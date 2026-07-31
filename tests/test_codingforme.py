import os
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from litellm.types.utils import ModelResponse

import codingforme as mini_pkg
from codingforme import (
    FakeModelClient,
    CodingForMe,
    OpenAICompatibleModelClient,
    SessionStore,
    WorkspaceContext,
    build_welcome,
)
from codingforme.models import _CompatBackendCustomLLM, final_answer, force_tool_choice, tool_call
from codingforme.tools import to_openai_function_specs


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs, **kwargs):
    """默认走原生 function-calling——和真实后端一致。

    脚本化输出用 `tool_call()` / `final_answer()` 构造，形状就是
    `model_client.complete()` 的返回值。`text_protocol=True` 只是把 client 标成
    "不吃 tools=" 的纯传输开关,不再改变教给模型的协议。
    """
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".codingforme" / "sessions")
    approval_policy = kwargs.pop("approval_policy", "auto")
    text_protocol = kwargs.pop("text_protocol", False)
    return CodingForMe(
        model_client=FakeModelClient(outputs, supports_native_tool_calls=not text_protocol),
        workspace=workspace,
        session_store=store,
        approval_policy=approval_policy,
        **kwargs,
    )


def test_agent_runs_tool_then_final(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            tool_call("read_file", path="hello.txt", start=1, end=2),
            final_answer("Read the file successfully."),
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Read the file successfully."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    assert "hello.txt" in agent.session["memory"]["files"]


def test_agent_updates_task_summary_on_each_request(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            final_answer("First pass."),
            final_answer("Second pass."),
        ],
    )

    assert agent.ask("First request") == "First pass."
    assert agent.session["memory"]["working"]["task_summary"] == "First request"

    assert agent.ask("Second request") == "Second pass."
    assert agent.session["memory"]["working"]["task_summary"] == "Second request"


def test_agent_only_stores_reusable_epistemic_notes(tmp_path):
    (tmp_path / "facts.txt").write_text("deploy key is red\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            tool_call("read_file", path="facts.txt", start=1, end=1),
            final_answer("Done."),
            final_answer("It is red."),
        ],
    )

    assert agent.ask("Read the file and remember the fact") == "Done."
    notes = agent.session["memory"]["episodic_notes"]
    assert any("deploy key is red" in note["text"] for note in notes)
    assert not any(note["text"] == "Done." for note in notes)
    assert not any(note["text"] == "Done." for note in notes)

    resumed = CodingForMe.from_session(
        model_client=FakeModelClient([final_answer("It is red.")]),
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("What color is the deploy key?") == "It is red."
    prompt = resumed.model_client.prompts[-1]
    assert "Relevant memory" in prompt
    assert "deploy key is red" in prompt


def test_file_summary_cache_is_invalidated_on_out_of_band_edit_and_path_spelling(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    agent.memory.set_file_summary("./sample.txt", "sample.txt: alpha")
    agent.memory.remember_file("./sample.txt")
    assert agent.memory.to_dict()["file_summaries"]["sample.txt"]["freshness"]

    assert "sample.txt: alpha" in agent.memory.render_memory_text()
    file_path.write_text("beta\n", encoding="utf-8")

    resumed = CodingForMe.from_session(
        model_client=FakeModelClient([]),
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert "sample.txt: alpha" not in resumed.memory_text()
    resumed.memory.invalidate_file_summary("sample.txt")
    assert "sample.txt" not in resumed.memory.to_dict()["file_summaries"]


def test_agent_retries_after_empty_model_output(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            final_answer(""),
            final_answer("Recovered after retry."),
        ],
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after retry."
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("empty response" in item for item in notices)


def test_agent_retries_after_malformed_tool_payload(tmp_path):
    """原生路径下的畸形调用：结构合法，但 arguments 不是一个 JSON 对象。

    真实后端会把模型吐出的 arguments 字符串原样带回来，`complete()` 解不出 dict
    时就把原字符串放进 args——这里直接脚本化那个形状。
    """
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            {"text": "", "tool_calls": [{"name": "read_file", "args": "bad"}]},
            tool_call("read_file", path="hello.txt", start=1, end=1),
            final_answer("Recovered after malformed tool output."),
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Recovered after malformed tool output."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("function-calling interface" in item for item in notices)


def test_tolerant_tag_reading_still_rescues_a_turn(tmp_path):
    """标签解析降级成"宽容读取"之后仍要能捞回那一轮。

    prompt 不再教这套写法，但模型偶尔还是会把调用写成文本（推理模型尤甚）。
    那时既不能崩、也不能白白浪费一轮：JSON 式 <tool>、适合多行内容的 XML 式
    <tool ...>、以及 <final> 都要照旧解析出来。
    """
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":"bad"}</tool>',
            '<tool name="write_file" path="hello.py"><content>print("hi")\n</content></tool>',
            "<final>Done.</final>",
        ],
    )

    answer = agent.ask("Create hello.py")

    assert answer == "Done."
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == 'print("hi")\n'
    # 宽容读取捞回来的调用会被计数，好让"协议漂移"可观测。
    assert agent.text_protocol_tool_calls == 1
    assert agent.last_prompt_metadata["text_protocol_tool_calls"] == 1


def test_native_tool_calls_do_not_count_as_protocol_drift(tmp_path):
    """走标准接口的调用不能被记成漂移，否则这个指标没有意义。"""
    agent = build_agent(
        tmp_path,
        [
            tool_call("list_files", path="."),
            final_answer("Done."),
        ],
    )

    assert agent.ask("list the files") == "Done."
    assert agent.text_protocol_tool_calls == 0


def test_retries_do_not_consume_the_whole_budget(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            final_answer(""),
            final_answer(""),
            final_answer("Recovered after several retries."),
        ],
        max_steps=1,
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after several retries."


def test_agent_saves_and_resumes_session(tmp_path):
    agent = build_agent(tmp_path, [final_answer("First pass.")])
    assert agent.ask("Start a session") == "First pass."

    resumed = CodingForMe.from_session(
        model_client=FakeModelClient([final_answer("Resumed.")]),
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.session["history"][0]["content"] == "Start a session"
    assert resumed.ask("Continue") == "Resumed."


def test_delegate_uses_child_agent(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            tool_call("delegate", task="inspect README", max_steps=2),
            final_answer("Child result."),
            final_answer("Parent incorporated the child result."),
        ],
    )

    answer = agent.ask("Use delegation")

    assert answer == "Parent incorporated the child result."
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert tool_events[0]["name"] == "delegate"
    assert "delegate_result" in tool_events[0]["content"]


def test_patch_file_replaces_exact_match(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool(
        "patch_file",
        {
            "path": "sample.txt",
            "old_text": "world",
            "new_text": "agent",
        },
    )

    assert result == "patched sample.txt"
    assert file_path.read_text(encoding="utf-8") == "hello agent\n"


def test_invalid_risky_tool_does_not_prompt_for_approval(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="ask")

    with patch("builtins.input") as mock_input:
        result = agent.run_tool("write_file", {})

    assert result.startswith("error: invalid arguments for write_file: 'path'")
    # 报错信息只示范参数对象，不示范调用形式——调用形式由 function-calling 接口决定。
    assert 'example arguments: {"path": "binary_search.py"' in result
    assert "<tool" not in result
    mock_input.assert_not_called()


def test_list_files_hides_internal_agent_state(tmp_path):
    agent = build_agent(tmp_path, [])
    (tmp_path / ".codingforme").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")

    result = agent.run_tool("list_files", {})

    assert ".codingforme" not in result
    assert ".git" not in result
    assert "[F] hello.txt" in result


def test_repeated_identical_tool_call_is_rejected(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "1"})
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "2"})

    result = agent.run_tool("list_files", {})

    assert result == "error: repeated identical tool call for list_files; choose a different tool or return a final answer"


def test_welcome_screen_keeps_box_shape_for_long_paths(tmp_path):
    deep = tmp_path / "very" / "long" / "path" / "for" / "the" / "mini" / "agent" / "welcome" / "screen"
    deep.mkdir(parents=True)
    agent = build_agent(deep, [])

    welcome = build_welcome(agent, model="qwen3.5:4b")
    lines = welcome.splitlines()

    assert len(lines) >= 5
    # 纯文本（无颜色）下，圆角框每一行的字符宽度都应一致。
    assert len({len(line) for line in lines}) == 1
    # 长路径会被中间截断，出现省略号。
    assert "..." in welcome
    # Claude Code 风格的火花与标题，以及 ASCII 机器人写代码场景 logo。
    assert "✻" in welcome
    assert "Welcome to coding-for-me" in welcome
    assert "while (1)" in welcome
    assert "code();" in welcome
    # 现场信息行存在。
    assert "cwd" in welcome
    assert "model" in welcome
    assert "qwen3.5:4b" in welcome
    # 旧版本的 ASCII 猫和文案已被替换掉。
    assert "(  o o  )" not in welcome
    assert "coding for me" not in welcome
    assert "calm shell" not in welcome



def test_openai_compatible_client_posts_expected_chat_completions_payload():
    captured = {}

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "backend text"}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete("hello", 42)

    assert result == {"text": "backend text", "tool_calls": None}
    assert captured["url"] == "https://right.codes/v1/chat/completions"
    assert captured["timeout"] == 30
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["headers"]["Accept"] == "application/json"
    assert captured["headers"]["User-agent"] == "coding-for-me/0.1"
    assert captured["body"] == {
        "model": "right.codes/codex-mini",
        "messages": [
            {
                "role": "user",
                "content": "hello",
            }
        ],
        "max_tokens": 42,
        "stream": False,
        "temperature": 0.2,
    }


def test_openai_compatible_client_sends_prompt_cache_fields_and_records_usage():
    captured = {}

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "output_text": "backend text",
                    "usage": {
                        "input_tokens": 2048,
                        "input_tokens_details": {"cached_tokens": 1536},
                        "output_tokens": 32,
                        "total_tokens": 2080,
                    },
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete(
            "hello",
            42,
            prompt_cache_key="prefix-hash-123",
            prompt_cache_retention="in_memory",
        )

    assert result == {"text": "backend text", "tool_calls": None}
    assert captured["body"]["prompt_cache_key"] == "prefix-hash-123"
    assert captured["body"]["prompt_cache_retention"] == "in_memory"
    assert client.last_completion_metadata["prompt_cache_supported"] is True
    assert client.last_completion_metadata["cached_tokens"] == 1536
    assert client.last_completion_metadata["cache_hit"] is True
    assert client.last_completion_metadata["input_tokens"] == 2048


def test_openai_compatible_client_extracts_text_from_event_stream():
    class FakeResponse:
        headers = {"Content-Type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                'data: {"type":"response.created","response":{"id":"resp_1","output":[]}}\n'
                'data: {"type":"response.completed","response":{"output":[{"content":[{"text":"streamed backend text"}]}]}}\n'
                "data: [DONE]\n"
            ).encode("utf-8")

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        result = client.complete("hello", 42)

    assert result == {"text": "streamed backend text", "tool_calls": None}


def test_openai_compatible_client_extracts_text_from_event_stream_deltas():
    class FakeResponse:
        headers = {"Content-Type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","delta":"streamed "}\n'
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","delta":"done"}\n'
                'event: response.output_text.done\n'
                'data: {"type":"response.output_text.done","text":"streamed done"}\n'
                "data: [DONE]\n"
            ).encode("utf-8")

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        result = client.complete("hello", 42)

    assert result == {"text": "streamed done", "tool_calls": None}


# ---------------------------------------------------------------------------
# 原生 function-calling 协议：parse() 的 tool_calls 分支、CustomLLM 桥接、
# tool_signature() 对 schema 翻译的敏感度。
# ---------------------------------------------------------------------------


def test_parse_accepts_native_tool_call():
    kind, payload = CodingForMe.parse({"text": "", "tool_calls": [{"name": "list_files", "args": {"path": "."}}]})
    assert kind == "tool"
    assert payload == {"name": "list_files", "args": {"path": "."}}


def test_parse_rejects_more_than_one_native_tool_call():
    kind, payload = CodingForMe.parse(
        {
            "text": "",
            "tool_calls": [
                {"name": "list_files", "args": {"path": "."}},
                {"name": "read_file", "args": {"path": "README.md"}},
            ],
        }
    )
    assert kind == "retry"


def test_parse_retries_when_native_tool_call_args_are_not_a_dict():
    kind, _ = CodingForMe.parse({"text": "", "tool_calls": [{"name": "list_files", "args": "not-json"}]})
    assert kind == "retry"


def test_parse_retries_when_native_tool_call_missing_name():
    kind, _ = CodingForMe.parse({"text": "", "tool_calls": [{"name": "", "args": {}}]})
    assert kind == "retry"


def test_parse_falls_back_to_text_tags_when_no_native_tool_calls():
    # 后端不支持 tools=（或者就是 FakeModelClient 的裸字符串）时，
    # dict 里 tool_calls 是 None/空，走原来的文本标签兜底解析。
    kind, payload = CodingForMe.parse({"text": '<tool>{"name":"list_files","args":{"path":"."}}</tool>', "tool_calls": None})
    assert kind == "tool"
    assert payload == {"name": "list_files", "args": {"path": "."}}

    kind, final = CodingForMe.parse({"text": "<final>plain text answer</final>", "tool_calls": []})
    assert kind == "final"
    assert final == "plain text answer"


def test_agent_executes_native_tool_call_end_to_end(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            {"text": "", "tool_calls": [{"name": "list_files", "args": {"path": "."}}]},
            {"text": "Done.", "tool_calls": None},
        ],
    )
    assert agent.ask("list the files") == "Done."


def test_to_openai_function_specs_marks_required_vs_optional():
    specs = to_openai_function_specs(
        {
            "read_file": {
                "schema": {"path": "str", "start": "int=1", "end": "int=200"},
                "risky": False,
                "description": "Read a UTF-8 file by line range.",
            }
        }
    )
    assert len(specs) == 1
    function = specs[0]["function"]
    assert function["name"] == "read_file"
    assert function["parameters"]["properties"]["path"] == {"type": "string"}
    assert function["parameters"]["properties"]["start"] == {"type": "integer"}
    assert function["parameters"]["required"] == ["path"]


def test_prompt_only_ever_teaches_the_function_calling_protocol(tmp_path):
    """prompt 里只存在一套协议，且不随后端能力变化。

    实测过：prefix 教 <tool> 文本协议、同时又发 tools= 原生 schema 时，
    同一个后端会一半走原生、一半吐文本标签，推理模型还会把 <tool> 标签埋进
    reasoning_content 里整轮作废。当时的修法是"两套互斥、按后端二选一"，现在
    直接只留一套——互斥这条约束因此不再需要被维护，它没有可违反的对象了。
    """
    agent = build_agent(tmp_path, [])
    assert "function-calling interface" in agent.prefix
    assert "<tool>" not in agent.prefix
    assert "<final>" not in agent.prefix
    assert "Valid response examples:" not in agent.prefix
    # 工具清单本身仍然保留：JSON Schema 里没有 approval/risk 这层信息。
    assert "approval required" in agent.prefix

    # 后端声明吃不吃 tools= 只影响传输，不再改变教给模型的协议。
    text_agent = build_agent(tmp_path, [], text_protocol=True)
    assert text_agent.native_tool_calls is False
    assert text_agent.prefix == agent.prefix

    # 报错信息是另一条通道，同样不能漏出标签写法。
    assert "<tool" not in agent.run_tool("write_file", {})


def test_ask_only_sends_tools_when_client_supports_native_tool_calls(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")

    seen = {}

    class _RecordingClient(FakeModelClient):
        def complete(self, prompt, max_new_tokens, **kwargs):
            seen.setdefault("tools", []).append(kwargs.get("tools"))
            return super().complete(prompt, max_new_tokens, **kwargs)

    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".codingforme" / "sessions")
    client = _RecordingClient([final_answer("done")], supports_native_tool_calls=False)
    agent = CodingForMe(
        model_client=client,
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )
    agent.ask("hi")
    assert seen["tools"] == [None]

    seen.clear()
    client2 = _RecordingClient([final_answer("done")])
    agent2 = CodingForMe(
        model_client=client2,
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )
    agent2.ask("hi")
    assert seen["tools"][0], "native client must receive the function specs"
    assert {spec["function"]["name"] for spec in seen["tools"][0]} == set(agent2.tools)


def test_retry_notice_points_back_at_the_only_protocol():
    """出错重试的提示是写回 history 的，模型下一轮会读到它。

    它必须指向 prefix 里那唯一一套协议：曾经在这里教 <tool> 标签，等于在模型
    出错的那一刻把它推向一套我们既没发 schema、也不打算支持的协议。
    """
    notice = CodingForMe.retry_notice("boom")
    assert "function-calling interface" in notice
    assert "<tool>" not in notice
    assert "<final>" not in notice


def test_custom_backend_llm_retries_instead_of_crashing_on_reasoning_only_response():
    """推理模型可能把全部输出放进 reasoning_content、让 content 为空。

    这种响应结构合法但这一轮没有可用输出，必须归约成 retry 让模型重来，
    而不是抛异常把整个 ask() 打断（实测跑基准时曾因此整任务崩溃）。
    """
    handler = _CompatBackendCustomLLM()

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "",
                                "role": "assistant",
                                "tool_calls": None,
                                "reasoning_content": "thinking out loud, never finished",
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
                }
            ).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        response = handler.completion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            api_base="https://example.test/v1",
            api_key="k",
            timeout=5,
            optional_params={},
            model_response=ModelResponse(),
        )

    assert response.choices[0].message.content == ""
    assert CodingForMe.parse({"text": "", "tool_calls": None})[0] == "retry"


def test_capabilities_are_declared_not_guessed_from_url():
    from codingforme.models import DEFAULT_CAPABILITIES, resolve_capabilities

    # 未知后端拿保守默认值，而不是一个碰巧匹配上的 substring 猜测。
    unknown = resolve_capabilities("https://api.unknown-vendor.test/v1")
    assert unknown == DEFAULT_CAPABILITIES
    assert unknown["prompt_cache_key"] is False
    assert unknown["native_tool_calls"] is True

    # 已知后端表生效。
    assert resolve_capabilities("https://api.openai.com/v1")["prompt_cache_key"] is True

    # 显式覆盖赢过已知后端表，None 表示"没意见"、让位给前面两层。
    assert resolve_capabilities(
        "https://api.openai.com/v1", {"prompt_cache_key": False}
    )["prompt_cache_key"] is False
    assert resolve_capabilities(
        "https://api.openai.com/v1", {"prompt_cache_key": None}
    )["prompt_cache_key"] is True

    with pytest.raises(ValueError):
        resolve_capabilities("https://x.test/v1", {"no_such_capability": True})


def test_client_capabilities_drive_both_switches():
    client = OpenAICompatibleModelClient(
        model="m",
        base_url="https://api.unknown-vendor.test",
        api_key="k",
        temperature=0.0,
        timeout=5,
        capabilities={"native_tool_calls": False, "prompt_cache_key": True},
    )
    assert client.supports_native_tool_calls is False
    assert client.supports_prompt_cache is True
    assert client.observed == {"prompt_cache_hit": False, "native_tool_calls": False}


def test_metadata_separates_declared_capability_from_observed_behaviour():
    """cache_hit 为真而 prompt_cache_supported 为假是正常的，不是矛盾。

    实测后端不认 prompt_cache_key 字段，却一直在做自动前缀缓存。旧 metadata
    只有一个 prompt_cache_supported，读起来像"没有缓存"，严重误导。
    """
    client = OpenAICompatibleModelClient(
        model="m",
        base_url="https://api.unknown-vendor.test",
        api_key="k",
        temperature=0.0,
        timeout=5,
    )
    assert client.supports_prompt_cache is False

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 5,
                        "total_tokens": 105,
                        "prompt_tokens_details": {"cached_tokens": 64},
                    },
                }
            ).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        client.complete("hi", 32, prompt_cache_key="abc")

    meta = client.last_completion_metadata
    assert meta["prompt_cache_supported"] is False
    assert meta["prompt_cache_key_sent"] is False, "未声明支持时不应该发 cache key"
    assert meta["cache_hit"] is True, "后端自动前缀缓存仍然生效"
    assert meta["cached_tokens"] == 64
    assert client.observed["prompt_cache_hit"] is True


def test_scripted_outputs_use_the_same_shape_as_a_real_backend():
    """脚本化输出的形状必须和 complete() 的真实返回值一致。

    这是"测试覆盖生产路径"的守门测试：只要 FakeModelClient 默认走原生协议、
    tool_call()/final_answer() 产出的就是 tool_calls 分支的形状，一整套用例
    验证的就不再是生产环境永远不走的文本解析路径。
    """
    assert FakeModelClient([]).supports_native_tool_calls is True

    call = tool_call("read_file", path="a.txt", start=1, end=2)
    assert call == {"text": "", "tool_calls": [{"name": "read_file", "args": {"path": "a.txt", "start": 1, "end": 2}}]}
    assert CodingForMe.parse(call) == ("tool", {"name": "read_file", "args": {"path": "a.txt", "start": 1, "end": 2}})

    answer = final_answer("Done.")
    assert answer == {"text": "Done.", "tool_calls": None}
    assert CodingForMe.parse(answer) == ("final", "Done.")


def test_forced_tool_choice_applies_once_and_then_clears():
    """安全评测强制的那次工具调用不能泄漏到后续请求。

    tool_choice 锁定一个工具后模型就给不出最终答案了，只会被逼着一直调工具
    直到步数耗尽——所以它必须严格只作用于紧接着的一次 complete()。
    """
    client = OpenAICompatibleModelClient(
        model="m",
        base_url="https://api.unknown-vendor.test",
        api_key="k",
        temperature=0.0,
        timeout=5,
    )
    specs = to_openai_function_specs(
        {"list_files": {"schema": {"path": "str='.'"}, "risky": False, "description": "List files."}}
    )
    sent = []

    def fake_completion(**kwargs):
        sent.append(kwargs.get("tool_choice"))
        return ModelResponse(
            choices=[{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]
        )

    client.pending_tool_choice = force_tool_choice("list_files")
    with patch("codingforme.models.litellm.completion", side_effect=fake_completion):
        client.complete("go", 32, tools=specs)
        assert client.last_completion_metadata["tool_choice_forced"] is True
        client.complete("go again", 32, tools=specs)
        assert client.last_completion_metadata["tool_choice_forced"] is False

    assert sent == [{"type": "function", "function": {"name": "list_files"}}, "auto"]
    assert client.pending_tool_choice is None


def test_forced_tool_choice_is_dropped_when_tools_are_not_sent():
    """没发 tools= 的那一轮也要把强制意图取走，否则会落到下一次无关请求上。"""
    client = OpenAICompatibleModelClient(
        model="m",
        base_url="https://api.unknown-vendor.test",
        api_key="k",
        temperature=0.0,
        timeout=5,
    )

    def fake_completion(**kwargs):
        assert "tool_choice" not in kwargs
        return ModelResponse(
            choices=[{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]
        )

    client.pending_tool_choice = force_tool_choice("list_files")
    with patch("codingforme.models.litellm.completion", side_effect=fake_completion):
        client.complete("go", 32)

    assert client.pending_tool_choice is None
    assert client.last_completion_metadata["tool_choice_forced"] is False


def test_tool_signature_changes_when_schema_changes(tmp_path):
    agent = build_agent(tmp_path, [])
    original = agent.tool_signature()
    agent.tools["list_files"]["schema"] = dict(agent.tools["list_files"]["schema"], extra="str='x'")
    changed = agent.tool_signature()
    assert original != changed


def test_custom_backend_llm_extracts_tool_calls_from_standard_choices_response():
    handler = _CompatBackendCustomLLM()

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "list_files", "arguments": '{"path": "."}'},
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            ).encode("utf-8")

    model_response = ModelResponse()
    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        result = handler.completion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            api_base="https://example.test/v1",
            api_key="sk-test",
            timeout=10,
            optional_params={},
            model_response=model_response,
        )

    tool_calls = result.choices[0].message.tool_calls
    assert tool_calls[0].function.name == "list_files"
    assert tool_calls[0].function.arguments == '{"path": "."}'
    assert result.usage.prompt_tokens == 10


def test_custom_backend_llm_handles_output_text_shape_without_choices():
    # 已知的真实后端怪癖之一：/chat/completions 上返回 Responses-API 风格的
    # output_text，而不是标准的 choices。litellm 内建传输遇到这种形状会直接
    # 报错，CustomLLM 桥接必须继续兼容它。
    handler = _CompatBackendCustomLLM()

    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "output_text": "backend text",
                    "usage": {"input_tokens": 2048, "input_tokens_details": {"cached_tokens": 1536}, "output_tokens": 32, "total_tokens": 2080},
                }
            ).encode("utf-8")

    model_response = ModelResponse()
    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        result = handler.completion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            api_base="https://example.test/v1",
            api_key="sk-test",
            timeout=10,
            optional_params={},
            model_response=model_response,
        )

    assert result.choices[0].message.content == "backend text"
    assert result.usage.prompt_tokens_details.cached_tokens == 1536


def test_custom_backend_llm_handles_sse_despite_non_stream_request():
    # 另一个已知怪癖：声明 stream:false 但后端仍返回 SSE。
    handler = _CompatBackendCustomLLM()

    class FakeResponse:
        headers = {"Content-Type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                'data: {"type":"response.output_text.delta","delta":"streamed "}\n'
                'data: {"type":"response.output_text.delta","delta":"done"}\n'
                'data: {"type":"response.output_text.done","text":"streamed done"}\n'
                "data: [DONE]\n"
            ).encode("utf-8")

    model_response = ModelResponse()
    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        result = handler.completion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            api_base="https://example.test/v1",
            api_key="sk-test",
            timeout=10,
            optional_params={},
            model_response=model_response,
        )

    assert result.choices[0].message.content == "streamed done"


class _FakeStreamResponse:
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self._lines)


def test_custom_backend_llm_streaming_parses_standard_delta_chunks():
    handler = _CompatBackendCustomLLM()
    lines = [
        b'data: {"choices":[{"delta":{"content":"Hello","role":"assistant"},"finish_reason":null,"index":0}]}\n',
        b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null,"index":0}]}\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,'
        b'"prompt_tokens_details":{"cached_tokens":3}}}\n',
        b"data: [DONE]\n",
    ]
    with patch("urllib.request.urlopen", return_value=_FakeStreamResponse(lines)):
        chunks = list(
            handler.streaming(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                api_base="https://example.test/v1",
                api_key="sk-test",
                timeout=10,
                optional_params={},
                model_response=ModelResponse(),
            )
        )

    text_chunks = [chunk["text"] for chunk in chunks if chunk["text"]]
    assert text_chunks == ["Hello", " world"]
    final_chunk = chunks[-1]
    assert final_chunk["is_finished"] is True
    assert final_chunk["finish_reason"] == "stop"
    assert final_chunk["usage"]["prompt_tokens"] == 10
    assert final_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == 3


def test_openai_compatible_client_streams_tokens_via_on_token_callback():
    lines = [
        b'data: {"choices":[{"delta":{"content":"1\\n","role":"assistant"},"finish_reason":null,"index":0}]}\n',
        b'data: {"choices":[{"delta":{"content":"2\\n"},"finish_reason":null,"index":0}]}\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],'
        b'"usage":{"prompt_tokens":20,"completion_tokens":4,"total_tokens":24}}\n',
        b"data: [DONE]\n",
    ]
    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url="https://right.codes/v1",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )
    received = []
    with patch("urllib.request.urlopen", return_value=_FakeStreamResponse(lines)):
        result = client.complete("count", 42, on_token=received.append)

    assert received == ["1\n", "2\n"]
    assert result == {"text": "1\n2\n", "tool_calls": None}
    assert client.last_completion_metadata["input_tokens"] == 20
    assert client.last_completion_metadata["total_tokens"] == 24


def test_build_agent_uses_openai_provider_and_model_override(tmp_path):
    args = type(
        "Args",
        (),
        {
            "cwd": str(tmp_path),
            "model": "override-model",
            "base_url": None,
            "openai_timeout": 300,
            "temperature": 0.2,
            "resume": None,
            "approval": "ask",
            "secret_env_names": [],
            "max_steps": 6,
            "max_new_tokens": 512,
        },
    )()

    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE": "https://www.right.codes/codex/v1",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "env-model",
        },
        clear=False,
    ):
        with patch("codingforme.cli.OpenAICompatibleModelClient") as mock_openai:
            fake_client = mock_openai.return_value
            agent = mini_pkg.build_agent(args)

    mock_openai.assert_called_once()
    assert mock_openai.call_args.kwargs["model"] == "override-model"
    assert mock_openai.call_args.kwargs["base_url"] == "https://www.right.codes/codex/v1"
    assert mock_openai.call_args.kwargs["api_key"] == "sk-test"
    assert agent.model_client is fake_client



def test_build_agent_uses_openai_provider_by_default(tmp_path):
    args = mini_pkg.build_arg_parser().parse_args(["--cwd", str(tmp_path)])

    with patch.dict(
        os.environ,
        {
            "OPENAI_API_BASE": "https://www.right.codes/codex/v1",
            "OPENAI_API_KEY": "sk-test",
        },
        clear=False,
    ):
        with patch("codingforme.cli.OpenAICompatibleModelClient") as mock_openai:
            fake_client = mock_openai.return_value
            agent = mini_pkg.build_agent(args)

    mock_openai.assert_called_once()
    assert mock_openai.call_args.kwargs["model"] == "gpt-5.4"
    assert mock_openai.call_args.kwargs["base_url"] == "https://www.right.codes/codex/v1"
    assert mock_openai.call_args.kwargs["api_key"] == "sk-test"
    assert agent.model_client is fake_client


def test_successful_run_persists_run_artifacts_and_stop_reason(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            tool_call("read_file", path="hello.txt", start=1, end=2),
            final_answer("Finished."),
        ],
    )

    assert agent.ask("Do the thing") == "Finished."

    runs_root = tmp_path / ".codingforme" / "runs"
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    task_state = json.loads((run_dir / "task_state.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    trace_lines = (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()

    assert task_state["task_id"] != task_state["run_id"]
    assert run_dir.name == task_state["run_id"]
    assert (run_dir / "task_state.json").exists()
    assert (run_dir / "trace.jsonl").exists()
    assert (run_dir / "report.json").exists()
    assert task_state["stop_reason"] == "final_answer_returned"
    assert task_state["final_answer"] == "Finished."
    assert report["stop_reason"] == "final_answer_returned"
    assert report["task_state"]["stop_reason"] == "final_answer_returned"
    assert report["run_id"] == task_state["run_id"]
    trace_events = [json.loads(line)["event"] for line in trace_lines]
    assert trace_events[0] == "run_started"
    assert trace_events[-1] == "run_finished"
    assert trace_events.count("prompt_built") == 2
    assert "tool_executed" in trace_events


def test_trace_and_report_redact_secret_env_values(tmp_path):
    secret = "sk-test-secret-123"
    with patch.dict(os.environ, {"OPENAI_API_KEY": secret}, clear=True):
        agent = build_agent(
            tmp_path,
            [
                tool_call("run_shell", command="printf '%s' 'sk-test-secret-123'", timeout=20),
                final_answer("Masked."),
            ],
        )

        assert agent.ask("Mask the secret") == "Masked."

    runs_root = tmp_path / ".codingforme" / "runs"
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    trace_text = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
    report_text = (run_dir / "report.json").read_text(encoding="utf-8")
    trace_events = [json.loads(line) for line in trace_text.splitlines()]

    assert secret not in trace_text
    assert secret not in report_text

    prompt_events = [event for event in trace_events if event["event"] == "prompt_built"]
    assert prompt_events
    assert prompt_events[0]["prompt_metadata"]["secret_env_count"] >= 1
    assert "OPENAI_API_KEY" in prompt_events[0]["prompt_metadata"]["secret_env_names"]

    tool_events = [event for event in trace_events if event["event"] == "tool_executed"]
    assert tool_events
    assert "<redacted>" in tool_events[0]["args"]["command"]
    assert "<redacted>" in tool_events[0]["result"]


def test_prompt_budget_metadata_records_budget_decisions(tmp_path):
    agent = build_agent(tmp_path, [final_answer("Done.")])
    agent.memory.append_note("alpha episodic note " + ("A" * 120), tags=("recall",), created_at="2026-04-07T10:00:00+00:00")
    agent.memory.append_note("beta episodic recall note " + ("B" * 120), created_at="2026-04-07T10:01:00+00:00")
    agent.memory.append_note("gamma episodic note " + ("C" * 120), tags=("recall",), created_at="2026-04-07T10:02:00+00:00")

    for index in range(4):
        agent.record(
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"history-{index}-" + ("A" * 240),
                "created_at": f"2026-04-07T10:0{index}:00+00:00",
            }
        )

    agent.context_manager.total_budget = 1000
    agent.context_manager.section_budgets = {
        "prefix": 80,
        "memory": 80,
        "relevant_memory": 80,
        "history": 80,
    }

    assert agent.ask("recall") == "Done."

    trace_events = [
        json.loads(line)
        for line in (agent.run_store.trace_path(agent.current_task_state).read_text(encoding="utf-8").splitlines())
    ]
    prompt_events = [event for event in trace_events if event["event"] == "prompt_built"]
    assert prompt_events
    metadata = prompt_events[0]["prompt_metadata"]
    relevant_section = agent.model_client.prompts[0].split("Relevant memory:\n", 1)[1].split("\n\nTranscript:", 1)[0]

    assert metadata["relevant_memory"]["selected_count"] == 3
    assert len(metadata["relevant_memory"]["rendered_notes"]) == 3
    assert len([line for line in relevant_section.splitlines() if line.startswith("- ")]) == 3
    assert "alpha episodic" in relevant_section
    assert "beta episodic" in relevant_section
    assert "gamma episodic" in relevant_section
    assert metadata["current_request"]["text"] == "recall"
    assert metadata["current_request"]["rendered_chars"] == len("recall")


def test_prompt_metadata_refreshes_prefix_when_workspace_changes(tmp_path):
    agent = build_agent(tmp_path, [])

    first = agent.prompt_metadata("first", "")
    second = agent.prompt_metadata("second", "")

    assert first["prefix_hash"] == second["prefix_hash"]
    assert second["prefix_changed"] is False
    assert second["workspace_changed"] is False

    (tmp_path / "README.md").write_text("demo changed\n", encoding="utf-8")

    third = agent.prompt_metadata("third", "")

    assert third["prefix_hash"] != second["prefix_hash"]
    assert third["prefix_changed"] is True
    assert third["workspace_changed"] is True
    assert "demo changed" in agent.prefix


def test_agent_creates_checkpoint_when_context_reduction_happens_and_artifacts_only_reference_it(tmp_path):
    agent = build_agent(tmp_path, [final_answer("Done after checkpoint.")])
    for index in range(10):
        agent.record(
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"history-{index}-" + ("A" * 260),
                "created_at": f"2026-04-07T10:{index:02d}:00+00:00",
            }
        )
    agent.memory.append_note("checkpoint note " + ("B" * 220), tags=("checkpoint",), created_at="2026-04-07T11:00:00+00:00")
    agent.context_manager.total_budget = 900
    agent.context_manager.section_budgets = {
        "prefix": 120,
        "memory": 120,
        "relevant_memory": 120,
        "history": 160,
    }

    assert agent.ask("Resume the long task") == "Done after checkpoint."

    checkpoint_state = agent.session["checkpoints"]
    checkpoint = checkpoint_state["items"][checkpoint_state["current_id"]]
    assert checkpoint["checkpoint_id"] == checkpoint_state["current_id"]
    assert checkpoint["schema_version"] == "phase1-v1"
    assert checkpoint["current_goal"] == "Resume the long task"
    assert checkpoint["key_files"] == []
    assert checkpoint["current_blocker"] == ""
    assert checkpoint["next_step"]

    task_state = json.loads(agent.run_store.task_state_path(agent.current_task_state).read_text(encoding="utf-8"))
    report = json.loads(agent.run_store.report_path(agent.current_task_state).read_text(encoding="utf-8"))
    trace_events = [
        json.loads(line)
        for line in agent.run_store.trace_path(agent.current_task_state).read_text(encoding="utf-8").splitlines()
    ]

    assert task_state["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert report["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert report["task_state"]["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert "current_goal" not in task_state
    assert "current_goal" not in report
    checkpoint_events = [event for event in trace_events if event["event"] == "checkpoint_created"]
    assert checkpoint_events
    assert checkpoint_events[-1]["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert "current_goal" not in checkpoint_events[-1]


def test_resume_prompt_uses_checkpoint_state_not_just_history(tmp_path):
    agent = build_agent(tmp_path, [final_answer("checkpoint ready.")])
    agent.session["checkpoints"] = {
        "current_id": "ckpt_manual",
        "items": {
            "ckpt_manual": {
                "checkpoint_id": "ckpt_manual",
                "parent_checkpoint_id": "",
                "schema_version": "phase1-v1",
                "created_at": "2026-04-14T09:00:00+00:00",
                "current_goal": "Fix failing resume flow",
                "completed": ["Read runtime.py"],
                "excluded": ["Do not add branch summary"],
                "current_blocker": "Need to re-anchor stale file facts",
                "next_step": "Re-read runtime.py and refresh the checkpoint",
                "key_files": [{"path": "runtime.py", "freshness": "abc"}],
                "freshness": {"runtime.py": "abc"},
                "summary": "Resume from the latest checkpoint",
                "runtime_identity": {"workspace_fingerprint": "old-fingerprint"},
            }
        },
    }
    agent.session_store.save(agent.session)

    resumed = CodingForMe.from_session(
        model_client=FakeModelClient([final_answer("Resumed.")]),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("Continue the task") == "Resumed."

    prompt = resumed.model_client.prompts[-1]
    assert "Task checkpoint:" in prompt
    assert "Current goal: Fix failing resume flow" in prompt
    assert "Current blocker: Need to re-anchor stale file facts" in prompt
    assert "Next step: Re-read runtime.py and refresh the checkpoint" in prompt


def test_resume_invalidates_stale_file_summaries_and_marks_partial_stale(tmp_path):
    file_path = tmp_path / "runtime.py"
    file_path.write_text("alpha\n", encoding="utf-8")
    agent = build_agent(tmp_path, [final_answer("checkpoint ready.")])
    agent.memory.set_file_summary("runtime.py", "runtime.py: alpha")
    freshness = agent.memory.to_dict()["file_summaries"]["runtime.py"]["freshness"]
    agent.session["checkpoints"] = {
        "current_id": "ckpt_stale",
        "items": {
            "ckpt_stale": {
                "checkpoint_id": "ckpt_stale",
                "parent_checkpoint_id": "",
                "schema_version": "phase1-v1",
                "created_at": "2026-04-14T09:00:00+00:00",
                "current_goal": "Fix stale summary handling",
                "completed": [],
                "excluded": [],
                "current_blocker": "",
                "next_step": "Re-read runtime.py",
                "key_files": [{"path": "runtime.py", "freshness": freshness}],
                "freshness": {"runtime.py": freshness},
                "summary": "runtime.py is important",
                "runtime_identity": {"workspace_fingerprint": agent.workspace.fingerprint()},
            }
        },
    }
    agent.session_store.save(agent.session)
    file_path.write_text("beta\n", encoding="utf-8")

    resumed = CodingForMe.from_session(
        model_client=FakeModelClient([final_answer("Resumed.")]),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("Continue the task") == "Resumed."

    assert "runtime.py" not in resumed.memory.to_dict()["file_summaries"]
    assert resumed.last_prompt_metadata["resume_status"] == "partial-stale"
    assert resumed.last_prompt_metadata["stale_summary_invalidations"] == 1


def test_run_shell_nonzero_with_workspace_change_is_recorded_as_partial_success(tmp_path):
    agent = build_agent(tmp_path, [])

    result = agent.run_tool(
        "run_shell",
        {
            "command": "printf 'changed\\n' > README.md && exit 1",
            "timeout": 20,
        },
    )

    assert "exit_code: 1" in result
    assert agent._last_tool_result_metadata["tool_status"] == "partial_success"
    assert agent._last_tool_result_metadata["affected_paths"] == ["README.md"]
    assert agent._last_tool_result_metadata["workspace_changed"] is True


def test_resume_marks_workspace_mismatch_when_checkpoint_runtime_identity_is_stale(tmp_path):
    agent = build_agent(tmp_path, [final_answer("checkpoint ready.")])
    agent.session["checkpoints"] = {
        "current_id": "ckpt_workspace",
        "items": {
            "ckpt_workspace": {
                "checkpoint_id": "ckpt_workspace",
                "parent_checkpoint_id": "",
                "schema_version": "phase1-v1",
                "created_at": "2026-04-14T09:00:00+00:00",
                "current_goal": "Continue after drift",
                "completed": [],
                "excluded": [],
                "current_blocker": "",
                "next_step": "Rebuild runtime state",
                "key_files": [],
                "freshness": {},
                "summary": "workspace changed",
                "runtime_identity": {"workspace_fingerprint": "outdated-fingerprint"},
            }
        },
    }
    agent.session_store.save(agent.session)

    resumed = CodingForMe.from_session(
        model_client=FakeModelClient([final_answer("Resumed.")]),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("Continue the task") == "Resumed."
    assert resumed.last_prompt_metadata["resume_status"] == "workspace-mismatch"


def test_write_file_trace_records_minimum_tool_contract_fields(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            tool_call("write_file", path="notes.txt", content="hello\n"),
            final_answer("Done."),
        ],
    )

    assert agent.ask("Create notes.txt") == "Done."

    trace_events = [
        json.loads(line)
        for line in agent.run_store.trace_path(agent.current_task_state).read_text(encoding="utf-8").splitlines()
    ]
    tool_event = [event for event in trace_events if event["event"] == "tool_executed"][-1]

    assert tool_event["name"] == "write_file"
    assert tool_event["risk_level"] == "high"
    assert tool_event["read_only"] is False
    assert tool_event["tool_status"] == "ok"
    assert tool_event["affected_paths"] == ["notes.txt"]
    assert tool_event["workspace_changed"] is True
    assert tool_event["diff_summary"] == ["created:notes.txt"]


def test_resume_marks_schema_mismatch_when_checkpoint_version_is_incompatible(tmp_path):
    agent = build_agent(tmp_path, [final_answer("checkpoint ready.")])
    agent.session["checkpoints"] = {
        "current_id": "ckpt_schema",
        "items": {
            "ckpt_schema": {
                "checkpoint_id": "ckpt_schema",
                "parent_checkpoint_id": "",
                "schema_version": "legacy-v0",
                "created_at": "2026-04-14T09:00:00+00:00",
                "current_goal": "Continue after schema change",
                "completed": [],
                "excluded": [],
                "current_blocker": "",
                "next_step": "Migrate checkpoint",
                "key_files": [],
                "freshness": {},
                "summary": "schema changed",
                "runtime_identity": {"workspace_fingerprint": agent.workspace.fingerprint()},
            }
        },
    }
    agent.session_store.save(agent.session)

    resumed = CodingForMe.from_session(
        model_client=FakeModelClient([final_answer("Resumed.")]),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("Continue the task") == "Resumed."
    assert resumed.last_prompt_metadata["resume_status"] == "schema-mismatch"


def test_resume_marks_no_checkpoint_when_session_has_no_checkpoint_state(tmp_path):
    agent = build_agent(tmp_path, [final_answer("checkpoint ready.")])
    agent.session.pop("checkpoints", None)
    agent.session_store.save(agent.session)

    resumed = CodingForMe.from_session(
        model_client=FakeModelClient([final_answer("Resumed.")]),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("Continue the task") == "Resumed."
    assert resumed.last_prompt_metadata["resume_status"] == "no-checkpoint"
    assert "Task checkpoint:" not in resumed.model_client.prompts[-1]


def test_freshness_mismatch_creates_checkpoint_before_model_completion(tmp_path):
    file_path = tmp_path / "runtime.py"
    file_path.write_text("alpha\n", encoding="utf-8")
    agent = build_agent(tmp_path, [final_answer("Resumed.")])
    agent.memory.set_file_summary("runtime.py", "runtime.py: alpha")
    freshness = agent.memory.to_dict()["file_summaries"]["runtime.py"]["freshness"]
    agent.session["checkpoints"] = {
        "current_id": "ckpt_freshness",
        "items": {
            "ckpt_freshness": {
                "checkpoint_id": "ckpt_freshness",
                "parent_checkpoint_id": "",
                "schema_version": "phase1-v1",
                "created_at": "2026-04-14T09:00:00+00:00",
                "current_goal": "Handle freshness mismatch",
                "completed": [],
                "excluded": [],
                "current_blocker": "",
                "next_step": "Re-read runtime.py",
                "key_files": [{"path": "runtime.py", "freshness": freshness}],
                "freshness": {"runtime.py": freshness},
                "summary": "runtime.py changed",
                "runtime_identity": {"workspace_fingerprint": agent.workspace.fingerprint()},
            }
        },
    }
    agent.session_store.save(agent.session)
    file_path.write_text("beta\n", encoding="utf-8")

    assert agent.ask("Continue the task") == "Resumed."

    trace_events = [
        json.loads(line)
        for line in agent.run_store.trace_path(agent.current_task_state).read_text(encoding="utf-8").splitlines()
    ]
    checkpoint_events = [event for event in trace_events if event["event"] == "checkpoint_created"]

    assert checkpoint_events
    assert checkpoint_events[0]["trigger"] == "freshness_mismatch"


def test_runtime_identity_persists_key_execution_metadata(tmp_path):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".codingforme" / "sessions")
    agent = CodingForMe(
        model_client=FakeModelClient([final_answer("Done.")]),
        workspace=workspace,
        session_store=store,
        approval_policy="never",
        max_steps=9,
        max_new_tokens=1024,
        feature_flags={"memory": True, "relevant_memory": False},
    )

    runtime_identity = agent.session["runtime_identity"]

    assert runtime_identity["session_id"] == agent.session["id"]
    assert runtime_identity["cwd"] == str(tmp_path)
    assert runtime_identity["approval_policy"] == "never"
    assert runtime_identity["read_only"] is False
    assert runtime_identity["max_steps"] == 9
    assert runtime_identity["max_new_tokens"] == 1024
    assert runtime_identity["feature_flags"]["memory"] is True
    assert runtime_identity["feature_flags"]["relevant_memory"] is False
    assert runtime_identity["shell_env_allowlist"] == list(agent.shell_env_allowlist)


def test_resume_records_runtime_identity_mismatch_fields_in_metadata_and_trace(tmp_path):
    agent = build_agent(tmp_path, [final_answer("checkpoint ready.")])
    agent.session["checkpoints"] = {
        "current_id": "ckpt_identity",
        "items": {
            "ckpt_identity": {
                "checkpoint_id": "ckpt_identity",
                "parent_checkpoint_id": "",
                "schema_version": "phase1-v1",
                "created_at": "2026-04-14T09:00:00+00:00",
                "current_goal": "Resume with a different runtime identity",
                "completed": [],
                "excluded": [],
                "current_blocker": "",
                "next_step": "Rebuild runtime identity",
                "key_files": [],
                "freshness": {},
                "summary": "identity changed",
                "runtime_identity": {
                    "workspace_fingerprint": agent.workspace.fingerprint(),
                    "approval_policy": "auto",
                    "read_only": False,
                    "max_steps": 6,
                    "max_new_tokens": 512,
                    "model": "old-model",
                    "model_client": "FakeModelClient",
                    "feature_flags": {"memory": True, "relevant_memory": True},
                    "shell_env_allowlist": ["PATH"],
                    "session_id": agent.session["id"],
                    "cwd": str(tmp_path),
                },
            }
        },
    }
    agent.session_store.save(agent.session)

    resumed = CodingForMe.from_session(
        model_client=FakeModelClient([final_answer("Resumed.")]),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="never",
        max_steps=9,
        max_new_tokens=1024,
        feature_flags={"memory": True, "relevant_memory": False},
    )

    resumed.ask("Continue the task")

    assert resumed.last_prompt_metadata["resume_status"] == "workspace-mismatch"
    assert resumed.last_prompt_metadata["runtime_identity_mismatch_fields"] == [
        "approval_policy",
        "feature_flags",
        "max_new_tokens",
        "max_steps",
        "model",
        "shell_env_allowlist",
    ]

    trace_events = [
        json.loads(line)
        for line in resumed.run_store.trace_path(resumed.current_task_state).read_text(encoding="utf-8").splitlines()
    ]
    mismatch_events = [event for event in trace_events if event["event"] == "runtime_identity_mismatch"]
    assert mismatch_events
    assert mismatch_events[0]["fields"] == [
        "approval_policy",
        "feature_flags",
        "max_new_tokens",
        "max_steps",
        "model",
        "shell_env_allowlist",
    ]


def test_partial_success_creates_process_note_for_exploration_history(tmp_path):
    agent = build_agent(tmp_path, [])

    agent.run_tool(
        "run_shell",
        {
            "command": "printf 'changed\\n' > README.md && exit 1",
            "timeout": 20,
        },
    )

    process_notes = [
        note
        for note in agent.memory.to_dict()["episodic_notes"]
        if note.get("kind") == "process"
    ]

    assert process_notes
    assert process_notes[-1]["text"] == "run_shell partial_success on README.md; inspect diff before retry"
    assert "partial_success" in process_notes[-1]["tags"]
    assert "README.md" in process_notes[-1]["tags"]


def test_explicit_memory_promotion_persists_durable_memory_topics(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            final_answer(
                "Project convention: Use constrained tools instead of guessing.\n"
                "Project convention: Preserve local agent state under .codingforme/.\n"
                "Decision: Keep durable memory topic-based and lightweight."
            ),
        ],
    )

    answer = agent.ask(
        "Capture the stable facts you already discovered as durable memory. "
        "Respond with exactly the long-term facts."
    )

    assert "Project convention:" in answer

    index_path = tmp_path / ".codingforme" / "memory" / "MEMORY.md"
    conventions_path = tmp_path / ".codingforme" / "memory" / "topics" / "project-conventions.md"
    decisions_path = tmp_path / ".codingforme" / "memory" / "topics" / "key-decisions.md"
    report = json.loads(agent.run_store.report_path(agent.current_task_state).read_text(encoding="utf-8"))

    assert index_path.exists()
    assert conventions_path.exists()
    assert decisions_path.exists()
    assert "project-conventions" in index_path.read_text(encoding="utf-8")
    assert "Use constrained tools instead of guessing." in conventions_path.read_text(encoding="utf-8")
    assert "Keep durable memory topic-based and lightweight." in decisions_path.read_text(encoding="utf-8")
    assert report["durable_promotions"] == [
        "project-conventions: Use constrained tools instead of guessing.",
        "project-conventions: Preserve local agent state under .codingforme/.",
        "key-decisions: Keep durable memory topic-based and lightweight.",
    ]


def test_explicit_memory_promotion_supports_chinese_intent_and_labels(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            final_answer(
                "项目约定：优先使用受约束工具，不要靠猜。\n"
                "决策：持久记忆保持轻量、按 topic 管理。"
            ),
        ],
    )

    answer = agent.ask("请把下面这些稳定事实记住，作为长期记忆保存下来。")

    assert "项目约定：" in answer

    conventions_path = tmp_path / ".codingforme" / "memory" / "topics" / "project-conventions.md"
    decisions_path = tmp_path / ".codingforme" / "memory" / "topics" / "key-decisions.md"

    assert "优先使用受约束工具，不要靠猜。" in conventions_path.read_text(encoding="utf-8")
    assert "持久记忆保持轻量、按 topic 管理。" in decisions_path.read_text(encoding="utf-8")


def test_explicit_memory_promotion_rejects_secret_shaped_and_transient_lines(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            final_answer(
                "Project convention: Use constrained tools instead of guessing.\n"
                "Dependency: API key is sk-live-secret-abc.\n"
                "Decision: Current goal is fix flaky tests.\n"
                "Dependency: stdout: FAIL test_one FAIL test_two FAIL test_three."
            ),
        ],
    )

    agent.ask("Capture these stable facts into durable memory.")

    report = json.loads(agent.run_store.report_path(agent.current_task_state).read_text(encoding="utf-8"))
    conventions_path = tmp_path / ".codingforme" / "memory" / "topics" / "project-conventions.md"
    dependency_path = tmp_path / ".codingforme" / "memory" / "topics" / "dependency-facts.md"

    assert report["durable_promotions"] == [
        "project-conventions: Use constrained tools instead of guessing.",
    ]
    assert report["durable_rejections"] == [
        "dependency-facts:secret_shaped",
        "key-decisions:transient_task_state",
        "dependency-facts:noisy_output",
    ]
    assert "Use constrained tools instead of guessing." in conventions_path.read_text(encoding="utf-8")
    assert not dependency_path.exists()


def test_explicit_memory_promotion_supersedes_matching_durable_fact(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            final_answer("Dependency: Python runtime is 3.11."),
            final_answer("Dependency: Python runtime is 3.12."),
        ],
    )

    assert agent.ask("Capture this stable dependency fact into durable memory.") == "Dependency: Python runtime is 3.11."
    assert agent.ask("Save the updated dependency fact into durable memory.") == "Dependency: Python runtime is 3.12."

    dependency_path = tmp_path / ".codingforme" / "memory" / "topics" / "dependency-facts.md"
    report = json.loads(agent.run_store.report_path(agent.current_task_state).read_text(encoding="utf-8"))
    text = dependency_path.read_text(encoding="utf-8")

    assert "Python runtime is 3.12." in text
    assert "Python runtime is 3.11." not in text
    assert report["durable_superseded"] == [
        "dependency-facts: Python runtime is 3.11. -> Python runtime is 3.12.",
    ]


def test_explicit_memory_promotion_dedupes_duplicate_durable_note(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            final_answer("Project convention: Use constrained tools instead of guessing."),
            final_answer("Project convention: Use constrained tools instead of guessing."),
        ],
    )

    agent.ask("Capture the stable fact into durable memory.")
    agent.ask("Capture the stable fact into durable memory again.")

    conventions_path = tmp_path / ".codingforme" / "memory" / "topics" / "project-conventions.md"
    text = conventions_path.read_text(encoding="utf-8")

    assert text.count("Use constrained tools instead of guessing.") == 1


def test_agent_records_model_cache_metadata_in_last_prompt_metadata(tmp_path):
    class CacheAwareFakeModelClient(FakeModelClient):
        def complete(self, prompt, max_new_tokens, **kwargs):
            self.last_completion_metadata = {
                "prompt_cache_supported": True,
                "cached_tokens": 512,
                "cache_hit": True,
                "input_tokens": 1024,
            }
            return super().complete(prompt, max_new_tokens, **kwargs)

    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".codingforme" / "sessions")
    agent = CodingForMe(
        model_client=CacheAwareFakeModelClient([final_answer("Done.")]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )

    assert agent.ask("Cache aware run") == "Done."

    assert agent.last_prompt_metadata["prompt_cache_supported"] is True
    assert agent.last_prompt_metadata["cached_tokens"] == 512
    assert agent.last_prompt_metadata["cache_hit"] is True
    assert agent.last_prompt_metadata["prefix_hash"]
    assert agent.last_prompt_metadata["prompt_cache_key"] == agent.last_prompt_metadata["prefix_hash"]


def test_recent_transcript_entries_stay_richer_than_older_ones(tmp_path):
    agent = build_agent(tmp_path, [final_answer("Done.")])
    old_text = "OLD-" + ("A" * 320)
    recent_text = "RECENT-" + ("B" * 320)

    agent.record({"role": "user", "content": old_text, "created_at": "2026-04-07T09:00:00+00:00"})
    agent.record({"role": "assistant", "content": old_text, "created_at": "2026-04-07T09:01:00+00:00"})
    agent.record({"role": "user", "content": recent_text, "created_at": "2026-04-07T09:02:00+00:00"})
    agent.record({"role": "assistant", "content": recent_text, "created_at": "2026-04-07T09:03:00+00:00"})
    agent.record({"role": "user", "content": recent_text, "created_at": "2026-04-07T09:04:00+00:00"})
    agent.record({"role": "assistant", "content": recent_text, "created_at": "2026-04-07T09:05:00+00:00"})
    agent.record({"role": "user", "content": recent_text, "created_at": "2026-04-07T09:06:00+00:00"})
    agent.record({"role": "assistant", "content": recent_text, "created_at": "2026-04-07T09:07:00+00:00"})

    assert agent.ask("Check the transcript") == "Done."

    prompt = agent.model_client.prompts[-1]

    assert recent_text in prompt
    assert old_text not in prompt


def test_public_api_exports_resolve_through_package_path():
    assert callable(build_welcome)
    assert FakeModelClient is not None
    assert CodingForMe is not None
    assert SessionStore is not None
    assert WorkspaceContext is not None
    assert Path(mini_pkg.__file__).as_posix().endswith("/codingforme/__init__.py")


def test_reviewer_skeleton_docs_exist():
    review_pack = Path("docs/review-pack/README.md")
    architecture = Path("docs/architecture/agent-harness-v1-overview.md")

    assert review_pack.exists()
    assert architecture.exists()

    review_text = review_pack.read_text(encoding="utf-8")
    assert "Project pitch" in review_text
    assert "Architecture map" in review_text
    assert "Benchmark evidence" in review_text
    assert "Sample run artifact list" in review_text

    architecture_text = architecture.read_text(encoding="utf-8")
    assert "Agent Harness v1" in architecture_text
    assert "task state" in architecture_text.lower()


def test_package_import_surface_includes_cli_entrypoints():
    assert callable(mini_pkg.main)
    assert callable(mini_pkg.build_agent)
    assert callable(mini_pkg.build_arg_parser)


def test_module_execution_help_works():
    result = subprocess.run(
        [sys.executable, "-m", "codingforme", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
