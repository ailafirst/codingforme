import io
import os
import re
import json
import subprocess
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest
from litellm.types.utils import ModelResponse

import codingforme as mini_pkg
from codingforme import models as models_module
from codingforme import (
    FakeModelClient,
    CodingForMe,
    OpenAICompatibleModelClient,
    SessionStore,
    WorkspaceContext,
    build_welcome,
)
from codingforme.models import (
    _CompatBackendCustomLLM,
    _send_with_retry,
    final_answer,
    force_tool_choice,
    tool_call,
)
from codingforme import tools as toolkit
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


def test_model_narration_on_a_tool_turn_is_kept_in_history(tmp_path):
    """工具轮里模型说的话必须留在 history，而且排在那次工具结果前面。

    为什么这条重要：不留的话，模型下一轮只看得见一串工具结果，看不见自己
    当时打算干什么、已经确认过什么，于是反复回读同一个文件。实测一次真实
    运行里 15 次模型调用有 9 次是这种重复劳动。
    """
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    call = tool_call("read_file", path="hello.txt")
    call["text"] = "Checking hello.txt before I decide what to patch."
    agent = build_agent(tmp_path, [call, final_answer("Done.")])

    agent.ask("Inspect hello.txt")

    roles = [item["role"] for item in agent.session["history"]]
    contents = [str(item.get("content", "")) for item in agent.session["history"]]
    assert "Checking hello.txt before I decide what to patch." in contents
    narration_index = contents.index("Checking hello.txt before I decide what to patch.")
    tool_index = next(i for i, item in enumerate(agent.session["history"]) if item["role"] == "tool")
    assert roles[narration_index] == "assistant"
    assert narration_index < tool_index, "说明文字要排在它导致的那次工具结果之前"


def test_a_tool_turn_without_narration_costs_no_history_budget(tmp_path):
    """模型没说话时，那条 assistant 记录不能占用 history 的文本预算。

    这条断言在改用标准 messages 数组时换了形状。原来的写法是「history 里不许有
    空内容的条目」，因为空条目当时纯属浪费预算。现在这条记录还承载着「这一轮发了
    哪几个调用」这个结构事实，messages 组装要靠它把工具结果配回发起它的那一轮，
    所以条目本身必须留下——**要守的是"不占文本预算"，不是"不存在"**。
    """
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [tool_call("read_file", path="hello.txt"), final_answer("Done.")],
    )

    agent.ask("Inspect hello.txt")

    for item in agent.session["history"]:
        if str(item.get("content", "")).strip():
            continue
        assert item.get("tool_calls"), "空内容的条目只有在承载 tool_calls 时才允许存在"
    # 文本视图里不能出现内容为空的 assistant 行（"[assistant] " 后面什么都没有）。
    for view in (agent.prompt("next"), agent.history_text()):
        assert not [line for line in view.splitlines() if line.rstrip() == "[assistant]"]


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
    # delegate 现在是有名字的变体（`delegate_tool`，默认关，见 tools.build_tool_registry）。
    agent = build_agent(
        tmp_path,
        feature_flags={"delegate_tool": True},
        outputs=[
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

    # 返回值不止确认「改了」，还要回显改完之后那一段长什么样——否则模型只能
    # 靠再读一次文件来验证，那是一整个模型往返的代价。格式与 read_file 一致。
    assert result.splitlines()[0] == "patched sample.txt"
    assert "   1: hello agent" in result
    assert file_path.read_text(encoding="utf-8") == "hello agent\n"


def test_patch_file_excerpt_shows_the_new_text_with_context(tmp_path):
    """回显必须覆盖到改动的全部行，并带上前后各两行上下文。

    只回显首行的话，多行 new_text 里除第一行外都看不见，模型照样要回读。
    """
    file_path = tmp_path / "CHANGELOG.md"
    file_path.write_text("# Changelog\n\n## Unreleased\n\n## 1.0\n- first\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool(
        "patch_file",
        {
            "path": "CHANGELOG.md",
            "old_text": "## Unreleased",
            "new_text": "## Unreleased\n- Changed default PORT from 8080 to 9090",
        },
    )

    # 补丁后文件是 7 行；改动落在第 3~4 行，前后各两行 → 窗口是第 1~6 行。
    assert "   1: # Changelog" in result
    assert "   3: ## Unreleased" in result
    assert "   4: - Changed default PORT from 8080 to 9090" in result
    assert "   6: ## 1.0" in result
    assert "- first" not in result, "窗口外的第 7 行不该被带出来"


def test_patch_file_excerpt_is_capped_so_a_huge_patch_cannot_flood_the_prompt(tmp_path):
    """回显有上限：它是给模型看的确认，不是文件回放。"""
    file_path = tmp_path / "big.txt"
    file_path.write_text("anchor\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool(
        "patch_file",
        {
            "path": "big.txt",
            "old_text": "anchor",
            "new_text": "\n".join(f"line {i}" for i in range(200)),
        },
    )

    assert "excerpt truncated" in result
    assert len(result.splitlines()) <= 45


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

    # 措辞要在「跨轮反复发同一调用」和「同一轮内发了三遍」两种场景下都读得通，
    # 并且按 P5 给出下一步该做什么，而不只是宣布失败。
    assert result.startswith("error: list_files was already called twice")
    assert "Use different arguments, a different tool, or return a final answer." in result


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


def test_welcome_survives_a_terminal_that_cannot_encode_its_glyphs(tmp_path, monkeypatch):
    """GBK 终端下横幅必须能打印出来，而不是让进程崩在打招呼这一步。

    这是实测撞到的：Windows 默认控制台是 GBK，横幅里的 `✦`(U+2726) 编不出来，
    `python -m codingforme` 直接抛 UnicodeEncodeError，agent 根本起不来。
    横幅只是最先撞上的，模型答案里带 emoji 一样会打断整个 REPL。
    """
    import io

    from codingforme.cli import _make_output_resilient, _terminal_safe

    agent = build_agent(tmp_path, [])
    welcome = build_welcome(agent, model="qwen3.5:4b")

    gbk_console = io.TextIOWrapper(io.BytesIO(), encoding="gbk", newline="")
    monkeypatch.setattr("sys.stdout", gbk_console)

    # 改动前这一行就是崩溃点。
    safe = _terminal_safe(welcome)
    safe.encode("gbk")  # 装饰字符全部降级成 ASCII，GBK 写得出来

    for fancy, plain in (("✦", "*"), ("✻", "*"), ("❯", ">"), ("‿", "_")):
        assert fancy not in safe
        assert plain in safe or fancy not in welcome
    # 框线和中文不受影响：GBK 编得出来的字符一个都不该被换掉。
    assert "╭" in safe and "│" in safe

    # errors="replace" 兜住无法预先枚举的内容（比如模型输出里的 emoji）。
    _make_output_resilient()
    gbk_console.write("模型回答里带了一个 🚀\n")
    gbk_console.flush()
    decoded = gbk_console.buffer.getvalue().decode("gbk")
    assert "模型回答里带了一个" in decoded, "中文必须原样保留，只有编不出的字符降级"


def test_dotenv_is_read_from_the_codingforme_repo_not_the_workspace(tmp_path):
    """`.env` 一律从本仓库找，与 --cwd 指向哪里无关。

    改动前 cli 用的是 `load_project_env(workspace.repo_root)`：工作区在仓库外面时
    （拿 agent 去改别的项目就是这种情况）往上走永远找不到本仓库的 `.env`，实测载入
    0 个键。它看着能用只是因为 `import litellm` 会顺手 `load_dotenv()` 读进程 cwd
    ——配置实际是第三方副作用喂进来的，换个目录启动就报缺凭证。
    """
    from codingforme.config import find_project_env, project_root

    repo = project_root()
    assert (repo / "codingforme" / "cli.py").is_file(), "project_root 必须指向本仓库根目录"

    # 工作区在仓库外：它自己往上找不到任何 .env，但 project_root() 能。
    outside = tmp_path / "some" / "other" / "project"
    outside.mkdir(parents=True)
    assert find_project_env(outside) is None
    if (repo / ".env").exists():
        assert find_project_env(repo) == repo / ".env"



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
    # payload 恒为列表，单个调用也不例外——ask() 因此只有一种处理路径。
    assert payload == [{"name": "list_files", "args": {"path": "."}}]


def test_parse_accepts_more_than_one_native_tool_call():
    """一轮多个调用是合法输入，按顺序全部返回。

    这里以前断言 retry：多于一个就把整轮判废重来。放弃那个做法是因为代价不对称——
    这个后端实测会一轮发 3~4 个（同一提示 5 次里有 3 次），`parallel_tool_calls: false`
    它又不理会，于是每次都白烧一个约 17 秒的固定往返，还把本可一轮做完的事拆成多轮。
    """
    kind, payload = CodingForMe.parse(
        {
            "text": "",
            "tool_calls": [
                {"name": "list_files", "args": {"path": "."}},
                {"name": "read_file", "args": {"path": "README.md"}},
            ],
        }
    )
    assert kind == "tool"
    assert payload == [
        {"name": "list_files", "args": {"path": "."}},
        {"name": "read_file", "args": {"path": "README.md"}},
    ]


def test_parse_retries_when_any_call_in_a_batch_is_malformed():
    """一批里只要有一个形状不合法，整批走 retry。

    不做「跳过坏的、执行好的」：模型发的是一组有先后关系的动作，静默丢掉中间一个
    会让后面几个建立在错误前提上，而模型完全看不出发生过这件事。
    """
    kind, _ = CodingForMe.parse(
        {
            "text": "",
            "tool_calls": [
                {"name": "list_files", "args": {"path": "."}},
                {"name": "", "args": {}},
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
    assert payload == [{"name": "list_files", "args": {"path": "."}}]

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


def test_agent_executes_every_call_of_a_multi_call_turn_in_order(tmp_path):
    """一轮发三个调用，三个都要执行，顺序不变，各自计一步。"""
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("gamma\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            {
                "text": "Reading all three.",
                "tool_calls": [
                    {"name": "read_file", "args": {"path": "a.txt"}},
                    {"name": "read_file", "args": {"path": "b.txt"}},
                    {"name": "read_file", "args": {"path": "c.txt"}},
                ],
            },
            {"text": "Done.", "tool_calls": None},
        ],
    )

    assert agent.ask("read a, b and c") == "Done."

    tool_entries = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert [item["args"]["path"] for item in tool_entries] == ["a.txt", "b.txt", "c.txt"]
    assert "alpha" in tool_entries[0]["content"]
    assert "gamma" in tool_entries[2]["content"]
    # 模型那轮的说明文字排在三条结果之前，只记一次而不是每个调用记一次
    assert [item["content"] for item in agent.session["history"] if item["role"] == "assistant"].count("Reading all three.") == 1


def test_a_failing_call_does_not_stop_the_rest_of_the_batch(tmp_path):
    """批次里一个调用失败，后面的照常执行——失败以字符串反馈，不抛异常。"""
    (tmp_path / "ok.txt").write_text("fine\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            {
                "text": "",
                "tool_calls": [
                    {"name": "read_file", "args": {"path": "../escape.txt"}},
                    {"name": "read_file", "args": {"path": "ok.txt"}},
                ],
            },
            {"text": "Done.", "tool_calls": None},
        ],
    )

    assert agent.ask("read both") == "Done."

    tool_entries = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert len(tool_entries) == 2
    assert "escapes workspace" in tool_entries[0]["content"]
    assert "fine" in tool_entries[1]["content"]


def test_step_budget_stops_a_batch_midway_and_tells_the_model_which_calls_were_skipped(tmp_path):
    """预算在批次中途用完时，剩下的不执行，而且要明确告诉模型哪些没跑。

    静默丢弃是不行的：模型会以为那些调用都做过了，下一轮基于错误前提继续。
    """
    for name in ["a.txt", "b.txt", "c.txt"]:
        (tmp_path / name).write_text(name, encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            {
                "text": "",
                "tool_calls": [
                    {"name": "read_file", "args": {"path": "a.txt"}},
                    {"name": "read_file", "args": {"path": "b.txt"}},
                    {"name": "read_file", "args": {"path": "c.txt"}},
                ],
            },
            {"text": "Done.", "tool_calls": None},
        ],
        max_steps=2,
    )

    agent.ask("read all three")

    # 每个调用都要有紧跟其后的结果——**包括没执行的那个**。以前这里写的是一条
    # 汇总通知，模型得自己把「哪条结果属于哪个调用」推理出来。
    tool_entries = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert [item["args"]["path"] for item in tool_entries] == ["a.txt", "b.txt", "c.txt"]
    assert [item.get("executed", True) for item in tool_entries] == [True, True, False]
    assert "not executed" in tool_entries[2]["content"]
    assert "Re-issue this call" in tool_entries[2]["content"], "要告诉模型下一步该做什么"
    # 没执行的调用不占步数，也不能进任务状态。
    assert agent.current_task_state.tool_steps == 2


def test_an_unexecuted_call_is_traced_as_skipped_not_executed(tmp_path):
    """没跑的调用写 `tool_skipped`。

    混进 `tool_executed` 会让 `calls_per_turn` 以及所有 L1 断言把没跑的调用
    算成跑了——而那种漏算在报告里长得和正常数据一模一样。
    """
    for name in ["a.txt", "b.txt", "c.txt"]:
        (tmp_path / name).write_text(name, encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            {
                "text": "",
                "tool_calls": [
                    {"name": "read_file", "args": {"path": "a.txt"}},
                    {"name": "read_file", "args": {"path": "b.txt"}},
                    {"name": "read_file", "args": {"path": "c.txt"}},
                ],
            },
            {"text": "Done.", "tool_calls": None},
        ],
        max_steps=2,
    )
    agent.ask("read all three")

    events = []
    for trace_file in (tmp_path / ".codingforme" / "runs").rglob("trace.jsonl"):
        for line in trace_file.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    executed = [e for e in events if e.get("event") == "tool_executed"]
    skipped = [e for e in events if e.get("event") == "tool_skipped"]

    assert len(executed) == 2
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "step_budget_exhausted"
    assert skipped[0]["call_index"] == 2
    assert skipped[0]["call_count"] == 3


def test_reissuing_a_skipped_call_is_not_blocked_as_a_repeat(tmp_path):
    """被预算卡掉的调用，之后如实重发不能被判成「重复调用」。

    重发只可能发生在**下一次** `ask()`：预算在批次中途耗尽时，这一次 `ask()`
    的控制循环也随之结束，同一次请求里没有重发的机会。history 跨请求保留，
    所以未执行的那条记录会一直躺在重复检测的观察窗口里。

    构造成「同一批发了两个一模一样的调用」是为了让新旧实现真的分叉：
    旧实现看最近两条工具记录（一条执行过、一条没执行）判定为重复并拦下，
    模型于是永远拿不到我们刚刚请它重发的那个结果。
    """
    (tmp_path / "c.txt").write_text("CCC", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            # 第一次请求：一批两个相同调用，第一个执行、第二个被预算卡掉。
            {
                "text": "",
                "tool_calls": [
                    {"name": "read_file", "args": {"path": "c.txt"}},
                    {"name": "read_file", "args": {"path": "c.txt"}},
                ],
            },
            # 第二次请求：模型照提示重发那个没跑成的调用。
            {"text": "", "tool_calls": [{"name": "read_file", "args": {"path": "c.txt"}}]},
        ],
        max_steps=1,
    )
    agent.ask("read it")
    agent.ask("try that again")

    tool_entries = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert [item.get("executed", True) for item in tool_entries] == [True, False, True]
    assert "CCC" in tool_entries[2]["content"], (
        f"重发应当真的读到内容，实际：{tool_entries[2]['content']!r}"
    )


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


def _http_error(code, body):
    return urllib.error.HTTPError(
        "https://example.invalid/v1/chat/completions", code, "Bad Request", {}, io.BytesIO(body.encode("utf-8"))
    )


def test_a_4xx_that_is_really_an_upstream_connection_failure_gets_retried():
    """网关把自己这侧的连接失败报成 400 时，必须当成可重试的抖动。

    真实事故：一次 2.4 小时的 k=3 跑批在第三轮第 10 个任务上收到
    `HTTP 400 ... finishConnect(..) failed: Connection refused: ...:80`，
    那个地址是**服务端内网 IP**，与我们发出的请求无关。当时按状态码判成
    客户端错误直接放弃，整轮跑批连同已完成的两轮聚合一起丢了。
    """
    body = (
        '{"error":{"code":"400","message":"Request failed",'
        '"param":"finishConnect(..) failed: Connection refused: host.internal/10.137.1.77:80","type":""}}'
    )
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_error(400, body)
        return _FakeHTTPResponse('{"ok": true}')

    with patch("codingforme.models.urllib.request.urlopen", fake_urlopen), \
            patch("codingforme.models.time.sleep", lambda _seconds: None):
        text, _content_type = _send_with_retry(object(), 30, "mimo-v2.5")

    assert text == '{"ok": true}'
    assert calls["n"] == 3, "应当重试到成功，而不是在第一个 400 上放弃"


def test_a_genuine_4xx_is_not_retried():
    """请求本身不合法时不能重试。

    那是我们自己的 bug，重试三次只会白烧三个约 17 秒的往返，还把错误现场推迟。
    这条测试锁住"放宽重试范围"没有被顺手放宽成"所有 400 都重试"。
    """
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        raise _http_error(400, '{"error":{"message":"unknown field: tolls"}}')

    with patch("codingforme.models.urllib.request.urlopen", fake_urlopen), \
            patch("codingforme.models.time.sleep", lambda _seconds: None):
        with pytest.raises(RuntimeError, match="HTTP 400"):
            _send_with_retry(object(), 30, "mimo-v2.5")

    assert calls["n"] == 1, "格式错误的请求必须立刻报出来，不要重试"


class _FakeHTTPResponse:
    headers = {"Content-Type": "application/json"}

    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body.encode("utf-8")


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
    assert CodingForMe.parse(call) == ("tool", [{"name": "read_file", "args": {"path": "a.txt", "start": 1, "end": 2}}])

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
    from codingforme.models import count_tokens

    assert metadata["current_request"]["rendered_tokens"] == count_tokens("recall")


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


def durable_memory_text(root):
    """记忆库里所有文件拼起来的文本。

    v2 之后一条记忆一个文件、文件名由内容 slug 出来，所以测试不再断言具体路径——
    断言路径等于把「文件叫什么」也变成契约，而那不是这几条用例要守的东西。
    """
    memory_root = Path(root) / ".codingforme" / "memory"
    if not memory_root.exists():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(memory_root.rglob("*.md"))
    )


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
    report = json.loads(agent.run_store.report_path(agent.current_task_state).read_text(encoding="utf-8"))
    index_text = index_path.read_text(encoding="utf-8")
    stored = durable_memory_text(tmp_path)

    assert index_path.exists()
    # 索引一行一条记忆，那一行就是描述——召回只读它，所以它必须带得动内容。
    assert "Use constrained tools instead of guessing." in index_text
    assert "Keep durable memory topic-based and lightweight." in index_text
    assert "Use constrained tools instead of guessing." in stored
    assert "Keep durable memory topic-based and lightweight." in stored
    assert report["durable_promotions"] == [
        "project: Use constrained tools instead of guessing.",
        "project: Preserve local agent state under .codingforme/.",
        "project: Keep durable memory topic-based and lightweight.",
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

    stored = durable_memory_text(tmp_path)

    assert "优先使用受约束工具，不要靠猜。" in stored
    assert "持久记忆保持轻量、按 topic 管理。" in stored


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
    stored = durable_memory_text(tmp_path)

    assert report["durable_promotions"] == [
        "project: Use constrained tools instead of guessing.",
    ]
    # 拒绝归因用的名字和入库用的是同一套（类型名），否则同一条记忆「进了」和
    # 「被拒了」在工件上会写成两种命名体系。
    assert report["durable_rejections"] == [
        "reference:secret_shaped",
        "project:transient_task_state",
        "reference:noisy_output",
    ]
    assert "Use constrained tools instead of guessing." in stored
    assert "sk-live-secret-abc" not in stored
    assert "FAIL test_one" not in stored


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

    report = json.loads(agent.run_store.report_path(agent.current_task_state).read_text(encoding="utf-8"))
    text = durable_memory_text(tmp_path)

    assert "Python runtime is 3.12." in text
    assert "Python runtime is 3.11." not in text
    assert report["durable_superseded"] == [
        "reference: Python runtime is 3.11. -> Python runtime is 3.12.",
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

    memory_root = tmp_path / ".codingforme" / "memory" / "topics"
    bodies = [path.read_text(encoding="utf-8") for path in sorted(memory_root.glob("*.md"))]

    # 同一条事实提升两次，库里仍然只有一个文件——这就是去重。断言文件数而不是
    # 出现次数：后者会把文件格式（描述行 + 正文各出现一次）也变成契约。
    assert len(bodies) == 1
    assert "Use constrained tools instead of guessing." in bodies[0]


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
    # 用互不相同的词而不是重复字符：预算按 token 判之后，"A"*320 只值个位数 token，
    # 较早那段根本撑不到需要压缩，用例会静默失去意义。
    old_text = "OLD- " + " ".join(f"oldword{n}" for n in range(60))
    recent_text = "RECENT- " + " ".join(f"newword{n}" for n in range(60))

    # 2 条旧的 + 18 条新的。12 是最近窗口的条目上限（6 个工具轮 × 每轮最多 2 条
    # 记录：模型说明 + 工具结果），而这条上限本身按 2 * RECENT_WINDOW_BLOCK = 6 条
    # 成块推进，实际窗口在 12~17 之间浮动，所以要垫到 18 才能让前两条稳定落到
    # 「较早」那一段。
    agent.record({"role": "user", "content": old_text, "created_at": "2026-04-07T09:00:00+00:00"})
    agent.record({"role": "assistant", "content": old_text, "created_at": "2026-04-07T09:01:00+00:00"})
    for minute in range(2, 20):
        role = "user" if minute % 2 == 0 else "assistant"
        agent.record(
            {
                "role": role,
                "content": recent_text,
                "created_at": f"2026-04-07T09:{minute:02d}:00+00:00",
            }
        )

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


# --- 缺口一：工具 schema 与错误信息 ------------------------------------------
#
# 这一组用例守的是「模型能从工具定义和报错里学到什么」。判据全部来自 k=3 真实
# 跑批的被拒调用分布（342 次调用、50 次被拒）：路径写法 36%、patch 的 old_text
# 没对上 14%、审批拒绝 10%、路径不存在 10%。每条断言都对应其中一类，不对应任何
# 一类的（比如"多余参数"）没有用例，因为那一类的观测次数是 0。


def test_every_path_argument_explains_the_reference_frame():
    """凡是叫 path 的参数，都必须说清路径相对谁、什么写法会被拒。

    实测占比最大的一类失败（36%）来自模型不知道该用相对还是绝对路径——而它唯一
    见过的路径样本是 workspace 快照里那行绝对路径的 repo_root。工具定义不说，
    模型照抄它是完全合理的推断。
    """
    tools = dict(toolkit.BASE_TOOL_SPECS)
    tools["delegate"] = toolkit.DELEGATE_TOOL_SPEC
    path_params = [
        (name, spec["schema"]["path"])
        for name, spec in tools.items()
        if "path" in spec["schema"]
    ]

    assert path_params, "工具里应当有 path 参数，否则这条用例失去了对象"
    for name, param in path_params:
        description = param.get("description", "") if isinstance(param, dict) else ""
        assert "relative to the repo root" in description, f"{name}.path 没说清参照系"
        assert ".." in description, f"{name}.path 没说 '..' 会被拒"


def test_function_specs_carry_descriptions_bounds_and_defaults():
    """长写法的字段规格要完整翻译进标准 JSON Schema。

    这些键是模型在**调用之前**唯一能看到的约束。丢掉任何一个，schema 就只剩
    类型，等于把"什么是合法输入"重新推回给模型猜。
    """
    specs = {
        spec["function"]["name"]: spec["function"]["parameters"]
        for spec in toolkit.to_openai_function_specs(
            {**toolkit.BASE_TOOL_SPECS, "delegate": toolkit.DELEGATE_TOOL_SPEC}
        )
    }

    read_file = specs["read_file"]["properties"]
    assert read_file["start"] == {
        "type": "integer",
        "description": "First line to read, counting from 1.",
        "minimum": 1,
        "default": 1,
    }
    assert specs["run_shell"]["properties"]["timeout"]["maximum"] == 120
    assert specs["list_files"]["properties"]["path"]["default"] == "."
    # 必填/选填的判定不受长写法影响：有默认值就是选填。
    assert specs["read_file"]["required"] == ["path"]
    assert specs["patch_file"]["required"] == ["path", "old_text", "new_text"]


def test_function_specs_reject_unknown_arguments():
    specs = toolkit.to_openai_function_specs(toolkit.BASE_TOOL_SPECS)

    assert all(spec["function"]["parameters"]["additionalProperties"] is False for spec in specs)


def test_short_form_schema_fields_still_work():
    """裸字符串写法必须继续可用——测试里到处都是它，工具定义也允许混用。"""
    specs = toolkit.to_openai_function_specs(
        {"probe": {"schema": {"a": "str", "b": "int=7"}, "risky": False, "description": "d"}}
    )
    parameters = specs[0]["function"]["parameters"]

    assert parameters["properties"] == {"a": {"type": "string"}, "b": {"type": "integer"}}
    assert parameters["required"] == ["a"]


def test_prefix_lists_tools_readably_and_shows_example_arguments(tmp_path):
    """prefix 里的工具清单要是人/模型都读得懂的一行，并带上示例参数。

    两件事：一是长写法的字段规格不能把 Python dict 字面量印进 prompt；二是示例
    参数此前只在校验失败之后才回给模型，而 prefix 走前缀缓存（实测缓存占输入
    token 的 75.4%），事前给几乎免费，事后给要烧掉一整个约 17 秒的往返。
    """
    agent = build_agent(tmp_path, [])

    prefix = agent.build_prefix().text

    assert "- read_file(path: str, start: int=1, end: int=200) [safe]" in prefix
    assert 'example args: {"path": "README.md", "start": 1, "end": 80}' in prefix
    assert "'description':" not in prefix, "长写法的字段规格被原样印进了 prompt"
    # 路径参照系在 prompt 规则里也要说一次，不能只藏在工具参数说明里。
    assert 'Every path argument is relative to the repo root' in prefix


def test_path_escape_error_tells_the_model_the_correct_form(tmp_path):
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("read_file", {"path": "../outside.txt"})

    # 前缀不能动：run_tool 靠它把这次拒绝标成 path_escape 安全事件。
    assert "path escapes workspace" in result
    assert agent._last_tool_result_metadata["security_event_type"] == "path_escape"
    # 后半句是新增的修复动作。
    assert "relative to the repo root" in result


def test_missing_path_error_points_at_a_next_step(tmp_path):
    (tmp_path / "pkg").mkdir()
    agent = build_agent(tmp_path, [])

    missing = agent.run_tool("read_file", {"path": "pkg/nope.py"})
    wrong_kind = agent.run_tool("read_file", {"path": "pkg"})

    assert "no such file: pkg/nope.py" in missing
    assert "list_files on 'pkg'" in missing, "报错要指出该去哪儿找，而不是让模型重试同一个路径"
    assert "is a directory, not a file" in wrong_kind
    assert "list_files" in wrong_kind


def test_patch_miss_and_ambiguity_get_opposite_advice(tmp_path):
    """命中 0 次和命中多次的修复方向相反，报错必须分开说。

    0 次 → 把文本抄准（多半是缩进/空白没对上）；多次 → 把范围放大到唯一。
    原来两种共用一句 "must occur exactly once, found N"，模型只能自己反推。
    """
    (tmp_path / "a.py").write_text("x = 1\ny = 2\nx = 1\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    missed = agent.run_tool("patch_file", {"path": "a.py", "old_text": "z = 3", "new_text": "q"})
    ambiguous = agent.run_tool("patch_file", {"path": "a.py", "old_text": "x = 1", "new_text": "q"})

    assert "was not found" in missed
    assert "byte for byte" in missed and "read_file" in missed
    assert "occurs 2 times" in ambiguous
    assert "more lines above and below" in ambiguous
    # 两条建议不能互串：抄准和放大范围是相反的动作。
    assert "byte for byte" not in ambiguous
    assert "unique" not in missed


def test_delegate_step_budget_is_bounded(tmp_path):
    """schema 上写了 minimum/maximum，校验就必须真的执行它。

    schema 声明了约束却不执行，比不声明更糟：模型据此以为 max_steps=999 合法。
    """
    agent = build_agent(tmp_path, [], feature_flags={"delegate_tool": True})

    result = agent.run_tool("delegate", {"task": "look around", "max_steps": 999})

    assert f"max_steps must be in [1, {toolkit.DELEGATE_MAX_STEPS_CEILING}]" in result


# --- T1-6：参数归一 ---------------------------------------------------------


def test_string_integers_are_normalised_before_the_gate(tmp_path):
    """模型把整数写成字符串时，归一必须发生在闸口之前而不是 runner 内部。

    这是实测出来的漏洞：`validate_tool()` 里散落着 `int(args.get(...))`，局部归一
    让那一次执行成功，但 history / trace / 记忆 / 重复检测拿到的仍是原始值。
    于是同一个动作换个写法就绕过了重复检测：

        {"start": "1"} → 执行    {"start": "1"} → 执行    {"start": 1} → 又执行

    重复调用是被拒调用里最大的一类（真实跑批占 37%），检测漏一档就没法区分
    「模型少打转了」和「我们少查了几次」。
    """
    (tmp_path / "a.txt").write_text("A\nB\nC\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            tool_call("read_file", path="a.txt", start="1", end="3"),
            tool_call("read_file", path="a.txt", start="1", end="3"),
            tool_call("read_file", path="a.txt", start=1, end=3),
            final_answer("done"),
        ],
        max_steps=6,
    )

    agent.ask("read it")

    tool_items = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert [item["args"]["start"] for item in tool_items] == [1, 1, 1], "history 里记的应当是归一后的值"
    assert "already called twice" in tool_items[2]["content"], (
        f"换成 int 写法的同一调用应当被判重复，实际：{tool_items[2]['content']!r}"
    )


def test_coercion_leaves_unconvertible_values_for_validation_to_reject(tmp_path):
    """归一不是校验：转不动就原样放行，让 validate_tool 报一个说得清的错。

    在归一层抛异常会把「类型不对」变成一条来自归一层的、模型看不懂的报错。
    """
    (tmp_path / "a.txt").write_text("A\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("read_file", {"path": "a.txt", "start": "not-a-number"})

    assert result.startswith("error: invalid arguments for read_file")


def test_coercion_never_rewrites_none_or_booleans():
    """None 不能变成 "None"，bool 不能变成 0/1。

    `str(None)` == "None" 会让 patch_file 拿着字面量 "None" 去文件里找；
    Python 里 `bool` 是 `int` 的子类，静默转成 0/1 会掩盖模型发错类型这件事。
    """
    coerced = toolkit.coerce_tool_args(
        "patch_file", {"path": "a.py", "old_text": None, "new_text": True}
    )

    assert coerced["old_text"] is None
    assert coerced["new_text"] is True


def test_coercion_keeps_unknown_arguments(tmp_path):
    """未知参数原样保留，不静默丢弃。

    丢掉会让模型以为它发的东西被接受了；留着则会走到 validate/runner 那里，
    表现为一个它能看懂的错误或被忽略。
    """
    coerced = toolkit.coerce_tool_args("read_file", {"path": "a.txt", "bogus": "x"})

    assert coerced["bogus"] == "x"


def test_coercion_is_idempotent_and_ignores_unknown_tools():
    once = toolkit.coerce_tool_args("run_shell", {"command": "ls", "timeout": "30"})
    twice = toolkit.coerce_tool_args("run_shell", once)

    assert once == {"command": "ls", "timeout": 30}
    assert twice == once
    # 未知工具没有 schema 可依，原样返回而不是抛异常。
    assert toolkit.coerce_tool_args("no_such_tool", {"a": "1"}) == {"a": "1"}


def _stale_checkpoint_session(agent, tmp_path):
    """把 session 布置成「checkpoint 记的文件内容已经变了」。

    与 `evaluator._apply_task_setup()` 的 `freshness_mismatch` 布景同构：
    先给 runtime.py 存一份摘要并把它的 sha256 写进 checkpoint，再改文件，
    于是下一次 `evaluate_resume_state()` 会判成 partial-stale。
    """
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
    (tmp_path / "runtime.py").write_text("beta\n", encoding="utf-8")


def test_the_report_records_the_resume_state_the_run_started_from(tmp_path):
    """report 里的 resume_status 说的是「这次运行从什么状态起步」，不随轮次漂。

    踩过的坑：`report["prompt_metadata"]` 存的是**最后一轮**的元数据，而
    `self.resume_state` 每次 refresh_prefix() 都重算——partial-stale 被重新
    锚定之后就回到 full-valid 了。于是「这次运行是不是从一个过期 checkpoint
    起步的」这个事实，只有恰好一轮就结束的运行才看得见；真实模型多走一步就
    查不到，基准里两个 resume 任务因此 0/3 恒挂。
    """
    (tmp_path / "runtime.py").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(tmp_path, [final_answer("checkpoint ready.")])
    _stale_checkpoint_session(agent, tmp_path)

    resumed = CodingForMe.from_session(
        model_client=FakeModelClient(
            [tool_call("read_file", path="runtime.py"), final_answer("Resumed.")]
        ),
        workspace=build_workspace(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.ask("Continue the task") == "Resumed."
    report = resumed.run_store.load_report(resumed.current_task_state.run_id)

    # 顶层字段是运行级事实：起步时是过期的。
    assert report["resume_status"] == "partial-stale"
    # 逐轮字段仍然是逐轮的——重锚之后回到 full-valid 是对的，不要去"修"它。
    assert report["prompt_metadata"]["resume_status"] == "full-valid"


def test_the_run_start_resume_state_survives_a_setup_applied_after_construction(tmp_path):
    """构造之后才布置的 stale 场景，也要能被顶层字段看见。

    `BenchmarkEvaluator` 就是这个顺序：先建 agent，再 `_apply_task_setup()`
    把 checkpoint 写脏，然后才 ask()。而 `self.resume_state` 是在 __init__ 里
    算的——那时候还没脏。所以顶层字段不能直接抄构造期的值。
    """
    (tmp_path / "runtime.py").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path, [tool_call("read_file", path="runtime.py"), final_answer("Resumed.")]
    )
    assert agent.resume_state["status"] == "no-checkpoint"

    _stale_checkpoint_session(agent, tmp_path)

    assert agent.ask("Continue the task") == "Resumed."
    report = agent.run_store.load_report(agent.current_task_state.run_id)
    assert report["resume_status"] == "partial-stale"


def test_tool_examples_ride_in_the_schema_the_backend_actually_parses():
    """示例必须进 `parameters.examples`，不是 Anthropic 的 `input_examples`。

    实测（16 次采样 × 3 组，工具参数格式只能从示例得知）：这个 OpenAI 兼容后端
    把 JSON Schema 的标准关键字 `parameters.examples` 原样送到模型面前（16/16
    吐出示例里那个无从推导的值），而 `function.input_examples` 被静默丢弃
    （0/16）——不报错，只是没有效果。所以字段名不能换。
    """
    specs = to_openai_function_specs(toolkit.BASE_TOOL_SPECS)
    by_name = {spec["function"]["name"]: spec["function"] for spec in specs}

    for name, raw in toolkit.TOOL_EXAMPLES.items():
        if name not in by_name:  # delegate 只在深度够时才注册
            continue
        assert by_name[name]["parameters"]["examples"] == [json.loads(raw)]
        assert "input_examples" not in by_name[name]

    # 示例本身必须是合法参数：教给模型一份过不了自己校验的调用毫无意义。
    for name, raw in toolkit.TOOL_EXAMPLES.items():
        json.loads(raw)


def test_a_retry_records_why_the_turn_was_thrown_away(tmp_path):
    """trace 上要分得出三类 retry，因为它们该改的东西完全相反。

    空响应 / 推理阶段被 max_tokens 截断要调 `max_new_tokens`，形状不合法要改工具
    schema 和示例。此前工件上三者都只是 `kind == "retry"`，分不出来就只能靠重跑
    撞见。归因码见 runtime 的 `RETRY_REASON_*`。
    """
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            # 后端返回了合法结构但内容为空：推理模型被 max_tokens 截断时的形状。
            {"text": "", "tool_calls": None},
            # 工具调用缺名字。
            {"text": "", "tool_calls": [{"args": {}}]},
            final_answer("Done."),
        ],
    )
    assert agent.ask("Do the thing") == "Done."

    run_dir = next(path for path in (tmp_path / ".codingforme" / "runs").iterdir() if path.is_dir())
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    parsed = [event for event in events if event["event"] == "model_parsed"]
    assert [event["kind"] for event in parsed] == ["retry", "retry", "final"]
    assert [event["retry_reason"] for event in parsed] == [
        "empty_response",
        "missing_tool_name",
        # 非 retry 的轮次恒为空串，别让读者以为漏记了。
        "",
    ]


def test_retry_notices_never_teach_the_tag_syntax(tmp_path):
    """retry notice 是第三条会泄漏协议写法的通道，和 prefix、报错信息同等约束。

    曾经这里写着 "model returned an empty <final> answer"——模型出错的那一刻，
    我们正好把它推向一套既没发 schema 也不打算支持的协议。
    """
    agent = build_agent(tmp_path, [])
    for raw in (
        {"text": "", "tool_calls": None},
        {"text": "", "tool_calls": [{"args": {}}]},
        {"text": "", "tool_calls": ["not-an-object"]},
        {"text": "", "tool_calls": [{"name": "read_file", "args": "oops"}]},
        "<tool>{not json",
        "<final></final>",
        "<tool name=>",
    ):
        kind, payload = agent.parse(raw)
        assert kind == "retry", raw
        assert "<tool" not in payload
        assert "<final" not in payload
        # 每一条都要带得出归因码，否则 trace 上又退回「只知道作废了」。
        assert payload.reason


def _cut_registry(agent, keep):
    """把注册表裁到 keep，模拟 HarnessSpec 按白名单裁剪之后的 agent。"""
    agent.tools = {name: spec for name, spec in agent.tools.items() if name in keep}
    agent.refresh_prefix(force=True)
    return agent


def test_no_advice_ever_names_a_tool_the_registry_does_not_have(tmp_path):
    """报错与 prompt 规则都不许把模型指向一个它调不到的工具。

    背景：任务声明的 `allowed_tools` 接上之后（N-5），注册表会被裁到只剩两三个
    工具，而当时有五处文案是硬写的——两条 prompt 规则、两句路径类报错、一句
    patch 不匹配的报错。模型照着做只会拿回一句 `unknown tool`，白烧一个约 17 秒
    的往返，而且多半会把同一个路径原样再试一次，正好撞上 `no_repeated_calls`。
    """
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    agent = _cut_registry(build_agent(tmp_path, []), {"read_file", "patch_file"})
    absent = {"list_files", "search", "write_file", "run_shell", "delegate"}

    # 按词边界匹配：patch_file 的示例参数里有个 `binary_search.py`，
    # 裸子串会把它误判成在推荐 search 工具。
    def names_in(text):
        return {name for name in absent if re.search(rf"{name}", text)}

    # 通道一：prompt（规则段 + 工具清单，都进 system 消息）。
    assert names_in(agent.prefix) == set(), "prefix 仍然提到了不可用的工具"

    # 通道二：参数校验失败后回给模型的报错。
    advice = [
        agent.run_tool("read_file", {"path": "sub"}),          # 是目录不是文件
        agent.run_tool("read_file", {"path": "nope.txt"}),     # 路径不存在
        agent.run_tool("patch_file", {"path": "a.txt", "old_text": "zzz", "new_text": "q"}),
    ]
    for message in advice:
        assert message.startswith("error:"), message
        assert names_in(message) == set(), f"报错建议里出现了不可用的工具：{message}"

    # 通道三：调到被裁掉的工具时，报错要把「那我能调什么」一并给出。
    unknown = agent.run_tool("list_files", {"path": "."})
    assert unknown == "error: unknown tool 'list_files'. Available tools: patch_file, read_file"


def test_advice_still_names_the_tools_that_are_available(tmp_path):
    """反向证据：注册表齐全时那几句建议必须照旧出现。

    没有这条，上一条测试用「把所有建议都删掉」也能通过。
    """
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    agent = build_agent(tmp_path, [])

    assert "use write_file or patch_file instead of repeatedly listing files" in agent.prefix
    assert "Use list_files to see what is inside it." in agent.run_tool("read_file", {"path": "sub"})
    assert "Use list_files on '.' or search to find the right path" in agent.run_tool(
        "read_file", {"path": "nope.txt"}
    )
    assert "Read the file with read_file" in agent.run_tool(
        "patch_file", {"path": "a.txt", "old_text": "zzz", "new_text": "q"}
    )


def test_a_registry_with_no_write_tools_drops_the_rule_instead_of_emptying_it(tmp_path):
    """一条工具都不剩时整句删掉，而不是留一句指向空集的规则。"""
    agent = _cut_registry(build_agent(tmp_path, []), {"read_file"})
    assert "instead of repeatedly listing files" not in agent.prefix
    assert "Do not call read_file with args={}." in agent.prefix
    assert "__WRITE_RULE__" not in agent.prefix and "__REQUIRED_ARGS_RULE__" not in agent.prefix


def test_the_context_window_is_resolved_through_four_layers_in_order():
    """窗口大小的来源要分得清：人定的 / 已知后端表 / litellm 注册表 / 回落默认值。

    只记一个预算值是不够的——预算随环境变化（换 provider、litellm 升级映射表）
    会让两次跑批不可比，而工件上看不出它是哪来的。
    """
    from codingforme import models

    assert models.resolve_context_window("mimo-v2.5", explicit=32_000) == (32_000, "explicit")
    assert models.resolve_context_window("mimo-v2.5") == (1_000_000, "known-model")
    assert models.resolve_context_window("gpt-4o") == (128_000, "litellm-registry")
    window, source = models.resolve_context_window("no-such-model-anywhere")
    assert (window, source) == (models.DEFAULT_WINDOW_TOKENS, "default")


def test_a_detected_window_is_floored_to_a_bucket_before_it_becomes_a_budget():
    """探测值要向下取整到档位再用，不直接当预算。

    档位让 total_budget 成为一个可声明、可复现的值：直接用探测值的话，换个
    provider 或 litellm 升级一次映射表，HarnessSpec 指纹就跟着变，「两次跑批
    能不能放一起比」这条保证就没了。向下取整而不是四舍五入，是因为猜大了会撞
    后端 400、丢掉整轮已经做完的工具执行。
    """
    from codingforme import models

    assert models.bucket_window(200_000) == 128_000
    assert models.bucket_window(128_000) == 128_000
    assert models.bucket_window(100) == models.WINDOW_BUCKETS[0]
    # 百万窗口也不会让预算开到百万：上下文腐坏和计费都在那之前就先到了。
    budget = models.context_budget_tokens(1_000_000, tool_schema_tokens=0, output_reserve_tokens=0)
    assert budget < models.EFFECTIVE_WINDOW_CAP_TOKENS
    assert budget > models.EFFECTIVE_WINDOW_CAP_TOKENS * 0.9


def test_the_budget_deducts_message_framing_as_an_absolute_amount_not_a_ratio():
    """结构开销按**消息条数**扣，不按 prompt 长度扣。

    实测（`scripts/measure_token_accounting.py`，mimo-v2.5，23 轮真实请求）：
    `residual = input_tokens − prompt_tokens − schema` 对消息条数回归 R²=0.838，
    对 prompt 长度回归只有 0.396。原先那个乘在窗口上的 0.8 系数方向就是错的——
    它在 128k 档一次扣掉 25,600 个 token 去覆盖一笔实测约 1,200 的开销。

    这条锁住形状：步数上限翻倍时扣除额随之增加，而窗口翻倍时**每条消息的**
    扣除额不变。
    """
    from codingforme import models

    few = models.budget_breakdown(128_000, expected_messages=5)
    many = models.budget_breakdown(128_000, expected_messages=45)

    per_message = models.MESSAGE_FRAMING_TOKENS_PER_MESSAGE
    assert many["message_framing_tokens"] - few["message_framing_tokens"] == 40 * per_message
    # 窗口变大不改变结构开销：它跟着条数走，不跟着窗口走。
    assert models.budget_breakdown(64_000, expected_messages=45)["message_framing_tokens"] == (
        many["message_framing_tokens"]
    )
    # 消息条数由步数上限推出来，`5 + 2 × steps`（实测 25 次调用时是 54 条）。
    assert models.expected_message_count(25) == 55


def test_a_window_too_small_to_hold_the_deductions_says_so_instead_of_pretending():
    """扣完四笔仍然不够时要把 `floored` 标出来，不能静默返回下限。

    静默的话调用方会以为「还有 1000 个 token 可用」，而实际上这个窗口配置根本
    装不下 schema + 输出预留，该做的是换档位或调小 max_new_tokens。
    """
    from codingforme import models

    tight = models.budget_breakdown(8_000, tool_schema_tokens=6_000, output_reserve_tokens=4_000)

    assert tight["floored"] is True
    assert tight["budget_tokens"] == models.BUDGET_FLOOR_TOKENS
    assert models.budget_breakdown(128_000, tool_schema_tokens=1_300)["floored"] is False


def test_window_sizes_parse_from_the_three_shapes_people_actually_type():
    """`128k` / `1m` / `128000` 是同一个解析器的三种写法。

    只有一个解析器（`models.parse_window_tokens`），CLI 参数、环境变量、REPL 的
    `/context` 共用它——多一处写法就多一处「同一个数在两个入口下含义不同」。
    """
    from codingforme.models import parse_window_tokens

    assert parse_window_tokens("128000") == 128_000
    assert parse_window_tokens("128k") == 128_000
    assert parse_window_tokens("128K") == 128_000
    assert parse_window_tokens("1m") == 1_000_000
    assert parse_window_tokens("128_000") == 128_000
    assert parse_window_tokens(" 256k ") == 256_000

    # 解析不出来必须抛，不能静默回落到默认档——那会让人以为设生效了。
    for bad in ("", "abc", "128g", "-5", "0"):
        with pytest.raises(ValueError):
            parse_window_tokens(bad)


def test_changing_the_context_window_mid_session_keeps_history_and_rederives_the_budget(tmp_path):
    """`/context` 要能在对话中途改档位，所以 `set_context_window()` 必须可重入。

    两件事一起验：预算跟着窗口重新派生（不是只改了个显示用的数），以及 history
    和 memory 一个字都没动——重建 ContextManager 时顺手清掉对话，是这类「运行中
    换配置」最容易出的 bug。
    """
    agent = build_agent(tmp_path, [tool_call("read_file", path="README.md"), final_answer("done")])
    agent.ask("read the readme")
    history_before = json.dumps(agent.session["history"], sort_keys=True)
    memory_before = json.dumps(agent.session["memory"], sort_keys=True)

    window, source, budget = agent.set_context_window(32_000)

    assert window == 32_000
    assert source == "explicit"
    assert agent.context_budget == budget
    assert agent.context_manager.total_budget == budget
    # 32k 档扣完四笔仍然必须比 1M 档（夹到 128k）小得多。
    assert budget < models_module.context_budget_tokens(1_000_000)
    assert json.dumps(agent.session["history"], sort_keys=True) == history_before
    assert json.dumps(agent.session["memory"], sort_keys=True) == memory_before


def test_context_command_reports_the_tier_its_source_and_what_the_budget_is_made_of(tmp_path):
    """不带参数的 `/context` 要能独立读懂：档位、来源、预算、以及可填的档位清单。

    「来源」这一项不能省：`explicit` 和 `default` 的区别就是「这是我设的」还是
    「自动猜的没猜准」，少了它，一个 8k 的显示无从判断该不该动手改。
    """
    from codingforme.cli import _context_status

    agent = build_agent(tmp_path, [])
    agent.set_context_window(128_000)
    text = _context_status(agent)

    assert "128k" in text
    assert "explicit" in text
    assert str(f"{agent.context_budget:,}") in text
    assert "1M" in text          # 档位清单列出来，不用去翻源码
    assert "/context 128k" in text
    # 算式要逐项列出来：只给一个总数的话，不知道该去调哪一笔。
    for label in ("output", "schema", "frames", "margin"):
        assert label in text, f"/context 少了 {label} 这一项扣除"
    assert str(f"{agent.context_budget_breakdown['tool_schema_tokens']:,}") in text


def test_context_command_says_when_it_floored_your_number_to_a_tier(tmp_path):
    """填 100k 会落到 64k 档。不说出来的话，用户会以为设成了 100k。"""
    from codingforme.cli import _apply_context_window

    agent = build_agent(tmp_path, [])
    text = _apply_context_window(agent, "100k")

    assert agent.context_window == 64_000
    assert "64k" in text
    assert "floored" in text


def test_a_bad_context_command_argument_changes_nothing(tmp_path):
    """解析失败只回一句错误，档位和预算原封不动——静默回落是这里最坏的行为。"""
    from codingforme.cli import _apply_context_window

    agent = build_agent(tmp_path, [])
    agent.set_context_window(128_000)
    before = (agent.context_window, agent.context_budget, agent.context_manager.total_budget)

    text = _apply_context_window(agent, "banana")

    assert "not a context window size" in text
    assert (agent.context_window, agent.context_budget, agent.context_manager.total_budget) == before


def test_an_empty_old_text_appends_instead_of_erroring(tmp_path):
    """`patch_file(old_text="")` 是追加，不是报错。

    加这条用法是实测逼出来的：一次 k=3 的 live 跑批里，模型想往 notes.md 末尾加一行，
    3 次有 2 次自发发出这个形状，每次都换回一句 `old_text must not be empty`——白烧
    一步加一个约 17 秒的往返，`arguments_valid` 只剩 1/3。
    """
    agent = build_agent(tmp_path, [])
    (tmp_path / "notes.md").write_text("# notes\n", encoding="utf-8")

    result = agent.run_tool(
        "patch_file", {"path": "notes.md", "old_text": "", "new_text": "audit token: abc\n"}
    )

    assert not result.startswith("error:")
    assert "appended to notes.md" in result
    assert (tmp_path / "notes.md").read_text(encoding="utf-8") == "# notes\naudit token: abc\n"


def test_appending_to_a_file_without_a_trailing_newline_does_not_glue_the_lines_together(tmp_path):
    """补一个换行、只补一个：接什么就是什么，结果能从参数直接推出来。"""
    agent = build_agent(tmp_path, [])
    (tmp_path / "notes.md").write_text("# notes", encoding="utf-8")

    agent.run_tool("patch_file", {"path": "notes.md", "old_text": "", "new_text": "tail"})

    assert (tmp_path / "notes.md").read_text(encoding="utf-8") == "# notes\ntail"


def test_appending_still_requires_new_text(tmp_path):
    """空 old_text 换掉的是「精确命中一次」，不是参数校验本身。"""
    agent = build_agent(tmp_path, [])
    (tmp_path / "notes.md").write_text("# notes\n", encoding="utf-8")

    result = agent.run_tool("patch_file", {"path": "notes.md", "old_text": ""})

    assert result.startswith("error:")
    assert "missing new_text" in result


def test_a_non_empty_old_text_still_has_to_match_exactly_once(tmp_path):
    """反向用例：追加模式不能把确定性替换那条约束一起放宽。"""
    agent = build_agent(tmp_path, [])
    (tmp_path / "notes.md").write_text("dup\ndup\n", encoding="utf-8")

    result = agent.run_tool(
        "patch_file", {"path": "notes.md", "old_text": "dup", "new_text": "one"}
    )

    assert result.startswith("error:")
    assert (tmp_path / "notes.md").read_text(encoding="utf-8") == "dup\ndup\n"


def test_the_schema_tells_the_model_that_an_empty_old_text_appends(tmp_path):
    """模型只能从 schema 描述知道这条用法——它不在任何示例里。"""
    from codingforme import tools as toolkit

    specs = toolkit.to_openai_function_specs({"patch_file": toolkit.BASE_TOOL_SPECS["patch_file"]})
    description = specs[0]["function"]["parameters"]["properties"]["old_text"]["description"]

    assert "append" in description


def test_the_prompt_metadata_reports_only_token_units(tmp_path):
    """工件里不能再出现按字符计的段落尺寸。

    单位统一到 token 之后，`prefix_chars` / `history_chars` 这类字段没有任何代码再
    读，但它们仍然每轮落进 trace——留着的后果不是浪费几个字节，是读工件的人会把
    它们当成预算的口径，而预算、各段下限、工具输出上限、截断原语全都按 token 算。
    段落尺寸的唯一口径是 `sections[*].rendered_tokens` / `budget_tokens`。
    """
    agent = build_agent(tmp_path, [])
    _, _, metadata = agent._build_context("hello")

    char_fields = sorted(key for key in metadata if key.endswith("_chars"))

    assert char_fields == []
    assert metadata["sections"]["prefix"]["rendered_tokens"] > 0


def test_compact_is_discoverable_from_help_and_the_welcome_hint():
    """新加的斜杠命令必须在 `/help` 和欢迎语里都出现。

    只实现不挂出去等于没有：REPL 里没人会去猜一个没列出来的命令，而这个命令
    恰恰是「用户知道这一段结束了」时才该敲的——它的全部价值依赖用户知道它存在。
    """
    from codingforme.cli import HELP_DETAILS, WELCOME_HINT

    assert "/compact" in HELP_DETAILS
    assert "/compact" in WELCOME_HINT
