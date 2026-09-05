import copy
import json
import re
from pathlib import Path

from codingforme import FakeModelClient, CodingForMe, SessionStore, WorkspaceContext
from codingforme import context_manager
from codingforme import models
from codingforme.context_manager import ContextManager
from codingforme.models import count_tokens


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs, **kwargs):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".codingforme" / "sessions")
    approval_policy = kwargs.pop("approval_policy", "auto")
    return CodingForMe(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy=approval_policy,
        **kwargs,
    )


def test_context_manager_assembles_sections_in_expected_order(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.memory.append_note("deploy key is red", tags=("deploy",), created_at="2026-04-07T10:00:00+00:00")
    agent.record({"role": "user", "content": "old request", "created_at": "2026-04-07T09:59:00+00:00"})
    agent.record({"role": "assistant", "content": "old answer", "created_at": "2026-04-07T10:00:30+00:00"})

    prompt, metadata = ContextManager(agent).build("Where is the deploy key?")

    assert prompt.index("You are coding-for-me") < prompt.index("Memory:")
    assert prompt.index("Memory:") < prompt.index("Relevant memory:")
    assert prompt.index("Relevant memory:") < prompt.index("Transcript:")
    assert prompt.index("Transcript:") < prompt.index("Current user request:")
    assert prompt.rstrip().endswith("Current user request:\nWhere is the deploy key?")
    assert metadata["section_order"] == ["prefix", "memory", "relevant_memory", "history", "current_request"]


def test_context_manager_reduces_relevant_memory_before_history_and_preserves_newer_context(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.prefix = "PREFIX " + ("A" * 600)
    agent.memory.render_memory_text = lambda: "MEMORY " + ("B" * 600)
    agent.memory.append_note("keep episodic note one " + ("C" * 220), tags=("keep",), created_at="2026-04-07T10:00:00+00:00")
    agent.memory.append_note("keep episodic note two " + ("D" * 220), tags=("keep",), created_at="2026-04-07T10:01:00+00:00")
    agent.memory.append_note("keep episodic note three " + ("E" * 220), tags=("keep",), created_at="2026-04-07T10:02:00+00:00")
    agent.record({"role": "user", "content": "OLD-CONTEXT " + ("D" * 260), "created_at": "2026-04-07T09:59:00+00:00"})
    for minute in range(1, 8):
        role = "assistant" if minute % 2 == 1 else "user"
        content = "RECENT-CONTEXT " + ("E" * 260) if minute == 7 else f"recent-{minute} " + ("E" * 180)
        agent.record({"role": role, "content": content, "created_at": f"2026-04-07T10:0{minute}:00+00:00"})

    manager = ContextManager(
        agent,
        total_budget=175,
        section_budgets={
            # prefix 不在这里：它没有额度这个概念，传进去也会被 __init__ 丢掉。
            "memory": 120,
            "relevant_memory": 120,
            "history": 400,
        },
    )

    prompt, metadata = manager.build("keep this request verbatim")

    for section in ("memory", "relevant_memory", "history"):
        assert metadata["sections"][section]["rendered_tokens"] <= metadata["sections"][section]["budget_tokens"]
    # prefix 反过来：它没有额度，而且一个 token 都没被裁掉。
    assert metadata["sections"]["prefix"]["budget_tokens"] is None
    assert (
        metadata["sections"]["prefix"]["rendered_tokens"]
        == metadata["sections"]["prefix"]["raw_tokens"]
    )

    reduction_sections = [entry["section"] for entry in metadata["budget_reductions"]]
    assert reduction_sections[0] == "relevant_memory"
    assert reduction_sections
    assert "RECENT-CONTEXT" in prompt
    assert "OLD-CONTEXT" not in prompt
    assert "keep this request verbatim" in prompt


def test_context_manager_renders_top_three_episodic_notes_per_note_under_budget(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.memory.append_note("alpha episodic note " + ("A" * 120), tags=("recall",), created_at="2026-04-07T10:00:00+00:00")
    agent.memory.append_note("beta episodic recall note " + ("B" * 120), created_at="2026-04-07T10:01:00+00:00")
    agent.memory.append_note("gamma episodic note " + ("C" * 120), tags=("recall",), created_at="2026-04-07T10:02:00+00:00")
    agent.memory.append_note("older unmatched note", created_at="2026-04-07T09:59:00+00:00")
    agent.memory.append_note("Unrelated note", created_at="2026-04-07T11:00:00+00:00")

    prompt, metadata = ContextManager(
        agent,
        total_budget=65,
        section_budgets={
            "prefix": 60,
            "memory": 60,
            "relevant_memory": 80,
            "history": 60,
        },
    ).build("recall")

    assert metadata["relevant_memory"]["selected_count"] == 3
    assert metadata["relevant_memory"]["limit"] == 3
    assert metadata["relevant_memory"]["selected_notes"] == [
        "gamma episodic note " + ("C" * 120),
        "alpha episodic note " + ("A" * 120),
        "beta episodic recall note " + ("B" * 120),
    ]
    assert len(metadata["relevant_memory"]["rendered_notes"]) == 3
    assert metadata["relevant_memory"]["rendered_count"] == 3
    assert metadata["relevant_memory"]["rendered_notes"][0].startswith("gamma episodi")
    assert metadata["relevant_memory"]["rendered_notes"][1].startswith("alpha episodi")
    assert metadata["relevant_memory"]["rendered_notes"][2].startswith("beta episodi")
    relevant_section = prompt.split("Relevant memory:\n", 1)[1].split("\n\nTranscript:", 1)[0]
    assert len([line for line in relevant_section.splitlines() if line.startswith("- ")]) == 3
    assert "alpha episodi" in relevant_section
    assert "beta episodic" in relevant_section
    assert "gamma episodi" in relevant_section
    assert "older unmatched note" not in relevant_section


def test_context_manager_preserves_current_request_when_over_budget(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.prefix = "PREFIX " + ("A" * 600)
    agent.memory.render_memory_text = lambda: "MEMORY " + ("B" * 600)
    agent.memory.retrieval_view = lambda query, limit=3: "Relevant memory:\n" + "\n".join(f"- {i} " + ("C" * 220) for i in range(5))
    agent.history_text = lambda: "Transcript:\n" + "\n".join(f"[user] {i} " + ("D" * 220) for i in range(5))

    request = "please preserve this request exactly"
    prompt, metadata = ContextManager(
        agent,
        total_budget=65,
        section_budgets={
            "prefix": 80,
            "memory": 80,
            "relevant_memory": 80,
            "history": 80,
        },
    ).build(request)

    assert prompt.split("Current user request:\n", 1)[1] == request
    assert metadata["current_request"]["text"] == request
    from codingforme.models import count_tokens

    assert metadata["current_request"]["rendered_tokens"] == count_tokens(request)


def test_context_manager_head_tail_clips_current_request_when_it_alone_overflows_budget(tmp_path):
    agent = build_agent(tmp_path, [])

    # 用互不相同的词而不是一长串重复字符：预算按 token 判之后，"X"*2000 只值
    # 一百多个 token（重复串的压缩率极高），根本撑不爆预算，用例会静默失去意义。
    filler = " ".join(f"filler{index}" for index in range(800))
    huge_request = "START-MARKER " + filler + " END-MARKER"
    # 预算要大到能装下整个 prefix，否则这个用例测不到它想测的东西。prefix 现在不裁，
    # 光它一段就有几百个 token；预算给 200 时兜底分支把请求砍到零也退不出超预算，
    # 于是断言失败的原因变成"prefix 太大"，而不是"请求没被首尾保留地裁掉"。
    # 那条中文提示语本身还要约 50 个 token（中文一个字往往就是一个 token）。
    prompt, metadata = ContextManager(
        agent,
        total_budget=1600,
        section_budgets={
            "memory": 60,
            "relevant_memory": 60,
            "history": 60,
        },
    ).build(huge_request)

    assert metadata["prompt_tokens"] <= 1600
    assert metadata["current_request"]["truncated"] is True
    assert metadata["current_request"]["dropped_tokens"] > 0
    assert metadata["prompt_over_budget"] is False
    rendered_request = prompt.split("Current user request:\n", 1)[1]
    assert rendered_request.startswith("START-MARKER")
    assert "END-MARKER" in rendered_request
    assert "[omitted middle]" in rendered_request
    assert "请提示用户缩短请求或分步描述" in rendered_request


def test_context_manager_leaves_current_request_untruncated_when_it_fits(tmp_path):
    agent = build_agent(tmp_path, [])

    prompt, metadata = ContextManager(agent).build("a short request that fits easily")

    assert metadata["current_request"]["truncated"] is False
    assert metadata["current_request"]["dropped_tokens"] == 0
    assert prompt.rstrip().endswith("Current user request:\na short request that fits easily")


def test_context_manager_collapses_older_duplicate_reads_into_one_summary_line(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\nbeta\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])
    agent.memory.set_file_summary("sample.txt", "alpha | beta")
    agent.memory.remember_file("sample.txt")

    for created_at in ("2026-04-07T09:00:00+00:00", "2026-04-07T09:01:00+00:00"):
        agent.record(
            {
                "role": "tool",
                "name": "read_file",
                "args": {"path": "sample.txt", "start": 1, "end": 2},
                "content": "# sample.txt\nalpha\nbeta\n",
                "created_at": created_at,
            }
        )

    # 18 条而不是 12 条：最近窗口的条目上限是 2 * 6 个工具轮 = 12 条（一个工具轮
    # 会留下模型说明 + 工具结果两条记录），而这条上限本身**按 2 * RECENT_WINDOW_BLOCK
    # = 6 条成块推进**，所以实际窗口在 12~17 条之间浮动。要把上面那两次读稳定挤出
    # 窗口，就得垫够 18 条。
    for minute in range(2, 20):
        role = "user" if minute % 2 == 0 else "assistant"
        agent.record(
            {
                "role": role,
                "content": f"recent-{minute}",
                "created_at": f"2026-04-07T09:{minute:02d}:00+00:00",
            }
        )

    prompt, metadata = ContextManager(agent).build("check the file")
    transcript = prompt.split("\n\nTranscript:\n", 1)[1].split("\n\nCurrent user request:", 1)[0]

    assert transcript.count("[tool:read_file]") == 0
    assert "sample.txt -> alpha | beta" in transcript
    assert metadata["history"]["older_entries_count"] == 1
    assert metadata["history"]["collapsed_duplicate_reads"] == 1
    assert metadata["history"]["reused_file_summary_count"] == 1


def test_recent_window_counts_tool_turns_not_history_entries(tmp_path):
    """最近窗口要覆盖 6 个工具轮，不是 6 条 history 记录。

    这条是回归网：模型的工具轮说明开始写进 history 之后，一个工具轮从
    1 条记录变成 2 条。窗口若还按记录数算，就会悄悄从 6 个工具轮塌成 3 个，
    而这正是「模型忘了自己干过什么、于是反复回读」的成因。
    """
    agent = build_agent(tmp_path, [])
    for i in range(8):
        agent.record({"role": "assistant", "content": f"NARRATION-{i} why I am doing this", "created_at": ""})
        agent.record(
            {
                "role": "tool",
                "name": "read_file",
                "args": {"path": f"file_{i}.py"},
                "content": f"# file_{i}.py\nbody of file {i}",
                "created_at": "",
            }
        )

    prompt, _ = ContextManager(agent).build("keep going")
    transcript = prompt.split("\n\nTranscript:\n", 1)[1].split("\n\nCurrent user request:", 1)[0]

    # 最后 6 个工具轮是 file_2..file_7，它们的说明必须都在。
    for i in range(2, 8):
        assert f"NARRATION-{i}" in transcript, f"第 {i} 轮的说明掉出了最近窗口"


def test_the_newest_read_of_a_path_wins_over_the_stale_copy(tmp_path):
    """同一个文件读过多次时，留在 prompt 里的必须是最后那次的内容。

    旧实现保留最早那次，等于把过时内容当现状喂给模型：实测有一次运行里
    server.py 在 patch 前后各读过一次，留下来的会是改动前的 PORT = 8080。
    """
    agent = build_agent(tmp_path, [])
    for content in ("PORT = 8080", "PORT = 9090"):
        agent.record(
            {
                "role": "tool",
                "name": "read_file",
                "args": {"path": "server.py"},
                "content": f"# server.py\n{content}",
                "created_at": "",
            }
        )

    prompt, metadata = ContextManager(agent).build("what is the port")
    transcript = prompt.split("\n\nTranscript:\n", 1)[1].split("\n\nCurrent user request:", 1)[0]

    assert "PORT = 9090" in transcript
    assert "PORT = 8080" not in transcript, "过期的那份读取不该还留在 prompt 里"
    assert metadata["history"]["collapsed_duplicate_reads"] == 1


def test_two_ranges_of_the_same_file_are_both_kept(tmp_path):
    """同一个文件的不同行区间是互补内容，不是彼此的陈旧版本。

    去重键只按路径时，一轮里读 1–50 行和 51–100 行，前一段会被整条丢掉——
    模型下一轮看不到自己刚读过的前半段，而它以为看得到。单调用时几乎碰不到，
    一轮多调用把它放大成常见路径。
    """
    agent = build_agent(tmp_path, [])
    for start, end, marker in ((1, 50, "FIRST_HALF"), (51, 100, "SECOND_HALF")):
        agent.record(
            {
                "role": "tool",
                "name": "read_file",
                "args": {"path": "server.py", "start": start, "end": end},
                "content": f"# server.py\n{marker}",
                "created_at": "",
            }
        )

    prompt, metadata = ContextManager(agent).build("summarise server.py")
    transcript = prompt.split("\n\nTranscript:\n", 1)[1].split("\n\nCurrent user request:", 1)[0]

    assert "FIRST_HALF" in transcript, "先读的那一段被去重逻辑吞掉了"
    assert "SECOND_HALF" in transcript
    assert metadata["history"]["collapsed_duplicate_reads"] == 0


def test_an_unexecuted_read_never_shadows_a_real_one(tmp_path):
    """没执行的调用不能顶掉同一区间那次真读到的内容。

    未执行记录的 content 是一句「未执行」的说明。它如果参与去重并且排在后面，
    真读到的文件内容就会被丢掉——数据直接没了，而报告上看不出来。
    """
    agent = build_agent(tmp_path, [])
    agent.record(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "server.py"},
            "content": "# server.py\nPORT = 9090",
            "created_at": "",
        }
    )
    agent.record(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "server.py"},
            "content": CodingForMe.UNEXECUTED_CALL_RESULT,
            "created_at": "",
            "executed": False,
        }
    )

    prompt, _metadata = ContextManager(agent).build("what is the port")
    transcript = prompt.split("\n\nTranscript:\n", 1)[1].split("\n\nCurrent user request:", 1)[0]

    assert "PORT = 9090" in transcript, "真读到的内容被未执行记录顶掉了"
    assert "not executed" in transcript, "未执行这件事也要留在 prompt 里，模型才知道要重发"


def test_context_manager_summarizes_older_tool_output_into_one_line(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.record(
        {
            "role": "tool",
            "name": "run_shell",
            "args": {"command": "pytest -q"},
            "content": "FAIL test_one\nFAIL test_two\nFAIL test_three\nFAIL test_four\n",
            "created_at": "2026-04-07T09:00:00+00:00",
        }
    )

    # 同上：窗口上限按 6 条成块推进，在 12~17 之间浮动，垫满 18 条才能把这次
    # run_shell 稳定挤出最近窗口。
    for minute in range(1, 19):
        role = "user" if minute % 2 == 1 else "assistant"
        agent.record(
            {
                "role": role,
                "content": f"recent-{minute}",
                "created_at": f"2026-04-07T09:{minute:02d}:00+00:00",
            }
        )

    prompt, metadata = ContextManager(agent).build("check failures")
    transcript = prompt.split("\n\nTranscript:\n", 1)[1].split("\n\nCurrent user request:", 1)[0]

    assert 'pytest -q -> FAIL test_one | FAIL test_two | FAIL test_three' in transcript
    assert "FAIL test_four" not in transcript
    assert metadata["history"]["summarized_tool_count"] == 1
    assert metadata["history"]["reused_file_summary_count"] == 0


def test_context_manager_relevant_memory_can_mix_durable_notes(tmp_path):
    memory_root = tmp_path / ".codingforme" / "memory"
    topics_dir = memory_root / "topics"
    topics_dir.mkdir(parents=True)
    (memory_root / "MEMORY.md").write_text(
        "# Durable Memory Index\n\n"
        "- [project-conventions](topics/project-conventions.md): Project Conventions\n"
        "  - summary: Stable repository conventions.\n"
        "  - tags: convention\n",
        encoding="utf-8",
    )
    (topics_dir / "project-conventions.md").write_text(
        "# Project Conventions\n\n"
        "- topic: project-conventions\n"
        "- summary: Stable repository conventions.\n"
        "- tags: convention\n"
        "- updated_at: 2026-04-12T08:14:49+00:00\n\n"
        "## Notes\n"
        "- Use constrained tools instead of guessing.\n",
        encoding="utf-8",
    )

    agent = build_agent(tmp_path, [])

    prompt, metadata = ContextManager(agent).build("What conventions should I follow?")
    relevant_section = prompt.split("Relevant memory:\n", 1)[1].split("\n\nTranscript:", 1)[0]

    assert "Use constrained tools instead of guessing." in relevant_section
    assert any("Use constrained tools instead of guessing." in item for item in metadata["relevant_memory"]["selected_notes"])
    assert metadata["relevant_memory"]["selected_durable_count"] == 1
    assert metadata["relevant_memory"]["selected_sources"] == ["project-conventions"]
    assert metadata["relevant_memory"]["selected_kinds"] == ["durable"]


def _tool_record(agent, name, args, content, created_at):
    agent.record({
        "role": "tool", "name": name, "args": args, "content": content,
        "executed": True, "created_at": created_at,
    })


def test_dropped_history_leaves_a_digest_instead_of_vanishing(tmp_path):
    """预算放不下的历史要留下摘要，不能静默消失。

    原来的做法是条目放不下就跳过——模型完全看不出自己前面做过什么，于是重读
    已经读过的文件、重发已经发过的补丁。这条锁住 Anthropic《Effective context
    engineering》里 compaction 的要点：接近上限时要**总结再丢**，不是直接截断。
    """
    agent = build_agent(tmp_path, [])
    for minute in range(1, 7):
        _tool_record(
            agent, "read_file", {"path": f"old_{minute}.py"},
            "OLD-FILE-BODY " + ("Z" * 400), f"2026-04-07T09:0{minute}:00+00:00",
        )
    _tool_record(
        agent, "patch_file", {"path": "target.py"},
        "patched", "2026-04-07T09:59:00+00:00",
    )
    agent.record({"role": "assistant", "content": "RECENT " + ("E" * 200),
                  "created_at": "2026-04-07T10:00:00+00:00"})

    manager = ContextManager(
        agent,
        total_budget=225,
        section_budgets={"prefix": 120, "memory": 120, "relevant_memory": 120, "history": 300},
    )
    messages, _prompt, metadata = manager.build_all("finish the task")

    history = metadata["sections"]["history"]
    assert history["omitted_entry_count"] > 0, "这个预算下本就该丢掉一些条目"

    # 摘要必须真的发出去，而不是只出现在用于度量的压平文本里。
    state = [m["content"] for m in messages if m["role"] == "user"]
    digest = next(text for text in state if "Omitted context:" in text)
    assert "read_file x" in digest
    # 具体丢的是哪几条随预算浮动，锁死某个文件名会让这条测试变脆；
    # 要断言的是「丢了什么被记下来了」，不是「恰好丢了 old_1」。
    assert "files touched: old_" in digest
    assert "Omitted context: %d earlier" % history["omitted_entry_count"] in digest
    assert metadata["message_layout"]["omitted_digest_present"] is True


def test_no_digest_appears_when_the_budget_fits_everything(tmp_path):
    """预算够用时不该凭空多出一条摘要——它每轮都变，会压低前缀缓存命中率。"""
    agent = build_agent(tmp_path, [])
    _tool_record(agent, "read_file", {"path": "a.py"}, "body", "2026-04-07T10:00:00+00:00")

    manager = ContextManager(agent, total_budget=10000)
    messages, _prompt, metadata = manager.build_all("go")

    assert metadata["sections"]["history"]["omitted_entry_count"] == 0
    assert metadata["message_layout"]["omitted_digest_present"] is False
    assert all("Omitted context:" not in str(m["content"]) for m in messages)


def _agent_with_bloated_history(tmp_path, turns=40, chars=900):
    """造一个必然超预算的上下文：足够多、足够长的历史条目。"""
    agent = build_agent(tmp_path, [])
    for index in range(turns):
        agent.record({"role": "user", "content": f"q{index} " + "x" * chars})
        agent.record({"role": "assistant", "content": f"a{index} " + "y" * chars})
    return agent


def test_the_cached_prefix_is_never_sacrificed_to_the_char_budget(tmp_path):
    """prefix 超预算时也不许被裁——它就是前缀缓存认的那段公共前缀。

    踩过坑：prefix 曾经排在 reduction_order 最后当"最后手段"，而实测 632 轮里
    超预算的 67 轮**每一轮都用到了这个最后手段**。裁掉它省下几百字符，代价是
    那一轮缓存必然落空，而缓存命中率是花了一整轮 A/B 才从 32.7% 提到 77.6% 的。
    """
    agent = _agent_with_bloated_history(tmp_path)
    manager = ContextManager(agent, total_budget=1500)

    _, metadata = manager.build("What changed?")

    assert "prefix" not in manager.reduction_order
    assert metadata["protected_sections"] == ["prefix"]
    reduced_sections = {entry["section"] for entry in metadata["budget_reductions"]}
    assert reduced_sections, "这个用例必须真的触发裁剪，否则它什么都没验证"
    assert "prefix" not in reduced_sections
    # prefix 现在**没有额度**，原文原样进去，一个 token 都没让出去。
    # 从前这里断言的是"拿到了它完整的 section 额度"，而那个额度本身就在裁它：
    # 实测本仓库 prefix 原始 3696 token、额度 1450，每轮白丢 61%。
    assert metadata["sections"]["prefix"]["budget_tokens"] is None
    assert (
        metadata["sections"]["prefix"]["rendered_tokens"]
        == metadata["sections"]["prefix"]["raw_tokens"]
    )
    assert "prefix" not in manager.section_budgets


def test_passing_prefix_in_a_custom_reduction_order_does_not_re_enable_cutting_it(tmp_path):
    """自定义顺序也挡得住:"prefix 不被裁"是不变量，不是默认值。"""
    agent = _agent_with_bloated_history(tmp_path)
    manager = ContextManager(
        agent, total_budget=1500, reduction_order=("prefix", "history", "memory")
    )

    _, metadata = manager.build("What changed?")

    assert manager.reduction_order == ("history", "memory")
    assert "prefix" not in {entry["section"] for entry in metadata["budget_reductions"]}


def test_running_out_of_room_is_reported_instead_of_silently_overflowing(tmp_path):
    """可裁的都裁到底仍放不下时，要有一个字段说出来。

    它和"刚好装下"在 prompt_chars / budget_reductions 上看不出区别，而含义相反：
    前者说明预算对当前负载偏小，该调预算，不是该继续裁。
    """
    agent = _agent_with_bloated_history(tmp_path)
    manager = ContextManager(agent, total_budget=900)

    _, metadata = manager.build("What changed?")

    assert metadata["prompt_over_budget"] is True
    assert metadata["budget_floor_exhausted"] is True


def test_protecting_the_prefix_does_not_start_clipping_normal_user_requests(tmp_path):
    """保护 prefix 不能把代价转嫁到用户请求上。

    首尾保留裁剪只服务于"请求自己就放不下"这一种情况。prefix 不再参与裁剪之后
    可减空间少了一截，如果进入条件只看"prompt 超没超预算"，一个正常长度的请求
    也会被砍——那是把一个 bug 换成了另一个更难发现的 bug。
    """
    agent = _agent_with_bloated_history(tmp_path)
    manager = ContextManager(agent, total_budget=900)

    request = "Please summarise what changed in the last few turns."
    prompt, metadata = manager.build(request)

    assert metadata["prompt_over_budget"] is True, "这个用例要求真的超预算"
    assert metadata["current_request"]["truncated"] is False
    assert request in prompt


def test_the_budget_is_denominated_in_tokens_not_characters(tmp_path):
    """预算的单位是 token，而且工件里**只有** token 这一种单位。

    从前它是 12000 个**字符**，而后端按 token 计费和截断。实测 318 轮真实请求，
    字符数与 token 数的比值中位 2.06、最小 0.83、最大 2.51——相差 3 倍，因为中文
    一个字往往就是一个 token，英文和代码要 3~4 个字符才一个。

    同时摆两种单位比只用错的那种更糟：读的人无从知道某个数是哪一种。所以这里
    连带断言 `*_chars` 字段已经从 metadata 里彻底消失。
    """
    agent = build_agent(tmp_path, [])
    manager = ContextManager(agent, total_budget=4000)

    _, metadata = manager.build("hello")

    assert metadata["prompt_budget_tokens"] == 4000
    assert metadata["prompt_tokens"] > 0
    leftovers = sorted(key for key in metadata if key.endswith("_chars"))
    assert leftovers == [], f"metadata 里还留着按字符计的字段：{leftovers}"
    section_leftovers = sorted(
        f"{section}.{key}"
        for section, fields in metadata["sections"].items()
        for key in fields
        if key.endswith("_chars")
    )
    assert section_leftovers == [], f"section 里还留着按字符计的字段：{section_leftovers}"


def test_chinese_and_ascii_of_the_same_length_do_not_cost_the_same_budget(tmp_path):
    """同样长的中文和英文占用的预算不同——这正是换单位的理由。"""
    from codingforme.models import count_tokens

    chinese = "上下文预算必须按词元计算而不是按字符计算" * 4
    ascii_text = "context budget must be counted in tokens not chars" * 3
    # 80 个中文字符 vs 150 个英文字符：更短的那串反而更贵（52 vs 27 个 token）。
    assert len(chinese) < len(ascii_text)
    assert count_tokens(chinese) > count_tokens(ascii_text), (
        "更短的中文串应该比更长的英文串更费 token；这条挂了说明计数退化成了字符数"
    )


def test_the_tool_schema_is_charged_once_when_the_budget_is_derived(tmp_path):
    """工具 schema 的账只能记一次：派生预算时扣掉，测量上下文时不再加。

    两处都算会让 total_budget 有一个隐形下限——预算低于 schema 大小时永远不可满足，
    裁剪循环把每个 section 都砍到底也退不出去。
    """
    from codingforme.models import context_budget_tokens

    agent = build_agent(tmp_path, [])
    schema_tokens = agent.tool_schema_tokens()
    assert schema_tokens > 0, "注册表非空时 schema 不可能是 0 个 token"

    from codingforme import models

    budget = context_budget_tokens(
        32_000, tool_schema_tokens=schema_tokens, output_reserve_tokens=1024, expected_messages=5
    )
    framing = models.MESSAGE_FRAMING_BASE_TOKENS + models.MESSAGE_FRAMING_TOKENS_PER_MESSAGE * 5
    deducted = 32_000 - 1024 - schema_tokens - framing
    assert budget == deducted - int(deducted * models.TOKENIZER_MARGIN_RATIO)

    # 预算设得比 schema 还小时，测量值仍然只反映上下文本身，不会凭空多出 schema。
    manager = ContextManager(agent, total_budget=max(1, schema_tokens // 2))
    _, metadata = manager.build("hi")
    assert metadata["tool_schema_tokens"] == schema_tokens
    assert metadata["prompt_tokens"] < schema_tokens


def test_a_prompt_that_fits_is_never_trimmed_by_a_fixed_per_section_quota(tmp_path):
    """预算够用时，每一段都原样进上下文。

    这是阶段二的核心不变量。从前各段有一组绝对额度（prefix 1450 / memory 520 /
    relevant_memory 900 / history 2500），它们和 total_budget 完全脱钩：预算涨到
    十万量级之后，裁剪循环永远触发不了，但各段照样每轮被砍——实测本仓库 prefix
    原始 3696 token 被裁到 1450，白丢 61%，而此时预算还空着 11 万。
    """
    agent = build_agent(tmp_path, [])
    agent.prefix = "PREFIX " + " ".join(f"word{index}" for index in range(400))
    agent.memory.render_memory_text = lambda: "MEMORY " + " ".join(f"mem{index}" for index in range(300))
    agent.memory.append_note("note " + " ".join(f"n{index}" for index in range(200)), tags=("keep",))
    for index in range(10):
        agent.record({"role": "user", "content": f"turn{index} " + " ".join(f"t{n}" for n in range(80))})

    manager = ContextManager(agent, total_budget=100_000)
    _, metadata = manager.build("keep this request verbatim")

    assert metadata["prompt_over_budget"] is False
    assert metadata["budget_reductions"] == []
    for section in ("prefix", "memory", "relevant_memory", "history"):
        sizes = metadata["sections"][section]
        assert sizes["rendered_tokens"] == sizes["raw_tokens"], (
            f"{section} 在预算充足时被裁了：{sizes}"
        )


def test_a_budget_smaller_than_the_prefix_still_assembles_a_valid_prompt(tmp_path):
    """预算比 prefix 还小时不许炸，要照常发出去并把超预算如实记下来。

    这是取消各段额度之后最可能出问题的地方：从前四段额度合计 5370 是个绝对值，
    在 8k 兜底档（total_budget 4335）下比总预算还大 24%，每轮都在跑裁剪循环；
    现在 prefix 不裁了，一个小窗口下它自己就能撑爆预算。此时正确的行为是
    **超出去并记账**，不是抛异常，也不是把 prefix 砍掉。
    """
    agent = build_agent(tmp_path, [])
    agent.prefix = "PREFIX " + " ".join(f"word{index}" for index in range(600))
    for index in range(10):
        agent.record({"role": "user", "content": f"turn{index} " + " ".join(f"t{n}" for n in range(60))})

    messages, prompt, metadata = ContextManager(agent, total_budget=200).build_all("go")

    assert metadata["prompt_over_budget"] is True
    assert metadata["budget_floor_exhausted"] is True
    # prefix 一个 token 都没让出去，哪怕它自己就比整个预算大。
    prefix_sizes = metadata["sections"]["prefix"]
    assert prefix_sizes["rendered_tokens"] == prefix_sizes["raw_tokens"]
    assert prefix_sizes["budget_tokens"] is None
    # 仍然是一份能发出去的 prompt：messages 非空，最后一条是当前请求。
    assert messages and messages[-1]["role"] == "user"
    assert "go" in messages[-1]["content"]
    assert prompt.strip()


def _history_with_tool_reads(agent, count, content):
    """给 agent 塞 `count` 轮 read_file 历史，每轮结果都是 `content`。"""
    for index in range(count):
        call_id = f"call-{index}"
        agent.record(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": call_id, "name": "read_file", "args": {"path": f"big{index}.py"}}],
            }
        )
        agent.record(
            {
                "role": "tool",
                "name": "read_file",
                "args": {"path": f"big{index}.py"},
                "content": content,
                "call_id": call_id,
                "executed": True,
            }
        )


def test_the_tool_output_floor_matches_the_workspace_constant():
    """派生上限的下限必须和 `workspace.MAX_TOOL_OUTPUT` 同值。

    两处写着同一个数（context_manager 不 import workspace，避免多一条依赖），
    改了一处忘了另一处，小档位的行为会静默漂移。
    """
    from codingforme import context_manager as cm
    from codingforme import workspace

    assert cm.TOOL_OUTPUT_MIN_TOKENS == workspace.MAX_TOOL_OUTPUT


def test_a_bigger_budget_actually_lets_more_of_a_tool_result_through(tmp_path):
    """把预算放大，模型看到的工具结果就该变多。

    这条锁的是阶段二那个病的第二层：`MAX_TOOL_OUTPUT = 1320` 和 history 里的
    `line_limit = 430` 都是与 `total_budget` 脱钩的绝对常量。改动前实测把预算从
    4,335 放大到 118,335（×27），发出去的 prompt 一个 token 都不变（3,490 →
    3,490），10 条 tool 消息逐条相同——预算再大也没用，因为卡住内容的是别的东西。
    """
    from codingforme.context_manager import tool_output_limit

    big = "\n".join(f"def f{index}(): return {index}  # padding padding padding" for index in range(400))

    def rendered_tool_tokens(total_budget):
        workspace_root = tmp_path / f"ws{total_budget}"
        workspace_root.mkdir()
        agent = build_agent(workspace_root, [])
        manager = ContextManager(agent, total_budget=total_budget)
        # 内容先过入口上限，和 run_tool() 存进 history 时的处理一致。
        _history_with_tool_reads(agent, 10, models.clip_tokens(big, tool_output_limit(total_budget)))
        messages, _, metadata = manager.build_all("continue")
        tool_tokens = [count_tokens(m["content"]) for m in messages if m.get("role") == "tool"]
        return max(tool_tokens), metadata["sections"]["history"]["rendered_tokens"]

    small_max, small_history = rendered_tool_tokens(4335)
    large_max, large_history = rendered_tool_tokens(118335)

    assert large_max > small_max * 2, (
        f"预算放大 27 倍，单条工具结果只从 {small_max} 变到 {large_max}——"
        "说明还有一个与预算脱钩的常量卡在中间"
    )
    assert large_history > small_history


def test_a_cleared_tool_result_says_so_instead_of_echoing_its_own_call(tmp_path):
    """掉出最近窗口的工具结果要留一句明确的占位，不能只剩调用签名。

    从前这里渲染成 `[tool:read_file] {"path": "a.py"}`——那是这次调用自己的签名，
    出现在一条 role:"tool" 消息里，最接近的读法是「这次调用什么都没返回」。
    官方 `clear_tool_uses` 用的是显式的 `[cleared to save context]`。
    """
    from codingforme.context_manager import CLEARED_RESULT_MARKER, RECENT_TOOL_TURNS

    agent = build_agent(tmp_path, [])
    # run_shell 走的是另一条摘要分支（保留 stdout 前三行），这里用它之外的工具。
    for index in range(RECENT_TOOL_TURNS + 4):
        call_id = f"call-{index}"
        agent.record(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": call_id, "name": "list_files", "args": {"path": f"dir{index}"}}],
            }
        )
        agent.record(
            {
                "role": "tool",
                "name": "list_files",
                "args": {"path": f"dir{index}"},
                "content": "[F] a.py\n[F] b.py",
                "call_id": call_id,
                "executed": True,
            }
        )

    messages, _, _ = ContextManager(agent).build_all("continue")
    cleared = [m["content"] for m in messages if m.get("role") == "tool" and CLEARED_RESULT_MARKER in m["content"]]

    assert cleared, "掉出窗口的工具结果没有留下任何「内容已清除」的说明"
    # 调用签名仍然在（模型要知道这次调用发生过），只是后面多了那句占位。
    assert "[tool:list_files]" in cleared[0]


def test_a_reduction_clears_at_least_a_tenth_of_the_budget(tmp_path):
    """裁剪一次要腾出预算的 1/10，不能只裁「刚好够」的量。

    只裁刚好够会让下一轮几乎必然再裁一次，每次都作废 history 之后的全部前缀缓存。
    官方 context management API 的 `clear_at_least` 就是为这件事存在的。
    """
    from codingforme.context_manager import CLEAR_AT_LEAST_DIVISOR

    agent = build_agent(tmp_path, [])
    _history_with_tool_reads(agent, 12, " ".join(f"word{index}" for index in range(400)))

    total_budget = 3000
    _, _, metadata = ContextManager(agent, total_budget=total_budget).build_all("continue")
    reductions = metadata["budget_reductions"]

    assert reductions, "这份历史应该撑爆 3000 的预算并触发裁剪"
    first = reductions[0]
    assert first["target_tokens"] >= total_budget // CLEAR_AT_LEAST_DIVISOR
    assert first["target_tokens"] >= first["overflow_tokens"]


def test_an_uncapped_section_reports_no_budget_in_the_artifact(tmp_path):
    """没有额度的段在工件里记 None，不是 total_budget。

    起始额度等于 total_budget 表达的是「这一段不设上限」；照原样记进工件会被读成
    「分到了这么多」，三段加起来还等于总预算的三倍。只有调用方显式给了额度、或者
    裁剪真的把它压下去过，才算真额度。
    """
    agent = build_agent(tmp_path, [])
    agent.record({"role": "user", "content": "hello"})

    _, _, metadata = ContextManager(agent, total_budget=118335).build_all("continue")

    assert metadata["budget_reductions"] == []
    for section in ("memory", "relevant_memory", "history"):
        assert metadata["sections"][section]["budget_tokens"] is None, (
            f"{section} 没有额度却记成了 {metadata['sections'][section]['budget_tokens']}"
        )

    # 调用方显式给的额度照旧记成数字。
    _, _, capped = ContextManager(agent, section_budgets={"history": 400}).build_all("continue")
    assert capped["sections"]["history"]["budget_tokens"] == 400


def _long_history(agent, turns=24, filler=90):
    """造一段只有 user/assistant 文本的历史。

    刻意不放工具结果：`_compressed_history_entries()` 只对 `role == "tool"` 的条目
    做清理/去重/摘要，工具结果那条路径上的压力会被落盘和最近窗口吃掉。纯对话文本
    是这两级都碰不到的唯一成分，也是唯一能把预算真正顶起来的负载形状。
    """
    for index in range(turns):
        agent.record({"role": "user", "content": f"requirement {index}: " + ("spec " * filler)})
        agent.record({"role": "assistant", "content": f"acknowledged {index}: " + ("recap " * filler)})


def test_graded_compression_starts_before_the_prompt_is_over_budget(tmp_path):
    """触发点在 85%，不是 100%——压完还要留出余量。

    从前只有一根线：`prompt_tokens > total_budget`。合成压力探针实测的后果是 prompt
    被死死顶在 100.0% 上（24 轮里第 13 轮起每一轮都是 99.3% 以上），一条稍长的用户
    消息就溢出。
    """
    agent = build_agent(tmp_path, [])
    # 会话摘要排在逐段硬裁之前,开着它这条负载的压力会被它先吃掉——这里要测的是
    # 分级闸门本身,所以把它钉死成关。
    agent.feature_flags = {**agent.feature_flags, "session_summary": False}
    _long_history(agent)

    _, _, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    pressure = metadata["context_pressure"]
    assert pressure["graded"] is True
    assert pressure["triggered"] is True
    assert pressure["trigger_tokens"] < 4823, "触发点必须严格低于硬预算"
    assert pressure["target_tokens"] < pressure["trigger_tokens"], "压完要压到触发点以下，否则下一轮必然再压"
    assert pressure["occupancy_after"] <= 0.75, pressure
    assert metadata["prompt_over_budget"] is False


def test_turning_off_graded_compression_pins_the_prompt_at_the_budget(tmp_path):
    """消融开关必须严格退回「只有 100% 一根线」的老行为。

    没有这条，A/B 可能两边跑的是同一份代码，而工件上看不出来。
    """
    agent = build_agent(tmp_path, [])
    _long_history(agent)
    agent.feature_flags = {**agent.feature_flags, "graded_compression": False, "session_summary": False}

    _, _, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    pressure = metadata["context_pressure"]
    assert pressure["graded"] is False
    assert pressure["trigger_tokens"] == pressure["target_tokens"] == 4823
    # 老行为是裁到刚好不超预算为止，所以占用率贴着 100%。
    assert 0.90 <= pressure["occupancy_after"] <= 1.0, pressure


def test_graded_compression_sends_fewer_tokens_than_the_single_gate(tmp_path):
    """同一份历史、同一个预算，唯一变量是那个开关：分级压缩发出去的 prompt 更小。"""
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    graded = build_agent(tmp_path / "on", [])
    # 会话摘要排在逐段硬裁之前,开着它这条负载的压力会被它先吃掉——这里要测的是
    # 分级闸门本身,所以把它钉死成关。
    graded.feature_flags = {**graded.feature_flags, "session_summary": False}
    _long_history(graded)
    _, on_prompt, _ = ContextManager(graded, total_budget=4823).build_all("next")

    ungraded = build_agent(_subdir(tmp_path, "off"), [])
    _long_history(ungraded)
    ungraded.feature_flags = {**ungraded.feature_flags, "graded_compression": False, "session_summary": False}
    _, off_prompt, _ = ContextManager(ungraded, total_budget=4823).build_all("next")

    assert count_tokens(on_prompt, None) < count_tokens(off_prompt, None)


# --- 会话摘要(阶段二:L5 便宜的那一半)-----------------------------------------


def _dialogue_turns(agent, budget=4823, turns=24, filler=90, prefix="requirement"):
    """一轮一轮地长对话,每轮真的组一次上下文,返回逐轮的 (metadata, 历史第一条正文)。

    **必须一轮一轮地组**,不能造完 24 轮再组一次:这套东西要验的正是「同一条边界能
    撑几轮」,一次性组装的话跨轮状态根本没有机会体现。
    """
    rows = []
    for index in range(turns):
        agent.record({"role": "user", "content": f"{prefix} {index}: " + ("spec " * filler)})
        agent.record({"role": "assistant", "content": f"acknowledged {index}: " + ("recap " * filler)})
        messages, prompt, metadata = ContextManager(agent, total_budget=budget).build_all("next")
        dialogue = [item for item in messages if item["role"] in ("user", "assistant")]
        # messages[0] 是仓库快照那条 user,history 的第一条紧随其后。
        head = dialogue[1]["content"] if len(dialogue) > 1 else ""
        rows.append((metadata, prompt, head))
    return rows


def _head_moves(rows):
    return sum(1 for before, after in zip(rows, rows[1:]) if before[2] != after[2])


def test_the_session_summary_moves_the_history_boundary_in_jumps_not_every_turn(tmp_path):
    """这是阶段二存在的**唯一理由**,所以它是第一条测试。

    阶段一量出来的病:对话单调增长时,预算驱动的丢弃边界每轮都要往前爬一格,history
    的第一条每轮都变,它之后的前缀缓存每轮作废(实测 24 轮里动了 14 次)。迟滞治不了
    这个(见 context_manager 顶部那段注释)。摘要的做法是「偶尔跳一大步」:一次把最早
    的一批条目整体换成概述,并把剩下的压到预算的 45% 以下,于是下一次跳之前边界一动
    不动。
    """
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    with_summary = _dialogue_turns(build_agent(tmp_path / "on", []))

    without = build_agent(_subdir(tmp_path, "off"), [])
    without.feature_flags = {**without.feature_flags, "session_summary": False}
    without_summary = _dialogue_turns(without)

    assert _head_moves(with_summary) < _head_moves(without_summary)
    # 而且不是「少动一点」,是量级上的差别:关掉之后压力一到就每轮都动。
    assert _head_moves(with_summary) <= 3
    assert _head_moves(without_summary) >= 6


def test_the_session_summary_keeps_the_oldest_user_request_verbatim(tmp_path):
    """被换掉的那一段里,用户说过的话必须原样留在摘要里。

    阶段零那条 24 轮探针最后丢了 39 条历史,上下文里只剩一行「dropped 39 entries」,
    而用户在第 1 轮定下的规格已经不在里面、模型却仍被要求遵守它。这条断言就是冲着
    那个缺陷来的:第 0 条要求的原文必须仍然出现在 prompt 里,尽管它对应的那条历史
    条目已经不在 messages 中了。
    """
    rows = _dialogue_turns(build_agent(tmp_path, []))
    metadata, prompt, _ = rows[-1]

    summary = metadata["context_pressure"]["session_summary"]
    assert summary["covered_entries"] > 0, "这条负载必须真的触发过摘要,否则下面断言什么都没查"
    assert "requirement 0:" in prompt
    assert metadata["message_layout"]["session_summary_present"] is True


def test_the_summary_never_covers_the_recent_window(tmp_path):
    """覆盖点不许吃进最近那几条。

    把刚发生的一轮换成一句概述,等于让模型忘掉自己上一步做了什么——而「模型看得见
    自己上一轮调了什么」正是走到终点率 76% -> 97% 的那个机制。
    """
    agent = build_agent(tmp_path, [])
    rows = _dialogue_turns(agent)
    covered = rows[-1][0]["context_pressure"]["session_summary"]["covered_entries"]
    history_length = len(agent.session["history"])

    assert covered > 0
    assert history_length - covered >= context_manager.SESSION_SUMMARY_MIN_KEPT_ENTRIES


def test_the_summary_rides_along_with_the_session_file(tmp_path):
    """摘要状态必须跟着 session 一起落盘、一起 resume。

    覆盖点是 history 的下标:重启之后如果只恢复了 history 而丢了覆盖点,同一段内容
    会既留在摘要里、又整段回到历史里(白发一遍),或者反过来凭空消失。
    """
    agent = build_agent(tmp_path, [])
    _dialogue_turns(agent)
    before = dict(agent.session["context_summary"])
    assert before["covered"] > 0

    restored = json.loads(json.dumps(agent.session))
    revived = CodingForMe(
        model_client=FakeModelClient([]),
        workspace=build_workspace(tmp_path),
        session_store=SessionStore(tmp_path / ".codingforme" / "sessions"),
        session=restored,
        approval_policy="auto",
    )
    _, prompt, metadata = ContextManager(revived, total_budget=4823).build_all("next")

    assert metadata["context_pressure"]["session_summary"]["covered_entries"] == before["covered"]
    assert "requirement 0:" in prompt


def test_a_reset_clears_the_summary_along_with_the_history(tmp_path):
    """`/reset` 清空 history,覆盖点必须一起清——否则下一轮拿一个陈旧的下标去切空历史。"""
    agent = build_agent(tmp_path, [])
    _dialogue_turns(agent)
    assert agent.session["context_summary"]["covered"] > 0

    agent.reset()
    _, prompt, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    assert int(agent.session["context_summary"].get("covered", 0) or 0) == 0
    assert metadata["context_pressure"]["session_summary"]["covered_entries"] == 0
    assert "requirement 0:" not in prompt


def test_the_summary_block_is_written_even_when_nothing_was_compacted(tmp_path):
    """零值也要写。

    「机制一次都没执行」在报告里长得和「没问题」一模一样——这是这套东西上一轮踩过
    的坑(`run_plan` 的调用数、落盘的 spill 计数都是同一个教训)。
    """
    agent = build_agent(tmp_path, [])
    agent.record({"role": "user", "content": "short"})

    _, _, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    summary = metadata["context_pressure"]["session_summary"]
    assert summary == {
        "enabled": True,
        "compactions": 0,
        "covered_entries": 0,
        "summary_tokens": 0,
        "covered_tokens": 0,
        "refreshes": 0,
    }
    assert metadata["message_layout"]["session_summary_present"] is False


def test_turning_off_the_session_summary_falls_back_to_dropping_entries(tmp_path):
    """消融开关必须严格退回机制出现之前的行为:按预算逐条丢,丢掉的只剩一行 digest。

    没有这条,A/B 两边可能跑的是同一份代码,而工件上看不出来。
    """
    agent = build_agent(tmp_path, [])
    agent.feature_flags = {**agent.feature_flags, "session_summary": False}
    rows = _dialogue_turns(agent)
    metadata, prompt, _ = rows[-1]

    assert metadata["context_pressure"]["session_summary"]["enabled"] is False
    assert metadata["context_pressure"]["session_summary"]["covered_entries"] == 0
    assert metadata["sections"]["history"]["omitted_entry_count"] > 0
    assert metadata["message_layout"]["omitted_digest_present"] is True
    assert "requirement 0:" not in prompt


def test_the_summary_is_inert_once_the_protected_tail_alone_exceeds_the_target(tmp_path):
    """这条锁的是**这个机制的适用条件**,不是它的收益。

    摘要能不能省下缓存作废,取决于一件事:被保护的那截最近历史(覆盖点的上限,
    `_max_summary_coverage()`)本身放不放得进目标点。放得进,一次跳跃就能腾出好几轮
    的空间,边界几轮才动一次;放不进,覆盖点就被死死顶在那个上限上,而上限每轮涨 2
    条、对话每轮也长 2 条——于是边界照旧每轮动,和关掉它完全一样。**这和阶段一那个
    被撤掉的迟滞栽在同一个结构上**,区别只是这次上限是「不许动最近这几条」而不是
    「不许比上一轮少丢」。

    实测(预算 4,823、目标点 2,170、24 轮对话):每条约 96 token 时受保护窗口 1,152,
    边界移动 7 → 1;每条约 206 token 时窗口 2,472,14 → 5;每条约 306 token 时窗口
    3,672,19 → 19,一点没省。下面用的就是最后那个量级。

    留这条测试是为了**别再靠调比例去救它**:目标点从 45% 调到 30% 也不行,受保护的
    那截本来就在目标点之上。真要在这个量级上省缓存,只能动「保护多少条」这个约束,
    而那条约束换来的是「模型看得见自己上一轮做了什么」。
    """
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    heavy = 300

    with_summary = _dialogue_turns(build_agent(tmp_path / "on", []), filler=heavy)
    without = build_agent(_subdir(tmp_path, "off"), [])
    without.feature_flags = {**without.feature_flags, "session_summary": False}
    without_summary = _dialogue_turns(without, filler=heavy)

    assert _head_moves(with_summary) >= _head_moves(without_summary) - 1, (
        "这个量级下摘要救不了边界移动;如果这条挂了,说明适用条件变了,去更新上面那段说明"
    )


# --- 最近窗口的条目上限也按块推进(阶段三)-------------------------------------


def _prefix_rewrites(agent, budget=12423, turns=24, filler=150):
    """一轮一轮地组上下文,数「messages 前缀被从中间改写」的轮次。

    **最后两条本来就每轮都变**:倒数第一条是当前请求,倒数第二条是运行状态
    (working memory + relevant memory + 摘要 + digest)。摆位设计就是把每轮都变的
    东西放到最后,所以它们变不算前缀作废——真正要看的是它们前面那一段。
    """
    rewrites = 0
    previous = None
    starts = []
    for index in range(turns):
        agent.record({"role": "user", "content": f"requirement {index}: " + ("spec " * filler)})
        agent.record({"role": "assistant", "content": f"acknowledged {index}: " + ("recap " * filler)})
        messages, _, metadata = ContextManager(agent, total_budget=budget).build_all("next")
        starts.append(metadata["history"]["recent_start"])
        current = [(m.get("role"), m.get("content")) for m in messages]
        if previous is not None:
            before, after = previous[:-2], current[:-2]
            shared = min(len(before), len(after))
            first_diff = next((i for i in range(shared) if before[i] != after[i]), shared)
            if first_diff < len(before):
                rewrites += 1
        previous = current
    return rewrites, starts


def test_the_recent_window_entry_cap_advances_in_blocks_not_every_turn(tmp_path):
    """条目数那条上限也要成块推进,否则 `RECENT_WINDOW_BLOCK` 只做了一半。

    `_blocked_recent_window()` 量化的是**工具轮**那条边界;条目数上限
    (`len(history) - 2 * tool_turns`)管的是「工具很少、对话很多」的历史,而它从前
    每多两条对话就往前推两格。后果实测得到:24 轮真实压力探针、预算 12,423、
    **一次裁剪都没触发过**,23 次组装里仍有 15 次的前缀在中间位置被改写,改写点恰好
    每轮前移 2——全部来自这一条。量化之后同一条负载降到 5 次。

    这是这套上下文工程里少见的、**在没有任何预算压力的常规运行上**就能兑现的收益,
    而常规运行正是真实跑批的形态(三个窗口档位 178 个真实预算轮次,裁剪触发 0 次)。
    """
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    blocked_rewrites, blocked_starts = _prefix_rewrites(build_agent(tmp_path / "on", []))

    unblocked = build_agent(_subdir(tmp_path, "off"), [])
    unblocked.feature_flags = {**unblocked.feature_flags, "recent_window_block": False}
    plain_rewrites, plain_starts = _prefix_rewrites(unblocked)

    assert blocked_rewrites * 2 < plain_rewrites, (blocked_rewrites, plain_rewrites)
    # 边界只落在 6 的倍数上（2 条记录 × RECENT_WINDOW_BLOCK）。
    assert {start % 6 for start in blocked_starts} == {0}, blocked_starts
    # 关掉之后它每轮都在动，这正是要治的那个病。
    assert len(set(plain_starts)) > len(set(blocked_starts))


def test_squeezed_entries_are_counted_apart_from_dropped_ones(tmp_path):
    """「压扁了」和「整条丢了」必须在工件上分得开。

    两者对模型的后果差得很远:一个是「这一轮发生过、内容看不清」,一个是「这一轮
    根本不存在」。它们从前都只体现为 history 变短,谁也数不出各占多少。
    """
    agent = build_agent(tmp_path, [])
    # 关掉会话摘要:开着它,最早那批条目会被整体换成摘要,保留循环根本轮不到,
    # 而这条测试要验的正是保留循环那两档(压扁 / 丢弃)分不分得开。
    agent.feature_flags = {**agent.feature_flags, "session_summary": False}
    _long_history(agent, turns=24, filler=200)

    _, _, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    history = metadata["history"]
    assert "squeezed_entry_count" in history
    dropped = metadata["sections"]["history"]["omitted_entry_count"]
    assert history["squeezed_entry_count"] > 0 and dropped > 0, (history["squeezed_entry_count"], dropped)


def _reduction_targets(agent, total_budget):
    _, prompt, metadata = ContextManager(agent, total_budget=total_budget).build_all("continue")
    return prompt, metadata, metadata["budget_reductions"]


def test_clear_at_least_never_binds_while_graded_compression_is_on(tmp_path):
    """分级压缩开着的时候，`clear_at_least` 恒不生效——这是算术上必然的。

    `overflow = prompt_tokens - target_tokens`，而循环只在 `prompt_tokens > trigger`
    时才进得来，所以 overflow 至少是 `(0.85 - 0.70) × 预算 = 预算的 15%`，而
    `clear_at_least` 只要求腾出预算的 10%。15% > 10%，`max()` 永远取前者。

    这条不是在锁「机制没用」，是在锁**两个常量之间的关系**：谁要是把触发点和目标点
    拉近到 1/10 以内（比如 0.85 / 0.80），`clear_at_least` 就会立刻变成那个说了算的
    约束，而这条测试会挂，逼他知道自己改的不只是一个阈值。
    """
    from codingforme.context_manager import (
        CLEAR_AT_LEAST_DIVISOR,
        COMPRESSION_TARGET_RATIO,
        COMPRESSION_TRIGGER_RATIO,
    )

    assert COMPRESSION_TRIGGER_RATIO - COMPRESSION_TARGET_RATIO > 1 / CLEAR_AT_LEAST_DIVISOR

    agent = build_agent(tmp_path, [])
    agent.feature_flags = {**agent.feature_flags, "session_summary": False}
    _history_with_tool_reads(agent, 12, " ".join(f"word{index}" for index in range(400)))

    _, _, reductions = _reduction_targets(agent, 3000)

    assert reductions, "这份历史应该顶过触发点"
    for entry in reductions:
        assert entry["target_tokens"] == entry["overflow_tokens"], entry


def test_clear_at_least_leaves_headroom_when_it_is_the_only_thing_holding_the_line(tmp_path):
    """把分级关掉，`clear_at_least` 就有作用对象了：它把「刚好压回预算线」变成
    「压到线下面留一截余量」。

    余量就是这个机制的**全部**意义——不留余量的话，下一轮再多一句话就又要裁一次，
    而每裁一次 history 就作废它之后的全部前缀缓存。
    """
    def build(clear_at_least):
        root = tmp_path / ("on" if clear_at_least else "off")
        root.mkdir()
        agent = build_agent(root, [])
        agent.feature_flags = {
            **agent.feature_flags,
            "session_summary": False,
            "graded_compression": False,
            "clear_at_least": clear_at_least,
        }
        # 170 个词是特意挑的：它让 overflow（31）远小于预算的 1/10（300），
        # 也就是 `max()` 里 `clear_at_least` 那一项说了算的唯一区间。
        _history_with_tool_reads(agent, 12, " ".join(f"word{index}" for index in range(170)))
        return _reduction_targets(agent, 3000)

    _, on_meta, on_reductions = build(True)
    _, off_meta, off_reductions = build(False)

    assert on_reductions and off_reductions
    assert on_reductions[0]["overflow_tokens"] == off_reductions[0]["overflow_tokens"], "两边的压力要一样大"
    assert on_reductions[0]["target_tokens"] == 3000 // 10
    assert off_reductions[0]["target_tokens"] == off_reductions[0]["overflow_tokens"]
    # 开着的那边压完离预算线更远，这就是省下的那次重裁。
    assert on_meta["prompt_tokens"] < off_meta["prompt_tokens"] - 100, (
        on_meta["prompt_tokens"],
        off_meta["prompt_tokens"],
    )


def test_turning_reversible_squeeze_off_drops_the_entry_instead_of_stubbing_it(tmp_path):
    """可逆折叠的消融：关掉之后窗口外放不下的条目整条消失，不再留残句。

    关掉要**严格**退回这个机制出现之前的行为（直接丢弃），不是退回某个第三种形态，
    否则 A/B 测的就不是这个机制。
    """
    def build(reversible_squeeze):
        root = tmp_path / ("on" if reversible_squeeze else "off")
        root.mkdir()
        agent = build_agent(root, [])
        agent.feature_flags = {
            **agent.feature_flags,
            "session_summary": False,
            "reversible_squeeze": reversible_squeeze,
        }
        _long_history(agent, turns=24, filler=200)
        _, prompt, metadata = ContextManager(agent, total_budget=4823).build_all("next")
        return prompt, metadata

    on_prompt, on_meta = build(True)
    _, off_meta = build(False)

    assert on_meta["history"]["squeezed_entry_count"] > 0
    assert off_meta["history"]["squeezed_entry_count"] == 0
    # 残句留住的那几条，关掉之后就整条掉进 `Omitted context:` 那一行了。
    on_dropped = on_meta["sections"]["history"]["omitted_entry_count"]
    off_dropped = off_meta["sections"]["history"]["omitted_entry_count"]
    assert off_dropped > on_dropped, (on_dropped, off_dropped)
    # 残句保留开头，所以那一轮的编号仍然认得出来——这就是「模型知道这一轮发生过」。
    assert "Transcript:" in on_prompt


def test_a_squeezed_stub_still_names_the_turn_it_stands_for(tmp_path):
    """残句必须仍然认得出是哪一轮，否则它和「整条丢掉」在模型眼里没区别。

    `_squeeze` 走的是保留开头的裁剪，所以条目开头那句 `requirement N:` 活得下来。
    这条锁的是**裁剪方向**：换成保留结尾的话残句会变成一串没有主语的尾巴，
    10 个 token 全花在废话上。
    """
    agent = build_agent(tmp_path, [])
    agent.feature_flags = {**agent.feature_flags, "session_summary": False}
    _long_history(agent, turns=24, filler=200)

    _, prompt, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    assert metadata["history"]["squeezed_entry_count"] > 0
    stubs = [line for line in prompt.splitlines() if line.strip().endswith("...")]
    assert stubs, "没有找到任何残句"
    assert any("requirement" in line or "acknowledged" in line for line in stubs), stubs[:5]


def _scan_turns(agent, budget=4823, turns=22, chunk=100):
    """模拟「分块扫一个长文件」：每轮读下一段区间，并写一句结论。

    这是 `long_log_audit_token` 的形状。真实跑批里它在第 14 轮重新读了第 2 轮就
    已经读过的 1~100 行——因为摘要把 7 次 `read_file` 压成了一个计数 `read_file x7`，
    「读过哪几段」这件事在上下文里根本不存在了。
    """
    rows = []
    for index in range(turns):
        start = index * chunk + 1
        end = start + chunk - 1
        # 垫料是必需的：阶段零量过，工具结果那一路会被 L1/L3 吃干净，
        # 唯一能把 history 撑到触发压缩的成分是 user / assistant 的文本。
        agent.record({"role": "user", "content": f"keep scanning chunk {index}: " + ("spec " * 90)})
        agent.record(
            {
                "role": "tool",
                "name": "read_file",
                "args": {"path": "audit.log", "start": start, "end": end},
                "content": f"line {start} ... line {end} " + ("noise " * 40),
            }
        )
        agent.record(
            {"role": "assistant", "content": f"no token in lines {start}-{end}; " + ("recap " * 90)}
        )
        _, prompt, metadata = (lambda cm: (cm.build_all("next")))(ContextManager(agent, total_budget=budget))
        rows.append((metadata, prompt))
    return rows


def test_the_summary_keeps_which_ranges_were_already_scanned(tmp_path):
    """摘要必须说出「读过哪几段」，不能只说「读了 7 次」。

    实测缺陷（这条测试就是为它补的）：`_render_session_summary()` 从前把每条调用的
    签名压成聚合计数 `already run: read_file x7`，于是分块扫长文件的任务丢掉了区间
    信息，模型在第 14 轮又把第 2 轮读过的 1~100 行读了一遍——白烧一个约 17 秒的
    往返，还正好撞上重复调用检测。
    """
    agent = build_agent(tmp_path, [])
    rows = _scan_turns(agent)
    metadata, prompt = rows[-1]

    summary = metadata["context_pressure"]["session_summary"]
    assert summary["covered_entries"] > 0, "这条负载必须真的触发过摘要，否则下面断言什么都没查"
    # 被摘要覆盖掉的那几段区间，仍然要在 prompt 里说得出来。连着读的区间会合并成
    # 一段（22 条签名合并成一条，否则 289 token 的摘要预算根本装不下），所以这里
    # 断言的是"合并后的那一段确实盖住了扫过的范围"，不是逐条区间。
    span = re.search(r"read_file audit\.log:(\d+)-(\d+)", prompt)
    assert span, prompt[-1500:]
    assert int(span.group(1)) == 1 and int(span.group(2)) >= 1000, span.groups()
    assert "read_file x" not in prompt, "退回纯计数说明区间信息又丢了"


def test_the_summary_keeps_the_models_own_conclusions(tmp_path):
    """模型自己的结论比工具结果更该留：文件能重读，推理不能。

    从前这里只写一行 `N assistant turn(s) not repeated here`，等于把整段推理抹掉，
    而「可恢复才可丢弃」这条原则恰恰指向相反的取舍。
    """
    agent = build_agent(tmp_path, [])
    rows = _scan_turns(agent)
    metadata, prompt = rows[-1]

    assert metadata["context_pressure"]["session_summary"]["covered_entries"] > 0
    assert "earlier finding:" in prompt
    assert "assistant turn(s) not repeated here" not in prompt


def test_the_summary_falls_back_to_counts_when_the_signatures_do_not_fit(tmp_path):
    """预算紧时退回纯计数，而不是把清单截成半句话。

    优先级是：用户原话 > 模型结论 > 带区间的签名 > 文件清单。签名那一行装不下时
    必须整行换成计数——半个清单比没有更糟，模型会以为剩下的没做过。
    """
    agent = build_agent(tmp_path, [])
    items = [
        {"role": "tool", "name": "read_file", "args": {"path": f"pkg/module_{i}.py", "start": 1, "end": 400}, "content": "x"}
        for i in range(40)
    ]
    manager = ContextManager(agent, total_budget=4823)
    rendered = manager._render_session_summary(items)

    assert "read_file x40" in rendered, rendered
    assert len(rendered.split("\n")) < 8, "退回计数之后不该还有一长串签名"


def _read_then_write_history(agent, write_name="patch_file", wrote=None, write_content="patched"):
    """读一个文件 → 改同一个文件。两条都留在最近窗口里。"""
    agent.record({"role": "user", "content": "fix the greeting"})
    agent.record(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "app.py"},
            "content": "def greet():\n    return 'OLD TEXT ANCHOR'\n",
        }
    )
    entry = {
        "role": "tool",
        "name": write_name,
        "args": {"path": "app.py"},
        "content": write_content,
    }
    if wrote is not None:
        entry["wrote"] = wrote
    agent.record(entry)


def test_a_read_is_marked_stale_once_the_same_file_has_been_written(tmp_path):
    """写过之后，先前那次读的**全文**不能再原样呈现。

    实测动机：那段内容是改动**之前**的，而它在最近窗口里是全文，和「文件现在就长
    这样」在模型眼里完全一样。模型于是从这段过期全文里抄 `old_text` 去发下一次
    `patch_file`，命中 0 次被打回（k=3 跑批里 `old_text` 没命中占被拒调用的 14%），
    白烧一个约 17 秒的往返。
    """
    agent = build_agent(tmp_path, [])
    _read_then_write_history(agent, wrote=["app.py"])

    _, prompt, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    assert "OLD TEXT ANCHOR" not in prompt, "改动前的全文仍然原样留在上下文里"
    assert context_manager.STALE_READ_MARKER in prompt
    # 调用记录本身要留着：模型得知道自己读过这个文件，否则它会以为没读过。
    assert "read_file" in prompt
    assert metadata["history"]["stale_read_count"] == 1


def test_turning_stale_read_invalidation_off_brings_the_old_content_back(tmp_path):
    """证伪用例：关掉开关必须精确退回「全文照旧」，而不是第三种形态。

    没有这一条，上面那条断言可能是被别的机制（去重、窗口外清理）顺手做掉的，
    而不是这个机制在起作用。
    """
    agent = build_agent(tmp_path, [], feature_flags={"stale_read_invalidation": False})
    _read_then_write_history(agent, wrote=["app.py"])

    _, prompt, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    assert "OLD TEXT ANCHOR" in prompt
    assert context_manager.STALE_READ_MARKER not in prompt
    assert metadata["history"]["stale_read_count"] == 0


def test_a_write_that_failed_does_not_invalidate_the_read_it_needs(tmp_path):
    """失败的 patch 一个字节都没改，那次读仍然新鲜——而且正是重试要抄的东西。

    方向搞反的后果比不做还糟：模型手里唯一正确的 `old_text` 来源被抹掉，它只能
    重新 read_file，多烧一个往返，还可能撞上重复调用检测。
    """
    agent = build_agent(tmp_path, [])
    # runtime 报的是快照差异，失败的写自然是空列表。
    _read_then_write_history(agent, wrote=[], write_content="error: old_text must occur exactly once, found 0")

    _, prompt, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    assert "OLD TEXT ANCHOR" in prompt
    assert metadata["history"]["stale_read_count"] == 0


def test_an_old_session_without_the_wrote_field_still_gets_invalidation(tmp_path):
    """老会话的 history 没有 `wrote` 字段，要能从 args 里的 path 回落判断。

    这半边覆盖不了 `run_plan`（内层调用的路径不在 args 里），但覆盖得了
    write_file / patch_file——也就是绝大多数情况。
    """
    agent = build_agent(tmp_path, [])
    _read_then_write_history(agent, wrote=None)

    _, prompt, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    assert "OLD TEXT ANCHOR" not in prompt
    assert metadata["history"]["stale_read_count"] == 1


def test_a_read_taken_after_the_write_is_not_stale(tmp_path):
    """写之后再读一次，那次读是新鲜的，不许被作废。

    没有这条，「作废」可以退化成「只要这个文件被写过，所有读都作废」——那样模型
    永远看不到自己刚读到的当前内容。
    """
    agent = build_agent(tmp_path, [])
    _read_then_write_history(agent, wrote=["app.py"])
    agent.record(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "./app.py"},  # 故意换个写法，验证路径归一
            "content": "def greet():\n    return 'FRESH TEXT ANCHOR'\n",
        }
    )

    _, prompt, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    assert "FRESH TEXT ANCHOR" in prompt
    assert "OLD TEXT ANCHOR" not in prompt
    assert metadata["history"]["stale_read_count"] == 1


def test_a_write_to_another_file_leaves_the_read_alone(tmp_path):
    """只作废被写过的那个文件的读，别的文件不受牵连。"""
    agent = build_agent(tmp_path, [])
    agent.record(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "app.py"},
            "content": "OLD TEXT ANCHOR",
        }
    )
    agent.record(
        {
            "role": "tool",
            "name": "patch_file",
            "args": {"path": "other.py"},
            "content": "patched",
            "wrote": ["other.py"],
        }
    )

    _, prompt, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    assert "OLD TEXT ANCHOR" in prompt
    assert metadata["history"]["stale_read_count"] == 0


# --- P5：窗口外的落盘指针也要标过期 -------------------------------------------


SPILL_PATH = ".codingforme/tool_outputs/r1/1-read_file.txt"


def _spilled_read(path, anchor):
    """一条**落过盘**的 read_file 结果：预览 + 末尾那行指针。"""
    marker = context_manager.spill_marker(SPILL_PATH, 9000, total_lines=300, chunk_lines=200)
    return {
        "role": "tool",
        "name": "read_file",
        "args": {"path": path},
        "content": f"{anchor}\npreview only\n{marker}",
    }


def _push_out_of_window(agent, count=12):
    """垫够工具轮，把前面的条目挤出最近窗口（窗口是 6~8 个工具轮，按块浮动）。"""
    for index in range(count):
        agent.record(
            {
                "role": "tool",
                "name": "read_file",
                "args": {"path": f"filler_{index}.py"},
                "content": f"filler body {index}",
            }
        )


def test_a_spilled_pointer_says_the_saved_copy_is_stale_once_the_file_is_written(tmp_path):
    """窗口外的落盘指针,在同一个文件被写过之后必须说出「盘上那份是旧版」。

    这是 P3 漏掉的那一半。P3 的判断写在 `_policy_for()` 的 `if recent:` 内部,只管
    最近窗口。掉出窗口之后这条记录只剩一行指针,而指针指向的
    `.codingforme/tool_outputs/...` 是**读的那一刻**写下的快照——原文件被改了它不会
    跟着变,指针原文却正好在教模型去读它(还给了可以照抄的 `read_file(path=...)`)。
    不标出来,模型取回的是改动之前那一版,而且看起来像是刚读的。
    """
    agent = build_agent(tmp_path, [])
    agent.record(_spilled_read("big.py", "OLD SPILLED ANCHOR"))
    agent.record(
        {
            "role": "tool",
            "name": "patch_file",
            "args": {"path": "big.py"},
            "content": "patched",
            "wrote": ["big.py"],
        }
    )
    _push_out_of_window(agent)

    _, prompt, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    assert context_manager.STALE_SPILL_MARKER in prompt
    assert metadata["history"]["stale_pointer_count"] == 1
    # 它仍然是一条指针,所以照旧计进 `spilled_pointer_count`——那个数回答的是
    # 「有几条被清掉的结果还留着取回线索」,过期与否不改变这个事实。
    assert metadata["history"]["spilled_pointer_count"] >= 1


def test_a_stale_pointer_still_keeps_the_path_so_the_full_text_stays_recoverable(tmp_path):
    """过期**不等于**该丢掉指针:盘上那份仍是取回全文的唯一线索。

    如果这里退化成 `CLEARED_RESULT_MARKER`,这条记录就从「可恢复」掉回「不可恢复」,
    而 L1 落盘存在的全部理由就是可恢复(Manus 那条「可恢复才可丢弃」)。正确的做法
    是指针照留、后面接一句说清它是旧版。
    """
    agent = build_agent(tmp_path, [])
    agent.record(_spilled_read("big.py", "OLD SPILLED ANCHOR"))
    agent.record(
        {
            "role": "tool",
            "name": "patch_file",
            "args": {"path": "big.py"},
            "content": "patched",
            "wrote": ["big.py"],
        }
    )
    _push_out_of_window(agent)

    _, prompt, _ = ContextManager(agent, total_budget=4823).build_all("next")

    line = next(row for row in prompt.split("\n") if context_manager.STALE_SPILL_MARKER in row)
    # 一行之内三样都要在：原路径（重读用）、落盘路径（取回用）、过期警告。
    assert SPILL_PATH in line
    assert "big.py" in line
    assert context_manager.CLEARED_RESULT_MARKER not in line
    assert "OLD SPILLED ANCHOR" not in prompt, "预览段本来就该随窗口外清理一起消失"


def test_an_untouched_spill_pointer_keeps_its_plain_marker(tmp_path):
    """反向用例：没被写过的落盘指针一个字都不该变。

    没有这一条，上面那两条可能是被别的机制顺手做掉的——而「所有指针都带警告」
    等于这条警告什么都没说。
    """
    agent = build_agent(tmp_path, [])
    agent.record(_spilled_read("big.py", "OLD SPILLED ANCHOR"))
    agent.record(
        {
            "role": "tool",
            "name": "patch_file",
            "args": {"path": "other.py"},
            "content": "patched",
            "wrote": ["other.py"],
        }
    )
    _push_out_of_window(agent)

    _, prompt, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    assert SPILL_PATH in prompt, "指针本身还要在"
    assert context_manager.STALE_SPILL_MARKER not in prompt
    assert metadata["history"]["stale_pointer_count"] == 0


def test_turning_stale_read_invalidation_off_also_turns_off_the_pointer_warning(tmp_path):
    """同一个消融开关管两半。分成两个开关的话，`no_stale_read` 这个变体就只关掉
    一半机制，跑出来的 A/B 测的是「半个供给侧新鲜度」，而那不是任何人想要的对照。
    """
    agent = build_agent(tmp_path, [], feature_flags={"stale_read_invalidation": False})
    agent.record(_spilled_read("big.py", "OLD SPILLED ANCHOR"))
    agent.record(
        {
            "role": "tool",
            "name": "patch_file",
            "args": {"path": "big.py"},
            "content": "patched",
            "wrote": ["big.py"],
        }
    )
    _push_out_of_window(agent)

    _, prompt, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    assert context_manager.STALE_SPILL_MARKER not in prompt
    assert SPILL_PATH in prompt
    assert metadata["history"]["stale_pointer_count"] == 0


def test_the_two_stale_markers_say_different_things(tmp_path):
    """窗口内外两句标记不能是同一句,这正是把它们拆成两个常量的理由。

    窗口内那条的内容还在上下文里,「re-read it」指的就是原文件;窗口外只剩一行指针,
    模型手边最容易照抄的重读动作恰恰是去读那个**同样过期**的落盘快照。一句话通用
    的话,这条警告会把模型推向它要防的那件事。
    """
    assert context_manager.STALE_SPILL_MARKER != context_manager.STALE_READ_MARKER
    assert "saved copy" in context_manager.STALE_SPILL_MARKER
    assert "original path" in context_manager.STALE_SPILL_MARKER


def test_a_real_write_through_run_tool_reports_the_paths_it_changed(tmp_path):
    """端到端：`wrote` 必须来自**工作区快照的前后差异**，不是 args 里的 path。

    这是这个机制的供给侧。取快照差异而不是 args，是为了覆盖三种 args 读不出路径
    的情况：`run_shell` 改文件、`run_plan` 内层调用改文件、以及「写进去内容完全
    相同因此文件其实没变」。
    """
    (tmp_path / "app.py").write_text("old body\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            models.tool_call("patch_file", path="app.py", old_text="old body", new_text="new body"),
            models.final_answer("done"),
        ],
    )
    agent.ask("update app.py")

    writes = [item for item in agent.session["history"] if item.get("name") == "patch_file"]
    assert writes and writes[0]["wrote"] == ["app.py"]
    # 只读工具照样带这个键，值是空列表——`_written_paths()` 靠「键在不在」区分
    # 「runtime 说了、就是没改」和「老会话没这个字段、只能从 args 猜」。
    reads = [item for item in agent.session["history"] if item.get("name") == "read_file"]
    assert all(item.get("wrote") == [] for item in reads)


# --- P4：手动压缩(/compact)与 L4 可逆性 ---------------------------------------


def _subdir(tmp_path, name):
    """每个 agent 一个独立的工作区目录。`build_workspace()` 直接写 README.md，
    父目录不存在会 FileNotFoundError。"""
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dialogue_history(agent, turns=24, filler=40):
    """一段够长、能真的触发摘要的对话历史。"""
    for index in range(turns):
        agent.record({"role": "user", "content": f"spec item {index}: " + "detail " * filler})
        agent.record({"role": "assistant", "content": f"conclusion {index}: " + "reasoning " * filler})


def test_compact_now_pushes_the_summary_as_far_as_the_structure_allows(tmp_path):
    """`/compact` 不等占用率到 85%，当场把覆盖点推到结构允许的最远处。

    存在的理由：自动压缩是**占用率驱动**的，而用户知道一些系统不知道的事——
    「这一段调查结束了」「换个话题」。那一刻主动压，压缩点比等它涨到 85% 更靠前，
    被作废的前缀缓存也更少。
    """
    agent = build_agent(tmp_path, [])
    _dialogue_history(agent)

    report = agent.compact_context()

    assert report["compacted"] is True
    assert report["newly_covered"] > 0
    assert report["covered"] == report["newly_covered"]
    # 手动压缩必须真的省下东西，否则这个命令只是把摘要正文塞进上下文。
    assert report["after_tokens"] < report["before_tokens"]
    assert report["saved_tokens"] == report["before_tokens"] - report["after_tokens"]
    assert report["summary"].startswith(context_manager.SESSION_SUMMARY_HEADER)


def test_compact_now_acts_without_waiting_for_the_pressure_threshold(tmp_path):
    """手动压缩的价值不是「压得更狠」，是「不用等压力上来」。

    量过了：一旦占用率真的到了触发点，自动那条也会一跳跳到
    `_max_summary_coverage()` 的上限，和手动一模一样（本仓库这段负载上都是 36）——
    上限由热尾决定，不由预算决定。所以两者的差别只在**什么时候动手**：同一份历史、
    同一个预算，占用率没到触发点时自动那条一条都不压，手动那条照压不误。
    """
    budget = 4823  # 这段负载在这个预算下**不会**触发自动压缩（占用率没到 85%）。
    auto = build_agent(_subdir(tmp_path, "auto"), [])
    _dialogue_history(auto)
    ContextManager(auto, total_budget=budget).build_all("next")
    assert int(auto.session.get("context_summary", {}).get("covered", 0)) == 0, (
        "这个预算下不该触发自动压缩，否则这条测试测的是另一件事"
    )

    manual = build_agent(_subdir(tmp_path, "manual"), [])
    _dialogue_history(manual)
    ContextManager(manual, total_budget=budget).build_all("next")

    assert manual.compact_context()["covered"] > 0


def test_compact_says_why_it_did_nothing(tmp_path):
    """推不动时要说出**哪一种**推不动，三种情况用户该做的事完全不同。"""
    short = build_agent(_subdir(tmp_path, "short"), [])
    short.record({"role": "user", "content": "hello"})
    assert "too short" in short.compact_context()["reason"]

    done = build_agent(_subdir(tmp_path, "done"), [])
    _dialogue_history(done)
    done.compact_context()
    again = done.compact_context()
    assert again["compacted"] is False
    assert "already compacted" in again["reason"]

    off = build_agent(_subdir(tmp_path, "off"), [], feature_flags={"session_summary": False})
    _dialogue_history(off)
    assert "turned off" in off.compact_context()["reason"]


def test_compact_persists_so_a_resume_does_not_undo_it(tmp_path):
    """压完要落盘。不落盘的话下一次 resume 会拿回压缩之前的状态，用户敲的那条命令
    等于没执行——而 REPL 上看不出任何区别。
    """
    agent = build_agent(tmp_path, [])
    _dialogue_history(agent)
    covered = agent.compact_context()["covered"]

    saved = json.loads(Path(agent.session_path).read_text(encoding="utf-8"))

    assert saved["context_summary"]["covered"] == covered
    assert saved["context_summary"]["text"].startswith(context_manager.SESSION_SUMMARY_HEADER)


def test_compression_never_mutates_the_transcript_it_projects(tmp_path):
    """L4「完全可逆」这句话的凭据：无论压得多狠，`history` 一个字节都不变。

    这是整个投影层的地基。它错了不会报错，只会**悄悄**丢数据——而丢了之后
    没有任何工件能看出来，因为唯一的真相源已经被改掉了。
    """
    agent = build_agent(tmp_path, [])
    _dialogue_history(agent)
    for path in ("a.py", "b.py", "c.py"):
        agent.record(
            {
                "role": "tool",
                "name": "read_file",
                "args": {"path": path},
                "content": f"UNIQUE-BODY-{path} " + "line " * 200,
            }
        )
    before = copy.deepcopy(agent.session["history"])

    # 从宽松到掐死，每一档都渲染一次。
    for budget in (60_000, 12_000, 4_823, 900, 300):
        ContextManager(agent, total_budget=budget).build_all("next")
        assert agent.session["history"] == before, f"budget={budget} 改动了 history"

    # 手动压缩同理：它改的是 `context_summary` 里的覆盖点，不是历史本身。
    agent.compact_context()
    assert agent.session["history"] == before


def test_a_squeezed_transcript_comes_back_in_full_at_a_bigger_budget(tmp_path):
    """可逆性的正面验证：同一份历史，把预算放大，被压掉的内容原样回来。

    只断言「history 没被改」还不够——那只证明数据还在，不证明**投影读得回来**。
    这条走的是 Claude Code `projectView()` 那个语义：压缩是呈现层的事，换个预算
    重新投影一次就还原了。
    """
    agent = build_agent(tmp_path, [])
    _dialogue_history(agent, turns=6)
    agent.record(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "deep.py"},
            "content": "UNIQUE-BODY-MARKER " + "line " * 300,
        }
    )
    _dialogue_history(agent, turns=6)

    _, tight, _ = ContextManager(agent, total_budget=900).build_all("next")
    assert "UNIQUE-BODY-MARKER" not in tight

    # 覆盖点要清掉：手动/自动压缩推进过它之后，摘要会一直顶在那儿——那是
    # **状态**，不是投影。可逆的是投影，状态要显式回退。
    agent.session["context_summary"] = {"covered": 0, "text": "", "covered_tokens": 0, "refreshes": 0}
    _, roomy, _ = ContextManager(agent, total_budget=120_000).build_all("next")

    assert "UNIQUE-BODY-MARKER" in roomy


def test_the_compact_report_names_the_reason_when_it_did_nothing(tmp_path):
    """REPL 上显示的那段文字本身也要说清原因，不能只回一句「没压」。"""
    from codingforme.cli import _compact_report

    text = _compact_report({"compacted": False, "reason": "the transcript is too short to compact (1 entries; at least 6 are always kept)", "entries": 1, "covered": 0})
    assert "too short" in text
    assert "1 transcript entries" in text

    agent = build_agent(tmp_path, [])
    _dialogue_history(agent)
    done = _compact_report(agent.compact_context())
    assert "compacted" in done
    assert "tokens (saved" in done
    # 摘要正文要当场显示：压缩是用户主动发起的，只报一个 token 差值等于让他
    # 相信压掉的部分无关紧要。
    assert context_manager.SESSION_SUMMARY_HEADER in done

# --- P7:压缩要在工件上留痕,占用率要在 REPL 上看得见 -------------------------


def _trace_events(agent, name):
    """把这个 agent 写下的所有 trace 事件里叫 `name` 的挑出来。"""
    root = Path(agent.workspace.repo_root) / ".codingforme" / "runs"
    found = []
    for path in root.rglob("trace.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event") == name:
                found.append(event)
    return found


def test_a_manual_compaction_leaves_a_trace_event(tmp_path):
    """`/compact` 改的是模型看得见什么,这件事必须在 trace 上留痕。

    在这条测试之前手动压缩**一个字节都不写**:压完落盘、resume 接着用,而工件上
    完全看不出上下文被换过。于是「这次运行为什么不记得第 1 轮的规格」只能靠重算
    history 去猜。
    """
    agent = build_agent(tmp_path, [models.final_answer("done")])
    agent.ask("kick off a run so there is a trace to write into")
    _long_history(agent, turns=24, filler=200)

    report = agent.compact_context()

    assert report["compacted"] is True
    events = _trace_events(agent, "context_compacted")
    assert len(events) == 1, events
    event = events[0]
    assert event["trigger"] == "manual"
    assert event["newly_covered"] == report["newly_covered"]
    assert event["covered_entries"] == report["covered"]
    assert event["transcript_entries"] == len(agent.session["history"])
    # 换进去的比换掉的小多少。这个数可以是负数（摘要比它替代的还长），所以断的是
    # 恒等式而不是「大于零」——把它 clamp 成 0 就等于把帮倒忙的情况藏起来。
    assert event["saved_tokens"] == event["covered_tokens"] - event["summary_tokens"]


def test_an_automatic_compaction_writes_the_same_event_with_a_different_trigger(tmp_path):
    """压力驱动那条走同一个事件、同一份字段,只有 `trigger` 不同。

    分 manual / auto 是因为它们对调参是相反的信号:auto 变多说明触发点或者预算该调,
    manual 变多说明用户在替系统判断「这一段结束了」。混成一个事件就分不出来了。
    """
    agent = build_agent(tmp_path, [models.final_answer("ok")])
    _long_history(agent, turns=24, filler=200)

    agent.ask("next")

    events = _trace_events(agent, "context_compacted")
    assert events, "这份历史应该把 8k 档的预算顶过触发点"
    assert {event["trigger"] for event in events} == {"auto"}
    assert events[0]["newly_covered"] > 0
    assert events[0]["covered_entries"] > 0


def test_a_compaction_before_any_run_writes_no_trace_instead_of_opening_one(tmp_path):
    """一轮都没跑过就 /compact:压照压,但不为这条事件凭空开一个 run。

    开一个没有任何模型调用的 run 会把所有「按 run 求平均」的指标算歪,而那批指标
    正是这套评测体系的分母。
    """
    agent = build_agent(tmp_path, [])
    _long_history(agent, turns=24, filler=200)

    report = agent.compact_context()

    assert report["compacted"] is True
    runs = Path(agent.workspace.repo_root) / ".codingforme" / "runs"
    assert not runs.exists() or not list(runs.rglob("trace.jsonl"))


def test_context_status_shows_how_close_the_session_is_to_auto_compaction(tmp_path):
    """`/context` 要说出现在占了多少、系统会在哪个点自己动手。

    整个上下文治理里唯一给用户的手是 `/compact`,而在这之前它是**盲的**——看得到
    预算总额,看不到自己此刻在哪、也看不到触发点在哪,于是无从判断现在该不该压。
    """
    from codingforme.cli import _context_status

    agent = build_agent(tmp_path, [models.final_answer("ok")])
    _long_history(agent, turns=6, filler=40)
    agent.ask("next")

    text = _context_status(agent)

    pressure = [line for line in text.splitlines() if line.startswith("pressure")]
    assert pressure and "as of the last turn" in pressure[0], text
    used = int(agent.last_prompt_metadata["prompt_tokens"])
    assert f"{used:,}" in pressure[0]
    thresholds = agent.context_manager.compression_thresholds()
    assert f"{thresholds['trigger_tokens']:,}" in text
    assert f"{thresholds['target_tokens']:,}" in text
    # 摘要覆盖到哪儿,是「再敲一次 /compact 还有没有用」的唯一依据。
    assert "transcript entries covered" in text


def test_context_status_says_it_has_no_reading_yet_instead_of_showing_zero(tmp_path):
    """一轮都没跑过时明说「还没有」,不显示 0%。

    「占用率是零」和「还没量过」在同一个数字上分不出来,而后者根本不该拿来做决策。
    """
    from codingforme.cli import _context_status

    agent = build_agent(tmp_path, [])

    text = _context_status(agent)

    assert "no turn has been assembled yet" in text
    assert "0.0%" not in text


def test_context_status_says_so_when_graded_compression_is_switched_off(tmp_path):
    """分级关掉时触发点等于预算,这件事要说出来——否则看起来像显示错了。"""
    from codingforme.cli import _context_status

    agent = build_agent(tmp_path, [], feature_flags={"graded_compression": False})

    text = _context_status(agent)

    assert "graded compression is OFF" in text
    assert "compress down to this" not in text
