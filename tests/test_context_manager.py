from codingforme import FakeModelClient, CodingForMe, SessionStore, WorkspaceContext
from codingforme.context_manager import ContextManager


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
        total_budget=700,
        section_budgets={
            "prefix": 120,
            "memory": 120,
            "relevant_memory": 120,
            "history": 400,
        },
    )

    prompt, metadata = manager.build("keep this request verbatim")

    for section in ("prefix", "memory", "relevant_memory", "history"):
        assert metadata["sections"][section]["rendered_chars"] <= metadata["sections"][section]["budget_chars"]

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
        total_budget=250,
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
        total_budget=250,
        section_budgets={
            "prefix": 80,
            "memory": 80,
            "relevant_memory": 80,
            "history": 80,
        },
    ).build(request)

    assert prompt.split("Current user request:\n", 1)[1] == request
    assert metadata["current_request"]["text"] == request
    assert metadata["current_request"]["rendered_chars"] == len(request)


def test_context_manager_head_tail_clips_current_request_when_it_alone_overflows_budget(tmp_path):
    agent = build_agent(tmp_path, [])

    huge_request = "START-MARKER " + ("X" * 2000) + " END-MARKER"
    prompt, metadata = ContextManager(
        agent,
        total_budget=250,
        section_budgets={
            "prefix": 60,
            "memory": 60,
            "relevant_memory": 60,
            "history": 60,
        },
    ).build(huge_request)

    assert len(prompt) <= 250
    assert metadata["current_request"]["truncated"] is True
    assert metadata["current_request"]["dropped_chars"] > 0
    assert metadata["prompt_over_budget"] is False
    rendered_request = prompt.split("Current user request:\n", 1)[1]
    assert rendered_request.startswith("START-MARKER")
    assert "END-MARKER" in rendered_request
    assert "中间已省略" in rendered_request
    assert "请提示用户缩短请求或分步描述" in rendered_request


def test_context_manager_leaves_current_request_untruncated_when_it_fits(tmp_path):
    agent = build_agent(tmp_path, [])

    prompt, metadata = ContextManager(agent).build("a short request that fits easily")

    assert metadata["current_request"]["truncated"] is False
    assert metadata["current_request"]["dropped_chars"] == 0
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

    # 12 条而不是 6 条：最近窗口的条目上限是 2 * 6 个工具轮 = 12 条
    # （一个工具轮现在会留下模型说明 + 工具结果两条记录）。要把上面那两次读
    # 挤出窗口，就得垫够 12 条。
    for minute in range(2, 14):
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

    # 同上：垫满 12 条才能把这次 run_shell 挤出最近窗口。
    for minute in range(1, 13):
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
        total_budget=900,
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

    manager = ContextManager(agent, total_budget=40000)
    messages, _prompt, metadata = manager.build_all("go")

    assert metadata["sections"]["history"]["omitted_entry_count"] == 0
    assert metadata["message_layout"]["omitted_digest_present"] is False
    assert all("Omitted context:" not in str(m["content"]) for m in messages)
