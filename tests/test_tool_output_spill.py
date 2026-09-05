"""阶段三 L1:超长工具结果落盘,以及掉出最近窗口之后的可恢复占位。

对齐的是 Claude Code 的 Tool Result Budget(单条结果超上限就全文写盘、上下文里
只留预览 + 路径,模型用已有的读工具取回)和 Microcompact 的 Path A(窗口外的结果
换成占位)。这份文件锁住四件在工件上「做了」和「没做」长得一模一样的事:落盘真的
发生了、指针真的能读回全文、占位真的带着路径、错误真的没被清掉。
"""

import re

from codingforme import CodingForMe, FakeModelClient, SessionStore, WorkspaceContext
from codingforme import context_manager as cm
from codingforme.context_manager import ContextManager
from codingforme.models import count_tokens


def build_agent(tmp_path, outputs=(), **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".codingforme" / "sessions")
    return CodingForMe(
        model_client=FakeModelClient(list(outputs)),
        workspace=workspace,
        session_store=store,
        approval_policy=kwargs.pop("approval_policy", "auto"),
        **kwargs,
    )


def _oversized_file(tmp_path, name="big.txt", lines=4000):
    body = "\n".join(f"line {index} alpha beta gamma delta epsilon" for index in range(lines))
    (tmp_path / name).write_text(body, encoding="utf-8")
    return lines


def _marker_path(marker):
    return marker[len(cm.SPILL_MARKER_PREFIX) :].split(" ", 1)[0]


def test_an_oversized_tool_result_is_spilled_and_can_be_read_back(tmp_path):
    """落盘的全部意义在于「可恢复」——指针必须真的能被 read_file 打开。

    这是 Manus 那条「可恢复才可丢弃」的验收点。指针指到工作区根之外、或者带了
    反斜杠,`path()` 都会把模型挡在门外,而那样的占位比不给还糟:它看起来像
    给了出路,实际每次尝试都白烧一个往返。
    """
    agent = build_agent(tmp_path)
    lines = _oversized_file(tmp_path)
    limit = agent.tool_output_limit()

    result = agent.run_tool("read_file", {"path": "big.txt", "start": 1, "end": lines})

    marker = cm.find_spill_marker(result)
    assert marker, result[-300:]
    # 落盘之后放进上下文的那一份仍然守着单条上限。
    assert count_tokens(result) <= limit
    relative = _marker_path(marker)
    # 正斜杠、工作区相对:这串路径会被模型原样喂回 read_file。
    assert "\\" not in relative
    assert relative.startswith(".codingforme/" + cm.SPILL_DIR_NAME + "/")
    assert (tmp_path / relative).exists()
    # 落盘不截断:全文都在。
    spilled = (tmp_path / relative).read_text(encoding="utf-8")
    assert "line 0 alpha" in spilled and f"line {lines - 1} alpha" in spilled
    # 模型自己取回来:走已有的工具,不新增工具。
    recovered = agent.run_tool("read_file", {"path": relative, "start": 1, "end": 3})
    assert not recovered.startswith("error:")
    assert "line 0 alpha" in recovered


def test_a_result_that_fits_is_left_alone_and_writes_nothing(tmp_path):
    agent = build_agent(tmp_path)

    result = agent.run_tool("read_file", {"path": "README.md"})

    assert cm.find_spill_marker(result) == ""
    assert not (tmp_path / ".codingforme" / cm.SPILL_DIR_NAME).exists()
    assert agent._last_tool_result_metadata["tool_output_spilled"] is False
    assert agent._last_tool_result_metadata["tool_output_spill_path"] == ""
    assert agent._last_tool_result_metadata["tool_output_full_tokens"] == 0


def test_the_trace_metadata_says_whether_the_spill_actually_fired(tmp_path):
    """S2:「触发了」和「一次都没触发」不能在工件上长得一样。

    没有这三个字段,阶段三的收益就没有验收通道——报告里看不出这套机制是天天生效
    还是从来没跑过,而这两件事对应的下一步完全相反。
    """
    agent = build_agent(tmp_path)
    lines = _oversized_file(tmp_path)

    agent.run_tool("read_file", {"path": "big.txt", "start": 1, "end": lines})

    metadata = agent._last_tool_result_metadata
    assert metadata["tool_output_spilled"] is True
    assert metadata["tool_output_spill_path"].endswith("-read_file.txt")
    assert metadata["tool_output_full_tokens"] > agent.tool_output_limit()


def test_a_failed_spill_falls_back_to_plain_clipping(tmp_path, monkeypatch):
    """落盘写不进去时退回普通截断,不能把一次成功的工具调用变成异常。

    `run_tool()` 的契约是任何失败都返回字符串,好让模型下一轮消费这份反馈;
    落盘只是省上下文的手段,它自己失败不该改变这个契约。
    """
    agent = build_agent(tmp_path)
    lines = _oversized_file(tmp_path)
    monkeypatch.setattr(
        CodingForMe,
        "_spill_dir",
        lambda self: (_ for _ in ()).throw(OSError("read-only filesystem")),
    )

    result = agent.run_tool("read_file", {"path": "big.txt", "start": 1, "end": lines})

    assert cm.find_spill_marker(result) == ""
    assert count_tokens(result) <= agent.tool_output_limit()
    assert agent._last_tool_result_metadata["tool_output_spilled"] is False


def test_a_failed_spill_is_distinguishable_from_a_result_that_never_needed_one(tmp_path, monkeypatch):
    """落盘失败必须在工件上和「结果本来就没超上限」分得开。

    这两件事从前记出来一模一样(`tool_output_spilled: False`、两个 token 字段都是 0),
    而含义相反:一个是机制没必要跑,一个是机制该跑却跑挂了——模型手里那份结果被
    截断,而且**不可恢复**(没有落盘文件,指针也没有)。这正是这个仓库反复踩的那类坑:
    「机制生效了」和「机制一次都没触发」在工件上分不出来。
    """
    agent = build_agent(tmp_path)
    lines = _oversized_file(tmp_path)

    # 一、没超上限的调用:两个标志都是假,失败原因为空。
    agent.run_tool("read_file", {"path": "README.md", "start": 1, "end": 5})
    fine = dict(agent._last_tool_result_metadata)
    assert fine["tool_output_spilled"] is False
    assert fine["tool_output_spill_failed"] is False
    assert fine["tool_output_spill_error"] == ""
    assert fine["tool_output_full_tokens"] == 0

    # 二、超了上限但落盘挂掉:spilled 仍是假,但 failed 为真、带上原因和原始大小。
    monkeypatch.setattr(
        CodingForMe,
        "_spill_dir",
        lambda self: (_ for _ in ()).throw(OSError("read-only filesystem")),
    )
    agent.run_tool("read_file", {"path": "big.txt", "start": 1, "end": lines})
    broken = dict(agent._last_tool_result_metadata)

    assert broken["tool_output_spilled"] is False
    assert broken["tool_output_spill_failed"] is True
    assert "read-only filesystem" in broken["tool_output_spill_error"]
    # 原始大小照写:它说的是这次静默截断到底丢了多少。
    assert broken["tool_output_full_tokens"] > agent.tool_output_limit()

    # 三、正常落盘时 failed 必须是假,否则这个字段就成了噪声。
    monkeypatch.undo()
    agent.run_tool("read_file", {"path": "big.txt", "start": 1, "end": lines})
    good = dict(agent._last_tool_result_metadata)
    assert good["tool_output_spilled"] is True
    assert good["tool_output_spill_failed"] is False


def test_the_spilled_file_goes_through_redaction(tmp_path, monkeypatch):
    """落盘工件也要脱敏:它既进工作区,又会被模型原样读回来。"""
    agent = build_agent(tmp_path)
    lines = _oversized_file(tmp_path)
    monkeypatch.setattr(CodingForMe, "redact_text", lambda self, text: str(text).replace("alpha", "<redacted>"))

    result = agent.run_tool("read_file", {"path": "big.txt", "start": 1, "end": lines})

    relative = _marker_path(cm.find_spill_marker(result))
    spilled = (tmp_path / relative).read_text(encoding="utf-8")
    assert "alpha" not in spilled
    assert "<redacted>" in spilled
    assert "alpha" not in result


def _tool_item(name, args, content, call_id="c1"):
    return {"role": "tool", "name": name, "args": args, "content": content, "call_id": call_id, "executed": True}


def test_a_cleared_result_keeps_the_pointer_when_it_was_spilled(tmp_path):
    """窗口外的占位必须带路径,否则内容就是不可恢复的。

    `CLEARED_RESULT_MARKER` 只说「被清了」——对 read_file 还能靠路径重读,对
    `search` / `run_shell` 的输出就是彻底没了。
    """
    agent = build_agent(tmp_path)
    manager = ContextManager(agent)
    marker = cm.spill_marker(".codingforme/tool_outputs/run-1/001-search.txt", 9000)
    item = _tool_item("search", {"pattern": "def "}, "hit one\nhit two\n" + marker)

    line = manager._summarize_old_tool_item(item)

    assert marker in line
    assert cm.CLEARED_RESULT_MARKER not in line
    # 没落盘过的照旧是纯占位。
    plain = manager._summarize_old_tool_item(_tool_item("search", {"pattern": "def "}, "hit one\nhit two"))
    assert cm.CLEARED_RESULT_MARKER in plain


def test_a_shell_result_that_was_spilled_keeps_the_pointer_too(tmp_path):
    """run_shell 有自己的摘要分支(只留前三行),指针恒在末尾,会正好被切掉。"""
    agent = build_agent(tmp_path)
    manager = ContextManager(agent)
    marker = cm.spill_marker(".codingforme/tool_outputs/run-1/002-run_shell.txt", 40000)
    item = _tool_item("run_shell", {"command": "pytest -q"}, "a\nb\nc\nd\ne\n" + marker)

    assert marker in manager._summarize_old_tool_item(item)


def test_an_error_result_is_not_cleared_when_it_leaves_the_window(tmp_path):
    """把错误留在上下文里(Manus)。

    清掉失败的观察,等于让模型忘了自己刚踩过什么,于是把同一个调用再发一遍——
    正好撞上重复调用检测,白烧一个约 17 秒的往返。错误串本来就只有一两句话。
    """
    agent = build_agent(tmp_path)
    manager = ContextManager(agent)
    text = "error: patch_file could not find that text in src/app.py; read the file first."
    item = _tool_item("patch_file", {"path": "src/app.py"}, text)

    line = manager._summarize_old_tool_item(item)

    assert "could not find that text" in line
    assert cm.CLEARED_RESULT_MARKER not in line


def test_a_spilled_read_file_keeps_its_pointer_instead_of_the_memory_file_summary(tmp_path):
    """落盘过的 `read_file` 掉出窗口后留的是指针,不是记忆里那句文件摘要。

    这条走的是**真正的渲染入口** `_compressed_history_entries()`,而不是上面三条
    那样直接调 `_summarize_old_tool_item()`——差别是致命的:那个入口里「重用文件
    摘要」那一支曾经排在前面,并且对每一次执行成功的 `read_file` 都命中,于是指针
    那一支永远轮不到。一次 live 压力探针实测:5 次落盘、15 条掉出窗口、指针 0 条,
    模型随后开始重复读已经读过的文件、耗光步数预算。

    两条后果决定了顺序必须反过来:`read_file` 是唯一会产生大段文件内容的工具,
    它被遮蔽等于 `spilled_pointer_count` 在最该生效的地方是死的;而留下的那句摘要
    是从**落盘后的预览段**生成的,等于用一句描述开头那点内容的话代表整个文件,
    还不给任何线索说明全文在盘上。
    """
    agent = build_agent(tmp_path)
    lines = _oversized_file(tmp_path, name="huge.py", lines=4000)
    spilled = agent.run_tool("read_file", {"path": "huge.py", "start": 1, "end": lines})
    assert cm.find_spill_marker(spilled), "fixture must actually spill"

    # 把这次调用连同结果放进 history,再垫够条目把它推出最近窗口。
    agent.record({"role": "assistant", "content": "", "tool_calls": [{"id": "call-0", "name": "read_file", "args": {"path": "huge.py"}}]})
    agent.record(_tool_item("read_file", {"path": "huge.py"}, spilled, call_id="call-0"))
    # 再走一次真实的普通读取(过 run_tool,这样记忆里才有它的文件摘要),用来证明
    # 「重用文件摘要」那一支没有被这次调整误伤。
    plain = agent.run_tool("read_file", {"path": "README.md", "start": 1, "end": 5})
    agent.record({"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "name": "read_file", "args": {"path": "README.md"}}]})
    agent.record(_tool_item("read_file", {"path": "README.md"}, plain, call_id="call-1"))
    for index in range(1, cm.RECENT_TOOL_TURNS + cm.RECENT_WINDOW_BLOCK + 1):
        _record_tool_turn(agent, index, f"body of file {index}")

    messages, _, metadata = ContextManager(agent).build_all("go")
    rendered = "\n".join(str(message.get("content", "")) for message in messages)

    assert metadata["history"]["spilled_pointer_count"] == 1
    assert cm.SPILL_MARKER_PREFIX in rendered
    # 没落过盘的那些普通 read_file 仍然走文件摘要重用,这一支没有被误伤。
    assert metadata["history"]["reused_file_summary_count"] >= 1


def test_the_pointer_hands_back_a_range_that_actually_fits(tmp_path):
    """指针必须给出**照抄就能用**的取回调用,不能只给 token 数。

    这条是 live 实测逼出来的:第一版指针写 `(N tokens); use read_file on that path
    to see the rest`,模型照做了——两个压力探针里 4 次去读指针路径,4 次**又落了
    一次盘**(43,130 token → 留 14,804)。根因是单位对不上:`read_file` 的参数是
    行号,指针给的是 token 数,而模型既不知道每行多少 token 也不知道单条上限是多少。
    于是指针指向的东西永远读不完整,两个探针都因此没拿到答案。

    所以这里断言的不是文案,是**行为**:照着指针里那个 start/end 读回去,不能再落盘。
    """
    agent = build_agent(tmp_path)
    lines = _oversized_file(tmp_path, name="big.txt", lines=4000)
    marker = cm.find_spill_marker(agent.run_tool("read_file", {"path": "big.txt", "start": 1, "end": lines}))
    assert marker, "fixture must actually spill"

    relative = _marker_path(marker)
    chunk = int(re.search(r"end=(\d+)", marker).group(1))
    assert chunk > 0
    assert f"{lines} lines" in marker or f"{lines + 1} lines" in marker

    back = agent.run_tool("read_file", {"path": relative, "start": 1, "end": chunk})
    assert not back.startswith("error:"), back[:200]
    assert cm.find_spill_marker(back) == "", "跟着指针读回来的一段不能再触发一次落盘"
    assert count_tokens(back, None) <= agent.tool_output_limit()


def test_reading_a_spilled_file_does_not_number_the_lines_twice(tmp_path):
    """落盘件按原样回,不再编号。

    两个理由,第二个是硬的:一、它存的就是上一次 `read_file` 结果的逐字节副本,
    而那份已经带行号,再编一次会得到 `198: 198:...`(live 实测);二、指针里那个
    「一段读多少行」是按落盘时的每行 token 数算的,再加一层行号会把每行撑大,
    建议值当场失真、取回又会撞上单条上限。
    """
    agent = build_agent(tmp_path)
    lines = _oversized_file(tmp_path, name="big.txt", lines=4000)
    marker = cm.find_spill_marker(agent.run_tool("read_file", {"path": "big.txt", "start": 1, "end": lines}))
    relative = _marker_path(marker)

    back = agent.run_tool("read_file", {"path": relative, "start": 1, "end": 5})

    # 落盘件装的就是上一次 read_file 结果的逐字节副本,所以它自己第一行就是那份
    # 结果的 `# big.txt` 头,第二行才是带行号的正文。
    lines_back = back.splitlines()
    assert lines_back[0] == "# big.txt", lines_back[0]
    assert lines_back[1].startswith("   1: line 0 alpha"), lines_back[1]
    # 关键断言:不能出现第二层编号。
    assert not re.search(r"^\s*\d+:\s+\d+: ", back, re.M), back[:200]
    # 普通文件照旧编号,这一支没有被误伤。
    plain = agent.run_tool("read_file", {"path": "README.md", "start": 1, "end": 5})
    assert plain.startswith("# README.md\n   1: ")


def test_old_spill_directories_are_swept_but_never_the_current_run(tmp_path):
    """落盘目录只增不减会让工作区无上限地长——而取回尝试本身还会再落一次盘。

    两条边界都要锁住:超过 `SPILL_KEEP_BYTES` 时按运行目录整个删、最旧的先删;
    **本次运行的目录永远不删**,它里面的文件正被 history 里的指针引用着,删掉就把
    「可恢复」变成一句谎话。
    """
    from codingforme import runtime as runtimelib

    agent = build_agent(tmp_path)
    stale = agent.spill_root() / "run_old"
    stale.mkdir(parents=True)
    (stale / "001-search.txt").write_text("x" * 4096, encoding="utf-8")

    original = runtimelib.SPILL_KEEP_BYTES
    runtimelib.SPILL_KEEP_BYTES = 1024
    try:
        lines = _oversized_file(tmp_path, name="big.txt", lines=4000)
        marker = cm.find_spill_marker(agent.run_tool("read_file", {"path": "big.txt", "start": 1, "end": lines}))
    finally:
        runtimelib.SPILL_KEEP_BYTES = original

    assert marker
    assert not stale.exists(), "超出上限的旧运行目录应当被整个清掉"
    assert (tmp_path / _marker_path(marker)).exists(), "本次运行的落盘件不能被自己的清理删掉"


def _record_tool_turn(agent, index, content):
    call_id = f"call-{index}"
    agent.record({"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "name": "read_file", "args": {"path": f"f{index}.py"}}]})
    agent.record(_tool_item("read_file", {"path": f"f{index}.py"}, content, call_id=call_id))


def test_the_recent_window_advances_in_blocks_not_one_entry_per_turn(tmp_path):
    """最近窗口按块推进,而不是每轮往前滑一格。

    每轮滑一格意味着每轮都有一条 `tool` 消息从全文变成占位,消息数组从那个位置
    往后全变,前缀缓存**每轮**作废一次(命中率 32.7% → 77.6% 是花一整轮 A/B 换
    来的)。Claude Code 为同一件事写了服务端的 `cache_edits`,我们没有那个 API,
    只能靠让边界少动几次来近似。
    """
    agent = build_agent(tmp_path)
    turns = cm.RECENT_TOOL_TURNS + 2 * cm.RECENT_WINDOW_BLOCK + 1
    cleared_counts = []
    windows = []
    for index in range(turns):
        _record_tool_turn(agent, index, f"body of file {index}")
        details = ContextManager(agent).build_all("go")[2]["history"]
        cleared_counts.append(details["summarized_tool_count"])
        windows.append(details["recent_tool_window"])

    moves = sum(1 for before, after in zip(cleared_counts, cleared_counts[1:]) if after != before)
    # 每轮滑一格的话,边界会动 (轮数 - 窗口) 次。
    slides_if_unblocked = turns - cm.RECENT_TOOL_TURNS
    assert moves == slides_if_unblocked // cm.RECENT_WINDOW_BLOCK < slides_if_unblocked
    # 每次动都是整块地动,不是一条一条挪。
    steps = {after - before for before, after in zip(cleared_counts, cleared_counts[1:]) if after != before}
    assert steps == {cm.RECENT_WINDOW_BLOCK}
    # 窗口大小在 [base, base + block) 之间浮动,不会无限制地变胖。
    assert all(cm.RECENT_TOOL_TURNS <= window < cm.RECENT_TOOL_TURNS + cm.RECENT_WINDOW_BLOCK for window in windows)


def test_a_plan_transcript_is_not_spilled_but_its_inner_calls_are(tmp_path):
    """聚合型工具(`aggregates_calls`)自己管上限,不走落盘。

    两个理由。一、转录的额度是 `plans.transcript_limit()` = max(4000, 3 x 单条上限),
    拿单条上限去卡它等于让那个 3 倍额度当场作废——实测 6 次读文件的转录 3,993 token
    会被砍到 1,320,同一次 read_file 放进计划里反而看得更少。二、传给落盘的已经是
    裁过的那一份,写进文件的不是全文,而指针那句话说的是「full output」,是假承诺。
    真正的大输出在内层每个调用上就已经各自落过盘了。
    """
    agent = build_agent(tmp_path, feature_flags={"plan_tool": True})
    body = " ".join(["x"] * 6000)
    for index in range(6):
        (tmp_path / f"m{index}.txt").write_text(body, encoding="utf-8")
    plan = 'for p in ["m0.txt","m1.txt","m2.txt","m3.txt","m4.txt","m5.txt"]:\n    read_file(path=p)\n'

    result = agent.run_tool("run_plan", {"plan": plan})

    assert agent._last_tool_result_metadata["tool_output_spilled"] is False
    assert not list((tmp_path / ".codingforme" / cm.SPILL_DIR_NAME).rglob("*-run_plan.txt"))
    # 转录里确实有指针,但那是**内层调用**的——落盘发生在正确的那一层。
    assert cm.SPILL_MARKER_PREFIX in result
    # 转录拿到的是它自己的额度,不是单条上限。
    assert count_tokens(result) > 3 * agent.tool_output_limit() - 100
    # 内层的大输出照样各自落了盘——落盘发生在正确的那一层。
    spilled = sorted((tmp_path / ".codingforme" / cm.SPILL_DIR_NAME).rglob("*-read_file.txt"))
    assert len(spilled) == 6


def test_a_spilled_run_leaves_no_workspace_regression(tmp_path):
    """落盘写进的是工作区,而基准的隐式 P2P 要求「声明可改的文件之外逐字节一致」。

    两者只靠一个巧合对上:`_spill_dir()` 落在 `.codingforme/` 下,而
    `checks.SNAPSHOT_EXCLUDED_DIRS` 恰好排除这个目录名。把落盘目录挪出去,
    每个触发落盘的任务都会被判成「改坏了别的文件」——而当前 14 个基准任务
    一次都触发不了落盘,所以没有任何一次跑批会发现这件事。
    """
    from codingforme.eval.checks import SNAPSHOT_EXCLUDED_DIRS, workspace_regressions, workspace_snapshot

    agent = build_agent(tmp_path)
    lines = _oversized_file(tmp_path)
    before = workspace_snapshot(tmp_path)

    result = agent.run_tool("read_file", {"path": "big.txt", "start": 1, "end": lines})

    marker = cm.find_spill_marker(result)
    assert marker, result[-300:]
    relative = _marker_path(marker)
    assert (tmp_path / relative).exists()
    # 落盘目录的第一段必须是被快照排除的那个名字,否则下面这条断言不可能成立。
    assert relative.split("/")[0] in set(SNAPSHOT_EXCLUDED_DIRS)
    assert workspace_regressions(before, workspace_snapshot(tmp_path), ()) == []


def test_spilled_files_stay_out_of_list_files_and_search(tmp_path):
    """落盘产物不能出现在列目录和搜索结果里,否则会把自己搜出来再落一次盘。"""
    agent = build_agent(tmp_path)
    lines = _oversized_file(tmp_path)
    marker = cm.find_spill_marker(agent.run_tool("read_file", {"path": "big.txt", "start": 1, "end": lines}))
    assert marker
    relative = _marker_path(marker)

    listed = agent.run_tool("list_files", {"path": ".", "format": "paths"})
    assert relative not in listed
    assert cm.SPILL_DIR_NAME not in listed
    hits = agent.run_tool("search", {"query": "line 7 alpha", "format": "paths"})
    assert cm.SPILL_DIR_NAME not in hits


# ---------------------------------------------------------------- 消融开关

def test_turning_off_the_spill_flag_restores_plain_truncation(tmp_path):
    """`no_tool_output_spill` 变体必须真的退回「截断丢尾巴」，否则 A/B 两边跑的是同一份代码。

    这条锁的是消融本身的有效性，不是落盘的正确性：没有它，一次「开 vs 关」的
    对照实验可能两边都开着，而工件上看不出来——两边的数字于是必然一样，
    然后被读成「机制没有收益」。
    """
    _oversized_file(tmp_path)
    agent = build_agent(tmp_path, feature_flags={"tool_output_spill": False})
    result = agent.run_tool("read_file", {"path": "big.txt", "start": 1, "end": 4000})

    assert cm.SPILL_MARKER_PREFIX not in result
    assert not list((tmp_path / ".codingforme" / "tool_outputs").glob("**/*.txt"))
    assert count_tokens(result, None) <= agent.tool_output_limit()
    # 快路径两边必须一致：装得下的结果在关掉落盘时也一个字节都不该变。
    small = agent.run_tool("read_file", {"path": "README.md", "start": 1, "end": 5})
    assert "demo" in small


def test_turning_off_the_window_block_slides_the_boundary_every_turn(tmp_path):
    """`no_window_block` 变体必须退回「每个工具轮推一格」。

    观测通道是 `metadata["history"]["recent_tool_window"]`：按块推进时它在
    6~8 之间浮动，关掉之后恒等于 RECENT_TOOL_TURNS。没有这条，一次「开 vs 关」
    的对照可能两边跑的是同一份代码，而工件上看不出来。
    """
    seen = {}
    for blocked in (True, False):
        agent = build_agent(tmp_path, feature_flags={"recent_window_block": blocked})
        windows = set()
        for index in range(cm.RECENT_TOOL_TURNS + 2 * cm.RECENT_WINDOW_BLOCK + 1):
            _record_tool_turn(agent, index, f"body of file {index}")
            windows.add(int(ContextManager(agent).build_all("go")[2]["history"]["recent_tool_window"]))
        seen[blocked] = windows

    assert seen[False] == {cm.RECENT_TOOL_TURNS}, seen[False]
    assert seen[True] - {cm.RECENT_TOOL_TURNS}, seen[True]


def test_a_spilled_read_goes_stale_end_to_end_once_the_file_is_patched(tmp_path):
    """端到端:真落一次盘、真改一次文件,窗口外那条指针必须带上过期标记。

    上面那批 P5 用例是手搓 history 的,验的是投影层的分支。这一条验的是**两半真的
    接上了**:落盘发生在 `run_tool()` 里,「写了哪个文件」来自工作区快照的 sha256
    前后差异,两者中间隔着 history 条目的 `wrote` 字段——任何一环对不上,盘上那份
    过期快照就会被当成新鲜内容原样指给模型,而工件上看不出任何异常。
    """
    agent = build_agent(tmp_path)
    lines = _oversized_file(tmp_path)

    spilled = agent.run_tool("read_file", {"path": "big.txt", "start": 1, "end": lines})
    assert cm.find_spill_marker(spilled), "前提没成立:这次读根本没落盘"
    agent.session["history"].append(
        {"role": "tool", "name": "read_file", "args": {"path": "big.txt", "start": 1, "end": lines},
         "content": spilled, "call_id": "c0"}
    )
    # 真的改一次文件：`wrote` 要由 run_tool 自己从快照差异里算出来，不是我们填的。
    agent.run_tool("patch_file", {"path": "big.txt", "old_text": "line 0 alpha", "new_text": "line 0 ALPHA"})
    agent.session["history"].append(
        {"role": "tool", "name": "patch_file", "args": {"path": "big.txt"}, "content": "patched",
         "call_id": "c1",
         "wrote": list((agent._last_tool_result_metadata or {}).get("affected_paths") or ())}
    )
    assert agent.session["history"][-1]["wrote"] == ["big.txt"], agent.session["history"][-1]["wrote"]

    # 垫够工具轮，把这两条挤出最近窗口（窗口是 6~8 个工具轮，按块浮动）。
    for index in range(12):
        agent.session["history"].append(
            {"role": "tool", "name": "read_file", "args": {"path": "README.md"},
             "content": f"filler {index}", "call_id": f"f{index}"}
        )

    _, prompt, metadata = ContextManager(agent, total_budget=4823).build_all("next")

    assert cm.STALE_SPILL_MARKER in prompt
    assert metadata["history"]["stale_pointer_count"] == 1
    # 指针本身还在：全文仍在盘上，那是取回的唯一线索。
    line = next(row for row in prompt.split("\n") if cm.STALE_SPILL_MARKER in row)
    assert ".codingforme/" + cm.SPILL_DIR_NAME + "/" in line
