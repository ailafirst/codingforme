"""P0 验收：三级身份落盘、TraceIndex 保真、HarnessSpec 装配、统一结果 schema。"""

import json

import pytest

from codingforme.eval import harness as harness_module
from codingforme.eval.code_signature import (
    MODEL_FACING_MODULES,
    SOURCE_UNAVAILABLE,
    module_signatures,
    normalized_source,
)
from codingforme.eval.compat import legacy_run_aggregate
from codingforme.eval.harness import BUILTIN_HARNESS_SPECS, DEFAULT_HARNESS, HarnessSpec, get_harness
from codingforme.eval.report import (
    AXIS_CAPABILITY,
    LEVEL_TASK,
    aggregate_cases,
    build_case_result,
    build_eval_result,
    render_eval_result_markdown,
    write_eval_result,
)
from codingforme.eval.economy import render_economy_markdown, summarize_economy
from codingforme.eval.trace import RunRecord, TraceIndex, TurnRecord
from codingforme import tools as toolkit
from codingforme.metrics import aggregate_run_artifacts
from codingforme.models import FakeModelClient, final_answer, tool_call
from codingforme.runtime import SessionStore
from codingforme import runtime


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    (tmp_path / "sample.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    return tmp_path


def build_agent(tmp_path, outputs, harness=None, session=None, session_store=None):
    harness = harness or DEFAULT_HARNESS
    return harness.build(
        FakeModelClient(list(outputs)),
        tmp_path,
        session=session,
        session_store=session_store,
    )


# --- 三级身份：session -> run -> turn ---------------------------------------


def test_trace_events_carry_session_run_and_turn(tmp_path):
    build_workspace(tmp_path)
    agent = build_agent(
        tmp_path,
        [tool_call("read_file", path="README.md", start=1, end=2), final_answer("done")],
    )

    agent.ask("read the readme")

    trace_path = agent.run_store.trace_path(agent.current_task_state)
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert events, "trace should not be empty"
    for event in events:
        assert event["session_id"] == agent.session["id"]
        assert event["run_id"] == agent.current_task_state.run_id
        assert event["run_seq"] == 1
        assert "turn" in event

    # run_started 发生在第一轮之前，turn 为 0；之后每轮递增。
    assert next(event for event in events if event["event"] == "run_started")["turn"] == 0
    prompt_turns = [event["turn"] for event in events if event["event"] == "prompt_built"]
    assert prompt_turns == [1, 2]


def test_run_seq_increments_within_a_session_and_survives_resume(tmp_path):
    build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".codingforme" / "sessions")
    agent = build_agent(tmp_path, [final_answer("one"), final_answer("two")], session_store=store)
    agent.ask("first")
    session_id = agent.session["id"]
    assert agent.current_task_state.run_seq == 1

    agent.ask("second")
    assert agent.current_task_state.run_seq == 2

    # 换一个进程/agent 实例接着同一个 session 跑，序号必须接着涨而不是归零。
    resumed = build_agent(
        tmp_path,
        [final_answer("three")],
        session=store.load(session_id),
        session_store=store,
    )
    resumed.ask("third")
    assert resumed.current_task_state.run_seq == 3
    assert resumed.session["id"] == session_id


def test_report_and_task_state_carry_the_join_key(tmp_path):
    build_workspace(tmp_path)
    agent = build_agent(tmp_path, [final_answer("done")])
    agent.ask("hello")

    report = agent.run_store.load_report(agent.current_task_state.run_id)
    task_state = agent.run_store.load_task_state(agent.current_task_state)

    assert report["session_id"] == agent.session["id"]
    assert report["run_seq"] == 1
    assert task_state["session_id"] == agent.session["id"]
    assert task_state["run_seq"] == 1


# --- TraceIndex -------------------------------------------------------------


def test_trace_index_rebuilds_session_run_turn_hierarchy(tmp_path):
    build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".codingforme" / "sessions")
    agent = build_agent(
        tmp_path,
        [tool_call("read_file", path="README.md", start=1, end=2), final_answer("first")],
        session_store=store,
    )
    agent.ask("first request")
    agent.model_client = FakeModelClient([final_answer("second")])
    agent.ask("second request")

    index = TraceIndex.load(
        tmp_path / ".codingforme" / "runs",
        tmp_path / ".codingforme" / "sessions",
    )

    assert len(index.sessions) == 1
    session = index.sessions[0]
    assert session.session_id == agent.session["id"]
    assert session.run_count == 2
    assert [run.run_seq for run in session.runs] == [1, 2]
    assert not session.synthetic
    # 会话状态被一并读进来，跨 session 的分析可以直接拿到 history/memory
    assert session.session_state.get("id") == agent.session["id"]

    first_run = session.runs[0]
    assert len(first_run.turns) == 2
    tool_turn = first_run.turns[0]
    assert tool_turn.kind == "tool"
    assert tool_turn.tool_name == "read_file"
    assert tool_turn.prompt_metadata, "prompt_built 的元数据应挂在同一轮上"
    assert tool_turn.prompt_cache_key
    assert first_run.turns[1].kind == "final"

    coverage = index.coverage()
    assert coverage["run_count"] == 2
    assert coverage["session_count"] == 1
    assert coverage["turn_count"] == 3
    assert coverage["runs_with_session_id"] == 2
    assert coverage["synthetic_sessions"] == 0
    assert coverage["multi_run_sessions"] == 1


def test_trace_index_pairs_events_by_turn_not_by_file_order(tmp_path):
    build_workspace(tmp_path)
    agent = build_agent(
        tmp_path,
        [
            tool_call("read_file", path="README.md", start=1, end=2),
            tool_call("read_file", path="sample.txt", start=1, end=2),
            final_answer("done"),
        ],
    )
    agent.ask("read both files")

    index = TraceIndex.load(tmp_path / ".codingforme" / "runs")
    run = index.runs[0]

    assert [turn.turn for turn in run.turns] == [1, 2, 3]
    assert [turn.tool_name for turn in run.turns] == ["read_file", "read_file", ""]
    # 每一轮的工具事件都落在自己那一轮上，没有串轮
    assert run.turns[0].tool["args"]["path"] == "README.md"
    assert run.turns[1].tool["args"]["path"] == "sample.txt"
    assert run.turns[2].tool is None


def test_trace_index_reads_legacy_artifacts_without_identity_fields(tmp_path):
    """老工件没有 session_id/run_seq/turn，索引要能靠事件次序补齐。"""
    build_workspace(tmp_path)
    agent = build_agent(
        tmp_path,
        [tool_call("read_file", path="README.md", start=1, end=2), final_answer("done")],
    )
    agent.ask("read it")

    runs_root = tmp_path / ".codingforme" / "runs"
    run_dir = next(path for path in runs_root.iterdir() if path.is_dir())
    trace_path = run_dir / "trace.jsonl"
    legacy_events = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        for key in ("session_id", "run_id", "run_seq", "turn"):
            event.pop(key, None)
        legacy_events.append(event)
    trace_path.write_text("\n".join(json.dumps(event) for event in legacy_events) + "\n", encoding="utf-8")
    for name in ("task_state.json", "report.json"):
        payload = json.loads((run_dir / name).read_text(encoding="utf-8"))
        payload.pop("session_id", None)
        payload.pop("run_seq", None)
        (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    index = TraceIndex.load(runs_root)

    assert len(index.runs) == 1
    run = index.runs[0]
    assert run.synthetic_session is True
    assert run.session_id.startswith("unknown:")
    # turn 靠 prompt_built 的出现次序精确还原
    assert [turn.turn for turn in run.turns] == [1, 2]
    assert run.turns[0].tool_name == "read_file"
    assert index.coverage()["runs_with_session_id"] == 0
    assert index.coverage()["synthetic_sessions"] == 1


def test_trace_index_survives_a_truncated_trace_line(tmp_path):
    build_workspace(tmp_path)
    agent = build_agent(tmp_path, [final_answer("done")])
    agent.ask("hello")

    runs_root = tmp_path / ".codingforme" / "runs"
    trace_path = next(runs_root.iterdir()) / "trace.jsonl"
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "run_fin')

    index = TraceIndex.load(runs_root)
    assert len(index.runs) == 1
    assert index.coverage()["turn_count"] == 1


def test_trace_index_merges_independent_workspaces(tmp_path):
    indexes = []
    for name in ("alpha", "beta"):
        root = tmp_path / name
        root.mkdir()
        build_workspace(root)
        agent = build_agent(root, [final_answer("done")])
        agent.ask(f"hello {name}")
        indexes.append(
            TraceIndex.load(root / ".codingforme" / "runs", root / ".codingforme" / "sessions")
        )

    merged = TraceIndex.merge(indexes)
    assert merged.coverage()["run_count"] == 2
    assert merged.coverage()["session_count"] == 2


# --- 保真度：新底座必须和旧口径逐字段一致 ------------------------------------


def test_legacy_aggregate_is_reproduced_exactly_from_the_index(tmp_path):
    """P0 只换底座、不动数字：同一批工件上两条路径的输出必须完全相同。"""
    build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".codingforme" / "sessions")
    agent = build_agent(
        tmp_path,
        [
            tool_call("read_file", path="README.md", start=1, end=2),
            tool_call("read_file", path="../escape.txt", start=1, end=2),
            final_answer("done"),
        ],
        session_store=store,
    )
    agent.ask("first")
    agent.model_client = FakeModelClient(
        [tool_call("list_files", path="."), final_answer("second")]
    )
    agent.ask("second")

    runs_root = tmp_path / ".codingforme" / "runs"
    legacy = aggregate_run_artifacts(runs_root)
    rebuilt = legacy_run_aggregate(TraceIndex.load(runs_root))

    assert rebuilt == legacy
    # 顺带确认这批工件确实有内容可比，不是两个空 dict 相等
    assert legacy["run_count"] == 2
    assert legacy["tool_name_counts"]["read_file"] == 2
    assert legacy["security_event_counts"], "路径逃逸应当留下安全事件"


# --- HarnessSpec ------------------------------------------------------------


def test_default_harness_matches_the_previously_inlined_assembly(tmp_path):
    build_workspace(tmp_path)
    agent = DEFAULT_HARNESS.build(FakeModelClient([]), tmp_path)

    assert agent.approval_policy == "auto"
    assert agent.read_only is False
    assert agent.max_steps == 6
    assert agent.max_depth == 1
    assert agent.feature_flags == {
        "memory": True,
        "relevant_memory": True,
        "context_reduction": True,
        "prompt_cache": True,
        # 受限编排默认关：开着它跑出来的数据和关着跑出来的不可比，
        # 所以它是一个要显式打开的变体（`plan_tool`），不是默认能力。
        "plan_tool": False,
        "delegate_tool": False,
        # 这七个是消融开关，默认开——关掉才是变体（`no_tool_output_spill` /
        # `no_window_block` / `no_graded_compression` / `no_session_summary` /
        # `no_stale_read` / `no_clear_at_least` / `no_reversible_squeeze`）。
        # 写在这里是为了「默认变体等于机制全开」这句话有一处会失败的断言兜着。
        "tool_output_spill": True,
        "recent_window_block": True,
        "graded_compression": True,
        "session_summary": True,
        "stale_read_invalidation": True,
        "clear_at_least": True,
        "reversible_squeeze": True,
    }
    assert "run_plan" not in agent.tools


def test_harness_fingerprint_changes_with_behaviour_not_with_description():
    base = get_harness("full")
    renamed = base.derive(description="换个说明文字")
    changed = base.derive(feature_flags={"memory": False})

    assert renamed.fingerprint() == base.fingerprint()
    assert changed.fingerprint() != base.fingerprint()


def test_harness_spec_round_trips_through_dict():
    spec = HarnessSpec(
        name="probe",
        approval_policy="never",
        read_only=True,
        feature_flags={"memory": False},
        tools_allowlist=("read_file",),
        total_budget=900,
        section_budgets={"history": 100},
    )

    restored = HarnessSpec.from_dict(json.loads(json.dumps(spec.to_dict())))

    assert restored.fingerprint() == spec.fingerprint()
    assert restored.tools_allowlist == ("read_file",)
    assert restored.total_budget == 900


def test_harness_variants_apply_their_configuration(tmp_path):
    build_workspace(tmp_path)

    read_only = get_harness("read_only").build(FakeModelClient([]), tmp_path)
    assert read_only.read_only is True

    no_memory = get_harness("no_memory").build(FakeModelClient([]), tmp_path)
    assert no_memory.feature_flags["memory"] is False
    assert no_memory.feature_flags["relevant_memory"] is False

    strict = get_harness("strict_approval").build(FakeModelClient([]), tmp_path)
    assert strict.approval_policy == "never"


def test_harness_tools_allowlist_prunes_registry_and_rebuilds_prefix(tmp_path):
    build_workspace(tmp_path)
    full = DEFAULT_HARNESS.build(FakeModelClient([]), tmp_path)
    pruned = DEFAULT_HARNESS.derive(tools_allowlist=("read_file",)).build(FakeModelClient([]), tmp_path)

    assert set(pruned.tools) == {"read_file"}
    # 工具集是 prefix 的一部分，裁剪后 prefix hash 和 tool_signature 都必须跟着变，
    # 否则发给模型的工具列表会和实际注册表对不上。
    assert pruned.prefix_state.hash != full.prefix_state.hash
    assert pruned.prefix_state.tool_signature != full.prefix_state.tool_signature
    assert [spec["function"]["name"] for spec in toolkit.to_openai_function_specs(pruned.tools)] == ["read_file"]


def test_harness_rejects_an_unknown_tool_in_the_allowlist(tmp_path):
    build_workspace(tmp_path)
    spec = DEFAULT_HARNESS.derive(tools_allowlist=("read_file", "teleport"))

    with pytest.raises(ValueError, match="teleport"):
        spec.build(FakeModelClient([]), tmp_path)


def test_harness_applies_context_budget_overrides(tmp_path):
    build_workspace(tmp_path)
    spec = DEFAULT_HARNESS.derive(total_budget=900, section_budgets={"history": 120})
    agent = spec.build(FakeModelClient([]), tmp_path)

    assert agent.context_manager.total_budget == 900
    assert agent.context_manager.section_budgets["history"] == 120


def test_a_window_tier_variant_moves_the_whole_derivation_chain(tmp_path):
    """`context_window` 走 `set_context_window()`，不是只改那个总数。

    和 `total_budget` 的区别就在这里：直接写 `total_budget` 会让
    `context_window_tokens` / `context_budget_breakdown` 停留在探测出来的旧值，
    工件上两者对不上；而单条工具结果上限、计划转录上限都是从预算派生的，
    降档如果不带着它们一起走，等于只换了闸门、没换任何一道入口上限。
    """
    build_workspace(tmp_path)

    small = get_harness("window_16k").build(FakeModelClient([]), tmp_path)
    large = get_harness("window_32k").build(FakeModelClient([]), tmp_path)

    assert (small.context_window, large.context_window) == (16_000, 32_000)
    # 预算、单条上限、breakdown 三者必须同时跟着走
    assert small.context_manager.total_budget < large.context_manager.total_budget
    assert small.tool_output_limit() < large.tool_output_limit()
    assert small.context_budget_breakdown["window_tokens"] == 16_000
    assert large.context_budget_breakdown["window_tokens"] == 32_000


def test_builtin_variants_have_distinct_fingerprints():
    fingerprints = {name: spec.fingerprint() for name, spec in BUILTIN_HARNESS_SPECS.items()}
    assert len(set(fingerprints.values())) == len(fingerprints)


# --- 统一结果 schema --------------------------------------------------------


def test_build_eval_result_has_the_six_top_level_blocks():
    cases = [
        build_case_result("task_a", LEVEL_TASK, True, axis_values={AXIS_CAPABILITY: 1.0}),
        build_case_result("task_b", LEVEL_TASK, False, axis_values={AXIS_CAPABILITY: 0.0}),
    ]

    result = build_eval_result(
        suite="unit",
        harness=DEFAULT_HARNESS,
        dataset={"source": "unit", "task_count": 2},
        cases=cases,
        run_context={"mode": "scripted"},
        trace_summary={"coverage": {"run_count": 2}},
    )

    assert set(result) >= {
        "schema_version",
        "suite",
        "run_context",
        "harness",
        "dataset",
        "cases",
        "aggregates",
        "trace",
    }
    assert result["harness"]["fingerprint"] == DEFAULT_HARNESS.fingerprint()
    assert result["aggregates"]["pass_rate"] == 0.5
    assert result["aggregates"]["by_level"][LEVEL_TASK]["total"] == 2


def test_case_result_rejects_unknown_levels_and_axes():
    with pytest.raises(ValueError, match="unknown eval level"):
        build_case_result("x", "L9-nope", True)
    with pytest.raises(ValueError, match="unknown eval axes"):
        build_case_result("x", LEVEL_TASK, True, axis_values={"vibes": 1.0})


def test_aggregate_cases_groups_by_level():
    cases = [
        build_case_result("a", LEVEL_TASK, True),
        build_case_result("b", LEVEL_TASK, False),
        build_case_result("c", "L1-trajectory", True),
    ]

    aggregates = aggregate_cases(cases)

    assert aggregates["total"] == 3
    assert aggregates["by_level"][LEVEL_TASK]["pass_rate"] == 0.5
    assert aggregates["by_level"]["L1-trajectory"]["pass_rate"] == 1.0


def test_eval_result_writes_and_renders(tmp_path):
    result = build_eval_result(
        suite="unit",
        harness=DEFAULT_HARNESS,
        dataset={"source": "unit", "task_count": 1},
        cases=[build_case_result("a", LEVEL_TASK, True)],
        trace_summary={"coverage": {"run_count": 1, "session_count": 1, "turn_count": 2, "runs_with_session_id": 1}},
        notes=["仅底座"],
    )
    path = write_eval_result(tmp_path / "nested" / "eval.json", result)

    assert json.loads(path.read_text(encoding="utf-8"))["suite"] == "unit"

    rendered = render_eval_result_markdown(result)
    assert "评测结果：unit" in rendered
    assert "L2-task" in rendered
    assert "仅底座" in rendered


def test_harness_fingerprint_covers_the_prompt_template(monkeypatch):
    """改提示词必须改指纹。

    此前 `fingerprint()` 只吃配置字段，提示词文本不是配置字段。实测后果：本仓库
    所有跑批的指纹恒为 `sha256:1e0dcb0f0a19`——期间提示词改过四轮（加规则、加
    few-shot 示范、全部撤回、修 dedent）、模型输出协议改过一轮，指纹一次没变。
    而这个字段在结果 schema 里的定义正是「用来判定两次结果是否可比」。
    """
    base = get_harness("full").fingerprint()
    monkeypatch.setattr(runtime, "PROMPT_TEMPLATE", runtime.PROMPT_TEMPLATE + "\n- One more rule.")

    assert get_harness("full").fingerprint() != base


def test_harness_fingerprint_covers_the_tool_schema(monkeypatch):
    """改工具定义必须改指纹。

    和提示词同理：工具 schema 决定模型看到什么、能发什么，却不是配置字段。
    缺口一会改它，改完之后新旧两份数据必须能在工件层面区分开。
    """
    base = get_harness("full").fingerprint()
    patched = dict(toolkit.BASE_TOOL_SPECS)
    patched["read_file"] = {
        **patched["read_file"],
        "description": patched["read_file"]["description"] + " Paths are workspace-relative.",
    }
    monkeypatch.setattr(toolkit, "BASE_TOOL_SPECS", patched)

    assert get_harness("full").fingerprint() != base


def test_code_signature_is_independent_of_configuration():
    """代码签名只反映代码，不反映配置。

    两者分开报，指纹变了的时候才能立刻回答「变的是配置还是代码」。
    """
    variants = [spec.code_signature() for spec in BUILTIN_HARNESS_SPECS.values()]

    assert len(set(variants)) == 1, "不同配置变体的代码签名应当相同"
    assert len({spec.fingerprint() for spec in BUILTIN_HARNESS_SPECS.values()}) == len(variants)


def test_code_signature_covers_runtime_logic_not_just_text_constants(monkeypatch):
    """代码签名必须覆盖**运行时逻辑**，不能只覆盖提示词文本和工具 schema。

    证据是踩过的坑：T2-1（把上下文从「压平成一段文本塞进单条 user message」换成
    标准 messages 数组）改动前后各跑了一次 12 任务 × 3 轮的正式跑批，两份工件里
    的 `code_signature` 完全相同（`sha256:db7c156bcbee7`）。而那个改动让走到终点
    率从 76% 变成 97%——这个字段在结果 schema 里的定义正是「用来判定两次结果是否
    可比」，它当时给的是错误答案。
    """
    base = get_harness("full").code_signature()
    monkeypatch.setattr(harness_module, "model_facing_code_signature", lambda: "pretend-assembly-changed")

    assert get_harness("full").code_signature() != base


def test_the_code_signature_ignores_comments_and_docstrings():
    """只有真代码变才算变。

    这个仓库里 docstring 常常比实现长几倍，而且改得最勤。如果写一段说明就让签名
    变，签名很快就没人信了——每份工件都显示「和上一次不可比」，等于没有信号。
    """
    with_docs = 'def f(x):\n    """一段说明。"""\n    # 一条注释\n    return x + 1\n'
    other_docs = 'def f(x):\n    """换成完全不同的一段说明。"""\n    return x + 1\n'
    changed_code = 'def f(x):\n    return x + 2\n'

    assert normalized_source(with_docs) == normalized_source(other_docs)
    assert normalized_source(with_docs) != normalized_source(changed_code)


def test_every_signed_module_actually_resolves():
    """签名清单里的模块名必须都能拿到源码。

    拼错一个模块名不会报错，只会让那一项静默退化成 `<source-unavailable>` 常量——
    结果是这个模块怎么改签名都不变，而报告上看不出任何异常。
    """
    digests = module_signatures()

    assert set(digests) == set(MODEL_FACING_MODULES)
    assert SOURCE_UNAVAILABLE not in digests.values()
    assert "codingforme.context_manager" in digests, "上下文组装是最需要被签进去的一块"


def test_economy_summary_reports_token_usage_from_the_trace():
    """economy 轴的数据一直在 trace 里，此前从没进过报告。

    k=3 基线上有 337 次带 usage 的模型调用、合计 73.8 万 input token，报告却
    渲染成「未覆盖」，同一份报告的备注还写着「economy 轴有数」，自相矛盾。
    """
    index = _index_with_usage(
        [
            {"input_tokens": 1000, "output_tokens": 100, "reasoning_tokens": 80, "cached_tokens": 750},
            {"input_tokens": 3000, "output_tokens": 300, "reasoning_tokens": 240, "cached_tokens": 2250},
        ]
    )
    summary = summarize_economy(index)

    assert summary["turns_with_usage"] == 2
    assert summary["input_tokens"] == 4000
    assert summary["output_tokens"] == 400
    # 思维链计入 output，不是额外的一份。
    assert summary["reasoning_share_of_output"] == pytest.approx(0.8)
    assert summary["cached_share_of_input"] == pytest.approx(0.75)
    assert summary["input_tokens_per_turn"] == pytest.approx(2000.0)

    rendered = render_economy_markdown(summary)
    assert "80.0%" in rendered and "75.0%" in rendered
    assert "未覆盖" not in rendered


def test_economy_says_why_it_is_empty_under_oracle_replay():
    """没有数据时要说明原因，而不是省略这一节。

    「未覆盖」和「回放模式本来就不产生 usage」是两件事，读者要能分辨。
    """
    rendered = render_economy_markdown(summarize_economy(_index_with_usage([])))

    assert "无数据" in rendered
    assert "FakeModelClient" in rendered


def _index_with_usage(metadatas):
    """造一个只带 completion_metadata 的最小索引。"""
    index = TraceIndex([], [])
    run = RunRecord(run_dir=None, run_id="run_x", session_id="s", run_seq=1)
    for turn_no, metadata in enumerate(metadatas, start=1):
        record = TurnRecord(session_id="s", run_id="run_x", run_seq=1, turn=turn_no)
        record.completion_metadata = dict(metadata)
        run.turns.append(record)
    index.runs.append(run)
    return index
