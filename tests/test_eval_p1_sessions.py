"""P1-② 验收：L3 跨会话断言、进程重启后的连续性、数据集契约。

这一层最容易写成「怎么跑都通过」的假测试，所以用例的重心在**证伪**：
关掉记忆之后必须恰好是那几条召回类断言挂掉，其余照常通过。
只有当断言在该失败的时候真的失败，12/12 才有意义。
"""

import json
from pathlib import Path

import pytest

from codingforme.eval.harness import DEFAULT_HARNESS, get_harness
from codingforme.eval.report import LEVEL_SESSION
from codingforme.eval.scorers import rendered_notes, score_session, session_cases
from codingforme.eval.session_suite import (
    DEFAULT_SESSION_BENCHMARK_PATH,
    expectations_for,
    load_session_benchmark,
    run_session_suite,
    run_session_task,
    scripted_outputs,
)
from codingforme.eval.trace import TurnRecord

QUESTION_TYPES = {
    "single_session_recall",
    "cross_session_aggregation",
    "knowledge_update",
    "temporal_reasoning",
}


def failures(result):
    return {case["case_id"]: case["detail"] for case in result["cases"] if not case["passed"]}


def task_by_id(benchmark, task_id):
    return next(task for task in benchmark["tasks"] if task["id"] == task_id)


# --- 数据集契约 --------------------------------------------------------------


def test_session_benchmark_covers_the_four_question_types():
    """四种 LongMemEval 题型必须齐全；压力探针是另一类，不占它们的名额。

    `context_pressure` 不是记忆题型——它测的是「上下文压力有没有被压下去」，
    加进 QUESTION_TYPES 会让「四种题型齐不齐」这个契约变松。所以这里查的是
    **包含**四种，而不是恰好等于四种；「每个任务都得有断言」那条对所有任务照旧。
    """
    benchmark = load_session_benchmark(DEFAULT_SESSION_BENCHMARK_PATH)

    assert benchmark["schema_version"] == 1
    assert QUESTION_TYPES <= {task["question_type"] for task in benchmark["tasks"]}
    for task in benchmark["tasks"]:
        assert len(task["turns"]) >= 2, "跨会话任务至少要两轮，否则测的还是单轮"
        assert any(turn.get("expect") for turn in task["turns"]), f"{task['id']} 没有任何断言"


def test_expectations_are_keyed_by_run_seq():
    benchmark = load_session_benchmark(DEFAULT_SESSION_BENCHMARK_PATH)
    task = task_by_id(benchmark, "recall_release_tag_across_restarts")

    expectations = expectations_for(task)

    # 第 2、3 轮标了重启；第 3 轮同时带召回期望。
    assert sorted(expectations) == [2, 3]
    assert expectations[2]["restart_before"] is True
    assert expectations[3]["notes_contain"] == ["v0.2.0"]


def test_scripted_outputs_rejects_shapes_it_does_not_understand():
    assert scripted_outputs({"outputs": [{"final": "ok"}]})[0]["text"] == "ok"
    assert scripted_outputs({"outputs": [{"tool": "read_file", "args": {"path": "a"}}]})[0]["tool_calls"]

    with pytest.raises(ValueError):
        scripted_outputs({"outputs": [{"speak": "hi"}]})
    with pytest.raises(ValueError):
        scripted_outputs({"outputs": []})


def test_unsupported_schema_version_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 99, "tasks": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_session_benchmark(path)


# --- 端到端：全机制开启 ------------------------------------------------------


def test_full_harness_passes_every_session_assertion(tmp_path):
    result = run_session_suite(workspace_root=tmp_path / "ws")

    assert failures(result) == {}
    assert result["aggregates"]["total"] == 14
    assert all(case["level"] == LEVEL_SESSION for case in result["cases"])
    assert len({case["case_id"] for case in result["cases"]}) == len(result["cases"])


def test_the_suite_actually_produces_multi_run_sessions(tmp_path):
    """这就是 P1-② 存在的理由：改动前 multi_run_sessions 恒为 0。"""
    result = run_session_suite(workspace_root=tmp_path / "ws")

    coverage = result["trace"]["coverage"]
    assert coverage["multi_run_sessions"] == 5
    assert coverage["session_count"] == 5
    # 13 = 四条记忆会话的轮次和；+24 = `long_dialogue_pressure` 那条压力探针
    assert coverage["run_count"] == 37
    assert coverage["synthetic_sessions"] == 0


# --- 证伪：关掉记忆之后必须挂在该挂的地方 ------------------------------------


def test_disabling_memory_fails_exactly_the_recall_assertions(tmp_path):
    """断言必须能证伪。否则 12/12 只是说明它什么都没查。"""
    full = run_session_suite(harness=get_harness("full"), workspace_root=tmp_path / "full")
    none = run_session_suite(harness=get_harness("no_memory"), workspace_root=tmp_path / "none")

    assert failures(full) == {}
    broken = failures(none)
    assert len(broken) == 4
    assert all(case_id.endswith(":session_evidence_surfaced") for case_id in broken)
    # 归因要说清「什么没带回来」，而不只是「不通过」。
    for detail in broken.values():
        offender = detail["offenders"][0]
        assert offender["missing"]
        assert offender["rendered_notes"] == []

    # 时间线与重启连续性跟记忆无关，不该被连坐。
    assert none["aggregates"]["by_axis"]["reliability"]["pass_rate"] == 1.0


# --- 逐题型 ------------------------------------------------------------------


def test_knowledge_update_supersedes_the_old_value_on_both_sides(tmp_path):
    """新值出现 + 旧值消失 + 工件里留下 supersede 记录，三者都要有。"""
    benchmark = load_session_benchmark(DEFAULT_SESSION_BENCHMARK_PATH)
    task = task_by_id(benchmark, "update_indentation_convention")

    session_id, index = run_session_task(task, tmp_path / "ws", harness=DEFAULT_HARNESS)
    session = index.session(session_id)
    verdicts = {item.assertion_id: item for item in score_session(session, expectations_for(task))}

    assert verdicts["session_evidence_surfaced"].passed
    assert verdicts["superseded_evidence_absent"].passed
    assert verdicts["supersede_recorded"].passed

    # 直接看证据本身：最后一轮的 prompt 里只有新值。
    last = session.runs[-1]
    blob = "\n".join(rendered_notes(last.first_turn))
    assert "2 spaces" in blob
    assert "4 spaces" not in blob
    # 覆盖记录落在发生覆盖的那一轮，而不是最后一轮。
    assert session.runs[2].report["durable_superseded"]


def test_restarting_the_agent_keeps_one_continuous_session(tmp_path):
    benchmark = load_session_benchmark(DEFAULT_SESSION_BENCHMARK_PATH)
    task = task_by_id(benchmark, "recall_release_tag_across_restarts")

    session_id, index = run_session_task(task, tmp_path / "ws", harness=DEFAULT_HARNESS)
    session = index.session(session_id)

    # 三次 ask() 分属三个 agent 实例，但必须是同一条会话、序号连续。
    assert [run.run_seq for run in session.runs] == [1, 2, 3]
    assert {run.session_id for run in session.runs} == {session_id}
    assert not session.synthetic

    verdicts = {item.assertion_id: item for item in score_session(session, expectations_for(task))}
    assert verdicts["continuity_survives_restart"].passed
    assert verdicts["continuity_survives_restart"].detail["restart_points"] == [2, 3]


def test_aggregation_is_selective_not_a_memory_dump(tmp_path):
    """跨会话聚合要带回相关的两条，也不能把无关的第三条一起倒进来。"""
    benchmark = load_session_benchmark(DEFAULT_SESSION_BENCHMARK_PATH)
    task = task_by_id(benchmark, "aggregate_lint_and_dependency")

    session_id, index = run_session_task(task, tmp_path / "ws", harness=DEFAULT_HARNESS)
    session = index.session(session_id)
    blob = "\n".join(rendered_notes(session.runs[-1].first_turn))

    assert "ruff check" in blob and "litellm" in blob
    assert "architecture reports" not in blob


# --- 不适用语义与判分器本身 --------------------------------------------------


def test_tasks_without_a_restart_do_not_claim_restart_coverage(tmp_path):
    benchmark = load_session_benchmark(DEFAULT_SESSION_BENCHMARK_PATH)
    task = task_by_id(benchmark, "recall_build_command")

    session_id, index = run_session_task(task, tmp_path / "ws", harness=DEFAULT_HARNESS)
    session = index.session(session_id)
    verdicts = {item.assertion_id: item for item in score_session(session, expectations_for(task))}

    # 没测过重启就不能记成通过。
    assert "continuity_survives_restart" not in verdicts
    assert "supersede_recorded" not in verdicts
    assert verdicts["session_timeline_reconstructable"].passed


def test_rendered_notes_falls_back_to_selected_notes():
    """老工件可能只有 selected_notes；不能因此把召回读成空。"""
    turn = TurnRecord(
        session_id="s",
        run_id="r",
        run_seq=1,
        turn=1,
        prompt_metadata={"relevant_memory": {"selected_notes": ["a fact"]}},
    )

    assert rendered_notes(turn) == ["a fact"]
    assert rendered_notes(TurnRecord(session_id="s", run_id="r", run_seq=1, turn=1)) == []


def test_session_cases_are_stable_across_recomputation(tmp_path):
    benchmark = load_session_benchmark(DEFAULT_SESSION_BENCHMARK_PATH)
    task = task_by_id(benchmark, "recall_build_command")
    session_id, index = run_session_task(task, tmp_path / "ws", harness=DEFAULT_HARNESS)
    session = index.session(session_id)
    expectations = expectations_for(task)

    first = session_cases(session, expectations, case_prefix=task["id"])
    second = session_cases(session, expectations, case_prefix=task["id"])

    assert first == second
    assert [case["case_id"] for case in first] == [
        "recall_build_command:session_evidence_surfaced",
        "recall_build_command:session_timeline_reconstructable",
    ]


def test_benchmark_path_default_points_at_the_repo_dataset():
    assert Path(DEFAULT_SESSION_BENCHMARK_PATH).name == "session_tasks.json"
    assert load_session_benchmark(DEFAULT_SESSION_BENCHMARK_PATH)["tasks"]


def test_the_pressure_probe_can_fail_when_nothing_is_compressed(tmp_path):
    """证伪：把窗口放大到压力够不着，这条断言必须挂。

    没有这条，`context_pressure_absorbed` 通过只说明它什么都没查——而这正是这套
    上下文工程踩过两轮的坑：机制「实现了、有测试、真实跑批里一次都不执行」，
    在报告里长得和「没问题」一模一样。
    """
    tight = run_session_suite(workspace_root=tmp_path / "tight")
    loose = run_session_suite(harness=get_harness("window_32k"), workspace_root=tmp_path / "loose")

    def pressure_case(result):
        return next(
            case for case in result["cases"]
            if case["detail"].get("assertion_id") == "context_pressure_absorbed"
        )

    assert pressure_case(tight)["passed"] is True
    loose_case = pressure_case(loose)
    assert loose_case["passed"] is False
    assert loose_case["detail"]["reduced_turns"] == 0
