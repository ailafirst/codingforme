"""P2-b 验收：预算降级（第一步）与 pass^k / 方差（第二步）。

这批用例的重点不是「新代码跑得通」，而是**它们真的会挂**：
- 预算降级如果写反，`test_budget_still_gates_oracle_replay` 会挂；
- 可靠性聚合如果把 flaky 读成通过，`test_pass_hat_k_separates_flaky_from_stable` 会挂。
没有这两条，新增的字段只是好看的装饰。
"""

from pathlib import Path

from codingforme.eval.reliability import (
    STABILITY_FLAKY,
    STABILITY_STABLE_FAIL,
    STABILITY_STABLE_PASS,
    render_reliability_markdown,
    summarize_reliability,
)
from codingforme.eval.report import EXECUTION_MODE_LIVE_MODEL, EXECUTION_MODE_ORACLE_REPLAY
from codingforme.evaluator import BenchmarkEvaluator
from codingforme.models import final_answer, tool_call


def _row(task_id, passed, tool_steps, *, budget=4, ceiling=False, within=None):
    return {
        "id": task_id,
        "passed": passed,
        "tool_steps": tool_steps,
        "step_budget": budget,
        "within_budget": (tool_steps <= budget) if within is None else within,
        "step_ceiling_hit": ceiling,
        "failure_category": None if passed else "verifier_failed",
    }


# ----------------------------------------------------------------- 第二步：pass^k


def test_pass_hat_k_separates_flaky_from_stable():
    """三种稳定性必须被分开：轮轮过 / 轮轮挂 / 跳变。

    这是这一层存在的理由——单轮通过率把第三种读成前两种之一。
    """
    groups = [
        [_row("always", True, 3), _row("never", False, 5), _row("flaky", True, 2)],
        [_row("always", True, 3), _row("never", False, 5), _row("flaky", False, 9)],
        [_row("always", True, 3), _row("never", False, 5), _row("flaky", True, 4)],
    ]
    summary = summarize_reliability(groups)

    assert summary["repeats"] == 3
    assert summary["by_task"]["always"]["stability"] == STABILITY_STABLE_PASS
    assert summary["by_task"]["never"]["stability"] == STABILITY_STABLE_FAIL
    assert summary["by_task"]["flaky"]["stability"] == STABILITY_FLAKY
    assert summary["flaky_tasks"] == ["flaky"]

    # flaky 的 pass@1 是 2/3，但 pass^k 是 0——「过了两次」不等于「可靠」。
    assert summary["by_task"]["flaky"]["pass_at_1"] == 2 / 3
    assert summary["by_task"]["flaky"]["pass_hat_k"] == 0.0
    # 整体：3 个任务里只有 1 个轮轮都过。
    assert summary["pass_hat_k"] == 1 / 3
    assert summary["tasks_ever_passed"] == 2


def test_pass_at_1_is_the_mean_not_ever_passed():
    """pass@1 取平均，不能退化成「至少过一次」——那会把 1/3 读成 1.0。"""
    groups = [[_row("t", True, 2)], [_row("t", False, 2)], [_row("t", False, 2)]]
    summary = summarize_reliability(groups)
    assert summary["pass_at_1"] == 1 / 3
    assert summary["tasks_ever_passed"] == 1


def test_censored_steps_are_flagged_so_they_are_not_fitted_into_budgets():
    """顶到步数上限的运行必须被标出来：它的 tool_steps 只是下界。"""
    groups = [
        [_row("t", False, 15, budget=2, ceiling=True)],
        [_row("t", False, 8, budget=2)],
    ]
    summary = summarize_reliability(groups)
    assert summary["censored_runs"] == 1
    assert summary["tasks_with_censored_steps"] == 1
    assert summary["by_task"]["t"]["steps_censored"] is True
    assert "顶到了步数上限" in render_reliability_markdown(summary)


def test_single_repeat_report_says_the_layer_does_not_hold():
    """k=1 时不能照常印出 pass^k——那是个恒等于 pass@1 的假数字。"""
    summary = summarize_reliability([[_row("t", True, 2)]])
    text = render_reliability_markdown(summary)
    assert "k = 1" in text
    assert "这一层不成立" in text


# --------------------------------------------------- 第一步：预算降级只在 live 模式


def _run_one_task(tmp_path, benchmark_path, model_client_factory=None, step_budget_override=None):
    evaluator = BenchmarkEvaluator(
        benchmark_path=benchmark_path,
        artifact_path=tmp_path / "artifact.json",
        workspace_root=tmp_path / "ws",
        model_client_factory=model_client_factory,
        step_budget_override=step_budget_override,
    )
    return evaluator.run()


def _over_budget_solution(**_kwargs):
    """绕了 5 步（声明预算是 4）但最终把任务做对的解法。

    必须配 `step_budget_override` 才有意义：不给 agent 余量的话，它会在第 4 步
    就耗尽上限、根本走不到最终答案，那测到的是「没收敛」而不是「超预算」。
    """
    from codingforme.models import FakeModelClient

    return FakeModelClient([
        tool_call("read_file", path="README.md"),
        tool_call("read_file", path="README.md"),
        tool_call("list_files", path="."),
        tool_call("read_file", path="README.md"),
        tool_call(
            "patch_file",
            path="README.md",
            old_text="This is a placeholder benchmark fixture.",
            new_text="This fixture is a locked benchmark workspace.",
        ),
        final_answer("Done."),
    ])


def _one_task_benchmark(tmp_path):
    """从正式基准里切出单个任务，避免这组用例依赖全量数据集。"""
    import json

    source = Path("benchmarks/coding_tasks.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    data["tasks"] = [t for t in data["tasks"] if t["id"] == "readme_intro_locked"]
    # 基准路径的父目录的父目录被当作 repo_root，所以必须落在 benchmarks/ 同层结构下。
    target = tmp_path / "benchmarks" / "one_task.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    # fixture_repo 是相对 repo_root 的路径，把 repo_root 指回真实仓库。
    (tmp_path / "tests").mkdir(exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return target


def test_budget_still_gates_oracle_replay(tmp_path, monkeypatch):
    """回放模式下预算仍然是通过条件——这条基线不许动。"""
    monkeypatch.chdir(Path.cwd())
    artifact = _run_one_task(tmp_path, Path("benchmarks/coding_tasks.json"))
    assert artifact["execution_mode"] == EXECUTION_MODE_ORACLE_REPLAY
    for row in artifact["rows"]:
        assert row["budget_enforced"] is True


def test_live_model_records_budget_without_gating_on_it(tmp_path):
    """live 模式下超预算不再判失败，但 within_budget 仍逐条记录。"""
    artifact = _run_one_task(
        tmp_path,
        Path("benchmarks/coding_tasks.json"),
        model_client_factory=_over_budget_solution,
        step_budget_override=10,
    )
    assert artifact["execution_mode"] == EXECUTION_MODE_LIVE_MODEL

    target = next(row for row in artifact["rows"] if row["id"] == "readme_intro_locked")
    assert target["budget_enforced"] is False
    # 5 步 > 声明预算 4：确实超了，而且这个事实仍然被逐条记录下来。
    assert target["tool_steps"] == 5
    assert target["step_budget"] == 4
    assert target["within_budget"] is False
    # 但它不再是失败——任务做对了、也正常收尾了。
    assert target["passed"] is True
    assert target["failure_category"] is None


def test_session_suite_declares_the_execution_mode_it_actually_ran_in(tmp_path):
    """L3 套件的口径字段必须跟着实际执行方式走。

    不传 execution_mode 会默认成 oracle-replay，于是一次真实模型跑批被标成参考解
    回放，报告还会渲染出「capability 轴评的是参考解」这段假声明。
    """
    from codingforme.eval.session_suite import DEFAULT_SESSION_BENCHMARK_PATH, run_session_suite
    from codingforme.models import FakeModelClient

    scripted = run_session_suite(
        benchmark_path=DEFAULT_SESSION_BENCHMARK_PATH,
        workspace_root=tmp_path / "scripted",
    )
    assert scripted["run_context"]["execution_mode"] == EXECUTION_MODE_ORACLE_REPLAY

    injected = run_session_suite(
        benchmark_path=DEFAULT_SESSION_BENCHMARK_PATH,
        workspace_root=tmp_path / "injected",
        model_client_factory=lambda *a, **k: FakeModelClient([final_answer("ok")] * 40),
    )
    assert injected["run_context"]["execution_mode"] == EXECUTION_MODE_LIVE_MODEL


def test_step_budget_assertion_judges_against_the_limit_that_actually_applied():
    """判据必须取这次运行生效的上限，不是 base harness 声明的那个。

    评测里 harness 按任务派生 max_steps，判分器却只拿得到派生前的配置。
    写反的话，一个在自己上限内跑完的运行会被判成违规。
    """
    from types import SimpleNamespace

    from codingforme.eval.harness import DEFAULT_HARNESS
    from codingforme.eval.scorers import check_step_budget_respected

    # check_* 返回的是 (assertion_id, check_class, axis, passed, detail) 元组。
    # base harness 的上限是 6；这次运行实际被允许跑 15 步，用了 8 步。
    assert DEFAULT_HARNESS.max_steps == 6
    _, _, _, passed, detail = check_step_budget_respected(
        SimpleNamespace(tool_steps=8, max_steps=15), DEFAULT_HARNESS
    )
    assert passed, "在自己的上限内跑完，不该被 base harness 的 6 判成违规"
    assert detail["max_steps"] == 15
    assert detail["limit_source"] == "run"

    # 真的超了自己的上限，仍然要挂。
    _, _, _, over_passed, _ = check_step_budget_respected(
        SimpleNamespace(tool_steps=16, max_steps=15), DEFAULT_HARNESS
    )
    assert not over_passed

    # 老工件没落这个字段时回退到 harness 声明值，行为与改动前一致。
    _, _, _, legacy_passed, legacy_detail = check_step_budget_respected(
        SimpleNamespace(tool_steps=8, max_steps=None), DEFAULT_HARNESS
    )
    assert not legacy_passed
    assert legacy_detail["limit_source"] == "harness"


def test_budget_demotion_does_not_excuse_a_run_that_never_converged(tmp_path):
    """放宽的只是「超出声明预算」，不是「压根没收敛」。

    步数耗尽会让 stop_reason 变成 step_limit_reached，这条仍然判失败——
    否则降级就成了「跑多久都算过」。
    """
    from codingforme.models import FakeModelClient

    def never_finishes(**_kwargs):
        return FakeModelClient([tool_call("read_file", path="README.md")] * 30)

    artifact = _run_one_task(
        tmp_path,
        Path("benchmarks/coding_tasks.json"),
        model_client_factory=never_finishes,
    )
    target = next(row for row in artifact["rows"] if row["id"] == "readme_intro_locked")
    assert target["passed"] is False
    assert target["stop_reason"] != "final_answer_returned"
    assert target["step_ceiling_hit"] is True
