"""P2-0（参考解正名 + 执行模式）与 P2-d（P2P 判定 + 去 shell）的验收。

这里最重要的不是「12/12 还是 12/12」，而是**新加的那半边判定真的会挂**：
如果 pass-to-pass 在任何情况下都不报警，那它就等于没加。所以下面前三条用例都在
构造「任务完成了、但顺手破坏了别的东西」的解法，逼 P2P 出手。
"""

import json
from pathlib import Path

import pytest

from codingforme.eval.checks import (
    CheckContext,
    run_checks,
    workspace_regressions,
    workspace_snapshot,
)
from codingforme.eval.report import (
    EXECUTION_MODE_LIVE_MODEL,
    EXECUTION_MODE_ORACLE_REPLAY,
    build_run_context,
    render_eval_result_markdown,
)
from codingforme.eval.harness import get_harness
from codingforme.eval.suite import run_benchmark_suite
from codingforme.evaluator import BenchmarkEvaluator, validate_benchmark
from codingforme.models import FakeModelClient, final_answer, tool_call

BENCHMARK_PATH = Path("benchmarks/coding_tasks.json")


def _evaluator(tmp_path, outputs_by_task=None):
    """按需注入自定义解法；不注入就是回放数据集自带的参考解。"""
    factory = None
    if outputs_by_task:
        def factory(task, workspace):
            return FakeModelClient(list(outputs_by_task[task["id"]]))

    return BenchmarkEvaluator(
        benchmark_path=BENCHMARK_PATH,
        artifact_path=tmp_path / "artifact.json",
        workspace_root=tmp_path / "workspaces",
        model_client_factory=factory,
    )


def _task(evaluator, task_id):
    return next(task for task in evaluator.load()["tasks"] if task["id"] == task_id)


# ------------------------------------------------------- P2-d：P2P 真的会挂


def test_pass_to_pass_catches_a_solution_that_finishes_the_task_but_breaks_the_rest(tmp_path):
    """F2P 全过、P2P 挂掉——这正是只有 F2P 时会被漏掉的那一类失败。"""
    saboteur = [
        tool_call(
            "patch_file",
            path="README.md",
            old_text="This is a placeholder benchmark fixture.",
            new_text="This fixture is a locked benchmark workspace.",
        ),
        # 任务没要求动这一行。改了它，仓库就被破坏了。
        tool_call(
            "patch_file",
            path="README.md",
            old_text="- Placeholder note about the repo.",
            new_text="- (wiped)",
        ),
        final_answer("Done."),
    ]
    evaluator = _evaluator(tmp_path, {"readme_intro_locked": saboteur})

    row = evaluator.run_task(_task(evaluator, "readme_intro_locked"))

    assert row["fail_to_pass_passed"] is True, "任务本身确实完成了"
    assert row["pass_to_pass_passed"] is False, "但原有内容被破坏了，P2P 必须报警"
    assert row["passed"] is False
    assert row["failure_category"] == "regression_detected"

    broken = [item for item in row["check_results"]["pass_to_pass"] if not item["passed"]]
    assert [item["target"] for item in broken] == ["README.md"]
    assert "Placeholder note about the repo." in broken[0]["detail"]


def test_pass_to_pass_catches_stray_files_outside_mutable_paths(tmp_path):
    """显式基线断言查不到的那一半，由隐式快照对比兜住。"""
    littering = [
        tool_call(
            "patch_file",
            path="README.md",
            old_text="This is a placeholder benchmark fixture.",
            new_text="This fixture is a locked benchmark workspace.",
        ),
        tool_call("write_file", path="STRAY.md", content="not part of the task"),
        final_answer("Done."),
    ]
    evaluator = _evaluator(tmp_path, {"readme_intro_locked": littering})

    # 这里刻意把任务的 allowed_tools 放宽到含 write_file。理由：接上 N-5 之后
    # 任务白名单会真的裁掉工具注册表，而数据集里没有任何一个任务允许 write_file，
    # 于是「模型乱建文件」这个场景在原任务上物理不可发生——测不到本条要测的
    # 隐式 P2P 快照对比。放宽的是这一条用例自己的任务副本，数据集不动。
    task = dict(_task(evaluator, "readme_intro_locked"))
    task["allowed_tools"] = [*task["allowed_tools"], "write_file"]

    row = evaluator.run_task(task)

    assert row["fail_to_pass_passed"] is True
    # 显式基线全过：README 的原有内容一个字都没少。
    assert all(item["passed"] for item in row["check_results"]["pass_to_pass"])
    # 但工作区多了一个计划外的文件。
    assert row["regressions"] == [{"path": "STRAY.md", "change": "created"}]
    assert row["pass_to_pass_passed"] is False
    assert row["failure_category"] == "regression_detected"


def test_setup_mutations_are_scenery_not_regressions(tmp_path):
    """基线在 setup 之后拍：布景改文件不算 agent 的回归。

    freshness 任务的 setup 会把 sample.txt 改脏来制造 stale 状态，而这个任务的
    mutable_paths 是空的。如果基线在 setup 之前拍，它会被判成回归——那是错的，
    因为 agent 根本没碰过那个文件。
    """
    evaluator = _evaluator(tmp_path)

    row = evaluator.run_task(_task(evaluator, "freshness_reanchor_resume"))

    assert row["mutable_paths"] == []
    assert row["regressions"] == []
    assert row["pass_to_pass_passed"] is True
    assert row["passed"] is True


def test_workspace_regressions_reports_every_change_class(tmp_path):
    before = {"a.txt": "sha256:1", "b.txt": "sha256:2", "keep.txt": "sha256:3"}
    after = {"a.txt": "sha256:9", "c.txt": "sha256:4", "keep.txt": "sha256:3"}

    assert workspace_regressions(before, after, mutable_paths=()) == [
        {"path": "a.txt", "change": "modified"},
        {"path": "b.txt", "change": "deleted"},
        {"path": "c.txt", "change": "created"},
    ]
    # 声明为可改的路径整条退出比较。
    assert workspace_regressions(before, after, mutable_paths=("a.txt", "b.txt", "c.txt")) == []


def test_workspace_snapshot_skips_agent_own_artifacts(tmp_path):
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    runs = tmp_path / ".codingforme" / "runs" / "r1"
    runs.mkdir(parents=True)
    (runs / "trace.jsonl").write_text("{}", encoding="utf-8")

    snapshot = workspace_snapshot(tmp_path)

    assert list(snapshot) == ["keep.txt"]


# ------------------------------------------------------------ P2-d：不再走 shell


def test_verification_never_shells_out(tmp_path, monkeypatch):
    """判定不得再起 shell 子进程。

    改动前每个任务的判定是 `python3 -c "..."` + `shell=True`——Windows 上没有
    python3、POSIX 引号规则也不成立，失败原因是执行环境而不是被测系统。
    """
    import codingforme.evaluator as evaluator_module

    real_run = evaluator_module.subprocess.run

    def guarded(args, **kwargs):
        if kwargs.get("shell") or isinstance(args, str):
            raise AssertionError(f"verification must not shell out: {args!r}")
        return real_run(args, **kwargs)

    monkeypatch.setattr(evaluator_module.subprocess, "run", guarded)

    evaluator = _evaluator(tmp_path)
    row = evaluator.run_task(_task(evaluator, "readme_intro_locked"))

    assert row["passed"] is True


# --------------------------------------------------- P2-d：判定本身可被校验


def _benchmark_with_checks(checks, mutable_paths=("README.md",)):
    return {
        "schema_version": 2,
        "tasks": [
            {
                "id": "probe",
                "prompt": "probe",
                "fixture_repo": "tests/fixtures/bench_repo_readme",
                "allowed_tools": ["read_file"],
                "step_budget": 2,
                "expected_artifact": "README.md",
                "mutable_paths": list(mutable_paths),
                "checks": checks,
                "category": "documentation",
            }
        ],
    }


@pytest.mark.parametrize(
    "checks, message",
    [
        (
            {"fail_to_pass": [{"kind": "no_such_kind", "path": "README.md"}], "pass_to_pass": [{"kind": "file_exists", "path": "README.md"}]},
            "unknown check kind",
        ),
        (
            {"fail_to_pass": [{"kind": "file_contains", "path": "README.md"}], "pass_to_pass": [{"kind": "file_exists", "path": "README.md"}]},
            "missing required keys",
        ),
        (
            {"fail_to_pass": [{"kind": "file_exists", "path": "README.md"}], "pass_to_pass": []},
            "pass_to_pass must be a non-empty list",
        ),
        (
            {"fail_to_pass": [{"kind": "file_exists", "path": "README.md"}]},
            "pass_to_pass must be a non-empty list",
        ),
    ],
)
def test_invalid_checks_are_rejected_at_load_time(checks, message):
    """判定现在是数据，所以加载时就能校验——不必等到跑完评测才发现写错了。"""
    with pytest.raises(ValueError, match=message):
        validate_benchmark(_benchmark_with_checks(checks), repo_root=Path.cwd())


def test_mutable_paths_may_not_escape_the_fixture():
    benchmark = _benchmark_with_checks(
        {"fail_to_pass": [{"kind": "file_exists", "path": "README.md"}], "pass_to_pass": [{"kind": "file_exists", "path": "README.md"}]},
        mutable_paths=("../outside.txt",),
    )

    with pytest.raises(ValueError, match="stay inside the fixture"):
        validate_benchmark(benchmark, repo_root=Path.cwd())


@pytest.mark.parametrize("escaping", ["../outside.txt", "/etc/passwd", "a/../../outside.txt"])
def test_check_paths_may_not_escape_the_fixture(escaping):
    """判定器的路径必须和 `CodingForMe.path()` 一样受约束。

    一条指向宿主文件的断言能让任务凭空「通过」——判定读错文件比工具读错文件更严重，
    因为它直接决定那个数字。
    """
    benchmark = _benchmark_with_checks(
        {
            "fail_to_pass": [{"kind": "file_contains", "path": escaping, "text": "x"}],
            "pass_to_pass": [{"kind": "file_exists", "path": "README.md"}],
        },
        mutable_paths=(),
    )

    with pytest.raises(ValueError, match="stay inside the fixture"):
        validate_benchmark(benchmark, repo_root=Path.cwd())


def test_check_execution_confines_paths_even_through_symlinks(tmp_path):
    """静态校验挡 `../`，运行时这道挡 symlink——解析之后才知道它指向哪儿。"""
    outside = tmp_path / "outside.txt"
    outside.write_text("secret-outside", encoding="utf-8")
    root = tmp_path / "fixture"
    root.mkdir()
    try:
        (root / "link.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("本机不允许创建 symlink")

    context = CheckContext(root=root)

    assert context.resolve("link.txt") is None
    results = run_checks(
        [
            {"kind": "file_contains", "path": "link.txt", "text": "secret-outside"},
            {"kind": "file_exists", "path": "link.txt"},
            # 读不到时不得被当成「不包含」而通过。
            {"kind": "file_not_contains", "path": "link.txt", "text": "secret-outside"},
        ],
        context,
    )

    assert [item["passed"] for item in results] == [False, False, False]
    assert all("outside the workspace" in item["detail"] for item in results)


def test_workspace_snapshot_records_symlinks_without_dereferencing(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret-outside", encoding="utf-8")
    root = tmp_path / "fixture"
    root.mkdir()
    (root / "real.txt").write_text("in-tree", encoding="utf-8")
    try:
        (root / "link.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("本机不允许创建 symlink")

    snapshot = workspace_snapshot(root)

    assert snapshot["real.txt"].startswith("sha256:")
    # 记的是链接目标，不是被链接文件的内容——快照绝不读工作区外的字节。
    assert snapshot["link.txt"].startswith("symlink:")
    assert "secret-outside" not in json.dumps(snapshot)


def test_run_checks_reports_every_failure_not_just_the_first(tmp_path):
    """不短路：第一条挂掉就返回会丢掉其余归因，而归因清单才是这层判定的产出。"""
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    context = CheckContext(root=tmp_path, report={"a": {"b": 1}}, trace_events=[{"event": "x"}])

    results = run_checks(
        [
            {"kind": "file_contains", "path": "README.md", "text": "missing-one"},
            {"kind": "file_contains", "path": "README.md", "text": "missing-two"},
            {"kind": "report_equals", "field": "a.b", "value": 1},
            {"kind": "trace_event", "event": "never-emitted"},
        ],
        context,
    )

    assert [item["passed"] for item in results] == [False, False, True, False]
    assert all(item["detail"] for item in results if not item["passed"])


def test_report_and_trace_checks_read_persisted_artifacts(tmp_path):
    context = CheckContext(
        root=tmp_path,
        report={"checkpoint_id": "ckpt_1", "prompt_metadata": {"resume_status": "partial-stale"}, "items": [1, 2]},
        trace_events=[{"event": "checkpoint_created", "trigger": "freshness_mismatch"}],
    )

    results = run_checks(
        [
            {"kind": "report_truthy", "field": "checkpoint_id"},
            {"kind": "report_equals", "field": "prompt_metadata.resume_status", "value": "partial-stale"},
            {"kind": "report_length", "field": "items", "length": 2},
            {"kind": "trace_event", "event": "checkpoint_created", "where": {"trigger": "freshness_mismatch"}},
            {"kind": "trace_event", "event": "checkpoint_created", "where": {"trigger": "context_reduction"}},
        ],
        context,
    )

    assert [item["passed"] for item in results] == [True, True, True, True, False]


# ------------------------------------------------- P2-0：数字自带口径声明


def test_execution_mode_distinguishes_oracle_replay_from_live_model(tmp_path):
    assert _evaluator(tmp_path).execution_mode == EXECUTION_MODE_ORACLE_REPLAY
    assert _evaluator(tmp_path, {"x": []}).execution_mode == EXECUTION_MODE_LIVE_MODEL


def test_build_run_context_rejects_unknown_execution_mode():
    with pytest.raises(ValueError, match="unknown execution_mode"):
        build_run_context(execution_mode="scripted-ish")


def test_oracle_replay_results_carry_the_caveat_and_live_model_results_do_not():
    """回放模式下，口径声明必须和数字一起出现，而且排在数字**之前**。

    没有它，capability 轴会被读成「这套 harness 的能力」，可它评的是数据集里
    那份参考解。
    """
    def render(execution_mode):
        return render_eval_result_markdown(
            {
                "suite": "fixed-benchmark",
                "run_context": build_run_context(execution_mode=execution_mode),
                "harness": {"name": "full"},
                "aggregates": {"total": 1, "passed": 1, "pass_rate": 1.0, "by_level": {}, "by_axis": {}},
            }
        )

    replay = render(EXECUTION_MODE_ORACLE_REPLAY)
    live = render(EXECUTION_MODE_LIVE_MODEL)

    assert "## 这些数字测的是什么" in replay
    assert "capability 轴评的是参考解脚本" in replay
    assert replay.index("## 这些数字测的是什么") < replay.index("## 结果")

    assert "## 这些数字测的是什么" not in live
    assert f"- 执行模式：{EXECUTION_MODE_LIVE_MODEL}" in live


def test_artifact_and_rows_record_the_new_verification_shape(tmp_path):
    evaluator = _evaluator(tmp_path)

    artifact = evaluator.run()

    assert artifact["execution_mode"] == EXECUTION_MODE_ORACLE_REPLAY
    assert artifact["reproducibility"]["execution_mode"] == EXECUTION_MODE_ORACLE_REPLAY
    assert artifact["summary"]["pass_rate"] == 1.0

    for row in artifact["rows"]:
        # 旧的 shell 判定字段必须彻底消失，否则会有人继续读那个口径。
        assert "verifier" not in row
        assert "verifier_exit_code" not in row
        assert row["check_results"]["fail_to_pass"]
        assert row["check_results"]["pass_to_pass"]
        assert row["regressions"] == []

    persisted = json.loads((tmp_path / "artifact.json").read_text(encoding="utf-8"))
    assert persisted == artifact


def test_a_failed_repeat_does_not_destroy_the_repeats_that_finished(tmp_path, monkeypatch):
    """一轮跑挂了，已经跑完的那几轮必须还在。

    踩过的坑：一次 2.4 小时的 k=3 跑批，r1、r2 各约 50 分钟都跑完了，r3 在第 10 个
    任务上撞到服务端 `HTTP 400 / Connection refused`，异常一路上抛——而聚合发生在
    重复循环**之后**，于是两轮完整数据留在磁盘上却没有任何结果文件。
    """
    from codingforme.eval import suite as suite_module

    real_run = suite_module.BenchmarkEvaluator.run
    calls = {"n": 0}

    def flaky_run(self):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("OpenAI-compatible error: HTTP 400 Connection refused")
        return real_run(self)

    monkeypatch.setattr(suite_module.BenchmarkEvaluator, "run", flaky_run)

    result = run_benchmark_suite(
        harness=get_harness("full"),
        workspace_root=tmp_path / "ws",
        artifact_path=tmp_path / "artifact.json",
        repeats=3,
    )

    context = result["run_context"]
    assert context["repeats_requested"] == 3
    assert context["repeats_completed"] == 2, "跑完的两轮必须保住"
    assert len(context["aborted_repeats"]) == 1
    assert "Connection refused" in context["aborted_repeats"][0]["error"]
    # k 要按**实际完成**的轮数报，不能报成 3。
    assert result["aggregates"]["reliability"]["repeats"] == 2
    # 中止这件事必须出现在报告里，而且在所有数字之前。
    markdown = render_eval_result_markdown(result)
    assert "实际完成" in markdown
    assert markdown.index("实际完成") < markdown.index("## 结果")


def test_every_repeat_failing_raises_instead_of_writing_an_empty_report(tmp_path, monkeypatch):
    """一轮都没跑成时必须抛——静默产出一份空报告比崩掉更危险。"""
    from codingforme.eval import suite as suite_module

    def always_fails(self):
        raise RuntimeError("provider down")

    monkeypatch.setattr(suite_module.BenchmarkEvaluator, "run", always_fails)

    with pytest.raises(RuntimeError, match="every repeat aborted"):
        run_benchmark_suite(
            harness=get_harness("full"),
            workspace_root=tmp_path / "ws",
            artifact_path=tmp_path / "artifact.json",
            repeats=2,
        )


def test_run_context_records_where_the_benchmark_actually_ran(tmp_path):
    """工作区路径、以及它是不是嵌在某个 git 仓库里，都要落进工件。

    踩过坑：评测工作区落到了本仓库目录下，`workspace.py` 的快照被悄悄放大成外层
    仓库，一整次 12 任务 x 3 轮的跑批因此作废——而当时工件里根本看不出这批数据跑
    在哪，只能靠人肉复盘。
    """
    clean = tmp_path / "outside"
    clean.mkdir()
    context = build_run_context(workspace_root=clean)
    assert context["workspace_root"] == str(clean.resolve())
    # 不在任何仓库里时恒为空串——这才是正常状态。
    assert context["workspace_git_root"] == ""

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    nested = repo / "eval-workspace"
    nested.mkdir()
    dirty = build_run_context(workspace_root=nested)
    assert dirty["workspace_git_root"] == str(repo.resolve())

    # 警告必须出现在所有数字之前，否则读者会先把不可比的数字读进去。
    markdown = render_eval_result_markdown(
        {"suite": "fixed-benchmark", "run_context": dirty, "cases": [], "aggregates": {}}
    )
    assert "评测工作区落在 git 仓库" in markdown
    assert markdown.index("评测工作区落在 git 仓库") < markdown.index("## 结果")

    # 不传 workspace_root 时字段仍在，只是为空——schema 不随调用点摇摆。
    assert build_run_context()["workspace_root"] == ""
