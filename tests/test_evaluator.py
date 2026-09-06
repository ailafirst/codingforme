import json
from pathlib import Path
from collections import Counter

import pytest

from codingforme.eval.scorers import ASSERTION_SUBJECTS
from codingforme.models import FakeModelClient, final_answer
from codingforme.evaluator import (
    BenchmarkEvaluator,
    load_benchmark,
    validate_benchmark,
    run_harness_regression_v2,
    run_fixed_benchmark,
    summarize_rows,
)

BENCHMARK_PATH = Path("benchmarks/coding_tasks.json")


def test_load_benchmark_validates_fixed_schema():
    benchmark = load_benchmark(Path("benchmarks/coding_tasks.json"))

    assert benchmark["schema_version"] == 2
    assert len(benchmark["tasks"]) == 15
    assert Counter(task["category"] for task in benchmark["tasks"]) == {
        "documentation": 2,
        "text-edit": 2,
        "tool-boundary": 3,
        "recovery": 3,
        "durable-contract": 2,
        # fan-out：先把 8 个模块都读一遍，再改其中少数几个。加这一类是因为
        # 另外 12 个任务全是串行（读一个改一个），结构上量不出受限编排的收益。
        "fan-out": 2,
        # long-context：单条工具结果必然超过 `tool_output_limit()` 的任务。加它是因为
        # 其余 14 个任务的 fixture 加起来只有 477 个 token，而 1M 档下单条上限是
        # 14,810——落盘/指针/窗口推进那整套机制在常规跑批里一次都触发不了，
        # 于是「机制生效了」和「一次都没跑起来」在工件上长得一模一样。
        "long-context": 1,
    }
    for task in benchmark["tasks"]:
        assert {
            "id",
            "prompt",
            "fixture_repo",
            "allowed_tools",
            "step_budget",
            "expected_artifact",
            "checks",
            "mutable_paths",
            "category",
        } <= set(task)
        assert isinstance(task["allowed_tools"], list)
        assert task["step_budget"] > 0
        # 每个任务两半判定都得有：只有 fail-to-pass 时，「改对一处、砸坏三处」照样满分。
        assert task["checks"]["fail_to_pass"]
        assert task["checks"]["pass_to_pass"]


def test_load_benchmark_rejects_missing_required_task_fields(tmp_path):
    benchmark_path = tmp_path / "bad-benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "tasks": [
                    {
                        "id": "broken",
                        "prompt": "Missing required task keys.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required"):
        load_benchmark(benchmark_path)


def test_run_fixed_benchmark_uses_fresh_fixture_copy_and_fresh_run_directory(tmp_path):
    artifact_path = tmp_path / "benchmark-v1.json"
    evaluator = BenchmarkEvaluator(
        benchmark_path=Path("benchmarks/coding_tasks.json"),
        artifact_path=artifact_path,
        workspace_root=tmp_path / "workspaces",
    )

    original_fixture = Path("tests/fixtures/bench_repo_patch/sample.txt").read_text(encoding="utf-8")
    artifact = evaluator.run()

    row = next(item for item in artifact["rows"] if item["id"] == "sample_beta_locked")
    copied_fixture = (tmp_path / "workspaces" / row["fixture_copy_relpath"]).resolve()
    run_dir = (tmp_path / "workspaces" / row["run_dir_relpath"]).resolve()

    assert artifact_path.exists()
    assert copied_fixture.exists()
    assert run_dir.exists()
    assert not row["fixture_copy_relpath"].startswith("/")
    assert not row["run_dir_relpath"].startswith("/")
    assert row["initial_history_empty"] is True
    assert row["initial_memory_empty"] is True
    assert row["initial_task_summary_empty"] is True
    assert Path("tests/fixtures/bench_repo_patch/sample.txt").read_text(encoding="utf-8") == original_fixture
    assert "beta-locked" in (copied_fixture / "sample.txt").read_text(encoding="utf-8")


def test_run_fixed_benchmark_reports_metadata_and_success_definition(tmp_path):
    artifact_path = tmp_path / "benchmark-v1.json"
    artifact = run_fixed_benchmark(
        benchmark_path=Path("benchmarks/coding_tasks.json"),
        artifact_path=artifact_path,
        workspace_root=tmp_path / "workspaces",
    )

    assert artifact_path.exists()
    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert persisted == artifact

    assert artifact["schema_version"] == 2
    # 数字必须自己声明它测的是什么：这批任务是回放参考解跑出来的，不是被测系统自己解的。
    assert artifact["execution_mode"] == "oracle-replay"
    assert artifact["summary"] == {
        "total_tasks": 15,
        "passed": 15,
        "failed": 0,
        "pass_rate": 1.0,
        "within_budget": 15,
        "verifier_passes": 15,
        "within_budget_rate": 1.0,
        "verifier_pass_rate": 1.0,
        "failure_category_counts": {},
    }
    assert artifact["failure_category_counts"] == {}

    reproducibility = artifact["reproducibility"]
    assert reproducibility["model_name"] == "FakeModelClient"
    assert reproducibility["model_version"] == "scripted-deterministic"
    assert reproducibility["fixture_snapshot_id"].startswith("sha256:")
    assert reproducibility["decoding"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 1024,
    }
    assert reproducibility["timezone"] == "Asia/Shanghai"
    # locale 是**记录**用的复现字段，不是被测行为：断言它等于某个具体值，等于把
    # 测试钉死在一种宿主环境上（`C.UTF-8` 只在 POSIX 容器里成立）。要断言的是
    # 「这次运行的 locale 被记下来了」。
    assert isinstance(reproducibility["locale"], str)
    assert reproducibility["locale"]

    for row in artifact["rows"]:
        assert not row["fixture_copy_relpath"].startswith("/")
        assert not row["run_dir_relpath"].startswith("/")
        assert not row["task_state_relpath"].startswith("/")
        assert not row["report_relpath"].startswith("/")
        assert row["status"] == "pass"
        assert row["passed"] is True
        assert row["within_budget"] is True
        assert row["verifier_passed"] is True
        assert row["fail_to_pass_passed"] is True
        assert row["pass_to_pass_passed"] is True
        assert row["regressions"] == []
        assert row["expected_artifact_exists"] is True
        assert row["non_failure_stop_reason"] is True
        assert row["stop_reason"] == "final_answer_returned"


def test_run_fixed_benchmark_covers_recovery_and_durable_contract_rows(tmp_path):
    artifact = run_fixed_benchmark(
        benchmark_path=Path("benchmarks/coding_tasks.json"),
        artifact_path=tmp_path / "benchmark-v1.json",
        workspace_root=tmp_path / "workspaces",
    )

    context_row = next(item for item in artifact["rows"] if item["id"] == "context_reduction_checkpoint")
    durable_row = next(item for item in artifact["rows"] if item["id"] == "durable_promotion_reject")

    trace_path = (tmp_path / "workspaces" / context_row["run_dir_relpath"] / "trace.jsonl").resolve()
    trace_events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

    assert any(
        event.get("event") == "checkpoint_created" and event.get("trigger") == "context_reduction"
        for event in trace_events
    )
    assert durable_row["report"]["durable_rejections"] == [
        "reference:secret_shaped",
        "project:transient_task_state",
    ]


def test_run_harness_regression_v2_writes_named_artifact(tmp_path):
    artifact_path = tmp_path / "artifacts" / "harness-regression-v2.json"

    artifact = run_harness_regression_v2(
        benchmark_path=Path("benchmarks/coding_tasks.json"),
        artifact_path=artifact_path,
        workspace_root=tmp_path / "workspaces",
    )

    assert artifact_path.exists()
    assert artifact["summary"]["total_tasks"] == 15
    assert artifact["summary"]["pass_rate"] == 1.0
    assert artifact["summary"]["within_budget_rate"] == 1.0
    assert artifact["summary"]["verifier_pass_rate"] == 1.0


def test_run_task_anchors_paths_to_fixture_copy_even_inside_repo_workspace():
    evaluator = BenchmarkEvaluator(
        benchmark_path=Path("benchmarks/coding_tasks.json"),
        artifact_path=Path("docs/review-pack/benchmark-v1.json"),
        workspace_root=Path("."),
    )

    task = next(item for item in evaluator.load()["tasks"] if item["id"] == "readme_intro_locked")
    row = evaluator.run_task(task)

    assert row["status"] == "pass"
    fixture_copy = Path(row["fixture_copy_relpath"])
    readme_path = fixture_copy / "README.md"
    assert "This fixture is a locked benchmark workspace." in readme_path.read_text(encoding="utf-8")


def test_summarize_rows_counts_failure_categories():
    summary = summarize_rows(
        [
            {
                "status": "pass",
                "within_budget": True,
                "verifier_passed": True,
                "expected_artifact_exists": True,
                "non_failure_stop_reason": True,
            },
            {
                "status": "fail",
                "within_budget": False,
                "verifier_passed": False,
                "expected_artifact_exists": False,
                "non_failure_stop_reason": False,
                "failure_category": "verifier_failed",
            },
            {
                "status": "fail",
                "within_budget": False,
                "verifier_passed": True,
                "expected_artifact_exists": True,
                "non_failure_stop_reason": False,
                "failure_category": "budget_exceeded",
            },
        ]
    )

    assert summary["total_tasks"] == 3
    assert summary["passed"] == 1
    assert summary["failed"] == 2
    assert summary["pass_rate"] == pytest.approx(1 / 3)
    assert summary["within_budget"] == 1
    assert summary["verifier_passes"] == 2
    assert summary["failure_category_counts"] == {
        "budget_exceeded": 1,
        "verifier_failed": 1,
    }


def test_every_content_target_the_model_must_produce_appears_in_the_prompt():
    """判据要求写进文件的字面串，必须在提示词里出现过。

    踩过的坑：`invalid_patch_recovery` 这类任务的 F2P 要求 README 里出现
    'recovered after invalid patch args'，而提示词从没提过这串字。参考解
    (`ORACLE_SOLUTIONS`) 知道它，任何真实模型都猜不到——于是这些任务在 live
    模式下 0/3 恒挂，把 pass@1 的分母污染成「测不出能力的能力指标」。

    只查 `file_contains`：`file_not_contains` 说的是「把原有内容删掉」，
    report/trace 类判据查的是运行状态而不是模型产出，都不适用这条约束。

    真正的约束是「模型有办法拿到这串字」，提示词只是最常见的那个来源。
    `spill_pointer_recovery` 是另一个来源：那串 token 藏在工作区的日志里，任务
    要考的就是模型能不能把它找出来——所以 `setup.marker` 也算数。放宽到「提示词
    **或** setup 声明的内容」，而不是给这个任务开豁免：豁免会让下一个写错的任务
    照样溜过去。
    """
    tasks = load_benchmark(Path("benchmarks/coding_tasks.json"))["tasks"]

    def obtainable(task, text):
        if text in task["prompt"]:
            return True
        marker = str((task.get("setup") or {}).get("marker", ""))
        return bool(marker) and marker.split(": ", 1)[-1] in text

    missing = [
        (task["id"], check["text"])
        for task in tasks
        for check in task["checks"]["fail_to_pass"]
        if check["kind"] == "file_contains" and not obtainable(task, check["text"])
    ]
    assert missing == []


def test_a_task_never_forbids_changing_a_file_it_asks_the_model_to_change():
    """提示词要求改文件，就不能同时把该文件的原内容列进基线断言。

    踩过的坑：`context_reduction_checkpoint` 的提示词说 'Finish the README
    update'，而它的 `mutable_paths` 是空的、pass-to-pass 又要求 README 逐字
    不变——照做必挂，不照做才过。这种自相矛盾在回放模式下看不见，因为参考解
    根本不动文件。
    """
    tasks = load_benchmark(Path("benchmarks/coding_tasks.json"))["tasks"]
    for task in tasks:
        if task["mutable_paths"]:
            continue
        baseline_paths = {
            check.get("path")
            for check in task["checks"]["pass_to_pass"]
            if check.get("path")
        }
        for path in baseline_paths:
            stem = path.split("/")[-1].split(".")[0].lower()
            assert stem not in task["prompt"].lower() or "do not modify" in task["prompt"].lower(), (
                f"{task['id']}: 提示词提到了 {path}，但 mutable_paths 为空且它进了基线断言"
            )


def test_a_task_cannot_declare_an_assertion_that_does_not_exist():
    """`expected_trajectory_failures` 的名字要对着真实断言表校验。

    写错一个字母，这条声明就永远不生效，而报告上看不出任何异常——只会看到那条
    失败一直混在 L1 通过率的分母里。所以在加载数据集时就把它拦下来。
    """
    data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    declaring = [
        task for task in data["tasks"] if task.get("expected_trajectory_failures")
    ]
    # 数据集本身要真的用上这个字段，否则这条校验没有保护对象。
    assert declaring, "no task declares expected_trajectory_failures"
    for task in declaring:
        for assertion_id in task["expected_trajectory_failures"]:
            assert assertion_id in ASSERTION_SUBJECTS

    data["tasks"][0]["expected_trajectory_failures"] = ["path_confinedd"]
    with pytest.raises(ValueError, match="unknown trajectory assertion"):
        validate_benchmark(data, repo_root=Path.cwd())


def test_intersecting_two_allowlists_keeps_both_promises():
    from codingforme.eval.harness import intersect_tools_allowlist

    # None 的含义是「不限制」，所以两边任意一侧为 None 时结果就是另一侧。
    assert intersect_tools_allowlist(None, ["read_file", "patch_file"]) == ("patch_file", "read_file")
    assert intersect_tools_allowlist(("read_file", "search"), None) == ("read_file", "search")
    # 变体说「只给这三个」、任务说「只该用这两个」，两句话都要成立。
    assert intersect_tools_allowlist(("read_file", "search", "list_files"), ["read_file", "patch_file"]) == (
        "read_file",
    )
    # 交集为空时报错，而不是装配一个一个工具都没有的 agent——后者在工件上
    # 看起来像「模型不会做这道题」。
    with pytest.raises(ValueError, match="intersection is empty"):
        intersect_tools_allowlist(("search",), ["read_file"])


def test_a_task_that_allowlists_read_file_cannot_reach_write_file(tmp_path):
    """N-5 的回归网：`allowed_tools` 必须真的进 agent 装配。

    这个字段此前只被校验、只被抄进结果行，模型照样能调 `write_file`。这里
    脚本化一个越权调用，断言它在工具注册表这一层就不存在。
    """
    from codingforme.models import FakeModelClient, final_answer, tool_call

    evaluator = BenchmarkEvaluator(
        benchmark_path=BENCHMARK_PATH,
        artifact_path=tmp_path / "benchmark.json",
        workspace_root=tmp_path / "workspaces",
        model_client_factory=lambda task, workspace: FakeModelClient(
            [
                tool_call("write_file", path="pwn.txt", content="escaped"),
                final_answer("Project convention: Preserve benchmark reproducibility."),
            ]
        ),
    )
    task = next(item for item in evaluator.load()["tasks"] if item["id"] == "durable_promotion_accept")
    assert task["allowed_tools"] == ["read_file"]

    row = evaluator.run_task(task)

    fixture_copy = tmp_path / "workspaces" / row["fixture_copy_relpath"]
    assert not (fixture_copy / "pwn.txt").exists()
    trace = (tmp_path / "workspaces" / row["run_dir_relpath"] / "trace.jsonl").read_text(encoding="utf-8")
    assert "unknown tool 'write_file'" in trace


def test_the_trace_records_both_the_declared_allowlist_and_the_actual_registry(tmp_path):
    """判分侧的地基：两个字段都要落盘，缺一条就查不出 N-5 那种「声明了却没执行」。

    `tools_allowlist` 是数据集声称的边界，`tool_names` 是模型实际看得到的工具。
    只有 `tool_count` 这一个数字时（原状），一个声明 `["read_file"]` 却装配了
    全部 6 个工具的运行，在工件上和正常运行长得一模一样。
    """
    import json as _json

    from codingforme.models import FakeModelClient, final_answer

    evaluator = BenchmarkEvaluator(
        benchmark_path=BENCHMARK_PATH,
        artifact_path=tmp_path / "benchmark.json",
        workspace_root=tmp_path / "workspaces",
        model_client_factory=lambda task, workspace: FakeModelClient([final_answer("done")]),
    )
    task = next(item for item in evaluator.load()["tasks"] if item["id"] == "durable_promotion_accept")
    row = evaluator.run_task(task)

    trace_path = tmp_path / "workspaces" / row["run_dir_relpath"] / "trace.jsonl"
    built = [
        _json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if _json.loads(line).get("event") == "prompt_built"
    ]
    assert built, "没有 prompt_built 事件"
    metadata = built[0]["prompt_metadata"]
    assert metadata["tools_allowlist"] == ["read_file"]
    assert metadata["tool_names"] == ["read_file"]


def test_a_fixture_never_carries_a_previous_runs_session_into_the_workspace(tmp_path):
    """样板仓库里残留的 `.codingforme/` 不能被复制进任务工作区。

    live 跑批会在样板仓库里写下 session 与 run 工件（它们本来就该落在工作区里，
    而 live 跑批的工作区一度就是样板仓库本身）。判分不受影响——工作区快照和
    P2P 的回归快照都各自排除了这个目录——但复制过去会让每个任务凭空多出一份
    别的运行留下的 session，resume 那几个任务尤其容易被误读成「恢复成功了」。
    """
    fixture = tmp_path / "tests" / "fixtures" / "bench_repo_readme"
    (fixture / ".codingforme" / "sessions").mkdir(parents=True)
    (fixture / ".codingforme" / "sessions" / "stray.json").write_text("{}", encoding="utf-8")
    (fixture / "__pycache__").mkdir()
    (fixture / "__pycache__" / "stale.pyc").write_bytes(b"\x00")
    (fixture / "README.md").write_text("hello\n", encoding="utf-8")

    benchmark_path = tmp_path / "benchmarks" / "mini.json"
    benchmark_path.parent.mkdir(parents=True)
    benchmark_path.write_text("{}", encoding="utf-8")

    evaluator = BenchmarkEvaluator(
        benchmark_path=benchmark_path,
        artifact_path=tmp_path / "artifact.json",
        workspace_root=tmp_path / "workspaces",
        model_client_factory=lambda task, workspace: FakeModelClient([final_answer("Done.")]),
    )
    row = evaluator.run_task(
        {
            "id": "mini_task",
            "prompt": "Do nothing.",
            "fixture_repo": "tests/fixtures/bench_repo_readme",
            "allowed_tools": ["read_file"],
            "step_budget": 2,
            "expected_artifact": "README.md",
            "checks": {
                "fail_to_pass": [{"kind": "file_contains", "path": "README.md", "text": "hello"}],
                "pass_to_pass": [{"kind": "file_exists", "path": "README.md"}],
            },
            "mutable_paths": ["README.md"],
            "category": "documentation",
        }
    )

    copied = (tmp_path / "workspaces" / row["fixture_copy_relpath"]).resolve()
    assert (copied / "README.md").exists()
    # 这次运行自己写下的 session 会在这里，所以断言的是「那个残留的文件」不在。
    assert not (copied / ".codingforme" / "sessions" / "stray.json").exists()
    assert not (copied / "__pycache__").exists()
