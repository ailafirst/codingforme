"""端到端入口：跑一次基准，产出统一 schema 的结果。

它把原语串起来演示一遍完整链路：

    HarnessSpec（被测对象）→ BenchmarkEvaluator（执行）
        → TraceIndex（把工件重建成 session/run/turn）
        → L2 任务判定 + L1 轨迹断言 → 统一结果 schema

L2 的 cases 全部来自基准已有的判定（这一层不引入新口径）；L1 的 cases 由
`scorers.py` 在同一份 TraceIndex 上跑确定性断言得到——同一次执行，两个层级，
不需要重跑。
"""

import sys
from pathlib import Path

from ..evaluator import DEFAULT_BENCHMARK_PATH, BenchmarkEvaluator
from .harness import DEFAULT_HARNESS
from .report import (
    AXIS_CAPABILITY,
    AXIS_EFFICIENCY,
    EXECUTION_MODE_LIVE_MODEL,
    LEVEL_TASK,
    aggregate_cases,
    build_case_result,
    build_eval_result,
    build_run_context,
    render_eval_result_markdown,
    write_eval_result,
)
from .economy import summarize_economy
from .reliability import summarize_reliability
from .tool_usage import summarize_tool_usage
from .scorers import (
    mark_expected_failures,
    score_index,
    summarize_assertions,
    trajectory_cases,
)
from .trace import TraceIndex

_BASE_NOTE = (
    "L2 的判定同时包含 fail-to-pass（任务完成了）与 pass-to-pass（没破坏别的东西）；"
    "L1 全部是 trace 上的确定性断言，不使用模型裁判。"
)
_ORACLE_ECONOMY_NOTE = (
    "economy 轴未覆盖：回放模式下 FakeModelClient 不产 usage 数据，成本类指标须由真实 provider 跑批。"
)
_LIVE_ECONOMY_NOTE = (
    "economy 轴见「成本」一节：真实 provider 会返回 usage。注意思维链 token 计入 output_tokens，"
    "按 output_tokens - reasoning_tokens 算成本会系统性低估（k=3 基线实测思维链占输出 80.5%）。"
)


def suite_notes(execution_mode):
    """备注要跟着执行模式变。

    写死「economy 轴未覆盖」曾经是对的——那时只有回放模式。接上真实 provider 之后
    它就成了报告里的假话，而这套东西的立场恰恰是「数字必须自己声明它测的是什么」。
    """
    economy = _LIVE_ECONOMY_NOTE if execution_mode == EXECUTION_MODE_LIVE_MODEL else _ORACLE_ECONOMY_NOTE
    return (_BASE_NOTE, economy)


def index_for_benchmark_artifact(artifact, workspace_root):
    """把基准跑出来的 N 份独立仓库工件合并成一个索引。

    基准给每个任务复制一份 fixture 仓库，`.codingforme/runs` 因此是散的；
    跨任务分析必须先合并。
    """
    workspace_root = Path(workspace_root)
    indexes = []
    for row in artifact.get("rows", []):
        relpath = str(row.get("fixture_copy_relpath", "")).strip()
        if not relpath:
            continue
        task_root = workspace_root / relpath
        indexes.append(
            TraceIndex.load(
                task_root / ".codingforme" / "runs",
                task_root / ".codingforme" / "sessions",
            )
        )
    return TraceIndex.merge(indexes)


def cases_from_benchmark_artifact(artifact, repeat_index=None):
    """把一次跑批的 rows 变成 L2 cases。

    `repeat_index` 只在重复跑批时给：case id 必须逐条唯一，否则 k 轮的同一个任务
    会互相覆盖，聚合出来的分母也不对。单轮时保持原样不加后缀，老结果照旧可比。
    """
    cases = []
    for row in artifact.get("rows", []):
        case_id = row["id"] if repeat_index is None else f"{row['id']}#r{repeat_index}"
        cases.append(
            build_case_result(
                case_id,
                LEVEL_TASK,
                bool(row.get("passed")),
                axis_values={
                    AXIS_CAPABILITY: 1.0 if row.get("verifier_passed") else 0.0,
                    AXIS_EFFICIENCY: int(row.get("tool_steps", 0) or 0),
                },
                detail={
                    "task_id": row["id"],
                    "repeat_index": repeat_index,
                    "budget_enforced": bool(row.get("budget_enforced", True)),
                    "step_ceiling_hit": bool(row.get("step_ceiling_hit")),
                    "category": row.get("category", ""),
                    "failure_category": row.get("failure_category"),
                    "stop_reason": row.get("stop_reason", ""),
                    "within_budget": bool(row.get("within_budget")),
                    "verifier_passed": bool(row.get("verifier_passed")),
                    "attempts": int(row.get("attempts", 0) or 0),
                },
                trace_ref={
                    "run_id": row.get("run_id", ""),
                    "run_dir_relpath": row.get("run_dir_relpath", ""),
                },
            )
        )
    return cases


def run_benchmark_suite(
    harness=None,
    benchmark_path=DEFAULT_BENCHMARK_PATH,
    artifact_path=None,
    workspace_root=None,
    result_path=None,
    markdown_path=None,
    model_client_factory=None,
    step_budget_override=None,
    max_new_tokens=None,
    repeats=1,
):
    harness = harness or DEFAULT_HARNESS
    evaluator_kwargs = {}
    if max_new_tokens:
        evaluator_kwargs["max_new_tokens"] = int(max_new_tokens)
    # 真实模型跑批时把模型名透传下去。不传的话 `BenchmarkEvaluator` 会用默认值
    # `FakeModelClient`，于是一次 live-model 跑批的工件里 `model` 字段写着
    # "FakeModelClient"——事后没人能从工件判断这批数字是哪个模型产生的。
    # （已确认 C1、B4 等既有 live 工件都带着这个错，它们的 model 字段不可信。）
    factory_model_name = getattr(model_client_factory, "model_name", "")
    if factory_model_name:
        evaluator_kwargs["model_name"] = str(factory_model_name)
    repeats = max(1, int(repeats or 1))
    base_artifact_path = Path(artifact_path) if artifact_path else Path("artifacts") / f"benchmark-{harness.name}.json"

    artifacts = []
    indexes = []
    cases = []
    aborted_repeats = []
    for repeat_index in range(1, repeats + 1):
        # 每轮各自一份工件与工作区：共用会让后一轮把前一轮的证据覆盖掉，
        # 而可靠性这层要的恰恰是「同一题的 k 份独立证据」。
        if repeats > 1:
            run_artifact_path = base_artifact_path.with_name(
                f"{base_artifact_path.stem}-r{repeat_index}{base_artifact_path.suffix}"
            )
            run_workspace_root = Path(workspace_root) / f"r{repeat_index}" if workspace_root else None
        else:
            run_artifact_path = base_artifact_path
            run_workspace_root = workspace_root
        evaluator = BenchmarkEvaluator(
            benchmark_path=benchmark_path,
            artifact_path=run_artifact_path,
            workspace_root=run_workspace_root,
            model_client_factory=model_client_factory,
            harness=harness,
            step_budget_override=step_budget_override,
            **evaluator_kwargs,
        )
        try:
            artifact = evaluator.run()
        except Exception as exc:
            # 一轮跑挂了不能连累已经跑完的那几轮。
            #
            # 踩过的坑：一次 2.4 小时的 k=3 跑批，r1、r2 各约 50 分钟都跑完了，
            # r3 在第 10 个任务上撞到服务端 `HTTP 400 / Connection refused`，异常
            # 一路上抛——而聚合发生在这个循环**之后**，于是两轮完整数据留在磁盘上
            # 却没有任何结果文件，只能靠临时脚本离线救回。
            #
            # 刻意只在**轮**这一层兜，不在**任务**那一层兜：任务级的异常大多是
            # provider 故障，把它记成「这个任务没通过」会把基础设施故障混进能力
            # 指标里，比直接失败更糟。这一轮的证据（trace）本来就已落盘，需要时
            # 可以离线补算。
            aborted_repeats.append({"repeat": repeat_index, "error": f"{type(exc).__name__}: {exc}"})
            print(
                f"[repeat {repeat_index}/{repeats}] aborted: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        artifacts.append(artifact)
        indexes.append(index_for_benchmark_artifact(artifact, evaluator.workspace_root))
        cases.extend(
            cases_from_benchmark_artifact(artifact, repeat_index=repeat_index if repeats > 1 else None)
        )

    if not artifacts:
        # 一轮都没跑成就没有可聚合的东西，这时必须抛——静默产出一份空报告
        # 比崩掉更危险。
        raise RuntimeError(
            "every repeat aborted; no benchmark artifact was produced: "
            + "; ".join(item["error"] for item in aborted_repeats)
        )

    artifact = artifacts[0]
    index = indexes[0] if len(indexes) == 1 else TraceIndex.merge(indexes)

    # L1 直接吃同一份索引：轨迹层不需要重跑任务，它评的是刚才那批运行的过程。
    assertions = score_index(index, harness)
    # 数据集声明的「这个任务必然会让某条断言挂掉」按 run_id 落到具体断言上。
    # 映射走 rows 而不是 trace：只有 rows 同时知道基准任务 id 和它跑出来的 run_id。
    expected_by_run = {
        str(row.get("run_id", "")): list(row.get("expected_trajectory_failures", []) or [])
        for artifact_item in artifacts
        for row in artifact_item.get("rows", [])
        if row.get("expected_trajectory_failures")
    }
    assertions, unobserved_expected = mark_expected_failures(assertions, expected_by_run)
    cases = cases + trajectory_cases(index, harness, assertions=assertions)
    aggregates = aggregate_cases(cases)
    aggregates["trajectory"] = summarize_assertions(assertions)
    # 声明了、但这次没触发的那些。oracle-replay 下参考解确定性，非空即声明过期；
    # live 下非空只说明模型这次没踩，属正常。
    aggregates["trajectory"]["expected_failures_not_observed"] = unobserved_expected
    # 可靠性：k 轮的同一批任务放在一起看「稳不稳」。k=1 时照样产出，
    # 但 render 会明说这一层不成立——不产出会让读者以为忘了算。
    aggregates["reliability"] = summarize_reliability([a.get("rows", []) for a in artifacts])
    # 成本：数据一直在 trace 的 completion_metadata 里，此前从没进过报告。
    aggregates["economy"] = summarize_economy(index)
    # 工具用量：哪些工具真的被调用过、run_plan 省下多少上下文。不记的话
    # 「机制生效了」和「机制一次都没触发」在工件上分不出来（见 tool_usage.py）。
    aggregates["tool_usage"] = summarize_tool_usage(index)

    result = build_eval_result(
        suite="fixed-benchmark",
        harness=harness,
        dataset={
            "source": artifact["benchmark"]["source"],
            "task_count": artifact["benchmark"]["task_count"],
            "fixture_snapshot_id": artifact["reproducibility"]["fixture_snapshot_id"],
        },
        cases=cases,
        run_context=build_run_context(
            mode="scripted" if model_client_factory is None else "custom",
            # 谁解的题由 evaluator 说了算：没注入 model_client_factory 就是回放参考解。
            execution_mode=evaluator.execution_mode,
            model=artifact["reproducibility"]["model_name"],
            repo_root=evaluator.repo_root,
            workspace_root=evaluator.workspace_root,
            # 实际给 agent 的步数上限。不记下来的话报告只能显示 HarnessSpec 声明的
            # max_steps，而那是**被覆盖前**的值——读者会看到"步数上限 6"配上一张
            # 步数 13、15、16 的表，只能得出"数据自相矛盾"的结论。
            extra={
                "step_budget_override": int(step_budget_override) if step_budget_override else None,
                # 请求了几轮、实际完成几轮、哪几轮中止。不记的话一份 k=2 的结果
                # 和「本来就只要 k=2」在工件层面长得一模一样。
                "repeats_requested": repeats,
                "repeats_completed": len(artifacts),
                "aborted_repeats": aborted_repeats,
            },
        ),
        trace_summary=index.to_dict(),
        aggregates=aggregates,
        notes=suite_notes(evaluator.execution_mode),
    )

    if result_path:
        write_eval_result(result_path, result)
    if markdown_path:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_eval_result_markdown(result), encoding="utf-8")
    return result
