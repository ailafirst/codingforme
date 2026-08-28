"""统一评测结果 schema。

为什么存在：
此前每类实验各写一份 JSON 形状（`artifact_type` 各不相同、字段名互不兼容），
结果是没法把两次评测放在一起比，也没法回答「这个数字是在什么条件下测出来的」。

这里的 schema 参照公开评测结果统一化的通行做法，固定六个顶层块：

    run_context   在什么环境跑的（版本、commit、时间、模式）
    harness       被测对象是谁（HarnessSpec 及其指纹）
    dataset       评的是哪份数据
    cases         逐条结果
    aggregates    聚合结果，按「层 / 轴」分组
    trace         底层工件的覆盖情况（来自 TraceIndex.coverage()）

任何新增评测都往这个形状里填，而不是再发明一种 artifact_type。
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

EVAL_RESULT_SCHEMA_VERSION = 1

# 评估对象的五个层级。指标必须声明自己属于哪一层，
# 否则聚合出来的数字说不清是在评什么。
LEVEL_COMPONENT = "L0-component"
LEVEL_TRAJECTORY = "L1-trajectory"
LEVEL_TASK = "L2-task"
LEVEL_SESSION = "L3-session"
LEVEL_SYSTEM = "L4-system"
LEVELS = (LEVEL_COMPONENT, LEVEL_TRAJECTORY, LEVEL_TASK, LEVEL_SESSION, LEVEL_SYSTEM)

# 五条正交的轴。同一层的指标要在这五个角度上都有说法，
# 只报能力（准确率）而不报代价的评测是没有信息量的。
AXIS_CAPABILITY = "capability"
AXIS_EFFICIENCY = "efficiency"
AXIS_ECONOMY = "economy"
AXIS_SAFETY = "safety"
AXIS_RELIABILITY = "reliability"
AXES = (AXIS_CAPABILITY, AXIS_EFFICIENCY, AXIS_ECONOMY, AXIS_SAFETY, AXIS_RELIABILITY)

# 这批任务是**谁**解的。区分它是这套结果里最容易被误读的一件事：
#   oracle-replay  回放数据集自带的参考解（对齐 Terminal-Bench 的四件套里那份
#                  reference solution）。任务不是被测系统自己解的。
#   live-model     真实模型自己解，被测的才是完整链路。
# 这个字段存在的理由：让报出来的数字**自己声明它测的是什么**。
EXECUTION_MODE_ORACLE_REPLAY = "oracle-replay"
EXECUTION_MODE_LIVE_MODEL = "live-model"
EXECUTION_MODES = (EXECUTION_MODE_ORACLE_REPLAY, EXECUTION_MODE_LIVE_MODEL)

# 参考解回放模式下必须随结果一起呈现的口径声明。
# 不是免责声明，是判定边界：不写清楚的话，capability 轴的数字会被读成
# 「这套 harness 的能力」，而它实际评的是数据集里那段固定脚本。
ORACLE_REPLAY_CAVEAT = (
    "本次结果由**参考解回放**（oracle-replay）产生：任务不是被测系统自己解的，"
    "而是回放数据集自带的参考解。因此这些数字的归属必须分开读——",
    "**capability 轴评的是参考解脚本**，不是 harness，也不是模型。"
    "典型表现：`read_before_patch` 偏低是参考解第一步就直接改文件，与 harness 无关。",
    "**safety / efficiency / reliability 三条轴评的才是 harness 本身**："
    "约束有没有被遵守、步数有没有超预算、是否走到终态、协议有没有漂移。",
    "**economy 轴无数据**：回放不产生 token 用量，成本类指标必须由真实 provider 跑批。",
    "换成 live-model 跑批之后，capability 轴才开始度量能力。",
)


def _git_value(args, fallback="", cwd=None):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd or Path.cwd(),
            capture_output=True,
            text=True,
            # 同 evaluator._git_value：不指定编码会按宿主 ANSI 代码页解码 git 输出。
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or fallback
    except Exception:
        return fallback


def enclosing_git_root(path):
    """`path` 落在哪个 git 仓库里（不在任何仓库里则返回空串）。

    踩过坑：评测工作区一旦落在某个仓库目录下，`workspace.py` 的快照就会被悄悄
    放大成那个外层仓库，一整次 12 任务 x 3 轮的跑批因此作废——而当时的工件里
    根本看不出这批数据跑在哪，只能靠人肉复盘。

    走文件系统而不是 `git rev-parse`：临时工作区在写报告时可能已经被清掉，
    子进程的 cwd 会直接失败，而它的父目录仍然在，答案照样算得出来。
    """
    try:
        current = Path(path).resolve()
    except OSError:
        return ""
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return str(candidate)
    return ""


def build_run_context(
    mode="scripted",
    model="",
    provider="",
    repo_root=None,
    workspace_root=None,
    extra=None,
    execution_mode=EXECUTION_MODE_ORACLE_REPLAY,
):
    execution_mode = str(execution_mode)
    if execution_mode not in EXECUTION_MODES:
        raise ValueError(
            f"unknown execution_mode: {execution_mode!r} (known: {', '.join(EXECUTION_MODES)})"
        )
    context = {
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "commit_sha": _git_value(["rev-parse", "HEAD"], cwd=repo_root),
        "branch": _git_value(["branch", "--show-current"], cwd=repo_root),
        # scripted = FakeModelClient 确定性回放；real = 真实 provider
        "mode": str(mode),
        # 谁解的题。mode 说的是「模型客户端是什么」，execution_mode 说的是
        # 「这批任务的解法从哪来」——后者才决定数字该怎么读。
        "execution_mode": execution_mode,
        "model": str(model),
        "provider": str(provider),
        # 这批任务在哪个目录里跑的，以及那个目录是不是落在某个 git 仓库内部。
        # 后者恒为空串才是正常的；非空说明工作区嵌在仓库里，快照会被放大，
        # 这份数据多半不能和别的跑批相比。见 enclosing_git_root()。
        "workspace_root": str(Path(workspace_root).resolve()) if workspace_root else "",
        "workspace_git_root": enclosing_git_root(workspace_root) if workspace_root else "",
    }
    context.update(dict(extra or {}))
    return context


def build_case_result(case_id, level, passed, *, axis_values=None, detail=None, trace_ref=None):
    """一条 case 的结果。

    axis_values 是这条 case 在五条轴上的观测值（缺哪条就不填），
    detail 放判定依据，trace_ref 指回产生它的运行工件。
    """
    level = str(level)
    if level not in LEVELS:
        raise ValueError(f"unknown eval level: {level!r} (known: {', '.join(LEVELS)})")
    unknown_axes = sorted(set(axis_values or {}) - set(AXES))
    if unknown_axes:
        raise ValueError(f"unknown eval axes: {', '.join(unknown_axes)}")
    return {
        "case_id": str(case_id),
        "level": level,
        "passed": bool(passed),
        "axis_values": dict(axis_values or {}),
        "detail": dict(detail or {}),
        "trace_ref": dict(trace_ref or {}),
    }


def aggregate_cases(cases):
    """按层、按轴聚合通过率。

    by_axis 统计的是「声明了这条轴的 case 有多少、通过多少」，因此它同时是一份
    **覆盖度**报告：某条轴的 total 为 0，说明这次评测在那个角度上根本没有证据，
    而不是表现完美。只报能力不报代价的评测没有信息量，这个空缺必须看得见。
    """
    cases = list(cases)
    by_level = {}
    by_axis = {}
    for case in cases:
        bucket = by_level.setdefault(case["level"], {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += 1 if case["passed"] else 0
        for axis in case.get("axis_values", {}):
            axis_bucket = by_axis.setdefault(axis, {"total": 0, "passed": 0})
            axis_bucket["total"] += 1
            axis_bucket["passed"] += 1 if case["passed"] else 0
    for bucket in list(by_level.values()) + list(by_axis.values()):
        bucket["pass_rate"] = (bucket["passed"] / bucket["total"]) if bucket["total"] else 0.0
    total = len(cases)
    passed = sum(1 for case in cases if case["passed"])
    return {
        "total": total,
        "passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "by_level": by_level,
        "by_axis": by_axis,
    }


def build_eval_result(
    suite,
    harness,
    dataset,
    cases,
    *,
    run_context=None,
    trace_summary=None,
    aggregates=None,
    notes=None,
):
    cases = list(cases)
    harness_payload = harness.to_dict() if hasattr(harness, "to_dict") else dict(harness)
    harness_payload["fingerprint"] = harness.fingerprint() if hasattr(harness, "fingerprint") else ""
    # 单独也存一份：指纹变了的时候，这个字段能立刻回答「变的是配置还是代码」。
    harness_payload["code_signature"] = (
        harness.code_signature() if hasattr(harness, "code_signature") else ""
    )
    # 分项再存一份。合并成一个哈希之后，「变了」和「哪里变了」就分开了；两份工件
    # 的签名对不上时，没有分项就只能靠 git 逐个模块比对，而工件往往比源码活得久。
    if hasattr(harness, "code_signature_parts"):
        harness_payload["code_signature_parts"] = dict(harness.code_signature_parts())
    return {
        "schema_version": EVAL_RESULT_SCHEMA_VERSION,
        "suite": str(suite),
        "run_context": dict(run_context or build_run_context()),
        "harness": harness_payload,
        "dataset": dict(dataset or {}),
        "cases": cases,
        "aggregates": dict(aggregates) if aggregates is not None else aggregate_cases(cases),
        "trace": dict(trace_summary or {}),
        "notes": list(notes or []),
    }


def write_eval_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _step_cap_text(harness, context):
    """报告里"步数上限"那一格。

    HarnessSpec 声明的 `max_steps` 未必是这次真正给 agent 的上限：固定基准跑批
    会按任务覆盖它（见 `scripts/run_eval_suite.py` 的 `DEFAULT_STEP_BUDGET`）。
    只显示声明值会让整张表读不通——"步数上限 6"配上一列 13/15/16 的实际步数。
    覆盖生效时把两个数都写出来，并说明哪个是哪个。
    """
    declared = harness.get("max_steps", "")
    override = context.get("step_budget_override")
    if not override:
        return str(declared)
    return f"{override}（本次跑批覆盖；变体声明 {declared}）"


def render_eval_result_markdown(result):
    aggregates = result.get("aggregates", {})
    harness = result.get("harness", {})
    context = result.get("run_context", {})
    coverage = (result.get("trace") or {}).get("coverage", {})
    lines = [
        f"# 评测结果：{result.get('suite', '')}",
        "",
        "## 运行上下文",
        f"- 采集时间：{context.get('captured_at', '')}",
        f"- commit：{context.get('commit_sha', '')[:12]} ({context.get('branch', '')})",
        f"- 模式：{context.get('mode', '')}",
        f"- 执行模式：{context.get('execution_mode', '')}",
    ]
    if context.get("workspace_root"):
        lines.append(f"- 评测工作区：{context['workspace_root']}")
    if context.get("workspace_git_root"):
        # 排在所有数字之前：工作区嵌在 git 仓库里会让 workspace.py 的快照放大到
        # 外层仓库，这批数字和别的跑批不可比，读者必须先知道这件事。
        lines.extend([
            "",
            f"> ⚠ **评测工作区落在 git 仓库 `{context['workspace_git_root']}` 内部**："
            "仓库快照会被放大成该仓库，本次数据与工作区在仓库之外的跑批不可比。",
        ])
    aborted = context.get("aborted_repeats") or []
    if aborted:
        # 中止的轮次必须出现在**所有数字之前**：一份 k=2 的结果和「本来就只要
        # k=2」在数字上长得一模一样，读者要先知道这批数据缺了什么。
        lines.extend([
            "",
            f"> ⚠ **请求 {context.get('repeats_requested', '?')} 轮，实际完成 "
            f"{context.get('repeats_completed', '?')} 轮**；下列轮次中止，其数据未计入本报告：",
        ])
        lines.extend(f"> - 第 {item.get('repeat')} 轮：{item.get('error', '')}" for item in aborted)
    if context.get("execution_mode") == EXECUTION_MODE_ORACLE_REPLAY:
        # 口径声明紧跟运行上下文，排在所有数字**之前**：
        # 放在末尾等于默认读者会读到最后，而实际上没人这么读表。
        lines.extend(["", "## 这些数字测的是什么"])
        lines.extend(f"- {line}" for line in ORACLE_REPLAY_CAVEAT)
    lines.extend([
        "",
        "## 被测 harness",
        f"- 变体：`{harness.get('name', '')}` — {harness.get('description', '')}",
        # 两个签名都列：指纹变了的时候要能立刻分辨「改的是配置」还是「改的是
        # 提示词/工具 schema」。code 签名相同 = 两次跑批用的是同一套提示词与工具定义。
        f"- 指纹：{harness.get('fingerprint', '')[:19]}"
        + (f" · 代码签名：{harness.get('code_signature', '')[:19]}" if harness.get("code_signature") else ""),
        f"- 审批策略：{harness.get('approval_policy', '')} · 只读：{harness.get('read_only', False)} · 步数上限：{_step_cap_text(harness, context)}",
        f"- feature flags：{harness.get('feature_flags', {})}",
        "",
        "## 结果",
        f"- 总计 {aggregates.get('total', 0)} 条，通过 {aggregates.get('passed', 0)} 条，通过率 {aggregates.get('pass_rate', 0.0):.2%}",
    ])
    by_level = aggregates.get("by_level", {})
    if by_level:
        lines.extend(["", "| 层级 | 通过 / 总数 | 通过率 |", "|---|---|---|"])
        for level in LEVELS:
            bucket = by_level.get(level)
            if not bucket:
                continue
            lines.append(f"| {level} | {bucket['passed']} / {bucket['total']} | {bucket['pass_rate']:.2%} |")
    by_axis = aggregates.get("by_axis")
    if by_axis is not None:
        # 五条轴全部列出：没有数据的那条要显式写成「未覆盖」，
        # 否则一张只剩三行的表会被读成「另外两条没问题」。
        lines.extend(["", "| 轴 | 通过 / 总数 | 通过率 |", "|---|---|---|"])
        economy = aggregates.get("economy") or {}
        for axis in AXES:
            bucket = by_axis.get(axis)
            if not bucket:
                # economy 是成本轴，没有「通过」这个概念，所以它不进 by_axis 的
                # 通过率统计。但只要 trace 里有 usage，它就**不是**未覆盖——
                # 以前这里一律写「未覆盖」，而数据其实躺在 completion_metadata 里。
                if axis == AXIS_ECONOMY and economy.get("turns_with_usage"):
                    lines.append(
                        f"| {axis} | {economy['input_tokens']:,} + {economy['output_tokens']:,} token "
                        "| 见下方「成本」一节 |"
                    )
                    continue
                lines.append(f"| {axis} | — | 未覆盖 |")
                continue
            lines.append(f"| {axis} | {bucket['passed']} / {bucket['total']} | {bucket['pass_rate']:.2%} |")
    reliability = aggregates.get("reliability")
    if reliability and reliability.get("task_count"):
        # 排在断言明细**之前**：k 轮跳变的通过率会让下面每一个百分比都失去意义，
        # 读者必须先知道这批数字稳不稳，再去读它们是多少。
        from .reliability import render_reliability_markdown

        lines.extend(["", render_reliability_markdown(reliability).rstrip()])
    economy_summary = aggregates.get("economy")
    if economy_summary:
        # 成本紧跟可靠性：两者都是「读能力数字之前该先知道的背景」。
        from .economy import render_economy_markdown

        lines.extend(["", render_economy_markdown(economy_summary).rstrip()])
    tool_usage = aggregates.get("tool_usage")
    if tool_usage:
        # 紧跟成本：「省了多少上下文」和「花了多少 token」是同一个问题的两面。
        from .tool_usage import render_tool_usage_markdown

        lines.extend(["", render_tool_usage_markdown(tool_usage).rstrip()])
    trajectory = aggregates.get("trajectory")
    if trajectory:
        lines.extend(
            [
                "",
                # 同一个块服务 L1 轨迹断言和 L3 会话断言，标题保持中性。
                "## 断言明细",
                # 三个桶而不是通过率一个数：数据集声明「这个任务允许挂」的那几条
                # （故意越界、故意缺必填参数）既不该拉低通过率，也不该被藏起来。
                f"- 断言 {trajectory.get('total', 0)} 条：通过 {trajectory.get('passed', 0)}"
                f" · 预期内失败 {trajectory.get('expected_failures', 0)}"
                f" · **预期外失败 {trajectory.get('unexpected_failures', 0)}**"
                f"（把预期内失败算作通过后为 {trajectory.get('clean_rate', trajectory.get('pass_rate', 0.0)):.2%}）",
            ]
        )
        by_subject = trajectory.get("by_subject") or {}
        if by_subject:
            lines.extend(
                [
                    "",
                    "按「判的是谁的行为」分：`model` 是模型自己发的调用序列"
                    "（回放模式下即参考解脚本），`harness` 是闸口 / 预算 / 上下文组装。",
                    "",
                    "| 判的是谁 | 通过 / 适用 | 通过率 |",
                    "|---|---|---|",
                ]
            )
            for subject, bucket in sorted(by_subject.items()):
                lines.append(
                    f"| `{subject}` | {bucket['passed']} / {bucket['total']} | {bucket['pass_rate']:.2%} |"
                )
        lines.extend(
            [
                "",
                "| 断言 | 类别 | 轴 | 判的是谁 | 通过 / 适用 | 通过率 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for assertion_id, bucket in trajectory.get("by_assertion", {}).items():
            lines.append(
                f"| `{assertion_id}` | {bucket.get('check_class', '')} | {bucket.get('axis', '')} "
                f"| {bucket.get('subject', '')} "
                f"| {bucket['passed']} / {bucket['total']} | {bucket['pass_rate']:.2%} |"
            )
        unobserved = trajectory.get("expected_failures_not_observed") or []
        if unobserved:
            lines.extend(
                [
                    "",
                    "数据集声明了、但这次没触发的失败（回放模式下出现即声明过期）：",
                ]
            )
            lines.extend(
                f"- `{item.get('assertion_id')}` @ {item.get('run_id')}" for item in unobserved[:10]
            )
        failures = trajectory.get("failures", [])
        if failures:
            lines.extend(["", "失败归因（前 10 条）："])
            for failure in failures[:10]:
                tag = "（预期内）" if failure.get("expected_failure") else ""
                lines.append(
                    f"- `{failure['assertion_id']}`{tag} @ {failure['run_id']} — {failure.get('detail', {})}"
                )
    if coverage:
        lines.extend(
            [
                "",
                "## trace 覆盖",
                f"- 会话 {coverage.get('session_count', 0)} 条 / 运行 {coverage.get('run_count', 0)} 次 / 轮次 {coverage.get('turn_count', 0)} 轮",
                f"- 带 session_id 的运行：{coverage.get('runs_with_session_id', 0)} / {coverage.get('run_count', 0)}",
                f"- 多轮会话（run_count > 1）：{coverage.get('multi_run_sessions', 0)}",
            ]
        )
    notes = result.get("notes", [])
    if notes:
        lines.extend(["", "## 备注"])
        lines.extend(f"- {note}" for note in notes)
    lines.append("")
    return "\n".join(lines)
