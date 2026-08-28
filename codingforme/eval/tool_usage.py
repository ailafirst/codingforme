"""这次跑批里每个工具真的被调用了几次，以及 run_plan 省下了多少上下文。

存在的理由是踩过坑：一次 12 任务 × 3 轮的 live 跑批做完之后，「`run_plan`
到底有没有被用过」在工件里查不到——评测工作区是临时目录、跑完就删，报告的
case detail 只记步数和终止原因、不记工具名。于是**「机制生效了」和「机制一次
都没触发」在报告里长得一模一样**，而这恰恰是决定要不要开启它的那个数。

和其它指标一样的约定：只跟 `TraceIndex` 的 `Session → Run → Turn` 三级结构
打交道，不直接解析文件；没数据时产出零值而不是省略字段——缺字段会被读成
「这块没问题」，零值才读得出「这块没发生」。

`run_plan` 的内层调用带 `via: "run_plan"`，所以它们既进 `by_tool`（那确实是
真的执行过的调用，各自计了一步、各自过了闸口），也单独在 `plan.inner_calls`
里数一遍。两个口径都要，因为它们回答的是不同的问题：前者是「这次跑批一共
做了多少事」，后者是「其中多少是编排出来的」。
"""

TOOL_USAGE_SCHEMA_VERSION = 1

# `run_tool()` 里被判成「真的执行了」的状态。其余（rejected / error）分开计，
# 因为「调用了 20 次 patch_file」和「发起 20 次、被闸口挡掉 18 次」是两回事。
_APPLIED_STATUSES = ("ok", "applied")

EMPTY_PLAN_USAGE = {
    "plans_attempted": 0,
    "plans_executed": 0,
    "plans_rejected": 0,
    "runs_with_plan": 0,
    "inner_calls": 0,
    "result_bytes": 0,
    "transcript_chars": 0,
    "saved_chars": 0,
    "filtered_plans": 0,
    "clipped_plans": 0,
}


def _plan_usage(index):
    """`run_plan` 专属口径，全部取自 `plan_executed` 事件里已落盘的字段。

    `saved_chars` = 内层结果原始总字节 − 真正回给模型的字符数。没有这个差值，
    「过滤真省了」和「过滤什么都没省」在工件上分不出来。

    **`attempted` 和 `executed` 必须分开数。** `plan_executed` 这条事件在计划被
    静态检查打回时**照样会写**（`run_tool()` 的校验失败也走同一条落盘路径），
    只是不带 `plan_calls` 字段。把两者混成一个数,「模型爱用这个工具」和「模型
    每次都写错、每次白烧一个约 17 秒的往返」就分不出来了——而这两件事的应对
    完全相反:前者该扩大适用面,后者该改语法白名单或工具描述。这个坑是本函数
    自己踩的:第一版只有 `plans_executed`,在一次 14 任务的 live 跑批上报出 13,
    实际真正跑起来的只有 7。
    """
    usage = dict(EMPTY_PLAN_USAGE)
    runs_with_plan = set()
    for run in index.runs:
        for event in run.events:
            if event.get("event") != "plan_executed":
                continue
            usage["plans_attempted"] += 1
            if event.get("plan_calls") is None:
                # 静态检查打回:一个工具都没执行,但确实烧掉了一个模型往返。
                usage["plans_rejected"] += 1
                continue
            runs_with_plan.add(run.run_id)
            usage["plans_executed"] += 1
            usage["result_bytes"] += int(event.get("plan_result_bytes", 0) or 0)
            usage["transcript_chars"] += int(event.get("plan_transcript_chars", 0) or 0)
            if event.get("plan_results_echoed") is False:
                usage["filtered_plans"] += 1
            if event.get("plan_transcript_clipped"):
                usage["clipped_plans"] += 1
    usage["runs_with_plan"] = len(runs_with_plan)
    usage["saved_chars"] = usage["result_bytes"] - usage["transcript_chars"]
    usage["inner_calls"] = sum(
        1
        for run in index.runs
        for turn in run.turns
        for call in turn.tools
        if call.get("via") == "run_plan"
    )
    return usage


def summarize_tool_usage(index):
    """把一次跑批的工具调用汇总成 `by_tool` + `plan` 两块。"""
    by_tool = {}
    runs_by_tool = {}
    calls_total = 0
    for run in index.runs:
        for turn in run.turns:
            for call in turn.tools:
                name = str(call.get("name") or "")
                if not name:
                    continue
                calls_total += 1
                row = by_tool.setdefault(name, {"calls": 0, "applied": 0, "rejected": 0, "error": 0, "runs": 0})
                row["calls"] += 1
                status = str(call.get("tool_status") or "")
                if status in _APPLIED_STATUSES:
                    row["applied"] += 1
                elif status == "rejected":
                    row["rejected"] += 1
                else:
                    row["error"] += 1
                runs_by_tool.setdefault(name, set()).add(run.run_id)
    for name, run_ids in runs_by_tool.items():
        by_tool[name]["runs"] = len(run_ids)
    return {
        "schema_version": TOOL_USAGE_SCHEMA_VERSION,
        "runs_total": len(index.runs),
        "calls_total": calls_total,
        "by_tool": dict(sorted(by_tool.items())),
        "plan": _plan_usage(index),
    }


def render_tool_usage_markdown(summary):
    """渲染成一节。零调用时明说，而不是省略这一节——省略会被读成「没问题」。"""
    if not summary or not summary.get("calls_total"):
        return (
            "## 工具用量\n\n"
            "- **无数据**：本次跑批没有记录到任何工具调用。\n"
        )

    lines = ["## 工具用量", ""]
    lines.append(
        f"- {summary['runs_total']} 次运行共 {summary['calls_total']} 次工具调用"
        "（`run_plan` 的内层调用各自计一次，和它们各自消耗一步、各自过闸口的语义一致）"
    )
    lines.extend(["", "| 工具 | 调用次数 | 执行成功 | 被闸口拒绝 | 出错 | 用到它的运行数 |", "|---|---|---|---|---|---|"])
    for name, row in summary.get("by_tool", {}).items():
        lines.append(
            f"| `{name}` | {row['calls']} | {row['applied']} | {row['rejected']} | {row['error']}"
            f" | {row['runs']} / {summary['runs_total']} |"
        )

    plan = summary.get("plan") or {}
    lines.extend(["", "### 受限编排（`run_plan`）", ""])
    if not plan.get("plans_attempted"):
        # 这一行就是加这个模块的全部理由：以前它在工件里根本不存在，
        # 于是「这个变体没开」和「开了但模型一次都没用」读起来一模一样。
        lines.append(
            "- **一次都没用过**。这不等于机制有问题：`run_plan` 只在 `plan_tool` 变体下"
            "进注册表，且串行任务本来就用不上它（Anthropic 在 τ²-bench 上的实测同样是"
            "「分数不变、成本还高约 8%」）。要量收益得用 fan-out 形状的任务。"
        )
        return "\n".join(lines) + "\n"

    # 写错的计划照样烧掉一个约 17 秒的模型往返,所以「写了几段」和「跑起来几段」
    # 要分开报——两者的应对完全相反:前者低说明工具没人用,后者低说明语法白名单
    # 或工具描述有问题。
    attempted = plan["plans_attempted"]
    rejected = plan["plans_rejected"]
    lines.append(
        f"- 模型写了 {attempted} 段计划,其中 **{plan['plans_executed']} 段真的跑起来了、"
        f"{rejected} 段被静态检查打回**（打回率 {rejected / attempted:.0%}，每次白烧一个模型往返）"
    )
    if not plan["plans_executed"]:
        lines.append("- **没有一段跑起来**：语法白名单或工具描述和模型的写法对不上,先看打回原因。")
        return "\n".join(lines) + "\n"

    saved = plan["saved_chars"]
    ratio = (plan["transcript_chars"] / plan["result_bytes"]) if plan["result_bytes"] else 0.0
    lines.append(
        f"- 跑起来的那些覆盖 {plan['runs_with_plan']} / {summary['runs_total']} 次运行；"
        f"内层调用 {plan['inner_calls']} 次"
    )
    lines.append(
        f"- **内层结果 {plan['result_bytes']:,} 字节 → 实际回给模型 {plan['transcript_chars']:,} 字符"
        f"（{ratio:.1%}），省下 {saved:,}**"
        "——差值就是没进下一轮 prompt 的那部分；没有这个数，「真省了」和「什么都没省」分不出来。"
    )
    lines.append(
        f"- 其中 {plan['filtered_plans']} 段用了 `print()` 走过滤模式"
        f"（其余全文回显），{plan['clipped_plans']} 段撞上 `MAX_TRANSCRIPT_CHARS` 被裁剪"
    )
    return "\n".join(lines) + "\n"
