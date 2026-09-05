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
    "transcript_tokens": 0,
    "saved_tokens": 0,
    "filtered_plans": 0,
    "clipped_plans": 0,
}


def _plan_usage(index):
    """`run_plan` 专属口径，全部取自 `plan_executed` 事件里已落盘的字段。

    `saved_tokens` = 内层结果原始总字节 − 真正回给模型的 token 数。没有这个差值，
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
            usage["transcript_tokens"] += int(event.get("plan_transcript_tokens", 0) or 0)
            if event.get("plan_results_echoed") is False:
                usage["filtered_plans"] += 1
            if event.get("plan_transcript_clipped"):
                usage["clipped_plans"] += 1
    usage["runs_with_plan"] = len(runs_with_plan)
    usage["saved_tokens"] = usage["result_bytes"] - usage["transcript_tokens"]
    usage["inner_calls"] = sum(
        1
        for run in index.runs
        for turn in run.turns
        for call in turn.tools
        if call.get("via") == "run_plan"
    )
    return usage


EMPTY_SPILL_USAGE = {
    "spilled_calls": 0,
    "runs_with_spill": 0,
    "full_tokens": 0,
    "kept_tokens": 0,
    "saved_tokens": 0,
    # 落盘该发生却抛异常的次数。**必须和 `spilled_calls` 分开数**：失败那支退回
    # 普通截断，于是在工件上和「结果本来就没超上限」长得一模一样，而含义相反——
    # 一个是机制没必要跑，一个是机制跑挂了、模型手里那份结果被截断且不可恢复。
    # `failed_tokens` 记的是这些失败本来要落盘多大一份，也就是静默丢了多少。
    "failed_calls": 0,
    "failed_tokens": 0,
    "failures": [],
    "by_tool": {},
}


def _spill_usage(index):
    """阶段三 L1（超长工具结果落盘）到底触发了几次、省下多少。

    和 `run_plan` 那一块是同一个理由:没有这几个数,「机制天天生效」和「一次都
    没触发」在报告里长得一模一样,而这两件事的下一步完全相反——前者要看省下的
    量值不值这份复杂度,后者要看是不是上限设得根本够不着（工具结果的入口上限
    已经从写死的 1,320 换成 `clamp(total_budget // 8, 1320, 25000)`，1M 档 =
    14,791，能触发落盘的负载因此少了一个数量级）。

    `saved_tokens` = 原始 token − 留在上下文里的 token，口径和 `plan.saved_tokens`
    一致。**注意它不是净省**：全文落了盘，模型可以再花一次 `read_file` 把它读回来。
    """
    usage = {key: (dict(value) if isinstance(value, dict) else value) for key, value in EMPTY_SPILL_USAGE.items()}
    runs_with_spill = set()
    for run in index.runs:
        for turn in run.turns:
            for call in turn.tools:
                if call.get("tool_output_spill_failed"):
                    usage["failed_calls"] += 1
                    usage["failed_tokens"] += int(call.get("tool_output_full_tokens", 0) or 0)
                    if len(usage["failures"]) < 20:
                        usage["failures"].append({
                            "run_id": run.run_id,
                            "turn": turn.turn,
                            "name": str(call.get("name") or ""),
                            "full_tokens": int(call.get("tool_output_full_tokens", 0) or 0),
                            "error": str(call.get("tool_output_spill_error") or ""),
                        })
                if not call.get("tool_output_spilled"):
                    continue
                name = str(call.get("name") or "")
                usage["spilled_calls"] += 1
                usage["full_tokens"] += int(call.get("tool_output_full_tokens", 0) or 0)
                usage["kept_tokens"] += int(call.get("tool_output_kept_tokens", 0) or 0)
                usage["by_tool"][name] = usage["by_tool"].get(name, 0) + 1
                runs_with_spill.add(run.run_id)
    usage["runs_with_spill"] = len(runs_with_spill)
    usage["saved_tokens"] = usage["full_tokens"] - usage["kept_tokens"]
    usage["by_tool"] = dict(sorted(usage["by_tool"].items()))
    return usage


EMPTY_DELEGATE_USAGE = {
    "delegations": 0,
    "runs_with_delegate": 0,
    "child_tool_steps": 0,
    "child_transcript_tokens": 0,
    "result_tokens": 0,
    "saved_tokens": 0,
    "child_input_tokens": 0,
    "child_stop_reasons": {},
}


def _delegate_usage(index):
    """受限委派（`delegate`）触发了几次、把多少上下文挡在了主上下文之外。

    这是四层上下文治理里「不让内容进来」那一层唯一的机制，而它在此前六批 live
    跑批里的调用数**恒为 0**——不是因为不好用，是因为每个任务的 `allowed_tools`
    都把它裁掉了，而它当时还不是元工具。所以这一块的零值特别容易被误读成
    「这个机制没问题」，渲染时必须明说是「一次都没执行」。

    `saved_tokens` = 子 agent 自己积累的转录 − 主上下文实际收到的结论，口径和
    `plan.saved_tokens`、`spill.saved_tokens` 一致。**它同样不是净省**：子 agent
    那几轮模型往返照样要花钱，省下的只有主上下文的**长度**。
    """
    usage = {key: (dict(value) if isinstance(value, dict) else value) for key, value in EMPTY_DELEGATE_USAGE.items()}
    # 子 agent 是索引里的独立 run，所以它烧掉的输入 token 拿 run_id 就能对上。
    # 这一项必须和 `saved_tokens` 并排放：省下的是**主上下文的长度**，花掉的是
    # 子 agent 那几轮真实的输入 token，两者不同量纲也不同方向。只报前者，就会
    # 把一次「省了 6 千、多花 22 万」的交易读成纯收益。
    input_by_run = {run.run_id: sum(turn.input_tokens for turn in run.turns) for run in index.runs}
    runs = set()
    for run in index.runs:
        for turn in run.turns:
            for call in turn.tools:
                if str(call.get("name") or "") != "delegate":
                    continue
                if not call.get("delegate_child_run_id"):
                    # 深度耗尽、参数不合法这类调用不会跑出子 agent，没有记账可言。
                    continue
                usage["delegations"] += 1
                runs.add(run.run_id)
                usage["child_tool_steps"] += int(call.get("delegate_child_tool_steps", 0) or 0)
                usage["child_transcript_tokens"] += int(call.get("delegate_child_transcript_tokens", 0) or 0)
                usage["result_tokens"] += int(call.get("delegate_result_tokens", 0) or 0)
                usage["child_input_tokens"] += int(input_by_run.get(str(call.get("delegate_child_run_id") or ""), 0))
                reason = str(call.get("delegate_child_stop_reason") or "")
                usage["child_stop_reasons"][reason] = usage["child_stop_reasons"].get(reason, 0) + 1
    usage["runs_with_delegate"] = len(runs)
    usage["saved_tokens"] = usage["child_transcript_tokens"] - usage["result_tokens"]
    usage["child_stop_reasons"] = dict(sorted(usage["child_stop_reasons"].items()))
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
        "spill": _spill_usage(index),
        "delegate": _delegate_usage(index),
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

    spill = summary.get("spill") or {}
    lines.extend(["", "### 超长工具结果落盘（阶段三 L1）", ""])
    if not spill.get("spilled_calls"):
        # 零值也要写出来:省略这一节会被读成「这块没问题」，而实际含义是
        # 「这批负载里没有一次工具输出大到需要落盘」——那是关于负载的结论，
        # 不是关于机制的结论。
        lines.append(
            "- **一次都没触发**。单条上限是 `clamp(total_budget // 8, 1320, 25000)`，"
            "这批任务的工具输出全都装得下。要量收益得用会刷屏的负载（大仓库上的 "
            "`search`、几万行的 `run_shell` 输出）。"
        )
    else:
        by_tool = "、".join(f"`{name}` {count} 次" for name, count in spill.get("by_tool", {}).items())
        lines.append(
            f"- {spill['spilled_calls']} 次调用的结果落了盘（{by_tool}），"
            f"覆盖 {spill['runs_with_spill']} / {summary['runs_total']} 次运行"
        )
        lines.append(
            f"- **原始 {spill['full_tokens']:,} token → 留在上下文里 {spill['kept_tokens']:,}，"
            f"少进 {spill['saved_tokens']:,}**"
            "。这不是净省：全文在工作区里，模型可以再花一次 `read_file` 读回来。"
        )

    # 故障单独一段，排在用量之外：落盘失败不是「用了多少」，是「机制该接却接漏了」。
    # 零值也照写(只在触发过落盘时)，否则读者无从知道这个通道被查过。
    if spill.get("failed_calls"):
        lines.append(
            f"- ⚠️ **落盘失败 {spill['failed_calls']} 次**（本该落盘 "
            f"{spill['failed_tokens']:,} token，实际退回普通截断、内容不可恢复）："
            + "；".join(
                f"`{item['name']}` @ {item['run_id']} turn {item['turn']} — {item['error']}"
                for item in spill.get("failures", [])[:5]
            )
        )
    elif spill.get("spilled_calls"):
        lines.append("- 落盘失败 0 次（每一次超限都成功落了盘）")

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

    saved = plan["saved_tokens"]
    ratio = (plan["transcript_tokens"] / plan["result_bytes"]) if plan["result_bytes"] else 0.0
    lines.append(
        f"- 跑起来的那些覆盖 {plan['runs_with_plan']} / {summary['runs_total']} 次运行；"
        f"内层调用 {plan['inner_calls']} 次"
    )
    lines.append(
        f"- **内层结果 {plan['result_bytes']:,} 字节 → 实际回给模型 {plan['transcript_tokens']:,} token"
        f"（{ratio:.1%}），省下 {saved:,}**"
        "——差值就是没进下一轮 prompt 的那部分；没有这个数，「真省了」和「什么都没省」分不出来。"
    )
    lines.append(
        f"- 其中 {plan['filtered_plans']} 段用了 `print()` 走过滤模式"
        f"（其余全文回显），{plan['clipped_plans']} 段撞上 `MAX_TRANSCRIPT_TOKENS` 被裁剪"
    )
    lines.extend(_render_delegate_section(summary))
    return "\n".join(lines) + "\n"


def _render_delegate_section(summary):
    """受限委派那一节。零调用时明说「一次都没执行」，不省略——省略会被读成没问题。"""
    usage = summary.get("delegate") or {}
    lines = ["", "### 受限委派（`delegate`）", ""]
    if not usage.get("delegations"):
        lines.append(
            "- **一次都没执行**。它只在 `delegate_tool` 变体下进注册表；这个数为零时"
            "先确认跑的是不是那个变体，再谈机制本身。"
        )
        return lines
    saved = usage["saved_tokens"]
    lines.append(
        f"- 委派 {usage['delegations']} 次，覆盖 {usage['runs_with_delegate']} / "
        f"{summary['runs_total']} 次运行；子 agent 合计走了 {usage['child_tool_steps']} 个工具步"
    )
    lines.append(
        f"- **子上下文积累 {usage['child_transcript_tokens']:,} token → 主上下文只收到 "
        f"{usage['result_tokens']:,} token，挡在外面 {saved:,}**"
    )
    lines.append(
        f"- **代价：子 agent 自己烧掉 {usage['child_input_tokens']:,} 输入 token。**"
        "省下的是主上下文的**长度**，花掉的是真金白银的输入 token——两者不同量纲，"
        "这一行不写就会把交易读成纯收益。"
    )
    if usage.get("child_stop_reasons"):
        detail = "，".join(f"{reason or '(空)'} x{count}" for reason, count in usage["child_stop_reasons"].items())
        lines.append(f"- 子 agent 的终止原因：{detail}")
    return lines
