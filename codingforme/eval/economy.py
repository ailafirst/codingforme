"""economy 轴：一次跑批花了多少 token。

为什么单独一个模块：这套评测的立场是「只报能力不报代价的评测没有信息量」，
而在此之前 economy 轴一直渲染成「未覆盖」——**尽管数据早就在 trace 里**。
k=3 基线上有 337 次模型调用带 usage、合计 73.8 万 input token，报告却一个数
都没报，同一份报告的备注还写着「economy 轴有数」，自相矛盾。

为什么不塞进 `by_axis`：`aggregate_cases()` 的 `by_axis` 是**通过率**表
（total / passed / pass_rate），而成本没有「通过」这个概念。硬套进去只会得到
一列和 capability 一模一样的通过率，看着像有数据，实际什么也没说。所以成本
单独成块，`by_axis` 那一行改成指向这里。
"""

# 回放模式（FakeModelClient）不产生 usage，这时整块没有数据。
EMPTY_ECONOMY = {
    "turns_with_usage": 0,
    "turns_total": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens": 0,
    "cached_tokens": 0,
    "total_tokens": 0,
    "reasoning_share_of_output": 0.0,
    "cached_share_of_input": 0.0,
    "input_tokens_per_turn": 0.0,
    "by_run": {},
}


def summarize_economy(index):
    """从 `TraceIndex` 汇总 token 用量。

    只跟 `Session → Run → Turn` 三级结构打交道，不再直接解析文件——和其它
    指标一样的约定。
    """
    turns = [turn for run in index.runs for turn in run.turns]
    with_usage = [turn for turn in turns if turn.has_usage]
    if not with_usage:
        return dict(EMPTY_ECONOMY, turns_total=len(turns))

    input_tokens = sum(turn.input_tokens for turn in with_usage)
    output_tokens = sum(turn.output_tokens for turn in with_usage)
    reasoning_tokens = sum(turn.reasoning_tokens for turn in with_usage)
    cached_tokens = sum(turn.cached_tokens for turn in with_usage)

    by_run = {}
    for run in index.runs:
        run_turns = [turn for turn in run.turns if turn.has_usage]
        if not run_turns:
            continue
        by_run[run.run_id] = {
            "task_id": run.task_id,
            "turns": len(run_turns),
            "input_tokens": sum(turn.input_tokens for turn in run_turns),
            "output_tokens": sum(turn.output_tokens for turn in run_turns),
            "reasoning_tokens": sum(turn.reasoning_tokens for turn in run_turns),
            "cached_tokens": sum(turn.cached_tokens for turn in run_turns),
        }

    return {
        "turns_with_usage": len(with_usage),
        "turns_total": len(turns),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        # 思维链 token 计入 output_tokens，不是额外的一份。单列出来是因为
        # 它在这个后端上占比极高，不看这个数会以为"输出很便宜"。
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": input_tokens + output_tokens,
        "reasoning_share_of_output": (reasoning_tokens / output_tokens) if output_tokens else 0.0,
        "cached_share_of_input": (cached_tokens / input_tokens) if input_tokens else 0.0,
        # 每轮平均 input ≈ 上下文规模，是 context 裁剪是否生效的直接观测量。
        "input_tokens_per_turn": input_tokens / len(with_usage),
        "by_run": by_run,
    }


def render_economy_markdown(summary):
    """渲染成一节。没有数据时明说原因，而不是省略掉这一节。"""
    if not summary or not summary.get("turns_with_usage"):
        return (
            "## 成本（economy 轴）\n\n"
            "- **无数据**：本次跑批没有任何一轮带 usage。参考解回放走 `FakeModelClient`，"
            "不发请求给模型，因此不产生 token 用量——成本类指标必须由真实 provider 跑批。\n"
        )

    lines = ["## 成本（economy 轴）", ""]
    lines.append(
        f"- 带 usage 的模型调用：{summary['turns_with_usage']} / {summary['turns_total']} 轮"
    )
    lines.append(
        f"- **input {summary['input_tokens']:,} token · output {summary['output_tokens']:,} token**"
        f"（合计 {summary['total_tokens']:,}）"
    )
    lines.append(
        f"- 每轮平均 input {summary['input_tokens_per_turn']:,.0f} token"
        "（≈ 上下文规模，是裁剪是否生效的直接观测量）"
    )
    lines.append(
        f"- **思维链占输出 {summary['reasoning_share_of_output']:.1%}**"
        f"（{summary['reasoning_tokens']:,} / {summary['output_tokens']:,}）"
        "——思维链 token **计入** output_tokens，不是额外的一份；"
        "按 `output - reasoning` 估成本会系统性低估。"
    )
    lines.append(
        f"- **缓存命中占输入 {summary['cached_share_of_input']:.1%}**"
        f"（{summary['cached_tokens']:,} / {summary['input_tokens']:,}）"
        "——命中的是稳定前缀那一段。"
    )
    lines.append("")

    by_run = summary.get("by_run") or {}
    if by_run:
        # 按 input 从多到少列前 5 个：最贵的那几次运行是最值得看的。
        top = sorted(by_run.items(), key=lambda kv: kv[1]["input_tokens"], reverse=True)[:5]
        lines.append("input token 最多的 5 次运行：")
        lines.append("")
        lines.append("| 运行 | 任务 | 轮数 | input | output | 其中思维链 | 缓存命中 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for run_id, row in top:
            lines.append(
                f"| `{run_id[:24]}` | `{row['task_id'] or '-'}` | {row['turns']} | "
                f"{row['input_tokens']:,} | {row['output_tokens']:,} | "
                f"{row['reasoning_tokens']:,} | {row['cached_tokens']:,} |"
            )
        lines.append("")
    return "\n".join(lines)
