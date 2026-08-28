"""重复跑同一份基准之后，回答「这个数字可信吗」。

为什么存在：
单次跑批的通过率在真实模型下基本是抛硬币。实测同一个任务、temperature=0，
步数能跑出 [2, 2, 15]，通过与否也在轮次间跳变——**单轮结论会把运气读成能力**。
τ-bench 提出 pass^k 就是为了把「稳不稳」从「准不准」里分离出来。

这一层此前被标成阻塞，理由是「回放参考解是确定性的，方差恒为 0，跑了没信息量」。
接上真实 provider 之后确定性消失，它才成立。

三个刻意的口径：

- **pass@1 取的是平均通过率**（k 轮里过了几轮），不是「至少过一次」。后者会把
  一个 1/3 的任务读成通过。
- **pass^k 估计的是「k 轮全过」的任务占比**。k=3 是很粗的估计量，样本少时它系统性
  偏低，报数时必须连 k 一起给。
- **撞到步数上限的运行会被单独标出**（censored）。那些运行的 `tool_steps` 只是
  「≥上限」，拿去算中位数会得到一个偏低且看不出问题的数——预算就是这样被拟合歪的。
"""

import statistics

RELIABILITY_SCHEMA_VERSION = 1

STABILITY_STABLE_PASS = "stable-pass"
STABILITY_STABLE_FAIL = "stable-fail"
STABILITY_FLAKY = "flaky"


def _stability(passes):
    if all(passes):
        return STABILITY_STABLE_PASS
    if not any(passes):
        return STABILITY_STABLE_FAIL
    return STABILITY_FLAKY


def summarize_reliability(row_groups):
    """把 k 轮的 rows 合成可靠性画像。

    输入 `row_groups` 是 [rows_of_repeat_1, rows_of_repeat_2, ...]，
    每份 rows 就是 benchmark artifact 里的 `rows`。
    """
    row_groups = [list(rows or []) for rows in row_groups or []]
    repeats = len(row_groups)
    by_task = {}
    order = []
    for rows in row_groups:
        for row in rows:
            task_id = row["id"]
            if task_id not in by_task:
                by_task[task_id] = {
                    "passes": [],
                    "tool_steps": [],
                    "within_budget": [],
                    "ceiling_hits": 0,
                    "step_budget": int(row.get("step_budget", 0) or 0),
                    "failure_categories": [],
                }
                order.append(task_id)
            entry = by_task[task_id]
            entry["passes"].append(bool(row.get("passed")))
            entry["tool_steps"].append(int(row.get("tool_steps", 0) or 0))
            entry["within_budget"].append(bool(row.get("within_budget")))
            if row.get("step_ceiling_hit"):
                entry["ceiling_hits"] += 1
            entry["failure_categories"].append(row.get("failure_category"))

    tasks = {}
    censored_runs = 0
    for task_id in order:
        entry = by_task[task_id]
        passes = entry["passes"]
        steps = entry["tool_steps"]
        censored_runs += entry["ceiling_hits"]
        tasks[task_id] = {
            "runs": len(passes),
            "passes": passes,
            "pass_at_1": (sum(passes) / len(passes)) if passes else 0.0,
            # k 轮全过才算 1。样本少时这是个偏保守的估计量。
            "pass_hat_k": 1.0 if passes and all(passes) else 0.0,
            "stability": _stability(passes),
            "tool_steps": steps,
            "steps_median": statistics.median(steps) if steps else 0,
            "steps_max": max(steps) if steps else 0,
            "steps_spread": (max(steps) - min(steps)) if steps else 0,
            "step_budget": entry["step_budget"],
            "within_budget_runs": sum(1 for ok in entry["within_budget"] if ok),
            # 撞上限的轮次数。非 0 时这一行的步数统计不可用于拟合预算。
            "ceiling_hits": entry["ceiling_hits"],
            "steps_censored": entry["ceiling_hits"] > 0,
            "failure_categories": entry["failure_categories"],
        }

    task_count = len(order)
    flaky = [tid for tid in order if tasks[tid]["stability"] == STABILITY_FLAKY]
    return {
        "schema_version": RELIABILITY_SCHEMA_VERSION,
        "repeats": repeats,
        "task_count": task_count,
        "pass_at_1": (sum(tasks[t]["pass_at_1"] for t in order) / task_count) if task_count else 0.0,
        "pass_hat_k": (sum(tasks[t]["pass_hat_k"] for t in order) / task_count) if task_count else 0.0,
        "tasks_ever_passed": sum(1 for t in order if any(tasks[t]["passes"])),
        "tasks_always_passed": sum(1 for t in order if all(tasks[t]["passes"]) and tasks[t]["passes"]),
        "flaky_tasks": flaky,
        "flaky_count": len(flaky),
        # 步数被截断的运行总数。这个数非 0 时，任何「模型需要几步」的结论都是下界。
        "censored_runs": censored_runs,
        "tasks_with_censored_steps": sum(1 for t in order if tasks[t]["steps_censored"]),
        "by_task": tasks,
    }


def render_reliability_markdown(summary):
    """渲染成表。k=1 时明确说明这一层不成立，而不是照常印出一堆 1.00。"""
    if not summary or not summary.get("task_count"):
        return "## 可靠性\n\n- 无数据。\n"
    repeats = summary["repeats"]
    lines = ["## 可靠性（重复 k 次）", ""]
    if repeats < 2:
        lines.append(
            f"- **k = {repeats}，这一层不成立**：只跑一轮无从谈起「稳不稳」，"
            "下面的 pass^k 恒等于 pass@1，不要当结论读。"
        )
        lines.append("")
    lines.append(f"- k = {repeats}，任务 {summary['task_count']} 个")
    lines.append(f"- **pass@1 = {summary['pass_at_1']:.3f}**（平均每轮过多少）")
    lines.append(f"- **pass^{repeats} = {summary['pass_hat_k']:.3f}**（{summary['tasks_always_passed']} 个任务轮轮都过）")
    lines.append(f"- 至少过一次的任务：{summary['tasks_ever_passed']} / {summary['task_count']}")
    lines.append(f"- **结果在轮次间跳变的任务：{summary['flaky_count']} 个**"
                 + (f"（{', '.join(summary['flaky_tasks'])}）" if summary["flaky_tasks"] else ""))
    if summary["censored_runs"]:
        lines.append(
            f"- ⚠ **{summary['censored_runs']} 次运行顶到了步数上限**，涉及 "
            f"{summary['tasks_with_censored_steps']} 个任务：这些运行的步数只是下界，"
            "不能用来拟合预算。"
        )
    lines.append("")
    # 「声明预算」这一列几乎必然被误读，所以说明必须紧贴着表、而不是丢给外部文档：
    # 它记的是**参考解**走完这题需要几步（2~6），不是这次给模型的步数上限。
    # 真实模型要探索要试错，普遍走十几步，于是「预算内轮次」几乎全是 0/k——
    # 那是两个口径的差，不是"任务全部超预算"。
    lines.append(
        "- 读表须知：**「声明预算」是数据集记录的「参考解最短路径」，"
        "不是本次给模型的步数上限**（后者见上方「被测 harness」一栏）。"
        "「预算内轮次」按参考解那个数判定，所以真实模型跑批时普遍是 0/k，属预期。"
    )
    lines.append("")
    lines.append("| 任务 | pass@1 | 稳定性 | 各轮步数 | 中位 | 极差 | 声明预算 | 预算内轮次 |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|")
    for task_id, row in summary["by_task"].items():
        steps = ", ".join(str(s) for s in row["tool_steps"])
        if row["steps_censored"]:
            steps += " ⚠"
        lines.append(
            f"| `{task_id}` | {row['pass_at_1']:.2f} | {row['stability']} | {steps} | "
            f"{row['steps_median']:.0f} | {row['steps_spread']} | {row['step_budget']} | "
            f"{row['within_budget_runs']}/{row['runs']} |"
        )
    lines.append("")
    return "\n".join(lines)
