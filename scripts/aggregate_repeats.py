"""把多轮重复测量合成一张预算对照表，重点看方差。

一个样本决定不了预算：同一个任务两次跑出过 5 步和 4 步。这个脚本回答的是
「实测步数稳不稳」，以及「按某条规则设的预算能覆盖几轮里的几个任务」。

吃的是 `run_eval_suite.py --repeats k` 落下的**每轮 benchmark 工件**
（`--benchmark-artifact` 那一份，不是 `--output-json` 的聚合结果）。k>1 时
后缀自动变成 `-r1` / `-r2` / …，把它们全部传进来：

    python scripts/aggregate_repeats.py artifacts/benchmark-live-k3-r{1,2,3}.json
"""
import json
import pathlib
import statistics
import sys
from collections import Counter, defaultdict

paths = [pathlib.Path(p) for p in sys.argv[1:]]
artifacts = [json.loads(p.read_text(encoding="utf-8")) for p in paths if p.is_file()]
if not artifacts:
    raise SystemExit("no artifacts")
print(f"轮次 {len(artifacts)} 份: {', '.join(p.name for p in paths if p.is_file())}\n")

steps = defaultdict(list)
budget = {}
cats = defaultdict(list)
passed = defaultdict(list)
order = []
for art in artifacts:
    for row in art["rows"]:
        tid = row["id"]
        if tid not in budget:
            budget[tid] = row["step_budget"]
            order.append(tid)
        steps[tid].append(row["tool_steps"])
        cats[tid].append(row.get("failure_category") or "pass")
        passed[tid].append(bool(row["passed"]))

hdr = f"{'task':<30} {'声明':>4} {'各轮步数':<14} {'最大':>4} {'均值':>5} {'极差':>4}  {'归因（各轮）'}"
print(hdr)
print("-" * 118)
for tid in order:
    vals = steps[tid]
    spread = max(vals) - min(vals)
    flag = " <<" if spread >= 3 else ""
    print(
        f"{tid:<30} {budget[tid]:>4} {str(vals):<14} {max(vals):>4} "
        f"{statistics.mean(vals):>5.1f} {spread:>4}  {','.join(c[:12] for c in cats[tid])}{flag}"
    )

print()
# 候选规则：预算 = k × 参考解预算，看能覆盖多少「任务×轮次」
print("规则覆盖率（分母 = 任务数 × 轮次数 =", len(order) * len(artifacts), "）")
for label, fn in (
    ("原样不动 (1x)", lambda b: b),
    ("1.5x 向上取整", lambda b: -(-b * 3 // 2)),
    ("2x", lambda b: b * 2),
    ("2x 且下限 6", lambda b: max(b * 2, 6)),
    ("3x", lambda b: b * 3),
):
    ok = sum(1 for tid in order for v in steps[tid] if v <= fn(budget[tid]))
    tasks_all_ok = sum(1 for tid in order if all(v <= fn(budget[tid]) for v in steps[tid]))
    print(f"  {label:<16} 覆盖 {ok:>3}/{len(order) * len(artifacts)}  |  三轮全过的任务 {tasks_all_ok}/{len(order)}")

print()
print("失败归因合计:", dict(Counter(c for tid in order for c in cats[tid])))
print("通过率（各轮）:", [sum(art_pass) for art_pass in zip(*[passed[t] for t in order])] if order else [])
print("按轮次通过数:", [sum(1 for tid in order if passed[tid][i]) for i in range(len(artifacts))])
