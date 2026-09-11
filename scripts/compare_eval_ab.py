"""把两份（或多份）评测套件工件并排成一张对照表。

为什么存在：
`scripts/compare_context_ab.py` 对的是压力探针的工件，形状是「一个探针一行」；
评测套件的工件（`run_eval_suite.py --output-json`）形状完全不同——L2 结果层、
L1 逐条断言、按 subject 分的两半、经济性四个数。手工对着两份 JSON 读，最常犯
的两个错都不是算错，而是**读法错**：

1. **只看总数，不看分母。** L1 从 150/156 变成 139/150，两个分母就不一样——
   适用的断言本来就随轨迹变。真正能比的是**逐条断言**那一层，同一条断言两边
   的分母往往才相同。
2. **把抖动读成因果。** k=1 时一条断言只有几次适用判定，4/6 → 2/6 看着吓人，
   Fisher 精确检验 p=0.57，也就是「靠运气就能翻出来」。所以这个脚本对每一条
   计数都当场给出 p 值，而不是让读者自己判断这个差距算不算数。

它还会在所有数字之前检查两臂**可不可比**：`code_signature_parts` 里除了提示词
之外还有别的项不同，就说明这次不止换了一个变量，对照本身不成立。

用法：

    python scripts/compare_eval_ab.py \
        --arm current=artifacts/eval-live-p1prompt.json \
        --arm pre_phase1=artifacts/eval-live-prompt-legacy.json \
        --output-markdown docs/metrics/prompt-ab.md
"""

import argparse
import json
import math
import sys
from pathlib import Path


def _ratio(block):
    if not block:
        return None
    return int(block.get("passed") or 0), int(block.get("total") or 0)


# 逐条断言之外还要并排的几个总量。每个都带一句「越大越好还是越小越好」，因为
# 一张只有数字的对照表，读者第一步要做的判断恰恰是这个。
SUMMARY_ROWS = (
    ("L2 结果层（任务真做完了没）", "higher", lambda d: _ratio(d["aggregates"]["by_level"].get("L2-task"))),
    ("L1 轨迹层（过程干净率）", "higher", lambda d: _ratio(d["aggregates"]["trajectory"])),
    ("L1 · subject=harness（闸口/预算/白名单）", "higher",
     lambda d: _ratio(d["aggregates"]["trajectory"]["by_subject"].get("harness"))),
    ("L1 · subject=model（模型自己的调用序列）", "higher",
     lambda d: _ratio(d["aggregates"]["trajectory"]["by_subject"].get("model"))),
)

ECONOMY_ROWS = (
    ("每轮输入 token", "lower", "input_tokens_per_turn", 1),
    ("输入 token 合计", "lower", "input_tokens", 0),
    ("输出 token 合计", "lower", "output_tokens", 0),
    ("模型往返轮数", "lower", "turns_total", 0),
    ("缓存命中占输入", "higher", "cached_share_of_input", None),
)


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fisher_exact_two_sided(a, b, c, d):
    """2×2 表的 Fisher 精确检验（双侧），纯标准库。

    用它而不是卡方：这些格子里的数常常是个位数（一条断言在 15 个任务上只适用
    6 次），卡方的近似在这个量级上不成立。返回 None 表示某一行或某一列全是 0，
    此时检验没有意义。
    """
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    if not (row1 and row2 and col1 and col2):
        return None

    def prob(x):
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    observed = prob(a)
    # 1e-9 的容差是为了让「和观测一样极端」的那些格子别因为浮点误差被漏掉。
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed + 1e-9))


def _fmt_ratio(value):
    if not value:
        return "—"
    passed, total = value
    if not total:
        return "0/0"
    return f"{passed}/{total} ({passed / total * 100:.1f}%)"


def _fmt_number(value, digits):
    if value is None:
        return "—"
    if digits is None:
        return f"{value * 100:.1f}%"
    return f"{value:.{digits}f}" if digits else f"{int(value)}"


def comparability(arms):
    """两臂到底只差了提示词，还是连别的都变了。"""
    notes = []
    parts = {}
    for name, payload in arms:
        harness = payload.get("harness", {}) or {}
        parts[name] = harness.get("code_signature_parts", {}) or {}
        context = payload.get("run_context", {}) or {}
        mode = context.get("execution_mode")
        if mode != "live-model":
            notes.append(
                f"{name} 的 execution_mode 是 {mode!r}——回放参考解时 capability 轴评的是"
                "参考解脚本，提示词改动在它上面**结构上**不可能有作用对象。"
            )
        if context.get("workspace_git_root"):
            notes.append(f"{name} 的评测工作区嵌在一个 git 仓库内部，快照被放大，和别的跑批不可比。")

    names = [name for name, _ in arms]
    if len(names) == 2:
        left, right = parts[names[0]], parts[names[1]]
        differing = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        expected = {"prompt_variant", "prompt_template"}
        extra = [key for key in differing if key not in expected]
        if extra:
            notes.append(
                "两臂除了提示词之外还有别的代码不同：" + ", ".join(extra)
                + "。这不是一次单变量对照，差值不能归因给提示词。"
            )
        if not differing:
            notes.append("两臂的 code_signature_parts 完全相同——大概率跑的是同一个变体，对照不成立。")
    return notes


def render(arms):
    names = [name for name, _ in arms]
    lines = []

    notes = comparability(arms)
    if notes:
        lines.append("## 读这张表之前")
        lines.append("")
        for note in notes:
            lines.append(f"- **{note}**")
        lines.append("")

    lines.append("## 总量")
    lines.append("")
    lines.append("| 指标 | 方向 | " + " | ".join(names) + " |")
    lines.append("|---|---|" + "---|" * len(names))
    for label, direction, getter in SUMMARY_ROWS:
        cells = []
        for _, payload in arms:
            try:
                cells.append(_fmt_ratio(getter(payload)))
            except (KeyError, TypeError):
                cells.append("—")
        arrow = "越高越好" if direction == "higher" else "越低越好"
        lines.append(f"| {label} | {arrow} | " + " | ".join(cells) + " |")
    for label, direction, key, digits in ECONOMY_ROWS:
        cells = [
            _fmt_number((payload["aggregates"].get("economy") or {}).get(key), digits)
            for _, payload in arms
        ]
        arrow = "越高越好" if direction == "higher" else "越低越好"
        lines.append(f"| {label} | {arrow} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 逐条断言（分母相同的那一层才真的可比）")
    lines.append("")
    header = "| 断言 | subject | " + " | ".join(names) + " |"
    if len(names) == 2:
        header = header[:-1] + " 差得算不算数 |"
    lines.append(header)
    lines.append("|---|---|" + "---|" * (len(names) + (1 if len(names) == 2 else 0)))

    by_assertion = [(payload["aggregates"]["trajectory"].get("by_assertion") or {}) for _, payload in arms]
    subjects = {}
    for block in by_assertion:
        for name, stats in block.items():
            subjects.setdefault(name, stats.get("subject", ""))

    for assertion in sorted(subjects):
        cells = []
        counts = []
        for block in by_assertion:
            stats = block.get(assertion)
            if not stats:
                cells.append("—")
                counts.append(None)
                continue
            passed, total = int(stats.get("passed") or 0), int(stats.get("total") or 0)
            cells.append(f"{passed}/{total}")
            counts.append((passed, total - passed))
        row = f"| `{assertion}` | {subjects[assertion]} | " + " | ".join(cells)
        if len(names) == 2:
            verdict = "—"
            if all(counts) and counts[0] != counts[1]:
                p = fisher_exact_two_sided(counts[0][0], counts[0][1], counts[1][0], counts[1][1])
                if p is None:
                    verdict = "无法检验"
                elif p < 0.05:
                    verdict = f"**测得出区别**（p={p:.4f}）"
                else:
                    verdict = f"测不出区别（p={p:.4f}）"
            elif all(counts):
                verdict = "两边一样"
            row += f" | {verdict}"
        lines.append(row + " |")
    lines.append("")

    lines.append("## 每臂各自没过的任务")
    lines.append("")
    for name, payload in arms:
        failed = [
            case["case_id"]
            for case in payload.get("cases", [])
            if case.get("level") == "L2-task" and not case.get("passed")
        ]
        lines.append(f"- **{name}**：" + (", ".join(f"`{item}`" for item in failed) if failed else "全过"))
    lines.append("")
    lines.append(
        "> 失败**换了任务**（两边名单不重合）是抖动的形状，不是某条改动稳定压出来的形状。"
        "两边名单相同、数量也相同时，那条改动在结果层上没有作用对象。"
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="一臂：名字=工件路径，至少给两个",
    )
    parser.add_argument("--output-markdown", default="")
    args = parser.parse_args(argv)

    if len(args.arm) < 2:
        parser.error("至少要两臂才谈得上对照")

    arms = []
    for item in args.arm:
        if "=" not in item:
            parser.error(f"--arm 要写成 NAME=PATH，收到 {item!r}")
        name, _, path = item.partition("=")
        arms.append((name, _load(path)))

    text = render(arms)
    print(text)
    if args.output_markdown:
        out = Path(args.output_markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\n[written] {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
