"""把两份（或多份）压力探针工件并排成一张对照表。

为什么存在：
上下文工程的三个阶段之前只验证到「机制正确、没有回归」，缺的是最后一层——
**机制到底有没有收益**。收益只有一种量法：同一份负载跑两次，只换一个消融开关
（`--harness full` vs `--harness no_tool_output_spill` / `no_window_block`），
再看四个数怎么变。手工对着两份 JSON 读容易只挑对自己有利的那一列，所以把
对照口径写死在脚本里。

四个数分别回答一件事，缺一不可：

    answered        答对没有。省了上下文却把答案弄丢，那是负收益，不是收益。
    steps           花了几步。同样答对，步数少的那个才叫省。
    input_tokens    烧了多少输入 token。这是「省上下文」这句话的直接度量。
    cached_share    缓存命中占输入的比例。按块推进那条改动唯一想改善的就是它。

用法：

    python scripts/compare_context_ab.py \
        --arm full=artifacts/stress-ab-full.json \
        --arm no_spill=artifacts/stress-ab-nospill.json
"""

import argparse
import json
import sys
from pathlib import Path


def _load(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {probe["probe"]: probe for probe in payload.get("probes", [])}


def _cached_share(probe):
    total = int(probe.get("input_tokens") or 0)
    if total <= 0:
        return None
    return int(probe.get("cached_tokens") or 0) / total


def _fmt(value, kind):
    if value is None:
        return "—"
    if kind == "share":
        return "%.1f%%" % (value * 100)
    if kind == "bool":
        return "yes" if value else "NO"
    return str(value)


METRICS = (
    ("answered", "答对", "bool", lambda p: bool(p.get("answered"))),
    ("tool_calls", "工具调用数", "int", lambda p: int(p.get("tool_calls") or 0)),
    ("turns", "模型往返", "int", lambda p: int(p.get("turns") or 0)),
    ("input_tokens", "输入 token", "int", lambda p: int(p.get("input_tokens") or 0)),
    ("cached_share", "缓存命中占输入", "share", _cached_share),
    ("spilled", "落盘次数", "int", lambda p: int(p.get("spilled") or 0)),
    ("spill_failed", "落盘失败", "int", lambda p: int(p.get("spill_failed") or 0)),
    ("spill_readbacks", "照指针取回", "int", lambda p: int(p.get("spill_readbacks") or 0)),
    ("pointer_max", "窗口外指针占位", "int", lambda p: int(p.get("spilled_pointer_count_max") or 0)),
    ("stale_max", "过期读作废", "int", lambda p: int(p.get("stale_read_count_max") or 0)),
    ("stale_ptr_max", "过期指针标记", "int", lambda p: int(p.get("stale_pointer_count_max") or 0)),
    ("reductions", "裁剪次数", "int", lambda p: int(p.get("budget_reductions") or 0)),
    ("clear_bound", "按 1/10 裁的次数", "int", lambda p: int(p.get("clear_at_least_bound") or 0)),
    ("squeezed_max", "压扁成残句", "int", lambda p: int(p.get("squeezed_entry_max") or 0)),
    ("compactions", "自动压缩次数", "int", lambda p: int(p.get("compactions_auto") or 0)),
    ("delegates", "委派次数", "int", lambda p: int(p.get("delegate_calls") or 0)),
    ("delegate_saved", "委派省下的 token", "int", lambda p: int(p.get("delegate_saved_tokens") or 0)),
    ("stop_reason", "终止原因", "str", lambda p: str(p.get("stop_reason") or "")),
)


def render(arms):
    """arms: [(名字, {probe: summary})]，按探针分组输出 markdown 表。"""
    names = []
    for _, probes in arms:
        for name in probes:
            if name not in names:
                names.append(name)

    out = []
    for probe_name in names:
        present = [(label, probes[probe_name]) for label, probes in arms if probe_name in probes]
        if len(present) < 2:
            # 只有一条臂的探针照样列出来——它不是对照，但零值也要看得见。
            out.append("### %s（只有一条臂，不构成对照）\n" % probe_name)
        else:
            out.append("### %s\n" % probe_name)
        goal = present[0][1].get("goal", "")
        if goal:
            out.append("测的是：%s\n" % goal)
        header = "| 指标 | " + " | ".join(label for label, _ in present) + " |"
        out.append(header)
        out.append("|---|" + "---|" * len(present))
        for _, title, kind, getter in METRICS:
            cells = [_fmt(getter(probe), kind) for _, probe in present]
            out.append("| %s | %s |" % (title, " | ".join(cells)))
        out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="一条臂：显示名=工件路径。至少给两条才叫对照。",
    )
    parser.add_argument("--output-markdown", default="")
    args = parser.parse_args(argv)

    arms = []
    for item in args.arm:
        if "=" not in item:
            raise SystemExit("--arm 要写成 LABEL=PATH，收到的是 %r" % item)
        label, path = item.split("=", 1)
        arms.append((label, _load(path)))
    if not arms:
        raise SystemExit("至少给一个 --arm")

    text = render(arms)
    # Windows 控制台默认 GBK，直接 print 会在 emoji 上抛 UnicodeEncodeError。
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    if args.output_markdown:
        target = Path(args.output_markdown)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print("wrote %s" % target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
