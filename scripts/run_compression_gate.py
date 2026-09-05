"""阶段 4 的开工判据:逐段硬裁到底还会不会被触发。

一句话:**这个脚本只回答一个问题——在合成压力负载上把会话摘要开着跑一遍,
`budget_reductions` 是不是空的。** 空了,`DEFAULT_REDUCTION_ORDER` + `SECTION_FLOORS`
那套不可逆的逐段硬裁才退化成纯兜底,阶段 4(把它删掉)才有开工的依据;非空就不能删——
删了而上面几级顶不住,超预算时就没有任何应对。

为什么要写成脚本而不是跑一次记个数:这个判据此前只存在于迁移文档的一句话里
(「每条对话 306 token 那一档摘要完全无效、边界移动 19 次全靠硬裁扛住」)。
一句记下来的结论没法在改完代码之后**重跑一遍**,于是「现在还成不成立」永远是猜的。

不调模型,全部确定性:负载是合成的对话历史,渲染走真实的 `ContextManager.build_all()`。
所以它跑一次几秒钟,可以在每次动上下文代码之后当回归跑。

用法:

    python scripts/run_compression_gate.py --output-json artifacts/compression-gate.json
    python scripts/run_compression_gate.py --variants full no_session_summary
"""

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codingforme import CodingForMe, FakeModelClient, SessionStore, WorkspaceContext  # noqa: E402
from codingforme.context_manager import ContextManager  # noqa: E402
from codingforme.eval.harness import BUILTIN_HARNESS_SPECS, get_harness  # noqa: E402

# 每条对话多大。三档取自阶段 2 那张适用条件表——96 token 那档摘要有效、206 那档
# 开始帮倒忙、306 那档完全无效。判据要在**三档上都成立**才算数:只在最轻的那档
# 上「硬裁没触发」不能说明问题,压力大的负载才是这套东西存在的理由。
TURN_TOKEN_SIZES = (96, 206, 306)
TURNS = 24
# 8k 兜底档。刻意用最小的档位:大窗口下压缩机制根本没有作用对象(三次全量 live
# 共 189 轮里 186 轮的占用率中位数只有 0.70%),在那儿测「硬裁触发了没有」等于
# 在测「压力到了没有」。
WINDOW_TOKENS = 8_000


def _filler_words(tokens):
    """造一段大约 `tokens` 个 token 的英文填充文本。

    用固定词表而不是随机串:同一份负载要能跨次复现,否则两次跑批不可比。
    """
    return "spec detail " * max(1, tokens // 2)


def _transcript_head(prompt):
    """组装好的 prompt 里 history 段的头两行。

    这两行一变，它之后的全部前缀缓存作废——所以「变了几次」是压缩策略的真实代价，
    比「裁掉了多少 token」更能说明问题。取正文而不是 token 数，是因为两轮之间
    token 数可能碰巧相同而内容已经换了一批。
    """
    marker = "Transcript:"
    if marker not in prompt:
        return ""
    tail = prompt[prompt.index(marker) + len(marker):].lstrip("\n")
    return "\n".join(tail.splitlines()[:2])[:120]


def _build_agent(root, variant):
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(root)
    spec = get_harness(variant)
    return CodingForMe(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=SessionStore(root / ".codingforme" / "sessions"),
        feature_flags=spec.resolved_feature_flags(),
        context_window=WINDOW_TOKENS,
    )


def run_arm(variant, turn_tokens, root):
    """逐轮把对话加长,每一轮都真的组一次上下文,记下这一轮发生了什么。

    逐轮渲染而不是一次性喂满,是因为要观测的两件事都只在**跨轮**上才成立:
    硬裁触发了几轮、history 的起点被改写了几次(每改写一次,它之后的前缀缓存全废)。
    """
    agent = _build_agent(root, variant)
    budget = int(agent.context_budget)
    filler = _filler_words(turn_tokens)
    rows = []
    for index in range(TURNS):
        agent.record({"role": "user", "content": "requirement %d: %s" % (index, filler)})
        agent.record({"role": "assistant", "content": "acknowledged %d: %s" % (index, filler)})
        _, prompt, metadata = ContextManager(agent, total_budget=budget).build_all("continue")
        pressure = metadata.get("context_pressure") or {}
        summary_state = pressure.get("session_summary") or {}
        history = metadata.get("history") or {}
        section = (metadata.get("sections") or {}).get("history") or {}
        # 段的 metadata 里只有 token 计数，没有正文——所以头一行从组装出来的 prompt
        # 里取。第一版写的是 `section.get("rendered")`，那个键根本不存在，于是这一列
        # 恒为空串、`history_head_rewrites` 恒为 0，看起来像「边界一次都没动」，
        # 而真实情况正相反。**不存在的键取出来是空值，不是报错**，这类缺陷只能靠
        # 盯着结果里那个「好得可疑」的 0 才发现。
        rendered_head = _transcript_head(prompt)
        rows.append(
            {
                "turn": index + 1,
                "prompt_tokens": int(metadata.get("prompt_tokens") or 0),
                "occupancy": round(float(pressure.get("occupancy_before") or 0.0), 4),
                "reductions": len(metadata.get("budget_reductions") or []),
                "summary_compactions": int(summary_state.get("compactions") or 0),
                "summary_covered": int(summary_state.get("covered_entries") or 0),
                "squeezed": int(history.get("squeezed_entry_count") or 0),
                "omitted": int(section.get("omitted_entry_count") or 0),
                # history 段渲染出来的头两行。它一变,它之后的前缀缓存全部作废,
                # 所以「变了几次」比「裁了多少 token」更能说明代价。
                "history_head": rendered_head,
            }
        )
    heads = [row["history_head"] for row in rows]
    rewrites = sum(1 for a, b in zip(heads, heads[1:]) if a != b)
    reduction_turns = [row["turn"] for row in rows if row["reductions"]]
    return {
        "variant": variant,
        "turn_tokens": turn_tokens,
        "budget_tokens": budget,
        # **这就是阶段 4 的判据本身。** 空列表 = 这一档上逐段硬裁一次都没被触发。
        "reduction_turns": reduction_turns,
        "gate_open": not reduction_turns,
        "peak_occupancy": max(row["occupancy"] for row in rows),
        "total_prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "median_prompt_tokens": int(statistics.median(row["prompt_tokens"] for row in rows)),
        "summary_compaction_turns": [row["turn"] for row in rows if row["summary_compactions"]],
        "max_summary_covered": max(row["summary_covered"] for row in rows),
        "max_squeezed": max(row["squeezed"] for row in rows),
        "max_omitted": max(row["omitted"] for row in rows),
        "history_head_rewrites": rewrites,
        "rows": rows,
    }


def render_markdown(results):
    out = [
        "# 阶段 4 开工判据:逐段硬裁还会不会被触发",
        "",
        "判据只有一条:**同一条压力负载上把会话摘要开着跑一遍,`budget_reductions` 为空。**",
        "空了才说明上面几级(会话摘要 / 可逆折叠 / 分级压缩)真的把不可逆的逐段硬裁替掉了,",
        "阶段 4「删掉硬裁」才有依据。每一行是一次 %d 轮的合成对话,窗口固定在 8k 兜底档;" % TURNS,
        "「每条对话多大」这一列是负载强度,三档分别对应摘要有效 / 开始帮倒忙 / 完全无效。",
        "",
        "| 变体 | 每条对话多大 | 预算 | 硬裁触发的轮次 | 判据 | 峰值占用 | 摘要推进的轮次 | 压扁条数 | 丢弃条数 | history 起点改写 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in results:
        out.append(
            "| `%s` | %d token | %s | %s | %s | %.1f%% | %s | %d | %d | %d 次 |"
            % (
                item["variant"],
                item["turn_tokens"],
                format(item["budget_tokens"], ","),
                ", ".join(str(turn) for turn in item["reduction_turns"]) or "一次都没有",
                "**可以开工**" if item["gate_open"] else "不能删",
                100 * item["peak_occupancy"],
                ", ".join(str(turn) for turn in item["summary_compaction_turns"]) or "一次都没有",
                item["max_squeezed"],
                item["max_omitted"],
                item["history_head_rewrites"],
            )
        )
    default_arms = [item for item in results if item["variant"] == "full"]
    open_all = bool(default_arms) and all(item["gate_open"] for item in default_arms)
    out.extend(
        [
            "",
            "## 结论",
            "",
            "默认变体(`full`)在**全部 %d 档**负载上硬裁都没触发 → 阶段 4 可以开工。" % len(default_arms)
            if open_all
            else "默认变体(`full`)至少有一档负载仍然要靠逐段硬裁扛住 → **阶段 4 不能开工**,"
            "删掉硬裁之后这一档就没有任何应对。",
        ]
    )
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--variants", nargs="+", default=["full", "no_session_summary"],
        choices=sorted(BUILTIN_HARNESS_SPECS),
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-markdown", default="")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    root = Path(tempfile.mkdtemp(prefix="cfm-gate-"))
    results = []
    for variant in args.variants:
        for turn_tokens in TURN_TOKEN_SIZES:
            item = run_arm(variant, turn_tokens, root / ("%s-%d" % (variant, turn_tokens)))
            results.append(item)
            print(
                "%-22s %3d token/turn -> reductions on turns %s"
                % (variant, turn_tokens, item["reduction_turns"] or "none"),
                flush=True,
            )

    markdown = render_markdown(results)
    print()
    print(markdown)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"window_tokens": WINDOW_TOKENS, "turns": TURNS, "results": results},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("wrote %s" % path)
    if args.output_markdown:
        path = Path(args.output_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
