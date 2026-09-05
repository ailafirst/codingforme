"""128k 档(默认生产窗口)在不同压力强度下的压缩率:复刻阶段 1 最初那次评测的
方法论,只是换成默认生产档位、扫多档压力强度,而不是只在 8k 档上跑一条负载。

## 方法论和阶段 1 是同一套

阶段 1 最初那次 A/B(`docs/architecture/compression-pipeline-migration.md` §3)
只在一条 24 轮压力负载、8k 兜底档上量过一次:峰值占用 100.0% → 77.4%,24 轮
prompt token 合计 88,721 → 69,725(**−21.4%**)。这个脚本是同一套方法论——同一条
增长式负载,压缩开 vs 压缩关,比总 token 数——搬到 128k 档,并且扫多档压力强度,
而不是只测一条。

**压缩率的定义**:`1 − (压缩开着时 N 轮 prompt token 合计) / (压缩完全关闭时的合计)`。
分母用 `no_context_reduction` 变体的输出——它是唯一一个让 prompt 完全不受预算
约束、按内容原样增长的配置(`context_manager._render_sections_without_reduction()`
直接跳过整条裁剪流水线),是"没有压缩"这句话在代码里唯一对应的状态,不是某个
消融开关关掉一个机制之后的"半压缩"状态。

## 为什么是 128k,不是之前测过的 8k

之前(阶段 0/1/10)的压力探针默认用 8k 兜底档,因为那是最容易压出效果的档位;
但真实后端的窗口解析结果向下取整到档位后,`EFFECTIVE_WINDOW_CAP_TOKENS = 128_000`
本身就是**实际生效的上限**——1M 档解析出来，预算算式也是按 128k 算的（阶段 0
之前的文档已经量过这一点）。所以"默认 128k 上下文"不是一个新增的测试档位，是
生产环境实际在用的那个档位，第一次拿它当扫描对象而不是当"陪衬"。

## 为什么要扫多档压力,不是一条负载

阶段 11(`run_system_gate.py`)已经证明单条固定负载在 128k 档上几乎测不到压力
(24 轮 × 最重的 306 token/轮,峰值占用只有 9.4%)。要看清"128k 档的压缩率随压力
怎么变"，必须让负载强度本身成为一个扫描维度，而不是固定一档。

用法:

    python scripts/run_compression_ratio.py --output-json artifacts/compression-ratio.json --output-markdown artifacts/compression-ratio.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codingforme import CodingForMe, FakeModelClient, SessionStore, WorkspaceContext  # noqa: E402
from codingforme.context_manager import ContextManager  # noqa: E402
from codingforme.eval.harness import get_harness  # noqa: E402

import run_compression_gate as gate  # noqa: E402  复用 `_filler_words`

WINDOW_TOKENS = 128_000
TURNS = 24

# 压力强度分档:每条对话(user + assistant 各一条)大约多少 token。跨度刻意拉大——
# 从"明显在触发点之下"到"明显远超预算好几倍",才看得出压缩率随压力怎么变,
# 而不是只有一个点。
PRESSURE_LEVELS = (
    ("light", 150),
    ("moderate", 600),
    ("near_trigger", 1500),
    ("heavy", 3000),
    ("extreme", 6000),
    # 前 5 档实测峰值占用最高只到 84.6%(extreme 那档,刚好卡在 85% 触发点之下)——
    # 分级压缩、会话摘要、逐段硬裁全程一次都没被触发过。不加下面两档的话,这份
    # 报告只能证明"系统在没到触发点之前不会误触发",证明不了"压力真的过了触发点
    # 之后压缩率怎么变"——而后者才是用户要看的那部分。
    ("saturating", 9000),
    ("past_trigger", 13000),
)


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


def run_side(variant, turn_tokens, root):
    """跑一条 24 轮增长负载,返回逐轮 prompt_tokens 与占用率。"""
    agent = _build_agent(root, variant)
    budget = int(agent.context_budget)
    filler = gate._filler_words(turn_tokens)
    rows = []
    for index in range(TURNS):
        agent.record({"role": "user", "content": "requirement %d: %s" % (index, filler)})
        agent.record({"role": "assistant", "content": "acknowledged %d: %s" % (index, filler)})
        _, _, metadata = ContextManager(agent, total_budget=budget).build_all("continue")
        pressure = metadata.get("context_pressure") or {}
        summary_state = pressure.get("session_summary") or {}
        rows.append(
            {
                "turn": index + 1,
                "prompt_tokens": int(metadata.get("prompt_tokens") or 0),
                "occupancy": round(float(pressure.get("occupancy_before") or 0.0), 4),
                "reductions": len(metadata.get("budget_reductions") or []),
                "summary_compactions": int(summary_state.get("compactions") or 0),
            }
        )
    return {
        "variant": variant,
        "budget_tokens": budget,
        "total_prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "peak_occupancy": max(row["occupancy"] for row in rows),
        # 硬裁(第三道最后一步,`DEFAULT_REDUCTION_ORDER` + `SECTION_FLOORS`)触发过的轮次。
        "reduction_turns": [row["turn"] for row in rows if row["reductions"]],
        # 会话摘要(第三道更早一步,阶段 2)真的换掉过内容的轮次——它比硬裁先触发,
        # 只看 `reduction_turns` 会漏掉"压力已经越过触发点、但摘要一步就兜住了、
        # 没轮到硬裁出手"这种情况,而这种情况在这份负载里恰恰是主要形态。
        "summary_turns": [row["turn"] for row in rows if row["summary_compactions"]],
        "rows": rows,
    }


def run_level(level_name, turn_tokens, root):
    compressed = run_side("full", turn_tokens, root / ("%s-full" % level_name))
    uncompressed = run_side("no_context_reduction", turn_tokens, root / ("%s-raw" % level_name))

    raw_total = uncompressed["total_prompt_tokens"]
    compressed_total = compressed["total_prompt_tokens"]
    # raw_total 理论上不可能是 0(哪怕最轻的一档也有仓库快照 + 24 轮对话),
    # 这里仍然防一手除零，避免脚本在边界配置下直接崩溃。
    ratio = 1 - (compressed_total / raw_total) if raw_total else 0.0

    return {
        "level": level_name,
        "turn_tokens": turn_tokens,
        "budget_tokens": compressed["budget_tokens"],
        "raw_total_prompt_tokens": raw_total,
        "compressed_total_prompt_tokens": compressed_total,
        "compression_ratio": round(ratio, 4),
        "peak_occupancy_compressed": compressed["peak_occupancy"],
        # 未压缩那一臂的"名义占用率"：raw_total 换算成占预算的比例,只是给读者一个
        # 直觉——"这条负载原始内容大概相当于多少倍预算",不是真实触发过的占用率
        # (那一臂没有触发裁剪逻辑,`occupancy` 字段本身在关掉 context_reduction 时
        # 也不会被裁剪循环使用)。
        "raw_content_vs_budget_ratio": round(raw_total / compressed["budget_tokens"], 2)
        if compressed["budget_tokens"]
        else 0.0,
        "reduction_turns": compressed["reduction_turns"],
        "hard_cut_triggered": bool(compressed["reduction_turns"]),
        "summary_turns": compressed["summary_turns"],
        "session_summary_triggered": bool(compressed["summary_turns"]),
    }


def render_markdown(results):
    out = [
        "# 128k 档在不同压力强度下的压缩率",
        "",
        "一句话:**压缩率 = 1 − 压缩开着时 24 轮 prompt token 合计 / 压缩完全关闭时的合计。**",
        "分母(`no_context_reduction`)是「没有压缩」这句话在代码里唯一对应的状态——",
        "prompt 完全不受预算约束、按内容原样增长。窗口固定 128k(生产环境实际生效的",
        "上限),压力强度(每条对话大约多少 token)从明显低于触发点扫到明显越过触发点。",
        "",
        "**这个比率不是单一机制的功劳,是两层叠加的结果,读表前先分清楚**:",
        "",
        "- **第二道(无条件,和占用率无关)**:最近 6 轮之外的旧对话,不论压力大不大,",
        "  一律换成 110 token 的残句(`RECENT_TOOL_TURNS` / `POLICY_CLIPPED`)。原始内容",
        "  越大,这一刀省下的绝对 token 数就越多——这是**全部档位**压缩率不为零的主因,",
        "  和「压没压到触发点」无关。",
        "- **第三道(占用率驱动)**:峰值占用越过 85% 触发点后,会话摘要 / 逐段硬裁才会",
        "  出手。「触发过会话摘要」「触发过硬裁」两列标的就是这道有没有真正被压力逼出来。",
        "",
        "| 压力强度 | 每条对话多大 | 预算 | 未压缩合计 | 压缩后合计 | **压缩率** | "
        "压缩臂峰值占用 | 原始内容/预算 | 触发过会话摘要 | 触发过硬裁 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in results:
        out.append(
            "| %s | %s token | %s | %s | %s | **%.1f%%** | %.1f%% | %.1fx | %s | %s |"
            % (
                item["level"],
                format(item["turn_tokens"], ","),
                format(item["budget_tokens"], ","),
                format(item["raw_total_prompt_tokens"], ","),
                format(item["compressed_total_prompt_tokens"], ","),
                100 * item["compression_ratio"],
                100 * item["peak_occupancy_compressed"],
                item["raw_content_vs_budget_ratio"],
                "是(%d/24 轮)" % len(item["summary_turns"]) if item["session_summary_triggered"] else "否",
                "是(%d/24 轮)" % len(item["reduction_turns"]) if item["hard_cut_triggered"] else "否",
            )
        )

    out.extend(["", "## 结论", ""])
    below_trigger = [item for item in results if item["peak_occupancy_compressed"] < 0.85]
    above_trigger = [item for item in results if item["peak_occupancy_compressed"] >= 0.85]
    if below_trigger:
        out.append(
            "- 峰值占用没到 85%% 触发点的档位(%s):压缩率**不是** 0,但全部来自第二道——"
            "会话摘要和硬裁一次都没触发,**这是正确行为,不是没测出效果**。"
            % "、".join(item["level"] for item in below_trigger)
        )
    if above_trigger:
        out.append(
            "- 峰值占用越过触发点的档位(%s):第三道真正被逼出来了——**这是阶段 1~10"
            "之前从没在 128k 档上覆盖到过的区间**,压缩率里开始包含会话摘要 / 硬裁的贡献,"
            "不再只是第二道的固定效果。"
            % "、".join(item["level"] for item in above_trigger)
        )
    else:
        out.append(
            "- **本次全部档位的峰值占用都没有越过 85% 触发点**——说明当前扫描的压力强度"
            "上限仍然不够大,128k 档的第三道(会话摘要 / 硬裁)在这次实验里全程没有机会"
            "出手,表里的压缩率**完全来自第二道**;这不是「机制在 128k 档上不起作用」,"
            "是这份负载还没能把占用率推过触发点。"
        )
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-markdown", default="")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    import tempfile

    root = Path(tempfile.mkdtemp(prefix="cfm-ratio-"))
    results = []
    for level_name, turn_tokens in PRESSURE_LEVELS:
        item = run_level(level_name, turn_tokens, root / level_name)
        results.append(item)
        print(
            "%-14s %6d token/turn -> ratio=%.1f%% peak_occupancy=%.1f%% hard_cut=%s"
            % (
                level_name,
                turn_tokens,
                100 * item["compression_ratio"],
                100 * item["peak_occupancy_compressed"],
                item["hard_cut_triggered"],
            ),
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
