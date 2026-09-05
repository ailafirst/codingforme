"""真实工具调用负载下的上下文压缩测试体系(P0~P2 全量)。

## 这一版相对上一版改了什么、为什么

上一版(16 格,message×tool 4×4,tool 档到 4x)跑完后发现一个结构性问题,
不是脚本 bug:

    `_store_tool_output()`(runtime.py:930-939)——`limit = self.tool_output_limit()`
    无条件计算,`full_tokens <= limit` 才直接放行;超过时,`tool_output_spill`
    关掉不代表"不裁",退回的是 `clip(text, limit)`,一样卡在同一个 limit。
    而 `models.budget_breakdown()`(models.py:314)里 `capped = min(window_tokens,
    EFFECTIVE_WINDOW_CAP_TOKENS=128_000)`——调大 `context_window` 也顶不破,
    `total_budget` 结构上就到不了 200,000(触发 `TOOL_OUTPUT_MAX_TOKENS=25000`
    需要的量)。

结论:**tool 结果一旦超过 `tool_output_limit()`(128k 档约 14,879 token),
在这套系统里没有任何配置能拿到"完全不裁"的基线**——不是某个开关的问题,是
两层无条件上限(`_store_tool_output` 的 limit 检查 + `EFFECTIVE_WINDOW_CAP_TOKENS`)
叠在一起决定的。上一版 tool=1.5x/4x 那 8 格,raw 臂和 compressed 臂在工具结果这
一层已经被砍到同一大小,"压缩率"这个指标在那个区间失去了对比对象,且 1.5x/4x
两档数字完全重复(纯浪费:每格 180~270s)。

对应的修法,P0~P2 四块:

- **P0-a(收窄矩阵)**:tool 档全部收窄到 `tool_output_limit()` 以内
  (0.3/0.6/0.85/0.95x),压缩率矩阵里的每一格都是真的"压缩 vs 完全不压缩"。
- **P0-b(可恢复性检查)**:tool≥limit 这个真实会发生的场景,换一个能测的问题——
  不问"省了多少 token"(结构上答不出来),问"内容还能不能拿回来"。落盘开着时
  超限内容全文还在磁盘上、有指针能读回来;落盘关掉时内容随着 `clip()` 截断
  永久丢失、没有任何取回线索。两者对模型后续任务的影响完全不同。
- **P1-a(机制拆解)**:一个总 ratio 掩盖了"最近窗口裁剪省的"和"落盘省的"分别是
  多少;拆开成 `windowing_saved_tokens` / `spill_saved_tokens` 两个数。
- **P1-b(窗口维度)**:只在 128k 一档测,看不出 `tool_output_limit()` 随预算收缩
  之后的行为——8k 档它被压到下限 1,320,原本"安全"的档位可能反而超限。
- **P2-a(会话摘要触发场景)**:上一版 16 格全部 0 次触发 `session_summary`,是
  `docs/architecture` 记录过的已知条件复现(受保护尾巴本身超过目标点,覆盖点
  顶死),不是 bug。补一组专门去触发它的负载(轮次多、单轮增量小)做对照。
- **P2-b(stale-read / PTC 探针)**:之前的变量表规划过、这次基础测试主动搁置的
  两个机制,现在补上最小可行的验证。

## 用法

    # 全量(P0~P2 六组实验,顺序跑完,每组落一份独立工件到 --output-dir)
    python scripts/run_tool_pressure_matrix.py --mode all --output-dir artifacts

    # 只跑一组
    python scripts/run_tool_pressure_matrix.py --mode ratio --output-json artifacts/x.json --output-markdown artifacts/x.md

    # 冒烟(每组都缩到最小规模,验证脚本本身跑得通)
    python scripts/run_tool_pressure_matrix.py --mode all --smoke
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codingforme import FakeModelClient, models  # noqa: E402
from codingforme.context_manager import ContextManager  # noqa: E402
from codingforme.context_manager import tool_output_limit as _tool_output_limit  # noqa: E402
from codingforme.eval.harness import get_harness  # noqa: E402

import run_compression_gate as gate  # noqa: E402  复用 _filler_words

WINDOW_TOKENS = 128_000
WINDOW_TIERS = (8_000, 32_000, 128_000)
TURNS = 24
# 同一份工具结果文件里循环读取的不同区段数。`repeated_tool_call()` 只比较
# "最近两次真实执行过的工具调用"是否 name+args 完全相同,所以只要连续两轮的
# 读取区间不同就不会撞上这条闸口——不需要为每一轮生成一个独立文件。
WINDOW_VARIANTS = 4

MESSAGE_TOKEN_SIZES = (150, 1_500, 6_000, 9_000)
# P0-a:全部 <1x,压缩率矩阵里每一格 raw 臂都是真的不裁。0.95x 留 5% 余量,
# 躲开落盘指针文案里 `run_id` 造成的 ±1~2 token 抖动和校准的"至少达到"误差。
TOOL_RATIOS = (0.3, 0.6, 0.85, 0.95)
# P1-b 窗口扫描用:刻意包含一个 >1x 档(1.5x),只看 compressed 臂自己的行为
# (是否落盘、占用率),不touch raw 臂,所以不受"超限没有基线"这条限制。
WINDOW_SWEEP_RATIOS = (0.3, 0.6, 0.85, 1.5)
WINDOW_SWEEP_MESSAGE_TOKENS = 1_500

RECOVERABILITY_RATIO = 3.0  # 远超 limit,确保两臂都真正超限
SUMMARY_PROBE_TURNS = 48
SUMMARY_PROBE_MESSAGE_TOKENS = 300
SUMMARY_PROBE_TOOL_RATIO = 0.3

_FILLER_LINE = "spec detail " * 6


# --------------------------------------------------------------------------
# 共享基础设施(校准 / 建文件 / 脚本化输出 / 建 agent)
# --------------------------------------------------------------------------


def _window_text(path_display, start, lines):
    """和 `tools.tool_read_file` 完全相同的渲染格式,供离线校准 token 数用。"""
    numbered = ["%4d: %s" % (n, _FILLER_LINE) for n in range(start, start + lines)]
    body = "\n".join(numbered)
    return "# %s\n%s" % (path_display, body)


def _calibrate_window_lines(target_tokens, path_display):
    """二分搜索:找到让 `_window_text` 渲染结果 >= target_tokens 的最小行数。"""
    hi = 1
    while models.count_tokens(_window_text(path_display, 1, hi)) < target_tokens:
        hi *= 2
        if hi > 4_000_000:
            break
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if models.count_tokens(_window_text(path_display, 1, mid)) < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    return max(1, lo)


def _build_tool_tier_file(root, lines_per_window, rel_path="data/tool_tier.txt", variants=WINDOW_VARIANTS, tail_markers=None):
    """写一个 `variants * lines_per_window` 行的填充文件,循环读取其中几个不重叠区段。

    `tail_markers`:可选,`{window_index: marker_text}`——把标记行放在该窗口的
    **最后一行**,供可恢复性检查用(截断/预览都是保留开头,标记只会出现在
    落盘的全文里,不会出现在进上下文的预览里)。
    """
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    tail_markers = tail_markers or {}
    for k in range(variants):
        window_lines = [_FILLER_LINE] * lines_per_window
        if k in tail_markers:
            window_lines[-1] = tail_markers[k]
        lines.extend(window_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rel_path.replace("\\", "/")


def _scripted_read_outputs(tool_rel_path, lines_per_window, turns, variants=WINDOW_VARIANTS):
    outputs = []
    for turn in range(turns):
        k = turn % variants
        start = k * lines_per_window + 1
        end = start + lines_per_window - 1
        outputs.append(models.tool_call("read_file", path=tool_rel_path, start=start, end=end))
        outputs.append(models.final_answer("turn-%d-done" % turn))
    return outputs


def _build_agent(feature_flags, root, outputs, name, tools_allowlist=("read_file",), max_steps=4, context_window=WINDOW_TOKENS):
    spec = get_harness("full").derive(
        name=name,
        feature_flags=dict(feature_flags),
        tools_allowlist=tools_allowlist,
        max_steps=max_steps,
    )
    return spec.build(
        FakeModelClient(outputs), root, repo_root_override=root, context_window=context_window
    )


def _run_turns(agent, budget, message_tokens, turns):
    """跑 `turns` 轮 `agent.ask()`,每轮附带一条大小可控的用户消息,逐轮记录
    prompt 组装结果。返回逐轮记录列表。"""
    rows = []
    for turn in range(turns):
        filler = gate._filler_words(message_tokens) if message_tokens else ""
        agent.ask("turn %d requirement (msg~%d tok): %s" % (turn, message_tokens, filler))
        _, _, metadata = ContextManager(agent, total_budget=budget).build_all("continue")
        pressure = metadata.get("context_pressure") or {}
        summary_state = pressure.get("session_summary") or {}
        tool_meta = dict(getattr(agent, "_last_tool_result_metadata", None) or {})
        rows.append(
            {
                "turn": turn + 1,
                "prompt_tokens": int(metadata.get("prompt_tokens") or 0),
                "occupancy": round(float(pressure.get("occupancy_before") or 0.0), 4),
                "reductions": len(metadata.get("budget_reductions") or []),
                "summary_compactions": int(summary_state.get("compactions") or 0),
                "tool_output_spilled": bool(tool_meta.get("tool_output_spilled")),
                "tool_output_full_tokens": int(tool_meta.get("tool_output_full_tokens") or 0),
                "tool_output_kept_tokens": int(tool_meta.get("tool_output_kept_tokens") or 0),
                "history_stale_read_count": int((metadata.get("history") or {}).get("stale_read_count") or 0),
                "history_stale_pointer_count": int((metadata.get("history") or {}).get("stale_pointer_count") or 0),
            }
        )
    return rows


def run_side(feature_flags, message_tokens, tool_rel_path, lines_per_window, root, name, turns=TURNS):
    """跑真实 `ask()`:每轮一次大小可控的用户消息 + 一次大小可控的工具读取。

    P1-a:除了总量,还把"落盘省的"(`spill_saved_tokens`,逐轮
    `tool_output_full_tokens - tool_output_kept_tokens` 累加,只在真触发落盘的
    轮次非零)单独记出来,让 `windowing_saved = total_saved - spill_saved`
    这个拆解在 `run_cell` 里能算。
    """
    outputs = _scripted_read_outputs(tool_rel_path, lines_per_window, turns)
    agent = _build_agent(feature_flags, root, outputs, name)
    budget = int(agent.context_budget)
    rows = _run_turns(agent, budget, message_tokens, turns)
    memory_state = (agent.session.get("memory") or {}) if isinstance(agent.session, dict) else {}
    working = memory_state.get("working") or {}
    spill_saved = sum(
        max(0, r["tool_output_full_tokens"] - r["tool_output_kept_tokens"])
        for r in rows
        if r["tool_output_spilled"]
    )
    return {
        "budget_tokens": budget,
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "peak_occupancy": max(r["occupancy"] for r in rows),
        "reduction_turns": [r["turn"] for r in rows if r["reductions"]],
        "summary_turns": [r["turn"] for r in rows if r["summary_compactions"]],
        "spill_turns": [r["turn"] for r in rows if r["tool_output_spilled"]],
        "spill_saved_tokens": int(spill_saved),
        "tool_output_spilled_last_turn": bool(rows[-1]["tool_output_spilled"]) if rows else False,
        "memory_file_summaries": len(memory_state.get("file_summaries") or {}),
        "memory_recent_files": len(working.get("recent_files") or []),
    }


def _raw_flags():
    full_flags = dict(get_harness("full").resolved_feature_flags())
    raw_flags = dict(full_flags)
    raw_flags["context_reduction"] = False
    raw_flags["tool_output_spill"] = False
    return full_flags, raw_flags


# --------------------------------------------------------------------------
# P0-a:压缩率矩阵(tool 档全部 <1x,raw 臂真实有效)
# --------------------------------------------------------------------------


def run_cell(message_tokens, tool_ratio, tool_limit, root):
    target_tool_tokens = int(round(tool_limit * tool_ratio))
    tag = "msg%d_tool%.2fx" % (message_tokens, tool_ratio)
    lines_per_window = _calibrate_window_lines(target_tool_tokens, "data/tool_tier.txt")

    full_flags, raw_flags = _raw_flags()

    compressed_root = root / ("%s-full" % tag)
    raw_root = root / ("%s-raw" % tag)
    tool_rel_path_c = _build_tool_tier_file(compressed_root, lines_per_window)
    tool_rel_path_r = _build_tool_tier_file(raw_root, lines_per_window)

    compressed = run_side(
        full_flags, message_tokens, tool_rel_path_c, lines_per_window, compressed_root,
        "ptm-full-%s" % tag,
    )
    raw = run_side(
        raw_flags, message_tokens, tool_rel_path_r, lines_per_window, raw_root,
        "ptm-raw-%s" % tag,
    )

    raw_total = raw["total_prompt_tokens"]
    compressed_total = compressed["total_prompt_tokens"]
    ratio = 1 - (compressed_total / raw_total) if raw_total else 0.0
    total_saved = raw_total - compressed_total
    spill_saved = compressed["spill_saved_tokens"]
    windowing_saved = total_saved - spill_saved

    return {
        "message_tokens": message_tokens,
        "tool_ratio": tool_ratio,
        "target_tool_tokens": target_tool_tokens,
        "lines_per_window": lines_per_window,
        "budget_tokens": compressed["budget_tokens"],
        "raw_total_prompt_tokens": raw_total,
        "compressed_total_prompt_tokens": compressed_total,
        "compression_ratio": round(ratio, 4),
        "total_saved_tokens": int(total_saved),
        "windowing_saved_tokens": int(windowing_saved),
        "spill_saved_tokens": int(spill_saved),
        "peak_occupancy_compressed": compressed["peak_occupancy"],
        "reduction_turns": compressed["reduction_turns"],
        "hard_cut_triggered": bool(compressed["reduction_turns"]),
        "summary_turns": compressed["summary_turns"],
        "session_summary_triggered": bool(compressed["summary_turns"]),
        "tool_output_spilled_compressed": compressed["tool_output_spilled_last_turn"],
        "tool_output_spilled_raw": raw["tool_output_spilled_last_turn"],
        "memory_file_summaries_compressed": compressed["memory_file_summaries"],
        "memory_recent_files_compressed": compressed["memory_recent_files"],
    }


def render_ratio_markdown(results, tool_limit):
    out = [
        "# P0-a:128k 档,message × tool 4×4 压缩率矩阵(tool 档全部 <1x)",
        "",
        "一句话:tool 档收窄到 `tool_output_limit()` 以内(0.3/0.6/0.85/0.95x),",
        "raw 臂(`context_reduction`+`tool_output_spill` 都关)在这个区间内没有任何",
        "裁剪,是真正的\"完全不压缩\"基线;因此这份表里每一格的压缩率都是有效对比,",
        "不再有上一版 tool≥1.5x 那种\"两臂被同一硬顶截到同一大小\"的失真。",
        "",
        "`tool_output_limit(128k 档预算)` = %s token,tool 档按它的比例定义。" % format(tool_limit, ","),
        "",
        "压缩率 = 1 − compressed_total/raw_total;`windowing_saved`/`spill_saved`",
        "是把省下的 token 拆成\"最近窗口裁剪省的\"和\"落盘省的\"两部分——本表内",
        "`spill_saved` 应恒为 0(没有一格真正超过落盘阈值),这本身就是证据:",
        "P0 收窄之后,压缩率矩阵测到的 100% 是 windowing 机制的贡献。",
        "",
        "| message 档(token/轮) | tool 档(比例/token) | 压缩率 | windowing 省的 token | spill 省的 token | 压缩臂峰值占用 | 触发过硬裁 | 记忆:file_summaries |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in results:
        out.append(
            "| %s | %.2fx (%s) | **%.1f%%** | %s | %s | %.1f%% | %s | %d |"
            % (
                format(item["message_tokens"], ","),
                item["tool_ratio"],
                format(item["target_tool_tokens"], ","),
                100 * item["compression_ratio"],
                format(item["windowing_saved_tokens"], ","),
                format(item["spill_saved_tokens"], ","),
                100 * item["peak_occupancy_compressed"],
                "是(%d/24)" % len(item["reduction_turns"]) if item["hard_cut_triggered"] else "否",
                item["memory_file_summaries_compressed"],
            )
        )
    return "\n".join(out) + "\n"


def run_mode_ratio(root, tool_limit, smoke, checkpoint_path):
    message_sizes = MESSAGE_TOKEN_SIZES[:1] if smoke else MESSAGE_TOKEN_SIZES
    tool_ratios = TOOL_RATIOS[:1] if smoke else TOOL_RATIOS
    results = []
    for message_tokens in message_sizes:
        for tool_ratio in tool_ratios:
            started = time.time()
            item = run_cell(message_tokens, tool_ratio, tool_limit, root)
            results.append(item)
            print(
                "[ratio] msg=%5d tool=%.2fx(%6d) -> ratio=%.1f%% windowing_saved=%d spill_saved=%d "
                "peak_occ=%.1f%% hard_cut=%s elapsed=%.0fs"
                % (
                    message_tokens, tool_ratio, item["target_tool_tokens"],
                    100 * item["compression_ratio"], item["windowing_saved_tokens"], item["spill_saved_tokens"],
                    100 * item["peak_occupancy_compressed"], item["hard_cut_triggered"], time.time() - started,
                ),
                flush=True,
            )
            _checkpoint(checkpoint_path, {"tool_output_limit": tool_limit, "results": results})
    return {"tool_output_limit": tool_limit, "results": results}, render_ratio_markdown(results, tool_limit)


# --------------------------------------------------------------------------
# P0-b:可恢复性检查(落盘 vs 直接截断,tool 远超 limit)
# --------------------------------------------------------------------------


def run_mode_recoverability(root, tool_limit, smoke):
    ratio = RECOVERABILITY_RATIO
    target_tool_tokens = int(round(tool_limit * ratio))
    lines_per_window = _calibrate_window_lines(target_tool_tokens, "data/recoverability.txt")
    turns = 1 if smoke else 2

    entries = []
    for label, spill_on in (("spill_on", True), ("spill_off", False)):
        arm_root = root / ("recoverability-%s" % label)
        marker = "RECOVERABILITY-MARKER-%s" % label
        tool_rel_path = _build_tool_tier_file(
            arm_root, lines_per_window, variants=1, tail_markers={0: marker}
        )
        full_flags = dict(get_harness("full").resolved_feature_flags())
        flags = dict(full_flags)
        flags["tool_output_spill"] = spill_on
        outputs = []
        for turn in range(turns):
            outputs.append(
                models.tool_call("read_file", path=tool_rel_path, start=1, end=lines_per_window)
            )
            outputs.append(models.final_answer("turn-%d-done" % turn))
        agent = _build_agent(flags, arm_root, outputs, "ptm-recoverability-%s" % label)
        for turn in range(turns):
            agent.ask("turn %d: read the oversized file" % turn)
        tool_meta = dict(getattr(agent, "_last_tool_result_metadata", None) or {})
        spilled = bool(tool_meta.get("tool_output_spilled"))
        spill_path = str(tool_meta.get("tool_output_spill_path") or "")
        # 进上下文的最后一条 tool 消息里,标记是否可见(应恒为 False——落盘预览
        # 和直接截断都是保留开头,标记刻意放在窗口最后一行)。
        history = agent.session.get("history", [])
        last_tool_text = ""
        for item in reversed(history):
            if isinstance(item, dict) and item.get("role") == "tool":
                last_tool_text = str(item.get("content", ""))
                break
        marker_visible_in_context = marker in last_tool_text
        recovered_ok = False
        if spilled and spill_path:
            # 第一版这里传 {"path": spill_path}(吃默认 end=200)——标记埋在
            # 最后一行,直接截断在前 200 行外,读不到。改成传大 end 覆盖全文
            # 还是错:落盘文件本身就有 lines_per_window 行、原始体量,`read_file`
            # 不认"这是不是落盘产物",只看这次请求的范围有多大——整段读回去
            # 依然超过 tool_output_limit(),会被 `_store_tool_output()` 当成
            # 一次新的超限调用**再落盘一次**,拿到的还是头部预览,标记还是看不到。
            # 正确做法是像真实取回场景一样只要最后几行:窗口小,不会二次触发落盘。
            tail_start = max(1, lines_per_window - 20)
            recovered = agent.run_tool(
                "read_file", {"path": spill_path, "start": tail_start, "end": lines_per_window + 50}
            )
            recovered_ok = marker in recovered
        entries.append(
            {
                "arm": label,
                "tool_output_spill_flag": spill_on,
                "target_tool_tokens": target_tool_tokens,
                "spilled": spilled,
                "spill_path": spill_path,
                "marker_visible_in_context": marker_visible_in_context,
                "recoverable_via_read_file": recovered_ok,
            }
        )
        print(
            "[recoverability] arm=%s spilled=%s marker_in_context=%s recoverable=%s"
            % (label, spilled, marker_visible_in_context, recovered_ok),
            flush=True,
        )
    payload = {"tool_output_limit": tool_limit, "ratio": ratio, "target_tool_tokens": target_tool_tokens, "entries": entries}
    md = [
        "# P0-b:可恢复性检查(tool = %.1fx limit,远超 %s token)" % (ratio, format(tool_limit, ",")),
        "",
        "一句话:超过 `tool_output_limit()` 后,两臂进上下文的预览都不含埋在窗口",
        "**最后一行**的标记(截断和落盘预览都是保留开头)——区别在标记还能不能",
        "从别处找回来。",
        "",
        "| 分支 | `tool_output_spill` | 触发落盘 | 上下文里能看到标记 | 能靠 read_file 读回标记 |",
        "|---|---|---|---|---|",
    ]
    for e in entries:
        md.append(
            "| %s | %s | %s | %s | %s |"
            % (
                e["arm"], e["tool_output_spill_flag"], e["spilled"],
                e["marker_visible_in_context"], e["recoverable_via_read_file"],
            )
        )
    md.append("")
    return payload, "\n".join(md) + "\n"


# --------------------------------------------------------------------------
# P1-b:窗口档位扫描(compressed 臂,不涉及 raw 基线问题)
# --------------------------------------------------------------------------


def run_mode_window_sweep(root, smoke, checkpoint_path=None):
    windows = WINDOW_TIERS[-1:] if smoke else WINDOW_TIERS
    ratios = WINDOW_SWEEP_RATIOS[:1] if smoke else WINDOW_SWEEP_RATIOS
    results = []
    for window_tokens in windows:
        probe_root = root / ("sweep-probe-%d" % window_tokens)
        _build_tool_tier_file(probe_root, 4, rel_path="data/probe.txt")
        probe_agent = _build_agent(
            dict(get_harness("full").resolved_feature_flags()), probe_root,
            [models.final_answer("noop")], "ptm-sweep-probe-%d" % window_tokens,
            context_window=window_tokens,
        )
        window_tool_limit = _tool_output_limit(int(probe_agent.context_budget))
        for tool_ratio in ratios:
            target_tool_tokens = int(round(window_tool_limit * tool_ratio))
            lines_per_window = _calibrate_window_lines(target_tool_tokens, "data/sweep.txt")
            tag = "win%d_tool%.2fx" % (window_tokens, tool_ratio)
            cell_root = root / tag
            tool_rel_path = _build_tool_tier_file(cell_root, lines_per_window)
            full_flags = dict(get_harness("full").resolved_feature_flags())
            started = time.time()
            outputs = _scripted_read_outputs(tool_rel_path, lines_per_window, TURNS)
            agent = _build_agent(full_flags, cell_root, outputs, "ptm-sweep-%s" % tag, context_window=window_tokens)
            budget = int(agent.context_budget)
            rows = _run_turns(agent, budget, WINDOW_SWEEP_MESSAGE_TOKENS, TURNS)
            spilled_turns = [r["turn"] for r in rows if r["tool_output_spilled"]]
            hard_cut_turns = [r["turn"] for r in rows if r["reductions"]]
            item = {
                "window_tokens": window_tokens,
                "tool_output_limit": window_tool_limit,
                "tool_ratio": tool_ratio,
                "target_tool_tokens": target_tool_tokens,
                "budget_tokens": budget,
                "peak_occupancy": max(r["occupancy"] for r in rows),
                "spilled": bool(spilled_turns),
                "spill_turn_count": len(spilled_turns),
                "hard_cut_triggered": bool(hard_cut_turns),
                "hard_cut_turn_count": len(hard_cut_turns),
            }
            results.append(item)
            print(
                "[window-sweep] window=%6d tool=%.2fx(%6d,limit=%6d) -> peak_occ=%.1f%% "
                "spilled=%s(%d) hard_cut=%s(%d) elapsed=%.0fs"
                % (
                    window_tokens, tool_ratio, target_tool_tokens, window_tool_limit,
                    100 * item["peak_occupancy"], item["spilled"], item["spill_turn_count"],
                    item["hard_cut_triggered"], item["hard_cut_turn_count"], time.time() - started,
                ),
                flush=True,
            )
            _checkpoint(checkpoint_path, {"message_tokens": WINDOW_SWEEP_MESSAGE_TOKENS, "results": results})
    payload = {"message_tokens": WINDOW_SWEEP_MESSAGE_TOKENS, "results": results}
    md = [
        "# P1-b:窗口档位扫描(8k/32k/128k,compressed 臂,message 固定 %s token)"
        % format(WINDOW_SWEEP_MESSAGE_TOKENS, ","),
        "",
        "一句话:只看 compressed 臂自己的行为(不比 raw),`tool_output_limit()` 随",
        "窗口收缩后,同一个相对比例(如 1.5x)在小窗口下是不是更容易落盘/触发硬裁。",
        "",
        "| 窗口档 | tool_output_limit(该档) | tool 档(比例/token) | 峰值占用 | 落盘轮次 | 硬裁轮次 |",
        "|---|---|---|---|---|---|",
    ]
    for item in results:
        md.append(
            "| %s | %s | %.2fx (%s) | %.1f%% | %d/24 | %d/24 |"
            % (
                format(item["window_tokens"], ","), format(item["tool_output_limit"], ","),
                item["tool_ratio"], format(item["target_tool_tokens"], ","),
                100 * item["peak_occupancy"], item["spill_turn_count"], item["hard_cut_turn_count"],
            )
        )
    return payload, "\n".join(md) + "\n"


# --------------------------------------------------------------------------
# P2-a:会话摘要触发场景探针
# --------------------------------------------------------------------------


def run_mode_summary_trigger(root, tool_limit, smoke):
    turns = 8 if smoke else SUMMARY_PROBE_TURNS
    target_tool_tokens = int(round(tool_limit * SUMMARY_PROBE_TOOL_RATIO))
    lines_per_window = _calibrate_window_lines(target_tool_tokens, "data/summary_probe.txt")
    tool_rel_path = _build_tool_tier_file(root, lines_per_window)
    full_flags = dict(get_harness("full").resolved_feature_flags())
    outputs = _scripted_read_outputs(tool_rel_path, lines_per_window, turns)
    started = time.time()
    agent = _build_agent(full_flags, root, outputs, "ptm-summary-probe")
    budget = int(agent.context_budget)
    rows = _run_turns(agent, budget, SUMMARY_PROBE_MESSAGE_TOKENS, turns)
    summary_turns = [r["turn"] for r in rows if r["summary_compactions"]]
    elapsed = time.time() - started
    payload = {
        "turns": turns,
        "message_tokens": SUMMARY_PROBE_MESSAGE_TOKENS,
        "tool_ratio": SUMMARY_PROBE_TOOL_RATIO,
        "target_tool_tokens": target_tool_tokens,
        "peak_occupancy": max(r["occupancy"] for r in rows),
        "summary_turns": summary_turns,
        "session_summary_triggered": bool(summary_turns),
        "elapsed_seconds": round(elapsed, 1),
    }
    print(
        "[summary-trigger] turns=%d msg=%d tool=%.2fx -> peak_occ=%.1f%% summary_triggered=%s(%d/%d) elapsed=%.0fs"
        % (
            turns, SUMMARY_PROBE_MESSAGE_TOKENS, SUMMARY_PROBE_TOOL_RATIO,
            100 * payload["peak_occupancy"], payload["session_summary_triggered"],
            len(summary_turns), turns, elapsed,
        ),
        flush=True,
    )
    md = [
        "# P2-a:会话摘要触发场景探针(%d 轮,message 固定 %d token,tool 固定 %.1fx)"
        % (turns, SUMMARY_PROBE_MESSAGE_TOKENS, SUMMARY_PROBE_TOOL_RATIO),
        "",
        "一句话:主矩阵(message 持续增长)16 格里 `session_summary` 全部 0 次触发——",
        "这不是 bug,是已知条件复现(受保护尾巴本身超过目标点时覆盖点顶死);",
        "这里换成\"轮次多、单轮增量恒定且小\"的负载,专门验证条件不成立时它会不会触发。",
        "",
        "触发:**%s**(%d/%d 轮),峰值占用 %.1f%%。"
        % ("是" if payload["session_summary_triggered"] else "否", len(summary_turns), turns, 100 * payload["peak_occupancy"]),
    ]
    return payload, "\n".join(md) + "\n"


# --------------------------------------------------------------------------
# P2-b:过期读作废(stale-read)探针
# --------------------------------------------------------------------------


def run_mode_stale_read(root, smoke):
    entries = []
    for label, invalidation_on in (("invalidation_on", True), ("invalidation_off", False)):
        arm_root = root / ("stale-%s" % label)
        rel_path = "data/stale.txt"
        path = arm_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("version-1 marker\n" * 5, encoding="utf-8")

        full_flags = dict(get_harness("full").resolved_feature_flags())
        flags = dict(full_flags)
        flags["stale_read_invalidation"] = invalidation_on
        # 两次 read_file 必须用**不同的行区间**(dedup key 是 `(path,start,end)`,
        # 见 `context_manager._read_dedup_key`)。同一区间读两次会先被"重复读取
        # 去重"机制吞掉(旧的整条 `POLICY_DROPPED`),根本轮不到过期读判定——
        # 这不是缺陷,是"模型重读了同一段,旧的自然没用了"这个更强的结论优先;
        # 过期读要测的是"没重读、旧内容还留在上下文里"这个场景,所以第二次读
        # 换一个区间,不覆盖第一次读过的那段。
        outputs = [
            models.tool_call("read_file", path=rel_path, start=1, end=5),
            models.final_answer("read-1-done"),
            models.tool_call("write_file", path=rel_path, content="version-2 marker\n" * 5),
            models.final_answer("write-done"),
            models.tool_call("read_file", path=rel_path, start=1, end=3),
            models.final_answer("read-2-done"),
        ]
        agent = _build_agent(
            flags, arm_root, outputs, "ptm-stale-%s" % label,
            tools_allowlist=("read_file", "write_file"), max_steps=2,
        )
        budget = int(agent.context_budget)
        for turn, msg in enumerate(["read once", "overwrite the file", "read again"]):
            agent.ask("turn %d: %s" % (turn, msg))
        _, _, metadata = ContextManager(agent, total_budget=budget).build_all("continue")
        history_meta = metadata.get("history") or {}
        entries.append(
            {
                "arm": label,
                "stale_read_invalidation_flag": invalidation_on,
                "stale_read_count": int(history_meta.get("stale_read_count") or 0),
                "stale_pointer_count": int(history_meta.get("stale_pointer_count") or 0),
            }
        )
        print(
            "[stale-read] arm=%s flag=%s stale_read_count=%d stale_pointer_count=%d"
            % (label, invalidation_on, entries[-1]["stale_read_count"], entries[-1]["stale_pointer_count"]),
            flush=True,
        )
    payload = {"entries": entries}
    md = [
        "# P2-b:过期读作废(stale-read)探针",
        "",
        "一句话:读一个文件 → 覆盖同一路径 → 再读一次;开关打开时第一次读的记录",
        "应该被标记为过期(`stale_read_count=1`),关掉时不应该标记(退回改动之前",
        "的行为)。",
        "",
        "| 分支 | `stale_read_invalidation` | stale_read_count | stale_pointer_count |",
        "|---|---|---|---|",
    ]
    for e in entries:
        md.append(
            "| %s | %s | %d | %d |"
            % (e["arm"], e["stale_read_invalidation_flag"], e["stale_read_count"], e["stale_pointer_count"])
        )
    return payload, "\n".join(md) + "\n"


# --------------------------------------------------------------------------
# P2-b:受限编排(run_plan / PTC)探针
# --------------------------------------------------------------------------


def run_mode_ptc(root, smoke):
    variants = 2 if smoke else WINDOW_VARIANTS
    lines_per_window = 40  # 小文件即可,这里测的是round trip/token省了多少,不是落盘
    _build_tool_tier_file(root / "ptc", lines_per_window, variants=variants)

    # 基线:N 轮各自单独 read_file。
    baseline_root = root / "ptc-baseline"
    baseline_path = _build_tool_tier_file(baseline_root, lines_per_window, variants=variants)
    baseline_outputs = _scripted_read_outputs(baseline_path, lines_per_window, variants, variants=variants)
    baseline_agent = _build_agent(
        dict(get_harness("full").resolved_feature_flags()), baseline_root, baseline_outputs, "ptm-ptc-baseline",
    )
    baseline_budget = int(baseline_agent.context_budget)
    baseline_rows = _run_turns(baseline_agent, baseline_budget, 0, variants)

    # PTC:1 轮,一段 run_plan 程序读完全部窗口,只 print 每次结果的长度
    # (演示"过滤模式":结果不回显全文,只回显 print 出来的东西)。
    plan_root = root / "ptc-plan"
    plan_path = _build_tool_tier_file(plan_root, lines_per_window, variants=variants)
    plan_lines = []
    for k in range(variants):
        start = k * lines_per_window + 1
        end = start + lines_per_window - 1
        plan_lines.append("r = read_file(path=%r, start=%d, end=%d)" % (plan_path, start, end))
        plan_lines.append("print(len(r))")
    plan_source = "\n".join(plan_lines)
    plan_flags = dict(get_harness("full").resolved_feature_flags())
    plan_flags["plan_tool"] = True
    plan_outputs = [models.tool_call("run_plan", plan=plan_source), models.final_answer("plan-done")]
    plan_agent = _build_agent(
        plan_flags, plan_root, plan_outputs, "ptm-ptc-plan",
        tools_allowlist=("read_file",), max_steps=variants + 2,
    )
    plan_budget = int(plan_agent.context_budget)
    plan_rows = _run_turns(plan_agent, plan_budget, 0, 1)

    payload = {
        "variants": variants,
        "baseline": {
            "round_trips": len(baseline_rows),
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in baseline_rows),
        },
        "plan": {
            "round_trips": len(plan_rows),
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in plan_rows),
            "plan_source": plan_source,
        },
    }
    print(
        "[ptc] baseline round_trips=%d tokens=%d | plan round_trips=%d tokens=%d"
        % (
            payload["baseline"]["round_trips"], payload["baseline"]["total_prompt_tokens"],
            payload["plan"]["round_trips"], payload["plan"]["total_prompt_tokens"],
        ),
        flush=True,
    )
    md = [
        "# P2-b:受限编排(run_plan/PTC)探针",
        "",
        "一句话:同样读 %d 个窗口,基线是 %d 次模型往返;`run_plan` 把它压进 1 次" % (variants, variants),
        "往返(每个内层调用仍各计一步、各过闸口,只是不再各花一次约 17 秒的模型",
        "调用)。用 `print(len(r))` 演示过滤模式:回给模型的不是 4 段窗口全文,",
        "是 4 个数字。",
        "",
        "| 分支 | 模型往返次数 | 累计 prompt token |",
        "|---|---|---|",
        "| 基线(逐次 read_file) | %d | %s |" % (payload["baseline"]["round_trips"], format(payload["baseline"]["total_prompt_tokens"], ",")),
        "| run_plan(1 次,过滤模式) | %d | %s |" % (payload["plan"]["round_trips"], format(payload["plan"]["total_prompt_tokens"], ",")),
        "",
        "```python",
        plan_source,
        "```",
    ]
    return payload, "\n".join(md) + "\n"


# --------------------------------------------------------------------------
# 落盘与入口
# --------------------------------------------------------------------------


def _checkpoint(path, payload):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["complete"] = False
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _finalize(path, payload):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["complete"] = True
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote %s" % path)


MODES = ("ratio", "recoverability", "window-sweep", "summary-trigger", "stale-read", "ptc")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=MODES + ("all",), default="all")
    parser.add_argument("--output-json", default="", help="单组模式(--mode 不是 all)时的输出路径")
    parser.add_argument("--output-markdown", default="", help="单组模式(--mode 不是 all)时的输出路径")
    parser.add_argument("--output-dir", default="artifacts", help="--mode all 时,每组各写一份到这个目录")
    parser.add_argument("--smoke", action="store_true", help="每组都缩到最小规模,验证脚本本身跑得通")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    # 只是为了拿一次真实的 agent.context_budget(128k 档),不参与后面的正式跑批。
    probe_root = Path(tempfile.mkdtemp(prefix="cfm-ptm-probe-"))
    _build_tool_tier_file(probe_root, 4)
    probe_agent = _build_agent(
        dict(get_harness("full").resolved_feature_flags()), probe_root, [models.final_answer("noop")], "ptm-probe"
    )
    tool_limit = _tool_output_limit(int(probe_agent.context_budget))

    root = Path(tempfile.mkdtemp(prefix="cfm-ptm-"))
    modes = list(MODES) if args.mode == "all" else [args.mode]

    for mode in modes:
        mode_root = root / mode
        mode_root.mkdir(parents=True, exist_ok=True)
        print("\n=== running mode: %s ===" % mode, flush=True)
        started = time.time()

        if args.mode == "all":
            json_path = Path(args.output_dir) / ("tool-pressure-%s.json" % mode)
            md_path = Path(args.output_dir) / ("tool-pressure-%s.md" % mode)
        else:
            json_path = Path(args.output_json) if args.output_json else None
            md_path = Path(args.output_markdown) if args.output_markdown else None

        if mode == "ratio":
            payload, markdown = run_mode_ratio(mode_root, tool_limit, args.smoke, json_path)
        elif mode == "recoverability":
            payload, markdown = run_mode_recoverability(mode_root, tool_limit, args.smoke)
        elif mode == "window-sweep":
            payload, markdown = run_mode_window_sweep(mode_root, args.smoke, json_path)
        elif mode == "summary-trigger":
            payload, markdown = run_mode_summary_trigger(mode_root, tool_limit, args.smoke)
        elif mode == "stale-read":
            payload, markdown = run_mode_stale_read(mode_root, args.smoke)
        elif mode == "ptc":
            payload, markdown = run_mode_ptc(mode_root, args.smoke)
        else:
            raise ValueError("unknown mode: %s" % mode)

        print("\n" + markdown, flush=True)
        print("[%s] elapsed=%.0fs" % (mode, time.time() - started), flush=True)

        _finalize(json_path, payload)
        if md_path:
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(markdown, encoding="utf-8")
            print("wrote %s" % md_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
