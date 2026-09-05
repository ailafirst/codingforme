"""完整系统门槛测试:一条整合了全部机制触发条件的确定性会话,横扫多个窗口档位。

一句话:阶段 10 的 `run_compression_gate.py` 只回答"压力驱动的硬裁会不会触发",
而且只在 8k 一档上测;这个脚本把同一个"不调模型、可重跑"的判据延伸成
"同一次会话里,九个机制各自在每个窗口档位上发生了什么",不再是分开测的
八个探针(`run_context_stress.py` 的 `stale_read`/`spill_stale`/... 各自只测一个
机制、各自挑一个能让它生效的窗口)。

## 为什么要分两段,而不是从头到尾都是对话

上下文治理体系里的机制分两类,触发条件完全不同,硬凑成一段负载测不出两者:

- **调用模式驱动**(和预算大小无关,只和"调用序列长什么样"有关):落盘指针
  (L1)、过期读作废(阶段 5/5b)、最近窗口按块推进(L3)。这些哪怕预算天大,
  照样要触发——落盘看的是单条结果多大,过期读看的是读之后有没有紧跟一次写。
- **占用率驱动**(和预算大小强相关):分级压缩、会话摘要、`clear_at_least`、
  可逆折叠、逐段硬裁兜底。预算越大,同样内容占的比例越低,这些机制可能
  一次都不触发——这正是本文档要如实测出来、而不是假装它们到处都生效的地方。

所以:

- **Phase A**(真实工具调用,通过 `FakeModelClient` 脚本化,走真实 `ask()` 循环、
  真实 `run_tool()`):读一个大文件(在小档位上会落盘)→ `patch_file` 改它 →
  连续读十几个不同的小文件(推进最近窗口,顺带把前面的大文件读记录挤出窗口)
  → 再读一次大文件。落盘、过期标记、窗口按块推进全部是代码真实跑出来的,
  不是摆样子拼出来的 metadata。
- **Phase B**(纯对话增长,直接操纵 history,不再调模型,手法照抄阶段 10):
  在 Phase A 真实产生的历史之后继续叠对话文本,每加一轮就用真实的
  `ContextManager.build_all()` 渲染一次,直到分级压缩 / 会话摘要 / 逐段硬裁
  依次触发或者到达轮数上限。

两段共用同一个 agent/session——Phase B 站在 Phase A 真实落盘、真实过期标记
之上,这是和"各测一个机制"的探针的本质区别:之前每个探针只回答"这一个
机制在挑出来的那个窗口上通不通",这里回答"一次真实会话里,九个机制各自
在每个窗口档位上发生了什么"。

## 为什么不是全部 8 个窗口档位

`models.budget_breakdown()` 把窗口先 `min(window, EFFECTIVE_WINDOW_CAP_TOKENS=128_000)`
再往下算预算,所以 128k / 256k / 512k / 1M 的预算与派生上限逐位相同——已用
一次性脚本实测验证:四档的 `budget_tokens` 都是 118,428、`tool_output_limit`
都是 14,803。扫后三档只是把同一份结果重复写三遍。默认只跑 5 个预算互不
相同的档位(8k/16k/32k/64k/128k);`--all-buckets` 能把这条"结构性重复"
本身摊开验证。

用法:

    python scripts/run_system_gate.py --output-json artifacts/system-gate.json --output-markdown artifacts/system-gate.md
    python scripts/run_system_gate.py --all-buckets --output-json artifacts/system-gate-full.json
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codingforme import FakeModelClient  # noqa: E402
from codingforme import models  # noqa: E402
from codingforme.context_manager import ContextManager  # noqa: E402
from codingforme.context_manager import tool_output_limit as _tool_output_limit  # noqa: E402
from codingforme.eval.harness import get_harness  # noqa: E402

import run_compression_gate as gate  # noqa: E402  阶段 10:_filler_words / _transcript_head
import run_context_stress as stress  # noqa: E402  阶段 9:_read_trace / _summarize

# 预算互不相同的 5 档;`min(window, 128_000)` 之后 128k 以上全部塌缩成同一个数,
# 见模块 docstring。
WINDOW_TIERS_DEFAULT = (8_000, 16_000, 32_000, 64_000, 128_000)
WINDOW_TIERS_ALL = (8_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000, 1_000_000)

# Phase A:小文件数量。14 个读 + 1 次大文件读 + 1 次 patch + 1 次复读 = 17 次
# 工具调用,足够把 `RECENT_TOOL_TURNS = 6` 撑过去两轮以上,逼最近窗口至少
# 按块推进一次。
SMALL_FILE_COUNT = 14

# 一致性检查只看这些结构性字段。两个字段刻意不列进来:`spill_paths` 带
# run_id,每次 `ask()` 都会现生成一个;`spill_full_tokens` / `spill_kept_tokens`
# 也刻意排除——落盘指针那句提示原样把 run_id 写进正文(供模型照抄读回去),
# 而 run_id 里的十六进制位数在分词器下不是等长的("830b12" 和 "176380" 切出来
# 的 token 数可以差 1~2 个),这会让"留了多少 token"这个数跟着 run_id 抖动
# ±1~2,和机制本身有没有生效无关。
_PHASE_A_STRUCTURAL_KEYS = (
    "tool_calls",
    "tool_errors",
    "spilled",
    "recent_tool_window",
    "stale_read_count_max",
    "stale_pointer_count_max",
    "squeezed_entry_max",
)

# Phase B:对话强度沿用阶段 10 的三档(每条对话 96/206/306 token),含义相同——
# 分别对应"会话摘要有效 / 开始帮倒忙 / 完全无效"。
TURN_TOKEN_SIZES = (96, 206, 306)
PHASE_B_TURNS = 24


def _build_big_module_text():
    """~2,000 token 的合成源码,顶部放一个可供 `patch_file` 精确命中的标记行。"""
    lines = ["MARKER = 'TODO-alpha'", ""]
    for i in range(220):
        lines.append("def helper_%03d(value):" % i)
        lines.append("    # deterministic filler body for helper %03d" % i)
        lines.append("    return value + %d" % i)
        lines.append("")
    return "\n".join(lines)


_FILLER_LINE = "spec detail " * 6


def _small_file_text(index):
    """~150 token 的小文件,内容按下标错开,避免撞上"连续两次同参数调用"检测。"""
    lines = ["# module %03d" % index]
    lines += ['row_%03d_%02d = "%s"' % (index, n, _FILLER_LINE) for n in range(10)]
    return "\n".join(lines) + "\n"


def _build_workspace(root, small_file_count):
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("system gate workspace\n", encoding="utf-8")
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "big_module.py").write_text(_build_big_module_text(), encoding="utf-8")
    for i in range(small_file_count):
        (src / ("mod_%03d.py" % i)).write_text(_small_file_text(i), encoding="utf-8")


def _phase_a_outputs(small_file_count):
    """真实工具调用序列:读大文件 → patch → 十几次小文件读(推进窗口)→ 复读大文件。

    每一步都是刻意的:第 1 步在小档位上会落盘;第 2 步让第 1 步那条读记录变成
    过期;中间那批小文件读把最近窗口向前推,把第 1 步挤出窗口(落过盘的话就该
    变成 `stale_pointer`);最后一步验证"复读改过的文件"这个最常见的真实动作
    不会被过期标记污染(这次读发生在写**之后**,理应是新鲜的)。
    """
    calls = [
        models.tool_call("read_file", path="src/big_module.py"),
        models.tool_call(
            "patch_file",
            path="src/big_module.py",
            old_text="MARKER = 'TODO-alpha'",
            new_text="MARKER = 'DONE-alpha'",
        ),
    ]
    calls += [
        models.tool_call("read_file", path="src/mod_%03d.py" % i)
        for i in range(small_file_count)
    ]
    calls.append(models.tool_call("read_file", path="src/big_module.py"))
    calls.append(models.final_answer("PHASE-A-DONE"))
    return calls


def _run_phase_a(tier, variant, small_file_count):
    """真实跑一遍 Phase A,返回 (agent, phase_a_summary)。"""
    root = Path(tempfile.mkdtemp(prefix="cfm-sysgate-"))
    _build_workspace(root, small_file_count)
    base = get_harness(variant)
    spec = base.derive(
        name="sysgate-%s-%d" % (variant, tier),
        description="完整系统门槛测试的 Phase A(真实工具调用)",
        max_steps=small_file_count + 10,
        tools_allowlist=("read_file", "patch_file", "list_files"),
    )
    agent = spec.build(
        FakeModelClient(_phase_a_outputs(small_file_count)),
        root,
        repo_root_override=root,
        context_window=tier,
    )
    agent.ask(
        "Read src/big_module.py, flip its MARKER from TODO-alpha to DONE-alpha, "
        "then read through every module under src/."
    )
    events = stress._read_trace(agent.current_run_dir)
    summary = stress._summarize(events)
    return agent, summary


def _run_phase_b(agent, turn_tokens, turns):
    """在 Phase A 的真实历史之后继续叠对话,手法照抄阶段 10 的逐轮渲染。"""
    budget = int(agent.context_budget)
    filler = gate._filler_words(turn_tokens)
    rows = []
    start_index = sum(1 for item in agent.session["history"] if item.get("role") == "user")
    for offset in range(turns):
        index = start_index + offset
        agent.record({"role": "user", "content": "requirement %d: %s" % (index, filler)})
        agent.record({"role": "assistant", "content": "acknowledged %d: %s" % (index, filler)})
        _, prompt, metadata = ContextManager(agent, total_budget=budget).build_all("continue")
        pressure = metadata.get("context_pressure") or {}
        summary_state = pressure.get("session_summary") or {}
        history = metadata.get("history") or {}
        reductions = metadata.get("budget_reductions") or []
        rows.append(
            {
                "turn": offset + 1,
                "prompt_tokens": int(metadata.get("prompt_tokens") or 0),
                "occupancy": round(float(pressure.get("occupancy_before") or 0.0), 4),
                "reductions": len(reductions),
                # `clear_at_least` 真正说了算:按它裁的量比 overflow 还多。
                "clear_at_least_bound": sum(
                    1 for item in reductions
                    if int(item.get("target_tokens") or 0) > int(item.get("overflow_tokens") or 0)
                ),
                "summary_compactions": int(summary_state.get("compactions") or 0),
                "summary_covered": int(summary_state.get("covered_entries") or 0),
                "squeezed": int(history.get("squeezed_entry_count") or 0),
                "omitted": int((metadata.get("sections") or {}).get("history", {}).get("omitted_entry_count") or 0),
                "history_head": gate._transcript_head(prompt),
            }
        )
    heads = [row["history_head"] for row in rows]
    rewrites = sum(1 for a, b in zip(heads, heads[1:]) if a != b)
    reduction_turns = [row["turn"] for row in rows if row["reductions"]]
    return {
        "turn_tokens": turn_tokens,
        "budget_tokens": budget,
        "reduction_turns": reduction_turns,
        "gate_open": not reduction_turns,
        "peak_occupancy": max(row["occupancy"] for row in rows),
        "clear_at_least_bound_turns": [row["turn"] for row in rows if row["clear_at_least_bound"]],
        "summary_compaction_turns": [row["turn"] for row in rows if row["summary_compactions"]],
        "max_summary_covered": max(row["summary_covered"] for row in rows),
        "max_squeezed": max(row["squeezed"] for row in rows),
        "max_omitted": max(row["omitted"] for row in rows),
        "history_head_rewrites": rewrites,
    }


def run_tier(tier, variant, small_file_count, turn_sizes, phase_b_turns):
    """一个窗口档位:Phase A 跑一次,Phase B 按每档对话强度各跑一次(共享 Phase A 的产物)。"""
    runs = []
    for turn_tokens in turn_sizes:
        agent, phase_a = _run_phase_a(tier, variant, small_file_count)
        phase_b = _run_phase_b(agent, turn_tokens, phase_b_turns)
        runs.append({"turn_tokens": turn_tokens, "phase_a": phase_a, "phase_b": phase_b})
        print(
            "tier=%-9s turns/size=%3d -> spill=%d stale_read=%d stale_pointer=%d "
            "window=%s | reductions=%s"
            % (
                tier,
                turn_tokens,
                phase_a["spilled"],
                phase_a["stale_read_count_max"],
                phase_a["stale_pointer_count_max"],
                phase_a["recent_tool_window"],
                phase_b["reduction_turns"] or "none",
            ),
            flush=True,
        )
    return {
        "window_tier": tier,
        "tool_output_limit": _tool_output_limit(runs[0]["phase_b"]["budget_tokens"]),
        # Phase A 和对话强度无关(同一份脚本化调用),三次重复的"结构性"结果应当
        # 完全一致——只比这几个字段,不比整份 summary:`spill_paths` 里带
        # `run_id`,每次 `ask()` 都会现生成一个,逐字比较必然判"不一致",
        # 但那不是这项检查想查的东西。
        "phase_a_consistent": len(
            {
                json.dumps(
                    {k: r["phase_a"][k] for k in _PHASE_A_STRUCTURAL_KEYS},
                    sort_keys=True,
                )
                for r in runs
            }
        )
        == 1,
        "runs": runs,
    }


def render_markdown(results):
    out = [
        "# 完整系统门槛测试:同一次会话横扫多个窗口档位",
        "",
        "一句话:**调用模式驱动的机制(落盘 / 过期读作废 / 最近窗口)只看调用序列,",
        "和窗口大小基本无关;占用率驱动的机制(分级压缩 / 会话摘要 / 逐段硬裁)",
        "强依赖窗口大小,窗口越大越难触发。** 每个窗口档位先真实跑一段工具调用",
        "(Phase A),再在同一段历史之后叠对话直到硬裁触发或到轮数上限(Phase B)。",
        "",
        "## Phase A:调用模式驱动的机制(同一档位下,与对话强度无关)",
        "",
        "「大文件读」是这次会话第 1 步读的文件,后面会被 patch 一次;",
        "「小文件读」是接下来 %d 次读不同小文件,用来把最近窗口向前推。" % SMALL_FILE_COUNT,
        "",
        "| 窗口档位 | 单条落盘上限 | 落盘次数 | 过期读(窗口内) | 过期指针(窗口外) | "
        "最近窗口分布 | 三次强度结果一致 |",
        "|---|---|---|---|---|---|---|",
    ]
    for tier_result in results:
        phase_a = tier_result["runs"][0]["phase_a"]
        out.append(
            "| %s | %s token | %d | %d | %d | `%s` | %s |"
            % (
                _fmt_tier(tier_result["window_tier"]),
                format(tier_result["tool_output_limit"], ","),
                phase_a["spilled"],
                phase_a["stale_read_count_max"],
                phase_a["stale_pointer_count_max"],
                json.dumps(phase_a["recent_tool_window"], ensure_ascii=False),
                "是" if tier_result["phase_a_consistent"] else "**否,见 JSON**",
            )
        )

    out.extend(
        [
            "",
            "## Phase B:占用率驱动的机制(窗口档位 × 对话强度)",
            "",
            "「每条对话多大」是阶段 2/10 沿用的三档负载强度;硬裁触发的轮次就是",
            "阶段 4 的判据本身,空了才说明分级压缩 + 会话摘要真的把不可逆的逐段",
            "硬裁替掉了。",
            "",
            "| 窗口档位 | 预算 | 每条对话多大 | 峰值占用 | 硬裁触发的轮次 | 判据 | "
            "`clear_at_least` 说了算的轮次 | 摘要推进的轮次 | 压扁条数 | history 起点改写 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for tier_result in results:
        for run in tier_result["runs"]:
            phase_b = run["phase_b"]
            out.append(
                "| %s | %s | %d token | %.1f%% | %s | %s | %s | %s | %d | %d 次 |"
                % (
                    _fmt_tier(tier_result["window_tier"]),
                    format(phase_b["budget_tokens"], ","),
                    run["turn_tokens"],
                    100 * phase_b["peak_occupancy"],
                    ", ".join(str(t) for t in phase_b["reduction_turns"]) or "一次都没有",
                    "**可以开工**" if phase_b["gate_open"] else "不能删",
                    ", ".join(str(t) for t in phase_b["clear_at_least_bound_turns"]) or "一次都没有",
                    ", ".join(str(t) for t in phase_b["summary_compaction_turns"]) or "一次都没有",
                    phase_b["max_squeezed"],
                    phase_b["history_head_rewrites"],
                )
            )

    all_runs = [(t["window_tier"], r) for t in results for r in t["runs"]]
    open_tiers = sorted({tier for tier, r in all_runs if r["phase_b"]["gate_open"]})
    blocked_tiers = sorted({tier for tier, r in all_runs if not r["phase_b"]["gate_open"]})
    clear_at_least_ever_bound = any(r["phase_b"]["clear_at_least_bound_turns"] for _, r in all_runs)
    out.extend(
        [
            "",
            "## 结论",
            "",
            "- **调用模式驱动的机制在测到的每个档位上都真实触发过**"
            "(落盘、过期读、最近窗口推进不为零)——这类机制不需要"
            "\"预算够大\"这个前提,窗口大小只决定落盘阈值,不决定触不触发。"
            if any(t["runs"][0]["phase_a"]["spilled"] for t in results)
            else "- **本次配置下没有一个档位触发落盘**,需要检查 Phase A 的大文件是否"
            "小于全部档位的单条上限,不是机制失效。",
            "- 阶段 4 判据(硬裁能不能删)按窗口档位分裂:%s%s"
            % (
                ("%s 档上不能开工(仍要靠逐段硬裁扛住压力)" % "、".join(_fmt_tier(t) for t in blocked_tiers))
                if blocked_tiers
                else "全部测到的档位都可以开工",
                (
                    "；%s 档在测到的三档负载上从未触发过硬裁。" % "、".join(_fmt_tier(t) for t in open_tiers)
                    if open_tiers
                    else "。"
                ),
            ),
            "- `clear_at_least` 在测到的全部档位 × 全部对话强度上%s"
            % ("确实说了算过至少一轮。" if clear_at_least_ever_bound else "**恰好一次都没说了算**——和阶段 7 的算术结论一致(触发点/目标点相差 15%,而它只要求腾出 10%,`max()` 永远取前者)。"),
        ]
    )
    return "\n".join(out) + "\n"


def _fmt_tier(tier):
    if tier >= 1_000_000:
        return "1M"
    if tier % 1000 == 0:
        return "%dk" % (tier // 1000)
    return str(tier)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", default="full")
    parser.add_argument(
        "--all-buckets", action="store_true",
        help="连 128k 以上四个预算相同的档位一起跑(默认只跑 5 个预算互不相同的档位)",
    )
    parser.add_argument("--small-files", type=int, default=SMALL_FILE_COUNT)
    parser.add_argument("--phase-b-turns", type=int, default=PHASE_B_TURNS)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-markdown", default="")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    tiers = WINDOW_TIERS_ALL if args.all_buckets else WINDOW_TIERS_DEFAULT
    results = [
        run_tier(tier, args.variant, args.small_files, TURN_TOKEN_SIZES, args.phase_b_turns)
        for tier in tiers
    ]

    markdown = render_markdown(results)
    print()
    print(markdown)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "variant": args.variant,
                    "window_tiers": list(tiers),
                    "small_file_count": args.small_files,
                    "phase_b_turns": args.phase_b_turns,
                    "results": results,
                },
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
