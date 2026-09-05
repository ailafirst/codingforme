"""完整系统门槛测试(`scripts/run_system_gate.py`)的**形状**不变量。

和 `test_stress_probes.py` 同一个理由:脚本不是库代码，正常不该有测试；这几条锁的
是"如果被顺手整理掉，整份系统门槛测试会静默测不到它本来要测的东西"那几处结构性
前提，不是脚本的完整行为。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_system_gate as sysgate  # noqa: E402

from codingforme import FakeModelClient, models  # noqa: E402
from codingforme.context_manager import RECENT_TOOL_TURNS, RECENT_WINDOW_BLOCK  # noqa: E402
from codingforme.eval.harness import get_harness  # noqa: E402


def test_the_patch_anchor_in_the_big_module_is_unique():
    """`patch_file` 要求 `old_text` 恰好命中一次；这里的锚点是那一整行。"""
    text = sysgate._build_big_module_text()
    assert text.count("MARKER = 'TODO-alpha'") == 1


def test_the_big_module_spills_at_small_tiers_but_not_at_large_ones(tmp_path):
    """Phase A 能不能测到落盘,取决于大文件读出来是不是超过当前档位的单条上限——
    这条断言把这个前提钉死,而不是留给"跑起来看看"。

    **必须查真实的 `tool_output_spilled` 标记,不能拿裁剪后的返回值去比 token 数**:
    `run_tool()` 会在返回前就把超限结果裁到 `tool_output_limit()` 附近(留出指针的
    位置),所以"裁剪后的结果有多少 token"这个数无论原始内容多大都卡在同一个上限
    附近,拿它去判断"有没有落盘"必然得出"从来没有"这个假结论。落盘与否只能看
    `_last_tool_result_metadata["tool_output_spilled"]` 这个布尔量本身。
    也**不能量文件本身**:`read_file` 不传 `end` 时默认只读前 200 行
    (`tools.tool_read_file`),这个大文件有 880 行,量整份文件同样会把门槛算错。
    """
    sysgate._build_workspace(tmp_path, 1)
    spec = get_harness("full").derive(
        name="test-big-module-spill", tools_allowlist=("read_file",)
    )

    small_agent = spec.build(
        FakeModelClient([]), tmp_path, repo_root_override=tmp_path, context_window=8_000
    )
    small_agent.run_tool("read_file", {"path": "src/big_module.py"})
    assert small_agent._last_tool_result_metadata["tool_output_spilled"], "小档位测不到落盘,大文件不够大"

    large_agent = spec.build(
        FakeModelClient([]), tmp_path, repo_root_override=tmp_path, context_window=32_000
    )
    large_agent.run_tool("read_file", {"path": "src/big_module.py"})
    assert not large_agent._last_tool_result_metadata["tool_output_spilled"], (
        "32k 档也落盘的话,Phase A 就失去了跨档位的区分度"
    )


def test_the_small_file_count_pushes_the_recent_window_past_one_full_block():
    """小文件读的次数要能把最近窗口推过 `RECENT_TOOL_TURNS`,否则最近窗口
    这条边界在 Phase A 里一次都不会移动——测的就是"没测到"。
    """
    # Phase A 的工具调用总数 = 大文件读 + patch + 小文件读 + 复读
    total_tool_turns = 2 + sysgate.SMALL_FILE_COUNT + 1
    assert total_tool_turns > RECENT_TOOL_TURNS + RECENT_WINDOW_BLOCK


def test_phase_a_scripted_outputs_match_the_small_file_count():
    """脚本化输出的条数必须和小文件数量对得上,否则要么读不全,要么
    `FakeModelClient` 会在跑到一半时报"fake model ran out of outputs"。
    """
    outputs = sysgate._phase_a_outputs(sysgate.SMALL_FILE_COUNT)
    tool_call_outputs = [o for o in outputs if o.get("tool_calls")]
    final_outputs = [o for o in outputs if not o.get("tool_calls")]
    assert len(tool_call_outputs) == 2 + sysgate.SMALL_FILE_COUNT + 1
    assert len(final_outputs) == 1


def test_window_tiers_above_128k_collapse_to_the_same_budget():
    """`models.budget_breakdown()` 把窗口先 `min(window, 128_000)`,所以
    128k 以上的档位理论上必须给出完全相同的预算——这是"默认只跑 5 档、
    `--all-buckets` 才测后三档"这个设计决策的依据,不是假设。
    """
    capped = [t for t in sysgate.WINDOW_TIERS_ALL if t >= 128_000]
    assert len(capped) == 4
    budgets = {models.budget_breakdown(t)["budget_tokens"] for t in capped}
    assert len(budgets) == 1, "128k 以上的档位不该给出不同的预算，否则默认跳过它们就错了"


def test_default_window_tiers_have_pairwise_distinct_budgets():
    """反过来,默认扫的 5 档必须两两不同——否则"预算互不相同的 5 档"这句话是假的。"""
    budgets = [models.budget_breakdown(t)["budget_tokens"] for t in sysgate.WINDOW_TIERS_DEFAULT]
    assert len(set(budgets)) == len(sysgate.WINDOW_TIERS_DEFAULT)
