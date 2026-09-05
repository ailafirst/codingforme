"""压力探针的**形状**不变量。

这些探针不是库代码，正常不该有测试。有这一份是因为踩过一个具体的坑：`stale_read`
探针的 fixture 把两处要改的默认值放进了两个互相独立的函数，于是「第二次 patch 的
锚点会不会受第一次改动影响」这件事从来没成立过——机制照常触发（计数器是 1），
开和关跑出来的结果却一模一样。**探针失去作用对象时，它长得和「机制没问题」一样。**

所以这里锁的不是代码行为，是 fixture 的那几条结构性质：它们一旦被「整理」掉，
对应的 live A/B 就静默退化成一个什么都不测的跑批。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_context_stress as stress  # noqa: E402

from codingforme.context_manager import RECENT_TOOL_TURNS, tool_output_limit  # noqa: E402
from codingforme.models import count_tokens  # noqa: E402


def test_the_stale_read_fixture_forces_the_second_anchor_through_the_first_edit(tmp_path):
    """第二处改动的 `old_text` 必须绕不开第一处——这是这个探针唯一的作用对象。

    `patch_file` 要求 `old_text` 恰好命中一次。所以只要
    (a) 目标行本身在文件里出现两次、(b) 它下面那行也出现两次，
    唯一能消歧的邻居就只剩它**上面**那行，而那正是第一次 patch 刚改过的。
    """
    stress.PROBES["stale_read"]["build"](tmp_path)
    lines = (tmp_path / "src" / "render.py").read_text(encoding="utf-8").splitlines()

    target = '    kind = config.get("kind", "legacy")'
    first_edit = '    mode = config.get("mode", "legacy")'
    assert lines.count(target) == 2, "目标行只出现一次的话，单行锚点就够用了"
    index = lines.index(target)
    assert lines[index - 1] == first_edit, "目标行上面那行必须就是第一次要改的那行"
    assert lines.count(lines[index + 1]) == 2, "下面那行也得重复，否则往下取上下文就能绕开"
    # 第一次要改的那行本身必须唯一，否则模型连第一处都定位不到。
    assert lines.count(first_edit) == 1


def test_the_spill_stale_fixture_actually_spills_at_its_declared_tier(tmp_path):
    """整份读必须真的超过那一档的单条上限，否则不会落盘，也就没有指针可以标过期。

    这条是算出来的，不是跑出来的：文件 token 数与 `tool_output_limit()` 直接比。
    """
    config = stress.PROBES["spill_stale"]
    config["build"](tmp_path)
    body = (tmp_path / "src" / "core.py").read_text(encoding="utf-8")

    # 8k 档的预算约 4.4k，单条上限 1,320。留一倍余量，别卡在边界上。
    limit = tool_output_limit(4428)
    assert count_tokens(body) > 2 * limit, (count_tokens(body), limit)
    assert config["window"] == 8_000


def test_the_spill_stale_fixture_has_enough_padding_to_push_the_read_out_of_window(tmp_path):
    """后面那些小文件的数量要够把落盘的那条读记录挤出最近窗口。

    不够的话它一直是窗口**内**的形态，测到的是 `stale_read` 而不是 `stale_pointer`——
    两个机制，同一个开关，但呈现完全不同。
    """
    stress.PROBES["spill_stale"]["build"](tmp_path)
    padding = sorted((tmp_path / "src").glob("mod_*.py"))

    # 1 次大读 + 1 次 patch 之后，还要有超过 RECENT_TOOL_TURNS 个工具轮把它顶出去。
    assert len(padding) > RECENT_TOOL_TURNS, (len(padding), RECENT_TOOL_TURNS)
    assert stress.PROBES["spill_stale"]["max_steps"] >= len(padding) + 2


def test_the_delegate_probe_is_too_big_to_read_in_the_parent_context(tmp_path):
    """委派要划算，主 agent 全读一遍就得放不下——否则模型自己读完也一样，
    这个探针测不出「不让内容进主上下文」这一层的价值。
    """
    config = stress.PROBES["delegate_survey"]
    config["build"](tmp_path)
    modules = sorted((tmp_path / "src").glob("*.py"))
    total = sum(count_tokens(path.read_text(encoding="utf-8")) for path in modules)

    assert len(modules) == 12
    # 16k 档的预算是 12,087（`window_16k` 变体的描述里写着这个数）。
    assert total > 12_087, total
    # 正确答案必须是唯一的最长模块。
    longest = max(modules, key=lambda path: len(path.read_text(encoding="utf-8").splitlines()))
    assert config["expect"] == ("LONGEST=%s" % longest.name,)


def test_every_probe_declares_a_marker_the_answer_cannot_stumble_into(tmp_path):
    """判据串要是刻意埋的标记，不能是「还剩几处」那种数字。

    踩过的坑：`stale_read` 第一版的 `expect` 是 `("0",)`，被答案里
    `config.get("id", 0)` 这种无关文本蒙中，两臂都「答对」，这条判据什么都没查。
    """
    for name, config in stress.PROBES.items():
        expects = config.get("expect", ())
        assert expects, name
        for marker in expects:
            assert len(str(marker)) >= 4, (name, marker)
            assert not str(marker).isdigit(), (name, marker)
