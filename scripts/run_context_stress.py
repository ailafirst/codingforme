"""上下文工程压力探针：造出会真正触发落盘 / 窗口推进 / 裁剪的负载。

为什么存在：
固定基准（`benchmarks/coding_tasks.json`）的 11 个 fixture 文件加起来只有 477
token，最大的单文件 115 token，而三个阶段的机制门槛分别是「单条工具结果超
`tool_output_limit`」「一次运行超过 9 次工具调用」「整份 prompt 超 `total_budget`」。
两者相差 100~250 倍，所以在那份基准上跑多少轮，这些机制都是零触发——
「机制生效了」和「机制一次都没跑起来」在工件上长得一模一样。

这个脚本造三块合成工作区，每块只为触发一件事：

    spill_readback   单条工具结果 > 单条上限 → 落盘 + 模型照指针取回（1M 档，
                     绑定生产环境真实用的那个阈值 14,791）
    fanout_window    >9 次工具调用 + 若干超限结果 + 一次必然失败的调用 →
                     最近窗口推进、掉出窗口的落盘结果变指针、错误结果不被清
    budget_pressure  history 撑到超预算 → 真实裁剪 + 被丢弃条目的确定性摘要
    stale_read       读过的文件随后被改 → 那条读记录不再以改动前的全文呈现
                     （唯一带写工具的探针：其余六个白名单里没有写工具，读进来的
                     内容永远不会过期，这个机制在它们身上一次都不会触发）

后两块跑在 16k 档：`tool_output_limit` 和 `total_budget` 都从窗口派生，档位小
只是把同一条代码路径的阈值等比缩小，跑起来便宜得多；顺带补上「只验证过 1M
一个档」这个缺口。

用法（必须真实模型，回放没有意义——这些机制全都由模型发多少调用、调用回来
多大的结果决定）：

    python scripts/run_context_stress.py --probe all \
        --output-json artifacts/context-stress.json
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codingforme import context_manager as cm  # noqa: E402
from codingforme.config import load_project_env, project_root, provider_env  # noqa: E402
from codingforme.eval.harness import BUILTIN_HARNESS_SPECS, get_harness  # noqa: E402
from codingforme.models import OpenAICompatibleModelClient  # noqa: E402

# ---------------------------------------------------------------- 合成工作区


def _write(root, relative, text):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _module_source(name, blocks, header=True):
    """一个体积可控的假模块。每个 block 4 行，行长稳定，token 数好预估。"""
    stem = name.replace(".py", "")
    lines = ["# module: " + name] if header else ["# (header line intentionally missing)"]
    lines.append("")
    for index in range(blocks):
        lines.append("def %s_step_%03d(payload, index=%d):" % (stem, index, index))
        lines.append("    # Deterministic filler so the file has a predictable size.")
        lines.append('    return {"name": "%s", "step": %d, "payload": payload}' % (name, index))
        lines.append("")
    return "\n".join(lines)


def build_spill_workspace(root):
    """一个远超单条上限的日志文件，答案藏在末尾（预览段之外）。"""
    _write(root, "README.md", "# log-triage\n\n`logs/server.log` 是一份 4000 行的服务日志。\n")
    lines = []
    for index in range(1, 4001):
        if index == 3877:
            lines.append("2026-08-30T04:12:07Z AUDIT-TOKEN: run-9f31c7d2-final-marker")
        else:
            lines.append(
                "2026-08-30T04:%02d:%02dZ INFO worker=%02d request=%06d latency_ms=%03d status=200 route=/api/v1/items"
                % (index % 60, (index * 7) % 60, index % 16, index, (index * 13) % 900)
            )
    _write(root, "logs/server.log", "\n".join(lines) + "\n")
    return (
        "logs/server.log 是一份 4000 行的服务日志。文件里恰好有一行以 `AUDIT-TOKEN:` 开头。"
        "请把整个文件读进来，然后把那一行的完整内容原样告诉我。"
    )


def build_fanout_workspace(root):
    """20 个模块：4 个超过 16k 档的单条上限、3 个缺 header，外加一个不存在的文件。"""
    _write(root, "README.md", "# module-audit\n\n`src/` 下的每个 .py 文件第一行都应该是 `# module: <文件名>`。\n")
    missing_header = {"parser.py", "router.py", "cache.py"}
    oversized = {"codec.py", "scheduler.py", "indexer.py", "planner.py"}
    names = [
        "parser.py", "router.py", "cache.py", "codec.py", "scheduler.py",
        "indexer.py", "planner.py", "logger.py", "config.py", "retry.py",
        "queue.py", "metrics.py", "auth.py", "tracer.py", "buffer.py",
        "hasher.py", "pool.py", "clock.py", "serde.py", "limiter.py",
    ]
    for name in names:
        blocks = 60 if name in oversized else 4
        _write(root, "src/" + name, _module_source(name, blocks, header=name not in missing_header))
    return (
        # 措辞必须逼出「整份读」：第一版写的是「读一遍第一行」，模型于是每个文件只取
        # start=1,end=3，4 个超限文件一个都没落盘，探针白跑一次。
        "请先读 src/legacy_codec.py（我不确定它在不在）。然后把 src/ 下每个 .py 文件都"
        "从第 1 行读到第 400 行、完整读一遍，逐个告诉我：文件的第一行是不是 "
        "`# module: <文件名>`，以及这个文件一共有多少行。最后汇总列出所有不符合这条约定的文件名。"
    )


def build_pressure_workspace(root):
    """14 个模块，每个都刚好在单条上限之下——内容全量进 history，把预算撑爆。"""
    _write(root, "README.md", "# line-census\n\n`src/` 下有 14 个模块，需要逐个统计行数。\n")
    names = [
        "alpha.py", "bravo.py", "charlie.py", "delta.py", "echo.py", "foxtrot.py",
        "golf.py", "hotel.py", "india.py", "juliet.py", "kilo.py", "lima.py",
        "mike.py", "november.py",
    ]
    for index, name in enumerate(names):
        # 加上行号之后约 1,150 token，稳稳落在 8k 档的单条上限 1,320 之下：
        # 这个探针要的是「内容全量进 history 然后把预算撑爆」，一旦落盘就测不到裁剪了。
        body = _module_source(name, 13 + (index % 4), header=True)
        # 标记埋在**按字母序第一个**文件里：等模型读完 14 个文件要回答时，
        # 这条内容早已被裁剪压掉。所以这个判据问的是「裁剪有没有把答案弄丢」——
        # 弄丢了模型就得回头重读，那正是要量的代价。
        if name == "alpha.py":
            body = body.replace(
                "# module: alpha.py",
                "# module: alpha.py\n# CENSUS-TOKEN: census-7e21b9-alpha",
                1,
            )
        _write(root, "src/" + name, body)
    return (
        "src/ 下有 14 个 .py 文件。请按字母序把每个文件都完整读一遍（每个都不到 200 行）。"
        "其中恰好有一个文件里有一行以 `# CENSUS-TOKEN:` 开头。全部读完之后告诉我："
        "是哪个文件，以及那个 token 的完整值。"
    )


def build_pointer_workspace(root):
    """一个会刷屏的 `search` 结果 + 10 个小文件。

    为什么不用 `read_file`：`_compressed_history_entries()` 里「重用文件摘要」
    那一支排在 `_summarize_old_tool_item()` 之前，且对每一次执行过的
    `read_file` 都命中——于是落过盘的 `read_file` 结果掉出窗口时被换成
    记忆里的文件摘要，而不是那条指针。`search` 不走那一支。
    """
    _write(root, "README.md", "# log-and-modules\n\n`logs/access.log` 很大，`src/` 下有 10 个小模块。\n")
    lines = []
    for index in range(1, 2401):
        lines.append(
            "2026-08-30T05:%02d:%02dZ GET /api/v1/items?page=%04d latency_ms=%03d status=200"
            % (index % 60, (index * 11) % 60, index, (index * 17) % 900)
        )
    _write(root, "logs/access.log", "\n".join(lines) + "\n")
    names = [
        "alpha.py", "bravo.py", "charlie.py", "delta.py", "echo.py",
        "foxtrot.py", "golf.py", "hotel.py", "india.py", "juliet.py",
    ]
    for name in names:
        body = _module_source(name, 3, header=True)
        # 判据不能用「一共多少行」：文件以空行结尾，13 还是 14 全看谁来数，
        # 模型答哪个都不算错，A/B 于是测的是数数习惯而不是机制。埋一个唯一
        # 字面串，答对答错没有第二种读法。
        if name == "golf.py":
            body += "\n# MODULE-TOKEN: mod-4b17ca-golf\n"
        _write(root, "src/" + name, body)
    return (
        "先用 search 把 logs/access.log 里所有包含 `latency_ms=` 的行找出来（会很多）。"
        "然后逐个读一遍 src/ 下的 10 个 .py 文件：其中恰好有一个文件里有一行以 "
        "`# MODULE-TOKEN:` 开头。告诉我是哪个文件，以及那个 token 的完整值。"
    )


def build_threshold_band_workspace(root):
    """一个大小刚好卡在两个候选阈值之间的文件（约 17,000 token）。

    为什么要有这个探针：`spill_readback` 量不了阈值本身。那份日志有 156,103 token，
    远超任何候选阈值，模型读多宽的区间完全由它自己决定——实测同一道题它一次挑
    `start=1,end=4000`（落盘、5 步答对），一次拆成 20 次窄读（不落盘、耗光步数）。
    这种方差把 33% 的阈值变化整个淹掉了。

    这里把文件做成「一次整份读恰好落在 14,810（除数 8）和 19,747（除数 6）之间」：
    除数 8 时它落盘、模型只拿到预览加指针；除数 6 时它整份进上下文。**唯一的变量
    就是阈值**，模型的读法不再影响结论——提示词明说整份读，而整份读只有一种大小。
    """
    lines = ["# service configuration reference", ""]
    for index in range(1, 661):
        lines.append(
            "setting_%04d: value=%s timeout_ms=%d retries=%d region=%s"
            % (index, "abcdefgh"[index % 8] * 6, (index * 37) % 9000, index % 5, "region-%02d" % (index % 12))
        )
        if index == 600:
            lines.append("BAND-TOKEN: band-5c93ea11-deep-marker")
    _write(root, "config/reference.txt", "\n".join(lines) + "\n")
    _write(root, "README.md", "# config-audit\n\n`config/reference.txt` 是一份配置参考。\n")
    # 663 行 → `read_file` 整份输出约 17,300 token（行号也算进去了），两个候选阈值
    # 各留约 2,500 的余量。改这个函数时必须重新量一次，否则探针会静默失去意义：
    # 掉到 14,810 之下两边都不落盘，涨到 19,747 之上两边都落盘，两种情况都测不出差别。
    return (
        "请把 config/reference.txt 从头到尾完整读一遍（它有 663 行，用 start=1、end=663 一次读完）。"
        "文件里恰好有一行以 `BAND-TOKEN:` 开头，位置很靠后。把那一行的完整内容原样告诉我。"
    )


def build_stale_read_workspace(root):
    """一个「第二处改动的锚点必须包含第一处改动」的文件。

    为什么要这么设计（第一版是**错的**，这段是修它的）：过期读作废要产生后果，
    得让模型给 `patch_file` 写 `old_text` 时**绕不开**第一处已经被改掉的那行。
    第一版把两处默认值放进两个互相独立的函数（`render` / `describe`），于是第二次
    patch 的锚点从来不受第一次影响——机制照常触发（`stale_read_count` 是 1），
    但开和关跑出来的结果一模一样：6 轮、5 次调用、0 次工具错误、两边都答对。
    **触发了不等于起作用了**，而这两件事在计数器上分不出来。

    现在的形状是结构性的，不靠运气：

    - 要改的两行**相邻**，都在 `render()` 里：先改 `mode` 那行，再改 `kind` 那行。
    - `kind = config.get("kind", "legacy")` 这行在 `preview()` 里**一字不差地又出现
      一次**，所以它单独拿出来当 `old_text` 会命中 2 次，被 `patch_file` 的
      「必须恰好出现一次」打回。
    - 它**后面**那行 `return build(mode, kind)` 同样在 `preview()` 里重复，所以往下
      取上下文也消不掉歧义。
    - 于是唯一能消歧的邻居是它**上面**那行——正是第一次 patch 刚改过的那行。

    结果：关掉作废时模型从过期全文里抄出来的 `old_text` 带着 `"legacy"`，命中 0 次，
    白烧一个约 17 秒的往返；开着时它先被告知那条读记录已过期，重读一次再写锚点。
    """
    body = [
        "def build(mode, kind):",
        '    return mode + "/" + kind',
        "",
        "",
        "def render(config):",
        '    mode = config.get("mode", "legacy")',
        '    kind = config.get("kind", "legacy")',
        "    return build(mode, kind)",
        "",
        "",
        "def preview(config):",
        '    mode = "modern"',
        '    kind = config.get("kind", "legacy")',
        "    return build(mode, kind)",
        "",
    ]
    _write(root, "src/render.py", "\n".join(body) + "\n")
    _write(root, "README.md", "# render-service\n\n`src/render.py` 的 `render()` 有两处默认值要改。\n")
    return (
        "src/render.py 的 `render()` 函数里有两处默认值写成了 `legacy`（`mode` 一处、`kind` 一处）。"
        "请把**这两处**都改成 `modern`，**分两次 patch_file 调用完成，一次改一处，先改 mode 再改 kind**。"
        "`preview()` 里的那处不要动。改完之后在答案末尾原样写上一行 `STALE-PROBE-DONE`。"
    )


def build_spill_stale_workspace(root):
    """读一个大到会落盘的文件 → 改它 → 再做十次调用把它挤出最近窗口。

    这是 `stale_pointer`（窗口外的落盘指针标过期）在真实负载上的**唯一**作用对象。
    在这个探针之前它一次都没执行过：七个探针里只有 `stale_read` 带写工具，而那份
    fixture 只有几十行、在任何档位都不会落盘；14 个基准任务的 trace 扫下来
    `spilled_pointer_count` 合计为 0。机制正确、有测试、真实负载上从不执行——
    这个状态在报告里长得和「没问题」一模一样。

    形状要同时满足三件事，缺一个就退化成别的探针：
    **落盘**（8k 兜底档单条上限 1,320，`core.py` 约 2,600 token，整份读必然超）、
    **被写过**（紧接着 patch 同一个文件）、**掉出最近窗口**（后面十个小文件各读一次，
    工具轮数超过 `RECENT_TOOL_TURNS = 6`）。
    """
    # 要改的那行放在文件**开头**：落盘之后回给模型的是「头部预览 + 指针」，
    # 标记在末尾的话预览里根本看不到它，模型只能照指针一段段读回来——第一次
    # 冒烟跑批就是这样，16 步全花在读回落盘文件上，`patch_file` 一次都没发出去，
    # 于是「文件被写过」这个前提不成立，`stale_pointer` 照样触发不了。
    core = ["# core module", "", "MARKER = 'TODO-alpha'", ""]
    for index in range(160):
        core.append("def handler_%03d(payload):" % index)
        core.append('    """处理第 %d 类事件，返回规范化后的载荷。"""' % index)
        core.append("    return {'kind': %d, 'payload': payload}" % index)
        core.append("")
    _write(root, "src/core.py", "\n".join(core) + "\n")

    words = ("alpha", "bravo", "charlie", "delta", "echo",
             "foxtrot", "golf", "hotel", "india", "juliett")
    for index, word in enumerate(words, start=1):
        _write(root, "src/mod_%02d.py" % index, "token: %s\n" % word)
    _write(root, "README.md", "# spill-stale\n\n`src/core.py` 很大，`src/mod_*.py` 各有一个 token。\n")
    return (
        "第一步：完整读一遍 `src/core.py`（**不要**用行号区间，要整份读），"
        "把里面的 `TODO-alpha` 改成 `DONE-alpha`。"
        "第二步：依次读 `src/mod_01.py` 到 `src/mod_10.py` 这十个文件，"
        "把每个文件里 `token:` 后面的那个词按编号顺序拼成一串（用 `-` 连接）。"
        "在答案末尾原样写上一行 `CHAIN=<拼好的那一串>`。"
    )


def build_delegate_survey_workspace(root):
    """一份「读起来很大、结论只有一行」的调查任务——委派这个机制的标准形状。

    为什么需要它：`delegate` 是四层上下文治理里「不让内容进主上下文」那一层的
    **唯一**机制，而它在真实负载上一次都没执行过——六批 live 跑批里调用数恒为 0，
    因为每个基准任务的 `allowed_tools` 都把它裁掉了。现在它是元工具（穿过白名单，
    `tools.META_TOOLS`），所以只要变体打开 `delegate_tool` 就用得上。

    形状：12 个模块，每个约 700~900 token，主 agent 全读一遍必然撑爆 16k 档的
    预算；而结论只有一个文件名。委派把这 12 份内容留在子 agent 的上下文里，
    主上下文只收到一行结论——省下多少由 `delegate_saved_tokens` 记账。
    """
    names = ("parser", "router", "cache", "codec", "sched", "audit",
             "queue", "retry", "trace", "vault", "shard", "probe")
    longest = "codec"
    for name in names:
        # 22 行 ≈ 1,100 token < 16k 档的单条上限 1,477：单个模块**不落盘**。
        # 冒烟跑批里每个模块 2,005 token，12 次读全落了盘，测到的是落盘不是委派。
        # 而 12 个加起来 14,000 多 token 仍然远超预算 12,087，委派照样划算。
        rows = 22 if name != longest else 40
        body = ["# module: %s" % name, ""]
        for index in range(rows):
            body.append("def %s_step_%02d(payload, options=None):" % (name, index))
            body.append('    """%s 的第 %d 步：规范化载荷并返回。"""' % (name, index))
            body.append("    return {'module': '%s', 'step': %d, 'payload': payload}" % (name, index))
            body.append("")
        _write(root, "src/%s.py" % name, "\n".join(body) + "\n")
    _write(root, "README.md", "# survey\n\n`src/` 下有 12 个模块，需要知道哪个最长。\n")
    return (
        "`src/` 目录下有 12 个 Python 模块。请查出其中**行数最多**的是哪一个，"
        "并在答案末尾原样写上一行 `LONGEST=<模块的文件名>`（例如 `LONGEST=foo.py`）。"
    )


PROBES = {
    "stale_read": {
        # 1M 档：预算充裕，history 不会被压力机制动到，读记录稳稳留在最近窗口里——
        # 而那正是过期读作废唯一生效的地方。换成小档位会让别的机制一起动，A/B 测的
        # 就不是这一个变量了。
        "window": 1_000_000,
        "max_steps": 10,
        "tools": ("read_file", "patch_file", "list_files"),
        "build": build_stale_read_workspace,
        # 判据串是刻意埋的标记,不是「还剩几处」那个数字。第一版写的是 `("0",)`,
        # 两处都错:正确答案是 2(另外两处 `legacy` 是分支判断和返回值前缀,不是默认值),
        # 而且 `"0"` 会被答案里 `config.get("id", 0)` 这种无关文本蒙中——两臂都
        # `expect_hits: ["0"]`,等于这条判据什么都没查。
        "expect": ("STALE-PROBE-DONE",),
        "goal": "读过的文件随后被改 → 第二处改动的锚点绕不开第一处，过期全文必然写错",
    },
    "spill_stale": {
        # 8k 兜底档：单条上限 1,320，`src/core.py` 约 2,600 token，整份读必然落盘。
        # 后面十次 read_file 把它推出最近窗口（`RECENT_TOOL_TURNS = 6`），而它在这
        # 之前已经被 patch 过 → 窗口外的那条指针要被标成过期。
        "window": 8_000,
        # 20 步不是 16：一次落盘 + 一次 patch + 十次小读已经是 12 步，冒烟跑批里
        # 模型还会先 list_files 探一次路，16 步在中途就被截停。
        "max_steps": 20,
        "tools": ("read_file", "patch_file", "list_files"),
        "build": build_spill_stale_workspace,
        "expect": ("CHAIN=alpha-bravo-charlie-delta-echo-foxtrot-golf-hotel-india-juliett",),
        "goal": "落过盘的读记录先被写过、再掉出最近窗口 → 指针照留但要标明盘上那份已过期",
    },
    "threshold_band": {
        # 1M 档。文件约 17,000 token：除数 8（上限 14,810）时落盘，除数 6（19,747）时不落盘。
        "window": 1_000_000,
        "max_steps": 8,
        "tools": ("read_file", "list_files"),
        "build": build_threshold_band_workspace,
        "expect": ("band-5c93ea11-deep-marker",),
        "goal": "整份读的大小卡在两个候选阈值之间 → 只有单条上限这一个变量在动",
    },
    "spill_readback": {
        "window": 1_000_000,
        "max_steps": 8,
        "tools": ("read_file", "list_files"),
        "build": build_spill_workspace,
        "expect": ("run-9f31c7d2-final-marker",),
        "goal": "单条工具结果超过 tool_output_limit → 落盘留指针，模型照指针取回",
    },
    "fanout_window": {
        "window": 16_000,
        "max_steps": 28,
        "tools": ("read_file", "list_files"),
        "build": build_fanout_workspace,
        "expect": ("parser.py", "router.py", "cache.py"),
        "goal": "工具调用数超过最近窗口 → 窗口按块推进、落盘结果变指针、错误结果不被清",
    },
    "pointer_placeholder": {
        # 8k 兜底档：`search` 的结果必然超过单条上限 1,320 而落盘，
        # 后面 10 次 read_file 把它推出最近窗口 → spilled_pointer_count > 0。
        "window": 8_000,
        "max_steps": 16,
        "tools": ("read_file", "list_files", "search"),
        "build": build_pointer_workspace,
        "expect": ("mod-4b17ca-golf",),
        "goal": "落过盘的结果掉出最近窗口 → 占位换成那条可恢复的指针",
    },
    "forced_spill_1m": {
        # 1M 档(生产环境真实用的那个档位)下**模型无法回避**的落盘。
        #
        # 为什么单独有这个探针：`spill_readback` 也在 1M 档，但那里模型可以自己
        # 挑 `read_file` 的行号区间，实测它就是这么做的——一次 15 任务的 live 跑批
        # `spilled_calls` 是 0，落盘一次都没触发，于是「1M 档能不能落盘」在真实
        # 负载上从来没被验证过。`search` 没有区间参数：命中多少行就回多少行，
        # 2400 行日志 × 约 45 token ≈ 10 万 token，必然超过 1M 档的单条上限
        # 14,810。这个探针问的是：那一刻落盘会不会正常发生、模型能不能接着干活。
        "window": 1_000_000,
        "max_steps": 16,
        "tools": ("read_file", "list_files", "search"),
        "build": build_pointer_workspace,
        "expect": ("mod-4b17ca-golf",),
        "goal": "1M 档下模型无法回避的超限结果 → 落盘在生产档位上真的会发生",
    },
    "delegate_survey": {
        # 16k 档：12 个模块全读进主上下文必然撑爆预算 27,287 → 委派才是划算的做法。
        # **必须配 `--harness delegate_tool` 跑**，否则注册表里根本没有这个工具，
        # 那一臂就是「模型自己全读一遍」的对照组。
        "window": 16_000,
        "max_steps": 26,
        "tools": ("read_file", "list_files"),
        "build": build_delegate_survey_workspace,
        "expect": ("LONGEST=codec.py",),
        "goal": "读起来很大、结论只有一行 → 整段调查在子 agent 的上下文里跑完",
    },
    "budget_pressure": {
        # 刻意用最保守的 8k 兜底档：预算 4,486、单条上限 1,320，6 个文件的内容就
        # 必然撑爆预算。顺带补上「只在 1M 档验证过」这个缺口的另一端。
        "window": 8_000,
        "max_steps": 24,
        "tools": ("read_file", "list_files"),
        "build": build_pressure_workspace,
        "expect": ("census-7e21b9-alpha",),
        "goal": "history 撑到超预算 → 真实裁剪 + 被丢弃条目的确定性摘要",
    },
}


# ---------------------------------------------------------------- 跑一个探针


def _model_client(timeout):
    load_project_env(project_root())
    api_key = provider_env("CODINGFORME_OPENAI_API_KEY", ("OPENAI_API_KEY",))
    if not api_key:
        raise SystemExit("stress probes need a real model: set CODINGFORME_OPENAI_API_KEY in .env")
    return OpenAICompatibleModelClient(
        model=provider_env("CODINGFORME_OPENAI_MODEL", ("OPENAI_MODEL",), "gpt-5.4"),
        base_url=provider_env("CODINGFORME_OPENAI_API_BASE", ("OPENAI_API_BASE",), "https://api.openai.com/v1"),
        api_key=api_key,
        temperature=0.0,
        timeout=timeout,
    )


def _read_trace(run_dir):
    path = Path(run_dir) / "trace.jsonl"
    events = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def _meta(event):
    return event.get("prompt_metadata") or {}


def _summarize(events):
    """只统计三个阶段各自的验收字段。零值照写——省略会被读成「没问题」。"""
    prompts = [e for e in events if e.get("event") == "prompt_built"]
    tools = [e for e in events if e.get("event") == "tool_executed"]
    compactions = [e for e in events if e.get("event") == "context_compacted"]
    delegates = [e for e in tools if e.get("name") == "delegate"]
    history = [_meta(e).get("history") or {} for e in prompts]

    spills = [e for e in tools if e.get("tool_output_spilled")]
    spill_paths = [e.get("tool_output_spill_path") for e in spills if e.get("tool_output_spill_path")]
    readbacks = [
        e for e in tools
        if e.get("name") == "read_file"
        and any(str(p) in json.dumps(e.get("args") or {}, ensure_ascii=False) for p in spill_paths)
    ]

    windows = {}
    for item in history:
        key = int(item.get("recent_tool_window") or 0)
        windows[key] = windows.get(key, 0) + 1

    reductions = []
    for event in prompts:
        for item in _meta(event).get("budget_reductions") or []:
            reductions.append(item)

    def _max(getter):
        return max([getter(item) for item in history] or [0])

    return {
        "turns": len(prompts),
        "tool_calls": len(tools),
        "tool_names": sorted({str(e.get("name")) for e in tools}),
        "tool_errors": sum(1 for e in tools if e.get("tool_status") == "error"),
        # 阶段二 / 三：落盘
        # 落盘失败单独数：它在 trace 里退回普通截断，和「结果本来就没超上限」
        # 长得一样,不单独数就查不出来。
        "spill_failed": sum(1 for e in tools if e.get("tool_output_spill_failed")),
        "spill_failed_detail": [
            {"name": e.get("name"), "full_tokens": e.get("tool_output_full_tokens"),
             "error": e.get("tool_output_spill_error")}
            for e in tools if e.get("tool_output_spill_failed")
        ],
        "spilled": len(spills),
        "spill_paths": spill_paths,
        "spill_full_tokens": [e.get("tool_output_full_tokens") for e in spills],
        "spill_kept_tokens": [e.get("tool_output_kept_tokens") for e in spills],
        "spill_readbacks": len(readbacks),
        # 受限委派的上下文记账。零值也照写：`delegate` 此前六批 live 跑批里调用数
        # 恒为 0，而「模型不用它」和「这个探针没测到它」在省略掉这几项时长得一样。
        "delegate_calls": len(delegates),
        "delegate_saved_tokens": sum(int(e.get("delegate_saved_tokens") or 0) for e in delegates),
        "delegate_child_transcript_tokens": sum(
            int(e.get("delegate_child_transcript_tokens") or 0) for e in delegates
        ),
        "delegate_result_tokens": sum(int(e.get("delegate_result_tokens") or 0) for e in delegates),
        # 子 agent 撞步数上限被截停的次数。实测这是最常见的失败形态（一次跑批 5 次
        # 委派有 4 次如此），而它回给父的字符串和跑完的那次形状一样，不单独数就看不见。
        "delegate_child_incomplete": sum(
            1 for e in delegates
            if str(e.get("delegate_child_stop_reason") or "") not in ("", "final_answer_returned")
        ),
        # 阶段三：窗口与占位
        "recent_tool_window": windows,
        "spilled_pointer_count_max": _max(lambda i: int(i.get("spilled_pointer_count") or 0)),
        "preserved_error_count_max": _max(lambda i: int(i.get("preserved_error_count") or 0)),
        # 供给侧新鲜度：这一轮有几条 read_file 结果因为文件后来被写过而没有全文呈现。
        "stale_read_count_max": _max(lambda i: int(i.get("stale_read_count") or 0)),
        # 窗口**外**那一半：指针还在，但它指向的落盘快照已经不是文件现状。和上面
        # 那个分开数——一个是模型手里的全文过期了，一个是它照指针取回来的会过期。
        "stale_pointer_count_max": _max(lambda i: int(i.get("stale_pointer_count") or 0)),
        # 可逆折叠（L4）：这一轮有几条窗口外的条目被压成 10 token 的残句。它和
        # `omitted_digest_present` 是一对——一个是「压扁了还在」，一个是「整条没了」，
        # 不分开数的话两者在工件上长得一模一样。
        "squeezed_entry_max": _max(lambda i: int(i.get("squeezed_entry_count") or 0)),
        "older_entries_max": _max(lambda i: int(i.get("older_entries_count") or 0)),
        "summarized_tool_max": _max(lambda i: int(i.get("summarized_tool_count") or 0)),
        "reused_file_summary_max": _max(lambda i: int(i.get("reused_file_summary_count") or 0)),
        # 阶段一：预算
        "prompt_tokens_max": max([int(_meta(e).get("prompt_tokens") or 0) for e in prompts] or [0]),
        "budget_tokens": next((int(_meta(e).get("prompt_budget_tokens") or 0) for e in prompts), 0),
        "prompt_over_budget": sum(1 for e in prompts if _meta(e).get("prompt_over_budget")),
        "budget_reductions": len(reductions),
        # 会话摘要真的换掉过内容的次数，按触发方式分开。零值也照写：「一次都没压」
        # 和「这块没问题」在省略掉这一项时长得一模一样。
        "compactions_auto": sum(1 for e in compactions if e.get("trigger") == "auto"),
        "compactions_manual": sum(1 for e in compactions if e.get("trigger") == "manual"),
        "compaction_saved_tokens": sum(int(e.get("saved_tokens") or 0) for e in compactions),
        # `clear_at_least` 真正说了算的次数：按它裁的量比 overflow 还多。分级压缩
        # 开着时这个数恒为 0（15% 的触发-目标间距压过 1/10），非 0 说明分级被关了
        # 或者两个比例被拉近过。
        "clear_at_least_bound": sum(
            1 for item in reductions
            if int(item.get("target_tokens") or 0) > int(item.get("overflow_tokens") or 0)
        ),
        "budget_floor_exhausted": sum(1 for e in prompts if _meta(e).get("budget_floor_exhausted")),
        "omitted_digest_present": sum(
            1 for e in prompts if (_meta(e).get("message_layout") or {}).get("omitted_digest_present")
        ),
        "context_window": next((int(_meta(e).get("context_window_tokens") or 0) for e in prompts), 0),
        "context_window_source": next((str(_meta(e).get("context_window_source") or "") for e in prompts), ""),
        "cached_tokens": sum(int((e.get("completion_metadata") or {}).get("cached_tokens") or 0) for e in events),
        "input_tokens": sum(int((e.get("completion_metadata") or {}).get("input_tokens") or 0) for e in events),
    }


def run_probe(name, *, timeout, keep, variant="full", max_steps=0):
    config = PROBES[name]
    root = Path(tempfile.mkdtemp(prefix="cfm-stress-%s-" % name))
    started = time.time()
    try:
        prompt = config["build"](root)
        # 变体走 `BUILTIN_HARNESS_SPECS`，不在这里手搓 feature_flags——消融的
        # 唯一入口是 HarnessSpec，这样 A/B 两边的指纹才落在同一套口径上。
        base = get_harness(variant)
        spec = base.derive(
            name="stress-%s-%s" % (name, variant),
            description=config["goal"],
            max_steps=int(max_steps) or config["max_steps"],
            tools_allowlist=config["tools"],
        )
        agent = spec.build(
            _model_client(timeout),
            root,
            repo_root_override=root,
            context_window=config["window"],
        )
        answer = agent.ask(prompt)
        events = _read_trace(agent.current_run_dir)
        summary = _summarize(events)
        summary.update({
            "probe": name,
            "variant": variant,
            "harness_fingerprint": spec.fingerprint(),
            "max_steps": spec.max_steps,
            "goal": config["goal"],
            "declared_window": config["window"],
            "tool_output_limit": cm.tool_output_limit(summary["budget_tokens"] or 1),
            "workspace": str(root),
            "elapsed_s": round(time.time() - started, 1),
            # A/B 需要一个硬的结果指标，不能只看 token 数省了多少：机制省了上下文
            # 但把答案弄丢了，那是负收益。`expect` 是这道题的正确答案里必然出现的
            # 字面串（藏在日志里的那个标记、缺 header 的三个文件名、总行数），
            # 全部命中才算答对。判据写死在探针定义里，不用模型当裁判。
            "expect": list(config.get("expect", ())),
            "expect_hits": [item for item in config.get("expect", ()) if item in str(answer)],
            "answered": all(item in str(answer) for item in config.get("expect", ())),
            "answer_head": str(answer)[:400],
            "stop_reason": next(
                (e.get("stop_reason") for e in reversed(events) if e.get("event") == "run_finished"), ""
            ),
        })
        return summary
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", default="all", choices=["all"] + sorted(PROBES))
    parser.add_argument("--output-json", default="")
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument(
        "--harness",
        default="full",
        choices=sorted(BUILTIN_HARNESS_SPECS),
        help="消融变体；A/B 就是同一个探针跑两次、只换这个名字",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="覆盖探针自带的步数上限（0=不覆盖）")
    args = parser.parse_args(argv)

    # Windows 控制台默认 GBK，而探针会把模型的答案原样打出来——答案里出现一个
    # emoji 就 UnicodeEncodeError，整个跑批在**最后一步**炸掉、工件一个字没写。
    # 已经踩过一次：一次跑满 44 步的 fanout 因为答案里有个 ✅ 全白跑了。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    names = sorted(PROBES) if args.probe == "all" else [args.probe]
    results = []

    def _flush(payload_results):
        """每跑完一个探针就落一次盘：跑批越长，中途挂掉丢掉全部结果的代价越大。"""
        if not args.output_json:
            return
        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "artifact_type": "context-stress-probe",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "harness": args.harness,
            "probes": payload_results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in names:
        print("=== probe: %s (%s) ===" % (name, args.harness), flush=True)
        result = run_probe(
            name,
            timeout=args.request_timeout,
            keep=args.keep_workspace,
            variant=args.harness,
            max_steps=args.max_steps,
        )
        results.append(result)
        _flush(results)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    payload = {
        "artifact_type": "context-stress-probe",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "harness": args.harness,
        "probes": results,
    }
    if args.output_json:
        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote %s" % target, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
