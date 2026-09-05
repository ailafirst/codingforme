"""Prompt 组装与上下文预算控制。

这个模块负责决定：每一轮到底把多少 prefix、memory、相关笔记、历史
以及当前用户请求送进模型。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import models
from .models import count_tokens


# 预算的单位是 **token**，不是字符。
#
# 从前它是 12000 个字符。实测 318 轮真实请求，`prompt_chars / input_tokens`
# 的比值中位 2.06、最小 0.83、最大 2.51——相差 3 倍，因为中文一个字往往就是一个
# token，而英文和代码大约 3~4 个字符才一个。用字符表示一个 token 上限，误差本身
# 就吃掉了任何精确设定的努力。
#
# **各段额度和下限同样是 token。** 系统里不存在第二种单位：预算、下限、每段额度、
# 所有比较、所有截断一律按 token 算。曾经的混合制（闸门按 token、各段按字符）逼出过
# 一段没有原理可言的折算代码——把 token 溢出量当字符去减各段额度——那段已经删掉。
#
# **各段没有固定额度。** 默认上限就是 `total_budget` 本身，只有整份 prompt 真的
# 放不下时，才按 `DEFAULT_REDUCTION_ORDER` 逐段往下压到 `SECTION_FLOORS`。
#
# 从前这里有一组绝对常量（prefix 1450 / memory 520 / relevant_memory 900 /
# history 2500）。它们和 `total_budget` 完全脱钩，实测两个方向都错：
#
#   1M 档（total_budget 118,335）   四段合计 5,370 = 预算的 4.5%。裁剪循环永远
#                                   触发不了，但各段照样被卡死——本仓库 prefix
#                                   原始 3,696 token 被裁到 1,450，每轮白丢 2,246
#                                   （61%），而此时预算还空着 11 万。
#   8k 兜底档（total_budget 4,335） 四段合计 5,370 = 预算的 123.9%。额度之和比总
#                                   预算还大，于是每一轮都在跑裁剪循环。
#
# 约束改到**来源入口**去加，这一点和 Claude Code 一致：它组装时没有分段额度，每块
# 整个进上下文，限制加在读进来的那一刻（auto memory 取前 200 行或 25KB、工具结果
# 25,000 token）。我们对应的入口上限是 `workspace.MAX_SNAPSHOT_TOKENS`、
# `workspace.MAX_TOOL_OUTPUT`、`memory.NOTE_TOKENS` / `TASK_SUMMARY_TOKENS`。
#
# 那组常量是从更早的字符额度折算来的（换单位时为了容量不变）。折算比值留在这里，
# 它仍是「同一段文本按字符数和按 token 数差多少」的唯一实测记录，样本取自本仓库
# 源码 + 一次真实 `ask()` 的转录（`mimo-v2.5` 分词器，和 `count_tokens()` 同一个）：
#
#   prefix（规则 + 工具清单 + 仓库快照）  2.48  快照里全是路径，token 很密
#   memory（working：英文摘要 + 路径）    3.06
#   relevant_memory（模型写的中文笔记）   1.31  中文一个字往往就是一个 token
#   history（中文请求 + 工具输出混合）    2.07
#   工具输出（源码）                      3.02
#   工具 schema（JSON）                   3.95
#   整段 prompt                           2.63

# 只在 `ContextManager(agent)` 不传预算时兜底。真实路径的预算一律由
# `models.budget_breakdown()` 从上下文窗口派生、经 `runtime.set_context_window()`
# 传进来，走不到这个值。
FALLBACK_TOTAL_BUDGET = 5600
# 裁剪的下限：压到这里就不再往下压。**没有 prefix 这一项**，而且不是漏写——
# 给它一个下限等于承认它可以被压到某个值，而它压根不参与裁剪（见 PROTECTED_SECTIONS）。
SECTION_FLOORS = {
    "memory": 130,
    "relevant_memory": 230,
    "history": 720,
}
# 当 prompt 超预算时，会优先压缩这些 section。
#
# **`prefix` 不在这个顺序里，这是刻意的。** 它曾经排在最后当兜底项，理由是
# "最后手段"，但实测它每一次都被用到了：632 轮真实跑批里有 67 轮（10.6%）超预算
# 触发裁剪，而这 67 轮**四个 section 全被裁到底、包括 prefix**——裁剪循环每次都
# 走完了整条顺序。
#
# 代价不对等：prefix 就是前缀缓存认的那段公共前缀（`system` 消息 + 第一条 `user`
# 消息的内容）。把命中率从 32.7% 提到 77.6% 靠的就是让它跨请求逐字节相同，裁它
# 一次那轮必然落空，而省下来的只有几百个 token。所以现在的取舍是：**宁可让 prompt
# 超出这个内部预算，也不动 prefix**。
#
# 超出去是安全的：`total_budget` 是我们自己的软预算，不是后端的硬上限，两者之间
# 隔着安全边际（见 `docs/architecture/context-budget-sizing.md`）。超了会如实记进
# `prompt_over_budget` 和 `budget_floor_exhausted`，不是静默行为。
DEFAULT_REDUCTION_ORDER = ("relevant_memory", "history", "memory")
# 永不参与裁剪的 section。放在这里而不是只靠 reduction_order 少写一项，是为了让
# 自定义 reduction_order 的调用方也挡得住——把 prefix 传进去不会生效。
#
# 「不参与裁剪」现在是字面意思：prefix **完全没有额度**，`_render_sections()` 原样
# 渲染它，调用方显式传进来的 prefix 额度也会被 `__init__` 丢掉。从前它只是不参与
# 超预算时的**进一步**压缩，而基础额度 1450 每轮照裁不误——实测本仓库因此每轮丢掉
# prefix 的 61%，且因为 `_tail_clip` 保留开头，丢掉的全是仓库快照那一段。
PROTECTED_SECTIONS = ("prefix",)
SECTION_ORDER = ("prefix", "memory", "relevant_memory", "history", "current_request")
CURRENT_REQUEST_SECTION = "current_request"
RELEVANT_MEMORY_LIMIT = 3
# prefix 段内部的分界线：它之前是规则和工具清单（跨任务恒定），从这一行起是仓库
# 快照和 resume checkpoint（每个任务都不同）。`_assemble_messages()` 按它切开，
# 把恒定的那半留在 system 里，好让前缀缓存吃得到工具 schema 那一大块。
# 值必须和 runtime.PROMPT_TEMPLATE 里 __WORKSPACE_TEXT__ 展开后的首行对上，
# 也就是 workspace.WorkspaceContext.text() 的第一行。
PREFIX_WORKSPACE_MARKER = "\nWorkspace:\n"
# prefix 段末尾还可能挂着 resume checkpoint（`render_checkpoint_text()` 的产物）。
# 它和仓库快照不是一类东西：快照一次运行内恒定，checkpoint **每执行一次工具就重
# 渲染一次**。所以它既不能留在 system，也不该跟快照一起放在第一条 user——那等于
# 每轮都在最靠前的位置制造一处变化。它跟 working memory 是同一类「当前任务状态」，
# 一起摆在最后。
PREFIX_CHECKPOINT_MARKER = "\nTask checkpoint:\n"
# 摆位方案的名字。进 trace 的 prompt_metadata，好让事后分析能区分「这份数据是
# 哪种摆位跑出来的」——缓存命中率跨跑批对比时，不知道摆位就没法解释差异。
MESSAGE_LAYOUT_NAME = "stable-system-v3"


def _tail_clip(text, limit, model=None):
    """把文本裁到 `limit` 个 **token** 以内，保留开头。

    单位是 token，不是字符——整个上下文子系统只有这一种单位。混合制（闸门按
    token、各段额度按字符）曾经存在过，它逼出一段没有原理可言的折算代码：
    把 token 溢出量当字符去减各段额度。现在预算、下限、各段额度、比较、截断
    全部同单位，那段折算随之删掉。
    """
    return models.clip_tokens(text, limit, model, marker="..." if limit > 3 else "")


RECENT_TOOL_TURNS = 6

# 单个工具结果能占多少 token。**从 `total_budget` 派生，不是绝对常量。**
#
# 为什么改：`workspace.MAX_TOOL_OUTPUT = 1320` 和 history 渲染里的 `line_limit = 430`
# 都是字符时代折算来的绝对值（4000 ÷ 3.02、900 ÷ 2.07），与 `total_budget` 完全脱钩
# ——这正是阶段二从各段配额里删掉的那个病，只是活在下一层。实测证据：构造 10 轮
# read_file 历史，只改 `total_budget`（4,335 → 118,335，×27），发出去的 prompt 一个
# token 都没变（3,490 → 3,490），10 条 tool 消息也逐条相同（`[14,14,14,14,430×6]`）。
# 一个 5,599 token 的文件到模型眼前只剩 430 token，**只有 7.7%**，而预算空着 114,562。
#
# 三个数的来历：
# - 除数 8 —— 单个结果最多占预算的 1/8。塞满 8 个正好触发全局裁剪循环，而全局裁剪
#   本来就是为「装不下」准备的；再大就会让一次读文件独占预算。
#   **调大过一次,数据说不该调,已改回来。** 造了个 663 行、整份读恰好 17,295 token
#   的文件（卡在 div=8 的 14,810 和 div=6 的 19,747 之间，提示词明说一次读完，
#   所以阈值是唯一的变量）：两边都答对、都是 2 次调用 3 个往返，而 div=6 多烧 6.1%
#   的输入 token（48,319 vs 45,521）。落盘那条路径**没有多花一个往返**——预览加指针
#   就够模型把题做完，既然省不下往返，把 17,295 token 全塞进上下文就是纯成本。
#   受影响的面也小：本仓库 61 个 .py 里，8→6 只让 2 个（metrics.py 19,621、
#   models.py 15,266）从落盘变成整份进来，runtime.py 和 test_codingforme.py 两边都落盘。
#   探针是 `scripts/run_context_stress.py --probe threshold_band`，要重开这个决定就重跑它。
#   顺带：div=4 在 128k/1M 档正好撞上下面那个 25,000 的上限，等于把「从预算派生」
#   这个性质在最常用的档位上作废，直接出局。
# - 上限 25,000 —— 与 Claude Code 给工具结果的上限一致（200k 窗口的 12.5%）。
# - 下限 1320 —— 旧的绝对值。保留它是为了让 8k 兜底档行为完全不变
#   （4,335 // 8 = 541 < 1320），改动只在大窗口档位上生效。
TOOL_OUTPUT_BUDGET_DIVISOR = 8
# 与 `workspace.MAX_TOOL_OUTPUT` 同值。不 import 过去是为了不让 context_manager
# 依赖 workspace；两处都改动时由
# `tests/test_context_manager.py::test_the_tool_output_floor_matches_the_workspace_constant` 锁住。
TOOL_OUTPUT_MIN_TOKENS = 1320
TOOL_OUTPUT_MAX_TOKENS = 25000


def tool_output_limit(total_budget):
    """单个工具结果的上限（token），从总预算派生。"""
    derived = int(total_budget) // TOOL_OUTPUT_BUDGET_DIVISOR
    return max(TOOL_OUTPUT_MIN_TOKENS, min(TOOL_OUTPUT_MAX_TOKENS, derived))


# 一次裁剪至少要裁掉这么多：`total_budget // CLEAR_AT_LEAST_DIVISOR`。
#
# 只裁「刚好够」的量会让下一轮几乎必然再裁一次，而每裁一次 history 就作废它之后
# 的全部前缀缓存（命中率 32.7% → 77.6% 是花一整轮 A/B 换来的）。Anthropic 的
# context management API 为同一件事留了 `clear_at_least` 这个参数，官方说明就是
# 「ensures cache invalidation worthwhile」。10 = 一次腾出预算的 10%。
CLEAR_AT_LEAST_DIVISOR = 10

# --- 分级压缩(阶段一)---------------------------------------------------------
#
# 从前只有一根线:`prompt_tokens > total_budget` 才动手,一动就走完整条
# `DEFAULT_REDUCTION_ORDER`。合成压力探针(`long_dialogue_pressure`,24 轮纯对话)
# 把这条线的两个毛病都照出来了:
#
# 一、**动手太晚**。压力一旦持续,prompt 就被死死顶在 100.0% 上(实测第 17~24 轮
#    每一轮都是 4,816~4,823 / 4,823),没有任何余量,再来一条稍长的用户消息就溢出。
# 二、**每轮重算**。`build_all()` 每轮都从 `budgets = {section: total_budget}` 重新
#    开始,上一轮压到 3,228 的额度下一轮又回到 4,823,于是每轮都要重裁一遍——12 个
#    触发轮里丢弃条目数变了 11 次(3→8→15→17→23→25→28→31→33→35→39),history 段的
#    开头每轮都在变,**它之后的前缀缓存每轮作废一次**。`CLEAR_AT_LEAST_DIVISOR`
#    防不住这个:第 17 轮一次腾出 1,553 token(预算的 32%,远超 1/10),下一轮照样重裁。
#
# 对应 Claude Code 的两件事:按占用率分级触发(他们的 autocompact 在 83.5%~87%,不是
# 100%),以及 L2 那个「上一级刚释放过就别跟着动」的抑制器。这里的落地形态是:
#
#   触发点   占用率 > TRIGGER_RATIO 就开始压,不等到超预算
#   目标点   一压就压到 TARGET_RATIO 以下,留出余量(这才是 clear_at_least 的本意)
#
# (原计划还有第三件「迟滞」,做出来实测无效已撤掉,见下面那段注释。)
# 两个数都是比例,不是从 Claude Code 抄来的绝对常量(他们那三个百分比有
# 83.5 / 87 / ~90 三个互相冲突的版本,全是第三方逆向)。
COMPRESSION_TRIGGER_RATIO = 0.85
COMPRESSION_TARGET_RATIO = 0.70
# **迟滞这一半试过了,做不到,原因是结构性的——别再实现一遍。** 曾经加过一个跨轮
# 保持、按块前进的 history 丢弃边界(`_history_drop_floor`),想让边界少动几次。实测
# 它从第 13 轮起**一次都没绑定过**:边界是**预算驱动**的——填充从最新一条往回装、
# 装到预算用完为止,丢掉的永远比这个下限多。而且下限本身受「最近窗口里的东西不丢」
# 这条约束封顶,那个上限每轮只涨 2(对话每轮长 2 条),永远追不上实际丢弃数
# (第 24 轮:上限 33、实际丢了 43——连最近窗口都装不下了)。
#
# 更根本的一条:**对话单调增长时,任何遵守预算的丢弃策略都必须每轮多丢一点**,
# 于是 history 段的第一条每轮都在变,它之后的前缀缓存每轮作废。这不是「少动一点」
# 能解决的,只能靠 L5 那种「偶尔一次大跳」——把丢掉的那一段换成一份不随每轮变化的
# 摘要。所以缓存churn 归阶段二(会话摘要),不归迟滞。

# --- 消息投影:每条历史条目的「呈现形态」(阶段三)-------------------------------
#
# history 是**只追加、一个字节不改**的事实记录;真正发给模型的那份是它的一次投影
# (`project(history, policy) -> messages`)。「压缩」在这一层的含义因此变成**换一种
# 呈现形态**,而不是改动或丢弃事实——这就是 Claude Code 那个 `projectView()` 的等价物,
# 也是他们的 L4 能标成「完全可逆」的原因:压力退下去,同一条历史照样能再展开。
#
# 这些形态原本是散在 `_compressed_history_entries()` 里的一串 if 分支,谁也数不清
# 一共有几种、哪几种可以同时出现。拎成常量之后它们可枚举、可单独测、可按压力组合。
#
#   structural    只有 tool_calls 没有说明文字的 assistant 记录。文本侧等于不存在,
#                 但**条目必须留下**——messages 组装要靠它把调用和结果配成对。
#   full          原样(仍受单条工具结果上限 `tool_output_limit()` 约束)。
#   clipped       窗口外的 user/assistant 文本,尾裁到 CLIPPED_TOKENS。
#   stale_read    窗口内、但之后同一个文件又被写过的 read_file:内容换成
#                 STALE_READ_MARKER。那段全文是**改动之前**的那一版。
#   file_summary  窗口外、memory 里已有摘要的 read_file:换成那句摘要。
#   pointer       窗口外、落过盘的结果:换成落盘指针(可恢复才可丢弃)。
#   stale_pointer 同上,但那个文件之后又被写过:指针照留(全文仍在盘上,是取回的
#                 唯一线索),再加一句 STALE_SPILL_MARKER 说清盘上那份是旧版。
#   error_kept    窗口外、`error:` 开头的结果:**不清内容**(Manus 那条「把错误留在
#                 上下文里」),只裁到 ERROR_KEEP_TOKENS。
#   shell_head    窗口外的 run_shell:命令 + 输出前三行。
#   not_executed  预算用完没跑成的调用:显式说出「没执行」,否则它和正常执行长得一样。
#   cleared       其余窗口外的工具结果:内容换成 CLEARED_RESULT_MARKER。
#   squeezed      预算实在不够时的最后一档:把已经定好的那份呈现再压到 SQUEEZED_TOKENS,
#                 留半句残话。等价于 Claude Code 的 L4(可逆折叠)——history 一个字节
#                 没动,压力退下去同一条还能按原形态展开。它由保留循环回填,不是
#                 `_policy_for()` 选出来的。
#   dropped       过期的重复读:整条不进上下文。它在 messages 侧仍由
#                 `DROPPED_RESULT_NOTE` 占位补上配对,所以「不进上下文」不等于
#                 「那次调用没发生过」。
#
# 优先级就是下面这个顺序在 `_policy_for()` 里的书写顺序,改动前先读那个函数的注释:
# 其中 error 必须排在 pointer 和 shell_head 之前(错误串是一整句话,被 `lines[:3]`
# 切开就读不成句;指针恒在末尾,同样会被切掉)。
POLICY_STRUCTURAL = "structural"
POLICY_FULL = "full"
POLICY_CLIPPED = "clipped"
POLICY_FILE_SUMMARY = "file_summary"
POLICY_POINTER = "pointer"
POLICY_ERROR_KEPT = "error_kept"
POLICY_SHELL_HEAD = "shell_head"
POLICY_NOT_EXECUTED = "not_executed"
POLICY_CLEARED = "cleared"
POLICY_SQUEEZED = "squeezed"
POLICY_DROPPED = "dropped"
POLICY_STALE_READ = "stale_read"
POLICY_STALE_POINTER = "stale_pointer"
HISTORY_POLICIES = (
    POLICY_STRUCTURAL,
    POLICY_FULL,
    POLICY_CLIPPED,
    POLICY_FILE_SUMMARY,
    POLICY_POINTER,
    POLICY_ERROR_KEPT,
    POLICY_SHELL_HEAD,
    POLICY_NOT_EXECUTED,
    POLICY_CLEARED,
    POLICY_SQUEEZED,
    POLICY_DROPPED,
    POLICY_STALE_READ,
    POLICY_STALE_POINTER,
)
# 压扁后留多少。10 个 token 只够留半句话——它的作用不是让模型读懂那一轮,而是让
# 「这一轮发生过」这个事实不消失。再少就退化成噪声,再多就起不到腾空间的作用。
SQUEEZED_TOKENS = 10
# 窗口外的 user/assistant 文本留多少。60 曾经够用是因为那里只会出现 runtime 自己
# 写回的重试提示;模型的工具轮说明进 history 之后,60 会把它砍成一句没头没尾的话。
CLIPPED_TOKENS = 110

# --- 会话摘要(阶段二:L5 便宜的那一半)-----------------------------------------
#
# Claude Code 的 autocompact 分两段:先读磁盘上维护好的会话摘要(**不调模型**),
# 失败才 fork 一个 agent 去写 `<summary>`。这里只抄第一段——一次模型调用是一个
# 约 17 秒的固定往返而且不可逆,没有数据证明必要之前不引入。
#
# 要解决的是阶段一量出来、迟滞治不了的那个病:对话单调增长时,预算驱动的丢弃边界
# **每轮都要往前爬一格**,history 段的第一条每轮都变,它之后的前缀缓存每轮作废
# (实测 24 轮里边界动了 14 次)。摘要把「每轮爬一格」换成「几轮跳一次」:压力到达
# 触发点时一次性把最早的一批条目**整体**换成一份确定性摘要,并把剩下的历史压到
# 预算的 KEEP_RATIO 以下,留出好几轮的增长空间;在下一次跳之前,history 的第一条
# 逐字节不变。
#
# 摘要本身也必须**不随每轮变化**,否则只是把 churn 换了个位置。所以它只在真正推进
# 覆盖点的那一轮重新生成,其余轮次原样复用;生成过程确定性(用过哪些工具、碰过哪些
# 文件、用户提过什么要求),同样不调模型——理由和 `_omitted_digest()` 一样。
#
# 三个数都是相对 total_budget 的比例,不抄绝对值。官方那几个(首次 8k、每 +15k 更新
# 一次、每节 <=2000、整份 <=12000)是按 200k 窗口给的,折成比例约 4% / 7.5% / 1% / 6%,
# 下面的 6% 就是照最后那个折的。
# 压完之后整份 prompt 的占用率目标。**必须按渲染后的实际占用算,不能按原始 history
# token 算**——踩过的坑:第一版判据是「剩下的原始历史 <= 预算的 45%」,而老条目在
# `_compressed_history_entries()` 里早就被逐条裁到 110 token,原始量和真正发出去的量
# 差着数倍,于是「跳一次」跳得太浅,下一轮照样越过触发点、照样再跳一格——边界每轮
# 都在动,和没做这件事完全一样(真实评测负载实测:24 轮里边界动了 14 次,与关掉它
# 一模一样)。0.45 是相对触发点 0.85 留出 40% 的增长空间,大约够五六轮。
SESSION_SUMMARY_TARGET_RATIO = 0.45
SESSION_SUMMARY_BUDGET_RATIO = 0.06
SESSION_SUMMARY_MIN_TOKENS = 135
SESSION_SUMMARY_MAX_TOKENS = 1200
# 覆盖点永远不许吃进最近这几条。把刚发生的一轮换成一句摘要,等于让模型忘掉自己
# 上一步做了什么——而「模型看得见自己上一轮调了什么」正是走到终点率 76% -> 97%
# 的那个机制。最近工具窗口那条线(`_recent_start_by_tool_turns`)同样不许越过。
SESSION_SUMMARY_MIN_KEPT_ENTRIES = 6
SESSION_SUMMARY_HEADER = "Session summary (older transcript, compacted):"
# 摘要里保留几条模型自己的结论，每条裁到多少。
#
# 为什么留结论：Manus 那条「可恢复才可丢弃」在这里的推论是——工具结果可以重跑
# （落盘指针、重新 read_file），**模型自己的推理不能**。所以摘要要是只能留一样，
# 该留的是结论而不是工具输出。取最近 3 条是因为同一件事后面的说法覆盖前面的；
# 每条 60 token 约等于一两句结论，再长就该由用户原话那一段承担了。
SESSION_SUMMARY_MAX_FINDINGS = 3
SESSION_SUMMARY_FINDING_TOKENS = 60
# 用户原话最多占摘要预算的多少。
#
# 必须有这个上限，否则优先级就退化成「排第一的那项吃光全部预算」：实测 22 轮的
# 扫描负载里，user 那几条每条 100+ token，摘要预算只有 289，结论和调用签名一条都
# 排不进去——而它们恰恰是这次修复要保住的东西。0.6 是让三类信息都拿得到位置的
# 最小值：用户原话仍然占大头（它最不可重建），剩下的 40% 够放 1~2 条结论加一行签名。
SESSION_SUMMARY_REQUEST_SHARE = 0.6


SESSION_SUMMARY_MAX_SPANS = 4


def _merge_spans(spans):
    """把区间合并成尽量少的几段。相邻（end + 1 == next start）也算连着。

    合并不是为了好看，是为了让「读过哪几段」这件事**装得进摘要预算**：22 次连着
    读 100 行，不合并要 22 条签名 ≈ 220 token（摘要总预算只有 289），合并之后是
    一条 `audit.log:1-2200`。信息一点没少，反而更接近模型真正需要的那句话。
    """
    ordered = sorted(spans)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _render_call_signatures(calls):
    """把一串 (工具名, 参数) 压成可读的调用签名，读过的区间合并后写出来。

    存在的理由是实测缺陷：摘要从前只写 `already run: read_file x7`，扫长文件的任务
    因此丢掉「我已经读过哪几段」，第 14 轮又去读第 2 轮读过的 1~100 行。计数回答
    「做过多少次」，签名回答「做过哪几次」——后者才是模型下一步要用的。
    """
    grouped = {}
    for name, args in calls:
        args = args or {}
        label = str(args.get("path", "")).strip() or str(args.get("pattern", "")).strip()
        key = (str(name), label)
        entry = grouped.setdefault(key, [])
        start, end = args.get("start"), args.get("end")
        if start is not None and end is not None:
            try:
                entry.append((int(start), int(end)))
            except (TypeError, ValueError):
                pass
    lines = []
    for (name, label), spans in grouped.items():
        if not label:
            lines.append(name)
            continue
        if not spans:
            lines.append("%s %s" % (name, label))
            continue
        merged = _merge_spans(spans)
        shown = merged[:SESSION_SUMMARY_MAX_SPANS]
        text = ",".join("%d-%d" % (start, end) for start, end in shown)
        if len(merged) > len(shown):
            text += ",+%d more" % (len(merged) - len(shown))
        lines.append("%s %s:%s" % (name, label, text))
    return lines


def session_summary_budget(total_budget):
    """摘要本身的 token 上限,从预算派生(不是绝对常量,理由见上面那段注释)。"""
    scaled = int(int(total_budget) * SESSION_SUMMARY_BUDGET_RATIO)
    return max(SESSION_SUMMARY_MIN_TOKENS, min(SESSION_SUMMARY_MAX_TOKENS, scaled))


# 工具结果因为超出最近窗口而被丢掉内容时，留在它位置上的标记。
#
# **必须是一句显式的话，不能只留下调用签名。** 从前这里渲染成 `[tool:read_file]
# {"path": "a.py"}`——那是这次调用自己的签名，模型看到的是一条「结果」消息里装着
# 「调用」，最接近的读法是「这次调用什么都没返回」。官方 `clear_tool_uses` 用的是
# 一句明确的 `"[cleared to save context]"`，保留调用记录、把内容换成占位。
CLEARED_RESULT_MARKER = "[cleared to save context; re-read if needed]"
# 一次 read_file 的结果，在它之后同一个文件又被写过。
#
# 为什么要显式说出来：这条记录里的内容是**改动之前**的那一版，而它在最近窗口里是
# 全文呈现的——和「文件现在就长这样」在模型眼里完全一样。实测后果是 `patch_file`
# 的 `old_text` 从这段过期全文里抄出来，于是命中 0 次被打回（k=3 跑批里 `old_text`
# 没命中占被拒调用的 14%），白烧一个约 17 秒的往返。
#
# 这是「供给侧新鲜度」：记忆层早就在做同一件事——`update_memory_after_tool()` 在
# 写操作之后让 file_summaries 失效，判据是文件内容的 sha256。history 一直没有这一
# 步，于是同一个事实在两个地方一个是新的、一个是旧的。
STALE_READ_MARKER = "[stale: the file was modified after this read; re-read it before quoting or patching]"
# 哪些工具会让先前读到的内容失效。`run_shell` **不在**里面，这是已知盲点而不是
# 遗漏：它能改文件，但改了哪个文件在 args 里读不出来（`run_tool()` 的前后快照知道
# 差异，history 条目上没有）。`run_plan` 靠内层调用自己上报（见 runtime 那边的
# `wrote` 字段），不靠这张表。
WRITE_TOOL_NAMES = frozenset({"write_file", "patch_file"})
# 一条**落过盘**的 read_file 结果,在它之后同一个文件又被写过。
#
# 为什么不能复用 STALE_READ_MARKER:那句话说的是「re-read it」,而这条记录在上下文里
# 只剩一行指针,指针指的是 `.codingforme/tool_outputs/<run_id>/<n>-read_file.txt`——
# 那是**读的那一刻**写下的快照,原文件后来被改了它不会跟着变。照着 STALE_READ_MARKER
# 去「重读」,模型最可能重读的就是指针里那个可以照抄的 `read_file(path=...)` 调用,
# 于是又拿回同一份旧内容——而且这一次它看起来是刚读的。所以这里必须显式说出
# 「盘上那份是旧版、要读的是原路径」。原路径就在同一行的调用签名里,不必再插一遍。
#
# 这是 P3(供给侧新鲜度)漏掉的那一半:P3 的判断写在 `_policy_for()` 的 `if recent:`
# 内部,只覆盖最近窗口。窗口外另外三种形态里,file_summary 已经被记忆层按 sha256
# 作废过一次、cleared 根本不带内容,只有 pointer 仍然携带「可取回但已陈旧」的内容。
STALE_SPILL_MARKER = (
    "[stale: the file was modified after this read; the saved copy is the "
    "pre-change version - re-read the original path above, not the saved copy]"
)


def path_key(raw):
    """把一个路径归一到可比较的形式：正斜杠、去掉 `./` 前缀和首尾斜杠。

    存在的理由是两侧的路径来源不同：**写**的那一侧是
    `diff_workspace_snapshots()` 算出来的工作区相对 posix 路径（权威、规范），
    **读**的那一侧是模型自己敲进 `read_file` 的字符串（可能写成 `./a.py`，
    Windows 上还可能是 `a\b.py`）。不归一就会出现「明明写的是同一个文件、
    过期读却没被作废」——而这种漏判在工件上长得和「本来就没有过期读」一模一样。
    """
    text = str(raw or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")

# 超长工具结果落盘之后，留在上下文里的那一行。
#
# 形状照抄 Claude Code 的 L1（Tool Result Budget）：全文写盘，上下文里只留
# 「预览 + 路径 + 原始大小」，模型用已有的读工具自己取回，不新增工具。这条是
# Manus 那句「可恢复才可丢弃」的落地——`CLEARED_RESULT_MARKER` 只说「被清了」
# 而不给路径，对 `search` / `run_shell` 的输出等于不可恢复。
#
# 路径一律用正斜杠的工作区相对路径：它会被模型原样喂回 `read_file`，而
# `Path.relative_to()` 在 Windows 上给反斜杠，同一段上下文在两个平台上形状不同
# 是最难查的一类问题（`list_files(format="paths")` 踩过同一个坑）。
SPILL_DIR_NAME = "tool_outputs"
SPILL_MARKER_PREFIX = "[full output saved to "


def spill_marker(relative_path, full_tokens, total_lines=0, chunk_lines=0):
    """落盘指针那一行。前缀是常量，`find_spill_marker()` 靠它把这行认回来。

    **必须给出行数和建议的分段大小**，只给 token 数是不够的。踩过的坑：第一版写的是
    `(N tokens); use read_file on that path to see the rest`，模型照做了——一次
    live 压力探针里 4 次去读指针路径，4 次**又落了一次盘**（43,130 token → 留
    14,804），因为落盘文件的全文按定义就大于单条上限，而读它走的是同一个
    `read_file`、受同一个 `tool_output_limit`。于是指针指向的东西永远读不完整，
    两个探针都因此没拿到答案。

    根因是单位对不上：`read_file` 的参数是**行号**，而指针给的是 **token 数**，
    模型无从换算（它既不知道每行多少 token，也不知道上限是多少）。所以这里直接
    把换算做完，给一个可以照抄的调用：`chunk_lines` 由 `_store_tool_output()` 按
    「预览额度 ÷ 每行 token」算出来，留了一成余量。
    """
    head = f"{SPILL_MARKER_PREFIX}{relative_path} ({int(full_tokens)} tokens"
    if int(total_lines or 0) > 0:
        head += f", {int(total_lines)} lines"
    head += ")"
    if int(chunk_lines or 0) > 0:
        return (
            f"{head}; read it back in chunks of about {int(chunk_lines)} lines, "
            f'e.g. read_file(path="{relative_path}", start=1, end={int(chunk_lines)})]'
        )
    return f"{head}; use read_file on that path to see the rest]"


def find_spill_marker(text):
    """从一段工具结果里认出落盘指针那一行，没有就返回空串。

    从后往前找：指针恒在末尾，而预览部分本身可能含有类似的文本。
    """
    for line in reversed(str(text or "").splitlines()):
        stripped = line.strip()
        if stripped.startswith(SPILL_MARKER_PREFIX) and stripped.endswith("]"):
            return stripped
    return ""


# 掉出最近窗口的工具结果里，以 `error:` 开头的那些**不清内容**，只裁到这么多 token。
#
# 理由是 Manus 那条「把错误留在上下文里」：失败的观察是模型据以改变行为的最强信号，
# 清掉它等于让模型忘了自己刚才踩过什么，于是把同一个错误调用再发一遍——而那正好
# 撞上重复调用检测，白烧一个约 17 秒的往返。代价可控：`run_tool()` 的错误串都是
# 一两句话，实测都在 60 个 token 以内，这个上限几乎不会绑定。
ERROR_KEEP_TOKENS = 90

# 最近窗口按**块**推进，不是每轮滑一格。
#
# 为什么：`_recent_start_by_tool_turns()` 每轮重算，历史一超过 RECENT_TOOL_TURNS，
# 每多一个工具轮就恰好把边界往前推一格——上一轮还是全文的那条 `tool` 消息，这一轮
# 变成占位行。消息数组从那个位置往后全变，前缀缓存**每轮**作废一次。Manus 那条
# 「上下文只追加、不修改」说的就是这件事；Claude Code 专门为它写了 Microcompact
# 的 Path B（`cache_edits`，服务端删除、本地消息一个字节不动），而我们的后端没有
# 这个 API，只能退而求其次：把「要清几条」向下取整到 3 的倍数，于是实际保留的窗口
# 在 6~8 条之间浮动，改写频率从每轮一次降到每三轮一次。
#
# 取 3 不取更大：块越大，窗口外那几条推迟得越久、上下文越肥。3 是「省下三分之二的
# 缓存作废」和「最多多留 2 条工具结果」之间的折中。
RECENT_WINDOW_BLOCK = 3


def _recent_start_by_tool_turns(history, tool_turns, block=RECENT_WINDOW_BLOCK):
    """最近窗口按「工具轮」数，不按 history 条目数。

    条目数会骗人：一个工具轮从前只留下一条记录（工具结果），现在还会多一条
    模型的说明文字。按条目数算 6，等于把窗口从 6 个工具轮悄悄砍成 3 个——
    实测就是这样，前 3 轮的说明连同它们的工具结果一起掉出窗口。
    这里从末尾往回数工具条目，数满 tool_turns 个就停，说明文字跟着它那轮走。
    """
    seen = 0
    start = 0
    for index in range(len(history) - 1, -1, -1):
        if history[index]["role"] != "tool":
            continue
        seen += 1
        if seen > tool_turns:
            start = index + 1
            break
    # 条目数上限：一个工具轮最多留下两条记录（模型说明 + 工具结果），
    # 所以窗口不该超过 2 * tool_turns 条。没有这个上限时，一段工具调用很少、
    # 对话很多的历史会被整段判成「最近」，全部按最近条目的额度渲染，预算一次吃光。
    #
    # **这条上限也要按块推进**，理由和 `RECENT_WINDOW_BLOCK` 一模一样，只是作用对象
    # 不同：上面那个 `start` 管的是工具轮那条边界，已经由 `_blocked_recent_window()`
    # 量化过；这条上限管的是「工具很少、对话很多」的历史，而它从前是每多两条对话就
    # 往前推两格。实测(24 轮真实压力探针，预算 12,423，**完全没有触发过裁剪**)：
    # 23 次组装里有 15 次的前缀在中间位置被改写，改写点恰好每轮前移 2——全部来自
    # 这一条。量化之后同一条负载降到 5 次（每三轮一次）。
    # 代价和工具那条一致：窗口浮动着变大，最多多留 2 * block - 1 条。
    cap_start = len(history) - 2 * int(tool_turns)
    block = int(block)
    if block > 1 and cap_start > 0:
        step = 2 * block
        cap_start = (cap_start // step) * step
    return max(start, cap_start)


def _blocked_recent_window(history, base_turns, block=RECENT_WINDOW_BLOCK):
    """按块推进的最近窗口大小，见 `RECENT_WINDOW_BLOCK` 的注释。

    「要清掉几条工具结果」向下取整到 `block` 的倍数，于是实际保留的条数在
    `base_turns` 到 `base_turns + block - 1` 之间浮动，而窗口边界（也就是消息
    数组里第一条被改写的位置）每 `block` 个工具轮才动一次。
    """
    block = int(block)
    base_turns = int(base_turns)
    total = sum(1 for item in history if item.get("role") == "tool")
    surplus = total - base_turns
    if surplus <= 0 or block <= 1:
        return base_turns
    return total - (surplus // block) * block


def _head_tail_clip(text, limit, model=None):
    """保留首尾、省略中间的裁剪——仅用于 current_request 兜底分支。

    为什么不用 `_tail_clip`：用户请求的关键约束经常同时出现在开头(要做什么)
    和结尾(边界条件、"不要动 xxx" 之类的补充说明)，只砍尾巴容易连着关键信息
    一起丢掉。这里退而求其次，两头都留一点，只丢中间。
    """
    return models.head_tail_clip_tokens(text, limit, model)


def _budget_or_none(budgets, section):
    """某段这一轮的额度，没有额度就返回 None。

    `PROTECTED_SECTIONS` 恒为 None；关掉裁剪时所有段都是 None。折成 0 会让工件里
    「不设上限」和「额度是零」变成同一个数，而两者含义相反。
    """
    if section in PROTECTED_SECTIONS:
        return None
    value = budgets.get(section)
    return None if value is None else int(value)


@dataclass
class SectionRender:
    raw: str
    # None = 这一段没有额度（prefix 恒定如此；关掉裁剪时所有段都如此）。
    # 不要写成 0：0 会在工件里被读成"额度是零"，含义正相反。
    budget: int | None
    rendered: str
    details: dict | None = None
    # 裁剪之后**活下来的条目**，每条带回它的来源 history item。
    #
    # 为什么不放进 details：details 会整体进 metadata → trace → 落盘，而这里挂着
    # 原始 item（含完整文件内容）。放进去等于把每轮 prompt 的全文再抄一份到
    # trace 里。这条通道只服务于 `_assemble_messages()`，是进程内的。
    kept_entries: list | None = None

    # 度量一律按 token。工件里不再出现任何 `*_chars` 字段——同时摆两种单位，
    # 读的人无从知道某个数是哪一种，这正是要消掉的问题。
    @property
    def raw_tokens(self):
        return count_tokens(self.raw)

    @property
    def rendered_tokens(self):
        return count_tokens(self.rendered)


class ContextManager:
    def __init__(
        self,
        agent,
        total_budget=FALLBACK_TOTAL_BUDGET,
        section_budgets=None,
        section_floors=None,
        reduction_order=None,
    ):
        self.agent = agent
        # token 预算。名字沿用 total_budget（它是 HarnessSpec 的字段名），
        # 但单位已经从字符换成 token，见模块顶部。
        self.total_budget = int(total_budget)
        # 默认为空 = 各段不设固定上限，起始额度就是 total_budget。这里只收调用方
        # **显式**指定的那几段（HarnessSpec 的消融变体、evaluator 的任务 setup、
        # 测试用例都靠它构造"预算很小"的场景）。
        #
        # prefix 被显式挡掉：它没有额度这个概念，收下一个 prefix 额度就等于给了
        # 调用方一条绕过 PROTECTED_SECTIONS 的路。
        self.section_budgets = {}
        if section_budgets:
            self.section_budgets.update(
                {
                    str(key): int(value)
                    for key, value in section_budgets.items()
                    if str(key) not in PROTECTED_SECTIONS
                }
            )
        self._section_floor_overrides = {str(key): int(value) for key, value in (section_floors or {}).items()}
        self.section_floors = self._compute_section_floors()
        # 过滤 PROTECTED_SECTIONS：调用方（包括老的 HarnessSpec、测试）可能仍然
        # 传进一个含 prefix 的顺序，这里统一挡掉，保证"prefix 不被裁"是不变量而不是默认值。
        self.reduction_order = tuple(
            section
            for section in (reduction_order or DEFAULT_REDUCTION_ORDER)
            if section not in PROTECTED_SECTIONS
        )

    def _graded_compression_enabled(self):
        """消融开关。关掉就严格退回分级压缩出现之前的行为:触发点和目标点都是
        100%,history 没有跨轮保持的丢弃边界。不是「换成第三种形态」——那样 A/B
        测的就不是这个机制。"""
        checker = getattr(self.agent, "feature_enabled", None)
        if not callable(checker):
            return True
        return bool(checker("graded_compression"))

    def _compression_thresholds(self):
        """返回 `(触发点, 目标点)`,单位都是 token。"""
        if not self._graded_compression_enabled():
            return self.total_budget, self.total_budget
        trigger = int(self.total_budget * COMPRESSION_TRIGGER_RATIO)
        target = int(self.total_budget * COMPRESSION_TARGET_RATIO)
        # 目标点不能低于各段下限之和,否则循环必然走到 floor_exhausted、把
        # 「压不下去」这个本该报警的状态变成常态。
        floor_sum = sum(int(self.section_floors.get(section, 0)) for section in self.reduction_order)
        return max(trigger, floor_sum), max(target, floor_sum)

    def _clear_at_least_enabled(self):
        """消融开关。关掉就退回「只裁刚好够的量」——一次裁剪只把 overflow 那么多
        腾出来,于是下一轮几乎必然再裁一次,而每裁一次 history 就作废它之后的全部
        前缀缓存。这是 `CLEAR_AT_LEAST_DIVISOR` 出现之前的行为,不是第三种形态。"""
        checker = getattr(self.agent, "feature_enabled", None)
        if not callable(checker):
            return True
        return bool(checker("clear_at_least"))

    def _reversible_squeeze_enabled(self):
        """消融开关。关掉就退回「放不下直接丢」——窗口外的条目不再压扁成残句,
        整条从投影里消失,只在 `Omitted context:` 那一行里留个数。

        这个开关存在的理由和别的不太一样:压扁这一步在被命名成 `POLICY_SQUEEZED`
        之前就藏在保留循环里了,一直没有对照组,所以「留个残句」到底比「整条丢掉」
        好在哪,在工件上从来没有被量过。
        """
        checker = getattr(self.agent, "feature_enabled", None)
        if not callable(checker):
            return True
        return bool(checker("reversible_squeeze"))

    def compression_thresholds(self):
        """公开入口:`/context` 要显示「现在离自动压缩还有多远」。

        单独开一个公开方法而不是让 CLI 去读 `_compression_thresholds()`,是因为
        分级关掉时两个点都回到 100%,这件事必须跟着数字一起报出来——否则用户看到
        「触发点 = 预算」会以为是显示错了。
        """
        trigger, target = self._compression_thresholds()
        return {
            "graded": self._graded_compression_enabled(),
            "budget_tokens": int(self.total_budget),
            "trigger_tokens": int(trigger),
            "target_tokens": int(target),
        }

    # --- 会话摘要(阶段二)-------------------------------------------------------

    def _session_summary_enabled(self):
        checker = getattr(self.agent, "feature_enabled", None)
        if not callable(checker):
            return True
        return bool(checker("session_summary"))

    def _session_summary_state(self):
        """摘要状态存在 `session["context_summary"]` 里,跟着 session 一起落盘、一起 resume。

        迁移计划里写的是单开一个 `.codingforme/sessions/<id>.summary.md`;没那么做,
        因为 session 本身就是那份「可恢复状态」的落盘产物,再开一个文件要自带
        load / save / resume 三条路径,换不到任何东西。四个字段:
        `covered`(摘要覆盖了 history 的前几条)、`text`(摘要正文)、
        `covered_tokens`(被换掉的那一段原始有多大)、`refreshes`(跳了几次)。
        """
        session = getattr(self.agent, "session", None)
        if not isinstance(session, dict):
            return {"covered": 0, "text": "", "covered_tokens": 0, "refreshes": 0}
        state = session.get("context_summary")
        if not isinstance(state, dict):
            state = {}
            session["context_summary"] = state
        state.setdefault("covered", 0)
        state.setdefault("text", "")
        state.setdefault("covered_tokens", 0)
        state.setdefault("refreshes", 0)
        # 防御性夹紧:`/reset` 之后 history 归零,一个陈旧的 covered 会把整段历史切没。
        # reset() 自己也会清这份状态,这里是第二道保险(老 session 文件同理)。
        if int(state["covered"] or 0) > len(session.get("history", []) or []):
            state["covered"] = 0
            state["text"] = ""
            state["covered_tokens"] = 0
        return state

    def _summary_covered(self):
        if not self._session_summary_enabled():
            return 0
        return int(self._session_summary_state().get("covered", 0) or 0)

    def _summary_text(self):
        if not self._session_summary_enabled():
            return ""
        return str(self._session_summary_state().get("text", "") or "")

    def _max_summary_coverage(self, history):
        """覆盖点的硬上限:最近工具窗口之前,且至少留下 MIN_KEPT_ENTRIES 条。"""
        block = RECENT_WINDOW_BLOCK
        checker = getattr(self.agent, "feature_enabled", None)
        if callable(checker) and not checker("recent_window_block"):
            block = 1
        recent_window = _blocked_recent_window(history, RECENT_TOOL_TURNS, block=block)
        recent_start = _recent_start_by_tool_turns(history, recent_window, block=block)
        return max(0, min(recent_start, len(history) - SESSION_SUMMARY_MIN_KEPT_ENTRIES))

    def _compact_session_summary(self, entries_budget):
        """把最早的一批历史条目整体换成一份摘要。真的推进了覆盖点才返回 True。

        `entries_budget` 是「压完之后,history 里的条目最多还能占多少 token」,由
        调用方从整份 prompt 的占用率目标倒算(见 `_summary_entries_budget()`)。
        目标不是「刚好够」,而是**一次跳到目标点以下**:只腾刚好够的量正是阶段一
        量出来的那个病——下一轮几乎必然再动一次,每动一次就作废 history 之后的
        全部前缀缓存。

        它排在逐段硬裁**之前**:硬裁是不可逆地丢内容,而这一步把内容换成摘要,
        丢的是细节不是「发生过这件事」本身。
        """
        if not self._session_summary_enabled():
            return False
        history = list(getattr(self.agent, "session", {}).get("history", []))
        state = self._session_summary_state()
        covered = int(state.get("covered", 0) or 0)
        ceiling = self._max_summary_coverage(history)
        if covered >= ceiling:
            return False
        # 每条条目**渲染之后**占多少,而不是原始有多大:老条目会被裁到 110 token、
        # 过期的重复读会被整条去掉、工具结果会被换成一行摘要或落盘指针。按原始量
        # 算会严重高估剩余量、于是跳得太浅(见 SESSION_SUMMARY_TARGET_RATIO 的注释)。
        model = self._model_name()
        tail = history[covered:]
        block = RECENT_WINDOW_BLOCK
        checker = getattr(self.agent, "feature_enabled", None)
        if callable(checker) and not checker("recent_window_block"):
            block = 1
        recent_start = _recent_start_by_tool_turns(
            tail, _blocked_recent_window(tail, RECENT_TOOL_TURNS, block=block), block=block
        )
        entries, _ = self._compressed_history_entries(tail, recent_start)
        costs = {}
        for entry in entries:
            costs[id(entry.get("item"))] = count_tokens("\n".join(entry.get("lines", [])), model) + 1
        remaining = sum(costs.values())
        new_covered = covered
        while new_covered < ceiling and remaining > entries_budget:
            remaining -= costs.get(id(history[new_covered]), 0)
            new_covered += 1
        if new_covered <= covered:
            return False
        state["covered"] = new_covered
        # 被换掉的那一段**原始**有多大。它只进工件、不参与任何判据——回答的是
        # 「这次跳省下了多少」,所以要按原始量记,不是按渲染后的量。
        state["covered_tokens"] = count_tokens(
            self._raw_history_text(history[:new_covered]), self._model_name()
        )
        state["text"] = self._render_session_summary(history[:new_covered])
        state["refreshes"] = int(state.get("refreshes", 0) or 0) + 1
        return True

    def compact_now(self):
        """用户主动要求压缩:把摘要覆盖点一次推到结构允许的最远处。

        走的是**同一个** `_compact_session_summary()`,只是把「压完之后 history 还能
        占多少 token」这个参数设成 0——于是那个 while 循环一直推到 `_max_summary_coverage()`
        为止。刻意不另写一条压缩路径:第二条路径意味着「手动压」和「自动压」会长出
        两种不同的结果形状,而它们落进同一个 `session["context_summary"]`、跟着同一份
        session 落盘和 resume。

        为什么需要手动入口:自动压缩是**占用率驱动**的,到 85% 才动手。而用户知道
        一些系统不知道的事——「这一段调查结束了」「换个话题」——那一刻主动压掉,
        比等它自己涨到 85% 再压更省(压缩点越靠前,被作废的前缀缓存越少)。

        返回一份可直接显示的报告 dict,**推不动时也返回**(带 `reason`)。不能返回
        None 或抛异常:「压过了但没什么可压的」和「这个功能没开」对用户是两件事。
        """
        history = list(getattr(self.agent, "session", {}).get("history", []))
        state = self._session_summary_state()
        before_covered = int(state.get("covered", 0) or 0)
        model = self._model_name()
        before_tokens = count_tokens(self._raw_history_text(history), model)
        if not self._session_summary_enabled():
            return {
                "compacted": False,
                "reason": "session_summary is turned off for this agent",
                "covered": before_covered,
                "entries": len(history),
            }
        ceiling = self._max_summary_coverage(history)
        if before_covered >= ceiling:
            # 说清楚为什么推不动。三种情况长得一样但用户该做的事完全不同:历史太短
            # (再聊几轮就行)、热尾挡住了(正常,压过头会把刚发生的事丢掉)、已经压到
            # 头了(不用再敲)。只回一句「没压」会被读成命令没生效。
            if len(history) <= SESSION_SUMMARY_MIN_KEPT_ENTRIES:
                reason = "the transcript is too short to compact (%d entries; at least %d are always kept)" % (
                    len(history),
                    SESSION_SUMMARY_MIN_KEPT_ENTRIES,
                )
            elif before_covered:
                reason = "already compacted up to the protected recent tail"
            else:
                reason = "nothing left to compact; the protected recent tail already starts here"
            return {
                "compacted": False,
                "reason": reason,
                "covered": before_covered,
                "entries": len(history),
            }
        self._compact_session_summary(0)
        state = self._session_summary_state()
        covered = int(state.get("covered", 0) or 0)
        summary_text = str(state.get("text", "") or "")
        # 「压完之后 history 这一段还有多大」= 摘要本身 + 没被覆盖的那些条目原文。
        after_tokens = count_tokens(summary_text, model) + count_tokens(
            self._raw_history_text(history[covered:]), model
        )
        return {
            "compacted": covered > before_covered,
            "reason": "",
            "covered": covered,
            "newly_covered": covered - before_covered,
            "entries": len(history),
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "saved_tokens": max(0, before_tokens - after_tokens),
            "summary": summary_text,
        }

    def _summary_entries_budget(self, prompt_tokens, history_render):
        """压完之后 history 里的条目还能占多少 token。

        从整份 prompt 的占用率目标倒算:先减掉这一轮 history 之外的那些段(它们不受
        摘要影响),再减掉摘要自己那份(按上限估,保守方向)。
        """
        non_history = max(0, int(prompt_tokens) - int(history_render.rendered_tokens))
        prompt_target = int(self.total_budget * SESSION_SUMMARY_TARGET_RATIO)
        return max(0, prompt_target - non_history - session_summary_budget(self.total_budget))

    def _render_session_summary(self, items):
        """确定性地把一批历史条目压成一份摘要。**不调模型。**

        比 `_omitted_digest()` 多出来的是三样东西，每一样都是踩过坑补的：

        1. **用户说过什么**（原文）。阶段零那条 24 轮探针最后丢了 39 条历史，只剩
           一行「dropped 39 entries」，而用户在第 1 轮定下的那条规格已经不在上下文里、
           模型却仍被要求遵守它。
        2. **模型自己得出过什么结论**。从前这里只写一行 `N assistant turn(s) not
           repeated here`——那等于把整段推理抹掉。**文件可以重读，推理不能**，所以在
           「可恢复才可丢弃」这条原则下，assistant 的结论比工具结果更该留。
        3. **调用签名带区间**，不只是计数。从前写的是 `already run: read_file x7`，
           于是分块扫长文件的任务丢掉了「我已经读过哪几段」——实测
           `long_log_audit_token` 在第 14 轮重新读了第 2 轮就已经读过的 1~100 行，
           白烧一个约 17 秒的往返，还正好撞上重复调用检测。

        预算紧时按优先级退让：用户原话 > 模型结论 > 带区间的签名（退回纯计数）>
        文件清单。用户那几条走首尾保留裁剪：最早那条通常是任务定义，最近几条是刚
        追加的要求，中间那些才是可以省的。
        """
        if not items:
            return ""
        model = self._model_name()
        budget = session_summary_budget(self.total_budget)
        requests = []
        findings = []
        tool_counts = {}
        calls = []
        paths = []
        for item in items:
            role = item.get("role")
            if role == "tool":
                name = str(item.get("name", "")) or "?"
                tool_counts[name] = tool_counts.get(name, 0) + 1
                args = item.get("args") or {}
                path = str(args.get("path", "")).strip()
                if path and path not in paths:
                    paths.append(path)
                calls.append((name, args))
                continue
            content = " ".join(str(item.get("content", "")).split())
            if not content:
                continue
            if role == "user":
                requests.append(content)
            else:
                findings.append(content)

        head = SESSION_SUMMARY_HEADER + "\n- covers the first %d transcript entries" % len(items)
        spent = count_tokens(head, model)
        # 用户原话单独走首尾保留裁剪，所以先把它的额度扣掉，剩下的才归下面几行分。
        request_label = "\n- earlier user requests (verbatim, oldest first):\n"
        request_body = "\n".join("  %d. %s" % (index + 1, text) for index, text in enumerate(requests))
        request_cost = 0
        if requests:
            request_cost = min(
                count_tokens(request_label + request_body, model),
                max(20, int((budget - spent) * SESSION_SUMMARY_REQUEST_SHARE)),
            )
        room = budget - request_cost

        def fit(line):
            """装得下就收下，装不下就整行不要——半句话的清单比没有更糟。"""
            nonlocal head, spent
            cost = count_tokens("\n" + line, model)
            if spent + cost > room:
                return False
            head += "\n" + line
            spent += cost
            return True

        kept_findings = 0
        # 取最近的几条：同一件事，后面的说法覆盖前面的。
        for text in findings[-SESSION_SUMMARY_MAX_FINDINGS:]:
            if not fit("- earlier finding: " + _tail_clip(text, SESSION_SUMMARY_FINDING_TOKENS, model)):
                break
            kept_findings += 1
        if findings and not kept_findings:
            # 一条都放不下时至少说出「有过这些轮」，别让它们无声消失。
            fit("- %d assistant turn(s) not repeated here" % len(findings))
        signatures = _render_call_signatures(calls)
        if signatures:
            if not fit("- already run: " + "; ".join(signatures)):
                counts = ", ".join("%s x%d" % (name, count) for name, count in sorted(tool_counts.items()))
                fit("- already run: " + counts)
        if paths:
            # 签名行装得下时这行是冗余的，所以排最后。
            fit("- files touched: " + ", ".join(paths))

        if not requests:
            return _tail_clip(head, budget, model)
        available = max(20, budget - count_tokens(head + request_label, model))
        return head + request_label + _head_tail_clip(request_body, available, model)

    def _model_name(self):
        client = getattr(self.agent, "model_client", None)
        return getattr(client, "model", None)

    def _tool_schema_tokens(self):
        """工具 schema 每轮都要重发，所以它算在预算里。

        实测基础 6 个工具的 `tools=` 数组是 4615 字符 / 1169 token，约占实际输入的
        53%，而它一个 token 都不进 `prompt_tokens`——不计等于预算漏掉了输入的一半。
        取值走 agent 那边（它才持有注册表），拿不到就当 0，不让缺字段变成异常。
        """
        getter = getattr(self.agent, "tool_schema_tokens", None)
        if not callable(getter):
            return 0
        try:
            return int(getter())
        except Exception:
            return 0

    def _measure(self, prompt):
        """这一轮组装出来的上下文有多少 token。

        **不含工具 schema，这是刻意的。** schema 确实每轮都发、确实占实际输入的
        一半，但它已经在派生预算时扣掉了一次（`models.context_budget_tokens()`
        用 `窗口 × 安全系数 − schema − 输出预留` 算出 total_budget）。这里再加一次
        就是重复计账，还会让 total_budget 有一个隐形下限——低于 schema 大小的预算
        永远不可满足，裁剪循环会把所有 section 砍到底也退不出去。

        所以分工是：**派生时扣一次，测量时不管它**；schema 的实际大小照旧写进
        metadata 的 `tool_schema_tokens`，全貌在工件里仍然读得到。
        """
        return count_tokens(prompt, self._model_name())

    def build(self, user_message):
        """`build_all()` 的兼容入口，只返回 `(prompt, metadata)`。

        真正发给模型的是 messages 数组，见 `build_all()`。这里保留纯文本形态，
        是因为预算裁剪和 trace 里的 prompt 元数据都按压平后的这段文本算 token——
        它仍然是**同一次组装**的产物，不是另一条并行链路。
        """
        _, prompt, metadata = self.build_all(user_message)
        return prompt, metadata

    def build_all(self, user_message):
        """按预算组装一轮完整上下文，产出标准 messages 数组。

        为什么存在：
        仅靠用户这一轮输入，模型并不知道当前仓库状态、会话里已经读过什么、
        哪些旧信息还值得继续参考。这个函数负责把“稳定基线 + 工作记忆 +
        相关笔记 + 历史 + 当前请求”拼成真正发给模型的 prompt。

        输入 / 输出：
        - 输入：`user_message`，也就是用户当前这一轮的新请求。
        - 输出：`(messages, prompt, metadata)`。
          `messages` 是真正发给模型的标准对话数组（system / user / assistant
          带 tool_calls / tool 带 tool_call_id）；
          `prompt` 是同一份内容压平成的纯文本，预算裁剪和各项字符数度量都按它算；
          `metadata` 记录了每个 section 的原始长度、裁剪后的长度、是否触发了
          预算收缩等信息，后续会进入 trace/report，便于解释这轮 prompt
          是怎么被拼出来的。

        在 agent 链路里的位置：
        它位于 `CodingForMe.ask()` 的每轮模型调用之前，是“真正发请求给模型”
        的最后一道组装工序。`WorkspaceContext` 提供稳定前缀，`LayeredMemory`
        提供工作记忆，这个函数则把它们和当前请求合成一份可控大小的 prompt。
        """
        user_message = str(user_message)
        self.section_floors = self._compute_section_floors()
        memory_enabled = True
        relevant_memory_enabled = True
        context_reduction_enabled = True
        if hasattr(self.agent, "feature_enabled"):
            memory_enabled = self.agent.feature_enabled("memory")
            relevant_memory_enabled = self.agent.feature_enabled("relevant_memory")
            context_reduction_enabled = self.agent.feature_enabled("context_reduction")
        section_texts = {
            "prefix": str(getattr(self.agent, "prefix", "")),
            "memory": "Memory:\n- disabled" if not memory_enabled else str(self.agent.memory_text()),
            "history": "",
            CURRENT_REQUEST_SECTION: f"Current user request:\n{user_message}",
        }
        checkpoint_text = ""
        if hasattr(self.agent, "render_checkpoint_text"):
            checkpoint_text = str(self.agent.render_checkpoint_text() or "").strip()
        if checkpoint_text:
            section_texts["prefix"] = section_texts["prefix"] + "\n\n" + checkpoint_text
        selected_notes = []
        if memory_enabled and relevant_memory_enabled and hasattr(self.agent, "memory") and hasattr(self.agent.memory, "retrieval_candidates"):
            selected_notes = self.agent.memory.retrieval_candidates(user_message, limit=RELEVANT_MEMORY_LIMIT)

        if not context_reduction_enabled:
            rendered = self._render_sections_without_reduction(section_texts, selected_notes=selected_notes)
            prompt = self._assemble_prompt(rendered)
            metadata = self._metadata(
                prompt=prompt,
                rendered=rendered,
                budgets={section: render.budget for section, render in rendered.items() if section != CURRENT_REQUEST_SECTION},
                reduction_log=[],
                selected_notes=selected_notes,
                user_message=user_message,
                section_texts=section_texts,
            )
            messages, placement = self._assemble_messages(rendered, user_message)
            self._record_message_metadata(metadata, messages, placement)
            return messages, prompt, metadata

        # 起始额度：不设上限的段拿到 total_budget（"你最多可以用完整个预算"），
        # 调用方显式指定的段用它自己的值。prefix 不在其中——它没有额度。
        budgets = {section: self.total_budget for section in self.reduction_order}
        budgets.update(self.section_budgets)
        rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)
        prompt = self._assemble_prompt(rendered)
        reduction_log = []

        # 只有整份 prompt 顶到触发点时才走到下面这个循环——各段没有固定额度，
        # 装得下就一个 token 都不裁。顺序体现平台偏好：先牺牲 relevant_memory，
        # 再牺牲 history，最后才动 memory。prefix 和当前用户请求都不在里面。
        prompt_tokens = self._measure(prompt)
        # 分级：触发点 85%、目标点 70%。关掉消融开关时两者都回到 100%，循环条件
        # 和目标额度就都退回改动之前的样子。
        trigger_tokens, target_tokens = self._compression_thresholds()
        occupancy_before = (prompt_tokens / self.total_budget) if self.total_budget else 0.0
        # "可裁的都裁到底了、仍然放不下"。判据是裁剪循环自己报的：走完整条
        # reduction_order 一个 section 都动不了。
        #
        # 不能再像从前那样查 `budgets[section] <= floor`：各段起始额度现在是
        # total_budget，一个天然就比下限还小的 section 永远不会被写回一个 <= floor
        # 的额度（它一进循环就因为"没得可裁"被跳过），于是那个查法恒为假。
        floor_exhausted = False
        summary_compactions = 0
        while prompt_tokens > trigger_tokens:
            # 额度、下限、overflow 同为 token，直接相减。（这里曾经写着"overflow 折成
            # 字符"，那是混合制时期的遗留：闸门按 token 而各段额度按字符，得先折算。
            # 单位统一之后折算就不存在了。）
            # 「超了多少」相对**目标点**算，不是相对硬预算——一压就压到目标点以下，
            # 留出余量。分级关掉时目标点就是硬预算，这一行退回原样。
            overflow = prompt_tokens - target_tokens
            reduced = False
            # 会话摘要排在逐段硬裁**之前**:硬裁是不可逆地丢内容,摘要是把最早那一批
            # 条目换成一份确定性的、跨轮不变的概述——先做便宜且可交代的那一步,这就是
            # Claude Code 那条「按代价从低到高排」的组织原则。推进成功就重组一次再量,
            # 它推不动了(已经压到 KEEP_RATIO 以下、或者只剩最近窗口)才轮到硬裁。
            entries_budget = self._summary_entries_budget(prompt_tokens, rendered["history"])
            if self._compact_session_summary(entries_budget):
                rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)
                prompt = self._assemble_prompt(rendered)
                prompt_tokens = self._measure(prompt)
                summary_compactions += 1
                continue
            for section in self.reduction_order:
                floor = int(self.section_floors.get(section, 0))
                current_budget = int(budgets.get(section, 0))
                # 起始额度是 total_budget，而某段实际只渲染出几百个 token；直接拿名义
                # 额度去减 overflow，减出来的新额度仍然远大于实际大小，这一轮什么都
                # 没裁掉，prompt 一个 token 都没变，循环于是要跑上万次才收敛。
                # 先收到实际渲染大小，再减——每一轮就都真的裁掉了东西。
                current_budget = min(current_budget, int(rendered[section].rendered_tokens))
                if current_budget <= floor:
                    continue
                # `clear_at_least`：一次至少腾出预算的 1/10，而不是只裁刚好够的量。
                # 理由见 CLEAR_AT_LEAST_DIVISOR——最小裁剪等于每轮都要重裁一次，
                # 每次都作废 history 之后的全部前缀缓存。
                target = overflow
                if self._clear_at_least_enabled():
                    target = max(overflow, self.total_budget // CLEAR_AT_LEAST_DIVISOR)
                new_budget = max(floor, current_budget - target)
                if new_budget >= current_budget:
                    continue
                reduction_log.append(
                    {
                        "section": section,
                        "before_tokens": current_budget,
                        "after_tokens": new_budget,
                        "overflow_tokens": overflow,
                        # 实际按多少裁的。和 overflow 不同就说明 clear_at_least 生效了。
                        "target_tokens": target,
                    }
                )
                budgets[section] = new_budget
                rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)
                prompt = self._assemble_prompt(rendered)
                prompt_tokens = self._measure(prompt)
                reduced = True
                break
            if not reduced:
                floor_exhausted = True
                break

        # 兜底分支：只有当 current_request 自己就超过 total_budget、且其它
        # section 已经全部压到 floor 仍不够时才会走到这里。正常情况下
        # current_request 完全不受上面这个压缩循环影响——这里不是把它纳入
        # 常规裁剪顺序，而是处理"怎么裁都裁不出空间"的极端情况：与其让一个
        # 超预算的 prompt 原样发给模型、指望后端报错兜底，不如在这一步就把
        # 决定权收回来，用可审计的方式砍它，并显式告诉模型这一情况。
        # 进入条件必须显式写出"请求自己放不下"，不能只看 prompt 超没超预算。
        # 从前这两者等价是**碰巧**的：prefix 能被裁到 floor，所有 section 压到底之后
        # 剩余空间很小，于是 prompt 仍然超预算就意味着请求本身很大。现在 prefix 不再
        # 参与裁剪，可减的空间少了一截，光看"超预算"会把一个正常长度的用户请求也砍掉。
        current_request_truncated = False
        current_request_dropped_tokens = 0
        # 判据就是文档里那一句原话：**请求自己撑爆了 total_budget**。不要写成
        # "其它 section 占完之后还剩多少"——那个量会随 prefix 保护、随 floor 配置
        # 变化，`total_budget` 比下限之和还小时它甚至恒为真，于是正常长度的请求
        # 也会被砍掉。
        header = "Current user request:\n"
        note = (
            "\n\n[注意：用户原始请求过长，已保留首尾并省略中间部分；"
            "如果任务因此显得不完整，请提示用户缩短请求或分步描述。]"
        )
        request_cannot_fit = (
            count_tokens(header + user_message + note, self._model_name()) > self.total_budget
        )
        if prompt_tokens > self.total_budget and request_cannot_fit:
            model = self._model_name()
            current_render = rendered[CURRENT_REQUEST_SECTION]
            # 单位统一之后这里是一次算式，不再是「估算 → 裁 → 重新量」的迭代。
            # 从前预算是 token 而截断按字符，得先用本轮的字符/token 比折算出字符额度；
            # 而那个比值是按**裁剪前**的文本估的，被一长串重复字符撑高之后算出的额度
            # 偏大，裁完仍然超预算（实测一次裁完还剩 132 个 token、预算 65），只能循环
            # 逼近。现在额度和截断同为 token，一步到位。
            other_tokens = prompt_tokens - count_tokens(current_render.rendered, model)
            overhead = count_tokens(header, model) + count_tokens(note, model)
            available = max(0, self.total_budget - other_tokens - overhead)
            clipped_message = _head_tail_clip(user_message, available, model)
            if count_tokens(clipped_message, model) < count_tokens(user_message, model):
                current_request_truncated = True
                current_request_dropped_tokens = count_tokens(user_message, model) - count_tokens(
                    clipped_message, model
                )
                rendered[CURRENT_REQUEST_SECTION] = SectionRender(
                    raw=current_render.raw,
                    budget=0,
                    rendered=header + clipped_message + note,
                    details={"truncated": True},
                )
                prompt = self._assemble_prompt(rendered)
                prompt_tokens = self._measure(prompt)

        # 工件口径：起始额度等于 total_budget 表达的是「这一段不设上限」，但记成
        # 118335 会被读成「分到了这么多」。只有两种情况才是真额度：调用方显式给了
        # 一个值，或者裁剪循环真的把它压下去过。其余记 None，和 prefix 一致。
        reduced_sections = {str(entry["section"]) for entry in reduction_log}
        reported_budgets = {
            section: (
                value
                if section in self.section_budgets or section in reduced_sections
                else None
            )
            for section, value in budgets.items()
        }
        metadata = self._metadata(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            rendered=rendered,
            budgets=reported_budgets,
            reduction_log=reduction_log,
            selected_notes=selected_notes,
            user_message=user_message,
            section_texts=section_texts,
            current_request_truncated=current_request_truncated,
            current_request_dropped_tokens=current_request_dropped_tokens,
            floor_exhausted=floor_exhausted,
        )
        # 分级压缩的观测口径。零值也写:「没触发」和「触发了但没省下什么」在别的
        # 字段上分不出来,而这套东西上一轮踩的坑就是「机制一次都没执行」在报告里
        # 长得和「没问题」一模一样。占用率是相对硬预算算的,不是相对触发点——
        # 读的人关心的是「离撑爆还有多远」。
        history_details = rendered["history"].details if "history" in rendered else {}
        summary_state = self._session_summary_state()
        metadata["context_pressure"] = {
            "graded": self._graded_compression_enabled(),
            "trigger_tokens": int(trigger_tokens),
            "target_tokens": int(target_tokens),
            "occupancy_before": round(float(occupancy_before), 4),
            "occupancy_after": round(
                (prompt_tokens / self.total_budget) if self.total_budget else 0.0, 4
            ),
            # 「这一轮做了压缩」= 任意一级动过手,不只是逐段硬裁。阶段二加进会话摘要
            # 之后两者不再等价:摘要一步就把占用率压回目标点以下,`reduction_log` 是空的,
            # 而压缩确确实实发生了。分级看下面的 `session_summary.compactions`。
            "triggered": bool(reduction_log) or summary_compactions > 0,
            "freed_tokens": max(0, int(occupancy_before * self.total_budget) - int(prompt_tokens)),
            "omitted_entry_count": int(history_details.get("omitted_entry_count", 0) or 0),
            # 会话摘要(阶段二)。`compactions` 是这一轮真的推进了几次覆盖点——
            # 恒为 0 就说明这套东西一次都没执行,而那正是上一轮踩的坑:「机制没跑」
            # 在报告里长得和「没问题」一模一样。`covered_entries` / `covered_tokens`
            # 记的是累计换掉了多大一段,`refreshes` 记它一共跳过几次。
            "session_summary": {
                "enabled": self._session_summary_enabled(),
                "compactions": int(summary_compactions),
                "covered_entries": int(history_details.get("summary_covered_entries", 0) or 0),
                "summary_tokens": int(history_details.get("summary_tokens", 0) or 0),
                "covered_tokens": int(summary_state.get("covered_tokens", 0) or 0),
                "refreshes": int(summary_state.get("refreshes", 0) or 0),
            },
        }
        messages, placement = self._assemble_messages(rendered, user_message)
        self._record_message_metadata(metadata, messages, placement)
        return messages, prompt, metadata

    def _record_message_metadata(self, metadata, messages, placement=None):
        """把 messages 的形状记进 metadata，供 trace/report 观测。

        只记形状不记内容：条数、角色分布、token 总量、以及**批量轮次**——
        `assistant` 消息里 `tool_calls` 多于一个的有几轮。最后这个数字是
        T2-1 想要影响的量，没有它就只能靠事后翻 trace 数。
        `messages_tokens` 与 `prompt_tokens` 会有出入（结构字段替代了文本前缀），
        两个都留着，差值本身就是这次改动的开销。

        `message_layout` 记的是**摆位**：快照有没有从 system 里分出去、memory
        有没有单独成条。缓存命中率是否合理只能对着摆位读——同一份内容摆错位置，
        命中率会差一倍以上，而光看条数和角色分布看不出摆位。
        """
        roles = {}
        for message in messages:
            roles[str(message.get("role", ""))] = roles.get(str(message.get("role", "")), 0) + 1
        metadata.update(
            {
                "message_count": len(messages),
                "message_roles": roles,
                "messages_tokens": sum(
                    count_tokens(str(message.get("content") or ""), self._model_name())
                    for message in messages
                ),
                "history_tool_calls_replayed": sum(len(message.get("tool_calls") or []) for message in messages),
                "history_batched_turns": sum(1 for message in messages if len(message.get("tool_calls") or []) > 1),
                "message_layout": {
                    "name": MESSAGE_LAYOUT_NAME,
                    "workspace_split_out": bool((placement or {}).get("prefix_split")),
                    "memory_own_message": bool((placement or {}).get("memory_message")),
                    # 这一轮有没有把「被裁掉的历史」压成摘要带上。恒为 false
                    # 说明预算一直够用；非常频繁地为 true 说明预算给小了。
                    "omitted_digest_present": bool((placement or {}).get("omitted_digest")),
                    # 这一轮的上下文里有没有会话摘要。它和上面那个不一样:digest 说的是
                    # 「有几条放不下被丢了」,摘要说的是「最早那一段已经被换成概述了」。
                    "session_summary_present": bool((placement or {}).get("session_summary")),
                    "checkpoint_split_out": bool((placement or {}).get("checkpoint_message")),
                },
            }
        )
        return metadata

    def _render_sections_without_reduction(self, section_texts, selected_notes=None):
        selected_notes = selected_notes or []
        relevant_lines = ["Relevant memory:"]
        if selected_notes:
            relevant_lines.extend(f"- {note['text']}" for note in selected_notes)
        else:
            relevant_lines.append("- none")
        relevant_raw = "\n".join(relevant_lines)
        history = list(getattr(self.agent, "session", {}).get("history", []))
        history_raw = self._raw_history_text(history)
        # 关掉裁剪时每段都没有额度，budget 一律记 None。从前这里写的是 `len(text)`，
        # 那是**字符数**，会经 metadata 落进 `budget_tokens` 字段——单位是错的，
        # 而系统里只该有 token 一种单位。
        return {
            "prefix": SectionRender(raw=section_texts["prefix"], budget=None, rendered=section_texts["prefix"], details={}),
            "memory": SectionRender(raw=section_texts["memory"], budget=None, rendered=section_texts["memory"], details={}),
            "relevant_memory": SectionRender(
                raw=relevant_raw,
                budget=None,
                rendered=relevant_raw,
                details={
                    "selected_notes": [note["text"] for note in selected_notes],
                    "rendered_notes": [note["text"] for note in selected_notes],
                    "selected_count": len(selected_notes),
                    "rendered_count": len(selected_notes),
                    "note_budget": 0,
                },
            ),
            "history": SectionRender(
                raw=history_raw,
                budget=None,
                rendered=history_raw,
                details={"rendered_entries": []},
                # 关掉裁剪时每条历史都原样保留，lines 用不裁剪的渲染结果。
                kept_entries=[
                    {"recent": True, "item": item, "lines": self._render_history_item(item, len(history_raw) or 1)}
                    for item in history
                ],
            ),
            CURRENT_REQUEST_SECTION: SectionRender(
                raw=section_texts[CURRENT_REQUEST_SECTION],
                budget=0,
                rendered=section_texts[CURRENT_REQUEST_SECTION],
                details={},
            ),
        }

    def _compute_section_floors(self):
        """裁剪下限。

        从前它是各段额度的四分之一算出来的；额度取消之后没有可除的东西了，改成
        `SECTION_FLOORS` 里的显式常量。

        **调用方显式给了某段额度时，下限退回旧的"四分之一"规则**（取它和常量下限里
        小的那个）。不能直接拿额度本身当下限：那样 floor == budget，这一段一进裁剪
        循环就因为"已经到底"被跳过，一步都走不动——实测就是这样，一个显式给了
        120 额度的 relevant_memory 一个 token 都没被裁掉，`budget_reductions` 全空。
        """
        floors = dict(SECTION_FLOORS)
        for section, budget in self.section_budgets.items():
            if section in floors:
                floors[section] = min(floors[section], max(20, int(budget) // 4))
        floors.update(self._section_floor_overrides)
        return floors

    def _render_sections(self, section_texts, budgets, selected_notes=None):
        rendered = {}
        for section in SECTION_ORDER:
            budget = budgets.get(section)
            if section == CURRENT_REQUEST_SECTION:
                raw = section_texts[section]
                rendered[section] = SectionRender(raw=raw, budget=0, rendered=raw, details={})
            elif section == "relevant_memory":
                rendered[section] = self._render_relevant_memory(selected_notes or [], int(budget or 0))
            elif section == "history":
                rendered[section] = self._render_history_section(int(budget or 0))
            elif section in PROTECTED_SECTIONS:
                # prefix 原样进去，一个 token 不裁。budget 记 None 表示"没有额度"，
                # 不是 0——0 会在工件里被读成"额度是零"。
                raw = section_texts[section]
                rendered[section] = SectionRender(raw=raw, budget=None, rendered=raw, details={})
            else:
                raw = section_texts[section]
                rendered_text = _tail_clip(raw, int(budget), self._model_name()) if budget is not None else raw
                rendered[section] = SectionRender(raw=raw, budget=int(budget) if budget is not None else 0, rendered=rendered_text, details={})
        return rendered

    def _render_relevant_memory(self, selected_notes, budget):
        model = self._model_name()
        header = "Relevant memory:"
        note_texts = [str(note.get("text", "")) for note in selected_notes if str(note.get("text", "")).strip()]
        raw_lines = [header] + [f"- {text}" for text in note_texts]
        raw = "\n".join(raw_lines) if note_texts else "\n".join([header, "- none"])
        if not note_texts:
            rendered = raw
            return SectionRender(
                raw=raw,
                budget=budget,
                rendered=rendered,
                details={
                    "selected_notes": [],
                    "rendered_notes": [],
                    "selected_count": 0,
                    "rendered_count": 0,
                    "note_budget": 0,
                },
            )

        per_note_budget = self._per_note_budget(budget, len(note_texts), header)
        rendered_notes = []
        while True:
            # 让每条 note 平分这一段的预算，避免一条超长笔记把其他笔记都挤掉。
            rendered_notes = [_tail_clip(text, per_note_budget, model) for text in note_texts]
            rendered = "\n".join([header] + [f"- {text}" for text in rendered_notes])
            if count_tokens(rendered, model) <= budget or per_note_budget <= 1:
                break
            per_note_budget -= 1

        if count_tokens(rendered, model) > budget and budget > 0:
            rendered = _tail_clip(raw, budget, model)
            rendered_notes = [rendered]

        return SectionRender(
            raw=raw,
            budget=budget,
            rendered=rendered,
            details={
                "selected_notes": note_texts,
                "rendered_notes": rendered_notes,
                "selected_count": len(note_texts),
                "rendered_count": len(rendered_notes),
                "note_budget": per_note_budget,
            },
        )

    def _per_note_budget(self, budget, note_count, header):
        """每条笔记平分这一段的额度，单位是 token。

        overhead 也得按 token 算：`len(header)` 是字符数，拿它去减一个 token 额度
        会凭空多扣掉三四倍的量——实测这条把 80 个 token 的额度扣成个位数，三条
        笔记全被截成一个词。
        """
        if note_count <= 0:
            return 0
        model = self._model_name()
        # 每条笔记多出来的 "- " 前缀和换行，按一个 token 估。
        overhead = count_tokens(header, model) + 2 * note_count
        usable = max(0, budget - overhead)
        return max(1, usable // note_count)

    def _render_history_section(self, budget):
        model = self._model_name()
        history = list(getattr(self.agent, "session", {}).get("history", []))
        # 会话摘要覆盖的那一段整体退出 history:它已经被换成 summary_text,再渲染
        # 一遍就是把同一件事说两遍。切在任意位置都是安全的——`_history_messages()`
        # 本来就会给孤儿工具条目补出它的 assistant 轮,配对约束不会破。
        summary_covered = min(self._summary_covered(), len(history))
        summary_text = self._summary_text() if summary_covered else ""
        history = history[summary_covered:]
        summary_tokens = count_tokens(summary_text, model) if summary_text else 0
        # 摘要先占掉自己那份,剩下的才是条目可用的额度——否则这一段会悄悄超出 budget。
        entry_budget = max(0, int(budget) - summary_tokens)
        summary_details = {
            "summary_covered_entries": summary_covered,
            "summary_tokens": summary_tokens,
        }
        raw = self._raw_history_text(history)
        if not history:
            rendered = "\n\n".join(part for part in (summary_text, "Transcript:\n- empty") if part)
            return SectionRender(
                raw=raw,
                budget=budget,
                rendered=rendered,
                details={
                    "rendered_entries": [],
                    "older_entries_count": 0,
                    "collapsed_duplicate_reads": 0,
                    "reused_file_summary_count": 0,
                    "summarized_tool_count": 0,
                    "spilled_pointer_count": 0,
                    "preserved_error_count": 0,
                    "stale_read_count": 0,
                    "stale_pointer_count": 0,
                    **summary_details,
                },
                kept_entries=[],
            )

        # 优先保留最近的历史，因为下一步决策通常最依赖刚刚发生的工具结果。
        # block=1 等价于「每个工具轮把边界推一格」，也就是按块推进之前的老行为；
        # 它是一个消融开关，不是配置项——见 `RECENT_WINDOW_BLOCK` 的注释。
        block = RECENT_WINDOW_BLOCK
        if hasattr(self.agent, "feature_enabled") and not self.agent.feature_enabled("recent_window_block"):
            block = 1
        recent_window = _blocked_recent_window(history, RECENT_TOOL_TURNS, block=block)
        recent_start = _recent_start_by_tool_turns(history, recent_window, block=block)
        history_entries, history_details = self._compressed_history_entries(history, recent_start)
        rendered_entries = []
        # kept 与 rendered_entries 严格同步：任何一次接受 candidate_entries，
        # 都要把对应条目（带上它**最终采用的那份 lines**）插到 kept 最前面。
        # messages 组装完全依赖这个列表，两者一旦漂开，发出去的对话就和
        # prompt 文本说的不是一回事了。
        kept = []
        for entry in reversed(history_entries):
            recent = bool(entry.get("recent", False))
            candidate_lines = list(entry.get("lines", []))
            candidate_entries = candidate_lines + rendered_entries
            candidate_rendered = "\n".join(["Transcript:", *candidate_entries])
            if count_tokens(candidate_rendered, model) <= entry_budget:
                rendered_entries = candidate_entries
                kept.insert(0, {**entry, "lines": candidate_lines})
                continue
            if recent:
                available = entry_budget - count_tokens("Transcript:", model)
                if rendered_entries:
                    available -= sum(count_tokens(line, model) + 1 for line in rendered_entries)
                available = max(10, available - 1)
                candidate_lines = [_tail_clip(line, available, model) for line in candidate_lines]
                candidate_entries = candidate_lines + rendered_entries
                candidate_rendered = "\n".join(["Transcript:", *candidate_entries])
                if count_tokens(candidate_rendered, model) <= entry_budget:
                    rendered_entries = candidate_entries
                    # 最近窗口内的条目是「裁到剩下的空间正好放得下」,不是压扁——
                    # 它没有固定尺寸,所以不算进 squeezed。
                    kept.insert(0, {**entry, "lines": candidate_lines})
            elif self._reversible_squeeze_enabled():
                # 放不下就先**压扁**再谈丢弃:留 10 个 token 的残句,至少让模型知道
                # 这一轮发生过。这一步等价于 Claude Code 的 L4(可逆折叠)——history
                # 一个字节没动,压力退下去同一条还能按原形态展开——只是它一直藏在这个
                # 循环里、既没有名字也没有计数。给它一个形态名和一个计数:压扁和整条
                # 丢弃在工件上从前长得一模一样,而两者对模型的后果差得很远。
                smaller_lines = self._squeeze(candidate_lines)
                smaller_entries = smaller_lines + rendered_entries
                smaller_rendered = "\n".join(["Transcript:", *smaller_entries])
                if count_tokens(smaller_rendered, model) <= entry_budget:
                    rendered_entries = smaller_entries
                    kept.insert(0, {**entry, "lines": smaller_lines, "policy": POLICY_SQUEEZED})
                    history_details["squeezed_entry_count"] += 1
        rendered = "\n".join(["Transcript:", *rendered_entries])

        if count_tokens(rendered, model) > entry_budget and entry_budget > 0:
            # 兜底分支：整段按 token 尾裁，条目边界已经不成立了。
            # messages 侧没有对应的"半条消息"概念，所以这里退回全部条目——
            # 宁可让 messages 比 prompt 文本多带一点，也不能发出结构上残缺的对话。
            rendered = _tail_clip(raw, entry_budget, model)
            kept = [{**entry, "lines": list(entry.get("lines", []))} for entry in history_entries]

        # 哪几条没活下来。`kept` 里存的是 entry 的副本，靠底层 item 的身份来配对。
        kept_item_ids = {id(entry.get("item")) for entry in kept}
        dropped = [entry for entry in history_entries if id(entry.get("item")) not in kept_item_ids]
        omitted_digest = self._omitted_digest(dropped)
        if summary_text:
            rendered = summary_text + "\n\n" + rendered

        return SectionRender(
            raw=raw,
            budget=budget,
            rendered=rendered,
            details={
                "recent_window": recent_window,
                "recent_start": recent_start,
                "rendered_entries": rendered_entries,
                "omitted_entry_count": len(dropped),
                "omitted_digest": omitted_digest,
                "summary_text": summary_text,
                **summary_details,
                **history_details,
            },
            kept_entries=kept,
        )

    @staticmethod
    def _read_dedup_key(item):
        """read_file 结果去重的键：`(路径, 起始行, 结束行)`。

        **不能只按路径。** 一轮里读 `a.txt` 的 1–50 行和 51–100 行是两段互补内容，
        不是彼此的陈旧版本；只按路径去重会把前一段整条丢掉，模型下一轮看不到自己
        刚读过的前半段，而它以为看得到。单调用时几乎碰不到（分两轮读同一文件的
        不同区间很少见），一轮多调用把它放大成常见路径。

        默认值要和 `tools.py` 里 `read_file` 的 schema 保持一致（start=1, end=200），
        否则「显式写了默认值的调用」和「没写的调用」会被算成两个不同区间。
        """
        args = item.get("args", {}) or {}
        path = str(args.get("path", "")).strip()
        try:
            start = int(args.get("start", 1))
        except (TypeError, ValueError):
            start = 1
        try:
            end = int(args.get("end", 200))
        except (TypeError, ValueError):
            end = 200
        return (path, start, end)

    @classmethod
    def _newest_read_index(cls, history):
        """每个「路径+区间」最后一次 read_file 落在 history 的哪一条。

        旧实现只在「较早的那一段」里按路径去重，而且保留的是**最早**那次读。
        两个毛病：一是同一个文件被读过 N 次时，最近窗口里还是会有 N 份全文，
        白白吃掉 history 预算；二是保留最早那份等于把过时内容当现状喂给模型——
        实测有一次运行里 server.py 在改动前后各读过一次，留下来的会是改动前的。

        没有真正执行过的调用不参与去重：它们的内容是一句「未执行」的错误说明，
        让它顶掉同一区间那次真读到的内容会直接丢数据。
        """
        newest = {}
        for index, item in enumerate(history):
            if item.get("role") != "tool" or item.get("name") != "read_file":
                continue
            if item.get("executed", True) is False:
                continue
            newest[cls._read_dedup_key(item)] = index
        return newest

    @staticmethod
    def _written_paths(item):
        """这条历史记录让哪些文件的既有读取失效了。

        两个来源，合成一条路径而不是两条分支：

        - `wrote` 字段（runtime 在记录这一轮时填的，见 `CodingForMe._drain_written_paths()`）。
          有它就以它为准——**它是唯一能覆盖 `run_plan` 的来源**：一段计划在 history 里
          只有一条 `run_plan` 消息，内层那次 `patch_file` 的路径在 args 里根本不存在。
        - 没有这个字段时（老会话，或直接构造出来的 history）回落到 args 里的 `path`，
          并且要求这次调用**没有**返回错误——失败的 patch 一个字节都没改，让它去作废
          那次读会把模型重试时唯一能抄的正确文本抹掉，方向正好反了。
        """
        declared = item.get("wrote")
        if declared is not None:
            return {path_key(path) for path in declared if path_key(path)}
        if item.get("name") not in WRITE_TOOL_NAMES:
            return set()
        if str(item.get("content", "")).strip().startswith("error:"):
            return set()
        path = path_key((item.get("args") or {}).get("path", ""))
        return {path} if path else set()

    @classmethod
    def _stale_read_indexes(cls, history):
        """哪几条 read_file 记录在它之后被写过。返回 history 下标的集合。"""
        stale = set()
        reads_by_path = {}
        for index, item in enumerate(history):
            if item.get("role") != "tool":
                continue
            if item.get("name") == "read_file" and item.get("executed", True) is not False:
                path = path_key((item.get("args") or {}).get("path", ""))
                if path:
                    reads_by_path.setdefault(path, []).append(index)
            for path in cls._written_paths(item):
                # 只作废这次写**之前**的读；之后的读是新鲜的。
                stale.update(reads_by_path.pop(path, ()))
        return stale

    def _history_policies(self, history, recent_start):
        """给每条历史条目定一个呈现形态,**不做任何渲染**。

        拆开「决定怎么呈现」和「渲染成什么样」是这一层的全部意义:决策是纯函数
        (输入历史 + 窗口边界,输出一串形态名),因此可以整组枚举着测——而在此之前,
        这些分支只能靠构造出恰好触发它的历史来间接命中,谁也说不清一共有几种。
        """
        newest_read = self._newest_read_index(history)
        stale_reads = self._stale_read_indexes(history) if self._stale_read_enabled() else frozenset()
        plan = []
        for index, item in enumerate(history):
            recent = index >= recent_start
            plan.append((item, recent, self._policy_for(item, index, recent, newest_read, stale_reads)))
        return plan

    def _stale_read_enabled(self):
        """消融开关。关掉 → 过期读照旧全文呈现，也就是这个机制出现之前的行为。"""
        checker = getattr(self.agent, "feature_enabled", None)
        if checker is None:
            return True
        return bool(checker("stale_read_invalidation"))

    def _policy_for(self, item, index, recent, newest_read, stale_reads=frozenset()):
        """单条条目的形态。**书写顺序就是优先级**,每一条错位都有过实测后果。"""
        role = item.get("role")
        # 只带 tool_calls、没有说明文字的 assistant 记录:文本侧等于不存在,结构侧
        # 照常参与——messages 组装要靠它把这一轮的调用和结果配成对。
        if role != "tool" and not str(item.get("content", "")).strip():
            return POLICY_STRUCTURAL
        # 同一个「文件+区间」的过期读:不论在不在窗口内,只留最后那次全文。未执行的
        # 调用不在 newest_read 里,所以不会被误判成过期读丢掉——模型要靠它知道这个
        # 调用需要重发。
        if (
            role == "tool"
            and item.get("name") == "read_file"
            and item.get("executed", True) is not False
            and newest_read.get(self._read_dedup_key(item)) != index
        ):
            return POLICY_DROPPED
        if recent:
            # 过期读**只在窗口内**改形态。窗口外那几种形态本来就不再是全文
            # （摘要 / 指针 / cleared），而记忆层的 file_summaries 已经按 sha256
            # 作废过一次，所以那半边不需要再来一遍；只在这里改，还能让
            # `older_entries_count` 这套老口径的计数一个都不动。
            if index in stale_reads:
                return POLICY_STALE_READ
            return POLICY_FULL
        if role != "tool":
            return POLICY_CLIPPED
        policy = self._old_tool_policy(item)
        # 窗口外唯一还携带「可取回但已陈旧」内容的形态。指针指向的落盘文件是**读的
        # 那一刻**写下的快照,原文件后来被改了它不会跟着变,而指针原文正是在教模型去
        # 读它——不标出来,取回的就是改动之前那一版,且看起来像是刚读的。
        if policy == POLICY_POINTER and index in stale_reads:
            return POLICY_STALE_POINTER
        return policy

    def _old_tool_policy(self, item):
        """掉出最近窗口的那条工具结果换成什么形态。五选一,顺序即优先级。"""
        if item.get("executed", True) is False:
            return POLICY_NOT_EXECUTED
        content = str(item.get("content", ""))
        marker = find_spill_marker(content)
        # 落盘过的结果不能换成文件摘要:那句摘要是从**落盘后的预览段**生成的,等于用
        # 一句描述开头 14,804 token 的话代表整个 140,000 token 的文件,而且不再有任何
        # 线索说明全文还在盘上。踩过的坑:这一支原本无条件排在最前,于是 read_file
        # ——唯一会产生大段文件内容的工具——永远轮不到 pointer;一次 live 压力探针实测
        # 5 次落盘、15 条掉出窗口、指针 0 条,模型随后开始重复读同一个文件、耗光步数。
        if (
            item.get("name") == "read_file"
            and not marker
            and self._reusable_file_summary(str((item.get("args") or {}).get("path", "")).strip())
        ):
            return POLICY_FILE_SUMMARY
        # 错误要排在 pointer 和 shell_head 之前:错误串是一整句话,被 `lines[:3]` 切开
        # 就读不成句;指针恒在末尾,同样会被切掉。
        if content.strip().startswith("error:"):
            return POLICY_ERROR_KEPT
        if marker:
            return POLICY_POINTER
        if item.get("name") == "run_shell":
            return POLICY_SHELL_HEAD
        return POLICY_CLEARED

    def _squeeze(self, lines):
        """压扁:把已经定好的那份呈现再裁到 SQUEEZED_TOKENS。保留循环和投影层共用。"""
        return [_tail_clip(line, SQUEEZED_TOKENS, self._model_name()) for line in lines]

    def _render_policy(self, item, policy):
        """按形态渲染一条条目。**只认形态名**,不再重新判断这条是什么。"""
        if policy in (POLICY_STRUCTURAL, POLICY_DROPPED):
            return []
        if policy == POLICY_SQUEEZED:
            # 压扁是叠在别的形态之上的:先按这条本来的形态渲染,再压。
            base = self._old_tool_policy(item) if item.get("role") == "tool" else POLICY_CLIPPED
            return self._squeeze(self._render_policy(item, base))
        if policy == POLICY_FULL:
            # 不做第二道裁剪:内容在 `run_tool()` 存进 history 的那一刻已经被
            # `tool_output_limit()` 限过一次,这里再砍一刀既重复、又与预算脱钩
            # (旧值是写死的 430)。放不下由按 budget 走的保留循环负责。
            return self._render_history_item(item, tool_output_limit(self.total_budget))
        if policy == POLICY_CLIPPED:
            return self._render_history_item(item, CLIPPED_TOKENS)
        if policy == POLICY_STALE_READ:
            # 调用签名照留（模型要知道自己读过这个文件），内容换成显式标记。
            # 和 `CLEARED_RESULT_MARKER` 是同一条原则：保留调用记录、把内容换成
            # 一句说得出原因的占位，而不是让它静默变成别的样子。
            return [f"{self._render_history_item(item, 60)[0]} -> {STALE_READ_MARKER}"]
        if policy == POLICY_FILE_SUMMARY:
            path = str((item.get("args") or {}).get("path", "")).strip()
            return [f"{path} -> {self._reusable_file_summary(path)}"]
        signature = self._render_history_item(item, 60)[0]
        content = str(item.get("content", ""))
        if policy == POLICY_NOT_EXECUTED:
            # 老条目一律只留 `[tool:name] {args}`,内容被丢掉——对未执行的调用来说,
            # 那会让它和一次正常执行长得一模一样。这里显式保留「没执行」这个事实。
            return [f"{signature} -> not executed"]
        if policy == POLICY_ERROR_KEPT:
            return [f"{signature} -> {_tail_clip(content.strip(), ERROR_KEEP_TOKENS, self._model_name())}"]
        if policy == POLICY_POINTER:
            return [f"{signature} -> {find_spill_marker(content)}"]
        if policy == POLICY_STALE_POINTER:
            # 指针**照留**:全文还在盘上,那是取回的唯一线索,丢掉它就等于把这条记录
            # 降级成 cleared。只在后面接一句说清盘上那份已经是旧版。
            return [f"{signature} -> {find_spill_marker(content)} {STALE_SPILL_MARKER}"]
        if policy == POLICY_SHELL_HEAD:
            command = str((item.get("args") or {}).get("command", "")).strip() or "shell"
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            return [f"{command} -> " + (" | ".join(lines[:3]) if lines else "(empty)")]
        return [f"{signature} -> {CLEARED_RESULT_MARKER}"]

    # 形态 → details 里的计数键。**老口径一个不改**:`older_entries_count` 只数窗口外的
    # 工具条目,窗口外的对话文本不算在内(它从前就没算,改了会让跨版本的工件对不上)。
    POLICY_COUNTERS = {
        POLICY_DROPPED: ("collapsed_duplicate_reads",),
        # 只有这一条不带 `older_entries_count`：它按定义就在窗口内。
        POLICY_STALE_READ: ("stale_read_count",),
        POLICY_FILE_SUMMARY: ("older_entries_count", "reused_file_summary_count"),
        POLICY_NOT_EXECUTED: ("older_entries_count", "summarized_tool_count"),
        POLICY_ERROR_KEPT: ("older_entries_count", "summarized_tool_count", "preserved_error_count"),
        POLICY_POINTER: ("older_entries_count", "summarized_tool_count", "spilled_pointer_count"),
        # 仍然计进 `spilled_pointer_count`:它就是一条指针,只是多带了一句标记。
        # `stale_pointer_count` 是它的子集,回答的是另一个问题——「有几条指针指向的
        # 盘上快照已经过期」。混成一个数,这两件事就分不出来了。
        POLICY_STALE_POINTER: (
            "older_entries_count",
            "summarized_tool_count",
            "spilled_pointer_count",
            "stale_pointer_count",
        ),
        POLICY_SHELL_HEAD: ("older_entries_count", "summarized_tool_count"),
        POLICY_CLEARED: ("older_entries_count", "summarized_tool_count"),
    }

    def _compressed_history_entries(self, history, recent_start):
        """投影:历史 + 窗口边界 → 一串带形态的待渲染条目。

        `details` 里那几个计数是这套东西唯一的观测通道:「指针接上了」和「内容静默
        消失了」在别的字段上长得一模一样,而那正是要验收的那件事。
        """
        details = {
            "older_entries_count": 0,
            "collapsed_duplicate_reads": 0,
            "reused_file_summary_count": 0,
            "summarized_tool_count": 0,
            "spilled_pointer_count": 0,
            "preserved_error_count": 0,
            # 被后续写操作作废、因此没有全文呈现的 read_file 条数。零值也照写——
            # 「没有过期读」和「这个机制没生效」在别的字段上分不出来。
            "stale_read_count": 0,
            # 同上,但发生在最近窗口**之外**:指针指向的落盘快照已经不是文件现状。
            "stale_pointer_count": 0,
            # 预算不够时被压扁(而不是整条丢弃)的条目数,由保留循环回填。
            "squeezed_entry_count": 0,
        }
        entries = []
        for item, recent, policy in self._history_policies(history, recent_start):
            for key in self.POLICY_COUNTERS.get(policy, ()):
                details[key] += 1
            if policy == POLICY_DROPPED:
                continue
            entries.append(
                {
                    "recent": recent,
                    "lines": self._render_policy(item, policy),
                    "item": item,
                    "policy": policy,
                }
            )
        return entries, details

    def _reusable_file_summary(self, path):
        memory = getattr(self.agent, "memory", None)
        if memory is None or not hasattr(memory, "to_dict"):
            return ""
        snapshot = memory.to_dict()
        summary = snapshot.get("file_summaries", {}).get(str(path), {})
        if not summary:
            return ""
        return str(summary.get("summary", "")).strip()

    def _summarize_old_tool_item(self, item):
        """窗口外的单条工具结果长什么样。形态判定与渲染都在投影层,这里只是入口。"""
        return self._render_policy(item, self._old_tool_policy(item))[0]

    def _raw_history_text(self, history):
        if not history:
            return "Transcript:\n- empty"
        lines = []
        for item in history:
            if item["role"] == "tool":
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(str(item["content"]))
            else:
                # 见 runtime.history_text()：只带 tool_calls 的 assistant 记录
                # 在纯文本视图里没有内容可写。
                if not str(item.get("content", "")).strip():
                    continue
                lines.append(f"[{item['role']}] {item['content']}")
        return "\n".join(["Transcript:", *lines])

    # 这份摘要最多占多少 **token**（`_tail_clip` 按 token 裁，这里必须同单位——
    # 换单位时这一处漏掉了，注释还写着「字符」而代码已经按 token 判，等于把上限
    # 悄悄放宽了约 3 倍）。摘要要放进「运行状态」那条 user 消息，它每轮都在变，
    # 撑大它会直接压低前缀缓存命中率。原先 400 字符，内容是英文短句加路径、
    # 实测约 3 字符/token，折算 400 ÷ 3 ≈ 135。
    OMITTED_DIGEST_BUDGET = 135

    @classmethod
    def _omitted_digest(cls, dropped_entries):
        """把「没能活过预算的那几轮」压成一行确定性摘要。

        为什么需要:原来的做法是**静默丢弃**——条目放不下就跳过,模型完全看不出
        自己前面做过什么。它于是会重读已经读过的文件、重发已经发过的补丁。
        Anthropic《Effective context engineering》里 compaction 的要点正是这个:
        接近上限时**要总结再丢**,不能直接截断。

        刻意不调模型来生成摘要:那会在组上下文的路径里插进一次约 17 秒的往返,
        而这里真正要保住的信息(做过哪些调用、碰过哪些文件)完全可以从条目本身
        确定性地算出来。
        """
        if not dropped_entries:
            return ""
        tool_counts = {}
        paths = []
        message_turns = 0
        for entry in dropped_entries:
            item = entry.get("item") or {}
            if item.get("role") != "tool":
                if str(item.get("content", "")).strip():
                    message_turns += 1
                continue
            name = str(item.get("name", "")) or "?"
            tool_counts[name] = tool_counts.get(name, 0) + 1
            path = str((item.get("args") or {}).get("path", "")).strip()
            if path and path not in paths:
                paths.append(path)
        parts = [f"{len(dropped_entries)} earlier transcript entries dropped to fit the budget"]
        if tool_counts:
            done = ", ".join(f"{name} x{count}" for name, count in sorted(tool_counts.items()))
            parts.append(f"already run: {done}")
        if paths:
            parts.append("files touched: " + ", ".join(paths))
        if message_turns:
            parts.append(f"{message_turns} message turn(s)")
        return _tail_clip("Omitted context: " + "; ".join(parts) + ".", cls.OMITTED_DIGEST_BUDGET)

    def _render_history_item(self, item, line_limit):
        if item["role"] == "tool":
            prefix = f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}"
            content = _tail_clip(item["content"], max(10, line_limit), self._model_name())
            return [prefix, content]
        return [f"[{item['role']}] {_tail_clip(item['content'], line_limit, self._model_name())}"]

    # 某个工具结果没能活过裁剪时，占位用的内容。
    #
    # 为什么必须有占位而不能直接不发：OpenAI 的协议要求 assistant 消息里每一个
    # tool_call 都恰好对应一条 role:"tool" 消息，缺一条整个请求会被后端拒掉。
    # 而反过来把那个 tool_call 从 assistant 消息里删掉也不行——那等于告诉模型
    # "你没调过这个"，它下一轮可能重发一遍已经做过的写操作。
    DROPPED_RESULT_NOTE = "(result dropped to fit the context budget; call it again if you still need it)"

    @staticmethod
    def _entry_body(entry):
        """把一条 history 条目的**最终渲染行**还原成消息正文。

        `_render_history_item()` 对工具条目产出 `[prefix, content]` 两行、对
        user/assistant 产出一行 `[role] text`。到了 messages 里，角色和工具名都由
        结构字段承载，这些文本前缀就是纯冗余，所以在这里剥掉。
        取 `lines[-1]` 而不是预存正文，是为了拿到**裁剪之后**的那一份。
        """
        lines = list(entry.get("lines", []))
        if not lines:
            return ""
        item = entry.get("item", {}) or {}
        if item.get("role") == "tool":
            # 被摘要成一行的老条目没有独立正文，那一行本身就是全部信息。
            return lines[-1]
        text = lines[0]
        prefix = f"[{item.get('role', '')}] "
        return text[len(prefix):] if text.startswith(prefix) else text

    @staticmethod
    def _wire_tool_call(call):
        """history 里记的调用 → OpenAI 线上格式的 tool_call。"""
        return {
            "id": str(call.get("id", "")),
            "type": "function",
            "function": {
                "name": str(call.get("name", "")),
                "arguments": json.dumps(call.get("args", {}) or {}, sort_keys=True),
            },
        }

    def _history_messages(self, kept_entries):
        """把活下来的历史条目还原成真正的对话轮次。

        为什么存在：
        在此之前，整段历史被 `_raw_history_text()` 渲染成一串 `[tool:read_file] {...}`
        文本塞进单条 user message。那样做丢掉了**回合边界**——模型"一轮发了两个
        调用"和"两轮各发一个"渲染出来一模一样，于是它无从知道自己批量过。实测把
        历史摆成标准结构后，开放式请求下的多调用率从 15% 升到 44%
        （Fisher 精确检验 p=0.0031，见 docs/architecture/multi-tool-call-blueprint.md §5.4）。
        另外两个收益同样重要：文件内容不再和用户指令混在同一条消息里（注入面），
        以及工具结果和它的调用有了 `tool_call_id` 配对。

        两条硬约束，破坏任一条后端就会拒请求或者模型会被误导：
        1. **每个 tool_call 必须恰好配一条 role:"tool" 消息。** 结果被裁掉时补
           占位，而不是把调用一起删掉。
        2. **孤儿工具条目要补出它的 assistant 轮。** 老会话没有 `tool_calls`
           字段，它那一轮的 assistant 记录也可能被裁掉；两种情况都在这里补齐。
        """
        by_call_id = {}
        for index, entry in enumerate(kept_entries):
            item = entry.get("item", {}) or {}
            if item.get("role") == "tool" and item.get("call_id"):
                by_call_id[str(item["call_id"])] = index

        messages = []
        consumed = set()
        for index, entry in enumerate(kept_entries):
            if index in consumed:
                continue
            item = entry.get("item", {}) or {}
            role = item.get("role")
            body = self._entry_body(entry)

            if role == "assistant":
                calls = list(item.get("tool_calls") or [])
                if not calls:
                    if body.strip():
                        messages.append({"role": "assistant", "content": body})
                    continue
                # 说明文字取 item 原文而不是 body：body 可能被裁到只剩几十字符，
                # 而 assistant 的说明本来就短，裁它省不下多少、却会让下一轮看不懂。
                messages.append(
                    {
                        "role": "assistant",
                        "content": str(item.get("content") or ""),
                        "tool_calls": [self._wire_tool_call(call) for call in calls],
                    }
                )
                for call in calls:
                    call_id = str(call.get("id", ""))
                    result_index = by_call_id.get(call_id)
                    if result_index is None:
                        messages.append({"role": "tool", "tool_call_id": call_id, "content": self.DROPPED_RESULT_NOTE})
                        continue
                    consumed.add(result_index)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": self._entry_body(kept_entries[result_index]),
                        }
                    )
                continue

            if role == "tool":
                call_id = str(item.get("call_id") or "") or f"call_orphan_{index}"
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            self._wire_tool_call({"id": call_id, "name": item.get("name", ""), "args": item.get("args", {})})
                        ],
                    }
                )
                messages.append({"role": "tool", "tool_call_id": call_id, "content": body})
                continue

            messages.append({"role": "user", "content": body})
        return messages

    def _split_prefix(self, prefix_text):
        """把 prefix 按「多久变一次」切成三段。

        - 规则 + 工具清单：同一个 harness 下逐字节相同 → 留在 `system`。
        - 仓库快照（从 `Workspace:` 那行起）：一次运行内恒定、跨任务不同 → 第一条 `user`。
        - resume checkpoint（从 `Task checkpoint:` 那行起）：**每执行一次工具就重
          渲染** → 和 working memory 一起压到最后。

        切不开时（预算把它裁没了、或模板改过）该段就并回前一段，最坏情况退化成
        「整段都在 `system`」，也就是改动前的摆法：不会出错，只是省不下缓存。
        """
        checkpoint = ""
        index = prefix_text.find(PREFIX_CHECKPOINT_MARKER)
        if index >= 0:
            prefix_text, checkpoint = prefix_text[:index], prefix_text[index:].strip()
        index = prefix_text.find(PREFIX_WORKSPACE_MARKER)
        if index < 0:
            return prefix_text.strip(), "", checkpoint
        return prefix_text[:index].strip(), prefix_text[index:].strip(), checkpoint

    def _assemble_messages(self, rendered, user_message):
        """把各 section 摆成标准 messages 数组。

        **摆位是按前缀缓存设计的，不是把 `SECTION_ORDER` 直接翻译成消息。**
        这个后端（以及多数 OpenAI-compatible 后端）做的是自动前缀缓存：认序列化
        之后的最长公共 token 前缀，不认我们发的 cache key。于是规则变成「越靠前
        的内容越要恒定」——一段每轮都变的文本放得越靠前，被它作废掉的后续内容
        就越多。

        四类位置：
        - `system` = prefix 里规则 + 工具清单那一段。跨任务、跨轮次逐字节相同，
          工具 schema 那一大块（占 prefix 的大头）因此能一直命中。
        - 第一条 `user` = 仓库快照 + resume checkpoint。一次运行内恒定、跨任务不同。
        - 中间 = 真正的对话轮次（`_history_messages()`），只在末尾追加。
        - 倒数第二条 `user` = resume checkpoint + working memory + relevant memory。
          这三样都**每执行一次工具就变**，所以必须压到最后，否则它们会把前面所有
          内容一起挤出缓存。最后一条 `user` 则是当前请求。

        为什么不是「合成一条 system」：那是上一版的做法，实测把前缀缓存命中率从
        78.8% 打到 32.7%——快照随任务变、memory 每轮变，两者都在 system 里，公共
        前缀于是止步于规则文本末尾，工具定义那一大段每次都要重算。改成现在这个
        摆法后，真实后端上跨任务 77.7%、同运行内 83.1%，两项都优于改动前。实测
        见 docs/architecture/multi-tool-call-blueprint.md §5.7。

        memory 单独占一条 user 消息而不是拼进当前请求，是为了不让「工具读回来的
        文件内容」（它经 working memory 的摘要泄进上下文）和用户亲口说的话共用同
        一条消息——见 tests/test_messages_protocol.py 里那条注入面的用例。
        """
        prefix_stable, workspace_text, checkpoint_text = self._split_prefix(rendered["prefix"].rendered)
        placement = {"prefix_split": bool(workspace_text), "checkpoint_message": bool(checkpoint_text)}
        messages = [{"role": "system", "content": prefix_stable}]
        if workspace_text:
            messages.append({"role": "user", "content": workspace_text})
        messages.extend(self._history_messages(rendered["history"].kept_entries or []))
        # checkpoint 排在 memory 之前，和改动前它们在 system 里的相对顺序一致。
        state_text = "\n\n".join(
            part
            for part in (
                checkpoint_text,
                rendered["memory"].rendered,
                rendered["relevant_memory"].rendered,
                # 被裁掉的历史的摘要挂在最后：它和 memory 一样属于「每轮都在变的
                # 运行状态」，放这条消息里不会污染前面跨轮次逐字节相同的部分。
                # **不能挂进 history 的消息序列**——那串消息受 assistant.tool_calls
                # 与 role:"tool" 必须一一配对的约束，插一条自由文本会打坏配对。
                # 会话摘要和 digest 一样属于「每轮都可能变的运行状态」,所以挂在这条
                # 消息上而不是 history 的消息序列里(那串消息受 assistant.tool_calls
                # 与 role:"tool" 一一配对的约束,插自由文本会被后端拒掉)。
                # 它排在 digest 之前:摘要覆盖的是更早的一段。
                str(rendered["history"].details.get("summary_text", "")),
                str(rendered["history"].details.get("omitted_digest", "")),
            )
            if part
        ).strip()
        placement["memory_message"] = bool(state_text)
        placement["omitted_digest"] = bool(rendered["history"].details.get("omitted_digest"))
        placement["session_summary"] = bool(rendered["history"].details.get("summary_text"))
        if state_text:
            messages.append({"role": "user", "content": state_text})
        # current_request 段的文本带着 "Current user request:" 这个标签，那是
        # 压平成一段文本时用来标界的；现在界由消息边界划出，标签只剩噪声。
        # 但**必须走 rendered 而不是原始 user_message**，否则超预算兜底的
        # 首尾保留裁剪会被绕过。
        request_text = rendered[CURRENT_REQUEST_SECTION].rendered
        label = "Current user request:\n"
        if request_text.startswith(label):
            request_text = request_text[len(label):]
        messages.append({"role": "user", "content": request_text})
        return messages, placement

    def _assemble_prompt(self, rendered):
        # 顺序是刻意设计的：稳定规则放前面，最新请求放最后。
        return "\n\n".join(
            [
                rendered["prefix"].rendered,
                rendered["memory"].rendered,
                rendered["relevant_memory"].rendered,
                rendered["history"].rendered,
                rendered[CURRENT_REQUEST_SECTION].rendered,
            ]
        ).strip()

    def _metadata(
        self,
        prompt,
        rendered,
        budgets,
        reduction_log,
        selected_notes,
        user_message,
        section_texts,
        current_request_truncated=False,
        current_request_dropped_tokens=0,
        prompt_tokens=None,
        floor_exhausted=False,
    ):
        # 不裁剪的那条路径（context_reduction 关掉时）不走闸门，也就没算过 token，
        # 这里补算一次，好让两条路径产出同一组字段。
        if prompt_tokens is None:
            prompt_tokens = self._measure(prompt)
        section_metadata = {}
        for section in SECTION_ORDER[:-1]:
            section_metadata[section] = {
                "raw_tokens": rendered[section].raw_tokens,
                # 受保护的段没有额度，记 None。写 0 会被读成"额度是零"，而它的
                # 含义正相反：不设上限。
                "budget_tokens": _budget_or_none(budgets, section),
                "rendered_tokens": rendered[section].rendered_tokens,
            }
            # 「这一段丢了几条、丢的是什么」必须能从工件里读出来：静默丢弃和
            # 丢了但留了摘要，在别的字段上长得一模一样。
            if "omitted_entry_count" in rendered[section].details:
                section_metadata[section]["omitted_entry_count"] = int(
                    rendered[section].details["omitted_entry_count"]
                )
                section_metadata[section]["omitted_digest"] = str(
                    rendered[section].details.get("omitted_digest", "")
                )
        section_metadata[CURRENT_REQUEST_SECTION] = {
            "raw_tokens": count_tokens(section_texts[CURRENT_REQUEST_SECTION], self._model_name()),
            "budget_tokens": None,
            "rendered_tokens": rendered[CURRENT_REQUEST_SECTION].rendered_tokens,
        }
        return {

            # 闸门判的是 token，字符只是给人看的。两个都留：字符能和源码里的
            # section budget 对上，token 能和后端回报的 input_tokens 对上。
            "prompt_tokens": int(prompt_tokens),
            "prompt_budget_tokens": self.total_budget,
            "tool_schema_tokens": self._tool_schema_tokens(),
            # 「可裁的都裁到底了、仍然放不下」必须能单独读出来。它和「刚好装下」
            # 在 prompt_chars / budget_reductions 上都看不出区别，而两者的含义相反：
            # 前者说明这个预算对当前负载偏小，该调预算，不是该继续裁。
            "budget_floor_exhausted": bool(
                prompt_tokens > self.total_budget and floor_exhausted
            ),
            # prefix 不参与裁剪是不变量，把它落进工件，免得日后有人改了顺序
            # 而所有断言照常通过。
            "protected_sections": list(PROTECTED_SECTIONS),
            "prompt_over_budget": int(prompt_tokens) > self.total_budget,
            "section_order": list(SECTION_ORDER),
            "section_budgets": {
                section: (
                    None
                    if section == CURRENT_REQUEST_SECTION
                    else _budget_or_none(budgets, section)
                )
                for section in SECTION_ORDER
            },
            "sections": section_metadata,
            "budget_reductions": reduction_log,
            "reduction_order": list(self.reduction_order),
            "relevant_memory": {
                "limit": RELEVANT_MEMORY_LIMIT,
                "selected_count": len(selected_notes),
                "selected_notes": [note["text"] for note in selected_notes],
                "selected_sources": [str(note.get("source", "")).strip() for note in selected_notes],
                "selected_kinds": [str(note.get("kind", "episodic")).strip() or "episodic" for note in selected_notes],
                "selected_durable_count": sum(
                    1 for note in selected_notes if (str(note.get("kind", "episodic")).strip() or "episodic") == "durable"
                ),
                "raw_tokens": rendered["relevant_memory"].raw_tokens,
                "rendered_tokens": rendered["relevant_memory"].rendered_tokens,
                "rendered_notes": list(rendered["relevant_memory"].details.get("rendered_notes", [])),
                "rendered_count": int(rendered["relevant_memory"].details.get("rendered_count", 0)),
            },
            "history": {
                "raw_tokens": rendered["history"].raw_tokens,
                "rendered_tokens": rendered["history"].rendered_tokens,
                "older_entries_count": int(rendered["history"].details.get("older_entries_count", 0)),
                "collapsed_duplicate_reads": int(rendered["history"].details.get("collapsed_duplicate_reads", 0)),
                "reused_file_summary_count": int(rendered["history"].details.get("reused_file_summary_count", 0)),
                "summarized_tool_count": int(rendered["history"].details.get("summarized_tool_count", 0)),
                # 最近窗口这一轮实际保留了几条工具结果。按块推进之后它在
                # [RECENT_TOOL_TURNS, +RECENT_WINDOW_BLOCK) 之间浮动,不再恒等于
                # 常量——恒等于常量的那个版本每轮都把边界往前推一格,前缀缓存
                # 每轮作废一次,而工件上完全看不出来。
                "recent_tool_window": int(rendered["history"].details.get("recent_window", 0)),
                # 被清掉的结果里,有多少还留着落盘指针(可恢复)、有多少是刻意
                # 保留全文的错误。没有这两个数,「指针接上了」和「内容静默消失了」
                # 在工件上长得一模一样。
                "spilled_pointer_count": int(rendered["history"].details.get("spilled_pointer_count", 0)),
                "preserved_error_count": int(rendered["history"].details.get("preserved_error_count", 0)),
                # 被后续写操作作废的 read_file 条数。没有它,「这轮没有过期读」和
                # 「作废机制根本没跑」在工件上长得一模一样——和上面两个是同一个理由。
                "stale_read_count": int(rendered["history"].details.get("stale_read_count", 0)),
                # 窗口**外**那一半:指针还在(全文仍可从盘上取回),但盘上那份已经
                # 是改动之前的快照。和上面那个分开数——一个说明模型手里的全文过期
                # 了,一个说明它照着指针去取回来的东西会过期,应对不同。
                "stale_pointer_count": int(rendered["history"].details.get("stale_pointer_count", 0)),
                # 预算不够时被**压扁**(压到 SQUEEZED_TOKENS 的残句)而不是整条丢弃的
                # 条目数。压扁和丢弃从前在工件上长得一模一样,而两者对模型的后果差得
                # 很远:一个是「这一轮发生过、内容看不清」,一个是「这一轮不存在」。
                "squeezed_entry_count": int(rendered["history"].details.get("squeezed_entry_count", 0)),
                # 最近窗口边界这一轮落在哪条。它每变一次,消息数组从那里往后全部改写、
                # 前缀缓存作废——按块推进就是为了让这个数少变几次,所以它必须可观测。
                "recent_start": int(rendered["history"].details.get("recent_start", 0)),
            },
            "current_request": {
                "text": user_message,
                "raw_tokens": count_tokens(user_message, self._model_name()),
                "rendered_tokens": count_tokens(user_message, self._model_name()),
                "section_tokens": rendered[CURRENT_REQUEST_SECTION].rendered_tokens,
                "truncated": bool(current_request_truncated),
                "dropped_tokens": int(current_request_dropped_tokens),
            },
        }
