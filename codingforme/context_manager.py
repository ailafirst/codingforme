"""Prompt 组装与上下文预算控制。

这个模块负责决定：每一轮到底把多少 prefix、memory、相关笔记、历史
以及当前用户请求送进模型。
"""

from __future__ import annotations

import json
from dataclasses import dataclass


DEFAULT_TOTAL_BUDGET = 12000
DEFAULT_SECTION_BUDGETS = {
    "prefix": 3600,
    "memory": 1600,
    "relevant_memory": 1200,
    "history": 5200,
}
DEFAULT_SECTION_FLOORS = {
    "prefix": 1200,
    "memory": 400,
    "relevant_memory": 300,
    "history": 1500,
}
# 当 prompt 超预算时，会优先压缩这些 section。
DEFAULT_REDUCTION_ORDER = ("relevant_memory", "history", "memory", "prefix")
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


def _tail_clip(text, limit):
    text = str(text)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


RECENT_TOOL_TURNS = 6


def _recent_start_by_tool_turns(history, tool_turns):
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
    # 对话很多的历史会被整段判成「最近」，全部按 900 字符渲染，预算一次吃光。
    return max(start, len(history) - 2 * tool_turns)


def _head_tail_clip(text, limit):
    """保留首尾、省略中间的裁剪——仅用于 current_request 兜底分支。

    为什么不用 `_tail_clip`：用户请求的关键约束经常同时出现在开头(要做什么)
    和结尾(边界条件、"不要动 xxx" 之类的补充说明)，只砍尾巴容易连着关键信息
    一起丢掉。这里退而求其次，两头都留一点，只丢中间。
    """
    text = str(text)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    marker = "...[中间已省略]..."
    if limit <= len(marker):
        return text[:limit]
    remaining = limit - len(marker)
    head_len = remaining // 2
    tail_len = remaining - head_len
    return text[:head_len] + marker + text[len(text) - tail_len :]


@dataclass
class SectionRender:
    raw: str
    budget: int
    rendered: str
    details: dict | None = None
    # 裁剪之后**活下来的条目**，每条带回它的来源 history item。
    #
    # 为什么不放进 details：details 会整体进 metadata → trace → 落盘，而这里挂着
    # 原始 item（含完整文件内容）。放进去等于把每轮 prompt 的全文再抄一份到
    # trace 里。这条通道只服务于 `_assemble_messages()`，是进程内的。
    kept_entries: list | None = None

    @property
    def raw_chars(self):
        return len(self.raw)

    @property
    def rendered_chars(self):
        return len(self.rendered)


class ContextManager:
    def __init__(
        self,
        agent,
        total_budget=DEFAULT_TOTAL_BUDGET,
        section_budgets=None,
        section_floors=None,
        reduction_order=None,
    ):
        self.agent = agent
        self.total_budget = int(total_budget)
        self.section_budgets = dict(DEFAULT_SECTION_BUDGETS)
        if section_budgets:
            self.section_budgets.update({str(key): int(value) for key, value in section_budgets.items()})
        self._section_floor_overrides = {str(key): int(value) for key, value in (section_floors or {}).items()}
        self.section_floors = self._compute_section_floors()
        self.reduction_order = tuple(reduction_order or DEFAULT_REDUCTION_ORDER)

    def build(self, user_message):
        """`build_all()` 的兼容入口，只返回 `(prompt, metadata)`。

        真正发给模型的是 messages 数组，见 `build_all()`。这里保留纯文本形态，
        是因为预算裁剪、`prompt_chars` 这些度量、以及 trace 里的 prompt 元数据
        全部按字符数算——它仍然是**同一次组装**的产物，不是另一条并行链路。
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

        budgets = dict(self.section_budgets)
        rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)
        prompt = self._assemble_prompt(rendered)
        reduction_log = []

        # 如果 prompt 超预算，就按固定顺序不断压缩。
        # 这里的顺序体现了平台偏好：
        # 先牺牲 relevant_memory，再牺牲 history，然后才动 memory 和 prefix。
        # 最新用户请求永远不裁剪，因为那是本轮最重要的输入。
        while len(prompt) > self.total_budget:
            overflow = len(prompt) - self.total_budget
            reduced = False
            for section in self.reduction_order:
                floor = int(self.section_floors.get(section, 0))
                current_budget = int(budgets.get(section, 0))
                if current_budget <= floor:
                    continue
                new_budget = max(floor, current_budget - overflow)
                if new_budget >= current_budget:
                    continue
                reduction_log.append(
                    {
                        "section": section,
                        "before_chars": current_budget,
                        "after_chars": new_budget,
                        "overflow_chars": overflow,
                    }
                )
                budgets[section] = new_budget
                rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)
                prompt = self._assemble_prompt(rendered)
                reduced = True
                break
            if not reduced:
                break

        # 兜底分支：只有当 current_request 自己就超过 total_budget、且其它
        # section 已经全部压到 floor 仍不够时才会走到这里。正常情况下
        # current_request 完全不受上面这个压缩循环影响——这里不是把它纳入
        # 常规裁剪顺序，而是处理"怎么裁都裁不出空间"的极端情况：与其让一个
        # 超预算的 prompt 原样发给模型、指望后端报错兜底，不如在这一步就把
        # 决定权收回来，用可审计的方式砍它，并显式告诉模型这一情况。
        current_request_truncated = False
        current_request_dropped_chars = 0
        if len(prompt) > self.total_budget:
            current_render = rendered[CURRENT_REQUEST_SECTION]
            header = "Current user request:\n"
            note = (
                "\n\n[注意：用户原始请求过长，已保留首尾并省略中间部分；"
                "如果任务因此显得不完整，请提示用户缩短请求或分步描述。]"
            )
            other_chars = len(prompt) - len(current_render.rendered)
            available = self.total_budget - other_chars - len(note) - len(header)
            available_for_message = max(0, available)
            if available_for_message < len(user_message):
                current_request_truncated = True
                clipped_message = _head_tail_clip(user_message, available_for_message)
                current_request_dropped_chars = len(user_message) - len(clipped_message)
                rendered[CURRENT_REQUEST_SECTION] = SectionRender(
                    raw=current_render.raw,
                    budget=0,
                    rendered=header + clipped_message + note,
                    details={"truncated": True},
                )
                prompt = self._assemble_prompt(rendered)

        metadata = self._metadata(
            prompt=prompt,
            rendered=rendered,
            budgets=budgets,
            reduction_log=reduction_log,
            selected_notes=selected_notes,
            user_message=user_message,
            section_texts=section_texts,
            current_request_truncated=current_request_truncated,
            current_request_dropped_chars=current_request_dropped_chars,
        )
        messages, placement = self._assemble_messages(rendered, user_message)
        self._record_message_metadata(metadata, messages, placement)
        return messages, prompt, metadata

    @staticmethod
    def _record_message_metadata(metadata, messages, placement=None):
        """把 messages 的形状记进 metadata，供 trace/report 观测。

        只记形状不记内容：条数、角色分布、字符总量、以及**批量轮次**——
        `assistant` 消息里 `tool_calls` 多于一个的有几轮。最后这个数字是
        T2-1 想要影响的量，没有它就只能靠事后翻 trace 数。
        `messages_chars` 与 `prompt_chars` 会有出入（结构字段替代了文本前缀），
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
                "messages_chars": sum(len(str(message.get("content") or "")) for message in messages),
                "history_tool_calls_replayed": sum(len(message.get("tool_calls") or []) for message in messages),
                "history_batched_turns": sum(1 for message in messages if len(message.get("tool_calls") or []) > 1),
                "message_layout": {
                    "name": MESSAGE_LAYOUT_NAME,
                    "workspace_split_out": bool((placement or {}).get("prefix_split")),
                    "memory_own_message": bool((placement or {}).get("memory_message")),
                    # 这一轮有没有把「被裁掉的历史」压成摘要带上。恒为 false
                    # 说明预算一直够用；非常频繁地为 true 说明预算给小了。
                    "omitted_digest_present": bool((placement or {}).get("omitted_digest")),
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
        return {
            "prefix": SectionRender(raw=section_texts["prefix"], budget=len(section_texts["prefix"]), rendered=section_texts["prefix"], details={}),
            "memory": SectionRender(raw=section_texts["memory"], budget=len(section_texts["memory"]), rendered=section_texts["memory"], details={}),
            "relevant_memory": SectionRender(
                raw=relevant_raw,
                budget=len(relevant_raw),
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
                budget=len(history_raw),
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
        floors = {
            section: max(20, int(budget) // 4)
            for section, budget in self.section_budgets.items()
        }
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
            else:
                raw = section_texts[section]
                rendered_text = _tail_clip(raw, int(budget)) if budget is not None else raw
                rendered[section] = SectionRender(raw=raw, budget=int(budget) if budget is not None else 0, rendered=rendered_text, details={})
        return rendered

    def _render_relevant_memory(self, selected_notes, budget):
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
            rendered_notes = [_tail_clip(text, per_note_budget) for text in note_texts]
            rendered = "\n".join([header] + [f"- {text}" for text in rendered_notes])
            if len(rendered) <= budget or per_note_budget <= 1:
                break
            per_note_budget -= 1

        if len(rendered) > budget and budget > 0:
            rendered = _tail_clip(raw, budget)
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
        if note_count <= 0:
            return 0
        overhead = len(header) + 3 * note_count
        usable = max(0, budget - overhead)
        return max(1, usable // note_count)

    def _render_history_section(self, budget):
        history = list(getattr(self.agent, "session", {}).get("history", []))
        raw = self._raw_history_text(history)
        if not history:
            rendered = "Transcript:\n- empty"
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
                },
                kept_entries=[],
            )

        # 优先保留最近的历史，因为下一步决策通常最依赖刚刚发生的工具结果。
        recent_window = RECENT_TOOL_TURNS
        recent_start = _recent_start_by_tool_turns(history, recent_window)
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
            if len(candidate_rendered) <= budget:
                rendered_entries = candidate_entries
                kept.insert(0, {**entry, "lines": candidate_lines})
                continue
            if recent:
                available = budget - len("Transcript:")
                if rendered_entries:
                    available -= sum(len(line) + 1 for line in rendered_entries)
                available = max(20, available - 1)
                candidate_lines = [_tail_clip(line, available) for line in candidate_lines]
                candidate_entries = candidate_lines + rendered_entries
                candidate_rendered = "\n".join(["Transcript:", *candidate_entries])
                if len(candidate_rendered) <= budget:
                    rendered_entries = candidate_entries
                    kept.insert(0, {**entry, "lines": candidate_lines})
            else:
                smaller_lines = [_tail_clip(line, 20) for line in candidate_lines]
                smaller_entries = smaller_lines + rendered_entries
                smaller_rendered = "\n".join(["Transcript:", *smaller_entries])
                if len(smaller_rendered) <= budget:
                    rendered_entries = smaller_entries
                    kept.insert(0, {**entry, "lines": smaller_lines})
        rendered = "\n".join(["Transcript:", *rendered_entries])

        if len(rendered) > budget and budget > 0:
            # 兜底分支：整段按字符尾裁，条目边界已经不成立了。
            # messages 侧没有对应的"半条消息"概念，所以这里退回全部条目——
            # 宁可让 messages 比 prompt 文本多带一点，也不能发出结构上残缺的对话。
            rendered = _tail_clip(raw, budget)
            kept = [{**entry, "lines": list(entry.get("lines", []))} for entry in history_entries]

        # 哪几条没活下来。`kept` 里存的是 entry 的副本，靠底层 item 的身份来配对。
        kept_item_ids = {id(entry.get("item")) for entry in kept}
        dropped = [entry for entry in history_entries if id(entry.get("item")) not in kept_item_ids]
        omitted_digest = self._omitted_digest(dropped)

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

    def _compressed_history_entries(self, history, recent_start):
        entries = []
        newest_read = self._newest_read_index(history)
        details = {
            "older_entries_count": 0,
            "collapsed_duplicate_reads": 0,
            "reused_file_summary_count": 0,
            "summarized_tool_count": 0,
        }

        for index, item in enumerate(history):
            recent = index >= recent_start

            # 只带 tool_calls、没有说明文字的 assistant 记录在文本里没内容可渲染，
            # 但**条目本身必须留下**——messages 组装要靠它把这一轮的调用和结果
            # 配成对。给它一份空的 lines：文本侧等于不存在，结构侧照常参与。
            if item.get("role") != "tool" and not str(item.get("content", "")).strip():
                entries.append({"recent": recent, "lines": [], "item": item})
                continue

            # 同一个「文件+区间」的过期读：不论在窗口内还是窗口外，只留最后那次全文。
            # 未执行的调用不在 newest_read 里，所以这里不会把它误判成过期读丢掉——
            # 它必须留在 history 里，模型要靠它知道这个调用需要重发。
            if (
                item.get("role") == "tool"
                and item.get("name") == "read_file"
                and item.get("executed", True) is not False
            ):
                if newest_read.get(self._read_dedup_key(item)) != index:
                    details["collapsed_duplicate_reads"] += 1
                    continue

            if recent:
                line_limit = 900
                entries.append(
                    {
                        "recent": True,
                        "lines": self._render_history_item(item, line_limit),
                        "item": item,
                    }
                )
                continue

            # 未执行的调用不能被替换成文件摘要：那会把「这个调用没跑」变成
            # 「这个文件的内容是 ...」，模型于是以为它已经读到了。
            if (
                item["role"] == "tool"
                and item["name"] == "read_file"
                and item.get("executed", True) is not False
            ):
                path = str(item["args"].get("path", "")).strip()
                summary = self._reusable_file_summary(path)
                if summary:
                    entries.append({"recent": False, "lines": [f"{path} -> {summary}"], "item": item})
                    details["older_entries_count"] += 1
                    details["reused_file_summary_count"] += 1
                    continue

            if item["role"] == "tool":
                summary_line = self._summarize_old_tool_item(item)
                entries.append({"recent": False, "lines": [summary_line], "item": item})
                details["older_entries_count"] += 1
                details["summarized_tool_count"] += 1
                continue

            # 非最近轮的 assistant 文本。60 字符曾经够用，是因为这里只会出现
            # runtime 自己写回的重试提示；现在模型的工具轮说明也进 history 了，
            # 60 字符会把它砍成一句没头没尾的话，等于修了又白修。
            entries.append({"recent": False, "lines": self._render_history_item(item, 220), "item": item})

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
        # 老条目一律只留 `[tool:name] {args}`，内容被丢掉——对未执行的调用来说，
        # 那会让它和一次正常执行长得一模一样。这里显式保留「没执行」这个事实。
        if item.get("executed", True) is False:
            return f"{self._render_history_item(item, 60)[0]} -> not executed"
        if item["name"] == "run_shell":
            command = str(item["args"].get("command", "")).strip() or "shell"
            lines = [line.strip() for line in str(item.get("content", "")).splitlines() if line.strip()]
            summary = " | ".join(lines[:3]) if lines else "(empty)"
            return f"{command} -> {summary}"
        return self._render_history_item(item, 60)[0]

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

    # 一条被裁掉的历史最多在摘要里占多少字符。摘要本身要放进「运行状态」那条
    # user 消息，它每轮都在变，撑大它会直接压低前缀缓存命中率。
    OMITTED_DIGEST_BUDGET = 400

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
            content = _tail_clip(item["content"], max(20, line_limit))
            return [prefix, content]
        return [f"[{item['role']}] {_tail_clip(item['content'], line_limit)}"]

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
                str(rendered["history"].details.get("omitted_digest", "")),
            )
            if part
        ).strip()
        placement["memory_message"] = bool(state_text)
        placement["omitted_digest"] = bool(rendered["history"].details.get("omitted_digest"))
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
        current_request_dropped_chars=0,
    ):
        section_metadata = {}
        for section in SECTION_ORDER[:-1]:
            section_metadata[section] = {
                "raw_chars": rendered[section].raw_chars,
                "budget_chars": int(budgets.get(section, 0)),
                "rendered_chars": rendered[section].rendered_chars,
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
            "raw_chars": len(section_texts[CURRENT_REQUEST_SECTION]),
            "budget_chars": None,
            "rendered_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
        }
        return {
            "prompt_chars": len(prompt),
            "prompt_budget_chars": self.total_budget,
            "prompt_over_budget": len(prompt) > self.total_budget,
            "section_order": list(SECTION_ORDER),
            "section_budgets": {
                section: (None if section == CURRENT_REQUEST_SECTION else int(budgets.get(section, 0)))
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
                "raw_chars": rendered["relevant_memory"].raw_chars,
                "rendered_chars": rendered["relevant_memory"].rendered_chars,
                "rendered_notes": list(rendered["relevant_memory"].details.get("rendered_notes", [])),
                "rendered_count": int(rendered["relevant_memory"].details.get("rendered_count", 0)),
            },
            "history": {
                "raw_chars": rendered["history"].raw_chars,
                "rendered_chars": rendered["history"].rendered_chars,
                "older_entries_count": int(rendered["history"].details.get("older_entries_count", 0)),
                "collapsed_duplicate_reads": int(rendered["history"].details.get("collapsed_duplicate_reads", 0)),
                "reused_file_summary_count": int(rendered["history"].details.get("reused_file_summary_count", 0)),
                "summarized_tool_count": int(rendered["history"].details.get("summarized_tool_count", 0)),
            },
            "current_request": {
                "text": user_message,
                "raw_chars": len(user_message),
                "rendered_chars": len(user_message),
                "section_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
                "truncated": bool(current_request_truncated),
                "dropped_chars": int(current_request_dropped_chars),
            },
        }
