"""多步 agent 运行时使用的轻量工作记忆。

session history 负责保存完整事件流；这个模块只保存更小的一层工作集：
当前任务摘要、最近接触的文件、文件短摘要，以及少量跨轮笔记。
这样下一轮 prompt 还能接上上一轮，但不会被整段历史塞满。
"""

import hashlib
import math
import shutil
from datetime import datetime
import re
from pathlib import Path

from .workspace import clip, now


# 记忆里两处长度上限，单位是 **token**——和上下文预算同一种。
#
# 它们限的是最终要进 prompt 的文本（笔记会被召回渲染进 relevant_memory，
# task_summary 每轮都在 working memory 里），所以必须和预算用同一把尺子量。
# 数值由原先的 500 / 300 字符按**笔记这种内容**实测出的比值折算，不是统一折半：
# 笔记和 task_summary 都由模型用用户的语言写，实测中文 1.31 字符/token（中文一个字
# 往往就是一个 token），500 ÷ 1.31 ≈ 380、300 ÷ 1.31 ≈ 230。统一折半会给 250/150，
# 把中文笔记砍掉三分之一——而这两处限的恰恰是最可能是中文的那类文本。
NOTE_TOKENS = 380
TASK_SUMMARY_TOKENS = 230

# 一条记忆的「描述」上限。描述是**专门用来判相关性**的那一行：召回只对它打分，
# 不再对正文打分（阶段二）。80 token 在中文约 105 字、英文约 245 字，和 Claude Code
# 索引单行 ~150 字符是同一个量级——它那条上限的作用同样是「一行，别把正文写进索引」。
DESCRIPTION_TOKENS = 80

WORKING_FILE_LIMIT = 8
EPISODIC_NOTE_LIMIT = 12
FILE_SUMMARY_LIMIT = 6
# 仪表盘上最多列几个记忆名。v2 之后这个列表是**每条记忆一个名字**（不再是 4 个
# 主题），不设上限的话它会随库线性增长——而真正要给模型看的是带描述的索引。
DURABLE_TOPIC_DISPLAY_LIMIT = 8

# 长期记忆的四种类型，取自 Claude Code 的类型系统。分类的实际作用不是归档，
# 是**召回时的保守程度**：user / project 描述的是「用户长期在做什么」，不该被
# 一个字面重叠的问题勾出来。目前我们只存类型、按类型做覆盖判定，还没按类型
# 调排序——那要等有足够多的真实记忆能量出误召回率。
MEMORY_TYPES = ("user", "feedback", "project", "reference")
DEFAULT_MEMORY_TYPE = "project"

# 旧的 4 个封闭主题 → 新类型。它现在只是**迁移映射表**，不再是可写入的主题集合。
LEGACY_TOPIC_TYPES = {
    "project-conventions": "project",
    "key-decisions": "project",
    "dependency-facts": "reference",
    "user-preferences": "user",
}

# 记忆库的磁盘格式版本，写在 `.codingforme/memory/.schema` 里。
# v1 = 4 个封闭主题、一个主题文件里堆多条笔记；v2 = 一事一文件 + frontmatter。
DURABLE_SCHEMA_VERSION = 2

# `name` 的合法字符集，和 Claude Code 的 `/^[a-z0-9_-]+$/` 一致。不合规的一律
# 转成 kebab-case；纯中文标题产生不了 ASCII 词，回落到内容哈希（见 `slugify_memory_name`）。
_NAME_PATTERN = re.compile(r"^[a-z0-9_-]+$")

DURABLE_TOPIC_DEFAULTS = {
    "project-conventions": {
        "title": "Project Conventions",
        "summary": "Stable repository conventions.",
        "tags": ["convention"],
    },
    "key-decisions": {
        "title": "Key Decisions",
        "summary": "Long-lived decisions and rationale anchors.",
        "tags": ["decision"],
    },
    "dependency-facts": {
        "title": "Dependency Facts",
        "summary": "Stable dependency and environment facts.",
        "tags": ["dependency"],
    },
    "user-preferences": {
        "title": "User Preferences",
        "summary": "Stable user preferences.",
        "tags": ["preference"],
    },
}


def default_memory_state():
    # 用一个小而结构化的状态，而不是一大段自由文本摘要。
    return {
        "working": {
            "task_summary": "",
            "recent_files": [],
        },
        "episodic_notes": [],
        "file_summaries": {},
        "task": "",
        "files": [],
        "notes": [],
        "next_note_index": 0,
    }


class DurableMemoryStore:
    """长期记忆库：磁盘上的那份，跨会话存活。

    两种磁盘格式同时存在，靠文件第一行区分：

    - **v1（旧）**：4 个封闭主题，一个 `topics/<topic>.md` 里堆多条笔记。
    - **v2（新）**：一事一文件，开头三行元信息（`name` / `description` / `metadata`）。

    v1 **只读不写**：`memory_types` 关掉时走的是原样保留的旧写入路径（消融要求
    「关掉严格退回机制出现之前的行为」），开着时第一次写入会把 v1 整库迁移成 v2。
    两种格式长期并存是不允许的——并存会让索引上限的计算分叉。
    """

    def __init__(self, root, cjk_recall=True, description_scoring=True, memory_types=True):
        self.root = Path(root)
        self.index_path = self.root / "MEMORY.md"
        self.topics_dir = self.root / "topics"
        self.schema_path = self.root / ".schema"
        self.cjk_recall = bool(cjk_recall)
        self.description_scoring = bool(description_scoring)
        self.memory_types = bool(memory_types)

    def topic_slugs(self):
        return [topic["topic"] for topic in self.load_index()]

    # ------------------------------------------------------------------
    # v2 读写
    # ------------------------------------------------------------------

    def schema_version(self):
        try:
            return int(str(self.schema_path.read_text(encoding="utf-8")).strip() or 1)
        except Exception:
            return 1

    def _write_schema(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema_path.write_text(f"{DURABLE_SCHEMA_VERSION}\n", encoding="utf-8")

    def _entry_paths(self):
        if not self.topics_dir.exists():
            return []
        return sorted(self.topics_dir.glob("*.md"))

    @staticmethod
    def resolve_type(kind):
        """把「调用方给的那个名字」归一成四种类型之一。

        调用方可能给类型名（`project`），也可能给旧的主题 slug（`project-conventions`）
        ——后者来自 `DURABLE_MEMORY_LINE_PATTERNS`，那份表还按主题命名。两种都认，
        认不出的一律落到默认类型而不是报错：提升路径上抛异常会把一次已经答完的
        运行整个打断。
        """
        kind = str(kind or "").strip().lower()
        if kind in MEMORY_TYPES:
            return kind
        return LEGACY_TOPIC_TYPES.get(kind, DEFAULT_MEMORY_TYPE)

    @staticmethod
    def _parse_entry(path):
        """读一个 v2 记忆文件；不是 v2 格式就返回 None（交给旧读法）。

        自己解析而不是引 yaml：运行时依赖只有 litellm 这一条是仓库硬约束，而
        frontmatter 在这里只有两层、值全是标量，正则足够。
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            return None
        lines = raw.splitlines()
        if not lines or lines[0].strip() != "---":
            return None
        meta = {}
        head = {}
        body_start = len(lines)
        in_metadata = False
        for index in range(1, len(lines)):
            line = lines[index]
            if line.strip() == "---":
                body_start = index + 1
                break
            match = re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
            if not match:
                continue
            indent, key, value = match.group(1), match.group(2), match.group(3).strip()
            if not indent:
                in_metadata = key == "metadata"
                if not in_metadata:
                    head[key] = value
            elif in_metadata:
                meta[key] = value
        body = "\n".join(lines[body_start:]).strip()
        text = body
        links = []
        link_match = re.search(r"^Links:\s*(.+)$", body, re.M)
        if link_match:
            links = re.findall(r"\[\[([^\]]+)\]\]", link_match.group(1))
            text = body[: link_match.start()].strip()
        name = head.get("name", "") or path.stem
        return {
            "name": name,
            "title": name,
            "description": head.get("description", ""),
            "text": text,
            "type": DurableMemoryStore.resolve_type(meta.get("type", DEFAULT_MEMORY_TYPE)),
            "tags": [tag.strip() for tag in meta.get("tags", "").split(",") if tag.strip()],
            "created_at": meta.get("created_at", "") or now(),
            "updated_at": meta.get("updated_at", "") or meta.get("created_at", "") or now(),
            "origin_session_id": meta.get("origin_session_id", ""),
            "origin_run_seq": meta.get("origin_run_seq", ""),
            "stale": str(meta.get("stale", "")).strip().lower() in {"1", "true", "yes"},
            "incomplete": str(meta.get("incomplete", "")).strip().lower() in {"1", "true", "yes"},
            "links": links,
            "source": name,
            "kind": "durable",
        }

    def _write_entry(self, entry):
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "---",
            f"name: {entry['name']}",
            f"description: {entry.get('description', '')}",
            "metadata:",
            f"  type: {entry.get('type', DEFAULT_MEMORY_TYPE)}",
            f"  created_at: {entry.get('created_at') or now()}",
            f"  updated_at: {entry.get('updated_at') or now()}",
        ]
        if entry.get("tags"):
            lines.append(f"  tags: {', '.join(entry['tags'])}")
        if entry.get("origin_session_id"):
            lines.append(f"  origin_session_id: {entry['origin_session_id']}")
        if str(entry.get("origin_run_seq", "")).strip():
            lines.append(f"  origin_run_seq: {entry['origin_run_seq']}")
        if entry.get("stale"):
            lines.append("  stale: true")
        if entry.get("incomplete"):
            lines.append("  incomplete: true")
        lines.extend(["---", "", entry.get("text", "")])
        if entry.get("links"):
            lines.extend(["", "Links: " + ", ".join(f"[[{name}]]" for name in entry["links"])])
        (self.topics_dir / f"{entry['name']}.md").write_text(
            "\n".join(lines).rstrip() + "\n", encoding="utf-8"
        )

    def _modern_entries(self):
        entries = []
        for path in self._entry_paths():
            entry = self._parse_entry(path)
            if entry is not None:
                entries.append(entry)
        return entries

    def _legacy_entries(self):
        entries = []
        for path in self._entry_paths():
            if self._parse_entry(path) is not None:
                continue
            slug = path.stem
            title = DURABLE_TOPIC_DEFAULTS.get(slug, {}).get("title", slug)
            for note in self.load_topic_notes(slug):
                note = dict(note)
                note["name"] = slug
                note["title"] = title
                note["description"] = derive_description(note.get("text", ""))
                note["type"] = LEGACY_TOPIC_TYPES.get(slug, DEFAULT_MEMORY_TYPE)
                note["updated_at"] = note.get("created_at", "")
                note["links"] = []
                entries.append(note)
        return entries

    def entries(self):
        """库里现有的全部记忆，两种磁盘格式合成一份列表。"""
        if not self.memory_types:
            return self._legacy_entries()
        return self._modern_entries() + self._legacy_entries()

    def _unique_name(self, text, taken):
        # 名字从**描述**切出来，不是从正文：正文截前 48 个字符会把「这条记忆顺带
        # 提到的东西」也带进名字，而名字是参与召回打分的（`scoring_tokens`）——
        # 那等于从后门把正文关键词放回了打分对象，描述层就白做了。
        base = slugify_memory_name(derive_description(text))
        if base not in taken:
            return base
        for suffix in range(2, 100):
            candidate = f"{base}-{suffix}"
            if candidate not in taken:
                return candidate
        return slugify_memory_name(text, fallback_seed=now())

    def load_index(self):
        if not self.index_path.exists():
            return []
        lines = self.index_path.read_text(encoding="utf-8").splitlines()
        topics = []
        current = None
        for raw in lines:
            line = raw.strip()
            match = re.match(r"- \[([^\]]+)\]\([^)]+\):\s*(.+)", line)
            if match:
                current = {
                    "topic": match.group(1).strip(),
                    "title": match.group(2).strip(),
                    "summary": "",
                    "tags": [],
                }
                topics.append(current)
                continue
            if current is None:
                continue
            summary_match = re.match(r"- summary:\s*(.+)", line)
            if summary_match:
                current["summary"] = summary_match.group(1).strip()
                continue
            tags_match = re.match(r"- tags:\s*(.+)", line)
            if tags_match:
                current["tags"] = [tag.strip() for tag in tags_match.group(1).split(",") if tag.strip()]
        return topics

    def load_topic_notes(self, topic):
        path = self.topics_dir / f"{topic}.md"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        notes = []
        capture = False
        updated_at = ""
        tags = []
        for raw in lines:
            line = raw.strip()
            if line.startswith("- tags:"):
                tags = [tag.strip() for tag in line.split(":", 1)[1].split(",") if tag.strip()]
            elif line.startswith("- updated_at:"):
                updated_at = line.split(":", 1)[1].strip()
            elif line == "## Notes":
                capture = True
            elif capture and line.startswith("- "):
                notes.append(
                    {
                        "text": line[2:].strip(),
                        "tags": tags,
                        "source": topic,
                        "created_at": updated_at or now(),
                        "kind": "durable",
                    }
                )
        return notes

    @staticmethod
    def _subject_key(text, cjk=True):
        """一条事实的「主语」，用来判断新事实该覆盖哪条旧事实。

        它和召回共用 `_tokenize`，所以中文分词一开，中文事实的覆盖语义就从
        「永不生效」变成「生效」——这是修复不是回归。英文侧的键必须逐字节不变，
        由 `tests/test_memory_phases.py` 锁住。
        """
        text = str(text).strip()
        patterns = (
            r"^(.+?)\s+is\s+.+$",
            r"^(.+?)\s+are\s+.+$",
            r"^(.+?)\s+uses?\s+.+$",
            r"^(.+?)\s+should\s+.+$",
            r"^(.+?)是.+$",
            r"^(.+?)使用.+$",
        )
        for pattern in patterns:
            match = re.match(pattern, text, re.I)
            if match:
                subject = " ".join(sorted(_tokenize(match.group(1), cjk)))
                return subject or None
        return None

    def scoring_tokens(self, entry):
        """一条记忆拿去和提问比对的那组词。

        `description_scoring` 开着时**只看名字、描述和标签,不看正文**——这就是
        「描述即索引」那条设计的落点:正文改长改短不该影响它被不被召回。关掉时
        退回旧行为(对正文全文打分),这样消融测的才是同一个机制。
        """
        tags = {str(tag).lower() for tag in entry.get("tags", [])}
        if self.description_scoring:
            return (
                _tokenize(entry.get("description", ""), self.cjk_recall)
                | _tokenize(entry.get("name", ""), self.cjk_recall)
                | tags
            )
        return (
            _tokenize(entry.get("text", ""), self.cjk_recall)
            | _tokenize(entry.get("title", ""), self.cjk_recall)
            | tags
        )

    def retrieval_candidates(self, query, limit=3):
        query_tokens = _tokenize(query, self.cjk_recall)
        ranked = []
        for entry in self.entries():
            note_tags = {str(tag).lower() for tag in entry.get("tags", [])}
            note_tokens = self.scoring_tokens(entry)
            exact_tag_match = int(bool(query_tokens & note_tags))
            overlap = _overlap_score(query_tokens, note_tokens, self.description_scoring)
            if exact_tag_match == 0 and overlap == 0:
                continue
            recency = _parse_timestamp(entry.get("updated_at") or entry.get("created_at"))
            ranked.append(((exact_tag_match, overlap, recency), entry))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [note for _, note in ranked[:limit]]

    def index_view(self, limit_tokens, model=None):
        """索引:一行一条记忆,只有名字和那一行描述,不含正文。

        它每轮都进上下文(这就是 Claude Code 遥测里的 `listed` 那一层),所以上限
        从预算派生、不写死。**截断必须报出来**:静默截断在工件上和「库里就这么多」
        长得一模一样,而两者的应对相反。
        """
        from .models import count_tokens

        entries = sorted(
            self.entries(),
            key=lambda entry: _parse_timestamp(entry.get("updated_at") or entry.get("created_at")),
            reverse=True,
        )
        limit_tokens = max(int(limit_tokens or 0), 0)
        header = "Durable memory index:"
        kept = []
        for entry in entries:
            description = str(entry.get("description", "")).strip() or derive_description(entry.get("text", ""))
            # `[stale]` 进索引行、`[incomplete]` 不进，这是刻意的不对称：
            # stale 罕见，而且直接改变模型该怎么用这条记忆（「它点名的文件可能已经
            # 不在了」）；incomplete 在写入器打开之前**每一条都会命中**（确定性生成的
            # 描述不可能带「为什么」），一个 100% 命中的标记不传递任何信息，却要在
            # 每一行上花掉约 13 个 token——而索引是有预算的。它照样写进文件元数据，
            # 并在下面的统计里给出条数，这样「有多少条缺理由」仍然量得到。
            marker = " [stale]" if entry.get("stale") else ""
            kept.append(f"- {entry.get('name', '')} ({entry.get('type', DEFAULT_MEMORY_TYPE)}): {description}{marker}")
        incomplete = sum(1 for entry in entries if entry.get("incomplete"))
        truncated = 0
        while kept and count_tokens("\n".join([header] + kept), model) > limit_tokens:
            kept.pop()
            truncated += 1
        text = "\n".join([header] + kept) if kept else ""
        return {
            "text": text,
            "entries": len(kept),
            "total_entries": len(entries),
            "truncated_entries": truncated,
            "incomplete_entries": incomplete,
            "tokens": count_tokens(text, model) if text else 0,
        }

    def _write_index(self, topics):
        self.root.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        lines = ["# Durable Memory Index", ""]
        for topic in topics:
            lines.append(f"- [{topic['topic']}](topics/{topic['topic']}.md): {topic['title']}")
            lines.append(f"  - summary: {topic['summary']}")
            lines.append(f"  - tags: {', '.join(topic['tags'])}")
        self.index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _write_topic(self, topic, notes):
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        meta = DURABLE_TOPIC_DEFAULTS[topic]
        lines = [
            f"# {meta['title']}",
            "",
            f"- topic: {topic}",
            f"- summary: {meta['summary']}",
            f"- tags: {', '.join(meta['tags'])}",
            f"- updated_at: {now()}",
            "",
            "## Notes",
        ]
        for note in notes:
            lines.append(f"- {note}")
        (self.topics_dir / f"{topic}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _write_modern_index(self, entries):
        self.root.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        ordered = sorted(
            entries,
            key=lambda entry: (
                -_parse_timestamp(entry.get("updated_at") or entry.get("created_at")),
                entry.get("name", ""),
            ),
        )
        lines = ["# Durable Memory Index", ""]
        for entry in ordered:
            lines.append(
                f"- [{entry['name']}](topics/{entry['name']}.md): "
                f"{entry.get('description', '') or derive_description(entry.get('text', ''))}"
            )
            lines.append(f"  - type: {entry.get('type', DEFAULT_MEMORY_TYPE)}")
            lines.append(f"  - updated_at: {entry.get('updated_at') or entry.get('created_at') or now()}")
        self.index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def migrate_legacy(self):
        """把 v1 的主题文件整库迁移成 v2 的一事一文件。返回迁移了几条。

        **先整目录备份再改写**,失败就整目录还原——这是方案里给「迁移丢数据」
        这条风险留的处置口。
        """
        if not self.memory_types:
            return 0
        legacy_paths = [path for path in self._entry_paths() if self._parse_entry(path) is None]
        if not legacy_paths:
            if self.root.exists():
                self._write_schema()
            return 0
        backup = self.root.parent / f"memory.bak-{now().replace(':', '').replace('-', '')}"
        try:
            shutil.copytree(self.root, backup, dirs_exist_ok=True)
        except Exception:
            backup = None
        try:
            entries = {entry["name"]: entry for entry in self._modern_entries()}
            migrated = 0
            for path in legacy_paths:
                slug = path.stem
                mtype = LEGACY_TOPIC_TYPES.get(slug, DEFAULT_MEMORY_TYPE)
                for note in self.load_topic_notes(slug):
                    text = str(note.get("text", "")).strip()
                    if not text or any(entry["text"] == text for entry in entries.values()):
                        continue
                    name = self._unique_name(text, entries)
                    entries[name] = {
                        "name": name,
                        "description": derive_description(text),
                        "text": text,
                        "type": mtype,
                        "tags": list(note.get("tags", [])),
                        "created_at": note.get("created_at") or now(),
                        "updated_at": note.get("created_at") or now(),
                        "origin_session_id": "",
                        "origin_run_seq": "",
                        "links": [],
                        "incomplete": _needs_rationale(mtype, text),
                    }
                    migrated += 1
            for path in legacy_paths:
                path.unlink()
            for entry in entries.values():
                self._write_entry(entry)
            self._write_modern_index(entries.values())
            self._write_schema()
            return migrated
        except Exception:
            if backup is not None and backup.exists():
                shutil.rmtree(self.root, ignore_errors=True)
                shutil.copytree(backup, self.root, dirs_exist_ok=True)
            raise

    def consolidate(self, workspace_root=None):
        """整理:去重、标记过期、重建索引。全部确定性,不调模型。

        「过期」是**标记不是删除**——记忆里点名的文件不见了,可能是改名也可能是
        真删了,前者删掉记忆就把线索一起丢了。这和 Claude Code 那条「记忆说 X 存在
        和 X 现在存在不是一回事」是同一件事,只是我们只做得到确定性的那一半。
        """
        if not self.memory_types:
            return {"removed_duplicates": 0, "marked_stale": [], "entries": len(self.entries()), "ran": False}
        entries = sorted(
            self._modern_entries(),
            key=lambda entry: _parse_timestamp(entry.get("updated_at") or entry.get("created_at")),
        )
        keep = {}
        removed = 0
        for entry in entries:
            subject = self._subject_key(entry.get("text", ""), self.cjk_recall)
            key = (entry.get("type"), subject) if subject else ("text", entry.get("text", ""))
            previous = keep.get(key)
            if previous is not None:
                removed += 1
                path = self.topics_dir / f"{previous['name']}.md"
                if path.exists():
                    path.unlink()
            keep[key] = entry
        marked_stale = []
        for entry in keep.values():
            stale = _names_a_missing_file(entry.get("text", ""), workspace_root)
            if stale and not entry.get("stale"):
                entry["stale"] = True
                marked_stale.append(entry["name"])
            self._write_entry(entry)
        self._write_modern_index(keep.values())
        self._write_schema()
        return {
            "removed_duplicates": removed,
            "marked_stale": marked_stale,
            "entries": len(keep),
            "ran": True,
        }

    def promote(self, promotions, origin=None):
        if not promotions:
            return [], []
        if self.memory_types:
            return self._promote_typed(promotions, origin or {})
        return self._promote_legacy(promotions)

    def _promote_typed(self, promotions, origin):
        self.migrate_legacy()
        entries = {entry["name"]: entry for entry in self._modern_entries()}
        results = []
        superseded = []
        for kind, note_text in promotions:
            note_text = str(note_text).strip()
            if not note_text:
                continue
            mtype = self.resolve_type(kind)
            if any(entry["text"] == note_text for entry in entries.values()):
                continue
            subject = self._subject_key(note_text, self.cjk_recall)
            replaced = None
            if subject:
                for entry in entries.values():
                    if entry.get("type") != mtype:
                        continue
                    if self._subject_key(entry.get("text", ""), self.cjk_recall) == subject:
                        replaced = entry
                        break
            if replaced is not None:
                superseded.append(f"{mtype}: {replaced['text']} -> {note_text}")
                replaced["text"] = note_text
                replaced["description"] = derive_description(note_text)
                replaced["updated_at"] = now()
                replaced["incomplete"] = _needs_rationale(mtype, note_text)
                entry = replaced
            else:
                name = self._unique_name(note_text, entries)
                entry = {
                    "name": name,
                    "description": derive_description(note_text),
                    "text": note_text,
                    "type": mtype,
                    "tags": [],
                    "created_at": now(),
                    "updated_at": now(),
                    "origin_session_id": str(origin.get("session_id", "") or ""),
                    "origin_run_seq": str(origin.get("run_seq", "") or ""),
                    "links": _extract_links(note_text),
                    "incomplete": _needs_rationale(mtype, note_text),
                }
                entries[name] = entry
            self._write_entry(entry)
            results.append(f"{mtype}: {note_text}")
        self._write_modern_index(entries.values())
        self._write_schema()
        return results, superseded

    def _promote_legacy(self, promotions):
        topics = {topic["topic"]: topic for topic in self.load_index()}
        topic_notes = {slug: [note["text"] for note in self.load_topic_notes(slug)] for slug in topics}
        results = []
        superseded = []
        for topic, note_text in promotions:
            meta = DURABLE_TOPIC_DEFAULTS[topic]
            topics.setdefault(
                topic,
                {
                    "topic": topic,
                    "title": meta["title"],
                    "summary": meta["summary"],
                    "tags": list(meta["tags"]),
                },
            )
            existing = topic_notes.setdefault(topic, [])
            if note_text in existing:
                continue
            new_subject = self._subject_key(note_text, self.cjk_recall)
            replaced = False
            if new_subject:
                for index, old_text in enumerate(list(existing)):
                    if self._subject_key(old_text, self.cjk_recall) == new_subject:
                        superseded.append(f"{topic}: {old_text} -> {note_text}")
                        existing[index] = note_text
                        replaced = True
                        break
            if not replaced:
                existing.append(note_text)
            results.append(f"{topic}: {note_text}")
        self._write_index([topics[slug] for slug in sorted(topics)])
        for topic, notes in topic_notes.items():
            self._write_topic(topic, notes)
        return results, superseded


def _ensure_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def resolve_workspace_path(raw_path, workspace_root=None):
    path = Path(str(raw_path))
    if workspace_root is None:
        return path

    root = Path(workspace_root).resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def canonicalize_path(raw_path, workspace_root=None):
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None:
        return Path(str(raw_path)).as_posix()
    if workspace_root is None:
        return Path(str(raw_path)).as_posix()
    root = Path(workspace_root).resolve()
    return resolved.relative_to(root).as_posix()


def file_freshness(raw_path, workspace_root=None):
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


_ASCII_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
# 中日韩统一表意文字 + 扩展 A + 日文假名 + 韩文音节。用「有没有词边界」这个性质
# 分组，不是按语言分：这几段文字都不用空格分词，所以都要走下面的二元切分。
_CJK_RUN_PATTERN = re.compile(r"[㐀-䶿一-鿿぀-ヿ가-힯]+")


def _tokenize(text, cjk=True):
    """把一段文字切成检索词。

    英文侧一直是 `[A-Za-z0-9_]+`；中文侧从前**一个词都切不出来**，所以中文笔记
    在结构上就召不回来（`benchmarks/session_tasks.json` 里那句「关键词一律用英文」
    就是在为这个洞让路）。

    中文切成**相邻两字的组合**（「构建命令」→ 构建 / 建命 / 命令），不是单字也
    不是分词器：单字会让「的」「是」命中一切；真正的分词器要么加依赖、要么自己
    维护词表（那同样是数据文件），而运行时依赖只有 litellm 这一条是仓库硬约束。

    已知副作用：同一句话中文切出的词远多于英文（20 字中文约 19 个、英文约 4 个），
    所以关键词重叠数在混合语言的库里会系统性偏向中文条目。**排序侧的长度归一化
    才是解药**（见 `_overlap_score`），这里不做任何补偿——分词只管切词。
    """
    text = str(text)
    tokens = {token.lower() for token in _ASCII_TOKEN_PATTERN.findall(text)}
    if not cjk:
        return tokens
    for run in _CJK_RUN_PATTERN.findall(text):
        if len(run) == 1:
            # 单字段落保留该字本身，否则「读 A 写 B」这种夹在 ASCII 之间的
            # 单字永远产生不了任何检索词。
            tokens.add(run)
            continue
        tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _overlap_score(query_tokens, candidate_tokens, normalize):
    """关键词重叠的打分。

    `normalize=False` 是这个函数出现之前的行为：纯重叠个数。它对长文本天然有利，
    中文二元切分之后这个偏差会放大到不可接受（同一句话切出的词数差 4 倍以上）。

    `normalize=True` 除以候选词数的平方根——用平方根而不是直接除以词数，是因为
    后者会矫枉过正：一条只有两个词的记忆只要命中一个就 0.5，稳压任何长条目。
    """
    overlap = len(query_tokens & candidate_tokens)
    if not normalize or overlap == 0:
        return float(overlap)
    return overlap / math.sqrt(max(len(candidate_tokens), 1))


def derive_description(text, limit=DESCRIPTION_TOKENS):
    """从正文推出一行描述。

    阶段二**刻意不引入模型调用**：描述取正文第一句并截断。这是降级版，说清楚——
    它保证了「召回只读一行」这个结构，但那一行的质量要等写入器（阶段四）接管才
    真正兑现。中英文的句末标点都要认，否则中文正文永远切不出第一句。
    """
    text = str(text or "").strip()
    if not text:
        return ""
    # 保留句末标点。丢掉它会让「描述包含这句话」这种断言在句号上失手，而描述
    # 本来就是要被人读的一行。中英文的句末标点都要认，否则中文正文切不出第一句。
    match = re.search(r"^(.+?[。！？!?]|.+?[a-z0-9\)\]一-鿿]\.(?=\s|$))", text, re.S)
    first = (match.group(1) if match else text).strip() or text
    return clip(first, limit)


_FILE_PATH_PATTERN = re.compile(r"(?<![\w/\\.])([\w./\\-]+\.(?:py|md|json|toml|txt|ya?ml|cfg|ini|lock))\b")
_LINK_PATTERN = re.compile(r"\[\[([a-z0-9_-]+)\]\]")


def _extract_links(text):
    """`[[name]]` 互链。链到一个还不存在的名字不算错误——它标记的是一件值得
    以后写的事,这是 Claude Code 提示词里明说的,也是卡片盒笔记法本来的用法。"""
    return _dedupe_preserve_order(_LINK_PATTERN.findall(str(text or "")))


def _needs_rationale(memory_type, text):
    """feedback / project 类型要求正文带「为什么」和「什么时候适用」两行。

    这里只做**校验与标记**,不代写:标记会进索引行末尾,提醒读到它的人这条记忆
    只能机械照搬、判断不了边界情况。真正把这两行写出来是写入器(阶段四)的事。
    """
    if memory_type not in {"feedback", "project"}:
        return False
    lowered = str(text or "").lower()
    has_why = "why:" in lowered or "为什么:" in lowered or "为什么：" in lowered
    has_how = "how to apply:" in lowered or "适用" in lowered
    return not (has_why and has_how)


def _names_a_missing_file(text, workspace_root):
    """这条记忆点名了某个文件,而那个文件已经不在了。

    「记忆说 X 存在」和「X 现在存在」不是一回事——这是整理器唯一能确定性判定的
    那半边(函数名和 flag 那半边要 grep 源码,不做)。
    """
    if workspace_root is None:
        return False
    root = Path(workspace_root)
    for candidate in _FILE_PATH_PATTERN.findall(str(text or "")):
        resolved = resolve_workspace_path(candidate, root)
        if resolved is None:
            continue
        if not resolved.exists():
            return True
    return False


def slugify_memory_name(text, fallback_seed=""):
    """把一句事实变成合法的记忆文件名（`[a-z0-9_-]+`）。

    纯中文的事实产生不了 ASCII 词，这时回落到内容哈希——名字不再可读，但它的
    职责只是**唯一标识与互链锚点**，可读性由 `description` 承担。
    """
    words = _ASCII_TOKEN_PATTERN.findall(str(text or "").lower())
    slug = "-".join(words)[:48].strip("-")
    if slug and _NAME_PATTERN.match(slug):
        return slug
    digest = hashlib.sha256(f"{text}{fallback_seed}".encode("utf-8")).hexdigest()[:10]
    return f"mem-{digest}"


def _parse_timestamp(value):
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except Exception:
        return 0.0


def _normalize_note(note, index):
    if isinstance(note, str):
        text = clip(note.strip(), NOTE_TOKENS)
        return {
            "text": text,
            "description": derive_description(text),
            "tags": [],
            "source": "",
            "created_at": now(),
            "note_index": index,
            "kind": "episodic",
        }

    if not isinstance(note, dict):
        text = clip(str(note).strip(), NOTE_TOKENS)
        return {
            "text": text,
            "description": derive_description(text),
            "tags": [],
            "source": "",
            "created_at": now(),
            "note_index": index,
            "kind": "episodic",
        }

    text = clip(str(note.get("text", "")).strip(), NOTE_TOKENS)
    tags = [str(tag).strip() for tag in _ensure_list(note.get("tags", [])) if str(tag).strip()]
    source = str(note.get("source", "")).strip()
    created_at = str(note.get("created_at", "")).strip() or now()
    note_index = int(note.get("note_index", index))
    kind = str(note.get("kind", "episodic")).strip() or "episodic"
    description = clip(str(note.get("description", "")).strip(), DESCRIPTION_TOKENS) or derive_description(text)
    return {
        "text": text,
        "description": description,
        "tags": _dedupe_preserve_order(tags),
        "source": source,
        "created_at": created_at,
        "note_index": note_index,
        "kind": kind,
    }


def normalize_memory_state(state, workspace_root=None):
    if state is None:
        state = default_memory_state()
    elif not isinstance(state, dict):
        raise TypeError("memory state must be a mapping")

    # 规范化层的作用，是把“磁盘里可能长得不太一样的旧状态”
    # 统一整理成当前 runtime 可直接使用的紧凑结构。
    working = state.get("working")
    if not isinstance(working, dict):
        working = {}
    working.setdefault("task_summary", "")
    working.setdefault("recent_files", [])
    working["task_summary"] = clip(str(working.get("task_summary", "")).strip(), TASK_SUMMARY_TOKENS)
    working["recent_files"] = _dedupe_preserve_order(
        [
            canonicalize_path(path, workspace_root)
            for path in _ensure_list(working.get("recent_files", []))
            if str(path).strip()
        ]
    )[-WORKING_FILE_LIMIT:]
    state["working"] = working

    if not str(working["task_summary"]).strip() and state.get("task"):
        working["task_summary"] = clip(str(state.get("task", "")).strip(), TASK_SUMMARY_TOKENS)
    if not working["recent_files"] and state.get("files"):
        working["recent_files"] = _dedupe_preserve_order(
            [
                canonicalize_path(path, workspace_root)
                for path in _ensure_list(state.get("files", []))
                if str(path).strip()
            ]
        )[-WORKING_FILE_LIMIT:]

    episodic_notes = state.get("episodic_notes")
    if not isinstance(episodic_notes, list):
        episodic_notes = []

    if not episodic_notes and state.get("notes"):
        episodic_notes = [
            _normalize_note(note, index)
            for index, note in enumerate(_ensure_list(state.get("notes", [])))
            if str(note).strip()
        ]
    else:
        normalized_notes = []
        for index, note in enumerate(episodic_notes):
            if isinstance(note, str) and not str(note).strip():
                continue
            normalized_notes.append(_normalize_note(note, index))
        episodic_notes = normalized_notes
    episodic_notes = episodic_notes[-EPISODIC_NOTE_LIMIT:]
    state["episodic_notes"] = episodic_notes

    file_summaries = state.get("file_summaries")
    if not isinstance(file_summaries, dict):
        file_summaries = {}
    normalized_file_summaries = {}
    for path, summary in file_summaries.items():
        path = canonicalize_path(path, workspace_root)
        if isinstance(summary, dict):
            text = clip(str(summary.get("summary", "")).strip(), NOTE_TOKENS)
            created_at = str(summary.get("created_at", "")).strip() or now()
            freshness = summary.get("freshness")
            freshness = None if freshness in (None, "") else str(freshness).strip() or None
        else:
            text = clip(str(summary).strip(), NOTE_TOKENS)
            created_at = now()
            freshness = None
        if not path or not text:
            continue
        normalized_file_summaries[path] = {
            "summary": text,
            "created_at": created_at,
            "freshness": freshness,
        }
    state["file_summaries"] = normalized_file_summaries

    next_note_index = state.get("next_note_index")
    if not isinstance(next_note_index, int) or next_note_index < 0:
        next_note_index = 0
    max_index = max([note["note_index"] for note in episodic_notes], default=-1)
    state["next_note_index"] = max(next_note_index, max_index + 1)

    state["task"] = working["task_summary"]
    state["files"] = list(working["recent_files"])
    state["notes"] = [note["text"] for note in episodic_notes]
    durable_root = Path(workspace_root) / ".codingforme" / "memory" if workspace_root is not None else None
    durable_store = DurableMemoryStore(durable_root) if durable_root is not None else None
    state["durable_topics"] = durable_store.topic_slugs() if durable_store is not None else []
    return state


def set_task_summary(state, summary, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    state["working"]["task_summary"] = clip(str(summary).strip(), TASK_SUMMARY_TOKENS)
    state["task"] = state["working"]["task_summary"]
    return state


def remember_file(state, path, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if not path:
        return state
    files = [item for item in state["working"]["recent_files"] if item != path]
    files.append(path)
    state["working"]["recent_files"] = files[-WORKING_FILE_LIMIT:]
    state["files"] = list(state["working"]["recent_files"])
    return state


def append_note(state, text, tags=(), source="", created_at=None, workspace_root=None, kind="episodic"):
    state = normalize_memory_state(state, workspace_root)
    text = clip(str(text).strip(), NOTE_TOKENS)
    if not text:
        return state

    normalized_tags = _dedupe_preserve_order(
        [str(tag).strip() for tag in _ensure_list(tags) if str(tag).strip()]
    )
    note = {
        "text": text,
        "tags": normalized_tags,
        "source": str(source).strip(),
        "created_at": str(created_at).strip() if created_at else now(),
        "note_index": int(state.get("next_note_index", 0)),
        "kind": str(kind).strip() or "episodic",
    }
    state["next_note_index"] = note["note_index"] + 1

    notes = [item for item in state["episodic_notes"] if item["text"] != note["text"]]
    notes.append(note)
    state["episodic_notes"] = notes[-EPISODIC_NOTE_LIMIT:]
    state["notes"] = [item["text"] for item in state["episodic_notes"]]
    return state
def set_file_summary(state, path, summary, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    summary = clip(str(summary).strip(), NOTE_TOKENS)
    if not path or not summary:
        return state
    state["file_summaries"][path] = {
        "summary": summary,
        "created_at": now(),
        "freshness": file_freshness(path, workspace_root),
    }
    return state


def invalidate_file_summary(state, path, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if not path:
        return state
    state["file_summaries"].pop(path, None)
    return state


def invalidate_stale_file_summaries(state, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    invalidated = []
    for path, summary in list(state["file_summaries"].items()):
        current_freshness = file_freshness(path, workspace_root)
        if summary.get("freshness") == current_freshness:
            continue
        invalidated.append(path)
        state["file_summaries"].pop(path, None)
    return state, invalidated


def summarize_read_result(result, limit=180):
    # 我们不会把完整文件内容塞进记忆层，
    # 这里只保留足够提醒下一轮“刚刚读到了什么”的短摘要。
    lines = [line.strip() for line in str(result).splitlines() if line.strip()]
    if not lines:
        return "(empty)"
    if lines[0].startswith("# "):
        lines = lines[1:]
    if not lines:
        return "(empty)"
    summary = " | ".join(lines[:3])
    return clip(summary, limit)


def retrieval_candidates(state, query, limit=3, workspace_root=None, cjk=True, description_scoring=True, memory_types=True):
    state = normalize_memory_state(state, workspace_root)
    query_tokens = _tokenize(query, cjk)
    ranked = []
    for note in state["episodic_notes"]:
        # 召回逻辑故意保持简单透明：先看 tag 精确命中，再看关键词重叠，
        # 最后看新旧程度。这里不引入向量检索——运行时依赖只有 litellm，
        # 而且相似度分数会让「为什么这条被召回」不可复查。
        note_tags = {tag.lower() for tag in note.get("tags", [])}
        if description_scoring:
            note_tokens = _tokenize(note.get("description", ""), cjk) | _tokenize(note.get("source", ""), cjk) | note_tags
        else:
            note_tokens = _tokenize(note.get("text", ""), cjk) | _tokenize(note.get("source", ""), cjk) | note_tags
        exact_tag_match = int(bool(query_tokens & note_tags))
        overlap = _overlap_score(query_tokens, note_tokens, description_scoring)
        if exact_tag_match == 0 and overlap == 0:
            continue
        recency = _parse_timestamp(note.get("created_at"))
        note_index = int(note.get("note_index", 0))
        ranked.append(((exact_tag_match, overlap, recency, note_index), note))

    if workspace_root is not None:
        durable_store = DurableMemoryStore(
            Path(workspace_root) / ".codingforme" / "memory",
            cjk_recall=cjk,
            description_scoring=description_scoring,
            memory_types=memory_types,
        )
        for note in durable_store.retrieval_candidates(query, limit=limit):
            note_tags = {tag.lower() for tag in note.get("tags", [])}
            note_tokens = durable_store.scoring_tokens(note)
            exact_tag_match = int(bool(query_tokens & note_tags))
            overlap = _overlap_score(query_tokens, note_tokens, description_scoring)
            recency = _parse_timestamp(note.get("updated_at") or note.get("created_at"))
            ranked.append(((exact_tag_match, overlap, recency, -1), note))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [note for _, note in ranked[:limit]]


def retrieval_view(state, query, limit=3, workspace_root=None, **kwargs):
    candidates = retrieval_candidates(state, query, limit=limit, workspace_root=workspace_root, **kwargs)
    lines = ["Relevant memory:"]
    if not candidates:
        lines.append("- none")
        return "\n".join(lines)
    for note in candidates:
        lines.append(f"- {note['text']}")
    return "\n".join(lines)


def render_memory_text(state, workspace_root=None, index_text=""):
    state = normalize_memory_state(state, workspace_root)
    # 这里渲染的是给模型看的紧凑“仪表盘”，不是完整回放。
    # 笔记正文默认不展开，只有在相关召回时才按需拿出来。
    lines = [
        "Memory:",
        f"- task: {state['working']['task_summary'] or '-'}",
        f"- recent_files: {', '.join(state['working']['recent_files']) or '-'}",
    ]

    summaries = []
    for path in state["working"]["recent_files"][:FILE_SUMMARY_LIMIT]:
        summary = state["file_summaries"].get(path, {})
        current_freshness = file_freshness(path, workspace_root)
        if summary.get("summary", "") and summary.get("freshness") == current_freshness:
            summaries.append(f"- {path}: {summary['summary']}")
    if summaries:
        lines.append("- file_summaries:")
        lines.extend(f"  {line}" for line in summaries)
    else:
        lines.append("- file_summaries: -")

    lines.append(f"- episodic_notes: {len(state['episodic_notes'])}")
    durable_topics = state.get("durable_topics", [])
    lines.append(f"- durable_topics: {', '.join(durable_topics[:DURABLE_TOPIC_DISPLAY_LIMIT]) or '-'}")
    if len(durable_topics) > DURABLE_TOPIC_DISPLAY_LIMIT:
        lines[-1] += f" (+{len(durable_topics) - DURABLE_TOPIC_DISPLAY_LIMIT} more)"
    index_text = str(index_text or "").strip()
    if index_text:
        # 索引全文跟着 working memory 一起进上下文——它是 Claude Code 遥测里
        # `listed` 那一层的等价物：模型至少得知道库里有哪些记忆，才谈得上引用它们。
        lines.append(index_text)
    return "\n".join(lines)


def is_effectively_empty(state, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    return (
        not str(state["working"]["task_summary"]).strip()
        and not state["working"]["recent_files"]
        and not state["episodic_notes"]
        and not state["file_summaries"]
    )


class LayeredMemory:
    def __init__(self, state=None, workspace_root=None, cjk_recall=True, description_index=True, memory_types=True):
        self.workspace_root = workspace_root
        # 三个开关从 runtime 的 feature flags 一路传到这里。它们不是配置项，是
        # **消融维度**：每一个关掉都必须严格退回该机制出现之前的行为。
        self.cjk_recall = bool(cjk_recall)
        self.description_index = bool(description_index)
        self.memory_types = bool(memory_types)
        self.state = normalize_memory_state(state, workspace_root)
        self.durable_store = (
            DurableMemoryStore(
                Path(workspace_root) / ".codingforme" / "memory",
                cjk_recall=self.cjk_recall,
                description_scoring=self.description_index,
                memory_types=self.memory_types,
            )
            if workspace_root is not None
            else None
        )

    def to_dict(self):
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return self.state

    def canonical_path(self, path):
        return canonicalize_path(path, self.workspace_root)

    def set_task_summary(self, summary):
        self.state = set_task_summary(self.state, summary, self.workspace_root)
        return self

    def remember_file(self, path):
        self.state = remember_file(self.state, path, self.workspace_root)
        return self

    def append_note(self, text, tags=(), source="", created_at=None, kind="episodic"):
        self.state = append_note(
            self.state,
            text,
            tags=tags,
            source=source,
            created_at=created_at,
            workspace_root=self.workspace_root,
            kind=kind,
        )
        return self

    def set_file_summary(self, path, summary):
        self.state = set_file_summary(self.state, path, summary, self.workspace_root)
        return self

    def invalidate_file_summary(self, path):
        self.state = invalidate_file_summary(self.state, path, self.workspace_root)
        return self

    def invalidate_stale_file_summaries(self):
        self.state, invalidated = invalidate_stale_file_summaries(self.state, self.workspace_root)
        return invalidated

    def _recall_kwargs(self):
        return {
            "cjk": self.cjk_recall,
            "description_scoring": self.description_index,
            "memory_types": self.memory_types,
        }

    def retrieval_candidates(self, query, limit=3):
        return retrieval_candidates(
            self.state, query, limit=limit, workspace_root=self.workspace_root, **self._recall_kwargs()
        )

    def retrieval_view(self, query, limit=3):
        return retrieval_view(
            self.state, query, limit=limit, workspace_root=self.workspace_root, **self._recall_kwargs()
        )

    def durable_index(self, limit_tokens, model=None):
        """索引视图 + 它的验收字段。零值也返回——调用方要把它原样写进工件。"""
        if self.durable_store is None or not self.description_index:
            return {"text": "", "entries": 0, "total_entries": 0, "truncated_entries": 0, "incomplete_entries": 0, "tokens": 0}
        return self.durable_store.index_view(limit_tokens, model=model)

    def render_memory_text(self, index_text=""):
        return render_memory_text(self.state, self.workspace_root, index_text=index_text)

    def migrate_durable(self):
        if self.durable_store is None:
            return 0
        return self.durable_store.migrate_legacy()

    def consolidate_durable(self):
        if self.durable_store is None:
            return {"removed_duplicates": 0, "marked_stale": [], "entries": 0, "ran": False}
        result = self.durable_store.consolidate(workspace_root=self.workspace_root)
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return result

    def promote_durable(self, promotions, origin=None):
        if self.durable_store is None:
            return [], []
        self.state = normalize_memory_state(self.state, self.workspace_root)
        promoted, superseded = self.durable_store.promote(promotions, origin=origin)
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return promoted, superseded
