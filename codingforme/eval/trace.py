"""统一 trace 索引：把散在 .codingforme/runs/ 下的事件流重建成三级结构。

为什么存在：
在这之前，每个评测/实验各自打开 report.json 或 trace.jsonl，各写一遍聚合逻辑。
结果是同一个概念（比如“缓存命中率”）在不同地方口径不一样，而且谁也说不清
自己算的是哪一轮。TraceIndex 把“读工件”这件事收敛成唯一入口：

    Session（一条会话）
      └── Run（一次 ask()）
            └── Turn（一轮 attempt：组 prompt -> 调模型 -> 也许执行工具）

指标层只跟这三个对象打交道，不再直接碰文件。这样新增指标是写一个纯函数，
而不是再抄一遍解析代码。

对老工件的兼容：
0.2.x 及更早的 trace 里没有 session_id / run_seq / turn 三个字段。索引会
自行补齐——turn 按 prompt_built 出现的次序重建（每轮恰好一次，可精确还原），
session 则退化成“每个 run 自成一会话”并标记 synthetic=True。历史数据因此
仍然可以被重新计算，只是跨 session 维度上是空的。
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

TRACE_INDEX_SCHEMA_VERSION = 1

# 一轮里三个关键事件。索引按 (run_id, turn) 把它们配成一个 TurnRecord。
EVENT_PROMPT_BUILT = "prompt_built"
EVENT_MODEL_REQUESTED = "model_requested"
EVENT_MODEL_PARSED = "model_parsed"
EVENT_TOOL_EXECUTED = "tool_executed"
# 一轮里没轮到执行就被步数预算卡住的调用。它**不是** tool_executed 的一种：
# 混进去会让 calls_per_turn 和所有 L1 断言把没跑的调用算成跑了。
EVENT_TOOL_SKIPPED = "tool_skipped"
EVENT_RUN_STARTED = "run_started"
EVENT_RUN_FINISHED = "run_finished"
EVENT_CHECKPOINT_CREATED = "checkpoint_created"


def parse_iso8601(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


@dataclass
class TurnRecord:
    """一次 attempt 的全部可观测事实。"""

    session_id: str
    run_id: str
    run_seq: int
    turn: int
    created_at: str = ""
    # 模型调用前组好的 prompt 元数据（分段字符数、prefix hash、resume 状态……）
    prompt_metadata: dict = field(default_factory=dict)
    # 模型调用后从后端拿到的元数据（usage、cached_tokens、是否走原生工具调用……）
    completion_metadata: dict = field(default_factory=dict)
    # parse() 的归约结果：tool / final / retry
    kind: str = ""
    # kind == "retry" 时的归因码（runtime 的 RETRY_REASON_*），其余情况恒为空串。
    # 没有它，trace 上三类 retry（空响应 / 被截断 / 形状不合法）长得一模一样，
    # 而它们该改的东西完全不同。老工件没有这个字段，退化成空串。
    retry_reason: str = ""
    text_protocol_tool_call: bool = False
    prompt_build_ms: float = 0.0
    model_ms: float = 0.0
    # 该轮执行过的工具调用，按发生顺序；没执行工具时是空列表。
    #
    # 是列表而不是单个：runtime 允许模型一轮发多个工具调用并逐个执行，所以一轮里
    # 会有多条 tool_executed 事件。这里以前是 `tool: dict`，后来的事件会把先来的
    # 覆盖掉——L1 的安全断言（path_confined 之类）就只查得到每轮最后一个调用，
    # 前面的静默漏检。判分器一律遍历这个列表，不要回退到只看某一个。
    tools: list = field(default_factory=list)
    # 该轮申请了但没执行的调用（预算在批次中途耗尽）。与 tools 分开存是刻意的：
    # 合在一起就无法区分「做了 3 件事」和「申请 3 件、只做成 2 件」。
    skipped_tools: list = field(default_factory=list)
    checkpoint_triggers: list = field(default_factory=list)

    @property
    def tool(self):
        """该轮的第一个工具调用；没有则为 None。

        只为「一轮一个调用」时代写的读法保留，新代码请用 `tools`。
        """
        return self.tools[0] if self.tools else None

    @property
    def prompt_cache_key(self):
        return str(self.prompt_metadata.get("prompt_cache_key", "") or "")

    @property
    def cached_tokens(self):
        return int(self.completion_metadata.get("cached_tokens", 0) or 0)

    @property
    def input_tokens(self):
        return int(self.completion_metadata.get("input_tokens", 0) or 0)

    @property
    def output_tokens(self):
        return int(self.completion_metadata.get("output_tokens", 0) or 0)

    @property
    def reasoning_tokens(self):
        """思维链 token。**它是 output_tokens 的一部分，不是额外的。**"""
        return int(self.completion_metadata.get("reasoning_tokens", 0) or 0)

    @property
    def has_usage(self):
        """这一轮的后端响应带没带 usage。回放模式恒为 False。"""
        return self.completion_metadata.get("input_tokens") is not None

    @property
    def cache_hit(self):
        return bool(self.completion_metadata.get("cache_hit"))

    @property
    def tool_name(self):
        return str((self.tool or {}).get("name", "") or "")

    @property
    def tool_status(self):
        return str((self.tool or {}).get("tool_status", "") or "")

    @property
    def tool_ms(self):
        return float((self.tool or {}).get("duration_ms", 0.0) or 0.0)

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "run_seq": self.run_seq,
            "turn": self.turn,
            "created_at": self.created_at,
            "kind": self.kind,
            "retry_reason": self.retry_reason,
            "text_protocol_tool_call": self.text_protocol_tool_call,
            "prompt_build_ms": self.prompt_build_ms,
            "model_ms": self.model_ms,
            "tool_name": self.tool_name,
            "tool_status": self.tool_status,
            "tool_ms": self.tool_ms,
            # 一轮可能执行了多个工具调用；上面三个字段只描述第一个，这里给出全貌。
            "tool_count": len(self.tools),
            "tool_names": [str(item.get("name", "") or "") for item in self.tools],
            "prompt_cache_key": self.prompt_cache_key,
            "cached_tokens": self.cached_tokens,
            "input_tokens": self.input_tokens,
            "cache_hit": self.cache_hit,
            "checkpoint_triggers": list(self.checkpoint_triggers),
        }


@dataclass
class RunRecord:
    """一次 ask() 的全部工件。"""

    run_id: str
    session_id: str
    run_seq: int
    run_dir: Path
    turns: list = field(default_factory=list)
    events: list = field(default_factory=list)
    report: dict = field(default_factory=dict)
    task_state: dict = field(default_factory=dict)
    synthetic_session: bool = False

    @property
    def task_id(self):
        return str(self.task_state.get("task_id", "") or self.report.get("task_id", "") or "")

    @property
    def status(self):
        return str(self.task_state.get("status", "") or self.report.get("status", "") or "")

    @property
    def stop_reason(self):
        return str(self.task_state.get("stop_reason", "") or self.report.get("stop_reason", "") or "")

    @property
    def max_steps(self):
        """这次运行实际生效的步数上限；老工件没这个字段时返回 None。

        返回 None 而不是兜个默认值是刻意的：判分器得以区分「上限是 6」和
        「不知道上限是多少」，后者应当回退到 harness 声明值而不是硬判。
        """
        value = self.report.get("max_steps")
        return int(value) if value else None

    @property
    def resume_status(self):
        return str(self.task_state.get("resume_status", "") or self.report.get("resume_status", "") or "")

    @property
    def tool_steps(self):
        return int(self.task_state.get("tool_steps", 0) or 0)

    @property
    def attempts(self):
        return int(self.task_state.get("attempts", 0) or 0)

    @property
    def started_at(self):
        started = next((event for event in self.events if event.get("event") == EVENT_RUN_STARTED), None)
        if started:
            return str(started.get("created_at", ""))
        return str(self.events[0].get("created_at", "")) if self.events else ""

    @property
    def finished_event(self):
        return next(
            (event for event in reversed(self.events) if event.get("event") == EVENT_RUN_FINISHED),
            None,
        )

    @property
    def duration_ms(self):
        finished = self.finished_event
        if finished and finished.get("run_duration_ms") is not None:
            return float(finished["run_duration_ms"])
        start_dt = parse_iso8601(self.started_at)
        end_dt = parse_iso8601((finished or {}).get("created_at"))
        if start_dt is None or end_dt is None:
            return 0.0
        return max(0.0, (end_dt - start_dt).total_seconds() * 1000.0)

    @property
    def first_turn(self):
        return self.turns[0] if self.turns else None

    @property
    def declared_tools_allowlist(self):
        """这次运行声明的工具白名单；没声明（不限制）或老工件缺字段时为 None。

        取首轮而不是末轮：白名单在装配时就定死，一次运行内不会变，而首轮
        必然存在（`prompt_built` 是每轮的第一个事件）。
        """
        turn = self.first_turn
        names = (turn.prompt_metadata.get("tools_allowlist") if turn else None) or None
        return tuple(str(name) for name in names) if names else None

    @property
    def tool_registry(self):
        """这次运行模型实际能看到的工具名；老工件缺字段时为 None（而不是空集）。"""
        turn = self.first_turn
        names = (turn.prompt_metadata.get("tool_names") if turn else None) or None
        return tuple(str(name) for name in names) if names else None

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "run_seq": self.run_seq,
            "task_id": self.task_id,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "resume_status": self.resume_status,
            "tool_steps": self.tool_steps,
            "attempts": self.attempts,
            "turn_count": len(self.turns),
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "synthetic_session": self.synthetic_session,
        }


@dataclass
class SessionRecord:
    """一条会话：按 run_seq 排好序的若干次 ask()。"""

    session_id: str
    runs: list = field(default_factory=list)
    session_state: dict = field(default_factory=dict)
    synthetic: bool = False

    @property
    def turns(self):
        return [turn for run in self.runs for turn in run.turns]

    @property
    def run_count(self):
        return len(self.runs)

    @property
    def started_at(self):
        return self.runs[0].started_at if self.runs else ""

    @property
    def span_ms(self):
        if not self.runs:
            return 0.0
        start_dt = parse_iso8601(self.runs[0].started_at)
        last = self.runs[-1]
        end_dt = parse_iso8601((last.finished_event or {}).get("created_at")) or parse_iso8601(last.started_at)
        if start_dt is None or end_dt is None:
            return 0.0
        return max(0.0, (end_dt - start_dt).total_seconds() * 1000.0)

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "run_count": self.run_count,
            "turn_count": len(self.turns),
            "started_at": self.started_at,
            "span_ms": self.span_ms,
            "synthetic": self.synthetic,
        }


def _read_jsonl(path):
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # trace 是追加写的，进程被打断时最后一行可能是半截 JSON。
            # 跳过它，而不是让整个索引失败——跑一半的运行同样有分析价值。
            continue
    return events


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _build_turns(run_id, session_id, run_seq, events):
    """把事件流按 (turn) 归拢成 TurnRecord。

    老 trace 没有 turn 字段，这里用 prompt_built 的出现次序重建：
    ask() 的主循环每个 attempt 恰好发一次 prompt_built，所以这个还原是精确的。
    """
    turns = {}
    order = []
    fallback_turn = 0
    for event in events:
        name = str(event.get("event", ""))
        if name == EVENT_PROMPT_BUILT:
            fallback_turn += 1
        turn_no = event.get("turn")
        if turn_no is None:
            turn_no = fallback_turn
        turn_no = int(turn_no or 0)
        if turn_no <= 0:
            # run_started 之类发生在第一轮之前的事件，不属于任何 attempt。
            continue
        if turn_no not in turns:
            turns[turn_no] = TurnRecord(
                session_id=session_id,
                run_id=run_id,
                run_seq=run_seq,
                turn=turn_no,
                created_at=str(event.get("created_at", "")),
            )
            order.append(turn_no)
        record = turns[turn_no]
        if name == EVENT_PROMPT_BUILT:
            record.prompt_metadata = dict(event.get("prompt_metadata", {}) or {})
            record.prompt_build_ms = float(event.get("duration_ms", 0.0) or 0.0)
        elif name == EVENT_MODEL_PARSED:
            record.kind = str(event.get("kind", "") or "")
            record.retry_reason = str(event.get("retry_reason", "") or "")
            record.text_protocol_tool_call = bool(event.get("text_protocol_tool_call"))
            record.completion_metadata = dict(event.get("completion_metadata", {}) or {})
            record.model_ms = float(event.get("duration_ms", 0.0) or 0.0)
        elif name == EVENT_TOOL_EXECUTED:
            record.tools.append(dict(event))
        elif name == EVENT_TOOL_SKIPPED:
            record.skipped_tools.append(dict(event))
        elif name == EVENT_CHECKPOINT_CREATED:
            trigger = str(event.get("trigger", "") or "")
            if trigger:
                record.checkpoint_triggers.append(trigger)
    return [turns[turn_no] for turn_no in sorted(order)]


def _group_sessions(run_records, session_states):
    grouped = {}
    for run in run_records:
        grouped.setdefault(run.session_id, []).append(run)
    sessions = []
    for session_id in sorted(grouped):
        runs = sorted(grouped[session_id], key=lambda item: (item.run_seq, item.started_at, item.run_id))
        sessions.append(
            SessionRecord(
                session_id=session_id,
                runs=runs,
                session_state=dict(session_states.get(session_id, {})),
                synthetic=all(run.synthetic_session for run in runs),
            )
        )
    return sessions


class TraceIndex:
    """runs/ + sessions/ 的只读索引。所有指标都从这里取数。"""

    def __init__(self, runs=None, sessions=None, runs_root=None, sessions_root=None):
        self.runs = list(runs or [])
        self.sessions = list(sessions or [])
        self.runs_root = Path(runs_root) if runs_root else None
        self.sessions_root = Path(sessions_root) if sessions_root else None

    @property
    def turns(self):
        return [turn for run in self.runs for turn in run.turns]

    @classmethod
    def load(cls, runs_root, sessions_root=None):
        runs_root = Path(runs_root)
        run_records = []
        run_dirs = sorted(path for path in runs_root.glob("*") if path.is_dir()) if runs_root.is_dir() else []
        for run_dir in run_dirs:
            trace_path = run_dir / "trace.jsonl"
            events = _read_jsonl(trace_path) if trace_path.exists() else []
            report_path = run_dir / "report.json"
            report = _read_json(report_path) if report_path.exists() else {}
            task_state_path = run_dir / "task_state.json"
            task_state = _read_json(task_state_path) if task_state_path.exists() else {}

            run_id = (
                str(task_state.get("run_id", ""))
                or str(report.get("run_id", ""))
                or next((str(event.get("run_id", "")) for event in events if event.get("run_id")), "")
                or run_dir.name
            )
            session_id = (
                str(task_state.get("session_id", ""))
                or str(report.get("session_id", ""))
                or next((str(event.get("session_id", "")) for event in events if event.get("session_id")), "")
            )
            synthetic = not session_id
            if synthetic:
                # 老工件没有 session_id。退化成“每个 run 自成一会话”，
                # 这样 run 层指标照常可算，只是跨 session 维度为空。
                session_id = f"unknown:{run_id}"
            run_seq = int(task_state.get("run_seq", 0) or report.get("run_seq", 0) or 0)
            if not run_seq:
                run_seq = next((int(event.get("run_seq", 0) or 0) for event in events if event.get("run_seq")), 0)

            run_records.append(
                RunRecord(
                    run_id=run_id,
                    session_id=session_id,
                    run_seq=run_seq,
                    run_dir=run_dir,
                    turns=_build_turns(run_id, session_id, run_seq, events),
                    events=events,
                    report=report,
                    task_state=task_state,
                    synthetic_session=synthetic,
                )
            )

        session_states = {}
        sessions_root = Path(sessions_root) if sessions_root else None
        if sessions_root and sessions_root.is_dir():
            for path in sorted(sessions_root.glob("*.json")):
                state = _read_json(path)
                session_states[str(state.get("id", path.stem))] = state

        return cls(
            runs=run_records,
            sessions=_group_sessions(run_records, session_states),
            runs_root=runs_root,
            sessions_root=sessions_root,
        )

    @classmethod
    def merge(cls, indexes):
        """把多个 workspace 的索引合成一个。

        基准评测给每个任务复制一份独立仓库，工件因此散在 N 个
        .codingforme/runs 下；跨任务的分析必须先合并再算。
        """
        indexes = list(indexes)
        runs = [run for index in indexes for run in index.runs]
        session_states = {}
        for index in indexes:
            for session in index.sessions:
                if session.session_state:
                    session_states[session.session_id] = session.session_state
        return cls(runs=runs, sessions=_group_sessions(runs, session_states))

    def run(self, run_id):
        return next((run for run in self.runs if run.run_id == run_id), None)

    def session(self, session_id):
        return next((item for item in self.sessions if item.session_id == session_id), None)

    def events(self, name=None):
        for run in self.runs:
            for event in run.events:
                if name is None or event.get("event") == name:
                    yield event

    def coverage(self):
        """索引自检：有多少工件带了三级身份，有多少是靠兼容路径补出来的。

        这个数字本身就是一条指标——它回答“我的评测底座覆盖了多少数据”，
        并且能立刻暴露“新字段没落盘”这类回归。
        """
        turns = self.turns
        return {
            "schema_version": TRACE_INDEX_SCHEMA_VERSION,
            "run_count": len(self.runs),
            "session_count": len(self.sessions),
            "turn_count": len(turns),
            "runs_with_session_id": sum(1 for run in self.runs if not run.synthetic_session),
            "synthetic_sessions": sum(1 for item in self.sessions if item.synthetic),
            "runs_with_report": sum(1 for run in self.runs if run.report),
            "turns_with_prompt_metadata": sum(1 for turn in turns if turn.prompt_metadata),
            "turns_with_completion_metadata": sum(1 for turn in turns if turn.completion_metadata),
            "multi_run_sessions": sum(1 for item in self.sessions if item.run_count > 1),
        }

    def to_dict(self):
        return {
            "coverage": self.coverage(),
            "sessions": [item.to_dict() for item in self.sessions],
            "runs": [run.to_dict() for run in self.runs],
        }
