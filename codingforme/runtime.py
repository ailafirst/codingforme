"""Agent 运行时核心逻辑。

CodingForMe 就是包在模型外面的控制循环：负责组 prompt、解析模型输出、
校验并执行工具、写 trace、更新工作记忆，以及在合适的时候停下来。
"""

import json
import os
import re
import shutil
import textwrap
import uuid
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import memory as memorylib
from . import models
from . import context_manager
from .context_manager import ContextManager
from .run_store import RunStore
from .task_state import TaskState
from . import plans
from . import tools as toolkit
from .workspace import IGNORED_PATH_NAMES, MAX_HISTORY, MAX_TOOL_OUTPUT, WorkspaceContext, clip, now

# 落盘目录（`.codingforme/tool_outputs/`）的总字节上限。超过就按运行目录整个删、
# 最旧的先删，本次运行的目录永远不删（见 `_sweep_spill_dirs()`）。
#
# 8 MiB 是拍的，但有尺度：一次 live 压力探针里单次运行写出 7 个文件、合计约
# 120 KB，所以这个值大约是六十次同量级运行的量；它限的是「跑批跑久了工作区
# 无上限地长」，不是单次运行的行为。
SPILL_KEEP_BYTES = 8 * 1024 * 1024

SENSITIVE_ENV_NAME_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
REDACTED_VALUE = "<redacted>"
DEFAULT_SHELL_ENV_ALLOWLIST = ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "LOGNAME", "PATH", "PWD", "SHELL", "TERM", "TMPDIR", "TMP", "TEMP", "USER")
DEFAULT_FEATURE_FLAGS = {
    "memory": True,
    "relevant_memory": True,
    "context_reduction": True,
    "prompt_cache": True,
    # 受限编排（run_plan）。**默认关**：它会改变模型的行为，开着跑出来的数据
    # 和关着跑出来的不可比，所以让它成为一个要显式打开的变体维度。
    "plan_tool": False,
    # 受限委派（delegate）。**默认关**，和 plan_tool 同一个理由。它是四层上下文治理
    # 里「不让内容进主上下文」那一层唯一的机制：整段调查在子 agent 的独立上下文里
    # 跑完，主上下文只收到一行结论。CLI 在 `cli.build_agent()` 里显式打开。
    "delegate_tool": False,
    # 下面两个都**默认开**，存在只是为了能关掉做对照——机制本身的收益，只有
    # 同一份负载跑一次开、跑一次关才量得出来。关掉时都退回到这两个机制出现
    # 之前的行为，不是退回到某个第三种形态。
    #
    # tool_output_spill：关 → 超限的工具结果直接按 `clip()` 截断丢尾巴，不落盘、
    # 不给指针（也就是阶段三 L1 之前的样子）。
    "tool_output_spill": True,
    # recent_window_block：关 → 最近窗口每个工具轮滑一格（block=1），也就是
    # `RECENT_WINDOW_BLOCK` 那条注释里说的「每轮作废一次前缀缓存」的老行为。
    "recent_window_block": True,
    # 分级压缩：触发点 85% / 目标点 70%。关掉退回「只有 100% 一根线、超了才动手、
    # 只裁刚好够的量」的老行为。
    "graded_compression": True,
    # 会话摘要（L5 便宜的那一半）：压力到触发点时，把最早的一批历史条目整体换成
    # 一份确定性摘要，并一次跳到预算的 45% 以下。关掉 → 完全不生成摘要，历史仍按
    # 预算逐条丢弃（也就是这个机制出现之前的样子）。
    "session_summary": True,
    # 供给侧新鲜度：一次 read_file 之后同一个文件又被写过，那条读记录在最近窗口里
    # 仍然是**改动前**的全文。关掉 → 照旧全文呈现，也就是这个机制出现之前的样子。
    "stale_read_invalidation": True,
    # 一次裁剪至少腾出预算的 1/10（`CLEAR_AT_LEAST_DIVISOR`）。关掉 → 只裁刚好够的
    # 量，也就是这个机制出现之前的样子：下一轮几乎必然再裁一次，而每裁一次 history
    # 就作废它之后的全部前缀缓存。
    "clear_at_least": True,
    # 可逆折叠（L4）：窗口外放不下的条目先压扁成 10 个 token 的残句，压不下才整条
    # 丢弃。关掉 → 直接丢弃，只在 `Omitted context:` 那一行里留个数。两者的区别是
    # 「模型知不知道这一轮发生过」，而在这个开关出现之前它们在工件上分不出来。
    "reversible_squeeze": True,
}
# 稳定前缀的模板。`__TOOL_TEXT__` / `__WORKSPACE_TEXT__` 由 `build_prefix()` 在
# dedent **之后**替换。
#
# 为什么用占位符而不是 f-string：`textwrap.dedent` 取的是所有行的**公共**缩进，
# 而 f-string 是先插值再 dedent 的——工具清单和 workspace 快照都是多行且行首无
# 空格，一展开公共缩进就变成 0，dedent 于是什么都不做，整段 prompt 带着 12 个
# 空格的缩进发给模型（实测每行都有，约占 prefix 的 11%）。占位符是单行且带着
# 模板缩进，dedent 能正常算出公共前缀。`workspace.py` 的 `text()` 同理。
#
# 为什么提到模块级：它是**会改变模型行为、却不属于任何配置字段**的东西。
# 提出来之后 `prompt_template_signature()` 才能哈希它，让 harness 指纹覆盖
# 「提示词改过没有」——否则改四轮提示词，指纹一次不变（见 eval/harness.py）。
#
# prefix 只教一套协议：标准 function-calling。这里曾经按后端能力二选一地教
# 「原生」或「<tool> 文本标签」。放弃文本那套是因为它从来不是真正的故障降级——
# 协议在构造 client 时就定死了，一次请求里不会从原生掉到标签，所谓"兜底"其实
# 从不触发。而两套协议一旦可能同时出现，代价是实打实的：实测同一个后端 19 次
# 走原生、17 次吐文本标签，推理模型还会把 <tool> 标签埋进 reasoning_content
# 里让整轮作废。现在 parse() 里的标签解析降级成"宽容读取"：模型万一自己吐了
# 标签也照收，但我们绝不再教它这么做。
#
# prompt 刻意**不规定一轮发几个工具调用**。压制方向试过：规则加示范能把多调用率
# 从 42% 压到 8%，但那是概率性的——用户只要说一句「这几个文件都看一下」，模型就
# 会站在用户那边，而每次压不住就白烧一个约 17 秒的固定往返。鼓励方向也试过：
# Anthropic 官方那两个模板在这个后端上，短版 20/60 → 20/60（p=1.0000）、
# 强版 20/60 → 25/60（p=0.4509），都不显著。runtime 两种都接，prompt 因此
# 不需要也不应该对这件事表态。
PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are coding-for-me, a small local coding agent working inside a local repository.

    Rules:
    - Use tools instead of guessing about the workspace.
    - Call tools through the provided function-calling interface.
    - Never write tool calls as text or XML in your reply; use the tool-call interface.
    - When you are done, reply with the answer as plain text and no tool call.
    - Never invent tool results.
    - Keep answers concise and concrete.
    - If asked to remember/save/persist something so it survives future sessions, restate the fact as a line starting with exactly one of `Project convention:`, `Decision:`, `Dependency:`, `Preference:` (or 项目约定：/决策：/依赖：/偏好：) — only lines in that exact shape are kept long-term; a plain "I'll remember that" is not.
    __WRITE_RULE__
    - Before writing tests for existing code, read the implementation first.
    - When writing tests, match the current implementation unless the user explicitly asked you to change the code.
    - New files should be complete and runnable, including obvious imports.
    - Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.
    __REQUIRED_ARGS_RULE__
    - Every path argument is relative to the repo root below. Absolute paths and paths containing ".." are rejected.

    Tools:
    __TOOL_TEXT__

    __WORKSPACE_TEXT__
    """
).strip()


def _english_list(names):
    """把工具名拼成 `a`、`a or b`、`a, b, or c`，用于 prompt 里点名工具的规则句。"""
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    if len(names) == 2:
        return f"{names[0]} or {names[1]}"
    return ", ".join(names[:-1]) + f", or {names[-1]}"


def prompt_template_signature():
    """稳定前缀模板的 sha256。

    存在的理由：`HarnessSpec.fingerprint()` 只吃配置字段，而**提示词文本不是
    配置字段**。实测后果是所有跑批的指纹恒为同一个值（`sha256:1e0dcb0f0a19`），
    提示词改过四轮、协议改过一轮，指纹一次没变——而那个字段在结果 schema 里的
    定义正是「用来判定两次结果是否可比」。
    """
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


CHECKPOINT_SCHEMA_VERSION = "phase1-v1"
CHECKPOINT_NONE_STATUS = "no-checkpoint"
CHECKPOINT_FULL_VALID_STATUS = "full-valid"
CHECKPOINT_PARTIAL_STALE_STATUS = "partial-stale"
CHECKPOINT_WORKSPACE_MISMATCH_STATUS = "workspace-mismatch"
CHECKPOINT_SCHEMA_MISMATCH_STATUS = "schema-mismatch"
DURABLE_MEMORY_INTENT_PATTERN = re.compile(r"(?i)\b(capture|remember|save|store|persist|note)\b")
DURABLE_MEMORY_INTENT_ZH_PATTERN = re.compile(r"(记住|保存|记录|沉淀|长期记忆|持久记忆)")
DURABLE_MEMORY_LINE_PATTERNS = (
    ("project-conventions", re.compile(r"(?i)^Project convention:\s*(.+)$")),
    ("key-decisions", re.compile(r"(?i)^Decision:\s*(.+)$")),
    ("dependency-facts", re.compile(r"(?i)^Dependency:\s*(.+)$")),
    ("user-preferences", re.compile(r"(?i)^Preference:\s*(.+)$")),
    ("project-conventions", re.compile(r"^项目约定：\s*(.+)$")),
    ("key-decisions", re.compile(r"^决策：\s*(.+)$")),
    ("dependency-facts", re.compile(r"^依赖：\s*(.+)$")),
    ("user-preferences", re.compile(r"^偏好：\s*(.+)$")),
)
SECRET_SHAPED_TEXT_PATTERN = re.compile(r"(?i)(\b(api[_ -]?key|token|secret|password)\b|sk-[A-Za-z0-9_-]{6,})")

# retry 的机器可读原因码。写进 trace 的 `model_parsed` 事件，因为「这一轮为什么
# 作废」在工件上此前只体现为 kind == "retry"，而三类 retry 的应对完全相反：
# 空响应 / 推理阶段被 max_tokens 截断要调 max_new_tokens，形状不合法要改工具
# schema 和示例。分不出是哪一类，就只能靠重跑撞见。
RETRY_REASON_EMPTY_RESPONSE = "empty_response"
RETRY_REASON_EMPTY_FINAL = "empty_final_answer"
RETRY_REASON_TOOL_CALL_NOT_OBJECT = "tool_call_not_an_object"
RETRY_REASON_MISSING_TOOL_NAME = "missing_tool_name"
RETRY_REASON_TOOL_ARGS_NOT_OBJECT = "tool_args_not_an_object"
RETRY_REASON_MALFORMED_TOOL_JSON = "malformed_tool_json"
RETRY_REASON_TOOL_PAYLOAD_NOT_OBJECT = "tool_payload_not_an_object"
RETRY_REASON_UNPARSABLE_TOOL_TAG = "unparsable_tool_tag"


class RetryNotice(str):
    """retry 的 notice 文本，外加一个机器可读的原因码。

    做成 str 的子类而不是把 `parse()` 的返回形状改成三元组：payload 仍然**就是**
    一个字符串，既有调用方、既有用例、写回 history 的那条 runtime notice 全部
    不受影响，只有 trace 的发射点会多取一个 `.reason`。
    """

    def __new__(cls, text, reason):
        notice = super().__new__(cls, text)
        notice.reason = str(reason)
        return notice



@dataclass
class PromptPrefix:
    # prefix 除了文本本身，还带一小份元数据，
    # 这样 runtime 才能明确判断 prefix 是否可以复用。
    text: str
    hash: str
    workspace_fingerprint: str
    tool_signature: str
    built_at: str


class SessionStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id):
        return self.root / f"{session_id}.json"

    def save(self, session):
        path = self.path(session["id"])
        path.write_text(json.dumps(session, indent=2), encoding="utf-8")
        return path

    def load(self, session_id):
        return json.loads(self.path(session_id).read_text(encoding="utf-8"))

    def latest(self):
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return files[-1].stem if files else None


class CodingForMe:
    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        run_store=None,
        approval_policy="ask",
        # 步数上限是「卡住时止损的看门狗」，不是「这件事该花几步」的估计。
        # 20 的来历：实测同一批任务下真实模型做对一次用 3~10 步，通过率在 10 步处
        # 饱和（再放宽到 30 步一个成功都不多增），取饱和点的 2 倍留余量。
        # 评测侧的上限不跟这个走——`eval/harness.py` 的 HarnessSpec 自己声明。
        max_steps=20,
        # 单轮输出上限。512 太紧：实测一次真实运行里有一轮输出正好等于 512 被截断，
        # 整轮作废重来，白花 42 秒（占那次运行总耗时的 11%）。截断的代价是一整个
        # 模型往返，而放宽的代价只是偶尔多生成几十个 token，两边不对称。
        # 评测侧仍是 512——`eval/harness.py` 的 HarnessSpec 自己声明。
        max_new_tokens=1024,
        depth=0,
        max_depth=1,
        read_only=False,
        shell_env_allowlist=None,
        secret_env_names=None,
        feature_flags=None,
        on_token=None,
        # 上下文窗口（token）。None 表示自动解析：已知后端表 → litellm 注册表 →
        # 保守默认值，解析结果向下取整到档位。见 models.resolve_context_window()。
        context_window=None,
    ):
        self.model_client = model_client
        # 可选的流式回调：传入时 model_client.complete() 会走 SSE 流式，
        # 每个文本增量都会实时回调给它（比如 REPL 想要边生成边显示）。
        # 不传（默认）时行为和之前完全一样，一次性拿完整结果。
        self.on_token = on_token
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.shell_env_allowlist = tuple(shell_env_allowlist or DEFAULT_SHELL_ENV_ALLOWLIST)
        self.secret_env_names = {str(name).upper() for name in (secret_env_names or ())}
        self.feature_flags = dict(DEFAULT_FEATURE_FLAGS)
        if feature_flags:
            self.feature_flags.update({str(key): bool(value) for key, value in feature_flags.items()})
        self.run_store = run_store or RunStore(Path(workspace.repo_root) / ".codingforme" / "runs")
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": memorylib.default_memory_state(),
        }
        self._ensure_session_shape()
        self.memory = memorylib.LayeredMemory(
            self.session.setdefault("memory", memorylib.default_memory_state()),
            workspace_root=self.root,
        )
        self.session["memory"] = self.memory.to_dict()
        self.tools = self.build_tools()
        # 装配方声明的工具白名单，仅供工件记录用；真正的裁剪由装配方直接改
        # `self.tools` 完成（见 eval/harness.py 的 HarnessSpec.build）。两者
        # 分开记是刻意的：判分器要能查出「声明了却没裁」这种漏接。
        self.declared_tools_allowlist = None
        # 数据集声明的「回答所依赖的关键串」，由评测层盖上来，见 `_context_evidence_report()`。
        self.declared_context_evidence = None
        # 受限编排（run_plan）与控制循环之间的三个瞬时状态。放在实例上而不是
        # 参数里，是因为它们要穿过 `run_tool()` 这层通用闸口——闸口的签名
        # `(name, args) -> str` 对所有工具一致，不该为一个工具开口子。
        self._active_task_state = None      # 当前 run 的 TaskState，内层调用写 trace 要用
        # 除 run_plan 这一步之外还能再跑几个调用。`ask()` 每次调工具前都会重设它；
        # 初值取 max_steps 是为了让绕过控制循环的直接调用（测试、metrics 的安全
        # 场景复现）也拿到一个合理预算，而不是静默退化成"只跑第一个调用"。
        self._steps_available = int(max_steps)
        self._extra_tool_steps = 0          # run_plan 额外吃掉了几步，由 ask() 加回预算
        self._last_plan_result = None
        # 由 model client 声明自己能不能吃标准 function-calling 的 tools=。
        # prefix 的协议说明、ask() 要不要发 tools=、retry notice 的措辞都看它，
        # 保证"发出去的协议"和"教给模型的协议"永远是同一套。
        self.native_tool_calls = bool(getattr(model_client, "supports_native_tool_calls", False))
        self.prefix_state = self.build_prefix()
        self.prefix = self.prefix_state.text
        # 上下文预算从窗口派生，而不是写死一个数。三项扣除都不能省，理由见
        # models.context_budget_tokens() 与 docs/architecture/context-budget-sizing.md：
        # 工具 schema 每轮重发且占实际输入约一半；输出预留要按实际输出算（实测
        # 输出 token 里 66.9% 是推理过程）；安全系数留给分词器估算误差。
        self.set_context_window(context_window)
        self.resume_state = self.evaluate_resume_state()
        self.session_path = self.session_store.save(self.session)
        self.current_task_state = None
        self.current_run_dir = None
        self.last_prompt_metadata = {}
        self.last_completion_metadata = {}
        self.last_durable_promotions = []
        self.last_durable_rejections = []
        self.last_durable_superseded = []
        self._last_tool_result_metadata = {}
        self._last_delegate_stats = {}   # delegate 的上下文记账，由 tools.tool_delegate() 填
        # 落盘过的超长工具结果的序号，只保证一次运行内单调递增。
        self._spill_seq = 0
        # 有多少次工具调用是靠 parse() 的宽容标签读取捞回来的（而不是走标准
        # function-calling 接口）。健康状态下应当恒为 0，非 0 说明模型在偏离
        # 我们唯一宣传的那套协议。
        self.text_protocol_tool_calls = 0
        # trace 事件的三级身份坐标（session -> run -> turn）里的后两级，
        # 由 ask() 在推进过程中维护，emit_trace() 只负责盖章。
        self.current_run_seq = 0
        self.current_turn = 0
        self._last_prefix_refresh = {
            "workspace_changed": False,
            "prefix_changed": False,
        }

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    def _ensure_session_shape(self):
        self.session.setdefault("history", [])
        self.session.setdefault("memory", memorylib.default_memory_state())
        checkpoints = self.session.setdefault("checkpoints", {})
        if not isinstance(checkpoints, dict):
            checkpoints = {}
            self.session["checkpoints"] = checkpoints
        checkpoints.setdefault("current_id", "")
        checkpoints.setdefault("items", {})
        runtime_identity = self.session.setdefault("runtime_identity", {})
        if not isinstance(runtime_identity, dict):
            self.session["runtime_identity"] = {}
        resume_state = self.session.setdefault("resume_state", {})
        if not isinstance(resume_state, dict):
            self.session["resume_state"] = {}
        # 会话内的 ask() 计数。存进 session 而不是进程内变量，
        # 这样 --resume 之后序号能接着涨，跨 session 的时间线不会断。
        self.session.setdefault("run_seq", 0)
        # 会话摘要（context_manager 的阶段二）。放进 session 是为了跟着落盘、跟着
        # resume——它记的是「history 的前 N 条已经被换成这份概述」，这个覆盖点必须
        # 和 history 一起恢复，否则重启之后同一段内容会既在摘要里又在历史里。
        summary = self.session.setdefault("context_summary", {})
        if not isinstance(summary, dict):
            self.session["context_summary"] = {}

    def current_runtime_identity(self):
        return {
            "session_id": self.session.get("id", ""),
            "cwd": str(self.root),
            "model": str(getattr(self.model_client, "model", "")),
            "model_client": self.model_client.__class__.__name__,
            "approval_policy": self.approval_policy,
            "read_only": bool(self.read_only),
            "max_steps": int(self.max_steps),
            "max_new_tokens": int(self.max_new_tokens),
            "feature_flags": dict(self.feature_flags),
            "shell_env_allowlist": list(self.shell_env_allowlist),
            "workspace_fingerprint": getattr(getattr(self, "prefix_state", None), "workspace_fingerprint", self.workspace.fingerprint()),
            "tool_signature": self.tool_signature(),
        }

    def checkpoint_state(self):
        self._ensure_session_shape()
        return self.session["checkpoints"]

    def current_checkpoint(self):
        state = self.checkpoint_state()
        checkpoint_id = str(state.get("current_id", "")).strip()
        if not checkpoint_id:
            return None
        return state.get("items", {}).get(checkpoint_id)

    def invalidate_stale_memory(self):
        invalidated = self.memory.invalidate_stale_file_summaries()
        self.session["memory"] = self.memory.to_dict()
        return invalidated

    def evaluate_resume_state(self):
        previous_resume_state = dict(self.session.get("resume_state", {}) or {})
        invalidated = self.invalidate_stale_memory()
        checkpoint = self.current_checkpoint()
        status = CHECKPOINT_NONE_STATUS
        stale_paths = list(invalidated)
        mismatch_fields = []
        if checkpoint:
            if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                status = CHECKPOINT_SCHEMA_MISMATCH_STATUS
            else:
                for item in checkpoint.get("key_files", []):
                    path = str(item.get("path", "")).strip()
                    if not path:
                        continue
                    expected = item.get("freshness")
                    current = memorylib.file_freshness(path, self.root)
                    if expected != current and path not in stale_paths:
                        stale_paths.append(path)
                saved_identity = dict(checkpoint.get("runtime_identity", {}) or self.session.get("runtime_identity", {}) or {})
                current_identity = self.current_runtime_identity()
                identity_keys = (
                    "cwd",
                    "model",
                    "model_client",
                    "approval_policy",
                    "read_only",
                    "max_steps",
                    "max_new_tokens",
                    "feature_flags",
                    "shell_env_allowlist",
                    "workspace_fingerprint",
                    "tool_signature",
                )
                for key in identity_keys:
                    if key not in saved_identity:
                        continue
                    if saved_identity.get(key) != current_identity.get(key):
                        mismatch_fields.append(key)
                mismatch_fields.sort()
                if stale_paths:
                    status = CHECKPOINT_PARTIAL_STALE_STATUS
                elif mismatch_fields:
                    status = CHECKPOINT_WORKSPACE_MISMATCH_STATUS
                else:
                    status = CHECKPOINT_FULL_VALID_STATUS

        resume_state = {
            "status": status,
            "stale_paths": stale_paths,
            "runtime_identity_mismatch_fields": mismatch_fields,
            "stale_summary_invalidations": max(
                len(invalidated),
                int(previous_resume_state.get("stale_summary_invalidations", 0))
                if status == CHECKPOINT_PARTIAL_STALE_STATUS
                else 0,
            ),
        }
        self.session["resume_state"] = resume_state
        self.session["runtime_identity"] = self.current_runtime_identity()
        return resume_state

    def render_checkpoint_text(self):
        checkpoint = self.current_checkpoint()
        if not checkpoint:
            return ""
        lines = [
            "Task checkpoint:",
            f"- Resume status: {self.resume_state.get('status', CHECKPOINT_NONE_STATUS)}",
            f"- Current goal: {checkpoint.get('current_goal', '-') or '-'}",
            f"- Current blocker: {checkpoint.get('current_blocker', '-') or '-'}",
            f"- Next step: {checkpoint.get('next_step', '-') or '-'}",
        ]
        key_files = [str(item.get("path", "")).strip() for item in checkpoint.get("key_files", []) if str(item.get("path", "")).strip()]
        lines.append(f"- Key files: {', '.join(key_files) or '-'}")
        if checkpoint.get("completed"):
            lines.append("- Completed: " + " | ".join(str(item) for item in checkpoint.get("completed", [])))
        if checkpoint.get("excluded"):
            lines.append("- Excluded: " + " | ".join(str(item) for item in checkpoint.get("excluded", [])))
        if self.resume_state.get("stale_paths"):
            lines.append("- Stale paths: " + ", ".join(self.resume_state["stale_paths"]))
        summary = str(checkpoint.get("summary", "")).strip()
        if summary:
            lines.append(f"- Summary: {summary}")
        return "\n".join(lines)

    @staticmethod
    def remember(bucket, item, limit):
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    def build_tools(self):
        return toolkit.build_tool_registry(self)

    def tool_signature(self):
        payload = []
        for name in sorted(self.tools):
            tool = self.tools[name]
            payload.append(
                {
                    "name": name,
                    "schema": tool["schema"],
                    "risky": tool["risky"],
                    "description": tool["description"],
                }
            )
        # 同时哈希翻译成标准 function-calling 的 JSON Schema 形状，
        # 这样万一 to_openai_function_specs() 的翻译逻辑出 bug、悄悄改变了
        # 发给模型的 tools= schema，也能被这份签名和 prompt cache/resume
        # 的一致性检查捕捉到，而不是一个没有测试保护的隐藏副作用。
        payload.append({"function_specs": toolkit.to_openai_function_specs(self.tools)})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def tool_schema_tokens(self):
        """`tools=` 数组本身要占多少 token——它每一轮都要原样重发。

        为什么必须算进上下文预算：实测基础 6 个工具的 schema 是 4615 字符 /
        1169 token，约占实际输入的 53%（对照同批跑批 `input_tokens` 中位 2188），
        而它一个字符都不进 `prompt_chars`。不计它，预算就漏掉了输入的一半——
        这是「字符/token 比最低到 0.83」那个不可能的比值的来源。

        按 `tool_signature()` 缓存：注册表在一次运行里几乎不变，而计数虽然只要
        约 1ms，也没必要每轮重做。
        """
        signature = self.tool_signature()
        cached = getattr(self, "_tool_schema_tokens_cache", None)
        if cached and cached[0] == signature:
            return cached[1]
        specs = toolkit.to_openai_function_specs(self.tools)
        tokens = models.count_tokens(
            json.dumps(specs, ensure_ascii=False), getattr(self.model_client, "model", None)
        )
        self._tool_schema_tokens_cache = (signature, tokens)
        return tokens

    def set_context_window(self, window_tokens=None):
        """(重新)确定上下文窗口档位，并按它派生预算、换掉 ContextManager。

        构造期调一次；REPL 的 `/context` 命令在运行中再调,所以这段必须是**幂等
        且可重入**的——换句话说它只能读 `self` 上那些不随对话变化的东西(模型名、
        工具注册表、max_new_tokens),不能碰 history 或 memory。

        为什么值得让人手动设:自动探测对大窗口后端会给出一个不该直接用的数
        (mimo-v2.5 官方标 1M,探测就是 1M),而 128k 之外的部分既要计费又会踩上
        下文腐坏。档位是策略,不是模型能力,所以得留一个人工入口。

        换窗口会连带换掉裁剪点,进而换掉前缀缓存认的那段公共前缀——运行中改一次
        等于主动放弃一次缓存命中。这是刻意接受的:调档位是低频动作。

        返回 `(window_tokens, source, budget_tokens)`,好让调用方直接拿去显示,
        不必再去读三个属性。
        """
        self.context_window, self.context_window_source = models.resolve_context_window(
            getattr(self.model_client, "model", None), explicit=window_tokens
        )
        # 四项扣除都不能省，理由见 models.budget_breakdown() 与
        # docs/architecture/context-budget-sizing.md：工具 schema 每轮重发且占实际
        # 输入约一半；输出预留要按实际输出算（实测输出 token 里 66.9% 是推理过程）；
        # 消息结构开销按条数走（实测 residual ≈ 191 + 20 × 消息条数，R²=0.838）；
        # 最后的分词器容差是唯一按比例走的一项。
        #
        # 逐项都留着（`context_budget_breakdown`），不只留那个总数：`/context` 要把
        # 算式显示出来，而「窗口太小、预算已经触底」和「算出来正好是这个数」在一个
        # 整数上分不出来。
        self.context_budget_breakdown = models.budget_breakdown(
            self.context_window,
            tool_schema_tokens=self.tool_schema_tokens(),
            output_reserve_tokens=self.max_new_tokens,
            expected_messages=models.expected_message_count(self.max_steps),
        )
        self.context_budget = self.context_budget_breakdown["budget_tokens"]
        self.context_manager = ContextManager(self, total_budget=self.context_budget)
        return self.context_window, self.context_window_source, self.context_budget

    def build_prefix(self):
        tool_lines = []
        for name, tool in self.tools.items():
            fields = toolkit.render_schema_fields(tool["schema"])
            risk = "approval required" if tool["risky"] else "safe"
            tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
            # 示例参数此前只在**参数校验失败之后**才回给模型。放到 prefix 里是
            # 因为这个通道的成本结构完全不同：prefix 走前缀缓存（实测缓存命中占
            # 输入 token 的 75.4%），多这几十个 token 近乎免费；而一次校验失败
            # 要烧掉一整个约 17 秒的模型往返。同一份文本，事前给远比事后给便宜。
            example = toolkit.tool_example(name, self.tools)
            if example:
                tool_lines.append(f"    example args: {example}")
        tool_text = "\n".join(tool_lines)
        # prefix 只教一套协议：标准 function-calling。
        #
        # 这里曾经按后端能力二选一地教「原生」或「<tool> 文本标签」。放弃文本那套
        # 是因为它从来不是真正的故障降级——协议在构造 client 时就定死了，一次请求
        # 里不会从原生掉到标签，所谓"兜底"其实从不触发。而两套协议一旦有可能同时
        # 出现，代价却是实打实的：实测同一个后端 19 次走原生、17 次吐文本标签，
        # 推理模型还会把 <tool> 标签埋进 reasoning_content 里让整轮作废。
        #
        # 现在 parse() 里的标签解析降级成"宽容读取"：模型万一自己吐了标签也照收，
        # 但我们绝不再教它这么做。见 CodingForMe.parse()。
        #
        # prefix 可以理解成 agent 的“工作手册”：
        # 它是谁、工具怎么调用、当前仓库是什么状态，都写在这里。
        #
        # prompt 刻意**不规定一轮发几个工具调用**。这里曾经用规则加示范去压模型，
        # 让它一轮只发一个，实测能把多调用率从 42% 压到 8%（五种提示形态各采样 12 次）。
        # 那条路后来被放弃了：压得再好也是概率性的——用户只要说一句「这几个文件都看一下」，
        # 模型就会站在用户那边，而每次压不住就白烧一个约 17 秒的固定往返。
        # 现在改成 runtime 两种都接（见 parse() 与 ask()），模型怎么发都不作废，
        # prompt 因此不需要也不应该再对这件事表态。
        # 插值用占位符而不是直接写 {tool_text}：`textwrap.dedent` 取的是所有行的**公共**
        # 缩进，而 f-string 是先插值再 dedent 的——tool_text 和 workspace.text() 都是多行
        # 且行首无空格，一展开公共缩进就变成 0，dedent 于是什么都不做，整段 prompt 带着
        # 12 个空格的缩进发给模型（实测每行都有，约占 prefix 的 11%）。占位符是单行且带着
        # 模板缩进，dedent 能正常算出公共前缀，替换放在 dedent 之后。
        # 规则里凡是**点名某个工具**的句子，都必须按当前注册表现算，不能硬写。
        # 注册表会被变体白名单 ∩ 任务白名单裁掉一部分（见 eval/harness.py），
        # 硬写的规则于是会在工具清单只剩 read_file/patch_file 的运行里，仍然
        # 指着 write_file、run_shell、delegate 说话——模型下一轮照着做，只会
        # 拿到一句 `unknown tool`，白烧一个约 17 秒的往返。一条工具都不剩时
        # 整句删掉，而不是留一句指向空集的规则。
        write_tools = [name for name in ("write_file", "patch_file") if name in self.tools]
        write_rule = (
            "- If the user asks you to create or update a specific file and the path is clear, use "
            f"{_english_list(write_tools)} instead of repeatedly listing files.\n"
            if write_tools
            else ""
        )
        required_arg_tools = [
            name
            for name, tool in self.tools.items()
            if any("=" not in toolkit.schema_field_type(field) for field in tool["schema"].values())
        ]
        required_args_rule = (
            # 措辞是「必须给全」而不是「不能为空」：`patch_file` 的 old_text 传空串是
            # 一种**合法用法**（追加到文件末尾），说成「不能为空」会把模型从那条路上推开。
            "- Required tool arguments must all be provided. Do not call "
            f"{_english_list(required_arg_tools)} with args={{}}.\n"
            if required_arg_tools
            else ""
        )
        text = (
            PROMPT_TEMPLATE
            .replace("__TOOL_TEXT__", tool_text)
            .replace("__WRITE_RULE__\n", write_rule)
            .replace("__REQUIRED_ARGS_RULE__\n", required_args_rule)
            .replace("__WORKSPACE_TEXT__", self.workspace.text())
        )
        return PromptPrefix(
            text=text,
            hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            workspace_fingerprint=self.workspace.fingerprint(),
            tool_signature=self.tool_signature(),
            built_at=now(),
        )

    def _apply_prefix_state(self, prefix_state):
        self.prefix_state = prefix_state
        self.prefix = prefix_state.text

    def refresh_prefix(self, force=False):
        previous_hash = getattr(getattr(self, "prefix_state", None), "hash", None)
        previous_workspace_fingerprint = getattr(getattr(self, "prefix_state", None), "workspace_fingerprint", None)

        # 工作区事实相对稳定，所以这里按整体刷新；
        # 只有这些事实真的变化了，才重建完整 prefix。
        #
        # **必须把 repo_root 钉回 self.root**。不传这个参数时 `WorkspaceContext.build()`
        # 会用 `git rev-parse --show-toplevel` 现算仓库根，于是一个刻意被限定在子目录里
        # 的 workspace（评测把每个任务的样板仓库复制到临时目录、并用 repo_root_override
        # 限定在那份拷贝上）会在**第一次刷新时被悄悄放大成整个外层 git 仓库**。
        #
        # 实测后果（2026-08-19，评测工作区放在本仓库内的那次跑批）：
        # 1. 快照里混进了本仓库自己的 AGENTS.md / README.md / pyproject.toml（合计
        #    3237 字符）和本仓库的 git status，prefix 从约 3400 涨到约 7400 字符，
        #    超过 3600 的段预算；而裁剪是保头去尾的，于是**样板仓库自己的快照和
        #    resume checkpoint 被整段截掉**，两个 resume 任务因此全部撞步数上限。
        # 2. 更严重的是 `delegate`：子 agent 用父 agent 的 workspace 构造，而
        #    `self.root = Path(workspace.repo_root)`——放大之后的 repo_root 让子
        #    agent 拿到了比父 agent 更宽的根目录，实测子 agent 能读到外层仓库的
        #    AGENTS.md。子 agent 的权限只应更小，不可能更大。
        #
        # 不变量：**agent 的根目录在构造之后永不改变。** 普通 CLI 用法下 self.root
        # 本来就等于 git 顶层，传不传这个参数结果相同，所以这条修正不影响日常使用。
        refreshed_workspace = WorkspaceContext.build(self.root, repo_root_override=self.root)
        refreshed_workspace_fingerprint = refreshed_workspace.fingerprint()
        workspace_changed = force or refreshed_workspace_fingerprint != previous_workspace_fingerprint
        if workspace_changed:
            self.workspace = refreshed_workspace

        prefix_state = self.build_prefix() if workspace_changed or force or previous_hash is None else self.prefix_state
        prefix_changed = force or previous_hash != prefix_state.hash
        if prefix_changed:
            self._apply_prefix_state(prefix_state)

        self._last_prefix_refresh = {
            "workspace_changed": workspace_changed,
            "prefix_changed": prefix_changed,
        }
        return dict(self._last_prefix_refresh)

    def memory_text(self):
        return self.memory.render_memory_text()

    def compact_context(self):
        """手动压缩当前会话的上下文(`/compact` 走这里)。

        自动压缩是占用率驱动的,到 85% 才动手。这个入口让用户在**它自己知道
        「这一段结束了」的那一刻**主动压——压缩点越靠前,被作废的前缀缓存越少。

        `history` 一个字节都不动:压缩改的是 `session["context_summary"]` 里的
        覆盖点和摘要正文,也就是这份历史的**投影**。把覆盖点清零就能完整还原,
        这正是 L4「完全可逆」这句话在手动入口上的含义。压完要落盘,否则下一次
        resume 会拿回压缩之前的状态。
        """
        before_covered = self._summary_covered()
        report = self.context_manager.compact_now()
        if report.get("compacted"):
            self.session_path = self.session_store.save(self.session)
            self.emit_context_compacted("manual", before_covered)
        return report

    def _summary_covered(self):
        """摘要当前覆盖到 history 的第几条。压缩前后各取一次就是「这次推进了几条」。"""
        state = (self.session or {}).get("context_summary") or {}
        return int(state.get("covered", 0) or 0)

    def emit_context_compacted(self, trigger, before_covered):
        """把一次上下文压缩写进 trace。

        存在的理由:压缩改变的是**模型看得见什么**,而它此前几乎不在工件上留痕——
        手动 `/compact` 一个字节都不写,自动那条只体现为 `prompt_built` 里
        `context_pressure.session_summary.compactions` 这个嵌套计数。于是「这次运行的
        上下文里为什么没有第 1 轮的规格」只能靠重算 history 去猜,而 resume 又会把
        压缩后的状态原样接着用。

        `trigger` 分 `manual` / `auto` 是刻意的:两者对同一份 session 做同一件事,但
        调参时是相反的信号——auto 多说明触发点或预算该调,manual 多说明用户在替系统
        判断「这一段结束了」,那是好事。

        两条路径填**同一份**字段,取值全部来自 `session["context_summary"]` 这一个源,
        不各自从自己的返回值里凑——凑出来的两种形状会让聚合代码要写两遍。
        """
        task_state = getattr(self, "current_task_state", None)
        if task_state is None:
            # 一轮都还没跑过(REPL 里刚进来就 /compact)。不为这条事件凭空开一个 run:
            # 工件里会多出一次没有任何模型调用的「运行」,把所有按 run 求平均的
            # 指标都算歪。
            return None
        state = (self.session or {}).get("context_summary") or {}
        covered = int(state.get("covered", 0) or 0)
        covered_tokens = int(state.get("covered_tokens", 0) or 0)
        summary_tokens = models.count_tokens(
            str(state.get("text", "") or ""), getattr(self.model_client, "model", None)
        )
        return self.emit_trace(
            task_state,
            "context_compacted",
            {
                "trigger": str(trigger),
                "covered_entries": covered,
                "newly_covered": max(0, covered - int(before_covered)),
                "transcript_entries": len(self.session.get("history", []) or []),
                "covered_tokens": covered_tokens,
                "summary_tokens": summary_tokens,
                # 换进去的比换掉的小多少。可能是负数——摘要比它替代的那几条还长,
                # 那是这个机制在这份负载上帮倒忙,必须能看出来而不是被 max(0,...) 抹平。
                "saved_tokens": covered_tokens - summary_tokens,
                "refreshes": int(state.get("refreshes", 0) or 0),
                # 这条事件挂在哪次运行下、那次运行当时是什么状态。手动压缩发生在
                # 两轮之间时挂的是**上一次**运行,不说明的话读的人会以为它发生在
                # 那次运行进行中。
                "run_status": str(getattr(task_state, "status", "")),
            },
        )

    def history_text(self):
        history = self.session["history"]
        if not history:
            return "- empty"

        lines = []
        seen_reads = set()
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if item["role"] == "tool" and item["name"] == "read_file" and not recent:
                path = str(item["args"].get("path", ""))
                if path in seen_reads:
                    continue
                seen_reads.add(path)

            if item["role"] == "tool":
                # 单位是 token（`clip()` 已统一），由原先的字符值按转录实测的
                # 2.07 字符/token 折算（900 ÷ 2.07 ≈ 430，180 ÷ 2.07 ≈ 87）。
                # 最近的条目留得比早先的厚，是刻意的：下一步决策最依赖刚刚发生
                # 的工具结果。
                limit = 430 if recent else 87
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(clip(item["content"], limit))
            else:
                # 只带 tool_calls、没有说明文字的 assistant 记录在文本视图里没有
                # 内容可写（调用本身已经由紧随其后的 [tool:...] 行表达了），
                # 渲染出来只会是一行空的 "[assistant] "。
                if not str(item.get("content", "")).strip():
                    continue
                limit = 430 if recent else 106
                lines.append(f"[{item['role']}] {clip(item['content'], limit)}")

        return clip("\n".join(lines), MAX_HISTORY)

    def tool_output_limit(self):
        """单个工具结果进 history 时的 token 上限。

        从 `total_budget` 派生（`context_manager.tool_output_limit()`），不是写死的
        `MAX_TOOL_OUTPUT`。旧值 1320 与预算脱钩：1M 档预算 118,335 时它仍然只让一个
        文件进来 1320 个 token，而预算空着十几万；实测把预算放大 27 倍，发出去的
        prompt 一个 token 都不变。

        `context_manager` 还没装配好时回落到 `MAX_TOOL_OUTPUT`——构造期的调用
        （以及不带 context_manager 的测试替身）走这条路，行为与改动前一致。
        """
        budget = getattr(getattr(self, "context_manager", None), "total_budget", None)
        if not budget:
            return MAX_TOOL_OUTPUT
        return context_manager.tool_output_limit(budget)

    def spill_root(self):
        """所有运行的落盘根目录。`read_file` 靠它认出「这是一个落盘件」。"""
        return self.root / ".codingforme" / context_manager.SPILL_DIR_NAME

    def _spill_dir(self):
        """超长工具结果的落盘目录。

        **必须在 workspace root 之下**（不是 `repo_root`）：这条路径会原样交给模型，
        模型拿它去调 `read_file`，而所有路径都要过 `path()` 的锚定检查——落在根之外
        的话，指针指向一个模型永远打不开的地方，等于没给。
        """
        run_id = Path(self.current_run_dir).name if self.current_run_dir else "adhoc"
        return self.spill_root() / run_id

    def _sweep_spill_dirs(self, keep_dir):
        """把落盘总量压到 `SPILL_KEEP_BYTES` 之下，按运行目录整个删、最旧的先删。

        为什么需要：这个目录只增不减，而**取回尝试本身还会再落一次盘**——一次
        live 压力探针里，一次运行就写出 7 个文件。跑批跑久了它会一直长。

        两条刻意的边界：
        - **本次运行的目录永远不删**（`keep_dir`）。它里面的文件正被 history 里的
          指针引用着，删掉就把「可恢复」变成了一句谎话。
        - 按**运行目录**整个删，不按单文件删。同一次运行里的文件是一组互相引用的
          证据，删一半留一半只会让指针指向一个残缺的集合。

        代价要认账：老会话 resume 之后，history 里可能还留着指向已被清掉的运行
        目录的指针，那次 `read_file` 会拿到 `error: no such file`。这是可接受的——
        错误串会被保留在上下文里（`preserved_error_count`），模型据此回头读原文件；
        而不清理的代价是工作区无上限地长。
        """
        root = self.spill_root()
        if not root.is_dir():
            return
        entries = []
        for child in root.iterdir():
            if not child.is_dir() or child == keep_dir:
                continue
            try:
                size = sum(item.stat().st_size for item in child.rglob("*") if item.is_file())
                entries.append((child.stat().st_mtime, size, child))
            except OSError:
                continue
        total = sum(size for _, size, _ in entries)
        for _, size, child in sorted(entries):
            if total <= SPILL_KEEP_BYTES:
                break
            shutil.rmtree(child, ignore_errors=True)
            total -= size

    def _store_tool_output(self, name, text):
        """工具结果超出单条上限时全文落盘，上下文里只留预览 + 指针。

        形状照抄 Claude Code 的 L1（Tool Result Budget）：**落盘不截断**，上下文里
        留固定大小的头部预览，指针自带路径和原始大小，取回走已有的 `read_file`
        而不新增工具。和「掉出最近窗口后清内容」的区别是时机——这里在**写进
        history 的那一刻**就定形，以后每轮都是同一串文本；事后回改历史会把前缀
        缓存从那个位置往后全部作废（Claude Code 为此专门写了服务端的
        `cache_edits`，我们的后端没有）。

        返回 `(放进 history 的文本, 落盘信息 dict)`；没触发时后者是空 dict。
        任何一步失败都退回普通截断——工具结果的契约是「失败也返回字符串」，
        落盘不成功不该把一次成功的工具调用变成异常。
        """
        text = str(text)
        # 聚合型工具(`aggregates_calls`，目前只有 `run_plan`)自己管上限，这里
        # 一律放行。两个理由:
        # 一、它的结果是 N 个内层调用的转录，额度由 `plans.transcript_limit()`
        #     给(= max(4000, 3 × 单条上限))。拿单条上限去卡它，那个 3 倍额度当场
        #     作废——实测 6 次读文件的转录 3,993 token 会被砍到 1,320，同一次
        #     `read_file` 放进计划里反而看得更少，`run_plan` 变成纯负收益。
        #     **这个洞在落盘之前就有**(那时是直接 `clip()` 截断)，落盘只是照出来了。
        # 二、就算按聚合上限判，落盘也是**假承诺**:传进来的已经是
        #     `PlanResult.transcript`(裁过的那一份)，写进文件的不是全文，而指针
        #     那句话说的是「full output」。转录自己的裁剪标记已经带了「use print()」
        #     这条正确的下一步，再叠一条指向残缺文件的指针只会误导。
        #     内层每个调用各自走过这条落盘路径了，真正的大输出在那一层就已经落过盘。
        tool = self.tools.get(name) or {}
        if tool.get("aggregates_calls"):
            return text, {}
        limit = self.tool_output_limit()
        model = getattr(self.model_client, "model", None)
        full_tokens = models.count_tokens(text, model)
        if full_tokens <= limit:
            return text, {}
        # 消融开关：关掉落盘就退回这个机制出现之前的做法——直接截断丢尾巴。
        # 放在这里（而不是函数开头）是刻意的：`full_tokens <= limit` 那条快路径
        # 两个变体必须完全一致，否则 A/B 里连「有没有超限」都不可比。
        if not self.feature_enabled("tool_output_spill"):
            return clip(text, limit), {}
        # 落盘的是工具原始输出，可能带密钥（`run_shell` 打环境变量之类），
        # 而这个文件既进工作区又会被模型读回来。落盘工件一律先脱敏。
        redacted = self.redact_text(text)
        try:
            directory = self._spill_dir()
            directory.mkdir(parents=True, exist_ok=True)
            self._spill_seq += 1
            target = directory / f"{self._spill_seq:03d}-{name}.txt"
            target.write_text(redacted, encoding="utf-8")
            relative = target.relative_to(self.root).as_posix()
            self._sweep_spill_dirs(directory)
        except Exception as error:
            # 落盘失败要在工件上留痕，不能和「结果本来就没超上限」记成同一种。
            # 这两件事此前在 trace 里长得一模一样（`tool_output_spilled: False`、
            # 两个 token 字段都是 0），而含义相反：一个是机制没必要跑，一个是
            # 机制该跑却跑挂了、模型手里那份结果被截断且**不可恢复**。
            # 踩过的坑是同一类：「机制生效了」和「机制一次都没触发」分不出来。
            # `failed` 这个键让调用方把两者分开写，`spilled` 仍然只在真落盘时为真。
            return clip(text, limit), {
                "failed": True,
                "error": f"{type(error).__name__}: {error}"[:200],
                "full_tokens": full_tokens,
            }
        # 指针要给出**可以照抄的取回调用**，不能只给 token 数——`read_file` 的参数
        # 是行号，模型换算不出来，实测 4 次取回 4 次又落一次盘。这里把换算做完：
        # 每行平均 token = full_tokens / 行数，一段能装下的行数 = 预览额度 / 每行，
        # 再留一成余量。`read_file` 读落盘件时不再加行号（见 tools.tool_read_file），
        # 所以这个比例在取回时仍然成立。
        total_lines = max(1, redacted.count("\n") + 1)
        tokens_per_line = max(1.0, float(full_tokens) / total_lines)
        marker = context_manager.spill_marker(relative, full_tokens, total_lines, total_lines)
        preview_budget = max(1, limit - models.count_tokens(marker, model) - 1)
        chunk_lines = max(1, int(preview_budget * 0.9 / tokens_per_line))
        marker = context_manager.spill_marker(relative, full_tokens, total_lines, chunk_lines)
        preview = models.clip_tokens(redacted, preview_budget, model, marker="")
        return (
            "\n".join([preview, marker]),
            {
                "path": relative,
                "full_tokens": full_tokens,
                "full_lines": total_lines,
                "chunk_lines": chunk_lines,
                "preview_tokens": models.count_tokens(preview, model),
            },
        )

    def feature_enabled(self, name):
        return bool(self.feature_flags.get(str(name), False))

    def prompt(self, user_message):
        _, prompt, _ = self._build_context(user_message)
        return prompt

    def record(self, item):
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    def tool_call_id(self, index):
        """一次工具调用在 messages 里的标识，用来把 assistant 的调用和它的结果配对。

        刻意是**确定性**的而不是随机 uuid：session 会落盘、会被 resume 重放，
        随机 id 会让同一段历史每次加载都长得不一样，既没法比对，也让"两次运行
        是否可比"这件事凭空多一个变量。`run_seq` + `turn` + 轮内序号三者
        在一个会话里唯一确定一次调用。
        """
        return f"call_{int(self.current_run_seq)}_{int(self.current_turn)}_{int(index)}"

    @staticmethod
    def looks_sensitive_env_name(name):
        upper = str(name).upper()
        return any(upper == marker or upper.endswith(marker) or upper.endswith(f"_{marker}") for marker in SENSITIVE_ENV_NAME_MARKERS)

    def is_secret_env_name(self, name):
        upper = str(name).upper()
        return upper in self.secret_env_names or self.looks_sensitive_env_name(upper)

    def configured_secret_env_items(self):
        items = [
            (name, value)
            for name, value in os.environ.items()
            if str(name).upper() in self.secret_env_names and value
        ]
        items.sort(key=lambda item: item[0])
        return items

    def detected_secret_env_items(self):
        items = [
            (name, value)
            for name, value in os.environ.items()
            if self.is_secret_env_name(name) and value
        ]
        items.sort(key=lambda item: item[0])
        return items

    def secret_env_summary(self):
        names = [name for name, _ in self.configured_secret_env_items()]
        return {
            "secret_env_count": len(names),
            "secret_env_names": names,
        }

    def detected_secret_env_summary(self):
        names = [name for name, _ in self.detected_secret_env_items()]
        return {
            "secret_env_count": len(names),
            "secret_env_names": names,
        }

    def redact_text(self, text):
        text = str(text)
        for _, value in sorted(self.detected_secret_env_items(), key=lambda item: len(item[1]), reverse=True):
            text = text.replace(value, REDACTED_VALUE)
        return text

    def redact_artifact(self, value, key=None):
        if key and self.is_secret_env_name(key):
            return REDACTED_VALUE
        if isinstance(value, dict):
            return {
                str(item_key): self.redact_artifact(item_value, key=item_key)
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [self.redact_artifact(item, key=key) for item in value]
        if isinstance(value, tuple):
            return [self.redact_artifact(item, key=key) for item in value]
        if isinstance(value, str):
            redacted = self.redact_text(value)
            return redacted
        return value

    def shell_env(self):
        env = {
            name: os.environ[name]
            for name in self.shell_env_allowlist
            if name in os.environ
        }
        env["PWD"] = str(self.root)
        if "PATH" not in env and os.environ.get("PATH"):
            env["PATH"] = os.environ["PATH"]
        return env

    def prompt_metadata(self, user_message, prompt):
        _, _, metadata = self._build_context(user_message)
        return metadata

    def _context_evidence_report(self, prompt):
        """检查这一轮 prompt 里还留着哪些「回答所依赖的关键内容」。

        `declared_context_evidence` 由数据集声明、评测层盖上来（和
        `declared_tools_allowlist` 同一个套路）；没声明就返回 `None`，判分器据此
        判「不适用」而不是硬凑成通过。

        刻意做成**子串精确匹配**：一旦引入模糊匹配，这条断言就从确定性判据退化成
        另一个需要校准的判断，而 L1 的全部价值就在于它不需要模型当裁判。
        """
        declared = [str(item) for item in (self.declared_context_evidence or []) if str(item)]
        if not declared:
            return None
        missing = [item for item in declared if item not in prompt]
        return {"declared": declared, "missing": missing}

    def _build_context(self, user_message):
        """组一轮上下文，返回 `(messages, prompt, metadata)`。

        `messages` 是真正发出去的标准对话数组；`prompt` 是同一份内容压平成的
        文本，只用于度量与 trace。**预算裁剪按 token 算，不按字符**——这句话
        从前写的是「按它的字符数算」，那是混合制时期留下的，早已不成立。
        """
        refresh = self.refresh_prefix()
        self.resume_state = self.evaluate_resume_state()
        messages, prompt, metadata = self.context_manager.build_all(user_message)
        # 这里把“这轮 prompt 是怎么拼出来的”连同缓存相关状态一起记下来，
        # 后面 trace/report 才能解释清楚：为什么这一轮 prefix 变了、缓存有没有命中。
        metadata.update(
            {
                # 这里从前还落了 prefix/workspace/memory/history/request 五个
                # `*_chars` 字段。删掉是因为单位统一到 token 之后没有任何代码再读
                # 它们，而工件里留着一组字符数会被读成「预算就是按这个量的」——
                # 各段的真实口径在 `sections[*].rendered_tokens` / `budget_tokens`。
                #
                # 窗口是从哪来的必须写进工件。预算随环境变化（换 provider、
                # litellm 升级一次映射表）会让两次跑批不可比，而只记一个预算值
                # 看不出它是人定的、查表查到的、还是回落到默认值的。
                "context_window_tokens": int(getattr(self, "context_window", 0) or 0),
                "context_window_source": str(getattr(self, "context_window_source", "") or ""),
                # 预算是怎么算出来的,逐项落进工件。少了这一项,「预算触底」——窗口
                # 小到扣完四笔就不够了、只好回落到 BUDGET_FLOOR_TOKENS——在任何
                # 字段上都看不出来,而它和「预算正好是这个数」的含义完全相反。
                #
                # 注意这里记的是**派生出来的**预算。`HarnessSpec.total_budget` 和
                # `evaluator._apply_task_setup()` 都能在构造之后直接改
                # `context_manager.total_budget`,那时两者会对不上——真正生效的
                # 是同一份 metadata 里的 `prompt_budget_tokens`,以它为准。
                "context_budget_breakdown": dict(getattr(self, "context_budget_breakdown", {}) or {}),
                "tool_count": len(self.tools),
                # 注册表的**名字**，不只是个数。判分器要靠它把「声明了白名单」
                # 和「白名单真的生效了」分开：N-5 那次故障里，任务声明
                # `["read_file"]`、注册表却是全部 6 个工具，而工件上只有一个
                # `tool_count: 6`——没有任何字段能看出这份声明被无视了。
                "tool_names": sorted(self.tools),
                "tools_allowlist": (
                    list(self.declared_tools_allowlist) if self.declared_tools_allowlist else None
                ),
                # 「回答所依赖的内容,这一轮到底在不在上下文里」。
                #
                # 为什么必须在这里算、而不是事后从 trace 复算:trace 里**没有 prompt
                # 原文**(只有分段计数),所以判分器根本没有可以 grep 的对象。上下文
                # 工程的全部意义就是把对的东西放进 prompt,而在这个字段出现之前,
                # 一次 live 压力探针里 harness 把答案所在那一行丢出了上下文、模型
                # 因此答不出来,而**全部 L1 断言照样是绿的**——它们查的是路径没越界、
                # 预算没超、白名单守住了,没有一条查内容还在不在。
                "context_evidence": self._context_evidence_report(prompt),
                "workspace_docs": len(self.workspace.project_docs),
                "recent_commits": len(self.workspace.recent_commits),
                "prefix_hash": self.prefix_state.hash,
                "prompt_cache_key": self.prefix_state.hash,
                "workspace_fingerprint": self.prefix_state.workspace_fingerprint,
                "tool_signature": self.prefix_state.tool_signature,
                "workspace_changed": refresh["workspace_changed"],
                "prefix_changed": refresh["prefix_changed"],
                "prompt_cache_supported": bool(getattr(self.model_client, "supports_prompt_cache", False)),
                "native_tool_calls": self.native_tool_calls,
                "text_protocol_tool_calls": self.text_protocol_tool_calls,
                "resume_status": self.resume_state.get("status", CHECKPOINT_NONE_STATUS),
                "stale_summary_invalidations": int(self.resume_state.get("stale_summary_invalidations", 0)),
                "stale_paths": list(self.resume_state.get("stale_paths", [])),
                "runtime_identity_mismatch_fields": list(self.resume_state.get("runtime_identity_mismatch_fields", [])),
            }
        )
        metadata.update(self.detected_secret_env_summary())
        return messages, prompt, metadata

    def emit_trace(self, task_state, event, payload=None):
        payload = self.redact_artifact(payload or {})
        payload["event"] = event
        payload["created_at"] = now()
        # trace 是运行中的逐事件时间线，适合回答“这一轮 agent 到底做了什么”。
        # 每条事件都盖上 session -> run -> turn 三级身份：
        # session_id/run_seq 让散在 runs/ 下的事件能重新拼成会话时间线；
        # turn 让同一轮的 prompt_built / model_parsed / tool_executed 可以
        # 按字段配对，而不是靠它们在文件里的先后顺序。
        payload["session_id"] = getattr(task_state, "session_id", "") or str(self.session.get("id", ""))
        payload["run_id"] = task_state.run_id
        payload["run_seq"] = int(self.current_run_seq)
        payload["turn"] = int(self.current_turn)
        self.run_store.append_trace(task_state, payload)
        return payload

    def capture_workspace_snapshot(self):
        snapshot = {}
        for path in self.root.rglob("*"):
            try:
                relative_parts = path.relative_to(self.root).parts
            except ValueError:
                continue
            if any(part in IGNORED_PATH_NAMES for part in relative_parts):
                continue
            if not path.is_file():
                continue
            try:
                snapshot[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                continue
        return snapshot

    @staticmethod
    def diff_workspace_snapshots(before, after):
        changed_paths = []
        summaries = []
        all_paths = sorted(set(before) | set(after))
        for path in all_paths:
            if before.get(path) == after.get(path):
                continue
            changed_paths.append(path)
            if path not in before:
                summaries.append(f"created:{path}")
            elif path not in after:
                summaries.append(f"deleted:{path}")
            else:
                summaries.append(f"modified:{path}")
        return changed_paths, summaries

    def create_checkpoint(self, task_state, user_message, trigger):
        state = self.checkpoint_state()
        current = self.current_checkpoint()
        checkpoint_id = "ckpt_" + uuid.uuid4().hex[:8]
        key_files = []
        freshness = {}
        for path in self.memory.to_dict()["working"]["recent_files"]:
            file_freshness = memorylib.file_freshness(path, self.root)
            freshness[path] = file_freshness
            key_files.append({"path": path, "freshness": file_freshness})
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": current.get("checkpoint_id", "") if current else "",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "created_at": now(),
            "current_goal": str(user_message),
            "completed": [task_state.final_answer] if task_state.final_answer else [],
            "excluded": [],
            "current_blocker": "" if str(task_state.stop_reason or "") in ("", "final_answer_returned") else str(task_state.stop_reason),
            "next_step": self.infer_next_step(task_state),
            "key_files": key_files,
            "freshness": freshness,
            "summary": f"{trigger}: {clip(str(user_message), 120)}",
            "runtime_identity": self.current_runtime_identity(),
        }
        state["items"][checkpoint_id] = checkpoint
        state["current_id"] = checkpoint_id
        task_state.checkpoint_id = checkpoint_id
        self.session["runtime_identity"] = checkpoint["runtime_identity"]
        self.session_path = self.session_store.save(self.session)
        return checkpoint

    def infer_next_step(self, task_state):
        if task_state.status == "completed":
            return "No next step recorded."
        if task_state.stop_reason == "step_limit_reached":
            return "Resume from the latest checkpoint and continue the task."
        if task_state.last_tool:
            return f"Decide the next action after {task_state.last_tool}."
        return "Continue the task from the latest checkpoint."

    def update_memory_after_tool(self, name, args, result):
        """把少量高价值工具结果沉淀到 working memory。

        为什么存在：
        并不是每个工具结果都值得长期带进下一轮 prompt。完整结果已经进了
        `history`，这里只挑少量“下一轮大概率还会用到”的事实做提纯，
        例如最近读写过哪些文件、某个文件读出来的短摘要。

        输入 / 输出：
        - 输入：工具名 `name`、参数 `args`、执行结果 `result`
        - 输出：无显式返回值，副作用是更新 `self.memory`

        在 agent 链路里的位置：
        它发生在 `run_tool()` 真正执行完工具之后、下一轮 prompt 组装之前。
        也就是说：工具结果先进入完整历史，再由这个函数择优沉淀成轻量记忆。
        """
        if not self.feature_enabled("memory"):
            return
        path = args.get("path")
        if not path:
            return

        canonical_path = self.memory.canonical_path(path)
        # 不是所有工具结果都进入工作记忆。
        # 读文件会生成摘要；写文件/patch 会让旧摘要失效，因为它们可能过期了。
        if name in {"read_file", "write_file", "patch_file"}:
            self.memory.remember_file(canonical_path)
        if name == "read_file":
            summary = memorylib.summarize_read_result(result)
            self.memory.set_file_summary(canonical_path, summary)
            self.memory.append_note(summary, tags=(canonical_path,), source=canonical_path)
        elif name in {"write_file", "patch_file"}:
            self.memory.invalidate_file_summary(canonical_path)

    def note_tool(self, name, args, result):
        self.update_memory_after_tool(name, args, result)

    def record_process_note_for_tool(self, name, metadata):
        status = str(metadata.get("tool_status", "")).strip()
        if status not in {"partial_success", "error", "rejected"}:
            return
        affected_paths = [str(path).strip() for path in metadata.get("affected_paths", []) if str(path).strip()]
        path_text = ", ".join(affected_paths) or "workspace"
        if status == "partial_success":
            text = f"{name} partial_success on {path_text}; inspect diff before retry"
        elif status == "error":
            text = f"{name} error on {path_text}; check the failure before retry"
        else:
            text = f"{name} rejected; choose a different action before retry"
        tags = ["process", status, *affected_paths]
        self.memory.append_note(text, tags=tuple(tags), source=name, kind="process")
        self.session["memory"] = self.memory.to_dict()

    def reject_durable_reason(self, note_text):
        text = str(note_text or "").strip()
        lowered = text.lower()
        if not text:
            return "empty"
        if REDACTED_VALUE in text or SECRET_SHAPED_TEXT_PATTERN.search(text):
            return "secret_shaped"
        checkpoint_like_prefixes = (
            "current goal",
            "current blocker",
            "next step",
            "current phase",
            "key files",
            "freshness",
            "当前目标",
            "当前卡点",
            "下一步",
            "当前阶段",
            "关键文件",
            "已完成",
            "已排除",
        )
        if any(lowered.startswith(prefix) for prefix in checkpoint_like_prefixes):
            return "transient_task_state"
        if re.search(r"(?i)\b(stdout|stderr|traceback|exit_code)\b", text) or len(text) > 220:
            return "noisy_output"
        return ""

    def extract_durable_promotions(self, user_message, final_answer):
        user_text = str(user_message or "")
        if not (DURABLE_MEMORY_INTENT_PATTERN.search(user_text) or DURABLE_MEMORY_INTENT_ZH_PATTERN.search(user_text)):
            return [], []
        promotions = []
        rejections = []
        for line in str(final_answer or "").splitlines():
            text = line.strip()
            if not text or REDACTED_VALUE in text:
                continue
            for topic, pattern in DURABLE_MEMORY_LINE_PATTERNS:
                match = pattern.match(text)
                if not match:
                    continue
                note_text = match.group(1).strip()
                if note_text:
                    reason = self.reject_durable_reason(note_text)
                    if reason:
                        rejections.append(f"{topic}:{reason}")
                        break
                    promotions.append((topic, note_text))
                break
        return promotions, rejections

    def promote_durable_memory(self, user_message, final_answer):
        promotions, rejections = self.extract_durable_promotions(user_message, final_answer)
        promoted, superseded = self.memory.promote_durable(promotions)
        self.session["memory"] = self.memory.to_dict()
        self.last_durable_promotions = promoted
        self.last_durable_rejections = rejections
        self.last_durable_superseded = superseded
        return promoted, rejections, superseded

    def ask(self, user_message):
        """执行一次完整的 agent 回合，直到产出最终答案或命中停止条件。

        为什么存在：
        `ask()` 是整个 runtime 的总调度器。它把“用户提一个请求”扩展成一条
        可持续推进的控制循环：记录会话、组 prompt、调用模型、执行工具、
        写 trace/report、更新状态，直到模型给出最终答案或系统主动停下。

        输入 / 输出：
        - 输入：`user_message`，即用户这一次的任务描述
        - 输出：字符串形式的最终回答；如果中途达到步数上限或重试上限，
          返回的是一条停止原因说明

        在 agent 链路里的位置：
        它是 CLI 和底层工具/模型之间的核心桥梁。CLI 收到用户输入后基本只做
        一件事：调用 `agent.ask()`。而 `ask()` 内部再去驱动 `ContextManager`
        组 prompt、`model_client.complete()` 调模型、`run_tool()` 执行动作。
        如果新人想理解 coding-for-me 是怎么“从一句话跑成一个 agent 流程”的，
        这里就是最关键的入口。
        """
        run_started_at = time.monotonic()
        self.memory.set_task_summary(user_message)
        self.record({"role": "user", "content": user_message, "created_at": now()})

        self.session["run_seq"] = int(self.session.get("run_seq", 0) or 0) + 1
        self.current_run_seq = int(self.session["run_seq"])
        self.current_turn = 0
        task_state = TaskState.create(
            run_id=self.new_run_id(),
            task_id=self.new_task_id(),
            user_request=user_message,
            session_id=str(self.session.get("id", "")),
            run_seq=self.current_run_seq,
        )
        # 这里只是个初值：`self.resume_state` 是构造期算的，而调用方完全可能在
        # 构造之后才把 session 改脏（`BenchmarkEvaluator._apply_task_setup()` 就是
        # 这个顺序）。真正的运行级取值在首轮组完 prompt 之后钉死，见下面的循环。
        task_state.resume_status = self.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        self.current_task_state = task_state
        self.current_run_dir = self.run_store.start_run(task_state)
        self.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),
            },
        )

        tool_steps = 0
        attempts = 0
        max_attempts = max(self.max_steps * 3, self.max_steps + 4)

        # 这是 agent 的主循环，可以按“感知 -> 决策 -> 行动 -> 记录”来理解：
        # 1. 感知：重新组 prompt，把当前状态整理给模型看
        # 2. 决策：让模型返回一个工具调用，或一个最终答案
        # 3. 行动：如果是工具调用，就执行工具
        # 4. 记录：把结果写回 history / task_state / trace / memory
        # 然后进入下一轮，直到停机条件满足
        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1
            task_state.record_attempt()
            self.current_turn = attempts
            self.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()
            summary_covered_before = self._summary_covered()
            messages, _, prompt_metadata = self._build_context(user_message)
            if attempts == 1:
                # **只在首轮钉一次。** `task_state.resume_status` 回答的是「这次运行
                # 从什么状态起步」，是运行级事实；而 `prompt_metadata["resume_status"]`
                # 是逐轮事实，重新锚定之后回到 full-valid 是对的。两者别混用：
                # report 里存的 prompt_metadata 是**最后一轮**的，拿它判「这次运行有没有
                # 从过期 checkpoint 恢复」，只有恰好一轮就结束的运行才判得对。
                task_state.resume_status = str(
                    prompt_metadata.get("resume_status", CHECKPOINT_NONE_STATUS)
                )
            self.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            # 组上下文的过程中会话摘要推进过覆盖点 → 单独写一条事件。判据取**覆盖点
            # 本身**,不取 `context_pressure.session_summary.compactions`:后者是这一轮
            # 循环跑了几次,而这条事件回答的是「模型看得见的东西被换掉了多少」。
            if self._summary_covered() > summary_covered_before:
                self.emit_context_compacted("auto", summary_covered_before)
            if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
                checkpoint = self.create_checkpoint(task_state, user_message, trigger="freshness_mismatch")
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "freshness_mismatch",
                    },
                )
            elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                self.emit_trace(
                    task_state,
                    "runtime_identity_mismatch",
                    {
                        "fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", [])),
                    },
                )
                checkpoint = self.create_checkpoint(task_state, user_message, trigger="workspace_mismatch")
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "workspace_mismatch",
                    },
                )
            if prompt_metadata.get("budget_reductions"):
                checkpoint = self.create_checkpoint(task_state, user_message, trigger="context_reduction")
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "context_reduction",
                    },
                )
            self.emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )
            prompt_cache_key = None
            prompt_cache_retention = None
            if getattr(self.model_client, "supports_prompt_cache", False):
                # 只有后端明确支持时，才把稳定前缀的 hash 作为 cache key 发出去。
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"
            model_started_at = time.monotonic()
            raw = self.model_client.complete(
                messages,
                self.max_new_tokens,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
                tools=toolkit.to_openai_function_specs(self.tools) if self.native_tool_calls else None,
                on_token=self.on_token,
            )
            completion_metadata = dict(getattr(self.model_client, "last_completion_metadata", {}) or {})
            if completion_metadata:
                # 把后端返回的 usage/cache 统计并回 prompt_metadata，
                # 方便统一写入 report 和 trace。
                prompt_metadata.update(completion_metadata)
            self.last_completion_metadata = completion_metadata
            self.last_prompt_metadata = prompt_metadata
            kind, payload = self.parse(raw)
            # 协议漂移的观测点：prompt 只教标准 function-calling，所以这里但凡
            # 是靠标签解析出来的工具调用，都说明模型没走我们给的接口。之前那次
            # 19:17 的分裂只能靠真实后端跑基准才偶然看见，现在它是 trace 和
            # report 里的一个数字。
            text_protocol_tool_call = kind == "tool" and not (isinstance(raw, dict) and raw.get("tool_calls"))
            if text_protocol_tool_call:
                self.text_protocol_tool_calls += 1
                # prompt_metadata 是在模型调用**之前**组好的，这里补一次，
                # 让写进 report 的计数包含当前这一轮而不是滞后一轮。
                prompt_metadata["text_protocol_tool_calls"] = self.text_protocol_tool_calls
            self.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    # 只有 retry 有归因；其余 kind 恒为空串。归因码见 RETRY_REASON_*，
                    # 配合 completion_metadata 里的 finish_reason 才能分清「模型没说话」
                    # 和「推理阶段就被 max_tokens 截断了」。
                    "retry_reason": getattr(payload, "reason", "") if kind == "retry" else "",
                    "text_protocol_tool_call": text_protocol_tool_call,
                    "completion_metadata": completion_metadata,
                    "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                },
            )

            if kind == "tool":
                # 模型伴随工具调用产生的说明文字也要进 history。丢掉它等于让模型
                # 每轮失忆：下一轮它只看得见一串工具结果，看不见自己当时打算干什么、
                # 已经确认过什么。实测一次真实运行里 14 次工具调用只留下 2 条
                # assistant 记录（还都是 runtime 自己写回的），模型因此反复回读同一个
                # 文件——15 次模型调用里 9 次是重复劳动。它本来就为这段文字付了
                # 每轮 94~232 个 output token，丢掉纯属白花。
                narration = str(raw.get("text") or "").strip() if isinstance(raw, dict) else ""
                # 这条 assistant 记录**即使没有说明文字也要写**，因为它同时承载了
                # 「这一轮发了哪几个调用」这个结构事实。组 messages 时，每条 tool
                # 消息都要有一个发起它的 assistant 消息与之配对；少了这条，一轮多
                # 调用就退化成「几个来路不明的工具结果」，回合边界重新丢失，
                # 而回合边界正是 T2-1 要修的那个东西。
                call_ids = [self.tool_call_id(index) for index in range(len(payload))]
                self.record(
                    {
                        "role": "assistant",
                        "content": narration,
                        "tool_calls": [
                            {"id": call_id, "name": call.get("name", ""), "args": call.get("args", {})}
                            for call_id, call in zip(call_ids, payload)
                        ],
                        "created_at": now(),
                    }
                )

                # payload 恒为列表；一轮可以有多个调用，按模型发来的顺序逐个执行。
                #
                # 三条刻意的规则：
                # 1. **每个调用各自计一步**。步数预算限制的是"做了多少事"，不是
                #    "往返了几次"——否则模型一轮发 20 个调用就能绕过 max_steps，
                #    连带绕过 read_only 与审批的粒度。省下来的是模型往返（真正的
                #    时间成本），预算语义因此不必跟着改。
                # 2. **一个失败不打断后面的**。`run_tool()` 任何失败都返回字符串
                #    而不抛异常，所以后续调用照常执行，模型下一轮能一次看到全部反馈。
                # 3. **预算在轮内用完就停下**，剩余调用不执行，并写一条 notice 告诉
                #    模型哪些没跑——静默丢弃会让它以为那些都做过了。
                for index, call in enumerate(payload):
                    if tool_steps >= self.max_steps:
                        self.record_unexecuted_calls(
                            task_state, payload, index, call_ids, reason="step_budget_exhausted"
                        )
                        self.run_store.write_task_state(task_state)
                        break
                    tool_steps += 1
                    name = call.get("name", "")
                    args = call.get("args", {})
                    # `run_plan` 是一批调用的容器，不是一次调用。它的记账走
                    # execute_plan()：内层每个调用自己 record_tool + 写 trace，
                    # 这里只把额外吃掉的步数加回预算。一个调用都没跑成时（计划
                    # 非法）才由这里补记一步，好让 task_state.tool_steps 与控制
                    # 循环的 tool_steps 始终对得上。
                    is_plan = name == "run_plan"
                    if not is_plan:
                        task_state.record_tool(name)
                    self._active_task_state = task_state
                    self._steps_available = max(0, self.max_steps - tool_steps)
                    self._extra_tool_steps = 0
                    self._last_plan_result = None
                    tool_started_at = time.monotonic()
                    result = self.run_tool(name, args)
                    if is_plan:
                        tool_steps += int(self._extra_tool_steps)
                        if not (self._last_plan_result and self._last_plan_result.calls):
                            task_state.record_tool(name)
                    self.record(
                        {
                            "role": "tool",
                            "name": name,
                            "args": args,
                            "content": result,
                            "call_id": call_ids[index],
                            "created_at": now(),
                            # 这次调用实际改到了哪些文件。取的是**工作区快照前后
                            # 的 sha256 差异**，不是 args 里的 path——三个理由：
                            # `run_shell` 改了哪个文件在 args 里读不出来；一段
                            # `run_plan` 在 history 里只有一条消息，内层那次
                            # patch_file 的路径同样不在 args 里；而失败的写、或者
                            # 写进去内容完全相同的写，args 有 path 但文件没变，
                            # 按 args 判会把一条其实仍然新鲜的读误判成过期。
                            #
                            # 空列表也照写，不省略这个键：`_written_paths()` 靠
                            # 「有没有这个键」区分「runtime 说了、就是没改」和
                            # 「老会话没这个字段、只能从 args 猜」。
                            "wrote": list(
                                (self._last_tool_result_metadata or {}).get("affected_paths") or ()
                            ),
                        }
                    )
                    self.run_store.write_task_state(task_state)
                    # 计划写的是 `plan_executed`：内层调用已经各自写过一条
                    # `tool_executed`，外层再写一条会让同一次执行被 trace 数两遍，
                    # 所有按调用聚合的指标（calls_per_turn、L1 各条断言）都会翻倍。
                    plan_result = self._last_plan_result if is_plan else None
                    self.emit_trace(
                        task_state,
                        "plan_executed" if is_plan else "tool_executed",
                        {
                            "name": name,
                            "args": args,
                            "result": clip(result, 500),
                            "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                            # 同一轮里的第几个调用。没有它，trace 里一轮的多条
                            # tool_executed 只能靠文件顺序区分，而顺序不是承诺。
                            "call_index": index,
                            "call_count": len(payload),
                            **(
                                {
                                    "plan_calls": len(plan_result.calls),
                                    "plan_ops": plan_result.ops,
                                    "plan_stopped_reason": plan_result.stopped_reason,
                                    # 「结果不进上下文」省了多少，只能靠这三个数
                                    # 验收：省下的字节 = result_bytes -
                                    # transcript_chars。没有它们，真省了和没省
                                    # 在工件上长得一模一样。
                                    "plan_result_bytes": plan_result.result_bytes,
                                    # 度量按 token：转录直接进下一轮 prompt，
                                    # 而预算按 token 判，两边同单位才对得上。
                                    "plan_transcript_tokens": models.count_tokens(
                                        plan_result.transcript,
                                        getattr(self.model_client, "model", None),
                                    ),
                                    "plan_transcript_clipped": (
                                        len(plan_result.transcript_full) > len(plan_result.transcript)
                                    ),
                                    "plan_results_echoed": plan_result.echo_results,
                                }
                                if plan_result
                                else {}
                            ),
                            **dict(self._last_tool_result_metadata or {}),
                        },
                    )
                    checkpoint = self.create_checkpoint(task_state, user_message, trigger="tool_executed")
                    self.run_store.write_task_state(task_state)
                    self.emit_trace(
                        task_state,
                        "checkpoint_created",
                        {
                            "checkpoint_id": checkpoint["checkpoint_id"],
                            "trigger": "tool_executed",
                        },
                    )
                continue

            if kind == "retry":
                self.record({"role": "assistant", "content": payload, "created_at": now()})
                self.run_store.write_task_state(task_state)
                continue

            raw_text = raw["text"] if isinstance(raw, dict) else str(raw)
            final = (payload or raw_text).strip()
            self.record({"role": "assistant", "content": final, "created_at": now()})
            task_state.finish_success(final)
            self.promote_durable_memory(user_message, final)
            checkpoint = self.create_checkpoint(task_state, user_message, trigger="run_finished")
            self.run_store.write_task_state(task_state)
            self.emit_trace(
                task_state,
                "checkpoint_created",
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "trigger": "run_finished",
                },
            )
            self.emit_trace(
                task_state,
                "run_finished",
                {
                    "status": task_state.status,
                    "stop_reason": task_state.stop_reason,
                    "final_answer": final,
                    "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                },
            )
            self.run_store.write_report(task_state, self.redact_artifact(self.build_report(task_state)))
            return final

        if attempts >= max_attempts and tool_steps < self.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            task_state.stop_retry_limit(final)
        else:
            final = "Stopped after reaching the step limit without a final answer."
            task_state.stop_step_limit(final)
        self.record({"role": "assistant", "content": final, "created_at": now()})
        self.promote_durable_memory(user_message, final)
        self.run_store.write_task_state(task_state)
        checkpoint = self.create_checkpoint(task_state, user_message, trigger=task_state.stop_reason or "run_stopped")
        self.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": task_state.stop_reason or "run_stopped",
            },
        )
        self.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        self.run_store.write_report(task_state, self.redact_artifact(self.build_report(task_state)))
        return final

    def execute_plan(self, source):
        """驱动一段受限编排计划：解释器负责语法，这里负责闸口、预算与 trace。

        为什么要在 runtime 里而不是在 `plans.py` 里：`plans.py` 刻意不认识 agent，
        它只知道「有个 `call_tool(name, args) -> str` 可以调」。三件必须由这里
        承担的事，都是控制循环层面的：

        1. **每个内层调用照样走 `run_tool()`。** 存在性、参数校验、重复检测、
           审批、只读、脱敏、记忆回写、工具白名单，一个都不少。编排不是绕过
           闸口的旁路，它只是替模型省掉一次约 17 秒的模型往返。
        2. **每个内层调用各自计一步。** 语义和「一轮发多个工具调用」完全一致：
           预算限制的是做了多少事，不是往返了几次。否则一段 `for` 循环就能在
           一步之内做完二十件事，`max_steps` 连同它保护的一切都失效。`run_plan`
           这一步本身覆盖第一个调用，之后每个调用再吃一步——所以一段 N 个调用
           的计划正好消耗 max(1, N) 步，和把它们拆成 N 个普通调用一样贵。
        3. **每个内层调用各写一条 `tool_executed`。** 不这么做的话，L1 的
           `path_confined` / `read_before_patch` / `tools_allowlist_respected`
           全都看不见计划里发生了什么——编排会变成整套评测体系的一个盲区，
           而盲区在报告里长得和"通过"一模一样。事件上多一个 `via: "run_plan"`，
           这样"模型自己发的调用"和"计划里派生的调用"仍然分得开。

        计划本身的成败不进 `tool_executed`：外层那条事件叫 `plan_executed`
        （由 `ask()` 写），否则同一次工具执行会被 trace 数两遍。
        """
        task_state = self._active_task_state
        budget = int(self._steps_available)
        call_index = [0]

        def call_tool(name, args):
            tool_started_at = time.monotonic()
            result = self.run_tool(name, args)
            if task_state is not None:
                task_state.record_tool(name)
                self.emit_trace(
                    task_state,
                    "tool_executed",
                    {
                        "name": name,
                        "args": args,
                        "result": clip(result, 500),
                        "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                        # 轮内序号。计划派生的调用和模型自己发的调用共用这个字段，
                        # 靠 `via` 区分来源——同一轮里两种都有时序号会重号，那是
                        # 刻意的：序号标的是"在自己这一批里的第几个"。
                        "call_index": call_index[0],
                        "via": "run_plan",
                        **dict(self._last_tool_result_metadata or {}),
                    },
                )
            call_index[0] += 1
            return result

        result = plans.PlanResult()
        # 转录的聚合上限跟着工具结果上限走（原设计就是「三个工具调用的额度」）。
        # 不跟着走的话，1M 档下单个结果能有 14,791 而整段转录仍卡在 4,000，
        # 同一次 read_file 放进计划里反而看得更少。
        result.transcript_limit = plans.transcript_limit(self.tool_output_limit())
        failure = ""
        try:
            result = plans.execute_plan(
                source, call_tool, toolkit.plan_callable_tools(self), budget, result=result
            )
        except plans.PlanError as exc:
            # 能走到这里的只剩静态检查看不出来的东西——主要是没赋值就使用的变量
            # （作用域要跟着 for 和推导式走，静态判太容易误伤）。`validate_tool`
            # 已经用 check_plan 把语法和工具名那一层挡在前面了。
            failure = str(exc)
        if failure and not result.calls:
            # 一个调用都没跑成：抛出去，让 run_tool 的兜底把它记成 tool_status
            # "error"。返回字符串的话这次执行会被记成"成功"，而 trace 上的
            # 「成功但结果是一句报错」是最难看出问题的一种形状。
            self._extra_tool_steps = 0
            self._last_plan_result = None
            raise ValueError(f"invalid plan: {failure}")
        self._last_plan_result = result
        self._extra_tool_steps = max(0, len(result.calls) - 1)
        header = f"plan executed {len(result.calls)} tool call(s)."
        if not result.echo_results and result.calls:
            # 规则是隐式的（"写了 print 就不回显"），所以必须在这里说出来。
            # 不说的话，模型看到的是调用清单后面少了结果，最可能的反应是把
            # 同一批调用原样再发一遍——正好撞上重复调用检测，白烧一个往返。
            #
            # **必须明说"这就是全部输出"。** live 验证里上一版结尾写的是
            # "Print what you need next time."，模型在已经印对了的情况下把它读成
            # "你没拿全，再来一次"，于是把整个 fan-out 重跑了一遍。这句话的作用
            # 是**关闭**这一轮，不是催下一轮。
            header += (
                " The plan used print(), so tool results were not echoed. "
                "The printed output below is complete; nothing was cut off."
            )
        if failure:
            # 跑了一半才失败：已经执行的调用必须原样回给模型。丢掉它们等于
            # 让模型以为那些都没发生过，而其中可能包含已经落盘的写操作。
            result.stopped_reason = result.stopped_reason or "plan_error"
            header += f" The plan then failed: {failure}"
        if result.stopped_reason == "step_budget_exhausted":
            header += (
                " The step budget ran out part way through, so the rest of the plan did not run. "
                "Re-issue what is still needed in your next reply."
            )
        return "\n".join([header, result.transcript]) if result.transcript else header

    def run_tool(self, name, args):
        """执行一次工具调用，并在执行前后套上完整护栏。

        为什么存在：
        在 agent 系统里，真正危险的不是“模型会不会想调用工具”，而是
        “平台有没有在执行前把边界守住”。这个函数就是工具层的总闸口：
        所有工具调用都必须先经过它，不能让模型直接碰到底层函数。

        输入 / 输出：
        - 输入：工具名 `name`，参数字典 `args`
        - 输出：字符串结果。无论是成功结果还是错误信息，都会统一返回文本，
          这样模型下一轮都能继续消费这份反馈。

        在 agent 链路里的位置：
        它位于 `ask()` 的“模型决定要调用工具”之后，是控制循环里真正把模型
        意图落到外部世界的一步。因此这里串起了几乎所有安全与可控设计：
        工具是否存在、参数是否合法、是否重复、是否需要审批、执行结果是否裁剪、
        是否需要回写记忆。
        """
        # 工具执行不是“直接调函数”，而是一条带护栏的流水线：
        # 工具是否存在 -> 参数是否合法 -> 是否重复调用 -> 是否通过审批
        # -> 真正执行 -> 更新记忆。
        # parse() 已经归一过一次；这里再来一次是为了直接调用 run_tool() 的那些
        # 入口（测试、metrics.py 的安全场景复现）也拿到同样的语义。归一是幂等的。
        args = toolkit.coerce_tool_args(name, args)
        tool = self.tools.get(name)
        if tool is None:
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "unknown_tool",
                "security_event_type": "",
                "risk_level": "high",
                "read_only": False,
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            # 带上当前注册表：模型看到 `unknown tool` 时最需要知道的是"那我能调
            # 什么"。工具清单会被白名单裁掉一部分，而模型的先验里那些工具是存在
            # 的（它在别的仓库、别的会话里见过），只说"没这个工具"它多半会换一个
            # 同样不在表里的名字再试一次。
            return f"error: unknown tool '{name}'. Available tools: {', '.join(sorted(self.tools)) or '(none)'}"
        try:
            self.validate_tool(name, args)
        except Exception as exc:
            example = self.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample arguments: {example}"
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "invalid_arguments",
                "security_event_type": security_event_type,
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return message
        if self.repeated_tool_call(name, args):
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "repeated_identical_call",
                "security_event_type": "",
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            # 措辞要在两种场景下都读得通：跨轮反复发同一调用，以及**同一轮内**
            # 把同一个调用发了三遍（多调用之后才会出现的形状）。原来的说法是
            # "repeated identical tool call"，批内场景下读起来像 runtime 出了 bug，
            # 而实际上判断是对的——那确实是空转。这里改成陈述事实 + 给下一步（P5）。
            return (
                f"error: {name} was already called twice with these exact arguments and the result "
                "will not change. Use different arguments, a different tool, or return a final answer."
            )
        if tool["risky"] and not self.approve(name, args):
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "approval_denied",
                "security_event_type": "read_only_block" if self.read_only else "approval_denied",
                "risk_level": "high",
                "read_only": False,
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            # 只说"被拒了"不够：模型无法区分"这次没批准，换个说法再试"和
            # "这个能力本轮根本不开放"，实测的反应就是换个工具往同一个文件写
            # （patch_file 被只读态挡住后改用 run_shell echo）。这里把原因和
            # 下一步都说清楚。只读是配置决定的，重试一万次也不会变。
            if self.read_only:
                return (
                    f"error: approval denied for {name}: this agent is running read-only and cannot "
                    "modify the workspace. Do not retry with another tool; report what you found instead."
                )
            return (
                f"error: approval denied for {name}: the user declined this action. "
                "Do not retry it; continue with what you are allowed to do, or explain what you need."
            )
        # `run_plan` 自己不需要审批（risky=False，审批发生在每个内层调用上），
        # 但它可能通过内层调用改到工作区。不拍快照的话，trace 里那条记录会说
        # 「没改动过任何文件」，而那是假的。
        snapshots_needed = bool(tool["risky"] or tool.get("aggregates_calls"))
        before_snapshot = self.capture_workspace_snapshot() if snapshots_needed else {}
        after_snapshot = before_snapshot
        try:
            result, spill = self._store_tool_output(name, tool["run"](args))
            after_snapshot = self.capture_workspace_snapshot() if snapshots_needed else before_snapshot
            affected_paths, diff_summary = self.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            tool_status = "ok"
            tool_error_code = ""
            if name == "run_shell":
                match = re.search(r"exit_code:\s*(-?\d+)", result)
                exit_code = int(match.group(1)) if match else 0
                if exit_code != 0 and workspace_changed:
                    tool_status = "partial_success"
                    tool_error_code = "tool_partial_success"
                elif exit_code != 0:
                    tool_status = "error"
                    tool_error_code = "tool_failed"
            self.update_memory_after_tool(name, args, result)
            self._last_tool_result_metadata = {
                "tool_status": tool_status,
                "tool_error_code": tool_error_code,
                "security_event_type": "",
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "affected_paths": affected_paths,
                "workspace_changed": workspace_changed,
                "workspace_fingerprint": self.workspace.fingerprint(),
                "diff_summary": diff_summary,
            }
            # S2：落盘触发了没有，必须在工件上看得见。零值也照常写——
            # 「一次都没触发」和「这个字段不存在」是两回事，后者会被读成没问题。
            self._last_tool_result_metadata.update(
                {
                    # 真落盘才算 spilled：失败那支同样返回非空 dict（它要捎带
                    # 失败原因和原始大小），所以这里不能只写 `bool(spill)`。
                    "tool_output_spilled": bool(spill) and not spill.get("failed"),
                    "tool_output_spill_failed": bool(spill.get("failed")),
                    "tool_output_spill_error": str(spill.get("error", "")),
                    "tool_output_spill_path": spill.get("path", ""),
                    # 失败时这个数照样写：它说的是「本来要落盘多大一份」，
                    # 也就是这次静默截断到底丢了多少。
                    "tool_output_full_tokens": int(spill.get("full_tokens", 0)),
                    # 落盘之后真正留在上下文里的量。省下多少 = full - kept,
                    # 和 run_plan 的 `result_bytes - transcript_tokens` 同一个
                    # 口径:没有这个差值,「真省了」和「什么都没省」在工件上
                    # 分不出来。
                    "tool_output_kept_tokens": int(spill.get("preview_tokens", 0)),
                }
            )
            # 委派的上下文记账，由 `tools.tool_delegate()` 现算好放在这里。合并进
            # 同一份 metadata 而不是另发一条事件：一次 delegate 就是一次工具调用，
            # 两条事件会让同一次执行被 trace 数两遍（`run_plan` 的内层调用踩过）。
            if name == "delegate":
                self._last_tool_result_metadata.update(
                    dict(getattr(self, "_last_delegate_stats", {}) or {})
                )
                self._last_delegate_stats = {}
            self.record_process_note_for_tool(name, self._last_tool_result_metadata)
            return result
        except Exception as exc:
            after_snapshot = self.capture_workspace_snapshot() if snapshots_needed else before_snapshot
            affected_paths, diff_summary = self.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            self._last_tool_result_metadata = {
                "tool_status": "partial_success" if workspace_changed else "error",
                "tool_error_code": "tool_partial_success" if workspace_changed else "tool_failed",
                "security_event_type": security_event_type,
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "affected_paths": affected_paths,
                "workspace_changed": workspace_changed,
                "workspace_fingerprint": self.workspace.fingerprint(),
                "diff_summary": diff_summary,
            }
            self.record_process_note_for_tool(name, self._last_tool_result_metadata)
            return f"error: tool {name} failed: {exc}"

    # 一轮里没轮到执行就被步数预算卡住的调用，回给模型的结果文本。
    #
    # 为什么每个调用各写一条、而不是写一条汇总通知：模型发出的是一组调用，
    # 它需要知道**具体哪一个**没跑。写成汇总的话，模型看到的是「3 个调用、
    # 2 条结果、外加一句话说还有一个没跑」，配对得靠它自己推理。
    # 这也是 Anthropic 明确要求的形状（每个 tool_use 都要有配对的 tool_result，
    # 没执行的也要回并标成错误），Claude Code 里有专门函数维持同一个不变量。
    UNEXECUTED_CALL_RESULT = (
        "error: not executed — the step budget was exhausted earlier in this turn. "
        "Re-issue this call in your next reply if it is still needed."
    )

    def record_unexecuted_calls(self, task_state, payload, start_index, call_ids, reason):
        """把一批里从 `start_index` 起没能执行的调用，逐个写回 history 与 trace。

        三条边界是刻意的，破坏任何一条都会污染既有度量：

        1. **不计步**——它没执行，不该占 `tool_steps`。
        2. **不进 `task_state.record_tool()`**——否则任务状态里会出现从未发生的工具。
        3. **trace 写 `tool_skipped` 而不是 `tool_executed`**——混进后者会让
           `calls_per_turn` 以及所有 L1 断言把没跑的调用算成跑了。

        history 条目上打 `executed: False` 标记，供两处消费：`repeated_tool_call()`
        不能把它当成一次真实调用（否则模型下一轮如实重发会被判成重复而拦掉，
        正好堵死我们想要的自愈路径），`ContextManager` 的 read 去重也不能让它
        顶掉同一区间那次真实读到的内容。
        """
        for offset, call in enumerate(payload[start_index:]):
            index = start_index + offset
            name = str(call.get("name", ""))
            args = call.get("args", {})
            self.record(
                {
                    "role": "tool",
                    "name": name,
                    "args": args,
                    "content": self.UNEXECUTED_CALL_RESULT,
                    "call_id": call_ids[index],
                    "created_at": now(),
                    "executed": False,
                }
            )
            self.emit_trace(
                task_state,
                "tool_skipped",
                {
                    "name": name,
                    "args": args,
                    "reason": reason,
                    "call_index": index,
                    "call_count": len(payload),
                },
            )

    @staticmethod
    def was_executed(item):
        """这条 history 记录是否对应一次真实执行过的工具调用。

        老工件里没有 `executed` 字段，缺省视为执行过——只有显式写了 `False`
        的（`record_unexecuted_calls` 写的那些）才算没执行。
        """
        return item.get("executed", True) is not False

    def repeated_tool_call(self, name, args):
        # agent 很常见的一种坏循环，是在没有新信息的情况下反复发起同一调用。
        # 这里提前挡掉最简单的这种循环。
        #
        # 只看真实执行过的调用：没执行的那些（预算卡住的）本来就该被重发，
        # 把它们计进来会让重发第一次就撞上"重复调用"。
        tool_events = [
            item
            for item in self.session["history"]
            if item["role"] == "tool" and self.was_executed(item)
        ]
        if len(tool_events) < 2:
            return False
        recent = tool_events[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)

    @staticmethod
    def new_task_id():
        return "task_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    @staticmethod
    def new_run_id():
        return "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def build_report(self, task_state):
        # report 是一次运行的最终摘要；
        # 和 trace 的区别在于，trace 关注过程，report 关注结果与关键指标。
        return {
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
            "session_id": getattr(task_state, "session_id", ""),
            "run_seq": getattr(task_state, "run_seq", 0),
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": task_state.final_answer,
            "tool_steps": task_state.tool_steps,
            # 这次运行**实际**生效的步数上限。评测里 harness 会按任务派生出不同的
            # max_steps，而判分器只拿得到 base harness——不把它落进工件，
            # 「步数没超上限」这条断言就会拿一个与本次运行无关的数字去判。
            "max_steps": int(self.max_steps),
            "attempts": task_state.attempts,
            "checkpoint_id": task_state.checkpoint_id,
            "resume_status": task_state.resume_status,
            "task_state": task_state.to_dict(),
            "prompt_metadata": self.last_prompt_metadata,
            "durable_promotions": list(self.last_durable_promotions),
            "durable_rejections": list(self.last_durable_rejections),
            "durable_superseded": list(self.last_durable_superseded),
            "redacted_env": self.detected_secret_env_summary(),
        }

    def tool_example(self, name):
        # 传注册表：`run_plan` 的示例要按当前能调到的工具现算，静态示例会教
        # 模型去调一个白名单裁掉的工具（见 tools.plan_example 的注释）。
        return toolkit.tool_example(name, self.tools)

    def validate_tool(self, name, args):
        """把通用工具校验和 runtime 级额外约束串起来。"""
        toolkit.validate_tool(self, name, args)
        if name == "delegate":
            if self.depth >= self.max_depth:
                raise ValueError("delegate depth exceeded")

    def tool_list_files(self, args):
        return toolkit.tool_list_files(self, args)

    def tool_read_file(self, args):
        return toolkit.tool_read_file(self, args)

    def tool_search(self, args):
        return toolkit.tool_search(self, args)

    def tool_run_shell(self, args):
        return toolkit.tool_run_shell(self, args)

    def tool_write_file(self, args):
        return toolkit.tool_write_file(self, args)

    def tool_patch_file(self, args):
        return toolkit.tool_patch_file(self, args)

    def tool_delegate(self, args):
        return toolkit.tool_delegate(self, args)

    def approve(self, name, args):
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse(raw):
        """把模型原始输出解析成 runtime 可执行的动作或最终答案。

        为什么存在：
        模型输出首先是自然语言文本，而 runtime 需要的是结构化决策：
        “这是工具调用”还是“这是最终答案”。如果没有这层解析，后面的工具校验、
        审批和执行链路就没法可靠工作。

        输入 / 输出：
        - 输入：`raw`，或者是模型返回的原始文本（旧协议 / `FakeModelClient`），
          或者是 `model_client.complete()` 的结构化结果
          `{"text": str, "tool_calls": [...] | None}`（原生 function-calling）。
        - 输出：`(kind, payload)`，其中 `kind` 可能是 `tool`、`final`、`retry`

        在 agent 链路里的位置：
        它位于 `model_client.complete()` 之后、`run_tool()` 之前，是模型输出
        进入平台控制流的第一道结构化关口。

        **`tool` 的 payload 恒为列表**，哪怕只有一个调用。一轮里发几个由模型决定：
        这个后端实测会在一轮里发 3~4 个（同一提示跑 5 次有 3 次如此），而
        `parallel_tool_calls: false` 它并不理会。以前这里把「多于一个」判成 `retry`，
        代价是整轮作废重来——白烧一个约 17 秒的固定往返，还把本可一轮做完的三件事
        拆成三轮。现在按模型发来的顺序全部执行，`ask()` 逐个过 `run_tool()` 闸口，
        每个调用各自计一步、各自审批、各自记 trace。归约成 `retry` 的只剩真正无法
        执行的形状（缺工具名、args 不是对象），而不是"数量不合我们的预期"。

        标准路径只有一条：`tool_calls`。下面的 `<tool>`/`<final>` 标签解析是
        **宽容读取**，不是第二套协议——prompt 里不再教它，工具报错信息里也不再
        举它的例子（见 `build_prefix()`）。留着它只为一件事：模型万一自作主张
        把调用写成文本（推理模型偶发、后端不吃 `tools=` 时的即兴发挥），别让那
        一轮白白作废。走到这条分支属于异常而非常态，`ask()` 会把它记进 trace 和
        report 的 `text_protocol_tool_calls`，这样"协议漂移"是一个能被观测到的
        数字，而不是只能靠真实后端跑基准才偶然发现的现象。
        """
        if isinstance(raw, dict):
            tool_calls = raw.get("tool_calls")
            text = raw.get("text", "")
        else:
            tool_calls = None
            text = raw

        if tool_calls:
            calls = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    return "retry", CodingForMe.retry_notice(
                        "a tool call was not a JSON object", RETRY_REASON_TOOL_CALL_NOT_OBJECT
                    )
                name = str(call.get("name", "")).strip()
                if not name:
                    return "retry", CodingForMe.retry_notice(
                        "tool call is missing a tool name", RETRY_REASON_MISSING_TOOL_NAME
                    )
                args = call.get("args", {})
                if args is None:
                    args = {}
                elif not isinstance(args, dict):
                    return "retry", CodingForMe.retry_notice(
                        "tool arguments must be a JSON object", RETRY_REASON_TOOL_ARGS_NOT_OBJECT
                    )
                # 归一放在这里而不是各个 runner 里：从这一点往后，history、trace、
                # 记忆、重复检测拿到的都是同一份归一后的参数。见 coerce_tool_args()。
                calls.append({"name": name, "args": toolkit.coerce_tool_args(name, args)})
            return "tool", calls

        raw = str(text)
        # 走到这里说明后端没有返回原生 tool_calls，回退到文本兜底协议。
        # 这里支持两种工具格式：
        # 1. <tool>...</tool> 里包 JSON，适合简短调用
        # 2. XML 风格属性/子标签，适合写文件这类多行内容
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = CodingForMe.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:
                return "retry", CodingForMe.retry_notice(
                    "model returned malformed tool JSON", RETRY_REASON_MALFORMED_TOOL_JSON
                )
            if not isinstance(payload, dict):
                return "retry", CodingForMe.retry_notice(
                    "tool payload must be a JSON object", RETRY_REASON_TOOL_PAYLOAD_NOT_OBJECT
                )
            if not str(payload.get("name", "")).strip():
                return "retry", CodingForMe.retry_notice(
                    "tool payload is missing a tool name", RETRY_REASON_MISSING_TOOL_NAME
                )
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", CodingForMe.retry_notice(
                    "tool arguments must be a JSON object", RETRY_REASON_TOOL_ARGS_NOT_OBJECT
                )
            # 宽容读取捞回来的调用也要归一：两条路径产出的 payload 形状必须一致，
            # 否则协议漂移时连带把参数类型也漂了。
            payload["args"] = toolkit.coerce_tool_args(
                str(payload.get("name", "")).strip(), payload.get("args", {})
            )
            # 宽容读取一次只捞得出一个调用，但仍然包成列表：payload 的形状不随
            # 走了哪条分支而变，`ask()` 因此只有一种处理路径。
            return "tool", [payload]
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = CodingForMe.parse_xml_tool(raw)
            if payload is not None:
                return "tool", [payload]
            return "retry", CodingForMe.retry_notice(
                "model emitted a tool tag that could not be parsed", RETRY_REASON_UNPARSABLE_TOOL_TAG
            )
        if "<final>" in raw:
            final = CodingForMe.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", CodingForMe.retry_notice(
                "model returned an empty answer", RETRY_REASON_EMPTY_FINAL
            )
        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", CodingForMe.retry_notice(
            "model returned an empty response", RETRY_REASON_EMPTY_RESPONSE
        )

    @staticmethod
    def retry_notice(problem=None, reason=RETRY_REASON_MALFORMED_TOOL_JSON):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        # 这条 notice 会被写回 history，模型下一轮就读到它。它教的必须是
        # prefix 里那唯一一套协议——曾经在这里教 <tool> 标签，等于在模型出错
        # 的那一刻把它推向一套我们既没发 schema、也不打算支持的协议。
        # 不提"一次发几个"：runtime 两种都接，这条 notice 只该指出形状不合法。
        return RetryNotice(
            f"{prefix}. Either emit one or more tool calls through the function-calling "
            "interface, or reply with a non-empty plain-text answer.",
            reason,
        )

    @staticmethod
    def parse_xml_tool(raw):
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.S)
        if not match:
            return None
        attrs = CodingForMe.parse_attrs(match.group("attrs"))
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None

        body = match.group("body")
        args = dict(attrs)
        for key in ("content", "old_text", "new_text", "command", "task", "pattern", "path"):
            if f"<{key}>" in body:
                args[key] = CodingForMe.extract_raw(body, key)

        body_text = body.strip("\n")
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        if name == "delegate" and "task" not in args and body_text:
            args["task"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text):
        attrs = {}
        for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self):
        self.session["history"] = []
        # 摘要的覆盖点是 history 的下标，history 清空了它必须一起清——否则下一轮
        # 会拿一个陈旧的覆盖点去切一段空历史。
        self.session["context_summary"] = {}
        self.session["memory"].clear()
        self.session["memory"].update(memorylib.default_memory_state())
        self.memory = memorylib.LayeredMemory(self.session["memory"], workspace_root=self.root)
        self.session_store.save(self.session)

    def path(self, raw_path):
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        # 所有文件类工具都被锚定在 workspace root 之下。
        # 这样既能防住 "../" 逃逸，也能防住符号链接解析后跳出仓库。
        if os.path.commonpath([str(self.root), str(resolved)]) != str(self.root):
            # 措辞里必须保留 "path escapes workspace" 这个前缀：run_tool() 靠它把
            # 这次拒绝标成 `security_event_type: "path_escape"`，安全事件统计和
            # L1 的 path_confined 断言都从那里读。
            #
            # 后半句是补上去的可执行修复动作。这一类占 k=3 跑批被拒调用的 36%
            # （`../` 26% + 绝对路径 10%），而模型此前拿到的是一句纯陈述——它
            # 既不知道正确写法长什么样，也不知道自己是踩了哪一条。
            raise ValueError(
                f"path escapes workspace: {raw_path}. Paths are relative to the repo root; "
                "write them like 'src/app.py', with no leading '/' or drive letter and no '..'."
            )
        return resolved

