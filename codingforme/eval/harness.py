"""HarnessSpec：把「被测的那套控制循环配置」变成一等对象。

为什么存在：
这个项目的卖点是包在模型外面的控制循环，那评测对象就该是控制循环本身，
而不只是模型。此前每个实验各自手搓一份 agent 装配代码（metrics.py 里的
`_build_real_agent` / `_security_agent`、evaluator.py 里的内联装配、测试里
各文件重复定义的 `build_agent`），配置差异散在参数里，无法命名、无法落盘、
无法互相对比。

HarnessSpec 把这些配置固化成一个可命名、可指纹、可序列化的变体，于是
「消融实验」和「评测」合并成同一件事：同一份数据集 × 多个 HarnessSpec。

它刻意不包含模型客户端：模型是另一条正交的轴，由调用方注入，这样
「固定模型、只变 harness」和「固定 harness、只变模型」两种对比都能表达。
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..run_store import RunStore
from .code_signature import model_facing_code_signature, module_signatures
from .. import tools as toolkit
from ..runtime import DEFAULT_FEATURE_FLAGS, CodingForMe, SessionStore, prompt_template_signature
from ..tools import base_tool_schema_signature
from ..workspace import WorkspaceContext

HARNESS_SPEC_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HarnessSpec:
    """一个被测的 harness 变体。

    字段刻意只覆盖「会改变 agent 行为」的配置。跑在哪个目录、用哪个模型、
    落盘到哪里这些属于执行环境，由 build() 的参数传入，不进指纹。
    """

    name: str
    description: str = ""
    approval_policy: str = "auto"
    max_steps: int = 6
    # 和运行时保持同一个数（runtime.py 的 max_new_tokens）。512 太紧：实测
    # 真实模型有整轮输出正好顶到上限被截断、整轮作废的情况。注意这个字段进
    # fingerprint()，改它会让所有变体的指纹变化——这是预期的，指纹本来就该
    # 反映「这次评的不是同一个配置」。
    max_new_tokens: int = 1024
    read_only: bool = False
    max_depth: int = 1
    feature_flags: dict = field(default_factory=dict)
    # 只保留这些工具（None = 全量）。裁剪工具注册表是逼出特定行为的手段，
    # 也是「工具集是 harness 的一部分」这个观点的落点。
    tools_allowlist: tuple = None
    # context_manager 的预算覆盖（None = 用默认值）
    total_budget: int = None
    section_budgets: dict = None

    def resolved_feature_flags(self):
        flags = dict(DEFAULT_FEATURE_FLAGS)
        flags.update({str(key): bool(value) for key, value in (self.feature_flags or {}).items()})
        return flags

    def derive(self, name=None, **overrides):
        """派生一个新变体。用于按任务覆盖 max_steps 这类逐条不同的参数。"""
        if name is not None:
            overrides["name"] = name
        return replace(self, **overrides)

    def to_dict(self):
        payload = dataclasses.asdict(self)
        payload["feature_flags"] = self.resolved_feature_flags()
        payload["tools_allowlist"] = list(self.tools_allowlist) if self.tools_allowlist else None
        payload["section_budgets"] = dict(self.section_budgets) if self.section_budgets else None
        payload["schema_version"] = HARNESS_SPEC_SCHEMA_VERSION
        return payload

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        payload.pop("schema_version", None)
        payload.pop("fingerprint", None)
        allowlist = payload.get("tools_allowlist")
        payload["tools_allowlist"] = tuple(allowlist) if allowlist else None
        known = {item.name for item in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})

    def code_signature(self):
        """配置之外、但会改变模型行为的**代码**的签名。

        为什么需要它：`HarnessSpec` 的字段只覆盖「配置」（审批策略、步数上限、
        只读、feature flags、工具白名单、context 预算）。但两次跑批不可比的原因
        不止配置，还有三类都不是配置字段的东西：**提示词文本**、**工具 schema**、
        以及**运行时逻辑本身**（上下文怎么组装、模型输出怎么解析、记忆怎么召回）。

        两次实测后果，都是这个字段给出错误答案：
        - 截至 2026-08-17，本仓库所有跑批的 harness 指纹恒为 `sha256:1e0dcb0f0a19`。
          期间提示词改过四轮、模型输出协议改过一轮，指纹一次没变。补上前两项之后
          这个洞堵上了。
        - 2026-08-18 复盘发现第三个洞：T2-1 把上下文从「压平成一段文本」换成标准
          messages 数组，改动前后两次 12 任务 × 3 轮的正式跑批，`code_signature`
          仍然完全相同（`sha256:db7c156bcbee7`）——一个让走到终点率 76% → 97%
          的改动，在「可不可比」这个字段上完全不可见。`model_facing_code_signature()`
          就是为这一条补的，细节见 eval/code_signature.py。

        刻意**不**把 workspace 快照算进来：那是每个任务各自的执行环境，不是
        被测对象的属性，算进去会让同一个 harness 在每个任务上指纹都不同。
        """
        return "sha256:" + hashlib.sha256(
            json.dumps(
                {
                    "prompt_template": prompt_template_signature(),
                    "tool_schema": base_tool_schema_signature(),
                    "model_facing_code": model_facing_code_signature(),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def code_signature_parts():
        """签名的分项，用来回答「这次是哪一部分变了」。

        合并成一个哈希之后，"变了"和"哪里变了"就分开了；没有这个入口，事后
        只能靠 git 逐个模块比对。不进指纹，只进报告和排查。
        """
        parts = {
            "prompt_template": prompt_template_signature(),
            "tool_schema": base_tool_schema_signature(),
        }
        parts.update(module_signatures())
        return parts

    def fingerprint(self):
        """配置 + 相关代码的 sha256。进评测结果 schema，用来判定两次结果是否可比。"""
        payload = self.to_dict()
        payload.pop("description", None)
        # 提示词与工具 schema 一起进指纹，理由见 code_signature()。
        payload["code_signature"] = self.code_signature()
        return "sha256:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()

    def build(
        self,
        model_client,
        workspace_root,
        *,
        session=None,
        session_store=None,
        run_store=None,
        workspace=None,
        repo_root_override=None,
        on_token=None,
        depth=0,
        **agent_kwargs,
    ):
        """按这份配置装配一个 agent。

        这是唯一的装配入口——任何评测都不应再手写 CodingForMe(...)，
        否则「变体」这个概念就又漏了。

        workspace 可以由调用方预先构建后传进来（比如需要先拿它去造模型
        客户端的场景），不传则按 workspace_root 现场构建。
        """
        workspace_root = Path(workspace_root)
        if workspace is None:
            workspace = WorkspaceContext.build(
                workspace_root,
                repo_root_override=repo_root_override if repo_root_override is not None else workspace_root,
            )
        session_store = session_store or SessionStore(workspace_root / ".codingforme" / "sessions")
        run_store = run_store or RunStore(workspace_root / ".codingforme" / "runs")

        agent = CodingForMe(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session,
            run_store=run_store,
            approval_policy=self.approval_policy,
            max_steps=int(self.max_steps),
            max_new_tokens=int(self.max_new_tokens),
            read_only=bool(self.read_only),
            depth=depth,
            max_depth=int(self.max_depth),
            feature_flags=self.resolved_feature_flags(),
            on_token=on_token,
            **agent_kwargs,
        )

        if self.tools_allowlist:
            allowed = {str(name) for name in self.tools_allowlist}
            missing = sorted(allowed - set(agent.tools))
            if missing:
                raise ValueError(f"harness {self.name!r} allowlists unknown tools: {', '.join(missing)}")
            # 元工具穿过白名单（见 tools.META_TOOLS）：它们不提供新能力，只改变
            # 调用形式，而它们能调到的工具**仍然**只有裁剪之后剩下的这些。
            keep = allowed | (toolkit.META_TOOLS & set(agent.tools))
            agent.tools = {name: spec for name, spec in agent.tools.items() if name in keep}
            agent.declared_tools_allowlist = tuple(sorted(allowed))
            # 工具集变了，prefix 和 tool_signature 都得跟着变，
            # 否则发给模型的工具列表和实际注册表会对不上。
            agent.refresh_prefix(force=True)

        if self.total_budget is not None:
            agent.context_manager.total_budget = int(self.total_budget)
        if self.section_budgets:
            agent.context_manager.section_budgets.update(
                {str(key): int(value) for key, value in self.section_budgets.items()}
            )

        return agent


def intersect_tools_allowlist(variant_allowlist, task_allowlist):
    """变体的工具白名单 ∩ 任务声明的白名单，结果可直接喂给 `derive()`。

    为什么是交集而不是覆盖：两句话都该成立——变体说「这个消融只给只读工具」，
    任务说「这道题只该用 read_file」。谁覆盖谁都会让另一句话失效。`None` 的
    含义是「不限制」，所以 `None ∩ X = X`、`X ∩ None = X`。

    踩过的坑（N-5）：`benchmarks/coding_tasks.json` 的 `allowed_tools` 从加载
    起就被校验、被原样抄进结果行，**但从来没有参与过 agent 的装配**——一个
    声明 `["read_file"]` 的任务，模型照样调到了 `write_file` 并在仓库根建了
    `MEMORY.md`。字段是纯装饰的，而报告里看不出这一点。

    交集为空时直接报错，不装配一个没有工具的 agent：后者会让模型第一轮就
    无事可做，而失败在工件上看起来像「模型不会做这道题」。
    """
    variant = tuple(str(name) for name in (variant_allowlist or ()))
    task = tuple(str(name) for name in (task_allowlist or ()))
    if not variant:
        return tuple(sorted(set(task))) or None
    if not task:
        return tuple(sorted(set(variant)))
    merged = set(variant) & set(task)
    if not merged:
        raise ValueError(
            "tools allowlist intersection is empty: "
            f"variant={sorted(set(variant))} task={sorted(set(task))}"
        )
    return tuple(sorted(merged))


# 内建变体。这批就是现有消融实验里那些散落的配置，收敛成有名字的东西。
DEFAULT_HARNESS = HarnessSpec(
    name="full",
    description="全部机制开启的基线配置",
)

BUILTIN_HARNESS_SPECS = {
    spec.name: spec
    for spec in (
        DEFAULT_HARNESS,
        HarnessSpec(
            name="no_memory",
            description="关闭记忆注入与相关记忆召回",
            feature_flags={"memory": False, "relevant_memory": False},
        ),
        HarnessSpec(
            name="no_context_reduction",
            description="关闭预算裁剪，prompt 不做压缩",
            feature_flags={"context_reduction": False},
        ),
        HarnessSpec(
            name="no_prompt_cache",
            description="关闭 prompt 缓存相关行为",
            feature_flags={"prompt_cache": False},
        ),
        HarnessSpec(
            name="read_only",
            description="只读：写类工具全部被闸口挡住",
            read_only=True,
        ),
        HarnessSpec(
            name="strict_approval",
            description="高风险工具一律拒绝",
            approval_policy="never",
        ),
        # 受限编排是**加**了一个能力，不是关掉一个——所以它和上面几个消融变体
        # 方向相反，但仍然是同一个东西：一个有名字、有指纹、可以单独跑一批
        # 数据来对比的变体。默认变体不带它，两边的数字因此可比。
        HarnessSpec(
            name="plan_tool",
            description="开启受限编排 run_plan：模型可以用一小段程序把有依赖的多步调用串起来",
            feature_flags={"plan_tool": True},
        ),
    )
}


def get_harness(name):
    spec = BUILTIN_HARNESS_SPECS.get(str(name))
    if spec is None:
        raise KeyError(f"unknown harness variant: {name!r} (known: {', '.join(sorted(BUILTIN_HARNESS_SPECS))})")
    return spec
