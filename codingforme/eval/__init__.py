"""CFM-Eval：评测体系的公共原语。

P0 只落三件事——被测对象（HarnessSpec）、数据底座（TraceIndex）、
结果形状（统一 schema）。数据集、判分器、指标注册表、门禁是后续阶段。

刻意不在这里 import suite/compat：它们会拉起 evaluator/metrics，
而 evaluator 反过来要 import 本包的 harness，直接 import 会成环。
需要时按模块路径显式导入即可。
"""

from .harness import (
    BUILTIN_HARNESS_SPECS,
    DEFAULT_HARNESS,
    HARNESS_SPEC_SCHEMA_VERSION,
    HarnessSpec,
    get_harness,
)
from .report import (
    AXES,
    EVAL_RESULT_SCHEMA_VERSION,
    LEVELS,
    aggregate_cases,
    build_case_result,
    build_eval_result,
    build_run_context,
    render_eval_result_markdown,
    write_eval_result,
)
from .scorers import (
    CHECK_CLASSES,
    SESSION_CHECKS,
    TRAJECTORY_CHECKS,
    TRAJECTORY_SCORER_SCHEMA_VERSION,
    Assertion,
    score_index,
    score_run,
    score_session,
    session_cases,
    summarize_assertions,
    trajectory_cases,
)
from .trace import (
    TRACE_INDEX_SCHEMA_VERSION,
    RunRecord,
    SessionRecord,
    TraceIndex,
    TurnRecord,
)

__all__ = [
    "AXES",
    "CHECK_CLASSES",
    "SESSION_CHECKS",
    "TRAJECTORY_CHECKS",
    "TRAJECTORY_SCORER_SCHEMA_VERSION",
    "Assertion",
    "BUILTIN_HARNESS_SPECS",
    "DEFAULT_HARNESS",
    "EVAL_RESULT_SCHEMA_VERSION",
    "HARNESS_SPEC_SCHEMA_VERSION",
    "HarnessSpec",
    "LEVELS",
    "RunRecord",
    "SessionRecord",
    "TRACE_INDEX_SCHEMA_VERSION",
    "TraceIndex",
    "TurnRecord",
    "aggregate_cases",
    "build_case_result",
    "build_eval_result",
    "build_run_context",
    "get_harness",
    "render_eval_result_markdown",
    "score_index",
    "score_run",
    "score_session",
    "session_cases",
    "summarize_assertions",
    "trajectory_cases",
    "write_eval_result",
]
