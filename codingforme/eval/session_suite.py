"""L3 跨会话套件：在**同一条会话**里连续跑多次 ask()，然后判会话层断言。

为什么单独有一个执行器：
固定基准（`evaluator.py`）的形状是「一个任务 = 一次 ask()」，每个任务各自一份
仓库、各自一条会话。那个形状测不了跨会话——实测 `multi_run_sessions = 0`，
P0 花全部力气建的三级身份在上面只用到了 session→run 一层。

这里的形状是「一个任务 = 一条会话 = N 次 ask()」，并且支持在任意两轮之间
**重建 agent 实例**（`restart_before`），走一遍真实的 resume 路径——跨进程的
连续性只有这样才测得到，同一个对象连着调用 N 次是测不出来的。

判分交给 `scorers.py` 的 L3 断言，本模块只负责「按脚本把会话跑出来」。
"""

import json
import tempfile
from pathlib import Path

from ..models import FakeModelClient, final_answer, tool_call
from ..run_store import RunStore
from ..runtime import SessionStore
from .economy import summarize_economy
from .tool_usage import summarize_tool_usage
from .harness import DEFAULT_HARNESS
from .report import (
    EXECUTION_MODE_LIVE_MODEL,
    EXECUTION_MODE_ORACLE_REPLAY,
    aggregate_cases,
    build_eval_result,
    build_run_context,
    render_eval_result_markdown,
    write_eval_result,
)
from .scorers import score_session, session_cases, summarize_assertions
from .trace import TraceIndex

DEFAULT_SESSION_BENCHMARK_PATH = Path("benchmarks") / "session_tasks.json"

SESSION_SUITE_NOTES = (
    "L3 断言的是「harness 有没有把先前建立的事实重新放回 prompt」，不是「模型有没有答对」——"
    "脚本化模型不真的读 prompt，而把对的东西放进上下文本来就是 harness 的职责。",
    "标了 restart_before 的轮次会重建 agent 实例走真实 resume 路径，跨进程连续性才测得到。",
)


def load_session_benchmark(path=DEFAULT_SESSION_BENCHMARK_PATH):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported session benchmark schema_version: {payload.get('schema_version')!r}")
    return payload


def scripted_outputs(turn):
    """把数据集里的 JSON 描述翻成 `complete()` 的真实返回形状。

    只认 {"final": ...} 和 {"tool": ..., "args": {...}} 两种，走的都是
    models.py 的构造函数——评测脚本不该自己拼那个 dict 的形状。
    """
    outputs = []
    for item in turn.get("outputs", []):
        if "final" in item:
            outputs.append(final_answer(str(item["final"])))
        elif "tool" in item:
            outputs.append(tool_call(str(item["tool"]), **dict(item.get("args", {}))))
        else:
            raise ValueError(f"unrecognised scripted output: {item!r}")
    if not outputs:
        raise ValueError("a session turn must script at least one model output")
    return outputs


def expectations_for(task):
    """`{run_seq: expect}`。run_seq 从 1 开始，与 ask() 的计数对齐。"""
    expectations = {}
    for index, turn in enumerate(task.get("turns", []), start=1):
        expect = dict(turn.get("expect", {}) or {})
        if turn.get("restart_before"):
            expect["restart_before"] = True
        if expect:
            expectations[index] = expect
    return expectations


def run_session_task(task, workspace_root, harness=None, model_client_factory=None):
    """按脚本跑完一条会话，返回 (session_id, TraceIndex)。"""
    harness = harness or DEFAULT_HARNESS
    workspace_root = Path(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    session_store = SessionStore(workspace_root / ".codingforme" / "sessions")
    run_store = RunStore(workspace_root / ".codingforme" / "runs")

    def build(session=None):
        # 刻意每次都重新构建 workspace 快照：restart 要模拟的是新进程，
        # 复用同一个快照对象就把「重启后重新感知仓库」这一步跳过了。
        outputs_client = model_client_factory() if model_client_factory else FakeModelClient([])
        return harness.build(
            outputs_client,
            workspace_root,
            session=session,
            session_store=session_store,
            run_store=run_store,
        )

    agent = build()
    session_id = agent.session["id"]

    for turn in task.get("turns", []):
        if turn.get("restart_before"):
            # 真实 resume：换一个 agent 实例接着同一条 session 跑。
            agent = build(session=session_store.load(session_id))
        if model_client_factory is None:
            agent.model_client = FakeModelClient(scripted_outputs(turn))
        agent.ask(str(turn["request"]))

    index = TraceIndex.load(
        workspace_root / ".codingforme" / "runs",
        workspace_root / ".codingforme" / "sessions",
    )
    return session_id, index


def run_session_suite(
    harness=None,
    benchmark_path=DEFAULT_SESSION_BENCHMARK_PATH,
    workspace_root=None,
    result_path=None,
    markdown_path=None,
    model_client_factory=None,
    repo_root=None,
):
    harness = harness or DEFAULT_HARNESS
    benchmark = load_session_benchmark(benchmark_path)
    # 默认落在临时目录,不是仓库里的 `.codingforme/eval-sessions`。旧默认有两个问题,
    # 都是踩过的:一、它嵌在本仓库内部,`workspace.py` 的快照会被悄悄放大成整个项目
    # (`run_context.workspace_git_root` 因此恒为非空,这批数字和别的跑批不可比);
    # 二、它跨次调用复用同一个根,`TraceIndex` 把上一次的 run 一起索引进来——连跑三个
    # 变体会看到会话数 17 → 22 → 27 这种累加,每个变体的数字里都混着前一个变体的。
    if workspace_root:
        workspace_root = Path(workspace_root)
    else:
        workspace_root = Path(tempfile.mkdtemp(prefix="codingforme-sessions-"))
    workspace_root.mkdir(parents=True, exist_ok=True)

    cases = []
    assertions = []
    indexes = []
    for task in benchmark["tasks"]:
        task_root = workspace_root / str(task["id"])
        session_id, index = run_session_task(
            task,
            task_root,
            harness=harness,
            model_client_factory=model_client_factory,
        )
        indexes.append(index)
        session = index.session(session_id)
        expectations = expectations_for(task)
        cases.extend(session_cases(session, expectations, case_prefix=str(task["id"])))
        assertions.extend(score_session(session, expectations))

    merged = TraceIndex.merge(indexes)
    aggregates = aggregate_cases(cases)
    aggregates["trajectory"] = summarize_assertions(assertions)
    # 跨会话套件同样要报成本：多轮会话的 input token 只会比单轮更大，
    # 不报的话「记忆机制到底省不省上下文」这个问题就没有观测量。
    aggregates["economy"] = summarize_economy(merged)
    aggregates["tool_usage"] = summarize_tool_usage(merged)

    result = build_eval_result(
        suite="cross-session",
        harness=harness,
        dataset={
            "source": benchmark["source"],
            "task_count": len(benchmark["tasks"]),
            "question_types": sorted({str(task["question_type"]) for task in benchmark["tasks"]}),
        },
        cases=cases,
        run_context=build_run_context(
            mode="scripted" if model_client_factory is None else "custom",
            # 不传 execution_mode 会默认成 oracle-replay，于是一次真实模型跑批
            # 会被报告标成参考解回放，并渲染出一段「capability 轴评的是参考解」的
            # 假声明。口径字段必须跟着实际执行方式走。
            execution_mode=(
                EXECUTION_MODE_ORACLE_REPLAY if model_client_factory is None else EXECUTION_MODE_LIVE_MODEL
            ),
            # 同 suite.py：模型名要跟着实际执行方式走，否则 live 跑批的工件
            # 说不清是哪个模型产生的（这里此前是空字符串）。
            model=str(getattr(model_client_factory, "model_name", "") or ""),
            repo_root=repo_root,
            workspace_root=workspace_root,
        ),
        trace_summary=merged.to_dict(),
        aggregates=aggregates,
        notes=SESSION_SUITE_NOTES,
    )

    if result_path:
        write_eval_result(result_path, result)
    if markdown_path:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_eval_result_markdown(result), encoding="utf-8")
    return result
