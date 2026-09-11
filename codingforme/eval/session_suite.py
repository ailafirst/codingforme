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
import shutil
import tempfile
from pathlib import Path

from ..models import FakeModelClient, final_answer, tool_call
from ..run_store import RunStore
from ..runtime import SessionStore
from .economy import summarize_economy
from .tool_usage import summarize_tool_usage
from .harness import DEFAULT_HARNESS
from .report import (
    AXIS_RELIABILITY,
    EXECUTION_MODE_LIVE_MODEL,
    EXECUTION_MODE_ORACLE_REPLAY,
    LEVEL_SESSION,
    aggregate_cases,
    build_case_result,
    build_eval_result,
    build_run_context,
    render_eval_result_markdown,
    write_eval_result,
)
from .scorers import (
    ASSERTION_SUBJECTS,
    score_index,
    score_session,
    session_cases,
    summarize_assertions,
    trajectory_cases,
)
from .trace import TraceIndex
from ..workspace import IGNORED_PATH_NAMES

DEFAULT_SESSION_BENCHMARK_PATH = Path("benchmarks") / "session_tasks.json"

# 会话分两档，理由是成本而不是重要性。`core` 是每次跑批、每次 pytest 都要跑的
# 回归网；`extended` 是 200/300 轮那种「问的是极限在哪里」的阶梯——它的答案只在
# 记忆层改动时才会变，而一条 300 轮会话就是 300 次 ask()，把它塞进默认集合等于
# 让每个人每次跑测试都替这个问题付一遍钱。
#
# **分档不是把它们降级成一次性脚本**：两档走同一个执行器、同一批断言、同一份
# 结果 schema，`--session-tier extended` 就能单独跑一批出来。数据集里没写 tier
# 的任务一律算 core（新增字段不能让既有任务悄悄从默认集合里消失）。
SESSION_TIER_CORE = "core"
SESSION_TIER_EXTENDED = "extended"
SESSION_TIERS = (SESSION_TIER_CORE, SESSION_TIER_EXTENDED)
DEFAULT_SESSION_TIERS = (SESSION_TIER_CORE,)

SESSION_SUITE_NOTES = (
    "L3 断言的是「harness 有没有把先前建立的事实重新放回 prompt」，不是「模型有没有答对」——"
    "脚本化模型不真的读 prompt，而把对的东西放进上下文本来就是 harness 的职责。",
    "标了 restart_before 的轮次会重建 agent 实例走真实 resume 路径，跨进程连续性才测得到。",
)


def load_session_benchmark(path=DEFAULT_SESSION_BENCHMARK_PATH, repo_root=None, tiers=DEFAULT_SESSION_TIERS):
    """读数据集并按档位过滤。

    `tiers=None` 表示不过滤（全部档位都要）。**校验永远跑在过滤之前**：一个
    写错了 fixture 路径的 extended 任务，不该因为这次没跑它就躲过校验——那正是
    「声明退化成永不生效的注释」那个坑的又一种形态。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported session benchmark schema_version: {payload.get('schema_version')!r}")
    _validate_session_benchmark(payload, repo_root=repo_root)
    payload = dict(payload)
    selected = None if tiers is None else tuple(str(item) for item in tiers)
    if selected is not None:
        unknown = [name for name in selected if name not in SESSION_TIERS]
        if unknown:
            raise ValueError(f"unknown session tier(s): {unknown}")
        payload["tasks"] = [task for task in payload["tasks"] if task_tier(task) in selected]
    payload["tiers"] = list(selected) if selected is not None else list(SESSION_TIERS)
    return payload


def task_tier(task):
    """没写 `tier` 的任务算 core——新增字段不能让既有任务悄悄退出默认集合。"""
    return str(task.get("tier") or SESSION_TIER_CORE)


def _validate_session_benchmark(payload, repo_root=None):
    """加载时就把两个可选字段查一遍，别让它们退化成永不生效的注释。

    踩过的同类坑在固定基准那边：字段被校验、被抄进结果行、却从没传进装配，
    于是整条断言在整份数据集上一次都没触发过，而报告里长得和「没问题」一样。
    """
    root = Path(repo_root) if repo_root else Path.cwd()
    for task in payload.get("tasks", []):
        task_id = task.get("id", "<unnamed>")
        fixture = task.get("fixture_repo")
        if fixture is not None:
            if not str(fixture).strip():
                raise ValueError(f"session task {task_id} has an empty fixture_repo")
            if not (root / str(fixture)).is_dir():
                raise ValueError(f"session task {task_id} fixture repo does not exist: {fixture}")
        steps = task.get("max_steps")
        if steps is not None and (not isinstance(steps, int) or steps < 1):
            raise ValueError(f"session task {task_id} max_steps must be a positive integer")
        tier = task.get("tier")
        if tier is not None and str(tier) not in SESSION_TIERS:
            raise ValueError(f"session task {task_id} has an unknown tier: {tier!r}")
        # `asserts` 是任务对「我存在是为了让哪几条断言真的判一次」的声明。有它的任务
        # 可以不写 `expect`——重复读那两条的判据是 L1 形状，会话层没有对应的期望字段，
        # 而没有任何声明的任务等于什么都没测。名字对着注册表校验：写错一个字母会变成
        # 一条永不生效的注释，而那正是这个字段要防的东西。
        declared = task.get("asserts")
        if declared is not None:
            if not isinstance(declared, list) or not declared:
                raise ValueError(f"session task {task_id} asserts must be a non-empty list")
            unknown = [name for name in declared if str(name) not in ASSERTION_SUBJECTS]
            if unknown:
                raise ValueError(f"session task {task_id} declares unknown assertions: {unknown}")
        if not declared and not any(turn.get("expect") for turn in task.get("turns", [])):
            raise ValueError(f"session task {task_id} asserts nothing: give it an expect or an asserts list")
        for index, turn in enumerate(task.get("turns", []), start=1):
            evidence = turn.get("context_evidence")
            if evidence is None:
                continue
            if not isinstance(evidence, list) or not evidence:
                raise ValueError(f"session task {task_id} turn {index} context_evidence must be a non-empty list")
            if any(not str(item).strip() for item in evidence):
                raise ValueError(f"session task {task_id} turn {index} has an empty context_evidence entry")


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


def prepare_session_workspace(task, workspace_root, repo_root=None):
    """把任务声明的样板仓库复制进这条会话的工作区。

    没有 `fixture_repo` 的任务照旧跑在空目录里（纯对话的召回类会话不需要文件）。
    有的话必须复制而不是直接指向 `tests/fixtures/` 下那份——会话会往里写文件、
    写 `.codingforme/`，跑一次就把样板仓库改脏了，下一次跑批的基线就不是基线。

    `IGNORED_PATH_NAMES` 取自 workspace 那份同一个名单：样板仓库里可能残留上一次
    live 跑批写下的 `.codingforme/`，复制过去会让这条会话的工作区里凭空多出一份
    别人的 session，而这个套件恰恰在测 resume。
    """
    workspace_root = Path(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    fixture = task.get("fixture_repo")
    if not fixture:
        return workspace_root
    source = (Path(repo_root) if repo_root else Path.cwd()) / str(fixture)
    if not source.is_dir():
        raise ValueError(f"session task {task.get('id')} fixture repo does not exist: {fixture}")
    shutil.copytree(
        source,
        workspace_root,
        ignore=shutil.ignore_patterns(*IGNORED_PATH_NAMES),
        dirs_exist_ok=True,
    )
    return workspace_root


def run_session_task(task, workspace_root, harness=None, model_client_factory=None, repo_root=None):
    """按脚本跑完一条会话，返回 (session_id, TraceIndex)。"""
    harness = harness or DEFAULT_HARNESS
    # 步数上限跟着任务走：默认 6 步是给「读一个文件、答一句」那种会话定的，而
    # 长期任务会话里一轮就可能十来个调用（读完一批模块再改一处）。不给数据集这个
    # 旋钮的话，这类会话会在中途被 `step_limit_reached` 截停，而截停之后写出来的
    # 工件长得像「模型自己停了」——那正是这批 case 要测的东西之一，会被自己污染。
    if task.get("max_steps"):
        harness = harness.derive(
            name="%s+steps%d" % (harness.name, int(task["max_steps"])),
            max_steps=int(task["max_steps"]),
        )
    workspace_root = prepare_session_workspace(task, workspace_root, repo_root=repo_root)

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
        # 数据集声明的「这一轮的回答依赖哪几个串」。必须逐轮盖上去（不是整条会话
        # 一次），因为 `_context_evidence_report()` 在组 prompt 的那一刻现算，而
        # 「第 1 轮立的规格，到第 100 轮还在不在上下文里」这个问题只有在第 100 轮
        # 那一刻查得到——事后从 trace 复算不出来，trace 里只有分段计数，没有 prompt 原文。
        # 一轮都没声明时显式清成 None，否则上一轮的声明会悄悄漏到这一轮。
        agent.declared_context_evidence = [str(item) for item in (turn.get("context_evidence") or [])] or None
        agent.ask(str(turn["request"]))

    index = TraceIndex.load(
        workspace_root / ".codingforme" / "runs",
        workspace_root / ".codingforme" / "sessions",
    )
    return session_id, index


def declared_assertion_cases(task, assertions):
    """任务声明的 `asserts` 必须真的被产出过，否则记一条失败的 case。

    这是「探针失去作用对象」那个坑的守卫，做成 case 而不是只做成测试：数据集或
    fixture 哪天改得让断言不再适用，跑批当场就红，而不是等到有人想起来去读单测。
    """
    declared = [str(name) for name in (task.get("asserts") or [])]
    if not declared:
        return []
    observed = {assertion.assertion_id for assertion in assertions}
    missing = [name for name in declared if name not in observed]
    return [
        build_case_result(
            "%s:asserts_declared" % task["id"],
            LEVEL_SESSION,
            not missing,
            axis_values={AXIS_RELIABILITY: 0.0 if missing else 1.0},
            detail={"declared": declared, "missing": missing, "observed": sorted(observed)},
            trace_ref={"task_id": task["id"]},
        )
    ]


def run_session_suite(
    harness=None,
    benchmark_path=DEFAULT_SESSION_BENCHMARK_PATH,
    workspace_root=None,
    result_path=None,
    markdown_path=None,
    model_client_factory=None,
    repo_root=None,
    tiers=DEFAULT_SESSION_TIERS,
):
    harness = harness or DEFAULT_HARNESS
    benchmark = load_session_benchmark(benchmark_path, repo_root=repo_root, tiers=tiers)
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
            repo_root=repo_root,
        )
        indexes.append(index)
        session = index.session(session_id)
        expectations = expectations_for(task)
        cases.extend(session_cases(session, expectations, case_prefix=str(task["id"])))
        assertions.extend(score_session(session, expectations))
        # L1 也在这里跑一遍。加它是因为「有没有重复读没变过的文件」「patch 前读没读」
        # 这类判据是逐次运行的轨迹形状，写成会话断言反而别扭；而在这之前跨会话套件
        # 只跑 SESSION_CHECKS 那几条，任何 L1 形状的断言写了也不会被执行。
        run_assertions = score_index(index, harness)
        assertions.extend(run_assertions)
        cases.extend(trajectory_cases(index, harness, assertions=run_assertions))
        # 数据集声明「我存在是为了让这几条断言真的判一次」的，就地验一次。
        # 断言返回 None（不适用）时不会产出任何 case，于是「查过了、没问题」和
        # 「压根没查」在报告上长得一模一样——这是这个套件反复踩过的那个坑。
        cases.extend(
            declared_assertion_cases(task, run_assertions + score_session(session, expectations))
        )

    merged = TraceIndex.merge(indexes)
    aggregates = aggregate_cases(cases)
    # 一张表同时列 L1 和 L3：`by_assertion` 天然按断言名分开，谁是谁一眼看得出，
    # 而「L3 到底几条通过」由 `by_level` 回答（L1-trajectory / L3-session 两个桶）。
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
            # 档位要落进工件：两批数字的任务数不同时，「跑的是哪一档」是唯一能
            # 解释差异的字段，缺了它 core 与 extended 的报告看起来像同一件事跑歪了。
            "tiers": list(benchmark.get("tiers") or DEFAULT_SESSION_TIERS),
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
