"""长期任务 case 的验收与证伪。

这批 case 补的是三个洞：超长会话里第一轮的信息还在不在、会不会重复读一个没变过的
文件、会不会照着旧信息动手。**每一类至少两条 case**，理由是单条挂掉分不清是机制坏了
还是那一个 fixture 恰好触发不到——而「探针失去作用对象」在报告里长得和「机制没问题」
一模一样（`stale_read` 探针第一版就栽在这里，两臂跑出来完全相同）。

用例分两类，缺一不可：

- **契约与适用性**：数据集的形状、断言在正确轨迹上确实**适用**（不是恒返回「不适用」
  混过去）。
- **证伪**：手搓一条违规轨迹，断言它真的会被判挂。回放模式下模型输出是我们自己写的，
  所以数据集里那八条 case 只能证明「判据接通了」，证明不了「模型会不会犯这个错」——
  会犯的那一半只有 `--live-model` 才测得到，而判据本身会不会漏检，只能靠这里的用例。
"""

import json
from pathlib import Path

import pytest

from codingforme.eval.harness import DEFAULT_HARNESS, get_harness
from codingforme.eval.scorers import (
    ASSERTION_SUBJECTS,
    SESSION_CHECKS,
    TRAJECTORY_CHECKS,
    check_forbidden_paths_untouched,
    check_no_redundant_reread,
    check_no_redundant_reread_across_runs,
    check_stale_reads_flagged,
    score_session,
)
from codingforme.eval.session_suite import (
    DEFAULT_SESSION_BENCHMARK_PATH,
    load_session_benchmark,
    prepare_session_workspace,
    run_session_task,
)
from codingforme.eval.trace import RunRecord, SessionRecord, TurnRecord

LONG_HORIZON_IDS = (
    "long_horizon_spec_recall_20",
    "long_horizon_spec_recall_100",
    "long_horizon_spec_recall_200",
    "long_horizon_spec_recall_300",
)
REREAD_IDS = (
    "reread_after_unrelated_calls",
    "reread_across_restart",
    "long_horizon_reread_across_runs_24",
    "long_horizon_reread_in_run_26",
)
STALE_IDS = (
    "stale_patch_anchor",
    "stale_read_across_runs",
    "stale_spill_pointer",
    "long_horizon_stale_anchor_24",
    "long_horizon_stale_pointer_26",
)
# 200/300 轮那两条在 extended 档，默认跑批不加载它们。数据集契约这一类测试查的是
# 「数据集里写对了没有」，必须看全部档位——只看 core 的话，extended 里写错的 fixture
# 路径、写错的断言名会一直躲着，而那正是加载期校验存在的理由。
ALL_TIERS = None


def tasks_by_id():
    benchmark = load_session_benchmark(DEFAULT_SESSION_BENCHMARK_PATH, tiers=ALL_TIERS)
    return {task["id"]: task for task in benchmark["tasks"]}


def read_call(path, start=1, end=40, spilled=False, status="ok"):
    return {
        "name": "read_file",
        "args": {"path": path, "start": start, "end": end},
        "tool_status": status,
        "tool_output_spilled": spilled,
        "affected_paths": [],
    }


def write_call(path):
    return {
        "name": "patch_file",
        "args": {"path": path, "old_text": "a", "new_text": "b"},
        "tool_status": "ok",
        "affected_paths": [path],
    }


def synthetic_run(call_groups, run_seq=1, run_id="r1", window=6, prior_runs=()):
    """一次运行：`call_groups` 的每个元素是一轮，元素本身是这一轮里的调用列表。"""
    turns = [
        TurnRecord(
            session_id="s1",
            run_id=run_id,
            run_seq=run_seq,
            turn=index,
            kind="tool",
            prompt_metadata={"history": {"recent_tool_window": window}},
            tools=[dict(call) for call in group],
        )
        for index, group in enumerate(call_groups, start=1)
    ]
    return RunRecord(
        run_id=run_id,
        session_id="s1",
        run_seq=run_seq,
        run_dir=Path("."),
        turns=turns,
        task_state={"stop_reason": "final_answer_returned", "status": "completed"},
        prior_runs=list(prior_runs),
    )


# --- 数据集契约 --------------------------------------------------------------


def test_every_long_horizon_kind_has_at_least_two_cases():
    """每类至少两条，且形状必须不同。

    只有一条 case 时，它挂了分不清是机制坏了还是这个 fixture 恰好触发不到；
    两条形状不同的同时挂，才是机制问题。这条按类计数，不按总数——总数够而某一类
    只剩一条，正是这条要拦的情况。
    """
    tasks = tasks_by_id()
    buckets = {}
    for task in tasks.values():
        buckets.setdefault(task["question_type"], []).append(task["id"])

    for kind in ("long_horizon_recall", "redundant_work", "stale_information"):
        assert len(buckets.get(kind, [])) >= 2, f"{kind} 只有一条 case，偶然性排除不掉"

    # 长程召回那几条是同一条事实的 20 / 100 / 200 / 300 轮阶梯——差的必须只有长度，
    # 否则「第几档开始挂」这个读数就不是长度造成的，而是问法或事实本身变了。
    ladder = [tasks[task_id] for task_id in LONG_HORIZON_IDS]
    assert [len(task["turns"]) for task in ladder] == [20, 100, 200, 300]
    first = ladder[0]
    for task in ladder[1:]:
        assert task["turns"][0]["request"] == first["turns"][0]["request"], task["id"]
        assert task["turns"][-1] == first["turns"][-1], task["id"]


def test_the_recall_ladder_splits_across_tiers_without_losing_the_control():
    """阶梯分两档是成本决定的，但**对照组必须留在默认档**。

    200/300 轮那两条的答案只有记忆层改动时才会变，一条 300 轮会话就是 300 次
    ask()，进默认集合等于每个人每次跑测试都替这个问题付一遍钱。但 20 轮那条
    对照组一旦跟着挪进 extended，默认跑批里 100 轮那条挂掉时就又分不清是
    「召回本身不成立」还是「长度造成的」——那正是对照组存在的全部理由。
    """
    tasks = tasks_by_id()
    tiers = {task_id: tasks[task_id].get("tier", "core") for task_id in LONG_HORIZON_IDS}
    assert tiers["long_horizon_spec_recall_20"] == "core"
    assert tiers["long_horizon_spec_recall_100"] == "core"
    assert tiers["long_horizon_spec_recall_200"] == "extended"
    assert tiers["long_horizon_spec_recall_300"] == "extended"
    # 默认档加载出来的任务里不能混进 extended，否则分档等于没分。
    core_ids = {task["id"] for task in load_session_benchmark(DEFAULT_SESSION_BENCHMARK_PATH)["tasks"]}
    assert "long_horizon_spec_recall_20" in core_ids
    assert "long_horizon_spec_recall_200" not in core_ids


def test_a_broken_extended_task_still_fails_at_load_time(tmp_path):
    """校验必须跑在过滤之前，否则 extended 里的错误可以一直躲着。

    默认档不加载 extended 任务。如果校验跟着过滤走，一个 fixture 路径写错、
    断言名拼错的 extended 任务就永远不会报错——直到有人真去跑那一档，而那
    可能是几个月以后。这正是「声明退化成永不生效的注释」那个坑的又一种形态。
    """
    payload = {
        "schema_version": 1,
        "source": "probe",
        "tasks": [
            {
                "id": "ok_core",
                "question_type": "single_session_recall",
                "turns": [{"request": "hi", "outputs": [{"final": "hi"}], "expect": {"min_notes": 0}}],
            },
            {
                "id": "broken_extended",
                "tier": "extended",
                "question_type": "single_session_recall",
                "fixture_repo": "tests/fixtures/this_does_not_exist",
                "turns": [{"request": "hi", "outputs": [{"final": "hi"}], "expect": {"min_notes": 0}}],
            },
        ],
    }
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # 默认档只跑 core，但坏的那条仍然必须让加载失败。
    with pytest.raises(ValueError, match="fixture repo does not exist"):
        load_session_benchmark(path, repo_root=Path("."))


def test_an_unknown_tier_is_rejected_rather_than_silently_dropped(tmp_path):
    """写错档位名不能变成「这条任务从此不跑了」——那和删掉它没区别。"""
    payload = {
        "schema_version": 1,
        "source": "probe",
        "tasks": [
            {
                "id": "typo",
                "tier": "extendd",
                "question_type": "single_session_recall",
                "turns": [{"request": "hi", "outputs": [{"final": "hi"}], "expect": {"min_notes": 0}}],
            }
        ],
    }
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown tier"):
        load_session_benchmark(path, repo_root=Path("."))


def test_every_long_horizon_case_outside_the_ladder_runs_at_least_twenty_turns():
    """「长期任务」这四个字得有下限，否则 3 轮的 case 也能挂在这个名字下。

    阶梯那四条自带轮数，不在这条的管辖范围。
    """
    tasks = tasks_by_id()
    for task_id in ("long_horizon_reread_across_runs_24", "long_horizon_reread_in_run_26",
                    "long_horizon_stale_anchor_24", "long_horizon_stale_pointer_26",
                    "long_horizon_constraint_retained_100"):
        assert len(tasks[task_id]["turns"]) >= 20, task_id


def test_the_twenty_turn_case_is_the_control_for_the_hundred_turn_one():
    """对照组存在的全部理由：把「召回本身不行」和「被压缩丢了」分开。

    两条同时挂 = 召回压根不成立；只有 100 轮那条挂 = 确实是长度造成的。
    没有对照组的话，100 轮那条挂掉时这两种解释在报告上分不出来。
    """
    tasks = tasks_by_id()
    assert tasks["long_horizon_spec_recall_20"]["question_type"] == "long_horizon_recall"
    assert tasks["long_horizon_spec_recall_100"]["question_type"] == "long_horizon_recall"


def test_long_horizon_cases_declare_the_marker_on_both_channels():
    """标记串必须同时走两条独立的观测通道，只走一条等于只测了半个机制。

    `context_evidence` 查的是**组 prompt 那一刻它在不在 prompt 里**（harness 有没有
    把对的东西放进去）；`notes_contain` 查的是**它是被哪个机制放进去的**（召回选中了
    那条记忆）。前者过、后者挂说明它是靠原始历史蒙混过关的，那在 100 轮上不可持续。
    """
    tasks = tasks_by_id()
    for task_id in LONG_HORIZON_IDS + ("long_horizon_constraint_retained_100",):
        final = tasks[task_id]["turns"][-1]
        assert final["context_evidence"], task_id
        marker = final["context_evidence"][0]
        assert marker in final["expect"]["notes_contain"], task_id


def test_the_constraint_case_names_the_files_it_forbids():
    """禁令类 case 必须点名具体路径，否则 `forbidden_paths_untouched` 无从判起。"""
    task = tasks_by_id()["long_horizon_constraint_retained_100"]
    forbidden = task["turns"][-1]["expect"]["forbidden_paths"]
    assert forbidden == ["vendor/lib.py"]
    assert (Path(task["fixture_repo"]) / "vendor" / "lib.py").is_file()


def test_stale_cases_declare_which_half_of_the_mechanism_they_exercise():
    """窗口内和窗口外是两条代码路径，数据集必须分开声明，不能合成一个数。

    合成一个数会让「窗口内那半根本没跑」被窗口外那半的计数盖过去。
    """
    tasks = tasks_by_id()
    declared = {}
    for task_id in STALE_IDS:
        for turn in tasks[task_id]["turns"]:
            declared.update({key: turn["expect"][key] for key in turn.get("expect", {}) if key.startswith("stale_")})
        keys = {key for turn in tasks[task_id]["turns"] for key in turn.get("expect", {}) if key.startswith("stale_")}
        declared[task_id] = keys
    assert declared["stale_patch_anchor"] == {"stale_reads_min"}
    assert declared["stale_read_across_runs"] == {"stale_reads_min"}
    assert declared["stale_spill_pointer"] == {"stale_pointers_min"}


def test_the_spill_case_reads_a_file_that_actually_overflows(tmp_path):
    """落盘那条 case 的大文件必须**算出来**真的超过那一档的单条上限。

    按文件大小挑探针是错的：`read_file` 的返回值带行号前缀和路径头，比原文大约 35%，
    所以要卡阈值只能量 `run_tool()` 的返回值。这条挂掉说明 fixture 缩水了，那条
    case 会静默退化成一条普通的「读文件」会话，而报告照样全绿。
    """
    from codingforme import models
    from codingforme.context_manager import tool_output_limit
    from codingforme.models import FakeModelClient

    task = tasks_by_id()["stale_spill_pointer"]
    root = prepare_session_workspace(task, tmp_path / "ws")
    agent = DEFAULT_HARNESS.build(FakeModelClient([]), root)
    limit = tool_output_limit(agent.context_manager.total_budget)

    big = agent.run_tool("read_file", {"path": "src/core.py", "start": 1, "end": 400})
    assert (agent._last_tool_result_metadata or {}).get("tool_output_spilled") is True
    assert models.count_tokens(big, None) <= limit

    # 用来把它挤出最近窗口的填充模块，自己一个都不许落盘——落了就变成另一个探针。
    small = agent.run_tool("read_file", {"path": "src/mod00.py", "start": 1, "end": 20})
    assert (agent._last_tool_result_metadata or {}).get("tool_output_spilled") is not True
    assert models.count_tokens(small, None) < limit


def test_the_stale_anchor_fixture_forces_the_second_patch_through_the_first(tmp_path):
    """第二处改动的锚点必须绕不开第一处，否则「照着旧信息动手」没有作用对象。

    形状要求：`kind` 那行在 `preview()` 里一字不差地又出现一次（单独当 old_text 会
    命中两次、被 `patch_file` 的「恰好一次」打回），它下面那行同样重复（往下取上下文
    也消不掉歧义），于是唯一能消歧的邻居只剩它上面那行——正是第一次 patch 刚改过的
    那行。缺了这个结构，作废机制照常触发而开关两臂跑出来完全相同。
    """
    task = tasks_by_id()["stale_patch_anchor"]
    root = prepare_session_workspace(task, tmp_path / "ws")
    lines = (root / "src" / "render.py").read_text(encoding="utf-8").splitlines()

    kind_line = '    kind = config.get("kind", "legacy")'
    below = "    return build(mode, kind)"
    assert lines.count(kind_line) == 2, "锚点行不重复，第二次 patch 不需要借第一处消歧"
    assert lines.count(below) == 2, "锚点下面那行不重复，往下取上下文就能消歧"
    above = lines[lines.index(kind_line) - 1]
    assert above == '    mode = config.get("mode", "legacy")', "唯一能消歧的邻居必须是第一处改动"


# --- 适用性：断言在正确轨迹上必须**判过**，不能一律「不适用」 ----------------


def test_the_two_reread_cases_exercise_two_different_checks(tmp_path):
    """两条 case 分工不同，各自让一条断言真的**判过**。

    正确的做法是「改完再读一遍」——那次重读命中写操作豁免。适用性如果判在豁免之后，
    这两条断言在整份数据集上一次都不会产出，而「查过了、没问题」和「压根没查」在
    工件上又长得一模一样。写成同一个形状同样不行：那样其中一条永远没有作用对象。
    """
    tasks = tasks_by_id()

    # 跨运行那条：config.py 的两次读分在第 1 次和第 3 次运行里。
    _, index = run_session_task(tasks["reread_after_unrelated_calls"], tmp_path / "across")
    across = check_no_redundant_reread_across_runs(index.sessions[0], {})
    assert across is not None, "跨运行重读断言不适用，这条 case 什么都没测"
    assert across[3] is True, across[4]

    # 运行内那条：config.py 只在最后一次运行里碰，读→改→再读全在同一次运行内。
    _, index = run_session_task(tasks["reread_across_restart"], tmp_path / "within")
    session = index.sessions[0]
    within = [check_no_redundant_reread(run, DEFAULT_HARNESS) for run in session.runs]
    assert any(item is not None for item in within), "运行内重读断言不适用"
    assert all(item[3] for item in within if item is not None)


def test_the_stale_cases_actually_push_the_counters(tmp_path):
    """三条 stale case 必须真的把 harness 的计数器顶起来。

    `stale_read_invalidation` 在 14 个基准任务上 `stale_read_count` 合计为 0——
    机制正确、有单测、真实负载上一次都没执行过。这条用例就是那个状态的守卫。
    """
    tasks = tasks_by_id()
    observed = {}
    for task_id in STALE_IDS:
        task = tasks[task_id]
        _, index = run_session_task(task, tmp_path / task_id)
        session = index.sessions[0]
        counters = [
            ((turn.prompt_metadata or {}).get("history") or {})
            for run in session.runs
            for turn in run.turns
        ]
        observed[task_id] = (
            max([int(item.get("stale_read_count", 0) or 0) for item in counters] or [0]),
            max([int(item.get("stale_pointer_count", 0) or 0) for item in counters] or [0]),
        )
    assert observed["stale_patch_anchor"][0] >= 1
    assert observed["stale_read_across_runs"][0] >= 1
    assert observed["stale_spill_pointer"][1] >= 1


# --- 证伪：违规轨迹必须被判挂 ------------------------------------------------


def test_a_reread_of_unchanged_content_is_caught_even_across_other_calls():
    """`no_repeated_calls` 覆盖不到的那一半：隔着别的调用再绕回来读同一个文件。

    闸口的 `repeated_tool_call()` 只看 history 里最后两条工具记录，中间隔一个别的
    调用就完全查不到。真实跑批里的浪费恰恰是这个形状（读 a → 读 b → 读 c → 又读 a），
    每一次都白烧一个约 17 秒的往返。
    """
    run = synthetic_run(
        [
            [read_call("config.py")],
            [read_call("handlers/alpha.py")],
            [read_call("handlers/bravo.py")],
            [read_call("config.py")],
        ]
    )

    verdict = check_no_redundant_reread(run, DEFAULT_HARNESS)

    assert verdict is not None
    assert verdict[3] is False
    offender = verdict[4]["offenders"][0]
    assert offender["path"] == "config.py"
    assert offender["first_turn"] == 1 and offender["turn"] == 4
    assert offender["intervening_calls"] == 2


@pytest.mark.parametrize(
    "middle, reason",
    [
        ([write_call("config.py")], "内容变了，重读是必须的"),
        ([read_call("f%d.py" % index) for index in range(8)], "前一次读已经掉出最近窗口，重读是设计好的取回路径"),
    ],
)
def test_a_legitimate_reread_is_not_reported(middle, reason):
    """三条豁免必须真的豁免，否则这条断言会把上下文治理的正常工作记成模型的毛病。"""
    run = synthetic_run([[read_call("config.py")]] + [[call] for call in middle] + [[read_call("config.py")]])

    verdict = check_no_redundant_reread(run, DEFAULT_HARNESS)

    assert verdict is not None, "豁免不该连适用性一起豁免掉：" + reason
    assert verdict[3] is True, verdict[4]


def test_a_spilled_read_may_be_read_again():
    """落过盘的结果在上下文里只剩预览和一行指针，指针本身就在教模型再读一次。"""
    run = synthetic_run(
        [[read_call("src/core.py", end=400, spilled=True)], [read_call("a.py")], [read_call("src/core.py", end=400)]]
    )

    verdict = check_no_redundant_reread(run, DEFAULT_HARNESS)

    assert verdict is not None
    assert verdict[3] is True, verdict[4]


def test_a_write_by_another_tool_still_counts_as_a_change():
    """「写过哪些文件」取自工作区快照的 sha256 差异，不是 args 里的 path。

    args 读不出 `run_shell` 改的文件、`run_plan` 内层改的文件，也读不出「args 里有
    path 但一个字节都没写成」的失败写。用 args 判的话这条必漏。
    """
    shell = {
        "name": "run_shell",
        "args": {"command": "sed -i s/7/9/ config.py"},
        "tool_status": "ok",
        "affected_paths": ["config.py"],
    }
    run = synthetic_run([[read_call("config.py")], [shell], [read_call("config.py")]])

    verdict = check_no_redundant_reread(run, DEFAULT_HARNESS)

    assert verdict is not None
    assert verdict[3] is True, verdict[4]


def test_a_reread_after_a_restart_is_caught_by_the_session_level_check():
    """重启之后重读一个没变过的文件，该动的是 resume 带了什么，不是模型。

    所以这条的 subject 是 harness，而运行内那条是 model——同一套判据、不同的主语。
    """
    first = synthetic_run([[read_call("config.py")]], run_seq=1, run_id="r1")
    second = synthetic_run([[read_call("config.py")]], run_seq=2, run_id="r2", prior_runs=[first])
    session = SessionRecord(session_id="s1", runs=[first, second])

    verdict = check_no_redundant_reread_across_runs(session, {})

    assert verdict is not None
    assert verdict[3] is False
    offender = verdict[4]["offenders"][0]
    assert offender["first_run_seq"] == 1 and offender["run_seq"] == 2
    assert ASSERTION_SUBJECTS["no_redundant_reread_across_runs"] == "harness"
    assert ASSERTION_SUBJECTS["no_redundant_reread"] == "model"


def test_the_session_level_check_does_not_double_report_within_one_run():
    """运行内的重复由 L1 那条报，会话层不能把同一件事再数一遍。"""
    run = synthetic_run([[read_call("a.py")], [read_call("b.py")], [read_call("a.py")]])
    session = SessionRecord(session_id="s1", runs=[run])

    assert check_no_redundant_reread_across_runs(session, {}) is None
    assert check_no_redundant_reread(run, DEFAULT_HARNESS)[3] is False


def test_a_missing_stale_flag_is_reported_with_the_counter_that_fell_short():
    """归因要说清是哪一半没跑起来（窗口内 / 窗口外），只说「不通过」没法定位。"""
    run = synthetic_run([[read_call("a.py")]])
    run.turns[0].prompt_metadata = {"history": {"stale_read_count": 0, "stale_pointer_count": 0}}
    session = SessionRecord(session_id="s1", runs=[run])

    verdict = check_stale_reads_flagged(session, {1: {"stale_reads_min": 1, "stale_pointers_min": 2}})

    assert verdict[3] is False
    fields = {offender["field"]: offender for offender in verdict[4]["offenders"]}
    assert fields["stale_read_count"]["observed"] == 0
    assert fields["stale_pointer_count"]["min"] == 2


def test_stale_flags_are_not_asserted_when_the_dataset_says_nothing():
    """没声明就不适用，不能拿没测过的事当通过。"""
    run = synthetic_run([[read_call("a.py")]])
    session = SessionRecord(session_id="s1", runs=[run])

    assert check_stale_reads_flagged(session, {}) is None
    assert check_forbidden_paths_untouched(session, {}) is None


def test_touching_a_forbidden_path_is_caught_no_matter_which_run_does_it():
    """禁令是第 1 轮立的，违反它的动作可能发生在第 100 轮——按轮判会漏掉全部。"""
    first = synthetic_run([[read_call("src/app.py")]], run_seq=1, run_id="r1")
    second = synthetic_run([[write_call("vendor/lib.py")]], run_seq=2, run_id="r2", prior_runs=[first])
    session = SessionRecord(session_id="s1", runs=[first, second])

    verdict = check_forbidden_paths_untouched(session, {1: {"forbidden_paths": ["vendor/lib.py"]}})

    assert verdict[3] is False
    offender = verdict[4]["offenders"][0]
    assert offender["run_seq"] == 2 and offender["touched"] == ["vendor/lib.py"]


# --- 注册契约 ----------------------------------------------------------------


def test_every_new_assertion_declares_whose_behaviour_it_judges():
    """漏标 subject 会把模型的问题记到 harness 头上（缺省值是 harness）。"""
    names = {
        "no_redundant_reread",
        "no_redundant_reread_across_runs",
        "stale_reads_flagged",
        "forbidden_paths_untouched",
    }
    assert names <= set(ASSERTION_SUBJECTS)
    registered = {check.__name__ for check in TRAJECTORY_CHECKS} | {check.__name__ for check in SESSION_CHECKS}
    assert {"check_" + name for name in names} <= registered


def test_disabling_stale_read_invalidation_fails_exactly_the_stale_cases(tmp_path):
    """证伪：关掉过期读作废，挂的必须**只是** stale 那五条（3 条短会话 + 2 条 >=20 轮）。

    没有这条，五条 case 全绿既可能是「作废机制成立了」，也可能是「这几条根本没触发
    过它」——两者在报告上长得一模一样。长会话那两条尤其要单独验：读与改排在会话
    末尾才在窗口内，多垫几轮噪声就会退化成「窗口外、不计数」，那时它测的是反面。
    """
    tasks = tasks_by_id()
    off = get_harness("no_stale_read")
    for task_id in STALE_IDS:
        task = tasks[task_id]
        _, index = run_session_task(task, tmp_path / task_id, harness=off)
        session = index.sessions[0]
        expectations = {
            seq: dict(turn.get("expect", {}) or {})
            for seq, turn in enumerate(task["turns"], start=1)
            if turn.get("expect")
        }
        verdict = check_stale_reads_flagged(session, expectations)
        assert verdict is not None and verdict[3] is False, f"{task_id}：关掉作废之后仍然通过"

    # 召回类会话跟这个开关无关，不该被连坐。
    recall = tasks["recall_build_command"]
    _, index = run_session_task(recall, tmp_path / "recall", harness=off)
    assertions = score_session(index.sessions[0], {2: {"notes_contain": ["uv run pytest"], "min_notes": 1}})
    assert all(item.passed for item in assertions), [item.assertion_id for item in assertions if not item.passed]
