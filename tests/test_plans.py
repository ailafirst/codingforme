"""受限编排（`run_plan`）的验收：沙箱边界、闸口、记账、trace。

分八组，对应八个「做错了就会很贵」的地方：

1. **沙箱** —— 一段计划能做什么、不能做什么。这里的用例大多是逃逸尝试，
   它们必须在**一个工具都还没执行**的时候就被打回。
2. **闸口** —— 计划里的调用和模型直接发的调用受同一套约束：路径不越界、
   只读态挡写、工具白名单、审批。编排不是绕过闸口的旁路。
3. **记账** —— 一段 N 个调用的计划恰好消耗 max(1, N) 步。少算一步，
   `max_steps` 连同它保护的一切（只读粒度、审批粒度）都会被一个 for 循环绕过。
4. **trace** —— 每个内层调用各写一条 `tool_executed`，否则整套 L1 判分对
   计划里发生的事完全失明，而失明在报告里长得和「通过」一模一样。
5. **转录预算与结果过滤** —— 回给模型多少由计划自己决定（`print` 优先），
   两块内容不能交织。这两条都是 live 抓出来的，当时单测全绿。
6. **计数与累加** —— `+=` 和 `sum` 是 live 里最常见的两种打回；`try` 相反，
   它是**被否掉的**，因为它唯一能捕到的是沙箱守卫自己抛的异常。
7. **结构化输出** —— `format="paths"` 让计划不必对工具输出做字符串手术，
   两条 search 实现路径也必须给出同一种形状。
8. **live 数据点名的三个缺件** —— 每条都对应探针里一段被打回的真实计划。
"""

import json

import pytest

from codingforme import plans
from codingforme.eval.harness import HarnessSpec, get_harness
from codingforme.models import FakeModelClient, final_answer, tool_call
from codingforme.tools import _PLAN_EXAMPLES, plan_callable_tools, to_openai_function_specs

READ_EVERY_PY = (
    'listing = list_files(path=".")\n'
    "for row in lines(listing):\n"
    '    if "[F] " in row and ".py" in row:\n'
    '        read_file(path=replace(row, "[F] ", ""), start=1, end=80)\n'
)


def build_workspace(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("hi\n", encoding="utf-8")
    return tmp_path


def build_agent(tmp_path, outputs, **kwargs):
    """装一个开着 run_plan 的 agent。默认变体不带它——见 DEFAULT_FEATURE_FLAGS。"""
    flags = {"plan_tool": True}
    flags.update(kwargs.pop("feature_flags", {}))
    spec = HarnessSpec(name="plan-test", feature_flags=flags, **kwargs)
    return spec.build(FakeModelClient(list(outputs)), build_workspace(tmp_path))


def trace_events(tmp_path):
    runs = sorted((tmp_path / ".codingforme" / "runs").iterdir())
    lines = (runs[-1] / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def fake_call_tool(name, args):
    if name == "list_files":
        return "[F] a.py\n[F] b.py"
    return f"<{name} {args}>"


# --- 1. 沙箱 -----------------------------------------------------------------


@pytest.mark.parametrize(
    "source, needle",
    [
        # 属性访问是整个沙箱最关键的那条边：能取属性，`().__class__.__bases__`
        # 那条经典逃逸路径就成立，值域也就不再限于 str/int/list/dict。
        ('read_file(path=().__class__.__name__)', "attribute access is not allowed"),
        # import 单独给一条针对性的话：实测它是最常见的打回（7 次计划里 3 次），
        # 而模型拿到一句 "Import is not allowed" 会换成 `from os import path` 再试。
        ("import os", "A plan is not a Python program"),
        ("from os import path", "no standard library"),
        ("__import__('os')", "is not a tool you can call"),
        ("open('x')", "is not a tool you can call"),
        ("eval('1')", "is not a tool you can call"),
        ("exec('1')", "is not a tool you can call"),
        ("while True:\n    read_file(path='a')", "While is not allowed"),
        ("def f():\n    read_file(path='a')", "FunctionDef is not allowed"),
        ("try:\n    read_file(path='a')\nexcept: pass", "Try is not allowed"),
        ("with open('x') as f:\n    read_file(path='a')", "With is not allowed"),
        ("lambda: read_file(path='a')", "Lambda is not allowed"),
        # 乘法被排除是因为 "x" * 10**9 能在一步之内把内存撑爆。
        ("read_file(path='a' * 99)", "Mult is not allowed"),
        # `+=` 现在是允许的（见第 6 组），但 `*=` 仍然不是——同一个理由。
        ("read_file(path='a')\nx *= 99", "Mult is not allowed"),
        # 嵌套编排：内层能写的东西外层都能写，只会让步数记账多一层递归。
        ("run_plan(plan='read_file(path=\"a\")')", "is not a tool you can call"),
    ],
)
def test_the_sandbox_rejects_it_before_any_tool_runs(source, needle):
    with pytest.raises(plans.PlanError) as excinfo:
        plans.check_plan(source, {"read_file", "list_files"})
    assert needle in str(excinfo.value)


def test_a_plan_that_calls_no_tool_is_rejected():
    """一个工具都不调的"计划"没有意义，而且会白占一步。"""
    with pytest.raises(plans.PlanError, match="at least one tool"):
        plans.check_plan("x = len('abc')", {"read_file"})


def test_the_limits_are_enforced():
    """四条硬上限。防的不是攻击，是模型写出一个死循环把进程挂住。"""
    # 源码长度
    with pytest.raises(plans.PlanError, match="the limit is"):
        plans.check_plan("read_file(path='a')" + ("# pad" + chr(10)) * 3000, {"read_file"})
    # range() 产出的条数
    with pytest.raises(plans.PlanError, match="the limit is"):
        plans.execute_plan(
            "for i in range(9999):" + chr(10) + "    read_file(path=str(i))",
            fake_call_tool,
            {"read_file"},
            budget=999,
        )
    # 循环条数（迭代一个手写的长列表）
    long_list = "[" + ", ".join(f"'{index}'" for index in range(plans.MAX_LOOP_ITEMS + 1)) + "]"
    with pytest.raises(plans.PlanError, match="exceeds the limit"):
        plans.execute_plan(
            f"for i in {long_list}:" + chr(10) + "    read_file(path=i)",
            fake_call_tool,
            {"read_file"},
            budget=999,
        )
    # 解释器运算步数（两层循环各自都在上限之内，乘起来不是）
    with pytest.raises(plans.PlanError, match="interpreter steps"):
        plans.execute_plan(
            "read_file(path='a')" + chr(10)
            + "for i in range(200):" + chr(10)
            + "    for j in range(200):" + chr(10)
            + "        x = i",
            fake_call_tool,
            {"read_file", "list_files"},
            budget=999,
        )


def test_a_plan_that_fails_half_way_keeps_the_calls_that_already_ran(tmp_path):
    """跑了一半才失败时，已经执行的调用必须原样回给模型。

    丢掉它们等于让模型以为那些都没发生过，而其中可能包含已经落盘的写操作——
    它下一轮会基于一个错误的前提继续推。
    """
    plan = 'read_file(path="a.py", start=1, end=5)' + chr(10) + "read_file(path=undefined_name)" + chr(10)
    agent = build_agent(tmp_path, [tool_call("run_plan", plan=plan), final_answer("done")])
    agent.ask("go")

    executed = [e for e in trace_events(tmp_path) if e["event"] == "tool_executed"]
    assert [e["name"] for e in executed] == ["read_file"]
    plan_event = [e for e in trace_events(tmp_path) if e["event"] == "plan_executed"][0]
    assert "'undefined_name' is not defined" in plan_event["result"]
    assert "plan executed 1 tool call(s)." in plan_event["result"]


def test_the_documented_example_actually_runs(tmp_path):
    """工具示例是进 prefix 和报错信息的，它必须是真能跑的，不能是伪代码。

    这条是踩过坑补的：`run_shell` 的示例曾经写着本机开发用的 `uv run --with
    pytest ...`，而评测 fixture 仓库里既没有 uv 也没装 pytest——等于主动教模型
    烧掉一个约 17 秒的往返。
    """
    agent = build_agent(tmp_path, [])
    plan = json.loads(agent.tool_example("run_plan"))["plan"]
    agent = build_agent(tmp_path, [tool_call("run_plan", plan=plan), final_answer("done")])
    agent.ask("read every python file")

    executed = [e for e in trace_events(tmp_path) if e["event"] == "tool_executed"]
    assert [e["name"] for e in executed] == ["list_files", "read_file", "read_file"]
    assert {e["args"]["path"] for e in executed if e["name"] == "read_file"} == {"a.py", "b.py"}


# --- 2. 闸口 -----------------------------------------------------------------


def test_a_path_escape_inside_a_plan_is_rejected_like_any_other_call(tmp_path):
    agent = build_agent(
        tmp_path,
        [tool_call("run_plan", plan='read_file(path="../outside.txt", start=1, end=5)'), final_answer("done")],
    )
    agent.ask("peek outside")

    executed = [e for e in trace_events(tmp_path) if e["event"] == "tool_executed"]
    assert len(executed) == 1
    assert executed[0]["tool_status"] == "rejected"
    assert executed[0]["security_event_type"] == "path_escape"


def test_read_only_blocks_the_write_inside_a_plan_but_not_the_read(tmp_path):
    """审批粒度按**内层调用**走，不按计划走。

    这是把 `run_plan` 标成 `risky: False` 的全部理由：标成 risky 的话，一段
    只读的计划在只读变体下也会被整段挡掉，那是错的。
    """
    plan = 'read_file(path="a.py", start=1, end=5)\npatch_file(path="a.py", old_text="x = 1", new_text="x = 2")\n'
    agent = build_agent(tmp_path, [tool_call("run_plan", plan=plan), final_answer("done")], read_only=True)
    agent.ask("bump x")

    executed = [e for e in trace_events(tmp_path) if e["event"] == "tool_executed"]
    assert [(e["name"], e["tool_status"]) for e in executed] == [
        ("read_file", "ok"),
        ("patch_file", "rejected"),
    ]
    assert executed[1]["security_event_type"] == "read_only_block"
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\n"


def test_a_plan_cannot_reach_a_tool_the_allowlist_removed(tmp_path):
    """工具白名单自动生效：解释器只认识注册表里的名字。

    而且是在**校验阶段**就打回——计划里前面那个合法的调用一个都不会执行，
    工作区不会停在半截状态。
    """
    spec = HarnessSpec(
        name="plan-readonly-tools",
        feature_flags={"plan_tool": True},
        tools_allowlist=("read_file", "run_plan"),
    )
    plan = 'read_file(path="a.py", start=1, end=5)\nwrite_file(path="pwn.txt", content="x")\n'
    agent = spec.build(
        FakeModelClient([tool_call("run_plan", plan=plan), final_answer("done")]),
        build_workspace(tmp_path),
    )
    agent.ask("write something")

    assert not (tmp_path / "pwn.txt").exists()
    assert [e["event"] for e in trace_events(tmp_path) if e["event"] == "tool_executed"] == []
    plan_events = [e for e in trace_events(tmp_path) if e["event"] == "plan_executed"]
    assert "'write_file' is not a tool you can call" in plan_events[0]["result"]


def test_the_registry_is_empty_of_run_plan_unless_the_flag_is_on(tmp_path):
    """默认变体不带这个工具——两边的跑批数据因此可比。"""
    default_agent = get_harness("full").build(FakeModelClient([]), build_workspace(tmp_path))
    assert "run_plan" not in default_agent.tools
    assert "run_plan" not in default_agent.prefix
    assert get_harness("plan_tool").resolved_feature_flags()["plan_tool"] is True


# --- 3. 记账 -----------------------------------------------------------------


def test_a_plan_of_n_calls_costs_exactly_n_steps(tmp_path):
    """否则一个 for 循环就能在一步之内做完二十件事，`max_steps` 形同虚设。"""
    agent = build_agent(tmp_path, [tool_call("run_plan", plan=READ_EVERY_PY), final_answer("done")], max_steps=8)
    agent.ask("read every python file")

    runs = sorted((tmp_path / ".codingforme" / "runs").iterdir())
    state = json.loads((runs[-1] / "task_state.json").read_text(encoding="utf-8"))
    assert state["tool_steps"] == 3  # list_files + 两个 read_file
    assert state["stop_reason"] == "final_answer_returned"


def test_the_budget_stops_a_plan_part_way_and_says_so(tmp_path):
    """预算在计划中途用完时停下，已经跑完的调用保留，并明确告诉模型还有没跑的。

    静默丢弃会让模型以为整段都执行过了——那是最贵的一种错误，因为它会基于
    一个没发生过的写操作继续往下推。
    """
    agent = build_agent(tmp_path, [tool_call("run_plan", plan=READ_EVERY_PY), final_answer("done")], max_steps=2)
    agent.ask("read every python file")

    executed = [e for e in trace_events(tmp_path) if e["event"] == "tool_executed"]
    assert [e["name"] for e in executed] == ["list_files", "read_file"]
    plan_event = [e for e in trace_events(tmp_path) if e["event"] == "plan_executed"][0]
    assert plan_event["plan_stopped_reason"] == "step_budget_exhausted"
    assert "step budget ran out" in plan_event["result"]


def test_an_invalid_plan_still_costs_one_step_and_returns_a_string(tmp_path):
    """`run_tool()` 的契约：任何失败都变成模型下一轮能消费的字符串，不抛异常。"""
    agent = build_agent(tmp_path, [tool_call("run_plan", plan="read_file(path=q)"), final_answer("done")])
    agent.ask("go")

    runs = sorted((tmp_path / ".codingforme" / "runs").iterdir())
    state = json.loads((runs[-1] / "task_state.json").read_text(encoding="utf-8"))
    assert state["tool_steps"] == 1
    plan_event = [e for e in trace_events(tmp_path) if e["event"] == "plan_executed"][0]
    # 一个调用都没跑成 → 记成执行失败，而不是"成功但结果是一句报错"。
    assert plan_event["tool_status"] == "error"
    assert "'q' is not defined" in plan_event["result"]


# --- 4. trace ----------------------------------------------------------------


def test_every_inner_call_lands_in_the_trace_as_its_own_tool_executed(tmp_path):
    """没有这条，L1 的 path_confined / read_before_patch / 工具白名单三条断言
    对计划里发生的一切完全失明，而失明在报告里长得和「通过」一模一样。"""
    plan = 'read_file(path="a.py", start=1, end=5)\npatch_file(path="a.py", old_text="x = 1", new_text="x = 2")\n'
    agent = build_agent(tmp_path, [tool_call("run_plan", plan=plan), final_answer("done")])
    agent.ask("bump x")

    events = trace_events(tmp_path)
    executed = [e for e in events if e["event"] == "tool_executed"]
    assert [e["name"] for e in executed] == ["read_file", "patch_file"]
    assert [e["call_index"] for e in executed] == [0, 1]
    assert all(e["via"] == "run_plan" for e in executed)
    # 外层那条**不是** tool_executed：混进去会让同一次执行被数两遍，所有按
    # 调用聚合的指标（calls_per_turn、L1 各条断言）都翻倍。
    assert [e["event"] for e in events].count("plan_executed") == 1
    assert not any(e["name"] == "run_plan" for e in executed)


def test_the_outer_plan_event_reports_the_workspace_change(tmp_path):
    """`run_plan` 自己不需要审批，但它通过内层调用改到了文件。

    不在它前后拍工作区快照的话，trace 里那条记录会说「没改动过任何文件」——
    那是假的，而且是最难查的一种假：所有按 affected_paths 做的分析都会漏掉它。
    """
    plan = 'read_file(path="a.py", start=1, end=5)\npatch_file(path="a.py", old_text="x = 1", new_text="x = 2")\n'
    agent = build_agent(tmp_path, [tool_call("run_plan", plan=plan), final_answer("done")])
    agent.ask("bump x")

    plan_event = [e for e in trace_events(tmp_path) if e["event"] == "plan_executed"][0]
    assert plan_event["workspace_changed"] is True
    assert plan_event["affected_paths"] == ["a.py"]
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 2\n"


def test_the_history_keeps_exactly_one_tool_message_for_the_whole_plan(tmp_path):
    """messages 协议的硬约束：assistant 的每个 tool_call 必须恰好配一条 tool 消息。

    内层调用不是 assistant 发出的 tool_call，所以它们**不能**各写一条 tool 消息
    ——多出来的那几条会让后端直接拒掉整个请求。它们只进 trace。
    """
    agent = build_agent(tmp_path, [tool_call("run_plan", plan=READ_EVERY_PY), final_answer("done")], max_steps=8)
    agent.ask("read every python file")

    tool_messages = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["name"] == "run_plan"
    assert "plan executed 3 tool call(s)." in tool_messages[0]["content"]


def test_the_task_allowlist_keeps_run_plan_but_still_governs_what_it_can_call(tmp_path):
    """元工具穿过白名单，它能调到的东西不穿过。

    没有前半句，这个变体在整份固定基准上直接是空操作——12 个任务声明的
    `allowed_tools` 全是 `read_file`/`patch_file`，`run_plan` 会被交集裁掉，
    而"变体开了但什么都没发生"在报告里长得和"变体没用"一模一样。
    没有后半句，它就成了绕过白名单的旁路。
    """
    spec = HarnessSpec(
        name="plan-with-task-allowlist",
        feature_flags={"plan_tool": True},
        tools_allowlist=("read_file", "patch_file"),
    )
    agent = spec.build(FakeModelClient([]), build_workspace(tmp_path))

    assert sorted(agent.tools) == ["patch_file", "read_file", "run_plan"]
    # 声明值里没有 run_plan——它不是数据集声称的能力之一。
    assert agent.declared_tools_allowlist == ("patch_file", "read_file")
    # 计划里能调的仍然只有白名单剩下的那些，而且 run_plan 不能套 run_plan。
    assert plans.check_plan('read_file(path="a.py", start=1, end=5)', plan_callable_tools(agent))
    with pytest.raises(plans.PlanError, match="'write_file' is not a tool"):
        plans.check_plan('write_file(path="x", content="y")', plan_callable_tools(agent))
    with pytest.raises(plans.PlanError, match="'run_plan' is not a tool"):
        plans.check_plan('run_plan(plan="read_file(path=\'a\')")', plan_callable_tools(agent))


def test_print_writes_into_the_transcript_instead_of_stdout():
    """`print` 必须可用,而且要落进回给模型的转录里。

    这条是探针实验补的:模型自发写出的 9 段计划里有 4 段带 `print(...)`——它在写
    Python,`print` 是反射性的。`print` 不在白名单里时整段计划会在校验阶段被打回、
    一个工具都不执行,那一次模型往返(约 17 秒)就白烧了。
    """
    result = plans.execute_plan(
        'print("checking")' + chr(10) + 'read_file(path="a.py")',
        fake_call_tool,
        {"read_file"},
        budget=9,
    )
    assert "checking" in result.transcript.splitlines()
    assert len(result.calls) == 1


@pytest.mark.parametrize(
    "allowlist",
    [None, ("read_file", "patch_file"), ("read_file",)],
)
def test_the_example_only_ever_calls_tools_the_registry_actually_has(tmp_path, allowlist):
    """示例本身必须是一段**在当前注册表下能跑通**的计划。

    这条是真实跑批抓到的,单元测试当时全绿:静态示例写的是 `list_files(...)`,而
    12 个基准任务的白名单里一个都没有 `list_files`。模型照抄示例,13 次计划 13 次
    被打回,每次白烧一个约 17 秒的往返——而报错信息里附的示例还是同一份,于是它
    照着改也改不对。示例进两个通道(prefix 的工具清单、`tools=` schema 的
    `parameters.examples`),两条都得对。
    """
    spec = HarnessSpec(
        name="example-check",
        feature_flags={"plan_tool": True},
        tools_allowlist=allowlist and (*allowlist, "run_plan"),
    )
    agent = spec.build(FakeModelClient([]), build_workspace(tmp_path))

    example = agent.tool_example("run_plan")
    assert example, f"注册表 {sorted(agent.tools)} 下没有可用示例"
    plan = json.loads(example)["plan"]
    # 用真正的校验器验:示例必须能通过 run_plan 自己的参数校验。
    plans.check_plan(plan, plan_callable_tools(agent))

    # 两个通道给的是同一份。
    specs = to_openai_function_specs(agent.tools)
    schema_example = next(
        item["function"]["parameters"]["examples"][0]
        for item in specs
        if item["function"]["name"] == "run_plan"
    )
    assert schema_example["plan"] == plan
    assert f"example args: {example}" in agent.prefix


def test_the_example_shows_a_dependency_not_just_a_batch(tmp_path):
    """示例必须展示「后一个调用依赖前一个」,那是这个工具唯一比批量调用多出来的能力。

    只并排列几个调用的示例,等于在教模型用它做一件批量调用已经能做的事——而实测
    模型本来就已经在这么用它了(探针:3 段计划全部把 search 的结果硬编码进列表)。
    """
    agent = build_agent(tmp_path, [])
    plan = json.loads(agent.tool_example("run_plan"))["plan"]
    # 依赖的表现形式:循环变量、或者把上一个调用的返回值接着用。
    assert "for " in plan or " in " in plan


# --- 5. 转录预算与结果过滤 -----------------------------------------------------
#
# 这一组管的是「计划回给模型多少东西」。在它出现之前，每个内层调用的输出都被
# 原样倒进转录、且整段转录没有任何聚合上限——一个 for 循环读 20 个文件就能把
# 下一轮 prompt 撑爆。参照的是 Anthropic programmatic tool calling 的语义：
# 工具结果留在执行环境里，只有代码 print 出来的进上下文。
# https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling


NOISY_BODY = "line of noise\n" * 400


def noisy_call_tool(name, args):
    if name == "list_files":
        return "[F] a.py\n[F] b.py"
    return NOISY_BODY


def test_a_plan_that_prints_keeps_tool_results_out_of_the_reply():
    """同一段计划，加不加 print 决定结果进不进上下文。

    这是这个工具省上下文的唯一途径。不做的话「一次读 10 个文件再总结」和
    「分 10 轮读」占的上下文一样多，只省了往返。
    """
    read = 'text = read_file(path="a.py", start=1, end=999)\n'
    quiet = plans.execute_plan(read, noisy_call_tool, {"read_file"}, budget=9)
    loud = plans.execute_plan(
        read + 'print("noise lines:", len(lines(text)))', noisy_call_tool, {"read_file"}, budget=9
    )

    assert quiet.echo_results is True
    assert "line of noise" in quiet.transcript
    assert loud.echo_results is False
    assert "line of noise" not in loud.transcript
    assert "noise lines: 400" in loud.transcript
    # 两边都真的调了工具、都记了原始字节；差别只在回给模型多少。
    assert quiet.result_bytes == loud.result_bytes == len(NOISY_BODY)
    assert len(loud.transcript) < len(quiet.transcript) / 10


def test_filtering_never_hides_that_a_call_happened():
    """不变量：内层调用不能在回给模型的文本里静默消失。

    尤其是写操作——删掉那一行等于告诉模型 patch_file 没发生过，它可能把一个
    已经落盘的写再发一遍。所以过滤掉的只是**结果**，调用清单两种模式都写。
    """
    plan = (
        'read_file(path="a.py", start=1, end=999)\n'
        'patch_file(path="a.py", old_text="x", new_text="y")\n'
        'print("done")'
    )
    result = plans.execute_plan(plan, noisy_call_tool, {"read_file", "patch_file"}, budget=9)

    assert result.echo_results is False
    assert "[1] read_file(" in result.transcript
    assert "[2] patch_file(" in result.transcript
    assert "new_text='y'" in result.transcript
    # 结果本身没回显，但体积回了——模型据此知道那边有多少东西。
    assert f"-> {len(NOISY_BODY)} chars" in result.transcript


def test_print_detection_is_static_so_the_shape_never_depends_on_a_branch():
    """`print` 写在一个走不到的分支里，也算「这段计划要自己控制输出」。

    做成运行时判断的话，同一段计划两次执行可能给出不同形状的转录（这次进了
    分支就没结果、下次没进就有结果），模型无从建立预期。
    """
    plan = (
        'text = read_file(path="a.py", start=1, end=999)\n'
        'if "never" in text:\n'
        '    print("unreachable")\n'
    )
    result = plans.execute_plan(plan, noisy_call_tool, {"read_file"}, budget=9)
    assert result.echo_results is False
    assert "line of noise" not in result.transcript


def test_a_runaway_transcript_is_clipped_and_the_notice_says_what_to_do():
    """安全网：没写 print 的计划也不能把上下文撑爆。

    单个工具输出各自受 workspace.clip 的 4000 字符限制，但在这条上限出现之前
    N 个加起来完全没有聚合裁剪。
    """
    plan = "\n".join(f'read_file(path="f{i}.py", start=1, end=999)' for i in range(6))
    result = plans.execute_plan(plan, noisy_call_tool, {"read_file"}, budget=9)

    assert len(result.calls) == 6
    assert result.result_bytes == 6 * len(NOISY_BODY)
    from codingforme.models import count_tokens

    # 上限的单位是 token，断言也必须按 token 量——拿字符长度去比一个 token
    # 上限正是这次要消掉的那种混用。
    assert count_tokens(result.transcript_full) > plans.MAX_TRANSCRIPT_TOKENS
    assert count_tokens(result.transcript) <= plans.MAX_TRANSCRIPT_TOKENS
    assert "tokens of plan output omitted" in result.transcript
    # 撞上截断是模型最可能学会「自己过滤」的时刻，提示必须就在这里。
    assert "use print()" in result.transcript
    # 首尾都要留：开头是最早的调用，结尾是最后的调用与失败信息。
    assert result.transcript.startswith("[1] read_file(")
    assert "line of noise" in result.transcript.splitlines()[-1]


def test_the_reply_explains_why_results_are_missing(tmp_path):
    """规则是隐式的（写了 print 就不回显），所以 runtime 必须把它说出来。

    不说的话，模型看到调用清单后面少了结果，最可能的反应是把同一批调用原样
    再发一遍——正好撞上重复调用检测，白烧一个约 17 秒的往返。
    """
    plan = 'text = read_file(path="a.py", start=1, end=5)\nprint(len(lines(text)))'
    agent = build_agent(tmp_path, [tool_call("run_plan", plan=plan), final_answer("done")])
    agent.ask("count lines")

    content = [item for item in agent.session["history"] if item["role"] == "tool"][0]["content"]
    assert "tool results were not echoed" in content
    # 这句必须**关闭**这一轮而不是催下一轮。上一版结尾是 "Print what you need
    # next time."，live 验证里模型在已经印对了的情况下把它读成"你没拿全，再来
    # 一次"，于是把整个 fan-out 重跑——8 个样本里 5 个这样耗光步数预算。
    assert "complete; nothing was cut off" in content
    assert "next time" not in content


def test_the_plan_event_records_how_much_context_was_saved(tmp_path):
    """没有这几个字段，「结果不进上下文」在工件上和「什么都没省」长得一样。"""
    plan = 'text = read_file(path="big.py", start=1, end=999)\nprint(len(lines(text)))'
    agent = build_agent(tmp_path, [tool_call("run_plan", plan=plan), final_answer("done")])
    # 文件要足够大，否则"省下的上下文"是负数——一个 6 字节的文件，光调用清单
    # 那一行就比结果本身长。这也是这个机制真实的适用边界。
    (tmp_path / "big.py").write_text("z = 0\n" * 300, encoding="utf-8")
    agent.ask("count lines")

    event = [e for e in trace_events(tmp_path) if e["event"] == "plan_executed"][0]
    assert event["plan_results_echoed"] is False
    assert event["plan_transcript_clipped"] is False
    assert event["plan_result_bytes"] > event["plan_transcript_tokens"]


def test_the_examples_all_show_filtering_not_just_calling(tmp_path):
    """三档示例每一档都要 print，否则等于在教模型放弃这一半收益。"""
    for needed, text in _PLAN_EXAMPLES:
        plan = json.loads(text)["plan"]
        assert "print(" in plan, f"{needed} 那一档示例没有展示过滤"


def test_printed_output_is_one_block_not_interleaved_with_the_call_list():
    """过滤模式下 print 的内容必须自成一段完整清单，不能和调用清单交织。

    这条是 live 验证抓到的，单元测试当时全绿：交织版本里，一次读 12 个文件的
    计划只有 4 个文件命中并 print 出来，清单最后三行后面空空如也，模型自己写下
    "The plan output got truncated. Let me rerun it to ensure I capture all files"
    然后把整个 fan-out 又跑了一遍——8 个样本里 5 个这样耗光步数预算。骗人的地方
    在于 print 发生在循环体内，输出行落在**下一个**调用的清单行之前，看起来像
    结果和调用错位、末尾被截断。
    """
    plan = (
        'for path in ["a.py", "b.py", "c.py", "d.py"]:\n'
        '    text = read_file(path=path, start=1, end=99)\n'
        '    if "a" in path:\n'
        '        print(path, "hit")\n'
    )
    result = plans.execute_plan(plan, noisy_call_tool, {"read_file"}, budget=9)
    lines = result.transcript.splitlines()

    assert lines[0] == "Printed output:"
    printed_end = lines.index("Calls made (results not echoed):")
    printed = [line for line in lines[1:printed_end] if line]
    listing = [line for line in lines[printed_end + 1 :] if line]
    # 印出来的那一块里只有 print 的内容，没有任何调用清单行。
    assert printed == ["a.py hit"]
    assert not any(line.startswith("[") for line in printed)
    # 调用清单自成一块，四条齐全，一条 print 都不混在里面。
    assert len(listing) == 4
    assert all(line.startswith("[") and "chars" in line for line in listing)


def test_a_plan_that_prints_nothing_says_so_instead_of_showing_a_blank():
    """`print` 写在走不到的分支里时，输出块是空的——必须明说，不能给一片空白。"""
    plan = (
        'text = read_file(path="a.py", start=1, end=99)\n'
        'if "never-matches" in text:\n'
        '    print("hit")\n'
    )
    result = plans.execute_plan(plan, noisy_call_tool, {"read_file"}, budget=9)
    assert "(the plan printed nothing)" in result.transcript
    assert "line of noise" not in result.transcript


# --- 6. 计数与累加（补齐两个最常见的打回） -------------------------------------


COUNT_ROWS = 'n = 0\nfor row in lines(list_files(path=".")):\n    n += 1\nprint(n)'


def test_augmented_assignment_counts_without_a_temporary():
    """`n += 1` 是 live 里最常见的一种打回，而它就是 `n = n + 1`。"""
    result = plans.execute_plan(COUNT_ROWS, fake_call_tool, ["list_files"], budget=5)

    assert "2" in result.transcript


def test_augmented_assignment_keeps_the_same_operator_limits_as_binop():
    """只留 += 和 -=：`*=` 能在一步之内把内存撑爆，理由和 BinOp 那条一致。

    `*=` 在**静态检查**阶段就被打回（`ast.Mult` 不在白名单里），也就是一个工具
    都还没执行的时候——比走到解释器里再拦更好。解释器里那条 `_fail` 是第二道
    防线，正常路径够不到它。
    """
    with pytest.raises(plans.PlanError) as excinfo:
        plans.check_plan('n = 2\nn *= 500000\nread_file(path="a.py")', ["read_file"])

    assert "Mult is not allowed" in str(excinfo.value)


def test_augmented_assignment_on_an_unset_name_says_what_to_do():
    source = 'read_file(path="a.py")\nn += 1'
    with pytest.raises(plans.PlanError) as excinfo:
        plans.execute_plan(source, fake_call_tool, ["read_file"], budget=5)

    assert "assign it before using +=" in str(excinfo.value)


def test_sum_counts_matching_rows():
    """fan-out 计划最常见的收尾动作：数有多少个命中。"""
    source = 'rows = lines(list_files(path="."))\nprint(sum([1 for row in rows if ".py" in row]))'
    result = plans.execute_plan(source, fake_call_tool, ["list_files"], budget=5)

    assert "2" in result.transcript


def test_sum_on_strings_explains_what_to_write_instead():
    """报错是回给模型看的：Python 原生的 TypeError 说不出该改哪里。"""
    with pytest.raises(plans.PlanError) as excinfo:
        plans.execute_plan(
            'read_file(path="a.py")\nprint(sum(["a", "b"]))', fake_call_tool, ["read_file"], budget=5
        )

    message = str(excinfo.value)
    assert "item 0 is str" in message
    assert "sum([1 for row in rows if ...])" in message


def test_try_stays_banned_because_it_would_swallow_the_sandbox_guards():
    """`try` 不是漏掉的，是否掉的，而且理由和别的语法不一样。

    内层调用**不抛异常**——`run_tool()` 任何失败都返回 `error:` 开头的字符串。
    所以 `try` 在这里唯一能捕到的是 `PlanError`，也就是运算步数 / 循环条数 /
    字符串长度这几条硬上限自己抛的那个异常。放开它等于给模型一个吞掉沙箱守卫
    的语法。
    """
    source = 'try:\n    read_file(path="a.py")\nexcept:\n    pass\n'
    with pytest.raises(plans.PlanError):
        plans.check_plan(source, ["read_file"])


def test_a_failing_inner_call_returns_an_error_string_rather_than_raising(tmp_path):
    """这条是上一条的正面证据：不需要 try，因为失败根本不是异常。"""
    agent = build_agent(
        tmp_path,
        [
            tool_call("run_plan", plan='text = read_file(path="nope.py")\nprint(text)'),
            final_answer("done"),
        ],
    )
    reply = agent.ask("read a missing file inside a plan")

    assert reply == "done"
    events = [item for item in trace_events(tmp_path) if item.get("event") == "plan_executed"]
    assert len(events) == 1


# --- 7. 结构化输出（format="paths"） ------------------------------------------


def test_list_files_paths_format_needs_no_string_surgery(tmp_path):
    """`format="paths"` 直接给裸文件路径，计划不必再把 `[F] ` 抠掉。

    Anthropic 的 tool design 建议第一条就是「返回结构化数据」，理由是代码要
    反序列化工具结果。我们不能照抄 JSON——计划沙箱没有属性访问也没有标准库，
    给它一段 JSON 等于给它一段拆不开的字符串。所以这里的「结构化」是一行一个
    裸路径：`lines()` 直接就能用。
    """
    agent = build_agent(tmp_path, [])

    tree = agent.run_tool("list_files", {"path": "."})
    paths = agent.run_tool("list_files", {"path": ".", "format": "paths"})

    assert "[F] a.py" in tree
    assert "[F]" not in paths
    assert sorted(paths.splitlines()) == ["a.py", "b.py", "notes.md"]


def test_paths_format_never_lists_a_directory(tmp_path):
    """目录不该出现：这份输出的用途是逐个喂给 read_file。"""
    build_workspace(tmp_path)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("z = 3\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    assert "[D]" in agent.run_tool("list_files", {"path": "."})
    assert "sub" not in agent.run_tool("list_files", {"path": ".", "format": "paths"}).splitlines()


def test_listing_always_uses_forward_slashes(tmp_path):
    """Windows 上 `relative_to` 给的是反斜杠，而这串东西会被喂回 read_file。

    同一段计划在两个平台上形状不同，是最难查的一类问题。
    """
    build_workspace(tmp_path)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("z = 3\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    listing = agent.run_tool("list_files", {"path": "sub", "format": "paths"})

    assert listing == "sub/c.py"
    assert "\\" not in listing


def test_search_paths_format_returns_each_file_once(tmp_path):
    agent = build_agent(tmp_path, [])

    lines = agent.run_tool("search", {"pattern": "=", "path": "."})
    paths = agent.run_tool("search", {"pattern": "=", "path": ".", "format": "paths"})

    assert ":" in lines
    rows = paths.splitlines()
    assert rows == sorted(set(rows), key=rows.index)
    assert all(":" not in row for row in rows)
    assert "a.py" in rows


def test_search_paths_are_workspace_relative_not_host_absolute(tmp_path):
    """rg 按我们传进去的绝对路径原样打印，纯 Python 回退打印的是相对路径。

    两条实现路径必须给出同一种形状——否则同一段计划在装了 rg 和没装 rg 的机器上
    拿到的东西不一样。顺带这也不再把宿主的绝对路径漏进模型上下文。Windows 上
    还有个额外的坑：绝对路径带盘符，`E:\\ws\\a.py:1:text` 按 ':' 一切，头一段是 `E`。
    """
    agent = build_agent(tmp_path, [])

    for fmt in ("lines", "paths"):
        output = agent.run_tool("search", {"pattern": "x = 1", "path": ".", "format": fmt})
        assert not output.startswith("(no matches)"), output
        for row in output.splitlines():
            head = row.split(":")[0]
            assert head == "a.py", row


def test_an_unknown_format_is_rejected_with_the_legal_values(tmp_path):
    """静默退回默认值最糟：模型会以为拿到的是 paths，然后去拆一段 tree。"""
    agent = build_agent(tmp_path, [])

    message = agent.run_tool("list_files", {"path": ".", "format": "json"})

    assert "format must be one of tree, paths" in message


def test_a_plan_can_fan_out_over_the_structured_listing(tmp_path):
    """端到端：新格式让「列目录再逐个读」写成三行，且不含任何字符串手术。"""
    plan = (
        'for path in lines(list_files(path=".", format="paths")):\n'
        '    if ".py" in path:\n'
        "        text = read_file(path=path, start=1, end=20)\n"
        '        print(path, len(lines(text)))'
    )
    agent = build_agent(tmp_path, [tool_call("run_plan", plan=plan), final_answer("done")])

    assert agent.ask("survey the python files") == "done"
    event = next(item for item in trace_events(tmp_path) if item.get("event") == "plan_executed")
    assert event["plan_results_echoed"] is False
    inner = [
        item
        for item in trace_events(tmp_path)
        if item.get("event") == "tool_executed" and item.get("via") == "run_plan"
    ]
    assert [item["name"] for item in inner] == ["list_files", "read_file", "read_file"]


# --- 8. live 数据点名的三个缺件 ------------------------------------------------
#
# 这一组每条都对应 live 探针里一段**被打回的真实计划**。挑它们不是因为好补，
# 是因为 24 段模型自发写出的计划里有 11 段没通过静态检查，而这三样占了 5 段。


def test_the_most_common_way_to_count_is_a_generator_not_a_list():
    """`sum(1 for x in y if c)` —— live 里 3 段计划栽在这上面。

    生成器表达式当列表推导算:语义差别只有惰性,而这里的值域和 MAX_LOOP_ITEMS
    守卫让惰性没有任何意义。不认它的唯一后果就是把最常见的计数写法打回,而那
    几段计划的其余部分完全合法。
    """
    source = (
        'text = read_file(path="a.py")\n'
        'print(sum(1 for line in lines(text) if "read_file" in line))'
    )
    result = plans.execute_plan(source, fake_call_tool, ["read_file"], budget=3)

    assert result.printed_lines == ["1"]


def test_a_generator_comprehension_obeys_the_same_loop_ceiling_as_a_list_one():
    """放开语法不等于放开守卫:两种推导式走同一条求值路径,上限自然一致。"""
    long_list = "[" + ", ".join(f"'{index}'" for index in range(plans.MAX_LOOP_ITEMS + 1)) + "]"
    source = f'read_file(path="a.py")\nprint(sum(1 for i in {long_list} if i))'
    with pytest.raises(plans.PlanError, match="exceeds the limit"):
        plans.execute_plan(source, fake_call_tool, ["read_file"], budget=3)


def test_join_and_count_exist_because_attribute_access_does_not():
    """`join(sep, items)` / `count(text, sub)` —— 和 split/strip/replace 同一条理由。

    live 里模型写过 `"..".join(results)` 和 `text.count("TODO")`,两者都是属性
    访问、沙箱必拒,而它们又是汇总 fan-out 结果时最自然的两个动作。
    """
    source = (
        'rows = ["alpha", "beta"]\n'
        'text = read_file(path="a.py")\n'
        'print(join("|", rows))\n'
        'print(count(text, "read_file"))'
    )
    result = plans.execute_plan(source, fake_call_tool, ["read_file"], budget=3)

    assert result.printed_lines == ["alpha|beta", "1"]


def test_the_exact_plan_the_live_model_wrote_and_got_rejected_now_validates():
    """回归锚点:这段源码逐字来自 live 探针,当时被 `GeneratorExp is not allowed` 打回。"""
    source = (
        'for path in lines(list_files(path=".", format="paths")):\n'
        "    text = read_file(path=path, start=1, end=500)\n"
        '    count = sum(1 for line in lines(text) if "TODO" in line)\n'
        "    if count > 0:\n"
        '        print(f"{path}: {count}")\n'
    )

    plans.check_plan(source, {"list_files", "read_file"})


def test_the_transcript_cap_never_falls_below_a_single_tool_result():
    """转录的聚合上限不能小于单个工具结果的上限。

    小于的话，同一次 `read_file` 放进计划里反而看得更少，`run_plan` 变成纯负收益。
    这个坑是工具结果上限改成从 `total_budget` 派生之后新出现的:1M 档下单条能有
    14,791，而 `MAX_TRANSCRIPT_TOKENS` 还写死在 4,000。原设计的意思是「三个工具
    调用的额度」(12000 字符 ≈ 3 × 4000 字符)，倍数关系要跟着一起走。
    """
    from codingforme import plans
    from codingforme.context_manager import tool_output_limit

    for total_budget in (4335, 32000, 118335, 1_000_000):
        single = tool_output_limit(total_budget)
        aggregate = plans.transcript_limit(single)
        assert aggregate >= single, (
            f"total_budget={total_budget}: 转录上限 {aggregate} < 单条结果上限 {single}"
        )
        assert aggregate >= plans.MAX_TRANSCRIPT_TOKENS
