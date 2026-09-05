"""受限编排：让模型用一小段代码把多步工具调用串起来，而不是一次只发一个调用。

## 为什么存在

标准 function-calling 一轮能发多个工具调用，但那些调用之间**不能有依赖**——
参数必须在发出时就写死。于是「列出目录，然后把里面每个 .py 都读一遍」这类
最常见的形状，无论如何都要拆成两轮：先 `list_files`，等结果回来，再发 N 个
`read_file`。在这个后端上一次模型往返的固定开销约 17 秒，而工具执行本身在
35 次运行、140 次调用里合计只有 0.9 秒（占总耗时 0.02%）。省往返才有意义，
并行执行工具没有意义。

实测的头顶空间（在 12 任务 × 3 轮的真实跑批 trace 上重算）：保守口径（只合并
连续的只读调用）14/121 轮 ≈ 11.6%，激进口径（整次运行压成一段程序）55/121
≈ 45.5%。

## 为什么不是 run_shell

`run_shell` 是宿主 shell（只做了环境变量白名单）。把编排建在它上面，等于把
「模型只能调用声明过的工具」换成「模型能跑任意代码」——那是安全模型的实质
变更，不是一个性能优化该付的代价。所以这里自己写了一个解释器：

- **只解析，不 exec/eval。** `ast.parse` 之后自己遍历语法树，白名单之外的
  节点类型一律拒绝。没有 `import`、没有 `open`、没有 `__builtins__`。
- **没有属性访问。** `ast.Attribute` 直接判非法。这是最关键的一条：只要能取
  属性，`().__class__.__bases__` 那条经典逃逸路径就成立；禁掉之后，值域被限死
  在 str/int/bool/None/list/dict 这几种没有可达危险方法的对象上。
- **可调用的东西只有两类**：当前注册表里的工具（由调用方注入的 `call_tool`
  决定，因此工具白名单自动生效），以及下面 `_BUILTINS` 里那几个纯函数。
- **有硬上限**：源码长度、解释器运算步数、循环条数、字符串长度。防的不是
  攻击，是模型写出一个死循环把进程挂住。

## 回给模型多少

计划里写了 `print(` 时，内层调用的结果**不进上下文**，只留一行调用清单 +
`print` 出来的内容；没写则全文回显。照抄 Anthropic programmatic tool calling
的语义（工具结果留在执行环境里，只有代码输出进上下文），做成条件触发是因为
实测模型自发写的计划里只有不到一半带 `print`。另有一条聚合上限
`MAX_TRANSCRIPT_TOKENS`——防的是上下文被撑爆，和上面四条硬上限性质不同。
过滤模式下两块内容必须**分开**而不是交织，理由见 `PlanResult.transcript_full`。

## 不做什么

不支持函数定义、类、导入、异常、while、赋值给下标、lambda、推导式以外的
嵌套作用域。这不是「还没做」，是**刻意的**：每多一种语法，安全论证就多一条
要维护的边；而上面那些对「把几个工具调用串起来」毫无必要。

**`try/except` 是被否掉的，不是漏掉的，而且理由和别的不一样。** 它看起来是
「内层调用失败时别让整段崩掉」的解法，但内层调用**根本不抛异常**——
`run_tool()` 任何失败都返回一个以 `error:` 开头的字符串，计划照常往下跑。
所以 `try` 在这里唯一能捕到的是 `PlanError`，也就是运算步数、循环条数、
字符串长度这四条硬上限自己抛的那个异常。放开它等于给模型一个吞掉沙箱守卫的
语法。要判断一次调用成没成功，正确写法是查返回字符串的前缀，这一点由工具
描述明说。
"""

import ast

# 源码长度上限。超过多半意味着模型在写程序而不是在编排调用。
MAX_PLAN_CHARS = 4000
# 解释器自己的运算步数上限（每求值一个节点算一步）。防死循环，不防攻击。
MAX_PLAN_OPS = 20000
# 单次循环最多迭代多少项；也是 range() 能产出的最大长度。
MAX_LOOP_ITEMS = 200
# 任何一个字符串值的长度上限。挡住 "x" * n 这类把内存撑爆的写法。
MAX_STRING_CHARS = 200000
# 一段计划**回给模型**的转录长度上限。上面四条防的是进程被挂住，这条防的是
# 上下文被撑爆：一个 for 循环读 20 个文件，转录会原样进入下一轮 prompt，而在
# 这条上限出现之前它完全没有聚合裁剪（单个工具输出各自受 workspace.clip 的
# 单次额度限制，20 个加起来就是二十倍）。取三个普通工具调用的额度：计划的价值是
# 省模型往返，不是省下 N 倍的上下文预算；要回更多就该用 print() 在计划里先过滤。
#
# **单位是 token**，和上下文预算、工具输出上限同一种。转录直接进下一轮 prompt，
# 拿字符当上限就等于在同一条链路上摆两把尺子。数值由原先的 12000 字符按**转录里
# 实际装的内容**实测出的比值折算：转录就是若干个工具输出拼起来，和 `MAX_TOOL_OUTPUT`
# 同一种内容，实测 3.02 字符/token，12000 ÷ 3.02 ≈ 4000。统一折半会给 6000，等于
# 凭空把这条聚合上限放宽 1.5 倍。
# **这是下限。** 真正生效的是 `transcript_limit()` 从 `total_budget` 派生出来的值，
# 由 `runtime.execute_plan()` 写进 `PlanResult.transcript_limit`。
MAX_TRANSCRIPT_TOKENS = 4000
# 转录能装几个「一个工具结果那么大」的东西。原设计就是这个意思——12000 字符
# ≈ 三个普通工具调用的额度；工具结果上限跟着预算走之后，这个倍数关系要一起走，
# 否则聚合上限会小于单个元素的上限。
TRANSCRIPT_TOOL_RESULT_MULTIPLE = 3


def transcript_limit(tool_output_limit):
    """一段计划的转录最多回给模型多少 token。"""
    return max(MAX_TRANSCRIPT_TOKENS, TRANSCRIPT_TOOL_RESULT_MULTIPLE * int(tool_output_limit))


def _clip_transcript(text, limit=MAX_TRANSCRIPT_TOKENS, model=None):
    """转录超预算时保留首尾、省略中间，并**明确告知模型**省了多少、该怎么办。

    为什么保留首尾而不是只砍尾巴：转录的开头是最早几个调用（模型据此知道计划
    真的跑起来了），结尾是 print 出来的结论和失败信息（模型下一步要用的东西）。
    只砍一头都会切掉其中一类。

    省略标记里带一句 `use print()`：模型撞上截断的那一刻，正是它最可能学会
    「自己过滤」的时刻——这条提示比工具描述里的同一句话更容易被读进去。
    """
    from .models import count_tokens, head_tail_clip_tokens

    text = str(text)
    if limit <= 0:
        return ""
    total = count_tokens(text, model)
    if total <= limit:
        return text
    dropped = total - limit
    marker = (
        f"\n...[{dropped} tokens of plan output omitted. "
        f"The limit is {limit}; use print() to return only what you need]...\n"
    )
    marker_tokens = count_tokens(marker, model)
    if limit <= marker_tokens:
        return head_tail_clip_tokens(text, limit, model)
    body = head_tail_clip_tokens(text, limit - marker_tokens, model)
    # head_tail_clip_tokens 自带一个中性的省略标记，这里换成带 print() 建议的那条：
    # 模型撞上截断的那一刻，正是它最可能学会自己过滤的时刻。
    return body.replace("\n...[omitted middle]\n", marker, 1)


class PlanError(Exception):
    """计划本身不合法（语法、越界语法节点、越界上限）。

    与「某个工具调用失败」严格区分：后者是正常结果，会被写进 transcript 让
    模型下一轮消费；PlanError 则意味着这段计划根本没法执行。
    """


def _fail(node, message):
    line = getattr(node, "lineno", None)
    where = f"line {line}: " if line else ""
    raise PlanError(f"{where}{message}")


def _builtin_lines(text):
    """把一段文本切成行的列表。没有属性访问，所以 splitlines 得以函数形式提供。"""
    return str(text).splitlines()


def _builtin_sum(values, start=0):
    """数值求和。

    只收数字：`sum` 撞上字符串列表时 Python 抛的是 `TypeError: unsupported
    operand type(s)`，那条信息对着一段计划源码读不出该改哪里。这里换成一句
    说得出「第几个元素是什么类型」的话——计划的报错是回给模型看的，模型下一轮
    能不能改对，取决于这句话说清楚了没有。
    """
    if not isinstance(values, (list, tuple)):
        raise PlanError(f"sum() needs a list of numbers, got {type(values).__name__}")
    total = start
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PlanError(
                f"sum() needs numbers; item {index} is {type(value).__name__}. "
                "To count matching rows write sum([1 for row in rows if ...])."
            )
        total = total + value
    return total


def _extreme(name, chooser, values):
    """max/min 共用的实现。

    报错跟着 `sum` 的口径走:说得出「传进来的是什么」,因为这句话是回给模型看的,
    模型下一轮能不能改对取决于它说清楚了没有。不支持 `key=`——那需要 lambda,
    而 lambda 不在白名单里;要按别的字段取极值就先用列表推导把那个字段抽出来。
    """
    if not isinstance(values, (list, tuple)):
        raise PlanError(f"{name}() needs a list, got {type(values).__name__}")
    if not values:
        raise PlanError(f"{name}() got an empty list")
    return chooser(values)


def _builtin_max(values):
    return _extreme("max", max, values)


def _builtin_min(values):
    return _extreme("min", min, values)


def _builtin_range(*args):
    values = list(range(*[int(value) for value in args]))
    if len(values) > MAX_LOOP_ITEMS:
        raise PlanError(f"range() produced {len(values)} items, the limit is {MAX_LOOP_ITEMS}")
    return values


# 允许调用的纯函数。全部无副作用、不碰文件系统、不碰网络。
#
# 为什么有 split/strip/replace 这三个看起来很基础的东西：**因为禁掉了属性访问**，
# 而工具的输出是给人读的文本，不是结构化数据。`list_files` 回的是 `[F] a.py`
# 这种形状，模型要拿到 `a.py` 就必须能做字符串处理，而 `row.replace(...)` 这条
# 路已经被沙箱堵死了。不补这几个函数，"列目录再逐个读"——也就是这个工具唯一
# 真正值钱的用法——根本写不出来。
_BUILTINS = {
    "len": lambda value: len(value),
    "sorted": lambda value: sorted(value),
    "str": lambda value: str(value),
    "int": lambda value: int(value),
    "list": lambda value: list(value),
    "lines": _builtin_lines,
    "range": _builtin_range,
    # 计数是 fan-out 计划最常见的收尾动作（"有几个文件带 TODO"），而没有 sum
    # 就只能写成 `n = 0` 加一层循环 `n += 1`。实测这是模型第二常见的打回原因。
    "sum": _builtin_sum,
    "split": lambda text, sep=None: str(text).split(sep),
    "strip": lambda text: str(text).strip(),
    "replace": lambda text, old, new: str(text).replace(str(old), str(new)),
    # join 和 count 是同一条理由的延续，而且是 live 数据点名的：模型写出的
    # `join(...)` 方法形式和 `text.count("TODO")` 各被打回过——两者都是属性
    # 访问，沙箱必拒，可它们又是汇总 fan-out 结果时最自然的两个动作。
    # 参数顺序跟着 replace 走（被操作的东西在前），join 例外是因为反过来读不通。
    "join": lambda sep, items: str(sep).join(str(item) for item in items),
    "count": lambda text, sub: str(text).count(str(sub)),
    # startswith 是补 `error:` 前缀约定的那一半：工具描述让模型「查返回值的前缀」，
    # 而在它出现之前沙箱里根本没有查前缀的手段——实测模型于是写了
    # `text.startswith("error")`，被属性访问挡掉，白烧一个约 17 秒的往返。
    "startswith": lambda text, prefix: str(text).startswith(str(prefix)),
    "lower": lambda text: str(text).lower(),
    "upper": lambda text: str(text).upper(),
    "repr": lambda value: repr(value),
    "max": _builtin_max,
    "min": _builtin_min,
}

# 允许出现在语法树里的全部节点类型。**白名单而不是黑名单**：新版本 Python 加了
# 什么新语法，默认都是拒绝，不需要我们追着补。`check_plan()` 拿它做静态检查，
# 因此 `while True:` 这类东西在**一个工具都还没执行**的时候就被打回——半截执行
# 留下的工作区状态是最难解释的一种工件。解释器里那些 `_fail` 分支是第二道防线。
_ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Pass,
    ast.Load, ast.Store,
    ast.Name, ast.Constant, ast.JoinedStr, ast.FormattedValue,
    ast.List, ast.Tuple, ast.Dict, ast.Subscript, ast.Slice,
    ast.Compare, ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not, ast.USub,
    ast.BinOp, ast.Add, ast.Sub,
    ast.ListComp, ast.GeneratorExp, ast.comprehension,
    ast.Call, ast.keyword,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
)

_COMPARE_OPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


class _Printer:
    """`print()` 的实现：把内容追加进转录。

    为什么必须有它：探针实验里模型自发写出的 9 段计划有 4 段带 `print(...)`——
    它在写 Python，`print` 是反射性的。而 `print` 不在白名单里时，整段计划会在
    校验阶段被打回、一个工具都不执行，那一次模型往返（约 17 秒）就白烧了。
    实现成"写进转录"而不是真的打到 stdout：agent 的输出通道是回给模型的字符串，
    不是进程的标准输出。
    """

    __slots__ = ("result",)

    def __init__(self, result):
        self.result = result

    def __call__(self, *values):
        line = " ".join(str(value) for value in values)
        # 过滤模式下 print 的内容单独成块，不和调用清单交织——理由见
        # `PlanResult.transcript_full`。
        if self.result.echo_results:
            self.result.transcript_lines.append(line)
        else:
            self.result.printed_lines.append(line)
        return None


class PlanResult:
    """一次计划执行的全部可观测事实。

    `calls` 是逐个调用的记录（工具名、参数、结果），调用方拿它去写 trace 和
    算步数；`transcript` 是回给模型的文本。两者分开是刻意的：模型看到的是
    人读的转录，判分器看到的是结构化事实，不该互相将就。

    `result_bytes` 与 `transcript` 的长度之差就是这次编排**省下的上下文**。
    没有这两个数，「结果不进上下文」这件事在工件上完全不可验收——它和
    「什么都没省」长得一模一样。
    """

    __slots__ = (
        "calls",
        "transcript_lines",
        "printed_lines",
        "call_lines",
        "stopped_reason",
        "ops",
        "result_bytes",
        "echo_results",
        "transcript_limit",
    )

    def __init__(self):
        self.calls = []
        # 回显模式下用它：结果必须紧挨着自己那次调用，交织是对的。
        self.transcript_lines = []
        # 过滤模式下分成两块：print 出来的内容，和调用清单。
        self.printed_lines = []
        self.call_lines = []
        self.stopped_reason = ""
        self.ops = 0
        # 内层调用返回的原始字节总数（未经任何裁剪）。
        self.result_bytes = 0
        # 是否把每个调用的结果全文回显给模型。计划里写了 print() 就关掉——
        # 见 `_Interpreter.run_call` 的说明。
        self.echo_results = True
        # 转录的聚合上限。默认是 `MAX_TRANSCRIPT_TOKENS`，真实运行里由
        # `runtime.execute_plan()` 换成从 `total_budget` 派生的值——否则单个工具
        # 结果的上限（1M 档 14,791）会超过整段转录的上限（4,000），同一次
        # `read_file` 放进计划里反而看得更少，`run_plan` 变成纯负收益。
        self.transcript_limit = MAX_TRANSCRIPT_TOKENS

    @property
    def transcript(self):
        """回给模型的文本，**已按 `transcript_limit` 裁剪**。"""
        return _clip_transcript(self.transcript_full, self.transcript_limit)

    @property
    def transcript_full(self):
        """裁剪之前的转录。只给度量用，不要回给模型。

        **过滤模式下 print 的内容和调用清单必须分成两块，不能交织。** 这条是
        live 验证抓到的：交织版本里，一次读 12 个文件的计划只有 4 个文件命中并
        print 出来，于是清单最后三行后面空空如也——模型自己写下 "The plan output
        got truncated. Let me rerun it to ensure I capture all files"，然后把整个
        fan-out 又跑了一遍。8 个样本里 5 个这样耗光步数预算。交织之所以骗人：
        `print` 发生在循环体内，输出行落在**下一个**调用的清单行之前，看起来像
        是结果和调用错位、末尾被截断。分块之后 print 的输出自成一段完整清单，
        没有"缺了几行"的错觉。
        """
        if self.echo_results:
            return "\n".join(self.transcript_lines)
        printed = "\n".join(self.printed_lines) if self.printed_lines else "(the plan printed nothing)"
        blocks = ["Printed output:", printed]
        if self.call_lines:
            blocks.append("")
            blocks.append("Calls made (results not echoed):")
            blocks.extend(self.call_lines)
        return "\n".join(blocks)


class _BudgetExhausted(Exception):
    """内部信号：步数预算用完，停在当前语句。不是错误，正常收尾。"""


class _Interpreter:
    def __init__(self, call_tool, tool_names, budget, result):
        self.call_tool = call_tool
        self.tool_names = set(tool_names)
        self.budget = int(budget)
        self.result = result
        self.env = {}
        # print 要能写进这次执行的转录，所以它是按实例绑定的，不在模块级
        # `_BUILTINS` 里。查找顺序上两者等价（见 eval_call）。
        self.builtins = dict(_BUILTINS)
        self.builtins["print"] = _Printer(result)

    # --- 记账 -------------------------------------------------------------

    def tick(self, node):
        self.result.ops += 1
        if self.result.ops > MAX_PLAN_OPS:
            _fail(node, f"plan exceeded {MAX_PLAN_OPS} interpreter steps; it is probably looping")

    def guard_value(self, node, value):
        if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
            _fail(node, f"a string value grew past {MAX_STRING_CHARS} characters")
        return value

    # --- 语句 -------------------------------------------------------------

    def exec_body(self, body):
        for node in body:
            self.exec_stmt(node)

    def exec_stmt(self, node):
        self.tick(node)
        if isinstance(node, ast.Expr):
            self.eval(node.value)
            return
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                _fail(node, "assignment must have exactly one plain variable on the left, like x = ...")
            self.env[node.targets[0].id] = self.eval(node.value)
            return
        if isinstance(node, ast.AugAssign):
            # `x += y` 就是 `x = x + y`，值域和守卫完全一致——补它纯粹是因为
            # 「计数」这个动作没有它写不顺，而实测这是模型最常见的一种打回。
            # 和 BinOp 一样只留 + 和 -：`*=` 能在一步之内把内存撑爆。
            if not isinstance(node.target, ast.Name):
                _fail(node, "augmented assignment needs a plain variable on the left, like n += 1")
            name = node.target.id
            if name not in self.env:
                _fail(node, f"'{name}' is not defined; assign it before using +=")
            if isinstance(node.op, ast.Add):
                self.env[name] = self.guard_value(node, self.env[name] + self.eval(node.value))
                return
            if isinstance(node.op, ast.Sub):
                self.env[name] = self.guard_value(node, self.env[name] - self.eval(node.value))
                return
            _fail(node, f"{type(node.op).__name__} is not allowed; only += and -= are supported")
        if isinstance(node, ast.For):
            if not isinstance(node.target, ast.Name):
                _fail(node, "the loop variable must be a plain name, like: for path in paths:")
            if node.orelse:
                _fail(node, "for/else is not supported")
            items = self.eval(node.iter)
            if not isinstance(items, (list, tuple, str)):
                _fail(node, f"cannot loop over {type(items).__name__}; loop over a list")
            if len(items) > MAX_LOOP_ITEMS:
                _fail(node, f"loop over {len(items)} items exceeds the limit of {MAX_LOOP_ITEMS}")
            for item in items:
                self.env[node.target.id] = item
                self.exec_body(node.body)
            return
        if isinstance(node, ast.If):
            if self.eval(node.test):
                self.exec_body(node.body)
            else:
                self.exec_body(node.orelse)
            return
        if isinstance(node, ast.Pass):
            return
        _fail(node, f"{type(node).__name__} is not allowed in a plan")

    # --- 表达式 -----------------------------------------------------------

    def eval(self, node):
        self.tick(node)
        if isinstance(node, ast.Constant):
            return self.guard_value(node, node.value)
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            _fail(node, f"'{node.id}' is not defined; assign it before using it")
        if isinstance(node, ast.JoinedStr):
            parts = []
            for piece in node.values:
                if isinstance(piece, ast.FormattedValue):
                    parts.append(str(self.eval(piece.value)))
                else:
                    parts.append(str(self.eval(piece)))
            return self.guard_value(node, "".join(parts))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.eval(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {self.eval(key): self.eval(value) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.Subscript):
            container = self.eval(node.value)
            if isinstance(node.slice, ast.Slice):
                lower = self.eval(node.slice.lower) if node.slice.lower else None
                upper = self.eval(node.slice.upper) if node.slice.upper else None
                if node.slice.step is not None:
                    _fail(node, "slice steps are not supported")
                return container[lower:upper]
            key = self.eval(node.slice)
            try:
                return container[key]
            except (KeyError, IndexError, TypeError) as exc:
                _fail(node, f"cannot index that value: {exc}")
        if isinstance(node, ast.Compare):
            left = self.eval(node.left)
            for operator, comparator in zip(node.ops, node.comparators):
                handler = _COMPARE_OPS.get(type(operator))
                if handler is None:
                    _fail(node, f"comparison {type(operator).__name__} is not allowed")
                right = self.eval(comparator)
                if not handler(left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                value = True
                for item in node.values:
                    value = self.eval(item)
                    if not value:
                        return value
                return value
            value = False
            for item in node.values:
                value = self.eval(item)
                if value:
                    return value
            return value
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return not self.eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -self.eval(node.operand)
            _fail(node, f"unary {type(node.op).__name__} is not allowed")
        if isinstance(node, ast.BinOp):
            # 只留 + 和 -。乘法被排除是因为 "x" * n 和 [x] * n 能在一步之内
            # 把内存撑爆，而编排工具调用完全用不上它。
            if isinstance(node.op, ast.Add):
                return self.guard_value(node, self.eval(node.left) + self.eval(node.right))
            if isinstance(node.op, ast.Sub):
                return self.eval(node.left) - self.eval(node.right)
            _fail(node, f"operator {type(node.op).__name__} is not allowed; only + and - are")
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            # 生成器表达式当列表推导算。语义差别只有惰性,而这里的值域和
            # MAX_LOOP_ITEMS 守卫让惰性没有任何意义;不认它的唯一后果是
            # `sum(1 for line in lines(text) if "TODO" in line)` 这个**最常见的
            # 计数写法**被打回——实测 11 次被拒的计划里有 3 次栽在这上面,
            # 而且那几段计划的其余部分完全合法。
            return self.eval_listcomp(node)
        if isinstance(node, ast.Call):
            return self.eval_call(node)
        if isinstance(node, ast.Attribute):
            # 单独报这一条而不是落到兜底，是因为模型最容易踩的就是它
            # （`text.splitlines()`），而它同时也是整个沙箱最关键的那条边。
            _fail(node, "attribute access is not allowed; use lines(text), len(x) or sorted(x) instead")
        _fail(node, f"{type(node).__name__} is not allowed in a plan")

    def eval_listcomp(self, node):
        if len(node.generators) != 1:
            _fail(node, "a comprehension may have only one 'for'")
        generator = node.generators[0]
        if generator.is_async or not isinstance(generator.target, ast.Name):
            _fail(node, "the comprehension variable must be a plain name")
        items = self.eval(generator.iter)
        if not isinstance(items, (list, tuple, str)):
            _fail(node, f"cannot loop over {type(items).__name__}")
        if len(items) > MAX_LOOP_ITEMS:
            _fail(node, f"comprehension over {len(items)} items exceeds the limit of {MAX_LOOP_ITEMS}")
        saved = self.env.get(generator.target.id)
        output = []
        for item in items:
            self.env[generator.target.id] = item
            if all(self.eval(condition) for condition in generator.ifs):
                output.append(self.eval(node.elt))
        if saved is None:
            self.env.pop(generator.target.id, None)
        else:
            self.env[generator.target.id] = saved
        return output

    def eval_call(self, node):
        if not isinstance(node.func, ast.Name):
            _fail(node, "only plain function names can be called")
        name = node.func.id
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            _fail(node, "*args is not supported")
        if any(keyword.arg is None for keyword in node.keywords):
            _fail(node, "**kwargs is not supported")
        if name in self.builtins:
            if node.keywords:
                _fail(node, f"{name}() takes positional arguments only")
            try:
                return self.guard_value(node, self.builtins[name](*[self.eval(arg) for arg in node.args]))
            except PlanError:
                raise
            except Exception as exc:
                _fail(node, f"{name}() failed: {exc}")
        if name not in self.tool_names:
            available = ", ".join(sorted(self.tool_names | set(self.builtins)))
            _fail(node, f"'{name}' is not a tool you can call. Available: {available}")
        if node.args:
            _fail(node, f"{name}() takes keyword arguments only, like {name}(path=\"src/app.py\")")
        args = {keyword.arg: self.eval(keyword.value) for keyword in node.keywords}
        return self.run_call(node, name, args)

    def run_call(self, node, name, args):
        # 第一个调用由 run_plan 这一步本身覆盖；从第二个起，每个调用各吃一步。
        # 语义和「一轮发多个工具调用」完全一致：预算限制的是做了多少事，
        # 不是往返了几次，否则一段计划就能绕过 max_steps。
        if self.result.calls and self.budget <= 0:
            self.result.stopped_reason = "step_budget_exhausted"
            raise _BudgetExhausted()
        if self.result.calls:
            self.budget -= 1
        output = self.call_tool(name, args)
        self.result.calls.append({"name": name, "args": args, "result": output})
        self.result.result_bytes += len(str(output))
        index = len(self.result.calls)
        rendered = ", ".join(f"{key}={value!r}" for key, value in args.items())
        # 调用清单行**在两种模式下都写**。这是一条不变量：内层调用不能在回给
        # 模型的文本里静默消失，否则模型会以为那次 patch_file 没发生过，可能
        # 把一个已经落盘的写操作再发一遍。
        header = f"[{index}] {name}({rendered})"
        if self.result.echo_results:
            self.result.transcript_lines.append(header)
            self.result.transcript_lines.append(str(output))
        else:
            # 计划里写了 print()：模型已经表明要自己挑什么回来，就不再把每个
            # 结果全文倒进上下文。这是 Anthropic programmatic tool calling 的
            # 核心语义（工具结果留在执行环境里，只有代码的输出进上下文）。
            # 之所以做成「有 print 才关」而不是无条件关：探针里模型自发写的
            # 9 段计划只有 4 段带 print，无条件关会让另外 5 段拿回一句空转录、
            # 白烧一个约 17 秒的模型往返。字节数照报，模型据此知道结果有多大。
            self.result.call_lines.append(f"{header} -> {len(str(output))} chars")
        return self.guard_value(node, str(output))


def check_plan(source, tool_names):
    """静态检查：能不能解析、语法节点是否全部合法、调用的名字是否都存在。

    **不执行任何工具**，所以可以放在参数校验阶段（`validate_tool`）里跑。
    价值在于：一段引用了不存在的工具、或者用了属性访问的计划，会在动手之前
    就被打回，而不是执行到一半留下半截改动——那种半截状态在工件上最难解释。
    """
    if not isinstance(source, str) or not source.strip():
        raise PlanError("plan must be a non-empty string")
    if len(source) > MAX_PLAN_CHARS:
        raise PlanError(f"plan is {len(source)} characters, the limit is {MAX_PLAN_CHARS}")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise PlanError(f"line {exc.lineno}: {exc.msg}") from exc
    known = set(tool_names) | set(_BUILTINS) | {"print"}
    calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            _fail(node, "attribute access is not allowed; use lines(text), len(x) or sorted(x) instead")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # 单独报这一条，因为实测它是最常见的一种打回：k=3 跑批里 7 次计划有
            # 3 次栽在 `import os` / `os.walk` / `open()` 上。模型在写 Python，
            # 默认脚下有个解释器和一套标准库。一句 "Import is not allowed" 不足以
            # 纠正这个先验——它下一轮会换成 `from os import path` 再试一次。
            _fail(
                node,
                "imports are not available. A plan is not a Python program: there is no standard "
                "library, no os, no open(), and no filesystem access except through the tools "
                "listed above. To look at files, call the tools.",
            )
        if not isinstance(node, _ALLOWED_NODES):
            _fail(node, f"{type(node).__name__} is not allowed in a plan")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                _fail(node, "only plain function names can be called")
            if node.func.id not in known:
                _fail(node, f"'{node.func.id}' is not a tool you can call. Available: {', '.join(sorted(known))}")
            if node.func.id in tool_names:
                calls += 1
    if not calls:
        raise PlanError("a plan must call at least one tool; if you only need one call, call the tool directly")
    return tree


def plan_prints(tree):
    """这段计划里有没有调用 `print`。

    决定了执行时要不要把每个工具结果全文回显给模型（见 `_Interpreter.run_call`）。
    做成**静态判断**而不是运行时"有没有 print 过"：后者会让转录形态取决于分支
    有没有走到，同一段计划两次执行可能给出不同形状的输出，模型无从建立预期。
    """
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
        for node in ast.walk(tree)
    )


def execute_plan(source, call_tool, tool_names, budget, result=None):
    """跑一段计划。返回 `PlanResult`。

    `call_tool(name, args) -> str` 由调用方注入——真实运行里就是
    `CodingForMe.run_tool`，所以每一个内层调用照样走完整闸口（存在性、参数
    校验、重复检测、审批、只读、脱敏、记忆回写），工具白名单也自动生效。
    这个模块**不认识 agent**，也拿不到文件系统。

    调用方可以传入一个自己持有的 `result`：计划跑到一半抛 `PlanError` 时，
    已经执行过的调用仍然记在里面。丢掉它们等于让模型以为那些都没发生过，
    而其中可能包含已经落盘的写操作。

    `budget` 是「除第一个调用之外还能再跑几个」。用完就停在当前语句，
    `stopped_reason` 记为 `step_budget_exhausted`，已经跑完的调用照常保留——
    静默丢弃会让模型以为整段都没执行。
    """
    result = result if result is not None else PlanResult()
    tree = check_plan(source, tool_names)
    # 计划自己写了 print()，就由它决定什么进上下文；没写则维持全文回显。
    result.echo_results = not plan_prints(tree)
    interpreter = _Interpreter(call_tool, tool_names, budget, result)
    try:
        interpreter.exec_body(tree.body)
    except _BudgetExhausted:
        pass
    return result
