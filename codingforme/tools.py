"""工具定义与执行辅助逻辑。

可以把这个文件看成 agent 的能力白名单：模型能申请哪些动作、这些动作
如何做参数校验，以及最终如何执行，都是在这里定义的。
"""

import hashlib
import json
import shutil
import subprocess
import textwrap
from functools import partial
from pathlib import Path

from . import plans
from .workspace import IGNORED_PATH_NAMES, clip

# 路径类参数共用的一句话说明。
#
# 为什么值得单独提出来：k=3 真实跑批里 50 次被拒调用中，26% 是 `../` 逃出工作区、
# 10% 是绝对路径，合计 36%——是最大的一类。而此前模型能看到的关于路径的全部信息，
# 只有 workspace 快照里那行 `repo_root: <绝对路径>`：它告诉模型仓库在哪，却从没说过
# **工具参数该用哪种写法**。看到一个绝对路径就照抄一个绝对路径，是完全合理的推断。
PATH_ARGUMENT_NOTE = (
    "Path relative to the repo root, like 'src/app.py'. "
    "Absolute paths and paths containing '..' are rejected."
)

# 工具定义。参数值有两种写法，都合法：
#
#   "int=1"                                   —— 只有类型和默认值的短写法
#   {"type": "int=1", "description": ..., "minimum": 1}   —— 带说明和约束
#
# 短写法保留是因为它够用（测试里大量使用），也因为把两种写法收敛成一种会让
# 「这个参数需要解释」这件事失去标记。`"type"` 里的 DSL 两种写法完全一致：
# 有 `=` 表示有默认值、因此非必填；没有 `=` 表示必填。
#
# description 写给谁：写给模型，不是写给读代码的人。判断标准是「模型不知道这件事
# 时会犯什么错」——所以 `old_text` 的说明里写「逐字节一致、包含缩进」，因为实测
# 14% 的被拒调用是它没做到；而 `pattern` 就只有一句话，因为没有证据显示它出过问题。
BASE_TOOL_SPECS = {
    "list_files": {
        "schema": {
            "format": {
                "type": "str='tree'",
                "description": (
                    "'tree' (default) returns one line per entry as '[F] name' or '[D] name'. "
                    "'paths' returns one bare file path per line with no prefix and no "
                    "directories, which is what you want when code will consume the result."
                ),
            },
            "path": {
                "type": "str='.'",
                "description": "Directory to list, e.g. '.' or 'src'. " + PATH_ARGUMENT_NOTE,
            },
        },
        "risky": False,
        "description": "List the files and directories directly inside one workspace directory.",
    },
    "read_file": {
        "schema": {
            "path": {"type": "str", "description": "File to read. " + PATH_ARGUMENT_NOTE},
            "start": {
                "type": "int=1",
                "description": "First line to read, counting from 1.",
                "minimum": 1,
            },
            "end": {
                "type": "int=200",
                "description": (
                    "Last line to read, inclusive. Raise it if the file is longer than the "
                    "excerpt you get back."
                ),
                "minimum": 1,
            },
        },
        "risky": False,
        "description": "Read a UTF-8 text file over a line range. The result is line-numbered.",
    },
    "search": {
        "schema": {
            "format": {
                "type": "str='lines'",
                "description": (
                    "'lines' (default) returns 'path:line:text' for every match. 'paths' returns "
                    "each matching file once, one bare path per line -- far smaller when you only "
                    "need to know which files match."
                ),
            },
            "pattern": {"type": "str", "description": "Text or regular expression to look for."},
            "path": {
                "type": "str='.'",
                "description": "File or directory to search under. " + PATH_ARGUMENT_NOTE,
            },
        },
        "risky": False,
        "description": "Search the workspace for a pattern and return matching lines with line numbers.",
    },
    "run_shell": {
        "schema": {
            "command": {
                "type": "str",
                "description": "Command to run, e.g. 'python -m pytest -q'. It runs from the repo root.",
            },
            "timeout": {
                "type": "int=20",
                "description": "Seconds to wait before the command is killed.",
                "minimum": 1,
                "maximum": 120,
            },
        },
        "risky": True,
        "description": "Run one shell command in the repo root and return its exit code, stdout and stderr.",
    },
    "write_file": {
        "schema": {
            "path": {
                "type": "str",
                "description": "File to write; missing parent directories are created. " + PATH_ARGUMENT_NOTE,
            },
            "content": {
                "type": "str",
                "description": (
                    "The complete new content of the file. It replaces the file entirely, so "
                    "include every line you want to keep."
                ),
            },
        },
        "risky": True,
        "description": "Create a file or replace it whole. To change part of an existing file, use patch_file.",
    },
    "patch_file": {
        "schema": {
            "path": {"type": "str", "description": "File to modify. " + PATH_ARGUMENT_NOTE},
            "old_text": {
                "type": "str",
                "description": (
                    "The text to replace, copied from the file byte for byte, including "
                    "indentation and line breaks. It must occur exactly once, so include "
                    "enough surrounding lines to make it unique. Read the file first."
                ),
            },
            "new_text": {
                "type": "str",
                "description": "The replacement text, indented to match the surrounding code.",
            },
        },
        "risky": True,
        "description": "Replace one exact block of text in an existing file.",
    },
}

DELEGATE_TOOL_SPEC = {
    "schema": {
        "task": {"type": "str", "description": "What the child agent should find out, in one sentence."},
        "max_steps": {
            "type": "int=3",
            "description": "How many tool calls the child agent may make.",
            "minimum": 1,
            "maximum": 8,
        },
    },
    "risky": False,
    "description": "Ask a bounded read-only child agent to investigate something and report back.",
}

# 受限编排工具。**默认不注册**：由 feature flag `plan_tool` 打开（见
# `build_tool_registry`），这样它是一个可以单独消融的变体维度，而不是一个
# 悄悄改变所有跑批的全局改动。语法与安全边界见 `plans.py` 的模块 docstring。
#
# `aggregates_calls` 这个键只有它有：它自己不需要审批（`risky: False`），
# 因为审批与只读判定发生在**每一个内层调用**上——粒度不该因为套了一层编排
# 就变粗。但它可能通过内层调用改到工作区，所以 `run_tool()` 仍要在它前后拍
# 工作区快照，否则 trace 里那条记录会说「没改动过任何文件」，而那是假的。
PLAN_TOOL_SPEC = {
    "schema": {
        "plan": {
            "type": "str",
            # 这段文字**每一轮都要重发一遍**：实测 `run_plan` 的存在让每轮固定
            # 多花 642 个输入 token，而它的 JSON spec 一个人就顶其余 6 个工具的
            # 一半。所以这里的取舍标准不是「挑短的留」，而是**每一句都得说出
            # 一件模型猜不到、且实测猜错过的事**：
            #
            #   - 「不是真 Python」——最常见的一类打回（import os 之类）；
            #   - 禁属性访问那两个改写示范——打回里最具体的一条，删了模型只会
            #     再写一次 `text.splitlines()`；
            #   - 允许的语法与函数清单——无从推导，只能明说；
            #   - `print` 的过滤语义——整个功能的核心，且它是隐式规则；
            #   - 内层调用照样过闸口、各计一步——安全相关；
            #   - 失败返回 `error:` 前缀——顶替掉「为什么没有 try」那一整段。
            #
            # 删掉的是和**工具**描述重复的那句「参数依赖前一次结果时用它」，
            # 以及把同一件事换个说法再讲一遍的从句。
            "description": (
                "A tiny Python-like program that calls other tools, so a later call can use an "
                "earlier call's result without another round trip. It is not real Python: no "
                "imports, no standard library, no open(), no while/def/try, and no attribute "
                "access -- every string operation is a plain function, so write lines(text) not "
                "text.splitlines(), replace(row, a, b) not row.replace(a, b), startswith(row, x) "
                "not row.startswith(x). Tools take keyword arguments and return strings; a call "
                "that fails returns a string starting with 'error:' rather than raising, so test "
                "it with startswith(result, 'error:'). Allowed: assignment, += and -=, for, if, "
                "f-strings, list/dict literals, indexing, comparisons, +, -, comprehensions, and "
                "len, sorted, sum, max, min, str, int, list, repr, lines, range, split, strip, "
                "replace, join, count, startswith, lower, upper, print. Every inner call is "
                "checked and approved exactly like a direct call and costs a step. If the plan "
                "calls print(), tool results are NOT sent back to you and only what you print "
                "is -- use that to filter: read a lot, print the few lines you need. Without "
                "print(), every result comes back in full."
            ),
        },
    },
    "risky": False,
    "aggregates_calls": True,
    "description": (
        "Run several dependent tool calls in one step by writing a short program. "
        "Reach for it when you must read something before you know what to do next, "
        "or to scan many files and report only a summary. For a single call, call the "
        "tool directly."
    ),
}

# 参数校验失败时回给模型的示例。**只举参数对象，不举调用形式**：调用形式由
# 标准 function-calling 接口决定，这里再示范一遍 <tool> 标签，等于在模型出错的
# 那一刻把它推向一套 prompt 里根本没教、我们也没发 schema 的协议——协议互斥的
# 问题曾经就是从这个通道漏出去的。
TOOL_EXAMPLES = {
    "list_files": '{"path": "."}',
    "read_file": '{"path": "README.md", "start": 1, "end": 80}',
    "search": '{"pattern": "binary_search", "path": ".", "format": "paths"}',
    # 示例现在会进 prefix（见 runtime.build_prefix），所以它必须是**在被测仓库里
    # 真能跑起来的命令**，不能是本机开发用的那条。原来写的是 `uv run --with pytest
    # ...`，而评测 fixture 仓库里既没有 uv 也没装 pytest——模型照抄一条注定失败的
    # 命令，就等于我们主动教它烧掉一个约 17 秒的往返。
    "run_shell": '{"command": "python -m pytest -q", "timeout": 20}',
    "write_file": '{"path": "binary_search.py", "content": "def binary_search(nums, target):\\n    return -1\\n"}',
    "patch_file": '{"path": "binary_search.py", "old_text": "return -1", "new_text": "return mid"}',
    "delegate": '{"task": "inspect README.md", "max_steps": 3}',
    # 示例刻意展示「后一个调用的参数来自前一个调用的结果」这件事——那正是
    # 这个工具唯一存在的理由，而它也是模型最不容易自己想到的用法。
}


def build_tool_registry(agent):
    # 工具不是动态发现的，而是显式注册的。
    # 这样模型看到的是一个有边界、可审计的动作集合。
    tools = {
        name: {**spec, "run": partial(_TOOL_RUNNERS[name], agent)}
        for name, spec in BASE_TOOL_SPECS.items()
    }
    # 子 agent 是刻意做成受限能力的：一旦深度耗尽，
    # 就连 delegate 这个工具都不再暴露给模型。
    if agent.depth < agent.max_depth:
        tools["delegate"] = {**DELEGATE_TOOL_SPEC, "run": partial(tool_delegate, agent)}
    # 受限编排默认关闭。开着它跑出来的数据和关着它跑出来的不可比，所以让它
    # 成为一个有名字的变体（`HarnessSpec(feature_flags={"plan_tool": True})`），
    # 而不是一个所有跑批都自动带上的改动。
    if agent.feature_enabled("plan_tool"):
        tools["run_plan"] = {**PLAN_TOOL_SPEC, "run": partial(tool_run_plan, agent)}
    return tools


# `run_plan` 的示例**不能是静态的**——它得展示怎么调别的工具，而"别的工具"是
# 什么，取决于白名单裁剪之后剩下哪些。踩过的坑（真实跑批才看得见）：静态示例写
# 的是 `list_files(...)`，而 12 个基准任务的白名单里一个都没有 `list_files`；
# 模型照抄示例，13 次计划 13 次被打回，每次白烧一个约 17 秒的往返。
#
# 三档示例都要展示**依赖**（后一个调用的参数来自前一个的结果或判断），因为那
# 才是这个工具唯一比批量调用多出来的能力；只会并排列几个调用的示例，等于在教
# 模型用它做一件批量调用已经能做的事。
#
# 三档示例还都必须展示**过滤**：把结果留在计划里，只 print 结论。这是这个工具
# 省上下文的唯一途径（模型写了 print，执行器就不再把每个结果全文回显，见
# plans.py 的 run_call）。原来的示例只调用不 print，等于在教模型放弃这一半收益。
_PLAN_EXAMPLES = (
    (
        ("list_files", "read_file"),
        # 这份示例里的 `startswith(text, "error:")` 是刻意放的,不是凑数:实测
        # 描述里用散文说「写 startswith(row, x) 而不是 row.startswith(x)」**不管用**——
        # 一次 14 任务的 live 跑批里 5 段被打回的计划有 3 段正是 `result.startswith(...)`。
        # 模型照抄的是示例,不是规则。
        '{"plan": "for path in lines(list_files(path=\\".\\", format=\\"paths\\")):\\n    if \\".py\\" in path:\\n        text = read_file(path=path, start=1, end=200)\\n        if startswith(text, \\"error:\\"):\\n            print(path, \\"unreadable\\")\\n        else:\\n            defs = [ln for ln in lines(text) if \\"def \\" in ln]\\n            print(path, len(defs), \\"functions\\")"}',
    ),
    (
        ("read_file", "patch_file"),
        '{"plan": "text = read_file(path=\\"README.md\\", start=1, end=60)\\nif \\"TODO\\" in text:\\n    patch_file(path=\\"README.md\\", old_text=\\"TODO\\", new_text=\\"done\\")\\n    print(\\"patched README.md\\")\\nelse:\\n    print(\\"no TODO in README.md\\")"}',
    ),
    (
        ("read_file",),
        '{"plan": "for path in [\\"README.md\\", \\"sample.txt\\"]:\\n    text = read_file(path=path, start=1, end=80)\\n    if \\"TODO\\" in text:\\n        print(path, \\"has TODO\\")"}',
    ),
)


def plan_example(tool_names):
    """按当前注册表挑一份 `run_plan` 的示例；没有能凑出示例的工具时返回空串。"""
    available = set(tool_names)
    for needed, text in _PLAN_EXAMPLES:
        if available.issuperset(needed):
            return text
    return ""


def tool_example(name, tool_names=None):
    """参数示例。`run_plan` 的那份按注册表现算，其余是静态的。

    `tool_names` 传当前注册表（`agent.tools` 或 `to_openai_function_specs` 拿到
    的那份 dict 都行）；不传时退回全部已知工具，供不认识 agent 的调用方使用。
    """
    if name == "run_plan":
        names = set(tool_names) if tool_names is not None else set(BASE_TOOL_SPECS)
        return plan_example(names - META_TOOLS)
    return TOOL_EXAMPLES.get(name, "")


_SCHEMA_TYPE_TO_JSON_TYPE = {"str": "string", "int": "integer"}
# 从字段规格里往 JSON Schema 直接透传的键。列成白名单而不是"除 type 外全传"，
# 是为了让「往规格里加一个键」变成一次需要动这行代码的显式决定。
_SCHEMA_PASSTHROUGH_KEYS = ("description", "minimum", "maximum", "enum")


def schema_field_type(field_value):
    """取出字段规格里的类型 DSL 串（`"int=1"` 这种）。

    短写法（裸字符串）和长写法（带 description 的 dict）在这里合流，
    其余代码就不必知道当前这个字段用的是哪种写法。
    """
    if isinstance(field_value, dict):
        return str(field_value.get("type", "str"))
    return str(field_value)


def render_schema_fields(schema):
    """把一份工具 schema 渲染成 prefix 里那行 `path: str, start: int=1`。

    放在 tools.py 而不是 runtime 里：微型 DSL 的形状属于工具层，runtime 只该
    拿到一行现成的文本。此前 runtime 直接 `f"{key}: {value}"`，值一旦变成 dict
    就会把整个 Python 字典字面量印进 prompt。
    """
    return ", ".join(f"{key}: {schema_field_type(value)}" for key, value in schema.items())


def tool_schema(name):
    """按工具名取 schema，**不需要 agent 实例**。

    `parse()` 是静态方法，拿不到 `self.tools`，但归一必须发生在那里（理由见
    `coerce_tool_args`）。工具白名单可能被 harness 裁剪过，但同名工具的 schema
    在各变体之间是同一份，所以按名字查全集是安全的。
    """
    if name == "delegate":
        return DELEGATE_TOOL_SPEC["schema"]
    spec = BASE_TOOL_SPECS.get(name)
    return spec["schema"] if spec else None


def _coerce_scalar(value, type_name):
    # 归一不是校验：**转不动就原样放行**，让 validate_tool() 去报一个说得清楚的错。
    # 在这里抛异常会把"类型不对"变成一条来自归一层的、模型看不懂的报错。
    if value is None or isinstance(value, bool):
        # None 不能转成 "None"（那会让 old_text 去文件里找字面量 None），
        # bool 不能转成 0/1（Python 里 bool 是 int 的子类，静默转换会掩盖模型的错）。
        return value
    if type_name == "int":
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else value
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return value
        return value
    if type_name == "str":
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        return value
    return value


def coerce_tool_args(name, args):
    """把模型发来的参数按 schema 归一成声明的类型。

    为什么必须在**进入闸口之前**做，而不是各个 runner 自己 `int(...)` 一下：
    局部归一只让那一次执行成功，`run_tool()` 拿到的、写进 history 与 trace 的、
    进记忆的仍是模型发来的原始值。实测后果是重复调用检测会漏：

        第 1 次  {"path": "a.txt", "start": "1"}   执行
        第 2 次  {"path": "a.txt", "start": "1"}   执行
        第 3 次  {"path": "a.txt", "start": 1}     **又执行了一次**

    第 3 次和前两次是完全相同的动作，但 `repeated_tool_call()` 比的是原始字典，
    `1 != "1"` 于是判不出重复。重复调用是被拒调用里最大的一类（实测占 37%），
    检测本身漏一档，就没法区分「模型少打转了」和「我们少查了几次」。

    未知参数原样保留：静默丢掉会让模型以为它发的东西被接受了。
    """
    schema = tool_schema(name)
    if not schema or not isinstance(args, dict):
        return args
    coerced = dict(args)
    for key, value in args.items():
        field = schema.get(key)
        if field is None:
            continue
        type_name = schema_field_type(field).partition("=")[0].strip()
        coerced[key] = _coerce_scalar(value, type_name)
    return coerced


def _schema_field_to_json_schema(field_value):
    # 微型 DSL 里 "int=20" 表示"有默认值 20，因此非必填"；
    # 没有 "=" 的字段（比如 "str"）表示必填，没有默认值。
    type_text = schema_field_type(field_value)
    type_name, _, default_text = type_text.partition("=")
    is_required = "=" not in type_text
    json_type = _SCHEMA_TYPE_TO_JSON_TYPE.get(type_name.strip(), "string")
    prop = {"type": json_type}
    if isinstance(field_value, dict):
        for key in _SCHEMA_PASSTHROUGH_KEYS:
            if key in field_value:
                prop[key] = field_value[key]
        # 默认值只在长写法里透传。短写法保持原样是为了让既有断言
        # （`properties["start"] == {"type": "integer"}`）继续成立：那些用例
        # 验证的是"必填/选填怎么判"，不该因为这次加字段而全部重写。
        if default_text:
            prop["default"] = _coerce_default(json_type, default_text.strip())
    return prop, is_required


def _coerce_default(json_type, default_text):
    if json_type == "integer":
        try:
            return int(default_text)
        except ValueError:
            return default_text
    return default_text.strip("'\"")


def to_openai_function_specs(tools):
    """把 `BASE_TOOL_SPECS` 风格的微型 DSL 翻译成标准 OpenAI function-calling 的 `tools=` 数组。

    `BASE_TOOL_SPECS`/`DELEGATE_TOOL_SPEC` 是这份 schema 的唯一来源；这里只是
    多提供一种读法，不引入第二份手写的 JSON Schema，避免两份定义互相漂移。
    """
    specs = []
    for name, tool in tools.items():
        properties = {}
        required = []
        for field_name, field_value in tool["schema"].items():
            prop, is_required = _schema_field_to_json_schema(field_value)
            properties[field_name] = prop
            if is_required:
                required.append(field_name)
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        # 多余的键直接判非法，而不是被静默忽略。k=3 基线里**一次
                        # 都没观测到**模型发多余参数，所以这不是在修一个已知问题，
                        # 只是把"schema 应当完整描述合法输入"这条补齐——顺手做，
                        # 别把它当成会带来收益的改动。
                        "additionalProperties": False,
                    },
                },
            }
        )
        # 示例走 JSON Schema 的标准 `examples` 关键字，放在 parameters 上。
        # 三件事是实测定的（16 次采样 × 3 组，工具的参数格式只能从示例得知）：
        # 这个后端把 `parameters.examples` **原样送到模型面前**（16/16 吐出示例
        # 里那个无从推导的值）；Anthropic 的 `input_examples` 字段名在这里被静默
        # 丢弃（0/16）；什么都不给时模型 12/16 直接拒绝调用。
        #
        # 为什么还要在 prefix 的工具清单里留一份：两个通道成本都近乎为零（都走
        # 前缀缓存），而它们喂给模型的解析路径不同——prefix 是自然语言，tools=
        # 是结构化 schema。去掉哪一份都属于单独的实验，不在这次改动里做。
        example_text = tool_example(name, tools)
        if example_text:
            try:
                specs[-1]["function"]["parameters"]["examples"] = [json.loads(example_text)]
            except ValueError:
                # 示例写坏了不该让整个请求发不出去；prefix 里那份仍然在。
                pass
    return specs


def base_tool_schema_signature():
    """全部工具定义的 sha256，**不需要 agent 实例**。

    和 `CodingForMe.tool_signature()` 哈希同样的东西（名字/schema/risky/描述，
    外加翻译成标准 function-calling 的 JSON Schema 形状），区别只是它从
    `BASE_TOOL_SPECS` 直接算，因此可以在没有 workspace、没有装配 agent 的地方
    调用——`HarnessSpec.fingerprint()` 就是这种地方。

    存在的理由见 `runtime.prompt_template_signature()` 的 docstring：工具 schema
    和提示词一样，是「会改变模型行为、却不属于任何配置字段」的东西，不进指纹的话
    改了 schema 的前后两份评测数据在工件层面无法区分。
    """
    tools = dict(BASE_TOOL_SPECS)
    tools["delegate"] = DELEGATE_TOOL_SPEC
    tools["run_plan"] = PLAN_TOOL_SPEC
    payload = [
        {
            "name": name,
            "schema": tools[name]["schema"],
            "risky": tools[name]["risky"],
            "description": tools[name]["description"],
        }
        for name in sorted(tools)
    ]
    payload.append({"function_specs": to_openai_function_specs(tools)})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


DELEGATE_MAX_STEPS_CEILING = 8


def available_tools(agent):
    """当前注册表里**真正存在**的工具名集合。

    `agent.tools` 是唯一真相，不是 `BASE_TOOL_SPECS`：`HarnessSpec.build()` 会按
    变体白名单 ∩ 任务白名单把注册表裁掉一部分（见 `eval/harness.py`），所以规格里
    有的工具，模型不一定调得到。
    """
    return set(getattr(agent, "tools", None) or ())


def patch_match_error(agent, count, relative_path):
    """`old_text` 没有恰好命中一次时的报错。

    命中 0 次和命中多次是**两种完全不同的错误**，修复动作正相反：0 次要模型把
    文本抄准（多半是缩进或空白没对上），多次要模型把范围放大到唯一。原来两种
    共用一句 `old_text must occur exactly once, found N`，模型只能自己反推该往
    哪个方向改。这一类占 k=3 跑批被拒调用的 14%。

    带 `agent` 是为了拿注册表：0 次那句建议模型去 `read_file`，而白名单可能
    根本没给它这个工具——见 `wrong_kind_error` 的说明。
    """
    if count == 0:
        how = (
            "Read the file with read_file and copy the lines exactly as they appear, "
            "without the line numbers."
            if "read_file" in available_tools(agent)
            else "Copy the lines exactly as they appear in the file."
        )
        return ValueError(
            f"old_text was not found in {relative_path}. It must match the file byte for byte, "
            f"including indentation and blank lines. {how}"
        )
    return ValueError(
        f"old_text occurs {count} times in {relative_path} and must occur exactly once. "
        "Include more lines above and below it so the block is unique."
    )


def _relative_to_root(agent, path):
    try:
        return str(path.relative_to(agent.root)).replace("\\", "/")
    except ValueError:  # 解析后落在 root 之外的情形已被 agent.path() 拦掉，这里只是兜底
        return str(path)


def wrong_kind_error(agent, path, expected):
    """路径存在性/类型不对时的报错，**带上下一步该做什么**。

    为什么不是一句 `path is not a file` 就够了：k=3 真实跑批里 10% 的被拒调用
    是这一类，而原来的措辞既没说清是"不存在"还是"是个目录"，也没给修复动作。
    模型拿到它只能猜，通常就是把同一个路径再试一遍——这正是 30% 的重复调用
    里的一部分来源。

    `expected` 是 "file" 或 "directory"。

    **建议里只能出现注册表里真有的工具。** 这三句话原本是硬写的，而任务声明的
    `allowed_tools` 接上之后（N-5）注册表会被裁到只剩 `read_file`/`patch_file`：
    模型于是被指去调一个它在工具清单里根本看不到的工具，下一轮只能瞎猜，或者
    把同一个路径原样再试一遍——那正好撞上 `no_repeated_calls` 这条断言。一句
    错的建议比不给建议更糟，所以一个都不可用时宁可回退到"查工作区快照"。
    """
    relative = _relative_to_root(agent, path)
    have = available_tools(agent)
    if path.is_dir():
        hint = " Use list_files to see what is inside it." if "list_files" in have else ""
        return ValueError(f"path is a directory, not a file: {relative}.{hint}")
    if path.is_file():
        hint = " Use read_file to read it." if "read_file" in have else ""
        return ValueError(f"path is a file, not a directory: {relative}.{hint}")
    parent = _relative_to_root(agent, path.parent)
    ways = []
    if "list_files" in have:
        ways.append(f"list_files on '{parent or '.'}'")
    if "search" in have:
        ways.append("search")
    how = (
        f"Use {' or '.join(ways)} to find the right path"
        if ways
        else "Check the workspace snapshot in the first message for the right path"
    )
    return ValueError(f"no such {expected}: {relative}. {how} instead of retrying this one.")


# 「元工具」：本身不提供任何新能力，只改变已有能力的**调用形式**。
#
# 为什么要单列：工具白名单声明的是「这道题该动用哪些能力」。`run_plan` 能调到的
# 东西恰好是注册表里已有的那些（见 `plan_callable_tools`），它一件新事都做不了，
# 只是让几个调用能在一次模型往返里发完。把它计进白名单的话，数据集里每一个任务
# 都要重新声明一个不给任何权限的名字，否则这个变体在整份基准上直接变成空操作——
# 而"变体开了但什么都没发生"在报告里长得和"变体没用"一模一样。
META_TOOLS = frozenset({"run_plan"})


def plan_callable_tools(agent):
    """一段计划里允许调用的工具名。

    刻意排除 `run_plan` 自己：嵌套编排没有任何表达力上的收益（内层能写的东西
    外层都能写），却会让步数记账和 trace 的轮内序号多一层需要论证的递归。
    """
    return available_tools(agent) - {"run_plan"}


def validate_tool(agent, name, args):
    args = args or {}

    if name == "run_plan":
        # 静态检查，**一个工具都不执行**：语法、越界语法节点、调用了不存在的
        # 工具，都在这里就打回。放在校验阶段而不是执行阶段是刻意的——一段
        # 写错的计划如果执行到一半才报错，工作区会停在半截状态，而那种状态
        # 在工件上最难解释、也最难让模型自己收拾。
        try:
            plans.check_plan(str(args.get("plan", "")), plan_callable_tools(agent))
        except plans.PlanError as exc:
            raise ValueError(str(exc)) from exc
        return

    if name == "list_files":
        path = agent.path(args.get("path", "."))
        if not path.is_dir():
            raise wrong_kind_error(agent, path, "directory")
        _validate_format(args, ("tree", "paths"))
        return

    if name == "read_file":
        path = agent.path(args["path"])
        if not path.is_file():
            raise wrong_kind_error(agent, path, "file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        return

    if name == "search":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        agent.path(args.get("path", "."))
        _validate_format(args, ("lines", "paths"))
        return

    if name == "run_shell":
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        return

    if name == "write_file":
        path = agent.path(args["path"])
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")
        if "content" not in args:
            raise ValueError("missing content")
        return

    if name == "patch_file":
        # patch_file 故意做得很严格：old_text 必须精确命中且只能出现一次，
        # 这样修改行为才是确定的，失败原因也更容易解释。
        path = agent.path(args["path"])
        if not path.is_file():
            raise wrong_kind_error(agent, path, "file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise patch_match_error(agent, count, _relative_to_root(agent, path))
        return

    if name == "delegate":
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        max_steps = int(args.get("max_steps", 3))
        if max_steps < 1 or max_steps > DELEGATE_MAX_STEPS_CEILING:
            raise ValueError(f"max_steps must be in [1, {DELEGATE_MAX_STEPS_CEILING}]")
        return


def _validate_format(args, allowed):
    """`format` 只收白名单里的值，写错在校验阶段就打回。

    报错里带上合法取值，是因为这条信息**只能从这里拿到**：模型看到的 schema
    描述会被上下文裁剪，而一个不认识的 format 静默退回默认值的话，模型会以为
    自己拿到的是 paths 格式，然后按 paths 去拆一段 tree 格式的输出。
    """
    value = args.get("format")
    if value is None:
        return
    if str(value) not in allowed:
        raise ValueError(f"format must be one of {', '.join(allowed)}; got {value!r}")


def _split_match_row(agent, row):
    """把一行 `path:line:text` 拆成 (工作区相对路径, 其余)。

    两处都要小心，都是踩过的：

    - **Windows 盘符**。rg 拿到的是绝对路径，于是一行长得像
      `E:\\ws\\src\\a.py:1:text`，直接 `split(":", 1)` 切出来的是 `E`。
    - **两条实现路径的输出必须一致**。装了 rg 走 rg、没装走纯 Python 回退，
      而 rg 按我们传进去的绝对路径原样打印、回退那条打印的是相对路径。
      同一段计划在两台机器上拿到不同形状，是最难查的一类问题。顺带这也
      不再把宿主的绝对路径漏进模型上下文。
    """
    head, sep, rest = row.partition(":")
    # 盘符：`E:` 后面必然还跟着一段路径，把它接回去再切一次。
    if sep and len(head) == 1 and head.isalpha() and rest[:1] in ("/", "\\"):
        more_head, sep, rest = rest.partition(":")
        head = head + ":" + more_head
    if not sep:
        return "", row
    candidate = Path(head)
    try:
        head = candidate.resolve().relative_to(agent.root.resolve()).as_posix()
    except (ValueError, OSError):
        head = head.replace("\\", "/")
    return head, rest


def _search_output(agent, text, args):
    """把 rg / 回退两条路径的输出统一成工作区相对路径，再按 format 组装。"""
    rows = []
    for row in text.splitlines():
        head, rest = _split_match_row(agent, row)
        rows.append((head, f"{head}:{rest}" if head else row))
    if str(args.get("format", "lines")) != "paths":
        return "\n".join(item[1] for item in rows)
    seen = []
    for head, _ in rows:
        if head and head not in seen:
            seen.append(head)
    return "\n".join(seen)


def _workspace_relative(agent, entry):
    """始终用正斜杠。

    `Path.relative_to` 在 Windows 上给的是 `src\\cache.py`，而这串东西会被喂回
    `read_file(path=...)`，也会出现在计划的字符串处理里。同一段计划在两个平台上
    形状不同，是最难查的一类问题。
    """
    return entry.relative_to(agent.root).as_posix()


def tool_list_files(agent, args):
    path = agent.path(args.get("path", "."))
    if not path.is_dir():
        raise wrong_kind_error(agent, path, "directory")
    entries = [
        item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        if item.name not in IGNORED_PATH_NAMES
    ]
    if str(args.get("format", "tree")) == "paths":
        # 一行一个裸文件路径，不带 `[F] ` 前缀、不含目录。这是给 run_plan 里的
        # 代码用的：计划沙箱没有属性访问，`row.replace(...)` 走不通，而带前缀的
        # 输出必须先拆才能喂回 read_file。
        names = [_workspace_relative(agent, item) for item in entries[:200] if item.is_file()]
        return "\n".join(names) or "(no files)"
    lines = []
    for entry in entries[:200]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {_workspace_relative(agent, entry)}")
    return "\n".join(lines) or "(empty)"


def tool_read_file(agent, args):
    path = agent.path(args["path"])
    if not path.is_file():
        raise wrong_kind_error(agent, path, "file")
    start = int(args.get("start", 1))
    end = int(args.get("end", 200))
    if start < 1 or end < start:
        raise ValueError("invalid line range")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
    return f"# {path.relative_to(agent.root)}\n{body}"


def tool_search(agent, args):
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    path = agent.path(args.get("path", "."))

    if shutil.which("rg"):
        # 优先用 rg，因为搜索会非常频繁，搜索延迟会直接影响 agent 控制循环。
        result = subprocess.run(
            ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
            cwd=agent.root,
            capture_output=True,
            text=True,
            # 见 tool_run_shell 里那段注释：不指定编码时，捕获输出会**静默变成
            # None**，而不是抛异常。
            encoding="utf-8",
            errors="replace",
        )
        text = (result.stdout or "").strip() or (result.stderr or "").strip()
        return _search_output(agent, text, args) or "(no matches)"

    matches = []
    files = [path] if path.is_file() else [
        item for item in path.rglob("*")
        if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(agent.root).parts)
    ]
    for file_path in files:
        for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pattern.lower() in line.lower():
                matches.append(f"{_workspace_relative(agent, file_path)}:{number}:{line}")
                if len(matches) >= 200:
                    return _search_output(agent, "\n".join(matches), args)
    return _search_output(agent, "\n".join(matches), args) or "(no matches)"


def tool_run_shell(agent, args):
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be in [1, 120]")
    result = subprocess.run(
        command,
        cwd=agent.root,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        # 必须显式指定编码。不指定时 Python 按宿主 ANSI 代码页解码，中文 Windows
        # 上是 GBK——而 `capture_output=True` 的读取发生在**子线程**里，那里抛出的
        # UnicodeDecodeError 传不回主线程：`result.stdout` 会**静默变成 None**，
        # 下一句 `.strip()` 于是报 "'NoneType' object has no attribute 'strip'"。
        #
        # 症状比 CLAUDE.md 里记的更隐蔽（那里写的是"直接抛 UnicodeDecodeError"，
        # 那是 capture_output=False 时的形状）。k=3 真实跑批里这条吃掉了 4 次调用，
        # 触发条件是 `git log`——fixture 副本本身不是 git 仓库，git 会往上走到本
        # 仓库，于是读到中文提交信息。
        encoding="utf-8",
        errors="replace",
        # 这里传入的是过滤后的环境变量，而不是直接继承整个父 shell 环境，
        # 目的是减少敏感信息被意外带进命令执行环境的风险。
        env=agent.shell_env(),
    )
    return textwrap.dedent(
        f"""\
        exit_code: {result.returncode}
        stdout:
        {(result.stdout or "").strip() or "(empty)"}
        stderr:
        {(result.stderr or "").strip() or "(empty)"}
        """
    ).strip()


def tool_write_file(agent, args):
    path = agent.path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(agent.root)} ({len(content)} chars)"


PATCH_EXCERPT_CONTEXT_LINES = 2
PATCH_EXCERPT_MAX_LINES = 40


def _patched_excerpt(updated_text, start_offset, new_text):
    """把改动落地后的那一段按 read_file 的格式回显出来。

    为什么存在：`patch_file` 从前只回一句 `patched server.py`，模型没有任何办法
    确认改动落成了什么样，于是紧接着回读同一个文件来验证。实测一次真实运行里
    有 3 次 read_file 纯粹是在做这件事，每次要花掉一个约 20 秒的模型往返。
    回显的代价是几十个 token，回读的代价是一整轮，两边不对称。

    格式刻意和 `tool_read_file` 一致（`{行号:>4}: {正文}`），这样模型看到的
    “文件现在长什么样”在两个工具之间是同一种东西。
    """
    lines = updated_text.splitlines()
    start_line = updated_text[:start_offset].count("\n") + 1
    end_line = start_line + str(new_text).count("\n")
    first = max(1, start_line - PATCH_EXCERPT_CONTEXT_LINES)
    last = min(len(lines), end_line + PATCH_EXCERPT_CONTEXT_LINES)
    truncated = (last - first + 1) > PATCH_EXCERPT_MAX_LINES
    if truncated:
        last = first + PATCH_EXCERPT_MAX_LINES - 1
    body = "\n".join(
        f"{number:>4}: {line}" for number, line in enumerate(lines[first - 1:last], start=first)
    )
    if truncated:
        body += "\n      ... (excerpt truncated)"
    return body


def tool_patch_file(agent, args):
    path = agent.path(args["path"])
    if not path.is_file():
        raise wrong_kind_error(agent, path, "file")
    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ValueError("old_text must not be empty")
    if "new_text" not in args:
        raise ValueError("missing new_text")
    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        raise patch_match_error(agent, count, _relative_to_root(agent, path))
    new_text = str(args["new_text"])
    start_offset = text.index(old_text)
    updated = text.replace(old_text, new_text, 1)
    path.write_text(updated, encoding="utf-8")
    relative = path.relative_to(agent.root)
    return f"patched {relative}\n{_patched_excerpt(updated, start_offset, new_text)}"


def tool_delegate(agent, args):
    if agent.depth >= agent.max_depth:
        raise ValueError("delegate depth exceeded")
    task = str(args.get("task", "")).strip()
    if not task:
        raise ValueError("task must not be empty")
    max_steps = int(args.get("max_steps", 3))
    if max_steps < 1 or max_steps > DELEGATE_MAX_STEPS_CEILING:
        raise ValueError(f"max_steps must be in [1, {DELEGATE_MAX_STEPS_CEILING}]")

    from .runtime import CodingForMe

    child = CodingForMe(
        model_client=agent.model_client,
        workspace=agent.workspace,
        session_store=agent.session_store,
        run_store=agent.run_store,
        approval_policy="never",
        max_steps=max_steps,
        max_new_tokens=agent.max_new_tokens,
        depth=agent.depth + 1,
        max_depth=agent.max_depth,
        read_only=True,
        secret_env_names=agent.secret_env_names,
        shell_env_allowlist=agent.shell_env_allowlist,
    )
    # 委派的目标是“调查”，不是“放权执行”。
    # 子 agent 以只读方式运行、步数更少，最后只把结论文本返回给父 agent。
    child.session["memory"]["task"] = task
    child.session["memory"]["notes"] = [clip(agent.history_text(), 300)]
    return "delegate_result:\n" + child.ask(task)


def tool_run_plan(agent, args):
    """`run_plan` 的执行入口——薄封装，真正的驱动在 `runtime.CodingForMe.execute_plan()`。

    为什么驱动逻辑不在这里：它要拿步数预算、要往 trace 里写每个内层调用、要用
    当前 run 的 task_state，全都是 runtime 层的东西。tools.py 保持"定义 + 校验 +
    执行"的职责，不去碰控制循环的状态。
    """
    return agent.execute_plan(str(args.get("plan", "")))


_TOOL_RUNNERS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search": tool_search,
    "run_shell": tool_run_shell,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
}
