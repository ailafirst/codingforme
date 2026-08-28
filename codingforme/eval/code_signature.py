"""被测代码本身的签名：哪些源码变了，就该判定两次跑批不可比。

为什么存在：
`HarnessSpec.fingerprint()` 的定义是「用来判定两次结果是否可比」，但它一开始
只吃**配置字段**（审批策略、步数上限、只读、feature flags、工具白名单、context
预算）。后来补上了提示词模板和工具 schema 两个哈希，仍然不够——因为改变模型
行为的东西还有第三类：**上下文是怎么组装的、模型输出是怎么解析的、记忆是怎么
召回的**，这些既不是配置也不是文本常量，而是代码。

实测后果（2026-08-18 复盘）：T2-1 把上下文从「压平成一段文本塞进单条 user
message」换成了标准 messages 数组，改动前后各跑了一次 12 任务 × 3 轮的正式
跑批，两份工件里的 `code_signature` 完全相同（`sha256:db7c156bcbee7`）。也就是
说，一个足以让走到终点率从 76% 变成 97% 的改动，在「这两份数据可不可比」这个
字段上完全不可见。

做法：把「会改变模型行为」的那几个模块的源码规范化之后一起哈希。规范化 =
`ast` 解析 → 去掉 docstring → `ast.unparse` 重新生成，因此**注释、docstring、
空行、缩进风格的改动不会让签名变**，只有真正的代码结构变化才会变。

刻意接受的取舍：**宁可误报不可比，不可漏报可比。** 一次与模型行为无关的重构
（改个变量名、拆个函数）也会让签名变，于是两份其实等价的数据被标成不可比——
代价是多跑一次。反过来漏报的代价是把两份不可比的数据画进同一张图，那是错误
结论，不是多花时间。

刻意**不**包含的东西：`codingforme/eval/` 自己（评测代码改了不影响被测对象的
行为）、`cli.py` / `config.py` / `run_store.py`（交互与落盘，不进模型的输入
输出链路）。
"""

import ast
import hashlib
import importlib
import inspect

# 会改变模型行为的模块。判断标准是「它的产物进不进模型的输入，或者它解不解释
# 模型的输出」：
MODEL_FACING_MODULES = (
    "codingforme.runtime",          # 控制循环、工具闸口、prompt 前缀、输出解析
    "codingforme.context_manager",  # 上下文分段、预算裁剪、messages 摆位
    "codingforme.memory",           # 三层记忆与召回排序
    "codingforme.tools",            # 工具定义、参数校验、执行
    "codingforme.models",           # 请求组装、function-calling、流式、usage 提取
    "codingforme.workspace",        # 仓库快照文本
    "codingforme.plans",            # 受限编排的语法与上限——直接决定模型能写出什么计划
)

SOURCE_UNAVAILABLE = "<source-unavailable>"


def _strip_docstrings(tree):
    """去掉模块、类、函数的 docstring。

    docstring 在这个仓库里是大头（很多函数的说明比实现长几倍），而且改得最勤。
    不去掉的话，写一段解释就会让签名变，签名很快就没人信了。
    """
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return tree


def normalized_source(source):
    """源码 → 规范化后的等价源码（无注释、无 docstring、统一格式）。

    解析不了的（理论上不该发生）原样返回：宁可让签名对格式敏感，也不能让它
    静默变成一个和源码无关的常数。
    """
    try:
        return ast.unparse(_strip_docstrings(ast.parse(source)))
    except SyntaxError:
        return source


def module_signatures(module_names=MODEL_FACING_MODULES):
    """逐模块的签名。签名变了时用它定位是哪个模块变的。"""
    digests = {}
    for name in module_names:
        try:
            source = inspect.getsource(importlib.import_module(name))
        except (OSError, TypeError, ImportError):
            # 拿不到源码（打包成 zip、frozen）时记一个显式标记而不是跳过：
            # 跳过等于让两个不同的构建产出同一个签名，正是这个模块要防的事。
            digests[name] = SOURCE_UNAVAILABLE
            continue
        digests[name] = hashlib.sha256(normalized_source(source).encode("utf-8")).hexdigest()
    return digests


def model_facing_code_signature(module_names=MODEL_FACING_MODULES):
    """全部模型相关模块的合并签名，进 `HarnessSpec.code_signature()`。"""
    digests = module_signatures(module_names)
    payload = "\n".join("%s=%s" % (name, digests[name]) for name in sorted(digests))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
