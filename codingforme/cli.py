"""命令行入口。

这个模块负责把“用户怎么启动 coding-for-me”翻译成 runtime 能理解的对象：
解析参数、挑模型后端、构建工作区快照、恢复或新建 session，
最后进入 one-shot 或交互式循环。
"""

import argparse
import os
import shutil
import sys
import textwrap
import unicodedata

from . import models
from .config import load_project_env, project_root, provider_env
from .models import OpenAICompatibleModelClient
from .runtime import CodingForMe, SessionStore
from .workspace import WorkspaceContext, middle

DEFAULT_SECRET_ENV_NAMES = (
    "CODINGFORME_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_API_TOKEN",
    "GITHUB_PAT",
    "GH_PAT",
)

# Claude Code 风格的欢迎界面：一个 ASCII 场景 logo（机器人侧坐在电脑前写代码）
# + ✻ 火花 + 一两行平静的介绍。
# 场景故意只用纯 ASCII，避免不同终端对 unicode 宽度的解释不一致而把框线撑歪。
# 各行宽度不一致没关系，build_welcome 会在居中前把整块补齐到等宽再统一居中。
WELCOME_MASCOT = (
    r"          (✦)                                  ",
    r"       ╭──────╮          ┌────────────────┐    ",
    r"       │ ● ‿ ●│          │  > while (1) { │     ",
    r"       ╰──┬───╯          │      code();   │     ",
    r"      ╭────┴────╮        │  }             │     ",
    r"      │  [==]   │──╮     └────────────────┘     ",
    r"      ╰──┬───┬──╯  ╰──[ =keyboard= ]            ",
    r"        _╯   ╰_                                 ",
    r"       (__) (__)                               ",
)
WELCOME_SPARK = "✻"
WELCOME_TITLE = "Welcome to coding-for-me"
WELCOME_INTRO = (
    "A small local coding agent that lives inside your repo.",
    "It reads, edits, and runs code via an OpenAI-compatible model.",
)
WELCOME_HINT = "/help for commands · /context · /compact · /memory · /session · /reset · /exit"
HELP_DETAILS = textwrap.dedent(
    """\
    Commands:
    /help    Show this help message.
    /context Show the context window tier and budget; /context 128k sets the tier.
    /compact Compact the transcript now instead of waiting for the pressure threshold.
    /memory  Show the agent's distilled working memory.
    /session Show the path to the saved session file.
    /reset   Clear the current session history and memory.
    /exit    Exit the agent.
    """
).strip()


DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_OPENAI_BASE_URL = "https://www.right.codes/codex/v1"
SECRET_ENV_NAMES_VAR = "CODINGFORME_SECRET_ENV_NAMES"


def _effective_model(args):
    # 模型选择优先级：
    # 1. 用户显式传入 --model
    # 2. 环境变量 CODINGFORME_OPENAI_MODEL
    # 3. 代码里的默认值
    explicit_model = getattr(args, "model", None)
    if explicit_model:
        return explicit_model
    return provider_env("CODINGFORME_OPENAI_MODEL", ("OPENAI_MODEL",)) or DEFAULT_OPENAI_MODEL


def _configured_context_window(args):
    """上下文窗口的显式覆盖：CLI `--context-window` 优先于 `CODINGFORME_CONTEXT_WINDOW`。

    两者都没给就返回 None，交给 `models.resolve_context_window()` 去查已知后端表
    和 litellm 注册表。优先级和 provider 那几个配置项保持一致：CLI > 环境变量 > 自动。
    """
    if getattr(args, "context_window", None):
        return int(args.context_window)
    raw = provider_env("CODINGFORME_CONTEXT_WINDOW")
    if not raw:
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        # 配错了不要静默当成没配——那会让人以为窗口生效了，实际回落到了默认档。
        raise SystemExit(f"CODINGFORME_CONTEXT_WINDOW must be an integer, got: {raw!r}")


def _format_tokens(value):
    """把 token 数写成人能一眼读懂的形式：1000000 → 1M，128000 → 128k。"""
    value = int(value)
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return f"{value:,}"


def _compact_report(report):
    """`/compact` 显示什么。

    推不动时也要说出**原因**,不能只回一句「没压」——「热尾就那么长、没什么可压」
    和「这个功能被变体关掉了」对用户是两件事,前者是正常的,后者要去改配置。
    """
    if not report.get("compacted"):
        return "nothing compacted: %s\n%d transcript entries, %d already covered by the summary" % (
            report.get("reason") or "unknown reason",
            int(report.get("entries", 0)),
            int(report.get("covered", 0)),
        )
    lines = [
        "compacted %d more transcript entries (%d of %d now covered)"
        % (
            int(report.get("newly_covered", 0)),
            int(report.get("covered", 0)),
            int(report.get("entries", 0)),
        ),
        "transcript %d -> %d tokens (saved %d)"
        % (
            int(report.get("before_tokens", 0)),
            int(report.get("after_tokens", 0)),
            int(report.get("saved_tokens", 0)),
        ),
        # 摘要正文要给出来。压缩是**用户主动**发起的,他有权当场看到换进去的是什么;
        # 只报一个 token 差值等于让他相信压掉的部分无关紧要。
        "",
        str(report.get("summary", "")).strip() or "(the summary is empty)",
    ]
    return "\n".join(lines)


def _context_status(agent):
    """`/context` 不带参数时显示什么。

    四样缺一不可：窗口档位、它是怎么定下来的（explicit / known-model /
    litellm-registry / default——不写来源就没法判断「这是我设的」还是「自动猜的」）、
    预算算式的**逐项**扣除（只给一个总数的话，想不通「1M 的窗口为什么只剩 11 万」，
    也不知道该去调哪一笔），以及可选档位清单（不列出来，用户就得去翻源码才知道
    能填什么）。窗口小到扣完仍不够时还要多一行警告——触底和「预算正好是这个数」
    含义相反。
    """
    window = int(getattr(agent, "context_window", 0) or 0)
    source = str(getattr(agent, "context_window_source", "") or "unknown")
    budget = int(getattr(agent, "context_budget", 0) or 0)
    breakdown = dict(getattr(agent, "context_budget_breakdown", {}) or {})
    lines = [
        f"window     {_format_tokens(window)} tokens  (source: {source})",
        f"budget     {budget:,} tokens for prompt assembly",
    ]
    if breakdown:
        # 算式逐项写出来，不然「1M 的窗口为什么只有 10 万预算」看不出所以然，
        # 而且四笔扣除各自该调什么也无从判断。每行保持在 60 字符以内：REPL 的
        # 面板会硬折行，折断的数字比不显示更难读。
        capped = int(breakdown.get("capped_tokens", 0))
        capped_note = "  (policy cap)" if breakdown.get("window_capped") else ""
        lines.extend(
            [
                f"  usable   {capped:,}{capped_note}",
                f"  - output {int(breakdown.get('output_reserve_tokens', 0)):,}"
                f"  (max_new_tokens)",
                f"  - schema {int(breakdown.get('tool_schema_tokens', 0)):,}"
                f"  (resent every turn)",
                f"  - frames {int(breakdown.get('message_framing_tokens', 0)):,}"
                f"  ({int(breakdown.get('expected_messages', 0))} messages)",
                f"  - margin {int(breakdown.get('tokenizer_margin_tokens', 0)):,}"
                f"  ({models.TOKENIZER_MARGIN_RATIO:.0%} tokenizer drift)",
            ]
        )
        if breakdown.get("floored"):
            # 触底是「这个窗口装不下」，不是「预算就是这么多」。不说出来的话，
            # 用户会以为还有 1000 个 token 可用，而实际上扣完四笔已经是负数。
            lines.append(
                f"  WARNING  window too small; floored to {models.BUDGET_FLOOR_TOKENS:,}"
            )
    lines.extend(_context_pressure_lines(agent))
    lines.extend(
        [
            "tiers      " + " ".join(_format_tokens(b) for b in models.WINDOW_BUCKETS),
            "usage      /context 128k   (128000 and 1m work too)",
        ]
    )
    return "\n".join(lines)


def _context_pressure_lines(agent):
    """`/context` 里「现在占了多少、离自动压缩还有多远」那一段。

    加它是因为整个上下文治理里唯一给用户的手是 `/compact`,而在这之前它是**盲的**:
    用户看得到窗口档位和预算总额,看不到自己此刻占了多少、也看不到系统会在哪个点
    自己动手。要决定「现在该不该压」,这两个数才是判据。

    还没跑过任何一轮时明说「还没有」,不显示 0%——「占用率是零」和「还没量过」在
    同一个数字上分不出来,而后者根本不该拿来做决策。
    """
    manager = getattr(agent, "context_manager", None)
    thresholds = manager.compression_thresholds() if manager is not None else {}
    if not thresholds:
        return []
    budget = int(thresholds.get("budget_tokens", 0) or 0)
    metadata = dict(getattr(agent, "last_prompt_metadata", {}) or {})
    used = int(metadata.get("prompt_tokens") or 0)
    if used and budget:
        lines = ["pressure   %s / %s tokens (%.1f%%) as of the last turn"
                 % (f"{used:,}", f"{budget:,}", 100.0 * used / budget)]
    else:
        lines = ["pressure   no turn has been assembled yet in this session"]
    if thresholds.get("graded"):
        lines.extend(
            [
                "  trigger  %s  (compress above this)" % f"{int(thresholds.get('trigger_tokens', 0)):,}",
                "  target   %s  (compress down to this)" % f"{int(thresholds.get('target_tokens', 0)):,}",
            ]
        )
    else:
        # 分级被关掉时两个点都是 100%,不说出来的话「触发点 = 预算」看起来像显示错了。
        lines.append("  trigger  %s  (graded compression is OFF: only overflow triggers)"
                     % f"{int(thresholds.get('trigger_tokens', 0)):,}")
    state = dict((getattr(agent, "session", {}) or {}).get("context_summary") or {})
    entries = len((getattr(agent, "session", {}) or {}).get("history") or [])
    covered = int(state.get("covered", 0) or 0)
    # 摘要覆盖到哪儿,是用户判断「再敲一次 /compact 还有没有用」的唯一依据。
    lines.append("  summary  %d of %d transcript entries covered" % (covered, entries))
    return lines


def _apply_context_window(agent, raw):
    """`/context <size>`：换档位并重新派生预算。

    解析失败**不改任何东西**，只回一句错误——静默回落会让人以为设生效了，
    而实际还跑在原来的档位上，这正是 `parse_window_tokens()` 抛异常的理由。
    """
    try:
        requested = models.parse_window_tokens(raw)
    except ValueError:
        return f"not a context window size: {raw!r} (try 128k, 128000, or 1m)"
    window, source, budget = agent.set_context_window(requested)
    note = ""
    if window != requested:
        # 档位是向下取整的，用户填 100k 会落到 64k。不说出来的话，他会以为设成了 100k。
        note = f" (floored from {requested:,} to the nearest tier)"
    return (
        f"window     {_format_tokens(window)} tokens  (source: {source}){note}\n"
        f"budget     {budget:,} tokens for prompt assembly"
    )


def _configured_secret_names(args):
    configured_secret_names = set(DEFAULT_SECRET_ENV_NAMES)
    configured_secret_names.update(str(name).upper() for name in args.secret_env_names)
    extra_names = os.environ.get(SECRET_ENV_NAMES_VAR, "")
    if extra_names.strip():
        configured_secret_names.update(
            item.strip().upper()
            for item in extra_names.split(",")
            if item.strip()
        )
    return sorted(configured_secret_names)


def _env_flag(name):
    """把一个可选的布尔环境变量读成 True / False / None。

    None 表示"没配"，交给 models.resolve_capabilities() 的默认值和已知后端表，
    而不是在这里替用户拍板。
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _configured_capabilities():
    """从环境变量读后端能力覆盖。

    这样接一个新后端是改配置，不是改 models.py 里的 host 白名单。
    """
    return {
        "native_tool_calls": _env_flag("CODINGFORME_OPENAI_NATIVE_TOOL_CALLS"),
        "prompt_cache_key": _env_flag("CODINGFORME_OPENAI_PROMPT_CACHE_KEY"),
    }


def _build_model_client(args):
    model = _effective_model(args)
    base_url = getattr(args, "base_url", None) or provider_env("CODINGFORME_OPENAI_API_BASE", ("OPENAI_API_BASE",), DEFAULT_OPENAI_BASE_URL)
    api_key = provider_env("CODINGFORME_OPENAI_API_KEY", ("OPENAI_API_KEY",))
    return OpenAICompatibleModelClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=args.temperature,
        timeout=getattr(args, "openai_timeout", 300),
        capabilities=_configured_capabilities(),
    )


def _make_output_resilient():
    """让 stdout/stderr 遇到宿主编码表达不了的字符时降级，而不是崩掉整个进程。

    踩过的坑：Windows 默认控制台是 GBK，横幅里的 `✦`(U+2726) 编不出来，于是
    `python -m codingforme` 在打招呼那一行就抛 UnicodeEncodeError，agent 根本起不来。
    横幅只是最先撞上的那个——模型答案里出现 emoji、工具回显里带上非 GBK 字符，
    一样会把整个 REPL 打断。

    只改 `errors` 不改 `encoding` 是刻意的：强行改成 utf-8 会让 GBK 终端把**所有**
    中文渲染成乱码，而本项目的输出大量是中文。保持宿主编码 + `replace`，
    代价只是少数装饰字符显示成 `?`，中文和框线（GBK 都能编）完全不受影响。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            # 被重定向成不支持 reconfigure 的对象（测试里的 StringIO 等）就跳过。
            pass


# UI 里用到的、GBK 编不出来的全部 4 个装饰字符及其 ASCII 近似物。
# （框线那 25 个 unicode 字符 GBK 都能编，不需要降级。）
_GLYPH_FALLBACKS = {
    "✦": "*",   # 吉祥物头顶的火花
    "✻": "*",   # 标题火花
    "❯": ">",   # 输入提示符
    "‿": "_",   # 吉祥物的嘴
}


def _terminal_safe(text):
    """把当前终端编码写不出的装饰字符换成 ASCII 近似物。

    只作用于我们自己的 UI chrome（横幅、提示符、回答框标题）。光靠
    `_make_output_resilient()` 的 `errors="replace"` 不会崩，但会印出一串 `?`；
    这里给出的是有意设计的降级形态，而不是"坏掉的样子"。

    模型输出不走这里——那部分内容无法预先枚举，交给 `errors="replace"` 兜底。
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except (UnicodeEncodeError, LookupError):
        pass
    for fancy, plain in _GLYPH_FALLBACKS.items():
        text = text.replace(fancy, plain)
    return text


def _display_width(text):
    """按终端显示宽度计字符宽度（CJK / 全角算 2，其余算 1）。"""
    width = 0
    for ch in str(text):
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def _supports_color():
    """判断 stdout 是否适合输出 ANSI 颜色；在 Windows 上顺带开启 VT 处理。"""
    if os.environ.get("NO_COLOR"):
        return False
    stream = sys.stdout
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            return False
    return True


def _welcome_palette(color):
    if not color:
        return {key: "" for key in ("accent", "bold", "dim", "border", "reset")}
    return {
        "accent": "\033[38;5;209m",  # Claude 风格的暖橙色火花
        "bold": "\033[1m",
        "dim": "\033[38;5;245m",
        "border": "\033[38;5;240m",
        "reset": "\033[0m",
    }


def _box_width(cols=None):
    """欢迎框 / 对话框统一的外框宽度：撑满当前终端宽度（留出两列边距），
    最窄 58，避免在窄终端里折行。撑满是为了不在右侧留一大片空白。"""
    if cols is None:
        cols = shutil.get_terminal_size((80, 20)).columns
    return max(58, cols - 2)


def _wrap_display(text, width):
    """按终端显示宽度把文本折到指定宽度，返回纯文本行列表。

    用 `_display_width` 计宽，所以中英文混排不会把右边框撑歪；
    保留原文里的换行，长行（含无空格的 CJK）会按宽度硬折。
    """
    width = max(1, width)
    lines = []
    for raw in str(text).split("\n"):
        current = ""
        current_w = 0
        for ch in raw:
            ch_w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if current and current_w + ch_w > width:
                lines.append(current)
                current, current_w = "", 0
            current += ch
            current_w += ch_w
        lines.append(current)
    return lines or [""]


def _render_box(text, *, title=None, color=False, accent_border=False):
    """把一段文本渲染进一个圆角框，返回多行字符串。

    为什么存在：
    欢迎横幅、每一轮的回答都共用同一套圆角框 UI，这样对话过程中框线
    始终在屏幕上，不会“聊几句就看不到 UI 了”。

    - `title` 会嵌在顶边线上：`╭─ title ───╮`
    - `color=True` 时插入 ANSI 颜色，padding 仍按可见宽度计算
    - 文本按当前框宽自动折行（中英文混排安全）
    """
    pal = _welcome_palette(color)
    border, reset, accent = pal["border"], pal["reset"], pal["accent"]
    box_width = _box_width()
    inner = box_width - 4
    line_border = accent if accent_border else border

    if title:
        label = f"─ {title} "
        label_w = _display_width(label)
        dashes = max(0, box_width - 2 - label_w)
        colored = f"{line_border}╭{accent}{label}{reset}{line_border}{'─' * dashes}╮{reset}"
        rows = [colored]
    else:
        rows = [f"{line_border}╭{'─' * (box_width - 2)}╮{reset}"]

    for plain in _wrap_display(text, inner):
        pad = max(0, inner - _display_width(plain))
        rows.append(f"{line_border}│{reset} {plain}{' ' * pad} {line_border}│{reset}")
    rows.append(f"{line_border}╰{'─' * (box_width - 2)}╯{reset}")
    return "\n".join(rows)


def build_welcome(agent, model, color=False):
    """构建 Claude Code 风格的欢迎横幅：圆角框 + ✻ 火花 + 平静的介绍。

    为什么存在：
    启动时给用户一个干净、一眼能看懂当前现场（工作区、模型、分支、审批、会话）的入口画面。

    输入 / 输出：
    - 输入：已装配好的 agent、要显示的 model 名、是否上色
    - 输出：一整段多行字符串

    `color=False` 时输出纯文本（测试和重定向到文件时保持等宽对齐）；
    `color=True` 时插入 ANSI 颜色，padding 仍按可见文本宽度计算，框线不会错位。
    """
    box_width = _box_width()
    inner = box_width - 4  # 去掉左右各「框线 + 一个空格」
    pal = _welcome_palette(color)
    border, reset, dim = pal["border"], pal["reset"], pal["dim"]

    def hline(left_corner, right_corner):
        return f"{border}{left_corner}{'─' * (box_width - 2)}{right_corner}{reset}"

    def line(plain, colored=None):
        colored = plain if colored is None else colored
        pad = max(0, inner - _display_width(plain))
        return f"{border}│{reset} {colored}{' ' * pad} {border}│{reset}"

    def center(plain, colored=None):
        colored = plain if colored is None else colored
        total = max(0, inner - _display_width(plain))
        left = total // 2
        right = total - left
        return f"{border}│{reset} {' ' * left}{colored}{' ' * right} {border}│{reset}"

    def info(label, value):
        value = middle(value, inner - 12)
        plain = f"  {label:<9}{value}"
        colored = f"  {dim}{label:<9}{reset}{value}"
        return line(plain, colored)

    title_plain = f"{WELCOME_SPARK} {WELCOME_TITLE}"
    title_colored = f"{pal['accent']}{WELCOME_SPARK}{reset} {pal['bold']}{WELCOME_TITLE}{reset}"

    rows = [hline("╭", "╮"), line("")]
    # 先把整块 logo 补齐到等宽，再逐行居中，这样多行场景作为一个整体对齐。
    art_width = max((_display_width(art) for art in WELCOME_MASCOT), default=0)
    for art in WELCOME_MASCOT:
        padded = art + " " * max(0, art_width - _display_width(art))
        rows.append(center(padded, f"{pal['accent']}{padded}{reset}"))
    rows.append(line(""))
    rows.append(line(title_plain, title_colored))
    rows.append(line(""))
    for intro in WELCOME_INTRO:
        text = "  " + middle(intro, inner - 2)
        rows.append(line(text, f"{dim}{text}{reset}"))
    rows.append(line(""))
    for label, value in (
        ("cwd", agent.workspace.cwd),
        ("model", model),
        ("branch", agent.workspace.branch),
        ("approval", agent.approval_policy),
        ("session", agent.session["id"]),
    ):
        rows.append(info(label, value))
    rows.append(line(""))
    hint = middle(WELCOME_HINT, inner - 2)
    rows.append(line("  " + hint, f"  {dim}{hint}{reset}"))
    rows.append(hline("╰", "╯"))
    return "\n".join(rows)


def build_agent(args, on_token=None):
    """根据 CLI 参数装配出一个可运行的 CodingForMe 实例。

    为什么存在：
    命令行参数只是字符串和开关，runtime 需要的是已经装配好的对象图：
    model client、workspace snapshot、session store、secret 配置等。
    这个函数负责把“启动参数”翻译成“agent 运行现场”。

    输入 / 输出：
    - 输入：`argparse` 解析后的 `args`
    - 输出：一个新的 `CodingForMe`，或一个从旧 session 恢复出来的 `CodingForMe`

    在 agent 链路里的位置：
    它是整个程序启动链路里最靠近 runtime 的装配点。`main()` 先调它，
    得到 agent 后，后面无论是 one-shot 还是 REPL 模式，都会落到 `ask()`。
    """
    # 这里是 CLI 到 runtime 的装配点：
    # 先采集工作区快照和加载项目级环境，再整理 secret 名单、模型后端和 session。
    workspace = WorkspaceContext.build(args.cwd)
    configured_secret_names = _configured_secret_names(args)
    store = SessionStore(workspace.repo_root + "/.codingforme/sessions")
    model = _build_model_client(args)
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
    if session_id:
        return CodingForMe.from_session(
            model_client=model,
            workspace=workspace,
            session_store=store,
            session_id=session_id,
            approval_policy=args.approval,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            context_window=_configured_context_window(args),
            secret_env_names=configured_secret_names,
            feature_flags={"delegate_tool": True},
            on_token=on_token,
        )
    return CodingForMe(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        context_window=_configured_context_window(args),
        secret_env_names=configured_secret_names,
        # 交互式用法保留 delegate：它默认关是为了让评测跑批之间可比，不是因为
        # 这个能力有问题。
        feature_flags={"delegate_tool": True},
        on_token=on_token,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal coding agent for OpenAI-compatible models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override. Defaults to CODINGFORME_OPENAI_MODEL env var, or gpt-5.4.",
    )
    parser.add_argument("--base-url", default=None, help="API base URL.")
    parser.add_argument("--openai-timeout", type=int, default=300, help="Request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    parser.add_argument("--approval", choices=("ask", "auto", "never"), default="ask", help="Approval policy for risky tools.")
    parser.add_argument(
        "--secret-env-name",
        dest="secret_env_names",
        action="append",
        default=[],
        help="Extra environment variable names to treat as secrets for trace/report redaction.",
    )
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum model output tokens per step.")
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help=(
            "Context window in tokens. Overrides auto-detection "
            "(known-model table, then the litellm registry). "
            "Also settable via CODINGFORME_CONTEXT_WINDOW."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    # 必须在任何 print 之前：横幅是第一处输出，也是第一处会因编码崩掉的地方。
    _make_output_resilient()
    # `.env` 只从本仓库读，与 --cwd 指向哪里无关（见 config.project_root 的说明）。
    #
    # 放在 main() 而不是 build_agent() 里：往 os.environ 里灌东西是**进程级副作用**，
    # 只该由 CLI 启动这一处做。放进 build_agent 会让直接调它的测试读到开发机上真实的
    # `.env`，于是"默认模型是什么"这类断言变成随开发机配置而定——实测正是这样挂掉了
    # 4 条用例。必须排在 build_agent 之前，模型/密钥的解析都依赖它。
    load_project_env(project_root())
    use_color = _supports_color()
    pal = _welcome_palette(use_color)

    def on_token(text):
        # 模型还在流式生成时的实时预览：用暗色打印，和后面 show() 渲染的
        # 最终圆角框答案区分开，避免看起来像是同一段内容被打印了两遍。
        print(f"{pal['dim']}{text}{pal['reset']}", end="", flush=True)

    agent = build_agent(args, on_token=on_token)

    model = getattr(agent.model_client, "model", DEFAULT_OPENAI_MODEL)
    print(_terminal_safe(build_welcome(agent, model=model, color=use_color)))
    # 输入提示符：始终在屏幕上的一段 UI chrome，配合每轮回答的框，
    # 让对话过程中圆角框风格不会“滚一会儿就消失”。
    prompt_str = f"\n{pal['accent']}{_terminal_safe('❯')}{pal['reset']} "
    answer_title = _terminal_safe(f"{WELCOME_SPARK} coding-for-me")

    def show(text, *, title=None):
        print()
        print(_render_box(text, title=title or answer_title, color=use_color))

    if args.prompt:
        # one-shot 模式：只跑一次 ask，不进入 REPL 循环。
        prompt = " ".join(args.prompt).strip()
        if prompt:
            try:
                show(agent.ask(prompt))
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        return 0

    while True:
        # 交互模式：每次读取一条用户输入，交给同一个 agent，
        # 因此 session history 和 working memory 会跨轮延续。
        try:
            user_input = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            show(HELP_DETAILS, title="/help")
            continue
        if user_input == "/context" or user_input.startswith("/context "):
            argument = user_input[len("/context"):].strip()
            if argument:
                show(_apply_context_window(agent, argument), title="/context")
            else:
                show(_context_status(agent), title="/context")
            continue
        if user_input == "/compact":
            show(_compact_report(agent.compact_context()), title="/compact")
            continue
        if user_input == "/memory":
            show(agent.memory_text(), title="/memory")
            continue
        if user_input == "/session":
            show(agent.session_path, title="/session")
            continue
        if user_input == "/reset":
            agent.reset()
            show("session reset", title="/reset")
            continue

        try:
            show(agent.ask(user_input))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
