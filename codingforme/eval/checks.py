"""声明式验证器：把「怎么算做对了」从 shell 命令变成可校验的数据。

为什么存在（改动前的三个问题）：

1. **依赖宿主环境**。此前每个任务的判定是一条 `python3 -c "..."` 的 shell 命令，
   靠 `subprocess.run(..., shell=True)` 执行。Windows 上没有 `python3`、`shell=True`
   会落到 cmd.exe（POSIX 引号规则不成立），本机 3 个基准测试因此长期失败——
   **失败原因是执行环境，不是被测系统**，这在成熟评测体系里是不该出现的状态。
2. **判定本身不可校验**。它是一段字符串，写错了只有跑起来才知道；
   `validate_benchmark()` 能校验步数预算却校验不了判定。
3. **只有一半的判定**。参照 SWE-bench 的做法，判定应当同时包含两类：

       fail-to-pass (F2P)   改动后必须成立的断言 —— 证明任务真的完成了
       pass-to-pass (P2P)   改动前后都必须成立的断言 —— 证明没有顺手破坏别的东西

   只有 F2P 时，「改对一处、砸坏三处」的解法照样满分。SWE-bench+ 的实测是
   约 31% 的通过实例被不够强的测试放过，我们那条单断言正是这种弱测试的极端形态。

于是判定改成一组**数据**：每条 check 是 `{"kind": ..., ...}` 字典，由当前解释器
在进程内执行，不起子进程、不碰 shell、不 eval 任何字符串。

P2P 由两部分合成：
  - **显式**：数据集里声明的基线断言（原有内容必须存活）。
  - **隐式**：`workspace_regressions()`——除任务声明的 `mutable_paths` 外，
    工作区必须与「agent 动手前」逐字节一致。基线在 setup **之后**拍摄是刻意的：
    setup 改文件属于布景，断言的是 agent 有没有动它不该动的东西。
"""

import hashlib
import os
from pathlib import Path

CHECKS_SCHEMA_VERSION = 1

# 带文件路径的 check kind。这些路径要和 `CodingForMe.path()` 同样的立场处理：
# 既挡 `../`，也挡 symlink 解析之后的逃逸。判定器读的必须是这次运行自己的工作区
# ——一条指向宿主文件的断言能让任务凭空「通过」，那比读错文件严重得多。
PATH_BEARING_KINDS = ("file_contains", "file_not_contains", "file_exists")

# 工作区快照要跳过的目录：agent 自己的产物（session/run/durable memory）本来就会变，
# 它们不是「被破坏的仓库内容」。
SNAPSHOT_EXCLUDED_DIRS = (".codingforme",)

GROUP_FAIL_TO_PASS = "fail_to_pass"
GROUP_PASS_TO_PASS = "pass_to_pass"
CHECK_GROUPS = (GROUP_FAIL_TO_PASS, GROUP_PASS_TO_PASS)

_MISSING = object()


class CheckContext:
    """一条 check 能看到的全部证据。

    刻意只有这三样：工作区文件、report.json、trace 事件流。判定只能基于**已落盘的
    工件**，不能去问模型、也不能重新执行 agent——这样同一批工件重算多少次结果都一样。
    """

    def __init__(self, root, report=None, trace_events=None):
        self.root = Path(root)
        self.report = dict(report or {})
        self.trace_events = list(trace_events or [])

    def resolve(self, relpath):
        """把路径锚定在 root 之下；越界返回 None。

        `validate_check()` 已经静态挡掉了 `../` 和绝对路径，这里是运行时的第二道：
        symlink 要解析之后才知道它指向哪儿。两道都要有——静态那道让错误在加载数据集时
        就暴露，运行时这道才真正决定读到的是哪个文件。
        """
        root = self.root.resolve()
        try:
            candidate = (root / str(relpath)).resolve()
        except OSError:
            return None
        if candidate != root and root not in candidate.parents:
            return None
        return candidate

    def read_text(self, relpath):
        path = self.resolve(relpath)
        if path is None or not path.is_file():
            return None
        return path.read_text(encoding="utf-8")


# ------------------------------------------------------------------ 字段查找


def _lookup(payload, dotted):
    """按点号路径取值，取不到返回 _MISSING（区别于「取到了 None」）。"""
    current = payload
    for part in str(dotted).split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.lstrip("-").isdigit():
            index = int(part)
            if -len(current) <= index < len(current):
                current = current[index]
                continue
        return _MISSING
    return current


def _event_matches(event, where):
    return all(event.get(key) == value for key, value in (where or {}).items())


# -------------------------------------------------------------- check 实现
#
# 每个 runner 返回 (passed, detail)。detail 是失败归因，通过时通常为空字符串——
# 和 L1 判分器一样，这份归因才是真正有用的产出。


def _check_file_contains(check, context):
    text = context.read_text(check["path"])
    if text is None:
        # 越界和不存在都归到这里：两种情况下断言都只能**失败**，绝不能通过。
        return False, f"file not found or outside the workspace: {check['path']}"
    if str(check["text"]) in text:
        return True, ""
    return False, f"{check['path']} does not contain {check['text']!r}"


def _check_file_not_contains(check, context):
    text = context.read_text(check["path"])
    if text is None:
        # 文件读不到时**不算**「不包含」：那会把「路径写错了」和「内容已清掉」
        # 混成同一个通过结果。
        return False, f"file not found or outside the workspace: {check['path']}"
    if str(check["text"]) not in text:
        return True, ""
    return False, f"{check['path']} still contains {check['text']!r}"


def _check_file_exists(check, context):
    path = context.resolve(check["path"])
    if path is not None and path.is_file():
        return True, ""
    return False, f"file not found or outside the workspace: {check['path']}"


def _check_report_equals(check, context):
    actual = _lookup(context.report, check["field"])
    if actual is _MISSING:
        return False, f"report field missing: {check['field']}"
    if actual == check["value"]:
        return True, ""
    return False, f"report.{check['field']} == {actual!r}, expected {check['value']!r}"


def _check_report_truthy(check, context):
    actual = _lookup(context.report, check["field"])
    if actual is _MISSING:
        return False, f"report field missing: {check['field']}"
    if actual:
        return True, ""
    return False, f"report.{check['field']} is falsy: {actual!r}"


def _check_report_length(check, context):
    actual = _lookup(context.report, check["field"])
    if actual is _MISSING:
        return False, f"report field missing: {check['field']}"
    try:
        length = len(actual)
    except TypeError:
        return False, f"report.{check['field']} has no length: {actual!r}"
    if length == int(check["length"]):
        return True, ""
    return False, f"len(report.{check['field']}) == {length}, expected {check['length']}"


def _check_trace_event(check, context):
    where = dict(check.get("where") or {})
    where["event"] = check["event"]
    if any(_event_matches(event, where) for event in context.trace_events):
        return True, ""
    return False, f"no trace event matching {where}"


# 注册表：kind -> (必填字段, runner, 用于归因的 target 字段)。
# 新增 kind 只需在这里加一行；validate_checks() 自动跟着变严。
CHECK_KINDS = {
    "file_contains": (("path", "text"), _check_file_contains, "path"),
    "file_not_contains": (("path", "text"), _check_file_not_contains, "path"),
    "file_exists": (("path",), _check_file_exists, "path"),
    "report_equals": (("field", "value"), _check_report_equals, "field"),
    "report_truthy": (("field",), _check_report_truthy, "field"),
    "report_length": (("field", "length"), _check_report_length, "field"),
    "trace_event": (("event",), _check_trace_event, "event"),
}


# ------------------------------------------------------------------ 校验


def validate_check(check, where=""):
    """校验一条 check 的结构。数据集加载时就会跑，写错不必等到跑评测才发现。"""
    if not isinstance(check, dict):
        raise ValueError(f"{where}check must be a mapping")
    kind = str(check.get("kind", "")).strip()
    if kind not in CHECK_KINDS:
        raise ValueError(f"{where}unknown check kind: {kind!r} (known: {', '.join(sorted(CHECK_KINDS))})")
    required, _, _ = CHECK_KINDS[kind]
    missing = [key for key in required if key not in check]
    if missing:
        raise ValueError(f"{where}check {kind!r} is missing required keys: {', '.join(missing)}")
    normalized = dict(check)
    normalized["kind"] = kind
    if kind in PATH_BEARING_KINDS:
        relative = str(check["path"]).strip().replace("\\", "/")
        if not relative:
            raise ValueError(f"{where}check {kind!r} has an empty path")
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(
                f"{where}check {kind!r} path must stay inside the fixture: {check['path']!r}"
            )
        normalized["path"] = relative
    return normalized


def validate_check_groups(checks, where=""):
    """校验 `{"fail_to_pass": [...], "pass_to_pass": [...]}` 这个形状。

    两组都要求**非空**：允许 P2P 为空，就等于允许作者跳过「没破坏别的东西」这半边，
    而那正是这次改动要补上的东西。
    """
    if not isinstance(checks, dict):
        raise ValueError(f"{where}checks must be a mapping")
    unknown = sorted(set(checks) - set(CHECK_GROUPS))
    if unknown:
        raise ValueError(f"{where}unknown check groups: {', '.join(unknown)}")
    normalized = {}
    for group in CHECK_GROUPS:
        items = checks.get(group)
        if not isinstance(items, list) or not items:
            raise ValueError(f"{where}checks.{group} must be a non-empty list")
        normalized[group] = [
            validate_check(item, where=f"{where}checks.{group}[{index}] ")
            for index, item in enumerate(items)
        ]
    return normalized


# ------------------------------------------------------------------ 执行


def run_check(check, context):
    _, runner, target_key = CHECK_KINDS[str(check["kind"])]
    passed, detail = runner(check, context)
    return {
        "kind": str(check["kind"]),
        "target": str(check.get(target_key, "")),
        "passed": bool(passed),
        "detail": detail,
    }


def run_checks(checks, context):
    """跑完一组 check——**不短路**。

    第一条挂掉就返回会丢掉其余归因，而归因清单才是这层判定的真正产出。
    """
    return [run_check(check, context) for check in checks or []]


def checks_passed(results):
    return all(result["passed"] for result in results)


# -------------------------------------------------------- 隐式 P2P：工作区快照


def workspace_snapshot(root, excluded_dirs=SNAPSHOT_EXCLUDED_DIRS):
    """工作区里每个文件的 sha256，按相对路径索引。

    **不跟随 symlink**（`os.walk(followlinks=False)` + 单独处理链接项）：跟随的话
    快照会去读工作区外的文件，一条指向宿主目录的链接就能让这次对比读到不该读的东西。
    symlink 记的是链接目标字符串本身——创建链接、改指向照样能被发现，但绝不解引用。
    """
    root = Path(root)
    excluded = {str(name) for name in excluded_dirs}
    snapshot = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        # 就地裁剪：被排除的目录不再往下走，symlink 目录也不进去。
        dirnames[:] = [
            name for name in dirnames
            if name not in excluded and not (current / name).is_symlink()
        ]
        for name in dirnames + filenames:
            path = current / name
            if not path.is_symlink() and name not in filenames:
                continue
            relative = path.relative_to(root)
            if excluded & set(relative.parts):
                continue
            key = relative.as_posix()
            if path.is_symlink():
                snapshot[key] = "symlink:" + str(path.readlink()).replace("\\", "/")
            elif path.is_file():
                snapshot[key] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(snapshot.items()))


def workspace_regressions(before, after, mutable_paths=()):
    """对比前后两份快照，列出 `mutable_paths` 之外的一切改动。

    新增文件也算：一次干净的运行不该在仓库里留下计划外的产物。
    """
    mutable = {str(path).replace("\\", "/") for path in mutable_paths or ()}
    regressions = []
    for path in sorted(set(before) | set(after)):
        if path in mutable:
            continue
        old = before.get(path)
        new = after.get(path)
        if old == new:
            continue
        if old is None:
            regressions.append({"path": path, "change": "created"})
        elif new is None:
            regressions.append({"path": path, "change": "deleted"})
        else:
            regressions.append({"path": path, "change": "modified"})
    return regressions
