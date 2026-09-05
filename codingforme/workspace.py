"""工作区快照工具。

这个模块负责在 agent 按需读文件之前，先给它一份便宜的“仓库第一印象”。
这份快照刻意保持小而稳定：主要包含 Git 事实和少量白名单项目文档。
"""

import subprocess
import textwrap
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

# 这两个上限的单位是 **token**，和上下文预算同一种单位。
#
# 它们限的都是「进模型上下文的量」——工具输出会原样回给模型，压平的历史文本同理——
# 所以必须和 `ContextManager.total_budget` 用同一把尺子量。曾经它们是字符数，于是
# 同一条链路上摆着两种单位：预算按 token 判，而喂进预算的那些文本按字符裁，谁也
# 换算不到谁。
#
# 数值由原先的字符值按**这条链路自己实际装的内容**实测出的字符/token 比折算（统一
# 折半是错的，比值随内容类型差 2.4 倍，见 `context_manager.py` 顶部那张表）：工具
# 输出主要是源码，实测 3.02，4000 ÷ 3.02 ≈ 1320；压平历史是中文请求与工具输出混合，
# 实测 2.07，12000 ÷ 2.07 ≈ 5800。
# **这是下限，不是上限。** 真正生效的是 `context_manager.tool_output_limit()`
# 从 `total_budget` 派生出来的值，本常量只在小档位（或 context_manager 尚未装配时）
# 兜底。写死一个绝对值与预算脱钩正是阶段二从各段配额里删掉的那个病。
MAX_TOOL_OUTPUT = 1320
MAX_HISTORY = 5800
# 整份仓库快照的上限。它是**入口上限**，不是组装时的额度：`context_manager` 那边
# 各段已经没有固定额度了，快照能进多少由这里决定，进去之后一个 token 都不再裁。
#
# 这个位置和 Claude Code 一致——它组装时不分配额度，限制加在读进来的那一刻
# （auto memory 取前 200 行或 25KB，Codex 的 `project_doc_max_bytes` 是 32 KiB）。
#
# 8000 是**兜底**，正常仓库碰不到：实测本仓库整份快照 3058 token，其中
# project_docs 2562（README/AGENTS/pyproject，每份各自已被 clip 到 1200）、
# status 200、recent_commits 129、project_tree 只有 104。最坏情况是 4 份 doc 都满
# （4×1200）加 status 满（1500），约 6500，仍在这条线以下。它防的是 status 或某份
# doc 异常膨胀把整个 prefix 撑爆，而 prefix 现在是不裁的——没有这条线，一个
# `git status` 刷出几千行的仓库会直接把预算吃光。
MAX_SNAPSHOT_TOKENS = 8000
# 这些文件最可能直接影响 agent 的行动方式。
# 我们不会预加载整个仓库，只会先给模型一小份“导航包”。
DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
IGNORED_PATH_NAMES = {".git", ".codingforme", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}


def now():
    return datetime.now(timezone.utc).isoformat()


def clip(text, limit=MAX_TOOL_OUTPUT, model=None):
    """把文本裁到 `limit` 个 **token** 以内。

    单位是 token，因为它裁的东西最终都要进模型上下文（工具输出、压平的历史），
    而上下文预算是按 token 判的。两边用同一把尺子，才谈得上「这段占了预算的多少」。
    实现委托给 `models.clip_tokens()`，全系统只有那一个截断原语。
    """
    from .models import clip_tokens

    text = str(text)
    return clip_tokens(text, limit, model, marker=f"\n...[truncated to {int(limit)} tokens]")


def middle(text, limit):
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]


def build_file_tree(paths):
    # 目录 + 扩展名聚合：根目录文件原样列出；子目录不逐个列出文件名，
    # 只汇总文件总数和扩展名分布。体积由“顶层目录数 x 扩展名多样性”
    # 决定，不随文件总数线性增长。文件名级别的细节交给 list_files/search
    # 工具按需查看（实测见 docs/architecture/repo-snapshot-optimization.md）。
    root_files = []
    dirs = {}
    for raw in paths:
        path = str(raw).strip().replace("\\", "/")
        if not path:
            continue
        parts = path.split("/", 1)
        if len(parts) == 1:
            if parts[0] not in IGNORED_PATH_NAMES:
                root_files.append(parts[0])
            continue
        top, rest = parts
        if top in IGNORED_PATH_NAMES:
            continue
        entry = dirs.setdefault(top, {"count": 0, "ext_counts": {}})
        entry["count"] += 1
        ext = Path(rest).suffix or "noext"
        entry["ext_counts"][ext] = entry["ext_counts"].get(ext, 0) + 1
    return {"root_files": sorted(root_files), "dirs": dirs}


def render_file_tree(file_tree):
    lines = [f"- {name}" for name in file_tree.get("root_files", [])]
    for name, info in sorted(file_tree.get("dirs", {}).items()):
        ext_items = sorted(info["ext_counts"].items(), key=lambda kv: -kv[1])
        shown = [f"{count} {ext}" for ext, count in ext_items[:5]]
        rest = sum(count for _, count in ext_items[5:])
        if rest:
            shown.append(f"+{rest} more")
        lines.append(f"- {name}/ ({info['count']} files: {', '.join(shown)})")
    return "\n".join(lines) or "- (empty)"


def _walk_relative_paths(root, limit=400):
    """把工作区里的文件列出来，跳过 `IGNORED_PATH_NAMES` 里那些噪声目录。

    只在「这个工作区不是它所在 git 仓库的顶层」时才用得上——那时 git 索引讲的是
    别人的事。限制条数是为了别让一个大目录把 prefix 撑爆。
    """
    root = Path(root)
    found = []
    for path in sorted(root.rglob("*")):
        if len(found) >= limit:
            break
        if not path.is_file():
            continue
        if any(part in IGNORED_PATH_NAMES for part in path.relative_to(root).parts):
            continue
        found.append(str(path.relative_to(root)).replace("\\", "/"))
    return found


class WorkspaceContext:
    def __init__(self, cwd, repo_root, branch, default_branch, status, recent_commits, project_docs, file_tree):
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.default_branch = default_branch
        self.status = status
        self.recent_commits = recent_commits
        self.project_docs = project_docs
        self.file_tree = file_tree

    @classmethod
    def build(cls, cwd, repo_root_override=None):
        cwd = Path(cwd).resolve()

        def git(args, fallback=""):
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    # 显式指定编码：不指定时按宿主 ANSI 代码页解码 git 输出，
                    # 中文 Windows 上遇到非 ASCII 的提交信息会抛 UnicodeDecodeError。
                    encoding="utf-8",
                    errors="replace",
                    check=True,
                    timeout=5,
                )
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        git_toplevel = Path(git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
        repo_root = Path(repo_root_override).resolve() if repo_root_override is not None else git_toplevel
        # 调用方显式指定了根、而这个根又不是 git 顶层时（评测把每个任务的样板仓库
        # 复制到某处、再限定在那份拷贝上，而那个位置恰好落在某个 git 仓库里面），
        # git 报出来的分支、状态、提交记录讲的是**外层仓库**的事，不是这个工作区的。
        # 原样塞进快照有两个害处：一是给模型一堆与任务无关的文件名（实测把 prefix
        # 从约 3400 撑到约 7400 字符，超过段预算后样板仓库自己的快照反而被截掉），
        # 二是把外层仓库的文件名泄露给一个本不该看到它们的 agent。
        foreign_git = repo_root != git_toplevel
        docs = {}
        # 同时扫描 repo_root 和 cwd，这样在子目录启动时也能看到本地文档；
        # 但用相对路径做 key，避免同一份文档被重复收集。
        for base in (repo_root, cwd):
            for name in DOC_NAMES:
                path = base / name
                if not path.exists():
                    continue
                key = str(path.relative_to(repo_root))
                if key in docs:
                    continue
                docs[key] = clip(path.read_text(encoding="utf-8", errors="replace"), 1200)

        # tracked + 未被 .gitignore 忽略的 untracked 文件；不走文件系统 rglob，
        # 直接吃 git 索引，天然遵守 .gitignore。
        # 同理：外层仓库的 git 索引列的是外层的文件，而且这份拷贝多半整个被
        # .gitignore 掉了（于是列出来是空的）。这种情况下退回文件系统遍历，
        # 至少让模型看得见自己工作区里真实有哪些文件。
        if foreign_git:
            file_tree = build_file_tree(_walk_relative_paths(repo_root))
        else:
            tracked_output = git(["ls-files", "--cached", "--others", "--exclude-standard"], "")
            file_tree = build_file_tree(line for line in tracked_output.splitlines() if line.strip())

        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch="-" if foreign_git else (git(["branch", "--show-current"], "-") or "-"),
            default_branch="-" if foreign_git else (
                lambda branch: branch[len("origin/") :] if branch.startswith("origin/") else branch
            )(git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main"),
            status="clean" if foreign_git else clip(git(["status", "--short"], "clean") or "clean", 1500),
            recent_commits=[] if foreign_git else [line for line in git(["log", "--oneline", "-5"]).splitlines() if line],
            project_docs=docs,
            file_tree=file_tree,
        )

    def text(self):
        # 这段文本会被塞进 prompt prefix，作为相对稳定的基线上下文。
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        docs = "\n".join(f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()) or "- none"
        tree = render_file_tree(self.file_tree)
        # 多行插值一律走占位符，替换放在 dedent **之后**。
        # 直接写 {self.status} / {commits} / {docs} / {tree} 会让 dedent 失效：
        # f-string 是先插值再 dedent 的，而这几个值都是多行且行首无空格，一展开
        # 所有行的公共缩进就变成 0，dedent 于是什么都不做——整段 Workspace 带着
        # 12 个空格的缩进发给模型（实测 15 行里有 14 行中招）。这和 runtime.py
        # 的 build_prefix() 是同一个坑，两处修法保持一致。
        template = textwrap.dedent(
            """\
            Workspace:
            - cwd: __CWD__
            - repo_root: __REPO_ROOT__
            - branch: __BRANCH__
            - default_branch: __DEFAULT_BRANCH__
            - status:
            __STATUS__
            - recent_commits:
            __COMMITS__
            - project_docs:
            __DOCS__
            - project_tree:
            __TREE__
            """
        )
        for placeholder, value in (
            ("__CWD__", self.cwd),
            ("__REPO_ROOT__", self.repo_root),
            ("__BRANCH__", self.branch),
            ("__DEFAULT_BRANCH__", self.default_branch),
            ("__STATUS__", self.status),
            ("__COMMITS__", commits),
            ("__DOCS__", docs),
            ("__TREE__", tree),
        ):
            template = template.replace(placeholder, str(value))
        # 入口上限。clip() 保留开头，所以裁掉的是 project_tree 那一段（模板里排最后），
        # 而 cwd / branch / status / recent_commits 这些定位信息一定活着。
        return clip(template.strip(), MAX_SNAPSHOT_TOKENS)

    def fingerprint(self):
        # 这个指纹用来判断仓库状态是否发生了足够大的变化，
        # 从而决定是否需要重建缓存中的 prompt prefix。
        payload = {
            "cwd": self.cwd,
            "repo_root": self.repo_root,
            "branch": self.branch,
            "default_branch": self.default_branch,
            "status": self.status,
            "recent_commits": list(self.recent_commits),
            "project_docs": dict(self.project_docs),
            "file_tree": self.file_tree,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
