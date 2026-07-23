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

MAX_TOOL_OUTPUT = 4000
MAX_HISTORY = 12000
# 这些文件最可能直接影响 agent 的行动方式。
# 我们不会预加载整个仓库，只会先给模型一小份“导航包”。
DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
IGNORED_PATH_NAMES = {".git", ".codingforme", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}


def now():
    return datetime.now(timezone.utc).isoformat()


def clip(text, limit=MAX_TOOL_OUTPUT):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


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
                    check=True,
                    timeout=5,
                )
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        repo_root = (
            Path(repo_root_override).resolve()
            if repo_root_override is not None
            else Path(git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
        )
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
        tracked_output = git(["ls-files", "--cached", "--others", "--exclude-standard"], "")
        file_tree = build_file_tree(line for line in tracked_output.splitlines() if line.strip())

        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch=git(["branch", "--show-current"], "-") or "-",
            default_branch=(
                lambda branch: branch[len("origin/") :] if branch.startswith("origin/") else branch
            )(git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main"),
            status=clip(git(["status", "--short"], "clean") or "clean", 1500),
            recent_commits=[line for line in git(["log", "--oneline", "-5"]).splitlines() if line],
            project_docs=docs,
            file_tree=file_tree,
        )

    def text(self):
        # 这段文本会被塞进 prompt prefix，作为相对稳定的基线上下文。
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        docs = "\n".join(f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()) or "- none"
        tree = render_file_tree(self.file_tree)
        return textwrap.dedent(
            f"""\
            Workspace:
            - cwd: {self.cwd}
            - repo_root: {self.repo_root}
            - branch: {self.branch}
            - default_branch: {self.default_branch}
            - status:
            {self.status}
            - recent_commits:
            {commits}
            - project_docs:
            {docs}
            - project_tree:
            {tree}
            """
        ).strip()

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
