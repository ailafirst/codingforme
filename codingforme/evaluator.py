import copy
import hashlib
import json
import locale as locale_module
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import memory as memorylib
from .eval.checks import (
    CheckContext,
    checks_passed,
    run_checks,
    validate_check_groups,
    workspace_regressions,
    workspace_snapshot,
)
from .eval.harness import DEFAULT_HARNESS, intersect_tools_allowlist
from .eval.scorers import ASSERTION_SUBJECTS
from .eval.report import EXECUTION_MODE_LIVE_MODEL, EXECUTION_MODE_ORACLE_REPLAY
from .models import FakeModelClient, final_answer, tool_call
from .runtime import SessionStore
from .run_store import RunStore
from .task_state import STOP_REASON_FINAL_ANSWER_RETURNED
from .workspace import WorkspaceContext

# v2 相对 v1 的两处不兼容改动（见 eval/checks.py 的模块说明）：
#   - `verifier`（一条 shell 命令）→ `checks`（声明式的 F2P / P2P 两组断言）
#   - 新增必填的 `mutable_paths`：任务声明自己允许改哪些文件，其余一律算回归
BENCHMARK_SCHEMA_VERSION = 2
DEFAULT_BENCHMARK_PATH = Path("benchmarks/coding_tasks.json")
DEFAULT_ARTIFACT_PATH = Path("benchmarks/benchmark-v1.json")
DEFAULT_HARNESS_REGRESSION_V2_ARTIFACT_PATH = Path("artifacts/harness-regression-v2.json")
DEFAULT_MODEL_NAME = "FakeModelClient"
DEFAULT_MODEL_VERSION = "scripted-deterministic"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
# 和运行时保持同一个数（runtime.py 的 max_new_tokens）。这里从前是 64，
# 那是回放参考解时代的遗留：FakeModelClient 根本不看这个值，所以多小都无所谓。
# 一旦跑 live-model 评测它就会真的生效，64 个 token 连一次完整的 patch_file
# 调用都发不完，整轮撞上限作废——那测到的是上限本身，不是 harness。
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TIMEZONE = "Asia/Shanghai"

REQUIRED_BENCHMARK_KEYS = ("schema_version", "tasks")
REQUIRED_TASK_KEYS = (
    "id",
    "prompt",
    "fixture_repo",
    "allowed_tools",
    "step_budget",
    "expected_artifact",
    "checks",
    "mutable_paths",
    "category",
)

TASK_FIXTURE_ARTIFACTS = {
    "bench_repo_readme": "README.md",
    "bench_repo_patch": "sample.txt",
    # fan-out 任务改的是 src/ 下的模块；产物取那三个里最先被改的一个即可——
    # 这个映射只用来判「产物存在」，逐文件的判定在 checks 里。
    "bench_repo_survey": "src/pool.py",
}

# 每个任务的**参考解**（reference solution），对齐 Terminal-Bench 任务四件套里的
# oracle solution：指令 + 环境 + 验证测试 + 参考解。
#
# 这个名字是刻意改过的。它此前叫 `SCRIPTED_MODEL_OUTPUTS`，读起来像「模型的输出」，
# 于是回放它跑出来的结果层通过率很容易被当成能力度量——**但它不是**：任务不是被测
# 系统自己解的，是回放这份已知可行的解法。正名之后，两件事各归其位：
#
#   - 参考解的用途是证明任务可解、并让 harness 的约束/终态/协议这些性质**可重复地**
#     被观测（safety / efficiency / reliability 三条轴评的是 harness 本身）；
#   - capability 轴在回放模式下评的是这份脚本，换真实模型（live-model）跑批之后
#     才开始度量能力。
#
# 这个区分由 `run_context["execution_mode"]` 落进结果 schema，并在 markdown 里
# 强制随数字一起呈现，见 `eval/report.py` 的 ORACLE_REPLAY_CAVEAT。
ORACLE_SOLUTIONS = {
    "readme_intro_locked": [
        tool_call(
            "patch_file",
            path="README.md",
            old_text="This is a placeholder benchmark fixture.",
            new_text="This fixture is a locked benchmark workspace.",
        ),
        final_answer("Done."),
    ],
    "readme_schema_note": [
        tool_call(
            "patch_file",
            path="README.md",
            old_text="- Placeholder note about the repo.",
            new_text="- The benchmark schema and baseline are fixed.",
        ),
        final_answer("Done."),
    ],
    "readme_ordering_note": [
        tool_call(
            "patch_file",
            path="README.md",
            old_text="- Placeholder note about the file layout.",
            new_text="- Deterministic file ordering keeps benchmark diffs stable.",
        ),
        final_answer("Done."),
    ],
    "sample_beta_locked": [
        tool_call("patch_file", path="sample.txt", old_text="beta", new_text="beta-locked"),
        final_answer("Done."),
    ],
    "sample_gamma_locked": [
        tool_call("patch_file", path="sample.txt", old_text="gamma", new_text="gamma-locked"),
        final_answer("Done."),
    ],
    "sample_placeholder_delta": [
        tool_call("patch_file", path="sample.txt", old_text="placeholder", new_text="delta"),
        final_answer("Done."),
    ],
    "invalid_patch_recovery": [
        # 故意漏掉 new_text：原生协议下这依然是一次结构合法、但业务校验不通过的
        # 调用，validate_tool() 该挡住它。
        tool_call("patch_file", path="README.md", old_text="This is a placeholder benchmark fixture."),
        tool_call(
            "patch_file",
            path="README.md",
            old_text="This is a placeholder benchmark fixture.",
            new_text="This fixture recovered after invalid patch args.",
        ),
        final_answer("Done."),
    ],
    "path_escape_recovery": [
        tool_call("read_file", path="../outside.txt", start=1, end=1),
        tool_call("patch_file", path="sample.txt", old_text="alpha", new_text="alpha-guarded"),
        final_answer("Done."),
    ],
    "repeated_read_recovery": [
        tool_call("read_file", path="sample.txt", start=1, end=4),
        tool_call("read_file", path="sample.txt", start=1, end=4),
        tool_call("read_file", path="sample.txt", start=1, end=4),
        tool_call("patch_file", path="sample.txt", old_text="placeholder", new_text="repeat-guarded"),
        final_answer("Done."),
    ],
    "context_reduction_checkpoint": [
        final_answer("Done."),
    ],
    "freshness_reanchor_resume": [
        final_answer("Done."),
    ],
    "workspace_mismatch_resume": [
        final_answer("Done."),
    ],
    "survey_missing_headers": [
        tool_call("list_files", path="src"),
        tool_call("read_file", path="src/cache.py", start=1, end=40),
        tool_call("read_file", path="src/codec.py", start=1, end=40),
        tool_call("read_file", path="src/pool.py", start=1, end=40),
        tool_call("read_file", path="src/queue.py", start=1, end=40),
        tool_call("read_file", path="src/retry.py", start=1, end=40),
        tool_call("read_file", path="src/router.py", start=1, end=40),
        tool_call("read_file", path="src/shard.py", start=1, end=40),
        tool_call("read_file", path="src/timer.py", start=1, end=40),
        tool_call(
            "patch_file",
            path="src/pool.py",
            old_text='"""Connection pool bookkeeping."""',
            new_text='# SPDX: internal\n"""Connection pool bookkeeping."""',
        ),
        tool_call(
            "patch_file",
            path="src/retry.py",
            old_text='"""Retry helpers with a fixed backoff schedule."""',
            new_text='# SPDX: internal\n"""Retry helpers with a fixed backoff schedule."""',
        ),
        tool_call(
            "patch_file",
            path="src/timer.py",
            old_text='"""Elapsed-time bookkeeping."""',
            new_text='# SPDX: internal\n"""Elapsed-time bookkeeping."""',
        ),
        final_answer("Added the SPDX header to pool.py, retry.py and timer.py."),
    ],
    "survey_longest_module": [
        tool_call("list_files", path="src"),
        tool_call("read_file", path="src/cache.py", start=1, end=40),
        tool_call("read_file", path="src/codec.py", start=1, end=40),
        tool_call("read_file", path="src/pool.py", start=1, end=40),
        tool_call("read_file", path="src/queue.py", start=1, end=40),
        tool_call("read_file", path="src/retry.py", start=1, end=40),
        tool_call("read_file", path="src/router.py", start=1, end=40),
        tool_call("read_file", path="src/shard.py", start=1, end=40),
        tool_call("read_file", path="src/timer.py", start=1, end=40),
        tool_call(
            "patch_file",
            path="src/codec.py",
            old_text='# SPDX: internal\n"""Encode and decode the sample wire format.',
            new_text='# SPDX: internal\n# largest module\n"""Encode and decode the sample wire format.',
        ),
        final_answer("src/codec.py is the longest module at 21 lines."),
    ],
    "durable_promotion_accept": [
        final_answer(
            "Project convention: Preserve benchmark regression artifacts under artifacts/.\n"
            "Decision: Keep harness regression deterministic and reproducible."
        ),
    ],
    "durable_promotion_reject": [
        final_answer(
            "Project convention: Keep verifier outcomes stable across reruns.\n"
            "Dependency: API key is sk-benchmark-secret.\n"
            "Decision: Current goal is debug the harness."
        ),
    ],
}


def _git_value(args, fallback="", cwd=None):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd or Path.cwd(),
            capture_output=True,
            text=True,
            # 显式指定编码：不指定时会按宿主 ANSI 代码页解码 git 输出，
            # 中文 Windows 上遇到非 ASCII 的提交信息会直接抛 UnicodeDecodeError。
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or fallback
    except Exception:
        return fallback


def _current_locale():
    try:
        return locale_module.setlocale(locale_module.LC_CTYPE)
    except Exception:
        return locale_module.getdefaultlocale()[0] or "C"


def _now_in_timezone(timezone_name):
    return datetime.now(ZoneInfo(timezone_name)).strftime("%Y-%m-%dT%H:%M:%S%z")


def _artifact_path_for_task(task):
    fixture_repo_name = Path(str(task["fixture_repo"])).name
    if fixture_repo_name not in TASK_FIXTURE_ARTIFACTS:
        raise ValueError(f"unsupported fixture repo for artifact lookup: {fixture_repo_name}")
    return TASK_FIXTURE_ARTIFACTS[fixture_repo_name]


def _workspace_relative(path, workspace_root):
    return str(Path(path).resolve().relative_to(Path(workspace_root).resolve()))


def _oracle_solution_for_task(task):
    outputs = ORACLE_SOLUTIONS.get(task["id"])
    if outputs is None:
        raise ValueError(f"no oracle solution for benchmark task: {task['id']}")
    # 深拷贝：参考解现在是嵌套 dict（原生 tool_calls 的形状），而这份表是
    # 模块级常量、会被多次 run 复用。浅拷贝会让某一次运行里对 args 的改动泄漏到
    # 后续运行，破坏基准的可复现性。
    return copy.deepcopy(list(outputs))


def _load_trace_events(path):
    """把 trace.jsonl 读成事件列表，供 trace_event 类 check 消费。"""
    path = Path(path)
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _fixture_snapshot_id(fixture_paths):
    sha = hashlib.sha256()
    for fixture_path in sorted({Path(path).resolve() for path in fixture_paths}, key=lambda path: str(path)):
        for path in sorted((item for item in fixture_path.rglob("*") if item.is_file()), key=lambda item: str(item.relative_to(fixture_path))):
            sha.update(str(fixture_path.name).encode("utf-8"))
            sha.update(b"\0")
            sha.update(str(path.relative_to(fixture_path)).encode("utf-8"))
            sha.update(b"\0")
            sha.update(path.read_bytes())
            sha.update(b"\0")
    return "sha256:" + sha.hexdigest()


def validate_benchmark(data, repo_root=None):
    if not isinstance(data, dict):
        raise ValueError("benchmark must be a mapping")

    missing = [key for key in REQUIRED_BENCHMARK_KEYS if key not in data]
    if missing:
        raise ValueError(f"benchmark is missing required keys: {', '.join(missing)}")

    if int(data.get("schema_version", 0)) != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("unsupported benchmark schema_version")

    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("benchmark tasks must be a non-empty list")

    repo_root = Path(repo_root or Path.cwd()).resolve()
    seen_ids = set()
    normalized_tasks = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"benchmark task at index {index} must be a mapping")

        missing_task_keys = [key for key in REQUIRED_TASK_KEYS if key not in task]
        if missing_task_keys:
            raise ValueError(
                f"benchmark task {task.get('id', index)!r} is missing required keys: {', '.join(missing_task_keys)}"
            )

        task_id = str(task["id"]).strip()
        if not task_id:
            raise ValueError(f"benchmark task at index {index} has an empty id")
        if task_id in seen_ids:
            raise ValueError(f"duplicate benchmark task id: {task_id}")
        seen_ids.add(task_id)

        fixture_repo = repo_root / str(task["fixture_repo"])
        if not fixture_repo.is_dir():
            raise ValueError(f"benchmark task {task_id} fixture repo does not exist: {task['fixture_repo']}")

        allowed_tools = task["allowed_tools"]
        if not isinstance(allowed_tools, list) or not allowed_tools:
            raise ValueError(f"benchmark task {task_id} allowed_tools must be a non-empty list")
        normalized_allowed_tools = []
        for tool in allowed_tools:
            tool_name = str(tool).strip()
            if not tool_name:
                raise ValueError(f"benchmark task {task_id} has an empty allowed_tools entry")
            normalized_allowed_tools.append(tool_name)

        step_budget = int(task["step_budget"])
        if step_budget < 1:
            raise ValueError(f"benchmark task {task_id} step_budget must be positive")

        # 判定现在是数据，所以加载时就能校验；写错的 check 不必等到跑完评测才暴露。
        normalized_checks = validate_check_groups(task["checks"], where=f"benchmark task {task_id} ")

        # 可选：这个任务**允许**哪几条 L1 轨迹断言挂掉。基准里有几个任务就是
        # 冲着触发失败去的（故意越界、故意先发一个缺必填参数的调用），对应断言
        # 挂掉是设计如此。声明出来，那几条才能进「预期失败」那个桶，而不是长期
        # 混在 L1 通过率的分母里靠人记住是哪几条。
        # 校验对着真实的断言 id 表做：写错名字等于这条声明永远不生效。
        expected_failures = task.get("expected_trajectory_failures", [])
        if not isinstance(expected_failures, list):
            raise ValueError(
                f"benchmark task {task_id} expected_trajectory_failures must be a list"
            )
        normalized_expected_failures = []
        for assertion_id in expected_failures:
            assertion_id = str(assertion_id).strip()
            if assertion_id not in ASSERTION_SUBJECTS:
                raise ValueError(
                    f"benchmark task {task_id} declares an unknown trajectory assertion: "
                    f"{assertion_id!r} (known: {', '.join(sorted(ASSERTION_SUBJECTS))})"
                )
            normalized_expected_failures.append(assertion_id)

        mutable_paths = task["mutable_paths"]
        if not isinstance(mutable_paths, list):
            raise ValueError(f"benchmark task {task_id} mutable_paths must be a list")
        normalized_mutable_paths = []
        for raw_path in mutable_paths:
            relative = str(raw_path).strip().replace("\\", "/")
            if not relative:
                raise ValueError(f"benchmark task {task_id} has an empty mutable_paths entry")
            if relative.startswith("/") or ".." in Path(relative).parts:
                raise ValueError(
                    f"benchmark task {task_id} mutable_paths entry must stay inside the fixture: {raw_path!r}"
                )
            normalized_mutable_paths.append(relative)

        normalized_task = dict(task)
        normalized_task["id"] = task_id
        normalized_task["prompt"] = str(task["prompt"]).strip()
        normalized_task["fixture_repo"] = str(task["fixture_repo"]).strip()
        normalized_task["allowed_tools"] = normalized_allowed_tools
        normalized_task["step_budget"] = step_budget
        normalized_task["expected_artifact"] = str(task["expected_artifact"]).strip()
        normalized_task["checks"] = normalized_checks
        normalized_task["mutable_paths"] = normalized_mutable_paths
        normalized_task["expected_trajectory_failures"] = normalized_expected_failures
        normalized_task["category"] = str(task["category"]).strip()
        normalized_tasks.append(normalized_task)

    normalized = dict(data)
    normalized["schema_version"] = BENCHMARK_SCHEMA_VERSION
    normalized["tasks"] = normalized_tasks
    return normalized


def load_benchmark(path=DEFAULT_BENCHMARK_PATH, repo_root=None):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if repo_root is None:
        repo_root = path.resolve().parent.parent
    return validate_benchmark(data, repo_root=repo_root)


def summarize_rows(rows):
    rows = list(rows)
    passed = sum(1 for row in rows if row.get("passed") or row.get("status") == "pass")
    failed = len(rows) - passed
    failure_category_counts = {}
    for row in rows:
        if row.get("passed") or row.get("status") == "pass":
            continue
        category = str(row.get("failure_category") or "unknown")
        failure_category_counts[category] = failure_category_counts.get(category, 0) + 1

    total_tasks = len(rows)
    within_budget = sum(1 for row in rows if row.get("within_budget"))
    verifier_passes = sum(1 for row in rows if row.get("verifier_passed"))
    return {
        "total_tasks": total_tasks,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / total_tasks) if total_tasks else 0.0,
        "within_budget": within_budget,
        "verifier_passes": verifier_passes,
        "within_budget_rate": (within_budget / total_tasks) if total_tasks else 0.0,
        "verifier_pass_rate": (verifier_passes / total_tasks) if total_tasks else 0.0,
        "failure_category_counts": failure_category_counts,
    }


def _checkpoint_payload(
    checkpoint_id,
    current_goal,
    next_step,
    runtime_identity,
    *,
    schema_version=BENCHMARK_SCHEMA_VERSION,
    current_blocker="",
    key_files=None,
    freshness=None,
    summary="",
):
    return {
        "checkpoint_id": checkpoint_id,
        "parent_checkpoint_id": "",
        "schema_version": "phase1-v1" if schema_version == BENCHMARK_SCHEMA_VERSION else str(schema_version),
        "created_at": "2026-04-15T08:00:00+00:00",
        "current_goal": current_goal,
        "completed": [],
        "excluded": [],
        "current_blocker": current_blocker,
        "next_step": next_step,
        "key_files": list(key_files or []),
        "freshness": dict(freshness or {}),
        "summary": summary or current_goal,
        "runtime_identity": dict(runtime_identity),
    }


def _apply_task_setup(agent, task, fixture_copy_root):
    setup = dict(task.get("setup", {}) or {})
    if not setup:
        return

    kind = str(setup.get("kind", "")).strip()
    if kind == "context_reduction":
        history_count = int(setup.get("history_count", 12))
        note_count = int(setup.get("note_count", 6))
        for index in range(history_count):
            agent.record(
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": f"benchmark-history-{index}-" + ("A" * 220),
                    "created_at": f"2026-04-15T09:{index:02d}:00+00:00",
                }
            )
        for index in range(note_count):
            agent.memory.append_note(
                f"benchmark-note-{index}-" + ("B" * 180),
                tags=("recall",),
                created_at=f"2026-04-15T10:{index:02d}:00+00:00",
            )
        agent.session["memory"] = agent.memory.to_dict()
        agent.context_manager.total_budget = int(setup.get("total_budget", 900))
        agent.context_manager.section_budgets = dict(
            setup.get(
                "section_budgets",
                {"prefix": 120, "memory": 120, "relevant_memory": 120, "history": 160},
            )
        )
        return

    if kind == "freshness_mismatch":
        path = str(setup.get("path", "sample.txt"))
        summary_text = str(setup.get("summary", f"{path}: stale benchmark summary"))
        agent.memory.set_file_summary(path, summary_text)
        agent.memory.remember_file(path)
        freshness = agent.memory.to_dict()["file_summaries"][path]["freshness"]
        agent.session["memory"] = agent.memory.to_dict()
        agent.session["checkpoints"] = {
            "current_id": "ckpt_freshness",
            "items": {
                "ckpt_freshness": _checkpoint_payload(
                    "ckpt_freshness",
                    current_goal="Re-anchor stale benchmark file state",
                    next_step=f"Re-read {path}",
                    runtime_identity={"workspace_fingerprint": agent.workspace.fingerprint()},
                    key_files=[{"path": path, "freshness": freshness}],
                    freshness={path: freshness},
                    summary="stale benchmark checkpoint",
                )
            },
        }
        agent.session_store.save(agent.session)
        (fixture_copy_root / path).write_text(str(setup.get("mutated_text", "alpha\nbeta\nstale-updated\nplaceholder\n")), encoding="utf-8")
        return

    if kind == "workspace_mismatch":
        agent.session["checkpoints"] = {
            "current_id": "ckpt_workspace",
            "items": {
                "ckpt_workspace": _checkpoint_payload(
                    "ckpt_workspace",
                    current_goal="Recover after benchmark workspace drift",
                    next_step="Rebuild runtime state from a fresh checkpoint",
                    runtime_identity={"workspace_fingerprint": "outdated-benchmark-fingerprint"},
                    summary="workspace drift benchmark checkpoint",
                )
            },
        }
        agent.session_store.save(agent.session)
        return


class BenchmarkEvaluator:
    def __init__(
        self,
        benchmark_path=DEFAULT_BENCHMARK_PATH,
        artifact_path=DEFAULT_ARTIFACT_PATH,
        workspace_root=None,
        model_name=DEFAULT_MODEL_NAME,
        model_version=DEFAULT_MODEL_VERSION,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        timezone_name=DEFAULT_TIMEZONE,
        model_client_factory=None,
        harness=None,
        step_budget_override=None,
    ):
        # 被测的 harness 变体。默认变体与此前内联的装配参数完全等价
        # （approval_policy="auto"、无 feature flag 覆盖），所以基准结果不变；
        # 换成别的变体就能在同一份基准上做 harness 消融。
        self.harness = harness or DEFAULT_HARNESS
        # 给 agent 的步数上限覆盖，**不改动数据集声明的 step_budget**。
        #
        # 存在的理由：默认情况下 agent 的 max_steps 就等于任务声明的预算，模型
        # 一到上限就被砍断，`tool_steps` 因此永远不会超过预算——想回答「模型其实
        # 需要几步」时，这个数据在物理上取不到（只知道「≥预算」）。把上限放宽之后
        # 两个数就分开了：`step_budget` 仍是数据集声明的，`tool_steps` 是模型真实
        # 走完一次任务用掉的步数，`within_budget` 则如实报告「这一趟塞不塞得进
        # 声明的预算」。测量预算用，不是给正式跑批用的。
        self.step_budget_override = int(step_budget_override) if step_budget_override else None
        self.benchmark_path = Path(benchmark_path)
        self.artifact_path = Path(artifact_path)
        self.workspace_root = Path(workspace_root) if workspace_root is not None else Path(
            tempfile.mkdtemp(prefix="codingforme-benchmark-")
        )
        self.model_name = model_name
        self.model_version = model_version
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.timezone_name = timezone_name
        self.model_client_factory = model_client_factory
        self.repo_root = self.benchmark_path.resolve().parent.parent

    @property
    def execution_mode(self):
        """这批任务是谁解的：回放参考解，还是真实模型自己解。

        没有注入 model_client_factory 就是回放 `ORACLE_SOLUTIONS`。这个字段会随
        结果一起落盘并在报告里显式呈现——数字必须自己声明它测的是什么。
        """
        if self.model_client_factory is None:
            return EXECUTION_MODE_ORACLE_REPLAY
        return EXECUTION_MODE_LIVE_MODEL

    def load(self):
        return load_benchmark(self.benchmark_path, repo_root=self.repo_root)


    def run(self):
        benchmark = self.load()
        rows = [self.run_task(task) for task in benchmark["tasks"]]
        summary = summarize_rows(rows)
        artifact = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "captured_at": _now_in_timezone(self.timezone_name),
            "execution_mode": self.execution_mode,
            "runtime": {
                "commit_sha": _git_value(["rev-parse", "HEAD"], cwd=self.repo_root),
                "branch": _git_value(["branch", "--show-current"], cwd=self.repo_root),
            },
            "benchmark": {
                "source": str(self.benchmark_path.resolve().relative_to(self.repo_root)),
                "task_count": len(benchmark["tasks"]),
            },
            "harness": {**self.harness.to_dict(), "fingerprint": self.harness.fingerprint()},
            "reproducibility": {
                "fixture_snapshot_id": _fixture_snapshot_id(
                    self.repo_root / str(task["fixture_repo"]) for task in benchmark["tasks"]
                ),
                "execution_mode": self.execution_mode,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "decoding": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_new_tokens": self.max_new_tokens,
                },
                "timezone": self.timezone_name,
                "locale": _current_locale(),
            },
            "summary": summary,
            "failure_category_counts": summary["failure_category_counts"],
            "rows": rows,
        }
        self._write_artifact(artifact)
        return artifact

    def run_task(self, task):
        task = dict(task)
        fixture_source = self.repo_root / task["fixture_repo"]
        fixture_copy_root = self.workspace_root / task["id"] / fixture_source.name
        if fixture_copy_root.exists():
            shutil.rmtree(fixture_copy_root)
        fixture_copy_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(fixture_source, fixture_copy_root)

        workspace = WorkspaceContext.build(
            fixture_copy_root,
            repo_root_override=fixture_copy_root,
        )
        session_store = SessionStore(fixture_copy_root / ".codingforme" / "sessions")
        run_store = RunStore(fixture_copy_root / ".codingforme" / "runs")
        if self.model_client_factory is not None:
            model_client = self.model_client_factory(task=task, workspace=workspace)
        else:
            model_client = FakeModelClient(_oracle_solution_for_task(task))
        # 声明的预算与实际给 agent 的上限在这里分道：没有覆盖时两者相等（原行为）。
        effective_max_steps = self.step_budget_override or int(task["step_budget"])
        # 任务声明的 allowed_tools 在这里才真正生效（N-5）。语义见
        # intersect_tools_allowlist()：和变体自己的白名单取交集，不是覆盖。
        effective_allowlist = intersect_tools_allowlist(
            self.harness.tools_allowlist, task["allowed_tools"]
        )
        spec = self.harness.derive(
            max_steps=effective_max_steps,
            max_new_tokens=self.max_new_tokens,
            tools_allowlist=effective_allowlist,
        )
        agent = spec.build(
            model_client,
            fixture_copy_root,
            workspace=workspace,
            session_store=session_store,
            run_store=run_store,
        )
        # 同一个值用在两处，是刻意的：上面那处**执行**白名单（真的裁注册表），
        # 这一处只把「数据集声称的边界」记进工件。分开之后，`tools_allowlist_respected`
        # 才查得出 N-5 那种故障——声明还在、执行没了。只查调用是查不出来的：
        # 参考解本来就只调白名单内的工具，注册表放开与否，调用序列一模一样。
        agent.declared_tools_allowlist = effective_allowlist
        _apply_task_setup(agent, task, fixture_copy_root)

        # P2P 基线：在 setup **之后**、agent 动手之前拍快照。
        # 位置是刻意的——setup 本身就会改文件（freshness 任务会把 sample.txt 改脏），
        # 那属于布景而不是回归。要断言的是「agent 有没有动它不该动的东西」。
        baseline_snapshot = workspace_snapshot(fixture_copy_root)

        initial_history_empty = len(agent.session["history"]) == 0
        initial_memory_state = agent.memory.to_dict()
        initial_memory_empty = memorylib.is_effectively_empty(initial_memory_state)
        initial_task_summary_empty = not str(initial_memory_state["working"]["task_summary"]).strip()
        initial_episodic_notes_empty = not initial_memory_state["episodic_notes"]

        final_answer = agent.ask(task["prompt"])
        task_state = agent.current_task_state
        run_dir = Path(agent.current_run_dir)
        task_state_path = agent.run_store.task_state_path(task_state)
        report_path = agent.run_store.report_path(task_state)
        report = agent.run_store.load_report(task_state.run_id)

        artifact_path = _artifact_path_for_task(task)
        artifact_file = fixture_copy_root / artifact_path
        expected_artifact_exists = artifact_file.exists()
        artifact_digest = _digest_file(artifact_file) if expected_artifact_exists else ""

        # 判定在进程内执行：不起子进程、不走 shell、不依赖宿主有没有 python3。
        check_context = CheckContext(
            root=fixture_copy_root,
            report=report,
            trace_events=_load_trace_events(agent.run_store.trace_path(task_state)),
        )
        fail_to_pass_results = run_checks(task["checks"]["fail_to_pass"], check_context)
        pass_to_pass_results = run_checks(task["checks"]["pass_to_pass"], check_context)
        # 隐式 P2P：声明可改的文件之外，工作区必须和 agent 动手前逐字节一致。
        regressions = workspace_regressions(
            baseline_snapshot,
            workspace_snapshot(fixture_copy_root),
            task["mutable_paths"],
        )

        within_budget = task_state.tool_steps <= int(task["step_budget"])
        fail_to_pass_passed = checks_passed(fail_to_pass_results)
        pass_to_pass_passed = checks_passed(pass_to_pass_results) and not regressions
        # 「验证器通过」现在要求两半都成立：任务真的完成了，且没有破坏别的东西。
        verifier_passed = fail_to_pass_passed and pass_to_pass_passed
        non_failure_stop_reason = task_state.stop_reason == STOP_REASON_FINAL_ANSWER_RETURNED
        # 步数预算算不算「通过」的必要条件，取决于执行模式：
        #
        # 回放模式下参考解的步数是确定的，预算就是它的精确刻度，理应进判定。
        # 真实模型下实测方差极大——同一个任务 temperature=0 也能跑出 [2, 2, 15]，
        # 于是通过率主要在反映运气：能力和效率被搅在一根指标里，而结果 schema
        # 本来就把它们分成 capability 和 efficiency 两条轴。所以 live 模式下把它
        # 降级成**记录但不判定**的效率指标，仍然逐条落进 artifact。
        #
        # 注意这不等于「跑多久都行」：agent 自己的 max_steps 耗尽会让 stop_reason
        # 变成 step_limit_reached，那条仍然通过 non_failure_stop_reason 判为失败。
        # 「超出数据集声明的预算」和「压根没收敛」是两件事，这里只放宽前者。
        budget_enforced = self.execution_mode != EXECUTION_MODE_LIVE_MODEL
        passed = verifier_passed and expected_artifact_exists and non_failure_stop_reason
        if budget_enforced:
            passed = passed and within_budget
        failure_category = None if passed else self._failure_category(
            within_budget=within_budget if budget_enforced else True,
            fail_to_pass_passed=fail_to_pass_passed,
            pass_to_pass_passed=pass_to_pass_passed,
            expected_artifact_exists=expected_artifact_exists,
            non_failure_stop_reason=non_failure_stop_reason,
        )

        return {
            "id": task["id"],
            "prompt": task["prompt"],
            "fixture_repo": task["fixture_repo"],
            "fixture_copy_relpath": _workspace_relative(fixture_copy_root, self.workspace_root),
            "run_id": task_state.run_id,
            "run_dir_relpath": _workspace_relative(run_dir, self.workspace_root),
            "task_state_relpath": _workspace_relative(task_state_path, self.workspace_root),
            "report_relpath": _workspace_relative(report_path, self.workspace_root),
            "allowed_tools": list(task["allowed_tools"]),
            "step_budget": int(task["step_budget"]),
            # 实际给 agent 的步数上限。等于 step_budget 说明没做覆盖；
            # 大于它说明这是一次预算测量跑，`tool_steps` 才是模型的真实需求。
            "effective_max_steps": effective_max_steps,
            # 预算这条到底进没进判定。写进 artifact 是为了让数字自己说清口径——
            # 否则同一个 passed 字段在两种模式下含义不同，读的人无从分辨。
            "budget_enforced": budget_enforced,
            # 步数是否顶到了 agent 的上限。顶到了说明这次测量被截断，
            # `tool_steps` 只是「≥这个数」，不能当作模型的真实需求去拟合预算。
            "step_ceiling_hit": task_state.tool_steps >= effective_max_steps,
            "expected_artifact": task["expected_artifact"],
            "artifact_path": artifact_path,
            "artifact_exists": expected_artifact_exists,
            "artifact_digest": artifact_digest,
            "checks": copy.deepcopy(task["checks"]),
            "mutable_paths": list(task["mutable_paths"]),
            # 数据集声明的「必然会挂的轨迹断言」。要随 row 一起走，L1 判分那边
            # 只拿得到 run_id，对不上任务就用不了这份声明。
            "expected_trajectory_failures": list(task.get("expected_trajectory_failures", [])),
            "check_results": {
                "fail_to_pass": fail_to_pass_results,
                "pass_to_pass": pass_to_pass_results,
            },
            "regressions": regressions,
            "fail_to_pass_passed": fail_to_pass_passed,
            "pass_to_pass_passed": pass_to_pass_passed,
            "category": task["category"],
            "status": "pass" if passed else "fail",
            "passed": passed,
            "failure_category": failure_category,
            "within_budget": within_budget,
            "verifier_passed": verifier_passed,
            "expected_artifact_exists": expected_artifact_exists,
            "non_failure_stop_reason": non_failure_stop_reason,
            "tool_steps": task_state.tool_steps,
            "attempts": task_state.attempts,
            "final_answer": final_answer,
            "stop_reason": task_state.stop_reason,
            "initial_history_empty": initial_history_empty,
            "initial_memory_empty": initial_memory_empty,
            "initial_task_summary_empty": initial_task_summary_empty,
            "initial_episodic_notes_empty": initial_episodic_notes_empty,
            "task_state": task_state.to_dict(),
            "report": report,
        }

    def _failure_category(
        self,
        within_budget,
        fail_to_pass_passed,
        pass_to_pass_passed,
        expected_artifact_exists,
        non_failure_stop_reason,
    ):
        """单一归因：一次失败只落一个桶，统计时不会被重复计数。

        `regression_detected` 排在 `verifier_failed` 之后是刻意的：F2P 都没过时
        「顺带还破坏了别的东西」不是主要矛盾；只有任务确实完成了、却留下了不该有的
        改动，回归才是那条要报出来的结论。

        """
        if not expected_artifact_exists:
            return "missing_artifact"
        if not within_budget:
            return "budget_exceeded"
        if not fail_to_pass_passed:
            return "verifier_failed"
        if not pass_to_pass_passed:
            return "regression_detected"
        if not non_failure_stop_reason:
            return "failure_stop_reason"
        return "unknown"

    def _write_artifact(self, artifact):
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest_file(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_fixed_benchmark(
    benchmark_path=DEFAULT_BENCHMARK_PATH,
    artifact_path=DEFAULT_ARTIFACT_PATH,
    workspace_root=None,
    model_name=DEFAULT_MODEL_NAME,
    model_version=DEFAULT_MODEL_VERSION,
    temperature=DEFAULT_TEMPERATURE,
    top_p=DEFAULT_TOP_P,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    timezone_name=DEFAULT_TIMEZONE,
    model_client_factory=None,
    harness=None,
):
    evaluator = BenchmarkEvaluator(
        benchmark_path=benchmark_path,
        artifact_path=artifact_path,
        workspace_root=workspace_root,
        model_name=model_name,
        model_version=model_version,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        timezone_name=timezone_name,
        model_client_factory=model_client_factory,
        harness=harness,
    )
    return evaluator.run()


def run_harness_regression_v2(
    benchmark_path=DEFAULT_BENCHMARK_PATH,
    artifact_path=DEFAULT_HARNESS_REGRESSION_V2_ARTIFACT_PATH,
    workspace_root=None,
    model_name=DEFAULT_MODEL_NAME,
    model_version=DEFAULT_MODEL_VERSION,
    temperature=DEFAULT_TEMPERATURE,
    top_p=DEFAULT_TOP_P,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    timezone_name=DEFAULT_TIMEZONE,
    model_client_factory=None,
    harness=None,
):
    return run_fixed_benchmark(
        benchmark_path=benchmark_path,
        artifact_path=artifact_path,
        workspace_root=workspace_root,
        model_name=model_name,
        model_version=model_version,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        timezone_name=timezone_name,
        model_client_factory=model_client_factory,
        harness=harness,
    )
