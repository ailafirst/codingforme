"""跑一次评测套件，产出统一 schema 的结果与 markdown。

两个套件：
    fixed-benchmark   L2 任务判定 + L1 轨迹断言（12 个单轮任务）
    cross-session     L3 跨会话断言（4 条多轮会话，含进程重启）

用法：
    uv run python scripts/run_eval_suite.py --suite fixed-benchmark --harness full \
        --output-json artifacts/eval-full.json \
        --output-markdown docs/metrics/eval-full.md

    uv run python scripts/run_eval_suite.py --suite cross-session --harness full \
        --output-json artifacts/eval-session-full.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codingforme.config import load_project_env, project_root, provider_env  # noqa: E402
from codingforme.eval.harness import BUILTIN_HARNESS_SPECS, get_harness  # noqa: E402
from codingforme.models import OpenAICompatibleModelClient  # noqa: E402
from codingforme.eval.report import render_eval_result_markdown  # noqa: E402
from codingforme.eval.session_suite import (  # noqa: E402
    DEFAULT_SESSION_BENCHMARK_PATH,
    run_session_suite,
)
from codingforme.eval.suite import run_benchmark_suite  # noqa: E402

SUITE_FIXED = "fixed-benchmark"
SUITE_SESSION = "cross-session"

# 固定基准跑批时 agent 的步数上限，覆盖数据集里声明的 step_budget。
#
# 为什么不用数据集的值：那里的 2~6 步是 ORACLE_SOLUTIONS 参考解的**最短路径**，
# 真实模型要探索、要验证、会试错，在那之下连终点都到不了。实测（live-model，
# 12 个任务）：用数据集原值时 11 个任务撞步数上限停，结果层通过率 1/12；放宽到 12
# 变成 4/12、跑到终点 9/12；再放宽到 16 仍是 4/12——曲线在 12 附近就平了，
# 剩下没过的那几个是在打转，不是步数不够。取 16 是在拐点之上留一点余量。
#
# 数据集里的 step_budget 保持不动：它记录的是「参考解需要几步」这个事实，
# step_budget_respected 断言仍然按它判。
DEFAULT_STEP_BUDGET = 16


def build_live_model_factory(timeout=180):
    """从 `.env` 造一个真实 provider 的客户端工厂。

    注入它就把 `execution_mode` 从 `oracle-replay` 切成 `live-model`——
    任务改由模型自己解，而不是回放 `ORACLE_SOLUTIONS` 的参考解。

    刻意放在脚本层而不是 `eval/harness.py`：模型是与 harness 正交的另一条轴,
    `HarnessSpec` 不含模型客户端是它的设计决定,不要在那边开口子。
    """
    load_project_env(project_root())
    api_key = provider_env("CODINGFORME_OPENAI_API_KEY", ("OPENAI_API_KEY",))
    if not api_key:
        raise SystemExit("live model requires CODINGFORME_OPENAI_API_KEY (or OPENAI_API_KEY) in .env")
    model = provider_env("CODINGFORME_OPENAI_MODEL", ("OPENAI_MODEL",), "gpt-5.4")
    base_url = provider_env("CODINGFORME_OPENAI_API_BASE", ("OPENAI_API_BASE",), "https://api.openai.com/v1")

    def factory(task=None, workspace=None):
        # 每个任务一个新客户端：`observed` / `last_completion_metadata` 是按实例
        # 累积的，复用同一个实例会让任务之间的观测数据互相污染。
        return OpenAICompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
            timeout=timeout,
        )

    factory.model_name = model
    factory.base_url = base_url
    return factory


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run a CFM-Eval suite and emit a unified result artifact.")
    parser.add_argument(
        "--suite",
        default=SUITE_FIXED,
        choices=(SUITE_FIXED, SUITE_SESSION),
        help="which suite to run",
    )
    parser.add_argument(
        "--harness",
        default="full",
        choices=sorted(BUILTIN_HARNESS_SPECS),
        help="harness variant under evaluation",
    )
    parser.add_argument("--benchmark", default="", help="benchmark path (defaults per suite)")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-markdown", default="")
    parser.add_argument("--workspace-root", default="", help="where to materialise evaluation workspaces")
    parser.add_argument(
        "--benchmark-artifact",
        default="",
        help="where to write the raw benchmark artifact (fixed-benchmark only)",
    )
    parser.add_argument(
        "--live-model",
        action="store_true",
        help="let a real provider (from .env) solve the tasks instead of replaying oracle solutions",
    )
    parser.add_argument(
        "--step-budget-override",
        type=int,
        default=None,
        help=(
            f"agent step cap for the fixed-benchmark suite; defaults to {DEFAULT_STEP_BUDGET}. "
            "The dataset's own step_budget (2-6) is the oracle solution's shortest path, which a "
            "real model cannot finish inside. Pass 0 to fall back to the dataset's declared value."
        ),
    )
    parser.add_argument(
        "--session-tier",
        default="core",
        choices=("core", "extended", "all"),
        help=(
            "which cross-session tier to run. core (default) is the always-on regression net; "
            "extended holds the 200/300-turn recall ladder, whose answer only changes when the "
            "memory layer changes and which costs one ask() per turn."
        ),
    )
    parser.add_argument("--request-timeout", type=int, default=180, help="per-request timeout for the live model")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "run the whole benchmark k times and report pass@1 / pass^k (fixed-benchmark only). "
            "Only meaningful against a live model: replaying oracle solutions is deterministic."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help=(
            "per-step output cap. The benchmark default (64) was tuned for oracle replay, where "
            "FakeModelClient ignores it; a reasoning model burns that entire budget on chain-of-thought "
            "and returns empty content, which the runtime reduces to retry. Live runs want >=1024."
        ),
    )
    return parser


def _force_utf8_stdout():
    """把 stdout/stderr 切成 UTF-8，否则中文 Windows 控制台会在最后一步崩掉。

    报告里有 ⚠、—、中文这些非 ASCII 字符，而中文 Windows 控制台默认是 GBK
    代码页：`print(render_eval_result_markdown(result))` 会抛
    `UnicodeEncodeError: 'gbk' codec can't encode character '\\u26a0'`。

    坑在于**这时候 JSON 与 markdown 两份产物其实已经落盘了**，崩的只是终端
    渲染那一步——于是看起来像"整轮评测失败"，实际数据完好；反过来如果没注意到
    traceback，又会以为报告没生成。跑一次要一个多小时，这个误判代价很大。

    和 `CLAUDE.md` 里那条"`subprocess.run` 一律显式带 `encoding='utf-8'`"是
    同一类问题的两面：那条管**读**子进程输出，这条管**写**自己的输出。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            # stdout 被替换成了不支持 reconfigure 的对象（测试里常见），
            # 或底层不是文本流。不是致命问题，让它按原编码走。
            pass


def main(argv=None):
    _force_utf8_stdout()
    args = build_arg_parser().parse_args(argv)
    harness = get_harness(args.harness)
    result_path = Path(args.output_json) if args.output_json else None
    markdown_path = Path(args.output_markdown) if args.output_markdown else None
    workspace_root = Path(args.workspace_root) if args.workspace_root else None

    model_client_factory = build_live_model_factory(timeout=args.request_timeout) if args.live_model else None
    if model_client_factory is not None:
        print(f"[live-model] {model_client_factory.model_name} @ {model_client_factory.base_url}", file=sys.stderr)

    if args.suite == SUITE_SESSION:
        if args.step_budget_override is not None:
            raise SystemExit("--step-budget-override only applies to the fixed-benchmark suite")
        if args.repeats > 1:
            raise SystemExit("--repeats only applies to the fixed-benchmark suite")
        result = run_session_suite(
            harness=harness,
            benchmark_path=Path(args.benchmark) if args.benchmark else DEFAULT_SESSION_BENCHMARK_PATH,
            workspace_root=workspace_root,
            result_path=result_path,
            markdown_path=markdown_path,
            model_client_factory=model_client_factory,
            tiers=None if args.session_tier == "all" else (args.session_tier,),
        )
    else:
        if args.session_tier != "core":
            raise SystemExit("--session-tier only applies to the cross-session suite")
        result = run_benchmark_suite(
            harness=harness,
            benchmark_path=Path(args.benchmark) if args.benchmark else Path("benchmarks/coding_tasks.json"),
            artifact_path=Path(args.benchmark_artifact) if args.benchmark_artifact else None,
            workspace_root=workspace_root,
            result_path=result_path,
            markdown_path=markdown_path,
            model_client_factory=model_client_factory,
            # 不传 -> 用 DEFAULT_STEP_BUDGET；显式传 0 -> 回落到数据集声明的 step_budget。
            step_budget_override=(
                DEFAULT_STEP_BUDGET if args.step_budget_override is None else args.step_budget_override
            ) or None,
            max_new_tokens=args.max_new_tokens or None,
            repeats=args.repeats,
        )
    print(render_eval_result_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
