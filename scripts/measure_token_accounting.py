"""量一件事:我们自己数出来的 prompt token,和后端实际计费的 input_token,差多少。

## 为什么需要这个脚本

`models.context_budget_tokens()` 里有一个余量系数。它防的是「我们数少了」——
`count_tokens()` 走 litellm 的分词器,而自建后端(mimo-v2.5 这类)不在 litellm 的
注册表里,回落到别的分词器,和后端真正用的那个不是同一个。数少了的后果是:我们以为
还有余量,实际请求已经超出窗口,后端直接 400,整轮已经做完的工具执行全部丢掉。

这个系数原先取 0.8,依据是「实测同一段中英混合文本,gpt-4o 的分词器算 2160 个、
未知模型回落分词器算 2400 个,高 11%」。但那个 11% 说的是**回落分词器数得更多**,
也就是保守方向;真正危险的方向——我们数得比后端少——从来没测过。

本脚本直接测这个差:发真实请求,把两个数配对。

## 差里面有什么

    input_tokens (后端) − prompt_tokens (我们)
        = 工具 schema            我们单独算,`tool_schema_tokens`,不进 prompt_tokens
        + 消息结构开销            role 字段、消息分隔符、tool_call 的封装,后端计费
                                  我们的压平文本里没有
        + 分词器差异              正负都可能

预算需要覆盖的是**整个差**,不是只覆盖分词器那一项——所以这里报的
`residual_ratio = (input_tokens − prompt_tokens − schema) / prompt_tokens`
就是余量系数应该取的下界。

## 怎么跑

    python scripts/measure_token_accounting.py --output-json artifacts/token-accounting.json

需要 `.env` 里的三行 provider 配置(和其它 live 脚本一样)。默认在临时目录里
建一个仿真工作区(把本仓库的 `codingforme/*.py` 复制进去并 git init),这样
prefix 里的仓库快照体量和真实使用接近,而不会污染本仓库的 `.codingforme/runs`。
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codingforme import CodingForMe, SessionStore, WorkspaceContext  # noqa: E402
from codingforme.config import load_project_env, project_root, provider_env  # noqa: E402
from codingforme.models import OpenAICompatibleModelClient  # noqa: E402

# 探针请求。刻意从短到长排,好让样本覆盖不同的 prompt 体量——差值如果随体量变化,
# 一个固定比例的余量就不够用,那是必须看出来的事。
DEFAULT_PROBES = (
    "What files are in this workspace?",
    "Read models.py and tell me what count_tokens does.",
    "Read context_manager.py and summarize how the prompt budget is enforced.",
    "Read runtime.py and list the tools the agent can call.",
    "Compare how models.py and workspace.py each clip text, and say which is authoritative.",
)


def build_probe_workspace(root: Path) -> Path:
    """造一个体量接近真实项目的临时工作区。

    用真实源码而不是几个空文件:prefix 里的仓库快照按文件路径和 git 状态生成,
    空目录会让 prefix 小一个量级,量出来的比例就不能外推到真实使用。
    """
    src = Path(__file__).resolve().parents[1] / "codingforme"
    dst = root / "codingforme"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (root / "README.md").write_text(
        "# probe workspace\n\nCopied from coding-for-me for token accounting.\n",
        encoding="utf-8",
    )
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=probe@example.com", "-c", "user.name=probe",
         "commit", "-q", "-m", "probe workspace"],
    ):
        subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            # 中文 Windows 上不指定编码会按 ANSI 代码页解码 git 输出，遇到非 ASCII
            # 直接抛 UnicodeDecodeError。同 workspace.py / evaluator.py。
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    return root


def build_model_client(timeout: int) -> OpenAICompatibleModelClient:
    load_project_env(project_root())
    api_key = provider_env("CODINGFORME_OPENAI_API_KEY", ("OPENAI_API_KEY",))
    if not api_key:
        raise SystemExit("需要 .env 里的 CODINGFORME_OPENAI_API_KEY(或 OPENAI_API_KEY)")
    return OpenAICompatibleModelClient(
        model=provider_env("CODINGFORME_OPENAI_MODEL", ("OPENAI_MODEL",), "gpt-5.4"),
        base_url=provider_env("CODINGFORME_OPENAI_API_BASE", ("OPENAI_API_BASE",), "https://api.openai.com/v1"),
        api_key=api_key,
        temperature=0.0,
        timeout=timeout,
    )


def collect_turns(runs_root: Path):
    """从 trace 里把每一轮的两个数配对。

    按 `turn` 配对而不按文件顺序:同一轮的 `prompt_built` 和 `model_parsed` 是两条
    事件,而一轮里可能有多条 `tool_executed` 夹在中间。
    """
    turns = []
    for trace_path in sorted(runs_root.glob("*/trace.jsonl")):
        by_turn = {}
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            key = (event.get("run_id"), event.get("turn"))
            slot = by_turn.setdefault(key, {})
            if event.get("event") == "prompt_built":
                metadata = event.get("prompt_metadata") or {}
                slot["prompt_tokens"] = metadata.get("prompt_tokens")
                slot["schema_tokens"] = metadata.get("tool_schema_tokens")
                slot["message_count"] = metadata.get("message_count")
                slot["tool_calls_replayed"] = metadata.get("history_tool_calls_replayed")
            elif event.get("event") == "model_parsed":
                completion = event.get("completion_metadata") or {}
                slot["input_tokens"] = completion.get("input_tokens")
                slot["cached_tokens"] = completion.get("cached_tokens")
        for (run_id, turn), slot in sorted(by_turn.items(), key=lambda item: (str(item[0][0]), item[0][1] or 0)):
            if slot.get("prompt_tokens") is None or slot.get("input_tokens") is None:
                continue
            prompt_tokens = int(slot["prompt_tokens"])
            schema_tokens = int(slot.get("schema_tokens") or 0)
            input_tokens = int(slot["input_tokens"])
            residual = input_tokens - prompt_tokens - schema_tokens
            turns.append(
                {
                    "run_id": run_id,
                    "turn": turn,
                    "prompt_tokens": prompt_tokens,
                    "schema_tokens": schema_tokens,
                    "input_tokens": input_tokens,
                    "cached_tokens": slot.get("cached_tokens"),
                    "message_count": slot.get("message_count"),
                    "tool_calls_replayed": slot.get("tool_calls_replayed"),
                    "gap": input_tokens - prompt_tokens,
                    "residual": residual,
                    # 余量系数要覆盖的就是这个比例。
                    "residual_ratio": residual / prompt_tokens if prompt_tokens else None,
                    # 如果余量的本质是「每条消息的结构开销」,这一列才该是常数,
                    # 而 residual_ratio 会随 prompt 变大而变小。哪一列稳,
                    # 就决定了余量该写成绝对量还是比例。
                    "residual_per_message": (
                        residual / slot["message_count"] if slot.get("message_count") else None
                    ),
                }
            )
    return turns


def summarize(turns):
    ratios = [t["residual_ratio"] for t in turns if t["residual_ratio"] is not None]
    prompts = [t["prompt_tokens"] for t in turns]
    per_message = [t["residual_per_message"] for t in turns if t.get("residual_per_message")]
    residuals = [t["residual"] for t in turns]
    if not ratios:
        return {"sample_size": 0}
    return {
        "residual_min": min(residuals),
        "residual_median": statistics.median(residuals),
        "residual_max": max(residuals),
        "residual_per_message_min": min(per_message) if per_message else None,
        "residual_per_message_median": statistics.median(per_message) if per_message else None,
        "residual_per_message_max": max(per_message) if per_message else None,
        "sample_size": len(ratios),
        "prompt_tokens_min": min(prompts),
        "prompt_tokens_max": max(prompts),
        "residual_ratio_min": min(ratios),
        "residual_ratio_median": statistics.median(ratios),
        "residual_ratio_max": max(ratios),
        "residual_ratio_mean": statistics.fmean(ratios),
        # 建议取的余量:最大值再留一档。报出来而不是直接写进代码——常量该由人来定。
        "suggested_margin": round(max(ratios) * 1.5, 3) if max(ratios) > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-json", default=None, help="把逐轮数据和汇总写到这里")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--context-window", default=None, help="覆盖窗口档位,例如 128k")
    args = parser.parse_args()

    model = build_model_client(args.request_timeout)
    with tempfile.TemporaryDirectory(prefix="token-accounting-") as tmp:
        root = build_probe_workspace(Path(tmp))
        workspace = WorkspaceContext.build(str(root))
        agent = CodingForMe(
            model_client=model,
            workspace=workspace,
            session_store=SessionStore(str(root / ".codingforme" / "sessions")),
            approval_policy="never",
            # 只读:探针只需要真实体量的上下文,不需要它改任何东西。
            read_only=True,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            context_window=args.context_window,
        )
        print(
            f"[probe] model={model.model} window={agent.context_window} "
            f"source={agent.context_window_source} budget={agent.context_budget}",
            file=sys.stderr,
        )
        for index, probe in enumerate(DEFAULT_PROBES, start=1):
            print(f"[probe {index}/{len(DEFAULT_PROBES)}] {probe}", file=sys.stderr)
            try:
                agent.ask(probe)
            except Exception as error:  # 一个探针失败不该丢掉已经收到的样本
                print(f"[probe {index}] failed: {error}", file=sys.stderr)
        turns = collect_turns(root / ".codingforme" / "runs")

    summary = summarize(turns)
    payload = {
        "model": model.model,
        "context_window": agent.context_window,
        "context_window_source": agent.context_window_source,
        "context_budget": agent.context_budget,
        "summary": summary,
        "turns": turns,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        destination = Path(args.output_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {destination}", file=sys.stderr)


if __name__ == "__main__":
    main()
