"""量一件事：在给定档位下，哪个工具的输出**够得着**落盘门槛。

为什么存在：
一次 15 任务的 live 跑批里 `spilled_calls` 是 0，一开始被当成「基准文件太小」。
真跑了探针才发现不止如此——1M 档的单条上限是 14,791 token，而除 `read_file`
之外的每个工具都有**自己的入口上限**，落在这个门槛之下：`search` 的 rg 调用带
`--max-count 200`，纯 Python 回退也在 200 条命中处返回，实测封顶 12,699 token
（门槛的 86%），永远差一点点。也就是说 1M 档的落盘只有一条路径能触发：
`read_file` 读一个 45 KB 以上的范围。

这个脚本把这件事测出来，而不是靠读代码推断——两个入口上限分别写在
`tools.py`（200 条命中）和 `context_manager.py`（`total_budget // 8`），
改任何一个都会让上面那句话失效。

用法：

    python scripts/measure_tool_output_ceiling.py --window 1m
    python scripts/measure_tool_output_ceiling.py --window 8k --json artifacts/tool-ceiling-8k.json
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codingforme import CodingForMe, FakeModelClient, SessionStore, WorkspaceContext  # noqa: E402
from codingforme.models import count_tokens, parse_window_tokens  # noqa: E402


def build_worst_case(root):
    """刻意造出每个工具各自能产生的**最大**输出。

    日志 4000 行、树下 300 个文件、每个文件都含 `latency_ms` —— search 的命中数
    远超它自己的 200 条上限，list_files 的树也远超一份普通仓库。
    """
    (root / "logs").mkdir()
    (root / "logs" / "server.log").write_text(
        "\n".join(
            "2026-08-30T04:%02d:%02dZ INFO worker=%02d request=%06d latency_ms=%03d status=200 route=/api/v1/items"
            % (index % 60, (index * 7) % 60, index % 16, index, (index * 13) % 900)
            for index in range(1, 4001)
        ),
        encoding="utf-8",
    )
    (root / "src").mkdir()
    for index in range(300):
        (root / "src" / ("mod_%03d.py" % index)).write_text(
            "# module %d\nlatency_ms = %d\n" % (index, index), encoding="utf-8"
        )
    (root / "README.md").write_text("demo\n", encoding="utf-8")


PROBES = (
    ("read_file 整份 4000 行日志", "read_file", {"path": "logs/server.log", "start": 1, "end": 4000}),
    ("search latency_ms（命中数远超 200 条上限）", "search", {"pattern": "latency_ms", "path": "."}),
    ("list_files 300 个文件的树", "list_files", {"path": "."}),
    ("search 输出成裸路径", "search", {"pattern": "latency_ms", "path": ".", "format": "paths"}),
)


def measure(window_tokens):
    root = Path(tempfile.mkdtemp(prefix="cfm-ceiling-"))
    build_worst_case(root)
    agent = CodingForMe(
        model_client=FakeModelClient([]),
        workspace=WorkspaceContext.build(root),
        session_store=SessionStore(root / ".codingforme" / "sessions"),
        approval_policy="auto",
    )
    agent.set_context_window(window_tokens)
    limit = agent.tool_output_limit()
    rows = []
    for label, name, args in PROBES:
        # 直接调底层 runner 再手动过一次落盘判定：这里要的是「原始输出有多大」，
        # 而 `run_tool()` 拿到的已经是定形之后的那一份。
        raw = agent.tools[name]["run"](args)
        kept, spill = agent._store_tool_output(name, raw)
        rows.append({
            "label": label,
            "tool": name,
            "raw_tokens": count_tokens(raw, None),
            "kept_tokens": count_tokens(kept, None),
            "spilled": bool(spill),
        })
    return {
        "context_window_tokens": window_tokens,
        "total_budget_tokens": agent.context_manager.total_budget,
        "tool_output_limit": limit,
        "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", default="1m", help="上下文窗口，如 8k / 128k / 1m")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    payload = measure(parse_window_tokens(args.window))
    print("窗口 %d → 预算 %d → 单条上限 %d" % (
        payload["context_window_tokens"], payload["total_budget_tokens"], payload["tool_output_limit"]))
    for row in payload["rows"]:
        print("%-42s raw=%7d  kept=%6d  落盘=%s" % (
            row["label"], row["raw_tokens"], row["kept_tokens"], "是" if row["spilled"] else "否"))
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote %s" % target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
