"""把旧的聚合口径原样架在 TraceIndex 之上。

为什么存在：
新建一层索引最大的风险是「它以为自己读到的和旧代码读到的是同一份数据」。
这个模块把 `metrics.aggregate_run_artifacts()` 的输出逐字段复现一遍，
差别只在数据来源换成了 TraceIndex。配套的测试断言两者在同一批工件上
产出完全相同的 dict——这是 P0 的验收条件，也是后续把指标迁过来的凭据。

注意它复现的是**旧口径**，包括旧口径已知的偏差（cache 相关字段取自
report.json，也就是一次运行最后一轮的元数据）。修口径是 P1 的事，
P0 只负责把底座换掉而不动数字。
"""

from .trace import EVENT_PROMPT_BUILT, EVENT_TOOL_EXECUTED

# 刻意不从 metrics.py 导入这两个helper：metrics 会 import evaluator，
# 而 evaluator 现在要 import eval.harness，那就成了环。它们各自只有几行。


def _safe_mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_ratio(numerator, denominator):
    if not denominator:
        return 0.0
    return numerator / denominator


def legacy_run_aggregate(index):
    """等价于 metrics.aggregate_run_artifacts(runs_root)，但从索引取数。"""
    reports = [run.report for run in index.runs if run.report]
    tool_status_counts = {}
    tool_name_counts = {}
    security_event_counts = {}
    run_durations = []
    tool_durations = []
    prompt_durations = []
    stop_reasons = {}

    for run in index.runs:
        run_durations.append(run.duration_ms)
        for event in run.events:
            if event.get("event") == EVENT_PROMPT_BUILT and event.get("duration_ms") is not None:
                prompt_durations.append(float(event["duration_ms"]))
            if event.get("event") != EVENT_TOOL_EXECUTED:
                continue
            tool_name = str(event.get("name", "")).strip()
            if tool_name:
                tool_name_counts[tool_name] = tool_name_counts.get(tool_name, 0) + 1
            tool_status = str(event.get("tool_status", "")).strip()
            if tool_status:
                tool_status_counts[tool_status] = tool_status_counts.get(tool_status, 0) + 1
            security_event = str(event.get("security_event_type", "")).strip()
            if security_event:
                security_event_counts[security_event] = security_event_counts.get(security_event, 0) + 1
            if event.get("duration_ms") is not None:
                tool_durations.append(float(event["duration_ms"]))

    tool_steps = [int(report.get("tool_steps", 0)) for report in reports]
    attempts = [int(report.get("attempts", 0)) for report in reports]
    prompt_tokens = [int((report.get("prompt_metadata") or {}).get("prompt_tokens", 0)) for report in reports]
    cached_tokens = [int((report.get("prompt_metadata") or {}).get("cached_tokens", 0) or 0) for report in reports]
    cache_hits = [bool((report.get("prompt_metadata") or {}).get("cache_hit")) for report in reports]
    input_tokens = [int((report.get("prompt_metadata") or {}).get("input_tokens", 0) or 0) for report in reports]
    prefix_reused = [
        not bool((report.get("prompt_metadata") or {}).get("prefix_changed"))
        for report in reports
        if "prefix_changed" in (report.get("prompt_metadata") or {})
    ]
    for report in reports:
        stop_reason = str(report.get("stop_reason", "")).strip()
        if stop_reason:
            stop_reasons[stop_reason] = stop_reasons.get(stop_reason, 0) + 1

    return {
        "run_count": len(reports) if reports else len(index.runs),
        "avg_tool_steps": _safe_mean(tool_steps),
        "avg_attempts": _safe_mean(attempts),
        "avg_prompt_tokens": _safe_mean(prompt_tokens),
        "cache_hit_rate": _safe_ratio(sum(1 for hit in cache_hits if hit), len(cache_hits)),
        "cached_token_ratio": _safe_ratio(sum(cached_tokens), sum(input_tokens)),
        "avg_cached_tokens": _safe_mean(cached_tokens),
        "prefix_reuse_rate": _safe_ratio(sum(1 for reused in prefix_reused if reused), len(prefix_reused)),
        "tool_status_counts": tool_status_counts,
        "tool_name_counts": tool_name_counts,
        "security_event_counts": security_event_counts,
        "stop_reason_counts": stop_reasons,
        "avg_run_duration_ms": _safe_mean(run_durations),
        "avg_tool_duration_ms": _safe_mean(tool_durations),
        "avg_prompt_build_duration_ms": _safe_mean(prompt_durations),
    }
