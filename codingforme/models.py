"""模型后端适配层。

runtime 只关心一件事：给我一个 prompt（和可选的工具 schema），我拿回一个结构化结果
（文本 + 可能的原生 tool_calls）。不同 provider 在 HTTP 接口、响应结构、是否支持
function-calling、是否支持 prompt cache 上都有差异，这些差异都在这里被抹平成统一的
complete() 接口——具体传输经由 litellm 完成，但已知会返回非标准响应体的后端
（声明 stream:false 却仍返回 SSE；返回 Responses-API 风格的 output_text 而非
标准 choices）绕开 litellm 自带的严格 schema 校验，走 CustomLLM 桥接自行解析。
"""

import json
import time
from http.client import RemoteDisconnected
import urllib.error
import urllib.request

import litellm
from litellm import CustomLLM
from litellm.types.utils import ChatCompletionMessageToolCall, Function

OPENAI_COMPATIBLE_USER_AGENT = "coding-for-me/0.1"


def tool_call(name, **args):
    """脚本化一次原生工具调用，形状和真实后端返回的完全一致。

    这是测试/评测里表达"模型这一轮决定调某个工具"的标准写法。以前写成
    `'<tool>{"name":"read_file","args":{...}}</tool>'` 这种字符串，等于让测试
    去验证一条生产环境永远不走的文本解析路径——真实后端走的是 tool_calls 分支。
    """
    return {"text": "", "tool_calls": [{"name": str(name), "args": dict(args)}]}


def final_answer(text):
    """脚本化一次最终答案：有文本、没有工具调用。

    原生协议下"结束"的信号就是这个形状，不需要 <final> 标签来标记。
    """
    return {"text": str(text), "tool_calls": None}


def to_messages(messages):
    """把 `complete()` 的第一个参数归一成标准 messages 数组。

    字符串是**便利形式**而不是第二套协议：线上永远发数组，这里只是让直接调
    `complete("...")` 的测试和脚本不必手工包一层。
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return [dict(message) for message in messages]


def flatten_messages(messages):
    """messages 数组 → 纯文本视图。

    只服务于观测与断言（FakeModelClient 的 `prompts`、trace 里的可读快照），
    **不用于发请求**。工具调用渲染成 `[tool:name] {args}`，和 `ContextManager`
    压平历史时的写法一致，这样"某段文字在不在上下文里"这类断言换了载体也照旧成立。
    """
    if isinstance(messages, str):
        return messages
    blocks = []
    for message in messages:
        content = str(message.get("content") or "")
        if content:
            blocks.append(content)
        for call in message.get("tool_calls") or []:
            function = call.get("function", {}) or {}
            blocks.append(f"[tool:{function.get('name', '')}] {function.get('arguments', '')}")
    return "\n\n".join(blocks)


class FakeModelClient:
    def __init__(self, outputs, supports_native_tool_calls=True):
        self.outputs = list(outputs)
        self.prompts = []
        self.messages = []
        self.supports_prompt_cache = False
        # 默认按原生 function-calling 走，和真实后端一致——脚本化输出用
        # tool_call()/final_answer() 构造。传 False 只表示"这个后端不吃 tools="，
        # 是个纯传输开关：prompt 教什么协议已经不再跟着它变。
        self.supports_native_tool_calls = bool(supports_native_tool_calls)
        self.pending_tool_choice = None
        self.last_completion_metadata = {}

    def complete(self, messages, max_new_tokens, **kwargs):
        # 两份记录各有用处：`messages` 是真正发出去的结构，断言"这一轮以 assistant
        # 身份重放了哪几个 tool_call"只能查它；`prompts` 是压平的文本视图，
        # 断言"某段文字在不在上下文里"用它更直接，也让既有用例不必全部重写。
        self.messages.append(to_messages(messages))
        self.prompts.append(flatten_messages(messages))
        if not getattr(self, "last_completion_metadata", None):
            self.last_completion_metadata = {}
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)



def _normalize_versioned_base_url(base_url):
    base = str(base_url).rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


# 后端能力是"声明"出来的，不是从 URL 猜出来的。
#
# 之前这里是一句 `any(host in base_url for host in ("openai.com", "right.codes"))`，
# 问题有二：一是加 provider 就要改这行代码；二是 substring 猜测会同时猜错两个方向
# ——实测在用的 xiaomimimo 后端不在白名单里，于是 supports_prompt_cache=False，
# 但它每次都返回 cached_tokens 非零（有自动前缀缓存），字段名严重误导。
#
# 现在分三层，后面的覆盖前面的：默认值 → 已知后端表 → 显式传入/环境变量。
# 未知后端拿到的是保守默认值，而不是一个碰巧匹配上的猜测。
DEFAULT_CAPABILITIES = {
    # 后端能吃标准 function-calling 的 tools= 数组。默认 True：这是 OpenAI-compatible
    # 的既定标准，不支持的后端会走 parse() 的文本标签兜底，代价可控。
    "native_tool_calls": True,
    # 后端认 prompt_cache_key / prompt_cache_retention 这两个自定义字段。
    # 默认 False：发一个后端不认的伪参数没有收益，只有风险。
    "prompt_cache_key": False,
}

KNOWN_BACKEND_CAPABILITIES = (
    ("openai.com", {"prompt_cache_key": True}),
    ("right.codes", {"prompt_cache_key": True}),
)

CAPABILITY_NAMES = tuple(DEFAULT_CAPABILITIES)


def force_tool_choice(name):
    """构造 OpenAI 的 tool_choice：强制这一次调用必须命中指定工具。

    安全评测需要让**真实模型**确定性地发起某一次危险调用（越权读、只读态写盘
    等），才能验证平台闸口挡不挡得住。以前的做法是在 prompt 里塞一段
    `<tool>{...}</tool>` 让模型照抄，那既依赖模型的复读意愿，又和"绝不要把工具
    调用写成文本"的协议说明直接冲突。改用 tool_choice 之后，"调哪个工具"由协议
    保证，prompt 只需要用自然语言说清参数。
    """
    return {"type": "function", "function": {"name": str(name)}}


def resolve_capabilities(base_url, overrides=None):
    """把"默认 → 已知后端 → 显式覆盖"三层合成一份能力声明。

    `overrides` 里值为 None 的键表示"没有意见"，会让位给前两层；
    这样 CLI 可以无脑把没配的环境变量传成 None，不用自己判断。
    """
    capabilities = dict(DEFAULT_CAPABILITIES)
    for fragment, known in KNOWN_BACKEND_CAPABILITIES:
        if fragment in str(base_url):
            capabilities.update(known)
    for name, value in (overrides or {}).items():
        if name not in DEFAULT_CAPABILITIES:
            raise ValueError(f"unknown model capability: {name}")
        if value is not None:
            capabilities[name] = bool(value)
    return capabilities


def _extract_openai_text(data):
    if data.get("output_text"):
        return data["output_text"]

    for item in data.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict):
                text = content.get("text")
                if text:
                    return text

    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        return text

    return ""


def _extract_openai_text_from_sse(body_text):
    last_response = None
    deltas = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type", "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
            continue
        if event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                return text
        part = event.get("part")
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text:
                return text
        item = event.get("item")
        if isinstance(item, dict):
            text = _extract_openai_text({"output": [item]})
            if text:
                return text
        response = event.get("response")
        if isinstance(response, dict):
            last_response = response
            text = _extract_openai_text(response)
            if text:
                return text
        text = _extract_openai_text(event)
        if text:
            return text
    if deltas:
        return "".join(deltas)
    if isinstance(last_response, dict):
        return _extract_openai_text(last_response)
    return ""


def _extract_openai_response_from_sse(body_text):
    last_response = None
    deltas = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        response = event.get("response")
        if isinstance(response, dict):
            last_response = response
            if event.get("type") == "response.completed":
                text = _extract_openai_text(response)
                if text:
                    return text, response
        event_type = event.get("type", "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
        elif event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                return text, last_response or {}
        else:
            text = _extract_openai_text(event)
            if text:
                return text, event
    if deltas:
        return "".join(deltas), last_response or {}
    if isinstance(last_response, dict):
        return _extract_openai_text(last_response), last_response
    return "", {}


def _extract_usage_cache_details(data):
    # 把不同 OpenAI-compatible 返回里的 usage 字段整理成统一结构，
    # 让 runtime/trace/report 不需要关心 provider 细节。
    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached_tokens = int(input_details.get("cached_tokens") or 0)
    # 推理模型把思维链的开销记在 completion_tokens_details.reasoning_tokens 里，
    # 且它**已经计入** output_tokens。不单独取出来会有两个后果：
    #   1. 成本被严重低估——实测 MiMo 一次简单收尾 115 个输出 token 里有 92 个是思维链；
    #   2. 看不出 max_new_tokens 是被思维链吃光的，只会看到"content 莫名其妙是空的"。
    output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    reasoning_tokens = int(output_details.get("reasoning_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": cached_tokens,
        "cache_hit": cached_tokens > 0,
    }


# 网关把**自己这一侧的连接失败**当成 4xx 报出来时，响应体里会留下这些痕迹。
# 按状态码判定的话它是"客户端请求有问题"（不可重试），按语义则是一次典型的
# 上游抖动（应当重试）。踩过的坑：一次 2.4 小时的 k=3 跑批在第三轮第 10 个任务
# 上收到 `HTTP 400 {"message":"Request failed","param":"finishConnect(..) failed:
# Connection refused: ...10.137.1.77:80"}` —— 那个 IP 是**服务端内网地址**，
# 和我们发的请求没有关系。当时按 400 直接放弃，整轮跑批连同已完成的部分一起丢了。
_TRANSPORT_FAILURE_MARKERS = (
    "connection refused",
    "connection reset",
    "connection timed out",
    "connect timed out",
    "finishconnect",
    "no route to host",
    "broken pipe",
    "upstream connect error",
)


def _is_transport_failure_body(body):
    """这个 4xx 是不是网关在替上游报连接失败。

    刻意只认连接类痕迹，**不放宽成"所有 400 都重试"**：真正的请求格式错误
    （我们自己的 bug）重试三次只会白烧三个约 17 秒的往返，还把错误现场推迟。
    """
    lowered = str(body).lower()
    return any(marker in lowered for marker in _TRANSPORT_FAILURE_MARKERS)


def _send_with_retry(request, timeout, model, attempts=3):
    """发起请求并对 5xx / 连接类错误做退避重试，返回 (body_text, content_type)。

    这段重试逻辑和"这个后端返回的到底是什么形状"无关，因此从
    `_CompatBackendCustomLLM.completion()` 里拆出来单独复用。
    """
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body_text = response.read().decode("utf-8")
                headers = getattr(response, "headers", {}) or {}
                content_type = headers.get("Content-Type", "")
            return body_text, content_type
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            # 4xx 里混着的连接类失败与 5xx 同类处理，但退避拉长：上游抖动恢复
            # 通常以秒计，0.5 秒重试大概率撞在同一次故障窗口里。
            transport_failure = exc.code < 500 and _is_transport_failure_body(body)
            if (exc.code >= 500 or transport_failure) and attempt < attempts - 1:
                time.sleep((2.0 if transport_failure else 0.5) * (attempt + 1))
                continue
            raise RuntimeError(f"OpenAI-compatible request failed with HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, RemoteDisconnected) as exc:
            if attempt < attempts - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise RuntimeError(
                "Could not reach the OpenAI-compatible backend.\n"
                f"Base URL: {request.full_url}\n"
                f"Model: {model}"
            ) from exc


def _usage_dict_from_object(usage_obj):
    # usage_obj 可能是 litellm 的 Usage pydantic 对象（非流式 / 流式最终 chunk），
    # 也可能是 CustomLLM.streaming() 里自己拼的普通 dict——这里统一成一份
    # {"prompt_tokens","completion_tokens","total_tokens","cached_tokens"}。
    if usage_obj is None:
        return {}
    if isinstance(usage_obj, dict):
        get = usage_obj.get
        details = usage_obj.get("prompt_tokens_details")
        out_details = usage_obj.get("completion_tokens_details")
    else:
        get = lambda key, default=None: getattr(usage_obj, key, default)  # noqa: E731
        details = getattr(usage_obj, "prompt_tokens_details", None)
        out_details = getattr(usage_obj, "completion_tokens_details", None)
    cached_tokens = 0
    if details is not None:
        cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    # 思维链的开销：它**已经计入** completion_tokens，但不单独取出来就看不见。
    # 见 _extract_usage_cache_details 里同一处注释。
    reasoning_tokens = 0
    if out_details is not None:
        reasoning_tokens = (
            out_details.get("reasoning_tokens") if isinstance(out_details, dict)
            else getattr(out_details, "reasoning_tokens", None)
        )
    return {
        "prompt_tokens": get("prompt_tokens"),
        "completion_tokens": get("completion_tokens"),
        "reasoning_tokens": int(reasoning_tokens or 0),
        "total_tokens": get("total_tokens"),
        "cached_tokens": int(cached_tokens or 0),
    }


def _tool_calls_from_message(message):
    # message 是解析出来的原始 JSON dict（标准 chat-completions 形状下的
    # choices[0].message），这里把它里面的 tool_calls 翻成 litellm 认识的类型，
    # 好让上层统一用 response.choices[0].message.tool_calls 读取。
    raw_calls = (message or {}).get("tool_calls")
    if not raw_calls:
        return None
    calls = []
    for call in raw_calls:
        function = (call or {}).get("function") or {}
        calls.append(
            ChatCompletionMessageToolCall(
                id=str(call.get("id") or ""),
                type="function",
                function=Function(
                    name=str(function.get("name") or ""),
                    arguments=str(function.get("arguments") or ""),
                ),
            )
        )
    return calls


class _CompatBackendCustomLLM(CustomLLM):
    """把这个项目已知的怪异 OpenAI-compatible 响应形状包进 litellm 的 provider 接口。

    litellm 内建的 openai 传输对响应体做严格 schema 校验，遇到过两种真实会
    报错的形状：(1) 后端声明 stream:false 但仍返回 SSE；(2) 后端返回
    `{"output_text":...}` 而不是标准的 `{"choices":[...]}`。这里自己发请求、
    自己解析（沿用原来 `_extract_openai_text`/`_extract_openai_response_from_sse`
    这套兼容逻辑），解析完再塞回 litellm 的 `ModelResponse`，这样上层依然可以
    统一用 `litellm.completion()` 调用，且能拿到统一的 tool_calls/usage 对象。
    """

    def completion(self, model, messages, api_base, api_key, timeout, optional_params, model_response, **kwargs):
        payload = {"model": model, "messages": messages, "stream": False}
        for key in ("temperature", "max_tokens", "tools", "tool_choice"):
            if optional_params.get(key) is not None:
                payload[key] = optional_params[key]
        extra_body = optional_params.get("extra_body") or {}
        payload.update(extra_body)

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": OPENAI_COMPATIBLE_USER_AGENT,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            api_base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        body_text, content_type = _send_with_retry(request, timeout or 300, model)

        text = None
        if content_type.startswith("text/event-stream") or body_text.lstrip().startswith("data:"):
            text, response_data = _extract_openai_response_from_sse(body_text)
            data = response_data if isinstance(response_data, dict) else {}
        else:
            try:
                data = json.loads(body_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "OpenAI-compatible error: backend returned non-JSON content that could not be parsed"
                ) from exc
            if data.get("error"):
                raise RuntimeError(f"OpenAI-compatible error: {data['error']}")

        choices = data.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        has_message = bool(choices) and isinstance(choices[0].get("message"), dict)
        if text is None:
            if isinstance(message.get("content"), str):
                text = message["content"]
            elif has_message:
                # 推理模型可能把全部输出放进 reasoning_content 而让 content 缺席，
                # 或者在推理阶段就被 max_tokens 截断（finish_reason:"length"）。
                # 响应结构本身是合法的，只是这一轮没有可用输出——交给 parse()
                # 走 retry，让模型下一轮重来，而不是抛异常把整个 ask() 打断。
                text = ""
            else:
                text = _extract_openai_text(data)
        if not text and not message.get("tool_calls") and not has_message:
            raise RuntimeError("OpenAI-compatible error: could not extract text from response")

        model_response.choices[0].message.content = text or ""
        tool_calls = _tool_calls_from_message(message)
        if tool_calls:
            model_response.choices[0].message.tool_calls = tool_calls

        usage = _extract_usage_cache_details(data)
        if usage.get("input_tokens") is not None:
            model_response.usage = litellm.Usage(
                prompt_tokens=usage["input_tokens"] or 0,
                completion_tokens=usage.get("output_tokens") or 0,
                total_tokens=usage.get("total_tokens") or 0,
                prompt_tokens_details={"cached_tokens": usage.get("cached_tokens") or 0},
                # 思维链开销要一路带到 last_completion_metadata。漏掉这一项的后果不是
                # 少个字段，而是成本被系统性低估——它是 completion_tokens 的一部分，
                # 实测能占到输出的八成。
                completion_tokens_details={"reasoning_tokens": usage.get("reasoning_tokens") or 0},
            )
        return model_response

    def streaming(self, model, messages, api_base, api_key, timeout, optional_params, model_response, **kwargs):
        """真正逐块读取后端的 SSE 流，实测这个项目实际对接的后端在 `stream:true`
        下用的是标准 `choices[0].delta.content`/`.tool_calls` 增量格式,所以这里
        按标准格式解析；如果一整条流读完什么都没解析出来（说明遇到了某个非标准
        怪癖，比如 Responses 风格的事件），退回 `completion()` 那套宽容解析逻辑，
        一次性把结果当成"一个 chunk"回放给调用方——不堆两套怪癖兼容逻辑。
        """
        payload = {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}}
        for key in ("temperature", "max_tokens", "tools", "tool_choice"):
            if optional_params.get(key) is not None:
                payload[key] = optional_params[key]
        extra_body = optional_params.get("extra_body") or {}
        payload.update(extra_body)

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": OPENAI_COMPATIBLE_USER_AGENT,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            api_base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        saw_any_event = False
        tool_fragments = {}
        finish_reason = ""
        usage_block = None
        with urllib.request.urlopen(request, timeout=timeout or 300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_text = line[len("data:"):].strip()
                if not data_text or data_text == "[DONE]":
                    continue
                try:
                    event = json.loads(data_text)
                except json.JSONDecodeError:
                    continue
                saw_any_event = True
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        yield {
                            "text": content,
                            "tool_use": None,
                            "is_finished": False,
                            "finish_reason": "",
                            "usage": None,
                            "index": 0,
                        }
                    for call in delta.get("tool_calls") or []:
                        index = call.get("index", 0)
                        slot = tool_fragments.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        if call.get("id"):
                            slot["id"] = call["id"]
                        function = call.get("function") or {}
                        if function.get("name"):
                            slot["name"] = function["name"]
                        if function.get("arguments"):
                            slot["arguments"] += function["arguments"]
                    reason = choices[0].get("finish_reason")
                    if reason:
                        finish_reason = reason
                if event.get("usage"):
                    usage_block = event["usage"]

        if not saw_any_event or (not tool_fragments and not finish_reason and usage_block is None):
            # 标准增量格式什么都没解析出来：大概率是遇到了非标准怪癖响应，
            # 退回非流式的宽容解析逻辑重新完整请求一次，当成单个 chunk 回放。
            fallback = self.completion(model, messages, api_base, api_key, timeout, optional_params, model_response, **kwargs)
            message = fallback.choices[0].message
            tool_use = None
            if getattr(message, "tool_calls", None):
                call = message.tool_calls[0]
                tool_use = {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                    "index": 0,
                }
            usage = None
            if fallback.usage is not None:
                usage = {
                    "prompt_tokens": fallback.usage.prompt_tokens,
                    "completion_tokens": fallback.usage.completion_tokens,
                    "total_tokens": fallback.usage.total_tokens,
                    "prompt_tokens_details": getattr(fallback.usage, "prompt_tokens_details", None) and dict(fallback.usage.prompt_tokens_details),
                }
            yield {
                "text": message.content or "",
                "tool_use": tool_use,
                "is_finished": True,
                "finish_reason": "tool_calls" if tool_use else "stop",
                "usage": usage,
                "index": 0,
            }
            return

        tool_use = None
        if tool_fragments:
            first = tool_fragments[min(tool_fragments)]
            tool_use = {
                "id": first["id"],
                "type": "function",
                "function": {"name": first["name"], "arguments": first["arguments"]},
                "index": 0,
            }
        usage = None
        if usage_block:
            usage = {
                "prompt_tokens": usage_block.get("prompt_tokens", 0),
                "completion_tokens": usage_block.get("completion_tokens", 0),
                "total_tokens": usage_block.get("total_tokens", 0),
                "prompt_tokens_details": usage_block.get("prompt_tokens_details") or usage_block.get("input_tokens_details"),
            }
        yield {
            "text": "",
            "tool_use": tool_use,
            "is_finished": True,
            "finish_reason": finish_reason or ("tool_calls" if tool_use else "stop"),
            "usage": usage,
            "index": 0,
        }


class OpenAICompatibleModelClient:
    def __init__(self, model, base_url, api_key, temperature, timeout, capabilities=None):
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.capabilities = resolve_capabilities(self.base_url, capabilities)
        # 是否往请求里放 prompt cache 的自定义字段。注意它只表示"我们主动发不发
        # cache key"，**不表示后端有没有缓存**：实测有的后端不认这些字段，却一直
        # 在做自动前缀缓存（响应里 cached_tokens 非零）。两者分别由
        # last_completion_metadata 里的 prompt_cache_key_sent 和 cache_hit 反映。
        self.supports_prompt_cache = self.capabilities["prompt_cache_key"]
        # 是否用标准 function-calling 把 tools= 发出去。它同时决定 prefix 教哪套
        # 协议——两套协议不能同时摆在模型面前，见 CodingForMe.build_prefix()。
        self.supports_native_tool_calls = self.capabilities["native_tool_calls"]
        # 观测到的后端行为（跨调用累积），和上面"声明的能力"分开记：
        # 声明是我们发什么，观测是后端实际回什么。
        self.observed = {"prompt_cache_hit": False, "native_tool_calls": False}
        # 强制下一次调用命中某个工具（见 force_tool_choice）。**只对紧接着的一次
        # complete() 生效，用完即清**：安全评测要的是让模型确定性地发起某一次
        # 危险调用，而不是把整轮循环锁死在一个工具上——那样模型永远给不出最终
        # 答案，只会一直被逼着调工具直到步数耗尽。
        self.pending_tool_choice = None
        self.last_completion_metadata = {}
        # provider 名字带 id(self)，是为了在同一进程里可能存在多个
        # OpenAICompatibleModelClient 实例时（比如测试），互不覆盖彼此在
        # litellm 全局 custom_provider_map 里注册的 handler。
        self._provider_name = f"codingforme_compat_{id(self)}"
        litellm.custom_provider_map = [
            item for item in (litellm.custom_provider_map or [])
            if item.get("provider") != self._provider_name
        ] + [{"provider": self._provider_name, "custom_handler": _CompatBackendCustomLLM()}]

    def complete(self, messages, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None, tools=None, on_token=None):
        """向 OpenAI-compatible 后端发起一次模型调用，经由 litellm 传输。

        为什么存在：
        runtime 不应该知道 HTTP 细节、SSE 细节、usage 字段长什么样，也不应该
        自己判断 prompt cache 参数要不要带、tool_calls 怎么从响应里抠出来、
        要不要走流式。这个函数把这些后端细节都包起来，对上层暴露统一的
        `complete()` 行为。

        输入 / 输出：
        - 输入：标准 messages 数组（system / user / assistant 带 tool_calls /
          tool 带 tool_call_id；传字符串则包成单条 user message）、
          最大输出 token、可选的 prompt cache 参数、
          可选的 `tools`（`tools.to_openai_function_specs()` 产出的标准
          function-calling schema 列表）、可选的 `on_token` 回调（传入时走
          SSE 流式，每个文本增量都会实时回调；不传时走一次性非流式请求）。
        - 输出：`{"text": str, "tool_calls": [{"name":..., "args": {...}}] | None}`；
          同时把 usage / cached_tokens 等元数据写进 `self.last_completion_metadata`

        在 agent 链路里的位置：
        它位于 `CodingForMe.ask()` 的模型调用阶段，是稳定前缀缓存复用链路、
        原生 function-calling 协议、以及流式传输真正落到 provider API 的地方。
        """
        self.last_completion_metadata = {}
        extra_body = {}
        # runtime 传入的是“稳定前缀”的签名，而不是整段 prompt 的签名。
        # 这样缓存复用针对的是稳定段，不会因为动态 history 每轮变化而失效。
        if self.supports_prompt_cache and prompt_cache_key:
            extra_body["prompt_cache_key"] = prompt_cache_key
        if self.supports_prompt_cache and prompt_cache_retention:
            extra_body["prompt_cache_retention"] = prompt_cache_retention

        kwargs = {
            "model": f"{self._provider_name}/{self.model}",
            "messages": to_messages(messages),
            "api_base": self.base_url,
            "api_key": self.api_key,
            "max_tokens": max_new_tokens,
            "timeout": self.timeout,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        # 无论这一轮有没有 tools 都要取走，否则一次没发 tools= 的调用会把强制
        # 意图留到下一轮，落在完全无关的一次请求上。
        forced_tool_choice = self.pending_tool_choice
        self.pending_tool_choice = None
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = forced_tool_choice or "auto"
        if extra_body:
            kwargs["extra_body"] = extra_body

        if on_token is None:
            text, raw_tool_calls, usage = self._complete_blocking(kwargs)
        else:
            text, raw_tool_calls, usage = self._complete_streaming(kwargs, tools, on_token)

        cached_tokens = int(usage.get("cached_tokens") or 0)
        if cached_tokens > 0:
            self.observed["prompt_cache_hit"] = True
        if raw_tool_calls:
            self.observed["native_tool_calls"] = True
        self.last_completion_metadata = {
            # 声明侧：我们发了什么
            "prompt_cache_supported": self.supports_prompt_cache,
            "prompt_cache_key_sent": bool(extra_body.get("prompt_cache_key")),
            "native_tool_calls_sent": bool(tools),
            "tool_choice_forced": bool(tools and forced_tool_choice),
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
            # 观测侧：后端实际回了什么。cache_hit 为真而 prompt_cache_supported
            # 为假是完全正常的——说明后端在做自动前缀缓存，只是不认我们的 key。
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            # 思维链 token。它是 output_tokens 的一部分，不是额外的。
            # reasoning_tokens 接近 max_new_tokens 时，说明额度在推理阶段就被吃光、
            # content 根本没轮到写——那一轮会被 parse() 归约成 retry。
            "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
            "total_tokens": usage.get("total_tokens"),
            "cached_tokens": cached_tokens,
            "cache_hit": cached_tokens > 0,
            "native_tool_calls_observed": bool(raw_tool_calls),
        }

        tool_calls = None
        if raw_tool_calls:
            tool_calls = []
            for raw in raw_tool_calls:
                try:
                    args = json.loads(raw["arguments"]) if raw["arguments"] else {}
                except (json.JSONDecodeError, TypeError):
                    # 保留原始字符串（而不是包成 dict），这样 parse() 里
                    # "args 必须是 dict" 的既有校验会自然触发 retry。
                    args = raw["arguments"]
                tool_calls.append({"name": raw["name"], "args": args})

        return {"text": text, "tool_calls": tool_calls}

    def _complete_blocking(self, kwargs):
        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"OpenAI-compatible error: {exc}") from exc

        message = response.choices[0].message
        raw_tool_calls = [
            {
                "name": getattr(getattr(call, "function", None), "name", "") or "",
                "arguments": getattr(getattr(call, "function", None), "arguments", "") or "",
            }
            for call in (getattr(message, "tool_calls", None) or [])
        ]
        return message.content or "", raw_tool_calls, _usage_dict_from_object(getattr(response, "usage", None))

    def _complete_streaming(self, kwargs, tools, on_token):
        kwargs = dict(kwargs, stream=True, stream_options={"include_usage": True})
        try:
            stream = litellm.completion(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"OpenAI-compatible error: {exc}") from exc

        text_parts = []
        tool_fragments = {}
        usage = {}
        # 没有原生 tool_calls 的场景（后端不支持 tools=，走文本兜底协议）下，
        # 一旦累积文本里出现闭合标签，本地就不用再等后面的 token 了——省下的
        # 是我们这边的等待延迟，不代表 provider 那侧一定会停止计费/生成。
        watch_for_closing_tag = not tools
        for chunk in stream:
            choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
            if choice is not None:
                delta = choice.delta
                content = getattr(delta, "content", None) if delta else None
                if content:
                    text_parts.append(content)
                    on_token(content)
                    if watch_for_closing_tag:
                        accumulated = "".join(text_parts)
                        if "</tool>" in accumulated or "</final>" in accumulated:
                            break
                for call in getattr(delta, "tool_calls", None) or []:
                    index = getattr(call, "index", 0)
                    slot = tool_fragments.setdefault(index, {"name": "", "arguments": ""})
                    function = getattr(call, "function", None)
                    if function and getattr(function, "name", None):
                        slot["name"] = function.name
                    if function and getattr(function, "arguments", None):
                        slot["arguments"] += function.arguments
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = _usage_dict_from_object(chunk_usage)

        raw_tool_calls = [
            {"name": slot["name"], "arguments": slot["arguments"]}
            for slot in tool_fragments.values()
        ]
        return "".join(text_parts), raw_tool_calls, usage


