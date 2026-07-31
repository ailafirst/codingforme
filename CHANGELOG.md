# 更新日志

本项目的版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。0.x 阶段,次版本号的变化即可能包含破坏性改动。

## [0.2.0] - 2026-07-31

这一版把模型交互层从「自定义文本协议 + 手写 HTTP」换成了「标准 function-calling + LiteLLM 传输」,并顺带补上了流式输出。

### 破坏性改动

- **运行时不再是零依赖**:新增唯一的运行时依赖 `litellm>=1.60.0`。此前 README 承诺的"运行时只用 Python 标准库"不再成立。
- **旧会话首次 `--resume` 会报 `workspace-mismatch`**,并自动新建 checkpoint。这是预期行为,不是 bug:工具 schema 现在会被翻译成 JSON Schema 一并计入 `tool_signature()`,prompt 前缀本身也改了,两者都进 `runtime_identity`。会话历史本身不会丢。
- **prompt 缓存键随之失效一次**。`prompt_cache_key` 就是前缀的 hash,前缀变了,升级后的第一批请求不会命中缓存。
- **移除文本标签协议**。`<tool>...</tool>` / `<final>...</final>` 不再是受支持的输出协议(详见下)。若你依赖 `CodingForMe.parse()` 或 `retry_notice()` 的签名,注意 `native_tool_calls` 参数已删除。

### 新增

- **标准 OpenAI function-calling 协议**。`tools.py` 新增 `to_openai_function_specs()`,把 `BASE_TOOL_SPECS` 这份微型 DSL 翻译成标准 JSON Schema 的 `tools=` 数组。`BASE_TOOL_SPECS` 仍是唯一声明处,不引入第二份手写 schema。
- **SSE 流式输出**。`model_client.complete()` 接受可选的 `on_token` 回调;REPL 默认启用,模型生成时暗色实时预览,最终答案仍按原样渲染成圆角框。流式请求带 `stream_options: {"include_usage": true}`,否则携带 usage/cache 的最后一个 chunk 不会出现。
- **后端能力声明**。`models.resolve_capabilities()` 按「保守默认值 → 已知后端表 → 显式覆盖」三层合成,取代了原先靠 URL 子串猜测的写法。新增两个可选环境变量:
  - `CODINGFORME_OPENAI_NATIVE_TOOL_CALLS` —— 后端吃不吃 `tools=` 参数
  - `CODINGFORME_OPENAI_PROMPT_CACHE_KEY` —— 后端认不认 `prompt_cache_key` 字段

  未设置表示"没意见",交给前两层决定。
- **协议漂移计数 `text_protocol_tool_calls`**。进 trace 的 `model_parsed` 事件和 report 的 `prompt_metadata`,健康状态下恒为 0。
- **`models.tool_call()` / `models.final_answer()`**:构造脚本化模型输出的公开辅助函数,产出的就是 `complete()` 的真实返回形状。
- **`models.force_tool_choice()`** 与 `client.pending_tool_choice`(一次性):让评测能在协议层强制某一次工具调用。

### 变更

- **`complete()` 的返回值从裸字符串改为 `{"text": str, "tool_calls": [...] | None}`**,抹平了各家后端的差异。
- **prompt 里只教一套协议**。`build_prefix()` 不再按后端能力二选一地教「原生」或「文本标签」;`retry_notice()` 和 `tools.py` 的 `TOOL_EXAMPLES` 也一并收敛——后者现在只举参数对象,不举调用形式。这条通道曾经是协议说明的隐蔽泄漏点(它进的是参数校验失败的报错信息)。
- **`supports_native_tool_calls` 降级成纯传输开关**:只决定要不要真的发 `tools=`,不再影响教给模型的内容。
- **`parse()` 里的标签解析降级为「宽容读取」**:prompt 不再教这套写法,但模型万一自作主张写成文本(推理模型偶发),仍会被解析出来,不让那一轮白白作废。
- **已知非标准响应形状走 `litellm.CustomLLM` 桥接**自行解析,绕开 litellm 的严格 schema 校验。两个已知怪癖:声明 `stream: false` 却仍返回 SSE;返回 Responses-API 风格的 `output_text` 而非标准 `choices`。
- **`last_completion_metadata` 区分「声明」与「观测」**:`prompt_cache_key_sent` / `native_tool_calls_sent` 是我们发了什么,`cache_hit` / `native_tool_calls_observed` 是后端实际回了什么。前者为 false 而后者为 true 是正常的——有的后端不认我们的 cache key,却一直在做自动前缀缓存。
- **真实模型安全评测改用 `tool_choice` + 裁剪工具注册表**逼出危险调用,不再依赖模型照抄一段 `<tool>` 文本(那与 prompt 里"绝不要把工具调用写成文本"直接冲突)。新增 `tool_call_fidelity` 记录复现保真度,`max_steps=1` 保证记录到的是被强制的那次调用而非模型被拦后的绕路尝试。
- **测试与评测的脚本化输出全部改用原生 dict 形状**。此前上百个用例验证的是生产环境永远不走的文本解析分支。

### 修复

- **推理模型的空 content 不再打断整轮运行**。模型可能把全部输出放进 `reasoning_content` 而让 `content` 缺席,或在推理阶段就被 `max_tokens` 截断(`finish_reason: "length"`)。这两种响应此前会抛异常、丢掉此前所有工具执行成果,现在归约成 `retry` 让模型下一轮重来。只有连 `choices[0].message` 都找不到时才算真正无法解析。
- **协议摇摆**。此前 prompt 教文本标签、同时又发原生 `tools=`,实测同一个后端 19 次走原生 / 17 次吐文本标签,推理模型还会把 `<tool>` 标签埋进 `reasoning_content` 里让整轮作废。

### 实测数据

真实后端跑完整 12 个基准任务(47 次模型调用):

| 指标 | v0.1.0 | v0.2.0 |
|---|---|---|
| 走标准 function-calling | 0 | 45 |
| 走文本标签解析 | 41 | 0 |
| retry 次数 | 15 | 0 |
| 进程级崩溃 | 0 | 0 |
| 任务通过率 | 1/12 | 1/12 |

通过率**没有变化**:基准的 4–6 步预算是按脚本化假模型的最优路径卡的,真实模型会先 `list_files`/`read_file` 探查,两三步就耗尽预算(12 个任务里 9 个 `stop=step_limit_reached`)。这是基准设计的问题,不是协议改动的效果。

真实模型安全评测 10 个场景,9 个成功复现危险调用且**全部被闸口正确拒绝**。唯一未复现的 `patch_missing_new_text` 是预期结论而非缺陷:原生 function-calling 下模型发不出缺必填参数的调用,JSON Schema 在协议层就挡住了。作为对照,同样非法但 schema 合法的两条(超范围 `timeout=121`、空字符串 `task=""`)模型照发不误,由 `validate_tool()` 兜住。

### 未包含

LiteLLM 目前**只用作协议类型层,不是网关**。所有流量都经 `CustomLLM` 绕开了它的内建 provider 路径,因此它的重试、成本核算、fallback、多 provider 路由都没有生效。`.env` 里的 `CODINGFORME_ANTHROPIC_*` / `CODINGFORME_DEEPSEEK_*` 仍是未接线的配置。

## [0.1.0]

首个版本。
