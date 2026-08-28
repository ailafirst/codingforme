# 更新日志

本项目的版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。0.x 阶段,次版本号的变化即可能包含破坏性改动。

## [0.2.5] - 2026-08-28

这一版把"评测"从散在各处的实验脚本收敛成一套有坐标系的体系,并让控制循环真正用上标准 messages 数组:一轮可以发多个工具调用,上下文按变更频率分层摆位。附带一个默认关闭的受限编排工具 `run_plan`。

### 破坏性改动

- **基准数据集 schema v1 → v2**。`verifier`(一条 `python3 -c "..."` 命令)换成 `checks`(声明式的 `fail_to_pass` / `pass_to_pass` 两组断言),并新增必填的 `mutable_paths`。判定由当前解释器进程内执行,不起子进程、不走 shell,因此不再依赖宿主装没装 `python3`。两组都要求非空:允许 P2P 为空就等于允许作者跳过"没破坏别的东西"这半边。旧数据集加载时直接报错。
- **`complete()` 的第一个参数从 `prompt` 字符串变成 `messages` 数组**。`to_messages()` 兼容裸字符串输入,但自定义 model client 的签名需要跟着改;要按文本判断上下文内容的脚本化 client 用 `flatten_messages()` 压平。
- **`ContextManager.build()` 降级为兼容入口**,真正的产出是 `build_all()` 的 `(messages, prompt, metadata)`。`agent._build_prompt_and_metadata()` 换成 `_build_context()`。
- **`SCRIPTED_MODEL_OUTPUTS` 改名 `ORACLE_SOLUTIONS`**。旧名字读起来像"模型的输出",于是回放它跑出来的通过率容易被当成能力度量,而任务根本不是被测系统自己解的。
- **`tool_signature()` 的哈希变了**(工具 schema 新增 `format` 参数、可选注册 `run_plan`),老会话首次 `--resume` 会报 `workspace-mismatch` 并自动新建 checkpoint。这是预期行为,会话历史不丢。
- **`pyproject.toml` 的 `packages` 增加 `codingforme.eval`**,可编辑安装需要重装一次。

### 新增

- **评测体系底座 `codingforme/eval/`**,四个原语全部纯标准库:
  - `HarnessSpec`(harness.py)—— 被测配置固化成可命名、可指纹、可序列化的一等对象,并作为**唯一的 agent 装配入口**。7 个内建变体:`full` / `no_memory` / `no_context_reduction` / `no_prompt_cache` / `read_only` / `strict_approval` / `plan_tool`,消融就是换一个名字重跑。`fingerprint()` 含 `code_signature()`——提示词模板、工具 schema 与七个模型相关模块的规范化源码哈希(`ast` 去掉 docstring 后 `unparse`,所以改注释不影响),避免"改了代码没改配置"的两次跑批被当成可比。
  - `TraceIndex`(trace.py)—— 把 `.codingforme/runs/` 重建成 `Session → Run → Turn` 三级结构,所有指标只跟这三个对象打交道。同一轮的事件靠 `turn` 字段配对,不靠文件顺序。`TurnRecord.tools` 是**列表**(一轮可有多条 `tool_executed`)。
  - **统一结果 schema**(report.py)—— 六个顶层块 `run_context / harness / dataset / cases / aggregates / trace`。每条 case 必须声明层(`L0-component` ~ `L4-system`)与轴(`capability` / `efficiency` / `economy` / `safety` / `reliability`),写错名字直接 `ValueError`;聚合里五条轴**永远全列**,没数据的渲染成"未覆盖"。
  - **判分器**(scorers.py)—— 13 条 L1 轨迹断言分四类(`factual` / `constraint` / `sequence` / `reachability`)+ 5 条 L3 会话断言。判据只取 trace 里已落盘的字段,不用模型当裁判;不适用返回 `None` 而不是硬凑成通过;失败必须带归因(哪个 run、哪一轮、哪条 path、哪组参数)。
- **每条断言标注判定对象**(`ASSERTION_SUBJECTS`:`model` 7 条 / `harness` 6 条),聚合里 `by_subject` 分开报。分界线是"这条挂了该动谁"。
- **数据集可声明预期失败**(`expected_trajectory_failures`)。基准里有几个任务就是冲着触发失败去的(`path_escape_recovery` 要先越界一次、`invalid_patch_recovery` 要先发一个缺 `new_text` 的调用),对应断言挂掉是设计如此。L1 因此报三个桶:通过 / 预期内失败 / 预期外失败。
- **L3 跨会话套件**(`benchmarks/session_tasks.json` + session_suite.py)—— 4 条会话覆盖 LongMemEval 的四类题型(单会话召回 / 跨会话聚合 / 知识更新追踪 / 时序推理),标了 `restart_before` 的轮次**重建 agent 实例**走真实 resume 路径。断言的是"harness 有没有把先前建立的事实重新放回 prompt",不是"模型有没有答对"——后者不是 harness 的职责。
- **工具用量统计**(tool_usage.py)—— `by_tool` 分开数调用 / 执行成功 / 被闸口拒绝 / 出错("调用了 20 次 patch_file"和"发起 20 次、被挡掉 18 次"是两回事);`plan` 分开数 `plans_attempted` / `plans_executed` / `plans_rejected`("模型爱用这个工具"和"每次都写错、每次白烧一个往返"应对完全相反)。零调用时产出零值并在 markdown 里明说,不省略这一节。
- **评测统一入口 `scripts/run_eval_suite.py`**,把 HarnessSpec → 执行 → TraceIndex → L1/L2 判分 → 统一 schema 整条链路串起来。`--suite fixed-benchmark` / `cross-session`,`--harness <变体名>`,`--live-model` 切换真实模型。另有 `scripts/aggregate_repeats.py` 聚合 k>1 的重复跑批。
- **一轮多工具调用**。`parse()` 的 `"tool"` payload 恒为列表,单个调用也包成一元列表。执行语义三条:每个调用**各自计一步**(否则一轮发 20 个就绕过 `max_steps`,连带绕过只读与审批的粒度)、一个失败不打断后面的、预算在轮内用完就停下并写 notice 报出未执行的调用名。
- **受限编排 `run_plan`**(plans.py,**默认关闭**,由 `HarnessSpec(feature_flags={"plan_tool": True})` 打开)。模型写一小段程序把有依赖的多步调用串起来,在一次模型往返里发完。**不建在 `run_shell` 上**——那等于把"只能调用声明过的工具"换成"能跑任意代码"。解释器 `ast.parse` 后自行遍历(不 `exec`),白名单节点,**禁属性访问**(挡掉 `().__class__.__bases__` 那条经典逃逸),四条硬上限防死循环。内层调用照样走 `run_tool()`、各自计一步、各写一条带 `via: "run_plan"` 的 `tool_executed`。计划源码里静态出现 `print(` 时结果不再全文进转录,另有 `MAX_TRANSCRIPT_CHARS = 12000` 的聚合上限。
- **retry 归因码**(`RETRY_REASON_*`,由 `RetryNotice` 这个 str 子类捎带),落进 trace 的 `model_parsed` 事件与 `TurnRecord.retry_reason`。三类 retry 在工件上原本长得一模一样而应对相反:`empty_response` 要调 `max_new_tokens`,`tool_args_not_an_object` 要改工具 schema 和示例。
- **会话身份字段** `TaskState.session_id` / `run_seq`,`emit_trace()` 给每条事件盖 `session_id` / `run_id` / `run_seq` / `turn`。没有这条 join 键,run 工件回连不到 session,跨会话分析在数据层就做不了。老工件缺字段时索引兼容并标 `synthetic=True`。
- **工具输出的第二种形状**。`list_files` 的 `format="paths"` 与 `search` 的 `format="paths"` 回一行一个工作区相对路径(正斜杠)。计划沙箱没有属性访问也没有标准库,"结构化"在这里的正确形态是**不需要解析**。未知 `format` 值一律报错,不静默退回默认值。
- **被裁掉的历史留下确定性摘要**。放不下的条目压成一行 `Omitted context: N earlier transcript entries dropped ...`,挂在"运行状态"那条 `user` 消息里(不能挂进 history,那串消息受配对约束)。刻意不调模型生成摘要——那会在组上下文的路径里插进一次约 17 秒的往返。
- **fan-out 类别的两个基准任务**(`survey_missing_headers` / `survey_longest_module`),共用新的 `tests/fixtures/bench_repo_survey`(README + `src/` 下 8 个模块)。原有 12 个任务全是"读一个文件、改一个文件"的串行形状,结构上量不出受限编排的收益。这个仓库也补上了回归检测原本缺的对象:此前两个样板仓库各自只有一个文件,而那个文件恰好就是允许改的那个。

### 变更

- **发出去的是标准 messages 数组,摆位按"这段内容多久变一次"分层**,不是把 `SECTION_ORDER` 直接翻译成消息:`system` 只放跨任务逐字节相同的规则与工具清单,第一条 `user` 放仓库快照与 resume checkpoint,中间还原对话轮次(`assistant` 带 `tool_calls`、每个结果一条 `role:"tool"` 带 `tool_call_id`),倒数第二条 `user` 放 working / relevant memory,最后一条是当前请求。**不变量**:`assistant` 里每一个 `tool_call` 必须紧跟恰好一条同 `tool_call_id` 的 `tool` 消息,少一条、多一条、顺序错了后端都会直接拒掉整个请求。
- **`run_context` 新增 `execution_mode`**(`oracle-replay` / `live-model`)。回放模式下 markdown 报告在**所有数字之前**强制渲染口径声明:capability 轴评的是参考解脚本,safety / efficiency / reliability 才是在评 harness。
- **`tools_allowlist_respected` 改判"声明有没有落地"**,读的是这次运行自己工件里的 `tools_allowlist` 与 `tool_names`,而不是变体的 `HarnessSpec`。固定基准跑的是 `full` 变体、白名单是 `None`,于是这条断言此前 36 次运行里一次都没触发过——报告里长得像"没问题"、实际是"没查"。
- **`run_context` 记录 `workspace_root` 与 `workspace_git_root`**。后者非空说明评测工作区嵌在某个 git 仓库内部,快照会被悄悄放大成那个外层仓库,这批数字和别的跑批不可比,markdown 报告会在所有数字之前渲染警告。判定走文件系统逐层找 `.git` 而不是 `git rev-parse`——临时工作区在写报告时可能已被清掉。
- **`no_repeated_calls` 的 offenders 带上被重发的那组参数**。原来只记 `{turn, name}`,一次 36 运行的跑批报出三条"`read_file` at turn 5",光看报告无法判断是死读同一个文件还是三个任务各撞了别的东西。
- **prompt 只教一套 function-calling 协议**这条不变量扩展到 retry notice:那里曾经写着 `model returned an empty <final> answer`,等于在模型出错的那一刻把它推向一套我们既没发 schema 也不打算支持的协议。
- **凡是点名某个工具的文案都按 `agent.tools` 现算**。注册表会被"变体白名单 ∩ 任务白名单"裁掉一部分,硬写的文案会在只剩 `read_file` / `patch_file` 的运行里仍然指着 `write_file` 说话——模型照做只会拿回一句 `unknown tool`,白烧一个往返。
- **工具示例改走 JSON Schema 的标准 `examples` 关键字**,放在 `parameters` 上。实测(16 次采样 × 3 组):这个后端把 `parameters.examples` 原样送到模型面前(16/16 吐出示例里那个无从推导的值),Anthropic 的 `input_examples` 字段名被静默丢弃(0/16),什么都不给时模型 12/16 直接拒绝调用。
- **`.env` 一律从 `config.project_root()` 找**,不看工作区、也不看进程 cwd。此前用的是工作区的 `repo_root`,而工作区经常在仓库外面,实测载入 0 个键——配置实际是被 `import litellm` 顺手 `load_dotenv()` 的副作用喂进来的。
- **默认单轮输出上限 `DEFAULT_MAX_NEW_TOKENS` 提到 1024**(评测侧此前是 64,回放时代的遗留)。它进变体指纹,所以旧工件与新工件不再算同一配置。

### 修复

- **`TurnRecord.tools` 从 `dict` 改成列表**。原来 `record.tool = dict(event)` 会让同一轮后来的调用覆盖先来的,`path_confined` / `tools_allowlist_respected` 这些安全断言于是只查得到每轮最后一个调用,前面的静默漏检——**而漏检在报告里长得和通过一模一样**。
- **`resume_status` 分清运行级与逐轮两个口径**。report 顶层那个在首轮组完 prompt 之后钉死一次,`prompt_metadata` 里那个是最后一轮的元数据。基准的两个 resume 任务原本查的是后者,参考解正好一轮所以回放全绿,真实模型多走一步就 0/3 恒挂。
- **复制样板仓库时忽略残留的运行工件**。live 跑批会在样板仓库里写下 `.codingforme/`,原样复制会让每个任务的工作区里凭空多出一份别的运行留下的 session,两个 resume 任务尤其容易被误读成"恢复成功了"。忽略名单取自 `workspace.IGNORED_PATH_NAMES`,避免两处各写一份慢慢漂开。
- **REPL 输出对非 UTF-8 终端更耐受**(`cli._make_output_resilient()` / `_terminal_safe()`),中文 Windows 控制台不再因编码抛错打断会话。

### 实测数据

**固定基准 · 真实模型**(12 个任务 × 重复 3 轮 = 36 次运行,`full` 变体,步数上限 16):

| 指标 | 结果 | 读法 |
|---|---|---|
| L2 任务层 | 36 / 36 | 五重判定同时成立:产物存在 + 步数在预算内 + F2P 全过 + P2P 全过 + 终止原因正常 |
| pass^k | 12 / 12 题三次全过 | 重复采样此前是空白,因为回放的方差恒为 0 |
| L1 轨迹层 | 305 / 309,干净率 99.0% | 309 = 36 次运行 × 每次适用的断言条数(不适用的退出分母) |
| 按判定对象 | harness 96/96,model 209/213 | 闸口、预算、白名单这半边零失误;挂掉的 4 条全在模型自己发的调用序列上 |
| 成本 | 19.0 万 token / 121 轮往返 | 输入 17.4 万(62.4% 命中缓存,每轮均 1434),输出 1.7 万(66.9% 是推理过程) |

**messages 数组改造的收益和当初的预期不是一回事。** 探针实验(假仓库、开放式请求)测出多调用率 15% → 44%(Fisher 精确检验 p=0.0031),但 12 任务 × 3 轮的真实跑批里**批量率纹丝不动**(7.1% → 9.7%,p=0.37)——基准任务多是"读完才知道改哪"的串行依赖,没有可合并的独立调用。真正兑现的是别的:走到终点 76% → 97%(p=0.0142)、没打转 74% → 94%(p=0.0249)、每次运行工具步数中位 6 → 4(Mann-Whitney U p=0.0012)、模型往返 313 → 201 轮。机制是模型现在看得见自己上一轮调了什么,不再反复回读同一个文件。

**前缀缓存命中率 32.7% → 77.6%**(同一批 12 任务 × 3 轮跑批)。病因是"prefix + memory + relevant_memory 合成一条 `system`":仓库快照随任务变、working memory 每轮变,放进 `system` 会截短跨请求的公共前缀,连带把工具定义那一大段挤出缓存。修法就是上面那个分层摆位。

**跨会话套件 · 真实模型 7 / 12**(回放模式 12/12)。挂的 4 条全是"先前建立的事实有没有被放回上下文",归因字段显示实际带回 0 条笔记。原因是笔记的产生依赖模型最终答案里出现特定句式,模型换个说法就不触发——**这是 harness 的问题,不是模型的问题**,尚未修复。

**证伪证据**:`test_disabling_memory_fails_exactly_the_recall_assertions` 断言关掉记忆后**恰好**那 4 条召回断言挂掉、reliability 轴不受连坐。没有这条,12/12 只说明断言什么都没查。

**受限编排别指望在固定基准上量出收益。** Anthropic 在 τ²-bench 上的实测是:每轮只有一两个串行调用的负载,programmatic tool calling 分数不变、成本还高约 8%;我们的基准任务正是这个形状。注册表里有 `run_plan` 的运行每轮输入 token 多 642 个(1414 → 2057)。收益要用 fan-out 形状的探针量:3.72 工具/轮,内层结果 468,485 字节 → 转录 9,635 字符(−97.9%),任务完成率 25% → 83%(Fisher p=0.0123)。

**回归**:353 条测试,350 通过 / 3 个已知的 Windows 环境失败(`run_shell` 用 POSIX 语法落到 cmd.exe;一条断言指向被 gitignore 的 `docs/`)。ruff 默认规则集 clean。

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
