# 更新日志

本项目的版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。0.x 阶段,次版本号的变化即可能包含破坏性改动。

## [0.3.0] - 2026-09-05

这一版把上下文治理从"一根裁剪线"做成分级流水线(阶段三~十二):`history` 变成只追加的事实记录、发给模型的是它按预算与新鲜度算出的一次投影;超限工具结果落盘可取回、过期读会被标记、最近窗口按块推进以保住前缀缓存;预算改由上下文窗口派生而不是写死常量。另修复了一个生产级缺口——系统提示词此前从未教模型长期记忆的写法,导致"记住这件事"类请求静默不生效。

### 新增

- **消息投影层(阶段三)**:`history` 是只追加、一个字节不改的事实记录,发给模型的那份是它的一次投影。「这一条这次怎么呈现」从散在 `_compressed_history_entries()` 里的一串 if 分支变成十个一等的形态常量(`POLICY_FULL` / `CLIPPED` / `FILE_SUMMARY` / `POINTER` / `ERROR_KEPT` / `SHELL_HEAD` / `NOT_EXECUTED` / `CLEARED` / `SQUEEZED` / `DROPPED` 加 `STRUCTURAL`),决策(`_policy_for()` / `_old_tool_policy()`)与渲染(`_render_policy()`)拆开。**分支行为一条没变**,变的是它们可以被整组验:`test_every_policy_combination_keeps_the_tool_calls_paired` 枚举 10³ 种形态组合 × 「哪几条活过预算」的掩码,逐个验「每个 `tool_call` 恰好配一条同 `tool_call_id` 的 `tool` 消息」——这是唯一一条错了会让整个请求被后端拒掉、而不是降级的约束;`test_every_policy_is_reachable_from_a_real_history` 保证没有永远选不中的死形态。
- **最近窗口的条目数上限也按块推进**(`_recent_start_by_tool_turns(..., block=)`)。`RECENT_WINDOW_BLOCK` 从前只量化了工具轮那条边界,条目数上限 `len(history) - 2×tool_turns`(管「工具很少、对话很多」的历史)每多两条对话就往前推两格,消息数组从那里往后全部改写。实测 24 轮真实压力探针、预算 12,423、**一次裁剪都没触发**:前缀被改写的轮次 **15 → 5**,估算前缀命中率 46.3% → **73.1%**,估算未命中 token **52,510 → 27,266(−48.1%)**;代价是总 prompt token +3.7%(向下取整让窗口浮动着变大,最多多留 5 条全文条目,方向与 `_blocked_recent_window()` 一致)。**这是这套上下文工程里少见的、在没有任何预算压力的常规运行上就能兑现的收益**,而常规运行正是真实跑批的形态。关掉 `recent_window_block` 时两条边界一起退回老行为。
- **`prompt_metadata["history"]` 新增 `squeezed_entry_count` 与 `recent_start`**。前者是「预算不够时被压扁成 10 token 残句、而不是整条丢弃」的条目数——压扁就是 Claude Code L4(可逆折叠)的等价物,它一直藏在保留循环里,既没有名字也没有计数,于是和整条丢弃在工件上长得一模一样;后者是最近窗口边界这一轮落在哪条,按块推进的效果只能对着它读。
- **会话摘要(阶段二:Claude Code 自动压缩里**便宜的那一半**)**:压力到触发点时,把最早的一批历史条目整体换成一份确定性摘要(**不调模型**),覆盖点与摘要文本存进 `session["context_summary"]`,跟着落盘与 resume;摘要只在推进覆盖点的那一轮重新生成。配 `session_summary` 消融开关(默认开)、`no_session_summary` 变体,以及 `context_pressure.session_summary` 六个字段(零值也写)与 `message_layout.session_summary_present`。**兑现的是信息保全**:从前 24 轮压力探针最后只剩一行「dropped 39 entries」、用户第 1 轮定的规格已经不在上下文里,现在它的原文仍在。**没兑现缓存 churn,而且适用条件可以量出来**:覆盖点不许吃进最近那截历史,那截本身超过目标点时覆盖点就被顶死在一个每轮只涨 2 的上限上——实测每条对话 96 token 时边界移动 7 → 1、token 省 5.7%,206 token 时 14 → 5、token 多花 8.2%,306 token 时 19 → 19 完全无效。真实探针属于最后一档。详见 `docs/architecture/compression-pipeline-migration.md` §2.3,并由 `test_the_summary_is_inert_once_the_protected_tail_alone_exceeds_the_target` 锁住,免得下一轮靠调比例去救它。
- **`context_pressure_absorbed` 的判据改成「任意一级压缩」**(逐段硬裁 `budget_reductions` 或会话摘要 `session_summary.compactions`)。只数前者的话,摘要一步把占用率压回目标点以下时 `budget_reductions` 是空的,这条断言在阶段二之后会变成恒挂。
- **分级压缩(迁移 Claude Code 五级流水线的阶段一)**:上下文占用率到 `COMPRESSION_TRIGGER_RATIO = 0.85` 就开始压、一压压到 `COMPRESSION_TARGET_RATIO = 0.70` 以下,取代从前「只有 `prompt_tokens > total_budget` 一根线、裁到刚好不超为止」的做法。配 `graded_compression` 消融开关(默认开)与 `no_graded_compression` 变体,以及 `prompt_metadata["context_pressure"]`(occupancy_before/after、trigger/target、freed_tokens,零值也写)。同一条 24 轮压力会话 A/B:峰值占用 100.0% → 77.4%,24 轮 prompt token 合计 88,721 → 69,725(**−21.4%**),代价是多丢 4 条历史条目。**迟滞那一半试过、实测无效已撤掉**——边界是预算驱动的,对话单调增长时任何守预算的丢弃策略都必须每轮多丢一点,只能靠会话摘要那种「偶尔一次大跳」解决。
- **`HarnessSpec.context_window` 与 `window_32k` / `window_16k` 变体**:走 `set_context_window()`,整条派生链(预算算式、`tool_output_limit()`、计划转录上限、`context_budget_breakdown`)跟着走;直接写 `total_budget` 只改总数,工件上 `context_window_tokens` 会和它对不上。
- **压力探针会话 `long_dialogue_pressure`(24 轮纯对话)与 L3 断言 `context_pressure_absorbed`**:三个条件同时成立才通过——裁剪轮次达标(确实压到了)、没有一轮超预算(压住了)、没有一轮裁到底还不够。加它是因为工具结果那条路径上的压力被 L1 落盘和 L3 热尾吃干净了,三个窗口档位 178 个真实预算轮次`budget_reductions` 恒为 0;user/assistant 文本是这两级都碰不到的唯一成分。
- **`patch_file` 新增追加用法：`old_text` 传空串 = 把 `new_text` 接到文件末尾**（文件不以换行结尾时先补一个换行），回一句 `appended to <path>` 加同样格式的片段回显。这不是放宽「`old_text` 必须精确命中一次」那条约束——空串在任何文件里都出现无数次，本来就永远不可能是一次合法替换。加它是因为实测模型想追加一行时**自发**就发这个形状（一次 k=3 的 live 跑批里 3 次有 2 次），而它从前只换回一句 `old_text must not be empty`，白烧一步加一个约 17 秒的往返，`arguments_valid` 只剩 1/3。prefix 里那条规则同步改成 “Required tool arguments must all be provided”（原文 “must not be empty” 会把模型从这条合法用法上推开）。**注意这会改 `tool_signature()`**，老 session 第一次 resume 报 `workspace-mismatch` 是预期行为。
- **两个消融开关,把上下文工程的收益变成可测量的**:`no_tool_output_spill`(关掉超限结果的落盘与指针,退回 `clip()` 截断)与 `no_window_block`(关掉最近窗口的按块推进,退回每个工具轮推一格)。两者默认都是**开**,关掉时严格退回机制出现之前的行为。加它们是因为三个阶段此前只验证到「机制正确、没有回归」,而「机制有没有收益」在工件上根本无从判断——同一份负载跑一次开、跑一次关,是唯一的量法。
- **`scripts/run_context_stress.py` 新增 `--harness` / `--max-steps`,以及第五个探针 `forced_spill_1m`**;每跑完一个探针就落一次盘(一次跑满 44 步的探针曾因为模型答案里有个 emoji 在最后一步 `UnicodeEncodeError`,整批结果全丢)。探针定义里新增 `expect`:这道题正确答案里必然出现的字面串,全部命中才记 `answered`——没有硬的结果指标,「省了上下文」和「省了上下文但把答案弄丢了」分不出来。
- **`scripts/compare_context_ab.py`**:把两份探针工件并排成对照表(答对 / 步数 / 输入 token / 缓存命中占比 / 落盘与占位计数)。对照口径写死在脚本里,免得手工读 JSON 时只挑对自己有利的那一列。
- **`scripts/measure_tool_output_ceiling.py`**:量「哪个工具的输出够得着落盘门槛」。实测 1M 档(单条上限 14,791)下只有 `read_file` 够得着——`search` 因为自己的 200 条命中上限封顶 12,699 token(门槛的 86%),`list_files` 的树更小。这解释了为什么真实跑批里 `spilled_calls` 长期是 0:不只是基准文件太小,而是入口上限分层之后,1M 档的落盘只剩 `read_file` 一条触发路径。
- **超长工具结果落盘(阶段三 L1)**。单次工具输出超过 `context_manager.tool_output_limit(total_budget)` 时,全文写进 `<workspace>/.codingforme/tool_outputs/<run_id>/<n>-<tool>.txt`,上下文里只留头部预览加一行指针:`[full output saved to <正斜杠相对路径> (N tokens); use read_file on that path to see the rest]`。取回走已有的 `read_file`,不新增工具。落盘目录必须在 workspace root 之下(否则 `path()` 会挡住模型读回来),落盘前过 `redact_text()`,任何一步失败退回普通 `clip()` 而不抛异常。实测本仓库 8k 档:`search("def ")` 28,416 → 1,287 token,`read_file("codingforme/runtime.py", 1..3000)` 35,679 → 1,286。
- **`tool_executed` 事件新增 `tool_output_spilled` / `tool_output_spill_path` / `tool_output_full_tokens` / `tool_output_kept_tokens`**,`prompt_metadata["history"]` 新增 `recent_tool_window` / `spilled_pointer_count` / `preserved_error_count`,跑批汇总新增 `aggregates["tool_usage"]["spill"]`。零值也照常渲染,markdown 里明说「一次都没触发」——省略这一节会被读成「这块没问题」。
- **`/context` 斜杠命令**:不带参数显示当前窗口档位、来源和**逐项**预算算式,`/context 128k` 现场换档(`128000` / `1m` 同样认,不在档位上的值向下取整并明说)。配套 `CodingForMe.set_context_window()` 与 `models.parse_window_tokens()`——后者是 CLI 参数、环境变量、斜杠命令共用的唯一解析器。
- **`--context-window` 参数与 `CODINGFORME_CONTEXT_WINDOW` 环境变量**,覆盖自动探测。
- **`prompt_metadata` 新增 `context_window_tokens` / `context_window_source` / `budget_floor_exhausted` / `protected_sections`**。`budget_floor_exhausted` 和「刚好装下」在别的字段上看不出区别,而含义相反:该调预算,不是该继续裁。
- **供给侧新鲜度(阶段五)**:`read_file` 之后同一个文件又被写过,那条记录里是**改动之前**的内容,而它和"文件现在就长这样"在模型眼里完全一样——实测后果是 `patch_file` 的 `old_text` 从这段过期内容里抄出来、命中 0 次被打回。窗口内的过期读整条换成 `STALE_READ_MARKER`(调用签名照留);窗口外已落盘的过期读,指针照留(那是取回的唯一线索)、后面追加 `STALE_SPILL_MARKER` 说明盘上那份也是旧版。判定看 history 里"这条 `read_file` 之后有没有对同一路径的写"(`wrote` 字段来自快照 sha256 差异,不是 args 里的 `path`,因此 `run_shell`/`run_plan` 改的文件与失败的写都算得对)。配 `stale_read_invalidation` 消融开关(默认开)/ 变体 `no_stale_read`,观测 `prompt_metadata["history"]` 的 `stale_read_count` / `stale_pointer_count`。
- **可逆折叠有了对照组**:`reversible_squeeze` 消融开关(默认开)/ 变体 `no_reversible_squeeze`。关掉就退回它被命名之前的行为——窗口外放不下的条目整条丢弃,只在 `Omitted context:` 里留个数;开着则压成一句保留开头的 10 token 残句。两条属性测试锁住"L4 完全可逆"这句话:压缩不改写 `session["history"]` 本身,预算放大后被压掉的内容原样回来。
- **`scripts/run_compression_gate.py`**(阶段四开工判据):合成 24 轮对话 × 三档强度,查逐段硬裁还会不会被触发,不调模型、几秒钟跑完,只测 8k 一档。
- **`scripts/run_system_gate.py`**(阶段十一):同一份判据横扫 8k/16k/32k/64k/128k 五个窗口档位,外加真实工具调用(落盘 / 过期读 / 最近窗口),查同一次会话里九个机制各自在每个档位上发生了什么,不调模型、约 2 分钟跑完。
- **`scripts/run_compression_ratio.py`**(阶段十二):128k 档(生产实际生效窗口)在 7 档压力强度下的压缩率,压缩开(`full`)对比完全关(`no_context_reduction`)。
- **`scripts/run_tool_pressure_matrix.py`**:量工具输出落盘与最近窗口机制在不同压力形状下的可恢复性——一次读回落盘内容会不会因为超过单条上限而被再次截断(实测全量重读会再次触发落盘,只有窄尾部窗口读才能避开)。
- **`long_log_audit_token` 基准任务**(`tests/fixtures/bench_repo_spill`):4000 行日志里唯一一行 `AUDIT-TOKEN:` 在未知位置,工具白名单只给 `read_file` / `patch_file`(没有 `search`),逼模型手动分窗口读——专门用来触发超长工具结果落盘。

### 修复(续)

- **durable-memory 的格式约定第一次被写进系统提示词**。`extract_durable_promotions()` 只认以 `Project convention:` / `Decision:` / `Dependency:` / `Preference:`(或中文 项目约定：/决策：/依赖：/偏好：)开头的行,但这套格式此前从未出现在 `PROMPT_TEMPLATE` 里——模型说"好的我记住了"之类的自然语言永远不会命中,长期记忆提升因此在生产里静默失败。补一条规则后,cross-session 套件的真实模型跑批里 `session_evidence_surfaced` 0/4 → 4/4、`supersede_recorded` 0/1 → 1/1,L3 整体 8/14(57.1%)→ 13/14(92.9%),回归测试 469/472 基线不变(3 个已知 Windows 环境失败之外零新增失败)。

### 破坏性改动

- **窗口外的工具结果占位改成三种形态**。落过盘的换成带路径的指针(可恢复才可丢弃);内容以 `error:` 开头的**不清**,只裁到 `ERROR_KEEP_TOKENS = 90`(清掉失败的观察会让模型把同一个调用再发一遍,正好撞上重复调用检测);其余仍是 `CLEARED_RESULT_MARKER`。两个新分支都排在 `run_shell` 的摘要分支之前——后者只留前三行,而指针恒在末尾。
- **最近窗口按块推进**(`RECENT_WINDOW_BLOCK = 3`)。`_blocked_recent_window()` 把「要清几条」向下取整到 3 的倍数,实际保留的工具结果在 6~8 条之间浮动。从前每轮重算意味着边界每轮往前推一格,消息数组从那个位置往后全变,前缀缓存每轮作废一次。
- **`read_file` / `write_file` / `patch_file` 回给模型的路径统一成正斜杠**。`path.relative_to(agent.root)` 在 Windows 上给反斜杠,而这串东西会被模型原样喂回 `read_file`(落盘指针正是这么用的),同一段上下文在两个平台上形状不同是最难查的一类问题。`list_files(format="paths")` 早前踩过同一个坑。
- **工具结果的入口上限从 `total_budget` 派生**。`workspace.MAX_TOOL_OUTPUT = 1320` 降级成下限,真正生效的是 `context_manager.tool_output_limit(total_budget)` = `clamp(total_budget // 8, 1320, 25000)`;`_compressed_history_entries()` 里写死的 `line_limit = 430` 取同一个值,等于取消 history 里的第二道裁剪。这两个常量是字符时代折算来的(4000÷3.02、900÷2.07)、与预算脱钩,实测把 `total_budget` 从 4,335 放大到 118,335(×27)时**发出去的 prompt 一个 token 都不变**(3,490 → 3,490),一个 5,599 token 的文件到模型眼前只剩 430 token。改后同一份 10 轮历史在 1M 档下:单条结果 430 → 5,599、history 段 2,723 → 33,794、整份 prompt 3,490 → 34,564(预算 118,335)。**输入 token 会明显上涨**,这是预算终于生效的直接后果。8k 兜底档不受影响(4335//8 = 541 < 1320 下限)。
- **掉出最近窗口的工具结果改成显式占位**。`_summarize_old_tool_item()` 从前返回这次调用**自己的签名**(`[tool:read_file] {"path":"a.py"}`)、放在一条 `role:"tool"` 消息里,读起来像「这次调用什么都没返回」;现在追加 `CLEARED_RESULT_MARKER = "[cleared to save context; re-read if needed]"`,与官方 `clear_tool_uses_20250919` 的占位同义。每条被清的记录多约 13 个 token。
- **计划转录的聚合上限跟着工具结果上限走**。`plans.MAX_TRANSCRIPT_TOKENS = 4000` 降级成下限,真正生效的是 `plans.transcript_limit(tool_output_limit)` = `max(4000, 3 × 单条上限)`,由 `runtime.execute_plan()` 写进新的 `PlanResult.transcript_limit`。倍数 3 是原设计就有的(「12000 字符 ≈ 三个普通工具调用的额度」);不跟着走的话 1M 档下单条结果能有 14,791 而整段转录仍卡在 4,000,同一次 `read_file` 放进计划里反而看得更少,`run_plan` 变成纯负收益。
- **工件里没有额度的段记 `None` 而不是 `total_budget`**。起始额度等于总预算表达的是「不设上限」,记成数字会被读成「分到了这么多」——三段加起来是总预算的三倍。只有调用方显式给了额度、或裁剪循环真的把它压下去过,`sections[*].budget_tokens` 才是数字。
- **上下文预算的单位从字符换成 token,系统里不再有第二种单位**。各段额度与下限、工具输出上限(`workspace.MAX_TOOL_OUTPUT`)、压平历史上限(`MAX_HISTORY`)、计划转录上限(`plans.MAX_TRANSCRIPT_TOKENS`)、记忆里的笔记与 task_summary 上限(`memory.NOTE_TOKENS` / `TASK_SUMMARY_TOKENS`)全部按 token 算。工件里所有 `*_chars` 字段改名 `*_tokens`,`prompt_chars` / `prompt_budget_chars` 直接删除——同时摆两种单位,读的人无从知道某个数是哪一种。
- **`total_budget` 从上下文窗口派生,不再写死**。`models.resolve_context_window()` 四层解析(显式配置 > 已知后端表 > litellm 注册表 > 保守默认 8k),向下取整到 `WINDOW_BUCKETS` 的档位;`budget_breakdown()` 再扣输出预留、工具 schema、消息结构开销,最后乘分词器容差。`HarnessSpec` 的 `total_budget` 字段含义随之变化,跨版本的评测工件不可直接比较。
- **各段固定额度取消**。`DEFAULT_SECTION_BUDGETS`(prefix 1450 / memory 520 / relevant_memory 900 / history 2500)与 `DEFAULT_SECTION_FLOORS` 删除,换成起始额度 = `total_budget`、下限 = `SECTION_FLOORS` 常量。那组数是绝对值、与 `total_budget` 脱钩,实测两个方向都错:1M 档下四段合计 5,370 只占预算(118,335)的 4.5%,裁剪循环永远触发不了、各段却照样每轮被砍(本仓库 prefix 原始 3,696 token 被裁到 1,450,白丢 61%);8k 兜底档下 5,370 是预算(4,335)的 123.9%,额度之和比总预算还大,每轮都在跑裁剪循环。`prompt_metadata` 里各段的 `budget_tokens` 随之改变含义。
- **`prefix` 完全没有额度**。从前 `PROTECTED_SECTIONS` 只挡住超预算时的**进一步**压缩,基础额度 1450 每轮照裁不误,且因为 `_tail_clip` 保留开头,丢掉的全是仓库快照那一段。现在 `section_budgets` 里传 `prefix` 会被 `ContextManager.__init__` 和 `evaluator._apply_task_setup()` 两处一起丢掉,工件里 `sections.prefix.budget_tokens` 恒为 `None`。
- **数据集 `setup` 新增 `section_floors`**,并且 `setup.section_budgets` 里的 `prefix` 会被忽略。`benchmarks/coding_tasks.json` 的 `context_reduction_checkpoint` 随之调整:它靠"裁剪真的发生"来触发 `checkpoint_created`,而常量下限(history 720)比这个 fixture 的历史(约 440 token)还大,够不着。

### 修复

- **`budget_floor_exhausted` 的判据改成由裁剪循环自己报**(走完整条 `reduction_order` 一个 section 都动不了)。旧判据查 `budgets[section] <= floor`,而各段起始额度现在是 `total_budget`,一个天然比下限还小的 section 一进循环就被跳过、额度永远不会被写回,那个查法恒为假。
- **关掉裁剪的那条路径不再拿字符数冒充 token 额度**。`_render_sections_without_reduction()` 里 `budget=len(text)` 会经 metadata 落进 `budget_tokens` 字段,单位是错的;现在一律记 `None`。
- **一次裁剪至少腾出预算的 1/10**(`CLEAR_AT_LEAST_DIVISOR = 10`)。从前 `new_budget = max(floor, current_budget - overflow)` 只裁刚好够的量,下一轮几乎必然再裁一次,而每裁一次 history 就作废它之后的全部前缀缓存。官方 context management API 为同一件事留了 `clear_at_least`(「ensures cache invalidation worthwhile」)。`budget_reductions[]` 新增 `target_tokens`,和 `overflow_tokens` 不同即说明它生效了。
- **仓库快照有了入口上限 `workspace.MAX_SNAPSHOT_TOKENS = 8000`**。prefix 现在不裁,没有这条线时一个 `git status` 刷出几千行的仓库能直接把预算吃光。正常仓库碰不到:实测本仓库整份快照 3,058 token(project_docs 2,562 / status 200 / recent_commits 129 / project_tree 104),最坏情况约 6,500。
- **预算算式里的乘性安全系数换成按消息条数的绝对扣除**。`BUDGET_SAFETY_RATIO = 0.8` 删除,改为 `MESSAGE_FRAMING_BASE_TOKENS + 20 × 预计消息条数` 加一个 5% 的 `TOKENIZER_MARGIN_RATIO`。实测 23 轮真实请求(`scripts/measure_token_accounting.py`),`input_tokens − prompt_tokens − schema` 对消息条数回归 R²=0.838、对 prompt 长度回归只有 0.396——开销跟着有几条消息走,不跟着 prompt 多长走。旧系数在 128k 档一次扔掉 25,600 token 去覆盖一笔实测约 1,200 的开销,在 8k 档又把 schema 和输出预留一起打了八折。mimo-v2.5 的预算 100,063 → 118,335。
- **预算触底不再静默**。`context_budget_tokens()` 成为 `budget_breakdown()` 的薄封装,后者返回逐项字典并带 `floored`;逐项进 `prompt_metadata["context_budget_breakdown"]` 和 `/context` 的显示。
- **`prefix` 不再参与超预算裁剪**。它曾排在裁剪顺序最后当兜底项,而实测 632 轮真实跑批里超预算的 67 轮(10.6%)**每一轮都用到了这个兜底项**。prefix 就是前缀缓存认的那段公共前缀,裁一次那轮必然落空,而省下的只有几百 token。现在由 `PROTECTED_SECTIONS` 过滤,自定义裁剪顺序也挡得住。
- **字符上限折算成 token 时按内容类型分别折算**,不再统一折半。实测字符/token 比随内容差 2.4 倍(模型写的中文笔记 1.31、工具 schema 3.95),一个除数会让一部分限额凭空放宽 1.5 倍、另一部分砍掉四成。
- **`OMITTED_DIGEST_BUDGET` 的单位对齐**:它走按 token 裁剪的 `_tail_clip`,而值还是当年的 400「字符」,等于把上限悄悄放宽约 3 倍;现按内容实测折算成 135 token。
- **计数与截断统一用 `litellm.encode`**。`token_counter()` 和 `encode()` 对同一模型名结果不一致,中文上差 40%(按 encode 裁到 290 token,用 token_counter 数只有 175),额度会被静默吃掉四成。
- **评测 fixture 复制时排除 `.codingforme` 等运行产物**(`shutil.copytree(..., ignore=...)`),上一次运行的 session 不再泄进下一次的工作区。

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
