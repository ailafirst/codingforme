# coding-for-me

> 把一个会写代码的助手,装进你的仓库里——它在终端里,看得见你的文件,改得动你的代码。

![coding-for-me](assets/screenshots/coding-for-me-start.png)

这就是 `coding-for-me` 的全部体验:一个提示符,一句话,它去读、去改、去跑,然后把结果摆在你面前。没有 IDE 插件,没有网页,没有上下文丢失——只有终端和你的仓库。

---

## 它做这些

- **修测试。** 你说哪个挂了,它去看 traceback、定位、改代码、再跑一遍。
- **改现有代码。** 基于仓库里真实存在的文件做小步迭代,而不是凭空生成一段你还得自己拼进去的片段。
- **读懂一个陌生仓库。** 进去转一圈,告诉你结构、入口、关键约定在哪。
- **跑一次性任务。** 重命名、补类型、改配置——一条命令,一次性搞定。
- **记住上下文。** 关掉再回来,`--resume latest` 接着上次继续。

生成过程是流式的:模型还在想的时候,你就能看到字一个个出来,不用对着空屏幕等。

## 它不做这些

- 不是聊天框。它的每句话背后都对应一次真实的文件读写或命令执行。
- 不偷偷跑命令。`run_shell` / `write_file` / `patch_file` 默认每一步都问过你才动手。
- 不绑定某一家模型。任何 OpenAI 兼容的 `/chat/completions` 端点都能接。
- 不拖一堆依赖。运行时只有 `litellm` 一个依赖(为了标准 function-calling 和流式),其余全是标准库。

---

## 三步上手

```bash
# 1. 装(需要 Python 3.10+)
uv sync                 # 或者 pip install -e .

# 2. 配一个模型
cp .env.example .env    # 然后把下面三行填进去

# 3. 在仓库里开跑
uv run coding-for-me
```

`.env` 里要填的三行:

```bash
CODINGFORME_OPENAI_API_BASE="https://your-api.example/v1"
CODINGFORME_OPENAI_API_KEY="your-api-key"
CODINGFORME_OPENAI_MODEL="your-model"
```

还有两个可选开关,平时不用管——只在你的后端行为特殊时才需要:

```bash
CODINGFORME_OPENAI_NATIVE_TOOL_CALLS=0   # 后端不吃标准 tools= 参数时关掉
CODINGFORME_OPENAI_PROMPT_CACHE_KEY=1    # 后端认 prompt_cache_key 字段时打开
CODINGFORME_CONTEXT_WINDOW=131072        # 后端的上下文窗口(token),自动查不到时才需要填
```

> 用小米 MiMo?把 base 换成 `https://token-plan-cn.xiaomimimo.com/v1`、模型填 `mimo-v2.5` 就行。
> 谁都不填时,默认指向 `https://www.right.codes/codex/v1` 的 `gpt-5.4`。
> 优先级:命令行 `--model` / `--base-url` > `.env` 里的 `CODINGFORME_OPENAI_*` > 裸 `OPENAI_*` > 代码默认。

## 几种用法

```bash
uv run coding-for-me                         # 交互模式(REPL)
uv run coding-for-me "把 README 里的死链修掉" # 说完就跑,跑完就退
uv run coding-for-me --cwd /path/to/repo     # 对另一个仓库下手
uv run coding-for-me --resume latest         # 接着上次的会话
python -m codingforme                        # 等价的模块入口
```

进了 REPL,这几个命令随时可用:`/help` `/context`(看/改上下文窗口档位)`/memory`(看它记住了什么)`/session`(会话文件在哪)`/reset`(清空重来)`/exit`。

## 想再拧几个旋钮

| 旋钮 | 干什么 | 默认 |
|---|---|---|
| `--approval` | 高风险工具是 `ask` 逐个问 / `auto` 自动 / `never` 不许 | `ask` |
| `--max-new-tokens` | 模型**每一步**最多吐多少 token | `512` |
| `--max-steps` | 一次请求里最多迭代几轮工具 | `6` |
| `--temperature` | 采样温度 | `0.2` |
| `--model` / `--base-url` | 临时换模型 / 换端点,不动 `.env` | 取自 `.env` |
| `--resume` | 会话 id,或 `latest` | 无 |
| `--context-window` | 后端上下文窗口(token),覆盖自动探测 | 自动 |

> 回答被截断了?多半是 `--max-new-tokens` 默认 512 偏小——它是单步硬上限。写长代码时调到 `2048` 试试。

> 上下文窗口一般不用管:自己会去查(已知后端表 → litellm 的模型注册表 → 保守默认 8k),
> 查到的值向下取整到 8k/16k/32k/64k/128k/256k/512k/1M 某一档再用。只有在用一个自建的、
> 冷门的后端、而且日志里显示回落到了默认档时,才需要手动填 `--context-window`。
> **填大了比填小了危险**:小了只是少带点历史,大了会撞后端报错、丢掉整轮已经做完的工具执行。

> 不想重启也能改:REPL 里 `/context` 看当前档位、来源和预算是怎么算出来的,
> `/context 128k` 直接换档(`128000`、`1m` 一样认,不在档位上的值向下取整并告诉你取了)。
> 窗口特别大的后端(比如 mimo-v2.5 官方标 1M)尤其用得上——自动探测出来的 1M
> 不是该直接用的数,预算无论如何都会先夹到 128k 上限,再乘安全系数、扣掉工具 schema
> 和输出预留。

---

## 幕后:它信不过自己

这个 agent 的设计前提是「模型会犯错」,所以护栏不在模型那边,在平台这边:

- **危险操作要过闸。** 所有工具调用都先经过一个总闸口,审批策略说了不算就是不算;`patch_file` 还要求 `old_text` 在文件里唯一匹配,改不准就拒绝。
- **子任务只读。** 它派生出去的子 agent 一律只读、不许审批、步数预算更小——能看不能动。
- **环境是过滤过的。** `run_shell` 只拿到一份白名单环境变量,密钥名会被脱敏,不会把你的完整 env 漏给子进程。
- **工具调用走标准协议。** 用的是 OpenAI 标准的 function-calling:工具的参数被翻译成 JSON Schema 一并发给模型,而不是让它在自由文本里"写"一段调用格式。少一层字符串解析,就少一类模型能把调用写歪的方式——实测缺必填参数的调用在协议层就发不出来。
- **一切留痕。** 每次运行都在 `.codingforme/runs/<run_id>/` 落下 `trace.jsonl`(逐事件追加,跑一半也能看)、`task_state.json`、`report.json`;写进去之前,密钥的值已经换成 `<redacted>`。

会话本身存在 `.codingforme/sessions/`,这些本地产物都被 `.gitignore` 挡在仓库之外。

有意思的一点:**`coding-for-me` 的开发对象就是它自己。** 仓库里大量测试,是把这个 agent 放进 `tests/fixtures/` 的样板仓库里跑出来的。

## 改它 / 测它

```bash
uv run pytest            # 全量测试
uv run ruff check .      # lint
```

源码里的注释是中文,标识符和 CLI 文案是英文——动手改的时候,跟着这个习惯走。
