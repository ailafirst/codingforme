import os
import shlex
import sys
from unittest.mock import patch

from codingforme import FakeModelClient, CodingForMe, SessionStore, WorkspaceContext
from codingforme import cli as mini_cli
from codingforme.models import final_answer, tool_call
from codingforme.task_state import TaskState


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs, **kwargs):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".codingforme" / "sessions")
    approval_policy = kwargs.pop("approval_policy", "auto")
    return CodingForMe(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy=approval_policy,
        **kwargs,
    )


def test_workspace_escape_is_rejected(tmp_path):
    (tmp_path / "outside.txt").write_text("outside\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("read_file", {"path": "../outside.txt"})

    assert "path escapes workspace" in result


def test_symlink_path_traversal_is_rejected(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (tmp_path / "linked.txt").symlink_to(outside)
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("read_file", {"path": "linked.txt"})

    assert "path escapes workspace" in result


def test_risky_tool_deny_behavior(tmp_path):
    """被拒的调用要返回错误字符串而不是抛异常，且不得留下任何执行痕迹。

    断言从「等于某一句话」放宽成「是这次拒绝、且不是成功」，是刻意的：这条用例
    守的是**拒绝这件事**，不是措辞。措辞是要随证据演进的——只读态下模型此前会
    换用 run_shell 往同一个文件 echo，所以现在的文案要说清"别换个工具再试"。
    """
    agent = build_agent(tmp_path, [], approval_policy="never")

    result = agent.run_tool("run_shell", {"command": "echo hi", "timeout": 20})

    assert result.startswith("error: approval denied for run_shell")
    assert agent._last_tool_result_metadata["tool_status"] == "rejected"
    assert agent._last_tool_result_metadata["workspace_changed"] is False


def test_read_only_denial_says_it_is_read_only_and_not_to_reroute(tmp_path):
    """只读态的拒绝要说明原因是只读，并明说别换个工具重试。

    证据：实测 patch_file 被只读态挡住之后，模型改用 run_shell 往同一个文件
    echo——它把"这次没批准"读成了"换个说法可能行"。这两件事必须在文案里分开。
    """
    agent = build_agent(tmp_path, [], approval_policy="never", read_only=True)

    result = agent.run_tool("write_file", {"path": "a.txt", "content": "x"})

    assert "read-only" in result
    assert "another tool" in result
    assert not (tmp_path / "a.txt").exists()


def test_cli_build_agent_wires_secret_env_names_from_parser(tmp_path):
    class DummyModelClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def complete(self, prompt, max_new_tokens):
            raise AssertionError("model should not be invoked")

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    with patch.dict(os.environ, {"GITHUB_PAT": "ghp-1", "GH_PAT": "ghp-2"}, clear=True), patch(
        "codingforme.cli.OpenAICompatibleModelClient",
        DummyModelClient,
    ):
        args = mini_cli.build_arg_parser().parse_args(
            [
                "--cwd",
                str(tmp_path),
                "--approval",
                "auto",
                "--secret-env-name",
                "GITHUB_PAT",
                "--secret-env-name",
                "GH_PAT",
            ]
        )
        agent = mini_cli.build_agent(args)
        assert set(agent.secret_env_summary()["secret_env_names"]) == {"GITHUB_PAT", "GH_PAT"}


def test_cli_build_agent_uses_default_configured_secret_names(tmp_path):
    class DummyModelClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def complete(self, prompt, max_new_tokens):
            raise AssertionError("model should not be invoked")

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    with patch.dict(os.environ, {"GH_PAT": "ghp-default-1"}, clear=True), patch(
        "codingforme.cli.OpenAICompatibleModelClient",
        DummyModelClient,
    ):
        args = mini_cli.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--approval", "auto"])
        agent = mini_cli.build_agent(args)
        assert agent.secret_env_summary()["secret_env_names"] == ["GH_PAT"]



def test_cli_build_agent_reads_secret_names_from_environment_config(tmp_path):
    class DummyModelClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def complete(self, prompt, max_new_tokens):
            raise AssertionError("model should not be invoked")

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    with patch.dict(
        os.environ,
        {
            "MCA_CUSTOM_SECRET": "custom-secret-value",
            "CODINGFORME_SECRET_ENV_NAMES": "MCA_CUSTOM_SECRET",
        },
        clear=True,
    ), patch("codingforme.cli.OpenAICompatibleModelClient", DummyModelClient):
        args = mini_cli.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--approval", "auto"])
        agent = mini_cli.build_agent(args)
        assert agent.secret_env_summary()["secret_env_names"] == ["MCA_CUSTOM_SECRET"]


def test_run_shell_uses_allowlisted_environment_only(tmp_path):
    secret = "shh-allowlist-secret"
    agent = build_agent(tmp_path, [], approval_policy="auto")
    script = 'import os; print(os.getenv("MCA_ALLOWLIST_SECRET", "missing"))'
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    with patch.dict(os.environ, {"MCA_ALLOWLIST_SECRET": secret}, clear=False):
        result = agent.run_tool("run_shell", {"command": command, "timeout": 20})

    assert secret not in result
    assert "missing" in result


def test_bound_tool_methods_delegate_into_tools_module(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="auto")

    with patch("codingforme.tools.subprocess.run") as fake_run:
        fake_run.return_value = type(
            "Result",
            (),
            {"returncode": 0, "stdout": "toolkit-shell\n", "stderr": ""},
        )()
        shell_result = agent.tool_run_shell({"command": "echo bypass", "timeout": 20})

    assert "toolkit-shell" in shell_result
    fake_run.assert_called_once()
    assert agent.tool_run_shell.__func__.__module__ == "codingforme.runtime"

    with patch("codingforme.tools.tool_delegate", return_value="toolkit-delegate") as fake_delegate:
        delegate_result = agent.tool_delegate({"task": "inspect README.md", "max_steps": 2})

    assert delegate_result == "toolkit-delegate"
    fake_delegate.assert_called_once()


def test_delegate_depth_limit_is_enforced(tmp_path):
    agent = build_agent(tmp_path, [], depth=1, max_depth=1)

    try:
        agent.validate_tool("delegate", {"task": "inspect README.md", "max_steps": 2})
    except ValueError as exc:
        assert "delegate depth exceeded" in str(exc)
    else:
        raise AssertionError("delegate depth validation did not fail")


def test_delegate_child_is_read_only(tmp_path):
    target = tmp_path / "child-was-not-allowed.txt"
    agent = build_agent(
        tmp_path,
        [
            tool_call("delegate", task="write a file", max_steps=2),
            tool_call("write_file", path="child-was-not-allowed.txt", content="nope"),
            final_answer("child done"),
            final_answer("parent done"),
        ],
    )

    result = agent.ask("Delegate the work")

    assert result == "parent done"
    assert not target.exists()
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert tool_events[0]["name"] == "delegate"
    assert "delegate_result" in tool_events[0]["content"]


def test_configured_secret_env_names_are_redacted_in_trace_and_report(tmp_path):
    github_pat = "ghp_configured_secret_123"
    gh_pat = "ghp_configured_secret_456"
    with patch.dict(os.environ, {"GITHUB_PAT": github_pat, "GH_PAT": gh_pat}, clear=True):
        agent = build_agent(
            tmp_path,
            [],
            secret_env_names=("GITHUB_PAT", "GH_PAT"),
        )
        state = TaskState.create(run_id="run_001", task_id="task_001", user_request="Mask configured secrets")
        agent.run_store.start_run(state)

        assert set(agent.secret_env_summary()["secret_env_names"]) == {"GITHUB_PAT", "GH_PAT"}

        payload = {
            "GITHUB_PAT": github_pat,
            "GH_PAT": gh_pat,
            "nested": {"GITHUB_PAT": github_pat, "GH_PAT": gh_pat},
            "list": [github_pat, gh_pat],
        }
        agent.emit_trace(state, "tool_executed", payload)
        agent.run_store.write_report(
            state,
            agent.redact_artifact({"task_state": state.to_dict(), "payload": payload}),
        )

    run_dir = agent.run_store.run_dir(state.run_id)
    trace_text = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
    report_text = (run_dir / "report.json").read_text(encoding="utf-8")

    assert github_pat not in trace_text
    assert gh_pat not in trace_text
    assert github_pat not in report_text
    assert gh_pat not in report_text
    assert trace_text.count("<redacted>") >= 4
    assert report_text.count("<redacted>") >= 4


def test_captured_subprocess_output_survives_non_utf8_locale(tmp_path):
    """子进程输出必须显式按 UTF-8 解码，否则捕获到的是 None 而不是异常。

    `capture_output=True` 的读取发生在子线程里，那里抛出的 UnicodeDecodeError
    传不回主线程——`result.stdout` 会**静默变成 None**，下一句 `.strip()` 报
    "'NoneType' object has no attribute 'strip'"。这比 `capture_output=False`
    时直接抛异常隐蔽得多：k=3 真实跑批里它吃掉了 4 次调用，报错文本看不出病因。

    这里直接查调用参数而不是造一个 GBK 环境：宿主代码页不是测试能可靠控制的东西，
    而"有没有显式指定编码"恰恰是这个 bug 的充要条件。
    """
    import inspect

    from codingforme import tools as toolkit

    for func in (toolkit.tool_run_shell, toolkit.tool_search):
        source = inspect.getsource(func)
        assert 'encoding="utf-8"' in source, f"{func.__name__} 没有显式指定编码"
        assert 'errors="replace"' in source, f"{func.__name__} 没有指定解码错误策略"
        # 即便如此也要能扛住 None：显式编码是防御，容错是兜底。
        assert "result.stdout.strip()" not in source, f"{func.__name__} 直接对 stdout 调了 .strip()"
        assert "result.stderr.strip()" not in source, f"{func.__name__} 直接对 stderr 调了 .strip()"


def test_refreshing_the_prefix_never_widens_the_agent_root(tmp_path):
    """刷新工作区快照不许把 agent 的根目录放大。

    踩过的坑：`refresh_prefix()` 用 `WorkspaceContext.build(self.root)` 重建快照，
    没传 `repo_root_override`。那个参数缺席时 `build()` 会用
    `git rev-parse --show-toplevel` 现算仓库根——于是一个**刻意被限定在子目录里**
    的 workspace（评测把每个任务的样板仓库复制到某个目录、再用 override 限定在
    那份拷贝上）会在第一次刷新时被悄悄放大成整个外层 git 仓库。

    两个后果，第二个是安全问题：
    1. 快照里混进外层仓库的 AGENTS.md / README.md / pyproject.toml 和 git status，
       prefix 撑爆段预算，样板仓库自己的快照和 resume checkpoint 被截掉。
    2. `delegate` 的子 agent 用父 agent 的 workspace 构造，而 `CodingForMe.__init__`
       里 `self.root = Path(workspace.repo_root)`——放大后的 repo_root 会让**子
       agent 拿到比父 agent 更宽的根目录**。子 agent 的权限只应更小。所以下面
       断言 `workspace.repo_root` 不变，等价于断言子 agent 拿不到更宽的根。

    这里用一个真的 git 仓库套一个子目录 workspace 来复现：不套 git 仓库的话
    `show-toplevel` 会失败并回落到 cwd，恰好掩盖这个 bug。
    """
    import subprocess

    outer = tmp_path / "outer"
    (outer / "inner").mkdir(parents=True)
    (outer / "SECRET-OUTSIDE.md").write_text("outer repo file\n", encoding="utf-8")
    (outer / "inner" / "README.md").write_text("inner workspace\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=outer, check=True, capture_output=True)

    inner = outer / "inner"
    agent = CodingForMe(
        model_client=FakeModelClient([]),
        workspace=WorkspaceContext.build(inner, repo_root_override=inner),
        session_store=SessionStore(inner / ".codingforme" / "sessions"),
        approval_policy="auto",
    )
    root_before = os.path.realpath(str(agent.root))

    agent.refresh_prefix(force=True)

    assert os.path.realpath(str(agent.root)) == root_before, "agent 的根目录在刷新之后变了"
    assert os.path.realpath(str(agent.workspace.repo_root)) == os.path.realpath(str(inner)), (
        "快照的 repo_root 被放大到了外层仓库；delegate 的子 agent 会继承这个更宽的根"
    )
    assert "SECRET-OUTSIDE.md" not in agent.prefix, "外层仓库的文件泄进了 prefix"
