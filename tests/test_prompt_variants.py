"""提示词变体：让「换了措辞」和「关掉记忆」一样能跑出对照。

为什么需要这一层：在它之前 `PROMPT_TEMPLATE` 是模块级常量，改一句措辞只让
`prompt_template_signature()` 换个值——两批跑批从此不可比，**却没有任何办法把
新旧两版放进同一张对照表**。实测这个退化有多严重：15 个固定任务里能被措辞影响
的只有 6 次适用判定（`no_redundant_reread`），k=1 下 4/6 → 2/6，Fisher 精确检验
p=0.57，读不出任何因果。

一版提示词是**模板 + 各工具描述后面追加的那句话**（阶段三之后 per-tool 的用法
规则住在工具描述里）。三版：

    current      阶段三：per-tool 规则在工具描述里，规则块跨任务逐字节相同
    pre_phase3   阶段一之后、阶段三之前：per-tool 规则还在规则块里，两条要现算
    pre_phase1   最早那版：防重复按「同一组参数」判、禁令没有出口、记忆规则在骗模型

这份测试锁四件事：每一版都能渲染干净、三版真的是三份不同的东西、名字写错会当场
炸、以及变体真的改变了发给模型的内容并且进了指纹与工件。
"""

import pytest

from codingforme.eval.harness import BUILTIN_HARNESS_SPECS, HarnessSpec, get_harness
from codingforme.models import FakeModelClient, final_answer
from codingforme.runtime import (
    DEFAULT_PROMPT_VARIANT,
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_PRE_PHASE1,
    PROMPT_TEMPLATE_PRE_PHASE3,
    PROMPT_VARIANTS,
    CodingForMe,
    SessionStore,
    get_prompt_template,
    get_tool_guidance,
    prompt_template_signature,
)
from codingforme.workspace import WorkspaceContext


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    return tmp_path


def build_agent(tmp_path, harness):
    return harness.build(FakeModelClient([final_answer("done")]), tmp_path)


def test_every_variant_renders_without_leaving_a_placeholder_behind(tmp_path):
    """每一版都必须能渲染干净。

    模板里的占位符靠 `str.replace()` 填，而 replace 找不到目标时**什么都不做、也
    不报错**。所以两种故障都要挡：占位符没被替换（模型看到一行 `__WRITE_RULE__`），
    以及某一版少写了 `__TOOL_TEXT__` / `__WORKSPACE_TEXT__`（工具清单或仓库快照
    整段消失，而两臂差的就不再只是措辞了）。
    """
    build_workspace(tmp_path)
    for name in sorted(PROMPT_VARIANTS):
        agent = build_agent(tmp_path, get_harness("full").derive(prompt_variant=name))
        assert "__" not in agent.prefix, f"{name} 渲染后仍有占位符"
        assert "Tools:" in agent.prefix and "Workspace:" in agent.prefix, name
        # 每一版都不许教文本协议——这条不变量对全部变体成立，不只是当前那份。
        assert "<tool>" not in agent.prefix and "<final>" not in agent.prefix, name


def test_the_three_arms_are_actually_three_different_prompts():
    """三臂要真的互不相同，而且不同点就是三个阶段各自改的那几处。"""
    templates = {PROMPT_TEMPLATE, PROMPT_TEMPLATE_PRE_PHASE3, PROMPT_TEMPLATE_PRE_PHASE1}
    assert len(templates) == 3
    signatures = {prompt_template_signature(name) for name in PROMPT_VARIANTS}
    assert len(signatures) == len(PROMPT_VARIANTS)

    # 阶段一之前：防重复按「同一组参数」判、禁令没有出口、记忆规则宣称只有前缀行才算数。
    assert "the same tool call with the same arguments" in PROMPT_TEMPLATE_PRE_PHASE1
    assert "only lines in that exact shape are kept long-term" in PROMPT_TEMPLATE_PRE_PHASE1
    assert "If no available tool fits" not in PROMPT_TEMPLATE_PRE_PHASE1

    # 阶段一之后：按目标判并附理由、补上出口、去掉那句已经不成立的断言。
    assert "does not need to be fetched again" in PROMPT_TEMPLATE_PRE_PHASE3
    assert "If no available tool fits" in PROMPT_TEMPLATE_PRE_PHASE3
    assert "only lines in that exact shape are kept long-term" not in PROMPT_TEMPLATE_PRE_PHASE3

    # 阶段三：per-tool 规则搬走了，规则块里不再有占位符要现算，防重复那句挪进了
    # read_file 的描述。
    assert "__WRITE_RULE__" not in PROMPT_TEMPLATE
    assert "__REQUIRED_ARGS_RULE__" not in PROMPT_TEMPLATE
    assert "does not need to be fetched again" not in PROMPT_TEMPLATE
    assert "reading it again costs a full turn" in get_tool_guidance("current")["read_file"]
    assert get_tool_guidance("pre_phase3") == {}
    assert get_tool_guidance("pre_phase1") == {}


def test_an_unknown_variant_blows_up_instead_of_falling_back(tmp_path):
    """静默退回默认模板 = 整批跑批看起来跑了对照臂、实际两臂完全相同。"""
    with pytest.raises(ValueError):
        get_prompt_template("nope")
    with pytest.raises(ValueError):
        HarnessSpec(name="typo", prompt_variant="pre-phase1")   # 连字符不是下划线

    build_workspace(tmp_path)
    with pytest.raises(ValueError):
        CodingForMe(
            model_client=FakeModelClient([final_answer("done")]),
            workspace=WorkspaceContext.build(tmp_path, repo_root_override=tmp_path),
            session_store=SessionStore(tmp_path / ".codingforme" / "sessions"),
            prompt_variant="nope",
        )


def test_the_variant_changes_what_the_model_actually_sees(tmp_path):
    """指纹变了但发出去的文本没变，等于一个不做事的对照臂。"""
    build_workspace(tmp_path)
    current = build_agent(tmp_path, get_harness("full"))
    phase1 = build_agent(tmp_path, get_harness("prompt_pre_phase3"))
    legacy = build_agent(tmp_path, get_harness("prompt_pre_phase1"))

    # 防重复那句话三臂各在一个地方：工具描述里 / 规则块里 / 完全是另一种措辞。
    assert "reading it again costs a full turn" in current.prefix.split("Tools:")[1]
    assert "does not need to be fetched again" in phase1.prefix.split("Tools:")[0]
    assert "the same tool call with the same arguments" in legacy.prefix.split("Tools:")[0]

    # 阶段三的规则块里一个工具名都没有；另外两版靠现算，所以有。
    assert "list_files" not in current.prefix.split("Tools:")[0]
    assert "patch_file" in phase1.prefix.split("Tools:")[0]

    for agent in (current, phase1, legacy):
        assert "approval required" in agent.prefix


def test_the_variant_lands_in_the_fingerprint_and_in_the_artifact(tmp_path):
    """跑完之后要能从工件上读出这一批用的是哪一版提示词。"""
    full = get_harness("full")
    legacy = get_harness("prompt_pre_phase1")

    assert full.fingerprint() != legacy.fingerprint()
    assert full.code_signature() != legacy.code_signature()
    assert full.code_signature_parts()["prompt_variant"] == "current"
    assert legacy.code_signature_parts()["prompt_variant"] == "pre_phase1"
    assert full.to_dict()["prompt_variant"] == DEFAULT_PROMPT_VARIANT
    assert HarnessSpec.from_dict(legacy.to_dict()).prompt_variant == "pre_phase1"

    workspace = build_workspace(tmp_path)
    agent = legacy.build(FakeModelClient([final_answer("done")]), workspace)
    agent.ask("say hi")
    metadata = agent.last_prompt_metadata
    assert metadata["prompt_variant"] == "pre_phase1"
    assert metadata["prompt_template_signature"] == prompt_template_signature("pre_phase1")
    assert metadata["prompt_template_signature"] != prompt_template_signature("current")


def test_the_signature_covers_the_tool_guidance_not_just_the_template(monkeypatch):
    """只哈希模板的话，两版只差工具描述附文时会算出同一个签名。

    阶段三之后 per-tool 规则住在附文里，所以那种「签名相同、发给模型的内容不同」
    正是这条轴要消灭的故障——工件上两批数据看起来可比，实际不是。
    """
    from codingforme import runtime

    base = prompt_template_signature("current")
    monkeypatch.setattr(
        runtime,
        "PROMPT_VARIANTS",
        {
            **runtime.PROMPT_VARIANTS,
            "current": runtime.PromptVariant(runtime.PROMPT_TEMPLATE, {"read_file": " Extra."}),
        },
    )

    assert prompt_template_signature("current") != base


def test_the_two_arms_differ_in_exactly_one_field():
    """对照臂只许换提示词。多换一个字段，A/B 测的就是两件事叠在一起。"""
    full = dict(get_harness("full").to_dict())
    legacy = dict(get_harness("prompt_pre_phase3").to_dict())
    for payload in (full, legacy):
        payload.pop("name")
        payload.pop("description")
    differing = {key for key in full if full[key] != legacy[key]}
    assert differing == {"prompt_variant"}, differing


def test_every_builtin_harness_names_a_known_prompt_variant():
    """`__post_init__` 的守卫要覆盖内建表本身，而不只是外部调用方。"""
    for name, spec in BUILTIN_HARNESS_SPECS.items():
        assert spec.prompt_variant in PROMPT_VARIANTS, name
