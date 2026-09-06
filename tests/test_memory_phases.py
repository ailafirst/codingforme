"""记忆层改造(阶段一到四)的验收测试。

每个阶段两组:一组证明机制**生效**,一组把对应的消融开关关掉、证明这些断言
**恰好会挂**。只有前一组的话,「全绿」既可能是机制成立,也可能是断言什么都没查——
这份仓库在 L3 套件上踩过同一个坑(见 `test_disabling_memory_fails_exactly_the_recall_assertions`)。

施工图见 `docs/architecture/memory-implementation-spec.md`。
"""

import json
from pathlib import Path

from codingforme import CodingForMe, FakeModelClient, SessionStore, WorkspaceContext
from codingforme import memory as memorylib
from codingforme.context_manager import durable_index_limit
from codingforme.models import final_answer


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs, **kwargs):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".codingforme" / "sessions")
    return CodingForMe(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy=kwargs.pop("approval_policy", "auto"),
        **kwargs,
    )


def durable_dir(tmp_path):
    return tmp_path / ".codingforme" / "memory"


def store_for(tmp_path, **flags):
    return memorylib.DurableMemoryStore(durable_dir(tmp_path), **flags)


def seed(store, *facts, memory_type="project"):
    store.promote([(memory_type, fact) for fact in facts])


# ---------------------------------------------------------------------------
# 阶段一:中文召回
# ---------------------------------------------------------------------------


def test_a_chinese_fact_can_be_recalled_by_a_chinese_question(tmp_path):
    """改造前这条**结构上**不可能通过:`[A-Za-z0-9_]+` 对中文一个词都切不出来。"""
    store = store_for(tmp_path)
    seed(store, "构建命令是 uv run pytest。", "架构报告一律用中文写。")

    hits = store.retrieval_candidates("这个项目的构建命令是什么?", limit=3)

    assert [hit["text"] for hit in hits][:1] == ["构建命令是 uv run pytest。"]


def test_turning_cjk_recall_off_makes_the_same_chinese_fact_unreachable(tmp_path):
    """证伪证据:关掉开关,上一条断言必须挂。"""
    store = store_for(tmp_path)
    seed(store, "构建命令是 uv run pytest。")

    ascii_only = store_for(tmp_path, cjk_recall=False)

    assert ascii_only.retrieval_candidates("这个项目的构建命令是什么?", limit=3) == []


def test_the_english_subject_key_is_byte_identical_with_and_without_cjk(tmp_path):
    """分词改动不许动英文侧。

    主语键决定「新事实覆盖哪条旧事实」,它和召回共用 `_tokenize`——所以中文一开,
    英文侧必须证明自己没被顺带改掉。
    """
    for text in ("Python runtime is 3.11.", "lint runs with ruff check", "indentation is 4 spaces"):
        with_cjk = memorylib.DurableMemoryStore._subject_key(text, True)
        without = memorylib.DurableMemoryStore._subject_key(text, False)
        assert with_cjk == without


def test_chinese_facts_now_supersede_each_other(tmp_path):
    """中文事实的覆盖语义从「永不生效」变成「生效」——这是修复不是回归。"""
    store = store_for(tmp_path)
    seed(store, "缩进是 4 个空格。")

    promoted, superseded = store.promote([("project", "缩进是 2 个空格。")])

    assert promoted == ["project: 缩进是 2 个空格。"]
    assert superseded == ["project: 缩进是 4 个空格。 -> 缩进是 2 个空格。"]


# ---------------------------------------------------------------------------
# 阶段二:描述层与索引
# ---------------------------------------------------------------------------


def test_a_longer_body_does_not_change_where_a_memory_ranks(tmp_path):
    """R2 的判据:召回只读描述那一行,正文改长改短不影响名次。"""
    short = store_for(tmp_path)
    seed(short, "lint runs with ruff check. Nothing else matters here.")
    ranked_before = [hit["name"] for hit in short.retrieval_candidates("which lint command runs", limit=3)]

    entry = short._modern_entries()[0]
    entry["text"] = entry["text"] + ("\npadding sentence about unrelated deployment topics." * 40)
    short._write_entry(entry)
    ranked_after = [hit["name"] for hit in short.retrieval_candidates("which lint command runs", limit=3)]

    assert ranked_before == ranked_after != []


def test_a_keyword_that_only_lives_in_the_body_no_longer_recalls_the_memory(tmp_path):
    store = store_for(tmp_path)
    seed(store, "lint runs with ruff check. The deployment pipeline is unrelated trivia.")

    assert store.retrieval_candidates("ruff", limit=3) != []
    assert store.retrieval_candidates("deployment pipeline trivia", limit=3) == []


def test_turning_the_index_cap_off_brings_body_scoring_back(tmp_path):
    """证伪证据:同一条查询,在关掉开关的那一臂上**反向成立**。"""
    store = store_for(tmp_path)
    seed(store, "lint runs with ruff check. The deployment pipeline is unrelated trivia.")

    legacy = store_for(tmp_path, description_scoring=False)

    assert legacy.retrieval_candidates("deployment pipeline trivia", limit=3) != []


def test_the_index_is_capped_by_the_budget_and_reports_what_it_dropped(tmp_path):
    """R3 的判据:小预算下截断且**说出来**,大预算下不截断。

    静默截断在工件上和「库里就这么多」长得一模一样,而两者的应对相反——所以
    `truncated_entries` 必须是个数字,不能靠读者自己推。
    """
    store = store_for(tmp_path)
    seed(store, *[f"fact number {index} about subsystem {index}" for index in range(120)])

    tight = store.index_view(400)
    roomy = store.index_view(3000)

    assert tight["total_entries"] == 120
    assert tight["truncated_entries"] > 0
    assert tight["entries"] + tight["truncated_entries"] == 120
    assert tight["tokens"] <= 400
    assert roomy["truncated_entries"] < tight["truncated_entries"]


def test_the_index_budget_is_derived_from_the_context_budget_not_written_down():
    """上限跟着预算走。写死绝对常量正是阶段二从各段配额里删掉的那个病。"""
    assert durable_index_limit(4335) == 400          # 8k 兜底档打到下限
    assert durable_index_limit(32000) == 2000        # 中间档按 1/16 算
    assert durable_index_limit(118335) == 3000       # 1M 档打到上限


def test_the_index_reaches_the_prompt_and_its_stats_reach_the_artifact(tmp_path):
    agent = build_agent(tmp_path, [final_answer("Project convention: lint runs with ruff check.")])
    agent.ask("Please remember how lint runs.")

    agent2 = build_agent(tmp_path, [final_answer("ok")])
    agent2.ask("How does lint run?")

    prompt_text = agent2.model_client.prompts[-1]
    metadata = agent2.last_prompt_metadata["relevant_memory"]

    assert "Durable memory index:" in prompt_text
    assert "lint runs with ruff check" in prompt_text
    assert metadata["index_entries"] == 1
    # 零值也要在场:没截断是 0,不是这个字段不存在。
    assert metadata["index_truncated_entries"] == 0
    assert metadata["index_tokens"] > 0


# ---------------------------------------------------------------------------
# 阶段三:类型系统、解耦提升、价值过滤
# ---------------------------------------------------------------------------


def write_legacy_library(tmp_path):
    root = durable_dir(tmp_path)
    topics = root / "topics"
    topics.mkdir(parents=True)
    (root / "MEMORY.md").write_text(
        "# Durable Memory Index\n\n"
        "- [project-conventions](topics/project-conventions.md): Project Conventions\n"
        "  - summary: Stable repository conventions.\n"
        "  - tags: convention\n",
        encoding="utf-8",
    )
    (topics / "project-conventions.md").write_text(
        "# Project Conventions\n\n"
        "- topic: project-conventions\n"
        "- summary: Stable repository conventions.\n"
        "- tags: convention\n"
        "- updated_at: 2026-04-12T08:14:49+00:00\n\n"
        "## Notes\n"
        "- Use constrained tools instead of guessing.\n"
        "- Preserve local agent state under .codingforme/.\n",
        encoding="utf-8",
    )
    (topics / "dependency-facts.md").write_text(
        "# Dependency Facts\n\n"
        "- topic: dependency-facts\n"
        "- summary: Stable dependency facts.\n"
        "- tags: dependency\n"
        "- updated_at: 2026-04-12T08:14:49+00:00\n\n"
        "## Notes\n"
        "- The runtime depends on litellm only.\n",
        encoding="utf-8",
    )


def test_a_legacy_library_migrates_without_losing_a_single_entry(tmp_path):
    write_legacy_library(tmp_path)
    store = store_for(tmp_path)

    migrated = store.migrate_legacy()
    entries = store.entries()

    assert migrated == 3
    assert len(entries) == 3
    assert {entry["text"] for entry in entries} == {
        "Use constrained tools instead of guessing.",
        "Preserve local agent state under .codingforme/.",
        "The runtime depends on litellm only.",
    }
    # 旧主题按映射表落到类型上,不是全塞进默认类型。
    assert {entry["text"]: entry["type"] for entry in entries}["The runtime depends on litellm only."] == "reference"
    assert store.schema_version() == memorylib.DURABLE_SCHEMA_VERSION
    assert not (durable_dir(tmp_path) / "topics" / "project-conventions.md").exists()
    # 备份目录留着,这是「迁移丢数据」那条风险的处置口。
    assert list(durable_dir(tmp_path).parent.glob("memory.bak-*"))


def test_a_migrated_entry_carries_the_three_frontmatter_keys(tmp_path):
    agent = build_agent(tmp_path, [final_answer("Decision: release tag is v0.3.0.")])
    agent.ask("Please remember the release tag.")

    files = sorted((durable_dir(tmp_path) / "topics").glob("*.md"))
    text = files[0].read_text(encoding="utf-8")

    assert text.startswith("---\n")
    assert "name: " in text and "description: " in text and "metadata:" in text
    # 溯源:这条记忆能回指是哪次运行写的。
    assert f"origin_session_id: {agent.session['id']}" in text
    assert "origin_run_seq: 1" in text


def test_intent_alone_is_enough_to_promote(tmp_path):
    """R5 的一半:用户说了「记住」,模型用自己的话答的,这条以前会被静默丢弃。"""
    agent = build_agent(tmp_path, [final_answer("The deploy window is Tuesday 09:00 UTC.")])
    agent.ask("Please remember this for later sessions.")

    assert agent.last_durable_promotions == ["project: The deploy window is Tuesday 09:00 UTC."]


def test_a_prefix_line_alone_is_enough_to_promote(tmp_path):
    """R5 的另一半:用户没说「记住」,但模型给出了结构化结论。"""
    agent = build_agent(tmp_path, [final_answer("Decision: the release cadence is monthly.")])
    agent.ask("What cadence did we settle on?")

    assert agent.last_durable_promotions == ["project: the release cadence is monthly."]


def test_low_value_notes_are_rejected_and_the_reason_says_which_rule(tmp_path):
    """R6 的三条规则各一个正例。拒绝要带归因,否则调阈值时无从下手。"""
    agent = build_agent(tmp_path, [final_answer("ok")])
    (tmp_path / "CLAUDE.md").write_text(
        "本项目的评测入口是 scripts/run_eval_suite.py，它把整条链路串起来。\n",
        encoding="utf-8",
    )
    agent._doc_line_tokens = None

    assert agent.reject_durable_reason("codingforme/runtime.py") == "low_value"
    assert agent.reject_durable_reason("Fixed the off-by-one in codingforme/plans.py") == "low_value"
    assert agent.reject_durable_reason("本项目的评测入口是 scripts/run_eval_suite.py") == "low_value"
    # 反例:真正不可从代码推出的偏好照样进得去。
    assert agent.reject_durable_reason("The user prefers Chinese in architecture reports") == ""


def test_the_value_filter_never_weakens_the_safety_filters(tmp_path):
    """新增的价值维度是**加**在安全拦截上的,不是替换。"""
    agent = build_agent(tmp_path, [final_answer("ok")])

    assert agent.reject_durable_reason("API key is sk-live-secret-abc") == "secret_shaped"
    assert agent.reject_durable_reason("Current goal is fix flaky tests") == "transient_task_state"
    assert agent.reject_durable_reason("stdout: FAIL one FAIL two") == "noisy_output"


def test_turning_memory_types_off_restores_the_closed_topic_layout(tmp_path):
    """证伪证据:关掉之后,R4 / R5 / R6 三组断言必须**同时**回到改造前的样子。"""
    agent = build_agent(
        tmp_path,
        [final_answer("The deploy window is Tuesday 09:00 UTC.")],
        feature_flags={"memory_types": False},
    )
    agent.ask("Please remember this for later sessions.")

    # R5 回退:只有意图、没有前缀行 → 静默丢弃。
    assert agent.last_durable_promotions == []
    # R6 回退:价值过滤不再生效。
    assert agent.reject_durable_reason("codingforme/runtime.py") == ""

    agent2 = build_agent(
        tmp_path,
        [final_answer("Project convention: keep artifacts under artifacts/.")],
        feature_flags={"memory_types": False},
    )
    agent2.ask("Capture this convention.")

    # R4 回退:写回 4 个封闭主题的那种布局。
    assert (durable_dir(tmp_path) / "topics" / "project-conventions.md").exists()
    assert agent2.last_durable_promotions == ["project-conventions: keep artifacts under artifacts/."]


def test_a_project_memory_without_a_rationale_is_marked_incomplete(tmp_path):
    """feedback / project 要求正文带「为什么」和「什么时候适用」。

    阶段三只做校验与标记(标记进索引行),真正写这两行是写入器的事。
    """
    store = store_for(tmp_path)
    seed(store, "The deploy window is Tuesday 09:00 UTC.")

    assert store._modern_entries()[0]["incomplete"] is True
    # 标记进文件元数据和索引统计，**不**进索引行：写入器打开之前每一条都会命中，
    # 一个 100% 命中的标记不传递信息，却要在每行上花掉约 13 个 token。
    assert store.index_view(3000)["incomplete_entries"] == 1
    assert "[incomplete]" not in store.index_view(3000)["text"]


# ---------------------------------------------------------------------------
# 阶段四:写入器与整理器(默认关闭)
# ---------------------------------------------------------------------------


def test_the_extractor_writes_a_memory_without_the_user_saying_remember(tmp_path):
    """R8:一次没有任何「记住」字样的会话结束后,库里多出正确的一条。"""
    agent = build_agent(
        tmp_path,
        [
            final_answer("Understood, I will keep reports in Chinese."),
            {"text": "MEMORY: user | 用户要求架构报告用中文写 | The user wants architecture reports written in Chinese.", "tool_calls": None},
        ],
        feature_flags={"memory_extractor": True},
    )
    agent.ask("From now on write the architecture reports in Chinese, not English.")
    if agent._memory_extraction_thread is not None:
        agent._memory_extraction_thread.join(timeout=10)

    result = agent.last_memory_extraction

    assert result["status"] == "ok"
    assert result["promoted"] == ["user: The user wants architecture reports written in Chinese."]
    assert "architecture reports written in Chinese" in "\n".join(
        path.read_text(encoding="utf-8") for path in (durable_dir(tmp_path) / "topics").glob("*.md")
    )


def test_the_extractor_is_off_unless_the_variant_turns_it_on(tmp_path):
    agent = build_agent(tmp_path, [final_answer("noted")])
    agent.ask("Write the reports in Chinese from now on.")

    assert agent._memory_extraction_thread is None
    assert agent.extract_memories()["status"] == "disabled"


def test_the_extractor_skips_a_turn_that_carries_no_new_user_signal(tmp_path):
    """照抄的两个跳过条件之一:纯工具循环没有可抽取的信号。"""
    agent = build_agent(tmp_path, [final_answer("done")], feature_flags={"memory_extractor": True})
    agent.session["history"] = []
    agent.session["memory_extraction_cursor"] = 0

    assert agent.extract_memories()["skipped_reason"] == "no_new_user_signal"


def test_the_extractor_output_still_goes_through_the_write_filters(tmp_path):
    """写入器不是绕过写入拦截的旁路。"""
    agent = build_agent(
        tmp_path,
        [{"text": "MEMORY: reference | key | API key is sk-live-secret-abc", "tool_calls": None}],
        feature_flags={"memory_extractor": True},
    )
    agent.session["history"] = [{"role": "user", "content": "please note the credentials we discussed"}]

    result = agent.extract_memories(force=True)

    assert result["promoted"] == []
    assert result["rejections"] == ["reference:secret_shaped"]


def test_consolidation_dedupes_and_marks_stale_without_deleting(tmp_path):
    """R7:同一件事写两遍只剩一条;点名的文件不见了是**标记**不是删除。"""
    store = store_for(tmp_path)
    store.promote([("project", "The parser lives in gone.py and handles retries.")])
    # 两条不同措辞、同一个主语 → 整理时算重复。
    store.promote([("reference", "The runtime is 3.11.")])
    store.promote([("reference", "The runtime is 3.12.")])

    result = store.consolidate(workspace_root=tmp_path)
    texts = {entry["text"] for entry in store.entries()}

    assert result["ran"] is True
    assert "The runtime is 3.11." not in texts
    assert "The runtime is 3.12." in texts
    # 过期的那条还在,只是被标出来了。
    assert "The parser lives in gone.py and handles retries." in texts
    assert result["marked_stale"]
    assert "[stale]" in store.index_view(3000)["text"]


def test_the_consolidation_gate_counts_sessions_before_it_runs(tmp_path):
    agent = build_agent(tmp_path, [final_answer("ok")], feature_flags={"memory_consolidator": True})
    store_for(tmp_path).promote([("project", "keep artifacts under artifacts/ for later comparison")])

    first = agent.consolidate_memory()

    assert first["status"] == "gated"
    assert first["sessions_since"] == 1


def test_only_one_process_consolidates_at_a_time(tmp_path):
    """跨进程锁:两个会话同时过闸门时只有一个真的跑。"""
    agent = build_agent(tmp_path, [final_answer("ok")], feature_flags={"memory_consolidator": True})
    store = store_for(tmp_path)
    store.promote([("project", "keep artifacts under artifacts/ for later comparison")])
    lock = durable_dir(tmp_path) / ".consolidate-lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("held", encoding="utf-8")

    result = agent.consolidate_memory(force=True)

    assert result["status"] == "locked"
    assert result["ran"] is False


def test_the_consolidator_is_off_unless_the_variant_turns_it_on(tmp_path):
    agent = build_agent(tmp_path, [final_answer("ok")])

    assert agent.consolidate_memory()["status"] == "disabled"


def test_the_memory_artifacts_carry_zero_values_too(tmp_path):
    """空库时字段照写。省略会被读成「这块没问题」。"""
    agent = build_agent(tmp_path, [final_answer("nothing to remember here")])
    agent.ask("Just answer, do not store anything.")

    report = json.loads(agent.run_store.report_path(agent.current_task_state).read_text(encoding="utf-8"))

    assert report["durable_index"] == {
        "entries": 0,
        "total_entries": 0,
        "truncated_entries": 0,
        "incomplete_entries": 0,
        "tokens": 0,
    }
    assert Path(agent.run_store.report_path(agent.current_task_state)).exists()
