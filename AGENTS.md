# Repository Guidelines

## Project Structure & Module Organization

`codingforme/` contains the application package. The CLI entry point is
`codingforme/cli.py`; `runtime.py` coordinates the agent loop, while `tools.py`,
`workspace.py`, and `models.py` implement guarded tools, workspace access, and
provider communication. `plans.py` holds the restricted orchestration
interpreter behind the optional `run_plan` tool: it parses with `ast` and walks
the tree itself (never `exec`), allows a whitelist of node types with no
attribute access, and can only call tools already in the agent's registry.
Evaluation support lives in `codingforme/eval/`.

Tests are in `tests/`, with sample repositories under `tests/fixtures/`.
Benchmark definitions live in `benchmarks/`; runnable experiment helpers are in
`scripts/`. Keep generated run/session data inside `.codingforme/` (ignored by
Git), not in source directories.

## Build, Test, and Development Commands

Run commands from the repository root:

```bash
uv sync                              # Install Python 3.10+ dependencies
uv run coding-for-me                 # Start the interactive CLI
uv run pytest                        # Run the full test suite
uv run pytest tests/test_memory.py   # Run one test module
uv run pytest -k "patch_file"         # Select tests by name
uv run ruff check .                  # Run linting
```

`pip install -e .` and `python -m pytest` are suitable fallbacks when `uv` is
unavailable. Root execution matters because some tests use repository-relative
benchmark paths.

## Coding Style & Naming Conventions

Use four-space Python indentation, standard-library-first imports, and clear
`snake_case` names for functions, variables, and modules; use `PascalCase` for
classes. Keep source comments in Chinese, but keep identifiers and CLI text in
English, matching the existing codebase. Run `ruff check .` before submitting.

Keep runtime dependencies minimal: `litellm` is the only production dependency.
Route filesystem paths through `CodingForMe.path()` and tool requests through
`run_tool()` so path containment, approval, validation, and trace behavior stay
intact. That applies to calls made from inside a `run_plan` program too: each
one goes back through `run_tool()`, costs its own step, and emits its own
`tool_executed` trace event. Redact secrets before writing any new artifact.

What a `run_plan` program sends back to the model is capped and filtered. When
the plan's source statically contains a `print(` call, tool results stay out of
the reply and only printed output is returned; otherwise every result is echoed
in full. Either way the transcript is clipped to `plans.MAX_TRANSCRIPT_CHARS`,
and the per-call listing line is always written -- an inner call must never
vanish silently from what the model reads.

In that filtered mode the printed output and the call listing go in two separate
labelled blocks, never interleaved, and the reply states that the printed output
is complete. Both rules exist because the interleaved first version read as
truncated output and made the model re-run whole fan-outs until it ran out of
steps.

Evaluation results must record which tools actually ran. `eval/tool_usage.py`
summarizes per-tool call counts and `run_plan` context savings from the trace
index, and reports zeros with an explicit "never executed" line rather than
omitting the section -- an omitted section reads as "nothing wrong here".
Trajectory assertion failures must carry enough detail to locate the offending
call (arguments, not just a turn number).

`list_files` and `search` accept `format="paths"`, which returns one bare workspace-relative path per line (forward slashes on every platform) instead of a display format a plan would have to take apart. Keep both search implementations -- ripgrep and the pure-Python fallback -- emitting the same shape, and reject an unknown `format` rather than silently falling back to the default.

Prose rules in a tool description do not change how the model writes plans; the worked example in `_PLAN_EXAMPLES` does. Two live runs measured the same 62% plan rejection rate, and most remaining rejections were the exact method-call form the description told the model not to use. Demonstrate a helper in the example rather than describing it.

Which helper functions the plan sandbox needs is decided from rejected plans in live runs, not from guesswork -- and adding a helper is only half the fix, because the model writes the method form until the tool description spells out the function form (`startswith(row, x)`, not `row.startswith(x)`). Report plan attempts, executions and rejections separately: a rejected plan still burns a model round trip, and a single combined count hides that.

`run_plan`'s tool description is re-sent on every single turn, so its length is itself a cost: keep only sentences stating something the model cannot guess and has been observed getting wrong. Changes to it must be validated against a real model, not just unit tests.

Any message that names a specific tool -- prompt rules, validation errors,
retry notices -- must be built from `agent.tools` at runtime rather than
hard-coded. Variant and task allowlists trim the registry, so a hard-coded
sentence can point the model at a tool it cannot call.

## Testing Guidelines

Add focused `pytest` coverage for behavior changes. Name test files
`test_<area>.py` and tests `test_<expected_behavior>`. Prefer
`FakeModelClient` and `tmp_path` to external providers or real workspaces.
For changes affecting tools, paths, approvals, or environment handling, run
`uv run pytest tests/test_safety_invariants.py` in addition to relevant tests.

## Commit & Pull Request Guidelines

Follow the established concise Conventional Commit pattern, such as
`feat: add session report` or `fix: redact artifact values`. Keep commits
single-purpose. Pull requests should explain the behavior change, list tests
run, link relevant issues, and include CLI output or screenshots when the
user-facing interface changes.

The context budget is denominated in tokens, not characters, and the cached
prefix is never spent to satisfy it. Characters and tokens differ by up to 3x on
real traffic (measured `prompt_chars / input_tokens` across 318 turns: median
2.06, min 0.83, max 2.51), so a character budget cannot express a token limit.
Count with `models.count_tokens()` -- litellm ships the counter and is already a
dependency; do not add a tokenizer package. Per-section budgets and floors are
tokens too: there is exactly one unit in this system.

When converting a character limit to tokens, do **not** divide everything by one
ratio. The chars-per-token ratio varies 2.4x by content type -- measured on this
repo's sources plus one real `ask()` transcript with the `mimo-v2.5` tokenizer:
prefix 2.48 (repo snapshot is mostly paths), memory 3.06, relevant_memory 1.31
(model-written Chinese notes), history 2.07, tool output 3.02, tool schema 3.95,
whole prompt 2.63. A blanket halving loosened some limits 1.5x and cut others by
40%. Convert each limit by the ratio of the content that limit actually holds.

`prefix` is excluded from `DEFAULT_REDUCTION_ORDER` and filtered out of any
caller-supplied order via `PROTECTED_SECTIONS`. It used to sit last as a "last
resort" and was reached on every single over-budget turn (67 of 632 measured
turns cut all four sections down to their floors). Cutting it saves a few hundred
characters and costs that turn's prefix-cache hit, and the hit rate was raised
from 32.7% to 77.6% by keeping that span byte-identical across requests.
Overflowing the internal budget is the cheaper failure, and it is recorded in
`prompt_over_budget` and `budget_floor_exhausted` rather than happening silently.

"Excluded from reduction" is now literal: `prefix` has no quota at all, a `prefix`
entry in `section_budgets` is dropped by both `ContextManager.__init__` and
`evaluator._apply_task_setup()`, and its `budget_tokens` is `None` in artifacts --
never 0, which reads as "the quota is zero" and means the opposite.

Sections carry no fixed quota either. Each reducible section starts at
`total_budget` and is only cut when the assembled prompt genuinely does not fit,
down to `SECTION_FLOORS`. The removed constants (prefix 1450, memory 520,
relevant_memory 900, history 2500) were absolute and therefore decoupled from
`total_budget`, and measurement showed both directions were wrong: at the 1M tier
they sum to 5,370 -- 4.5% of a 118,335 budget, so the reduction loop never fires
while every section is still cut each turn (this repo's prefix measured 3,696
tokens and was clipped to 1,450, discarding 61% with 116k of budget unused); at
the 8k fallback tier the same 5,370 is 123.9% of the 4,335 budget, so every turn
runs the reduction loop. Put limits at the source instead
(`workspace.MAX_SNAPSHOT_TOKENS`, `MAX_TOOL_OUTPUT`, `memory.NOTE_TOKENS`), which
is where Claude Code puts them: it allocates no assembly-time quota, each block
goes in whole, and the limit applies when the content is read (auto memory keeps
the first 200 lines or 25KB; tool results cap at 25,000 tokens).

Derive the source-entry caps from `total_budget` too. Phase two removed the
per-section quotas but left the same disease one level down: `MAX_TOOL_OUTPUT =
1320` and the hardcoded `line_limit = 430` in `_compressed_history_entries()` are
character-era constants (4000/3.02 and 900/2.07) decoupled from the budget.
Measured: with ten `read_file` turns in history, scaling `total_budget` from 4,335
to 118,335 (27x) changed the outgoing prompt by exactly zero tokens (3,490 ->
3,490) and left all ten `tool` messages byte-identical; a 5,599-token file reached
the model as 430 tokens, 7.7% of itself, with 114,562 tokens of budget unused.
Both now go through `context_manager.tool_output_limit(total_budget)` =
`clamp(total_budget // 8, 1320, 25000)`. Divisor 8 means one result may take an
eighth of the budget, so eight of them trip the global reduction loop that exists
for exactly that case; 25,000 matches Claude Code's tool-result cap; the 1320
floor keeps the 8k fallback tier byte-identical (4335//8 = 541). `line_limit`
takes the same value, which removes the second clip entirely -- the content was
already capped once when `run_tool()` stored it. Same ten-turn history at the 1M
tier: 430 -> 5,599 tokens per recent result, history 2,723 -> 33,794, whole prompt
3,490 -> 34,564 against a 118,335 budget. The 10x rise in input tokens is the
price of the budget finally applying; it is not free.

A tool result that falls out of the recent window must say so. It used to render
as `[tool:read_file] {"path":"a.py"}` -- its own call signature, sitting inside a
`role:"tool"` message, which reads as "this call returned nothing". Append
`CLEARED_RESULT_MARKER` instead, the same contract as the official
`clear_tool_uses_20250919` placeholder: keep the call record, replace the content
with an explicit marker. Costs about 13 tokens per cleared record.

A tool result over the single-result cap is spilled to disk in full; the context
keeps a preview plus a pointer (phase three, L1). `run_tool()` calls
`_store_tool_output()`, which writes the whole thing to
`<workspace>/.codingforme/tool_outputs/<run_id>/<n>-<tool>.txt` and returns a head
preview plus one line from `context_manager.spill_marker()`:
`[full output saved to <forward-slash relative path> (N tokens); use read_file on
that path to see the rest]`. The shape copies Claude Code's Tool Result Budget
(their wording is `Output too large (X MB). Full output saved to: ...`), and four
properties are deliberate: **spill without truncating**; **the pointer carries the
path and the original size** (Manus: droppable only if recoverable --
`CLEARED_RESULT_MARKER` only says "cleared", which leaves `search` / `run_shell`
output unrecoverable); **recovery goes through the existing `read_file`, no new
tool**; **the spill happens when the result enters history, not by rewriting
history later** (a later rewrite invalidates every cached prefix from that point;
Claude Code built server-side `cache_edits` for exactly this and we have no such
API). Three hard constraints: the directory must sit under the **workspace root**
(not `repo_root`), or `path()` blocks the model from reading it back; every path
goes through `as_posix()` (`relative_to()` yields backslashes on Windows and this
string is fed straight back into `read_file` -- the paths echoed by `read_file`,
`write_file` and `patch_file` were normalised for the same reason); and the file
goes through `redact_text()` before it is written, because it lands in the
workspace *and* comes back to the model. Any failure falls back to a plain
`clip()` and never raises. Measured on this repo (8k fallback tier, 1,320 cap):
`search("def ")` 28,416 -> 1,287, `read_file("codingforme/runtime.py", 1..3000)`
35,679 -> 1,286, `list_files` 34 tokens and no spill.

When an entry leaves the recent window, a spilled result keeps its pointer and an
`error:` result keeps its text (capped at `ERROR_KEEP_TOKENS = 90`). The second is
Manus's "keep the wrong turns in the context": clearing a failed observation makes
the model forget what it just hit, so it reissues the same call and lands on the
repeated-call check, burning another ~17-second round trip. Both branches must come
*before* the `run_shell` branch, which keeps only the first three lines -- the
pointer always sits last and an error string is one sentence.

Read results go stale, and the context has to say so (supply-side freshness).
Once a file is written, an earlier `read_file` record holds the *pre-change*
text, which looks exactly like the current file to the model -- measured effect:
the model copies `old_text` out of that stale text, matches zero times and gets
rejected (14% of rejected calls in a k=3 batch). `_stale_read_indexes()` walks
history and marks every read of a path that a later write touched (only reads
*before* the write). **Which files were written comes from the entry's `wrote`
field, not `args.path`**: `wrote` is the sha256 diff of the workspace snapshot,
and `args` cannot see three real cases -- a `run_shell` write, an inner
`patch_file` inside a `run_plan` (one history entry for the whole plan), and a
*failed* write (path present, file unchanged; invalidating there would erase the
only correct `old_text` the model can copy for its retry).

The presentation splits in two, and the two markers must not be merged. Inside
the window, `POLICY_STALE_READ` keeps the call signature and replaces the body
with `STALE_READ_MARKER`. Outside it, `POLICY_STALE_POINTER` **keeps the spill
pointer** -- the full text is still on disk and that pointer is the only way back
to it; downgrading to `cleared` would turn a recoverable record into an
unrecoverable one, which is the whole reason L1 spills in the first place -- and
appends `STALE_SPILL_MARKER`. They are separate constants because
`STALE_READ_MARKER` says "re-read it": inside the window the body is still there
so "it" means the original file, but outside the window only the pointer remains
and the easiest re-read to copy is the *equally stale* spill file
(`.codingforme/tool_outputs/<run_id>/<n>-read_file.txt` is a snapshot taken at
read time and does not follow the original). So `STALE_SPILL_MARKER` says
explicitly that the saved copy is the old version and the original path is what
to re-read. The other two out-of-window shapes need nothing: `file_summary` is
already invalidated by sha256 in the memory layer, and `cleared` carries no body.

One ablation flag covers both halves: `stale_read_invalidation` (on by default) /
variant `no_stale_read`. Splitting it in two would make `no_stale_read` disable
only half the mechanism, so the A/B would measure half a feature. Observability is
`prompt_metadata["history"]["stale_read_count"]` and `stale_pointer_count`, both
written even at zero; the latter still counts into `spilled_pointer_count` (it is
a pointer, just annotated), so the two numbers answer different questions: how
many cleared results still carry a way back, and how many of those ways back are
already out of date.

The recent window advances in blocks, not one entry per turn
(`RECENT_WINDOW_BLOCK = 3`). `_recent_start_by_tool_turns()` is recomputed every
turn, so once history exceeds `RECENT_TOOL_TURNS = 6` each new tool turn pushes the
boundary forward by exactly one: the `tool` message that was full text last turn
becomes a placeholder this turn, every message after it changes, and **the prefix
cache is invalidated every single turn**. `_blocked_recent_window()` rounds "how
many to clear" down to a multiple of 3, so the live window floats between 6 and 8
and the boundary is rewritten once every three turns. It is the poor-man's version
of Claude Code's Microcompact Path B.

Live check (14 tasks x 1 round, mimo-v2.5, `artifacts/eval-live-phase3.json` against the
phase-two baseline `artifacts/eval-live-p0p1p2.json`): L2 12/14 -> 12/14 (different tasks
fail; k=1 jitter), **L1 `subject: harness` 55/55 -> 53/53, all green** (no harness
regression), `subject: model` 86/88 -> 80/84, safety 40/40 -> 38/38. Those numbers prove
no regression and nothing else: across every trace, `tool_output_spilled` was true **0
times** and `spilled_pointer_count` / `preserved_error_count` both summed to **0** -- three
of the four changes never executed on this benchmark. Only the block-advancing window ran
(`recent_tool_window` distribution `{6: 50, 7: 3}`).

Observability fields are always written, zeros included: `tool_executed` carries
`tool_output_spilled` / `tool_output_spill_path` / `tool_output_full_tokens` /
`tool_output_kept_tokens`; `prompt_metadata["history"]` carries
`recent_tool_window` / `spilled_pointer_count` / `preserved_error_count`; the batch
rollup is `aggregates["tool_usage"]["spill"]`, and the markdown says "never fired"
rather than omitting the section (omission reads as "nothing wrong here" -- the
`run_plan` section already made that mistake). `saved_tokens = full_tokens -
kept_tokens`, **which is not a net saving**: the full text is still in the
workspace and the model can spend a `read_file` to get it back. Locked by
`tests/test_tool_output_spill.py` (9 cases) and two cases in
`tests/test_eval_p1.py`. **None of the 14 benchmark tasks trigger L1** -- their
tool outputs all fit -- so that suite proves no regression, not any benefit.

One reduction must free at least a tenth of the budget
(`CLEAR_AT_LEAST_DIVISOR = 10`). Trimming exactly `overflow` guarantees another
trim next turn, and each history trim invalidates every cached prefix after it
(the 32.7% -> 77.6% hit rate cost a full A/B round to win). The official context
management API keeps `clear_at_least` for this reason -- "ensures cache
invalidation worthwhile". `budget_reductions[].target_tokens` records what was
actually used; differing from `overflow_tokens` means it fired.

**In the default configuration it never fires, and that is arithmetic, not luck.**
`overflow = prompt_tokens - target`, and the loop is only entered when
`prompt_tokens > trigger`, so overflow is at least `(0.85 - 0.70) * budget` = 15% of
the budget, while this rule only asks for 10%; `max()` always picks the former. It is
**not** deleted: "no object to act on" is not "structurally incapable of having one".
Turn graded compression off (`no_graded_compression`) and it binds immediately -- on
the same load, overflow 31 is trimmed by 300, landing 324 below the budget line
instead of 3. The relationship between the two constants is locked by
`tests/test_context_manager.py::test_clear_at_least_never_binds_while_graded_compression_is_on`,
so narrowing trigger/target to within 1/10 fails loudly rather than silently changing
which mechanism is in charge. Ablation flag `clear_at_least` (on by default), variant
`no_clear_at_least`.

Reversible folding now has a control arm too: flag `reversible_squeeze` (on by
default), variant `no_reversible_squeeze`. Off, out-of-window entries that do not fit
are dropped whole and only counted in the `Omitted context:` line -- exactly the
behaviour that predates naming the step `POLICY_SQUEEZED`. The switch exists because
the squeeze had been hiding unnamed inside the retention loop, so "a 10-token stub
beats dropping the entry" had never been measured. On the same 24-turn dialogue load
(budget 4,823): on -> 4 squeezed / 31 dropped; off -> 0 squeezed / 35 dropped, i.e.
the stubs are exactly what those four entries survived on. The stub keeps the *head*
of the entry, so the turn is still identifiable; keeping the tail would spend all ten
tokens on a subjectless fragment and be no better than dropping it.

Every context compaction writes a `context_compacted` trace event with
`trigger: manual | auto`. Before this, `/compact` wrote nothing at all and the
automatic path only showed up as a nested counter inside `prompt_built`
(`context_pressure.session_summary.compactions`), so "why does this run not
remember the spec from turn 1" could only be guessed by recomputing the history --
while resume happily carries the compacted state forward. Both paths fill the *same*
fields from the *same* source, `session["context_summary"]` (`covered_entries`,
`newly_covered`, `transcript_entries`, `covered_tokens`, `summary_tokens`,
`saved_tokens`, `refreshes`, `run_status`); neither assembles its own shape from its
own return value, or the aggregation code would have to be written twice.
`saved_tokens = covered_tokens - summary_tokens` and **may be negative** (the summary
came out longer than what it replaced) -- do not clamp it to 0, that is exactly the
case where the mechanism is a net loss. Compacting before any run has started still
compacts but writes no trace: opening a run with no model call in it would skew every
per-run average. A manual compaction between turns is filed under the *previous* run,
hence the `run_status` field.

`/context` also reports pressure: `pressure` (last turn's prompt tokens over the
budget, with a percentage), `trigger` / `target` (the 85% / 70% points; with graded
compression off only one line is shown, spelling out `graded compression is OFF`, or
"trigger == budget" looks like a display bug), and `summary` (how many transcript
entries the summary already covers -- the only basis for deciding whether another
`/compact` would do anything). With no turn assembled yet it says so rather than
printing 0.0%: "occupancy is zero" and "never measured" are not the same fact.
Thresholds come from `ContextManager.compression_thresholds()`.

Two live findings that run *against* the reasoning the mechanisms were built on --
recorded rather than acted on, because "measured no benefit here" is not the deletion
criterion; "structurally incapable of acting on this workload class" is.

First, `stale_read_invalidation` is a net cost on the probe built to exercise it. The
fixture forces the second edit's `old_text` to span the first edit (the target line and
the line below it both appear twice in the file, so the only disambiguating neighbour is
the line just patched). Over 3 runs per arm, the ablated arm had **zero** rejected
`patch_file` calls -- the model remembers its own edit rather than copying the stale
full text -- while the enabled arm spent ~3 extra tool calls and ~30% more input tokens
re-reading. The evidence that motivated the mechanism came from a different load (k=3
benchmark batches, where `old_text` misses were 14% of rejected calls); the two have not
been reconciled.

Second, `DELEGATE_MAX_STEPS_CEILING = 8` and `TOOL_OUTPUT_BUDGET_DIVISOR = 8` are the
same number, and that is a structural conflict. The divisor means "eight tool results
fill the budget exactly, which is when global trimming starts", so any investigation big
enough to be worth delegating necessarily needs more than eight steps -- exactly the
child agent's hard ceiling. All 6 delegations across 4 live runs were truncated
(`delegate_child_incomplete`), and the delegating arms burned more than twice the input
tokens of the control. Neither constant was changed: the ceiling is a permission
boundary, and the divisor moves the whole derivation chain.

To size a fixture right at the spill threshold, measure what `run_tool("read_file", ...)`
returns, not the file: line-number prefixes and the path header make the output about 35%
larger than the source (1,105 -> 1,493 tokens measured).

The plan transcript's aggregate cap has to move with it.
`plans.MAX_TRANSCRIPT_TOKENS = 4000` becomes a floor; the live value is
`plans.transcript_limit(tool_output_limit)` = `max(4000, 3 * single-result cap)`,
written into `PlanResult.transcript_limit` by `runtime.execute_plan()` so `plans.py`
still knows nothing about the agent. The multiple of 3 is the original design
("12000 chars is about three ordinary tool calls"); without it, a single result may
be 14,791 tokens at the 1M tier while the whole transcript stays capped at 4,000, so
the same `read_file` shows the model *less* inside a plan and `run_plan` becomes a
pure loss.

Report `None`, not `total_budget`, for a section with no cap. A starting
allowance equal to `total_budget` means "no ceiling here"; writing the number into
the artifact reads as "this section was allocated 118,335", and the three of them
sum to three times the budget. Only a caller-supplied allowance or an allowance
the reduction loop actually pushed down gets a number.

Deliberately not done: reordering `DEFAULT_REDUCTION_ORDER` by re-fetchability.
The principle (clear cheap re-fetchable tool results before summarising
unrecoverable reasoning) is already honoured inside `history` -- out-of-window
tool results are cleared to a placeholder before whole entries are dropped into
`_omitted_digest()`. The section-level order answers a different question, and
`relevant_memory -> history -> memory` sacrifices the smallest, most regenerable
section first: notes are re-selected from the note store every turn, while
`history` carries the `assistant.tool_calls`/`role:"tool"` pairing constraint and
is the mechanism behind the 76% -> 97% terminal-state rate.

Derive `budget_floor_exhausted` from the reduction loop itself -- a full pass over
`reduction_order` that could not move any section. Do not test
`budgets[section] <= floor`: sections now start at `total_budget`, so one that is
naturally smaller than its floor is skipped without its budget ever being written
back, and that test is then always false.

Derive `total_budget` from the model's window instead of hardcoding it:
`models.resolve_context_window()` checks an explicit setting, then
`KNOWN_MODEL_WINDOWS` for self-hosted backends litellm does not map, then the
litellm registry, then a conservative default -- and floors the answer to a
`WINDOW_BUCKETS` step. Buckets keep the value declarable and reproducible, which
matters because `total_budget` feeds the `HarnessSpec` fingerprint and because a
moving budget moves the truncation point and therefore the cache prefix. Charge
the tool schema against the budget exactly once, when deriving it; charging it
again at measurement time gives `total_budget` an invisible floor that no amount
of trimming can satisfy. Note that `litellm.get_max_tokens()` returns
`max_output_tokens`, not the context window.

Every deduction in `models.budget_breakdown()` is an absolute amount except the
last one. The backend bills more input tokens than `prompt_tokens` reports,
because the request is a messages array while `prompt_tokens` counts the
flattened text: role fields, message delimiters, `assistant.tool_calls` wrappers
and `tool_call_id` strings are billed and are not in that text. Measured over 23
live turns (`scripts/measure_token_accounting.py`), that residual regresses on
message count as `191 + 20 x count` with R^2 = 0.838, and on prompt length with
only R^2 = 0.396 -- so reserve it per message, not as a fraction of the window.
Estimate the count from the step ceiling (`5 + 2 x max_steps`), because the
budget is fixed before the run knows how many steps it will take. Only the
trailing `TOKENIZER_MARGIN_RATIO` stays proportional; it covers length-scaled
estimation error, which is a different thing. When the deductions exhaust the
window the budget falls back to `BUDGET_FLOOR_TOKENS`, and that fallback must be
visible (`floored`) -- "the window cannot hold this" and "the budget happens to
be this number" call for opposite responses.

There is exactly one unit in this system, and it is the token. Budgets, floors,
per-section allowances, every comparison and every truncation use it: the context
budget, `workspace.MAX_TOOL_OUTPUT` and `MAX_HISTORY`, `plans.MAX_TRANSCRIPT_TOKENS`,
and `memory.NOTE_TOKENS` / `TASK_SUMMARY_TOKENS`. Artifacts carry no `*_chars`
fields either. `models.clip_tokens()` and `head_tail_clip_tokens()` are the only
truncation primitives; they clip exactly via `litellm.encode`/`decode`.

Count with the same tokenizer you clip with. `token_counter()` and `encode()`
disagree by 40% on Chinese for the same model name, so clipping to 290 tokens by
`encode` and then counting with `token_counter` reported 175 -- four tenths of the
allowance silently wasted. `count_tokens()` is therefore `len(encode(...))`.

The mixed regime that preceded this -- a token gate over character-denominated
section allowances -- forced two pieces of code that had no principle behind them:
converting a token overflow into a character decrement, and an estimate-clip-remeasure
loop that ran up to five times because the characters-per-token ratio was estimated
from the pre-clip text. Both are gone. When a fixture needs to exceed a budget,
build it from distinct words: a run of repeated characters costs almost no tokens
and the test will pass while measuring nothing.
