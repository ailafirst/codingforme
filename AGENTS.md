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
