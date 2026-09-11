# Stale-Info Fixture

`src/render.py` has two defaults that must be changed one at a time, and the
second one cannot be anchored without quoting the line the first change just
rewrote. `src/core.py` is large on purpose: a whole-file read of it overflows
the per-result limit at the 8k budget tier and therefore spills to disk.
