# bench/

One executable per `bench/<name>.cpp` (target `bench_<name>`), for reproducible performance and
memory measurements that back numbers quoted in docs and commit messages. Run them through
`tools/run/guard.py`; print machine-readable results (JSON lines) to stdout. Heavier,
case-based comparisons live in `tools/bench/` (Python: SLEPc baseline, `compare.py`).
