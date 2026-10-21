# Eval report: not yet run

No eval has been run against a real model yet: this repo was built without an API key.
Run `make eval` (or `python -m evals.run_evals --model <provider:model>`) to overwrite this file
with measured execution accuracy, first-try success, attempts and latency.

The gold set itself is verified: `python -m evals.run_evals --check-gold` runs all 20 gold
queries and `tests/test_evals.py` asserts each returns non-empty, non-NULL results.
