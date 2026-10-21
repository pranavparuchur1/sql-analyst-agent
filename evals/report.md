# Eval report: `claude-cli:sonnet`

Run at 2026-10-21 07:35. 20 questions, each on a fresh thread, approvals auto-granted.

| Metric | Value |
|---|---|
| Execution accuracy | 100% (20/20) |
| - easy | 100% (6/6) |
| - medium | 100% (8/8) |
| - hard | 100% (6/6) |
| First-try success | 100% (20/20) |
| Avg SQL attempts | 1.00 |
| Latency p50 / p95 | 19.7s / 23.7s |
| LLM calls (avg per question) | 80 (4.0) |
| API-equivalent cost (total / per question) | $0.54 / $0.027 |

Run on a Claude subscription via `claude -p` with 4 parallel worker(s). API-equivalent cost is what these tokens would cost on the API; nothing was billed. Latency includes starting a `claude` CLI process for every LLM call, so it is higher than API latency.

## Failures

None.
