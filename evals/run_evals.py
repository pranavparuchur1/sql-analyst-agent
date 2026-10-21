"""Execution-accuracy eval over the golden set.

    python -m evals.run_evals                      # uses LLM_MODEL
    python -m evals.run_evals --model openai:gpt-4.1-mini
    python -m evals.run_evals --check-gold         # offline: just verify gold SQL runs

We compare *results*, not SQL text: two very different queries can both be right.
A prediction is correct when every gold result column matches some predicted
column (sorted values, floats rounded to 2 dp). Extra predicted columns are
fine, since the agent often adds helpful context like a count.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden.jsonl"
REPORT = HERE / "report.md"


def _norm(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int, float)):
        return f"{round(float(value), 2):.2f}"  # 500 == 500.0, 3.767 == 3.77
    return str(value).strip().lower()


def column_values(rows: list[Any], index: int) -> list[str]:
    return sorted(_norm(r[index]) for r in rows)


def results_match(gold_rows: list[Any], pred_rows: list[Any]) -> bool:
    """True if each gold column equals (as a sorted multiset) some predicted column."""
    if len(gold_rows) != len(pred_rows):
        return False
    if not gold_rows:
        return True
    pred_cols = [column_values(pred_rows, j) for j in range(len(pred_rows[0]))]
    return all(column_values(gold_rows, i) in pred_cols for i in range(len(gold_rows[0])))


def load_golden(path: Path = GOLDEN) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Outcome:
    id: str
    level: str
    question: str
    correct: bool
    status: str
    attempts: int
    latency_s: float
    sql: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)


def run_one(graph: Any, db: Any, item: dict[str, Any]) -> Outcome:
    config = {"configurable": {"thread_id": f"eval-{item['id']}-{uuid.uuid4().hex[:6]}"}}
    start = time.perf_counter()
    try:
        result = graph.invoke({"question": item["question"]}, config)
        # The eval measures SQL quality, so approvals are granted automatically.
        while result.get("__interrupt__"):
            result = graph.invoke(Command(resume={"approved": True}), config)
    except Exception as e:  # noqa: BLE001 - one bad item must not kill the run
        return Outcome(item["id"], item["level"], item["question"], False, "crashed", 0,
                       time.perf_counter() - start, error=f"{type(e).__name__}: {e}")  # fmt: skip
    latency = time.perf_counter() - start
    gold = db.run(item["gold_sql"])
    correct = result.get("status") == "answered" and results_match(gold.rows, result.get("rows", []))
    return Outcome(item["id"], item["level"], item["question"], correct, result.get("status", "?"),
                   result.get("attempts", 0), latency, sql=result.get("sql", ""),
                   error=result.get("error") or "")  # fmt: skip


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}% ({n}/{d})" if d else "n/a"


def summarize(outcomes: list[Outcome], model: str, extra_rows: list[str] | None = None, note: str = "") -> str:
    total = len(outcomes)
    correct = sum(o.correct for o in outcomes)
    first_try = sum(o.correct and o.attempts == 1 for o in outcomes)
    latencies = sorted(o.latency_s for o in outcomes)
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))] if latencies else 0.0
    attempted = [o.attempts for o in outcomes if o.attempts]
    by_level: dict[str, list[Outcome]] = defaultdict(list)
    for o in outcomes:
        by_level[o.level].append(o)

    lines = [
        f"# Eval report: `{model}`",
        "",
        f"Run at {time.strftime('%Y-%m-%d %H:%M')}. {total} questions, each on a fresh thread, approvals auto-granted.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Execution accuracy | {_pct(correct, total)} |",
        *[f"| - {lvl} | {_pct(sum(o.correct for o in by_level[lvl]), len(by_level[lvl]))} |"
          for lvl in ("easy", "medium", "hard") if lvl in by_level],
        f"| First-try success | {_pct(first_try, total)} |",
        f"| Avg SQL attempts | {statistics.mean(attempted):.2f} |" if attempted else "| Avg SQL attempts | n/a |",
        f"| Latency p50 / p95 | {statistics.median(latencies):.1f}s / {p95:.1f}s |" if latencies else "| Latency | n/a |",
        *(extra_rows or []),
        "",
        *([note, ""] if note else []),
        "## Failures",
        "",
    ]  # fmt: skip
    failures = [o for o in outcomes if not o.correct]
    if not failures:
        lines.append("None.")
    for o in failures:
        lines += [f"### {o.id} ({o.level}): {o.question}", f"- status: `{o.status}`, attempts: {o.attempts}"]
        if o.error:
            lines.append(f"- last error: `{o.error}`")
        if o.sql:
            lines += ["", "```sql", o.sql, "```"]
        lines.append("")
    return "\n".join(lines) + "\n"


def check_gold(db: Any, items: list[dict[str, Any]]) -> None:
    for item in items:
        result = db.run(item["gold_sql"])
        assert result.rows, f"{item['id']}: gold query returned no rows"
        print(f"{item['id']:<4} {item['level']:<7} {len(result.rows):>3} rows  {result.rows[0]}")


def main() -> None:
    from app.config import get_llm, get_settings
    from app.db import SQLiteDatabase
    from app.graph import build_graph

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Override LLM_MODEL, e.g. anthropic:claude-haiku-5-5")
    parser.add_argument("--ids", default=None, help="Comma-separated subset, e.g. e1,m3,h2")
    parser.add_argument("--out", type=Path, default=REPORT)
    parser.add_argument("--workers", type=int, default=1, help="Questions run in parallel (helps slow claude-cli mode).")
    parser.add_argument("--check-gold", action="store_true", help="Only verify that every gold query runs.")
    args = parser.parse_args()

    settings = get_settings()
    db = SQLiteDatabase(settings.db_path, timeout_s=settings.query_timeout_s, max_rows=settings.row_limit)
    items = load_golden()
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [i for i in items if i["id"] in wanted]
    if args.check_gold:
        check_gold(db, items)
        return

    model = args.model or settings.llm_model
    llm = get_llm(settings, model)
    graph = build_graph(llm=llm, db=db, settings=settings, checkpointer=InMemorySaver())

    def one(item: dict[str, Any]) -> Outcome:
        outcome = run_one(graph, db, item)
        mark = "PASS" if outcome.correct else "FAIL"
        print(f"{mark} {item['id']:<4} attempts={outcome.attempts} {outcome.latency_s:5.1f}s  {item['question'][:60]}",
              flush=True)  # fmt: skip
        return outcome

    # Each question has its own thread_id, so runs are independent and safe to parallelise.
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        outcomes = list(pool.map(one, items))

    extra, note = [], ""
    calls = getattr(llm, "calls", None)  # populated by ClaudeCLIModel (subscription mode)
    if calls:
        extra = [
            f"| LLM calls (avg per question) | {len(calls)} ({len(calls) / len(items):.1f}) |",
            f"| API-equivalent cost (total / per question) | ${sum(c.api_equiv_usd for c in calls):.2f} / "
            f"${sum(c.api_equiv_usd for c in calls) / len(items):.3f} |",
        ]
        note = (
            f"Run on a Claude subscription via `claude -p` with {args.workers} parallel worker(s). "
            "API-equivalent cost is what these tokens would cost on the API; nothing was billed. "
            "Latency includes starting a `claude` CLI process for every LLM call, so it is higher than API latency."
        )
    report = summarize(outcomes, model, extra, note)
    args.out.write_text(report, encoding="utf-8")
    print("\n" + report.split("## Failures")[0])
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
