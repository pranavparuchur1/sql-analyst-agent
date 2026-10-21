"""Tests for the eval harness itself: a wrong scorer silently makes every number meaningless."""

from __future__ import annotations

from collections import Counter

from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_graph
from app.schemas import FinalAnswer, IntentDecision, SQLQuery, TableSelection
from evals.run_evals import load_golden, results_match, run_one, summarize
from tests.fakes import ScriptedLLM


def test_match_ignores_row_order_column_order_and_float_noise():
    gold = [("4K Monitor", 137029.71000000002), ("Yoga Mat", 54449.4)]
    pred = [(54449.401, "Yoga Mat", 2), (137029.71, "4K Monitor", 9)]  # extra column is fine
    assert results_match(gold, pred)


def test_match_treats_int_and_float_equal():
    assert results_match([(500,)], [(500.0,)])


def test_match_rejects_wrong_values_and_wrong_row_count():
    assert not results_match([(500,)], [(499,)])
    assert not results_match([("a",), ("b",)], [("a",)])
    assert not results_match([("a", 1)], [("a", 2)])


def test_match_requires_each_gold_column_as_a_whole():
    # Same set of values overall, but one column's multiset differs: not a match.
    assert not results_match([("a", 1), ("b", 2)], [("a", 1), ("b", 1)])


def test_known_limitation_row_pairing_is_not_checked():
    # Columns are compared independently, so swapped pairings still "match". This is the
    # price of tolerating column reordering/extra columns; documented in the README.
    assert results_match([("a", 1), ("b", 2)], [("a", 2), ("b", 1)])


def test_match_handles_null_and_case():
    assert results_match([(None, "TX")], [(None, "tx")])


def test_golden_set_shape():
    items = load_golden()
    assert Counter(i["level"] for i in items) == {"easy": 6, "medium": 8, "hard": 6}
    assert len({i["id"] for i in items}) == 20


def test_every_gold_query_runs_and_returns_rows(db):
    for item in load_golden():
        result = db.run(item["gold_sql"])
        assert result.rows, item["id"]
        assert not any(v is None for row in result.rows for v in row), item["id"]


def test_run_one_scores_a_scripted_answer(db, settings):
    item = next(i for i in load_golden() if i["id"] == "e5")
    llm = ScriptedLLM().add(IntentDecision, {"intent": "data_question", "reason": "r"})
    llm.add(TableSelection, {"tables": ["customers"], "reason": "r"})
    llm.add(SQLQuery, {"sql": "SELECT COUNT(*) AS n FROM customers WHERE state = 'TX'", "explanation": "x"})
    llm.add(FinalAnswer, {"answer": "122", "caveats": []})
    graph = build_graph(llm=llm, db=db, settings=settings, checkpointer=InMemorySaver())
    outcome = run_one(graph, db, item)
    assert outcome.correct and outcome.attempts == 1
    report = summarize([outcome], "fake")
    assert "100% (1/1)" in report
