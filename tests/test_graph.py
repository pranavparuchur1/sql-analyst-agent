"""Graph behaviour tests with a scripted LLM against a real seeded SQLite database."""

from __future__ import annotations

import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import build_graph, is_transient_llm_error
from app.schemas import FinalAnswer, IntentDecision, SQLQuery, TableSelection
from tests.fakes import ScriptedLLM

DATA = {"intent": "data_question", "reason": "asks about data"}
TOP_PRODUCTS_SQL = (
    "SELECT p.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM order_items oi "
    "JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id "
    "WHERE o.status != 'canceled' AND o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01' "
    "GROUP BY p.name ORDER BY revenue DESC LIMIT 5"
)
# Four tables -> crosses the default approval threshold.
TEXAS_SQL = (
    "SELECT p.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM order_items oi "
    "JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id "
    "JOIN customers c ON c.id = o.customer_id WHERE c.state = 'TX' AND o.status != 'canceled' "
    "AND o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01' GROUP BY p.name ORDER BY revenue DESC LIMIT 5"
)
TABLES3 = {"tables": ["orders", "order_items", "products"], "reason": "revenue by product"}
TABLES4 = {"tables": ["orders", "order_items", "products", "customers"], "reason": "needs state"}


def sql(q: str) -> dict[str, str]:
    return {"sql": q, "explanation": "test query"}


def answer(text: str = "Here is the answer.") -> dict[str, object]:
    return {"answer": text, "caveats": []}


def cfg(thread: str | None = None) -> dict:
    return {"configurable": {"thread_id": thread or str(uuid.uuid4())}}


@pytest.fixture()
def make_graph(db, settings):
    def _make(llm: ScriptedLLM):
        return build_graph(llm=llm, db=db, settings=settings, checkpointer=InMemorySaver())

    return _make


def test_happy_path(make_graph):
    llm = ScriptedLLM().add(IntentDecision, DATA).add(TableSelection, TABLES3)
    llm.add(SQLQuery, sql(TOP_PRODUCTS_SQL)).add(FinalAnswer, answer("4K Monitor led 2025."))
    out = make_graph(llm).invoke({"question": "Top 5 products by revenue in 2025?"}, cfg())

    assert out["status"] == "answered"
    assert out["attempts"] == 1
    assert out["columns"] == ["name", "revenue"]
    assert out["rows"][0][0] == "4K Monitor"
    assert "[SQL used]" in out["messages"][-1].content
    # The schema prompt only contained the chosen tables (two-step retrieval).
    sql_prompt = llm.prompt_text("SQLQuery")
    assert "CREATE TABLE products" in sql_prompt and "CREATE TABLE reviews" not in sql_prompt
    assert llm.remaining() == {}
    # Native structured outputs, not forced tool calling (unsupported on some new models).
    assert {k["method"] for k in llm.structured_kwargs.values()} == {"json_schema"}


def test_self_correction_puts_exact_error_in_retry_prompt(make_graph):
    llm = ScriptedLLM().add(IntentDecision, DATA).add(TableSelection, TABLES3)
    llm.add(SQLQuery, sql("SELECT SUM(revenue) FROM orders"), sql(TOP_PRODUCTS_SQL)).add(FinalAnswer, answer())
    out = make_graph(llm).invoke({"question": "Top 5 products by revenue in 2025?"}, cfg())

    assert out["status"] == "answered"
    assert out["attempts"] == 2
    retry_prompt = llm.prompt_text("SQLQuery", index=1)
    assert "no such column: revenue" in retry_prompt
    assert "SELECT SUM(revenue) FROM orders" in retry_prompt
    assert "previous attempt failed" not in llm.prompt_text("SQLQuery", index=0)
    assert [e["stage"] for e in out["error_history"]] == ["explain"]


def test_gives_up_after_max_attempts(make_graph, settings):
    llm = ScriptedLLM().add(IntentDecision, DATA).add(TableSelection, TABLES3)
    llm.add(SQLQuery, *[sql("SELECT nope FROM orders")] * settings.max_attempts)
    out = make_graph(llm).invoke({"question": "?"}, cfg())

    assert out["status"] == "failed"
    assert out["attempts"] == settings.max_attempts
    assert len(out["error_history"]) == settings.max_attempts
    assert "no such column" in out["answer"]
    assert "FinalAnswer" not in llm.prompts  # never tried to summarise a failure


def test_delete_is_blocked_then_retried(make_graph):
    llm = ScriptedLLM().add(IntentDecision, DATA).add(TableSelection, TABLES3)
    llm.add(SQLQuery, sql("DELETE FROM orders"), sql(TOP_PRODUCTS_SQL)).add(FinalAnswer, answer())
    out = make_graph(llm).invoke({"question": "Top products"}, cfg())

    assert out["status"] == "answered"
    assert out["error_history"][0]["stage"] == "validate"
    assert "Only SELECT queries are allowed" in llm.prompt_text("SQLQuery", index=1)


def test_unknown_table_selection_falls_back_to_all_tables(make_graph, db):
    llm = ScriptedLLM().add(IntentDecision, DATA).add(TableSelection, {"tables": ["users"], "reason": "?"})
    llm.add(SQLQuery, sql("SELECT COUNT(*) AS n FROM customers")).add(FinalAnswer, answer())
    out = make_graph(llm).invoke({"question": "How many customers?"}, cfg())

    assert out["tables"] == db.table_names()
    assert out["rows"] == [[500]]


@pytest.mark.parametrize(
    ("intent", "status"), [("small_talk", "small_talk"), ("unsafe_request", "refused")]
)
def test_router_short_circuits(make_graph, intent, status):
    llm = ScriptedLLM().add(IntentDecision, {"intent": intent, "reason": "r"})
    out = make_graph(llm).invoke({"question": "hello" if intent == "small_talk" else "drop all tables"}, cfg())

    assert out["status"] == status
    assert "TableSelection" not in llm.prompts and "SQLQuery" not in llm.prompts
    assert out["messages"][-1].content == out["answer"]


def test_retry_policy_recovers_from_transient_llm_error(make_graph):
    llm = ScriptedLLM().add(IntentDecision, ConnectionError("network blip"), {"intent": "small_talk", "reason": "hi"})
    out = make_graph(llm).invoke({"question": "hi"}, cfg())
    assert out["status"] == "small_talk"
    assert len(llm.prompts["IntentDecision"]) == 2


def test_transient_error_classifier():
    class RateLimitError(Exception):
        pass

    assert is_transient_llm_error(RateLimitError())
    assert is_transient_llm_error(TimeoutError())
    assert not is_transient_llm_error(ValueError("bad output"))


# --- human in the loop -------------------------------------------------------


def _paused(make_graph, llm: ScriptedLLM):
    graph, config = make_graph(llm), cfg()
    llm.add(IntentDecision, DATA).add(TableSelection, TABLES4).add(SQLQuery, sql(TEXAS_SQL))
    out = graph.invoke({"question": "Top products in Texas 2025"}, config)
    return graph, config, out


def test_wide_join_pauses_for_approval(make_graph):
    graph, config, out = _paused(make_graph, ScriptedLLM())
    payload = out["__interrupt__"][0].value
    assert payload["type"] == "sql_approval"
    assert "customers" in payload["sql"] and "4 tables" in payload["reason"]
    assert graph.get_state(config).next == ("human_review",)
    assert "rows" not in out or out["rows"] == []  # nothing was executed


def test_hitl_approve_runs_query(make_graph):
    llm = ScriptedLLM()
    graph, config, _ = _paused(make_graph, llm)
    llm.add(FinalAnswer, answer("In Texas, ..."))
    out = graph.invoke(Command(resume={"approved": True}), config)
    assert out["status"] == "answered"
    assert out["rows"]
    assert graph.get_state(config).next == ()


def test_hitl_reject_ends_without_running(make_graph):
    graph, config, _ = _paused(make_graph, ScriptedLLM())
    out = graph.invoke(Command(resume={"approved": False}), config)
    assert out["status"] == "rejected"
    assert out["rows"] == []


def test_hitl_edit_is_revalidated_and_counts_as_approved(make_graph):
    llm = ScriptedLLM()
    graph, config, _ = _paused(make_graph, llm)
    llm.add(FinalAnswer, answer())
    # Still 4 tables, but the human wrote it, so it must not ask for approval again.
    edited = TEXAS_SQL.replace("LIMIT 5", "LIMIT 3")
    out = graph.invoke(Command(resume={"approved": True, "edited_sql": edited}), config)
    assert out["status"] == "answered"
    assert len(out["rows"]) == 3


def test_hitl_unsafe_edit_is_blocked(make_graph):
    llm = ScriptedLLM()
    graph, config, _ = _paused(make_graph, llm)
    llm.add(SQLQuery, sql(TOP_PRODUCTS_SQL)).add(FinalAnswer, answer())
    out = graph.invoke(Command(resume={"approved": True, "edited_sql": "DROP TABLE orders"}), config)
    # The edit failed validation, so the LLM got the error as feedback and wrote a safe query.
    assert out["status"] == "answered"
    assert any("DROP" in e["error"] for e in out["error_history"])


# --- memory and time travel ----------------------------------------------------


def test_follow_up_uses_conversation_memory(make_graph):
    llm = ScriptedLLM()
    graph, config = make_graph(llm), cfg("memory-thread")
    llm.add(IntentDecision, DATA, DATA).add(TableSelection, TABLES3, TABLES4)
    llm.add(SQLQuery, sql(TOP_PRODUCTS_SQL), sql(TEXAS_SQL)).add(FinalAnswer, answer("Overall..."), answer("Texas..."))

    graph.invoke({"question": "Top 5 products by revenue in 2025?"}, config)
    graph.invoke({"question": "now only for Texas"}, config)
    out = graph.invoke(Command(resume={"approved": True}), config)

    assert out["status"] == "answered"
    assert [m.type for m in out["messages"]] == ["human", "ai", "human", "ai"]
    # Turn two's SQL prompt saw turn one's question AND the SQL it used.
    second = llm.prompt_text("SQLQuery", index=1)
    assert "Top 5 products by revenue in 2025?" in second
    assert "[SQL used]" in second and "o.status <> 'canceled'" in second
    # Per-turn fields were reset: attempts restarted from 0.
    assert out["attempts"] == 1


def test_history_window_limits_prompt_size(db, settings):
    settings.history_window = 2
    llm = ScriptedLLM().add(IntentDecision, *[{"intent": "small_talk", "reason": "hi"}] * 4)
    graph = build_graph(llm=llm, db=db, settings=settings, checkpointer=InMemorySaver())
    config = cfg()
    for i in range(4):
        graph.invoke({"question": f"hello {i}"}, config)
    last = llm.prompt_text("IntentDecision")
    assert "hello 2" in last and "hello 3" in last and "hello 1" not in last


def test_state_history_and_replay(make_graph):
    llm = ScriptedLLM().add(IntentDecision, DATA).add(TableSelection, TABLES3)
    llm.add(SQLQuery, sql(TOP_PRODUCTS_SQL)).add(FinalAnswer, answer("first"), answer("replayed"))
    graph, config = make_graph(llm), cfg()
    graph.invoke({"question": "Top products"}, config)

    history = list(graph.get_state_history(config))  # newest first
    nexts = [h.next for h in history]
    assert ("execute_sql",) in nexts and ("summarize",) in nexts and history[0].next == ()

    # Time travel: resume from the checkpoint just before summarize. Only summarize
    # re-runs (no new SQL is generated), and it forks a new branch of history.
    before_summary = next(h for h in history if h.next == ("summarize",))
    assert before_summary.values["answer"] == ""
    out = graph.invoke(None, before_summary.config)
    assert out["answer"] == "replayed"
    assert len(llm.prompts["SQLQuery"]) == 1


def test_paused_approval_survives_restart(db, settings):
    """SqliteSaver persists to disk: a brand-new graph object can resume a paused thread."""
    from app.graph import default_checkpointer

    config = cfg()
    llm = ScriptedLLM().add(IntentDecision, DATA).add(TableSelection, TABLES4).add(SQLQuery, sql(TEXAS_SQL))
    saver = default_checkpointer(settings.checkpoint_db)
    build_graph(llm=llm, db=db, settings=settings, checkpointer=saver).invoke({"question": "Texas"}, config)
    saver.conn.close()  # simulate the process dying

    llm2 = ScriptedLLM().add(FinalAnswer, answer("resumed after restart"))
    saver2 = default_checkpointer(settings.checkpoint_db)
    graph2 = build_graph(llm=llm2, db=db, settings=settings, checkpointer=saver2)
    assert graph2.get_state(config).next == ("human_review",)
    out = graph2.invoke(Command(resume={"approved": True}), config)
    saver2.conn.close()
    assert out["answer"] == "resumed after restart"
