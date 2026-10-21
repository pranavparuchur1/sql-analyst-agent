"""HTTP-level tests: the full approval flow and multi-turn threads, with a scripted LLM."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.api import create_app
from app.graph import build_graph
from app.schemas import FinalAnswer, IntentDecision, SQLQuery, TableSelection
from tests.fakes import ScriptedLLM
from tests.test_graph import DATA, TABLES3, TABLES4, TEXAS_SQL, TOP_PRODUCTS_SQL, answer, sql


@pytest.fixture()
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture()
def client(llm, db, settings):
    graph = build_graph(llm=llm, db=db, settings=settings, checkpointer=InMemorySaver())
    with TestClient(create_app(graph)) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_full_approval_flow(client, llm):
    llm.add(IntentDecision, DATA).add(TableSelection, TABLES4).add(SQLQuery, sql(TEXAS_SQL))
    r = client.post("/ask", json={"question": "Top products in Texas in 2025"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "needs_approval"
    assert "customers" in body["approval"]["sql"]
    thread = body["thread_id"]

    # While paused, a new question on the same thread is refused...
    assert client.post("/ask", json={"question": "something else", "thread_id": thread}).status_code == 409
    # ...and the thread view shows what is pending.
    assert client.get(f"/threads/{thread}").json()["pending_approval"]["type"] == "sql_approval"

    llm.add(FinalAnswer, answer("Texas: 4K Monitor leads."))
    r = client.post("/approve", json={"thread_id": thread, "approved": True})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["rows"] and body["answer"].startswith("Texas")

    # Nothing left to approve.
    assert client.post("/approve", json={"thread_id": thread, "approved": True}).status_code == 409


def test_reject_via_api(client, llm):
    llm.add(IntentDecision, DATA).add(TableSelection, TABLES4).add(SQLQuery, sql(TEXAS_SQL))
    thread = client.post("/ask", json={"question": "Texas"}).json()["thread_id"]
    body = client.post("/approve", json={"thread_id": thread, "approved": False}).json()
    assert body["status"] == "rejected" and body["rows"] == []


def test_follow_up_thread_and_history(client, llm):
    llm.add(IntentDecision, DATA, {"intent": "small_talk", "reason": "thanks"})
    llm.add(TableSelection, TABLES3).add(SQLQuery, sql(TOP_PRODUCTS_SQL)).add(FinalAnswer, answer())

    first = client.post("/ask", json={"question": "Top 5 products by revenue in 2025?"}).json()
    assert first["status"] == "answered"
    second = client.post("/ask", json={"question": "thanks!", "thread_id": first["thread_id"]}).json()
    assert second["status"] == "small_talk"

    thread = client.get(f"/threads/{first['thread_id']}").json()
    assert [m["role"] for m in thread["messages"]] == ["human", "ai", "human", "ai"]
    assert "[SQL used]" in thread["messages"][1]["content"]

    history = client.get(f"/threads/{first['thread_id']}/history").json()["checkpoints"]
    assert len(history) > 10  # one checkpoint per step, across both turns
    assert history[0]["next"] == []
    assert any(c["next"] == ["execute_sql"] for c in history)


def test_unknown_thread_is_404(client):
    assert client.get("/threads/nope").status_code == 404
    assert client.get("/threads/nope/history").status_code == 404


def test_validation_error_on_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_llm_failure_is_502(client, llm):
    llm.add(IntentDecision, ValueError("model returned garbage"))
    assert client.post("/ask", json={"question": "hi"}).status_code == 502
