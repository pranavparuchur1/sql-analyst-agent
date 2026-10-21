"""HTTP API over the agent graph.

`create_app(graph)` takes an already-built graph so tests can inject one wired to
a fake LLM; with no graph, the lifespan builds the real one at startup.
Endpoints are sync `def`s: FastAPI runs them in a thread pool, which suits the
blocking LLM and SQLite calls inside graph.invoke.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, description="Reuse to ask follow-ups in the same conversation.")


class ApproveRequest(BaseModel):
    thread_id: str
    approved: bool
    edited_sql: str | None = None


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _to_response(thread_id: str, result: dict[str, Any]) -> dict[str, Any]:
    if result.get("__interrupt__"):
        return {"thread_id": thread_id, "status": "needs_approval", "approval": result["__interrupt__"][0].value}
    return {
        "thread_id": thread_id,
        "status": result.get("status"),
        "answer": result.get("answer"),
        "sql": result.get("sql") or None,
        "columns": result.get("columns", []),
        "rows": result.get("rows", []),
        "attempts": result.get("attempts", 0),
    }


def create_app(graph: CompiledStateGraph | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if graph is not None:
            app.state.graph = graph
            yield
            return
        from app.graph import build_graph, default_checkpointer

        saver = default_checkpointer()
        app.state.graph = build_graph(checkpointer=saver)
        yield
        saver.conn.close()

    app = FastAPI(title="SQL Analyst Agent", version="1.0.0", lifespan=lifespan)

    def _graph(request: Request) -> CompiledStateGraph:
        return request.app.state.graph

    def _invoke(g: CompiledStateGraph, payload: Any, thread_id: str) -> dict[str, Any]:
        try:
            return g.invoke(payload, _config(thread_id))
        except Exception as e:  # noqa: BLE001 - the agent handles SQL errors; anything here is upstream (LLM/infra)
            raise HTTPException(status_code=502, detail=f"Agent error: {type(e).__name__}: {e}") from e

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> dict[str, Any]:
        g = _graph(request)
        thread_id = body.thread_id or str(uuid.uuid4())
        # A paused thread is waiting on /approve. Starting a new turn would silently
        # orphan the pending approval, so make the client resolve it first.
        if g.get_state(_config(thread_id)).next:
            raise HTTPException(status_code=409, detail="Thread is waiting for approval; call /approve first.")
        return _to_response(thread_id, _invoke(g, {"question": body.question}, thread_id))

    @app.post("/approve")
    def approve(body: ApproveRequest, request: Request) -> dict[str, Any]:
        g = _graph(request)
        if "human_review" not in g.get_state(_config(body.thread_id)).next:
            raise HTTPException(status_code=409, detail="No query is waiting for approval on this thread.")
        decision = {"approved": body.approved, "edited_sql": body.edited_sql}
        return _to_response(body.thread_id, _invoke(g, Command(resume=decision), body.thread_id))

    @app.get("/threads/{thread_id}")
    def get_thread(thread_id: str, request: Request) -> dict[str, Any]:
        snapshot = _graph(request).get_state(_config(thread_id))
        if not snapshot.values:
            raise HTTPException(status_code=404, detail="Unknown thread.")
        values = snapshot.values
        return {
            "thread_id": thread_id,
            "status": values.get("status"),
            "pending_approval": snapshot.interrupts[0].value if snapshot.interrupts else None,
            "messages": [{"role": m.type, "content": m.content} for m in values.get("messages", [])],
            "last_sql": values.get("sql") or None,
            "error_history": values.get("error_history", []),
        }

    @app.get("/threads/{thread_id}/history")
    def get_history(thread_id: str, request: Request) -> dict[str, Any]:
        # Every checkpoint, newest first: the full step-by-step audit trail of the thread.
        checkpoints = [
            {
                "checkpoint_id": s.config["configurable"]["checkpoint_id"],
                "step": (s.metadata or {}).get("step"),
                "created_at": s.created_at,
                "next": list(s.next),
                "status": s.values.get("status"),
                "sql": s.values.get("sql") or None,
                "attempts": s.values.get("attempts"),
            }
            for s in _graph(request).get_state_history(_config(thread_id))
        ]
        if not checkpoints:
            raise HTTPException(status_code=404, detail="Unknown thread.")
        return {"thread_id": thread_id, "checkpoints": checkpoints}

    return app


# `uvicorn app.api:app` entry point. Building is deferred to the lifespan, so
# importing this module (e.g. in tests) never needs an API key.
app = create_app()
