"""The graph's shared state.

Every node receives the current state and returns a *partial* update. For most
keys the update simply overwrites the old value. Keys annotated with a reducer
are merged instead: that is how memory and the error log accumulate.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Conversation memory. add_messages appends (and de-duplicates by message id)
    # instead of overwriting, and the checkpointer persists it per thread_id.
    messages: Annotated[list[AnyMessage], add_messages]
    # Append-only audit log of every failed attempt across all turns of a thread.
    error_history: Annotated[list[dict[str, Any]], operator.add]

    # --- per-turn fields: reset by prepare_turn at the start of every question ---
    question: str
    intent: str
    tables: list[str]
    schema_context: str
    sql: str
    sql_explanation: str
    attempts: int
    error: str | None
    needs_approval: bool
    approval_reason: str
    approved_sql: str | None
    columns: list[str]
    rows: list[list[Any]]
    answer: str
    status: str  # answered | rejected | failed | needs_approval | small_talk | refused
