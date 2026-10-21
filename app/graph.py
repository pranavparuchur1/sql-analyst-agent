"""The agent graph: nodes, routing functions, wiring, compile and checkpointer.

    prepare_turn -> classify ─┬─> small_talk -> END
                              ├─> refuse -> END
                              └─> select_tables -> generate_sql -> validate_sql ─┬─> generate_sql (retry)
                                                                               ├─> give_up -> END
                                                                               ├─> human_review ─┬─> execute_sql
                                                                               │                 ├─> validate_sql (edited)
                                                                               │                 └─> END (rejected)
                                                                               └─> execute_sql ─┬─> summarize -> END
                                                                                                ├─> generate_sql (retry)
                                                                                                └─> give_up -> END

Two routing styles are used on purpose:
- `Command(goto=...)` returned *from a node* (classify, human_review): the node that
  makes the decision also writes state, in one atomic step. Best when the decision
  comes from work the node just did (an LLM call, a human's answer).
- `add_conditional_edges` with a pure routing function (after validate/execute):
  the node just records facts, and a separate function reads state and picks the
  next hop. Best when routing is a policy over state you want to unit-test alone.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, RetryPolicy, interrupt

from app.config import Settings, get_llm, get_settings
from app.db import BUSINESS_RULES, Database, SQLiteDatabase
from app.prompts import ANSWER_PROMPT, ROUTER_PROMPT, SQL_PROMPT, TABLE_SELECTION_PROMPT
from app.schemas import FinalAnswer, IntentDecision, SQLQuery, TableSelection
from app.state import AgentState
from app.validator import validate_sql as run_validator

SMALL_TALK_REPLY = (
    "Hi! I answer questions about the shop's sales data: customers, products, orders, payments "
    "and reviews. Try \"Top 5 products by revenue in 2025?\" or \"Monthly order count for 2026\"."
)
REFUSE_REPLY = (
    "I can only read and analyse data. I can't modify, delete or create anything, and I won't "
    "change how I operate. Ask me a question about the data instead."
)

# Exception class names that mean "try again later", across OpenAI/Anthropic/httpx.
# Matching by name avoids importing every provider SDK just to reference its classes.
_TRANSIENT_NAMES = {
    "RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError",
    "ServiceUnavailableError", "OverloadedError", "ConnectError", "ReadTimeout",
}  # fmt: skip


def is_transient_llm_error(exc: Exception) -> bool:
    """Retry network blips and rate limits, never bugs or bad output (retrying those just burns money)."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return type(exc).__name__ in _TRANSIENT_NAMES or getattr(exc, "status_code", None) in {429, 500, 502, 503, 529}


LLM_RETRY = RetryPolicy(max_attempts=3, initial_interval=0.5, backoff_factor=2.0, retry_on=is_transient_llm_error)


def default_checkpointer(path: Path | str | None = None) -> SqliteSaver:
    """A SQLite-backed checkpointer: conversations and paused approvals survive a restart."""
    path = Path(path or get_settings().checkpoint_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False because FastAPI serves requests from a thread pool;
    # SqliteSaver serialises access with its own lock.
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def _history(state: AgentState, window: int) -> list[AnyMessage]:
    # The last message is the current question, which the prompts take separately.
    return list(state.get("messages", []))[:-1][-window:]


def _format_rows(columns: list[str], rows: list[list[Any]], limit: int = 50) -> str:
    lines = [" | ".join(columns)] + [" | ".join(map(str, r)) for r in rows[:limit]]
    return "\n".join(lines)


def _failure(state: AgentState, stage: str, sql: str, error: str) -> dict[str, Any]:
    return {
        "error": error,
        "error_history": [
            {"question": state.get("question"), "attempt": state.get("attempts", 0), "stage": stage,
             "sql": sql, "error": error}
        ],
    }  # fmt: skip


def build_graph(
    llm: BaseChatModel | Any = None,
    db: Database | None = None,
    settings: Settings | None = None,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    """Wire and compile the agent. Every dependency is injectable so tests can pass fakes."""
    settings = settings or get_settings()
    llm = llm or get_llm(settings)
    db = db or SQLiteDatabase(settings.db_path, timeout_s=settings.query_timeout_s, max_rows=settings.row_limit)
    checkpointer = checkpointer if checkpointer is not None else default_checkpointer(settings.checkpoint_db)
    window = settings.history_window
    method = settings.structured_output_method

    # LCEL: `prompt | model` composes two Runnables into one. .invoke(dict) fills the
    # template, calls the model, and returns a validated Pydantic object.
    router_chain = ROUTER_PROMPT | llm.with_structured_output(IntentDecision, method=method)
    table_chain = TABLE_SELECTION_PROMPT | llm.with_structured_output(TableSelection, method=method)
    sql_chain = SQL_PROMPT | llm.with_structured_output(SQLQuery, method=method)
    answer_chain = ANSWER_PROMPT | llm.with_structured_output(FinalAnswer, method=method)

    def normalize(sql: str) -> str:
        result = run_validator(sql, db.table_names(), dialect=db.dialect, row_limit=settings.row_limit)
        return result.sql if result.ok else sql

    # --- nodes ---------------------------------------------------------------

    def prepare_turn(state: AgentState) -> dict[str, Any]:
        # The checkpointer restores *all* of last turn's state for this thread, so
        # without this reset a new question would inherit last turn's sql, attempts,
        # approval... Only `messages` and `error_history` are meant to carry over.
        return {
            "messages": [HumanMessage(state["question"])],
            "intent": "", "tables": [], "schema_context": "", "sql": "", "sql_explanation": "",
            "attempts": 0, "error": None, "needs_approval": False, "approval_reason": "",
            "approved_sql": None, "columns": [], "rows": [], "answer": "", "status": "running",
        }  # fmt: skip

    def classify(state: AgentState) -> Command[Literal["select_tables", "small_talk", "refuse"]]:
        decision = router_chain.invoke({"history": _history(state, window), "question": state["question"]})
        goto = {"data_question": "select_tables", "small_talk": "small_talk", "unsafe_request": "refuse"}
        # Command = "update these keys AND go here next", decided by the node itself.
        return Command(update={"intent": decision.intent}, goto=goto[decision.intent])

    def select_tables(state: AgentState) -> dict[str, Any]:
        # Two-step schema retrieval: the LLM sees a cheap one-line overview of every
        # table, then only the chosen tables get full DDL + samples. This keeps the SQL
        # prompt small, which matters a lot with 200-table warehouses.
        choice = table_chain.invoke(
            {"overview": db.table_overview(), "rules": BUSINESS_RULES,
             "history": _history(state, window), "question": state["question"]}
        )  # fmt: skip
        known = set(db.table_names())
        tables = [t for t in dict.fromkeys(choice.tables) if t in known]
        if not tables or len(tables) != len(set(choice.tables)):
            # A hallucinated table name means the model is confused; showing it the
            # whole (small) schema is cheaper than failing the question.
            tables = db.table_names()
        return {"tables": tables, "schema_context": db.schema_for(tables)}

    def generate_sql(state: AgentState) -> dict[str, Any]:
        feedback = ""
        if state.get("error"):
            # Self-correction: show the model its own previous SQL and the *exact*
            # error. Specific feedback fixes most errors in one retry.
            feedback = (
                "\n\nYour previous attempt failed. Fix it.\n"
                f"Previous SQL:\n{state.get('sql', '')}\nError:\n{state['error']}"
            )
        query = sql_chain.invoke(
            {"dialect": db.dialect, "row_limit": settings.row_limit, "rules": BUSINESS_RULES,
             "schema": state["schema_context"], "history": _history(state, window),
             "question": state["question"], "retry_feedback": feedback}
        )  # fmt: skip
        return {"sql": query.sql, "sql_explanation": query.explanation, "attempts": state.get("attempts", 0) + 1}

    def validate_sql(state: AgentState) -> dict[str, Any]:
        result = run_validator(state["sql"], db.table_names(), dialect=db.dialect, row_limit=settings.row_limit)
        if not result.ok:
            return _failure(state, "validate", state["sql"], result.error or "invalid SQL")
        try:
            # EXPLAIN catches bad columns/functions without executing anything.
            db.explain(result.sql)
        except Exception as e:  # noqa: BLE001 - any DB error is feedback for the LLM
            return {"sql": result.sql, **_failure(state, "explain", result.sql, f"{type(e).__name__}: {e}")}

        approved = state.get("approved_sql")
        already_approved = approved is not None and normalize(approved) == result.sql
        needs_approval = result.table_count >= settings.approval_join_threshold and not already_approved
        return {
            "sql": result.sql,
            "error": None,
            "needs_approval": needs_approval,
            "approval_reason": (
                f"Query joins {result.table_count} tables ({', '.join(result.tables)}); "
                f"the approval threshold is {settings.approval_join_threshold}."
                if needs_approval else ""
            ),
            "status": "needs_approval" if needs_approval else "running",
        }  # fmt: skip

    def human_review(state: AgentState) -> Command[Literal["execute_sql", "validate_sql", "__end__"]]:
        # interrupt() saves a checkpoint and stops the run. The value we pass is what
        # the caller sees. When resumed with Command(resume=x), this node runs again
        # from the top and interrupt() returns x. So nothing before it may have side effects.
        decision = interrupt(
            {"type": "sql_approval", "sql": state["sql"], "explanation": state.get("sql_explanation", ""),
             "reason": state.get("approval_reason", ""), "tables": state.get("tables", [])}
        )  # fmt: skip
        if isinstance(decision, bool):
            decision = {"approved": decision}
        edited = (decision.get("edited_sql") or "").strip()
        if edited:
            # A human edit is still untrusted input: it goes back through validation.
            return Command(update={"sql": edited, "approved_sql": edited, "needs_approval": False}, goto="validate_sql")
        if decision.get("approved"):
            return Command(update={"approved_sql": state["sql"], "needs_approval": False}, goto="execute_sql")
        reply = "Okay, I didn't run that query. Feel free to rephrase or narrow the question."
        return Command(
            update={"status": "rejected", "needs_approval": False, "answer": reply, "messages": [AIMessage(reply)]},
            goto=END,
        )

    def execute_sql(state: AgentState) -> dict[str, Any]:
        try:
            result = db.run(state["sql"])
        except Exception as e:  # noqa: BLE001 - every DB error becomes retry feedback
            return _failure(state, "execute", state["sql"], f"{type(e).__name__}: {e}")
        return {"columns": result.columns, "rows": [list(r) for r in result.rows], "error": None}

    def summarize(state: AgentState) -> dict[str, Any]:
        rows = state.get("rows", [])
        truncated = len(rows) >= settings.row_limit
        final = answer_chain.invoke(
            {"history": _history(state, window), "question": state["question"], "sql": state["sql"],
             "columns": ", ".join(state.get("columns", [])), "row_count": len(rows),
             "truncated_note": f", truncated at {settings.row_limit}" if truncated else "",
             "rows": _format_rows(state.get("columns", []), rows)}
        )  # fmt: skip
        answer = final.answer
        if final.caveats:
            answer += "\n\nNotes: " + "; ".join(final.caveats)
        # The SQL goes into memory with the answer, so a follow-up like "now only for
        # Texas" can modify the previous query instead of starting from scratch.
        return {"answer": answer, "status": "answered", "messages": [AIMessage(f"{answer}\n\n[SQL used]\n{state['sql']}")]}

    def give_up(state: AgentState) -> dict[str, Any]:
        reply = (
            f"Sorry, I couldn't build a working query after {state.get('attempts', 0)} attempts. "
            f"The last problem was: {state.get('error')}. Try rephrasing, or name the metric and time range explicitly."
        )
        return {"answer": reply, "status": "failed", "messages": [AIMessage(reply)]}

    def small_talk(state: AgentState) -> dict[str, Any]:
        return {"answer": SMALL_TALK_REPLY, "status": "small_talk", "messages": [AIMessage(SMALL_TALK_REPLY)]}

    def refuse(state: AgentState) -> dict[str, Any]:
        return {"answer": REFUSE_REPLY, "status": "refused", "messages": [AIMessage(REFUSE_REPLY)]}

    # --- routing functions (pure: state in, next node name out) ---------------

    def route_after_validation(state: AgentState) -> str:
        if state.get("error"):
            return "give_up" if state.get("attempts", 0) >= settings.max_attempts else "generate_sql"
        return "human_review" if state.get("needs_approval") else "execute_sql"

    def route_after_execution(state: AgentState) -> str:
        if not state.get("error"):
            return "summarize"
        return "give_up" if state.get("attempts", 0) >= settings.max_attempts else "generate_sql"

    # --- wiring ---------------------------------------------------------------

    graph = StateGraph(AgentState)
    graph.add_node("prepare_turn", prepare_turn)
    # RetryPolicy re-runs a node on transient errors (rate limits, timeouts) with
    # exponential backoff. Only LLM nodes get it: DB errors are handled as feedback.
    graph.add_node("classify", classify, retry_policy=LLM_RETRY)
    graph.add_node("select_tables", select_tables, retry_policy=LLM_RETRY)
    graph.add_node("generate_sql", generate_sql, retry_policy=LLM_RETRY)
    graph.add_node("validate_sql", validate_sql)
    graph.add_node("human_review", human_review)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("summarize", summarize, retry_policy=LLM_RETRY)
    graph.add_node("give_up", give_up)
    graph.add_node("small_talk", small_talk)
    graph.add_node("refuse", refuse)

    graph.add_edge(START, "prepare_turn")
    graph.add_edge("prepare_turn", "classify")
    # classify and human_review route themselves with Command; their return-type
    # annotations tell LangGraph the possible destinations (for drawing and checks).
    graph.add_edge("select_tables", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")
    graph.add_conditional_edges(
        "validate_sql", route_after_validation, ["generate_sql", "give_up", "human_review", "execute_sql"]
    )
    graph.add_conditional_edges("execute_sql", route_after_execution, ["summarize", "generate_sql", "give_up"])
    for terminal in ("summarize", "give_up", "small_talk", "refuse"):
        graph.add_edge(terminal, END)

    return graph.compile(checkpointer=checkpointer)
