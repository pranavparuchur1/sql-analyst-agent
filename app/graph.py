"""The agent graph: nodes, wiring and compile.

Stage 3 version: a straight line with no branches or loops, to show the basics.
A node is a plain function `state -> partial update`. Edges say what runs next.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings, get_llm, get_settings
from app.db import BUSINESS_RULES, Database, SQLiteDatabase
from app.prompts import ANSWER_PROMPT, SQL_PROMPT, TABLE_SELECTION_PROMPT
from app.schemas import FinalAnswer, SQLQuery, TableSelection
from app.state import AgentState
from app.validator import validate_sql as run_validator


def _history(state: AgentState, window: int) -> list[AnyMessage]:
    # The last message is the current question, which the prompts take separately.
    return list(state.get("messages", []))[:-1][-window:]


def _format_rows(columns: list[str], rows: list[list[Any]], limit: int = 50) -> str:
    lines = [" | ".join(columns)] + [" | ".join(map(str, r)) for r in rows[:limit]]
    return "\n".join(lines)


def build_graph(
    llm: BaseChatModel | Any = None,
    db: Database | None = None,
    settings: Settings | None = None,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    settings = settings or get_settings()
    llm = llm or get_llm(settings)
    db = db or SQLiteDatabase(settings.db_path, timeout_s=settings.query_timeout_s, max_rows=settings.row_limit)

    # LCEL: `prompt | model` composes two Runnables into one. Calling .invoke(dict)
    # fills the template, sends it to the model, and parses into the Pydantic schema.
    table_chain = TABLE_SELECTION_PROMPT | llm.with_structured_output(TableSelection)
    sql_chain = SQL_PROMPT | llm.with_structured_output(SQLQuery)
    answer_chain = ANSWER_PROMPT | llm.with_structured_output(FinalAnswer)

    def prepare_turn(state: AgentState) -> dict[str, Any]:
        return {"messages": [HumanMessage(state["question"])], "attempts": 0, "error": None}

    def select_tables(state: AgentState) -> dict[str, Any]:
        choice = table_chain.invoke(
            {"overview": db.table_overview(), "rules": BUSINESS_RULES,
             "history": _history(state, settings.history_window), "question": state["question"]}
        )  # fmt: skip
        return {"tables": choice.tables, "schema_context": db.schema_for(choice.tables)}

    def generate_sql(state: AgentState) -> dict[str, Any]:
        query = sql_chain.invoke(
            {"dialect": db.dialect, "row_limit": settings.row_limit, "rules": BUSINESS_RULES,
             "schema": state["schema_context"], "history": _history(state, settings.history_window),
             "question": state["question"], "retry_feedback": ""}
        )  # fmt: skip
        return {"sql": query.sql, "sql_explanation": query.explanation, "attempts": state.get("attempts", 0) + 1}

    def validate_sql(state: AgentState) -> dict[str, Any]:
        result = run_validator(state["sql"], db.table_names(), dialect=db.dialect, row_limit=settings.row_limit)
        return {"sql": result.sql, "error": result.error}

    def execute_sql(state: AgentState) -> dict[str, Any]:
        result = db.run(state["sql"])
        return {"columns": result.columns, "rows": [list(r) for r in result.rows]}

    def summarize(state: AgentState) -> dict[str, Any]:
        final = answer_chain.invoke(
            {"history": _history(state, settings.history_window), "question": state["question"],
             "sql": state["sql"], "columns": ", ".join(state["columns"]), "row_count": len(state["rows"]),
             "truncated_note": "", "rows": _format_rows(state["columns"], state["rows"])}
        )  # fmt: skip
        return {
            "answer": final.answer,
            "status": "answered",
            "messages": [AIMessage(f"{final.answer}\n\n[SQL used]\n{state['sql']}")],
        }

    graph = StateGraph(AgentState)
    for name, fn in [("prepare_turn", prepare_turn), ("select_tables", select_tables),
                     ("generate_sql", generate_sql), ("validate_sql", validate_sql),
                     ("execute_sql", execute_sql), ("summarize", summarize)]:  # fmt: skip
        graph.add_node(name, fn)
    graph.add_edge(START, "prepare_turn")
    graph.add_edge("prepare_turn", "select_tables")
    graph.add_edge("select_tables", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")
    graph.add_edge("validate_sql", "execute_sql")
    graph.add_edge("execute_sql", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile(checkpointer=checkpointer)
