"""Interactive terminal chat that shows the agent thinking, step by step.

    python -m app.cli [--thread my-session] [--model openai:gpt-4.1-mini]

stream_mode="updates" yields one {node_name: partial_update} dict per finished
node, so every SQL attempt and error is printed as it happens.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from typing import Any

from langgraph.types import Command

from app.config import get_llm, get_settings
from app.graph import build_graph, default_checkpointer


def _show(node: str, update: dict[str, Any]) -> None:
    if node == "classify":
        print(f"  [route]    intent = {update.get('intent')}")
    elif node == "select_tables":
        print(f"  [tables]   {', '.join(update.get('tables', []))}")
    elif node == "generate_sql":
        print(f"  [sql #{update.get('attempts')}]   {update.get('sql')}")
    elif node == "validate_sql":
        if update.get("error"):
            print(f"  [blocked]  {update['error']}")
        elif update.get("needs_approval"):
            print(f"  [approval] {update.get('approval_reason')}")
        else:
            print("  [valid]    passed guardrails + EXPLAIN")
    elif node == "execute_sql":
        if update.get("error"):
            print(f"  [db error] {update['error']}")
        else:
            print(f"  [ran]      {len(update.get('rows', []))} rows")
    elif node in {"summarize", "give_up", "small_talk", "refuse", "human_review"} and update.get("answer"):
        print(f"\nagent> {update['answer']}\n")


def _ask_approval(payload: dict[str, Any]) -> dict[str, Any]:
    print(f"\n  ! Approval needed: {payload.get('reason')}\n  SQL: {payload.get('sql')}")
    while True:
        choice = input("  Run it? [y]es / [n]o / [e]dit: ").strip().lower()
        if choice in {"y", "yes"}:
            return {"approved": True}
        if choice in {"n", "no"}:
            return {"approved": False}
        if choice in {"e", "edit"}:
            return {"approved": True, "edited_sql": input("  Edited SQL: ")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the SQL analyst agent.")
    parser.add_argument("--thread", default=None, help="Thread id; reuse it to continue a conversation.")
    parser.add_argument("--model", default=None, help="Override LLM_MODEL, e.g. openai:gpt-4.1-mini")
    args = parser.parse_args()

    settings = get_settings()
    saver = default_checkpointer(settings.checkpoint_db)
    graph = build_graph(llm=get_llm(settings, args.model), settings=settings, checkpointer=saver)
    config = {"configurable": {"thread_id": args.thread or f"cli-{uuid.uuid4().hex[:8]}"}}
    print(f"SQL analyst ({args.model or settings.llm_model}), thread {config['configurable']['thread_id']}. Ctrl+C to quit.")

    try:
        while True:
            question = input("you> ").strip()
            if not question:
                continue
            payload: Any = {"question": question}
            try:
                # Loop because one question may pause for approval, and resuming may pause again.
                while payload is not None:
                    interrupt_value = None
                    for chunk in graph.stream(payload, config, stream_mode="updates"):
                        for node, update in chunk.items():
                            if node == "__interrupt__":
                                interrupt_value = update[0].value
                            elif update:
                                _show(node, update)
                    payload = Command(resume=_ask_approval(interrupt_value)) if interrupt_value else None
            except (KeyboardInterrupt, EOFError):
                raise
            except Exception as e:  # noqa: BLE001 - keep the session alive on LLM/network errors
                print(f"\n  [error] {type(e).__name__}: {str(e)[:300]}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
    finally:
        saver.conn.close()


if __name__ == "__main__":
    sys.exit(main())
