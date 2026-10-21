"""All prompt templates in one place, so prompt changes are reviewable as a diff.

Large reference text (business rules, schema) is passed as template variables
rather than baked in with f-strings: values are never parsed for {braces}, so a
schema or SQL containing `{` can't break the template.
"""

from __future__ import annotations

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the front door of a data assistant for an online shop's sales database "
            "(customers, products, orders, payments, reviews). Classify the user's latest message. "
            "Use the conversation history to recognise follow-ups, which are data questions.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

TABLE_SELECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Pick the database tables needed to answer the question.\n\nAvailable tables:\n{overview}\n\n"
            "Business rules:\n{rules}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

# Few-shot examples teach house style (aliases, date filtering, excluding canceled
# orders) more reliably than more instructions would.
FEW_SHOT_EXAMPLES = [
    {
        "question": "How many orders were delivered in March 2026?",
        "sql": "SELECT COUNT(*) AS delivered_orders FROM orders "
        "WHERE status = 'delivered' AND order_date >= '2026-03-01' AND order_date < '2026-04-01'",
    },
    {
        "question": "Revenue by product category in 2025",
        "sql": "SELECT p.category, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
        "FROM order_items oi JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id "
        "WHERE o.status != 'canceled' AND o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01' "
        "GROUP BY p.category ORDER BY revenue DESC",
    },
    {
        "question": "Which 3 cities have the most customers?",
        "sql": "SELECT city, state, COUNT(*) AS customers FROM customers "
        "GROUP BY city, state ORDER BY customers DESC LIMIT 3",
    },
]

_example_prompt = ChatPromptTemplate.from_messages([("human", "{question}"), ("ai", "{sql}")])
FEW_SHOT = FewShotChatMessagePromptTemplate(examples=FEW_SHOT_EXAMPLES, example_prompt=_example_prompt)

SQL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior analytics engineer writing {dialect} SQL for a business user.\n"
            "Write exactly one read-only SELECT query (CTEs allowed). Return at most {row_limit} rows.\n"
            "Only use tables and columns shown in the schema. If the question is a follow-up, build on "
            "the SQL from earlier in the conversation.\n\n"
            "Business rules:\n{rules}\n\nSchema:\n{schema}\n\nExamples of good queries follow.",
        ),
        FEW_SHOT,
        MessagesPlaceholder("history"),
        # retry_feedback is empty on the first attempt and carries the previous SQL plus
        # its exact error on retries: the self-correction loop's whole interface.
        ("human", "{question}{retry_feedback}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You explain query results to a non-technical business user. Be concise and concrete: lead "
            "with the answer, then the key numbers. Use only the numbers in the result. Round money to 2 "
            "decimals. If the result is empty, say so plainly and suggest why.",
        ),
        MessagesPlaceholder("history"),
        (
            "human",
            "Question: {question}\n\nSQL that was run:\n{sql}\n\nResult columns: {columns}\n"
            "Result rows ({row_count} shown{truncated_note}):\n{rows}",
        ),
    ]
)
