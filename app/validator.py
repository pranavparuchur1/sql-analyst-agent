"""SQL guardrails: pure Python, deterministic, no LLM.

We parse into an AST with sqlglot instead of regex-matching keywords. A regex
for "DELETE" rejects a harmless column named `deleted_flag` and misses
`WITH x AS (...) DELETE ...`. The AST knows what each token *is*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

# Anything that writes, changes schema, or escapes the SQL sandbox. `Command` is
# sqlglot's catch-all for statements it doesn't model (VACUUM, GRANT, ...), so
# rejecting it makes the validator fail closed on anything unfamiliar.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command,
    exp.Merge, exp.TruncateTable, exp.Pragma, exp.Attach, exp.Detach,
)  # fmt: skip


@dataclass
class ValidationResult:
    ok: bool
    sql: str = ""
    error: str | None = None
    tables: list[str] = field(default_factory=list)

    @property
    def table_count(self) -> int:
        return len(self.tables)


def validate_sql(
    sql: str,
    allowed_tables: list[str] | set[str],
    *,
    dialect: str = "sqlite",
    row_limit: int = 200,
) -> ValidationResult:
    """Check `sql` is a single safe read query over known tables and return it with a LIMIT.

    Error messages are written for the LLM, because they become retry feedback.
    """
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return ValidationResult(False, error="Empty SQL.")

    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except ParseError as e:
        return ValidationResult(False, sql=sql, error=f"SQL syntax error: {e.errors[0]['description'] if e.errors else e}")

    if len(statements) != 1:
        return ValidationResult(False, sql=sql, error=f"Expected exactly one statement, got {len(statements)}.")
    root = statements[0]

    if not isinstance(root, exp.Query):
        return ValidationResult(False, sql=sql, error=f"Only SELECT queries are allowed, got {type(root).__name__.upper()}.")

    # Walk the whole tree: a write can hide inside a CTE or subquery.
    for node in root.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return ValidationResult(False, sql=sql, error=f"Forbidden operation in query: {type(node).__name__.upper()}.")

    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    referenced = {t.name.lower() for t in root.find_all(exp.Table) if t.name}
    real_tables = sorted(referenced - cte_names)
    allowed = {t.lower() for t in allowed_tables}
    unknown = [t for t in real_tables if t not in allowed]
    if unknown:
        return ValidationResult(
            False, sql=sql, tables=real_tables,
            error=f"Unknown table(s): {', '.join(unknown)}. Available tables: {', '.join(sorted(allowed))}.",
        )  # fmt: skip

    # A missing LIMIT is the most common way an innocent question returns 1M rows.
    # (The DB layer also caps fetched rows; this keeps the database from doing the work.)
    if root.args.get("limit") is None:
        root = root.limit(row_limit)

    return ValidationResult(True, sql=root.sql(dialect=dialect), tables=real_tables)
