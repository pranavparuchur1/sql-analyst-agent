"""Read-only database access, schema discovery and the semantic layer.

The agent only ever talks to the database through the `Database` protocol, so a
Postgres implementation (read-only role + statement_timeout) can replace
`SQLiteDatabase` without touching the graph.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# --- Semantic layer ---------------------------------------------------------
# Column names alone don't tell an LLM what "revenue" means here. This is the
# business knowledge a human analyst would carry in their head.

TABLE_DOCS: dict[str, str] = {
    "customers": "One row per registered customer, with home city, 2-letter US state code and signup date.",
    "products": "Product catalog: name, category (electronics, home, fashion, books, sports) and current list price.",
    "orders": "One row per order: who placed it, order_date, and status (delivered, shipped, canceled).",
    "order_items": "Line items of an order: product, quantity and the unit_price actually charged (after discounts).",
    "payments": "One payment per non-canceled order: method (card, paypal, gift_card), amount and paid_at date.",
    "reviews": "Customer product reviews on delivered orders, score 1 (worst) to 5 (best).",
}

BUSINESS_RULES = """\
- Revenue = SUM(order_items.quantity * order_items.unit_price) over orders whose status != 'canceled'.
  unit_price is what was actually charged, so discounts are already included.
  Never use products.price for revenue: it is today's list price, not what was charged.
- Order counts use COUNT(DISTINCT orders.id). "Orders" excludes canceled ones unless the question says otherwise.
- Average order value (AOV) = revenue / number of non-canceled orders.
- Dates are ISO text 'YYYY-MM-DD'. For a year use order_date >= '2025-01-01' AND order_date < '2026-01-01'.
  Month: strftime('%Y-%m', order_date). Quarter: (CAST(strftime('%m', order_date) AS INTEGER) + 2) / 3.
- Data covers orders from 2025-01-01 to 2026-06-30.
- States are 2-letter codes: Texas = 'TX', California = 'CA', New York = 'NY'.
- Average rating = AVG(reviews.score); mention how many reviews it is based on.
- Category values are lowercase: electronics, home, fashion, books, sports.
"""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool = False
    elapsed_ms: float = 0.0


class Database(Protocol):
    """What the agent needs from a database. Implement this for Postgres to swap engines."""

    dialect: str

    def table_names(self) -> list[str]: ...
    def table_overview(self) -> str: ...
    def schema_for(self, tables: list[str]) -> str: ...
    def explain(self, sql: str) -> None: ...
    def run(self, sql: str) -> QueryResult: ...


@dataclass
class SQLiteDatabase:
    path: Path
    timeout_s: float = 5.0
    max_rows: int = 200
    dialect: str = "sqlite"
    _tables: list[str] | None = field(default=None, repr=False)

    def _connect(self) -> sqlite3.Connection:
        if not Path(self.path).exists():
            raise FileNotFoundError(f"{self.path} not found. Run `python -m data.seed` first.")
        # mode=ro: the database file itself refuses writes, so even a SQL statement
        # that slipped past the validator cannot modify data. Defense in depth.
        uri = f"file:{Path(self.path).resolve().as_posix()}?mode=ro"
        return sqlite3.connect(uri, uri=True, check_same_thread=False)

    # --- schema discovery -------------------------------------------------
    def table_names(self) -> list[str]:
        if self._tables is None:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
            self._tables = [r[0] for r in rows]
        return self._tables

    def _columns(self, conn: sqlite3.Connection, table: str) -> list[str]:
        return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]

    def table_overview(self) -> str:
        """One cheap line per table: enough for the LLM to pick tables, small enough to always send."""
        with closing(self._connect()) as conn:
            lines = [
                f"- {t}: {TABLE_DOCS.get(t, '(no description)')} Columns: {', '.join(self._columns(conn, t))}"
                for t in self.table_names()
            ]
        return "\n".join(lines)

    def schema_for(self, tables: list[str]) -> str:
        """Full detail (DDL + meaning + 3 sample rows) for the chosen tables only.

        Sample rows matter: they show value formats ('TX' vs 'Texas', ISO dates)
        that DDL alone hides, which prevents a whole class of empty-result bugs.
        """
        parts = []
        with closing(self._connect()) as conn:
            for t in tables:
                ddl = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone()
                if not ddl:
                    continue
                cur = conn.execute(f'SELECT * FROM "{t}" LIMIT 3')
                cols = [d[0] for d in cur.description]
                samples = "\n".join("  " + " | ".join(map(str, r)) for r in cur.fetchall())
                parts.append(
                    f"-- {t}: {TABLE_DOCS.get(t, '')}\n{ddl[0]};\n-- sample rows ({' | '.join(cols)}):\n{samples}"
                )
        return "\n\n".join(parts)

    # --- execution --------------------------------------------------------
    def explain(self, sql: str) -> None:
        """Compile the query without running it. Raises on unknown columns/tables, costs ~nothing."""
        with closing(self._connect()) as conn:
            conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()

    def run(self, sql: str) -> QueryResult:
        conn = self._connect()
        deadline = time.monotonic() + self.timeout_s
        # SQLite calls this every N VM instructions; returning non-zero aborts the
        # query with "interrupted". It's the SQLite equivalent of statement_timeout.
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        start = time.perf_counter()
        try:
            cur = conn.execute(sql)
            rows = cur.fetchmany(self.max_rows + 1)
            columns = [d[0] for d in cur.description or []]
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e):
                raise TimeoutError(f"Query exceeded the {self.timeout_s}s timeout") from e
            raise
        finally:
            conn.close()
        truncated = len(rows) > self.max_rows
        return QueryResult(columns, rows[: self.max_rows], truncated, (time.perf_counter() - start) * 1000)
