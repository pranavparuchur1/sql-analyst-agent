import pytest

from app.validator import validate_sql

TABLES = ["customers", "products", "orders", "order_items", "payments", "reviews"]


def check(sql: str):
    return validate_sql(sql, TABLES, row_limit=100)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "UPDATE orders SET status = 'x'",
        "INSERT INTO orders VALUES (1, 1, '2025-01-01', 'delivered')",
        "DROP TABLE orders",
        "CREATE TABLE t (a INT)",
        "ALTER TABLE orders ADD COLUMN x INT",
        "PRAGMA writable_schema = 1",
        "ATTACH DATABASE 'other.db' AS other",
        "VACUUM",
    ],
)
def test_rejects_writes_and_non_queries(sql):
    result = check(sql)
    assert not result.ok
    assert result.error


def test_rejects_write_hidden_in_cte():
    result = check("WITH gone AS (DELETE FROM orders RETURNING id) SELECT * FROM gone")
    assert not result.ok


def test_rejects_multiple_statements():
    result = check("SELECT 1; DROP TABLE orders")
    assert not result.ok
    assert "exactly one statement" in result.error


def test_trailing_semicolon_is_fine():
    assert check("SELECT id FROM orders;").ok


def test_allows_cte_names_as_tables():
    result = check("WITH big AS (SELECT * FROM orders WHERE status = 'delivered') SELECT COUNT(*) FROM big")
    assert result.ok, result.error
    assert result.tables == ["orders"]


def test_column_named_like_a_keyword_passes():
    # The reason we use a parser and not a regex: 'deleted_flag' is not a DELETE.
    result = validate_sql("SELECT deleted_flag, updated_at FROM orders", TABLES + [])
    assert result.ok, result.error


def test_rejects_unknown_table_and_lists_known_ones():
    result = check("SELECT * FROM users")
    assert not result.ok
    assert "users" in result.error and "customers" in result.error


def test_syntax_error_is_reported():
    result = check("SELEC id FROM orders WHERE")
    assert not result.ok


def test_adds_limit_when_missing_and_keeps_existing():
    assert check("SELECT id FROM orders").sql.endswith("LIMIT 100")
    assert check("SELECT id FROM orders LIMIT 5").sql.endswith("LIMIT 5")


def test_limit_added_to_union():
    result = check("SELECT id FROM orders UNION SELECT id FROM payments")
    assert result.ok and "LIMIT 100" in result.sql


def test_counts_distinct_tables_for_approval_rule():
    sql = (
        "SELECT c.state, SUM(oi.quantity * oi.unit_price) FROM customers c "
        "JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id "
        "JOIN products p ON p.id = oi.product_id JOIN orders o2 ON o2.id = o.id GROUP BY c.state"
    )
    result = check(sql)
    assert result.ok
    assert result.table_count == 4  # orders joined twice still counts once


def test_empty_sql():
    assert not check("   ").ok
