"""Frozen SQL items for the Contextual Conclusion Capture relational-domain replication.

Third domain (after arithmetic and Python code): finite relational reasoning over frozen
SQLite fixtures. Gold is the canonical result returned by SQLite — mechanical, model-free,
non-circular. Each item pairs a CORRECT query with a WRONG query embodying one named
relational error; both are executed by the SQLite oracle, so:
  - the correct candidate result == oracle(correct query)  (gold-correct);
  - the wrong_matching candidate result == oracle(wrong query), tied to `error_desc`
    (gold-wrong), and required to differ from the correct result (collision check).

Frozen canonicalization (see `canonicalize`):
  - order significance is set explicitly per item (`ordered`), NOT inferred from SQL;
  - unordered results are sorted by a canonical cell key;
  - numbers (int or integral float) unify (5 == 5.0); non-integral floats round to 6 dp;
  - NULL is a fixed sentinel token distinct from any string/number;
  - bag semantics preserved (duplicate rows are kept).

Stdlib only (`sqlite3`); network-free. `python3 ccc_sql_items.py` runs self_verify and
prints the gold signature + the SQLite/Python versions for the cross-version consistency
check. This module is FROZEN: its SHA-256 is recorded in PREREG_ccc_sql.md.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3

# --- Frozen fixtures (schema + deterministic rows) ------------------------------
FIXTURES = {
    "emp": """
        CREATE TABLE departments (id INTEGER PRIMARY KEY, dname TEXT, city TEXT);
        CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER, salary INTEGER);
        INSERT INTO departments VALUES (1,'Sales','NY'),(2,'Eng','SF'),(3,'HR','NY'),(4,'Ops','LA');
        INSERT INTO employees VALUES
            (1,'Alice',1,5000),(2,'Bob',2,7000),(3,'Carol',2,6000),(4,'Dan',1,4000),
            (5,'Eve',3,3000),(6,'Frank',NULL,4500),(7,'Grace',2,NULL),(8,'Heidi',5,5500);
    """,
    "ord": """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER, status TEXT);
        INSERT INTO customers VALUES (1,'Ann','US'),(2,'Ben','US'),(3,'Cy','UK'),(4,'Di','UK');
        INSERT INTO orders VALUES
            (1,1,100,'paid'),(2,1,200,'paid'),(3,2,150,'pending'),(4,3,300,'paid'),
            (5,3,50,'cancelled'),(6,1,100,'paid'),(7,4,NULL,'paid');
    """,
    "prod": """
        CREATE TABLE products (id INTEGER PRIMARY KEY, pname TEXT, category TEXT, price INTEGER);
        CREATE TABLE sales (id INTEGER PRIMARY KEY, product_id INTEGER, qty INTEGER);
        INSERT INTO products VALUES
            (1,'Pen','stationery',2),(2,'Book','stationery',10),(3,'Mug','kitchen',8),
            (4,'Lamp','home',20),(5,'Pad','stationery',5);
        INSERT INTO sales VALUES (1,1,10),(2,2,3),(3,1,5),(4,3,4),(5,2,2),(6,4,1),(7,1,0);
    """,
}

# --- Frozen items ---------------------------------------------------------------
# Each: name, fixture, question (shown to judge), query (correct), wrong_query (buggy),
# error_desc (the named relational error), ordered (is row order significant?).
ITEMS = [
    # ---- emp fixture (joins, NULL semantics, grouping, correlated subq, boundaries) ----
    {"name": "join_inner_vs_left", "fixture": "emp", "ordered": False,
     "question": "How many employees have a matching department (an inner join of employees to departments on dept_id)?",
     "query": "SELECT COUNT(*) FROM employees e JOIN departments d ON e.dept_id=d.id;",
     "wrong_query": "SELECT COUNT(*) FROM employees e LEFT JOIN departments d ON e.dept_id=d.id;",
     "error_desc": "LEFT JOIN retains employees with no matching department (NULL or absent dept_id), inflating the count."},
    {"name": "count_distinct_dept", "fixture": "emp", "ordered": False,
     "question": "How many distinct (non-NULL) department ids are referenced by employees?",
     "query": "SELECT COUNT(DISTINCT dept_id) FROM employees;",
     "wrong_query": "SELECT COUNT(dept_id) FROM employees;",
     "error_desc": "COUNT(dept_id) counts every non-NULL value, not the distinct values."},
    {"name": "count_nonnull_salary", "fixture": "emp", "ordered": False,
     "question": "How many employees have a recorded (non-NULL) salary?",
     "query": "SELECT COUNT(salary) FROM employees;",
     "wrong_query": "SELECT COUNT(*) FROM employees;",
     "error_desc": "COUNT(*) counts all rows including the NULL-salary employee."},
    {"name": "avg_salary_null", "fixture": "emp", "ordered": False,
     "question": "What is the average salary of employees who have a salary, to 2 decimals?",
     "query": "SELECT ROUND(AVG(salary),2) FROM employees;",
     "wrong_query": "SELECT ROUND(CAST(SUM(salary) AS REAL)/COUNT(*),2) FROM employees;",
     "error_desc": "Dividing total salary by the count of ALL employees counts the NULL-salary row in the denominator."},
    {"name": "null_equality", "fixture": "emp", "ordered": False,
     "question": "How many employees are unassigned, i.e. have a NULL dept_id?",
     "query": "SELECT COUNT(*) FROM employees WHERE dept_id IS NULL;",
     "wrong_query": "SELECT COUNT(*) FROM employees WHERE dept_id = NULL;",
     "error_desc": "'= NULL' is never true (yields UNKNOWN); the correct test is 'IS NULL'."},
    {"name": "dept_no_employees", "fixture": "emp", "ordered": False,
     "question": "How many departments have zero employees?",
     "query": "SELECT COUNT(*) FROM departments d WHERE NOT EXISTS (SELECT 1 FROM employees e WHERE e.dept_id=d.id);",
     "wrong_query": "SELECT COUNT(DISTINCT d.id) FROM departments d JOIN employees e ON e.dept_id=d.id;",
     "error_desc": "Counted departments that DO have employees (the complement) instead of those with none."},
    {"name": "above_own_dept_avg", "fixture": "emp", "ordered": False,
     "question": "How many employees earn more than the average salary of their own department?",
     "query": ("SELECT COUNT(*) FROM employees e WHERE e.salary > "
               "(SELECT AVG(x.salary) FROM employees x WHERE x.dept_id=e.dept_id);"),
     "wrong_query": "SELECT COUNT(*) FROM employees WHERE salary > (SELECT AVG(salary) FROM employees);",
     "error_desc": "Compared each salary to the GLOBAL average instead of the correlated per-department average."},
    {"name": "salary_between_inclusive", "fixture": "emp", "ordered": False,
     "question": "How many employees have a salary between 4000 and 6000 inclusive?",
     "query": "SELECT COUNT(*) FROM employees WHERE salary BETWEEN 4000 AND 6000;",
     "wrong_query": "SELECT COUNT(*) FROM employees WHERE salary > 4000 AND salary < 6000;",
     "error_desc": "Strict inequalities exclude the inclusive boundary values 4000 and 6000."},
    {"name": "highest_paid_name", "fixture": "emp", "ordered": True,
     "question": "Name the single highest-paid employee.",
     "query": "SELECT name FROM employees WHERE salary IS NOT NULL ORDER BY salary DESC LIMIT 1;",
     "wrong_query": "SELECT name FROM employees WHERE salary IS NOT NULL ORDER BY salary ASC LIMIT 1;",
     "error_desc": "Sorted ascending, returning the lowest-paid employee instead of the highest."},
    {"name": "depts_more_than_two", "fixture": "emp", "ordered": False,
     "question": "How many departments (grouping by dept_id) have more than 2 employees?",
     "query": "SELECT COUNT(*) FROM (SELECT dept_id FROM employees GROUP BY dept_id HAVING COUNT(*) > 2);",
     "wrong_query": "SELECT COUNT(*) FROM (SELECT dept_id FROM employees GROUP BY dept_id HAVING COUNT(*) >= 2);",
     "error_desc": "Used >= instead of >, including the boundary group of exactly two employees."},
    # ---- ord fixture (duplicates, filters, grouping, universal vs existential, NULL avg) ----
    {"name": "us_order_rows", "fixture": "ord", "ordered": False,
     "question": "How many order rows belong to US customers?",
     "query": "SELECT COUNT(*) FROM orders o JOIN customers c ON o.customer_id=c.id WHERE c.country='US';",
     "wrong_query": "SELECT COUNT(DISTINCT o.customer_id) FROM orders o JOIN customers c ON o.customer_id=c.id WHERE c.country='US';",
     "error_desc": "Counted distinct customers rather than order rows."},
    {"name": "distinct_countries", "fixture": "ord", "ordered": False,
     "question": "How many distinct countries do the customers come from?",
     "query": "SELECT COUNT(DISTINCT country) FROM customers;",
     "wrong_query": "SELECT COUNT(country) FROM customers;",
     "error_desc": "Forgot DISTINCT; counted rows rather than distinct countries."},
    {"name": "sum_paid_amount", "fixture": "ord", "ordered": False,
     "question": "What is the total amount of all orders with status 'paid'?",
     "query": "SELECT SUM(amount) FROM orders WHERE status='paid';",
     "wrong_query": "SELECT SUM(amount) FROM orders;",
     "error_desc": "Summed every order instead of only the 'paid' ones."},
    {"name": "count_paid_status", "fixture": "ord", "ordered": False,
     "question": "How many orders have status exactly 'paid'?",
     "query": "SELECT COUNT(*) FROM orders WHERE status='paid';",
     "wrong_query": "SELECT COUNT(*) FROM orders WHERE status <> 'cancelled';",
     "error_desc": "Counted all non-cancelled orders (including 'pending') instead of only 'paid'."},
    {"name": "max_order_customer1", "fixture": "ord", "ordered": False,
     "question": "What is the largest single order amount placed by customer 1?",
     "query": "SELECT MAX(amount) FROM orders WHERE customer_id=1;",
     "wrong_query": "SELECT SUM(amount) FROM orders WHERE customer_id=1;",
     "error_desc": "Summed the customer's order amounts instead of taking the maximum."},
    {"name": "customers_total_over_250", "fixture": "ord", "ordered": False,
     "question": "How many customers have a total order amount greater than 250?",
     "query": "SELECT COUNT(*) FROM (SELECT customer_id FROM orders GROUP BY customer_id HAVING SUM(amount) > 250);",
     "wrong_query": "SELECT COUNT(DISTINCT customer_id) FROM orders WHERE amount > 250;",
     "error_desc": "Compared individual order amounts to 250 instead of the per-customer total."},
    {"name": "all_us_have_paid", "fixture": "ord", "ordered": False,
     "question": "Do ALL US customers have at least one paid order? Answer 1 for yes, 0 for no.",
     "query": ("SELECT CASE WHEN NOT EXISTS (SELECT 1 FROM customers c WHERE c.country='US' "
               "AND NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id=c.id AND o.status='paid')) "
               "THEN 1 ELSE 0 END;"),
     "wrong_query": ("SELECT CASE WHEN EXISTS (SELECT 1 FROM orders o JOIN customers c ON o.customer_id=c.id "
                     "WHERE c.country='US' AND o.status='paid') THEN 1 ELSE 0 END;"),
     "error_desc": "Tested whether ANY US customer has a paid order (existential) instead of ALL (universal)."},
    {"name": "avg_di_orders_null", "fixture": "ord", "ordered": False,
     "question": "What is the average amount of customer 4's orders, to 2 decimals (NULL if none countable)?",
     "query": "SELECT ROUND(AVG(amount),2) FROM orders WHERE customer_id=4;",
     "wrong_query": "SELECT ROUND(COALESCE(AVG(amount),0),2) FROM orders WHERE customer_id=4;",
     "error_desc": "Treated the missing amount as 0, yielding 0 instead of NULL (no countable values)."},
    # ---- prod fixture (category grouping, ordering ties avoided, never-sold, float avg) ----
    {"name": "stationery_product_count", "fixture": "prod", "ordered": False,
     "question": "How many products are in the 'stationery' category?",
     "query": "SELECT COUNT(*) FROM products WHERE category='stationery';",
     "wrong_query": "SELECT COUNT(*) FROM products p JOIN sales s ON s.product_id=p.id WHERE p.category='stationery';",
     "error_desc": "Counted stationery SALES rows rather than the distinct stationery products."},
    {"name": "total_qty_product1", "fixture": "prod", "ordered": False,
     "question": "What is the total quantity sold of product 1?",
     "query": "SELECT SUM(qty) FROM sales WHERE product_id=1;",
     "wrong_query": "SELECT COUNT(*) FROM sales WHERE product_id=1;",
     "error_desc": "Counted the number of sale rows instead of summing the quantity."},
    {"name": "two_most_expensive", "fixture": "prod", "ordered": True,
     "question": "List the names of the two most expensive products, most expensive first.",
     "query": "SELECT pname FROM products ORDER BY price DESC LIMIT 2;",
     "wrong_query": "SELECT pname FROM products ORDER BY price ASC LIMIT 2;",
     "error_desc": "Sorted ascending, returning the two cheapest products."},
    {"name": "never_sold_products", "fixture": "prod", "ordered": False,
     "question": "How many products have never been sold?",
     "query": "SELECT COUNT(*) FROM products p WHERE NOT EXISTS (SELECT 1 FROM sales s WHERE s.product_id=p.id);",
     "wrong_query": "SELECT COUNT(DISTINCT product_id) FROM sales;",
     "error_desc": "Counted the products that HAVE been sold instead of those never sold."},
    {"name": "avg_stationery_price", "fixture": "prod", "ordered": False,
     "question": "What is the average price of 'stationery' products, to 2 decimals?",
     "query": "SELECT ROUND(AVG(price),2) FROM products WHERE category='stationery';",
     "wrong_query": "SELECT ROUND(AVG(price),2) FROM products;",
     "error_desc": "Averaged over all products instead of only the stationery ones."},
    {"name": "categories_with_sales", "fixture": "prod", "ordered": False,
     "question": "How many distinct categories have at least one sale?",
     "query": "SELECT COUNT(DISTINCT p.category) FROM products p JOIN sales s ON s.product_id=p.id;",
     "wrong_query": "SELECT COUNT(DISTINCT s.product_id) FROM sales s;",
     "error_desc": "Counted distinct products with sales rather than distinct categories."},
]

_NULL = "∅NULL"   # sentinel: distinct from any string or number rendering


def _cell(v) -> str:
    if v is None:
        return _NULL
    if isinstance(v, bool):
        return "n:" + str(int(v))
    if isinstance(v, (int, float)):
        f = round(float(v), 6)
        return "n:" + (str(int(f)) if f.is_integer() else repr(f))
    if isinstance(v, bytes):
        return "b:" + v.hex()
    return "s:" + str(v)


def canonicalize(rows, ordered: bool) -> str:
    """Frozen canonical serialization of a result set. `ordered` fixes whether row
    order is significant (set per item, never inferred from the SQL)."""
    serial = [[_cell(c) for c in row] for row in rows]
    if not ordered:
        serial = sorted(serial)
    return json.dumps(serial, separators=(",", ":"))


def _run(conn, sql):
    cur = conn.execute(sql)
    return cur.fetchall()


def _fresh(fixture_key):
    conn = sqlite3.connect(":memory:")
    conn.executescript(FIXTURES[fixture_key])
    return conn


def result_of(item, which: str) -> str:
    """Canonical result string for an item's 'query' (correct) or 'wrong_query'."""
    conn = _fresh(item["fixture"])
    try:
        rows = _run(conn, item["query"] if which == "correct" else item["wrong_query"])
        return canonicalize(rows, item["ordered"])
    finally:
        conn.close()


def readable(item, which: str) -> str:
    """Human-facing rendering of a result (for prompts / stimuli, built later)."""
    conn = _fresh(item["fixture"])
    try:
        rows = _run(conn, item["query"] if which == "correct" else item["wrong_query"])
    finally:
        conn.close()
    def fmt(v):
        return "NULL" if v is None else (str(v) if not isinstance(v, str) else v)
    if len(rows) == 1 and len(rows[0]) == 1:      # scalar
        return fmt(rows[0][0])
    return "[" + ", ".join("(" + ", ".join(fmt(c) for c in r) + ")" for r in rows) + "]"


def self_verify(repeats: int = 2) -> None:
    """Assert the frozen invariants: names unique; every item's correct and wrong
    results are computable, DIFFER (collision check), and are deterministic across
    `repeats` executions."""
    names = [it["name"] for it in ITEMS]
    assert len(names) == len(set(names)), "duplicate item name"
    first = None
    for r in range(repeats):
        snap = []
        for it in ITEMS:
            c = result_of(it, "correct")
            w = result_of(it, "wrong")
            assert c != w, f"{it['name']}: wrong_query result equals correct (collision)"
            snap.append((it["name"], c, w))
        if first is None:
            first = snap
        else:
            assert snap == first, "non-deterministic SQL result across repeats"
    print(f"self_verify OK: {len(ITEMS)} items; correct != wrong for all; deterministic across {repeats} runs")


def gold_signature() -> str:
    sig = [(it["name"], result_of(it, "correct"), result_of(it, "wrong")) for it in ITEMS]
    return hashlib.sha256(json.dumps(sig, separators=(",", ":")).encode()).hexdigest()


def items_sha256() -> str:
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


if __name__ == "__main__":
    import sys
    self_verify()
    print("gold signature:", gold_signature())
    print("items sha256 (LF):", items_sha256())
    print("python:", sys.version.split()[0], "| sqlite:", sqlite3.sqlite_version)
