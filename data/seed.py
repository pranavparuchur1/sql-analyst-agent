"""Build a deterministic synthetic e-commerce database.

`python -m data.seed` writes data/shop.db and prints row counts. A fixed seed
means the eval's gold answers stay valid: same seed -> same rows -> same results.
"""

from __future__ import annotations

import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "shop.db"

SCHEMA = """
CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    city        TEXT NOT NULL,
    state       TEXT NOT NULL,
    signup_date TEXT NOT NULL
);
CREATE TABLE products (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    category TEXT NOT NULL,
    price    REAL NOT NULL
);
CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date  TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('delivered', 'shipped', 'canceled'))
);
CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE payments (
    id       INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL UNIQUE REFERENCES orders(id),
    method   TEXT NOT NULL,
    amount   REAL NOT NULL,
    paid_at  TEXT NOT NULL
);
CREATE TABLE reviews (
    id          INTEGER PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(id),
    product_id  INTEGER NOT NULL REFERENCES products(id),
    score       INTEGER NOT NULL CHECK (score BETWEEN 1 AND 5),
    review_date TEXT NOT NULL
);
-- Indexes on the join keys and the date column most questions filter on.
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_items_order ON order_items(order_id);
CREATE INDEX idx_items_product ON order_items(product_id);
CREATE INDEX idx_reviews_product ON reviews(product_id);
"""

CITIES = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("San Francisco", "CA"), ("Chicago", "IL"),
    ("Houston", "TX"), ("Austin", "TX"), ("Dallas", "TX"), ("Seattle", "WA"),
    ("Miami", "FL"), ("Boston", "MA"), ("Denver", "CO"), ("Atlanta", "GA"),
]

PRODUCTS: dict[str, tuple[tuple[float, float], list[str]]] = {
    "electronics": ((25, 900), ["Wireless Earbuds", "Smartwatch", "4K Monitor", "Mechanical Keyboard",
                                "Bluetooth Speaker", "Laptop Stand", "USB-C Hub", "Webcam HD",
                                "Noise-Cancelling Headphones", "Portable Charger"]),
    "home": ((10, 300), ["Chef Knife", "Cast Iron Skillet", "Coffee Grinder", "Throw Blanket",
                         "Desk Lamp", "Air Purifier", "Ceramic Vase", "Bath Towel Set",
                         "French Press", "Storage Baskets"]),
    "fashion": ((15, 250), ["Denim Jacket", "Running Shoes", "Leather Belt", "Wool Scarf",
                            "Canvas Backpack", "Sunglasses", "Rain Jacket", "Chino Pants",
                            "Linen Shirt", "Ankle Boots"]),
    "books": ((8, 60), ["Data Engineering Handbook", "Mystery Novel", "Cookbook Essentials",
                        "Sci-Fi Anthology", "History of Rome", "Startup Playbook", "Poetry Collection",
                        "Travel Guide USA", "Kids Picture Book", "Python Crash Course"]),
    "sports": ((12, 500), ["Yoga Mat", "Dumbbell Set", "Tennis Racket", "Cycling Helmet",
                           "Camping Tent", "Water Bottle", "Resistance Bands", "Soccer Ball",
                           "Hiking Poles", "Foam Roller"]),
}

FIRST = ["Ava", "Liam", "Mia", "Noah", "Zoe", "Ethan", "Ivy", "Lucas", "Nora", "Omar",
         "Priya", "Raj", "Sofia", "Wei", "Aisha", "Diego", "Hana", "Kofi", "Lena", "Mateo"]
LAST = ["Smith", "Garcia", "Chen", "Patel", "Johnson", "Kim", "Nguyen", "Brown", "Lopez",
        "Singh", "Williams", "Davis", "Martin", "Ali", "Rossi", "Okafor"]

START, END = date(2025, 1, 1), date(2026, 6, 30)


def _rand_date(rng: random.Random, lo: date, hi: date) -> date:
    return lo + timedelta(days=rng.randint(0, (hi - lo).days))


def build(path: Path = DEFAULT_PATH) -> dict[str, int]:
    """(Re)create the database at `path` and return row counts per table."""
    rng = random.Random(42)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("PRAGMA foreign_keys = ON;" + SCHEMA)

    customers = []
    for cid in range(1, 501):
        city, state = rng.choice(CITIES)
        signup = _rand_date(rng, date(2024, 1, 1), date(2026, 3, 31))
        customers.append((cid, f"{rng.choice(FIRST)} {rng.choice(LAST)}", city, state, signup.isoformat()))
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)

    products, weights, quality = [], [], []
    pid = 0
    for category, ((lo, hi), names) in PRODUCTS.items():
        for name in names:
            pid += 1
            products.append((pid, name, category, round(rng.uniform(lo, hi), 2)))
            # Skewed popularity so "top products" has a clear, non-random answer.
            weights.append(rng.paretovariate(1.5))
            quality.append(rng.uniform(-1.2, 0.6))  # per-product review bias
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", products)
    price = {p[0]: p[3] for p in products}
    pids = [p[0] for p in products]

    orders, items, payments, reviews = [], [], [], []
    item_id = 0
    for oid in range(1, 3001):
        cust = rng.choice(customers)
        signup = date.fromisoformat(cust[4])
        od = _rand_date(rng, max(START, signup), END)
        # Recent orders are more likely still in transit.
        if od > END - timedelta(days=14):
            status = rng.choices(["shipped", "delivered", "canceled"], [70, 25, 5])[0]
        else:
            status = rng.choices(["delivered", "shipped", "canceled"], [86, 6, 8])[0]
        orders.append((oid, cust[0], od.isoformat(), status))

        chosen = set()
        total = 0.0
        for _ in range(rng.choices([1, 2, 3, 4], [50, 30, 15, 5])[0]):
            p = rng.choices(pids, weights)[0]
            if p in chosen:
                continue
            chosen.add(p)
            discount = rng.uniform(0.1, 0.3) if rng.random() < 0.15 else 0.0
            unit = round(price[p] * (1 - discount), 2)
            qty = rng.choices([1, 2, 3], [75, 20, 5])[0]
            item_id += 1
            items.append((item_id, oid, p, qty, unit))
            total += qty * unit

        if status != "canceled":
            method = rng.choices(["card", "paypal", "gift_card"], [70, 25, 5])[0]
            payments.append((len(payments) + 1, oid, method, round(total, 2), od.isoformat()))
        if status == "delivered" and rng.random() < 0.5:
            p = rng.choice(sorted(chosen))
            base = 4.2 + quality[p - 1] + rng.gauss(0, 0.9)
            score = max(1, min(5, round(base)))
            rd = od + timedelta(days=rng.randint(3, 20))
            reviews.append((len(reviews) + 1, oid, p, score, rd.isoformat()))

    conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)
    conn.executemany("INSERT INTO payments VALUES (?,?,?,?,?)", payments)
    conn.executemany("INSERT INTO reviews VALUES (?,?,?,?,?)", reviews)
    conn.commit()

    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ["customers", "products", "orders", "order_items", "payments", "reviews"]}
    conn.close()
    return counts


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    for table, n in build(target).items():
        print(f"{table:<12} {n:>6}")
    print(f"wrote {target}")
