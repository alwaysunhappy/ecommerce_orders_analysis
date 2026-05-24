from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from src.config import DB_PATH, SQL_DIR, ensure_directories


def build_orders_mart(db_path: Path = DB_PATH, sql_path: Path | None = None) -> None:
    ensure_directories()
    sql_path = sql_path or SQL_DIR / "create_orders_mart.sql"

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run load_data.py first.")

    sql = sql_path.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(sql)
        mart_rows = conn.execute("SELECT COUNT(*) FROM orders_mart").fetchone()[0]

    print(f"orders_mart created: rows={mart_rows}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--sql-path", type=Path, default=SQL_DIR / "create_orders_mart.sql")
    args = parser.parse_args()
    build_orders_mart(args.db_path, args.sql_path)


if __name__ == "__main__":
    main()
