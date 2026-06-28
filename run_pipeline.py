from __future__ import annotations

import argparse
from pathlib import Path

from src.build_mart import build_orders_mart, build_seller_mart
from src.config import DB_PATH, RAW_DIR, ensure_directories
from src.eda import make_figures
from src.hypothesis_testing import run_hypothesis_tests
from src.load_data import load_csvs_to_sqlite
from src.metrics import save_metrics


def run_pipeline(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> None:
    ensure_directories()

    print("Step 1/6: loading CSV files to SQLite")
    load_csvs_to_sqlite(raw_dir, db_path)

    print("Step 2/6: building orders_mart")
    build_orders_mart(db_path)

    print("Step 3/6: building seller marts")
    build_seller_mart(db_path)

    print("Step 4/6: calculating metrics")
    save_metrics(db_path)

    print("Step 5/6: creating figures")
    make_figures(db_path)

    print("Step 6/6: running hypothesis tests")
    run_hypothesis_tests(db_path)

    print("\nDone.")
    print("Main outputs:")
    print("- data/processed/olist_analysis.sqlite")
    print("- reports/tables/*.csv")
    print("- reports/figures/*.png")
    print("- dashboard/app.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run e-commerce customer experience analytics project.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Directory with Olist CSV files.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="Path to output SQLite database.")
    args = parser.parse_args()
    run_pipeline(args.raw_dir, args.db_path)


if __name__ == "__main__":
    main()
