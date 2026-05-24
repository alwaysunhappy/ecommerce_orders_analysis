from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_PATH, EXPECTED_FILES, RAW_DIR, ensure_directories


class MissingDataError(FileNotFoundError):
    pass


def validate_raw_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    missing = [raw_dir / filename for filename in EXPECTED_FILES.values() if not (raw_dir / filename).exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise MissingDataError(
            "Не найдены обязательные CSV-файлы:\n"
            f"{missing_text}\n\n"
            "Загрузите данные командой `make data` или положите CSV-файлы Olist в data/raw/."
        )
    return [raw_dir / filename for filename in EXPECTED_FILES.values()]


def load_csvs_to_sqlite(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> None:
    ensure_directories()
    validate_raw_files(raw_dir)

    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        for table_name, filename in EXPECTED_FILES.items():
            path = raw_dir / filename
            df = pd.read_csv(path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"Loaded {filename:<45} -> {table_name:<25} rows={len(df)}")

    print(f"SQLite database saved to {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    args = parser.parse_args()
    load_csvs_to_sqlite(args.raw_dir, args.db_path)


if __name__ == "__main__":
    main()
