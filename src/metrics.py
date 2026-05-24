from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DB_PATH, TABLES_DIR, ensure_directories


def read_mart(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM orders_mart", conn)


def _safe_rate(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce")
    if series.dropna().empty:
        return np.nan
    return float(series.mean())


def calculate_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    customer_orders = df.groupby("customer_unique_id")["order_id"].nunique()
    repeat_purchase_rate = float((customer_orders > 1).mean()) if len(customer_orders) else np.nan

    metrics = {
        "total_orders": len(df),
        "delivered_orders": int(df["is_delivered"].sum()),
        "cancelled_orders": int(df["is_cancelled"].sum()),
        "gmv": float(df["total_price"].sum()),
        "aov": float(df["total_price"].mean()),
        "avg_review_score": float(pd.to_numeric(df["review_score"], errors="coerce").mean()),
        "bad_review_rate": _safe_rate(df["is_bad_review"]),
        "delay_rate": _safe_rate(df.loc[df["is_delivered"] == 1, "is_delayed"]),
        "avg_delivery_time_days": float(pd.to_numeric(df["delivery_time_days"], errors="coerce").mean()),
        "avg_delay_days": float(pd.to_numeric(df["delay_days"], errors="coerce").mean()),
        "cancelled_rate": _safe_rate(df["is_cancelled"]),
        "avg_freight_share": float(pd.to_numeric(df["freight_share"], errors="coerce").mean()),
        "repeat_purchase_rate": repeat_purchase_rate,
    }
    return pd.DataFrame([metrics]).round(4)


def aggregate_segment(df: pd.DataFrame, group_col: str, min_orders: int = 30) -> pd.DataFrame:
    result = (
        df.groupby(group_col, dropna=False)
        .agg(
            orders=("order_id", "nunique"),
            gmv=("total_price", "sum"),
            aov=("total_price", "mean"),
            avg_review_score=("review_score", "mean"),
            bad_review_rate=("is_bad_review", "mean"),
            delay_rate=("is_delayed", "mean"),
            avg_delivery_time_days=("delivery_time_days", "mean"),
            avg_freight_share=("freight_share", "mean"),
        )
        .reset_index()
    )
    result = result[result["orders"] >= min_orders]
    return result.sort_values(["bad_review_rate", "orders"], ascending=[False, False]).round(4)


def calculate_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.groupby("order_year_month")
        .agg(
            orders=("order_id", "nunique"),
            gmv=("total_price", "sum"),
            aov=("total_price", "mean"),
            avg_review_score=("review_score", "mean"),
            bad_review_rate=("is_bad_review", "mean"),
            delay_rate=("is_delayed", "mean"),
            avg_delivery_time_days=("delivery_time_days", "mean"),
        )
        .reset_index()
        .sort_values("order_year_month")
    )
    return result.round(4)


def save_metrics(db_path: Path = DB_PATH, output_dir: Path = TABLES_DIR) -> None:
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_mart(db_path)

    outputs = {
        "metrics_summary.csv": calculate_summary_metrics(df),
        "category_metrics.csv": aggregate_segment(df, "product_category_name", min_orders=30),
        "customer_state_metrics.csv": aggregate_segment(df, "customer_state", min_orders=30),
        "seller_state_metrics.csv": aggregate_segment(df, "seller_state", min_orders=30),
        "monthly_metrics.csv": calculate_monthly_metrics(df),
    }

    for filename, table in outputs.items():
        path = output_dir / filename
        table.to_csv(path, index=False)
        print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()
    save_metrics(args.db_path, args.output_dir)


if __name__ == "__main__":
    main()
