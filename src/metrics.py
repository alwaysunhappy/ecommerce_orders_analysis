from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DB_PATH, TABLES_DIR, ensure_directories


Z95 = 1.959963984540054


def read_mart(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM orders_mart", conn)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def read_seller_mart(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, "seller_mart"):
            return pd.DataFrame()
        return pd.read_sql_query("SELECT * FROM seller_mart", conn)


def read_order_seller(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, "order_seller"):
            return pd.DataFrame()
        return pd.read_sql_query("SELECT * FROM order_seller", conn)


def _wilson_ci(success: float, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = success / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (center - half, center + half)


def _safe_rate(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce")
    if series.dropna().empty:
        return np.nan
    return float(series.mean())


def calculate_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    customer_orders = df.groupby("customer_unique_id")["order_id"].nunique()
    repeat_purchase_rate = float((customer_orders > 1).mean()) if len(customer_orders) else np.nan

    reviewed_mask = pd.to_numeric(df["review_count"], errors="coerce").fillna(0) > 0

    metrics = {
        "total_orders": len(df),
        "delivered_orders": int(df["is_delivered"].sum()),
        "cancelled_orders": int(df["is_cancelled"].sum()),
        "reviewed_orders": int(reviewed_mask.sum()),
        "review_coverage": float(reviewed_mask.mean()) if len(df) else np.nan,
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


DELAY_BUCKET_LABELS = ["Нет задержки", "1-3 дня", "4-7 дней", "8-14 дней", "14+ дней"]
DELAY_BUCKET_BINS = [-np.inf, 0, 3, 7, 14, np.inf]
DELAY_STAGE_LABELS = [
    "Поздняя передача продавцом",
    "Долгая транспортировка",
    "Оба этапа",
    "Другое",
]


def delay_bucket_series(df: pd.DataFrame) -> pd.Series:
    delay = pd.to_numeric(df.get("delay_days"), errors="coerce")
    delivered = pd.to_numeric(df.get("is_delivered"), errors="coerce")
    bucket = pd.cut(delay, bins=DELAY_BUCKET_BINS, labels=DELAY_BUCKET_LABELS)
    bucket = bucket.astype("object")
    bucket[(delivered != 1) | delay.isna()] = np.nan
    return pd.Categorical(bucket, categories=DELAY_BUCKET_LABELS, ordered=True)


def delay_bucket_metrics(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["delay_bucket"] = delay_bucket_series(data)
    data = data[data["delay_bucket"].notna()].copy()
    if data.empty:
        return pd.DataFrame()
    data["is_bad_review"] = pd.to_numeric(data["is_bad_review"], errors="coerce")
    data["review_score"] = pd.to_numeric(data["review_score"], errors="coerce")

    rows: list[dict[str, object]] = []
    for bucket in DELAY_BUCKET_LABELS:
        group = data[data["delay_bucket"] == bucket]
        reviewed = group[group["is_bad_review"].notna()]
        n_reviewed = len(reviewed)
        bad = float(reviewed["is_bad_review"].sum())
        rate = bad / n_reviewed if n_reviewed else np.nan
        ci_low, ci_high = _wilson_ci(bad, n_reviewed)
        rows.append({
            "delay_bucket": bucket,
            "orders": int(group["order_id"].nunique()),
            "reviewed_orders": int(n_reviewed),
            "bad_review_rate": rate,
            "bad_review_rate_ci_low": ci_low,
            "bad_review_rate_ci_high": ci_high,
            "avg_review_score": float(reviewed["review_score"].mean()) if n_reviewed else np.nan,
        })
    return pd.DataFrame(rows).round(4)


def delay_stage_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["is_delayed"] = pd.to_numeric(data["is_delayed"], errors="coerce")
    seller_over = pd.to_numeric(data.get("seller_overrun_days"), errors="coerce")
    transit_over = pd.to_numeric(data.get("transit_overrun_days"), errors="coerce")
    data = data[(data["is_delayed"] == 1) & seller_over.notna() & transit_over.notna()].copy()
    if data.empty:
        return pd.DataFrame()
    seller_over = pd.to_numeric(data["seller_overrun_days"], errors="coerce")
    transit_over = pd.to_numeric(data["transit_overrun_days"], errors="coerce")
    data["delay_stage"] = np.select(
        [
            (seller_over > 0) & (transit_over > 0),
            (seller_over > 0) & (transit_over <= 0),
            (transit_over > 0) & (seller_over <= 0),
        ],
        ["Оба этапа", "Поздняя передача продавцом", "Долгая транспортировка"],
        default="Другое",
    )
    data["is_bad_review"] = pd.to_numeric(data["is_bad_review"], errors="coerce")
    data["review_score"] = pd.to_numeric(data["review_score"], errors="coerce")
    data["delay_days"] = pd.to_numeric(data["delay_days"], errors="coerce")

    total = len(data)
    rows: list[dict[str, object]] = []
    for stage in DELAY_STAGE_LABELS:
        group = data[data["delay_stage"] == stage]
        if group.empty:
            continue
        reviewed = group[group["is_bad_review"].notna()]
        rows.append({
            "delay_stage": stage,
            "delayed_orders": int(len(group)),
            "share_of_delayed": round(len(group) / total, 4) if total else np.nan,
            "avg_delay_days": round(float(group["delay_days"].mean()), 2),
            "bad_review_rate": round(float(reviewed["is_bad_review"].mean()), 4) if len(reviewed) else np.nan,
            "avg_review_score": round(float(reviewed["review_score"].mean()), 4) if len(reviewed) else np.nan,
        })
    return pd.DataFrame(rows)


def aggregate_sellers(seller_df: pd.DataFrame, min_orders: int = 30) -> pd.DataFrame:
    if seller_df.empty:
        return seller_df
    columns = [
        "seller_id",
        "seller_state",
        "top_category",
        "orders",
        "items",
        "single_seller_orders",
        "distinct_categories",
        "gmv",
        "delay_rate",
        "late_handover_rate",
        "bad_review_rate",
        "avg_review_score",
        "review_coverage",
        "avg_delivery_time_days",
        "avg_handover_time_days",
        "avg_transit_time_days",
    ]
    existing = [column for column in columns if column in seller_df.columns]
    result = seller_df[existing].copy()
    result = result[result["orders"] >= min_orders]
    return result.sort_values(["bad_review_rate", "orders"], ascending=[False, False]).round(4)


def seller_within_category_comparison(
    order_seller_df: pd.DataFrame,
    min_seller_orders: int = 30,
    min_sellers: int = 3,
    single_seller_only: bool = True,
) -> pd.DataFrame:
    if order_seller_df.empty:
        return pd.DataFrame()

    df = order_seller_df.copy()
    if single_seller_only and "is_single_seller_order" in df.columns:
        df = df[pd.to_numeric(df["is_single_seller_order"], errors="coerce") == 1]

    for column in ["is_bad_review", "is_delayed", "is_late_handover", "review_score",
                   "avg_handover_time_days", "handover_time_days", "transit_time_days"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    reviewed = df[df["is_bad_review"].notna()]
    if reviewed.empty:
        return pd.DataFrame()

    category_rate = reviewed.groupby("product_category_name")["is_bad_review"].mean()

    per_seller = (
        reviewed.groupby(["product_category_name", "seller_id"])
        .agg(
            orders=("order_id", "nunique"),
            bad_reviews=("is_bad_review", "sum"),
            bad_review_rate=("is_bad_review", "mean"),
            avg_review_score=("review_score", "mean"),
            delay_rate=("is_delayed", "mean"),
            late_handover_rate=("is_late_handover", "mean"),
            avg_handover_time_days=("handover_time_days", "mean"),
            avg_transit_time_days=("transit_time_days", "mean"),
        )
        .reset_index()
    )
    per_seller = per_seller[per_seller["orders"] >= min_seller_orders]
    if per_seller.empty:
        return pd.DataFrame()

    seller_counts = per_seller.groupby("product_category_name")["seller_id"].transform("size")
    per_seller = per_seller[seller_counts >= min_sellers]
    if per_seller.empty:
        return pd.DataFrame()

    per_seller["category_bad_review_rate"] = per_seller["product_category_name"].map(category_rate)
    per_seller["bad_review_rate_diff"] = per_seller["bad_review_rate"] - per_seller["category_bad_review_rate"]
    per_seller["n_sellers_in_category"] = per_seller.groupby("product_category_name")["seller_id"].transform("size")

    ci = per_seller.apply(
        lambda row: _wilson_ci(float(row["bad_reviews"]), int(row["orders"])),
        axis=1,
        result_type="expand",
    )
    per_seller["bad_review_rate_ci_low"] = ci[0]
    per_seller["bad_review_rate_ci_high"] = ci[1]

    ordered_columns = [
        "product_category_name",
        "seller_id",
        "n_sellers_in_category",
        "orders",
        "bad_review_rate",
        "bad_review_rate_ci_low",
        "bad_review_rate_ci_high",
        "category_bad_review_rate",
        "bad_review_rate_diff",
        "delay_rate",
        "late_handover_rate",
        "avg_review_score",
        "avg_handover_time_days",
        "avg_transit_time_days",
    ]
    per_seller = per_seller[ordered_columns]
    return per_seller.sort_values(
        ["product_category_name", "bad_review_rate_diff"], ascending=[True, False]
    ).round(4)


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

    delay_buckets = delay_bucket_metrics(df)
    if not delay_buckets.empty:
        outputs["delay_bucket_metrics.csv"] = delay_buckets

    delay_stages = delay_stage_breakdown(df)
    if not delay_stages.empty:
        outputs["delay_stage_breakdown.csv"] = delay_stages

    seller_df = read_seller_mart(db_path)
    seller_metrics = aggregate_sellers(seller_df, min_orders=30)
    if not seller_metrics.empty:
        outputs["seller_metrics.csv"] = seller_metrics

    order_seller_df = read_order_seller(db_path)
    seller_comparison = seller_within_category_comparison(order_seller_df)
    if not seller_comparison.empty:
        outputs["seller_category_comparison.csv"] = seller_comparison

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
