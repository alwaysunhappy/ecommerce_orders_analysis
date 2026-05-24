from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
SQL_DIR = ROOT_DIR / "sql"
DB_PATH = PROCESSED_DIR / "olist_analysis.sqlite"

EXPECTED_FILES = {
    "raw_customers": "olist_customers_dataset.csv",
    "raw_orders": "olist_orders_dataset.csv",
    "raw_order_items": "olist_order_items_dataset.csv",
    "raw_order_payments": "olist_order_payments_dataset.csv",
    "raw_order_reviews": "olist_order_reviews_dataset.csv",
    "raw_products": "olist_products_dataset.csv",
    "raw_sellers": "olist_sellers_dataset.csv",
    "raw_category_translation": "product_category_name_translation.csv",
}

DATE_COLUMNS = {
    "raw_orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "raw_order_items": ["shipping_limit_date"],
    "raw_order_reviews": ["review_creation_date", "review_answer_timestamp"],
}

REQUIRED_DIRS = [RAW_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR]


def ensure_directories() -> None:
    for path in REQUIRED_DIRS:
        path.mkdir(parents=True, exist_ok=True)
