from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from src.interactive_charts import (
    PLOTLY_CONFIG,
    build_category_priority_figure,
    build_delay_attribution_figure,
    build_delay_bucket_figure,
    build_delay_review_figure,
    build_freight_share_figure,
    build_gmv_monthly_figure,
    build_kpi_summary,
    build_monthly_overview_figure,
    build_payment_type_figure,
    build_review_distribution_figure,
    build_seller_customer_heatmap,
    build_seller_quality_scatter,
    build_seller_within_category_figure,
    build_stage_decomposition_figure,
    build_state_experience_figure,
    build_top_category_risk_figure,
    category_metrics,
    prepare_order_seller,
    prepare_orders_mart,
    seller_metrics,
    state_metrics,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "processed" / "olist_analysis.sqlite"


@st.cache_data(show_spinner=False)
def load_orders_mart() -> pd.DataFrame:
    if not DB_PATH.exists():
        st.error("База не найдена")
        st.stop()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM orders_mart", conn)
    return prepare_orders_mart(df)


@st.cache_data(show_spinner=False)
def load_order_seller() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='order_seller'"
        ).fetchone()
        if table is None:
            return pd.DataFrame()
        df = pd.read_sql_query("SELECT * FROM order_seller", conn)
    return prepare_order_seller(df)


def fmt_int(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def fmt_money(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def fmt_float(value: float, ndigits: int = 2) -> str:
    return f"{value:.{ndigits}f}"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def metric_delta(current: float, base: float, kind: str = "number") -> str:
    if pd.isna(current) or pd.isna(base):
        return ""
    diff = current - base
    if kind == "pct":
        return f"{diff * 100:+.1f} п.п. vs overall"
    if kind == "money":
        return f"{diff:+,.0f} vs overall".replace(",", " ")
    return f"{diff:+.2f} vs overall"


def show_metric_cards(filtered: pd.DataFrame, overall: pd.DataFrame) -> None:
    current = build_kpi_summary(filtered)
    base = build_kpi_summary(overall)

    row1 = st.columns(5)
    row1[0].metric("Заказы", fmt_int(current["orders"]))
    row1[1].metric("GMV", fmt_money(current["gmv"]), metric_delta(current["gmv"], base["gmv"], "money"))
    row1[2].metric("AOV", fmt_float(current["aov"], 1), metric_delta(current["aov"], base["aov"]))
    row1[3].metric("Средняя оценка", fmt_float(current["avg_review_score"], 2), metric_delta(current["avg_review_score"], base["avg_review_score"]))
    row1[4].metric("Плохие отзывы", fmt_pct(current["bad_review_rate"]), metric_delta(current["bad_review_rate"], base["bad_review_rate"], "pct"))

    row2 = st.columns(4)
    row2[0].metric("Доля задержек", fmt_pct(current["delay_rate"]), metric_delta(current["delay_rate"], base["delay_rate"], "pct"))
    row2[1].metric("Среднее время доставки", f"{current['avg_delivery_time_days']:.1f} дней", metric_delta(current["avg_delivery_time_days"], base["avg_delivery_time_days"]))
    row2[2].metric("Доля отмен", fmt_pct(current["cancel_rate"]), metric_delta(current["cancel_rate"], base["cancel_rate"], "pct"))
    row2[3].metric("Повторные покупки", fmt_pct(current["repeat_purchase_rate"]), metric_delta(current["repeat_purchase_rate"], base["repeat_purchase_rate"], "pct"))


def make_download_button(df: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
    )


st.set_page_config(
    page_title="E-commerce Customer Experience Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("E-commerce Customer Experience Analytics")
st.caption(
    "Анализ заказов, доставки, отзывов и проблемных сегментов."
)

orders = load_orders_mart()
order_seller = load_order_seller()

with st.sidebar:
    st.header("Фильтры")

    categories = sorted(orders["product_category_name"].dropna().unique().tolist())
    customer_states = sorted(orders["customer_state"].dropna().unique().tolist())
    seller_states = sorted(orders["seller_state"].dropna().unique().tolist())
    payment_types = sorted(orders["main_payment_type"].dropna().unique().tolist())

    selected_categories = st.multiselect("Категории", categories, default=[])
    selected_customer_states = st.multiselect("Регионы покупателей", customer_states, default=[])
    selected_seller_states = st.multiselect("Регионы продавцов", seller_states, default=[])
    selected_payment_types = st.multiselect("Тип оплаты", payment_types, default=[])

    min_date = orders["order_purchase_timestamp"].min().date()
    max_date = orders["order_purchase_timestamp"].max().date()
    date_range = st.date_input("Период", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    st.divider()
    min_orders = st.slider("Минимум заказов в сегменте", min_value=10, max_value=1000, value=50, step=10)
    top_n = st.slider("Top-N сегментов", min_value=5, max_value=30, value=15, step=1)

filtered = orders.copy()
if selected_categories:
    filtered = filtered[filtered["product_category_name"].isin(selected_categories)]
if selected_customer_states:
    filtered = filtered[filtered["customer_state"].isin(selected_customer_states)]
if selected_seller_states:
    filtered = filtered[filtered["seller_state"].isin(selected_seller_states)]
if selected_payment_types:
    filtered = filtered[filtered["main_payment_type"].isin(selected_payment_types)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        (filtered["order_purchase_timestamp"].dt.date >= start_date)
        & (filtered["order_purchase_timestamp"].dt.date <= end_date)
    ]

if filtered.empty:
    st.warning("После применения фильтров не осталось данных.")
    st.stop()

filtered_sellers = order_seller.copy()
if not filtered_sellers.empty:
    if selected_categories:
        filtered_sellers = filtered_sellers[filtered_sellers["product_category_name"].isin(selected_categories)]
    if selected_customer_states:
        filtered_sellers = filtered_sellers[filtered_sellers["customer_state"].isin(selected_customer_states)]
    if selected_seller_states:
        filtered_sellers = filtered_sellers[filtered_sellers["seller_state"].isin(selected_seller_states)]
    if isinstance(date_range, tuple) and len(date_range) == 2 and "order_purchase_timestamp" in filtered_sellers.columns:
        start_date, end_date = date_range
        filtered_sellers = filtered_sellers[
            (filtered_sellers["order_purchase_timestamp"].dt.date >= start_date)
            & (filtered_sellers["order_purchase_timestamp"].dt.date <= end_date)
        ]

st.subheader("Ключевые метрики")
show_metric_cards(filtered, orders)

st.divider()

tab_overview, tab_quality, tab_categories, tab_geo, tab_sellers, tab_money, tab_data = st.tabs(
    [
        "1. Обзор",
        "2. Доставка и отзывы",
        "3. Категории",
        "4. География",
        "5. Продавцы",
        "6. Деньги и оплата",
        "7. Данные",
    ]
)

with tab_overview:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(build_monthly_overview_figure(filtered), use_container_width=True, config=PLOTLY_CONFIG, key='monthly_overview_chart')
    with col2:
        st.plotly_chart(build_gmv_monthly_figure(filtered), use_container_width=True, config=PLOTLY_CONFIG, key='gmv_monthly_chart')

    st.plotly_chart(build_review_distribution_figure(filtered), use_container_width=True, config=PLOTLY_CONFIG, key='review_distribution_chart')

with tab_quality:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(build_delay_review_figure(filtered), use_container_width=True, config=PLOTLY_CONFIG, key='delay_review_chart')
    with col2:
        st.plotly_chart(build_freight_share_figure(filtered), use_container_width=True, config=PLOTLY_CONFIG, key='freight_share_delay_tab_chart')

    st.plotly_chart(build_delay_bucket_figure(filtered), use_container_width=True, config=PLOTLY_CONFIG, key='delay_bucket_chart')

with tab_categories:
    st.plotly_chart(build_category_priority_figure(filtered, min_orders=min_orders), use_container_width=True, config=PLOTLY_CONFIG, key='category_priority_chart')
    st.plotly_chart(build_top_category_risk_figure(filtered, min_orders=min_orders, top_n=top_n), use_container_width=True, config=PLOTLY_CONFIG, key='top_category_risk_chart')

    cat_table = category_metrics(filtered, min_orders=min_orders).sort_values("business_priority_score", ascending=False)
    st.subheader("Таблица категорий")
    st.dataframe(
        cat_table,
        use_container_width=True,
        column_config={
            "bad_review_rate": st.column_config.ProgressColumn("bad_review_rate", format="%.1f", min_value=0, max_value=1),
            "delay_rate": st.column_config.ProgressColumn("delay_rate", format="%.1f", min_value=0, max_value=1),
            "gmv": st.column_config.NumberColumn("gmv", format="%.0f"),
            "business_priority_score": st.column_config.NumberColumn("business_priority_score", format="%.0f"),
        },
    )
    make_download_button(cat_table, "category_metrics_interactive.csv", "Скачать таблицу категорий")

with tab_geo:
    st.plotly_chart(build_state_experience_figure(filtered, min_orders=min_orders), use_container_width=True, config=PLOTLY_CONFIG, key='state_experience_chart')
    st.plotly_chart(build_seller_customer_heatmap(filtered, min_orders=max(20, min_orders // 2)), use_container_width=True, config=PLOTLY_CONFIG, key='seller_customer_heatmap_chart')

    state_table = state_metrics(filtered, min_orders=min_orders).sort_values(["delay_rate", "bad_review_rate"], ascending=False)
    st.subheader("Таблица регионов покупателей")
    st.dataframe(state_table, use_container_width=True)
    make_download_button(state_table, "customer_state_metrics_interactive.csv", "Скачать таблицу регионов")

with tab_sellers:
    if filtered_sellers.empty:
        st.info("Витрина продавцов недоступна. Запустите пайплайн: `make all`.")
    else:
        seller_min_orders = max(10, min_orders // 2)
        st.plotly_chart(
            build_delay_attribution_figure(filtered),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key='delay_attribution_chart',
        )
        st.plotly_chart(
            build_stage_decomposition_figure(filtered_sellers),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key='stage_decomposition_chart',
        )
        st.plotly_chart(
            build_seller_quality_scatter(filtered_sellers, min_orders=seller_min_orders),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key='seller_quality_scatter_chart',
        )
        st.plotly_chart(
            build_seller_within_category_figure(filtered_sellers, min_orders=seller_min_orders),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key='seller_within_category_chart',
        )

        seller_table = seller_metrics(filtered_sellers, min_orders=seller_min_orders).sort_values(
            "bad_review_rate", ascending=False
        )
        st.subheader("Таблица продавцов")
        st.dataframe(
            seller_table,
            use_container_width=True,
            column_config={
                "bad_review_rate": st.column_config.ProgressColumn("bad_review_rate", format="%.2f", min_value=0, max_value=1),
                "delay_rate": st.column_config.ProgressColumn("delay_rate", format="%.2f", min_value=0, max_value=1),
                "late_handover_rate": st.column_config.ProgressColumn("late_handover_rate", format="%.2f", min_value=0, max_value=1),
                "gmv": st.column_config.NumberColumn("gmv", format="%.0f"),
            },
        )
        make_download_button(seller_table, "seller_metrics_interactive.csv", "Скачать таблицу продавцов")

with tab_money:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(build_freight_share_figure(filtered), use_container_width=True, config=PLOTLY_CONFIG, key='freight_share_money_tab_chart')
    with col2:
        st.plotly_chart(build_payment_type_figure(filtered), use_container_width=True, config=PLOTLY_CONFIG, key='payment_type_chart')

with tab_data:
    st.subheader("Витрина orders_mart")
    display_columns = [
        "order_id",
        "order_purchase_timestamp",
        "order_status",
        "product_category_name",
        "customer_state",
        "seller_state",
        "total_price",
        "total_freight",
        "freight_share",
        "main_payment_type",
        "review_score",
        "is_bad_review",
        "is_delayed",
        "delivery_time_days",
        "delay_days",
    ]
    existing_columns = [column for column in display_columns if column in filtered.columns]
    st.dataframe(filtered[existing_columns].head(10_000), use_container_width=True)
    make_download_button(filtered[existing_columns], "orders_mart_filtered.csv", "Скачать данные")