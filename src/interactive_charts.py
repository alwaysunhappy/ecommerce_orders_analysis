from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "olist_chart", "scale": 2},
}

TEMPLATE = "plotly_white"
COLOR_BAD = "#D64545"
COLOR_GOOD = "#2E7D59"
COLOR_NEUTRAL = "#4C78A8"
COLOR_ACCENT = "#F2A541"
COLOR_MUTED = "#8A94A6"


def prepare_orders_mart(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for column in date_columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")

    for column in [
        "total_price",
        "total_freight",
        "payment_value",
        "review_score",
        "delivery_time_days",
        "delay_days",
        "freight_share",
    ]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    for column in ["is_bad_review", "is_delayed", "is_delivered", "is_cancelled"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)

    result["product_category_name"] = result["product_category_name"].fillna("unknown")
    result["customer_state"] = result["customer_state"].fillna("unknown")
    result["seller_state"] = result["seller_state"].fillna("unknown")
    result["main_payment_type"] = result["main_payment_type"].fillna("unknown")

    if "order_purchase_timestamp" in result.columns:
        result["order_month"] = result["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()

    result["bad_review_label"] = np.where(result["is_bad_review"] == 1, "Плохой отзыв", "Не плохой отзыв")
    result["delay_group"] = np.where(result["is_delayed"] == 1, "С задержкой", "Без задержки")
    result["delivered_group"] = np.where(result["is_delivered"] == 1, "Доставлен", "Не доставлен")

    return result


def _format_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.1f}%"


def _safe_mean(series: pd.Series) -> float:
    value = series.dropna().mean()
    return float(value) if pd.notna(value) else 0.0


def _clean_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def add_title(fig: go.Figure, title: str, subtitle: str | None = None) -> go.Figure:
    full_title = title if not subtitle else f"{title}<br><sup>{subtitle}</sup>"
    fig.update_layout(
        template=TEMPLATE,
        title={"text": full_title, "x": 0.01, "xanchor": "left"},
        hovermode="closest",
        margin={"l": 30, "r": 30, "t": 90 if subtitle else 70, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        font={"family": "Arial, sans-serif", "size": 13},
    )
    return fig


def build_kpi_summary(df: pd.DataFrame) -> dict[str, float]:
    delivered = df[df["is_delivered"] == 1]
    repeat_purchase_rate = 0.0
    if "customer_unique_id" in df.columns and len(df) > 0:
        orders_per_customer = df.groupby("customer_unique_id")["order_id"].nunique()
        repeat_purchase_rate = float((orders_per_customer > 1).mean()) if len(orders_per_customer) else 0.0

    return {
        "orders": float(df["order_id"].nunique()),
        "gmv": float(df["total_price"].sum(skipna=True)),
        "aov": float(df["total_price"].mean(skipna=True)) if len(df) else 0.0,
        "avg_review_score": float(df["review_score"].mean(skipna=True)) if len(df) else 0.0,
        "bad_review_rate": float(df["is_bad_review"].mean(skipna=True)) if len(df) else 0.0,
        "delay_rate": float(delivered["is_delayed"].mean(skipna=True)) if len(delivered) else 0.0,
        "avg_delivery_time_days": float(delivered["delivery_time_days"].mean(skipna=True)) if len(delivered) else 0.0,
        "cancel_rate": float(df["is_cancelled"].mean(skipna=True)) if len(df) else 0.0,
        "repeat_purchase_rate": repeat_purchase_rate,
    }


def monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if "order_month" not in df.columns:
        return pd.DataFrame()
    result = (
        df.dropna(subset=["order_month"])
        .groupby("order_month", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            gmv=("total_price", "sum"),
            aov=("total_price", "mean"),
            avg_review_score=("review_score", "mean"),
            bad_review_rate=("is_bad_review", "mean"),
            delay_rate=("is_delayed", "mean"),
        )
        .sort_values("order_month")
    )
    if len(result) >= 3:
        result["orders_ma3"] = result["orders"].rolling(3, min_periods=1).mean()
        result["gmv_ma3"] = result["gmv"].rolling(3, min_periods=1).mean()
    else:
        result["orders_ma3"] = result["orders"]
        result["gmv_ma3"] = result["gmv"]
    return _clean_for_plot(result)


def category_metrics(df: pd.DataFrame, min_orders: int = 50) -> pd.DataFrame:
    base_bad_rate = _safe_mean(df["is_bad_review"])
    result = (
        df.groupby("product_category_name", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            gmv=("total_price", "sum"),
            aov=("total_price", "mean"),
            avg_review_score=("review_score", "mean"),
            bad_review_rate=("is_bad_review", "mean"),
            delay_rate=("is_delayed", "mean"),
            avg_delivery_time_days=("delivery_time_days", "mean"),
        )
    )
    result = result[result["orders"] >= min_orders].copy()
    result["bad_reviews"] = result["orders"] * result["bad_review_rate"]
    result["expected_bad_reviews"] = result["orders"] * base_bad_rate
    result["excess_bad_reviews"] = result["bad_reviews"] - result["expected_bad_reviews"]
    result["business_priority_score"] = result["excess_bad_reviews"].clip(lower=0) * np.log1p(result["gmv"])
    return _clean_for_plot(result)


def state_metrics(df: pd.DataFrame, min_orders: int = 50) -> pd.DataFrame:
    delivered = df[df["is_delivered"] == 1].copy()
    result = (
        delivered.groupby("customer_state", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            gmv=("total_price", "sum"),
            avg_review_score=("review_score", "mean"),
            bad_review_rate=("is_bad_review", "mean"),
            delay_rate=("is_delayed", "mean"),
            avg_delivery_time_days=("delivery_time_days", "mean"),
            avg_delay_days=("delay_days", "mean"),
        )
    )
    return _clean_for_plot(result[result["orders"] >= min_orders])


def payment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.groupby("main_payment_type", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            gmv=("total_price", "sum"),
            aov=("total_price", "mean"),
            avg_review_score=("review_score", "mean"),
            bad_review_rate=("is_bad_review", "mean"),
        )
        .sort_values("orders", ascending=False)
    )
    return _clean_for_plot(result)


def freight_bins(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    bins = [-0.001, 0.05, 0.10, 0.20, 0.35, 0.50, np.inf]
    labels = ["0-5%", "5-10%", "10-20%", "20-35%", "35-50%", "50%+"]
    data["freight_share_bin"] = pd.cut(data["freight_share"], bins=bins, labels=labels)
    result = (
        data.dropna(subset=["freight_share_bin"])
        .groupby("freight_share_bin", observed=False, as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            avg_review_score=("review_score", "mean"),
            bad_review_rate=("is_bad_review", "mean"),
            delay_rate=("is_delayed", "mean"),
            aov=("total_price", "mean"),
        )
    )
    result["freight_share_bin"] = result["freight_share_bin"].astype(str)
    return _clean_for_plot(result)


def build_monthly_overview_figure(df: pd.DataFrame) -> go.Figure:
    data = monthly_metrics(df)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=data["order_month"],
            y=data["orders"],
            name="Заказы",
            marker_color=COLOR_NEUTRAL,
            opacity=0.72,
            hovertemplate="Месяц: %{x|%Y-%m}<br>Заказы: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data["order_month"],
            y=data["orders_ma3"],
            name="Заказы, MA 3 мес.",
            mode="lines+markers",
            line={"color": COLOR_ACCENT, "width": 3},
            hovertemplate="Месяц: %{x|%Y-%m}<br>MA 3 мес.: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data["order_month"],
            y=data["bad_review_rate"],
            name="Доля плохих отзывов",
            mode="lines+markers",
            line={"color": COLOR_BAD, "width": 2},
            hovertemplate="Месяц: %{x|%Y-%m}<br>Плохие отзывы: %{y:.1%}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Количество заказов", secondary_y=False)
    fig.update_yaxes(title_text="Доля плохих отзывов", tickformat=".0%", secondary_y=True)
    fig.update_xaxes(title_text="Месяц")
    return add_title(
        fig,
        "Динамика заказов и качества клиентского опыта",
    )


def build_gmv_monthly_figure(df: pd.DataFrame) -> go.Figure:
    data = monthly_metrics(df)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=data["order_month"],
            y=data["gmv"],
            name="GMV",
            marker_color=COLOR_GOOD,
            opacity=0.72,
            hovertemplate="Месяц: %{x|%Y-%m}<br>GMV: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data["order_month"],
            y=data["aov"],
            name="Средний чек",
            mode="lines+markers",
            line={"color": COLOR_ACCENT, "width": 3},
            hovertemplate="Месяц: %{x|%Y-%m}<br>AOV: %{y:,.1f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="GMV", secondary_y=False)
    fig.update_yaxes(title_text="AOV", secondary_y=True)
    fig.update_xaxes(title_text="Месяц")
    return add_title(fig, "Динамика GMV и среднего чека")


def build_review_distribution_figure(df: pd.DataFrame) -> go.Figure:
    data = (
        df.dropna(subset=["review_score"])["review_score"]
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("review_score")
        .reset_index(name="orders")
    )
    total = data["orders"].sum()
    data["share"] = data["orders"] / total if total else 0
    data["label"] = data["share"].map(lambda x: f"{x:.1%}")

    fig = px.bar(
        data,
        x="review_score",
        y="orders",
        text="label",
        color="review_score",
        color_continuous_scale=[[0, COLOR_BAD], [0.5, COLOR_ACCENT], [1, COLOR_GOOD]],
        hover_data={"orders": ":,.0f", "share": ":.1%", "review_score": True, "label": False},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title_text="Оценка заказа", dtick=1)
    fig.update_yaxes(title_text="Количество заказов")
    return add_title(fig, "Распределение оценок заказов")


def build_delay_review_figure(df: pd.DataFrame) -> go.Figure:
    data = (
        df[df["is_delivered"] == 1]
        .groupby("delay_group", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            avg_review_score=("review_score", "mean"),
            bad_review_rate=("is_bad_review", "mean"),
            avg_delivery_time_days=("delivery_time_days", "mean"),
        )
    )
    order = ["Без задержки", "С задержкой"]
    data["delay_group"] = pd.Categorical(data["delay_group"], categories=order, ordered=True)
    data = data.sort_values("delay_group")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=data["delay_group"].astype(str),
            y=data["avg_review_score"],
            name="Средняя оценка",
            marker_color=[COLOR_GOOD if x == "Без задержки" else COLOR_BAD for x in data["delay_group"].astype(str)],
            text=data["avg_review_score"].map(lambda x: f"{x:.2f}"),
            textposition="outside",
            hovertemplate="Группа: %{x}<br>Средняя оценка: %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data["delay_group"].astype(str),
            y=data["bad_review_rate"],
            name="Доля плохих отзывов",
            mode="lines+markers+text",
            text=data["bad_review_rate"].map(lambda x: f"{x:.1%}"),
            textposition="top center",
            line={"color": COLOR_ACCENT, "width": 3},
            marker={"size": 10},
            hovertemplate="Группа: %{x}<br>Плохие отзывы: %{y:.1%}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Средняя оценка", range=[0, 5.4], secondary_y=False)
    fig.update_yaxes(title_text="Доля плохих отзывов", tickformat=".0%", secondary_y=True)
    fig.update_xaxes(title_text="Статус доставки")
    return add_title(
        fig,
        "Связь задержки доставки с оценкой заказа",
    )


def build_category_priority_figure(df: pd.DataFrame, min_orders: int = 50) -> go.Figure:
    data = category_metrics(df, min_orders=min_orders)
    if data.empty:
        return add_title(go.Figure(), "Матрица приоритизации категорий", "Недостаточно данных для выбранного порога заказов.")

    avg_bad_rate = _safe_mean(df["is_bad_review"])
    median_orders = float(data["orders"].median()) if len(data) else 0.0

    fig = px.scatter(
        data,
        x="orders",
        y="bad_review_rate",
        size="gmv",
        color="avg_review_score",
        color_continuous_scale=[[0, COLOR_BAD], [0.5, COLOR_ACCENT], [1, COLOR_GOOD]],
        hover_name="product_category_name",
        hover_data={
            "orders": ":,.0f",
            "gmv": ":,.0f",
            "bad_review_rate": ":.1%",
            "avg_review_score": ":.2f",
            "delay_rate": ":.1%",
            "excess_bad_reviews": ":,.1f",
            "business_priority_score": ":,.0f",
        },
        size_max=48,
        log_x=True,
    )
    top_labels = data.sort_values("business_priority_score", ascending=False).head(8)
    for _, row in top_labels.iterrows():
        fig.add_annotation(
            x=row["orders"],
            y=row["bad_review_rate"],
            text=str(row["product_category_name"]),
            showarrow=True,
            arrowhead=2,
            ax=24,
            ay=-24,
            font={"size": 10},
        )
    fig.add_hline(
        y=avg_bad_rate,
        line_dash="dash",
        line_color=COLOR_BAD,
        annotation_text=f"Средняя доля плохих отзывов: {avg_bad_rate:.1%}",
        annotation_position="top left",
    )
    if median_orders > 0:
        fig.add_vline(
            x=median_orders,
            line_dash="dot",
            line_color=COLOR_MUTED,
            annotation_text="Медианный объём",
            annotation_position="top right",
        )
    fig.update_xaxes(title_text="Количество заказов в категории, лог. шкала")
    fig.update_yaxes(title_text="Доля плохих отзывов", tickformat=".0%")
    fig.update_layout(coloraxis_colorbar={"title": "Средняя оценка"})
    return add_title(
        fig,
        "Матрица приоритизации категорий",
    )


def build_top_category_risk_figure(df: pd.DataFrame, min_orders: int = 50, top_n: int = 15) -> go.Figure:
    data = category_metrics(df, min_orders=min_orders)
    data = data.sort_values("business_priority_score", ascending=False).head(top_n).sort_values("business_priority_score")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=data["business_priority_score"],
            y=data["product_category_name"],
            orientation="h",
            marker_color=np.where(data["bad_review_rate"] > _safe_mean(df["is_bad_review"]), COLOR_BAD, COLOR_NEUTRAL),
            customdata=np.stack(
                [
                    data["orders"],
                    data["gmv"],
                    data["bad_review_rate"],
                    data["avg_review_score"],
                    data["excess_bad_reviews"],
                ],
                axis=-1,
            ) if len(data) else None,
            hovertemplate=(
                "Категория: %{y}<br>Priority score: %{x:,.0f}"
                "<br>Заказы: %{customdata[0]:,.0f}"
                "<br>GMV: %{customdata[1]:,.0f}"
                "<br>Плохие отзывы: %{customdata[2]:.1%}"
                "<br>Средняя оценка: %{customdata[3]:.2f}"
                "<br>Excess bad reviews: %{customdata[4]:,.1f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title_text="Business priority score")
    fig.update_yaxes(title_text="Категория")
    return add_title(
        fig,
        "Категории с максимальным приоритетом для улучшения",
    )


def build_state_experience_figure(df: pd.DataFrame, min_orders: int = 50) -> go.Figure:
    data = state_metrics(df, min_orders=min_orders)
    if data.empty:
        return add_title(go.Figure(), "Матрица регионов доставки", "Недостаточно данных для выбранного порога заказов.")
    avg_delay_rate = _safe_mean(df[df["is_delivered"] == 1]["is_delayed"])
    avg_review = _safe_mean(df["review_score"])

    fig = px.scatter(
        data,
        x="delay_rate",
        y="avg_review_score",
        size="orders",
        color="bad_review_rate",
        color_continuous_scale=[[0, COLOR_GOOD], [0.5, COLOR_ACCENT], [1, COLOR_BAD]],
        hover_name="customer_state",
        text="customer_state",
        hover_data={
            "orders": ":,.0f",
            "gmv": ":,.0f",
            "delay_rate": ":.1%",
            "bad_review_rate": ":.1%",
            "avg_delivery_time_days": ":.1f",
            "avg_review_score": ":.2f",
        },
        size_max=52,
    )
    fig.update_traces(textposition="top center")
    fig.add_vline(
        x=avg_delay_rate,
        line_dash="dash",
        line_color=COLOR_BAD,
        annotation_text=f"Средняя задержка: {avg_delay_rate:.1%}",
        annotation_position="top right",
    )
    fig.add_hline(
        y=avg_review,
        line_dash="dash",
        line_color=COLOR_MUTED,
        annotation_text=f"Средняя оценка: {avg_review:.2f}",
        annotation_position="bottom left",
    )
    fig.update_xaxes(title_text="Доля задержанных доставок", tickformat=".0%")
    fig.update_yaxes(title_text="Средняя оценка заказа", range=[max(0, data["avg_review_score"].min() - 0.2), 5.05])
    fig.update_layout(coloraxis_colorbar={"title": "Плохие отзывы"})
    return add_title(
        fig,
        "Матрица регионов доставки",
    )


def build_freight_share_figure(df: pd.DataFrame) -> go.Figure:
    data = freight_bins(df)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=data["freight_share_bin"],
            y=data["orders"],
            name="Заказы",
            marker_color=COLOR_NEUTRAL,
            opacity=0.68,
            hovertemplate="Доля доставки: %{x}<br>Заказы: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data["freight_share_bin"],
            y=data["bad_review_rate"],
            name="Плохие отзывы",
            mode="lines+markers+text",
            text=data["bad_review_rate"].map(lambda x: f"{x:.1%}"),
            textposition="top center",
            line={"color": COLOR_BAD, "width": 3},
            hovertemplate="Доля доставки: %{x}<br>Плохие отзывы: %{y:.1%}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=data["freight_share_bin"],
            y=data["avg_review_score"],
            name="Средняя оценка",
            mode="lines+markers",
            line={"color": COLOR_GOOD, "width": 3},
            hovertemplate="Доля доставки: %{x}<br>Средняя оценка: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Количество заказов", secondary_y=False)
    fig.update_yaxes(title_text="Доля плохих отзывов / средняя оценка", secondary_y=True)
    fig.update_xaxes(title_text="Стоимость доставки / стоимость товара")
    return add_title(
        fig,
        "Стоимость доставки относительно цены товара и клиентский опыт",
    )


def build_payment_type_figure(df: pd.DataFrame) -> go.Figure:
    data = payment_metrics(df)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=data["main_payment_type"],
            y=data["orders"],
            name="Заказы",
            marker_color=COLOR_NEUTRAL,
            hovertemplate="Тип оплаты: %{x}<br>Заказы: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data["main_payment_type"],
            y=data["bad_review_rate"],
            name="Плохие отзывы",
            mode="lines+markers+text",
            text=data["bad_review_rate"].map(lambda x: f"{x:.1%}"),
            textposition="top center",
            line={"color": COLOR_BAD, "width": 3},
            hovertemplate="Тип оплаты: %{x}<br>Плохие отзывы: %{y:.1%}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Количество заказов", secondary_y=False)
    fig.update_yaxes(title_text="Доля плохих отзывов", tickformat=".0%", secondary_y=True)
    fig.update_xaxes(title_text="Тип оплаты")
    return add_title(fig, "Метрики по способу оплаты")


def build_seller_customer_heatmap(df: pd.DataFrame, min_orders: int = 30) -> go.Figure:
    data = (
        df[df["is_delivered"] == 1]
        .groupby(["seller_state", "customer_state"], as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            delay_rate=("is_delayed", "mean"),
            avg_delivery_time_days=("delivery_time_days", "mean"),
            bad_review_rate=("is_bad_review", "mean"),
        )
    )
    data = data[data["orders"] >= min_orders].copy()
    if data.empty:
        return add_title(go.Figure(), "Тепловая карта логистических направлений", "Недостаточно данных для выбранного порога заказов.")

    states_order = sorted(set(data["seller_state"]).union(data["customer_state"]))
    matrix = data.pivot(index="seller_state", columns="customer_state", values="delay_rate").reindex(index=states_order, columns=states_order)
    orders_matrix = data.pivot(index="seller_state", columns="customer_state", values="orders").reindex(index=states_order, columns=states_order)
    delivery_matrix = data.pivot(index="seller_state", columns="customer_state", values="avg_delivery_time_days").reindex(index=states_order, columns=states_order)

    customdata = np.dstack([
        orders_matrix.fillna(0).to_numpy(),
        delivery_matrix.fillna(0).to_numpy(),
    ])

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=[[0, COLOR_GOOD], [0.5, COLOR_ACCENT], [1, COLOR_BAD]],
            colorbar={"title": "Доля задержек", "tickformat": ".0%"},
            customdata=customdata,
            hovertemplate=(
                "Регион продавца: %{y}<br>Регион покупателя: %{x}"
                "<br>Доля задержек: %{z:.1%}"
                "<br>Заказы: %{customdata[0]:,.0f}"
                "<br>Среднее время доставки: %{customdata[1]:.1f} дней"
                "<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title_text="Регион покупателя")
    fig.update_yaxes(title_text="Регион продавца")
    return add_title(
        fig,
        "Тепловая карта логистических направлений",
    )


def build_all_figures(df: pd.DataFrame, min_orders: int = 50, top_n: int = 15) -> dict[str, go.Figure]:
    return {
        "01_monthly_overview": build_monthly_overview_figure(df),
        "02_gmv_monthly": build_gmv_monthly_figure(df),
        "03_review_distribution": build_review_distribution_figure(df),
        "04_delay_vs_review": build_delay_review_figure(df),
        "05_category_priority_matrix": build_category_priority_figure(df, min_orders=min_orders),
        "06_top_category_risk": build_top_category_risk_figure(df, min_orders=min_orders, top_n=top_n),
        "07_state_experience_matrix": build_state_experience_figure(df, min_orders=min_orders),
        "08_freight_share": build_freight_share_figure(df),
        "09_payment_type": build_payment_type_figure(df),
        "10_seller_customer_heatmap": build_seller_customer_heatmap(df, min_orders=max(20, min_orders // 2)),
    }
