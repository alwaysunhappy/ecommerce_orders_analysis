from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from src.config import DB_PATH, FIGURES_DIR, ensure_directories
from src.metrics import read_mart

BRAND = "#1F4E79"
BRAND_LIGHT = "#D9EAF7"
ACCENT = "#E67E22"
NEGATIVE = "#C0392B"
POSITIVE = "#2E7D32"
NEUTRAL = "#6B7280"
LIGHT_GREY = "#E5E7EB"
DARK = "#111827"
GRID = "#E8EEF5"
BACKGROUND = "#FFFFFF"

plt.rcParams.update(
    {
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "axes.edgecolor": "#D1D5DB",
        "axes.labelcolor": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "text.color": DARK,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "legend.frameon": False,
        "savefig.facecolor": BACKGROUND,
    }
)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=1.4)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def clean_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")


def pct_formatter(x: float, _pos: int | None = None) -> str:
    return f"{x:.0f}%"


def pct1_formatter(x: float, _pos: int | None = None) -> str:
    return f"{x:.1f}%"


def money_formatter(x: float, _pos: int | None = None) -> str:
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f}K"
    return f"{x:.0f}"


def int_formatter(x: float, _pos: int | None = None) -> str:
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f}K"
    return f"{x:.0f}"


def fmt_int(x: float | int | np.integer | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f} млн"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f} тыс"
    return f"{x:.0f}"


def fmt_money(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f} млн"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f} тыс"
    return f"{x:.0f}"


def fmt_pct(x: float | None, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x) * 100:.{digits}f}%"


def wrap_labels(labels: Iterable[object], width: int = 24) -> list[str]:
    result = []
    for label in labels:
        text = "Не указано" if pd.isna(label) else str(label)
        result.append("\n".join(textwrap.wrap(text, width=width, break_long_words=False)))
    return result


def add_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    subtitle_wrapped = None
    if subtitle:
        subtitle_wrapped = "\n".join(textwrap.wrap(subtitle, width=118, break_long_words=False))

    title_pad = 52 if subtitle_wrapped and "\n" in subtitle_wrapped else 34
    ax.set_title(title, loc="left", pad=title_pad)

    if subtitle_wrapped:
        ax.text(
            0,
            1.015,
            subtitle_wrapped,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color=NEUTRAL,
            linespacing=1.35,
        )


def add_bar_labels(
    ax: plt.Axes,
    bars,
    labels: list[str] | None = None,
    orientation: str = "vertical",
    padding: float = 3,
    fontsize: int = 9,
) -> None:
    for i, bar in enumerate(bars):
        label = labels[i] if labels else f"{bar.get_height():.1f}"
        if orientation == "horizontal":
            width = bar.get_width()
            ax.annotate(
                label,
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(padding, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=fontsize,
                color=DARK,
            )
        else:
            height = bar.get_height()
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, padding),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=DARK,
            )


def prepare_mart(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    numeric_cols = [
        "total_price",
        "total_freight",
        "payment_value",
        "review_score",
        "is_bad_review",
        "is_delivered",
        "is_cancelled",
        "delivery_time_days",
        "estimated_delivery_time_days",
        "delay_days",
        "is_delayed",
        "freight_share",
    ]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    for col in ["product_category_name", "customer_state", "seller_state"]:
        if col in result.columns:
            result[col] = result[col].fillna("unknown")

    if "order_year_month" in result.columns:
        result["order_month_dt"] = pd.to_datetime(result["order_year_month"].astype(str) + "-01", errors="coerce")

    result["bad_review_flag"] = np.where(result["review_score"].notna(), result["is_bad_review"], np.nan)
    return result


def plot_review_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    scores = pd.Series(range(1, 6), name="review_score")
    counts = (
        df["review_score"]
        .dropna()
        .astype(int)
        .value_counts()
        .reindex(scores, fill_value=0)
        .sort_index()
    )
    total = counts.sum()
    shares = counts / total * 100 if total else counts.astype(float)
    avg_score = df["review_score"].mean()
    bad_rate = df["bad_review_flag"].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [NEGATIVE if score <= 2 else ACCENT if score == 3 else BRAND for score in counts.index]
    bars = ax.bar(counts.index.astype(str), shares.values, color=colors, width=0.65)

    add_title(
        ax,
        "Распределение оценок заказов",
    )
    ax.set_xlabel("Оценка review_score")
    ax.set_ylabel("Доля заказов")
    ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax.set_ylim(0, max(shares.max() * 1.25, 10))
    clean_axes(ax)

    labels = [f"{share:.1f}%\n{fmt_int(count)}" for share, count in zip(shares.values, counts.values)]
    add_bar_labels(ax, bars, labels=labels, fontsize=9)

    ax.text(
        0.02,
        0.94,
        f"Средняя оценка: {avg_score:.2f}\nДоля плохих отзывов: {fmt_pct(bad_rate)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=LIGHT_GREY),
    )
    savefig(output_dir / "01_review_score_distribution.png")


def _delay_group_table(df: pd.DataFrame) -> pd.DataFrame:
    data = (
        df[df["review_score"].notna()]
        .assign(delivery_group=lambda x: x["is_delayed"].map({0: "Без задержки", 1: "С задержкой"}).fillna("Не доставлено"))
        .query("delivery_group in ['Без задержки', 'С задержкой']")
        .groupby("delivery_group")
        .agg(
            avg_review_score=("review_score", "mean"),
            bad_review_rate=("bad_review_flag", "mean"),
            orders=("order_id", "nunique"),
        )
        .reindex(["Без задержки", "С задержкой"])
    )
    return data


def plot_delayed_vs_review(df: pd.DataFrame, output_dir: Path) -> None:
    data = _delay_group_table(df).dropna(subset=["orders"])
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = [POSITIVE if group == "Без задержки" else NEGATIVE for group in data.index]
    bars = ax.bar(data.index, data["avg_review_score"], color=colors, width=0.55)

    no_delay = data.loc["Без задержки", "avg_review_score"] if "Без задержки" in data.index else np.nan
    delayed = data.loc["С задержкой", "avg_review_score"] if "С задержкой" in data.index else np.nan
    delta = delayed - no_delay if pd.notna(no_delay) and pd.notna(delayed) else np.nan

    add_title(
        ax,
        "Сравнение доставленных заказов с известной оценкой",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Средняя оценка")
    ax.set_ylim(0, 5)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    clean_axes(ax)

    labels = [f"{row.avg_review_score:.2f}\nn={fmt_int(row.orders)}" for row in data.itertuples()]
    add_bar_labels(ax, bars, labels=labels, fontsize=10)

    if pd.notna(delta):
        ax.text(
            0.5,
            0.14,
            f"Разница: {delta:.2f} балла",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=NEGATIVE if delta < 0 else POSITIVE,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=LIGHT_GREY),
        )

    savefig(output_dir / "02_avg_review_by_delay_group.png")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    values = data["bad_review_rate"] * 100
    bars = ax.bar(data.index, values, color=colors, width=0.55)

    no_delay_rate = data.loc["Без задержки", "bad_review_rate"] if "Без задержки" in data.index else np.nan
    delayed_rate = data.loc["С задержкой", "bad_review_rate"] if "С задержкой" in data.index else np.nan
    lift = (delayed_rate / no_delay_rate - 1) if pd.notna(no_delay_rate) and no_delay_rate > 0 and pd.notna(delayed_rate) else np.nan

    add_title(
        ax,
        "Доля негативных отзывов по статусу доставки"
    )
    ax.set_xlabel("")
    ax.set_ylabel("Доля плохих отзывов")
    ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax.set_ylim(0, max(values.max() * 1.35, 5))
    clean_axes(ax)

    labels = [f"{value:.1f}%\nn={fmt_int(order)}" for value, order in zip(values.values, data["orders"].values)]
    add_bar_labels(ax, bars, labels=labels, fontsize=10)

    if pd.notna(lift):
        ax.text(
            0.5,
            0.86,
            f"Lift риска плохого отзыва: {lift * 100:+.1f}%",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=NEGATIVE if lift > 0 else POSITIVE,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=LIGHT_GREY),
        )

    savefig(output_dir / "03_bad_review_rate_by_delay_group.png")


def plot_bad_reviews_by_category(df: pd.DataFrame, output_dir: Path, min_orders: int = 300) -> None:
    global_rate = df["bad_review_flag"].mean()
    data = (
        df[df["review_score"].notna()]
        .groupby("product_category_name", dropna=False)
        .agg(
            orders=("order_id", "nunique"),
            gmv=("total_price", "sum"),
            avg_review_score=("review_score", "mean"),
            bad_review_rate=("bad_review_flag", "mean"),
        )
        .reset_index()
    )
    data["bad_reviews"] = data["orders"] * data["bad_review_rate"]
    data = data[(data["orders"] >= min_orders) & (data["bad_review_rate"] >= global_rate)]
    if data.empty:
        data = (
            df[df["review_score"].notna()]
            .groupby("product_category_name", dropna=False)
            .agg(
                orders=("order_id", "nunique"),
                gmv=("total_price", "sum"),
                bad_review_rate=("bad_review_flag", "mean"),
            )
            .reset_index()
        )
        data["bad_reviews"] = data["orders"] * data["bad_review_rate"]

    data = data.sort_values("bad_reviews", ascending=False).head(15).sort_values("bad_review_rate")

    fig_height = max(6, 0.42 * len(data) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    colors = [NEGATIVE if rate >= global_rate else BRAND for rate in data["bad_review_rate"]]
    bars = ax.barh(wrap_labels(data["product_category_name"], 28), data["bad_review_rate"] * 100, color=colors)

    add_title(
        ax,
        "Приоритетные категории по плохому клиентскому опыту",
    )
    ax.set_xlabel("Доля плохих отзывов")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax.axvline(global_rate * 100, color=NEUTRAL, linestyle="--", linewidth=1.4, label=f"Среднее по всем заказам: {fmt_pct(global_rate)}")
    clean_axes(ax, grid_axis="x")
    ax.legend(loc="lower right")

    labels = [f"{rate * 100:.1f}% · n={fmt_int(order)}" for rate, order in zip(data["bad_review_rate"], data["orders"])]
    add_bar_labels(ax, bars, labels=labels, orientation="horizontal", fontsize=8)

    xmax = max((data["bad_review_rate"] * 100).max() * 1.35, global_rate * 100 * 1.4, 5)
    ax.set_xlim(0, xmax)
    savefig(output_dir / "04_bad_review_rate_by_category.png")


def plot_delivery_by_state(df: pd.DataFrame, output_dir: Path, min_orders: int = 300) -> None:
    delivered = df[df["is_delivered"] == 1].copy()
    global_delay_rate = delivered["is_delayed"].mean()
    global_delivery_time = delivered["delivery_time_days"].mean()

    data = (
        delivered.groupby("customer_state", dropna=False)
        .agg(
            orders=("order_id", "nunique"),
            delay_rate=("is_delayed", "mean"),
            avg_delivery_time_days=("delivery_time_days", "mean"),
            bad_review_rate=("bad_review_flag", "mean"),
        )
        .reset_index()
    )
    data = data[data["orders"] >= min_orders]
    if data.empty:
        data = (
            delivered.groupby("customer_state", dropna=False)
            .agg(
                orders=("order_id", "nunique"),
                delay_rate=("is_delayed", "mean"),
                avg_delivery_time_days=("delivery_time_days", "mean"),
                bad_review_rate=("bad_review_flag", "mean"),
            )
            .reset_index()
        )

    top_delay = data.sort_values("delay_rate", ascending=False).head(15).sort_values("delay_rate")
    fig_height = max(5.5, 0.4 * len(top_delay) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    colors = [NEGATIVE if rate >= global_delay_rate else BRAND for rate in top_delay["delay_rate"]]
    bars = ax.barh(top_delay["customer_state"], top_delay["delay_rate"] * 100, color=colors)

    add_title(
        ax,
        "Регионы с повышенной долей задержек доставки",
    )
    ax.set_xlabel("Доля задержанных доставок")
    ax.set_ylabel("Регион покупателя")
    ax.xaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax.axvline(global_delay_rate * 100, color=NEUTRAL, linestyle="--", linewidth=1.4, label=f"Среднее: {fmt_pct(global_delay_rate)}")
    clean_axes(ax, grid_axis="x")
    ax.legend(loc="lower right")
    labels = [f"{rate * 100:.1f}% · n={fmt_int(order)}" for rate, order in zip(top_delay["delay_rate"], top_delay["orders"])]
    add_bar_labels(ax, bars, labels=labels, orientation="horizontal", fontsize=8)
    ax.set_xlim(0, max((top_delay["delay_rate"] * 100).max() * 1.35, global_delay_rate * 100 * 1.4, 5))
    savefig(output_dir / "05_delay_rate_by_customer_state.png")

    top_time = data.sort_values("avg_delivery_time_days", ascending=False).head(15).sort_values("avg_delivery_time_days")
    fig_height = max(5.5, 0.4 * len(top_time) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    colors = [NEGATIVE if days >= global_delivery_time else BRAND for days in top_time["avg_delivery_time_days"]]
    bars = ax.barh(top_time["customer_state"], top_time["avg_delivery_time_days"], color=colors)

    add_title(
        ax,
        "Регионы с наиболее длительным временем доставки",
    )
    ax.set_xlabel("Среднее время доставки, дней")
    ax.set_ylabel("Регион покупателя")
    ax.axvline(global_delivery_time, color=NEUTRAL, linestyle="--", linewidth=1.4, label=f"Среднее: {global_delivery_time:.1f} дн.")
    clean_axes(ax, grid_axis="x")
    ax.legend(loc="lower right")
    labels = [f"{days:.1f} дн. · n={fmt_int(order)}" for days, order in zip(top_time["avg_delivery_time_days"], top_time["orders"])]
    add_bar_labels(ax, bars, labels=labels, orientation="horizontal", fontsize=8)
    ax.set_xlim(0, max(top_time["avg_delivery_time_days"].max() * 1.25, global_delivery_time * 1.4, 1))
    savefig(output_dir / "06_avg_delivery_time_by_customer_state.png")


def _freight_bin_label(interval: pd.Interval) -> str:
    left = max(interval.left, 0)
    right = interval.right
    return f"{left * 100:.0f}–{right * 100:.0f}%"


def plot_freight_share_vs_review(df: pd.DataFrame, output_dir: Path) -> None:
    data = df[(df["freight_share"].notna()) & (df["review_score"].notna()) & (df["freight_share"] >= 0)].copy()
    if data.empty:
        return

    upper = data["freight_share"].quantile(0.98)
    data = data[data["freight_share"] <= upper].copy()
    data["freight_share_bin"] = pd.qcut(data["freight_share"], q=6, duplicates="drop")
    agg = (
        data.groupby("freight_share_bin", observed=True)
        .agg(
            avg_review_score=("review_score", "mean"),
            bad_review_rate=("bad_review_flag", "mean"),
            orders=("order_id", "nunique"),
        )
        .reset_index()
    )
    agg["bin_label"] = agg["freight_share_bin"].map(_freight_bin_label)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(agg["bin_label"], agg["avg_review_score"], color=BRAND, linewidth=2.6, marker="o", markersize=7)
    ax.fill_between(
        range(len(agg)),
        agg["avg_review_score"].min(),
        agg["avg_review_score"],
        color=BRAND_LIGHT,
        alpha=0.35,
    )

    add_title(
        ax,
        "Средняя оценка заказа в зависимости от доли стоимости доставки",
    )
    ax.set_xlabel("Доля доставки в стоимости товаров")
    ax.set_ylabel("Средняя оценка")
    ax.set_ylim(max(0, agg["avg_review_score"].min() - 0.4), 5)
    clean_axes(ax)

    for i, row in agg.iterrows():
        ax.annotate(
            f"{row.avg_review_score:.2f}\nn={fmt_int(row.orders)}",
            xy=(i, row.avg_review_score),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=DARK,
        )

    savefig(output_dir / "07_avg_review_by_freight_share_bin.png")


def _monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    data = (
        df.dropna(subset=["order_month_dt"])
        .groupby("order_month_dt")
        .agg(
            orders=("order_id", "nunique"),
            gmv=("total_price", "sum"),
            bad_review_rate=("bad_review_flag", "mean"),
            delay_rate=("is_delayed", "mean"),
        )
        .reset_index()
        .sort_values("order_month_dt")
    )
    data["orders_ma3"] = data["orders"].rolling(3, min_periods=1).mean()
    data["gmv_ma3"] = data["gmv"].rolling(3, min_periods=1).mean()
    return data


def plot_monthly_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    data = _monthly_data(df)
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(data["order_month_dt"], data["orders"], color=BRAND_LIGHT, linewidth=1.5, marker="o", markersize=4, label="Заказы")
    ax.plot(data["order_month_dt"], data["orders_ma3"], color=BRAND, linewidth=2.8, label="Скользящее среднее 3 мес.")

    peak = data.loc[data["orders"].idxmax()]
    last = data.iloc[-1]
    ax.scatter([peak["order_month_dt"]], [peak["orders"]], color=ACCENT, s=55, zorder=5)
    ax.annotate(
        f"Пик: {fmt_int(peak['orders'])}",
        xy=(peak["order_month_dt"], peak["orders"]),
        xytext=(10, 18),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2),
        fontsize=9,
        color=DARK,
    )
    ax.annotate(
        f"Последний месяц: {fmt_int(last['orders'])}",
        xy=(last["order_month_dt"], last["orders"]),
        xytext=(-110, -28),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=NEUTRAL, lw=1.2),
        fontsize=9,
        color=DARK,
    )

    add_title(
        ax,
        "Динамика количества заказов по месяцам",
    )
    ax.set_xlabel("Месяц заказа")
    ax.set_ylabel("Количество заказов")
    ax.yaxis.set_major_formatter(FuncFormatter(int_formatter))
    clean_axes(ax)
    ax.legend(loc="upper left")
    fig.autofmt_xdate(rotation=35)
    savefig(output_dir / "08_orders_by_month.png")

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(data["order_month_dt"], data["gmv"], color=BRAND_LIGHT, linewidth=1.5, marker="o", markersize=4, label="GMV")
    ax.plot(data["order_month_dt"], data["gmv_ma3"], color=BRAND, linewidth=2.8, label="Скользящее среднее 3 мес.")

    peak = data.loc[data["gmv"].idxmax()]
    last = data.iloc[-1]
    ax.scatter([peak["order_month_dt"]], [peak["gmv"]], color=ACCENT, s=55, zorder=5)
    ax.annotate(
        f"Пик: {fmt_money(peak['gmv'])}",
        xy=(peak["order_month_dt"], peak["gmv"]),
        xytext=(10, 18),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2),
        fontsize=9,
        color=DARK,
    )
    ax.annotate(
        f"Последний месяц: {fmt_money(last['gmv'])}",
        xy=(last["order_month_dt"], last["gmv"]),
        xytext=(-120, -28),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=NEUTRAL, lw=1.2),
        fontsize=9,
        color=DARK,
    )

    add_title(
        ax,
        "Динамика GMV по месяцам",
    )
    ax.set_xlabel("Месяц заказа")
    ax.set_ylabel("GMV")
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    clean_axes(ax)
    ax.legend(loc="upper left")
    fig.autofmt_xdate(rotation=35)
    savefig(output_dir / "09_gmv_by_month.png")


def plot_category_impact_matrix(df: pd.DataFrame, output_dir: Path, min_orders: int = 300) -> None:
    global_bad_rate = df["bad_review_flag"].mean()
    data = (
        df[df["review_score"].notna()]
        .groupby("product_category_name", dropna=False)
        .agg(
            orders=("order_id", "nunique"),
            gmv=("total_price", "sum"),
            bad_review_rate=("bad_review_flag", "mean"),
        )
        .reset_index()
    )
    data["bad_reviews"] = data["orders"] * data["bad_review_rate"]
    data = data[data["orders"] >= min_orders]
    if data.empty:
        data = (
            df[df["review_score"].notna()]
            .groupby("product_category_name", dropna=False)
            .agg(
                orders=("order_id", "nunique"),
                gmv=("total_price", "sum"),
                bad_review_rate=("bad_review_flag", "mean"),
            )
            .reset_index()
        )
        data["bad_reviews"] = data["orders"] * data["bad_review_rate"]
    if data.empty:
        return

    median_orders = data["orders"].median()
    sizes = 120 + 950 * (data["gmv"] / data["gmv"].max()).clip(0, 1)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = np.where(data["bad_review_rate"] >= global_bad_rate, NEGATIVE, BRAND)
    ax.scatter(
        data["orders"],
        data["bad_review_rate"] * 100,
        s=sizes,
        c=colors,
        alpha=0.72,
        edgecolors="white",
        linewidth=1.0,
    )
    ax.axhline(global_bad_rate * 100, color=NEUTRAL, linestyle="--", linewidth=1.2, label=f"Средняя доля плохих отзывов: {fmt_pct(global_bad_rate)}")
    ax.axvline(median_orders, color=NEUTRAL, linestyle=":", linewidth=1.2, label=f"Медианный объём категории: {fmt_int(median_orders)}")

    label_data = data.sort_values("bad_reviews", ascending=False).head(10)
    for _, row in label_data.iterrows():
        ax.annotate(
            str(row["product_category_name"])[:28],
            xy=(row["orders"], row["bad_review_rate"] * 100),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8,
            color=DARK,
        )

    add_title(
        ax,
        "Матрица приоритизации категорий",
    )
    ax.set_xlabel("Количество заказов в категории")
    ax.set_ylabel("Доля плохих отзывов")
    ax.xaxis.set_major_formatter(FuncFormatter(int_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(pct1_formatter))
    clean_axes(ax)
    ax.legend(loc="upper right")
    savefig(output_dir / "10_category_impact_matrix.png")


def plot_state_experience_matrix(df: pd.DataFrame, output_dir: Path, min_orders: int = 300) -> None:
    delivered = df[(df["is_delivered"] == 1) & (df["review_score"].notna())].copy()
    data = (
        delivered.groupby("customer_state", dropna=False)
        .agg(
            orders=("order_id", "nunique"),
            delay_rate=("is_delayed", "mean"),
            avg_review_score=("review_score", "mean"),
            gmv=("total_price", "sum"),
        )
        .reset_index()
    )
    data = data[data["orders"] >= min_orders]
    if data.empty:
        data = (
            delivered.groupby("customer_state", dropna=False)
            .agg(
                orders=("order_id", "nunique"),
                delay_rate=("is_delayed", "mean"),
                avg_review_score=("review_score", "mean"),
                gmv=("total_price", "sum"),
            )
            .reset_index()
        )
    if data.empty:
        return

    global_delay = delivered["is_delayed"].mean()
    global_review = delivered["review_score"].mean()
    sizes = 130 + 850 * (data["orders"] / data["orders"].max()).clip(0, 1)
    colors = np.where((data["delay_rate"] >= global_delay) & (data["avg_review_score"] <= global_review), NEGATIVE, BRAND)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(
        data["delay_rate"] * 100,
        data["avg_review_score"],
        s=sizes,
        c=colors,
        alpha=0.72,
        edgecolors="white",
        linewidth=1.0,
    )
    ax.axvline(global_delay * 100, color=NEUTRAL, linestyle="--", linewidth=1.2, label=f"Средняя доля задержек: {fmt_pct(global_delay)}")
    ax.axhline(global_review, color=NEUTRAL, linestyle=":", linewidth=1.2, label=f"Средняя оценка: {global_review:.2f}")

    label_data = data.sort_values("orders", ascending=False).head(12)
    for _, row in label_data.iterrows():
        ax.annotate(
            str(row["customer_state"]),
            xy=(row["delay_rate"] * 100, row["avg_review_score"]),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=9,
            color=DARK,
        )

    add_title(
        ax,
        "Карта регионов: задержки доставки и средняя оценка",
    )
    ax.set_xlabel("Доля задержанных доставок")
    ax.set_ylabel("Средняя оценка")
    ax.xaxis.set_major_formatter(FuncFormatter(pct1_formatter))
    ax.set_ylim(max(0, data["avg_review_score"].min() - 0.25), min(5, data["avg_review_score"].max() + 0.25))
    clean_axes(ax)
    ax.legend(loc="lower left")
    savefig(output_dir / "11_state_experience_matrix.png")


def make_figures(db_path: Path = DB_PATH, output_dir: Path = FIGURES_DIR) -> None:
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_mart(read_mart(db_path))

    plot_review_distribution(df, output_dir)
    plot_delayed_vs_review(df, output_dir)
    plot_bad_reviews_by_category(df, output_dir)
    plot_delivery_by_state(df, output_dir)
    plot_freight_share_vs_review(df, output_dir)
    plot_monthly_metrics(df, output_dir)
    plot_category_impact_matrix(df, output_dir)
    plot_state_experience_matrix(df, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    make_figures(args.db_path, args.output_dir)


if __name__ == "__main__":
    main()
