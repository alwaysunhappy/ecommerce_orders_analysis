from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DB_PATH, TABLES_DIR, ensure_directories
from src.metrics import (
    DELAY_BUCKET_LABELS,
    delay_bucket_series,
    read_mart,
    read_order_seller,
)

Z95 = 1.959963984540054
EFFECT_LABELS = ["незначимый", "слабый", "умеренный", "сильный"]


def _format_pvalue(p: float) -> float:
    if pd.isna(p):
        return np.nan
    return float(p)


def _wilson_ci(success: float, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = success / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (center - half, center + half)


def _mean_ci(values: pd.Series, z: float = Z95) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2:
        return (np.nan, np.nan)
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(n)
    return (mean - z * se, mean + z * se)


def _spearman_ci(rho: float, n: int, z: float = Z95) -> tuple[float, float]:
    if n < 4 or abs(rho) >= 1:
        return (np.nan, np.nan)
    fisher = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    return (float(np.tanh(fisher - z * se)), float(np.tanh(fisher + z * se)))


def _cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    k = min(r - 1, c - 1)
    if n == 0 or k == 0:
        return np.nan
    return float(np.sqrt(chi2 / (n * k)))


def _rank_biserial(u: float, n1: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return np.nan
    return float(1 - (2 * u) / (n1 * n2))


def _magnitude(value: float, thresholds: tuple[float, ...] = (0.1, 0.3, 0.5)) -> str:
    if pd.isna(value):
        return "не определена"
    a = abs(value)
    for threshold, label in zip(thresholds, EFFECT_LABELS):
        if a < threshold:
            return label
    return EFFECT_LABELS[-1]


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    idx = np.where(~np.isnan(p))[0]
    m = len(idx)
    if m == 0:
        return adjusted
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adjusted[order] = np.clip(adj, 0, 1)
    return adjusted


def _cochran_armitage(counts: np.ndarray, totals: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    n = float(totals.sum())
    r = float(counts.sum())
    if n == 0 or r == 0 or r == n:
        return (np.nan, np.nan)
    p_bar = r / n
    t = float((scores * (counts - totals * p_bar)).sum())
    var = p_bar * (1 - p_bar) * float((totals * scores ** 2).sum() - (totals * scores).sum() ** 2 / n)
    if var <= 0:
        return (np.nan, np.nan)
    z = t / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return (float(z), float(p))


def _proportion_posthoc(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    min_orders: int,
) -> pd.DataFrame:
    data = frame.copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data[data[value_col].notna()]
    if data.empty:
        return pd.DataFrame()

    total_n = len(data)
    total_success = float(data[value_col].sum())
    overall_rate = total_success / total_n

    rows: list[dict[str, object]] = []
    for segment, group in data.groupby(group_col):
        n = len(group)
        if n < min_orders:
            continue
        success = float(group[value_col].sum())
        rate = success / n
        rest_n = total_n - n
        rest_success = total_success - success
        contingency = np.array([[success, n - success], [rest_success, rest_n - rest_success]])
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        ci_low, ci_high = _wilson_ci(success, n)
        rows.append({
            group_col: segment,
            "orders": n,
            "rate": round(rate, 4),
            "rate_ci_low": round(ci_low, 4),
            "rate_ci_high": round(ci_high, 4),
            "overall_rate": round(overall_rate, 4),
            "rate_diff": round(rate - overall_rate, 4),
            "direction": "выше среднего" if rate > overall_rate else "ниже среднего",
            "statistic": round(float(chi2), 4),
            "p_value": _format_pvalue(p),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_value_adjusted"] = _benjamini_hochberg(result["p_value"].to_numpy())
    result["significant"] = result["p_value_adjusted"] < 0.05
    return result.sort_values("rate_diff", ascending=False).round(6)


def run_category_posthoc(df: pd.DataFrame, output_dir: Path = TABLES_DIR, min_orders: int = 100) -> pd.DataFrame:
    reviewed = df[df["is_bad_review"].notna()]
    result = _proportion_posthoc(reviewed, "product_category_name", "is_bad_review", min_orders)
    if not result.empty:
        path = output_dir / "category_posthoc.csv"
        result.to_csv(path, index=False)
        print(f"Saved {path}")
    return result


def run_state_posthoc(df: pd.DataFrame, output_dir: Path = TABLES_DIR, min_orders: int = 100) -> pd.DataFrame:
    delivered = df[(pd.to_numeric(df["is_delivered"], errors="coerce") == 1) & df["is_delayed"].notna()]
    result = _proportion_posthoc(delivered, "customer_state", "is_delayed", min_orders)
    if not result.empty:
        path = output_dir / "state_posthoc.csv"
        result.to_csv(path, index=False)
        print(f"Saved {path}")
    return result


def _interpret(row: pd.Series) -> str:
    if pd.isna(row["p_value"]):
        return "Тест не проводился"
    base = "статистически значимо" if row["significant"] else "статистически незначимо"
    return (
        f"После поправки BH: {base} (p_adj={row['p_value_adjusted']:.4g}); "
        f"эффект {row['effect_magnitude']} ({row['effect_size_name']}={row['effect_size']:.3g})"
    )


def run_seller_within_category_tests(
    order_seller_df: pd.DataFrame,
    output_dir: Path = TABLES_DIR,
    min_seller_orders: int = 30,
    min_sellers: int = 3,
    single_seller_only: bool = True,
) -> pd.DataFrame:
    if order_seller_df.empty:
        return pd.DataFrame()

    df = order_seller_df.copy()
    if single_seller_only and "is_single_seller_order" in df.columns:
        df = df[pd.to_numeric(df["is_single_seller_order"], errors="coerce") == 1]
    df["is_bad_review"] = pd.to_numeric(df["is_bad_review"], errors="coerce")
    reviewed = df[df["is_bad_review"].notna()]
    if reviewed.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for category, group in reviewed.groupby("product_category_name"):
        seller_orders = group.groupby("seller_id")["order_id"].nunique()
        eligible_sellers = seller_orders[seller_orders >= min_seller_orders].index
        if len(eligible_sellers) < min_sellers:
            continue
        focal = group[group["seller_id"].isin(eligible_sellers)]
        contingency = pd.crosstab(focal["seller_id"], focal["is_bad_review"])
        if contingency.shape[0] < 2 or contingency.shape[1] != 2:
            continue
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        n = int(contingency.to_numpy().sum())
        v = _cramers_v(chi2, n, contingency.shape[0], contingency.shape[1])
        seller_rates = focal.groupby("seller_id")["is_bad_review"].mean()
        rows.append({
            "product_category_name": category,
            "n_sellers": int(len(eligible_sellers)),
            "n_observations": n,
            "category_bad_review_rate": float(focal["is_bad_review"].mean()),
            "min_seller_bad_review_rate": float(seller_rates.min()),
            "max_seller_bad_review_rate": float(seller_rates.max()),
            "statistic": float(chi2),
            "effect_size_name": "Cramér's V",
            "effect_size": v,
            "effect_magnitude": _magnitude(v),
            "p_value": _format_pvalue(p),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_adjusted"] = _benjamini_hochberg(result["p_value"].to_numpy())
        result["significant"] = result["p_value_adjusted"] < 0.05
        result = result.sort_values("n_observations", ascending=False).round(6)
        path = output_dir / "seller_within_category_tests.csv"
        result.to_csv(path, index=False)
        print(f"Saved {path}")
    return result


def run_hypothesis_tests(db_path: Path = DB_PATH, output_dir: Path = TABLES_DIR) -> pd.DataFrame:
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_mart(db_path)
    rows: list[dict[str, object]] = []

    delivered = df[df["is_delayed"].notna()].copy()
    delivered["is_delayed"] = delivered["is_delayed"].astype(int)

    h1 = delivered[delivered["is_bad_review"].notna()]
    contingency = pd.crosstab(h1["is_delayed"], h1["is_bad_review"])
    if contingency.shape == (2, 2):
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        n = int(contingency.to_numpy().sum())
        delayed = h1.loc[h1["is_delayed"] == 1, "is_bad_review"]
        not_delayed = h1.loc[h1["is_delayed"] == 0, "is_bad_review"]
        ci_d = _wilson_ci(float(delayed.sum()), len(delayed))
        ci_nd = _wilson_ci(float(not_delayed.sum()), len(not_delayed))
        v = _cramers_v(chi2, n, 2, 2)
        rows.append({
            "hypothesis": "H1: доля плохих отзывов зависит от факта задержки доставки",
            "test": "chi-square test of independence",
            "scope": "доставленные заказы с отзывом",
            "metric_1": "bad_review_rate_delayed",
            "value_1": float(delayed.mean()),
            "ci_low_1": ci_d[0],
            "ci_high_1": ci_d[1],
            "metric_2": "bad_review_rate_not_delayed",
            "value_2": float(not_delayed.mean()),
            "ci_low_2": ci_nd[0],
            "ci_high_2": ci_nd[1],
            "n_observations": n,
            "statistic": chi2,
            "effect_size_name": "Cramér's V",
            "effect_size": v,
            "effect_magnitude": _magnitude(v),
            "p_value": _format_pvalue(p),
        })

    delayed_scores = delivered.loc[(delivered["is_delayed"] == 1) & delivered["review_score"].notna(), "review_score"].astype(float)
    not_delayed_scores = delivered.loc[(delivered["is_delayed"] == 0) & delivered["review_score"].notna(), "review_score"].astype(float)
    if len(delayed_scores) > 0 and len(not_delayed_scores) > 0:
        u, p = stats.mannwhitneyu(delayed_scores, not_delayed_scores, alternative="two-sided")
        rb = _rank_biserial(u, len(delayed_scores), len(not_delayed_scores))
        ci_d = _mean_ci(delayed_scores)
        ci_nd = _mean_ci(not_delayed_scores)
        rows.append({
            "hypothesis": "H2: оценки заказов с задержкой отличаются от оценок заказов без задержки",
            "test": "Mann-Whitney U test",
            "scope": "доставленные заказы с отзывом",
            "metric_1": "mean_review_delayed",
            "value_1": float(delayed_scores.mean()),
            "ci_low_1": ci_d[0],
            "ci_high_1": ci_d[1],
            "metric_2": "mean_review_not_delayed",
            "value_2": float(not_delayed_scores.mean()),
            "ci_low_2": ci_nd[0],
            "ci_high_2": ci_nd[1],
            "n_observations": int(len(delayed_scores) + len(not_delayed_scores)),
            "statistic": float(u),
            "effect_size_name": "rank-biserial correlation",
            "effect_size": rb,
            "effect_magnitude": _magnitude(rb),
            "p_value": _format_pvalue(p),
        })

    corr_df = df[["freight_share", "review_score"]].dropna()
    corr_df = corr_df[np.isfinite(corr_df["freight_share"]) & np.isfinite(corr_df["review_score"])]
    if len(corr_df) > 10:
        rho, p = stats.spearmanr(corr_df["freight_share"], corr_df["review_score"])
        ci = _spearman_ci(rho, len(corr_df))
        rows.append({
            "hypothesis": "H3: доля доставки в стоимости заказа связана с оценкой",
            "test": "Spearman correlation",
            "scope": "заказы с отзывом и известной долей доставки",
            "metric_1": "spearman_rho",
            "value_1": float(rho),
            "ci_low_1": ci[0],
            "ci_high_1": ci[1],
            "metric_2": "n_observations",
            "value_2": int(len(corr_df)),
            "ci_low_2": np.nan,
            "ci_high_2": np.nan,
            "n_observations": int(len(corr_df)),
            "statistic": float(rho),
            "effect_size_name": "Spearman rho",
            "effect_size": float(rho),
            "effect_magnitude": _magnitude(rho),
            "p_value": _format_pvalue(p),
        })

    reviewed = df[df["is_bad_review"].notna()]
    category_counts = reviewed.groupby("product_category_name")["order_id"].nunique()
    good_categories = category_counts[category_counts >= 30].index
    cat_df = reviewed[reviewed["product_category_name"].isin(good_categories)]
    contingency_cat = pd.crosstab(cat_df["product_category_name"], cat_df["is_bad_review"])
    if contingency_cat.shape[0] > 1 and contingency_cat.shape[1] == 2:
        chi2, p, _, _ = stats.chi2_contingency(contingency_cat)
        n = int(contingency_cat.to_numpy().sum())
        v = _cramers_v(chi2, n, contingency_cat.shape[0], 2)
        rows.append({
            "hypothesis": "H4: категории товаров различаются по доле плохих отзывов",
            "test": "chi-square test of independence",
            "scope": "заказы с отзывом, категории с >= 30 заказами",
            "metric_1": "n_categories",
            "value_1": int(len(good_categories)),
            "ci_low_1": np.nan,
            "ci_high_1": np.nan,
            "metric_2": "min_orders_per_category",
            "value_2": 30,
            "ci_low_2": np.nan,
            "ci_high_2": np.nan,
            "n_observations": n,
            "statistic": chi2,
            "effect_size_name": "Cramér's V",
            "effect_size": v,
            "effect_magnitude": _magnitude(v),
            "p_value": _format_pvalue(p),
        })

    state_counts = delivered.groupby("customer_state")["order_id"].nunique()
    good_states = state_counts[state_counts >= 30].index
    state_df = delivered[delivered["customer_state"].isin(good_states)]
    contingency_state = pd.crosstab(state_df["customer_state"], state_df["is_delayed"])
    if contingency_state.shape[0] > 1 and contingency_state.shape[1] == 2:
        chi2, p, _, _ = stats.chi2_contingency(contingency_state)
        n = int(contingency_state.to_numpy().sum())
        v = _cramers_v(chi2, n, contingency_state.shape[0], 2)
        rows.append({
            "hypothesis": "H5: регионы покупателей различаются по доле задержанных доставок",
            "test": "chi-square test of independence",
            "scope": "доставленные заказы, регионы с >= 30 заказами",
            "metric_1": "n_states",
            "value_1": int(len(good_states)),
            "ci_low_1": np.nan,
            "ci_high_1": np.nan,
            "metric_2": "min_orders_per_state",
            "value_2": 30,
            "ci_low_2": np.nan,
            "ci_high_2": np.nan,
            "n_observations": n,
            "statistic": chi2,
            "effect_size_name": "Cramér's V",
            "effect_size": v,
            "effect_magnitude": _magnitude(v),
            "p_value": _format_pvalue(p),
        })

    h7 = df[df["is_late_handover"].notna() & df["is_bad_review"].notna()].copy()
    if not h7.empty:
        h7["is_late_handover"] = pd.to_numeric(h7["is_late_handover"], errors="coerce").astype(int)
        contingency_h7 = pd.crosstab(h7["is_late_handover"], h7["is_bad_review"])
        if contingency_h7.shape == (2, 2):
            chi2, p, _, _ = stats.chi2_contingency(contingency_h7)
            n = int(contingency_h7.to_numpy().sum())
            late = h7.loc[h7["is_late_handover"] == 1, "is_bad_review"]
            on_time = h7.loc[h7["is_late_handover"] == 0, "is_bad_review"]
            ci_late = _wilson_ci(float(late.sum()), len(late))
            ci_on_time = _wilson_ci(float(on_time.sum()), len(on_time))
            v = _cramers_v(chi2, n, 2, 2)
            rows.append({
                "hypothesis": "H7: поздняя передача перевозчику (зона продавца) связана с плохим отзывом",
                "test": "chi-square test of independence",
                "scope": "заказы с отзывом и известным фактом просрочки отгрузки",
                "metric_1": "bad_review_rate_late_handover",
                "value_1": float(late.mean()),
                "ci_low_1": ci_late[0],
                "ci_high_1": ci_late[1],
                "metric_2": "bad_review_rate_on_time_handover",
                "value_2": float(on_time.mean()),
                "ci_low_2": ci_on_time[0],
                "ci_high_2": ci_on_time[1],
                "n_observations": n,
                "statistic": chi2,
                "effect_size_name": "Cramér's V",
                "effect_size": v,
                "effect_magnitude": _magnitude(v),
                "p_value": _format_pvalue(p),
            })

    buckets = delivered.copy()
    buckets["delay_bucket"] = delay_bucket_series(buckets)
    buckets = buckets[buckets["delay_bucket"].notna() & buckets["is_bad_review"].notna()]
    if not buckets.empty:
        score_map = {label: i for i, label in enumerate(DELAY_BUCKET_LABELS)}
        buckets = buckets.assign(bucket_score=buckets["delay_bucket"].map(score_map).astype(float))
        counts = np.array([
            buckets.loc[buckets["delay_bucket"] == label, "is_bad_review"].sum()
            for label in DELAY_BUCKET_LABELS
        ], dtype=float)
        totals = np.array([
            int((buckets["delay_bucket"] == label).sum())
            for label in DELAY_BUCKET_LABELS
        ], dtype=float)
        scores = np.arange(len(DELAY_BUCKET_LABELS), dtype=float)
        z, p = _cochran_armitage(counts, totals, scores)
        if not pd.isna(p):
            rho, _ = stats.spearmanr(buckets["bucket_score"], buckets["is_bad_review"])
            rate_no_delay = float(buckets.loc[buckets["delay_bucket"] == DELAY_BUCKET_LABELS[0], "is_bad_review"].mean())
            rate_max_delay = float(buckets.loc[buckets["delay_bucket"] == DELAY_BUCKET_LABELS[-1], "is_bad_review"].mean())
            rows.append({
                "hypothesis": "H8: вероятность плохого отзыва растёт с длительностью задержки",
                "test": "Cochran-Armitage trend test",
                "scope": "доставленные заказы с отзывом, по корзинам длительности задержки",
                "metric_1": "bad_review_rate_no_delay",
                "value_1": rate_no_delay,
                "ci_low_1": np.nan,
                "ci_high_1": np.nan,
                "metric_2": "bad_review_rate_14plus",
                "value_2": rate_max_delay,
                "ci_low_2": np.nan,
                "ci_high_2": np.nan,
                "n_observations": int(totals.sum()),
                "statistic": float(z),
                "effect_size_name": "Spearman rho (корзина vs плохой отзыв)",
                "effect_size": float(rho),
                "effect_magnitude": _magnitude(rho),
                "p_value": _format_pvalue(p),
            })

    run_category_posthoc(df, output_dir)
    run_state_posthoc(df, output_dir)

    seller_tests = run_seller_within_category_tests(read_order_seller(db_path), output_dir)
    if not seller_tests.empty:
        focal = seller_tests.iloc[0]
        rows.append({
            "hypothesis": "H6: внутри категории продавцы различаются по доле плохих отзывов",
            "test": "chi-square test of independence",
            "scope": f"single-seller заказы с отзывом, фокусная категория {focal['product_category_name']} (>= 30 заказов на продавца)",
            "metric_1": "n_sellers",
            "value_1": int(focal["n_sellers"]),
            "ci_low_1": np.nan,
            "ci_high_1": np.nan,
            "metric_2": "category_bad_review_rate",
            "value_2": float(focal["category_bad_review_rate"]),
            "ci_low_2": float(focal["min_seller_bad_review_rate"]),
            "ci_high_2": float(focal["max_seller_bad_review_rate"]),
            "n_observations": int(focal["n_observations"]),
            "statistic": float(focal["statistic"]),
            "effect_size_name": "Cramér's V",
            "effect_size": float(focal["effect_size"]),
            "effect_magnitude": focal["effect_magnitude"],
            "p_value": _format_pvalue(float(focal["p_value"])),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_adjusted"] = _benjamini_hochberg(result["p_value"].to_numpy())
        result["significant"] = result["p_value_adjusted"] < 0.05
        result["interpretation"] = result.apply(_interpret, axis=1)

    result = result.round(6)
    path = output_dir / "hypothesis_tests.csv"
    result.to_csv(path, index=False)
    print(f"Saved {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()
    run_hypothesis_tests(args.db_path, args.output_dir)


if __name__ == "__main__":
    main()
