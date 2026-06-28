from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DB_PATH, TABLES_DIR, ensure_directories
from src.metrics import read_mart

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


def _interpret(row: pd.Series) -> str:
    if pd.isna(row["p_value"]):
        return "Тест не проводился"
    base = "статистически значимо" if row["significant"] else "статистически незначимо"
    return (
        f"После поправки BH: {base} (p_adj={row['p_value_adjusted']:.4g}); "
        f"эффект {row['effect_magnitude']} ({row['effect_size_name']}={row['effect_size']:.3g})"
    )


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
