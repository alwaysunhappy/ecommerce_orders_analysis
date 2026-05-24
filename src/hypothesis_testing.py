from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DB_PATH, TABLES_DIR, ensure_directories
from src.metrics import read_mart


def _format_pvalue(p: float) -> float:
    if pd.isna(p):
        return np.nan
    return float(p)


def run_hypothesis_tests(db_path: Path = DB_PATH, output_dir: Path = TABLES_DIR) -> pd.DataFrame:
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_mart(db_path)
    rows: list[dict[str, object]] = []

    contingency = pd.crosstab(df["is_delayed"], df["is_bad_review"])
    if contingency.shape == (2, 2):
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        delayed_bad_rate = df.loc[df["is_delayed"] == 1, "is_bad_review"].mean()
        not_delayed_bad_rate = df.loc[df["is_delayed"] == 0, "is_bad_review"].mean()
        rows.append({
            "hypothesis": "H1: доля плохих отзывов зависит от факта задержки доставки",
            "test": "chi-square test of independence",
            "metric_1": "bad_review_rate_delayed",
            "value_1": delayed_bad_rate,
            "metric_2": "bad_review_rate_not_delayed",
            "value_2": not_delayed_bad_rate,
            "statistic": chi2,
            "p_value": _format_pvalue(p),
            "interpretation": "Различие статистически значимо" if p < 0.05 else "Нет оснований считать различие статистически значимым",
        })

    delayed_scores = df.loc[(df["is_delayed"] == 1) & df["review_score"].notna(), "review_score"].astype(float)
    not_delayed_scores = df.loc[(df["is_delayed"] == 0) & df["review_score"].notna(), "review_score"].astype(float)
    if len(delayed_scores) > 0 and len(not_delayed_scores) > 0:
        stat, p = stats.mannwhitneyu(delayed_scores, not_delayed_scores, alternative="two-sided")
        rows.append({
            "hypothesis": "H2: оценки заказов с задержкой отличаются от оценок заказов без задержки",
            "test": "Mann-Whitney U test",
            "metric_1": "mean_review_delayed",
            "value_1": delayed_scores.mean(),
            "metric_2": "mean_review_not_delayed",
            "value_2": not_delayed_scores.mean(),
            "statistic": stat,
            "p_value": _format_pvalue(p),
            "interpretation": "Различие статистически значимо" if p < 0.05 else "Нет оснований считать различие статистически значимым",
        })

    corr_df = df[["freight_share", "review_score"]].dropna()
    corr_df = corr_df[np.isfinite(corr_df["freight_share"]) & np.isfinite(corr_df["review_score"])]
    if len(corr_df) > 10:
        rho, p = stats.spearmanr(corr_df["freight_share"], corr_df["review_score"])
        rows.append({
            "hypothesis": "H3: доля доставки в стоимости заказа связана с оценкой",
            "test": "Spearman correlation",
            "metric_1": "spearman_rho",
            "value_1": rho,
            "metric_2": "n_observations",
            "value_2": len(corr_df),
            "statistic": rho,
            "p_value": _format_pvalue(p),
            "interpretation": "Связь статистически значима" if p < 0.05 else "Нет оснований считать связь статистически значимой",
        })

    category_counts = df.groupby("product_category_name")["order_id"].nunique()
    good_categories = category_counts[category_counts >= 30].index
    cat_df = df[df["product_category_name"].isin(good_categories)]
    contingency_cat = pd.crosstab(cat_df["product_category_name"], cat_df["is_bad_review"])
    if contingency_cat.shape[0] > 1 and contingency_cat.shape[1] == 2:
        chi2, p, _, _ = stats.chi2_contingency(contingency_cat)
        rows.append({
            "hypothesis": "H4: категории товаров различаются по доле плохих отзывов",
            "test": "chi-square test of independence",
            "metric_1": "n_categories",
            "value_1": len(good_categories),
            "metric_2": "min_orders_per_category",
            "value_2": 30,
            "statistic": chi2,
            "p_value": _format_pvalue(p),
            "interpretation": "Различия между категориями статистически значимы" if p < 0.05 else "Нет оснований считать различия статистически значимыми",
        })

    result = pd.DataFrame(rows).round(6)
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
