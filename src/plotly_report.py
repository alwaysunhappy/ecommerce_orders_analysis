from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DB_PATH, REPORTS_DIR  # noqa: E402
from src.interactive_charts import build_all_figures, build_kpi_summary, prepare_orders_mart  # noqa: E402

INTERACTIVE_DIR = REPORTS_DIR / "interactive"


def load_orders_mart(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}."
        )
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM orders_mart", conn)
    return prepare_orders_mart(df)


def _format_int(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_html_report(df: pd.DataFrame) -> str:
    summary = build_kpi_summary(df)
    figures = build_all_figures(df, min_orders=50, top_n=15)

    cards = [
        ("Заказы", _format_int(summary["orders"])),
        ("GMV", _format_int(summary["gmv"])),
        ("AOV", f"{summary['aov']:.1f}"),
        ("Средняя оценка", f"{summary['avg_review_score']:.2f}"),
        ("Плохие отзывы", _format_pct(summary["bad_review_rate"])),
        ("Доля задержек", _format_pct(summary["delay_rate"])),
        ("Среднее время доставки", f"{summary['avg_delivery_time_days']:.1f} дней"),
        ("Доля отмен", _format_pct(summary["cancel_rate"])),
    ]

    card_html = "".join(
        f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value in cards
    )

    figure_html_parts: list[str] = []
    first = True
    for name, fig in figures.items():
        figure_html_parts.append(
            f'<section class="chart"><h2>{name.replace("_", " ")}</h2>'
            + fig.to_html(full_html=False, include_plotlyjs=True if first else False)
            + "</section>"
        )
        first = False

    return f"""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <title>Olist E-commerce Interactive Analytics Report</title>
  <style>
    body {{
      margin: 0;
      padding: 0;
      background: #f5f7fb;
      color: #1f2937;
      font-family: Arial, sans-serif;
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px 24px 56px;
    }}
    .header {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 20px;
      padding: 28px 32px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
    }}
    .subtitle {{
      margin: 0;
      color: #667085;
      line-height: 1.5;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
      margin: 24px 0;
    }}
    .card {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }}
    .label {{
      color: #667085;
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .value {{
      font-size: 26px;
      font-weight: 700;
    }}
    .chart {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 20px;
      padding: 18px;
      margin: 20px 0;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }}
    .chart h2 {{
      margin: 0 0 4px 12px;
      font-size: 16px;
      color: #334155;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="header">
      <p class="subtitle">
        Анализ клиентского опыта, задержек доставки, плохих отзывов, категорий и регионов.
      </p>
    </section>
    <section class="cards">{card_html}</section>
    {''.join(figure_html_parts)}
  </main>
</body>
</html>
"""


def save_report(db_path: Path = DB_PATH, output_dir: Path = INTERACTIVE_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_orders_mart(db_path)
    html = build_html_report(df)
    output_path = output_dir / "interactive_report.html"
    output_path.write_text(html, encoding="utf-8")

    figures = build_all_figures(df, min_orders=50, top_n=15)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.write_html(charts_dir / f"{name}.html", include_plotlyjs="cdn", full_html=True)

    return output_path


def main() -> None:
    output_path = save_report()
    print(f"Report saved to {output_path}")
    print(f"HTML files saved to {output_path.parent / 'charts'}")


if __name__ == "__main__":
    main()
