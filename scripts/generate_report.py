"""
Generate a comprehensive HTML/Markdown research report.

Usage:
    python scripts/generate_report.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_json, get_logger, ensure_dir

logger = get_logger(__name__)


def generate_markdown_report(output_dir: str = "outputs") -> str:
    """Generate a Markdown research report from pipeline outputs."""
    output_path = Path(output_dir)
    report_parts = []

    report_parts.append(f"""# Alpha Factor Discovery & Quantitative Backtesting Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Executive Summary

This report presents the results of an end-to-end quantitative research pipeline
that engineers alpha factors from historical financial data and evaluates ML models
for generating profitable trading signals under realistic backtesting assumptions.

---
""")

    # Metrics
    metrics_path = output_path / "metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path)

        report_parts.append("## 2. Model Performance\n\n")

        if "classification" in metrics:
            clf = metrics["classification"]
            report_parts.append("### Classification Metrics\n\n")
            report_parts.append("| Metric | Value |\n|--------|-------|\n")
            for k, v in clf.items():
                report_parts.append(f"| {k.replace('_', ' ').title()} | {v:.4f} |\n")
            report_parts.append("\n")

        if "trading" in metrics:
            trd = metrics["trading"]
            report_parts.append("### Trading Metrics\n\n")
            report_parts.append("| Metric | Value |\n|--------|-------|\n")
            for k, v in trd.items():
                if isinstance(v, float):
                    if "ratio" in k or "rate" in k or "factor" in k:
                        report_parts.append(f"| {k.replace('_', ' ').title()} | {v:.4f} |\n")
                    else:
                        report_parts.append(f"| {k.replace('_', ' ').title()} | {v:.4f} |\n")
            report_parts.append("\n")

        if "statistical" in metrics:
            stat = metrics["statistical"]
            report_parts.append("### Statistical Tests\n\n")
            if "sharpe_bootstrap" in stat:
                sb = stat["sharpe_bootstrap"]
                report_parts.append(f"**Sharpe Ratio Bootstrap (95% CI)**: "
                                    f"{sb.get('ci_lower', 0):.3f} — {sb.get('ci_upper', 0):.3f}\n\n")
            if "mean_return_ttest" in stat:
                tt = stat["mean_return_ttest"]
                sig = "✅ Yes" if tt.get("significant_5pct", False) else "❌ No"
                report_parts.append(f"**Mean Return t-test**: t={tt.get('t_stat', 0):.3f}, "
                                    f"p={tt.get('p_value', 1):.4f}, significant: {sig}\n\n")

    # Plots
    plots_dir = output_path / "plots"
    if plots_dir.exists():
        report_parts.append("---\n\n## 3. Visualizations\n\n")
        for plot in sorted(plots_dir.glob("*.png")):
            name = plot.stem.replace("_", " ").title()
            report_parts.append(f"### {name}\n\n")
            report_parts.append(f"![{name}]({plot.name})\n\n")

    # Ablation
    ablation_path = output_path / "ablation_results.csv"
    if ablation_path.exists():
        import pandas as pd
        ablation = pd.read_csv(ablation_path, index_col=0)
        report_parts.append("---\n\n## 4. Ablation Study\n\n")
        report_parts.append(ablation.to_markdown())
        report_parts.append("\n\n")

    # Save
    report_text = "".join(report_parts)
    report_file = output_path / "final_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("Report generated → %s", report_file)
    return str(report_file)


if __name__ == "__main__":
    generate_markdown_report()
