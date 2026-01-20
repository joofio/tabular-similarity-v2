"""Plot top proxy metrics against delta-U from run CSVs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _sanitize_filename(value: str) -> str:
    """Create a filesystem-safe filename fragment."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _select_top_metrics(stats: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Select the top metrics by absolute Spearman correlation."""
    stats = stats.dropna(subset=["spearman"]).copy()
    if stats.empty:
        return stats
    stats["abs_spearman"] = stats["spearman"].abs()
    stats = stats.sort_values("abs_spearman", ascending=False)
    return stats.head(top_n)


def _iter_dataset_dirs(runs_dir: Path) -> Iterable[Path]:
    """Yield dataset directories containing summary and stats CSVs."""
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir():
            continue
        if (path / "summary.csv").exists() and (path / "stats.csv").exists():
            yield path


def _plot_metric(
    dataset: str,
    metric: str,
    summary: pd.DataFrame,
    spearman: float,
    outdir: Path,
    dpi: int,
) -> None:
    """Create a scatter plot for a single metric."""
    proxy_col = f"proxy_{metric}_median"
    if proxy_col not in summary.columns:
        return

    data = summary[[proxy_col, "delta_u_median"]].dropna()
    if len(data) < 2:
        return

    x = data[proxy_col].to_numpy()
    y = data["delta_u_median"].to_numpy()
    r2 = float(spearman) ** 2

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x, y, alpha=0.75, edgecolors="none")
    ax.set_xlabel(f"{metric} (proxy median)")
    ax.set_ylabel("delta_u_median")
    ax.set_title(f"{dataset} - {metric}")
    ax.text(
        0.02,
        0.98,
        f"Spearman R^2 = {r2:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    filename = f"{_sanitize_filename(metric)}.png"
    fig.savefig(outdir / filename, dpi=dpi)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Plot top proxy metrics vs delta-U.")
    parser.add_argument(
        "--runs-dir",
        default="runs/experiment1",
        help="Run directory containing dataset subfolders",
    )
    parser.add_argument(
        "--outdir",
        default="runs/experiment1/plots_top_metrics",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top metrics to plot per dataset",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the plotting utility."""
    parser = build_parser()
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    outdir = Path(args.outdir)

    for dataset_dir in _iter_dataset_dirs(runs_dir):
        summary = pd.read_csv(dataset_dir / "summary.csv")
        stats = pd.read_csv(dataset_dir / "stats.csv")
        top = _select_top_metrics(stats, args.top_n)
        if top.empty:
            continue

        dataset_name = dataset_dir.name
        dataset_outdir = outdir / dataset_name
        for _, row in top.iterrows():
            metric = str(row["metric"])
            spearman = float(row["spearman"])
            _plot_metric(dataset_name, metric, summary, spearman, dataset_outdir, args.dpi)


if __name__ == "__main__":
    main()
