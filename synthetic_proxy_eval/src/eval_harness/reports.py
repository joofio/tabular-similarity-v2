"""Reporting utilities for saving results and plots."""

from __future__ import annotations

from typing import Dict

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save dataframe to parquet."""
    df.to_parquet(path, index=False)


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save dataframe to CSV."""
    df.to_csv(path, index=False)


def plot_scatter(df: pd.DataFrame, metric: str, outpath: str) -> None:
    """Plot scatter of proxy metric vs delta-U."""
    plt.figure(figsize=(6, 4))
    plt.scatter(df[metric], df["delta_u_median"], alpha=0.7)
    plt.xlabel(metric)
    plt.ylabel("delta_u_median")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_roc_curve(fpr: pd.Series, tpr: pd.Series, outpath: str) -> None:
    """Save ROC curve plot."""
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_pr_curve(precision: pd.Series, recall: pd.Series, outpath: str) -> None:
    """Save precision-recall curve plot."""
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_degradation(df: pd.DataFrame, outpath: str) -> None:
    """Plot degradation curves for proxy metrics."""
    plt.figure(figsize=(6, 4))
    for metric, group in df.groupby("metric"):
        plt.plot(group["lambda"], group["value"], marker="o", label=metric)
    plt.xlabel("lambda")
    plt.ylabel("metric_value")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def write_reports(
    summary: pd.DataFrame,
    utility_long: pd.DataFrame,
    proxy_long: pd.DataFrame,
    fidelity: pd.DataFrame,
    stats_summary: pd.DataFrame,
    outdir: str,
    save_plots: bool = True,
) -> Dict[str, str]:
    """Write tables and plots to disk."""
    ensure_dir(outdir)
    outputs: Dict[str, str] = {}

    results_path = os.path.join(outdir, "summary.csv")
    save_csv(summary, results_path)
    outputs["results"] = results_path

    save_csv(summary, os.path.join(outdir, "summary.csv"))
    save_csv(utility_long, os.path.join(outdir, "utility_long.csv"))
    save_csv(proxy_long, os.path.join(outdir, "proxy_long.csv"))
    save_csv(fidelity, os.path.join(outdir, "fidelity.csv"))
    save_csv(stats_summary, os.path.join(outdir, "stats.csv"))

    if save_plots:
        plot_dir = os.path.join(outdir, "plots")
        ensure_dir(plot_dir)
        if not proxy_long.empty:
            metric = proxy_long["metric"].iloc[0]
            plot_scatter(summary, f"proxy_{metric}_median", os.path.join(plot_dir, "scatter.png"))
        if "lambda" in proxy_long.columns and not proxy_long.empty:
            plot_degradation(proxy_long, os.path.join(plot_dir, "degradation.png"))

    return outputs
