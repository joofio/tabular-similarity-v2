"""Reporting utilities for saving results and plots."""

from __future__ import annotations

from typing import Dict, List

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def _read_csv(path: str) -> pd.DataFrame:
    """Read a CSV file into a dataframe."""
    return pd.read_csv(path)


def _collect_dataset_paths(parent_outdir: str) -> List[str]:
    """Collect dataset output directories (excluding aggregate)."""
    dirs = []
    for entry in os.listdir(parent_outdir):
        full_path = os.path.join(parent_outdir, entry)
        if entry == "aggregate":
            continue
        if os.path.isdir(full_path):
            dirs.append(full_path)
    return sorted(dirs)


def _aggregate_summary(summary_all: pd.DataFrame) -> pd.DataFrame:
    """Aggregate summary metrics across datasets."""
    group_cols = ["generator", "size"]
    metric_cols = [
        col
        for col in summary_all.columns
        if col not in group_cols + ["dataset_id", "seed", "source_dataset"]
        and np.issubdtype(summary_all[col].dtype, np.number)
    ]
    agg_df = (
        summary_all.groupby(group_cols)[metric_cols]
        .median()
        .reset_index()
        .rename(columns={col: f"{col}_median" for col in metric_cols})
    )
    counts = summary_all.groupby(group_cols).agg(
        dataset_count=("source_dataset", "nunique"),
        synth_count=("dataset_id", "count"),
    )
    agg_df = agg_df.merge(counts.reset_index(), on=group_cols, how="left")
    return agg_df


def aggregate_reports(parent_outdir: str) -> Dict[str, str]:
    """Aggregate results across multiple dataset runs."""
    dataset_dirs = _collect_dataset_paths(parent_outdir)
    if not dataset_dirs:
        return {}

    summary_frames = []
    utility_frames = []
    proxy_frames = []
    fidelity_frames = []
    stats_frames = []

    for dataset_dir in dataset_dirs:
        dataset_id = os.path.basename(dataset_dir)
        summary_path = os.path.join(dataset_dir, "summary.csv")
        if not os.path.exists(summary_path):
            continue

        summary = _read_csv(summary_path)
        summary["source_dataset"] = dataset_id
        summary_frames.append(summary)

        for name, collector in [
            ("utility_long.csv", utility_frames),
            ("proxy_long.csv", proxy_frames),
            ("fidelity.csv", fidelity_frames),
            ("stats.csv", stats_frames),
        ]:
            path = os.path.join(dataset_dir, name)
            if os.path.exists(path):
                df = _read_csv(path)
                df["source_dataset"] = dataset_id
                collector.append(df)

    if not summary_frames:
        return {}

    summary_all = pd.concat(summary_frames, ignore_index=True)
    utility_all = pd.concat(utility_frames, ignore_index=True) if utility_frames else pd.DataFrame()
    proxy_all = pd.concat(proxy_frames, ignore_index=True) if proxy_frames else pd.DataFrame()
    fidelity_all = pd.concat(fidelity_frames, ignore_index=True) if fidelity_frames else pd.DataFrame()
    stats_all = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame()

    agg_dir = os.path.join(parent_outdir, "aggregate")
    ensure_dir(agg_dir)

    outputs: Dict[str, str] = {}
    outputs["summary_all"] = os.path.join(agg_dir, "summary_all.csv")
    save_csv(summary_all, outputs["summary_all"])

    if not utility_all.empty:
        outputs["utility_all"] = os.path.join(agg_dir, "utility_all.csv")
        save_csv(utility_all, outputs["utility_all"])
    if not proxy_all.empty:
        outputs["proxy_all"] = os.path.join(agg_dir, "proxy_all.csv")
        save_csv(proxy_all, outputs["proxy_all"])
    if not fidelity_all.empty:
        outputs["fidelity_all"] = os.path.join(agg_dir, "fidelity_all.csv")
        save_csv(fidelity_all, outputs["fidelity_all"])
    if not stats_all.empty:
        outputs["stats_all"] = os.path.join(agg_dir, "stats_all.csv")
        save_csv(stats_all, outputs["stats_all"])

    aggregated_summary = _aggregate_summary(summary_all)
    outputs["summary_aggregated"] = os.path.join(agg_dir, "summary_aggregated.csv")
    save_csv(aggregated_summary, outputs["summary_aggregated"])

    return outputs
