"""Statistical validation utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def spearman_with_bootstrap(
    m_values: np.ndarray,
    delta_u_values: np.ndarray,
    n_resamples: int,
    seed: int,
) -> Dict[str, float]:
    """Compute Spearman correlation with bootstrap confidence interval."""
    corr = spearmanr(m_values, delta_u_values).correlation
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(m_values), size=len(m_values))
        boot_corr = spearmanr(m_values[idx], delta_u_values[idx]).correlation
        if np.isnan(boot_corr):
            continue
        boot.append(boot_corr)

    if not boot:
        return {"spearman": float(corr), "ci_low": float("nan"), "ci_high": float("nan")}

    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))
    return {"spearman": float(corr), "ci_low": ci_low, "ci_high": ci_high}


def _threshold_for_precision(
    scores: np.ndarray,
    labels: np.ndarray,
    precision_target: float,
) -> Tuple[float, float, float]:
    """Find threshold achieving target precision with maximal recall."""
    thresholds = sorted(set(scores), reverse=True)
    best_thr = float("nan")
    best_recall = -1.0
    best_precision = 0.0
    for thr in thresholds:
        preds = scores >= thr
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        if precision >= precision_target and recall > best_recall:
            best_recall = recall
            best_precision = precision
            best_thr = float(thr)
    return best_thr, best_precision, best_recall


def screening_metrics(
    m_values: np.ndarray,
    delta_u_values: np.ndarray,
    acceptable_threshold: float,
    precision_target: float,
) -> Dict[str, float]:
    """Compute screening metrics for acceptable synthetic datasets."""
    labels = delta_u_values <= acceptable_threshold
    if len(np.unique(labels)) < 2:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "threshold": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }

    roc_auc = roc_auc_score(labels, m_values)
    pr_auc = average_precision_score(labels, m_values)
    thr, prec, rec = _threshold_for_precision(m_values, labels, precision_target)
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": float(thr),
        "precision": float(prec),
        "recall": float(rec),
    }


def stability_std(proxy_long: pd.DataFrame) -> pd.DataFrame:
    """Compute standard deviation of proxy metrics across seeds."""
    return (
        proxy_long.groupby(["dataset_id", "metric"])["value"]
        .std()
        .reset_index()
        .rename(columns={"value": "std"})
    )


def boundary_crossing_rate(
    proxy_long: pd.DataFrame, thresholds: Dict[str, float]
) -> pd.DataFrame:
    """Compute boundary-crossing rate for proxy metrics."""
    rows = []
    for metric, thr in thresholds.items():
        subset = proxy_long[proxy_long["metric"] == metric]
        crosses = 0
        total = 0
        for _, group in subset.groupby("dataset_id"):
            values = group["value"].to_numpy()
            if len(values) == 0:
                continue
            total += 1
            if np.min(values) < thr < np.max(values):
                crosses += 1
        rate = crosses / total if total > 0 else float("nan")
        rows.append({"metric": metric, "boundary_crossing_rate": rate})
    return pd.DataFrame(rows)
