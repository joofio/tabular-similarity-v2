"""Task definitions and metric directions."""

from __future__ import annotations

from typing import Dict, List


CLASSIFICATION_METRICS: List[str] = [
    "auroc",
    "auprc",
    "macro_f1",
    "balanced_accuracy",
    "brier",
]

REGRESSION_METRICS: List[str] = ["mae", "rmse"]


def metric_directions(task_type: str) -> Dict[str, str]:
    """Return per-metric directionality."""
    if task_type == "classification":
        return {
            "auroc": "higher",
            "auprc": "higher",
            "macro_f1": "higher",
            "balanced_accuracy": "higher",
            "brier": "lower",
        }
    if task_type == "regression":
        return {"mae": "lower", "rmse": "lower"}
    raise ValueError(f"Unknown task type: {task_type}")


def metrics_for_task(task_type: str) -> List[str]:
    """Return list of metrics for a given task."""
    if task_type == "classification":
        return list(CLASSIFICATION_METRICS)
    if task_type == "regression":
        return list(REGRESSION_METRICS)
    raise ValueError(f"Unknown task type: {task_type}")
