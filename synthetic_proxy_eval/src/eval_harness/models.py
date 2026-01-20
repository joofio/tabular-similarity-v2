"""Model factories for classification and regression tasks."""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge


def get_models(task_type: str, seed: int) -> Dict[str, object]:
    """Return a dictionary of model name to estimator instance."""
    if task_type == "classification":
        return {
            "logreg": LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=seed,
            ),
            "rf": RandomForestClassifier(
                n_estimators=100,
                random_state=seed,
            ),
        }
    if task_type == "regression":
        return {
            "ridge": Ridge(alpha=1.0),
            "rf": RandomForestRegressor(
                n_estimators=100,
                random_state=seed,
            ),
        }
    raise ValueError(f"Unknown task type: {task_type}")
