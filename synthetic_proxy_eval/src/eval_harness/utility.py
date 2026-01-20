"""Utility metrics and delta-U computation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

from eval_harness.io import Schema
from eval_harness.models import get_models
from eval_harness.preprocess import PreprocessResult, split_features_target, transform_features
from eval_harness.tasks import metric_directions, metrics_for_task


def _evaluate_classification(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Fit and evaluate classification metrics."""
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)

    y_true = y_test.to_numpy()
    # Use model's classes to ensure alignment between y_bin and proba
    classes = model.classes_

    if proba.shape[1] == 2:
        # Binary classification: encode y_true to 0/1 based on model's classes
        y_encoded = (y_true == classes[1]).astype(int)
        pos_prob = proba[:, 1]
        auroc = roc_auc_score(y_encoded, pos_prob)
        auprc = average_precision_score(y_encoded, pos_prob)
        brier = brier_score_loss(y_encoded, pos_prob)
    else:
        y_bin = label_binarize(y_true, classes=classes)
        auroc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
        auprc = average_precision_score(y_bin, proba, average="macro")
        brier = float(np.mean(np.sum((y_bin - proba) ** 2, axis=1)))

    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "brier": float(brier),
    }


def _evaluate_regression(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Fit and evaluate regression metrics."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return {"mae": float(mae), "rmse": float(rmse)}


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
) -> Dict[str, float]:
    """Evaluate a model for the given task type."""
    if task_type == "classification":
        return _evaluate_classification(model, X_train, y_train, X_test, y_test)
    if task_type == "regression":
        return _evaluate_regression(model, X_train, y_train, X_test, y_test)
    raise ValueError(f"Unknown task type: {task_type}")


def compute_model_metrics(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    preprocess: PreprocessResult,
    schema: Schema,
    model_names: List[str],
    seeds: List[int],
) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Compute model metrics for each model/seed combination."""
    X_train = transform_features(preprocess, df_train, schema)
    y_train = split_features_target(df_train, schema)[1]
    X_test = transform_features(preprocess, df_test, schema)
    y_test = split_features_target(df_test, schema)[1]

    results: Dict[Tuple[str, int], Dict[str, float]] = {}
    for seed in seeds:
        models = get_models(schema.task_type, seed)
        for name, model in models.items():
            if name not in model_names:
                continue
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test, schema.task_type)
            results[(name, seed)] = metrics
    return results


def compute_delta_u(
    rr_metrics: Dict[str, float],
    sr_metrics: Dict[str, float],
    task_type: str,
    eps: float,
) -> Dict[str, float]:
    """Compute delta-U for a set of metrics."""
    directions = metric_directions(task_type)
    deltas: Dict[str, float] = {}
    for metric in metrics_for_task(task_type):
        rr = rr_metrics[metric]
        sr = sr_metrics[metric]
        direction = directions[metric]
        if direction == "higher":
            deltas[metric] = rr - sr
        else:
            deltas[metric] = (sr - rr) / (rr + eps)
    return deltas


def aggregate_delta_u(values: List[float]) -> Tuple[float, float]:
    """Aggregate delta-U values via median and IQR."""
    arr = np.array(values, dtype=float)
    median = float(np.median(arr))
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    return median, iqr
