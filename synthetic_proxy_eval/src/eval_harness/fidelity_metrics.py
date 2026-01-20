"""Fidelity metrics comparing real and synthetic data distributions."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from eval_harness.io import Schema
from eval_harness.preprocess import PreprocessResult, transform_features


def _wasserstein_numeric(df_real: pd.DataFrame, df_synth: pd.DataFrame, cols: List[str]) -> float:
    """Average Wasserstein distance for numeric marginals."""
    if not cols:
        return 0.0
    distances = []
    for col in cols:
        real_vals = df_real[col].dropna().to_numpy()
        synth_vals = df_synth[col].dropna().to_numpy()
        if len(real_vals) == 0 or len(synth_vals) == 0:
            distances.append(0.0)
        else:
            distances.append(float(wasserstein_distance(real_vals, synth_vals)))
    return float(np.mean(distances))


def _jsd_categorical(df_real: pd.DataFrame, df_synth: pd.DataFrame, cols: List[str]) -> float:
    """Average Jensen-Shannon distance for categorical marginals."""
    if not cols:
        return 0.0
    distances = []
    for col in cols:
        real_counts = df_real[col].astype(str).value_counts(normalize=True)
        synth_counts = df_synth[col].astype(str).value_counts(normalize=True)
        categories = sorted(set(real_counts.index).union(set(synth_counts.index)))
        real_probs = np.array([real_counts.get(cat, 0.0) for cat in categories], dtype=float)
        synth_probs = np.array([synth_counts.get(cat, 0.0) for cat in categories], dtype=float)
        distances.append(float(jensenshannon(real_probs, synth_probs)))
    return float(np.mean(distances))


def _correlation_distance(df_real: pd.DataFrame, df_synth: pd.DataFrame, cols: List[str]) -> float:
    """Frobenius norm distance between correlation matrices."""
    if len(cols) < 2:
        return 0.0
    corr_real = df_real[cols].corr().to_numpy()
    corr_synth = df_synth[cols].corr().to_numpy()
    diff = corr_real - corr_synth
    return float(np.linalg.norm(diff, ord="fro") / len(cols))


def _propensity_auc(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    preprocess: PreprocessResult,
    schema: Schema,
    seed: int,
    test_size: float,
    max_iter: int,
) -> float:
    """Train a propensity classifier to distinguish real vs synthetic."""
    df_real = df_real.copy()
    df_synth = df_synth.copy()
    df_real["_label"] = 1
    df_synth["_label"] = 0
    df_all = pd.concat([df_real, df_synth], ignore_index=True)

    X = transform_features(preprocess, df_all, schema)
    y = df_all["_label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    clf = LogisticRegression(max_iter=max_iter, solver="lbfgs", random_state=seed)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, probs))


def compute_fidelity_metrics(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    schema: Schema,
    preprocess: PreprocessResult,
    seed: int,
    test_size: float,
    max_iter: int,
) -> Dict[str, float]:
    """Compute fidelity metrics for a synthetic dataset."""
    metrics = {
        "wasserstein_numeric": _wasserstein_numeric(df_real, df_synth, schema.numeric),
        "jsd_categorical": _jsd_categorical(df_real, df_synth, schema.categorical),
        "correlation_distance": _correlation_distance(df_real, df_synth, schema.numeric),
        "propensity_auc": _propensity_auc(
            df_real,
            df_synth,
            preprocess=preprocess,
            schema=schema,
            seed=seed,
            test_size=test_size,
            max_iter=max_iter,
        ),
    }
    return metrics
