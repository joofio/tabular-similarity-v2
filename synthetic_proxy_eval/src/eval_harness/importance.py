"""Feature importance and ranking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from eval_harness.io import Schema
from eval_harness.models import get_models
from eval_harness.preprocess import PreprocessResult, split_features_target, transform_features


@dataclass
class Ranking:
    """Feature importance ranking container."""

    importance: Dict[str, float]
    ranks: Dict[str, float]
    order: List[str]


def _group_importances(
    importances: np.ndarray,
    feature_group: List[str],
) -> Dict[str, float]:
    """Group importances by original feature name."""
    grouped: Dict[str, float] = {}
    for value, group in zip(importances, feature_group):
        grouped[group] = grouped.get(group, 0.0) + abs(float(value))
    return grouped


def rank_importances(importance: Dict[str, float]) -> Ranking:
    """Rank features by absolute importance with deterministic tie handling."""
    series = pd.Series(importance)
    ranks = series.rank(ascending=False, method="average")
    order = sorted(importance.items(), key=lambda x: (-x[1], x[0]))
    ordered_features = [name for name, _ in order]
    return Ranking(importance=importance, ranks=ranks.to_dict(), order=ordered_features)


def compute_importance_rankings(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    preprocess: PreprocessResult,
    schema: Schema,
    model_names: List[str],
    seeds: List[int],
    scoring: str | None,
    n_repeats: int,
) -> Dict[Tuple[str, int], Ranking]:
    """Compute permutation importance rankings for each model/seed."""
    X_train = transform_features(preprocess, df_train, schema)
    y_train = split_features_target(df_train, schema)[1]
    X_test = transform_features(preprocess, df_test, schema)
    y_test = split_features_target(df_test, schema)[1]

    results: Dict[Tuple[str, int], Ranking] = {}
    for seed in seeds:
        models = get_models(schema.task_type, seed)
        for name, model in models.items():
            if name not in model_names:
                continue
            model.fit(X_train, y_train)
            perm = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=n_repeats,
                random_state=seed,
                scoring=scoring,
            )
            grouped = _group_importances(perm.importances_mean, preprocess.feature_group)
            ranking = rank_importances(grouped)
            results[(name, seed)] = ranking
    return results
