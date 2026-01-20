"""Preprocessing utilities with deterministic, leakage-free fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from eval_harness.io import Schema


@dataclass
class PreprocessResult:
    """Fitted preprocessing pipeline and feature metadata."""

    transformer: ColumnTransformer
    feature_names: List[str]
    feature_group: List[str]
    original_features: List[str]


def _make_onehot_encoder() -> OneHotEncoder:
    """Create a version-tolerant one-hot encoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _get_scaler(scaler_type: str):
    """Get scaler instance based on type string."""
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler()
    if scaler_type in ("none", None):
        return None
    raise ValueError(f"Unknown scaler type: {scaler_type}. Use 'standard', 'minmax', or 'none'.")


def build_preprocessor(schema: Schema, scaler_type: str = "standard") -> ColumnTransformer:
    """Build preprocessing transformer for numeric and categorical features."""
    scaler = _get_scaler(scaler_type)

    if scaler is not None:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", scaler),
            ]
        )
    else:
        num_pipe = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_onehot_encoder()),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipe, schema.numeric),
            ("cat", cat_pipe, schema.categorical),
        ],
        remainder="drop",
    )
    return transformer


def fit_preprocess(
    df_train: pd.DataFrame, schema: Schema, scaler_type: str = "standard"
) -> PreprocessResult:
    """Fit preprocessing on training data only and return metadata."""
    transformer = build_preprocessor(schema, scaler_type=scaler_type)
    X_train = df_train[schema.features]
    transformer.fit(X_train)

    feature_names: List[str] = []
    feature_group: List[str] = []

    for name in schema.numeric:
        feature_names.append(name)
        feature_group.append(name)

    if schema.categorical:
        encoder = transformer.named_transformers_["cat"].named_steps["onehot"]
        for feature, cats in zip(schema.categorical, encoder.categories_):
            for cat in cats:
                feature_names.append(f"{feature}__{cat}")
                feature_group.append(feature)

    original_features = schema.features

    return PreprocessResult(
        transformer=transformer,
        feature_names=feature_names,
        feature_group=feature_group,
        original_features=original_features,
    )


def split_features_target(df: pd.DataFrame, schema: Schema) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into feature matrix and target vector."""
    return df[schema.features], df[schema.target]


def transform_features(
    preprocess: PreprocessResult,
    df: pd.DataFrame,
    schema: Schema,
) -> pd.DataFrame:
    """Transform dataframe to model-ready features with consistent columns."""
    X, _ = split_features_target(df, schema)
    array = preprocess.transformer.transform(X)
    return pd.DataFrame(array, columns=preprocess.feature_names)
