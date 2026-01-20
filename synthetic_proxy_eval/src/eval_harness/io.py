"""IO utilities for configuration, schema inference, and dataset loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Schema:
    """Schema description for tabular datasets."""

    target: str
    task_type: str
    numeric: List[str]
    categorical: List[str]
    target_categories: Optional[List[str]]

    @property
    def features(self) -> List[str]:
        """Return ordered feature list (numeric + categorical)."""
        return list(self.numeric) + list(self.categorical)


def load_config(path: str) -> Dict:
    """Load YAML configuration from disk, merging include_datasets if present."""
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    include_path = config.pop("include_datasets", None)
    if include_path:
        if not os.path.isabs(include_path):
            base_dir = os.path.dirname(path)
            include_path = os.path.join(base_dir, include_path)

        with open(include_path, "r", encoding="utf-8") as fh:
            datasets_config = yaml.safe_load(fh)

        if "datasets" in datasets_config:
            config["datasets"] = datasets_config["datasets"]
        if "defaults" in datasets_config:
            for key, val in datasets_config["defaults"].items():
                config.setdefault(key, val)

    return config


def get_dataset_entry(config: Dict, dataset_id: str) -> Dict:
    """Look up a dataset entry by ID or name from config."""
    datasets = config.get("datasets", [])
    for entry in datasets:
        if str(entry.get("id", "")) == str(dataset_id):
            return entry
        if entry.get("name") == dataset_id:
            return entry
    raise ValueError(f"Dataset '{dataset_id}' not found in config")


def get_task_def(entry: Dict, task_id: Optional[str] = None) -> Dict:
    """Get task definition from dataset entry. Uses first task if task_id not specified."""
    task_defs = entry.get("task_defs", [])
    if not task_defs:
        raise ValueError(f"No task_defs found for dataset '{entry.get('id')}'")

    if task_id is None:
        return task_defs[0]

    for task in task_defs:
        if task.get("id") == task_id:
            return task
    raise ValueError(f"Task '{task_id}' not found in dataset '{entry.get('id')}'")


def _infer_task_type(series: pd.Series) -> str:
    """Infer task type from target series."""
    if pd.api.types.is_numeric_dtype(series):
        unique_count = series.nunique(dropna=True)
        if unique_count <= 10:
            return "classification"
        return "regression"
    return "classification"


def apply_filters(df: pd.DataFrame, filters: Optional[Dict]) -> pd.DataFrame:
    """Apply filters (e.g., drop_columns) to dataframe."""
    if filters is None:
        return df

    df = df.copy()
    drop_cols = filters.get("drop_columns", [])
    existing_drops = [c for c in drop_cols if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)

    return df


def apply_label_map(df: pd.DataFrame, target: str, label_map: Optional[Dict]) -> pd.DataFrame:
    """Apply label mapping to target column."""
    if label_map is None:
        return df

    df = df.copy()
    df[target] = df[target].map(label_map)
    return df


def apply_derived_target(
    df: pd.DataFrame,
    task_def: Dict,
) -> Tuple[pd.DataFrame, str]:
    """Create derived target column from existing column."""
    target = task_def.get("target", "")

    if not target.startswith("__derived__:"):
        return df, target

    derived_name = target.replace("__derived__:", "")
    source_col = task_def.get("derived_from")
    if source_col is None:
        raise ValueError(f"derived_from required for derived target '{derived_name}'")

    df = df.copy()

    map_gt = task_def.get("map_greater_than")
    if map_gt:
        threshold = map_gt["threshold"]
        out_leq = map_gt["out_if_leq"]
        out_gt = map_gt["out_if_gt"]
        df[derived_name] = df[source_col].apply(
            lambda x: out_gt if x > threshold else out_leq
        )
    else:
        df[derived_name] = df[source_col]

    return df, derived_name


def infer_schema(
    df: pd.DataFrame,
    target: str,
    override: Optional[Dict] = None,
    task_type: str = "auto",
) -> Schema:
    """Infer schema from dataframe with optional override."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataframe")

    if override is None:
        override = {}

    feature_cols = [c for c in df.columns if c != target]

    numeric = override.get("numeric")
    categorical = override.get("categorical")

    # Handle "__infer__" sentinel
    if numeric == "__infer__" or numeric is None:
        numeric = None
    if categorical == "__infer__" or categorical is None:
        categorical = None

    # Infer numeric columns
    if numeric is None:
        if categorical is not None:
            numeric = [c for c in feature_cols if c not in categorical]
        else:
            numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    # Infer categorical columns
    if categorical is None:
        categorical = [c for c in feature_cols if c not in numeric]

    inferred_task = _infer_task_type(df[target]) if task_type == "auto" else task_type

    target_categories: Optional[List[str]] = None
    if inferred_task == "classification":
        target_categories = sorted([str(x) for x in df[target].dropna().unique()])

    return Schema(
        target=target,
        task_type=inferred_task,
        numeric=list(numeric),
        categorical=list(categorical),
        target_categories=target_categories,
    )


def load_dataset(
    path: str,
    target: str,
    schema_override: Optional[Dict] = None,
    task_type: str = "auto",
) -> Tuple[pd.DataFrame, Schema]:
    """Load CSV dataset and infer schema."""
    df = pd.read_csv(path)
    schema = infer_schema(df, target=target, override=schema_override, task_type=task_type)
    return df, schema


def load_dataset_from_config(
    config: Dict,
    dataset_id: str,
    task_id: Optional[str] = None,
    data_root: str = ".",
) -> Tuple[pd.DataFrame, Schema, Dict, Dict]:
    """Load dataset using configuration from datasets config.

    Returns:
        Tuple of (dataframe, schema, task_def, dataset_entry)
    """
    entry = get_dataset_entry(config, dataset_id)
    task_def = get_task_def(entry, task_id)

    # Resolve path
    path = entry.get("path", "")
    if not os.path.isabs(path):
        path = os.path.join(data_root, path)

    # Load CSV
    df = pd.read_csv(path)

    # Apply filters (drop_columns, etc.)
    df = apply_filters(df, entry.get("filters"))

    # Apply derived target if needed
    df, target = apply_derived_target(df, task_def)

    # Apply label map if present
    label_map = task_def.get("label_map")
    df = apply_label_map(df, target, label_map)

    # Determine task type
    task_type_str = task_def.get("type", "auto")
    if task_type_str in ("binary_classification", "multiclass_classification"):
        task_type_str = "classification"

    # Build schema with overrides from config
    schema_override = entry.get("schema", {})
    schema = infer_schema(df, target=target, override=schema_override, task_type=task_type_str)

    return df, schema, task_def, entry


def split_train_test(
    df: pd.DataFrame,
    schema: Schema,
    seed: int,
    test_size: float = 0.2,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform deterministic train/test split with optional stratification."""
    stratify_vals = None
    if stratify and schema.task_type == "classification":
        stratify_vals = df[schema.target]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_vals,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
