"""Synthetic data generators and dataset bank creation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from eval_harness.io import Schema


class SynthGenerator:
    """Base interface for synthetic data generators."""

    name: str = "base"

    def fit(self, df_train: pd.DataFrame, schema: Schema, seed: int) -> None:
        """Fit the generator to training data."""
        raise NotImplementedError

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Sample synthetic data of size n."""
        raise NotImplementedError


class BootstrapGenerator(SynthGenerator):
    """Bootstrap resampling generator."""

    name = "bootstrap"

    def __init__(self) -> None:
        self._df_train: pd.DataFrame | None = None

    def fit(self, df_train: pd.DataFrame, schema: Schema, seed: int) -> None:
        """Store the training data for bootstrap sampling."""
        del schema
        del seed
        self._df_train = df_train.reset_index(drop=True)

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Sample rows with replacement from training data."""
        if self._df_train is None:
            raise ValueError("Generator must be fit before sampling")
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(self._df_train), size=n)
        return self._df_train.iloc[idx].reset_index(drop=True)


class GaussianCopulaGenerator(SynthGenerator):
    """Gaussian copula-like generator for numeric features."""

    name = "gaussian_copula"

    def __init__(self) -> None:
        self._numeric_cols: List[str] = []
        self._categorical_cols: List[str] = []
        self._mean: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._cat_probs: Dict[str, Dict] = {}
        self._cat_dtypes: Dict[str, type] = {}
        self._col_order: List[str] = []

    def fit(self, df_train: pd.DataFrame, schema: Schema, seed: int) -> None:
        """Fit numeric and categorical distributions from training data."""
        del seed
        numeric_cols = list(schema.numeric)
        categorical_cols = list(schema.categorical)

        if schema.task_type == "regression":
            numeric_cols.append(schema.target)
        else:
            categorical_cols.append(schema.target)

        self._numeric_cols = numeric_cols
        self._categorical_cols = categorical_cols
        self._col_order = list(df_train.columns)

        if self._numeric_cols:
            numeric_df = df_train[self._numeric_cols].copy()
            means = numeric_df.mean(axis=0, skipna=True).to_numpy()
            numeric_filled = numeric_df.fillna(numeric_df.mean())
            cov = np.cov(numeric_filled.to_numpy().T)
            cov = cov + np.eye(cov.shape[0]) * 1e-6
            self._mean = means
            self._cov = cov

        self._cat_probs = {}
        self._cat_dtypes = {}
        for col in self._categorical_cols:
            # Preserve original dtype for later conversion
            self._cat_dtypes[col] = df_train[col].dtype
            counts = df_train[col].value_counts(dropna=False)
            probs = (counts / counts.sum()).to_dict()
            self._cat_probs[col] = {k: float(v) for k, v in probs.items()}

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Sample synthetic data using the fitted distributions."""
        rng = np.random.default_rng(seed)
        parts: Dict[str, pd.Series] = {}

        if self._numeric_cols:
            if self._mean is None or self._cov is None:
                raise ValueError("Numeric parameters not fit")
            samples = rng.multivariate_normal(self._mean, self._cov, size=n)
            for idx, col in enumerate(self._numeric_cols):
                parts[col] = pd.Series(samples[:, idx])

        for col in self._categorical_cols:
            probs = self._cat_probs.get(col, {})
            categories = list(probs.keys())
            weights = np.array(list(probs.values()), dtype=float)
            weights = weights / weights.sum()
            sampled = rng.choice(categories, size=n, p=weights)
            parts[col] = pd.Series(sampled, dtype=self._cat_dtypes.get(col))

        df = pd.DataFrame(parts)
        for col in self._col_order:
            if col not in df.columns:
                df[col] = np.nan
        return df[self._col_order].reset_index(drop=True)


class TVAELikeGenerator(SynthGenerator):
    """TVAE-like generator using a Gaussian mixture for numeric features."""

    name = "tvae"

    def __init__(self) -> None:
        self._numeric_cols: List[str] = []
        self._categorical_cols: List[str] = []
        self._gmm: GaussianMixture | None = None
        self._cat_probs: Dict[int, Dict[str, Dict]] = {}
        self._fallback_probs: Dict[str, Dict] = {}
        self._cat_dtypes: Dict[str, type] = {}
        self._col_order: List[str] = []

    def fit(self, df_train: pd.DataFrame, schema: Schema, seed: int) -> None:
        """Fit a Gaussian mixture and conditional categorical distributions."""
        numeric_cols = list(schema.numeric)
        categorical_cols = list(schema.categorical)

        if schema.task_type == "regression":
            numeric_cols.append(schema.target)
        else:
            categorical_cols.append(schema.target)

        self._numeric_cols = numeric_cols
        self._categorical_cols = categorical_cols
        self._col_order = list(df_train.columns)

        self._cat_probs = {}
        self._fallback_probs = {}
        self._cat_dtypes = {}

        for col in self._categorical_cols:
            self._cat_dtypes[col] = df_train[col].dtype
            counts = df_train[col].value_counts(dropna=False)
            probs = (counts / counts.sum()).to_dict()
            self._fallback_probs[col] = {k: float(v) for k, v in probs.items()}

        if not self._numeric_cols:
            self._gmm = None
            self._cat_probs = {0: self._fallback_probs}
            return

        numeric_df = df_train[self._numeric_cols].copy()
        numeric_filled = numeric_df.fillna(numeric_df.mean())
        n_components = min(5, max(1, int(np.sqrt(len(numeric_filled)) // 2)))
        n_components = min(n_components, len(numeric_filled))
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=seed,
        )
        gmm.fit(numeric_filled.to_numpy())
        labels = gmm.predict(numeric_filled.to_numpy())
        self._gmm = gmm

        for component in range(n_components):
            component_probs: Dict[str, Dict] = {}
            mask = labels == component
            for col in self._categorical_cols:
                if mask.sum() == 0:
                    component_probs[col] = self._fallback_probs[col]
                else:
                    counts = df_train.loc[mask, col].value_counts(dropna=False)
                    if counts.sum() == 0:
                        component_probs[col] = self._fallback_probs[col]
                    else:
                        probs = (counts / counts.sum()).to_dict()
                        component_probs[col] = {k: float(v) for k, v in probs.items()}
            self._cat_probs[component] = component_probs

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Sample synthetic data using the fitted mixture model."""
        rng = np.random.default_rng(seed)
        parts: Dict[str, pd.Series] = {}

        if self._numeric_cols and self._gmm is None:
            raise ValueError("Numeric parameters not fit")

        if self._gmm is None:
            labels = np.zeros(n, dtype=int)
        else:
            samples, labels = self._gmm.sample(n, random_state=seed)
            for idx, col in enumerate(self._numeric_cols):
                parts[col] = pd.Series(samples[:, idx])

        for col in self._categorical_cols:
            sampled = []
            for label in labels:
                probs = self._cat_probs.get(int(label), self._fallback_probs).get(
                    col, self._fallback_probs.get(col, {})
                )
                categories = list(probs.keys())
                weights = np.array(list(probs.values()), dtype=float)
                weights = weights / weights.sum()
                sampled.append(rng.choice(categories, p=weights))
            parts[col] = pd.Series(sampled, dtype=self._cat_dtypes.get(col))

        df = pd.DataFrame(parts)
        for col in self._col_order:
            if col not in df.columns:
                df[col] = np.nan
        return df[self._col_order].reset_index(drop=True)


@dataclass
class SyntheticDataset:
    """Metadata for a synthetic dataset."""

    name: str
    generator: str
    seed: int
    size_multiplier: float
    df: pd.DataFrame
    path: str


def build_generators() -> List[SynthGenerator]:
    """Build the list of supported generators."""
    return [BootstrapGenerator(), GaussianCopulaGenerator(), TVAELikeGenerator()]


def generate_synthetic_bank(
    df_train: pd.DataFrame,
    schema: Schema,
    seeds: List[int],
    sizes: List[float],
    outdir: str,
) -> List[SyntheticDataset]:
    """Generate and persist a bank of synthetic datasets."""
    datasets: List[SyntheticDataset] = []
    for gen in build_generators():
        for seed in seeds:
            gen.fit(df_train, schema=schema, seed=seed)
            for size in sizes:
                n = max(1, int(len(df_train) * size))
                df_synth = gen.sample(n=n, seed=seed)
                name = f"{gen.name}_seed{seed}_size{size}"
                path = f"{outdir}/{name}.csv"
                df_synth.to_csv(path, index=False)
                datasets.append(
                    SyntheticDataset(
                        name=name,
                        generator=gen.name,
                        seed=seed,
                        size_multiplier=size,
                        df=df_synth,
                        path=path,
                    )
                )
    return datasets
