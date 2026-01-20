"""Experiment orchestration for proxy metric evaluation."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from eval_harness import reports
from eval_harness.fidelity_metrics import compute_fidelity_metrics
from eval_harness.generators import SyntheticDataset, generate_synthetic_bank
from eval_harness.importance import compute_importance_rankings
from eval_harness.io import (
    Schema,
    load_config,
    load_dataset,
    load_dataset_from_config,
    split_train_test,
)
from eval_harness.preprocess import PreprocessResult, fit_preprocess
from eval_harness.proxy_metrics import compute_proxy_metrics
from eval_harness.stats import (
    boundary_crossing_rate,
    screening_metrics,
    spearman_with_bootstrap,
    stability_std,
)
from eval_harness.utility import (
    aggregate_delta_u,
    compute_delta_u,
    compute_model_metrics,
)


def _select_scoring(config: Dict) -> str | None:
    """Select scoring for permutation importance from config."""
    scoring = config["preprocess"].get("importance_scoring", "auto")
    if scoring == "auto":
        return None
    return scoring


def _evaluate_proxy(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_synth: pd.DataFrame,
    schema: Schema,
    preprocess: PreprocessResult,
    model_names: List[str],
    seeds: List[int],
    ndcg_ks: List[int],
    rbo_ps: List[float],
    jaccard_ks: List[int],
    overlap_ks: List[int],
    scoring: str | None,
    n_repeats: int,
) -> Tuple[List[Dict], Dict[str, Dict[str, float]]]:
    """Compute proxy metrics for a synthetic dataset against the real baseline."""
    base_rankings = compute_importance_rankings(
        df_train,
        df_test,
        preprocess,
        schema,
        model_names,
        seeds,
        scoring=scoring,
        n_repeats=n_repeats,
    )
    synth_rankings = compute_importance_rankings(
        df_synth,
        df_test,
        preprocess,
        schema,
        model_names,
        seeds,
        scoring=scoring,
        n_repeats=n_repeats,
    )

    records: List[Dict] = []
    metric_summary: Dict[str, Dict[str, float]] = {}

    for (model_name, seed), base_rank in base_rankings.items():
        synth_rank = synth_rankings[(model_name, seed)]
        metrics = compute_proxy_metrics(
            base_rank.order,
            synth_rank.order,
            base_rank.ranks,
            synth_rank.ranks,
            ndcg_ks=ndcg_ks,
            rbo_ps=rbo_ps,
            jaccard_ks=jaccard_ks,
            overlap_ks=overlap_ks,
        )
        for metric_name, value in metrics.items():
            records.append(
                {
                    "model": model_name,
                    "model_seed": seed,
                    "metric": metric_name,
                    "value": float(value),
                }
            )

    proxy_long = pd.DataFrame(records)
    for metric in proxy_long["metric"].unique():
        values = proxy_long.loc[proxy_long["metric"] == metric, "value"].to_numpy()
        median = float(np.median(values))
        iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
        metric_summary[metric] = {"median": median, "iqr": iqr}

    return records, metric_summary


def _evaluate_dataset(
    synth: SyntheticDataset,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    schema: Schema,
    preprocess: PreprocessResult,
    config: Dict,
    base_utilities: Dict[Tuple[str, int], Dict[str, float]],
) -> Tuple[Dict, List[Dict], List[Dict], Dict[str, float]]:
    """Evaluate utility, proxy, and fidelity metrics for one synthetic dataset."""
    model_names = config["utility"]["models"]
    seeds = config["generators"]["seeds"]
    eps = config["utility"]["eps"]

    synth_metrics = compute_model_metrics(
        synth.df,
        df_test,
        preprocess,
        schema,
        model_names,
        seeds,
    )

    utility_records: List[Dict] = []
    delta_values: List[float] = []
    for key, rr_metrics in base_utilities.items():
        sr_metrics = synth_metrics[key]
        deltas = compute_delta_u(rr_metrics, sr_metrics, schema.task_type, eps)
        for metric_name, delta in deltas.items():
            utility_records.append(
                {
                    "dataset_id": synth.name,
                    "generator": synth.generator,
                    "seed": synth.seed,
                    "size": synth.size_multiplier,
                    "model": key[0],
                    "model_seed": key[1],
                    "metric": metric_name,
                    "delta_u": float(delta),
                }
            )
            delta_values.append(float(delta))

    delta_median, delta_iqr = aggregate_delta_u(delta_values)

    proxy_records, proxy_summary = _evaluate_proxy(
        df_train,
        df_test,
        synth.df,
        schema,
        preprocess,
        model_names,
        seeds,
        ndcg_ks=config["proxy_metrics"]["ndcg_ks"],
        rbo_ps=config["proxy_metrics"]["rbo_ps"],
        jaccard_ks=config["proxy_metrics"]["jaccard_ks"],
        overlap_ks=config["proxy_metrics"]["overlap_ks"],
        scoring=_select_scoring(config),
        n_repeats=config["preprocess"]["n_repeats_importance"],
    )

    fidelity = compute_fidelity_metrics(
        df_train,
        synth.df,
        schema,
        preprocess,
        seed=config["seed"],
        test_size=config["fidelity"]["propensity"]["test_size"],
        max_iter=config["fidelity"]["propensity"]["max_iter"],
    )

    summary = {
        "dataset_id": synth.name,
        "generator": synth.generator,
        "seed": synth.seed,
        "size": synth.size_multiplier,
        "delta_u_median": delta_median,
        "delta_u_iqr": delta_iqr,
    }

    for metric, stats in proxy_summary.items():
        summary[f"proxy_{metric}_median"] = stats["median"]
        summary[f"proxy_{metric}_iqr"] = stats["iqr"]

    for key, value in fidelity.items():
        summary[f"fidelity_{key}"] = value

    for record in proxy_records:
        record.update(
            {
                "dataset_id": synth.name,
                "generator": synth.generator,
                "seed": synth.seed,
                "size": synth.size_multiplier,
            }
        )

    return summary, utility_records, proxy_records, fidelity


def _apply_degradation(
    df: pd.DataFrame,
    schema: Schema,
    kind: str,
    lam: float,
    seed: int,
) -> pd.DataFrame:
    """Apply a single degradation to a dataset."""
    rng = np.random.default_rng(seed)
    df_out = df.copy()

    if kind == "mean_variance_drift":
        for col in schema.numeric:
            std = df_out[col].std(skipna=True)
            if std == 0 or np.isnan(std):
                continue
            df_out[col] = (df_out[col] - df_out[col].mean()) * (1 + lam) + df_out[col].mean()
            df_out[col] = df_out[col] + lam * std
    elif kind == "gaussian_noise":
        for col in schema.numeric:
            std = df_out[col].std(skipna=True)
            if std == 0 or np.isnan(std):
                continue
            df_out[col] = df_out[col] + rng.normal(0, lam * std, size=len(df_out))
    elif kind == "rare_category_collapse":
        for col in schema.categorical:
            counts = df_out[col].value_counts(normalize=True)
            if counts.empty:
                continue
            rare = counts[counts < 0.1].index
            mode = counts.idxmax()
            mask = df_out[col].isin(rare)
            replace = rng.random(len(df_out)) < lam
            df_out.loc[mask & replace, col] = mode
    elif kind == "correlation_attenuation":
        for col in schema.numeric:
            shuffled = df_out[col].sample(frac=1.0, random_state=seed).to_numpy()
            df_out[col] = (1 - lam) * df_out[col].to_numpy() + lam * shuffled
    elif kind == "synthetic_missingness":
        for col in schema.features:
            mask = rng.random(len(df_out)) < lam
            df_out.loc[mask, col] = np.nan
    else:
        raise ValueError(f"Unknown degradation kind: {kind}")

    return df_out


def _apply_invariance_pair(
    df_real_train: pd.DataFrame,
    df_real_test: pd.DataFrame,
    df_synth: pd.DataFrame,
    schema: Schema,
    kind: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply an invariance transform to real train/test and synthetic datasets."""
    rng = np.random.default_rng(seed)
    real_train = df_real_train.copy()
    real_test = df_real_test.copy()
    synth = df_synth.copy()

    if kind == "column_reorder":
        columns = schema.features
        shuffled = columns.copy()
        rng.shuffle(shuffled)
        reorder = shuffled + [schema.target]
        return real_train[reorder], real_test[reorder], synth[reorder]

    if kind == "categorical_bijection":
        for col in schema.categorical:
            categories = sorted(
                pd.concat([real_train[col], real_test[col], synth[col]]).astype(str).unique()
            )
            permuted = categories.copy()
            rng.shuffle(permuted)
            mapping = dict(zip(categories, permuted))
            real_train[col] = real_train[col].astype(str).map(mapping)
            real_test[col] = real_test[col].astype(str).map(mapping)
            synth[col] = synth[col].astype(str).map(mapping)
        return real_train, real_test, synth

    if kind == "numeric_affine":
        scale = 1.0 + rng.uniform(0.1, 0.2)
        shift = rng.uniform(-1.0, 1.0)
        for col in schema.numeric:
            real_train[col] = real_train[col] * scale + shift
            real_test[col] = real_test[col] * scale + shift
            synth[col] = synth[col] * scale + shift
        return real_train, real_test, synth

    raise ValueError(f"Unknown invariance kind: {kind}")


def run_pipeline(
    config_path: str,
    data_path: str,
    target: str,
    outdir: str,
) -> Dict[str, str]:
    """Run the full evaluation pipeline end-to-end."""
    config = load_config(config_path)
    df, schema = load_dataset(data_path, target=target, task_type=config["utility"]["task_type"])

    os.makedirs(outdir, exist_ok=True)
    synth_dir = os.path.join(outdir, config["generators"]["output_dir"])
    os.makedirs(synth_dir, exist_ok=True)

    df_train, df_test = split_train_test(
        df,
        schema,
        seed=config["seed"],
        test_size=config["split"]["test_size"],
        stratify=config["split"]["stratify"],
    )

    scaler_type = config.get("preprocess", {}).get("scaler", "standard")
    preprocess = fit_preprocess(df_train, schema, scaler_type=scaler_type)

    synth_bank = generate_synthetic_bank(
        df_train,
        schema,
        seeds=config["generators"]["seeds"],
        sizes=config["generators"]["sizes"],
        outdir=synth_dir,
    )

    base_utilities = compute_model_metrics(
        df_train,
        df_test,
        preprocess,
        schema,
        model_names=config["utility"]["models"],
        seeds=config["generators"]["seeds"],
    )

    summary_rows: List[Dict] = []
    utility_rows: List[Dict] = []
    proxy_rows: List[Dict] = []
    fidelity_rows: List[Dict] = []

    for synth in synth_bank:
        summary, utility_records, proxy_records, fidelity = _evaluate_dataset(
            synth,
            df_train,
            df_test,
            schema,
            preprocess,
            config,
            base_utilities,
        )
        summary_rows.append(summary)
        utility_rows.extend(utility_records)
        proxy_rows.extend(proxy_records)
        fidelity_rows.append({"dataset_id": synth.name, **fidelity})

    summary_df = pd.DataFrame(summary_rows)
    utility_long = pd.DataFrame(utility_rows)
    proxy_long = pd.DataFrame(proxy_rows)
    fidelity_df = pd.DataFrame(fidelity_rows)

    stats_rows: List[Dict] = []
    thresholds: Dict[str, float] = {}
    for metric_col in [c for c in summary_df.columns if c.startswith("proxy_") and c.endswith("_median")]:
        raw_metric = metric_col[len("proxy_") : -len("_median")]
        m_values = summary_df[metric_col].to_numpy()
        delta_u = summary_df["delta_u_median"].to_numpy()
        spearman_stats = spearman_with_bootstrap(
            m_values,
            delta_u,
            n_resamples=config["stats"]["bootstrap"]["n_resamples"],
            seed=config["stats"]["bootstrap"]["seed"],
        )
        screening = screening_metrics(
            m_values,
            delta_u,
            acceptable_threshold=config["stats"]["acceptable_delta_u"],
            precision_target=config["stats"]["precision_target"],
        )
        thresholds[raw_metric] = screening["threshold"]
        stats_rows.append(
            {
                "metric": raw_metric,
                **spearman_stats,
                **screening,
            }
        )

    stats_df = pd.DataFrame(stats_rows)

    if not proxy_long.empty:
        stability_df = stability_std(proxy_long)
        boundary_df = boundary_crossing_rate(proxy_long, thresholds)
        stats_df = stats_df.merge(boundary_df, on="metric", how="left")
        stability_df.to_csv(os.path.join(outdir, "stability.csv"), index=False)

    outputs = reports.write_reports(
        summary_df,
        utility_long,
        proxy_long,
        fidelity_df,
        stats_df,
        outdir,
        save_plots=config["reporting"]["save_plots"],
    )

    if config["reporting"]["save_plots"] and not summary_df.empty:
        proxy_cols = [c for c in summary_df.columns if c.startswith("proxy_") and c.endswith("_median")]
        if proxy_cols:
            metric_col = proxy_cols[0]
            labels = summary_df["delta_u_median"].to_numpy() <= config["stats"]["acceptable_delta_u"]
            scores = summary_df[metric_col].to_numpy()
            if len(np.unique(labels)) > 1:
                fpr, tpr, _ = roc_curve(labels, scores)
                precision, recall, _ = precision_recall_curve(labels, scores)
                plot_dir = os.path.join(outdir, "plots")
                reports.ensure_dir(plot_dir)
                reports.plot_roc_curve(fpr, tpr, os.path.join(plot_dir, "roc_curve.png"))
                reports.plot_pr_curve(precision, recall, os.path.join(plot_dir, "pr_curve.png"))

    _run_sensitivity_tests(
        df_train,
        df_test,
        schema,
        preprocess,
        config,
        base_utilities,
        outdir,
    )
    _run_invariance_tests(
        df_train,
        df_test,
        schema,
        preprocess,
        config,
        synth_bank[0],
        outdir,
    )

    return outputs


def run_pipeline_from_config(
    config: Dict,
    dataset_id: str,
    task_id: Optional[str] = None,
    outdir: str = "output",
    data_root: str = ".",
) -> Dict[str, str]:
    """Run the full evaluation pipeline using dataset config.

    Args:
        config: Loaded configuration dictionary (with datasets merged)
        dataset_id: ID of dataset to run (from datasets config)
        task_id: Optional task ID within dataset (uses first if not specified)
        outdir: Output directory for results
        data_root: Root directory for resolving relative dataset paths
    """
    # Load dataset using config
    df, schema, task_def, entry = load_dataset_from_config(
        config, dataset_id, task_id=task_id, data_root=data_root
    )

    # Get task-specific metrics if defined
    task_metrics = task_def.get("metrics")

    # Get split settings (per-dataset overrides global)
    dataset_split = entry.get("split", {})
    test_size = dataset_split.get("test_size", config["split"]["test_size"])
    stratify = dataset_split.get("stratify", config["split"]["stratify"])

    os.makedirs(outdir, exist_ok=True)
    synth_dir = os.path.join(outdir, config["generators"]["output_dir"])
    os.makedirs(synth_dir, exist_ok=True)

    df_train, df_test = split_train_test(
        df,
        schema,
        seed=config["seed"],
        test_size=test_size,
        stratify=stratify,
    )

    scaler_type = config.get("preprocess", {}).get("scaler", "standard")
    preprocess = fit_preprocess(df_train, schema, scaler_type=scaler_type)

    synth_bank = generate_synthetic_bank(
        df_train,
        schema,
        seeds=config["generators"]["seeds"],
        sizes=config["generators"]["sizes"],
        outdir=synth_dir,
    )

    base_utilities = compute_model_metrics(
        df_train,
        df_test,
        preprocess,
        schema,
        model_names=config["utility"]["models"],
        seeds=config["generators"]["seeds"],
    )

    summary_rows: List[Dict] = []
    utility_rows: List[Dict] = []
    proxy_rows: List[Dict] = []
    fidelity_rows: List[Dict] = []

    for synth in synth_bank:
        summary, utility_records, proxy_records, fidelity = _evaluate_dataset(
            synth,
            df_train,
            df_test,
            schema,
            preprocess,
            config,
            base_utilities,
        )
        summary_rows.append(summary)
        utility_rows.extend(utility_records)
        proxy_rows.extend(proxy_records)
        fidelity_rows.append({"dataset_id": synth.name, **fidelity})

    summary_df = pd.DataFrame(summary_rows)
    utility_long = pd.DataFrame(utility_rows)
    proxy_long = pd.DataFrame(proxy_rows)
    fidelity_df = pd.DataFrame(fidelity_rows)

    stats_rows: List[Dict] = []
    thresholds: Dict[str, float] = {}
    for metric_col in [c for c in summary_df.columns if c.startswith("proxy_") and c.endswith("_median")]:
        raw_metric = metric_col[len("proxy_") : -len("_median")]
        m_values = summary_df[metric_col].to_numpy()
        delta_u = summary_df["delta_u_median"].to_numpy()
        spearman_stats = spearman_with_bootstrap(
            m_values,
            delta_u,
            n_resamples=config["stats"]["bootstrap"]["n_resamples"],
            seed=config["stats"]["bootstrap"]["seed"],
        )
        screening = screening_metrics(
            m_values,
            delta_u,
            acceptable_threshold=config["stats"]["acceptable_delta_u"],
            precision_target=config["stats"]["precision_target"],
        )
        thresholds[raw_metric] = screening["threshold"]
        stats_rows.append(
            {
                "metric": raw_metric,
                **spearman_stats,
                **screening,
            }
        )

    stats_df = pd.DataFrame(stats_rows)

    if not proxy_long.empty:
        stability_df = stability_std(proxy_long)
        boundary_df = boundary_crossing_rate(proxy_long, thresholds)
        stats_df = stats_df.merge(boundary_df, on="metric", how="left")
        stability_df.to_csv(os.path.join(outdir, "stability.csv"), index=False)

    outputs = reports.write_reports(
        summary_df,
        utility_long,
        proxy_long,
        fidelity_df,
        stats_df,
        outdir,
        save_plots=config["reporting"]["save_plots"],
    )

    if config["reporting"]["save_plots"] and not summary_df.empty:
        proxy_cols = [c for c in summary_df.columns if c.startswith("proxy_") and c.endswith("_median")]
        if proxy_cols:
            metric_col = proxy_cols[0]
            labels = summary_df["delta_u_median"].to_numpy() <= config["stats"]["acceptable_delta_u"]
            scores = summary_df[metric_col].to_numpy()
            if len(np.unique(labels)) > 1:
                fpr, tpr, _ = roc_curve(labels, scores)
                precision, recall, _ = precision_recall_curve(labels, scores)
                plot_dir = os.path.join(outdir, "plots")
                reports.ensure_dir(plot_dir)
                reports.plot_roc_curve(fpr, tpr, os.path.join(plot_dir, "roc_curve.png"))
                reports.plot_pr_curve(precision, recall, os.path.join(plot_dir, "pr_curve.png"))

    _run_sensitivity_tests(
        df_train,
        df_test,
        schema,
        preprocess,
        config,
        base_utilities,
        outdir,
    )
    _run_invariance_tests(
        df_train,
        df_test,
        schema,
        preprocess,
        config,
        synth_bank[0],
        outdir,
    )

    return outputs


def _run_sensitivity_tests(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    schema: Schema,
    preprocess: PreprocessResult,
    config: Dict,
    base_utilities: Dict[Tuple[str, int], Dict[str, float]],
    outdir: str,
) -> None:
    """Run sensitivity tests with degradations."""
    lambdas = config["stats"]["sensitivity_lambdas"]
    kinds = [
        "mean_variance_drift",
        "gaussian_noise",
        "rare_category_collapse",
        "correlation_attenuation",
        "synthetic_missingness",
    ]

    rows: List[Dict] = []
    for kind in kinds:
        for lam in lambdas:
            degraded = _apply_degradation(df_train, schema, kind, lam, seed=config["seed"])
            synth = SyntheticDataset(
                name=f"degraded_{kind}_{lam}",
                generator="degradation",
                seed=config["seed"],
                size_multiplier=1.0,
                df=degraded,
                path="",
            )
            summary, _, _, _ = _evaluate_dataset(
                synth,
                df_train,
                df_test,
                schema,
                preprocess,
                config,
                base_utilities,
            )
            rows.append(
                {
                    "kind": kind,
                    "lambda": lam,
                    "metric": "delta_u_median",
                    "value": summary["delta_u_median"],
                }
            )
            for key, value in summary.items():
                if key.startswith("proxy_") and key.endswith("_median"):
                    raw_metric = key[len("proxy_") : -len("_median")]
                    rows.append(
                        {
                            "kind": kind,
                            "lambda": lam,
                            "metric": raw_metric,
                            "value": value,
                        }
                    )
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "sensitivity.csv")
    df_out.to_csv(csv_path, index=False)
    if config["reporting"]["save_plots"] and not df_out.empty:
        plot_dir = os.path.join(outdir, "plots")
        reports.ensure_dir(plot_dir)
        reports.plot_degradation(df_out, os.path.join(plot_dir, "degradation.png"))


def _run_invariance_tests(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    schema: Schema,
    preprocess: PreprocessResult,
    config: Dict,
    synth: SyntheticDataset,
    outdir: str,
) -> None:
    """Run invariance tests for proxy metrics."""
    base_records, base_summary = _evaluate_proxy(
        df_train,
        df_test,
        synth.df,
        schema,
        preprocess,
        config["utility"]["models"],
        config["generators"]["seeds"],
        ndcg_ks=config["proxy_metrics"]["ndcg_ks"],
        rbo_ps=config["proxy_metrics"]["rbo_ps"],
        jaccard_ks=config["proxy_metrics"]["jaccard_ks"],
        overlap_ks=config["proxy_metrics"]["overlap_ks"],
        scoring=_select_scoring(config),
        n_repeats=config["preprocess"]["n_repeats_importance"],
    )
    del base_records

    rows: List[Dict] = []
    for kind in ["column_reorder", "categorical_bijection", "numeric_affine"]:
        r_train, r_test, s_df = _apply_invariance_pair(
            df_train,
            df_test,
            synth.df,
            schema,
            kind,
            seed=config["seed"],
        )
        scaler_type = config.get("preprocess", {}).get("scaler", "standard")
        preprocess_inv = fit_preprocess(r_train, schema, scaler_type=scaler_type)
        _, transformed_summary = _evaluate_proxy(
            r_train,
            r_test,
            s_df,
            schema,
            preprocess_inv,
            config["utility"]["models"],
            config["generators"]["seeds"],
            ndcg_ks=config["proxy_metrics"]["ndcg_ks"],
            rbo_ps=config["proxy_metrics"]["rbo_ps"],
            jaccard_ks=config["proxy_metrics"]["jaccard_ks"],
            overlap_ks=config["proxy_metrics"]["overlap_ks"],
            scoring=_select_scoring(config),
            n_repeats=config["preprocess"]["n_repeats_importance"],
        )
        for metric, stats in base_summary.items():
            delta = abs(stats["median"] - transformed_summary[metric]["median"])
            rows.append(
                {
                    "transform": kind,
                    "metric": metric,
                    "delta": delta,
                }
            )

    inv_df = pd.DataFrame(rows)
    inv_df.to_csv(os.path.join(outdir, "invariance.csv"), index=False)
