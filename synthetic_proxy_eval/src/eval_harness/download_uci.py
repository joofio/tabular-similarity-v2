"""Download UCI datasets listed in a YAML config using ucimlrepo."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml
from ucimlrepo import fetch_ucirepo
from ucimlrepo.fetch import DatasetNotFoundError


def _load_config(path: str) -> Dict[str, Any]:
    """Load YAML config containing dataset entries."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_uci_fetch_kwargs(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve kwargs for fetch_ucirepo from a config entry."""
    if "uci_id" in entry and entry["uci_id"] is not None:
        return {"id": int(entry["uci_id"])}

    uci_name = entry.get("uci_name")
    if uci_name:
        return {"name": str(uci_name)}

    dataset_id = entry.get("id")
    dataset_id_str = None
    if dataset_id is None:
        dataset_id_str = None
    elif isinstance(dataset_id, int):
        return {"id": int(dataset_id)}
    else:
        dataset_id_str = str(dataset_id).strip()
        if dataset_id_str.isdigit():
            return {"id": int(dataset_id_str)}

    name = entry.get("name")
    if name:
        return {"name": str(name)}

    if dataset_id_str:
        return {"name": dataset_id_str}

    raise ValueError("Dataset entry must define uci_id, uci_name, name, or id")


def _dataset_to_dataframe(dataset: Any) -> pd.DataFrame:
    """Construct a dataframe from a ucimlrepo dataset object."""
    if hasattr(dataset.data, "dataframe") and dataset.data.dataframe is not None:
        return dataset.data.dataframe.copy()

    parts: List[pd.DataFrame] = []
    if getattr(dataset.data, "features", None) is not None:
        parts.append(dataset.data.features)
    if getattr(dataset.data, "targets", None) is not None:
        parts.append(dataset.data.targets)

    if not parts:
        raise ValueError("No features/targets found in fetched dataset")
    return pd.concat(parts, axis=1)


def _output_path(root: Path, relative_path: str) -> Path:
    """Resolve output path relative to the root directory."""
    rel = Path(relative_path)
    if rel.is_absolute():
        return rel
    return root / rel


def download_datasets(
    config_path: str,
    data_root: str,
    only_ids: Iterable[str] | None,
    force: bool,
) -> List[Path]:
    """Download datasets defined in a config file."""
    config = _load_config(config_path)
    datasets = config.get("datasets", [])
    only_set = set(only_ids or [])

    outputs: List[Path] = []
    root = Path(data_root)
    for entry in datasets:
        dataset_id = entry.get("id", "")
        if only_set and dataset_id not in only_set:
            continue
        if "path" not in entry:
            raise ValueError(f"Dataset '{dataset_id}' missing path in config")

        fetch_kwargs = _resolve_uci_fetch_kwargs(entry)
        try:
            dataset = fetch_ucirepo(**fetch_kwargs)
        except DatasetNotFoundError as exc:
            warnings.warn(
                f"Skipping dataset '{dataset_id}': {exc}",
                RuntimeWarning,
            )
            continue
        df = _dataset_to_dataframe(dataset)

        # Strip whitespace from string columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()

        out_path = _output_path(root, entry["path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not force:
            raise FileExistsError(f"File exists: {out_path}. Use --force to overwrite.")
        df.to_csv(out_path, index=False)
        outputs.append(out_path)

    return outputs


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for dataset download."""
    parser = argparse.ArgumentParser(description="Download UCI datasets via ucimlrepo")
    parser.add_argument(
        "--config",
        default="configs/datasets_uci_health.yaml",
        help="Path to datasets config YAML",
    )
    parser.add_argument(
        "--data-root",
        default=".",
        help="Root directory for dataset paths (default: project root)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional dataset ids to download (from config)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files if they already exist",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    """Run the download command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    download_datasets(args.config, args.data_root, args.only, args.force)


if __name__ == "__main__":
    main()
