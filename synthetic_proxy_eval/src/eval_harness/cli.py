"""CLI entrypoint for evaluation harness."""

from __future__ import annotations

import argparse
import os
import sys

from eval_harness.experiments import run_pipeline, run_pipeline_from_config
from eval_harness.io import load_config


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Synthetic proxy metric evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single dataset by ID (from datasets config):
  python -m eval_harness.cli --config configs/base.yaml --dataset 17 --outdir runs/exp1

  # Run all datasets from config:
  python -m eval_harness.cli --config configs/base.yaml --all --outdir runs/exp1

  # Legacy mode (explicit path/target):
  python -m eval_harness.cli --config configs/base.yaml --data data/ckd/ckd.csv --target class --outdir runs/exp1
""",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--outdir", required=True, help="Output directory")

    # Dataset selection (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--dataset",
        help="Dataset ID from config (e.g., '17' or 'Breast Cancer Wisconsin')",
    )
    data_group.add_argument(
        "--all",
        action="store_true",
        help="Run all datasets defined in config",
    )
    data_group.add_argument(
        "--data",
        help="Path to CSV data (legacy mode, requires --target)",
    )

    parser.add_argument(
        "--target",
        help="Target column name (required with --data, ignored with --dataset)",
    )
    parser.add_argument(
        "--task",
        help="Task ID within dataset (optional, uses first task if not specified)",
    )
    parser.add_argument(
        "--data-root",
        default=".",
        help="Root directory for resolving relative dataset paths (default: current dir)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI main entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Legacy mode: --data and --target
    if args.data:
        if not args.target:
            parser.error("--target is required when using --data")
        run_pipeline(args.config, args.data, args.target, args.outdir)
        return

    # Load config for dataset-based modes
    config = load_config(args.config)
    datasets = config.get("datasets", [])

    if not datasets:
        print("Error: No datasets found in config", file=sys.stderr)
        sys.exit(1)

    # Determine which datasets to run
    if args.all:
        dataset_ids = [str(entry.get("id", "")) for entry in datasets]
    else:
        dataset_ids = [args.dataset]

    # Run each dataset
    for dataset_id in dataset_ids:
        print(f"\n{'='*60}")
        print(f"Running dataset: {dataset_id}")
        print(f"{'='*60}\n")

        # Create per-dataset output directory
        outdir = os.path.join(args.outdir, str(dataset_id))

        try:
            run_pipeline_from_config(
                config=config,
                dataset_id=dataset_id,
                task_id=args.task,
                outdir=outdir,
                data_root=args.data_root,
            )
            print(f"\nCompleted dataset {dataset_id} -> {outdir}")
        except Exception as e:
            print(f"\nError processing dataset {dataset_id}: {e}", file=sys.stderr)
            if not args.all:
                sys.exit(1)


if __name__ == "__main__":
    main()
