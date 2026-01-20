# Synthetic Proxy Eval

Evaluation harness for validating proxy metrics on synthetic healthcare tabular datasets.

## Example run

python -m eval_harness.cli \
  --config configs/base.yaml \
  --data path/to/data.csv \
  --target target_column \
  --outdir runs/experiment1
