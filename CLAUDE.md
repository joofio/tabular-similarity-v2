# CLAUDE.md - AI Assistant Guide

**Last Updated:** 2026-01-20
**Repository:** tabular-similarity-v2
**Project:** Synthetic Proxy Eval

## Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Structure](#codebase-structure)
3. [Architecture & Design Principles](#architecture--design-principles)
4. [Development Workflow](#development-workflow)
5. [Key Conventions & Patterns](#key-conventions--patterns)
6. [Testing Guidelines](#testing-guidelines)
7. [Configuration Management](#configuration-management)
8. [Common Tasks](#common-tasks)
9. [Git Workflow](#git-workflow)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

### Purpose
Synthetic Proxy Eval is a Python evaluation harness that validates proxy metrics used to assess the quality of synthetic healthcare tabular datasets. The system provides a complete pipeline to:

- Load and preprocess UCI healthcare datasets with mixed numeric/categorical features
- Generate synthetic datasets using multiple methods (Bootstrap, Gaussian Copula, TVAE-like)
- Compute utility metrics (ΔU - performance gap between models trained on synthetic vs. real data)
- Calculate proxy metrics that correlate with utility gap (RBO, Kendall tau, Jaccard@k, etc.)
- Assess fidelity (how well synthetic data preserves real data distributions)
- Perform sensitivity and invariance testing
- Generate statistical analysis and visualization reports

### Key Use Cases
1. Validate whether proxy metrics can reliably screen synthetic datasets without expensive utility evaluations
2. Compare different synthetic data generation methods
3. Understand which proxy metrics best correlate with actual utility gaps
4. Detect and measure synthetic data quality across multiple healthcare datasets

### Technology Stack
- **Language:** Python 3.9+
- **Core Dependencies:** numpy, pandas, scipy, scikit-learn, pyyaml, joblib, matplotlib, ucimlrepo
- **Testing:** pytest
- **Build System:** setuptools (pyproject.toml)
- **Package Name:** synthetic-proxy-eval v0.1.0

---

## Codebase Structure

```
/home/user/tabular-similarity-v2/
├── .gitignore                         # Python standard + data/ and runs/
├── agents.md                          # Agent workflow documentation (legacy)
└── synthetic_proxy_eval/              # Main project directory
    ├── pyproject.toml                 # Project configuration
    ├── pytest.ini                     # Test configuration
    ├── README.md                      # Basic project README
    ├── agents.md                      # Agent workflow documentation
    ├── plot_top_metrics.py            # Utility for plotting metric results
    ├── configs/                       # YAML configuration files
    │   ├── base.yaml                  # Base configuration template
    │   └── datasets_uci_health.yaml   # UCI healthcare dataset definitions
    ├── src/eval_harness/              # Main Python package (15 modules)
    │   ├── __init__.py
    │   ├── cli.py                     # CLI entrypoint (123 lines)
    │   ├── io.py                      # Data loading & schema (269 lines)
    │   ├── preprocess.py              # Feature preprocessing (123 lines)
    │   ├── generators.py              # Synthetic generators (276 lines)
    │   ├── tasks.py                   # Task definitions & metrics (40 lines)
    │   ├── models.py                  # Model factory (43 lines)
    │   ├── importance.py              # Feature importance (80 lines)
    │   ├── proxy_metrics.py           # Ranking metrics (181 lines)
    │   ├── utility.py                 # Utility gap ΔU (149 lines)
    │   ├── fidelity_metrics.py        # Distribution fidelity (115 lines)
    │   ├── experiments.py             # Pipeline orchestration (760 lines - LARGEST)
    │   ├── stats.py                   # Statistical validation (121 lines)
    │   ├── reports.py                 # Report generation (221 lines)
    │   └── download_uci.py            # UCI dataset downloader (156 lines)
    └── tests/                         # Test suite (13 test modules)
        ├── conftest.py                # Pytest fixtures
        ├── test_cli_integration.py
        ├── test_determinism.py
        ├── test_experiments_*.py
        ├── test_fidelity_metrics.py
        ├── test_generators.py
        ├── test_golden.py
        ├── test_importance.py
        ├── test_io.py
        ├── test_preprocess.py
        ├── test_proxy_metrics.py
        └── test_utility.py
```

### Module Responsibilities

| Module | Purpose | Key Functions/Classes |
|--------|---------|----------------------|
| `cli.py` | Command-line interface | `build_parser()`, `main()` |
| `io.py` | Data loading & schema management | `load_dataset()`, `load_config()`, `infer_schema()` |
| `preprocess.py` | Feature preprocessing (NO LEAKAGE) | `fit_preprocess()`, `transform_features()` |
| `generators.py` | Synthetic data generation | `BootstrapGenerator`, `GaussianCopulaGenerator`, `TVAELikeGenerator` |
| `tasks.py` | Task type definitions | `metric_directions()`, `metrics_for_task()` |
| `models.py` | Model factory | `get_models()` |
| `importance.py` | Feature importance ranking | `compute_importance_rankings()` |
| `proxy_metrics.py` | Ranking comparison metrics | NDCG, RBO, Kendall tau, Jaccard@k, Overlap@k |
| `utility.py` | Utility gap computation | `compute_delta_u()` |
| `fidelity_metrics.py` | Distribution fidelity | Wasserstein, Jensen-Shannon, correlation distance, propensity AUC |
| `experiments.py` | Pipeline orchestration | `run_pipeline()`, sensitivity/invariance testing |
| `stats.py` | Statistical analysis | Spearman correlation, bootstrap CI, ROC/PR curves |
| `reports.py` | Output generation | Parquet/CSV saving, plotting functions |
| `download_uci.py` | UCI dataset fetching | `fetch_datasets_from_config()` |

---

## Architecture & Design Principles

### Core Principles

1. **NO DATA LEAKAGE**: Preprocessing must ONLY fit on training data, then transform test data
2. **DETERMINISM**: All randomness uses explicit seeds via `np.random.default_rng(seed)`
3. **MODULARITY**: Clear separation between IO, preprocessing, generation, metrics, statistics, reporting
4. **TYPE SAFETY**: Full use of type hints with `from __future__ import annotations`
5. **FEATURE GROUPING**: Track which expanded features (e.g., one-hot encoded) map to original columns

### Pipeline Flow

```
Load Config (YAML)
    ↓
Load Dataset (CSV/UCI)
    ↓
Apply Filters & Derived Targets
    ↓
Infer/Override Schema
    ↓
Train/Test Split (80/20, stratified)
    ↓
Fit Preprocessing on Train Data ONLY
    ↓
Compute Base Importance Rankings & Utility (Real vs Real)
    ↓
Generate Synthetic Bank (multiple sizes × seeds × generators)
    ↓
For Each Synthetic Dataset:
    ├─ Compute Model Metrics (Train Synth → Test Real)
    ├─ Compute Utility Gap (ΔU)
    ├─ Compute Importance Rankings
    ├─ Compute Proxy Metrics (RBO, Kendall, Jaccard, etc.)
    └─ Compute Fidelity Metrics (Wasserstein, JSD, Corr, Propensity AUC)
    ↓
Statistical Analysis:
    ├─ Spearman Correlation with Bootstrap CI
    ├─ Screening Metrics (ROC-AUC, PR-AUC)
    ├─ Precision at Target Threshold
    ├─ Metric Stability (std)
    └─ Boundary Crossing Rate
    ↓
Sensitivity Testing: Inject perturbations (noise, drift, rare collapse)
    ↓
Invariance Testing: Column reordering, categorical bijections, numeric transforms
    ↓
Generate Reports:
    ├─ results.parquet (all metric values)
    ├─ summary.csv (aggregated by synthetic dataset)
    ├─ stats.csv (correlation analysis)
    └─ Visualization Plots (scatter, ROC, PR, degradation)
```

### Critical Design Constraints

#### 1. No Leakage Pattern
```python
# CORRECT: Fit on train, transform both
transformer = fit_preprocess(df_train, schema, scaler_type)
X_train = transform_features(df_train, transformer)
X_test = transform_features(df_test, transformer)

# INCORRECT: Fitting on test data or combined data
transformer = fit_preprocess(pd.concat([df_train, df_test]), schema, scaler_type)  # ❌
```

#### 2. Feature Grouping Pattern
```python
# After one-hot encoding: "sex__M", "sex__F" both map to group "sex"
feature_names: ["age", "bmi", "sex__M", "sex__F"]
feature_group: ["age", "bmi", "sex", "sex"]

# Importance is computed per GROUP, not per expanded feature
```

#### 3. Determinism Pattern
```python
# CORRECT: Explicit seed
rng = np.random.default_rng(seed)
model.fit(X_train, y_train, random_state=seed)

# INCORRECT: Implicit randomness
np.random.shuffle(data)  # ❌ No seed control
```

#### 4. Metric Normalization
```python
# All similarity metrics clamped to [0, 1]
def _clamp_unit(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
```

---

## Development Workflow

### Original Agent-Based Development

This project was developed using a three-agent workflow (see `agents.md`):

1. **Architect Agent**: Design module structure, define interfaces, enforce no-leakage
2. **Coding Agent**: Implement all modules, ensure determinism and full functionality
3. **QA Agent**: Verify correctness, test for leakage, validate full pipeline

**Key Validation Checkpoints:**
- Architecture conforms EXACTLY to spec
- No data leakage in preprocessing
- Correct grouping of one-hot features
- Correct ΔU normalization
- Correct RBO implementation
- All modules import successfully
- CLI runs end-to-end on a dataset

### Current Development Workflow

When modifying this codebase:

1. **Read Before Modify**: Always read files before editing them
2. **Maintain Determinism**: Use explicit seeds for all randomness
3. **Test First**: Run relevant tests before and after changes
4. **No Breaking Changes**: Maintain backward compatibility with existing configs
5. **Document Changes**: Update docstrings and comments
6. **Check Leakage**: Verify preprocessing still fits only on train data
7. **Run Integration Test**: Execute `test_cli_integration.py` after significant changes

---

## Key Conventions & Patterns

### Naming Conventions

| Pattern | Usage | Example |
|---------|-------|---------|
| `df_*` | DataFrame variables | `df_train`, `df_test`, `df_synth` |
| `X_*`, `y_*` | Model features and targets | `X_train`, `y_train` |
| `*_scores`, `*_metrics` | Collections of results | `proxy_scores`, `fidelity_metrics` |
| `config`, `schema` | Key configuration objects | `config["preprocess"]` |
| `seed`, `rng` | Random number generation | `rng = np.random.default_rng(seed)` |
| `_function()` | Internal/private functions | `_clamp_unit()`, `_aligned_rank_arrays()` |

### Code Style Patterns

#### 1. Type Hints (Always Use)
```python
from __future__ import annotations
from typing import Dict, List, Tuple

def function(x: int, y: str) -> Dict[str, float]:
    return {"result": float(x)}
```

#### 2. Dataclass Usage
```python
from dataclasses import dataclass

@dataclass
class Schema:
    target: str
    task_type: str
    numeric: List[str]
    categorical: List[str]
```

#### 3. Pipeline Composition
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
])
```

#### 4. Error Handling for Edge Cases
```python
if np.isnan(value):
    return 0.0
if not columns:
    return 0.0
if len(rank_real) == 0 or len(rank_synth) == 0:
    return 0.0
```

#### 5. Module Exports
```python
# In __init__.py
__all__ = ["cli", "io", "preprocess", "generators", ...]
```

### Critical Patterns to Follow

#### Pattern: OneHotEncoder with Unknown Categories
```python
OneHotEncoder(
    drop="first",          # Avoid multicollinearity
    sparse_output=False,   # Return dense arrays
    handle_unknown="ignore"  # CRITICAL: Handle new categories in test data
)
```

#### Pattern: Feature Group Tracking
```python
# Track which expanded features belong to which original column
feature_names_out = transformer.get_feature_names_out()
feature_group = []
for name in feature_names_out:
    # Extract original column name (before "__" separator)
    original_col = name.split("__")[0] if "__" in name else name
    feature_group.append(original_col)
```

#### Pattern: Stratified Splitting
```python
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    stratify=df[target] if stratify else None,
    random_state=seed
)
```

---

## Testing Guidelines

### Test Structure

**Framework:** pytest with custom markers

**Configuration** (`pytest.ini`):
```ini
[pytest]
addopts = -ra
pythonpath = src
markers = integration
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_generators.py

# Run tests matching pattern
pytest -k "test_determinism"

# Run only integration tests
pytest -m integration

# Verbose output
pytest -v

# Show local variables on failure
pytest -l
```

### Test Categories

| Category | Purpose | Key Tests |
|----------|---------|-----------|
| **Unit Tests** | Test individual functions | All `test_*.py` except integration |
| **Integration Tests** | Test full pipeline | `test_cli_integration.py` |
| **Determinism Tests** | Verify reproducibility | `test_determinism.py` |
| **Golden Tests** | Snapshot testing | `test_golden.py` |
| **Sensitivity Tests** | Metric robustness | `test_experiments_sensitivity.py` |
| **Invariance Tests** | Transformation stability | `test_experiments_invariance.py` |

### Key Test Fixtures (conftest.py)

```python
@pytest.fixture
def toy_df():
    """200-row synthetic dataset with age, BMI, sex, smoker, Class"""

@pytest.fixture
def toy_schema(toy_df):
    """Schema inferred from toy_df with numeric/categorical split"""

@pytest.fixture
def test_config():
    """Minimal config: 2 seeds, 1 size, 2 models, 50 bootstrap resamples"""
```

### Testing Best Practices

1. **Always Use Fixtures**: Reuse `toy_df`, `toy_schema`, `test_config` for consistency
2. **Test Identity**: `metric(X, X) should be 1.0 or 0.0 depending on similarity/distance`
3. **Test Determinism**: Same seed → same output
4. **Test Bounds**: Metrics should stay within [0, 1] or expected ranges
5. **Test No Leakage**: Verify preprocessing fits only on train data
6. **Test Edge Cases**: Empty data, single column, all same values

### Adding New Tests

When adding new functionality:

1. Create test file: `tests/test_<module_name>.py`
2. Import fixtures from `conftest.py`
3. Test determinism first
4. Test edge cases
5. Add integration test if CLI behavior changes

---

## Configuration Management

### Configuration Files

#### base.yaml Structure
```yaml
include_datasets: "datasets_uci_health.yaml"  # External dataset definitions

seed: 42                                      # Global random seed
split:
  test_size: 0.2                             # Train/test split ratio
  stratify: true                             # Stratified split by target

preprocess:
  scaler: "standard"                         # "standard" | "minmax" | "none"
  n_repeats_importance: 5                    # Permutation importance repeats
  importance_scoring: "auto"                 # Scoring metric for importance

generators:
  seeds: [0, 1, 2]                          # Random seeds for synthetic generation
  sizes: [0.5, 1.0]                         # Dataset size multipliers
  output_dir: "synthetic"                   # Output directory name

utility:
  models: ["logreg", "rf", "gb", "knn", "ridge", "gbr"]
  eps: 1.0e-8                               # Epsilon for numerical stability
  task_type: "auto"                         # "auto" | "classification" | "regression"

proxy_metrics:
  ndcg_ks: [10, 20, 50]                    # Top-k values for NDCG
  rbo_ps: [0.9, 0.95]                      # Bias parameters for RBO
  overlap_ks: [10, 20, 50]                 # Top-k for Overlap@k
  jaccard_ks: [10, 20, 50]                 # Top-k for Jaccard@k

fidelity:
  propensity:
    test_size: 0.3                         # Test size for propensity classifier
    max_iter: 500                          # Max iterations for LogReg

stats:
  bootstrap:
    n_resamples: 200                       # Bootstrap resamples
    seed: 123                              # Bootstrap seed
  acceptable_delta_u: 0.02                 # Threshold for acceptable ΔU
  precision_target: 0.95                   # Target precision for screening
  invariance_epsilon: 0.01                 # Tolerance for invariance tests
  sensitivity_lambdas: [0.0, 0.25, 0.5, 0.75, 1.0]  # Degradation levels

reporting:
  save_plots: true                         # Generate visualizations
  figure_dpi: 150                          # Plot resolution
```

#### datasets_uci_health.yaml Structure
```yaml
datasets:
  - id: 296                                # Unique dataset ID
    name: "Diabetes 130-US hospitals"
    source: "UCI"
    path: "data/diabetic.csv"
    task_defs:
      - id: "readmitted"
        target: "readmitted"
        type: "binary_classification"
        label_map:
          "<30": "YES"
          ">30": "YES"
          "NO": "NO"
        metrics: ["auroc", "auprc", "f1_macro", "balanced_accuracy"]
    schema:
      categorical: ["race", "gender", "age", ...]
      numeric: ["time_in_hospital", "num_lab_procedures", ...]
    filters:
      drop_columns: ["encounter_id", "patient_nbr"]

defaults:                                  # Global fallback values
  split:
    test_size: 0.2
    stratify: true
  proxy_metrics:
    required: ["kendall_tau", "rbo_0.9", ...]
    ndcg_ks: [10, 20, 50]
  models:
    classification: ["logreg", "rf", "gb", "knn"]
    regression: ["ridge", "rfr", "gbr", "knnr"]
```

### Modifying Configurations

#### Adding a New Dataset
```yaml
# In datasets_uci_health.yaml
datasets:
  - id: 999                                # Choose unique ID
    name: "My Custom Dataset"
    source: "UCI"                          # Or "local"
    path: "data/my_dataset.csv"           # Relative to data_root
    task_defs:
      - id: "my_task"
        target: "outcome"
        type: "binary_classification"     # Or "regression" | "multiclass_classification"
        metrics: ["auroc", "auprc"]
    schema:
      categorical: ["category_col"]
      numeric: ["numeric_col"]
```

#### Adding a New Model
```python
# In models.py
def get_models(task_type: str, seed: int) -> Dict[str, Any]:
    if task_type == "classification":
        models["my_model"] = MyClassifier(random_state=seed)
    return models
```

Then update `base.yaml`:
```yaml
utility:
  models: ["logreg", "rf", "gb", "my_model"]
```

---

## Common Tasks

### Running the Pipeline

#### Single Dataset
```bash
python -m eval_harness.cli \
  --config configs/base.yaml \
  --dataset 336 \
  --outdir runs/exp1
```

#### All Datasets
```bash
python -m eval_harness.cli \
  --config configs/base.yaml \
  --all \
  --outdir runs/full
```

#### Legacy Mode (Direct CSV)
```bash
python -m eval_harness.cli \
  --config configs/base.yaml \
  --data data/ckd.csv \
  --target class \
  --outdir runs/ckd
```

#### With Data Root
```bash
python -m eval_harness.cli \
  --config configs/base.yaml \
  --dataset 336 \
  --data-root /path/to/data \
  --outdir runs/exp1
```

### Development Tasks

#### Install in Development Mode
```bash
cd synthetic_proxy_eval
pip install -e .
```

#### Install with Dev Dependencies
```bash
pip install -e ".[dev]"
```

#### Run Tests
```bash
pytest                          # All tests
pytest -v                       # Verbose
pytest tests/test_io.py        # Specific file
pytest -k "test_determinism"   # Pattern matching
```

#### Download UCI Datasets
```bash
python -m eval_harness.download_uci --config configs/base.yaml
```

#### Plot Top Metrics
```bash
python plot_top_metrics.py runs/exp1/stats.csv
```

### Analyzing Results

#### Output Directory Structure
```
<outdir>/<dataset_id>/
├── synthetic/                      # Generated synthetic datasets
│   ├── bootstrap_seed0_size0.5.parquet
│   ├── gaussian_copula_seed0_size1.0.parquet
│   └── tvae_seed0_size1.0.parquet
├── results.parquet                 # All metric values (rows = synth dataset)
├── summary.csv                     # Aggregated statistics
├── stats.csv                       # Correlation analysis (metric vs delta_u)
├── degradation.csv                 # Sensitivity test results
├── invariance.csv                  # Invariance test results
└── plots/                          # Matplotlib figures
    ├── scatter_rbo_0.9.png
    ├── roc_curve.png
    ├── pr_curve.png
    └── degradation_curves.png
```

#### Key Result Files

**results.parquet**: Full results
```python
import pandas as pd
df = pd.read_parquet("runs/exp1/336/results.parquet")
# Columns: generator, seed, size, delta_u, rbo_0.9, kendall_tau, ...
```

**stats.csv**: Correlation analysis
```python
df = pd.read_csv("runs/exp1/336/stats.csv")
# Columns: metric, spearman_rho, p_value, ci_lower, ci_upper, roc_auc, pr_auc, precision_at_target
```

---

## Git Workflow

### Branch Naming Convention
- Feature branches: `claude/feature-name-<session-id>`
- Example: `claude/add-tvae-generator-ABC123`

### Current Working Branch
```bash
git branch --show-current
# Output: claude/add-claude-documentation-ZijRn
```

### Commit Guidelines

1. **Clear, Descriptive Messages**: Focus on "why" not "what"
2. **Present Tense**: "Add feature" not "Added feature"
3. **No Emojis**: Keep commits professional
4. **Atomic Commits**: One logical change per commit

**Good Commit Messages:**
```
Add TVAE-like generator with GaussianMixture
Fix preprocessing leakage in test split
Update proxy metrics to handle tied rankings
```

**Bad Commit Messages:**
```
updates
fix bug
WIP
🎉 Added cool feature
```

### Pushing Changes

```bash
# Always push to the designated branch with -u flag
git push -u origin claude/add-claude-documentation-ZijRn
```

### Common Git Operations

```bash
# Check status
git status

# View recent commits
git log --oneline -5

# View diff
git diff

# Stage files
git add src/eval_harness/new_module.py

# Commit
git commit -m "Add new module for X functionality"

# Push to feature branch
git push -u origin claude/<branch-name>
```

---

## Troubleshooting

### Common Issues

#### Issue: Import Errors
**Symptom:** `ModuleNotFoundError: No module named 'eval_harness'`

**Solution:**
```bash
cd synthetic_proxy_eval
pip install -e .
```

#### Issue: Test Failures Due to Randomness
**Symptom:** Tests pass sometimes, fail other times

**Solution:** Check all random operations use explicit seeds:
```python
# CORRECT
rng = np.random.default_rng(seed)
model.fit(X, y, random_state=seed)

# INCORRECT
np.random.shuffle(X)  # No seed control
```

#### Issue: Preprocessing Leakage
**Symptom:** Test performance unrealistically high

**Solution:** Verify preprocessing fits ONLY on train:
```python
# CORRECT
transformer = fit_preprocess(df_train, schema, scaler_type)
X_train = transform_features(df_train, transformer)
X_test = transform_features(df_test, transformer)

# INCORRECT
X = transform_features(pd.concat([df_train, df_test]), transformer)
```

#### Issue: OneHotEncoder Errors on Test Data
**Symptom:** `ValueError: Found unknown categories`

**Solution:** Always use `handle_unknown="ignore"`:
```python
OneHotEncoder(handle_unknown="ignore")
```

#### Issue: Feature Importance Ranking Incorrect
**Symptom:** Too many/too few features in rankings

**Solution:** Check feature grouping tracks one-hot expansion correctly:
```python
# After one-hot: sex__M, sex__F → both map to group "sex"
feature_names: ["age", "sex__M", "sex__F"]
feature_group: ["age", "sex", "sex"]
```

#### Issue: CLI Fails to Find Dataset
**Symptom:** `FileNotFoundError: data/dataset.csv not found`

**Solution:** Use `--data-root` to specify data directory:
```bash
python -m eval_harness.cli \
  --config configs/base.yaml \
  --dataset 336 \
  --data-root /absolute/path/to/data \
  --outdir runs/exp1
```

#### Issue: Metrics Outside [0, 1] Range
**Symptom:** RBO or other metrics > 1.0 or < 0.0

**Solution:** Always clamp metrics:
```python
def _clamp_unit(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
```

### Debugging Tips

1. **Enable Verbose Logging**: Add print statements or use Python debugger
2. **Check Intermediate Outputs**: Verify shapes and types at each pipeline stage
3. **Use Small Test Data**: Run with `toy_df` fixture for faster iteration
4. **Check Random Seeds**: Ensure deterministic behavior for debugging
5. **Validate Schemas**: Print schema after inference to verify correct types
6. **Test in Isolation**: Use pytest with `-k` to run specific tests

### Performance Issues

#### Slow Test Execution
**Solutions:**
- Reduce `n_resamples` in `test_config` fixture
- Use fewer models in test configs
- Run specific tests with `pytest -k pattern`
- Parallelize with `pytest -n auto` (requires pytest-xdist)

#### Out of Memory
**Solutions:**
- Reduce dataset sizes in config (`sizes: [0.5]` instead of `[0.5, 1.0, 2.0]`)
- Reduce number of synthetic seeds
- Process datasets sequentially instead of with `--all`

---

## Best Practices Summary

### DO:
✅ Always read files before modifying them
✅ Use explicit random seeds everywhere
✅ Fit preprocessing ONLY on training data
✅ Track feature groups for one-hot encoded features
✅ Clamp metrics to valid ranges
✅ Write deterministic tests
✅ Use type hints consistently
✅ Document changes with clear commit messages
✅ Run tests before and after changes
✅ Handle edge cases (empty data, NaN values)

### DON'T:
❌ Fit preprocessing on test or combined data
❌ Use implicit randomness without seeds
❌ Modify code without reading it first
❌ Break backward compatibility with existing configs
❌ Skip writing tests for new functionality
❌ Leave TODOs or placeholders in production code
❌ Use emojis in code or commits
❌ Ignore test failures
❌ Forget to handle unknown categories in OneHotEncoder
❌ Create new files when editing existing ones would work

---

## Quick Reference

### Key Files to Know

| File | Purpose | When to Modify |
|------|---------|----------------|
| `cli.py` | CLI interface | Adding command-line arguments |
| `experiments.py` | Pipeline orchestration | Changing pipeline flow |
| `io.py` | Data loading | Adding new data sources |
| `generators.py` | Synthetic generation | Adding new generators |
| `proxy_metrics.py` | Ranking metrics | Adding new proxy metrics |
| `fidelity_metrics.py` | Distribution metrics | Adding new fidelity metrics |
| `base.yaml` | Main configuration | Changing default parameters |
| `datasets_uci_health.yaml` | Dataset definitions | Adding new datasets |
| `pyproject.toml` | Project metadata | Adding dependencies |
| `pytest.ini` | Test configuration | Changing test behavior |

### Essential Commands

```bash
# Development
pip install -e .                        # Install in dev mode
pip install -e ".[dev]"                 # Install with dev dependencies

# Testing
pytest                                   # Run all tests
pytest -v                                # Verbose output
pytest -k "pattern"                      # Run matching tests
pytest tests/test_file.py               # Run specific file

# Running Pipeline
python -m eval_harness.cli --config configs/base.yaml --dataset 336 --outdir runs/exp1

# Git
git status                               # Check status
git add <file>                           # Stage changes
git commit -m "message"                  # Commit
git push -u origin <branch>              # Push to branch
```

---

**Document Version:** 1.0
**Created:** 2026-01-20
**For:** AI Assistants working with tabular-similarity-v2
**Maintainer:** Auto-generated by Claude Code
