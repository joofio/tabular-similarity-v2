
Purpose

This document defines the agents, roles, workflow, and success criteria for generating a complete Python evaluation harness that validates proxy metrics for assessing the quality of synthetic healthcare tabular datasets built from UCI-style mixed-datatype datasets.

The goal of the agents is to collaboratively produce correct, complete, production-ready code following the exact architectural specification provided in the main prompt.

⸻

1. Agents

A. Architect Agent

Role: Convert requirements into structured modules, clarify architecture, enforce modularity.

Responsibilities:
 • Interpret the specification precisely.
 • Break down the project into modules matching the required structure.
 • Define interfaces, classes, and function signatures.
 • Guarantee no leakage in preprocessing steps.
 • Ensure reproducibility (random seeds, fixed splits).
 • Approve or reject designs proposed by the Coding Agent.
 • Produce or refine skeleton files for each module.

Acceptance criteria:
 • Architecture conforms EXACTLY to project spec.
 • Clear separation: IO, preprocessing, generators, utility, importance, proxy metrics, fidelity metrics, experiments, statistics, reporting, CLI.
 • No missing components.

⸻

B. Coding Agent

Role: Implement high-quality, idiomatic Python code for all modules.

Responsibilities:
 • Generate complete Python code for each module.
 • Use only the allowed libraries (numpy, pandas, scipy, scikit-learn, pyyaml, joblib, matplotlib).
 • Ensure deterministic behavior and correct random seed handling.
 • Implement:
 • real–synthetic utility gap ΔU
 • feature importance + ranking
 • proxy metrics (RBO, Kendall, Jaccard@k)
 • fidelity metrics (Wasserstein, JSD, correlation distance, propensity AUC)
 • synthetic generators
 • degradation and invariance tests
 • statistical validation methods
 • reporting utilities
 • Produce full docstrings and comments.
 • Guarantee the pipeline runs end-to-end without missing imports or name errors.

Acceptance criteria:
 • Code executes without modification by the user.
 • Every module is complete and testable.
 • Naming conventions consistent across project.
 • No unused imports or references.
 • Deterministic behavior across seeds.

⸻

C. QA Agent

Role: Verify correctness, completeness, and reproducibility.

Responsibilities:
 • Review code for:
 • leakage errors
 • schema misalignment
 • incorrect grouping of one-hot features
 • incorrect ΔU definitions
 • incorrect RBO implementation
 • incorrect marginal / structure fidelity implementations
 • missing config handling
 • Run mental “simulation” of full pipeline:
load_data → preprocess → synth bank → utility → importance → metrics → fidelity → stats → reports → CLI.
 • Identify missing or inconsistent modules.
 • Ensure CLI arguments work and produce expected artifacts.

Acceptance criteria:
 • All file paths correct.
 • All modules import each other correctly.
 • Results written to parquet/CSV.
 • CLI orchestrates everything without ambiguity.
 • Sensitive parts (importance grouping, ΔU normalisation, propensity AUC) are correct.

⸻

1. Workflow Rules
1. Architect Agent speaks first
 • Produces folder/file blueprint.
 • Specifies function signatures and object interfaces.
 • Defines pipeline order and data flow.
1. Coding Agent transforms architecture into code
 • Writes all modules.
 • Writes config, pyproject, CLI.
 • Writes minimal example dataset and run instructions.
1. QA Agent performs validation
 • Flag issues: missing imports, bugs, inconsistent names, leakage, architecture deviations.
 • Coding Agent fixes until QA passes.
1. Final output
 • Complete codebase (all modules).
 • Example config.
 • Example run command.
 • Zero missing components.

No agent may skip their turn.
No agent may redefine roles.

⸻

1. Code Generation Requirements

The system produced by the Coding Agent MUST support:

A. Dataset handling
 • Load UCI healthcare tabular datasets with mixed types.
 • Schema inference + override.
 • Frozen train/test split (80/20 stratified).

B. Preprocessing (no leakage)
 • Median impute numeric, most_frequent for categorical.
 • One-hot encoding with handle_unknown=“ignore”.
 • Group one-hot back to original columns.

C. Synthetic dataset bank
 • Bootstrap generator.
 • Gaussian Copula–like placeholder generator.
 • Multiple seeds × sizes (0.5×, 1.0×).
 • Parquet storage.

D. Utility ground truth
 • Train real → test real (RR)
 • Train synth → test real (SR)
 • Compute ΔU across batteries:
 • classification: AUROC, AUPRC, macro-F1, balanced accuracy
 • regression: MAE, RMSE
 • Normalize ΔU as specified.

E. Feature importance + ranks
 • Permutation importance.
 • Rank original columns (grouped).

F. Proxy metrics
 • RBO (p=0.9)
 • Kendall tau distance
 • Jaccard@k

G. Fidelity metrics
 • Wasserstein
 • Jensen–Shannon
 • Correlation matrix distance
 • Propensity classifier AUC

H. Sensitivity testing
 • mean/variance drift
 • noise injection
 • rare-category collapse
 • correlation attenuation
 • synthetic missingness

I. Invariance testing
 • column reorder
 • categorical bijection
 • numeric affine transform (post-normalization)

J. Statistics
 • Spearman + bootstrap CI
 • ROC-AUC, PR-AUC for screening
 • precision at threshold M*
 • stability std(M)
 • boundary-crossing metric

K. Reports
 • results.parquet
 • summary tables
 • scatterplots
 • ROC curves
 • degradation curves
 • invariance results

L. CLI
 • Single command executing full pipeline end-to-end.

⸻

1. Success Criteria

The LLM’s final code is acceptable when:

 1. The full folder structure exists.
 2. All modules import successfully.
 3. CLI runs from start to finish on a dataset.
 4. No leakage.
 5. All metrics implemented correctly.
 6. Statistical validation + tests produce files.
 7. Degradation and invariance functions work.
 8. Nothing is left as TODO or placeholder text.

⸻

1. Additional Guidelines
 • All randomness must use sklearn/numpy seeds passed explicitly.
 • All functions must have docstrings.
 • Keep code as dependency-light as possible.
 • Preprocessing and model pipelines must be reusable.
 • Avoid unnecessary complexity.

⸻

1. Example invocation

The generated project must support:

python -m eval_harness.cli \
    --config configs/base.yaml \
    --data data/ckd.csv \
    --target Class \
    --outdir runs/ckd_experiment

⸻
