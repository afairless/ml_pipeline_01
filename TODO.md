# Implementation Plan: Extract a Reusable Machine-Learning Pipeline Template

Source: `docs/research/plan-extract-generic-ml-pipeline.md`

Implement the plan in the order below. Each row is one commit: implement only
that row, run the gates, commit, then stop before starting the next row. The
plan document is the specification; this file is the step contract.

## Prerequisites

These must hold before row 1. They are not commits.

- **P1 — External source tree.** The source repository is supplied out of band
  at implementation time. Its path is deliberately **not** recorded in this
  repository, in any file, commit message, or test. It is needed for the
  `.port-staging/inventory.md` role inventory (plan §2) and for the denylist
  derivation (plan §6.1a). Confirmed available: the source whose
  `src/{core,project,stages}` layout matches plan §3/§7.
- **P2 — Toolchain.** `pixi 0.71.2`, `/usr/bin/python3` = 3.12.3, and `git`
  are present. Python is pinned to 3.12; every command runs through
  `pixi run`, never bare `python`.
- **P3 — Repository bootstrap (this planning commit).** `git init`, the
  `.gitignore` from plan §3, this file, and the plan document are committed
  before row 1. Those bootstrap commits are vocabulary-gated through
  `scrub_repo.py --history` once the gate exists (row 3), not before.
- **P4 — Hook installation.** After row 3, run
  `git config core.hooksPath .githooks`. It is per-clone, so it is never
  treated as proof of a clean commit.
- **P5 — Denylist.** `.scrub/denylist.txt` is derived from the source tree
  (plan §6.1a), human-reviewed, and hashed in row 3. Until it exists and is
  reviewed, the gate fails closed: no ported code may be committed.

## Steps

| # | Commit message | Logical unit | Key deliverables | Tests | Plan §9 |
| --- | --- | --- | --- | --- | --- |
| 1 | chore: add repository tooling skeleton | Toolchain skeleton | `pyproject.toml`, `pixi.toml`, `README.md` stub, `docs/research/README.md`, package `__init__.py` for `src/`, `src/core/`, `src/project/`, `src/stages/`, `tests/`; extend `.gitignore` | Smoke | 1 |
| 2 | chore: pin the development environment | Locked environment | `pixi.lock`; untracked `.port-staging/audit.md` dependency-audit record for every added third-party package | Smoke | 2 |
| 3 | feat: add vocabulary scrub gate | Vocabulary gate | `scripts/__init__.py`, `scripts/build_denylist.py`, `scripts/scrub_repo.py`, `.githooks/pre-commit`, `tests/test_vocabulary_scrub.py`, `tests/vocabulary_denylist.sha256` | Unit, Property-based | 3 |
| 4 | feat: port core path and context primitives | Core paths and context | `src/core/paths.py` (config Protocol, `StageSpec`, `PipelineLayout`, `PipelineContext`, `ensure`), `tests/test_paths.py`, `tests/test_context.py`, `tests/test_module_imports.py` | Unit, Smoke | 5 |
| 5 | feat: port core manifest kinds and run discovery | Core manifests | `src/core/manifest.py` (`ArtifactRef`, `ModelManifest`, `PredictionManifest`, `FeatureManifest`, read/write, run discovery), `tests/test_manifest.py` | Unit, Property-based | 6 |
| 6 | feat: port core metrics | Regression metrics | `src/core/metrics.py`, `tests/test_core_metrics.py` | Unit | 7 |
| 7 | feat: port core binary metrics and threshold helper | Binary metrics | `src/core/binary_metrics.py` (incl. `threshold_labels`), `tests/test_core_binary_metrics.py`, `tests/test_core_binary_metrics_plots.py` | Unit | 8 |
| 8 | feat: port core plotting, serialization, and binning | Plot, I/O, and binning helpers | `src/core/plots.py`, `src/core/io.py`, `src/core/binning.py`, `tests/test_core_plots.py`, `tests/test_core_io.py`, `tests/test_core_binning.py` | Unit | 9 |
| 9 | feat: port core partition layer | Partition contract | `src/core/splits.py` (`DataPartition`, `PARTITION_ORDER`, `holdout_partitions`), `tests/test_splits.py` | Unit, Property-based | 10 |
| 10 | feat: port core estimator layer | Estimator protocol and registry | `src/core/estimator.py`, `src/core/estimators/__init__.py`, `src/core/estimators/xgboost.py`, `tests/test_core_estimators.py` | Unit | 11 |
| 11 | feat: port core training mechanics | Training mechanics | `src/core/training.py` (`fit_single`, `run_param_search`), `tests/test_core_training.py` | Unit | 12 |
| 12 | feat: add project column vocabulary | Column vocabulary | `src/project/columns.py`, `tests/test_columns.py` | Unit | 13a |
| 13 | feat: add project configuration | Project config | `src/project/config.py` (frozen `ProjectConfig`, `CONFIG`, reporting policy tables), `tests/test_project_config.py` | Unit | 13b |
| 14 | feat: add project outcome registry | Outcome registry | `src/project/outcomes.py` (`Outcome`, `OUTCOMES`, `OutcomeComparison`, `OUTCOME_COMPARISONS`, `resolve_transform`), `tests/test_outcomes.py`, `tests/test_outcome_column_vocabulary.py` | Unit, Property-based | 13c |
| 15 | feat: add project boundary schemas | Boundary schemas | `src/project/schemas.py` (`ColumnRule`, `validate`, all `S0x_*_SCHEMA`), `tests/test_schemas.py` | Unit | 14 |
| 16 | feat: add project external-input registry | Input registry | `src/project/inputs.py` (`InputTableSpec`, `INPUT_TABLES`, eager read, lazy `inspect_input_table`), `tests/test_inputs.py` | Unit | 15 |
| 17 | feat: add project feature policies | Feature policies | `src/project/features.py` (`FeaturePolicy`, `IdentityFeatures`, `OutcomePlusNoise`, `PureNoise`, `FeatureSubset`), `tests/test_features.py` | Unit | 16 |
| 18 | feat: add project model strategies | Model strategies | `src/project/models.py` (`ModelStrategy`, 8-entry `MODEL_STRATEGIES`), `tests/test_model_strategies.py` | Unit | 17 |
| 19 | feat: add project estimator factory | Estimator factory | `src/project/estimators.py` (`ESTIMATOR_ID`, `make_single_fit_adapter`, `make_search_adapter_factory`, `booster_of`), `tests/test_project_estimators.py` | Unit | 18 |
| 20 | feat: add project artifact I/O | Artifact I/O | `src/project/artifacts.py` (`SplitTables`, `load_split_tables`, `save_results`, feature-importance I/O, model manifest production), `tests/test_s04_extract_feature_importances.py`, `tests/test_s04_load_feature_importances.py`, `tests/test_s04_model_manifest.py` | Unit, Property-based | 19 |
| 21 | feat: add project training driver | Training driver | `src/project/training.py` (model-ready data, split sequencing, `train_model_run`), `tests/test_project_training.py` | Integration | 20 |
| 22 | feat: add stage table and orchestration registry | Stage table | `src/project/pipeline.py` (`StageDescriptor`, `STAGES`, `EXPLICIT_STAGES`, `REGISTRY_STAGES`, `StageBinding`, `run_pipeline`, `build_paths`), `tests/test_project_pipeline.py`, `tests/test_contracts.py` | Unit | 21 |
| 23 | feat: add run-stage planner and CLI wrapper | Run-stage planner | `src/project/run_stage.py` (planner, `render_list`, `render_plan`), `scripts/run_stage.py`, `tests/test_run_stage.py` | Unit, Smoke | 22 |
| 24 | feat: add deterministic demo data | Demo source table | `scripts/make_demo_data.py`, `data/demo/source_table_1.parquet` | Unit | 23 |
| 25 | feat: add project adapters | Demo adapters | `src/project/adapters.py` (`build_outcome_table`, `build_feature_table`), `tests/test_adapters.py` | Unit | 24 |
| 26 | feat: add s01 outcome stage | s01 outcome stage | `src/stages/s01_outcome.py`, `tests/test_s01_outcome.py` | Integration | 25 |
| 27 | feat: add s01b outcome-comparison stage | s01b analysis stage | `src/stages/s01b_outcome_analysis.py`, `tests/test_s01b_outcome_analysis.py` | Integration | 26 |
| 28 | feat: add s02 feature stage | s02 feature stage | `src/stages/s02_feature.py`, `tests/test_s02_feature.py` | Integration | 27 |
| 29 | feat: add s03 divide stage | s03 divide stage | `src/stages/s03_divide_data.py`, `tests/test_s03_divide.py` | Integration, Property-based | 28 |
| 30 | feat: add s04 train stage | s04 train stage | `src/stages/s04_train.py`, `tests/test_s04_train.py` | Integration | 29 |
| 31 | feat: add s05 predict stage | s05 predict stages | `src/stages/s05_predict_model.py`, `src/stages/s05_predict_model_main.py`, `tests/test_s05_manifest_reader.py`, `tests/test_s05_prediction_manifest.py` | Integration | 30 |
| 32 | feat: add s06 metric and report helpers | s06 metric helpers | `src/stages/s06_evaluate_model.py`, `tests/test_s06_metrics.py`, `tests/test_s06_display_table.py` | Unit | 31a |
| 33 | feat: add s06 evaluate stage driver | s06 driver | `src/stages/s06_evaluate_model_main.py`, `tests/test_s06_evaluate.py` | Integration | 31b |
| 34 | feat: add pipeline entry point and smoke tests | Entry point | `src/main.py` (with `__main__` guard), `tests/test_smoke.py`, `tests/test_smoke_pipeline.py`, `tests/test_smoke_main.py` | Smoke, Integration | 32 |
| 35 | test: add golden baseline harness | Golden harness | `tests/golden/__init__.py`, `aliases.py`, `capture.py`, `diff.py`, `paths.py`, `runner.py`, `tests/test_golden_harness.py` | Unit, Smoke | 33a |
| 36 | test: capture golden baseline and diff tests | Golden baseline | `tests/golden/baseline/output/**` (manifests, metrics JSON, statistics text, display CSVs only), `tests/test_golden_baseline.py`, `tests/test_golden_diff.py` | Integration | 33b |
| 37 | docs: write architecture, agent, and user documentation | Documentation | `docs/ARCHITECTURE.md`, `AGENTS.md`, `README.md`, `docs/research/README.md` | — | 34 |

## Per-step gates

Every row, before its commit:

```bash
pixi run ruff check .
pixi run ruff format --check .
pixi run pyright
pixi run pytest -q
```

plus the vocabulary gate once row 3 lands:

```bash
pixi run python scripts/scrub_repo.py --worktree
pixi run python scripts/scrub_repo.py --staged
```

Immediately after the row 3 commit, run
`pixi run python scripts/scrub_repo.py --history` to verify the bootstrap
commits retroactively (plan §6.3, §9). From row 4 onward the gate is
mandatory for every commit and every commit message.

## Final acceptance (no commit)

Plan §11, verified after row 37. No commit is produced, so this is a checklist
rather than a step:

- `pixi run pytest -q` green.
- `OMP_NUM_THREADS=1 pixi run python -m src.main` produces the plan §8.5
  output tree.
- `ruff check`, `ruff format --check`, and `pyright` clean.
- `scrub_repo.py --worktree`, `--staged`, and `--history` all exit `0`.
- Every commit subject in `git log --format=%B` passes the same tokenizer.
- `tests/test_module_imports.py` proves `src.core` imports no `src.project`
  and no stage module.
- The golden baseline diffs exactly on a second run and contains only the
  plan §8.5 baseline artifacts.
- Documentation names `src/project/adapters.py` as the extension point and
  contains no subject-matter vocabulary; the README has no development
  history.
- `.scrub/denylist.txt` was derived from the supplied source tree and reviewed
  before row 3; only the committed hash list gates commits.
- `.port-staging/audit.md` records a clean dependency audit for every added
  third-party package (untracked by design).

## Mapping notes

- Plan §9 row 0 (`git init`, `.gitignore`, hook config) is prerequisite P3/P4,
  not a commit row.
- Plan §9 row 4 (`docs: add extraction plan`) is the bootstrap commit made
  when this file was added, so it has no row here.
- Plan §9 row 35 (`chore: final acceptance audit`) produces no commit; it is
  the **Final acceptance** checklist above.
- Plan §9 rows 13, 31, and 33 are split (`13a`–`13c`, `31a`–`31b`,
  `33a`–`33b`) so each commit stays one verifiable unit.
- Rows 1–2 predate the gate and are verified retroactively with `--history`
  right after row 3, exactly like the plan-document bootstrap commit. Rows 3–37
  run `--worktree` and `--staged` before every commit.
- Each row's named pytest files are the step's minimum; the full suite must
  also stay green.
