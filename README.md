# Machine-Learning Pipeline Template

A reusable template for a batch machine-learning pipeline. It reads external
source tables, derives an outcome table and a feature table, splits cases by
date, trains one or more model strategies, generates predictions, and writes
evaluation metrics and reports.

The template ships a deterministic synthetic demo, so the whole flow runs
without supplied data. Replace the demo producers in `src/project/adapters.py`
with your own transformations; the stage plumbing, orchestration, and artifact
contracts stay unchanged.

## Pipeline stages

| Stage | Responsibility |
| --- | --- |
| `s01_outcome` | Read the external source table and build the outcome table. |
| `s01b_outcome_analysis` | Describe the outcome table and the registered outcome comparisons. |
| `s02_feature` | Read the external source table and build the feature table. |
| `s03_divide` | Join one outcome with the features and split cases into `train`, `valid`, and `test` by `case_date`. |
| `s04_train` | Train each registered model strategy for each modeled outcome. |
| `s05_predict` | Generate predictions for every declared partition of every trained run. |
| `s06_evaluate` | Compute metrics, plots, feature importances, and the per-outcome display table. |

```text
external source table
        │
        ▼
s01_outcome ──► s01b_outcome_analysis
        │
        ▼
s02_feature
        │
        ▼
s03_divide
        │
        ▼
s04_train
        │
        ▼
s05_predict
        │
        ▼
s06_evaluate
```

## Requirements

- [pixi](https://pixi.sh) for environment and command management.
- Python 3.12, declared in `pixi.toml` and `pyproject.toml`.
- The declared environment targets `linux-64`.

Runtime dependencies are declared in `pixi.toml` and pinned in `pixi.lock`.
Test, lint, and type-check tooling lives in the `dev` feature of the same
environment.

## Quick start

```bash
pixi install
OMP_NUM_THREADS=1 pixi run python -m src.main
```

The run writes the complete output tree under `output/`. The environment sets
`OMP_NUM_THREADS=1` on activation; passing it explicitly keeps the command
reproducible when Python is launched outside `pixi run`.

Run the test suite and the static checks:

```bash
pixi run pytest -q
pixi run ruff check .
pixi run ruff format --check .
pixi run pyright
```

## Output tree

```text
output/
  s01_outcome/
    outcome_df.parquet
    outcome_statistics.txt
  s01b_outcome_analysis/
    outcome_differences.parquet
    outcome_differences.csv
    outcome_differences_statistics.txt
    <left>_vs_<right>_{histogram,kde,difference_histogram,difference_kde}.png
  s02_feature/
    feature_df.parquet
    feature_df_head.csv
    feature_manifest.json
  s03_divide/{outcome}/
    train_df.parquet, valid_df.parquet, test_df.parquet
    feature_head_df.parquet, outcome_head_df.parquet
    statistics.txt
  s04_train/{outcome}/{model}/
    manifest.json
    model.json, model_pretty.json, model.txt
    {partition}_key_df.{parquet,csv}
    {partition}_feature_df.{parquet,csv}
    feature_importances.{parquet,csv}
  s05_predict/{outcome}/{model}/
    manifest.json
    {model}_{partition}_predictions.{parquet,csv}
  s06_evaluate/{outcome}/
    {outcome}_display.csv
    {model}_feature_importances.png
    {model}_{partition}/
      {model}_regression_metrics.json
      {model}_transform_binary_<threshold>_metrics.json
      *.png
```

Manifests are the artifact index. Downstream stages read a run's manifest to
find its declared artifacts instead of discovering files by name or directory
depth.

## Repository layout

```text
src/core/       reusable, project-agnostic machinery
src/project/    configuration, registries, schemas, adapters, orchestration
src/stages/     thin stage entry points
scripts/        demo generation, stage runner, repository checks
tests/          unit, property-based, integration, smoke, and golden tests
docs/           architecture and design notes
data/demo/      committed deterministic demo source table
output/         runtime output (gitignored)
```

## Configuration

`src/project/config.py` defines the frozen `ProjectConfig` dataclass and the
module-level `CONFIG` instance used by the entry point:

| Setting | Meaning |
| --- | --- |
| `data_root` | Directory holding external input tables, relative to the project root. |
| `train_end_date` | Last `case_date` in the `train` partition. |
| `test_start_date` | First `case_date` in the `test` partition. |
| `feature_columns_to_drop` | Feature columns excluded before training. |
| `binary_thresholds` | Thresholds used to derive binary labels from regression values. |
| `evaluated_partitions` | Partitions that receive metrics and reports. |
| `feature_importance_plot_top_n` | Number of features drawn in the importance plot. |
| `demo_seed`, `demo_case_count` | Seed and row count for the demo source table. |
| `xgb_*` | Estimator search and single-fit defaults. |

Outcome identity, transforms, and analysis pairs live in
`src/project/outcomes.py`. Feature policies and the model-strategy table live
in `src/project/features.py` and `src/project/models.py`. External input tables
are declared in `src/project/inputs.py` and validated against the boundary
schemas in `src/project/schemas.py`.

## Customizing the template

1. Put your source parquet file under `data/demo/` (or change `data_root`) and
   update the `InputTableSpec` entry in `src/project/inputs.py`.
2. Update `SOURCE_TABLE_SCHEMA` and the stage schemas in
   `src/project/schemas.py` to match your columns.
3. Replace `build_outcome_table` and `build_feature_table` in
   `src/project/adapters.py` with your calculations. Keep the canonical key
   columns and the key-alignment contract.
4. Register your outcomes, transforms, and analysis comparisons in
   `src/project/outcomes.py`.
5. Adjust the feature policies, model strategies, and configuration values.
6. Run the pipeline and verify the output tree.

The stage functions, manifest readers, and orchestration registry do not need
to change for a new project as long as the adapters return frames that satisfy
the boundary schemas.

## Running individual stages

The stage runner plans and executes stages against the declarative stage table:

```bash
# Show stages, outcomes, and model ids
pixi run python scripts/run_stage.py --list

# Show the ordered calls without importing or executing a stage
pixi run python scripts/run_stage.py --stage s03_divide --dry-run

# Run one stage, or a stage plus its prerequisite closure
pixi run python scripts/run_stage.py --stage s03_divide --with-prereqs

# Restrict a per-outcome or per-model run
pixi run python scripts/run_stage.py --stage s04_train --outcome outcome_1 --model model_00_signal_0
```

The output root defaults to `<repository>/output` and can be overridden with
`--output-root`.

## Regenerating the demo source table

The committed demo table is reproducible from the configured seed:

```bash
pixi run python scripts/make_demo_data.py
```

The output path must stay inside `data/demo/`. The script writes a temporary
sibling and replaces the target atomically.

## Reproducibility

The demo is reproducible under the test configuration because:

- the source table is committed;
- every demo adapter uses its own
  `numpy.random.default_rng(config.demo_seed)` and never touches the global
  `numpy` random state;
- feature policies and the estimator draw from the global `numpy` random
  state, seeded once in `tests/conftest.py` and before every golden run;
- `OMP_NUM_THREADS=1` and the estimator's single-thread setting are applied
  before the numeric libraries run;
- the golden tests compare the committed baseline byte for byte.

A bare `python -m src.main` run uses the process's initial global RNG state for
feature-policy noise. Use the golden test harness when byte-level
reproducibility matters.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module map, data contracts,
  design decisions, testing strategy, and limitations.
- [`AGENTS.md`](AGENTS.md) — commands and conventions for coding agents.
- [`docs/research/`](docs/research/) — design notes and the extraction plan.
