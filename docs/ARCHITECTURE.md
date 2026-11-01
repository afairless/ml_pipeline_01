# Architecture

## 1. Overview

This repository is a template for a batch machine-learning pipeline. It reads
one or more external source tables, derives outcomes and features, divides
cases into date-based holdout partitions, trains model strategies, predicts on
every declared partition, and evaluates the results.

The architecture is optimized for three properties:

- **Reusable machinery.** Project-agnostic path, manifest, metric, plotting,
  serialization, partition, estimator, and training code lives in `src/core/`.
- **A single extension seam.** A new project changes configuration,
  registries, boundary schemas, and the two demo producers in
  `src/project/adapters.py`; the orchestration and stage plumbing stay intact.
- **Inspectable artifacts.** Every run writes a manifest that declares its
  identity, configuration, and artifact inventory. Downstream consumers read
  manifests rather than guessing filenames or directory depth.

The audience is a developer starting a new batch machine-learning project from
this template, or an operator running and extending an instantiated copy.

## 2. Layering

```text
┌──────────────────────────────────────────────┐
│ src/stages/    thin stage entry points       │
│ s01 … s06      resolve, validate, delegate   │
└──────────────────┬───────────────────────────┘
                   │ depends on
┌──────────────────▼───────────────────────────┐
│ src/project/   configuration and registries  │
│ adapters, schemas, inputs, stage table,      │
│ artifact I/O, training driver                │
└──────────────────┬───────────────────────────┘
                   │ depends on
┌──────────────────▼───────────────────────────┐
│ src/core/      reusable, project-agnostic    │
│ paths, manifests, metrics, plots, I/O,       │
│ partitions, estimator protocol, training     │
└──────────────────────────────────────────────┘
```

The dependency direction is one-way. `src/core/` never imports
`src/project/` or any stage module; `tests/test_module_imports.py` enforces
this with an AST walk. The core layer types the configuration it needs through
the `ProjectConfig` protocol in `src/core/paths.py`, and the project dataclass
satisfies that protocol structurally.

### 2.1 Module map

| Module | Responsibility | Key types / functions |
| --- | --- | --- |
| `src/core/paths.py` | Declarative, side-effect-free stage layout and the value every stage receives. | `StageSpec`, `PipelineLayout`, `PipelineContext`, `ProjectConfig` protocol, `ensure` |
| `src/core/manifest.py` | Frozen model, prediction, and feature manifest kinds plus structural run discovery. | `ArtifactRef`, `ModelManifest`, `PredictionManifest`, `FeatureManifest` |
| `src/core/metrics.py` | Regression metrics on the original and model target scales. | `RegressionMetrics` |
| `src/core/binary_metrics.py` | Binary metrics and the single threshold-to-label rule. | `threshold_labels`, `BinaryClassificationMetrics` |
| `src/core/plots.py` | Figure-owning plotting helpers that always write their declared artifact. | Histogram, KDE, importance, and metric helpers |
| `src/core/io.py` | Serialization helpers for table, head-preview, and full-table artifacts. | Table writers |
| `src/core/binning.py` | Pure uniform-binning utility. | Binning helper |
| `src/core/splits.py` | Canonical holdout partition vocabulary and alignment contract. | `DataPartition`, `PARTITION_ORDER`, `holdout_partitions` |
| `src/core/estimator.py` | Estimator adapter protocol and registry. | `EstimatorAdapter`, `EstimatorRegistry` |
| `src/core/estimators/xgboost.py` | The shipped estimator adapter family. | `XGBoostAdapter`, `ESTIMATOR_ID` |
| `src/core/training.py` | Estimator-agnostic training mechanics. | `fit_single`, `run_param_search` |
| `src/project/config.py` | Frozen project configuration and reporting policy tables. | `ProjectConfig`, `CONFIG` |
| `src/project/columns.py` | Canonical column vocabulary. | `case_id`, `case_date`, `KEY_COLUMNS`, feature and importance columns |
| `src/project/outcomes.py` | Outcome registry, comparisons, and transform resolver. | `Outcome`, `OUTCOMES`, `OutcomeComparison`, `OUTCOME_COMPARISONS`, `resolve_transform` |
| `src/project/schemas.py` | Boundary schemas and the frame validator. | `ColumnRule`, `TableSchema`, `validate`, `S01_OUTCOME_SCHEMA`, …, `S05_PREDICTION_SCHEMA` |
| `src/project/inputs.py` | External input-table registry and ingestion boundary. | `InputTableSpec`, `INPUT_TABLES`, `read_input_table`, `inspect_input_table` |
| `src/project/adapters.py` | Demo outcome and feature producers; the documented extension point. | `build_outcome_table`, `build_feature_table` |
| `src/project/features.py` | Feature-selection and substitution policies. | `FeaturePolicy`, `IdentityFeatures`, `OutcomePlusNoise`, `PureNoise`, `FeatureSubset` |
| `src/project/models.py` | Model-strategy registry. | `ModelStrategy`, `MODEL_STRATEGIES` |
| `src/project/estimators.py` | Estimator-specific construction for the training driver. | `ESTIMATOR_ID`, adapter factories, `booster_of` |
| `src/project/artifacts.py` | Split-table loading and trained-run artifact I/O. | `SplitTables`, `load_split_tables`, `save_results`, importance I/O |
| `src/project/training.py` | Model-ready data preparation and the training driver. | `train_model_run` |
| `src/project/pipeline.py` | Declarative stage table and orchestration registry. | `StageDescriptor`, `STAGES`, `EXPLICIT_STAGES`, `REGISTRY_STAGES`, `StageBinding`, `run_pipeline`, `build_paths` |
| `src/project/run_stage.py` | Importable stage planner and executor. | `StageSelection`, `PlannedCall`, `plan_calls`, `run_stages`, `render_list`, `render_plan` |
| `src/stages/` | Thin stage entry points. | `s01_outcome`, `s01b_outcome_analysis`, `s02_feature`, `s03_divide_data`, `s04_train`, `s05_predict_model_main`, `s06_evaluate_model_main` |
| `src/main.py` | Entry point: three explicit stages followed by the registry-driven run. | `main` |
| `scripts/` | Demo generation, stage runner, and repository checks. | `make_demo_data.py`, `run_stage.py`, `scrub_repo.py`, `build_denylist.py` |
| `tests/` | Unit, property-based, integration, smoke, and golden tests. | `test_*`, `golden/` |
| `tests/golden/` | Baseline capture, alias selection, and byte-exact diffing. | `capture`, `diff`, `runner`, `aliases`, `paths` |

## 3. Data flow

### 3.1 Stage sequence

`src/main.py` runs the first three stages explicitly because they either
ingest external data or run once per pipeline run:

1. `s01_outcome` reads `source_table_1` through the input registry, calls
   `build_outcome_table`, validates `S01_OUTCOME_SCHEMA`, and writes the
   outcome parquet and statistics text.
2. `s01b_outcome_analysis` reads and validates the outcome table, differences
   every registered outcome comparison as `left - right`, and writes the
   difference table, its CSV projection, statistics, and four plots per
   comparison.
3. `s02_feature` reads `source_table_1` and the outcome keys, calls
   `build_feature_table`, asserts the feature keys equal the outcome keys
   exactly, validates `S02_FEATURE_SCHEMA`, and writes the feature parquet,
   head CSV, and feature manifest.

The remaining stages are registry-driven. `run_pipeline` executes them in
stage-table order and fans them out over the outcome and model registries:

4. `s03_divide` joins one outcome column and the feature table on
   `(case_id, case_date)`, asserts key uniqueness and row conservation, and
   splits the rows by `case_date`:
   - `train`: `case_date <= train_end_date`
   - `valid`: `train_end_date < case_date < test_start_date`
   - `test`: `case_date >= test_start_date`

   Each per-outcome directory carries only that outcome's column.
5. `s04_train` trains every model strategy for each modeled outcome.
6. `s05_predict` discovers trained runs through their manifests and predicts
   on every declared partition.
7. `s06_evaluate` joins predictions to split tables, writes regression and
   thresholded binary metrics, feature-importance plots, and the per-outcome
   display table.

### 3.2 Artifact and manifest contract

Every consumed artifact has one declaration site:

- `src/project/pipeline.py` declares each stage's directory template and
  logical artifact filenames.
- `src/project/inputs.py` declares each external input table.
- `src/project/schemas.py` declares each boundary schema.
- `src/core/manifest.py` declares each run manifest kind and schema version.

Model and prediction runs write a `manifest.json` beside their artifacts; the
feature run writes `feature_manifest.json`. The manifest records the run
identity, configuration, partition names, and an artifact inventory. A
consumer resolves artifacts through that inventory; it never globs a directory
or infers identity from a filename. An unknown manifest version or a missing
declared artifact is an error, not a partial load.

`s04_train` and `s05_predict` declare no stage-level artifact list because
each run directory is discovered through its own manifest. `s06_evaluate`
declares the per-outcome display table at the stage-directory root.

### 3.3 Runtime output tree

```text
output/
  s01_outcome/
    outcome_df.parquet, outcome_statistics.txt
  s01b_outcome_analysis/
    outcome_differences.parquet, outcome_differences.csv
    outcome_differences_statistics.txt
    <left>_vs_<right>_{histogram,kde,difference_histogram,difference_kde}.png
  s02_feature/
    feature_df.parquet, feature_df_head.csv, feature_manifest.json
  s03_divide/{outcome}/
    train_df.parquet, valid_df.parquet, test_df.parquet
    feature_head_df.parquet, outcome_head_df.parquet, statistics.txt
  s04_train/{outcome}/{model}/
    manifest.json, model.json, model_pretty.json, model.txt
    {partition}_key_df.{parquet,csv}, {partition}_feature_df.{parquet,csv}
    feature_importances.{parquet,csv}
  s05_predict/{outcome}/{model}/
    manifest.json, {model}_{partition}_predictions.{parquet,csv}
  s06_evaluate/{outcome}/
    {outcome}_display.csv, {model}_feature_importances.png
    {model}_{partition}/
      {model}_regression_metrics.json
      {model}_transform_binary_<threshold>_metrics.json
      *.png
```

## 4. Data contracts

### 4.1 Identity

`case_id` identifies one predicted unit and `case_date` is the split key.
`KEY_COLUMNS = (case_id, case_date)` is the canonical join, sort, and
alignment order; the tuple is never reversed.

### 4.2 Boundary schemas

Every frame that crosses a stage boundary is validated against a
`TableSchema`: ordered column rules, optional key-column requirements, optional
minimum row counts, and optional mutually exclusive column groups. `validate`
checks presence, exact dtype, key uniqueness, row count, and exclusivity, and
raises a boundary-`ValueError` naming the schema label and the failing rule.
Stages validate on ingress and egress.

### 4.3 Partitions

The canonical partitions are `train`, `valid`, and `test`, in that order. They
are disjoint, exhaustive, and non-empty. `tests/test_s03_divide.py` pins the
boundary dates and the partition-size invariant.

### 4.4 Manifests

| Manifest | Schema version | Produced by |
| --- | --- | --- |
| `FeatureManifest` | `1` | `s02_feature` |
| `ModelManifest` | `2` | `s04_train` |
| `PredictionManifest` | `1` | `s05_predict` |

Each kind records `schema_version`, `key_columns`, row counts, partitions, and
the run's declared artifacts. Reads reject an unknown version.

### 4.5 Registry tables

Four immutable tables drive the pipeline:

| Registry | Module | Role |
| --- | --- | --- |
| Outcomes and comparisons | `src/project/outcomes.py` | Which outcomes run, in which order, with which transforms; which pairs the analysis stage differences. |
| Model strategies | `src/project/models.py` | Which feature policy, training mechanic, and objective each model run uses. |
| Stage table | `src/project/pipeline.py` | Directory templates, artifacts, prerequisites, and lazy entry-point module paths. |
| External input tables | `src/project/inputs.py` | Logical names, filename tokens, required metadata columns, and boundary schemas. |

Each registry is a read-only mapping or tuple. Iterating a registry reproduces
the pipeline's run order. A mutation attempt raises instead of silently
changing a stage path, manifest name, or fan-out.

## 5. Design decisions

### Decision: Keep a three-layer split with a one-way dependency

- **Context:** Reusable machinery must be separable from project-specific data
  derivation.
- **Decision:** `src/core/` is project-agnostic, `src/project/` is the
  extension layer, and `src/stages/` contains thin entry points. Core types the
  project configuration through a protocol and never imports the project or
  stage layers.
- **Consequences:** (+) The engine is testable without project data; a
  project can be replaced without touching core. (−) Configuration has two
  declarations (protocol and dataclass) that must stay in sync; a conformance
  test pins them.

### Decision: Make the stage table and manifests the artifact contract

- **Context:** Stages fan out over outcomes and models, and consumers must not
  infer identity from filenames or directory depth.
- **Decision:** Declare each stage's directory template, artifacts, and
  prerequisites once. Declare each run's identity and artifact inventory in a
  manifest. Resolve stages lazily through dotted module paths.
- **Alternatives considered:** Hard-coded path helpers per stage (duplicated
  naming rules); filesystem globbing (fragile under a changed stage table).
- **Consequences:** (+) One place to change a path or artifact name; imports
  stay acyclic; an incomplete run fails loudly. (−) The stage table is a
  central file that must be updated when a stage is added.

### Decision: Pass a frozen context and an explicit output root

- **Context:** Module-level path constants and `Path.cwd()` defaults make a
  pipeline non-reproducible across working directories.
- **Decision:** Every stage receives a frozen `PipelineContext` (layout plus
  configuration) and an explicit output root. Paths stay relative until the
  caller joins them. Directory creation happens only through `context.ensure`.
- **Consequences:** (+) Tests inject temporary roots; the entry point is
  cwd-independent. (−) Every stage entry point must accept the context and
  output root, even when it uses only one.

### Decision: Validate every stage boundary

- **Context:** A wrong-but-matching input file or a misaligned join can
  propagate silently through several stages.
- **Decision:** Describe every boundary frame with a `TableSchema` and call
  `validate` at ingress and egress. Resolve external files through an
  exact-one registry instead of guessing filenames.
- **Alternatives considered:** Ad-hoc assertions inside each stage
  (inconsistent labels and coverage); relying on the type system alone
  (dataframe columns are runtime values).
- **Consequences:** (+) Failures name the boundary and the rule; schemas are
  testable in isolation. (−) Schema changes require a matching test update.

### Decision: Keep `src/project/adapters.py` as the only required user edit

- **Context:** Users need to supply their own outcome and feature
  calculations without rewriting orchestration.
- **Decision:** `build_outcome_table` and `build_feature_table` are the
  documented replacement point. The stage modules read inputs, call the
  adapters, validate, and write artifacts.
- **Consequences:** (+) A new project replaces two functions and its
  registries; the stage graph is unchanged. (−) The adapters must honor the
  canonical key columns and the key-alignment contract.

### Decision: Train regression strategies and threshold for binary metrics

- **Context:** The template needs both regression and binary reporting without
  a second objective or estimator adapter.
- **Decision:** Every model strategy uses the `reg:absoluteerror` objective.
  Binary labels are derived at evaluation time with
  `threshold_labels(value, threshold)`, where the positive class is strictly
  greater than the threshold.
- **Alternatives considered:** A separate binary objective and adapter
  (additional estimator surface); a per-strategy objective switch (more
  configuration for no template benefit).
- **Consequences:** (+) One training path and one adapter family. (−) Binary
  metrics are only as meaningful as the thresholded regression output;
  classification objectives are a future extension.

### Decision: Ship a deterministic demo and an exact golden baseline

- **Context:** The repository must run and be verifiable after a clean clone
  without external data.
- **Decision:** Commit a generated source table, seed every generator
  explicitly, seed the global estimator RNG once in the test configuration,
  pin `OMP_NUM_THREADS=1` and the estimator threads, and commit a baseline of
  manifests, metrics JSON, statistics text, and display CSVs. Compare the
  baseline byte for byte.
- **Alternatives considered:** A live external dataset (not reproducible);
  excluding estimator-dependent output entirely (loses end-to-end coverage).
- **Consequences:** (+) The full flow is a regression test. (−) The baseline is
  sensitive to intentional formatting changes and must be recaptured only with
  a reviewed reason; parquet and PNG bytes are excluded from the baseline.

### Decision: Resolve stage entry points lazily

- **Context:** The stage table imports from the project layer, while stages
  import the project layer back. Eager imports would create cycles and pull the
  estimator library into registry-only commands.
- **Decision:** Store each entry point as a dotted module path and import it
  only when the stage executes.
- **Consequences:** (+) The planner, the `--list` mode, and the `--dry-run`
  mode import no stage module. (−) A wrong module path fails at execution, not
  at import; the registry-consistency check and tests cover the table.

## 6. Reproducibility and idempotency

- The committed demo source table is the only external input.
- Each demo adapter owns a private `numpy.random.default_rng(demo_seed)` and
  never touches the global RNG. Feature policies and the estimator draw from
  the global `numpy` RNG, seeded once in `tests/conftest.py` and before every
  golden run. A bare entry-point run uses the process's initial RNG state for
  feature-policy noise; byte-level reproducibility is a property of the test
  harness.
- `OMP_NUM_THREADS=1` is set on environment activation and passed explicitly
  in acceptance commands; the estimator uses `nthread=1`.
- Plot helpers close every figure they create and write their declared file
  even when the data is insufficient for a draw.
- A stage replaces only its own declared artifacts, using a temporary sibling
  and an atomic replace. Re-running a stage is idempotent; it never deletes
  another stage's output.
- `tests/golden/diff.py` compares the committed baseline byte for byte on a
  second run.

## 7. Testing strategy

| Tier | Coverage |
| --- | --- |
| Unit | Every core module and every project registry, schema, adapter, and helper. |
| Property-based | Manifest write/read round trips, partition-size sums, and tokenizer canonicalization. |
| Integration | Each stage on the committed demo artifacts, the training driver, and the entry point. |
| Smoke | Every stage, every model strategy, and the registry-driven `s03` → `s06` run. |
| Golden | Baseline capture, alias selection, and exact diff on a fresh demo run. |
| Contract | Import direction, registry consistency, and configuration conformance. |

The suite is part of the default `pytest` run, including the vocabulary gate
test. Golden recapture is a deliberate, reviewed operation, never an automatic
way to hide drift.

## 8. Quality gates

Every change must pass:

```bash
pixi run ruff check .
pixi run ruff format --check .
pixi run pyright
pixi run pytest -q
```

The repository also ships a commit-time vocabulary gate. It hashes tracked
text, path components, commit messages, and text columns of committed parquet
files, then compares the digests with a committed hash list. The plain term
list is local and untracked. Run it before committing:

```bash
pixi run python scripts/scrub_repo.py --worktree
pixi run python scripts/scrub_repo.py --staged
```

## 9. Operational considerations

- **Execution model:** A single-process batch run. There is no service,
  network listener, credential, or remote dependency at runtime.
- **Inputs:** External tables are local parquet files resolved under
  `data_root`. Missing, ambiguous, or schema-invalid inputs fail at ingestion.
- **Outputs:** The output root is caller-supplied. Stages create only their
  own directories and replace only their own declared artifacts.
- **Observability:** Artifacts are the record: manifests carry run identity
  and configuration; statistics text and metrics JSON are stable and
  human-readable; the display table compares models across a partition.
- **Failure behavior:** Boundary validation, manifest version checks, missing
  artifacts, and registry inconsistencies raise instead of falling back to a
  partial or guessed result.
- **Secrets:** The template requires no secrets. Do not commit credentials or
  local paths.

## 10. Limitations and future work

Known limitations of the template as shipped:

- Only one external input table is registered in the demo.
- Only one estimator family (`xgboost`) is registered.
- Every strategy is a regression strategy; binary reporting is derived by
  thresholding regression values.
- Outcome payloads are numeric only. Elapsed-duration outcomes are not
  supported.
- Feature selection uses explicit policies; there is no feature-family or
  prefix/tier registry.
- The holdout contract is date-based `train`/`valid`/`test`; there is no
  cross-validation partition.

Future extensions, in rough order of value:

1. A generic feature-family registry with longest-prefix attribution and
   as-of tiers, replacing explicit feature subsets.
2. Elapsed-duration outcome columns and a numeric-or-duration schema helper.
3. Additional estimator adapters registered through
   `src/core/estimator.py`, plus a real binary classification objective and
   task dispatch.
4. A CI workflow that runs the test suite, the full demo, and the history
   vocabulary check.
5. Packaging as an installable library with a console entry point.
6. A profiling or dry-run execution surface for large runs.
7. A generic cross-validation partition in addition to the holdout contract.
