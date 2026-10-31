# Plan — Extract a Reusable Machine-Learning Pipeline Template

**Status:** Proposed. No implementation code written.

**Source:** An external, separately versioned repository whose pipeline code has
already been split into a project-agnostic layer and a project-specific layer.
Its location is supplied with the task and is deliberately **not** recorded in
any file in this repository. Do not add it as a remote, copy its history, or
copy its prose.

**Deliverable:** A new repository whose committed history contains only generic
machine-learning pipeline vocabulary. Every artifact, module, fixture, test,
docstring, comment, and commit message must pass the vocabulary gate in §6
before it is committed.

**Prerequisites (must exist before step 1):**

1. The source repository is available at a path supplied out of band (the path
   is never written into this repository).
2. `.scrub/denylist.txt` is **derived from the source tree's vocabulary** by
   `scripts/build_denylist.py --derive` and reviewed by a human in step 3,
   before any ported code is committed. It is untracked and gitignored. The
   derivation procedure and review requirement are in §6.1a. Until that file
   exists and has been reviewed, the §6 gate fails closed and no ported code
   may be committed; steps 1–2 only add tooling and are verified retroactively
   with `--history` after step 3.
3. Python is pinned to **3.12** (verified locally: `/usr/bin/python3` is
   3.12.3; `pixi` is 0.71.2). All commands run through `pixi run`; the bare
   `python` command is not assumed to exist.

---

## 1. Goal and hard invariants

Build a reusable template for a batch machine-learning project with this flow:

```text
external source table(s)
        │
        ▼
s01_outcome   derive the outcome table (outcome variables)
        │
        ├──────────────► s01b_outcome_analysis (descriptive analysis)
        ▼
s02_feature   derive the feature table (predictor variables)
        │
        ▼
s03_divide    join outcomes + features, split by case date
        │
        ▼
s04_train     train one or more model strategies
        │
        ▼
s05_predict   generate predictions per model run
        │
        ▼
s06_evaluate  metrics, plots, feature importances, display table
```

The template ships a deterministic synthetic demo so the whole flow runs with
no external data. Users replace the demo producers in `src/project/adapters.py`
with their own outcome and feature calculations.

### Invariants

1. **No subject-matter vocabulary in any commit.** The commit-time gate is the
   hash-based denylist in §6; the §4 table is the naming contract for new code
   and the reference used to derive that denylist. See §4 and §6.
2. **`src.core` never imports `src.project` or any stage module.** Enforced by
   `tests/test_module_imports.py`.
3. **Every stage receives a frozen `PipelineContext` and an explicit output
   root.** No module-level path constants, no `Path.cwd()` defaults, no
   import-time directory creation.
4. **The stage table and per-run manifests are the artifact contract.** No
   consumer parses a filename or directory depth to recover identity.
5. **The repository is runnable after a clean clone:** `pixi run python -m src.main`
   produces a complete `output/` tree, and `pixi run pytest -q` is green.
   `src/main.py` ends with an `if __name__ == '__main__': main()` guard so that
   `-m src.main` executes it.
6. **The demo is deterministic.** Reproducibility requires all of: the
   committed source table; a fixed RNG seed applied before any draw; the demo
   adapters using `numpy.random.default_rng(config.demo_seed)`; the estimator
   drawing from the global `numpy` RNG seeded once in `tests/conftest.py`; and
   `OMP_NUM_THREADS=1` plus the estimator's `nthread=1` set **before numpy or
   the estimator library is imported**. §8.4 specifies each mechanism.
7. **Execution is idempotent.** Re-running a stage overwrites its own declared
   artifacts atomically (write to a temporary sibling, then replace). A stage
   never deletes artifacts it does not own. §7.3 states the rule per stage.
8. **Quality gates run every step:** lint (`ruff check`), format check
   (`ruff format --check`), type check (`pyright`), and the test suite. Adding
   `xgboost` or `polars` requires a dependency audit record (§9 step 2).

---

## 2. Source inventory by role

The source tree is triaged by responsibility, not by subject. Use this table to
decide what to port, rewrite, or leave behind. Before porting, capture a
per-module inventory of every file in this table: module path, public
signatures, and the recorded dependency versions. That inventory is untracked
work product under `.port-staging/inventory.md`; it makes the dispositions
below verifiable and prevents a "port" from silently inventing an interface.

| Source role | Disposition |
| --- | --- |
| Project-agnostic path/layout resolution, context, and manifest kinds | **Port** with docstring scrub and the config-protocol edit in §7.1 |
| Project-agnostic metrics, binary metrics, plotting, serialization, binning, partition contract | **Port** with label scrub |
| Project-agnostic estimator protocol, estimator registry, and the shipped estimator adapter | **Port** |
| Project-agnostic training mechanics (`fit_single`, `run_param_search`) | **Port** |
| Project-layer configuration, common column vocabulary, outcome registry, transforms, and comparison registry | **Rewrite** generically from the source's shape |
| Project-layer stage table, prerequisite graph, orchestration registry, and run-stage planner | **Rewrite** generically, preserving the design |
| Project-layer boundary schema validator and external-input registry | **Rewrite** generically with one demo input table |
| Project-layer artifact I/O, split loading, model-manifest producer, feature-importance I/O | **Rewrite** generically |
| Project-layer training-data preparation and training driver | **Rewrite** generically; move estimator-specific factories out (§7.2) |
| Project-layer feature policies and model strategy table | **Rewrite** generically; drop the prefix-family/tier registry |
| Subject-matter outcome producers, feature producers, and analysis calculations | **Replace** with demo adapters and documented stubs |
| Subject-matter helper modules and stage subpackages | **Do not port** |
| Core and project tests | **Triage**: port generic tests, rewrite project tests, drop subject-matter tests |
| Architecture, agent, and user documentation | **Rewrite from scratch**; never copy prose |

The source's domain-specific stage subpackages and helper modules are not
listed here by name. They are identified during the port by their imports and
their vocabulary; anything that fails the §6 gate is not portable.

---

## 3. Destination layout

```text
.
├── .githooks/
│   └── pre-commit
├── .gitignore
├── AGENTS.md
├── README.md
├── data/
│   └── demo/
│       └── source_table_1.parquet
├── docs/
│   ├── ARCHITECTURE.md
│   └── research/
│       ├── README.md
│       └── plan-extract-generic-ml-pipeline.md
├── pixi.lock
├── pixi.toml
├── pyproject.toml
├── scripts/
│   ├── __init__.py
│   ├── build_denylist.py
│   ├── make_demo_data.py
│   ├── run_stage.py
│   └── scrub_repo.py
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── paths.py
│   │   ├── manifest.py
│   │   ├── metrics.py
│   │   ├── binary_metrics.py
│   │   ├── plots.py
│   │   ├── io.py
│   │   ├── binning.py
│   │   ├── splits.py
│   │   ├── estimator.py
│   │   ├── training.py
│   │   └── estimators/
│   │       ├── __init__.py
│   │       └── xgboost.py
│   ├── project/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── columns.py
│   │   ├── outcomes.py
│   │   ├── schemas.py
│   │   ├── inputs.py
│   │   ├── artifacts.py
│   │   ├── estimators.py
│   │   ├── training.py
│   │   ├── features.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── run_stage.py
│   │   └── adapters.py
│   └── stages/
│       ├── __init__.py
│       ├── s01_outcome.py
│       ├── s01b_outcome_analysis.py
│       ├── s02_feature.py
│       ├── s03_divide_data.py
│       ├── s04_train.py
│       ├── s05_predict_model.py
│       ├── s05_predict_model_main.py
│       ├── s06_evaluate_model.py
│       └── s06_evaluate_model_main.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── fixtures.py
    ├── test_vocabulary_scrub.py
    ├── vocabulary_denylist.sha256
    ├── test_core_*.py
    ├── test_manifest.py
    ├── test_splits.py
    ├── test_paths.py
    ├── test_context.py
    ├── test_contracts.py
    ├── test_module_imports.py
    ├── test_project_*.py
    ├── test_outcomes.py
    ├── test_outcome_column_vocabulary.py
    ├── test_columns.py
    ├── test_schemas.py
    ├── test_inputs.py
    ├── test_core_estimators.py
    ├── test_project_estimators.py
    ├── test_core_training.py
    ├── test_project_training.py
    ├── test_features.py
    ├── test_model_strategies.py
    ├── test_run_stage.py
    ├── test_adapters.py
    ├── test_s01_*.py
    ├── test_s01b_*.py
    ├── test_s02_*.py
    ├── test_s03_*.py
    ├── test_s04_*.py
    ├── test_s05_*.py
    ├── test_s06_*.py
    ├── test_smoke*.py
    ├── test_property_*.py
    ├── test_golden_*.py
    └── golden/
        ├── __init__.py
        ├── aliases.py
        ├── capture.py
        ├── diff.py
        ├── paths.py
        ├── runner.py
        ├── baseline/
        │   └── output/...
        └── fixtures/
            └── (only if a staged fixture is smaller than a live demo run)
```

No `src/common/`. Ingestion belongs in `src/project/inputs.py`; stage modules
belong in `src/stages/`.

`.gitignore` must be created before `git init` and must contain at least:

```text
# local vocabulary work product — never committed
.scrub/
.port-staging/

# environment and tooling
.pixi/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.pyright/

# runtime output — leading slash so tests/golden/baseline/output/ stays tracked
/output/
```

The leading `/` on `/output/` is mandatory: a bare `output/` pattern would also
ignore the committed `tests/golden/baseline/output/` tree.

---

## 4. Destination vocabulary

This is the complete naming contract. Add nothing outside it without updating
this section and the §6 denylist.

| Concept | Name |
| --- | --- |
| A row / unit of prediction | `case` |
| Case identity column | `case_id` |
| Case date column (split key) | `case_date` |
| Canonical key columns | `KEY_COLUMNS = (CASE_ID, CASE_DATE)` |
| Outcome / predicted variable | `outcome_<n>` |
| Predictor variable | `feature_<n>` |
| Raw source value (demo only) | `value_<n>` |
| Demo source table | `source_table_1` |
| Model identity | `model_00_signal_0`, `model_00_signal_1`, `model_00_noise_0`, `model_00_noise_1`, `model_01a`, `model_01b`, `model_02_subset_a`, `model_02_subset_b` |
| Holdout partitions | `train`, `valid`, `test` |
| Stages | `s01_outcome`, `s01b_outcome_analysis`, `s02_feature`, `s03_divide`, `s04_train`, `s05_predict`, `s06_evaluate` |
| Stage output root | `output/s<stage>/...` |
| Model run directory | `output/s04_train/{outcome}/{model}/` |
| Prediction run directory | `output/s05_predict/{outcome}/{model}/` |
| Evaluation result directory | `output/s06_evaluate/{outcome}/{model}_{partition}/` |

### Outcome registry (demo)

| Name | Purpose | Modeled? | Transform |
| --- | --- | --- | --- |
| `outcome_1` | primary regression outcome | yes | `identity` |
| `outcome_2` | analysis counterpart | no | `identity` |
| `outcome_3` | secondary regression outcome | yes | `identity` |
| `outcome_4` | analysis counterpart | no | `identity` |

`s01b` compares `(outcome_1, outcome_2)` and `(outcome_3, outcome_4)`. The
comparison direction is always `left - right`; no subject-matter sign rule is
encoded.

### Feature registry (demo)

`feature_1` … `feature_8`. Feature policies select or substitute these columns
by name. No prefix/tier family registry is ported in v1 (see §5.4).

---

## 5. Design decisions for the extraction

### 5.1 Keep the three-layer split

- `src/core/` — reusable machinery. Port and keep stable.
- `src/project/` — the extension point. Users edit this.
- `src/stages/` — thin stage entry points. Users mostly leave these alone;
  the outcome/feature producers they call live in `src/project/adapters.py`.

This preserves the source's strongest property: the reusable engine has no
dependency on the project layer, enforced by an AST-based import test.

### 5.2 Keep the declarative stage table and manifests

Port `StageSpec`, `PipelineLayout`, `PipelineContext`, `ModelManifest`,
`PredictionManifest`, `FeatureManifest`, structural run discovery, and the
`STAGE_REGISTRY` / `run_stage` planner. Rename the source's per-model binding
flag to `modeled_outcome_only`; the flag means "run this stage only for
outcomes whose registry entry is modeled."

### 5.3 Keep boundary schemas and the external-input registry

Port the `validate(frame, schema, label)` boundary error wrapper. Keep one
generic demo input table (`source_table_1`) in `INPUT_TABLES` so the eager and
lazy ingestion paths are exercised. Remove every subject-matter input spec.

### 5.4 Deliberate simplifications versus the source

These are scope reductions, not behavior changes to the retained machinery:

1. **Drop the prefix/tier feature-family registry.** The source's family
   registry is useful but carries subject-matter constants. v1 ships
   `IdentityFeatures`, `OutcomePlusNoise`, `PureNoise`, and `FeatureSubset`
   (an explicit column list). A generic family registry is a documented future
   extension (§13).
2. **Simplify `FeatureManifest`.** The canonical field list is
   `schema_version`, `key_columns`, `n_rows`, `partitions`, `feature_columns`,
   `categorical_columns`, `metadata_columns`, `artifacts`. Set
   `FEATURE_SCHEMA_VERSION` to `1` in the new repo (it starts fresh; no
   migration is needed) and reject any other version on read. `MODEL_SCHEMA_VERSION`
   and `PREDICTION_SCHEMA_VERSION` keep the source values. §5.4 and §7.1 agree
   on this list; there are no additional manifest fields.
3. **Numeric-only outcome payloads.** The source supports elapsed-time
   (`Duration`) outcome columns. v1 accepts numeric outcomes only and documents
   the extension point in `schemas.py`. This keeps the demo and the metric
   layer simple.
4. **Move estimator-specific construction out of the training driver.** Add
   `src/project/estimators.py` with the single-fit adapter factory, the search
   adapter factory, and the booster-normalization helper. `src/project/training.py`
   then contains only estimator-agnostic flow, mirroring the core/project split.
5. **Make the context injectable.** `build_context(config: ProjectConfig = CONFIG)`
   so tests can supply a temporary data root without patching module globals.
6. **Reduce demo spend.** Use smaller estimator search settings than the source
   so the full suite and the golden run stay fast. Document the values as
   template defaults.

### 5.5 Demo adapters are the user extension point

`src/project/adapters.py` is the only file most users need to replace:

```python
def build_outcome_table(source_df, config): ...
def build_feature_table(source_df, outcome_df, config): ...
```

`s01_outcome.py` and `s02_feature.py` are thin: resolve paths, read the demo
source table through the input registry, call the adapter, validate the frame,
write the artifact and manifest. Replace the adapter body; leave the stage
plumbing intact.

---

## 6. Vocabulary scrub protocol

This is a hard gate, not a style check. Run it before **every** commit.

### 6.1 Denylist representation

The forbidden terms must not appear in any tracked file, including the test
that enforces the gate. Store only `sha256` hashes.

- `.scrub/denylist.txt` — **untracked and gitignored**. One normalized term or
  phrase per line, derived from the source tree's subject-matter vocabulary by
  §6.1a. This file is never committed.
- `tests/vocabulary_denylist.sha256` — committed. One lowercase hex digest per
  line, sorted. Contains no terms.
- `.port-staging/` — **untracked and gitignored**. Copy source candidates here,
  rename and scrub them here, then move the cleaned files into `src/`. Never
  copy a raw source file directly into a tracked path.

### 6.1a Deriving the denylist from the source tree

The source path is supplied out of band and is never recorded in this
repository. Derivation is a one-time, human-reviewed step performed before
step 1:

1. `pixi run python scripts/build_denylist.py --derive <source-path>`.
2. The script uses the §2 inventory to collect identifiers, file and directory
   stems, fixture column names, and docstring nouns from the source tree,
   subtracts the §4 vocabulary and a small generic-programming allowlist
   (`test`, `data`, `frame`, `path`, `value`, `config`, and similar), and writes
   candidate terms to `.scrub/denylist.candidates`, sorted and de-duplicated.
3. A human reviews the candidates, deletes false positives, and saves the
   result to `.scrub/denylist.txt`.
4. `pixi run python scripts/build_denylist.py --hashes` writes
   `tests/vocabulary_denylist.sha256` from the reviewed file.

Fail-closed rules: if `.scrub/denylist.txt` is missing, empty, or tracked by
Git, both `--derive` and `--hashes` exit non-zero and `scrub_repo.py` exits
non-zero with `denylist unavailable`. An empty denylist is never treated as a
clean result.

### 6.2 Scripts

`scripts/build_denylist.py`

1. Refuse to run if `.scrub/denylist.txt` is tracked by Git.
2. Normalize each line (lowercase, collapse whitespace).
3. Write `sha256(normalized_term)` to `tests/vocabulary_denylist.sha256`,
   sorted, one per line.

`scripts/scrub_repo.py`

Modes:

- `--worktree` — inspect tracked files plus untracked files that are not
  ignored.
- `--staged` — inspect the index (`git ls-files --stage` + `git cat-file`) and
  staged path names.
- `--history` — inspect every commit reachable from `--all`: commit message,
  tree path names, and blob contents. Used before the first push and at the
  final acceptance gate.

Inspection rules:

1. Tokenize text on non-alphanumeric characters; lowercase. Split camelCase
   and PascalCase boundaries before lowercasing so `fooBar` becomes `foo`,
   `bar`.
2. Form n-grams by joining the constituent tokens with a **single space**
   (canonical form), then hash the canonical string. `build_denylist.py`
   hashes each normalized entry the same way, including multi-token phrases.
3. Check unigrams, bigrams, and trigrams of the text against the hash set.
4. Check the normalized file stem and every directory component (same
   tokenizer and canonical n-grams).
5. For `.parquet`, check column names and string-column values through
   `polars`; do not decode numeric payloads.
6. For other binary files, fail closed unless the path is on a tiny, hashed
   allowlist (e.g. compiled caches are gitignored, not committed).
7. On a hit, print `path:line: fingerprint=<first 8 hex>` only. Do not print
   the matched term; look it up in the untracked source list.
8. Exit `1` on any hit or when the denylist is unavailable. Exit `0` only when
   the inspection is clean.

### 6.3 Test and hook

- `tests/test_vocabulary_scrub.py` imports the tokenizer from
  `scripts/scrub_repo.py`, walks `git ls-files`, and fails with the fingerprint
  and location. It is part of the normal suite.
- `.githooks/pre-commit` runs `pixi run python scripts/scrub_repo.py --staged`
  and refuses the commit on failure.
- Installation is a one-time local step:
  `git config core.hooksPath .githooks`.
- `git config core.hooksPath` is per-clone; §11 therefore does not treat the
  hook as proof of a clean commit. The test suite and the explicit
  `--worktree`/`--staged` checks are authoritative.

### 6.4 Commit gate checklist

Before staging:

- [ ] The change was developed in `.port-staging/` or written from scratch.
- [ ] No source file with a subject-matter name was copied in unchanged.
- [ ] Docstrings, comments, identifiers, filenames, fixture columns, and test
      names use only §4 vocabulary.
- [ ] `pixi run python scripts/scrub_repo.py --worktree` exits `0`.
- [ ] After staging, `pixi run python scripts/scrub_repo.py --staged` exits `0`.
- [ ] The commit message uses only §4 vocabulary.

If a violating commit is created, rewrite it before doing anything else. The
repository is local and unpublished during extraction; `git commit --amend`,
`git rebase`, or `git reset` is acceptable. Never leave a violating commit in
history and never push before `--history` is clean.

---

## 7. Module-by-module port plan

### 7.1 `src/core/`

| File | Action | Notes |
| --- | --- | --- |
| `core/paths.py` | Port | Keep a `ProjectConfig` **Protocol** here whose field names mirror the project dataclass in §7.2 (no methods, no project imports). Keep `{outcome}` validation, `StageSpec`, `PipelineLayout`, `PipelineContext`, `ensure`. Scrub docstrings. |
| `core/manifest.py` | Port + simplify | Keep `ArtifactRef`, `ModelManifest`, `PredictionManifest`, `write_manifest`, `read_manifest`, `declared_artifact_path`, `load_model_runs`, `load_prediction_runs`. Replace `FeatureManifest` with the §5.4 shape. Re-tuple `feature_columns`, `categorical_columns`, `metadata_columns`, `key_columns`, `partitions`, `artifacts`. |
| `core/metrics.py` | Port | `RegressionMetrics` is generic. Scrub field names only if a subject-matter noun appears. |
| `core/binary_metrics.py` | Port | Keep `BinaryClassificationMetrics`, `BestThresholdMetric`, `get_binary_classification_metrics`, and the plot helpers. Scrub axis labels. |
| `core/plots.py` | Port | Keep figure ownership: every helper creates, decorates, saves, and closes its own figure. Scrub titles/labels. |
| `core/io.py` | Port | Keep `save_dataframe_to_csv_and_parquet` and `save_full_dataframe_to_parquet_and_csv`. |
| `core/binning.py` | Port | Pure helper, no changes beyond docstring scrub. |
| `core/splits.py` | Port | Replace the source's key-pair docstring examples with `(case_id, case_date)`. Keep `DataPartition`, `PARTITION_ORDER`, `holdout_partitions`. |
| `core/estimator.py` | Port | Keep `EstimatorAdapter`, `EstimatorRegistry`, `estimator_registry`. |
| `core/estimators/xgboost.py` | Port | Keep the adapter and serialization. Verify no subject-matter attribute names remain. |
| `core/training.py` | Port | Keep `fit_single` and `run_param_search` estimator-agnostic. |

**Acceptance for this layer:** `tests/test_module_imports.py` proves
`src.core` imports no `src.project` and (once stages exist) no `src.stages`
module; the ported core tests pass. This test lands with step 3, not step 12.
While `src/stages/` is empty or absent the test asserts only the core→project
rule and does not fail on missing stage files.

### 7.2 `src/project/`

#### `config.py`

Frozen `ProjectConfig` and one module-level `CONFIG`. `ProjectConfig` is a
frozen dataclass in the project layer; the field names match the
`ProjectConfig` Protocol in `src/core/paths.py` so core never imports this
module. There is no `input_filenames` field: filenames are owned by the
`InputTableSpec` entries in `inputs.py` (single source of truth).

```python
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    data_root: str = 'data/demo'  # resolved against an explicit project_root
    feature_columns_to_drop: tuple[str, ...] = ()
    train_end_date: date = date(2024, 1, 1)
    test_start_date: date = date(2024, 7, 1)
    binary_thresholds: tuple[float, ...] = (-0.5, 0.0, 0.5)
    evaluated_partitions: tuple[str, ...] = ('train', 'valid')
    feature_importance_plot_top_n: int = 10
    demo_seed: int = 12345
    demo_case_count: int = 420
    xgb_single_fit_params: Mapping[str, Any] = field(default_factory=dict)
    xgb_num_boost_round: int = 40
    xgb_early_stopping_rounds: int = 5
    xgb_regressor_seed: int = 7
    xgb_search_seed: int = 11
    xgb_param_distributions: Mapping[str, Any] = field(default_factory=dict)
    xgb_search_n_iter: int = 4
    xgb_search_cv: int = 3


CONFIG = ProjectConfig()
```

`data_root` is a relative path and is **always** resolved against an explicit
`project_root` argument (`build_context(config=CONFIG, project_root=...)` or
`source_data_dir(project_root)`); no function resolves it against
`Path.cwd()`. Tests inject a temporary project root and a config with a
matching `data_root`. Mapping fields are read-only by convention; mutable
defaults use `field(default_factory=dict)` because dataclasses reject bare
`{}`.

Remove every source configuration field and directory helper that carries
subject-matter meaning. Keep the module-level
`LOG_PLOT_BASES`, `DISPLAY_REGRESSION_METRICS`, and `DISPLAY_METRIC_DECIMALS`
tables; they are generic reporting policy.

#### `columns.py`

```python
CASE_ID = 'case_id'
CASE_DATE = 'case_date'
KEY_COLUMNS = (CASE_ID, CASE_DATE)
FEATURE_COLNAME = 'feature'
FEATURE_ESTIMATOR_ID_COLNAME = 'feature_estimator_id'
FEATURE_IMPORTANCE_COLNAME = 'importance'
```

`KEY_COLUMNS` order is canonical and must never be reversed.

#### `outcomes.py`

Keep the registry shape: frozen `Outcome(name, outcome_transform,
reporting_backtransform, modeled)`, `OUTCOMES` insertion order = run order,
`modeled_outcomes()`, `outcome_names()`. Add `OutcomeComparison(left, right)`
and `OUTCOME_COMPARISONS` for `s01b`.

`resolve_transform` must have more than one real case, so register exactly two
forward/inverse pairs:

| Forward | Inverse | Domain rule |
| --- | --- | --- |
| `identity` | `identity` | any numeric |
| `log1p` | `expm1` | input must be `> -1`; raise a boundary error otherwise |

All four demo outcomes in §4 use `identity`. The `log1p`/`expm1` pair is
exercised only by `tests/test_outcomes.py` (forward/inverse round-trip and the
domain rejection); it is not used by the demo because `outcome_3` can be
negative. Document this explicitly so a future user knows a non-identity
transform must supply a matching inverse before enabling it.

#### `schemas.py`

Keep the `validate(frame, schema, label)` wrapper and its boundary-labelled
error preservation. Define:

- `SOURCE_TABLE_SCHEMA` — `case_id`, `case_date`, `value_1..value_3`.
- `S01_OUTCOME_SCHEMA` — keys + the four outcome columns, numeric.
- `S02_FEATURE_SCHEMA` — keys + numeric feature columns.
- `S03_SPLIT_SCHEMA` — keys + numeric payload.
- `S04_PARTITION_KEY_SCHEMA`, `S04_PARTITION_FEATURE_SCHEMA`,
  `S04_FEATURE_IMPORTANCES_SCHEMA`.
- `S05_PREDICTION_SCHEMA` — keys + `predictions`.

Drop all subject-matter input schemas. Numeric-only payloads; document the
elapsed-duration extension point in the module docstring without naming a
subject.

**Schema representation.** A schema is an ordered mapping
`column -> ColumnRule(dtype, nullable, unique=False, min=None, max=None)`, plus
optional table-level rules (`row_count`, `key_columns`). `validate` checks, in
order: presence of required columns, exact dtype, nullability, then the
constraint rules, and raises a `BoundaryError` that preserves `label` and the
failing column/rule. It does **not** coerce types or fill nulls; coercion is a
transformation concern.

Per-table constraints beyond dtype:

| Schema | Constraints |
| --- | --- |
| `SOURCE_TABLE_SCHEMA` | `case_id` non-null and unique; `case_date` non-null and strictly increasing; `value_1..value_3` non-null; `row_count > 0` |
| `S01_OUTCOME_SCHEMA` | keys as above, unique and non-null; all four outcomes non-null; row count preserved from the source table |
| `S02_FEATURE_SCHEMA` | keys unique and non-null; `feature_1..feature_8` non-null; one row per source row |
| `S03_SPLIT_SCHEMA` | keys unique within a partition; the single outcome column non-null; partition rows are a subset of the joined keys |
| `S04_*` | key columns unique; feature matrix has no nulls |
| `S05_PREDICTION_SCHEMA` | keys unique per partition; `predictions` finite (no NaN/inf) |

Each constraint above gets exactly one failure test asserting the boundary
label and the failing rule, matching §10.5.

#### `inputs.py`

Keep `InputTableSpec`, `INPUT_TABLES`, `resolve_input_files`,
`resolve_input_table_path`, `read_input_table`, `InputTableFilepaths`, and
`InputTables`. Port the lazy metadata-only reader as `inspect_input_table`
(the generic rename avoids the source's ambiguous verb). Ship exactly one
spec:

```python
INPUT_TABLES = {
    'source_table_1': InputTableSpec(
        logical_name='source_table_1',
        filename_tokens=('source_table_1',),
        required_columns=('case_id', 'case_date', 'value_1', 'value_2', 'value_3'),
        schema=SOURCE_TABLE_SCHEMA,
    ),
}
```

Keep the eager validated read and the lazy metadata-only inspection. Remove
the source's table-name sets and every subject-matter schema reference.

#### `artifacts.py`

Port `SplitTables`, `load_split_tables`, `save_results`,
`extract_feature_importances`, `load_feature_importances`, and the model
serialization writer. Read key columns from `src.project.columns`, not from a
caller argument. Remove subject-matter column literals. Keep manifest
production (`MODEL_SCHEMA_VERSION` remains `2`, `partitions` included,
`best_parameters` in the manifest only).

#### `estimators.py` (new)

Own the estimator-specific construction that the source kept in the training
driver:

- `ESTIMATOR_ID = 'xgboost'`
- `make_single_fit_adapter()`
- `make_search_adapter_factory(objective, regressor_seed)`
- `booster_of(adapter)`

`src/project/training.py` imports these and stays estimator-agnostic in flow.

#### `training.py`

Port `TVTFeatureOutcomeData`, `ModelReadyData`,
`divide_sort_train_valid_test_data`, `prepare_model_ready_data`, and
`train_model_run`. The driver:

1. Loads split tables through `load_split_tables`.
2. Applies `strategy.feature_policy.apply(...)`.
3. Prepares arrays.
4. Dispatches on `strategy.trainer` (`fit_single` / `run_param_search`).
5. Sets feature-name and outcome-name attributes on the booster.
6. Calls `save_results`.

Every setting comes from `context.config`. No module-level `CONFIG` import.
Preserve the source's documented numpy draw-order contract (the golden seed
depends on outcome order, partition order, and policy draw order).

#### `features.py`

Keep the `FeaturePolicy` Protocol (`apply`, `feature_colnames`) and:

- `IdentityFeatures` — pass through, full feature list in head order.
- `OutcomePlusNoise` — replace the first feature with the outcome plus a
  small seeded normal draw; keep the second feature; model order `[second,
  first]`.
- `PureNoise` — replace the first two features with pure seeded draws.
- `FeatureSubset` — select an explicit tuple of feature columns, preserving
  head order.

Drop the source's family-policy classes, their registry constants, and the
prefix registry.

#### `models.py`

Frozen `ModelStrategy(id, feature_policy, trainer, objective)` and this exact
eight-entry `MODEL_STRATEGIES` table (insertion order = run order):

| `id` | `feature_policy` | `trainer` | `objective` |
| --- | --- | --- | --- |
| `model_00_signal_0` | `IdentityFeatures` | `fit_single` | `reg:absoluteerror` |
| `model_00_signal_1` | `OutcomePlusNoise` | `fit_single` | `reg:absoluteerror` |
| `model_00_noise_0` | `PureNoise` | `fit_single` | `reg:absoluteerror` |
| `model_00_noise_1` | `FeatureSubset(('feature_1', 'feature_8'))` | `fit_single` | `reg:absoluteerror` |
| `model_01a` | `IdentityFeatures` | `run_param_search` | `reg:absoluteerror` |
| `model_01b` | `OutcomePlusNoise` | `run_param_search` | `reg:absoluteerror` |
| `model_02_subset_a` | `FeatureSubset(('feature_1', 'feature_2', 'feature_3'))` | `fit_single` | `reg:absoluteerror` |
| `model_02_subset_b` | `FeatureSubset(('feature_4', 'feature_5', 'feature_6', 'feature_7'))` | `fit_single` | `reg:absoluteerror` |

`TrainerKind` remains `Literal['fit_single', 'run_param_search']`. The
objective is a strategy fact; hyperparameters stay in `CONFIG`. The source's
`reg:absoluteerror` default is retained (open item resolved).

**Task and binary metrics.** Every strategy is a regression strategy, so every
manifest records `task='regression'`. `s06` produces binary classification
metrics by thresholding the evaluated partition's actual outcome and the
model's prediction at each value in `CONFIG.binary_thresholds`; the negative
class is `<= threshold`, the positive class is `> threshold`. This rule is
implemented once in `core/binary_metrics.py` as
`threshold_labels(values, threshold)` and is the only source of `task='binary'`
inputs. No `binary:*` objective is registered in v1 (§13 item 3).

#### `pipeline.py`

Port the design with generic names:

- `STAGES` as a mapping of stage name to a `StageDescriptor` that stores the
  **module path as a string** (for example `'src.stages.s03_divide_data'`) and
  the artifact declarations. Importing `pipeline.py` must not import a stage
  module; the descriptor is resolved lazily only when the stage is executed.
- `EXPLICIT_STAGES = ('s01_outcome', 's01b_outcome_analysis', 's02_feature')`
  and `REGISTRY_STAGES = ('s03_divide', 's04_train', 's05_predict',
  's06_evaluate')`. `STAGES` contains all seven descriptors;
  `defined_stages()` returns all seven names, and `explicit_stages()` returns
  the first three so `--list` can show the full graph while `run_pipeline`
  executes only `REGISTRY_STAGES`.
- `StageSpec.prerequisites` with `build_layout()` referential-integrity and
  cycle checks.
- `StageBinding` with `modeled_outcome_only` (renamed from the source's flag).
- `_standard_args` / `_strategy_args`.
- `_s04_train_bindings()` derived from `MODEL_STRATEGIES`.
- `run_pipeline(context, output_root)` iterating `OUTCOMES` order.
- `ProjectPaths` / `build_paths`.

Remove the source's subject-matter directory helpers. Keep one
`source_data_dir(project_root)` helper if useful.

`s01_outcome`, `s01b_outcome_analysis`, and `s02_feature` stay explicit calls
in `src/main.py`; s03–s06 are registry-driven.

#### `run_stage.py`

Port the importable planner: `defined_stages()`, `explicit_stages()`,
`model_ids()`, `allowed_outcomes()`, `check_registry_consistency()`,
`StageSelection`, `PlannedCall`, `plan_calls`, `run_stages`, `render_list`,
`render_plan`. Rename the modeled-outcome filter. Importing the planner must
not import a stage module or the estimator library.

#### `adapters.py` (new)

Demo producers and the documented extension point:

```python
def build_outcome_table(source_df: pl.DataFrame, config: ProjectConfig) -> pl.DataFrame: ...
def build_feature_table(source_df: pl.DataFrame, outcome_df: pl.DataFrame,
                        config: ProjectConfig) -> pl.DataFrame: ...
```

Use only deterministic arithmetic. Each adapter creates its own generator:
`build_outcome_table` uses `numpy.random.default_rng(config.demo_seed)` and
draws in the §8.2 order; `build_feature_table` uses a **fresh**
`numpy.random.default_rng(config.demo_seed)` and makes no random draws (all
§8.3 features are deterministic functions of the source columns). Adapters do
not read or mutate the global `numpy` RNG; the estimator is the only consumer
of the global RNG. The demo formulas are specified in §8.

Alignment contract: both adapters return frames sorted by `KEY_COLUMNS` and
carry `case_id` for every source row. `s02_feature` validates that the returned
feature keys equal the outcome keys exactly (same set, same order) before
writing, so the later join cannot drop or duplicate rows.

### 7.3 `src/stages/`

| File | Action |
| --- | --- |
| `s01_outcome.py` | Thin stage: resolve dir via layout, `context.ensure`, read `source_table_1` through `read_input_table`, call `build_outcome_table`, validate `S01_OUTCOME_SCHEMA`, write `outcome_df.parquet` + statistics text. Docstring names `src/project/adapters.py` as the replacement point. |
| `s01b_outcome_analysis.py` | Generic analysis: read and validate the outcome table, project `OUTCOME_COMPARISONS` into a wide difference table (`left - right`), write parquet + full CSV + statistics + four plots per comparison (overlaid histogram, overlaid KDE, difference histogram, difference KDE). Every plot helper saves its declared PNG even when data is insufficient. |
| `s02_feature.py` | Thin stage: read the source table and the outcome keys, call `build_feature_table`, assert its keys equal the outcome keys exactly (set and order), validate `S02_FEATURE_SCHEMA`, write `feature_df.parquet`, `feature_df_head.csv`, and `feature_manifest.json`. Docstring names the replacement point. |
| `s03_divide_data.py` | Port and genericize: for one modeled outcome, join that outcome column + features on `KEY_COLUMNS`, assert key uniqueness and row conservation, split on `case_date`, validate each partition, write `train_df` / `valid_df` / `test_df` + head frames + statistics under `s03_divide/{outcome}/`. Each `{outcome}` split contains the keys, that single outcome column, and the feature columns; outcomes are not duplicated across directories. Split boundaries are `train: case_date <= train_end_date`, `valid: train_end_date < case_date < test_start_date`, `test: case_date >= test_start_date`; assert `test_start_date > train_end_date` and that every partition is non-empty before writing. |
| `s04_train.py` | Port: one entry point per strategy id; create `s04_train/{outcome}/{strategy_id}` via `context.ensure`; delegate to `train_model_run`. |
| `s05_predict_model.py` | Port: manifest-driven `load_model_run`, `generate_and_save_predictions`, `save_prediction_run`. Read artifacts from the manifest inventory, never by globbing. |
| `s05_predict_model_main.py` | Port: discover s04 runs with `load_model_runs`, predict every declared partition, write `PredictionManifest`. |
| `s06_evaluate_model.py` | Port: metric/report writers, `partition_tables`, `get_model_tasks`, feature-importance ranking and plotting, regression and binary savers, display-table builder, log-scale helper. Binary metrics come from `threshold_labels(values, threshold)` (§7.2 `models.py` rule), not from a classification objective. Replace any subject-matter column naming with `context`-driven names. |
| `s06_evaluate_model_main.py` | Port: iterate evaluated partitions, join predictions to split tables, dispatch by manifest `task`, write per-model importance PNG, per-partition metrics + plots, and the per-outcome display CSV. |

**Stage responsibilities and idempotency.** Each stage is split conceptually into
three responsibilities even though it lives in one function: (1) *ingestion* —
read and validate through `src/project/inputs.py`, never mutate the incoming
frame; (2) *transformation* — the adapters, joins, splits, training, prediction,
and metric computation; (3) *output* — the artifact/manifest/plot/statistics
writers, which convert internal frames to the declared on-disk format. Writers
are the only code allowed to change representation (for example parquet ↔ CSV,
figures closed after save). Re-running a stage atomically replaces only the
artifacts that stage declares; it never deletes another stage's output and
never reads a previous run's file by glob or directory depth. `context.ensure`
creates directories but never creates files at import time.

### 7.4 `scripts/`

- `scripts/run_stage.py` — argparse wrapper over `src/project/run_stage.py`;
  `--stage`, `--outcome`, `--model`, `--output-root`, `--with-prereqs`,
  `--dry-run`, `--list`; planner `ValueError` → `parser.error` (exit `2`).
- `scripts/make_demo_data.py` — regenerate `source_table_1.parquet`
  deterministically from `CONFIG.demo_seed`. Accepts `--output <path>`
  (default `data/demo/source_table_1.parquet`). The resolved output path must
  remain inside the resolved `data/demo/` directory; otherwise exit `2`. This
  keeps the containment rule while letting determinism tests write to a
  `tmp_path` that still resolves under the repository's `data/demo/` (tests
  pass a temporary project root whose `data/demo/` is the containment base).
  Writes are atomic (temp sibling + replace).
- `scripts/build_denylist.py`, `scripts/scrub_repo.py` — §6.

### 7.5 `src/main.py`

```python
def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    context = build_context(config=CONFIG, project_root=project_root)
    output_root = project_root / 'output'
    s01_outcome(context, output_root)
    s01b_outcome_analysis(context, output_root)
    s02_feature(context, output_root)
    run_pipeline(context, output_root)


if __name__ == '__main__':
    main()
```

Docstring states that the first three are explicit because they read external
data or run once per run, and that s03–s06 are registry-driven.

---

## 8. Demo data contract

### 8.1 Source table

`data/demo/source_table_1.parquet`, 420 rows, columns:

| Column | Dtype | Derivation |
| --- | --- | --- |
| `case_id` | `String` | `case_{i:04d}` |
| `case_date` | `Date` | `2022-06-01 + 2*i days` |
| `value_1` | `Float64` | `sin(i/7) + 0.05 * rng.normal()` |
| `value_2` | `Float64` | `cos(i/11) + 0.05 * rng.normal()` |
| `value_3` | `Float64` | `value_1 - value_2 + 0.02 * rng.normal()` |

With 420 rows at two-day spacing, the dates span roughly 2022-06 to 2024-09,
so all three partitions are populated against
`train_end_date = 2024-01-01` and `test_start_date = 2024-07-01`.

Partition contract (the single definition used by `s03` and its tests):

| Partition | Predicate |
| --- | --- |
| `train` | `case_date <= train_end_date` |
| `valid` | `train_end_date < case_date < test_start_date` |
| `test` | `case_date >= test_start_date` |

Preconditions: `test_start_date > train_end_date`; every partition is non-empty
on the committed demo data. No date belongs to two partitions, and every source
row belongs to exactly one.

### 8.2 Outcomes (`build_outcome_table`)

```text
outcome_1 = 0.60 * value_1 + 0.30 * value_2 + 0.10 * rng.normal()
outcome_2 = value_1 - value_3
outcome_3 = value_1 * value_2 + 0.30 * value_3 + 0.05 * rng.normal()
outcome_4 = value_3
```

Both modeled outcomes are centered near zero with a spread wide enough that
the configured thresholds `(-0.5, 0.0, 0.5)` yield both classes in the train
and valid partitions. During implementation, assert that each evaluated
threshold produces at least one positive and one negative label; if a future
demo change breaks that, adjust the demo distribution or the thresholds rather
than weakening the metric code.

Binary labels are defined exactly as: `label = value > threshold` (strict),
where `value` is the partitioned actual outcome for actual labels and the
model prediction for predicted labels. Both sides use the same rule, and the
threshold list applies to the `train` and `valid` partitions in
`CONFIG.evaluated_partitions` only.

### 8.3 Features (`build_feature_table`)

```text
feature_1 = value_1
feature_2 = value_2
feature_3 = value_3
feature_4 = value_1 * value_2
feature_5 = value_1 ** 2
feature_6 = value_2 ** 2
feature_7 = value_1 + value_2 + value_3
feature_8 = (value_1 - value_2) / (1 + abs(value_3))
```

### 8.4 Determinism rules

There are three independent RNG consumers; each has exactly one mechanism:

1. **Demo source data** — `scripts/make_demo_data.py` uses
   `numpy.random.default_rng(CONFIG.demo_seed)` in fixed `i` order and writes
   parquet with explicit writer settings (`compression='zstd'`, statistics on).
   The pinned `polars` version is recorded in `pixi.lock`; a parquet writer or
   polars upgrade requires recapturing the baseline with a reviewed reason.
2. **Demo adapters** — each adapter creates a fresh
   `numpy.random.default_rng(CONFIG.demo_seed)` (§7.2 `adapters.py`). Outcome
   draws follow §8.2 order; feature construction makes no draws.
3. **Estimator** — the only consumer of the global `numpy` RNG, seeded once in
   `tests/conftest.py` (`numpy.random.seed(0)`) before the estimator is
   imported.

Environment: `OMP_NUM_THREADS=1` is set in the `pixi.toml` test/run
environments **and** in `tests/conftest.py` before importing `numpy`,
`polars`, or `xgboost`; the estimator is also constructed with `nthread=1`.
Setting the variable after import is not sufficient and must not be relied on.

The golden harness runs stages in registry order. PNGs and estimator
serializations are excluded from the golden diff; §14 records what the
baseline contains.

### 8.5 Expected output tree (demo)

```text
output/
  s01_outcome/outcome_df.parquet
  s01_outcome/outcome_statistics.txt
  s01b_outcome_analysis/...
  s02_feature/feature_df.parquet
  s02_feature/feature_df_head.csv
  s02_feature/feature_manifest.json
  s03_divide/{outcome}/{train,valid,test}_df.parquet
  s03_divide/{outcome}/statistics.txt
  s04_train/{outcome}/{model}/manifest.json + model + split tables + importances
  s05_predict/{outcome}/{model}/manifest.json + per-partition predictions
  s06_evaluate/{outcome}/{model}_{partition}/metrics + plots
  s06_evaluate/{outcome}/{model}_feature_importances.png
  s06_evaluate/{outcome}/{outcome}_display.csv
```

The committed golden baseline (decided in §14) contains only the manifests,
metrics JSON, statistics text, and display CSVs from this tree. It does **not**
contain split-table parquet files, PNGs, or estimator serializations.

---

## 9. Commit-by-commit implementation plan

Each row is one small, independently verifiable commit. Run the §6 gate before
every commit **except the bootstrap commits 1–3** that introduce the tooling
and the gate itself. Those three are verified retroactively with
`scrub_repo.py --history` immediately after step 3; from step 4 onward the gate
is mandatory. Do not batch steps.

Every verification also runs, and must pass:

```bash
pixi run ruff check .
pixi run ruff format --check .
pixi run pyright
```

plus the named pytest command. A step is not complete until all of these are
clean. Verification commands use `pixi run`; the bare `python` executable is
not assumed to exist.

| # | Commit message | Files | Verification |
| --- | --- | --- | --- |
| 0 | *(not a commit)* | `git init`; create `.gitignore` (§3), `.scrub/`, `.port-staging/`; `git config core.hooksPath .githooks` | `git status` shows only intended untracked setup; `.scrub/` and `.port-staging/` are ignored |
| 1 | `chore: add repository tooling skeleton` | `.gitignore`, `pyproject.toml`, `pixi.toml`, `README.md` stub, `docs/research/README.md`, package `__init__.py` files | `pixi install`; `pixi run python -c "import src"` |
| 2 | `chore: pin the development environment` | `pixi.lock`; untracked `.port-staging/audit.md` dependency-audit record | `pixi install`; `pixi run ruff --version`; `pixi run pyright --version`; audit recorded |
| 3 | `feat: add vocabulary scrub gate` | `scripts/__init__.py`, `scripts/build_denylist.py`, `scripts/scrub_repo.py`, `.githooks/pre-commit`, `tests/test_vocabulary_scrub.py`, `tests/vocabulary_denylist.sha256` | derive `.scrub/denylist.txt` from the source (§6.1a), human review, `--hashes`; `pixi run pytest tests/test_vocabulary_scrub.py -q`; immediately after commit run `scrub_repo.py --history` |
| 4 | `docs: add extraction plan` | this plan | `scrub_repo.py --worktree` and `--staged` exit `0` |
| 5 | `feat: port core path and context primitives` | `src/core/paths.py`, `tests/test_paths.py`, `tests/test_context.py`, `tests/test_module_imports.py` | named tests green; import test tolerates an absent `src/stages/` |
| 6 | `feat: port core manifest kinds and run discovery` | `src/core/manifest.py`, `tests/test_manifest.py` | named tests green |
| 7 | `feat: port core metrics` | `src/core/metrics.py`, `tests/test_core_metrics.py` | named tests green |
| 8 | `feat: port core binary metrics and threshold helper` | `src/core/binary_metrics.py`, `tests/test_core_binary_metrics.py`, `tests/test_core_binary_metrics_plots.py` | named tests green; `threshold_labels` boundary tests (equal-to-threshold → negative) |
| 9 | `feat: port core plotting, serialization, and binning` | `src/core/plots.py`, `src/core/io.py`, `src/core/binning.py`, their tests | named tests green |
| 10 | `feat: port core partition layer` | `src/core/splits.py`, `tests/test_splits.py` | named tests green |
| 11 | `feat: port core estimator layer` | `src/core/estimator.py`, `src/core/estimators/`, `tests/test_core_estimators.py` | named tests green |
| 12 | `feat: port core training mechanics` | `src/core/training.py`, `tests/test_core_training.py` | named tests green |
| 13 | `feat: add project vocabulary, config, and outcomes` | `src/project/columns.py`, `config.py`, `outcomes.py`; `tests/test_columns.py`, `test_project_config.py`, `test_outcomes.py`, `test_outcome_column_vocabulary.py` | named tests green; config dataclass constructs `CONFIG`; log1p/expm1 round-trip and domain rejection |
| 14 | `feat: add project boundary schemas` | `src/project/schemas.py`, `tests/test_schemas.py` | named tests green; one failure test per §7.2 constraint |
| 15 | `feat: add project external-input registry` | `src/project/inputs.py`, `tests/test_inputs.py` | named tests green; eager and lazy paths both covered |
| 16 | `feat: add project feature policies` | `src/project/features.py`, `tests/test_features.py` | named tests green; head-order preservation |
| 17 | `feat: add project model strategies` | `src/project/models.py`, `tests/test_model_strategies.py` | named tests green; table matches §7.2 exactly |
| 18 | `feat: add project estimator factory` | `src/project/estimators.py`, `tests/test_project_estimators.py` | named tests green |
| 19 | `feat: add project artifact I/O` | `src/project/artifacts.py`, `tests/test_s04_extract_feature_importances.py`, `tests/test_s04_load_feature_importances.py`, `tests/test_s04_model_manifest.py` | named tests green; manifest round-trip |
| 20 | `feat: add project training driver` | `src/project/training.py`, `tests/test_project_training.py` | named tests green; one strategy through `fit_single` on demo split tables |
| 21 | `feat: add stage table and orchestration registry` | `src/project/pipeline.py`, `tests/test_project_pipeline.py`, `tests/test_contracts.py` | named tests green; cycle/referential-integrity checks; lazy stage imports |
| 22 | `feat: add run-stage planner and CLI wrapper` | `src/project/run_stage.py`, `scripts/run_stage.py`; `tests/test_run_stage.py` | named tests green; `--list`/`--dry-run` work; importing planner imports no stage |
| 23 | `feat: add deterministic demo data` | `scripts/make_demo_data.py`, `data/demo/source_table_1.parquet` | regenerate twice into two temp roots; byte-identical; containment rejection test |
| 24 | `feat: add project adapters` | `src/project/adapters.py`, `tests/test_adapters.py` | adapters deterministic, key-aligned, shape-correct |
| 25 | `feat: add s01 outcome stage` | `src/stages/s01_outcome.py`, `tests/test_s01_outcome.py` | stage test green; schema boundary label asserted |
| 26 | `feat: add s01b outcome-analysis stage` | `src/stages/s01b_outcome_analysis.py`, `tests/test_s01b_outcome_analysis.py` | four plots per comparison emitted even on insufficient data |
| 27 | `feat: add s02 feature stage` | `src/stages/s02_feature.py`, `tests/test_s02_feature.py` | stage test green; feature manifest round-trips; key alignment |
| 28 | `feat: add s03 divide stage` | `src/stages/s03_divide_data.py`, `tests/test_s03_divide.py` | partitions are disjoint, exhaustive, non-empty; boundary dates tested |
| 29 | `feat: add s04 train stage` | `src/stages/s04_train.py`, `tests/test_s04_train.py` | one strategy; manifest kind/version correct; `task='regression'` |
| 30 | `feat: add s05 predict stage` | `src/stages/s05_predict_model.py`, `s05_predict_model_main.py`; `tests/test_s05_manifest_reader.py`, `tests/test_s05_prediction_manifest.py` | predictions cover every declared partition through manifests only |
| 31 | `feat: add s06 evaluate stage` | `src/stages/s06_evaluate_model.py`, `s06_evaluate_model_main.py`; `tests/test_s06_*.py` | metrics JSON + display CSV; binary labels from thresholding |
| 32 | `feat: add pipeline entry point and smoke tests` | `src/main.py`, `tests/test_smoke.py`, `tests/test_smoke_pipeline.py`, `tests/test_smoke_main.py` | full demo run produces §8.5 tree; `-m src.main` guard exercised |
| 33 | `test: add golden baseline harness and baseline` | `tests/golden/*`, `tests/golden/baseline/*`, `tests/test_golden_*.py` | capture twice; diff is exact and stable; baseline contains only §8.5 non-parquet/non-PNG artifacts |
| 34 | `docs: write architecture, agent, and user documentation` | `docs/ARCHITECTURE.md`, `AGENTS.md`, `README.md`, `docs/research/README.md` | docs use §4 vocabulary; `scrub_repo.py --worktree` clean; README has no development history |
| 35 | `chore: final acceptance audit` *(verification only; no commit if clean)* | — | `scrub_repo.py --history`; full suite; full run; commit-message check |

**Step contract.** `TODO.md` is the machine-readable form of this table and is
what implementation follows; this table remains the specification. The two
differ only in how a row maps to a commit:

- Row 0 (`git init`, `.gitignore`, hook configuration) is a prerequisite, not a
  commit.
- Row 4 (`docs: add extraction plan`) is the bootstrap commit that adds this
  plan together with `TODO.md`, so `TODO.md` has no separate row for it. Rows 0
  and 4, plus the tooling commits 1–2, are vocabulary-gated retroactively with
  `--history` immediately after the row-3 gate commit.
- Row 35 has no commit; it is a non-committed acceptance checklist in
  `TODO.md`.
- Rows 13, 31, and 33 are split into `13a`–`13c` (column vocabulary, config,
  outcome registry), `31a`–`31b` (s06 metric helpers, s06 driver), and
  `33a`–`33b` (golden harness, baseline capture) so each commit stays one
  verifiable unit. Each split row keeps this row's verification contract.

### Why this order

- The gate scripts are bootstrapped by commits 1–3, then enforced for every
  later commit. The bootstrap range is verified after the fact with
  `--history` immediately after step 3.
- The core layer lands before the project layer, preserving the one-way
  dependency from the first commit; `test_module_imports.py` lands with the
  first core module.
- Each module is its own commit, so a failing port can be reverted without
  unwinding unrelated work.
- The demo data and adapters land before the stages that consume them.
- s03 lands before s04, s04 before s05, and s05 before s06, so each stage's test
  can use real upstream artifacts rather than hand-built stand-ins.
- The golden baseline lands last, after the demo is stable, so it is recorded
  once and never rewritten to hide drift.

---

## 10. Testing strategy

### 10.1 Port unchanged (with vocabulary scrub)

- Core metrics, binary metrics, metric plots, plots, I/O, binning.
- Core manifest read/write, schema-version rejection, run discovery.
- Core partition contract.
- Core estimator adapter and training mechanics.
- Path/layout resolution and context `ensure`.
- Import contract (`src.core` imports no `src.project`).
- Config/context conformance.

### 10.2 Rewrite for the generic template

- Column vocabulary and outcome registry.
- Boundary schemas and the external-input registry (one demo table).
- Feature policies (identity, outcome-plus-noise, pure noise, subset).
- Model strategy table identity and trainer-kind dispatch.
- Artifact I/O and model-manifest round trips.
- Training driver on the demo split tables.
- Stage table, prerequisite graph, registry consistency, run-stage planner.
- s01/s01b/s02 demo stages.
- s03–s06 stages on the demo artifacts.
- Smoke: every stage, every model strategy, the registry-driven s03→s06 run.
- Golden harness (capture, diff, path/alias declarations).

### 10.3 Drop

- Tests whose imports or fixtures reference the source's subject-matter
  modules, stage subpackages, helper modules, or fixtures.
- Tests that assert subject-matter column names, formulas, or comparison
  semantics.
- The source's legacy-residue guard test (it encodes a subject-matter noun).

### 10.4 Add

- `tests/test_vocabulary_scrub.py` — §6.
- `tests/test_adapters.py` — demo builders are deterministic, key-aligned, and
  shape-correct.
- `tests/test_s03_divide.py` and stage-specific tests where the source had
  only a smoke test.
- `tests/test_golden_*` — baseline reproducibility.
- `tests/test_property_*.py` — `hypothesis`-based property tests (dependency
  added in step 1, lock in step 2) for the invariants that genuinely hold:
  * manifest `write → read` round-trip preserves every field for arbitrary
    valid manifests;
  * `train`/`valid`/`test` partition sizes sum to the input height for
    arbitrary dates and a valid `train_end_date < test_start_date`;
  * the scrub tokenizer/n-gram canonicalizer is stable across repeated
    application (`canonicalize(canonicalize(x)) == canonicalize(x)`).
- `tests/test_project_training.py` — the project training driver on the demo
  split tables (previously missing).

### 10.5 Coverage goals

- Every public function in `src/core/` has at least one unit test.
- Every stage has a smoke test on the committed demo data.
- Every manifest kind has a write→read round-trip test and an
  unknown-version rejection test.
- Every constraint enumerated in §7.2 has exactly one failure test asserting
  the boundary label and the failing rule.
- Binary label derivation has a boundary test at exactly `value == threshold`
  (negative class) for actual and predicted labels.
- The scrub test is part of the default suite.

---

## 11. Definition of done

1. `pixi run pytest -q` is green.
2. `OMP_NUM_THREADS=1 pixi run python -m src.main` produces the §8.5 tree via
   the `if __name__ == '__main__': main()` guard in `src/main.py`.
3. `pixi run ruff check .`, `pixi run ruff format --check .`, and
   `pixi run pyright` are clean.
4. `pixi run python scripts/scrub_repo.py --worktree` exits `0`.
5. `pixi run python scripts/scrub_repo.py --history` exits `0`.
6. Every commit message in `git log --format=%B` passes the same tokenizer.
7. `tests/test_module_imports.py` proves `src.core` imports no `src.project`
   and no stage module (tolerating an absent `src/stages/` before step 25).
8. The golden baseline diffs exactly on a second run and contains only the
   artifacts listed in §8.5's baseline note (no parquet, PNG, or estimator
   serialization).
9. `docs/ARCHITECTURE.md`, `AGENTS.md`, and `README.md` describe the current
   template, name `src/project/adapters.py` as the extension point, and contain
   no subject-matter vocabulary. The README contains no development history.
10. No untracked `.scrub/` or `.port-staging/` content is needed to run the
    repository; only the committed hash list and the scrub scripts are
    authoritative.
11. `.port-staging/audit.md` records a clean dependency audit for every added
    third-party package; the file itself is untracked (it may name packages).
12. `.scrub/denylist.txt` was derived from the supplied source tree and
    reviewed before step 3; the committed hash list is the only gate input.

---

## 12. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| A subject-matter noun survives in a docstring, comment, test name, or fixture column | Hash-based worktree/staged/history gate; develop in `.port-staging/`; never copy raw source |
| The denylist test itself introduces a forbidden term | Store only `sha256` digests; keep the term list untracked |
| False positives from common programming words | Curate entries as unambiguous tokens or multi-token phrases; report fingerprints, not terms |
| A pre-existing violating commit | Rewrite local history immediately; do not push until `--history` is clean |
| Core gains a project import during the port | `tests/test_module_imports.py` from the first core commit |
| Demo non-determinism (thread scheduling, RNG order) | `OMP_NUM_THREADS=1` and `nthread=1` set before numpy/xgboost import; one global estimator seed; per-adapter `default_rng`; fixed registry order; capture twice |
| Golden baseline is recaptured to hide a bug | Recapture only with an explicit, reviewed reason; diff must be exact |
| Parquet output changes across polars versions or writer settings | Pin `polars` in `pixi.lock`; explicit `compression='zstd'` and statistics; recapture only with a reviewed reason |
| Scrub misses camelCase or multi-token terms | Split camelCase/PascalCase; canonical n-grams joined with a single space; property test for canonicalizer stability |
| Denylist derivation over- or under-includes terms | Derivation writes candidates, never the final list; a human review is mandatory before `--hashes`; the gate fails closed on an empty list |
| Scope creep from porting subject-matter feature engineering | §5.4 lists the deliberate omissions; family registry is a future extension |
| Binary demo fixture accidentally carries subject-matter metadata | `scrub_repo.py` reads parquet column names and string values |
| Python/estimator version drift | Pin Python 3.12 and record the resolved `pixi.lock`; dependency audit in step 2 |
| A stage leaves stale artifacts from an earlier run | Each stage atomically replaces only its declared artifacts; smoke tests run each stage twice and assert stable output |

---

## 13. Future extensions (out of scope)

These are documented so the template's shape is understood, but they are not
part of this plan:

1. A generic feature-family registry with longest-prefix attribution and
   as-of tiers, replacing the explicit `FeatureSubset` policy.
2. Elapsed-duration outcome columns and a numeric-or-duration schema helper.
3. Additional estimator adapters (linear models, gradient-boosted trees from
   other libraries) registered through `src/core/estimator.py`, and a real
   `binary:*` classification objective with its own task dispatch (v1 derives
   binary metrics by thresholding regression outputs).
4. A CI workflow that runs the test suite, the full demo, and the history
   scrub.
5. Packaging as an installable library with a console entry point.
6. A `--profile` or `--dry-run` execution surface for large runs.
7. A generic cross-validation partition in addition to the holdout contract.

---

## 14. Resolved decisions

The three items previously left open are now decided. No open questions remain.

| Item | Decision | Rationale |
| --- | --- | --- |
| Python version | Pin **3.12** (local `/usr/bin/python3` is 3.12.3); mirror the source's generic dependency pins; record `pixi.lock` in step 2 | Matches the available interpreter and keeps the port reproducible |
| Golden baseline content | Commit **manifests, metrics JSON, statistics text, and display CSVs only**. Exclude parquet split tables, PNGs, and estimator serializations | Smaller, less version-sensitive baseline; parquet bytes are covered by the step 23 determinism test instead |
| Estimator objective | Keep **`reg:absoluteerror`** | Minimizes porting risk; switching is a one-line config change deferred to a future extension |
| Denylist provisioning | Derived from the supplied source tree by `build_denylist.py --derive`, human-reviewed, then hashed (§6.1a) | Satisfies the hard gate without recording source prose; fails closed if unavailable |
| Binary metrics | Threshold regression outputs at `CONFIG.binary_thresholds`; `task='regression'`; strict `value > threshold` | No new objective or adapter needed; consistent actual/predicted rule |

Implementation-time checks that are **not** open questions (they are
verification steps with unambiguous pass criteria):

- Confirm the source dependency pins resolve under Python 3.12 and record the
  lock in step 2.
- Confirm the committed demo yields at least one positive and one negative
  label at every configured threshold in `train` and `valid` (step 31 test).
- Confirm the baseline was captured twice and diffs exactly (step 33).
