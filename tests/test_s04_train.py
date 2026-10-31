"""Integration tests for the ``s04_train`` stage.

The stage is a thin shell: it resolves one model run's directory from the
declarative layout, creates it through ``context.ensure``, and delegates to the
project training driver. The tests therefore drive the whole path on a
temporary project root seeded with the committed demo table and the real
s01/s02/s03 stages, and pin the on-disk contract of one strategy:

* the run directory is ``s04_train/{outcome}/{strategy_id}``;
* the manifest is a model manifest of the accepted schema version whose
  identity, task, objective, columns, partitions, transform id, and tuned
  parameters match the strategy table and the injected configuration;
* the manifest's artifact inventory is exactly the run directory's file set,
  so a consumer can read the run through the manifest alone;
* every declared partition has its key and feature tables, the
  feature-importance table round-trips through the manifest reader, and the
  run is discoverable through ``load_model_runs``;
* re-running the stage replaces the same run rather than accumulating files;
* the registry binding targets this module's single entry point with the
  strategy id as its leading argument;
* an unknown strategy fails before the run directory is created, and a
  missing s03 prerequisite leaves an empty run directory behind.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from src.core.manifest import (
    MANIFEST_FILENAME,
    MODEL_SCHEMA_VERSION,
    ModelManifest,
    load_model_runs,
    read_manifest,
)
from src.core.paths import PipelineContext
from src.core.splits import PARTITION_ORDER
from src.project.artifacts import load_feature_importances
from src.project.columns import (
    FEATURE_COLNAME,
    FEATURE_ESTIMATOR_ID_COLNAME,
    FEATURE_IMPORTANCE_COLNAME,
    KEY_COLUMNS,
)
from src.project.config import CONFIG
from src.project.estimators import ESTIMATOR_ID
from src.project.models import MODEL_STRATEGIES
from src.project.outcomes import OUTCOMES, modeled_outcomes
from src.project.pipeline import STAGE_BINDINGS, build_context
from src.project.schemas import FEATURE_COLUMNS
from src.stages.s01_outcome import s01_outcome
from src.stages.s02_feature import s02_feature
from src.stages.s03_divide_data import s03_divide
from src.stages.s04_train import main, model_run_dir, s04_train

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Repository root, used only to locate the committed demo input table."""

SOURCE_FILENAME = "source_table_1.parquet"
"""Committed demo filename, copied into each temporary project root."""

STRATEGY_ID = "model_00_signal_0"
"""The single strategy these integration tests train."""

MODELED_OUTCOME_NAME = modeled_outcomes()[0].name
"""The demo outcome the stage's own tests train."""

STAGE_NAME = "s04_train"
"""The stage name the run directory and manifest record."""


@dataclass(frozen=True, slots=True)
class Trained:
    """One completed stage run: the temp root, its context, and the run dir."""

    project_root: Path
    context: PipelineContext
    output_root: Path
    run_dir: Path

    @property
    def manifest(self) -> ModelManifest:
        """The model manifest of the trained run."""
        manifest = read_manifest(self.run_dir / MANIFEST_FILENAME)
        assert isinstance(manifest, ModelManifest)
        return manifest


def _copy_demo_source(project_root: Path) -> None:
    """Place the committed demo source table under a temporary root."""
    data_dir = project_root / CONFIG.data_root
    data_dir.mkdir(parents=True)
    shutil.copyfile(
        PROJECT_ROOT / CONFIG.data_root / SOURCE_FILENAME,
        data_dir / SOURCE_FILENAME,
    )


def _seed_through_divide(context: PipelineContext, output_root: Path) -> None:
    """Run the s01, s02, and s03 stages so s04 has its inputs."""
    s01_outcome(context, output_root)
    s02_feature(context, output_root)
    s03_divide(MODELED_OUTCOME_NAME, context, output_root)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Return a temporary project root containing the demo source table."""
    _copy_demo_source(tmp_path)
    return tmp_path


@pytest.fixture
def context(project_root: Path) -> PipelineContext:
    """Return the pipeline context bound to the temporary project root."""
    return build_context(config=CONFIG, project_root=project_root)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    """Train one strategy once on a fully seeded temporary project root."""
    project_root = tmp_path_factory.mktemp("s04_trained")
    _copy_demo_source(project_root)
    context = build_context(config=CONFIG, project_root=project_root)
    output_root = project_root / "output"
    _seed_through_divide(context, output_root)
    main(STRATEGY_ID, MODELED_OUTCOME_NAME, context, output_root)
    run_dir = model_run_dir(STRATEGY_ID, MODELED_OUTCOME_NAME, context, output_root)
    return Trained(project_root, context, output_root, run_dir)


# --- manifest and artifact contract ---------------------------------------


def test_stage_writes_the_run_directory_the_layout_declares(
    trained: Trained,
) -> None:
    expected = trained.output_root / STAGE_NAME / MODELED_OUTCOME_NAME / STRATEGY_ID

    assert trained.run_dir == expected
    assert trained.run_dir.is_dir()
    assert (trained.run_dir / MANIFEST_FILENAME).is_file()


def test_manifest_records_model_identity_and_the_regression_task(
    trained: Trained,
) -> None:
    manifest = trained.manifest

    assert manifest.schema_version == MODEL_SCHEMA_VERSION
    assert manifest.manifest_kind == "model"
    assert manifest.model_name == STRATEGY_ID
    assert manifest.stage == STAGE_NAME
    assert manifest.outcome == MODELED_OUTCOME_NAME
    assert manifest.estimator == ESTIMATOR_ID
    assert manifest.task == "regression"
    assert manifest.objective == MODEL_STRATEGIES[STRATEGY_ID].objective
    assert manifest.outcome_colname == MODELED_OUTCOME_NAME
    assert manifest.feature_colnames == tuple(FEATURE_COLUMNS)
    assert manifest.partitions == PARTITION_ORDER
    assert manifest.target_transform == (
        OUTCOMES[MODELED_OUTCOME_NAME].outcome_transform
    )
    assert manifest.best_parameters == dict(CONFIG.xgb_single_fit_params)


def test_manifest_inventory_is_exactly_the_run_directory(
    trained: Trained,
) -> None:
    declared = {artifact.filename for artifact in trained.manifest.artifacts}
    on_disk = {path.name for path in trained.run_dir.iterdir() if path.is_file()}
    other_entries = {
        path.name for path in trained.run_dir.iterdir() if not path.is_file()
    }

    assert len(declared) == len(trained.manifest.artifacts)
    assert on_disk == declared | {MANIFEST_FILENAME}
    assert other_entries == set()


def test_every_declared_partition_has_its_key_and_feature_tables(
    trained: Trained,
) -> None:
    split_dir = trained.output_root / "s03_divide" / MODELED_OUTCOME_NAME

    for partition in trained.manifest.partitions:
        key_df = pl.read_parquet(trained.run_dir / f"{partition}_key_df.parquet")
        feature_df = pl.read_parquet(
            trained.run_dir / f"{partition}_feature_df.parquet"
        )
        source_df = pl.read_parquet(split_dir / f"{partition}_df.parquet")

        assert tuple(key_df.columns) == KEY_COLUMNS
        assert tuple(feature_df.columns) == (*KEY_COLUMNS, *FEATURE_COLUMNS)
        assert key_df.height == source_df.height > 0
        assert feature_df.select(list(KEY_COLUMNS)).equals(key_df)


def test_feature_importances_round_trip_through_the_manifest(
    trained: Trained,
) -> None:
    importances = load_feature_importances(trained.manifest, trained.run_dir)

    assert tuple(importances.columns) == (
        FEATURE_COLNAME,
        FEATURE_ESTIMATOR_ID_COLNAME,
        FEATURE_IMPORTANCE_COLNAME,
    )
    # The estimator's positional ids map back to the declared feature order.
    assert dict(
        zip(
            importances[FEATURE_ESTIMATOR_ID_COLNAME].to_list(),
            importances[FEATURE_COLNAME].to_list(),
            strict=True,
        )
    ) == {f"f{index}": name for index, name in enumerate(FEATURE_COLUMNS)}
    assert importances.height == len(FEATURE_COLUMNS)
    assert (importances[FEATURE_IMPORTANCE_COLNAME] > 0).all()


def test_run_is_discoverable_through_its_manifest(trained: Trained) -> None:
    runs = load_model_runs(trained.run_dir.parent, MODELED_OUTCOME_NAME)

    assert [manifest.model_name for manifest in runs] == [STRATEGY_ID]
    assert runs[0] == trained.manifest


def test_registry_binding_targets_the_single_entry_point(
    trained: Trained,
) -> None:
    binding = next(
        binding
        for binding in STAGE_BINDINGS
        if binding.stage == STAGE_NAME and binding.binding_id == STRATEGY_ID
    )

    assert binding.module == "src.stages.s04_train"
    assert binding.callable_name == "main"
    assert binding.resolve() is main
    assert binding.build_args(
        MODELED_OUTCOME_NAME, trained.context, trained.output_root
    ) == (STRATEGY_ID, MODELED_OUTCOME_NAME, trained.context, trained.output_root)


def test_stage_entry_point_is_exported_under_its_stage_name() -> None:
    assert s04_train is main


# --- rerun and failure cases ----------------------------------------------


def test_rerunning_the_stage_replaces_the_same_run(
    project_root: Path,
    context: PipelineContext,
) -> None:
    output_root = project_root / "output"
    _seed_through_divide(context, output_root)
    main(STRATEGY_ID, MODELED_OUTCOME_NAME, context, output_root)
    run_dir = model_run_dir(STRATEGY_ID, MODELED_OUTCOME_NAME, context, output_root)
    first_files = sorted(path.name for path in run_dir.iterdir())
    first_manifest = (run_dir / MANIFEST_FILENAME).read_bytes()

    main(STRATEGY_ID, MODELED_OUTCOME_NAME, context, output_root)

    assert sorted(path.name for path in run_dir.iterdir()) == first_files
    assert (run_dir / MANIFEST_FILENAME).read_bytes() == first_manifest
    assert [path.name for path in run_dir.parent.iterdir()] == [STRATEGY_ID]


def test_unknown_strategy_fails_before_creating_the_run(
    project_root: Path,
    context: PipelineContext,
) -> None:
    output_root = project_root / "output"

    with pytest.raises(ValueError, match="unknown model strategy 's99_missing'"):
        main("s99_missing", MODELED_OUTCOME_NAME, context, output_root)

    assert not (output_root / STAGE_NAME).exists()
    with pytest.raises(ValueError, match="unknown model strategy"):
        model_run_dir("s99_missing", MODELED_OUTCOME_NAME, context, output_root)


def test_missing_prerequisites_fail_before_writing_any_artifact(
    project_root: Path,
    context: PipelineContext,
) -> None:
    output_root = project_root / "output"

    with pytest.raises(FileNotFoundError):
        main(STRATEGY_ID, MODELED_OUTCOME_NAME, context, output_root)

    run_dir = model_run_dir(STRATEGY_ID, MODELED_OUTCOME_NAME, context, output_root)
    assert run_dir.is_dir()
    assert not any(run_dir.iterdir())
