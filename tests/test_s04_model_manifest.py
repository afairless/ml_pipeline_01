"""Unit tests for the s04 split loader and model-manifest producer.

``save_results`` persists a trained run's artifacts and closes the run with
``manifest.json``: a frozen ``ModelManifest`` recording model identity
(``model_name`` is the strategy id), the producing stage and outcome, the
estimator/task/objective, the outcome and feature column names, the outcome's
registered model-space ``target_transform`` id, the tuned ``best_parameters``,
the run's declared ``partitions``, and the ``ArtifactRef`` inventory of every
file written. These tests exercise the producer directly on a tiny trained
model and pin the manifest fields, the write/read round trip, and the on-disk
inventory match: every written file is declared exactly once, nothing
undeclared is recorded, the manifest never lists itself, and a declared
artifact missing from disk makes the reader raise. The tuned parameters live
in the manifest's ``best_parameters`` field only — no standalone parameters
file is written.

The split-loader tests stage the five s03 artifacts behind a minimal layout and
pin that ``load_split_tables`` reads through the layout's logical names, uses
the shared key vocabulary, and validates each split partition at the read
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from src.core.manifest import (
    MANIFEST_FILENAME,
    MODEL_SCHEMA_VERSION,
    ModelManifest,
    read_manifest,
)
from src.core.paths import PipelineLayout, StageSpec
from src.core.splits import PARTITION_ORDER
from src.project.artifacts import (
    FEATURE_IMPORTANCES_LOGICAL_NAME,
    S03_SPLIT_READ_BOUNDARY,
    load_split_tables,
    save_results,
    validate_partition_feature_df,
    validate_partition_key_df,
)
from src.project.columns import CASE_DATE, CASE_ID, KEY_COLUMNS
from src.project.outcomes import OUTCOMES
from src.project.schemas import FEATURE_COLUMNS, BoundaryError

FEATURE_COLNAMES = list(FEATURE_COLUMNS)
TINY_BEST_PARAMETERS = {"learning_rate": 0.01, "max_depth": 1}
_TINY_N = 16


@dataclass(frozen=True)
class _ModelReadyData:
    """The structural model-ready value the artifact writer consumes."""

    train_key_df: pl.DataFrame
    train_feature_select_df: pl.DataFrame
    valid_key_df: pl.DataFrame
    valid_feature_select_df: pl.DataFrame
    test_key_df: pl.DataFrame
    test_feature_select_df: pl.DataFrame


def _keys() -> pl.DataFrame:
    return pl.DataFrame(
        {
            CASE_ID: [f"case_{index:04d}" for index in range(_TINY_N)],
            CASE_DATE: [
                date(2024, 1, 1) + timedelta(days=index) for index in range(_TINY_N)
            ],
        }
    )


def _features() -> pl.DataFrame:
    base = np.linspace(-1.0, 1.0, _TINY_N)
    return pl.DataFrame(
        {
            name: base * (index + 1) + float(index)
            for index, name in enumerate(FEATURE_COLNAMES)
        }
    )


def _feature_df() -> pl.DataFrame:
    """The training feature frame the writer maps gains against."""
    return _features()


def _make_model_ready_data() -> _ModelReadyData:
    keys = _keys()
    features = _features()
    return _ModelReadyData(
        train_key_df=keys,
        train_feature_select_df=features,
        valid_key_df=keys,
        valid_feature_select_df=features,
        test_key_df=keys,
        test_feature_select_df=features,
    )


def _train_tiny_model() -> xgb.Booster:
    """A small regression booster over the tiny feature frame."""
    features = _features().to_numpy()
    target = 2.0 * features[:, 0] - features[:, 1] + 0.5 * features[:, 2]
    dmtx = xgb.DMatrix(features, label=target)
    return xgb.train(
        {"objective": "reg:absoluteerror", "max_depth": 2, "nthread": 1},
        dmtx,
        num_boost_round=3,
    )


def _save_tiny_run(
    tmp_path: Path,
    *,
    strategy_id: str = "model_00_signal_0",
    outcome_name: str = "outcome_1",
    objective: str = "reg:absoluteerror",
) -> Path:
    """Run ``save_results`` on a tiny trained model and return the run dir."""
    output_dir = tmp_path / strategy_id
    output_dir.mkdir()
    save_results(
        _train_tiny_model(),
        dict(TINY_BEST_PARAMETERS),
        _make_model_ready_data(),
        _feature_df(),
        FEATURE_COLNAMES,
        output_dir,
        strategy_id=strategy_id,
        outcome_name=outcome_name,
        objective=objective,
    )
    return output_dir


def _manifest_of(output_dir: Path) -> ModelManifest:
    manifest = read_manifest(output_dir / MANIFEST_FILENAME)
    assert isinstance(manifest, ModelManifest)
    return manifest


# --- the manifest file itself ----------------------------------------------


def test_save_results_writes_manifest_into_run_dir(tmp_path: Path) -> None:
    output_dir = _save_tiny_run(tmp_path)
    assert (output_dir / MANIFEST_FILENAME).is_file()
    _manifest_of(output_dir)


def test_manifest_records_identity_and_configuration(tmp_path: Path) -> None:
    output_dir = _save_tiny_run(
        tmp_path,
        strategy_id="model_01a",
        outcome_name="outcome_3",
    )
    manifest = _manifest_of(output_dir)

    assert manifest.schema_version == MODEL_SCHEMA_VERSION
    assert manifest.manifest_kind == "model"
    assert manifest.model_name == "model_01a"  # the strategy id
    assert manifest.stage == "s04_train"
    assert manifest.outcome == "outcome_3"
    assert manifest.estimator == "xgboost"
    assert manifest.task == "regression"  # derived from the objective
    assert manifest.objective == "reg:absoluteerror"
    assert manifest.outcome_colname == "outcome_3"
    assert manifest.feature_colnames == tuple(FEATURE_COLNAMES)
    assert manifest.target_transform == OUTCOMES["outcome_3"].outcome_transform
    assert manifest.best_parameters == TINY_BEST_PARAMETERS


def test_manifest_records_partitions(tmp_path: Path) -> None:
    """The manifest declares the run's split shape in canonical order."""
    manifest = _manifest_of(_save_tiny_run(tmp_path))
    assert manifest.partitions == PARTITION_ORDER
    assert set(manifest.partitions) == {"train", "valid", "test"}


@pytest.mark.parametrize(
    ("objective", "task"),
    [
        ("reg:absoluteerror", "regression"),
        ("reg:squarederror", "regression"),
        ("binary:logistic", "binary"),
    ],
)
def test_task_derived_from_objective(
    tmp_path: Path,
    objective: str,
    task: str,
) -> None:
    output_dir = _save_tiny_run(tmp_path, objective=objective)
    assert _manifest_of(output_dir).task == task


def test_save_results_rejects_unclassifiable_objective(tmp_path: Path) -> None:
    """An objective that is neither reg:* nor binary:* never reaches the
    manifest, so a recorded manifest always carries a classifiable task."""
    output_dir = tmp_path / "model_00_signal_0"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="objective"):
        save_results(
            _train_tiny_model(),
            dict(TINY_BEST_PARAMETERS),
            _make_model_ready_data(),
            _feature_df(),
            FEATURE_COLNAMES,
            output_dir,
            strategy_id="model_00_signal_0",
            outcome_name="outcome_1",
            objective="rank:ndcg",
        )


# --- artifact inventory -----------------------------------------------------


def test_save_results_written_partition_artifacts_pass_their_schemas(
    tmp_path: Path,
) -> None:
    output_dir = _save_tiny_run(tmp_path)

    for partition in PARTITION_ORDER:
        key_df = pl.read_parquet(output_dir / f"{partition}_key_df.parquet")
        feature_df = pl.read_parquet(output_dir / f"{partition}_feature_df.parquet")
        assert validate_partition_key_df(key_df, partition).equals(key_df)
        assert validate_partition_feature_df(feature_df, partition).equals(feature_df)


def test_save_results_rejects_a_malformed_partition_key_frame(
    tmp_path: Path,
) -> None:
    """A null key fails before the key artifact is written."""
    mrd = _make_model_ready_data()
    malformed = replace(
        mrd,
        train_key_df=mrd.train_key_df.with_columns(
            pl.lit(None, dtype=pl.Date).alias(CASE_DATE)
        ),
    )
    output_dir = tmp_path / "model_00_signal_0"
    output_dir.mkdir()

    with pytest.raises(BoundaryError) as excinfo:
        save_results(
            _train_tiny_model(),
            dict(TINY_BEST_PARAMETERS),
            malformed,
            _feature_df(),
            FEATURE_COLNAMES,
            output_dir,
            strategy_id="model_00_signal_0",
            outcome_name="outcome_1",
            objective="reg:absoluteerror",
        )

    assert "s04 train_key_df" in str(excinfo.value)
    assert not (output_dir / "train_key_df.parquet").exists()


def test_save_results_rejects_a_malformed_partition_feature_frame(
    tmp_path: Path,
) -> None:
    """A non-numeric feature fails before the feature artifact is written."""
    mrd = _make_model_ready_data()
    malformed = replace(
        mrd,
        train_feature_select_df=mrd.train_feature_select_df.with_columns(
            pl.lit("not-a-feature").alias(FEATURE_COLNAMES[0])
        ),
    )
    output_dir = tmp_path / "model_00_signal_0"
    output_dir.mkdir()

    with pytest.raises(BoundaryError) as excinfo:
        save_results(
            _train_tiny_model(),
            dict(TINY_BEST_PARAMETERS),
            malformed,
            _feature_df(),
            FEATURE_COLNAMES,
            output_dir,
            strategy_id="model_00_signal_0",
            outcome_name="outcome_1",
            objective="reg:absoluteerror",
        )

    assert "s04 train_feature_df" in str(excinfo.value)
    assert not (output_dir / "train_feature_df.parquet").exists()


def test_artifacts_inventory_matches_written_files(tmp_path: Path) -> None:
    """Every written file is declared exactly once; the manifest never lists
    itself."""
    output_dir = _save_tiny_run(tmp_path)
    manifest = _manifest_of(output_dir)
    produced = {path.name for path in output_dir.iterdir()}
    inventory = {ref.filename for ref in manifest.artifacts}

    assert inventory == produced - {MANIFEST_FILENAME}
    assert len(inventory) == len(produced) - 1


def test_artifacts_inventory_expects_full_run_file_set(tmp_path: Path) -> None:
    """The inventory is the complete run artifact set."""
    data_stems = (
        "train_key_df",
        "train_feature_df",
        "valid_key_df",
        "valid_feature_df",
        "test_key_df",
        "test_feature_df",
    )
    expected = {
        f"{stem}.{suffix}" for stem in data_stems for suffix in ("parquet", "csv")
    }
    expected.update(
        {
            "feature_importances.parquet",
            "feature_importances.csv",
            "model.json",
            "model_pretty.json",
            "model.txt",
        }
    )

    manifest = _manifest_of(_save_tiny_run(tmp_path))
    assert {ref.filename for ref in manifest.artifacts} == expected


def test_training_artifacts_are_not_partition_tagged(tmp_path: Path) -> None:
    manifest = _manifest_of(_save_tiny_run(tmp_path))

    assert all(ref.partition is None for ref in manifest.artifacts)
    for ref in manifest.artifacts:
        assert ref.logical_name in {
            "train_key_df",
            "train_feature_df",
            "valid_key_df",
            "valid_feature_df",
            "test_key_df",
            "test_feature_df",
            FEATURE_IMPORTANCES_LOGICAL_NAME,
            "model",
        }


def test_every_declared_partition_has_its_split_tables_on_disk(
    tmp_path: Path,
) -> None:
    output_dir = _save_tiny_run(tmp_path)
    manifest = _manifest_of(output_dir)
    produced = {path.name for path in output_dir.iterdir()}
    inventory = {ref.filename for ref in manifest.artifacts}

    for partition in manifest.partitions:
        for stem in (f"{partition}_key_df", f"{partition}_feature_df"):
            for suffix in (".parquet", ".csv"):
                filename = f"{stem}{suffix}"
                assert filename in produced
                assert filename in inventory


# --- round trip and compatibility -------------------------------------------


def test_manifest_read_twice_is_stable(tmp_path: Path) -> None:
    output_dir = _save_tiny_run(tmp_path)
    first = _manifest_of(output_dir)
    second = _manifest_of(output_dir)

    assert second == first
    assert isinstance(second.feature_colnames, tuple)
    assert isinstance(second.artifacts, tuple)


def test_best_parameters_live_only_in_the_manifest(tmp_path: Path) -> None:
    output_dir = _save_tiny_run(tmp_path)
    produced = {path.name for path in output_dir.iterdir()}
    manifest = _manifest_of(output_dir)

    assert "best_parameters.json" not in produced
    assert not any(ref.logical_name == "best_parameters" for ref in manifest.artifacts)
    assert manifest.best_parameters == TINY_BEST_PARAMETERS


# --- split-table loading ----------------------------------------------------


def _layout() -> PipelineLayout:
    artifacts = {
        "feature_head_df": "feature_head_df.parquet",
        "outcome_head_df": "outcome_head_df.parquet",
        "train_df": "train_df.parquet",
        "valid_df": "valid_df.parquet",
        "test_df": "test_df.parquet",
    }
    return PipelineLayout(
        stages={
            "s03_divide": StageSpec(
                name="s03_divide",
                dir_template="s03_divide/{outcome}",
                artifacts=artifacts,
            )
        },
        outcome_names=frozenset({"outcome_1"}),
    )


def _split_frame() -> pl.DataFrame:
    keys = _keys()
    frame = keys.hstack(_features())
    return frame.with_columns(pl.lit(1.0).alias("outcome_1"))


def _write_split_artifacts(
    root: Path,
    layout: PipelineLayout,
    *,
    outcome_name: str = "outcome_1",
    train_df: pl.DataFrame | None = None,
) -> dict[str, pl.DataFrame]:
    stage_dir = root / layout.stage_dir("s03_divide", outcome_name)
    stage_dir.mkdir(parents=True)
    frames = {
        "feature_head_df": _features().head(3),
        "outcome_head_df": _keys().head(3),
        "train_df": _split_frame() if train_df is None else train_df,
        "valid_df": _split_frame(),
        "test_df": _split_frame(),
    }
    for logical, frame in frames.items():
        frame.write_parquet(root / layout.artifact("s03_divide", logical, outcome_name))
    return frames


def test_load_split_tables_reads_declared_artifacts(tmp_path: Path) -> None:
    layout = _layout()
    frames = _write_split_artifacts(tmp_path, layout)

    tables = load_split_tables(layout, "outcome_1", tmp_path)

    assert tables.key_colnames == list(KEY_COLUMNS)
    assert tables.feature_head_df.equals(frames["feature_head_df"])
    assert tables.outcome_head_df.equals(frames["outcome_head_df"])
    assert tables.train_df.equals(frames["train_df"])
    assert tables.valid_df.equals(frames["valid_df"])
    assert tables.test_df.equals(frames["test_df"])


def test_load_split_tables_validates_split_partitions(tmp_path: Path) -> None:
    layout = _layout()
    malformed = _split_frame().with_columns(
        pl.lit("not-a-feature").alias(FEATURE_COLNAMES[0])
    )
    _write_split_artifacts(tmp_path, layout, train_df=malformed)

    with pytest.raises(BoundaryError) as excinfo:
        load_split_tables(layout, "outcome_1", tmp_path)

    assert S03_SPLIT_READ_BOUNDARY.format(partition="train_df") in str(excinfo.value)
    assert excinfo.value.column == FEATURE_COLNAMES[0]
