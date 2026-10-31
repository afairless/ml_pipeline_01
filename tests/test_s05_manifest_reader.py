"""Integration tests for the s05 manifest reader.

``load_model_run`` reads one trained model run from its ``manifest.json``
(written by ``src.project.artifacts.save_results``): the split tables and the
model are resolved through the manifest's declared artifact inventory, and
model identity/``task``/``objective`` come from the manifest — never from
scanning filenames or parsing a directory name. The reader also reconstructs
the model through the estimator registry: ``load_model_run`` deserializes the
declared ``model.json`` via ``estimator_registry.load(manifest.estimator,
path)`` as an ``XGBoostAdapter``, whose derived ``task`` is validated against
the manifest's.

These tests exercise the reader against real ``save_results``-produced run
directories (the producer/reader round trip), and pin the failure modes: a
missing run directory, a declared artifact missing from disk, an artifact the
manifest never declared, an unknown estimator id, and an unknown ``task`` all
raise instead of silently loading a wrong or partial run.

``load_model_run`` is the only loading path in
``src.stages.s05_predict_model``: no filename-chain reader, container, or
wrapper-type detector remains in the module.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from src.core.estimators.xgboost import XGBoostAdapter
from src.core.manifest import (
    Manifest,
    load_model_runs,
    read_manifest,
    write_manifest,
)
from src.project.artifacts import save_results
from src.project.outcomes import modeled_outcomes
from src.project.schemas import FEATURE_COLUMNS
from src.project.training import ModelReadyData
from src.stages.s05_predict_model import LoadedModelRun, load_model_run

FEATURE_COLNAMES = list(FEATURE_COLUMNS[:2])
"""The model-input columns; the saved partition matrix carries all features."""

OUTCOME_NAME = modeled_outcomes()[0].name
"""The modeled outcome the tiny test runs use."""

_TINY_N = 16
"""Enough varied rows that xgboost grows splits on the tiny design matrix."""


def _keys() -> pl.DataFrame:
    """A tiny unique key frame with the shared key dtypes."""
    return pl.DataFrame(
        {
            "case_id": [f"case_{index:04d}" for index in range(_TINY_N)],
            "case_date": pl.Series(list(range(_TINY_N))).cast(pl.Date),
        }
    )


def _full_features() -> pl.DataFrame:
    """The complete declared feature matrix, in head order."""
    base = np.linspace(-1.0, 1.0, _TINY_N)
    return pl.DataFrame(
        {name: base + index * 0.1 for index, name in enumerate(FEATURE_COLUMNS)}
    )


def _make_model_ready_data(*, binary: bool = False) -> ModelReadyData:
    """A small model-ready value whose key/feature frames save cleanly."""
    keys = _keys()
    features = _full_features()
    selected = features.select(FEATURE_COLNAMES)
    if binary:
        outcome = pl.DataFrame(
            {
                OUTCOME_NAME: pl.Series(
                    [1.0 if index % 2 == 0 else 0.0 for index in range(_TINY_N)]
                )
            }
        )
    else:
        outcome = pl.DataFrame(
            {OUTCOME_NAME: 2.0 * selected[FEATURE_COLNAMES[0]].to_numpy()}
        )
    key_feature_df = pl.concat([keys, features], how="horizontal")

    return ModelReadyData(
        train_key_df=keys,
        train_feature_df=key_feature_df,
        train_feature_select_df=selected,
        train_outcome_select_df=outcome,
        x_train=selected.to_numpy(),
        y_train=outcome.to_numpy(),
        valid_key_df=keys,
        valid_feature_df=key_feature_df,
        valid_feature_select_df=selected,
        valid_outcome_select_df=outcome,
        x_valid=selected.to_numpy(),
        y_valid=outcome.to_numpy(),
        test_key_df=keys,
        test_feature_df=key_feature_df,
        test_feature_select_df=selected,
        test_outcome_select_df=outcome,
        x_test=selected.to_numpy(),
        y_test=outcome.to_numpy(),
    )


def _feature_df() -> pl.DataFrame:
    """The full feature frame ``save_results`` derives importances from."""
    return _full_features()


def _train_tiny_model(*, binary: bool = False) -> xgb.Booster:
    """A small booster on the tiny feature data, regression or binary."""
    mrd = _make_model_ready_data(binary=binary)
    dmtx = xgb.DMatrix(mrd.x_train, label=mrd.y_train)
    objective = "binary:logistic" if binary else "reg:absoluteerror"
    return xgb.train(
        {"objective": objective, "max_depth": 2},
        dmtx,
        num_boost_round=3,
    )


def _write_tiny_run(
    tmp_path: Path,
    *,
    strategy_id: str = "model_00a",
    outcome_name: str = OUTCOME_NAME,
    binary: bool = False,
) -> Path:
    """Run ``save_results`` on a tiny trained model; return its run dir."""
    output_dir = tmp_path / strategy_id
    output_dir.mkdir(parents=True, exist_ok=True)
    objective = "binary:logistic" if binary else "reg:absoluteerror"
    save_results(
        _train_tiny_model(binary=binary),
        {"learning_rate": 0.01},
        _make_model_ready_data(binary=binary),
        _feature_df(),
        FEATURE_COLNAMES,
        output_dir,
        strategy_id=strategy_id,
        outcome_name=outcome_name,
        objective=objective,
    )
    return output_dir


def _load_run(
    tmp_path: Path,
    *,
    strategy_id: str = "model_00a",
    outcome_name: str = OUTCOME_NAME,
    binary: bool = False,
) -> tuple[Path, LoadedModelRun]:
    """Produce a tiny run and load it back through the real discovery path."""
    run_dir = _write_tiny_run(
        tmp_path,
        strategy_id=strategy_id,
        outcome_name=outcome_name,
        binary=binary,
    )
    (manifest,) = load_model_runs(tmp_path, outcome_name)
    return run_dir, load_model_run(manifest, tmp_path)


def _rewrite_manifest(run_dir: Path, manifest: Manifest) -> None:
    """Replace the run's manifest file with ``manifest`` (same path)."""
    write_manifest(manifest, run_dir / "manifest.json")


# --- happy path: producer/reader round trip ---------------------------------


def test_load_model_run_reads_split_tables_from_manifest(tmp_path: Path) -> None:
    """The reader loads the same split tables the producer saved, keyed by
    the manifest's declared partition names."""
    _, loaded = _load_run(tmp_path)
    mrd = _make_model_ready_data()

    assert loaded.key_data["train"].equals(mrd.train_key_df)
    assert loaded.key_data["valid"].equals(mrd.valid_key_df)
    assert loaded.key_data["test"].equals(mrd.test_key_df)
    assert loaded.feature_data["train"].equals(mrd.train_feature_select_df)
    assert loaded.feature_data["valid"].equals(mrd.valid_feature_select_df)
    assert loaded.feature_data["test"].equals(mrd.test_feature_select_df)


def test_partitions_come_from_the_manifest(tmp_path: Path) -> None:
    """The run's partitions are the manifest's declared ``partitions``: the
    reader loads one key/feature pair per declared partition — the manifest
    owns the split shape, not a hardcoded partition list."""
    _, loaded = _load_run(tmp_path)

    assert loaded.manifest.partitions == ("train", "valid", "test")
    assert list(loaded.key_data) == list(loaded.manifest.partitions)
    assert list(loaded.feature_data) == list(loaded.manifest.partitions)


def test_loaded_model_type_follows_manifest_task(tmp_path: Path) -> None:
    """The reader reconstructs the model through the estimator registry as an
    ``XGBoostAdapter``; its derived ``task`` follows the manifest's
    regression/binary declaration — never a raw xgboost wrapper."""
    reg_stage = tmp_path / "reg_stage"
    bin_stage = tmp_path / "bin_stage"

    _, reg_run = _load_run(reg_stage)
    assert isinstance(reg_run.loaded_model, XGBoostAdapter)
    assert not isinstance(reg_run.loaded_model, xgb.XGBRegressor)
    assert reg_run.loaded_model.task == "regression"
    assert reg_run.loaded_model.task == reg_run.manifest.task

    _, bin_run = _load_run(bin_stage, binary=True)
    assert isinstance(bin_run.loaded_model, XGBoostAdapter)
    assert not isinstance(bin_run.loaded_model, xgb.XGBClassifier)
    assert bin_run.loaded_model.task == "binary"
    assert bin_run.loaded_model.task == bin_run.manifest.task


def test_identity_and_objective_come_from_manifest(tmp_path: Path) -> None:
    """Model identity, outcome, objective, and task all come from the
    manifest, not from any file or directory name."""
    _, loaded = _load_run(tmp_path, strategy_id="model_01")

    assert loaded.manifest.model_name == "model_01"
    assert loaded.manifest.outcome == OUTCOME_NAME
    assert loaded.manifest.objective == "reg:absoluteerror"
    assert loaded.manifest.task == "regression"
    assert loaded.manifest.feature_colnames == tuple(FEATURE_COLNAMES)
    assert loaded.run_dir == tmp_path / "model_01"


def test_identity_does_not_depend_on_directory_name(tmp_path: Path) -> None:
    """A run directory whose name carries no model prefix still loads:
    identity comes from the manifest only."""
    _, loaded = _load_run(tmp_path, strategy_id="run-x")

    assert loaded.manifest.model_name == "run-x"
    assert loaded.run_dir == tmp_path / "run-x"


def test_reader_does_not_touch_the_pretty_dump(tmp_path: Path) -> None:
    """Deleting the pretty/text model dumps does not affect loading: the
    reader resolves only the declared plain JSON serialization."""
    run_dir, _ = _load_run(tmp_path)
    (run_dir / "model_pretty.json").unlink()
    (run_dir / "model.txt").unlink()

    (manifest,) = load_model_runs(tmp_path, OUTCOME_NAME)
    loaded = load_model_run(manifest, tmp_path)

    assert isinstance(loaded.loaded_model, XGBoostAdapter)
    assert loaded.loaded_model.task == "regression"


def test_loaded_model_predicts_on_run_features(tmp_path: Path) -> None:
    """The reconstructed model predicts on the run's feature tables with the
    expected output shape (regression: one value per row)."""
    _, loaded = _load_run(tmp_path)

    predictions = loaded.loaded_model.predict(loaded.feature_data["train"].to_numpy())

    assert predictions.shape == (_TINY_N,)
    assert np.isfinite(predictions).all()


def test_composed_with_load_model_runs_is_sorted(tmp_path: Path) -> None:
    """``load_model_runs`` plus ``load_model_run`` reproduce the sorted run
    order the s05 stage loop relies on."""
    _write_tiny_run(tmp_path, strategy_id="model_01")
    _write_tiny_run(tmp_path, strategy_id="model_00a")

    loaded_runs = [
        load_model_run(manifest, tmp_path)
        for manifest in load_model_runs(tmp_path, OUTCOME_NAME)
    ]

    assert [run.manifest.model_name for run in loaded_runs] == [
        "model_00a",
        "model_01",
    ]


# --- failure modes ----------------------------------------------------------


def test_missing_run_dir_raises(tmp_path: Path) -> None:
    """A manifest whose declared run directory is absent raises instead of
    silently loading nothing."""
    _write_tiny_run(tmp_path)
    (manifest,) = load_model_runs(tmp_path, OUTCOME_NAME)
    empty_stage = tmp_path / "empty_stage"
    empty_stage.mkdir()

    with pytest.raises(FileNotFoundError, match="run directory"):
        load_model_run(manifest, empty_stage)


def test_missing_declared_artifact_raises(tmp_path: Path) -> None:
    """A declared artifact missing from disk raises ``FileNotFoundError``
    instead of loading a partial run."""
    run_dir, _ = _load_run(tmp_path)
    (run_dir / "train_key_df.parquet").unlink()

    (manifest,) = load_model_runs(tmp_path, OUTCOME_NAME)
    with pytest.raises(FileNotFoundError, match="train_key_df"):
        load_model_run(manifest, tmp_path)


def test_undeclared_artifact_raises(tmp_path: Path) -> None:
    """A manifest that never declared a logical artifact the stage needs
    raises ``ValueError``: the reader never guesses a filename."""
    run_dir = _write_tiny_run(tmp_path)
    manifest = read_manifest(run_dir / "manifest.json")
    _rewrite_manifest(
        run_dir,
        replace(
            manifest,
            artifacts=tuple(
                ref
                for ref in manifest.artifacts
                if ref.logical_name != "valid_feature_df"
            ),
        ),
    )

    (manifest,) = load_model_runs(tmp_path, OUTCOME_NAME)
    with pytest.raises(ValueError, match="valid_feature_df"):
        load_model_run(manifest, tmp_path)


def test_unknown_task_rejected(tmp_path: Path) -> None:
    """A manifest recording a task the loaded model does not report is
    rejected at load: the adapter derives its own task from the serialized
    model, and the manifest's declared ``task`` must agree."""
    run_dir = _write_tiny_run(tmp_path)
    manifest = read_manifest(run_dir / "manifest.json")
    _rewrite_manifest(run_dir, replace(manifest, task="ranking"))

    (manifest,) = load_model_runs(tmp_path, OUTCOME_NAME)
    with pytest.raises(ValueError, match="task"):
        load_model_run(manifest, tmp_path)


def test_unknown_estimator_id_rejected(tmp_path: Path) -> None:
    """A manifest recording an estimator id the registry does not know raises
    at the registry boundary, never a silent fallback."""
    run_dir = _write_tiny_run(tmp_path)
    manifest = read_manifest(run_dir / "manifest.json")
    _rewrite_manifest(run_dir, replace(manifest, estimator="lightgbm"))

    (manifest,) = load_model_runs(tmp_path, OUTCOME_NAME)
    with pytest.raises(ValueError, match="unknown estimator id"):
        load_model_run(manifest, tmp_path)


def test_registered_id_predictions_match_direct_construction(
    tmp_path: Path,
) -> None:
    """Predictions through the registered ``xgboost`` adapter are exactly the
    values a hand-constructed reference wrapper produces: registering the
    estimator id must not change any prediction."""
    reg_run_dir, reg_loaded = _load_run(tmp_path)
    reg_reference = xgb.XGBRegressor()
    reg_reference.load_model(reg_run_dir / "model.json")
    reg_x = reg_loaded.feature_data["train"].to_numpy()
    np.testing.assert_array_equal(
        reg_loaded.loaded_model.predict(reg_x),
        reg_reference.predict(reg_x),
    )

    bin_run_dir, bin_loaded = _load_run(tmp_path / "bin_stage", binary=True)
    bin_reference = xgb.XGBClassifier()
    bin_reference.load_model(bin_run_dir / "model.json")
    bin_x = bin_loaded.feature_data["valid"].to_numpy()
    np.testing.assert_array_equal(
        bin_loaded.loaded_model.predict_proba(bin_x)[:, 1],
        bin_reference.predict_proba(bin_x)[:, 1],
    )


# --- the filename-chain reader is absent ------------------------------------


def test_filename_chain_reader_is_absent() -> None:
    """The old filename-chain reader, its container, and its wrapper-type
    detector are absent: model identity now comes from each run's manifest."""
    import src.stages.s05_predict_model as s05

    assert not hasattr(s05, "load_model_data")
    assert not hasattr(s05, "ModelData")
    assert not hasattr(s05, "detect_xgb_classifier_or_regressor")
