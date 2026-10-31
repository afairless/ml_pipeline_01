"""Integration tests for the prediction-manifest producer.

``save_prediction_run`` writes one model run's partition prediction artifacts
into its per-model run directory (``s05_predict/{outcome}/{model_name}/``,
mirroring the s04 run-dir layout) and assembles the run's
``PredictionManifest``: ``manifest_kind='predictions'``,
``stage='s05_predict'``, ``model_name``, ``outcome``, the declared
``partitions``, and one partition-tagged ``ArtifactRef`` per written file. The
s05 stage main resolves the run dir from the layout, ``ensure``s it, and
closes the run with ``write_manifest(run_dir / 'manifest.json')``.

These tests exercise the producer against a real trained-and-loaded model run
and pin the manifest fields, the write/read round trip through
``read_manifest`` and ``load_prediction_runs``, and the on-disk inventory
match: every written file is declared exactly once, nothing undeclared is
recorded, and a mismatched partition table count is rejected up front.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from src.core.estimators.xgboost import XGBoostAdapter
from src.core.manifest import (
    MANIFEST_FILENAME,
    PREDICTION_SCHEMA_VERSION,
    PredictionManifest,
    load_model_runs,
    load_prediction_runs,
    read_manifest,
    write_manifest,
)
from src.project.artifacts import save_results
from src.project.outcomes import modeled_outcomes
from src.project.pipeline import build_context
from src.project.schemas import FEATURE_COLUMNS
from src.project.training import ModelReadyData
from src.stages.s05_predict_model import (
    LoadedModelRun,
    generate_and_save_predictions,
    load_model_run,
    save_prediction_run,
)
from src.stages.s05_predict_model_main import main

FEATURE_COLNAMES = list(FEATURE_COLUMNS[:2])
"""The model-input columns; the saved partition matrix carries all features."""

OUTCOME_NAME = modeled_outcomes()[0].name
"""The modeled outcome the tiny test runs use."""

_TINY_N = 16
"""Enough varied rows that xgboost grows splits on the tiny design matrix."""

PARTITIONS = ("train", "valid", "test")
"""The three prediction partitions the producer writes, in write order."""


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


def _load_tiny_model_run(
    tmp_path: Path,
    *,
    strategy_id: str = "model_00a",
    outcome_name: str = OUTCOME_NAME,
    binary: bool = False,
) -> tuple[Path, LoadedModelRun]:
    """s04-produce a tiny run and load it back through the real discovery
    path (``load_model_runs`` plus ``load_model_run``), mirroring the s05
    stage."""
    model_run_dir = tmp_path / "s04_stage" / strategy_id
    model_run_dir.mkdir(parents=True, exist_ok=True)
    objective = "binary:logistic" if binary else "reg:absoluteerror"
    save_results(
        _train_tiny_model(binary=binary),
        {"learning_rate": 0.01},
        _make_model_ready_data(binary=binary),
        _feature_df(),
        FEATURE_COLNAMES,
        model_run_dir,
        strategy_id=strategy_id,
        outcome_name=outcome_name,
        objective=objective,
    )
    (manifest,) = load_model_runs(tmp_path / "s04_stage", outcome_name)
    return model_run_dir, load_model_run(manifest, tmp_path / "s04_stage")


def _produce_prediction_run(
    tmp_path: Path,
    *,
    strategy_id: str = "model_00a",
    outcome_name: str = OUTCOME_NAME,
    binary: bool = False,
) -> tuple[Path, PredictionManifest]:
    """Run ``save_prediction_run`` on a loaded tiny run; return the run dir
    and the assembled manifest (not yet written by the caller)."""
    _, loaded = _load_tiny_model_run(
        tmp_path,
        strategy_id=strategy_id,
        outcome_name=outcome_name,
        binary=binary,
    )
    run_dir = tmp_path / "s05_stage" / strategy_id
    run_dir.mkdir(parents=True)
    manifest = save_prediction_run(
        loaded.loaded_model,
        loaded.manifest.task,
        loaded.manifest.partitions,
        [loaded.key_data[partition] for partition in loaded.manifest.partitions],
        [loaded.feature_data[partition] for partition in loaded.manifest.partitions],
        loaded.manifest.model_name,
        loaded.manifest.outcome,
        run_dir,
    )
    return run_dir, manifest


def _manifest_of(run_dir: Path) -> PredictionManifest:
    manifest = read_manifest(run_dir / MANIFEST_FILENAME)
    assert isinstance(manifest, PredictionManifest)
    return manifest


# --- the producer itself ----------------------------------------------------


def test_producer_writes_partition_artifacts_into_the_run_dir(
    tmp_path: Path,
) -> None:
    """The three partition prediction files (parquet plus CSV each) are
    written into the run dir, not the flat stage dir."""
    run_dir, _ = _produce_prediction_run(tmp_path)

    assert {path.name for path in run_dir.iterdir()} == {
        f"model_00a_{partition}_predictions{suffix}"
        for partition in PARTITIONS
        for suffix in (".parquet", ".csv")
    }


def test_manifest_records_identity_and_partitions(tmp_path: Path) -> None:
    """``model_name`` is the strategy id; kind/stage/outcome and the
    partition names are all recorded."""
    _, manifest = _produce_prediction_run(
        tmp_path, strategy_id="model_01", outcome_name=OUTCOME_NAME
    )

    assert manifest.schema_version == PREDICTION_SCHEMA_VERSION
    assert manifest.manifest_kind == "predictions"
    assert manifest.model_name == "model_01"
    assert manifest.stage == "s05_predict"
    assert manifest.outcome == OUTCOME_NAME
    assert manifest.partitions == PARTITIONS


def test_producer_writes_exactly_the_declared_partitions(
    tmp_path: Path,
) -> None:
    """The prediction producer predicts exactly the partition set its caller
    passes (the model run's declared ``ModelManifest.partitions``) — no
    hardcoded partition list inside the producer."""
    _, loaded = _load_tiny_model_run(tmp_path)
    run_dir = tmp_path / "s05_stage" / "model_00a"
    run_dir.mkdir(parents=True)
    declared = ("train", "valid")

    manifest = save_prediction_run(
        loaded.loaded_model,
        loaded.manifest.task,
        declared,
        [loaded.key_data[partition] for partition in declared],
        [loaded.feature_data[partition] for partition in declared],
        loaded.manifest.model_name,
        loaded.manifest.outcome,
        run_dir,
    )

    produced = {path.name for path in run_dir.iterdir()}
    assert produced == {
        f"model_00a_{partition}_predictions{suffix}"
        for partition in declared
        for suffix in (".parquet", ".csv")
    }
    assert manifest.partitions == declared
    assert {ref.partition for ref in manifest.artifacts} == set(declared)


def test_artifacts_are_partition_tagged(tmp_path: Path) -> None:
    """Every artifact carries its partition in the ref and in the filename;
    logical names follow the ``<partition>_predictions`` convention."""
    _, manifest = _produce_prediction_run(tmp_path)

    assert len(manifest.artifacts) == 3 * 2
    actual = {
        (ref.filename, ref.partition, ref.logical_name) for ref in manifest.artifacts
    }
    expected = {
        (
            f"model_00a_{partition}_predictions{suffix}",
            partition,
            f"{partition}_predictions",
        )
        for partition in PARTITIONS
        for suffix in (".parquet", ".csv")
    }

    assert actual == expected


def test_artifacts_inventory_matches_written_files(tmp_path: Path) -> None:
    """Every file the run wrote is declared by exactly one ``ArtifactRef``
    (and vice versa); the manifest never lists itself."""
    run_dir, manifest = _produce_prediction_run(tmp_path)
    write_manifest(manifest, run_dir / MANIFEST_FILENAME)

    produced = {path.name for path in run_dir.iterdir()}
    inventory = {ref.filename for ref in manifest.artifacts}

    assert inventory == produced - {MANIFEST_FILENAME}
    assert len(inventory) == len(produced) - 1


def test_producer_requires_one_table_per_partition(tmp_path: Path) -> None:
    """A mismatched number of key/feature tables is rejected up front."""
    _, loaded = _load_tiny_model_run(tmp_path)
    run_dir = tmp_path / "bad_run"
    run_dir.mkdir()

    with pytest.raises(AssertionError, match="per partition"):
        save_prediction_run(
            loaded.loaded_model,
            loaded.manifest.task,
            loaded.manifest.partitions,
            [loaded.key_data[partition] for partition in ("train", "valid")],
            [
                loaded.feature_data[partition]
                for partition in loaded.manifest.partitions
            ],
            loaded.manifest.model_name,
            loaded.manifest.outcome,
            run_dir,
        )


# --- round trip and discovery -----------------------------------------------


def test_write_read_round_trip_and_discovery(tmp_path: Path) -> None:
    """Writing the manifest into the run dir and reading it back preserves
    every field; ``load_prediction_runs`` discovers the run structurally."""
    run_dir, manifest = _produce_prediction_run(tmp_path)
    write_manifest(manifest, run_dir / MANIFEST_FILENAME)

    loaded = _manifest_of(run_dir)
    assert loaded == manifest
    assert isinstance(loaded.partitions, tuple)
    assert isinstance(loaded.artifacts, tuple)

    stage_dir = run_dir.parent
    runs = load_prediction_runs(stage_dir, OUTCOME_NAME)
    assert [run.model_name for run in runs] == ["model_00a"]
    assert runs[0] == manifest


def test_round_trip_preserves_artifact_set(tmp_path: Path) -> None:
    """The read-back manifest's partition-tagged refs are unchanged by the
    write/read boundary."""
    run_dir, manifest = _produce_prediction_run(tmp_path)
    write_manifest(manifest, run_dir / MANIFEST_FILENAME)

    loaded = _manifest_of(run_dir)

    assert [(ref.filename, ref.partition) for ref in loaded.artifacts] == [
        (ref.filename, ref.partition) for ref in manifest.artifacts
    ]


# --- regression predictions are the raw model output ------------------------


def test_regression_predictions_equal_raw_predict_output(
    tmp_path: Path,
) -> None:
    """The saved regression predictions are ``loaded_model.predict(x)`` with
    no intercept subtraction.

    The model is fit with a deliberately non-zero ``base_score`` so its
    ``intercept_`` is non-zero, pinning the fix: re-introducing a spurious
    intercept shift would move every saved value and fail this test.
    """
    feature_a = np.linspace(-1.0, 1.0, _TINY_N)
    feature_b = np.linspace(10.0, 20.0, _TINY_N)
    features = np.column_stack((feature_a, feature_b))
    outcome = 2.0 * feature_a + 5.0
    model = xgb.XGBRegressor(max_depth=2, n_estimators=3, base_score=13.0)
    model.fit(features, outcome)
    assert not np.allclose(model.intercept_, 0.0), (
        "test requires a non-zero intercept to pin the no-subtraction fix"
    )

    key_df = _keys()
    output_path = tmp_path / "predictions"
    output_path.mkdir()
    adapter = XGBoostAdapter(model=model)

    generate_and_save_predictions(
        adapter,
        "regression",
        features,
        key_df,
        "model_00a",
        "train",
        output_path,
    )

    saved = pl.read_parquet(output_path / "model_00a_train_predictions.parquet")
    assert saved.columns == ["case_id", "case_date", "predictions"]
    np.testing.assert_array_equal(
        saved["predictions"].to_numpy(), model.predict(features)
    )


def test_manifest_task_mismatch_with_adapter_rejected(
    tmp_path: Path,
) -> None:
    """``generate_and_save_predictions`` validates the manifest's ``task``
    against the adapter: a binary claim on a regression adapter raises before
    any prediction is written."""
    _, loaded = _load_tiny_model_run(tmp_path)
    output_path = tmp_path / "predictions"
    output_path.mkdir()

    with pytest.raises(ValueError, match="task"):
        generate_and_save_predictions(
            loaded.loaded_model,
            "binary",
            loaded.feature_data["train"].to_numpy(),
            loaded.key_data["train"],
            "model_00a",
            "train",
            output_path,
        )


def test_predictions_preserve_keys_and_split_sizes(tmp_path: Path) -> None:
    """The written train/valid/test tables keep the key columns plus the
    predictions column and match the split sizes the model data defined."""
    run_dir, _ = _produce_prediction_run(tmp_path)

    for partition in PARTITIONS:
        table = pl.read_parquet(run_dir / f"model_00a_{partition}_predictions.parquet")
        assert table.columns == ["case_id", "case_date", "predictions"]
        assert table.height == _TINY_N
        assert table["predictions"].is_null().sum() == 0


def test_binary_run_writes_classification_predictions(tmp_path: Path) -> None:
    """A binary-task run produces the same per-partition files; probability
    predictions stay within ``[0, 1]``."""
    run_dir, manifest = _produce_prediction_run(tmp_path, binary=True)

    assert manifest.partitions == PARTITIONS
    train_table = pl.read_parquet(run_dir / "model_00a_train_predictions.parquet")
    predictions = train_table["predictions"]

    assert cast(float, predictions.min()) >= 0.0
    assert cast(float, predictions.max()) <= 1.0


# --- the stage entry point discovers every model run ------------------------


def test_stage_main_predicts_every_declared_partition(tmp_path: Path) -> None:
    """The stage main discovers the s04 runs through their manifests, writes
    one prediction run per model, and closes each with its manifest."""
    output_root = tmp_path / "output"
    model_run_dir = output_root / "s04_train" / OUTCOME_NAME / "model_00a"
    model_run_dir.mkdir(parents=True)
    save_results(
        _train_tiny_model(),
        {"learning_rate": 0.01},
        _make_model_ready_data(),
        _feature_df(),
        FEATURE_COLNAMES,
        model_run_dir,
        strategy_id="model_00a",
        outcome_name=OUTCOME_NAME,
        objective="reg:absoluteerror",
    )

    context = build_context()
    main(OUTCOME_NAME, context, output_root)

    s05_dir = output_root / context.layout.stage_dir("s05_predict", OUTCOME_NAME)
    run_dir = s05_dir / "model_00a"
    manifest = _manifest_of(run_dir)

    assert manifest.manifest_kind == "predictions"
    assert manifest.model_name == "model_00a"
    assert manifest.outcome == OUTCOME_NAME
    assert manifest.partitions == PARTITIONS
    assert {ref.filename for ref in manifest.artifacts} == {
        f"model_00a_{partition}_predictions{suffix}"
        for partition in PARTITIONS
        for suffix in (".parquet", ".csv")
    }
    assert [run.model_name for run in load_prediction_runs(s05_dir, OUTCOME_NAME)] == [
        "model_00a"
    ]


def test_stage_main_rejects_unknown_outcome(tmp_path: Path) -> None:
    """The layout's ``{outcome}`` validation fires before any write."""
    context = build_context()

    with pytest.raises(ValueError, match="unknown outcome"):
        main("outcome_missing", context, tmp_path / "output")
