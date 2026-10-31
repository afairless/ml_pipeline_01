"""Tests for the project training driver and its data-preparation helpers.

Three layers are covered:

* ``divide_sort_train_valid_test_data`` cuts each split partition by the
  feature/outcome heads and sorts every frame on the canonical key columns, so
  the feature and outcome halves stay row-aligned regardless of on-disk row
  order.
* ``prepare_model_ready_data`` separates keys from model data, applies the
  strategy's estimator input order, keeps the full partition feature frames
  for the artifact writer, and builds the design/target matrices.
* ``train_model_run`` runs end to end on staged s03 split tables through both
  trainer kinds and across feature policies: the run directory receives the
  declared artifacts and a model manifest whose identity, task, feature order,
  and tuned parameters match the strategy and context it was given.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.core.estimator import estimator_registry
from src.core.manifest import MANIFEST_FILENAME, ModelManifest, read_manifest
from src.core.paths import PipelineContext, PipelineLayout, StageSpec
from src.core.splits import PARTITION_ORDER
from src.project.columns import CASE_DATE, CASE_ID, KEY_COLUMNS
from src.project.config import CONFIG, ProjectConfig
from src.project.schemas import FEATURE_COLUMNS
from src.project.training import (
    ModelReadyData,
    TVTFeatureOutcomeData,
    divide_sort_train_valid_test_data,
    prepare_model_ready_data,
    train_model_run,
)

OUTCOME_NAME = "outcome_1"
FEATURE_COLNAMES = list(FEATURE_COLUMNS)
_TRAIN_COUNT = 12
_VALID_COUNT = 6
_TEST_COUNT = 6


def _keys(count: int, start: date) -> pl.DataFrame:
    return pl.DataFrame(
        {
            CASE_ID: [f"case_{index:04d}" for index in range(count)],
            CASE_DATE: [start + timedelta(days=index) for index in range(count)],
        }
    )


def _features(count: int, start: date) -> pl.DataFrame:
    """A deterministic, non-degenerate feature matrix with unique keys."""
    base = np.linspace(-1.0, 1.0, count)
    features = {
        name: base * (index + 1) + float(index)
        for index, name in enumerate(FEATURE_COLNAMES)
    }
    return _keys(count, start).hstack(pl.DataFrame(features))


def _outcome(features: pl.DataFrame, outcome_name: str = OUTCOME_NAME) -> pl.Series:
    """A deterministic target that depends on the feature columns."""
    return (
        2.0 * features[FEATURE_COLNAMES[0]]
        - features[FEATURE_COLNAMES[1]]
        + 0.5 * features[FEATURE_COLNAMES[2]]
        + 0.25
    ).alias(outcome_name)


def _split_frames() -> dict[str, pl.DataFrame]:
    """The five s03 artifacts: two heads and the three partitions."""
    train = _features(_TRAIN_COUNT, date(2023, 12, 1)).with_columns(
        _outcome(_features(_TRAIN_COUNT, date(2023, 12, 1)))
    )
    valid = _features(_VALID_COUNT, date(2024, 2, 1)).with_columns(
        _outcome(_features(_VALID_COUNT, date(2024, 2, 1)))
    )
    test = _features(_TEST_COUNT, date(2024, 7, 1)).with_columns(
        _outcome(_features(_TEST_COUNT, date(2024, 7, 1)))
    )
    feature_head = train.select([*KEY_COLUMNS, *FEATURE_COLNAMES])
    outcome_head = train.select([*KEY_COLUMNS, OUTCOME_NAME])
    return {
        "feature_head_df": feature_head,
        "outcome_head_df": outcome_head,
        "train_df": train,
        "valid_df": valid,
        "test_df": test,
    }


def _layout() -> PipelineLayout:
    """The minimal s03/s04 stage table the driver resolves paths against."""
    return PipelineLayout(
        stages={
            "s03_divide": StageSpec(
                name="s03_divide",
                dir_template="s03_divide/{outcome}",
                artifacts={
                    "feature_head_df": "feature_head_df.parquet",
                    "outcome_head_df": "outcome_head_df.parquet",
                    "train_df": "train_df.parquet",
                    "valid_df": "valid_df.parquet",
                    "test_df": "test_df.parquet",
                },
            ),
            "s04_train": StageSpec(
                name="s04_train",
                dir_template="s04_train/{outcome}",
                artifacts={},
            ),
        },
        outcome_names=frozenset({"outcome_1", "outcome_3"}),
    )


def _test_config() -> ProjectConfig:
    """Small, fast estimator settings; the frozen fields stay the same."""
    return replace(
        CONFIG,
        xgb_single_fit_params={
            "max_depth": 2,
            "learning_rate": 0.1,
            "nthread": 1,
        },
        xgb_num_boost_round=5,
        xgb_early_stopping_rounds=2,
    )


def _search_config() -> ProjectConfig:
    """A one-candidate, two-fold search so the dispatch branch stays cheap."""
    return replace(
        _test_config(),
        xgb_param_distributions={"max_depth": [1, 2], "n_estimators": [2]},
        xgb_search_n_iter=1,
        xgb_search_cv=2,
        xgb_regressor_seed=3,
    )


def _write_split_artifacts(
    output_root: Path,
    layout: PipelineLayout,
    *,
    frames: dict[str, pl.DataFrame] | None = None,
    outcome_name: str = OUTCOME_NAME,
) -> None:
    stage_dir = output_root / layout.stage_dir("s03_divide", outcome_name)
    stage_dir.mkdir(parents=True)
    written = _split_frames() if frames is None else frames
    for logical, frame in written.items():
        frame.write_parquet(
            output_root / layout.artifact("s03_divide", logical, outcome_name)
        )


def _run(
    tmp_path: Path,
    strategy_id: str,
    *,
    config: ProjectConfig | None = None,
    outcome_name: str = OUTCOME_NAME,
) -> tuple[PipelineLayout, Path]:
    layout = _layout()
    _write_split_artifacts(tmp_path, layout, outcome_name=outcome_name)
    context = PipelineContext(layout=layout, config=config or _test_config())
    output_dir = tmp_path / layout.stage_dir("s04_train", outcome_name) / strategy_id
    context.ensure(output_dir)
    train_model_run(strategy_id, outcome_name, context, tmp_path)
    return layout, output_dir


def _manifest_of(output_dir: Path) -> ModelManifest:
    manifest = read_manifest(output_dir / MANIFEST_FILENAME)
    assert isinstance(manifest, ModelManifest)
    return manifest


# --- divide / sort ---------------------------------------------------------


def test_divide_sorts_partitions_and_selects_heads() -> None:
    frames = _split_frames()
    shuffled = frames["train_df"].reverse()

    tvt = divide_sort_train_valid_test_data(
        shuffled,
        frames["valid_df"],
        frames["test_df"],
        frames["feature_head_df"],
        frames["outcome_head_df"],
        list(KEY_COLUMNS),
    )

    assert isinstance(tvt, TVTFeatureOutcomeData)
    assert tvt.train_feature_df.columns == list(frames["feature_head_df"].columns)
    assert tvt.train_outcome_df.columns == list(frames["outcome_head_df"].columns)
    assert tvt.train_feature_df[CASE_DATE].to_list() == sorted(
        tvt.train_feature_df[CASE_DATE].to_list()
    )
    # The feature and outcome halves address the same cases in the same order.
    assert tvt.train_feature_df[CASE_ID].to_list() == (
        tvt.train_outcome_df[CASE_ID].to_list()
    )
    assert tvt.train_feature_df.height == _TRAIN_COUNT
    assert tvt.valid_feature_df.height == _VALID_COUNT
    assert tvt.test_feature_df.height == _TEST_COUNT


def test_divide_keeps_feature_and_outcome_frames_aligned_on_valid_and_test() -> None:
    frames = _split_frames()
    tvt = divide_sort_train_valid_test_data(
        frames["train_df"].reverse(),
        frames["valid_df"].reverse(),
        frames["test_df"].reverse(),
        frames["feature_head_df"],
        frames["outcome_head_df"],
        list(KEY_COLUMNS),
    )

    for feature_df, outcome_df in (
        (tvt.valid_feature_df, tvt.valid_outcome_df),
        (tvt.test_feature_df, tvt.test_outcome_df),
    ):
        assert feature_df[CASE_ID].to_list() == outcome_df[CASE_ID].to_list()
        assert feature_df[CASE_DATE].to_list() == outcome_df[CASE_DATE].to_list()


# --- prepare ---------------------------------------------------------------


def _prepared(feature_colnames: list[str] | None = None) -> ModelReadyData:
    frames = _split_frames()
    tvt = divide_sort_train_valid_test_data(
        frames["train_df"],
        frames["valid_df"],
        frames["test_df"],
        frames["feature_head_df"],
        frames["outcome_head_df"],
        list(KEY_COLUMNS),
    )
    return prepare_model_ready_data(
        tvt.train_feature_df,
        tvt.train_outcome_df,
        tvt.valid_feature_df,
        tvt.valid_outcome_df,
        tvt.test_feature_df,
        tvt.test_outcome_df,
        list(KEY_COLUMNS),
        feature_colnames or FEATURE_COLNAMES,
        OUTCOME_NAME,
    )


def test_prepare_builds_key_selections_and_arrays() -> None:
    mrd = _prepared()

    assert mrd.train_key_df.columns == list(KEY_COLUMNS)
    assert mrd.train_feature_select_df.columns == FEATURE_COLNAMES
    assert mrd.train_outcome_select_df.columns == [OUTCOME_NAME]
    assert mrd.x_train.shape == (_TRAIN_COUNT, len(FEATURE_COLNAMES))
    assert mrd.y_train.shape == (_TRAIN_COUNT, 1)
    assert mrd.x_valid.shape == (_VALID_COUNT, len(FEATURE_COLNAMES))
    assert mrd.x_test.shape == (_TEST_COUNT, len(FEATURE_COLNAMES))
    assert np.isfinite(mrd.x_train).all()
    assert np.isfinite(mrd.y_train).all()


def test_prepare_keeps_full_feature_frames_and_applies_model_order() -> None:
    reversed_features = list(reversed(FEATURE_COLNAMES))
    mrd = _prepared(reversed_features)

    # The model selection follows the declared estimator order ...
    assert mrd.train_feature_select_df.columns == reversed_features
    # ... while the full partition frame keeps every feature in head order for
    # the artifact writer.
    assert mrd.train_feature_df.columns == [*KEY_COLUMNS, *FEATURE_COLNAMES]
    assert mrd.train_feature_df.equals(_split_frames()["train_df"].drop(OUTCOME_NAME))
    assert mrd.x_train.shape == (_TRAIN_COUNT, len(FEATURE_COLNAMES))


# --- train_model_run -------------------------------------------------------


def test_train_model_run_single_fit_writes_manifest_and_artifacts(
    tmp_path: Path,
) -> None:
    config = _test_config()
    _, output_dir = _run(tmp_path, "model_00_signal_0", config=config)
    manifest = _manifest_of(output_dir)

    assert manifest.model_name == "model_00_signal_0"
    assert manifest.stage == "s04_train"
    assert manifest.outcome == OUTCOME_NAME
    assert manifest.estimator == "xgboost"
    assert manifest.task == "regression"
    assert manifest.objective == "reg:absoluteerror"
    assert manifest.outcome_colname == OUTCOME_NAME
    assert manifest.feature_colnames == tuple(FEATURE_COLNAMES)
    assert manifest.partitions == PARTITION_ORDER
    assert manifest.best_parameters == config.xgb_single_fit_params

    # Every declared artifact exists, and the partitions carry the full
    # feature matrix.
    for artifact in manifest.artifacts:
        assert (output_dir / artifact.filename).is_file()
    for partition in PARTITION_ORDER:
        feature_df = pl.read_parquet(output_dir / f"{partition}_feature_df.parquet")
        assert feature_df.columns == [*KEY_COLUMNS, *FEATURE_COLNAMES]
        key_df = pl.read_parquet(output_dir / f"{partition}_key_df.parquet")
        assert key_df.columns == list(KEY_COLUMNS)


def test_train_model_run_persists_a_loadable_model(tmp_path: Path) -> None:
    config = _test_config()
    _, output_dir = _run(tmp_path, "model_00_signal_0", config=config)
    manifest = _manifest_of(output_dir)

    adapter = estimator_registry.load("xgboost", output_dir / "model.json")
    predictions = adapter.predict(_prepared().x_valid)
    assert predictions.shape[0] == _VALID_COUNT
    assert np.isfinite(predictions).all()
    assert adapter.task == "regression"
    assert manifest.task == adapter.task


def test_train_model_run_subset_strategy_keeps_full_partition_features(
    tmp_path: Path,
) -> None:
    """A feature-subset model records only its inputs but persists all
    partition features, so the s04 boundary schema holds for every run."""
    _, output_dir = _run(tmp_path, "model_00_noise_1")
    manifest = _manifest_of(output_dir)

    assert manifest.feature_colnames == ("feature_1", "feature_8")
    train_features = pl.read_parquet(output_dir / "train_feature_df.parquet")
    assert train_features.columns == [*KEY_COLUMNS, *FEATURE_COLNAMES]
    importances = pl.read_parquet(output_dir / "feature_importances.parquet")
    assert set(importances["feature"].to_list()) <= {"feature_1", "feature_8"}


def test_train_model_run_outcome_plus_noise_uses_policy_feature_order(
    tmp_path: Path,
) -> None:
    _, output_dir = _run(tmp_path, "model_00_signal_1")
    manifest = _manifest_of(output_dir)

    # OutcomePlusNoise exposes the estimator order [second, first].
    assert manifest.feature_colnames == ("feature_2", "feature_1")
    train_features = pl.read_parquet(output_dir / "train_feature_df.parquet")
    assert train_features.columns == [*KEY_COLUMNS, *FEATURE_COLNAMES]


def test_train_model_run_search_strategy_records_tuned_parameters(
    tmp_path: Path,
) -> None:
    config = _search_config()
    _, output_dir = _run(tmp_path, "model_01a", config=config)
    manifest = _manifest_of(output_dir)

    assert manifest.model_name == "model_01a"
    assert manifest.task == "regression"
    assert manifest.feature_colnames == tuple(FEATURE_COLNAMES)
    assert set(manifest.best_parameters) <= {"max_depth", "n_estimators"}
    assert manifest.best_parameters


def test_train_model_run_rejects_unknown_strategy(tmp_path: Path) -> None:
    layout = _layout()
    _write_split_artifacts(tmp_path, layout)
    context = PipelineContext(layout=layout, config=_test_config())

    with pytest.raises(ValueError, match="unknown model strategy"):
        train_model_run("model_99_missing", OUTCOME_NAME, context, tmp_path)


def test_train_model_run_writes_into_the_strategy_run_directory(
    tmp_path: Path,
) -> None:
    layout, output_dir = _run(tmp_path, "model_02_subset_a")

    expected = tmp_path / layout.stage_dir("s04_train", OUTCOME_NAME)
    assert output_dir.parent == expected
    assert output_dir.name == "model_02_subset_a"
    assert (output_dir / MANIFEST_FILENAME).is_file()
