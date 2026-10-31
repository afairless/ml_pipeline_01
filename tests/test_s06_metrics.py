"""Unit tests for the s06 metric and report helpers.

Pins the helper contracts the evaluation driver relies on: the outcome-derived
evaluation column names, the log-transform column table and finite shared-shift
branch, quantile-bin averaging, prediction joins with the reporting inverse and
its non-positive mask, feature-importance ranking and plotting, and the
regression/binary metric writers. Binary labels are pinned to the shared strict
``value > threshold`` rule.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import polars as pl
import pytest
from matplotlib import pyplot as plt

from src.core.binary_metrics import threshold_labels
from src.core.metrics import RegressionMetrics
from src.project.config import LOG_PLOT_BASES
from src.project.outcomes import Outcome
from src.stages.s06_evaluate_model import (
    ORIGINAL_TARGET_PRED_COLNAME,
    PREDICTION_COLNAME,
    OutcomeColumnNames,
    calculate_log_predictions_and_actuals,
    calculate_predictions_and_actuals_statistic_by_uniform_bins,
    join_dataframe_with_predictions,
    log_base_suffix,
    log_plot_column_names,
    outcome_column_names,
    save_classification_model_prediction_results,
    save_feature_importance_plot,
    save_regression_model_prediction_results,
    top_feature_importances,
)


@pytest.fixture(autouse=True)
def _close_all_figures_after_each_test():
    """Never let a leaked figure contaminate the next test."""
    yield
    plt.close("all")


# --- outcome-derived column names ------------------------------------------


def test_identity_outcome_reports_on_the_model_scale() -> None:
    """An identity target is already on the reporting scale."""

    names = outcome_column_names(Outcome(name="outcome_1", modeled=True))

    assert names == OutcomeColumnNames(
        original_target_true_colname="outcome_1",
        original_target_pred_colname="predictions",
        model_target_true_colname="outcome_1",
        model_target_pred_colname="predictions",
    )


def test_non_identity_outcome_uses_the_backtransformed_column() -> None:
    """A transformed model target reports through the joined column."""

    outcome = Outcome(
        name="outcome_3",
        outcome_transform="log1p",
        reporting_backtransform="log1p",
        modeled=True,
    )

    names = outcome_column_names(outcome)

    assert names.original_target_true_colname == "outcome_3"
    assert names.original_target_pred_colname == ORIGINAL_TARGET_PRED_COLNAME
    assert names.model_target_true_colname == "outcome_3"
    assert names.model_target_pred_colname == PREDICTION_COLNAME


# --- log-scale helper ------------------------------------------------------


def _reference_log(value: float, base: float) -> float:
    return math.log(value, base)


def test_log_plot_columns_follow_the_configured_base_order() -> None:
    """The six expected names follow ``LOG_PLOT_BASES`` order."""

    assert LOG_PLOT_BASES == (10.0, 2.0)
    assert log_base_suffix(10.0) == "log10"
    assert log_base_suffix(2.0) == "log2"
    assert log_plot_column_names("original_target") == [
        "original_target_pred_log10",
        "original_target_true_log10",
        "original_target_pred_log2",
        "original_target_true_log2",
    ]


def test_positive_input_uses_direct_logs() -> None:
    """An all-positive common minimum takes the direct-log branch."""

    y_pred = pl.Series([1.0, 2.0])
    y_true = pl.Series([1.5, 4.0])

    log_df = calculate_log_predictions_and_actuals("original_target", y_pred, y_true)

    assert log_df.columns == log_plot_column_names("original_target")
    for base in LOG_PLOT_BASES:
        suffix = log_base_suffix(base)
        assert log_df[f"original_target_pred_{suffix}"].to_list() == pytest.approx(
            [_reference_log(value, base) for value in y_pred]
        )
        assert log_df[f"original_target_true_{suffix}"].to_list() == pytest.approx(
            [_reference_log(value, base) for value in y_true]
        )


def test_zero_minimum_uses_one_shared_shift() -> None:
    """A zero prediction maps to zero through the shared shift."""

    y_pred = pl.Series([0.0, 1.0, 2.0])
    y_true = pl.Series([0.5, 1.5, 2.5])
    shift = 1.0

    log_df = calculate_log_predictions_and_actuals("original_target", y_pred, y_true)

    for base in LOG_PLOT_BASES:
        suffix = log_base_suffix(base)
        pred_log = log_df[f"original_target_pred_{suffix}"]
        true_log = log_df[f"original_target_true_{suffix}"]

        assert pred_log.to_list() == pytest.approx(
            [_reference_log((value + shift) / shift, base) for value in y_pred]
        )
        assert true_log.to_list() == pytest.approx(
            [_reference_log((value + shift) / shift, base) for value in y_true]
        )
        true_min = true_log.min()
        assert isinstance(true_min, float)
        assert pred_log.min() == pytest.approx(0.0)
        assert true_min > 0.0


def test_negative_minimum_keeps_every_value_finite() -> None:
    """A negative common minimum shifts both series, never dropping rows."""

    y_pred = pl.Series([-1.0, 0.0, 2.0])
    y_true = pl.Series([-0.5, 1.0, 3.0])
    shift = abs(-1.0) + 1

    log_df = calculate_log_predictions_and_actuals("original_target", y_pred, y_true)

    assert log_df.height == 3
    assert log_df.null_count().sum_horizontal().item() == 0
    for name in log_df.columns:
        assert all(math.isfinite(value) for value in log_df[name])
    for base in LOG_PLOT_BASES:
        suffix = log_base_suffix(base)
        assert log_df[f"original_target_pred_{suffix}"].min() == pytest.approx(
            -_reference_log(shift, base)
        )


def test_old_var_type_vocabulary_is_rejected() -> None:
    """The pre-rename names still raise."""

    series = pl.Series([1.0, 2.0])
    for old_var_type in ("transform", "change"):
        with pytest.raises(AssertionError):
            calculate_log_predictions_and_actuals(old_var_type, series, series)


def test_integer_log_inputs_are_rejected() -> None:
    """Integer series trip the float-minimum asserts."""

    integer_series = pl.Series([1, 2])
    with pytest.raises(AssertionError):
        calculate_log_predictions_and_actuals(
            "original_target", integer_series, integer_series
        )


# --- bin statistics and prediction joins -----------------------------------


def test_uniform_bin_statistics_returns_one_row_per_label() -> None:
    """Both series are averaged into the requested ordered bins."""

    y_pred = pl.Series("predictions", [float(value) for value in range(8)])
    y_true = pl.Series("outcome_1", [float(value * 2) for value in range(8)])

    binned = calculate_predictions_and_actuals_statistic_by_uniform_bins(
        y_pred, y_true, ["0", "50"]
    )

    assert binned.columns == ["bins", "outcome_1", "predictions"]
    assert binned["bins"].to_list() == ["0", "50"]
    assert binned.height == 2


def test_join_applies_the_inverse_only_to_positive_predictions() -> None:
    """The reporting inverse is masked; non-positive rows report zero."""

    split_df = pl.DataFrame(
        {
            "case_id": ["a", "b", "c"],
            "case_date": [1, 2, 3],
            "outcome_1": [1.0, 2.0, 3.0],
        }
    )
    predictions_df = pl.DataFrame(
        {
            "case_id": ["a", "b", "c"],
            "case_date": [1, 2, 3],
            "predictions": [-1.0, 0.0, 2.0],
        }
    )

    name, joined = join_dataframe_with_predictions(
        split_df,
        "stem",
        predictions_df,
        ["case_id", "case_date"],
        inverse=lambda value: value * 10.0,
    )

    assert name == "stem"
    assert joined.height == split_df.height
    assert joined[ORIGINAL_TARGET_PRED_COLNAME].to_list() == [0.0, 0.0, 20.0]


# --- feature-importance helpers --------------------------------------------


def _producer_shaped_importances() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": ["feature_1", "feature_3", "feature_2"],
            "feature_estimator_id": ["f0", "f1", "f2"],
            "importance": [0.1, 0.9, 0.5],
        }
    )


def test_top_feature_importances_selects_top_n_ascending() -> None:
    """The top N reported rows come back ascending (highest last/top)."""

    out = top_feature_importances(_producer_shaped_importances(), 2)

    assert out["feature"].to_list() == ["feature_2", "feature_3"]
    assert out["importance"].to_list() == [0.5, 0.9]


def test_top_feature_importances_keeps_ties_stable() -> None:
    """Stable sorts keep tied rows in artifact order, including at the cut."""

    df = pl.DataFrame(
        {
            "feature": ["tie_a", "other", "tie_b", "tie_c"],
            "feature_estimator_id": ["f0", "f1", "f2", "f3"],
            "importance": [0.5, 0.1, 0.5, 0.5],
        }
    )

    out = top_feature_importances(df, 2)

    assert out["feature"].to_list() == ["tie_a", "tie_b"]


def test_top_feature_importances_rejects_top_n_below_one() -> None:
    """A non-positive top_n is a configuration contract failure."""

    with pytest.raises(ValueError, match="top_n"):
        top_feature_importances(_producer_shaped_importances(), 0)


def test_save_feature_importance_plot_writes_a_non_empty_png(
    tmp_path: Path,
) -> None:
    """The writer emits the flat per-model PNG and leaves no figure open."""

    output_filepath = tmp_path / "model_00a_feature_importances.png"

    save_feature_importance_plot(
        _producer_shaped_importances(), "model_00a", 2, tmp_path
    )

    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


# --- regression and binary writers -----------------------------------------


def _prediction_frame() -> pl.DataFrame:
    """A synthetic regression frame with distinct target pairs."""

    row_n = 100
    rows = list(range(row_n))
    return pl.DataFrame(
        {
            "original_target_true": pl.Series(
                [float(row + 1) for row in rows], dtype=pl.Float64
            ),
            "original_target_pred": pl.Series(
                [
                    0.0 if row == 0 else float(1 + row) + ((row * 7) % 11) * 0.5
                    for row in rows
                ],
                dtype=pl.Float64,
            ),
            "model_target_true": pl.Series(
                [float(row) * 0.01 + 0.1 for row in rows], dtype=pl.Float64
            ),
            "model_target_pred": pl.Series(
                [float(row) * 0.009 + 0.15 + ((row * 5) % 7) * 0.02 for row in rows],
                dtype=pl.Float64,
            ),
        }
    )


def test_regression_writer_emits_all_artifacts_and_returns_metrics(
    tmp_path: Path,
) -> None:
    """The writer emits metrics, quantile, log, and error artifacts."""

    metrics = save_regression_model_prediction_results(
        _prediction_frame(),
        "original_target_pred",
        "original_target_true",
        "model_target_pred",
        "model_target_true",
        binary_thresholds=[],
        inverse=lambda value: value,
        output_stem="stem",
        output_path=tmp_path,
    )

    assert isinstance(metrics, RegressionMetrics)

    log_filenames = [
        f"stem_pred_original_target_{log_base_suffix(base)}_{kind}.png"
        for base in LOG_PLOT_BASES
        for kind in ("scatter", "kde", "resid")
    ]
    preexisting_filenames = [
        "stem_regression_metrics.json",
        "stem_original_target_quantile_bin_plot.png",
        "stem_model_target_quantile_bin_plot.png",
        *[
            f"stem_pred_original_target_{kind}.png"
            for kind in ("scatter", "kde", "resid")
        ],
        *[f"stem_pred_model_target_{kind}.png" for kind in ("scatter", "kde", "resid")],
        "stem_error_abs_vs_true.png",
    ]

    for filename in log_filenames + preexisting_filenames:
        output_filepath = tmp_path / filename
        assert output_filepath.is_file(), f"missing artifact {filename}"
        assert output_filepath.stat().st_size > 0, f"empty artifact {filename}"


def test_regression_binary_metrics_use_the_strict_threshold_rule(
    tmp_path: Path,
) -> None:
    """Actual and predicted labels both come from ``value > threshold``."""

    frame = _prediction_frame()
    threshold = 0.5

    save_regression_model_prediction_results(
        frame,
        "original_target_pred",
        "original_target_true",
        "model_target_pred",
        "model_target_true",
        binary_thresholds=[threshold],
        inverse=lambda value: value,
        output_stem="stem",
        output_path=tmp_path,
    )

    metrics_path = tmp_path / "stem_transform_binary_0,5_metrics.json"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))

    y_true = threshold_labels(frame["model_target_true"].to_numpy(), threshold)
    y_pred = threshold_labels(frame["model_target_pred"].to_numpy(), threshold)
    true_positive = int(np.sum(y_true & y_pred))
    true_negative = int(np.sum(~y_true & ~y_pred))
    false_positive = int(np.sum(~y_true & y_pred))
    false_negative = int(np.sum(y_true & ~y_pred))

    assert data["true_positive"] == true_positive
    assert data["true_negative"] == true_negative
    assert data["false_positive"] == false_positive
    assert data["false_negative"] == false_negative
    assert data["classification_threshold"] == threshold
    assert data["threshold_back_transformed"] == threshold


def test_classification_writer_emits_binary_metrics_and_plots(
    tmp_path: Path,
) -> None:
    """The classification writer emits its confusion, ROC, PRC, and metrics."""

    row_n = 40
    frame = pl.DataFrame(
        {
            "outcome_1": pl.Series([index < row_n // 2 for index in range(row_n)]),
            "predictions": pl.Series(
                [index / row_n for index in range(row_n)],
                dtype=pl.Float64,
            ),
        }
    )

    save_classification_model_prediction_results(
        frame, "outcome_1", "predictions", "stem", tmp_path
    )

    for suffix in ("confusion.png", "roc.png", "prc.png", "metrics.json"):
        output_filepath = tmp_path / f"stem_binary_1_{suffix}"
        assert output_filepath.is_file()
        assert output_filepath.stat().st_size > 0
