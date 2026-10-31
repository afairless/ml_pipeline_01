"""Unit tests for the core binary metrics and threshold helper.

Pins three invariants. First, ``threshold_labels`` uses the strict
``value > threshold`` rule, so a value exactly at the threshold is a negative
label and actual and predicted labels share one derivation. Second, the
confusion matrix is always built against explicit ``[False, True]`` labels, so
a partition with a single observed class still yields a 2x2 matrix and the
count unpacking cannot fail. Third, the serialization helpers convert numpy
integers and arrays without touching Python values.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import sklearn.metrics as skm

from src.core.binary_metrics import (
    BestThresholdMetric,
    BinaryClassificationMetrics,
    get_binary_classification_metrics,
    get_binary_classification_threshold_and_best_metric,
    threshold_labels,
)


def _metrics_all_same(
    y_true_value: bool, y_pred_value: bool
) -> BinaryClassificationMetrics:
    """Construct metrics where ``y_true`` is all one class."""
    y_true = np.full(10, y_true_value, dtype=bool)
    y_pred = np.full(10, y_pred_value, dtype=bool)
    with warnings.catch_warnings():
        # sklearn emits UndefinedMetricWarning for undefined recall/precision
        # on single-class partitions; the object must still construct.
        warnings.simplefilter("ignore")
        return BinaryClassificationMetrics(y_true=y_true, y_pred=y_pred)


def _metrics_with_probabilities() -> BinaryClassificationMetrics:
    """Construct metrics from clearly separated labels and probabilities."""
    return BinaryClassificationMetrics(
        y_true=np.array([False, False, True, True]),
        y_pred=np.array([False, False, True, True]),
        y_pred_proba=np.array([0.1, 0.2, 0.8, 0.9]),
        classification_threshold=0.5,
    )


def test_threshold_labels_strictly_greater_than_threshold() -> None:
    """Only values above the threshold are positive labels."""
    values = np.array([-1.0, 0.0, 0.499, 0.5, 0.501, 1.0])

    labels = threshold_labels(values, 0.5)

    assert labels.dtype == bool
    assert labels.tolist() == [False, False, False, False, True, True]


def test_threshold_labels_equal_to_threshold_is_negative() -> None:
    """The boundary value itself is always the negative class."""
    threshold = 1.25
    values = np.array([threshold])

    assert threshold_labels(values, threshold).tolist() == [False]


def test_threshold_labels_actual_and_predicted_share_the_rule() -> None:
    """Actual and predicted labels are derived by the same strict rule."""
    threshold = 0.0
    actual_values = np.array([-0.5, 0.0, 0.5])
    predicted_values = np.array([-0.5, 0.0, 0.5])

    actual = threshold_labels(actual_values, threshold)
    predicted = threshold_labels(predicted_values, threshold)

    assert actual.tolist() == [False, False, True]
    assert np.array_equal(actual, predicted)


def test_threshold_labels_treats_nan_as_negative() -> None:
    """A NaN comparison is false, so a missing value is not positive."""
    labels = threshold_labels(np.array([np.nan, 1.0]), 0.5)

    assert labels.tolist() == [False, True]


def test_threshold_labels_empty_input_returns_empty_labels() -> None:
    """An empty input produces an empty boolean array, not an error."""
    labels = threshold_labels(np.array([]), 0.5)

    assert labels.shape == (0,)
    assert labels.dtype == bool


def test_all_false_observation_still_yields_2x2_matrix() -> None:
    """An all-False ``y_true`` gives tn=10 and a valid 2x2 unpacking."""
    metrics = _metrics_all_same(y_true_value=False, y_pred_value=False)
    assert metrics.true_negative == 10
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
    assert metrics.true_positive == 0
    assert metrics.population == 10
    assert metrics.prevalence == 0


def test_all_true_observation_still_yields_2x2_matrix() -> None:
    """An all-True ``y_true`` gives tp=10 and a valid 2x2 unpacking."""
    metrics = _metrics_all_same(y_true_value=True, y_pred_value=True)
    assert metrics.true_positive == 10
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
    assert metrics.true_negative == 0
    assert metrics.population == 10
    assert metrics.prevalence == 1


def test_single_observed_class_with_wrong_predictions() -> None:
    """A one-class partition with fully wrong predictions still unpacks."""
    metrics = _metrics_all_same(y_true_value=False, y_pred_value=True)
    assert metrics.true_negative == 0
    assert metrics.false_positive == 10
    assert metrics.false_negative == 0
    assert metrics.true_positive == 0
    assert metrics.population == 10


def test_derived_statistics_match_reference_scorers() -> None:
    """Every alias and derived statistic matches its sklearn reference."""
    metrics = _metrics_with_probabilities()

    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.recall == metrics.sensitivity == metrics.tpr == 1.0
    assert metrics.specificity == metrics.tnr == 1.0
    assert metrics.precision == metrics.ppv == 1.0
    assert metrics.npv == 1.0
    assert metrics.fpr == 0.0
    assert metrics.false_omission_rate == 0.0
    assert metrics.informedness == 1.0
    assert metrics.markedness == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.mcc == pytest.approx(1.0)
    assert metrics.brier == pytest.approx(
        skm.brier_score_loss(metrics.y_true, metrics.y_pred_proba)
    )
    assert metrics.roc_auc == pytest.approx(
        skm.roc_auc_score(metrics.y_true, metrics.y_pred_proba)
    )


def test_probability_metrics_stay_nan_without_probabilities() -> None:
    """Without probabilities the Brier and ROC AUC fields remain NaN."""
    metrics = _metrics_all_same(y_true_value=False, y_pred_value=False)

    assert np.isnan(metrics.brier)
    assert np.isnan(metrics.roc_auc)
    assert metrics.y_pred_proba.size == 0


def test_round_floats_rounds_float_fields_only() -> None:
    """Rounding touches float fields and leaves the label arrays intact."""
    metrics = _metrics_with_probabilities()
    metrics.brier = 0.123456

    metrics.round_floats(digits=3)

    assert metrics.brier == 0.123
    assert np.array_equal(metrics.y_true, np.array([False, False, True, True]))


def test_convert_int_to_float_converts_numpy_ints_only() -> None:
    """Numpy counts become floats while a Python int survives untouched."""
    metrics = _metrics_all_same(y_true_value=False, y_pred_value=False)
    metrics.accuracy = 1  # a Python int, not a numpy scalar

    assert isinstance(metrics.true_negative, np.integer)
    assert isinstance(metrics.population, np.integer)

    metrics.convert_int_to_float()

    assert isinstance(metrics.true_negative, float)
    assert isinstance(metrics.population, float)
    assert metrics.true_negative == 10.0
    assert type(metrics.accuracy) is int


def test_convert_int_to_float_covers_non_int64_widths() -> None:
    """The check is not limited to int64: an int32 count converts too."""
    metrics = _metrics_all_same(y_true_value=False, y_pred_value=False)
    # __post_init__ stores numpy counts in float-annotated fields; on a 32-bit
    # platform confusion_matrix returns int32, so the check must not be
    # int64-specific.
    metrics.false_positive = cast(float, np.int32(3))

    metrics.convert_int_to_float()

    assert isinstance(metrics.false_positive, float)
    assert metrics.false_positive == 3.0


def test_convert_arrays_to_strings_replaces_array_fields() -> None:
    """Array fields become their list string form for JSON serialization."""
    metrics = _metrics_with_probabilities()

    metrics.convert_arrays_to_strings()

    assert isinstance(metrics.y_true, str)
    assert metrics.y_true == str([False, False, True, True])


def test_write_json_round_trips_converted_values(tmp_path: Path) -> None:
    """A converted metrics object writes plain JSON with native values."""
    metrics = _metrics_with_probabilities()
    metrics.convert_int_to_float()
    metrics.convert_arrays_to_strings()
    output_path = tmp_path / "metrics.json"

    metrics.write_json(output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["classification_threshold"] == 0.5
    assert data["true_positive"] == 2.0
    assert data["y_true"] == str([False, False, True, True])


def test_best_threshold_helper_returns_first_maximizing_grid_value() -> None:
    """The threshold search reports the score at the selected grid point."""
    y_true = np.array([False, False, True, True])
    y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9])

    result = get_binary_classification_threshold_and_best_metric(
        y_true, y_pred_proba, skm.f1_score
    )

    assert isinstance(result, BestThresholdMetric)
    assert result.metric == pytest.approx(1.0)
    assert result.threshold == pytest.approx(0.2, abs=0.011)


def test_get_binary_classification_metrics_uses_best_threshold() -> None:
    """The scored labels come from the winning threshold, not the raw output."""
    y_true = np.array([False, False, True, True])
    y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = get_binary_classification_metrics(y_true, y_pred_proba, skm.f1_score)

    assert metrics.classification_threshold == pytest.approx(0.2, abs=0.011)
    assert np.array_equal(metrics.y_pred, y_true)
    assert metrics.f1 == pytest.approx(1.0)
