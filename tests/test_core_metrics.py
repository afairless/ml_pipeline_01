"""Unit tests for the core regression metrics.

Pins the invariant that ``RegressionMetrics`` scores the true-first argument
order of the asymmetric sklearn metrics, drops NaN rows per scale, and
serializes only its scalar fields. A failure here means a reported score
could be plausible but wrong, or a missing value could contaminate the
other target scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sklearn.metrics as skm

from src.core.metrics import RegressionMetrics


def _build_metrics() -> RegressionMetrics:
    return RegressionMetrics(
        original_target_pred=np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
        original_target_true=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        model_target_pred=np.array([1.5, 3.9, 6.4, 7.9, 10.2]),
        model_target_true=np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
    )


def test_pinned_arrays_discriminate_the_argument_order() -> None:
    """The hand-built pairs must yield different r2/d2 under an arg swap.

    If the arrays happened to be symmetric, the test could not distinguish a
    correct true-first call from a swapped one.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    assert skm.r2_score(y_true, y_pred) != skm.r2_score(y_pred, y_true)
    assert skm.d2_absolute_error_score(y_true, y_pred) != (
        skm.d2_absolute_error_score(y_pred, y_true)
    )


def test_r2_and_d2_use_true_first_argument_order() -> None:
    """RegressionMetrics r2/d2 fields equal the true-first sklearn calls."""
    metrics = _build_metrics()

    assert metrics.original_target_r2 == skm.r2_score(
        metrics.original_target_true, metrics.original_target_pred
    )
    assert metrics.model_target_r2 == skm.r2_score(
        metrics.model_target_true, metrics.model_target_pred
    )
    assert metrics.original_target_d2_absolute_error == (
        skm.d2_absolute_error_score(
            metrics.original_target_true, metrics.original_target_pred
        )
    )
    assert metrics.model_target_d2_absolute_error == (
        skm.d2_absolute_error_score(
            metrics.model_target_true, metrics.model_target_pred
        )
    )


def test_symmetric_metrics_match_sklearn_reference() -> None:
    """RMSE/MAE/median-AE stay consistent with the reference scorer calls."""
    metrics = _build_metrics()

    assert metrics.original_target_rmse == skm.root_mean_squared_error(
        metrics.original_target_true, metrics.original_target_pred
    )
    assert metrics.model_target_rmse == skm.root_mean_squared_error(
        metrics.model_target_true, metrics.model_target_pred
    )
    assert metrics.original_target_mean_ae == skm.mean_absolute_error(
        metrics.original_target_true, metrics.original_target_pred
    )
    assert metrics.model_target_mean_ae == skm.mean_absolute_error(
        metrics.model_target_true, metrics.model_target_pred
    )
    assert metrics.original_target_median_ae == skm.median_absolute_error(
        metrics.original_target_true, metrics.original_target_pred
    )
    assert metrics.model_target_median_ae == skm.median_absolute_error(
        metrics.model_target_true, metrics.model_target_pred
    )


def test_nan_rows_are_dropped_independently_per_scale() -> None:
    """A NaN in one scale's pair does not affect the other scale's scores."""
    original_pred = np.array([1.0, np.nan, 3.0, 4.0])
    original_true = np.array([1.0, 2.0, np.nan, 4.0])
    model_pred = np.array([2.0, 3.0, 4.0, np.nan])
    model_true = np.array([1.0, 3.0, 4.0, 5.0])

    metrics = RegressionMetrics(
        original_target_pred=original_pred,
        original_target_true=original_true,
        model_target_pred=model_pred,
        model_target_true=model_true,
    )

    # Only rows 0 and 3 survive on the original scale, rows 0-2 on the model
    # scale, so the two scales must be scored against different row counts.
    assert metrics.original_target_rmse == skm.root_mean_squared_error(
        np.array([1.0, 4.0]), np.array([1.0, 4.0])
    )
    assert metrics.model_target_rmse == skm.root_mean_squared_error(
        np.array([1.0, 3.0, 4.0]), np.array([2.0, 3.0, 4.0])
    )


def test_to_dict_returns_every_field() -> None:
    """``to_dict`` includes both the input arrays and the computed metrics."""
    metrics = _build_metrics()
    data = metrics.to_dict()

    assert set(data) == {
        "original_target_pred",
        "original_target_true",
        "model_target_pred",
        "model_target_true",
        "original_target_rmse",
        "model_target_rmse",
        "original_target_r2",
        "model_target_r2",
        "original_target_mean_ae",
        "model_target_mean_ae",
        "original_target_median_ae",
        "model_target_median_ae",
        "original_target_d2_absolute_error",
        "model_target_d2_absolute_error",
    }
    original_pred = data["original_target_pred"]
    assert isinstance(original_pred, np.ndarray)
    assert np.array_equal(original_pred, metrics.original_target_pred)


def test_write_json_serializes_only_scalar_metrics(tmp_path: Path) -> None:
    """The JSON file holds the float metrics as native values, not arrays."""
    metrics = _build_metrics()
    output_path = tmp_path / "metrics.json"

    metrics.write_json(output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(data) == {
        "original_target_rmse",
        "model_target_rmse",
        "original_target_r2",
        "model_target_r2",
        "original_target_mean_ae",
        "model_target_mean_ae",
        "original_target_median_ae",
        "model_target_median_ae",
        "original_target_d2_absolute_error",
        "model_target_d2_absolute_error",
    }
    assert data["original_target_r2"] == metrics.original_target_r2
    assert all(isinstance(value, float) for value in data.values())
