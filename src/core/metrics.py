"""Core regression metrics shared across pipeline stages.

``RegressionMetrics`` computes error statistics on two aligned prediction/
truth pairs: the original target scale and the model target scale. Each pair
drops rows where either side is NaN before scoring, so a missing prediction
does not poison the other scale. The module is project-agnostic and depends
only on ``json``, ``numpy``, ``scikit-learn``, and ``pathlib``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import sklearn.metrics as skm


def _drop_nan_pairs(
    pred: np.ndarray, true: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the prediction/truth entries where neither side is NaN."""
    mask = ~np.isnan(pred) & ~np.isnan(true)
    return pred[mask], true[mask]


@dataclass
class RegressionMetrics:
    """Regression error statistics for original- and model-scale targets.

    The four input arrays are aligned row by row. ``__post_init__`` computes
    every metric field, dropping NaN rows independently per scale. The
    asymmetric sklearn scorers are called with ``y_true=``/``y_pred=`` by
    name so true values always come first.

    Attributes:
        original_target_pred: Predictions on the original target scale.
        original_target_true: Truth values on the original target scale.
        model_target_pred: Predictions on the model target scale.
        model_target_true: Truth values on the model target scale.
        original_target_rmse: Root mean squared error on the original scale.
        model_target_rmse: Root mean squared error on the model scale.
        original_target_r2: R2 score on the original scale.
        model_target_r2: R2 score on the model scale.
        original_target_mean_ae: Mean absolute error on the original scale.
        model_target_mean_ae: Mean absolute error on the model scale.
        original_target_median_ae: Median absolute error on the original scale.
        model_target_median_ae: Median absolute error on the model scale.
        original_target_d2_absolute_error: D2 absolute error score on the
            original scale.
        model_target_d2_absolute_error: D2 absolute error score on the model
            scale.
    """

    original_target_pred: np.ndarray
    original_target_true: np.ndarray
    model_target_pred: np.ndarray
    model_target_true: np.ndarray

    original_target_rmse: float = np.nan
    model_target_rmse: float = np.nan
    original_target_r2: float = np.nan
    model_target_r2: float = np.nan

    original_target_mean_ae: float = np.nan
    model_target_mean_ae: float = np.nan
    original_target_median_ae: float = np.nan
    model_target_median_ae: float = np.nan
    original_target_d2_absolute_error: float = np.nan
    model_target_d2_absolute_error: float = np.nan

    def __post_init__(self) -> None:
        original_pred, original_true = _drop_nan_pairs(
            self.original_target_pred, self.original_target_true
        )
        model_pred, model_true = _drop_nan_pairs(
            self.model_target_pred, self.model_target_true
        )

        self.original_target_rmse = skm.root_mean_squared_error(
            original_true, original_pred
        )
        self.model_target_rmse = skm.root_mean_squared_error(model_true, model_pred)

        self.original_target_r2 = skm.r2_score(
            y_true=original_true, y_pred=original_pred
        )
        self.model_target_r2 = skm.r2_score(y_true=model_true, y_pred=model_pred)

        self.original_target_mean_ae = skm.mean_absolute_error(
            y_true=original_true, y_pred=original_pred
        )
        self.model_target_mean_ae = skm.mean_absolute_error(
            y_true=model_true, y_pred=model_pred
        )

        self.original_target_median_ae = skm.median_absolute_error(
            y_true=original_true, y_pred=original_pred
        )
        self.model_target_median_ae = skm.median_absolute_error(
            y_true=model_true, y_pred=model_pred
        )

        self.original_target_d2_absolute_error = skm.d2_absolute_error_score(
            y_true=original_true, y_pred=original_pred
        )
        self.model_target_d2_absolute_error = skm.d2_absolute_error_score(
            y_true=model_true, y_pred=model_pred
        )

    def to_dict(self) -> dict[str, object]:
        """Return every field, including the input arrays, keyed by name."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def write_json(self, output_path: Path) -> None:
        """Write the scalar metric fields to ``output_path`` as JSON.

        The input arrays are intentionally omitted: only the computed float
        metrics are serialized, with numpy scalars converted to native Python
        values so the file is plain JSON. Parent directories are not created;
        the caller ensures the output directory first.
        """

        def convert_to_native_types(obj: object) -> object:
            if isinstance(obj, np.generic):
                return obj.item()
            return obj

        data = {
            key: convert_to_native_types(value)
            for key, value in self.to_dict().items()
            if isinstance(value, float)
        }
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)
