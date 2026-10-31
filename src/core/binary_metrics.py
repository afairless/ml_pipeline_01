"""Core binary-classification metrics and figure helpers.

``BinaryClassificationMetrics`` scores a boolean prediction against a boolean
truth vector, always building the confusion matrix against the explicit label
order ``[False, True]`` so a partition that happens to contain only one
observed class still yields a 2x2 matrix. ``threshold_labels`` is the single
place where a continuous value is turned into a binary label: the positive
class is the value **strictly greater** than the threshold, so a value equal
to the threshold is negative. ``get_binary_classification_metrics`` searches a
fixed grid for the threshold that maximizes a caller-supplied scorer and then
scores the resulting labels.

The plot helpers own the figures they create: each one builds, decorates,
saves, and closes its own figure without touching the caller's pyplot state.
The module is project-agnostic and imports only ``json``, ``numpy``,
``scikit-learn``, and ``matplotlib``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import cast

import numpy as np
import sklearn.metrics as skm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def threshold_labels(values: np.ndarray, threshold: float) -> np.ndarray:
    """Return boolean labels for ``values`` at ``threshold``.

    The positive class is ``value > threshold`` (strict). A value exactly
    equal to the threshold is therefore a negative label, which keeps the
    actual and predicted label rules identical.

    Args:
        values: One-dimensional array of continuous values to threshold.
        threshold: Cut point separating the positive from the negative class.

    Returns:
        A boolean array of the same shape as ``values``.
    """
    return np.asarray(values) > threshold


@dataclass
class BestThresholdMetric:
    """The best threshold found on a grid and its score.

    Attributes:
        threshold: Grid value that produced ``metric``.
        metric: Score returned by the caller-supplied scorer at that value.
    """

    threshold: float
    metric: float


@dataclass
class BinaryClassificationMetrics:
    """Binary confusion-matrix counts and derived classification statistics.

    ``__post_init__`` computes every metric from the aligned ``y_true`` and
    ``y_pred`` label vectors. When ``y_pred_proba`` is non-empty it also
    scores the probability-based metrics (Brier loss and ROC AUC).

    Attributes:
        y_true: Boolean truth labels, one per row.
        y_pred: Boolean predicted labels, one per row.
        y_pred_proba: Optional predicted probabilities for the positive class.
        classification_threshold: Threshold used to derive ``y_pred``, if any.
        threshold_back_transformed: Threshold on the original target scale,
            if any.
        true_positive: Count of correctly predicted positives.
        true_negative: Count of correctly predicted negatives.
        false_positive: Count of negatives predicted positive.
        false_negative: Count of positives predicted negative.
        population: Total number of scored rows.
        prevalence: Share of rows whose truth label is positive.
        accuracy: Share of rows predicted correctly.
        recall: Positive-class recall (sensitivity).
        sensitivity: Alias for ``recall``.
        true_positive_rate: Alias for ``recall``.
        tpr: Alias for ``recall``.
        specificity: Negative-class recall.
        true_negative_rate: Alias for ``specificity``.
        tnr: Alias for ``specificity``.
        precision: Positive predictive value.
        positive_predictive_value: Alias for ``precision``.
        ppv: Alias for ``precision``.
        negative_predictive_value: Share of predicted negatives that are true
            negatives.
        npv: Alias for ``negative_predictive_value``.
        false_positive_rate: Share of true negatives predicted positive.
        fpr: Alias for ``false_positive_rate``.
        false_omission_rate: Share of true negatives predicted negative.
        f1: Harmonic mean of precision and recall.
        roc_auc: Area under the ROC curve from ``y_pred_proba``.
        balanced_accuracy: Mean of sensitivity and specificity.
        informedness: ``recall + specificity - 1``.
        markedness: ``ppv + npv - 1``.
        mcc: Matthews correlation coefficient.
        brier: Brier loss from ``y_pred_proba``.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    y_pred_proba: np.ndarray = field(default_factory=lambda: np.array([]))
    classification_threshold: float = np.nan
    threshold_back_transformed: float = np.nan

    true_positive: float = np.nan
    true_negative: float = np.nan
    false_positive: float = np.nan
    false_negative: float = np.nan

    population: float = np.nan
    prevalence: float = np.nan

    accuracy: float = np.nan
    recall: float = np.nan
    sensitivity: float = np.nan
    true_positive_rate: float = np.nan
    tpr: float = np.nan
    specificity: float = np.nan
    true_negative_rate: float = np.nan
    tnr: float = np.nan
    precision: float = np.nan
    positive_predictive_value: float = np.nan
    ppv: float = np.nan
    negative_predictive_value: float = np.nan
    npv: float = np.nan
    false_positive_rate: float = np.nan
    fpr: float = np.nan
    false_omission_rate: float = np.nan
    f1: float = np.nan
    roc_auc: float = np.nan

    balanced_accuracy: float = np.nan
    informedness: float = np.nan
    markedness: float = np.nan
    mcc: float = np.nan

    brier: float = np.nan

    def __post_init__(self) -> None:
        assert self.y_true.ndim == 1
        assert self.y_pred.ndim == 1
        assert len(self.y_true) == len(self.y_pred)

        confusion = skm.confusion_matrix(self.y_true, self.y_pred, labels=[False, True])
        true_negative, false_positive, false_negative, true_positive = confusion.ravel()
        self.true_positive = true_positive
        self.true_negative = true_negative
        self.false_positive = false_positive
        self.false_negative = false_negative

        self.population = (
            true_positive + true_negative + false_positive + false_negative
        )
        self.prevalence = (true_positive + false_negative) / self.population

        self.accuracy = skm.accuracy_score(self.y_true, self.y_pred)
        self.recall = skm.recall_score(self.y_true, self.y_pred)
        self.sensitivity = self.recall
        self.true_positive_rate = self.recall
        self.tpr = self.true_positive_rate
        self.specificity = true_negative / (true_negative + false_positive)
        self.true_negative_rate = self.specificity
        self.tnr = self.true_negative_rate
        self.precision = skm.precision_score(self.y_true, self.y_pred)
        self.positive_predictive_value = self.precision
        self.ppv = self.positive_predictive_value
        self.negative_predictive_value = true_negative / (
            true_negative + false_negative
        )
        self.npv = self.negative_predictive_value
        self.false_positive_rate = false_positive / (false_positive + true_negative)
        self.fpr = self.false_positive_rate
        self.false_omission_rate = false_negative / (false_negative + true_negative)
        self.f1 = skm.f1_score(self.y_true, self.y_pred)

        self.balanced_accuracy = skm.balanced_accuracy_score(self.y_true, self.y_pred)
        self.informedness = self.recall + self.specificity - 1
        self.markedness = self.ppv + self.npv - 1
        self.mcc = skm.matthews_corrcoef(self.y_true, self.y_pred)

        if self.y_pred_proba.size > 0:
            self.brier = skm.brier_score_loss(self.y_true, self.y_pred_proba)
            self.roc_auc = skm.roc_auc_score(self.y_true, self.y_pred_proba)

    def round_floats(self, digits: int = 2) -> None:
        """Round every float-valued field to ``digits`` decimal places."""
        for data_field in fields(self):
            if isinstance(getattr(self, data_field.name), float):
                setattr(
                    self,
                    data_field.name,
                    round(getattr(self, data_field.name), digits),
                )

    def convert_int_to_float(self) -> None:
        """Convert numpy integer counts to Python floats for serialization.

        Python ``int`` values are left untouched. The check uses the bare
        ``np.integer`` class because the ``np.int64`` stub alias carries type
        arguments, which ``isinstance`` rejects.
        """
        for data_field in fields(self):
            if isinstance(getattr(self, data_field.name), np.integer):
                setattr(self, data_field.name, float(getattr(self, data_field.name)))

    def convert_arrays_to_strings(self) -> None:
        """Replace array-valued fields with their list string form."""
        for data_field in fields(self):
            if isinstance(getattr(self, data_field.name), np.ndarray):
                setattr(
                    self, data_field.name, str(getattr(self, data_field.name).tolist())
                )

    def to_dict(self) -> dict[str, object]:
        """Return every field, including the input arrays, keyed by name."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def write_json(self, output_path: Path) -> None:
        """Write the metrics to ``output_path`` as JSON.

        Callers that want array fields serialized should call
        ``convert_arrays_to_strings`` first; numpy scalars are converted to
        native Python values here. Parent directories are not created.
        """

        def convert_to_native_types(obj: object) -> object:
            if isinstance(obj, np.generic):
                return obj.item()
            return obj

        data = {
            key: convert_to_native_types(value) for key, value in self.to_dict().items()
        }
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)


def get_binary_classification_threshold_and_best_metric(
    true_y_binary_categories: np.ndarray,
    y_prediction_probabilities: np.ndarray,
    scoring_func: Callable[[np.ndarray, np.ndarray], float],
) -> BestThresholdMetric:
    """Find the grid threshold that maximizes ``scoring_func``.

    The grid runs from 0.01 to 0.99 in 99 evenly spaced steps; ties resolve to
    the first (lowest) threshold that attains the maximum.

    Args:
        true_y_binary_categories: Boolean truth labels.
        y_prediction_probabilities: Positive-class probabilities.
        scoring_func: Two-argument scorer called as
            ``scoring_func(y_true, y_pred)``.

    Returns:
        The winning threshold and its score.
    """
    assert y_prediction_probabilities.ndim == 1
    assert true_y_binary_categories.ndim == 1
    assert len(y_prediction_probabilities) == len(true_y_binary_categories)

    thresholds = np.linspace(0.01, 0.99, 99)
    metrics = [
        scoring_func(true_y_binary_categories, y_prediction_probabilities > threshold)
        for threshold in thresholds
    ]
    best_metric = np.max(metrics)
    threshold = thresholds[np.argmax(metrics)]

    return BestThresholdMetric(float(threshold), float(best_metric))


def get_binary_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold_metric_func: Callable[[np.ndarray, np.ndarray], float],
) -> BinaryClassificationMetrics:
    """Threshold probabilities with the best grid cut and score the labels.

    Args:
        y_true: Boolean truth labels.
        y_pred_proba: Positive-class probabilities.
        threshold_metric_func: Scorer used to pick the classification
            threshold.

    Returns:
        Metrics for the labels derived from the winning threshold.
    """
    best_threshold_metric = get_binary_classification_threshold_and_best_metric(
        y_true, y_pred_proba, threshold_metric_func
    )

    y_pred = y_pred_proba > best_threshold_metric.threshold

    return BinaryClassificationMetrics(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        classification_threshold=best_threshold_metric.threshold,
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
    output_filepath: Path,
) -> None:
    """Plot a confusion matrix with positives in the top-left cell.

    Args:
        y_true: Boolean truth labels.
        y_pred_proba: Positive-class probabilities, thresholded here.
        threshold: Strict cut point for the positive class.
        output_filepath: PNG path written by the owned figure.
    """
    y_pred = threshold_labels(y_pred_proba, threshold)
    confusion = skm.confusion_matrix(y_true, y_pred, labels=[False, True])[
        ::-1, ::-1
    ].transpose()

    plot = skm.ConfusionMatrixDisplay(confusion, display_labels=[1, 0])
    fig, ax = plt.subplots(figsize=(4, 3))
    plot.plot(ax=ax)
    ax.set_ylabel("Predicted label", fontsize=14)
    ax.set_xlabel("True label", fontsize=14)
    fig.tight_layout()

    fig.savefig(output_filepath)
    plt.close(fig)


def plot_confusion_matrix_2(
    y_true: np.ndarray, y_pred: np.ndarray, output_filepath: Path
) -> None:
    """Plot a confusion matrix from already-thresholded labels.

    Args:
        y_true: Boolean truth labels.
        y_pred: Boolean predicted labels.
        output_filepath: PNG path written by the owned figure.
    """
    confusion = skm.confusion_matrix(y_true, y_pred, labels=[False, True])[
        ::-1, ::-1
    ].transpose()

    plot = skm.ConfusionMatrixDisplay(confusion, display_labels=[1, 0])
    fig, ax = plt.subplots(figsize=(4, 3))
    plot.plot(ax=ax)
    ax.set_ylabel("Predicted label", fontsize=14)
    ax.set_xlabel("True label", fontsize=14)
    fig.tight_layout()

    fig.savefig(output_filepath)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    tpr: float,
    fpr: float,
    prevalence: float,
    output_filepath: Path,
    plot_balance: bool = True,
) -> None:
    """Plot a receiver operating characteristic curve.

    Args:
        y_true: Boolean truth labels.
        y_pred_proba: Positive-class probabilities.
        tpr: Observed true-positive rate to mark on the curve.
        fpr: Observed false-positive rate to mark on the curve.
        prevalence: Positive-class prevalence, used for the chance line when
            ``plot_balance`` is false.
        output_filepath: PNG path written by the owned figure.
        plot_balance: Draw the diagonal balance line when true, otherwise draw
            the prevalence-based chance line.
    """
    slope = (1 - prevalence) / prevalence
    intercept = tpr - (slope * fpr)
    fprs = np.linspace(0.01, 0.99, 99)
    tprs = slope * fprs + intercept

    display = skm.RocCurveDisplay(tpr=tpr, fpr=fpr).from_predictions(
        y_true, y_pred_proba
    )
    display.ax_.scatter(fpr, tpr, marker="o")

    if plot_balance:
        display.ax_.plot([0, 1], [0, 1], color="black", linestyle="dotted")
    else:
        display.ax_.plot(fprs, tprs, color="black", linestyle="dotted")

    figure = cast(Figure, display.figure_)
    figure.savefig(output_filepath)
    plt.close(figure)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    recall: float,
    precision: float,
    informedness: float,
    markedness: float,
    prevalence: float,
    output_filepath: Path,
) -> None:
    """Plot a precision-recall curve with the observed point marked.

    Args:
        y_true: Boolean truth labels.
        y_pred_proba: Positive-class probabilities.
        recall: Observed recall to mark on the curve.
        precision: Observed precision to mark on the curve.
        informedness: Observed informedness, accepted for call-site symmetry
            and currently not drawn.
        markedness: Observed markedness, accepted for call-site symmetry and
            currently not drawn.
        prevalence: Positive-class prevalence, drawn as the horizontal chance
            line.
        output_filepath: PNG path written by the owned figure.
    """
    display = skm.PrecisionRecallDisplay(
        precision=precision, recall=recall
    ).from_predictions(y_true=y_true, y_score=y_pred_proba)
    display.ax_.scatter(recall, precision, marker="o")
    display.ax_.axhline(prevalence, color="black", linestyle="dotted")

    figure = cast(Figure, display.figure_)
    figure.savefig(output_filepath)
    plt.close(figure)
