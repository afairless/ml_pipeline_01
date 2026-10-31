"""s06 evaluation helpers: reporting, metrics writers, and display tables.

This module holds the reusable half of the evaluation stage. It joins a
partition's predictions to its split table, derives the reporting column names
from the outcome registry, computes and writes the regression and binary
metric artifacts, ranks feature importances, and assembles the cross-model
display table. The stage driver in ``s06_evaluate_model_main`` owns directory
resolution and the evaluation loop.

The module is project-facing but subject-matter agnostic: the true column is
always the outcome's registered name, the prediction column is the shared
``predictions`` vocabulary, and scale semantics come from the outcome's
registered transforms. Binary metrics are derived from continuous values with
``threshold_labels`` (strict ``value > threshold``) rather than from a
classification objective; every current strategy is a regressor, so this is
the only source of binary labels in v1.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import sklearn.metrics as skm
from scipy.stats import pearsonr, spearmanr

from src.core.binary_metrics import (
    BinaryClassificationMetrics,
    get_binary_classification_metrics,
    plot_confusion_matrix_2,
    plot_precision_recall_curve,
    plot_roc_curve,
    threshold_labels,
)
from src.core.manifest import ModelManifest, load_model_runs
from src.core.metrics import RegressionMetrics
from src.core.plots import (
    plot_feature_importances_bar,
    plot_prediction_joint_plots,
    plot_scatter_by_bin,
    plot_two_series_with_differences,
)
from src.project.columns import FEATURE_COLNAME, FEATURE_IMPORTANCE_COLNAME
from src.project.config import LOG_PLOT_BASES
from src.project.outcomes import Outcome

PREDICTION_COLNAME = "predictions"
"""The shared prediction column every s05 prediction table carries."""

ORIGINAL_TARGET_PRED_COLNAME = "predictions_original_target"
"""Column appended by the join when a non-identity target is back-transformed."""

__all__ = [
    "ORIGINAL_TARGET_PRED_COLNAME",
    "PREDICTION_COLNAME",
    "OutcomeColumnNames",
    "build_regression_display_table",
    "calculate_log_predictions_and_actuals",
    "calculate_predictions_and_actuals_statistic_by_uniform_bins",
    "get_model_tasks",
    "join_dataframe_with_predictions",
    "join_dataframes_with_predictions",
    "log_base_suffix",
    "log_plot_column_names",
    "outcome_column_names",
    "record_regression_metrics",
    "save_classification_model_prediction_results",
    "save_feature_importance_plot",
    "save_regression_display_table",
    "save_regression_model_prediction_results",
    "top_feature_importances",
]


@dataclass(frozen=True, slots=True)
class OutcomeColumnNames:
    """The four evaluation column names for one outcome.

    Attributes:
        original_target_true_colname: True-value column on the reporting scale.
        original_target_pred_colname: Prediction column on the reporting scale.
        model_target_true_colname: True-value column on the model scale.
        model_target_pred_colname: Prediction column on the model scale.
    """

    original_target_true_colname: str
    original_target_pred_colname: str
    model_target_true_colname: str
    model_target_pred_colname: str


def join_dataframe_with_predictions(
    df: pl.DataFrame,
    df_name: str,
    predictions_df: pl.DataFrame,
    key_colnames: list[str],
    inverse: Callable[[float], float],
) -> tuple[str, pl.DataFrame]:
    """Join a split table to its predictions and report on the original scale.

    The join is an inner join on the canonical key columns and must preserve
    the split table's row count exactly. The reporting ``inverse`` is applied
    only where the model prediction is positive; every other row reports zero,
    so the mask lives in the stage while the registered inverse stays
    unmasked.

    Args:
        df: The split table with the ground-truth outcome.
        df_name: Name carried beside the joined frame for the caller.
        predictions_df: The run's prediction table.
        key_colnames: Key columns to join on, in canonical order.
        inverse: Registered reporting back-transform.

    Returns:
        ``(df_name, joined_frame)`` with a new
        ``predictions_original_target`` column.

    Raises:
        AssertionError: The keys are missing, row counts differ, or the joined
            predictions contain nulls.
    """

    for colname in key_colnames:
        assert colname in df.columns, f'Column "{colname}" not in "df"'
        assert colname in predictions_df.columns, (
            f'Column "{colname}" not in "predictions_df"'
        )

    assert predictions_df.height == df.height, (
        "Data table and predictions dataframe must have same number of rows"
    )

    join_df = df.join(predictions_df, on=key_colnames, how="inner")

    assert join_df.height == df.height, "Joined dataframe must have same number of rows"
    assert PREDICTION_COLNAME in join_df.columns, (
        f'Column "{PREDICTION_COLNAME}" must be in joined dataframe'
    )
    assert join_df[PREDICTION_COLNAME].is_null().sum() == 0, (
        f'Column "{PREDICTION_COLNAME}" must not have any null values'
    )

    join_df_2 = join_df.with_columns(
        pl.when(pl.col(PREDICTION_COLNAME) > 0)
        .then(pl.col(PREDICTION_COLNAME).map_elements(inverse, return_dtype=pl.Float64))
        .otherwise(0.0)
        .alias(ORIGINAL_TARGET_PRED_COLNAME)
    )

    return df_name, join_df_2


def join_dataframes_with_predictions(
    dfs: list[pl.DataFrame],
    predictions: list[tuple[int, str, pl.DataFrame]],
    key_colnames: list[str],
    inverse: Callable[[float], float],
) -> list[tuple[str, pl.DataFrame]]:
    """Join each split table to its matching prediction table.

    Args:
        dfs: Split tables in the same order as ``predictions``.
        predictions: ``(index, stem, table)`` tuples from the prediction
            manifest grouping.
        key_colnames: Key columns to join on.
        inverse: Registered reporting back-transform.

    Returns:
        One ``(stem, joined_frame)`` pair per prediction, in input order.

    Raises:
        AssertionError: ``predictions`` is empty.
    """

    assert len(predictions) > 0, (
        'The "predictions" list must contain at least one dataframe'
    )

    joined: list[tuple[str, pl.DataFrame]] = []
    for i, entry in enumerate(predictions):
        joined.append(
            join_dataframe_with_predictions(
                dfs[i], entry[1], entry[2], key_colnames, inverse
            )
        )

    return joined


def get_model_tasks(
    model_stage_dir: Path,
    outcome: str,
) -> tuple[dict[str, str], dict[str, str], list[ModelManifest]]:
    """Read each model run's task and outcome column from its manifest.

    Discovery is structural and manifest-driven: ``load_model_runs`` treats
    one immediate child directory holding a ``ModelManifest`` as one run and
    validates kind, outcome, and schema version. The manifest's own ``task``
    is the regression/binary branch decision; the recorded objective is never
    decoded.

    Args:
        model_stage_dir: The outcome-scoped s04 stage directory.
        outcome: The outcome whose runs are read.

    Returns:
        ``(model_name -> task, model_name -> outcome_colname, manifests)``
        from the same single manifest read.

    Raises:
        FileNotFoundError: The stage directory does not exist.
        ValueError: A run manifest is missing, of the wrong kind, declares a
            different outcome, or has an unsupported schema version.
    """

    manifests = load_model_runs(model_stage_dir, outcome)
    task_dict = {manifest.model_name: manifest.task for manifest in manifests}
    outcome_dict = {
        manifest.model_name: manifest.outcome_colname for manifest in manifests
    }

    return task_dict, outcome_dict, manifests


def top_feature_importances(df: pl.DataFrame, top_n: int) -> pl.DataFrame:
    """Select and order the top-N reported feature importances for plotting.

    Rows are ranked by importance descending (stable), truncated to ``top_n``,
    then sorted ascending (stable) because ``barh`` draws the first row at the
    bottom: the most important feature ends up at the top of the chart.

    Args:
        df: The reported feature-importance table.
        top_n: Number of rows to keep.

    Returns:
        The selected rows in ascending importance order.

    Raises:
        ValueError: ``top_n`` is less than 1.
    """

    if top_n < 1:
        raise ValueError(
            f"feature-importance plot top_n must be at least 1, got {top_n}"
        )
    return (
        df.sort(FEATURE_IMPORTANCE_COLNAME, descending=True, maintain_order=True)
        .head(top_n)
        .sort(FEATURE_IMPORTANCE_COLNAME, maintain_order=True)
    )


def save_feature_importance_plot(
    importances_df: pl.DataFrame,
    model_name: str,
    top_n: int,
    output_path: Path,
) -> None:
    """Write one model's top-N feature-importance bar PNG."""

    plot_df = top_feature_importances(importances_df, top_n)
    output_filepath = output_path / f"{model_name}_feature_importances.png"
    plot_feature_importances_bar(
        plot_df, FEATURE_COLNAME, FEATURE_IMPORTANCE_COLNAME, output_filepath
    )


def outcome_column_names(outcome: Outcome) -> OutcomeColumnNames:
    """Derive the s06 evaluation column names from the outcome registry.

    An ``Outcome.name`` is the outcome column name in the outcome table, so
    the true column is the outcome's name on both scales. The model prediction
    column is always ``predictions``. When the model target is untransformed
    (``outcome_transform == 'identity'``) the model and reporting scales
    coincide, so the raw prediction column reports in place; otherwise the
    join's back-transformed column is used.

    Args:
        outcome: The registry outcome being evaluated.

    Returns:
        The frozen evaluation column names for the outcome.
    """

    model_target_true_colname = outcome.name
    original_target_true_colname = outcome.name
    model_target_pred_colname = PREDICTION_COLNAME

    if outcome.outcome_transform == "identity":
        original_target_pred_colname = PREDICTION_COLNAME
    else:
        original_target_pred_colname = ORIGINAL_TARGET_PRED_COLNAME

    return OutcomeColumnNames(
        original_target_true_colname=original_target_true_colname,
        original_target_pred_colname=original_target_pred_colname,
        model_target_true_colname=model_target_true_colname,
        model_target_pred_colname=model_target_pred_colname,
    )


def log_base_suffix(base: float) -> str:
    """Return the column suffix used for a logarithmic plot base.

    Natural log is named ``ln``; all other bases use ``log<base>`` with a
    trailing ``.0`` removed (``10.0`` -> ``log10``, ``2.0`` -> ``log2``).
    """

    if math.isclose(base, math.e):
        return "ln"
    return f"log{base:g}"


def log_plot_column_names(var_type: str) -> list[str]:
    """Return the ordered log-scale column names for ``var_type``.

    The order follows ``LOG_PLOT_BASES`` and alternates the prediction and
    truth column for each base.
    """

    names: list[str] = []
    for base in LOG_PLOT_BASES:
        suffix = log_base_suffix(base)
        names.append(f"{var_type}_pred_{suffix}")
        names.append(f"{var_type}_true_{suffix}")
    return names


def _log_values(values: pl.Series, base: float) -> pl.Series:
    """Log-transform ``values`` with ``base``."""

    return values.log(base)


def calculate_log_predictions_and_actuals(
    var_type: str,
    y_pred: pl.Series,
    y_true: pl.Series,
) -> pl.DataFrame:
    """Log-transform predictions and actuals on every configured base.

    The common minimum of the two series decides the branch. When it is
    positive both series are logged directly with offset zero. Otherwise both
    are shifted upward by the same ``abs(var_min) + 1``; the shift is removed
    on the log scale as ``log(y + shift) - log(shift)``. One shared shift
    keeps the prediction/actual relationship intact, and zero-valued
    predictions stay finite instead of becoming ``-inf``.

    Args:
        var_type: Either ``model_target`` or ``original_target``.
        y_pred: Predictions on that scale.
        y_true: Actual values on that scale.

    Returns:
        A frame with the ordered ``{var_type}_pred_*`` / ``{var_type}_true_*``
        columns.

    Raises:
        AssertionError: ``var_type`` is unknown, the minima are not floats, or
            a shifted value is not positive (for example integer input).
    """

    assert var_type in ["model_target", "original_target"], (
        '"var_type" must equal either "model_target" or "original_target"'
    )

    y_pred_min = y_pred.min()
    y_true_min = y_true.min()
    assert isinstance(y_pred_min, float), "y_pred_min must be a float"
    assert isinstance(y_true_min, float), "y_true_min must be a float"
    var_min = min(y_pred_min, y_true_min)

    shift = abs(var_min) + 1 if var_min <= 0 else 0.0
    y_pred_shift = y_pred + shift
    y_true_shift = y_true + shift

    assert (y_pred_shift > 0).all(), "y_pred_shift must > 0"
    assert (y_true_shift > 0).all(), "y_true_shift must > 0"

    log_columns: dict[str, pl.Series] = {}
    for base in LOG_PLOT_BASES:
        suffix = log_base_suffix(base)
        offset = _log_values(pl.Series([shift]), base)[0] if shift else 0.0
        log_columns[f"{var_type}_pred_{suffix}"] = (
            _log_values(y_pred_shift, base) - offset
        )
        log_columns[f"{var_type}_true_{suffix}"] = (
            _log_values(y_true_shift, base) - offset
        )

    return pl.DataFrame(log_columns)


def calculate_predictions_and_actuals_statistic_by_uniform_bins(
    y_pred: pl.Series,
    y_true: pl.Series,
    bin_labels: list[str] = ["0", "25", "50", "75"],
) -> pl.DataFrame:
    """Average predictions and actuals in the same quantile bins.

    Args:
        y_pred: Predictions.
        y_true: Actual values.
        bin_labels: Ordered quantile labels; the count sets the bin count.

    Returns:
        A frame with ``bins`` plus the mean prediction and truth columns.
    """

    quantile_n = len(bin_labels)

    y_pred_bins = y_pred.qcut(quantile_n, labels=bin_labels, allow_duplicates=True)
    y_true_bins = y_true.qcut(quantile_n, labels=bin_labels, allow_duplicates=True)

    y_pred_bin_df = (
        pl.DataFrame({y_pred.name: y_pred, "bins": y_pred_bins.cast(pl.String)})
        .group_by("bins")
        .mean()
    )
    y_true_bin_df = (
        pl.DataFrame({y_true.name: y_true, "bins": y_true_bins.cast(pl.String)})
        .group_by("bins")
        .mean()
    )

    return (
        pl.DataFrame(bin_labels, schema=["bin_labels"])
        .join(
            y_true_bin_df,
            left_on="bin_labels",
            right_on="bins",
            how="full",
            maintain_order="left",
        )
        .join(y_pred_bin_df, on="bins", how="full", maintain_order="left")
        .with_columns(pl.col("bins").fill_null(pl.col("bins_right")))
        .drop(["bin_labels", "bins_right"])
        .fill_null(strategy="forward")
    )


def save_classification_model_prediction_results(
    prediction_df: pl.DataFrame,
    y_true_colname: str,
    y_pred_proba_colname: str,
    output_stem: str,
    output_path: Path,
) -> None:
    """Write one model's binary-classification metrics and plots.

    Args:
        prediction_df: Frame carrying the true and predicted-probability
            columns.
        y_true_colname: True-label column.
        y_pred_proba_colname: Predicted-probability column.
        output_stem: Filename stem shared by every artifact.
        output_path: Existing directory that receives the artifacts.
    """

    y_true = prediction_df[y_true_colname]
    y_pred_proba = prediction_df[y_pred_proba_colname]

    binary_metrics = get_binary_classification_metrics(
        y_true.to_numpy(), y_pred_proba.to_numpy(), skm.balanced_accuracy_score
    )
    bm = binary_metrics

    y_pearson_corr, _ = pearsonr(y_pred_proba, y_true)
    y_spearman_corr, _ = spearmanr(y_pred_proba, y_true)

    threshold_str = y_true_colname.split("_")[-1]

    output_filepath = output_path / (
        output_stem + f"_binary_{threshold_str}_confusion.png"
    )
    plot_confusion_matrix_2(bm.y_true, bm.y_pred, output_filepath)

    output_filepath = output_path / (output_stem + f"_binary_{threshold_str}_roc.png")
    plot_roc_curve(
        bm.y_true,
        bm.y_pred_proba,
        bm.tpr,
        bm.fpr,
        bm.prevalence,
        output_filepath,
    )

    output_filepath = output_path / (output_stem + f"_binary_{threshold_str}_prc.png")
    plot_precision_recall_curve(
        bm.y_true,
        bm.y_pred_proba,
        bm.recall,
        bm.precision,
        bm.informedness,
        bm.markedness,
        bm.prevalence,
        output_filepath,
    )

    output_filepath = output_path / (
        output_stem + f"_binary_{threshold_str}_metrics.json"
    )
    bm.convert_arrays_to_strings()
    bm.write_json(output_filepath)

    quantiles = [str(j) for j in range(0, 101, 5)]
    y_bin_df = calculate_predictions_and_actuals_statistic_by_uniform_bins(
        y_pred_proba, y_true, quantiles
    )

    y_bin_pearson_corr, _ = pearsonr(y_bin_df[:, 1], y_bin_df[:, 2])
    y_bin_spearman_corr, _ = spearmanr(y_bin_df[:, 1], y_bin_df[:, 2])

    title = (
        f"Pearson r = {y_bin_pearson_corr:.2f}, Spearman r = {y_bin_spearman_corr:.2f}"
    )
    output_filepath = output_path / (output_stem + "_y_quantile_bin_plot.png")
    plot_two_series_with_differences(
        y_bin_df,
        "bins",
        y_true_colname,
        y_pred_proba_colname,
        title,
        output_filepath,
    )

    title = f"Pearson r = {y_pearson_corr:.2f}, Spearman r = {y_spearman_corr:.2f}"
    plot_prediction_joint_plots(
        prediction_df,
        y_true_colname,
        y_pred_proba_colname,
        "proba_vs_binary",
        title,
        output_stem,
        output_path,
    )


def save_regression_model_prediction_results(
    prediction_df: pl.DataFrame,
    original_target_pred_colname: str,
    original_target_true_colname: str,
    model_target_pred_colname: str,
    model_target_true_colname: str,
    binary_thresholds: list[float],
    inverse: Callable[[float], float],
    output_stem: str,
    output_path: Path,
) -> RegressionMetrics:
    """Write one regression model's metrics and plots on both scales.

    Binary classification metrics are derived by thresholding both the actual
    and predicted model-scale values with the shared ``threshold_labels``
    rule, so every threshold uses the same strict ``value > threshold``
    convention. The reporting inverse is applied unmasked to record each
    threshold's original-scale value.

    Args:
        prediction_df: Frame carrying both scales' prediction and true
            columns.
        original_target_pred_colname: Reporting-scale prediction column.
        original_target_true_colname: Reporting-scale true column.
        model_target_pred_colname: Model-scale prediction column.
        model_target_true_colname: Model-scale true column.
        binary_thresholds: Model-scale thresholds to report.
        inverse: Registered reporting back-transform.
        output_stem: Filename stem shared by every artifact.
        output_path: Existing directory that receives the artifacts.

    Returns:
        The computed ``RegressionMetrics`` so the caller can build the
        cross-model display table without re-reading JSON.
    """

    original_target_pred = prediction_df[original_target_pred_colname]
    original_target_true = prediction_df[original_target_true_colname]
    model_target_true = prediction_df[model_target_true_colname]
    model_target_pred = prediction_df[model_target_pred_colname]

    original_target_ae = (original_target_pred - original_target_true).abs()

    regression_metrics = RegressionMetrics(
        original_target_pred.to_numpy(),
        original_target_true.to_numpy(),
        model_target_pred.to_numpy(),
        model_target_true.to_numpy(),
    )
    output_filepath = output_path / (output_stem + "_regression_metrics.json")
    regression_metrics.write_json(output_filepath)

    original_target_pearson_corr, _ = pearsonr(
        original_target_pred, original_target_true
    )
    original_target_spearman_corr, _ = spearmanr(
        original_target_pred, original_target_true
    )
    model_target_pearson_corr, _ = pearsonr(model_target_pred, model_target_true)
    model_target_spearman_corr, _ = spearmanr(model_target_pred, model_target_true)

    quantiles = [str(j) for j in range(0, 101, 2)]
    original_target_bin_df = (
        calculate_predictions_and_actuals_statistic_by_uniform_bins(
            original_target_pred, original_target_true, quantiles
        )
    )
    model_target_bin_df = calculate_predictions_and_actuals_statistic_by_uniform_bins(
        model_target_pred, model_target_true, quantiles
    )

    original_target_bin_pearson_corr, _ = pearsonr(
        original_target_bin_df[:, 1], original_target_bin_df[:, 2]
    )
    original_target_bin_spearman_corr, _ = spearmanr(
        original_target_bin_df[:, 1], original_target_bin_df[:, 2]
    )
    model_target_bin_pearson_corr, _ = pearsonr(
        model_target_bin_df[:, 1], model_target_bin_df[:, 2]
    )
    model_target_bin_spearman_corr, _ = spearmanr(
        model_target_bin_df[:, 1], model_target_bin_df[:, 2]
    )

    title = (
        f"Pearson r = {original_target_bin_pearson_corr:.2f}, "
        f"Spearman r = {original_target_bin_spearman_corr:.2f}"
    )
    output_filepath = output_path / (
        output_stem + "_original_target_quantile_bin_plot.png"
    )
    plot_two_series_with_differences(
        original_target_bin_df,
        "bins",
        original_target_true_colname,
        original_target_pred_colname,
        title,
        output_filepath,
    )

    title = (
        f"Pearson r = {model_target_bin_pearson_corr:.2f}, "
        f"Spearman r = {model_target_bin_spearman_corr:.2f}"
    )
    output_filepath = output_path / (
        output_stem + "_model_target_quantile_bin_plot.png"
    )
    plot_two_series_with_differences(
        model_target_bin_df,
        "bins",
        model_target_true_colname,
        model_target_pred_colname,
        title,
        output_filepath,
    )

    original_target_log_df = calculate_log_predictions_and_actuals(
        "original_target", original_target_pred, original_target_true
    )
    prediction_df_2 = pl.concat(
        [prediction_df, original_target_log_df], how="horizontal"
    )

    title = (
        f"Pearson r = {original_target_pearson_corr:.2f}, "
        f"Spearman r = {original_target_spearman_corr:.2f}"
    )
    plot_prediction_joint_plots(
        prediction_df_2,
        original_target_true_colname,
        original_target_pred_colname,
        "original_target",
        title,
        output_stem,
        output_path,
    )

    for base in LOG_PLOT_BASES:
        suffix = log_base_suffix(base)
        plot_prediction_joint_plots(
            prediction_df_2,
            f"original_target_true_{suffix}",
            f"original_target_pred_{suffix}",
            f"original_target_{suffix}",
            title,
            output_stem,
            output_path,
        )

    title = (
        f"Pearson r = {model_target_pearson_corr:.2f}, "
        f"Spearman r = {model_target_spearman_corr:.2f}"
    )
    plot_prediction_joint_plots(
        prediction_df_2,
        model_target_true_colname,
        model_target_pred_colname,
        "model_target",
        title,
        output_stem,
        output_path,
    )

    for binary_threshold in binary_thresholds:
        threshold_back_transformed = inverse(binary_threshold)
        model_target_true_binary = threshold_labels(
            model_target_true.to_numpy(), binary_threshold
        )
        model_target_pred_binary = threshold_labels(
            model_target_pred.to_numpy(), binary_threshold
        )

        change_binary_metrics = BinaryClassificationMetrics(
            y_true=model_target_true_binary,
            y_pred=model_target_pred_binary,
            classification_threshold=binary_threshold,
            threshold_back_transformed=threshold_back_transformed,
        )

        threshold_str = str(binary_threshold).replace(".", ",")

        output_filepath = output_path / (
            output_stem + f"_transform_binary_{threshold_str}_confusion.png"
        )
        plot_confusion_matrix_2(
            change_binary_metrics.y_true,
            change_binary_metrics.y_pred,
            output_filepath,
        )

        output_filepath = output_path / (
            output_stem + f"_transform_binary_{threshold_str}_roc.png"
        )
        plot_roc_curve(
            change_binary_metrics.y_true,
            change_binary_metrics.y_pred,
            change_binary_metrics.tpr,
            change_binary_metrics.fpr,
            change_binary_metrics.prevalence,
            output_filepath,
        )

        output_filepath = output_path / (
            output_stem + f"_transform_binary_{threshold_str}_prc.png"
        )
        plot_precision_recall_curve(
            change_binary_metrics.y_true,
            change_binary_metrics.y_pred,
            change_binary_metrics.recall,
            change_binary_metrics.precision,
            change_binary_metrics.informedness,
            change_binary_metrics.markedness,
            change_binary_metrics.prevalence,
            output_filepath,
        )

        output_filepath = output_path / (
            output_stem + f"_transform_binary_{threshold_str}_metrics.json"
        )
        change_binary_metrics.convert_arrays_to_strings()
        change_binary_metrics.write_json(output_filepath)

    bin_n = 50
    suptitle = "Model prediction absolute error vs. true values by equal-sized bin"
    output_filepath = output_path / (output_stem + "_error_abs_vs_true.png")
    plot_scatter_by_bin(
        bin_n,
        original_target_true,
        original_target_ae,
        f'True values of "{original_target_true_colname}", '
        f"{original_target_true_colname}",
        f"Model absolute error, {original_target_pred_colname}",
        suptitle,
        output_stem,
        output_filepath,
    )

    return regression_metrics


def record_regression_metrics(
    task: str,
    model_name: str,
    partition: str,
    metrics: RegressionMetrics,
    metrics_by_model: dict[str, dict[str, RegressionMetrics]],
) -> None:
    """Record one model/partition's metrics for the display table.

    A non-regression task is a no-op, so a binary model is omitted entirely
    rather than given a blank row.

    Raises:
        ValueError: ``(model_name, partition)`` has already been recorded.
    """

    if task != "regression":
        return

    model_metrics = metrics_by_model.setdefault(model_name, {})
    if partition in model_metrics:
        raise ValueError(
            "Duplicate regression metrics for model "
            f"{model_name!r} partition {partition!r}"
        )
    model_metrics[partition] = metrics


def _display_metric_fields(metric_name: str) -> tuple[str, ...]:
    """Expand a configured metric into the fields the table renders.

    The project config names metric *suffixes* (for example ``rmse``), so the
    table renders both the original-scale and model-scale field. A caller may
    also pass an already-prefixed field name, which is used as-is.
    """

    if metric_name.startswith(("original_target_", "model_target_")):
        return (metric_name,)
    return (
        f"original_target_{metric_name}",
        f"model_target_{metric_name}",
    )


def build_regression_display_table(
    metrics_by_model: Mapping[str, Mapping[str, RegressionMetrics]],
    partitions: Sequence[str],
    metric_names: Sequence[str],
    decimals: int,
) -> pl.DataFrame:
    """Build the cross-model regression metrics display frame.

    Rows are sorted by model name ascending. Columns are ``model`` followed
    by every expanded metric field and partition pair, with each metric's
    partitions adjacent in the requested order. Every metric value is rounded
    numerically to ``decimals`` places; the frame's dtypes are explicit so
    the header-only case still carries the full declared schema.

    Raises:
        ValueError: A model supplies a partition other than exactly the
            requested ones; the message names the model and partition.
    """

    fields: list[str] = []
    for metric_name in metric_names:
        fields.extend(_display_metric_fields(metric_name))

    columns = [
        "model",
        *[f"{field}_{partition}" for field in fields for partition in partitions],
    ]
    schema = {
        column: pl.String if column == "model" else pl.Float64 for column in columns
    }

    rows: list[dict[str, str | float]] = []
    for model_name in sorted(metrics_by_model):
        model_metrics = metrics_by_model[model_name]

        for partition in partitions:
            if partition not in model_metrics:
                raise ValueError(
                    f"Model {model_name!r} is missing partition {partition!r}"
                )
        for partition in model_metrics:
            if partition not in partitions:
                raise ValueError(
                    f"Model {model_name!r} has unexpected partition {partition!r}"
                )

        row: dict[str, str | float] = {"model": model_name}
        for field in fields:
            for partition in partitions:
                value = getattr(model_metrics[partition], field)
                row[f"{field}_{partition}"] = round(value, decimals)
        rows.append(row)

    return pl.DataFrame(rows, schema=schema)


def save_regression_display_table(display_df: pl.DataFrame, output_path: Path) -> None:
    """Write the display table as exactly one CSV at ``output_path``."""

    display_df.write_csv(output_path)
