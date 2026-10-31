"""Evaluation stage entry point, driven by the pipeline context.

Accepts the frozen ``PipelineContext`` plus the executor-supplied output root
and evaluates one modeled outcome across every model run its upstream stages
produced. Nothing is discovered by filename: the s06 directory is resolved from
the layout (``layout.stage_dir('s06_evaluate', outcome)``), the trained runs
come from ``load_model_runs`` over ``s04_train/{outcome}``, the prediction runs
come from ``load_prediction_runs`` over ``s05_predict/{outcome}``, and every
split and prediction table is resolved through its run's declared artifact
inventory.

For each model run the driver writes one feature-importance PNG into the
stage directory, then evaluates exactly the partitions named by
``config.evaluated_partitions``. Each evaluated partition joins the s03 split
table to its s05 prediction table on the canonical key columns and reports in
its own ``{model}_{partition}/`` directory, so the on-disk layout carries both
identities without a consumer parsing a mixed filename. The regression and
binary writers are selected by the model manifest's ``task``; the manifest is
the single source of truth for that branch, never the objective string or a
filename suffix. Regression metrics are also accumulated into the per-outcome
cross-model display CSV that the stage declares.

The three responsibilities stay separate even though they meet here:
*ingestion* reads validated split, prediction, and importance tables through
the project artifact readers; *transformation* is the join, the reporting
inverse, and the metric computation; *output* is the metric/plot/display
writers. Re-running the stage replaces only the artifacts it declares and never
touches another stage's output.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.core.manifest import (
    ModelManifest,
    PredictionManifest,
    declared_artifact_path,
    load_prediction_runs,
)
from src.core.metrics import RegressionMetrics
from src.core.paths import PipelineContext
from src.project.artifacts import (
    SplitTables,
    load_feature_importances,
    load_split_tables,
)
from src.project.columns import KEY_COLUMNS
from src.project.config import (
    DISPLAY_METRIC_DECIMALS,
    DISPLAY_REGRESSION_METRICS,
)
from src.project.outcomes import OUTCOMES, resolve_transform
from src.stages.s06_evaluate_model import (
    PREDICTION_COLNAME,
    build_regression_display_table,
    get_model_tasks,
    join_dataframes_with_predictions,
    outcome_column_names,
    record_regression_metrics,
    save_classification_model_prediction_results,
    save_feature_importance_plot,
    save_regression_display_table,
    save_regression_model_prediction_results,
)

STAGE_NAME = "s06_evaluate"
"""Stage-table key whose directory and declared artifacts this stage writes."""

S03_STAGE_NAME = "s03_divide"
"""Stage-table key whose split partitions this stage evaluates."""

S04_STAGE_NAME = "s04_train"
"""Stage-table key whose model runs and manifests this stage consumes."""

S05_STAGE_NAME = "s05_predict"
"""Stage-table key whose prediction runs and manifests this stage consumes."""

__all__ = [
    "S03_STAGE_NAME",
    "S04_STAGE_NAME",
    "S05_STAGE_NAME",
    "STAGE_NAME",
    "main",
    "s06_evaluate",
]


def _prediction_runs_by_model(
    stage_dir: Path,
    outcome_name: str,
) -> dict[str, PredictionManifest]:
    """Index one outcome's discovered prediction runs by model name.

    ``load_prediction_runs`` returns runs sorted by directory name but does not
    guarantee unique model names, so a duplicate would make a lookup
    ambiguous — the driver raises instead of silently keeping one of them.

    Raises:
        ValueError: Two discovered prediction manifests declare the same
            ``model_name``.
    """
    prediction_runs: dict[str, PredictionManifest] = {}
    for manifest in load_prediction_runs(stage_dir, outcome_name):
        if manifest.model_name in prediction_runs:
            raise ValueError(
                f"duplicate prediction run for model {manifest.model_name!r} "
                f"under {stage_dir}"
            )
        prediction_runs[manifest.model_name] = manifest
    return prediction_runs


def _prediction_table(
    manifest: PredictionManifest,
    run_dir: Path,
    partition: str,
) -> pl.DataFrame:
    """Read one partition's prediction table through its manifest inventory.

    Raises:
        FileNotFoundError: The declared prediction parquet is absent from
            ``run_dir``.
    """
    path = declared_artifact_path(
        manifest,
        run_dir,
        f"{partition}_predictions",
        ".parquet",
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"prediction run {manifest.model_name!r} is missing its declared "
            f"{partition!r} artifact {path}"
        )
    return pl.read_parquet(path)


def _evaluated_partitions(
    outcome_name: str,
    model_manifest: ModelManifest,
    prediction_manifest: PredictionManifest,
    evaluated_partitions: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the configured partitions both upstream runs declare.

    The model manifest declares the run's split shape and the prediction
    manifest declares which partitions were predicted; a configured partition
    missing from either run means the evaluation cannot be completed, so it
    raises instead of skipping silently.

    Raises:
        ValueError: A configured partition is missing from one of the two
            manifests.
    """
    missing_from_model = [
        partition
        for partition in evaluated_partitions
        if partition not in model_manifest.partitions
    ]
    if missing_from_model:
        raise ValueError(
            f"model run {model_manifest.model_name!r} for {outcome_name!r} "
            f"does not declare evaluated partition(s) {missing_from_model}; "
            f"declared partitions are {list(model_manifest.partitions)}"
        )

    missing_from_predictions = [
        partition
        for partition in evaluated_partitions
        if partition not in prediction_manifest.partitions
    ]
    if missing_from_predictions:
        raise ValueError(
            f"prediction run {prediction_manifest.model_name!r} for "
            f"{outcome_name!r} does not declare evaluated partition(s) "
            f"{missing_from_predictions}; declared partitions are "
            f"{list(prediction_manifest.partitions)}"
        )

    return evaluated_partitions


def _partition_frame(split_tables: SplitTables, partition: str) -> pl.DataFrame:
    """Return one loaded s03 split partition by its manifest-style name.

    Raises:
        ValueError: ``partition`` is not one of the split tables' partitions.
    """
    attribute = f"{partition}_df"
    frame = getattr(split_tables, attribute, None)
    if not isinstance(frame, pl.DataFrame):
        raise ValueError(
            f"unknown evaluated partition {partition!r}; no {attribute!r} "
            "split table was loaded"
        )
    return frame


def _evaluate_model(
    manifest: ModelManifest,
    prediction_manifest: PredictionManifest,
    context: PipelineContext,
    output_root: Path,
    output_path: Path,
    split_tables: SplitTables,
    metrics_by_model: dict[str, dict[str, RegressionMetrics]],
) -> None:
    """Evaluate one model run and write its per-partition artifacts.

    A regression task delegates to the regression writer and records the
    returned metrics for the display table; a binary task delegates to the
    classification writer. Any other task is rejected before a partition
    directory is created.

    Raises:
        ValueError: The task is unknown, or a configured evaluated partition
            is missing from either upstream manifest.
        FileNotFoundError: A declared split or prediction artifact is absent.
    """
    model_name = manifest.model_name
    evaluated = _evaluated_partitions(
        manifest.outcome,
        manifest,
        prediction_manifest,
        context.config.evaluated_partitions,
    )

    inverse = resolve_transform(
        OUTCOMES[manifest.outcome].reporting_backtransform
    ).inverse
    prediction_run_dir = (
        output_root
        / context.layout.stage_dir(S05_STAGE_NAME, manifest.outcome)
        / model_name
    )
    prediction_tables = [
        (
            index,
            partition,
            _prediction_table(prediction_manifest, prediction_run_dir, partition),
        )
        for index, partition in enumerate(evaluated)
    ]
    joined_tables = join_dataframes_with_predictions(
        [_partition_frame(split_tables, partition) for partition in evaluated],
        prediction_tables,
        list(KEY_COLUMNS),
        inverse,
    )

    column_names = outcome_column_names(OUTCOMES[manifest.outcome])
    for partition, joined_df in zip(evaluated, joined_tables, strict=True):
        partition_name, prediction_df = joined_df
        partition_dir = output_path / f"{model_name}_{partition}"
        context.ensure(partition_dir)

        if manifest.task == "regression":
            metrics = save_regression_model_prediction_results(
                prediction_df,
                column_names.original_target_pred_colname,
                column_names.original_target_true_colname,
                column_names.model_target_pred_colname,
                column_names.model_target_true_colname,
                list(context.config.binary_thresholds),
                inverse,
                model_name,
                partition_dir,
            )
            record_regression_metrics(
                manifest.task,
                model_name,
                partition_name,
                metrics,
                metrics_by_model,
            )
        elif manifest.task == "binary":
            save_classification_model_prediction_results(
                prediction_df,
                column_names.model_target_true_colname,
                PREDICTION_COLNAME,
                model_name,
                partition_dir,
            )
        else:
            raise ValueError(
                f"unknown task {manifest.task!r} for model run "
                f"{model_name!r}; task must be 'regression' or 'binary'"
            )


def main(outcome_name: str, context: PipelineContext, output_root: Path) -> None:
    """Evaluate one outcome's model runs and write the s06 artifacts.

    The stage directory is resolved from the layout under the supplied output
    root and created through the context's one explicit side effect,
    ``ensure``. The s04 model runs supply identity, task, partitions, and
    feature importances; the s05 prediction runs supply the partition-tagged
    prediction tables; and the s03 split tables supply the ground truth each
    prediction is joined to.

    Args:
        outcome_name: The modeled outcome being evaluated.
        context: The frozen pipeline context (layout + config).
        output_root: The pipeline output root the relative layout paths are
            resolved under.

    Raises:
        FileNotFoundError: An upstream stage directory or a declared
            split/prediction/importance artifact is missing.
        ValueError: A manifest is inconsistent with its run, a configured
            partition is not declared upstream, a model task is unknown, or
            the outcome is not declared by the layout.
        BoundaryError: A consumed table violates its declared schema.
    """
    output_root = Path(output_root)
    output_path = output_root / context.layout.stage_dir(STAGE_NAME, outcome_name)
    context.ensure(output_path)

    model_stage_dir = output_root / context.layout.stage_dir(
        S04_STAGE_NAME, outcome_name
    )
    prediction_stage_dir = output_root / context.layout.stage_dir(
        S05_STAGE_NAME, outcome_name
    )
    _task_by_model, _outcome_by_model, manifests = get_model_tasks(
        model_stage_dir, outcome_name
    )
    prediction_runs = _prediction_runs_by_model(prediction_stage_dir, outcome_name)
    split_tables = load_split_tables(context.layout, outcome_name, output_root)

    metrics_by_model: dict[str, dict[str, RegressionMetrics]] = {}
    for manifest in manifests:
        model_name = manifest.model_name
        if model_name not in prediction_runs:
            raise FileNotFoundError(
                f"model run {model_name!r} for {outcome_name!r} has no "
                f"prediction run under {prediction_stage_dir}"
            )

        model_run_dir = model_stage_dir / model_name
        importances_df = load_feature_importances(manifest, model_run_dir)
        save_feature_importance_plot(
            importances_df,
            model_name,
            context.config.feature_importance_plot_top_n,
            output_path,
        )

        _evaluate_model(
            manifest,
            prediction_runs[model_name],
            context,
            output_root,
            output_path,
            split_tables,
            metrics_by_model,
        )

    display_df = build_regression_display_table(
        metrics_by_model,
        context.config.evaluated_partitions,
        DISPLAY_REGRESSION_METRICS,
        DISPLAY_METRIC_DECIMALS,
    )
    display_path = output_root / context.layout.artifact(
        STAGE_NAME, "display_table", outcome_name
    )
    save_regression_display_table(display_df, display_path)


s06_evaluate = main
"""Stage entry point under the name used by the registry executor."""
