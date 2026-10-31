"""Integration tests for the ``s06_evaluate`` stage driver.

The driver is the consumer end of the registry-driven half of the pipeline: it
reads the s04 model manifests, the s05 prediction manifests, and the s03 split
tables, joins each prediction table to its ground truth on the canonical keys,
and writes the per-model feature-importance PNG, the per-partition metrics and
plots, and the per-outcome cross-model display CSV.

These tests drive the whole path on a temporary project root seeded with the
committed demo table and the real s01/s02/s03/s04/s05 stages, then pin:

* the stage directory and declared display artifact match the layout;
* only the configured evaluated partitions receive a ``{model}_{partition}``
  directory, so the held-out test partition stays unevaluated;
* every evaluated partition writes regression metrics and the binary metrics
  derived from the configured thresholds, with the strict ``value > threshold``
  rule reflected in classification counts;
* the feature-importance PNG is written once per model run;
* the display CSV carries one row per regression model, every configured
  metric field and evaluated partition, rounded to the configured precision;
* re-running the stage replaces the same artifacts deterministically; and
* a missing upstream run fails instead of writing a partial evaluation.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import polars as pl
import pytest

from src.core.paths import PipelineContext
from src.core.splits import PARTITION_ORDER
from src.project.config import (
    CONFIG,
    DISPLAY_METRIC_DECIMALS,
    DISPLAY_REGRESSION_METRICS,
)
from src.project.outcomes import modeled_outcomes
from src.project.pipeline import build_context
from src.stages.s01_outcome import s01_outcome
from src.stages.s02_feature import s02_feature
from src.stages.s03_divide_data import s03_divide
from src.stages.s04_train import main as s04_main
from src.stages.s05_predict_model_main import main as s05_main
from src.stages.s06_evaluate_model_main import STAGE_NAME, main, s06_evaluate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Repository root, used only to locate the committed demo input table."""

SOURCE_FILENAME = "source_table_1.parquet"
"""Committed demo filename, copied into each temporary project root."""

OUTCOME_NAME = modeled_outcomes()[0].name
"""The modeled outcome these integration tests evaluate."""

STRATEGY_ID = "model_00_signal_0"
"""The single strategy the seeded pipeline trains and evaluates."""

DISPLAY_ARTIFACT = "display_table"
"""The logical artifact name the stage table declares for the CSV."""


@dataclass(frozen=True, slots=True)
class Evaluated:
    """One completed evaluation: the temp root, context, and stage directory."""

    project_root: Path
    context: PipelineContext
    output_root: Path
    stage_dir: Path

    @property
    def display_path(self) -> Path:
        """The declared per-outcome display CSV path."""
        return self.output_root / self.context.layout.artifact(
            STAGE_NAME, DISPLAY_ARTIFACT, OUTCOME_NAME
        )

    def partition_dir(self, partition: str, model_name: str = STRATEGY_ID) -> Path:
        """The stage's per-model, per-partition output directory."""
        return self.stage_dir / f"{model_name}_{partition}"


def _copy_demo_source(project_root: Path) -> None:
    """Place the committed demo source table under a temporary root."""
    data_dir = project_root / CONFIG.data_root
    data_dir.mkdir(parents=True)
    shutil.copyfile(
        PROJECT_ROOT / CONFIG.data_root / SOURCE_FILENAME,
        data_dir / SOURCE_FILENAME,
    )


def _seed_through_predict(context: PipelineContext, output_root: Path) -> None:
    """Run s01 through s05 so s06 has every upstream artifact it reads."""
    s01_outcome(context, output_root)
    s02_feature(context, output_root)
    s03_divide(OUTCOME_NAME, context, output_root)
    s04_main(STRATEGY_ID, OUTCOME_NAME, context, output_root)
    s05_main(OUTCOME_NAME, context, output_root)


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory: pytest.TempPathFactory) -> Evaluated:
    """Evaluate one strategy once on a fully seeded temporary project root."""
    project_root = tmp_path_factory.mktemp("s06_evaluated")
    _copy_demo_source(project_root)
    context = build_context(config=CONFIG, project_root=project_root)
    output_root = project_root / "output"
    _seed_through_predict(context, output_root)
    main(OUTCOME_NAME, context, output_root)
    stage_dir = output_root / context.layout.stage_dir(STAGE_NAME, OUTCOME_NAME)
    return Evaluated(project_root, context, output_root, stage_dir)


def _metrics_path(evaluated: Evaluated, partition: str) -> Path:
    """The regression metrics JSON written for one evaluated partition."""
    return evaluated.partition_dir(partition) / f"{STRATEGY_ID}_regression_metrics.json"


def _threshold_suffix(threshold: float) -> str:
    """The artifact stem fragment the writer uses for one threshold."""
    return str(threshold).replace(".", ",")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


# --- layout and top-level contract -----------------------------------------


def test_entry_point_alias_matches_main() -> None:
    """The registry alias is the stage's only entry point."""
    assert s06_evaluate is main


def test_stage_writes_into_the_layout_directory(evaluated: Evaluated) -> None:
    """The stage directory and display artifact come from the layout."""
    expected_dir = evaluated.output_root / evaluated.context.layout.stage_dir(
        STAGE_NAME, OUTCOME_NAME
    )

    assert evaluated.stage_dir == expected_dir
    assert evaluated.stage_dir.is_dir()
    assert evaluated.display_path == (
        evaluated.stage_dir / f"{OUTCOME_NAME}_display.csv"
    )
    assert evaluated.display_path.is_file()


# --- per-partition evaluation ----------------------------------------------


def test_only_configured_partitions_are_evaluated(evaluated: Evaluated) -> None:
    """The held-out ``test`` partition receives no evaluation directory."""
    for partition in PARTITION_ORDER:
        partition_dir = evaluated.partition_dir(partition)
        if partition in CONFIG.evaluated_partitions:
            assert partition_dir.is_dir()
        else:
            assert not partition_dir.exists()


def test_regression_metrics_written_for_every_evaluated_partition(
    evaluated: Evaluated,
) -> None:
    """Each evaluated partition writes its regression metrics JSON."""
    expected_fields = {
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

    for partition in CONFIG.evaluated_partitions:
        metrics_path = _metrics_path(evaluated, partition)
        assert metrics_path.is_file()
        data = _read_json(metrics_path)
        assert set(data) == expected_fields
        assert all(
            isinstance(value, float) and math.isfinite(value) for value in data.values()
        )


def test_binary_metrics_use_the_configured_thresholds(
    evaluated: Evaluated,
) -> None:
    """Every configured threshold writes binary metrics with both classes."""
    for partition in CONFIG.evaluated_partitions:
        split_df = pl.read_parquet(
            evaluated.output_root
            / evaluated.context.layout.artifact(
                "s03_divide", f"{partition}_df", OUTCOME_NAME
            )
        )
        for threshold in CONFIG.binary_thresholds:
            suffix = _threshold_suffix(threshold)
            metrics_path = (
                evaluated.partition_dir(partition)
                / f"{STRATEGY_ID}_transform_binary_{suffix}_metrics.json"
            )
            assert metrics_path.is_file()

            data = _read_json(metrics_path)
            assert data["classification_threshold"] == threshold
            assert data["threshold_back_transformed"] == pytest.approx(threshold)

            true_positive = int(data["true_positive"])
            true_negative = int(data["true_negative"])
            false_positive = int(data["false_positive"])
            false_negative = int(data["false_negative"])

            assert true_positive + true_negative + false_positive + false_negative == (
                split_df.height
            )
            assert true_positive + false_negative > 0
            assert true_negative + false_positive > 0


def test_regression_metrics_match_the_joined_prediction_table(
    evaluated: Evaluated,
) -> None:
    """The metrics are computed on the same rows the split table carries."""
    for partition in CONFIG.evaluated_partitions:
        split_height = pl.read_parquet(
            evaluated.output_root
            / evaluated.context.layout.artifact(
                "s03_divide", f"{partition}_df", OUTCOME_NAME
            )
        ).height
        threshold = CONFIG.binary_thresholds[0]
        data = _read_json(
            evaluated.partition_dir(partition)
            / (
                f"{STRATEGY_ID}_transform_binary_"
                f"{_threshold_suffix(threshold)}_metrics.json"
            )
        )
        positive = int(data["true_positive"]) + int(data["false_negative"])
        negative = int(data["true_negative"]) + int(data["false_positive"])
        assert positive + negative == split_height
        assert positive > 0
        assert negative > 0


def test_feature_importance_plot_is_written_once_per_model(
    evaluated: Evaluated,
) -> None:
    """The per-model top-N importance PNG sits at the stage-directory root."""
    plot_path = evaluated.stage_dir / f"{STRATEGY_ID}_feature_importances.png"

    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0


# --- cross-model display table ---------------------------------------------


def test_display_table_has_one_row_per_regression_model(
    evaluated: Evaluated,
) -> None:
    """The display CSV follows the configured metric and partition order."""
    display_df = pl.read_csv(evaluated.display_path)

    metric_fields = [
        field
        for metric_name in DISPLAY_REGRESSION_METRICS
        for field in (
            f"original_target_{metric_name}",
            f"model_target_{metric_name}",
        )
    ]
    expected_columns = [
        "model",
        *[
            f"{field}_{partition}"
            for field in metric_fields
            for partition in CONFIG.evaluated_partitions
        ],
    ]

    assert display_df.columns == expected_columns
    assert display_df["model"].to_list() == [STRATEGY_ID]
    assert display_df.height == 1
    assert display_df.null_count().sum_horizontal().item() == 0


def test_display_values_are_the_rounded_regression_metrics(
    evaluated: Evaluated,
) -> None:
    """A displayed value is the metrics JSON value rounded to the configured
    precision, so the table never silently disagrees with the artifacts."""
    display_df = pl.read_csv(evaluated.display_path)
    partition = CONFIG.evaluated_partitions[0]
    field = f"original_target_{DISPLAY_REGRESSION_METRICS[0]}"
    metrics = _read_json(_metrics_path(evaluated, partition))

    expected = round(float(metrics[field]), DISPLAY_METRIC_DECIMALS)
    assert display_df[f"{field}_{partition}"].to_list() == pytest.approx([expected])


# --- idempotency and failure mode ------------------------------------------


def test_rerun_replaces_the_same_artifacts(evaluated: Evaluated) -> None:
    """Re-running the stage reproduces the same metrics and display CSV."""
    metrics_path = _metrics_path(evaluated, CONFIG.evaluated_partitions[0])
    before_metrics = metrics_path.read_bytes()
    before_display = evaluated.display_path.read_bytes()

    main(OUTCOME_NAME, evaluated.context, evaluated.output_root)

    assert metrics_path.read_bytes() == before_metrics
    assert evaluated.display_path.read_bytes() == before_display


def test_missing_upstream_runs_raise_instead_of_writing_a_partial_result(
    tmp_path: Path,
) -> None:
    """A root with no s04 runs fails before any evaluation directory is
    populated."""
    context = build_context(config=CONFIG, project_root=tmp_path)
    output_root = tmp_path / "output"

    with pytest.raises(FileNotFoundError):
        main(OUTCOME_NAME, context, output_root)

    stage_dir = output_root / context.layout.stage_dir(STAGE_NAME, OUTCOME_NAME)
    assert list(stage_dir.iterdir()) == []
