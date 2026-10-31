"""Integration smoke test for the complete demo pipeline run.

The test seeds a temporary project root with the committed demo source table,
runs the same explicit stage sequence as ``src/main.py`` followed by the
registry-driven ``run_pipeline`` executor, and asserts the plan's expected
output tree: every explicit stage writes its declared artifacts, every
registered outcome is divided, every model strategy writes a training and a
prediction manifest, and every evaluated partition receives regression and
thresholded binary metrics. Analysis-only outcomes must not receive model
stages.

A separate test re-runs the explicit stages and the deterministic divide stage
twice and compares the non-figure artifacts, so a stage that accumulates state
or emits unstable table bytes fails here.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import pytest

from src.core.manifest import MANIFEST_FILENAME
from src.core.splits import PARTITION_ORDER
from src.project.config import CONFIG
from src.project.models import MODEL_STRATEGIES
from src.project.outcomes import modeled_outcomes, outcome_names
from src.project.pipeline import PipelineContext, build_context, run_pipeline
from src.stages.s01_outcome import s01_outcome
from src.stages.s01b_outcome_analysis import s01b_outcome_analysis
from src.stages.s02_feature import s02_feature
from src.stages.s03_divide_data import s03_divide

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Repository root, used only to locate the committed demo input table."""

SOURCE_FILENAME = "source_table_1.parquet"
"""Committed demo filename, copied into each temporary project root."""

EXPLICIT_STAGE_NAMES = ("s01_outcome", "s01b_outcome_analysis", "s02_feature")
"""Stages the entry point calls before the registry-driven executor."""

MODELED_OUTCOME_NAMES = tuple(outcome.name for outcome in modeled_outcomes())
"""Outcomes that receive the training, prediction, and evaluation stages."""

ANALYSIS_OUTCOME_NAMES = tuple(
    name for name in outcome_names() if name not in MODELED_OUTCOME_NAMES
)
"""Outcomes that exist only for descriptive analysis."""


@dataclass(frozen=True, slots=True)
class CompletedRun:
    """The temporary project root and output tree produced by one full run."""

    project_root: Path
    context: PipelineContext
    output_root: Path


def _seed_project_root(project_root: Path) -> None:
    """Place the committed demo source table under a temporary root."""
    data_dir = project_root / CONFIG.data_root
    data_dir.mkdir(parents=True)
    shutil.copyfile(
        PROJECT_ROOT / CONFIG.data_root / SOURCE_FILENAME,
        data_dir / SOURCE_FILENAME,
    )


def _run_pipeline(project_root: Path) -> CompletedRun:
    """Run the entry point's explicit stages followed by the registry."""
    _seed_project_root(project_root)
    context = build_context(config=CONFIG, project_root=project_root)
    output_root = project_root / "output"
    s01_outcome(context, output_root)
    s01b_outcome_analysis(context, output_root)
    s02_feature(context, output_root)
    run_pipeline(context, output_root)
    return CompletedRun(project_root, context, output_root)


@pytest.fixture(scope="module")
def completed(tmp_path_factory: pytest.TempPathFactory) -> CompletedRun:
    """Run the full demo pipeline once for the tree assertions."""
    root = tmp_path_factory.mktemp("smoke_pipeline")
    return _run_pipeline(root)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _stage_dir(completed: CompletedRun, stage: str, outcome: str | None = None) -> Path:
    """Resolve one stage directory through the pipeline layout."""
    return completed.output_root / completed.context.layout.stage_dir(stage, outcome)


def _artifact(
    completed: CompletedRun,
    stage: str,
    logical: str,
    outcome: str | None = None,
) -> Path:
    """Resolve one declared stage artifact through the pipeline layout."""
    return completed.output_root / completed.context.layout.artifact(
        stage, logical, outcome
    )


# --- output tree contract ---------------------------------------------------


def test_full_run_creates_exactly_the_declared_stage_directories(
    completed: CompletedRun,
) -> None:
    """The output root holds one directory per declared stage, no extras."""
    assert {path.name for path in completed.output_root.iterdir() if path.is_dir()} == {
        "s01_outcome",
        "s01b_outcome_analysis",
        "s02_feature",
        "s03_divide",
        "s04_train",
        "s05_predict",
        "s06_evaluate",
    }


@pytest.mark.parametrize("stage_name", EXPLICIT_STAGE_NAMES)
def test_explicit_stages_write_every_declared_artifact(
    completed: CompletedRun,
    stage_name: str,
) -> None:
    """Each explicitly called stage leaves its complete declared artifact set."""
    descriptor = completed.context.layout.stages[stage_name]
    for logical_name in descriptor.artifacts:
        assert _artifact(completed, stage_name, logical_name).is_file()


@pytest.mark.parametrize("outcome_name", outcome_names())
def test_divide_stage_writes_every_split_for_every_outcome(
    completed: CompletedRun,
    outcome_name: str,
) -> None:
    """Every registered outcome receives all three split tables and statistics."""
    for logical_name in (
        "feature_head_df",
        "outcome_head_df",
        "train_df",
        "valid_df",
        "test_df",
        "statistics",
    ):
        assert _artifact(completed, "s03_divide", logical_name, outcome_name).is_file()


@pytest.mark.parametrize("outcome_name", ANALYSIS_OUTCOME_NAMES)
def test_analysis_only_outcomes_receive_no_model_stages(
    completed: CompletedRun,
    outcome_name: str,
) -> None:
    """The model stages run only for outcomes whose registry entry is modeled."""
    for stage_name in ("s04_train", "s05_predict", "s06_evaluate"):
        assert not _stage_dir(completed, stage_name, outcome_name).exists()


# --- model fan-out ----------------------------------------------------------


@pytest.mark.parametrize("outcome_name", MODELED_OUTCOME_NAMES)
@pytest.mark.parametrize("model_name", tuple(MODEL_STRATEGIES))
def test_every_model_strategy_writes_train_and_prediction_manifests(
    completed: CompletedRun,
    outcome_name: str,
    model_name: str,
) -> None:
    """Every modeled outcome trains and predicts every registered strategy."""
    train_dir = _stage_dir(completed, "s04_train", outcome_name) / model_name
    prediction_dir = _stage_dir(completed, "s05_predict", outcome_name) / model_name

    train_manifest = _read_json(train_dir / MANIFEST_FILENAME)
    prediction_manifest = _read_json(prediction_dir / MANIFEST_FILENAME)

    assert train_manifest["manifest_kind"] == "model"
    assert train_manifest["model_name"] == model_name
    assert train_manifest["outcome"] == outcome_name
    assert train_manifest["task"] == "regression"
    assert train_manifest["partitions"] == list(PARTITION_ORDER)

    assert prediction_manifest["manifest_kind"] == "predictions"
    assert prediction_manifest["model_name"] == model_name
    assert prediction_manifest["outcome"] == outcome_name
    assert prediction_manifest["partitions"] == list(PARTITION_ORDER)


@pytest.mark.parametrize("outcome_name", MODELED_OUTCOME_NAMES)
@pytest.mark.parametrize("model_name", tuple(MODEL_STRATEGIES))
def test_every_model_run_is_evaluated_for_the_configured_partitions(
    completed: CompletedRun,
    outcome_name: str,
    model_name: str,
) -> None:
    """Every evaluated partition gets regression and binary metric files."""
    stage_dir = _stage_dir(completed, "s06_evaluate", outcome_name)
    assert (stage_dir / f"{model_name}_feature_importances.png").is_file()

    for partition in CONFIG.evaluated_partitions:
        partition_dir = stage_dir / f"{model_name}_{partition}"
        assert (partition_dir / f"{model_name}_regression_metrics.json").is_file()
        for threshold in CONFIG.binary_thresholds:
            threshold_stem = str(threshold).replace(".", ",")
            metrics_path = (
                partition_dir
                / f"{model_name}_transform_binary_{threshold_stem}_metrics.json"
            )
            assert metrics_path.is_file()


@pytest.mark.parametrize("outcome_name", MODELED_OUTCOME_NAMES)
def test_every_modeled_outcome_writes_a_display_table(
    completed: CompletedRun,
    outcome_name: str,
) -> None:
    """The per-outcome cross-model display CSV is declared and written."""
    assert _artifact(completed, "s06_evaluate", "display_table", outcome_name).is_file()


# --- repeat-run stability ---------------------------------------------------


def _non_figure_snapshot(output_root: Path) -> dict[str, bytes]:
    """Return every non-PNG artifact's bytes, keyed by relative path."""
    return {
        str(path.relative_to(output_root)): path.read_bytes()
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.suffix != ".png"
    }


def test_repeating_explicit_and_divide_stages_is_byte_stable(tmp_path: Path) -> None:
    """Re-running the deterministic stages replaces the same bytes."""
    _seed_project_root(tmp_path)
    context = build_context(config=CONFIG, project_root=tmp_path)
    output_root = tmp_path / "output"

    def run() -> None:
        s01_outcome(context, output_root)
        s01b_outcome_analysis(context, output_root)
        s02_feature(context, output_root)
        for outcome_name in outcome_names():
            s03_divide(outcome_name, context, output_root)

    run()
    first = _non_figure_snapshot(output_root)
    run()
    second = _non_figure_snapshot(output_root)

    assert first == second
    assert first
