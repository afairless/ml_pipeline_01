"""Integration tests for the ``s01_outcome`` stage.

The stage is the first thin entry point in the pipeline: it ingests the
external source table through the input registry, delegates outcome derivation
to ``src/project/adapters.py``, validates the result at the outcome boundary,
and writes the two artifacts the stage table declares. These tests exercise
that whole path on a temporary project root seeded with the committed demo
table, so they pin the on-disk contract rather than an in-memory helper.

The assertions cover the artifact set and filenames, the written frame and its
schema, deterministic statistics text, repeated-run stability, the explicit
``project_root`` seam for an output root outside the project tree, and the
boundary failure cases: an adapter that returns a non-finite value and an
adapter that drops a row. Both failures must preserve the stage's boundary
label instead of surfacing an anonymous validation error.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import polars as pl
import pytest

import src.stages.s01_outcome as stage_module
from src.core.paths import PipelineContext
from src.project.adapters import build_outcome_table
from src.project.columns import CASE_DATE, CASE_ID
from src.project.config import CONFIG, ProjectConfig
from src.project.inputs import read_input_table
from src.project.pipeline import build_context
from src.project.schemas import (
    OUTCOME_COLUMNS,
    S01_OUTCOME_SCHEMA,
    BoundaryError,
    validate,
)
from src.stages.s01_outcome import (
    OUTCOME_BOUNDARY_LABEL,
    main,
    outcome_statistics_text,
    s01_outcome,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Repository root, used only to locate the committed demo input table."""

SOURCE_FILENAME = "source_table_1.parquet"
"""Committed demo filename, copied into each temporary project root."""


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Return a temporary project root containing the demo source table."""
    data_dir = tmp_path / CONFIG.data_root
    data_dir.mkdir(parents=True)
    shutil.copyfile(
        PROJECT_ROOT / CONFIG.data_root / SOURCE_FILENAME,
        data_dir / SOURCE_FILENAME,
    )
    return tmp_path


@pytest.fixture
def context(project_root: Path) -> PipelineContext:
    """Return the pipeline context bound to the temporary project root."""
    return build_context(config=CONFIG, project_root=project_root)


def _run(
    project_root: Path,
    context: PipelineContext,
    *,
    output_root: Path | None = None,
) -> tuple[Path, Path]:
    """Run the stage and return its two declared artifact paths."""
    resolved_output = project_root / "output" if output_root is None else output_root
    main(context, resolved_output)
    stage_dir = resolved_output / "s01_outcome"
    return (
        stage_dir / "outcome_df.parquet",
        stage_dir / "outcome_statistics.txt",
    )


def _expected_outcome_df(project_root: Path) -> pl.DataFrame:
    """Rebuild the expected outcome frame through the same public path."""
    source_df = read_input_table(
        "source_table_1",
        project_root / CONFIG.data_root,
    )
    return build_outcome_table(source_df, CONFIG)


# --- successful stage run --------------------------------------------------


def test_stage_writes_exactly_the_declared_artifacts(
    project_root: Path,
    context: PipelineContext,
) -> None:
    outcome_path, statistics_path = _run(project_root, context)

    stage_dir = outcome_path.parent
    assert sorted(path.name for path in stage_dir.iterdir()) == [
        "outcome_df.parquet",
        "outcome_statistics.txt",
    ]
    assert outcome_path.is_file()
    assert statistics_path.is_file()
    assert statistics_path.stat().st_size > 0


def test_stage_writes_the_adapter_outcome_table(
    project_root: Path,
    context: PipelineContext,
) -> None:
    outcome_path, _ = _run(project_root, context)

    written = pl.read_parquet(outcome_path)
    expected = _expected_outcome_df(project_root)

    assert written.equals(expected)
    assert written.height == CONFIG.demo_case_count
    assert tuple(written.columns) == (CASE_ID, CASE_DATE, *OUTCOME_COLUMNS)
    assert (
        validate(
            written,
            S01_OUTCOME_SCHEMA,
            "written outcome table",
            expected_row_count=written.height,
        )
        is written
    )


def test_stage_writes_deterministic_statistics_text(
    project_root: Path,
    context: PipelineContext,
) -> None:
    _, statistics_path = _run(project_root, context)

    expected_text = outcome_statistics_text(_expected_outcome_df(project_root))
    assert statistics_path.read_text(encoding="utf-8") == expected_text
    assert expected_text.startswith("Outcome statistics\n")
    assert f"rows: {CONFIG.demo_case_count}\n" in expected_text
    for name in OUTCOME_COLUMNS:
        assert any(line.startswith(f"{name} ") for line in expected_text.splitlines())


def test_outcome_statistics_text_is_stable() -> None:
    frame = pl.DataFrame(
        {
            CASE_ID: ["case_a", "case_b"],
            CASE_DATE: pl.Series(
                [date(2024, 1, 1), date(2024, 1, 2)],
                dtype=pl.Date,
            ),
            **{name: [1.0, 2.0] for name in OUTCOME_COLUMNS},
        }
    )

    assert outcome_statistics_text(frame) == outcome_statistics_text(frame)
    assert "1.500000" in outcome_statistics_text(frame)


def test_rerunning_the_stage_replaces_the_same_artifacts(
    project_root: Path,
    context: PipelineContext,
) -> None:
    outcome_path, statistics_path = _run(project_root, context)
    first_outcome_bytes = outcome_path.read_bytes()
    first_statistics = statistics_path.read_text(encoding="utf-8")

    outcome_path, statistics_path = _run(project_root, context)

    assert outcome_path.read_bytes() == first_outcome_bytes
    assert statistics_path.read_text(encoding="utf-8") == first_statistics
    assert sorted(path.name for path in outcome_path.parent.iterdir()) == [
        "outcome_df.parquet",
        "outcome_statistics.txt",
    ]


def test_explicit_project_root_resolves_the_data_directory(
    project_root: Path,
    context: PipelineContext,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    output_root = tmp_path_factory.mktemp("elsewhere") / "output"

    stage_module.main(context, output_root, project_root=project_root)

    written = pl.read_parquet(output_root / "s01_outcome" / "outcome_df.parquet")
    assert written.equals(_expected_outcome_df(project_root))


def test_stage_entry_point_is_exported_under_its_stage_name() -> None:
    assert s01_outcome is main


# --- boundary failures -----------------------------------------------------


def test_non_finite_adapter_result_fails_at_the_outcome_boundary(
    project_root: Path,
    context: PipelineContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def malformed(source_df: pl.DataFrame, config: ProjectConfig) -> pl.DataFrame:
        del config
        frame = source_df.select(CASE_ID, CASE_DATE)
        for name in OUTCOME_COLUMNS:
            frame = frame.with_columns(pl.lit(1.0).alias(name))
        return frame.with_columns(pl.lit(float("nan")).alias("outcome_2"))

    monkeypatch.setattr(stage_module, "build_outcome_table", malformed)

    with pytest.raises(BoundaryError) as caught:
        _run(project_root, context)

    error = caught.value
    assert error.label == OUTCOME_BOUNDARY_LABEL
    assert error.column == "outcome_2"
    assert error.rule == "finite"
    assert OUTCOME_BOUNDARY_LABEL in str(error)


def test_row_dropping_adapter_result_fails_at_the_outcome_boundary(
    project_root: Path,
    context: PipelineContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def truncated(source_df: pl.DataFrame, config: ProjectConfig) -> pl.DataFrame:
        return build_outcome_table(source_df, config).head(source_df.height - 1)

    monkeypatch.setattr(stage_module, "build_outcome_table", truncated)

    with pytest.raises(BoundaryError) as caught:
        _run(project_root, context)

    error = caught.value
    expected_height = _expected_outcome_df(project_root).height
    assert error.label == OUTCOME_BOUNDARY_LABEL
    assert error.column is None
    assert error.rule == "row_count"
    assert str(expected_height) in str(error)
    assert str(expected_height - 1) in str(error)
