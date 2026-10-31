"""Integration tests for the ``s01b_outcome_analysis`` stage.

The stage consumes the outcome parquet produced by ``s01_outcome`` and turns
the registered outcome comparisons into a wide difference table plus its
on-disk reports. These tests exercise the complete on-disk path on a temporary
project root seeded with the committed demo inputs: the s01 stage runs first,
then s01b reads its declared artifact through the layout.

The assertions cover the declared artifact set and filenames, the exact
left-minus-right difference table, the full CSV projection, deterministic
statistics text, repeated-run stability, and the four plots per comparison.
A constant-value outcome table is used to prove that every plot helper still
writes its declared PNG when a KDE cannot be estimated, and a malformed
outcome artifact proves the stage preserves its boundary label instead of
surfacing an anonymous validation error.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.core.paths import PipelineContext
from src.project.columns import CASE_DATE, CASE_ID, KEY_COLUMNS
from src.project.config import CONFIG
from src.project.outcomes import OUTCOME_COMPARISONS
from src.project.pipeline import STAGES, build_context
from src.project.schemas import (
    OUTCOME_COLUMNS,
    S01_OUTCOME_SCHEMA,
    BoundaryError,
    validate,
)
from src.stages.s01_outcome import main as s01_main
from src.stages.s01b_outcome_analysis import (
    OUTCOME_BOUNDARY_LABEL,
    build_difference_table,
    comparison_stem,
    difference_column_name,
    difference_statistics_text,
    main,
    s01b_outcome_analysis,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Repository root, used only to locate the committed demo input table."""

SOURCE_FILENAME = "source_table_1.parquet"
"""Committed demo filename, copied into each temporary project root."""

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
"""First eight bytes of every PNG artifact the plot helpers write."""

DIFFERENCE_COLUMNS = tuple(
    difference_column_name(comparison) for comparison in OUTCOME_COMPARISONS
)
"""Wide difference columns, in comparison registry order."""


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


def _declared_artifact_paths(output_root: Path) -> list[Path]:
    """Return every s01b artifact path declared by the stage table."""
    stage_dir = output_root / "s01b_outcome_analysis"
    return [
        stage_dir / filename
        for filename in STAGES["s01b_outcome_analysis"].artifacts.values()
    ]


def _run(project_root: Path, context: PipelineContext) -> Path:
    """Run s01 and then s01b, returning the s01b stage directory."""
    output_root = project_root / "output"
    s01_main(context, output_root)
    main(context, output_root)
    return output_root / "s01b_outcome_analysis"


def _expected_outcome_df(project_root: Path, context: PipelineContext) -> pl.DataFrame:
    """Return the s01 outcome artifact written by the prerequisite stage."""
    return pl.read_parquet(
        project_root / "output" / context.layout.artifact("s01_outcome", "outcome_df")
    )


def _write_constant_outcome_table(
    project_root: Path,
    context: PipelineContext,
) -> None:
    """Write a valid but constant outcome artifact for the insufficient-data case.

    Every outcome equals the same constant, so both overlaid KDE curves and the
    difference KDE cannot estimate a density. The stage must still emit all
    four PNGs per comparison.
    """
    output_root = project_root / "output"
    outcome_path = output_root / context.layout.artifact("s01_outcome", "outcome_df")
    (output_root / "s01_outcome").mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            CASE_ID: ["case_a", "case_b"],
            CASE_DATE: pl.Series([date(2024, 1, 1), date(2024, 1, 2)], dtype=pl.Date),
            **{name: [5.0, 5.0] for name in OUTCOME_COLUMNS},
        }
    )
    validate(frame, S01_OUTCOME_SCHEMA, "constant outcome table")
    frame.write_parquet(outcome_path)


# --- successful stage run --------------------------------------------------


def test_stage_writes_exactly_the_declared_artifacts(
    project_root: Path,
    context: PipelineContext,
) -> None:
    stage_dir = _run(project_root, context)

    expected_names = sorted(STAGES["s01b_outcome_analysis"].artifacts.values())
    assert sorted(path.name for path in stage_dir.iterdir()) == expected_names
    for path in _declared_artifact_paths(project_root / "output"):
        assert path.is_file()
        assert path.stat().st_size > 0


def test_difference_table_is_left_minus_right_for_every_comparison(
    project_root: Path,
    context: PipelineContext,
) -> None:
    stage_dir = _run(project_root, context)
    written = pl.read_parquet(stage_dir / "outcome_differences.parquet")
    outcome_df = _expected_outcome_df(project_root, context)

    assert tuple(written.columns) == (*KEY_COLUMNS, *DIFFERENCE_COLUMNS)
    assert written.height == outcome_df.height == CONFIG.demo_case_count
    for comparison in OUTCOME_COMPARISONS:
        column = difference_column_name(comparison)
        expected = outcome_df[comparison.left] - outcome_df[comparison.right]
        assert written[column].equals(expected)
    assert written.equals(build_difference_table(outcome_df))


def test_full_csv_projection_matches_the_parquet_table(
    project_root: Path,
    context: PipelineContext,
) -> None:
    stage_dir = _run(project_root, context)
    csv_path = stage_dir / "outcome_differences.csv"
    parquet_df = pl.read_parquet(stage_dir / "outcome_differences.parquet")
    csv_df = pl.read_csv(csv_path, try_parse_dates=True)

    assert csv_df.height == parquet_df.height
    assert tuple(csv_df.columns) == tuple(parquet_df.columns)
    assert csv_df[CASE_ID].to_list() == parquet_df[CASE_ID].to_list()
    assert csv_df[CASE_DATE].to_list() == parquet_df[CASE_DATE].to_list()
    for column in DIFFERENCE_COLUMNS:
        assert csv_df[column].to_list() == pytest.approx(parquet_df[column].to_list())


def test_statistics_text_is_deterministic_and_covers_every_comparison(
    project_root: Path,
    context: PipelineContext,
) -> None:
    stage_dir = _run(project_root, context)
    statistics_path = stage_dir / "outcome_differences_statistics.txt"
    difference_df = pl.read_parquet(stage_dir / "outcome_differences.parquet")
    text = statistics_path.read_text(encoding="utf-8")

    assert text == difference_statistics_text(difference_df)
    assert text.startswith("Outcome difference statistics\n")
    assert f"rows: {difference_df.height}\n" in text
    for comparison in OUTCOME_COMPARISONS:
        assert difference_column_name(comparison) in text


def test_every_comparison_emits_four_plot_artifacts(
    project_root: Path,
    context: PipelineContext,
) -> None:
    stage_dir = _run(project_root, context)

    for comparison in OUTCOME_COMPARISONS:
        stem = comparison_stem(comparison)
        for suffix in (
            "histogram",
            "kde",
            "difference_histogram",
            "difference_kde",
        ):
            png_path = stage_dir / f"{stem}_{suffix}.png"
            assert png_path.is_file()
            assert png_path.read_bytes().startswith(PNG_SIGNATURE)


def test_plot_artifacts_are_still_written_for_constant_outcomes(
    project_root: Path,
    context: PipelineContext,
) -> None:
    _write_constant_outcome_table(project_root, context)

    main(context, project_root / "output")

    for comparison in OUTCOME_COMPARISONS:
        stem = comparison_stem(comparison)
        for suffix in (
            "histogram",
            "kde",
            "difference_histogram",
            "difference_kde",
        ):
            png_path = (
                project_root
                / "output"
                / "s01b_outcome_analysis"
                / f"{stem}_{suffix}.png"
            )
            assert png_path.is_file()
            assert png_path.read_bytes().startswith(PNG_SIGNATURE)


def test_rerunning_the_stage_replaces_the_same_artifacts(
    project_root: Path,
    context: PipelineContext,
) -> None:
    stage_dir = _run(project_root, context)
    output_root = stage_dir.parent
    first_bytes = {
        path.name: path.read_bytes() for path in _declared_artifact_paths(output_root)
    }

    _run(project_root, context)

    assert {
        path.name: path.read_bytes() for path in _declared_artifact_paths(output_root)
    } == first_bytes
    assert sorted(path.name for path in stage_dir.iterdir()) == sorted(first_bytes)


def test_stage_entry_point_is_exported_under_its_stage_name() -> None:
    assert s01b_outcome_analysis is main


# --- boundary failures -----------------------------------------------------


def test_non_finite_outcome_artifact_fails_at_the_outcome_boundary(
    project_root: Path,
    context: PipelineContext,
) -> None:
    output_root = project_root / "output"
    s01_main(context, output_root)
    outcome_path = output_root / context.layout.artifact("s01_outcome", "outcome_df")
    malformed = pl.read_parquet(outcome_path).with_columns(
        pl.lit(float("nan")).alias("outcome_2")
    )
    malformed.write_parquet(outcome_path)

    with pytest.raises(BoundaryError) as caught:
        main(context, output_root)

    error = caught.value
    assert error.label == OUTCOME_BOUNDARY_LABEL
    assert error.column == "outcome_2"
    assert error.rule == "finite"
    assert OUTCOME_BOUNDARY_LABEL in str(error)


def test_missing_outcome_artifact_is_reported(
    project_root: Path,
    context: PipelineContext,
) -> None:
    with pytest.raises(FileNotFoundError):
        main(context, project_root / "output")
