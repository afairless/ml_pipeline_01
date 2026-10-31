"""Describe the outcome table through the registered outcome comparisons.

The stage is a generic descriptive analysis of the outcome table written by
``s01_outcome``. It reads that artifact through the declarative layout, checks
it against :data:`~src.project.schemas.S01_OUTCOME_SCHEMA`, and projects the
:data:`~src.project.outcomes.OUTCOME_COMPARISONS` vocabulary into a wide
difference table. Every comparison is always evaluated as ``left - right``; the
stage encodes no other sign rule and no subject-matter outcome name.

The wide table carries the canonical key columns plus one difference column per
comparison, named ``<left>_minus_<right>``. The stage writes the declared
parquet, a full CSV projection of the same table, a deterministic statistics
text, and four plot artifacts per comparison: an overlaid histogram of the two
outcomes, an overlaid KDE of the two outcomes, a histogram of the difference,
and a KDE of the difference. The plotting helpers are owned by
:mod:`src.core.plots` and always save their declared PNG, annotating the axes
when the data cannot support the requested curve.

The three responsibilities stay separate: ingestion reads and validates the s01
artifact without mutating it; transformation computes the difference columns;
output serialises the parquet/CSV/statistics and draws the figures. Re-running
the stage atomically replaces exactly the artifacts this stage declares and
never touches another stage's output.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import polars as pl

from src.core.paths import PipelineContext
from src.core.plots import (
    plot_distribution_histogram,
    plot_distribution_kde,
    plot_overlaid_distribution_histograms,
    plot_overlaid_distribution_kdes,
)
from src.project.columns import CASE_DATE, CASE_ID, KEY_COLUMNS
from src.project.outcomes import OUTCOME_COMPARISONS, OutcomeComparison
from src.project.schemas import S01_OUTCOME_SCHEMA, validate

STAGE_NAME = "s01b_outcome_analysis"
"""Stage-table key whose directory and artifact filenames this module uses."""

SOURCE_STAGE = "s01_outcome"
"""Stage whose declared outcome parquet this analysis ingests."""

OUTCOME_BOUNDARY_LABEL = "s01b input outcome table"
"""Boundary label used when the s01 outcome artifact violates its schema."""

PARQUET_COMPRESSION = "zstd"
"""Compression codec of the difference parquet artifact."""

STATISTICS_TITLE = "Outcome difference statistics"
"""Title line of the deterministic statistics text."""

_STATISTIC_HEADERS = ("count", "mean", "std", "min", "median", "max")
"""Numeric summary columns, in render order."""

_NAME_WIDTH = 32
"""Width of the comparison column in the statistics table."""

_VALUE_WIDTH = 14
"""Width of every numeric statistics column."""


def difference_column_name(comparison: OutcomeComparison) -> str:
    """Return the wide-table column name of one comparison's difference.

    The name is derived from the comparison vocabulary only, so a renamed or
    added registry entry automatically receives a matching column without an
    edit here.

    Args:
        comparison: The registered left/right comparison.

    Returns:
        ``'<left>_minus_<right>'``, e.g. ``'outcome_1_minus_outcome_2'``.
    """
    return f"{comparison.left}_minus_{comparison.right}"


def comparison_stem(comparison: OutcomeComparison) -> str:
    """Return the filename stem shared by one comparison's four plots.

    Args:
        comparison: The registered left/right comparison.

    Returns:
        ``'<left>_vs_<right>'``, matching the artifact keys declared by the
        stage table.
    """
    return f"{comparison.left}_vs_{comparison.right}"


def build_difference_table(
    outcome_df: pl.DataFrame,
    comparisons: Sequence[OutcomeComparison] = OUTCOME_COMPARISONS,
) -> pl.DataFrame:
    """Project the comparisons into a wide left-minus-right table.

    The returned frame keeps the canonical key columns in their declared order
    and adds one ``Float64`` difference column per comparison. The input frame
    is not mutated, and no outcome column is dropped from the caller's frame;
    only the new frame is narrower.

    Args:
        outcome_df: A validated outcome frame carrying every compared outcome
            column.
        comparisons: The comparisons to evaluate, in report order. Defaults to
            the project registry.

    Returns:
        A frame with ``KEY_COLUMNS`` followed by one difference column per
        comparison, in ``comparisons`` order.
    """
    expressions = [
        (pl.col(comparison.left) - pl.col(comparison.right)).alias(
            difference_column_name(comparison)
        )
        for comparison in comparisons
    ]
    return outcome_df.select(
        pl.col(CASE_ID),
        pl.col(CASE_DATE),
        *expressions,
    )


def _format_number(value: float | None) -> str:
    """Render one statistics value with a fixed, deterministic width.

    A missing value (for example a standard deviation of a one-row frame) is
    rendered as ``n/a``. Finite values use six decimal places; the outcome
    schema rejects non-finite inputs, so the non-finite branch is defensive.
    """
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return repr(value)
    return f"{value:.6f}"


def _statistics_header() -> str:
    """Return the aligned header row of the statistics table."""
    return f"{'difference':<{_NAME_WIDTH}}" + "".join(
        f"{name:>{_VALUE_WIDTH}}" for name in _STATISTIC_HEADERS
    )


def difference_statistics_text(difference_df: pl.DataFrame) -> str:
    """Return the deterministic statistics text for a difference table.

    The summary covers every non-key column in frame order: the row count and
    the count/mean/standard deviation/minimum/median/maximum of each
    difference. Column widths and decimal precision are fixed, so the same
    frame always produces the same bytes on a pinned ``polars`` version.

    Args:
        difference_df: A frame carrying the key columns and one or more numeric
            difference columns.

    Returns:
        A newline-terminated text report.
    """
    difference_columns = [
        name for name in difference_df.columns if name not in KEY_COLUMNS
    ]
    rows = [
        STATISTICS_TITLE,
        "=" * len(STATISTICS_TITLE),
        f"rows: {difference_df.height}",
        f"comparisons: {len(difference_columns)}",
        "",
        _statistics_header(),
    ]
    for name in difference_columns:
        series = difference_df[name]
        values = tuple(
            None if value is None else cast(float, value)
            for value in (
                series.mean(),
                series.std(),
                series.min(),
                series.median(),
                series.max(),
            )
        )
        rows.append(
            f"{name:<{_NAME_WIDTH}}{series.len():>{_VALUE_WIDTH}d}"
            + "".join(f"{_format_number(value):>{_VALUE_WIDTH}}" for value in values)
        )
    return "\n".join(rows) + "\n"


def _write_atomically(path: Path, writer: Callable[[Path], object]) -> None:
    """Write one artifact through a temporary sibling and an atomic replace.

    The temporary file lives in the destination directory, so the final
    :func:`os.replace` never crosses a filesystem boundary. A failed write
    removes its temporary file and leaves the previous artifact untouched.

    Args:
        path: Destination artifact path whose parent already exists.
        writer: Callback that serializes the artifact to the temporary path.
    """
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        writer(temporary_path)
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_difference_parquet(path: Path, difference_df: pl.DataFrame) -> None:
    """Serialize the difference frame with deterministic writer settings."""
    _write_atomically(
        path,
        lambda target: difference_df.write_parquet(
            target,
            compression=PARQUET_COMPRESSION,
            statistics=True,
        ),
    )


def _write_difference_csv(path: Path, difference_df: pl.DataFrame) -> None:
    """Serialize the full difference frame as CSV."""
    _write_atomically(path, lambda target: difference_df.write_csv(target))


def _write_statistics(path: Path, difference_df: pl.DataFrame) -> None:
    """Serialize the difference statistics text as UTF-8 with ``\\n`` newlines."""
    text = difference_statistics_text(difference_df)
    _write_atomically(
        path,
        lambda target: target.write_text(text, encoding="utf-8"),
    )


def _read_outcome_table(context: PipelineContext, output_root: Path) -> pl.DataFrame:
    """Read and validate the s01 outcome artifact.

    Raises:
        FileNotFoundError: The prerequisite stage has not produced its outcome
            parquet under ``output_root``.
        BoundaryError: The artifact violates
            :data:`~src.project.schemas.S01_OUTCOME_SCHEMA`.
    """
    outcome_path = Path(output_root) / context.layout.artifact(
        SOURCE_STAGE,
        "outcome_df",
    )
    outcome_df = pl.read_parquet(outcome_path)
    validate(outcome_df, S01_OUTCOME_SCHEMA, OUTCOME_BOUNDARY_LABEL)
    return outcome_df


def _plot_comparison(
    context: PipelineContext,
    output_root: Path,
    outcome_df: pl.DataFrame,
    difference_df: pl.DataFrame,
    comparison: OutcomeComparison,
) -> None:
    """Write the four declared plots for one comparison.

    The two overlay plots read the original outcome columns; the two difference
    plots read the projected difference column. Every helper is figure-owning
    and saves its declared PNG even when the data is insufficient for the
    requested curve.
    """
    layout = context.layout
    stem = comparison_stem(comparison)
    difference_column = difference_column_name(comparison)
    suptitle = f"{comparison.left} vs {comparison.right}"

    histogram_path = Path(output_root) / layout.artifact(
        STAGE_NAME,
        f"{stem}_histogram",
    )
    kde_path = Path(output_root) / layout.artifact(STAGE_NAME, f"{stem}_kde")
    difference_histogram_path = Path(output_root) / layout.artifact(
        STAGE_NAME,
        f"{stem}_difference_histogram",
    )
    difference_kde_path = Path(output_root) / layout.artifact(
        STAGE_NAME,
        f"{stem}_difference_kde",
    )

    plot_overlaid_distribution_histograms(
        outcome_df,
        comparison.left,
        comparison.right,
        suptitle,
        "Overlaid outcome histograms",
        histogram_path,
    )
    plot_overlaid_distribution_kdes(
        outcome_df,
        comparison.left,
        comparison.right,
        suptitle,
        "Overlaid outcome KDEs",
        kde_path,
    )
    plot_distribution_histogram(
        difference_df,
        difference_column,
        suptitle,
        f"Histogram of {difference_column}",
        difference_histogram_path,
    )
    plot_distribution_kde(
        difference_df,
        difference_column,
        suptitle,
        f"KDE of {difference_column}",
        difference_kde_path,
        zero_line=True,
    )


def main(context: PipelineContext, output_root: Path) -> None:
    """Run the outcome-analysis stage once.

    Reads the declared s01 outcome parquet, validates it at the outcome
    boundary, projects the registered comparisons into a wide difference
    table, and writes every artifact the stage table declares: the difference
    parquet and full CSV, the statistics text, and four plots per comparison.

    Args:
        context: The frozen pipeline context carrying the layout and config.
        output_root: The explicit output root the relative layout paths are
            joined with.

    Raises:
        FileNotFoundError: The s01 outcome parquet is missing.
        BoundaryError: The s01 outcome artifact violates its declared schema.
    """
    layout = context.layout
    stage_dir = Path(output_root) / layout.stage_dir(STAGE_NAME)
    context.ensure(stage_dir)

    outcome_df = _read_outcome_table(context, output_root)
    difference_df = build_difference_table(outcome_df)

    difference_path = Path(output_root) / layout.artifact(
        STAGE_NAME,
        "difference_df",
    )
    csv_path = Path(output_root) / layout.artifact(
        STAGE_NAME,
        "difference_df_csv",
    )
    statistics_path = Path(output_root) / layout.artifact(
        STAGE_NAME,
        "difference_statistics",
    )

    _write_difference_parquet(difference_path, difference_df)
    _write_difference_csv(csv_path, difference_df)
    _write_statistics(statistics_path, difference_df)

    for comparison in OUTCOME_COMPARISONS:
        _plot_comparison(context, output_root, outcome_df, difference_df, comparison)


s01b_outcome_analysis = main
"""Stage entry point under the name used by the explicit pipeline sequence."""

__all__ = [
    "OUTCOME_BOUNDARY_LABEL",
    "build_difference_table",
    "comparison_stem",
    "difference_column_name",
    "difference_statistics_text",
    "main",
    "s01b_outcome_analysis",
]
