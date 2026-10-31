"""Derive the pipeline outcome table from the external source table.

This stage is deliberately thin. It resolves its output directory through the
declarative layout, reads the external source table through the input registry,
delegates the actual outcome construction to
:func:`src.project.adapters.build_outcome_table`, validates the result at the
``S01_OUTCOME_SCHEMA`` boundary, and writes the two artifacts the stage table
declares: the outcome parquet and a deterministic statistics text.

``src/project/adapters.py`` is the replacement point for a new project. Replace
the adapter body with real outcome derivation; this module and every other
stage keep working unchanged as long as the adapter returns the canonical key
columns plus the registered outcome columns.

The stage separates its three responsibilities explicitly: ingestion reads and
validates the source table without mutating it; transformation is the adapter
call; output serializes the validated frame and its summary. Both artifacts are
written through a temporary sibling and moved into place, so a re-run replaces
only this stage's declared files and a failed write never leaves a partial
artifact behind.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import polars as pl

from src.core.paths import PipelineContext
from src.project.adapters import build_outcome_table
from src.project.columns import CASE_DATE
from src.project.config import ProjectConfig
from src.project.inputs import read_input_table
from src.project.schemas import OUTCOME_COLUMNS, S01_OUTCOME_SCHEMA, validate

STAGE_NAME = "s01_outcome"
"""Stage-table key whose directory and artifact filenames this module uses."""

SOURCE_TABLE = "source_table_1"
"""Logical name of the external input table this stage ingests."""

OUTCOME_BOUNDARY_LABEL = "s01 outcome table"
"""Boundary label used when the adapter result violates its schema."""

PARQUET_COMPRESSION = "zstd"
"""Compression codec of the outcome parquet artifact."""

STATISTICS_TITLE = "Outcome statistics"
"""Title line of the deterministic statistics text."""

_STATISTIC_HEADERS = ("count", "mean", "std", "min", "median", "max")
"""Numeric summary columns, in render order."""

_NAME_WIDTH = 16
"""Width of the outcome-name column in the statistics table."""

_VALUE_WIDTH = 14
"""Width of every numeric statistics column."""


def _format_number(value: float | None) -> str:
    """Render one statistics value with a fixed, deterministic width.

    A missing value (for example a standard deviation of a one-row frame) is
    rendered as ``n/a`` instead of an empty cell. Finite values use six decimal
    places, and a non-finite value is rendered by its Python spelling; the
    outcome schema rejects non-finite values before statistics are written, so
    that branch is defensive rather than expected.
    """
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return repr(value)
    return f"{value:.6f}"


def _statistics_header() -> str:
    """Return the aligned header row of the statistics table."""
    return f"{'outcome':<{_NAME_WIDTH}}" + "".join(
        f"{name:>{_VALUE_WIDTH}}" for name in _STATISTIC_HEADERS
    )


def outcome_statistics_text(outcome_df: pl.DataFrame) -> str:
    """Return the deterministic statistics text for an outcome table.

    The summary is computed from the registered outcome columns only: the row
    count, the case-date range, and the count/mean/standard deviation/minimum/
    median/maximum of every outcome. Rendering fixes the column widths and the
    decimal precision, so the same frame always produces the same bytes on a
    pinned ``polars`` version.

    Args:
        outcome_df: A frame carrying the key columns and every registered
            outcome. The caller normally validates it against
            :data:`~src.project.schemas.S01_OUTCOME_SCHEMA` first.

    Returns:
        A newline-terminated text report, with a trailing blank line kept out of
        the table body.
    """
    rows = [
        STATISTICS_TITLE,
        "=" * len(STATISTICS_TITLE),
        f"rows: {outcome_df.height}",
        f"case_date_range: {outcome_df[CASE_DATE].min()} .. "
        f"{outcome_df[CASE_DATE].max()}",
        "",
        _statistics_header(),
    ]
    for name in OUTCOME_COLUMNS:
        series = outcome_df[name]
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


def _write_outcome_parquet(path: Path, outcome_df: pl.DataFrame) -> None:
    """Serialize the outcome frame with explicit deterministic writer settings."""
    _write_atomically(
        path,
        lambda target: outcome_df.write_parquet(
            target,
            compression=PARQUET_COMPRESSION,
            statistics=True,
        ),
    )


def _write_statistics(path: Path, outcome_df: pl.DataFrame) -> None:
    """Serialize the outcome statistics text as UTF-8 with ``\\n`` newlines."""
    text = outcome_statistics_text(outcome_df)
    _write_atomically(
        path,
        lambda target: target.write_text(text, encoding="utf-8"),
    )


def _source_data_dir(
    context: PipelineContext,
    output_root: Path,
    project_root: Path | None,
) -> Path:
    """Resolve the external data directory for this stage's run.

    ``output_root`` is the pipeline output root, normally
    ``<project_root>/output``, so its parent is the project root that a
    relative ``config.data_root`` is anchored to. A caller whose output root is
    elsewhere passes ``project_root`` explicitly instead.
    """
    root = Path(output_root).parent if project_root is None else Path(project_root)
    data_root = Path(context.config.data_root)
    return data_root if data_root.is_absolute() else root / data_root


def main(
    context: PipelineContext,
    output_root: Path,
    *,
    project_root: Path | None = None,
) -> None:
    """Run the outcome stage once.

    Reads ``source_table_1`` from the configured data directory, builds the
    outcome frame through the project adapter, validates it against
    :data:`~src.project.schemas.S01_OUTCOME_SCHEMA` with the source row count as
    the expected count, and writes the declared parquet and statistics
    artifacts under ``output_root/s01_outcome``.

    Args:
        context: The frozen pipeline context carrying the layout and config.
        output_root: The explicit output root the relative layout paths are
            joined with.
        project_root: Optional explicit root that a relative
            ``config.data_root`` is resolved against. Defaults to the parent of
            ``output_root``, matching the pipeline's
            ``<project_root>/output`` convention.

    Raises:
        FileNotFoundError: The source table cannot be found under the resolved
            data directory.
        ValueError: Input discovery is ambiguous, or the adapter result
            violates its schema.
        BoundaryError: The source table or the adapter result violates its
            declared boundary schema.
    """
    layout = context.layout
    stage_dir = Path(output_root) / layout.stage_dir(STAGE_NAME)
    context.ensure(stage_dir)

    data_dir = _source_data_dir(context, output_root, project_root)
    source_df = read_input_table(SOURCE_TABLE, data_dir)

    outcome_df = build_outcome_table(
        source_df,
        cast(ProjectConfig, context.config),
    )
    validate(
        outcome_df,
        S01_OUTCOME_SCHEMA,
        OUTCOME_BOUNDARY_LABEL,
        expected_row_count=source_df.height,
    )

    outcome_path = Path(output_root) / layout.artifact(STAGE_NAME, "outcome_df")
    statistics_path = Path(output_root) / layout.artifact(
        STAGE_NAME,
        "outcome_statistics",
    )
    _write_outcome_parquet(outcome_path, outcome_df)
    _write_statistics(statistics_path, outcome_df)


s01_outcome = main
"""Stage entry point under the name used by the explicit pipeline sequence."""

__all__ = [
    "OUTCOME_BOUNDARY_LABEL",
    "main",
    "outcome_statistics_text",
    "s01_outcome",
]
