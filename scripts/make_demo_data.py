"""Deterministic generator for the committed demo source table.

The template ships one external input table so the whole pipeline can run
without any supplied data. This script regenerates that table from
``CONFIG.demo_seed`` and ``CONFIG.demo_case_count``: one row per case, a
two-day date step, and three raw values whose noise comes from a single
``numpy.random.default_rng`` consumed in row order. Regenerating the table is
therefore reproducible byte for byte as long as the pinned ``polars`` and
``numpy`` versions do not change.

Usage:

``python scripts/make_demo_data.py``
    Rewrite ``<repository>/data/demo/source_table_1.parquet``.
``python scripts/make_demo_data.py --output data/demo/other.parquet``
    Write to another path. A relative path resolves against the project root,
    never against the current working directory.

The resolved output path must stay strictly inside the project's
``data/demo/`` directory. A path outside it is a usage error (exit status
``2``), so a mistyped ``--output`` cannot scatter data files through the
repository. Writes are atomic: the table is serialized to a temporary sibling
and moved into place, so a failed write never leaves a half-written parquet
file behind.

``project_root`` is a keyword-only test seam. Tests pass a temporary project
root and a relative ``--output`` so both the containment base and the written
file live under the temporary directory.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project.config import CONFIG  # noqa: E402
from src.project.schemas import SOURCE_TABLE_SCHEMA, validate  # noqa: E402

DEMO_DATA_DIR = Path("data") / "demo"
"""Project-root-relative directory that bounds every output path."""

DEFAULT_OUTPUT = DEMO_DATA_DIR / "source_table_1.parquet"
"""Project-root-relative output path used when ``--output`` is omitted."""

FIRST_CASE_DATE = date(2022, 6, 1)
"""Date of the first demo case; later cases step forward from here."""

CASE_DATE_STEP_DAYS = 2
"""Day step between consecutive demo cases."""

PARQUET_COMPRESSION = "zstd"
"""Compression codec of the committed parquet artifact."""

SOURCE_BOUNDARY_LABEL = "demo source table"
"""Boundary label used when the generated frame violates its schema."""


def build_source_table(*, seed: int, case_count: int) -> pl.DataFrame:
    """Build the demo source table in memory.

    The three raw values follow the documented demo contract: a slow
    sine/cosine signal plus small noise, and a third value derived from the
    first two. Noise draws happen once per row and per value, in row order, so
    the frame depends only on ``seed`` and ``case_count``.

    Args:
        seed: Seed for the generator's private ``numpy`` random generator.
        case_count: Number of case rows to emit; must be positive.

    Returns:
        A frame with ``case_id``, ``case_date``, ``value_1``, ``value_2``,
        and ``value_3``, validated against
        :data:`~src.project.schemas.SOURCE_TABLE_SCHEMA`.

    Raises:
        ValueError: ``case_count`` is not positive.
        BoundaryError: The generated frame violates the source-table schema.
    """
    if case_count < 1:
        raise ValueError(f"case_count must be positive; got {case_count}")

    generator = np.random.default_rng(seed)
    case_ids: list[str] = []
    case_dates: list[date] = []
    values_1: list[float] = []
    values_2: list[float] = []
    values_3: list[float] = []

    for index in range(case_count):
        value_1 = math.sin(index / 7.0) + 0.05 * generator.normal()
        value_2 = math.cos(index / 11.0) + 0.05 * generator.normal()
        value_3 = value_1 - value_2 + 0.02 * generator.normal()

        case_ids.append(f"case_{index:04d}")
        case_dates.append(FIRST_CASE_DATE + timedelta(days=CASE_DATE_STEP_DAYS * index))
        values_1.append(value_1)
        values_2.append(value_2)
        values_3.append(value_3)

    frame = pl.DataFrame(
        {
            "case_id": pl.Series(case_ids, dtype=pl.String),
            "case_date": pl.Series(case_dates, dtype=pl.Date),
            "value_1": pl.Series(values_1, dtype=pl.Float64),
            "value_2": pl.Series(values_2, dtype=pl.Float64),
            "value_3": pl.Series(values_3, dtype=pl.Float64),
        }
    )
    return validate(
        frame,
        SOURCE_TABLE_SCHEMA,
        SOURCE_BOUNDARY_LABEL,
        expected_row_count=case_count,
    )


def resolve_output_path(output: Path | None, *, project_root: Path) -> Path:
    """Resolve ``output`` and enforce the ``data/demo/`` containment rule.

    A relative path is resolved against ``project_root`` so the result does
    not depend on the current working directory. Symlinks are resolved before
    the containment check, so a link inside the demo directory cannot be used
    to escape it.

    Args:
        output: Requested output path, or ``None`` for
            :data:`DEFAULT_OUTPUT`.
        project_root: Root the relative default and the containment base are
            anchored to.

    Returns:
        The absolute output path.

    Raises:
        ValueError: The resolved path equals the containment base or lies
            outside it.
    """
    root = Path(project_root).resolve()
    base = (root / DEMO_DATA_DIR).resolve()
    candidate = DEFAULT_OUTPUT if output is None else Path(output)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()

    if resolved == base:
        raise ValueError(
            f"output path must name a file inside {base}, not the directory itself"
        )
    if not resolved.is_relative_to(base):
        raise ValueError(f"output path must stay inside {base}; got {resolved}")
    return resolved


def write_source_table(frame: pl.DataFrame, output_path: Path) -> None:
    """Serialize ``frame`` to ``output_path`` atomically.

    The parquet file is written to a temporary sibling and moved into place
    with :func:`os.replace`, so readers observe either the previous file or
    the complete new one. Missing parent directories are created.

    Args:
        frame: Frame to serialize.
        output_path: Destination parquet path, already containment-checked.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        frame.write_parquet(
            temporary_path,
            compression=PARQUET_COMPRESSION,
            statistics=True,
        )
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    """Return the generator's argument parser."""
    parser = argparse.ArgumentParser(
        prog="make_demo_data",
        description=(
            "Regenerate the deterministic demo source table from the project "
            "configuration."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help=(
            "output parquet path, relative to the project root unless "
            f"absolute (default: {DEFAULT_OUTPUT.as_posix()})"
        ),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    project_root: Path | None = None,
) -> int:
    """Regenerate the demo source table and return the process exit status.

    Args:
        argv: Arguments to parse; ``None`` reads :data:`sys.argv`.
        project_root: Root that anchors the output path and the containment
            base; ``None`` uses this script's repository root. Tests pass a
            temporary root so they never touch the committed artifact.

    Returns:
        ``0`` on success. A path that escapes the demo directory exits with
        status ``2`` through :meth:`argparse.ArgumentParser.error`.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)
    root = PROJECT_ROOT if project_root is None else Path(project_root)

    try:
        output_path = resolve_output_path(arguments.output, project_root=root)
    except ValueError as error:
        parser.error(str(error))

    frame = build_source_table(
        seed=CONFIG.demo_seed,
        case_count=CONFIG.demo_case_count,
    )
    write_source_table(frame, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
