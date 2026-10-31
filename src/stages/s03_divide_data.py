"""Divide one outcome's joined table into the canonical holdout partitions.

This stage is the bridge between the outcome and feature producers and the
training stage. For one registered outcome it joins that single outcome column
to the feature table on the canonical key columns, asserts the key uniqueness
and row-conservation contract, splits the joined rows on ``case_date``, and
writes the three holdout partitions plus the two head frames and a
deterministic statistics text under ``s03_divide/{outcome}/``.

Each per-outcome directory carries only that outcome's column. A later stage
loads one directory at a time, so duplicating every outcome into every
directory would make the artifact contract ambiguous; the outcome name is part
of the stage directory, not a column selector the consumer has to know about.

The split boundaries are fixed by the configuration and are half-open in the
sense that a case belongs to exactly one partition:

- ``train``: ``case_date <= config.train_end_date``
- ``valid``: ``config.train_end_date < case_date < config.test_start_date``
- ``test``: ``case_date >= config.test_start_date``

The stage asserts ``test_start_date > train_end_date`` before reading anything,
that every partition is non-empty before writing anything, and that the three
partitions are pairwise disjoint and together cover the joined table exactly.
A violated contract fails before the first artifact is written, so a failed run
never leaves a partial split set behind.

The three responsibilities are kept separate: ingestion reads and validates the
s01 outcome and s02 feature artifacts without mutating them; transformation is
the join and the date split; output serializes the validated partitions, their
head frames, and the statistics text through a temporary sibling and an atomic
replace. Re-running the stage replaces only the artifacts this stage declares
and never touches another stage's output.

``src/project/adapters.py`` remains the replacement point for a new project:
this stage consumes whatever outcome and feature tables those adapters produce
as long as the frames satisfy the declared boundary schemas.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import cast

import polars as pl

from src.core.paths import PipelineContext
from src.core.splits import PARTITION_ORDER
from src.project.columns import CASE_DATE, KEY_COLUMNS
from src.project.schemas import (
    FEATURE_COLUMNS,
    S01_OUTCOME_SCHEMA,
    S02_FEATURE_SCHEMA,
    S03_SPLIT_SCHEMA,
    validate,
)

STAGE_NAME = "s03_divide"
"""Stage-table key whose directory and artifact filenames this module uses."""

OUTCOME_STAGE = "s01_outcome"
"""Stage whose declared outcome parquet supplies the modeled outcome column."""

FEATURE_STAGE = "s02_feature"
"""Stage whose declared feature parquet supplies the feature matrix."""

OUTCOME_BOUNDARY_LABEL = "s03 input outcome table"
"""Boundary label used when the s01 outcome artifact violates its schema."""

FEATURE_BOUNDARY_LABEL = "s03 input feature table"
"""Boundary label used when the s02 feature artifact violates its schema."""

SPLIT_BOUNDARY_LABEL = "s03 {partition} split"
"""Boundary label used when one partition violates its split schema."""

PARQUET_COMPRESSION = "zstd"
"""Compression codec of every parquet artifact this stage writes."""

HEAD_ROW_COUNT = 5
"""Number of leading joined rows written to the head frames."""

STATISTICS_TITLE = "Split statistics"
"""Title line of the deterministic statistics text."""

_PARTITION_WIDTH = 10
"""Width of the partition-name column in the statistics table."""

_ROW_WIDTH = 8
"""Width of the row-count column in the statistics table."""

_DATE_WIDTH = 12
"""Width of the start/end date columns in the statistics table."""

_VALUE_WIDTH = 14
"""Width of every numeric statistics column."""

_STATISTIC_HEADERS = (
    "rows",
    "start",
    "end",
    "mean",
    "std",
    "min",
    "median",
    "max",
)
"""Statistics columns, in render order."""


def _require_key_columns(frame: pl.DataFrame, label: str) -> pl.DataFrame:
    """Return only the canonical key columns, in declared order.

    Args:
        frame: A frame that must carry every canonical key column.
        label: Boundary name used in the failure message.

    Raises:
        ValueError: A key column is missing from ``frame``.
    """
    missing = [name for name in KEY_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{label} is missing key column(s) {missing!r}; expected the "
            f"canonical key columns {list(KEY_COLUMNS)!r}"
        )
    return frame.select(list(KEY_COLUMNS))


def assert_date_boundaries(train_end_date: date, test_start_date: date) -> None:
    """Assert the configured split boundaries leave a non-empty valid range.

    The validation partition is defined by ``train_end_date < case_date <
    test_start_date``, so a ``test_start_date`` at or before the
    ``train_end_date`` would silently make the validation range impossible.
    The stage rejects that configuration before reading any artifact.

    Args:
        train_end_date: Inclusive last train date.
        test_start_date: Inclusive first test date.

    Raises:
        ValueError: ``test_start_date`` is not strictly after
            ``train_end_date``.
    """
    if test_start_date <= train_end_date:
        raise ValueError(
            "test_start_date must be strictly after train_end_date so the "
            "validation range is non-empty; got "
            f"train_end_date={train_end_date!r} and "
            f"test_start_date={test_start_date!r}"
        )


def join_outcome_and_features(
    feature_df: pl.DataFrame,
    outcome_df: pl.DataFrame,
    outcome_name: str,
) -> pl.DataFrame:
    """Join one outcome column to the feature matrix on the canonical keys.

    The join is an inner join on ``KEY_COLUMNS``. Both inputs must carry the
    same key set exactly once; a key that appears in only one table would be
    dropped by the inner join, so the stage checks the key sets first and then
    asserts that the joined table still holds every input row. The result is
    sorted on the canonical keys so the split and the written artifacts are
    deterministic regardless of the producers' row order.

    Args:
        feature_df: The feature table: canonical keys plus feature columns.
            Extra producer columns are ignored.
        outcome_df: The outcome table: canonical keys plus the registered
            outcome columns.
        outcome_name: The single outcome column this split directory carries.

    Returns:
        A frame with the canonical keys, the registered feature columns in
        ``FEATURE_COLUMNS`` order, and exactly one outcome column, sorted by
        the canonical keys.

    Raises:
        ValueError: A key column is missing, the outcome column is missing,
            the two tables do not carry the same key set, or the join did not
            conserve every row.
    """
    if outcome_name not in outcome_df.columns:
        raise ValueError(
            f"outcome {outcome_name!r} is not a column of the outcome table; "
            f"available columns: {outcome_df.columns!r}"
        )

    feature_keys = _require_key_columns(feature_df, "feature table").sort(
        list(KEY_COLUMNS)
    )
    outcome_keys = _require_key_columns(outcome_df, "outcome table").sort(
        list(KEY_COLUMNS)
    )
    if not feature_keys.equals(outcome_keys):
        feature_rows = set(feature_keys.iter_rows())
        outcome_rows = set(outcome_keys.iter_rows())
        missing = sorted(outcome_rows - feature_rows)
        unexpected = sorted(feature_rows - outcome_rows)
        raise ValueError(
            "feature and outcome tables do not carry the same keys: "
            f"{len(missing)} outcome key(s) are missing from the feature "
            f"table (first: {missing[:3]!r}) and {len(unexpected)} feature "
            f"key(s) are absent from the outcome table (first: "
            f"{unexpected[:3]!r}); the producers must emit one row per case"
        )

    feature_projection = feature_df.select([*KEY_COLUMNS, *FEATURE_COLUMNS])
    outcome_projection = outcome_df.select([*KEY_COLUMNS, outcome_name])
    joined_df = feature_projection.join(
        outcome_projection,
        on=list(KEY_COLUMNS),
        how="inner",
    ).sort(list(KEY_COLUMNS))

    if joined_df.height != feature_df.height or joined_df.height != outcome_df.height:
        raise ValueError(
            "the key join did not conserve rows: the feature table has "
            f"{feature_df.height} row(s), the outcome table has "
            f"{outcome_df.height}, but the joined table has "
            f"{joined_df.height}; every input row must appear exactly once"
        )
    if joined_df.select(list(KEY_COLUMNS)).is_duplicated().any():
        raise ValueError(
            "the joined table contains duplicate canonical keys; both "
            "producers must emit each case exactly once"
        )
    return joined_df


def _assert_partitions_partition_the_table(
    joined_df: pl.DataFrame,
    partitions: Mapping[str, pl.DataFrame],
) -> None:
    """Assert the partitions are pairwise disjoint and exhaustive.

    Disjointness is checked on the canonical key tuples, not on full rows, so
    two partitions that happen to share only some payload values are still
    rejected. Exhaustiveness is checked both by the summed row count and by
    the union of key tuples, which catches a partition that both drops a row
    and duplicates another.

    Raises:
        ValueError: The partitions overlap, drop a joined row, or cover fewer
            rows than the joined table.
    """
    joined_keys = set(_require_key_columns(joined_df, "joined table").iter_rows())
    seen: set[tuple[object, ...]] = set()
    total = 0
    for name in PARTITION_ORDER:
        keys = set(
            _require_key_columns(partitions[name], f"{name} partition").iter_rows()
        )
        overlap = seen & keys
        if overlap:
            examples = sorted(overlap)[:3]
            raise ValueError(
                f"the {name} partition overlaps an earlier partition on "
                f"{len(overlap)} key(s), first: {examples!r}; each case must "
                "belong to exactly one holdout partition"
            )
        seen |= keys
        total += partitions[name].height

    if total != joined_df.height:
        raise ValueError(
            "holdout partitions are not exhaustive: their row counts sum to "
            f"{total} but the joined table has {joined_df.height} row(s)"
        )
    if seen != joined_keys:
        missing = sorted(joined_keys - seen)
        unexpected = sorted(seen - joined_keys)
        raise ValueError(
            "holdout partitions do not cover the joined keys exactly: "
            f"{len(missing)} joined key(s) are missing (first: "
            f"{missing[:3]!r}) and {len(unexpected)} unknown key(s) are "
            f"present (first: {unexpected[:3]!r})"
        )


def split_by_case_date(
    joined_df: pl.DataFrame,
    train_end_date: date,
    test_start_date: date,
) -> dict[str, pl.DataFrame]:
    """Split a joined table into the canonical train/valid/test partitions.

    The predicates are mutually exclusive and ordered, so every joined row
    falls into exactly one partition. An empty partition is rejected: three
    non-empty holdout partitions are part of the split contract, and a
    configuration that produces an empty one cannot train, validate, or test
    the model as declared. All checks run before the caller writes anything.

    Args:
        joined_df: The key-joined table produced by
            :func:`join_outcome_and_features`.
        train_end_date: Inclusive last train date.
        test_start_date: Inclusive first test date.

    Returns:
        A mapping keyed ``'train'``, ``'valid'``, and ``'test'`` in
        ``PARTITION_ORDER`` order, each holding that partition's rows in the
        joined table's column order and key-sorted row order.

    Raises:
        ValueError: The boundaries are invalid, a partition is empty, or the
            partitions are not disjoint and exhaustive.
    """
    assert_date_boundaries(train_end_date, test_start_date)

    partitions = {
        "train": joined_df.filter(pl.col(CASE_DATE) <= train_end_date),
        "valid": joined_df.filter(
            (pl.col(CASE_DATE) > train_end_date) & (pl.col(CASE_DATE) < test_start_date)
        ),
        "test": joined_df.filter(pl.col(CASE_DATE) >= test_start_date),
    }
    for name in PARTITION_ORDER:
        if partitions[name].height == 0:
            raise ValueError(
                f"the {name} partition is empty for the configured "
                f"boundaries train_end_date={train_end_date!r} and "
                f"test_start_date={test_start_date!r}; every holdout "
                "partition must contain at least one case"
            )

    _assert_partitions_partition_the_table(joined_df, partitions)
    return partitions


def _format_number(value: float | None) -> str:
    """Render one statistics value with a fixed, deterministic width.

    A missing value (for example a standard deviation of a one-row partition)
    is rendered as ``n/a``. Finite values use six decimal places; the split
    schema rejects non-finite outcome values before statistics are written, so
    the non-finite branch is defensive rather than expected.
    """
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return repr(value)
    return f"{value:.6f}"


def _format_date(value: object) -> str:
    """Render a partition's boundary date, or ``n/a`` for an empty series."""
    return "n/a" if value is None else str(value)


def _statistics_header() -> str:
    """Return the aligned header row of the statistics table."""
    return (
        f"{'partition':<{_PARTITION_WIDTH}}"
        f"{_STATISTIC_HEADERS[0]:>{_ROW_WIDTH}}"
        f"{_STATISTIC_HEADERS[1]:>{_DATE_WIDTH}}"
        f"{_STATISTIC_HEADERS[2]:>{_DATE_WIDTH}}"
        + "".join(f"{name:>{_VALUE_WIDTH}}" for name in _STATISTIC_HEADERS[3:])
    )


def partition_statistics_text(
    outcome_name: str,
    partitions: Mapping[str, pl.DataFrame],
    train_end_date: date,
    test_start_date: date,
) -> str:
    """Return the deterministic statistics text for one outcome's split set.

    The report states the configuration boundaries, the total and per-partition
    row counts, and each partition's outcome mean, standard deviation,
    minimum, median, and maximum. Fixed column widths and six decimal places
    make the same split set produce the same bytes on a pinned ``polars``
    version.

    Args:
        outcome_name: The outcome column carried by every partition.
        partitions: The split mapping returned by :func:`split_by_case_date`.
        train_end_date: Inclusive last train date.
        test_start_date: Inclusive first test date.

    Returns:
        A newline-terminated text report.
    """
    total_rows = sum(partitions[name].height for name in PARTITION_ORDER)
    rows = [
        STATISTICS_TITLE,
        "=" * len(STATISTICS_TITLE),
        f"outcome: {outcome_name}",
        f"train_end_date: {train_end_date}",
        f"test_start_date: {test_start_date}",
        f"rows: {total_rows}",
        f"train_rows: {partitions['train'].height}",
        f"valid_rows: {partitions['valid'].height}",
        f"test_rows: {partitions['test'].height}",
        "",
        _statistics_header(),
    ]
    for name in PARTITION_ORDER:
        frame = partitions[name]
        series = frame[outcome_name]
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
            f"{name:<{_PARTITION_WIDTH}}{frame.height:>{_ROW_WIDTH}d}"
            f"{_format_date(frame[CASE_DATE].min()):>{_DATE_WIDTH}}"
            f"{_format_date(frame[CASE_DATE].max()):>{_DATE_WIDTH}}"
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


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    """Serialize one frame with explicit deterministic writer settings."""
    _write_atomically(
        path,
        lambda target: frame.write_parquet(
            target,
            compression=PARQUET_COMPRESSION,
            statistics=True,
        ),
    )


def _write_statistics(path: Path, text: str) -> None:
    """Serialize the statistics text as UTF-8 with ``\\n`` newlines."""
    _write_atomically(
        path,
        lambda target: target.write_text(text, encoding="utf-8"),
    )


def _read_feature_table(
    context: PipelineContext,
    output_root: Path,
) -> pl.DataFrame:
    """Read and validate the s02 feature artifact.

    Raises:
        FileNotFoundError: The feature stage has not produced its parquet.
        BoundaryError: The artifact violates ``S02_FEATURE_SCHEMA``.
    """
    feature_path = Path(output_root) / context.layout.artifact(
        FEATURE_STAGE,
        "feature_df",
    )
    return validate(
        pl.read_parquet(feature_path),
        S02_FEATURE_SCHEMA,
        FEATURE_BOUNDARY_LABEL,
    )


def _read_outcome_table(
    context: PipelineContext,
    output_root: Path,
) -> pl.DataFrame:
    """Read and validate the s01 outcome artifact.

    Raises:
        FileNotFoundError: The outcome stage has not produced its parquet.
        BoundaryError: The artifact violates ``S01_OUTCOME_SCHEMA``.
    """
    outcome_path = Path(output_root) / context.layout.artifact(
        OUTCOME_STAGE,
        "outcome_df",
    )
    return validate(
        pl.read_parquet(outcome_path),
        S01_OUTCOME_SCHEMA,
        OUTCOME_BOUNDARY_LABEL,
    )


def main(
    outcome_name: str,
    context: PipelineContext,
    output_root: Path,
) -> None:
    """Divide one outcome into the canonical holdout partitions.

    Reads the s01 outcome and s02 feature artifacts through the declarative
    layout, joins the modeled outcome column to the feature matrix on the
    canonical keys, splits the joined table on ``case_date``, validates every
    partition at the ``S03_SPLIT_SCHEMA`` boundary, and writes the declared
    head frames, partition parquets, and statistics text under
    ``output_root/s03_divide/{outcome_name}/``.

    The configuration boundaries are validated before any artifact is read,
    and every partition is validated before any artifact is written, so a
    failure leaves no partial split set.

    Args:
        outcome_name: The modeled outcome this split directory carries; it is
            also the partition's single outcome column.
        context: The frozen pipeline context carrying the layout and config.
        output_root: The explicit output root the relative layout paths are
            joined with.

    Raises:
        Failure propagation:
            ``FileNotFoundError`` when a prerequisite parquet is missing,
            ``BoundaryError`` when a prerequisite or partition violates its
            schema, and ``ValueError`` for an invalid boundary configuration,
            a key mismatch, a row-conservation failure, an empty partition, or
            a non-partitioning split.
    """
    layout = context.layout
    config = context.config
    assert_date_boundaries(config.train_end_date, config.test_start_date)

    stage_dir = Path(output_root) / layout.stage_dir(STAGE_NAME, outcome_name)
    context.ensure(stage_dir)

    feature_df = _read_feature_table(context, output_root)
    outcome_df = _read_outcome_table(context, output_root)
    joined_df = join_outcome_and_features(feature_df, outcome_df, outcome_name)

    partitions = split_by_case_date(
        joined_df,
        config.train_end_date,
        config.test_start_date,
    )
    for name in PARTITION_ORDER:
        validate(
            partitions[name],
            S03_SPLIT_SCHEMA,
            SPLIT_BOUNDARY_LABEL.format(partition=name),
        )

    feature_head_df = joined_df.select([*KEY_COLUMNS, *FEATURE_COLUMNS]).head(
        HEAD_ROW_COUNT
    )
    outcome_head_df = joined_df.select([*KEY_COLUMNS, outcome_name]).head(
        HEAD_ROW_COUNT
    )

    _write_parquet(
        Path(output_root)
        / layout.artifact(STAGE_NAME, "feature_head_df", outcome_name),
        feature_head_df,
    )
    _write_parquet(
        Path(output_root)
        / layout.artifact(STAGE_NAME, "outcome_head_df", outcome_name),
        outcome_head_df,
    )
    for name in PARTITION_ORDER:
        _write_parquet(
            Path(output_root) / layout.artifact(STAGE_NAME, f"{name}_df", outcome_name),
            partitions[name],
        )
    _write_statistics(
        Path(output_root) / layout.artifact(STAGE_NAME, "statistics", outcome_name),
        partition_statistics_text(
            outcome_name,
            partitions,
            config.train_end_date,
            config.test_start_date,
        ),
    )


s03_divide = main
"""Stage entry point under the name used by the registry executor."""

__all__ = [
    "FEATURE_BOUNDARY_LABEL",
    "HEAD_ROW_COUNT",
    "OUTCOME_BOUNDARY_LABEL",
    "SPLIT_BOUNDARY_LABEL",
    "assert_date_boundaries",
    "join_outcome_and_features",
    "main",
    "partition_statistics_text",
    "s03_divide",
    "split_by_case_date",
]
