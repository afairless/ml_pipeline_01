"""Boundary schemas and the single frame validator for the project layer.

Every dataframe that crosses a stage boundary is described by a
:class:`TableSchema`: an ordered mapping of column name to :class:`ColumnRule`
plus the optional table-level rules ``key_columns``, ``min_rows``, and
``exclusive_groups``. The ``S0x_*_SCHEMA`` constants below are the data
contracts the stages, the external-input registry, and their tests share. A
stage names the schema it expects and calls :func:`validate`; nothing else
re-derives column names or dtypes from a filename.

:func:`validate` is deliberately strict and side-effect free. It checks, in
order, that required columns are present, that present columns have exactly the
declared dtype, that non-nullable columns contain no nulls, and then the
remaining constraints (uniqueness, numeric bounds, finiteness, strict
monotonicity, the one-of group, key uniqueness, and row counts). It never
casts a column and never fills a null: coercion and imputation are
transformation concerns that belong before the boundary, not at it. A
violation raises :class:`BoundaryError`, which preserves the caller's ``label``
together with the failing column and rule so a stage failure names both the
artifact it was reading and the contract it broke.

:func:`validate` accepts an optional ``expected_row_count`` keyword. Schemas
whose contract says "the row count is preserved from an earlier table" cannot
know that count by themselves, so the calling stage passes it and the check is
reported under the same ``row_count`` rule as ``min_rows``.

Payload columns are numeric in v1. A future project that needs a non-numeric
payload (for example a duration or categorical value) must extend
:class:`ColumnRule` or add a casting transformation before the boundary; the
elapsed-duration case is a documented extension point, not a hidden fallback.
Extra columns that a schema does not mention are allowed, so a caller may pass
a wider frame to a narrower schema. Columns listed with ``required=False`` are
optional; the only current use is the single outcome column of a split table,
which :class:`TableSchema` forces to be exactly one of the registered outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import polars as pl

from src.project.columns import (
    CASE_DATE,
    CASE_ID,
    FEATURE_COLNAME,
    FEATURE_ESTIMATOR_ID_COLNAME,
    FEATURE_IMPORTANCE_COLNAME,
    KEY_COLUMNS,
)
from src.project.outcomes import OUTCOMES

FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"feature_{index}" for index in range(1, 9)
)
"""The demo feature columns, in head order: ``feature_1`` through
``feature_8``."""

OUTCOME_COLUMNS: Final[tuple[str, ...]] = tuple(OUTCOMES)
"""Every registered outcome column, in registry (run) order."""

SOURCE_VALUE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"value_{index}" for index in range(1, 4)
)
"""The raw demo value columns: ``value_1`` through ``value_3``."""

PREDICTION_COLNAME: Final[str] = "predictions"
"""The prediction value column every prediction frame carries."""


class BoundaryError(ValueError):
    """A frame violated the schema declared for its stage boundary.

    The error keeps the structured facts a caller or test needs to diagnose the
    failure without parsing the message: the caller-supplied ``label``, the
    offending ``column`` (``None`` for a table-level rule), the rule name, and a
    human-readable ``detail``. It derives from :class:`ValueError` so existing
    callers that already handle invalid data keep working.

    Attributes:
        label: Name of the boundary, usually the artifact or table being read.
        column: Failing column name, or ``None`` when the rule is table-level.
        rule: Short rule name such as ``presence``, ``dtype``, ``nullable``,
            ``unique``, ``finite``, ``strictly_increasing``, ``exactly_one``,
            or ``row_count``.
        detail: Explanation of what the frame contained and what was required.
    """

    def __init__(
        self,
        label: str,
        column: str | None,
        rule: str,
        detail: str,
    ) -> None:
        self.label = label
        self.column = column
        self.rule = rule
        self.detail = detail
        location = f"column {column!r}" if column is not None else "table"
        super().__init__(f"{label}: {location} failed the {rule} rule: {detail}")


@dataclass(frozen=True, slots=True)
class ColumnRule:
    """The contract for one column inside a :class:`TableSchema`.

    Attributes:
        dtype: The exact ``polars`` dtype the column must have. A rule never
            accepts a compatible-but-different dtype; the producer must cast
            before the boundary.
        nullable: Whether nulls are allowed. Defaults to ``False`` because a
            missing identity or payload value is almost always a producer bug.
        unique: Whether every value in the column must be distinct. Nulls are
            checked first, so a non-nullable unique column has no null
            ambiguity.
        min: Inclusive lower bound for a numeric column, or ``None`` for no
            lower bound.
        max: Inclusive upper bound for a numeric column, or ``None`` for no
            upper bound.
        finite: Whether every value must be finite (no NaN or infinity). Only
            valid for floating-point columns.
        strictly_increasing: Whether the values must be sorted in strictly
            increasing order. Only meaningful for an ordered column such as
            ``case_date``; uniqueness is implied by strict monotonicity.
        required: Whether the column must be present. An optional column is
            still dtype-, null-, and constraint-checked when a frame carries
            it.

    Raises:
        ValueError: ``min`` exceeds ``max``, ``finite`` is set on a non-float
            column, or ``strictly_increasing`` is set on a float column that
            would admit NaN.
    """

    dtype: type[pl.DataType]
    nullable: bool = False
    unique: bool = False
    min: float | None = None
    max: float | None = None
    finite: bool = False
    strictly_increasing: bool = False
    required: bool = True

    def __post_init__(self) -> None:
        """Reject a rule whose own settings contradict each other."""
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f"column rule has min {self.min!r} greater than max {self.max!r}; "
                "the bounds are inclusive and must be ordered"
            )
        if self.finite and self.dtype not in (pl.Float32, pl.Float64):
            raise ValueError(
                f"the finite rule needs a floating-point dtype; got {self.dtype}"
            )
        if self.strictly_increasing and self.dtype in (pl.Float32, pl.Float64):
            raise ValueError(
                "the strictly_increasing rule is for ordered non-float columns; "
                "a float column admits NaN and must be ordered by its producer"
            )


@dataclass(frozen=True, slots=True)
class TableSchema:
    """A named collection of column rules plus table-level constraints.

    Attributes:
        columns: Ordered column name -> :class:`ColumnRule` mapping. The mapping
            is copied into a read-only proxy at construction, so a caller
            cannot mutate a module-level schema after import.
        key_columns: Columns whose combined tuple must be unique. An empty
            tuple means the table has no declared identity contract.
        min_rows: Minimum row count. Defaults to ``1``: an empty table is
            almost always a failed read rather than a valid empty result.
        exclusive_groups: Groups of column names where exactly one member must
            be present. Used by a split table, which carries one outcome column
            chosen from the registry.

    Raises:
        ValueError: ``columns`` is empty, a key or group column is not
            declared, a group is smaller than two columns, a column appears in
            two groups, ``min_rows`` is negative, or a column is required and
            also part of an exclusive group.
    """

    columns: Mapping[str, ColumnRule]
    key_columns: tuple[str, ...] = ()
    min_rows: int = 1
    exclusive_groups: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze the schema definition."""
        if not self.columns:
            raise ValueError("a table schema needs at least one column rule")
        if self.min_rows < 0:
            raise ValueError(f"min_rows must be non-negative; got {self.min_rows}")
        declared = set(self.columns)
        for key in self.key_columns:
            if key not in declared:
                raise ValueError(f"key column {key!r} has no rule in this table schema")
        seen_group_columns: set[str] = set()
        for group in self.exclusive_groups:
            if len(group) < 2:
                raise ValueError(
                    f"exclusive group {group!r} needs at least two columns"
                )
            for name in group:
                if name not in declared:
                    raise ValueError(
                        f"exclusive group column {name!r} has no rule in this "
                        "table schema"
                    )
                if name in seen_group_columns:
                    raise ValueError(
                        f"column {name!r} appears in more than one exclusive group"
                    )
                seen_group_columns.add(name)
                if self.columns[name].required:
                    raise ValueError(
                        f"exclusive group column {name!r} must be optional; "
                        "exactly one group member is present per frame"
                    )
        object.__setattr__(self, "columns", MappingProxyType(dict(self.columns)))

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return every column the schema requires, in declaration order."""
        return tuple(name for name, rule in self.columns.items() if rule.required)

    @property
    def optional_columns(self) -> tuple[str, ...]:
        """Return every optional column, in declaration order."""
        return tuple(name for name, rule in self.columns.items() if not rule.required)


def _fail(
    label: str,
    column: str | None,
    rule: str,
    detail: str,
) -> BoundaryError:
    """Build a :class:`BoundaryError`; keeps call sites short and uniform."""
    return BoundaryError(label=label, column=column, rule=rule, detail=detail)


def _check_presence(
    frame: pl.DataFrame,
    schema: TableSchema,
    label: str,
) -> None:
    """Check that every required column is present, in declaration order."""
    present = set(frame.columns)
    for name in schema.required_columns:
        if name not in present:
            raise _fail(
                label,
                name,
                "presence",
                f"required column is missing; frame has {frame.columns!r}",
            )


def _check_dtypes(
    frame: pl.DataFrame,
    schema: TableSchema,
    label: str,
) -> None:
    """Check the exact dtype of every present, declared column."""
    present = set(frame.columns)
    for name, rule in schema.columns.items():
        if name not in present:
            continue
        actual = frame.schema[name]
        if actual != rule.dtype:
            raise _fail(
                label,
                name,
                "dtype",
                f"expected {rule.dtype} but found {actual}",
            )


def _check_nullability(
    frame: pl.DataFrame,
    schema: TableSchema,
    label: str,
) -> None:
    """Check that every present non-nullable column has no nulls."""
    present = set(frame.columns)
    for name, rule in schema.columns.items():
        if name not in present or rule.nullable:
            continue
        null_count = frame[name].null_count()
        if null_count:
            raise _fail(
                label,
                name,
                "nullable",
                f"column is not nullable but has {null_count} null value(s)",
            )


def _check_column_constraints(
    frame: pl.DataFrame,
    schema: TableSchema,
    label: str,
) -> None:
    """Check uniqueness, bounds, finiteness, and monotonicity per column."""
    present = set(frame.columns)
    for name, rule in schema.columns.items():
        if name not in present:
            continue
        series = frame[name]
        if rule.unique and series.is_duplicated().any():
            raise _fail(
                label,
                name,
                "unique",
                "column values must be distinct but contain duplicates",
            )
        if rule.min is not None and not bool((series >= rule.min).all()):
            raise _fail(
                label,
                name,
                "min",
                f"column values must be >= {rule.min!r}",
            )
        if rule.max is not None and not bool((series <= rule.max).all()):
            raise _fail(
                label,
                name,
                "max",
                f"column values must be <= {rule.max!r}",
            )
        if rule.finite and not bool(series.is_finite().all()):
            raise _fail(
                label,
                name,
                "finite",
                "column values must be finite (no NaN or infinity)",
            )
        if rule.strictly_increasing:
            if series.n_unique() != series.len() or not series.is_sorted():
                raise _fail(
                    label,
                    name,
                    "strictly_increasing",
                    "column values must be sorted in strictly increasing order",
                )


def _check_exclusive_groups(
    frame: pl.DataFrame,
    schema: TableSchema,
    label: str,
) -> None:
    """Check that exactly one member of each declared group is present."""
    present = set(frame.columns)
    for group in schema.exclusive_groups:
        found = [name for name in group if name in present]
        if len(found) != 1:
            raise _fail(
                label,
                None,
                "exactly_one",
                f"expected exactly one of {group!r} but found {found!r}",
            )


def _check_keys(
    frame: pl.DataFrame,
    schema: TableSchema,
    label: str,
) -> None:
    """Check that the declared key tuple is unique across the frame."""
    if not schema.key_columns:
        return
    present = set(frame.columns)
    for name in schema.key_columns:
        if name not in present:
            raise _fail(
                label,
                name,
                "presence",
                f"key column is missing; frame has {frame.columns!r}",
            )
    key_columns = list(schema.key_columns)
    if frame.select(key_columns).is_duplicated().any():
        raise _fail(
            label,
            ",".join(schema.key_columns),
            "unique",
            f"the key tuple {schema.key_columns!r} must be unique",
        )


def _check_row_count(
    frame: pl.DataFrame,
    schema: TableSchema,
    label: str,
    expected_row_count: int | None,
) -> None:
    """Check the minimum and, when supplied, the preserved row count."""
    if frame.height < schema.min_rows:
        raise _fail(
            label,
            None,
            "row_count",
            f"frame has {frame.height} row(s) but at least {schema.min_rows} "
            "are required",
        )
    if expected_row_count is not None and frame.height != expected_row_count:
        raise _fail(
            label,
            None,
            "row_count",
            f"frame has {frame.height} row(s) but the expected row count is "
            f"{expected_row_count}",
        )


def validate(
    frame: pl.DataFrame,
    schema: TableSchema,
    label: str,
    *,
    expected_row_count: int | None = None,
) -> pl.DataFrame:
    """Validate ``frame`` against ``schema`` and return the frame unchanged.

    The check order matches the documented contract: required-column presence,
    exact dtype, nullability, per-column constraints, exclusive groups, key
    uniqueness, and row counts. The function never mutates or casts the input,
    so a caller can safely keep using the returned frame.

    Args:
        frame: The dataframe crossing the boundary.
        schema: The :class:`TableSchema` that boundary declares.
        label: Human-readable boundary name, usually the artifact path or table
            logical name. It is preserved on the raised error.
        expected_row_count: Optional row count the frame must match exactly,
            used where the contract is "one row per source row" or "row count
            preserved from the source table".

    Returns:
        ``frame`` itself, so validation can be chained into a stage call.

    Raises:
        TypeError: ``frame`` is not a ``polars`` dataframe.
        BoundaryError: ``frame`` violates ``schema``; the error carries the
            label, failing column (or ``None``), rule, and detail.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(
            f"schema validation needs a polars DataFrame; got {type(frame).__name__}"
        )
    _check_presence(frame, schema, label)
    _check_dtypes(frame, schema, label)
    _check_nullability(frame, schema, label)
    _check_column_constraints(frame, schema, label)
    _check_exclusive_groups(frame, schema, label)
    _check_keys(frame, schema, label)
    _check_row_count(frame, schema, label, expected_row_count)
    return frame


def _numeric_rule(**overrides: object) -> ColumnRule:
    """Build the common non-null, finite ``Float64`` payload rule.

    Every numeric payload column in this module shares the same shape; keeping
    it in one helper prevents a schema from quietly dropping the finiteness
    check.
    """
    settings: dict[str, object] = {
        "dtype": pl.Float64,
        "nullable": False,
        "finite": True,
    }
    settings.update(overrides)
    return ColumnRule(**settings)  # type: ignore[arg-type]


# --- Table schemas ---------------------------------------------------------

SOURCE_TABLE_SCHEMA: Final[TableSchema] = TableSchema(
    columns={
        CASE_ID: ColumnRule(dtype=pl.String, unique=True),
        CASE_DATE: ColumnRule(dtype=pl.Date, unique=True, strictly_increasing=True),
        "value_1": _numeric_rule(),
        "value_2": _numeric_rule(),
        "value_3": _numeric_rule(),
    },
    key_columns=KEY_COLUMNS,
    min_rows=1,
)
"""The external demo source table: identity, date, and three raw values.

``case_id`` is unique and non-null, ``case_date`` is non-null and strictly
increasing (the split predicates rely on that order), and every raw value is a
finite non-null float. The table must contain at least one row.
"""

S01_OUTCOME_SCHEMA: Final[TableSchema] = TableSchema(
    columns={
        CASE_ID: ColumnRule(dtype=pl.String, unique=True),
        CASE_DATE: ColumnRule(dtype=pl.Date, unique=True),
        **{name: _numeric_rule() for name in OUTCOME_COLUMNS},
    },
    key_columns=KEY_COLUMNS,
    min_rows=1,
)
"""The outcome table: the identity pair plus every registered outcome.

Callers that require the outcome stage to preserve the source row count pass
``expected_row_count=source_df.height`` to :func:`validate`.
"""

S02_FEATURE_SCHEMA: Final[TableSchema] = TableSchema(
    columns={
        CASE_ID: ColumnRule(dtype=pl.String, unique=True),
        CASE_DATE: ColumnRule(dtype=pl.Date, unique=True),
        **{name: _numeric_rule() for name in FEATURE_COLUMNS},
    },
    key_columns=KEY_COLUMNS,
    min_rows=1,
)
"""The feature table: the identity pair plus ``feature_1..feature_8``.

Callers that require one feature row per source row pass
``expected_row_count=source_df.height`` to :func:`validate`.
"""

S03_SPLIT_SCHEMA: Final[TableSchema] = TableSchema(
    columns={
        CASE_ID: ColumnRule(dtype=pl.String, unique=True),
        CASE_DATE: ColumnRule(dtype=pl.Date, unique=True),
        **{name: _numeric_rule() for name in FEATURE_COLUMNS},
        **{name: _numeric_rule(required=False) for name in OUTCOME_COLUMNS},
    },
    key_columns=KEY_COLUMNS,
    min_rows=1,
    exclusive_groups=(OUTCOME_COLUMNS,),
)
"""One partition of a split table: keys, features, and exactly one outcome.

A split directory is per outcome, so the frame carries the keys, the full
feature matrix, and that directory's single outcome column. The outcome member
is validated as a non-null finite float even though it is optional at the
schema level; the exclusive group requires exactly one registered outcome to
be present. Subset-of-joined-keys and partition non-emptiness are table
relationships the caller checks against the joined table; this schema checks
everything that is true of one partition frame on its own.
"""

S04_PARTITION_KEY_SCHEMA: Final[TableSchema] = TableSchema(
    columns={
        CASE_ID: ColumnRule(dtype=pl.String, unique=True),
        CASE_DATE: ColumnRule(dtype=pl.Date, unique=True),
    },
    key_columns=KEY_COLUMNS,
    min_rows=1,
)
"""The key frame of one partition: only the identity pair."""

S04_PARTITION_FEATURE_SCHEMA: Final[TableSchema] = TableSchema(
    columns={
        CASE_ID: ColumnRule(dtype=pl.String, unique=True),
        CASE_DATE: ColumnRule(dtype=pl.Date, unique=True),
        **{name: _numeric_rule() for name in FEATURE_COLUMNS},
    },
    key_columns=KEY_COLUMNS,
    min_rows=1,
)
"""The model feature matrix of one partition: keys plus the feature columns.

Every feature column is non-null and finite, so a trainer never receives a
NaN or an infinity in the design matrix. Extra columns, such as the partition's
outcome, are allowed.
"""

S04_FEATURE_IMPORTANCES_SCHEMA: Final[TableSchema] = TableSchema(
    columns={
        FEATURE_COLNAME: ColumnRule(dtype=pl.String),
        FEATURE_ESTIMATOR_ID_COLNAME: ColumnRule(dtype=pl.String),
        FEATURE_IMPORTANCE_COLNAME: _numeric_rule(),
    },
    min_rows=1,
)
"""The extracted feature-importance table written by the training stage.

One row per feature and estimator combination: the feature name, the estimator
identifier, and the finite importance value.
"""

S05_PREDICTION_SCHEMA: Final[TableSchema] = TableSchema(
    columns={
        CASE_ID: ColumnRule(dtype=pl.String, unique=True),
        CASE_DATE: ColumnRule(dtype=pl.Date, unique=True),
        PREDICTION_COLNAME: _numeric_rule(),
    },
    key_columns=KEY_COLUMNS,
    min_rows=1,
)
"""The prediction table for one model run and one partition.

The identity pair is unique within the partition, and every prediction is a
finite float: a NaN or infinity here would silently corrupt every metric
downstream, so it is rejected at the boundary instead.
"""
