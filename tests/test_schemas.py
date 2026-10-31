"""Unit tests for the project boundary schemas and the frame validator.

The schemas are the data contract every stage boundary shares, so the tests
pin two things. First, the validator's mechanics: presence, exact dtype,
nullability, per-column constraints, exclusive groups, key uniqueness, and row
counts are checked in that documented order, and a failure raises a
``BoundaryError`` that preserves the boundary label and names the failing
column and rule. Second, the declared constants: every ``S0x_*_SCHEMA`` names
the columns and dtypes the stages expect, and each table-level constraint from
the plan gets a failure test.

The tests build small, explicit frames rather than sharing a fixture so each
failure is traceable to the exact column and rule it targets.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from typing import Any

import polars as pl
import pytest

from src.project.columns import (
    CASE_DATE,
    CASE_ID,
    FEATURE_COLNAME,
    FEATURE_ESTIMATOR_ID_COLNAME,
    FEATURE_IMPORTANCE_COLNAME,
    KEY_COLUMNS,
)
from src.project.schemas import (
    FEATURE_COLUMNS,
    OUTCOME_COLUMNS,
    PREDICTION_COLNAME,
    S01_OUTCOME_SCHEMA,
    S02_FEATURE_SCHEMA,
    S03_SPLIT_SCHEMA,
    S04_FEATURE_IMPORTANCES_SCHEMA,
    S04_PARTITION_FEATURE_SCHEMA,
    S04_PARTITION_KEY_SCHEMA,
    S05_PREDICTION_SCHEMA,
    SOURCE_TABLE_SCHEMA,
    SOURCE_VALUE_COLUMNS,
    BoundaryError,
    ColumnRule,
    TableSchema,
    validate,
)

CASES = ("case_0000", "case_0001", "case_0002")
DATES = (date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3))


# --- Frame builders --------------------------------------------------------


def _source_rows() -> dict[str, list[Any]]:
    """Return a valid source-table frame as a mutable column mapping."""
    return {
        CASE_ID: list(CASES),
        CASE_DATE: list(DATES),
        "value_1": [0.1, 0.2, 0.3],
        "value_2": [-0.1, 0.0, 0.1],
        "value_3": [0.2, 0.2, 0.2],
    }


def _outcome_rows() -> dict[str, list[Any]]:
    """Return a valid outcome-table frame as a mutable column mapping."""
    rows = _source_rows()
    return {
        CASE_ID: rows[CASE_ID],
        CASE_DATE: rows[CASE_DATE],
        "outcome_1": [1.0, 2.0, 3.0],
        "outcome_2": [0.5, 1.5, 2.5],
        "outcome_3": [-1.0, 0.0, 1.0],
        "outcome_4": [0.0, 0.0, 0.0],
    }


def _feature_rows() -> dict[str, list[Any]]:
    """Return a valid feature-table frame as a mutable column mapping."""
    rows = _source_rows()
    result: dict[str, list[Any]] = {
        CASE_ID: rows[CASE_ID],
        CASE_DATE: rows[CASE_DATE],
    }
    for index, name in enumerate(FEATURE_COLUMNS):
        result[name] = [float(index + row) for row in range(len(CASES))]
    return result


def _frame(rows: dict[str, list[Any]]) -> pl.DataFrame:
    """Build a dataframe from a column mapping with inferred dtypes."""
    return pl.DataFrame(rows)


def _split_rows() -> dict[str, list[Any]]:
    """Return one split partition with exactly one outcome column."""
    rows = _feature_rows()
    rows["outcome_1"] = [1.0, 2.0, 3.0]
    return rows


def _prediction_rows() -> dict[str, list[Any]]:
    """Return a valid prediction frame."""
    rows = _source_rows()
    return {
        CASE_ID: rows[CASE_ID],
        CASE_DATE: rows[CASE_DATE],
        PREDICTION_COLNAME: [0.1, 0.2, 0.3],
    }


def _assert_boundary_error(
    caught: pytest.ExceptionInfo[BoundaryError],
    *,
    label: str,
    column: str | None,
    rule: str,
) -> None:
    """Assert the structured boundary facts and their message rendering."""
    error = caught.value
    assert isinstance(error, BoundaryError)
    assert error.label == label
    assert error.column == column
    assert error.rule == rule
    assert label in str(error)
    if column is None:
        assert "table" in str(error)
    else:
        assert repr(column) in str(error)


# --- ColumnRule construction ----------------------------------------------


def test_column_rule_defaults_are_strict() -> None:
    """A plain rule requires presence, rejects nulls, and sets no bounds."""
    rule = ColumnRule(dtype=pl.Float64)

    assert rule.dtype is pl.Float64
    assert rule.nullable is False
    assert rule.unique is False
    assert rule.min is None
    assert rule.max is None
    assert rule.finite is False
    assert rule.strictly_increasing is False
    assert rule.required is True


def test_column_rule_is_frozen() -> None:
    """A module-level schema's rules cannot be mutated after import."""
    rule = ColumnRule(dtype=pl.String)

    with pytest.raises(FrozenInstanceError):
        setattr(rule, "nullable", True)


def test_column_rule_rejects_inverted_bounds() -> None:
    """The inclusive bounds must be ordered."""
    with pytest.raises(ValueError, match="greater than max"):
        ColumnRule(dtype=pl.Float64, min=1.0, max=0.0)


def test_column_rule_accepts_equal_bounds() -> None:
    """Equal bounds describe a single allowed value and are valid."""
    rule = ColumnRule(dtype=pl.Float64, min=1.0, max=1.0)

    assert rule.min == rule.max == 1.0


def test_column_rule_rejects_finite_on_a_non_float_column() -> None:
    """Finiteness is only defined for floating-point dtypes."""
    with pytest.raises(ValueError, match="floating-point dtype"):
        ColumnRule(dtype=pl.String, finite=True)


def test_column_rule_rejects_strict_order_on_a_float_column() -> None:
    """A float column admits NaN, so strict order needs a non-float dtype."""
    with pytest.raises(ValueError, match="non-float"):
        ColumnRule(dtype=pl.Float64, strictly_increasing=True)


# --- TableSchema construction ---------------------------------------------


def test_table_schema_requires_at_least_one_column() -> None:
    """An empty schema cannot describe any frame."""
    with pytest.raises(ValueError, match="at least one column rule"):
        TableSchema(columns={})


def test_table_schema_rejects_a_negative_minimum_row_count() -> None:
    """A negative minimum row count is meaningless."""
    with pytest.raises(ValueError, match="min_rows must be non-negative"):
        TableSchema(columns={CASE_ID: ColumnRule(dtype=pl.String)}, min_rows=-1)


def test_table_schema_rejects_an_undeclared_key_column() -> None:
    """Every key column needs its own dtype and nullability rule."""
    with pytest.raises(ValueError, match="key column"):
        TableSchema(
            columns={CASE_ID: ColumnRule(dtype=pl.String)},
            key_columns=(CASE_DATE,),
        )


def test_table_schema_rejects_a_one_column_exclusive_group() -> None:
    """A one-member group would require and forbid the same column."""
    with pytest.raises(ValueError, match="at least two columns"):
        TableSchema(
            columns={CASE_ID: ColumnRule(dtype=pl.String, required=False)},
            exclusive_groups=((CASE_ID,),),
        )


def test_table_schema_rejects_an_undeclared_exclusive_group_column() -> None:
    """Every group member needs a declared rule."""
    with pytest.raises(ValueError, match="has no rule"):
        TableSchema(
            columns={CASE_ID: ColumnRule(dtype=pl.String, required=False)},
            exclusive_groups=((CASE_ID, "missing"),),
        )


def test_table_schema_rejects_a_column_in_two_exclusive_groups() -> None:
    """A column belongs to at most one one-of group."""
    with pytest.raises(ValueError, match="more than one exclusive group"):
        TableSchema(
            columns={
                CASE_ID: ColumnRule(dtype=pl.String, required=False),
                CASE_DATE: ColumnRule(dtype=pl.Date, required=False),
            },
            exclusive_groups=((CASE_ID, CASE_DATE), (CASE_ID, CASE_DATE)),
        )


def test_table_schema_rejects_a_required_exclusive_group_column() -> None:
    """A required member would make the one-of group unsatisfiable with two."""
    with pytest.raises(ValueError, match="must be optional"):
        TableSchema(
            columns={
                CASE_ID: ColumnRule(dtype=pl.String),
                CASE_DATE: ColumnRule(dtype=pl.Date, required=False),
            },
            exclusive_groups=((CASE_ID, CASE_DATE),),
        )


def test_table_schema_copies_columns_into_a_read_only_mapping() -> None:
    """Mutating the caller's mapping after construction changes no schema."""
    columns = {CASE_ID: ColumnRule(dtype=pl.String)}
    schema = TableSchema(columns=columns)

    columns[CASE_DATE] = ColumnRule(dtype=pl.Date)

    assert tuple(schema.columns) == (CASE_ID,)
    with pytest.raises(TypeError):
        schema.columns[CASE_DATE] = ColumnRule(dtype=pl.Date)  # type: ignore[index]


def test_table_schema_reports_required_and_optional_columns() -> None:
    """The two projections keep declaration order."""
    schema = TableSchema(
        columns={
            CASE_ID: ColumnRule(dtype=pl.String),
            CASE_DATE: ColumnRule(dtype=pl.Date, required=False),
        }
    )

    assert schema.required_columns == (CASE_ID,)
    assert schema.optional_columns == (CASE_DATE,)


# --- Validator mechanics ---------------------------------------------------


def test_validate_returns_the_same_frame() -> None:
    """Validation is a check, not a copy or a cast."""
    frame = _frame(_source_rows())

    validated = validate(frame, SOURCE_TABLE_SCHEMA, "source")

    assert validated is frame
    assert validated.equals(frame)


def test_validate_rejects_a_non_dataframe() -> None:
    """The validator only accepts a polars dataframe."""
    with pytest.raises(TypeError, match="polars DataFrame"):
        validate([], SOURCE_TABLE_SCHEMA, "source")  # type: ignore[arg-type]


def test_validate_reports_a_missing_required_column() -> None:
    """A required column is reported before any dtype check."""
    rows = _source_rows()
    del rows["value_2"]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), SOURCE_TABLE_SCHEMA, "source")

    _assert_boundary_error(caught, label="source", column="value_2", rule="presence")


def test_validate_allows_an_extra_column() -> None:
    """A wider frame is valid for a narrower schema."""
    rows = _source_rows()
    rows["extra"] = [1, 2, 3]

    frame = validate(_frame(rows), SOURCE_TABLE_SCHEMA, "source")

    assert "extra" in frame.columns


def test_validate_reports_a_dtype_mismatch() -> None:
    """Dtypes must match exactly; the validator never casts."""
    rows = _source_rows()
    rows["value_1"] = [1, 2, 3]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), SOURCE_TABLE_SCHEMA, "source")

    _assert_boundary_error(caught, label="source", column="value_1", rule="dtype")


def test_validate_reports_a_null_in_a_non_nullable_column() -> None:
    """A null in a non-nullable column is a producer bug, not data to fill."""
    rows = _source_rows()
    rows["value_1"] = [0.1, None, 0.3]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), SOURCE_TABLE_SCHEMA, "source")

    _assert_boundary_error(caught, label="source", column="value_1", rule="nullable")


def test_validate_allows_a_null_in_a_nullable_column() -> None:
    """A nullable rule opts into nulls while keeping the dtype check."""
    schema = TableSchema(columns={CASE_ID: ColumnRule(dtype=pl.String, nullable=True)})
    frame = pl.DataFrame({CASE_ID: ["case_0000", None]})

    validate(frame, schema, "optional-null")


def test_validate_skips_constraints_on_an_absent_optional_column() -> None:
    """An absent optional column is not a failure by itself."""
    schema = TableSchema(
        columns={
            CASE_ID: ColumnRule(dtype=pl.String),
            "optional": ColumnRule(dtype=pl.Float64, required=False, finite=True),
        }
    )

    validate(pl.DataFrame({CASE_ID: ["case_0000"]}), schema, "optional")


def test_validate_checks_an_optional_column_when_present() -> None:
    """Presence is optional; a present column still obeys its rule."""
    schema = TableSchema(
        columns={
            CASE_ID: ColumnRule(dtype=pl.String),
            "optional": ColumnRule(dtype=pl.Float64, required=False, finite=True),
        }
    )
    frame = pl.DataFrame({CASE_ID: ["case_0000"], "optional": [float("nan")]})

    with pytest.raises(BoundaryError) as caught:
        validate(frame, schema, "optional")

    _assert_boundary_error(caught, label="optional", column="optional", rule="finite")


def test_validate_reports_a_duplicate_in_a_unique_column() -> None:
    """The per-column unique rule rejects a repeated value."""
    schema = TableSchema(columns={"name": ColumnRule(dtype=pl.String, unique=True)})
    frame = pl.DataFrame({"name": ["a", "a"]})

    with pytest.raises(BoundaryError) as caught:
        validate(frame, schema, "unique")

    _assert_boundary_error(caught, label="unique", column="name", rule="unique")


def test_validate_reports_a_value_below_the_minimum() -> None:
    """The lower bound is inclusive and reported as the min rule."""
    schema = TableSchema(columns={"value": ColumnRule(dtype=pl.Float64, min=0.0)})
    frame = pl.DataFrame({"value": [0.0, -0.5]})

    with pytest.raises(BoundaryError) as caught:
        validate(frame, schema, "bounded")

    _assert_boundary_error(caught, label="bounded", column="value", rule="min")


def test_validate_reports_a_value_above_the_maximum() -> None:
    """The upper bound is inclusive and reported as the max rule."""
    schema = TableSchema(columns={"value": ColumnRule(dtype=pl.Float64, max=1.0)})
    frame = pl.DataFrame({"value": [1.0, 1.5]})

    with pytest.raises(BoundaryError) as caught:
        validate(frame, schema, "bounded")

    _assert_boundary_error(caught, label="bounded", column="value", rule="max")


def test_validate_accepts_the_exact_bounds() -> None:
    """Both bounds are inclusive, so the endpoints are valid."""
    schema = TableSchema(
        columns={"value": ColumnRule(dtype=pl.Float64, min=0.0, max=1.0)}
    )

    validate(pl.DataFrame({"value": [0.0, 0.5, 1.0]}), schema, "bounded")


def test_validate_reports_a_non_finite_value() -> None:
    """NaN is not finite, even though it is not null."""
    schema = TableSchema(columns={"value": ColumnRule(dtype=pl.Float64, finite=True)})
    frame = pl.DataFrame({"value": [0.0, float("nan")]})

    with pytest.raises(BoundaryError) as caught:
        validate(frame, schema, "finite")

    _assert_boundary_error(caught, label="finite", column="value", rule="finite")


def test_validate_reports_an_infinite_value() -> None:
    """Infinity is rejected by the same finiteness rule."""
    schema = TableSchema(columns={"value": ColumnRule(dtype=pl.Float64, finite=True)})
    frame = pl.DataFrame({"value": [0.0, float("inf")]})

    with pytest.raises(BoundaryError) as caught:
        validate(frame, schema, "finite")

    _assert_boundary_error(caught, label="finite", column="value", rule="finite")


def test_validate_reports_an_unsorted_strictly_increasing_column() -> None:
    """A decreasing pair breaks the strict order rule."""
    rows = _source_rows()
    rows[CASE_DATE] = [DATES[0], DATES[2], DATES[1]]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), SOURCE_TABLE_SCHEMA, "source")

    _assert_boundary_error(
        caught, label="source", column=CASE_DATE, rule="strictly_increasing"
    )


def test_validate_reports_a_repeated_date_in_an_ordered_column() -> None:
    """The ordered date column rejects a repeated value."""
    rows = _source_rows()
    rows[CASE_DATE] = [DATES[0], DATES[0], DATES[1]]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), SOURCE_TABLE_SCHEMA, "source")

    _assert_boundary_error(caught, label="source", column=CASE_DATE, rule="unique")


def test_validate_reports_a_duplicate_key_tuple() -> None:
    """The composite key is checked even when no single column is unique."""
    schema = TableSchema(
        columns={
            "left": ColumnRule(dtype=pl.String),
            "right": ColumnRule(dtype=pl.Date),
        },
        key_columns=("left", "right"),
    )
    frame = pl.DataFrame({"left": ["a", "a"], "right": [DATES[0], DATES[0]]})

    with pytest.raises(BoundaryError) as caught:
        validate(frame, schema, "keys")

    _assert_boundary_error(caught, label="keys", column="left,right", rule="unique")


def test_validate_reports_a_missing_key_column() -> None:
    """A declared key column that is absent is a presence failure."""
    schema = TableSchema(
        columns={
            "left": ColumnRule(dtype=pl.String, required=False),
            "right": ColumnRule(dtype=pl.Date, required=False),
        },
        key_columns=("left", "right"),
    )

    with pytest.raises(BoundaryError) as caught:
        validate(pl.DataFrame({"left": ["a"]}), schema, "keys")

    _assert_boundary_error(caught, label="keys", column="right", rule="presence")


def test_validate_reports_an_empty_frame() -> None:
    """The default minimum row count rejects an empty table."""
    frame = pl.DataFrame(
        {
            CASE_ID: pl.Series([], dtype=pl.String),
            CASE_DATE: pl.Series([], dtype=pl.Date),
            "value_1": pl.Series([], dtype=pl.Float64),
            "value_2": pl.Series([], dtype=pl.Float64),
            "value_3": pl.Series([], dtype=pl.Float64),
        }
    )

    with pytest.raises(BoundaryError) as caught:
        validate(frame, SOURCE_TABLE_SCHEMA, "source")

    _assert_boundary_error(caught, label="source", column=None, rule="row_count")


def test_validate_reports_a_preserved_row_count_mismatch() -> None:
    """A caller-supplied expected count is checked as the row_count rule."""
    frame = _frame(_source_rows())

    with pytest.raises(BoundaryError) as caught:
        validate(
            frame,
            SOURCE_TABLE_SCHEMA,
            "source",
            expected_row_count=len(CASES) + 1,
        )

    _assert_boundary_error(caught, label="source", column=None, rule="row_count")


def test_validate_accepts_a_matching_expected_row_count() -> None:
    """A preserved count with the right value passes."""
    frame = _frame(_source_rows())

    validate(
        frame,
        SOURCE_TABLE_SCHEMA,
        "source",
        expected_row_count=len(CASES),
    )


# --- Schema declaration contract ------------------------------------------


def test_feature_and_outcome_column_vocabulary_is_pinned() -> None:
    """The generated column tuples match the destination vocabulary."""
    assert FEATURE_COLUMNS == tuple(f"feature_{index}" for index in range(1, 9))
    assert SOURCE_VALUE_COLUMNS == ("value_1", "value_2", "value_3")
    assert OUTCOME_COLUMNS == ("outcome_1", "outcome_2", "outcome_3", "outcome_4")
    assert OUTCOME_COLUMNS == tuple(OUTCOME_COLUMNS)


def test_all_schema_constants_are_table_schemas() -> None:
    """Every public schema constant is a frozen table schema."""
    schemas = (
        SOURCE_TABLE_SCHEMA,
        S01_OUTCOME_SCHEMA,
        S02_FEATURE_SCHEMA,
        S03_SPLIT_SCHEMA,
        S04_PARTITION_KEY_SCHEMA,
        S04_PARTITION_FEATURE_SCHEMA,
        S04_FEATURE_IMPORTANCES_SCHEMA,
        S05_PREDICTION_SCHEMA,
    )

    for schema in schemas:
        assert isinstance(schema, TableSchema)
        assert schema.columns
        with pytest.raises(FrozenInstanceError):
            setattr(schema, "min_rows", 0)


def test_source_schema_declares_the_identity_and_value_columns() -> None:
    """The source contract is keys plus the three raw value columns."""
    assert tuple(SOURCE_TABLE_SCHEMA.columns) == (
        CASE_ID,
        CASE_DATE,
        *SOURCE_VALUE_COLUMNS,
    )
    assert SOURCE_TABLE_SCHEMA.key_columns == KEY_COLUMNS
    assert SOURCE_TABLE_SCHEMA.columns[CASE_ID].dtype is pl.String
    assert SOURCE_TABLE_SCHEMA.columns[CASE_DATE].dtype is pl.Date
    assert SOURCE_TABLE_SCHEMA.columns[CASE_ID].unique is True
    assert SOURCE_TABLE_SCHEMA.columns[CASE_DATE].strictly_increasing is True


def test_outcome_schema_declares_every_registered_outcome() -> None:
    """The outcome contract follows the registry column order."""
    assert tuple(S01_OUTCOME_SCHEMA.columns) == (
        CASE_ID,
        CASE_DATE,
        *OUTCOME_COLUMNS,
    )
    assert S01_OUTCOME_SCHEMA.key_columns == KEY_COLUMNS


def test_feature_schema_declares_the_full_feature_matrix() -> None:
    """The feature contract is keys plus every demo feature column."""
    assert tuple(S02_FEATURE_SCHEMA.columns) == (
        CASE_ID,
        CASE_DATE,
        *FEATURE_COLUMNS,
    )
    assert S02_FEATURE_SCHEMA.key_columns == KEY_COLUMNS


def test_split_schema_requires_exactly_one_registered_outcome() -> None:
    """The split contract is keys, features, and one outcome from the group."""
    assert S03_SPLIT_SCHEMA.exclusive_groups == (OUTCOME_COLUMNS,)
    assert S03_SPLIT_SCHEMA.optional_columns == OUTCOME_COLUMNS
    assert S03_SPLIT_SCHEMA.required_columns == (
        CASE_ID,
        CASE_DATE,
        *FEATURE_COLUMNS,
    )


def test_partition_key_schema_is_the_identity_pair_only() -> None:
    """The key frame carries no payload columns."""
    assert tuple(S04_PARTITION_KEY_SCHEMA.columns) == KEY_COLUMNS
    assert S04_PARTITION_KEY_SCHEMA.key_columns == KEY_COLUMNS


def test_partition_feature_schema_is_keys_plus_features() -> None:
    """The model matrix contract is keys plus the feature columns."""
    assert tuple(S04_PARTITION_FEATURE_SCHEMA.columns) == (
        CASE_ID,
        CASE_DATE,
        *FEATURE_COLUMNS,
    )
    assert S04_PARTITION_FEATURE_SCHEMA.key_columns == KEY_COLUMNS


def test_feature_importances_schema_declares_the_three_columns() -> None:
    """The importance contract names the feature, estimator, and value."""
    assert tuple(S04_FEATURE_IMPORTANCES_SCHEMA.columns) == (
        FEATURE_COLNAME,
        FEATURE_ESTIMATOR_ID_COLNAME,
        FEATURE_IMPORTANCE_COLNAME,
    )
    assert (
        S04_FEATURE_IMPORTANCES_SCHEMA.columns[FEATURE_IMPORTANCE_COLNAME].dtype
        is pl.Float64
    )


def test_prediction_schema_declares_the_prediction_column() -> None:
    """The prediction contract is keys plus the finite prediction value."""
    assert tuple(S05_PREDICTION_SCHEMA.columns) == (
        CASE_ID,
        CASE_DATE,
        PREDICTION_COLNAME,
    )
    assert S05_PREDICTION_SCHEMA.key_columns == KEY_COLUMNS
    assert S05_PREDICTION_SCHEMA.columns[PREDICTION_COLNAME].finite is True


# --- Source table contract -------------------------------------------------


def test_source_table_accepts_a_valid_frame() -> None:
    """The valid source frame passes every declared rule."""
    validate(_frame(_source_rows()), SOURCE_TABLE_SCHEMA, "source")


def test_source_table_preserves_a_matching_row_count() -> None:
    """The source table accepts its own row count as the preserved count."""
    frame = _frame(_source_rows())

    validate(
        frame,
        SOURCE_TABLE_SCHEMA,
        "source",
        expected_row_count=frame.height,
    )


def test_source_table_reports_a_null_case_id() -> None:
    """A missing identity value is never imputed at the boundary."""
    rows = _source_rows()
    rows[CASE_ID] = [CASES[0], None, CASES[2]]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), SOURCE_TABLE_SCHEMA, "source")

    _assert_boundary_error(caught, label="source", column=CASE_ID, rule="nullable")


def test_source_table_reports_a_duplicate_case_id() -> None:
    """A source case appears once."""
    rows = _source_rows()
    rows[CASE_ID] = [CASES[0], CASES[0], CASES[2]]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), SOURCE_TABLE_SCHEMA, "source")

    _assert_boundary_error(caught, label="source", column=CASE_ID, rule="unique")


# --- Outcome table contract ------------------------------------------------


def test_outcome_table_accepts_a_valid_frame() -> None:
    """The valid outcome frame passes every declared rule."""
    validate(_frame(_outcome_rows()), S01_OUTCOME_SCHEMA, "outcome")


def test_outcome_table_reports_a_missing_outcome_column() -> None:
    """Every registered outcome column is required."""
    rows = _outcome_rows()
    del rows["outcome_3"]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S01_OUTCOME_SCHEMA, "outcome")

    _assert_boundary_error(caught, label="outcome", column="outcome_3", rule="presence")


def test_outcome_table_reports_a_null_outcome() -> None:
    """An outcome value is never allowed to be null."""
    rows = _outcome_rows()
    rows["outcome_2"] = [0.5, None, 2.5]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S01_OUTCOME_SCHEMA, "outcome")

    _assert_boundary_error(caught, label="outcome", column="outcome_2", rule="nullable")


def test_outcome_table_reports_a_non_finite_outcome() -> None:
    """An infinite outcome would corrupt every metric downstream."""
    rows = _outcome_rows()
    rows["outcome_1"] = [1.0, float("inf"), 3.0]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S01_OUTCOME_SCHEMA, "outcome")

    _assert_boundary_error(caught, label="outcome", column="outcome_1", rule="finite")


def test_outcome_table_enforces_the_preserved_source_row_count() -> None:
    """The stage must pass the source count it promises to preserve."""
    frame = _frame(_outcome_rows())

    with pytest.raises(BoundaryError) as caught:
        validate(
            frame,
            S01_OUTCOME_SCHEMA,
            "outcome",
            expected_row_count=len(CASES) + 1,
        )

    _assert_boundary_error(caught, label="outcome", column=None, rule="row_count")


def test_outcome_table_reports_a_null_key() -> None:
    """The identity pair is always complete."""
    rows = _outcome_rows()
    rows[CASE_ID] = [CASES[0], None, CASES[2]]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S01_OUTCOME_SCHEMA, "outcome")

    _assert_boundary_error(caught, label="outcome", column=CASE_ID, rule="nullable")


def test_outcome_table_reports_duplicate_keys() -> None:
    """One outcome row exists per case."""
    rows = _outcome_rows()
    rows[CASE_ID] = [CASES[0], CASES[0], CASES[2]]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S01_OUTCOME_SCHEMA, "outcome")

    _assert_boundary_error(caught, label="outcome", column=CASE_ID, rule="unique")


# --- Feature table contract ------------------------------------------------


def test_feature_table_accepts_a_valid_frame() -> None:
    """The valid feature frame passes every declared rule."""
    validate(_frame(_feature_rows()), S02_FEATURE_SCHEMA, "feature")


def test_feature_table_reports_a_missing_feature_column() -> None:
    """Every feature column is required."""
    rows = _feature_rows()
    del rows["feature_8"]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S02_FEATURE_SCHEMA, "feature")

    _assert_boundary_error(caught, label="feature", column="feature_8", rule="presence")


def test_feature_table_reports_a_null_feature() -> None:
    """A feature value is never allowed to be null."""
    rows = _feature_rows()
    rows["feature_4"] = [0.0, None, 1.0]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S02_FEATURE_SCHEMA, "feature")

    _assert_boundary_error(caught, label="feature", column="feature_4", rule="nullable")


def test_feature_table_reports_a_non_float_feature_matrix() -> None:
    """The matrix must already be floating point; the boundary never casts."""
    rows = _feature_rows()
    rows["feature_1"] = [1, 2, 3]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S02_FEATURE_SCHEMA, "feature")

    _assert_boundary_error(caught, label="feature", column="feature_1", rule="dtype")


def test_feature_table_enforces_one_row_per_source_row() -> None:
    """The stage passes the source count as the preserved row count."""
    frame = _frame(_feature_rows()).head(len(CASES) - 1)

    with pytest.raises(BoundaryError) as caught:
        validate(
            frame,
            S02_FEATURE_SCHEMA,
            "feature",
            expected_row_count=len(CASES),
        )

    _assert_boundary_error(caught, label="feature", column=None, rule="row_count")


def test_feature_table_reports_a_null_key() -> None:
    """The feature table's identity pair is always complete."""
    rows = _feature_rows()
    rows[CASE_ID] = [CASES[0], None, CASES[2]]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S02_FEATURE_SCHEMA, "feature")

    _assert_boundary_error(caught, label="feature", column=CASE_ID, rule="nullable")


def test_feature_table_reports_duplicate_keys() -> None:
    """One feature row exists per case."""
    rows = _feature_rows()
    rows[CASE_ID] = [CASES[0], CASES[0], CASES[2]]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S02_FEATURE_SCHEMA, "feature")

    _assert_boundary_error(caught, label="feature", column=CASE_ID, rule="unique")


# --- Split partition contract ---------------------------------------------


def test_split_partition_accepts_exactly_one_outcome() -> None:
    """Keys, features, and one registered outcome pass."""
    validate(_frame(_split_rows()), S03_SPLIT_SCHEMA, "split")


def test_split_partition_accepts_each_registered_outcome() -> None:
    """The one present outcome may be any member of the registry."""
    for outcome_name in OUTCOME_COLUMNS:
        rows = _split_rows()
        rows.pop("outcome_1")
        rows[outcome_name] = [1.0, 2.0, 3.0]

        validate(_frame(rows), S03_SPLIT_SCHEMA, f"split:{outcome_name}")


def test_split_partition_reports_no_outcome_column() -> None:
    """A split frame with no outcome column violates the one-of group."""
    rows = _feature_rows()

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S03_SPLIT_SCHEMA, "split")

    _assert_boundary_error(caught, label="split", column=None, rule="exactly_one")


def test_split_partition_reports_two_outcome_columns() -> None:
    """A split frame duplicates no outcome across directories, so two fail."""
    rows = _split_rows()
    rows["outcome_2"] = [0.5, 1.5, 2.5]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S03_SPLIT_SCHEMA, "split")

    _assert_boundary_error(caught, label="split", column=None, rule="exactly_one")


def test_split_partition_reports_a_null_outcome() -> None:
    """The single present outcome must be non-null."""
    rows = _split_rows()
    rows["outcome_1"] = [1.0, None, 3.0]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S03_SPLIT_SCHEMA, "split")

    _assert_boundary_error(caught, label="split", column="outcome_1", rule="nullable")


def test_split_partition_reports_a_null_feature() -> None:
    """The split frame carries the full non-null feature matrix."""
    rows = _split_rows()
    rows["feature_2"] = [0.0, None, 2.0]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S03_SPLIT_SCHEMA, "split")

    _assert_boundary_error(caught, label="split", column="feature_2", rule="nullable")


def test_split_partition_reports_duplicate_keys() -> None:
    """A partition's identity pair must be unique within that partition."""
    rows = _split_rows()
    rows[CASE_ID] = [CASES[0], CASES[0], CASES[2]]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S03_SPLIT_SCHEMA, "split")

    _assert_boundary_error(caught, label="split", column=CASE_ID, rule="unique")


# --- S04 partition and importance contracts -------------------------------


def test_partition_key_frame_accepts_the_identity_pair() -> None:
    """A key frame with just the identity pair passes."""
    rows = _source_rows()
    frame = pl.DataFrame({CASE_ID: rows[CASE_ID], CASE_DATE: rows[CASE_DATE]})

    validate(frame, S04_PARTITION_KEY_SCHEMA, "key")


def test_partition_key_frame_reports_duplicate_keys() -> None:
    """The key frame's identity pair must be unique."""
    rows = _source_rows()
    rows[CASE_ID] = [CASES[0], CASES[0], CASES[2]]
    frame = pl.DataFrame({CASE_ID: rows[CASE_ID], CASE_DATE: rows[CASE_DATE]})

    with pytest.raises(BoundaryError) as caught:
        validate(frame, S04_PARTITION_KEY_SCHEMA, "key")

    _assert_boundary_error(caught, label="key", column=CASE_ID, rule="unique")


def test_partition_feature_frame_accepts_keys_and_features() -> None:
    """The model matrix passes without an outcome column."""
    validate(_frame(_feature_rows()), S04_PARTITION_FEATURE_SCHEMA, "matrix")


def test_partition_feature_frame_reports_a_null_feature() -> None:
    """The design matrix must contain no nulls."""
    rows = _feature_rows()
    rows["feature_7"] = [0.0, None, 1.0]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S04_PARTITION_FEATURE_SCHEMA, "matrix")

    _assert_boundary_error(caught, label="matrix", column="feature_7", rule="nullable")


def test_feature_importances_accepts_the_three_columns() -> None:
    """A feature/estimator/importance frame passes."""
    frame = pl.DataFrame(
        {
            FEATURE_COLNAME: ["feature_1", "feature_2"],
            FEATURE_ESTIMATOR_ID_COLNAME: ["xgboost", "xgboost"],
            FEATURE_IMPORTANCE_COLNAME: [0.75, 0.25],
        }
    )

    validate(frame, S04_FEATURE_IMPORTANCES_SCHEMA, "importance")


def test_feature_importances_reports_a_null_feature_name() -> None:
    """A feature name is the join key of the ranking and cannot be null."""
    frame = pl.DataFrame(
        {
            FEATURE_COLNAME: ["feature_1", None],
            FEATURE_ESTIMATOR_ID_COLNAME: ["xgboost", "xgboost"],
            FEATURE_IMPORTANCE_COLNAME: [0.75, 0.25],
        }
    )

    with pytest.raises(BoundaryError) as caught:
        validate(frame, S04_FEATURE_IMPORTANCES_SCHEMA, "importance")

    _assert_boundary_error(
        caught, label="importance", column=FEATURE_COLNAME, rule="nullable"
    )


def test_feature_importances_reports_a_non_finite_importance() -> None:
    """An infinite importance would break ranking and plotting."""
    frame = pl.DataFrame(
        {
            FEATURE_COLNAME: ["feature_1", "feature_2"],
            FEATURE_ESTIMATOR_ID_COLNAME: ["xgboost", "xgboost"],
            FEATURE_IMPORTANCE_COLNAME: [0.75, float("nan")],
        }
    )

    with pytest.raises(BoundaryError) as caught:
        validate(frame, S04_FEATURE_IMPORTANCES_SCHEMA, "importance")

    _assert_boundary_error(
        caught,
        label="importance",
        column=FEATURE_IMPORTANCE_COLNAME,
        rule="finite",
    )


# --- Prediction contract ---------------------------------------------------


def test_prediction_frame_accepts_finite_predictions() -> None:
    """Keys plus finite predictions pass."""
    validate(_frame(_prediction_rows()), S05_PREDICTION_SCHEMA, "prediction")


def test_prediction_frame_reports_a_missing_prediction_column() -> None:
    """The prediction value is required."""
    rows = _source_rows()
    frame = pl.DataFrame({CASE_ID: rows[CASE_ID], CASE_DATE: rows[CASE_DATE]})

    with pytest.raises(BoundaryError) as caught:
        validate(frame, S05_PREDICTION_SCHEMA, "prediction")

    _assert_boundary_error(
        caught, label="prediction", column=PREDICTION_COLNAME, rule="presence"
    )


def test_prediction_frame_reports_a_nan_prediction() -> None:
    """A NaN prediction is rejected by the finiteness rule."""
    rows = _prediction_rows()
    rows[PREDICTION_COLNAME] = [0.1, float("nan"), 0.3]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S05_PREDICTION_SCHEMA, "prediction")

    _assert_boundary_error(
        caught, label="prediction", column=PREDICTION_COLNAME, rule="finite"
    )


def test_prediction_frame_reports_duplicate_keys() -> None:
    """A prediction row exists once per case in its partition."""
    rows = _prediction_rows()
    rows[CASE_ID] = [CASES[0], CASES[0], CASES[2]]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S05_PREDICTION_SCHEMA, "prediction")

    _assert_boundary_error(caught, label="prediction", column=CASE_ID, rule="unique")


def test_prediction_frame_reports_a_null_key() -> None:
    """A prediction's identity pair is always complete."""
    rows = _prediction_rows()
    rows[CASE_ID] = [CASES[0], None, CASES[2]]

    with pytest.raises(BoundaryError) as caught:
        validate(_frame(rows), S05_PREDICTION_SCHEMA, "prediction")

    _assert_boundary_error(caught, label="prediction", column=CASE_ID, rule="nullable")
