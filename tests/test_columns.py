"""Unit tests for the canonical project column vocabulary.

Pins the identity contract the whole project layer relies on: the case
identity and case date names, their canonical ``(case_id, case_date)`` order,
and the derived feature/estimator/importance column names. A failure here
means boundary schemas, joins, and manifest writers could silently disagree
about a column name, so the assertions check exact values and uniqueness
rather than only "the module imports".

The vocabulary holds no data and no behavior, so the tests need no fixtures.
"""

from __future__ import annotations

import pytest

from src.project import columns


def test_key_column_constants_match_the_contract() -> None:
    """The two identity columns keep their exact public names."""
    assert columns.CASE_ID == "case_id"
    assert columns.CASE_DATE == "case_date"


def test_key_columns_is_canonical_order() -> None:
    """``KEY_COLUMNS`` is exactly the identity pair in join/sort order.

    The order is canonical: consumers join, sort, and assert alignment on the
    pair, so reversing the tuple would silently change artifact shapes.
    """
    assert columns.KEY_COLUMNS == ("case_id", "case_date")
    assert columns.KEY_COLUMNS == (columns.CASE_ID, columns.CASE_DATE)
    assert columns.KEY_COLUMNS != (columns.CASE_DATE, columns.CASE_ID)


def test_key_columns_is_an_immutable_tuple_of_distinct_names() -> None:
    """The canonical key set is immutable, non-empty, and duplicate-free."""
    assert isinstance(columns.KEY_COLUMNS, tuple)
    assert len(columns.KEY_COLUMNS) == 2
    assert len(set(columns.KEY_COLUMNS)) == len(columns.KEY_COLUMNS)
    assert all(columns.KEY_COLUMNS)


def test_derived_column_constants_match_the_contract() -> None:
    """The feature, estimator-id, and importance columns keep their names."""
    assert columns.FEATURE_COLNAME == "feature"
    assert columns.FEATURE_ESTIMATOR_ID_COLNAME == "feature_estimator_id"
    assert columns.FEATURE_IMPORTANCE_COLNAME == "importance"


@pytest.mark.parametrize(
    "name",
    [
        columns.CASE_ID,
        columns.CASE_DATE,
        columns.FEATURE_COLNAME,
        columns.FEATURE_ESTIMATOR_ID_COLNAME,
        columns.FEATURE_IMPORTANCE_COLNAME,
    ],
)
def test_column_names_are_lowercase_snake_case(name: str) -> None:
    """Every column name is a non-empty lowercase snake_case identifier.

    Stages build frames programmatically from these constants, so a stray
    space, uppercase letter, or empty string would produce a column that
    downstream schemas and joins cannot address consistently.
    """
    assert name
    assert name == name.strip()
    assert name == name.lower()
    assert name.replace("_", "").isalnum()
    assert not name.startswith("_")
    assert not name.endswith("_")
    assert "__" not in name


def test_derived_names_do_not_collide_with_keys_or_each_other() -> None:
    """Derived columns stay disjoint from the identity pair and one another."""
    derived = [
        columns.FEATURE_COLNAME,
        columns.FEATURE_ESTIMATOR_ID_COLNAME,
        columns.FEATURE_IMPORTANCE_COLNAME,
    ]
    assert set(derived).isdisjoint(columns.KEY_COLUMNS)
    assert len(set(derived)) == len(derived)


def test_key_columns_unpack_in_identity_order() -> None:
    """Unpacking the canonical tuple yields identity first and date second."""
    identity, date = columns.KEY_COLUMNS
    assert identity == columns.CASE_ID
    assert date == columns.CASE_DATE
