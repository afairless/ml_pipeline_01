"""Unit and property-based tests for the project outcome registry.

Pins the vocabulary and behavior the stages rely on: ``OUTCOMES`` insertion
order is the pipeline run order, every registry key equals its ``Outcome.name``
(which is the outcome-table column name), the modeled filter selects the
declared subset, and the comparison tuple preserves the declared left/right
pairs.

The transform resolver is covered by property tests rather than example-only
checks: the ``identity`` pair round-trips every non-NaN float, the
``log1p``/``expm1`` pair round-trips inside its domain, and every value at or
below the domain boundary is rejected with a boundary ``ValueError``. Unknown
transform ids are rejected for arbitrary text.

A failure here means an outcome could be ordered, transformed, or compared
inconsistently between the registry, the stage table, and the analysis stage.
"""

from __future__ import annotations

import operator
from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.project.outcomes import (
    OUTCOME_COMPARISONS,
    OUTCOMES,
    Outcome,
    OutcomeComparison,
    TransformPair,
    _validate_comparisons,
    modeled_outcomes,
    outcome_names,
    resolve_transform,
)

EXPECTED_OUTCOME_NAMES = ("outcome_1", "outcome_2", "outcome_3", "outcome_4")
EXPECTED_COMPARISONS = (
    ("outcome_1", "outcome_2"),
    ("outcome_3", "outcome_4"),
)

# Values strictly greater than -1, the domain of the log1p transform. Keeping
# the lower bound away from -1 itself avoids feeding the round-trip property a
# value where the forward transform is not defined.
LOG1P_DOMAIN_VALUES = st.floats(
    min_value=-0.999999,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
)


# --- None: construction rejects degenerate outcomes ------------------------


def test_outcome_rejects_an_empty_name() -> None:
    """An outcome name is a column name and a path placeholder; empty is
    never valid."""
    with pytest.raises(ValueError, match="non-empty"):
        Outcome(name="", modeled=True)


def test_outcome_rejects_an_unknown_model_transform() -> None:
    """A model transform id must resolve before the outcome is registered."""
    with pytest.raises(ValueError, match="unknown transform"):
        Outcome(name="outcome_1", outcome_transform="not_registered", modeled=True)


def test_outcome_rejects_an_unknown_reporting_transform() -> None:
    """A reporting transform id must resolve before the outcome is used."""
    with pytest.raises(ValueError, match="unknown transform"):
        Outcome(
            name="outcome_1",
            reporting_backtransform="not_registered",
            modeled=True,
        )


# --- One: a single Outcome ------------------------------------------------


def test_outcome_stores_the_declared_fields_with_identity_defaults() -> None:
    """The transform fields default to the no-op pair and ``modeled`` is
    carried exactly as declared."""
    outcome = Outcome(name="outcome_1", modeled=True)
    assert outcome.name == "outcome_1"
    assert outcome.outcome_transform == "identity"
    assert outcome.reporting_backtransform == "identity"
    assert outcome.modeled is True


def test_outcome_accepts_registered_non_identity_transforms() -> None:
    """A registered pair is accepted on both transform axes."""
    outcome = Outcome(
        name="outcome_1",
        outcome_transform="log1p",
        reporting_backtransform="log1p",
        modeled=False,
    )
    assert outcome.outcome_transform == "log1p"
    assert outcome.reporting_backtransform == "log1p"
    assert outcome.modeled is False


def test_outcome_field_order_is_the_documented_order() -> None:
    """Field order is part of the public shape: name, model transform,
    reporting transform, then the modeled flag."""
    assert tuple(field.name for field in fields(Outcome)) == (
        "name",
        "outcome_transform",
        "reporting_backtransform",
        "modeled",
    )


def test_outcome_accepts_positional_construction() -> None:
    """The documented four-field constructor order is usable directly."""
    outcome = Outcome("outcome_1", "identity", "identity", True)
    assert outcome == Outcome(name="outcome_1", modeled=True)


def test_outcome_is_frozen() -> None:
    """Outcomes are shared module state and cannot be edited in place."""
    outcome = Outcome(name="outcome_1", modeled=True)
    with pytest.raises(FrozenInstanceError):
        setattr(outcome, "name", "outcome_2")


# --- Many: the OUTCOMES registry -----------------------------------------


def test_registry_keys_equal_the_outcome_names() -> None:
    """The registry key is the outcome column name, never a separate label."""
    for name, outcome in OUTCOMES.items():
        assert outcome.name == name


def test_registry_is_read_only() -> None:
    """Mutating the registry raises instead of silently changing the
    vocabulary every stage resolves against."""
    outcome = Outcome(name="outcome_5", modeled=False)
    mutable = cast(Any, OUTCOMES)
    with pytest.raises(TypeError):
        operator.setitem(mutable, "outcome_5", outcome)


def test_outcome_names_pin_registry_order() -> None:
    """Insertion order is the pipeline run order and is pinned exactly."""
    assert outcome_names() == EXPECTED_OUTCOME_NAMES


def test_outcome_names_are_unique() -> None:
    """Each outcome name addresses exactly one outcome column."""
    names = outcome_names()
    assert len(names) == len(set(names))


def test_modeled_outcomes_pin_the_declared_subset_in_order() -> None:
    """The modeled set is outcome_1 and outcome_3, in registry order."""
    assert tuple(outcome.name for outcome in modeled_outcomes()) == (
        "outcome_1",
        "outcome_3",
    )


def test_modeled_outcomes_return_the_registry_values() -> None:
    """The filter returns frozen registry entries, not copies."""
    assert modeled_outcomes() == (OUTCOMES["outcome_1"], OUTCOMES["outcome_3"])


def test_every_registry_outcome_resolves_its_transform_ids() -> None:
    """Every registered outcome names two registered transforms."""
    for outcome in OUTCOMES.values():
        resolve_transform(outcome.outcome_transform)
        resolve_transform(outcome.reporting_backtransform)


def test_demo_outcomes_use_identity_on_both_scales() -> None:
    """The demo outcomes are all untransformed on the model and reporting
    scales; the non-identity pair is exercised by the tests only."""
    assert {outcome.outcome_transform for outcome in OUTCOMES.values()} == {"identity"}
    assert {outcome.reporting_backtransform for outcome in OUTCOMES.values()} == {
        "identity"
    }


# --- Transform resolution -------------------------------------------------


def test_resolve_transform_returns_a_transform_pair() -> None:
    """The resolver returns the frozen pair type for a known id."""
    assert isinstance(resolve_transform("identity"), TransformPair)


def test_unknown_transform_id_is_rejected() -> None:
    """A misspelled transform id must fail at resolution time."""
    with pytest.raises(ValueError, match="unknown transform"):
        resolve_transform("not_a_transform")


@given(st.text().filter(lambda value: value not in {"identity", "log1p"}))
def test_arbitrary_text_is_not_a_registered_transform(transform_id: str) -> None:
    """Only the two registered ids resolve; arbitrary text raises."""
    with pytest.raises(ValueError, match="unknown transform"):
        resolve_transform(transform_id)


@given(st.floats(allow_nan=False))
def test_identity_transform_round_trips_every_float(value: float) -> None:
    """The identity pair returns its input on both directions, including
    infinities."""
    pair = resolve_transform("identity")
    assert pair.forward(value) == value
    assert pair.inverse(value) == value


@given(LOG1P_DOMAIN_VALUES)
def test_log1p_forward_inverse_round_trip(value: float) -> None:
    """Inside the domain, applying log1p then expm1 recovers the input."""
    pair = resolve_transform("log1p")
    assert pair.inverse(pair.forward(value)) == pytest.approx(value, rel=1e-7, abs=1e-9)


@given(
    st.floats(
        min_value=-20.0,
        max_value=20.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_log1p_inverse_forward_round_trip(value: float) -> None:
    """The inverse direction is also a round trip because expm1 maps every
    real value into the log1p domain."""
    pair = resolve_transform("log1p")
    assert pair.forward(pair.inverse(value)) == pytest.approx(value, rel=1e-7, abs=1e-9)


@given(
    st.floats(
        max_value=-1.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_log1p_rejects_values_at_or_below_the_domain_boundary(
    value: float,
) -> None:
    """The forward transform raises a boundary error for value <= -1."""
    with pytest.raises(ValueError, match="domain"):
        resolve_transform("log1p").forward(value)


def test_log1p_rejects_nan() -> None:
    """NaN is outside the domain and must not pass through silently."""
    with pytest.raises(ValueError, match="domain"):
        resolve_transform("log1p").forward(float("nan"))


# --- None: construction rejects degenerate comparisons --------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("not_an_outcome", "outcome_2"),
        ("outcome_1", "not_an_outcome"),
    ],
)
def test_comparison_rejects_an_unknown_outcome(left: str, right: str) -> None:
    """Both sides must be registered outcome names."""
    with pytest.raises(ValueError, match="unknown outcome"):
        OutcomeComparison(left=left, right=right)


def test_comparison_rejects_comparing_an_outcome_with_itself() -> None:
    """A self-difference is always zero and is not a comparison."""
    with pytest.raises(ValueError, match="itself"):
        OutcomeComparison(left="outcome_1", right="outcome_1")


# --- One: a single OutcomeComparison -------------------------------------


def test_comparison_stores_the_left_and_right_outcomes() -> None:
    """The comparison carries exactly the ordered pair it differences."""
    comparison = OutcomeComparison(left="outcome_1", right="outcome_2")
    assert comparison.left == "outcome_1"
    assert comparison.right == "outcome_2"


def test_comparison_is_frozen() -> None:
    """Comparisons are shared module state and cannot be edited in place."""
    comparison = OutcomeComparison(left="outcome_1", right="outcome_2")
    with pytest.raises(FrozenInstanceError):
        setattr(comparison, "left", "outcome_3")


# --- Many: the OUTCOME_COMPARISONS registry ------------------------------


def test_comparisons_pin_the_declared_pairs_in_order() -> None:
    """The tuple's order is the analysis stage's report and plot order."""
    assert (
        tuple((comparison.left, comparison.right) for comparison in OUTCOME_COMPARISONS)
        == EXPECTED_COMPARISONS
    )


def test_comparisons_are_a_read_only_tuple() -> None:
    """A consumer cannot append a comparison the analysis artifacts do not
    declare."""
    assert isinstance(OUTCOME_COMPARISONS, tuple)
    with pytest.raises(AttributeError):
        getattr(OUTCOME_COMPARISONS, "append")


def test_comparison_pairs_are_unique() -> None:
    """Each ordered pair is compared exactly once."""
    pairs = [(comparison.left, comparison.right) for comparison in OUTCOME_COMPARISONS]
    assert len(pairs) == len(set(pairs))


def test_shipped_comparisons_pass_the_module_level_validation() -> None:
    """The shipped tuple satisfies the duplicate-pair check run at import."""
    _validate_comparisons(OUTCOME_COMPARISONS)


def test_a_repeated_comparison_pair_is_rejected() -> None:
    """Comparing the same ordered pair twice is a registry error."""
    comparisons = (
        OutcomeComparison(left="outcome_1", right="outcome_2"),
        OutcomeComparison(left="outcome_1", right="outcome_2"),
    )
    with pytest.raises(ValueError, match="more than once"):
        _validate_comparisons(comparisons)
