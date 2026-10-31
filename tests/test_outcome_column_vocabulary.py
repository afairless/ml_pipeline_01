"""Column-vocabulary tests for the project outcome registry.

The outcome registry is also a column vocabulary: every ``Outcome.name`` is
the outcome-table column name and the ``{outcome}`` placeholder value. These
tests pin the exact demo names, their ``outcome_<n>`` spelling, uniqueness,
registry order, and disjointness from the canonical key columns.

They are separate from ``tests/test_outcomes.py`` on purpose: that file covers
registry behavior and transforms, while this file guards the naming contract
shared with boundary schemas, stage paths, and the analysis stage. A failure
here means a registry edit would stop being addressable as a column.
"""

from __future__ import annotations

import re

from src.project.columns import KEY_COLUMNS
from src.project.outcomes import OUTCOMES, modeled_outcomes, outcome_names

EXPECTED_OUTCOME_NAMES = ("outcome_1", "outcome_2", "outcome_3", "outcome_4")

OUTCOME_NAME_PATTERN = re.compile(r"^outcome_[1-9][0-9]*$")
"""The destination vocabulary: outcomes are numbered columns."""


def test_outcome_names_are_pinned_exactly() -> None:
    """The demo registry exposes outcome_1 through outcome_4 in run order."""
    assert outcome_names() == EXPECTED_OUTCOME_NAMES


def test_registry_keys_match_the_column_names() -> None:
    """A registry key and its ``Outcome.name`` are the same column name."""
    assert tuple(OUTCOMES) == outcome_names()
    for name, outcome in OUTCOMES.items():
        assert outcome.name == name


def test_outcome_names_are_unique() -> None:
    """Two outcomes cannot share one column name."""
    names = outcome_names()
    assert len(names) == len(set(names))


def test_every_outcome_name_is_a_lowercase_snake_case_identifier() -> None:
    """Each name is a non-empty, lowercase, space-free column identifier."""
    for name in outcome_names():
        assert name
        assert name == name.strip()
        assert name == name.lower()
        assert " " not in name
        assert name.replace("_", "").isalnum()
        assert not name.startswith("_")
        assert not name.endswith("_")
        assert "__" not in name


def test_every_outcome_name_matches_the_numbered_vocabulary() -> None:
    """The complete name is ``outcome_<positive integer>``."""
    for name in outcome_names():
        assert OUTCOME_NAME_PATTERN.fullmatch(name) is not None


def test_outcome_names_are_ordered_by_numeric_suffix() -> None:
    """Registry order follows the numeric suffix, which is run order."""
    suffixes = [int(name.removeprefix("outcome_")) for name in outcome_names()]
    assert suffixes == sorted(suffixes)


def test_outcome_names_do_not_collide_with_key_columns() -> None:
    """Outcome columns and identity columns are disjoint vocabularies."""
    assert set(outcome_names()).isdisjoint(KEY_COLUMNS)


def test_modeled_outcome_names_are_a_subset_of_the_registry() -> None:
    """Every modeled outcome is a registered column, and the filter has at
    least one entry on each side of the modeled flag."""
    registered = set(outcome_names())
    modeled = {outcome.name for outcome in modeled_outcomes()}
    assert modeled
    assert modeled <= registered
    assert registered - modeled


def test_every_outcome_name_is_usable_as_a_registry_lookup() -> None:
    """Each name in the vocabulary addresses exactly one registry entry."""
    for name in outcome_names():
        assert OUTCOMES[name].name == name
