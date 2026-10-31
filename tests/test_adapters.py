"""Unit tests for the demo outcome and feature adapters.

The adapters are the template's extension point, so these tests pin the
contract a replacing project must preserve:

- shape and schema: the outcome builder emits the canonical keys plus
  ``outcome_1`` through ``outcome_4``, and the feature builder emits the keys
  plus ``feature_1`` through ``feature_8``, both satisfying the stage boundary
  schemas with one row per source row;
- alignment: both frames carry every source key exactly once, sorted by the
  canonical key columns, so the later joins cannot drop or duplicate rows;
- determinism: repeated calls are identical, a shuffled input yields the same
  frame, and each builder's private generator leaves the global ``numpy``
  random state untouched;
- the documented formulas and draw order, including the row-by-row
  ``outcome_1`` / ``outcome_3`` draw sequence and the draw-free feature
  formulas;
- the demo distribution guarantee that every configured threshold splits both
  evaluated partitions of every modeled outcome into two non-empty classes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.project.adapters import build_feature_table, build_outcome_table
from src.project.columns import CASE_DATE, CASE_ID, KEY_COLUMNS
from src.project.config import CONFIG
from src.project.inputs import read_input_table
from src.project.outcomes import modeled_outcomes
from src.project.schemas import (
    FEATURE_COLUMNS,
    OUTCOME_COLUMNS,
    S01_OUTCOME_SCHEMA,
    S02_FEATURE_SCHEMA,
    validate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Repository root, used only to locate the committed demo input table."""

OUTCOME_TABLE_COLUMNS = (CASE_ID, CASE_DATE, *OUTCOME_COLUMNS)
FEATURE_TABLE_COLUMNS = (CASE_ID, CASE_DATE, *FEATURE_COLUMNS)

PARTITION_PREDICATES = {
    "train": pl.col(CASE_DATE) <= CONFIG.train_end_date,
    "valid": (pl.col(CASE_DATE) > CONFIG.train_end_date)
    & (pl.col(CASE_DATE) < CONFIG.test_start_date),
    "test": pl.col(CASE_DATE) >= CONFIG.test_start_date,
}


@pytest.fixture(scope="module")
def demo_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return the committed source table and both adapter outputs."""
    source_df = read_input_table(
        "source_table_1",
        PROJECT_ROOT / CONFIG.data_root,
    )
    outcome_df = build_outcome_table(source_df, CONFIG)
    feature_df = build_feature_table(source_df, outcome_df, CONFIG)
    return source_df, outcome_df, feature_df


def _source_frame() -> pl.DataFrame:
    """Return a small, already key-sorted source frame with signed values."""
    return pl.DataFrame(
        {
            CASE_ID: pl.Series(["case_a", "case_b", "case_c"], dtype=pl.String),
            CASE_DATE: pl.Series(
                [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                dtype=pl.Date,
            ),
            "value_1": pl.Series([1.0, -2.0, 0.0], dtype=pl.Float64),
            "value_2": pl.Series([0.5, 3.0, -0.5], dtype=pl.Float64),
            "value_3": pl.Series([-1.0, 0.25, 2.0], dtype=pl.Float64),
        }
    )


def _assert_global_rng_untouched(builder: Callable[[], object]) -> None:
    """Assert that ``builder`` consumes no draws from the global RNG."""
    state = np.random.get_state()
    try:
        np.random.seed(4321)
        expected = np.random.random(4)
        np.random.seed(4321)
        builder()
        observed = np.random.random(4)
    finally:
        np.random.set_state(state)
    assert np.array_equal(observed, expected)


# --- outcome table ---------------------------------------------------------


def test_build_outcome_table_matches_the_documented_shape(
    demo_tables: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    source_df, outcome_df, _ = demo_tables

    assert outcome_df.height == source_df.height == CONFIG.demo_case_count
    assert tuple(outcome_df.columns) == OUTCOME_TABLE_COLUMNS
    assert outcome_df.schema == {
        CASE_ID: pl.String,
        CASE_DATE: pl.Date,
        **{name: pl.Float64 for name in OUTCOME_COLUMNS},
    }


def test_build_outcome_table_satisfies_the_outcome_schema(
    demo_tables: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    source_df, outcome_df, _ = demo_tables

    assert (
        validate(
            outcome_df,
            S01_OUTCOME_SCHEMA,
            "demo outcome table",
            expected_row_count=source_df.height,
        )
        is outcome_df
    )


def test_build_outcome_table_preserves_every_source_key_in_order(
    demo_tables: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    source_df, outcome_df, _ = demo_tables

    expected_keys = source_df.select(list(KEY_COLUMNS)).sort(list(KEY_COLUMNS))
    assert outcome_df.select(list(KEY_COLUMNS)).equals(expected_keys)
    assert outcome_df["case_id"].n_unique() == source_df.height
    assert outcome_df["case_id"].is_sorted()
    assert outcome_df["case_date"].is_sorted()


def test_build_outcome_table_is_deterministic() -> None:
    source_df = _source_frame()

    assert build_outcome_table(source_df, CONFIG).equals(
        build_outcome_table(source_df, CONFIG)
    )


def test_build_outcome_table_follows_the_documented_draw_order() -> None:
    source_df = _source_frame()
    outcome_df = build_outcome_table(source_df, CONFIG)

    generator = np.random.default_rng(CONFIG.demo_seed)
    expected_1: list[float] = []
    expected_2: list[float] = []
    expected_3: list[float] = []
    expected_4: list[float] = []
    # The demo contract consumes the outcome_1 draw before the outcome_3 draw
    # for each source row, so the expected sequence is built row by row.
    for row in source_df.iter_rows(named=True):
        value_1 = row["value_1"]
        value_2 = row["value_2"]
        value_3 = row["value_3"]
        expected_1.append(0.60 * value_1 + 0.30 * value_2 + 0.10 * generator.normal())
        expected_2.append(value_1 - value_3)
        expected_3.append(
            value_1 * value_2 + 0.30 * value_3 + 0.05 * generator.normal()
        )
        expected_4.append(value_3)

    assert outcome_df["outcome_1"].to_list() == pytest.approx(expected_1)
    assert outcome_df["outcome_2"].to_list() == pytest.approx(expected_2)
    assert outcome_df["outcome_3"].to_list() == pytest.approx(expected_3)
    assert outcome_df["outcome_4"].to_list() == pytest.approx(expected_4)


def test_build_outcome_table_seed_changes_only_the_noisy_outcomes() -> None:
    source_df = _source_frame()
    baseline = build_outcome_table(source_df, CONFIG)
    other = build_outcome_table(
        source_df,
        replace(CONFIG, demo_seed=CONFIG.demo_seed + 1),
    )

    assert not np.allclose(baseline["outcome_1"], other["outcome_1"])
    assert not np.allclose(baseline["outcome_3"], other["outcome_3"])
    assert baseline["outcome_2"].equals(other["outcome_2"])
    assert baseline["outcome_4"].equals(other["outcome_4"])


def test_build_outcome_table_accepts_a_shuffled_source_table() -> None:
    source_df = _source_frame()

    assert build_outcome_table(source_df.reverse(), CONFIG).equals(
        build_outcome_table(source_df, CONFIG)
    )


def test_build_outcome_table_does_not_touch_the_global_rng() -> None:
    source_df = _source_frame()
    _assert_global_rng_untouched(lambda: build_outcome_table(source_df, CONFIG))


def test_build_outcome_table_does_not_mutate_the_source() -> None:
    source_df = _source_frame()
    before = source_df.clone()

    build_outcome_table(source_df, CONFIG)

    assert source_df.equals(before)


# --- feature table ---------------------------------------------------------


def test_build_feature_table_matches_the_documented_shape(
    demo_tables: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    source_df, _, feature_df = demo_tables

    assert feature_df.height == source_df.height == CONFIG.demo_case_count
    assert tuple(feature_df.columns) == FEATURE_TABLE_COLUMNS
    assert feature_df.schema == {
        CASE_ID: pl.String,
        CASE_DATE: pl.Date,
        **{name: pl.Float64 for name in FEATURE_COLUMNS},
    }


def test_build_feature_table_satisfies_the_feature_schema(
    demo_tables: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    source_df, _, feature_df = demo_tables

    assert (
        validate(
            feature_df,
            S02_FEATURE_SCHEMA,
            "demo feature table",
            expected_row_count=source_df.height,
        )
        is feature_df
    )


def test_build_feature_table_preserves_every_source_key_in_order(
    demo_tables: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    source_df, outcome_df, feature_df = demo_tables

    expected_keys = source_df.select(list(KEY_COLUMNS)).sort(list(KEY_COLUMNS))
    assert feature_df.select(list(KEY_COLUMNS)).equals(expected_keys)
    assert feature_df.select(list(KEY_COLUMNS)).equals(
        outcome_df.select(list(KEY_COLUMNS))
    )
    assert feature_df["case_id"].n_unique() == source_df.height


def test_build_feature_table_applies_the_documented_formulas() -> None:
    source_df = _source_frame()
    feature_df = build_feature_table(source_df, source_df, CONFIG)

    value_1 = source_df["value_1"].to_list()
    value_2 = source_df["value_2"].to_list()
    value_3 = source_df["value_3"].to_list()
    expected = {
        "feature_1": value_1,
        "feature_2": value_2,
        "feature_3": value_3,
        "feature_4": [first * second for first, second in zip(value_1, value_2)],
        "feature_5": [first**2 for first in value_1],
        "feature_6": [second**2 for second in value_2],
        "feature_7": [
            first + second + third
            for first, second, third in zip(value_1, value_2, value_3)
        ],
        "feature_8": [
            (first - second) / (1 + abs(third))
            for first, second, third in zip(value_1, value_2, value_3)
        ],
    }

    for name, values in expected.items():
        assert feature_df[name].to_list() == pytest.approx(values)


def test_build_feature_table_is_deterministic() -> None:
    source_df = _source_frame()
    outcome_df = build_outcome_table(source_df, CONFIG)

    assert build_feature_table(source_df, outcome_df, CONFIG).equals(
        build_feature_table(source_df, outcome_df, CONFIG)
    )


def test_build_feature_table_makes_no_random_draws() -> None:
    source_df = _source_frame()
    outcome_df = build_outcome_table(source_df, CONFIG)

    baseline = build_feature_table(source_df, outcome_df, CONFIG)
    other = build_feature_table(
        source_df,
        outcome_df,
        replace(CONFIG, demo_seed=CONFIG.demo_seed + 1),
    )

    assert baseline.equals(other)


def test_build_feature_table_ignores_the_outcome_frame() -> None:
    source_df = _source_frame()
    outcome_df = build_outcome_table(source_df, CONFIG)
    key_only = source_df.select(list(KEY_COLUMNS))

    assert build_feature_table(source_df, outcome_df, CONFIG).equals(
        build_feature_table(source_df, key_only, CONFIG)
    )


def test_build_feature_table_accepts_a_shuffled_source_table() -> None:
    source_df = _source_frame()
    outcome_df = build_outcome_table(source_df, CONFIG)

    assert build_feature_table(source_df.reverse(), outcome_df, CONFIG).equals(
        build_feature_table(source_df, outcome_df, CONFIG)
    )


def test_build_feature_table_does_not_touch_the_global_rng() -> None:
    source_df = _source_frame()
    outcome_df = build_outcome_table(source_df, CONFIG)
    _assert_global_rng_untouched(
        lambda: build_feature_table(source_df, outcome_df, CONFIG)
    )


def test_build_feature_table_does_not_mutate_the_source() -> None:
    source_df = _source_frame()
    outcome_df = build_outcome_table(source_df, CONFIG)
    before = source_df.clone()

    build_feature_table(source_df, outcome_df, CONFIG)

    assert source_df.equals(before)


# --- demo distribution -----------------------------------------------------


@pytest.mark.parametrize(
    "outcome_name", [outcome.name for outcome in modeled_outcomes()]
)
@pytest.mark.parametrize("threshold", CONFIG.binary_thresholds)
@pytest.mark.parametrize("partition", CONFIG.evaluated_partitions)
def test_every_evaluated_partition_has_both_binary_classes(
    demo_tables: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
    outcome_name: str,
    threshold: float,
    partition: str,
) -> None:
    _, outcome_df, _ = demo_tables
    values = outcome_df.filter(PARTITION_PREDICATES[partition])[outcome_name]
    positives = int((values > threshold).sum())
    negatives = values.len() - positives

    assert positives > 0, (
        f"{outcome_name} has no value > {threshold} in the {partition} partition"
    )
    assert negatives > 0, (
        f"{outcome_name} has no value <= {threshold} in the {partition} partition"
    )
