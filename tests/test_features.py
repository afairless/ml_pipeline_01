"""Unit tests for the project feature policies.

The policies are the model strategies' feature contract: which columns the
estimator sees, in which order, and — for the substitution policies — the
exact seeded draw sequence. These tests pin:

- pass-through and selection shape: only the declared feature columns change
  or are projected, every other column and the column order stay identical,
  and the feature head keeps the key columns;
- the model-order ``feature_colnames`` each policy declares, including the
  outcome-plus-noise feature swap and the subset policy's head-order rule;
- the draw count, distribution, and train -> valid -> test order of the
  substitution policies, because a reordered draw shifts every later seeded
  value in the pipeline;
- the boundary contracts: a substitution policy rejects fewer than two
  feature columns, and a subset policy rejects empty, duplicate, unknown, and
  key-column selections.

Tests that draw call the global legacy RNG (the policies do), so they save
and restore its state around themselves; the suite must never leak a shifted
RNG state into a later test.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

import numpy as np
import polars as pl
import pytest

from src.project.features import (
    FeaturePolicy,
    FeatureSubset,
    IdentityFeatures,
    OutcomePlusNoise,
    PureNoise,
    SplitTablesLike,
)

SEED = 20261023

KEY_COLNAMES = ("case_id", "case_date")
FEATURE_COLNAMES = ("feature_1", "feature_2", "feature_3")
OUTCOME_COLNAME = "outcome_1"


@dataclass(frozen=True)
class FakeSplitTables:
    """The structural split-table shape the policies operate on."""

    feature_head_df: pl.DataFrame
    outcome_head_df: pl.DataFrame
    train_df: pl.DataFrame
    valid_df: pl.DataFrame
    test_df: pl.DataFrame
    key_colnames: Sequence[str]


def _frame(height: int, offset: int) -> pl.DataFrame:
    """Build one partition frame with distinct heights and values."""
    return pl.DataFrame(
        {
            "case_id": [f"case_{index:04d}" for index in range(height)],
            "case_date": [index + 1 for index in range(height)],
            "feature_1": [offset + index for index in range(height)],
            "feature_2": [offset + 100 + index for index in range(height)],
            "feature_3": [offset + 200 + index for index in range(height)],
            OUTCOME_COLNAME: [offset + 10 + index for index in range(height)],
        }
    )


@pytest.fixture
def split_tables() -> FakeSplitTables:
    """A tiny deterministic split set with distinct partition heights.

    Train 5, valid 3, and test 7 let the draw-order tests pin the
    train -> valid -> test sequence by the recorded ``size`` arguments.
    """
    head = _frame(1, 0)
    return FakeSplitTables(
        feature_head_df=head.select([*KEY_COLNAMES, *FEATURE_COLNAMES]),
        outcome_head_df=head.select([*KEY_COLNAMES, OUTCOME_COLNAME]),
        train_df=_frame(5, 0),
        valid_df=_frame(3, 1000),
        test_df=_frame(7, 2000),
        key_colnames=list(KEY_COLNAMES),
    )


@pytest.fixture
def single_feature_split_tables() -> FakeSplitTables:
    """A split set whose feature head declares a single feature column."""
    head = _frame(1, 0)
    return FakeSplitTables(
        feature_head_df=head.select([*KEY_COLNAMES, "feature_1"]),
        outcome_head_df=head.select([*KEY_COLNAMES, OUTCOME_COLNAME]),
        train_df=_frame(5, 0),
        valid_df=_frame(3, 1000),
        test_df=_frame(7, 2000),
        key_colnames=list(KEY_COLNAMES),
    )


@pytest.fixture
def rng_state() -> Iterator[None]:
    """Save and restore the global legacy RNG state around one test."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _partition_frames(split_tables: SplitTablesLike) -> list[pl.DataFrame]:
    """Return the partition frames in the one order policies draw in."""
    return [
        split_tables.train_df,
        split_tables.valid_df,
        split_tables.test_df,
    ]


def _unchanged_column_name(
    original: pl.DataFrame,
    substituted: pl.DataFrame,
    changed_colnames: set[str],
) -> str | None:
    """Return the first unchanged column that differs, or None."""
    for colname in original.columns:
        if colname in changed_colnames:
            continue
        if not substituted[colname].equals(original[colname]):
            return colname
    return None


def _install_recorder(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    name: str,
    calls: list[tuple[tuple[object, ...], dict[str, object]]],
) -> None:
    """Record calls to ``module.name`` while passing values through."""
    real = getattr(module, name)

    def recorder(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, recorder)


def test_every_policy_satisfies_the_policy_protocol() -> None:
    """The four declared policies expose the two protocol methods."""
    policies: tuple[FeaturePolicy, ...] = (
        IdentityFeatures(),
        OutcomePlusNoise(),
        PureNoise(),
        FeatureSubset(("feature_1",)),
    )
    for policy in policies:
        assert callable(policy.apply)
        assert callable(policy.feature_colnames)


class TestIdentityFeatures:
    """The pass-through policy used by the full-feature model."""

    def test_apply_returns_the_split_tables_unchanged(
        self,
        split_tables: FakeSplitTables,
        rng_state: Iterator[None],
    ) -> None:
        assert IdentityFeatures().apply(split_tables) is split_tables

    def test_feature_colnames_are_the_full_schema_in_head_order(
        self,
        split_tables: FakeSplitTables,
    ) -> None:
        assert IdentityFeatures().feature_colnames(split_tables) == list(
            FEATURE_COLNAMES
        )

    def test_apply_draws_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        split_tables: FakeSplitTables,
        rng_state: Iterator[None],
    ) -> None:
        normal_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        rint_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        _install_recorder(monkeypatch, np.random, "normal", normal_calls)
        _install_recorder(monkeypatch, np, "rint", rint_calls)

        IdentityFeatures().apply(split_tables)

        assert normal_calls == []
        assert rint_calls == []


class TestOutcomePlusNoise:
    """The outcome-plus-noise policy used by the dummy baseline models."""

    def test_changes_only_the_first_feature_column(
        self,
        split_tables: FakeSplitTables,
        rng_state: Iterator[None],
    ) -> None:
        np.random.seed(SEED)
        substituted = OutcomePlusNoise().apply(split_tables)

        np.random.seed(SEED)
        for partition, original in zip(
            ("train", "valid", "test"),
            _partition_frames(split_tables),
            strict=True,
        ):
            noise = np.random.normal(
                loc=0,
                scale=0.002,
                size=original.shape[0],
            )
            expected = original.with_columns(
                (pl.col(OUTCOME_COLNAME) + pl.Series(noise)).alias("feature_1")
            )
            substituted_frame = getattr(substituted, f"{partition}_df")
            assert substituted_frame.equals(expected)
            assert substituted_frame.columns == original.columns

        assert not substituted.train_df.equals(split_tables.train_df)
        assert substituted.train_df.height == split_tables.train_df.height
        for original, substituted_frame in zip(
            _partition_frames(split_tables),
            _partition_frames(substituted),
            strict=True,
        ):
            assert (
                _unchanged_column_name(
                    original,
                    substituted_frame,
                    {"feature_1"},
                )
                is None
            )

    def test_draws_exactly_three_normals_in_partition_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
        split_tables: FakeSplitTables,
        rng_state: Iterator[None],
    ) -> None:
        np.random.seed(SEED)
        normal_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        rint_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        _install_recorder(monkeypatch, np.random, "normal", normal_calls)
        _install_recorder(monkeypatch, np, "rint", rint_calls)

        OutcomePlusNoise().apply(split_tables)

        assert len(normal_calls) == 3
        assert rint_calls == []
        assert [kwargs["size"] for _, kwargs in normal_calls] == [5, 3, 7]
        for _, kwargs in normal_calls:
            assert kwargs["loc"] == 0
            assert kwargs["scale"] == 0.002

    def test_feature_colnames_reproduce_the_feature_swap(
        self,
        split_tables: FakeSplitTables,
    ) -> None:
        assert OutcomePlusNoise().feature_colnames(split_tables) == [
            "feature_2",
            "feature_1",
        ]

    def test_rejects_fewer_than_two_feature_columns(
        self,
        single_feature_split_tables: FakeSplitTables,
    ) -> None:
        policy = OutcomePlusNoise()
        with pytest.raises(ValueError, match="at least two feature columns"):
            policy.apply(single_feature_split_tables)
        with pytest.raises(ValueError, match="at least two feature columns"):
            policy.feature_colnames(single_feature_split_tables)


class TestPureNoise:
    """The pure-noise policy used by the negative-control models."""

    def test_changes_only_the_first_two_feature_columns(
        self,
        split_tables: FakeSplitTables,
        rng_state: Iterator[None],
    ) -> None:
        np.random.seed(SEED)
        substituted = PureNoise().apply(split_tables)

        np.random.seed(SEED)
        for partition, original in zip(
            ("train", "valid", "test"),
            _partition_frames(split_tables),
            strict=True,
        ):
            noise_1 = np.random.normal(
                loc=0,
                scale=0.2,
                size=original.shape[0],
            )
            noise_2 = np.rint(
                np.random.normal(
                    loc=100,
                    scale=5,
                    size=original.shape[0],
                ).astype(int)
            )
            expected = original.with_columns(
                pl.Series(noise_1).alias("feature_1"),
                pl.Series(noise_2).alias("feature_2"),
            )
            substituted_frame = getattr(substituted, f"{partition}_df")
            assert substituted_frame.equals(expected)
            assert substituted_frame.columns == original.columns

        assert not substituted.train_df.equals(split_tables.train_df)
        for original, substituted_frame in zip(
            _partition_frames(split_tables),
            _partition_frames(substituted),
            strict=True,
        ):
            assert (
                _unchanged_column_name(
                    original,
                    substituted_frame,
                    {"feature_1", "feature_2"},
                )
                is None
            )

    def test_draws_six_normals_and_three_rints_in_partition_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
        split_tables: FakeSplitTables,
        rng_state: Iterator[None],
    ) -> None:
        np.random.seed(SEED)
        normal_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        rint_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        _install_recorder(monkeypatch, np.random, "normal", normal_calls)
        _install_recorder(monkeypatch, np, "rint", rint_calls)

        PureNoise().apply(split_tables)

        assert len(normal_calls) == 6
        assert len(rint_calls) == 3
        assert [(kwargs["loc"], kwargs["scale"]) for _, kwargs in normal_calls] == [
            (0, 0.2),
            (100, 5),
            (0, 0.2),
            (100, 5),
            (0, 0.2),
            (100, 5),
        ]
        assert [kwargs["size"] for _, kwargs in normal_calls] == [
            5,
            5,
            3,
            3,
            7,
            7,
        ]
        assert [np.asarray(args[0]).shape[0] for args, _ in rint_calls] == [
            5,
            3,
            7,
        ]

    def test_feature_colnames_are_the_first_two_features_in_head_order(
        self,
        split_tables: FakeSplitTables,
    ) -> None:
        assert PureNoise().feature_colnames(split_tables) == [
            "feature_1",
            "feature_2",
        ]

    def test_rejects_fewer_than_two_feature_columns(
        self,
        single_feature_split_tables: FakeSplitTables,
    ) -> None:
        policy = PureNoise()
        with pytest.raises(ValueError, match="at least two feature columns"):
            policy.apply(single_feature_split_tables)
        with pytest.raises(ValueError, match="at least two feature columns"):
            policy.feature_colnames(single_feature_split_tables)


class TestFeatureSubset:
    """The explicit-selection policy used by the subset models."""

    def test_feature_colnames_preserve_head_order_not_argument_order(
        self,
        split_tables: FakeSplitTables,
    ) -> None:
        policy = FeatureSubset(("feature_3", "feature_1"))
        assert policy.feature_colnames(split_tables) == [
            "feature_1",
            "feature_3",
        ]

    def test_apply_projects_the_feature_head_to_keys_plus_selection(
        self,
        split_tables: FakeSplitTables,
    ) -> None:
        projected = FeatureSubset(("feature_3", "feature_1")).apply(split_tables)

        assert projected.feature_head_df.columns == [
            "case_id",
            "case_date",
            "feature_1",
            "feature_3",
        ]
        assert projected.outcome_head_df is split_tables.outcome_head_df
        assert projected.train_df is split_tables.train_df
        assert projected.valid_df is split_tables.valid_df
        assert projected.test_df is split_tables.test_df
        assert projected.key_colnames == list(KEY_COLNAMES)
        assert projected.key_colnames is not split_tables.key_colnames

    def test_apply_does_not_mutate_the_input(
        self,
        split_tables: FakeSplitTables,
    ) -> None:
        original_columns = list(split_tables.feature_head_df.columns)
        FeatureSubset(("feature_2",)).apply(split_tables)
        assert split_tables.feature_head_df.columns == original_columns

    def test_apply_draws_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        split_tables: FakeSplitTables,
        rng_state: Iterator[None],
    ) -> None:
        normal_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        rint_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        _install_recorder(monkeypatch, np.random, "normal", normal_calls)
        _install_recorder(monkeypatch, np, "rint", rint_calls)

        FeatureSubset(("feature_2",)).apply(split_tables)

        assert normal_calls == []
        assert rint_calls == []

    def test_rejects_an_empty_selection(self) -> None:
        with pytest.raises(ValueError, match="at least one feature column"):
            FeatureSubset(())

    def test_rejects_a_duplicate_column(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            FeatureSubset(("feature_1", "feature_1"))

    def test_rejects_an_empty_column_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            FeatureSubset(("feature_1", ""))

    def test_rejects_an_unknown_column(
        self,
        split_tables: FakeSplitTables,
    ) -> None:
        policy = FeatureSubset(("feature_99",))
        with pytest.raises(ValueError, match="not feature columns"):
            policy.feature_colnames(split_tables)
        with pytest.raises(ValueError, match="not feature columns"):
            policy.apply(split_tables)

    def test_rejects_a_key_column_selection(
        self,
        split_tables: FakeSplitTables,
    ) -> None:
        with pytest.raises(ValueError, match="not feature columns"):
            FeatureSubset(("case_id",)).apply(split_tables)

    def test_policy_value_is_frozen(self) -> None:
        policy = FeatureSubset(("feature_1",))
        with pytest.raises(AttributeError):
            setattr(policy, "columns", ("feature_2",))

    def test_normalizes_a_sequence_to_a_tuple(self) -> None:
        policy = FeatureSubset(["feature_1", "feature_2"])  # type: ignore[arg-type]
        assert policy.columns == ("feature_1", "feature_2")

    def test_replace_preserves_the_concrete_split_table_type(
        self,
        split_tables: FakeSplitTables,
    ) -> None:
        projected = replace(split_tables, train_df=split_tables.train_df)
        assert isinstance(projected, FakeSplitTables)
