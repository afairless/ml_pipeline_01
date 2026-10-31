"""Unit and property tests for the core data partition contract.

Pins the invariant that a ``DataPartition`` is a pure, validated dataframe
triple: construction rejects an unknown name, an empty frame, and a
row-misaligned triple, and ``holdout_partitions`` accepts exactly the three
canonical names in ``PARTITION_ORDER`` order while passing the given frames
through unchanged. A failure here means a stage could consume a misaligned
or misnamed partition and silently evaluate the wrong rows.

The contract types hold no layout, project names, or artifact I/O, so the
module needs no fixtures beyond plain polars frames. The property tests
exercise arbitrary partition sizes, names, and frame alignments with
Hypothesis, so the invariants are not pinned only to the hand-picked
examples. ``src.core.splits`` joins the import-contract scan automatically
once this file imports it.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.core.splits import PARTITION_ORDER, DataPartition, holdout_partitions


def _frames(n_rows: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Three aligned single-purpose frames with ``n_rows`` rows each.

    The column names make each frame's role visible in a failure diff: the key
    frame holds the ``(case_id, case_date)`` identity rows, the feature frame
    the model inputs, and the outcome frame the target column. All frames
    share the same row count by construction, so a misalignment test must
    build its own broken triple.
    """
    return (
        pl.DataFrame(
            {
                "case_id": [f"case_{i:04d}" for i in range(n_rows)],
                "case_date": [
                    dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_rows)
                ],
            }
        ),
        pl.DataFrame({"feature": list(range(n_rows))}),
        pl.DataFrame({"outcome": [float(i) for i in range(n_rows)]}),
    )


def _partition(name: str, n_rows: int) -> DataPartition:
    """A valid ``DataPartition`` for canonical name ``name`` with aligned frames."""
    key_df, feature_df, outcome_df = _frames(n_rows)
    return DataPartition(name, key_df, feature_df, outcome_df)


def test_partition_order_is_the_three_canonical_names_in_run_order() -> None:
    """Valid input: the canonical vocabulary is exactly train/valid/test in
    that order — the order downstream stages must consume them in."""
    assert PARTITION_ORDER == ("train", "valid", "test")


class TestDataPartitionConstruction:
    """Construction-time contract: canonical name, row alignment, non-empty."""

    @pytest.mark.parametrize("name", PARTITION_ORDER)
    def test_each_canonical_partition_name_is_accepted(self, name: str) -> None:
        """Valid input: every name in ``PARTITION_ORDER`` builds a valid
        partition, so ``holdout_partitions`` can receive any three of them."""
        partition = _partition(name, n_rows=4)
        assert partition.name == name
        assert partition.key_df.height == 4

    def test_one_row_partition_is_a_valid_partition(self) -> None:
        """Valid input: a single-row partition (the smallest non-empty cut)
        passes the alignment and non-emptiness checks."""
        partition = _partition("train", n_rows=1)
        assert partition.key_df.height == 1
        assert partition.feature_df.height == 1
        assert partition.outcome_df.height == 1

    def test_unknown_partition_name_is_rejected(self) -> None:
        """Invalid input: a name outside ``PARTITION_ORDER`` (the deferred
        refit/in_sample projection is not registered here) is rejected so a
        typo cannot produce a partition no consumer recognizes."""
        key_df, feature_df, outcome_df = _frames(n_rows=2)
        with pytest.raises(ValueError, match="invalid partition name"):
            DataPartition("refit", key_df, feature_df, outcome_df)

    def test_empty_string_partition_name_is_rejected(self) -> None:
        """Invalid input: the empty-string name is not one of the canonical
        partition names and raises the same construction error."""
        key_df, feature_df, outcome_df = _frames(n_rows=2)
        with pytest.raises(ValueError, match="invalid partition name"):
            DataPartition("", key_df, feature_df, outcome_df)

    def test_empty_partition_is_rejected(self) -> None:
        """Invalid input: a zero-row partition is not a cut of a real holdout
        table and would corrupt every downstream count, so its construction
        raises."""
        with pytest.raises(ValueError, match="empty"):
            _partition("train", n_rows=0)

    def test_key_frame_alignment_failure_raises(self) -> None:
        """Invalid input: feature and outcome frames cut from a different
        table than the key frame would silently pair the wrong rows, so a
        height mismatch raises at construction."""
        key_df, _, outcome_df = _frames(n_rows=2)
        misaligned_feature_df = pl.DataFrame({"feature": [0, 1, 2]})
        with pytest.raises(ValueError, match="not row-aligned"):
            DataPartition("train", key_df, misaligned_feature_df, outcome_df)

    def test_frozen_partition_cannot_be_mutated(self) -> None:
        """Invariant: the dataclass is frozen, so a consumer cannot rename a
        validated partition after construction."""
        partition = _partition("train", n_rows=2)
        with pytest.raises(AttributeError):
            partition.name = "valid"  # type: ignore[misc]


class TestHoldoutPartitions:
    """The builder: canonical triple in order, pure pass-through, no reshuffle."""

    def test_returns_train_valid_test_in_canonical_order(self) -> None:
        """Valid input: the three partitions come back named and ordered
        exactly as ``PARTITION_ORDER`` declares, with each partition's own
        row count intact."""
        result = holdout_partitions(
            _partition("train", n_rows=3),
            _partition("valid", n_rows=2),
            _partition("test", n_rows=1),
        )
        assert len(result) == 3
        assert [p.name for p in result] == ["train", "valid", "test"]
        assert [p.key_df.height for p in result] == [3, 2, 1]

    def test_pass_through_preserves_the_given_frames_and_row_counts(self) -> None:
        """Valid input: the builder is a pass-through — it returns the exact
        partition objects it was given (identity, never a rebuilt copy), so
        frame values and row counts survive untouched."""
        train = _partition("train", n_rows=3)
        valid = _partition("valid", n_rows=2)
        test = _partition("test", n_rows=1)
        result = holdout_partitions(train, valid, test)
        assert result[0] is train
        assert result[1] is valid
        assert result[2] is test
        for partition in result:
            assert partition.key_df.height == partition.feature_df.height
            assert partition.feature_df.height == partition.outcome_df.height

    def test_rejects_partitions_in_non_canonical_order(self) -> None:
        """Invalid input: a triple assembled in a different order (e.g. valid,
        train, test) must raise rather than silently reorder the holdout, so
        a swapped argument can never swap the evaluation scope."""
        with pytest.raises(ValueError, match="must be named"):
            holdout_partitions(
                _partition("valid", 2),
                _partition("train", 3),
                _partition("test", 1),
            )

    def test_rejects_duplicate_partition_names(self) -> None:
        """Invalid input: a triple that repeats a name (here two ``train``
        partitions) is not the canonical holdout and raises, so no partition
        can stand in for another."""
        with pytest.raises(ValueError, match="must be named"):
            holdout_partitions(
                _partition("train", 2),
                _partition("train", 3),
                _partition("test", 1),
            )


class TestDataPartitionProperties:
    """Hypothesis properties over arbitrary sizes, names, and alignments."""

    @given(
        name=st.sampled_from(PARTITION_ORDER),
        n_rows=st.integers(min_value=1, max_value=100),
    )
    def test_any_canonical_name_and_positive_size_builds(
        self, name: str, n_rows: int
    ) -> None:
        """Property: every canonical name and every non-empty size constructs
        a partition whose three frames all report exactly that size."""
        partition = DataPartition(name, *_frames(n_rows))
        assert partition.name == name
        assert partition.key_df.height == n_rows
        assert partition.feature_df.height == n_rows
        assert partition.outcome_df.height == n_rows

    @given(name=st.text().filter(lambda candidate: candidate not in PARTITION_ORDER))
    def test_any_name_outside_the_order_is_rejected(self, name: str) -> None:
        """Property: any name outside ``PARTITION_ORDER`` — including text
        that merely resembles a canonical name — is rejected."""
        key_df, feature_df, outcome_df = _frames(n_rows=2)
        with pytest.raises(ValueError, match="invalid partition name"):
            DataPartition(name, key_df, feature_df, outcome_df)

    @given(
        n_rows=st.integers(min_value=1, max_value=50),
        feature_delta=st.integers(min_value=-5, max_value=5),
        outcome_delta=st.integers(min_value=-5, max_value=5),
    )
    def test_alignment_holds_exactly_when_all_frame_sizes_match(
        self, n_rows: int, feature_delta: int, outcome_delta: int
    ) -> None:
        """Property: construction succeeds iff all three frames agree in row
        count; any differing height raises the row-alignment error."""
        key_df, feature_df, outcome_df = _frames(n_rows)
        feature_rows = max(n_rows + feature_delta, 0)
        outcome_rows = max(n_rows + outcome_delta, 0)
        broken_feature_df = pl.DataFrame({"feature": list(range(feature_rows))})
        broken_outcome_df = pl.DataFrame(
            {"outcome": [float(i) for i in range(outcome_rows)]}
        )
        aligned = feature_rows == outcome_rows == n_rows
        if aligned:
            partition = DataPartition(
                "train", key_df, broken_feature_df, broken_outcome_df
            )
            assert partition.key_df.height == n_rows
        else:
            with pytest.raises(ValueError, match="not row-aligned"):
                DataPartition("train", key_df, broken_feature_df, broken_outcome_df)

    @given(n_rows=st.integers(min_value=1, max_value=50))
    def test_holdout_partitions_is_an_identity_pass_through(self, n_rows: int) -> None:
        """Property: the builder never rebuilds or reorders its inputs — it
        returns the very same partition objects, in the given order."""
        partitions = tuple(_partition(part, n_rows) for part in PARTITION_ORDER)
        train, valid, test = partitions
        result = holdout_partitions(train, valid, test)
        assert len(result) == 3
        for given_partition, returned_partition in zip(partitions, result):
            assert returned_partition is given_partition
            assert returned_partition.name in PARTITION_ORDER
        assert [partition.name for partition in result] == list(PARTITION_ORDER)

    @given(
        order=st.permutations(PARTITION_ORDER).filter(
            lambda candidate: tuple(candidate) != PARTITION_ORDER
        )
    )
    def test_any_non_canonical_order_is_rejected(self, order: list[str]) -> None:
        """Property: every permutation other than ``PARTITION_ORDER`` raises,
        so no argument ordering can silently reorder the holdout."""
        train, valid, test = (_partition(name, n_rows=2) for name in order)
        with pytest.raises(ValueError, match="must be named"):
            holdout_partitions(train, valid, test)

    @given(name=st.sampled_from(PARTITION_ORDER))
    def test_empty_any_canonical_partition_is_rejected(self, name: str) -> None:
        """Property: an empty partition is rejected for every canonical name."""
        key_df, feature_df, outcome_df = _frames(n_rows=3)
        with pytest.raises(ValueError, match="empty"):
            DataPartition(
                name,
                key_df.head(0),
                feature_df.head(0),
                outcome_df.head(0),
            )
