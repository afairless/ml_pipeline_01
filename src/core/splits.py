"""Data partition contract: named holdout partitions as pure dataframe types.

Defines the vocabulary s04's split-table artifacts, s05's partition-named
prediction files, and s06's evaluated partition tables share: every partition
is a frozen ``DataPartition`` of three aligned frames — the key frame (the
``(case_id, case_date)`` identity rows), the feature frame, and the outcome
frame — carrying its canonical ``name`` from ``PARTITION_ORDER``.
``holdout_partitions`` arranges three such partitions into the canonical
``(train, valid, test)`` tuple, asserting the key-alignment contract on the
way through.

The module is deliberately pure dataframe logic: it holds no layout, no
project vocabulary, no artifact I/O, and no ``np.random`` draws — it only
*naming* and *ordering* partitions, so the core layer reuses it for every
consumer of the split shape. The refit/``in_sample`` projection is
deliberately **not** part of the contract: no trainer produces a refit run, so
adding one means adding its partition names here together with the producer
that enables it.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

PARTITION_ORDER: tuple[str, ...] = ("train", "valid", "test")
"""Canonical holdout partition names, in the order downstream stages consume.

The tuple is the one place the vocabulary lives: a ``DataPartition.name`` must
be one of these, and ``holdout_partitions`` returns its partitions in exactly
this order. A future refit projection adds its names here together with the
producer that enables a refit run.
"""


def _assert_key_alignment(partition: DataPartition) -> None:
    """Assert one partition's three frames agree in row count (and exist).

    The key, feature, and outcome frames of a partition are cuts of the same
    joined holdout table (s03 splits one table; ``prepare_model_ready_data``
    later derives the key/frame/outcome selections from it), so they must
    cover the same rows: a height mismatch means the partitions were built
    from different tables and would silently pair the wrong rows. An empty
    frame is rejected for the same reason — an empty partition cannot be a
    cut of a real holdout table and would corrupt every downstream count.
    """
    heights = {
        partition.key_df.height,
        partition.feature_df.height,
        partition.outcome_df.height,
    }
    if len(heights) != 1:
        raise ValueError(
            f"{partition.name} partition frames are not row-aligned: "
            f"key_df has {partition.key_df.height} rows, "
            f"feature_df has {partition.feature_df.height}, "
            f"outcome_df has {partition.outcome_df.height}"
        )
    if partition.key_df.is_empty():
        raise ValueError(
            f"empty {partition.name} partition: key_df, feature_df, and "
            "outcome_df must each hold at least one row"
        )


@dataclass(frozen=True)
class DataPartition:
    """One holdout partition: its key, feature, and outcome frames.

    Attributes:
        name: The canonical partition name — one of ``PARTITION_ORDER``
            (``'train'``, ``'valid'``, ``'test'``). An unknown name is
            rejected at construction so a typo cannot silently produce a
            partition no downstream stage recognizes.
        key_df: The partition's key (identity) rows, e.g. the
            ``(case_id, case_date)`` pair on every holdout row.
        feature_df: The partition's model-input features, one row per key.
        outcome_df: The partition's target column(s), one row per key.
    """

    name: str
    key_df: pl.DataFrame
    feature_df: pl.DataFrame
    outcome_df: pl.DataFrame

    def __post_init__(self) -> None:
        """Validate the partition at construction: canonical name and alignment.

        Construction-time validation keeps the invariant where every builder
        benefits from it: a ``DataPartition`` that exists is a valid partition,
        so ``holdout_partitions`` and every downstream partition table get the
        name, alignment, and non-emptiness checks for free instead of
        re-implementing them per consumer.
        """
        if self.name not in PARTITION_ORDER:
            raise ValueError(
                f"invalid partition name {self.name!r}: must be one of "
                f"{PARTITION_ORDER!r}"
            )
        _assert_key_alignment(self)


def holdout_partitions(
    train: DataPartition,
    valid: DataPartition,
    test: DataPartition,
) -> tuple[DataPartition, ...]:
    """Arrange three holdout partitions into the canonical train/valid/test tuple.

    Pure pass-through plus contract assertion: the partitions are returned
    exactly as given (same frames, same row counts), after checking that they
    are named and ordered per ``PARTITION_ORDER``. The key-alignment assertion
    is re-run at this boundary so a caller assembling a triple from
    already-constructed partitions gets the same failure whether a partition
    was built out of alignment here (e.g. from two different split tables) or
    at its own construction.

    Raises:
        ValueError: The three partitions are not named ``('train', 'valid',
            'test')`` in exactly that order, or any partition's frames are not
            row-aligned / are empty.
    """
    names = (train.name, valid.name, test.name)
    if names != PARTITION_ORDER:
        raise ValueError(
            f"holdout partitions must be named in {PARTITION_ORDER!r} order, "
            f"got {names!r}"
        )
    for partition in (train, valid, test):
        _assert_key_alignment(partition)
    return (train, valid, test)
