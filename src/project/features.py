"""Feature-selection and feature-substitution policies for model strategies.

A feature policy is the declarative answer to two questions about one model
strategy: which feature columns does the model train on, and in which order
does the estimator receive them? The four policies here cover the template's
strategies:

``IdentityFeatures``
    Pass the split tables through unchanged; the model feature list is the
    full feature schema in head order. Draws nothing from the global RNG.
``OutcomePlusNoise``
    Replace the first feature column with the outcome plus a small seeded
    normal draw, leave the second feature unchanged, and expose the model
    order ``[second, first]``. One draw per partition, in train -> valid ->
    test order.
``PureNoise``
    Replace the first two feature columns with pure seeded draws and expose
    the first two feature columns in head order. Two draws per partition, in
    train -> valid -> test order.
``FeatureSubset``
    Select an explicit tuple of feature columns, preserving the feature head
    order rather than the order the tuple was written in. Draws nothing.

The substitution policies use the legacy global ``numpy`` RNG on purpose:
the pipeline seeds that RNG once before training, so the exact draw order is
part of the value contract. Reordering a draw changes every later seeded
value and therefore the recorded baseline. The tests pin the per-partition
draw sequence before the training driver consumes these policies.

``apply(split_tables) -> split_tables`` is the policy's only consumed entry
point. It returns a fresh split-table value whose key and outcome frames are
preserved; a substitution policy replaces the partition frames, while a
selection policy projects the feature head and leaves the partition frames
full for the training driver to slice by name. ``feature_colnames`` exposes
the model-order feature list next to the substitution or selection that
defines it, so the draw order and the estimator input order stay in one place.

The split-table shape is expressed structurally with :class:`SplitTablesLike`
rather than imported from the artifact-I/O module. That keeps this module
independent of artifact loading and lets the concrete ``SplitTables`` value
added by the project artifact layer satisfy the same contract without a
circular import.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

import numpy as np
import polars as pl


class SplitTablesLike(Protocol):
    """The structural split-table shape every feature policy operates on.

    The concrete project artifact value carries the two schema/head frames,
    the three holdout partition frames, and the key-column vocabulary. A
    policy reads those attributes and returns a fresh value of the same
    concrete type through :func:`dataclasses.replace`; it never mutates the
    incoming value.
    """

    @property
    def feature_head_df(self) -> pl.DataFrame: ...

    @property
    def outcome_head_df(self) -> pl.DataFrame: ...

    @property
    def train_df(self) -> pl.DataFrame: ...

    @property
    def valid_df(self) -> pl.DataFrame: ...

    @property
    def test_df(self) -> pl.DataFrame: ...

    @property
    def key_colnames(self) -> Sequence[str]: ...


class FeaturePolicy(Protocol):
    """The feature-policy interface shared by every model strategy.

    A policy maps one outcome's split tables to the tables the model trains
    on and declares the model-order feature column list the training driver
    must hand to the estimator. Policies are stateless and frozen: the same
    instance is safe to reuse across runs, and only the incoming split tables
    plus the global numpy RNG influence the result.
    """

    def apply(self, split_tables: SplitTablesLike) -> SplitTablesLike:
        """Return the split tables with this policy applied.

        The feature columns may be projected or substituted, and a
        substitution may replace the partition frames. Keys, outcome columns,
        and their order are preserved. A substitution policy must draw from
        the global numpy RNG in the exact train -> valid -> test order its
        model requires.
        """
        ...

    def feature_colnames(self, split_tables: SplitTablesLike) -> list[str]:
        """Return the model feature columns in estimator input order."""
        ...


@dataclass(frozen=True, slots=True)
class IdentityFeatures:
    """No substitution: the model trains on the split tables verbatim.

    Draws nothing from the global RNG, so the partition frames and the feature
    list are exactly what the split stage wrote.
    """

    def apply(self, split_tables: SplitTablesLike) -> SplitTablesLike:
        """Return the split tables unchanged (no draw, no re-frame)."""
        return split_tables

    def feature_colnames(self, split_tables: SplitTablesLike) -> list[str]:
        """Return the full feature schema in head order."""
        return _input_feature_colnames(split_tables)


@dataclass(frozen=True, slots=True)
class OutcomePlusNoise:
    """Outcome-plus-noise substitution for a dummy baseline model.

    The first feature becomes the outcome plus a tiny normal draw, while the
    second feature stays verbatim. One ``normal(loc=0, scale=0.002)`` draw per
    partition, in train -> valid -> test order, so the seeded values depend on
    exactly three global RNG draws.
    """

    def apply(self, split_tables: SplitTablesLike) -> SplitTablesLike:
        """Substitute the first feature column with outcome plus noise.

        Raises:
            ValueError: The feature head declares fewer than two feature
                columns, or the outcome column cannot be resolved.
        """
        feature_names = _require_two_features(split_tables, "OutcomePlusNoise")
        fake_feature_colname = feature_names[0]
        outcome_colname = _outcome_colname(split_tables)
        _assert_columns_available(
            split_tables,
            [fake_feature_colname, outcome_colname],
        )

        substituted_frames = []
        for frame in _partition_frames(split_tables):
            noise = np.random.normal(
                loc=0,
                scale=0.002,
                size=frame.shape[0],
            )
            substituted_frames.append(
                frame.with_columns(
                    (pl.col(outcome_colname) + pl.Series(noise)).alias(
                        fake_feature_colname
                    )
                )
            )

        return _with_substituted_frames(split_tables, substituted_frames)

    def feature_colnames(self, split_tables: SplitTablesLike) -> list[str]:
        """Return the model order ``[second, first]``.

        The swap is model data: the estimator sees the unchanged second
        feature first, then the outcome-plus-noise feature, so the training
        driver must not fall back to the head order.
        """
        feature_names = _require_two_features(split_tables, "OutcomePlusNoise")
        return [feature_names[1], feature_names[0]]


@dataclass(frozen=True, slots=True)
class PureNoise:
    """Pure-noise substitution for a negative-control model.

    The first two feature columns are replaced with noise that carries no
    information about the outcome. Two draws per partition — first
    ``normal(loc=0, scale=0.2)``, then
    ``rint(normal(loc=100, scale=5).astype(int))`` — in train -> valid -> test
    order, so the seeded values depend on exactly six normal draws and three
    ``rint`` calls.
    """

    def apply(self, split_tables: SplitTablesLike) -> SplitTablesLike:
        """Replace the first two feature columns with pure-noise draws.

        Raises:
            ValueError: The feature head declares fewer than two feature
                columns.
        """
        feature_names = _require_two_features(split_tables, "PureNoise")
        fake_feature_colname_1 = feature_names[0]
        fake_feature_colname_2 = feature_names[1]
        _assert_columns_available(
            split_tables,
            [fake_feature_colname_1, fake_feature_colname_2],
        )

        substituted_frames = []
        for frame in _partition_frames(split_tables):
            noise_1 = np.random.normal(
                loc=0,
                scale=0.2,
                size=frame.shape[0],
            )
            noise_2 = np.rint(
                np.random.normal(
                    loc=100,
                    scale=5,
                    size=frame.shape[0],
                ).astype(int)
            )
            substituted_frames.append(
                frame.with_columns(
                    pl.Series(noise_1).alias(fake_feature_colname_1),
                    pl.Series(noise_2).alias(fake_feature_colname_2),
                )
            )

        return _with_substituted_frames(split_tables, substituted_frames)

    def feature_colnames(self, split_tables: SplitTablesLike) -> list[str]:
        """Return the first two feature columns in head order."""
        feature_names = _require_two_features(split_tables, "PureNoise")
        return feature_names[:2]


@dataclass(frozen=True, slots=True)
class FeatureSubset:
    """Select an explicit tuple of feature columns.

    The model feature order is the feature head order, not the order the
    requested tuple was written in. Selection projects the feature head to the
    key columns plus the selected feature columns; the partition frames stay
    full so the training driver can slice them by the projected head.

    Attributes:
        columns: The feature columns to keep. Must be non-empty and free of
            duplicates; unknown columns are rejected when the policy is
            applied to a split-table value.

    Raises:
        ValueError: ``columns`` is empty, contains a duplicate, or contains an
            empty name.
    """

    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize and validate the declared selection."""
        columns = tuple(self.columns)
        object.__setattr__(self, "columns", columns)
        if not columns:
            raise ValueError(
                "FeatureSubset needs at least one feature column; an empty "
                "selection would train on no features"
            )
        if any(not name for name in columns):
            raise ValueError(
                f"FeatureSubset column names must be non-empty; got {columns!r}"
            )
        duplicates = sorted({name for name in columns if columns.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"FeatureSubset lists column(s) more than once: {duplicates}; "
                "each selected feature is one model input"
            )

    def apply(self, split_tables: SplitTablesLike) -> SplitTablesLike:
        """Project the feature head to the keys plus the selected columns.

        Raises:
            ValueError: A requested column is not a feature column in the
                split tables.
        """
        selected = _selected_subset_colnames(split_tables, self.columns)
        return _with_feature_projection(split_tables, selected)

    def feature_colnames(self, split_tables: SplitTablesLike) -> list[str]:
        """Return the selected feature columns in feature head order.

        Raises:
            ValueError: A requested column is not a feature column in the
                split tables.
        """
        return _selected_subset_colnames(split_tables, self.columns)


def _partition_frames(split_tables: SplitTablesLike) -> tuple[pl.DataFrame, ...]:
    """Return the partition frames in the one order policies must draw in."""
    return (split_tables.train_df, split_tables.valid_df, split_tables.test_df)


def _input_feature_colnames(split_tables: SplitTablesLike) -> list[str]:
    """Return the feature schema columns in head order, keys excluded."""
    key_colnames = set(split_tables.key_colnames)
    return [
        colname
        for colname in split_tables.feature_head_df.columns
        if colname not in key_colnames
    ]


def _require_two_features(
    split_tables: SplitTablesLike,
    policy_name: str,
) -> list[str]:
    """Return the feature names, rejecting a head with fewer than two."""
    feature_names = _input_feature_colnames(split_tables)
    if len(feature_names) < 2:
        raise ValueError(
            f"{policy_name} needs at least two feature columns; "
            f"feature_head_df declares {feature_names!r}"
        )
    return feature_names


def _outcome_colname(split_tables: SplitTablesLike) -> str:
    """Return the one non-key outcome column in ``outcome_head_df``.

    The split stage writes the outcome head as the keys plus the modeled
    outcome column, so deriving the name from the split tables keeps the
    policies independent of the outcome registry.

    Raises:
        ValueError: The outcome head does not carry exactly one non-key
            column.
    """
    key_colnames = set(split_tables.key_colnames)
    outcome_colnames = [
        colname
        for colname in split_tables.outcome_head_df.columns
        if colname not in key_colnames
    ]
    if len(outcome_colnames) != 1:
        raise ValueError(
            "expected exactly one outcome column in outcome_head_df, got "
            f"{outcome_colnames!r}"
        )
    return outcome_colnames[0]


def _selected_subset_colnames(
    split_tables: SplitTablesLike,
    requested: Sequence[str],
) -> list[str]:
    """Return the requested feature columns in feature head order.

    Raises:
        ValueError: A requested name is not a feature column, or the
            selection is empty.
    """
    feature_names = _input_feature_colnames(split_tables)
    available = set(feature_names)
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(
            f"FeatureSubset requested column(s) that are not feature columns: "
            f"{unknown}; available feature columns: {feature_names!r}"
        )
    requested_set = set(requested)
    selected = [name for name in feature_names if name in requested_set]
    if not selected:
        raise ValueError(
            "FeatureSubset selection is empty; requested columns: "
            f"{list(requested)!r}, available feature columns: {feature_names!r}"
        )
    return selected


def _assert_columns_available(
    split_tables: SplitTablesLike,
    colnames: Sequence[str],
) -> None:
    """Raise if a column a substitution overwrites or reads is missing.

    The split stage guarantees the columns exist, so this is a contract guard
    on the calling shape rather than a recovery path: a policy consuming a
    split set with a drifted schema fails with the column name instead of a
    polars ``ColumnNotFoundError`` mid-substitution.
    """
    for frame in _partition_frames(split_tables):
        missing = [name for name in colnames if name not in frame.columns]
        if missing:
            raise ValueError(
                f"split frame columns {list(frame.columns)!r} are missing "
                f"{missing!r}; the feature/outcome schemas must match the "
                "partition frames"
            )


def _with_substituted_frames(
    split_tables: SplitTablesLike,
    substituted_frames: Sequence[pl.DataFrame],
) -> SplitTablesLike:
    """Return a fresh split-table value carrying substituted partitions.

    The head frames and key-column vocabulary are preserved; the key list is
    copied so the frozen value never shares a mutable list with the caller.
    """
    train_df, valid_df, test_df = substituted_frames
    return cast(
        SplitTablesLike,
        replace(
            cast(Any, split_tables),
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            key_colnames=list(split_tables.key_colnames),
        ),
    )


def _with_feature_projection(
    split_tables: SplitTablesLike,
    selected_colnames: Sequence[str],
) -> SplitTablesLike:
    """Return a fresh split-table value with a projected feature head.

    The keys stay in the head because the training driver slices every
    partition frame by the head columns and then sorts on the keys. The
    partition frames stay full, and the outcome head plus key vocabulary are
    preserved verbatim.
    """
    selected = set(selected_colnames)
    key_colnames = set(split_tables.key_colnames)
    head_colnames = [
        colname
        for colname in split_tables.feature_head_df.columns
        if colname in key_colnames or colname in selected
    ]
    return cast(
        SplitTablesLike,
        replace(
            cast(Any, split_tables),
            feature_head_df=split_tables.feature_head_df.select(head_colnames),
            key_colnames=list(split_tables.key_colnames),
        ),
    )
