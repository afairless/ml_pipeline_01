"""Demo outcome and feature producers — the pipeline's extension point.

This module is the seam between an external source table and the pipeline's
generic stages. The two builders below are the only functions most users need
to replace: a new project supplies its own outcome table and feature matrix
there and leaves the stage code, schemas, and orchestration untouched. The
returned frames are ordinary ``polars`` dataframes that already satisfy the
alignment contract, so a stage can validate them against
:data:`~src.project.schemas.S01_OUTCOME_SCHEMA` and
:data:`~src.project.schemas.S02_FEATURE_SCHEMA` and write them immediately.

The demo implementations use deterministic arithmetic only, so the committed
golden baseline is reproducible. Each builder creates its own private
``numpy.random.default_rng(config.demo_seed)`` and never reads or mutates the
global ``numpy`` random state; the estimator is the only consumer of that
global state. ``build_outcome_table`` draws in the documented formula order —
for each source row, the ``outcome_1`` noise draw precedes the ``outcome_3``
draw — while ``build_feature_table`` creates its fresh generator and makes no
draws at all, because every demo feature is a deterministic function of the
source values.

The alignment contract shared by both builders: the result is sorted by
:data:`~src.project.columns.KEY_COLUMNS` and carries one key row per source
row. ``build_feature_table`` receives the outcome frame so a user adapter can
derive features from outcomes, but the demo implementation deliberately
ignores it; the feature stage separately asserts that the feature keys equal
the outcome keys exactly, so a mismatch fails there instead of being hidden
here.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from src.project.columns import CASE_DATE, CASE_ID, KEY_COLUMNS
from src.project.config import ProjectConfig
from src.project.schemas import FEATURE_COLUMNS

_SORT_COLUMNS = list(KEY_COLUMNS)
"""Key columns in canonical sort order; polars takes a list of names."""


def _ordered(source_df: pl.DataFrame) -> pl.DataFrame:
    """Return ``source_df`` sorted by the canonical key columns.

    Sorting before the outcome loop makes the draw sequence a function of the
    row set rather than of the input row order: two frames with the same rows
    produce the same outcomes even if one was shuffled.
    """
    return source_df.sort(_SORT_COLUMNS)


def build_outcome_table(
    source_df: pl.DataFrame,
    config: ProjectConfig,
) -> pl.DataFrame:
    """Build the demo outcome table from the source table.

    Four outcomes are derived per source row, in the order documented by the
    demo specification:

    - ``outcome_1`` — a weighted mix of ``value_1`` and ``value_2`` plus a
      private seeded noise draw;
    - ``outcome_2`` — ``value_1 - value_3``, draw-free;
    - ``outcome_3`` — a product/interaction term plus a private seeded noise
      draw;
    - ``outcome_4`` — ``value_3``, draw-free.

    The private generator is seeded from ``config.demo_seed``. Draws are
    consumed row by row in the formula order above, so the outcome columns are
    reproducible for a fixed source table and seed. The global ``numpy``
    random state is never touched.

    Args:
        source_df: A source table carrying ``case_id``, ``case_date``, and the
            ``value_1``/``value_2``/``value_3`` raw columns. It is not mutated.
        config: Project configuration supplying ``demo_seed``.

    Returns:
        A frame with the canonical key columns plus ``outcome_1`` through
        ``outcome_4``, sorted by :data:`~src.project.columns.KEY_COLUMNS` and
        carrying one row per source row.
    """
    generator = np.random.default_rng(config.demo_seed)
    case_ids: list[str] = []
    case_dates: list[date] = []
    outcomes_1: list[float] = []
    outcomes_2: list[float] = []
    outcomes_3: list[float] = []
    outcomes_4: list[float] = []

    for row in _ordered(source_df).iter_rows(named=True):
        value_1 = row["value_1"]
        value_2 = row["value_2"]
        value_3 = row["value_3"]

        case_ids.append(row[CASE_ID])
        case_dates.append(row[CASE_DATE])
        outcomes_1.append(0.60 * value_1 + 0.30 * value_2 + 0.10 * generator.normal())
        outcomes_2.append(value_1 - value_3)
        outcomes_3.append(
            value_1 * value_2 + 0.30 * value_3 + 0.05 * generator.normal()
        )
        outcomes_4.append(value_3)

    frame = pl.DataFrame(
        {
            CASE_ID: pl.Series(case_ids, dtype=pl.String),
            CASE_DATE: pl.Series(case_dates, dtype=pl.Date),
            "outcome_1": pl.Series(outcomes_1, dtype=pl.Float64),
            "outcome_2": pl.Series(outcomes_2, dtype=pl.Float64),
            "outcome_3": pl.Series(outcomes_3, dtype=pl.Float64),
            "outcome_4": pl.Series(outcomes_4, dtype=pl.Float64),
        }
    )
    return frame.sort(_SORT_COLUMNS)


def _feature_expressions() -> tuple[pl.Expr, ...]:
    """Return the eight demo feature expressions, in ``FEATURE_COLUMNS`` order.

    Every expression is a deterministic function of the source's
    ``value_1``/``value_2``/``value_3`` columns. ``feature_8``'s denominator is
    always positive, so the feature is finite even when ``value_3`` is
    negative.
    """
    value_1 = pl.col("value_1")
    value_2 = pl.col("value_2")
    value_3 = pl.col("value_3")
    return (
        value_1,
        value_2,
        value_3,
        value_1 * value_2,
        value_1**2,
        value_2**2,
        value_1 + value_2 + value_3,
        (value_1 - value_2) / (1 + value_3.abs()),
    )


def build_feature_table(
    source_df: pl.DataFrame,
    outcome_df: pl.DataFrame,
    config: ProjectConfig,
) -> pl.DataFrame:
    """Build the demo feature table from the source table.

    The eight features are deterministic expressions of the source values, so
    this builder makes no random draws. It still constructs a fresh private
    generator from ``config.demo_seed`` to mirror the outcome builder's
    contract and to give a replacing project a ready generator; the demo
    implementation never advances it.

    ``outcome_df`` is part of the extension-point signature so a user adapter
    can derive outcome-informed features. The demo formulas do not need it, so
    it is not read. The feature stage validates key alignment against the
    outcome table afterwards.

    Args:
        source_df: A source table carrying ``case_id``, ``case_date``, and the
            three raw value columns. It is not mutated.
        outcome_df: The outcome table; accepted for API compatibility and for
            user adapters, unused by the demo formulas.
        config: Project configuration supplying ``demo_seed``.

    Returns:
        A frame with the canonical key columns plus ``feature_1`` through
        ``feature_8``, sorted by :data:`~src.project.columns.KEY_COLUMNS` and
        carrying one row per source row.
    """
    _generator = np.random.default_rng(config.demo_seed)
    frame = source_df.select(
        pl.col(CASE_ID),
        pl.col(CASE_DATE),
        *(
            expression.alias(name)
            for expression, name in zip(
                _feature_expressions(), FEATURE_COLUMNS, strict=True
            )
        ),
    )
    return frame.sort(_SORT_COLUMNS)
