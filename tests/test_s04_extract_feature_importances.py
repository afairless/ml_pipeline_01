"""Unit and property tests for feature-importance extraction.

``extract_feature_importances`` reads a model's gain scores and maps the
estimator's positional feature identifiers (``f0``, ``f1``, ...) back to the
training column names. A model that grew no splits reports no gain scores; the
extractor substitutes a small positive dummy gain per declared feature so the
importance contract still holds and the ranking has one row per input feature.

The property-based tests assert the parts that must hold for every feature
count and every positive gain dictionary: every declared feature keeps its
name in estimator order, every emitted importance is positive, and the
empty-gain fallback emits one row per feature.
"""

from __future__ import annotations

from typing import cast

import polars as pl
import pytest
import xgboost as xgb
from hypothesis import given
from hypothesis import strategies as st

from src.project.artifacts import extract_feature_importances
from src.project.columns import (
    FEATURE_COLNAME,
    FEATURE_ESTIMATOR_ID_COLNAME,
    FEATURE_IMPORTANCE_COLNAME,
)


class _SplitlessModel:
    """Stand-in for an ``xgb.Booster`` whose gain scores are empty."""

    def get_score(self, importance_type: str) -> dict[str, float]:
        assert importance_type == "gain"
        return {}


class _GainModel:
    """Stand-in for an ``xgb.Booster`` reporting the supplied gain scores."""

    def __init__(self, gains: dict[str, float]) -> None:
        self._gains = gains

    def get_score(self, importance_type: str) -> dict[str, float]:
        assert importance_type == "gain"
        return self._gains


def _feature_frame(feature_count: int) -> pl.DataFrame:
    """A feature frame whose column names follow the demo feature vocabulary."""
    return pl.DataFrame(
        {f"feature_{index + 1}": [1.0, 2.0, 3.0] for index in range(feature_count)}
    )


def test_empty_gain_falls_back_to_dummy_importances() -> None:
    feature_df = _feature_frame(2)

    out = extract_feature_importances(feature_df, cast(xgb.Booster, _SplitlessModel()))

    assert out.columns == [
        FEATURE_COLNAME,
        FEATURE_ESTIMATOR_ID_COLNAME,
        FEATURE_IMPORTANCE_COLNAME,
    ]
    assert out[FEATURE_COLNAME].to_list() == ["feature_1", "feature_2"]
    assert out[FEATURE_ESTIMATOR_ID_COLNAME].to_list() == ["f0", "f1"]
    # Each dummy gain clears the producer's strict positivity check.
    assert out[FEATURE_IMPORTANCE_COLNAME].to_list() == [0.00001, 0.00001]


def test_nonempty_gain_passes_through_unchanged() -> None:
    feature_df = _feature_frame(2)
    gains = {"f1": 0.8, "f0": 0.2}

    out = extract_feature_importances(feature_df, cast(xgb.Booster, _GainModel(gains)))

    # The extraction preserves the model's score order rather than re-sorting.
    assert out[FEATURE_COLNAME].to_list() == ["feature_2", "feature_1"]
    assert out[FEATURE_ESTIMATOR_ID_COLNAME].to_list() == ["f1", "f0"]
    assert out[FEATURE_IMPORTANCE_COLNAME].to_list() == [0.8, 0.2]


def test_unknown_feature_identifier_raises() -> None:
    feature_df = _feature_frame(1)

    with pytest.raises(ValueError, match="feature id"):
        extract_feature_importances(
            feature_df, cast(xgb.Booster, _GainModel({"f1": 1.0}))
        )


def test_non_positive_gain_raises() -> None:
    feature_df = _feature_frame(1)

    with pytest.raises(ValueError, match="greater than"):
        extract_feature_importances(
            feature_df, cast(xgb.Booster, _GainModel({"f0": 0.0}))
        )


def test_no_feature_columns_raises() -> None:
    with pytest.raises(ValueError, match="at least one column"):
        extract_feature_importances(
            pl.DataFrame(), cast(xgb.Booster, _SplitlessModel())
        )


@given(
    gains=st.lists(
        st.floats(
            min_value=1e-5,
            max_value=1e9,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=8,
    )
)
def test_every_reported_gain_keeps_its_positional_feature_name(
    gains: list[float],
) -> None:
    """Any non-empty gain vector maps one-to-one onto the feature names."""
    feature_names = [f"feature_{index + 1}" for index in range(len(gains))]
    feature_df = pl.DataFrame({name: [1.0, 2.0] for name in feature_names})
    model = _GainModel({f"f{index}": gain for index, gain in enumerate(gains)})

    out = extract_feature_importances(feature_df, cast(xgb.Booster, model))

    assert out[FEATURE_COLNAME].to_list() == feature_names
    assert out[FEATURE_ESTIMATOR_ID_COLNAME].to_list() == [
        f"f{index}" for index in range(len(gains))
    ]
    assert out[FEATURE_IMPORTANCE_COLNAME].to_list() == gains
    assert bool((out[FEATURE_IMPORTANCE_COLNAME] > 0.0).all())


@given(feature_count=st.integers(min_value=1, max_value=8))
def test_empty_gain_fallback_covers_every_feature(feature_count: int) -> None:
    """A splitless model still yields one positive row per declared feature."""
    feature_df = _feature_frame(feature_count)

    out = extract_feature_importances(feature_df, cast(xgb.Booster, _SplitlessModel()))

    assert out.height == feature_count
    assert out[FEATURE_COLNAME].to_list() == list(feature_df.columns)
    assert out[FEATURE_ESTIMATOR_ID_COLNAME].to_list() == [
        f"f{index}" for index in range(feature_count)
    ]
    assert out[FEATURE_IMPORTANCE_COLNAME].to_list() == [0.00001] * feature_count
