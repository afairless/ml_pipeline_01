"""Unit tests: the s06 cross-model regression display table.

Pins the display contract: rows sorted by model name, one column per expanded
metric field and evaluated partition with each field's partitions adjacent,
numeric rounding to the configured precision, regression-task models only,
and a header-only CSV when an outcome has no regression models. The project
config names metric suffixes, so the table renders both the original-scale
and model-scale field for every entry.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from src.project.config import (
    DISPLAY_METRIC_DECIMALS,
    DISPLAY_REGRESSION_METRICS,
)
from src.stages.s06_evaluate_model import (
    build_regression_display_table,
    record_regression_metrics,
    save_regression_display_table,
)

PARTITIONS = ("train", "valid")
METRIC_FIELDS = [
    field
    for metric_name in DISPLAY_REGRESSION_METRICS
    for field in (f"original_target_{metric_name}", f"model_target_{metric_name}")
]
EXPECTED_COLUMNS = [
    "model",
    *[f"{field}_{partition}" for field in METRIC_FIELDS for partition in PARTITIONS],
]


def _values(offset: float) -> dict[str, float]:
    """Distinct values for every displayed field on one partition."""

    values: dict[str, float] = {}
    for index, field in enumerate(METRIC_FIELDS):
        values[field] = round(offset + index + 0.123456, 6)
    return values


MODEL_A_TRAIN = _values(0.0)
MODEL_A_VALID = _values(10.0)
MODEL_B_TRAIN = _values(20.0)
MODEL_B_VALID = _values(30.0)


def _fake_metrics(values: dict[str, float]) -> Any:
    """A ``RegressionMetrics``-shaped object with only readable attributes.

    Undisplayed fields are deliberately present so a builder that ignored the
    configured metric list would leak them into the table.
    """

    attributes: dict[str, object] = dict(values)
    attributes.update(
        {
            "some_hidden_metric": 99.9999,
            "model_target_pred": None,
            "model_target_true": None,
            "original_target_pred": None,
            "original_target_true": None,
        }
    )
    return SimpleNamespace(**attributes)


def _metrics_by_model() -> dict[str, dict[str, Any]]:
    """Two models supplied in reverse name order, each with both partitions."""

    return {
        "model_b": {
            "train": _fake_metrics(MODEL_B_TRAIN),
            "valid": _fake_metrics(MODEL_B_VALID),
        },
        "model_a": {
            "train": _fake_metrics(MODEL_A_TRAIN),
            "valid": _fake_metrics(MODEL_A_VALID),
        },
    }


def _expected_cell(values: dict[str, float], field: str) -> float:
    return round(values[field], DISPLAY_METRIC_DECIMALS)


def test_build_orders_columns_rounds_and_sorts_models() -> None:
    """Columns are field-then-partition, rows ascending, values rounded."""

    display_df = build_regression_display_table(
        _metrics_by_model(),
        PARTITIONS,
        DISPLAY_REGRESSION_METRICS,
        DISPLAY_METRIC_DECIMALS,
    )

    assert display_df.columns == EXPECTED_COLUMNS
    assert display_df["model"].to_list() == ["model_a", "model_b"]

    for row_index, (train_values, valid_values) in enumerate(
        (
            (MODEL_A_TRAIN, MODEL_A_VALID),
            (MODEL_B_TRAIN, MODEL_B_VALID),
        )
    ):
        for field in METRIC_FIELDS:
            assert display_df[f"{field}_train"][row_index] == _expected_cell(
                train_values, field
            )
            assert display_df[f"{field}_valid"][row_index] == _expected_cell(
                valid_values, field
            )


def test_build_excludes_undeclared_metrics() -> None:
    """No metric outside the configured suffix list becomes a column."""

    display_df = build_regression_display_table(
        _metrics_by_model(),
        PARTITIONS,
        DISPLAY_REGRESSION_METRICS,
        DISPLAY_METRIC_DECIMALS,
    )

    assert not any("some_hidden_metric" in column for column in display_df.columns)
    assert "model_target_pred_train" not in display_df.columns


def test_build_raises_on_missing_partition() -> None:
    """A model missing a configured partition fails, naming both."""

    metrics_by_model = _metrics_by_model()
    del metrics_by_model["model_a"]["valid"]

    with pytest.raises(ValueError, match="model_a.*valid"):
        build_regression_display_table(
            metrics_by_model,
            PARTITIONS,
            DISPLAY_REGRESSION_METRICS,
            DISPLAY_METRIC_DECIMALS,
        )


def test_build_raises_on_unexpected_partition() -> None:
    """A partition outside the configured set fails, naming both."""

    metrics_by_model = _metrics_by_model()
    metrics_by_model["model_a"]["test"] = _fake_metrics(MODEL_A_VALID)

    with pytest.raises(ValueError, match="model_a.*test"):
        build_regression_display_table(
            metrics_by_model,
            PARTITIONS,
            DISPLAY_REGRESSION_METRICS,
            DISPLAY_METRIC_DECIMALS,
        )


def test_build_empty_input_is_header_only_with_declared_dtypes() -> None:
    """No regression models still yields the full typed, rowless frame."""

    display_df = build_regression_display_table(
        {}, PARTITIONS, DISPLAY_REGRESSION_METRICS, DISPLAY_METRIC_DECIMALS
    )

    assert display_df.height == 0
    assert display_df.columns == EXPECTED_COLUMNS
    assert display_df.schema["model"] == pl.String
    for column in EXPECTED_COLUMNS[1:]:
        assert display_df.schema[column] == pl.Float64


def test_build_accepts_prefixed_metric_names_as_is() -> None:
    """An already-prefixed field name is not expanded a second time."""

    display_df = build_regression_display_table(
        _metrics_by_model(),
        PARTITIONS,
        ("model_target_r2",),
        DISPLAY_METRIC_DECIMALS,
    )

    assert display_df.columns == [
        "model",
        "model_target_r2_train",
        "model_target_r2_valid",
    ]


def test_save_writes_one_csv_that_round_trips(tmp_path: Path) -> None:
    """The writer creates exactly the requested CSV, with no sidecar copy."""

    display_df = build_regression_display_table(
        _metrics_by_model(),
        PARTITIONS,
        DISPLAY_REGRESSION_METRICS,
        DISPLAY_METRIC_DECIMALS,
    )
    output_path = tmp_path / "outcome_1_display.csv"

    save_regression_display_table(display_df, output_path)

    assert list(tmp_path.iterdir()) == [output_path]
    read_df = pl.read_csv(output_path)
    assert read_df.columns == display_df.columns
    assert read_df.equals(display_df)


def test_record_regression_metrics_keeps_regression_only() -> None:
    """A regression task is recorded; a binary task is omitted entirely."""

    metrics_by_model: dict[str, dict[str, Any]] = {}
    metrics = _fake_metrics(MODEL_A_TRAIN)

    record_regression_metrics(
        "binary", "model_binary", "train", metrics, metrics_by_model
    )
    assert metrics_by_model == {}

    record_regression_metrics(
        "regression", "model_a", "train", metrics, metrics_by_model
    )
    assert metrics_by_model == {"model_a": {"train": metrics}}


def test_record_regression_metrics_rejects_duplicate_pair() -> None:
    """Recording the same model/partition twice fails, naming both."""

    metrics_by_model: dict[str, dict[str, Any]] = {}
    metrics = _fake_metrics(MODEL_A_TRAIN)
    record_regression_metrics(
        "regression", "model_a", "train", metrics, metrics_by_model
    )

    with pytest.raises(ValueError, match="model_a.*train"):
        record_regression_metrics(
            "regression", "model_a", "train", metrics, metrics_by_model
        )
