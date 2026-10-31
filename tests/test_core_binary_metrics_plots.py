"""Unit tests for the figure-owning binary-metric plot helpers.

These tests are hermetic: they force the Agg backend before importing the
module under test, clean up every opened figure after each test, and use small
deterministic arrays only. They pin the figure-ownership rule for the confusion
matrix, ROC, and precision-recall helpers: each helper writes a non-empty PNG
on the figure it owns and closes that figure, and never draws into or closes an
unrelated caller-owned sentinel figure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from src.core import binary_metrics as core_binary_metrics  # noqa: E402

PlotCall = Callable[..., None]


@pytest.fixture(autouse=True)
def _close_all_figures_after_each_test() -> Any:
    """Never let a leaked figure contaminate the next test."""
    yield
    plt.close("all")


def make_binary_data(n: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return small deterministic binary labels and separated probabilities.

    Both classes are guaranteed present so the sklearn displays receive a
    well-formed input, and the probabilities are split around 0.5 by class so a
    threshold recovers the labels exactly.
    """
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_true[0] = 0
    y_true[1] = 1
    y_pred_proba = np.where(
        y_true == 1,
        rng.uniform(0.6, 0.99, size=n),
        rng.uniform(0.01, 0.4, size=n),
    )
    return y_true, y_pred_proba


def assert_png_written_no_figure_open(output_filepath: Path) -> None:
    """The helper wrote a non-empty PNG and closed every figure it made."""
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


def run_confusion_matrix(
    y_true: np.ndarray, y_pred_proba: np.ndarray, output_filepath: Path
) -> None:
    """Call ``plot_confusion_matrix`` with a 0.5 classification threshold."""
    core_binary_metrics.plot_confusion_matrix(
        y_true, y_pred_proba, 0.5, output_filepath
    )


def run_confusion_matrix_2(
    y_true: np.ndarray, y_pred_proba: np.ndarray, output_filepath: Path
) -> None:
    """Call ``plot_confusion_matrix_2`` with thresholded predictions."""
    core_binary_metrics.plot_confusion_matrix_2(
        y_true, y_pred_proba > 0.5, output_filepath
    )


def run_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_filepath: Path,
    plot_balance: bool = True,
) -> None:
    """Call ``plot_roc_curve`` with observed-point and prevalence scalars."""
    core_binary_metrics.plot_roc_curve(
        y_true,
        y_pred_proba,
        tpr=0.9,
        fpr=0.2,
        prevalence=0.5,
        output_filepath=output_filepath,
        plot_balance=plot_balance,
    )


def run_precision_recall_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, output_filepath: Path
) -> None:
    """Call ``plot_precision_recall_curve`` with observed-point scalars."""
    core_binary_metrics.plot_precision_recall_curve(
        y_true,
        y_pred_proba,
        recall=0.9,
        precision=0.8,
        informedness=0.8,
        markedness=0.7,
        prevalence=0.5,
        output_filepath=output_filepath,
    )


PLOT_CALLS = [
    pytest.param("confusion_matrix", run_confusion_matrix, {}, id="confusion_matrix"),
    pytest.param(
        "confusion_matrix_2", run_confusion_matrix_2, {}, id="confusion_matrix_2"
    ),
    pytest.param("roc_curve_balance", run_roc_curve, {}, id="roc_curve_balance"),
    pytest.param(
        "roc_curve_chance",
        run_roc_curve,
        {"plot_balance": False},
        id="roc_curve_chance",
    ),
    pytest.param(
        "precision_recall_curve",
        run_precision_recall_curve,
        {},
        id="precision_recall_curve",
    ),
]


@pytest.mark.parametrize("stem,call,kwargs", PLOT_CALLS)
def test_plot_writes_non_empty_png_and_leaves_no_figure(
    stem: str, call: PlotCall, kwargs: dict[str, Any], tmp_path: Path
) -> None:
    """Each binary-metric plot writes a non-empty PNG and leaves no figure."""
    y_true, y_pred_proba = make_binary_data()
    output_filepath = tmp_path / f"{stem}.png"
    call(y_true, y_pred_proba, output_filepath, **kwargs)
    assert_png_written_no_figure_open(output_filepath)


@pytest.mark.parametrize("stem,call,kwargs", PLOT_CALLS)
def test_sentinel_figure_survives_and_is_not_drawn_into(
    stem: str, call: PlotCall, kwargs: dict[str, Any], tmp_path: Path
) -> None:
    """No binary-metric helper draws into or closes a caller-owned figure.

    A distinct sentinel figure stays open with its original axes and no new
    artists while each helper draws on, saves, and closes the figure it owns.
    """
    sentinel = plt.figure()
    sentinel_ax = sentinel.add_subplot(111)
    sentinel_ax.set_title("sentinel title")
    sentinel_ax.set_xlabel("sentinel x")

    y_true, y_pred_proba = make_binary_data()
    output_filepath = tmp_path / f"{stem}.png"
    call(y_true, y_pred_proba, output_filepath, **kwargs)

    # The sentinel is the only figure left open and carries no new artists.
    assert plt.get_fignums() == [sentinel.number]
    assert len(sentinel.axes) == 1
    assert sentinel.axes[0].get_title() == "sentinel title"
    assert sentinel.axes[0].get_xlabel() == "sentinel x"
    assert len(sentinel.axes[0].lines) == 0
    assert len(sentinel.axes[0].collections) == 0
    assert len(sentinel.axes[0].patches) == 0


def test_confusion_matrix_plot_single_class_input(tmp_path: Path) -> None:
    """A single-class confusion input still plots on an owned figure.

    ``confusion_matrix`` against explicit labels handles a degenerate
    single-class input; the plot helper must do the same, writing its PNG and
    closing its own figure without raising.
    """
    y_true = np.ones(20, dtype=int)
    y_pred_proba = np.full(20, 0.8)
    output_filepath = tmp_path / "confusion_single_class.png"

    core_binary_metrics.plot_confusion_matrix(
        y_true, y_pred_proba, 0.5, output_filepath
    )

    assert_png_written_no_figure_open(output_filepath)
