"""Unit tests for the figure-owning plotting helpers in ``src.core.plots``.

These tests are hermetic: they force the Agg backend before importing the
module under test, clean up every opened figure after each test, and use small
deterministic frames and arrays only. They pin figure ownership for every
helper, the joint helper's decorations and failure cleanup, the prediction
helper's per-style artifacts, the leak regression where a failed residual plot
used to corrupt a later artifact, and the always-write, null-filtering
behaviour of the distribution helpers.
"""

from __future__ import annotations

from typing import cast

import matplotlib

matplotlib.use("Agg", force=True)

from pathlib import Path

import matplotlib.axes
import matplotlib.figure
import numpy as np
import polars as pl
import pytest
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import complete
from scipy.spatial.distance import pdist

from src.core import plots as core_plots
from src.core.plots import JointPlotKind


@pytest.fixture(autouse=True)
def _close_all_figures_after_each_test():
    """Never let a leaked figure contaminate the next test."""
    yield
    plt.close("all")


def make_joint_df(n: int = 40, seed: int = 0) -> pl.DataFrame:
    """A small correlated frame with columns ``a`` and ``b``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pl.DataFrame({"a": x, "b": x + rng.normal(scale=0.5, size=n)})


def make_two_series_df(n: int = 30, seed: int = 0) -> pl.DataFrame:
    """A small frame for the two-series helper: x plus two correlated series."""
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=float)
    return pl.DataFrame(
        {
            "x": x,
            "actual": x + rng.normal(scale=0.5, size=n),
            "pred": x + rng.normal(scale=0.5, size=n),
        }
    )


def make_attribution_sample(
    n: int = 20, n_features: int = 4, seed: int = 0
) -> np.ndarray:
    """A small deterministic sample for the clustering helpers."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n_features))


def make_cluster_inputs(
    n: int = 20, n_features: int = 4, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """A sample plus the complete-linkage matrix the cluster helpers use.

    Mirrors ``plot_hierarchical_cluster_dendrogram``'s computation so the
    cluster helpers receive the same shape of linkage matrix the pipeline
    passes them.
    """
    sample = make_attribution_sample(n, n_features, seed)
    linkage = complete(pdist(sample))
    return sample, linkage


def capture_saved_figure(monkeypatch) -> dict:
    """Snapshot a figure's texts and lines at save time.

    Every helper closes its own figure right after saving, so a test must
    observe the figure through ``Figure.savefig`` rather than afterwards.
    """
    captured: dict = {}
    real_savefig = matplotlib.figure.Figure.savefig

    def capturing_savefig(self, *args, **kwargs):
        captured["fig"] = self
        captured["axes_texts"] = [
            [text.get_text() for text in ax.texts] for ax in self.axes
        ]
        captured["titles"] = [ax.get_title() for ax in self.axes]
        captured["xlabels"] = [ax.get_xlabel() for ax in self.axes]
        captured["lines"] = [
            [np.asarray(line.get_xdata()) for line in ax.lines] for ax in self.axes
        ]
        return real_savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capturing_savefig)
    return captured


@pytest.mark.parametrize("kind", ["scatter", "kde", "hist", "hex", "reg", "resid"])
def test_joint_kind_writes_non_empty_png_and_leaves_no_figure(
    tmp_path: Path, kind: JointPlotKind
) -> None:
    """Each supported joint kind writes a non-empty PNG and no open figure."""
    output_filepath = tmp_path / f"joint_{kind}.png"
    core_plots.plot_joint_with_rug_and_margin(
        make_joint_df(), "a", "b", kind, "suptitle", "title", output_filepath
    )
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


def test_joint_decorations_land_on_joint_axes(monkeypatch, tmp_path: Path) -> None:
    """Labels, title, limits, and reference lines go on the joint axes."""
    df = make_joint_df()
    captured = capture_saved_figure(monkeypatch)
    title = "the title"
    core_plots.plot_joint_with_rug_and_margin(
        df, "a", "b", "scatter", "the suptitle", title, tmp_path / "decor.png"
    )

    figure = captured["fig"]
    joint_axes = [ax for ax in figure.axes if ax.get_xlabel() == "a"]
    assert len(joint_axes) == 1
    ax = joint_axes[0]

    assert ax.get_ylabel() == "b"
    assert ax.get_title() == title

    x_values = df["a"].to_numpy()
    y_values = df["b"].to_numpy()
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    y_min = float(y_values.min())
    y_max = float(y_values.max())
    x_min_pad = x_min - 0.02 * (x_max - x_min)
    x_max_pad = x_max + 0.02 * (x_max - x_min)
    y_min_pad = y_min - 0.02 * (y_max - y_min)
    y_max_pad = y_max + 0.02 * (y_max - y_min)
    assert ax.get_xlim() == pytest.approx((x_min_pad, x_max_pad))
    assert ax.get_ylim() == pytest.approx((y_min_pad, y_max_pad))

    # identity line plus the two zero lines, all on the joint panel
    assert len(ax.get_lines()) >= 3
    marginal_axes = [other for other in figure.axes if other is not ax]
    assert all(len(other.get_lines()) == 0 for other in marginal_axes)
    assert plt.get_fignums() == []


def test_joint_unsupported_kind_raises_and_leaves_no_figure(
    tmp_path: Path,
) -> None:
    """An unsupported kind raises before writing and closes its figure."""
    output_filepath = tmp_path / "unsupported.png"
    with pytest.raises(ValueError, match="Unsupported"):
        core_plots.plot_joint_with_rug_and_margin(
            make_joint_df(),
            "a",
            "b",
            cast(JointPlotKind, "not-a-kind"),
            "suptitle",
            "title",
            output_filepath,
        )
    assert not output_filepath.exists()
    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame({"a": [], "b": []}),
        pl.DataFrame({"a": [None, None], "b": [1.0, 2.0]}),
        pl.DataFrame({"a": [1.0, 2.0], "b": [None, None]}),
    ],
)
def test_joint_empty_or_all_null_input_skips_artifact(
    df: pl.DataFrame, tmp_path: Path
) -> None:
    """Empty/all-null input prints, writes nothing, and does not raise."""
    output_filepath = tmp_path / "skip.png"
    core_plots.plot_joint_with_rug_and_margin(
        df, "a", "b", "scatter", "suptitle", "title", output_filepath
    )
    assert not output_filepath.exists()
    assert plt.get_fignums() == []


def test_joint_failure_closes_its_figure_and_reraises(
    monkeypatch, tmp_path: Path
) -> None:
    """A drawing failure leaves no new figures, keeps caller figures, re-raises."""
    sentinel = plt.figure()
    before = set(plt.get_fignums())

    def fake_joint_panel(*args, **kwargs):
        raise RuntimeError("simulated joint drawing failure")

    monkeypatch.setattr(core_plots, "_draw_joint_panel", fake_joint_panel)
    with pytest.raises(RuntimeError, match="simulated joint drawing failure"):
        core_plots.plot_joint_with_rug_and_margin(
            make_joint_df(),
            "a",
            "b",
            "scatter",
            "suptitle",
            "title",
            tmp_path / "fail.png",
        )
    assert set(plt.get_fignums()) == before
    assert plt.fignum_exists(sentinel.number)


def test_prediction_joint_plots_writes_all_kinds(tmp_path: Path) -> None:
    """The prediction helper writes scatter, kde, and residual PNGs."""
    df = make_joint_df().rename({"a": "true", "b": "pred"})
    core_plots.plot_prediction_joint_plots(
        df, "true", "pred", "param1", "joint title", "stem", tmp_path
    )
    for kind in ("scatter", "kde", "resid"):
        png = tmp_path / f"stem_pred_param1_{kind}.png"
        assert png.exists()
        assert png.stat().st_size > 0
    assert plt.get_fignums() == []


def test_failed_resid_then_scatter_bin_uses_fresh_figure(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """A failed residual plot leaks nothing, so the next helper gets its own axes.

    This pins the original corruption mechanism: before the ownership fix, a
    failed residual call left a partially built figure open and current, and
    the binned scatter then drew into foreign axes under mis-labelled titles.
    """
    real_joint_panel = core_plots._draw_joint_panel

    def fake_joint_panel(ax_joint, x_values, y_values, joint_plot_kind):
        if joint_plot_kind == "resid":
            raise RuntimeError("simulated resid failure")
        return real_joint_panel(ax_joint, x_values, y_values, joint_plot_kind)

    monkeypatch.setattr(core_plots, "_draw_joint_panel", fake_joint_panel)

    captured = {}
    real_savefig = matplotlib.figure.Figure.savefig

    def capturing_savefig(self, *args, **kwargs):
        captured["fig"] = self
        # the helper closes its own figure right after saving; snapshot the
        # axes state at save time, before the figure is disposed
        captured["n_axes"] = len(self.axes)
        captured["titles"] = [ax.get_title() for ax in self.axes]
        captured["xlabels"] = [ax.get_xlabel() for ax in self.axes]
        return real_savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capturing_savefig)

    df = make_joint_df().rename({"a": "true", "b": "pred"})
    core_plots.plot_prediction_joint_plots(
        df, "true", "pred", "p1", "joint title", "stem", tmp_path
    )
    assert (tmp_path / "stem_pred_p1_scatter.png").exists()
    assert (tmp_path / "stem_pred_p1_kde.png").exists()
    assert not (tmp_path / "stem_pred_p1_resid.png").exists()
    assert plt.get_fignums() == []

    captured.clear()
    rng = np.random.default_rng(1)
    x = pl.Series("x", rng.normal(size=30))
    y = pl.Series("y", rng.normal(size=30))
    output_filepath = tmp_path / "err.png"
    core_plots.plot_scatter_by_bin(
        3,
        x,
        y,
        "True values",
        "Model absolute error",
        "suptitle",
        "bin title",
        output_filepath,
    )

    # the failure message names the failing kind and the exception
    stdout = capsys.readouterr().out
    assert "resid" in stdout
    assert "simulated resid failure" in stdout

    # the scatter artifact is a fresh one-axes figure with its own labels
    figure = captured.get("fig")
    assert figure is not None
    assert captured["n_axes"] == 1
    assert captured["titles"] == ["bin title"]
    assert captured["xlabels"] == ["True values"]
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


def test_two_series_writes_non_empty_png_and_leaves_no_figure(
    monkeypatch, tmp_path: Path
) -> None:
    """The two-series helper owns its figure and keeps its current decoration."""
    captured = capture_saved_figure(monkeypatch)

    output_filepath = tmp_path / "two_series.png"
    core_plots.plot_two_series_with_differences(
        make_two_series_df(),
        "x",
        "actual",
        "pred",
        "two series title",
        output_filepath,
    )

    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []
    figure = captured["fig"]
    assert [text.get_text() for text in figure.texts] == [
        "Prediction vs. Actuals, pred vs. actual"
    ]
    assert captured["titles"] == ["", "two series title"]
    assert captured["xlabels"] == ["", "x"]


def test_scatter_by_bin_writes_non_empty_png_and_leaves_no_figure(
    tmp_path: Path,
) -> None:
    """The binned-scatter helper writes its PNG on its own figure and closes it."""
    rng = np.random.default_rng(1)
    x = pl.Series("x", rng.normal(size=30))
    y = pl.Series("y", rng.normal(size=30))
    output_filepath = tmp_path / "scatter_bin.png"
    core_plots.plot_scatter_by_bin(
        3,
        x,
        y,
        "True values",
        "Model absolute error",
        "bin suptitle",
        "bin title",
        output_filepath,
    )
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


def test_sentinel_figure_survives_and_is_not_drawn_into(
    tmp_path: Path,
) -> None:
    """Neither active helper draws into an unrelated open figure.

    A distinct caller-owned figure must stay open and untouched while each
    helper creates, draws on, saves, and closes its own figure.
    """
    sentinel = plt.figure()
    sentinel_ax = sentinel.add_subplot(111)
    sentinel_ax.set_title("sentinel title")
    sentinel_ax.set_xlabel("sentinel x")

    core_plots.plot_two_series_with_differences(
        make_two_series_df(),
        "x",
        "actual",
        "pred",
        "two series title",
        tmp_path / "two_series.png",
    )

    rng = np.random.default_rng(1)
    x = pl.Series("x", rng.normal(size=30))
    y = pl.Series("y", rng.normal(size=30))
    core_plots.plot_scatter_by_bin(
        3,
        x,
        y,
        "True values",
        "Model absolute error",
        "bin suptitle",
        "bin title",
        tmp_path / "scatter_bin.png",
    )

    # the sentinel is the only figure left open and carries no new artists
    assert plt.get_fignums() == [sentinel.number]
    assert len(sentinel.axes) == 1
    assert sentinel.axes[0].get_title() == "sentinel title"
    assert sentinel.axes[0].get_xlabel() == "sentinel x"
    assert len(sentinel.axes[0].lines) == 0
    assert len(sentinel.axes[0].collections) == 0
    assert len(sentinel.axes[0].patches) == 0


def test_feature_binary_difference_heatmap_writes_png_and_leaves_no_figure(
    tmp_path: Path,
) -> None:
    """The heatmap helper writes its PNG on its own figure and closes it."""
    plot_df = pl.DataFrame({"f1": [0.5, 0.2], "f2": [0.8, 0.1]})
    output_filepath = tmp_path / "heatmap.png"
    core_plots.plot_feature_binary_difference_heatmap(
        plot_df, "heatmap suptitle", "heatmap title", output_filepath
    )
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


def test_heatmap_sentinel_figure_survives_and_is_not_drawn_into(
    tmp_path: Path,
) -> None:
    """The heatmap helper draws on its own axes, never a caller-owned figure."""
    sentinel = plt.figure()
    sentinel_ax = sentinel.add_subplot(111)
    sentinel_ax.set_title("sentinel title")
    sentinel_ax.set_xlabel("sentinel x")

    plot_df = pl.DataFrame({"f1": [0.5, 0.2], "f2": [0.8, 0.1]})
    core_plots.plot_feature_binary_difference_heatmap(
        plot_df,
        "heatmap suptitle",
        "heatmap title",
        tmp_path / "heatmap.png",
    )

    assert plt.get_fignums() == [sentinel.number]
    assert len(sentinel.axes) == 1
    assert sentinel.axes[0].get_title() == "sentinel title"
    assert sentinel.axes[0].get_xlabel() == "sentinel x"
    assert len(sentinel.axes[0].lines) == 0
    assert len(sentinel.axes[0].collections) == 0
    assert len(sentinel.axes[0].patches) == 0


def test_scatter_with_identity_line_writes_png_and_leaves_no_figure(
    tmp_path: Path,
) -> None:
    """The identity-line scatter helper writes its PNG and closes its figure."""
    x = pl.Series("x", [1.0, 2.0, 3.0, 4.0])
    y = pl.Series("y", [1.1, 2.2, 2.9, 4.1])
    output_filepath = tmp_path / "identity_scatter.png"
    core_plots.plot_scatter_with_identity_line(
        x, y, "scatter suptitle", "scatter title", output_filepath
    )
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


def test_identity_line_scatter_sentinel_figure_survives_and_is_not_drawn_into(
    tmp_path: Path,
) -> None:
    """The identity-line helper draws on its own axes."""
    sentinel = plt.figure()
    sentinel_ax = sentinel.add_subplot(111)
    sentinel_ax.set_title("sentinel title")
    sentinel_ax.set_xlabel("sentinel x")

    x = pl.Series("x", [1.0, 2.0, 3.0, 4.0])
    y = pl.Series("y", [1.1, 2.2, 2.9, 4.1])
    core_plots.plot_scatter_with_identity_line(
        x,
        y,
        "scatter suptitle",
        "scatter title",
        tmp_path / "identity_scatter.png",
    )

    assert plt.get_fignums() == [sentinel.number]
    assert len(sentinel.axes) == 1
    assert sentinel.axes[0].get_title() == "sentinel title"
    assert sentinel.axes[0].get_xlabel() == "sentinel x"
    assert len(sentinel.axes[0].lines) == 0
    assert len(sentinel.axes[0].collections) == 0
    assert len(sentinel.axes[0].patches) == 0


def test_feature_importances_bar_writes_png_and_leaves_no_figure(
    tmp_path: Path,
) -> None:
    """The feature-importances helper writes its PNG and closes its figure."""
    df = pl.DataFrame({"feature": ["f1", "f2", "f3"], "importance": [0.6, 0.3, 0.1]})
    output_filepath = tmp_path / "importances.png"
    core_plots.plot_feature_importances_bar(
        df, "feature", "importance", output_filepath
    )
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


def test_scatter_with_central_tendency_writes_png_and_leaves_no_figure(
    tmp_path: Path,
) -> None:
    """The central-tendency helper writes its PNG and closes its figure."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.5, 2.5, 3.5]})
    mean_df = pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]})
    median_df = pl.DataFrame({"x": [1.0, 3.0], "y": [1.2, 3.4]})
    output_filepath = tmp_path / "central_tendency.png"
    core_plots.plot_scatter_with_central_tendency(
        df,
        mean_df,
        median_df,
        "x",
        "y",
        "x label",
        "y label",
        "ct suptitle",
        "ct title",
        output_filepath,
    )
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


def test_scatter_with_central_tendency_rejects_missing_column(
    tmp_path: Path,
) -> None:
    """A missing column raises a clear assertion before any figure is drawn."""
    df = pl.DataFrame({"x": [1.0], "y": [1.0]})
    with pytest.raises(AssertionError, match="not found"):
        core_plots.plot_scatter_with_central_tendency(
            df,
            df,
            df,
            "x",
            "missing",
            "x label",
            "y label",
            "suptitle",
            "title",
            tmp_path / "missing.png",
        )
    assert plt.get_fignums() == []


def test_two_scatter_series_writes_png_and_leaves_no_figure(
    tmp_path: Path,
) -> None:
    """The two-series scatter helper writes its PNG and closes its figure."""
    df_1 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.1, 2.2, 3.3]})
    df_2 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.5, 2.0, 2.8]})
    output_filepath = tmp_path / "two_scatter_series.png"
    core_plots.plot_two_scatter_series(
        df_1,
        df_2,
        "x",
        "y",
        "x label",
        "y label",
        0.5,
        "ts suptitle",
        "ts title",
        output_filepath,
    )
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


def test_dendrogram_writes_non_empty_png_and_leaves_no_figure(
    tmp_path: Path,
) -> None:
    """The dendrogram helper writes its PNG and returns the linkage matrix."""
    sample, _ = make_cluster_inputs()
    output_filepath = tmp_path / "dendrogram.png"
    linkage = core_plots.plot_hierarchical_cluster_dendrogram(sample, output_filepath)
    assert isinstance(linkage, np.ndarray)
    assert linkage.shape[0] == sample.shape[0] - 1
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


def test_attribution_clusters_by_color_writes_png_per_perplexity(
    tmp_path: Path,
) -> None:
    """The color-cluster helper writes one PNG per perplexity and closes it."""
    sample, linkage = make_cluster_inputs()
    output_filepath = tmp_path / "clusters.png"
    core_plots.plot_attribution_clusters_by_color(
        sample, linkage, 2.0, [3, 5], 0, output_filepath
    )
    for perplexity in (3, 5):
        png = tmp_path / f"clusters_{perplexity}.png"
        assert png.exists()
        assert png.stat().st_size > 0
    assert plt.get_fignums() == []


def test_attribution_clusters_by_marker_1_writes_png_per_perplexity(
    tmp_path: Path,
) -> None:
    """The marker_1 helper writes one PNG per perplexity and leaves nothing open."""
    sample, linkage = make_cluster_inputs()
    output_filepath = tmp_path / "clusters.png"
    core_plots.plot_attribution_clusters_by_marker_1(
        sample, linkage, 2.0, [3, 5], sample[:, 0], "f0", 0, output_filepath
    )
    for perplexity in (3, 5):
        png = tmp_path / f"clusters_mark1_{perplexity}_f0.png"
        assert png.exists()
        assert png.stat().st_size > 0
    assert plt.get_fignums() == []


def test_attribution_clusters_by_marker_2_writes_png_per_perplexity(
    tmp_path: Path,
) -> None:
    """The multi-panel marker_2 helper writes one PNG per perplexity."""
    sample, linkage = make_cluster_inputs()
    output_filepath = tmp_path / "clusters.png"
    core_plots.plot_attribution_clusters_by_marker_2(
        sample, linkage, 2.0, [3, 5], sample[:, 0], "f0", 0, output_filepath
    )
    for perplexity in (3, 5):
        png = tmp_path / f"clusters_mark2_{perplexity}_f0.png"
        assert png.exists()
        assert png.stat().st_size > 0
    assert plt.get_fignums() == []


def test_marker_2_sentinel_figure_survives_and_is_not_drawn_into(
    tmp_path: Path,
) -> None:
    """The multi-panel marker_2 helper draws on its own 2x2 grid."""
    sentinel = plt.figure()
    sentinel_ax = sentinel.add_subplot(111)
    sentinel_ax.set_title("sentinel title")
    sentinel_ax.set_xlabel("sentinel x")

    sample, linkage = make_cluster_inputs()
    output_filepath = tmp_path / "mark2.png"
    core_plots.plot_attribution_clusters_by_marker_2(
        sample, linkage, 2.0, [3, 5], sample[:, 0], "f0", 0, output_filepath
    )
    for perplexity in (3, 5):
        png = tmp_path / f"mark2_mark2_{perplexity}_f0.png"
        assert png.exists()
        assert png.stat().st_size > 0

    assert plt.get_fignums() == [sentinel.number]
    assert len(sentinel.axes) == 1
    assert sentinel.axes[0].get_title() == "sentinel title"
    assert sentinel.axes[0].get_xlabel() == "sentinel x"
    assert len(sentinel.axes[0].lines) == 0
    assert len(sentinel.axes[0].collections) == 0
    assert len(sentinel.axes[0].patches) == 0


NONFINITE_FRAME = pl.DataFrame(
    {
        "a": [1.0, float("nan"), float("inf"), 2.0],
        "b": [1.0, 2.0, 3.0, float("-inf")],
    }
)

UNDRAWABLE_FRAMES = [
    pl.DataFrame({"a": [], "b": []}),
    pl.DataFrame({"a": [None, None], "b": [None, None]}),
    pl.DataFrame({"a": [None, None], "b": [1.0, 2.0]}),
    pl.DataFrame({"a": [float("nan")], "b": [float("inf")]}),
]


def make_distribution_df() -> pl.DataFrame:
    """A small two-column frame with overlapping but distinct distributions."""
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, 2.5, 3.5, 4.0, 5.0],
            "b": [0.5, 1.5, 3.0, 3.5, 4.5, 6.0],
        }
    )


def test_overlaid_histograms_write_png_and_leave_no_figure(
    tmp_path: Path,
) -> None:
    """The overlay histogram helper writes its PNG and closes its figure."""
    output_filepath = tmp_path / "overlay_hist.png"
    core_plots.plot_overlaid_distribution_histograms(
        make_distribution_df(),
        "a",
        "b",
        "overlay suptitle",
        "overlay title",
        output_filepath,
    )
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


def test_overlaid_histograms_share_one_set_of_bins(monkeypatch, tmp_path: Path) -> None:
    """Both overlaid histograms are drawn with the same bin edges."""
    recorded_bins = []
    real_hist = matplotlib.axes.Axes.hist

    def spying_hist(self, x, bins=None, **kwargs):
        recorded_bins.append(np.asarray(bins))
        return real_hist(self, x, bins=bins, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "hist", spying_hist)
    core_plots.plot_overlaid_distribution_histograms(
        make_distribution_df(),
        "a",
        "b",
        "suptitle",
        "title",
        tmp_path / "shared_bins.png",
    )

    assert len(recorded_bins) == 2
    assert np.array_equal(recorded_bins[0], recorded_bins[1])


def test_overlaid_kdes_write_png_and_share_one_grid(
    monkeypatch, tmp_path: Path
) -> None:
    """Both KDE curves share one grid and the helper writes its PNG."""
    recorded_grids = []
    real_draw_kde = core_plots._draw_kde

    def spying_draw_kde(ax, values, grid, label, color):
        recorded_grids.append(np.asarray(grid))
        return real_draw_kde(ax, values, grid, label, color)

    monkeypatch.setattr(core_plots, "_draw_kde", spying_draw_kde)
    output_filepath = tmp_path / "shared_grid.png"
    core_plots.plot_overlaid_distribution_kdes(
        make_distribution_df(), "a", "b", "suptitle", "title", output_filepath
    )

    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []
    assert len(recorded_grids) == 2
    assert np.array_equal(recorded_grids[0], recorded_grids[1])


def test_overlaid_kde_constant_series_is_annotated_not_drawn(
    monkeypatch, tmp_path: Path
) -> None:
    """A constant overlay series is annotated; the other is still drawn."""
    captured = capture_saved_figure(monkeypatch)
    frame = pl.DataFrame({"a": [2.0, 2.0, 2.0], "b": [1.0, 2.0, 3.0]})
    output_filepath = tmp_path / "constant_overlay.png"
    core_plots.plot_overlaid_distribution_kdes(
        frame, "a", "b", "suptitle", "title", output_filepath
    )

    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []
    axes_texts = captured["axes_texts"][0]
    assert any("Insufficient data" in text and "a" in text for text in axes_texts)
    # only the varying series contributes a curve
    assert len(captured["lines"][0]) == 1


@pytest.mark.parametrize(
    "frame",
    UNDRAWABLE_FRAMES,
    ids=["empty", "all_null", "one_null", "non_finite"],
)
@pytest.mark.parametrize("helper", ["histograms", "kdes"])
def test_overlay_helpers_still_write_png_for_undrawable_input(
    monkeypatch, tmp_path: Path, frame: pl.DataFrame, helper: str
) -> None:
    """Empty/all-null/non-finite input still writes an annotated PNG."""
    captured = capture_saved_figure(monkeypatch)
    output_filepath = tmp_path / f"{helper}.png"
    plot = (
        core_plots.plot_overlaid_distribution_histograms
        if helper == "histograms"
        else core_plots.plot_overlaid_distribution_kdes
    )
    plot(frame, "a", "b", "suptitle", "title", output_filepath)

    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []
    assert any("Insufficient data" in text for text in captured["axes_texts"][0])


def test_distribution_histogram_writes_png_and_leave_no_figure(
    tmp_path: Path,
) -> None:
    """The single histogram helper writes its PNG and closes its figure."""
    output_filepath = tmp_path / "single_hist.png"
    core_plots.plot_distribution_histogram(
        make_distribution_df(), "a", "suptitle", "title", output_filepath
    )
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


def test_distribution_histogram_constant_series_writes_png(
    tmp_path: Path,
) -> None:
    """A constant series is still drawable as a single-bin histogram."""
    output_filepath = tmp_path / "constant_hist.png"
    core_plots.plot_distribution_histogram(
        pl.DataFrame({"a": [2.0, 2.0, 2.0]}),
        "a",
        "suptitle",
        "title",
        output_filepath,
    )
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


def test_distribution_kde_writes_png_and_leaves_no_figure(
    tmp_path: Path,
) -> None:
    """The single KDE helper writes its PNG and closes its figure."""
    output_filepath = tmp_path / "single_kde.png"
    core_plots.plot_distribution_kde(
        make_distribution_df(), "a", "suptitle", "title", output_filepath
    )
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []


def test_distribution_kde_constant_series_is_annotated(
    monkeypatch, tmp_path: Path
) -> None:
    """A constant series cannot be estimated, so the axes say so."""
    captured = capture_saved_figure(monkeypatch)
    output_filepath = tmp_path / "constant_kde.png"
    core_plots.plot_distribution_kde(
        pl.DataFrame({"a": [2.0, 2.0, 2.0]}),
        "a",
        "suptitle",
        "title",
        output_filepath,
    )

    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []
    assert captured["lines"][0] == []
    assert any("Insufficient data" in text for text in captured["axes_texts"][0])


@pytest.mark.parametrize(
    "frame",
    UNDRAWABLE_FRAMES,
    ids=["empty", "all_null", "one_null", "non_finite"],
)
@pytest.mark.parametrize("helper", ["histogram", "kde"])
def test_single_helpers_still_write_png_for_undrawable_input(
    monkeypatch, tmp_path: Path, frame: pl.DataFrame, helper: str
) -> None:
    """Empty/all-null/non-finite input still writes an annotated PNG."""
    captured = capture_saved_figure(monkeypatch)
    output_filepath = tmp_path / f"single_{helper}.png"
    if helper == "histogram":
        core_plots.plot_distribution_histogram(
            frame, "a", "suptitle", "title", output_filepath
        )
    else:
        core_plots.plot_distribution_kde(
            frame, "a", "suptitle", "title", output_filepath
        )

    assert output_filepath.exists()
    assert output_filepath.stat().st_size > 0
    assert plt.get_fignums() == []
    assert any("Insufficient data" in text for text in captured["axes_texts"][0])


def test_non_finite_values_are_filtered_before_plotting(
    monkeypatch, tmp_path: Path
) -> None:
    """Nulls and infinities never reach the KDE inputs."""
    consumed = []
    real_draw_kde = core_plots._draw_kde

    def spying_draw_kde(ax, values, grid, label, color):
        consumed.append(np.asarray(values))
        return real_draw_kde(ax, values, grid, label, color)

    monkeypatch.setattr(core_plots, "_draw_kde", spying_draw_kde)
    core_plots.plot_distribution_kde(
        NONFINITE_FRAME, "a", "suptitle", "title", tmp_path / "finite.png"
    )

    assert len(consumed) == 1
    assert np.array_equal(consumed[0], np.array([1.0, 2.0]))


def test_distribution_kde_zero_line_is_opt_in(monkeypatch, tmp_path: Path) -> None:
    """``zero_line=True`` adds the zero reference; the default omits it."""
    frame = make_distribution_df()

    captured = capture_saved_figure(monkeypatch)
    core_plots.plot_distribution_kde(
        frame,
        "a",
        "suptitle",
        "title",
        tmp_path / "with_zero.png",
        zero_line=True,
    )
    with_zero = captured["lines"][0]
    assert any(np.array_equal(xdata, np.array([0.0, 0.0])) for xdata in with_zero)
    assert plt.get_fignums() == []

    captured = capture_saved_figure(monkeypatch)
    core_plots.plot_distribution_kde(
        frame, "a", "suptitle", "title", tmp_path / "without_zero.png"
    )
    without_zero = captured["lines"][0]
    assert not any(
        np.array_equal(xdata, np.array([0.0, 0.0])) for xdata in without_zero
    )
    assert plt.get_fignums() == []


def test_distribution_helpers_leave_foreign_figure_untouched(
    tmp_path: Path,
) -> None:
    """No distribution helper draws into or closes a caller-owned figure."""
    sentinel = plt.figure()
    sentinel_ax = sentinel.add_subplot(111)
    sentinel_ax.set_title("sentinel title")
    sentinel_ax.set_xlabel("sentinel x")

    frame = make_distribution_df()
    core_plots.plot_overlaid_distribution_histograms(
        frame, "a", "b", "suptitle", "title", tmp_path / "overlay_hist.png"
    )
    core_plots.plot_overlaid_distribution_kdes(
        frame, "a", "b", "suptitle", "title", tmp_path / "overlay_kde.png"
    )
    core_plots.plot_distribution_histogram(
        frame, "a", "suptitle", "title", tmp_path / "single_hist.png"
    )
    core_plots.plot_distribution_kde(
        frame,
        "a",
        "suptitle",
        "title",
        tmp_path / "single_kde.png",
        zero_line=True,
    )

    assert plt.get_fignums() == [sentinel.number]
    assert len(sentinel.axes) == 1
    assert sentinel.axes[0].get_title() == "sentinel title"
    assert sentinel.axes[0].get_xlabel() == "sentinel x"
    assert len(sentinel.axes[0].lines) == 0
    assert len(sentinel.axes[0].collections) == 0
    assert len(sentinel.axes[0].patches) == 0
    assert len(sentinel.axes[0].texts) == 0
