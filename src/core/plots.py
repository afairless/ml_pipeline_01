"""Project-agnostic, figure-owning plotting helpers.

Every helper in this module creates its own figure, decorates it, saves it to
the caller's path, and closes it. None of them draws into or closes a
caller-owned figure, so a stage can call several helpers in a row without
managing pyplot state. Helpers that declare an artifact always write it, even
when the input has no drawable row; the axes then carry an explicit
insufficient-data annotation instead of producing a missing or empty file.

The module uses matplotlib, NumPy, polars, and SciPy only. It imports nothing
from ``src.project`` and holds no project path or column constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import complete, dendrogram, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE

from src.core.binning import create_uniform_bin_vector_of_given_length

# the joint-plot kinds the helper accepts, so a plain ``str`` cannot reach the
# kind dispatch
JointPlotKind = Literal["scatter", "kde", "hist", "hex", "reg", "resid"]

# shared histogram bin count and KDE grid size for the distribution helpers
_DISTRIBUTION_BIN_N = 30
_KDE_GRID_N = 200

_INSUFFICIENT_DATA_MESSAGE = "Insufficient data to plot"


def plot_feature_binary_difference_heatmap(
    plot_df: pl.DataFrame,
    suptitle: str,
    title: str,
    output_filepath: Path = Path("plot.png"),
) -> None:
    """Plot the frame's values as a heatmap with a color scale.

    The helper owns the figure it creates: it draws on its own ``ax``, adds a
    colorbar bound to that axes, puts the suptitle on the figure and the title
    on the axes, then saves and closes the figure without touching pyplot's
    current-figure state.

    Args:
        plot_df: Numeric frame whose rows and columns form the heatmap grid.
        suptitle: Figure-level title.
        title: Axes-level title.
        output_filepath: PNG path written by the owned figure.
    """

    fig, ax = plt.subplots()
    try:
        image = ax.imshow(
            plot_df.to_numpy(), cmap="Blues_r", aspect="auto", origin="upper"
        )
        ax.set_xticks(range(plot_df.width), plot_df.columns)
        ax.set_yticks(range(plot_df.height))
        fig.colorbar(image, ax=ax)

        fig.suptitle(suptitle)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_filepath)
    finally:
        plt.close(fig)


def plot_scatter_with_identity_line(
    x: pl.Series,
    y: pl.Series,
    suptitle: str,
    title: str,
    output_filepath: Path,
) -> None:
    """Plot ``x`` against ``y`` as a scatter with a gray identity line.

    The helper owns the figure it creates: it draws the scatter and the
    identity line on its own ``ax``, labels and titles it, then saves and
    closes the figure without touching pyplot's current-figure state.

    Args:
        x: Horizontal series.
        y: Vertical series.
        suptitle: Figure-level title.
        title: Axes-level title.
        output_filepath: PNG path written by the owned figure.
    """

    x_values = _finite_values(x)
    y_values = _finite_values(y)
    x_min = float(x_values.min())
    y_min = float(y_values.min())
    x_max = float(x_values.max())
    y_max = float(y_values.max())

    fig, ax = plt.subplots()
    try:
        ax.scatter(x.to_numpy(), y.to_numpy(), alpha=0.2)
        ax.plot([x_min, x_max], [y_min, y_max], color="gray")

        ax.set_xlabel(x.name + " version 1")
        ax.set_ylabel(y.name + " version 2")

        fig.suptitle(suptitle)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_filepath)
    finally:
        plt.close(fig)


def plot_joint_with_rug_and_margin(
    plot_df: pl.DataFrame,
    x_colname: str,
    y_colname: str,
    joint_plot_kind: JointPlotKind,
    suptitle: str,
    title: str,
    output_filepath: Path,
) -> None:
    """Plot a joint panel with marginal distributions, a rug, and reference lines.

    The joint panel shows ``y_colname`` against ``x_colname`` in the requested
    style; the top and right marginals summarise the two series. The identity
    line and the zero lines are drawn on the joint panel only, the limits are
    padded by two percent, and the labels and titles are attached to the axes
    and figure that this helper created.

    The helper owns its figure: it creates, decorates, saves, and closes that
    figure in ``finally``, so a failure while drawing cannot leak a partially
    built figure into pyplot's current state. Empty input, or input with no
    finite row pair, prints a message and returns without writing an artifact.

    Args:
        plot_df: Frame carrying both columns.
        x_colname: Column used for the horizontal axis.
        y_colname: Column used for the vertical axis.
        joint_plot_kind: Rendering style for the joint panel.
        suptitle: Figure-level title.
        title: Joint-axes title.
        output_filepath: PNG path written by the owned figure.
    """

    x_values, y_values = _finite_pair(plot_df, x_colname, y_colname)
    x = plot_df[x_colname]
    y = plot_df[y_colname]

    if x_values.size == 0 or y_values.size == 0:
        print(
            f"Skipping {joint_plot_kind} joint plot for {output_filepath}: "
            "input is empty or has no finite row pair"
        )
        return

    x_min = float(x_values.min())
    y_min = float(y_values.min())
    x_max = float(x_values.max())
    y_max = float(y_values.max())

    x_min_pad = x_min - 0.02 * (x_max - x_min)
    x_max_pad = x_max + 0.02 * (x_max - x_min)
    y_min_pad = y_min - 0.02 * (y_max - y_min)
    y_max_pad = y_max + 0.02 * (y_max - y_min)

    before = set(plt.get_fignums())
    figure: Figure | None = None
    try:
        figure = plt.figure()
        grid = figure.add_gridspec(
            2,
            2,
            width_ratios=(4, 1),
            height_ratios=(1, 4),
            wspace=0.05,
            hspace=0.05,
        )
        ax_joint = figure.add_subplot(grid[1, 0])
        ax_marg_x = figure.add_subplot(grid[0, 0], sharex=ax_joint)
        ax_marg_y = figure.add_subplot(grid[1, 1], sharey=ax_joint)

        _draw_joint_panel(ax_joint, x_values, y_values, joint_plot_kind)
        _draw_marginals(ax_marg_x, ax_marg_y, x_values, y_values, joint_plot_kind)

        _draw_rug(
            ax_joint, x_values, y_values, x_min_pad, x_max_pad, y_min_pad, y_max_pad
        )

        shared_min = max(x_min, y_min)
        shared_max = min(x_max, y_max)
        ax_joint.plot(
            [shared_min, shared_max],
            [shared_min, shared_max],
            color="gray",
            alpha=0.3,
        )
        ax_joint.plot([0, 0], [y_min, y_max], color="gray")
        ax_joint.plot([x_min, x_max], [0, 0], color="gray")

        ax_joint.set_xlim(x_min_pad, x_max_pad)
        ax_joint.set_ylim(y_min_pad, y_max_pad)

        ax_joint.set_xlabel(x.name)
        ax_joint.set_ylabel(y.name)
        ax_joint.set_title(title, fontsize=10)

        ax_marg_x.tick_params(labelbottom=False)
        ax_marg_y.tick_params(labelleft=False)

        figure.suptitle(suptitle)
        figure.subplots_adjust(top=0.9, right=0.9, wspace=0.05, hspace=0.05)
        figure.savefig(output_filepath)
    finally:
        if figure is not None:
            plt.close(figure)
        else:
            for fignum in set(plt.get_fignums()) - before:
                plt.close(fignum)


def plot_prediction_joint_plots(
    plot_df: pl.DataFrame,
    y_true_colname: str,
    y_pred_colname: str,
    param_str: str,
    title: str,
    output_stem: str,
    output_path: Path,
) -> None:
    """Write scatter, KDE, and residual prediction joint plots.

    One PNG is written per joint style, named
    ``<output_stem>_pred_<param_str>_<kind>.png``. A style that matplotlib
    cannot draw for this data (for example constant input) prints the failing
    style and exception and leaves the other artifacts in place.

    Args:
        plot_df: Frame carrying the truth and prediction columns.
        y_true_colname: Column holding the observed values.
        y_pred_colname: Column holding the predicted values.
        param_str: Parameter label embedded in each filename.
        title: Joint-axes title passed to every plot.
        output_stem: Filename stem shared by the three artifacts.
        output_path: Existing directory that receives the PNGs.
    """

    suptitle = f"Prediction vs. Actuals, {y_true_colname}"

    for joint_kind in cast(tuple[JointPlotKind, ...], ("scatter", "kde", "resid")):
        output_filename = output_stem + f"_pred_{param_str}_{joint_kind}.png"
        output_filepath = output_path / output_filename
        try:
            plot_joint_with_rug_and_margin(
                plot_df,
                y_true_colname,
                y_pred_colname,
                joint_kind,
                suptitle,
                title,
                output_filepath,
            )
        except Exception as exc:
            print(f"Error plotting {joint_kind} for {output_filename}: {exc}")


def plot_two_series_with_differences(
    plot_df: pl.DataFrame,
    x_colname: str,
    y1_colname: str,
    y2_colname: str,
    title: str,
    output_filepath: Path,
) -> None:
    """Plot two series on the upper panel and their difference on the lower.

    Null-valued rows are dropped first: a sparsely populated bin carries no
    drawable point. The helper owns the figure it creates: it draws on its own
    ``axs[0]``/``axs[1]``, puts the suptitle on the figure and the title and
    x label on the lower axes, then saves and closes the figure without
    touching pyplot state.

    Args:
        plot_df: Frame carrying the x column and the two series.
        x_colname: Column used for the shared horizontal axis.
        y1_colname: First series, drawn in blue.
        y2_colname: Second series, drawn in orange; it is the minuend of the
            difference panel.
        title: Lower-axes title.
        output_filepath: PNG path written by the owned figure.
    """

    plot_df = plot_df.filter(
        pl.col(x_colname).is_not_null()
        & pl.col(y1_colname).is_not_null()
        & pl.col(y2_colname).is_not_null()
    )

    y_diff = plot_df[y2_colname] - plot_df[y1_colname]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    try:
        x = plot_df[x_colname]
        y1 = plot_df[y1_colname]
        y2 = plot_df[y2_colname]

        axs[0].scatter(
            x.to_numpy(), y1.to_numpy(), alpha=0.7, color="blue", label=y1_colname
        )
        axs[0].scatter(
            x.to_numpy(), y2.to_numpy(), alpha=0.7, color="orange", label=y2_colname
        )
        axs[0].legend()

        axs[1].bar(x.to_numpy(), y_diff.to_numpy(), alpha=1.0, color="orange")
        axs[1].set_xlabel(x.name)

        suptitle = f"Prediction vs. Actuals, {y2_colname} vs. {y1_colname}"
        fig.suptitle(suptitle)
        axs[1].set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_filepath)
    finally:
        plt.close(fig)


def plot_scatter_by_bin(
    bin_n: int,
    x: pl.Series,
    y: pl.Series,
    x_label: str,
    y_label: str,
    suptitle: str,
    title: str,
    output_filepath: Path,
) -> None:
    """Scatter ``x`` and ``y`` after averaging them into ``bin_n`` equal bins.

    The rows are sorted by ``x`` and split into near-equal-count bins, then
    each bin's mean is plotted, which keeps a dense scatter readable. The
    helper owns the figure it creates, saves it, and closes it.

    Args:
        bin_n: Number of equal-count bins.
        x: Horizontal series.
        y: Vertical series; must be the same length as ``x``.
        x_label: Horizontal-axis label.
        y_label: Vertical-axis label.
        suptitle: Figure-level title.
        title: Axes-level title.
        output_filepath: PNG path written by the owned figure.
    """

    assert x.shape[0] == y.shape[0], "x and y must be the same length"

    frame = pl.DataFrame({"x": x, "y": y}).sort("x")

    bins = create_uniform_bin_vector_of_given_length(bin_n, frame.shape[0])
    binned = frame.with_columns(pl.Series("bin", bins))
    plot_df = binned.group_by("bin").mean().sort("x").drop("bin")

    fig, ax = plt.subplots()
    try:
        ax.scatter(plot_df["x"].to_numpy(), plot_df["y"].to_numpy(), alpha=0.7)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        fig.suptitle(suptitle)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_filepath)
    finally:
        plt.close(fig)


def plot_feature_importances_bar(
    df: pl.DataFrame,
    feature_name_colname: str,
    feature_importance_colname: str,
    output_filepath: Path,
) -> None:
    """Plot feature importances on a horizontal bar chart.

    The helper owns the figure it creates: it draws the grid and bars on its
    own ``ax``, labels and titles it, then saves and closes the figure without
    touching pyplot's current-figure state.

    Args:
        df: Frame carrying one row per feature.
        feature_name_colname: Column with the bar labels.
        feature_importance_colname: Column with the bar lengths.
        output_filepath: PNG path written by the owned figure.
    """

    fig, ax = plt.subplots()
    try:
        ax.grid(axis="x", zorder=0)
        ax.barh(
            df[feature_name_colname].to_numpy(),
            df[feature_importance_colname].to_numpy(),
            zorder=2,
        )
        ax.set_xlabel("Importances")
        ax.set_title(f"Top {len(df)} Features")

        fig.tight_layout(pad=3.0)
        fig.savefig(output_filepath)
    finally:
        plt.close(fig)


def plot_hierarchical_cluster_dendrogram(
    attribution_sample: np.ndarray, output_filepath: Path
) -> np.ndarray:
    """Plot a dendrogram of complete-linkage clusters and return the linkage.

    The helper owns the figure it creates: it builds its own ``fig``/``ax``,
    keeps the tight layout before the dendrogram, draws the dendrogram on
    ``ax``, then saves and closes the figure without touching pyplot's
    current-figure state. The returned complete-linkage matrix is the input
    the cluster-coloring helpers expect.

    Args:
        attribution_sample: Two-dimensional sample, one row per observation.
        output_filepath: PNG path written by the owned figure.

    Returns:
        The complete-linkage matrix built from the sample.
    """

    condensed_distance_matrix = pdist(attribution_sample)
    cluster_linkage_matrix = complete(condensed_distance_matrix)

    fig, ax = plt.subplots(figsize=(16, 8))
    try:
        ax.set_ylabel("Distance")
        ax.set_xlabel("Index")
        fig.tight_layout()
        dendrogram(cluster_linkage_matrix, leaf_rotation=90, ax=ax)
        fig.savefig(output_filepath)
    finally:
        plt.close(fig)

    return cluster_linkage_matrix


def plot_attribution_clusters_by_color(
    attribution_sample: np.ndarray,
    cluster_linkage_matrix: np.ndarray,
    max_d: float,
    perplexities: list[int],
    random_state: int,
    output_filepath: Path,
) -> None:
    """Plot two-dimensional embeddings at each perplexity, colored by cluster.

    One PNG is written per perplexity, with the perplexity appended to the
    stem. The helper owns one figure per perplexity: it creates, decorates,
    saves, and closes each figure within the loop, without touching pyplot's
    current-figure state.

    Args:
        attribution_sample: Two-dimensional sample fed to the embedding.
        cluster_linkage_matrix: Linkage matrix from the dendrogram helper.
        max_d: Distance cut used to assign cluster labels.
        perplexities: Perplexity values, one figure each.
        random_state: Seed for the embedding.
        output_filepath: Base PNG path; the perplexity is inserted into it.
    """

    color_choices = [
        "viridis",
        "brg",
        "rainbow_r",
        "tab10_r",
        "tab20_r",
        "turbo",
        "turbo_r",
        "Dark2_r",
    ]
    cmap = color_choices[1]

    component_colnames = ["Component 1", "Component 2"]

    for perplexity in perplexities:
        embedding = TSNE(
            n_components=2, perplexity=perplexity, random_state=random_state
        ).fit_transform(attribution_sample)
        cluster_labels = fcluster(cluster_linkage_matrix, max_d, criterion="distance")

        fig, ax = plt.subplots()
        try:
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=cluster_labels,
                cmap=cmap,
                alpha=0.3,
            )

            ax.set_ylabel(component_colnames[0])
            ax.set_xlabel(component_colnames[1])
            ax.set_title(f"{len(np.unique(cluster_labels))} clusters by color")
            fig.tight_layout()

            output_filename = (
                output_filepath.stem + f"_{perplexity}" + output_filepath.suffix
            )
            fig.savefig(output_filepath.parent / output_filename)
        finally:
            plt.close(fig)


def plot_attribution_clusters_by_marker_1(
    attribution_sample: np.ndarray,
    cluster_linkage_matrix: np.ndarray,
    max_d: float,
    perplexities: list[int],
    attribution_values: np.ndarray,
    feature_name: str,
    random_state: int,
    output_filepath: Path,
) -> None:
    """Plot one marker per cluster, colored by an attribution value.

    One PNG is written per perplexity, with the perplexity and feature name
    appended to the stem. The helper owns one figure per perplexity and closes
    it within the loop.

    Args:
        attribution_sample: Two-dimensional sample fed to the embedding.
        cluster_linkage_matrix: Linkage matrix from the dendrogram helper.
        max_d: Distance cut used to assign cluster labels.
        perplexities: Perplexity values, one figure each.
        attribution_values: One value per sample row, used for the color scale.
        feature_name: Feature label embedded in the filename and title.
        random_state: Seed for the embedding.
        output_filepath: Base PNG path; the suffix is inserted into it.
    """

    cmap = "CMRmap_r"
    markers = ["o", "^", "X", "+", "s", "v", "p", "P", "*", "D", "d", "H", "h"]
    component_colnames = ["Component 1", "Component 2"]

    for perplexity in perplexities:
        embedding = TSNE(
            n_components=2, perplexity=perplexity, random_state=random_state
        ).fit_transform(attribution_sample)
        cluster_labels = fcluster(cluster_linkage_matrix, max_d, criterion="distance")
        values = np.unique(cluster_labels)

        fig, ax = plt.subplots()
        try:
            for i, value in enumerate(values):
                value_idxs = np.where(cluster_labels == value)[0]
                ax.scatter(
                    embedding[value_idxs, 0],
                    embedding[value_idxs, 1],
                    marker=markers[i],
                    s=96,
                    c=attribution_values[value_idxs],
                    cmap=cmap,
                    alpha=0.5,
                )

            ax.set_ylabel(component_colnames[0])
            ax.set_xlabel(component_colnames[1])
            ax.set_title(f"{len(values)} clusters by marker, colored by {feature_name}")
            fig.tight_layout()

            output_filename = (
                output_filepath.stem
                + f"_mark1_{perplexity}_{feature_name}"
                + output_filepath.suffix
            )
            fig.savefig(output_filepath.parent / output_filename)
        finally:
            plt.close(fig)


def plot_attribution_clusters_by_marker_2(
    attribution_sample: np.ndarray,
    cluster_linkage_matrix: np.ndarray,
    max_d: float,
    perplexities: list[int],
    attribution_values: np.ndarray,
    feature_name: str,
    random_state: int,
    output_filepath: Path,
) -> None:
    """Plot up to four clusters in a shared 2x2 marker panel per perplexity.

    One PNG is written per perplexity, with the perplexity and feature name
    appended to the stem. The multi-panel helper owns one figure per
    perplexity: it builds and decorates its own 2x2 grid, puts the title on
    the figure, then saves and closes that figure within the loop.

    Args:
        attribution_sample: Two-dimensional sample fed to the embedding.
        cluster_linkage_matrix: Linkage matrix from the dendrogram helper.
        max_d: Distance cut used to assign cluster labels.
        perplexities: Perplexity values, one figure each.
        attribution_values: One value per sample row, used for the color scale.
        feature_name: Feature label embedded in the filename and title.
        random_state: Seed for the embedding.
        output_filepath: Base PNG path; the suffix is inserted into it.
    """

    cmap = "CMRmap_r"
    markers = ["o", "^", "X", "+", "s", "v", "p", "P", "*", "D", "d", "H", "h"]

    for perplexity in perplexities:
        embedding = TSNE(
            n_components=2, perplexity=perplexity, random_state=random_state
        ).fit_transform(attribution_sample)
        cluster_labels = fcluster(cluster_linkage_matrix, max_d, criterion="distance")
        values = np.unique(cluster_labels)

        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        try:
            grid = [[0, 0], [0, 1], [1, 0], [1, 1]]
            for i in range(4):
                if i >= values.size:
                    break
                value = values[i]
                value_idxs = np.where(cluster_labels == value)[0]
                ax[grid[i][0], grid[i][1]].scatter(
                    embedding[value_idxs, 0],
                    embedding[value_idxs, 1],
                    marker=markers[i],
                    s=96,
                    c=attribution_values[value_idxs],
                    cmap=cmap,
                    alpha=0.5,
                )

            fig.suptitle(f"{len(values)} clusters by marker, colored by {feature_name}")
            fig.tight_layout()

            output_filename = (
                output_filepath.stem
                + f"_mark2_{perplexity}_{feature_name}"
                + output_filepath.suffix
            )
            fig.savefig(output_filepath.parent / output_filename)
        finally:
            plt.close(fig)


def plot_scatter_with_central_tendency(
    df: pl.DataFrame,
    mean_df: pl.DataFrame,
    median_df: pl.DataFrame,
    x_colname: str,
    y_colname: str,
    x_label: str,
    y_label: str,
    suptitle: str,
    title: str,
    output_filepath: Path,
) -> None:
    """Scatter ``df`` and overlay the mean and median paths.

    The helper owns the figure it creates: it draws the scatter and the
    mean/median lines on its own ``ax``, labels, legends, and titles it, then
    saves and closes the figure without touching pyplot's current-figure
    state.

    Args:
        df: Frame carrying the full scatter.
        mean_df: Frame with the mean path, in the same two columns.
        median_df: Frame with the median path, in the same two columns.
        x_colname: Column used for the horizontal axis.
        y_colname: Column used for the vertical axis.
        x_label: Horizontal-axis label.
        y_label: Vertical-axis label.
        suptitle: Figure-level title.
        title: Axes-level title.
        output_filepath: PNG path written by the owned figure.
    """

    _require_columns(df, x_colname, y_colname, "df")
    _require_columns(mean_df, x_colname, y_colname, "mean_df")
    _require_columns(median_df, x_colname, y_colname, "median_df")

    fig, ax = plt.subplots()
    try:
        ax.scatter(df[x_colname].to_numpy(), df[y_colname].to_numpy(), alpha=0.05)
        ax.plot(
            mean_df[x_colname].to_numpy(),
            mean_df[y_colname].to_numpy(),
            alpha=1,
            color="orange",
            label="mean",
        )
        ax.plot(
            median_df[x_colname].to_numpy(),
            median_df[y_colname].to_numpy(),
            alpha=1,
            color="blue",
            label="median",
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()

        fig.suptitle(suptitle)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_filepath)
    finally:
        plt.close(fig)


def plot_two_scatter_series(
    df_1: pl.DataFrame,
    df_2: pl.DataFrame,
    x_colname: str,
    y_colname: str,
    x_label: str,
    y_label: str,
    alpha: float,
    suptitle: str,
    title: str,
    output_filepath: Path,
) -> None:
    """Plot two scatter series on one axes, one from each frame.

    The helper owns the figure it creates: it draws both series on its own
    ``ax``, labels, legends, and titles it, then saves and closes the figure
    without touching pyplot's current-figure state.

    Args:
        df_1: First frame, drawn in blue.
        df_2: Second frame, drawn in orange.
        x_colname: Column used for the horizontal axis.
        y_colname: Column used for the vertical axis.
        x_label: Horizontal-axis label.
        y_label: Vertical-axis label.
        alpha: Marker transparency for both series.
        suptitle: Figure-level title.
        title: Axes-level title.
        output_filepath: PNG path written by the owned figure.
    """

    _require_columns(df_1, x_colname, y_colname, "df_1")
    _require_columns(df_2, x_colname, y_colname, "df_2")

    fig, ax = plt.subplots()
    try:
        ax.scatter(
            df_1[x_colname].to_numpy(),
            df_1[y_colname].to_numpy(),
            alpha=alpha,
            color="blue",
            label="series 1",
        )
        ax.scatter(
            df_2[x_colname].to_numpy(),
            df_2[y_colname].to_numpy(),
            alpha=alpha,
            color="orange",
            label="series 2",
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()

        fig.suptitle(suptitle)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_filepath)
    finally:
        plt.close(fig)


def _require_columns(
    df: pl.DataFrame, x_colname: str, y_colname: str, frame_name: str
) -> None:
    """Assert that both named columns exist in ``df`` with a clear message."""
    assert x_colname in df.columns, (
        f'Column "{x_colname}" not found in dataframe "{frame_name}"'
    )
    assert y_colname in df.columns, (
        f'Column "{y_colname}" not found in dataframe "{frame_name}"'
    )


def _as_float_series(plot_df: pl.DataFrame, colname: str) -> pl.Series:
    """Return ``colname`` as a float series, keeping nulls in place.

    Non-numeric dtypes have no finite-value predicate, so cast them first: a
    boolean column becomes 0/1 and any unparseable value becomes a null that
    the distribution helpers drop. Numeric columns are used as-is.
    """

    series = plot_df[colname]
    if not series.dtype.is_numeric():
        series = series.cast(pl.Float64, strict=False)
    return series


def _finite_values(series: pl.Series) -> np.ndarray:
    """Return the finite, non-null values of ``series`` as a float array."""
    numeric = series
    if not numeric.dtype.is_numeric():
        numeric = numeric.cast(pl.Float64, strict=False)
    mask = numeric.is_not_null() & numeric.is_finite()
    return cast(np.ndarray, numeric.filter(mask).to_numpy())


def _finite_distribution_values(plot_df: pl.DataFrame, colname: str) -> np.ndarray:
    """Return the finite, non-null values of ``colname``.

    Neither nulls nor infinities can be drawn on a finite histogram or KDE
    grid, so both are dropped before any plotting.
    """

    return _finite_values(_as_float_series(plot_df, colname))


def _finite_distribution_pair(
    plot_df: pl.DataFrame, x1_colname: str, x2_colname: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return both columns' finite values on the rows where both are usable.

    An overlay compares the two distributions case by case, so a row with a
    null or non-finite value in either column is dropped from both series.
    """

    x1_series = _as_float_series(plot_df, x1_colname)
    x2_series = _as_float_series(plot_df, x2_colname)
    mask = (
        x1_series.is_not_null()
        & x1_series.is_finite()
        & x2_series.is_not_null()
        & x2_series.is_finite()
    )
    return (
        cast(np.ndarray, x1_series.filter(mask).to_numpy()),
        cast(np.ndarray, x2_series.filter(mask).to_numpy()),
    )


def _finite_pair(
    plot_df: pl.DataFrame, x_colname: str, y_colname: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return two columns' finite values on the rows where both are usable."""
    return _finite_distribution_pair(plot_df, x_colname, y_colname)


def _common_histogram_edges(arrays: list[np.ndarray]) -> list[float]:
    """Common bin edges spanning the combined finite range of ``arrays``.

    Sharing the edges makes the overlaid histograms directly comparable
    instead of giving each series its own binning. NumPy expands a degenerate
    (constant) range by half a unit on either side.
    """

    edges = np.histogram_bin_edges(np.concatenate(arrays), bins=_DISTRIBUTION_BIN_N)
    return edges.tolist()


def _shared_kde_grid(arrays: list[np.ndarray]) -> np.ndarray:
    """A shared evaluation grid spanning the combined finite range."""

    combined = np.concatenate(arrays)
    x_min = float(combined.min())
    x_max = float(combined.max())
    if x_min == x_max:
        x_min = x_min - 0.5
        x_max = x_max + 0.5
    return np.linspace(x_min, x_max, _KDE_GRID_N)


def _draw_kde(
    ax: Axes, values: np.ndarray, grid: np.ndarray, label: str, color: str
) -> bool:
    """Draw one KDE curve; return False when it cannot be estimated.

    A constant series has zero variance, so its density estimate is singular
    rather than merely flat; reporting that instead of drawing a spike keeps
    the artifact honest.
    """

    if np.unique(values).size < 2:
        return False
    ax.plot(grid, gaussian_kde(values)(grid), label=label, color=color)
    return True


def _draw_joint_panel(
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    joint_plot_kind: JointPlotKind,
) -> None:
    """Draw the joint panel in the requested style.

    ``kde`` colors the points by their joint density, ``hist`` and ``hex``
    aggregate the points into cells, ``reg`` adds a least-squares line, and
    ``resid`` plots the fitted value against the residual. ``scatter`` is a
    plain point cloud.
    """

    if joint_plot_kind == "scatter":
        ax.scatter(x_values, y_values, alpha=0.2)
    elif joint_plot_kind == "kde":
        density = gaussian_kde(np.vstack([x_values, y_values]))(
            np.vstack([x_values, y_values])
        )
        ax.scatter(x_values, y_values, c=density, s=8, alpha=0.6, cmap="viridis")
    elif joint_plot_kind == "hist":
        ax.hist2d(x_values, y_values, bins=30, cmap="Blues")
    elif joint_plot_kind == "hex":
        ax.hexbin(x_values, y_values, gridsize=30, cmap="Blues")
    elif joint_plot_kind == "reg":
        coefficients = np.polyfit(x_values, y_values, 1)
        line_x = np.linspace(float(x_values.min()), float(x_values.max()), 100)
        ax.scatter(x_values, y_values, alpha=0.2)
        ax.plot(line_x, np.polyval(coefficients, line_x), color="red")
    elif joint_plot_kind == "resid":
        coefficients = np.polyfit(x_values, y_values, 1)
        fitted = np.polyval(coefficients, x_values)
        ax.scatter(fitted, y_values - fitted, alpha=0.2)
    else:
        raise ValueError(f"Unsupported joint plot kind: {joint_plot_kind}")


def _draw_marginals(
    ax_marg_x: Axes,
    ax_marg_y: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    joint_plot_kind: JointPlotKind,
) -> None:
    """Draw the top and right marginal summaries for a joint panel.

    The ``kde`` style uses density curves when a series has variance and
    falls back to a histogram for a constant series, which has no estimable
    density. Every other style uses histograms.
    """

    for ax, values in ((ax_marg_x, x_values), (ax_marg_y, y_values)):
        drew_kde = False
        if joint_plot_kind == "kde":
            grid = _shared_kde_grid([values])
            drew_kde = _draw_kde(ax, values, grid, "", "blue")
        if not drew_kde:
            ax.hist(values, bins=_DISTRIBUTION_BIN_N, color="blue", alpha=0.5)


def _draw_rug(
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    """Draw short tick marks for each observation along both joint edges."""

    x_span = x_max - x_min
    y_span = y_max - y_min
    ax.vlines(
        x_values,
        y_min,
        y_min + 0.02 * y_span,
        color="gray",
        alpha=0.3,
        linewidth=0.5,
    )
    ax.hlines(
        y_values,
        x_min,
        x_min + 0.02 * x_span,
        color="gray",
        alpha=0.3,
        linewidth=0.5,
    )


def _save_insufficient_data_figure(
    fig: Figure,
    ax: Axes,
    suptitle: str,
    title: str,
    output_filepath: Path,
    message: str = _INSUFFICIENT_DATA_MESSAGE,
) -> None:
    """Annotate an undrawable figure's axes and save it.

    The distribution helpers always write their declared PNG, so a caller can
    rely on the artifact existing even when there is nothing to draw.
    """

    fig.suptitle(suptitle)
    ax.set_title(title, fontsize=10)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_filepath)


def plot_overlaid_distribution_histograms(
    plot_df: pl.DataFrame,
    x1_colname: str,
    x2_colname: str,
    suptitle: str,
    title: str,
    output_filepath: Path,
) -> None:
    """Plot two columns as overlaid histograms sharing one set of bins.

    Rows with a null or non-finite value in either column are dropped, the bin
    edges are computed once from the combined range, and both series are
    drawn with overlaid density-normalised bars and a legend. The helper owns
    the figure it creates: it saves and closes that figure without touching
    pyplot's current-figure state, and it always writes ``output_filepath``,
    annotating the axes when no row is left to draw.

    Args:
        plot_df: Frame carrying both columns.
        x1_colname: First series, drawn in blue.
        x2_colname: Second series, drawn in orange.
        suptitle: Figure-level title.
        title: Axes-level title.
        output_filepath: PNG path written by the owned figure.
    """

    x1, x2 = _finite_distribution_pair(plot_df, x1_colname, x2_colname)

    fig, ax = plt.subplots()
    try:
        if x1.size == 0 and x2.size == 0:
            _save_insufficient_data_figure(fig, ax, suptitle, title, output_filepath)
            return

        bins = _common_histogram_edges([x1, x2])
        if x1.size > 0:
            ax.hist(
                x1,
                bins=bins,
                density=True,
                alpha=0.5,
                color="blue",
                label=x1_colname,
            )
        if x2.size > 0:
            ax.hist(
                x2,
                bins=bins,
                density=True,
                alpha=0.5,
                color="orange",
                label=x2_colname,
            )

        ax.set_xlabel(f"{x1_colname} / {x2_colname}")
        ax.set_ylabel("Density")
        ax.legend()

        fig.suptitle(suptitle)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_filepath)
    finally:
        plt.close(fig)


def plot_overlaid_distribution_kdes(
    plot_df: pl.DataFrame,
    x1_colname: str,
    x2_colname: str,
    suptitle: str,
    title: str,
    output_filepath: Path,
) -> None:
    """Plot two columns as overlaid KDE curves on one shared grid.

    Rows with a null or non-finite value in either column are dropped, both
    densities are evaluated on the same grid so the curves are comparable,
    and a series that is constant (and therefore has no estimable density) is
    reported on the axes instead of being drawn. The helper owns the figure it
    creates, saves and closes it, and always writes ``output_filepath``.

    Args:
        plot_df: Frame carrying both columns.
        x1_colname: First series, drawn in blue.
        x2_colname: Second series, drawn in orange.
        suptitle: Figure-level title.
        title: Axes-level title.
        output_filepath: PNG path written by the owned figure.
    """

    x1, x2 = _finite_distribution_pair(plot_df, x1_colname, x2_colname)

    fig, ax = plt.subplots()
    try:
        if x1.size == 0 and x2.size == 0:
            _save_insufficient_data_figure(fig, ax, suptitle, title, output_filepath)
            return

        grid = _shared_kde_grid([x1, x2])
        skipped: list[str] = []
        drawn = 0
        if x1.size > 0:
            if _draw_kde(ax, x1, grid, x1_colname, "blue"):
                drawn += 1
            else:
                skipped.append(f"{x1_colname}: {_INSUFFICIENT_DATA_MESSAGE}")
        if x2.size > 0:
            if _draw_kde(ax, x2, grid, x2_colname, "orange"):
                drawn += 1
            else:
                skipped.append(f"{x2_colname}: {_INSUFFICIENT_DATA_MESSAGE}")

        if drawn == 0:
            _save_insufficient_data_figure(
                fig,
                ax,
                suptitle,
                title,
                output_filepath,
                "\n".join(skipped) or _INSUFFICIENT_DATA_MESSAGE,
            )
            return

        if skipped:
            ax.text(
                0.02,
                0.95,
                "\n".join(skipped),
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=8,
            )

        ax.set_xlabel(f"{x1_colname} / {x2_colname}")
        ax.set_ylabel("Density")
        ax.legend()

        fig.suptitle(suptitle)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_filepath)
    finally:
        plt.close(fig)


def plot_distribution_histogram(
    plot_df: pl.DataFrame,
    x_colname: str,
    suptitle: str,
    title: str,
    output_filepath: Path,
) -> None:
    """Plot one column as a density-normalised histogram.

    Null and non-finite rows are dropped. The helper owns the figure it
    creates, saves and closes it without touching pyplot's current-figure
    state, and always writes ``output_filepath``, annotating the axes when no
    row is left to draw.

    Args:
        plot_df: Frame carrying the column.
        x_colname: Column to summarise.
        suptitle: Figure-level title.
        title: Axes-level title.
        output_filepath: PNG path written by the owned figure.
    """

    values = _finite_distribution_values(plot_df, x_colname)

    fig, ax = plt.subplots()
    try:
        if values.size == 0:
            _save_insufficient_data_figure(fig, ax, suptitle, title, output_filepath)
            return

        ax.hist(values, bins=_DISTRIBUTION_BIN_N, density=True, color="blue")
        ax.set_xlabel(x_colname)
        ax.set_ylabel("Density")

        fig.suptitle(suptitle)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_filepath)
    finally:
        plt.close(fig)


def plot_distribution_kde(
    plot_df: pl.DataFrame,
    x_colname: str,
    suptitle: str,
    title: str,
    output_filepath: Path,
    zero_line: bool = False,
) -> None:
    """Plot one column as a KDE curve.

    Null and non-finite rows are dropped. ``zero_line`` adds a vertical
    reference at zero for difference series; it defaults to ``False`` so the
    helper stays generic. A constant series has no estimable density, so the
    axes report insufficient data instead of drawing a spike, and the helper
    always writes ``output_filepath``. The helper owns the figure it creates
    and never draws into or closes a caller-owned one.

    Args:
        plot_df: Frame carrying the column.
        x_colname: Column to summarise.
        suptitle: Figure-level title.
        title: Axes-level title.
        output_filepath: PNG path written by the owned figure.
        zero_line: Draw a dashed vertical reference at zero when true.
    """

    values = _finite_distribution_values(plot_df, x_colname)

    fig, ax = plt.subplots()
    try:
        if values.size == 0:
            _save_insufficient_data_figure(fig, ax, suptitle, title, output_filepath)
            return

        grid = _shared_kde_grid([values])
        if not _draw_kde(ax, values, grid, x_colname, "blue"):
            _save_insufficient_data_figure(fig, ax, suptitle, title, output_filepath)
            return

        if zero_line:
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)

        ax.set_xlabel(x_colname)
        ax.set_ylabel("Density")
        ax.legend()

        fig.suptitle(suptitle)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_filepath)
    finally:
        plt.close(fig)
