"""Frozen project configuration and generic reporting policy.

``ProjectConfig`` is the single value every stage reads its settings from: the
demo data location, the date boundaries that define the partitions, the binary
reporting thresholds, the feature-importance plot size, the demo-generation
seeds and size, and the estimator search knobs. It is a frozen, slotted
dataclass so a stage cannot mutate the configuration it was handed, and
``CONFIG`` is the one module-level instance the entry point and tests reuse.

The field names and order mirror the core-side ``ProjectConfig`` Protocol in
``src/core/paths.py`` exactly. Core types the configuration structurally and
never imports this module, so the two declarations must be edited together;
``tests/test_project_config.py`` pins that conformance.

``data_root`` is always a path relative to a caller-supplied project root. No
function in this module resolves it against the current working directory:
the project layer binds it when it builds a pipeline layout, and tests inject a
temporary root with a matching ``data_root``. There is deliberately no
``input_filenames`` setting — external input filenames belong to the
``InputTableSpec`` entries in ``src/project/inputs.py`` so they have exactly
one owner.

The module-level tables below are reporting policy, not data contracts:
``LOG_PLOT_BASES`` lists the logarithmic axis bases the report layer may use,
``DISPLAY_REGRESSION_METRICS`` names the regression metric families shown in
the per-outcome display table, and ``DISPLAY_METRIC_DECIMALS`` fixes how many
decimal places that table prints. They carry no subject-matter meaning, so a
new project can keep them unchanged or override them locally.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Immutable settings shared by every pipeline stage.

    Attributes:
        data_root: Directory holding external input tables, relative to an
            explicit project root. Never resolved against the process cwd.
        feature_columns_to_drop: Feature names the training stage must exclude
            before fitting, applied after the feature table is validated.
        train_end_date: Last case date belonging to the ``train`` partition.
        test_start_date: First case date belonging to the ``test`` partition;
            the ``valid`` partition spans the dates strictly between the two
            boundaries. Must be later than ``train_end_date``.
        binary_thresholds: Regression-scale thresholds used to derive binary
            labels, with the strict rule ``value > threshold``.
        evaluated_partitions: Partitions that receive metrics, plots, and
            binary labels. Held-out partitions stay unevaluated.
        feature_importance_plot_top_n: Number of top-ranked features drawn in
            the feature-importance plot.
        demo_seed: Seed for every demo data generator; a single value makes
            the whole demo reproducible.
        demo_case_count: Number of source rows the demo generator emits.
        xgb_single_fit_params: Fixed estimator parameters for a single fit.
        xgb_num_boost_round: Boosting rounds for a single fit.
        xgb_early_stopping_rounds: Rounds without improvement before a single
            fit stops.
        xgb_regressor_seed: Estimator seed for a single fit.
        xgb_search_seed: Estimator seed used during parameter search.
        xgb_param_distributions: Parameter-name -> candidate distribution map
            for the search strategy.
        xgb_search_n_iter: Number of sampled parameter settings per search.
        xgb_search_cv: Cross-validation folds per search candidate.
    """

    data_root: str = "data/demo"
    feature_columns_to_drop: tuple[str, ...] = ()
    train_end_date: date = date(2024, 1, 1)
    test_start_date: date = date(2024, 7, 1)
    binary_thresholds: tuple[float, ...] = (-0.5, 0.0, 0.5)
    evaluated_partitions: tuple[str, ...] = ("train", "valid")
    feature_importance_plot_top_n: int = 10
    demo_seed: int = 12345
    demo_case_count: int = 420
    xgb_single_fit_params: Mapping[str, Any] = field(default_factory=dict)
    xgb_num_boost_round: int = 40
    xgb_early_stopping_rounds: int = 5
    xgb_regressor_seed: int = 7
    xgb_search_seed: int = 11
    xgb_param_distributions: Mapping[str, Any] = field(default_factory=dict)
    xgb_search_n_iter: int = 4
    xgb_search_cv: int = 3


CONFIG: Final[ProjectConfig] = ProjectConfig()
"""The configuration instance the template entry point runs with."""


LOG_PLOT_BASES: Final[tuple[float, ...]] = (10.0, 2.0)
"""Logarithmic axis bases the report layer may use, most common first."""

DISPLAY_REGRESSION_METRICS: Final[tuple[str, ...]] = (
    "rmse",
    "r2",
    "mean_ae",
    "median_ae",
    "d2_absolute_error",
)
"""Regression metric families shown in the per-outcome display table.

Each name is a ``RegressionMetrics`` field suffix: the report layer renders
both ``original_target_<name>`` and ``model_target_<name>`` for every entry.
"""

DISPLAY_METRIC_DECIMALS: Final[int] = 4
"""Decimal places used when the display table renders metric values."""
