"""Unit tests for the frozen project configuration and reporting policy.

Pins three contracts the rest of the project layer depends on. First,
``ProjectConfig`` is frozen and slotted, so no stage can rewrite the settings
it was handed and no arbitrary attribute can appear at runtime. Second, its
field names, order, and types mirror the core-side ``ProjectConfig`` Protocol
exactly, so ``PipelineContext`` accepts ``CONFIG`` without core importing the
project layer. Third, the module-level reporting tables are immutable,
duplicate-free policy whose metric names correspond to real
``RegressionMetrics`` fields.

A failure here means a stage could run with different settings than the ones
the entry point declared, or the two config declarations could drift apart.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from datetime import date
from typing import Any, Final, get_args, get_origin, get_type_hints

import pytest

from src.core.metrics import RegressionMetrics
from src.core.paths import ProjectConfig as CoreProjectConfig
from src.project import config
from src.project.config import (
    CONFIG,
    DISPLAY_METRIC_DECIMALS,
    DISPLAY_REGRESSION_METRICS,
    LOG_PLOT_BASES,
    ProjectConfig,
)


def _unwrap_final(hint: Any) -> Any:
    """Return ``hint`` without a ``Final[...]`` wrapper, if it has one.

    The core Protocol declares every field read-only with ``Final``; the
    project dataclass expresses the same guarantee through ``frozen=True``.
    Unwrapping lets the type comparison focus on the value type both sides
    actually agree on.
    """
    if get_origin(hint) is Final:
        return get_args(hint)[0]
    return hint


def test_config_constructs_from_the_declared_defaults() -> None:
    """The default instance carries the template's documented settings.

    These values are the demo contract: the date boundaries split the demo
    case dates into train/valid/test, the thresholds must yield both binary
    classes on the evaluated partitions, and the seeds keep every generated
    artifact reproducible.
    """
    assert CONFIG.data_root == "data/demo"
    assert CONFIG.feature_columns_to_drop == ()
    assert CONFIG.train_end_date == date(2024, 1, 1)
    assert CONFIG.test_start_date == date(2024, 7, 1)
    assert CONFIG.binary_thresholds == (-0.5, 0.0, 0.5)
    assert CONFIG.evaluated_partitions == ("train", "valid")
    assert CONFIG.feature_importance_plot_top_n == 10
    assert CONFIG.demo_seed == 12345
    assert CONFIG.demo_case_count == 420
    assert CONFIG.xgb_single_fit_params == {}
    assert CONFIG.xgb_num_boost_round == 40
    assert CONFIG.xgb_early_stopping_rounds == 5
    assert CONFIG.xgb_regressor_seed == 7
    assert CONFIG.xgb_search_seed == 11
    assert CONFIG.xgb_param_distributions == {}
    assert CONFIG.xgb_search_n_iter == 4
    assert CONFIG.xgb_search_cv == 3


def test_module_level_config_is_the_default_instance() -> None:
    """``CONFIG`` is an ordinary ``ProjectConfig`` and the same object."""
    assert isinstance(CONFIG, ProjectConfig)
    assert config.CONFIG is CONFIG
    assert CONFIG == ProjectConfig()


def test_default_partition_boundaries_are_ordered() -> None:
    """The test boundary is strictly later than the train boundary.

    The split predicate relies on ``test_start_date > train_end_date``; equal
    or inverted dates would make ``valid`` empty or overlapping.
    """
    assert CONFIG.test_start_date > CONFIG.train_end_date


def test_config_is_frozen() -> None:
    """Assigning to a field raises instead of changing shared settings."""
    with pytest.raises(FrozenInstanceError):
        setattr(CONFIG, "demo_seed", 1)


def test_config_uses_slots_and_rejects_unknown_attributes() -> None:
    """A slotted config carries no ``__dict__`` and cannot grow attributes.

    Assigning an undeclared name fails; with a frozen slotted dataclass the
    interpreter may surface that as ``TypeError`` rather than
    ``AttributeError``, so the test accepts either rejection and then proves
    the attribute never appeared.
    """
    assert not hasattr(CONFIG, "__dict__")
    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        setattr(CONFIG, "not_a_declared_field", True)
    assert not hasattr(CONFIG, "not_a_declared_field")


def test_mapping_defaults_are_independent_per_instance() -> None:
    """Each instance gets its own empty mapping, never a shared mutable one.

    ``field(default_factory=dict)`` is required because a bare ``{}`` default
    would be shared across instances. The maps are read-only by convention,
    so isolation is what keeps one caller's edits out of another's config.
    """
    first = ProjectConfig()
    second = ProjectConfig()
    assert first.xgb_single_fit_params is not second.xgb_single_fit_params
    assert first.xgb_param_distributions is not second.xgb_param_distributions
    configured = ProjectConfig(xgb_single_fit_params={"n_estimators": 1})
    assert configured.xgb_single_fit_params == {"n_estimators": 1}
    assert second.xgb_single_fit_params == {}


def test_field_names_order_and_types_mirror_the_core_protocol() -> None:
    """Both declarations agree on every field, in the same positional order.

    ``PipelineContext`` is typed by the core Protocol, and tests build contexts
    from this dataclass. Matching names *and* order means the conformance is
    real rather than accidental, and a field added on one side must be added
    on the other.
    """
    project_hints = get_type_hints(ProjectConfig)
    core_hints = {
        name: _unwrap_final(hint)
        for name, hint in get_type_hints(CoreProjectConfig).items()
    }
    assert tuple(project_hints) == tuple(core_hints)
    assert project_hints == core_hints


def test_config_satisfies_the_core_protocol_at_runtime() -> None:
    """``src.core`` accepts ``CONFIG`` structurally, with no project import."""
    assert isinstance(CONFIG, CoreProjectConfig)


def test_log_plot_bases_are_immutable_distinct_and_valid() -> None:
    """Every log base is a usable, unique base greater than one."""
    assert isinstance(LOG_PLOT_BASES, tuple)
    assert LOG_PLOT_BASES
    assert all(isinstance(base, float) for base in LOG_PLOT_BASES)
    assert all(base > 1.0 for base in LOG_PLOT_BASES)
    assert len(set(LOG_PLOT_BASES)) == len(LOG_PLOT_BASES)


def test_display_regression_metrics_are_immutable_and_distinct() -> None:
    """The display table names each metric family at most once."""
    assert isinstance(DISPLAY_REGRESSION_METRICS, tuple)
    assert DISPLAY_REGRESSION_METRICS
    assert all(isinstance(name, str) and name for name in DISPLAY_REGRESSION_METRICS)
    assert len(set(DISPLAY_REGRESSION_METRICS)) == len(DISPLAY_REGRESSION_METRICS)


@pytest.mark.parametrize("metric_name", DISPLAY_REGRESSION_METRICS)
def test_display_metric_names_match_regression_metrics_fields(
    metric_name: str,
) -> None:
    """Each display name is a real ``RegressionMetrics`` field suffix.

    The report layer renders both the original-scale and model-scale value, so
    both prefixed fields must exist; a renamed metric in core would otherwise
    silently blank the display table.
    """
    metric_fields = {metric_field.name for metric_field in fields(RegressionMetrics)}
    assert f"original_target_{metric_name}" in metric_fields
    assert f"model_target_{metric_name}" in metric_fields


def test_display_metric_decimals_is_a_positive_int() -> None:
    """The decimal count is a plain positive ``int``, not a bool or float."""
    assert type(DISPLAY_METRIC_DECIMALS) is int
    assert DISPLAY_METRIC_DECIMALS > 0
