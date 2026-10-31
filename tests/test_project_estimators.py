"""Unit tests for the project estimator factory.

The factory is the one place the project layer names an estimator family. The
training driver will combine these helpers with the core training mechanics,
so a failure here means a strategy could be trained with the wrong estimator,
the wrong objective, a reused model template, or an adapter shape the artifact
writers cannot consume.

Covered contracts:

- ``ESTIMATOR_ID`` is the registry id and resolves to the adapter type;
- ``make_single_fit_adapter`` returns a fresh, unfitted adapter that the core
  single-fit mechanic can fit and whose wrapped booster is recoverable;
- ``make_search_adapter_factory`` captures the objective and estimator seed,
  yields a fresh base-estimator template per call, and works with the core
  randomized-search mechanic;
- ``booster_of`` normalizes both wrapped model shapes — a native booster and a
  sklearn regressor — to the native booster.

Every fit uses a tiny locally seeded frame, so these tests never touch the
pipeline's global-RNG draw-order discipline.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import xgboost as xgb

from src.core.estimator import estimator_registry
from src.core.estimators.xgboost import XGBoostAdapter
from src.core.training import fit_single, run_param_search
from src.project.estimators import (
    ESTIMATOR_ID,
    booster_of,
    make_search_adapter_factory,
    make_single_fit_adapter,
)

SINGLE_FIT_PARAMS: dict[str, object] = {
    "objective": "reg:squarederror",
    "max_depth": 2,
    "seed": 7,
}
"""Small, fast parameters for the single-fit path."""

SEARCH_PARAM_DISTRIBUTIONS: dict[str, list[object]] = {
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1],
}
"""A tiny search space; every candidate is a sub-second fit."""


@pytest.fixture(scope="module")
def tiny_data() -> tuple[np.ndarray, np.ndarray]:
    """A tiny deterministic regression frame (24 rows x 3 features)."""
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(24, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=24)
    return x, y


def _fit_booster(x: np.ndarray, y: np.ndarray) -> xgb.Booster:
    """A tiny fitted booster on the single-fit path."""
    return xgb.train(
        {"objective": "reg:squarederror", "max_depth": 2, "seed": 7},
        xgb.DMatrix(x, label=y),
        num_boost_round=5,
    )


def test_estimator_id_is_the_registered_family_id() -> None:
    """The project id names the registered adapter family."""
    assert ESTIMATOR_ID == "xgboost"
    assert estimator_registry.get(ESTIMATOR_ID) is XGBoostAdapter


def test_single_fit_adapter_is_fresh_and_unfitted() -> None:
    """Each call yields its own unfitted adapter, not a shared fitted model."""
    first = make_single_fit_adapter()
    second = make_single_fit_adapter()
    assert isinstance(first, XGBoostAdapter)
    assert isinstance(second, XGBoostAdapter)
    assert first is not second
    with pytest.raises(ValueError, match="wraps no fitted model"):
        _ = first.model


def test_single_fit_adapter_fits_through_the_core_mechanic(
    tiny_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """The factory output is directly usable by ``fit_single``."""
    x, y = tiny_data
    adapter = make_single_fit_adapter()
    fitted = fit_single(
        adapter,
        x[:16],
        y[:16],
        x[16:],
        y[16:],
        SINGLE_FIT_PARAMS,
        num_boost_round=10,
        early_stopping_rounds=3,
    )
    assert fitted is adapter
    assert fitted.task == "regression"
    assert fitted.objective == SINGLE_FIT_PARAMS["objective"]
    assert isinstance(booster_of(fitted), xgb.Booster)
    assert booster_of(fitted) is cast(Any, fitted).model


def test_search_factory_yields_fresh_seeded_templates() -> None:
    """The closure captures the objective and seed on every fresh template."""
    factory = make_search_adapter_factory("reg:absoluteerror", 11)
    first = factory()
    second = factory()
    assert isinstance(first, XGBoostAdapter)
    assert isinstance(second, XGBoostAdapter)
    assert first is not second
    first_model = first.model
    second_model = second.model
    assert isinstance(first_model, xgb.XGBRegressor)
    assert isinstance(second_model, xgb.XGBRegressor)
    assert first_model is not second_model
    assert first_model.get_params()["objective"] == "reg:absoluteerror"
    assert first_model.get_params()["random_state"] == 11
    assert second_model.get_params()["objective"] == "reg:absoluteerror"
    assert second_model.get_params()["random_state"] == 11


def test_search_factory_works_with_the_core_search(
    tiny_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """The template passes through ``run_param_search`` and stays a booster."""
    x, y = tiny_data
    factory = make_search_adapter_factory("reg:absoluteerror", 7)
    fitted, tuned = run_param_search(
        factory,
        x,
        y,
        SEARCH_PARAM_DISTRIBUTIONS,
        n_iter=2,
        cv=2,
        scoring="neg_mean_absolute_error",
        random_state=5,
        n_jobs=1,
    )
    assert isinstance(fitted, XGBoostAdapter)
    assert fitted.task == "regression"
    assert fitted.objective == "reg:absoluteerror"
    assert tuned
    booster = booster_of(fitted)
    assert isinstance(booster, xgb.Booster)
    assert booster is cast(Any, fitted).model.get_booster()


def test_booster_of_returns_the_native_booster_unchanged(
    tiny_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """A booster-shaped adapter is normalized without rebuilding the model."""
    x, y = tiny_data
    booster = _fit_booster(x, y)
    adapter = XGBoostAdapter(booster)
    assert booster_of(adapter) is booster


def test_booster_of_unwraps_a_sklearn_regressor(
    tiny_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """A regressor-shaped adapter yields the regressor's own booster."""
    x, y = tiny_data
    regressor = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=5, random_state=7
    )
    regressor.fit(x, y)
    adapter = XGBoostAdapter(regressor)
    assert booster_of(adapter) is regressor.get_booster()
