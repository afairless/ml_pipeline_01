"""Unit tests for the core training mechanics.

Covers the two estimator-agnostic helpers ``fit_single`` (single holdout fit
through an adapter) and ``run_param_search`` (randomized search over a
factory-supplied base estimator template): a single fit returns the fitted
adapter and honors the params it was given; a search returns a fitted adapter
plus a non-empty tuned mapping whose keys are a subset of the declared
distributions; both predict on a tiny frame. A failure here means the
mechanics would hand a stage an unfitted or mis-parameterized model.

All models run on a tiny, locally seeded frame so the tests never touch the
pipeline's global-RNG draw-order discipline. One layering guard mirrors
``tests/test_module_imports.py`` so ``src.core.training`` imports no
``src.project``/stage module.
"""

from __future__ import annotations

import ast
import importlib
import inspect

import numpy as np
import pytest
import xgboost as xgb

from src.core.estimators.xgboost import XGBoostAdapter
from src.core.training import fit_single, run_param_search


@pytest.fixture(scope="module")
def tiny_data() -> tuple[np.ndarray, np.ndarray]:
    """A tiny deterministic regression frame (24 rows x 3 features).

    Seeded locally (``np.random.default_rng``) exactly like
    ``tests/test_core_estimators.py``: small enough for sub-second fits, and
    untouched by the pipeline's global-RNG discipline.
    """
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(24, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=24)
    return x, y


SINGLE_FIT_PARAMS: dict[str, object] = {
    "objective": "reg:squarederror",
    "max_depth": 2,
    "seed": 7,
}
"""Small, fast single-fit parameters shared by the fit_single tests."""


def _search_param_distributions() -> dict[str, list[object]]:
    """A small search space: every candidate stays a sub-second fit.

    Plain lists only (no scipy frozen distributions) so the parameter draws
    come from sklearn's own seeded ``RandomState`` and never leak into the
    pipeline's global numpy RNG.
    """
    return {
        "max_depth": [2, 3],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }


def _search_adapter_factory() -> XGBoostAdapter:
    """A fresh adapter around a fresh, unfitted base estimator template.

    The exact factory shape a search strategy will hand ``run_param_search``:
    the base regressor carries its own seed (the pipeline's regressor-seed
    slot today) and the adapter wraps it.
    """
    return XGBoostAdapter(
        xgb.XGBRegressor(objective="reg:squarederror", n_estimators=10, random_state=7)
    )


class TestFitSingle:
    """Single holdout fit: one adapter, one fixed parameter mapping."""

    def test_returns_fitted_adapter_with_the_params_it_was_given(
        self, tiny_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = tiny_data
        adapter = XGBoostAdapter()
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
        # The helper fit in place and returned the same object, so the caller
        # keeps one adapter handle; the objective it was given is the one the
        # wrapped booster reports.
        assert fitted is adapter
        assert fitted.objective == SINGLE_FIT_PARAMS["objective"]
        assert fitted.task == "regression"
        predictions = fitted.predict(x)
        assert predictions.shape == (len(x),)
        assert np.isfinite(predictions).all()

    def test_predicts_on_a_tiny_frame_after_fit(
        self, tiny_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = tiny_data
        fitted = fit_single(
            XGBoostAdapter(),
            x[:16],
            y[:16],
            x[16:],
            y[16:],
            SINGLE_FIT_PARAMS,
            num_boost_round=10,
            early_stopping_rounds=3,
        )
        # Predictions for the training rows must track their targets far more
        # closely than for the held-out rows — evidence the fit actually used
        # the provided split rather than returning a no-op wrapper.
        train_error = np.mean(np.abs(fitted.predict(x[:16]) - y[:16]))
        valid_error = np.mean(np.abs(fitted.predict(x[16:]) - y[16:]))
        assert train_error < valid_error


class TestRunParamSearch:
    """Randomized search: a fitted adapter plus the tuned parameter mapping."""

    def test_returns_fitted_adapter_and_nonempty_tuned_mapping(
        self, tiny_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = tiny_data
        fitted, tuned = run_param_search(
            _search_adapter_factory,
            x,
            y,
            _search_param_distributions(),
            n_iter=3,
            cv=2,
            scoring="neg_mean_absolute_error",
            random_state=5,
            n_jobs=1,
        )
        assert isinstance(fitted, XGBoostAdapter)
        assert fitted.task == "regression"
        assert fitted.objective == "reg:squarederror"
        assert tuned, "the search must return at least one tuned parameter"
        predictions = fitted.predict(x)
        assert predictions.shape == (len(x),)
        assert np.isfinite(predictions).all()

    def test_tuned_params_are_a_subset_of_the_declared_distributions(
        self, tiny_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = tiny_data
        distributions = _search_param_distributions()
        _, tuned = run_param_search(
            _search_adapter_factory,
            x,
            y,
            distributions,
            n_iter=3,
            cv=2,
            scoring="neg_mean_absolute_error",
            random_state=5,
            n_jobs=1,
        )
        assert set(tuned) <= set(distributions)
        for name, value in tuned.items():
            assert value in distributions[name], (
                f"tuned {name}={value!r} was not drawn from its declared distribution"
            )


def _first_non_core_src_import(source: str) -> str | None:
    """Textual repr of the first ``src.*`` import outside ``src.core``, else None.

    Mirrors the AST technique of ``tests/test_module_imports.py`` so doc or
    comment mentions of ``src.project``/stage names cannot trip the check.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "src" or alias.name.startswith("src."):
                    if not (
                        alias.name == "src.core" or alias.name.startswith("src.core.")
                    ):
                        return f"import {alias.name}"
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == "src" or node.module.startswith("src."))
            and not (node.module == "src.core" or node.module.startswith("src.core."))
        ):
            return f"from {node.module} import ..."
    return None


def test_core_training_imports_no_project_or_stage_module() -> None:
    """``src.core.training`` carries no ``src.project``/stage import: a core
    module that imported the project layer would invert the one-way
    dependency."""
    module = importlib.import_module("src.core.training")
    forbidden = _first_non_core_src_import(inspect.getsource(module))
    assert forbidden is None, (
        f"src.core.training must not import project/stage modules; found: {forbidden}"
    )
