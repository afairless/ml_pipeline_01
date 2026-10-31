"""Project-side estimator construction.

The estimator-agnostic training mechanics live in ``src.core.training``; this
module owns the small amount of estimator-specific construction the project
training driver needs. Keeping it here means the driver's flow — load, apply a
feature policy, prepare arrays, dispatch on the strategy's trainer, persist —
contains no estimator class or library detail.

The module exposes four names:

``ESTIMATOR_ID``
    The registry id of the family every project strategy trains with. It is
    the manifest's ``estimator`` vocabulary and the key the project resolves
    through ``src.core.estimator.estimator_registry``.
``make_single_fit_adapter()``
    A fresh, unfitted adapter for the single-holdout-fit trainer. The caller
    supplies the parameter mapping when it calls the core ``fit_single``
    mechanic, so this factory carries no strategy-specific settings.
``make_search_adapter_factory(objective, regressor_seed)``
    A zero-argument factory yielding a fresh adapter around an unfitted base
    regressor template. The core randomized-search mechanic clones that
    template for every candidate; the closure fixes the strategy objective and
    the estimator-held seed the search must reproduce.
``booster_of(adapter)``
    The wrapped native booster of a fitted adapter, normalized across the two
    shapes the pipeline produces: the booster from a single fit and the
    sklearn regressor wrapping a search's best estimator. Artifact writers
    need one shape, so callers do not branch on the training mechanic.

Everything here is project-agnostic in the same sense as the rest of
``src.project``: it names an estimator family, never a stage, outcome, run
directory, or manifest field.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import xgboost as xgb

from src.core.estimator import EstimatorAdapter, estimator_registry
from src.core.estimators.xgboost import ESTIMATOR_ID

__all__ = [
    "ESTIMATOR_ID",
    "booster_of",
    "make_search_adapter_factory",
    "make_single_fit_adapter",
]


def make_single_fit_adapter() -> EstimatorAdapter:
    """Return a fresh, unfitted adapter for the single-fit trainer.

    The adapter is resolved through the estimator registry, so the project
    names one family id (``ESTIMATOR_ID``) rather than importing its concrete
    class. A new adapter per call keeps a caller from reusing a fitted model
    across runs; the core ``fit_single`` mechanic fits it in place.
    """
    adapter_type = estimator_registry.get(ESTIMATOR_ID)
    return adapter_type()


def make_search_adapter_factory(
    objective: str,
    regressor_seed: int,
) -> Callable[[], EstimatorAdapter]:
    """Return a factory for the randomized-search base-estimator template.

    ``run_param_search`` calls the returned factory once to obtain the
    template it hands to ``RandomizedSearchCV``; sklearn then clones that
    estimator for every candidate fit. A fresh template per factory call keeps
    the search from reusing a fitted estimator across runs.

    Args:
        objective: The native estimator objective the strategy declares.
        regressor_seed: The estimator-held seed for the search template. It is
            separate from the search's own random state because the two
            control different draws.

    Returns:
        A zero-argument callable returning a fresh unfitted adapter whose
        wrapped model is a fresh, sklearn-compatible base estimator template.
    """

    def factory() -> EstimatorAdapter:
        # The registry returns the adapter *type*; the protocol surface does
        # not declare a constructor that accepts a wrapped model, while the
        # concrete adapter contract does. Cast at this single construction
        # boundary rather than widening the protocol every family implements.
        adapter_type = cast(
            Callable[[Any], EstimatorAdapter],
            estimator_registry.get(ESTIMATOR_ID),
        )
        return adapter_type(
            xgb.XGBRegressor(objective=objective, random_state=regressor_seed)
        )

    return factory


def booster_of(adapter: EstimatorAdapter) -> xgb.Booster:
    """Return the native xgboost booster wrapped by a fitted adapter.

    A single-fit adapter wraps the ``xgb.Booster`` produced by ``xgb.train``;
    a search adapter re-wraps the winning sklearn regressor. Artifact writers
    need the booster in one shape, so the normalization lives here instead of
    in every caller. The wrapped model is read through the adapter's public
    ``model`` attribute, which the adapter contract documents.
    """
    model = cast(Any, adapter).model
    if isinstance(model, xgb.Booster):
        return model
    return model.get_booster()
