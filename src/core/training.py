"""Estimator-agnostic training mechanics.

Holds exactly the two helpers the project training driver combines with its
split loaders, feature policies, and artifact writers — and nothing else:

- ``fit_single``: one model, one holdout split. The adapter's own ``fit``
  reproduces the pipeline's single-boosting call shape, so this function only
  forwards the plain numeric frames and the parameter mapping.
- ``run_param_search``: a randomized hyperparameter search over a factory-
  supplied base estimator template, returning the fitted adapter plus the
  tuned parameter mapping (the ``best_parameters`` the model manifest
  records).

Both functions are deliberately dumb about the project: they accept an
adapter (or a factory yielding one) and plain ``numpy`` frames, and know
nothing about stage names, run directories, manifests, outcomes, or
``src.project`` — the same one-way layering ``tests/test_module_imports.py``
enforces for every core module. The sklearn ``RandomizedSearchCV`` call shape
matches the project's randomized-search strategy trainers, so a driver that
reproduces their argument order feeds the identical sequence of parameter
draws into the library: draw order is part of the value contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
from sklearn.model_selection import RandomizedSearchCV

from src.core.estimator import EstimatorAdapter


def fit_single(
    adapter: EstimatorAdapter,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: Mapping[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> EstimatorAdapter:
    """Fit one model on the holdout split and return the fitted adapter.

    The fitting itself lives on the adapter: ``EstimatorAdapter.fit`` runs
    the single-fit boosting call. This helper only fixes the argument order
    the driver calls every single-fit strategy with, so a strategy specifies
    *parameters* and the driver specifies *where they go*, never the library
    call.
    """
    adapter.fit(
        x_train,
        y_train,
        x_valid,
        y_valid,
        params,
        num_boost_round,
        early_stopping_rounds,
    )
    return adapter


def run_param_search(
    adapter_factory: Callable[[], EstimatorAdapter],
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_distributions: Mapping[str, Any],
    n_iter: int,
    cv: int,
    scoring: str,
    random_state: int,
    n_jobs: int,
) -> tuple[EstimatorAdapter, dict[str, Any]]:
    """Randomized search; returns the fitted adapter and the tuned mapping.

    ``adapter_factory`` is a zero-argument callable returning a fresh,
    unfitted adapter whose wrapped model is a fresh, sklearn-compatible
    **base estimator template** (e.g. a boosted-tree regressor wrapped by an
    adapter); the base estimator is exposed as the adapter's ``model``
    attribute. The search runs ``RandomizedSearchCV`` over that template —
    the randomized-search strategies' call shape: same estimator-held seed,
    same distributions, same ``n_iter``/``cv``/``scoring``/``random_state``/
    ``n_jobs`` arguments — and fits on the train frame only, because the
    randomized-search strategies never use a validation split for the search.
    The winning ``best_estimator_`` is re-wrapped in the template's adapter
    type and returned alongside ``best_params_``, so the caller gets one
    fitted, protocol-conforming object plus the tuned parameter mapping the
    model manifest records.
    """
    template = adapter_factory()
    # The search needs the template's wrapped estimator, which the adapter
    # protocol deliberately does not expose; the documented factory contract
    # requires it (``template.model``) together with an adapter type that can
    # wrap the winning estimator. Cast at this single library boundary rather
    # than widening the protocol every adapter must satisfy.
    template_with_model = cast(Any, template)
    search = RandomizedSearchCV(
        estimator=template_with_model.model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    search.fit(x_train, y_train)
    adapter_type = cast(Callable[[Any], EstimatorAdapter], type(template))
    fitted = adapter_type(search.best_estimator_)
    return fitted, dict(search.best_params_)
