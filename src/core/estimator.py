"""Estimator abstraction: the adapter protocol and the id registry.

The project-agnostic surface stages use to fit, predict, and deserialize
models without naming an estimator library. Two pieces:

- ``EstimatorAdapter``: the protocol every concrete wrapper implements —
  ``fit``/``predict``/``predict_proba`` for the model lifecycle,
  ``task``/``objective`` for identity (``task`` is the single source of
  truth; ``objective`` is informative), and
  ``feature_importances``/``save`` for artifacts. ``load`` is the
  deserialization entry point: the registry's generic ``load(id, path)``
  delegates to the registered adapter type's ``load`` classmethod, and each
  adapter module exposes the same function at module level.
- ``EstimatorRegistry``: maps the manifest's ``estimator`` id vocabulary to
  adapter types. ``get`` resolves an id to its adapter type; ``load``
  reconstructs a fitted adapter from a saved model file, so a consumer
  rebuilds a model from the manifest's estimator id plus the declared model
  artifact without importing any estimator class. An unknown id raises
  ``ValueError`` — never a silent ``None``/fallback — so a mistyped manifest
  value fails loudly at the registry boundary.

``estimator_registry`` is the pipeline's default instance with the registered
families; new families add one module under ``src.core.estimators`` and one
entry here. The module is project-agnostic (``src.core`` keeps no
``src.project`` import; ``tests/test_module_imports.py`` enforces this for
every core module): it knows estimator ids and model files, never a stage
name, outcome, or run directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl

from src.core.estimators.xgboost import ESTIMATOR_ID, XGBoostAdapter


class EstimatorAdapter(Protocol):
    """The interface every concrete estimator wrapper implements.

    Attributes:
        task: Canonical task, ``'regression'`` or ``'binary'`` — the single
            source of truth for how the model is scored and branched on.
        objective: The native (estimator-family) objective string; informative
            next to ``task``.
    """

    @property
    def task(self) -> str:
        """Canonical task, ``'regression'`` or ``'binary'``."""
        ...

    @property
    def objective(self) -> str:
        """The native estimator-family objective string."""
        ...

    @classmethod
    def load(cls, path: str | Path) -> EstimatorAdapter:
        """Deserialize a saved model file into a fitted adapter."""
        ...

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        params: Mapping[str, object],
        num_boost_round: int,
        early_stopping_rounds: int,
    ) -> None:
        """Fit on the holdout split; the adapter then wraps the trained model."""
        ...

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the target for the rows of ``x`` (a 2-D feature frame)."""
        ...

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities for a binary task; a regression-task adapter
        raises ``ValueError`` — rejection is by ``task``, not by wrapper type."""
        ...

    def feature_importances(self, feature_colnames: Sequence[str]) -> pl.DataFrame:
        """Importance of each training feature, mapped to its column name."""
        ...

    def save(self, path: Path) -> None:
        """Serialize the fitted model to ``path`` (estimator-native format)."""
        ...


class EstimatorRegistry:
    """Map estimator family ids (the manifest's ``estimator`` vocabulary) to types.

    A registered adapter type must expose the ``load`` classmethod of
    ``EstimatorAdapter``; the module-level ``load`` in the family's submodule
    is the same function, so both the registry and direct imports reach one
    deserialization implementation.
    """

    def __init__(self, adapters: Mapping[str, type[EstimatorAdapter]]) -> None:
        # A copy: callers handing in a mutable mapping cannot later change
        # what this registry resolves.
        self._adapters: dict[str, type[EstimatorAdapter]] = dict(adapters)

    def get(self, estimator_id: str) -> type[EstimatorAdapter]:
        """The adapter type registered under ``estimator_id``.

        Raises:
            ValueError: No adapter is registered under ``estimator_id``.
        """
        try:
            return self._adapters[estimator_id]
        except KeyError:
            raise ValueError(
                f"unknown estimator id {estimator_id!r}; registered ids: "
                f"{', '.join(sorted(self._adapters)) or '(none)'}"
            ) from None

    def load(self, estimator_id: str, path: str | Path) -> EstimatorAdapter:
        """Reconstruct a fitted adapter of ``estimator_id`` from ``path``.

        The id is resolved first, so an unknown id raises before ``path`` is
        touched; the deserialization itself is the registered adapter type's
        ``load`` classmethod.
        """
        return self.get(estimator_id).load(path)


estimator_registry = EstimatorRegistry({ESTIMATOR_ID: XGBoostAdapter})
"""The pipeline's default registry (id -> adapter type).

New estimator families register here, one entry per family submodule under
``src.core.estimators``; stages and drivers read the registry rather than
importing adapter classes directly.
"""
