"""XGBoost estimator adapter.

Wraps both model shapes the pipeline produces behind the
``EstimatorAdapter`` protocol (``src.core.estimator``): the ``xgb.Booster`` a
single ``xgb.train`` fit returns and the ``xgb.XGBRegressor`` a
randomized-search result yields (``best_estimator_``). Everything the
downstream stages need — ``predict``, ``predict_proba``, ``task`` /
``objective``, feature importances, serialization — goes through the adapter,
so no stage hand-constructs an ``xgb.XGBRegressor()``/``XGBClassifier()`` to
load a model.

The module is project-agnostic (``src.core`` keeps no ``src.project``
import; ``tests/test_module_imports.py`` enforces this for every core
module): it knows an xgboost model, never a stage name, run directory, or
manifest field.

``load`` is a classmethod on the adapter type *and* a module-level alias:
the registry's generic ``EstimatorRegistry.load(id, path)`` delegates to the
adapter type's ``load``, while ``from src.core.estimators.xgboost import
load`` remains a valid direct-use entry point.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

ESTIMATOR_ID = "xgboost"
"""The registry id of this adapter family (the manifest's ``estimator`` vocabulary)."""

FEATURE_COLNAME = "feature"
FEATURE_ESTIMATOR_ID_COLNAME = "feature_estimator_id"
FEATURE_IMPORTANCE_COLNAME = "importance"


class XGBoostAdapter:
    """Wrap a fitted xgboost model behind the ``EstimatorAdapter`` protocol.

    ``model`` is a fitted ``xgb.Booster`` (the single-``xgb.train`` path), a
    fitted ``xgb.XGBRegressor`` (the randomized-search path, via
    ``best_estimator_``), or ``None`` for an unfitted adapter completed by
    :meth:`fit`. ``task`` and ``objective`` are derived from the learner
    configuration of the wrapped booster — never from the wrapper type — so a
    booster and a regressor trained with the same objective report the same
    task (``task`` is the single source of truth, ``objective`` informative).
    """

    def __init__(self, model: Any = None) -> None:
        self._model = model

    @property
    def model(self) -> Any:
        """The wrapped model object (read-only; the constructor's ``model`` argument).

        Mirrors the constructor parameter so a consumer that must reach the
        concrete estimator — the randomized-search mechanics in
        ``src.core.training`` hand this to ``RandomizedSearchCV`` — does so
        through a public name instead of the private slot. An adapter built
        around an *unfitted* base estimator (the param-search factory
        template) exposes that template here; the fitted best estimator is
        then wrapped by the adapter type again.
        """
        return self._require_model()

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        params: Mapping[str, Any],
        num_boost_round: int,
        early_stopping_rounds: int,
        verbose_eval: bool = False,
    ) -> None:
        """Train a booster over the given holdout split and wrap it.

        Mirrors the pipeline's single-fit shape — ``xgb.train`` over DMatrix
        objects with early stopping on the validation split — so a driver
        that reproduces that call order feeds the identical sequence of model
        decisions into the library. Replaces the wrapped model in place; the
        adapter then behaves exactly like one constructed around the trained
        booster.
        """
        dmtx_train = xgb.DMatrix(x_train, label=y_train)
        dmtx_valid = xgb.DMatrix(x_valid, label=y_valid)
        self._model = xgb.train(
            params=dict(params),
            dtrain=dmtx_train,
            num_boost_round=num_boost_round,
            evals=[(dmtx_valid, "test")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the target for the rows of ``x`` (a 2-D feature frame).

        For a wrapped booster the prediction mirrors the sklearn wrapper's
        semantics under early stopping: when the booster records a
        ``best_iteration`` (an early-stopped ``xgb.train`` run), only the
        trees up to that iteration are used, exactly as
        ``xgb.XGBRegressor().load_model(...).predict()`` would apply early
        stopping. This keeps registry-loaded predictions identical to a
        sklearn wrapper loaded from the same file. A booster trained without
        early stopping has no ``best_iteration`` and predicts with all trees.
        """
        model = self._require_model()
        if isinstance(model, xgb.Booster):
            # ``xgb.Booster.predict`` takes a DMatrix, not a raw array; the
            # sklearn wrappers accept arrays directly.
            return model.predict(
                xgb.DMatrix(x), iteration_range=self._iteration_range()
            )
        return model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities (``n`` rows, 2 columns) for a binary adapter.

        A regression-task adapter raises ``ValueError`` — the rejection is by
        ``task``, not by wrapper type, so any future binary adapter (a
        classifier wrapper or a ``binary:*`` booster) reports the same
        behavior.
        """
        if self.task != "binary":
            raise ValueError(
                "predict_proba is only defined for binary-task adapters; "
                f"this adapter reports task {self.task!r}"
            )
        probabilities = self._booster().predict(
            xgb.DMatrix(x), iteration_range=self._iteration_range()
        )
        if probabilities.ndim == 2:
            return probabilities
        return np.column_stack((1.0 - probabilities, probabilities))

    @property
    def task(self) -> str:
        """Canonical task — ``'regression'`` or ``'binary'`` — for this model.

        Derived from the native objective (``'reg:*'`` → ``'regression'``,
        ``'binary:*'`` → ``'binary'``; anything else raises), so the manifest's
        task and the adapter's task cannot disagree.
        """
        objective = self.objective
        if objective.startswith("reg:"):
            return "regression"
        if objective.startswith("binary:"):
            return "binary"
        raise ValueError(
            f'unknown xgboost objective {objective!r}: expected a "reg:" or '
            '"binary:" objective string'
        )

    @property
    def objective(self) -> str:
        """The native xgboost objective read from the learner configuration.

        Reading the serialized config (rather than an attribute of the
        wrapper) keeps the answer identical for a ``xgb.Booster`` and for any
        sklearn wrapper whose booster carries the same objective.
        """
        config = json.loads(self._booster().save_config())
        objective_node = config.get("learner", {}).get("objective")
        name = objective_node.get("name") if isinstance(objective_node, dict) else None
        if name is None:
            raise ValueError(
                "cannot determine the native objective of the wrapped model: "
                "its learner configuration carries no objective"
            )
        return name

    def feature_importances(self, feature_colnames: Sequence[str]) -> pl.DataFrame:
        """Gain importances mapped back to the original feature names.

        ``feature_colnames`` are the training feature columns in training
        order: xgboost addresses features by column index (``f0``, ``f1``,
        ...), so the mapping is positional, exactly as the pipeline's
        artifact extractor maps them for the saved importances frame. A model
        referencing an undeclared feature index raises ``ValueError`` instead
        of silently dropping or mislabelling the row.
        """
        booster = self._booster()
        importances = booster.get_score(importance_type="gain")
        feature_id_to_name = {f"f{i}": name for i, name in enumerate(feature_colnames)}
        try:
            names = [feature_id_to_name[feature_id] for feature_id in importances]
        except KeyError as err:
            raise ValueError(
                f"feature {err.args[0]!r} from the model is not among the "
                "declared feature columns; pass the training columns in "
                "training order"
            ) from err
        return pl.DataFrame(
            {
                FEATURE_COLNAME: names,
                FEATURE_ESTIMATOR_ID_COLNAME: list(importances.keys()),
                FEATURE_IMPORTANCE_COLNAME: list(importances.values()),
            }
        )

    def save(self, path: str | Path) -> None:
        """Serialize the wrapped model to ``path`` (xgboost-native format)."""
        self._booster().save_model(path)

    @classmethod
    def load(cls, path: str | Path) -> XGBoostAdapter:
        """Deserialize an xgboost model file and wrap it as a fitted adapter.

        The deserialization entry point the registry delegates to; a booster
        loaded this way reports the objective/task recorded in its learner
        configuration, exactly like the adapter that saved it.
        """
        booster = xgb.Booster()
        booster.load_model(path)
        return cls(booster)

    def _require_model(self) -> Any:
        """The wrapped model, or a clear error for an unfitted adapter."""
        if self._model is None:
            raise ValueError(
                "this adapter wraps no fitted model; construct it around a "
                "trained booster/regressor or call fit() first"
            )
        return self._model

    def _iteration_range(self) -> tuple[int, int]:
        """The best-iteration prediction range of the wrapped booster.

        ``(0, best_iteration + 1)`` when the booster records an early-stopping
        checkpoint, ``(0, 0)`` ("all trees", xgboost's iteration_range
        vocabulary) otherwise — the same range the sklearn wrappers compute
        when ``best_iteration`` is set, so a booster and a wrapper trained
        identically predict identically.
        """
        try:
            best_iteration = self._booster().best_iteration
        except AttributeError:
            # Trained without early stopping: no checkpoint is recorded, and
            # ``Booster.best_iteration`` raises where ``XGBModel`` has no
            # attribute at all — both mean "use every tree".
            return (0, 0)
        return (0, best_iteration + 1)

    def _booster(self) -> xgb.Booster:
        """The wrapped booster, whatever wrapper type holds it."""
        model = self._require_model()
        if isinstance(model, xgb.Booster):
            return model
        return model.get_booster()


# The module-level deserialization entry point: the classmethod above and
# this name are the same function, so ``from src.core.estimators.xgboost
# import load`` works for direct use while the registry can reach the loader
# generically through the adapter type.
load = XGBoostAdapter.load
