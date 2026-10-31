"""Unit tests for the core estimator abstraction.

Covers the registry (unknown-id rejection for both ``get`` and ``load``, id ->
type resolution, load delegation) and the xgboost adapter (task/objective
reporting for both wrapper shapes — the ``xgb.Booster`` single-fit path and
the ``xgb.XGBRegressor`` search-result path —, save/load round-trips, the
single-fit call shape, early-stopping prediction semantics, binary
probability output, feature-importance name mapping, and rejection of an
unfitted adapter). A failure in any of these means a stage could name one
estimator in its manifest and fit or predict with another.

The import-contract guard for these modules lives in
``tests/test_module_imports.py``, which scans every ``src.core`` module for a
``src.project`` or stage import.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from src.core.estimator import EstimatorRegistry, estimator_registry
from src.core.estimators import load
from src.core.estimators.xgboost import XGBoostAdapter


@pytest.fixture(scope="module")
def tiny_data() -> tuple[np.ndarray, np.ndarray]:
    """A tiny deterministic regression frame (30 rows x 3 features).

    Small enough to keep every model in this file a sub-second fit, and
    seeded locally (``np.random.default_rng``) so the tests never touch the
    pipeline's global-RNG draw-order discipline.
    """
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(30, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=30)
    return x, y


def _fit_booster(x: np.ndarray, y: np.ndarray) -> xgb.Booster:
    """A tiny fitted booster on the single-fit (``xgb.train``) path."""
    return xgb.train(
        {"objective": "reg:squarederror", "max_depth": 2, "seed": 7, "nthread": 1},
        xgb.DMatrix(x, label=y),
        num_boost_round=5,
    )


def _fit_early_stopped_booster(x: np.ndarray, y: np.ndarray) -> xgb.Booster:
    """A tiny booster trained with early stopping, so it records a
    ``best_iteration``.

    Enough rounds and a shallow tree on the small frame make the run stop
    before the round budget in practice; the pinned semantics below
    hold either way (the assertions compare against the same range, never
    against a "stops early" assumption).
    """
    return xgb.train(
        {
            "objective": "reg:squarederror",
            "max_depth": 1,
            "seed": 7,
            "nthread": 1,
        },
        xgb.DMatrix(x, label=y),
        num_boost_round=50,
        evals=[(xgb.DMatrix(x, label=y), "train")],
        early_stopping_rounds=5,
    )


def _fit_regressor(x: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
    """A tiny fitted regressor on the search-result (``best_estimator_``) path."""
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=2,
        n_estimators=5,
        random_state=7,
        n_jobs=1,
    )
    model.fit(x, y)
    return model


class TestEstimatorRegistry:
    """Registry id resolution: known id -> adapter type, unknown id -> raise."""

    def test_get_rejects_unknown_id(self) -> None:
        with pytest.raises(ValueError, match="unknown estimator id"):
            estimator_registry.get("sklearn")

    def test_load_rejects_unknown_id_before_touching_path(self) -> None:
        # The path is never read: resolution happens first, so a missing file
        # would surface as FileNotFoundError only for a *known* id.
        missing = Path("/does/not/exist.json")
        with pytest.raises(ValueError, match="unknown estimator id"):
            estimator_registry.load("sklearn", missing)

    def test_get_resolves_registered_id_to_adapter_type(self) -> None:
        assert estimator_registry.get("xgboost") is XGBoostAdapter

    def test_empty_registry_rejects_every_id(self) -> None:
        registry = EstimatorRegistry({})
        with pytest.raises(ValueError, match="unknown estimator id"):
            registry.get("xgboost")

    def test_registry_load_round_trip(
        self, tiny_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        x, y = tiny_data
        adapter = XGBoostAdapter(_fit_regressor(x, y))
        path = tmp_path / "model.json"
        adapter.save(path)
        reloaded = estimator_registry.load("xgboost", path)
        assert reloaded.task == adapter.task
        assert reloaded.objective == adapter.objective
        np.testing.assert_array_equal(reloaded.predict(x), adapter.predict(x))


class TestXGBoostAdapter:
    """Adapter behavior over both wrapped shapes (booster and regressor)."""

    def test_booster_adapter_reports_regression_task_and_objective(
        self, tiny_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = tiny_data
        adapter = XGBoostAdapter(_fit_booster(x, y))
        assert adapter.task == "regression"
        assert adapter.objective == "reg:squarederror"

    def test_regressor_adapter_reports_regression_task_and_objective(
        self, tiny_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = tiny_data
        adapter = XGBoostAdapter(_fit_regressor(x, y))
        assert adapter.task == "regression"
        assert adapter.objective == "reg:squarederror"

    def test_unrecognized_objective_raises_from_task(
        self, tiny_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        # ``count:poisson`` is a real xgboost objective, but neither a
        # regression nor a binary task under this adapter's mapping: the
        # rejection must be loud rather than guessed.
        x, y = tiny_data
        booster = xgb.train(
            {"objective": "count:poisson", "max_depth": 1, "seed": 7, "nthread": 1},
            xgb.DMatrix(x, label=np.abs(y) + 1.0),
            num_boost_round=2,
        )
        adapter = XGBoostAdapter(booster)
        assert adapter.objective == "count:poisson"
        with pytest.raises(ValueError, match="unknown xgboost objective"):
            _ = adapter.task

    def test_regression_adapter_raises_on_predict_proba(
        self, tiny_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = tiny_data
        for adapter in (
            XGBoostAdapter(_fit_booster(x, y)),
            XGBoostAdapter(_fit_regressor(x, y)),
        ):
            with pytest.raises(ValueError, match="predict_proba"):
                adapter.predict_proba(x)

    def test_unfitted_adapter_rejects_every_model_operation(self) -> None:
        adapter = XGBoostAdapter()
        with pytest.raises(ValueError, match="wraps no fitted model"):
            _ = adapter.task
        with pytest.raises(ValueError, match="wraps no fitted model"):
            adapter.predict(np.zeros((1, 3)))
        with pytest.raises(ValueError, match="wraps no fitted model"):
            adapter.save(Path("/does/not/exist.json"))

    def test_save_load_round_trip_preserves_predictions(
        self, tiny_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        x, y = tiny_data
        for adapter in (
            XGBoostAdapter(_fit_booster(x, y)),
            XGBoostAdapter(_fit_regressor(x, y)),
        ):
            path = tmp_path / "model.json"
            adapter.save(path)
            reloaded = load(path)
            assert reloaded.task == adapter.task
            assert reloaded.objective == adapter.objective
            np.testing.assert_array_equal(reloaded.predict(x), adapter.predict(x))

    def test_fit_faithfully_ports_the_single_fit_call_shape(
        self, tiny_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = tiny_data
        params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "seed": 7,
            "nthread": 1,
        }
        adapter = XGBoostAdapter()
        adapter.fit(
            x[:20],
            y[:20],
            x[20:],
            y[20:],
            params,
            num_boost_round=5,
            early_stopping_rounds=3,
        )
        # The adapter's fit mirrors the single-fit call shape exactly, so it
        # must reproduce the direct ``xgb.train`` call on the same inputs,
        # wrap the trained booster, and report the same identity.
        direct = xgb.train(
            dict(params),
            xgb.DMatrix(x[:20], label=y[:20]),
            num_boost_round=5,
            evals=[(xgb.DMatrix(x[20:], label=y[20:]), "test")],
            early_stopping_rounds=3,
        )
        assert adapter.task == "regression"
        assert adapter.objective == "reg:squarederror"
        # Both sides predict with the adapter's iteration semantics: an
        # early-stopped booster predicts up to its recorded ``best_iteration``
        # (mirroring the sklearn wrapper), so the reference call passes the
        # same range explicitly instead of silently comparing an all-trees
        # predict against a best-iteration one.
        try:
            iteration_range = (0, direct.best_iteration + 1)
        except AttributeError:
            iteration_range = (0, 0)
        np.testing.assert_array_equal(
            adapter.predict(x[:20]),
            direct.predict(xgb.DMatrix(x[:20]), iteration_range=iteration_range),
        )

    def test_booster_predict_honors_best_iteration_like_the_wrapper(
        self, tiny_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """A booster trained with early stopping predicts with its recorded
        ``best_iteration``, exactly as the sklearn wrapper does when the same
        model file is loaded into an ``xgb.XGBRegressor``."""
        x, y = tiny_data
        booster = _fit_early_stopped_booster(x, y)
        adapter = XGBoostAdapter(booster)
        reference = xgb.XGBRegressor()
        path = tmp_path / "model.json"
        adapter.save(path)
        reference.load_model(path)

        best_iteration = booster.best_iteration
        np.testing.assert_array_equal(adapter.predict(x), reference.predict(x))
        np.testing.assert_array_equal(
            adapter.predict(x),
            booster.predict(xgb.DMatrix(x), iteration_range=(0, best_iteration + 1)),
        )

    def test_binary_predict_proba_honors_best_iteration_like_the_wrapper(
        self, tiny_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """The same early-stopping semantics apply to ``predict_proba``: a
        binary booster trained with early stopping matches the
        ``xgb.XGBClassifier``-loaded reference probabilities."""
        x, y = tiny_data
        y_bin = (y > y.mean()).astype(np.float64)
        booster = xgb.train(
            {
                "objective": "binary:logistic",
                "max_depth": 1,
                "seed": 7,
                "nthread": 1,
            },
            xgb.DMatrix(x, label=y_bin),
            num_boost_round=50,
            evals=[(xgb.DMatrix(x, label=y_bin), "train")],
            early_stopping_rounds=5,
        )
        adapter = XGBoostAdapter(booster)
        assert adapter.task == "binary"
        assert adapter.objective == "binary:logistic"
        reference = xgb.XGBClassifier()
        path = tmp_path / "model.json"
        adapter.save(path)
        reference.load_model(path)
        probabilities = adapter.predict_proba(x)
        assert probabilities.shape == (x.shape[0], 2)
        np.testing.assert_array_equal(
            probabilities[:, 1], reference.predict_proba(x)[:, 1]
        )

    def test_feature_importances_map_feature_ids_to_names(
        self, tiny_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = tiny_data
        feature_colnames = ["feature_1", "feature_2", "feature_3"]
        adapter = XGBoostAdapter(_fit_booster(x, y))
        frame = adapter.feature_importances(feature_colnames)
        assert frame.columns == [
            "feature",
            "feature_estimator_id",
            "importance",
        ]
        feature_id_to_name = {f"f{i}": name for i, name in enumerate(feature_colnames)}
        for feature_id, feature_name in zip(
            frame["feature_estimator_id"], frame["feature"]
        ):
            assert feature_id_to_name[feature_id] == feature_name
        assert (frame["importance"] >= 0).all()

    def test_feature_importances_reject_undeclared_model_features(
        self, tiny_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = tiny_data
        adapter = XGBoostAdapter(_fit_booster(x, y))
        feature_colnames = ["feature_1", "feature_2", "feature_3"]
        used_ids = adapter.feature_importances(feature_colnames)[
            "feature_estimator_id"
        ].to_list()
        # Declare fewer columns than the highest feature index the model uses,
        # leaving that index unmapped: the mapping must raise rather than
        # mislabel the row.
        highest_index = max(int(feature_id[1:]) for feature_id in used_ids)
        with pytest.raises(ValueError, match="not among the declared feature"):
            adapter.feature_importances(feature_colnames[:highest_index])
