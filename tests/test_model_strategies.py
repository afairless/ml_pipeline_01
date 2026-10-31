"""Unit tests for the project model-strategy registry.

The registry is the model vocabulary the training and orchestration layers
resolve every run directory and manifest against, so these tests pin:

- the exact eight-entry table: ids, their run order, each entry's feature
  policy type and selected columns, the training mechanic, and the objective;
- the registry's structural guarantees: read-only mapping, key equal to
  ``ModelStrategy.id``, frozen and slotted entries, and a policy that exposes
  the ``FeaturePolicy`` surface;
- the construction guards: empty ids, empty objectives, and unknown trainers
  fail loudly instead of registering an unexecutable strategy.

A failure here means a model could train on different features, dispatch to
the wrong training mechanic, or land in a different run directory than the
pipeline declares.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any, cast, get_args

import pytest

from src.project.features import (
    FeatureSubset,
    IdentityFeatures,
    OutcomePlusNoise,
    PureNoise,
)
from src.project.models import (
    MODEL_STRATEGIES,
    REGRESSION_OBJECTIVE,
    TRAINER_KINDS,
    ModelStrategy,
    TrainerKind,
)

EXPECTED_STRATEGIES: tuple[tuple[str, type[Any], tuple[str, ...], str], ...] = (
    ("model_00_signal_0", IdentityFeatures, (), "fit_single"),
    ("model_00_signal_1", OutcomePlusNoise, (), "fit_single"),
    ("model_00_noise_0", PureNoise, (), "fit_single"),
    ("model_00_noise_1", FeatureSubset, ("feature_1", "feature_8"), "fit_single"),
    ("model_01a", IdentityFeatures, (), "run_param_search"),
    ("model_01b", OutcomePlusNoise, (), "run_param_search"),
    (
        "model_02_subset_a",
        FeatureSubset,
        ("feature_1", "feature_2", "feature_3"),
        "fit_single",
    ),
    (
        "model_02_subset_b",
        FeatureSubset,
        ("feature_4", "feature_5", "feature_6", "feature_7"),
        "fit_single",
    ),
)
"""The §7.2 table: id, policy type, subset columns, and trainer."""


def test_registry_has_exactly_the_eight_declared_strategies() -> None:
    """The registry holds exactly the eight plan entries, in run order."""
    assert tuple(MODEL_STRATEGIES) == tuple(entry[0] for entry in EXPECTED_STRATEGIES)
    assert len(MODEL_STRATEGIES) == 8


def test_registry_keys_equal_the_entries_own_ids() -> None:
    """A registry key is the entry's ``id``; no alias can shadow a strategy."""
    for key, strategy in MODEL_STRATEGIES.items():
        assert isinstance(strategy, ModelStrategy)
        assert strategy.id == key


def test_each_strategy_matches_the_declared_table() -> None:
    """Every entry's policy, subset columns, trainer, and objective match."""
    for (model_id, policy_type, columns, trainer), strategy in zip(
        EXPECTED_STRATEGIES,
        MODEL_STRATEGIES.values(),
        strict=True,
    ):
        assert strategy.id == model_id
        assert type(strategy.feature_policy) is policy_type
        if isinstance(strategy.feature_policy, FeatureSubset):
            assert strategy.feature_policy.columns == columns
        else:
            assert columns == ()
        assert strategy.trainer == trainer
        assert strategy.objective == REGRESSION_OBJECTIVE


def test_registry_order_is_the_documented_run_order() -> None:
    """Run order is the plan's order, not sorted order."""
    assert tuple(MODEL_STRATEGIES) == (
        "model_00_signal_0",
        "model_00_signal_1",
        "model_00_noise_0",
        "model_00_noise_1",
        "model_01a",
        "model_01b",
        "model_02_subset_a",
        "model_02_subset_b",
    )


def test_every_strategy_is_a_regression_with_a_known_trainer() -> None:
    """All v1 strategies are regression models and dispatch to a known kind."""
    for strategy in MODEL_STRATEGIES.values():
        assert strategy.objective == "reg:absoluteerror"
        assert strategy.trainer in TRAINER_KINDS


def test_trainer_kind_covers_exactly_the_literal_values() -> None:
    """``TRAINER_KINDS`` and the ``TrainerKind`` literal cannot drift apart."""
    assert set(TRAINER_KINDS) == set(get_args(TrainerKind))
    assert TRAINER_KINDS == ("fit_single", "run_param_search")


def test_registry_is_a_read_only_mapping() -> None:
    """The registry rejects mutation instead of silently changing the models."""
    assert isinstance(MODEL_STRATEGIES, MappingProxyType)
    mutable = cast(Any, MODEL_STRATEGIES)
    with pytest.raises(TypeError):
        mutable["model_03"] = MODEL_STRATEGIES["model_01a"]


def test_strategy_is_frozen_and_slotted() -> None:
    """A registered strategy cannot be edited or grow runtime attributes."""
    strategy = MODEL_STRATEGIES["model_01a"]
    with pytest.raises(FrozenInstanceError):
        setattr(strategy, "id", "model_01c")
    with pytest.raises((AttributeError, TypeError)):
        setattr(strategy, "unexpected", "value")
    assert not hasattr(strategy, "__dict__")


def test_each_policy_exposes_the_feature_policy_surface() -> None:
    """Every declared policy is an instance with both protocol methods."""
    for strategy in MODEL_STRATEGIES.values():
        policy = strategy.feature_policy
        assert not isinstance(policy, type)
        assert callable(policy.apply)
        assert callable(policy.feature_colnames)


def test_equivalent_strategies_compare_equal() -> None:
    """Frozen dataclass equality compares the policy value, not its identity."""
    left = ModelStrategy(
        id="model_00_signal_0",
        feature_policy=IdentityFeatures(),
        trainer="fit_single",
        objective=REGRESSION_OBJECTIVE,
    )
    assert left == MODEL_STRATEGIES["model_00_signal_0"]
    assert hash(left) == hash(MODEL_STRATEGIES["model_00_signal_0"])


def test_empty_strategy_id_is_rejected() -> None:
    """An empty id would produce an empty run directory, so it is rejected."""
    with pytest.raises(ValueError, match="id must be non-empty"):
        ModelStrategy(
            id="",
            feature_policy=IdentityFeatures(),
            trainer="fit_single",
            objective=REGRESSION_OBJECTIVE,
        )


def test_empty_objective_is_rejected() -> None:
    """The adapter is constructed from the objective, so it cannot be empty."""
    with pytest.raises(ValueError, match="non-empty objective"):
        ModelStrategy(
            id="model_00_signal_0",
            feature_policy=IdentityFeatures(),
            trainer="fit_single",
            objective="",
        )


def test_unknown_trainer_is_rejected() -> None:
    """A trainer name no driver dispatches on fails at construction."""
    with pytest.raises(ValueError, match="unknown trainer"):
        ModelStrategy(
            id="model_00_signal_0",
            feature_policy=IdentityFeatures(),
            trainer="refit",  # type: ignore[arg-type]
            objective=REGRESSION_OBJECTIVE,
        )
