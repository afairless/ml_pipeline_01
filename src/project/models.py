"""Project model-strategy registry.

A model strategy is the declarative identity of one trained model: which
feature policy the training driver applies, which training mechanic it
dispatches to, and which estimator objective the adapter is constructed with.
Everything else about a run — hyperparameters, seeds, boosting rounds — stays
in :data:`src.project.config.CONFIG`, so two strategies can differ in what
they train on and how they search without duplicating tuning settings.

:data:`MODEL_STRATEGIES` maps each strategy id to one frozen
:class:`ModelStrategy`. The mapping is read-only and preserves insertion
order, so iterating it reproduces the pipeline's model run order and a
mutation attempt fails instead of silently changing the vocabulary the stage
paths and manifests resolve against. The id is also the ``{model}``
placeholder value in every per-model stage path, and the mapping key matches
the entry's own ``id`` so the registry cannot drift from the value it holds.

The eight entries are, in run order:

``model_00_signal_0`` / ``model_00_signal_1``
    The full feature set verbatim and its outcome-plus-noise counterpart.
``model_00_noise_0`` / ``model_00_noise_1``
    The pure-noise negative control and a fixed two-feature subset.
``model_01a`` / ``model_01b``
    The same full and outcome-plus-noise feature policies, trained through
    the randomized hyperparameter search instead of one holdout fit.
``model_02_subset_a`` / ``model_02_subset_b``
    Two disjoint fixed feature subsets.

Every strategy is a regression strategy and carries the ``reg:absoluteerror``
objective. Binary metrics are produced later by thresholding a regression
model's actual and predicted values at evaluation time, never by training a
classification strategy; no binary objective is registered. A strategy names
a trainer, not a callable, so importing this registry does not import an
estimator library or a training function.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from src.project.features import (
    FeaturePolicy,
    FeatureSubset,
    IdentityFeatures,
    OutcomePlusNoise,
    PureNoise,
)

TrainerKind = Literal["fit_single", "run_param_search"]
"""The training mechanics a strategy dispatches to.

``fit_single`` fits one model on one holdout split; ``run_param_search`` runs
a randomized hyperparameter search on the train partition. The driver maps
this literal to :func:`src.core.training.fit_single` or
:func:`src.core.training.run_param_search`; the registry itself never holds
the callable, so importing it stays estimator-free.
"""

TRAINER_KINDS: tuple[TrainerKind, ...] = ("fit_single", "run_param_search")
"""Every valid :data:`TrainerKind`, in dispatch order."""

REGRESSION_OBJECTIVE = "reg:absoluteerror"
"""The retained default objective: every v1 strategy is a regression model."""


@dataclass(frozen=True, slots=True)
class ModelStrategy:
    """One model's feature policy, training mechanic, and objective.

    Attributes:
        id: The model identity. It is also the registry key, the ``{model}``
            placeholder value in per-model stage paths, and the model-run
            directory name.
        feature_policy: The policy that decides which feature columns the
            estimator sees, in estimator input order.
        trainer: The training mechanic the driver dispatches to. A literal
            name, not a callable, so the registry cannot pull an estimator or
            a training implementation into import time.
        objective: The estimator objective the adapter is constructed with.
            Hyperparameters and seeds stay in the project config.

    Raises:
        ValueError: ``id`` or ``objective`` is empty, or ``trainer`` is not a
            registered :data:`TrainerKind`. Validating on construction keeps a
            typo from registering a strategy no driver can execute.
    """

    id: str
    feature_policy: FeaturePolicy
    trainer: TrainerKind
    objective: str

    def __post_init__(self) -> None:
        """Validate the frozen fields on construction."""
        if not self.id:
            raise ValueError(
                "model strategy id must be non-empty; ModelStrategy.id is the "
                "registry key, the {model} placeholder, and the run directory "
                "name"
            )
        if not self.objective:
            raise ValueError(
                f"model strategy {self.id!r} needs a non-empty objective; the "
                "adapter is constructed from it"
            )
        if self.trainer not in TRAINER_KINDS:
            raise ValueError(
                f"model strategy {self.id!r} declares unknown trainer "
                f"{self.trainer!r}; registered trainers are {TRAINER_KINDS}"
            )


# The registry: mapping key == ModelStrategy.id, insertion order == model run
# order. MappingProxyType makes the registry read-only, so a consumer that
# tries to mutate it gets a TypeError instead of silently changing the models
# the stage table and layout resolve against. Every policy instance is created
# once here, so the table is the single declaration of each strategy's feature
# behaviour.
MODEL_STRATEGIES: Mapping[str, ModelStrategy] = MappingProxyType(
    {
        "model_00_signal_0": ModelStrategy(
            id="model_00_signal_0",
            feature_policy=IdentityFeatures(),
            trainer="fit_single",
            objective=REGRESSION_OBJECTIVE,
        ),
        "model_00_signal_1": ModelStrategy(
            id="model_00_signal_1",
            feature_policy=OutcomePlusNoise(),
            trainer="fit_single",
            objective=REGRESSION_OBJECTIVE,
        ),
        "model_00_noise_0": ModelStrategy(
            id="model_00_noise_0",
            feature_policy=PureNoise(),
            trainer="fit_single",
            objective=REGRESSION_OBJECTIVE,
        ),
        "model_00_noise_1": ModelStrategy(
            id="model_00_noise_1",
            feature_policy=FeatureSubset(columns=("feature_1", "feature_8")),
            trainer="fit_single",
            objective=REGRESSION_OBJECTIVE,
        ),
        "model_01a": ModelStrategy(
            id="model_01a",
            feature_policy=IdentityFeatures(),
            trainer="run_param_search",
            objective=REGRESSION_OBJECTIVE,
        ),
        "model_01b": ModelStrategy(
            id="model_01b",
            feature_policy=OutcomePlusNoise(),
            trainer="run_param_search",
            objective=REGRESSION_OBJECTIVE,
        ),
        "model_02_subset_a": ModelStrategy(
            id="model_02_subset_a",
            feature_policy=FeatureSubset(
                columns=("feature_1", "feature_2", "feature_3")
            ),
            trainer="fit_single",
            objective=REGRESSION_OBJECTIVE,
        ),
        "model_02_subset_b": ModelStrategy(
            id="model_02_subset_b",
            feature_policy=FeatureSubset(
                columns=("feature_4", "feature_5", "feature_6", "feature_7")
            ),
            trainer="fit_single",
            objective=REGRESSION_OBJECTIVE,
        ),
    }
)
