"""Project-layer training-data preparation and the training driver.

This module is the model-data half of the training stage. It reorders each
split partition by its key columns, separates the model inputs from the
outcome, and converts them to the plain ``numpy`` arrays the estimator
mechanics consume. Nothing here touches the layout, artifacts, or manifests:
the artifact writer lives one module over, in :mod:`src.project.artifacts`,
and reads the prepared value through its structural contract.

The public surface is:

``TVTFeatureOutcomeData`` / ``divide_sort_train_valid_test_data``
    The three holdout partitions cut into aligned feature and outcome frames,
    in canonical train -> valid -> test order and sorted on the key columns.
``ModelReadyData`` / ``prepare_model_ready_data``
    The key frames, the model-input selections, and the ``numpy`` design
    matrices for one run. The full partition feature frames are kept as well,
    so the artifact writer can persist each partition's complete feature
    matrix even when the model trains on a feature subset.
``train_model_run``
    The driver: load one outcome's split tables, apply the strategy's feature
    policy, prepare the arrays, dispatch on the strategy's trainer kind
    through :mod:`src.core.training`, and close the run with ``save_results``.

Every setting comes from ``context.config``; this module never imports the
module-level ``CONFIG`` instance. The estimator-specific construction (the
fresh adapters and the booster normalization) lives in
:mod:`src.project.estimators`, so the driver's flow names no estimator class.

The ``numpy`` draw order is part of the value contract: a feature policy that
substitutes columns draws from the global RNG in the exact train -> valid ->
test order it declares, so a reordered draw is a declared recalibration rather
than a silent change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src.core.paths import PipelineContext
from src.core.training import fit_single, run_param_search
from src.project.artifacts import load_split_tables, save_results
from src.project.estimators import (
    booster_of,
    make_search_adapter_factory,
    make_single_fit_adapter,
)
from src.project.models import MODEL_STRATEGIES

PARAM_SEARCH_SCORING = "neg_mean_absolute_error"
"""The search scorer: minimizing absolute error matches the v1 objective."""

PARAM_SEARCH_N_JOBS = -1
"""Worker count for the randomized search.

Kept as a module constant rather than a config field: it is an execution
detail, not a data or model fact, so changing it must not change a recorded
run's artifacts.
"""


@dataclass(frozen=True, slots=True)
class TVTFeatureOutcomeData:
    """The three holdout partitions cut into feature and outcome frames.

    Every feature frame is the partition's key-plus-feature projection and
    every outcome frame is the partition's key-plus-outcome projection, both
    sorted on the shared key columns, so the corresponding rows of a feature
    and outcome frame describe the same case.

    Attributes:
        train_feature_df: The train partition's keys and model features.
        train_outcome_df: The train partition's keys and outcome column.
        valid_feature_df: The validation partition's keys and features.
        valid_outcome_df: The validation partition's keys and outcome column.
        test_feature_df: The test partition's keys and features.
        test_outcome_df: The test partition's keys and outcome column.
    """

    train_feature_df: pl.DataFrame
    train_outcome_df: pl.DataFrame
    valid_feature_df: pl.DataFrame
    valid_outcome_df: pl.DataFrame
    test_feature_df: pl.DataFrame
    test_outcome_df: pl.DataFrame


@dataclass(frozen=True, slots=True)
class ModelReadyData:
    """One run's partitioned data, ready for the estimator mechanics.

    The ``*_feature_df`` members are the partitions' complete key-plus-feature
    frames; the ``*_feature_select_df`` members are the model-input
    projections in estimator order (which may be a subset of the feature
    columns). Keeping both lets the artifact writer persist the full feature
    matrix while the estimator sees only the selected columns.

    Attributes:
        train_key_df: The train partition's key columns.
        train_feature_df: The train partition's full key-plus-feature frame.
        train_feature_select_df: The model's train input columns, in order.
        train_outcome_select_df: The train target column.
        x_train: The train design matrix.
        y_train: The train target matrix.
        valid_key_df: The validation partition's key columns.
        valid_feature_df: The validation partition's full feature frame.
        valid_feature_select_df: The model's validation input columns.
        valid_outcome_select_df: The validation target column.
        x_valid: The validation design matrix.
        y_valid: The validation target matrix.
        test_key_df: The test partition's key columns.
        test_feature_df: The test partition's full feature frame.
        test_feature_select_df: The model's test input columns.
        test_outcome_select_df: The test target column.
        x_test: The test design matrix.
        y_test: The test target matrix.
    """

    train_key_df: pl.DataFrame
    train_feature_df: pl.DataFrame
    train_feature_select_df: pl.DataFrame
    train_outcome_select_df: pl.DataFrame
    x_train: np.ndarray
    y_train: np.ndarray

    valid_key_df: pl.DataFrame
    valid_feature_df: pl.DataFrame
    valid_feature_select_df: pl.DataFrame
    valid_outcome_select_df: pl.DataFrame
    x_valid: np.ndarray
    y_valid: np.ndarray

    test_key_df: pl.DataFrame
    test_feature_df: pl.DataFrame
    test_feature_select_df: pl.DataFrame
    test_outcome_select_df: pl.DataFrame
    x_test: np.ndarray
    y_test: np.ndarray


def divide_sort_train_valid_test_data(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_head_df: pl.DataFrame,
    outcome_head_df: pl.DataFrame,
    key_colnames: list[str],
) -> TVTFeatureOutcomeData:
    """Cut each split partition into its feature and outcome frames.

    The feature and outcome head frames supply the column vocabulary: the
    feature head is the partition's key-plus-feature projection and the
    outcome head is its key-plus-outcome projection. Every frame is sorted on
    ``key_colnames`` in the canonical key order, so the three partitions and
    their feature/outcome halves stay row-aligned regardless of the on-disk
    row order the split stage wrote.

    Args:
        train_df: The train partition, carrying keys, features, and outcome.
        valid_df: The validation partition, with the same shape.
        test_df: The test partition, with the same shape.
        feature_head_df: The feature-table head that names the feature
            columns (keys first).
        outcome_head_df: The outcome-table head that names the outcome column
            (keys first).
        key_colnames: The canonical key columns, in sort order.

    Returns:
        The three partitions' feature and outcome frames.
    """

    def cut(partition_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        feature_df = partition_df.select(feature_head_df.columns).sort(key_colnames)
        outcome_df = partition_df.select(outcome_head_df.columns).sort(key_colnames)
        return feature_df, outcome_df

    train_feature_df, train_outcome_df = cut(train_df)
    valid_feature_df, valid_outcome_df = cut(valid_df)
    test_feature_df, test_outcome_df = cut(test_df)

    return TVTFeatureOutcomeData(
        train_feature_df=train_feature_df,
        train_outcome_df=train_outcome_df,
        valid_feature_df=valid_feature_df,
        valid_outcome_df=valid_outcome_df,
        test_feature_df=test_feature_df,
        test_outcome_df=test_outcome_df,
    )


def prepare_model_ready_data(
    train_feature_df: pl.DataFrame,
    train_outcome_df: pl.DataFrame,
    valid_feature_df: pl.DataFrame,
    valid_outcome_df: pl.DataFrame,
    test_feature_df: pl.DataFrame,
    test_outcome_df: pl.DataFrame,
    key_colnames: list[str],
    feature_colnames: list[str],
    outcome_colname: str,
) -> ModelReadyData:
    """Separate the keys from the model data and build the numpy matrices.

    The feature columns are selected in ``feature_colnames`` order — the
    strategy's estimator input order, which is not necessarily the feature
    head order — so the design matrix column order is exactly what the
    strategy declared. The full key-plus-feature frames are retained next to
    the selections for the artifact writer.

    Args:
        train_feature_df: The train partition's key-plus-feature frame.
        train_outcome_df: The train partition's key-plus-outcome frame.
        valid_feature_df: The validation partition's key-plus-feature frame.
        valid_outcome_df: The validation partition's key-plus-outcome frame.
        test_feature_df: The test partition's key-plus-feature frame.
        test_outcome_df: The test partition's key-plus-outcome frame.
        key_colnames: The canonical key columns.
        feature_colnames: The model input columns, in estimator order.
        outcome_colname: The modeled outcome column.

    Returns:
        The key frames, the full and selected feature frames, the outcome
        selections, and the design/target matrices for all three partitions.
    """

    def prepare(
        feature_df: pl.DataFrame,
        outcome_df: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray]:
        key_df = feature_df.select(key_colnames)
        feature_select_df = feature_df.select(feature_colnames)
        outcome_select_df = outcome_df.select(outcome_colname)
        return (
            key_df,
            feature_select_df,
            outcome_select_df,
            feature_select_df.to_numpy(),
            outcome_select_df.to_numpy(),
        )

    (
        train_key_df,
        train_feature_select_df,
        train_outcome_select_df,
        x_train,
        y_train,
    ) = prepare(train_feature_df, train_outcome_df)
    (
        valid_key_df,
        valid_feature_select_df,
        valid_outcome_select_df,
        x_valid,
        y_valid,
    ) = prepare(valid_feature_df, valid_outcome_df)
    (
        test_key_df,
        test_feature_select_df,
        test_outcome_select_df,
        x_test,
        y_test,
    ) = prepare(test_feature_df, test_outcome_df)

    return ModelReadyData(
        train_key_df=train_key_df,
        train_feature_df=train_feature_df,
        train_feature_select_df=train_feature_select_df,
        train_outcome_select_df=train_outcome_select_df,
        x_train=x_train,
        y_train=y_train,
        valid_key_df=valid_key_df,
        valid_feature_df=valid_feature_df,
        valid_feature_select_df=valid_feature_select_df,
        valid_outcome_select_df=valid_outcome_select_df,
        x_valid=x_valid,
        y_valid=y_valid,
        test_key_df=test_key_df,
        test_feature_df=test_feature_df,
        test_feature_select_df=test_feature_select_df,
        test_outcome_select_df=test_outcome_select_df,
        x_test=x_test,
        y_test=y_test,
    )


def train_model_run(
    strategy_id: str,
    outcome_name: str,
    context: PipelineContext,
    output_root: Path,
) -> None:
    """Train one model strategy end to end and persist its run.

    One function serves every model strategy. The flow loads the outcome's s03
    split tables, applies the strategy's feature policy (the policy's global
    ``numpy`` draws happen here, in the exact train -> valid -> test order it
    declares), prepares the model-ready arrays, dispatches on the strategy's
    trainer kind, records the feature and outcome names on the native booster,
    and closes the run with ``save_results`` — which writes the partition
    tables, feature importances, model serializations, and the model manifest.

    Everything the strategy needs that is not strategy identity data is read
    from ``context.config``: the single-fit parameter mapping, the search
    seeds, distributions, iteration and CV counts, and the base-estimator
    seed. The key columns come from the loaded split tables, and the run
    directory is resolved from the layout as
    ``s04_train/{outcome}/{strategy_id}``; the caller creates that directory
    through ``context.ensure`` before calling this driver.

    Args:
        strategy_id: The model strategy id — a ``MODEL_STRATEGIES`` key, the
            run-directory name, and the manifest ``model_name``.
        outcome_name: The modeled outcome, which is also the target column
            name.
        context: The frozen pipeline context (layout + config).
        output_root: The pipeline output root the relative layout paths are
            resolved under.

    Raises:
        ValueError: ``strategy_id`` is not a registered model strategy, or the
            strategy declares an unknown trainer kind.
    """
    try:
        strategy = MODEL_STRATEGIES[strategy_id]
    except KeyError:
        raise ValueError(
            f"unknown model strategy {strategy_id!r}; known strategies: "
            f"{', '.join(MODEL_STRATEGIES)}"
        ) from None

    output_dir = (
        output_root / context.layout.stage_dir("s04_train", outcome_name) / strategy_id
    )

    # LOAD: the s03 artifacts through the layout's logical names.
    split_tables = load_split_tables(context.layout, outcome_name, output_root)

    # SUBSTITUTE FEATURES: the policy's draws happen here, before any
    # estimator sees the data, in the policy's declared partition order.
    substituted = strategy.feature_policy.apply(split_tables)

    # The full feature head from the loaded split tables names every feature
    # column, while the partition frames carry the policy's substitutions. A
    # selection policy projects only the head, so cutting against the full
    # head keeps each partition's complete feature matrix available for the
    # artifact writer, which the s04 feature schema requires even when the
    # model trains on a subset.
    tvt_feature_outcome_data = divide_sort_train_valid_test_data(
        substituted.train_df,
        substituted.valid_df,
        substituted.test_df,
        split_tables.feature_head_df,
        split_tables.outcome_head_df,
        list(substituted.key_colnames),
    )

    # The estimator input order is strategy data, not the feature head order:
    # a substitution policy can swap or select columns.
    feature_colnames = strategy.feature_policy.feature_colnames(substituted)
    mrd = prepare_model_ready_data(
        tvt_feature_outcome_data.train_feature_df,
        tvt_feature_outcome_data.train_outcome_df,
        tvt_feature_outcome_data.valid_feature_df,
        tvt_feature_outcome_data.valid_outcome_df,
        tvt_feature_outcome_data.test_feature_df,
        tvt_feature_outcome_data.test_outcome_df,
        list(substituted.key_colnames),
        feature_colnames,
        outcome_name,
    )

    # TRAIN: estimator-agnostic mechanics; the estimator construction is
    # confined to src.project.estimators.
    if strategy.trainer == "fit_single":
        adapter = fit_single(
            make_single_fit_adapter(),
            mrd.x_train,
            mrd.y_train,
            mrd.x_valid,
            mrd.y_valid,
            # The objective is a strategy fact, so it is applied here rather
            # than left to a config default; the remaining settings are the
            # project's single-fit parameters.
            {**context.config.xgb_single_fit_params, "objective": strategy.objective},
            context.config.xgb_num_boost_round,
            context.config.xgb_early_stopping_rounds,
        )
        best_parameters = dict(context.config.xgb_single_fit_params)
    elif strategy.trainer == "run_param_search":
        adapter, best_parameters = run_param_search(
            make_search_adapter_factory(
                strategy.objective,
                context.config.xgb_regressor_seed,
            ),
            mrd.x_train,
            mrd.y_train,
            context.config.xgb_param_distributions,
            context.config.xgb_search_n_iter,
            context.config.xgb_search_cv,
            scoring=PARAM_SEARCH_SCORING,
            random_state=context.config.xgb_search_seed,
            n_jobs=PARAM_SEARCH_N_JOBS,
        )
    else:
        raise ValueError(
            f"strategy {strategy_id!r} declares unknown trainer kind "
            f"{strategy.trainer!r}; expected 'fit_single' or "
            "'run_param_search'"
        )

    best_model = booster_of(adapter)
    best_model.set_attr(feature_names=",".join(feature_colnames))
    best_model.set_attr(outcome_colname=outcome_name)

    # SAVE: partition tables, feature importances, model serializations, and
    # the run's model manifest. The tuned parameters are recorded in the
    # manifest's ``best_parameters`` field only.
    save_results(
        best_model,
        best_parameters,
        mrd,
        tvt_feature_outcome_data.train_feature_df,
        feature_colnames,
        output_dir,
        strategy_id=strategy_id,
        outcome_name=outcome_name,
        objective=strategy.objective,
    )
