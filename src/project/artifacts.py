"""Project-layer artifact I/O for the training stage.

This module is the artifact-writing half of the training driver. It reads one
outcome's divided tables through the layout's logical artifact names
(``layout.artifact('s03_divide', ...)``) and writes one trained run's
artifacts plus its ``manifest.json``. Reading through the layout keeps the
s03/s04 artifact contract in one declarative place, and reading the key-column
vocabulary from :mod:`src.project.columns` means a caller never passes key
column names per call.

The public surface is:

``SplitTables`` / ``load_split_tables``
    The divided tables of one outcome: the feature and outcome head frames plus
    the three holdout partitions. The split partitions are validated with
    ``S03_SPLIT_SCHEMA`` when they cross the read boundary; the head frames are
    column-shape hints only, so they are read without a data gate.
``extract_feature_importances`` / ``load_feature_importances``
    The model-to-table importance extraction and its manifest-driven reader.
    The reader resolves the declared artifact through the manifest inventory
    and validates it with ``S04_FEATURE_IMPORTANCES_SCHEMA`` at the consume
    boundary.
``save_xgboost_model``
    The native model serialization writer: a compact JSON model, a
    reader-friendly JSON copy, and a text dump.
``save_results``
    The model-manifest producer. It persists the run's partition tables,
    feature importances, and model serialization, then writes the frozen
    ``ModelManifest`` that declares every written file. Downstream stages read
    the run's identity, configuration, partitions, and artifact inventory from
    that manifest instead of parsing file or directory names, which is why the
    manifest is the single source of truth for the run shape.

The module is deliberately free of stage imports: ``save_results`` accepts a
structural model-ready value (see :class:`ModelReadyDataLike`) rather than
importing the training driver that constructs it, so artifact I/O and training
mechanics stay in separate layers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from src.core.io import save_dataframe_to_csv_and_parquet
from src.core.manifest import (
    MANIFEST_FILENAME,
    MODEL_SCHEMA_VERSION,
    ArtifactRef,
    ModelManifest,
    declared_artifact_path,
    write_manifest,
)
from src.core.paths import PipelineLayout
from src.core.splits import PARTITION_ORDER
from src.project.columns import (
    FEATURE_COLNAME,
    FEATURE_ESTIMATOR_ID_COLNAME,
    FEATURE_IMPORTANCE_COLNAME,
    KEY_COLUMNS,
)
from src.project.estimators import ESTIMATOR_ID
from src.project.outcomes import OUTCOMES
from src.project.schemas import (
    S03_SPLIT_SCHEMA,
    S04_FEATURE_IMPORTANCES_SCHEMA,
    S04_PARTITION_FEATURE_SCHEMA,
    S04_PARTITION_KEY_SCHEMA,
    validate,
)

STAGE_NAME = "s04_train"
"""The producing stage recorded in every model manifest."""

FEATURE_IMPORTANCES_LOGICAL_NAME = "feature_importances"
"""Logical name the run's importance table is declared under."""

FEATURE_IMPORTANCES_STEM = "feature_importances"
"""Filename stem of the importance table's parquet/CSV pair."""

MODEL_STEM = "model"
"""Filename stem of the run's model serializations."""

_MODEL_FILENAMES = (
    f"{MODEL_STEM}.json",
    f"{MODEL_STEM}_pretty.json",
    f"{MODEL_STEM}.txt",
)
"""The three xgboost serializations written per run."""

S03_SPLIT_READ_BOUNDARY = "s04 {partition} read"
"""Boundary label for an s03 split table read by the training stage."""

S04_PARTITION_KEY_BOUNDARY = "s04 {partition}_key_df"
"""Boundary label for a written partition key table."""

S04_PARTITION_FEATURE_BOUNDARY = "s04 {partition}_feature_df"
"""Boundary label for a written partition feature table."""

S04_FEATURE_IMPORTANCES_BOUNDARY = "s04 feature_importances read"
"""Boundary label for the consumed feature-importance table."""


class ModelReadyDataLike(Protocol):
    """The structural model-ready shape ``save_results`` persists.

    The concrete value is built by the training driver. Keeping the artifact
    layer on this structural contract avoids a circular import between the
    artifact writer and the training mechanics, while still documenting every
    frame the writer reads.

    ``*_feature_select_df`` holds the model's input features for one partition
    (without keys, in estimator order). A producer that keeps a full partition
    feature frame may expose it as ``<partition>_feature_df``; the writer uses
    that frame when present so a model that trains on a feature subset still
    records the partition's full feature matrix.
    """

    @property
    def train_key_df(self) -> pl.DataFrame: ...

    @property
    def train_feature_select_df(self) -> pl.DataFrame: ...

    @property
    def valid_key_df(self) -> pl.DataFrame: ...

    @property
    def valid_feature_select_df(self) -> pl.DataFrame: ...

    @property
    def test_key_df(self) -> pl.DataFrame: ...

    @property
    def test_feature_select_df(self) -> pl.DataFrame: ...


@dataclass(frozen=True)
class SplitTables:
    """One outcome's divided tables plus the shared key-column vocabulary.

    Attributes:
        feature_head_df: The feature table's head: the key columns followed by
            the feature columns in head order.
        outcome_head_df: The outcome table's head: the key columns followed by
            the modeled outcome column.
        train_df: The train partition, carrying keys, features, and the
            outcome.
        valid_df: The validation partition, with the same shape.
        test_df: The test partition, with the same shape.
        key_colnames: The canonical key columns, read from
            :mod:`src.project.columns` by :func:`load_split_tables`.
    """

    feature_head_df: pl.DataFrame
    outcome_head_df: pl.DataFrame
    train_df: pl.DataFrame
    valid_df: pl.DataFrame
    test_df: pl.DataFrame
    key_colnames: list[str]


def validate_s03_split_df(
    split_df: pl.DataFrame,
    partition: str,
) -> pl.DataFrame:
    """Validate one s03 split table read by :func:`load_split_tables`.

    The read-side schema guards the shape of the consumed split artifact, so a
    corrupted split fails here with the partition name instead of surfacing as
    a confusing alignment failure in the training driver.

    Args:
        split_df: The split partition read from disk.
        partition: The partition name, used in the boundary label.

    Returns:
        The validated frame, unchanged.

    Raises:
        BoundaryError: The frame violates ``S03_SPLIT_SCHEMA``.
    """
    return validate(
        split_df,
        S03_SPLIT_SCHEMA,
        S03_SPLIT_READ_BOUNDARY.format(partition=partition),
    )


def validate_partition_key_df(
    key_df: pl.DataFrame,
    partition: str,
) -> pl.DataFrame:
    """Validate one partition key table before it is written.

    Args:
        key_df: The partition's key frame.
        partition: The partition name, used in the boundary label.

    Returns:
        The validated frame, unchanged.

    Raises:
        BoundaryError: The frame violates ``S04_PARTITION_KEY_SCHEMA``.
    """
    return validate(
        key_df,
        S04_PARTITION_KEY_SCHEMA,
        S04_PARTITION_KEY_BOUNDARY.format(partition=partition),
    )


def validate_partition_feature_df(
    feature_df: pl.DataFrame,
    partition: str,
) -> pl.DataFrame:
    """Validate one partition feature table before it is written.

    The schema checks the full partition feature matrix: keys plus the
    feature columns, all non-null and finite.

    Args:
        feature_df: The partition's key-plus-feature frame.
        partition: The partition name, used in the boundary label.

    Returns:
        The validated frame, unchanged.

    Raises:
        BoundaryError: The frame violates ``S04_PARTITION_FEATURE_SCHEMA``.
    """
    return validate(
        feature_df,
        S04_PARTITION_FEATURE_SCHEMA,
        S04_PARTITION_FEATURE_BOUNDARY.format(partition=partition),
    )


def load_split_tables(
    layout: PipelineLayout,
    outcome_name: str,
    output_root: Path,
) -> SplitTables:
    """Load one outcome's divided tables through the layout's artifact names.

    The s03 filenames are declared once in the stage table, so this loader
    carries no hardcoded filename. The key-column vocabulary is the shared
    project constant ``KEY_COLUMNS``, never a caller argument.

    Args:
        layout: The pipeline layout declaring the ``s03_divide`` artifacts.
        outcome_name: The outcome whose split set is loaded.
        output_root: The output root the layout's relative paths resolve
            under.

    Returns:
        The outcome's split tables, with the split partitions validated.

    Raises:
        KeyError: The layout declares no such stage or artifact.
        ValueError: The outcome is not declared by the layout.
        BoundaryError: A split partition violates ``S03_SPLIT_SCHEMA``.
    """

    def s03_artifact(logical: str) -> Path:
        return output_root / layout.artifact("s03_divide", logical, outcome_name)

    return SplitTables(
        feature_head_df=pl.read_parquet(s03_artifact("feature_head_df")),
        outcome_head_df=pl.read_parquet(s03_artifact("outcome_head_df")),
        train_df=validate_s03_split_df(
            pl.read_parquet(s03_artifact("train_df")), "train_df"
        ),
        valid_df=validate_s03_split_df(
            pl.read_parquet(s03_artifact("valid_df")), "valid_df"
        ),
        test_df=validate_s03_split_df(
            pl.read_parquet(s03_artifact("test_df")), "test_df"
        ),
        key_colnames=list(KEY_COLUMNS),
    )


def extract_feature_importances(
    feature_df: pl.DataFrame,
    model: Any,
) -> pl.DataFrame:
    """Map a model's gain scores back to the training feature names.

    xgboost assigns features positional names (``f0``, ``f1``, ...) in the
    order the training frame supplied them, so the mapping is positional. A
    model that grew no splits reports no gain scores; that case falls back to a
    small positive value per declared feature so the importance contract holds
    and downstream ranking has one row per input feature.

    Args:
        feature_df: The training features in estimator input order.
        model: The fitted booster whose gain scores are extracted.

    Returns:
        A frame with the feature name, the estimator's feature identifier, and
        the positive importance value, in the model's reported order.

    Raises:
        ValueError: ``feature_df`` has no columns, the model reports a feature
            identifier that is not in ``feature_df``, or a reported importance
            is not greater than ``1e-6``.
    """
    if feature_df.width == 0:
        raise ValueError("feature_df must have at least one column")

    feature_id_to_name = {
        f"f{index}": name for index, name in enumerate(feature_df.columns)
    }
    scores = model.get_score(importance_type="gain")
    if not scores:
        scores = {f"f{index}": 0.00001 for index in range(feature_df.width)}

    unknown_ids = sorted(set(scores) - set(feature_id_to_name))
    if unknown_ids:
        raise ValueError(
            "the model reported feature id(s) that are not among the declared "
            f"feature columns: {unknown_ids}; pass the training columns in "
            "training order"
        )

    importances = pl.DataFrame(
        {
            FEATURE_COLNAME: [feature_id_to_name[feature_id] for feature_id in scores],
            FEATURE_ESTIMATOR_ID_COLNAME: list(scores),
            FEATURE_IMPORTANCE_COLNAME: [
                _importance_value(value) for value in scores.values()
            ],
        }
    )
    if not bool((importances[FEATURE_IMPORTANCE_COLNAME] > 1e-6).all()):
        raise ValueError(
            "feature importances must all be greater than 1e-6; got "
            f"{importances[FEATURE_IMPORTANCE_COLNAME].to_list()!r}"
        )
    return importances


def load_feature_importances(
    manifest: ModelManifest,
    run_dir: Path,
) -> pl.DataFrame:
    """Read one model run's declared ``feature_importances`` parquet.

    The artifact is resolved through the manifest's declared inventory, never
    by scanning ``run_dir``: exactly one ``feature_importances`` parquet must
    be declared, even though the producer also writes a CSV under the same
    logical name. The read is validated at the consume boundary so a malformed
    table fails here instead of surfacing later in the plotting layer.

    Args:
        manifest: The run's model manifest.
        run_dir: The run directory holding the manifest and its colocated
            artifacts.

    Returns:
        The validated feature-importance table, unchanged.

    Raises:
        ValueError: The manifest declares no matching parquet, or more than
            one.
        FileNotFoundError: The declared parquet is absent from ``run_dir``.
        BoundaryError: The table violates ``S04_FEATURE_IMPORTANCES_SCHEMA``.
    """
    path = declared_artifact_path(
        manifest,
        run_dir,
        FEATURE_IMPORTANCES_LOGICAL_NAME,
        ".parquet",
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"model run {manifest.model_name!r} is missing its declared "
            f"{FEATURE_IMPORTANCES_LOGICAL_NAME!r} artifact {path}"
        )
    return validate(
        pl.read_parquet(path),
        S04_FEATURE_IMPORTANCES_SCHEMA,
        S04_FEATURE_IMPORTANCES_BOUNDARY,
    )


def _importance_value(value: float | list[float]) -> float:
    """Return a scalar importance from a possibly multi-output score."""
    if isinstance(value, list):
        return float(value[0])
    return float(value)


def _booster_of(model: Any) -> Any:
    """Return the native booster wrapped by either supported model shape."""
    get_booster = getattr(model, "get_booster", None)
    if get_booster is not None:
        return get_booster()
    return model


def save_xgboost_model(
    model: Any,
    output_filename_stem: str,
    output_path: Path,
) -> None:
    """Write the native model in JSON, pretty JSON, and text form.

    The compact JSON file is the canonical serialization; the pretty JSON copy
    is for a human reading the tree, and the text dump is the library's
    tree-level view. The directory is not created here: the caller creates it
    first, so a missing directory raises instead of being silently created.

    Args:
        model: The fitted booster, or a regressor wrapping one.
        output_filename_stem: Filename stem shared by the three artifacts.
        output_path: Existing directory that receives the files.
    """
    booster = _booster_of(model)

    json_path = output_path / f"{output_filename_stem}.json"
    booster.save_model(json_path)

    pretty_path = output_path / f"{output_filename_stem}_pretty.json"
    with open(json_path, encoding="utf-8") as source:
        model_json = json.load(source)
    with open(pretty_path, "w", encoding="utf-8") as target:
        json.dump(model_json, target, indent=4, sort_keys=True)

    text_path = output_path / f"{output_filename_stem}.txt"
    booster.dump_model(text_path)


def _task_from_objective(objective: str) -> str:
    """Map a native objective string to the canonical manifest task."""
    if objective.startswith("reg:"):
        return "regression"
    if objective.startswith("binary:"):
        return "binary"
    raise ValueError(
        f"unknown objective {objective!r}: expected a 'reg:' or 'binary:' "
        "objective string"
    )


def _partition_feature_frame(
    mrd: ModelReadyDataLike,
    partition: str,
) -> pl.DataFrame:
    """Return one partition's key-plus-feature table for persistence.

    A producer may already carry the full feature frame for the partition; when
    it does, that frame is used directly. Otherwise the model's selected
    feature frame is combined with the partition keys. Either way the result
    carries the keys exactly once, so the written artifact satisfies
    ``S04_PARTITION_FEATURE_SCHEMA``.
    """
    key_df = getattr(mrd, f"{partition}_key_df")
    full_feature_df = getattr(mrd, f"{partition}_feature_df", None)
    feature_df = (
        full_feature_df
        if full_feature_df is not None
        else getattr(mrd, f"{partition}_feature_select_df")
    )
    if all(column in feature_df.columns for column in key_df.columns):
        return feature_df
    return pl.concat([key_df, feature_df], how="horizontal")


def _data_artifact_refs() -> list[ArtifactRef]:
    """The split-table artifact inventory, in deterministic partition order."""
    refs: list[ArtifactRef] = []
    for partition in PARTITION_ORDER:
        for stem in (f"{partition}_key_df", f"{partition}_feature_df"):
            refs.append(ArtifactRef(logical_name=stem, filename=f"{stem}.parquet"))
            refs.append(ArtifactRef(logical_name=stem, filename=f"{stem}.csv"))
    return refs


def save_results(
    best_model: Any,
    best_parameters: Mapping[str, Any],
    mrd: ModelReadyDataLike,
    train_feature_df: pl.DataFrame,
    feature_colnames: Sequence[str],
    output_path: Path,
    strategy_id: str,
    outcome_name: str,
    objective: str,
) -> None:
    """Persist one trained run's artifacts and write its model manifest.

    The run's per-partition key and feature tables are written in
    ``PARTITION_ORDER``, followed by the feature-importance table and the model
    serializations, and the run is closed with ``manifest.json``. The manifest
    records model identity (``model_name`` is the strategy id), the producing
    stage and outcome, the estimator/task/objective, the outcome and feature
    column names, the declared partitions, the outcome's registered
    model-space transform id, the tuned parameters, and an ``ArtifactRef``
    inventory of every file just written. The tuned parameters live only in
    the manifest's ``best_parameters`` field; no standalone parameters file is
    written.

    Args:
        best_model: The fitted model. Either a native booster or a regressor
            wrapping one; both are normalized before serialization.
        best_parameters: The tuned hyperparameters to record in the manifest.
        mrd: The prepared model-ready data holding each partition's key and
            feature frames.
        train_feature_df: The training features in estimator input order, used
            to map gain scores back to feature names.
        feature_colnames: The model's feature columns, in estimator order.
        output_path: The existing run directory that receives the artifacts.
        strategy_id: The model identity recorded as the manifest's
            ``model_name``.
        outcome_name: The modeled outcome; also the target column name.
        objective: The native objective; the manifest task is derived from it.

    Raises:
        ValueError: ``objective`` is not a ``reg:`` or ``binary:`` objective.
        BoundaryError: A partition frame violates its S04 schema before
            anything is written, or the extracted importances violate their
            schema.
    """
    booster = _booster_of(best_model)
    partitions = PARTITION_ORDER

    for partition in partitions:
        key_df = validate_partition_key_df(
            getattr(mrd, f"{partition}_key_df"),
            partition,
        )
        feature_df = validate_partition_feature_df(
            _partition_feature_frame(mrd, partition),
            partition,
        )
        save_dataframe_to_csv_and_parquet(key_df, f"{partition}_key_df", output_path)
        save_dataframe_to_csv_and_parquet(
            feature_df, f"{partition}_feature_df", output_path
        )

    artifact_refs = _data_artifact_refs()

    importance_df = extract_feature_importances(
        train_feature_df.select(list(feature_colnames)),
        booster,
    )
    importance_df = validate(
        importance_df,
        S04_FEATURE_IMPORTANCES_SCHEMA,
        S04_FEATURE_IMPORTANCES_BOUNDARY,
    )
    save_dataframe_to_csv_and_parquet(
        importance_df,
        FEATURE_IMPORTANCES_STEM,
        output_path,
    )
    artifact_refs.append(
        ArtifactRef(
            logical_name=FEATURE_IMPORTANCES_LOGICAL_NAME,
            filename=f"{FEATURE_IMPORTANCES_STEM}.parquet",
        )
    )
    artifact_refs.append(
        ArtifactRef(
            logical_name=FEATURE_IMPORTANCES_LOGICAL_NAME,
            filename=f"{FEATURE_IMPORTANCES_STEM}.csv",
        )
    )

    save_xgboost_model(booster, MODEL_STEM, output_path)
    for filename in _MODEL_FILENAMES:
        artifact_refs.append(ArtifactRef(logical_name=MODEL_STEM, filename=filename))

    manifest = ModelManifest(
        schema_version=MODEL_SCHEMA_VERSION,
        manifest_kind="model",
        model_name=strategy_id,
        stage=STAGE_NAME,
        outcome=outcome_name,
        estimator=ESTIMATOR_ID,
        task=_task_from_objective(objective),
        objective=objective,
        outcome_colname=outcome_name,
        feature_colnames=tuple(feature_colnames),
        partitions=partitions,
        target_transform=OUTCOMES[outcome_name].outcome_transform,
        best_parameters=best_parameters,
        artifacts=tuple(artifact_refs),
    )
    write_manifest(manifest, output_path / MANIFEST_FILENAME)
