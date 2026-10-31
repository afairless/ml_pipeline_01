"""Manifest-driven model loading and the prediction-manifest producer.

``load_model_run`` reads one trained run from its ``manifest.json``: model
identity (``model_name``), ``task``/``objective``, the declared
``partitions``, and the artifact inventory all come from the frozen
``ModelManifest`` the s04 producer writes next to the run's artifacts under
``s04_train/{outcome}/{model}/``. Every split table and the model is resolved
through that inventory, so a declared artifact missing from disk raises
instead of loading a partial run.

Reconstruction goes through the estimator registry: ``load_model_run``
deserializes the declared ``model.json`` with
``estimator_registry.load(manifest.estimator, path)``, and
``generate_and_save_predictions`` dispatches on the manifest ``task`` —
validated against the adapter's own report — through the adapter's
``predict`` / ``predict_proba`` surface. The manifest's derived ``task``,
never the objective string, decides the regression/binary branch.

``save_prediction_run`` is the prediction-manifest producer, mirroring the
s04 producer (``src.project.artifacts.save_results``): for every partition
the model manifest declares it writes the prediction artifact pair
``<model>_<partition>_predictions`` into the per-model run directory
``s05_predict/{outcome}/{model_name}/`` and assembles the run's
``PredictionManifest`` (kind ``'predictions'``, stage ``'s05_predict'``,
``partitions`` = the model run's declared partitions, one partition-tagged
``ArtifactRef`` per written file). The partition is therefore recorded both in
the artifact name and in the manifest, so no consumer infers it from a
filename. The s05 stage main resolves the run dir from the layout, ``ensure``s
it, and closes the run with ``write_manifest(run_dir/'manifest.json')``; s06
discovers the runs structurally through ``load_prediction_runs``.

The prediction artifacts are schema-guarded at the write boundary: every
prediction frame is validated against ``S05_PREDICTION_SCHEMA`` before
``save_dataframe_to_csv_and_parquet`` writes its parquet/csv pair, so a
malformed key or prediction column fails with the partition-named boundary
label and leaves no artifact behind.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src.core.estimator import EstimatorAdapter, estimator_registry
from src.core.io import save_dataframe_to_csv_and_parquet
from src.core.manifest import (
    PREDICTION_SCHEMA_VERSION,
    ArtifactRef,
    ModelManifest,
    PredictionManifest,
    declared_artifact_path,
)
from src.project.schemas import S05_PREDICTION_SCHEMA, validate

MODEL_JSON_FILENAME = "model.json"
"""The canonical model serialization the manifest declares under ``model``.

A run's manifest declares ``model.json``, ``model_pretty.json``, and
``model.txt`` under the same logical name, but only ``model.json`` is
loadable by the adapter, so the reader resolves exactly that filename.
"""

PREDICTION_DF_BOUNDARY = "s05 {partition}_predictions"
"""Boundary label for one partition's written prediction table.

The partition name is interpolated because every partition artifact is a
separate boundary."""

STAGE_NAME = "s05_predict"
"""The producing stage recorded in every prediction manifest."""

_ARTIFACT_SUFFIXES = (".parquet", ".csv")
"""The file suffixes ``save_dataframe_to_csv_and_parquet`` writes per
partition (parquet first, then the head-only CSV)."""

__all__ = [
    "LoadedModelRun",
    "MODEL_JSON_FILENAME",
    "PREDICTION_DF_BOUNDARY",
    "STAGE_NAME",
    "generate_and_save_predictions",
    "load_model_run",
    "save_prediction_run",
    "validate_prediction_df",
]


def validate_prediction_df(
    prediction_df: pl.DataFrame,
    partition: str,
) -> pl.DataFrame:
    """Validate one prediction table before its artifacts are written.

    The schema guards the partition keys and the ``predictions`` column shape,
    so a malformed frame fails at the produce boundary (naming the partition)
    instead of leaving a corrupt parquet/csv pair for s06 to evaluate.

    Args:
        prediction_df: The prediction table about to be saved.
        partition: The partition name, used in the boundary label.

    Returns:
        The validated frame, unchanged.

    Raises:
        BoundaryError: The frame fails ``S05_PREDICTION_SCHEMA``; the error
            carries :data:`PREDICTION_DF_BOUNDARY` for ``partition``.
    """
    return validate(
        prediction_df,
        S05_PREDICTION_SCHEMA,
        PREDICTION_DF_BOUNDARY.format(partition=partition),
    )


@dataclass(frozen=True, slots=True)
class LoadedModelRun:
    """One trained model run loaded from its manifest.

    Attributes:
        manifest: The run's ``ModelManifest`` — the single source for model
            identity (``model_name``), ``task``/``objective``, the declared
            ``partitions`` (the split shape of the run), and the artifact
            inventory.
        run_dir: The run directory holding the manifest and its colocated
            artifacts (``stage_dir / manifest.model_name``).
        key_data: The split key tables of the run, keyed by partition name —
            one entry per ``manifest.partitions`` entry, read from the
            manifest's declared parquet artifacts.
        feature_data: The model-input tables, keyed by partition name in the
            same shape as ``key_data``. Each table is the declared partition
            feature frame projected to the manifest's ``feature_colnames``, so
            a model that trained on a feature subset receives exactly its
            declared input columns in estimator order.
        loaded_model: The trained model reconstructed from the declared
            ``model.json`` through the estimator registry
            (``manifest.estimator``), wrapped as an ``EstimatorAdapter``.
    """

    manifest: ModelManifest
    run_dir: Path
    key_data: dict[str, pl.DataFrame]
    feature_data: dict[str, pl.DataFrame]
    loaded_model: EstimatorAdapter


def _declared_model_path(manifest: ModelManifest, run_dir: Path) -> Path:
    """Return the run-dir path of the declared loadable model serialization.

    The manifest declares three filenames under the ``model`` logical name;
    only the plain JSON serialization is loadable, so exactly that one is
    required.

    Raises:
        ValueError: The manifest declares no matching serialization, or more
            than one.
    """
    filenames = [
        ref.filename
        for ref in manifest.artifacts
        if ref.logical_name == "model" and ref.filename == MODEL_JSON_FILENAME
    ]
    if len(filenames) != 1:
        raise ValueError(
            f"model run {manifest.model_name!r} must declare exactly one "
            f"{MODEL_JSON_FILENAME!r} serialization, got {sorted(filenames)}"
        )
    return run_dir / filenames[0]


def load_model_run(manifest: ModelManifest, stage_dir: Path) -> LoadedModelRun:
    """Load one trained model run from its manifest.

    ``manifest`` is one entry of ``src.core.manifest.load_model_runs``, which
    already validated the run's manifest (kind, outcome, per-kind schema
    version); ``stage_dir`` is the outcome-scoped s04 stage directory the
    manifest was discovered under. The split tables and the model are
    resolved through the manifest's declared artifact inventory — never by
    scanning filenames — and identity/``task``/``objective``/``partitions``
    come from the manifest, never from parsing a directory name. Per-partition
    tables are read for exactly the ``manifest.partitions`` the run declares,
    never a fixed partition list.

    Args:
        manifest: The run's model manifest.
        stage_dir: The outcome-scoped s04 stage directory containing the run
            directory named by ``manifest.model_name``.

    Returns:
        The run's manifest, directory, partition key and model-input tables,
        and the reconstructed estimator adapter.

    Raises:
        FileNotFoundError: The manifest's run directory or a declared artifact
            file does not exist.
        ValueError: The manifest declares no (or several) artifacts for a
            logical name the stage needs, records an estimator id unknown to
            the registry, or declares a ``task`` the loaded model does not
            report.
    """
    run_dir = Path(stage_dir) / manifest.model_name
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"model run {manifest.model_name!r} declared by its manifest in "
            f"{stage_dir} has no run directory {run_dir}"
        )

    def read_table(logical_name: str) -> pl.DataFrame:
        path = declared_artifact_path(manifest, run_dir, logical_name)
        if not path.is_file():
            raise FileNotFoundError(
                f"model run {manifest.model_name!r} is missing its declared "
                f"{logical_name!r} artifact {path}"
            )
        return pl.read_parquet(path)

    model_path = _declared_model_path(manifest, run_dir)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"model run {manifest.model_name!r} is missing its declared "
            f"model serialization {model_path}"
        )

    loaded_model = estimator_registry.load(manifest.estimator, model_path)
    if loaded_model.task != manifest.task:
        raise ValueError(
            f"model run {manifest.model_name!r} task mismatch: manifest "
            f"records {manifest.task!r} but the loaded "
            f"{manifest.estimator!r} model reports {loaded_model.task!r}"
        )

    feature_colnames = list(manifest.feature_colnames)
    return LoadedModelRun(
        manifest=manifest,
        run_dir=run_dir,
        key_data={
            partition: read_table(f"{partition}_key_df")
            for partition in manifest.partitions
        },
        feature_data={
            partition: read_table(f"{partition}_feature_df").select(feature_colnames)
            for partition in manifest.partitions
        },
        loaded_model=loaded_model,
    )


def generate_and_save_predictions(
    loaded_model: EstimatorAdapter,
    task: str,
    x_array: np.ndarray,
    key_df: pl.DataFrame,
    model_filename: str,
    data_type_name: str,
    output_path: Path,
) -> None:
    """Generate one partition's predictions and save the artifact pair.

    The regression/binary branch is driven by the manifest's ``task`` — the
    single source of truth — which is validated against the adapter's own
    ``task``: a discrepancy fails before any prediction is written. The
    prediction call itself goes through the adapter's ``predict`` /
    ``predict_proba`` surface instead of type checks on concrete estimator
    wrappers.

    Args:
        loaded_model: The trained estimator adapter.
        task: The task recorded in the model manifest (``'regression'`` or
            ``'binary'``); it must equal the adapter's own task.
        x_array: The partition's model-input matrix.
        key_df: The partition's key frame, one row per prediction.
        model_filename: The model identity used in the artifact stem.
        data_type_name: The partition name used in the artifact stem and the
            boundary label.
        output_path: The existing run directory that receives the artifacts.

    Raises:
        ValueError: ``task`` does not match the adapter's task, or is neither
            ``'regression'`` nor ``'binary'``.
        BoundaryError: The assembled prediction frame violates
            ``S05_PREDICTION_SCHEMA``.
    """
    if task != loaded_model.task:
        raise ValueError(
            f"task {task!r} (from the model manifest) does not match the "
            f"loaded model task {loaded_model.task!r}"
        )

    if task == "regression":
        predictions = loaded_model.predict(x_array)
    elif task == "binary":
        predictions = loaded_model.predict_proba(x_array)[:, 1]
    else:
        raise ValueError(
            f"Unknown task {task!r}. Task must be 'regression' or 'binary'."
        )

    prediction_df = pl.concat(
        [
            key_df,
            pl.DataFrame({"predictions": predictions.astype(np.float64)}),
        ],
        how="horizontal",
    )
    prediction_df = validate_prediction_df(prediction_df, data_type_name)
    output_stem = f"{model_filename}_{data_type_name}_predictions"
    save_dataframe_to_csv_and_parquet(prediction_df, output_stem, output_path)


def save_prediction_run(
    loaded_model: EstimatorAdapter,
    task: str,
    partitions: tuple[str, ...],
    key_dfs: list[pl.DataFrame],
    feature_dfs: list[pl.DataFrame],
    model_name: str,
    outcome: str,
    run_dir: Path,
) -> PredictionManifest:
    """Write one model run's prediction artifacts and assemble its manifest.

    One partition prediction file pair is written per ``partitions`` entry
    with the ``<model>_<partition>_predictions`` names into the per-model run
    directory, and the returned manifest records one partition-tagged
    ``ArtifactRef`` per written file. The partition is thus self-describing —
    carried by the artifact name and by the manifest — so s06 resolves the
    run's tables from the manifest instead of globbing the stage directory.

    The caller resolves and ``ensure``s ``run_dir``, then closes the run with
    ``write_manifest(manifest, run_dir / MANIFEST_FILENAME)``; this function
    never creates the run directory itself.

    Args:
        loaded_model: The trained estimator adapter generating predictions.
        task: The model manifest's task; it must equal the adapter's task.
        partitions: The model run's declared partition names, in prediction
            order.
        key_dfs: The split key tables, one per partition, in the same order.
        feature_dfs: The model-input tables, one per partition, in the same
            order.
        model_name: The model identity (the strategy id), recorded as the
            manifest's ``model_name`` and used in the artifact stems.
        outcome: The modeled outcome.
        run_dir: The run directory the partition artifacts are written into.

    Returns:
        The assembled prediction manifest, not yet written to disk.

    Raises:
        AssertionError: The number of key tables, feature tables, and
            partitions differ.
    """
    assert len(key_dfs) == len(feature_dfs) == len(partitions), (
        "one key and feature table per partition is required, got "
        f"{len(key_dfs)} key and {len(feature_dfs)} feature tables for "
        f"{len(partitions)} partitions"
    )

    artifact_refs: list[ArtifactRef] = []
    for partition, key_df, feature_df in zip(
        partitions, key_dfs, feature_dfs, strict=True
    ):
        generate_and_save_predictions(
            loaded_model,
            task,
            feature_df.to_numpy(),
            key_df,
            model_name,
            partition,
            run_dir,
        )
        for suffix in _ARTIFACT_SUFFIXES:
            filename = f"{model_name}_{partition}_predictions{suffix}"
            artifact_refs.append(
                ArtifactRef(
                    logical_name=f"{partition}_predictions",
                    filename=filename,
                    partition=partition,
                )
            )

    return PredictionManifest(
        schema_version=PREDICTION_SCHEMA_VERSION,
        manifest_kind="predictions",
        model_name=model_name,
        stage=STAGE_NAME,
        outcome=outcome,
        partitions=partitions,
        artifacts=tuple(artifact_refs),
    )
