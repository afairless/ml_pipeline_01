"""Core per-run manifest types: model, prediction, and feature manifests.

Every model, prediction, and feature run records its identity, configuration,
and artifact inventory in a ``manifest.json`` next to its artifacts, and
downstream consumers read that file instead of guessing filenames or
directory depth. Three frozen JSON kinds, each with its own schema version:

- ``ModelManifest`` (``manifest_kind='model'``, ``MODEL_SCHEMA_VERSION == 2``):
  one training run — model name (= strategy id), stage, outcome, estimator/
  task/objective, outcome and feature column names, the declared partition
  names, the ``target_transform`` id (resolved by the project layer), the
  tuned ``best_parameters``, and the run's ``ArtifactRef`` inventory.
- ``PredictionManifest`` (``manifest_kind='predictions'``,
  ``PREDICTION_SCHEMA_VERSION == 1``): one prediction run — the
  partition-tagged prediction files written by s05 for one model run, with
  the partition names plus one ``ArtifactRef`` per file.
- ``FeatureManifest`` (``manifest_kind='feature'``,
  ``FEATURE_SCHEMA_VERSION == 1``): one s02 feature run — the assembled
  table's key columns and row count, its partitions, and the four disjoint
  column sections (model features, categorical columns, metadata columns,
  and the run's artifact inventory). It records no wall clock and no
  absolute source path, so the artifact is deterministic and
  golden-comparable.

The module is project-agnostic (``src.core`` keeps no ``src.project`` import;
``tests/test_module_imports.py`` enforces this for every core module):
manifests carry plain JSON-serializable values, and
``write_manifest``/``read_manifest`` are the only serialize/deserialize
boundary. JSON has no tuples, so ``read_manifest`` re-tuples the list-shaped
fields (``feature_colnames``, ``partitions``, ``artifacts``, ``key_columns``,
``feature_columns``, ``categorical_columns``, ``metadata_columns``) before
constructing the frozen dataclass, keeping the annotations truthful. An
unknown ``manifest_kind``, or a ``schema_version`` that is not the accepted
version of the manifest's own kind, is rejected up front — never read
silently. The kind is read first and each kind validates only its own
version, so the model schema can bump without forcing the prediction or
feature schema to move; a stale version fails with a "regenerate the run"
message.

``load_model_runs`` / ``load_prediction_runs`` provide structural stage-dir
discovery: one ``iterdir()`` over the outcome-scoped stage directory, each
immediate child *directory* being one model/prediction run that must hold
``MANIFEST_FILENAME``. Discovery is explicit — a run directory without a
manifest, a manifest of the wrong kind, or a manifest declaring a different
outcome raises instead of silently dropping the run. Both loaders return runs
sorted by directory name.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

MODEL_SCHEMA_VERSION = 2
"""The accepted ``schema_version`` of a model manifest.

Per-kind version; v2 records the run's ``partitions``, so the model schema
moves independently of the prediction schema."""

PREDICTION_SCHEMA_VERSION = 1
"""The accepted ``schema_version`` of a prediction manifest.

Per-kind version, independent of ``MODEL_SCHEMA_VERSION``."""

FEATURE_SCHEMA_VERSION = 1
"""The accepted ``schema_version`` of a feature manifest.

Per-kind version, independent of the model and prediction schemas. This
repository starts at version 1, so no migration path is required."""

MANIFEST_FILENAME = "manifest.json"
"""The manifest file name written into each run directory."""


@dataclass(frozen=True)
class ArtifactRef:
    """One artifact file belonging to a run.

    Attributes:
        logical_name: Logical artifact name, e.g. ``'model'`` or
            ``'train_key_df'``.
        filename: Actual file name, relative to the run directory: every
            artifact is colocated with the manifest that declares it.
        partition: Partition tag for prediction artifacts (e.g. ``'train'`` /
            ``'valid'`` / ``'test'``); ``None`` for artifacts that are not
            partition-tagged.
    """

    logical_name: str
    filename: str
    partition: str | None = None


@dataclass(frozen=True)
class ModelManifest:
    """The manifest of one model training run.

    Records model identity and configuration plus the run's artifact
    inventory, so downstream stages read identity, columns, and parameters
    from the manifest instead of parsing file or directory names. All
    attributes are JSON-serializable.

    Attributes:
        schema_version: Manifest schema version; must equal
            ``MODEL_SCHEMA_VERSION``.
        manifest_kind: Discriminator; must be ``'model'``.
        model_name: Model identity, i.e. the strategy id (e.g.
            ``'model_00_signal_0'``).
        stage: The training stage that produced the run (e.g. ``'s04_train'``).
        outcome: The modeled outcome (e.g. ``'outcome_1'``).
        estimator: Estimator id (e.g. ``'xgboost'``).
        task: Model task, ``'regression'`` or ``'binary'``.
        objective: The native estimator objective (informative only; ``task``
            is the single source of truth).
        outcome_colname: Column name of the model target.
        feature_colnames: Column names of the model features.
        partitions: The partition names of the run — one entry per split
            table artifact the run writes (e.g. ``('train', 'valid',
            'test')``), in ``PARTITION_ORDER`` order. s05 predicts exactly
            these partitions: the manifest declares the run's split shape
            instead of the consumer hardcoding ``('train', 'valid', 'test')``.
        target_transform: Id of the target's (inverse) transform; the project
            layer resolves the id to callables, and the manifest stores the id
            only because callables are not JSON-serializable.
        best_parameters: The tuned hyperparameters of the run.
        artifacts: The run's artifact inventory, colocated with the manifest
            in the run directory.
    """

    schema_version: int
    manifest_kind: Literal["model"]
    model_name: str
    stage: str
    outcome: str
    estimator: str
    task: str
    objective: str
    outcome_colname: str
    feature_colnames: tuple[str, ...]
    partitions: tuple[str, ...]
    target_transform: str
    best_parameters: Mapping[str, Any]
    artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if self.manifest_kind != "model":
            raise ValueError(
                f'ModelManifest requires manifest_kind="model", got '
                f"{self.manifest_kind!r}"
            )


@dataclass(frozen=True)
class PredictionManifest:
    """The manifest of one s05 prediction run.

    Records the partition-tagged prediction artifacts written for one model
    run, so s06 iterates the recorded partitions instead of filtering
    prediction filenames by substring.

    Attributes:
        schema_version: Manifest schema version; must equal
            ``PREDICTION_SCHEMA_VERSION``.
        manifest_kind: Discriminator; must be ``'predictions'``.
        model_name: Model identity, i.e. the strategy id (e.g.
            ``'model_00_signal_0'``).
        stage: The prediction stage that produced the run (e.g.
            ``'s05_predict'``).
        outcome: The modeled outcome (e.g. ``'outcome_1'``).
        partitions: Partition names of the run (e.g.
            ``('train', 'valid', 'test')``).
        artifacts: One partition-tagged ``ArtifactRef`` per prediction file.
    """

    schema_version: int
    manifest_kind: Literal["predictions"]
    model_name: str
    stage: str
    outcome: str
    partitions: tuple[str, ...]
    artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if self.manifest_kind != "predictions":
            raise ValueError(
                f'PredictionManifest requires manifest_kind="predictions", '
                f"got {self.manifest_kind!r}"
            )


@dataclass(frozen=True)
class FeatureManifest:
    """The manifest of one s02 feature-assembly run.

    Attributes the assembled feature table's columns to four disjoint
    sections, so a consumer can select model features, categorical columns,
    or metadata without reading filenames or stage code.

    Attributes:
        schema_version: Manifest schema version; must equal
            ``FEATURE_SCHEMA_VERSION``.
        manifest_kind: Discriminator; must be ``'feature'``.
        key_columns: The assembled table's key columns, in canonical order.
        n_rows: Number of assembled rows (one per source row).
        partitions: The partition names covered by the feature table.
        feature_columns: Model-feature columns, in table order.
        categorical_columns: Categorical feature columns, in table order.
        metadata_columns: Metadata columns that are not model features, in
            table order.
        artifacts: The run's artifact inventory, colocated with the manifest
            in the run directory.
    """

    schema_version: int
    manifest_kind: Literal["feature"]
    key_columns: tuple[str, ...]
    n_rows: int
    partitions: tuple[str, ...]
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if self.manifest_kind != "feature":
            raise ValueError(
                f'FeatureManifest requires manifest_kind="feature", got '
                f"{self.manifest_kind!r}"
            )


Manifest = ModelManifest | PredictionManifest | FeatureManifest


def write_manifest(manifest: Manifest, path: Path) -> None:
    """Serialize ``manifest`` to JSON at ``path``.

    Tuples are written as JSON lists (JSON has no tuples; ``read_manifest``
    re-tuples them on the way back). Output is deterministic (sorted keys,
    fixed indent), so identical manifests produce byte-identical files.
    Parent directories are *not* created: a stage must ``ensure`` the run
    directory before writing, so the artifact layer has no hidden mkdir side
    effect and writing under a missing parent raises ``FileNotFoundError``.
    """
    data = asdict(manifest)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def read_manifest(path: Path) -> Manifest:
    """Parse the manifest file at ``path`` and return its typed form.

    Dispatches on the ``manifest_kind`` field: ``'model'`` →
    ``ModelManifest``, ``'predictions'`` → ``PredictionManifest``,
    ``'feature'`` → ``FeatureManifest``. The kind is read first and each kind
    validates only its own schema version: the model manifest accepts
    ``MODEL_SCHEMA_VERSION``, the prediction manifest accepts
    ``PREDICTION_SCHEMA_VERSION``, and the feature manifest accepts
    ``FEATURE_SCHEMA_VERSION``, so bumping one kind's schema never forces
    another to move. An unknown ``manifest_kind`` or a wrong version is
    rejected with ``ValueError`` before any type is constructed — a stale
    version gets a "regenerate the run" message — and the list-shaped fields
    are re-tupled so the frozen dataclass annotations remain truthful.

    Raises:
        ValueError: The file is not valid JSON, or carries an unknown
            ``manifest_kind`` or a ``schema_version`` that is not the
            accepted version of its kind.
        TypeError: The document is not a JSON object, or a list-shaped field
            is present under another shape; the frozen dataclass
            constructors also raise ``TypeError`` (or ``ValueError`` from
            the manifest-kind guard) for undeclared or missing fields.
    """
    with open(path, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(
            f"{path} is not a manifest object: expected a JSON object, got "
            f"{type(data).__name__}"
        )
    kind = data.get("manifest_kind")
    if kind == "model":
        _check_schema_version(
            "model", data.get("schema_version"), MODEL_SCHEMA_VERSION, path
        )
        return ModelManifest(
            **_retupled(data, ("feature_colnames", "partitions", "artifacts"))
        )
    if kind == "predictions":
        _check_schema_version(
            "predictions", data.get("schema_version"), PREDICTION_SCHEMA_VERSION, path
        )
        return PredictionManifest(**_retupled(data, ("partitions", "artifacts")))
    if kind == "feature":
        _check_schema_version(
            "feature", data.get("schema_version"), FEATURE_SCHEMA_VERSION, path
        )
        return FeatureManifest(
            **_retupled(
                data,
                (
                    "key_columns",
                    "partitions",
                    "feature_columns",
                    "categorical_columns",
                    "metadata_columns",
                    "artifacts",
                ),
            )
        )
    raise ValueError(f"unknown manifest_kind {kind!r} in {path}")


def _check_schema_version(
    kind: str,
    version: Any,
    expected: int,
    path: Path,
) -> None:
    """Reject a manifest whose ``schema_version`` is not its kind's version.

    Each manifest kind validates only its own accepted version:
    ``MODEL_SCHEMA_VERSION`` for ``'model'``, ``PREDICTION_SCHEMA_VERSION``
    for ``'predictions'``, ``FEATURE_SCHEMA_VERSION`` for ``'feature'``.
    Manifests are not long-lived — a run regenerates them — so a stale
    version is unrecoverable on read: the error tells the operator to re-run
    the producing stage instead of guessing how to reinterpret the file.
    """
    if version != expected:
        raise ValueError(
            f"unsupported schema_version {version!r} for {kind!r} manifest "
            f"in {path}; expected {expected} — regenerate the run"
        )


def _retupled(
    data: Mapping[str, Any],
    tuple_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Normalize decoded JSON back into the frozen dataclass field shapes.

    Each field named in ``tuple_fields`` must have decoded to a JSON list
    (the form ``write_manifest`` produces for tuple fields) and is converted
    back to a tuple; ``artifacts`` entries are additionally converted to
    frozen ``ArtifactRef`` values. A field present under any other shape is a
    malformed manifest and raises ``TypeError`` rather than silently
    corrupting a type annotation.
    """
    out = dict(data)
    for field in tuple_fields:
        value = out.get(field)
        if not isinstance(value, list):
            raise TypeError(
                f"manifest field {field!r} must be a JSON list, got "
                f"{type(value).__name__}"
            )
        if field == "artifacts":
            refs = []
            for entry in value:
                if not isinstance(entry, dict):
                    raise TypeError(
                        f"artifact entry in {field!r} must be a JSON object, "
                        f"got {type(entry).__name__}"
                    )
                refs.append(ArtifactRef(**entry))
            out[field] = tuple(refs)
        else:
            out[field] = tuple(value)
    return out


def declared_artifact_path(
    manifest: Manifest,
    run_dir: Path,
    logical_name: str,
    suffix: str = ".parquet",
) -> Path:
    """Resolve one declared artifact to its path inside ``run_dir``.

    Every artifact is colocated with the manifest that declares it, so a
    consumer finds a file through the manifest's artifact inventory instead of
    scanning the run directory or parsing filenames. A logical name may be
    declared once per format (a parquet/csv pair), so the caller pins the
    format with ``suffix``; exactly one matching ``ArtifactRef.filename`` is
    required.

    Args:
        manifest: The run's model or prediction manifest.
        run_dir: The run directory holding the manifest and its colocated
            artifacts.
        logical_name: The logical artifact name to resolve, e.g.
            ``'train_key_df'``.
        suffix: The required filename suffix, pinning the artifact format.

    Returns:
        ``run_dir / filename`` of the single matching declared artifact.

    Raises:
        ValueError: The manifest declares no matching artifact, or more than
            one; the message names the run, logical name, suffix, and the
            sorted candidate filenames.
    """
    filenames = [
        ref.filename
        for ref in manifest.artifacts
        if ref.logical_name == logical_name and ref.filename.endswith(suffix)
    ]
    if len(filenames) != 1:
        label = getattr(manifest, "model_name", manifest.manifest_kind)
        raise ValueError(
            f"run {label!r} must declare exactly one "
            f"{logical_name!r} {suffix!r} artifact, got {sorted(filenames)}"
        )
    return Path(run_dir) / filenames[0]


def _run_manifest_paths(stage_dir: Path, outcome: str) -> Iterator[Path]:
    """Yield the manifest path of every run directory under ``stage_dir``.

    ``stage_dir`` is the outcome-scoped stage directory (e.g.
    ``layout.stage_dir('s04_train', 'outcome_1')``): each immediate child
    *directory* is one run and must hold ``MANIFEST_FILENAME``. Plain files
    in the stage directory are not runs and are skipped; children sorted by
    name keep discovery deterministic. A child directory without its
    manifest, or a missing stage directory, raises instead of silently
    dropping a run.
    """
    root = Path(stage_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"no {outcome!r} stage directory {root}")
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        manifest_path = child / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ValueError(
                f"run directory {child} has no {MANIFEST_FILENAME!r}; every "
                "run must declare its manifest"
            )
        yield manifest_path


def load_model_runs(stage_dir: Path, outcome: str) -> list[ModelManifest]:
    """Structurally load every model run under one outcome's stage directory.

    ``stage_dir`` is the outcome-scoped stage directory; each immediate child
    directory is one model run holding ``MANIFEST_FILENAME``. Runs are
    returned sorted by directory name.

    Raises:
        FileNotFoundError: ``stage_dir`` does not exist.
        ValueError: A run directory lacks its manifest, holds a non-model
            manifest, declares an outcome other than ``outcome``, or carries
            an unknown ``schema_version``/``manifest_kind``.
    """
    runs: list[ModelManifest] = []
    for manifest_path in _run_manifest_paths(stage_dir, outcome):
        manifest = read_manifest(manifest_path)
        if manifest.manifest_kind != "model":
            raise ValueError(
                f"{manifest_path} is a {manifest.manifest_kind!r} manifest, "
                "expected 'model'"
            )
        if manifest.outcome != outcome:
            raise ValueError(
                f"{manifest_path} declares outcome {manifest.outcome!r}, "
                f"expected {outcome!r}"
            )
        assert isinstance(manifest, ModelManifest), "kind guard guarantees this"
        runs.append(manifest)
    return runs


def load_prediction_runs(
    stage_dir: Path,
    outcome: str,
) -> list[PredictionManifest]:
    """Structurally load every prediction run under one outcome's stage dir.

    Mirrors ``load_model_runs`` for ``PredictionManifest``: one prediction
    run per model, holding the partition-tagged prediction artifacts of that
    run. Raises exactly as ``load_model_runs`` does.
    """
    runs: list[PredictionManifest] = []
    for manifest_path in _run_manifest_paths(stage_dir, outcome):
        manifest = read_manifest(manifest_path)
        if manifest.manifest_kind != "predictions":
            raise ValueError(
                f"{manifest_path} is a {manifest.manifest_kind!r} manifest, "
                "expected 'predictions'"
            )
        if manifest.outcome != outcome:
            raise ValueError(
                f"{manifest_path} declares outcome {manifest.outcome!r}, "
                f"expected {outcome!r}"
            )
        assert isinstance(manifest, PredictionManifest), "kind guard guarantees this"
        runs.append(manifest)
    return runs
