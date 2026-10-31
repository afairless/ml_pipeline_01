"""Unit and property tests for the core manifest types.

Pins the invariant that a manifest survives the serialize boundary
unchanged: ``write_manifest`` then ``read_manifest`` reproduces the frozen
dataclass exactly (tuples re-tupled from JSON lists), and the structural
run loaders discover every produced run while raising on a broken or
misplaced one instead of silently dropping it. A failure here means identity
or the declared artifact inventory could drift between producer and consumer
without a stage noticing.

The property tests exercise an unbounded space of manifest values with
Hypothesis: round-trip stability, tuple re-tupling, and deterministic
serialization hold for arbitrary JSON-serializable field values, not only
for the hand-picked examples.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.core.manifest import (
    FEATURE_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    MODEL_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    ArtifactRef,
    FeatureManifest,
    ModelManifest,
    PredictionManifest,
    declared_artifact_path,
    load_model_runs,
    load_prediction_runs,
    read_manifest,
    write_manifest,
)

MODEL_ARTIFACTS = (
    ArtifactRef(logical_name="model", filename="model.json"),
    ArtifactRef(logical_name="train_key_df", filename="train_key_df.parquet"),
    ArtifactRef(logical_name="feature_importances", filename="feature_importances.csv"),
)
MODEL_PARTITIONS = ("train", "valid", "test")
PREDICTION_PARTITIONS = ("train", "valid", "test")


def asdict_artifact(ref: ArtifactRef) -> dict[str, Any]:
    """The plain-dict form ``write_manifest`` produces for one artifact."""
    return {
        "logical_name": ref.logical_name,
        "filename": ref.filename,
        "partition": ref.partition,
    }


def make_model_manifest(**overrides: Any) -> ModelManifest:
    """A representative model manifest; ``overrides`` replace any field."""
    fields: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "manifest_kind": "model",
        "model_name": "model_00_signal_0",
        "stage": "s04_train",
        "outcome": "outcome_1",
        "estimator": "xgboost",
        "task": "regression",
        "objective": "reg:absoluteerror",
        "outcome_colname": "outcome_1",
        "feature_colnames": ("feature_1", "feature_2", "feature_3"),
        "partitions": MODEL_PARTITIONS,
        "target_transform": "identity",
        "best_parameters": {"eta": 0.05, "max_depth": 6},
        "artifacts": MODEL_ARTIFACTS,
    }
    fields.update(overrides)
    return ModelManifest(**fields)


def make_prediction_manifest(**overrides: Any) -> PredictionManifest:
    """A representative prediction manifest; ``overrides`` replace fields."""
    fields: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "manifest_kind": "predictions",
        "model_name": "model_00_signal_0",
        "stage": "s05_predict",
        "outcome": "outcome_1",
        "partitions": PREDICTION_PARTITIONS,
        "artifacts": tuple(
            ArtifactRef(
                logical_name=f"{partition}_predictions",
                filename=(f"model_00_signal_0_{partition}_predictions_df.parquet"),
                partition=partition,
            )
            for partition in PREDICTION_PARTITIONS
        ),
    }
    fields.update(overrides)
    return PredictionManifest(**fields)


def make_feature_manifest(**overrides: Any) -> FeatureManifest:
    """A representative feature manifest; ``overrides`` replace fields."""
    fields: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "manifest_kind": "feature",
        "key_columns": ("case_id", "case_date"),
        "n_rows": 3,
        "partitions": MODEL_PARTITIONS,
        "feature_columns": ("feature_1", "feature_2"),
        "categorical_columns": ("feature_8",),
        "metadata_columns": ("case_group",),
        "artifacts": (
            ArtifactRef(logical_name="feature_df", filename="feature_df.parquet"),
            ArtifactRef(logical_name="feature_head_df", filename="feature_df_head.csv"),
        ),
    }
    fields.update(overrides)
    return FeatureManifest(**fields)


def feature_json_dict() -> dict[str, Any]:
    """The feature manifest as decoded JSON: list-shaped tuple fields."""
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "manifest_kind": "feature",
        "key_columns": ["case_id", "case_date"],
        "n_rows": 2,
        "partitions": ["train", "valid", "test"],
        "feature_columns": ["feature_1", "feature_2"],
        "categorical_columns": ["feature_8"],
        "metadata_columns": ["case_group"],
        "artifacts": [
            {
                "logical_name": "feature_df",
                "filename": "feature_df.parquet",
                "partition": None,
            },
        ],
    }


def model_json_dict() -> dict[str, Any]:
    """The model manifest as decoded JSON: list-shaped tuple fields."""
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "manifest_kind": "model",
        "model_name": "model_00_signal_0",
        "stage": "s04_train",
        "outcome": "outcome_1",
        "estimator": "xgboost",
        "task": "regression",
        "objective": "reg:absoluteerror",
        "outcome_colname": "outcome_1",
        "feature_colnames": ["feature_1", "feature_2"],
        "partitions": ["train", "valid", "test"],
        "target_transform": "identity",
        "best_parameters": {"eta": 0.05},
        "artifacts": [
            {"logical_name": "model", "filename": "model.json", "partition": None},
        ],
    }


def prediction_json_dict() -> dict[str, Any]:
    """The prediction manifest as decoded JSON: list-shaped tuple fields."""
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "manifest_kind": "predictions",
        "model_name": "model_00_signal_0",
        "stage": "s05_predict",
        "outcome": "outcome_1",
        "partitions": ["train", "valid", "test"],
        "artifacts": [
            {
                "logical_name": "train_predictions",
                "filename": "model_00_signal_0_train_predictions_df.parquet",
                "partition": "train",
            },
        ],
    }


def write_json_doc(tmp_path: Path, data: Mapping[str, Any]) -> Path:
    """Write ``data`` as a JSON manifest file under ``tmp_path``."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_run(
    tmp_path: Path, run_name: str, manifest: ModelManifest | PredictionManifest
) -> Path:
    """Stage one run directory holding ``manifest.json`` under ``tmp_path``."""
    run_dir = tmp_path / run_name
    run_dir.mkdir()
    write_manifest(manifest, run_dir / MANIFEST_FILENAME)
    return run_dir


# --- declared artifact resolution ------------------------------------------


def test_declared_artifact_path_resolves_single_match(
    tmp_path: Path,
) -> None:
    """One declared parquet under the logical name resolves to its run-dir
    path; the csv sibling of another logical name is irrelevant."""
    manifest = make_model_manifest()
    assert declared_artifact_path(manifest, tmp_path, "train_key_df") == (
        tmp_path / "train_key_df.parquet"
    )


def test_declared_artifact_path_no_match_raises(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="model_00_signal_0.*'absent'.*'.parquet'.*\\[\\]",
    ):
        declared_artifact_path(make_model_manifest(), tmp_path, "absent")


def test_declared_artifact_path_two_matches_raises(
    tmp_path: Path,
) -> None:
    """Two parquet files declared under one logical name is an ambiguous
    inventory, so the resolver raises instead of picking one."""
    manifest = make_model_manifest(
        artifacts=(
            ArtifactRef(logical_name="train_key_df", filename="a.parquet"),
            ArtifactRef(logical_name="train_key_df", filename="b.parquet"),
        )
    )
    with pytest.raises(
        ValueError,
        match="model_00_signal_0.*'train_key_df'.*'\\.parquet'.*a\\.parquet.*b\\.parquet",
    ):
        declared_artifact_path(manifest, tmp_path, "train_key_df")


def test_declared_artifact_path_csv_only_default_suffix_raises(
    tmp_path: Path,
) -> None:
    """``feature_importances`` is declared as csv only in this inventory;
    the default parquet suffix finds no match and raises."""
    with pytest.raises(
        ValueError,
        match="model_00_signal_0.*'feature_importances'.*'\\.parquet'.*\\[\\]",
    ):
        declared_artifact_path(make_model_manifest(), tmp_path, "feature_importances")


def test_declared_artifact_path_custom_suffix_matches(
    tmp_path: Path,
) -> None:
    assert declared_artifact_path(
        make_model_manifest(), tmp_path, "feature_importances", suffix=".csv"
    ) == (tmp_path / "feature_importances.csv")


def test_declared_artifact_path_accepts_feature_manifest(
    tmp_path: Path,
) -> None:
    """The resolver reads the artifact inventory of every manifest kind."""
    assert declared_artifact_path(make_feature_manifest(), tmp_path, "feature_df") == (
        tmp_path / "feature_df.parquet"
    )


# --- frozen declarations and the manifest_kind discriminator ----------------


def test_artifact_ref_stores_declared_fields() -> None:
    ref = ArtifactRef(logical_name="model", filename="model.json")
    assert ref.logical_name == "model"
    assert ref.filename == "model.json"
    assert ref.partition is None


def test_artifact_ref_can_be_partition_tagged() -> None:
    ref = ArtifactRef(
        logical_name="test_predictions",
        filename="model_00_signal_0_test_predictions_df.parquet",
        partition="test",
    )
    assert ref.partition == "test"


def test_artifact_ref_is_frozen() -> None:
    ref = ArtifactRef(logical_name="model", filename="model.json")
    with pytest.raises(FrozenInstanceError):
        ref.filename = "train_key_df.parquet"  # type: ignore[misc]


def test_model_manifest_is_frozen() -> None:
    manifest = make_model_manifest()
    with pytest.raises(FrozenInstanceError):
        manifest.model_name = "model_01a"  # type: ignore[misc]


def test_prediction_manifest_is_frozen() -> None:
    manifest = make_prediction_manifest()
    with pytest.raises(FrozenInstanceError):
        manifest.partitions = ("test",)  # type: ignore[misc]


def test_feature_manifest_is_frozen() -> None:
    manifest = make_feature_manifest()
    with pytest.raises(FrozenInstanceError):
        manifest.n_rows = 4  # type: ignore[misc]


def test_feature_manifest_requires_feature_kind() -> None:
    with pytest.raises(ValueError, match="manifest_kind"):
        make_feature_manifest(manifest_kind="model")


def test_model_manifest_requires_model_kind() -> None:
    with pytest.raises(ValueError, match="manifest_kind"):
        make_model_manifest(manifest_kind="predictions")


def test_prediction_manifest_requires_predictions_kind() -> None:
    with pytest.raises(ValueError, match="manifest_kind"):
        make_prediction_manifest(manifest_kind="model")


# --- write/read round trip --------------------------------------------------


def test_write_read_round_trip_model_manifest(tmp_path: Path) -> None:
    manifest = make_model_manifest()
    path = tmp_path / "model_00_signal_0" / MANIFEST_FILENAME
    path.parent.mkdir()
    write_manifest(manifest, path)
    loaded = read_manifest(path)
    assert isinstance(loaded, ModelManifest)
    assert loaded == manifest


def test_write_read_round_trip_prediction_manifest(tmp_path: Path) -> None:
    manifest = make_prediction_manifest()
    path = tmp_path / MANIFEST_FILENAME
    write_manifest(manifest, path)
    loaded = read_manifest(path)
    assert isinstance(loaded, PredictionManifest)
    assert loaded == manifest
    assert loaded.partitions == PREDICTION_PARTITIONS


def test_write_read_round_trip_feature_manifest(tmp_path: Path) -> None:
    manifest = make_feature_manifest()
    path = tmp_path / "feature_manifest.json"
    write_manifest(manifest, path)
    loaded = read_manifest(path)
    assert isinstance(loaded, FeatureManifest)
    assert loaded == manifest
    assert loaded.key_columns == ("case_id", "case_date")
    assert loaded.partitions == MODEL_PARTITIONS
    assert loaded.feature_columns == ("feature_1", "feature_2")
    assert loaded.categorical_columns == ("feature_8",)
    assert loaded.metadata_columns == ("case_group",)
    assert loaded.artifacts[0].filename == "feature_df.parquet"


def test_write_manifest_writes_json_lists_with_kind_and_version(
    tmp_path: Path,
) -> None:
    """Tuples serialize as JSON lists and the discriminator is in the file."""
    path = tmp_path / MANIFEST_FILENAME
    write_manifest(make_model_manifest(), path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["manifest_kind"] == "model"
    assert raw["schema_version"] == MODEL_SCHEMA_VERSION
    assert raw["partitions"] == list(MODEL_PARTITIONS)
    assert raw["artifacts"] == [asdict_artifact(ref) for ref in MODEL_ARTIFACTS]


def test_write_manifest_is_deterministic(tmp_path: Path) -> None:
    """Property: identical manifests serialize to byte-identical files."""
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    write_manifest(make_model_manifest(), first)
    write_manifest(make_model_manifest(), second)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    write_manifest(make_prediction_manifest(), first)
    write_manifest(make_prediction_manifest(), second)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    write_manifest(make_feature_manifest(), first)
    write_manifest(make_feature_manifest(), second)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_write_manifest_does_not_create_parent_dirs(tmp_path: Path) -> None:
    """The artifact layer does no hidden mkdir; the stage must ``ensure``
    the run directory first. Without that rule a write to a missing
    directory would silently create it instead of failing at the unensured
    boundary."""
    with pytest.raises(FileNotFoundError):
        write_manifest(make_model_manifest(), tmp_path / "missing" / MANIFEST_FILENAME)


# --- read_manifest: dispatch, re-tupling, strict schema ---------------------


def test_read_manifest_dispatches_on_manifest_kind(tmp_path: Path) -> None:
    assert isinstance(
        read_manifest(write_json_doc(tmp_path, model_json_dict())), ModelManifest
    )
    assert isinstance(
        read_manifest(write_json_doc(tmp_path, prediction_json_dict())),
        PredictionManifest,
    )
    assert isinstance(
        read_manifest(write_json_doc(tmp_path, feature_json_dict())), FeatureManifest
    )


def test_read_manifest_retuples_json_lists(tmp_path: Path) -> None:
    """JSON has no tuples: the loader re-tuples list-shaped fields so the
    frozen dataclass annotations stay truthful. A reader that returned
    lists would let a consumer annotate a tuple but hold a list."""
    loaded = read_manifest(write_json_doc(tmp_path, model_json_dict()))
    assert isinstance(loaded, ModelManifest)
    assert isinstance(loaded.feature_colnames, tuple)
    assert loaded.feature_colnames == ("feature_1", "feature_2")
    # The model manifest records its partitions, re-tupled on read.
    assert isinstance(loaded.partitions, tuple)
    assert loaded.partitions == ("train", "valid", "test")
    assert isinstance(loaded.artifacts, tuple)
    assert all(isinstance(ref, ArtifactRef) for ref in loaded.artifacts)
    assert loaded.artifacts[0].filename == "model.json"
    assert loaded.artifacts[0].partition is None

    pred_loaded = read_manifest(write_json_doc(tmp_path, prediction_json_dict()))
    assert isinstance(pred_loaded, PredictionManifest)
    assert isinstance(pred_loaded.partitions, tuple)
    assert pred_loaded.partitions == ("train", "valid", "test")
    assert pred_loaded.artifacts[0].partition == "train"


def test_read_manifest_retuples_all_feature_tuple_fields(
    tmp_path: Path,
) -> None:
    """Every list-shaped feature field is re-tupled, including the artifact
    inventory, so a consumer never holds a JSON list under a tuple
    annotation."""
    loaded = read_manifest(write_json_doc(tmp_path, feature_json_dict()))
    assert isinstance(loaded, FeatureManifest)
    assert isinstance(loaded.key_columns, tuple)
    assert isinstance(loaded.partitions, tuple)
    assert isinstance(loaded.feature_columns, tuple)
    assert isinstance(loaded.categorical_columns, tuple)
    assert isinstance(loaded.metadata_columns, tuple)
    assert isinstance(loaded.artifacts, tuple)
    assert loaded.artifacts[0] == ArtifactRef(
        logical_name="feature_df", filename="feature_df.parquet"
    )


@pytest.mark.parametrize("bad_version", [1, 0, -1, "1", None])
def test_read_manifest_rejects_unknown_model_schema_version(
    tmp_path: Path,
    bad_version: Any,
) -> None:
    """An unknown model-manifest version is rejected, never read: a stale
    v1 (the pre-``partitions`` schema) or a future version gets the
    "regenerate the run" error instead of being read at v2. Silently
    reading a stale run is the failure this catches."""
    data = model_json_dict()
    data["schema_version"] = bad_version
    with pytest.raises(ValueError, match="schema_version.*regenerate the run"):
        read_manifest(write_json_doc(tmp_path, data))


@pytest.mark.parametrize("bad_version", [2, 0, -1, "1", None])
def test_read_manifest_rejects_unknown_prediction_schema_version(
    tmp_path: Path,
    bad_version: Any,
) -> None:
    """The predictions kind validates its own version (still 1): a bump it
    does not recognize is rejected, never read."""
    pred = prediction_json_dict()
    pred["schema_version"] = bad_version
    with pytest.raises(ValueError, match="schema_version.*regenerate the run"):
        read_manifest(write_json_doc(tmp_path, pred))


@pytest.mark.parametrize("bad_version", [2, 0, -1, "1", None])
def test_read_manifest_rejects_unknown_feature_schema_version(
    tmp_path: Path,
    bad_version: Any,
) -> None:
    """The feature kind validates its own version and rejects any other."""
    feature = feature_json_dict()
    feature["schema_version"] = bad_version
    with pytest.raises(ValueError, match="schema_version.*regenerate the run"):
        read_manifest(write_json_doc(tmp_path, feature))


def test_read_manifest_kind_first_version_validation(tmp_path: Path) -> None:
    """Version checks are per kind: the same wrong version number is an
    error only for the kind whose accepted version it is not. The model
    manifest accepts 2 and the prediction manifest accepts 1; each rejects
    the other kind's version. A single global version check would accept a
    manifest at the wrong kind's version."""
    model_at_prediction_version = model_json_dict()
    model_at_prediction_version["schema_version"] = 1
    with pytest.raises(ValueError, match="regenerate the run"):
        read_manifest(write_json_doc(tmp_path, model_at_prediction_version))

    predictions_at_model_version = prediction_json_dict()
    predictions_at_model_version["schema_version"] = 2
    with pytest.raises(ValueError, match="regenerate the run"):
        read_manifest(write_json_doc(tmp_path, predictions_at_model_version))


def test_read_manifest_rejects_unknown_manifest_kind(tmp_path: Path) -> None:
    data = model_json_dict()
    data["manifest_kind"] = "params"
    with pytest.raises(ValueError, match="manifest_kind"):
        read_manifest(write_json_doc(tmp_path, data))


@pytest.mark.parametrize("field", ["feature_colnames", "partitions", "artifacts"])
def test_read_manifest_rejects_non_list_tuple_field(
    tmp_path: Path,
    field: str,
) -> None:
    """A tuple-typed field that did not decode to a JSON list is malformed."""
    data = model_json_dict()
    data[field] = "not-a-list"
    with pytest.raises(TypeError, match=field):
        read_manifest(write_json_doc(tmp_path, data))


@pytest.mark.parametrize(
    "field",
    [
        "key_columns",
        "partitions",
        "feature_columns",
        "categorical_columns",
        "metadata_columns",
        "artifacts",
    ],
)
def test_read_manifest_rejects_non_list_feature_tuple_field(
    tmp_path: Path,
    field: str,
) -> None:
    data = feature_json_dict()
    data[field] = "not-a-list"
    with pytest.raises(TypeError, match=field):
        read_manifest(write_json_doc(tmp_path, data))


def test_read_manifest_rejects_non_object_artifact_entry(
    tmp_path: Path,
) -> None:
    data = model_json_dict()
    data["artifacts"] = ["model.json"]
    with pytest.raises(TypeError, match="artifact entry"):
        read_manifest(write_json_doc(tmp_path, data))


def test_model_manifest_requires_partitions(tmp_path: Path) -> None:
    """``partitions`` is a required field of the model schema: a manifest
    serialized without it is malformed and raises on read, never silently
    defaulted. A defaulted empty partition set would make s05 predict
    nothing without an error."""
    data = model_json_dict()
    del data["partitions"]
    with pytest.raises(TypeError, match="partitions"):
        read_manifest(write_json_doc(tmp_path, data))


def test_feature_manifest_requires_every_declared_section(
    tmp_path: Path,
) -> None:
    """Each canonical feature section is required: a manifest missing one
    is malformed, never silently read with a default."""
    for field in (
        "key_columns",
        "n_rows",
        "partitions",
        "feature_columns",
        "categorical_columns",
        "metadata_columns",
        "artifacts",
    ):
        data = feature_json_dict()
        del data[field]
        with pytest.raises(TypeError, match=field):
            read_manifest(write_json_doc(tmp_path, data))


def test_read_manifest_rejects_undeclared_fields(tmp_path: Path) -> None:
    """A manifest with undeclared fields breaks the frozen schema."""
    data = model_json_dict()
    data["extra_field"] = 1
    with pytest.raises(TypeError):
        read_manifest(write_json_doc(tmp_path, data))


def test_read_manifest_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / MANIFEST_FILENAME
    path.write_text("not json {", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_manifest(path)


def test_read_manifest_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / MANIFEST_FILENAME
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(TypeError, match="not a manifest object"):
        read_manifest(path)


# --- property space: round trips over representative manifests --------------


MODEL_VARIANTS = (
    make_model_manifest(),  # every field populated
    make_model_manifest(
        model_name="model_01a",
        outcome="outcome_3",
        task="binary",
        objective="binary:logistic",
        outcome_colname="outcome_3",
        feature_colnames=(),
        partitions=("train", "test"),
        target_transform="log10_shift",
        best_parameters={},
        artifacts=(),
    ),
    make_model_manifest(
        best_parameters={"nested": {"a": 1}, "flag": True, "none": None}
    ),
    make_model_manifest(
        artifacts=(ArtifactRef(logical_name="model", filename="model.json"),)
    ),
)
MODEL_VARIANT_IDS = ("full", "binary-empty", "nested-params", "single-artifact")

PREDICTION_VARIANTS = (
    make_prediction_manifest(),  # three partitions, three artifacts
    make_prediction_manifest(
        model_name="model_01a",
        outcome="outcome_3",
        partitions=("test",),
        artifacts=(
            ArtifactRef(
                logical_name="test_predictions",
                filename="model_01a_test_predictions_df.parquet",
                partition="test",
            ),
        ),
    ),
    make_prediction_manifest(partitions=()),
)
PREDICTION_VARIANT_IDS = ("holdout", "single-partition", "no-partitions")

FEATURE_VARIANTS = (
    make_feature_manifest(),
    make_feature_manifest(
        partitions=(),
        feature_columns=(),
        categorical_columns=(),
        metadata_columns=(),
        artifacts=(),
        n_rows=0,
    ),
    make_feature_manifest(
        categorical_columns=("feature_7", "feature_8"),
        metadata_columns=("case_group", "case_source"),
    ),
)
FEATURE_VARIANT_IDS = ("full", "empty", "expanded")


@pytest.mark.parametrize("manifest", MODEL_VARIANTS, ids=MODEL_VARIANT_IDS)
def test_model_round_trip_preserves_every_field(
    manifest: ModelManifest,
    tmp_path: Path,
) -> None:
    """Property: write -> read reproduces the frozen manifest exactly."""
    path = tmp_path / MANIFEST_FILENAME
    write_manifest(manifest, path)
    assert read_manifest(path) == manifest


@pytest.mark.parametrize("manifest", PREDICTION_VARIANTS, ids=PREDICTION_VARIANT_IDS)
def test_prediction_round_trip_preserves_every_field(
    manifest: PredictionManifest,
    tmp_path: Path,
) -> None:
    """Property: write -> read reproduces the frozen manifest exactly."""
    path = tmp_path / MANIFEST_FILENAME
    write_manifest(manifest, path)
    assert read_manifest(path) == manifest


@pytest.mark.parametrize("manifest", FEATURE_VARIANTS, ids=FEATURE_VARIANT_IDS)
def test_feature_round_trip_preserves_every_field(
    manifest: FeatureManifest,
    tmp_path: Path,
) -> None:
    """Property: write -> read reproduces the frozen manifest exactly."""
    path = tmp_path / MANIFEST_FILENAME
    write_manifest(manifest, path)
    assert read_manifest(path) == manifest


def test_read_manifest_twice_is_idempotent(tmp_path: Path) -> None:
    """Property: re-reading the same file yields an equal manifest."""
    path = tmp_path / MANIFEST_FILENAME
    write_manifest(make_prediction_manifest(), path)
    assert read_manifest(path) == read_manifest(path)


JSON_SCALAR = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(),
)


@given(
    model_name=st.text(),
    feature_colnames=st.lists(st.text(), max_size=6),
    partitions=st.lists(st.text(), max_size=4),
    best_parameters=st.dictionaries(st.text(), JSON_SCALAR, max_size=4),
)
def test_model_round_trip_property(
    model_name: str,
    feature_colnames: list[str],
    partitions: list[str],
    best_parameters: dict[str, Any],
) -> None:
    """Property: every arbitrary JSON-serializable model manifest survives
    write -> read unchanged, with tuples restored."""
    manifest = make_model_manifest(
        model_name=model_name,
        feature_colnames=tuple(feature_colnames),
        partitions=tuple(partitions),
        best_parameters=best_parameters,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / MANIFEST_FILENAME
        write_manifest(manifest, path)
        loaded = read_manifest(path)
    assert loaded == manifest
    assert isinstance(loaded, ModelManifest)
    assert isinstance(loaded.feature_colnames, tuple)
    assert isinstance(loaded.partitions, tuple)


@given(
    key_columns=st.lists(st.text(), max_size=4),
    n_rows=st.integers(min_value=0),
    partitions=st.lists(st.text(), max_size=4),
    feature_columns=st.lists(st.text(), max_size=6),
    categorical_columns=st.lists(st.text(), max_size=4),
    metadata_columns=st.lists(st.text(), max_size=4),
)
def test_feature_round_trip_property(
    key_columns: list[str],
    n_rows: int,
    partitions: list[str],
    feature_columns: list[str],
    categorical_columns: list[str],
    metadata_columns: list[str],
) -> None:
    """Property: every arbitrary JSON-serializable feature manifest survives
    write -> read unchanged, with every tuple field restored."""
    manifest = make_feature_manifest(
        key_columns=tuple(key_columns),
        n_rows=n_rows,
        partitions=tuple(partitions),
        feature_columns=tuple(feature_columns),
        categorical_columns=tuple(categorical_columns),
        metadata_columns=tuple(metadata_columns),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / MANIFEST_FILENAME
        write_manifest(manifest, path)
        loaded = read_manifest(path)
    assert loaded == manifest
    assert isinstance(loaded, FeatureManifest)
    assert isinstance(loaded.key_columns, tuple)
    assert isinstance(loaded.partitions, tuple)
    assert isinstance(loaded.feature_columns, tuple)
    assert isinstance(loaded.categorical_columns, tuple)
    assert isinstance(loaded.metadata_columns, tuple)


# --- load_model_runs: structural run discovery -------------------------------


def test_load_model_runs_discovers_sorted_run_dirs(tmp_path: Path) -> None:
    write_run(tmp_path, "model_01a", make_model_manifest(model_name="model_01a"))
    write_run(
        tmp_path,
        "model_00_signal_0",
        make_model_manifest(model_name="model_00_signal_0"),
    )
    write_run(
        tmp_path,
        "model_00_signal_1",
        make_model_manifest(model_name="model_00_signal_1"),
    )
    # plain files in the stage dir are not runs and are skipped
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    runs = load_model_runs(tmp_path, "outcome_1")
    assert [run.model_name for run in runs] == [
        "model_00_signal_0",
        "model_00_signal_1",
        "model_01a",
    ]
    assert all(isinstance(run, ModelManifest) for run in runs)


def test_load_model_runs_missing_stage_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="stage directory"):
        load_model_runs(tmp_path / "absent", "outcome_1")


def test_load_model_runs_empty_stage_dir_returns_empty(tmp_path: Path) -> None:
    assert load_model_runs(tmp_path, "outcome_1") == []


def test_load_model_runs_missing_manifest_raises(tmp_path: Path) -> None:
    (tmp_path / "model_00_signal_0").mkdir()
    with pytest.raises(ValueError, match="manifest"):
        load_model_runs(tmp_path, "outcome_1")


def test_load_model_runs_wrong_kind_raises(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "model_00_signal_0",
        make_prediction_manifest(model_name="model_00_signal_0"),
    )
    with pytest.raises(ValueError, match="expected 'model'"):
        load_model_runs(tmp_path, "outcome_1")


def test_load_model_runs_outcome_mismatch_raises(tmp_path: Path) -> None:
    """A run declaring a different outcome is misplaced and never read as if
    it belonged to the requested outcome."""
    write_run(tmp_path, "model_00_signal_0", make_model_manifest(outcome="outcome_3"))
    with pytest.raises(ValueError, match="declares outcome"):
        load_model_runs(tmp_path, "outcome_1")


def test_load_model_runs_unknown_schema_version_raises(
    tmp_path: Path,
) -> None:
    """A model run at the pre-v2 schema or at any version the model kind
    does not accept is rejected by discovery with the "regenerate the run"
    message; discovery must never return a stale run as if it were current."""
    write_run(
        tmp_path,
        "model_00_signal_0",
        make_model_manifest(schema_version=MODEL_SCHEMA_VERSION - 1),
    )
    with pytest.raises(ValueError, match="schema_version.*regenerate the run"):
        load_model_runs(tmp_path, "outcome_1")


# --- load_prediction_runs: mirrors load_model_runs --------------------------


def test_load_prediction_runs_discovers_sorted_run_dirs(
    tmp_path: Path,
) -> None:
    write_run(tmp_path, "model_01a", make_prediction_manifest(model_name="model_01a"))
    write_run(
        tmp_path,
        "model_00_signal_0",
        make_prediction_manifest(model_name="model_00_signal_0"),
    )
    runs = load_prediction_runs(tmp_path, "outcome_1")
    assert [run.model_name for run in runs] == ["model_00_signal_0", "model_01a"]
    assert all(isinstance(run, PredictionManifest) for run in runs)
    assert runs[0].partitions == PREDICTION_PARTITIONS


def test_load_prediction_runs_missing_stage_dir_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="stage directory"):
        load_prediction_runs(tmp_path / "absent", "outcome_1")


def test_load_prediction_runs_empty_stage_dir_returns_empty(
    tmp_path: Path,
) -> None:
    assert load_prediction_runs(tmp_path, "outcome_1") == []


def test_load_prediction_runs_missing_manifest_raises(tmp_path: Path) -> None:
    (tmp_path / "model_00_signal_0").mkdir()
    with pytest.raises(ValueError, match="manifest"):
        load_prediction_runs(tmp_path, "outcome_1")


def test_load_prediction_runs_wrong_kind_raises(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "model_00_signal_0",
        make_model_manifest(model_name="model_00_signal_0"),
    )
    with pytest.raises(ValueError, match="expected 'predictions'"):
        load_prediction_runs(tmp_path, "outcome_1")


def test_load_prediction_runs_outcome_mismatch_raises(tmp_path: Path) -> None:
    write_run(
        tmp_path, "model_00_signal_0", make_prediction_manifest(outcome="outcome_3")
    )
    with pytest.raises(ValueError, match="declares outcome"):
        load_prediction_runs(tmp_path, "outcome_1")
