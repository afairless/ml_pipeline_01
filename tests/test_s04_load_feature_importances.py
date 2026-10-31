"""Unit tests for the manifest-driven feature-importances reader.

``load_feature_importances`` resolves one run's ``feature_importances``
parquet through its ``ModelManifest`` artifact inventory, reads it, and
validates the frame at the consume boundary with
``S04_FEATURE_IMPORTANCES_SCHEMA``. These tests pin that behaviour on a staged
run: a valid table comes back unchanged, a run without (or with several)
declared parquet references raises ``ValueError``, a declared-but-absent file
raises ``FileNotFoundError``, and a malformed table raises a boundary error
naming the consume boundary.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.core.manifest import (
    MODEL_SCHEMA_VERSION,
    ArtifactRef,
    ModelManifest,
    write_manifest,
)
from src.project.artifacts import (
    S04_FEATURE_IMPORTANCES_BOUNDARY,
    load_feature_importances,
)
from src.project.columns import (
    FEATURE_COLNAME,
    FEATURE_ESTIMATOR_ID_COLNAME,
    FEATURE_IMPORTANCE_COLNAME,
)
from src.project.schemas import BoundaryError


def _valid_importances() -> pl.DataFrame:
    """A producer-shaped feature-importances table."""
    return pl.DataFrame(
        {
            FEATURE_COLNAME: ["feature_1", "feature_2"],
            FEATURE_ESTIMATOR_ID_COLNAME: ["f0", "f1"],
            FEATURE_IMPORTANCE_COLNAME: [0.8, 0.2],
        }
    )


def _manifest(artifacts: tuple[ArtifactRef, ...]) -> ModelManifest:
    """A minimal model manifest declaring ``artifacts``."""
    return ModelManifest(
        schema_version=MODEL_SCHEMA_VERSION,
        manifest_kind="model",
        model_name="model_00_signal_0",
        stage="s04_train",
        outcome="outcome_1",
        estimator="xgboost",
        task="regression",
        objective="reg:absoluteerror",
        outcome_colname="outcome_1",
        feature_colnames=("feature_1", "feature_2"),
        partitions=("train", "valid", "test"),
        target_transform="identity",
        best_parameters={},
        artifacts=artifacts,
    )


def _staged_run(
    run_dir: Path,
    artifacts: tuple[ArtifactRef, ...],
) -> ModelManifest:
    """Write a manifest into ``run_dir`` and return the typed form."""
    run_dir.mkdir(exist_ok=True)
    manifest = _manifest(artifacts)
    write_manifest(manifest, run_dir / "manifest.json")
    return manifest


def _importance_ref(filename: str) -> ArtifactRef:
    return ArtifactRef(
        logical_name="feature_importances",
        filename=filename,
    )


def test_returns_declared_frame_unchanged(tmp_path: Path) -> None:
    frame = _valid_importances()
    frame.write_parquet(tmp_path / "feature_importances.parquet")
    manifest = _staged_run(
        tmp_path,
        (
            _importance_ref("feature_importances.parquet"),
            _importance_ref("feature_importances.csv"),
        ),
    )

    out = load_feature_importances(manifest, tmp_path)

    assert out.equals(frame)


def test_no_declared_parquet_raises(tmp_path: Path) -> None:
    # The producer also writes a CSV under the same logical name; a reader
    # that pins the parquet format must reject a CSV-only declaration.
    manifest = _staged_run(
        tmp_path,
        (_importance_ref("feature_importances.csv"),),
    )

    with pytest.raises(ValueError, match="feature_importances"):
        load_feature_importances(manifest, tmp_path)


def test_two_declared_parquets_raise(tmp_path: Path) -> None:
    manifest = _staged_run(
        tmp_path,
        (
            _importance_ref("feature_importances.parquet"),
            _importance_ref("feature_importances_copy.parquet"),
        ),
    )

    with pytest.raises(ValueError, match="feature_importances"):
        load_feature_importances(manifest, tmp_path)


def test_declared_but_absent_file_raises(tmp_path: Path) -> None:
    manifest = _staged_run(
        tmp_path,
        (_importance_ref("feature_importances.parquet"),),
    )

    with pytest.raises(FileNotFoundError, match="feature_importances"):
        load_feature_importances(manifest, tmp_path)


def test_malformed_table_raises_boundary_error(tmp_path: Path) -> None:
    # A non-finite importance violates the schema's finiteness contract.
    malformed = _valid_importances().with_columns(
        pl.when(pl.col(FEATURE_COLNAME) == "feature_1")
        .then(pl.lit(float("nan")))
        .otherwise(pl.col(FEATURE_IMPORTANCE_COLNAME))
        .alias(FEATURE_IMPORTANCE_COLNAME)
    )
    malformed.write_parquet(tmp_path / "feature_importances.parquet")
    manifest = _staged_run(
        tmp_path,
        (_importance_ref("feature_importances.parquet"),),
    )

    with pytest.raises(BoundaryError) as excinfo:
        load_feature_importances(manifest, tmp_path)

    assert S04_FEATURE_IMPORTANCES_BOUNDARY in str(excinfo.value)
    assert excinfo.value.rule == "finite"
