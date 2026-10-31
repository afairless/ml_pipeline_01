"""Integration tests for the ``s02_feature`` stage.

The stage is the second thin entry point in the pipeline: it ingests the
external source table through the input registry, reads the key columns of the
s01 outcome artifact, delegates feature construction to
``src/project/adapters.py``, asserts key alignment against the outcome table,
validates the result at the feature boundary, and writes the three declared
artifacts. These tests exercise that whole path on a temporary project root
seeded with the committed demo table, so they pin the on-disk contract rather
than an in-memory helper.

The assertions cover the artifact set and filenames, the written frame and its
schema, the head CSV preview, the feature manifest round-trip and its column
sections, deterministic re-runs, the explicit ``project_root`` seam for an
output root outside the project tree, and the failure cases: an adapter that
drops a row, reorders the keys, or changes the key set must fail with an
alignment error, while a non-finite feature must fail at the feature boundary
with the stage's label and leave no partial artifact behind.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import polars as pl
import pytest

import src.stages.s02_feature as stage_module
from src.core.manifest import (
    FEATURE_SCHEMA_VERSION,
    FeatureManifest,
    declared_artifact_path,
    read_manifest,
)
from src.core.paths import PipelineContext
from src.core.splits import PARTITION_ORDER
from src.project.adapters import build_feature_table, build_outcome_table
from src.project.columns import CASE_DATE, CASE_ID, KEY_COLUMNS
from src.project.config import CONFIG, ProjectConfig
from src.project.inputs import read_input_table
from src.project.pipeline import build_context
from src.project.schemas import (
    FEATURE_COLUMNS,
    S01_OUTCOME_SCHEMA,
    S02_FEATURE_SCHEMA,
    BoundaryError,
    validate,
)
from src.stages.s01_outcome import s01_outcome
from src.stages.s02_feature import (
    FEATURE_BOUNDARY_LABEL,
    HEAD_ROW_COUNT,
    assert_key_alignment,
    build_feature_manifest,
    main,
    s02_feature,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Repository root, used only to locate the committed demo input table."""

SOURCE_FILENAME = "source_table_1.parquet"
"""Committed demo filename, copied into each temporary project root."""

FEATURE_TABLE_COLUMNS = (CASE_ID, CASE_DATE, *FEATURE_COLUMNS)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Return a temporary project root containing the demo source table."""
    data_dir = tmp_path / CONFIG.data_root
    data_dir.mkdir(parents=True)
    shutil.copyfile(
        PROJECT_ROOT / CONFIG.data_root / SOURCE_FILENAME,
        data_dir / SOURCE_FILENAME,
    )
    return tmp_path


@pytest.fixture
def context(project_root: Path) -> PipelineContext:
    """Return the pipeline context bound to the temporary project root."""
    return build_context(config=CONFIG, project_root=project_root)


def _seed_outcome(
    context: PipelineContext,
    output_root: Path,
    project_root: Path | None = None,
) -> None:
    """Run ``s01_outcome`` so the feature stage has its key source."""
    s01_outcome(context, output_root, project_root=project_root)


def _run(
    project_root: Path,
    context: PipelineContext,
    *,
    output_root: Path | None = None,
    explicit_project_root: bool = False,
) -> tuple[Path, Path, Path]:
    """Seed the outcome artifact, run the feature stage, return its artifacts."""
    resolved_output = project_root / "output" if output_root is None else output_root
    explicit_root = project_root if explicit_project_root else None
    _seed_outcome(context, resolved_output, explicit_root)
    main(context, resolved_output, project_root=explicit_root)
    stage_dir = resolved_output / "s02_feature"
    return (
        stage_dir / "feature_df.parquet",
        stage_dir / "feature_df_head.csv",
        stage_dir / "feature_manifest.json",
    )


def _source_df(project_root: Path) -> pl.DataFrame:
    """Read the temporary project's demo source table."""
    return read_input_table("source_table_1", project_root / CONFIG.data_root)


def _expected_outcome_df(project_root: Path) -> pl.DataFrame:
    """Rebuild the outcome frame through the same public path."""
    return build_outcome_table(_source_df(project_root), CONFIG)


def _expected_feature_df(project_root: Path) -> pl.DataFrame:
    """Rebuild the expected feature frame through the same public path."""
    source_df = _source_df(project_root)
    return build_feature_table(
        source_df,
        build_outcome_table(source_df, CONFIG),
        CONFIG,
    )


# --- successful stage run --------------------------------------------------


def test_stage_writes_exactly_the_declared_artifacts(
    project_root: Path,
    context: PipelineContext,
) -> None:
    feature_path, head_path, manifest_path = _run(project_root, context)

    assert sorted(path.name for path in feature_path.parent.iterdir()) == [
        "feature_df.parquet",
        "feature_df_head.csv",
        "feature_manifest.json",
    ]
    assert feature_path.is_file()
    assert head_path.is_file()
    assert manifest_path.is_file()
    assert manifest_path.stat().st_size > 0


def test_stage_writes_the_adapter_feature_table(
    project_root: Path,
    context: PipelineContext,
) -> None:
    feature_path, _, _ = _run(project_root, context)

    written = pl.read_parquet(feature_path)
    expected = _expected_feature_df(project_root)

    assert written.equals(expected)
    assert written.height == CONFIG.demo_case_count
    assert tuple(written.columns) == FEATURE_TABLE_COLUMNS
    assert (
        validate(
            written,
            S02_FEATURE_SCHEMA,
            "written feature table",
            expected_row_count=written.height,
        )
        is written
    )


def test_stage_writes_a_head_csv_preview(
    project_root: Path,
    context: PipelineContext,
) -> None:
    feature_path, head_path, _ = _run(project_root, context)

    written = pl.read_parquet(feature_path)
    preview = pl.read_csv(head_path, try_parse_dates=True)

    assert preview.equals(written.head(HEAD_ROW_COUNT))
    assert preview.height == HEAD_ROW_COUNT
    assert tuple(preview.columns) == FEATURE_TABLE_COLUMNS


def test_stage_feature_keys_match_the_outcome_keys(
    project_root: Path,
    context: PipelineContext,
) -> None:
    feature_path, _, _ = _run(project_root, context)

    feature_df = pl.read_parquet(feature_path)
    outcome_df = _expected_outcome_df(project_root)

    assert (
        validate(outcome_df, S01_OUTCOME_SCHEMA, "expected outcome table") is outcome_df
    )
    assert_key_alignment(feature_df, outcome_df)
    assert feature_df.select(list(KEY_COLUMNS)).equals(
        outcome_df.select(list(KEY_COLUMNS))
    )


# --- feature manifest ------------------------------------------------------


def test_stage_writes_a_round_tripping_feature_manifest(
    project_root: Path,
    context: PipelineContext,
) -> None:
    feature_path, _, manifest_path = _run(project_root, context)

    loaded = read_manifest(manifest_path)
    expected = build_feature_manifest(pl.read_parquet(feature_path), context.layout)

    assert isinstance(loaded, FeatureManifest)
    assert loaded == expected
    assert loaded.schema_version == FEATURE_SCHEMA_VERSION
    assert loaded.manifest_kind == "feature"
    assert loaded.key_columns == KEY_COLUMNS
    assert loaded.n_rows == CONFIG.demo_case_count
    assert loaded.partitions == PARTITION_ORDER
    assert loaded.feature_columns == FEATURE_COLUMNS


def test_feature_manifest_declares_the_written_artifacts(
    project_root: Path,
    context: PipelineContext,
) -> None:
    feature_path, head_path, manifest_path = _run(project_root, context)

    loaded = read_manifest(manifest_path)
    assert isinstance(loaded, FeatureManifest)
    declared = {
        declared_artifact_path(loaded, manifest_path.parent, logical, suffix)
        for logical, suffix in (
            ("feature_df", ".parquet"),
            ("feature_head_df", ".csv"),
        )
    }

    assert declared == {feature_path, head_path}
    assert all(path.is_file() for path in declared)
    assert sorted(
        {ref.filename for ref in loaded.artifacts} | {manifest_path.name}
    ) == sorted(path.name for path in manifest_path.parent.iterdir())


def test_feature_manifest_sections_partition_every_column(
    project_root: Path,
    context: PipelineContext,
) -> None:
    feature_path, _, manifest_path = _run(project_root, context)

    written = pl.read_parquet(feature_path)
    loaded = read_manifest(manifest_path)
    assert isinstance(loaded, FeatureManifest)
    sections = (
        loaded.key_columns,
        loaded.feature_columns,
        loaded.categorical_columns,
        loaded.metadata_columns,
    )
    declared_columns = [name for section in sections for name in section]

    assert len(declared_columns) == len(set(declared_columns))
    assert set(declared_columns) == set(written.columns)
    assert loaded.feature_columns == FEATURE_COLUMNS
    assert loaded.categorical_columns == ()
    assert loaded.metadata_columns == ()


def test_build_feature_manifest_records_extra_columns_as_metadata(
    context: PipelineContext,
) -> None:
    frame = pl.DataFrame(
        {
            CASE_ID: ["case_a"],
            CASE_DATE: pl.Series([date(2024, 1, 1)], dtype=pl.Date),
            **{name: [1.0] for name in FEATURE_COLUMNS},
            "case_group": ["group_a"],
        }
    )

    manifest = build_feature_manifest(frame, context.layout)

    assert manifest.metadata_columns == ("case_group",)
    assert manifest.feature_columns == FEATURE_COLUMNS
    assert manifest.n_rows == 1


# --- determinism and path seams --------------------------------------------


def test_rerunning_the_stage_replaces_the_same_artifacts(
    project_root: Path,
    context: PipelineContext,
) -> None:
    paths = _run(project_root, context)
    first_bytes = tuple(path.read_bytes() for path in paths)

    rerun_paths = _run(project_root, context)

    assert rerun_paths == paths
    assert tuple(path.read_bytes() for path in rerun_paths) == first_bytes
    assert sorted(path.name for path in paths[0].parent.iterdir()) == [
        "feature_df.parquet",
        "feature_df_head.csv",
        "feature_manifest.json",
    ]


def test_explicit_project_root_resolves_the_data_directory(
    project_root: Path,
    context: PipelineContext,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    output_root = tmp_path_factory.mktemp("elsewhere") / "output"

    feature_path, _, _ = _run(
        project_root,
        context,
        output_root=output_root,
        explicit_project_root=True,
    )

    assert pl.read_parquet(feature_path).equals(_expected_feature_df(project_root))


def test_stage_entry_point_is_exported_under_its_stage_name() -> None:
    assert s02_feature is main


# --- boundary and alignment failures ---------------------------------------


def test_missing_outcome_artifact_fails_before_writing(
    project_root: Path,
    context: PipelineContext,
) -> None:
    output_root = project_root / "output"

    with pytest.raises(FileNotFoundError):
        main(context, output_root)

    stage_dir = output_root / "s02_feature"
    assert not any(stage_dir.iterdir())


def test_row_dropping_adapter_result_fails_the_alignment_assertion(
    project_root: Path,
    context: PipelineContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def truncated(
        source_df: pl.DataFrame,
        outcome_df: pl.DataFrame,
        config: ProjectConfig,
    ) -> pl.DataFrame:
        return build_feature_table(source_df, outcome_df, config).head(
            source_df.height - 1
        )

    monkeypatch.setattr(stage_module, "build_feature_table", truncated)

    with pytest.raises(ValueError, match="not key-aligned") as caught:
        _run(project_root, context)

    assert "feature table has" in str(caught.value)
    assert str(CONFIG.demo_case_count - 1) in str(caught.value)


def test_reordered_adapter_keys_fail_the_alignment_assertion(
    project_root: Path,
    context: PipelineContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reordered(
        source_df: pl.DataFrame,
        outcome_df: pl.DataFrame,
        config: ProjectConfig,
    ) -> pl.DataFrame:
        return build_feature_table(source_df, outcome_df, config).sort(
            list(KEY_COLUMNS),
            descending=True,
        )

    monkeypatch.setattr(stage_module, "build_feature_table", reordered)

    with pytest.raises(ValueError, match="different order"):
        _run(project_root, context)


def test_changed_adapter_keys_fail_the_alignment_assertion(
    project_root: Path,
    context: PipelineContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def renamed(
        source_df: pl.DataFrame,
        outcome_df: pl.DataFrame,
        config: ProjectConfig,
    ) -> pl.DataFrame:
        frame = build_feature_table(source_df, outcome_df, config)
        return (
            frame.with_row_index()
            .with_columns(
                pl.when(pl.col("index") == 0)
                .then(pl.lit("case_unknown"))
                .otherwise(pl.col(CASE_ID))
                .alias(CASE_ID)
            )
            .drop("index")
        )

    monkeypatch.setattr(stage_module, "build_feature_table", renamed)

    with pytest.raises(ValueError, match="keys differ") as caught:
        _run(project_root, context)

    assert "outcome key(s) are missing" in str(caught.value)
    assert "case_unknown" in str(caught.value)


def test_non_finite_adapter_result_fails_at_the_feature_boundary(
    project_root: Path,
    context: PipelineContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def malformed(
        source_df: pl.DataFrame,
        outcome_df: pl.DataFrame,
        config: ProjectConfig,
    ) -> pl.DataFrame:
        frame = build_feature_table(source_df, outcome_df, config)
        return frame.with_columns(pl.lit(float("nan")).alias("feature_1"))

    monkeypatch.setattr(stage_module, "build_feature_table", malformed)

    with pytest.raises(BoundaryError) as caught:
        _run(project_root, context)

    error = caught.value
    assert error.label == FEATURE_BOUNDARY_LABEL
    assert error.column == "feature_1"
    assert error.rule == "finite"
    assert FEATURE_BOUNDARY_LABEL in str(error)


def test_failed_boundary_validation_writes_no_artifact(
    project_root: Path,
    context: PipelineContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def malformed(
        source_df: pl.DataFrame,
        outcome_df: pl.DataFrame,
        config: ProjectConfig,
    ) -> pl.DataFrame:
        frame = build_feature_table(source_df, outcome_df, config)
        return frame.with_columns(pl.lit(float("inf")).alias("feature_2"))

    monkeypatch.setattr(stage_module, "build_feature_table", malformed)

    with pytest.raises(BoundaryError):
        _run(project_root, context)

    stage_dir = project_root / "output" / "s02_feature"
    assert not any(stage_dir.iterdir())
