"""Assemble the pipeline feature table from the external source table.

This stage is deliberately thin. It resolves its output directory through the
declarative layout, ingests the external source table through the input
registry, reads the key columns of the outcome table written by
``s01_outcome``, delegates the actual feature construction to
:func:`src.project.adapters.build_feature_table`, asserts that the feature keys
equal the outcome keys exactly (same set, same order), validates the result at
the ``S02_FEATURE_SCHEMA`` boundary, and writes the three artifacts the stage
table declares: the feature parquet, a head CSV preview, and the feature
manifest.

``src/project/adapters.py`` is the replacement point for a new project.
Replace the ``build_feature_table`` body with real feature derivation; this
module and every other stage keep working unchanged as long as the adapter
returns the canonical key columns plus the registered feature columns, one row
per source row, sorted by the keys.

The key-alignment assertion is the stage's contribution to the alignment
contract: a later stage joins the outcome column and the features on the
canonical key columns, so a feature table with a missing, extra, or reordered
key row would silently drop or mispair rows. The stage therefore fails with
the mismatching key count or sample instead of writing an artifact that the
divide stage cannot join.

The stage separates its three responsibilities explicitly: ingestion reads and
validates the source and outcome tables without mutating them; transformation
is the adapter call plus the alignment assertion; output serializes the
validated frame, its head preview, and the manifest that declares both. Every
artifact is written through a temporary sibling and moved into place, so a
re-run replaces only this stage's declared files and a failed write never
leaves a partial artifact behind.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import polars as pl

from src.core.manifest import (
    FEATURE_SCHEMA_VERSION,
    ArtifactRef,
    FeatureManifest,
    write_manifest,
)
from src.core.paths import PipelineContext, PipelineLayout
from src.core.splits import PARTITION_ORDER
from src.project.adapters import build_feature_table
from src.project.columns import KEY_COLUMNS
from src.project.config import ProjectConfig
from src.project.inputs import read_input_table
from src.project.schemas import (
    FEATURE_COLUMNS,
    S01_OUTCOME_SCHEMA,
    S02_FEATURE_SCHEMA,
    validate,
)

STAGE_NAME = "s02_feature"
"""Stage-table key whose directory and artifact filenames this module uses."""

SOURCE_STAGE = "s01_outcome"
"""Stage whose declared outcome parquet supplies the key columns."""

SOURCE_TABLE = "source_table_1"
"""Logical name of the external input table this stage ingests."""

OUTCOME_BOUNDARY_LABEL = "s02 input outcome table"
"""Boundary label used when the s01 outcome artifact violates its schema."""

FEATURE_BOUNDARY_LABEL = "s02 feature table"
"""Boundary label used when the adapter result violates its schema."""

PARQUET_COMPRESSION = "zstd"
"""Compression codec of the feature parquet artifact."""

HEAD_ROW_COUNT = 5
"""Number of leading rows written to the head CSV preview."""


def _key_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Return only the canonical key columns, in their declared order."""
    return frame.select(list(KEY_COLUMNS))


def assert_key_alignment(
    feature_df: pl.DataFrame,
    outcome_df: pl.DataFrame,
) -> None:
    """Assert the feature keys equal the outcome keys exactly.

    The two frames must carry the same ``(case_id, case_date)`` rows in the
    same order. Row order matters: the split stage joins the features and the
    outcome column positionally by the split predicates, and the manifest
    records the table as written, so a reordered but otherwise equal key frame
    is a producer bug rather than an acceptable variant. Dtypes are compared
    too, because a join on a re-typed key would not match.

    Args:
        feature_df: The feature frame returned by the project adapter.
        outcome_df: The validated outcome frame written by ``s01_outcome``.

    Raises:
        ValueError: The key frames differ in row count, key set, key dtypes, or
            row order. The message names the mismatch instead of surfacing an
            anonymous assertion failure.
    """
    feature_keys = _key_frame(feature_df)
    outcome_keys = _key_frame(outcome_df)
    if feature_keys.equals(outcome_keys):
        return

    if feature_df.height != outcome_df.height:
        raise ValueError(
            "feature table is not key-aligned with the outcome table: the "
            f"feature table has {feature_df.height} rows but the outcome "
            f"table has {outcome_df.height}; the adapter must return exactly "
            "one feature row per outcome row"
        )

    feature_rows = set(feature_keys.iter_rows())
    outcome_rows = set(outcome_keys.iter_rows())
    if feature_rows != outcome_rows:
        missing = sorted(outcome_rows - feature_rows)
        unexpected = sorted(feature_rows - outcome_rows)
        raise ValueError(
            "feature table keys differ from the outcome table keys: "
            f"{len(missing)} outcome key(s) are missing from the feature "
            f"table (first: {missing[:3]!r}) and {len(unexpected)} feature "
            f"key(s) are absent from the outcome table (first: "
            f"{unexpected[:3]!r}); the adapter must preserve the key set"
        )

    raise ValueError(
        "feature table carries the same key set as the outcome table but in a "
        "different order; both tables must be sorted by "
        f"{list(KEY_COLUMNS)!r} so the divide stage can join them positionally"
    )


def _declared_filename(layout: PipelineLayout, logical: str) -> str:
    """Return the on-disk filename a stage artifact is declared under.

    The manifest is colocated with its artifacts, so it stores the basename of
    the layout declaration rather than a path relative to the output root.
    """
    return Path(layout.artifact(STAGE_NAME, logical)).name


def build_feature_manifest(
    feature_df: pl.DataFrame,
    layout: PipelineLayout,
) -> FeatureManifest:
    """Describe an assembled feature table as a ``FeatureManifest``.

    The manifest attributes every column of the written table to one of four
    disjoint sections: the canonical key columns, the registered model feature
    columns, categorical feature columns, and metadata columns. The demo table
    carries keys plus numeric features only, so the categorical section is
    empty and any remaining producer column is recorded as metadata rather
    than silently dropped from the description.

    The declared partition names are the canonical holdout partitions: the
    feature table covers every source row, so each partition's predicate
    selects a non-empty subset of it. The artifact inventory names the parquet
    and the head CSV by the filenames the stage table declares.

    Args:
        feature_df: The validated feature frame about to be written.
        layout: The layout whose stage table declares the artifact filenames.

    Returns:
        A frozen manifest recording ``feature_df``'s shape, columns, and
        artifact inventory. The manifest carries no wall clock and no absolute
        path, so it is deterministic and golden-comparable.
    """
    metadata_columns = tuple(
        name
        for name in feature_df.columns
        if name not in KEY_COLUMNS and name not in FEATURE_COLUMNS
    )
    return FeatureManifest(
        schema_version=FEATURE_SCHEMA_VERSION,
        manifest_kind="feature",
        key_columns=KEY_COLUMNS,
        n_rows=feature_df.height,
        partitions=PARTITION_ORDER,
        feature_columns=FEATURE_COLUMNS,
        categorical_columns=(),
        metadata_columns=metadata_columns,
        artifacts=(
            ArtifactRef(
                logical_name="feature_df",
                filename=_declared_filename(layout, "feature_df"),
            ),
            ArtifactRef(
                logical_name="feature_head_df",
                filename=_declared_filename(layout, "feature_head_df"),
            ),
        ),
    )


def _write_atomically(path: Path, writer: Callable[[Path], object]) -> None:
    """Write one artifact through a temporary sibling and an atomic replace.

    The temporary file lives in the destination directory, so the final
    :func:`os.replace` never crosses a filesystem boundary. A failed write
    removes its temporary file and leaves the previous artifact untouched.

    Args:
        path: Destination artifact path whose parent already exists.
        writer: Callback that serializes the artifact to the temporary path.
    """
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        writer(temporary_path)
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_feature_parquet(path: Path, feature_df: pl.DataFrame) -> None:
    """Serialize the feature frame with explicit deterministic writer settings."""
    _write_atomically(
        path,
        lambda target: feature_df.write_parquet(
            target,
            compression=PARQUET_COMPRESSION,
            statistics=True,
        ),
    )


def _write_feature_head_csv(path: Path, feature_df: pl.DataFrame) -> None:
    """Serialize the leading rows of the feature frame as CSV."""
    _write_atomically(
        path,
        lambda target: feature_df.head(HEAD_ROW_COUNT).write_csv(target),
    )


def _write_feature_manifest(
    path: Path,
    feature_df: pl.DataFrame,
    layout: PipelineLayout,
) -> None:
    """Serialize the feature manifest as deterministic sorted-key JSON."""
    manifest = build_feature_manifest(feature_df, layout)
    _write_atomically(path, lambda target: write_manifest(manifest, target))


def _read_outcome_table(context: PipelineContext, output_root: Path) -> pl.DataFrame:
    """Read and validate the s01 outcome artifact.

    Only the key columns are consumed by this stage, but the whole artifact is
    validated so a corrupted prerequisite fails here with the outcome boundary
    label instead of during the later key comparison.

    Raises:
        FileNotFoundError: The prerequisite stage has not produced its outcome
            parquet under ``output_root``.
        BoundaryError: The artifact violates
            :data:`~src.project.schemas.S01_OUTCOME_SCHEMA`.
    """
    outcome_path = Path(output_root) / context.layout.artifact(
        SOURCE_STAGE,
        "outcome_df",
    )
    outcome_df = pl.read_parquet(outcome_path)
    return validate(outcome_df, S01_OUTCOME_SCHEMA, OUTCOME_BOUNDARY_LABEL)


def _source_data_dir(
    context: PipelineContext,
    output_root: Path,
    project_root: Path | None,
) -> Path:
    """Resolve the external data directory for this stage's run.

    ``output_root`` is the pipeline output root, normally
    ``<project_root>/output``, so its parent is the project root that a
    relative ``config.data_root`` is anchored to. A caller whose output root is
    elsewhere passes ``project_root`` explicitly instead.
    """
    root = Path(output_root).parent if project_root is None else Path(project_root)
    data_root = Path(context.config.data_root)
    return data_root if data_root.is_absolute() else root / data_root


def main(
    context: PipelineContext,
    output_root: Path,
    *,
    project_root: Path | None = None,
) -> None:
    """Run the feature stage once.

    Reads ``source_table_1`` from the configured data directory and the s01
    outcome artifact from ``output_root``, builds the feature frame through the
    project adapter, asserts key alignment against the outcome table, validates
    the result against :data:`~src.project.schemas.S02_FEATURE_SCHEMA` with the
    source row count as the expected count, and writes the declared parquet,
    head CSV, and manifest under ``output_root/s02_feature``.

    Args:
        context: The frozen pipeline context carrying the layout and config.
        output_root: The explicit output root the relative layout paths are
            joined with.
        project_root: Optional explicit root that a relative
            ``config.data_root`` is resolved against. Defaults to the parent of
            ``output_root``, matching the pipeline's
            ``<project_root>/output`` convention.

    Raises:
        FileNotFoundError: The source table or the s01 outcome parquet cannot
            be found.
        ValueError: Input discovery is ambiguous, the adapter result is not
            key-aligned with the outcome table, or the feature table has the
            wrong row count.
        BoundaryError: The source table, the outcome artifact, or the adapter
            result violates its declared boundary schema.
    """
    layout = context.layout
    stage_dir = Path(output_root) / layout.stage_dir(STAGE_NAME)
    context.ensure(stage_dir)

    data_dir = _source_data_dir(context, output_root, project_root)
    source_df = read_input_table(SOURCE_TABLE, data_dir)
    outcome_df = _read_outcome_table(context, output_root)

    feature_df = build_feature_table(
        source_df,
        outcome_df,
        cast(ProjectConfig, context.config),
    )
    assert_key_alignment(feature_df, outcome_df)
    validate(
        feature_df,
        S02_FEATURE_SCHEMA,
        FEATURE_BOUNDARY_LABEL,
        expected_row_count=source_df.height,
    )

    feature_path = Path(output_root) / layout.artifact(STAGE_NAME, "feature_df")
    head_path = Path(output_root) / layout.artifact(STAGE_NAME, "feature_head_df")
    manifest_path = Path(output_root) / layout.artifact(
        STAGE_NAME,
        "feature_manifest",
    )

    _write_feature_parquet(feature_path, feature_df)
    _write_feature_head_csv(head_path, feature_df)
    _write_feature_manifest(manifest_path, feature_df, layout)


s02_feature = main
"""Stage entry point under the name used by the explicit pipeline sequence."""

__all__ = [
    "FEATURE_BOUNDARY_LABEL",
    "OUTCOME_BOUNDARY_LABEL",
    "assert_key_alignment",
    "build_feature_manifest",
    "main",
    "s02_feature",
]
