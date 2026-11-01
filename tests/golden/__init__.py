"""Golden-baseline harness for the demo pipeline.

The harness has five small parts:

``paths``
    Locate the repository, the live ``output/`` tree, and the committed
    baseline tree. All paths are derived from this package's own location, so
    the harness works from any current working directory.
``aliases``
    Declare which output artifacts belong in the baseline. The baseline keeps
    only manifests, metrics JSON, statistics text, and display CSVs; parquet
    tables, figures, and estimator serializations are deliberately excluded.
``capture``
    Copy the selected artifacts from a live output tree into a baseline tree.
``diff``
    Compare a live output tree with a baseline tree byte for byte and report
    every added, removed, or changed artifact.
``runner``
    Seed a temporary project root with the committed demo source table and run
    the same explicit-plus-registry sequence as ``src/main.py``.

The public helpers are re-exported here so a test or a maintenance command can
import them from :mod:`tests.golden` without knowing the module split.
"""

from __future__ import annotations

from tests.golden.aliases import (
    ARTIFACT_ALIASES,
    GOLDEN_ARTIFACT_ALIASES,
    ArtifactAlias,
    artifact_alias,
    is_golden_artifact,
    select_golden_artifacts,
)
from tests.golden.capture import (
    CapturedArtifact,
    CaptureError,
    capture_baseline,
    capture_golden_tree,
    collect_golden_artifacts,
)
from tests.golden.diff import (
    DiffEntry,
    DiffResult,
    compare_trees,
    diff_baseline,
    diff_directories,
    diff_golden_tree,
)
from tests.golden.paths import (
    BASELINE_OUTPUT_ROOT,
    BASELINE_ROOT,
    GOLDEN_PACKAGE_ROOT,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    iter_files,
    iter_relative_files,
    relative_output_path,
)
from tests.golden.runner import (
    DEMO_RNG_SEED,
    SOURCE_FILENAME,
    run_demo_pipeline,
    seed_demo_source,
    temporary_demo_run,
)

__all__ = [
    "ARTIFACT_ALIASES",
    "BASELINE_OUTPUT_ROOT",
    "BASELINE_ROOT",
    "DEMO_RNG_SEED",
    "GOLDEN_ARTIFACT_ALIASES",
    "GOLDEN_PACKAGE_ROOT",
    "OUTPUT_ROOT",
    "PROJECT_ROOT",
    "SOURCE_FILENAME",
    "ArtifactAlias",
    "CaptureError",
    "CapturedArtifact",
    "DiffEntry",
    "DiffResult",
    "artifact_alias",
    "capture_baseline",
    "capture_golden_tree",
    "collect_golden_artifacts",
    "compare_trees",
    "diff_baseline",
    "diff_directories",
    "diff_golden_tree",
    "is_golden_artifact",
    "iter_files",
    "iter_relative_files",
    "relative_output_path",
    "run_demo_pipeline",
    "seed_demo_source",
    "select_golden_artifacts",
    "temporary_demo_run",
]
