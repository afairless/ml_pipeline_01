"""Run the demo pipeline into a temporary project root for golden captures.

The runner mirrors the entry point's sequence exactly: seed the committed demo
source table, call the three explicit preprocessing stages, then execute the
registry-driven stages. It never writes into the repository's live ``output/``
tree, so a golden test can run repeatedly without touching local artifacts.

The source table is copied into ``<project_root>/data/demo/`` and the explicit
stages anchor their relative ``data_root`` to the output root's parent, so the
temporary run needs no environment-specific absolute paths.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from src.project.config import CONFIG
from src.project.pipeline import build_context, run_pipeline
from src.stages.s01_outcome import s01_outcome
from src.stages.s01b_outcome_analysis import s01b_outcome_analysis
from src.stages.s02_feature import s02_feature
from tests.golden.paths import PROJECT_ROOT

SOURCE_FILENAME = "source_table_1.parquet"
"""Committed demo input filename copied into each temporary project root."""

DEMO_RNG_SEED = CONFIG.demo_seed
"""Seed applied to the global ``numpy`` RNG before each demo run.

The feature policies draw their substitutions from the global ``numpy`` RNG,
so a run is reproducible only when that RNG is seeded immediately before the
first draw. The test session seeds it once as well; seeding here makes a
standalone capture or a maintenance command equally reproducible.
"""


def seed_demo_source(
    project_root: Path,
    source_project_root: Path = PROJECT_ROOT,
) -> Path:
    """Copy the committed demo source table into ``project_root``.

    Args:
        project_root: Temporary project root to seed.
        source_project_root: Root holding the committed ``data/demo`` table.

    Returns:
        The copied source table's path.

    Raises:
        FileNotFoundError: The committed source table is absent.
    """
    source = Path(source_project_root) / CONFIG.data_root / SOURCE_FILENAME
    if not source.is_file():
        raise FileNotFoundError(f"committed demo source table is missing: {source}")

    destination = Path(project_root) / CONFIG.data_root / SOURCE_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    return destination


def run_demo_pipeline(project_root: Path) -> Path:
    """Run the full demo pipeline under an explicit project root.

    Args:
        project_root: Root that owns the ``data/`` input and ``output/`` tree.
            It is seeded with the committed demo source table first.

    Returns:
        The run's output root, ``<project_root>/output``.
    """
    root = Path(project_root)
    seed_demo_source(root)
    np.random.seed(DEMO_RNG_SEED)
    context = build_context(config=CONFIG, project_root=root)
    output_root = root / "output"
    s01_outcome(context, output_root)
    s01b_outcome_analysis(context, output_root)
    s02_feature(context, output_root)
    run_pipeline(context, output_root)
    return output_root


@contextmanager
def temporary_demo_run() -> Iterator[Path]:
    """Yield an output root from a full run in a temporary project root.

    The temporary project root (including its output tree) is removed when the
    context exits.
    """
    with tempfile.TemporaryDirectory(prefix="golden-demo-") as temporary:
        yield run_demo_pipeline(Path(temporary))


run_pipeline_to_output = run_demo_pipeline
"""Alias for callers that read the function as an output-producing runner."""


__all__ = [
    "DEMO_RNG_SEED",
    "SOURCE_FILENAME",
    "run_demo_pipeline",
    "run_pipeline_to_output",
    "seed_demo_source",
    "temporary_demo_run",
]
