"""Shared pytest configuration for the pipeline test suite.

The demo's feature policies draw from the global ``numpy`` RNG, and the
estimator library is imported by modules that a test module imports at
collection time. The RNG therefore has to be seeded once, before any test
imports the pipeline: seeding it inside an individual test would run after
those imports and would not be reproducible across test orders. The same
mechanism must run in every environment, so it lives here rather than in a
per-file fixture.

The seed is the project's single demo seed. Individual golden runs reseed the
global RNG before drawing, so a full-suite run and a standalone capture
produce identical artifacts.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from src.project.config import CONFIG

np.random.seed(CONFIG.demo_seed)
"""Seed the global RNG before the estimator and any stage module are imported."""


def _run_demo() -> Iterator[Path]:
    """Run the demo pipeline once in a temporary project root."""
    from tests.golden.runner import temporary_demo_run

    with temporary_demo_run() as output_root:
        yield output_root


@pytest.fixture(scope="session")
def golden_output_root() -> Iterator[Path]:
    """Run the demo pipeline once and yield its output tree.

    A full demo run writes a large tree, so the golden tests share one run per
    session. The temporary project root, including its output tree, is removed
    when the session ends.
    """
    yield from _run_demo()


@pytest.fixture(scope="session")
def second_golden_output_root() -> Iterator[Path]:
    """Yield the output tree of an independent second demo run.

    The two session runs are compared with each other and with the committed
    baseline, so a single accidental capture cannot hide nondeterminism.
    """
    yield from _run_demo()
