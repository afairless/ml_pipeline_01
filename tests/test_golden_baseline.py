"""Integration tests for the committed golden baseline.

The baseline is a reviewable record of the demo's deterministic, text-valued
artifacts. These tests answer two questions a pull request must be able to
answer without running the pipeline: is the committed tree complete and
restricted to the declared artifact families, and does a fresh demo run
reproduce it byte for byte?

The pipeline is expensive, so the fresh run is a session-scoped fixture and is
shared with :mod:`tests.test_golden_diff`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.golden import (
    ARTIFACT_ALIASES,
    BASELINE_OUTPUT_ROOT,
    GOLDEN_ARTIFACT_ALIASES,
    artifact_alias,
    diff_golden_tree,
    iter_relative_files,
)

ALLOWED_SUFFIXES = {".json", ".txt", ".csv"}
"""The baseline records text artifacts only; parquet, PNG, and models are out."""

EXPECTED_ALIASES = {"manifest", "metrics", "statistics", "display_table"}
"""Every declared artifact family must be represented in a complete baseline."""


def _baseline_relative_paths() -> tuple[Path, ...]:
    """Return the committed baseline's output-relative files in stable order."""
    return tuple(iter_relative_files(BASELINE_OUTPUT_ROOT))


def test_committed_baseline_exists_and_is_not_empty() -> None:
    """The baseline tree is committed, not captured at test time."""
    assert BASELINE_OUTPUT_ROOT.is_dir()
    relative_paths = _baseline_relative_paths()
    assert relative_paths, "the golden baseline contains no artifacts"
    assert len(relative_paths) == len(set(relative_paths))


def test_baseline_covers_every_declared_artifact_family() -> None:
    """Manifests, metrics, statistics, and display CSVs are all recorded."""
    present = {
        alias.name
        for alias in (artifact_alias(path) for path in _baseline_relative_paths())
        if alias is not None
    }
    assert present == EXPECTED_ALIASES
    assert EXPECTED_ALIASES == {alias.name for alias in GOLDEN_ARTIFACT_ALIASES}
    assert set(ARTIFACT_ALIASES) == EXPECTED_ALIASES


def test_baseline_contains_only_declared_text_artifacts() -> None:
    """Parquet tables, figures, and estimator serializations never enter it."""
    for relative_path in _baseline_relative_paths():
        alias = artifact_alias(relative_path)
        assert alias is not None, f"undeclared baseline artifact: {relative_path}"
        assert relative_path.suffix in ALLOWED_SUFFIXES, (
            f"baseline artifact has a non-text suffix: {relative_path}"
        )


@pytest.mark.parametrize("relative_path", _baseline_relative_paths(), ids=str)
def test_committed_baseline_artifacts_are_complete(relative_path: Path) -> None:
    """A recorded artifact is parseable or non-empty, never a truncated stub."""
    path = BASELINE_OUTPUT_ROOT / relative_path
    assert path.is_file()
    payload = path.read_bytes()
    assert payload, f"baseline artifact is empty: {relative_path}"

    if relative_path.suffix == ".json":
        parsed = json.loads(payload)
        assert isinstance(parsed, dict) and parsed
    elif relative_path.suffix == ".csv":
        lines = payload.decode("utf-8").splitlines()
        assert len(lines) >= 2
        assert len(lines[0].split(",")) > 1
    else:
        assert payload.decode("utf-8").strip()


def test_a_fresh_demo_run_reproduces_the_committed_baseline(
    golden_output_root: Path,
) -> None:
    """The pipeline's deterministic output matches the committed tree exactly."""
    result = diff_golden_tree(BASELINE_OUTPUT_ROOT, golden_output_root)

    selected = tuple(
        path
        for path in iter_relative_files(golden_output_root)
        if artifact_alias(path) is not None
    )

    assert result.is_empty, result.render()
    assert selected
    assert selected == _baseline_relative_paths()
