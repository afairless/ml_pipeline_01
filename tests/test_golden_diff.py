"""Integration tests for golden baseline capture and byte-exact diffing.

The harness unit tests in :mod:`tests.test_golden_harness` cover the mechanics
with synthetic trees. These tests exercise the same mechanics on a real demo
run: two independent runs must agree, a capture from a fresh run must equal the
committed baseline, and a reviewable change must be reported for exactly the
artifact that changed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tests.golden import (
    BASELINE_OUTPUT_ROOT,
    capture_golden_tree,
    diff_baseline,
    diff_golden_tree,
    iter_relative_files,
    select_golden_artifacts,
)

CHANGED_RELATIVE_PATH = Path("s02_feature/feature_manifest.json")
"""A stable, single-line JSON artifact used to prove a real change is detected."""


def _selected_paths(output_root: Path) -> tuple[Path, ...]:
    """Return the artifacts a golden comparison would consider."""
    return select_golden_artifacts(tuple(iter_relative_files(output_root)))


def test_two_independent_demo_runs_are_byte_identical(
    golden_output_root: Path,
    second_golden_output_root: Path,
) -> None:
    """The demo is deterministic across runs and temporary project roots."""
    result = diff_golden_tree(golden_output_root, second_golden_output_root)

    assert result.is_empty, result.render()
    assert _selected_paths(golden_output_root) == _selected_paths(
        second_golden_output_root
    )


def test_second_demo_run_diffs_cleanly_against_the_committed_baseline(
    second_golden_output_root: Path,
) -> None:
    """A run other than the capture run still matches the committed tree."""
    result = diff_baseline(second_golden_output_root)

    assert result.is_empty, result.render()


def test_recapture_from_a_fresh_run_reproduces_the_committed_baseline(
    second_golden_output_root: Path,
    tmp_path: Path,
) -> None:
    """Recapturing twice yields the same bytes as the reviewed baseline."""
    first_recapture = tmp_path / "first" / "output"
    second_recapture = tmp_path / "second" / "output"

    capture_golden_tree(second_golden_output_root, first_recapture)
    capture_golden_tree(second_golden_output_root, second_recapture)

    assert diff_golden_tree(first_recapture, second_recapture).is_empty
    result = diff_golden_tree(BASELINE_OUTPUT_ROOT, first_recapture)
    assert result.is_empty, result.render()


def test_diff_reports_a_changed_baseline_artifact(
    tmp_path: Path,
) -> None:
    """A modified artifact is reported by path, never silently tolerated."""
    mutated_root = tmp_path / "output"
    shutil.copytree(BASELINE_OUTPUT_ROOT, mutated_root)

    changed_path = mutated_root / CHANGED_RELATIVE_PATH
    original = changed_path.read_bytes()
    changed_path.write_bytes(original.replace(b'"', b'" ', 1))

    result = diff_golden_tree(BASELINE_OUTPUT_ROOT, mutated_root)

    assert not result.is_empty
    assert result.changed == (CHANGED_RELATIVE_PATH,)
    assert str(CHANGED_RELATIVE_PATH) in result.render()


def test_diff_reports_a_missing_baseline_artifact(
    tmp_path: Path,
) -> None:
    """Deleting a baseline artifact is an explicit removal, not an empty diff."""
    mutated_root = tmp_path / "output"
    shutil.copytree(BASELINE_OUTPUT_ROOT, mutated_root)
    (mutated_root / CHANGED_RELATIVE_PATH).unlink()

    result = diff_golden_tree(BASELINE_OUTPUT_ROOT, mutated_root)

    assert result.removed == (CHANGED_RELATIVE_PATH,)
    assert result.changed == ()
    assert result.added == ()


def test_diff_ignores_excluded_artifacts_from_a_real_run(
    golden_output_root: Path,
) -> None:
    """Parquet, figures, and estimator files are not part of the contract."""
    excluded = tuple(
        path
        for path in iter_relative_files(golden_output_root)
        if path not in _selected_paths(golden_output_root)
    )
    excluded_names = {path.name for path in excluded}

    assert {"model.json", "model_pretty.json", "model.txt"} <= excluded_names
    assert any(path.suffix == ".parquet" for path in excluded)
    assert any(path.suffix == ".png" for path in excluded)
    assert diff_golden_tree(BASELINE_OUTPUT_ROOT, golden_output_root).is_empty
