"""Capture the selected artifacts from a live output tree as a baseline.

Capture is a two-phase operation: the selected files are first copied into a
temporary staging directory beside the destination, then the staging directory
replaces the old baseline. A failure while walking or copying the source
therefore leaves the previous baseline untouched. The copy is byte-for-byte;
capture never parses or rewrites an artifact.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tests.golden.aliases import ArtifactAlias, artifact_alias
from tests.golden.paths import BASELINE_OUTPUT_ROOT, iter_relative_files


class CaptureError(RuntimeError):
    """Raised when a golden baseline cannot be captured safely."""


@dataclass(frozen=True, slots=True)
class CapturedArtifact:
    """One source artifact selected for the baseline.

    Attributes:
        relative_path: Artifact path relative to the output root.
        alias: The :class:`~tests.golden.aliases.ArtifactAlias` that selected
            it.
        source: Absolute path in the live output tree.
    """

    relative_path: Path
    alias: ArtifactAlias
    source: Path


def collect_golden_artifacts(output_root: Path) -> tuple[CapturedArtifact, ...]:
    """Return every baseline artifact under ``output_root`` in stable order.

    Raises:
        FileNotFoundError: ``output_root`` does not exist.
        NotADirectoryError: ``output_root`` is not a directory.
    """
    root = Path(output_root)
    if not root.exists():
        raise FileNotFoundError(f"output root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"output root is not a directory: {root}")

    captured: list[CapturedArtifact] = []
    for relative_path in iter_relative_files(root):
        alias = artifact_alias(relative_path)
        if alias is None:
            continue
        captured.append(
            CapturedArtifact(
                relative_path=relative_path,
                alias=alias,
                source=root / relative_path,
            )
        )
    return tuple(captured)


def _replace_directory(staging: Path, destination: Path) -> None:
    """Replace ``destination`` with the completed ``staging`` directory."""
    if destination.exists():
        if not destination.is_dir():
            raise CaptureError(
                f"baseline destination is not a directory: {destination}"
            )
        shutil.rmtree(destination)
    staging.replace(destination)


def capture_golden_tree(
    output_root: Path,
    baseline_output_root: Path = BASELINE_OUTPUT_ROOT,
) -> tuple[Path, ...]:
    """Copy the selected artifacts from ``output_root`` into a baseline tree.

    Args:
        output_root: Live output tree to capture.
        baseline_output_root: Destination tree, normally
            :data:`~tests.golden.paths.BASELINE_OUTPUT_ROOT`.

    Returns:
        The captured output-relative paths, in stable order.

    Raises:
        CaptureError: No baseline artifact was found, or the destination
            cannot be replaced safely.
        FileNotFoundError: ``output_root`` does not exist.
        NotADirectoryError: ``output_root`` is not a directory.
    """
    artifacts = collect_golden_artifacts(output_root)
    if not artifacts:
        raise CaptureError(
            f"no baseline artifacts found under {output_root}; expected "
            "manifests, metrics JSON, statistics text, or display CSVs"
        )

    destination = Path(baseline_output_root)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-",
        dir=destination.parent,
    ) as temporary:
        staging = Path(temporary) / destination.name
        for artifact in artifacts:
            target = staging / artifact.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact.source, target)
        _replace_directory(staging, destination)

    return tuple(artifact.relative_path for artifact in artifacts)


def capture_baseline(
    output_root: Path,
    baseline_root: Path | None = None,
) -> tuple[Path, ...]:
    """Capture a baseline tree from an output root.

    This is a convenience wrapper selecting ``<baseline_root>/output`` as the
    destination. Passing ``None`` uses the committed baseline location.
    """
    destination = (
        BASELINE_OUTPUT_ROOT
        if baseline_root is None
        else Path(baseline_root) / "output"
    )
    return capture_golden_tree(output_root, destination)


def relative_paths(output_root: Path) -> tuple[Path, ...]:
    """Return only the selected output-relative paths under ``output_root``."""
    root = Path(output_root)
    return tuple(artifact.relative_path for artifact in collect_golden_artifacts(root))


__all__ = [
    "CapturedArtifact",
    "CaptureError",
    "capture_baseline",
    "capture_golden_tree",
    "collect_golden_artifacts",
    "relative_paths",
]
