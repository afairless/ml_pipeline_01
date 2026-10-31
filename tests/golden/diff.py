"""Byte-exact directory comparison for golden baseline tests.

The primary entry point, :func:`diff_baseline`, compares a live output tree
with the committed baseline using the shared artifact aliases, so excluded
files (parquet, figures, estimator serializations) are ignored on both sides.
A changed artifact reports the relative path, the first differing byte offset,
and the two byte lengths; an added or removed artifact reports only its path.

Text is compared as bytes. No newline, whitespace, or JSON normalization is
applied: the baseline contract says an exact diff, and a readable but
semantically equal change must still be reviewed and recaptured explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tests.golden.aliases import is_golden_artifact
from tests.golden.paths import BASELINE_OUTPUT_ROOT, iter_relative_files

DiffKind = Literal["added", "removed", "changed"]


@dataclass(frozen=True, slots=True)
class DiffEntry:
    """One difference between a baseline tree and an actual tree.

    Attributes:
        relative_path: Artifact path relative to the compared output root.
        kind: ``'added'`` (actual only), ``'removed'`` (baseline only), or
            ``'changed'`` (both, different bytes).
        byte_offset: First differing byte offset for ``'changed'``; ``None``
            for added or removed artifacts.
        baseline_size: Baseline file size for ``'changed'``; ``None`` for
            added or removed artifacts.
        actual_size: Actual file size for ``'changed'``; ``None`` for added
            or removed artifacts.
    """

    relative_path: Path
    kind: DiffKind
    byte_offset: int | None = None
    baseline_size: int | None = None
    actual_size: int | None = None

    def render(self) -> str:
        """Return a single-line, deterministic description of the entry."""
        path = self.relative_path.as_posix()
        if self.kind == "changed":
            return (
                f"changed {path} at byte {self.byte_offset} "
                f"(baseline {self.baseline_size} bytes, "
                f"actual {self.actual_size} bytes)"
            )
        return f"{self.kind} {path}"


@dataclass(frozen=True, slots=True)
class DiffResult:
    """The ordered differences between two trees.

    Attributes:
        entries: Differences sorted by path and then kind.
    """

    entries: tuple[DiffEntry, ...]

    @property
    def is_empty(self) -> bool:
        """Return whether the two trees match exactly."""
        return not self.entries

    @property
    def added(self) -> tuple[Path, ...]:
        """Return the paths present only in the actual tree."""
        return tuple(
            entry.relative_path for entry in self.entries if entry.kind == "added"
        )

    @property
    def removed(self) -> tuple[Path, ...]:
        """Return the paths present only in the baseline tree."""
        return tuple(
            entry.relative_path for entry in self.entries if entry.kind == "removed"
        )

    @property
    def changed(self) -> tuple[Path, ...]:
        """Return the paths whose bytes differ."""
        return tuple(
            entry.relative_path for entry in self.entries if entry.kind == "changed"
        )

    def render(self) -> str:
        """Return a newline-separated report, or an empty string on a match."""
        return "\n".join(entry.render() for entry in self.entries)

    def __bool__(self) -> bool:
        """Return ``True`` when at least one difference exists."""
        return bool(self.entries)

    def __str__(self) -> str:
        """Return the rendered report."""
        return self.render()


def _first_difference(left: Path, right: Path) -> tuple[int, int, int]:
    """Return ``(offset, left_size, right_size)`` for two differing files."""
    left_bytes = left.read_bytes()
    right_bytes = right.read_bytes()
    limit = min(len(left_bytes), len(right_bytes))
    offset = next(
        (index for index in range(limit) if left_bytes[index] != right_bytes[index]),
        limit,
    )
    return offset, len(left_bytes), len(right_bytes)


def _same_bytes(left: Path, right: Path) -> bool:
    """Return whether two files have identical bytes."""
    if left.stat().st_size != right.stat().st_size:
        return False
    return left.read_bytes() == right.read_bytes()


def _relative_files(
    root: Path,
    predicate: object | None,
) -> dict[Path, Path]:
    """Map relative file paths to absolute paths under ``root``."""
    anchor = Path(root)
    mapping: dict[Path, Path] = {}
    for relative_path in iter_relative_files(anchor):
        if callable(predicate) and not predicate(relative_path):
            continue
        mapping[relative_path] = anchor / relative_path
    return mapping


def diff_directories(
    baseline_root: Path,
    actual_root: Path,
    *,
    predicate: object | None = None,
) -> DiffResult:
    """Compare two directory trees and return their ordered differences.

    Args:
        baseline_root: Expected tree.
        actual_root: Actual tree.
        predicate: Optional callable taking an output-relative ``Path`` and
            returning whether the file should be compared. ``None`` compares
            every regular file.

    Raises:
        FileNotFoundError: Either root does not exist.
        NotADirectoryError: Either root is not a directory.
    """
    baseline_path = Path(baseline_root)
    actual_path = Path(actual_root)
    for label, root in (("baseline", baseline_path), ("actual", actual_path)):
        if not root.exists():
            raise FileNotFoundError(f"{label} root does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"{label} root is not a directory: {root}")

    baseline_files = _relative_files(baseline_path, predicate)
    actual_files = _relative_files(actual_path, predicate)

    entries: list[DiffEntry] = []
    for relative_path in sorted(baseline_files.keys() | actual_files.keys()):
        baseline_file = baseline_files.get(relative_path)
        actual_file = actual_files.get(relative_path)
        if baseline_file is None:
            entries.append(DiffEntry(relative_path=relative_path, kind="added"))
            continue
        if actual_file is None:
            entries.append(DiffEntry(relative_path=relative_path, kind="removed"))
            continue
        if _same_bytes(baseline_file, actual_file):
            continue
        offset, baseline_size, actual_size = _first_difference(
            baseline_file,
            actual_file,
        )
        entries.append(
            DiffEntry(
                relative_path=relative_path,
                kind="changed",
                byte_offset=offset,
                baseline_size=baseline_size,
                actual_size=actual_size,
            )
        )
    return DiffResult(entries=tuple(entries))


def diff_golden_tree(
    baseline_output_root: Path,
    output_root: Path,
) -> DiffResult:
    """Compare only the declared golden artifacts in two output trees."""
    return diff_directories(
        baseline_output_root,
        output_root,
        predicate=is_golden_artifact,
    )


def diff_baseline(
    output_root: Path,
    baseline_output_root: Path = BASELINE_OUTPUT_ROOT,
) -> DiffResult:
    """Compare a live output tree with a baseline tree.

    ``baseline_output_root`` may be either the baseline's ``output/`` tree
    itself or the baseline root that contains it. Accepting both keeps a
    caller from silently comparing an extra directory level.
    """
    candidate = Path(baseline_output_root)
    nested = candidate / "output"
    resolved = nested if nested.is_dir() else candidate
    return diff_golden_tree(resolved, output_root)


compare_trees = diff_directories
"""Alias for callers that want a neutral, all-file comparison."""


__all__ = [
    "DiffEntry",
    "DiffKind",
    "DiffResult",
    "compare_trees",
    "diff_baseline",
    "diff_directories",
    "diff_golden_tree",
]
