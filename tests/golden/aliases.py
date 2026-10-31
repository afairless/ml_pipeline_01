"""Declare which pipeline artifacts belong in the golden baseline.

The baseline intentionally contains a small, version-insensitive subset of the
full output tree: model, prediction, and feature manifests; regression and
thresholded binary metrics JSON; statistics text; and the s06 cross-model
display CSVs. Parquet split tables, PNG figures, and estimator serializations
(model JSON/text files) are excluded because their bytes are either covered by
dedicated determinism tests or depend on plotting and estimator versions.

An alias is a named output pattern. Capturing, diffing, and reporting all share
the same declarations, so the baseline cannot drift by using different
selection rules in different places.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ArtifactAlias:
    """One named artifact class selected by an output-relative glob.

    Attributes:
        name: Stable alias used in reports and tests, e.g. ``'manifest'``.
        pattern: POSIX glob matched against the output-relative path. ``*``
            and ``?`` follow :func:`fnmatch.fnmatchcase` semantics.
        description: Human-readable explanation of the selected files.
    """

    name: str
    pattern: str
    description: str

    def matches(self, relative_path: Path | str) -> bool:
        """Return whether an output-relative path belongs to this alias."""
        candidate = (
            relative_path
            if isinstance(relative_path, str)
            else relative_path.as_posix()
        )
        return fnmatchcase(candidate, self.pattern)


GOLDEN_ARTIFACT_ALIASES: tuple[ArtifactAlias, ...] = (
    ArtifactAlias(
        name="manifest",
        pattern="*/*manifest*.json",
        description="model, prediction, and feature run manifests",
    ),
    ArtifactAlias(
        name="metrics",
        pattern="s06_evaluate/*/*/*_metrics.json",
        description="regression and thresholded binary metrics JSON",
    ),
    ArtifactAlias(
        name="statistics",
        pattern="*/*statistics*.txt",
        description="outcome, comparison, and split statistics text",
    ),
    ArtifactAlias(
        name="display_table",
        pattern="s06_evaluate/*/*_display.csv",
        description="per-outcome cross-model display CSV",
    ),
)
"""The complete, ordered baseline artifact contract."""

ARTIFACT_ALIASES: MappingProxyType[str, ArtifactAlias] = MappingProxyType(
    {alias.name: alias for alias in GOLDEN_ARTIFACT_ALIASES}
)
"""Alias name -> declaration, for callers that look one alias up by name."""


def artifact_alias(relative_path: Path | str) -> ArtifactAlias | None:
    """Return the first alias matching an output-relative path, if any."""
    for alias in GOLDEN_ARTIFACT_ALIASES:
        if alias.matches(relative_path):
            return alias
    return None


def is_golden_artifact(relative_path: Path | str) -> bool:
    """Return whether an output-relative path belongs in the baseline."""
    return artifact_alias(relative_path) is not None


def select_golden_artifacts(
    relative_paths: tuple[Path, ...] | list[Path],
) -> tuple[Path, ...]:
    """Filter output-relative paths to the declared baseline artifacts.

    Selection preserves the caller's order, so callers can pass an already
    sorted file list and receive a deterministic result.
    """
    return tuple(path for path in relative_paths if is_golden_artifact(path))
