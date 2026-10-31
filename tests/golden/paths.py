"""Filesystem locations used by the golden-baseline harness.

Every path is anchored to the location of this package, never to the process
working directory. The live pipeline writes to the repository's ``output/``
directory; the committed baseline lives beside this module under
``baseline/output/``. The leading slash on the repository's ``/output/`` ignore
rule keeps the baseline tree tracked.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

GOLDEN_PACKAGE_ROOT = Path(__file__).resolve().parent
"""The ``tests/golden/`` directory holding this package."""

PROJECT_ROOT = GOLDEN_PACKAGE_ROOT.parents[1]
"""Repository root, derived from the package rather than the working directory."""

OUTPUT_ROOT = PROJECT_ROOT / "output"
"""Default live output tree written by the full demo pipeline."""

BASELINE_ROOT = GOLDEN_PACKAGE_ROOT / "baseline"
"""Directory holding the committed golden baseline."""

BASELINE_OUTPUT_ROOT = BASELINE_ROOT / "output"
"""The baseline's output tree, comparable with :data:`OUTPUT_ROOT`."""

FIXTURES_ROOT = GOLDEN_PACKAGE_ROOT / "fixtures"
"""Optional staging area for small fixtures used by golden tests."""


def relative_output_path(path: Path, root: Path = OUTPUT_ROOT) -> Path:
    """Return ``path`` relative to an output root.

    Args:
        path: A path inside (or equal to) ``root``.
        root: The output root to relativize against.

    Raises:
        ValueError: ``path`` is outside ``root``.
    """
    candidate = Path(path).resolve()
    anchor = Path(root).resolve()
    try:
        return candidate.relative_to(anchor)
    except ValueError as error:
        raise ValueError(f"{path} is not inside the output root {root}") from error


def iter_files(root: Path) -> Iterator[Path]:
    """Yield every regular file under ``root`` in stable relative-path order.

    Missing roots yield nothing. Sorting by the POSIX relative path makes the
    iteration independent of filesystem directory order.

    Raises:
        NotADirectoryError: ``root`` exists but is not a directory.
    """
    anchor = Path(root)
    if not anchor.exists():
        return
    if not anchor.is_dir():
        raise NotADirectoryError(f"expected a directory, found {anchor}")
    files = (path for path in anchor.rglob("*") if path.is_file())
    yield from sorted(files, key=lambda path: path.relative_to(anchor).as_posix())


def iter_relative_files(root: Path) -> Iterator[Path]:
    """Yield every regular file under ``root`` as a relative ``Path``."""
    anchor = Path(root)
    for path in iter_files(anchor):
        yield path.relative_to(anchor)
