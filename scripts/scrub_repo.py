"""Commit-time vocabulary gate.

The gate tokenizes and hashes every tracked text payload, every path
component, every commit message, and the text columns of committed
``.parquet`` files. The resulting digests are compared with the committed
denylist in ``tests/vocabulary_denylist.sha256``. The plain terms live in an
untracked file (``.scrub/denylist.txt``); only their digests are committed.

A hit reports the location and the first eight hexadecimal characters of the
matching digest, never the term itself, so this module cannot leak the
vocabulary it enforces.

Modes:

``--worktree``
    Tracked files plus untracked files that are not ignored.
``--staged``
    The index (staged path names and blob contents).
``--history``
    Every commit reachable from ``--all``: message, tree path names, and
    blob contents.

Exit status: ``0`` when clean, ``1`` on a hit or an unavailable denylist,
``2`` on a usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

HASHES_PATH = Path("tests/vocabulary_denylist.sha256")
LOCAL_TERMS_PATH = Path(".scrub/denylist.txt")

CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
TOKEN = re.compile(r"[0-9A-Za-z]+")
HEX_DIGEST = re.compile(r"[0-9a-f]{64}")

# Digests of path names that may legitimately be binary. Empty by design: the
# repository commits no opaque binaries, so an unknown binary fails closed.
BINARY_PATH_ALLOWLIST: frozenset[str] = frozenset()

PARQUET_SUFFIX = ".parquet"

# Reported when a payload cannot be inspected at all. It is deliberately not a
# real digest prefix, so a fail-closed hit is distinguishable from a match.
FAIL_CLOSED_FINGERPRINT = "0" * 8


class DenylistUnavailable(RuntimeError):
    """The gate cannot run: no usable denylist is available."""


@dataclass(frozen=True)
class Hit:
    """One denied digest at one location."""

    location: str
    line: int
    fingerprint: str

    def render(self) -> str:
        """Return the stable one-line report for this hit."""
        return f"{self.location}:{self.line}: fingerprint={self.fingerprint}"


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lowercase tokens.

    Tokens break on every non-alphanumeric character, and camelCase or
    PascalCase boundaries are split before lowercasing, so ``fooBarBaz``
    becomes ``foo``, ``bar``, ``baz``.
    """
    tokens: list[str] = []
    for raw in TOKEN.findall(text):
        tokens.extend(part.lower() for part in CAMEL_BOUNDARY.split(raw) if part)
    return tokens


def canonical_ngrams(tokens: Sequence[str], max_size: int = 3) -> Iterator[str]:
    """Yield the canonical unigrams, bigrams, and trigrams of ``tokens``."""
    for width in range(1, max_size + 1):
        for start in range(len(tokens) - width + 1):
            yield " ".join(tokens[start : start + width])


def canonicalize(text: str) -> str:
    """Return the canonical single-space form of ``text``."""
    return " ".join(tokenize(text))


def digest(term: str) -> str:
    """Return the sha256 hex digest of a canonical term."""
    return hashlib.sha256(term.encode("utf-8")).hexdigest()


def digest_path(path: str) -> str:
    """Return the sha256 hex digest of a normalized path name."""
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def scan_text(text: str, hashes: frozenset[str], location: str) -> list[Hit]:
    """Return the first denied n-gram of every line of ``text``."""
    hits: list[Hit] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for term in canonical_ngrams(tokenize(line)):
            fingerprint = digest(term)
            if fingerprint in hashes:
                hits.append(Hit(location, number, fingerprint[:8]))
                break
    return hits


def scan_name(name: str, hashes: frozenset[str], location: str) -> list[Hit]:
    """Return the first denied n-gram of a file, directory, or column name."""
    normalized = name.replace("\\", "/")
    for term in canonical_ngrams(tokenize(normalized)):
        fingerprint = digest(term)
        if fingerprint in hashes:
            return [Hit(location, 1, fingerprint[:8])]
    return []


def _parquet_hits(name: str, data: bytes, hashes: frozenset[str]) -> list[Hit]:
    """Inspect a parquet payload's column names and string values."""
    import polars as pl

    hits: list[Hit] = []
    frame = pl.read_parquet(io.BytesIO(data))
    for column, dtype in frame.schema.items():
        hits.extend(scan_name(column, hashes, name))
        if dtype != pl.String and dtype != pl.Categorical:
            continue
        values = frame[column].drop_nulls().unique().to_list()
        for value in values:
            if isinstance(value, str):
                hits.extend(scan_text(value, hashes, f"{name}[{column}]"))
    return hits


def _binary_hits(name: str) -> list[Hit]:
    """Return no hit for an allowlisted binary, otherwise a fail-closed hit."""
    if digest_path(name) in BINARY_PATH_ALLOWLIST:
        return []
    return [Hit(name, 1, FAIL_CLOSED_FINGERPRINT)]


def scan_blob(name: str, data: bytes, hashes: frozenset[str]) -> list[Hit]:
    """Inspect one payload, deciding text, parquet, or fail-closed binary.

    A payload containing a NUL byte, or one that is not valid UTF-8, is
    treated as binary and fails closed unless its hashed path is allowlisted.
    """
    if name.lower().endswith(PARQUET_SUFFIX):
        return _parquet_hits(name, data, hashes)
    if b"\x00" in data:
        return _binary_hits(name)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return _binary_hits(name)
    return scan_text(text, hashes, name)


def load_hashes(root: Path) -> frozenset[str]:
    """Read and validate the committed denylist digests.

    Raises:
        DenylistUnavailable: the file is missing, empty, or malformed.
    """
    hashes_file = root / HASHES_PATH
    if not hashes_file.is_file():
        raise DenylistUnavailable(f"{HASHES_PATH.as_posix()} is missing")
    entries = [
        line.strip() for line in hashes_file.read_text(encoding="utf-8").splitlines()
    ]
    entries = [entry for entry in entries if entry]
    if not entries:
        raise DenylistUnavailable(f"{HASHES_PATH.as_posix()} is empty")
    for entry in entries:
        if not HEX_DIGEST.fullmatch(entry):
            raise DenylistUnavailable(
                f"{HASHES_PATH.as_posix()} holds a malformed digest"
            )
    return frozenset(entries)


def is_tracked(root: Path, relative: str) -> bool:
    """Return whether ``relative`` is tracked in the repository at ``root``."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def local_terms(root: Path) -> list[str]:
    """Return the reviewed local terms, or an empty list when absent."""
    terms_file = root / LOCAL_TERMS_PATH
    if not terms_file.is_file():
        return []
    lines = terms_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def check_denylist_state(root: Path) -> frozenset[str]:
    """Load the effective denylist and enforce the fail-closed rules.

    The committed digest file is authoritative, so a clean clone can run the
    gate. When the untracked reviewed terms are present, they must be
    untracked, non-empty, and consistent with the committed digests.

    Raises:
        DenylistUnavailable: any fail-closed condition.
    """
    local = root / LOCAL_TERMS_PATH
    if is_tracked(root, LOCAL_TERMS_PATH.as_posix()):
        raise DenylistUnavailable("the local terms file is tracked")
    hashes = load_hashes(root)
    if local.is_file():
        terms = local_terms(root)
        if not terms:
            raise DenylistUnavailable("the local terms file is empty")
        expected = {digest(canonicalize(term)) for term in terms if tokenize(term)}
        if expected != set(hashes):
            raise DenylistUnavailable(
                "the local terms file does not match the committed digests; "
                "run build_denylist.py --hashes"
            )
    return hashes


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=True,
    )
    return result.stdout


def worktree_paths(root: Path) -> list[str]:
    """Return tracked and untracked-but-not-ignored paths, sorted."""
    raw = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    return sorted({part for part in raw.decode().split("\0") if part})


def scan_worktree(root: Path, hashes: frozenset[str]) -> list[Hit]:
    """Inspect every worktree path and payload in scope."""
    hits: list[Hit] = []
    for relative in worktree_paths(root):
        path = root / relative
        if not path.is_file():
            continue
        hits.extend(scan_name(relative, hashes, relative))
        hits.extend(scan_blob(relative, path.read_bytes(), hashes))
    return hits


def scan_staged(root: Path, hashes: frozenset[str]) -> list[Hit]:
    """Inspect the index path names and staged blob contents."""
    hits: list[Hit] = []
    raw = _git(root, "ls-files", "--stage", "-z")
    for record in raw.decode().split("\0"):
        if not record:
            continue
        meta, _, relative = record.partition("\t")
        fields = meta.split()
        mode = fields[0]
        blob = fields[1]
        if mode == "160000":
            hits.append(Hit(relative, 1, FAIL_CLOSED_FINGERPRINT))
            continue
        hits.extend(scan_name(relative, hashes, relative))
        hits.extend(scan_blob(relative, _git(root, "cat-file", "blob", blob), hashes))
    return hits


def scan_history(root: Path, hashes: frozenset[str]) -> list[Hit]:
    """Inspect every reachable commit's message, path names, and blobs."""
    hits: list[Hit] = []
    cache: dict[tuple[str, str], list[Hit]] = {}
    commits = _git(root, "rev-list", "--all").decode().split()
    for commit in commits:
        label = f"commit {commit[:12]}"
        message = _git(root, "log", "-1", "--format=%B", commit).decode(
            errors="replace"
        )
        hits.extend(scan_text(message, hashes, label))
        tree = _git(root, "ls-tree", "-r", "-z", commit)
        for record in tree.split(b"\0"):
            if not record:
                continue
            meta, _, raw_path = record.partition(b"\t")
            fields = meta.split()
            blob = fields[2].decode()
            relative = raw_path.decode(errors="replace")
            hits.extend(scan_name(relative, hashes, f"{label}:{relative}"))
            key = (blob, relative)
            if key not in cache:
                cache[key] = scan_blob(
                    relative, _git(root, "cat-file", "blob", blob), hashes
                )
            hits.extend(
                Hit(f"{label}:{hit.location}", hit.line, hit.fingerprint)
                for hit in cache[key]
            )
    return hits


def render(hits: Sequence[Hit]) -> None:
    """Print hits to standard output, one stable line each."""
    for hit in hits:
        print(hit.render())


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Reject commits whose tracked content uses denied vocabulary.",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--worktree", action="store_true", help="inspect the worktree")
    modes.add_argument("--staged", action="store_true", help="inspect the index")
    modes.add_argument("--history", action="store_true", help="inspect history")
    parser.add_argument(
        "--root",
        default=".",
        help="repository root (default: the current directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gate and return the process exit status."""
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    try:
        hashes = check_denylist_state(root)
        if args.worktree:
            hits = scan_worktree(root, hashes)
        elif args.staged:
            hits = scan_staged(root, hashes)
        else:
            hits = scan_history(root, hashes)
    except DenylistUnavailable as error:
        print(f"denylist unavailable: {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(f"git failed: {error.stderr.decode(errors='replace')}", file=sys.stderr)
        return 1
    if hits:
        render(hits)
        print(f"vocabulary gate failed: {len(hits)} hit(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
