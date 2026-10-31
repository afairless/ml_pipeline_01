"""Unit and property tests for the commit-time vocabulary gate."""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts import build_denylist, scrub_repo

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE = "widget"
PROBE_DIGEST = scrub_repo.digest(PROBE)
PROBE_HASHES = frozenset({PROBE_DIGEST})


def parquet_payload(frame: pl.DataFrame) -> bytes:
    """Return ``frame`` encoded as parquet bytes."""
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


def write_digest_file(root: Path, entries: list[str]) -> None:
    """Write a digest file under ``root`` with one entry per line."""
    path = root / scrub_repo.HASHES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries) + "\n", encoding="utf-8")


GIT_ENV = {
    "GIT_AUTHOR_NAME": "gate-test",
    "GIT_AUTHOR_EMAIL": "gate-test@example.invalid",
    "GIT_COMMITTER_NAME": "gate-test",
    "GIT_COMMITTER_EMAIL": "gate-test@example.invalid",
}


def git(root: Path, *arguments: str) -> str:
    """Run one git command in ``root`` and return its standard output."""
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=True,
        text=True,
        env={**os.environ, **GIT_ENV},
    )
    return result.stdout


def make_repo(tmp_path: Path) -> Path:
    """Create an empty repository that carries the probe digest file."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    write_digest_file(root, [PROBE_DIGEST])
    return root


# ---------------------------------------------------------------------------
# Tokenizer and digest primitives
# ---------------------------------------------------------------------------


def test_tokenize_splits_case_and_separators() -> None:
    assert scrub_repo.tokenize("fooBarBaz") == ["foo", "bar", "baz"]
    assert scrub_repo.tokenize("HTTPServer") == ["http", "server"]
    assert scrub_repo.tokenize("alpha_beta-42/gamma") == [
        "alpha",
        "beta",
        "42",
        "gamma",
    ]


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert scrub_repo.tokenize("A, B!") == ["a", "b"]
    assert scrub_repo.tokenize("") == []


def test_canonical_ngrams_uses_every_window() -> None:
    assert list(scrub_repo.canonical_ngrams(["a", "b", "c"])) == [
        "a",
        "b",
        "c",
        "a b",
        "b c",
        "a b c",
    ]


def test_canonical_ngrams_respects_max_size() -> None:
    assert list(scrub_repo.canonical_ngrams(["a", "b"], max_size=1)) == ["a", "b"]


def test_canonicalize_joins_tokens_with_single_spaces() -> None:
    assert scrub_repo.canonicalize("  Alpha_Beta  ") == "alpha beta"


def test_digest_matches_sha256_of_the_canonical_term() -> None:
    assert scrub_repo.digest(PROBE) == hashlib.sha256(PROBE.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Content, name, and payload inspection
# ---------------------------------------------------------------------------


def test_scan_text_reports_fingerprint_and_line() -> None:
    hits = scrub_repo.scan_text("clean\nlet WIDGET pass\nclean", PROBE_HASHES, "f.txt")
    assert [hit.render() for hit in hits] == [
        f"f.txt:2: fingerprint={PROBE_DIGEST[:8]}"
    ]


def test_scan_text_reports_one_hit_per_line() -> None:
    hits = scrub_repo.scan_text("widget widget", PROBE_HASHES, "f.txt")
    assert len(hits) == 1


def test_scan_text_returns_nothing_for_clean_content() -> None:
    assert scrub_repo.scan_text("nothing to see", PROBE_HASHES, "f.txt") == []


def test_scan_text_ignores_empty_text() -> None:
    assert scrub_repo.scan_text("", PROBE_HASHES, "f.txt") == []


def test_scan_name_flags_a_component() -> None:
    hits = scrub_repo.scan_name("pkg/widget_data.py", PROBE_HASHES, "path")
    assert [hit.render() for hit in hits] == [f"path:1: fingerprint={PROBE_DIGEST[:8]}"]


def test_scan_name_accepts_a_clean_component() -> None:
    assert scrub_repo.scan_name("pkg/clean_data.py", PROBE_HASHES, "path") == []


def test_scan_blob_reads_text() -> None:
    hits = scrub_repo.scan_blob("a.txt", b"widget", PROBE_HASHES)
    assert len(hits) == 1


def test_scan_blob_fails_closed_on_unknown_binary() -> None:
    hits = scrub_repo.scan_blob("asset.bin", b"\x00\x01\x02", PROBE_HASHES)
    assert [hit.render() for hit in hits] == [
        f"asset.bin:1: fingerprint={scrub_repo.FAIL_CLOSED_FINGERPRINT}"
    ]


def test_scan_blob_fails_closed_on_invalid_utf8() -> None:
    hits = scrub_repo.scan_blob("asset.bin", b"\xff\xfe", PROBE_HASHES)
    assert [hit.fingerprint for hit in hits] == [scrub_repo.FAIL_CLOSED_FINGERPRINT]


def test_scan_blob_allows_an_allowlisted_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scrub_repo,
        "BINARY_PATH_ALLOWLIST",
        frozenset({scrub_repo.digest_path("asset.bin")}),
    )
    assert scrub_repo.scan_blob("asset.bin", b"\x00\x01", PROBE_HASHES) == []


def test_scan_blob_reads_parquet_column_names() -> None:
    payload = parquet_payload(pl.DataFrame({"widget_value": [1.0]}))
    assert scrub_repo.scan_blob("t.parquet", payload, PROBE_HASHES)


def test_scan_blob_reads_parquet_text_values() -> None:
    payload = parquet_payload(pl.DataFrame({"label": ["A widget"]}))
    hits = scrub_repo.scan_blob("t.parquet", payload, PROBE_HASHES)
    assert [hit.location for hit in hits] == ["t.parquet[label]"]


def test_scan_blob_ignores_parquet_numeric_payload() -> None:
    payload = parquet_payload(pl.DataFrame({"amount": [123456.0]}))
    assert scrub_repo.scan_blob("t.parquet", payload, PROBE_HASHES) == []


# ---------------------------------------------------------------------------
# Denylist loading and fail-closed state
# ---------------------------------------------------------------------------


def test_load_hashes_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(scrub_repo.DenylistUnavailable):
        scrub_repo.load_hashes(tmp_path)


def test_load_hashes_rejects_an_empty_file(tmp_path: Path) -> None:
    write_digest_file(tmp_path, [])
    with pytest.raises(scrub_repo.DenylistUnavailable):
        scrub_repo.load_hashes(tmp_path)


def test_load_hashes_rejects_a_malformed_digest(tmp_path: Path) -> None:
    write_digest_file(tmp_path, ["not-a-digest"])
    with pytest.raises(scrub_repo.DenylistUnavailable):
        scrub_repo.load_hashes(tmp_path)


def test_load_hashes_accepts_valid_digests(tmp_path: Path) -> None:
    write_digest_file(tmp_path, [PROBE_DIGEST])
    assert scrub_repo.load_hashes(tmp_path) == frozenset({PROBE_DIGEST})


def test_check_state_rejects_a_mismatched_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scrub_repo, "is_tracked", lambda root, relative: False)
    write_digest_file(tmp_path, [PROBE_DIGEST])
    local = tmp_path / scrub_repo.LOCAL_TERMS_PATH
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("other_term\n", encoding="utf-8")
    with pytest.raises(scrub_repo.DenylistUnavailable):
        scrub_repo.check_denylist_state(tmp_path)


def test_check_state_rejects_an_empty_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scrub_repo, "is_tracked", lambda root, relative: False)
    write_digest_file(tmp_path, [PROBE_DIGEST])
    local = tmp_path / scrub_repo.LOCAL_TERMS_PATH
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("\n", encoding="utf-8")
    with pytest.raises(scrub_repo.DenylistUnavailable):
        scrub_repo.check_denylist_state(tmp_path)


def test_check_state_rejects_a_tracked_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scrub_repo, "is_tracked", lambda root, relative: True)
    write_digest_file(tmp_path, [PROBE_DIGEST])
    with pytest.raises(scrub_repo.DenylistUnavailable):
        scrub_repo.check_denylist_state(tmp_path)


def test_check_state_accepts_a_matching_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scrub_repo, "is_tracked", lambda root, relative: False)
    write_digest_file(tmp_path, [PROBE_DIGEST])
    local = tmp_path / scrub_repo.LOCAL_TERMS_PATH
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("Widget\n", encoding="utf-8")
    assert scrub_repo.check_denylist_state(tmp_path) == frozenset({PROBE_DIGEST})


def test_main_reports_an_unavailable_denylist(tmp_path: Path) -> None:
    assert scrub_repo.main(["--worktree", "--root", str(tmp_path)]) == 1


# ---------------------------------------------------------------------------
# Repository gate
# ---------------------------------------------------------------------------


def test_committed_digests_are_sorted_unique_lowercase_hex() -> None:
    lines = (REPO_ROOT / scrub_repo.HASHES_PATH).read_text(encoding="utf-8")
    entries = lines.splitlines()
    assert entries
    assert entries == sorted(set(entries))
    assert all(scrub_repo.HEX_DIGEST.fullmatch(entry) for entry in entries)


def test_local_terms_file_is_never_tracked() -> None:
    relative = scrub_repo.LOCAL_TERMS_PATH.as_posix()
    assert not scrub_repo.is_tracked(REPO_ROOT, relative)


def test_repository_worktree_passes_the_committed_gate() -> None:
    hashes = scrub_repo.load_hashes(REPO_ROOT)
    assert scrub_repo.scan_worktree(REPO_ROOT, hashes) == []


def test_pre_commit_hook_runs_the_staged_gate() -> None:
    hook = REPO_ROOT / ".githooks" / "pre-commit"
    assert hook.is_file()
    assert os.access(hook, os.X_OK)
    assert "--staged" in hook.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Index, history, and command-line modes
# ---------------------------------------------------------------------------


def test_scan_staged_accepts_clean_content(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "note.txt").write_text("nothing to see\n", encoding="utf-8")
    git(root, "add", "note.txt")
    assert scrub_repo.scan_staged(root, PROBE_HASHES) == []


def test_scan_staged_reports_a_denied_blob(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "note.txt").write_text("let widget pass\n", encoding="utf-8")
    git(root, "add", "note.txt")
    hits = scrub_repo.scan_staged(root, PROBE_HASHES)
    assert [hit.render() for hit in hits] == [
        f"note.txt:1: fingerprint={PROBE_DIGEST[:8]}"
    ]


def test_scan_staged_reports_a_denied_path(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "widget_notes.txt").write_text("clean\n", encoding="utf-8")
    git(root, "add", "widget_notes.txt")
    hits = scrub_repo.scan_staged(root, PROBE_HASHES)
    assert [hit.render() for hit in hits] == [
        f"widget_notes.txt:1: fingerprint={PROBE_DIGEST[:8]}"
    ]


def test_scan_staged_ignores_unstaged_edits(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "note.txt").write_text("clean\n", encoding="utf-8")
    git(root, "add", "note.txt")
    (root / "note.txt").write_text("widget\n", encoding="utf-8")
    assert scrub_repo.scan_staged(root, PROBE_HASHES) == []


def test_scan_history_accepts_a_clean_history(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "note.txt").write_text("clean\n", encoding="utf-8")
    git(root, "add", "note.txt")
    git(root, "commit", "-qm", "chore: add note")
    assert scrub_repo.scan_history(root, PROBE_HASHES) == []


def test_scan_history_reports_a_denied_blob(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "note.txt").write_text("widget\n", encoding="utf-8")
    git(root, "add", "note.txt")
    git(root, "commit", "-qm", "chore: add note")
    hits = scrub_repo.scan_history(root, PROBE_HASHES)
    assert [hit.fingerprint for hit in hits] == [PROBE_DIGEST[:8]]
    assert hits[0].location.startswith("commit ")


def test_scan_history_reports_a_denied_commit_message(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "note.txt").write_text("clean\n", encoding="utf-8")
    git(root, "add", "note.txt")
    git(root, "commit", "-qm", "feat: add widget")
    hits = scrub_repo.scan_history(root, PROBE_HASHES)
    assert [hit.fingerprint for hit in hits] == [PROBE_DIGEST[:8]]


def test_scan_history_labels_a_reused_blob_with_its_own_path(
    tmp_path: Path,
) -> None:
    root = make_repo(tmp_path)
    (root / "first.txt").write_text("widget\n", encoding="utf-8")
    git(root, "add", "first.txt")
    git(root, "commit", "-qm", "chore: add first")
    (root / "second.txt").write_text("widget\n", encoding="utf-8")
    git(root, "add", "second.txt")
    git(root, "commit", "-qm", "chore: add second")
    locations = {hit.location for hit in scrub_repo.scan_history(root, PROBE_HASHES)}
    assert any(location.endswith(":second.txt") for location in locations)


def test_main_reports_a_staged_hit(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "note.txt").write_text("widget\n", encoding="utf-8")
    git(root, "add", "note.txt")
    assert scrub_repo.main(["--staged", "--root", str(root)]) == 1


def test_main_accepts_a_clean_staged_index(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    (root / "note.txt").write_text("clean\n", encoding="utf-8")
    git(root, "add", "note.txt")
    assert scrub_repo.main(["--staged", "--root", str(root)]) == 0


# ---------------------------------------------------------------------------
# Derivation and hashing helpers
# ---------------------------------------------------------------------------


def make_source_tree(tmp_path: Path) -> Path:
    """Build a tiny synthetic source tree for derivation tests."""
    source = tmp_path / "source"
    (source / "src" / "core").mkdir(parents=True)
    (source / "src" / "project").mkdir(parents=True)
    (source / "src" / "core" / "frame_utils.py").write_text(
        "def keep_frame(value):\n    return value\n",
        encoding="utf-8",
    )
    (source / "src" / "project" / "quarantine_registry.py").write_text(
        "QUARANTINE = 'widget_batch'\n\n\ndef widget_count():\n    return 0\n",
        encoding="utf-8",
    )
    return source


def test_candidates_exclude_generic_and_stop_vocabulary(tmp_path: Path) -> None:
    candidates = build_denylist.candidates_from(make_source_tree(tmp_path))
    assert "quarantine" in candidates
    assert "widget" in candidates
    assert "frame" not in candidates
    assert "keep" not in candidates


def test_derive_writes_candidates_under_the_ignored_directory(tmp_path: Path) -> None:
    source = make_source_tree(tmp_path)
    root = tmp_path / "repo"
    root.mkdir()
    assert build_denylist.derive(root, source) == 0
    written = (root / build_denylist.CANDIDATES_PATH).read_text(encoding="utf-8")
    assert "quarantine" in written.splitlines()


def test_derive_rejects_a_missing_source(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    assert build_denylist.derive(root, tmp_path / "absent") == 1


def test_write_hashes_sorts_and_deduplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build_denylist, "is_tracked", lambda root, relative: False)
    local = tmp_path / build_denylist.LOCAL_TERMS_PATH
    local.parent.mkdir(parents=True)
    local.write_text("Widget\nwidget\n  alpha beta  \n", encoding="utf-8")
    assert build_denylist.write_hashes(tmp_path) == 0
    entries = (tmp_path / build_denylist.HASHES_PATH).read_text().splitlines()
    expected = sorted({scrub_repo.digest("widget"), scrub_repo.digest("alpha beta")})
    assert entries == expected


def test_write_hashes_refuses_a_missing_terms_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build_denylist, "is_tracked", lambda root, relative: False)
    assert build_denylist.write_hashes(tmp_path) == 1


def test_write_hashes_refuses_an_empty_terms_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build_denylist, "is_tracked", lambda root, relative: False)
    local = tmp_path / build_denylist.LOCAL_TERMS_PATH
    local.parent.mkdir(parents=True)
    local.write_text("\n\n", encoding="utf-8")
    assert build_denylist.write_hashes(tmp_path) == 1
    assert not (tmp_path / build_denylist.HASHES_PATH).exists()


def test_write_hashes_refuses_a_tracked_terms_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build_denylist, "is_tracked", lambda root, relative: True)
    local = tmp_path / build_denylist.LOCAL_TERMS_PATH
    local.parent.mkdir(parents=True)
    local.write_text("widget\n", encoding="utf-8")
    assert build_denylist.write_hashes(tmp_path) == 1
    assert not (tmp_path / build_denylist.HASHES_PATH).exists()


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------


@given(st.text())
def test_tokenize_output_is_normalized(text: str) -> None:
    tokens = scrub_repo.tokenize(text)
    assert all(token == token.lower() for token in tokens)
    assert all(token.isalnum() for token in tokens)
    assert scrub_repo.canonicalize(text) == " ".join(tokens)


@given(st.text())
def test_canonicalize_is_idempotent(text: str) -> None:
    once = scrub_repo.canonicalize(text)
    assert scrub_repo.canonicalize(once) == once


@given(st.lists(st.sampled_from(["alpha", "beta", "gamma"]), max_size=6))
def test_ngram_count_matches_the_window_arithmetic(tokens: list[str]) -> None:
    expected = sum(max(0, len(tokens) - width + 1) for width in (1, 2, 3))
    assert len(list(scrub_repo.canonical_ngrams(tokens))) == expected


@given(st.lists(st.sampled_from(["alpha", "beta", "gamma"]), min_size=1, max_size=5))
def test_every_planted_ngram_is_detected(tokens: list[str]) -> None:
    hashes = frozenset(scrub_repo.digest(term) for term in tokens)
    assert scrub_repo.scan_text(" ".join(tokens), hashes, "probe")


@given(st.text())
def test_digest_is_a_stable_sha256(text: str) -> None:
    canonical = scrub_repo.canonicalize(text)
    assert (
        scrub_repo.digest(canonical) == hashlib.sha256(canonical.encode()).hexdigest()
    )
