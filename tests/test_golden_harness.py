"""Tests for the golden-baseline harness declarations and mechanics."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.golden import (
    BASELINE_OUTPUT_ROOT,
    GOLDEN_ARTIFACT_ALIASES,
    GOLDEN_PACKAGE_ROOT,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    ArtifactAlias,
    CaptureError,
    artifact_alias,
    capture_baseline,
    capture_golden_tree,
    collect_golden_artifacts,
    compare_trees,
    diff_baseline,
    diff_directories,
    diff_golden_tree,
    is_golden_artifact,
    iter_files,
    iter_relative_files,
    relative_output_path,
    run_demo_pipeline,
    seed_demo_source,
    select_golden_artifacts,
    temporary_demo_run,
)

INCLUDED_PATHS = (
    Path("s01_outcome/outcome_statistics.txt"),
    Path("s01b_outcome_analysis/outcome_differences_statistics.txt"),
    Path("s02_feature/feature_manifest.json"),
    Path("s03_divide/outcome_1/statistics.txt"),
    Path("s04_train/outcome_1/model_00_signal_0/manifest.json"),
    Path("s06_evaluate/outcome_1/outcome_1_display.csv"),
    Path(
        "s06_evaluate/outcome_1/model_00_signal_0_test/"
        "model_00_signal_0_regression_metrics.json"
    ),
    Path(
        "s06_evaluate/outcome_1/model_00_signal_0_test/"
        "model_00_signal_0_transform_binary_0,5_metrics.json"
    ),
)

EXCLUDED_PATHS = (
    Path("s01_outcome/outcome_df.parquet"),
    Path("s02_feature/feature_df_head.csv"),
    Path("s04_train/outcome_1/model_00_signal_0/model.json"),
    Path("s04_train/outcome_1/model_00_signal_0/model.txt"),
    Path("s06_evaluate/outcome_1/model_00_signal_0_test/roc.png"),
)


def _write(root: Path, relative_path: Path, content: bytes) -> Path:
    """Write bytes to a path beneath ``root`` and return the file path."""
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _populate_output(root: Path) -> None:
    """Create one included and one excluded artifact per stage family."""
    for index, relative_path in enumerate(INCLUDED_PATHS):
        _write(root, relative_path, f"included-{index}\n".encode())
    for index, relative_path in enumerate(EXCLUDED_PATHS):
        _write(root, relative_path, f"excluded-{index}\n".encode())


# --- paths -----------------------------------------------------------------


def test_package_and_repository_paths_are_anchored_to_this_file() -> None:
    """The harness paths point at this repository, not the working directory."""
    assert GOLDEN_PACKAGE_ROOT == Path(__file__).resolve().parent / "golden"
    assert PROJECT_ROOT == Path(__file__).resolve().parent.parent
    assert OUTPUT_ROOT == PROJECT_ROOT / "output"
    assert BASELINE_OUTPUT_ROOT == GOLDEN_PACKAGE_ROOT / "baseline" / "output"


def test_relative_output_path_accepts_inside_and_rejects_outside(
    tmp_path: Path,
) -> None:
    """Relativization is explicit about a path outside the output root."""
    root = tmp_path / "output"
    inside = root / "stage" / "artifact.txt"
    assert relative_output_path(inside, root) == Path("stage/artifact.txt")

    with pytest.raises(ValueError, match="not inside"):
        relative_output_path(tmp_path / "elsewhere.txt", root)


def test_iter_files_is_sorted_and_missing_root_yields_nothing(
    tmp_path: Path,
) -> None:
    """File iteration is stable and treats an absent root as empty."""
    _write(tmp_path, Path("b/second.txt"), b"2")
    _write(tmp_path, Path("a/first.txt"), b"1")

    assert tuple(iter_relative_files(tmp_path)) == (
        Path("a/first.txt"),
        Path("b/second.txt"),
    )
    assert tuple(iter_relative_files(tmp_path / "absent")) == ()
    assert tuple(iter_files(tmp_path)) == (
        tmp_path / "a" / "first.txt",
        tmp_path / "b" / "second.txt",
    )


def test_iter_files_rejects_a_non_directory_root(tmp_path: Path) -> None:
    """A file passed as an output root is rejected, not silently empty."""
    path = tmp_path / "file.txt"
    path.write_text("content", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        tuple(iter_files(path))


# --- aliases ---------------------------------------------------------------


@pytest.mark.parametrize("relative_path", INCLUDED_PATHS, ids=str)
def test_declared_baseline_artifacts_are_selected(relative_path: Path) -> None:
    """Manifests, metrics, statistics, and display CSVs are baseline files."""
    assert is_golden_artifact(relative_path)
    assert artifact_alias(relative_path) is not None


@pytest.mark.parametrize("relative_path", EXCLUDED_PATHS, ids=str)
def test_parquet_figures_and_model_serializations_are_excluded(
    relative_path: Path,
) -> None:
    """Excluded artifact families never enter the baseline."""
    assert not is_golden_artifact(relative_path)
    assert artifact_alias(relative_path) is None


@pytest.mark.parametrize("relative_path", INCLUDED_PATHS, ids=str)
def test_alias_matching_accepts_posix_strings_and_paths(
    relative_path: Path,
) -> None:
    """The public matcher accepts both path objects and POSIX strings."""
    alias = artifact_alias(relative_path)
    assert alias is not None
    assert alias.matches(relative_path.as_posix())


def test_alias_names_are_unique_and_immutable() -> None:
    """Each alias has one stable name and the table is a tuple."""
    names = tuple(alias.name for alias in GOLDEN_ARTIFACT_ALIASES)
    assert len(names) == len(set(names))
    assert isinstance(GOLDEN_ARTIFACT_ALIASES, tuple)
    assert all(isinstance(alias, ArtifactAlias) for alias in GOLDEN_ARTIFACT_ALIASES)


def test_select_golden_artifacts_preserves_input_order() -> None:
    """Selection is a filter, not a re-sort."""
    candidates = (EXCLUDED_PATHS[0], INCLUDED_PATHS[0], INCLUDED_PATHS[1])
    assert select_golden_artifacts(candidates) == (
        INCLUDED_PATHS[0],
        INCLUDED_PATHS[1],
    )


# --- capture ---------------------------------------------------------------


def test_collect_golden_artifacts_selects_and_sorts(tmp_path: Path) -> None:
    """Collection applies the shared contract in stable path order."""
    _populate_output(tmp_path)
    artifacts = collect_golden_artifacts(tmp_path)

    assert tuple(artifact.relative_path for artifact in artifacts) == tuple(
        sorted(INCLUDED_PATHS, key=lambda path: path.as_posix())
    )
    assert {artifact.alias.name for artifact in artifacts} == {
        "manifest",
        "metrics",
        "statistics",
        "display_table",
    }
    assert all(artifact.source.is_file() for artifact in artifacts)


def test_capture_writes_only_selected_artifacts_byte_for_byte(
    tmp_path: Path,
) -> None:
    """A capture copies selected files exactly and omits the excluded ones."""
    output_root = tmp_path / "output"
    baseline_root = tmp_path / "baseline"
    _populate_output(output_root)

    captured = capture_baseline(output_root, baseline_root)

    assert set(captured) == set(INCLUDED_PATHS)
    for relative_path in INCLUDED_PATHS:
        assert (baseline_root / "output" / relative_path).read_bytes() == (
            output_root / relative_path
        ).read_bytes()
    for relative_path in EXCLUDED_PATHS:
        assert not (baseline_root / "output" / relative_path).exists()


def test_capture_is_repeatable_and_replaces_the_previous_tree(
    tmp_path: Path,
) -> None:
    """Recapturing replaces stale baseline files instead of accumulating them."""
    output_root = tmp_path / "output"
    baseline_root = tmp_path / "baseline"
    _populate_output(output_root)

    first = capture_golden_tree(output_root, baseline_root)
    first_bytes = {path: (baseline_root / path).read_bytes() for path in first}
    second = capture_golden_tree(output_root, baseline_root)

    assert first == second
    for path in second:
        assert (baseline_root / path).read_bytes() == first_bytes[path]
    assert not (baseline_root / EXCLUDED_PATHS[0]).exists()


def test_capture_raises_when_no_golden_artifacts_exist(tmp_path: Path) -> None:
    """An output tree with nothing eligible cannot produce an empty baseline."""
    output_root = tmp_path / "output"
    _write(output_root, EXCLUDED_PATHS[0], b"not selected")

    with pytest.raises(CaptureError, match="no baseline artifacts"):
        capture_golden_tree(output_root, tmp_path / "baseline")


def test_failed_capture_leaves_the_previous_baseline_untouched(
    tmp_path: Path,
) -> None:
    """An invalid source root fails before the destination is replaced."""
    output_root = tmp_path / "output"
    baseline_root = tmp_path / "baseline"
    _populate_output(output_root)
    capture_golden_tree(output_root, baseline_root)
    before = {
        path.relative_to(baseline_root): path.read_bytes()
        for path in iter_files(baseline_root)
    }

    with pytest.raises(FileNotFoundError, match="output root does not exist"):
        capture_golden_tree(tmp_path / "absent-output", baseline_root)

    after = {
        path.relative_to(baseline_root): path.read_bytes()
        for path in iter_files(baseline_root)
    }
    assert after == before


# --- diff ------------------------------------------------------------------


def test_identical_trees_have_an_empty_diff(tmp_path: Path) -> None:
    """Equal bytes produce no entries and a truthy-as-empty result."""
    baseline_root = tmp_path / "baseline"
    actual_root = tmp_path / "actual"
    _populate_output(baseline_root)
    _populate_output(actual_root)

    result = diff_directories(baseline_root, actual_root)

    assert result.is_empty
    assert not result
    assert result.render() == ""
    assert repr(result.entries) == "()"


def test_diff_reports_added_removed_and_changed_artifacts(
    tmp_path: Path,
) -> None:
    """Every change kind is reported with the exact relative path."""
    baseline_root = tmp_path / "baseline"
    actual_root = tmp_path / "actual"
    _populate_output(baseline_root)
    _populate_output(actual_root)

    changed = INCLUDED_PATHS[0]
    (actual_root / changed).write_bytes(b"different\n")
    (actual_root / INCLUDED_PATHS[1]).unlink()
    added = Path(
        "s06_evaluate/outcome_1/new_model_test/new_model_test_regression_metrics.json"
    )
    _write(actual_root, added, b"added\n")

    result = diff_directories(baseline_root, actual_root)

    assert result.changed == (changed,)
    assert result.removed == (INCLUDED_PATHS[1],)
    assert result.added == (added,)
    assert "changed s01_outcome/outcome_statistics.txt at byte 0" in result.render()
    assert "removed s01b_outcome_analysis" in result.render()
    assert "added s06_evaluate/outcome_1/new_model_test" in result.render()


def test_diff_reports_first_differing_byte_and_sizes(tmp_path: Path) -> None:
    """A changed entry carries the first byte offset and both file sizes."""
    baseline_root = tmp_path / "baseline"
    actual_root = tmp_path / "actual"
    relative_path = Path("s01_outcome/outcome_statistics.txt")
    _write(baseline_root, relative_path, b"abcde")
    _write(actual_root, relative_path, b"abXdef")

    entry = diff_directories(baseline_root, actual_root).entries[0]

    assert entry.kind == "changed"
    assert entry.byte_offset == 2
    assert entry.baseline_size == 5
    assert entry.actual_size == 6


def test_golden_diff_ignores_excluded_artifacts_on_both_sides(
    tmp_path: Path,
) -> None:
    """Parquet, figures, and model files never make a golden diff fail."""
    baseline_root = tmp_path / "baseline"
    actual_root = tmp_path / "actual"
    _populate_output(baseline_root)
    _populate_output(actual_root)

    _write(actual_root, EXCLUDED_PATHS[0], b"changed parquet")
    _write(actual_root, Path("new_stage/new_plot.png"), b"new figure")

    assert diff_golden_tree(baseline_root, actual_root).is_empty
    assert not diff_directories(baseline_root, actual_root).is_empty


def test_diff_baseline_uses_the_shared_contract(tmp_path: Path) -> None:
    """The committed-baseline entry point compares only selected artifacts."""
    output_root = tmp_path / "output"
    baseline_root = tmp_path / "baseline" / "output"
    _populate_output(output_root)
    _populate_output(baseline_root)
    _write(output_root, EXCLUDED_PATHS[0], b"ignored")

    assert diff_baseline(output_root, tmp_path / "baseline").is_empty

    (output_root / INCLUDED_PATHS[0]).write_bytes(b"changed")
    result = diff_baseline(output_root, tmp_path / "baseline")

    assert result.changed == (INCLUDED_PATHS[0],)


def test_diff_rejects_a_missing_root(tmp_path: Path) -> None:
    """A missing baseline or actual root fails before comparison."""
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileNotFoundError, match="baseline root does not exist"):
        diff_directories(tmp_path / "absent", existing)
    with pytest.raises(FileNotFoundError, match="actual root does not exist"):
        diff_directories(existing, tmp_path / "absent")


def test_compare_trees_is_an_all_file_alias(tmp_path: Path) -> None:
    """The neutral helper compares every file, including excluded artifacts."""
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write(left, Path("only-left.txt"), b"left")
    _write(right, Path("only-right.txt"), b"right")

    result = compare_trees(left, right)

    assert result.added == (Path("only-right.txt"),)
    assert result.removed == (Path("only-left.txt"),)


# --- runner (smoke) --------------------------------------------------------


def test_seed_demo_source_copies_the_committed_table(tmp_path: Path) -> None:
    """Seeding creates the expected demo input below the temporary root."""
    destination = seed_demo_source(tmp_path)

    assert destination == tmp_path / "data" / "demo" / "source_table_1.parquet"
    assert destination.is_file()
    assert destination.stat().st_size > 0


def test_run_demo_pipeline_runs_explicit_stages_then_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner reproduces the entry point's stage order without a real run."""
    calls: list[str] = []

    def record(name: str):
        def stub(*args: object, **kwargs: object) -> None:
            del args, kwargs
            calls.append(name)

        return stub

    import tests.golden.runner as runner_module

    monkeypatch.setattr(runner_module, "s01_outcome", record("s01_outcome"))
    monkeypatch.setattr(
        runner_module,
        "s01b_outcome_analysis",
        record("s01b_outcome_analysis"),
    )
    monkeypatch.setattr(runner_module, "s02_feature", record("s02_feature"))
    monkeypatch.setattr(runner_module, "run_pipeline", record("run_pipeline"))

    output_root = run_demo_pipeline(tmp_path)

    assert output_root == tmp_path / "output"
    assert calls == [
        "s01_outcome",
        "s01b_outcome_analysis",
        "s02_feature",
        "run_pipeline",
    ]
    assert (tmp_path / "data" / "demo" / "source_table_1.parquet").is_file()


def test_temporary_demo_run_cleans_up_its_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The temporary runner yields an output root and removes its project tree."""
    import tests.golden.runner as runner_module

    seen: list[Path] = []

    def fake_run(project_root: Path) -> Path:
        output_root = project_root / "output"
        output_root.mkdir(parents=True)
        seen.append(project_root)
        return output_root

    monkeypatch.setattr(runner_module, "run_demo_pipeline", fake_run)

    with temporary_demo_run() as output_root:
        assert output_root.is_dir()
        assert output_root.parent in seen

    assert seen and not seen[0].exists()
