"""Unit tests for the deterministic demo source-table generator.

Pins ``scripts/make_demo_data.py``: the in-memory table follows the documented
row count, key format, date step, and noise draw order; the split boundaries
leave every partition populated; regeneration into one temporary project root
is byte-identical to regeneration into another; the output path is confined to
``data/demo/`` (including through a symlink); writes are atomic and leave the
previous file in place on failure; and the committed parquet artifact matches
a fresh regeneration exactly.
"""

from __future__ import annotations

import math
import subprocess
import sys
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scripts import make_demo_data as generator
from src.project.config import CONFIG
from src.project.inputs import read_input_table
from src.project.schemas import SOURCE_TABLE_SCHEMA, validate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "make_demo_data.py"
COMMITTED_DEMO_PATH = PROJECT_ROOT / generator.DEFAULT_OUTPUT
DEMO_CASE_COUNT = CONFIG.demo_case_count


def _temporary_project_root(root: Path) -> Path:
    """Create a temporary project root with an empty demo data directory."""
    (root / generator.DEMO_DATA_DIR).mkdir(parents=True)
    return root


def _build(
    case_count: int = DEMO_CASE_COUNT, seed: int = CONFIG.demo_seed
) -> pl.DataFrame:
    """Build a demo frame with this project's seed and row count by default."""
    return generator.build_source_table(seed=seed, case_count=case_count)


# --- table contract --------------------------------------------------------


def test_default_output_is_the_committed_demo_table() -> None:
    assert generator.DEFAULT_OUTPUT == Path("data/demo/source_table_1.parquet")
    assert (
        generator.resolve_output_path(None, project_root=PROJECT_ROOT)
        == COMMITTED_DEMO_PATH
    )


def test_build_source_table_matches_the_documented_shape() -> None:
    frame = _build()

    assert frame.shape == (DEMO_CASE_COUNT, 5)
    assert tuple(frame.columns) == (
        "case_id",
        "case_date",
        "value_1",
        "value_2",
        "value_3",
    )
    assert frame.schema == {
        "case_id": pl.String,
        "case_date": pl.Date,
        "value_1": pl.Float64,
        "value_2": pl.Float64,
        "value_3": pl.Float64,
    }
    assert frame["case_id"][0] == "case_0000"
    assert frame["case_id"][-1] == f"case_{DEMO_CASE_COUNT - 1:04d}"


def test_build_source_table_satisfies_the_source_schema() -> None:
    frame = _build()

    assert validate(frame, SOURCE_TABLE_SCHEMA, "generated") is frame


def test_case_dates_step_forward_by_two_days() -> None:
    dates = _build()["case_date"].to_list()

    assert dates[0] == date(2022, 6, 1)
    assert dates[-1] == date(2022, 6, 1) + timedelta(days=2 * (DEMO_CASE_COUNT - 1))
    assert all(
        later - earlier == timedelta(days=2) for earlier, later in pairwise(dates)
    )


def test_every_configured_partition_is_populated() -> None:
    frame = _build()
    train = frame.filter(pl.col("case_date") <= CONFIG.train_end_date).height
    valid = frame.filter(
        (pl.col("case_date") > CONFIG.train_end_date)
        & (pl.col("case_date") < CONFIG.test_start_date)
    ).height
    test = frame.filter(pl.col("case_date") >= CONFIG.test_start_date).height

    assert min(train, valid, test) > 0
    assert train + valid + test == frame.height


# --- determinism -----------------------------------------------------------


def test_build_source_table_is_deterministic() -> None:
    assert _build().equals(_build())


def test_a_different_seed_changes_only_the_values() -> None:
    baseline = _build()
    other = _build(seed=CONFIG.demo_seed + 1)

    assert baseline["case_id"].equals(other["case_id"])
    assert baseline["case_date"].equals(other["case_date"])
    assert not baseline["value_1"].equals(other["value_1"])
    assert not baseline["value_2"].equals(other["value_2"])
    assert not baseline["value_3"].equals(other["value_3"])


def test_values_follow_the_documented_draw_order() -> None:
    case_count = 3
    random_generator = np.random.default_rng(CONFIG.demo_seed)
    expected: list[tuple[float, float, float]] = []
    for index in range(case_count):
        value_1 = math.sin(index / 7.0) + 0.05 * random_generator.normal()
        value_2 = math.cos(index / 11.0) + 0.05 * random_generator.normal()
        value_3 = value_1 - value_2 + 0.02 * random_generator.normal()
        expected.append((value_1, value_2, value_3))

    frame = _build(case_count=case_count)

    for column, position in (("value_1", 0), ("value_2", 1), ("value_3", 2)):
        assert frame[column].to_list() == pytest.approx(
            [row[position] for row in expected]
        )


def test_build_source_table_rejects_a_non_positive_case_count() -> None:
    with pytest.raises(ValueError, match="case_count"):
        generator.build_source_table(seed=CONFIG.demo_seed, case_count=0)


# --- output path containment ----------------------------------------------


def test_relative_output_resolves_against_the_project_root(tmp_path: Path) -> None:
    root = _temporary_project_root(tmp_path)
    resolved = generator.resolve_output_path(
        Path("data/demo/other.parquet"), project_root=root
    )

    assert resolved == (root / "data/demo/other.parquet").resolve()


def test_output_outside_the_demo_directory_is_rejected(tmp_path: Path) -> None:
    root = _temporary_project_root(tmp_path)

    with pytest.raises(ValueError, match="inside"):
        generator.resolve_output_path(Path("../escape.parquet"), project_root=root)
    with pytest.raises(ValueError, match="inside"):
        generator.resolve_output_path(tmp_path / "elsewhere.parquet", project_root=root)


def test_the_demo_directory_itself_is_rejected(tmp_path: Path) -> None:
    root = _temporary_project_root(tmp_path)

    with pytest.raises(ValueError, match="directory itself"):
        generator.resolve_output_path(Path("data/demo"), project_root=root)


def test_a_symlink_cannot_escape_the_demo_directory(tmp_path: Path) -> None:
    root = _temporary_project_root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / generator.DEMO_DATA_DIR / "link").symlink_to(outside)

    with pytest.raises(ValueError, match="inside"):
        generator.resolve_output_path(
            Path("data/demo/link/source_table_1.parquet"), project_root=root
        )


# --- atomic writes ---------------------------------------------------------


def test_write_source_table_leaves_no_temporary_sibling(tmp_path: Path) -> None:
    root = _temporary_project_root(tmp_path)
    output_path = root / generator.DEFAULT_OUTPUT
    frame = _build(case_count=5)

    generator.write_source_table(frame, output_path)

    assert [path.name for path in output_path.parent.iterdir()] == [output_path.name]
    assert pl.read_parquet(output_path).equals(frame)


def test_a_failed_write_keeps_the_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _temporary_project_root(tmp_path)
    output_path = root / generator.DEFAULT_OUTPUT
    output_path.write_bytes(b"previous")

    def _fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(generator.os, "replace", _fail_replace)

    with pytest.raises(OSError, match="disk full"):
        generator.write_source_table(_build(case_count=5), output_path)

    assert output_path.read_bytes() == b"previous"
    assert [path.name for path in output_path.parent.iterdir()] == [output_path.name]


# --- main ------------------------------------------------------------------


def test_main_regenerates_byte_identical_files(tmp_path: Path) -> None:
    first_root = _temporary_project_root(tmp_path / "first")
    second_root = _temporary_project_root(tmp_path / "second")

    assert generator.main([], project_root=first_root) == 0
    assert generator.main([], project_root=second_root) == 0

    assert (first_root / generator.DEFAULT_OUTPUT).read_bytes() == (
        second_root / generator.DEFAULT_OUTPUT
    ).read_bytes()


def test_main_rejects_an_output_outside_the_demo_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _temporary_project_root(tmp_path / "project")
    outside = tmp_path / "outside.parquet"

    with pytest.raises(SystemExit) as caught:
        generator.main(["--output", str(outside)], project_root=root)

    assert caught.value.code == 2
    assert "inside" in capsys.readouterr().err
    assert not outside.exists()


def test_cli_rejects_a_path_outside_the_repository(tmp_path: Path) -> None:
    outside = tmp_path / "outside.parquet"

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--output", str(outside)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "inside" in completed.stderr
    assert not outside.exists()


def test_committed_artifact_is_reproducible(tmp_path: Path) -> None:
    assert COMMITTED_DEMO_PATH.is_file()
    root = _temporary_project_root(tmp_path)

    assert generator.main([], project_root=root) == 0

    assert (root / generator.DEFAULT_OUTPUT).read_bytes() == (
        COMMITTED_DEMO_PATH.read_bytes()
    )


def test_committed_artifact_reads_through_the_input_registry() -> None:
    frame = read_input_table("source_table_1", PROJECT_ROOT / generator.DEMO_DATA_DIR)

    assert frame.height == DEMO_CASE_COUNT
    assert validate(frame, SOURCE_TABLE_SCHEMA, "committed") is frame
