"""Unit tests for the core serialization helpers.

Pins the invariant that ``save_dataframe_to_csv_and_parquet`` writes both
``<stem>.parquet`` and a ``<stem>.csv`` preview under the given output path,
and does not create that directory itself: a missing directory raises instead
of being silently created. Also pins the full-table counterpart, whose CSV
carries every row rather than a head preview. A failure here means an artifact
could be written to the wrong location, a preview could be dropped, or a
lookup CSV could silently lose rows without a stage noticing.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.core.io import (
    save_dataframe_to_csv_and_parquet,
    save_full_dataframe_to_parquet_and_csv,
)


def test_save_dataframe_to_csv_and_parquet_writes_both_artifacts(
    tmp_path: Path,
) -> None:
    frame = pl.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7]})
    save_dataframe_to_csv_and_parquet(frame, "results", tmp_path)

    parquet_path = tmp_path / "results.parquet"
    csv_path = tmp_path / "results.csv"

    assert parquet_path.is_file()
    assert csv_path.is_file()

    roundtrip = pl.read_parquet(parquet_path)
    assert roundtrip.equals(frame)

    csv_head = pl.read_csv(csv_path)
    assert csv_head.height == 5
    assert csv_head.equals(frame.head())


def test_save_full_dataframe_writes_every_row_to_both_formats(
    tmp_path: Path,
) -> None:
    """The full-table helper keeps all rows in parquet and CSV alike."""
    frame = pl.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7]})
    save_full_dataframe_to_parquet_and_csv(frame, "results", tmp_path)

    parquet_path = tmp_path / "results.parquet"
    csv_path = tmp_path / "results.csv"

    assert parquet_path.is_file()
    assert csv_path.is_file()

    parquet_frame = pl.read_parquet(parquet_path)
    csv_frame = pl.read_csv(csv_path)
    assert parquet_frame.equals(frame)
    assert csv_frame.equals(frame)
    assert csv_frame.height == frame.height == 7


def test_save_full_dataframe_csv_differs_from_the_head_preview(
    tmp_path: Path,
) -> None:
    """The full CSV keeps the rows the preview helper deliberately drops."""
    frame = pl.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7]})
    save_dataframe_to_csv_and_parquet(frame, "preview", tmp_path)
    save_full_dataframe_to_parquet_and_csv(frame, "full", tmp_path)

    preview_height = pl.read_csv(tmp_path / "preview.csv").height
    full_height = pl.read_csv(tmp_path / "full.csv").height
    assert preview_height == 5
    assert full_height == frame.height
    assert full_height > preview_height


def test_save_full_dataframe_writes_empty_frame_to_both_formats(
    tmp_path: Path,
) -> None:
    """An empty frame still writes both artifacts with its header intact."""
    frame = pl.DataFrame({"a": []})
    save_full_dataframe_to_parquet_and_csv(frame, "empty", tmp_path)

    parquet_path = tmp_path / "empty.parquet"
    csv_path = tmp_path / "empty.csv"
    assert parquet_path.is_file()
    assert csv_path.is_file()
    assert pl.read_parquet(parquet_path).height == 0
    assert csv_path.read_text().splitlines() == ["a"]


def test_save_full_dataframe_writes_all_null_frame_to_both_formats(
    tmp_path: Path,
) -> None:
    """An all-null frame keeps its row count and null cells in both formats."""
    frame = pl.DataFrame({"a": [None, None], "b": [None, None]})
    save_full_dataframe_to_parquet_and_csv(frame, "nulls", tmp_path)

    parquet_path = tmp_path / "nulls.parquet"
    csv_path = tmp_path / "nulls.csv"
    assert parquet_path.is_file()
    assert csv_path.is_file()

    parquet_frame = pl.read_parquet(parquet_path)
    assert parquet_frame.height == frame.height == 2
    assert parquet_frame.columns == frame.columns
    assert parquet_frame.null_count().sum_horizontal().item() == 4
    assert len(csv_path.read_text().splitlines()) == 3


def test_save_full_dataframe_does_not_create_the_output_directory(
    tmp_path: Path,
) -> None:
    """A missing output directory raises instead of being silently created."""
    missing_dir = tmp_path / "does_not_exist"
    frame = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(Exception):
        save_full_dataframe_to_parquet_and_csv(frame, "results", missing_dir)

    assert not missing_dir.exists()


def test_save_preview_does_not_create_the_output_directory(
    tmp_path: Path,
) -> None:
    """The preview helper raises for a missing directory and does not mkdir."""
    missing_dir = tmp_path / "also_missing"
    frame = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(Exception):
        save_dataframe_to_csv_and_parquet(frame, "results", missing_dir)

    assert not missing_dir.exists()
