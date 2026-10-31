"""Project-agnostic serialization helpers shared across pipeline stages.

Two writers cover the two artifact shapes the pipeline emits: a table plus a
head preview, and a full table in both parquet and CSV. Both helpers write
under a caller-supplied output directory and neither creates that directory;
callers ``ensure`` it first, so a missing directory raises instead of being
silently created.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def save_dataframe_to_csv_and_parquet(
    df: pl.DataFrame, filename_stem: str, output_path: Path
) -> None:
    """Write ``df`` as a parquet file plus a CSV preview of its first rows.

    The full frame is written to ``<filename_stem>.parquet`` and ``df.head()``
    to ``<filename_stem>.csv`` under ``output_path``. The directory is not
    created here: the caller creates it first, so this helper has no hidden
    mkdir side effect and a missing directory raises.

    Args:
        df: Frame to serialize.
        filename_stem: Filename without extension; both artifacts derive from
            it.
        output_path: Existing directory that receives the two files.
    """

    output_filepath = output_path / (filename_stem + ".parquet")
    df.write_parquet(output_filepath)

    output_filepath = output_path / (filename_stem + ".csv")
    df.head().write_csv(output_filepath)


def save_full_dataframe_to_parquet_and_csv(
    df: pl.DataFrame, filename_stem: str, output_path: Path
) -> None:
    """Write the full ``df`` to both ``<filename_stem>.parquet`` and ``.csv``.

    This is the lookup-artifact counterpart to
    ``save_dataframe_to_csv_and_parquet``, whose CSV is deliberately only a
    head preview. Both files here carry every row, so the parquet and CSV
    views cannot disagree about the table's contents. As with the preview
    helper, the directory is not created here: a missing directory raises
    instead of being silently created.

    Args:
        df: Frame to serialize in full.
        filename_stem: Filename without extension; both artifacts derive from
            it.
        output_path: Existing directory that receives the two files.
    """

    output_filepath = output_path / (filename_stem + ".parquet")
    df.write_parquet(output_filepath)

    output_filepath = output_path / (filename_stem + ".csv")
    df.write_csv(output_filepath)
