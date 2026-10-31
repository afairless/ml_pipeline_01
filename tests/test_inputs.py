"""Unit tests for the project external-input registry.

The registry declares each logical table once — filename tokens, the columns a
lazy metadata check must see, and the full boundary schema — and resolves it
to exactly one file. These tests pin discovery (exactly one file or raise),
exact-filename overrides, the eager schema-boundary label, the single-table
reader, the lazy metadata-only inspection, and the one-table vocabulary.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

import src.project.inputs
from src.project.inputs import (
    INPUT_TABLES,
    InputTableFilepaths,
    InputTables,
    InputTableSpec,
    inspect_input_table,
    read_input_table,
    resolve_input_files,
    resolve_input_table_path,
)
from src.project.schemas import SOURCE_TABLE_SCHEMA, BoundaryError

SOURCE_FILENAME = "source_table_1.parquet"
CASES = ("case_0000", "case_0001", "case_0002")
DATES = (date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3))


def _source_frame() -> pl.DataFrame:
    """Return a frame valid for the demo source-table schema."""
    return pl.DataFrame(
        {
            "case_id": list(CASES),
            "case_date": list(DATES),
            "value_1": [0.1, 0.2, 0.3],
            "value_2": [-0.1, 0.0, 0.1],
            "value_3": [0.2, 0.2, 0.2],
        }
    )


def _write_source(data_dir: Path, filename: str = SOURCE_FILENAME) -> Path:
    """Write the valid source frame under ``data_dir`` and return its path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / filename
    _source_frame().write_parquet(path)
    return path


# --- Registry declaration --------------------------------------------------


def test_the_demo_logical_name_is_exactly_declared() -> None:
    """The template ships one external table, keyed by its logical name."""
    assert tuple(INPUT_TABLES) == ("source_table_1",)
    spec = INPUT_TABLES["source_table_1"]
    assert spec.logical_name == "source_table_1"
    assert spec.schema is SOURCE_TABLE_SCHEMA
    assert spec.filename_tokens == ("source_table_1",)
    assert spec.required_columns == SOURCE_TABLE_SCHEMA.required_columns


def test_registry_is_read_only() -> None:
    """Resolution never mutates the module-level registry."""
    with pytest.raises(TypeError):
        INPUT_TABLES["other"] = INPUT_TABLES["source_table_1"]  # type: ignore[index]


def test_input_table_spec_is_frozen() -> None:
    """A spec cannot be changed after import."""
    spec = INPUT_TABLES["source_table_1"]
    with pytest.raises(AttributeError):
        setattr(spec, "logical_name", "other")


def test_input_table_spec_rejects_an_empty_logical_name() -> None:
    """A logical table needs a name for its boundary label."""
    with pytest.raises(ValueError, match="logical_name"):
        InputTableSpec(
            logical_name="",
            filename_tokens=("source",),
            required_columns=("case_id",),
            schema=SOURCE_TABLE_SCHEMA,
        )


def test_input_table_spec_rejects_empty_filename_tokens() -> None:
    """With no token, every parquet file would be a candidate."""
    with pytest.raises(ValueError, match="filename token"):
        InputTableSpec(
            logical_name="source",
            filename_tokens=(),
            required_columns=("case_id",),
            schema=SOURCE_TABLE_SCHEMA,
        )


def test_input_table_spec_rejects_empty_required_columns() -> None:
    """The lazy metadata contract cannot be empty."""
    with pytest.raises(ValueError, match="required column"):
        InputTableSpec(
            logical_name="source",
            filename_tokens=("source",),
            required_columns=(),
            schema=SOURCE_TABLE_SCHEMA,
        )


def test_input_table_spec_rejects_a_column_with_no_schema_rule() -> None:
    """Every required column must have a declared dtype."""
    with pytest.raises(ValueError, match="no rule"):
        InputTableSpec(
            logical_name="source",
            filename_tokens=("source",),
            required_columns=("missing",),
            schema=SOURCE_TABLE_SCHEMA,
        )


def test_input_table_spec_rejects_conflicting_tokens() -> None:
    """A token cannot be both required and excluded."""
    with pytest.raises(ValueError, match="requires and excludes"):
        InputTableSpec(
            logical_name="source",
            filename_tokens=("source",),
            required_columns=("case_id",),
            schema=SOURCE_TABLE_SCHEMA,
            excluded_tokens=("source",),
        )


# --- Path discovery --------------------------------------------------------


def test_resolves_one_path_per_logical_name(tmp_path: Path) -> None:
    """Every registered name resolves to the one matching file."""
    path = _write_source(tmp_path)
    (tmp_path / "unrelated.parquet").write_bytes(b"not read")

    resolved = resolve_input_files(tmp_path)

    assert resolved == {"source_table_1": path}
    assert resolved["source_table_1"].is_file()


def test_resolution_ignores_non_parquet_files(tmp_path: Path) -> None:
    """Only parquet files are candidates, matching what the reader opens."""
    _write_source(tmp_path)
    (tmp_path / "source_table_1.csv").write_text("not parquet")

    assert resolve_input_files(tmp_path)["source_table_1"].suffix == ".parquet"


def test_resolution_requires_the_filename_to_contain_every_token(
    tmp_path: Path,
) -> None:
    """A candidate must contain all required tokens, not just one."""
    _write_source(tmp_path, "source.parquet")

    with pytest.raises(FileNotFoundError) as caught:
        resolve_input_files(tmp_path)

    message = str(caught.value)
    assert "source_table_1" in message
    assert "required tokens" in message
    assert "['source_table_1']" in message


def test_zero_matches_raises_naming_the_table_and_tokens(tmp_path: Path) -> None:
    """A missing table names the logical table and its tokens."""
    tmp_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError) as caught:
        resolve_input_files(tmp_path)

    message = str(caught.value)
    assert "source_table_1" in message
    assert "no input file" in message


def test_two_matches_raises_with_both_filenames(tmp_path: Path) -> None:
    """Ambiguous discovery fails instead of silently picking one file."""
    _write_source(tmp_path)
    _write_source(tmp_path, "source_table_1_extra.parquet")

    with pytest.raises(ValueError) as caught:
        resolve_input_files(tmp_path)

    message = str(caught.value)
    assert "source_table_1" in message
    assert SOURCE_FILENAME in message
    assert "source_table_1_extra.parquet" in message


def test_excluded_tokens_disambiguate_candidates(tmp_path: Path) -> None:
    """An excluded token is part of the filename contract."""
    _write_source(tmp_path)
    _write_source(tmp_path, "source_table_1_extra.parquet")
    spec = InputTableSpec(
        logical_name="source_table_1",
        filename_tokens=("source_table_1",),
        required_columns=SOURCE_TABLE_SCHEMA.required_columns,
        schema=SOURCE_TABLE_SCHEMA,
        excluded_tokens=("extra",),
    )

    resolved = src.project.inputs._resolve_input_file(spec, tmp_path, {})

    assert resolved.name == SOURCE_FILENAME


def test_override_wins_over_token_discovery(tmp_path: Path) -> None:
    """An exact-filename override bypasses token discovery."""
    _write_source(tmp_path)
    custom = _write_source(tmp_path, "custom.parquet")

    resolved = resolve_input_files(tmp_path, {"source_table_1": custom.name})

    assert resolved["source_table_1"] == custom


def test_override_matching_missing_file_raises(tmp_path: Path) -> None:
    """An override must name an existing file."""
    _write_source(tmp_path)

    with pytest.raises(FileNotFoundError) as caught:
        resolve_input_files(tmp_path, {"source_table_1": "does-not-exist.parquet"})

    message = str(caught.value)
    assert "source_table_1" in message
    assert "does-not-exist.parquet" in message


def test_override_for_unknown_logical_name_raises(tmp_path: Path) -> None:
    """A typo in an override key cannot silently fall back to discovery."""
    _write_source(tmp_path)

    with pytest.raises(ValueError, match="unknown input table override"):
        resolve_input_files(tmp_path, {"not_a_table": SOURCE_FILENAME})


def test_resolve_input_table_path_returns_the_requested_file(
    tmp_path: Path,
) -> None:
    """The single-table path resolver shares discovery with the all-table one."""
    path = _write_source(tmp_path)

    assert resolve_input_table_path("source_table_1", tmp_path) == path


def test_resolve_input_table_path_unknown_name_raises(tmp_path: Path) -> None:
    """Only registered logical names resolve."""
    _write_source(tmp_path)

    with pytest.raises(ValueError) as caught:
        resolve_input_table_path("not_a_table", tmp_path)

    message = str(caught.value)
    assert "not_a_table" in message
    assert "source_table_1" in message


# --- Eager reads -----------------------------------------------------------


def test_read_input_table_reads_only_the_requested_table(
    tmp_path: Path,
) -> None:
    """The single-table reader does not require the registry's other files."""
    path = _write_source(tmp_path)

    frame = read_input_table("source_table_1", tmp_path)

    assert path.is_file()
    assert frame.equals(_source_frame())


def test_read_input_table_honours_an_exact_filename_override(
    tmp_path: Path,
) -> None:
    """The override is used verbatim by the reader too."""
    _write_source(tmp_path)
    custom = _write_source(tmp_path, "custom.parquet")

    frame = read_input_table(
        "source_table_1", tmp_path, {"source_table_1": custom.name}
    )

    assert frame.equals(_source_frame())


def test_read_input_table_unknown_name_raises(tmp_path: Path) -> None:
    """Reads go through the same registered-name check as path resolution."""
    _write_source(tmp_path)

    with pytest.raises(ValueError) as caught:
        read_input_table("not_a_table", tmp_path)

    message = str(caught.value)
    assert "not_a_table" in message
    assert "source_table_1" in message


def test_read_input_table_schema_failure_is_boundary_labelled(
    tmp_path: Path,
) -> None:
    """A wrong-but-matching file fails with the ingestion boundary label."""
    tmp_path.mkdir(exist_ok=True)
    _source_frame().drop("value_3").write_parquet(tmp_path / SOURCE_FILENAME)

    with pytest.raises(BoundaryError) as caught:
        read_input_table("source_table_1", tmp_path)

    error = caught.value
    assert error.label == "external input table 'source_table_1'"
    assert error.column == "value_3"
    assert error.rule == "presence"
    assert "source_table_1" in str(error)


def test_read_input_table_reports_a_duplicate_case(tmp_path: Path) -> None:
    """Value checks still run on the eager path."""
    tmp_path.mkdir(exist_ok=True)
    duplicate = _source_frame().with_columns(
        pl.when(pl.int_range(pl.len()) == 1)
        .then(pl.lit(CASES[0]))
        .otherwise(pl.col("case_id"))
        .alias("case_id")
    )
    duplicate.write_parquet(tmp_path / SOURCE_FILENAME)

    with pytest.raises(BoundaryError) as caught:
        read_input_table("source_table_1", tmp_path)

    assert caught.value.column == "case_id"
    assert caught.value.rule == "unique"


def test_input_table_filepaths_resolves_every_attribute(tmp_path: Path) -> None:
    """The filepath view exposes one attribute per logical name."""
    path = _write_source(tmp_path)

    filepaths = InputTableFilepaths(tmp_path)

    assert filepaths.source_table_1 == path
    assert filepaths.data_dir == tmp_path


def test_input_tables_reads_and_validates_every_table(tmp_path: Path) -> None:
    """The table view eagerly reads and validates the registry."""
    _write_source(tmp_path)

    tables = InputTables(tmp_path)

    assert tables.source_table_1.equals(_source_frame())


def test_input_tables_reports_the_boundary_label(tmp_path: Path) -> None:
    """The eager all-table view uses the same boundary label as a single read."""
    tmp_path.mkdir(exist_ok=True)
    _source_frame().with_columns(pl.lit(float("nan")).alias("value_1")).write_parquet(
        tmp_path / SOURCE_FILENAME
    )

    with pytest.raises(BoundaryError) as caught:
        InputTables(tmp_path)

    assert caught.value.label == "external input table 'source_table_1'"
    assert caught.value.column == "value_1"
    assert caught.value.rule == "finite"


# --- Lazy inspection -------------------------------------------------------


def test_inspect_input_table_returns_a_lazy_frame_without_reading_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspection checks metadata only and never materializes the table."""
    _write_source(tmp_path)

    def _forbid_eager_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("inspect_input_table must not read rows")

    monkeypatch.setattr(src.project.inputs.pl, "read_parquet", _forbid_eager_read)

    frame = inspect_input_table("source_table_1", tmp_path)

    assert isinstance(frame, pl.LazyFrame)
    assert frame.collect_schema()["value_1"] == pl.Float64


def test_inspect_input_table_rejects_a_missing_required_column(
    tmp_path: Path,
) -> None:
    """The metadata check names the logical table and the missing column."""
    tmp_path.mkdir(exist_ok=True)
    _source_frame().drop("value_2").write_parquet(tmp_path / SOURCE_FILENAME)

    with pytest.raises(ValueError) as caught:
        inspect_input_table("source_table_1", tmp_path)

    message = str(caught.value)
    assert "external input table 'source_table_1'" in message
    assert "missing column(s)" in message
    assert "value_2" in message


def test_inspect_input_table_rejects_a_mismatched_dtype(
    tmp_path: Path,
) -> None:
    """Declared dtypes must match the schema exactly."""
    tmp_path.mkdir(exist_ok=True)
    _source_frame().with_columns(pl.col("value_1").cast(pl.Int64)).write_parquet(
        tmp_path / SOURCE_FILENAME
    )

    with pytest.raises(ValueError) as caught:
        inspect_input_table("source_table_1", tmp_path)

    message = str(caught.value)
    assert "external input table 'source_table_1'" in message
    assert "dtype mismatch" in message
    assert "value_1" in message
    assert "Float64" in message and "Int64" in message


def test_inspect_input_table_allows_extra_declared_columns(
    tmp_path: Path,
) -> None:
    """A wider table remains valid for the registry's narrower contract."""
    tmp_path.mkdir(exist_ok=True)
    _source_frame().with_columns(pl.lit(1.0).alias("extra_measure")).write_parquet(
        tmp_path / SOURCE_FILENAME
    )

    frame = inspect_input_table("source_table_1", tmp_path)

    assert "extra_measure" in frame.collect_schema()


def test_inspect_input_table_unknown_name_raises(tmp_path: Path) -> None:
    """The lazy path shares the registered-name check."""
    _write_source(tmp_path)

    with pytest.raises(ValueError) as caught:
        inspect_input_table("not_a_table", tmp_path)

    message = str(caught.value)
    assert "not_a_table" in message
    assert "source_table_1" in message


def test_inspect_input_table_defers_value_checks_to_the_eager_path(
    tmp_path: Path,
) -> None:
    """A metadata-valid file with bad values is accepted lazily only."""
    tmp_path.mkdir(exist_ok=True)
    bad_values = _source_frame().with_columns(
        pl.lit(None, dtype=pl.String).alias("case_id")
    )
    bad_values.write_parquet(tmp_path / SOURCE_FILENAME)

    lazy = inspect_input_table("source_table_1", tmp_path)

    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect()["case_id"].to_list() == [None, None, None]
    with pytest.raises(BoundaryError):
        read_input_table("source_table_1", tmp_path)
