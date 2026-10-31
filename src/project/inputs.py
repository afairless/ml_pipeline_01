"""External input-table registry and ingestion boundary.

The pipeline reads its external tables from the configured data root. Each
logical table is declared once in :data:`INPUT_TABLES` as an
:class:`InputTableSpec`: the filename tokens a candidate must contain, the
boundary :class:`~src.project.schemas.TableSchema` its contents must satisfy,
and the columns a lazy inspection must be able to see in the parquet metadata.

Resolution is exact-one-or-raise. :func:`resolve_input_files` returns one path
per logical name; zero matches or several matches raise an error naming the
logical table and the tokens that identify it. An optional exact-filename
override bypasses discovery for a data export whose filenames do not follow
the token convention, but an override must still name an existing file.

The registry has two views. :class:`InputTableFilepaths` resolves every
logical name to a path, and :class:`InputTables` reads every table eagerly and
validates it against its schema. A caller that needs one table calls
:func:`read_input_table`, which resolves, reads, and validates only that entry
without requiring the other tables' files to exist. A malformed file therefore
fails at ingestion with a boundary label naming the logical table instead of
reaching a stage.

:func:`inspect_input_table` is the lazy counterpart: it resolves the same
path and checks the parquet's declared schema from file metadata, returning a
``LazyFrame`` without materializing rows. Value, null, uniqueness, and
monotonicity rules cannot be checked from metadata, so they remain the eager
:func:`read_input_table` contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import polars as pl

from src.project.schemas import SOURCE_TABLE_SCHEMA, TableSchema, validate

INPUT_BOUNDARY_PREFIX = "external input table"
"""Boundary-label prefix used when an input file violates its schema."""


@dataclass(frozen=True, slots=True)
class InputTableSpec:
    """One external input table's filename contract and content schema.

    Attributes:
        logical_name: Stable name shared by the registry key, the
            ``InputTableFilepaths`` / ``InputTables`` attribute, and the
            boundary label.
        filename_tokens: Substrings every candidate filename must contain,
            for example ``('source_table_1',)``.
        required_columns: Columns that must be present in the parquet's
            declared metadata for :func:`inspect_input_table`. The eager read
            validates the full :attr:`schema`; this tuple is the smaller
            metadata-only contract.
        schema: The table schema the eager read validates. The table is
            validated at ingestion, so a wrong-but-matching file cannot reach
            a stage.
        excluded_tokens: Substrings a candidate filename must not contain.
            Empty by default; an exclusion keeps two tables with overlapping
            tokens distinguishable.

    Raises:
        ValueError: ``logical_name``, ``filename_tokens``, or
            ``required_columns`` is empty, a required column has no rule in
            :attr:`schema`, or a token is both required and excluded.
    """

    logical_name: str
    filename_tokens: tuple[str, ...]
    required_columns: tuple[str, ...]
    schema: TableSchema
    excluded_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a spec that cannot identify one table unambiguously."""
        if not self.logical_name:
            raise ValueError("an input table spec needs a logical_name")
        if not self.filename_tokens:
            raise ValueError(
                f"input table {self.logical_name!r} needs at least one filename token"
            )
        if not self.required_columns:
            raise ValueError(
                f"input table {self.logical_name!r} needs at least one required column"
            )
        unknown_columns = [
            name for name in self.required_columns if name not in self.schema.columns
        ]
        if unknown_columns:
            raise ValueError(
                f"input table {self.logical_name!r} requires column(s) with no "
                f"rule in its schema: {unknown_columns}"
            )
        overlap = set(self.filename_tokens) & set(self.excluded_tokens)
        if overlap:
            raise ValueError(
                f"input table {self.logical_name!r} requires and excludes the "
                f"same token(s): {sorted(overlap)}"
            )


INPUT_TABLES: Mapping[str, InputTableSpec] = MappingProxyType(
    {
        "source_table_1": InputTableSpec(
            logical_name="source_table_1",
            filename_tokens=("source_table_1",),
            required_columns=SOURCE_TABLE_SCHEMA.required_columns,
            schema=SOURCE_TABLE_SCHEMA,
        ),
    }
)
"""Every external input table, keyed by logical name in resolution order."""


def _describe_tokens(tokens: tuple[str, ...]) -> str:
    """Render a token tuple for an error message."""
    return repr(list(tokens))


def _required_dtypes(spec: InputTableSpec) -> dict[str, type[pl.DataType]]:
    """Return the declared required column -> exact dtype mapping."""
    return {name: spec.schema.columns[name].dtype for name in spec.required_columns}


def _check_declared_schema(declared: pl.Schema, spec: InputTableSpec) -> None:
    """Reject a declared parquet schema that violates the metadata contract.

    Extra columns are allowed, matching the eager validator. The error names
    the logical table, every absent required column, and every dtype mismatch
    so a wrong-but-matching file is diagnosable without reading rows.
    """
    required = _required_dtypes(spec)
    missing = [name for name in required if name not in declared]
    mismatched = [
        f"{name}: expected {required[name]}, found {declared[name]}"
        for name in required
        if name in declared and declared[name] != required[name]
    ]
    if not missing and not mismatched:
        return
    problems: list[str] = []
    if missing:
        problems.append(f"missing column(s) {missing}")
    if mismatched:
        problems.append(f"dtype mismatch(es) {mismatched}")
    raise ValueError(
        f"{INPUT_BOUNDARY_PREFIX} {spec.logical_name!r} declared schema does "
        "not satisfy the registry spec: " + "; ".join(problems)
    )


def _reject_unknown_overrides(overrides: Mapping[str, str]) -> None:
    """Reject override keys that name no registered logical table."""
    unknown_overrides = sorted(set(overrides) - set(INPUT_TABLES))
    if unknown_overrides:
        raise ValueError(
            f"unknown input table override(s): {unknown_overrides}; "
            f"known logical names: {list(INPUT_TABLES)}"
        )


def _resolve_input_file(
    spec: InputTableSpec,
    data_dir: Path,
    overrides: Mapping[str, str],
) -> Path:
    """Resolve exactly one parquet path for a single input-table spec.

    Args:
        spec: The table's filename contract.
        data_dir: Directory holding the external input tables.
        overrides: Already-validated exact-filename overrides.

    Returns:
        The one file matching the spec, or the override's path verbatim.

    Raises:
        ValueError: Discovery matched more than one file.
        FileNotFoundError: Discovery matched no file, or the override names a
            file that does not exist.
    """
    if spec.logical_name in overrides:
        path = Path(data_dir) / overrides[spec.logical_name]
        if not path.is_file():
            raise FileNotFoundError(
                f"input table {spec.logical_name!r} override names a missing "
                f"file: {path}"
            )
        return path

    candidates = sorted(
        path
        for path in Path(data_dir).glob("*.parquet")
        if all(token in path.name for token in spec.filename_tokens)
        and not any(token in path.name for token in spec.excluded_tokens)
    )
    if not candidates:
        raise FileNotFoundError(
            f"no input file for logical table {spec.logical_name!r} in "
            f"{data_dir}: a filename must contain all of the required tokens "
            f"{_describe_tokens(spec.filename_tokens)}"
            + (
                " and none of the excluded tokens "
                f"{_describe_tokens(spec.excluded_tokens)}"
                if spec.excluded_tokens
                else ""
            )
        )
    if len(candidates) > 1:
        raise ValueError(
            f"multiple input files for logical table {spec.logical_name!r} in "
            f"{data_dir}: {[path.name for path in candidates]}; a filename "
            f"must contain all of the required tokens "
            f"{_describe_tokens(spec.filename_tokens)}"
            + (
                " and none of the excluded tokens "
                f"{_describe_tokens(spec.excluded_tokens)}"
                if spec.excluded_tokens
                else ""
            )
            + "; use an explicit exact-filename override to choose one"
        )
    return candidates[0]


def resolve_input_files(
    data_dir: Path,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Resolve exactly one parquet file per registered logical table.

    Candidate files are the ``*.parquet`` files directly inside ``data_dir``
    whose filename contains all of the spec's filename tokens and none of its
    excluded tokens. A logical name in ``overrides`` bypasses token discovery
    and uses the named file verbatim.

    Args:
        data_dir: Directory holding the external input tables.
        overrides: Optional logical name -> exact filename mapping. Unknown
            logical names are rejected so a typo cannot silently fall back to
            token discovery.

    Returns:
        A mapping from every registered logical name to its resolved path, in
        ``INPUT_TABLES`` order.

    Raises:
        ValueError: An override names an unknown logical table, or discovery
            matched more than one file for a logical table.
        FileNotFoundError: Discovery matched no file, or an override names a
            file that does not exist.
    """
    selected = {} if overrides is None else overrides
    _reject_unknown_overrides(selected)
    return {
        spec.logical_name: _resolve_input_file(spec, Path(data_dir), selected)
        for spec in INPUT_TABLES.values()
    }


def resolve_input_table_path(
    logical_name: str,
    data_dir: Path,
    overrides: Mapping[str, str] | None = None,
) -> Path:
    """Resolve exactly one registered input table's parquet path.

    Only the requested table is resolved; the other tables' files need not
    exist.

    Args:
        logical_name: The registered logical table to resolve.
        data_dir: Directory holding the external input tables.
        overrides: Optional logical name -> exact filename mapping.

    Returns:
        The single matching parquet path.

    Raises:
        ValueError: The logical name or an override is unknown, or discovery
            matched more than one file.
        FileNotFoundError: Discovery matched no file, or an override names a
            file that does not exist.
    """
    if logical_name not in INPUT_TABLES:
        raise ValueError(
            f"unknown input table {logical_name!r}; known logical names: "
            f"{list(INPUT_TABLES)}"
        )
    selected = {} if overrides is None else overrides
    _reject_unknown_overrides(selected)
    return _resolve_input_file(INPUT_TABLES[logical_name], Path(data_dir), selected)


def read_input_table(
    logical_name: str,
    data_dir: Path,
    overrides: Mapping[str, str] | None = None,
) -> pl.DataFrame:
    """Resolve, read, and validate exactly one registered input table.

    Only the requested logical table is resolved, so a stage that needs one
    table does not require the other tables' files to exist. Token discovery,
    override handling, and the boundary label are shared with
    :func:`resolve_input_files` and :class:`InputTables`, so a malformed
    external file fails here instead of reaching a stage.

    Args:
        logical_name: The registered logical table to read.
        data_dir: Directory holding the external input tables.
        overrides: Optional logical name -> exact filename mapping.

    Returns:
        The table's frame, validated against its ``InputTableSpec.schema``.

    Raises:
        ValueError: ``logical_name`` is not registered, an override names an
            unknown logical table, or discovery matched more than one file.
        FileNotFoundError: Discovery matched no file, or the override names a
            file that does not exist.
        BoundaryError: The file contents violate the spec's schema.
    """
    path = resolve_input_table_path(logical_name, data_dir, overrides)
    return _read_input_table(path, INPUT_TABLES[logical_name])


def _read_input_table(path: Path, spec: InputTableSpec) -> pl.DataFrame:
    """Read one input parquet and validate it at the ingestion boundary."""
    frame = pl.read_parquet(path)
    return validate(
        frame,
        spec.schema,
        f"{INPUT_BOUNDARY_PREFIX} {spec.logical_name!r}",
    )


def inspect_input_table(
    logical_name: str,
    data_dir: Path,
    overrides: Mapping[str, str] | None = None,
) -> pl.LazyFrame:
    """Resolve one registered input table and scan it without reading rows.

    The parquet's **declared** schema from file metadata must contain every
    column in the spec's ``required_columns`` with its exact dtype; extra
    columns are allowed. The frame is returned as a ``LazyFrame``, so no rows
    are materialized here.

    The check is metadata-only by design: values, nullability, uniqueness, and
    ordering cannot be proven without reading data, so those checks remain the
    eager :func:`read_input_table` / :class:`InputTables` contract.

    Args:
        logical_name: The registered logical table to inspect.
        data_dir: Directory holding the external input tables.
        overrides: Optional logical name -> exact filename mapping.

    Returns:
        A ``LazyFrame`` over the resolved table.

    Raises:
        ValueError: ``logical_name`` is not registered, an override is
            unknown, discovery matched more than one file, or the declared
            schema omits a required column or carries a mismatched dtype.
        FileNotFoundError: Discovery matched no file, or the override names a
            file that does not exist.
    """
    path = resolve_input_table_path(logical_name, data_dir, overrides)
    frame = pl.scan_parquet(path)
    _check_declared_schema(frame.collect_schema(), INPUT_TABLES[logical_name])
    return frame


@dataclass(frozen=True, slots=True)
class InputTableFilepaths:
    """Resolved parquet path per external input table.

    Every path comes from :func:`resolve_input_files`, so token discovery and
    override handling have one implementation.

    Attributes:
        data_dir: Directory holding the external input tables.
        overrides: Optional exact-filename overrides per logical name.
        source_table_1: Path to the source table.
    """

    data_dir: Path
    overrides: Mapping[str, str] | None = None
    source_table_1: Path = field(init=False)

    def __post_init__(self) -> None:
        """Resolve every logical table path through the registry."""
        filepaths = resolve_input_files(self.data_dir, self.overrides)
        for logical_name, path in filepaths.items():
            object.__setattr__(self, logical_name, path)


@dataclass(frozen=True, slots=True)
class InputTables:
    """Every external input table read and schema-validated as a frame.

    Each parquet is read through the path registry and validated against its
    ``InputTableSpec.schema``, so a wrong-but-matching file fails at ingestion
    with a boundary label instead of reaching a stage.

    Attributes:
        data_dir: Directory holding the external input tables.
        overrides: Optional exact-filename overrides per logical name.
        source_table_1: The source table.
    """

    data_dir: Path
    overrides: Mapping[str, str] | None = None
    source_table_1: pl.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        """Read and validate every registered input table."""
        filepaths = InputTableFilepaths(self.data_dir, self.overrides)
        for logical_name, spec in INPUT_TABLES.items():
            path = getattr(filepaths, logical_name)
            object.__setattr__(self, logical_name, _read_input_table(path, spec))
