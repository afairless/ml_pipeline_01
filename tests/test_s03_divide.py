"""Integration and property-based tests for the ``s03_divide`` stage.

The stage joins one modeled outcome to the feature matrix, splits the joined
rows on ``case_date``, validates every partition at the split boundary, and
writes the two head frames, the three partition parquets, and a statistics text
under a per-outcome directory.

The integration tests drive the whole path on a temporary project root seeded
with the committed demo table and the real s01/s02 stages, so they pin the
on-disk contract: the declared artifact set, the exact partition membership for
the configured boundaries, the per-outcome column set, disjointness,
exhaustiveness, non-emptiness, head-frame columns, the deterministic
statistics text, repeat-run stability, and the failure cases that must leave no
artifact behind (missing prerequisites, invalid boundaries, an empty partition,
a key mismatch, and a partition that violates its schema).

The property-based tests pin the split algebra and the join contract over
generated inputs: every row belongs to exactly one partition, the boundary
dates fall on the declared side, the generated partitions are non-empty and
exhaustive, and the key join conserves every row and preserves the declared
column order.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import src.stages.s03_divide_data as stage_module
from src.core.paths import PipelineContext
from src.project.adapters import build_feature_table, build_outcome_table
from src.project.artifacts import load_split_tables
from src.project.columns import CASE_DATE, CASE_ID, KEY_COLUMNS
from src.project.config import CONFIG, ProjectConfig
from src.project.inputs import read_input_table
from src.project.outcomes import OUTCOME_COMPARISONS, modeled_outcomes
from src.project.pipeline import build_context
from src.project.schemas import (
    FEATURE_COLUMNS,
    S03_SPLIT_SCHEMA,
    BoundaryError,
    validate,
)
from src.stages.s01_outcome import s01_outcome
from src.stages.s02_feature import s02_feature
from src.stages.s03_divide_data import (
    HEAD_ROW_COUNT,
    OUTCOME_BOUNDARY_LABEL,
    SPLIT_BOUNDARY_LABEL,
    assert_date_boundaries,
    join_outcome_and_features,
    main,
    partition_statistics_text,
    s03_divide,
    split_by_case_date,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Repository root, used only to locate the committed demo input table."""

SOURCE_FILENAME = "source_table_1.parquet"
"""Committed demo filename, copied into each temporary project root."""

MODELED_OUTCOME_NAME = modeled_outcomes()[0].name
"""The demo outcome the stage's own tests split."""

ANALYSIS_OUTCOME_NAME = OUTCOME_COMPARISONS[0].right
"""A second registered outcome, used to prove the directories stay separate."""

DECLARED_ARTIFACT_NAMES = sorted(
    [
        "feature_head_df.parquet",
        "outcome_head_df.parquet",
        "statistics.txt",
        "test_df.parquet",
        "train_df.parquet",
        "valid_df.parquet",
    ]
)
"""Every file the s03 stage table declares for one outcome."""

PARTITION_NAMES = ("train", "valid", "test")


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Return a temporary project root containing the demo source table."""
    data_dir = tmp_path / CONFIG.data_root
    data_dir.mkdir(parents=True)
    shutil.copyfile(
        PROJECT_ROOT / CONFIG.data_root / SOURCE_FILENAME,
        data_dir / SOURCE_FILENAME,
    )
    return tmp_path


@pytest.fixture
def context(project_root: Path) -> PipelineContext:
    """Return the pipeline context bound to the temporary project root."""
    return build_context(config=CONFIG, project_root=project_root)


def _source_df(project_root: Path) -> pl.DataFrame:
    """Read the temporary project's demo source table."""
    return read_input_table("source_table_1", project_root / CONFIG.data_root)


def _expected_joined_df(
    project_root: Path,
    outcome_name: str,
) -> pl.DataFrame:
    """Rebuild the joined frame through the same public adapters."""
    source_df = _source_df(project_root)
    outcome_df = build_outcome_table(source_df, CONFIG)
    feature_df = build_feature_table(source_df, outcome_df, CONFIG)
    return join_outcome_and_features(feature_df, outcome_df, outcome_name)


def _seed_prerequisites(
    context: PipelineContext,
    output_root: Path,
) -> None:
    """Run the s01 and s02 stages so s03 has both inputs."""
    s01_outcome(context, output_root)
    s02_feature(context, output_root)


def _run(
    project_root: Path,
    context: PipelineContext,
    outcome_name: str = MODELED_OUTCOME_NAME,
) -> Path:
    """Seed the prerequisites, run s03, and return the outcome directory."""
    output_root = project_root / "output"
    _seed_prerequisites(context, output_root)
    main(outcome_name, context, output_root)
    return output_root / "s03_divide" / outcome_name


def _read_partitions(stage_dir: Path) -> dict[str, pl.DataFrame]:
    """Read the three partition parquets from an outcome directory."""
    return {
        name: pl.read_parquet(stage_dir / f"{name}_df.parquet")
        for name in PARTITION_NAMES
    }


# --- successful stage run --------------------------------------------------


def test_stage_writes_exactly_the_declared_artifacts(
    project_root: Path,
    context: PipelineContext,
) -> None:
    stage_dir = _run(project_root, context)

    assert sorted(path.name for path in stage_dir.iterdir()) == DECLARED_ARTIFACT_NAMES
    assert all((stage_dir / name).is_file() for name in DECLARED_ARTIFACT_NAMES)


def test_stage_writes_the_expected_partitions(
    project_root: Path,
    context: PipelineContext,
) -> None:
    stage_dir = _run(project_root, context)
    expected = _expected_joined_df(project_root, MODELED_OUTCOME_NAME)
    boundaries = {
        "train": pl.col(CASE_DATE) <= CONFIG.train_end_date,
        "valid": (pl.col(CASE_DATE) > CONFIG.train_end_date)
        & (pl.col(CASE_DATE) < CONFIG.test_start_date),
        "test": pl.col(CASE_DATE) >= CONFIG.test_start_date,
    }

    for name, frame in _read_partitions(stage_dir).items():
        assert frame.equals(expected.filter(boundaries[name]))
        assert validate(frame, S03_SPLIT_SCHEMA, f"written {name}") is frame


def test_each_partition_carries_keys_features_and_one_outcome(
    project_root: Path,
    context: PipelineContext,
) -> None:
    stage_dir = _run(project_root, context)
    expected_columns = (CASE_ID, CASE_DATE, *FEATURE_COLUMNS, MODELED_OUTCOME_NAME)

    for frame in _read_partitions(stage_dir).values():
        assert tuple(frame.columns) == expected_columns
        assert MODELED_OUTCOME_NAME in frame.columns
        assert (
            sum(
                name in frame.columns
                for name in ("outcome_1", "outcome_2", "outcome_3", "outcome_4")
            )
            == 1
        )
        assert frame.height > 0


def test_partitions_are_disjoint_exhaustive_and_non_empty(
    project_root: Path,
    context: PipelineContext,
) -> None:
    stage_dir = _run(project_root, context)
    expected = _expected_joined_df(project_root, MODELED_OUTCOME_NAME)
    partitions = _read_partitions(stage_dir)

    key_sets = {
        name: set(frame.select(list(KEY_COLUMNS)).iter_rows())
        for name, frame in partitions.items()
    }
    for left, right in (
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ):
        assert key_sets[left].isdisjoint(key_sets[right])
    assert set().union(*key_sets.values()) == set(
        expected.select(list(KEY_COLUMNS)).iter_rows()
    )
    assert sum(frame.height for frame in partitions.values()) == expected.height
    assert all(frame.height > 0 for frame in partitions.values())


def test_split_boundaries_are_respected_on_the_demo_table(
    project_root: Path,
    context: PipelineContext,
) -> None:
    stage_dir = _run(project_root, context)
    partitions = _read_partitions(stage_dir)
    train_dates = partitions["train"][CASE_DATE]
    valid_dates = partitions["valid"][CASE_DATE]
    test_dates = partitions["test"][CASE_DATE]

    assert bool((train_dates <= CONFIG.train_end_date).all())
    assert bool((valid_dates > CONFIG.train_end_date).all())
    assert bool((valid_dates < CONFIG.test_start_date).all())
    assert bool((test_dates >= CONFIG.test_start_date).all())


def test_head_frames_name_the_partition_columns(
    project_root: Path,
    context: PipelineContext,
) -> None:
    stage_dir = _run(project_root, context)
    joined = _expected_joined_df(project_root, MODELED_OUTCOME_NAME)
    feature_head = pl.read_parquet(stage_dir / "feature_head_df.parquet")
    outcome_head = pl.read_parquet(stage_dir / "outcome_head_df.parquet")

    assert feature_head.equals(
        joined.select([*KEY_COLUMNS, *FEATURE_COLUMNS]).head(HEAD_ROW_COUNT)
    )
    assert outcome_head.equals(
        joined.select([*KEY_COLUMNS, MODELED_OUTCOME_NAME]).head(HEAD_ROW_COUNT)
    )
    assert tuple(outcome_head.columns) == (
        CASE_ID,
        CASE_DATE,
        MODELED_OUTCOME_NAME,
    )

    tables = load_split_tables(
        context.layout,
        MODELED_OUTCOME_NAME,
        project_root / "output",
    )
    assert tables.feature_head_df.equals(feature_head)
    assert tables.outcome_head_df.equals(outcome_head)
    assert tables.train_df.equals(pl.read_parquet(stage_dir / "train_df.parquet"))


def test_statistics_text_is_deterministic_and_names_each_partition(
    project_root: Path,
    context: PipelineContext,
) -> None:
    stage_dir = _run(project_root, context)
    partitions = _read_partitions(stage_dir)
    expected = partition_statistics_text(
        MODELED_OUTCOME_NAME,
        partitions,
        CONFIG.train_end_date,
        CONFIG.test_start_date,
    )
    written = (stage_dir / "statistics.txt").read_text(encoding="utf-8")

    assert written == expected
    assert written.startswith("Split statistics\n")
    assert f"outcome: {MODELED_OUTCOME_NAME}\n" in written
    assert f"train_end_date: {CONFIG.train_end_date}\n" in written
    assert f"test_start_date: {CONFIG.test_start_date}\n" in written
    assert f"rows: {CONFIG.demo_case_count}\n" in written
    for name in PARTITION_NAMES:
        assert any(line.startswith(f"{name} ") for line in written.splitlines())


def test_each_outcome_directory_carries_only_its_own_outcome(
    project_root: Path,
    context: PipelineContext,
) -> None:
    modeled_dir = _run(project_root, context, MODELED_OUTCOME_NAME)
    analysis_dir = _run(project_root, context, ANALYSIS_OUTCOME_NAME)

    assert modeled_dir != analysis_dir
    assert tuple(pl.read_parquet(modeled_dir / "train_df.parquet").columns)[-1] == (
        MODELED_OUTCOME_NAME
    )
    assert tuple(pl.read_parquet(analysis_dir / "train_df.parquet").columns)[-1] == (
        ANALYSIS_OUTCOME_NAME
    )
    assert sorted(path.name for path in modeled_dir.iterdir()) == (
        DECLARED_ARTIFACT_NAMES
    )
    assert sorted(path.name for path in analysis_dir.iterdir()) == (
        DECLARED_ARTIFACT_NAMES
    )


def test_rerunning_the_stage_replaces_the_same_artifacts(
    project_root: Path,
    context: PipelineContext,
) -> None:
    stage_dir = _run(project_root, context)
    first_bytes = {
        name: (stage_dir / name).read_bytes() for name in DECLARED_ARTIFACT_NAMES
    }

    rerun_dir = _run(project_root, context)

    assert rerun_dir == stage_dir
    assert sorted(path.name for path in stage_dir.iterdir()) == DECLARED_ARTIFACT_NAMES
    assert {
        name: (stage_dir / name).read_bytes() for name in DECLARED_ARTIFACT_NAMES
    } == first_bytes


def test_stage_entry_point_is_exported_under_its_stage_name() -> None:
    assert s03_divide is main


# --- failure cases ---------------------------------------------------------


def test_missing_prerequisites_fail_before_writing(
    project_root: Path,
    context: PipelineContext,
) -> None:
    output_root = project_root / "output"

    with pytest.raises(FileNotFoundError):
        main(MODELED_OUTCOME_NAME, context, output_root)

    stage_dir = output_root / "s03_divide" / MODELED_OUTCOME_NAME
    assert not any(stage_dir.iterdir())


def test_invalid_boundaries_fail_before_reading_artifacts(
    project_root: Path,
    context: PipelineContext,
) -> None:
    config = ProjectConfig(
        train_end_date=date(2024, 7, 1),
        test_start_date=date(2024, 7, 1),
    )
    invalid_context = build_context(config=config, project_root=project_root)

    with pytest.raises(ValueError, match="strictly after train_end_date"):
        main(MODELED_OUTCOME_NAME, invalid_context, project_root / "output")

    assert not (project_root / "output" / "s03_divide").exists()


def test_empty_partition_fails_before_writing(
    project_root: Path,
    context: PipelineContext,
) -> None:
    config = ProjectConfig(
        train_end_date=date(2023, 1, 1),
        test_start_date=date(2023, 1, 2),
    )
    output_root = project_root / "output"
    empty_context = build_context(config=config, project_root=project_root)
    s01_outcome(empty_context, output_root)
    s02_feature(empty_context, output_root)

    with pytest.raises(ValueError, match="valid partition is empty"):
        main(MODELED_OUTCOME_NAME, empty_context, output_root)

    stage_dir = output_root / "s03_divide" / MODELED_OUTCOME_NAME
    assert not any(stage_dir.iterdir())


def test_key_mismatch_fails_before_writing(
    project_root: Path,
    context: PipelineContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = stage_module._read_feature_table

    def truncated(
        context_: PipelineContext,
        output_root: Path,
    ) -> pl.DataFrame:
        return original(context_, output_root).head(CONFIG.demo_case_count - 1)

    monkeypatch.setattr(stage_module, "_read_feature_table", truncated)

    with pytest.raises(ValueError, match="do not carry the same keys"):
        _run(project_root, context)

    stage_dir = project_root / "output" / "s03_divide" / MODELED_OUTCOME_NAME
    assert not any(stage_dir.iterdir())


def test_partition_schema_violation_fails_before_writing(
    project_root: Path,
    context: PipelineContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = stage_module.split_by_case_date

    def malformed(
        joined_df: pl.DataFrame,
        train_end_date: date,
        test_start_date: date,
    ) -> dict[str, pl.DataFrame]:
        partitions = original(joined_df, train_end_date, test_start_date)
        partitions["train"] = partitions["train"].with_columns(
            pl.lit(float("nan")).alias(MODELED_OUTCOME_NAME)
        )
        return partitions

    monkeypatch.setattr(stage_module, "split_by_case_date", malformed)

    with pytest.raises(BoundaryError) as caught:
        _run(project_root, context)

    error = caught.value
    assert error.label == SPLIT_BOUNDARY_LABEL.format(partition="train")
    assert error.column == MODELED_OUTCOME_NAME
    assert error.rule == "finite"
    stage_dir = project_root / "output" / "s03_divide" / MODELED_OUTCOME_NAME
    assert not any(stage_dir.iterdir())


def test_outcome_boundary_label_is_used_for_a_bad_outcome_artifact(
    project_root: Path,
    context: PipelineContext,
) -> None:
    output_root = project_root / "output"
    _seed_prerequisites(context, output_root)
    outcome_path = output_root / "s01_outcome" / "outcome_df.parquet"
    outcome_df = pl.read_parquet(outcome_path)
    outcome_df.with_columns(pl.lit(float("inf")).alias("outcome_1")).write_parquet(
        outcome_path
    )

    with pytest.raises(BoundaryError) as caught:
        main(MODELED_OUTCOME_NAME, context, output_root)

    assert caught.value.label == OUTCOME_BOUNDARY_LABEL
    assert caught.value.rule == "finite"


# --- property-based split algebra -----------------------------------------


@st.composite
def _split_cases(
    draw: st.DrawFn,
) -> tuple[list[date], date, date]:
    """Draw unique sorted dates plus boundaries that keep all partitions full."""
    dates = draw(
        st.lists(
            st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 12, 31)),
            min_size=3,
            max_size=30,
            unique=True,
        ).map(sorted)
    )
    train_index = draw(st.integers(min_value=0, max_value=len(dates) - 3))
    test_index = draw(st.integers(min_value=train_index + 2, max_value=len(dates) - 1))
    return dates, dates[train_index], dates[test_index]


def _frame_for_dates(dates: list[date]) -> pl.DataFrame:
    """Build one joined-shape frame with unique keys and finite payloads."""
    width = len(dates)
    return pl.DataFrame(
        {
            CASE_ID: [f"case_{index:04d}" for index in range(width)],
            CASE_DATE: pl.Series(dates, dtype=pl.Date),
            **{
                name: [float(index + column) for index in range(width)]
                for column, name in enumerate(FEATURE_COLUMNS, start=1)
            },
            MODELED_OUTCOME_NAME: [float(index) - 0.5 for index in range(width)],
        }
    )


@given(split_case=_split_cases())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_every_generated_row_belongs_to_exactly_one_partition(
    split_case: tuple[list[date], date, date],
) -> None:
    dates, train_end_date, test_start_date = split_case
    frame = _frame_for_dates(dates)

    partitions = split_by_case_date(frame, train_end_date, test_start_date)

    assert tuple(partitions) == PARTITION_NAMES
    assert all(partition.height > 0 for partition in partitions.values())
    assert sum(partition.height for partition in partitions.values()) == len(dates)
    key_sets = {
        name: set(partition.select(list(KEY_COLUMNS)).iter_rows())
        for name, partition in partitions.items()
    }
    assert (
        key_sets["train"].isdisjoint(key_sets["valid"])
        and key_sets["train"].isdisjoint(key_sets["test"])
        and key_sets["valid"].isdisjoint(key_sets["test"])
    )
    assert set().union(*key_sets.values()) == set(
        frame.select(list(KEY_COLUMNS)).iter_rows()
    )


@given(split_case=_split_cases())
@settings(max_examples=50, deadline=None)
def test_generated_boundary_dates_land_on_the_declared_side(
    split_case: tuple[list[date], date, date],
) -> None:
    dates, train_end_date, test_start_date = split_case
    partitions = split_by_case_date(
        _frame_for_dates(dates),
        train_end_date,
        test_start_date,
    )
    train_dates = partitions["train"][CASE_DATE].to_list()
    valid_dates = partitions["valid"][CASE_DATE].to_list()
    test_dates = partitions["test"][CASE_DATE].to_list()

    assert train_dates == [value for value in dates if value <= train_end_date]
    assert valid_dates == [
        value for value in dates if train_end_date < value < test_start_date
    ]
    assert test_dates == [value for value in dates if value >= test_start_date]
    assert train_dates[-1] == train_end_date
    assert test_dates[0] == test_start_date


@given(
    train_end_date=st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 12, 31)),
    test_start_date=st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 12, 31)),
)
@settings(max_examples=50, deadline=None)
def test_assert_date_boundaries_accepts_exactly_the_ordered_ranges(
    train_end_date: date,
    test_start_date: date,
) -> None:
    if test_start_date > train_end_date:
        assert_date_boundaries(train_end_date, test_start_date)
    else:
        with pytest.raises(ValueError, match="strictly after train_end_date"):
            assert_date_boundaries(train_end_date, test_start_date)


# --- property-based join contract -----------------------------------------

_FINITE_FLOAT = st.floats(
    min_value=-1_000_000.0,
    max_value=1_000_000.0,
    allow_nan=False,
    allow_infinity=False,
)
"""Finite payload values, matching the numeric schema contract."""


@given(
    values=st.lists(_FINITE_FLOAT, min_size=1, max_size=20),
)
@settings(max_examples=50, deadline=None)
def test_join_conserves_rows_and_preserves_column_order(
    values: list[float],
) -> None:
    width = len(values)
    dates = [date(2024, 1, 1 + index) for index in range(width)]
    keys = pl.DataFrame(
        {
            CASE_ID: [f"case_{index:04d}" for index in reversed(range(width))],
            CASE_DATE: pl.Series(list(reversed(dates)), dtype=pl.Date),
        }
    )
    feature_df = keys.hstack(
        pl.DataFrame(
            {
                name: [value + offset for value in values]
                for offset, name in enumerate(FEATURE_COLUMNS)
            }
        )
    )
    outcome_df = keys.select(list(KEY_COLUMNS)).with_columns(
        pl.Series(MODELED_OUTCOME_NAME, values, dtype=pl.Float64)
    )

    joined = join_outcome_and_features(feature_df, outcome_df, MODELED_OUTCOME_NAME)

    assert joined.height == width
    assert tuple(joined.columns) == (
        CASE_ID,
        CASE_DATE,
        *FEATURE_COLUMNS,
        MODELED_OUTCOME_NAME,
    )
    assert joined.select(list(KEY_COLUMNS)).equals(
        keys.select(list(KEY_COLUMNS)).sort(list(KEY_COLUMNS))
    )
    by_case = {row[CASE_ID]: row for row in joined.iter_rows(named=True)}
    for index, value in enumerate(values):
        case_id = f"case_{width - 1 - index:04d}"
        assert by_case[case_id][MODELED_OUTCOME_NAME] == value
        assert by_case[case_id][FEATURE_COLUMNS[0]] == value


@given(values=st.lists(_FINITE_FLOAT, min_size=2, max_size=10))
@settings(max_examples=30, deadline=None)
def test_join_rejects_a_missing_key_instead_of_dropping_it(
    values: list[float],
) -> None:
    width = len(values)
    frame = pl.DataFrame(
        {
            CASE_ID: [f"case_{index:04d}" for index in range(width)],
            CASE_DATE: pl.Series(
                [date(2024, 1, 1 + index) for index in range(width)],
                dtype=pl.Date,
            ),
            **{name: [value for value in values] for name in FEATURE_COLUMNS},
            MODELED_OUTCOME_NAME: values,
        }
    )
    feature_df = frame.select([*KEY_COLUMNS, *FEATURE_COLUMNS])
    outcome_df = frame.select([*KEY_COLUMNS, MODELED_OUTCOME_NAME]).head(width - 1)

    with pytest.raises(ValueError, match="do not carry the same keys"):
        join_outcome_and_features(feature_df, outcome_df, MODELED_OUTCOME_NAME)
