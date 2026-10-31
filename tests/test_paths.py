"""Unit and property tests for the core path layout types.

Pins the invariant that ``src/core/paths.py`` resolves declared stage
directories and artifacts purely: constructing a ``PipelineLayout`` or
resolving a path never creates a directory, never depends on the current
working directory, and never interpolates an undeclared outcome into a
path. A regression here means a stage would resolve or create its output
outside the location the stage table declares — the failure each assertion
catches.

The property tests assert those invariants over the full declared input
space (every stage x every outcome) instead of sampled examples; the
project pins no hypothesis dependency for this layer, so the space is
enumerated with ``@pytest.mark.parametrize``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.core.paths import PipelineLayout, StageSpec

# A stage table shaped like the project layout: these specs exercise both
# static and per-outcome templates.
S01_OUTCOME = StageSpec(
    name="s01_outcome",
    dir_template="s01_outcome",
    artifacts={
        "outcome_df": "outcome_df.parquet",
        "outcome_statistics_df": "outcome_statistics_df.txt",
    },
)
S03_DIVIDE = StageSpec(
    name="s03_divide",
    dir_template="s03_divide/{outcome}",
    artifacts={
        "feature_head_df": "feature_head_df.parquet",
        "train_df": "train_df.parquet",
        "test_df": "test_df.parquet",
    },
)
S05_PREDICT = StageSpec(
    name="s05_predict",
    dir_template="s05_predict/{outcome}",
    artifacts={"predictions_df": "predictions_df.parquet"},
)
S06_EVALUATE = StageSpec(
    name="s06_evaluate",
    dir_template="s06_evaluate/{outcome}",
    artifacts={"display_table": "{outcome}_display.csv"},
)

STAGES = {
    spec.name: spec for spec in (S01_OUTCOME, S03_DIVIDE, S05_PREDICT, S06_EVALUATE)
}

# The outcome names the project registry declares. Kept local so this core
# test does not depend on the project layer.
OUTCOMES = frozenset({"outcome_1", "outcome_3"})

ALL_STAGES = (S01_OUTCOME, S03_DIVIDE, S05_PREDICT, S06_EVALUATE)


@pytest.fixture
def layout() -> PipelineLayout:
    return PipelineLayout(stages=STAGES, outcome_names=OUTCOMES)


@pytest.fixture
def outcome_layout() -> PipelineLayout:
    """A layout that declares no outcomes: every outcome is unknown."""
    return PipelineLayout(stages=STAGES)


# --- StageSpec: declaration and frozenness ---------------------------------


def test_stage_spec_stores_declared_fields() -> None:
    spec = S01_OUTCOME
    assert spec.name == "s01_outcome"
    assert spec.dir_template == "s01_outcome"
    assert spec.artifacts["outcome_df"] == "outcome_df.parquet"


def test_stage_spec_defaults_prerequisites_to_empty() -> None:
    """A spec without declared dependencies carries no prerequisites: the
    s01/s02-style source stages declare none."""
    spec = StageSpec(name="s02_feature", dir_template="s02_feature", artifacts={})
    assert spec.prerequisites == ()


def test_stage_spec_stores_prerequisites() -> None:
    spec = StageSpec(
        name="s04_train",
        dir_template="s04_train/{outcome}",
        artifacts={},
        prerequisites=("s03_divide",),
    )
    assert spec.prerequisites == ("s03_divide",)


def test_stage_spec_prerequisites_are_declarative_metadata() -> None:
    """Validation-free construction: core never checks that a prerequisite
    names a registered stage or that the edges are acyclic — referential
    integrity is the project layer's ``build_layout()`` job, and the layout
    never resolves prerequisites itself."""
    spec = StageSpec(
        name="s07_blend",
        dir_template="s07_blend/{outcome}",
        artifacts={},
        prerequisites=("s99_nope", "s01_outcome"),
    )
    assert spec.prerequisites == ("s99_nope", "s01_outcome")


def test_stage_spec_is_frozen() -> None:
    spec = S01_OUTCOME
    with pytest.raises(FrozenInstanceError):
        spec.name = "s01_renamed"  # type: ignore[misc]


def test_pipeline_layout_is_frozen() -> None:
    layout = PipelineLayout(stages=STAGES, outcome_names=OUTCOMES)
    with pytest.raises(FrozenInstanceError):
        layout.stages = STAGES  # type: ignore[misc]


# --- stage_dir: interpolation ----------------------------------------------


def test_stage_dir_static_stage_ignores_outcome(layout: PipelineLayout) -> None:
    assert layout.stage_dir("s01_outcome") == Path("s01_outcome")
    assert layout.stage_dir("s01_outcome", outcome="outcome_1") == Path("s01_outcome")


def test_stage_dir_interpolates_outcome(layout: PipelineLayout) -> None:
    assert layout.stage_dir("s03_divide", "outcome_1") == Path("s03_divide/outcome_1")
    assert layout.stage_dir("s05_predict", "outcome_3") == Path("s05_predict/outcome_3")


def test_stage_dir_outcome_required_for_outcome_template(
    layout: PipelineLayout,
) -> None:
    with pytest.raises(ValueError, match="requires an outcome"):
        layout.stage_dir("s03_divide")


def test_stage_dir_unknown_stage_raises(layout: PipelineLayout) -> None:
    with pytest.raises(KeyError):
        layout.stage_dir("s07_blend", "outcome_1")


# --- stage_dir / artifact: unknown-outcome rejection ------------------------


def test_stage_dir_unknown_outcome_raises(layout: PipelineLayout) -> None:
    with pytest.raises(ValueError, match="unknown outcome"):
        layout.stage_dir("s03_divide", "outcome_2")
    # validated for static stages too: the name never reaches a path
    with pytest.raises(ValueError, match="unknown outcome"):
        layout.stage_dir("s01_outcome", "outcome_2")


def test_layout_without_declared_outcomes_rejects_every_outcome(
    outcome_layout: PipelineLayout,
) -> None:
    with pytest.raises(ValueError, match="unknown outcome"):
        outcome_layout.stage_dir("s03_divide", "outcome_1")
    # outcome-free stages still resolve with no outcome
    assert outcome_layout.stage_dir("s01_outcome") == Path("s01_outcome")


# --- artifact: declared logical names --------------------------------------


def test_artifact_is_stage_dir_joined_with_filename(
    layout: PipelineLayout,
) -> None:
    expected = Path("s03_divide/outcome_1/train_df.parquet")
    assert layout.artifact("s03_divide", "train_df", "outcome_1") == expected
    assert layout.artifact("s03_divide", "train_df", "outcome_1").parent == Path(
        "s03_divide/outcome_1"
    )


def test_artifact_unknown_logical_raises(layout: PipelineLayout) -> None:
    with pytest.raises(KeyError):
        layout.artifact("s03_divide", "model", "outcome_1")


def test_artifact_unknown_stage_raises(layout: PipelineLayout) -> None:
    with pytest.raises(KeyError):
        layout.artifact("s07_blend", "predictions_df", "outcome_1")


def test_artifact_outcome_required_for_outcome_template(
    layout: PipelineLayout,
) -> None:
    with pytest.raises(ValueError, match="requires an outcome"):
        layout.artifact("s05_predict", "predictions_df")


def test_artifact_unknown_outcome_raises(layout: PipelineLayout) -> None:
    with pytest.raises(ValueError, match="unknown outcome"):
        layout.artifact("s05_predict", "predictions_df", "unknown")


# --- artifact: filenames carrying the outcome placeholder -------------------


@pytest.mark.parametrize("outcome", sorted(OUTCOMES))
def test_artifact_interpolates_outcome_in_filename(
    layout: PipelineLayout,
    outcome: str,
) -> None:
    """Property: a declared ``{outcome}`` filename resolves per declared
    outcome and never leaves the placeholder unformatted."""
    path = layout.artifact("s06_evaluate", "display_table", outcome)
    assert path == Path(f"s06_evaluate/{outcome}/{outcome}_display.csv")
    assert "{outcome}" not in path.as_posix()


def test_artifact_filename_outcome_required_for_static_template() -> None:
    """The artifact-only guard does not depend on the directory template:
    a static stage whose filename carries ``{outcome}`` still requires one."""
    static_spec = StageSpec(
        name="s07_static_blend",
        dir_template="s07_static_blend",
        artifacts={"display_table": "{outcome}_display.csv"},
    )
    static_layout = PipelineLayout(
        stages={static_spec.name: static_spec}, outcome_names=OUTCOMES
    )
    with pytest.raises(ValueError, match="requires an outcome"):
        static_layout.artifact("s07_static_blend", "display_table")
    assert static_layout.artifact(
        "s07_static_blend", "display_table", "outcome_1"
    ) == Path("s07_static_blend/outcome_1_display.csv")


def test_artifact_without_placeholder_resolves_without_outcome(
    layout: PipelineLayout,
) -> None:
    """A placeholder-free artifact of a static stage needs no outcome."""
    assert layout.artifact("s01_outcome", "outcome_df") == Path(
        "s01_outcome/outcome_df.parquet"
    )


# --- side-effect freedom ----------------------------------------------------


def test_layout_resolution_is_side_effect_free(
    layout: PipelineLayout,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Property: no construction or resolution writes to the filesystem.

    Whether the cwd is an empty temp dir or an untouched one, and whether
    the layout was built before or after, resolving paths must leave no
    trace: no directories created, and results that never depend on cwd.
    """
    monkeypatch.chdir(tmp_path)
    assert list(tmp_path.rglob("*")) == []
    layout.stage_dir("s03_divide", "outcome_1")
    layout.artifact("s05_predict", "predictions_df", "outcome_3")
    second = PipelineLayout(stages=STAGES, outcome_names=OUTCOMES)
    second.stage_dir("s05_predict", "outcome_3")
    assert list(tmp_path.rglob("*")) == []


# --- property space: every stage x every outcome ----------------------------


@pytest.mark.parametrize("spec", ALL_STAGES, ids=lambda spec: spec.name)
@pytest.mark.parametrize("outcome", sorted(OUTCOMES))
def test_stage_dir_resolves_to_clean_relative_path(
    spec: StageSpec,
    outcome: str,
    layout: PipelineLayout,
) -> None:
    """Property: every resolution is a clean, relative, placeholder-free
    path, and for static stages the outcome does not change the result."""
    path = layout.stage_dir(spec.name, outcome)
    assert not path.is_absolute()
    assert not any(part in ("", ".", "..") for part in path.parts)
    if "{outcome}" in spec.dir_template:
        assert path.as_posix().endswith("/" + outcome)
    else:
        assert path == Path(spec.dir_template)


@pytest.mark.parametrize("spec", ALL_STAGES, ids=lambda spec: spec.name)
@pytest.mark.parametrize("outcome", sorted(OUTCOMES))
def test_artifact_is_stage_dir_plus_declared_filename(
    spec: StageSpec,
    outcome: str,
    layout: PipelineLayout,
) -> None:
    """Property: each declared artifact resolves consistently with its
    stage dir, for every artifact and every outcome."""
    for logical, filename in spec.artifacts.items():
        resolved_filename = filename.format(outcome=outcome)
        assert (
            layout.artifact(spec.name, logical, outcome)
            == layout.stage_dir(spec.name, outcome) / resolved_filename
        )


def test_outcome_dirs_are_pairwise_distinct(layout: PipelineLayout) -> None:
    """Property: per-outcome stages resolve to distinct directories per
    outcome, while static stages resolve identically for every outcome."""
    for spec in ALL_STAGES:
        resolved = [
            layout.stage_dir(spec.name, outcome) for outcome in sorted(OUTCOMES)
        ]
        if "{outcome}" in spec.dir_template:
            assert len(resolved) == len(set(resolved))
        else:
            assert len(set(resolved)) == 1


@pytest.mark.parametrize(
    "outcome", ["", "..", "outcome_1 ", "OUTCOME_1", "outcome_1/outcome_3"]
)
def test_unknown_outcome_never_reaches_a_path(
    layout: PipelineLayout,
    outcome: str,
) -> None:
    """Property: any name outside the declared set - including path-shaped
    or normalized variants - is rejected before interpolation."""
    with pytest.raises(ValueError, match="unknown outcome"):
        layout.stage_dir("s03_divide", outcome)
