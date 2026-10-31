"""Unit tests for the pipeline context.

Pins the invariant that the context handed to every stage is frozen and
self-contained: it carries the project ``layout`` and ``config`` as a
core-side ``ProjectConfig`` Protocol (so ``src.core`` imports no
``src.project``), and its one explicit side effect, ``ensure``, is the
only place a stage creates a directory. A failure here means a stage could
mutate its configuration, import the project layer from core, or create
directories implicitly on construction — each of which would break cwd
independence and test isolation.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol, get_type_hints

import pytest

from src.core.paths import (
    PipelineContext,
    PipelineLayout,
    ProjectConfig,
    StageSpec,
)

SRC_DIR = Path(__file__).parent.parent / "src"
CORE_PATHS_PATH = SRC_DIR / "core" / "paths.py"


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    """A local stand-in for the project config, mirroring its full surface."""

    data_root: str = "data/demo"
    feature_columns_to_drop: tuple[str, ...] = ()
    train_end_date: date = date(2024, 1, 1)
    test_start_date: date = date(2024, 7, 1)
    binary_thresholds: tuple[float, ...] = (-0.5, 0.0, 0.5)
    evaluated_partitions: tuple[str, ...] = ("train", "valid")
    feature_importance_plot_top_n: int = 10
    demo_seed: int = 12345
    demo_case_count: int = 420
    xgb_single_fit_params: Mapping[str, Any] = field(default_factory=dict)
    xgb_num_boost_round: int = 40
    xgb_early_stopping_rounds: int = 5
    xgb_regressor_seed: int = 7
    xgb_search_seed: int = 11
    xgb_param_distributions: Mapping[str, Any] = field(default_factory=dict)
    xgb_search_n_iter: int = 4
    xgb_search_cv: int = 3


CONFIG = FrozenConfig()

OUTCOMES = frozenset({"outcome_1", "outcome_3"})

STAGES = {
    "s01_outcome": StageSpec(
        name="s01_outcome",
        dir_template="s01_outcome",
        artifacts={"outcome_df": "outcome_df.parquet"},
    ),
    "s03_divide": StageSpec(
        name="s03_divide",
        dir_template="s03_divide/{outcome}",
        artifacts={"train_df": "train_df.parquet"},
    ),
    "s05_predict": StageSpec(
        name="s05_predict",
        dir_template="s05_predict/{outcome}",
        artifacts={"predictions_df": "predictions_df.parquet"},
    ),
    "s06_evaluate": StageSpec(
        name="s06_evaluate",
        dir_template="s06_evaluate/{outcome}",
        artifacts={"display_table": "{outcome}_display.csv"},
    ),
}


def build_layout() -> PipelineLayout:
    """Build the local stage table the context tests resolve paths from."""
    return PipelineLayout(stages=STAGES, outcome_names=OUTCOMES)


def build_context(config: ProjectConfig = CONFIG) -> PipelineContext:
    """Build a context from an explicit config (no module-level patching)."""
    return PipelineContext(layout=build_layout(), config=config)


# --- context shape, identity, frozenness -----------------------------------


def test_context_binds_layout_and_config() -> None:
    ctx = build_context()
    assert isinstance(ctx, PipelineContext)
    assert ctx.layout == build_layout()
    assert ctx.config is CONFIG


def test_pipeline_context_is_frozen() -> None:
    ctx = build_context()
    with pytest.raises(FrozenInstanceError):
        ctx.layout = build_layout()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        ctx.config = CONFIG  # type: ignore[misc]


def test_context_resolves_the_stage_table() -> None:
    """The context resolves the stage table like ``build_layout()``:
    same dirs, same artifacts, same validation."""
    ctx = build_context()
    assert ctx.layout == build_layout()
    assert ctx.layout.stage_dir("s03_divide", "outcome_1") == Path(
        "s03_divide/outcome_1"
    )
    assert ctx.layout.stage_dir("s05_predict", "outcome_3") == Path(
        "s05_predict/outcome_3"
    )
    assert ctx.layout.artifact("s03_divide", "train_df", "outcome_1") == Path(
        "s03_divide/outcome_1/train_df.parquet"
    )
    with pytest.raises(ValueError, match="requires an outcome"):
        ctx.layout.stage_dir("s03_divide")
    with pytest.raises(ValueError, match="unknown outcome"):
        ctx.layout.stage_dir("s03_divide", "outcome_2")


def test_context_carries_only_layout_and_config() -> None:
    """The context carries ``layout`` and ``config``: outcome names reach a
    stage through the layout's ``outcome_names`` set, so a separate
    ``outcomes`` attribute would duplicate that vocabulary."""
    ctx = build_context()
    assert not hasattr(ctx, "outcomes")


def test_context_accepts_an_injected_config() -> None:
    """``build_context``-style injection: the config is a parameter, not a
    module global, so tests can supply their own config."""
    other = FrozenConfig(data_root="data/other", demo_seed=99)
    ctx = build_context(other)
    assert ctx.config is other


# --- config travels as a core-side Protocol, not a src.project import -------


def test_config_field_is_typed_by_a_core_side_protocol() -> None:
    """PipelineContext.config announces the core-side ProjectConfig Protocol,
    and a frozen config conforms to it structurally."""
    hints = get_type_hints(PipelineContext)
    assert hints["config"] is ProjectConfig
    assert issubclass(ProjectConfig, Protocol)
    assert "demo_seed" in get_type_hints(ProjectConfig)
    assert isinstance(CONFIG, ProjectConfig)


def test_protocol_has_real_surface_and_rejects_an_incomplete_config() -> None:
    """A config missing any declared field must not satisfy the Protocol."""

    class IncompleteConfig:
        data_root = "data/demo"
        demo_seed = 1

    assert not isinstance(IncompleteConfig(), ProjectConfig)
    assert isinstance(CONFIG, ProjectConfig)


def test_core_paths_imports_no_src_project() -> None:
    """The Protocol carries the project config without importing it:
    ``core/paths.py`` must import no ``src.project`` module, or core would
    depend on the project layer. ``test_module_imports`` enforces the same
    rule package-wide; this test pins the mechanism locally."""
    source = CORE_PATHS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                alias.name != "src.project"
                and not alias.name.startswith("src.project.")
                for alias in node.names
            ), f"core/paths.py must not import src.project; found {node.names}"
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module != "src.project" and not node.module.startswith(
                "src.project."
            ), f"core/paths.py must not import from {node.module}"


# --- side-effect-free construction, cwd-independent -------------------------


def test_context_construction_and_resolution_are_side_effect_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Property: constructing a context (even twice, from the same cwd) and
    resolving its layout writes nothing to disk and never depends on the
    cwd — only the explicit layout/config values matter."""
    monkeypatch.chdir(tmp_path)
    assert list(tmp_path.rglob("*")) == []
    ctx = build_context()
    ctx.layout.stage_dir("s03_divide", "outcome_1")
    ctx.layout.artifact("s03_divide", "train_df", "outcome_1")
    ctx.layout.stage_dir("s05_predict", "outcome_1")
    ctx.layout.stage_dir("s06_evaluate", "outcome_3")
    second = build_context()
    assert second.layout == ctx.layout
    assert second.config is CONFIG
    assert list(tmp_path.rglob("*")) == []


def test_context_does_not_depend_on_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Path.cwd() default: the same context resolves identically from any
    working directory."""
    monkeypatch.chdir(tmp_path)
    reference = build_context().layout
    reference.stage_dir("s03_divide", "outcome_1")
    assert reference == build_layout()


def test_core_paths_mkdir_lives_only_inside_ensure() -> None:
    """No import-time mkdir: the module's only ``mkdir`` call is the one
    inside ``PipelineContext.ensure``, so constructing or resolving a
    context cannot create a directory. A second ``mkdir`` call is the
    failure this catches."""
    tree = ast.parse(CORE_PATHS_PATH.read_text(encoding="utf-8"))
    mkdir_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mkdir"
    ]
    assert len(mkdir_calls) == 1
    # the single call lives inside the ensure method body
    ensure_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "ensure"
    )
    assert mkdir_calls[0] in ast.walk(ensure_fn)


def test_core_paths_calls_no_cwd() -> None:
    """No Path.cwd() default anywhere in core/paths.py (code, not prose)."""
    tree = ast.parse(CORE_PATHS_PATH.read_text(encoding="utf-8"))
    cwd_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cwd"
    ]
    assert cwd_calls == []


# --- explicit ensure: the context's one side effect -------------------------


def test_ensure_creates_directory_recursively(tmp_path: Path) -> None:
    ctx = build_context()
    target = tmp_path / "output" / "s06_evaluate" / "outcome_3"
    ctx.ensure(target)
    assert target.is_dir()


def test_ensure_creates_multiple_nested_output_dirs(tmp_path: Path) -> None:
    ctx = build_context()
    targets = [
        tmp_path / "output" / "s03_divide" / "outcome_1",
        tmp_path / "output" / "s04_train" / "outcome_1",
        tmp_path / "output" / "s05_predict" / "outcome_1",
    ]
    for target in targets:
        ctx.ensure(target)
    assert all(target.is_dir() for target in targets)


def test_ensure_is_idempotent(tmp_path: Path) -> None:
    ctx = build_context()
    target = tmp_path / "output" / "s05_predict" / "outcome_1"
    ctx.ensure(target)
    ctx.ensure(target)
    assert target.is_dir()


def test_ensure_creates_a_directory_at_the_given_path(tmp_path: Path) -> None:
    """ensure(path) makes *path itself* a directory, parent dirs included."""
    ctx = build_context()
    artifact = tmp_path / "output" / "s03_divide" / "outcome_1" / "train_df.parquet"
    ctx.ensure(artifact)
    assert artifact.is_dir()
    assert artifact.parent.is_dir()


def test_ensure_works_after_constructions_with_no_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contexts are pure until ensure is called; the first touch of disk is
    the explicit ensure invocation itself."""
    monkeypatch.chdir(tmp_path)
    ctx = build_context()
    assert list(tmp_path.rglob("*")) == []
    ctx.ensure(tmp_path / "one")
    assert list(tmp_path.rglob("*")) == [tmp_path / "one"]
