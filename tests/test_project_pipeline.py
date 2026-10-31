"""Unit tests for the declarative stage table and orchestration registry.

Pins the artifact contract in ``src/project/pipeline.py``: the seven stage
descriptors, their directory templates, logical artifacts, prerequisite
edges, and lazily resolved entry points; the explicit/registry split; the
s04 bindings derived from the model-strategy table; the executor's
outcome-registry traversal; and the ``ProjectPaths`` view. A table edit that
orphans a prerequisite, duplicates a filename, or imports a stage eagerly must
fail here rather than during a run.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType

import pytest

from src.core.paths import PipelineContext, PipelineLayout, StageSpec
from src.project.config import CONFIG
from src.project.models import MODEL_STRATEGIES
from src.project.outcomes import (
    OUTCOME_COMPARISONS,
    modeled_outcomes,
    outcome_names,
)
from src.project.pipeline import (
    EXPLICIT_STAGES,
    REGISTRY_STAGES,
    STAGE_BINDINGS,
    STAGES,
    ProjectPaths,
    StageBinding,
    StageDescriptor,
    build_context,
    build_layout,
    build_paths,
    defined_stages,
    explicit_stages,
    run_pipeline,
    source_data_dir,
)

OUTCOMES = outcome_names()
MODELED_OUTCOMES = {outcome.name for outcome in modeled_outcomes()}

EXPECTED_MODULES = {
    "s01_outcome": "src.stages.s01_outcome",
    "s01b_outcome_analysis": "src.stages.s01b_outcome_analysis",
    "s02_feature": "src.stages.s02_feature",
    "s03_divide": "src.stages.s03_divide_data",
    "s04_train": "src.stages.s04_train",
    "s05_predict": "src.stages.s05_predict_model_main",
    "s06_evaluate": "src.stages.s06_evaluate_model_main",
}

EXPECTED_DIR_TEMPLATES = {
    "s01_outcome": "s01_outcome",
    "s01b_outcome_analysis": "s01b_outcome_analysis",
    "s02_feature": "s02_feature",
    "s03_divide": "s03_divide/{outcome}",
    "s04_train": "s04_train/{outcome}",
    "s05_predict": "s05_predict/{outcome}",
    "s06_evaluate": "s06_evaluate/{outcome}",
}

EXPECTED_PREREQUISITES = {
    "s01_outcome": (),
    "s01b_outcome_analysis": ("s01_outcome",),
    "s02_feature": (),
    "s03_divide": ("s01_outcome", "s02_feature"),
    "s04_train": ("s03_divide",),
    "s05_predict": ("s04_train",),
    "s06_evaluate": ("s05_predict",),
}

MODELED_STAGES = ("s04_train", "s05_predict", "s06_evaluate")


# --- stage descriptors and table -------------------------------------------


def test_stage_table_declares_every_stage_in_run_order() -> None:
    assert tuple(STAGES) == tuple(EXPECTED_MODULES)


@pytest.mark.parametrize("name", EXPECTED_MODULES, ids=lambda name: name)
def test_stage_descriptor_declares_layout_and_module(name: str) -> None:
    descriptor = STAGES[name]
    assert isinstance(descriptor, StageDescriptor)
    assert descriptor.name == name
    assert descriptor.dir_template == EXPECTED_DIR_TEMPLATES[name]
    assert descriptor.module == EXPECTED_MODULES[name]
    assert descriptor.callable_name == "main"
    assert descriptor.prerequisites == EXPECTED_PREREQUISITES[name]


@pytest.mark.parametrize("name", EXPECTED_MODULES, ids=lambda name: name)
def test_modeled_flag_is_true_only_for_model_stages(name: str) -> None:
    assert STAGES[name].modeled_outcome_only is (name in MODELED_STAGES)


def test_stage_table_is_read_only() -> None:
    with pytest.raises(TypeError):
        STAGES["s99_nope"] = STAGES["s01_outcome"]  # type: ignore[index]


def test_stage_descriptor_is_frozen() -> None:
    descriptor = STAGES["s01_outcome"]
    with pytest.raises(FrozenInstanceError):
        descriptor.module = "src.stages.s99_nope"  # type: ignore[misc]


def test_stage_descriptor_rejects_empty_identity_fields() -> None:
    with pytest.raises(ValueError, match="non-empty module"):
        StageDescriptor(
            name="s01_outcome",
            dir_template="s01_outcome",
            artifacts={},
            module="",
        )


def test_stage_descriptor_to_stage_spec_preserves_layout() -> None:
    descriptor = STAGES["s03_divide"]
    spec = descriptor.to_stage_spec()
    assert isinstance(spec, StageSpec)
    assert spec.name == descriptor.name
    assert spec.dir_template == descriptor.dir_template
    assert spec.artifacts == descriptor.artifacts
    assert spec.prerequisites == descriptor.prerequisites


def test_s01_s02_and_s03_artifact_names() -> None:
    assert STAGES["s01_outcome"].artifacts == {
        "outcome_df": "outcome_df.parquet",
        "outcome_statistics": "outcome_statistics.txt",
    }
    assert STAGES["s02_feature"].artifacts == {
        "feature_df": "feature_df.parquet",
        "feature_head_df": "feature_df_head.csv",
        "feature_manifest": "feature_manifest.json",
    }
    assert STAGES["s03_divide"].artifacts == {
        "feature_head_df": "feature_head_df.parquet",
        "outcome_head_df": "outcome_head_df.parquet",
        "train_df": "train_df.parquet",
        "valid_df": "valid_df.parquet",
        "test_df": "test_df.parquet",
        "statistics": "statistics.txt",
    }


def test_s01b_artifacts_derive_from_the_comparison_registry() -> None:
    artifacts = STAGES["s01b_outcome_analysis"].artifacts
    assert artifacts["difference_df"] == "outcome_differences.parquet"
    assert artifacts["difference_df_csv"] == "outcome_differences.csv"
    assert artifacts["difference_statistics"] == ("outcome_differences_statistics.txt")
    assert len(artifacts) == 3 + 4 * len(OUTCOME_COMPARISONS)
    for comparison in OUTCOME_COMPARISONS:
        stem = f"{comparison.left}_vs_{comparison.right}"
        assert artifacts[f"{stem}_histogram"] == f"{stem}_histogram.png"
        assert artifacts[f"{stem}_kde"] == f"{stem}_kde.png"
        assert artifacts[f"{stem}_difference_histogram"] == (
            f"{stem}_difference_histogram.png"
        )
        assert artifacts[f"{stem}_difference_kde"] == (f"{stem}_difference_kde.png")


def test_model_run_stages_declare_no_stage_artifacts() -> None:
    assert STAGES["s04_train"].artifacts == {}
    assert STAGES["s05_predict"].artifacts == {}
    assert STAGES["s06_evaluate"].artifacts == {
        "display_table": "{outcome}_display.csv",
    }


# --- explicit / registry split ---------------------------------------------


def test_explicit_and_registry_stages_partition_the_table() -> None:
    assert EXPLICIT_STAGES == (
        "s01_outcome",
        "s01b_outcome_analysis",
        "s02_feature",
    )
    assert REGISTRY_STAGES == (
        "s03_divide",
        "s04_train",
        "s05_predict",
        "s06_evaluate",
    )
    assert set(EXPLICIT_STAGES).isdisjoint(REGISTRY_STAGES)
    assert EXPLICIT_STAGES + REGISTRY_STAGES == tuple(STAGES)


def test_name_helpers_track_the_table() -> None:
    assert defined_stages() == tuple(STAGES)
    assert explicit_stages() == EXPLICIT_STAGES


# --- build_layout ---------------------------------------------------------


def test_build_layout_returns_frozen_layout_over_the_table() -> None:
    layout = build_layout()
    assert isinstance(layout, PipelineLayout)
    assert layout.stages.keys() == STAGES.keys()
    assert layout.outcome_names == frozenset(OUTCOMES)
    with pytest.raises(FrozenInstanceError):
        layout.outcome_names = frozenset()  # type: ignore[misc]


def test_build_layout_resolves_current_layout() -> None:
    layout = build_layout()
    assert layout.stage_dir("s03_divide", "outcome_1") == Path("s03_divide/outcome_1")
    assert layout.artifact("s03_divide", "train_df", "outcome_1") == Path(
        "s03_divide/outcome_1/train_df.parquet"
    )
    assert layout.artifact("s06_evaluate", "display_table", "outcome_3") == Path(
        "s06_evaluate/outcome_3/outcome_3_display.csv"
    )


def test_build_layout_preserves_prerequisite_edges() -> None:
    layout = build_layout()
    assert layout.stages["s03_divide"].prerequisites == (
        "s01_outcome",
        "s02_feature",
    )
    assert layout.stages["s06_evaluate"].prerequisites == ("s05_predict",)


def test_build_layout_rejects_mismatched_registration_key() -> None:
    bad = dict(STAGES)
    bad["s01_renamed"] = STAGES["s01_outcome"]
    with pytest.raises(ValueError, match="mismatched spec name"):
        build_layout(stages=bad)


def test_build_layout_rejects_unknown_prerequisite_name() -> None:
    bad = {
        "s05_predict": StageDescriptor(
            name="s05_predict",
            dir_template="s05_predict/{outcome}",
            artifacts={},
            module="src.stages.s05_predict_model_main",
            prerequisites=("s99_nope",),
        ),
    }
    with pytest.raises(ValueError, match="unknown prerequisite"):
        build_layout(stages=bad)


def test_build_layout_rejects_self_referential_prerequisite() -> None:
    bad = {
        "s03_divide": StageSpec(
            name="s03_divide",
            dir_template="s03_divide/{outcome}",
            artifacts={},
            prerequisites=("s03_divide",),
        ),
    }
    with pytest.raises(ValueError, match="prerequisite cycle"):
        build_layout(stages=bad)


def test_build_layout_rejects_prerequisite_cycle() -> None:
    bad = {
        "s01_outcome": StageSpec(
            name="s01_outcome",
            dir_template="s01_outcome",
            artifacts={},
            prerequisites=("s02_feature",),
        ),
        "s02_feature": StageSpec(
            name="s02_feature",
            dir_template="s02_feature",
            artifacts={},
            prerequisites=("s01_outcome",),
        ),
    }
    with pytest.raises(ValueError, match="prerequisite cycle"):
        build_layout(stages=bad)


def test_build_layout_accepts_custom_stages_and_outcome_names() -> None:
    layout = build_layout(
        stages={"s01_outcome": STAGES["s01_outcome"]},
        outcome_names=frozenset({"outcome_1"}),
    )
    assert layout.stages.keys() == {"s01_outcome"}
    assert layout.outcome_names == frozenset({"outcome_1"})


def test_build_layout_rejects_an_unknown_table_value() -> None:
    with pytest.raises(TypeError, match="StageDescriptor or StageSpec"):
        build_layout(stages={"s01_outcome": object()})  # type: ignore[dict-item]


# --- registry bindings ----------------------------------------------------


def test_registry_binds_exactly_the_registry_stages() -> None:
    assert {binding.stage for binding in STAGE_BINDINGS} == set(REGISTRY_STAGES)
    assert not ({binding.stage for binding in STAGE_BINDINGS} & set(EXPLICIT_STAGES))


def test_non_model_bindings_are_standard_and_unnamed() -> None:
    for binding in STAGE_BINDINGS:
        if binding.stage == "s04_train":
            continue
        assert binding.binding_id is None
        assert (
            binding.modeled_outcome_only is STAGES[binding.stage].modeled_outcome_only
        )
        assert binding.callable_name == "main"
        assert binding.module == STAGES[binding.stage].module


def test_s04_bindings_derive_from_the_model_strategy_table() -> None:
    s04 = [binding for binding in STAGE_BINDINGS if binding.stage == "s04_train"]
    assert [binding.binding_id for binding in s04] == list(MODEL_STRATEGIES)
    for binding in s04:
        assert binding.module == "src.stages.s04_train"
        assert binding.callable_name == "main"
        assert binding.modeled_outcome_only is True


def test_s04_binding_build_args_prefix_the_strategy_id() -> None:
    binding = next(b for b in STAGE_BINDINGS if b.stage == "s04_train")
    context = build_context()
    args = binding.build_args("outcome_1", context, Path("out"))
    assert args == (binding.binding_id, "outcome_1", context, Path("out"))


def test_stage_binding_is_frozen() -> None:
    binding = STAGE_BINDINGS[0]
    with pytest.raises(FrozenInstanceError):
        binding.stage = "s99_nope"  # type: ignore[misc]


def test_stage_binding_resolves_its_callable_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("tests._fake_pipeline_stage")

    def main(*args: object) -> None:
        """Record the arguments the executor supplied."""

    module.main = main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    binding = StageBinding(
        stage="s03_divide",
        module=module.__name__,
        callable_name="main",
        build_args=lambda outcome, context, output_root: (
            outcome,
            context,
            output_root,
        ),
    )
    assert binding.resolve() is main


# --- lazy imports ---------------------------------------------------------


def test_importing_the_pipeline_imports_no_stage_module() -> None:
    """Import the module in a fresh interpreter and inspect ``sys.modules``.

    A stage module imports this module itself, so an eager stage import here
    would close an import cycle; the descriptor's module string must stay
    unresolved until execution.
    """
    code = (
        "import sys\n"
        "import src.project.pipeline\n"
        "assert not any(name.startswith('src.stages.') for name in sys.modules), "
        "[name for name in sys.modules if name.startswith('src.stages.')]\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


# --- run_pipeline ---------------------------------------------------------


def _fake_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[object, ...]], tuple[StageBinding, ...]]:
    """Install a recording fake stage module and return its bindings."""
    module = ModuleType("tests._fake_pipeline_executor")
    calls: list[tuple[object, ...]] = []

    def main(*args: object) -> None:
        calls.append(args)

    module.main = main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def standard_args(
        outcome_name: str,
        context: PipelineContext,
        output_root: Path,
    ) -> tuple[object, ...]:
        return (outcome_name, context, output_root)

    bindings = (
        StageBinding(
            stage="s03_divide",
            module=module.__name__,
            callable_name="main",
            build_args=standard_args,
        ),
        StageBinding(
            stage="s04_train",
            module=module.__name__,
            callable_name="main",
            build_args=lambda outcome, context, output_root: (
                "model-id",
                outcome,
                context,
                output_root,
            ),
            modeled_outcome_only=True,
            binding_id="model-id",
        ),
    )
    return calls, bindings


def test_run_pipeline_iterates_outcomes_and_skips_model_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, bindings = _fake_bindings(monkeypatch)
    context = build_context()
    output_root = Path("out")
    run_pipeline(context, output_root, bindings=bindings)

    expected: list[tuple[object, ...]] = []
    for outcome in OUTCOMES:
        expected.append((outcome, context, output_root))
        if outcome in MODELED_OUTCOMES:
            expected.append(("model-id", outcome, context, output_root))
    assert calls == expected


def test_run_pipeline_resolves_each_binding_before_calling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("tests._fake_pipeline_lazy")

    def main(*args: object) -> None:
        """No-op fake entry point."""

    module.main = main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    resolved: list[str] = []

    original_resolve = StageBinding.resolve

    def resolve(self: StageBinding) -> object:
        resolved.append(self.stage)
        return original_resolve(self)

    monkeypatch.setattr(StageBinding, "resolve", resolve)
    binding = StageBinding(
        stage="s03_divide",
        module=module.__name__,
        callable_name="main",
        build_args=lambda outcome, context, output_root: (),
    )
    run_pipeline(build_context(), Path("out"), bindings=(binding,))
    assert resolved == ["s03_divide"] * len(OUTCOMES)


# --- build_paths / ProjectPaths -------------------------------------------


def test_build_paths_resolves_project_output_root(tmp_path: Path) -> None:
    paths = build_paths(tmp_path, "outcome_1")
    assert isinstance(paths, ProjectPaths)
    assert paths.project_root == tmp_path
    assert paths.project_output == tmp_path / "output"


def test_project_paths_view_is_frozen(tmp_path: Path) -> None:
    paths = build_paths(tmp_path, "outcome_1")
    with pytest.raises(FrozenInstanceError):
        paths.outcome = "outcome_3"  # type: ignore[misc]


def test_build_paths_resolves_bound_outcome(tmp_path: Path) -> None:
    paths = build_paths(tmp_path, "outcome_1")
    assert paths.stage_dir("s03_divide") == (
        tmp_path / "output" / "s03_divide" / "outcome_1"
    )
    assert paths.stage_dir("s05_predict") == (
        tmp_path / "output" / "s05_predict" / "outcome_1"
    )


def test_build_paths_artifact_resolves_declared_filename(tmp_path: Path) -> None:
    paths = build_paths(tmp_path, "outcome_1")
    assert paths.artifact("s03_divide", "train_df") == (
        tmp_path / "output" / "s03_divide" / "outcome_1" / "train_df.parquet"
    )
    assert paths.artifact("s03_divide", "statistics") == (
        tmp_path / "output" / "s03_divide" / "outcome_1" / "statistics.txt"
    )


def test_build_paths_outcome_free_stages_resolve_without_outcome(
    tmp_path: Path,
) -> None:
    paths = build_paths(tmp_path)
    assert paths.stage_dir("s01_outcome") == tmp_path / "output" / "s01_outcome"
    assert paths.artifact("s02_feature", "feature_df") == (
        tmp_path / "output" / "s02_feature" / "feature_df.parquet"
    )


def test_build_paths_requires_outcome_for_outcome_stage(tmp_path: Path) -> None:
    paths = build_paths(tmp_path)
    with pytest.raises(ValueError, match="requires an outcome"):
        paths.stage_dir("s03_divide")
    with pytest.raises(ValueError, match="requires an outcome"):
        paths.artifact("s03_divide", "train_df")


def test_build_paths_rejects_unknown_outcome(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown outcome"):
        build_paths(tmp_path, "outcome_99")


def test_build_paths_per_call_outcome_overrides_bound(tmp_path: Path) -> None:
    paths = build_paths(tmp_path, "outcome_1")
    assert paths.stage_dir("s03_divide", "outcome_3") == (
        tmp_path / "output" / "s03_divide" / "outcome_3"
    )
    assert paths.artifact("s03_divide", "train_df", "outcome_3") == (
        tmp_path / "output" / "s03_divide" / "outcome_3" / "train_df.parquet"
    )


def test_build_paths_unknown_stage_and_artifact_raise(tmp_path: Path) -> None:
    paths = build_paths(tmp_path, "outcome_1")
    with pytest.raises(KeyError):
        paths.stage_dir("s07_blend")
    with pytest.raises(KeyError):
        paths.artifact("s03_divide", "model")


def test_build_paths_is_side_effect_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    project_root = tmp_path / "project"
    paths = build_paths(project_root, "outcome_1")
    paths.stage_dir("s03_divide")
    paths.artifact("s03_divide", "train_df")
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize("outcome", OUTCOMES)
@pytest.mark.parametrize("name", EXPECTED_DIR_TEMPLATES, ids=lambda name: name)
def test_every_stage_resolves_for_every_outcome(
    name: str,
    outcome: str,
    tmp_path: Path,
) -> None:
    paths = build_paths(tmp_path, outcome)
    expected = (
        tmp_path / "output" / EXPECTED_DIR_TEMPLATES[name].format(outcome=outcome)
    )
    assert paths.stage_dir(name) == expected


@pytest.mark.parametrize("outcome", OUTCOMES)
@pytest.mark.parametrize("stage", ["s01_outcome", "s02_feature", "s03_divide"])
def test_every_artifact_resolves_below_its_stage_dir(
    stage: str,
    outcome: str,
    tmp_path: Path,
) -> None:
    descriptor = STAGES[stage]
    paths = build_paths(tmp_path, outcome)
    stage_dir = tmp_path / "output" / descriptor.dir_template.format(outcome=outcome)
    for logical, filename in descriptor.artifacts.items():
        assert paths.artifact(stage, logical) == stage_dir / filename


# --- context and data root ------------------------------------------------


def test_build_context_binds_layout_and_config() -> None:
    context = build_context()
    assert isinstance(context, PipelineContext)
    assert context.layout == build_layout()
    assert context.config is CONFIG


def test_build_context_accepts_an_explicit_config_and_root() -> None:
    context = build_context(config=CONFIG, project_root=Path("/tmp/project"))
    assert context.config is CONFIG
    assert context.layout == build_layout()


def test_source_data_dir_resolves_against_the_project_root(tmp_path: Path) -> None:
    assert source_data_dir(tmp_path) == tmp_path / CONFIG.data_root
