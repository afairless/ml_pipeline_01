"""Cross-component contracts for the declarative stage table.

The stage table is read once by several independent consumers: the core
``PipelineLayout`` resolves paths from it, ``ProjectPaths`` binds those paths
to an output root, ``PipelineContext`` carries the layout to every stage, and
the orchestration registry derives one binding per registry stage (and one per
model strategy for s04). These tests own the invariants that no single
component can see:

- the descriptor table and the core layout are two views of the same
  declaration, including the prerequisite graph;
- every registry binding names a declared stage and matches that stage's
  module, callable, and modeled-outcome flag;
- the binding table covers ``REGISTRY_STAGES`` exactly and the s04 bindings
  reproduce ``MODEL_STRATEGIES``;
- declared artifacts are unique, relative, and free of placeholders other
  than ``{outcome}``.

A failure here means a downstream consumer would resolve a different artifact
than the table declares.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from src.core.paths import PipelineContext, ProjectConfig, StageSpec
from src.project.config import CONFIG
from src.project.models import MODEL_STRATEGIES
from src.project.outcomes import outcome_names
from src.project.pipeline import (
    EXPLICIT_STAGES,
    REGISTRY_STAGES,
    STAGE_BINDINGS,
    STAGES,
    StageBinding,
    StageDescriptor,
    build_context,
    build_layout,
    build_paths,
)

OUTCOMES = outcome_names()
OUTCOME_PLACEHOLDER = "{outcome}"
_PLACEHOLDER_PATTERN = re.compile(r"\{[^{}]*\}")
_MODELED_STAGES = frozenset(
    name for name, descriptor in STAGES.items() if descriptor.modeled_outcome_only
)


# --- descriptor table <-> core layout -------------------------------------


@pytest.mark.parametrize("name", tuple(STAGES), ids=lambda name: name)
def test_layout_spec_matches_the_descriptor(name: str) -> None:
    descriptor = STAGES[name]
    spec = build_layout().stages[name]
    assert isinstance(spec, StageSpec)
    assert spec == descriptor.to_stage_spec()
    assert spec.name == descriptor.name
    assert spec.dir_template == descriptor.dir_template
    assert dict(spec.artifacts) == dict(descriptor.artifacts)
    assert spec.prerequisites == descriptor.prerequisites


def test_layout_outcome_vocabulary_comes_from_the_registry() -> None:
    layout = build_layout()
    assert layout.outcome_names == frozenset(OUTCOMES)


def test_every_prerequisite_edge_targets_a_declared_stage() -> None:
    layout = build_layout()
    for name, spec in layout.stages.items():
        for prerequisite in spec.prerequisites:
            assert prerequisite in layout.stages, (
                f"{name} depends on undeclared stage {prerequisite}"
            )
            assert prerequisite != name


def test_prerequisite_edges_form_an_acyclic_graph() -> None:
    layout = build_layout()
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        assert name not in visiting, f"cycle re-entered {name}"
        if name in visited:
            return
        visiting.add(name)
        for prerequisite in layout.stages[name].prerequisites:
            visit(prerequisite)
        visiting.remove(name)
        visited.add(name)

    for name in layout.stages:
        visit(name)
    assert visited == set(layout.stages)


# --- descriptor / binding consistency -------------------------------------


def test_registry_bindings_cover_the_registry_stages_exactly() -> None:
    assert {binding.stage for binding in STAGE_BINDINGS} == set(REGISTRY_STAGES)
    assert {binding.stage for binding in STAGE_BINDINGS}.isdisjoint(EXPLICIT_STAGES)


@pytest.mark.parametrize("binding", STAGE_BINDINGS, ids=lambda b: b.stage)
def test_binding_matches_its_descriptor(binding: StageBinding) -> None:
    descriptor = STAGES[binding.stage]
    assert binding.module == descriptor.module
    assert binding.callable_name == descriptor.callable_name
    assert binding.modeled_outcome_only is descriptor.modeled_outcome_only


def test_model_stages_are_bound_as_modeled_outcome_only() -> None:
    for binding in STAGE_BINDINGS:
        assert binding.modeled_outcome_only is (binding.stage in _MODELED_STAGES)
    assert not (_MODELED_STAGES & set(EXPLICIT_STAGES))


def test_s04_bindings_reproduce_the_model_strategy_table() -> None:
    s04 = [binding for binding in STAGE_BINDINGS if binding.stage == "s04_train"]
    assert [binding.binding_id for binding in s04] == list(MODEL_STRATEGIES)
    assert len(s04) == len(MODEL_STRATEGIES)


def test_non_s04_bindings_have_no_model_identity() -> None:
    for binding in STAGE_BINDINGS:
        if binding.stage != "s04_train":
            assert binding.binding_id is None


def test_registry_stage_names_are_known_descriptors() -> None:
    for name in REGISTRY_STAGES:
        assert name in STAGES
        assert isinstance(STAGES[name], StageDescriptor)
    for name in EXPLICIT_STAGES:
        assert name in STAGES
        assert isinstance(STAGES[name], StageDescriptor)


# --- artifact declarations ------------------------------------------------


@pytest.mark.parametrize("name", tuple(STAGES), ids=lambda name: name)
def test_artifact_filenames_are_unique_within_a_stage(name: str) -> None:
    filenames = list(STAGES[name].artifacts.values())
    assert len(filenames) == len(set(filenames))


@pytest.mark.parametrize("name", tuple(STAGES), ids=lambda name: name)
def test_artifact_filenames_are_relative_and_use_only_outcome_placeholder(
    name: str,
) -> None:
    for logical, filename in STAGES[name].artifacts.items():
        path = Path(filename)
        assert not path.is_absolute(), f"{name}.{logical} is absolute"
        assert ".." not in path.parts, f"{name}.{logical} escapes the stage dir"
        placeholders = set(_PLACEHOLDER_PATTERN.findall(filename))
        assert placeholders <= {OUTCOME_PLACEHOLDER}, (
            f"{name}.{logical} uses unsupported placeholders {placeholders}"
        )


@pytest.mark.parametrize("name", tuple(STAGES), ids=lambda name: name)
def test_dir_templates_are_relative_and_use_only_outcome_placeholder(
    name: str,
) -> None:
    template = STAGES[name].dir_template
    path = Path(template)
    assert not path.is_absolute()
    assert ".." not in path.parts
    placeholders = set(_PLACEHOLDER_PATTERN.findall(template))
    assert placeholders <= {OUTCOME_PLACEHOLDER}


def test_bindings_point_at_stage_modules_named_by_their_descriptor() -> None:
    for binding in STAGE_BINDINGS:
        descriptor = STAGES[binding.stage]
        assert binding.module == descriptor.module
        assert binding.module.startswith("src.stages.")


# --- layout / paths / context agreement -----------------------------------


def test_project_paths_resolve_exactly_the_layout_paths(tmp_path: Path) -> None:
    layout = build_layout()
    outcome = OUTCOMES[0]
    paths = build_paths(tmp_path, outcome)
    for name, descriptor in STAGES.items():
        stage_outcome = (
            outcome if OUTCOME_PLACEHOLDER in descriptor.dir_template else None
        )
        expected = tmp_path / "output" / layout.stage_dir(name, stage_outcome)
        assert paths.stage_dir(name, stage_outcome) == expected
        for logical in descriptor.artifacts:
            expected_artifact = (
                tmp_path / "output" / layout.artifact(name, logical, stage_outcome)
            )
            assert paths.artifact(name, logical, stage_outcome) == expected_artifact


def test_context_carries_the_project_layout_and_config() -> None:
    context = build_context()
    assert isinstance(context, PipelineContext)
    assert isinstance(context.config, ProjectConfig)
    assert context.layout == build_layout()
    assert context.config is CONFIG


def test_stage_table_is_the_single_source_for_all_consumers() -> None:
    """The descriptor table, layout, and paths view expose the same stage set.

    No consumer may add or omit a stage on its own; otherwise ``run_pipeline``
    could execute a binding whose artifacts the layout never resolves.
    """
    table_names = tuple(STAGES)
    layout_names = tuple(build_layout().stages)
    binding_names = tuple(dict.fromkeys(binding.stage for binding in STAGE_BINDINGS))
    assert table_names == layout_names
    assert binding_names == REGISTRY_STAGES
    assert set(table_names) == set(EXPLICIT_STAGES) | set(REGISTRY_STAGES)


def test_layout_artifacts_are_exposed_through_a_read_only_mapping() -> None:
    layout = build_layout()
    for spec in layout.stages.values():
        assert isinstance(spec.artifacts, Mapping)
