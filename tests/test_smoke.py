"""Fast smoke tests for the pipeline's stage and model vocabularies.

These tests import every declared stage module, resolve every registry binding,
and confirm the committed demo input is present and readable. They deliberately
do not execute a stage: the full demo run lives in ``test_smoke_pipeline.py``.
The point here is to catch a broken module path, a missing entry-point
callable, a registry binding that no longer resolves, or a model strategy with
no training binding before the slower integration smoke test starts.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

from src.project.config import CONFIG
from src.project.inputs import read_input_table
from src.project.models import MODEL_STRATEGIES
from src.project.outcomes import modeled_outcomes, outcome_names
from src.project.pipeline import STAGE_BINDINGS, STAGES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Repository root, used only to locate the committed demo input table."""

SOURCE_TABLE = "source_table_1"
"""Logical name of the committed demo source table."""


@pytest.mark.parametrize("stage_name", tuple(STAGES), ids=lambda name: name)
def test_stage_module_exposes_its_entry_point(stage_name: str) -> None:
    """Every declared stage module imports and exposes its declared callable."""
    descriptor = STAGES[stage_name]
    module = import_module(descriptor.module)
    entry_point = getattr(module, descriptor.callable_name)
    assert callable(entry_point)


def test_every_registry_binding_resolves_to_a_callable() -> None:
    """The lazy binding registry resolves every stage callable."""
    assert STAGE_BINDINGS
    for binding in STAGE_BINDINGS:
        assert callable(binding.resolve())


def test_every_model_strategy_has_a_training_binding() -> None:
    """The training bindings reproduce the model-strategy table exactly."""
    bound_models = tuple(
        binding.binding_id for binding in STAGE_BINDINGS if binding.stage == "s04_train"
    )
    assert bound_models == tuple(MODEL_STRATEGIES)


def test_registry_covers_every_non_explicit_stage() -> None:
    """Every stage after the explicit prefix has at least one binding."""
    bound_stages = {binding.stage for binding in STAGE_BINDINGS}
    assert bound_stages == {
        "s03_divide",
        "s04_train",
        "s05_predict",
        "s06_evaluate",
    }


def test_model_bindings_target_only_modeled_outcomes() -> None:
    """Only modeled outcomes receive the model stages."""
    modeled_names = {outcome.name for outcome in modeled_outcomes()}
    assert modeled_names < set(outcome_names())
    for binding in STAGE_BINDINGS:
        descriptor = STAGES[binding.stage]
        if descriptor.modeled_outcome_only:
            assert binding.stage in {"s04_train", "s05_predict", "s06_evaluate"}


def test_committed_demo_source_table_is_readable() -> None:
    """The full demo run can find and read its committed external input."""
    source_df = read_input_table(SOURCE_TABLE, PROJECT_ROOT / CONFIG.data_root)
    assert source_df.height == CONFIG.demo_case_count
    assert source_df.height > 0
