"""Importable run-stage planner for the pipeline's declarative stage table.

The planner turns a small selection value into the exact ordered list of stage
calls a caller wants to run, and then executes those calls. It is the only
place that knows how a selected stage is fanned out across the outcome and
model registries:

- an outcome-free stage (one whose directory template carries no
  ``{outcome}`` placeholder) is planned once;
- a per-outcome stage is planned once per allowed outcome;
- the training stage is planned once per modeled outcome and model strategy;
- ``with_prerequisites`` closes the selection over the declarative
  prerequisite graph, then re-orders the result into table (run) order.

Selection validation is centralized in :class:`StageSelection`, so an unknown
stage name, outcome, or model is rejected before any import or filesystem
work. Planning is pure: :func:`plan_calls` reads the module tables and returns
frozen :class:`PlannedCall` values without importing a stage module, creating
a directory, or touching the current working directory.

Importing this module imports no stage module and no estimator library. The
stage table stores each entry point as a dotted module path, and
:func:`run_stages` resolves that path only when it is about to execute the
call. Tests pin both properties in a fresh interpreter.

The public surface is:

``defined_stages()`` / ``explicit_stages()`` / ``model_ids()`` / \
``allowed_outcomes()``
    Read-only views of the stage, model, and outcome vocabularies the planner
    plans against.
``check_registry_consistency()``
    Cross-check the stage table, the explicit/registry split, and the
    run-time bindings. It raises ``ValueError`` instead of letting a planner
    fan out over an inconsistent registry.
``StageSelection`` / ``PlannedCall``
    The frozen input selection and the frozen planned call.
``plan_calls()``
    Expand a selection into ordered calls without side effects.
``run_stages()``
    Resolve and execute planned calls, passing each entry point its declared
    positional arguments.
``render_list()`` / ``render_plan()``
    Deterministic text renderings for the command-line wrapper's ``--list``
    and ``--dry-run`` modes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from src.core.paths import OUTCOME_PLACEHOLDER, PipelineContext
from src.project.models import MODEL_STRATEGIES
from src.project.outcomes import OUTCOMES, modeled_outcomes, outcome_names
from src.project.pipeline import (
    EXPLICIT_STAGES,
    REGISTRY_STAGES,
    STAGE_BINDINGS,
    STAGES,
    defined_stages,
    explicit_stages,
)

__all__ = [
    "PlannedCall",
    "StageSelection",
    "allowed_outcomes",
    "check_registry_consistency",
    "defined_stages",
    "explicit_stages",
    "model_ids",
    "plan_calls",
    "render_list",
    "render_plan",
    "run_stages",
]

_TRAINING_STAGE = "s04_train"
"""The one stage that carries a model identity in its planned call."""


def model_ids() -> tuple[str, ...]:
    """Return every model strategy id in table (run) order."""
    return tuple(MODEL_STRATEGIES)


def allowed_outcomes() -> tuple[str, ...]:
    """Return every registered outcome name in registry (run) order."""
    return outcome_names()


def _modeled_outcome_names() -> tuple[str, ...]:
    """Return the modeled outcome names in registry (run) order."""
    return tuple(outcome.name for outcome in modeled_outcomes())


def _outcome_pool(stage: str) -> tuple[str, ...]:
    """Return the outcomes a per-outcome stage is planned for.

    A stage that sets ``modeled_outcome_only`` runs only for outcomes whose
    registry entry is modeled; every other per-outcome stage runs for the
    full registry.
    """
    if STAGES[stage].modeled_outcome_only:
        return _modeled_outcome_names()
    return allowed_outcomes()


def check_registry_consistency() -> None:
    """Validate the stage, model, and binding vocabularies against each other.

    The planner fans a selection out across three tables that are declared in
    separate modules: the stage table plus its explicit/registry split, the
    model-strategy registry, and the run-time binding registry. A disagreement
    between them would plan a call for a stage that cannot run, or skip a
    model binding that exists. This check makes such a disagreement a
    ``ValueError`` at the planner boundary instead of a confusing failure
    during execution.

    The check is pure metadata validation: it never imports a stage module
    and never touches the filesystem.

    Raises:
        ValueError: The split does not reproduce the stage table, a registry
            binding names a stage outside the registry set, the s04 bindings
            do not reproduce the model strategy table, a modeled-outcome-only
            stage is outside the registry set, or a descriptor lacks a module
            path.
    """
    names = defined_stages()
    if tuple(EXPLICIT_STAGES) + tuple(REGISTRY_STAGES) != names:
        raise ValueError(
            "explicit and registry stage names must partition the stage table "
            f"in run order; got {EXPLICIT_STAGES + REGISTRY_STAGES} for {names}"
        )
    overlap = set(EXPLICIT_STAGES) & set(REGISTRY_STAGES)
    if overlap:
        raise ValueError(
            f"stage names {sorted(overlap)} are both explicit and registry-driven"
        )
    for binding in STAGE_BINDINGS:
        if binding.stage not in REGISTRY_STAGES:
            raise ValueError(
                f"binding targets stage {binding.stage!r}, which is not one of "
                f"the registry stages {REGISTRY_STAGES}"
            )
    s04_models = tuple(
        binding.binding_id
        for binding in STAGE_BINDINGS
        if binding.stage == _TRAINING_STAGE
    )
    if s04_models != model_ids():
        raise ValueError(
            f"the {_TRAINING_STAGE} bindings {s04_models} must reproduce the "
            f"model strategy table {model_ids()}"
        )
    for name in names:
        descriptor = STAGES[name]
        if not descriptor.module:
            raise ValueError(f"stage {name!r} needs a non-empty module path")
        if descriptor.modeled_outcome_only and name not in REGISTRY_STAGES:
            raise ValueError(
                f"stage {name!r} is modeled-outcome only but is not a "
                "registry stage; the planner cannot fan it out"
            )


@dataclass(frozen=True, slots=True)
class StageSelection:
    """One caller's stage, outcome, and model selection.

    The selection is validated on construction so downstream planning can
    trust it: every stage name is a declared stage, an outcome is a registered
    outcome, and a model is a registered strategy id. Stage order is preserved
    as supplied; :func:`plan_calls` restores table order after expanding
    prerequisites.

    Attributes:
        stages: Selected stage names. An empty tuple selects nothing.
        outcome: Optional outcome filter for per-outcome stages. It is
            ignored by outcome-free stages.
        model: Optional model filter. It is only meaningful together with the
            training stage; selecting a model without that stage is rejected
            by :func:`plan_calls`.
        with_prerequisites: When True, the selection is closed over the
            declarative prerequisite graph before planning.

    Raises:
        ValueError: A stage name is not declared, a stage name is repeated, an
            outcome is not registered, or a model is not registered.
    """

    stages: tuple[str, ...] = ()
    outcome: str | None = None
    model: str | None = None
    with_prerequisites: bool = False

    def __post_init__(self) -> None:
        """Validate the selection against the module vocabularies."""
        known_stages = defined_stages()
        for name in self.stages:
            if name not in STAGES:
                raise ValueError(
                    f"unknown stage {name!r}; declared stages are {known_stages}"
                )
        if len(set(self.stages)) != len(self.stages):
            raise ValueError(
                f"stage selection {self.stages} repeats a stage name; a stage "
                "is planned at most once"
            )
        if self.outcome is not None and self.outcome not in OUTCOMES:
            raise ValueError(
                f"unknown outcome {self.outcome!r}; registered outcomes are "
                f"{allowed_outcomes()}"
            )
        if self.model is not None and self.model not in MODEL_STRATEGIES:
            raise ValueError(
                f"unknown model {self.model!r}; model strategy ids are {model_ids()}"
            )


@dataclass(frozen=True, slots=True)
class PlannedCall:
    """One fully resolved stage call the planner would execute.

    Attributes:
        stage: Declared stage name.
        outcome: Outcome the call runs for, or ``None`` for an outcome-free
            stage.
        model: Model strategy id for the training stage, otherwise ``None``.

    Raises:
        ValueError: ``stage`` is not a declared stage name. Catches a
            hand-built call before :func:`run_stages` resolves a module for
            it.
    """

    stage: str
    outcome: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        """Reject an undeclared stage name on construction."""
        if self.stage not in STAGES:
            raise ValueError(
                f"planned call names unknown stage {self.stage!r}; declared "
                f"stages are {defined_stages()}"
            )

    def render(self) -> str:
        """Return the stable one-line form of this call.

        The fields are space-separated: the stage name always, the outcome
        when present, and the model id when present.
        """
        parts = [self.stage]
        if self.outcome is not None:
            parts.append(self.outcome)
        if self.model is not None:
            parts.append(self.model)
        return " ".join(parts)


def _expand_prerequisites(selection: StageSelection) -> tuple[str, ...]:
    """Return the selected stages, with optional prerequisites, in table order."""
    required: set[str] = set()

    def visit(name: str) -> None:
        if name in required:
            return
        required.add(name)
        if not selection.with_prerequisites:
            return
        for prerequisite in STAGES[name].prerequisites:
            visit(prerequisite)

    for name in selection.stages:
        visit(name)
    return tuple(name for name in defined_stages() if name in required)


def plan_calls(selection: StageSelection) -> tuple[PlannedCall, ...]:
    """Expand a selection into the ordered calls a run would execute.

    Planning is pure and deterministic. Stages are emitted in stage-table
    order; a per-outcome stage is emitted in outcome-registry order; the
    training stage additionally nests model strategies in model-table order.

    Args:
        selection: The validated stage/outcome/model selection.

    Returns:
        The planned calls. An empty selection plans no calls.

    Raises:
        ValueError: Registry tables are inconsistent, a selected model is not
            paired with the training stage, or the outcome filter names an
            outcome the selected stage cannot run for (for example a
            non-modeled outcome with a model stage).
    """
    check_registry_consistency()
    names = _expand_prerequisites(selection)
    if selection.model is not None and _TRAINING_STAGE not in names:
        raise ValueError(
            f"model {selection.model!r} is selected but {_TRAINING_STAGE!r} "
            "is not in the planned stages; a model filter only applies to the "
            "training stage"
        )
    calls: list[PlannedCall] = []
    for name in names:
        descriptor = STAGES[name]
        if OUTCOME_PLACEHOLDER not in descriptor.dir_template:
            calls.append(PlannedCall(stage=name))
            continue
        pool = _outcome_pool(name)
        if selection.outcome is not None:
            if selection.outcome not in pool:
                raise ValueError(
                    f"stage {name!r} runs for outcomes {pool}; selected "
                    f"outcome {selection.outcome!r} is not one of them"
                )
            pool = (selection.outcome,)
        models: tuple[str | None, ...]
        if name == _TRAINING_STAGE:
            models = (selection.model,) if selection.model else model_ids()
        else:
            models = (None,)
        for outcome in pool:
            for model in models:
                calls.append(PlannedCall(stage=name, outcome=outcome, model=model))
    return tuple(calls)


def _resolve_stage(name: str) -> Callable[..., object]:
    """Import and return one stage entry point.

    The import is deliberately deferred to call time, so importing the
    planner never imports a stage module. The module path and callable name
    come from the declarative descriptor, the same declaration the orchestrator
    uses.
    """
    descriptor = STAGES[name]
    module = import_module(descriptor.module)
    return getattr(module, descriptor.callable_name)


def _call_args(
    call: PlannedCall,
    context: PipelineContext,
    output_root: Path,
) -> tuple[object, ...]:
    """Return the positional arguments one planned call's entry point expects.

    Outcome-free stages receive ``(context, output_root)``; per-outcome stages
    prepend the outcome name; the training stage prepends the model id before
    the outcome. No stage receives a path it did not get from this call.
    """
    if call.stage == _TRAINING_STAGE:
        if call.model is None or call.outcome is None:
            raise ValueError(
                f"a {_TRAINING_STAGE!r} call needs both a model id and an "
                f"outcome; got model={call.model!r}, outcome={call.outcome!r}"
            )
        return (call.model, call.outcome, context, output_root)
    if call.model is not None:
        raise ValueError(
            f"stage {call.stage!r} takes no model id; only "
            f"{_TRAINING_STAGE!r} is planned per model"
        )
    if call.outcome is None:
        return (context, output_root)
    return (call.outcome, context, output_root)


def run_stages(
    context: PipelineContext,
    output_root: Path,
    calls: Iterable[PlannedCall],
) -> None:
    """Execute planned calls in order, resolving each entry point lazily.

    Args:
        context: The frozen pipeline context every stage receives.
        output_root: The explicit output root the stage paths resolve under.
        calls: Planned calls, normally the result of :func:`plan_calls`.

    Raises:
        ValueError: The registry is inconsistent, or a call's stage/outcome/
            model combination cannot produce valid arguments.
        ImportError: A stage's declared module cannot be imported.
        AttributeError: A stage module lacks its declared entry point.
    """
    check_registry_consistency()
    for call in tuple(calls):
        arguments = _call_args(call, context, output_root)
        entry_point = _resolve_stage(call.stage)
        entry_point(*arguments)


def render_list() -> str:
    """Return the stage, model, and outcome inventory as deterministic text.

    The rendering is the ``--list`` payload. It reads only the declarative
    tables, so listing the pipeline never imports a stage module or touches
    the filesystem.
    """
    explicit = set(explicit_stages())
    lines = ["Stages:"]
    for name in defined_stages():
        descriptor = STAGES[name]
        mode = "explicit" if name in explicit else "registry"
        lines.append(f"  {name:<22} {descriptor.dir_template:<28} {mode}")
    lines.append("Models:")
    lines.extend(f"  {model_id}" for model_id in model_ids())
    lines.append("Outcomes:")
    for name in allowed_outcomes():
        kind = "modeled" if OUTCOMES[name].modeled else "analysis"
        lines.append(f"  {name:<22} {kind}")
    return "\n".join(lines)


def render_plan(calls: Iterable[PlannedCall]) -> str:
    """Return the ordered planned calls as deterministic text.

    One call is rendered per line through :meth:`PlannedCall.render`. An
    empty plan renders a single explanatory line instead of an empty string,
    so ``--dry-run`` output is never ambiguous.
    """
    lines = [call.render() for call in calls]
    return "\n".join(lines) if lines else "(no calls planned)"
