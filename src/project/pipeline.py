"""Declarative stage table and orchestration registry for the pipeline.

This module is the project's single declaration of the on-disk stage layout and
of the run-time bindings that execute the registry-driven stages. It binds the
project-agnostic core path types (:class:`~src.core.paths.StageSpec` and
:class:`~src.core.paths.PipelineLayout`) to this project's output tree, while
``src.core`` keeps no import of this module.

The public surface is:

``StageDescriptor`` / ``STAGES``
    One descriptor per stage, carrying the stage's relative directory
    template, its logical artifact filenames, its declarative prerequisite
    edges, and the **dotted module path** of its entry point. The module path
    is a string, so importing this module never imports a stage module; the
    callable is imported only when the stage is actually executed.
``EXPLICIT_STAGES`` / ``REGISTRY_STAGES``
    The stage names split by how they run: the first three are explicitly
    called by ``src/main.py`` because they ingest external data or run once,
    while the last four are executed by :func:`run_pipeline`. Together they
    reproduce :data:`STAGES` in declaration order.
``StageBinding`` / ``STAGE_BINDINGS``
    The run-time registry: a binding pairs one stage callable with the
    argument builder that feeds it. The s04 training stage has one binding per
    entry in :data:`~src.project.models.MODEL_STRATEGIES`; every other
    registry stage has exactly one binding. Bindings are resolved lazily.
``build_layout()`` / ``build_context()``
    Validate the stage table and assemble the frozen core ``PipelineLayout``
    and ``PipelineContext`` values every stage receives.
``run_pipeline()``
    Execute the registry-driven stages in outcome-registry order, running the
    model stages only for outcomes whose registry entry is modeled.
``ProjectPaths`` / ``build_paths()``
    Bind the declarative layout to an explicit project root and resolve stage
    directories and declared artifacts under ``<project_root>/output``.

The stage table is read-only configuration: :data:`STAGES`, the name tuples,
and :data:`STAGE_BINDINGS` are immutable. A consumer that tries to mutate them
gets a ``TypeError`` instead of silently changing the artifact contract every
stage and path consumer reads.

Every resolution in this module is pure: constructing a descriptor, a layout,
a context, or a ``ProjectPaths`` view never touches the filesystem and never
depends on the current working directory. Only a stage's own ``context.ensure``
call creates directories.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import MappingProxyType

from src.core.paths import PipelineContext, PipelineLayout, StageSpec
from src.project.config import CONFIG, ProjectConfig
from src.project.models import MODEL_STRATEGIES
from src.project.outcomes import (
    OUTCOME_COMPARISONS,
    modeled_outcomes,
)
from src.project.outcomes import (
    outcome_names as registry_outcome_names,
)

_EXPLICIT_STAGE_NAMES: tuple[str, ...] = (
    "s01_outcome",
    "s01b_outcome_analysis",
    "s02_feature",
)
"""Stages called explicitly by the entry point instead of by the executor."""

_REGISTRY_STAGE_NAMES: tuple[str, ...] = (
    "s03_divide",
    "s04_train",
    "s05_predict",
    "s06_evaluate",
)
"""Stages the registry-driven executor runs."""

EXPLICIT_STAGES: tuple[str, ...] = _EXPLICIT_STAGE_NAMES
"""The three stages left outside the orchestration registry, in run order."""

REGISTRY_STAGES: tuple[str, ...] = _REGISTRY_STAGE_NAMES
"""The four stages the orchestration registry executes, in run order."""


def _s01b_artifacts() -> dict[str, str]:
    """Return the s01b artifacts derived from the comparison registry.

    The analysis stage writes one differences table, its CSV projection, a
    statistics file, and four plots per registered outcome comparison. The
    filenames are declared here once, so the stage resolves them through
    ``layout.artifact`` instead of restating a naming rule. The plot and table
    stem is ``<left>_vs_<right>``, read from the comparison vocabulary.
    """
    artifacts = {
        "difference_df": "outcome_differences.parquet",
        "difference_df_csv": "outcome_differences.csv",
        "difference_statistics": "outcome_differences_statistics.txt",
    }
    for comparison in OUTCOME_COMPARISONS:
        stem = f"{comparison.left}_vs_{comparison.right}"
        artifacts[f"{stem}_histogram"] = f"{stem}_histogram.png"
        artifacts[f"{stem}_kde"] = f"{stem}_kde.png"
        artifacts[f"{stem}_difference_histogram"] = f"{stem}_difference_histogram.png"
        artifacts[f"{stem}_difference_kde"] = f"{stem}_difference_kde.png"
    return artifacts


@dataclass(frozen=True, slots=True)
class StageDescriptor:
    """One stage's declarative layout and lazily resolved entry point.

    A descriptor is static configuration: it records where a stage writes, the
    logical names of the artifacts it declares, the other stages it depends
    on, and the dotted module path plus callable name of its entry point. The
    module path is not imported when the descriptor is created, so importing
    :mod:`src.project.pipeline` never imports a stage module.

    Attributes:
        name: Stable stage identifier and the key under which the descriptor
            is registered in :data:`STAGES`.
        dir_template: Relative output-directory template, e.g.
            ``'s03_divide/{outcome}'`` or ``'s01_outcome'``.
        artifacts: Logical artifact name -> declared filename. A filename may
            carry the ``{outcome}`` placeholder.
        module: Dotted module path of the stage's entry point, e.g.
            ``'src.stages.s03_divide_data'``. Imported only when the stage is
            executed.
        prerequisites: Names of the stages whose outputs this stage reads.
            Declarative metadata; :func:`build_layout` checks referential
            integrity and acyclicity.
        callable_name: The entry-point callable's name in ``module``.
        modeled_outcome_only: When True the stage runs only for modeled
            outcomes. The analysis-only outcomes still receive the stages that
            declare this flag False.

    Raises:
        ValueError: ``name``, ``module``, ``callable_name``, or
            ``dir_template`` is empty. This catches a mistyped table entry at
            import time instead of during a run.
    """

    name: str
    dir_template: str
    artifacts: Mapping[str, str]
    module: str
    prerequisites: tuple[str, ...] = ()
    callable_name: str = "main"
    modeled_outcome_only: bool = False

    def __post_init__(self) -> None:
        """Validate the required identity fields on construction."""
        for attribute in ("name", "dir_template", "module", "callable_name"):
            if not getattr(self, attribute):
                raise ValueError(
                    f"stage descriptor {self.name!r} needs a non-empty {attribute}"
                )

    def to_stage_spec(self) -> StageSpec:
        """Return the core path spec this descriptor declares."""
        return StageSpec(
            name=self.name,
            dir_template=self.dir_template,
            artifacts=self.artifacts,
            prerequisites=self.prerequisites,
        )


# The declarative stage table: every stage in run order. s04_train is one
# outcome-scoped stage whose per-model run directories are runtime
# subdirectories named by MODEL_STRATEGIES, not declared stages. s04_train and
# s05_predict declare no stage-level artifacts (their contents are discovered
# through per-run manifests); s06_evaluate adds the per-outcome cross-model
# display table at the stage-directory root.
STAGES: Mapping[str, StageDescriptor] = MappingProxyType(
    {
        "s01_outcome": StageDescriptor(
            name="s01_outcome",
            dir_template="s01_outcome",
            artifacts={
                "outcome_df": "outcome_df.parquet",
                "outcome_statistics": "outcome_statistics.txt",
            },
            module="src.stages.s01_outcome",
        ),
        "s01b_outcome_analysis": StageDescriptor(
            name="s01b_outcome_analysis",
            dir_template="s01b_outcome_analysis",
            artifacts=_s01b_artifacts(),
            module="src.stages.s01b_outcome_analysis",
            prerequisites=("s01_outcome",),
        ),
        "s02_feature": StageDescriptor(
            name="s02_feature",
            dir_template="s02_feature",
            artifacts={
                "feature_df": "feature_df.parquet",
                "feature_head_df": "feature_df_head.csv",
                "feature_manifest": "feature_manifest.json",
            },
            module="src.stages.s02_feature",
        ),
        "s03_divide": StageDescriptor(
            name="s03_divide",
            dir_template="s03_divide/{outcome}",
            artifacts={
                "feature_head_df": "feature_head_df.parquet",
                "outcome_head_df": "outcome_head_df.parquet",
                "train_df": "train_df.parquet",
                "valid_df": "valid_df.parquet",
                "test_df": "test_df.parquet",
                "statistics": "statistics.txt",
            },
            module="src.stages.s03_divide_data",
            prerequisites=("s01_outcome", "s02_feature"),
        ),
        "s04_train": StageDescriptor(
            name="s04_train",
            dir_template="s04_train/{outcome}",
            artifacts={},
            module="src.stages.s04_train",
            prerequisites=("s03_divide",),
            modeled_outcome_only=True,
        ),
        "s05_predict": StageDescriptor(
            name="s05_predict",
            dir_template="s05_predict/{outcome}",
            artifacts={},
            module="src.stages.s05_predict_model_main",
            prerequisites=("s04_train",),
            modeled_outcome_only=True,
        ),
        "s06_evaluate": StageDescriptor(
            name="s06_evaluate",
            dir_template="s06_evaluate/{outcome}",
            artifacts={"display_table": "{outcome}_display.csv"},
            module="src.stages.s06_evaluate_model_main",
            prerequisites=("s05_predict",),
            modeled_outcome_only=True,
        ),
    }
)
"""Every stage descriptor, keyed by stage name in canonical run order."""


def defined_stages() -> tuple[str, ...]:
    """Return every declared stage name in table (run) order."""
    return tuple(STAGES)


def explicit_stages() -> tuple[str, ...]:
    """Return the explicitly called stage names, in run order."""
    return EXPLICIT_STAGES


def _check_prerequisite_graph(stages: Mapping[str, StageSpec]) -> None:
    """Raise ``ValueError`` if the prerequisite edges contain a cycle.

    Walks the graph depth-first from every stage; re-entering a stage on the
    current path closes a cycle, including an immediate self-reference.
    Referential integrity is checked before this walk, so every edge target is
    a registered stage.
    """
    path: list[str] = []

    def visit(name: str) -> None:
        if name in path:
            cycle = path[path.index(name) :] + [name]
            raise ValueError("prerequisite cycle: " + " -> ".join(cycle))
        path.append(name)
        for prerequisite in stages[name].prerequisites:
            visit(prerequisite)
        path.pop()

    for name in stages:
        visit(name)


def _normalize_stage_spec(key: str, entry: object) -> StageSpec:
    """Return the core ``StageSpec`` a table entry declares.

    Accepts both a :class:`StageDescriptor` (the project table's own value) and
    a pre-built core ``StageSpec`` so callers and tests can hand
    :func:`build_layout` a minimal table without duplicating the descriptor's
    module metadata.
    """
    if isinstance(entry, StageDescriptor):
        spec = entry.to_stage_spec()
    elif isinstance(entry, StageSpec):
        spec = entry
    else:
        raise TypeError(
            f"stage {key!r} must be a StageDescriptor or StageSpec; got "
            f"{type(entry).__name__}"
        )
    if spec.name != key:
        raise ValueError(
            f"stage {key!r} registered with mismatched spec name {spec.name!r}; "
            "the registration key must equal StageSpec.name"
        )
    return spec


def build_layout(
    stages: Mapping[str, StageDescriptor | StageSpec] | None = None,
    outcome_names: frozenset[str] | None = None,
) -> PipelineLayout:
    """Return the frozen core layout for the project stage table.

    Validates the table before exposing it: every registration key equals its
    stage name, every prerequisite names a registered stage, and the
    prerequisite edges are acyclic. The resolved layout stays relative; a
    caller binds it to an output root through :func:`build_paths` or by
    joining the returned paths.

    Args:
        stages: The stage table; defaults to :data:`STAGES`.
        outcome_names: The valid ``{outcome}`` values; defaults to the outcome
            registry's names.

    Raises:
        ValueError: A key does not match its stage's name, a prerequisite is
            not registered, or the prerequisite graph contains a cycle.
        TypeError: A table value is neither a ``StageDescriptor`` nor a
            ``StageSpec``.
    """
    table = STAGES if stages is None else stages
    outcomes = (
        frozenset(registry_outcome_names()) if outcome_names is None else outcome_names
    )
    specs = {key: _normalize_stage_spec(key, entry) for key, entry in table.items()}
    for key, spec in specs.items():
        for prerequisite in spec.prerequisites:
            if prerequisite not in specs:
                raise ValueError(
                    f"stage {key!r} declares unknown prerequisite "
                    f"{prerequisite!r}; every prerequisite must name a "
                    "registered stage"
                )
    _check_prerequisite_graph(specs)
    return PipelineLayout(stages=specs, outcome_names=outcomes)


def build_context(
    config: ProjectConfig = CONFIG,
    project_root: Path | None = None,
) -> PipelineContext:
    """Return the frozen context every stage receives.

    The context is the declarative layout plus the injected project
    configuration. ``config`` is typed by the core-side ``ProjectConfig``
    Protocol, so the project dataclass reaches stages without ``src.core``
    importing ``src.project``. The context is root-independent: stage paths
    stay relative until a caller joins them with an explicit output root.

    Args:
        config: The project configuration; defaults to the module ``CONFIG``.
        project_root: Accepted for uniform construction with the other project
            helpers. The context never resolves paths against it, so it is not
            stored; pass it to :func:`build_paths` / :func:`source_data_dir`
            instead.
    """
    del project_root
    return PipelineContext(layout=build_layout(), config=config)


def _standard_args(
    outcome_name: str,
    context: PipelineContext,
    output_root: Path,
) -> tuple[object, ...]:
    """Return the argument tuple shared by the s03/s05/s06 callables."""
    return (outcome_name, context, output_root)


def _strategy_args(
    strategy_id: str,
) -> Callable[[str, PipelineContext, Path], tuple[object, ...]]:
    """Return the s04 argument builder for one model strategy.

    Closes over a fixed strategy id so the single ``src.stages.s04_train``
    entry point receives it as its leading positional argument; the executor's
    loop state supplies the outcome, context, and output root. Deriving the
    bindings from ``MODEL_STRATEGIES`` means run-directory names, manifest
    model names, and the ``run_stage --model`` vocabulary all read from that
    one table.
    """

    def build_args(
        outcome_name: str,
        context: PipelineContext,
        output_root: Path,
    ) -> tuple[object, ...]:
        return (strategy_id, outcome_name, context, output_root)

    return build_args


@dataclass(frozen=True, slots=True)
class StageBinding:
    """One run-time registry binding: a stage callable and its argument builder.

    The callable is referenced by dotted module path and callable name and is
    imported only by :meth:`resolve`, so constructing the registry never
    imports a stage module (which would close an import cycle through the
    stage's own ``src.project.pipeline`` import). ``build_args`` is a pure
    function of the executor's loop state; it returns the callable's
    positional arguments.

    Attributes:
        stage: The declarative stage name (:data:`STAGES` key).
        module: Dotted module path holding the callable.
        callable_name: The callable's name in ``module``.
        build_args: Produces the callable's positional arguments from the
            outcome, context, and output root.
        modeled_outcome_only: When True the binding runs only for outcomes
            whose registry entry is modeled.
        binding_id: Stable per-model identifier for a binding that owns a
            model run (the s04 strategy id); ``None`` for stages without
            per-model identity.
    """

    stage: str
    module: str
    callable_name: str
    build_args: Callable[[str, PipelineContext, Path], tuple[object, ...]]
    modeled_outcome_only: bool = False
    binding_id: str | None = None

    def resolve(self) -> Callable[..., object]:
        """Import and return the bound callable.

        The import happens at call time, never at registry construction, so
        importing :mod:`src.project.pipeline` stays stage-free.

        Raises:
            ImportError: ``module`` cannot be imported.
            AttributeError: ``module`` has no ``callable_name`` attribute.
        """
        module = import_module(self.module)
        return getattr(module, self.callable_name)


def _s04_train_bindings() -> tuple[StageBinding, ...]:
    """Return one s04 binding per entry in ``MODEL_STRATEGIES``, in table order.

    Each binding points at the single ``src.stages.s04_train.main`` entry
    point, carries the strategy id as ``binding_id``, and closes over that id
    in its argument builder. Adding a model is a strategy-table entry, not a
    new module or a new binding.
    """
    descriptor = STAGES["s04_train"]
    return tuple(
        StageBinding(
            stage=descriptor.name,
            module=descriptor.module,
            callable_name=descriptor.callable_name,
            build_args=_strategy_args(strategy_id),
            modeled_outcome_only=descriptor.modeled_outcome_only,
            binding_id=strategy_id,
        )
        for strategy_id in MODEL_STRATEGIES
    )


def _registry_bindings() -> tuple[StageBinding, ...]:
    """Derive the binding registry from ``REGISTRY_STAGES`` and the table."""
    bindings: list[StageBinding] = []
    for stage_name in REGISTRY_STAGES:
        descriptor = STAGES[stage_name]
        if stage_name == "s04_train":
            bindings.extend(_s04_train_bindings())
            continue
        bindings.append(
            StageBinding(
                stage=descriptor.name,
                module=descriptor.module,
                callable_name=descriptor.callable_name,
                build_args=_standard_args,
                modeled_outcome_only=descriptor.modeled_outcome_only,
            )
        )
    return tuple(bindings)


STAGE_BINDINGS: tuple[StageBinding, ...] = _registry_bindings()
"""The run-time registry: every binding the executor may call, in run order."""


def run_pipeline(
    context: PipelineContext,
    output_root: Path,
    bindings: Iterable[StageBinding] | None = None,
) -> None:
    """Execute the registry-driven stages for every pipeline outcome.

    Iterates the outcome registry in declaration order. Every outcome receives
    the bindings whose ``modeled_outcome_only`` flag is False (the divide
    stage); the model stages run only for outcomes whose registry entry is
    modeled. Each callable is resolved lazily at call time.

    s01, s01b, and s02 are never executed here: the entry point keeps them as
    explicit calls.

    Args:
        context: The frozen pipeline context every stage receives.
        output_root: The output root the relative stage paths resolve under.
        bindings: The bindings to execute; defaults to :data:`STAGE_BINDINGS`.
            An explicit iterable lets a caller or test substitute a registry
            without mutating the module table.
    """
    selected = STAGE_BINDINGS if bindings is None else tuple(bindings)
    modeled_names = {outcome.name for outcome in modeled_outcomes()}
    for outcome_name in registry_outcome_names():
        is_modeled = outcome_name in modeled_names
        for binding in selected:
            if binding.modeled_outcome_only and not is_modeled:
                continue
            callable_ = binding.resolve()
            callable_(*binding.build_args(outcome_name, context, output_root))


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """A frozen view of the project stage layout under one output root.

    ``stage_dir`` and ``artifact`` delegate to the core layout and prefix every
    resolution with ``project_root / 'output'``. The optional ``outcome`` bound
    at construction supplies the ``{outcome}`` placeholder for per-outcome
    stages; an explicit per-call outcome wins. Constructing or resolving a view
    never touches the filesystem or the current working directory.

    Raises:
        ValueError: An outcome bound at construction is not one of the
            layout's declared outcome names.
    """

    project_root: Path
    outcome: str | None = None
    layout: PipelineLayout = field(default_factory=build_layout)

    def __post_init__(self) -> None:
        """Reject an unknown outcome bound at construction."""
        if self.outcome is not None and self.outcome not in self.layout.outcome_names:
            raise ValueError(
                f"unknown outcome {self.outcome!r} bound to project paths; "
                f"known outcomes: {sorted(self.layout.outcome_names)}"
            )

    @property
    def project_output(self) -> Path:
        """Return the pipeline output root: ``project_root / 'output'``."""
        return self.project_root / "output"

    def stage_dir(self, name: str, outcome: str | None = None) -> Path:
        """Return the project-absolute output directory of one stage.

        ``outcome`` defaults to the view's bound outcome; a per-call outcome
        overrides it. Raises exactly like the core layout for an unknown
        stage, a missing required outcome, or an unknown outcome.
        """
        resolved = self.outcome if outcome is None else outcome
        return self.project_output / self.layout.stage_dir(name, resolved)

    def artifact(
        self,
        stage: str,
        logical: str,
        outcome: str | None = None,
    ) -> Path:
        """Return the project-absolute path of one declared stage artifact."""
        resolved = self.outcome if outcome is None else outcome
        return self.project_output / self.layout.artifact(stage, logical, resolved)


def build_paths(
    project_root: Path,
    outcome: str | None = None,
) -> ProjectPaths:
    """Bind the declarative stage layout to an explicit project root.

    Args:
        project_root: The project root whose ``output/`` tree is resolved.
        outcome: The outcome to bind for ``{outcome}`` templates.
    """
    return ProjectPaths(project_root=project_root, outcome=outcome)


def source_data_dir(project_root: Path) -> Path:
    """Return the external input directory under an explicit project root.

    ``CONFIG.data_root`` is a relative path and is always resolved against the
    caller's ``project_root``, never against ``Path.cwd()``.
    """
    return Path(project_root) / CONFIG.data_root
