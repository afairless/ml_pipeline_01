"""Core path layout types and pipeline context.

Declarative, side-effect-free path resolution for the pipeline stage table.
A frozen ``StageSpec`` declares one stage's relative output-directory template
plus its logical artifact filenames; a frozen ``PipelineLayout`` resolves those
declarations to relative ``Path`` objects via ``stage_dir``/``artifact``.

The layout is pure: constructing a layout or resolving a path never touches
the filesystem and never depends on the current working directory. Resolved
paths are relative to the caller's project output root, which the project
layer binds when it builds the layout.

The frozen ``PipelineContext`` (``(layout, config)`` plus ``ensure``) is the
value every pipeline stage receives. ``config`` is typed by the core-side
``ProjectConfig`` Protocol, whose surface mirrors
``src.project.config.ProjectConfig`` so ``src.core`` keeps no import of
``src.project``; ``ensure`` is the context's single explicit side effect — a
stage creates its own stage directory only through it, never at import time
and never from a ``Path.cwd()`` default.

Outcome interpolation is validated, never formatted blindly: a ``{outcome}``
template — whether in a stage's directory template or in a declared artifact
filename — requires an explicit ``outcome`` (``None`` raises ``ValueError``),
and an outcome that is not among the layout's declared ``outcome_names``
raises ``ValueError`` before it can be interpolated into a filesystem path.
The project layer declares the known outcome names when it builds the layout,
and a layout that declares none rejects every outcome (only outcome-free
stages resolve).
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

OUTCOME_PLACEHOLDER = "{outcome}"


@dataclass(frozen=True)
class StageSpec:
    """One stage's relative output-directory template and logical artifacts.

    Attributes:
        name: Stable stage identifier, e.g. ``'s05_predict'``; must match
            the key under which the spec is registered in the layout.
        dir_template: Relative output-directory template, e.g.
            ``'s05_predict/{outcome}'`` for a per-outcome stage or
            ``'s01_outcome'`` for a static one. The only supported
            placeholder is ``{outcome}``; its presence (not a separate
            flag) marks the stage as per-outcome.
        artifacts: Logical artifact name -> filename for the artifacts this
            stage produces, e.g. ``{'train_df': 'train_df.parquet'}``. A
            filename may carry the ``{outcome}`` placeholder, e.g.
            ``{'display_table': '{outcome}_display.csv'}``; it is resolved
            with the same outcome validation as the directory template.
        prerequisites: Names of the stages whose outputs this stage depends
            on, e.g. ``('s01_outcome', 's02_feature')`` for a stage fed by
            the s01 and s02 outputs. Purely declarative metadata: the
            layout never resolves prerequisites itself — the project
            layer's ``build_layout()`` validates that every name is a
            registered stage and that the edges form an acyclic graph.
    """

    name: str
    dir_template: str
    artifacts: Mapping[str, str]
    prerequisites: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineLayout:
    """Resolves stage paths from a declarative stage table.

    ``stage_dir``/``artifact`` interpolate the ``{outcome}`` placeholder of
    the stage's ``dir_template`` and of an artifact filename. Validation
    happens before interpolation: a per-outcome template (or artifact
    filename) called without an outcome, or any call with an outcome outside
    ``outcome_names``, raises ``ValueError`` so an unvalidated name is never
    interpolated into a filesystem path.

    Attributes:
        stages: Stage table keyed by stage name (built by the project layer).
        outcome_names: The set of valid outcome names for ``{outcome}``
            interpolation. Declared by the project layer; an empty set
            rejects every outcome.
    """

    stages: Mapping[str, StageSpec]
    outcome_names: frozenset[str] = frozenset()

    def stage_dir(self, name: str, outcome: str | None = None) -> Path:
        """Return the stage's relative output directory.

        Raises:
            KeyError: No stage named ``name`` is registered.
            ValueError: The stage's ``dir_template`` requires ``{outcome}``
                but ``outcome`` is ``None``, or ``outcome`` is not one of
                the layout's declared outcome names.
        """
        spec = self.stages[name]
        self._validate_outcome(spec, outcome)
        return Path(spec.dir_template.format(outcome=outcome))

    def artifact(
        self,
        stage: str,
        logical: str,
        outcome: str | None = None,
    ) -> Path:
        """Return the relative path of one declared stage artifact.

        The path is the stage's resolved output directory joined with the
        declared filename for ``logical`` (``stage_dir`` + artifacts). The
        filename may itself carry ``{outcome}``, which is interpolated after
        the outcome has been validated.

        Raises:
            KeyError: No stage named ``stage`` or no artifact named
                ``logical`` is registered.
            ValueError: An outcome-validation failure, like ``stage_dir``,
                including an artifact filename that contains ``{outcome}``
                while ``outcome`` is ``None`` even when ``dir_template`` is
                static.
        """
        spec = self.stages[stage]
        self._validate_outcome(spec, outcome)
        filename = spec.artifacts[logical]
        if OUTCOME_PLACEHOLDER in filename:
            if outcome is None:
                raise ValueError(
                    f"artifact {logical!r} of stage {spec.name!r} requires "
                    f"an outcome (filename {filename!r} contains "
                    f"{OUTCOME_PLACEHOLDER!r}); pass outcome=<name>"
                )
            filename = filename.format(outcome=outcome)
        return Path(spec.dir_template.format(outcome=outcome)) / filename

    def _validate_outcome(self, spec: StageSpec, outcome: str | None) -> None:
        if OUTCOME_PLACEHOLDER in spec.dir_template and outcome is None:
            raise ValueError(
                f"stage {spec.name!r} requires an outcome (dir_template "
                f"{spec.dir_template!r} contains {OUTCOME_PLACEHOLDER!r}); "
                f"pass outcome=<name>"
            )
        if outcome is not None and outcome not in self.outcome_names:
            raise ValueError(
                f"unknown outcome {outcome!r} for stage {spec.name!r}; "
                f"known outcomes: {sorted(self.outcome_names)}"
            )


# --- core-side config protocol + pipeline context --------------------------


@runtime_checkable
class ProjectConfig(Protocol):
    """Core-side structural view of the project configuration.

    ``src.core`` keeps no import of ``src.project``; the frozen
    ``PipelineContext`` carries the project configuration to the stages
    through this Protocol instead. Its surface mirrors the frozen
    ``src.project.config.ProjectConfig`` dataclass exactly, so structural
    typing accepts the dataclass without an import, and the
    ``@runtime_checkable`` marker lets tests pin that conformance with an
    ``isinstance`` check.

    Every member is declared ``Final``, i.e. read-only, because the dataclass
    it mirrors is frozen: a writable Protocol member is invariant, so a
    frozen dataclass field (read-only) could not satisfy it and static
    checkers would reject ``PipelineContext(config=CONFIG)``. Read-only
    members also let the Protocol accept *both* frozen and mutable
    implementers.
    """

    data_root: Final[str]
    feature_columns_to_drop: Final[tuple[str, ...]]
    train_end_date: Final[datetime.date]
    test_start_date: Final[datetime.date]
    binary_thresholds: Final[tuple[float, ...]]
    evaluated_partitions: Final[tuple[str, ...]]
    feature_importance_plot_top_n: Final[int]
    demo_seed: Final[int]
    demo_case_count: Final[int]
    xgb_single_fit_params: Final[Mapping[str, Any]]
    xgb_num_boost_round: Final[int]
    xgb_early_stopping_rounds: Final[int]
    xgb_regressor_seed: Final[int]
    xgb_search_seed: Final[int]
    xgb_param_distributions: Final[Mapping[str, Any]]
    xgb_search_n_iter: Final[int]
    xgb_search_cv: Final[int]


@dataclass(frozen=True)
class PipelineContext:
    """The frozen, explicit configuration handed to every pipeline stage.

    Carries the declarative ``layout`` plus the project ``config`` in one
    value, so stages resolve their paths and read their settings from the
    context instead of module-level path constants. ``config`` is typed by
    the core-side ``ProjectConfig`` Protocol, keeping ``src.core`` free of
    any ``src.project`` import.

    Constructing a context never touches the filesystem and never depends on
    the current working directory. ``ensure`` is the context's one explicit
    side effect — a stage that is about to write under a directory calls it
    first.
    """

    layout: PipelineLayout
    config: ProjectConfig

    def ensure(self, path: Path) -> None:
        """Create the directory for ``path``, including any missing parents.

        The single explicit side effect of the context: a stage that is
        about to write under a directory must ``ensure`` it first; no path
        object creates a directory on import or construction.
        """
        path.mkdir(parents=True, exist_ok=True)
