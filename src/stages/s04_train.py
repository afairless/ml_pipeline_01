"""Train one model strategy for one modeled outcome.

This stage is the training entry point of the registry-driven half of the
pipeline. It owns exactly three things: it resolves the run directory
``s04_train/{outcome}/{strategy_id}`` from the declarative layout, it creates
that directory through ``context.ensure``, and it delegates the run to
:func:`src.project.training.train_model_run`. Everything else — loading the
s03 split tables, applying the strategy's feature policy, dispatching on the
strategy's trainer kind, and writing the partition tables, feature
importances, model serializations, and the model manifest — belongs to the
project training driver, so this module stays a thin stage shell.

There is one entry point, :func:`main`, and the registry calls it once per
entry in :data:`~src.project.models.MODEL_STRATEGIES` with the strategy id as
its leading positional argument. The run-directory name, the manifest
``model_name``, and the ``run_stage --model`` vocabulary therefore all read
from that one strategy table; adding a model is a table entry, not a new
module.

The strategy id is validated before the run directory is created, so a typo
fails without leaving an empty run behind. Every other failure surfaces from
the training driver after the directory exists: a missing s03 prerequisite or
a violated boundary leaves the run directory empty rather than partially
populated, because the driver validates before saving and the writer replaces
whole files.

The three responsibilities are kept separate even though they meet in the
driver: *ingestion* reads and validates the s03 split tables through the
layout's logical artifact names; *transformation* is the feature-policy
substitution and the estimator fit; *output* is the artifact and manifest
writer, which is the only code allowed to change representation. Re-running
the stage replaces only the artifacts this run declares under
``s04_train/{outcome}/{strategy_id}/`` and never touches another strategy's
run or another stage's output.

``src/project/adapters.py`` remains the replacement point for a new project:
this stage consumes whatever divided tables those adapters lead to as long as
the frames satisfy the declared boundary schemas.
"""

from __future__ import annotations

from pathlib import Path

from src.core.paths import PipelineContext
from src.project.models import MODEL_STRATEGIES
from src.project.training import train_model_run

STAGE_NAME = "s04_train"
"""Stage-table key whose directory template the run directory is resolved
under."""


def model_run_dir(
    strategy_id: str,
    outcome_name: str,
    context: PipelineContext,
    output_root: Path,
) -> Path:
    """Resolve one model run's directory without creating it.

    The directory is the stage directory the layout declares for the outcome,
    joined with the strategy id. Resolving it is pure: the caller creates it
    with ``context.ensure`` only once it is about to write, so importing or
    planning a run never touches the filesystem.

    Args:
        strategy_id: The model strategy id; a key of
            :data:`~src.project.models.MODEL_STRATEGIES`.
        outcome_name: The modeled outcome the run belongs to.
        context: The frozen pipeline context (layout + config).
        output_root: The output root the relative layout path resolves under.

    Returns:
        The absolute-or-relative run directory path; the strategy subdirectory
        may not exist yet.

    Raises:
        ValueError: ``strategy_id`` is not a registered model strategy, or
            ``outcome_name`` is not a declared pipeline outcome.
    """
    if strategy_id not in MODEL_STRATEGIES:
        raise ValueError(
            f"unknown model strategy {strategy_id!r}; known strategies: "
            f"{', '.join(MODEL_STRATEGIES)}"
        )
    return (
        Path(output_root)
        / context.layout.stage_dir(STAGE_NAME, outcome_name)
        / strategy_id
    )


def main(
    strategy_id: str,
    outcome_name: str,
    context: PipelineContext,
    output_root: Path,
) -> None:
    """Train one registered model strategy and persist its run.

    Creates ``s04_train/{outcome}/{strategy_id}`` through ``context.ensure``
    and hands the run to :func:`src.project.training.train_model_run`. The
    driver writes the run's partition tables, feature importances, model
    serializations, and ``manifest.json``.

    Args:
        strategy_id: The model strategy id, validated before any directory is
            created.
        outcome_name: The modeled outcome, which is also the target column
            name the driver reads from the s03 split tables.
        context: The frozen pipeline context (layout + config).
        output_root: The output root the relative layout paths resolve under.

    Raises:
        ValueError: ``strategy_id`` is not a registered model strategy.
        FileNotFoundError: The outcome's s03 split artifacts are missing.
        BoundaryError: An input or output table violates its declared schema.
    """
    run_dir = model_run_dir(strategy_id, outcome_name, context, output_root)
    context.ensure(run_dir)
    train_model_run(strategy_id, outcome_name, context, output_root)


s04_train = main
"""Stage entry point under the name used by the stage table's module path."""
