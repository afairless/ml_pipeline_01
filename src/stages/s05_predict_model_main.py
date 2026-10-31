"""Prediction stage entry point, driven by the pipeline context.

Accepts the frozen ``PipelineContext`` plus the executor-supplied output root.
Its output directory is resolved from the layout
(``layout.stage_dir('s05_predict', outcome_name)``) and created through the
context's explicit ``ensure``, so nothing re-derives a cwd-rooted path.
Trained model runs are read from their per-run manifests: ``load_model_runs``
discovers each model run's ``manifest.json`` structurally under
``layout.stage_dir('s04_train', outcome_name)`` (one immediate child directory
per run, holding all of that run's artifacts), and ``load_model_run`` resolves
the split tables, the model, and the run's identity from the manifest's
declared artifact inventory.

Each model run's predictions are written into its own run directory
``s05_predict/{outcome}/{model_name}/`` — the per-model run dir is resolved
from the layout (stage dir + ``model_name``, mirroring the s04 run-dir layout)
and ensured through the context — and the run is closed with its
``PredictionManifest``: ``save_prediction_run`` writes one partition artifact
pair per partition the model run's ``ModelManifest`` declares and assembles
the manifest, and this stage main persists it as ``manifest.json`` inside the
run directory. Prediction artifacts and their partitions are therefore both
declared, which is what s06 reads.

The stage shares the one ``(outcome_name, context, output_root)`` argument
shape with every other registry-driven stage.
"""

from __future__ import annotations

from pathlib import Path

from src.core.manifest import (
    MANIFEST_FILENAME,
    load_model_runs,
    write_manifest,
)
from src.core.paths import PipelineContext
from src.stages.s05_predict_model import load_model_run, save_prediction_run

STAGE_NAME = "s05_predict"
"""Stage-table key whose directory template the run directories resolve
under."""

S04_STAGE_NAME = "s04_train"
"""Stage-table key whose model runs this stage consumes."""

__all__ = ["S04_STAGE_NAME", "STAGE_NAME", "main", "s05_predict"]


def main(outcome_name: str, context: PipelineContext, output_root: Path) -> None:
    """Generate and save one outcome's model predictions.

    The s05 output directory is resolved from the layout under the
    executor-supplied ``output_root`` and created through the context's one
    explicit side effect, ``ensure``. The layout's ``{outcome}`` validation
    rejects an unknown outcome before it can be interpolated into a filesystem
    path.

    Args:
        outcome_name: The outcome whose model runs are predicted.
        context: The frozen pipeline context (layout + config) handed to every
            stage.
        output_root: The pipeline output root the relative layout paths are
            resolved under.

    Raises:
        FileNotFoundError: The outcome's s04 model stage directory or a
            declared model artifact is missing.
        ValueError: A manifest is inconsistent with its run directory or the
            loaded model, or the outcome is not declared by the layout.
        BoundaryError: An assembled prediction table violates
            ``S05_PREDICTION_SCHEMA``.
    """
    output_path = Path(output_root) / context.layout.stage_dir(STAGE_NAME, outcome_name)
    context.ensure(output_path)

    s04_model_stage_dir = Path(output_root) / context.layout.stage_dir(
        S04_STAGE_NAME, outcome_name
    )
    loaded_runs = [
        load_model_run(manifest, s04_model_stage_dir)
        for manifest in load_model_runs(s04_model_stage_dir, outcome_name)
    ]

    for model_run in loaded_runs:
        model_filename = model_run.manifest.model_name
        task = model_run.manifest.task

        run_dir = output_path / model_filename
        context.ensure(run_dir)

        partitions = model_run.manifest.partitions
        manifest = save_prediction_run(
            model_run.loaded_model,
            task,
            partitions,
            [model_run.key_data[partition] for partition in partitions],
            [model_run.feature_data[partition] for partition in partitions],
            model_filename,
            model_run.manifest.outcome,
            run_dir,
        )
        write_manifest(manifest, run_dir / MANIFEST_FILENAME)


s05_predict = main
"""Stage entry point under the name used by the registry executor."""
