"""Run the complete machine-learning pipeline on the committed demo data.

The entry point keeps the three preprocessing stages explicit because they
either ingest external data or run once per pipeline run, and it delegates the
remaining stages to the declarative orchestration registry:

* ``s01_outcome`` reads the external source table and builds the outcome table;
* ``s01b_outcome_analysis`` runs once over the registered outcome comparisons;
* ``s02_feature`` reads the external source table again and builds the feature
  table;
* ``s03_divide`` through ``s06_evaluate`` are registry-driven, so their
  outcome and model fan-out comes from the stage, outcome, and model tables
  rather than from code in this module.

The project root is derived from this file's location, never from the current
working directory, and the output root is the ``output/`` directory beside the
``src/`` package. Every stage resolves its own relative paths through the
frozen pipeline context it receives, so running the module from any working
directory produces the same tree.
"""

from __future__ import annotations

from pathlib import Path

from src.project.config import CONFIG
from src.project.pipeline import build_context, run_pipeline
from src.stages.s01_outcome import s01_outcome
from src.stages.s01b_outcome_analysis import s01b_outcome_analysis
from src.stages.s02_feature import s02_feature

__all__ = ["main"]


def main() -> None:
    """Run every pipeline stage once against the demo project root.

    The first three stages are called explicitly because they read external
    data or run once per run. ``run_pipeline`` then executes the
    registry-driven stages in stage-table order, fanning them out over the
    outcome and model registries. The output root is the project's
    ``output/`` directory.
    """
    project_root = Path(__file__).resolve().parent.parent
    context = build_context(config=CONFIG, project_root=project_root)
    output_root = project_root / "output"
    s01_outcome(context, output_root)
    s01b_outcome_analysis(context, output_root)
    s02_feature(context, output_root)
    run_pipeline(context, output_root)


if __name__ == "__main__":
    main()
