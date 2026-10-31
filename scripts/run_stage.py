"""Command-line wrapper over the importable run-stage planner.

The planner in :mod:`src.project.run_stage` does the selection validation,
fan-out, and execution; this module only parses arguments, renders the
read-only modes, and maps a planner ``ValueError`` to an argparse usage error
(exit status ``2``). It contains no stage knowledge of its own.

Usage:

``python scripts/run_stage.py --list``
    Print the stage, model, and outcome inventory. This mode imports no stage
    module.
``python scripts/run_stage.py --stage s03_divide --dry-run``
    Print the ordered calls a run would execute without importing a stage.
``python scripts/run_stage.py --stage s06_evaluate --with-prereqs``
    Execute the selected stage together with its prerequisite closure.

The output root defaults to ``<repository>/output``. The project root is
derived from this file's location, never from the current working directory,
so the wrapper behaves the same from any directory.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project.pipeline import build_context  # noqa: E402
from src.project.run_stage import (  # noqa: E402
    StageSelection,
    plan_calls,
    render_list,
    render_plan,
    run_stages,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
"""Output root used when ``--output-root`` is not supplied."""


def build_parser() -> argparse.ArgumentParser:
    """Return the wrapper's argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_stage",
        description=(
            "List, plan, or execute pipeline stages through the importable "
            "run-stage planner."
        ),
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "stage to run; repeat the option to select several stages "
            "(required unless --list is given)"
        ),
    )
    parser.add_argument(
        "--outcome",
        metavar="NAME",
        help="restrict per-outcome stages to one registered outcome",
    )
    parser.add_argument(
        "--model",
        metavar="ID",
        help="restrict the training stage to one registered model strategy id",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        metavar="PATH",
        help="output root for the run (default: <repository>/output)",
    )
    parser.add_argument(
        "--with-prereqs",
        action="store_true",
        help="also run the selected stages' prerequisite closure",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--list",
        action="store_true",
        help="print the stage, model, and outcome inventory and exit",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned calls and exit without executing them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the wrapper and return the process exit status.

    Args:
        argv: Arguments to parse; ``None`` reads :data:`sys.argv`.

    Returns:
        ``0`` on success. Argument and planning failures exit with status
        ``2`` through :meth:`argparse.ArgumentParser.error`.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.list:
        print(render_list())
        return 0
    if not arguments.stage:
        parser.error("--stage is required unless --list is given")

    try:
        selection = StageSelection(
            stages=tuple(arguments.stage),
            outcome=arguments.outcome,
            model=arguments.model,
            with_prerequisites=arguments.with_prereqs,
        )
        calls = plan_calls(selection)
    except ValueError as error:
        parser.error(str(error))

    if arguments.dry_run:
        print(render_plan(calls))
        return 0

    output_root = (
        DEFAULT_OUTPUT_ROOT if arguments.output_root is None else arguments.output_root
    )
    run_stages(build_context(), output_root, calls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
