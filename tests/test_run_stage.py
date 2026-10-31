"""Unit and smoke tests for the run-stage planner and its CLI wrapper.

Pins the planner contract in ``src/project/run_stage.py``: the vocabulary
helpers track the stage, model, and outcome registries; the registry
consistency check refuses an inconsistent declaration; a selection is
validated before planning; planning expands outcome-free, per-outcome, and
per-model stages in table order; prerequisite closure restores run order; the
renderings are deterministic; and execution resolves stage entry points
lazily. The smoke tests drive ``scripts/run_stage.py`` through ``--list`` and
``--dry-run`` and check that both work without importing a stage.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType

import pytest

from scripts import run_stage as cli
from src.core.paths import PipelineContext
from src.project import run_stage as planner
from src.project.models import MODEL_STRATEGIES
from src.project.outcomes import OUTCOMES, modeled_outcomes, outcome_names
from src.project.pipeline import (
    EXPLICIT_STAGES,
    STAGES,
    build_context,
)
from src.project.run_stage import (
    PlannedCall,
    StageSelection,
    allowed_outcomes,
    check_registry_consistency,
    defined_stages,
    explicit_stages,
    model_ids,
    plan_calls,
    render_list,
    render_plan,
    run_stages,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_PATH = PROJECT_ROOT / "scripts" / "run_stage.py"

OUTCOME_NAMES = outcome_names()
MODELED_OUTCOMES = tuple(outcome.name for outcome in modeled_outcomes())


# --- vocabulary helpers ----------------------------------------------------


def test_defined_stages_reports_the_stage_table_in_order() -> None:
    assert defined_stages() == tuple(STAGES)


def test_explicit_stages_reports_the_explicit_split() -> None:
    assert explicit_stages() == EXPLICIT_STAGES


def test_model_ids_follow_the_strategy_table() -> None:
    assert model_ids() == tuple(MODEL_STRATEGIES)


def test_allowed_outcomes_follow_the_outcome_registry() -> None:
    assert allowed_outcomes() == OUTCOME_NAMES


def test_registry_is_consistent() -> None:
    assert check_registry_consistency() is None


# --- StageSelection --------------------------------------------------------


def test_stage_selection_is_frozen() -> None:
    selection = StageSelection(stages=("s01_outcome",))
    with pytest.raises(FrozenInstanceError):
        selection.outcome = "outcome_1"  # type: ignore[misc]


def test_stage_selection_accepts_registered_vocabulary() -> None:
    selection = StageSelection(
        stages=("s04_train",),
        outcome=MODELED_OUTCOMES[0],
        model=model_ids()[0],
        with_prerequisites=True,
    )
    assert selection.stages == ("s04_train",)
    assert selection.outcome == MODELED_OUTCOMES[0]
    assert selection.model == model_ids()[0]
    assert selection.with_prerequisites is True


def test_stage_selection_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        StageSelection(stages=("s99_nope",))


def test_stage_selection_rejects_repeated_stage() -> None:
    with pytest.raises(ValueError, match="repeats a stage name"):
        StageSelection(stages=("s01_outcome", "s01_outcome"))


def test_stage_selection_rejects_unknown_outcome() -> None:
    with pytest.raises(ValueError, match="unknown outcome"):
        StageSelection(outcome="outcome_99")


def test_stage_selection_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        StageSelection(model="model_99")


# --- plan_calls ------------------------------------------------------------


def test_empty_selection_plans_no_calls() -> None:
    assert plan_calls(StageSelection()) == ()


def test_outcome_free_stage_is_planned_once() -> None:
    calls = plan_calls(StageSelection(stages=("s01_outcome",)))
    assert calls == (PlannedCall(stage="s01_outcome"),)


def test_outcome_filter_is_ignored_by_outcome_free_stage() -> None:
    calls = plan_calls(
        StageSelection(stages=("s01_outcome",), outcome=OUTCOME_NAMES[0]),
    )
    assert calls == (PlannedCall(stage="s01_outcome"),)


def test_per_outcome_stage_expands_in_registry_order() -> None:
    calls = plan_calls(StageSelection(stages=("s03_divide",)))
    assert calls == tuple(
        PlannedCall(stage="s03_divide", outcome=n) for n in OUTCOME_NAMES
    )


def test_training_stage_expands_over_modeled_outcomes_and_models() -> None:
    calls = plan_calls(StageSelection(stages=("s04_train",)))
    expected = tuple(
        PlannedCall(stage="s04_train", outcome=outcome, model=model)
        for outcome in MODELED_OUTCOMES
        for model in model_ids()
    )
    assert calls == expected


def test_training_stage_honours_the_outcome_filter() -> None:
    outcome = MODELED_OUTCOMES[0]
    calls = plan_calls(StageSelection(stages=("s04_train",), outcome=outcome))
    assert calls == tuple(
        PlannedCall(stage="s04_train", outcome=outcome, model=model)
        for model in model_ids()
    )


def test_training_stage_honours_the_model_filter() -> None:
    model = model_ids()[-1]
    calls = plan_calls(StageSelection(stages=("s04_train",), model=model))
    assert calls == tuple(
        PlannedCall(stage="s04_train", outcome=outcome, model=model)
        for outcome in MODELED_OUTCOMES
    )


def test_model_stage_rejects_a_non_modeled_outcome() -> None:
    analysis_outcome = next(n for n in OUTCOME_NAMES if n not in MODELED_OUTCOMES)
    with pytest.raises(ValueError, match="is not one of them"):
        plan_calls(StageSelection(stages=("s04_train",), outcome=analysis_outcome))


def test_model_filter_without_the_training_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="only applies to the training stage"):
        plan_calls(StageSelection(stages=("s03_divide",), model=model_ids()[0]))


def test_prerequisite_closure_restores_run_order() -> None:
    calls = plan_calls(
        StageSelection(stages=("s06_evaluate",), with_prerequisites=True),
    )
    stages = [call.stage for call in calls]
    assert stages[0] == "s01_outcome"
    assert stages[-1] == "s06_evaluate"
    assert set(stages) == {
        "s01_outcome",
        "s02_feature",
        "s03_divide",
        "s04_train",
        "s05_predict",
        "s06_evaluate",
    }
    assert stages == sorted(stages, key=defined_stages().index)


def test_prerequisite_closure_includes_explicit_prerequisites() -> None:
    calls = plan_calls(
        StageSelection(stages=("s01b_outcome_analysis",), with_prerequisites=True),
    )
    assert [call.stage for call in calls] == [
        "s01_outcome",
        "s01b_outcome_analysis",
    ]


def test_without_prerequisites_only_the_selection_is_planned() -> None:
    calls = plan_calls(StageSelection(stages=("s06_evaluate",)))
    assert {call.stage for call in calls} == {"s06_evaluate"}


def test_selection_order_does_not_change_plan_order() -> None:
    forward = plan_calls(StageSelection(stages=("s03_divide", "s05_predict")))
    reverse = plan_calls(StageSelection(stages=("s05_predict", "s03_divide")))
    assert forward == reverse
    assert list(dict.fromkeys(call.stage for call in forward)) == [
        "s03_divide",
        "s05_predict",
    ]


# --- PlannedCall and rendering --------------------------------------------


def test_planned_call_is_frozen() -> None:
    call = PlannedCall(stage="s01_outcome")
    with pytest.raises(FrozenInstanceError):
        call.stage = "s02_feature"  # type: ignore[misc]


def test_planned_call_rejects_an_unknown_stage() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        PlannedCall(stage="s99_nope")


def test_planned_call_render_includes_present_fields() -> None:
    assert PlannedCall(stage="s01_outcome").render() == "s01_outcome"
    assert (
        PlannedCall(stage="s03_divide", outcome="outcome_1").render()
        == "s03_divide outcome_1"
    )
    assert (
        PlannedCall(
            stage="s04_train",
            outcome="outcome_1",
            model="model_01a",
        ).render()
        == "s04_train outcome_1 model_01a"
    )


def test_render_plan_lists_one_call_per_line() -> None:
    calls = (
        PlannedCall(stage="s03_divide", outcome="outcome_1"),
        PlannedCall(stage="s04_train", outcome="outcome_1", model="model_01a"),
    )
    assert render_plan(calls) == "s03_divide outcome_1\ns04_train outcome_1 model_01a"


def test_render_plan_for_an_empty_plan_is_explicit() -> None:
    assert render_plan(()) == "(no calls planned)"


def test_render_list_names_every_stage_model_and_outcome() -> None:
    text = render_list()
    lines = text.splitlines()
    assert lines[0] == "Stages:"
    for name in defined_stages():
        assert any(line.strip().startswith(name) for line in lines)
    assert "Models:" in lines
    for model in model_ids():
        assert f"  {model}" in lines
    assert "Outcomes:" in lines
    for outcome in OUTCOME_NAMES:
        kind = "modeled" if OUTCOMES[outcome].modeled else "analysis"
        assert any(
            line.split()[0] == outcome and line.split()[1] == kind for line in lines
        )


def test_render_list_is_deterministic() -> None:
    assert render_list() == render_list()


# --- lazy imports ----------------------------------------------------------


def test_importing_the_planner_imports_no_stage_or_estimator() -> None:
    code = (
        "import sys\n"
        "import src.project.run_stage\n"
        "stage_modules = [n for n in sys.modules if n.startswith('src.stages.')]\n"
        "assert not stage_modules, stage_modules\n"
        "estimator_modules = "
        "[n for n in sys.modules if n.startswith('src.core.estimators')]\n"
        "assert not estimator_modules, estimator_modules\n"
        "assert 'xgboost' not in sys.modules\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


# --- run_stages ------------------------------------------------------------


def _install_fake_stage(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> list[tuple[object, ...]]:
    """Install a recording fake stage module and return its call log."""
    module = ModuleType(module_name)
    calls: list[tuple[object, ...]] = []

    def main(*args: object) -> None:
        calls.append(args)

    module.main = main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    return calls


def test_run_stages_passes_standard_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    divide_calls = _install_fake_stage(
        monkeypatch,
        STAGES["s03_divide"].module,
    )
    context = build_context()
    output_root = Path("out")
    run_stages(
        context,
        output_root,
        (PlannedCall(stage="s03_divide", outcome="outcome_1"),),
    )
    assert divide_calls == [("outcome_1", context, output_root)]


def test_run_stages_passes_outcome_free_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome_calls = _install_fake_stage(monkeypatch, STAGES["s01_outcome"].module)
    context = build_context()
    output_root = Path("out")
    run_stages(context, output_root, (PlannedCall(stage="s01_outcome"),))
    assert outcome_calls == [(context, output_root)]


def test_run_stages_prefixes_the_model_id_for_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_calls = _install_fake_stage(monkeypatch, STAGES["s04_train"].module)
    context = build_context()
    output_root = Path("out")
    run_stages(
        context,
        output_root,
        (
            PlannedCall(
                stage="s04_train",
                outcome="outcome_1",
                model="model_01a",
            ),
        ),
    )
    assert train_calls == [("model_01a", "outcome_1", context, output_root)]


def test_run_stages_executes_calls_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("tests._fake_run_stage_order")
    order: list[str] = []

    def main(*args: object) -> None:
        order.append(str(args[0]))

    module.main = main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        planner,
        "_resolve_stage",
        lambda name: module.main if name in STAGES else None,
    )
    run_stages(
        build_context(),
        Path("out"),
        (
            PlannedCall(stage="s03_divide", outcome="outcome_1"),
            PlannedCall(stage="s03_divide", outcome="outcome_2"),
        ),
    )
    assert order == ["outcome_1", "outcome_2"]


def test_run_stages_rejects_training_call_without_model() -> None:
    with pytest.raises(ValueError, match="needs both a model id and an outcome"):
        run_stages(
            build_context(),
            Path("out"),
            (PlannedCall(stage="s04_train", outcome="outcome_1"),),
        )


def test_run_stages_rejects_a_model_on_another_stage() -> None:
    with pytest.raises(ValueError, match="takes no model id"):
        run_stages(
            build_context(),
            Path("out"),
            (
                PlannedCall(
                    stage="s03_divide",
                    outcome="outcome_1",
                    model="model_01a",
                ),
            ),
        )


def test_run_stages_accepts_an_empty_call_list() -> None:
    assert run_stages(build_context(), Path("out"), ()) is None


# --- CLI smoke tests -------------------------------------------------------


def test_cli_list_prints_the_inventory(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--list"]) == 0
    output = capsys.readouterr().out
    assert output.splitlines()[0] == "Stages:"
    for name in defined_stages():
        assert name in output
    for model in model_ids():
        assert model in output
    for outcome in OUTCOME_NAMES:
        assert outcome in output


def test_cli_list_runs_as_a_script() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), "--list"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines()[0] == "Stages:"


def test_cli_dry_run_prints_one_call(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = model_ids()[0]
    assert (
        cli.main(
            [
                "--stage",
                "s04_train",
                "--outcome",
                MODELED_OUTCOMES[0],
                "--model",
                model,
                "--dry-run",
            ],
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == (
        f"s04_train {MODELED_OUTCOMES[0]} {model}"
    )


def test_cli_dry_run_expands_prerequisites(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--stage", "s06_evaluate", "--with-prereqs", "--dry-run"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("s01_outcome")
    assert lines[-1].startswith("s06_evaluate ")
    assert any(line.startswith("s03_divide ") for line in lines)
    assert any(line.startswith("s04_train ") for line in lines)


def test_cli_executes_the_planned_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[tuple[PipelineContext, Path, tuple[PlannedCall, ...]]] = []

    def fake_run_stages(
        context: PipelineContext,
        output_root: Path,
        calls: tuple[PlannedCall, ...],
    ) -> None:
        seen.append((context, output_root, tuple(calls)))

    monkeypatch.setattr(cli, "run_stages", fake_run_stages)
    assert (
        cli.main(
            ["--stage", "s01_outcome", "--output-root", str(tmp_path)],
        )
        == 0
    )
    assert len(seen) == 1
    _, output_root, calls = seen[0]
    assert output_root == tmp_path
    assert calls == (PlannedCall(stage="s01_outcome"),)


def test_cli_requires_a_stage_without_list(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2
    assert "--stage is required" in capsys.readouterr().err


def test_cli_maps_a_planner_error_to_exit_status_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--stage", "s99_nope"])
    assert exit_info.value.code == 2
    assert "unknown stage" in capsys.readouterr().err


def test_cli_list_and_dry_run_are_mutually_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--list", "--dry-run"])
    assert exit_info.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_cli_list_imports_no_stage_module() -> None:
    code = (
        "import sys\n"
        "import runpy\n"
        "sys.argv = ['run_stage.py', '--list']\n"
        "try:\n"
        "    runpy.run_path('scripts/run_stage.py', run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 0, exc.code\n"
        "stage_modules = [n for n in sys.modules if n.startswith('src.stages.')]\n"
        "assert not stage_modules, stage_modules\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
