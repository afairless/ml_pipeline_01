"""Smoke tests for the ``src/main.py`` entry point and its ``__main__`` guard.

The tests pin three properties: importing the module never runs the pipeline,
calling ``main`` executes the explicit stages in their declared order before
the registry-driven executor, and running the module with ``python -m
src.main`` executes the guard. The guard test builds a tiny temporary package
whose module tree mirrors the real one with stub stages, so it exercises the
real ``src/main.py`` source without paying for another full demo run.
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import src.main as main_module
import src.project.pipeline as pipeline_module
import src.stages.s01_outcome as s01_outcome_module
import src.stages.s01b_outcome_analysis as s01b_module
import src.stages.s02_feature as s02_feature_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Repository root holding the real ``src/main.py`` source."""

EXPECTED_CALL_ORDER = (
    "s01_outcome",
    "s01b_outcome_analysis",
    "s02_feature",
    "run_pipeline",
)
"""The entry point's fixed stage sequence."""


def _record(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Replace the real stage and executor with call-recording stubs."""

    def record(name: str):
        def stub(*args: object, **kwargs: object) -> None:
            del args, kwargs
            calls.append(name)

        return stub

    monkeypatch.setattr(main_module, "s01_outcome", record("s01_outcome"))
    monkeypatch.setattr(
        main_module,
        "s01b_outcome_analysis",
        record("s01b_outcome_analysis"),
    )
    monkeypatch.setattr(main_module, "s02_feature", record("s02_feature"))
    monkeypatch.setattr(main_module, "run_pipeline", record("run_pipeline"))


def _record_imported_sources(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
) -> None:
    """Patch the modules that ``src.main`` imports from when re-executed.

    ``runpy`` executes the module source again, so patching only the already
    imported ``src.main`` attributes would be overwritten by the source's
    ``from ... import ...`` statements. Patching the defining modules keeps
    the guard test fast while still executing the real entry-point module.
    """

    def record(name: str):
        def stub(*args: object, **kwargs: object) -> None:
            del args, kwargs
            calls.append(name)

        return stub

    monkeypatch.setattr(
        s01_outcome_module,
        "s01_outcome",
        record("s01_outcome"),
    )
    monkeypatch.setattr(
        s01b_module,
        "s01b_outcome_analysis",
        record("s01b_outcome_analysis"),
    )
    monkeypatch.setattr(
        s02_feature_module,
        "s02_feature",
        record("s02_feature"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_pipeline",
        record("run_pipeline"),
    )


def test_importing_main_does_not_execute_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the entry-point module must have no side effects."""
    calls: list[str] = []
    _record_imported_sources(monkeypatch, calls)

    sys.modules.pop("src.main", None)
    runpy.run_module("src.main")

    assert calls == []


def test_main_calls_the_explicit_stages_before_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main()`` runs s01, s01b, s02, then the registry executor."""
    calls: list[str] = []
    _record(monkeypatch, calls)

    main_module.main()

    assert tuple(calls) == EXPECTED_CALL_ORDER


def test_main_passes_the_project_root_output_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every stage receives the project's ``output/`` directory."""
    output_roots: list[Path] = []

    def record_output(*args: object) -> None:
        assert args
        output_roots.append(args[-1])  # type: ignore[arg-type]

    monkeypatch.setattr(main_module, "s01_outcome", record_output)
    monkeypatch.setattr(main_module, "s01b_outcome_analysis", record_output)
    monkeypatch.setattr(main_module, "s02_feature", record_output)
    monkeypatch.setattr(main_module, "run_pipeline", record_output)

    main_module.main()

    assert output_roots == [PROJECT_ROOT / "output"] * len(EXPECTED_CALL_ORDER)


def test_running_the_module_as_main_invokes_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-executing the module as ``__main__`` calls ``main()``."""
    calls: list[str] = []
    _record_imported_sources(monkeypatch, calls)

    sys.modules.pop("src.main", None)
    runpy.run_module("src.main", run_name="__main__")

    assert tuple(calls) == EXPECTED_CALL_ORDER


def _write_stub_package(root: Path) -> None:
    """Build a temporary package tree that mirrors the real import paths."""
    src = root / "src"
    project = src / "project"
    stages = src / "stages"
    for directory in (src, project, stages):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")

    (src / "main.py").write_text(
        (PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (project / "config.py").write_text(
        "CONFIG = object()\n",
        encoding="utf-8",
    )
    (project / "pipeline.py").write_text(
        "def build_context(config=None, project_root=None):\n"
        "    del config, project_root\n"
        "    return object()\n"
        "\n"
        "def run_pipeline(context, output_root):\n"
        "    del context, output_root\n"
        '    print("run_pipeline")\n',
        encoding="utf-8",
    )
    for module_name, callable_name in (
        ("s01_outcome", "s01_outcome"),
        ("s01b_outcome_analysis", "s01b_outcome_analysis"),
        ("s02_feature", "s02_feature"),
    ):
        (stages / f"{module_name}.py").write_text(
            f"def {callable_name}(context, output_root):\n"
            "    del context, output_root\n"
            f'    print("{callable_name}")\n',
            encoding="utf-8",
        )


def test_python_dash_m_executes_the_real_guard(tmp_path: Path) -> None:
    """``python -m src.main`` runs the guard in a real interpreter."""
    _write_stub_package(tmp_path)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-m", "src.main"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == list(EXPECTED_CALL_ORDER)
