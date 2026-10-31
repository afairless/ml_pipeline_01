"""Import contract test for the src packages.

Pins the invariant that every module under ``src/`` — ``src.core.*``,
``src.project.*``, and (once they exist) ``src.stages.*`` — imports cleanly,
and that the project-agnostic ``src.core`` package carries no
``src.project`` and no ``src.stages`` import. A dangling or backwards
import re-point fails here before it can reach a stage run.

``src.stages`` is allowed to be empty while the port is in progress: the
stage-module rule only asserts that no core module imports a stage module,
so an absent or empty ``src/stages`` package does not fail this test.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil

import pytest

import src


def _discover_src_module_names() -> list[str]:
    """Return one fully-qualified name per module under ``src/``, sorted.

    Walks ``src.__path__`` with ``pkgutil`` so the list tracks the on-disk
    layout automatically: a module added later is picked up without editing
    this file. ``src`` itself is prepended so the top-level package is also
    import-checked.
    """
    names = ["src"] + sorted(
        m.name for m in pkgutil.walk_packages(src.__path__, prefix="src.")
    )
    return names


SRC_MODULE_NAMES = _discover_src_module_names()

CORE_MODULE_NAMES = [
    name
    for name in SRC_MODULE_NAMES
    if name == "src.core" or name.startswith("src.core.")
]


def _is_forbidden_layer(module_name: str, layer: str) -> bool:
    """True when ``module_name`` is ``layer`` or nested inside it."""
    return module_name == layer or module_name.startswith(layer + ".")


def _forbidden_import(source: str, layer: str) -> str | None:
    """Return a textual repr of the first forbidden import, else None.

    Parses the module's source with ``ast`` (rather than grepping the text) so
    docstring/comment mentions of the layer cannot trip the check.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_layer(alias.name, layer):
                    return f"import {alias.name}"
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and _is_forbidden_layer(node.module, layer)
        ):
            imported = ", ".join(alias.name for alias in node.names)
            return f"from {node.module} import {imported}"
    return None


@pytest.mark.parametrize("module_name", SRC_MODULE_NAMES)
def test_every_src_module_imports_cleanly(module_name: str) -> None:
    """Import each module under ``src/``, incl. core and project packages."""
    imported = importlib.import_module(module_name)
    assert imported is not None


@pytest.mark.parametrize("core_module_name", CORE_MODULE_NAMES)
def test_core_modules_carry_no_project_imports(core_module_name: str) -> None:
    """Project-agnostic core modules must not import src.project."""
    module = importlib.import_module(core_module_name)
    forbidden = _forbidden_import(inspect.getsource(module), "src.project")
    assert forbidden is None, (
        f"{core_module_name} must not import src.project; found: {forbidden}"
    )


@pytest.mark.parametrize("core_module_name", CORE_MODULE_NAMES)
def test_core_modules_carry_no_stage_imports(core_module_name: str) -> None:
    """Core must not import a stage module; stages depend on core, not the
    reverse. An empty ``src/stages`` package is fine: this only inspects
    imports, so it never requires stage modules to exist."""
    module = importlib.import_module(core_module_name)
    forbidden = _forbidden_import(inspect.getsource(module), "src.stages")
    assert forbidden is None, (
        f"{core_module_name} must not import src.stages; found: {forbidden}"
    )
