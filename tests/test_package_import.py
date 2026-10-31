"""Smoke test for the package skeleton."""

import src
import src.core
import src.project
import src.stages


def test_package_modules_import() -> None:
    assert src.__doc__ is not None
    assert src.core.__doc__ is not None
    assert src.project.__doc__ is not None
    assert src.stages.__doc__ is not None
