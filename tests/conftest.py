from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
from typing import Any

import pytest


_PYTEST_TEMP_ROOT = (
    Path(__file__).resolve().parent
    / ".pytest-runtime"
    / "sandbox-compatible"
)
_GMSH_NUMERIC_OPTIONS = (
    "General.Terminal",
    "Mesh.ElementOrder",
    "Mesh.SecondOrderIncomplete",
    "Mesh.RecombineAll",
    "Mesh.MeshSizeFromPoints",
    "Mesh.MeshSizeFromCurvature",
    "Mesh.MeshSizeExtendFromBoundary",
    "Mesh.MeshSizeMin",
    "Mesh.MeshSizeMax",
    "Mesh.MeshSizeFactor",
    "Mesh.Algorithm",
    "Mesh.Algorithm3D",
    "Mesh.RecombinationAlgorithm",
    "Mesh.Recombine3DAll",
    "Mesh.SubdivisionAlgorithm",
)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    if (
        os.name != "nt"
        or config.option.basetemp is not None
        or "PYTEST_DEBUG_TEMPROOT" in os.environ
    ):
        return

    _PYTEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    temp_root = _PYTEST_TEMP_ROOT.resolve()
    original_mkdir = os.mkdir

    # Python 3.13 applies a restrictive ACL for mode 0o700 on Windows. The
    # sandbox token cannot reopen those directories, so inherit the workspace
    # ACL for pytest's own temporary tree.
    def sandbox_compatible_mkdir(
        path: Any,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        if dir_fd is None and mode == 0o700:
            candidate = Path(os.path.abspath(os.fsdecode(path)))
            if candidate.is_relative_to(temp_root):
                mode = 0o777

        if dir_fd is None:
            original_mkdir(path, mode)
        else:
            original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch = pytest.MonkeyPatch()
    config.add_cleanup(monkeypatch.undo)
    monkeypatch.setattr(os, "mkdir", sandbox_compatible_mkdir)
    monkeypatch.setenv("PYTEST_DEBUG_TEMPROOT", str(temp_root))


def _model_counts(gmsh: Any) -> Counter[str]:
    return Counter(str(name) for name in gmsh.model.list())


@pytest.fixture
def real_gmsh() -> Any:
    gmsh = pytest.importorskip(
        "gmsh",
        reason="the optional native Gmsh runtime is not installed",
    )
    owns_session = not bool(gmsh.isInitialized())
    if owns_session:
        gmsh.initialize()

    original_models = _model_counts(gmsh)
    original_current = str(gmsh.model.getCurrent())
    saved_options = {
        name: gmsh.option.getNumber(name) for name in _GMSH_NUMERIC_OPTIONS
    }
    gmsh.option.setNumber("General.Terminal", 0.0)

    try:
        yield gmsh
    finally:
        if bool(gmsh.isInitialized()):
            added_models = _model_counts(gmsh) - original_models
            for model_name in added_models.elements():
                gmsh.model.setCurrent(model_name)
                gmsh.model.remove()

            remaining_models = _model_counts(gmsh)
            if original_current in remaining_models:
                gmsh.model.setCurrent(original_current)
            for name, value in saved_options.items():
                gmsh.option.setNumber(name, value)

        if owns_session and bool(gmsh.isInitialized()):
            gmsh.finalize()
