from __future__ import annotations

from collections import Counter
import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest


_GMSH_FIXTURE_NAMES = frozenset({"live_gmsh", "real_gmsh"})
_PYTEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "temp" / "pytest-runtime"
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


def _requires_native_gmsh(item: pytest.Item) -> bool:
    test_name = str(getattr(item, "originalname", item.name)).split("[", 1)[0]
    fixture_names = set(getattr(item, "fixturenames", ()))
    return (
        item.get_closest_marker("gmsh") is not None
        or test_name.startswith("test_real_")
        or bool(fixture_names & _GMSH_FIXTURE_NAMES)
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    gmsh_available = importlib.util.find_spec("gmsh") is not None
    missing_gmsh = pytest.mark.skip(
        reason="the optional native Gmsh runtime is not installed"
    )
    for item in items:
        if not _requires_native_gmsh(item):
            continue
        item.add_marker(pytest.mark.integration)
        item.add_marker(pytest.mark.gmsh)
        if not gmsh_available:
            item.add_marker(missing_gmsh)


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
