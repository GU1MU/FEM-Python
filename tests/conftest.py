from __future__ import annotations

from collections import Counter
import importlib.util
from typing import Any

import pytest


_GMSH_FIXTURE_NAMES = frozenset({"live_gmsh", "real_gmsh"})
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
