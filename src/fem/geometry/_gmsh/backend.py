"""Lazy loading for the optional external Gmsh dependency."""

from __future__ import annotations

from typing import Any


_CAD_DEPENDENCY_MESSAGE = (
    "Gmsh geometry support requires the optional 'cad' dependencies. "
    'Install the project with: pip install -e ".[cad]"'
)


def load_gmsh() -> Any:
    """Load Gmsh when a geometry context is entered."""
    try:
        import gmsh
    except ModuleNotFoundError as exc:
        if exc.name != "gmsh":
            raise
        raise ModuleNotFoundError(_CAD_DEPENDENCY_MESSAGE) from exc
    return gmsh
