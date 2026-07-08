from __future__ import annotations

from typing import Sequence

from ..core.model import AnalysisStep, Edge, EdgeLoad, NodalLoad, Surface, SurfaceLoad


def nodal_load(
    step: AnalysisStep,
    target: str | int,
    component: int,
    value: float,
) -> NodalLoad:
    """Add a nodal load to a step using a 1-based component."""
    load = NodalLoad(target, component, value)
    step.cloads = tuple(step.cloads) + (load,)
    return load


def surface_traction(
    step: AnalysisStep,
    surface: str | Surface,
    vector: Sequence[float],
) -> SurfaceLoad:
    """Add a surface traction load to a step."""
    surface_name = surface.name if isinstance(surface, Surface) else str(surface)
    load = SurfaceLoad(surface_name, vector, load_type="traction")
    step.surface_loads = tuple(step.surface_loads) + (load,)
    return load


def surface_pressure(
    step: AnalysisStep,
    surface: str | Surface,
    magnitude: float,
) -> SurfaceLoad:
    """Add an inward pressure load to a step."""
    surface_name = surface.name if isinstance(surface, Surface) else str(surface)
    load = SurfaceLoad(surface_name, magnitude=magnitude, load_type="pressure")
    step.surface_loads = tuple(step.surface_loads) + (load,)
    return load


def edge_traction(
    step: AnalysisStep,
    edge: str | Edge,
    vector: Sequence[float],
) -> EdgeLoad:
    """Add an edge traction load to a step."""
    edge_name = edge.name if isinstance(edge, Edge) else str(edge)
    load = EdgeLoad(edge_name, vector, load_type="traction")
    step.edge_loads = tuple(step.edge_loads) + (load,)
    return load


def edge_pressure(
    step: AnalysisStep,
    edge: str | Edge,
    magnitude: float,
) -> EdgeLoad:
    """Add an inward pressure load to an edge collection."""
    edge_name = edge.name if isinstance(edge, Edge) else str(edge)
    load = EdgeLoad(edge_name, magnitude=magnitude, load_type="pressure")
    step.edge_loads = tuple(step.edge_loads) + (load,)
    return load
