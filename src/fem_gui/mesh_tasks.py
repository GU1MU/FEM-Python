"""Detached mesh generation shared by manual and Agent commands."""

from __future__ import annotations

from time import perf_counter

from fem.application import MeshTaskSnapshot
from fem.application.preprocessing import generate_fem_model
from fem.geometry.gmsh_coordinator import GmshExecutionCancelled

from .visualization.model_adapter import ModelGeometry, build_model_geometry
from .workers import TaskContext


def prepare_mesh_display(
    task: MeshTaskSnapshot, context: TaskContext,
) -> tuple[object, ModelGeometry, dict[str, float]]:
    """Generate and prepare display adapters without changing a Session."""

    context.report("Generating mesh...")
    context.checkpoint()
    started = perf_counter()
    try:
        model = generate_fem_model(task, cancelled=lambda: context.is_cancelled)
    except GmshExecutionCancelled:
        context.checkpoint()
        raise
    timings = {"Gmsh geometry and mesh": perf_counter() - started}
    context.checkpoint()
    context.report("Preparing display mesh...")
    started = perf_counter()
    display_geometry = build_model_geometry(model)
    timings["VTK display geometry construction"] = perf_counter() - started
    context.checkpoint()
    return model, display_geometry, timings
