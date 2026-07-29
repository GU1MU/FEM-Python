"""Contour surface and visible-edge rendering helpers."""

from __future__ import annotations

from typing import Any

CONTOUR_RENDER_SHADED = "shaded"
CONTOUR_RENDER_FILLED = "filled"

CONTOUR_EDGE_ALL = "all"
CONTOUR_EDGE_EXTERIOR = "exterior"
CONTOUR_EDGE_FEATURE = "feature"
CONTOUR_EDGE_FREE = "free"
CONTOUR_EDGE_NONE = "none"

FEATURE_EDGE_ANGLE_DEGREES = 30.0


def contour_surface_options(
    render_mode: str,
    *,
    is_line_mesh: bool,
) -> dict[str, float | bool]:
    """Return PyVista lighting options for one contour render mode."""

    if render_mode == CONTOUR_RENDER_FILLED:
        return {
            "lighting": False,
            "smooth_shading": False,
        }
    if render_mode != CONTOUR_RENDER_SHADED:
        raise ValueError(f"unknown contour render mode: {render_mode}")
    return {
        "lighting": True,
        "smooth_shading": not is_line_mesh,
        "ambient": 0.35,
        "diffuse": 0.65,
        "specular": 0.0,
    }


def extract_contour_edges(dataset: Any, edge_mode: str) -> Any | None:
    """Extract Abaqus-style visible edges from a result dataset."""

    if edge_mode == CONTOUR_EDGE_NONE:
        return None
    if edge_mode == CONTOUR_EDGE_ALL:
        return dataset.extract_all_edges(clear_data=True)

    connected_geometry = (
        dataset.cast_to_unstructured_grid().clean()
    )
    surface = connected_geometry.extract_surface(
        algorithm="dataset_surface"
    ).clean()
    if edge_mode == CONTOUR_EDGE_EXTERIOR:
        return surface.extract_all_edges(clear_data=True)
    if edge_mode == CONTOUR_EDGE_FEATURE:
        return surface.extract_feature_edges(
            feature_angle=FEATURE_EDGE_ANGLE_DEGREES,
            boundary_edges=False,
            feature_edges=True,
            manifold_edges=False,
            non_manifold_edges=False,
            clear_data=True,
        )
    if edge_mode == CONTOUR_EDGE_FREE:
        return surface.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
            clear_data=True,
        )
    raise ValueError(f"unknown contour edge mode: {edge_mode}")


__all__ = [
    "CONTOUR_EDGE_ALL",
    "CONTOUR_EDGE_EXTERIOR",
    "CONTOUR_EDGE_FEATURE",
    "CONTOUR_EDGE_FREE",
    "CONTOUR_EDGE_NONE",
    "CONTOUR_RENDER_FILLED",
    "CONTOUR_RENDER_SHADED",
    "FEATURE_EDGE_ANGLE_DEGREES",
    "contour_surface_options",
    "extract_contour_edges",
]
