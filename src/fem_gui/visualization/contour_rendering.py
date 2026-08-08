"""Contour surface and visible-edge rendering helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

CONTOUR_RENDER_SHADED = "shaded"
CONTOUR_RENDER_FILLED = "filled"

CONTOUR_EDGE_ALL = "all"
CONTOUR_EDGE_EXTERIOR = "exterior"
CONTOUR_EDGE_GEOMETRY = "geometry"
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
    if edge_mode == CONTOUR_EDGE_GEOMETRY:
        # Open surface boundaries preserve planar outer and hole contours;
        # angular features preserve solid-body geometric edges.
        return surface.extract_feature_edges(
            feature_angle=FEATURE_EDGE_ANGLE_DEGREES,
            boundary_edges=True,
            feature_edges=True,
            manifold_edges=False,
            non_manifold_edges=False,
            clear_data=True,
        )
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


def style_contour_edges(edges: Any, edge_style: str) -> Any:
    """Build real line gaps for styles unsupported by VTK OpenGL2."""

    if edge_style in {"solid", "bold"}:
        return edges
    dash_pattern = {
        "dashed": (0.58, 0.28),
        "short_dashed": (0.22, 0.14),
    }.get(edge_style)
    if dash_pattern is None:
        raise ValueError(f"unknown contour edge style: {edge_style}")

    dash_fraction, gap_fraction = dash_pattern
    source_points = np.asarray(edges.points)
    connectivity = np.asarray(edges.lines, dtype=np.int64).ravel()
    styled_points: list[np.ndarray] = []
    styled_lines: list[int] = []

    cursor = 0
    while cursor < connectivity.size:
        point_count = int(connectivity[cursor])
        point_ids = connectivity[cursor + 1 : cursor + 1 + point_count]
        for start_id, end_id in zip(point_ids[:-1], point_ids[1:]):
            start = source_points[int(start_id)]
            end = source_points[int(end_id)]
            direction = end - start
            position = 0.0
            while position < 1.0:
                dash_end = min(position + dash_fraction, 1.0)
                first_id = len(styled_points)
                styled_points.extend(
                    (
                        start + direction * position,
                        start + direction * dash_end,
                    )
                )
                styled_lines.extend((2, first_id, first_id + 1))
                position = dash_end + gap_fraction
        cursor += point_count + 1

    return type(edges)(
        np.asarray(styled_points, dtype=source_points.dtype),
        lines=np.asarray(styled_lines, dtype=np.int64),
    )


__all__ = [
    "CONTOUR_EDGE_ALL",
    "CONTOUR_EDGE_EXTERIOR",
    "CONTOUR_EDGE_FEATURE",
    "CONTOUR_EDGE_FREE",
    "CONTOUR_EDGE_GEOMETRY",
    "CONTOUR_EDGE_NONE",
    "CONTOUR_RENDER_FILLED",
    "CONTOUR_RENDER_SHADED",
    "FEATURE_EDGE_ANGLE_DEGREES",
    "contour_surface_options",
    "extract_contour_edges",
    "style_contour_edges",
]
