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
_NORMAL_SOURCE_POINT_ID = "normal_source_point_id"
_NORMAL_SOURCE_CELL_ID = "normal_source_cell_id"
_SCALAR_SOURCE_POINT_ID = "scalar_source_point_id"
_SCALAR_SOURCE_CELL_ID = "scalar_source_cell_id"


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
        "ambient": 0.7,
        "diffuse": 0.3,
        "specular": 0.0,
    }


def build_shaded_contour_surface(
    dataset: Any,
    cells: tuple[tuple[int, ...], ...],
    point_keys: tuple[tuple[int, int], ...],
    *,
    scalar_name: str,
    point_scalars: bool,
) -> Any:
    """Build an exterior scalar surface with geometry-owned point normals."""

    canonical_points: list[np.ndarray] = []
    canonical_by_key: dict[tuple[int, int], int] = {}
    point_to_canonical = np.empty(len(point_keys), dtype=np.int64)
    source_points = np.asarray(dataset.points)
    for point_index, key in enumerate(point_keys):
        canonical_index = canonical_by_key.get(key)
        if canonical_index is None:
            canonical_index = len(canonical_points)
            canonical_by_key[key] = canonical_index
            canonical_points.append(source_points[point_index])
        point_to_canonical[point_index] = canonical_index

    canonical_cells = tuple(
        tuple(int(point_to_canonical[index]) for index in cell)
        for cell in cells
    )
    flat_cells = np.fromiter(
        (
            value
            for cell in canonical_cells
            for value in (len(cell), *cell)
        ),
        dtype=np.int64,
        count=sum(len(cell) + 1 for cell in canonical_cells),
    )
    connected = type(dataset)(
        flat_cells,
        np.asarray(dataset.celltypes, dtype=np.uint8),
        np.asarray(canonical_points, dtype=float),
    )
    surface = connected.extract_surface(
        algorithm="dataset_surface",
        pass_pointid=True,
        pass_cellid=True,
    )
    if int(surface.n_faces) == 0:
        return dataset

    surface.point_data[_NORMAL_SOURCE_POINT_ID] = np.asarray(
        surface.point_data["vtkOriginalPointIds"],
        dtype=np.int64,
    )
    surface.cell_data[_NORMAL_SOURCE_CELL_ID] = np.asarray(
        surface.cell_data["vtkOriginalCellIds"],
        dtype=np.int64,
    )
    normal_surface = surface.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=True,
        feature_angle=FEATURE_EDGE_ANGLE_DEGREES,
    )

    normal_faces = np.asarray(normal_surface.faces, dtype=np.int64)
    normal_source_points = np.asarray(
        normal_surface.point_data[_NORMAL_SOURCE_POINT_ID],
        dtype=np.int64,
    )
    normal_source_cells = np.asarray(
        normal_surface.cell_data[_NORMAL_SOURCE_CELL_ID],
        dtype=np.int64,
    )
    normals = np.asarray(normal_surface.point_data.active_normals)
    render_points: list[np.ndarray] = []
    render_normals: list[np.ndarray] = []
    render_source_points: list[int] = []
    render_source_cells: list[int] = []
    render_faces: list[int] = []
    cursor = 0
    cell_index = 0
    while cursor < normal_faces.size:
        point_count = int(normal_faces[cursor])
        normal_point_ids = normal_faces[
            cursor + 1 : cursor + 1 + point_count
        ]
        source_cell_id = int(normal_source_cells[cell_index])
        canonical_cell = canonical_cells[source_cell_id]
        result_cell = cells[source_cell_id]
        first_render_point = len(render_points)
        render_faces.extend(
            (point_count, *range(first_render_point, first_render_point + point_count))
        )
        for normal_point_id in normal_point_ids:
            normal_index = int(normal_point_id)
            canonical_point_id = int(normal_source_points[normal_index])
            local_node = canonical_cell.index(canonical_point_id)
            result_point_id = result_cell[local_node]
            render_points.append(np.asarray(normal_surface.points[normal_index]))
            render_normals.append(normals[normal_index])
            render_source_points.append(result_point_id)
        render_source_cells.append(source_cell_id)
        cursor += point_count + 1
        cell_index += 1

    rendered = type(surface)(
        np.asarray(render_points, dtype=float),
        faces=np.asarray(render_faces, dtype=np.int64),
    )
    rendered.point_data["Normals"] = np.asarray(render_normals, dtype=float)
    rendered.GetPointData().SetNormals(
        rendered.GetPointData().GetArray("Normals")
    )
    rendered.point_data[_SCALAR_SOURCE_POINT_ID] = np.asarray(
        render_source_points,
        dtype=np.int64,
    )
    rendered.cell_data[_SCALAR_SOURCE_CELL_ID] = np.asarray(
        render_source_cells,
        dtype=np.int64,
    )
    bind_shaded_contour_scalars(
        rendered,
        dataset,
        scalar_name=scalar_name,
        point_scalars=point_scalars,
    )
    return rendered


def bind_shaded_contour_scalars(
    rendered: Any,
    dataset: Any,
    *,
    scalar_name: str,
    point_scalars: bool,
) -> bool:
    """Bind one source scalar array onto an existing shaded surface."""

    if point_scalars:
        if _SCALAR_SOURCE_POINT_ID not in rendered.point_data:
            return False
        source_ids = np.asarray(
            rendered.point_data[_SCALAR_SOURCE_POINT_ID],
            dtype=np.int64,
        )
        rendered.point_data[scalar_name] = np.asarray(
            dataset.point_data[scalar_name]
        )[source_ids]
        preference = "point"
    else:
        if _SCALAR_SOURCE_CELL_ID not in rendered.cell_data:
            return False
        source_ids = np.asarray(
            rendered.cell_data[_SCALAR_SOURCE_CELL_ID],
            dtype=np.int64,
        )
        rendered.cell_data[scalar_name] = np.asarray(
            dataset.cell_data[scalar_name]
        )[source_ids]
        preference = "cell"
    rendered.set_active_scalars(scalar_name, preference=preference)
    return True


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
    "bind_shaded_contour_scalars",
    "build_shaded_contour_surface",
    "contour_surface_options",
    "extract_contour_edges",
    "style_contour_edges",
]
