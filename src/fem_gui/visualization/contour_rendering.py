"""Contour surface and visible-edge rendering helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

CONTOUR_RENDER_SHADED = "shaded"
CONTOUR_RENDER_FILLED = "filled"
CONTOUR_RENDER_WIREFRAME = "wireframe"
CONTOUR_RENDER_HIDDEN_LINE = "hidden_line"

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


def extract_dataset_surface(dataset: Any, **options: Any) -> Any:
    """Extract a surface across supported PyVista versions."""

    try:
        return dataset.extract_surface(
            algorithm="dataset_surface",
            **options,
        )
    except TypeError as error:
        if "unexpected keyword argument 'algorithm'" not in str(error):
            raise
        return dataset.extract_surface(**options)


def contour_surface_options(
    render_mode: str,
    *,
    is_line_mesh: bool,
) -> dict[str, Any]:
    """Return PyVista rendering options for one contour render mode."""

    if render_mode == CONTOUR_RENDER_FILLED:
        return {
            "lighting": False,
            "smooth_shading": False,
        }
    if render_mode == CONTOUR_RENDER_WIREFRAME:
        return {
            "style": "wireframe",
            "lighting": False,
            "smooth_shading": False,
        }
    if render_mode == CONTOUR_RENDER_HIDDEN_LINE:
        # VTK's surface representation plus a separately extracted geometry
        # edge layer gives a stable hidden-line view across backends.  The
        # caller owns the edge extraction so result scalars remain untouched.
        return {
            "style": "surface",
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
    point_keys: tuple[tuple[object, ...], ...],
    *,
    scalar_name: str,
    point_scalars: bool,
) -> Any:
    """Build an exterior scalar surface with geometry-owned point normals."""

    canonical_points: list[np.ndarray] = []
    canonical_by_key: dict[tuple[object, ...], int] = {}
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
    connected.point_data[_NORMAL_SOURCE_POINT_ID] = np.arange(
        connected.n_points,
        dtype=np.int64,
    )
    connected.cell_data[_NORMAL_SOURCE_CELL_ID] = np.arange(
        connected.n_cells,
        dtype=np.int64,
    )
    surface = extract_dataset_surface(
        connected,
        pass_pointid=False,
        pass_cellid=False,
        nonlinear_subdivision=0,
    )
    if int(surface.n_faces_strict) == 0:
        return dataset

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
    # Materialize the VTK point array once.  Accessing ``normal_surface.points``
    # inside the per-face/per-point mapping loop creates a new PyVista wrapper
    # for every point and turns an otherwise linear NumPy operation into a
    # Python <-> VTK round trip hotspot on large meshes.
    normal_surface_points = np.asarray(normal_surface.points)
    normals = np.asarray(normal_surface.point_data.active_normals)
    # ``faces`` is a VTK packed array: ``(n, p0, ..., p[n-1], ...)``.  Keep
    # the packed layout, but decode the point ids once so the provenance
    # remap below can be performed as array indexing instead of repeatedly
    # calling ``tuple.index`` for every normal point.
    face_counts: list[int] = []
    face_offsets: list[int] = []
    cursor = 0
    while cursor < normal_faces.size:
        point_count = int(normal_faces[cursor])
        if point_count <= 0 or cursor + point_count >= normal_faces.size:
            raise ValueError("normal surface contains an invalid packed face")
        face_offsets.append(cursor)
        face_counts.append(point_count)
        cursor += point_count + 1
    if cursor != normal_faces.size:
        raise ValueError("normal surface packed faces are truncated")
    if len(face_counts) != len(normal_source_cells):
        raise ValueError("normal surface provenance does not match its faces")

    face_offsets_array = np.asarray(face_offsets, dtype=np.int64)
    face_counts_array = np.asarray(face_counts, dtype=np.int64)
    point_mask = np.ones(normal_faces.size, dtype=bool)
    point_mask[face_offsets_array] = False
    normal_point_ids = normal_faces[point_mask]
    if normal_point_ids.size != int(np.sum(face_counts_array)):
        raise ValueError("normal surface packed face point count is invalid")

    # The output point order is exactly the order of the source normal points;
    # only the packed face point ids need to be renumbered.  This preserves the
    # historical face and normal ordering while avoiding per-point Python lists.
    packed_render_faces = normal_faces.copy()
    packed_render_faces[point_mask] = np.arange(
        normal_point_ids.size,
        dtype=np.int64,
    )

    canonical_cell_count = len(canonical_cells)
    max_cell_points = max(len(cell) for cell in canonical_cells)
    canonical_cell_points = np.full(
        (canonical_cell_count, max_cell_points),
        -1,
        dtype=np.int64,
    )
    result_cell_points = np.full_like(canonical_cell_points, -1)
    for cell_index, (canonical_cell, result_cell) in enumerate(
        zip(canonical_cells, cells, strict=True)
    ):
        if len(canonical_cell) != len(result_cell):
            raise ValueError("canonical and result cell arities do not match")
        width = len(canonical_cell)
        canonical_cell_points[cell_index, :width] = canonical_cell
        result_cell_points[cell_index, :width] = result_cell

    source_cell_ids = np.asarray(normal_source_cells, dtype=np.int64)
    point_source_cell_ids = np.repeat(source_cell_ids, face_counts_array)
    canonical_point_ids = np.asarray(
        normal_source_points[normal_point_ids],
        dtype=np.int64,
    )
    matching_local_nodes = (
        canonical_cell_points[point_source_cell_ids]
        == canonical_point_ids[:, None]
    )
    if not np.all(np.any(matching_local_nodes, axis=1)):
        raise ValueError("normal surface provenance cannot map to result cells")
    local_nodes = np.argmax(matching_local_nodes, axis=1)
    render_source_points = result_cell_points[
        point_source_cell_ids,
        local_nodes,
    ]
    render_points = normal_surface_points[normal_point_ids]
    render_normals = normals[normal_point_ids]

    rendered = type(surface)(
        np.asarray(render_points, dtype=float),
        faces=packed_render_faces,
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
        source_cell_ids,
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


def update_shaded_contour_geometry(rendered: Any, dataset: Any) -> bool:
    """Move an existing split shaded surface onto updated source points."""

    if _SCALAR_SOURCE_POINT_ID not in rendered.point_data:
        return False
    source_ids = np.asarray(
        rendered.point_data[_SCALAR_SOURCE_POINT_ID],
        dtype=np.int64,
    )
    source_points = np.asarray(dataset.points)
    if source_ids.size and int(np.max(source_ids)) >= len(source_points):
        return False
    rendered.points = source_points[source_ids]
    rendered.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=False,
        inplace=True,
    )
    return True


def extract_contour_edges(dataset: Any, edge_mode: str) -> Any | None:
    """Extract Abaqus-style visible edges from a result dataset."""

    if edge_mode == CONTOUR_EDGE_NONE:
        return None
    if edge_mode == CONTOUR_EDGE_ALL:
        extractor = getattr(dataset, "extract_all_edges", None)
        return (
            dataset
            if not callable(extractor)
            else extractor(clear_data=True)
        )

    # A few lightweight viewport/test adapters expose only ``points``.  They
    # cannot compute feature edges, but keeping the source dataset lets the
    # caller still create its edge actor and preserves the old adapter
    # contract.  Real PyVista datasets always take the branch below.
    if not callable(getattr(dataset, "cast_to_unstructured_grid", None)):
        return dataset

    connected_geometry = (
        dataset.cast_to_unstructured_grid().clean()
    )
    surface = extract_dataset_surface(connected_geometry).clean()
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
    "CONTOUR_RENDER_HIDDEN_LINE",
    "CONTOUR_RENDER_SHADED",
    "CONTOUR_RENDER_WIREFRAME",
    "FEATURE_EDGE_ANGLE_DEGREES",
    "bind_shaded_contour_scalars",
    "build_shaded_contour_surface",
    "contour_surface_options",
    "extract_contour_edges",
    "style_contour_edges",
]
