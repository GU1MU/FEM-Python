from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Dict, Optional

import numpy as np


def append_legacy_ascii_unstructured_grid_geometry(
    lines: list[str],
    *,
    title: str,
    points: Sequence[Sequence[object]],
    cells: Sequence[Sequence[int]],
    cell_types: Sequence[int],
    numeric_declaration: str,
    format_float: Callable[[object], str],
) -> None:
    """Append one validated VTK Legacy ASCII 3.0 geometry prefix."""

    if type(lines) is not list or any(type(line) is not str for line in lines):
        raise TypeError("lines must be a list containing only strings")
    if type(title) is not str:
        raise TypeError("title must be a string")
    if not title or "\n" in title or "\r" in title:
        raise ValueError("title must be non-empty single-line text")
    try:
        title.encode("ascii", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("title must contain only ASCII text") from error
    if len(title) > 256:
        raise ValueError("title must not exceed 256 characters")
    if type(numeric_declaration) is not str:
        raise TypeError("numeric_declaration must be a string")
    if numeric_declaration not in {"float", "double"}:
        raise ValueError(
            "numeric_declaration must be VTK float or double"
        )
    if not callable(format_float):
        raise TypeError("format_float must be callable")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise TypeError("points must be a sequence")
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise TypeError("cells must be a sequence")
    if not isinstance(cell_types, Sequence) or isinstance(
        cell_types,
        (str, bytes),
    ):
        raise TypeError("cell_types must be a sequence")

    point_lines: list[str] = []
    for point in points:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
            raise TypeError("points must contain coordinate sequences")
        if len(point) != 3:
            raise ValueError("every point must contain three coordinates")
        formatted = tuple(format_float(component) for component in point)
        if any(
            type(value) is not str
            or not value
            or any(character.isspace() for character in value)
            for value in formatted
        ):
            raise ValueError(
                "format_float must return one non-empty numeric token"
            )
        point_lines.append(" ".join(formatted))

    cell_lines: list[str] = []
    point_count = len(points)
    for cell in cells:
        if not isinstance(cell, Sequence) or isinstance(cell, (str, bytes)):
            raise TypeError("cells must contain connectivity sequences")
        if not cell:
            raise ValueError("cells must not contain empty connectivity")
        connectivity = tuple(cell)
        if any(type(index) is not int for index in connectivity):
            raise TypeError("cell connectivity must contain integers")
        if len(set(connectivity)) != len(connectivity):
            raise ValueError("cell connectivity must not repeat point indexes")
        if any(index < 0 or index >= point_count for index in connectivity):
            raise ValueError(
                "cell connectivity references an unknown point index"
            )
        cell_lines.append(
            " ".join(
                (str(len(connectivity)), *(str(index) for index in connectivity))
            )
        )

    if len(cell_types) != len(cells):
        raise ValueError("cell_types length must match cells")
    checked_cell_types = tuple(cell_types)
    if any(type(cell_type) is not int for cell_type in checked_cell_types):
        raise TypeError("cell_types must contain integers")
    if any(cell_type <= 0 for cell_type in checked_cell_types):
        raise ValueError("cell_types must contain positive VTK type IDs")

    geometry = [
        "# vtk DataFile Version 3.0",
        title,
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {point_count} {numeric_declaration}",
        *point_lines,
        (
            f"CELLS {len(cells)} "
            f"{sum(len(cell) + 1 for cell in cells)}"
        ),
        *cell_lines,
        f"CELL_TYPES {len(checked_cell_types)}",
        *(str(cell_type) for cell_type in checked_cell_types),
    ]
    lines.extend(geometry)


def write(
    mesh,
    cells,
    cell_types,
    elems_for_cell,
    node_disp,
    field_data,
    path: str,
    nodal_fields: Optional[Dict[str, Dict[int, float]]] = None,
    points: Optional[Sequence] = None,
    point_node_ids: Optional[Sequence[int]] = None,
    nodal_point_fields: Optional[Dict[str, Sequence[float]]] = None,
):
    """Write VTK file from displacement and field dictionaries."""
    nodes = list(mesh.nodes if points is None else points)
    result_node_ids = list(
        (node.id for node in nodes) if point_node_ids is None else point_node_ids
    )
    if len(result_node_ids) != len(nodes):
        raise ValueError("point_node_ids length must match points length")
    num_points = len(nodes)
    num_cells = len(cells)

    cell_field_arrays: Dict[str, np.ndarray] = {}
    for field_name, field_dict in field_data.items():
        arr = np.zeros(num_cells, dtype=float)
        for cidx, elem in enumerate(elems_for_cell):
            eid = elem.id
            arr[cidx] = float(field_dict.get(eid, 0.0))
        cell_field_arrays[field_name] = arr

    is_3d = len(nodes) > 0 and hasattr(nodes[0], "z")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("FEM results from CSV\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"POINTS {num_points} float\n")
        for node in nodes:
            if is_3d:
                f.write(f"{node.x} {node.y} {node.z}\n")
            else:
                f.write(f"{node.x} {node.y} 0.0\n")

        total_ints = sum(len(conn) for conn in cells)
        f.write(f"\nCELLS {num_cells} {total_ints}\n")
        for conn in cells:
            f.write(" ".join(str(v) for v in conn) + "\n")

        f.write(f"\nCELL_TYPES {num_cells}\n")
        for ct in cell_types:
            f.write(f"{ct}\n")

        f.write(f"\nPOINT_DATA {num_points}\n")
        f.write("VECTORS displacement float\n")
        for node_id in result_node_ids:
            disp = node_disp.get(node_id, {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rz": 0.0})
            if is_3d:
                f.write(f"{disp.get('ux', 0.0)} {disp.get('uy', 0.0)} {disp.get('uz', 0.0)}\n")
            else:
                f.write(f"{disp.get('ux', 0.0)} {disp.get('uy', 0.0)} 0.0\n")

        if is_3d and getattr(mesh, "dofs_per_node", 0) == 6:
            f.write("\nVECTORS rotation float\n")
            for node_id in result_node_ids:
                rotation = node_disp.get(node_id, {})
                f.write(
                    f"{rotation.get('rx', 0.0)} {rotation.get('ry', 0.0)} "
                    f"{rotation.get('rz', 0.0)}\n"
                )

        if not is_3d and getattr(mesh, "dofs_per_node", 0) >= 3:
            has_any_rz = any(abs(d.get("rz", 0.0)) > 0.0 for d in node_disp.values())
            if has_any_rz:
                f.write("\nSCALARS rotz float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for node_id in result_node_ids:
                    f.write(f"{node_disp.get(node_id, {}).get('rz', 0.0)}\n")

        if nodal_fields:
            for field_name, field_dict in nodal_fields.items():
                f.write(f"\nSCALARS {field_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for node_id in result_node_ids:
                    f.write(f"{float(field_dict.get(node_id, 0.0))}\n")

        if nodal_point_fields:
            for field_name, values in nodal_point_fields.items():
                if len(values) != num_points:
                    raise ValueError(
                        f"Nodal point field {field_name!r} length {len(values)} "
                        f"does not match point count {num_points}"
                    )
                f.write(f"\nSCALARS {field_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for value in values:
                    f.write(f"{float(value)}\n")

        if cell_field_arrays:
            f.write(f"\nCELL_DATA {num_cells}\n")
            for field_name, arr in cell_field_arrays.items():
                f.write(f"\nSCALARS {field_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for val in arr:
                    f.write(f"{val}\n")
