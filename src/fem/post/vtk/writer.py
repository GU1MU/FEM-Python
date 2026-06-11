from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def write(
    mesh,
    cells,
    cell_types,
    elems_for_cell,
    node_disp,
    field_data,
    path: str,
    nodal_fields: Optional[Dict[str, Dict[int, float]]] = None,
):
    """Write VTK file from displacement and field dictionaries."""
    nodes = mesh.nodes
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
        for node in nodes:
            disp = node_disp.get(node.id, {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rz": 0.0})
            if is_3d:
                f.write(f"{disp.get('ux', 0.0)} {disp.get('uy', 0.0)} {disp.get('uz', 0.0)}\n")
            else:
                f.write(f"{disp.get('ux', 0.0)} {disp.get('uy', 0.0)} 0.0\n")

        if not is_3d and getattr(mesh, "dofs_per_node", 0) >= 3:
            has_any_rz = any(abs(d.get("rz", 0.0)) > 0.0 for d in node_disp.values())
            if has_any_rz:
                f.write("\nSCALARS rotz float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for node in nodes:
                    f.write(f"{node_disp.get(node.id, {}).get('rz', 0.0)}\n")

        if nodal_fields:
            for field_name, field_dict in nodal_fields.items():
                f.write(f"\nSCALARS {field_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for node in nodes:
                    f.write(f"{float(field_dict.get(node.id, 0.0))}\n")

        if cell_field_arrays:
            f.write(f"\nCELL_DATA {num_cells}\n")
            for field_name, arr in cell_field_arrays.items():
                f.write(f"\nSCALARS {field_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for val in arr:
                    f.write(f"{val}\n")


def write_unstructured(
    path: str,
    points: Sequence[Sequence[float]],
    cells: Sequence[Sequence[int]],
    cell_types: Sequence[int],
    point_vectors: Optional[Dict[str, Sequence[Sequence[float]]]] = None,
    point_data: Optional[Dict[str, Sequence[float | int]]] = None,
    cell_data: Optional[Dict[str, Sequence[float | int]]] = None,
) -> None:
    """Write a legacy VTK unstructured grid from explicit arrays."""
    point_vectors = point_vectors or {}
    point_data = point_data or {}
    cell_data = cell_data or {}
    num_points = len(points)
    num_cells = len(cells)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("FEM region-aware results from CSV\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"POINTS {num_points} float\n")
        for point in points:
            x, y, z = point
            f.write(f"{x} {y} {z}\n")

        total_ints = sum(len(conn) for conn in cells)
        f.write(f"\nCELLS {num_cells} {total_ints}\n")
        for conn in cells:
            f.write(" ".join(str(v) for v in conn) + "\n")

        f.write(f"\nCELL_TYPES {num_cells}\n")
        for ct in cell_types:
            f.write(f"{ct}\n")

        if point_vectors or point_data:
            f.write(f"\nPOINT_DATA {num_points}\n")
            for name, values in point_vectors.items():
                f.write(f"VECTORS {name} float\n")
                for vector in values:
                    vx, vy, vz = vector
                    f.write(f"{vx} {vy} {vz}\n")
            for name, values in point_data.items():
                _write_scalar_array(f, name, values)

        if cell_data:
            f.write(f"\nCELL_DATA {num_cells}\n")
            for name, values in cell_data.items():
                _write_scalar_array(f, name, values)


def _write_scalar_array(f, name: str, values: Sequence[float | int]) -> None:
    vtk_type = "int" if _all_ints(values) else "float"
    f.write(f"\nSCALARS {name} {vtk_type} 1\n")
    f.write("LOOKUP_TABLE default\n")
    for value in values:
        f.write(f"{value}\n")


def _all_ints(values: Sequence[float | int]) -> bool:
    return all(isinstance(value, (int, np.integer)) for value in values)
