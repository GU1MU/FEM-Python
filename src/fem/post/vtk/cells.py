from __future__ import annotations

from typing import Any, Dict


def build(mesh):
    """Build VTK cells, cell types, and matching mesh elements."""
    node_id_to_pt_idx: Dict[int, int] = {node.id: i for i, node in enumerate(mesh.nodes)}
    cells = []
    cell_types = []
    elems_for_cell = []

    for elem in mesh.elements:
        etype = str(elem.type).lower()
        vtk_conn = None
        vtk_type = None

        if "truss" in etype or "beam" in etype:
            if len(elem.node_ids) != 2:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [2] + pt_ids
            vtk_type = 3

        elif "tri3" in etype:
            if len(elem.node_ids) != 3:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [3] + pt_ids
            vtk_type = 5

        elif "quad4" in etype:
            if len(elem.node_ids) != 4:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [4] + pt_ids
            vtk_type = 9

        elif "quad8" in etype:
            if len(elem.node_ids) != 8:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [8] + pt_ids
            vtk_type = 23

        elif "tet4" in etype:
            if len(elem.node_ids) != 4:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [4] + pt_ids
            vtk_type = 10

        elif "tet10" in etype:
            if len(elem.node_ids) != 10:
                continue
            # Abaqus C3D10 and VTK quadratic tetrahedra use the same edge order.
            vtk_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            pt_ids = [node_id_to_pt_idx[elem.node_ids[i]] for i in vtk_order]
            vtk_conn = [10] + pt_ids
            vtk_type = 24

        elif "hex8" in etype:
            if len(elem.node_ids) != 8:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [8] + pt_ids
            vtk_type = 12

        else:
            raise ValueError(f"Unsupported element type for VTK export: {elem.type}")

        cells.append(vtk_conn)
        cell_types.append(vtk_type)
        elems_for_cell.append(elem)

    return cells, cell_types, elems_for_cell


def build_region_aware(mesh, region_rows: list[dict[str, float | int]]):
    """Build VTK cells using duplicated region/cluster-aware points."""
    node_lookup = {node.id: node for node in mesh.nodes}
    row_lookup = {
        (int(row["source_elem_id"]), int(row["source_local_node"])): row
        for row in region_rows
    }
    point_key_to_idx: dict[tuple[int, int, int], int] = {}
    points: list[tuple[float, float, float]] = []
    point_rows: list[dict[str, float | int]] = []
    cells = []
    cell_types = []

    for elem in mesh.elements:
        vtk_type, vtk_order = _vtk_type_and_order(elem)
        if vtk_type is None:
            continue

        pt_ids: list[int] = []
        first_row: dict[str, float | int] | None = None
        for local_zero_based in vtk_order:
            local_node = local_zero_based + 1
            key = (int(elem.id), local_node)
            if key not in row_lookup:
                raise ValueError(
                    "Region nodal stress CSV is missing row for "
                    f"elem {elem.id} local node {local_node}"
                )
            row = row_lookup[key]
            if first_row is None:
                first_row = row
            original_node_id = int(row["original_node_id"])
            point_key = (
                original_node_id,
                int(row["region_id"]),
                int(row["cluster_id"]),
            )
            if point_key not in point_key_to_idx:
                node = node_lookup[original_node_id]
                point_key_to_idx[point_key] = len(points)
                points.append((float(node.x), float(node.y), float(getattr(node, "z", 0.0))))
                point_rows.append(row)
            pt_ids.append(point_key_to_idx[point_key])

        cells.append([len(pt_ids)] + pt_ids)
        cell_types.append(vtk_type)

    return points, cells, cell_types, point_rows


def _vtk_type_and_order(elem: Any) -> tuple[int | None, list[int]]:
    """Return VTK cell type and source-node order for one element."""
    etype = str(elem.type).lower()
    if "truss" in etype or "beam" in etype:
        if len(elem.node_ids) != 2:
            return None, []
        return 3, [0, 1]
    if "tri3" in etype:
        if len(elem.node_ids) != 3:
            return None, []
        return 5, [0, 1, 2]
    if "quad4" in etype:
        if len(elem.node_ids) != 4:
            return None, []
        return 9, [0, 1, 2, 3]
    if "quad8" in etype:
        if len(elem.node_ids) != 8:
            return None, []
        return 23, list(range(8))
    if "tet4" in etype:
        if len(elem.node_ids) != 4:
            return None, []
        return 10, [0, 1, 2, 3]
    if "tet10" in etype:
        if len(elem.node_ids) != 10:
            return None, []
        return 24, list(range(10))
    if "hex8" in etype:
        if len(elem.node_ids) != 8:
            return None, []
        return 12, list(range(8))
    raise ValueError(f"Unsupported element type for VTK export: {elem.type}")
