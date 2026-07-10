from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict


@dataclass(frozen=True)
class ResultTopology:
    """VTK topology expanded for element-local nodal result rows."""

    points: tuple[Any, ...]
    point_node_ids: tuple[int, ...]
    point_rows: tuple[Any | None, ...]
    cells: tuple[list[int], ...]
    cell_types: tuple[int, ...]
    elems_for_cell: tuple[Any, ...]


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

        if "c3d20r" in etype:
            raise ValueError(f"Unsupported element type for VTK export: {elem.type}")

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

        elif "tri6" in etype:
            if len(elem.node_ids) != 6:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [6] + pt_ids
            vtk_type = 22

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

        elif "hex20" in etype or "c3d20" in etype:
            if len(elem.node_ids) != 20:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [20] + pt_ids
            vtk_type = 25

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


def build_result(mesh, nodal_rows=()) -> ResultTopology:
    """Build point/cell topology while preserving repeated raw nodal rows."""
    base_cells, cell_types, elems_for_cell = build(mesh)
    rows_by_node: dict[int, list[Any]] = {}
    for row in nodal_rows:
        rows_by_node.setdefault(int(row.node_id), []).append(row)

    node_lookup = {int(node.id): node for node in mesh.nodes}
    elem_lookup = {int(elem.id): elem for elem in mesh.elements}
    shared_point_index: dict[int, int] = {}
    local_point_index: dict[tuple[int, int], int] = {}
    points: list[Any] = []
    point_node_ids: list[int] = []
    point_rows: list[Any | None] = []

    for node in mesh.nodes:
        node_id = int(node.id)
        node_rows = rows_by_node.get(node_id, [])
        if len(node_rows) <= 1:
            shared_point_index[node_id] = len(points)
            points.append(node)
            point_node_ids.append(node_id)
            point_rows.append(node_rows[0] if node_rows else None)
            continue

        for row in node_rows:
            if row.averaged or row.elem_id is None or row.local_node is None:
                raise ValueError(
                    f"Repeated nodal stress row for node {node_id} requires "
                    "averaged=false with elem_id and local_node provenance"
                )
            elem = elem_lookup.get(int(row.elem_id))
            local_node = int(row.local_node)
            if (
                elem is None
                or local_node < 1
                or local_node > len(elem.node_ids)
                or int(elem.node_ids[local_node - 1]) != node_id
            ):
                raise ValueError(
                    f"Nodal stress row for node {node_id} with element {row.elem_id} "
                    f"local node {row.local_node} cannot be matched to mesh connectivity"
                )
            key = (int(row.elem_id), local_node)
            if key in local_point_index:
                raise ValueError(
                    f"Duplicate nodal stress provenance for element {row.elem_id} "
                    f"local node {row.local_node}"
                )
            local_point_index[key] = len(points)
            points.append(node_lookup[node_id])
            point_node_ids.append(node_id)
            point_rows.append(row)

    expanded_cells: list[list[int]] = []
    for base_cell, elem in zip(base_cells, elems_for_cell):
        point_ids: list[int] = []
        for local_node, node_id in enumerate(elem.node_ids, start=1):
            node_id = int(node_id)
            if len(rows_by_node.get(node_id, [])) > 1:
                key = (int(elem.id), local_node)
                if key not in local_point_index:
                    raise ValueError(
                        f"Repeated nodal stress rows for node {node_id} cannot be "
                        f"matched to element {elem.id} local node {local_node}"
                    )
                point_ids.append(local_point_index[key])
            else:
                point_ids.append(shared_point_index[node_id])
        expanded_cells.append([base_cell[0], *point_ids])

    unknown_nodes = set(rows_by_node).difference(node_lookup)
    if unknown_nodes:
        raise ValueError(
            f"Nodal stress CSV contains node ids missing from mesh: {sorted(unknown_nodes)}"
        )

    return ResultTopology(
        points=tuple(points),
        point_node_ids=tuple(point_node_ids),
        point_rows=tuple(point_rows),
        cells=tuple(expanded_cells),
        cell_types=tuple(cell_types),
        elems_for_cell=tuple(elems_for_cell),
    )
