from __future__ import annotations

from typing import Dict


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
