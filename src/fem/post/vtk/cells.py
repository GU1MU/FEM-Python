from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict

from ...elements import canonical_element_type


_VTK_CELL_TYPES = {
    "Truss2": (2, 3),
    "Beam2": (2, 3),
    "Tri3": (3, 5),
    "Tri6": (6, 22),
    "Quad4": (4, 9),
    "Quad8": (8, 23),
    "Tet4": (4, 10),
    "Tet10": (10, 24),
    "Hex8": (8, 12),
    "Hex20": (20, 25),
}


class UnsupportedVTKCellTypeError(ValueError):
    """Raised when a canonical FEM element has no VTK Legacy cell type."""


def vtk_cell_spec(canonical_type: str) -> tuple[int, int]:
    """Return required node count and VTK type for a canonical FEM type."""

    if type(canonical_type) is not str:
        raise TypeError("canonical_type must be a string")
    try:
        return _VTK_CELL_TYPES[canonical_type]
    except KeyError as error:
        raise UnsupportedVTKCellTypeError(
            "Unsupported canonical element type for VTK export: "
            f"{canonical_type}"
        ) from error


def vtk_cell_spec_for_element(element_type: str) -> tuple[int, int]:
    """Return required node count and VTK type for a raw element type or alias."""

    return vtk_cell_spec(canonical_element_type(element_type))


def vtk_cell_type(canonical_type: str) -> int:
    """Return the exact VTK Legacy cell type for a canonical FEM type."""

    return vtk_cell_spec(canonical_type)[1]


def vtk_cell_node_count(canonical_type: str) -> int:
    """Return the required connectivity length for one canonical VTK cell."""

    return vtk_cell_spec(canonical_type)[0]


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
        try:
            canonical_type = canonical_element_type(elem.type)
            vtk_type = vtk_cell_type(canonical_type)
        except (NotImplementedError, UnsupportedVTKCellTypeError) as error:
            raise UnsupportedVTKCellTypeError(
                f"Unsupported element type for VTK export: {elem.type}"
            ) from error
        node_count = _VTK_CELL_TYPES[canonical_type][0]
        if len(elem.node_ids) != node_count:
            continue
        pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
        vtk_conn = [node_count] + pt_ids

        cells.append(vtk_conn)
        cell_types.append(vtk_type)
        elems_for_cell.append(elem)

    return cells, cell_types, elems_for_cell


def _validate_nonaveraged_row(
    row: Any,
    node_id: int,
    elem_lookup: dict[int, Any],
) -> tuple[int, int] | None:
    """Validate element-local provenance and return its connectivity key."""
    if row.averaged:
        return None
    missing = []
    if row.elem_id is None:
        missing.append("elem_id")
    if row.local_node is None:
        missing.append("local_node")
    if missing:
        raise ValueError(
            f"Nodal stress row for node {node_id} with averaged=false requires "
            "elem_id and one-based local_node provenance; "
            f"missing {', '.join(missing)}"
        )

    elem_id = int(row.elem_id)
    local_node = int(row.local_node)
    elem = elem_lookup.get(elem_id)
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
    return elem_id, local_node


def build_result(mesh, nodal_rows=()) -> ResultTopology:
    """Build point/cell topology while preserving repeated raw nodal rows."""
    base_cells, cell_types, elems_for_cell = build(mesh)
    rows_by_node: dict[int, list[Any]] = {}
    for row in nodal_rows:
        rows_by_node.setdefault(int(row.node_id), []).append(row)

    node_lookup = {int(node.id): node for node in mesh.nodes}
    elem_lookup = {int(elem.id): elem for elem in mesh.elements}
    incident_keys_by_node: dict[int, set[tuple[int, int]]] = {}
    for elem in elems_for_cell:
        for local_node, node_id in enumerate(elem.node_ids, start=1):
            incident_keys_by_node.setdefault(int(node_id), set()).add(
                (int(elem.id), local_node)
            )
    shared_point_index: dict[int, int] = {}
    local_point_index: dict[tuple[int, int], int] = {}
    points: list[Any] = []
    point_node_ids: list[int] = []
    point_rows: list[Any | None] = []

    for node in mesh.nodes:
        node_id = int(node.id)
        node_rows = rows_by_node.get(node_id, [])
        if not node_rows or (len(node_rows) == 1 and node_rows[0].averaged):
            shared_point_index[node_id] = len(points)
            points.append(node)
            point_node_ids.append(node_id)
            point_rows.append(node_rows[0] if node_rows else None)
            continue

        raw_keys: set[tuple[int, int]] = set()
        for row in node_rows:
            if row.averaged:
                raise ValueError(
                    f"Repeated nodal stress row for node {node_id} requires "
                    "averaged=false with elem_id and local_node provenance"
                )
            key = _validate_nonaveraged_row(row, node_id, elem_lookup)
            assert key is not None
            if key in local_point_index:
                raise ValueError(
                    f"Duplicate nodal stress provenance for element {row.elem_id} "
                    f"local node {row.local_node}"
                )
            raw_keys.add(key)
            local_point_index[key] = len(points)
            points.append(node_lookup[node_id])
            point_node_ids.append(node_id)
            point_rows.append(row)

        missing_keys = incident_keys_by_node.get(node_id, set()).difference(raw_keys)
        if missing_keys:
            shared_point_index[node_id] = len(points)
            points.append(node_lookup[node_id])
            point_node_ids.append(node_id)
            point_rows.append(None)

    expanded_cells: list[list[int]] = []
    for base_cell, elem in zip(base_cells, elems_for_cell):
        point_ids: list[int] = []
        for local_node, node_id in enumerate(elem.node_ids, start=1):
            node_id = int(node_id)
            key = (int(elem.id), local_node)
            if key in local_point_index:
                point_ids.append(local_point_index[key])
            else:
                if node_id not in shared_point_index:
                    raise ValueError(
                        f"Nodal stress topology for node {node_id} has no point for "
                        f"element {elem.id} local node {local_node}"
                    )
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


__all__ = [
    "ResultTopology",
    "UnsupportedVTKCellTypeError",
    "build",
    "build_result",
    "vtk_cell_node_count",
    "vtk_cell_spec",
    "vtk_cell_spec_for_element",
    "vtk_cell_type",
]
