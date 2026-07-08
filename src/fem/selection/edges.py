from __future__ import annotations

import builtins
from typing import Any

from ..core.model import Edge, ElementEdge
from .nodes import _coord_matches


def boundary(mesh: Any) -> list[tuple[int, int, list[int]]]:
    """Return boundary edges as (elem_id, local_edge, node_ids)."""
    edge_count: dict[tuple[int, ...], int] = {}
    edge_store: dict[tuple[int, ...], list[tuple[int, int, list[int]]]] = {}

    for elem in mesh.elements:
        for local_edge, node_ids in enumerate(_element_edge_node_ids(elem)):
            key = _edge_key(node_ids)
            edge_count[key] = edge_count.get(key, 0) + 1
            edge_store.setdefault(key, []).append((elem.id, local_edge, node_ids))

    result = []
    for key, count in edge_count.items():
        if count == 1:
            result.extend(edge_store[key])
    return result


def all(mesh: Any) -> list[tuple[int, int, list[int]]]:
    """Return all edges as (elem_id, local_edge, node_ids)."""
    result = []
    for elem in mesh.elements:
        for local_edge, node_ids in enumerate(_element_edge_node_ids(elem)):
            result.append((elem.id, local_edge, node_ids))
    return result


def by_x(
    mesh: Any,
    x_value: float,
    tol: float = 1e-8,
    boundary_only: bool = True,
) -> list[tuple[int, int, list[int]]]:
    """Return edges whose all nodes match x within tol."""
    return by_coord(mesh, x=x_value, tol=tol, boundary_only=boundary_only)


def by_y(
    mesh: Any,
    y_value: float,
    tol: float = 1e-8,
    boundary_only: bool = True,
) -> list[tuple[int, int, list[int]]]:
    """Return edges whose all nodes match y within tol."""
    return by_coord(mesh, y=y_value, tol=tol, boundary_only=boundary_only)


def by_z(
    mesh: Any,
    z_value: float,
    tol: float = 1e-8,
    boundary_only: bool = True,
) -> list[tuple[int, int, list[int]]]:
    """Return edges whose all nodes match z within tol."""
    return by_coord(mesh, z=z_value, tol=tol, boundary_only=boundary_only)


def by_coord(
    mesh: Any,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    tol: float = 1e-8,
    boundary_only: bool = True,
) -> list[tuple[int, int, list[int]]]:
    """Return edges whose all nodes match given coordinates."""
    if x is None and y is None and z is None:
        raise ValueError("at least one coordinate must be provided")
    candidates = boundary(mesh) if boundary_only else all(mesh)
    node_lookup = {node.id: node for node in mesh.nodes}
    return [
        (elem_id, local_edge, node_ids)
        for elem_id, local_edge, node_ids in candidates
        if builtins.all(
            _coord_matches(node_lookup[node_id], x, y, z, tol)
            for node_id in node_ids
        )
    ]


def edge_by_x(
    mesh: Any,
    name: str,
    x_value: float,
    tol: float = 1e-8,
    boundary_only: bool = True,
) -> Edge:
    """Return a named edge collection selected by x."""
    return _edge_from_edges(name, by_x(mesh, x_value, tol, boundary_only))


def edge_by_y(
    mesh: Any,
    name: str,
    y_value: float,
    tol: float = 1e-8,
    boundary_only: bool = True,
) -> Edge:
    """Return a named edge collection selected by y."""
    return _edge_from_edges(name, by_y(mesh, y_value, tol, boundary_only))


def edge_by_z(
    mesh: Any,
    name: str,
    z_value: float,
    tol: float = 1e-8,
    boundary_only: bool = True,
) -> Edge:
    """Return a named edge collection selected by z."""
    return _edge_from_edges(name, by_z(mesh, z_value, tol, boundary_only))


def edge_by_coord(
    mesh: Any,
    name: str,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    tol: float = 1e-8,
    boundary_only: bool = True,
) -> Edge:
    """Return a named edge collection selected by coordinates."""
    return _edge_from_edges(
        name,
        by_coord(mesh, x=x, y=y, z=z, tol=tol, boundary_only=boundary_only),
    )


def _edge_from_edges(
    name: str,
    edge_entries: list[tuple[int, int, list[int]]],
) -> Edge:
    """Convert edge selection tuples to a named edge collection."""
    return Edge(
        name,
        [
            ElementEdge(elem_id, local_edge, node_ids)
            for elem_id, local_edge, node_ids in edge_entries
        ],
    )


def _element_edge_node_ids(elem: Any) -> list[list[int]]:
    """Return local edge node id lists for common 2D and 3D elements."""
    etype = str(elem.type).lower()
    node_ids = elem.node_ids

    if ("tri3" in etype or "cps3" in etype or "cpe3" in etype) and len(node_ids) == 3:
        return [
            [node_ids[0], node_ids[1]],
            [node_ids[1], node_ids[2]],
            [node_ids[2], node_ids[0]],
        ]

    if ("tri6" in etype or "cps6" in etype or "cpe6" in etype) and len(node_ids) == 6:
        return [
            [node_ids[0], node_ids[3], node_ids[1]],
            [node_ids[1], node_ids[4], node_ids[2]],
            [node_ids[2], node_ids[5], node_ids[0]],
        ]

    if (
        "quad4" in etype
        or "cps4" in etype
        or "cpe4" in etype
    ) and len(node_ids) == 4:
        return [
            [node_ids[0], node_ids[1]],
            [node_ids[1], node_ids[2]],
            [node_ids[2], node_ids[3]],
            [node_ids[3], node_ids[0]],
        ]

    if ("quad8" in etype or "cps8" in etype or "cpe8" in etype) and len(node_ids) == 8:
        return [
            [node_ids[0], node_ids[4], node_ids[1]],
            [node_ids[1], node_ids[5], node_ids[2]],
            [node_ids[2], node_ids[6], node_ids[3]],
            [node_ids[3], node_ids[7], node_ids[0]],
        ]

    if ("hex8" in etype or "c3d8" in etype) and len(node_ids) == 8:
        return [
            [node_ids[0], node_ids[1]],
            [node_ids[1], node_ids[2]],
            [node_ids[2], node_ids[3]],
            [node_ids[3], node_ids[0]],
            [node_ids[4], node_ids[5]],
            [node_ids[5], node_ids[6]],
            [node_ids[6], node_ids[7]],
            [node_ids[7], node_ids[4]],
            [node_ids[0], node_ids[4]],
            [node_ids[1], node_ids[5]],
            [node_ids[2], node_ids[6]],
            [node_ids[3], node_ids[7]],
        ]

    if ("tet10" in etype or "c3d10" in etype) and len(node_ids) == 10:
        return [
            [node_ids[0], node_ids[4], node_ids[1]],
            [node_ids[1], node_ids[5], node_ids[2]],
            [node_ids[2], node_ids[6], node_ids[0]],
            [node_ids[0], node_ids[7], node_ids[3]],
            [node_ids[1], node_ids[8], node_ids[3]],
            [node_ids[2], node_ids[9], node_ids[3]],
        ]

    if ("tet4" in etype or "c3d4" in etype) and len(node_ids) == 4:
        return [
            [node_ids[0], node_ids[1]],
            [node_ids[1], node_ids[2]],
            [node_ids[2], node_ids[0]],
            [node_ids[0], node_ids[3]],
            [node_ids[1], node_ids[3]],
            [node_ids[2], node_ids[3]],
        ]

    return []


def _edge_key(node_ids: list[int]) -> tuple[int, ...]:
    """Return topology key for an edge."""
    return tuple(sorted((node_ids[0], node_ids[-1])))
