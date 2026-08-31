"""Validation that depends only on model and mesh data."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from numbers import Real
from typing import Any


_MESH_ATTRIBUTES = (
    "nodes",
    "elements",
    "dofs_per_node",
    "dof_map",
    "num_dofs",
    "element_dofs",
)


def validate_mesh(mesh: Any) -> None:
    """Validate mesh identity, connectivity, coordinates, and DOF state."""

    missing = [name for name in _MESH_ATTRIBUTES if not hasattr(mesh, name)]
    if missing:
        raise TypeError(
            "mesh validation requires attributes "
            + ", ".join(_MESH_ATTRIBUTES)
            + f"; missing {', '.join(missing)}"
        )
    nodes = list(mesh.nodes)
    elements = list(mesh.elements)
    if not nodes:
        raise ValueError("mesh must contain at least one node")
    if not elements:
        raise ValueError("mesh must contain at least one element")
    node_ids = _unique_entity_ids(nodes, "node")
    _unique_entity_ids(elements, "element")
    _validate_coordinates(nodes)
    _validate_connectivity(elements, set(node_ids))
    _validate_dof_map(mesh, node_ids)


def _unique_entity_ids(entities: Sequence[Any], kind: str) -> list[int]:
    ids = [_entity_id(entity, kind) for entity in entities]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{kind} ids must be unique")
    return ids


def _entity_id(entity: Any, kind: str) -> int:
    if not hasattr(entity, "id"):
        raise TypeError(f"{kind} is missing id")
    return _integer_id(entity.id, f"{kind} id")


def _integer_id(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, got {value!r}")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{label} must be an integer, got {value!r}") from exc


def _validate_coordinates(nodes: Sequence[Any]) -> None:
    z_flags = [hasattr(node, "z") for node in nodes]
    if z_flags and any(z_flags) and not all(z_flags):
        raise ValueError("mesh nodes must use one consistent coordinate dimension")
    coordinate_names = ("x", "y", "z") if z_flags and all(z_flags) else ("x", "y")
    for node in nodes:
        node_id = _entity_id(node, "node")
        for name in coordinate_names:
            if not hasattr(node, name):
                raise TypeError(f"node {node_id} is missing coordinate {name}")
            raw_value = getattr(node, name)
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise TypeError(
                    f"node {node_id} coordinate {name} must be a real number, "
                    f"got {raw_value!r}"
                )
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(
                    f"node {node_id} coordinate {name} must be finite, got {value}"
                )


def _validate_connectivity(elements: Sequence[Any], node_ids: set[int]) -> None:
    for element in elements:
        element_id = _entity_id(element, "element")
        if not hasattr(element, "node_ids"):
            raise TypeError(f"element {element_id} is missing node_ids")
        connectivity = [
            _integer_id(raw_node_id, f"element {element_id} node id")
            for raw_node_id in element.node_ids
        ]
        if len(set(connectivity)) != len(connectivity):
            raise ValueError(f"element {element_id} node_ids must be unique")
        for node_id in connectivity:
            if node_id not in node_ids:
                raise KeyError(
                    f"element {element_id} references missing node {node_id}"
                )


def _validate_dof_map(mesh: Any, node_ids: Sequence[int]) -> None:
    dofs_per_node = _integer_id(mesh.dofs_per_node, "dofs_per_node")
    if dofs_per_node <= 0:
        raise ValueError("dofs_per_node must be positive")
    dof_map = mesh.dof_map
    expected_ids = sorted(node_ids)
    expected_lookup = {
        node_id: node_index for node_index, node_id in enumerate(expected_ids)
    }
    consistent = (
        int(getattr(dof_map, "dofs_per_node", -1)) == dofs_per_node
        and list(getattr(dof_map, "node_ids", ())) == expected_ids
        and dict(getattr(dof_map, "node_id_to_index", {})) == expected_lookup
        and int(mesh.num_dofs) == len(expected_ids) * dofs_per_node
    )
    if not consistent:
        raise ValueError(
            "mesh NodeDofMap is inconsistent with current nodes; "
            "call mesh.rebuild_dof_map()"
        )


__all__ = ["validate_mesh"]
