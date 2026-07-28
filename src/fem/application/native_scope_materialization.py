"""Materialize user-authored native scopes on an existing mesh artifact."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from fem.core.model import (
    Edge,
    ElementEdge,
    ElementFace,
    ElementSet,
    NodeSet,
    Surface,
)
from fem.geometry.references import LogicalEntityRef
from fem.selection import edges as mesh_edges
from fem.selection import faces as mesh_faces

from .definitions import MeshEntityRef


NATIVE_SCOPE_CATALOG_KEY = "_native_scope_catalog"


def has_native_scope_catalog(model: Any) -> bool:
    """Return whether a native model can update scopes without remeshing."""

    metadata = getattr(model, "metadata", None)
    return (
        isinstance(metadata, Mapping)
        and isinstance(metadata.get(NATIVE_SCOPE_CATALOG_KEY), Mapping)
    )


def can_materialize_native_scopes(
    model: Any,
    regions: Iterable[Any],
) -> bool:
    """Return whether every supplied scope can be rebuilt on this mesh."""

    references = tuple(
        reference
        for region in regions
        for reference in tuple(getattr(region, "references", ()))
    )
    return (
        not references
        or all(type(reference) is MeshEntityRef for reference in references)
        or (
            all(type(reference) is LogicalEntityRef for reference in references)
            and has_native_scope_catalog(model)
        )
    )


def materialize_native_scopes(
    model: Any,
    *,
    previous_names: Iterable[str],
    regions: Iterable[Any],
) -> Any:
    """Return a detached model with user scopes rebuilt from its mesh catalog."""

    metadata = getattr(model, "metadata", None)
    catalog = (
        metadata.get(NATIVE_SCOPE_CATALOG_KEY)
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(catalog, Mapping):
        catalog = {}

    updated = deepcopy(model)
    collections = (
        updated.node_sets,
        updated.element_sets,
        updated.edges,
        updated.surfaces,
    )
    for name in previous_names:
        normalized_name = str(name)
        for collection in collections:
            collection.pop(normalized_name, None)

    for region in regions:
        name = getattr(region, "name", None)
        references = tuple(getattr(region, "references", ()))
        if type(name) is not str or not name.strip():
            raise ValueError("scope name must be a non-empty string")
        if not references or any(
            type(reference) not in {MeshEntityRef, LogicalEntityRef}
            for reference in references
        ):
            raise TypeError(
                "native scopes require non-empty mesh references"
            )
        if all(type(reference) is MeshEntityRef for reference in references):
            _materialize_mesh_scope(updated, name.strip(), references)
        elif all(type(reference) is LogicalEntityRef for reference in references):
            _materialize_legacy_scope(
                updated,
                catalog,
                name.strip(),
                references,
            )
        else:
            raise TypeError("one native scope cannot mix mesh and logical refs")
    return updated


def _materialize_mesh_scope(
    model: Any,
    name: str,
    references: tuple[MeshEntityRef, ...],
) -> None:
    kind = references[0].kind
    if any(reference.kind != kind for reference in references):
        raise ValueError("one mesh scope cannot mix entity kinds")
    node_ids = {int(node.id) for node in model.mesh.nodes}
    element_ids = {int(element.id) for element in model.mesh.elements}
    if kind == "node":
        selected = tuple(int(reference.node_id) for reference in references)
        _require_ids(selected, node_ids, label="node")
        model.node_sets[name] = NodeSet(name, selected)
        return
    if kind == "element":
        selected = tuple(int(reference.element_id) for reference in references)
        _require_ids(selected, element_ids, label="element")
        model.element_sets[name] = ElementSet(name, selected)
        return
    if kind == "edge":
        lookup = {
            (int(element_id), int(local_index)): tuple(int(value) for value in ids)
            for element_id, local_index, ids in mesh_edges.all(model.mesh)
        }
        model.edges[name] = Edge(
            name,
            tuple(
                ElementEdge(
                    int(reference.element_id),
                    int(reference.local_index),
                    _validated_boundary_nodes(reference, lookup, label="edge"),
                )
                for reference in references
            ),
        )
        return
    if kind == "face":
        lookup = {
            (int(element_id), int(local_index)): tuple(int(value) for value in ids)
            for element_id, local_index, ids in mesh_faces.boundary(model.mesh)
        }
        model.surfaces[name] = Surface(
            name,
            tuple(
                ElementFace(
                    int(reference.element_id),
                    int(reference.local_index),
                    _validated_boundary_nodes(reference, lookup, label="face"),
                )
                for reference in references
            ),
        )
        return
    raise ValueError(f"unsupported mesh scope kind: {kind!r}")


def mesh_references_for_logical_entities(
    model: Any,
    references: Iterable[LogicalEntityRef],
    *,
    mesh_kind: str,
) -> tuple[MeshEntityRef, ...]:
    """Expand whole logical entities to their current mesh-level references."""

    if mesh_kind not in {"node", "edge", "face", "element"}:
        raise ValueError(f"unsupported mesh scope kind: {mesh_kind!r}")
    metadata = getattr(model, "metadata", None)
    catalog = (
        metadata.get(NATIVE_SCOPE_CATALOG_KEY)
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(catalog, Mapping):
        raise ValueError("current model has no geometry-to-mesh scope catalog")
    selected: set[MeshEntityRef] = set()
    for reference in references:
        if type(reference) is not LogicalEntityRef:
            raise TypeError("logical scope selection requires LogicalEntityRef")
        entry = catalog.get(reference.logical_id)
        if not isinstance(entry, Mapping):
            raise ValueError(
                "scope reference is absent from the current mesh: "
                f"{reference.logical_id}"
            )
        if mesh_kind == "node":
            selected.update(
                MeshEntityRef.node(node_id)
                for node_id in _integer_values(entry.get("node_ids", ()))
            )
        elif mesh_kind == "element":
            selected.update(
                MeshEntityRef.element(element_id)
                for element_id in _integer_values(
                    entry.get("element_ids", ())
                )
            )
        else:
            rows = _boundary_rows(
                entry.get(f"{mesh_kind}s", ()),
                label=mesh_kind,
            )
            constructor = (
                MeshEntityRef.edge
                if mesh_kind == "edge"
                else MeshEntityRef.face
            )
            selected.update(
                constructor(element_id, local_index, node_ids)
                for (element_id, local_index), node_ids in rows.items()
            )
    if not selected:
        raise ValueError(
            f"selected geometry does not resolve to mesh {mesh_kind} entities"
        )
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.kind,
                item.identity,
                item.node_ids,
            ),
        )
    )


def _validated_boundary_nodes(
    reference: MeshEntityRef,
    lookup: Mapping[tuple[int, int], tuple[int, ...]],
    *,
    label: str,
) -> tuple[int, ...]:
    key = (int(reference.element_id), int(reference.local_index))
    actual = lookup.get(key)
    if actual is None:
        raise ValueError(
            f"{label} reference is absent from the current mesh: {key}"
        )
    if tuple(reference.node_ids) != actual:
        raise ValueError(
            f"{label} reference connectivity is stale for the current mesh: {key}"
        )
    return actual


def _require_ids(
    selected: Iterable[int],
    available: set[int],
    *,
    label: str,
) -> None:
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError(
            f"{label} reference is absent from the current mesh: {missing[0]}"
        )


def _materialize_legacy_scope(
    model: Any,
    catalog: Mapping[str, Any],
    name: str,
    references: tuple[LogicalEntityRef, ...],
) -> None:
    node_ids: set[int] = set()
    element_ids: set[int] = set()
    edge_rows: dict[tuple[int, int], tuple[int, ...]] = {}
    face_rows: dict[tuple[int, int], tuple[int, ...]] = {}

    for reference in references:
        entry = catalog.get(reference.logical_id)
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"scope reference is absent from the current mesh: "
                f"{reference.logical_id}"
            )
        if entry.get("kind") != reference.kind:
            raise ValueError(
                f"scope reference kind does not match the current mesh: "
                f"{reference.logical_id}"
            )
        node_ids.update(_integer_values(entry.get("node_ids", ())))
        element_ids.update(_integer_values(entry.get("element_ids", ())))
        edge_rows.update(
            _boundary_rows(entry.get("edges", ()), label="edge")
        )
        face_rows.update(
            _boundary_rows(entry.get("faces", ()), label="face")
        )

    if not (node_ids or element_ids or edge_rows or face_rows):
        raise ValueError(f"scope {name!r} resolves to an empty mesh selection")
    if node_ids:
        model.node_sets[name] = NodeSet(name, sorted(node_ids))
    if element_ids:
        model.element_sets[name] = ElementSet(
            name,
            sorted(element_ids),
        )
    if edge_rows:
        model.edges[name] = Edge(
            name,
            tuple(
                ElementEdge(element_id, local_index, node_ids)
                for (element_id, local_index), node_ids
                in sorted(edge_rows.items())
            ),
        )
    if face_rows:
        model.surfaces[name] = Surface(
            name,
            tuple(
                ElementFace(element_id, local_index, node_ids)
                for (element_id, local_index), node_ids
                in sorted(face_rows.items())
            ),
        )


def _integer_values(values: Any) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError("mesh scope IDs must be an iterable of integers")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("mesh scope IDs must contain only integers")
        result.append(int(value))
    return tuple(result)


def _boundary_rows(
    values: Any,
    *,
    label: str,
) -> dict[tuple[int, int], tuple[int, ...]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"mesh scope {label} rows must be iterable")
    result: dict[tuple[int, int], tuple[int, ...]] = {}
    for value in values:
        row = tuple(value)
        if len(row) != 3:
            raise ValueError(
                f"mesh scope {label} rows require element, local index, nodes"
            )
        element_id, local_index, raw_node_ids = row
        if (
            isinstance(element_id, bool)
            or not isinstance(element_id, int)
            or isinstance(local_index, bool)
            or not isinstance(local_index, int)
        ):
            raise TypeError(
                f"mesh scope {label} element and local indices must be integers"
            )
        node_ids = _integer_values(raw_node_ids)
        result[(int(element_id), int(local_index))] = node_ids
    return result


__all__ = [
    "NATIVE_SCOPE_CATALOG_KEY",
    "can_materialize_native_scopes",
    "has_native_scope_catalog",
    "materialize_native_scopes",
    "mesh_references_for_logical_entities",
]
