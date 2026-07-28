"""Internal read-only projection of typed constraints onto runtime nodes.

These helpers never materialize named regions or mutate authoring definitions.
Expanded node IDs are transient solver/visualization data only.
"""

from __future__ import annotations

from typing import Any


DISPLACEMENT_TARGET_KINDS = frozenset({"node_set", "edge", "surface"})


def displacement_target_kind(constraint: Any) -> str:
    """Return the normalized region namespace for a displacement constraint."""

    kind = str(getattr(constraint, "target_kind", "node_set")).strip().casefold()
    if kind not in DISPLACEMENT_TARGET_KINDS:
        raise ValueError(f"unsupported displacement target kind: {kind!r}")
    return kind


def resolve_displacement_node_ids(
    model: Any,
    constraint: Any,
) -> tuple[int, ...]:
    """Return transient ordered node IDs without changing the model."""

    target = getattr(constraint, "target", None)
    kind = displacement_target_kind(constraint)
    if isinstance(target, int):
        if kind != "node_set":
            raise ValueError(
                "integer displacement targets require target_kind='node_set'"
            )
        return (int(target),)
    if not isinstance(target, str):
        raise TypeError("displacement target must be a node ID or region name")

    if kind == "node_set":
        collection = getattr(model, "node_sets", {})
        if target not in collection:
            raise KeyError(f"node set {target} is not defined")
        values = collection[target].node_ids
    elif kind == "edge":
        collection = getattr(model, "edges", {})
        if target not in collection:
            raise KeyError(f"edge {target} is not defined")
        values = (
            node_id
            for entry in collection[target].edges
            for node_id in entry.node_ids
        )
    else:
        collection = getattr(model, "surfaces", {})
        if target not in collection:
            raise KeyError(f"surface {target} is not defined")
        values = (
            node_id
            for entry in collection[target].faces
            for node_id in entry.node_ids
        )

    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        node_id = int(value)
        if node_id not in seen:
            seen.add(node_id)
            result.append(node_id)
    return tuple(result)

