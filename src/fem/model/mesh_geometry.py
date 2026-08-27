"""Geometry queries over canonical mesh entities."""

from __future__ import annotations

from typing import Any

import numpy as np


def line3d_geometry(
    mesh: Any,
    element: Any,
    node_lookup: dict[int, Any] | None = None,
) -> tuple[float, np.ndarray]:
    """Return length and unit tangent for a two-node spatial line."""

    if len(element.node_ids) != 2:
        raise ValueError(
            f"Line2 element {element.id} requires 2 nodes, "
            f"got {len(element.node_ids)}; node_ids={element.node_ids}"
        )
    lookup = (
        {int(node.id): node for node in mesh.nodes}
        if node_lookup is None
        else node_lookup
    )
    try:
        first = lookup[element.node_ids[0]]
        second = lookup[element.node_ids[1]]
    except KeyError as exc:
        raise KeyError(
            f"Element {element.id} references missing node {exc.args[0]}"
        ) from exc
    delta = np.array(
        [second.x - first.x, second.y - first.y, second.z - first.z],
        dtype=float,
    )
    length = float(np.linalg.norm(delta))
    if length <= 0.0:
        raise ValueError(f"Line2 element {element.id} has zero length")
    return length, delta / length


__all__ = ["line3d_geometry"]
