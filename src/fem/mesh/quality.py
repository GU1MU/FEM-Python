"""Backend-neutral mesh shape-quality analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np

from fem.core.mesh import MeshProtocol
from fem.elements.registry import get_element_capabilities


@dataclass(frozen=True, slots=True)
class MeshQualityReport:
    """Immutable normalized quality summary for one canonical mesh."""

    node_count: int
    element_count: int
    element_types: tuple[tuple[str, int], ...]
    checked_count: int
    unchecked_count: int
    unchecked_element_types: tuple[tuple[str, int], ...]
    minimum: float
    mean: float
    maximum: float
    worst_elements: tuple[tuple[int, float], ...]


def analyze_mesh(mesh: MeshProtocol) -> MeshQualityReport:
    """Return normalized shape scores where 1 is ideal and 0 is degenerate.

    Unsupported element types and malformed connectivity are counted as
    unchecked and never enter the score denominator.
    """

    if not isinstance(mesh, MeshProtocol):
        raise TypeError("mesh must implement fem.core.mesh.MeshProtocol")
    coordinates = {
        int(node.id): np.asarray(
            (float(node.x), float(node.y), float(getattr(node, "z", 0.0))),
            dtype=float,
        )
        for node in mesh.nodes
    }
    scores: list[tuple[int, float]] = []
    unchecked_types: Counter[str] = Counter()
    type_counts = Counter(str(element.type) for element in mesh.elements)
    for element in mesh.elements:
        element_type = str(element.type)
        try:
            capabilities = get_element_capabilities(element_type)
        except NotImplementedError:
            unchecked_types[element_type] += 1
            continue
        node_ids = tuple(int(node_id) for node_id in element.node_ids)
        if len(node_ids) != capabilities.node_count:
            unchecked_types[element_type] += 1
            continue
        try:
            points = np.asarray(
                [coordinates[node_id] for node_id in node_ids],
                dtype=float,
            )
        except KeyError:
            unchecked_types[element_type] += 1
            continue
        score = _element_quality(capabilities.canonical_type, points)
        if score is None or not math.isfinite(score):
            unchecked_types[element_type] += 1
            continue
        scores.append(
            (
                int(element.id),
                float(np.clip(score, 0.0, 1.0)),
            )
        )

    values = np.asarray([score for _element_id, score in scores], dtype=float)
    unchecked_count = sum(unchecked_types.values())
    return MeshQualityReport(
        node_count=len(mesh.nodes),
        element_count=len(mesh.elements),
        element_types=tuple(sorted(type_counts.items())),
        checked_count=len(scores),
        unchecked_count=unchecked_count,
        unchecked_element_types=tuple(sorted(unchecked_types.items())),
        minimum=float(values.min()) if len(values) else 0.0,
        mean=float(values.mean()) if len(values) else 0.0,
        maximum=float(values.max()) if len(values) else 0.0,
        worst_elements=tuple(
            sorted(scores, key=lambda item: (item[1], item[0]))[:10]
        ),
    )


def _element_quality(
    canonical_type: str,
    points: np.ndarray,
) -> float | None:
    if canonical_type in {"Tri3", "Tri6"}:
        return _triangle_quality(points[:3])
    if canonical_type in {"Quad4", "Quad8"}:
        return _quadrilateral_quality(points[:4])
    if canonical_type in {"Tet4", "Tet10"}:
        return _tetrahedron_quality(points[:4])
    if canonical_type in {"Hex8", "Hex20"}:
        return _hexahedron_quality(points[:8])
    if canonical_type in {"Truss2", "Beam2"}:
        return (
            1.0
            if float(np.linalg.norm(points[1] - points[0])) > 0.0
            else 0.0
        )
    return None


def _triangle_quality(points: np.ndarray) -> float:
    edges = (
        points[1] - points[0],
        points[2] - points[1],
        points[0] - points[2],
    )
    denominator = sum(float(np.dot(edge, edge)) for edge in edges)
    area = 0.5 * float(np.linalg.norm(np.cross(edges[0], -edges[2])))
    return (
        0.0
        if denominator <= 0.0
        else 4.0 * math.sqrt(3.0) * area / denominator
    )


def _quadrilateral_quality(points: np.ndarray) -> float:
    edges = tuple(
        points[(index + 1) % 4] - points[index] for index in range(4)
    )
    lengths = np.asarray([np.linalg.norm(edge) for edge in edges], dtype=float)
    maximum = float(lengths.max(initial=0.0))
    if maximum <= 0.0:
        return 0.0
    sine_values = []
    for index in range(4):
        previous = -edges[index - 1]
        current = edges[index]
        denominator = float(
            np.linalg.norm(previous) * np.linalg.norm(current)
        )
        sine_values.append(
            0.0
            if denominator <= 0.0
            else float(np.linalg.norm(np.cross(previous, current)))
            / denominator
        )
    return float(lengths.min() / maximum * min(sine_values))


def _tetrahedron_quality(points: np.ndarray) -> float:
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    denominator = sum(
        float(np.dot(points[j] - points[i], points[j] - points[i]))
        for i, j in pairs
    )
    volume = (
        abs(
            float(
                np.linalg.det(
                    np.column_stack(
                        (
                            points[1] - points[0],
                            points[2] - points[0],
                            points[3] - points[0],
                        )
                    )
                )
            )
        )
        / 6.0
    )
    return (
        0.0
        if denominator <= 0.0
        else 12.0 * (3.0 * volume) ** (2.0 / 3.0) / denominator
    )


def _hexahedron_quality(points: np.ndarray) -> float:
    pairs = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    lengths = np.asarray(
        [np.linalg.norm(points[j] - points[i]) for i, j in pairs],
        dtype=float,
    )
    maximum = float(lengths.max(initial=0.0))
    return 0.0 if maximum <= 0.0 else float(lengths.min() / maximum)


__all__ = ["MeshQualityReport", "analyze_mesh"]
