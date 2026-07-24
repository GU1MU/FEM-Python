"""Small backend-neutral mesh statistics and shape-quality checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class MeshQualityReport:
    node_count: int
    element_count: int
    element_types: tuple[tuple[str, int], ...]
    checked_count: int
    minimum: float
    mean: float
    maximum: float
    worst_elements: tuple[tuple[int, float], ...]


def analyze_mesh(model: Any) -> MeshQualityReport:
    """Return normalized shape scores where 1 is ideal and 0 is degenerate."""
    mesh = model.mesh
    coordinates = {
        int(node.id): np.asarray(
            (float(node.x), float(node.y), float(getattr(node, "z", 0.0))),
            dtype=float,
        )
        for node in mesh.nodes
    }
    scores: list[tuple[int, float]] = []
    for element in mesh.elements:
        points = np.asarray([coordinates[int(node_id)] for node_id in element.node_ids])
        score = _element_quality(str(element.type), points)
        if score is not None and math.isfinite(score):
            scores.append((int(element.id), float(np.clip(score, 0.0, 1.0))))
    values = np.asarray([score for _element_id, score in scores], dtype=float)
    type_counts = Counter(str(element.type) for element in mesh.elements)
    return MeshQualityReport(
        node_count=len(mesh.nodes),
        element_count=len(mesh.elements),
        element_types=tuple(sorted(type_counts.items())),
        checked_count=len(scores),
        minimum=float(values.min()) if len(values) else 0.0,
        mean=float(values.mean()) if len(values) else 0.0,
        maximum=float(values.max()) if len(values) else 0.0,
        worst_elements=tuple(sorted(scores, key=lambda item: item[1])[:10]),
    )


def _element_quality(element_type: str, points: np.ndarray) -> float | None:
    kind = element_type.casefold()
    if kind.startswith(("tri", "cps3", "cpe3", "s3")) and len(points) >= 3:
        return _triangle_quality(points[:3])
    if kind.startswith(("quad", "cps4", "cpe4", "s4")) and len(points) >= 4:
        return _quadrilateral_quality(points[:4])
    if kind.startswith(("tet", "c3d4", "c3d10")) and len(points) >= 4:
        return _tetrahedron_quality(points[:4])
    if kind.startswith(("hex", "c3d8", "c3d20")) and len(points) >= 8:
        return _hexahedron_quality(points[:8])
    if kind.startswith(("line", "b", "t")) and len(points) >= 2:
        return 1.0 if np.linalg.norm(points[1] - points[0]) > 0.0 else 0.0
    return None


def _triangle_quality(points: np.ndarray) -> float:
    edges = (
        points[1] - points[0],
        points[2] - points[1],
        points[0] - points[2],
    )
    denominator = sum(float(np.dot(edge, edge)) for edge in edges)
    area = 0.5 * float(np.linalg.norm(np.cross(edges[0], -edges[2])))
    return 0.0 if denominator <= 0.0 else 4.0 * math.sqrt(3.0) * area / denominator


def _quadrilateral_quality(points: np.ndarray) -> float:
    edges = tuple(points[(index + 1) % 4] - points[index] for index in range(4))
    lengths = np.asarray([np.linalg.norm(edge) for edge in edges], dtype=float)
    if lengths.max(initial=0.0) <= 0.0:
        return 0.0
    sine_values = []
    for index in range(4):
        previous = -edges[index - 1]
        current = edges[index]
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(current))
        sine_values.append(
            0.0 if denominator <= 0.0 else float(np.linalg.norm(np.cross(previous, current))) / denominator
        )
    return float(lengths.min() / lengths.max() * min(sine_values))


def _tetrahedron_quality(points: np.ndarray) -> float:
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    denominator = sum(float(np.dot(points[j] - points[i], points[j] - points[i])) for i, j in pairs)
    volume = abs(float(np.linalg.det(np.column_stack(
        (points[1] - points[0], points[2] - points[0], points[3] - points[0])
    )))) / 6.0
    return 0.0 if denominator <= 0.0 else 12.0 * (3.0 * volume) ** (2.0 / 3.0) / denominator


def _hexahedron_quality(points: np.ndarray) -> float:
    pairs = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    lengths = np.asarray([np.linalg.norm(points[j] - points[i]) for i, j in pairs])
    maximum = float(lengths.max(initial=0.0))
    return 0.0 if maximum <= 0.0 else float(lengths.min() / maximum)
