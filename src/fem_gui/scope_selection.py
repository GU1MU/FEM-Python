"""Geometry-semantic scope picking backed by mesh-level references."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from fem.application import MeshEntityRef
from fem.application.native_scope_materialization import (
    mesh_references_for_logical_entities,
)
from fem.geometry import LogicalEntityRef, NATIVE_GEOMETRY_TYPES
from fem.selection import edges as mesh_edges
from fem.selection import faces as mesh_faces

from .geometry_preview import GeometryPreview, build_geometry_preview


@dataclass(frozen=True, slots=True)
class ScopeSelectionTopology:
    """Selectable whole geometry entities and their exact mesh expansion."""

    preview: GeometryPreview
    mesh_references: dict[
        LogicalEntityRef,
        tuple[MeshEntityRef, ...],
    ]


def build_scope_selection_topology(
    model: Any,
    recipe: object | None = None,
    *,
    feature_angle_degrees: float = 30.0,
) -> ScopeSelectionTopology:
    """Prefer exact native CAD ownership and infer imported mesh features."""

    if isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        exact = _native_scope_selection_topology(model, recipe)
        if exact.mesh_references:
            return exact
    return _inferred_scope_selection_topology(
        model,
        feature_angle_degrees=feature_angle_degrees,
    )


def _native_scope_selection_topology(
    model: Any,
    recipe: object,
) -> ScopeSelectionTopology:
    preview = build_geometry_preview(recipe)
    logical_ids = {
        logical_id
        for logical_id in (
            *preview.edge_logical_ids,
            *preview.face_logical_ids,
            *preview.face_body_logical_ids,
            *preview.edge_body_logical_ids,
            *preview.point_body_logical_ids,
            preview.body_logical_id,
        )
        if logical_id is not None
    }
    references: dict[
        LogicalEntityRef,
        tuple[MeshEntityRef, ...],
    ] = {}
    for logical_id in sorted(logical_ids):
        logical = LogicalEntityRef(logical_id)
        if (
            logical.kind == "body"
            and preview.topological_dimension != 3
        ):
            continue
        mesh_kind = {
            "edge": (
                "element"
                if preview.topological_dimension == 1
                else "edge"
            ),
            "face": (
                "element"
                if preview.topological_dimension == 2
                else "face"
            ),
            "body": "element",
        }.get(logical.kind)
        if mesh_kind is None:
            continue
        try:
            expanded = mesh_references_for_logical_entities(
                model,
                (logical,),
                mesh_kind=mesh_kind,
            )
        except ValueError:
            continue
        references[logical] = expanded
    return ScopeSelectionTopology(preview, references)


def _inferred_scope_selection_topology(
    model: Any,
    *,
    feature_angle_degrees: float,
) -> ScopeSelectionTopology:
    mesh = model.mesh
    node_ids = tuple(int(node.id) for node in mesh.nodes)
    points = tuple(_node_coordinates(node) for node in mesh.nodes)
    point_index = {
        node_id: index for index, node_id in enumerate(node_ids)
    }
    coordinates = {
        node_id: np.asarray(points[index], dtype=float)
        for index, node_id in enumerate(node_ids)
    }
    face_rows = tuple(
        (
            int(element_id),
            int(local_index),
            tuple(int(node_id) for node_id in row_node_ids),
        )
        for element_id, local_index, row_node_ids
        in mesh_faces.boundary(mesh)
    )
    if face_rows:
        return _infer_solid_topology(
            mesh,
            points,
            point_index,
            coordinates,
            face_rows,
            feature_angle_degrees,
        )
    edge_rows = tuple(
        (
            int(element_id),
            int(local_index),
            tuple(int(node_id) for node_id in row_node_ids),
        )
        for element_id, local_index, row_node_ids
        in mesh_edges.boundary(mesh)
    )
    if edge_rows:
        return _infer_planar_topology(
            mesh,
            points,
            point_index,
            coordinates,
            edge_rows,
            feature_angle_degrees,
        )
    return _infer_line_topology(
        mesh,
        points,
        point_index,
        coordinates,
        feature_angle_degrees,
    )


def _infer_planar_topology(
    mesh: Any,
    points: tuple[tuple[float, float, float], ...],
    point_index: dict[int, int],
    coordinates: dict[int, np.ndarray],
    edge_rows: tuple[tuple[int, int, tuple[int, ...]], ...],
    feature_angle_degrees: float,
) -> ScopeSelectionTopology:
    segments = tuple(
        (row[2][0], row[2][-1])
        for row in edge_rows
        if len(row[2]) >= 2
    )
    groups = _smooth_segment_groups(
        segments,
        coordinates,
        feature_angle_degrees,
    )
    edge_cells: list[tuple[int, ...]] = []
    edge_logical_ids: list[str] = []
    references: dict[
        LogicalEntityRef,
        tuple[MeshEntityRef, ...],
    ] = {}
    for group_number, group in enumerate(groups, start=1):
        logical = LogicalEntityRef(f"edge:inferred-{group_number:03d}")
        group_refs: list[MeshEntityRef] = []
        for row_index in group:
            element_id, local_index, row_node_ids = edge_rows[row_index]
            edge_cells.append(
                tuple(point_index[node_id] for node_id in row_node_ids)
            )
            edge_logical_ids.append(logical.logical_id)
            group_refs.append(
                MeshEntityRef.edge(
                    element_id,
                    local_index,
                    row_node_ids,
                )
            )
        references[logical] = tuple(group_refs)

    element_rows = tuple(
        (
            int(element.id),
            _element_corner_node_ids(element),
        )
        for element in mesh.elements
        if len(_element_corner_node_ids(element)) >= 3
    )
    face_cells: list[tuple[int, ...]] = []
    face_logical_ids: list[str] = []
    for group_number, group in enumerate(
        _connected_element_groups(element_rows),
        start=1,
    ):
        logical = LogicalEntityRef(
            f"face:inferred-domain-{group_number:03d}"
        )
        group_refs: list[MeshEntityRef] = []
        for element_index in group:
            element_id, corner_node_ids = element_rows[element_index]
            face_cells.append(
                tuple(
                    point_index[node_id]
                    for node_id in corner_node_ids
                )
            )
            face_logical_ids.append(logical.logical_id)
            group_refs.append(MeshEntityRef.element(element_id))
        references[logical] = tuple(group_refs)
    preview = GeometryPreview(
        points=points,
        faces=tuple(face_cells),
        edges=tuple(edge_cells),
        face_logical_ids=tuple(face_logical_ids),
        edge_logical_ids=tuple(edge_logical_ids),
        topological_dimension=2,
    )
    return ScopeSelectionTopology(preview, references)


def _infer_line_topology(
    mesh: Any,
    points: tuple[tuple[float, float, float], ...],
    point_index: dict[int, int],
    coordinates: dict[int, np.ndarray],
    feature_angle_degrees: float,
) -> ScopeSelectionTopology:
    element_rows = tuple(
        (
            int(element.id),
            tuple(int(node_id) for node_id in element.node_ids),
        )
        for element in mesh.elements
        if len(tuple(element.node_ids)) >= 2
    )
    segments = tuple(
        (node_ids[0], node_ids[-1])
        for _element_id, node_ids in element_rows
    )
    edge_cells: list[tuple[int, ...]] = []
    edge_logical_ids: list[str] = []
    references: dict[
        LogicalEntityRef,
        tuple[MeshEntityRef, ...],
    ] = {}
    for group_number, group in enumerate(
        _smooth_segment_groups(
            segments,
            coordinates,
            feature_angle_degrees,
        ),
        start=1,
    ):
        logical = LogicalEntityRef(f"edge:inferred-{group_number:03d}")
        group_refs: list[MeshEntityRef] = []
        for element_index in group:
            element_id, node_ids = element_rows[element_index]
            edge_cells.append(
                tuple(point_index[node_id] for node_id in node_ids)
            )
            edge_logical_ids.append(logical.logical_id)
            group_refs.append(MeshEntityRef.element(element_id))
        references[logical] = tuple(group_refs)
    return ScopeSelectionTopology(
        GeometryPreview(
            points=points,
            faces=(),
            edges=tuple(edge_cells),
            edge_logical_ids=tuple(edge_logical_ids),
            topological_dimension=1,
        ),
        references,
    )


def _infer_solid_topology(
    mesh: Any,
    points: tuple[tuple[float, float, float], ...],
    point_index: dict[int, int],
    coordinates: dict[int, np.ndarray],
    face_rows: tuple[tuple[int, int, tuple[int, ...]], ...],
    feature_angle_degrees: float,
) -> ScopeSelectionTopology:
    face_corners = tuple(
        _face_corner_node_ids(row_node_ids)
        for _element_id, _local_index, row_node_ids in face_rows
    )
    normals = tuple(
        _polygon_normal(tuple(coordinates[node_id] for node_id in corners))
        for corners in face_corners
    )
    face_groups = _smooth_face_groups(
        face_corners,
        normals,
        feature_angle_degrees,
    )
    face_group_by_index = {
        face_index: group_number
        for group_number, group in enumerate(face_groups, start=1)
        for face_index in group
    }
    preview_faces: list[tuple[int, ...]] = []
    face_logical_ids: list[str] = []
    references: dict[
        LogicalEntityRef,
        tuple[MeshEntityRef, ...],
    ] = {}
    for group_number, group in enumerate(face_groups, start=1):
        logical = LogicalEntityRef(f"face:inferred-{group_number:03d}")
        group_refs: list[MeshEntityRef] = []
        for face_index in group:
            element_id, local_index, row_node_ids = face_rows[face_index]
            preview_faces.append(
                tuple(
                    point_index[node_id]
                    for node_id in face_corners[face_index]
                )
            )
            face_logical_ids.append(logical.logical_id)
            group_refs.append(
                MeshEntityRef.face(
                    element_id,
                    local_index,
                    row_node_ids,
                )
            )
        references[logical] = tuple(group_refs)

    feature_segments = _solid_feature_segments(
        face_corners,
        face_group_by_index,
    )
    element_edges = _element_edge_representatives(mesh)
    selectable_segments = tuple(
        segment
        for segment in feature_segments
        if tuple(sorted(segment)) in element_edges
    )
    edge_groups = _smooth_segment_groups(
        selectable_segments,
        coordinates,
        feature_angle_degrees,
    )
    preview_edges: list[tuple[int, ...]] = []
    edge_logical_ids: list[str] = []
    for group_number, group in enumerate(edge_groups, start=1):
        logical = LogicalEntityRef(f"edge:inferred-{group_number:03d}")
        group_refs: list[MeshEntityRef] = []
        for segment_index in group:
            segment = selectable_segments[segment_index]
            row = element_edges[tuple(sorted(segment))]
            preview_edges.append(
                tuple(point_index[node_id] for node_id in segment)
            )
            edge_logical_ids.append(logical.logical_id)
            group_refs.append(MeshEntityRef.edge(*row))
        references[logical] = tuple(group_refs)

    preview = GeometryPreview(
        points=points,
        faces=tuple(preview_faces),
        edges=tuple(preview_edges),
        face_logical_ids=tuple(face_logical_ids),
        edge_logical_ids=tuple(edge_logical_ids),
        body_logical_id="body:inferred-domain-001",
        topological_dimension=3,
    )
    references[LogicalEntityRef("body:inferred-domain-001")] = tuple(
        MeshEntityRef.element(int(element.id))
        for element in mesh.elements
    )
    return ScopeSelectionTopology(preview, references)


def _smooth_face_groups(
    faces: tuple[tuple[int, ...], ...],
    normals: tuple[np.ndarray, ...],
    feature_angle_degrees: float,
) -> tuple[tuple[int, ...], ...]:
    edges: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for start, end in zip(face, (*face[1:], face[0])):
            edges.setdefault(tuple(sorted((start, end))), []).append(
                face_index
            )
    groups = _DisjointGroups(len(faces))
    cosine = math.cos(math.radians(float(feature_angle_degrees)))
    for adjacent in edges.values():
        for first, second in zip(adjacent, adjacent[1:]):
            if abs(float(np.dot(normals[first], normals[second]))) >= cosine:
                groups.union(first, second)
    return groups.values()


def _smooth_segment_groups(
    segments: tuple[tuple[int, int], ...],
    coordinates: dict[int, np.ndarray],
    feature_angle_degrees: float,
) -> tuple[tuple[int, ...], ...]:
    attached: dict[int, list[int]] = {}
    for segment_index, (start, end) in enumerate(segments):
        attached.setdefault(start, []).append(segment_index)
        attached.setdefault(end, []).append(segment_index)
    groups = _DisjointGroups(len(segments))
    threshold = -math.cos(math.radians(float(feature_angle_degrees)))
    for node_id, adjacent in attached.items():
        if len(adjacent) != 2:
            continue
        first, second = adjacent
        first_other = _other_endpoint(segments[first], node_id)
        second_other = _other_endpoint(segments[second], node_id)
        first_vector = _unit(coordinates[first_other] - coordinates[node_id])
        second_vector = _unit(coordinates[second_other] - coordinates[node_id])
        if float(np.dot(first_vector, second_vector)) <= threshold:
            groups.union(first, second)
    return groups.values()


def _solid_feature_segments(
    faces: tuple[tuple[int, ...], ...],
    group_by_face: dict[int, int],
) -> tuple[tuple[int, int], ...]:
    adjacent: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for start, end in zip(face, (*face[1:], face[0])):
            adjacent.setdefault(tuple(sorted((start, end))), []).append(
                face_index
            )
    return tuple(
        edge
        for edge, face_indices in sorted(adjacent.items())
        if (
            len(face_indices) == 1
            or len({group_by_face[index] for index in face_indices}) > 1
        )
    )


def _element_edge_representatives(
    mesh: Any,
) -> dict[tuple[int, int], tuple[int, int, tuple[int, ...]]]:
    representatives: dict[
        tuple[int, int],
        tuple[int, int, tuple[int, ...]],
    ] = {}
    for element_id, local_index, raw_node_ids in mesh_edges.all(mesh):
        node_ids = tuple(int(node_id) for node_id in raw_node_ids)
        if len(node_ids) < 2:
            continue
        key = tuple(sorted((node_ids[0], node_ids[-1])))
        candidate = (int(element_id), int(local_index), node_ids)
        current = representatives.get(key)
        if current is None or candidate[:2] < current[:2]:
            representatives[key] = candidate
    return representatives


def _face_corner_node_ids(node_ids: tuple[int, ...]) -> tuple[int, ...]:
    if len(node_ids) == 8:
        return node_ids[:4]
    if len(node_ids) == 6:
        return node_ids[:3]
    return node_ids


def _element_corner_node_ids(element: Any) -> tuple[int, ...]:
    node_ids = tuple(int(node_id) for node_id in element.node_ids)
    element_type = str(element.type).lower()
    if any(name in element_type for name in ("tri6", "cps6", "cpe6")):
        return node_ids[:3]
    if any(name in element_type for name in ("quad8", "cps8", "cpe8")):
        return node_ids[:4]
    return node_ids


def _connected_element_groups(
    rows: tuple[tuple[int, tuple[int, ...]], ...],
) -> tuple[tuple[int, ...], ...]:
    groups = _DisjointGroups(len(rows))
    owners: dict[tuple[int, int], int] = {}
    for element_index, (_element_id, node_ids) in enumerate(rows):
        for start, end in zip(node_ids, (*node_ids[1:], node_ids[0])):
            key = tuple(sorted((start, end)))
            other = owners.get(key)
            if other is None:
                owners[key] = element_index
            else:
                groups.union(other, element_index)
    return groups.values()


def _polygon_normal(points: tuple[np.ndarray, ...]) -> np.ndarray:
    origin = points[0]
    for index in range(1, len(points) - 1):
        normal = np.cross(points[index] - origin, points[index + 1] - origin)
        length = float(np.linalg.norm(normal))
        if length > 1.0e-12:
            return normal / length
    return np.asarray((0.0, 0.0, 1.0), dtype=float)


def _node_coordinates(node: Any) -> tuple[float, float, float]:
    return (
        float(node.x),
        float(node.y),
        float(getattr(node, "z", 0.0)),
    )


def _other_endpoint(segment: tuple[int, int], node_id: int) -> int:
    return segment[1] if segment[0] == node_id else segment[0]


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-15:
        return np.zeros(3, dtype=float)
    return vector / length


class _DisjointGroups:
    def __init__(self, size: int) -> None:
        self._parents = list(range(size))

    def find(self, value: int) -> int:
        parent = self._parents[value]
        if parent != value:
            self._parents[value] = self.find(parent)
        return self._parents[value]

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self._parents[second_root] = first_root

    def values(self) -> tuple[tuple[int, ...], ...]:
        grouped: dict[int, list[int]] = {}
        for value in range(len(self._parents)):
            grouped.setdefault(self.find(value), []).append(value)
        return tuple(
            tuple(values)
            for _root, values in sorted(
                grouped.items(),
                key=lambda item: item[1][0],
            )
        )


__all__ = [
    "ScopeSelectionTopology",
    "build_scope_selection_topology",
]
