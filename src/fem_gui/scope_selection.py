"""Geometry-semantic scope picking backed by mesh-level references."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from fem.application import MeshEntityRef
from fem.application.definitions import mesh_entity_ref_sort_key
from fem.application.native_scope_materialization import (
    NATIVE_PART_OWNERSHIP_KEY,
    NATIVE_SCOPE_CATALOG_KEY,
    mesh_references_for_logical_entities,
)
from fem.geometry import LogicalEntityRef, NATIVE_GEOMETRY_TYPES
from fem.geometry.part_namespace import part_id_sort_key
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


@dataclass(frozen=True, slots=True)
class MeshSelectionTopology:
    """Deterministic mesh filters and whole-topology expansion."""

    topological_dimension: int
    part_elements: dict[str, tuple[MeshEntityRef, ...]]
    node_owners: dict[int, str]
    element_owners: dict[int, str]
    edge_expansions: dict[
        tuple[str, tuple[int, int]],
        tuple[MeshEntityRef, ...],
    ]
    face_expansions: dict[
        tuple[str, tuple[int, int]],
        tuple[MeshEntityRef, ...],
    ]

    def reference_kind(self, selection_filter: str) -> str:
        """Return the MeshEntityRef kind produced by one semantic filter."""

        if selection_filter == "point":
            return "node"
        if selection_filter in {"element", "body"}:
            return "element"
        if selection_filter == "edge":
            return "element" if self.topological_dimension == 1 else "edge"
        if selection_filter == "face":
            return "element" if self.topological_dimension == 2 else "face"
        raise ValueError("unsupported mesh selection filter")

    def canonical_reference(self, reference: MeshEntityRef) -> MeshEntityRef:
        """Attach the deterministic Part owner to one picked mesh reference."""

        if type(reference) is not MeshEntityRef:
            raise TypeError("mesh selection requires MeshEntityRef")
        identity = (
            int(reference.node_id)
            if reference.kind == "node"
            else int(reference.element_id)
        )
        owner = (
            self.node_owners.get(identity)
            if reference.kind == "node"
            else self.element_owners.get(identity)
        )
        if owner is None or reference.part_id == owner:
            return reference
        return MeshEntityRef(
            reference.kind,
            node_id=reference.node_id,
            element_id=reference.element_id,
            local_index=reference.local_index,
            node_ids=reference.node_ids,
            part_id=owner,
        )

    def expand(
        self,
        selection_filter: str,
        picked: MeshEntityRef,
    ) -> tuple[MeshEntityRef, ...]:
        """Expand one local hit into its complete semantic mesh entity."""

        reference = self.canonical_reference(picked)
        if selection_filter == "point":
            return (reference,) if reference.kind == "node" else ()
        if selection_filter == "element":
            return (reference,) if reference.kind == "element" else ()
        if selection_filter == "body":
            if reference.kind != "element" or reference.part_id is None:
                return ()
            return self.part_elements.get(reference.part_id, ())
        expansions = (
            self.edge_expansions
            if selection_filter == "edge"
            else self.face_expansions
            if selection_filter == "face"
            else None
        )
        if expansions is None:
            raise ValueError("unsupported mesh selection filter")
        return expansions.get((reference.kind, reference.identity), ())

    def pick_references(
        self,
        selection_filter: str,
    ) -> tuple[MeshEntityRef, ...]:
        """Return stable local references that may seed a topology pick."""

        expansions = (
            self.edge_expansions
            if selection_filter == "edge"
            else self.face_expansions
            if selection_filter == "face"
            else None
        )
        if expansions is None:
            raise ValueError("topology pick references require edge or face")
        return _canonical_mesh_references(
            reference
            for group in expansions.values()
            for reference in group
        )


def build_mesh_selection_topology(
    model: Any,
    *,
    feature_angle_degrees: float = 30.0,
    scope_topology: ScopeSelectionTopology | None = None,
) -> MeshSelectionTopology:
    """Build exact native or deterministically inferred mesh selection groups."""

    mesh = model.mesh
    dimension = (
        scope_topology.preview.topological_dimension
        if scope_topology is not None
        else _mesh_topological_dimension(mesh)
    )
    node_owners, element_owners, part_elements = _mesh_part_ownership(
        model,
        dimension,
        scope_topology=scope_topology,
    )
    metadata = getattr(model, "metadata", None)
    catalog = (
        metadata.get(NATIVE_SCOPE_CATALOG_KEY)
        if isinstance(metadata, Mapping)
        else None
    )
    if scope_topology is not None:
        edge_groups, face_groups = _scope_mesh_topology_groups(
            scope_topology
        )
    elif isinstance(catalog, Mapping) and catalog:
        edge_groups, face_groups = _native_mesh_topology_groups(
            catalog,
            dimension,
            element_owners,
        )
    else:
        inferred = _inferred_scope_selection_topology(
            model,
            feature_angle_degrees=feature_angle_degrees,
        )
        edge_groups = tuple(
            references
            for logical, references in inferred.mesh_references.items()
            if logical.kind == "edge"
        )
        face_groups = tuple(
            references
            for logical, references in inferred.mesh_references.items()
            if logical.kind == "face"
        )
    canonical_edges = _owned_topology_groups(edge_groups, element_owners)
    canonical_faces = _owned_topology_groups(face_groups, element_owners)
    return MeshSelectionTopology(
        topological_dimension=dimension,
        part_elements=part_elements,
        node_owners=node_owners,
        element_owners=element_owners,
        edge_expansions=_topology_expansion_index(canonical_edges),
        face_expansions=_topology_expansion_index(canonical_faces),
    )


def _mesh_topological_dimension(mesh: Any) -> int:
    if mesh_faces.boundary(mesh):
        return 3
    if mesh_edges.boundary(mesh):
        return 2
    return 1


def _mesh_part_ownership(
    model: Any,
    dimension: int,
    *,
    scope_topology: ScopeSelectionTopology | None = None,
) -> tuple[
    dict[int, str],
    dict[int, str],
    dict[str, tuple[MeshEntityRef, ...]],
]:
    mesh = model.mesh
    metadata = getattr(model, "metadata", None)
    raw_ownership = (
        metadata.get(NATIVE_PART_OWNERSHIP_KEY)
        if isinstance(metadata, Mapping)
        else None
    )
    mesh_element_ids = {int(element.id) for element in mesh.elements}
    if isinstance(raw_ownership, Mapping):
        exact_rows: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        for raw_part_id, raw in sorted(
            raw_ownership.items(),
            key=lambda item: part_id_sort_key(str(item[0])),
        ):
            if not isinstance(raw, Mapping):
                continue
            part_id = str(raw_part_id)
            node_ids = tuple(sorted(int(value) for value in raw.get("node_ids", ())))
            element_ids = tuple(
                sorted(int(value) for value in raw.get("element_ids", ()))
            )
            exact_rows.append((part_id, node_ids, element_ids))
        if {
            element_id
            for _part_id, _node_ids, element_ids in exact_rows
            for element_id in element_ids
        } == mesh_element_ids:
            node_owners = {
                node_id: part_id
                for part_id, node_ids, _element_ids in exact_rows
                for node_id in node_ids
            }
            element_owners = {
                element_id: part_id
                for part_id, _node_ids, element_ids in exact_rows
                for element_id in element_ids
            }
            return (
                node_owners,
                element_owners,
                {
                    part_id: tuple(
                        MeshEntityRef.element(
                            element_id,
                            part_id=part_id,
                        )
                        for element_id in element_ids
                    )
                    for part_id, _node_ids, element_ids in exact_rows
                },
            )

    elements = tuple(sorted(mesh.elements, key=lambda item: int(item.id)))
    groups = _scope_planar_element_groups(
        elements,
        scope_topology,
    )
    if groups is None:
        connected = _connected_mesh_element_groups(mesh, elements, dimension)
        groups = tuple(
            tuple(elements[index] for index in group)
            for group in connected
        )
    element_owners: dict[int, str] = {}
    part_elements: dict[str, tuple[MeshEntityRef, ...]] = {}
    for part_number, group in enumerate(groups, start=1):
        part_id = f"P{part_number}"
        element_ids = tuple(int(element.id) for element in group)
        element_owners.update(
            (element_id, part_id) for element_id in element_ids
        )
        part_elements[part_id] = tuple(
            MeshEntityRef.element(element_id, part_id=part_id)
            for element_id in element_ids
        )
    node_owners: dict[int, str] = {}
    for element in elements:
        owner = element_owners[int(element.id)]
        for raw_node_id in element.node_ids:
            node_id = int(raw_node_id)
            current = node_owners.get(node_id)
            if current is None or part_id_sort_key(owner) < part_id_sort_key(current):
                node_owners[node_id] = owner
    default_owner = next(iter(part_elements), None)
    if default_owner is not None:
        for node in mesh.nodes:
            node_owners.setdefault(int(node.id), default_owner)
    return node_owners, element_owners, part_elements


def _scope_planar_element_groups(
    elements: tuple[Any, ...],
    scope_topology: ScopeSelectionTopology | None,
) -> tuple[tuple[Any, ...], ...] | None:
    """Reuse inferred planar domains as imported Part connectivity groups."""

    if (
        scope_topology is None
        or scope_topology.preview.topological_dimension != 2
    ):
        return None
    element_by_id = {int(element.id): element for element in elements}
    groups = tuple(
        tuple(
            element_by_id[int(reference.element_id)]
            for reference in references
            if reference.kind == "element"
            and reference.element_id is not None
            and int(reference.element_id) in element_by_id
        )
        for logical, references in scope_topology.mesh_references.items()
        if logical.kind == "face"
    )
    groups = tuple(group for group in groups if group)
    grouped_element_ids = tuple(
        int(element.id)
        for group in groups
        for element in group
    )
    if (
        set(grouped_element_ids) != set(element_by_id)
        or len(grouped_element_ids) != len(element_by_id)
    ):
        return None
    return tuple(
        sorted(
            groups,
            key=lambda group: min(int(element.id) for element in group),
        )
    )


def _scope_mesh_topology_groups(
    scope_topology: ScopeSelectionTopology,
) -> tuple[
    tuple[tuple[MeshEntityRef, ...], ...],
    tuple[tuple[MeshEntityRef, ...], ...],
]:
    """Derive mesh edge/face expansion groups from one shared scope scan."""

    return tuple(
        tuple(references)
        for logical, references in scope_topology.mesh_references.items()
        if logical.kind == "edge" and references
    ), tuple(
        tuple(references)
        for logical, references in scope_topology.mesh_references.items()
        if logical.kind == "face" and references
    )


def _connected_mesh_element_groups(
    mesh: Any,
    elements: tuple[Any, ...],
    dimension: int,
) -> tuple[tuple[int, ...], ...]:
    element_index = {
        int(element.id): index for index, element in enumerate(elements)
    }
    groups = _DisjointGroups(len(elements))
    token_owner: dict[tuple[int, ...], int] = {}
    if dimension == 3:
        rows = (
            (
                int(element_id),
                tuple(sorted(_face_corner_node_ids(tuple(int(value) for value in node_ids)))),
            )
            for element_id, _local_index, node_ids in mesh_faces.all(mesh)
        )
    elif dimension == 2:
        rows = (
            (
                int(element_id),
                tuple(sorted((int(node_ids[0]), int(node_ids[-1])))),
            )
            for element_id, _local_index, node_ids in mesh_edges.all(mesh)
            if len(node_ids) >= 2
        )
    else:
        rows = (
            (int(element.id), (int(node_id),))
            for element in elements
            for node_id in element.node_ids
        )
    for element_id, token in rows:
        index = element_index.get(element_id)
        if index is None:
            continue
        previous = token_owner.get(token)
        if previous is None:
            token_owner[token] = index
        else:
            groups.union(previous, index)
    return groups.values()


def _native_mesh_topology_groups(
    catalog: Mapping[Any, Any],
    dimension: int,
    element_owners: Mapping[int, str],
) -> tuple[
    tuple[tuple[MeshEntityRef, ...], ...],
    tuple[tuple[MeshEntityRef, ...], ...],
]:
    edge_groups: list[tuple[MeshEntityRef, ...]] = []
    face_groups: list[tuple[MeshEntityRef, ...]] = []
    for _logical_id, raw in sorted(
        catalog.items(),
        key=lambda item: str(item[0]),
    ):
        if not isinstance(raw, Mapping):
            continue
        kind = str(raw.get("kind", ""))
        if kind == "edge":
            if dimension == 1:
                references = tuple(
                    MeshEntityRef.element(
                        int(element_id),
                        part_id=element_owners.get(int(element_id)),
                    )
                    for element_id in raw.get("element_ids", ())
                )
            else:
                references = _catalog_boundary_references(
                    raw.get("edges", ()),
                    "edge",
                    element_owners,
                )
            if references:
                edge_groups.append(references)
        elif kind == "face":
            if dimension == 2:
                references = tuple(
                    MeshEntityRef.element(
                        int(element_id),
                        part_id=element_owners.get(int(element_id)),
                    )
                    for element_id in raw.get("element_ids", ())
                )
            elif dimension == 3:
                references = _catalog_boundary_references(
                    raw.get("faces", ()),
                    "face",
                    element_owners,
                )
            else:
                references = ()
            if references:
                face_groups.append(references)
    return tuple(edge_groups), tuple(face_groups)


def _catalog_boundary_references(
    rows: Any,
    kind: str,
    element_owners: Mapping[int, str],
) -> tuple[MeshEntityRef, ...]:
    constructor = MeshEntityRef.edge if kind == "edge" else MeshEntityRef.face
    references: list[MeshEntityRef] = []
    for row in rows:
        element_id, local_index, node_ids = tuple(row)
        normalized_element_id = int(element_id)
        references.append(
            constructor(
                normalized_element_id,
                int(local_index),
                tuple(int(value) for value in node_ids),
                part_id=element_owners.get(normalized_element_id),
            )
        )
    return _canonical_mesh_references(references)


def _owned_topology_groups(
    groups: tuple[tuple[MeshEntityRef, ...], ...],
    element_owners: Mapping[int, str],
) -> tuple[tuple[MeshEntityRef, ...], ...]:
    owned: list[tuple[MeshEntityRef, ...]] = []
    for group in groups:
        by_part: dict[str | None, list[MeshEntityRef]] = {}
        for reference in group:
            owner = element_owners.get(int(reference.element_id))
            canonical = (
                reference
                if reference.part_id == owner
                else MeshEntityRef(
                    reference.kind,
                    node_id=reference.node_id,
                    element_id=reference.element_id,
                    local_index=reference.local_index,
                    node_ids=reference.node_ids,
                    part_id=owner,
                )
            )
            by_part.setdefault(owner, []).append(canonical)
        owned.extend(
            _canonical_mesh_references(references)
            for _part_id, references in sorted(
                by_part.items(),
                key=lambda item: (
                    -1 if item[0] is None else part_id_sort_key(item[0])
                ),
            )
            if references
        )
    return tuple(
        sorted(
            owned,
            key=lambda group: mesh_entity_ref_sort_key(group[0]),
        )
    )


def _topology_expansion_index(
    groups: tuple[tuple[MeshEntityRef, ...], ...],
) -> dict[
    tuple[str, tuple[int, int]],
    tuple[MeshEntityRef, ...],
]:
    return {
        (reference.kind, reference.identity): group
        for group in groups
        for reference in group
    }


def _canonical_mesh_references(
    references: Any,
) -> tuple[MeshEntityRef, ...]:
    return tuple(sorted(set(references), key=mesh_entity_ref_sort_key))


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
    "MeshSelectionTopology",
    "ScopeSelectionTopology",
    "build_mesh_selection_topology",
    "build_scope_selection_topology",
]
