from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
import operator
from typing import Any, Literal

from fem.core import (
    Edge,
    Element2D,
    Element3D,
    ElementEdge,
    ElementFace,
    ElementSet,
    FEMModel,
    Mesh2D,
    Mesh3D,
    Node2D,
    Node3D,
    NodeSet,
    Surface,
)
from fem.selection import edges as edge_selection
from fem.selection import faces as face_selection


@dataclass(frozen=True)
class _ElementSpec:
    gmsh_type: int
    gmsh_name: str
    dimension: Literal[1, 2, 3]
    order: int
    node_count: int
    primary_node_count: int
    fem_type: str | None
    connectivity_permutation: tuple[int, ...]


@dataclass(frozen=True)
class _ElementRecord:
    element_id: int
    spec: _ElementSpec
    node_ids: tuple[int, ...]


@dataclass(frozen=True)
class _BoundaryElementSpec:
    gmsh_type: int
    gmsh_name: str
    dimension: Literal[1, 2]
    order: int
    node_count: int
    primary_node_count: int
    shape: Literal["line", "triangle", "quadrilateral"]


@dataclass(frozen=True)
class _BoundaryElementRecord:
    element_tag: int
    spec: _BoundaryElementSpec
    node_ids: tuple[int, ...]


@dataclass(frozen=True)
class _BoundaryOwner:
    elem_id: int
    local_index: int
    node_ids: tuple[int, ...]


_BoundaryKey = tuple[str, tuple[int, ...]]
_BoundaryOwnerLookup = dict[_BoundaryKey, list[_BoundaryOwner]]


def _validated_element_specs(specs: tuple[_ElementSpec, ...]) -> dict[int, _ElementSpec]:
    validated: dict[int, _ElementSpec] = {}
    for spec in specs:
        if spec.gmsh_type in validated:
            raise RuntimeError(f"duplicate Gmsh element specification {spec.gmsh_type}")
        if sorted(spec.connectivity_permutation) != list(range(spec.node_count)):
            raise RuntimeError(
                f"invalid connectivity permutation for Gmsh element type "
                f"{spec.gmsh_type}: {spec.connectivity_permutation!r}"
            )
        validated[spec.gmsh_type] = spec
    return validated


_ELEMENT_SPECS = _validated_element_specs(
    (
        _ElementSpec(1, "Line 2", 1, 1, 2, 2, None, (0, 1)),
        _ElementSpec(2, "Triangle 3", 2, 1, 3, 3, "Tri3", (0, 1, 2)),
        _ElementSpec(
            3,
            "Quadrilateral 4",
            2,
            1,
            4,
            4,
            "Quad4",
            (0, 1, 2, 3),
        ),
        _ElementSpec(4, "Tetrahedron 4", 3, 1, 4, 4, "Tet4", (0, 1, 2, 3)),
        _ElementSpec(
            5,
            "Hexahedron 8",
            3,
            1,
            8,
            8,
            "Hex8",
            (0, 1, 2, 3, 4, 5, 6, 7),
        ),
        _ElementSpec(9, "Triangle 6", 2, 2, 6, 3, "Tri6", (0, 1, 2, 3, 4, 5)),
        _ElementSpec(
            16,
            "Quadrilateral 8",
            2,
            2,
            8,
            4,
            "Quad8",
            (0, 1, 2, 3, 4, 5, 6, 7),
        ),
        _ElementSpec(
            11,
            "Tetrahedron 10",
            3,
            2,
            10,
            4,
            "Tet10",
            (0, 1, 2, 3, 4, 5, 6, 7, 9, 8),
        ),
        _ElementSpec(
            17,
            "Hexahedron 20",
            3,
            2,
            20,
            8,
            "Hex20",
            (
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                11,
                13,
                9,
                16,
                18,
                19,
                17,
                10,
                12,
                14,
                15,
            ),
        ),
    )
)


def _validated_boundary_element_specs(
    specs: tuple[_BoundaryElementSpec, ...],
) -> dict[int, _BoundaryElementSpec]:
    validated: dict[int, _BoundaryElementSpec] = {}
    shape_contract = {
        "line": (1, 2),
        "triangle": (2, 3),
        "quadrilateral": (2, 4),
    }
    for spec in specs:
        if spec.gmsh_type in validated:
            raise RuntimeError(
                f"duplicate Gmsh boundary element specification {spec.gmsh_type}"
            )
        expected_dimension, expected_primary_nodes = shape_contract[spec.shape]
        if (
            spec.dimension != expected_dimension
            or spec.primary_node_count != expected_primary_nodes
            or spec.node_count < spec.primary_node_count
            or spec.order <= 0
        ):
            raise RuntimeError(
                f"invalid Gmsh boundary element specification {spec.gmsh_type}: "
                f"{spec!r}"
            )
        validated[spec.gmsh_type] = spec
    return validated


_BOUNDARY_ELEMENT_SPECS = _validated_boundary_element_specs(
    (
        _BoundaryElementSpec(1, "Line 2", 1, 1, 2, 2, "line"),
        _BoundaryElementSpec(8, "Line 3", 1, 2, 3, 2, "line"),
        _BoundaryElementSpec(2, "Triangle 3", 2, 1, 3, 3, "triangle"),
        _BoundaryElementSpec(9, "Triangle 6", 2, 2, 6, 3, "triangle"),
        _BoundaryElementSpec(
            3,
            "Quadrilateral 4",
            2,
            1,
            4,
            4,
            "quadrilateral",
        ),
        _BoundaryElementSpec(
            16,
            "Quadrilateral 8",
            2,
            2,
            8,
            4,
            "quadrilateral",
        ),
    )
)

_CLOCKWISE_TO_COUNTERCLOCKWISE = {
    "Tri3": (0, 2, 1),
    "Tri6": (0, 2, 1, 5, 4, 3),
    "Quad4": (0, 3, 2, 1),
    "Quad8": (0, 3, 2, 1, 7, 6, 5, 4),
}

_INCOMPLETE_SECOND_ORDER_ALTERNATIVES = {
    10: ("Quadrilateral 9", "Quad8"),
    12: ("Hexahedron 27", "Hex20"),
}

_PhysicalGroupRecord = dict[str, Any]
_PhysicalGroupMetadata = dict[
    str,
    _PhysicalGroupRecord | tuple[_PhysicalGroupRecord, ...],
]


@dataclass(frozen=True)
class GmshImportResult:
    """FEM data imported from the caller-owned active Gmsh model."""

    mesh: Mesh2D | Mesh3D
    node_sets: dict[str, NodeSet]
    element_sets: dict[str, ElementSet]
    metadata: dict[str, Any]
    edges: dict[str, Edge] = field(default_factory=dict)
    surfaces: dict[str, Surface] = field(default_factory=dict)

    def to_fem_model(self, name: str | None = None) -> FEMModel:
        """Return an analysis model without inventing material or load semantics."""
        return FEMModel(
            mesh=self.mesh,
            name=name,
            node_sets=dict(self.node_sets),
            element_sets=dict(self.element_sets),
            edges=dict(self.edges),
            surfaces=dict(self.surfaces),
            metadata=deepcopy(self.metadata),
        )


def from_model(
    *,
    dimension: Literal[1, 2, 3],
    gmsh_model: Any | None = None,
    line_element_type: Literal["Truss2", "Beam2"] | None = None,
    plane_type: str = "stress",
    thickness: float = 1.0,
    z_tolerance: float = 1e-10,
) -> GmshImportResult:
    """Import the generated mesh from a caller-owned active Gmsh model."""
    normalized_dimension = _validate_dimension(dimension)
    normalized_line_element_type = _validate_line_element_type(
        normalized_dimension,
        line_element_type,
    )
    normalized_plane_type = _validate_plane_type(plane_type)
    normalized_thickness = _validate_thickness(thickness)
    normalized_z_tolerance = _validate_z_tolerance(z_tolerance)

    gmsh_version = None
    if gmsh_model is None:
        gmsh_model, gmsh_version = _resolve_live_backend()

    return _from_backend(
        gmsh_model,
        dimension=normalized_dimension,
        line_element_type=normalized_line_element_type,
        plane_type=normalized_plane_type,
        thickness=normalized_thickness,
        z_tolerance=normalized_z_tolerance,
        gmsh_version=gmsh_version,
    )


def _resolve_live_backend() -> tuple[Any, str | None]:
    try:
        import gmsh
    except ModuleNotFoundError as exc:
        if exc.name != "gmsh":
            raise
        raise ModuleNotFoundError(
            "Gmsh is an optional dependency; install it with "
            "`python -m pip install -e .[cad]`."
        ) from exc

    if not gmsh.isInitialized():
        raise RuntimeError(
            "Gmsh is not initialized; the caller must call gmsh.initialize() "
            "before fem.io.gmsh.from_model()."
        )

    version = getattr(gmsh, "__version__", None)
    return gmsh.model, None if version is None else str(version)


def _validate_dimension(value: Any) -> Literal[1, 2, 3]:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 3):
        raise ValueError(f"dimension must be 1, 2, or 3, got {value!r}")
    return value


def _validate_line_element_type(
    dimension: Literal[1, 2, 3],
    value: Any,
) -> Literal["Truss2", "Beam2"] | None:
    if dimension == 1:
        if value not in ("Truss2", "Beam2"):
            raise ValueError(
                "line_element_type must be exactly 'Truss2' or 'Beam2' for "
                f"dimension 1, got {value!r}"
            )
        return value
    if value is not None:
        raise ValueError(
            "line_element_type is only valid for dimension 1, "
            f"got {value!r} for dimension {dimension}"
        )
    return None


def _validate_plane_type(value: Any) -> str:
    if not isinstance(value, str) or value.lower() not in {"stress", "strain"}:
        raise ValueError(
            f"plane_type must be 'stress' or 'strain', got {value!r}"
        )
    return value.lower()


def _validate_thickness(value: Any) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"thickness must be finite and > 0, got {value!r}") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"thickness must be finite and > 0, got {value!r}")
    return normalized


def _validate_z_tolerance(value: Any) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"z_tolerance must be finite and >= 0, got {value!r}"
        ) from exc
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"z_tolerance must be finite and >= 0, got {value!r}")
    return normalized


def _from_backend(
    gmsh_model: Any,
    *,
    dimension: Literal[1, 2, 3],
    line_element_type: Literal["Truss2", "Beam2"] | None,
    plane_type: str,
    thickness: float,
    z_tolerance: float,
    gmsh_version: str | None,
) -> GmshImportResult:
    records = _read_element_records(gmsh_model.mesh, dimension)
    referenced_node_ids = {
        node_id for record in records for node_id in record.node_ids
    }
    nodes, coordinates = _read_nodes(
        gmsh_model.mesh,
        referenced_node_ids,
        dimension=dimension,
        z_tolerance=z_tolerance,
    )
    elements = _build_elements(
        records,
        coordinates,
        dimension=dimension,
        line_element_type=line_element_type,
        plane_type=plane_type,
        thickness=thickness,
    )
    if dimension == 2:
        mesh = Mesh2D(nodes=nodes, elements=elements, dofs_per_node=2)
    elif dimension == 1:
        dofs_per_node = 3 if line_element_type == "Truss2" else 6
        mesh = Mesh3D(nodes=nodes, elements=elements, dofs_per_node=dofs_per_node)
    else:
        mesh = Mesh3D(nodes=nodes, elements=elements, dofs_per_node=3)

    node_sets, element_sets, edges, surfaces, physical_groups, skipped_groups = (
        _read_physical_groups(
            gmsh_model,
            mesh=mesh,
            dimension=dimension,
            retained_node_ids={node.id for node in nodes},
            retained_element_ids={element.id for element in elements},
        )
    )

    metadata: dict[str, Any] = {
        "source": "gmsh",
        "dimension": dimension,
        "physical_groups": physical_groups,
        "skipped_physical_groups": skipped_groups,
    }
    if dimension == 1:
        metadata["line_element_type"] = line_element_type
    if gmsh_version is not None:
        metadata["gmsh_version"] = gmsh_version
    return GmshImportResult(
        mesh=mesh,
        node_sets=node_sets,
        element_sets=element_sets,
        metadata=metadata,
        edges=edges,
        surfaces=surfaces,
    )


def _read_physical_groups(
    gmsh_model: Any,
    *,
    mesh: Mesh2D | Mesh3D,
    dimension: Literal[1, 2, 3],
    retained_node_ids: set[int],
    retained_element_ids: set[int],
) -> tuple[
    dict[str, NodeSet],
    dict[str, ElementSet],
    dict[str, Edge],
    dict[str, Surface],
    _PhysicalGroupMetadata,
    tuple[dict[str, Any], ...],
]:
    node_sets: dict[str, NodeSet] = {}
    element_sets: dict[str, ElementSet] = {}
    edges: dict[str, Edge] = {}
    surfaces: dict[str, Surface] = {}
    metadata_groups: _PhysicalGroupMetadata = {}
    skipped_groups: list[dict[str, Any]] = []
    seen_names = {
        "node_set": set(),
        "element_set": set(),
        "edge": set(),
        "surface": set(),
    }
    edge_owner_lookup: _BoundaryOwnerLookup | None = None
    face_owner_lookup: _BoundaryOwnerLookup | None = None

    for raw_group in list(gmsh_model.getPhysicalGroups()):
        try:
            raw_group_dimension, raw_group_tag = raw_group
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid Gmsh physical group descriptor {raw_group!r}"
            ) from exc
        group_dimension = _integer_value(
            raw_group_dimension,
            "physical-group dimensions must be integers",
        )
        group_tag = _positive_tag(raw_group_tag, "physical-group tags")
        if group_dimension < 0:
            raise ValueError("physical-group dimensions must be nonnegative integers")
        if group_dimension > dimension:
            continue

        kind = "element_set" if group_dimension == dimension else "node_set"
        raw_name = gmsh_model.getPhysicalName(group_dimension, group_tag)
        name = str(raw_name) if raw_name else f"physical_{group_dimension}_{group_tag}"
        if name in seen_names[kind]:
            namespace = "element-set" if kind == "element_set" else "node-set"
            raise ValueError(
                f"duplicate physical-group name {name!r} in {namespace} namespace"
            )
        seen_names[kind].add(name)

        boundary_kind = None
        if group_dimension == dimension - 1 and dimension in (2, 3):
            boundary_kind = "edge" if dimension == 2 else "surface"
        if boundary_kind is not None:
            if name in seen_names[boundary_kind]:
                raise ValueError(
                    f"duplicate physical-group name {name!r} in "
                    f"{boundary_kind} namespace"
                )
            seen_names[boundary_kind].add(name)

        record = {
            "dimension": group_dimension,
            "tag": group_tag,
            "kind": kind,
        }
        if kind == "node_set":
            member_ids = _physical_node_ids(
                gmsh_model.mesh,
                group_dimension,
                group_tag,
                retained_node_ids,
            )
        else:
            member_ids = _physical_element_ids(
                gmsh_model,
                group_dimension,
                group_tag,
                retained_element_ids,
            )

        if not member_ids:
            skipped_groups.append({"name": name, **record})
            continue
        if kind == "node_set":
            node_sets[name] = NodeSet(name, member_ids)
        else:
            element_sets[name] = ElementSet(name, member_ids)

        if boundary_kind is not None:
            boundary_records = _read_boundary_element_records(
                gmsh_model,
                group_name=name,
                group_dimension=group_dimension,
                group_tag=group_tag,
            )
            retained_records = _retained_boundary_element_records(
                boundary_records,
                retained_node_ids,
                group_name=name,
                group_dimension=group_dimension,
                group_tag=group_tag,
            )
            if not retained_records:
                context = _physical_group_context(
                    name,
                    group_dimension,
                    group_tag,
                )
                raise ValueError(
                    f"{context} has retained nodes but no retained boundary elements"
                )
            if boundary_kind == "edge":
                if edge_owner_lookup is None:
                    edge_owner_lookup = _build_edge_owner_lookup(mesh)
                owner_lookup = edge_owner_lookup
            else:
                if face_owner_lookup is None:
                    face_owner_lookup = _build_face_owner_lookup(mesh)
                owner_lookup = face_owner_lookup
            matched_entries = _match_boundary_elements(
                retained_records,
                owner_lookup,
                group_name=name,
                group_dimension=group_dimension,
                group_tag=group_tag,
                boundary_kind=boundary_kind,
            )
            if boundary_kind == "edge":
                edges[name] = Edge(name, matched_entries)
            else:
                surfaces[name] = Surface(name, matched_entries)
            record.update(
                {
                    "boundary_kind": boundary_kind,
                    "boundary_entry_count": len(matched_entries),
                }
            )
        _store_physical_group_metadata(metadata_groups, name, record)

    return (
        node_sets,
        element_sets,
        edges,
        surfaces,
        metadata_groups,
        tuple(skipped_groups),
    )


def _physical_node_ids(
    gmsh_mesh: Any,
    group_dimension: int,
    group_tag: int,
    retained_node_ids: set[int],
) -> list[int]:
    raw_membership = gmsh_mesh.getNodesForPhysicalGroup(group_dimension, group_tag)
    try:
        raw_node_ids = raw_membership[0]
    except (TypeError, IndexError) as exc:
        raise ValueError(
            f"getNodesForPhysicalGroup({group_dimension}, {group_tag}) "
            "returned malformed data"
        ) from exc
    node_ids = {
        _positive_tag(value, "physical-group node tags")
        for value in list(raw_node_ids)
    }
    return sorted(node_ids.intersection(retained_node_ids))


def _physical_group_context(name: str, dimension: int, tag: int) -> str:
    return f"physical group {name!r} (dimension={dimension}, tag={tag})"


def _entity_element_blocks(
    gmsh_mesh: Any,
    dimension: int,
    entity_tag: int,
    *,
    block_label: str,
    context: str | None = None,
) -> tuple[list[Any], list[Any], list[Any]]:
    raw_blocks = gmsh_mesh.getElements(dimension, entity_tag)
    call = f"getElements({dimension}, {entity_tag})"
    if context is not None:
        call += f" for {context}"
    try:
        raw_types, raw_element_tags, raw_connectivity = raw_blocks
        element_types = list(raw_types)
        element_tag_blocks = list(raw_element_tags)
        connectivity_blocks = list(raw_connectivity)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{call} returned malformed data") from exc
    if (
        len(element_tag_blocks) != len(element_types)
        or len(connectivity_blocks) != len(element_types)
    ):
        raise ValueError(
            f"{call} returned inconsistent {block_label} element block counts"
        )
    return element_types, element_tag_blocks, connectivity_blocks


def _read_boundary_element_records(
    gmsh_model: Any,
    *,
    group_name: str,
    group_dimension: int,
    group_tag: int,
) -> list[_BoundaryElementRecord]:
    context = _physical_group_context(group_name, group_dimension, group_tag)
    records: list[_BoundaryElementRecord] = []
    seen_element_tags: set[int] = set()
    raw_entity_tags = gmsh_model.getEntitiesForPhysicalGroup(
        group_dimension,
        group_tag,
    )
    try:
        entity_tags = list(raw_entity_tags)
    except TypeError as exc:
        raise ValueError(f"{context} entity tags returned malformed data") from exc
    for raw_entity_tag in entity_tags:
        entity_tag = _positive_tag(
            raw_entity_tag,
            f"{context} entity tags",
        )
        element_types, element_tag_blocks, connectivity_blocks = (
            _entity_element_blocks(
                gmsh_model.mesh,
                group_dimension,
                entity_tag,
                block_label="boundary",
                context=context,
            )
        )
        for block_index, (raw_type, raw_tags, raw_nodes) in enumerate(
            zip(
                element_types,
                element_tag_blocks,
                connectivity_blocks,
                strict=True,
            )
        ):
            element_type = _integer_value(
                raw_type,
                f"{context} boundary element types must be integers",
            )
            try:
                raw_element_tag_values = list(raw_tags)
            except TypeError as exc:
                raise ValueError(
                    f"{context} block {block_index} has a malformed boundary "
                    "element-tag block"
                ) from exc
            element_tags = [
                _positive_tag(value, f"{context} boundary element tags")
                for value in raw_element_tag_values
            ]
            reported = _reported_element_properties(gmsh_model.mesh, element_type)
            spec = _BOUNDARY_ELEMENT_SPECS.get(element_type)
            if spec is None or spec.dimension != group_dimension:
                (
                    gmsh_name,
                    reported_dimension,
                    order,
                    node_count,
                    primary_node_count,
                ) = reported
                raise ValueError(
                    f"{context} entity {entity_tag} boundary element tags "
                    f"{element_tags!r} use unsupported codimension-one Gmsh "
                    f"element type {element_type} ({gmsh_name!r}, "
                    f"dimension={reported_dimension}, order={order}, "
                    f"nodes={node_count}, primary_nodes={primary_node_count})"
                )
            expected = (
                spec.gmsh_name,
                spec.dimension,
                spec.order,
                spec.node_count,
                spec.primary_node_count,
            )
            if reported != expected:
                raise ValueError(
                    f"{context} boundary element tags {element_tags!r} type "
                    f"{element_type} properties do not match the adapter contract: "
                    f"reported {reported!r}, expected {expected!r}"
                )

            try:
                raw_node_values = list(raw_nodes)
            except TypeError as exc:
                raise ValueError(
                    f"{context} block {block_index} has a malformed boundary "
                    "connectivity block"
                ) from exc
            flat_node_ids = [
                _positive_tag(value, f"{context} boundary node tags")
                for value in raw_node_values
            ]
            expected_connectivity_count = len(element_tags) * spec.node_count
            if len(flat_node_ids) != expected_connectivity_count:
                raise ValueError(
                    f"{context} boundary block {block_index} type {element_type} "
                    f"flattened connectivity has length {len(flat_node_ids)}; "
                    f"expected {expected_connectivity_count} for "
                    f"{len(element_tags)} elements with {spec.node_count} nodes each"
                )

            for local_index, element_tag in enumerate(element_tags):
                if element_tag in seen_element_tags:
                    raise ValueError(
                        f"{context} has duplicate boundary element tag {element_tag}"
                    )
                seen_element_tags.add(element_tag)
                start = local_index * spec.node_count
                node_ids = tuple(
                    flat_node_ids[start : start + spec.node_count]
                )
                if len(set(node_ids)) != len(node_ids):
                    raise ValueError(
                        f"{context} Gmsh boundary element {element_tag} type "
                        f"{element_type} has repeated node tags {node_ids!r}"
                    )
                records.append(_BoundaryElementRecord(element_tag, spec, node_ids))
    return records


def _retained_boundary_element_records(
    records: list[_BoundaryElementRecord],
    retained_node_ids: set[int],
    *,
    group_name: str,
    group_dimension: int,
    group_tag: int,
) -> list[_BoundaryElementRecord]:
    context = _physical_group_context(group_name, group_dimension, group_tag)
    retained_records: list[_BoundaryElementRecord] = []
    for record in records:
        retained_ids = set(record.node_ids).intersection(retained_node_ids)
        if not retained_ids:
            continue
        if len(retained_ids) != len(record.node_ids):
            raise ValueError(
                f"{context} Gmsh boundary element {record.element_tag} type "
                f"{record.spec.gmsh_type} has partial retained-node membership: "
                f"retained {sorted(retained_ids)!r} from source "
                f"{sorted(record.node_ids)!r}"
            )
        retained_records.append(record)
    return retained_records


def _build_edge_owner_lookup(mesh: Mesh2D | Mesh3D) -> _BoundaryOwnerLookup:
    lookup: _BoundaryOwnerLookup = {}
    for elem_id, local_index, raw_node_ids in edge_selection.all(mesh):
        node_ids = tuple(int(node_id) for node_id in raw_node_ids)
        if len(node_ids) < 2:
            raise ValueError(
                f"FEM edge owner ({elem_id}, {local_index}) has fewer than two nodes"
            )
        key = ("line", tuple(sorted((node_ids[0], node_ids[-1]))))
        lookup.setdefault(key, []).append(
            _BoundaryOwner(int(elem_id), int(local_index), node_ids)
        )
    return lookup


def _build_face_owner_lookup(mesh: Mesh2D | Mesh3D) -> _BoundaryOwnerLookup:
    lookup: _BoundaryOwnerLookup = {}
    elements_by_id = {int(element.id): element for element in mesh.elements}
    face_contract = {
        "tet4": ("triangle", 3),
        "tet10": ("triangle", 3),
        "hex8": ("quadrilateral", 4),
        "hex20": ("quadrilateral", 4),
    }
    for elem_id, local_index, raw_node_ids in face_selection.all(mesh):
        normalized_elem_id = int(elem_id)
        element = elements_by_id[normalized_elem_id]
        shape, corner_count = face_contract[str(element.type).casefold()]
        node_ids = tuple(int(node_id) for node_id in raw_node_ids)
        if len(node_ids) < corner_count:
            raise ValueError(
                f"FEM face owner ({elem_id}, {local_index}) has fewer than "
                f"{corner_count} corner nodes"
            )
        key = (shape, tuple(sorted(node_ids[:corner_count])))
        lookup.setdefault(key, []).append(
            _BoundaryOwner(normalized_elem_id, int(local_index), node_ids)
        )
    return lookup


def _match_boundary_elements(
    records: list[_BoundaryElementRecord],
    owner_lookup: _BoundaryOwnerLookup,
    *,
    group_name: str,
    group_dimension: int,
    group_tag: int,
    boundary_kind: Literal["edge", "surface"],
) -> list[ElementEdge] | list[ElementFace]:
    context = _physical_group_context(group_name, group_dimension, group_tag)
    entries: list[ElementEdge] | list[ElementFace] = []
    seen_owner_pairs: set[tuple[int, int]] = set()
    for record in records:
        primary_corner_ids = record.node_ids[: record.spec.primary_node_count]
        key = (record.spec.shape, tuple(sorted(primary_corner_ids)))
        candidates = owner_lookup.get(key, [])
        if not candidates:
            raise ValueError(
                f"{context} Gmsh boundary element {record.element_tag} type "
                f"{record.spec.gmsh_type} has no FEM owner for key {key!r}"
            )
        if len(candidates) > 1:
            candidate_pairs = [
                (candidate.elem_id, candidate.local_index)
                for candidate in candidates
            ]
            raise ValueError(
                f"{context} Gmsh boundary element {record.element_tag} type "
                f"{record.spec.gmsh_type} has multiple FEM owners "
                f"{candidate_pairs!r}; internal interfaces and nonmanifold "
                "boundaries are unsupported"
            )

        owner = candidates[0]
        if set(record.node_ids) != set(owner.node_ids):
            raise ValueError(
                f"{context} Gmsh boundary element {record.element_tag} type "
                f"{record.spec.gmsh_type} full node set "
                f"{sorted(record.node_ids)!r} does not match FEM owner "
                f"({owner.elem_id}, {owner.local_index}) full node set "
                f"{sorted(owner.node_ids)!r}"
            )
        owner_pair = (owner.elem_id, owner.local_index)
        if owner_pair in seen_owner_pairs:
            raise ValueError(
                f"{context} Gmsh boundary element {record.element_tag} type "
                f"{record.spec.gmsh_type} has duplicate FEM owner "
                f"{owner_pair!r} match"
            )
        seen_owner_pairs.add(owner_pair)
        if boundary_kind == "edge":
            entries.append(
                ElementEdge(owner.elem_id, owner.local_index, owner.node_ids)
            )
        else:
            entries.append(
                ElementFace(owner.elem_id, owner.local_index, owner.node_ids)
            )
    entries.sort(key=lambda entry: (entry.elem_id, entry.local_index))
    return entries


def _physical_element_ids(
    gmsh_model: Any,
    group_dimension: int,
    group_tag: int,
    retained_element_ids: set[int],
) -> list[int]:
    member_ids: set[int] = set()
    raw_entity_tags = gmsh_model.getEntitiesForPhysicalGroup(
        group_dimension,
        group_tag,
    )
    for raw_entity_tag in list(raw_entity_tags):
        entity_tag = _positive_tag(raw_entity_tag, "physical-group entity tags")
        _, element_tag_blocks, _ = _entity_element_blocks(
            gmsh_model.mesh,
            group_dimension,
            entity_tag,
            block_label="physical-group",
        )
        for raw_ids in element_tag_blocks:
            member_ids.update(
                _positive_tag(value, "physical-group element tags")
                for value in list(raw_ids)
            )
    return sorted(member_ids.intersection(retained_element_ids))


def _store_physical_group_metadata(
    metadata_groups: _PhysicalGroupMetadata,
    name: str,
    record: _PhysicalGroupRecord,
) -> None:
    if name not in metadata_groups:
        metadata_groups[name] = record
        return

    existing = metadata_groups[name]
    if isinstance(existing, tuple):
        metadata_groups[name] = (*existing, record)
    else:
        metadata_groups[name] = (existing, record)


def _read_element_records(
    gmsh_mesh: Any,
    dimension: Literal[1, 2, 3],
) -> list[_ElementRecord]:
    raw_blocks = gmsh_mesh.getElements(dimension, -1)
    try:
        raw_types, raw_element_tags, raw_connectivity = raw_blocks
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "getElements() must return element types, element-tag blocks, "
            "and connectivity blocks"
        ) from exc

    element_types = list(raw_types)
    element_tag_blocks = list(raw_element_tags)
    connectivity_blocks = list(raw_connectivity)
    block_count = len(element_types)
    if len(element_tag_blocks) != block_count or len(connectivity_blocks) != block_count:
        raise ValueError(
            "getElements() returned inconsistent element block counts: "
            f"{block_count} types, {len(element_tag_blocks)} element-tag blocks, "
            f"and {len(connectivity_blocks)} connectivity blocks"
        )

    records: list[_ElementRecord] = []
    seen_element_ids: set[int] = set()
    for block_index, (raw_type, raw_tags, raw_nodes) in enumerate(
        zip(element_types, element_tag_blocks, connectivity_blocks, strict=True)
    ):
        element_type = _integer_value(raw_type, "Gmsh element types must be integers")
        reported = _reported_element_properties(gmsh_mesh, element_type)
        spec = _ELEMENT_SPECS.get(element_type)
        if spec is None:
            name, reported_dimension, order, node_count, primary_node_count = reported
            message = (
                f"unsupported Gmsh element type {element_type} ({name!r}, "
                f"dimension={reported_dimension}, order={order}, nodes={node_count}, "
                f"primary_nodes={primary_node_count})"
            )
            alternative = _INCOMPLETE_SECOND_ORDER_ALTERNATIVES.get(element_type)
            if alternative is not None and name == alternative[0]:
                message += (
                    f". {name} is unsupported because FEM-Python provides "
                    f"{alternative[1]}. Generate an incomplete second-order mesh "
                    f"with Mesh.SecondOrderIncomplete = 1."
                )
            if dimension == 1 and element_type == 8 and name == "Line 3":
                message += (
                    ". FEM-Python's current Truss2 and Beam2 formulations "
                    "support only first-order, two-node line elements."
                )
            raise ValueError(message)
        expected = (
            spec.gmsh_name,
            spec.dimension,
            spec.order,
            spec.node_count,
            spec.primary_node_count,
        )
        if reported != expected:
            raise ValueError(
                f"Gmsh element type {element_type} properties do not match the "
                f"adapter contract: reported {reported!r}, expected {expected!r}"
            )
        if spec.dimension != dimension:
            raise ValueError(
                f"Gmsh element type {element_type} has dimension {spec.dimension}, "
                f"but dimension {dimension} was requested"
            )

        element_ids = [
            _positive_tag(value, "element tags") for value in list(raw_tags)
        ]
        flat_node_ids = [
            _positive_tag(value, "connectivity node tags")
            for value in list(raw_nodes)
        ]
        expected_connectivity_count = len(element_ids) * spec.node_count
        if len(flat_node_ids) != expected_connectivity_count:
            raise ValueError(
                f"Gmsh element block {block_index} type {element_type} flattened "
                f"connectivity has length {len(flat_node_ids)}; expected "
                f"{expected_connectivity_count} for {len(element_ids)} elements "
                f"with {spec.node_count} nodes each"
            )

        for local_element_index, element_id in enumerate(element_ids):
            if element_id in seen_element_ids:
                raise ValueError(f"duplicate element tag {element_id} across blocks")
            seen_element_ids.add(element_id)
            start = local_element_index * spec.node_count
            raw_node_ids = flat_node_ids[start : start + spec.node_count]
            node_ids = tuple(
                raw_node_ids[index] for index in spec.connectivity_permutation
            )
            if len(set(node_ids)) != len(node_ids):
                raise ValueError(
                    f"Gmsh element {element_id} type {element_type} has repeated "
                    f"node tags {node_ids!r}"
                )
            records.append(_ElementRecord(element_id, spec, node_ids))

    if not records:
        raise ValueError(
            f"Gmsh model contains no top-dimensional elements for dimension {dimension}"
        )
    return records


def _reported_element_properties(
    gmsh_mesh: Any,
    element_type: int,
) -> tuple[str, int, int, int, int]:
    properties = gmsh_mesh.getElementProperties(element_type)
    try:
        name, dimension, order, node_count = properties[:4]
        primary_node_count = properties[5]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            f"getElementProperties({element_type}) returned malformed data"
        ) from exc
    return (
        str(name),
        _integer_value(dimension, "reported element dimension must be an integer"),
        _integer_value(order, "reported element order must be an integer"),
        _integer_value(node_count, "reported element node count must be an integer"),
        _integer_value(
            primary_node_count,
            "reported primary node count must be an integer",
        ),
    )


def _read_nodes(
    gmsh_mesh: Any,
    referenced_node_ids: set[int],
    *,
    dimension: Literal[1, 2, 3],
    z_tolerance: float,
) -> tuple[list[Node2D] | list[Node3D], dict[int, tuple[float, float, float]]]:
    raw_nodes = gmsh_mesh.getNodes()
    try:
        raw_node_tags, raw_coordinates = raw_nodes[:2]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "getNodes() must return node tags and a flattened coordinate vector"
        ) from exc

    node_ids: list[int] = []
    seen_node_ids: set[int] = set()
    for raw_node_id in list(raw_node_tags):
        node_id = _positive_tag(raw_node_id, "node tags")
        if node_id in seen_node_ids:
            raise ValueError(f"duplicate node tag {node_id}")
        seen_node_ids.add(node_id)
        node_ids.append(node_id)

    raw_coordinate_values = list(raw_coordinates)
    expected_coordinate_count = 3 * len(node_ids)
    if len(raw_coordinate_values) != expected_coordinate_count:
        raise ValueError(
            f"Gmsh node coordinate vector length is {len(raw_coordinate_values)}; "
            f"expected {expected_coordinate_count} for {len(node_ids)} node tags"
        )
    coordinates: list[float] = []
    for value in raw_coordinate_values:
        try:
            coordinate = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Gmsh node coordinates must be finite numbers") from exc
        if not math.isfinite(coordinate):
            raise ValueError("Gmsh node coordinates must be finite numbers")
        coordinates.append(coordinate)

    coordinate_by_node = {
        node_id: tuple(coordinates[index : index + 3])
        for index, node_id in zip(range(0, len(coordinates), 3), node_ids, strict=True)
    }
    missing_node_ids = sorted(referenced_node_ids.difference(coordinate_by_node))
    if missing_node_ids:
        raise ValueError(f"element references missing node tag {missing_node_ids[0]}")

    retained_nodes: list[Node2D] | list[Node3D] = []
    for node_id in node_ids:
        if node_id not in referenced_node_ids:
            continue
        x, y, z = coordinate_by_node[node_id]
        if dimension == 2:
            if abs(z) > z_tolerance:
                raise ValueError(
                    f"Gmsh node {node_id} has z={z!r} outside the XY-plane "
                    f"tolerance {z_tolerance!r}"
                )
            retained_nodes.append(Node2D(node_id, x, y))
        else:
            retained_nodes.append(Node3D(node_id, x, y, z))
    return retained_nodes, coordinate_by_node


def _build_elements(
    records: list[_ElementRecord],
    coordinates: dict[int, tuple[float, float, float]],
    *,
    dimension: Literal[1, 2, 3],
    line_element_type: Literal["Truss2", "Beam2"] | None,
    plane_type: str,
    thickness: float,
) -> list[Element2D] | list[Element3D]:
    if dimension in (1, 3):
        fem_types = [
            line_element_type if dimension == 1 else record.spec.fem_type
            for record in records
        ]
        if any(fem_type is None for fem_type in fem_types):
            raise RuntimeError("Gmsh element specification has no FEM element type")
        return [
            Element3D(
                record.element_id,
                list(record.node_ids),
                fem_type,
                {},
            )
            for record, fem_type in zip(records, fem_types, strict=True)
        ]

    elements_2d: list[Element2D] = []
    props = {"plane_type": plane_type, "thickness": thickness}
    for record in records:
        node_ids = list(record.node_ids)
        corner_node_ids = node_ids[: record.spec.primary_node_count]
        if _signed_twice_area(corner_node_ids, coordinates) < 0.0:
            permutation = _CLOCKWISE_TO_COUNTERCLOCKWISE[record.spec.fem_type]
            node_ids = [node_ids[index] for index in permutation]
        elements_2d.append(
            Element2D(
                record.element_id,
                node_ids,
                record.spec.fem_type,
                dict(props),
            )
        )
    return elements_2d


def _signed_twice_area(
    node_ids: list[int],
    coordinates: dict[int, tuple[float, float, float]],
) -> float:
    area = 0.0
    for index, node_id in enumerate(node_ids):
        next_node_id = node_ids[(index + 1) % len(node_ids)]
        x, y, _ = coordinates[node_id]
        next_x, next_y, _ = coordinates[next_node_id]
        area += x * next_y - next_x * y
    return area


def _integer_value(value: Any, message: str) -> int:
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(message) from exc


def _positive_tag(value: Any, label: str) -> int:
    try:
        tag = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be positive integers") from exc
    if tag <= 0:
        raise ValueError(f"{label} must be positive integers")
    return tag


__all__ = ["GmshImportResult", "from_model"]
