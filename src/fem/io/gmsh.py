from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import operator
from typing import Any, Literal

from fem.core import (
    Element2D,
    Element3D,
    ElementSet,
    FEMModel,
    Mesh2D,
    Mesh3D,
    Node2D,
    Node3D,
    NodeSet,
)


@dataclass(frozen=True)
class _ElementSpec:
    gmsh_type: int
    gmsh_name: str
    dimension: Literal[2, 3]
    order: int
    node_count: int
    fem_type: str
    connectivity_permutation: tuple[int, ...]


@dataclass(frozen=True)
class _ElementRecord:
    element_id: int
    spec: _ElementSpec
    node_ids: tuple[int, ...]


_ELEMENT_SPECS = {
    spec.gmsh_type: spec
    for spec in (
        _ElementSpec(2, "Triangle 3", 2, 1, 3, "Tri3", (0, 1, 2)),
        _ElementSpec(3, "Quadrilateral 4", 2, 1, 4, "Quad4", (0, 1, 2, 3)),
        _ElementSpec(4, "Tetrahedron 4", 3, 1, 4, "Tet4", (0, 1, 2, 3)),
        _ElementSpec(
            5,
            "Hexahedron 8",
            3,
            1,
            8,
            "Hex8",
            (0, 1, 2, 3, 4, 5, 6, 7),
        ),
    )
}

_CLOCKWISE_TO_COUNTERCLOCKWISE = {
    "Tri3": (0, 2, 1),
    "Quad4": (0, 3, 2, 1),
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

    def to_fem_model(self, name: str | None = None) -> FEMModel:
        """Return an analysis model without inventing physics or boundaries."""
        return FEMModel(
            mesh=self.mesh,
            name=name,
            node_sets=dict(self.node_sets),
            element_sets=dict(self.element_sets),
            metadata=deepcopy(self.metadata),
        )


def from_model(
    *,
    dimension: Literal[2, 3],
    gmsh_model: Any | None = None,
    plane_type: str = "stress",
    thickness: float = 1.0,
    z_tolerance: float = 1e-10,
) -> GmshImportResult:
    """Import the generated mesh from a caller-owned active Gmsh model."""
    normalized_dimension = _validate_dimension(dimension)
    normalized_plane_type = _validate_plane_type(plane_type)
    normalized_thickness = _validate_thickness(thickness)
    normalized_z_tolerance = _validate_z_tolerance(z_tolerance)

    gmsh_version = None
    if gmsh_model is None:
        gmsh_model, gmsh_version = _resolve_live_backend()

    return _from_backend(
        gmsh_model,
        dimension=normalized_dimension,
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


def _validate_dimension(value: Any) -> Literal[2, 3]:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (2, 3):
        raise ValueError(f"dimension must be 2 or 3, got {value!r}")
    return value


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
    dimension: Literal[2, 3],
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
        plane_type=plane_type,
        thickness=thickness,
    )
    if dimension == 2:
        mesh = Mesh2D(nodes=nodes, elements=elements, dofs_per_node=2)
    else:
        mesh = Mesh3D(nodes=nodes, elements=elements, dofs_per_node=3)

    node_sets, element_sets, physical_groups, skipped_groups = (
        _read_physical_groups(
            gmsh_model,
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
    if gmsh_version is not None:
        metadata["gmsh_version"] = gmsh_version
    return GmshImportResult(
        mesh=mesh,
        node_sets=node_sets,
        element_sets=element_sets,
        metadata=metadata,
    )


def _read_physical_groups(
    gmsh_model: Any,
    *,
    dimension: Literal[2, 3],
    retained_node_ids: set[int],
    retained_element_ids: set[int],
) -> tuple[
    dict[str, NodeSet],
    dict[str, ElementSet],
    _PhysicalGroupMetadata,
    tuple[dict[str, Any], ...],
]:
    node_sets: dict[str, NodeSet] = {}
    element_sets: dict[str, ElementSet] = {}
    metadata_groups: _PhysicalGroupMetadata = {}
    skipped_groups: list[dict[str, Any]] = []
    seen_names = {"node_set": set(), "element_set": set()}

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
        _store_physical_group_metadata(metadata_groups, name, record)

    return node_sets, element_sets, metadata_groups, tuple(skipped_groups)


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
        raw_blocks = gmsh_model.mesh.getElements(group_dimension, entity_tag)
        try:
            raw_types, raw_element_tags, raw_connectivity = raw_blocks
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"getElements({group_dimension}, {entity_tag}) returned malformed data"
            ) from exc
        element_types = list(raw_types)
        element_tag_blocks = list(raw_element_tags)
        connectivity_blocks = list(raw_connectivity)
        if (
            len(element_tag_blocks) != len(element_types)
            or len(connectivity_blocks) != len(element_types)
        ):
            raise ValueError(
                f"getElements({group_dimension}, {entity_tag}) returned inconsistent "
                "physical-group element block counts"
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
    dimension: Literal[2, 3],
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
            name, reported_dimension, order, node_count = reported
            raise ValueError(
                f"unsupported Gmsh element type {element_type} ({name!r}, "
                f"dimension={reported_dimension}, order={order}, nodes={node_count})"
            )
        expected = (spec.gmsh_name, spec.dimension, spec.order, spec.node_count)
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
            records.append(_ElementRecord(element_id, spec, node_ids))

    if not records:
        raise ValueError(
            f"Gmsh model contains no top-dimensional elements for dimension {dimension}"
        )
    return records


def _reported_element_properties(
    gmsh_mesh: Any,
    element_type: int,
) -> tuple[str, int, int, int]:
    properties = gmsh_mesh.getElementProperties(element_type)
    try:
        name, dimension, order, node_count = properties[:4]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"getElementProperties({element_type}) returned malformed data"
        ) from exc
    return (
        str(name),
        _integer_value(dimension, "reported element dimension must be an integer"),
        _integer_value(order, "reported element order must be an integer"),
        _integer_value(node_count, "reported element node count must be an integer"),
    )


def _read_nodes(
    gmsh_mesh: Any,
    referenced_node_ids: set[int],
    *,
    dimension: Literal[2, 3],
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
    dimension: Literal[2, 3],
    plane_type: str,
    thickness: float,
) -> list[Element2D] | list[Element3D]:
    if dimension == 3:
        return [
            Element3D(
                record.element_id,
                list(record.node_ids),
                record.spec.fem_type,
                {},
            )
            for record in records
        ]

    elements_2d: list[Element2D] = []
    props = {"plane_type": plane_type, "thickness": thickness}
    for record in records:
        node_ids = list(record.node_ids)
        if _signed_twice_area(node_ids, coordinates) < 0.0:
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
