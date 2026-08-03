"""Typed strict solid-Boolean lineage proof from one OCC compile."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .recipes import (
    BooleanBodyContext,
    BooleanLineageEntity,
    BooleanLineageMapping,
    FaceSeedConnectionProof,
)


@dataclass(frozen=True, slots=True)
class EntityEvidence:
    """Read-only analytic evidence captured before destructive OCC mutation."""

    logical_id: str
    dimension: int
    geometry_type: str
    bounds: tuple[float, float, float, float, float, float]
    direction: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class BooleanOperandEvidence:
    """Complete logical-to-OCC support evidence for one Boolean operand."""

    catalog: Any
    entities: tuple[EntityEvidence, ...]
    volume: Any
    volume_measure: float


@dataclass(frozen=True, slots=True)
class BooleanLineageProof:
    """One fully covered strict Boolean result."""

    result_volume: Any
    logical_entities: Mapping[str, tuple[Any, ...]]
    result_entities: tuple[BooleanLineageEntity, ...]
    topology_mappings: tuple[BooleanLineageMapping, ...]
    generated_intersections: tuple[str, ...]
    connection_proof: FaceSeedConnectionProof | None = None
    diagnostics: tuple[str, ...] = ()


class BooleanLineageResolutionError(ValueError):
    """A strict Boolean result cannot be assigned proven stable lineage."""


def validate_solid_boolean_input_map(
    boolean_result: Any,
    *,
    face_seed_connection: FaceSeedConnectionProof | None = None,
) -> None:
    """Require the OCC target/tool map to identify the single result volume."""

    if len(boolean_result.input_map) != 2:
        raise BooleanLineageResolutionError(
            "boolean.input-map.invalid: OCC did not return target/tool mapping"
        )
    if face_seed_connection is not None and type(
        face_seed_connection
    ) is not FaceSeedConnectionProof:
        raise TypeError("face_seed_connection must be FaceSeedConnectionProof")
    volumes = tuple(
        entity for entity in boolean_result.outputs if entity.dimension == 3
    )
    if not volumes:
        raise BooleanLineageResolutionError(
            "boolean.result.empty: Boolean produced no result volume"
        )
    if len(volumes) != 1:
        return
    result_volume = volumes[0]
    target_outputs, tool_outputs = boolean_result.input_map
    mapped = (*target_outputs, *tool_outputs)
    if face_seed_connection is not None and not mapped:
        return
    owns_result = result_volume in target_outputs or (
        face_seed_connection is not None and result_volume in tool_outputs
    )
    if (
        not owns_result
        or any(entity.dimension != 3 for entity in mapped)
        or any(entity not in boolean_result.outputs for entity in mapped)
    ):
        raise BooleanLineageResolutionError(
            "boolean.input-map.invalid: OCC target/tool mapping does not "
            "prove ownership of the result volume"
        )


def capture_boolean_operand_evidence(
    cad: Any,
    compiled: Any,
) -> BooleanOperandEvidence:
    """Capture source support evidence before a destructive OCC operation."""

    volumes = tuple(
        entity for entity in compiled.domain if entity.dimension == 3
    )
    if len(volumes) != 1:
        raise BooleanLineageResolutionError(
            "boolean.operand.single-volume: operand must contain one volume"
        )
    records: list[EntityEvidence] = []
    for logical_id, entities in compiled.logical_entities.items():
        for entity in entities:
            records.append(
                EntityEvidence(
                    logical_id,
                    entity.dimension,
                    _geometry_type(cad, entity),
                    tuple(float(value) for value in cad.bounding_box(entity)),
                    _geometry_direction(cad, entity),
                )
            )
    return BooleanOperandEvidence(
        compiled.catalog,
        tuple(records),
        volumes[0],
        float(cad.volume(volumes[0])),
    )


def resolve_solid_boolean_lineage(
    cad: Any,
    target_compiled: BooleanOperandEvidence,
    tool_compiled: BooleanOperandEvidence,
    boolean_result: Any,
    result_boundary: tuple[Any, ...],
    body_context: BooleanBodyContext,
    *,
    operation: str,
    face_seed_connection: FaceSeedConnectionProof | None = None,
    feature_id: str | None = None,
) -> BooleanLineageProof:
    """Prove complete face/edge/point coverage for one result volume."""

    if type(body_context) is not BooleanBodyContext:
        raise TypeError("body_context must be BooleanBodyContext")
    if operation not in {"fuse", "cut"}:
        raise ValueError("strict solid Boolean operation must be fuse or cut")
    if face_seed_connection is not None:
        if type(face_seed_connection) is not FaceSeedConnectionProof:
            raise TypeError("face_seed_connection must be FaceSeedConnectionProof")
        if operation != "fuse":
            raise ValueError("face seed connection proof is restricted to fuse")
    result_feature_id = body_context.feature_id if feature_id is None else feature_id
    if type(result_feature_id) is not str or not result_feature_id.strip():
        raise ValueError("Boolean result feature_id must not be empty")
    volumes = tuple(
        entity
        for entity in boolean_result.outputs
        if entity.dimension == 3
    )
    if len(volumes) != 1:
        raise BooleanLineageResolutionError(
            "boolean.result.volume-count: strict Boolean requires one volume"
        )
    result_volume = volumes[0]
    result_measure = float(cad.volume(result_volume))
    tolerance = max(
        1.0e-9,
        1.0e-9 * max(target_compiled.volume_measure, result_measure, 1.0),
    )
    if result_measure <= tolerance:
        raise BooleanLineageResolutionError(
            "boolean.result.empty: Boolean removed the target"
        )
    if operation == "cut" and math.isclose(
        result_measure,
        target_compiled.volume_measure,
        rel_tol=1.0e-9,
        abs_tol=tolerance,
    ):
        raise BooleanLineageResolutionError(
            "boolean.cut.no-op: tool did not remove target volume"
        )
    if operation == "fuse" and result_measure <= max(
        target_compiled.volume_measure,
        tool_compiled.volume_measure,
    ):
        raise BooleanLineageResolutionError(
            "boolean.fuse.no-op: result did not combine both operands"
        )
    accepted_connection_proof: FaceSeedConnectionProof | None = None
    if operation == "fuse":
        overlap_measure = (
            target_compiled.volume_measure
            + tool_compiled.volume_measure
            - result_measure
        )
        if overlap_measure <= tolerance:
            if face_seed_connection is None:
                raise BooleanLineageResolutionError(
                    "boolean.fuse.non-positive-overlap: operands only touch; "
                    "strict fuse requires positive shared volume"
                )
            _validate_face_seed_connection(
                face_seed_connection,
                target_compiled,
                tool_compiled,
            )
            accepted_connection_proof = face_seed_connection

    faces = _unique(
        entity for entity in result_boundary if entity.dimension == 2
    )
    if not faces:
        raise BooleanLineageResolutionError(
            "boolean.lineage.boundary-empty: result has no boundary faces"
        )
    edges = _unique(
        entity
        for entity in cad.boundary(faces, combined=False)
        if entity.dimension == 1
    )
    points = _unique(
        entity
        for entity in cad.boundary(edges, combined=False)
        if entity.dimension == 0
    )

    grouped: dict[str, list[Any]] = {"body:domain": [result_volume]}
    roles: dict[str, tuple[str, str]] = {
        "body:domain": ("body", "boolean.result")
    }
    mappings: set[BooleanLineageMapping] = {
        BooleanLineageMapping(
            "target",
            "body:domain",
            "body:domain",
            "preserved",
        )
    }
    provenance_by_entity: dict[Any, tuple[set[str], set[str]]] = {}
    mapping_provenance_by_entity: dict[Any, tuple[set[str], set[str]]] = {}

    for entity in faces:
        target_ids, tool_ids = _source_matches(
            cad,
            entity,
            target_compiled,
            tool_compiled,
        )
        logical_id, role = _logical_result_id(
            "face",
            result_feature_id,
            target_ids,
            tool_ids,
        )
        if logical_id is None:
            raise BooleanLineageResolutionError(
                "boolean.lineage.face-unclassified: result face has no "
                "target/tool support evidence "
                f"(type={_geometry_type(cad, entity)!r}, "
                f"bounds={cad.bounding_box(entity)!r})"
            )
        _record_group(
            grouped,
            roles,
            mappings,
            logical_id,
            role,
            entity,
            target_ids,
            tool_ids,
        )
        provenance_by_entity[entity] = (target_ids, tool_ids)
        mapping_provenance_by_entity[entity] = (target_ids, tool_ids)

    generated_intersections: set[str] = set()
    face_sources_by_edge: dict[
        tuple[int, int],
        list[tuple[set[str], set[str]]],
    ] = {}
    for face in faces:
        face_target, face_tool = provenance_by_entity[face]
        for edge in cad.boundary((face,), combined=False):
            face_sources_by_edge.setdefault(
                _entity_key(edge),
                [],
            ).append(
                (set(face_target), set(face_tool))
            )
    edge_roles: dict[Any, str] = {}
    for entity in edges:
        direct_target, direct_tool = _source_matches(
            cad,
            entity,
            target_compiled,
            tool_compiled,
        )
        adjacent = face_sources_by_edge.get(
            _entity_key(entity),
            [],
        )
        face_target = set().union(
            *(target_ids for target_ids, _tool_ids in adjacent)
        ) if adjacent else set()
        face_tool = set().union(
            *(tool_ids for _target_ids, tool_ids in adjacent)
        ) if adjacent else set()
        intersection = (
            any(target_ids and not tool_ids for target_ids, tool_ids in adjacent)
            and any(tool_ids and not target_ids for target_ids, tool_ids in adjacent)
        ) or (
            operation == "cut"
            and bool(direct_tool)
            and bool(face_target)
            and bool(face_tool)
        )
        if intersection:
            target_ids, tool_ids = face_target, face_tool
            logical_id = _generated_result_id(
                "edge",
                result_feature_id,
                "intersection",
                target_ids,
                tool_ids,
            )
            role = "boolean.intersection"
        elif direct_target or direct_tool:
            target_ids, tool_ids = direct_target, direct_tool
            logical_id, role = _logical_result_id(
                "edge",
                result_feature_id,
                target_ids,
                tool_ids,
            )
        else:
            target_ids, tool_ids = face_target, face_tool
            logical_id = _generated_result_id(
                "edge",
                result_feature_id,
                "intersection",
                target_ids,
                tool_ids,
            )
            role = "boolean.intersection"
        if logical_id is None:
            raise BooleanLineageResolutionError(
                "boolean.lineage.edge-unclassified: result edge has no "
                "support or adjacent-face evidence"
            )
        _record_group(
            grouped,
            roles,
            mappings,
            logical_id,
            role,
            entity,
            target_ids,
            tool_ids,
        )
        provenance_by_entity[entity] = (target_ids, tool_ids)
        mapping_provenance_by_entity[entity] = (
            (direct_target, direct_tool)
            if direct_target or direct_tool
            else (target_ids, tool_ids)
        )
        edge_roles[entity] = role
        if "/intersection/" in logical_id:
            generated_intersections.add(logical_id)

    edge_sources_by_point: dict[
        tuple[int, int],
        list[
            tuple[
                set[str],
                set[str],
                str,
                set[str],
                set[str],
            ]
        ],
    ] = {}
    for edge in edges:
        edge_target, edge_tool = provenance_by_entity[edge]
        mapped_target, mapped_tool = mapping_provenance_by_entity[edge]
        for point in cad.boundary((edge,), combined=False):
            edge_sources_by_point.setdefault(
                _entity_key(point),
                [],
            ).append(
                (
                    set(edge_target),
                    set(edge_tool),
                    edge_roles[edge],
                    set(mapped_target),
                    set(mapped_tool),
                )
            )
    for entity in points:
        direct_target, direct_tool = _source_matches(
            cad,
            entity,
            target_compiled,
            tool_compiled,
        )
        adjacent = edge_sources_by_point.get(
            _entity_key(entity),
            [],
        )
        edge_target = set().union(
            *(target_ids for target_ids, _tool_ids, _role, _mt, _mu in adjacent)
        ) if adjacent else set()
        edge_tool = set().union(
            *(tool_ids for _target_ids, tool_ids, _role, _mt, _mu in adjacent)
        ) if adjacent else set()
        intersection = any(
            role == "boolean.intersection"
            for _target_ids, _tool_ids, role, _mt, _mu in adjacent
        )
        if not intersection and (direct_target or direct_tool):
            target_ids, tool_ids = direct_target, direct_tool
            logical_id, role = _logical_result_id(
                "point",
                result_feature_id,
                target_ids,
                tool_ids,
            )
        else:
            target_ids = {*direct_target, *edge_target}
            tool_ids = {*direct_tool, *edge_tool}
            logical_id = _generated_result_id(
                "point",
                result_feature_id,
                "intersection" if intersection else "combined",
                target_ids,
                tool_ids,
            )
            role = (
                "boolean.intersection"
                if intersection
                else "boolean.combined"
            )
        if logical_id is None:
            raise BooleanLineageResolutionError(
                "boolean.lineage.point-unclassified: result point has no "
                "support or adjacent-edge evidence"
            )
        if not intersection and (direct_target or direct_tool):
            mapping_target = set(direct_target)
            mapping_tool = set(direct_tool)
        else:
            mapping_target = {
                *direct_target,
                *(
                    source_id
                    for (
                        _target_ids,
                        _tool_ids,
                        _role,
                        mapped_target,
                        _mapped_tool,
                    ) in adjacent
                    for source_id in mapped_target
                    if source_id.startswith(("point:", "edge:"))
                ),
            }
            mapping_tool = {
                *direct_tool,
                *(
                    source_id
                    for (
                        _target_ids,
                        _tool_ids,
                        _role,
                        _mapped_target,
                        mapped_tool,
                    ) in adjacent
                    for source_id in mapped_tool
                    if source_id.startswith(("point:", "edge:"))
                ),
            }
        if not mapping_target and target_ids:
            mapping_target = set(target_ids)
        if not mapping_tool and tool_ids:
            mapping_tool = set(tool_ids)
        _record_group(
            grouped,
            roles,
            mappings,
            logical_id,
            role,
            entity,
            target_ids,
            tool_ids,
            mapping_target_ids=mapping_target,
            mapping_tool_ids=mapping_tool,
        )
        if "/intersection/" in logical_id:
            generated_intersections.add(logical_id)

    result_entities: list[BooleanLineageEntity] = []
    for logical_id, (kind, role) in roles.items():
        topology_links: tuple[str, ...] = ()
        if role == "boolean.target-survivor":
            try:
                source_entity = target_compiled.catalog.entity(logical_id)
            except KeyError as error:
                raise BooleanLineageResolutionError(
                    "boolean.lineage.target-catalog-mismatch: preserved "
                    f"logical entity {logical_id!r} is absent"
                ) from error
            role = source_entity.semantic_role
            topology_links = tuple(
                link
                for link in source_entity.topology_links
                if link in grouped
            )
        result_entities.append(
            BooleanLineageEntity(
                kind,
                logical_id,
                role,
                topology_links,
            )
        )
    proof = BooleanLineageProof(
        result_volume,
        {
            logical_id: _unique(entities)
            for logical_id, entities in grouped.items()
        },
        tuple(result_entities),
        tuple(mappings),
        tuple(sorted(generated_intersections)),
        accepted_connection_proof,
        (
            ("boolean.fuse.face-seed-connection",)
            if accepted_connection_proof is not None
            else ()
        ),
    )
    _validate_coverage(
        (*faces, *edges, *points, result_volume),
        proof.logical_entities,
    )
    return proof


def _validate_face_seed_connection(
    proof: FaceSeedConnectionProof,
    target: BooleanOperandEvidence,
    tool: BooleanOperandEvidence,
) -> None:
    target_faces = {
        item.logical_id
        for item in target.entities
        if item.dimension == 2
    }
    tool_faces = {
        item.logical_id
        for item in tool.entities
        if item.dimension == 2
    }
    if proof.support_face_id not in target_faces:
        raise BooleanLineageResolutionError(
            "boolean.fuse.face-seed.target-mismatch: support face is absent "
            "from target evidence"
        )
    if proof.tool_start_face_id not in tool_faces:
        raise BooleanLineageResolutionError(
            "boolean.fuse.face-seed.tool-mismatch: start face is absent "
            "from tool evidence"
        )


def _source_matches(
    cad: Any,
    result: Any,
    target: BooleanOperandEvidence,
    tool: BooleanOperandEvidence,
) -> tuple[set[str], set[str]]:
    evidence = EntityEvidence(
        "",
        result.dimension,
        _geometry_type(cad, result),
        tuple(float(value) for value in cad.bounding_box(result)),
        _geometry_direction(cad, result),
    )
    return (
        _unambiguous_support_ids(
            evidence,
            target,
            operand_label="target",
        ),
        _unambiguous_support_ids(
            evidence,
            tool,
            operand_label="tool",
        ),
    )


def _unambiguous_support_ids(
    result: EntityEvidence,
    evidence: BooleanOperandEvidence,
    *,
    operand_label: str,
) -> set[str]:
    candidates = tuple(
        item
        for item in evidence.entities
        if item.dimension == result.dimension
        and _supports_same_geometry(result, item)
    )
    exact = tuple(
        item
        for item in candidates
        if _same_bounds(result.bounds, item.bounds)
    )
    selected = exact or candidates
    logical_ids = {item.logical_id for item in selected}
    if len(logical_ids) > 1:
        raise BooleanLineageResolutionError(
            "boolean.lineage.output-ambiguous: result entity matches "
            f"multiple {operand_label} logical supports {sorted(logical_ids)!r}"
        )
    return logical_ids


def _same_bounds(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> bool:
    scale = max(
        *(abs(value) for value in left),
        *(abs(value) for value in right),
        1.0,
    )
    tolerance = 5.0e-7 * scale
    return all(
        abs(left_value - right_value) <= tolerance
        for left_value, right_value in zip(left, right, strict=True)
    )


def _supports_same_geometry(
    result: EntityEvidence,
    source: EntityEvidence,
) -> bool:
    result_type = result.geometry_type.casefold()
    source_type = source.geometry_type.casefold()
    result_type_unknown = (
        result_type == "unknown"
        or result_type.startswith("dimension-")
    )
    if not result_type_unknown and result_type != source_type:
        return False
    if result.direction is not None and source.direction is not None:
        alignment = abs(
            sum(
                left * right
                for left, right in zip(
                    result.direction,
                    source.direction,
                    strict=True,
                )
            )
        )
        if alignment < 1.0 - 1.0e-7:
            return False
    scale = max(
        *(abs(value) for value in result.bounds),
        *(abs(value) for value in source.bounds),
        1.0,
    )
    tolerance = 5.0e-7 * scale
    result_fixed = tuple(
        axis
        for axis in range(3)
        if abs(result.bounds[axis + 3] - result.bounds[axis]) <= tolerance
    )
    source_fixed = tuple(
        axis
        for axis in range(3)
        if abs(source.bounds[axis + 3] - source.bounds[axis]) <= tolerance
    )
    if result_fixed != source_fixed:
        return False
    if any(
        abs(result.bounds[axis] - source.bounds[axis]) > tolerance
        for axis in result_fixed
    ):
        return False
    for axis in range(3):
        if axis in result_fixed:
            continue
        result_min, result_max = result.bounds[axis], result.bounds[axis + 3]
        source_min, source_max = source.bounds[axis], source.bounds[axis + 3]
        if result_max < source_min - tolerance or source_max < result_min - tolerance:
            return False
    return True


def _logical_result_id(
    kind: str,
    feature_id: str,
    target_ids: set[str],
    tool_ids: set[str],
) -> tuple[str | None, str]:
    target_names = _local_names(target_ids, kind)
    tool_names = _local_names(tool_ids, kind)
    if len(target_names) == 1 and not tool_names:
        logical_id = next(iter(target_ids))
        return logical_id, "boolean.target-survivor"
    if target_names and tool_names:
        return (
            _generated_result_id(
                kind,
                feature_id,
                "combined",
                target_ids,
                tool_ids,
            ),
            "boolean.combined",
        )
    if tool_names:
        logical_id = (
            f"{kind}:boolean/{feature_id}/tool/"
            f"{'+'.join(tool_names)}"
        )
        return logical_id, "boolean.tool-derived"
    if target_names:
        logical_id = (
            f"{kind}:boolean/{feature_id}/target/"
            f"{'+'.join(target_names)}"
        )
        return logical_id, "boolean.target-derived"
    return None, ""


def _generated_result_id(
    kind: str,
    feature_id: str,
    group: str,
    target_ids: set[str],
    tool_ids: set[str],
) -> str | None:
    target_names = _source_names(target_ids)
    tool_names = _source_names(tool_ids)
    if not target_names and not tool_names:
        return None
    return (
        f"{kind}:boolean/{feature_id}/{group}/"
        f"{'+'.join(target_names) or '-'}/"
        f"{'+'.join(tool_names) or '-'}"
    )


def _source_names(logical_ids: set[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            logical_id.split(":", 1)[1]
            for logical_id in logical_ids
        )
    )


def _local_names(logical_ids: set[str], kind: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                logical_id.split(":", 1)[1]
                if logical_id.startswith(f"{kind}:")
                else logical_id.replace(":", "-", 1)
            )
            for logical_id in logical_ids
        )
    )


def _record_group(
    grouped: dict[str, list[Any]],
    roles: dict[str, tuple[str, str]],
    mappings: set[BooleanLineageMapping],
    logical_id: str,
    role: str,
    entity: Any,
    target_ids: set[str],
    tool_ids: set[str],
    *,
    mapping_target_ids: set[str] | None = None,
    mapping_tool_ids: set[str] | None = None,
) -> None:
    kind = logical_id.split(":", 1)[0]
    grouped.setdefault(logical_id, []).append(entity)
    roles.setdefault(logical_id, (kind, role))
    for source, source_ids in (
        (
            "target",
            target_ids if mapping_target_ids is None else mapping_target_ids,
        ),
        (
            "tool",
            tool_ids if mapping_tool_ids is None else mapping_tool_ids,
        ),
    ):
        for source_id in source_ids:
            mappings.add(
                BooleanLineageMapping(
                    source,
                    source_id,
                    logical_id,
                    (
                        "preserved"
                        if source == "target" and source_id == logical_id
                        else "derived"
                    ),
                )
            )


def _geometry_type(cad: Any, entity: Any) -> str:
    query = getattr(cad, "geometry_type", None)
    if callable(query):
        return str(query(entity))
    return f"dimension-{entity.dimension}"


def _geometry_direction(
    cad: Any,
    entity: Any,
) -> tuple[float, float, float] | None:
    query = getattr(cad, "geometry_direction", None)
    if not callable(query):
        return None
    value = query(entity)
    if value is None:
        return None
    return tuple(float(component) for component in value)


def _validate_coverage(
    boundary: tuple[Any, ...],
    logical_entities: Mapping[str, tuple[Any, ...]],
) -> None:
    counts: dict[Any, int] = {}
    for entities in logical_entities.values():
        for entity in entities:
            counts[entity] = counts.get(entity, 0) + 1
    missing = tuple(entity for entity in boundary if counts.get(entity, 0) == 0)
    if missing:
        raise BooleanLineageResolutionError(
            "boolean.lineage.coverage-incomplete: result boundary contains "
            "anonymous entities"
        )
    duplicates = tuple(
        entity for entity in boundary if counts.get(entity, 0) != 1
    )
    if duplicates:
        raise BooleanLineageResolutionError(
            "boolean.lineage.catalog-mismatch: result entity belongs to "
            "multiple logical groups"
        )


def _unique(values) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def _entity_key(entity: Any) -> tuple[int, int]:
    return int(entity.dimension), int(entity.tag)


__all__ = [
    "BooleanLineageProof",
    "BooleanLineageResolutionError",
    "BooleanOperandEvidence",
    "EntityEvidence",
    "capture_boolean_operand_evidence",
    "resolve_solid_boolean_lineage",
    "validate_solid_boolean_input_map",
]
