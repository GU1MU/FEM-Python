"""Exact lineage proof for strict planar OCC Boolean features."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from .recipes import (
    BooleanLineageEntity,
    BooleanLineageMapping,
    PlanarBooleanContext,
)
from .references import LogicalEntityRef


@dataclass(frozen=True, slots=True)
class PlanarEntityEvidence:
    """Backend-neutral support evidence captured before OCC mutation."""

    logical_id: str
    dimension: int
    geometry_type: str
    bounds: tuple[float, float, float, float, float, float]
    direction: tuple[float, float, float] | None


@dataclass(frozen=True, slots=True)
class PlanarOperandEvidence:
    """Complete logical support evidence for one planar operand."""

    catalog: Any
    entities: tuple[PlanarEntityEvidence, ...]
    surface_areas: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class PlanarBooleanLineageProof:
    """One completely classified planar Boolean result."""

    result_surfaces: tuple[Any, ...]
    logical_entities: Mapping[str, tuple[Any, ...]]
    result_entities: tuple[BooleanLineageEntity, ...]
    topology_mappings: tuple[BooleanLineageMapping, ...]
    generated_intersections: tuple[str, ...]
    diagnostics: tuple[str, ...] = ()


class PlanarBooleanLineageResolutionError(ValueError):
    """A planar Boolean result cannot receive exact stable lineage."""


def capture_planar_operand_evidence(
    cad: Any,
    compiled: Any,
    face_ids: tuple[str, ...],
) -> PlanarOperandEvidence:
    """Capture source supports and selected material area before OCC mutation."""

    records: list[PlanarEntityEvidence] = []
    for logical_id, entities in compiled.logical_entities.items():
        for entity in entities:
            records.append(
                PlanarEntityEvidence(
                    logical_id,
                    entity.dimension,
                    _geometry_type(cad, entity),
                    tuple(float(value) for value in cad.bounding_box(entity)),
                    _geometry_direction(cad, entity),
                )
            )
    areas: dict[str, float] = {}
    for face_id in face_ids:
        surfaces = tuple(compiled.logical_entities.get(face_id, ()))
        if len(surfaces) != 1 or surfaces[0].dimension != 2:
            raise PlanarBooleanLineageResolutionError(
                "planar-boolean.operand.face-not-unique: "
                f"{face_id} does not resolve to one OCC surface"
            )
        areas[face_id] = float(cad.area(surfaces[0]))
    return PlanarOperandEvidence(compiled.catalog, tuple(records), areas)


def validate_planar_boolean_input_map(
    boolean_result: Any,
    *,
    tool_count: int,
    operation: str,
) -> None:
    """Validate OCC ownership mapping for one target and selected tools."""

    if len(boolean_result.input_map) != 1 + int(tool_count):
        raise PlanarBooleanLineageResolutionError(
            "planar-boolean.input-map.invalid: OCC did not return one "
            "target map and one map per tool Profile"
        )
    surfaces = tuple(
        entity for entity in boolean_result.outputs if entity.dimension == 2
    )
    if not surfaces:
        raise PlanarBooleanLineageResolutionError(
            "planar-boolean.result.empty: Boolean produced no material Face"
        )
    mapped = tuple(entity for outputs in boolean_result.input_map for entity in outputs)
    if any(
        entity.dimension != 2 or entity not in boolean_result.outputs
        for entity in mapped
    ):
        raise PlanarBooleanLineageResolutionError(
            "planar-boolean.input-map.invalid: OCC returned a foreign "
            "or wrong-dimensional mapping"
        )
    if not any(surface in boolean_result.input_map[0] for surface in surfaces):
        raise PlanarBooleanLineageResolutionError(
            "planar-boolean.input-map.invalid: target ownership is unproven"
        )
    if operation == "fuse" and any(
        not any(surface in outputs for surface in surfaces)
        for outputs in boolean_result.input_map[1:]
    ):
        raise PlanarBooleanLineageResolutionError(
            "planar-boolean.fuse.disjoint: every tool Profile must connect "
            "to the material result"
        )


def resolve_planar_boolean_lineage(
    cad: Any,
    target_evidence: PlanarOperandEvidence,
    tool_evidence: PlanarOperandEvidence,
    boolean_result: Any,
    context: PlanarBooleanContext,
    *,
    operation: str,
    unaffected_logical_entities: Mapping[str, tuple[Any, ...]] = (),
) -> PlanarBooleanLineageProof:
    """Prove every selectable result Face, edge, and point."""

    if type(context) is not PlanarBooleanContext:
        raise TypeError("context must be PlanarBooleanContext")
    if operation not in {"fuse", "cut"}:
        raise ValueError("strict planar Boolean operation must be fuse or cut")
    affected_faces = _unique(
        entity for entity in boolean_result.outputs if entity.dimension == 2
    )
    if operation == "fuse":
        unaffected_faces = _unique(
            entity
            for logical_id, entities in unaffected_logical_entities.items()
            if logical_id != "body:domain"
            for entity in entities
            if entity.dimension == 2
        )
        if any(
            distance <= 1.0e-10
            for affected_face in affected_faces
            for distance in cad.distances_to(affected_face, unaffected_faces)
        ):
            raise PlanarBooleanLineageResolutionError(
                "planar-boolean.fuse.unaffected-contact: fused material "
                "must not touch or overlap an unselected object Face"
            )
    target_area = float(target_evidence.surface_areas[context.target_face_id])
    result_area = sum(float(cad.area(face)) for face in affected_faces)
    tolerance = max(1.0e-9, 1.0e-9 * max(target_area, result_area, 1.0))
    if result_area <= tolerance:
        raise PlanarBooleanLineageResolutionError(
            "planar-boolean.result.empty: Boolean removed the target Face"
        )
    if operation == "cut" and math.isclose(
        result_area,
        target_area,
        rel_tol=1.0e-9,
        abs_tol=tolerance,
    ):
        raise PlanarBooleanLineageResolutionError(
            "planar-boolean.cut.no-op: tool did not remove target area"
        )
    if operation == "fuse":
        if len(affected_faces) != 1:
            raise PlanarBooleanLineageResolutionError(
                "planar-boolean.fuse.disjoint: fuse must form one connected "
                "material Face"
            )
        tool_area = sum(tool_evidence.surface_areas.values())
        if result_area <= max(target_area, tool_area) + tolerance:
            raise PlanarBooleanLineageResolutionError(
                "planar-boolean.fuse.no-op: result did not combine target "
                "and tool material"
            )

    grouped: dict[str, list[Any]] = {
        logical_id: list(entities)
        for logical_id, entities in unaffected_logical_entities.items()
        if logical_id != "body:domain"
    }
    roles: dict[str, tuple[str, str]] = {
        logical_id: (
            logical_id.split(":", 1)[0],
            "boolean.unaffected",
        )
        for logical_id in grouped
    }
    mappings: set[BooleanLineageMapping] = set()
    _map_unaffected_sources(
        target_evidence,
        grouped,
        mappings,
    )

    affected_face_ids: dict[Any, str] = {}
    for face in affected_faces:
        logical_id = (
            context.target_face_id
            if len(affected_faces) == 1
            else _generated_id(cad, "face", context.feature_id, "result", face)
        )
        _require_group_available(grouped, logical_id, face)
        grouped.setdefault(logical_id, []).append(face)
        roles[logical_id] = (
            "face",
            (
                "boolean.target-survivor"
                if logical_id == context.target_face_id
                else "boolean.split-result"
            ),
        )
        affected_face_ids[face] = logical_id
        for source_id in _face_source_ids(
            target_evidence,
            context.target_face_id,
        ):
            mappings.add(
                BooleanLineageMapping(
                    "target",
                    source_id,
                    logical_id,
                    ("preserved" if source_id == logical_id else "derived"),
                )
            )
        if operation == "fuse":
            for source_id in context.tool_face_ids:
                mappings.add(
                    BooleanLineageMapping(
                        "tool",
                        source_id,
                        logical_id,
                        "derived",
                    )
                )

    all_faces = _unique(
        entity
        for logical_id, entities in grouped.items()
        if logical_id.startswith("face:")
        for entity in entities
    )
    all_edges = _unique(
        entity
        for face in all_faces
        for entity in cad.boundary((face,), combined=False)
        if entity.dimension == 1
    )
    assigned_edges = {
        entity
        for logical_id, entities in grouped.items()
        if logical_id.startswith("edge:")
        for entity in entities
    }
    generated_intersections: set[str] = set()
    for edge in all_edges:
        if edge in assigned_edges:
            continue
        target_ids = _source_matches(cad, edge, target_evidence)
        tool_ids = _source_matches(cad, edge, tool_evidence)
        exact_target = _exact_primary_matches(
            cad,
            edge,
            target_ids,
            target_evidence,
        )
        if len(exact_target) == 1 and not tool_ids:
            logical_id = exact_target[0]
            role = "boolean.target-survivor"
        else:
            group = (
                "intersection"
                if tool_ids and (operation == "cut" or target_ids)
                else "result"
            )
            logical_id = _generated_id(
                cad,
                "edge",
                context.feature_id,
                group,
                edge,
            )
            role = (
                "boolean.intersection"
                if group == "intersection"
                else "boolean.result-boundary"
            )
            if group == "intersection":
                generated_intersections.add(logical_id)
        _require_group_available(grouped, logical_id, edge)
        grouped.setdefault(logical_id, []).append(edge)
        roles[logical_id] = ("edge", role)
        _record_same_kind_mappings(
            mappings,
            logical_id,
            target_ids,
            tool_ids,
        )
        if not target_ids and not tool_ids:
            # A boundary-only fuse can split and then merge coincident operand
            # edges. OCC may return a new edge whose support signature no
            # longer matches either original edge, while the result Face is
            # still fully proven. Keep same-dimensional provenance through
            # the selected Profiles' proven boundary sets in that case.
            _record_same_kind_mappings(
                mappings,
                logical_id,
                _face_boundary_source_ids(
                    target_evidence,
                    (context.target_face_id,),
                ),
                _face_boundary_source_ids(
                    tool_evidence,
                    context.tool_face_ids,
                ),
            )
        if logical_id in generated_intersections:
            mappings.add(
                BooleanLineageMapping(
                    "target",
                    context.target_face_id,
                    logical_id,
                    "derived",
                )
            )

    all_points = _unique(
        entity
        for edge in all_edges
        for entity in cad.boundary((edge,), combined=False)
        if entity.dimension == 0
    )
    assigned_points = {
        entity
        for logical_id, entities in grouped.items()
        if logical_id.startswith("point:")
        for entity in entities
    }
    edge_ids_by_entity = {
        entity: logical_id
        for logical_id, entities in grouped.items()
        if logical_id.startswith("edge:")
        for entity in entities
    }
    for point in all_points:
        if point in assigned_points:
            continue
        target_ids = _source_matches(cad, point, target_evidence)
        tool_ids = _source_matches(cad, point, tool_evidence)
        adjacent_edges = tuple(
            edge for edge in all_edges if point in cad.boundary((edge,), combined=False)
        )
        adjacent_edge_ids = {edge_ids_by_entity[edge] for edge in adjacent_edges}
        adjacent_target_sources = (
            set().union(
                *(
                    _source_matches(cad, edge, target_evidence)
                    for edge in adjacent_edges
                )
            )
            if adjacent_edges
            else set()
        )
        adjacent_tool_sources = (
            set().union(
                *(_source_matches(cad, edge, tool_evidence) for edge in adjacent_edges)
            )
            if adjacent_edges
            else set()
        )
        for mapping in mappings:
            if (
                mapping.target_logical_id not in adjacent_edge_ids
                or LogicalEntityRef(mapping.source_logical_id).kind != "edge"
            ):
                continue
            if mapping.source == "target":
                adjacent_target_sources.add(mapping.source_logical_id)
            else:
                adjacent_tool_sources.add(mapping.source_logical_id)
        exact_target = _exact_primary_matches(
            cad,
            point,
            target_ids,
            target_evidence,
        )
        intersection = (
            any(
                logical_id in generated_intersections
                for logical_id in adjacent_edge_ids
            )
            or bool(target_ids and tool_ids)
            or bool(adjacent_target_sources and adjacent_tool_sources)
            or (
                not target_ids
                and not tool_ids
                and bool(adjacent_target_sources or adjacent_tool_sources)
            )
        )
        if len(exact_target) == 1 and not tool_ids and not intersection:
            logical_id = exact_target[0]
            role = "boolean.target-survivor"
        else:
            group = "intersection" if intersection else "result"
            logical_id = _generated_id(
                cad,
                "point",
                context.feature_id,
                group,
                point,
            )
            role = "boolean.intersection" if intersection else "boolean.result-boundary"
            if intersection:
                generated_intersections.add(logical_id)
        _require_group_available(grouped, logical_id, point)
        grouped.setdefault(logical_id, []).append(point)
        roles[logical_id] = ("point", role)
        _record_same_kind_mappings(
            mappings,
            logical_id,
            target_ids,
            tool_ids,
        )
        if not target_ids and not tool_ids:
            _record_cross_dimension_mappings(
                mappings,
                logical_id,
                adjacent_target_sources,
                adjacent_tool_sources,
            )

    result_faces = tuple(
        sorted(
            (logical_id for logical_id in grouped if logical_id.startswith("face:")),
        )
    )
    grouped["body:domain"] = list(
        (
            *affected_faces,
            *(
                entity
                for logical_id, entities in unaffected_logical_entities.items()
                if logical_id.startswith("face:")
                for entity in entities
            ),
        )
    )
    grouped["body:domain"] = list(_unique(grouped["body:domain"]))
    roles["body:domain"] = ("body", "boolean.planar-result")
    mappings.add(
        BooleanLineageMapping(
            "target",
            "body:domain",
            "body:domain",
            "preserved",
        )
    )
    mappings.add(
        BooleanLineageMapping(
            "tool",
            "body:domain",
            "body:domain",
            "derived",
        )
    )

    logical_entities = {
        logical_id: _unique(entities) for logical_id, entities in grouped.items()
    }
    _validate_coverage(
        (*all_faces, *all_edges, *all_points),
        logical_entities,
    )
    links = _topology_links(
        cad,
        logical_entities,
        result_faces,
    )
    result_entities = tuple(
        BooleanLineageEntity(
            roles[logical_id][0],
            logical_id,
            roles[logical_id][1],
            links.get(logical_id, ()),
        )
        for logical_id in sorted(
            logical_entities,
            key=lambda value: (
                {"point": 0, "edge": 1, "face": 2, "body": 3}[value.split(":", 1)[0]],
                value,
            ),
        )
    )
    mapped_ids = {mapping.target_logical_id for mapping in mappings}
    for entity in result_entities:
        if entity.logical_id not in mapped_ids:
            raise PlanarBooleanLineageResolutionError(
                "planar-boolean.lineage.coverage-incomplete: "
                f"{entity.logical_id} has no source proof"
            )
    return PlanarBooleanLineageProof(
        affected_faces,
        logical_entities,
        result_entities,
        tuple(mappings),
        tuple(sorted(generated_intersections)),
    )


def _face_source_ids(
    evidence: PlanarOperandEvidence,
    target_face_id: str,
) -> tuple[str, ...]:
    target_records = tuple(
        record
        for record in evidence.entities
        if record.logical_id == target_face_id and record.dimension == 2
    )
    if len(target_records) != 1:
        return (target_face_id,)
    target = target_records[0]
    return tuple(
        sorted(
            {
                record.logical_id
                for record in evidence.entities
                if record.dimension == 2
                and LogicalEntityRef(record.logical_id).kind == "face"
                and _evidence_exact(record, target)
            }
        )
    )


def _map_unaffected_sources(
    evidence: PlanarOperandEvidence,
    grouped: Mapping[str, list[Any]],
    mappings: set[BooleanLineageMapping],
) -> None:
    for logical_id in grouped:
        mappings.add(
            BooleanLineageMapping(
                "target",
                logical_id,
                logical_id,
                "preserved",
            )
        )


def _record_same_kind_mappings(
    mappings: set[BooleanLineageMapping],
    target_id: str,
    target_sources: set[str],
    tool_sources: set[str],
) -> None:
    for source, source_ids in (
        ("target", target_sources),
        ("tool", tool_sources),
    ):
        for source_id in source_ids:
            mappings.add(
                BooleanLineageMapping(
                    source,
                    source_id,
                    target_id,
                    (
                        "preserved"
                        if source == "target" and source_id == target_id
                        else "derived"
                    ),
                )
            )


def _face_boundary_source_ids(
    evidence: PlanarOperandEvidence,
    face_ids: tuple[str, ...],
) -> set[str]:
    result: set[str] = set()
    for face_id in face_ids:
        try:
            face = evidence.catalog.entity(face_id)
        except KeyError:
            continue
        result.update(
            logical_id
            for logical_id in face.topology_links
            if logical_id.startswith("edge:")
        )
    return result


def _record_cross_dimension_mappings(
    mappings: set[BooleanLineageMapping],
    target_id: str,
    target_sources: set[str],
    tool_sources: set[str],
) -> None:
    for source, source_ids in (
        ("target", target_sources),
        ("tool", tool_sources),
    ):
        for source_id in source_ids:
            mappings.add(BooleanLineageMapping(source, source_id, target_id, "derived"))


def _source_matches(
    cad: Any,
    entity: Any,
    evidence: PlanarOperandEvidence,
) -> set[str]:
    result = PlanarEntityEvidence(
        "",
        entity.dimension,
        _geometry_type(cad, entity),
        tuple(float(value) for value in cad.bounding_box(entity)),
        _geometry_direction(cad, entity),
    )
    return {
        record.logical_id
        for record in evidence.entities
        if _support_matches(result, record)
    }


def _exact_primary_matches(
    cad: Any,
    entity: Any,
    source_ids: set[str],
    evidence: PlanarOperandEvidence,
) -> tuple[str, ...]:
    records = {
        record.logical_id: record
        for record in evidence.entities
        if record.logical_id in source_ids
    }
    if not records:
        return ()
    bounds = tuple(float(value) for value in cad.bounding_box(entity))
    exact = {
        logical_id
        for logical_id, record in records.items()
        if _bounds_close(record.bounds, bounds)
    }
    primary = {
        _canonical_logical_id(evidence.catalog, logical_id) for logical_id in exact
    }
    return tuple(sorted(primary))


def _canonical_logical_id(catalog: Any, logical_id: str) -> str:
    current = logical_id
    visited: set[str] = set()
    kind = current.split(":", 1)[0]
    while current not in visited:
        visited.add(current)
        try:
            entity = catalog.entity(current)
        except KeyError:
            return current
        links = tuple(
            link for link in entity.topology_links if link.startswith(f"{kind}:")
        )
        if len(links) != 1:
            return current
        current = links[0]
    return logical_id


def _topology_links(
    cad: Any,
    logical_entities: Mapping[str, tuple[Any, ...]],
    face_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    edge_by_entity = {
        entity: logical_id
        for logical_id, entities in logical_entities.items()
        if logical_id.startswith("edge:")
        for entity in entities
    }
    point_by_entity = {
        entity: logical_id
        for logical_id, entities in logical_entities.items()
        if logical_id.startswith("point:")
        for entity in entities
    }
    links: dict[str, tuple[str, ...]] = {"body:domain": face_ids}
    for logical_id, entities in logical_entities.items():
        if logical_id.startswith("face:"):
            links[logical_id] = tuple(
                sorted(
                    {
                        edge_by_entity[edge]
                        for face in entities
                        for edge in cad.boundary((face,), combined=False)
                        if edge in edge_by_entity
                    }
                )
            )
        elif logical_id.startswith("edge:"):
            links[logical_id] = tuple(
                sorted(
                    {
                        point_by_entity[point]
                        for edge in entities
                        for point in cad.boundary((edge,), combined=False)
                        if point in point_by_entity
                    }
                )
            )
    return links


def _validate_coverage(
    boundary: tuple[Any, ...],
    logical_entities: Mapping[str, tuple[Any, ...]],
) -> None:
    counts: dict[Any, int] = {}
    for logical_id, entities in logical_entities.items():
        if logical_id == "body:domain":
            continue
        for entity in entities:
            counts[entity] = counts.get(entity, 0) + 1
    if any(counts.get(entity, 0) == 0 for entity in boundary):
        raise PlanarBooleanLineageResolutionError(
            "planar-boolean.lineage.coverage-incomplete: anonymous result entity"
        )
    if any(counts.get(entity, 0) != 1 for entity in boundary):
        raise PlanarBooleanLineageResolutionError(
            "planar-boolean.lineage.catalog-mismatch: result entity has "
            "multiple logical identities"
        )


def _require_group_available(
    grouped: Mapping[str, list[Any]],
    logical_id: str,
    entity: Any,
) -> None:
    if logical_id in grouped and entity not in grouped[logical_id]:
        raise PlanarBooleanLineageResolutionError(
            "planar-boolean.lineage.signature-collision: canonical topology "
            f"signature for {logical_id} is not unique"
        )


def _generated_id(
    cad: Any,
    kind: str,
    feature_id: str,
    group: str,
    entity: Any,
) -> str:
    signature = {
        "dimension": int(entity.dimension),
        "geometry": _geometry_type(cad, entity).casefold(),
        "bounds": [_rounded(value) for value in cad.bounding_box(entity)],
    }
    if entity.dimension == 2:
        signature["area"] = _rounded(cad.area(entity))
    center_query = getattr(cad, "center_of_mass", None)
    if callable(center_query):
        signature["center"] = [_rounded(value) for value in center_query(entity)]
    digest = hashlib.sha256(
        json.dumps(
            signature,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"{kind}:boolean/{feature_id}/{group}/{digest}"


def _rounded(value: Any) -> float:
    return round(float(value), 12)


def _support_matches(
    result: PlanarEntityEvidence,
    source: PlanarEntityEvidence,
) -> bool:
    if result.dimension != source.dimension:
        return False
    result_type = result.geometry_type.casefold()
    source_type = source.geometry_type.casefold()
    if (
        result_type not in {"unknown", f"dimension-{result.dimension}"}
        and source_type not in {"unknown", f"dimension-{source.dimension}"}
        and result_type != source_type
    ):
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
        if result_min < source_min - tolerance or result_max > source_max + tolerance:
            return False
    return True


def _evidence_exact(
    left: PlanarEntityEvidence,
    right: PlanarEntityEvidence,
) -> bool:
    return (
        left.dimension == right.dimension
        and left.geometry_type.casefold() == right.geometry_type.casefold()
        and _bounds_close(left.bounds, right.bounds)
    )


def _bounds_close(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    scale = max(
        *(abs(value) for value in left),
        *(abs(value) for value in right),
        1.0,
    )
    return all(
        math.isclose(a, b, rel_tol=5.0e-7, abs_tol=5.0e-7 * scale)
        for a, b in zip(left, right, strict=True)
    )


def _geometry_type(cad: Any, entity: Any) -> str:
    query = getattr(cad, "geometry_type", None)
    return str(query(entity)) if callable(query) else f"dimension-{entity.dimension}"


def _geometry_direction(
    cad: Any,
    entity: Any,
) -> tuple[float, float, float] | None:
    query = getattr(cad, "geometry_direction", None)
    if not callable(query):
        return None
    value = query(entity)
    return None if value is None else tuple(float(item) for item in value)


def _unique(values) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


__all__ = [
    "PlanarBooleanLineageProof",
    "PlanarBooleanLineageResolutionError",
    "PlanarEntityEvidence",
    "PlanarOperandEvidence",
    "capture_planar_operand_evidence",
    "resolve_planar_boolean_lineage",
    "validate_planar_boolean_input_map",
]
