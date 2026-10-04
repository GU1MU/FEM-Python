"""A4 stable scopes and reversible material/section authoring."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Protocol

from fem.application import (
    BeamOrientation,
    MeshEntityRef,
    ModelDefinitions,
    NamedRegion,
    RegionAssignment,
    ScopedDefinitionBatch,
    SectionDefinition,
)
from fem.core.model import (
    AnalysisStep,
    BodyForce,
    DisplacementConstraint,
    EdgeLoad,
    GravityLoad,
    LineLoad,
    MaterialDefinition,
    NodalLoad,
    OutputRequest,
    SurfaceLoad,
)
from fem.geometry import LogicalEntityRef

from .authoring import (
    ModelOperation,
    OperationKind,
)


class _Snapshot(Protocol):
    named_regions: Mapping[str, NamedRegion]
    materials: Sequence[MaterialDefinition]
    sections: Sequence[SectionDefinition]
    assignments: Sequence[RegionAssignment]
    steps: Sequence[object]


def scoped_definition_batch_from_operations(
    operations: Sequence[ModelOperation],
    *,
    base_session_revision: int,
) -> ScopedDefinitionBatch:
    """Decode the closed A4 operation pair into one application command."""

    values = tuple(operations)
    if tuple(item.kind for item in values) != (
        OperationKind.UPSERT_NAMED_REGIONS,
        OperationKind.UPSERT_MODEL_DEFINITIONS,
    ):
        raise ValueError(
            "A4 changes require exact region and definition post-state operations"
        )
    regions = _decode_regions(values[0].parameters["regions"])
    definitions = _decode_definitions(
        values[1].parameters["definitions"],
    )
    return ScopedDefinitionBatch(
        base_session_revision,
        regions,
        definitions.materials,
        definitions.sections,
        definitions.assignments,
        definitions.steps,
    )


def inverse_operations_for_snapshot(
    snapshot: _Snapshot,
) -> tuple[ModelOperation, ModelOperation]:
    """Return the exact A4 pre-state as closed inverse operations."""

    return definition_state_operations(
        tuple(snapshot.named_regions.values()),
        ModelDefinitions(
            tuple(snapshot.materials),
            tuple(snapshot.sections),
            tuple(snapshot.assignments),
            tuple(snapshot.steps),
        ),
    )


def definition_state_operations(
    regions: Sequence[NamedRegion],
    definitions: ModelDefinitions,
) -> tuple[ModelOperation, ModelOperation]:
    return (
        ModelOperation(
            OperationKind.UPSERT_NAMED_REGIONS,
            {"regions": _encode_regions(regions)},
        ),
        ModelOperation(
            OperationKind.UPSERT_MODEL_DEFINITIONS,
            {"definitions": _encode_definitions(definitions)},
        ),
    )


def _encode_regions(
    regions: Sequence[NamedRegion],
) -> list[dict[str, object]]:
    return [
        {
            "name": region.name,
            "references": [
                _encode_reference(reference)
                for reference in region.references
            ],
        }
        for region in regions
    ]


def _encode_reference(
    reference: MeshEntityRef | LogicalEntityRef,
) -> dict[str, object]:
    if type(reference) is LogicalEntityRef:
        return {
            "reference_type": "logical",
            "logical_id": reference.logical_id,
        }
    if type(reference) is not MeshEntityRef:
        raise TypeError("unsupported scope reference")
    return {
        "reference_type": "mesh",
        "kind": reference.kind,
        "node_id": reference.node_id,
        "element_id": reference.element_id,
        "local_index": reference.local_index,
        "node_ids": list(reference.node_ids),
        "part_id": reference.part_id,
    }


def _decode_regions(value: object) -> tuple[NamedRegion, ...]:
    rows = _require_list(value, "regions")
    regions: list[NamedRegion] = []
    for row in rows:
        data = _require_mapping(row, "region")
        _require_keys(data, {"name", "references"}, "region")
        references = tuple(
            _decode_reference(item)
            for item in _require_list(
                data["references"],
                "region references",
            )
        )
        regions.append(NamedRegion(str(data["name"]), references))
    return tuple(regions)


def _decode_reference(
    value: object,
) -> MeshEntityRef | LogicalEntityRef:
    data = _require_mapping(value, "scope reference")
    reference_type = data.get("reference_type")
    if reference_type == "logical":
        _require_keys(
            data,
            {"reference_type", "logical_id"},
            "logical reference",
        )
        return LogicalEntityRef(str(data["logical_id"]))
    if reference_type != "mesh":
        raise ValueError("scope reference type is unknown")
    _require_keys(
        data,
        {
            "reference_type",
            "kind",
            "node_id",
            "element_id",
            "local_index",
            "node_ids",
            "part_id",
        },
        "mesh reference",
    )
    return MeshEntityRef(
        kind=str(data["kind"]),
        node_id=data["node_id"],
        element_id=data["element_id"],
        local_index=data["local_index"],
        node_ids=tuple(_require_list(data["node_ids"], "mesh node_ids")),
        part_id=(
            None
            if data["part_id"] is None
            else str(data["part_id"])
        ),
    )


def _encode_definitions(
    definitions: ModelDefinitions,
) -> dict[str, object]:
    return {
        "materials": [
            {
                "name": material.name,
                "properties": deepcopy(dict(material.properties)),
            }
            for material in definitions.materials
        ],
        "sections": [
            {
                "name": section.name,
                "material": section.material,
                "section_type": section.section_type,
                "properties": deepcopy(dict(section.properties)),
            }
            for section in definitions.sections
        ],
        "assignments": [
            _encode_assignment(assignment)
            for assignment in definitions.assignments
        ],
        "steps": [_encode_step(step) for step in definitions.steps],
    }


def _encode_assignment(assignment: RegionAssignment) -> dict[str, object]:
    encoded: dict[str, object] = {
        "section_name": assignment.section_name,
        "region_name": assignment.region_name,
    }
    if assignment.beam_orientation is not None:
        encoded["beam_orientation"] = list(
            assignment.beam_orientation.local_y_reference
        )
    return encoded


def _decode_definitions(
    value: object,
) -> ModelDefinitions:
    data = _require_mapping(value, "definitions")
    fields = frozenset(data)
    if fields != {"materials", "sections", "assignments", "steps"}:
        raise ValueError(
            "definitions fields do not match the strict A4/A5 schema"
        )
    materials = []
    for row in _require_list(data["materials"], "materials"):
        item = _require_mapping(row, "material")
        _require_keys(item, {"name", "properties"}, "material")
        materials.append(
            MaterialDefinition(
                str(item["name"]),
                deepcopy(dict(_require_mapping(
                    item["properties"],
                    "material properties",
                ))),
            )
        )
    sections = []
    for row in _require_list(data["sections"], "sections"):
        item = _require_mapping(row, "section")
        _require_keys(
            item,
            {
                "name",
                "material",
                "section_type",
                "properties",
            },
            "section",
        )
        sections.append(
            SectionDefinition(
                str(item["name"]),
                str(item["material"]),
                str(item["section_type"]),
                deepcopy(dict(_require_mapping(
                    item["properties"],
                    "section properties",
                ))),
            )
        )
    assignments = []
    for row in _require_list(data["assignments"], "assignments"):
        item = _require_mapping(row, "assignment")
        fields = set(item)
        if fields not in (
            {"section_name", "region_name"},
            {"section_name", "region_name", "beam_orientation"},
        ):
            raise ValueError(
                "assignment fields do not match the strict A4/A5 schema"
            )
        has_orientation = "beam_orientation" in item
        orientation = item.get("beam_orientation")
        assignments.append(
            RegionAssignment(
                str(item["section_name"]),
                str(item["region_name"]),
                (
                    None
                    if not has_orientation
                    else BeamOrientation(
                        _strict_number_array(
                            orientation,
                            "beam orientation",
                        )
                    )
                ),
            )
        )
    return ModelDefinitions(
        tuple(materials),
        tuple(sections),
        tuple(assignments),
        tuple(
            _decode_step(row)
            for row in _require_list(data["steps"], "steps")
        ),
    )


def _encode_step(step: object) -> dict[str, object]:
    if type(step) is not AnalysisStep:
        raise TypeError("analysis step must be exactly AnalysisStep")
    return {
        "name": step.name,
        "procedure": step.procedure,
        "metadata": deepcopy(dict(step.metadata)),
        "boundaries": [
            {
                "name": item.name,
                "target": item.target,
                "target_kind": item.target_kind,
                "first_component": item.first_component,
                "last_component": item.last_component,
                "value": item.value,
            }
            for item in step.boundaries
        ],
        "cloads": [
            {
                "name": item.name,
                "target": item.target,
                "component": item.component,
                "value": item.value,
            }
            for item in step.cloads
        ],
        "edge_loads": [
            {
                "name": item.name,
                "edge": item.edge,
                "vector": list(item.vector),
                "magnitude": item.magnitude,
                "load_type": item.load_type,
            }
            for item in step.edge_loads
        ],
        "surface_loads": [
            {
                "name": item.name,
                "surface": item.surface,
                "vector": list(item.vector),
                "magnitude": item.magnitude,
                "load_type": item.load_type,
            }
            for item in step.surface_loads
        ],
        "line_loads": [
            {
                "name": item.name,
                "target": item.target,
                "vector": list(item.vector),
                "coordinate_system": item.coordinate_system,
            }
            for item in step.line_loads
        ],
        "body_loads": [
            {
                "name": item.name,
                "target": item.target,
                "vector": list(item.vector),
            }
            for item in step.body_loads
        ],
        "gravity_loads": [
            {
                "name": item.name,
                "target": item.target,
                "acceleration": list(item.acceleration),
            }
            for item in step.gravity_loads
        ],
        "outputs": [
            {
                "name": item.name,
                "kind": item.kind,
                "target": item.target,
                "variables": list(item.variables),
                "metadata": deepcopy(dict(item.metadata)),
            }
            for item in step.outputs
        ],
    }


def _decode_step(value: object) -> AnalysisStep:
    data = _require_mapping(value, "analysis step")
    _require_keys(
        data,
        {
            "name",
            "procedure",
            "metadata",
            "boundaries",
            "cloads",
            "edge_loads",
            "surface_loads",
            "line_loads",
            "body_loads",
            "gravity_loads",
            "outputs",
        },
        "analysis step",
    )
    return AnalysisStep(
        name=_strict_string(data["name"], "analysis step name"),
        procedure=_strict_string(data["procedure"], "analysis procedure"),
        metadata=deepcopy(
            dict(_require_mapping(data["metadata"], "step metadata"))
        ),
        boundaries=tuple(
            _decode_boundary(item)
            for item in _require_list(data["boundaries"], "boundaries")
        ),
        cloads=tuple(
            _decode_cload(item)
            for item in _require_list(data["cloads"], "cloads")
        ),
        edge_loads=tuple(
            _decode_edge_load(item)
            for item in _require_list(data["edge_loads"], "edge_loads")
        ),
        surface_loads=tuple(
            _decode_surface_load(item)
            for item in _require_list(
                data["surface_loads"],
                "surface_loads",
            )
        ),
        line_loads=tuple(
            _decode_line_load(item)
            for item in _require_list(data["line_loads"], "line_loads")
        ),
        body_loads=tuple(
            _decode_body_load(item)
            for item in _require_list(data["body_loads"], "body_loads")
        ),
        gravity_loads=tuple(
            _decode_gravity_load(item)
            for item in _require_list(
                data["gravity_loads"],
                "gravity_loads",
            )
        ),
        outputs=tuple(
            _decode_output(item)
            for item in _require_list(data["outputs"], "outputs")
        ),
    )


def _decode_boundary(value: object) -> DisplacementConstraint:
    data = _require_mapping(value, "boundary")
    _require_keys(
        data,
        {
            "name",
            "target",
            "target_kind",
            "first_component",
            "last_component",
            "value",
        },
        "boundary",
    )
    return DisplacementConstraint(
        _strict_target(data["target"], "boundary target"),
        _strict_integer(data["first_component"], "first_component"),
        _strict_integer(data["last_component"], "last_component"),
        _strict_number(data["value"], "boundary value"),
        _strict_string(data["target_kind"], "boundary target_kind"),
        _optional_name(data["name"]),
    )


def _decode_cload(value: object) -> NodalLoad:
    data = _require_mapping(value, "nodal load")
    _require_keys(
        data,
        {"name", "target", "component", "value"},
        "nodal load",
    )
    return NodalLoad(
        _strict_target(data["target"], "nodal load target"),
        _strict_integer(data["component"], "nodal load component"),
        _strict_number(data["value"], "nodal load value"),
        _optional_name(data["name"]),
    )


def _decode_edge_load(value: object) -> EdgeLoad:
    data = _require_mapping(value, "edge load")
    _require_keys(
        data,
        {"name", "edge", "vector", "magnitude", "load_type"},
        "edge load",
    )
    return EdgeLoad(
        _strict_string(data["edge"], "edge load target"),
        _strict_number_array(data["vector"], "edge load vector"),
        (
            None
            if data["magnitude"] is None
            else _strict_number(data["magnitude"], "edge load magnitude")
        ),
        _strict_string(data["load_type"], "edge load type"),
        _optional_name(data["name"]),
    )


def _decode_surface_load(value: object) -> SurfaceLoad:
    data = _require_mapping(value, "surface load")
    _require_keys(
        data,
        {"name", "surface", "vector", "magnitude", "load_type"},
        "surface load",
    )
    return SurfaceLoad(
        _strict_string(data["surface"], "surface load target"),
        _strict_number_array(data["vector"], "surface load vector"),
        (
            None
            if data["magnitude"] is None
            else _strict_number(
                data["magnitude"],
                "surface load magnitude",
            )
        ),
        _strict_string(data["load_type"], "surface load type"),
        _optional_name(data["name"]),
    )


def _decode_line_load(value: object) -> LineLoad:
    data = _require_mapping(value, "line load")
    _require_keys(
        data,
        {"name", "target", "vector", "coordinate_system"},
        "line load",
    )
    return LineLoad(
        _strict_target(data["target"], "line load target"),
        _strict_number_array(data["vector"], "line load vector"),
        _strict_string(data["coordinate_system"], "coordinate system"),
        _optional_name(data["name"]),
    )


def _decode_body_load(value: object) -> BodyForce:
    data = _require_mapping(value, "body load")
    _require_keys(data, {"name", "target", "vector"}, "body load")
    return BodyForce(
        _strict_target(data["target"], "body load target"),
        _strict_number_array(data["vector"], "body load vector"),
        _optional_name(data["name"]),
    )


def _decode_gravity_load(value: object) -> GravityLoad:
    data = _require_mapping(value, "gravity load")
    _require_keys(
        data,
        {"name", "target", "acceleration"},
        "gravity load",
    )
    return GravityLoad(
        _strict_number_array(data["acceleration"], "gravity acceleration"),
        (
            None
            if data["target"] is None
            else _strict_target(data["target"], "gravity target")
        ),
        _optional_name(data["name"]),
    )


def _decode_output(value: object) -> OutputRequest:
    data = _require_mapping(value, "output request")
    _require_keys(
        data,
        {"name", "kind", "target", "variables", "metadata"},
        "output request",
    )
    return OutputRequest(
        _strict_string(data["kind"], "output kind"),
        _strict_string(data["target"], "output target"),
        tuple(
            _strict_string(item, "output variable")
            for item in _require_list(data["variables"], "variables")
        ),
        deepcopy(dict(_require_mapping(data["metadata"], "output metadata"))),
        None,
        _optional_name(data["name"]),
    )


def _optional_name(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value.strip():
        raise ValueError("analysis object name must be null or nonblank string")
    return value


def _strict_string(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{label} must be a nonblank exact string")
    return value


def _strict_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    return value


def _strict_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or type(value) not in {int, float}
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"{label} must be a finite JSON number")
    return float(value)


def _strict_target(value: object, label: str) -> str | int:
    if type(value) is int:
        return value
    return _strict_string(value, label)


def _strict_number_array(value: object, label: str) -> tuple[float, ...]:
    return tuple(
        _strict_number(item, f"{label}[{index}]")
        for index, item in enumerate(_require_list(value, label))
    )


def _require_mapping(
    value: object,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return value


def _require_keys(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields do not match the strict A4/A5 schema"
        )


__all__ = [
    "definition_state_operations",
    "inverse_operations_for_snapshot",
    "scoped_definition_batch_from_operations",
]
