"""Direct, incremental authoring for accepted native model definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
import math
from typing import Protocol

from fem.application import (
    BeamOrientation,
    ModelDefinitions,
    NamedRegion,
    RegionAssignment,
    ScopedDefinitionBatch,
    SectionDefinition,
)
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    EdgeLoad,
    MaterialDefinition,
    OutputRequest,
)

from .authoring import AuthoringContext, ModelPatch
from .definition_authoring import (
    build_eccentric_plate_scopes,
    definition_state_operations,
)
from .naming import NamePolicy


DIRECT_DEFINITION_ACTIONS = frozenset(
    {
        "create_plate_scopes",
        "create_material",
        "create_section",
        "assign_section",
        "create_static_step",
        "create_boundary_condition",
        "create_load",
        "create_result_request",
    }
)


class _Snapshot(Protocol):
    session_id: str
    session_revision: int
    source_kind: str | None
    named_regions: Mapping[str, NamedRegion]
    materials: Sequence[object]
    sections: Sequence[object]
    assignments: Sequence[object]
    steps: Sequence[object]
    artifact: object | None
    model_current: bool
    runs: Sequence[object]


def create_incremental_definition_patch(
    *,
    patch_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    snapshot: _Snapshot,
    draft_revision: int,
    action: object,
    parameters: object,
) -> ModelPatch:
    """Build one immediate patch for one definition-authoring action."""

    normalized_action = _exact_action(action)
    values = _mapping(parameters, "parameters")
    _require_live_native_context(context, snapshot)

    regions = tuple(snapshot.named_regions.values())
    materials = tuple(snapshot.materials)
    sections = tuple(snapshot.sections)
    assignments = tuple(snapshot.assignments)
    steps = deepcopy(tuple(snapshot.steps))
    created_names: tuple[str, ...]

    if normalized_action == "create_plate_scopes":
        _exact_fields(values, set())
        scopes = build_eccentric_plate_scopes(snapshot)
        existing = {item.name for item in regions}
        conflicts = existing.intersection(item.name for item in scopes.regions)
        if conflicts:
            raise ValueError("plate scope names already exist")
        regions = regions + scopes.regions
        created_names = tuple(item.name for item in scopes.regions)
    elif normalized_action == "create_material":
        _exact_fields(values, {"name", "properties"})
        name = _controlled_name(values["name"], "material name", "材料")
        if name in {str(item.name) for item in materials}:
            raise ValueError("material name already exists")
        properties = dict(_mapping(values["properties"], "material properties"))
        if not properties:
            raise ValueError("material properties must not be empty")
        materials = materials + (MaterialDefinition(name, properties),)
        created_names = (name,)
    elif normalized_action == "create_section":
        planar_fields = {
            "name", "material", "plane_type", "thickness", "properties"
        }
        line_fields = {"name", "material", "section_type", "properties"}
        supplied_fields = frozenset(values)
        if supplied_fields not in {
            frozenset(planar_fields),
            frozenset(line_fields),
        }:
            raise ValueError(
                "section requires either planar properties or strict line properties"
            )
        name = _controlled_name(values["name"], "section name", "截面")
        material = _nonblank(values["material"], "section material")
        if name in {str(item.name) for item in sections}:
            raise ValueError("section name already exists")
        if material not in {str(item.name) for item in materials}:
            raise ValueError("section material does not exist")
        if set(values) == planar_fields:
            compatibility_properties = _mapping(
                values["properties"],
                "section properties",
            )
            if compatibility_properties:
                raise ValueError("planar section properties must be empty")
            plane_type = _enum(
                values["plane_type"],
                "plane_type",
                {"stress", "strain"},
            )
            section_type = "solid"
            section_properties = {
                "plane_type": plane_type,
                "thickness": _positive(
                    values["thickness"],
                    "section thickness",
                ),
            }
        else:
            section_type = _enum(
                values["section_type"],
                "section_type",
                {
                    "solid",
                    "truss",
                    "rectangle",
                    "solid_circle",
                    "hollow_circle",
                },
            )
            line_properties = _mapping(
                values["properties"],
                "section properties",
            )
            expected_fields = {
                "solid": set(),
                "truss": {"area"},
                "rectangle": {"height", "width"},
                "solid_circle": {"radius"},
                "hollow_circle": {"outer_radius", "inner_radius"},
            }[section_type]
            _exact_fields(line_properties, expected_fields)
            section_properties = {
                field: _positive(
                    line_properties[field],
                    f"{section_type} section {field}",
                )
                for field in expected_fields
            }
            if (
                section_type == "hollow_circle"
                and section_properties["inner_radius"]
                >= section_properties["outer_radius"]
            ):
                raise ValueError(
                    "hollow_circle inner_radius must be smaller than outer_radius"
                )
        sections = sections + (
            SectionDefinition(name, material, section_type, section_properties),
        )
        created_names = (name,)
    elif normalized_action == "assign_section":
        supplied = set(values)
        if supplied not in (
            {"section_name", "region_name"},
            {"section_name", "region_name", "local_y_reference"},
        ):
            raise ValueError(
                "section assignment fields do not match the strict schema"
            )
        section_name = _nonblank(values["section_name"], "section name")
        region_name = _nonblank(values["region_name"], "region name")
        if section_name not in {str(item.name) for item in sections}:
            raise ValueError("assigned section does not exist")
        if region_name not in {item.name for item in regions}:
            raise ValueError("assignment region does not exist")
        orientation = (
            None
            if "local_y_reference" not in values
            else BeamOrientation(
                _vector(values["local_y_reference"], "Beam local-y reference", 3)
            )
        )
        assignment = RegionAssignment(section_name, region_name, orientation)
        if assignment in assignments:
            raise ValueError("section assignment already exists")
        assignments = assignments + (assignment,)
        created_names = (f"{section_name} → {region_name}",)
    elif normalized_action == "create_static_step":
        _exact_fields(values, {"name"})
        name = _controlled_name(values["name"], "analysis step name", "分析步")
        if name in {str(item.name) for item in steps}:
            raise ValueError("analysis step name already exists")
        steps = steps + (
            AnalysisStep(
                name,
                procedure="static",
                metadata={"nlgeom": False},
            ),
        )
        created_names = (name,)
    elif normalized_action == "create_boundary_condition":
        _exact_fields(
            values,
            {
                "name",
                "step_name",
                "target_scope",
                "target_kind",
                "first_component",
                "last_component",
                "value",
            },
        )
        name = _controlled_name(values["name"], "boundary name", "位移")
        step_name = _nonblank(values["step_name"], "analysis step name")
        target_scope = _require_scope(
            regions,
            values["target_scope"],
            values["target_kind"],
        )
        boundary = DisplacementConstraint(
            target_scope,
            _component(values["first_component"], "first_component"),
            _component(values["last_component"], "last_component"),
            _finite(values["value"], "boundary value"),
            str(values["target_kind"]),
            name,
        )
        if boundary.last_component < boundary.first_component:
            raise ValueError("boundary component range is reversed")
        steps = _append_step_child(
            steps,
            step_name,
            "boundaries",
            boundary,
            name,
        )
        created_names = (name,)
    elif normalized_action == "create_load":
        required = {"name", "step_name", "target_scope", "load_type"}
        allowed = required | {"vector", "magnitude"}
        _allowed_fields(values, allowed, required)
        name = _controlled_name(values["name"], "load name", "载荷")
        step_name = _nonblank(values["step_name"], "analysis step name")
        target_scope = _require_scope(regions, values["target_scope"], "edge")
        load_type = _enum(
            values["load_type"],
            "load_type",
            {"edge_traction", "edge_pressure"},
        )
        if load_type == "edge_traction":
            vector = _vector(values.get("vector"), "load vector", 2)
            if "magnitude" in values:
                raise ValueError("edge traction uses vector, not magnitude")
            load = EdgeLoad(target_scope, vector, None, "traction", name)
        else:
            if "vector" in values:
                raise ValueError("edge pressure uses magnitude, not vector")
            load = EdgeLoad(
                target_scope,
                (),
                _finite(values.get("magnitude"), "load magnitude"),
                "pressure",
                name,
            )
        steps = _append_step_child(
            steps,
            step_name,
            "edge_loads",
            load,
            name,
        )
        created_names = (name,)
    else:
        _exact_fields(
            values,
            {"name", "step_name", "target", "variables"},
        )
        name = _controlled_name(
            values["name"],
            "result request name",
            "结果请求",
        )
        step_name = _nonblank(values["step_name"], "analysis step name")
        target = _enum(values["target"], "result target", {"node", "element"})
        variables = _string_tuple(values["variables"], "result variables")
        supported = {"node": {"U", "RF"}, "element": {"S"}}
        if not set(variables).issubset(supported[target]):
            raise ValueError("result variables do not match their target")
        output = OutputRequest("field", target, variables, name=name)
        steps = _append_step_child(
            steps,
            step_name,
            "outputs",
            output,
            name,
        )
        created_names = (name,)

    definitions = ModelDefinitions(materials, sections, assignments, steps)
    result_invalidating = any(
        bool(getattr(run, "has_result", False)) for run in snapshot.runs
    )
    return ModelPatch.create(
        patch_id=patch_id,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=tuple(source_tool_call_ids),
        target_document_id=context.binding.document_id,
        target_session_id=context.binding.session_id,
        base_session_revision=context.binding.session_revision,
        draft_revision=draft_revision,
        operations=definition_state_operations(regions, definitions),
        preconditions={
            "authoring_mode": "direct_incremental",
            "direct_action": normalized_action,
            "source_kind": "native",
            "exact_session_revision": True,
        },
        expected_changes={
            "action": normalized_action,
            "created_names": list(created_names),
        },
        invalidation_impact={
            "model": True,
            "validation": True,
            "results": result_invalidating,
        },
        display_summary={
            "title": f"Agent {_ACTION_SUMMARIES[normalized_action]}",
            "summary": _ACTION_SUMMARIES[normalized_action],
            "objects": list(created_names),
            "undo_label": "撤销修改",
        },
    )


def require_incremental_definition_batch(
    snapshot: _Snapshot,
    batch: ScopedDefinitionBatch,
    action: object,
) -> None:
    """Verify that a direct patch contains exactly its declared addition."""

    normalized_action = _exact_action(action)
    before_regions = tuple(snapshot.named_regions.values())
    before_materials = tuple(snapshot.materials)
    before_sections = tuple(snapshot.sections)
    before_assignments = tuple(snapshot.assignments)
    before_steps = tuple(snapshot.steps)

    if normalized_action == "create_plate_scopes":
        _require_prefix("named regions", before_regions, batch.regions, 4)
        _require_equal_definition_groups(
            snapshot,
            batch,
            skip={"regions"},
        )
        return
    if normalized_action == "create_material":
        _require_prefix("materials", before_materials, batch.materials, 1)
        _require_equal_definition_groups(
            snapshot,
            batch,
            skip={"materials"},
        )
        return
    if normalized_action == "create_section":
        _require_prefix("sections", before_sections, batch.sections, 1)
        _require_equal_definition_groups(
            snapshot,
            batch,
            skip={"sections"},
        )
        return
    if normalized_action == "assign_section":
        _require_prefix(
            "assignments",
            before_assignments,
            batch.assignments,
            1,
        )
        _require_equal_definition_groups(
            snapshot,
            batch,
            skip={"assignments"},
        )
        return
    if normalized_action == "create_static_step":
        _require_prefix("analysis steps", before_steps, batch.steps, 1)
        _require_equal_definition_groups(
            snapshot,
            batch,
            skip={"steps"},
        )
        return

    _require_equal_definition_groups(snapshot, batch, skip={"steps"})
    if len(batch.steps) != len(before_steps):
        raise ValueError("direct step-child patch cannot add or remove a step")
    field = {
        "create_boundary_condition": "boundaries",
        "create_load": "edge_loads",
        "create_result_request": "outputs",
    }[normalized_action]
    changed = 0
    for before, after in zip(before_steps, batch.steps, strict=True):
        if before == after:
            continue
        changed += 1
        for name in (
            "name",
            "procedure",
            "metadata",
            "boundaries",
            "cloads",
            "surface_loads",
            "outputs",
            "edge_loads",
            "line_loads",
            "body_loads",
            "gravity_loads",
        ):
            before_value = getattr(before, name)
            after_value = getattr(after, name)
            if name == field:
                _require_prefix(
                    f"step {field}",
                    tuple(before_value),
                    tuple(after_value),
                    1,
                )
            elif after_value != before_value:
                raise ValueError(
                    f"direct {normalized_action} patch changed step field {name}"
                )
    if changed != 1:
        raise ValueError("direct step-child patch must change exactly one step")


_ACTION_SUMMARIES = {
    "create_plate_scopes": "已创建当前平板的语义作用域",
    "create_material": "已创建材料",
    "create_section": "已创建截面",
    "assign_section": "已创建截面指派",
    "create_static_step": "已创建线性静力分析步",
    "create_boundary_condition": "已创建边界条件",
    "create_load": "已创建载荷",
    "create_result_request": "已创建结果请求",
}


def _append_step_child(
    steps: Sequence[object],
    step_name: str,
    field: str,
    child: object,
    child_name: str,
) -> tuple[object, ...]:
    matches = [item for item in steps if getattr(item, "name", None) == step_name]
    if len(matches) != 1:
        raise ValueError("analysis step is missing or ambiguous")
    step = matches[0]
    for collection_name in (
        "boundaries",
        "cloads",
        "edge_loads",
        "surface_loads",
        "line_loads",
        "body_loads",
        "gravity_loads",
        "outputs",
    ):
        if any(
            getattr(item, "name", None) == child_name
            for item in tuple(getattr(step, collection_name))
        ):
            raise ValueError("step child name already exists")
    return tuple(
        replace(
            item,
            **{field: tuple(getattr(item, field)) + (child,)},
        )
        if item is step
        else item
        for item in steps
    )


def _require_live_native_context(
    context: AuthoringContext,
    snapshot: _Snapshot,
) -> None:
    binding = context.binding
    if (
        snapshot.source_kind != "native"
        or snapshot.artifact is None
        or not snapshot.model_current
        or binding.source_kind != "native"
        or binding.session_id != snapshot.session_id
        or binding.document_id != f"document:{snapshot.session_id}"
        or binding.session_revision != snapshot.session_revision
    ):
        raise ValueError("direct authoring context is stale or unavailable")


def _require_scope(
    regions: Sequence[NamedRegion],
    value: object,
    expected_kind: object,
) -> str:
    name = _nonblank(value, "target scope")
    kind = _enum(expected_kind, "target kind", {"node", "edge", "face", "element"})
    matches = [item for item in regions if item.name == name]
    if len(matches) != 1 or matches[0].entity_kind != kind:
        raise ValueError("target scope is missing or has the wrong entity kind")
    return name


def _require_equal_definition_groups(
    snapshot: _Snapshot,
    batch: ScopedDefinitionBatch,
    *,
    skip: set[str],
) -> None:
    groups = {
        "regions": (
            tuple(snapshot.named_regions.values()),
            tuple(batch.regions),
        ),
        "materials": (tuple(snapshot.materials), tuple(batch.materials)),
        "sections": (tuple(snapshot.sections), tuple(batch.sections)),
        "assignments": (
            tuple(snapshot.assignments),
            tuple(batch.assignments),
        ),
        "steps": (tuple(snapshot.steps), tuple(batch.steps)),
    }
    for label, (before, after) in groups.items():
        if label not in skip and after != before:
            raise ValueError(f"direct patch changed unrelated {label}")


def _require_prefix(
    label: str,
    before: Sequence[object],
    after: Sequence[object],
    added: int,
) -> None:
    if tuple(after[: len(before)]) != tuple(before) or len(after) != len(before) + added:
        raise ValueError(f"direct patch must append exactly {added} {label}")


def _exact_action(value: object) -> str:
    if type(value) is not str or value not in DIRECT_DEFINITION_ACTIONS:
        raise ValueError("unsupported direct definition action")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError("parameters do not match the selected action")


def _allowed_fields(
    value: Mapping[str, object],
    allowed: set[str],
    required: set[str],
) -> None:
    if not required.issubset(value) or set(value) - allowed:
        raise ValueError("parameters do not match the selected action")


def _nonblank(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value.strip()


def _controlled_name(
    value: object,
    label: str,
    object_type: str,
) -> str:
    name = NamePolicy().validate(_nonblank(value, label))
    if not name.startswith(f"{object_type}-"):
        raise ValueError(f"{label} must use the {object_type}- prefix")
    return name


def _enum(value: object, label: str, allowed: set[str]) -> str:
    normalized = _nonblank(value, label)
    if normalized not in allowed:
        raise ValueError(f"{label} is unsupported")
    return normalized


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def _positive(value: object, label: str) -> float:
    normalized = _finite(value, label)
    if normalized <= 0.0:
        raise ValueError(f"{label} must be positive")
    return normalized


def _component(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= 6:
        raise ValueError(f"{label} must be an integer from 1 to 6")
    return value


def _vector(value: object, label: str, size: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{label} must be an array")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{label} must be an array") from error
    if len(values) != size:
        raise ValueError(f"{label} must contain {size} values")
    return tuple(_finite(item, label) for item in values)


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{label} must be an array")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{label} must be an array") from error
    if not values or any(type(item) is not str or not item for item in values):
        raise ValueError(f"{label} must contain non-blank strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")
    return values


__all__ = [
    "DIRECT_DEFINITION_ACTIONS",
    "create_incremental_definition_patch",
    "require_incremental_definition_batch",
]
