"""Strict production actions for incremental native model definitions.

This module is the active A4/A5 boundary adapter.  It keeps each user-requested
definition change small and reversible while retaining the original strict
engineering checks for dimensions, directions, signs, units, scopes, and
result invalidation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Protocol

from fem.application import (
    ModelDefinitions,
    NamedRegion,
    RegionRef,
    ScopedDefinitionBatch,
    UnitContext,
    describe_model_capabilities,
    describe_region_capabilities,
    validate_logical_reference,
)
from fem.application.native_scope_materialization import (
    mesh_references_for_logical_entities,
)
from fem.core.model import (
    DisplacementConstraint,
    EdgeLoad,
    NodalLoad,
    OutputRequest,
    SurfaceLoad,
)
from fem.geometry import (
    LogicalEntityRef,
    geometry_dimension,
    namespace_part_logical_id,
)

from .analysis_authoring import (
    AnalysisAuthoringError,
    ConfirmedDisplacement,
    ConfirmedLoad,
    ConfirmedResultRequest,
)
from .authoring import (
    AgentProposal,
    AuthoringContext,
    ModelPatch,
    ProposalKind,
)
from .definition_authoring import definition_state_operations
from .incremental_authoring import create_incremental_definition_patch
from .naming import NamePolicy


STRICT_DEFINITION_ACTIONS = frozenset(
    {
        "create_named_region",
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
    active_part_id: str | None
    parts: Sequence[object]
    named_regions: Mapping[str, NamedRegion]
    materials: Sequence[object]
    sections: Sequence[object]
    assignments: Sequence[object]
    steps: Sequence[object]
    artifact: object | None
    model_current: bool
    runs: Sequence[object]
    unit_context: UnitContext | None


@dataclass(frozen=True, slots=True)
class DefinitionChange:
    """One strict direct patch or its result-invalidating confirmation proposal."""

    value: ModelPatch | AgentProposal
    action: str
    resume_object_type: str


def create_definition_change(
    *,
    patch_id: str,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    snapshot: _Snapshot,
    draft_revision: int,
    action: object,
    parameters: object,
) -> DefinitionChange:
    """Create one strict A4/A5 change and gate accepted-result invalidation."""

    normalized_action = _action(action)
    values = _mapping(parameters, "parameters")
    _require_live_native_context(context, snapshot)
    _require_unit_context(context, snapshot)

    if normalized_action == "create_named_region":
        patch = _create_named_region_patch(
            patch_id=patch_id,
            agent_session_id=agent_session_id,
            turn_id=turn_id,
            source_tool_call_ids=source_tool_call_ids,
            context=context,
            snapshot=snapshot,
            draft_revision=draft_revision,
            values=values,
        )
        resume_object_type = "named_region"
    elif normalized_action in {
        "create_boundary_condition",
        "create_load",
        "create_result_request",
    }:
        patch = _create_analysis_child_patch(
            patch_id=patch_id,
            agent_session_id=agent_session_id,
            turn_id=turn_id,
            source_tool_call_ids=source_tool_call_ids,
            context=context,
            snapshot=snapshot,
            draft_revision=draft_revision,
            action=normalized_action,
            values=values,
        )
        resume_object_type = "analysis_step"
    else:
        _validate_non_analysis_action(normalized_action, values, snapshot)
        patch = create_incremental_definition_patch(
            patch_id=patch_id,
            agent_session_id=agent_session_id,
            turn_id=turn_id,
            source_tool_call_ids=source_tool_call_ids,
            context=context,
            snapshot=snapshot,
            draft_revision=draft_revision,
            action=normalized_action,
            parameters=values,
        )
        resume_object_type = (
            "analysis_step"
            if normalized_action == "create_static_step"
            else "named_region"
        )

    if patch.invalidation_impact.get("results") is not True:
        return DefinitionChange(patch, normalized_action, resume_object_type)
    proposal = AgentProposal.create(
        proposal_id=proposal_id,
        proposal_kind=ProposalKind.DESTRUCTIVE_EDIT,
        agent_session_id=patch.agent_session_id,
        turn_id=patch.turn_id,
        source_tool_call_ids=patch.source_tool_call_ids,
        target_document_id=patch.target_document_id,
        target_session_id=patch.target_session_id,
        base_session_revision=patch.base_session_revision,
        draft_revision=patch.draft_revision,
        operations=patch.operations,
        preconditions={
            **patch.preconditions,
            "accepted_result_confirmation_required": True,
        },
        expected_changes=patch.expected_changes,
        invalidation_impact=patch.invalidation_impact,
        display_summary={
            **patch.display_summary,
            "title": "定义修改将使已有结果失效",
            "impact": "已有验证、作业和结果将失效",
            "confirm_label": "确认修改",
        },
    )
    return DefinitionChange(proposal, normalized_action, resume_object_type)


def require_strict_definition_batch(
    snapshot: _Snapshot,
    batch: ScopedDefinitionBatch,
    action: object,
) -> None:
    """Verify one strict custom action before the Session owns its values."""

    normalized_action = _action(action)
    before_regions = tuple(snapshot.named_regions.values())
    if normalized_action == "create_named_region":
        _require_appended("named region", before_regions, batch.regions)
        _require_unchanged_groups(snapshot, batch, skip={"regions"})
        return
    if normalized_action not in {
        "create_boundary_condition",
        "create_load",
        "create_result_request",
    }:
        raise ValueError("strict custom batch action is unsupported")

    _require_unchanged_groups(snapshot, batch, skip={"steps"})
    before_steps = tuple(snapshot.steps)
    if len(before_steps) != 1 or len(batch.steps) != 1:
        raise ValueError("strict analysis patch requires exactly one step")
    before = before_steps[0]
    after = batch.steps[0]
    expected_fields = {
        "create_boundary_condition": {"boundaries"},
        "create_load": {"cloads", "edge_loads", "surface_loads"},
        "create_result_request": {"outputs"},
    }[normalized_action]
    changed_fields: set[str] = set()
    for field in (
        "name",
        "procedure",
        "boundaries",
        "cloads",
        "surface_loads",
        "outputs",
        "metadata",
        "edge_loads",
        "line_loads",
        "body_loads",
        "gravity_loads",
    ):
        before_value = getattr(before, field)
        after_value = getattr(after, field)
        if before_value != after_value:
            changed_fields.add(field)
            if field not in expected_fields:
                raise ValueError(
                    f"strict {normalized_action} patch changed step field {field}"
                )
            _require_appended(
                f"step {field}",
                tuple(before_value),
                tuple(after_value),
            )
    if len(changed_fields) != 1:
        raise ValueError(
            "strict analysis patch must append exactly one requested object"
        )


def _create_named_region_patch(
    *,
    patch_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    snapshot: _Snapshot,
    draft_revision: int,
    values: Mapping[str, object],
) -> ModelPatch:
    _exact_fields(
        values,
        {"name", "part_id", "logical_ids", "mesh_kind", "expected_count"},
    )
    mesh_kind = _enum(
        values["mesh_kind"],
        "mesh_kind",
        {"node", "edge", "face", "element"},
    )
    name = _region_name(values["name"], mesh_kind)
    if name in snapshot.named_regions:
        raise ValueError("named region already exists")
    part_id = _nonblank(values["part_id"], "part_id")
    part = _one_part(snapshot, part_id)
    recipe = getattr(part, "geometry_recipe", None)
    if recipe is None:
        raise ValueError("selected Part has no geometry recipe")
    logical_ids = _string_tuple(values["logical_ids"], "logical_ids")
    logical_references: list[LogicalEntityRef] = []
    for logical_id in logical_ids:
        local = LogicalEntityRef(logical_id)
        validate_logical_reference(recipe, local, require_exact=True)
        logical_references.append(
            LogicalEntityRef(namespace_part_logical_id(part_id, logical_id))
        )
    model = getattr(snapshot.artifact, "model", None)
    if model is None:
        raise ValueError("current native artifact has no model")
    references = mesh_references_for_logical_entities(
        model,
        logical_references,
        mesh_kind=mesh_kind,
    )
    expected_count = _positive_int(values["expected_count"], "expected_count")
    if len(references) != expected_count:
        raise ValueError(
            "scope selection count does not match the explicit expected_count"
        )
    regions = tuple(snapshot.named_regions.values()) + (
        NamedRegion(name, references),
    )
    definitions = _definitions(snapshot)
    return _patch(
        patch_id=patch_id,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=source_tool_call_ids,
        context=context,
        snapshot=snapshot,
        draft_revision=draft_revision,
        action="create_named_region",
        regions=regions,
        definitions=definitions,
        created_names=(name,),
        details={
            "part_id": part_id,
            "logical_ids": list(logical_ids),
            "mesh_kind": mesh_kind,
            "matched_count": len(references),
        },
    )


def _create_analysis_child_patch(
    *,
    patch_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    snapshot: _Snapshot,
    draft_revision: int,
    action: str,
    values: Mapping[str, object],
) -> ModelPatch:
    steps = deepcopy(tuple(snapshot.steps))
    if len(steps) != 1:
        raise ValueError("analysis definition actions require exactly one static step")
    step_name = _controlled_name(
        values.get("step_name"),
        "analysis step name",
        "分析步",
    )
    if steps[0].name != step_name:
        raise ValueError("analysis step is missing or ambiguous")
    if steps[0].procedure != "static" or steps[0].metadata.get("nlgeom") is not False:
        raise ValueError("analysis actions require one linear static NLGEOM=false step")
    dimension = _part_dimension(snapshot)
    dofs_per_node = _model_dofs_per_node(snapshot)
    units = snapshot.unit_context
    assert units is not None

    if action == "create_boundary_condition":
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
                "unit",
                "distribution",
                "confirmed",
            },
        )
        confirmed = ConfirmedDisplacement(
            _controlled_name(values["name"], "boundary name", "位移"),
            step_name,
            _require_scope(
                snapshot.named_regions.values(),
                values["target_scope"],
                values["target_kind"],
            ),
            _enum(
                values["target_kind"],
                "target_kind",
                {"node_set", "edge", "surface"},
            ),
            _component(values["first_component"], "first_component"),
            _component(values["last_component"], "last_component"),
            _finite(values["value"], "boundary value"),
            _nonblank(values["unit"], "boundary unit"),
            _enum(values["distribution"], "distribution", {"uniform"}),
            _confirmed(values["confirmed"]),
        )
        if confirmed.last_component > dofs_per_node:
            raise AnalysisAuthoringError(
                "displacement DOF exceeds the current model capability"
            )
        if confirmed.unit != units.length:
            raise AnalysisAuthoringError(
                "displacement unit must exactly match the project length unit"
            )
        child = DisplacementConstraint(
            confirmed.target_scope,
            confirmed.first_component,
            confirmed.last_component,
            confirmed.value,
            confirmed.entity_type,
            confirmed.name,
        )
        steps = _append_step_child(
            steps,
            "boundaries",
            child,
            confirmed.name,
        )
        details = confirmed.summary()
        created_name = confirmed.name
    elif action == "create_load":
        required = {
            "name",
            "step_name",
            "target_scope",
            "entity_type",
            "load_type",
            "component",
            "vector",
            "magnitude",
            "direction",
            "unit",
            "distribution",
            "confirmed",
        }
        _exact_fields(values, required)
        load_type = _enum(
            values["load_type"],
            "load_type",
            {
                "nodal",
                "edge_traction",
                "edge_pressure",
                "surface_traction",
                "surface_pressure",
            },
        )
        entity_type = _enum(
            values["entity_type"],
            "entity_type",
            {"node", "edge", "surface"},
        )
        target_kind = "node" if entity_type == "node" else entity_type
        confirmed = ConfirmedLoad(
            _controlled_name(values["name"], "load name", "载荷"),
            step_name,
            _require_scope(
                snapshot.named_regions.values(),
                values["target_scope"],
                target_kind,
            ),
            entity_type,
            load_type,
            _optional_component(values["component"]),
            _optional_vector(values["vector"]),
            _optional_finite(values["magnitude"], "load magnitude"),
            _nonblank(values["direction"], "load direction"),
            _nonblank(values["unit"], "load unit"),
            _nonblank(values["distribution"], "load distribution"),
            _confirmed(values["confirmed"]),
        )
        _validate_load_dimension_and_unit(
            confirmed,
            dimension,
            dofs_per_node,
            units,
        )
        field, child = _load_child(confirmed)
        steps = _append_step_child(steps, field, child, confirmed.name)
        details = confirmed.summary()
        created_name = confirmed.name
    else:
        _exact_fields(
            values,
            {
                "name",
                "step_name",
                "target",
                "variables",
                "units",
                "confirmed",
            },
        )
        confirmed = ConfirmedResultRequest(
            _controlled_name(
                values["name"],
                "result request name",
                "结果请求",
            ),
            step_name,
            "field",
            _enum(values["target"], "result target", {"node", "element"}),
            _string_tuple(values["variables"], "result variables"),
            _string_tuple(values["units"], "result units"),
            _confirmed(values["confirmed"]),
        )
        expected_units = {
            "U": units.length,
            "RF": units.force,
            "S": units.stress,
        }
        expected = tuple(expected_units[item] for item in confirmed.variables)
        if confirmed.units != expected:
            raise AnalysisAuthoringError(
                "result request units do not match the project unit context"
            )
        child = OutputRequest(
            "field",
            confirmed.target,
            confirmed.variables,
            name=confirmed.name,
        )
        steps = _append_step_child(
            steps,
            "outputs",
            child,
            confirmed.name,
        )
        details = confirmed.summary()
        created_name = confirmed.name

    return _patch(
        patch_id=patch_id,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=source_tool_call_ids,
        context=context,
        snapshot=snapshot,
        draft_revision=draft_revision,
        action=action,
        regions=tuple(snapshot.named_regions.values()),
        definitions=ModelDefinitions(
            tuple(snapshot.materials),
            tuple(snapshot.sections),
            tuple(snapshot.assignments),
            steps,
        ),
        created_names=(created_name,),
        details=details,
    )


def _validate_non_analysis_action(
    action: str,
    values: Mapping[str, object],
    snapshot: _Snapshot,
) -> None:
    if action == "create_material":
        _exact_fields(values, {"name", "properties"})
        properties = _mapping(values["properties"], "material properties")
        if set(properties) - {"E", "nu", "density"} or not {"E", "nu"} <= set(
            properties
        ):
            raise ValueError(
                "linear elastic material requires exactly E, nu, and optional density"
            )
        if _finite(properties["E"], "Young modulus") <= 0.0:
            raise ValueError("Young modulus must be positive")
        poisson = _finite(properties["nu"], "Poisson ratio")
        if not -1.0 < poisson < 0.5:
            raise ValueError("Poisson ratio must be between -1 and 0.5")
        if "density" in properties and _finite(
            properties["density"], "density"
        ) <= 0.0:
            raise ValueError("density must be positive")
        return
    if action == "create_section":
        planar_fields = {
            "name", "material", "plane_type", "thickness", "properties"
        }
        truss_fields = {"name", "material", "section_type", "properties"}
        if set(values) == planar_fields:
            return
        if set(values) != truss_fields:
            raise ValueError(
                "section requires either planar properties or a truss area"
            )
        if values["section_type"] != "truss":
            raise ValueError("line section_type must be truss")
        properties = _mapping(values["properties"], "section properties")
        _exact_fields(properties, {"area"})
        if _finite(properties["area"], "truss section area") <= 0.0:
            raise ValueError("truss section area must be positive")
        _require_elastic_material(snapshot, values["material"])
        return
    if action == "assign_section":
        _exact_fields(values, {"section_name", "region_name"})
        _require_supported_section_assignment(snapshot, values)
        return
    if action == "create_static_step":
        _exact_fields(values, {"name"})
        if snapshot.steps:
            raise ValueError("first milestone supports exactly one analysis step")
        return
    raise ValueError("unsupported strict definition action")


def _patch(
    *,
    patch_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    snapshot: _Snapshot,
    draft_revision: int,
    action: str,
    regions: Sequence[NamedRegion],
    definitions: ModelDefinitions,
    created_names: Sequence[str],
    details: Mapping[str, object],
) -> ModelPatch:
    result_invalidating = any(
        bool(getattr(run, "has_result", False)) for run in snapshot.runs
    )
    summary = {
        "create_named_region": "已创建作用域",
        "create_boundary_condition": "已创建边界条件",
        "create_load": "已创建载荷",
        "create_result_request": "已创建结果请求",
    }[action]
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
            "authoring_mode": "strict_incremental",
            "direct_action": action,
            "source_kind": "native",
            "exact_session_revision": True,
            "engineering_fields_confirmed": True,
        },
        expected_changes={
            "action": action,
            "created_names": list(created_names),
            "details": dict(details),
        },
        invalidation_impact={
            "model": True,
            "validation": True,
            "results": result_invalidating,
        },
        display_summary={
            "title": f"Agent {summary}",
            "summary": summary,
            "objects": list(created_names),
            "details": dict(details),
            "undo_label": "撤销修改",
        },
    )


def _definitions(snapshot: _Snapshot) -> ModelDefinitions:
    return ModelDefinitions(
        tuple(snapshot.materials),
        tuple(snapshot.sections),
        tuple(snapshot.assignments),
        tuple(snapshot.steps),
    )


def _append_step_child(
    steps: Sequence[object],
    field: str,
    child: object,
    name: str,
) -> tuple[object, ...]:
    step = steps[0]
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
            getattr(item, "name", None) == name
            for item in tuple(getattr(step, collection_name))
        ):
            raise ValueError("analysis object name already exists")
    return (
        replace(
            step,
            **{field: tuple(getattr(step, field)) + (child,)},
        ),
    )


def _load_child(load: ConfirmedLoad) -> tuple[str, object]:
    if load.load_type == "nodal":
        return (
            "cloads",
            NodalLoad(
                load.target_scope,
                int(load.component),
                float(load.magnitude),
                load.name,
            ),
        )
    if load.load_type.startswith("edge_"):
        return (
            "edge_loads",
            EdgeLoad(
                load.target_scope,
                load.vector,
                load.magnitude,
                load.load_type.removeprefix("edge_"),
                load.name,
            ),
        )
    return (
        "surface_loads",
        SurfaceLoad(
            load.target_scope,
            load.vector,
            load.magnitude,
            load.load_type.removeprefix("surface_"),
            load.name,
        ),
    )


def _validate_load_dimension_and_unit(
    load: ConfirmedLoad,
    dimension: int,
    dofs_per_node: int,
    units: UnitContext,
) -> None:
    if load.load_type == "nodal":
        if int(load.component or 0) > dofs_per_node:
            raise AnalysisAuthoringError(
                "nodal load component exceeds the current model capability"
            )
        expected_direction = {
            1: "global_x",
            2: "global_y",
            3: "global_z",
        }[int(load.component)]
        if load.direction != expected_direction:
            raise AnalysisAuthoringError(
                "nodal load direction does not match its component"
            )
        expected_unit = units.force
    elif load.load_type.startswith("edge_"):
        if dimension != 2:
            raise AnalysisAuthoringError(
                "edge load requires a two-dimensional Part"
            )
        if (
            load.load_type == "edge_traction"
            and load.direction != "global_xy"
        ):
            raise AnalysisAuthoringError(
                "two-dimensional traction direction must be global_xy"
            )
        expected_unit = f"{units.force}/{units.length}"
    else:
        if dimension != 3:
            raise AnalysisAuthoringError(
                "surface load requires a three-dimensional Part"
            )
        if (
            load.load_type == "surface_traction"
            and load.direction != "global_xyz"
        ):
            raise AnalysisAuthoringError(
                "three-dimensional traction direction must be global_xyz"
            )
        expected_unit = units.stress
    if load.vector and len(load.vector) != dimension:
        raise AnalysisAuthoringError(
            "load vector dimension does not match the current Part"
        )
    if load.unit != expected_unit:
        raise AnalysisAuthoringError(
            "load unit does not match the project unit context"
        )


def _part_dimension(snapshot: _Snapshot) -> int:
    part = _one_part(snapshot, snapshot.active_part_id)
    recipe = getattr(part, "geometry_recipe", None)
    if recipe is None:
        raise AnalysisAuthoringError(
            "analysis Part dimension must be explicitly two or three"
        )
    return int(geometry_dimension(recipe))


def _model_dofs_per_node(snapshot: _Snapshot) -> int:
    model = getattr(snapshot.artifact, "model", None)
    if model is None:
        raise AnalysisAuthoringError(
            "analysis requires a current realized model capability"
        )
    report = describe_model_capabilities(model)
    if not report.compatible or report.dofs_per_node is None:
        raise AnalysisAuthoringError(
            "analysis model has no compatible nodal DOF capability"
        )
    return int(report.dofs_per_node)


def _require_elastic_material(snapshot: _Snapshot, value: object) -> None:
    material_name = _nonblank(value, "section material")
    matches = [
        material
        for material in snapshot.materials
        if str(getattr(material, "name", "")) == material_name
    ]
    if len(matches) != 1:
        raise ValueError("section material does not exist")
    properties = _mapping(
        getattr(matches[0], "properties", None),
        "material properties",
    )
    if "E" not in properties or "nu" not in properties:
        raise ValueError("truss section requires an elastic material")
    if _finite(properties["E"], "Young modulus") <= 0.0:
        raise ValueError("Young modulus must be positive")
    poisson = _finite(properties["nu"], "Poisson ratio")
    if not -1.0 < poisson < 0.5:
        raise ValueError("Poisson ratio must be between -1 and 0.5")


def _require_supported_section_assignment(
    snapshot: _Snapshot,
    values: Mapping[str, object],
) -> None:
    section_name = _nonblank(values["section_name"], "section name")
    matches = [
        section
        for section in snapshot.sections
        if str(getattr(section, "name", "")) == section_name
    ]
    if len(matches) != 1:
        raise ValueError("assigned section does not exist")
    if str(getattr(matches[0], "section_type", "")).casefold() != "truss":
        return
    region_name = _nonblank(values["region_name"], "region name")
    model = getattr(snapshot.artifact, "model", None)
    if model is None:
        raise ValueError("truss assignment requires a current realized model")
    capability = describe_region_capabilities(
        model,
        RegionRef("element_set", region_name),
    )
    if (
        not capability.compatible
        or not capability.homogeneous
        or capability.canonical_element_types != ("Truss2",)
    ):
        raise ValueError(
            "truss section can target only a non-empty pure Truss2 element region"
        )


def _one_part(snapshot: _Snapshot, part_id: object) -> object:
    clean_id = _nonblank(part_id, "part_id")
    matches = [
        part for part in snapshot.parts if getattr(part, "id", None) == clean_id
    ]
    if len(matches) != 1:
        raise ValueError("selected Part is missing or ambiguous")
    return matches[0]


def _require_scope(
    regions: Sequence[NamedRegion],
    value: object,
    expected_kind: object,
) -> str:
    name = _nonblank(value, "target scope")
    kind = _enum(
        expected_kind,
        "target kind",
        {"node", "node_set", "edge", "surface", "face", "element"},
    )
    normalized_kind = {"node_set": "node", "surface": "face"}.get(kind, kind)
    matches = [item for item in regions if item.name == name]
    if len(matches) != 1 or matches[0].entity_kind != normalized_kind:
        raise ValueError("target scope is missing or has the wrong entity kind")
    return name


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
        raise ValueError("definition authoring context is stale or unavailable")


def _require_unit_context(
    context: AuthoringContext,
    snapshot: _Snapshot,
) -> None:
    units = snapshot.unit_context
    summary = context.unit_context
    if units is None or summary is None:
        raise AnalysisAuthoringError(
            "definition authoring requires the current project unit context"
        )
    if (
        summary.length != units.length
        or summary.force != units.force
        or summary.stress != units.stress
    ):
        raise AnalysisAuthoringError("project unit context is stale")


def _action(value: object) -> str:
    if type(value) is not str or value not in STRICT_DEFINITION_ACTIONS:
        raise ValueError("unsupported strict definition action")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError("parameters do not match the selected action")


def _nonblank(value: object, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-blank trimmed string")
    return value


def _controlled_name(value: object, label: str, object_type: str) -> str:
    name = NamePolicy().validate(_nonblank(value, label))
    if not name.startswith(f"{object_type}-"):
        raise ValueError(f"{label} must use the {object_type}- prefix")
    return name


def _region_name(value: object, mesh_kind: str) -> str:
    object_type = {
        "node": "点",
        "edge": "边",
        "face": "面",
        "element": "域",
    }[mesh_kind]
    return _controlled_name(value, "region name", object_type)


def _enum(value: object, label: str, allowed: set[str]) -> str:
    normalized = _nonblank(value, label)
    if normalized not in allowed:
        raise ValueError(f"{label} is unsupported")
    return normalized


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    normalized = float(value)
    if not (-float("inf") < normalized < float("inf")):
        raise ValueError(f"{label} must be a finite number")
    return normalized


def _optional_finite(value: object, label: str) -> float | None:
    return None if value is None else _finite(value, label)


def _component(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= 3:
        raise ValueError(f"{label} must be an integer from 1 to 3")
    return value


def _optional_component(value: object) -> int | None:
    return None if value is None else _component(value, "load component")


def _optional_vector(value: object) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("load vector must be an array or null")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("load vector must be an array or null") from error
    if len(values) not in {2, 3}:
        raise ValueError("load vector must contain two or three values")
    return tuple(_finite(item, "load vector component") for item in values)


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{label} must be an array")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{label} must be an array") from error
    if (
        not values
        or any(
            type(item) is not str
            or not item.strip()
            or item != item.strip()
            for item in values
        )
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{label} must contain unique non-blank strings")
    return values


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _confirmed(value: object) -> bool:
    if value is not True:
        raise AnalysisAuthoringError(
            "engineering definition fields must be explicitly confirmed"
        )
    return True


def _require_unchanged_groups(
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
        if label not in skip and before != after:
            raise ValueError(f"strict patch changed unrelated {label}")


def _require_appended(
    label: str,
    before: Sequence[object],
    after: Sequence[object],
) -> None:
    if tuple(after[:-1]) != tuple(before) or len(after) != len(before) + 1:
        raise ValueError(f"strict patch must append exactly one {label}")


__all__ = [
    "DefinitionChange",
    "STRICT_DEFINITION_ACTIONS",
    "create_definition_change",
    "require_strict_definition_batch",
]
