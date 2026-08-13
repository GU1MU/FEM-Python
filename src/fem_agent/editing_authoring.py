"""Bounded catalogs and direct edits for accepted model definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from typing import Protocol

from fem.application import (
    DefinitionEditBatch,
    MeshEntityRef,
    ModelSession,
    NamedRegion,
    RegionAssignment,
    RenameIntent,
    ScopedDefinitionBatch,
    SectionDefinition,
    UnitContext,
    RegionRef,
    describe_model_capabilities,
    describe_region_capabilities,
    evaluate_native_line_load_candidate,
    validate_logical_reference,
)
from fem.application.results import project_output_request
from fem.application.changes import SessionDelta
from fem.application.native_scope_materialization import (
    mesh_references_for_logical_entities,
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
from fem.geometry import (
    LogicalEntityRef,
    geometry_dimension,
    namespace_part_logical_id,
)

from .analysis_authoring import (
    ConfirmedDisplacement,
    ConfirmedLoad,
    ConfirmedResultRequest,
    expected_result_units,
)
from .authoring import (
    AgentProposal,
    AuthoringContext,
    ModelPatch,
    ModelOperation,
    OperationKind,
    ProposalKind,
)
from .naming import NamePolicy


_LOAD_COLLECTIONS = (
    ("cloads", NodalLoad, "nodal"),
    ("edge_loads", EdgeLoad, "edge"),
    ("surface_loads", SurfaceLoad, "surface"),
    ("line_loads", LineLoad, "line"),
    ("body_loads", BodyForce, "body"),
    ("gravity_loads", GravityLoad, "gravity"),
)
_STEP_REFERENCE_FIELDS = (
    ("boundaries", "target"),
    ("cloads", "target"),
    ("edge_loads", "edge"),
    ("surface_loads", "surface"),
    ("line_loads", "target"),
    ("body_loads", "target"),
    ("gravity_loads", "target"),
)
_EDIT_TYPES = frozenset(
    {
        "named_region",
        "material",
        "section",
        "section_assignment",
        "analysis_step",
        "boundary_condition",
        "load",
        "result_request",
    }
)
_STEP_CHILD_EDIT_TYPES = frozenset(
    {"boundary_condition", "load", "result_request"}
)


class _Snapshot(Protocol):
    source_kind: str | None
    session_revision: int
    named_regions: Mapping[str, NamedRegion]
    materials: object
    sections: object
    assignments: object
    steps: object
    artifact: object | None
    parts: Sequence[object]
    active_part_id: str | None
    unit_context: UnitContext | None
    runs: Sequence[object]


@dataclass(frozen=True, slots=True)
class EditableObject:
    object_type: str
    target_id: str
    display_name: str
    step_name: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.object_type not in _EDIT_TYPES:
            raise ValueError("unsupported editable object type")
        for field_name in ("target_id", "display_name"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field_name} must be a non-blank string")
            normalized = value.strip()
            if len(normalized) > 160:
                raise ValueError(f"{field_name} must be at most 160 characters")
            object.__setattr__(self, field_name, normalized)
        if self.object_type in _STEP_CHILD_EDIT_TYPES:
            if type(self.step_name) is not str or not self.step_name.strip():
                raise ValueError("step_name is required for step child targets")
            normalized_step = self.step_name.strip()
            if len(normalized_step) > 160:
                raise ValueError("step_name must be at most 160 characters")
            object.__setattr__(self, "step_name", normalized_step)
        elif self.step_name is not None:
            raise ValueError("step_name is only valid for step child targets")
        object.__setattr__(self, "details", deepcopy(dict(self.details)))

    def to_provider_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "object_type": self.object_type,
            "target_id": self.target_id,
            "display_name": self.display_name,
            "details": deepcopy(dict(self.details)),
        }
        if self.step_name is not None:
            result["step_name"] = self.step_name
        return result


def editable_object_catalog(
    snapshot: _Snapshot,
    *,
    limit: int = 100,
) -> tuple[EditableObject, ...]:
    """Return bounded native definition objects with stable edit identities."""

    if type(limit) is not int or isinstance(limit, bool) or not 1 <= limit <= 128:
        raise ValueError("limit must be an integer between 1 and 128")
    if (
        getattr(snapshot, "source_kind", None) != "native"
        or getattr(snapshot, "artifact", None) is None
    ):
        return ()

    items: list[EditableObject] = []
    for region in tuple(snapshot.named_regions.values()):
        if len(region.name) > 160:
            continue
        items.append(
            EditableObject(
                "named_region",
                region.name,
                region.name,
                details={
                    "entity_kind": region.entity_kind,
                    "reference_keys": [
                        _reference_key(reference)
                        for reference in region.references
                    ],
                    "reference_count": len(region.references),
                    "editable_fields": [
                        "new_name",
                        "reference_keys",
                        "part_id",
                        "logical_ids",
                        "mesh_kind",
                        "expected_count",
                    ],
                },
            )
        )

    for material in tuple(snapshot.materials):
        name = _optional_name(material)
        if name is None or len(name) > 160:
            continue
        items.append(
            EditableObject(
                "material",
                name,
                name,
                details={
                    "properties": _provider_mapping(material.properties),
                    "editable_fields": ["new_name", "properties"],
                },
            )
        )

    for section in tuple(snapshot.sections):
        name = _optional_name(section)
        if name is None or len(name) > 160:
            continue
        items.append(
            EditableObject(
                "section",
                name,
                name,
                details={
                    "material": _bounded_text(section.material),
                    "section_type": _bounded_text(section.section_type),
                    "properties": _provider_mapping(section.properties),
                    "editable_fields": [
                        "new_name",
                        "material",
                        "section_type",
                        "properties",
                    ],
                },
            )
        )

    assignment_region_counts: dict[str, int] = {}
    for assignment in tuple(snapshot.assignments):
        region_name = str(getattr(assignment, "region_name", "")).strip()
        if region_name:
            assignment_region_counts[region_name] = (
                assignment_region_counts.get(region_name, 0) + 1
            )
    for assignment in tuple(snapshot.assignments):
        region_name = str(getattr(assignment, "region_name", "")).strip()
        if (
            not region_name
            or len(region_name) > 160
            or assignment_region_counts.get(region_name) != 1
        ):
            continue
        items.append(
            EditableObject(
                "section_assignment",
                region_name,
                region_name,
                details={
                    "region_name": region_name,
                    "section_name": _bounded_text(assignment.section_name),
                    "editable_fields": ["region_name", "section_name"],
                },
            )
        )

    units = getattr(snapshot, "unit_context", None)
    for step in tuple(snapshot.steps):
        step_name = str(getattr(step, "name", "")).strip()
        if not step_name or len(step_name) > 160:
            continue
        items.append(
            EditableObject(
                "analysis_step",
                step_name,
                step_name,
                details={
                    "procedure": _bounded_text(
                        getattr(step, "procedure", "static")
                    ),
                    "metadata": _provider_mapping(
                        getattr(step, "metadata", {})
                    ),
                    "boundary_count": len(tuple(step.boundaries)),
                    "load_count": sum(
                        len(tuple(getattr(step, collection_name, ())))
                        for collection_name, _expected, _kind in _LOAD_COLLECTIONS
                    ),
                    "result_request_count": len(tuple(step.outputs)),
                    "editable_fields": [
                        "new_name",
                        "procedure",
                        "metadata",
                    ],
                },
            )
        )
        for output in tuple(getattr(step, "outputs", ())):
            name = _optional_name(output)
            if name is None or len(name) > 160:
                continue
            details: dict[str, object] = {
                "output_kind": _bounded_text(output.kind),
                "target": _bounded_text(output.target),
                "variables": [
                    _bounded_text(value) for value in tuple(output.variables)[:32]
                ],
                "metadata": _provider_mapping(output.metadata),
                "source_evidence_present": output.source_evidence is not None,
                "editable_fields": [
                    "new_name",
                    "output_kind",
                    "kind",
                    "target",
                    "variables",
                    "metadata",
                    "units",
                    "confirmed",
                ],
            }
            canonical_variables = tuple(output.variables)
            if set(canonical_variables).issubset(
                {"U", "UR", "RF", "RM", "SF", "SM", "LE", "S"}
            ):
                units = _require_units(snapshot)
                details["units"] = list(
                    expected_result_units(units, canonical_variables)
                )
                details["confirmed"] = True
            items.append(
                EditableObject(
                    "result_request",
                    name,
                    name,
                    step_name,
                    details,
                )
            )
        if type(units) is not UnitContext:
            continue
        for boundary in tuple(getattr(step, "boundaries", ())):
            name = _optional_name(boundary)
            if name is None or len(name) > 160:
                continue
            items.append(
                EditableObject(
                    "boundary_condition",
                    name,
                    name,
                    step_name,
                    {
                        "target_scope": boundary.target,
                        "target_kind": boundary.target_kind,
                        "first_component": boundary.first_component,
                        "last_component": boundary.last_component,
                        "value": boundary.value,
                        "unit": units.length,
                        "distribution": "uniform",
                        "confirmed": True,
                        "editable_fields": [
                            "new_name",
                            "target_scope",
                            "first_component",
                            "last_component",
                            "value",
                            "unit",
                            "distribution",
                            "confirmed",
                        ],
                    },
                )
            )
        for collection_name, expected_type, load_kind in _LOAD_COLLECTIONS:
            for load in tuple(getattr(step, collection_name, ())):
                if type(load) is not expected_type:
                    continue
                name = _optional_name(load)
                if name is None or len(name) > 160:
                    continue
                items.append(
                    EditableObject(
                        "load",
                        name,
                        name,
                        step_name,
                        _load_details(snapshot, load, load_kind),
                    )
                )

    counts: dict[tuple[str, str, str | None], int] = {}
    for item in items:
        identity = (item.object_type, item.target_id, item.step_name)
        counts[identity] = counts.get(identity, 0) + 1
    unique = (
        item
        for item in items
        if counts[(item.object_type, item.target_id, item.step_name)] == 1
    )
    return tuple(
        sorted(
            unique,
            key=lambda item: (
                item.object_type,
                "" if item.step_name is None else item.step_name,
                item.target_id,
            ),
        )[:limit]
    )


def create_edit_proposal(
    *,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    snapshot: _Snapshot,
    draft_revision: int,
    object_type: object,
    target_id: object,
    changes: object,
    step_name: object = None,
) -> tuple[AgentProposal, EditableObject]:
    """Create one revision-bound edit proposal without mutating the Session."""

    target = _resolve_editable_object(
        snapshot,
        object_type,
        target_id,
        step_name,
    )
    normalized_changes, _replacement = _validated_edit(
        snapshot,
        target,
        changes,
    )
    parameters: dict[str, object] = {
        "object_type": target.object_type,
        "target_id": target.target_id,
        "changes": normalized_changes,
    }
    if target.step_name is not None:
        parameters["step_name"] = target.step_name
    label = {
        "named_region": "作用域",
        "material": "材料",
        "section": "截面",
        "section_assignment": "截面指派",
        "analysis_step": "分析步",
        "boundary_condition": "边界条件",
        "load": "载荷",
        "result_request": "结果请求",
    }[target.object_type]
    changed_fields = "、".join(normalized_changes)
    proposal = AgentProposal.create(
        proposal_id=proposal_id,
        proposal_kind=ProposalKind.DESTRUCTIVE_EDIT,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=source_tool_call_ids,
        target_document_id=context.binding.document_id,
        target_session_id=context.binding.session_id,
        base_session_revision=context.binding.session_revision,
        draft_revision=draft_revision,
        operations=(
            ModelOperation(OperationKind.EDIT_MODEL_OBJECT, parameters),
        ),
        preconditions={
            "source_kind": "native",
            "target_exists": True,
            "exact_session_revision": True,
        },
        expected_changes={
            "edited_object_type": target.object_type,
            "edited_target_id": target.target_id,
            "changed_fields": list(normalized_changes),
        },
        invalidation_impact={
            "model": True,
            "validation": True,
            "results": False,
            "historical_results_retained": True,
            "current_validation_reset": True,
            "current_result_display_reset": True,
        },
        display_summary={
            "title": f"编辑{label}：{target.display_name}",
            "summary": f"修改{label}“{target.display_name}”的{changed_fields}",
            "impact": "修改后需重新预检；历史作业和结果继续保留",
            "confirm_label": "确认修改",
        },
    )
    return proposal, target


def create_edit_patch(
    *,
    patch_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    snapshot: _Snapshot,
    draft_revision: int,
    object_type: object,
    target_id: object,
    changes: object,
    step_name: object = None,
) -> tuple[ModelPatch, EditableObject]:
    """Create one revision-bound patch for an immediate supported edit."""

    proposal, target = create_edit_proposal(
        proposal_id=f"validated-{patch_id}",
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=source_tool_call_ids,
        context=context,
        snapshot=snapshot,
        draft_revision=draft_revision,
        object_type=object_type,
        target_id=target_id,
        changes=changes,
        step_name=step_name,
    )
    label = {
        "named_region": "作用域",
        "material": "材料",
        "section": "截面",
        "section_assignment": "截面指派",
        "analysis_step": "分析步",
        "boundary_condition": "边界条件",
        "load": "载荷",
        "result_request": "结果请求",
    }[target.object_type]
    patch = ModelPatch.create(
        patch_id=patch_id,
        agent_session_id=proposal.agent_session_id,
        turn_id=proposal.turn_id,
        source_tool_call_ids=proposal.source_tool_call_ids,
        target_document_id=proposal.target_document_id,
        target_session_id=proposal.target_session_id,
        base_session_revision=proposal.base_session_revision,
        draft_revision=proposal.draft_revision,
        operations=proposal.operations,
        preconditions={
            **dict(proposal.preconditions),
            "authoring_mode": "direct_edit",
        },
        expected_changes=proposal.expected_changes,
        invalidation_impact={
            **proposal.invalidation_impact,
            "results": False,
        },
        display_summary={
            "title": f"Agent 已编辑{label}",
            "summary": str(proposal.display_summary["summary"]),
            "objects": [target.display_name],
            "undo_label": "撤销本次 Agent 修改",
        },
    )
    return patch, target


def apply_edit_operation(
    session: ModelSession,
    operation: ModelOperation,
    *,
    base_session_revision: int,
) -> SessionDelta:
    """Apply one confirmed edit as an atomic scoped-definition post-state."""

    if type(session) is not ModelSession:
        raise TypeError("session must be exactly ModelSession")
    if (
        type(operation) is not ModelOperation
        or operation.kind is not OperationKind.EDIT_MODEL_OBJECT
    ):
        raise TypeError("operation must be EDIT_MODEL_OBJECT")
    snapshot = session.snapshot()
    parameters = operation.parameters
    target = _resolve_editable_object(
        snapshot,
        parameters["object_type"],
        parameters["target_id"],
        parameters.get("step_name"),
    )
    _normalized, replacement = _validated_edit(
        snapshot,
        target,
        parameters["changes"],
    )
    regions = tuple(snapshot.named_regions.values())
    materials = tuple(snapshot.materials)
    sections = tuple(snapshot.sections)
    assignments = tuple(snapshot.assignments)
    steps = deepcopy(tuple(snapshot.steps))

    if target.object_type == "named_region":
        assert type(replacement) is NamedRegion
        old_name = target.target_id
        regions = tuple(
            replacement if item.name == old_name else item
            for item in regions
        )
        if replacement.name != old_name:
            assignments = tuple(
                replace(
                    item,
                    region_name=(
                        replacement.name
                        if item.region_name == old_name
                        else item.region_name
                    ),
                )
                for item in assignments
            )
            steps = tuple(
                _rename_step_scope(step, old_name, replacement.name)
                for step in steps
            )
        return session.apply_scoped_definition_batch(
            ScopedDefinitionBatch(
                base_session_revision,
                regions,
                materials,
                sections,
                assignments,
                steps,
            )
        )
    if target.object_type == "material":
        assert type(replacement) is MaterialDefinition
        materials = tuple(
            replacement if item.name == target.target_id else item
            for item in materials
        )
        material_renames = (
            (RenameIntent(target.target_id, replacement.name),)
            if replacement.name != target.target_id
            else ()
        )
        return session.apply_definition_edit(
            DefinitionEditBatch(
                base_session_revision,
                materials,
                sections,
                assignments,
                steps,
                material_renames=material_renames,
            )
        )
    if target.object_type == "section":
        assert type(replacement) is SectionDefinition
        sections = tuple(
            replacement if item.name == target.target_id else item
            for item in sections
        )
        section_renames = (
            (RenameIntent(target.target_id, replacement.name),)
            if replacement.name != target.target_id
            else ()
        )
        return session.apply_definition_edit(
            DefinitionEditBatch(
                base_session_revision,
                materials,
                sections,
                assignments,
                steps,
                section_renames=section_renames,
            )
        )
    if target.object_type == "section_assignment":
        assert type(replacement) is RegionAssignment
        assignments = tuple(
            replacement if item.region_name == target.target_id else item
            for item in assignments
        )
    elif target.object_type == "analysis_step":
        steps = tuple(
            replacement if step.name == target.target_id else step
            for step in steps
        )
    else:
        steps = tuple(
            _replace_step_child(step, target, replacement)
            if step.name == target.step_name
            else step
            for step in steps
        )

    return session.apply_definition_edit(
        DefinitionEditBatch(
            base_session_revision,
            materials,
            sections,
            assignments,
            steps,
        )
    )


def _resolve_editable_object(
    snapshot: _Snapshot,
    object_type: object,
    target_id: object,
    step_name: object,
) -> EditableObject:
    normalized_type = _required_text(object_type, "object_type")
    if normalized_type not in _EDIT_TYPES:
        raise ValueError("unsupported edit object_type")
    normalized_target = _required_text(target_id, "target_id")
    normalized_step = (
        None if step_name is None else _required_text(step_name, "step_name")
    )
    matches = tuple(
        item
        for item in editable_object_catalog(snapshot, limit=128)
        if item.object_type == normalized_type
        and item.target_id == normalized_target
        and item.step_name == normalized_step
    )
    if len(matches) != 1:
        raise ValueError("edit target is unavailable or ambiguous")
    return matches[0]


def _validated_edit(
    snapshot: _Snapshot,
    target: EditableObject,
    raw_changes: object,
) -> tuple[dict[str, object], object]:
    if not isinstance(raw_changes, Mapping) or not raw_changes:
        raise ValueError("changes must be a non-empty object")
    changes = dict(raw_changes)
    if target.object_type == "named_region":
        return _validated_region_edit(snapshot, target, changes)
    if target.object_type == "material":
        return _validated_material_edit(snapshot, target, changes)
    if target.object_type == "section":
        return _validated_section_edit(snapshot, target, changes)
    if target.object_type == "section_assignment":
        return _validated_assignment_edit(snapshot, target, changes)
    if target.object_type == "analysis_step":
        return _validated_step_edit(snapshot, target, changes)
    if target.object_type == "boundary_condition":
        return _validated_boundary_edit(snapshot, target, changes)
    if target.object_type == "load":
        return _validated_load_edit(snapshot, target, changes)
    return _validated_output_edit(snapshot, target, changes)


def _validated_region_edit(
    snapshot: _Snapshot,
    target: EditableObject,
    changes: dict[object, object],
) -> tuple[dict[str, object], NamedRegion]:
    topology_fields = {
        "part_id",
        "logical_ids",
        "mesh_kind",
        "expected_count",
    }
    allowed = {"new_name", "reference_keys", *topology_fields}
    _require_change_keys(changes, allowed)
    current = snapshot.named_regions[target.target_id]
    updates: dict[str, object] = {}
    normalized: dict[str, object] = {}
    if "new_name" in changes:
        name = _controlled_name(
            changes["new_name"],
            "new_name",
            {
                "node": "点",
                "edge": "边",
                "face": "面",
                "element": "域",
            }[current.entity_kind],
        )
        updates["name"] = name
        normalized["new_name"] = name
    if "reference_keys" in changes:
        if set(changes) & topology_fields:
            raise ValueError(
                "reference_keys cannot be combined with logical topology fields"
            )
        keys = _string_list(
            changes["reference_keys"],
            "reference_keys",
            max_items=128,
        )
        reference_map = _reference_map(snapshot)
        unknown = tuple(key for key in keys if key not in reference_map)
        if unknown:
            raise ValueError("reference_keys contains an unavailable identity")
        updates["references"] = tuple(reference_map[key] for key in keys)
        normalized["reference_keys"] = list(keys)
    elif set(changes) & topology_fields:
        if not topology_fields <= set(changes):
            raise ValueError(
                "logical topology redirect requires part_id, logical_ids, "
                "mesh_kind, and expected_count"
            )
        part_id = _required_text(changes["part_id"], "part_id")
        part = _require_part(snapshot, part_id)
        recipe = getattr(part, "geometry_recipe", None)
        if recipe is None:
            raise ValueError("selected Part has no geometry recipe")
        mesh_kind = _required_text(changes["mesh_kind"], "mesh_kind")
        if mesh_kind != current.entity_kind:
            raise ValueError(
                "scope redirect must preserve the current mesh entity kind"
            )
        logical_ids = _string_list(
            changes["logical_ids"],
            "logical_ids",
            max_items=128,
        )
        logical_references = []
        for logical_id in logical_ids:
            local = LogicalEntityRef(logical_id)
            validate_logical_reference(recipe, local, require_exact=True)
            logical_references.append(
                LogicalEntityRef(
                    namespace_part_logical_id(part_id, logical_id)
                )
            )
        model = getattr(snapshot.artifact, "model", None)
        if model is None:
            raise ValueError("current native artifact has no model")
        references = mesh_references_for_logical_entities(
            model,
            logical_references,
            mesh_kind=mesh_kind,
        )
        expected_count = _positive_int(
            changes["expected_count"],
            "expected_count",
        )
        if len(references) != expected_count:
            raise ValueError(
                "scope selection count does not match expected_count"
            )
        updates["references"] = references
        normalized.update(
            {
                "part_id": part_id,
                "logical_ids": list(logical_ids),
                "mesh_kind": mesh_kind,
                "expected_count": expected_count,
            }
        )
    replacement = replace(current, **updates)
    if replacement == current:
        raise ValueError("changes do not modify the selected named region")
    return normalized, replacement


def _validated_material_edit(
    snapshot: _Snapshot,
    target: EditableObject,
    changes: dict[object, object],
) -> tuple[dict[str, object], MaterialDefinition]:
    _require_change_keys(changes, {"new_name", "properties"})
    current = _find_named(snapshot.materials, target.target_id, "material")
    updates: dict[str, object] = {}
    normalized: dict[str, object] = {}
    if "new_name" in changes:
        name = _safe_name(changes["new_name"], "new_name")
        updates["name"] = name
        normalized["new_name"] = name
    if "properties" in changes:
        property_updates = _bounded_mapping(
            changes["properties"],
            "properties",
        )
        updates["properties"] = {
            **dict(current.properties),
            **property_updates,
        }
        normalized["properties"] = deepcopy(property_updates)
    replacement = replace(current, **updates)
    if replacement == current:
        raise ValueError("changes do not modify the selected material")
    return normalized, replacement


def _validated_section_edit(
    snapshot: _Snapshot,
    target: EditableObject,
    changes: dict[object, object],
) -> tuple[dict[str, object], SectionDefinition]:
    _require_change_keys(
        changes,
        {"new_name", "material", "section_type", "properties"},
    )
    current = _find_named(snapshot.sections, target.target_id, "section")
    updates: dict[str, object] = {}
    normalized: dict[str, object] = {}
    if "new_name" in changes:
        name = _safe_name(changes["new_name"], "new_name")
        updates["name"] = name
        normalized["new_name"] = name
    if "material" in changes:
        material = _required_text(changes["material"], "material")
        _find_named(snapshot.materials, material, "material")
        updates["material"] = material
        normalized["material"] = material
    if "section_type" in changes:
        section_type = _required_text(
            changes["section_type"], "section_type"
        ).casefold()
        if section_type not in {
            "solid",
            "truss",
            "rectangle",
            "solid_circle",
            "hollow_circle",
        }:
            raise ValueError("section_type is unsupported")
        updates["section_type"] = section_type
        normalized["section_type"] = section_type
    if "properties" in changes:
        property_updates = _bounded_mapping(
            changes["properties"],
            "properties",
        )
        updates["properties"] = {
            **dict(current.properties),
            **property_updates,
        }
        normalized["properties"] = deepcopy(property_updates)
    replacement = replace(current, **updates)
    if replacement == current:
        raise ValueError("changes do not modify the selected section")
    return normalized, replacement


def _validated_assignment_edit(
    snapshot: _Snapshot,
    target: EditableObject,
    changes: dict[object, object],
) -> tuple[dict[str, object], RegionAssignment]:
    _require_change_keys(changes, {"region_name", "section_name"})
    matches = tuple(
        assignment
        for assignment in tuple(snapshot.assignments)
        if assignment.region_name == target.target_id
    )
    if len(matches) != 1:
        raise ValueError("section assignment target is unavailable or ambiguous")
    current = matches[0]
    updates: dict[str, object] = {}
    normalized: dict[str, object] = {}
    if "region_name" in changes:
        region_name = _required_text(changes["region_name"], "region_name")
        region = _require_region(snapshot, region_name)
        if region.entity_kind != "element":
            raise ValueError("assignment region must be an element scope")
        if any(
            assignment is not current
            and assignment.region_name == region_name
            for assignment in tuple(snapshot.assignments)
        ):
            raise ValueError("assignment region already has an assignment")
        updates["region_name"] = region_name
        normalized["region_name"] = region_name
    if "section_name" in changes:
        section_name = _required_text(
            changes["section_name"], "section_name"
        )
        _find_named(snapshot.sections, section_name, "section")
        updates["section_name"] = section_name
        normalized["section_name"] = section_name
    replacement = replace(current, **updates)
    if replacement == current:
        raise ValueError("changes do not modify the selected section assignment")
    return normalized, replacement


def _validated_step_edit(
    snapshot: _Snapshot,
    target: EditableObject,
    changes: dict[object, object],
) -> tuple[dict[str, object], AnalysisStep]:
    _require_change_keys(changes, {"new_name", "procedure", "metadata"})
    current = _require_step(snapshot, target.target_id)
    if type(current) is not AnalysisStep:
        raise ValueError("analysis step target has an unsupported type")
    updates: dict[str, object] = {}
    normalized: dict[str, object] = {}
    if "new_name" in changes:
        name = _safe_name(changes["new_name"], "new_name")
        updates["name"] = name
        normalized["new_name"] = name
    if "procedure" in changes:
        procedure = _required_text(changes["procedure"], "procedure").casefold()
        if procedure != "static":
            raise ValueError("procedure is unsupported")
        updates["procedure"] = procedure
        normalized["procedure"] = procedure
    if "metadata" in changes:
        metadata_updates = _bounded_mapping(changes["metadata"], "metadata")
        updates["metadata"] = {
            **dict(current.metadata),
            **metadata_updates,
        }
        normalized["metadata"] = deepcopy(metadata_updates)
    replacement = replace(current, **updates)
    if replacement == current:
        raise ValueError("changes do not modify the selected analysis step")
    return normalized, replacement


def _validated_output_edit(
    snapshot: _Snapshot,
    target: EditableObject,
    changes: dict[object, object],
) -> tuple[dict[str, object], OutputRequest]:
    _require_change_keys(
        changes,
        {
            "new_name", "output_kind", "kind", "target", "variables",
            "metadata", "units", "confirmed",
        },
    )
    if "output_kind" in changes and "kind" in changes:
        raise ValueError("use either output_kind or kind, not both")
    confirmation_fields = {"units", "confirmed"} & set(changes)
    if confirmation_fields and confirmation_fields != {"units", "confirmed"}:
        raise ValueError(
            "result request edit units and confirmed must be provided together"
        )
    current = _find_output(snapshot, target)
    updates: dict[str, object] = {}
    normalized: dict[str, object] = {}
    if "new_name" in changes:
        name = _safe_name(changes["new_name"], "new_name")
        updates["name"] = name
        normalized["new_name"] = name
    kind_field = "output_kind" if "output_kind" in changes else "kind"
    if kind_field in changes:
        kind = _required_text(changes[kind_field], kind_field).casefold()
        if kind != "field":
            raise ValueError("output_kind is unsupported")
        updates["kind"] = kind
        normalized[kind_field] = kind
    if "target" in changes:
        output_target = _required_text(changes["target"], "target").casefold()
        if output_target not in {"node", "element"}:
            raise ValueError("result request target is unsupported")
        updates["target"] = output_target
        normalized["target"] = output_target
    if "variables" in changes:
        variables = _string_list(changes["variables"], "variables", max_items=16)
        variables = tuple(value.upper() for value in variables)
        updates["variables"] = variables
        normalized["variables"] = list(variables)
    if "metadata" in changes:
        metadata_updates = _bounded_mapping(changes["metadata"], "metadata")
        updates["metadata"] = {
            **dict(current.metadata),
            **metadata_updates,
        }
        normalized["metadata"] = deepcopy(metadata_updates)
    replacement = replace(current, **updates)
    supported = {
        "node": {"U", "UR", "RF", "RM"},
        "element": {"SF", "SM", "LE", "S"},
    }
    if replacement.kind != "field":
        raise ValueError("output_kind is unsupported")
    if replacement.target not in supported:
        raise ValueError("result request target is unsupported")
    if not replacement.variables:
        raise ValueError("result request variables must not be empty")
    if not set(replacement.variables).issubset(supported[replacement.target]):
        raise ValueError("result variables do not match their target")
    _require_output_edit_capability(snapshot, replacement)
    if confirmation_fields:
        units = _string_list(changes["units"], "units", max_items=16)
        confirmed = ConfirmedResultRequest(
            _controlled_name(replacement.name, "result request name", "结果请求"),
            _controlled_name(target.step_name, "step_name", "分析步"),
            replacement.kind,
            replacement.target,
            tuple(replacement.variables),
            units,
            _confirmed(changes["confirmed"]),
        )
        expected = expected_result_units(
            _require_units(snapshot), confirmed.variables
        )
        if confirmed.units != expected:
            raise ValueError(
                "result request units do not match the project unit context"
            )
    if replacement == current:
        raise ValueError("changes do not modify the selected result request")
    return normalized, replacement


def _require_output_edit_capability(
    snapshot: _Snapshot,
    request: OutputRequest,
) -> None:
    model = getattr(snapshot.artifact, "model", None)
    if model is None:
        raise ValueError("result request edit requires a current realized model")
    report = describe_model_capabilities(model)
    catalog = report.output_request_catalog
    if not report.compatible or catalog is None:
        raise ValueError("result request edit requires a compatible catalog")
    projection = project_output_request(request, catalog, request_index=0)
    executable = projection.executable_request
    projected = (
        ()
        if executable is None
        else tuple(
            item.canonical_variable.value
            for item in executable.variables
            if item.canonical_variable is not None
        )
    )
    if (
        projection.diagnostics
        or len(projected) != len(request.variables)
        or set(projected) != set(request.variables)
    ):
        raise ValueError("result request is not fully executable by this model")


def _validated_boundary_edit(
    snapshot: _Snapshot,
    target: EditableObject,
    changes: dict[object, object],
) -> tuple[dict[str, object], DisplacementConstraint]:
    engineering_fields = {"unit", "distribution", "confirmed"}
    allowed = {
        "new_name",
        "target_scope",
        "first_component",
        "last_component",
        "value",
        *engineering_fields,
    }
    _require_change_keys(changes, allowed)
    if not engineering_fields <= set(changes):
        raise ValueError(
            "boundary edit requires explicit unit, distribution, and confirmed"
        )
    current = _find_boundary(snapshot, target)
    updates: dict[str, object] = {}
    normalized: dict[str, object] = {}
    if "new_name" in changes:
        name = _controlled_name(changes["new_name"], "new_name", "位移")
        updates["name"] = name
        normalized["new_name"] = name
    if "target_scope" in changes:
        scope_name = _required_text(changes["target_scope"], "target_scope")
        region = _require_region(snapshot, scope_name)
        updates["target"] = scope_name
        updates["target_kind"] = _boundary_target_kind(region)
        normalized["target_scope"] = scope_name
    for field_name in ("first_component", "last_component"):
        if field_name in changes:
            value = _component(changes[field_name], field_name)
            updates[field_name] = value
            normalized[field_name] = value
    if "value" in changes:
        value = _finite_number(changes["value"], "value")
        updates["value"] = value
        normalized["value"] = value
    replacement = replace(current, **updates)
    if replacement.first_component > replacement.last_component:
        raise ValueError("first_component must not exceed last_component")
    if replacement == current:
        raise ValueError("changes do not modify the selected boundary condition")
    region = _require_region(snapshot, str(replacement.target))
    if _boundary_target_kind(region) != replacement.target_kind:
        raise ValueError(
            "boundary target kind does not match its target scope"
        )
    dimension = _part_dimension_for_region(snapshot, region)
    units = _require_units(snapshot)
    confirmed = ConfirmedDisplacement(
        _controlled_name(replacement.name, "boundary name", "位移"),
        _controlled_name(target.step_name, "step_name", "分析步"),
        str(replacement.target),
        str(replacement.target_kind),
        int(replacement.first_component),
        int(replacement.last_component),
        float(replacement.value),
        _required_text(changes["unit"], "unit"),
        _required_text(changes["distribution"], "distribution"),
        _confirmed(changes["confirmed"]),
    )
    if confirmed.last_component > dimension:
        raise ValueError("boundary component exceeds the current Part dimension")
    if confirmed.unit != units.length:
        raise ValueError(
            "boundary unit must exactly match the project length unit"
        )
    normalized.update(
        {
            "unit": confirmed.unit,
            "distribution": confirmed.distribution,
            "confirmed": True,
        }
    )
    return normalized, replacement


def _validated_load_edit(
    snapshot: _Snapshot,
    target: EditableObject,
    changes: dict[object, object],
) -> tuple[dict[str, object], object]:
    collection_name, current, load_kind = _find_load(snapshot, target)
    del collection_name
    engineering_fields = {
        "entity_type",
        "load_type",
        "direction",
        "unit",
        "distribution",
        "confirmed",
    }
    allowed_by_kind = {
        "nodal": {
            "new_name",
            "target_scope",
            "component",
            "value",
            *engineering_fields,
        },
        "edge": {
            "new_name",
            "target_scope",
            "vector",
            "magnitude",
            *engineering_fields,
        },
        "surface": {
            "new_name",
            "target_scope",
            "vector",
            "magnitude",
            *engineering_fields,
        },
        "line": {
            "new_name",
            "target_scope",
            "vector",
            "coordinate_system",
            "unit",
            "distribution",
            "confirmed",
        },
        "body": {
            "new_name", "target_scope", "vector", "direction", "unit",
            "distribution", "confirmed",
        },
        "gravity": {
            "new_name", "target_scope", "acceleration", "direction", "unit",
            "distribution", "confirmed",
        },
    }
    _require_change_keys(changes, allowed_by_kind[load_kind])
    strict_kind = load_kind in {"nodal", "edge", "surface"}
    if strict_kind and not engineering_fields <= set(changes):
        raise ValueError(
            "load edit requires explicit entity_type, load_type, direction, "
            "unit, distribution, and confirmed"
        )
    extended_required = {
        "line": {"coordinate_system", "unit", "distribution", "confirmed"},
        "body": {"direction", "unit", "distribution", "confirmed"},
        "gravity": {"direction", "unit", "distribution", "confirmed"},
    }
    if (
        load_kind in extended_required
        and not extended_required[load_kind] <= set(changes)
    ):
        raise ValueError(
            f"{load_kind} load edit requires explicit engineering fields"
        )
    updates: dict[str, object] = {}
    normalized: dict[str, object] = {}
    if "new_name" in changes:
        name = _controlled_name(changes["new_name"], "new_name", "载荷")
        updates["name"] = name
        normalized["new_name"] = name
    if "target_scope" in changes:
        raw_scope = changes["target_scope"]
        if load_kind == "gravity" and raw_scope is None:
            scope_name = None
        else:
            scope_name = _required_text(raw_scope, "target_scope")
            _require_region(snapshot, scope_name)
        target_field = {
            "edge": "edge",
            "surface": "surface",
        }.get(load_kind, "target")
        updates[target_field] = scope_name
        normalized["target_scope"] = scope_name
    if "component" in changes:
        value = _component(changes["component"], "component")
        updates["component"] = value
        normalized["component"] = value
    if "value" in changes:
        value = _finite_number(changes["value"], "value")
        updates["value"] = value
        normalized["value"] = value
    for field_name in ("vector", "acceleration"):
        if field_name in changes:
            values = (
                ()
                if field_name == "vector"
                and strict_kind
                and (
                    changes[field_name] is None
                    or changes[field_name] == ()
                    or changes[field_name] == []
                )
                else _number_list(changes[field_name], field_name)
            )
            updates[field_name] = values
            normalized[field_name] = list(values)
    if "magnitude" in changes:
        raw_magnitude = changes["magnitude"]
        magnitude = (
            None
            if raw_magnitude is None
            else _finite_number(raw_magnitude, "magnitude")
        )
        updates["magnitude"] = magnitude
        normalized["magnitude"] = magnitude
    if "coordinate_system" in changes:
        value = _required_text(
            changes["coordinate_system"],
            "coordinate_system",
        )
        updates["coordinate_system"] = value
        normalized["coordinate_system"] = value
    if strict_kind:
        load_type = _required_text(changes["load_type"], "load_type")
        expected_prefix = {
            "nodal": "nodal",
            "edge": "edge_",
            "surface": "surface_",
        }[load_kind]
        if (
            load_kind == "nodal"
            and load_type != expected_prefix
        ) or (
            load_kind != "nodal"
            and not load_type.startswith(expected_prefix)
        ):
            raise ValueError("load_type does not match the edited load entity")
        if load_kind in {"edge", "surface"}:
            kernel_type = load_type.removeprefix(expected_prefix)
            if kernel_type not in {"traction", "pressure"}:
                raise ValueError("load_type is unsupported")
            updates["load_type"] = kernel_type
        normalized["load_type"] = load_type
    elif "load_type" in changes:
        value = _required_text(changes["load_type"], "load_type")
        updates["load_type"] = value
        normalized["load_type"] = value
    replacement = replace(current, **updates)
    if replacement == current:
        raise ValueError("changes do not modify the selected load")
    if strict_kind:
        confirmed = _confirmed_load_after_edit(
            snapshot,
            target,
            replacement,
            load_kind,
            changes,
        )
        normalized.update(
            {
                "entity_type": confirmed.entity_type,
                "direction": confirmed.direction,
                "unit": confirmed.unit,
                "distribution": confirmed.distribution,
                "confirmed": True,
            }
        )
    else:
        normalized.update(
            _validated_extended_load_edit(
                snapshot, target, replacement, load_kind, changes
            )
        )
    return normalized, replacement


def _validated_extended_load_edit(
    snapshot: _Snapshot,
    target: EditableObject,
    replacement: object,
    load_kind: str,
    changes: Mapping[object, object],
) -> dict[str, object]:
    model = getattr(snapshot.artifact, "model", None)
    if model is None:
        raise ValueError("load edit requires a current realized model")
    capability = describe_model_capabilities(model)
    if not capability.compatible or capability.spatial_dimension not in {2, 3}:
        raise ValueError("load edit requires a compatible 2D or 3D model")
    dimension = int(capability.spatial_dimension)
    units = _require_units(snapshot)
    distribution = _required_text(changes["distribution"], "distribution")
    if distribution != "uniform":
        raise ValueError("distributed element load must be uniform")
    _confirmed(changes["confirmed"])

    raw_scope = getattr(replacement, "target")
    if raw_scope is None:
        if load_kind != "gravity":
            raise ValueError("only gravity may target the whole model")
        if "gravity" not in capability.load_kinds:
            raise ValueError(
                "current model capability does not support global gravity"
            )
        scope_name = None
    else:
        scope_name = _required_text(raw_scope, "target_scope")
        region = _require_region(snapshot, scope_name)
        if region.entity_kind != "element":
            raise ValueError("load target must be an element named region")
        if load_kind in {"body", "gravity"}:
            region_capability = describe_region_capabilities(
                model, RegionRef("element_set", scope_name)
            )
            if (
                not region_capability.compatible
                or load_kind not in region_capability.load_kinds
            ):
                raise ValueError(
                    f"target element region does not support {load_kind} load"
                )

    if load_kind == "line":
        if len(tuple(getattr(replacement, "vector"))) != 3:
            raise ValueError("line load vector must contain exactly 3 components")
        coordinate_system = _required_text(
            changes["coordinate_system"], "coordinate_system"
        )
        if coordinate_system not in {"global", "local"}:
            raise ValueError("coordinate_system must be global or local")
        region_capability = describe_region_capabilities(
            model, RegionRef("element_set", str(scope_name))
        )
        if (
            not region_capability.compatible
            or not region_capability.homogeneous
            or region_capability.canonical_element_types != ("Beam2",)
        ):
            raise ValueError(
                "line load target must be a non-empty pure Beam2 element region"
            )
        unit = _required_text(changes["unit"], "unit")
        if unit != f"{units.force}/{units.length}":
            raise ValueError("line load unit does not match the project unit context")
        if coordinate_system == "local":
            step = _require_step(snapshot, target.step_name)
            index = next(
                index
                for index, item in enumerate(tuple(step.line_loads))
                if _optional_name(item) == target.target_id
            )
            decision = evaluate_native_line_load_candidate(
                snapshot,
                replacement,
                str(target.step_name),
                candidate_index=index,
            )
            if not decision.can_submit:
                message = "; ".join(item.message for item in decision.diagnostics)
                raise ValueError(
                    message or "local line load requires resolved Beam orientation"
                )
        return {
            "coordinate_system": coordinate_system,
            "unit": unit,
            "distribution": distribution,
            "confirmed": True,
        }

    direction = _required_text(changes["direction"], "direction")
    if direction != "global":
        raise ValueError(f"{load_kind} load direction must be global")
    if load_kind == "body":
        if len(tuple(getattr(replacement, "vector"))) != dimension:
            raise ValueError("body force vector dimension does not match the model")
        expected_unit = f"{units.force}/{units.length}^3"
    else:
        if len(tuple(getattr(replacement, "acceleration"))) != dimension:
            raise ValueError("gravity acceleration dimension does not match the model")
        if units.acceleration is None:
            raise ValueError("gravity requires an explicit project acceleration unit")
        expected_unit = units.acceleration
    unit = _required_text(changes["unit"], "unit")
    if unit != expected_unit:
        raise ValueError("load unit does not match the project unit context")
    return {
        "direction": direction,
        "unit": unit,
        "distribution": distribution,
        "confirmed": True,
    }


def _confirmed_load_after_edit(
    snapshot: _Snapshot,
    target: EditableObject,
    replacement: object,
    load_kind: str,
    changes: Mapping[object, object],
) -> ConfirmedLoad:
    target_field = {
        "edge": "edge",
        "surface": "surface",
    }.get(load_kind, "target")
    scope_name = _required_text(
        getattr(replacement, target_field),
        "target_scope",
    )
    region = _require_region(snapshot, scope_name)
    entity_type = _required_text(changes["entity_type"], "entity_type")
    expected_region_kind = {
        "nodal": "node",
        "edge": "edge",
        "surface": "face",
    }[load_kind]
    if region.entity_kind != expected_region_kind:
        raise ValueError("load target scope has the wrong mesh entity kind")
    load_type = _required_text(changes["load_type"], "load_type")
    confirmed = ConfirmedLoad(
        _controlled_name(replacement.name, "load name", "载荷"),
        _controlled_name(target.step_name, "step_name", "分析步"),
        scope_name,
        entity_type,
        load_type,
        (
            int(getattr(replacement, "component"))
            if load_kind == "nodal"
            else None
        ),
        (
            tuple(getattr(replacement, "vector"))
            if load_kind in {"edge", "surface"}
            else ()
        ),
        (
            float(getattr(replacement, "value"))
            if load_kind == "nodal"
            else getattr(replacement, "magnitude")
        ),
        _required_text(changes["direction"], "direction"),
        _required_text(changes["unit"], "unit"),
        _required_text(changes["distribution"], "distribution"),
        _confirmed(changes["confirmed"]),
    )
    dimension = _part_dimension_for_region(snapshot, region)
    units = _require_units(snapshot)
    if confirmed.load_type == "nodal":
        if int(confirmed.component or 0) > dimension:
            raise ValueError("load component exceeds the current Part dimension")
        expected_direction = {
            1: "global_x",
            2: "global_y",
            3: "global_z",
        }[int(confirmed.component)]
        if confirmed.direction != expected_direction:
            raise ValueError("load direction does not match its component")
        expected_unit = units.force
    elif confirmed.load_type.startswith("edge_"):
        if dimension != 2:
            raise ValueError("edge load requires a two-dimensional Part")
        if (
            confirmed.load_type == "edge_traction"
            and confirmed.direction != "global_xy"
        ):
            raise ValueError("two-dimensional traction requires global_xy")
        expected_unit = f"{units.force}/{units.length}"
    else:
        if dimension != 3:
            raise ValueError("surface load requires a three-dimensional Part")
        if (
            confirmed.load_type == "surface_traction"
            and confirmed.direction != "global_xyz"
        ):
            raise ValueError(
                "three-dimensional traction requires global_xyz"
            )
        expected_unit = units.stress
    if confirmed.vector and len(confirmed.vector) != dimension:
        raise ValueError("load vector dimension does not match the current Part")
    if confirmed.unit != expected_unit:
        raise ValueError("load unit does not match the project unit context")
    return confirmed


def _load_details(
    snapshot: _Snapshot,
    load: object,
    load_kind: str,
) -> dict[str, object]:
    details: dict[str, object] = {
        "load_kind": load_kind,
        "editable_fields": {
            "nodal": ["new_name", "target_scope", "component", "value"],
            "edge": [
                "new_name",
                "target_scope",
                "vector",
                "magnitude",
                "load_type",
            ],
            "surface": [
                "new_name",
                "target_scope",
                "vector",
                "magnitude",
                "load_type",
            ],
            "line": [
                "new_name",
                "target_scope",
                "vector",
                "coordinate_system",
            ],
            "body": [
                "new_name", "target_scope", "vector", "direction",
            ],
            "gravity": [
                "new_name", "target_scope", "acceleration", "direction",
            ],
        }[load_kind],
    }
    if load_kind in {"nodal", "edge", "surface"}:
        details["editable_fields"] = [
            *details["editable_fields"],  # type: ignore[list-item]
            "entity_type",
            "direction",
            "unit",
            "distribution",
            "confirmed",
        ]
    target_field = {
        "edge": "edge",
        "surface": "surface",
    }.get(load_kind, "target")
    details["target_scope"] = getattr(load, target_field, None)
    for field_name in (
        "component",
        "value",
        "vector",
        "magnitude",
        "load_type",
        "coordinate_system",
        "acceleration",
    ):
        if hasattr(load, field_name):
            value = getattr(load, field_name)
            details[field_name] = (
                list(value) if isinstance(value, tuple) else value
            )
    if load_kind in {"nodal", "edge", "surface"}:
        entity_type = {
            "nodal": "node",
            "edge": "edge",
            "surface": "surface",
        }[load_kind]
        details["entity_type"] = entity_type
        if load_kind == "nodal":
            component = int(getattr(load, "component"))
            details["load_type"] = "nodal"
            details["direction"] = {
                1: "global_x",
                2: "global_y",
                3: "global_z",
            }.get(component)
        else:
            load_type = str(getattr(load, "load_type"))
            details["load_type"] = f"{load_kind}_{load_type}"
            details["direction"] = (
                f"global_{'xy' if load_kind == 'edge' else 'xyz'}"
                if load_type == "traction"
                else (
                    "inward_normal"
                    if float(getattr(load, "magnitude") or 0.0) > 0.0
                    else "outward_normal"
                )
            )
        region = _require_region(
            snapshot,
            _required_text(details["target_scope"], "target_scope"),
        )
        dimension = _part_dimension_for_region(snapshot, region)
        units = _require_units(snapshot)
        details["unit"] = (
            units.force
            if load_kind == "nodal"
            else f"{units.force}/{units.length}"
            if dimension == 2
            else units.stress
        )
        details["distribution"] = (
            "concentrated" if load_kind == "nodal" else "uniform"
        )
        details["confirmed"] = True
    else:
        details["editable_fields"] = [
            *details["editable_fields"],  # type: ignore[list-item]
            "unit",
            "distribution",
            "confirmed",
        ]
        units = _require_units(snapshot)
        details["direction"] = (
            str(getattr(load, "coordinate_system"))
            if load_kind == "line"
            else "global"
        )
        details["unit"] = {
            "line": f"{units.force}/{units.length}",
            "body": f"{units.force}/{units.length}^3",
            "gravity": units.acceleration,
        }[load_kind]
        details["distribution"] = "uniform"
        details["confirmed"] = True
    return details


def _find_boundary(
    snapshot: _Snapshot,
    target: EditableObject,
) -> DisplacementConstraint:
    step = _require_step(snapshot, target.step_name)
    matches = tuple(
        item
        for item in tuple(getattr(step, "boundaries", ()))
        if _optional_name(item) == target.target_id
    )
    if len(matches) != 1 or type(matches[0]) is not DisplacementConstraint:
        raise ValueError("boundary condition target is unavailable")
    return matches[0]


def _find_load(
    snapshot: _Snapshot,
    target: EditableObject,
) -> tuple[str, object, str]:
    step = _require_step(snapshot, target.step_name)
    matches: list[tuple[str, object, str]] = []
    for collection_name, expected_type, load_kind in _LOAD_COLLECTIONS:
        matches.extend(
            (collection_name, item, load_kind)
            for item in tuple(getattr(step, collection_name, ()))
            if type(item) is expected_type
            and _optional_name(item) == target.target_id
        )
    if len(matches) != 1:
        raise ValueError("load target is unavailable or ambiguous")
    return matches[0]


def _find_output(
    snapshot: _Snapshot,
    target: EditableObject,
) -> OutputRequest:
    step = _require_step(snapshot, target.step_name)
    matches = tuple(
        item
        for item in tuple(getattr(step, "outputs", ()))
        if _optional_name(item) == target.target_id
    )
    if len(matches) != 1 or type(matches[0]) is not OutputRequest:
        raise ValueError("result request target is unavailable or ambiguous")
    return matches[0]


def _replace_step_child(
    step: object,
    target: EditableObject,
    replacement: object,
) -> object:
    if target.object_type == "boundary_condition":
        return replace(
            step,
            boundaries=tuple(
                replacement
                if _optional_name(item) == target.target_id
                else item
                for item in tuple(getattr(step, "boundaries", ()))
            ),
        )
    if target.object_type == "result_request":
        return replace(
            step,
            outputs=tuple(
                replacement
                if _optional_name(item) == target.target_id
                else item
                for item in tuple(getattr(step, "outputs", ()))
            ),
        )
    collection_name, _current, _kind = _find_load(
        _StepSnapshot(step),
        target,
    )
    return replace(
        step,
        **{
            collection_name: tuple(
                replacement
                if _optional_name(item) == target.target_id
                else item
                for item in tuple(getattr(step, collection_name, ()))
            )
        },
    )


class _StepSnapshot:
    def __init__(self, step: object) -> None:
        self.steps = (step,)


def _rename_step_scope(step: object, old_name: str, new_name: str) -> object:
    updates = {}
    for collection_name, field_name in _STEP_REFERENCE_FIELDS:
        updates[collection_name] = tuple(
            replace(item, **{field_name: new_name})
            if getattr(item, field_name, None) == old_name
            else item
            for item in tuple(getattr(step, collection_name, ()))
        )
    return replace(step, **updates)


def _reference_map(
    snapshot: _Snapshot,
) -> dict[str, MeshEntityRef | LogicalEntityRef]:
    result: dict[str, MeshEntityRef | LogicalEntityRef] = {}
    for region in tuple(snapshot.named_regions.values()):
        for reference in region.references:
            key = _reference_key(reference)
            previous = result.get(key)
            if previous is not None and previous != reference:
                raise ValueError("scope reference identity collision")
            result[key] = reference
    return result


def _reference_key(reference: MeshEntityRef | LogicalEntityRef) -> str:
    if type(reference) is LogicalEntityRef:
        payload: dict[str, object] = {
            "type": "logical",
            "logical_id": reference.logical_id,
        }
    elif type(reference) is MeshEntityRef:
        payload = {
            "type": "mesh",
            "kind": reference.kind,
            "node_id": reference.node_id,
            "element_id": reference.element_id,
            "local_index": reference.local_index,
            "node_ids": list(reference.node_ids),
            "part_id": reference.part_id,
        }
    else:
        raise TypeError("unsupported named-region reference")
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"scope-ref-{digest}"


def _require_step(snapshot: object, step_name: str | None) -> object:
    matches = tuple(
        step
        for step in tuple(getattr(snapshot, "steps", ()))
        if getattr(step, "name", None) == step_name
    )
    if len(matches) != 1:
        raise ValueError("analysis step is unavailable or ambiguous")
    return matches[0]


def _find_named(
    values: Sequence[object],
    name: str,
    label: str,
) -> object:
    matches = tuple(item for item in values if getattr(item, "name", None) == name)
    if len(matches) != 1:
        raise ValueError(f"{label} target is unavailable or ambiguous")
    return matches[0]


def _require_region(snapshot: _Snapshot, name: str) -> NamedRegion:
    try:
        region = snapshot.named_regions[name]
    except KeyError as error:
        raise ValueError("target_scope is unavailable") from error
    if type(region) is not NamedRegion:
        raise TypeError("target_scope must resolve to NamedRegion")
    return region


def _boundary_target_kind(region: NamedRegion) -> str:
    return {
        "point": "node_set",
        "node": "node_set",
        "edge": "edge",
        "face": "surface",
    }.get(region.entity_kind) or _unsupported_boundary_scope()


def _unsupported_boundary_scope() -> str:
    raise ValueError("target_scope cannot support a displacement boundary")


def _require_part(snapshot: _Snapshot, part_id: str) -> object:
    matches = tuple(
        part
        for part in snapshot.parts
        if getattr(part, "id", None) == part_id
        and not bool(getattr(part, "suppressed", False))
    )
    if len(matches) != 1:
        raise ValueError("selected Part is unavailable or ambiguous")
    return matches[0]


def _part_dimension_for_region(
    snapshot: _Snapshot,
    region: NamedRegion,
) -> int:
    part_ids = {
        reference.part_id
        for reference in region.references
        if type(reference) is MeshEntityRef and reference.part_id is not None
    }
    if len(part_ids) > 1:
        raise ValueError("engineering scope spans multiple Parts")
    part_id = (
        next(iter(part_ids))
        if part_ids
        else snapshot.active_part_id
    )
    if part_id is None:
        raise ValueError("engineering scope has no exact owning Part")
    part = _require_part(snapshot, part_id)
    recipe = getattr(part, "geometry_recipe", None)
    if recipe is None:
        raise ValueError("engineering Part has no geometry recipe")
    return int(geometry_dimension(recipe))


def _require_units(snapshot: _Snapshot) -> UnitContext:
    units = snapshot.unit_context
    if type(units) is not UnitContext:
        raise ValueError(
            "engineering edit requires the current project unit context"
        )
    return units


def _require_change_keys(
    changes: Mapping[object, object],
    allowed: set[str],
) -> None:
    if any(type(key) is not str for key in changes):
        raise ValueError("changes keys must be strings")
    unknown = set(changes) - allowed
    if unknown:
        raise ValueError(
            "changes has unsupported fields: " + ", ".join(sorted(unknown))
        )


def _required_text(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    normalized = value.strip()
    if len(normalized) > 160:
        raise ValueError(f"{field_name} must be at most 160 characters")
    return normalized


def _controlled_name(
    value: object,
    field_name: str,
    object_type: str,
) -> str:
    name = NamePolicy().validate(_required_text(value, field_name))
    if not name.startswith(f"{object_type}-"):
        raise ValueError(
            f"{field_name} must use the {object_type}- prefix"
        )
    return name


def _safe_name(value: object, field_name: str) -> str:
    return NamePolicy().validate(_required_text(value, field_name))


def _bounded_text(value: object) -> str:
    text = str(value)
    return text if len(text) <= 160 else text[:157] + "..."


def _provider_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, object] = {}
    for raw_key, raw_value in tuple(value.items())[:32]:
        key = _bounded_text(raw_key)
        if isinstance(raw_value, bool) or raw_value is None:
            result[key] = raw_value
        elif isinstance(raw_value, (int, float)) and math.isfinite(
            float(raw_value)
        ):
            result[key] = raw_value
        elif type(raw_value) is str:
            result[key] = _bounded_text(raw_value)
        elif isinstance(raw_value, Sequence) and not isinstance(
            raw_value, (str, bytes, bytearray)
        ):
            values: list[object] = []
            for item in tuple(raw_value)[:16]:
                if isinstance(item, bool) or item is None:
                    values.append(item)
                elif isinstance(item, (int, float)) and math.isfinite(float(item)):
                    values.append(item)
                elif type(item) is str:
                    values.append(_bounded_text(item))
                else:
                    values.append(None)
            result[key] = values
        else:
            result[key] = None
    return result


def _bounded_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    if len(value) > 32:
        raise ValueError(f"{field_name} must contain at most 32 fields")
    result: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        key = _required_text(raw_key, f"{field_name} key")
        if len(key) > 96:
            raise ValueError(f"{field_name} keys must be at most 96 characters")
        if isinstance(raw_value, bool) or raw_value is None:
            normalized: object = raw_value
        elif isinstance(raw_value, (int, float)):
            normalized = _finite_number(raw_value, f"{field_name}.{key}")
        elif type(raw_value) is str:
            normalized = _required_text(raw_value, f"{field_name}.{key}")
        elif isinstance(raw_value, Sequence) and not isinstance(
            raw_value, (str, bytes, bytearray)
        ):
            items = tuple(raw_value)
            if not 1 <= len(items) <= 16:
                raise ValueError(
                    f"{field_name}.{key} arrays must contain 1 to 16 values"
                )
            normalized_items: list[object] = []
            for item in items:
                if isinstance(item, bool) or item is None:
                    normalized_items.append(item)
                elif isinstance(item, (int, float)):
                    normalized_items.append(
                        _finite_number(item, f"{field_name}.{key}")
                    )
                elif type(item) is str:
                    normalized_items.append(
                        _required_text(item, f"{field_name}.{key}")
                    )
                else:
                    raise ValueError(
                        f"{field_name}.{key} contains an unsupported value"
                    )
            normalized = normalized_items
        else:
            raise ValueError(f"{field_name}.{key} has an unsupported value")
        result[key] = normalized
    return result


def _confirmed(value: object) -> bool:
    if value is not True:
        raise ValueError("engineering edit fields must be explicitly confirmed")
    return True


def _positive_int(value: object, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _optional_name(value: object) -> str | None:
    name = getattr(value, "name", None)
    return name.strip() if type(name) is str and name.strip() else None


def _component(value: object, field_name: str) -> int:
    if type(value) is not int or not 1 <= value <= 6:
        raise ValueError(f"{field_name} must be an integer between 1 and 6")
    return value


def _finite_number(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


def _number_list(value: object, field_name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{field_name} must be an array")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{field_name} must be an array") from error
    if not 1 <= len(items) <= 3:
        raise ValueError(f"{field_name} must contain between 1 and 3 values")
    return tuple(_finite_number(item, field_name) for item in items)


def _string_list(
    value: object,
    field_name: str,
    *,
    max_items: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{field_name} must be an array")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{field_name} must be an array") from error
    if not 1 <= len(items) <= max_items:
        raise ValueError(
            f"{field_name} must contain between 1 and {max_items} values"
        )
    normalized = tuple(
        _required_text(item, field_name) for item in items
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique values")
    return normalized


__all__ = [
    "EditableObject",
    "apply_edit_operation",
    "create_edit_patch",
    "create_edit_proposal",
    "editable_object_catalog",
]
