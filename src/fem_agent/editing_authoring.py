"""Bounded catalogs and GUI-confirmed edits for accepted model definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from typing import Protocol

from fem.application import (
    MeshEntityRef,
    ModelSession,
    NamedRegion,
    ScopedDefinitionBatch,
)
from fem.application.changes import SessionDelta
from fem.core.model import (
    BodyForce,
    DisplacementConstraint,
    EdgeLoad,
    GravityLoad,
    LineLoad,
    NodalLoad,
    SurfaceLoad,
)
from fem.geometry import LogicalEntityRef

from .authoring import (
    AgentProposal,
    AuthoringContext,
    ModelOperation,
    OperationKind,
    ProposalKind,
)


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
_EDIT_TYPES = frozenset({"named_region", "boundary_condition", "load"})


class _Snapshot(Protocol):
    source_kind: str | None
    session_revision: int
    named_regions: Mapping[str, NamedRegion]
    materials: object
    sections: object
    assignments: object
    steps: object
    artifact: object | None


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
        if self.object_type in {"boundary_condition", "load"}:
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
    """Return current named scopes, boundaries, and loads with editable values."""

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
                    "editable_fields": ["new_name", "reference_keys"],
                },
            )
        )

    for step in tuple(snapshot.steps):
        step_name = str(getattr(step, "name", "")).strip()
        if not step_name or len(step_name) > 160:
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
                        "editable_fields": [
                            "new_name",
                            "target_scope",
                            "first_component",
                            "last_component",
                            "value",
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
                        _load_details(load, load_kind),
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
        "boundary_condition": "边界条件",
        "load": "载荷",
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
            "results": True,
        },
        display_summary={
            "title": f"编辑{label}：{target.display_name}",
            "summary": f"修改{label}“{target.display_name}”的{changed_fields}",
            "impact": "修改后相关预检、作业和结果将失效",
            "confirm_label": "确认修改",
        },
    )
    return proposal, target


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
    else:
        steps = tuple(
            _replace_step_child(step, target, replacement)
            if step.name == target.step_name
            else step
            for step in steps
        )

    return session.apply_scoped_definition_batch(
        ScopedDefinitionBatch(
            base_session_revision,
            regions,
            tuple(snapshot.materials),
            tuple(snapshot.sections),
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
    if target.object_type == "boundary_condition":
        return _validated_boundary_edit(snapshot, target, changes)
    return _validated_load_edit(snapshot, target, changes)


def _validated_region_edit(
    snapshot: _Snapshot,
    target: EditableObject,
    changes: dict[object, object],
) -> tuple[dict[str, object], NamedRegion]:
    allowed = {"new_name", "reference_keys"}
    _require_change_keys(changes, allowed)
    current = snapshot.named_regions[target.target_id]
    updates: dict[str, object] = {}
    normalized: dict[str, object] = {}
    if "new_name" in changes:
        name = _required_text(changes["new_name"], "new_name")
        updates["name"] = name
        normalized["new_name"] = name
    if "reference_keys" in changes:
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
    replacement = replace(current, **updates)
    if replacement == current:
        raise ValueError("changes do not modify the selected named region")
    return normalized, replacement


def _validated_boundary_edit(
    snapshot: _Snapshot,
    target: EditableObject,
    changes: dict[object, object],
) -> tuple[dict[str, object], DisplacementConstraint]:
    allowed = {
        "new_name",
        "target_scope",
        "first_component",
        "last_component",
        "value",
    }
    _require_change_keys(changes, allowed)
    current = _find_boundary(snapshot, target)
    updates: dict[str, object] = {}
    normalized: dict[str, object] = {}
    if "new_name" in changes:
        name = _required_text(changes["new_name"], "new_name")
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
    return normalized, replacement


def _validated_load_edit(
    snapshot: _Snapshot,
    target: EditableObject,
    changes: dict[object, object],
) -> tuple[dict[str, object], object]:
    collection_name, current, load_kind = _find_load(snapshot, target)
    del collection_name
    allowed_by_kind = {
        "nodal": {"new_name", "target_scope", "component", "value"},
        "edge": {
            "new_name",
            "target_scope",
            "vector",
            "magnitude",
            "load_type",
        },
        "surface": {
            "new_name",
            "target_scope",
            "vector",
            "magnitude",
            "load_type",
        },
        "line": {
            "new_name",
            "target_scope",
            "vector",
            "coordinate_system",
        },
        "body": {"new_name", "target_scope", "vector"},
        "gravity": {"new_name", "target_scope", "acceleration"},
    }
    _require_change_keys(changes, allowed_by_kind[load_kind])
    updates: dict[str, object] = {}
    normalized: dict[str, object] = {}
    if "new_name" in changes:
        name = _required_text(changes["new_name"], "new_name")
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
            values = _number_list(changes[field_name], field_name)
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
    for field_name in ("load_type", "coordinate_system"):
        if field_name in changes:
            value = _required_text(changes[field_name], field_name)
            updates[field_name] = value
            normalized[field_name] = value
    replacement = replace(current, **updates)
    if replacement == current:
        raise ValueError("changes do not modify the selected load")
    return normalized, replacement


def _load_details(load: object, load_kind: str) -> dict[str, object]:
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
            "body": ["new_name", "target_scope", "vector"],
            "gravity": ["new_name", "target_scope", "acceleration"],
        }[load_kind],
    }
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
    "create_edit_proposal",
    "editable_object_catalog",
]
