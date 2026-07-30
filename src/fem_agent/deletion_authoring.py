"""Bounded catalog, proposal, and execution helpers for Agent deletions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Protocol

from fem.application import ModelSession, ScopedDefinitionBatch
from fem.application.changes import SessionDelta

from .authoring import (
    AgentProposal,
    AuthoringContext,
    ModelOperation,
    OperationKind,
    ProposalKind,
)


_LOAD_COLLECTIONS = (
    "cloads",
    "edge_loads",
    "surface_loads",
    "line_loads",
    "body_loads",
    "gravity_loads",
)
_STEP_CHILD_TYPES = frozenset(
    {"boundary_condition", "load", "result_request"}
)
_DELETE_TYPES = frozenset(
    {
        "part",
        "generated_mesh",
        "named_region",
        "analysis_step",
        *_STEP_CHILD_TYPES,
    }
)
_TYPE_LABELS = {
    "part": "部件",
    "generated_mesh": "已生成网格",
    "named_region": "作用域",
    "analysis_step": "分析步",
    "boundary_condition": "边界条件",
    "load": "载荷",
    "result_request": "结果请求",
}


class _Snapshot(Protocol):
    source_kind: str | None
    session_revision: int
    parts: object
    part_revisions: Mapping[str, int]
    named_regions: Mapping[str, object]
    materials: object
    sections: object
    assignments: object
    steps: object
    artifact: object | None

    def part_revision(self, part_id: str) -> int: ...


@dataclass(frozen=True, slots=True)
class DeletableObject:
    object_type: str
    target_id: str
    display_name: str
    step_name: str | None = None
    impact: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.object_type not in _DELETE_TYPES:
            raise ValueError("unsupported deletable object type")
        for field_name in ("target_id", "display_name"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field_name} must be a non-blank string")
            normalized = value.strip()
            if len(normalized) > 160:
                raise ValueError(f"{field_name} must be at most 160 characters")
            object.__setattr__(self, field_name, normalized)
        if self.object_type in _STEP_CHILD_TYPES:
            if type(self.step_name) is not str or not self.step_name.strip():
                raise ValueError("step_name is required for step child targets")
            normalized_step = self.step_name.strip()
            if len(normalized_step) > 160:
                raise ValueError("step_name must be at most 160 characters")
            object.__setattr__(self, "step_name", normalized_step)
        elif self.step_name is not None:
            raise ValueError("step_name is only valid for step child targets")
        values = tuple(str(item).strip()[:240] for item in self.impact)
        object.__setattr__(self, "impact", tuple(item for item in values if item))

    def to_provider_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "object_type": self.object_type,
            "target_id": self.target_id,
            "display_name": self.display_name,
            "impact": list(self.impact),
        }
        if self.step_name is not None:
            result["step_name"] = self.step_name
        return result


def deletable_object_catalog(
    snapshot: _Snapshot,
    *,
    limit: int = 100,
) -> tuple[DeletableObject, ...]:
    """Return a bounded stable-identity catalog for one native snapshot."""

    if type(limit) is not int or isinstance(limit, bool) or not 1 <= limit <= 128:
        raise ValueError("limit must be an integer between 1 and 128")
    if getattr(snapshot, "source_kind", None) != "native":
        return ()

    items: list[DeletableObject] = []
    for part in tuple(getattr(snapshot, "parts", ())):
        part_id = str(getattr(part, "id", "")).strip()
        name = str(getattr(part, "name", "")).strip()
        if (
            part_id
            and name
            and len(part_id) <= 160
            and len(name) <= 160
        ):
            items.append(
                DeletableObject(
                    "part",
                    part_id,
                    name,
                    impact=(
                        "删除该部件及其几何、网格设置和部件作用域",
                        "依赖指派、分析定义、预检、作业和结果可能失效",
                    ),
                )
            )

    artifact = getattr(snapshot, "artifact", None)
    if artifact is not None:
        items.append(
            DeletableObject(
                "generated_mesh",
                "current",
                "当前已生成网格",
                impact=(
                    "清除当前生成模型和网格",
                    "网格作用域及其依赖定义、预检、作业和结果将失效",
                ),
            )
        )
        for region in tuple(
            getattr(snapshot, "named_regions", {}).values()
        ):
            name = str(getattr(region, "name", "")).strip()
            if name and len(name) <= 160:
                items.append(
                    DeletableObject(
                        "named_region",
                        name,
                        name,
                        impact=_named_region_impact(snapshot, name),
                    )
                )
        for step in tuple(getattr(snapshot, "steps", ())):
            step_name = str(getattr(step, "name", "")).strip()
            if not step_name or len(step_name) > 160:
                continue
            items.append(
                DeletableObject(
                    "analysis_step",
                    step_name,
                    step_name,
                    impact=(
                        "删除该分析步内全部边界条件、载荷和结果请求",
                        "相关预检、作业和结果将失效",
                    ),
                )
            )
            items.extend(_step_child_catalog(step, step_name))

    identity_counts: dict[tuple[str, str, str | None], int] = {}
    for item in items:
        identity = (item.object_type, item.target_id, item.step_name)
        identity_counts[identity] = identity_counts.get(identity, 0) + 1
    unique_items = (
        item
        for item in items
        if identity_counts[
            (item.object_type, item.target_id, item.step_name)
        ]
        == 1
    )
    return tuple(
        sorted(
            unique_items,
            key=lambda item: (
                item.object_type,
                "" if item.step_name is None else item.step_name,
                item.target_id,
            ),
        )[:limit]
    )


def resolve_deletable_object(
    snapshot: _Snapshot,
    object_type: object,
    target_id: object,
    step_name: object = None,
) -> DeletableObject:
    normalized_type = _required_text(object_type, "object_type")
    normalized_target = _required_text(target_id, "target_id")
    normalized_step = (
        None if step_name is None else _required_text(step_name, "step_name")
    )
    if normalized_type not in _DELETE_TYPES:
        raise ValueError("unsupported delete object_type")
    if normalized_type in _STEP_CHILD_TYPES and normalized_step is None:
        raise ValueError("step_name is required for this delete target")
    if normalized_type not in _STEP_CHILD_TYPES and normalized_step is not None:
        raise ValueError("step_name is not valid for this delete target")

    matches = tuple(
        item
        for item in deletable_object_catalog(snapshot, limit=128)
        if item.object_type == normalized_type
        and item.target_id == normalized_target
        and item.step_name == normalized_step
    )
    if len(matches) != 1:
        raise ValueError("delete target is unavailable or ambiguous")
    return matches[0]


def create_delete_proposal(
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
    step_name: object = None,
) -> tuple[AgentProposal, DeletableObject]:
    """Build one revision-bound, GUI-confirmed destructive proposal."""

    target = resolve_deletable_object(
        snapshot,
        object_type,
        target_id,
        step_name,
    )
    parameters: dict[str, object] = {
        "object_type": target.object_type,
        "target_id": target.target_id,
    }
    if target.step_name is not None:
        parameters["step_name"] = target.step_name
    label = _TYPE_LABELS[target.object_type]
    impact_text = "；".join(target.impact)
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
            ModelOperation(
                OperationKind.DELETE_MODEL_OBJECT,
                parameters,
            ),
        ),
        preconditions={
            "source_kind": "native",
            "target_exists": True,
            "exact_session_revision": True,
        },
        expected_changes={
            "deleted_object_type": target.object_type,
            "deleted_target_id": target.target_id,
        },
        invalidation_impact={
            "items": list(target.impact),
            "model": True,
            "validation": True,
            "results": True,
        },
        display_summary={
            "title": f"删除{label}：{target.display_name}",
            "summary": f"删除{label}“{target.display_name}”",
            "impact": impact_text,
            "confirm_label": "确认删除",
        },
    )
    return proposal, target


def apply_delete_operation(
    session: ModelSession,
    operation: ModelOperation,
    *,
    base_session_revision: int,
) -> SessionDelta:
    """Apply one already-confirmed delete through public Session commands."""

    if type(session) is not ModelSession:
        raise TypeError("session must be exactly ModelSession")
    if (
        type(operation) is not ModelOperation
        or operation.kind is not OperationKind.DELETE_MODEL_OBJECT
    ):
        raise TypeError("operation must be DELETE_MODEL_OBJECT")
    parameters = operation.parameters
    snapshot = session.snapshot()
    target = resolve_deletable_object(
        snapshot,
        parameters["object_type"],
        parameters["target_id"],
        parameters.get("step_name"),
    )
    if target.object_type == "part":
        return session.delete_native_part(
            target.target_id,
            expected_part_revision=snapshot.part_revision(target.target_id),
            expected_session_revision=base_session_revision,
        )
    if target.object_type == "generated_mesh":
        return session.clear_generated_model(
            expected_session_revision=base_session_revision,
        )

    regions = tuple(snapshot.named_regions.values())
    assignments = tuple(snapshot.assignments)
    steps = deepcopy(tuple(snapshot.steps))
    if target.object_type == "named_region":
        regions = tuple(
            item for item in regions if item.name != target.target_id
        )
        assignments = tuple(
            item
            for item in assignments
            if item.region_name != target.target_id
        )
        steps = tuple(
            _without_region_dependencies(step, target.target_id)
            for step in steps
        )
    elif target.object_type == "analysis_step":
        steps = tuple(
            step for step in steps if step.name != target.target_id
        )
    else:
        steps = tuple(
            _without_step_child(
                step,
                target.object_type,
                target.target_id,
            )
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


def _step_child_catalog(
    step: object,
    step_name: str,
) -> tuple[DeletableObject, ...]:
    items: list[DeletableObject] = []
    for boundary in tuple(getattr(step, "boundaries", ())):
        name = _optional_name(boundary)
        if name is not None and len(name) <= 160:
            items.append(
                DeletableObject(
                    "boundary_condition",
                    name,
                    name,
                    step_name,
                    (
                        "删除该边界条件",
                        "相关预检、作业和结果将失效",
                    ),
                )
            )
    for collection_name in _LOAD_COLLECTIONS:
        for load in tuple(getattr(step, collection_name, ())):
            name = _optional_name(load)
            if name is not None and len(name) <= 160:
                items.append(
                    DeletableObject(
                        "load",
                        name,
                        name,
                        step_name,
                        (
                            "删除该载荷",
                            "相关预检、作业和结果将失效",
                        ),
                    )
                )
    for output in tuple(getattr(step, "outputs", ())):
        name = _optional_name(output)
        if name is not None and len(name) <= 160:
            items.append(
                DeletableObject(
                    "result_request",
                    name,
                    name,
                    step_name,
                    (
                        "删除该结果请求",
                        "后续求解将不再生成对应请求结果",
                    ),
                )
            )
    return tuple(items)


def _named_region_impact(
    snapshot: _Snapshot,
    region_name: str,
) -> tuple[str, ...]:
    assignments = sum(
        1
        for item in tuple(getattr(snapshot, "assignments", ()))
        if getattr(item, "region_name", None) == region_name
    )
    definitions = sum(
        _step_region_dependency_count(step, region_name)
        for step in tuple(getattr(snapshot, "steps", ()))
    )
    impact = ["删除该作用域"]
    if assignments:
        impact.append(f"级联删除 {assignments} 个截面指派")
    if definitions:
        impact.append(f"级联删除 {definitions} 个依赖分析定义")
    impact.append("相关预检、作业和结果将失效")
    return tuple(impact)


def _step_region_dependency_count(step: object, region_name: str) -> int:
    return sum(
        1
        for collection_name in (
            "boundaries",
            *_LOAD_COLLECTIONS,
        )
        for item in tuple(getattr(step, collection_name, ()))
        if _object_targets_region(item, collection_name, region_name)
    )


def _without_region_dependencies(
    step: object,
    region_name: str,
) -> object:
    updates = {
        collection_name: tuple(
            item
            for item in tuple(getattr(step, collection_name, ()))
            if not _object_targets_region(
                item,
                collection_name,
                region_name,
            )
        )
        for collection_name in (
            "boundaries",
            *_LOAD_COLLECTIONS,
        )
    }
    return replace(step, **updates)


def _object_targets_region(
    item: object,
    collection_name: str,
    region_name: str,
) -> bool:
    field_name = {
        "boundaries": "target",
        "cloads": "target",
        "edge_loads": "edge",
        "surface_loads": "surface",
        "line_loads": "target",
        "body_loads": "target",
        "gravity_loads": "target",
    }[collection_name]
    return getattr(item, field_name, None) == region_name


def _without_step_child(
    step: object,
    object_type: str,
    target_id: str,
) -> object:
    if object_type == "boundary_condition":
        return replace(
            step,
            boundaries=tuple(
                item
                for item in tuple(getattr(step, "boundaries", ()))
                if _optional_name(item) != target_id
            ),
        )
    if object_type == "result_request":
        return replace(
            step,
            outputs=tuple(
                item
                for item in tuple(getattr(step, "outputs", ()))
                if _optional_name(item) != target_id
            ),
        )
    if object_type == "load":
        updates = {
            collection_name: tuple(
                item
                for item in tuple(getattr(step, collection_name, ()))
                if _optional_name(item) != target_id
            )
            for collection_name in _LOAD_COLLECTIONS
        }
        return replace(step, **updates)
    raise ValueError("unsupported step child delete type")


def _optional_name(value: object) -> str | None:
    name = getattr(value, "name", None)
    return name.strip() if type(name) is str and name.strip() else None


def _required_text(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    normalized = value.strip()
    if len(normalized) > 160:
        raise ValueError(f"{field_name} must be at most 160 characters")
    return normalized


__all__ = [
    "DeletableObject",
    "apply_delete_operation",
    "create_delete_proposal",
    "deletable_object_catalog",
    "resolve_deletable_object",
]
