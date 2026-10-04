"""Direct, incremental authoring for accepted native model definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from typing import Protocol

from fem.application import (
    BeamOrientation,
    ModelDefinitions,
    NamedRegion,
    RegionAssignment,
    ScopedDefinitionBatch,
    SectionDefinition,
    UnitContext,
)
from fem.core.model import (
    AnalysisStep,
    MaterialDefinition,
)

from .authoring import AuthoringContext, ModelPatch
from .definition_authoring import (
    definition_state_operations,
)
from .naming import NamePolicy


DIRECT_DEFINITION_ACTIONS = frozenset(
    {
        "create_material",
        "create_section",
        "assign_section",
        "create_static_step",
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
    unit_context: UnitContext | None


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

    if normalized_action == "create_material":
        _exact_fields(values, {"name", "properties"})
        name = _controlled_name(values["name"], "material name", "Material")
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
        name = _controlled_name(values["name"], "section name", "Section")
        material = _nonblank(values["material"], "section material")
        if name in {str(item.name) for item in sections}:
            raise ValueError("section name already exists")
        if material not in {str(item.name) for item in materials}:
            raise ValueError("section material does not exist")
        if set(values) == planar_fields:
            planar_properties = _mapping(
                values["properties"],
                "section properties",
            )
            if planar_properties:
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
        name = _controlled_name(values["name"], "analysis step name", "Step")
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
    definitions = ModelDefinitions(materials, sections, assignments, steps)
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
            "results": False,
            "historical_results_retained": True,
            "current_validation_reset": True,
            "current_result_display_reset": True,
        },
        display_summary={
            "title": f"Agent: {_ACTION_SUMMARIES[normalized_action]}",
            "summary": _ACTION_SUMMARIES[normalized_action],
            "objects": list(created_names),
            "undo_label": "Undo",
        },
    )


def require_incremental_definition_batch(
    snapshot: _Snapshot,
    batch: ScopedDefinitionBatch,
    action: object,
) -> None:
    """Verify that a direct patch contains exactly its declared addition."""

    normalized_action = _exact_action(action)
    before_materials = tuple(snapshot.materials)
    before_sections = tuple(snapshot.sections)
    before_assignments = tuple(snapshot.assignments)
    before_steps = tuple(snapshot.steps)

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


_ACTION_SUMMARIES = {
    "create_material": "Material created",
    "create_section": "Section created",
    "assign_section": "Section assignment created",
    "create_static_step": "Linear static step created",
}


def _require_live_native_context(
    context: AuthoringContext,
    snapshot: _Snapshot,
) -> None:
    binding = context.binding
    # Multi-document workspaces bind stable integer document identities such
    # as "2"; only session identity and revision prove freshness, so the
    # document_id format is deliberately not compared here.
    if (
        snapshot.source_kind != "native"
        or snapshot.artifact is None
        or not snapshot.model_current
        or binding.source_kind != "native"
        or binding.session_id != snapshot.session_id
        or binding.session_revision != snapshot.session_revision
    ):
        raise ValueError("direct authoring context is stale or unavailable")


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
    if not NamePolicy.has_type(name, object_type):
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


__all__ = [
    "DIRECT_DEFINITION_ACTIONS",
    "create_incremental_definition_patch",
    "require_incremental_definition_batch",
]
