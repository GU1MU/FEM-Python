"""Detached candidate evaluation for native authoring before meshing."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Iterable, Mapping

from fem.core.model import LineLoad
from fem.elements import (
    BEAM_ORIENTATION_PARALLEL_TOLERANCE,
    BeamOrientation,
    get_element_capabilities,
)
from fem.geometry.recipes import (
    MovedGeometry,
    RotatedGeometry,
    WireGeometry,
)

from .capabilities import (
    AuthoringCapability,
    AuthoringStatus,
    ModelCapabilityReport,
    RegionRef,
    describe_session_authoring,
)
from .definitions import RegionAssignment
from .diagnostics import (
    PreflightDiagnostic,
    PreflightSeverity,
    PreflightStage,
)
from .native_regions import (
    CompiledDomainRegionSource,
    LogicalReferencesRegionSource,
    MeshEntitiesRegionSource,
    describe_native_regions,
)
from .project_validation import validate_native_project_inputs


def evaluate_native_assignment_candidate(
    snapshot: Any,
    candidate: RegionAssignment,
    *,
    candidate_index: int | None = None,
) -> AuthoringCapability:
    """Evaluate one native section assignment without constructing a mesh."""

    if type(candidate) is not RegionAssignment:
        raise TypeError("candidate must be RegionAssignment")
    section = next(
        (
            item
            for item in tuple(getattr(snapshot, "sections", ()))
            if item.name == candidate.section_name
        ),
        None,
    )
    operation = (
        f"section.{candidate.section_name}"
        if section is None
        else _section_operation(section)
    )
    if section is None:
        return _rejected(
            operation,
            candidate.section_name,
            f"unknown section: {candidate.section_name}",
        )

    try:
        assignments = _candidate_assignments(
            tuple(getattr(snapshot, "assignments", ())),
            candidate,
            candidate_index,
        )
        _validate_candidate(
            snapshot,
            assignments=assignments,
            steps=tuple(getattr(snapshot, "steps", ())),
        )
    except (IndexError, TypeError, ValueError) as error:
        return _rejected(
            operation,
            candidate.region_name,
            str(error),
        )

    target = RegionRef("element_set", candidate.region_name)
    projection = describe_session_authoring(snapshot)
    assignment_capability = projection.target(target).operation(
        "section.assignment"
    )
    if not assignment_capability.can_submit:
        return AuthoringCapability(
            operation,
            AuthoringStatus.UNAVAILABLE,
            assignment_capability.diagnostics,
        )
    section_type = operation.removeprefix("section.")
    if not projection.report.region(target).supports_section(section_type):
        return _rejected(
            operation,
            target,
            f"target does not support section type {section_type!r}",
        )
    return _evaluate_beam_orientation(
        snapshot,
        operation,
        target,
        assignments,
        projection.report,
    )


def evaluate_native_line_load_candidate(
    snapshot: Any,
    candidate: LineLoad,
    step_name: str,
    *,
    candidate_index: int | None = None,
) -> AuthoringCapability:
    """Evaluate one native local Beam line load without constructing a mesh."""

    operation = "load.line.local"
    if type(candidate) is not LineLoad:
        raise TypeError("candidate must be LineLoad")
    if candidate.coordinate_system != "local":
        return _rejected(
            operation,
            candidate.target,
            "candidate is not a local LineLoad",
        )
    try:
        steps = _candidate_steps(
            tuple(getattr(snapshot, "steps", ())),
            candidate,
            step_name,
            candidate_index,
        )
        assignments = tuple(getattr(snapshot, "assignments", ()))
        _validate_candidate(
            snapshot,
            assignments=assignments,
            steps=steps,
        )
    except (IndexError, TypeError, ValueError) as error:
        return _rejected(
            operation,
            candidate.target,
            str(error),
        )

    target = RegionRef("element_set", str(candidate.target))
    projection = describe_session_authoring(snapshot)
    target_capability = projection.target(target).operation(operation)
    if not target_capability.can_enter:
        return AuthoringCapability(
            operation,
            AuthoringStatus.UNAVAILABLE,
            target_capability.diagnostics,
        )
    return _evaluate_beam_orientation(
        snapshot,
        operation,
        target,
        assignments,
        projection.report,
    )


def _candidate_assignments(
    existing: tuple[RegionAssignment, ...],
    candidate: RegionAssignment,
    candidate_index: int | None,
) -> tuple[RegionAssignment, ...]:
    assignments = list(existing)
    if candidate_index is None:
        assignments = [
            assignment
            for assignment in assignments
            if assignment.region_name != candidate.region_name
        ]
        assignments.append(deepcopy(candidate))
        return tuple(assignments)
    index = _candidate_index(
        candidate_index,
        len(assignments),
        "assignment",
    )
    assignments[index] = deepcopy(candidate)
    return tuple(assignments)


def _candidate_steps(
    existing: tuple[Any, ...],
    candidate: LineLoad,
    step_name: str,
    candidate_index: int | None,
) -> tuple[Any, ...]:
    steps = list(deepcopy(existing))
    normalized_step = str(step_name).strip()
    selected = next(
        (
            step
            for step in steps
            if str(getattr(step, "name", "")).strip() == normalized_step
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"unknown analysis step: {step_name!r}")
    line_loads = list(getattr(selected, "line_loads", ()))
    if candidate_index is None:
        line_loads.append(deepcopy(candidate))
    else:
        index = _candidate_index(
            candidate_index,
            len(line_loads),
            "line load",
        )
        line_loads[index] = deepcopy(candidate)
    selected.line_loads = tuple(line_loads)
    return tuple(steps)


def _candidate_index(value: Any, size: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} candidate_index must be an integer")
    index = int(value)
    if index < 0 or index >= size:
        raise IndexError(
            f"{label} candidate_index {index} is outside 0..{size - 1}"
        )
    return index


def _validate_candidate(
    snapshot: Any,
    *,
    assignments: tuple[RegionAssignment, ...],
    steps: tuple[Any, ...],
) -> None:
    recipe = getattr(snapshot, "geometry_recipe", None)
    if recipe is None or getattr(snapshot, "source_kind", None) != "native":
        raise ValueError("native candidate evaluation requires native geometry")
    validate_native_project_inputs(
        recipe,
        getattr(snapshot, "mesh_settings", None),
        _mapping_values(getattr(snapshot, "named_regions", ())),
        tuple(getattr(snapshot, "materials", ())),
        tuple(getattr(snapshot, "sections", ())),
        assignments,
        steps,
    )


def _evaluate_beam_orientation(
    snapshot: Any,
    operation: str,
    target: RegionRef,
    assignments: tuple[RegionAssignment, ...],
    capability_report: ModelCapabilityReport,
) -> AuthoringCapability:
    if not _requires_orientation_requirement(
        capability_report,
        operation,
        "beam.orientation.valid",
    ):
        return AuthoringCapability(
            operation,
            AuthoringStatus.ENABLED,
            (),
        )
    try:
        memberships, tangents = _wire_regions_and_tangents(snapshot)
        target_members = memberships[target.name]
        effective: dict[str, BeamOrientation | None] = {
            member: None
            for member in target_members
        }
        for assignment in assignments:
            assigned_members = memberships[assignment.region_name]
            for member in target_members.intersection(assigned_members):
                effective[member] = assignment.beam_orientation
    except (KeyError, TypeError, ValueError) as error:
        return _rejected(operation, target, str(error))

    parallel: list[PreflightDiagnostic] = []
    missing = tuple(
        sorted(
            member
            for member, orientation in effective.items()
            if orientation is None
        )
    )
    for member, orientation in effective.items():
        if orientation is None:
            continue
        reference = _normalized(orientation.local_y_reference)
        tangent = tangents[member]
        dot = sum(
            reference[index] * tangent[index]
            for index in range(3)
        )
        projected = tuple(
            reference[index] - dot * tangent[index]
            for index in range(3)
        )
        if _length(projected) <= BEAM_ORIENTATION_PARALLEL_TOLERANCE:
            parallel.append(
                PreflightDiagnostic(
                    code="beam.orientation.parallel",
                    severity=PreflightSeverity.ERROR,
                    stage=PreflightStage.CAPABILITY,
                    message=(
                        f"{member} 的 Beam 局部 y 参考方向与构件轴线平行"
                    ),
                    subject=target,
                    path=(
                        "capabilities",
                        "operations",
                        operation,
                    ),
                    remediation="请为该构件选择不平行于轴线的局部 y 参考方向。",
                    details={
                        "operation": operation,
                        "member": member,
                        "reference": reference,
                        "tangent": tangent,
                    },
                )
            )
    if parallel:
        return AuthoringCapability(
            operation,
            AuthoringStatus.UNAVAILABLE,
            tuple(parallel),
        )
    if missing and _requires_orientation_requirement(
        capability_report,
        operation,
        "beam.orientation.explicit",
    ):
        diagnostic = PreflightDiagnostic(
            code="beam.orientation.assumed",
            severity=PreflightSeverity.WARNING,
            stage=PreflightStage.CAPABILITY,
            message=(
                f"{operation} 的目标中有 {len(missing)} 个 Beam member "
                "尚未设置显式方向"
            ),
            subject=target,
            path=("capabilities", "operations", operation),
            remediation="请先为所有目标 Beam member 分配显式局部 y 参考方向。",
            details={
                "operation": operation,
                "members": missing,
            },
        )
        return AuthoringCapability(
            operation,
            AuthoringStatus.LIMITED,
            (diagnostic,),
        )
    return AuthoringCapability(
        operation,
        AuthoringStatus.ENABLED,
        (),
    )


def _wire_regions_and_tangents(
    snapshot: Any,
) -> tuple[
    dict[str, frozenset[str]],
    dict[str, tuple[float, float, float]],
]:
    recipe = getattr(snapshot, "geometry_recipe", None)
    points, members = _wire_geometry(recipe)
    tangents = {
        logical_id: _normalized(
            tuple(
                points[end][index] - points[start][index]
                for index in range(3)
            )
        )
        for logical_id, (start, end) in members.items()
    }
    all_members = frozenset(tangents)
    memberships: dict[str, frozenset[str]] = {}
    descriptors = describe_native_regions(
        recipe,
        _mapping_values(getattr(snapshot, "named_regions", ())),
        mesh_settings=getattr(snapshot, "mesh_settings", None),
    )
    for descriptor in descriptors:
        if "beam_element_set" not in descriptor.products:
            continue
        if isinstance(descriptor.source, CompiledDomainRegionSource):
            memberships[descriptor.name] = all_members
        elif isinstance(
            descriptor.source,
            LogicalReferencesRegionSource,
        ):
            memberships[descriptor.name] = frozenset(
                reference.logical_id
                for reference in descriptor.source.references
                if reference.kind == "edge"
            )
        elif isinstance(descriptor.source, MeshEntitiesRegionSource):
            memberships[descriptor.name] = _mesh_scope_wire_members(
                snapshot,
                descriptor.source.references,
                points,
                members,
            )
        else:
            raise ValueError(
                f"unsupported native Wire region source: {descriptor.name}"
            )
    return memberships, tangents


def _mesh_scope_wire_members(
    snapshot: Any,
    references: Iterable[Any],
    points: Mapping[str, tuple[float, float, float]],
    members: Mapping[str, tuple[str, str]],
) -> frozenset[str]:
    """Resolve selected line elements back to their authored wire members."""

    model = getattr(snapshot, "model", None)
    if model is None:
        raise ValueError("mesh Wire scopes require a generated model")
    selected_ids = {
        int(reference.element_id)
        for reference in references
        if getattr(reference, "kind", None) == "element"
    }
    element_lookup = {
        int(element.id): element
        for element in model.mesh.elements
    }
    node_lookup = {
        int(node.id): (
            float(node.x),
            float(node.y),
            float(getattr(node, "z", 0.0)),
        )
        for node in model.mesh.nodes
    }
    missing = selected_ids.difference(element_lookup)
    if missing:
        raise ValueError(
            f"mesh Wire scope references unknown element {min(missing)}"
        )
    result: set[str] = set()
    for logical_id, (start_name, end_name) in members.items():
        start = points[start_name]
        end = points[end_name]
        if any(
            all(
                _point_on_segment(node_lookup[int(node_id)], start, end)
                for node_id in element_lookup[element_id].node_ids
            )
            for element_id in selected_ids
        ):
            result.add(logical_id)
    return frozenset(result)


def _point_on_segment(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> bool:
    segment = tuple(end[index] - start[index] for index in range(3))
    offset = tuple(point[index] - start[index] for index in range(3))
    squared_length = sum(value * value for value in segment)
    if squared_length <= 0.0:
        return False
    parameter = sum(
        offset[index] * segment[index] for index in range(3)
    ) / squared_length
    tolerance = 1.0e-8
    if parameter < -tolerance or parameter > 1.0 + tolerance:
        return False
    residual = tuple(
        offset[index] - parameter * segment[index]
        for index in range(3)
    )
    return _length(residual) <= tolerance * max(
        1.0,
        math.sqrt(squared_length),
    )


def _wire_geometry(
    recipe: Any,
) -> tuple[
    dict[str, tuple[float, float, float]],
    dict[str, tuple[str, str]],
]:
    if isinstance(recipe, WireGeometry):
        return (
            {
                point.name: (point.x, point.y, point.z)
                for point in recipe.points
            },
            {
                f"edge:{member.name}": (member.start, member.end)
                for member in recipe.members
            },
        )
    if isinstance(recipe, MovedGeometry):
        points, members = _wire_geometry(recipe.base)
        return (
            {
                name: (
                    point[0] + recipe.dx,
                    point[1] + recipe.dy,
                    point[2] + recipe.dz,
                )
                for name, point in points.items()
            },
            members,
        )
    if isinstance(recipe, RotatedGeometry):
        points, members = _wire_geometry(recipe.base)
        angle = math.radians(recipe.angle_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)
        return (
            {
                name: _rotate_point(
                    point,
                    recipe.axis,
                    cosine,
                    sine,
                )
                for name, point in points.items()
            },
            members,
        )
    raise ValueError("Beam orientation requires native Wire geometry")


def _rotate_point(
    point: tuple[float, float, float],
    axis: str,
    cosine: float,
    sine: float,
) -> tuple[float, float, float]:
    x, y, z = point
    if axis == "x":
        return x, y * cosine - z * sine, y * sine + z * cosine
    if axis == "y":
        return x * cosine + z * sine, y, -x * sine + z * cosine
    return x * cosine - y * sine, x * sine + y * cosine, z


def _requires_orientation_requirement(
    capability_report: ModelCapabilityReport,
    operation: str,
    requirement_code: str,
) -> bool:
    return any(
        requirement.code == requirement_code
        and operation in requirement.operations
        for element_type in capability_report.canonical_element_types
        for requirement in get_element_capabilities(element_type).requirements
    )


def _section_operation(section: Any) -> str:
    section_type = str(section.section_type).strip().casefold()
    if section_type == "beam":
        section_type = str(
            section.properties.get("section_type", section_type)
        ).strip().casefold()
    return f"section.{section_type}"


def _normalized(
    vector: Iterable[float],
) -> tuple[float, float, float]:
    owned = tuple(float(value) for value in vector)
    if len(owned) != 3:
        raise ValueError("Beam vector must contain exactly three components")
    length = _length(owned)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("Beam vector must be finite and nonzero")
    return tuple(value / length for value in owned)


def _length(vector: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _mapping_values(value: Any) -> tuple[Any, ...]:
    return tuple(value.values()) if isinstance(value, Mapping) else tuple(value)


def _rejected(
    operation: str,
    subject: object,
    message: str,
) -> AuthoringCapability:
    diagnostic = PreflightDiagnostic(
        code="native.authoring.candidate.invalid",
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.CAPABILITY,
        message=message,
        subject=subject,
        path=("capabilities", "operations", operation),
        remediation="请修正候选定义或选择兼容的 native 区域。",
    )
    return AuthoringCapability(
        operation,
        AuthoringStatus.UNAVAILABLE,
        (diagnostic,),
    )


__all__ = [
    "evaluate_native_assignment_candidate",
    "evaluate_native_line_load_candidate",
]
