"""Read-only effective Beam2 frame queries for application consumers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from fem import materials
from fem.elements import (
    BEAM_LOCAL_Y_REFERENCE_KEY,
    BeamFrame,
    BeamOrientation,
    BeamOrientationError,
    get_element_capabilities,
    resolve_beam_frame,
)

from .capabilities import RegionRef
from .diagnostics import (
    PreflightDiagnostic,
    PreflightSeverity,
    PreflightStage,
)


@dataclass(frozen=True, slots=True)
class EffectiveBeamFrame:
    """One effective frame together with its section-assignment provenance."""

    element_id: int
    frame: BeamFrame
    assignment_index: int | None = None
    element_set: str | None = None
    section_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "element_id", int(self.element_id))
        if not isinstance(self.frame, BeamFrame):
            raise TypeError("effective Beam frame entry requires BeamFrame")
        if self.assignment_index is not None:
            object.__setattr__(
                self,
                "assignment_index",
                int(self.assignment_index),
            )
        if self.element_set is not None:
            object.__setattr__(self, "element_set", str(self.element_set))
        if self.section_type is not None:
            object.__setattr__(self, "section_type", str(self.section_type))


@dataclass(frozen=True, slots=True)
class BeamFrameReport:
    """Effective Beam2 frames and diagnostics for one typed target."""

    target: RegionRef | int
    element_ids: tuple[int, ...]
    entries: tuple[EffectiveBeamFrame, ...] = ()
    diagnostics: tuple[PreflightDiagnostic, ...] = ()
    suggested_orientation: BeamOrientation | None = None

    def __post_init__(self) -> None:
        if isinstance(self.target, bool) or not isinstance(
            self.target,
            (RegionRef, int),
        ):
            raise TypeError("Beam frame report target must be RegionRef or int")
        object.__setattr__(
            self,
            "element_ids",
            tuple(int(value) for value in self.element_ids),
        )
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if any(
            not isinstance(item, EffectiveBeamFrame)
            for item in self.entries
        ):
            raise TypeError(
                "Beam frame report entries must be EffectiveBeamFrame values"
            )
        if any(
            not isinstance(item, PreflightDiagnostic)
            for item in self.diagnostics
        ):
            raise TypeError(
                "Beam frame report diagnostics must be "
                "PreflightDiagnostic values"
            )
        if (
            self.suggested_orientation is not None
            and not isinstance(self.suggested_orientation, BeamOrientation)
        ):
            raise TypeError(
                "suggested orientation must be BeamOrientation or None"
            )

    @property
    def passed(self) -> bool:
        """Return whether every requested element has one valid frame."""

        return (
            not any(item.blocking for item in self.diagnostics)
            and len(self.entries) == len(self.element_ids)
        )

    @property
    def frames(self) -> tuple[BeamFrame, ...]:
        """Return frames in target mesh order."""

        return tuple(item.frame for item in self.entries)

    def for_element(self, element_id: int) -> EffectiveBeamFrame | None:
        """Return one element result, if its frame resolved."""

        target = int(element_id)
        return next(
            (item for item in self.entries if item.element_id == target),
            None,
        )


def resolve_effective_beam_frames(
    model: Any,
    target: RegionRef | int,
) -> BeamFrameReport:
    """Resolve assignment-aware Beam2 frames without mutating the model."""

    mesh = getattr(model, "mesh", None)
    elements = tuple(getattr(mesh, "elements", ()))
    element_lookup = {int(element.id): element for element in elements}
    if len(element_lookup) != len(elements):
        return _target_error_report(
            target,
            "element ids must be unique",
        )

    try:
        element_ids = _target_element_ids(model, target, element_lookup)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        return _target_error_report(target, str(error))

    try:
        resolution = materials.resolve_sections(
            model,
            element_lookup=element_lookup,
        )
    except Exception as error:
        diagnostic = PreflightDiagnostic(
            code="definition.section.invalid",
            severity=PreflightSeverity.ERROR,
            stage=PreflightStage.DEFINITIONS,
            message=str(error),
            subject=target,
            path=("definitions", "sections"),
            remediation="请修复材料、截面及其区域分配。",
            details={"error_type": type(error).__name__},
        )
        return BeamFrameReport(
            target,
            element_ids,
            diagnostics=(diagnostic,),
        )

    blocking_issues = _relevant_resolution_issues(
        model,
        resolution,
        element_ids,
        target,
    )
    blocking_issues = (
        *blocking_issues,
        *_declared_orientation_diagnostics(
            model,
            resolution,
            element_ids,
            target,
            element_lookup,
        ),
    )
    blocked_elements = {
        int(value)
        for diagnostic in blocking_issues
        for key, value in diagnostic.details
        if key == "element_id" and value is not None
    }
    node_lookup = {
        int(node.id): node
        for node in tuple(getattr(mesh, "nodes", ()))
    }
    entries: list[EffectiveBeamFrame] = []
    diagnostics = list(blocking_issues)
    properties_by_element: dict[int, dict[str, Any]] = {}

    for element_id in element_ids:
        if element_id in blocked_elements:
            continue
        element = element_lookup[element_id]
        try:
            descriptor = get_element_capabilities(str(element.type))
        except (NotImplementedError, TypeError, ValueError) as error:
            diagnostics.append(
                _unsupported_target_diagnostic(
                    target,
                    element_id,
                    str(error),
                )
            )
            continue
        if descriptor.family != "beam":
            diagnostics.append(
                _unsupported_target_diagnostic(
                    target,
                    element_id,
                    (
                        f"element {element_id} is {descriptor.canonical_type}, "
                        "not Beam2"
                    ),
                )
            )
            continue

        effective = resolution.for_element(element_id)
        properties = (
            materials.restored_element_properties(
                model,
                element_id,
                element,
            )
            if effective is None
            else deepcopy(effective.effective_properties)
        )
        try:
            frame = resolve_beam_frame(
                mesh,
                element,
                node_lookup,
                properties=properties,
            )
        except BeamOrientationError as error:
            diagnostics.append(
                _orientation_error_diagnostic(
                    error,
                    target,
                    element_id,
                    effective,
                )
            )
            continue
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            diagnostics.append(
                _structure_error_diagnostic(
                    error,
                    target,
                    element_id,
                )
            )
            continue

        properties_by_element[element_id] = properties
        entries.append(
            EffectiveBeamFrame(
                element_id=element_id,
                frame=frame,
                assignment_index=(
                    None if effective is None else effective.assignment_index
                ),
                element_set=(
                    None if effective is None else effective.element_set
                ),
                section_type=(
                    None if effective is None else effective.section_type
                ),
            )
        )

    suggested = (
        _suggest_orientation(
            mesh,
            element_lookup,
            node_lookup,
            tuple(entries),
            properties_by_element,
        )
        if (
            len(entries) == len(element_ids)
            and not any(item.blocking for item in diagnostics)
        )
        else None
    )
    return BeamFrameReport(
        target=target,
        element_ids=element_ids,
        entries=tuple(entries),
        diagnostics=tuple(diagnostics),
        suggested_orientation=suggested,
    )


def _target_element_ids(
    model: Any,
    target: RegionRef | int,
    element_lookup: dict[int, Any],
) -> tuple[int, ...]:
    if isinstance(target, bool):
        raise TypeError("Beam frame target must be a RegionRef or element id")
    if isinstance(target, int):
        if target not in element_lookup:
            raise KeyError(f"element {target} is not defined")
        return (target,)
    if not isinstance(target, RegionRef):
        raise TypeError("Beam frame target must be a RegionRef or element id")
    if target.kind != "element_set":
        raise ValueError("Beam frame region target must be an element_set")

    public = getattr(model, "element_sets", {})
    internal = getattr(model, "metadata", {}).get(
        "_abaqus_internal_element_sets",
        {},
    )
    if target.name in public:
        element_set = public[target.name]
    elif target.name in internal:
        element_set = internal[target.name]
    else:
        raise KeyError(f"element set {target.name} is not defined")

    requested = tuple(int(value) for value in element_set.element_ids)
    missing = tuple(
        element_id
        for element_id in requested
        if element_id not in element_lookup
    )
    if missing:
        raise KeyError(f"element {missing[0]} is not defined")
    requested_set = set(requested)
    return tuple(
        element_id
        for element_id in element_lookup
        if element_id in requested_set
    )


def _relevant_resolution_issues(
    model: Any,
    resolution: Any,
    element_ids: tuple[int, ...],
    target: RegionRef | int,
) -> tuple[PreflightDiagnostic, ...]:
    requested = set(element_ids)
    assignment_by_index = {
        int(assignment.assignment_index): assignment
        for assignment in resolution.assignments
    }
    diagnostics: list[PreflightDiagnostic] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for issue in resolution.issues:
        affected = (
            (int(issue.element_id),)
            if issue.element_id is not None
            else tuple(
                element_id
                for element_id in getattr(
                    assignment_by_index.get(issue.assignment_index),
                    "element_ids",
                    (),
                )
                if int(element_id) in requested
            )
        )
        for element_id in affected:
            if element_id not in requested:
                continue
            code = (
                issue.code
                if str(issue.code).startswith("beam.orientation.")
                else issue.code
            )
            identity = (str(code), issue.assignment_index, element_id)
            if identity in seen:
                continue
            seen.add(identity)
            details = {
                "element_id": element_id,
                "assignment_index": issue.assignment_index,
                "element_set": issue.element_set,
                "material": issue.material,
                "section_type": issue.section_type,
            }
            reference = _resolution_issue_reference(model, issue)
            if reference is not None:
                details["reference"] = reference
            diagnostics.append(
                PreflightDiagnostic(
                    code=code,
                    severity=PreflightSeverity.ERROR,
                    stage=PreflightStage.DEFINITIONS,
                    message=issue.message,
                    subject=target,
                    path=(
                        "definitions",
                        "sections",
                        str(issue.assignment_index),
                    ),
                    remediation=(
                        "请修复梁截面方向、材料或截面区域分配。"
                    ),
                    details=details,
                )
            )
    return tuple(diagnostics)


def _resolution_issue_reference(model: Any, issue: Any) -> Any:
    if not str(issue.code).startswith("beam.orientation."):
        return None
    assignment_index = issue.assignment_index
    sections = tuple(getattr(model, "sections", ()))
    if (
        assignment_index is None
        or assignment_index < 0
        or assignment_index >= len(sections)
    ):
        return None
    properties = getattr(sections[assignment_index], "properties", {})
    if BEAM_LOCAL_Y_REFERENCE_KEY not in properties:
        return None
    return deepcopy(properties[BEAM_LOCAL_Y_REFERENCE_KEY])


def _declared_orientation_diagnostics(
    model: Any,
    resolution: Any,
    element_ids: tuple[int, ...],
    target: RegionRef | int,
    element_lookup: dict[int, Any],
) -> tuple[PreflightDiagnostic, ...]:
    """Validate explicit directions even when a later assignment shadows them."""

    requested = set(element_ids)
    sections = tuple(getattr(model, "sections", ()))
    materials_by_name = getattr(model, "materials", {})
    diagnostics: list[PreflightDiagnostic] = []
    for assignment in resolution.assignments:
        assignment_index = int(assignment.assignment_index)
        if assignment_index < 0 or assignment_index >= len(sections):
            continue
        section = sections[assignment_index]
        properties = getattr(section, "properties", {})
        if BEAM_LOCAL_Y_REFERENCE_KEY not in properties:
            continue
        material = materials_by_name.get(str(assignment.material))
        if material is None:
            continue
        for raw_element_id in assignment.element_ids:
            element_id = int(raw_element_id)
            if element_id not in requested:
                continue
            element = element_lookup.get(element_id)
            if element is None:
                continue
            try:
                resolved = materials.resolve_section_properties(
                    str(element.type),
                    material.properties,
                    str(section.section_type),
                    properties,
                    baseline_properties=(
                        materials.restored_element_properties(
                            model,
                            element_id,
                            element,
                        )
                    ),
                )
            except (
                KeyError,
                NotImplementedError,
                TypeError,
                ValueError,
            ):
                # Section-resolution issues are reported by the companion
                # diagnostic path above.
                continue
            try:
                resolve_beam_frame(
                    model.mesh,
                    element,
                    properties=resolved.effective_properties,
                )
            except BeamOrientationError as error:
                diagnostics.append(
                    _orientation_error_diagnostic(
                        error,
                        target,
                        element_id,
                        assignment,
                    )
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                # Connectivity and geometry failures are reported by the
                # effective-frame pass below.  This preliminary pass exists
                # only to retain diagnostics for shadowed declarations.
                continue
    return tuple(diagnostics)


def _orientation_error_diagnostic(
    error: BeamOrientationError,
    target: RegionRef | int,
    element_id: int,
    effective: Any,
) -> PreflightDiagnostic:
    return PreflightDiagnostic(
        code=error.code,
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.DEFINITIONS,
        message=str(error),
        subject=target,
        path=_orientation_path(effective, element_id),
        remediation=(
            "请输入有限、非零且不与任何目标梁轴平行的全局局部 y "
            "参考方向。"
        ),
        details={
            "element_id": element_id,
            "assignment_index": (
                None if effective is None else effective.assignment_index
            ),
            "element_set": (
                None if effective is None else effective.element_set
            ),
            "reference": error.reference,
            "tangent": error.tangent,
        },
    )


def _structure_error_diagnostic(
    error: Exception,
    target: RegionRef | int,
    element_id: int,
) -> PreflightDiagnostic:
    return PreflightDiagnostic(
        code="model.structure.invalid",
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.STRUCTURE,
        message=str(error),
        subject=target,
        path=("mesh", "elements", str(element_id)),
        remediation="请修复梁单元连接关系、节点坐标及几何长度。",
        details={
            "element_id": element_id,
            "error_type": type(error).__name__,
        },
    )


def _orientation_path(
    effective: Any,
    element_id: int,
) -> tuple[str, ...]:
    if effective is not None:
        return (
            "definitions",
            "sections",
            str(effective.assignment_index),
            BEAM_LOCAL_Y_REFERENCE_KEY,
        )
    return (
        "elements",
        str(element_id),
        "properties",
        BEAM_LOCAL_Y_REFERENCE_KEY,
    )


def _unsupported_target_diagnostic(
    target: RegionRef | int,
    element_id: int,
    message: str,
) -> PreflightDiagnostic:
    return PreflightDiagnostic(
        code="beam.orientation.unsupported_target",
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.CAPABILITY,
        message=message,
        subject=target,
        path=("elements", str(element_id)),
        remediation="请选择只包含 Beam2 单元的目标区域。",
        details={"element_id": element_id},
    )


def _target_error_report(
    target: RegionRef | int,
    message: str,
) -> BeamFrameReport:
    diagnostic = PreflightDiagnostic(
        code="step.reference.invalid",
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.CAPABILITY,
        message=message,
        subject=target,
        path=("target",),
        remediation="请选择当前模型中存在的 Beam2 单元或单元集。",
    )
    return BeamFrameReport(
        target=target,
        element_ids=(),
        diagnostics=(diagnostic,),
    )


def _suggest_orientation(
    mesh: Any,
    element_lookup: dict[int, Any],
    node_lookup: dict[int, Any],
    entries: tuple[EffectiveBeamFrame, ...],
    properties_by_element: dict[int, dict[str, Any]],
) -> BeamOrientation | None:
    if not entries:
        return None

    authored = tuple(
        entry.frame.orientation
        for entry in entries
        if entry.frame.source == "explicit"
    )
    if (
        len(authored) == len(entries)
        and authored
        and all(value == authored[0] for value in authored)
    ):
        return authored[0]

    candidate = BeamOrientation(
        tuple(float(value) for value in entries[0].frame.local_y)
    )
    for entry in entries:
        properties = deepcopy(properties_by_element[entry.element_id])
        properties[BEAM_LOCAL_Y_REFERENCE_KEY] = (
            candidate.local_y_reference
        )
        try:
            candidate_frame = resolve_beam_frame(
                mesh,
                element_lookup[entry.element_id],
                node_lookup,
                properties=properties,
            )
        except (BeamOrientationError, KeyError, TypeError, ValueError):
            return None
        if not np.allclose(
            candidate_frame.rotation,
            entry.frame.rotation,
            rtol=0.0,
            atol=1e-10,
        ):
            return None
    return candidate


__all__ = [
    "BeamFrameReport",
    "EffectiveBeamFrame",
    "resolve_effective_beam_frames",
]
