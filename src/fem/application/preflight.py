"""Structured, selected-Step linear-static preflight."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from fem import materials
from fem.boundary.step import boundary_for_step, get_step
from fem.core.validation import (
    validate_analysis_step,
    validate_model_structure,
)
from fem.solvers import static_linear

from .beam_frames import resolve_effective_beam_frames
from .capabilities import (
    RegionRef,
    _assumed_orientation_diagnostic,
    _diagnostic_for_operation,
    _requires_explicit_beam_orientation,
    describe_model_capabilities,
)
from .diagnostics import (
    PreflightDiagnostic,
    PreflightFacts,
    PreflightReport,
    PreflightSeverity,
    PreflightStage,
    internal_error_report,
)
from .revisions import TaskToken


def run_static_preflight(
    model: Any,
    step: Any = None,
    *,
    token: TaskToken | None = None,
) -> PreflightReport:
    """Check one detached model Step and return stable diagnostics."""

    owned_model = deepcopy(model)
    requested_name = _requested_step_name(step, token)
    provenance = _report_provenance(token)
    diagnostics: list[PreflightDiagnostic] = []
    numerical_stability_checked = False
    selected_step = None
    boundary = None

    try:
        capability_report = describe_model_capabilities(owned_model)
        diagnostics.extend(capability_report.diagnostics)
    except Exception as error:
        capability_report = None
        diagnostics.append(
            _diagnostic(
                "model.capability.unsupported_mix",
                PreflightStage.CAPABILITY,
                error,
                subject="model",
                remediation=(
                    "请使用已注册且具有完整 capability descriptor 的单元。"
                ),
            )
        )

    structure_valid = True
    try:
        validate_model_structure(owned_model)
    except Exception as error:
        structure_valid = False
        diagnostics.append(
            _diagnostic(
                "model.structure.invalid",
                PreflightStage.STRUCTURE,
                error,
                subject="model",
                remediation="请修复网格、集合或模型结构引用。",
            )
        )

    try:
        selected_step = get_step(owned_model, step)
    except Exception as error:
        diagnostics.append(
            _diagnostic(
                "step.reference.invalid",
                PreflightStage.STEP,
                error,
                subject=requested_name,
                path=("steps", requested_name),
                remediation="请选择当前模型中存在的分析步。",
            )
        )
    report_step_name = (
        str(selected_step.name)
        if selected_step is not None
        else requested_name
    )

    step_valid = selected_step is not None
    if structure_valid and selected_step is not None:
        try:
            validate_analysis_step(owned_model, selected_step)
        except Exception as error:
            step_valid = False
            diagnostics.append(
                _diagnostic(
                    "step.reference.invalid",
                    PreflightStage.STEP,
                    error,
                    subject=report_step_name,
                    path=("steps", report_step_name),
                    remediation=(
                        "请修复当前分析步及其继承的 Initial 边界引用。"
                    ),
                )
            )

    procedure_valid = _validate_static_procedure(
        selected_step,
        diagnostics,
        report_step_name,
    )

    definitions_valid = _append_definition_diagnostics(
        owned_model,
        diagnostics,
    )
    _append_beam_orientation_diagnostics(
        owned_model,
        selected_step,
        diagnostics,
    )

    if structure_valid and step_valid and selected_step is not None:
        try:
            boundary = boundary_for_step(owned_model, selected_step)
        except Exception as error:
            diagnostics.append(
                _diagnostic(
                    "step.reference.invalid",
                    PreflightStage.BOUNDARY,
                    error,
                    subject=report_step_name,
                    path=("steps", report_step_name, "boundary"),
                    remediation="请修复当前分析步的边界和载荷定义。",
                )
            )
        else:
            if not boundary.prescribed_displacements:
                diagnostics.append(
                    PreflightDiagnostic(
                        code="static.boundary.missing_displacement",
                        severity=PreflightSeverity.ERROR,
                        stage=PreflightStage.BOUNDARY,
                        message=(
                            "The selected Step has no effective displacement "
                            "constraints."
                        ),
                        subject=report_step_name,
                        path=(
                            "steps",
                            report_step_name,
                            "boundaries",
                        ),
                        remediation="请为当前分析步或 Initial 步添加位移约束。",
                    )
                )

    _append_output_diagnostic(
        selected_step,
        diagnostics,
        report_step_name,
    )

    if (
        structure_valid
        and step_valid
        and procedure_valid
        and definitions_valid
        and boundary is not None
        and boundary.prescribed_displacements
        and not _has_blocking_diagnostic(diagnostics)
    ):
        numerical_stability_checked = True
        try:
            static_linear.validate_stiffness(
                owned_model,
                selected_step,
            )
        except Exception as error:
            diagnostics.append(
                _diagnostic(
                    "static.stiffness.singular",
                    PreflightStage.STIFFNESS,
                    error,
                    subject=report_step_name,
                    path=("steps", report_step_name, "stiffness"),
                    remediation=(
                        "请检查约束、材料、截面、单元连接和零刚度自由度。"
                    ),
                )
            )

    facts = _preflight_facts(
        owned_model,
        selected_step,
        boundary,
        report_step_name,
    )
    return PreflightReport(
        step_name=report_step_name,
        diagnostics=tuple(diagnostics),
        facts=facts,
        numerical_stability_checked=numerical_stability_checked,
        **provenance,
    )


def safe_static_preflight(
    model: Any,
    step: Any = None,
    *,
    token: TaskToken | None = None,
) -> PreflightReport:
    """Convert an unexpected preflight invariant failure into a typed report."""

    try:
        return run_static_preflight(model, step, token=token)
    except Exception as error:
        step_name = _requested_step_name(step, token)
        return internal_error_report(
            step_name,
            error,
            **_report_provenance(token),
        )


def _append_definition_diagnostics(
    model: Any,
    diagnostics: list[PreflightDiagnostic],
) -> bool:
    materials_by_name = getattr(model, "materials", {})
    sections = tuple(getattr(model, "sections", ()))
    if not materials_by_name:
        diagnostics.append(
            PreflightDiagnostic(
                code="definition.material.missing",
                severity=PreflightSeverity.ERROR,
                stage=PreflightStage.DEFINITIONS,
                message="The model has no material definitions.",
                subject="materials",
                path=("definitions", "materials"),
                remediation="请创建至少一个与单元族兼容的材料。",
            )
        )
    if not sections:
        diagnostics.append(
            PreflightDiagnostic(
                code="definition.section.missing",
                severity=PreflightSeverity.ERROR,
                stage=PreflightStage.DEFINITIONS,
                message="The model has no section assignments.",
                subject="sections",
                path=("definitions", "sections"),
                remediation="请创建截面并分配到单元集。",
            )
        )

    try:
        resolution = materials.resolve_sections(model)
    except Exception as error:
        diagnostics.append(
            _diagnostic(
                "definition.section.invalid",
                PreflightStage.DEFINITIONS,
                error,
                subject="sections",
                path=("definitions", "sections"),
                remediation="请修复材料和截面参数。",
            )
        )
        return False

    for issue in resolution.issues:
        code = (
            "definition.section.missing"
            if issue.code == "definition.section.reference_missing"
            else issue.code
        )
        subject: Any = (
            RegionRef("element_set", issue.element_set)
            if issue.element_set
            else issue.element_id
        )
        details = {
            "assignment_index": issue.assignment_index,
            "element_set": issue.element_set,
            "element_id": issue.element_id,
            "material": issue.material,
            "section_type": issue.section_type,
        }
        if str(code).startswith("beam.orientation."):
            details["operation"] = "section.assignment"
            assignment_index = issue.assignment_index
            if (
                assignment_index is not None
                and 0 <= assignment_index < len(sections)
            ):
                properties = getattr(
                    sections[assignment_index],
                    "properties",
                    {},
                )
                if "beam_local_y_reference" in properties:
                    details["reference"] = deepcopy(
                        properties["beam_local_y_reference"]
                    )
        diagnostics.append(
            PreflightDiagnostic(
                code=code,
                severity=PreflightSeverity.ERROR,
                stage=PreflightStage.DEFINITIONS,
                message=issue.message,
                subject=subject,
                path=(
                    "definitions",
                    "sections",
                    str(issue.assignment_index),
                ),
                remediation="请修复材料、截面及其单元集分配。",
                details=details,
            )
        )
    if resolution.uncovered_element_ids:
        diagnostics.append(
            PreflightDiagnostic(
                code="definition.section.unassigned_elements",
                severity=PreflightSeverity.ERROR,
                stage=PreflightStage.DEFINITIONS,
                message=(
                    f"{len(resolution.uncovered_element_ids)} elements have "
                    "no valid explicit section assignment."
                ),
                subject=resolution.uncovered_element_ids,
                path=("definitions", "sections", "coverage"),
                remediation="请将兼容截面分配到所有单元。",
                details={
                    "element_ids": resolution.uncovered_element_ids,
                },
            )
        )
    return (
        bool(materials_by_name)
        and bool(sections)
        and not resolution.issues
        and not resolution.uncovered_element_ids
    )


def _append_beam_orientation_diagnostics(
    model: Any,
    selected_step: Any,
    diagnostics: list[PreflightDiagnostic],
) -> None:
    """Validate effective installed frames before numerical stiffness."""

    visited_sections: set[RegionRef] = set()
    for section in getattr(model, "sections", ()):
        section_type = _effective_section_type(section)
        if section_type not in {
            "rectangle",
            "solid_circle",
            "hollow_circle",
        }:
            continue
        try:
            target = RegionRef(
                "element_set",
                str(getattr(section, "element_set", "")),
            )
        except ValueError:
            continue
        if target in visited_sections:
            continue
        visited_sections.add(target)
        report = resolve_effective_beam_frames(model, target)
        for diagnostic in report.diagnostics:
            if (
                not diagnostic.code.startswith("beam.orientation.")
                and diagnostic.code != "model.structure.invalid"
            ):
                continue
            _append_unique_diagnostic(
                diagnostics,
                _diagnostic_for_operation(
                    diagnostic,
                    "section.assignment",
                ),
            )
        rectangle_automatic = tuple(
            entry
            for entry in report.entries
            if (
                entry.frame.source != "explicit"
                and str(entry.section_type).strip().casefold()
                == "rectangle"
            )
        )
        if (
            rectangle_automatic
            and _requires_explicit_beam_orientation(
                model,
                target,
                "section.rectangle",
            )
        ):
            _append_unique_diagnostic(
                diagnostics,
                _assumed_orientation_diagnostic(
                    target,
                    "section.rectangle",
                    rectangle_automatic,
                ),
            )

    if selected_step is None:
        return
    for load_index, line_load in enumerate(
        getattr(selected_step, "line_loads", ())
    ):
        if (
            str(getattr(line_load, "coordinate_system", ""))
            .strip()
            .casefold()
            != "local"
        ):
            continue
        raw_target = getattr(line_load, "target", None)
        try:
            target: RegionRef | int = (
                RegionRef("element_set", raw_target)
                if isinstance(raw_target, str)
                else raw_target
            )
        except (TypeError, ValueError):
            continue
        report = resolve_effective_beam_frames(model, target)
        for diagnostic in report.diagnostics:
            if (
                not diagnostic.code.startswith("beam.orientation.")
                and diagnostic.code != "model.structure.invalid"
            ):
                continue
            contextual = _diagnostic_for_operation(
                diagnostic,
                "load.line.local",
            )
            details = contextual.details_dict()
            details["step"] = str(getattr(selected_step, "name", ""))
            details["load_index"] = load_index
            _append_unique_diagnostic(
                diagnostics,
                PreflightDiagnostic(
                    code=contextual.code,
                    severity=contextual.severity,
                    stage=contextual.stage,
                    message=contextual.message,
                    subject=contextual.subject,
                    path=contextual.path,
                    remediation=contextual.remediation,
                    details=details,
                ),
            )
        automatic = tuple(
            entry
            for entry in report.entries
            if entry.frame.source != "explicit"
        )
        if (
            automatic
            and _requires_explicit_beam_orientation(
                model,
                target,
                "load.line.local",
            )
        ):
            warning = _assumed_orientation_diagnostic(
                target,
                "load.line.local",
                automatic,
            )
            details = warning.details_dict()
            details["step"] = str(getattr(selected_step, "name", ""))
            details["load_index"] = load_index
            _append_unique_diagnostic(
                diagnostics,
                PreflightDiagnostic(
                    code=warning.code,
                    severity=warning.severity,
                    stage=warning.stage,
                    message=warning.message,
                    subject=warning.subject,
                    path=warning.path,
                    remediation=warning.remediation,
                    details=details,
                ),
            )


def _effective_section_type(section: Any) -> str:
    section_type = str(
        getattr(section, "section_type", "")
    ).strip().casefold()
    if section_type == "beam":
        section_type = str(
            getattr(section, "properties", {}).get(
                "section_type",
                section_type,
            )
        ).strip().casefold()
    return section_type


def _append_unique_diagnostic(
    diagnostics: list[PreflightDiagnostic],
    diagnostic: PreflightDiagnostic,
) -> None:
    identity = _diagnostic_identity(diagnostic)
    if any(
        _diagnostic_identity(existing) == identity
        for existing in diagnostics
    ):
        return
    diagnostics.append(diagnostic)


def _diagnostic_identity(
    diagnostic: PreflightDiagnostic,
) -> str:
    return repr(
        (
            diagnostic.code,
            diagnostic.severity.value,
            diagnostic.stage.value,
            diagnostic.subject,
            diagnostic.path,
            diagnostic.details,
        )
    )


def _validate_static_procedure(
    step: Any,
    diagnostics: list[PreflightDiagnostic],
    step_name: str,
) -> bool:
    if step is None:
        return False
    procedure = str(getattr(step, "procedure", "")).strip().casefold()
    metadata = getattr(step, "metadata", {})
    nlgeom = next(
        (
            value
            for key, value in metadata.items()
            if str(key).strip().casefold() == "nlgeom"
        ),
        None,
    )
    if procedure == "static" and not _truthy_option(nlgeom):
        return True
    message = (
        "The current solver supports only linear static Steps "
        "with nlgeom disabled."
    )
    diagnostics.append(
        PreflightDiagnostic(
            code="static.procedure.unsupported",
            severity=PreflightSeverity.ERROR,
            stage=PreflightStage.STEP,
            message=message,
            subject=step_name,
            path=("steps", step_name, "procedure"),
            remediation="请选择线性静力过程并关闭 nlgeom。",
            details={"procedure": procedure, "nlgeom": nlgeom},
        )
    )
    return False


def _append_output_diagnostic(
    step: Any,
    diagnostics: list[PreflightDiagnostic],
    step_name: str,
) -> None:
    outputs = tuple(getattr(step, "outputs", ())) if step is not None else ()
    if not outputs:
        return
    diagnostics.append(
        PreflightDiagnostic(
            code="output.request.not_executed",
            severity=PreflightSeverity.WARNING,
            stage=PreflightStage.OUTPUT,
            message=(
                "Output requests are preserved but are not executed by the "
                "current solver."
            ),
            subject=step_name,
            path=("steps", step_name, "outputs"),
            remediation="既有输出请求可查看或删除；求解结果不会按其裁剪。",
            details={"count": len(outputs)},
        )
    )


def _preflight_facts(
    model: Any,
    step: Any,
    boundary: Any,
    step_name: str,
) -> PreflightFacts:
    mesh = getattr(model, "mesh", None)
    return PreflightFacts(
        model_name=getattr(model, "name", None),
        step_name=step_name,
        procedure=(
            str(getattr(step, "procedure", "")) if step is not None else ""
        ),
        node_count=len(getattr(mesh, "nodes", ())),
        element_count=len(getattr(mesh, "elements", ())),
        dof_count=int(getattr(mesh, "num_dofs", 0)),
        material_count=len(getattr(model, "materials", {})),
        section_count=len(getattr(model, "sections", ())),
        displacement_count=(
            len(boundary.prescribed_displacements)
            if boundary is not None
            else 0
        ),
        nodal_load_count=(
            len(boundary.nodal_forces) if boundary is not None else 0
        ),
        edge_load_count=len(getattr(step, "edge_loads", ())),
        surface_load_count=len(getattr(step, "surface_loads", ())),
        line_load_count=len(getattr(step, "line_loads", ())),
        gravity_load_count=len(getattr(step, "gravity_loads", ())),
    )


def _requested_step_name(
    step: Any,
    token: TaskToken | None,
) -> str:
    if token is not None and token.step_name is not None:
        return str(token.step_name)
    if hasattr(step, "name"):
        return str(step.name)
    return "" if step is None else str(step)


def _report_provenance(
    token: TaskToken | None,
) -> dict[str, Any]:
    if token is None:
        return {
            "session_id": None,
            "artifact_id": None,
            "model_revision": None,
        }
    if token.task_kind != "validation":
        raise ValueError("preflight token must be a validation task token")
    revisions = dict(token.dependency_revisions)
    return {
        "session_id": token.session_id,
        "artifact_id": token.artifact_id,
        "model_revision": revisions.get("model_revision"),
    }


def _diagnostic(
    code: str,
    stage: PreflightStage,
    error: Any,
    *,
    subject: Any,
    path: Iterable[str] = (),
    remediation: str,
) -> PreflightDiagnostic:
    return PreflightDiagnostic(
        code=code,
        severity=PreflightSeverity.ERROR,
        stage=stage,
        message=str(error),
        subject=subject,
        path=tuple(path),
        remediation=remediation,
        details={"error_type": type(error).__name__},
    )


def _has_blocking_diagnostic(
    diagnostics: Iterable[PreflightDiagnostic],
) -> bool:
    return any(item.blocking for item in diagnostics)


def _truthy_option(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"", "0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


__all__ = ["run_static_preflight", "safe_static_preflight"]
