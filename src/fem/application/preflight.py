"""Structured, selected-Step linear-static preflight."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
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
    _aggregate_capabilities,
    _assumed_orientation_diagnostic,
    _diagnostic_for_operation,
    _evaluate_output_requests,
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


@dataclass(frozen=True, slots=True)
class PreparedPreflight:
    """One typed preflight report and its reusable prepared base system."""

    report: PreflightReport
    prepared_system: static_linear.PreparedSystem | None = None


def run_static_preflight(
    model: Any,
    step: Any = None,
    *,
    token: TaskToken | None = None,
    check_numerical_stability: bool = True,
    copy_model: bool = True,
    quick_check: bool = False,
) -> PreflightReport:
    """Check one detached model Step and return stable diagnostics."""

    return _evaluate_static_preflight(
        model,
        step,
        token=token,
        check_numerical_stability=check_numerical_stability,
        copy_model=copy_model,
        quick_check=quick_check,
        retain_prepared_system=False,
    ).report


def prepare_static_preflight(
    model: Any,
    step: Any = None,
    *,
    token: TaskToken | None = None,
    check_numerical_stability: bool = True,
    copy_model: bool = True,
    quick_check: bool = False,
) -> PreparedPreflight:
    """Run preflight and retain a successful numerical base system."""

    return _evaluate_static_preflight(
        model,
        step,
        token=token,
        check_numerical_stability=check_numerical_stability,
        copy_model=copy_model,
        quick_check=quick_check,
        retain_prepared_system=True,
    )


def _evaluate_static_preflight(
    model: Any,
    step: Any = None,
    *,
    token: TaskToken | None,
    check_numerical_stability: bool,
    copy_model: bool,
    quick_check: bool,
    retain_prepared_system: bool,
) -> PreparedPreflight:
    """Evaluate preflight once, optionally retaining prepared stiffness."""

    if type(check_numerical_stability) is not bool:
        raise TypeError("check_numerical_stability must be bool")
    if type(copy_model) is not bool:
        raise TypeError("copy_model must be bool")
    if type(quick_check) is not bool:
        raise TypeError("quick_check must be bool")
    owned_model = deepcopy(model) if copy_model else model
    requested_name = _requested_step_name(step, token)
    provenance = _report_provenance(token)
    diagnostics: list[PreflightDiagnostic] = []
    numerical_stability_checked = False
    prepared_system = None
    selected_step = None
    boundary = None

    if quick_check:
        _append_quick_capability_diagnostics(owned_model, diagnostics)
    else:
        try:
            capability_report = describe_model_capabilities(owned_model)
            diagnostics.extend(capability_report.diagnostics)
        except Exception as error:
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
                        "请修复当前分析步及其继承的前序分析步边界引用。"
                    ),
                )
            )

    procedure_valid = _validate_static_procedure(
        selected_step,
        diagnostics,
        report_step_name,
    )

    definitions_valid = (
        _append_quick_definition_diagnostics(
            owned_model,
            diagnostics,
        )
        if quick_check
        else _append_definition_diagnostics(
            owned_model,
            diagnostics,
        )
    )
    if not quick_check:
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
                        remediation="请为当前分析步或任一前序分析步添加位移约束。",
                    )
                )

    _append_output_diagnostic(
        owned_model,
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
        if check_numerical_stability:
            numerical_stability_checked = True
            try:
                candidate = static_linear.prepare(
                    owned_model,
                    copy_model=False,
                )
                static_linear.validate_stiffness(
                    candidate,
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
            else:
                if retain_prepared_system:
                    prepared_system = candidate
        else:
            diagnostics.append(
                PreflightDiagnostic(
                    code="static.stiffness.skipped_large_model",
                    severity=PreflightSeverity.WARNING,
                    stage=PreflightStage.STIFFNESS,
                    message=(
                        "大型模型快速检查已跳过全局刚度矩阵的数值分解。"
                    ),
                    subject=report_step_name,
                    path=("steps", report_step_name, "stiffness"),
                    remediation=(
                        "提交分析后，求解器仍会执行完整刚度矩阵装配与分解。"
                    ),
                )
            )

    facts = _preflight_facts(
        owned_model,
        selected_step,
        boundary,
        report_step_name,
    )
    return PreparedPreflight(
        report=PreflightReport(
            step_name=report_step_name,
            diagnostics=tuple(diagnostics),
            facts=facts,
            numerical_stability_checked=numerical_stability_checked,
            **provenance,
        ),
        prepared_system=prepared_system,
    )


def safe_static_preflight(
    model: Any,
    step: Any = None,
    *,
    token: TaskToken | None = None,
    check_numerical_stability: bool = True,
    copy_model: bool = True,
    quick_check: bool = False,
) -> PreflightReport:
    """Convert an unexpected preflight invariant failure into a typed report."""

    try:
        return run_static_preflight(
            model,
            step,
            token=token,
            check_numerical_stability=check_numerical_stability,
            copy_model=copy_model,
            quick_check=quick_check,
        )
    except Exception as error:
        step_name = _requested_step_name(step, token)
        return internal_error_report(
            step_name,
            error,
            **_report_provenance(token),
        )


def safe_prepare_static_preflight(
    model: Any,
    step: Any = None,
    *,
    token: TaskToken | None = None,
    check_numerical_stability: bool = True,
    copy_model: bool = True,
    quick_check: bool = False,
) -> PreparedPreflight:
    """Return an internal-error report with no cacheable system on failure."""

    try:
        return prepare_static_preflight(
            model,
            step,
            token=token,
            check_numerical_stability=check_numerical_stability,
            copy_model=copy_model,
            quick_check=quick_check,
        )
    except Exception as error:
        step_name = _requested_step_name(step, token)
        return PreparedPreflight(
            internal_error_report(
                step_name,
                error,
                **_report_provenance(token),
            )
        )


def _append_quick_capability_diagnostics(
    model: Any,
    diagnostics: list[PreflightDiagnostic],
) -> None:
    """Check the model-wide contract without expanding every named region."""

    try:
        elements = getattr(getattr(model, "mesh", None), "elements", ())
        element_types = tuple(
            dict.fromkeys(str(getattr(element, "type", "")) for element in elements)
        )
        aggregate = _aggregate_capabilities(
            element_types,
            subject="model",
        )
        diagnostics.extend(aggregate.diagnostics)
    except Exception as error:
        element_types = ()
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
    diagnostics.append(
        PreflightDiagnostic(
            code="model.capability.sampled_large_model",
            severity=PreflightSeverity.WARNING,
            stage=PreflightStage.CAPABILITY,
            message=(
                "大型模型快速检查按唯一单元类型验证能力与截面，"
                "未逐单元构造完整截面解析对象。"
            ),
            subject="model",
            path=("capabilities",),
            remediation=(
                "提交分析后，求解器仍会对实际单元执行完整截面与刚度验证。"
            ),
            details={"element_types": element_types},
        )
    )


def _append_quick_definition_diagnostics(
    model: Any,
    diagnostics: list[PreflightDiagnostic],
) -> bool:
    """Validate references, coverage, and one schema sample per element type."""

    materials_by_name = getattr(model, "materials", {})
    sections = tuple(getattr(model, "sections", ()))
    definitions_valid = bool(materials_by_name) and bool(sections)
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
        element_lookup = {
            int(element.id): element
            for element in getattr(
                getattr(model, "mesh", None),
                "elements",
                (),
            )
        }
        internal_sets = dict(
            getattr(model, "metadata", {}).get(
                "_abaqus_internal_element_sets",
                {},
            )
        )
        element_sets = {
            **internal_sets,
            **dict(getattr(model, "element_sets", {})),
        }
    except Exception as error:
        diagnostics.append(
            _diagnostic(
                "definition.section.invalid",
                PreflightStage.DEFINITIONS,
                error,
                subject="sections",
                path=("definitions", "sections"),
                remediation="请修复材料、截面参数及单元集引用。",
            )
        )
        return False

    targeted_element_ids: set[int] = set()
    for assignment_index, section in enumerate(sections):
        element_set_name = str(getattr(section, "element_set", ""))
        material_name = str(getattr(section, "material", ""))
        declared_type = str(getattr(section, "section_type", ""))
        section_path = (
            "definitions",
            "sections",
            str(assignment_index),
        )
        material = (
            materials_by_name.get(material_name)
            if hasattr(materials_by_name, "get")
            else None
        )
        element_set = element_sets.get(element_set_name)

        if material is None:
            definitions_valid = False
            _append_quick_section_diagnostic(
                diagnostics,
                code="definition.material.missing",
                message=f"material {material_name} is not defined",
                assignment_index=assignment_index,
                element_set_name=element_set_name,
                material_name=material_name,
                declared_type=declared_type,
                path=section_path,
            )
        if element_set is None:
            definitions_valid = False
            _append_quick_section_diagnostic(
                diagnostics,
                code="definition.section.missing",
                message=f"element set {element_set_name} is not defined",
                assignment_index=assignment_index,
                element_set_name=element_set_name,
                material_name=material_name,
                declared_type=declared_type,
                path=section_path,
            )
            continue

        raw_element_ids = getattr(element_set, "element_ids", None)
        if raw_element_ids is None:
            definitions_valid = False
            _append_quick_section_diagnostic(
                diagnostics,
                code="definition.section.missing",
                message=(
                    f"element set {element_set_name} has no element_ids"
                ),
                assignment_index=assignment_index,
                element_set_name=element_set_name,
                material_name=material_name,
                declared_type=declared_type,
                path=section_path,
            )
            continue

        representatives: dict[str, tuple[int, Any]] = {}
        missing_element_count = 0
        first_missing_element_id: int | None = None
        for raw_element_id in raw_element_ids:
            try:
                element_id = int(raw_element_id)
            except (TypeError, ValueError):
                definitions_valid = False
                missing_element_count += 1
                continue
            element = element_lookup.get(element_id)
            if element is None:
                definitions_valid = False
                missing_element_count += 1
                if first_missing_element_id is None:
                    first_missing_element_id = element_id
                continue
            targeted_element_ids.add(element_id)
            representatives.setdefault(
                str(getattr(element, "type", "")),
                (element_id, element),
            )

        if missing_element_count:
            _append_quick_section_diagnostic(
                diagnostics,
                code="definition.section.missing",
                message=(
                    f"element set {element_set_name} references "
                    f"{missing_element_count} missing or invalid elements"
                ),
                assignment_index=assignment_index,
                element_set_name=element_set_name,
                material_name=material_name,
                declared_type=declared_type,
                path=section_path,
                element_id=first_missing_element_id,
                extra_details={"missing_element_count": missing_element_count},
            )

        if material is None:
            continue
        material_properties = getattr(material, "properties", {})
        section_properties = getattr(section, "properties", {})
        for element_type, (element_id, element) in representatives.items():
            try:
                materials.resolve_section_properties(
                    element_type,
                    material_properties,
                    declared_type,
                    section_properties,
                    baseline_properties=materials.restored_element_properties(
                        model,
                        element_id,
                        element,
                    ),
                )
            except materials.SectionCompatibilityError as caught:
                code = "definition.section.incompatible"
                section_error = caught
            except materials.MaterialPropertyError as caught:
                code = "definition.material.invalid"
                section_error = caught
            except materials.SectionPropertyError as caught:
                code = getattr(
                    caught,
                    "code",
                    "definition.section.invalid",
                )
                section_error = caught
            except NotImplementedError as caught:
                code = "definition.section.incompatible"
                section_error = caught
            except Exception as caught:
                code = "definition.section.invalid"
                section_error = caught
            else:
                continue
            definitions_valid = False
            _append_quick_section_diagnostic(
                diagnostics,
                code=code,
                message=str(section_error),
                assignment_index=assignment_index,
                element_set_name=element_set_name,
                material_name=material_name,
                declared_type=declared_type,
                path=section_path,
                element_id=element_id,
                extra_details={
                    "element_type": element_type,
                    "representative_check": True,
                    "error_type": type(section_error).__name__,
                },
            )

    uncovered_count = 0
    uncovered_sample: list[int] = []
    for element_id in element_lookup:
        if element_id in targeted_element_ids:
            continue
        uncovered_count += 1
        if len(uncovered_sample) < 20:
            uncovered_sample.append(element_id)
    if uncovered_count:
        definitions_valid = False
        diagnostics.append(
            PreflightDiagnostic(
                code="definition.section.unassigned_elements",
                severity=PreflightSeverity.ERROR,
                stage=PreflightStage.DEFINITIONS,
                message=(
                    f"{uncovered_count} elements have no valid explicit "
                    "section assignment."
                ),
                subject=tuple(uncovered_sample),
                path=("definitions", "sections", "coverage"),
                remediation="请将兼容截面分配到所有单元。",
                details={
                    "element_count": uncovered_count,
                    "element_ids": tuple(uncovered_sample),
                    "truncated": uncovered_count > len(uncovered_sample),
                },
            )
        )
    return definitions_valid


def _append_quick_section_diagnostic(
    diagnostics: list[PreflightDiagnostic],
    *,
    code: str,
    message: str,
    assignment_index: int,
    element_set_name: str,
    material_name: str,
    declared_type: str,
    path: tuple[str, ...],
    element_id: int | None = None,
    extra_details: dict[str, Any] | None = None,
) -> None:
    try:
        subject: Any = RegionRef("element_set", element_set_name)
    except ValueError:
        subject = element_id if element_id is not None else element_set_name
    details = {
        "assignment_index": assignment_index,
        "element_set": element_set_name,
        "element_id": element_id,
        "material": material_name,
        "section_type": declared_type,
    }
    details.update(extra_details or {})
    diagnostics.append(
        PreflightDiagnostic(
            code=code,
            severity=PreflightSeverity.ERROR,
            stage=PreflightStage.DEFINITIONS,
            message=message,
            subject=subject,
            path=path,
            remediation="请修复材料、截面及其单元集分配。",
            details=details,
        )
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
    model: Any,
    step: Any,
    diagnostics: list[PreflightDiagnostic],
    step_name: str,
) -> None:
    outputs = tuple(getattr(step, "outputs", ())) if step is not None else ()
    if not outputs:
        return
    output_support = _evaluate_output_requests(model, outputs)
    _append_projected_output_diagnostics(
        output_support.projections,
        diagnostics,
        step_name,
    )


def _append_projected_output_diagnostics(
    projections: tuple[Any, ...],
    diagnostics: list[PreflightDiagnostic],
    step_name: str,
) -> None:
    """Adapt canonical result diagnostics only after the lifecycle gate opens."""

    for projection in projections:
        request = projection.authoring_request
        for diagnostic in projection.diagnostics:
            details = dict(diagnostic.details)
            details.update(
                {
                    "request_name": request.name,
                    "request_kind": request.kind,
                    "request_target": request.target,
                    "request_variables": tuple(request.variables),
                }
            )
            diagnostics.append(
                PreflightDiagnostic(
                    code=diagnostic.code,
                    severity=PreflightSeverity.WARNING,
                    stage=PreflightStage.OUTPUT,
                    message=diagnostic.message,
                    subject=step_name,
                    path=(
                        "steps",
                        step_name,
                        *(
                            str(part)
                            for part in diagnostic.path
                        ),
                    ),
                    remediation=diagnostic.remediation,
                    details=details,
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
        body_load_count=len(getattr(step, "body_loads", ())),
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


__all__ = [
    "PreparedPreflight",
    "prepare_static_preflight",
    "run_static_preflight",
    "safe_prepare_static_preflight",
    "safe_static_preflight",
]
