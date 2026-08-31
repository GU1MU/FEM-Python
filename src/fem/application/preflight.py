"""Structured, selected-Step static preflight."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from typing import Any, Iterable

import numpy as np
from scipy.sparse.linalg import spsolve

from fem import analysis as analysis_domain
from fem.analysis import (
    AnalysisRequest,
    DEFAULT_ANALYSIS_EXECUTOR,
    ExecutionStrategy,
    PreparedAnalysis,
    compile_analysis,
    compile_boundary,
    execution_plan_cache_scope,
    resolve_analysis_request,
    resolve_execution_plan,
)
from fem.analysis.compilation import DEFAULT_NONLINEAR_STATIC_CAPABILITIES
from fem.model import (
    DynamicProcedureKind,
    DynamicStepControls,
    StaticFormulation,
    resolve_analysis_step,
)
from fem.analysis.validation import (
    validate_analysis_step,
    validate_model_structure,
)
from fem.elements import get_element_capabilities

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
    """One typed preflight report and an optional reusable static base system.

    Dynamic procedures are fully compiled during numerical preflight, but
    their ``CompiledAnalysis`` is not a Session ``PreparedAnalysis`` cache
    entry.  It is therefore intentionally not exposed through this field.
    """

    report: PreflightReport
    prepared_system: PreparedAnalysis | None = None


def run_static_preflight(
    model: Any,
    step: Any = None,
    *,
    token: TaskToken | None = None,
    check_numerical_stability: bool = True,
    copy_model: bool = True,
    quick_check: bool = False,
) -> PreflightReport:
    """Check one detached linear or supported nonlinear static Step."""

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


@execution_plan_cache_scope()
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
        selected_step = resolve_analysis_step(owned_model, step)
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

    analysis_request: AnalysisRequest | None = None
    if selected_step is not None:
        try:
            analysis_request = resolve_analysis_request(selected_step)
        except Exception as error:
            diagnostics.append(
                _diagnostic(
                    "step.controls.invalid",
                    PreflightStage.STEP,
                    error,
                    subject=report_step_name,
                    path=("steps", report_step_name, "controls"),
                    remediation="请修复分析步的过程、NLGEOM 或增量控制参数。",
                )
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
        request=analysis_request,
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

    execution_plan = None
    if analysis_request is not None and definitions_valid:
        try:
            execution_plan = resolve_execution_plan(
                owned_model,
                analysis_request,
            )
        except Exception as error:
            diagnostics.append(
                _diagnostic(
                    "analysis.execution_plan.invalid",
                    PreflightStage.CAPABILITY,
                    error,
                    subject=report_step_name,
                    path=("steps", report_step_name, "execution_plan"),
                    remediation=(
                        "请检查几何模式、材料模型标识、截面和单元能力是否完整。"
                    ),
                )
            )
    nonlinear_step = (
        execution_plan is not None
        and execution_plan.strategy is ExecutionStrategy.INCREMENTAL_NEWTON
    )

    if structure_valid and step_valid and selected_step is not None:
        try:
            boundary = compile_boundary(owned_model, selected_step)
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
            if not boundary.constraints.prescribed_values:
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
        and boundary.constraints.prescribed_values
        and not _has_blocking_diagnostic(diagnostics)
    ):
        if nonlinear_step:
            nonlinear_valid, nonlinear_checked = _validate_nonlinear_entry(
                owned_model,
                selected_step,
                diagnostics,
                check_numerical_stability=check_numerical_stability,
            )
            numerical_stability_checked = nonlinear_checked
        elif check_numerical_stability:
            numerical_stability_checked = True
            try:
                if analysis_request is None:
                    raise ValueError(
                        "a valid analysis request is required for stiffness validation"
                    )
                candidate = DEFAULT_ANALYSIS_EXECUTOR.prepare(
                    owned_model,
                    analysis_request,
                )
                DEFAULT_ANALYSIS_EXECUTOR.validate_prepared(
                    owned_model,
                    selected_step,
                    analysis_request,
                    candidate,
                )
            except Exception as error:
                dynamic_request = (
                    analysis_request is not None
                    and analysis_request.step.procedure == "dynamic"
                )
                diagnostics.append(
                    _diagnostic(
                        (
                            "dynamic.problem.invalid"
                            if dynamic_request
                            else "static.stiffness.singular"
                        ),
                        (
                            PreflightStage.CAPABILITY
                            if dynamic_request
                            else PreflightStage.STIFFNESS
                        ),
                        error,
                        subject=report_step_name,
                        path=("steps", report_step_name, "stiffness"),
                        remediation=(
                            "请为动力学单元提供密度 rho，并检查质量矩阵、"
                            "材料、截面和边界条件。"
                            if dynamic_request
                            else "请检查约束、材料、截面、单元连接和零刚度自由度。"
                        ),
                    )
                )
            else:
                if analysis_request is not None and analysis_request.step.procedure == "dynamic":
                    _validate_dynamic_prepared(
                        candidate,
                        analysis_request,
                        diagnostics,
                        report_step_name,
                    )
                if (
                    retain_prepared_system
                    and isinstance(candidate, PreparedAnalysis)
                ):
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
        request=analysis_request,
        execution_plan=execution_plan,
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
                analysis_domain.resolve_section_properties(
                    element_type,
                    material_properties,
                    declared_type,
                    section_properties,
                    baseline_properties=analysis_domain.restored_element_properties(
                        model,
                        element_id,
                        element,
                    ),
                    constitutive_model=getattr(
                        material,
                        "constitutive_model",
                        None,
                    ),
                )
            except analysis_domain.SectionCompatibilityError as caught:
                code = "definition.section.incompatible"
                section_error = caught
            except analysis_domain.MaterialPropertyError as caught:
                code = "definition.material.invalid"
                section_error = caught
            except analysis_domain.SectionPropertyError as caught:
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
        resolution = analysis_domain.resolve_sections(model)
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
    *,
    request: AnalysisRequest | None = None,
) -> bool:
    if step is None:
        return False
    if request is None:
        procedure = str(getattr(step, "procedure", "")).strip().casefold()
        nlgeom = None
    else:
        procedure = request.step.procedure
        nlgeom = request.formulation is StaticFormulation.NONLINEAR
    if procedure in {"static", "dynamic"}:
        return True
    message = (
        "The current solver supports static, implicit dynamic, and explicit "
        "dynamic Steps; the selected procedure is not available yet."
    )
    diagnostics.append(
        PreflightDiagnostic(
            code="static.procedure.unsupported",
            severity=PreflightSeverity.ERROR,
            stage=PreflightStage.STEP,
            message=message,
            subject=step_name,
            path=("steps", step_name, "procedure"),
            remediation="请选择已实现的静力、隐式动力学或显式动力学过程。",
            details={"procedure": procedure, "nlgeom": nlgeom},
        )
    )
    return False


def _validate_nonlinear_entry(
    model: Any,
    step: Any,
    diagnostics: list[PreflightDiagnostic],
    *,
    check_numerical_stability: bool,
) -> tuple[bool, bool]:
    """Validate a registered nonlinear static entry without running a solve."""

    mesh = getattr(model, "mesh", None)
    valid = True
    try:
        dofs_per_node = int(mesh.dofs_per_node)
    except (AttributeError, TypeError, ValueError) as error:
        _append_nonlinear_diagnostic(
            diagnostics,
            "nonlinear.entry.invalid",
            PreflightStage.CAPABILITY,
            error,
            subject="mesh",
            remediation=(
                "当前材料或几何非线性路径需要已注册的平面或实体连续体单元，"
                "并且节点自由度必须与单元 capability 匹配。"
            ),
        )
        return False, False

    for element in getattr(mesh, "elements", ()):
        try:
            descriptor = get_element_capabilities(element.type)
            nonlinear_capability = DEFAULT_NONLINEAR_STATIC_CAPABILITIES.resolve(
                element.type
            )
            definition = nonlinear_capability.definition_factory()
            node_count = len(element.node_ids)
        except Exception as error:
            valid = False
            _append_nonlinear_diagnostic(
                diagnostics,
                "nonlinear.entry.unsupported_element",
                PreflightStage.CAPABILITY,
                error,
                subject=getattr(element, "id", "element"),
                remediation=(
                    "请使用已注册的增量静力算子，并使节点数和自由度与单元能力匹配。"
                ),
            )
            continue

        if (
            nonlinear_capability.canonical_type != descriptor.canonical_type
            or node_count != descriptor.node_count
            or node_count != definition.node_count
            or dofs_per_node != descriptor.dofs_per_node
        ):
            valid = False
            _append_nonlinear_diagnostic(
                diagnostics,
                "nonlinear.entry.unsupported_element",
                PreflightStage.CAPABILITY,
                ValueError(
                    f"element {element.id} has incompatible nonlinear capability "
                    f"{nonlinear_capability.canonical_type}, "
                    f"{node_count} nodes and mesh DOFs/node={dofs_per_node}"
                ),
                subject=element.id,
                remediation=(
                    "请使网格自由度、元素节点数、元素能力描述和非线性定义保持一致。"
                ),
            )

    if not valid:
        return False, False

    try:
        problem = compile_analysis(
            model,
            resolve_analysis_request(step),
        ).problem
        if not check_numerical_stability:
            diagnostics.append(
                PreflightDiagnostic(
                    code="nonlinear.stability.skipped_large_model",
                    severity=PreflightSeverity.WARNING,
                    stage=PreflightStage.STIFFNESS,
                    message=(
                        "大型非线性模型快速检查已跳过初始切线数值稳定性检查。"
                    ),
                    subject=str(getattr(step, "name", "")),
                    path=(
                        "steps",
                        str(getattr(step, "name", "")),
                        "nonlinear_stiffness",
                    ),
                    remediation="提交分析后，Newton 求解器仍会执行完整切线装配。",
                )
            )
            return True, False

        evaluation = problem.evaluate()
        tangent = evaluation.tangent
        rhs = np.ones(int(mesh.num_dofs), dtype=float)
        solution = spsolve(tangent, rhs)
        if not np.all(np.isfinite(solution)):
            raise ValueError(
                "the initial nonlinear tangent solve returned non-finite values"
            )
    except KeyError as error:
        valid = False
        _append_nonlinear_diagnostic(
            diagnostics,
            "nonlinear.material.property_missing",
            PreflightStage.DEFINITIONS,
            error,
            subject="materials",
            remediation=(
                "为所有参与非线性静力的材料提供 E、nu；"
                "若启用塑性，再提供 yield_stress，可选 hardening_modulus。"
            ),
        )
    except Exception as error:
        valid = False
        _append_nonlinear_diagnostic(
            diagnostics,
            "nonlinear.entry.invalid",
            PreflightStage.STIFFNESS,
            error,
            subject=str(getattr(step, "name", "")),
            remediation=(
                "请检查非线性材料参数、单元几何、边界约束和初始切线。"
            ),
        )
    return valid, check_numerical_stability and valid


def _append_nonlinear_diagnostic(
    diagnostics: list[PreflightDiagnostic],
    code: str,
    stage: PreflightStage,
    error: Any,
    *,
    subject: Any,
    remediation: str,
) -> None:
    diagnostics.append(
        _diagnostic(
            code,
            stage,
            error,
            subject=subject,
            path=("nonlinear",),
            remediation=remediation,
        )
    )


def _validate_dynamic_prepared(
    prepared: Any,
    request: AnalysisRequest,
    diagnostics: list[PreflightDiagnostic],
    step_name: str,
) -> None:
    """Check explicit stability without turning a compiled dynamic object into a cache."""

    controls = request.controls
    if not isinstance(controls, DynamicStepControls):
        return
    if controls.procedure_kind is not DynamicProcedureKind.EXPLICIT:
        return
    problem = getattr(prepared, "problem", None)
    stable_increment = getattr(problem, "stable_time_increment", None)
    if not callable(stable_increment):
        return
    try:
        estimate = float(stable_increment())
    except Exception as error:
        diagnostics.append(
            _diagnostic(
                "dynamic.stability.invalid",
                PreflightStage.STIFFNESS,
                error,
                subject=step_name,
                path=("steps", step_name, "stable_time_increment"),
                remediation="请检查质量密度、单元刚度和边界条件。",
            )
        )
        return
    if not np.isfinite(estimate):
        return
    effective = 0.9 * estimate
    if effective < controls.minimum_time_increment:
        diagnostics.append(
            _diagnostic(
                "dynamic.stability.below_minimum_increment",
                PreflightStage.STIFFNESS,
                ValueError(
                    "the explicit stable time increment is below the configured minimum"
                ),
                subject=step_name,
                path=("steps", step_name, "stable_time_increment"),
                remediation=(
                    "减小最小时间增量、改善网格质量或调整材料/密度参数。"
                ),
            )
        )
        return
    required = max(1, ceil(request.controls.time_period / effective - 1.0e-12))
    if required > controls.maximum_increments:
        diagnostics.append(
            _diagnostic(
                "dynamic.stability.insufficient_increments",
                PreflightStage.STEP,
                ValueError(
                    "maximum_increments is too small for the explicit stable time step"
                ),
                subject=step_name,
                path=("steps", step_name, "maximum_increments"),
                remediation="增大最大增量数，或设置更合理的时间步和模型尺度。",
            )
        )
    elif controls.initial_time_increment > effective:
        diagnostics.append(
            PreflightDiagnostic(
                code="dynamic.stability.initial_increment_capped",
                severity=PreflightSeverity.WARNING,
                stage=PreflightStage.STIFFNESS,
                message=(
                    f"显式稳定时间增量估计为 {effective:.3e}，"
                    "求解时会自动限制用户设置的初始增量。"
                ),
                subject=step_name,
                path=("steps", step_name, "initial_time_increment"),
                remediation="如需减少增量数，可增大密度或改善单元尺寸；也可保持当前设置。",
            )
        )


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
    *,
    request: AnalysisRequest | None,
    execution_plan: Any | None = None,
) -> PreflightFacts:
    mesh = getattr(model, "mesh", None)
    return PreflightFacts(
        model_name=getattr(model, "name", None),
        step_name=step_name,
        procedure=(
            str(getattr(step, "procedure", "")) if step is not None else ""
        ),
        formulation=None if request is None else request.formulation,
        node_count=len(getattr(mesh, "nodes", ())),
        element_count=len(getattr(mesh, "elements", ())),
        dof_count=int(getattr(mesh, "num_dofs", 0)),
        material_count=len(getattr(model, "materials", {})),
        section_count=len(getattr(model, "sections", ())),
        displacement_count=(
            len(boundary.constraints.prescribed_values)
            if boundary is not None
            else 0
        ),
        nodal_load_count=(
            len(boundary.loads.nodal_forces) if boundary is not None else 0
        ),
        edge_load_count=len(getattr(step, "edge_loads", ())),
        surface_load_count=len(getattr(step, "surface_loads", ())),
        line_load_count=len(getattr(step, "line_loads", ())),
        body_load_count=len(getattr(step, "body_loads", ())),
        gravity_load_count=len(getattr(step, "gravity_loads", ())),
        geometry_mode=(
            None
            if request is None or request.geometry_mode is None
            else request.geometry_mode.value
        ),
        execution_strategy=(
            None
            if execution_plan is None
            else execution_plan.strategy.value
        ),
        material_models=(
            ()
            if execution_plan is None
            else tuple(execution_plan.material_models)
        ),
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
