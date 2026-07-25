"""Apply editable GUI model definitions to the existing FEMModel kernel."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from fem.core.model import MaterialDefinition, SectionAssignment
from fem.materials import linear_elastic

from .document import FEMDocument, RegionAssignment, SectionDefinition


def hydrate_document_definitions(document: FEMDocument) -> None:
    """Expose parsed INP materials/sections through the same GUI dialogs."""
    model = document.model
    if model is None:
        return
    document.material_definitions = list(model.materials.values())
    document.section_definitions.clear()
    document.region_assignments.clear()
    for index, section in enumerate(model.sections, start=1):
        name = f"Section-{index}"
        document.section_definitions.append(
            SectionDefinition(
                name,
                section.material,
                section.section_type,
                dict(section.properties),
            )
        )
        document.region_assignments.append(
            RegionAssignment(
                name, section.element_set
            )
        )
    document.analysis_definitions = deepcopy(model.steps)


def apply_document_definitions(document: FEMDocument) -> None:
    """Compile GUI definitions into the current FEMModel without a new backend.

    Native models are rebuilt repeatedly after meshing.  The GUI definitions are
    therefore the durable source and this function is intentionally idempotent.
    """
    model = document.model
    if model is None:
        return
    materials = {
        material.name: MaterialDefinition(material.name, dict(material.properties))
        for material in document.material_definitions
    }
    model.materials = materials

    definitions = {section.name: section for section in document.section_definitions}
    element_sets = dict(
        model.metadata.get("_abaqus_internal_element_sets", {})
    )
    element_sets.update(model.element_sets)
    assignments: list[SectionAssignment] = []
    for assignment in document.region_assignments:
        section = definitions.get(assignment.section_name)
        if section is None:
            raise ValueError(f"截面 {assignment.section_name} 不存在")
        if section.material not in model.materials:
            raise ValueError(f"截面 {section.name} 引用了不存在的材料 {section.material}")
        if assignment.region_name not in element_sets:
            raise ValueError(f"区域 {assignment.region_name} 不是单元集")
        assignments.append(
            SectionAssignment(
                assignment.region_name,
                section.material,
                section.section_type,
                dict(section.properties),
            )
        )
    model.sections = assignments
    model.steps = deepcopy(document.analysis_definitions)


def compiled_model_snapshot(
    model: Any,
    material_definitions: tuple[Any, ...],
    section_definitions: tuple[SectionDefinition, ...],
    region_assignments: tuple[RegionAssignment, ...],
    analysis_definitions: tuple[Any, ...],
) -> Any:
    """Compile GUI definitions into an isolated model for background checks."""
    snapshot_document = FEMDocument(
        model=deepcopy(model),
        material_definitions=deepcopy(list(material_definitions)),
        section_definitions=deepcopy(list(section_definitions)),
        region_assignments=deepcopy(list(region_assignments)),
        analysis_definitions=deepcopy(list(analysis_definitions)),
    )
    apply_document_definitions(snapshot_document)
    issues = section_assignment_issues(snapshot_document)
    if issues:
        raise ValueError("；".join(issues))
    return snapshot_document.model


def section_assignment_issues(document: FEMDocument) -> tuple[str, ...]:
    """Return actionable input issues for a submit/check guard."""
    model = document.model
    if model is None:
        return ("尚未生成网格或打开 INP 模型",)
    if not model.materials:
        return ("尚未定义材料",)
    if not model.sections:
        return ("尚未将截面分配到单元区域",)

    issues: list[str] = []
    element_sets = dict(
        model.metadata.get("_abaqus_internal_element_sets", {})
    )
    element_sets.update(model.element_sets)
    covered_element_ids: set[int] = set()
    validated_materials: set[str] = set()

    for section in model.sections:
        material = model.materials.get(section.material)
        if material is None:
            issues.append(
                f"截面引用了不存在的材料：{section.material}"
            )
        elif material.name not in validated_materials:
            properties = material.properties
            if "E" not in properties or "nu" not in properties:
                issues.append(
                    f"材料 {material.name} 缺少线弹性参数（弹性模量、泊松比）"
                )
            else:
                try:
                    linear_elastic.material(
                        material.name,
                        properties["E"],
                        properties["nu"],
                    )
                except (TypeError, ValueError) as error:
                    issues.append(
                        f"材料 {material.name} 的线弹性参数无效：{error}"
                    )
            validated_materials.add(material.name)

        element_set = element_sets.get(section.element_set)
        if element_set is None:
            issues.append(
                f"截面引用了不存在的单元集：{section.element_set}"
            )
        else:
            covered_element_ids.update(element_set.element_ids)

    model_element_ids = {element.id for element in model.mesh.elements}
    missing = sorted(model_element_ids - covered_element_ids)
    if missing:
        preview = "、".join(str(element_id) for element_id in missing[:8])
        suffix = "…" if len(missing) > 8 else ""
        issues.append(
            f"有 {len(missing)} 个单元尚未分配截面"
            f"（单元 {preview}{suffix}）"
        )
    return tuple(issues)
