"""Pure adapters between editable definitions and the FEM kernel model.

Project inputs are owned by :class:`fem.application.ModelSession`.  Functions
in this module operate only on detached values: extraction returns tuples and
compilation returns a deep-copied model.  They never assign to a GUI document
or mutate a Session snapshot.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from fem.application import RegionAssignment, SectionDefinition
from fem.core.model import MaterialDefinition, SectionAssignment
from fem.materials import linear_elastic


DefinitionTuple = tuple[
    tuple[Any, ...],
    tuple[SectionDefinition, ...],
    tuple[RegionAssignment, ...],
    tuple[Any, ...],
]


def definitions_from_model(model: Any) -> DefinitionTuple:
    """Return detached editable definitions projected from one kernel model."""
    materials = deepcopy(tuple(getattr(model, "materials", {}).values()))
    sections: list[SectionDefinition] = []
    assignments: list[RegionAssignment] = []
    for index, section in enumerate(getattr(model, "sections", ()), start=1):
        name = f"Section-{index}"
        sections.append(
            SectionDefinition(
                name,
                str(section.material),
                str(section.section_type),
                deepcopy(dict(section.properties)),
            )
        )
        assignments.append(
            RegionAssignment(name, str(section.element_set))
        )
    steps = deepcopy(tuple(getattr(model, "steps", ())))
    return materials, tuple(sections), tuple(assignments), steps


def compile_model_definitions(
    model: Any,
    material_definitions: Mapping[str, Any] | Iterable[Any],
    section_definitions: Iterable[SectionDefinition],
    region_assignments: Iterable[RegionAssignment],
    analysis_definitions: Iterable[Any],
) -> Any:
    """Compile detached definition inputs into a detached model copy."""
    compiled = deepcopy(model)
    materials = tuple(
        material_definitions.values()
        if isinstance(material_definitions, Mapping)
        else material_definitions
    )
    sections = tuple(section_definitions)
    assignments = tuple(region_assignments)
    steps = tuple(analysis_definitions)

    material_map = {
        str(material.name): MaterialDefinition(
            str(material.name),
            deepcopy(dict(material.properties)),
        )
        for material in materials
    }
    section_map = {str(section.name): section for section in sections}
    element_sets = dict(
        getattr(compiled, "metadata", {}).get(
            "_abaqus_internal_element_sets",
            {},
        )
    )
    element_sets.update(getattr(compiled, "element_sets", {}))

    compiled_sections: list[SectionAssignment] = []
    for assignment in assignments:
        section = section_map.get(str(assignment.section_name))
        if section is None:
            raise ValueError(f"截面 {assignment.section_name} 不存在")
        if str(section.material) not in material_map:
            raise ValueError(
                f"截面 {section.name} 引用了不存在的材料 {section.material}"
            )
        region_name = str(assignment.region_name)
        if region_name not in element_sets:
            raise ValueError(f"区域 {region_name} 不是单元集")
        compiled_sections.append(
            SectionAssignment(
                region_name,
                str(section.material),
                str(section.section_type),
                deepcopy(dict(section.properties)),
            )
        )

    compiled.materials = material_map
    compiled.sections = compiled_sections
    compiled.steps = deepcopy(list(steps))
    return compiled


def compiled_model_snapshot(
    model: Any,
    material_definitions: Iterable[Any],
    section_definitions: Iterable[SectionDefinition],
    region_assignments: Iterable[RegionAssignment],
    analysis_definitions: Iterable[Any],
) -> Any:
    """Compile and validate an isolated model for checks/background work."""
    compiled = compile_model_definitions(
        model,
        material_definitions,
        section_definitions,
        region_assignments,
        analysis_definitions,
    )
    issues = section_assignment_issues(compiled)
    if issues:
        raise ValueError("；".join(issues))
    return compiled


def section_assignment_issues(model: Any | None) -> tuple[str, ...]:
    """Return actionable material/section issues for a detached model."""
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
            issues.append(f"截面引用了不存在的材料：{section.material}")
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


# Transitional names imported by older GUI modules.  Their signatures now
# accept detached values and their behavior is pure.
hydrate_document_definitions = definitions_from_model
apply_document_definitions = compile_model_definitions


__all__ = [
    "DefinitionTuple",
    "apply_document_definitions",
    "compile_model_definitions",
    "compiled_model_snapshot",
    "definitions_from_model",
    "hydrate_document_definitions",
    "section_assignment_issues",
]
