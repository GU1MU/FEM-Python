"""Small authoring operations over the editable model."""

from __future__ import annotations

from typing import Any

from ..entities import ElementSet, MaterialDefinition, SectionAssignment


def add_material(model: Any, material: MaterialDefinition) -> MaterialDefinition:
    """Add or replace one named material definition."""

    if not isinstance(material, MaterialDefinition):
        raise TypeError("material must be MaterialDefinition")
    model.materials[material.name] = material
    return material


def assign_section(
    model: Any,
    material: str | MaterialDefinition,
    element_set: str | ElementSet,
    section_type: str = "solid",
    **properties: Any,
) -> SectionAssignment:
    """Append a material/section assignment to an editable model."""

    material_name = (
        material.name if isinstance(material, MaterialDefinition) else str(material)
    )
    element_set_name = (
        element_set.name if isinstance(element_set, ElementSet) else str(element_set)
    )
    section = SectionAssignment(
        element_set_name,
        material_name,
        section_type,
        dict(properties),
    )
    model.sections.append(section)
    return section


__all__ = ["add_material", "assign_section"]
