from __future__ import annotations

from typing import Any

from ..core.model import ElementSet, MaterialDefinition, SectionAssignment


def add(model: Any, material: MaterialDefinition) -> MaterialDefinition:
    """Add a material definition to a model."""
    model.materials[material.name] = material
    return material


def assign(
    model: Any,
    material: str | MaterialDefinition,
    element_set: str | ElementSet,
    section_type: str = "solid",
    **properties: Any,
) -> SectionAssignment:
    """Assign a material to an element set."""
    material_name = material.name if isinstance(material, MaterialDefinition) else str(material)
    element_set_name = element_set.name if isinstance(element_set, ElementSet) else str(element_set)
    section = SectionAssignment(
        element_set_name,
        material_name,
        section_type,
        dict(properties),
    )
    model.sections.append(section)
    return section


def apply_sections(model: Any) -> None:
    """Copy assigned material and section data onto element props."""
    element_lookup = {elem.id: elem for elem in model.mesh.elements}
    section_keys = model.metadata.setdefault("_section_property_keys_by_element", {})

    for section in model.sections:
        if section.material not in model.materials:
            raise KeyError(f"material {section.material} is not defined")
        element_set = _section_element_set(model, section.element_set)

        props = dict(model.materials[section.material].properties)
        props.update(section.properties)
        props["material"] = section.material

        for element_id in element_set.element_ids:
            if element_id not in element_lookup:
                raise KeyError(f"element {element_id} is not defined")
            elem = element_lookup[element_id]
            for key in section_keys.get(element_id, ()):
                elem.props.pop(key, None)
            elem.props.update(props)
            section_keys[element_id] = tuple(props)


def _section_element_set(model: Any, name: str) -> ElementSet:
    """Return a public or importer-internal element set used by a section."""
    if name in model.element_sets:
        return model.element_sets[name]
    internal_sets = model.metadata.get("_abaqus_internal_element_sets", {})
    if name in internal_sets:
        return internal_sets[name]
    raise KeyError(f"element set {name} is not defined")
