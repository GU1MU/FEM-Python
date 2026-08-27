"""Immutable material/section assignments for analysis execution.

Authoring models are normalized here once at the application boundary so
numerical operators consume an explicit material table.  The table carries a
constitutive-model identifier; material properties never select a model by
their mere presence.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from fem.materials.behavior_registry import material_behavior_diagnostics

from .assignments import resolve_sections, restored_element_properties


@dataclass(frozen=True, slots=True)
class CompiledMaterialAssignment:
    """Effective material and section properties for one mesh element."""

    element_id: int
    element_type: str
    material_name: str | None
    section_type: str | None
    properties: Mapping[str, Any] = field(default_factory=dict)
    source: str = "direct"
    constitutive_model: str = "linear_elastic"
    algorithm: str | None = None

    def __post_init__(self) -> None:
        element_id = int(self.element_id)
        if element_id < 1:
            raise ValueError("element_id must be positive")
        element_type = str(self.element_type).strip()
        if not element_type:
            raise ValueError("element_type must be nonblank")
        material_name = self.material_name
        if material_name is not None:
            material_name = str(material_name).strip() or None
        section_type = self.section_type
        if section_type is not None:
            section_type = str(section_type).strip() or None
        source = str(self.source).strip().casefold()
        if source not in {"direct", "section"}:
            raise ValueError("compiled assignment source must be 'direct' or 'section'")
        constitutive_model = str(self.constitutive_model).strip().casefold()
        if not constitutive_model:
            raise ValueError("compiled constitutive_model must not be blank")
        algorithm = self.algorithm
        if algorithm is not None:
            algorithm = str(algorithm).strip().casefold() or None
        if not isinstance(self.properties, Mapping):
            raise TypeError("compiled assignment properties must be a mapping")
        object.__setattr__(self, "element_id", element_id)
        object.__setattr__(self, "element_type", element_type)
        object.__setattr__(self, "material_name", material_name)
        object.__setattr__(self, "section_type", section_type)
        object.__setattr__(
            self,
            "properties",
            MappingProxyType(deepcopy(dict(self.properties))),
        )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "constitutive_model", constitutive_model)
        object.__setattr__(self, "algorithm", algorithm)

    def __deepcopy__(self, memo: dict[int, Any]) -> "CompiledMaterialAssignment":
        memo[id(self)] = self
        return self


@dataclass(frozen=True, slots=True)
class CompiledMaterialAssignments:
    """Deterministic element-id lookup used by a compiled analysis model."""

    assignments: tuple[CompiledMaterialAssignment, ...]

    def __post_init__(self) -> None:
        assignments = tuple(self.assignments)
        if any(type(item) is not CompiledMaterialAssignment for item in assignments):
            raise TypeError(
                "assignments must contain only CompiledMaterialAssignment values"
            )
        ids = tuple(item.element_id for item in assignments)
        if len(ids) != len(set(ids)):
            raise ValueError("compiled material assignments must use unique element ids")
        if ids != tuple(sorted(ids)):
            raise ValueError("compiled material assignments must be sorted by element id")
        object.__setattr__(self, "assignments", assignments)

    @property
    def by_element(self) -> dict[int, CompiledMaterialAssignment]:
        """Return a detached element-id lookup."""

        return {item.element_id: item for item in self.assignments}

    def for_element(self, element_id: int) -> CompiledMaterialAssignment:
        target = int(element_id)
        for item in self.assignments:
            if item.element_id == target:
                return item
        raise KeyError(f"element {target} has no compiled material assignment")


def compile_material_assignments(model: Any) -> CompiledMaterialAssignments:
    """Compile effective material/section data without mutating the model."""

    elements = tuple(getattr(getattr(model, "mesh", None), "elements", ()))
    if not elements:
        raise ValueError("cannot compile material assignments for an empty mesh")
    element_lookup = {int(element.id): element for element in elements}
    resolution = resolve_sections(model, element_lookup=element_lookup)
    resolution.require_valid()
    effective = {
        item.element_id: item
        for item in resolution.effective_assignments
    }
    materials = getattr(model, "materials", {})
    compiled: list[CompiledMaterialAssignment] = []

    for element_id in sorted(element_lookup):
        element = element_lookup[element_id]
        resolved = effective.get(element_id)
        if resolved is not None:
            properties = dict(resolved.effective_properties)
            material_name = resolved.material
            section_type = resolved.section_type
            source = "section"
        else:
            properties = restored_element_properties(
                model,
                element_id,
                element,
            )
            material_name = properties.get("material")
            section_type = properties.get("section_type")
            source = "direct"
            if material_name is not None and material_name in materials:
                material = materials[material_name]
                material_properties = dict(getattr(material, "properties", {}))
                material_properties.update(properties)
                properties = material_properties

        material_definition = (
            materials.get(material_name)
            if material_name is not None
            else None
        )
        if material_definition is not None:
            diagnostics = material_behavior_diagnostics(material_definition)
            if diagnostics:
                raise ValueError(
                    f"material {material_definition.name!r} is invalid: "
                    + "; ".join(diagnostics)
                )
        constitutive_model = str(
            getattr(
                material_definition,
                "constitutive_model",
                properties.get("constitutive_model", "linear_elastic"),
            )
        ).strip().casefold()
        algorithm = getattr(material_definition, "algorithm", None)
        if algorithm is None:
            algorithm = properties.get("algorithm")

        compiled.append(
            CompiledMaterialAssignment(
                element_id=element_id,
                element_type=str(element.type),
                material_name=material_name,
                section_type=section_type,
                properties=properties,
                source=source,
                constitutive_model=constitutive_model,
                algorithm=algorithm,
            )
        )
    return CompiledMaterialAssignments(tuple(compiled))


__all__ = [
    "CompiledMaterialAssignment",
    "CompiledMaterialAssignments",
    "compile_material_assignments",
]
