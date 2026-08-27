"""Private authoring-to-runtime compilation services."""

from .assignments import (
    EffectiveSectionAssignment,
    ResolvedSectionAssignment,
    SectionResolution,
    SectionResolutionIssue,
    apply_sections,
    resolve_sections,
    restored_element_properties,
)
from .materials import (
    CompiledMaterialAssignment,
    CompiledMaterialAssignments,
    compile_material_assignments,
)
from .capabilities import (
    DEFAULT_NONLINEAR_STATIC_CAPABILITIES,
    NonlinearStaticCapability,
    NonlinearStaticCapabilityRegistry,
    nonlinear_static_capabilities,
    nonlinear_static_capability_for,
)

__all__ = [
    "CompiledMaterialAssignment",
    "CompiledMaterialAssignments",
    "EffectiveSectionAssignment",
    "ResolvedSectionAssignment",
    "SectionResolution",
    "SectionResolutionIssue",
    "apply_sections",
    "compile_material_assignments",
    "DEFAULT_NONLINEAR_STATIC_CAPABILITIES",
    "NonlinearStaticCapability",
    "NonlinearStaticCapabilityRegistry",
    "nonlinear_static_capabilities",
    "nonlinear_static_capability_for",
    "resolve_sections",
    "restored_element_properties",
]
