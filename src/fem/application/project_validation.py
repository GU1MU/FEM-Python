"""Pure validation for detached native authoring project inputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from fem.core.model import AnalysisStep, MaterialDefinition
from fem.geometry.recipe_topology import topology_fingerprint_for_recipe
from fem.mesh.settings import MeshSettings
from fem.elements import get_element_capabilities
from fem.materials.sections import section_family

from .definitions import (
    MeshEntityRef,
    NamedRegion,
    RegionAssignment,
    SectionDefinition,
    normalize_model_definitions,
)
from .native_regions import (
    NativeRegionDescriptor,
    require_native_region_product,
    validate_native_authoring_context,
)
from .native_mesh_contract import describe_native_mesh_contract


class NativeProjectValidationError(ValueError):
    """Detached native authoring values are mutually inconsistent."""


def validate_native_project_inputs(
    recipe: Any,
    mesh_settings: MeshSettings | None,
    named_regions: Sequence[NamedRegion],
    materials: Sequence[MaterialDefinition],
    sections: Sequence[SectionDefinition],
    assignments: Sequence[RegionAssignment],
    steps: Sequence[AnalysisStep],
    *,
    enforce_formulation_compatibility: bool = True,
) -> tuple[NativeRegionDescriptor, ...]:
    """Validate references, definition links, and native region capabilities."""

    if mesh_settings is not None and type(mesh_settings) is not MeshSettings:
        raise NativeProjectValidationError(
            "mesh_settings must be MeshSettings or None"
        )
    try:
        mesh_contract = describe_native_mesh_contract(recipe, mesh_settings)
    except (TypeError, ValueError, NotImplementedError) as error:
        raise NativeProjectValidationError(str(error)) from error
    region_values = tuple(named_regions)
    logical_region_values = tuple(
        region
        for region in region_values
        if any(
            type(reference) is not MeshEntityRef
            for reference in region.references
        )
    )
    material_values = tuple(materials)
    section_values = tuple(sections)
    assignment_values = tuple(assignments)
    step_values = tuple(steps)
    try:
        normalized = normalize_model_definitions(
            material_values,
            section_values,
            assignment_values,
            step_values,
        )
        material_values = normalized.materials
        section_values = normalized.sections
        assignment_values = normalized.assignments
        step_values = normalized.steps
        descriptors = validate_native_authoring_context(
            recipe,
            region_values,
            local_controls=(
                ()
                if mesh_settings is None
                else tuple(mesh_settings.local_controls)
            ),
            mesh_settings=mesh_settings,
            mesh_contract=mesh_contract,
        )
    except NativeProjectValidationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise NativeProjectValidationError(str(error)) from error

    fingerprint = topology_fingerprint_for_recipe(recipe)
    if not fingerprint.exact and (
        logical_region_values
        or (
            mesh_settings is not None
            and bool(mesh_settings.local_controls)
        )
        or analysis_steps_have_native_region_targets(step_values)
    ):
        raise NativeProjectValidationError(
            "non-exact topology cannot carry NamedRegion with logical references, "
            "LocalMeshControl, or a named AnalysisStep target"
        )

    sections_by_name = {
        section.name: section for section in section_values
    }
    if not mesh_contract.complete and (assignment_values or step_values):
        raise NativeProjectValidationError(
            "incomplete native wire projects cannot persist region assignments "
            "or analysis definitions before line_element_type is selected"
        )

    expected_section_family = None
    if mesh_contract.complete and enforce_formulation_compatibility:
        capabilities = get_element_capabilities(
            mesh_contract.canonical_element_type
        )
        expected_section_family = capabilities.section_families[0]
    for index, assignment in enumerate(assignment_values):
        section = sections_by_name.get(assignment.section_name)
        if section is None:
            raise NativeProjectValidationError(
                f"assignments[{index}].section_name references unknown section "
                f"{assignment.section_name!r}"
            )
        if not enforce_formulation_compatibility and section.section_type != "solid":
            raise NativeProjectValidationError(
                f"assignments[{index}] section type {section.section_type!r} "
                "is incompatible with current native continuum recipes"
            )
        if enforce_formulation_compatibility:
            try:
                assigned_family = section_family(section.section_type)
            except (TypeError, ValueError) as error:
                raise NativeProjectValidationError(
                    f"assignments[{index}] section {section.name!r} has an "
                    f"invalid section type {section.section_type!r}: {error}"
                ) from error
            if assigned_family != expected_section_family:
                raise NativeProjectValidationError(
                    f"assignments[{index}] section type {section.section_type!r} "
                    f"belongs to family {assigned_family!r}, while native element "
                    f"{mesh_contract.canonical_element_type!r} requires "
                    f"section family {expected_section_family!r}"
                )
        target_product = (
            "beam_element_set"
            if enforce_formulation_compatibility
            and expected_section_family == "beam"
            else "element_set"
        )
        _require_product(
            descriptors,
            assignment.region_name,
            target_product,
            f"assignments[{index}].region_name",
        )
        orientation = getattr(assignment, "beam_orientation", None)
        if (
            orientation is not None
            and enforce_formulation_compatibility
            and mesh_contract.dimension == 1
        ):
            if expected_section_family != "beam":
                raise NativeProjectValidationError(
                    f"assignments[{index}].beam_orientation is only valid "
                    "for Beam2 assignments"
                )
            _require_product(
                descriptors,
                assignment.region_name,
                "beam_element_set",
                f"assignments[{index}].region_name",
            )

    for step_index, step in enumerate(step_values):
        for index, boundary in enumerate(step.boundaries):
            target = _stable_target(
                boundary.target,
                f"steps[{step_index}].boundaries[{index}].target",
            )
            _require_product(
                descriptors,
                target,
                getattr(boundary, "target_kind", "node_set"),
                f"steps[{step_index}].boundaries[{index}].target",
            )
        for index, load in enumerate(step.cloads):
            target = _stable_target(
                load.target,
                f"steps[{step_index}].cloads[{index}].target",
            )
            _require_product(
                descriptors,
                target,
                "node_set",
                f"steps[{step_index}].cloads[{index}].target",
            )
        for index, load in enumerate(step.edge_loads):
            _require_product(
                descriptors,
                load.edge,
                "edge",
                f"steps[{step_index}].edge_loads[{index}].edge",
            )
        for index, load in enumerate(step.surface_loads):
            _require_product(
                descriptors,
                load.surface,
                "surface",
                f"steps[{step_index}].surface_loads[{index}].surface",
            )
        for index, load in enumerate(step.line_loads):
            target = _stable_target(
                load.target,
                f"steps[{step_index}].line_loads[{index}].target",
            )
            _require_product(
                descriptors,
                target,
                "beam_element_set",
                f"steps[{step_index}].line_loads[{index}].target",
            )
        for index, load in enumerate(step.body_loads):
            target = _stable_target(
                load.target,
                f"steps[{step_index}].body_loads[{index}].target",
            )
            _require_product(
                descriptors,
                target,
                "element_set",
                f"steps[{step_index}].body_loads[{index}].target",
            )
        for index, load in enumerate(step.gravity_loads):
            if load.target is None:
                continue
            target = _stable_target(
                load.target,
                f"steps[{step_index}].gravity_loads[{index}].target",
            )
            _require_product(
                descriptors,
                target,
                "element_set",
                f"steps[{step_index}].gravity_loads[{index}].target",
            )
    return descriptors


def _stable_target(value: Any, path: str) -> str:
    if type(value) is not str or not value.strip():
        raise NativeProjectValidationError(
            f"{path} must be a non-empty stable region name; "
            "mesh integer targets cannot be persisted"
        )
    return value


def _require_product(
    descriptors: tuple[NativeRegionDescriptor, ...],
    target: Any,
    product: str,
    path: str,
) -> None:
    if type(target) is not str or not target.strip():
        raise NativeProjectValidationError(
            f"{path} must be a non-empty stable region name"
        )
    try:
        require_native_region_product(descriptors, target, product)
    except (KeyError, TypeError, ValueError) as error:
        raise NativeProjectValidationError(f"{path}: {error}") from error


def analysis_step_has_native_region_target(step: AnalysisStep) -> bool:
    """Return whether one native step depends on a named geometry region."""

    return bool(
        step.boundaries
        or step.cloads
        or step.edge_loads
        or step.surface_loads
        or step.line_loads
        or step.body_loads
        or any(load.target is not None for load in step.gravity_loads)
    )


def analysis_steps_have_native_region_targets(
    steps: Iterable[AnalysisStep],
) -> bool:
    """Return whether any native step depends on a named geometry region."""

    return any(analysis_step_has_native_region_target(step) for step in steps)


__all__ = [
    "NativeProjectValidationError",
    "analysis_step_has_native_region_target",
    "analysis_steps_have_native_region_targets",
    "validate_native_project_inputs",
]
