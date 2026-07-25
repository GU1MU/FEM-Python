"""Headless editable definitions and their single detached compiler."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from fem.core.model import MaterialDefinition, SectionAssignment

from .capabilities import RegionRef
from .diagnostics import (
    PreflightDiagnostic,
    PreflightSeverity,
    PreflightStage,
)


@dataclass(frozen=True, slots=True)
class NativePart:
    """Small serialisable representation of one editable native part."""

    name: str = "Part-1"
    body_name: str = "Body-1"


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    """One item in the shallow native feature history."""

    name: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NamedRegion:
    """A logical native region mapped to mesh sets after regeneration."""

    name: str
    entity_kind: Literal["point", "edge", "face", "body"]
    entity_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class SectionDefinition:
    """Editable section definition with a material linked by name."""

    name: str
    material: str
    section_type: str = "solid"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RegionAssignment:
    """Assign one named section to an existing element region."""

    section_name: str
    region_name: str


@dataclass(frozen=True, slots=True)
class ModelDefinitions:
    """Detached, application-owned editable model definitions."""

    materials: tuple[MaterialDefinition, ...] = ()
    sections: tuple[SectionDefinition, ...] = ()
    assignments: tuple[RegionAssignment, ...] = ()
    steps: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "materials",
            deepcopy(tuple(self.materials)),
        )
        object.__setattr__(
            self,
            "sections",
            deepcopy(tuple(self.sections)),
        )
        object.__setattr__(
            self,
            "assignments",
            deepcopy(tuple(self.assignments)),
        )
        object.__setattr__(self, "steps", deepcopy(tuple(self.steps)))


@dataclass(frozen=True, slots=True)
class DefinitionCompileResult:
    """Result of compiling definitions without mutating the base model."""

    definitions: ModelDefinitions | None
    model: Any | None
    diagnostics: tuple[PreflightDiagnostic, ...] = ()

    @property
    def passed(self) -> bool:
        return self.model is not None and not any(
            diagnostic.blocking for diagnostic in self.diagnostics
        )

    def require_model(self) -> Any:
        """Return a detached compiled model or raise a typed rejection."""

        if not self.passed:
            raise DefinitionRejected(self.diagnostics)
        return deepcopy(self.model)


class DefinitionRejected(ValueError):
    """A definitions command was rejected before Session state changed."""

    def __init__(
        self,
        diagnostics: Iterable[PreflightDiagnostic],
    ) -> None:
        self.diagnostics = tuple(deepcopy(tuple(diagnostics)))
        message = "; ".join(
            diagnostic.message for diagnostic in self.diagnostics
        )
        super().__init__(message or "model definitions were rejected")

    @classmethod
    def from_error(cls, error: Exception) -> DefinitionRejected:
        """Create a stable rejection for one input-validation error."""

        return cls((_definition_diagnostic(error),))


def normalize_model_definitions(
    materials: (
        ModelDefinitions
        | Mapping[str, Any]
        | Iterable[Any]
    ) = (),
    sections: Iterable[SectionDefinition] | None = None,
    assignments: Iterable[RegionAssignment] | None = None,
    steps: Iterable[Any] | None = None,
) -> ModelDefinitions:
    """Own, normalize, and validate editable definition inputs."""

    try:
        return _normalize_model_definitions(
            materials,
            sections,
            assignments,
            steps,
        )
    except DefinitionRejected:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise DefinitionRejected.from_error(error) from error


def _normalize_model_definitions(
    materials: (
        ModelDefinitions
        | Mapping[str, Any]
        | Iterable[Any]
    ),
    sections: Iterable[SectionDefinition] | None,
    assignments: Iterable[RegionAssignment] | None,
    steps: Iterable[Any] | None,
) -> ModelDefinitions:
    if isinstance(materials, ModelDefinitions):
        if any(value is not None for value in (sections, assignments, steps)):
            raise TypeError(
                "separate definition collections cannot accompany "
                "ModelDefinitions"
            )
        source = deepcopy(materials)
        material_values = source.materials
        section_values = source.sections
        assignment_values = source.assignments
        step_values = source.steps
    else:
        material_values = _mapping_values(materials)
        section_values = tuple(() if sections is None else sections)
        assignment_values = tuple(
            () if assignments is None else assignments
        )
        step_values = tuple(() if steps is None else steps)

    owned_materials = tuple(
        MaterialDefinition(
            _required_name(material, "material"),
            deepcopy(dict(getattr(material, "properties", {}))),
        )
        for material in deepcopy(tuple(material_values))
    )
    owned_sections = tuple(
        SectionDefinition(
            name=_required_name(section, "section"),
            material=str(section.material).strip(),
            section_type=str(section.section_type).strip().casefold(),
            properties=deepcopy(dict(section.properties)),
        )
        for section in deepcopy(tuple(section_values))
    )
    owned_assignments = tuple(
        RegionAssignment(
            section_name=str(assignment.section_name).strip(),
            region_name=str(assignment.region_name).strip(),
        )
        for assignment in deepcopy(tuple(assignment_values))
    )
    owned_steps = deepcopy(tuple(step_values))
    for step in owned_steps:
        name = _required_name(step, "analysis step")
        step.name = name

    _validate_unique_names(owned_materials, "material")
    _validate_unique_names(owned_sections, "section")
    _validate_unique_names(owned_steps, "analysis step")
    _validate_definition_links(
        owned_materials,
        owned_sections,
        owned_assignments,
    )
    return ModelDefinitions(
        owned_materials,
        owned_sections,
        owned_assignments,
        owned_steps,
    )


def definitions_from_model(model: Any) -> ModelDefinitions:
    """Project a kernel model into one detached editable snapshot."""

    materials = deepcopy(tuple(getattr(model, "materials", {}).values()))
    sections: list[SectionDefinition] = []
    assignments: list[RegionAssignment] = []
    for index, section in enumerate(
        getattr(model, "sections", ()),
        start=1,
    ):
        name = f"Section-{index}"
        sections.append(
            SectionDefinition(
                name=name,
                material=str(section.material),
                section_type=str(section.section_type),
                properties=deepcopy(dict(section.properties)),
            )
        )
        assignments.append(
            RegionAssignment(
                section_name=name,
                region_name=str(section.element_set),
            )
        )
    return normalize_model_definitions(
        materials,
        sections,
        assignments,
        deepcopy(tuple(getattr(model, "steps", ()))),
    )


def compile_model_definitions(
    base_model: Any,
    definitions: (
        ModelDefinitions
        | Mapping[str, Any]
        | Iterable[Any]
    ),
    sections: Iterable[SectionDefinition] | None = None,
    assignments: Iterable[RegionAssignment] | None = None,
    steps: Iterable[Any] | None = None,
) -> DefinitionCompileResult:
    """Compile definitions into a detached model, reporting user errors."""

    try:
        normalized = normalize_model_definitions(
            definitions,
            sections,
            assignments,
            steps,
        )
        compiled = deepcopy(base_model)
        material_map = {
            material.name: MaterialDefinition(
                material.name,
                deepcopy(dict(material.properties)),
            )
            for material in normalized.materials
        }
        section_map = {
            section.name: section for section in normalized.sections
        }
        metadata = getattr(compiled, "metadata", {})
        element_sets = dict(
            metadata.get("_abaqus_internal_element_sets", {})
        )
        element_sets.update(dict(getattr(compiled, "element_sets", {})))

        compiled_sections: list[SectionAssignment] = []
        for assignment in normalized.assignments:
            section = section_map[assignment.section_name]
            if assignment.region_name not in element_sets:
                raise KeyError(
                    f"region {assignment.region_name!r} is not an element set"
                )
            compiled_sections.append(
                SectionAssignment(
                    element_set=assignment.region_name,
                    material=section.material,
                    section_type=section.section_type,
                    properties=deepcopy(dict(section.properties)),
                )
            )

        compiled.materials = material_map
        compiled.sections = compiled_sections
        compiled.steps = deepcopy(list(normalized.steps))
        if compiled_sections:
            from fem.materials import resolve_sections

            resolution = resolve_sections(compiled)
            if resolution.issues:
                return DefinitionCompileResult(
                    definitions=deepcopy(normalized),
                    model=None,
                    diagnostics=tuple(
                        _section_resolution_diagnostic(issue)
                        for issue in resolution.issues
                    ),
                )
    except DefinitionRejected as error:
        return DefinitionCompileResult(
            definitions=None,
            model=None,
            diagnostics=error.diagnostics,
        )
    except (KeyError, TypeError, ValueError) as error:
        diagnostic = _definition_diagnostic(error)
        return DefinitionCompileResult(
            definitions=None,
            model=None,
            diagnostics=(diagnostic,),
        )
    return DefinitionCompileResult(
        definitions=deepcopy(normalized),
        model=compiled,
        diagnostics=(),
    )


def compiled_model_snapshot(
    base_model: Any,
    definitions: ModelDefinitions,
) -> Any:
    """Return a detached compiled model or raise ``DefinitionRejected``."""

    return compile_model_definitions(
        base_model,
        definitions,
    ).require_model()


def _mapping_values(
    value: Mapping[Any, Any] | Iterable[Any],
) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        return tuple(value.values())
    return tuple(value)


def _required_name(value: Any, label: str) -> str:
    if not hasattr(value, "name"):
        raise TypeError(f"{label} is missing a name")
    name = str(value.name).strip()
    if not name:
        raise ValueError(f"{label} name must not be empty")
    return name


def _validate_unique_names(values: Iterable[Any], label: str) -> None:
    seen: dict[str, str] = {}
    for value in values:
        name = str(value.name)
        key = name.casefold()
        if key in seen:
            raise ValueError(
                f"{label} names must be unique ignoring case: "
                f"{seen[key]!r} and {name!r}"
            )
        seen[key] = name


def _validate_definition_links(
    materials: tuple[MaterialDefinition, ...],
    sections: tuple[SectionDefinition, ...],
    assignments: tuple[RegionAssignment, ...],
) -> None:
    material_names = {material.name for material in materials}
    section_names = {section.name for section in sections}
    for section in sections:
        if not section.material:
            raise ValueError(
                f"section {section.name!r} material must not be empty"
            )
        if section.material not in material_names:
            raise ValueError(
                f"section {section.name!r} references missing material "
                f"{section.material!r}"
            )
    for assignment in assignments:
        if not assignment.section_name:
            raise ValueError(
                "assignment section name must not be empty"
            )
        if not assignment.region_name:
            raise ValueError("assignment region name must not be empty")
        if assignment.section_name not in section_names:
            raise ValueError(
                "assignment references missing section "
                f"{assignment.section_name!r}"
            )


def _definition_diagnostic(error: Exception) -> PreflightDiagnostic:
    message = str(error)
    lowered = message.casefold()
    if "material" in lowered:
        code = "definition.material.missing"
    elif "section" in lowered:
        code = "definition.section.missing"
    else:
        code = "step.reference.invalid"
    return PreflightDiagnostic(
        code=code,
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.DEFINITIONS,
        message=message,
        subject="model_definitions",
        path=("definitions",),
        remediation="请修正名称、引用和目标区域后重试。",
        details={"error_type": type(error).__name__},
    )


def _section_resolution_diagnostic(issue: Any) -> PreflightDiagnostic:
    code = (
        "definition.section.missing"
        if issue.code == "definition.section.reference_missing"
        else issue.code
    )
    return PreflightDiagnostic(
        code=code,
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.DEFINITIONS,
        message=issue.message,
        subject=(
            RegionRef("element_set", issue.element_set)
            if issue.element_set is not None
            else issue.element_id
        ),
        path=(
            "definitions",
            "sections",
            str(issue.assignment_index),
        ),
        remediation="请修复材料、截面参数或目标单元集。",
        details={
            "element_id": issue.element_id,
            "material": issue.material,
            "section_type": issue.section_type,
        },
    )


__all__ = [
    "DefinitionCompileResult",
    "DefinitionRejected",
    "FeatureRecord",
    "ModelDefinitions",
    "NamedRegion",
    "NativePart",
    "RegionAssignment",
    "SectionDefinition",
    "compile_model_definitions",
    "compiled_model_snapshot",
    "definitions_from_model",
    "normalize_model_definitions",
]
