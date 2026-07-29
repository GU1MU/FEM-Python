"""Headless editable definitions and their single detached compiler."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from fem.core.model import MaterialDefinition, SectionAssignment
from fem.elements import (
    BEAM_LOCAL_Y_REFERENCE_KEY,
    BeamOrientation,
    BeamOrientationError,
    get_element_capabilities,
    parse_beam_orientation,
    resolve_beam_frame,
)
from fem.geometry.references import LogicalEntityRef, logical_ref_sort_key

from .capabilities import RegionRef
from .diagnostics import (
    PreflightDiagnostic,
    PreflightSeverity,
    PreflightStage,
)
from .analysis_identity import validate_analysis_object_names


from .native_part import NativePart


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    """One item in the shallow native feature history."""

    name: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MeshEntityRef:
    """Stable reference to one entity in one generated finite-element mesh."""

    kind: str
    node_id: int | None = None
    element_id: int | None = None
    local_index: int | None = None
    node_ids: tuple[int, ...] = ()
    part_id: str | None = None

    def __post_init__(self) -> None:
        if self.part_id is not None:
            from .native_part import normalize_part_id

            object.__setattr__(
                self,
                "part_id",
                normalize_part_id(self.part_id, "mesh entity part_id"),
            )
        if self.kind not in {"node", "edge", "face", "element"}:
            raise ValueError(f"unsupported mesh entity kind: {self.kind!r}")
        node_ids = tuple(int(node_id) for node_id in self.node_ids)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("mesh entity node_ids must be unique")
        object.__setattr__(self, "node_ids", node_ids)
        if self.kind == "node":
            if (
                isinstance(self.node_id, bool)
                or not isinstance(self.node_id, int)
                or self.element_id is not None
                or self.local_index is not None
                or node_ids
            ):
                raise ValueError("node reference requires only one integer node_id")
            return
        if (
            isinstance(self.element_id, bool)
            or not isinstance(self.element_id, int)
            or self.node_id is not None
        ):
            raise ValueError(
                f"{self.kind} reference requires one integer element_id"
            )
        if self.kind == "element":
            if self.local_index is not None or node_ids:
                raise ValueError(
                    "element reference does not accept local_index or node_ids"
                )
            return
        if (
            isinstance(self.local_index, bool)
            or not isinstance(self.local_index, int)
            or self.local_index < 0
            or not node_ids
        ):
            raise ValueError(
                f"{self.kind} reference requires local_index and node_ids"
            )

    @classmethod
    def node(
        cls,
        node_id: int,
        *,
        part_id: str | None = None,
    ) -> MeshEntityRef:
        return cls("node", node_id=int(node_id), part_id=part_id)

    @classmethod
    def element(
        cls,
        element_id: int,
        *,
        part_id: str | None = None,
    ) -> MeshEntityRef:
        return cls(
            "element",
            element_id=int(element_id),
            part_id=part_id,
        )

    @classmethod
    def edge(
        cls,
        element_id: int,
        local_index: int,
        node_ids: Iterable[int],
        *,
        part_id: str | None = None,
    ) -> MeshEntityRef:
        return cls(
            "edge",
            element_id=int(element_id),
            local_index=int(local_index),
            node_ids=tuple(int(node_id) for node_id in node_ids),
            part_id=part_id,
        )

    @classmethod
    def face(
        cls,
        element_id: int,
        local_index: int,
        node_ids: Iterable[int],
        *,
        part_id: str | None = None,
    ) -> MeshEntityRef:
        return cls(
            "face",
            element_id=int(element_id),
            local_index=int(local_index),
            node_ids=tuple(int(node_id) for node_id in node_ids),
            part_id=part_id,
        )

    @property
    def identity(self) -> tuple[int, int]:
        """Return the integer identity used for canonical ordering."""

        if self.kind == "node":
            return int(self.node_id), -1
        return int(self.element_id), (
            -1 if self.local_index is None else int(self.local_index)
        )


def mesh_entity_ref_sort_key(
    reference: MeshEntityRef,
) -> tuple[str, int, int, int, tuple[int, ...]]:
    """Return a deterministic ordering key for mesh entity references."""

    kind_order = {"node": 0, "edge": 1, "face": 2, "element": 3}
    primary, local_index = reference.identity
    return (
        "" if reference.part_id is None else reference.part_id,
        kind_order[reference.kind],
        primary,
        local_index,
        reference.node_ids,
    )


@dataclass(frozen=True, slots=True)
class NamedRegion:
    """One user-authored scope on a generated finite-element mesh.

    Logical references remain readable for compatibility with older project
    files. New GUI authoring always stores :class:`MeshEntityRef` values.
    """

    name: str
    references: tuple[MeshEntityRef | LogicalEntityRef, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ValueError("named region name must be a non-empty string")
        references = tuple(self.references)
        if not references:
            raise ValueError("named region references must not be empty")
        reference_types = {type(reference) for reference in references}
        if not reference_types.issubset({MeshEntityRef, LogicalEntityRef}):
            raise TypeError(
                "named region references must contain MeshEntityRef or "
                "LogicalEntityRef values"
            )
        if len(reference_types) != 1:
            raise ValueError(
                "one named region cannot mix mesh and logical references"
            )
        if len(set(references)) != len(references):
            raise ValueError("named region references must be unique")
        kinds = {reference.kind for reference in references}
        if len(kinds) != 1:
            raise ValueError("one named region cannot mix entity kinds")
        object.__setattr__(self, "name", self.name.strip())
        sort_key = (
            mesh_entity_ref_sort_key
            if reference_types == {MeshEntityRef}
            else logical_ref_sort_key
        )
        object.__setattr__(
            self,
            "references",
            tuple(sorted(references, key=sort_key)),
        )

    @property
    def entity_kind(self) -> str:
        """Return the single kind derived from the canonical references."""

        return self.references[0].kind

    @property
    def logical_ids(self) -> tuple[str, ...]:
        """Return legacy logical IDs for compatibility consumers."""

        return tuple(
            reference.logical_id
            for reference in self.references
            if type(reference) is LogicalEntityRef
        )


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
    beam_orientation: BeamOrientation | None = None


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

    owned_materials_list: list[MaterialDefinition] = []
    for index, material in enumerate(deepcopy(tuple(material_values))):
        name = _required_name(material, "material")
        properties = deepcopy(dict(getattr(material, "properties", {})))
        if BEAM_LOCAL_Y_REFERENCE_KEY in properties:
            raise DefinitionRejected(
                (
                    _beam_orientation_diagnostic(
                        code="beam.orientation.invalid",
                        message=(
                            f"material {name!r} must not define reserved "
                            f"property {BEAM_LOCAL_Y_REFERENCE_KEY!r}"
                        ),
                        subject=name,
                        path=(
                            "definitions",
                            "materials",
                            str(index),
                            "properties",
                            BEAM_LOCAL_Y_REFERENCE_KEY,
                        ),
                        details={
                            "material_index": index,
                            "property": BEAM_LOCAL_Y_REFERENCE_KEY,
                        },
                    ),
                )
            )
        owned_materials_list.append(MaterialDefinition(name, properties))
    owned_materials = tuple(owned_materials_list)
    owned_sections_list: list[SectionDefinition] = []
    for index, section in enumerate(deepcopy(tuple(section_values))):
        properties = deepcopy(dict(section.properties))
        if BEAM_LOCAL_Y_REFERENCE_KEY in properties:
            raise DefinitionRejected(
                (
                    _beam_orientation_diagnostic(
                        code="beam.orientation.invalid",
                        message=(
                            f"section {_required_name(section, 'section')!r} "
                            f"must not define reserved property "
                            f"{BEAM_LOCAL_Y_REFERENCE_KEY!r}"
                        ),
                        subject=_required_name(section, "section"),
                        path=(
                            "definitions",
                            "sections",
                            str(index),
                            "properties",
                            BEAM_LOCAL_Y_REFERENCE_KEY,
                        ),
                        details={
                            "section_index": index,
                            "property": BEAM_LOCAL_Y_REFERENCE_KEY,
                        },
                    ),
                )
            )
        owned_sections_list.append(
            SectionDefinition(
                name=_required_name(section, "section"),
                material=str(section.material).strip(),
                section_type=str(section.section_type).strip().casefold(),
                properties=properties,
            )
        )
    owned_sections = tuple(owned_sections_list)

    owned_assignments_list: list[RegionAssignment] = []
    for index, assignment in enumerate(
        deepcopy(tuple(assignment_values))
    ):
        section_name = str(assignment.section_name).strip()
        region_name = str(assignment.region_name).strip()
        orientation = _normalize_beam_orientation(
            getattr(assignment, "beam_orientation", None),
            assignment_index=index,
            region_name=region_name,
        )
        owned_assignments_list.append(
            RegionAssignment(
                section_name=section_name,
                region_name=region_name,
                beam_orientation=orientation,
            )
        )
    owned_assignments = tuple(owned_assignments_list)
    owned_steps = deepcopy(tuple(step_values))
    for step in owned_steps:
        name = _required_name(step, "analysis step")
        step.name = name

    _validate_unique_names(owned_materials, "material")
    _validate_unique_names(owned_sections, "section")
    _validate_unique_names(owned_steps, "analysis step")
    validate_analysis_object_names(owned_steps, require_all=False)
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
        properties = deepcopy(dict(section.properties))
        orientation = None
        if BEAM_LOCAL_Y_REFERENCE_KEY in properties:
            raw_orientation = properties.pop(BEAM_LOCAL_Y_REFERENCE_KEY)
            try:
                orientation = parse_beam_orientation(raw_orientation)
            except BeamOrientationError as error:
                raise DefinitionRejected(
                    (
                        _beam_orientation_diagnostic(
                            code="beam.orientation.invalid",
                            message=str(error),
                            subject=_element_set_subject(
                                str(section.element_set)
                            ),
                            path=(
                                "definitions",
                                "assignments",
                                str(index - 1),
                                "beam_orientation",
                            ),
                            details={
                                "assignment_index": index - 1,
                                "element_set": str(section.element_set),
                                "reference": deepcopy(raw_orientation),
                                "error_type": type(error).__name__,
                            },
                        ),
                    )
                ) from error
        sections.append(
            SectionDefinition(
                name=name,
                material=str(section.material),
                section_type=str(section.section_type),
                properties=properties,
            )
        )
        assignments.append(
            RegionAssignment(
                section_name=name,
                region_name=str(section.element_set),
                beam_orientation=orientation,
            )
        )
    definitions = normalize_model_definitions(
        materials,
        sections,
        assignments,
        deepcopy(tuple(getattr(model, "steps", ()))),
    )
    target_diagnostics = _orientation_target_diagnostics(
        model,
        definitions,
    )
    if target_diagnostics:
        raise DefinitionRejected(target_diagnostics)
    if assignments:
        orientation_diagnostics = _compiled_orientation_diagnostics(
            model,
            definitions,
        )
        if orientation_diagnostics:
            raise DefinitionRejected(orientation_diagnostics)
    return definitions


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
            properties = deepcopy(dict(section.properties))
            if assignment.beam_orientation is not None:
                properties[BEAM_LOCAL_Y_REFERENCE_KEY] = tuple(
                    assignment.beam_orientation.local_y_reference
                )
            compiled_sections.append(
                SectionAssignment(
                    element_set=assignment.region_name,
                    material=section.material,
                    section_type=section.section_type,
                    properties=properties,
                )
            )

        compiled.materials = material_map
        compiled.sections = compiled_sections
        compiled.steps = deepcopy(list(normalized.steps))
        if compiled_sections:
            target_diagnostics = _orientation_target_diagnostics(
                compiled,
                normalized,
            )
            if target_diagnostics:
                return DefinitionCompileResult(
                    definitions=deepcopy(normalized),
                    model=None,
                    diagnostics=target_diagnostics,
                )
            from fem.materials import resolve_sections

            resolution = resolve_sections(compiled)
            if resolution.issues:
                return DefinitionCompileResult(
                    definitions=deepcopy(normalized),
                    model=None,
                    diagnostics=tuple(
                        _section_resolution_diagnostic(
                            issue,
                            normalized,
                        )
                        for issue in resolution.issues
                    ),
                )
            orientation_diagnostics = (
                _compiled_orientation_diagnostics(
                    compiled,
                    normalized,
                )
            )
            if orientation_diagnostics:
                return DefinitionCompileResult(
                    definitions=deepcopy(normalized),
                    model=None,
                    diagnostics=orientation_diagnostics,
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


def _normalize_beam_orientation(
    value: Any,
    *,
    assignment_index: int,
    region_name: str,
) -> BeamOrientation | None:
    if value is None:
        return None

    if not isinstance(value, BeamOrientation):
        error = TypeError(
            "assignment beam_orientation must be BeamOrientation or None"
        )
        raise DefinitionRejected(
            (
                _beam_orientation_diagnostic(
                    code="beam.orientation.invalid",
                    message=str(error),
                    subject=_element_set_subject(region_name),
                    path=(
                        "definitions",
                        "assignments",
                        str(assignment_index),
                        "beam_orientation",
                    ),
                    details={
                        "assignment_index": assignment_index,
                        "element_set": region_name,
                        "value_type": type(value).__name__,
                    },
                ),
            )
        ) from error
    return deepcopy(value)


def _compiled_orientation_diagnostics(
    model: Any,
    definitions: ModelDefinitions,
) -> tuple[PreflightDiagnostic, ...]:
    """Validate every authored explicit direction against its whole target."""

    from fem.materials import (
        MaterialPropertyError,
        SectionCompatibilityError,
        SectionPropertyError,
        resolve_section_properties,
        restored_element_properties,
    )

    element_lookup = {
        int(element.id): element
        for element in getattr(
            getattr(model, "mesh", None),
            "elements",
            (),
        )
    }
    metadata = getattr(model, "metadata", {})
    element_sets = dict(
        metadata.get("_abaqus_internal_element_sets", {})
    )
    element_sets.update(dict(getattr(model, "element_sets", {})))
    core_sections = tuple(getattr(model, "sections", ()))
    materials = getattr(model, "materials", {})
    diagnostics: list[PreflightDiagnostic] = []

    for assignment_index, assignment in enumerate(definitions.assignments):
        orientation = assignment.beam_orientation
        if orientation is None:
            continue
        if assignment_index >= len(core_sections):
            continue
        core_section = core_sections[assignment_index]
        element_set = element_sets.get(assignment.region_name)
        material = materials.get(str(core_section.material))
        if element_set is None or material is None:
            continue
        for raw_element_id in getattr(element_set, "element_ids", ()):
            element_id = int(raw_element_id)
            element = element_lookup.get(element_id)
            if element is None:
                continue
            try:
                resolved = resolve_section_properties(
                    str(element.type),
                    material.properties,
                    str(core_section.section_type),
                    core_section.properties,
                    baseline_properties=restored_element_properties(
                        model,
                        element_id,
                        element,
                    ),
                )
            except (
                MaterialPropertyError,
                SectionCompatibilityError,
                SectionPropertyError,
                NotImplementedError,
                KeyError,
                TypeError,
                ValueError,
            ):
                # The ordinary section-resolution diagnostics own these
                # schema/reference failures.
                continue
            try:
                resolve_beam_frame(
                    model.mesh,
                    element,
                    properties=deepcopy(resolved.effective_properties),
                )
            except BeamOrientationError as error:
                details = {
                    "assignment_index": assignment_index,
                    "element_set": assignment.region_name,
                    "element_id": element_id,
                    "reference": tuple(orientation.local_y_reference),
                    "operation": "section.assignment",
                    "error_type": type(error).__name__,
                }
                tangent = getattr(error, "tangent", None)
                if tangent is not None:
                    details["element_tangent"] = deepcopy(tangent)
                diagnostics.append(
                    _beam_orientation_diagnostic(
                        code=_beam_frame_error_code(error),
                        message=str(error),
                        subject=_element_set_subject(
                            assignment.region_name
                        ),
                        path=(
                            "definitions",
                            "assignments",
                            str(assignment_index),
                            "beam_orientation",
                        ),
                        details=details,
                    )
                )
    return tuple(diagnostics)


def _orientation_target_diagnostics(
    model: Any,
    definitions: ModelDefinitions,
) -> tuple[PreflightDiagnostic, ...]:
    oriented_assignments = tuple(
        (index, assignment)
        for index, assignment in enumerate(definitions.assignments)
        if assignment.beam_orientation is not None
    )
    if not oriented_assignments:
        return ()

    element_lookup = {
        int(element.id): element
        for element in getattr(
            getattr(model, "mesh", None),
            "elements",
            (),
        )
    }
    element_sets = dict(
        getattr(model, "metadata", {}).get(
            "_abaqus_internal_element_sets",
            {},
        )
    )
    element_sets.update(dict(getattr(model, "element_sets", {})))
    diagnostics: list[PreflightDiagnostic] = []

    for assignment_index, assignment in oriented_assignments:
        orientation = assignment.beam_orientation
        assert orientation is not None
        element_set = element_sets.get(assignment.region_name)
        if element_set is None:
            # The ordinary section-resolution path reports this reference.
            continue
        target_element_ids = tuple(
            getattr(element_set, "element_ids", ())
        )
        if not target_element_ids:
            diagnostics.append(
                _beam_orientation_diagnostic(
                    code="beam.orientation.unsupported_target",
                    message=(
                        "Beam orientation requires a non-empty Beam2 "
                        f"element set; {assignment.region_name!r} is empty"
                    ),
                    subject=_element_set_subject(
                        assignment.region_name
                    ),
                    path=(
                        "definitions",
                        "assignments",
                        str(assignment_index),
                        "beam_orientation",
                    ),
                    details={
                        "assignment_index": assignment_index,
                        "element_set": assignment.region_name,
                        "reference": tuple(
                            orientation.local_y_reference
                        ),
                        "operation": "section.assignment",
                    },
                )
            )
            continue
        for raw_element_id in target_element_ids:
            element_id = int(raw_element_id)
            element = element_lookup.get(element_id)
            if element is None:
                continue
            try:
                descriptor = get_element_capabilities(str(element.type))
            except (KeyError, NotImplementedError, TypeError, ValueError):
                descriptor = None
            if (
                descriptor is None
                or descriptor.canonical_type != "Beam2"
            ):
                diagnostics.append(
                    _beam_orientation_diagnostic(
                        code="beam.orientation.unsupported_target",
                        message=(
                            "Beam orientation can target only Beam2 "
                            f"elements; element {element_id} has type "
                            f"{element.type!r}"
                        ),
                        subject=_element_set_subject(
                            assignment.region_name
                        ),
                        path=(
                            "definitions",
                            "assignments",
                            str(assignment_index),
                            "beam_orientation",
                        ),
                        details={
                            "assignment_index": assignment_index,
                            "element_set": assignment.region_name,
                            "element_id": element_id,
                            "element_type": str(element.type),
                            "reference": tuple(
                                orientation.local_y_reference
                            ),
                            "operation": "section.assignment",
                        },
                    )
                )
    return tuple(diagnostics)


def _beam_frame_error_code(error: Exception) -> str:
    code = getattr(error, "code", None)
    if code in {
        "beam.orientation.invalid",
        "beam.orientation.parallel",
        "beam.orientation.unsupported_target",
    }:
        return str(code)
    return "beam.orientation.invalid"


def _element_set_subject(name: str) -> Any:
    try:
        return RegionRef("element_set", name)
    except ValueError:
        return str(name)


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


def _beam_orientation_diagnostic(
    *,
    code: str,
    message: str,
    subject: Any,
    path: Iterable[str],
    details: Mapping[str, Any],
) -> PreflightDiagnostic:
    remediation = {
        "beam.orientation.invalid": (
            "请提供三个有限、非零的全局局部 y 参考方向分量。"
        ),
        "beam.orientation.parallel": (
            "请让参考方向与目标梁单元轴线保持明显非平行。"
        ),
        "beam.orientation.unsupported_target": (
            "请仅将 Beam orientation 用于完全由 Beam2 单元组成的区域。"
        ),
    }.get(code, "请修正 Beam orientation 后重试。")
    return PreflightDiagnostic(
        code=code,
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.DEFINITIONS,
        message=str(message),
        subject=subject,
        path=tuple(path),
        remediation=remediation,
        details=details,
    )


def _section_resolution_diagnostic(
    issue: Any,
    definitions: ModelDefinitions | None = None,
) -> PreflightDiagnostic:
    code = (
        "definition.section.missing"
        if issue.code == "definition.section.reference_missing"
        else issue.code
    )
    if str(code).startswith("beam.orientation."):
        assignment_index = issue.assignment_index
        orientation = None
        if (
            definitions is not None
            and assignment_index is not None
            and 0 <= int(assignment_index) < len(definitions.assignments)
        ):
            orientation = definitions.assignments[
                int(assignment_index)
            ].beam_orientation
        details: dict[str, Any] = {
            "assignment_index": assignment_index,
            "element_id": issue.element_id,
            "element_set": issue.element_set,
            "material": issue.material,
            "section_type": issue.section_type,
            "operation": "section.assignment",
        }
        if orientation is not None:
            details["reference"] = tuple(
                orientation.local_y_reference
            )
        return _beam_orientation_diagnostic(
            code=str(code),
            message=issue.message,
            subject=(
                _element_set_subject(issue.element_set)
                if issue.element_set is not None
                else issue.element_id
            ),
            path=(
                "definitions",
                "assignments",
                str(assignment_index),
                "beam_orientation",
            ),
            details=details,
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
