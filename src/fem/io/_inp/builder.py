from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ...core.mesh import Element2D, Element3D, Mesh2D, Mesh3D, Node2D, Node3D
from ...core.model import (
    AnalysisStep,
    DisplacementConstraint,
    Edge,
    EdgeLoad,
    ElementEdge,
    ElementFace,
    ElementSet,
    FEMModel,
    GravityLoad,
    MaterialDefinition,
    LineLoad,
    NodalLoad,
    NodeSet,
    OutputRequest,
    OutputSourceEvidence,
    SectionAssignment,
    Surface,
    SurfaceLoad,
)
from ...elements import (
    BEAM_DEFAULT_LOCAL_Y_REFERENCE,
    BEAM_DEFAULT_LOCAL_Y_REFERENCE_KEY,
    BEAM_FRAME_FIELD_KEY,
    BEAM_FRAME_FIELD_REFERENCE_KEY,
    BEAM_ELEMENT_LOCAL_Y_REFERENCE_KEY,
    BEAM_LOCAL_Y_REFERENCE_KEY,
    BeamFrameField,
    canonical_element_type,
    get_element_capabilities,
    resolve_beam_frame_field,
    validate_beam_frame_field,
)
from ...elements.line import line3d_geometry
from ...materials import resolve_sections
from ...selection import edges as edge_selection
from ...selection import faces as face_selection
from ..inp import (
    InpImportNotice,
    InpImportResult,
    InpKeywordCategory,
)
from .contracts import STANDARD_LINE_SUBSET, classify_keyword
from .deck import (
    AbaqusBeamSectionData,
    AbaqusBoundary,
    AbaqusDeck,
    AbaqusDistributedLoad,
    AbaqusElement,
    AbaqusSection,
    AbaqusSolidSectionData,
    AbaqusStep,
)
from .errors import (
    AbaqusBuildError,
    AbaqusSourceLocation,
    UnsupportedAbaqusFeatureError,
)
from .orientation import (
    AbaqusOrientationResolution,
    AbaqusOrientationResolutionReport,
    resolve_b31_orientations,
)


# Compatibility names remain available while the public value types are owned
# by ``fem.io.inp``.
AbaqusImportNotice = InpImportNotice
AbaqusBuildResult = InpImportResult


def build_model(deck: AbaqusDeck) -> FEMModel:
    """Build a model while intentionally discarding import notices.

    Call :func:`build_model_with_report` when the caller presents imported
    results to a user and therefore must retain formulation limitations.
    """

    return build_model_with_report(deck).model


def build_model_with_report(deck: AbaqusDeck) -> AbaqusBuildResult:
    """Build a detached model and retain non-authoritative source notices."""

    # Background import callers may retain or reuse the parser result.  Work on
    # an owned snapshot so source validation and set expansion cannot share
    # mutable parser state with the accepted task.
    deck = deck.snapshot()
    # Resolve source targets before constructing a canonical mesh.  This keeps
    # mixed-family and stale-set diagnostics at the Abaqus boundary instead of
    # letting a generic Mesh constructor error hide the responsible record.
    _audit_raw_targets(deck)
    orientation_resolution = resolve_b31_orientations(deck)
    orientation_resolution.raise_if_invalid()
    has_line_elements = _audit_line_subset(deck)
    _audit_source_mesh_capability(deck)
    model = _build_model(deck)
    _install_b31_source_orientations(model, orientation_resolution)
    section_resolution = _validate_declared_sections(model, deck)
    if has_line_elements:
        _validate_line_model(model, deck, section_resolution)
    notices = _build_import_notices(deck, orientation_resolution.report)
    return AbaqusBuildResult(model=model, notices=notices)


def _build_model(deck: AbaqusDeck) -> FEMModel:
    """Build a FEMModel from a parsed Abaqus input deck."""
    mesh = _build_mesh(deck)
    orientation_only_nodes = _orientation_only_node_ids(deck)
    node_sets = {
        name: NodeSet(
            name,
            tuple(
                node_id
                for node_id in _unique_ids(ids)
                if node_id not in orientation_only_nodes
            ),
        )
        for name, ids in deck.node_sets.items()
        if deck.node_set_scopes.get(name, "model") != "part"
    }
    element_sets = {
        name: ElementSet(name, _unique_ids(ids))
        for name, ids in deck.element_sets.items()
    }
    visible_element_sets = {
        name: element_set
        for name, element_set in element_sets.items()
        if not _is_internal_element_set(name)
    }
    internal_element_sets = {
        name: element_set
        for name, element_set in element_sets.items()
        if _is_internal_element_set(name)
    }
    materials = {
        name: MaterialDefinition(name, dict(material.properties))
        for name, material in deck.materials.items()
    }
    sections = _build_sections(
        deck,
        mesh,
        element_sets,
        internal_element_sets,
    )

    topology: dict[str, Any] = {
        "elements": {elem.id: elem for elem in mesh.elements},
    }
    surfaces, edges = _build_surfaces_and_edges(mesh, deck, element_sets, topology)
    steps = [
        _build_step(
            step, mesh, surfaces, edges, element_sets, step_index, topology
        )
        for step_index, step in enumerate(deck.steps)
    ]
    model = FEMModel(
        mesh=mesh,
        name=deck.name,
        node_sets=node_sets,
        element_sets=visible_element_sets,
        edges=edges,
        surfaces=surfaces,
        materials=materials,
        sections=sections,
        steps=steps,
        metadata={"_abaqus_internal_element_sets": internal_element_sets},
    )
    return model


def _install_b31_source_orientations(
    model: FEMModel,
    resolution: AbaqusOrientationResolution,
) -> None:
    """Project resolver output into the generic core Beam2 frame contract."""

    field = resolution.field
    for mesh_element in model.mesh.elements:
        if str(getattr(mesh_element, "props", {}).get("abaqus_type", "")).upper() != "B31":
            continue
        element_id = int(mesh_element.id)
        entries = field.for_element(element_id)
        if len(entries) != 2:
            locations = tuple(
                location
                for entry in entries
                for location in entry.source_locations
            )
            raise AbaqusBuildError(
                f"B31 element {element_id} does not have two resolved end frames",
                code="abaqus.b31.frame_field_invalid",
                location=locations[0] if locations else None,
                locations=locations,
                record={"element": element_id, "end_count": len(entries)},
            )
        ordered = tuple(sorted(entries, key=lambda entry: entry.local_end))
        try:
            length, tangent = line3d_geometry(model.mesh, mesh_element)
            frame_field = BeamFrameField.from_axes(
                length,
                ordered[0].tangent,
                ordered[0].n1,
                ordered[0].normal,
                ordered[1].tangent,
                ordered[1].n1,
                ordered[1].normal,
            )
            validate_beam_frame_field(
                frame_field,
                length=length,
                tangent=tangent,
                element_id=element_id,
            )
        except (TypeError, ValueError) as exc:
            locations = tuple(
                location
                for entry in ordered
                for location in entry.source_locations
            )
            raise AbaqusBuildError(
                f"B31 element {element_id} has invalid core frame field: {exc}",
                code=getattr(exc, "code", "abaqus.b31.frame_field_invalid"),
                location=locations[0] if locations else None,
                locations=locations,
                record={
                    "element": element_id,
                    "nodes": tuple(int(value) for value in mesh_element.node_ids),
                    "ends": tuple(
                        {
                            "local_end": entry.local_end,
                            "tangent": entry.tangent,
                            "n1": entry.n1,
                            "normal": entry.normal,
                        }
                        for entry in ordered
                    ),
                },
            ) from exc
        mesh_element.props[BEAM_FRAME_FIELD_KEY] = frame_field
        effective_reference = field.constant_reference(element_id)
        baseline_reference = tuple(
            float(value) for value in ordered[0].approximate_n1
        )
        if all(
            math.isclose(
                first,
                float(second),
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
            for first, second in zip(
                baseline_reference,
                ordered[1].approximate_n1,
                strict=True,
            )
        ):
            mesh_element.props[BEAM_FRAME_FIELD_REFERENCE_KEY] = tuple(
                baseline_reference
            )
        requires_override = any(
            entry.reference_source == "orientation-node"
            or entry.normal_source in {"element-normal", "node-normal"}
            for entry in entries
        )
        if effective_reference is not None and requires_override:
            mesh_element.props[BEAM_ELEMENT_LOCAL_Y_REFERENCE_KEY] = (
                effective_reference
            )


_LINE_KEYWORD_PARAMETERS: dict[str, frozenset[str]] = {
    "heading": frozenset(),
    "preprint": frozenset({"echo", "history", "model", "contact"}),
    "part": frozenset({"name"}),
    "end part": frozenset(),
    "assembly": frozenset({"name"}),
    "end assembly": frozenset(),
    "instance": frozenset({"name", "part"}),
    "end instance": frozenset(),
    "node": frozenset(),
    "element": frozenset({"type", "elset"}),
    "nset": frozenset({"nset", "instance"}),
    "elset": frozenset({"elset", "instance"}),
    "material": frozenset({"name"}),
    "elastic": frozenset(),
    "density": frozenset(),
    "solid section": frozenset({"elset", "material"}),
    "beam section": frozenset({"elset", "material", "section"}),
    "step": frozenset({"name", "nlgeom"}),
    "static": frozenset(),
    "boundary": frozenset(),
    "cload": frozenset(),
    "dload": frozenset(),
    "normal": frozenset({"type"}),
    "output": frozenset({"variable"}),
    "field output": frozenset({"name", "variable"}),
    "history output": frozenset({"name", "variable"}),
    "node output": frozenset({"nset"}),
    "element output": frozenset({"elset", "position", "directions"}),
    "end step": frozenset(),
}
_LINE_KEYWORD_FLAGS: dict[str, frozenset[str]] = {
    "nset": frozenset({"generate", "unsorted", "internal"}),
    "elset": frozenset({"generate", "unsorted", "internal"}),
    "output": frozenset({"field", "history"}),
}
_LINE_KEYWORD_REQUIRED: dict[str, frozenset[str]] = {
    "part": frozenset({"name"}),
    "instance": frozenset({"name", "part"}),
    "element": frozenset({"type"}),
    "nset": frozenset({"nset"}),
    "elset": frozenset({"elset"}),
    "material": frozenset({"name"}),
    "solid section": frozenset({"elset", "material"}),
    "beam section": frozenset({"elset", "material", "section"}),
    "step": frozenset({"name"}),
}


def _audit_raw_targets(deck: AbaqusDeck) -> None:
    """Validate section and DLOAD targets against raw source elements."""

    element_lookup = {
        int(element.id): element
        for element in deck.elements
    }
    for section in deck.sections:
        element_ids = _raw_section_target(deck, section, element_lookup)
        _raw_target_family(
            element_ids,
            element_lookup,
            location=section.keyword_location,
            subject=f"section target {section.element_set!r}",
        )

    for step in deck.steps:
        for load in step.distributed_loads:
            if (
                str(load.source).casefold() != "dload"
                or str(load.label).upper() == "GRAV"
            ):
                continue
            element_ids = _raw_dload_target(deck, load, element_lookup)
            _raw_target_family(
                element_ids,
                element_lookup,
                location=load.location,
                subject=f"*DLOAD target {load.target!r}",
            )


def _raw_section_target(
    deck: AbaqusDeck,
    section: AbaqusSection,
    element_lookup: dict[int, AbaqusElement],
) -> tuple[int, ...]:
    """Resolve a section target with definition-time empty-set evidence."""

    target_was_defined = getattr(section, "target_was_defined", None)
    captured_ids = _unique_ids(section.element_ids)
    if target_was_defined is True:
        element_ids = captured_ids
    elif target_was_defined is False:
        if captured_ids:
            # Compatibility for direct DTO construction predating the
            # target_was_defined field.  Parser-produced forward references
            # always carry an empty captured tuple.
            element_ids = captured_ids
        elif section.element_set not in deck.element_sets:
            raise AbaqusBuildError(
                f"element set {section.element_set!r} is not defined",
                code="abaqus.section.target_missing",
                location=section.keyword_location,
                remediation=(
                    "Define a non-empty ELSET before assigning the section."
                ),
            )
        else:
            element_ids = _unique_ids(deck.element_sets[section.element_set])
    else:
        # Compatibility for manually constructed pre-evidence DTOs.
        if (
            section.element_set not in deck.element_sets
            and not captured_ids
        ):
            raise AbaqusBuildError(
                f"element set {section.element_set!r} is not defined",
                code="abaqus.section.target_missing",
                location=section.keyword_location,
                remediation=(
                    "Define a non-empty ELSET before assigning the section."
                ),
            )
        element_ids = (
            captured_ids
            or _unique_ids(deck.element_sets.get(section.element_set, ()))
        )
    if not element_ids:
        raise AbaqusBuildError(
            f"element set {section.element_set!r} is empty",
            code="abaqus.section.target_empty",
            location=section.keyword_location,
            remediation="Assign the section to a non-empty element set.",
        )
    missing_ids = tuple(
        element_id
        for element_id in element_ids
        if element_id not in element_lookup
    )
    if missing_ids:
        raise AbaqusBuildError(
            (
                f"section target {section.element_set!r} references undefined "
                f"element IDs {missing_ids}"
            ),
            code="abaqus.section.element_missing",
            location=section.keyword_location,
            record=missing_ids,
            remediation="Remove undefined IDs or define the referenced elements.",
        )
    return element_ids


def _raw_dload_target(
    deck: AbaqusDeck,
    load: AbaqusDistributedLoad,
    element_lookup: dict[int, AbaqusElement],
) -> tuple[int, ...]:
    """Resolve one non-gravity DLOAD target without a canonical mesh."""

    if load.target is None:
        raise AbaqusBuildError(
            f"*DLOAD {load.label} requires an element target",
            code="abaqus.dload.target_missing",
            location=load.location,
            record=(load.target, load.label),
        )
    if isinstance(load.target, int):
        element_ids = (int(load.target),)
    else:
        if load.target not in deck.element_sets:
            raise AbaqusBuildError(
                f"element set {load.target} is not defined",
                code="abaqus.dload.target_undefined",
                location=load.location,
                record=load.target,
            )
        element_ids = _unique_ids(deck.element_sets[load.target])
    if not element_ids:
        raise AbaqusBuildError(
            f"*DLOAD target {load.target!r} is empty",
            code="abaqus.dload.target_empty",
            location=load.location,
            record=load.target,
        )
    missing_ids = tuple(
        element_id
        for element_id in element_ids
        if element_id not in element_lookup
    )
    if missing_ids:
        raise AbaqusBuildError(
            f"*DLOAD target references undefined element IDs {missing_ids}",
            code="abaqus.dload.element_missing",
            location=load.location,
            record=missing_ids,
        )
    return element_ids


def _raw_target_family(
    element_ids: tuple[int, ...],
    element_lookup: dict[int, AbaqusElement],
    *,
    location: AbaqusSourceLocation | None,
    subject: str,
) -> str:
    """Return one family using Abaqus elements without building a mesh."""

    families = {
        _source_element_family(element_lookup[element_id])
        for element_id in element_ids
    }
    if len(families) != 1:
        raise AbaqusBuildError(
            f"{subject} mixes element families {tuple(sorted(families))}",
            code="abaqus.target.family_mixed",
            location=location,
            record=element_ids,
            remediation="Split the target into one element family per record.",
        )
    return next(iter(families))


def _source_element_family(element: AbaqusElement) -> str:
    """Return the canonical family represented by one raw Abaqus element."""

    element_type = str(element.type).upper()
    aliases = {
        "B31": "Beam2",
        "T3D2": "Truss2",
    }
    candidate = aliases.get(element_type, element.type)
    try:
        canonical = canonical_element_type(candidate)
        return get_element_capabilities(canonical).family
    except (NotImplementedError, ValueError) as exc:
        raise UnsupportedAbaqusFeatureError(
            f"unsupported Abaqus element type {element.type!r}",
            code="abaqus.element_type.unsupported",
            location=element.keyword_location or element.data_location,
            record=element.type,
            remediation="Use an element type implemented by the FEM kernel.",
        ) from exc


def _audit_source_mesh_capability(deck: AbaqusDeck) -> None:
    """Fail closed when one source deck requires mixed canonical families."""

    families: dict[str, list[str]] = defaultdict(list)
    for element in deck.elements:
        family = _source_element_family(element)
        element_type = str(element.type).upper()
        if element_type not in families[family]:
            families[family].append(element_type)
    if len(families) <= 1:
        return
    first = next(iter(deck.elements), None)
    raise UnsupportedAbaqusFeatureError(
        (
            "one imported model cannot mix canonical element families "
            f"{tuple(sorted(families))}"
        ),
        code="abaqus.mesh.element_family_mixed",
        location=(
            None
            if first is None
            else first.keyword_location or first.data_location
        ),
        record={
            family: tuple(element_types)
            for family, element_types in families.items()
        },
        remediation="Import each canonical element family as a separate model.",
    )


def _audit_line_subset(deck: AbaqusDeck) -> bool:
    """Fail closed on engineering semantics in decks containing line elements."""

    wire_types = {str(element.type).upper() for element in deck.elements}
    standard_line_types = set(STANDARD_LINE_SUBSET.element_types)
    has_line_elements = bool(wire_types & standard_line_types)
    if not has_line_elements:
        return False
    # Legal continuum source types are not line-keyword violations.  A deck
    # that combines them with B31/T3D2 is rejected later by the typed mesh
    # capability audit, after target-level mixed-family diagnostics have had
    # an opportunity to identify the responsible section or DLOAD.
    for element in deck.elements:
        element_type = str(element.type).upper()
        if element_type in standard_line_types:
            continue
        try:
            _source_element_family(element)
        except UnsupportedAbaqusFeatureError as exc:
            raise UnsupportedAbaqusFeatureError(
                (
                    "unsupported element families are outside the Phase 6 "
                    f"line subset: {element_type!r}"
                ),
                code="abaqus.line.element_family_unsupported",
                location=element.keyword_location or element.data_location,
                record=tuple(sorted(wire_types)),
                remediation=(
                    "Use B31/T3D2 or a continuum element implemented by the "
                    "current FEM kernel."
                ),
            ) from exc

    for occurrence in deck.keyword_occurrences:
        name = occurrence.name
        if name == "dsload":
            raise UnsupportedAbaqusFeatureError(
                "B31/T3D2 distributed loads must not use *DSLOAD",
                code="abaqus.line.dsload_unsupported",
                location=occurrence.location,
                remediation="Use supported *DLOAD records for B31 or GRAV.",
            )
        category = classify_keyword(name)
        if category is InpKeywordCategory.UNSUPPORTED_ENGINEERING_SEMANTICS:
            remediation = _unsupported_keyword_remediation(name)
            raise UnsupportedAbaqusFeatureError(
                (
                    f"*{name.upper()} is outside the accepted "
                    "B31/T3D2 linear-static input vocabulary"
                ),
                code="abaqus.line.keyword_unsupported",
                location=occurrence.location,
                record=name,
                remediation=remediation,
            )
        if category is InpKeywordCategory.HARMLESS_IGNORED:
            continue
        if category in {
            InpKeywordCategory.POSTPROCESS_CANDIDATE,
            InpKeywordCategory.PRESERVED,
        }:
            # Output options are preserved authoring evidence.  Concrete
            # postprocessing support is classified after solve.
            continue
        params = {
            str(key).casefold(): value
            for key, value in dict(occurrence.params).items()
        }
        flags = {str(flag).casefold() for flag in occurrence.flags}
        allowed_params = _LINE_KEYWORD_PARAMETERS.get(name, frozenset())
        allowed_flags = _LINE_KEYWORD_FLAGS.get(name, frozenset())
        unknown_params = tuple(sorted(set(params) - set(allowed_params)))
        unknown_flags = tuple(sorted(flags - set(allowed_flags)))
        missing = tuple(
            sorted(
                set(_LINE_KEYWORD_REQUIRED.get(name, frozenset()))
                - set(params)
            )
        )
        if missing:
            raise AbaqusBuildError(
                f"*{name.upper()} is missing required parameters {missing}",
                code="abaqus.line.keyword_parameter_missing",
                location=occurrence.location,
                record=missing,
            )
        if unknown_params or unknown_flags:
            details = (*unknown_params, *unknown_flags)
            raise UnsupportedAbaqusFeatureError(
                (
                    f"*{name.upper()} uses unsupported parameters or flags "
                    f"{details}"
                ),
                code="abaqus.line.keyword_option_unsupported",
                location=occurrence.location,
                record=details,
                remediation=(
                    "Remove unsupported options and use the exact linear-static "
                    "B31/T3D2 subset."
                ),
            )
        if name == "step" and str(params.get("nlgeom", "no")).strip().casefold() not in {
            "no",
            "false",
            "0",
        }:
            raise UnsupportedAbaqusFeatureError(
                "NLGEOM=YES is not supported by the linear Beam2/Truss2 solver",
                code="abaqus.line.nlgeom_unsupported",
                location=occurrence.location,
                remediation="Use NLGEOM=NO or remove the NLGEOM option.",
            )

    _audit_line_steps(deck)
    _audit_line_materials(deck)
    return True


def _unsupported_keyword_remediation(name: str) -> str:
    if name == "normal":
        return (
            "Use *NORMAL, TYPE=ELEMENT with a B31 target, or remove the "
            "unsupported normal definition."
        )
    if name == "amplitude":
        return "Replace amplitude-dependent loading with constant step data."
    if name in {"dynamic", "frequency", "buckle", "visco", "plastic"}:
        return "Use a linear-elastic *STATIC procedure in the supported subset."
    return (
        "Remove the keyword or wait for an adapter capability that executes "
        "its engineering semantics."
    )


def _audit_line_steps(deck: AbaqusDeck) -> None:
    """Require one explicit STATIC procedure in every analysis step."""

    for step in deck.steps:
        step_location = getattr(step, "keyword_location", None)
        procedure_location = getattr(step, "procedure_location", None)
        count = getattr(step, "procedure_count", None)
        if count is None:
            present = getattr(step, "procedure_present", None)
            count = 1 if present is True else 0 if present is False else 1
        count = int(count)
        if str(step.name).casefold() == "initial" and step_location is None:
            if count == 0:
                continue
            raise AbaqusBuildError(
                "*STATIC appears outside an explicit *STEP block",
                code="abaqus.line.step.static_outside_step",
                location=procedure_location,
                record=count,
                remediation=(
                    "Place exactly one *STATIC inside each explicit *STEP; "
                    "the implicit Initial step may contain initial conditions "
                    "and boundaries only."
                ),
            )
        if count == 0:
            raise AbaqusBuildError(
                f"analysis step {step.name!r} is missing *STATIC",
                code="abaqus.line.step.static_missing",
                location=step_location,
                record=step.name,
                remediation=(
                    "Add exactly one *STATIC procedure after the *STEP record."
                ),
            )
        if count != 1:
            raise AbaqusBuildError(
                (
                    f"analysis step {step.name!r} contains {count} procedure "
                    "records; exactly one *STATIC is required"
                ),
                code="abaqus.line.step.static_count",
                location=procedure_location or step_location,
                record=(step.name, count),
                remediation="Retain exactly one *STATIC procedure in the step.",
            )
        if str(step.procedure).casefold() != "static":
            raise UnsupportedAbaqusFeatureError(
                f"unsupported line procedure {step.procedure!r}",
                code="abaqus.line.step.procedure_unsupported",
                location=procedure_location or step_location,
                record=step.procedure,
                remediation="Use the supported linear *STATIC procedure.",
            )


def _audit_line_materials(deck: AbaqusDeck) -> None:
    """Reject material tables that cannot map to one constant core material."""

    for material in deck.materials.values():
        elastic_records = tuple(
            getattr(material, "elastic_records", ())
        )
        density_records = tuple(
            getattr(material, "density_records", ())
        )
        elastic_keyword_count = getattr(
            material,
            "elastic_keyword_count",
            1 if "E" in material.properties else 0,
        )
        density_keyword_count = getattr(
            material,
            "density_keyword_count",
            1 if "rho" in material.properties else 0,
        )
        material_location = _material_keyword_location(deck, material.name)

        if int(elastic_keyword_count) == 0:
            raise AbaqusBuildError(
                (
                    f"line material {material.name!r} is missing the required "
                    "*ELASTIC declaration"
                ),
                code="abaqus.line.material.elastic_missing",
                location=material_location,
                record=material.name,
                remediation=(
                    "Declare exactly one *ELASTIC keyword with one constant "
                    "E, nu data record."
                ),
            )
        if int(elastic_keyword_count) != 1:
            location = _record_location(
                elastic_records,
                prefer_second=int(elastic_keyword_count) > 1,
            )
            raise AbaqusBuildError(
                (
                    f"line material {material.name!r} must declare exactly "
                    "one *ELASTIC keyword"
                ),
                code="abaqus.line.material.elastic_record_count",
                location=location or material_location,
                record=(material.name, int(elastic_keyword_count)),
                remediation=(
                    "Declare one *ELASTIC keyword with one E, nu data record."
                ),
            )
        if len(elastic_records) != 1:
            raise AbaqusBuildError(
                (
                    f"line material {material.name!r} must contain exactly "
                    "one *ELASTIC data record"
                ),
                code="abaqus.line.material.elastic_record_count",
                location=(
                    _record_location(
                        elastic_records,
                        prefer_second=len(elastic_records) > 1,
                    )
                    or material_location
                ),
                record=(material.name, len(elastic_records)),
                remediation="Provide one constant E, nu record without a table.",
            )
        elastic = elastic_records[0]
        if int(elastic.field_count) != 2:
            raise AbaqusBuildError(
                (
                    f"line material {material.name!r} *ELASTIC data must "
                    "contain exactly E and nu"
                ),
                code="abaqus.line.material.elastic_shape",
                location=elastic.location or material_location,
                record=elastic.fields,
                remediation=(
                    "Remove temperature/field columns and provide exactly "
                    "two constant values."
                ),
            )

        if int(density_keyword_count) == 0 and not density_records:
            continue
        if int(density_keyword_count) != 1:
            raise AbaqusBuildError(
                (
                    f"line material {material.name!r} may declare at most "
                    "one *DENSITY keyword"
                ),
                code="abaqus.line.material.density_record_count",
                location=(
                    _record_location(
                        density_records,
                        prefer_second=int(density_keyword_count) > 1,
                    )
                    or material_location
                ),
                record=(material.name, int(density_keyword_count)),
                remediation=(
                    "Use at most one *DENSITY keyword with one scalar record."
                ),
            )
        if len(density_records) != 1:
            raise AbaqusBuildError(
                (
                    f"line material {material.name!r} must contain exactly "
                    "one *DENSITY data record when density is declared"
                ),
                code="abaqus.line.material.density_record_count",
                location=(
                    _record_location(
                        density_records,
                        prefer_second=len(density_records) > 1,
                    )
                    or material_location
                ),
                record=(material.name, len(density_records)),
                remediation="Provide one constant density scalar without a table.",
            )
        density = density_records[0]
        if int(density.field_count) != 1:
            raise AbaqusBuildError(
                (
                    f"line material {material.name!r} *DENSITY data must "
                    "contain exactly one scalar"
                ),
                code="abaqus.line.material.density_shape",
                location=density.location or material_location,
                record=density.fields,
                remediation=(
                    "Remove temperature/field columns and provide one "
                    "constant density."
                ),
            )


def _record_location(
    records: tuple[Any, ...],
    *,
    prefer_second: bool,
) -> AbaqusSourceLocation | None:
    if not records:
        return None
    index = 1 if prefer_second and len(records) > 1 else 0
    return getattr(records[index], "location", None)


def _material_keyword_location(
    deck: AbaqusDeck,
    material_name: str,
) -> AbaqusSourceLocation | None:
    for occurrence in deck.keyword_occurrences:
        if occurrence.name != "material":
            continue
        params = {
            str(key).casefold(): str(value)
            for key, value in dict(occurrence.params).items()
        }
        if params.get("name", "").casefold() == str(material_name).casefold():
            return occurrence.location
    return None


def _validate_declared_sections(model: FEMModel, deck: AbaqusDeck) -> Any:
    """Validate every declared assignment without requiring full coverage."""

    resolution = resolve_sections(model)
    if not resolution.issues:
        return resolution
    issue = resolution.issues[0]
    location = _section_location(deck, issue.assignment_index)
    locations: tuple[AbaqusSourceLocation, ...] = ()
    record: Any = (
        issue.element_set,
        issue.material,
        issue.section_type,
        issue.element_id,
    )
    if (
        str(issue.code).startswith("beam.orientation.")
        and issue.element_id is not None
    ):
        element = next(
            (
                candidate
                for candidate in model.mesh.elements
                if int(candidate.id) == int(issue.element_id)
            ),
            None,
        )
        if element is not None:
            locations = _beam_orientation_locations(
                deck,
                element,
                issue.assignment_index,
            )
            if locations:
                location = locations[0]
            record = {
                "element": int(issue.element_id),
                "nodes": tuple(int(value) for value in element.node_ids),
                "reference_source": "explicit",
            }
    raise AbaqusBuildError(
        issue.message,
        code=issue.code,
        location=location,
        record=record,
        locations=locations,
        remediation="Correct the referenced material and section data.",
    )


def _validate_line_model(
    model: FEMModel,
    deck: AbaqusDeck,
    resolution: Any,
) -> dict[int, Any] | None:
    """Validate canonical line sections and each source-defined B31 frame."""

    line_ids = {
        int(element.id)
        for element in model.mesh.elements
        if get_element_capabilities(element.type).family in {"beam", "truss"}
    }
    uncovered = tuple(
        element_id
        for element_id in resolution.uncovered_element_ids
        if element_id in line_ids
    )
    if uncovered:
        locations = tuple(
            location
            for element_id in uncovered
            if (
                location := _element_location(deck, element_id)
            ) is not None
        )
        raise AbaqusBuildError(
            f"line elements {uncovered} do not have a supported section",
            code="abaqus.line.section_missing",
            location=locations[0] if locations else None,
            locations=locations,
            record=uncovered,
            remediation="Assign every B31/T3D2 element through a non-empty ELSET.",
        )

    effective = {
        item.element_id: item
        for item in resolution.effective_assignments
    }
    beam_elements = [
        element
        for element in model.mesh.elements
        if get_element_capabilities(element.type).family == "beam"
    ]
    if not beam_elements:
        return None
    return _validate_b31_frames(model, deck, beam_elements, effective)


def _validate_b31_frames(
    model: FEMModel,
    deck: AbaqusDeck,
    beam_elements: list[Any],
    effective: dict[int, Any],
) -> dict[int, Any]:
    """Validate one tangent/reference frame for every B31 element.

    Frame validity is local to the element and its resolved section
    assignment.  Shared nodes therefore retain one global DOF identity while
    adjacent members may resolve different effective frames.
    """

    frames: dict[int, Any] = {}
    for element in beam_elements:
        element_id = int(element.id)
        assignment = effective[element_id]
        try:
            frames[element_id] = resolve_beam_frame_field(
                model.mesh,
                element,
                properties=assignment.effective_properties,
            )
        except (KeyError, TypeError, ValueError) as exc:
            locations = _beam_orientation_locations(
                deck,
                element,
                assignment.assignment_index,
            )
            raise AbaqusBuildError(
                f"B31 element {element_id} has invalid section orientation: {exc}",
                code=getattr(exc, "code", "abaqus.b31.orientation_invalid"),
                location=locations[0] if locations else None,
                locations=locations,
                record={
                    "element": element_id,
                    "nodes": tuple(int(value) for value in element.node_ids),
                    "reference": getattr(
                        exc,
                        "reference",
                        assignment.effective_properties.get(
                            BEAM_LOCAL_Y_REFERENCE_KEY
                        ),
                    ),
                    "reference_source": (
                        "explicit"
                        if BEAM_LOCAL_Y_REFERENCE_KEY
                        in assignment.effective_properties
                        else "default"
                    ),
                },
                remediation=(
                    "Provide a finite nonzero n1 that is not parallel to every "
                    "B31 element tangent in the target ELSET."
                ),
            ) from exc
    return frames


def _build_import_notices(
    deck: AbaqusDeck,
    orientation_report: AbaqusOrientationResolutionReport | None = None,
) -> tuple[AbaqusImportNotice, ...]:
    locations: list[AbaqusSourceLocation] = []
    seen: set[tuple[object, ...]] = set()
    for element in deck.elements:
        if str(element.type).upper() != "B31":
            continue
        location = element.keyword_location or element.data_location
        if location is None:
            continue
        identity = (location.path, location.line, location.keyword)
        if identity in seen:
            continue
        seen.add(identity)
        locations.append(location)
    if not any(str(element.type).upper() == "B31" for element in deck.elements):
        return ()
    notices: list[AbaqusImportNotice] = [
        AbaqusImportNotice(
            code="abaqus.b31.linear_timoshenko_support_boundary",
            message=(
                "The supported Abaqus B31 input subset maps to the solver's "
                "linear, small-deformation, static Timoshenko Beam2 "
                "formulation, including transverse shear deformation. This "
                "does not claim full Abaqus B31 numerical equivalence: "
                "nonlinear and dynamic or mass behavior, Abaqus "
                "element-length slenderness compensation, user-defined "
                "transverse shear stiffness, warping or a seventh degree of "
                "freedom, and PIPE or arbitrary sections remain unsupported."
            ),
            locations=tuple(locations),
        ),
    ]
    generated_shared_groups = bool(
        orientation_report
        and any(
            len(group.identities) > 1 or group.split_reason
            for group in orientation_report.groups
        )
    )
    if generated_shared_groups:
        notices.append(
            AbaqusImportNotice(
                code="abaqus.b31.nodal_normal_generation_approximation",
                message=(
                    "The source does not provide nodal-normal records. The "
                    "orientation resolver generated element-end normals from "
                    "the detached topology and preserved shared-node source "
                    "connectivity, but this Beam2 projection does not claim "
                    "numerical equivalence to Abaqus nodal-normal generation "
                    "or averaging."
                ),
                locations=tuple(locations),
            )
        )
    return tuple(notices)


def _section_location(
    deck: AbaqusDeck,
    assignment_index: int | None,
) -> AbaqusSourceLocation | None:
    if assignment_index is None:
        return None
    if not 0 <= assignment_index < len(deck.sections):
        return None
    return deck.sections[assignment_index].keyword_location


def _element_location(
    deck: AbaqusDeck,
    element_id: int,
) -> AbaqusSourceLocation | None:
    for element in deck.elements:
        if int(element.id) == int(element_id):
            return element.data_location or element.keyword_location
    return None


def _node_location(
    deck: AbaqusDeck,
    node_id: int,
) -> AbaqusSourceLocation | None:
    record = deck.node_records.get(int(node_id))
    if record is None:
        return None
    return record.location or record.keyword_location


def _section_orientation_location(
    deck: AbaqusDeck,
    assignment_index: int | None,
) -> AbaqusSourceLocation | None:
    if assignment_index is None:
        return None
    if not 0 <= int(assignment_index) < len(deck.sections):
        return None
    section = deck.sections[int(assignment_index)]
    data = section.data
    if isinstance(data, AbaqusBeamSectionData):
        if data.orientation.present and data.orientation.location is not None:
            return data.orientation.location
    return section.keyword_location


def _beam_orientation_locations(
    deck: AbaqusDeck,
    element: Any,
    assignment_index: int | None,
) -> tuple[AbaqusSourceLocation, ...]:
    """Order n1/section, element, and endpoint evidence for diagnostics."""

    candidates = [
        _section_orientation_location(deck, assignment_index),
        _element_location(deck, int(element.id)),
        *(
            _node_location(deck, int(node_id))
            for node_id in element.node_ids
        ),
    ]
    return _unique_source_locations(candidates)


def _unique_source_locations(
    candidates: Any,
) -> tuple[AbaqusSourceLocation, ...]:
    result: list[AbaqusSourceLocation] = []
    identities: set[tuple[object, ...]] = set()
    for location in candidates:
        if location is None:
            continue
        identity = (location.path, location.line, location.keyword)
        if identity in identities:
            continue
        identities.add(identity)
        result.append(location)
    return tuple(result)


def _build_mesh(deck: AbaqusDeck) -> Any:
    """Build a mesh from deck nodes and elements."""
    if not deck.nodes:
        raise ValueError("Abaqus deck has no nodes")
    if not deck.elements:
        raise ValueError("Abaqus deck has no elements")

    orientation_only_nodes = _orientation_only_node_ids(deck)
    mesh_nodes = {
        node_id: coordinates
        for node_id, coordinates in deck.nodes.items()
        if node_id not in orientation_only_nodes
    }
    dimension = _mesh_dimension(deck.elements)
    if dimension == 2:
        nodes2d = [
            Node2D(node_id, coords[0], coords[1])
            for node_id, coords in sorted(mesh_nodes.items())
        ]
        elements2d = [
            Element2D(
                element.id,
                list(element.structural_node_ids),
                _element_type(element),
                _element_props(element),
            )
            for element in deck.elements
        ]
        return Mesh2D(nodes2d, elements2d)

    nodes3d = [
        Node3D(node_id, coords[0], coords[1], coords[2])
        for node_id, coords in sorted(mesh_nodes.items())
    ]
    elements3d = [
        Element3D(
            element.id,
            list(element.structural_node_ids),
            _element_type(element),
            _element_props(element),
        )
        for element in deck.elements
    ]
    element_types = {element.type for element in elements3d}
    if element_types == {"Beam2"}:
        dofs_per_node = 6
    elif element_types == {"Truss2"}:
        dofs_per_node = 3
    elif element_types.intersection({"Beam2", "Truss2"}):
        raise ValueError("mixed line and continuum element meshes are not supported")
    else:
        dofs_per_node = 3
    return Mesh3D(nodes3d, elements3d, dofs_per_node=dofs_per_node)


def _orientation_only_node_ids(deck: AbaqusDeck) -> set[int]:
    """Return additional-node IDs that are not structural connectivity IDs."""

    orientation_ids = {
        int(element.additional_orientation_node_id)
        for element in deck.elements
        if str(element.type).upper() == "B31"
        and element.additional_orientation_node_id is not None
    }
    structural_ids = {
        int(node_id)
        for element in deck.elements
        for node_id in element.structural_node_ids
    }
    return orientation_ids - structural_ids


def _build_sections(
    deck: AbaqusDeck,
    mesh: Any,
    element_sets: dict[str, ElementSet],
    internal_element_sets: dict[str, ElementSet],
) -> list[SectionAssignment]:
    """Map typed wire sections after resolving their target element family."""

    del element_sets  # The raw deck preserves scoped set identity and source order.
    element_lookup = {int(element.id): element for element in mesh.elements}
    assignments: list[SectionAssignment] = []
    for section_index, section in enumerate(deck.sections):
        section_element_set, element_ids = _section_target(
            deck,
            section,
            section_index,
            element_lookup,
            internal_element_sets,
        )
        family = _target_family(
            element_ids,
            element_lookup,
            location=section.keyword_location,
            subject=f"section target {section.element_set!r}",
        )
        data = section.data
        if isinstance(data, AbaqusSolidSectionData):
            section_type, properties = _map_solid_section(
                data,
                family,
                section,
            )
        elif isinstance(data, AbaqusBeamSectionData):
            section_type, properties = _map_beam_section(
                data,
                family,
                section,
            )
        else:
            raise AbaqusBuildError(
                (
                    f"section {section.element_set!r} has no supported typed "
                    "Abaqus section data"
                ),
                code="abaqus.section.data_missing",
                location=section.keyword_location,
                remediation=(
                    "Use *SOLID SECTION for T3D2/CPS/CPE/C3D elements or "
                    "*BEAM SECTION for B31 elements."
                ),
            )
        assignments.append(
            SectionAssignment(
                section_element_set,
                section.material,
                section_type,
                properties,
            )
        )
    return assignments


def _section_target(
    deck: AbaqusDeck,
    section: AbaqusSection,
    section_index: int,
    element_lookup: dict[int, Any],
    internal_element_sets: dict[str, ElementSet],
) -> tuple[str, tuple[int, ...]]:
    """Resolve a section set while preserving definition-time scoped IDs."""

    target_was_defined = getattr(section, "target_was_defined", None)
    captured_ids = _unique_ids(section.element_ids)
    if (
        target_was_defined is not True
        and section.element_set not in deck.element_sets
        and not captured_ids
    ):
        raise AbaqusBuildError(
            f"element set {section.element_set!r} is not defined",
            code="abaqus.section.target_missing",
            location=section.keyword_location,
            remediation="Define a non-empty ELSET before assigning the section.",
        )
    resolved_ids = _unique_ids(deck.element_sets.get(section.element_set, ()))
    if target_was_defined is True:
        element_ids = captured_ids
    elif target_was_defined is False:
        element_ids = captured_ids or resolved_ids
    else:
        element_ids = captured_ids or resolved_ids
    if not element_ids:
        raise AbaqusBuildError(
            f"element set {section.element_set!r} is empty",
            code="abaqus.section.target_empty",
            location=section.keyword_location,
            remediation="Assign the section to a non-empty element set.",
        )
    missing_ids = tuple(
        element_id
        for element_id in element_ids
        if element_id not in element_lookup
    )
    if missing_ids:
        raise AbaqusBuildError(
            (
                f"section target {section.element_set!r} references undefined "
                f"element IDs {missing_ids}"
            ),
            code="abaqus.section.element_missing",
            location=section.keyword_location,
            record=missing_ids,
            remediation="Remove undefined IDs or define the referenced elements.",
        )

    section_element_set = section.element_set
    if (
        target_was_defined is True
        and captured_ids != resolved_ids
    ) or (
        target_was_defined is None
        and captured_ids
        and captured_ids != resolved_ids
    ):
        section_element_set = f"_section_{section_index}_{section.element_set}"
        internal_element_sets[section_element_set] = ElementSet(
            section_element_set,
            captured_ids,
        )
    return section_element_set, element_ids


def _target_family(
    element_ids: tuple[int, ...],
    element_lookup: dict[int, Any],
    *,
    location: AbaqusSourceLocation | None,
    subject: str,
) -> str:
    """Return one canonical family for a resolved non-empty target."""

    families = {
        get_element_capabilities(element_lookup[element_id].type).family
        for element_id in element_ids
    }
    if len(families) != 1:
        raise AbaqusBuildError(
            f"{subject} mixes element families {tuple(sorted(families))}",
            code="abaqus.target.family_mixed",
            location=location,
            record=element_ids,
            remediation="Split the target into one element family per record.",
        )
    return next(iter(families))


def _map_solid_section(
    data: AbaqusSolidSectionData,
    family: str,
    section: AbaqusSection,
) -> tuple[str, dict[str, Any]]:
    """Interpret one overloaded SOLID SECTION record by target family."""

    location = data.location or section.keyword_location
    if family == "truss":
        if not data.record_present:
            raise AbaqusBuildError(
                "T3D2 *SOLID SECTION requires an area data record",
                code="abaqus.t3d2.area_record_missing",
                location=section.keyword_location,
                remediation=(
                    "Add one data line after *SOLID SECTION; use a blank first "
                    "field to request the Abaqus default area of 1.0."
                ),
            )
        if data.field_count != 1:
            raise AbaqusBuildError(
                "T3D2 *SOLID SECTION data must contain exactly one area field",
                code="abaqus.t3d2.area_shape",
                location=location,
                record=data.fields,
                remediation="Provide exactly one positive area value.",
            )
        area = 1.0 if data.attribute is None else float(data.attribute)
        _require_positive_finite(
            area,
            "T3D2 section area",
            location=location,
            code="abaqus.t3d2.area_invalid",
        )
        return "truss", {"area": area}

    if family == "plane_continuum":
        if data.field_count > 1:
            raise AbaqusBuildError(
                "*SOLID SECTION data must contain at most one thickness field",
                code="abaqus.solid_section.thickness_shape",
                location=location,
                record=data.fields,
            )
        if data.attribute is None:
            return "solid", {}
        thickness = float(data.attribute)
        _require_positive_finite(
            thickness,
            "*SOLID SECTION thickness",
            location=location,
            code="abaqus.solid_section.thickness_invalid",
        )
        return "solid", {"thickness": thickness}

    if family == "solid_continuum":
        if data.attribute is not None or any(
            str(value).strip() for value in data.fields
        ):
            raise AbaqusBuildError(
                (
                    "Abaqus SOLID SECTION thickness data is supported only "
                    "for two-dimensional CPS/CPE elements"
                ),
                code="abaqus.solid_section.3d_attribute_unsupported",
                location=location,
                record=data.fields,
                remediation=(
                    "Remove the geometry data record for a homogeneous C3D "
                    "section in the supported subset."
                ),
            )
        return "solid", {}

    raise AbaqusBuildError(
        f"*SOLID SECTION cannot target {family} elements",
        code="abaqus.section.target_incompatible",
        location=section.keyword_location,
        remediation="Use *BEAM SECTION for B31 elements.",
    )


def _map_beam_section(
    data: AbaqusBeamSectionData,
    family: str,
    section: AbaqusSection,
) -> tuple[str, dict[str, Any]]:
    """Map supported B31 profile geometry and approximate n1."""

    if family != "beam":
        raise AbaqusBuildError(
            f"*BEAM SECTION cannot target {family} elements",
            code="abaqus.section.target_incompatible",
            location=section.keyword_location,
            remediation="Use *BEAM SECTION only with a non-empty B31 ELSET.",
        )

    profile = " ".join(data.profile.upper().split())
    dimensions = tuple(float(value) for value in data.dimensions)
    geometry_location = data.geometry.location or section.keyword_location
    if profile == "RECT":
        _require_dimension_count(
            profile,
            dimensions,
            2,
            location=geometry_location,
        )
        height, width = dimensions
        properties: dict[str, Any] = {
            "height": height,
            "width": width,
        }
        section_type = "rectangle"
    elif profile == "CIRC":
        _require_dimension_count(
            profile,
            dimensions,
            1,
            location=geometry_location,
        )
        properties = {"radius": dimensions[0]}
        section_type = "solid_circle"
    elif profile == "THICK PIPE":
        _require_dimension_count(
            profile,
            dimensions,
            2,
            location=geometry_location,
        )
        outer_radius, thickness = dimensions
        _require_positive_finite(
            outer_radius,
            "THICK PIPE outer radius",
            location=geometry_location,
            code="abaqus.b31.section.dimension_invalid",
        )
        _require_positive_finite(
            thickness,
            "THICK PIPE wall thickness",
            location=geometry_location,
            code="abaqus.b31.section.dimension_invalid",
        )
        inner_radius = outer_radius - thickness
        if not math.isfinite(inner_radius) or inner_radius <= 0.0:
            raise AbaqusBuildError(
                "THICK PIPE wall thickness must be less than the outer radius",
                code="abaqus.b31.thick_pipe.invalid",
                location=geometry_location,
                record=dimensions,
                remediation="Use finite values satisfying 0 < thickness < radius.",
            )
        properties = {
            "outer_radius": outer_radius,
            "inner_radius": inner_radius,
        }
        section_type = "hollow_circle"
    else:
        if profile == "PIPE":
            remediation = (
                "Use SECTION=THICK PIPE for a thick annulus; thin-wall PIPE "
                "requires a dedicated section contract."
            )
        elif profile in {"RECTANGLE", "SOLID_CIRCLE", "HOLLOW_CIRCLE"}:
            remediation = "Use the standard RECT, CIRC, or THICK PIPE profile."
        else:
            remediation = (
                "Use one of the supported standard profiles: RECT, CIRC, "
                "or THICK PIPE."
            )
        raise UnsupportedAbaqusFeatureError(
            f"unsupported B31 beam profile {data.profile!r}",
            code="abaqus.b31.section_profile_unsupported",
            location=section.keyword_location,
            record=data.profile,
            remediation=remediation,
        )

    for name, value in tuple(properties.items()):
        _require_positive_finite(
            float(value),
            f"B31 {profile} {name}",
            location=geometry_location,
            code="abaqus.b31.section.dimension_invalid",
        )
    if data.approximate_n1 is not None:
        properties[BEAM_LOCAL_Y_REFERENCE_KEY] = tuple(
            float(value) for value in data.approximate_n1
        )
    return section_type, properties


def _require_dimension_count(
    profile: str,
    dimensions: tuple[float, ...],
    expected: int,
    *,
    location: AbaqusSourceLocation | None,
) -> None:
    if len(dimensions) != expected:
        raise AbaqusBuildError(
            (
                f"B31 SECTION={profile} requires {expected} geometry "
                f"value{'s' if expected != 1 else ''}, got {len(dimensions)}"
            ),
            code="abaqus.b31.section.geometry_shape",
            location=location,
            record=dimensions,
        )


def _require_positive_finite(
    value: float,
    label: str,
    *,
    location: AbaqusSourceLocation | None,
    code: str,
) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise AbaqusBuildError(
            f"{label} must be finite and greater than zero",
            code=code,
            location=location,
            record=value,
        )


def _build_surfaces_and_edges(
    mesh: Any,
    deck: AbaqusDeck,
    element_sets: dict[str, ElementSet],
    topology: dict[str, Any],
) -> tuple[dict[str, Surface], dict[str, Edge]]:
    """Build named model surfaces and edges from deck surface entries."""
    if mesh.dofs_per_node == 2:
        return {}, _build_edges(mesh, deck, element_sets, topology)
    return _build_surfaces(mesh, deck, element_sets, topology), {}


def _build_surfaces(
    mesh: Any,
    deck: AbaqusDeck,
    element_sets: dict[str, ElementSet],
    topology: dict[str, Any],
) -> dict[str, Surface]:
    """Build named model surfaces from deck surface entries."""
    if not any(
        deck.surface_scopes.get(name, "model") != "part"
        for name in deck.surfaces
    ):
        return {}
    face_lookup = _face_lookup(mesh, topology)
    elem_lookup = topology["elements"]
    surfaces: dict[str, Surface] = {}

    for name, entries in deck.surfaces.items():
        if deck.surface_scopes.get(name, "model") == "part":
            continue
        model_faces: list[ElementFace] = []
        for entry in entries:
            for element_id in _resolve_element_target(entry.target, element_sets):
                elem = _require_mesh_element(elem_lookup, element_id)
                local_index = _face_label_to_index(entry.face_label, elem.type)
                node_ids = face_lookup.get((element_id, local_index))
                if node_ids is None:
                    raise ValueError(
                        f"element {element_id} does not have Abaqus face {entry.face_label}"
                    )
                model_faces.append(ElementFace(element_id, local_index, node_ids))
        surfaces[name] = Surface(name, model_faces)

    return surfaces


def _build_edges(
    mesh: Any,
    deck: AbaqusDeck,
    element_sets: dict[str, ElementSet],
    topology: dict[str, Any],
) -> dict[str, Edge]:
    """Build named model edges from 2D Abaqus surface entries."""
    if not any(
        deck.surface_scopes.get(name, "model") != "part"
        for name in deck.surfaces
    ):
        return {}
    edge_lookup = _edge_lookup(mesh, topology)
    elem_lookup = topology["elements"]
    edges: dict[str, Edge] = {}

    for name, entries in deck.surfaces.items():
        if deck.surface_scopes.get(name, "model") == "part":
            continue
        model_edges: list[ElementEdge] = []
        for entry in entries:
            for element_id in _resolve_element_target(entry.target, element_sets):
                elem = _require_mesh_element(elem_lookup, element_id)
                local_index = _face_label_to_index(entry.face_label, elem.type)
                node_ids = edge_lookup.get((element_id, local_index))
                if node_ids is None:
                    raise ValueError(
                        f"element {element_id} does not have Abaqus edge {entry.face_label}"
                    )
                model_edges.append(ElementEdge(element_id, local_index, node_ids))
        edges[name] = Edge(name, model_edges)

    return edges


def _build_step(
    step: AbaqusStep,
    mesh: Any,
    surfaces: dict[str, Surface],
    edges: dict[str, Edge],
    element_sets: dict[str, ElementSet],
    step_index: int,
    topology: dict[str, Any],
) -> AnalysisStep:
    """Convert raw Abaqus step data to core step data."""
    boundaries: list[DisplacementConstraint] = []
    for boundary in step.boundaries:
        for first, last, value in _constraint_ranges(boundary, mesh.dofs_per_node):
            boundaries.append(
                DisplacementConstraint(boundary.target, first, last, value)
            )

    cloads = [
        NodalLoad(load.target, load.component, load.value)
        for load in step.cloads
    ]
    distributed_loads = [
        _build_distributed_load(
            load,
            mesh,
            surfaces,
            edges,
            element_sets,
            step.name,
            step_index,
            load_index,
            topology,
        )
        for load_index, load in enumerate(step.distributed_loads)
    ]
    surface_loads = [
        load for load in distributed_loads
        if isinstance(load, SurfaceLoad)
    ]
    edge_loads = [
        load for load in distributed_loads
        if isinstance(load, EdgeLoad)
    ]
    line_loads = [
        load for load in distributed_loads
        if isinstance(load, LineLoad)
    ]
    gravity_loads = [
        load for load in distributed_loads
        if isinstance(load, GravityLoad)
    ]
    outputs = [
        OutputRequest(
            output.kind,
            output.target,
            output.variables,
            output.metadata,
            OutputSourceEvidence(
                "abaqus",
                output.parent_parameters,
                output.parent_flags,
                output.child_parameters,
                output.child_flags,
            ),
        )
        for output in step.output_requests
    ]
    return AnalysisStep(
        step.name,
        procedure=step.procedure,
        boundaries=boundaries,
        cloads=cloads,
        surface_loads=surface_loads,
        edge_loads=edge_loads,
        line_loads=line_loads,
        gravity_loads=gravity_loads,
        outputs=outputs,
        metadata=dict(step.metadata),
    )


def _build_distributed_load(
    load: AbaqusDistributedLoad,
    mesh: Any,
    surfaces: dict[str, Surface],
    edges: dict[str, Edge],
    element_sets: dict[str, ElementSet],
    step_name: str,
    step_index: int,
    load_index: int,
    topology: dict[str, Any],
) -> SurfaceLoad | EdgeLoad | LineLoad | GravityLoad:
    """Convert an Abaqus DLOAD/DSLOAD line to a model distributed load."""
    label = load.label.upper()
    if label == "GRAV":
        return _build_gravity_load(load, mesh)
    if load.source == "dload":
        target_family = _distributed_load_target_family(
            load,
            element_sets,
            topology,
        )
        if target_family in {"beam", "truss"}:
            return _build_line_load(load, target_family)
    else:
        mesh_families = {
            get_element_capabilities(element.type).family
            for element in mesh.elements
        }
        if mesh_families & {"beam", "truss"}:
            raise UnsupportedAbaqusFeatureError(
                "B31/T3D2 distributed loads must use *DLOAD",
                code="abaqus.line.dsload_unsupported",
                location=load.location,
                record=(load.target, load.label),
                remediation="Use a supported B31 *DLOAD label or GRAV.",
            )
    if load.source == "dsload":
        target_name = str(load.target)
        if mesh.dofs_per_node == 2:
            if target_name not in edges:
                raise KeyError(f"edge {target_name} is not defined")
        elif target_name not in surfaces:
            raise KeyError(f"surface {target_name} is not defined")
    elif load.source == "dload":
        face_label = _dload_face_label(label)
        target_name = _generated_surface_name(step_name, step_index, load_index)
        if mesh.dofs_per_node == 2:
            edges[target_name] = _edge_from_element_target(
                mesh,
                target_name,
                load.target,
                face_label,
                element_sets,
                topology,
            )
        else:
            surfaces[target_name] = _surface_from_element_target(
                mesh,
                target_name,
                load.target,
                face_label,
                element_sets,
                topology,
            )
    else:
        raise ValueError(f"unsupported distributed load source: {load.source}")

    if mesh.dofs_per_node == 2:
        if label == "P" or label.startswith("P"):
            return EdgeLoad(target_name, magnitude=load.magnitude, load_type="pressure")
        if label == "TRVEC":
            return EdgeLoad(target_name, _scaled_traction_vector(load, mesh), load_type="traction")
        raise ValueError(f"unsupported Abaqus 2D distributed load label: {load.label}")

    if label == "P" or label.startswith("P"):
        return SurfaceLoad(target_name, magnitude=load.magnitude, load_type="pressure")
    if label == "TRVEC":
        return SurfaceLoad(target_name, _scaled_traction_vector(load, mesh), load_type="traction")
    if label == "TRSHR":
        return SurfaceLoad(
            target_name,
            _traction_direction(load, mesh, "TRSHR"),
            magnitude=load.magnitude,
            load_type="shear_traction",
        )
    raise ValueError(f"unsupported Abaqus distributed load label: {load.label}")


def _distributed_load_target_family(
    load: AbaqusDistributedLoad,
    element_sets: dict[str, ElementSet],
    topology: dict[str, Any],
) -> str:
    """Resolve a DLOAD target before interpreting overloaded label tokens."""

    if load.target is None:
        raise AbaqusBuildError(
            f"*DLOAD {load.label} requires an element target",
            code="abaqus.dload.target_missing",
            location=load.location,
            record=(load.target, load.label),
        )
    try:
        element_ids = _resolve_element_target(load.target, element_sets)
    except KeyError as exc:
        raise AbaqusBuildError(
            str(exc),
            code="abaqus.dload.target_undefined",
            location=load.location,
            record=load.target,
        ) from exc
    if not element_ids:
        raise AbaqusBuildError(
            f"*DLOAD target {load.target!r} is empty",
            code="abaqus.dload.target_empty",
            location=load.location,
            record=load.target,
        )
    element_lookup = topology["elements"]
    missing = tuple(
        element_id
        for element_id in element_ids
        if element_id not in element_lookup
    )
    if missing:
        raise AbaqusBuildError(
            f"*DLOAD target references undefined element IDs {missing}",
            code="abaqus.dload.element_missing",
            location=load.location,
            record=missing,
        )
    return _target_family(
        tuple(int(value) for value in element_ids),
        element_lookup,
        location=load.location,
        subject=f"*DLOAD target {load.target!r}",
    )


def _build_line_load(
    load: AbaqusDistributedLoad,
    target_family: str,
) -> LineLoad:
    """Map one constant standard B31 line-load record."""

    label = load.label.upper()
    if target_family == "truss":
        raise UnsupportedAbaqusFeatureError(
            f"T3D2 does not support non-gravity distributed load {label}",
            code="abaqus.t3d2.line_load_unsupported",
            location=load.location,
            record=(load.target, label, load.magnitude, *load.extra),
            remediation="Convert the resultant to nodal CLOAD records or remodel it.",
        )
    supported_labels = {"PX", "PY", "PZ", "P1", "P2"}
    if label not in supported_labels:
        if label in {"QGLOBAL", "QLOCAL"}:
            remediation = (
                "Replace QGLOBAL with scalar PX/PY/PZ records. Replace a "
                "transverse QLOCAL with P1/P2 records; a local-x component "
                "has no Phase 6 standard equivalent."
            )
        else:
            remediation = "Use one of PX, PY, PZ, P1, or P2."
        raise UnsupportedAbaqusFeatureError(
            f"unsupported B31 *DLOAD label {label!r}",
            code="abaqus.b31.dload_label_unsupported",
            location=load.location,
            record=label,
            remediation=remediation,
        )
    if load.keyword_params or load.keyword_flags:
        raise UnsupportedAbaqusFeatureError(
            "*DLOAD options are not supported for B31 constant line loads",
            code="abaqus.b31.dload_option_unsupported",
            location=load.keyword_location or load.location,
            record={
                "params": dict(load.keyword_params),
                "flags": tuple(load.keyword_flags),
            },
            remediation=(
                "Remove FOLLOWER and other options; the supported subset uses "
                "the undeformed frame in a linear-static step."
            ),
        )
    if load.extra:
        raise AbaqusBuildError(
            f"B31 {label} requires exactly one magnitude value",
            code="abaqus.b31.dload_shape",
            location=load.location,
            record=(load.target, label, load.magnitude, *load.extra),
        )
    magnitude = float(load.magnitude)
    if not math.isfinite(magnitude):
        raise AbaqusBuildError(
            f"B31 {label} magnitude must be finite",
            code="abaqus.b31.dload_magnitude_invalid",
            location=load.location,
            record=load.magnitude,
        )
    mapping = {
        "PX": ((magnitude, 0.0, 0.0), "global"),
        "PY": ((0.0, magnitude, 0.0), "global"),
        "PZ": ((0.0, 0.0, magnitude), "global"),
        "P1": ((0.0, magnitude, 0.0), "local"),
        "P2": ((0.0, 0.0, magnitude), "local"),
    }
    vector, coordinate_system = mapping[label]
    return LineLoad(load.target, vector, coordinate_system)


def _build_gravity_load(
    load: AbaqusDistributedLoad,
    mesh: Any,
) -> GravityLoad:
    """Convert one Abaqus DLOAD GRAV record to an acceleration load."""
    if str(load.source).casefold() != "dload":
        raise ValueError("DSLOAD GRAV is not supported; use DLOAD for gravity")

    try:
        magnitude = float(load.magnitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("GRAV magnitude must be a finite number") from exc
    if not math.isfinite(magnitude):
        raise ValueError(f"GRAV magnitude must be finite, got {load.magnitude!r}")

    if len(load.extra) != 3:
        raise ValueError(
            "GRAV requires 3 direction components, "
            f"got {len(load.extra)}"
        )
    try:
        direction = tuple(float(value) for value in load.extra)
    except (TypeError, ValueError) as exc:
        raise ValueError("GRAV direction components must be finite numbers") from exc
    if not all(math.isfinite(value) for value in direction):
        raise ValueError("GRAV direction components must be finite numbers")

    scale = max(abs(value) for value in direction)
    if scale == 0.0:
        raise ValueError("GRAV direction vector must be nonzero")
    scaled_direction = tuple(value / scale for value in direction)
    scaled_norm = math.sqrt(sum(value * value for value in scaled_direction))
    unit_direction = tuple(value / scaled_norm for value in scaled_direction)
    acceleration = tuple(magnitude * value for value in unit_direction)

    dim = 3 if mesh.nodes and hasattr(mesh.nodes[0], "z") else 2
    if dim == 2:
        if acceleration[2] != 0.0:
            raise ValueError(
                "GRAV out-of-plane acceleration must be zero for a 2D model, "
                f"got {acceleration[2]}"
            )
        acceleration = acceleration[:2]
    return GravityLoad(acceleration, target=load.target)


def _surface_from_element_target(
    mesh: Any,
    name: str,
    target: str | int,
    face_label: str,
    element_sets: dict[str, ElementSet],
    topology: dict[str, Any],
) -> Surface:
    """Build a generated surface from an element target and face label."""
    face_lookup = _face_lookup(mesh, topology)
    elem_lookup = topology["elements"]
    model_faces = []
    for element_id in _resolve_element_target(target, element_sets):
        elem = _require_mesh_element(elem_lookup, element_id)
        local_index = _face_label_to_index(face_label, elem.type)
        node_ids = face_lookup.get((element_id, local_index))
        if node_ids is None:
            raise ValueError(f"element {element_id} does not have Abaqus face {face_label}")
        model_faces.append(ElementFace(element_id, local_index, node_ids))
    return Surface(name, model_faces)


def _edge_from_element_target(
    mesh: Any,
    name: str,
    target: str | int,
    face_label: str,
    element_sets: dict[str, ElementSet],
    topology: dict[str, Any],
) -> Edge:
    """Build a generated edge collection from a 2D element target and face label."""
    edge_lookup = _edge_lookup(mesh, topology)
    elem_lookup = topology["elements"]
    model_edges = []
    for element_id in _resolve_element_target(target, element_sets):
        elem = _require_mesh_element(elem_lookup, element_id)
        local_index = _face_label_to_index(face_label, elem.type)
        node_ids = edge_lookup.get((element_id, local_index))
        if node_ids is None:
            raise ValueError(f"element {element_id} does not have Abaqus edge {face_label}")
        model_edges.append(ElementEdge(element_id, local_index, node_ids))
    return Edge(name, model_edges)


def _face_lookup(mesh: Any, topology: dict[str, Any]) -> dict[tuple[int, int], tuple[int, ...]]:
    lookup = topology.get("faces")
    if lookup is None:
        lookup = {
            (elem_id, face_index): node_ids
            for elem_id, face_index, node_ids in face_selection.all(mesh)
        }
        topology["faces"] = lookup
    return lookup


def _edge_lookup(mesh: Any, topology: dict[str, Any]) -> dict[tuple[int, int], tuple[int, ...]]:
    lookup = topology.get("edges")
    if lookup is None:
        lookup = {
            (elem_id, edge_index): node_ids
            for elem_id, edge_index, node_ids in edge_selection.all(mesh)
        }
        topology["edges"] = lookup
    return lookup


def _scaled_traction_vector(load: AbaqusDistributedLoad, mesh: Any) -> tuple[float, ...]:
    """Return TRVEC magnitude multiplied by its direction vector."""
    direction = _traction_direction(load, mesh, "TRVEC")
    return tuple(float(load.magnitude * value) for value in direction)


def _traction_direction(
    load: AbaqusDistributedLoad,
    mesh: Any,
    label: str,
) -> tuple[float, ...]:
    """Return a normalized Abaqus traction direction vector."""
    dim = 3 if mesh.nodes and hasattr(mesh.nodes[0], "z") else 2
    if len(load.extra) != dim:
        raise ValueError(
            f"{label} requires {dim} direction components, got {len(load.extra)}"
        )
    norm = sum(float(value) ** 2 for value in load.extra) ** 0.5
    if norm <= 0.0:
        raise ValueError(f"{label} direction vector must be nonzero")
    return tuple(float(value) / norm for value in load.extra)


def _mesh_dimension(elements: list[AbaqusElement]) -> int:
    """Infer mesh dimension from Abaqus element types."""
    dimensions = {_element_dimension(element.type) for element in elements}
    if len(dimensions) != 1:
        raise ValueError(f"mixed mesh dimensions are not supported: {dimensions}")
    return dimensions.pop()


def _element_dimension(element_type: str) -> int:
    """Return spatial dimension for an Abaqus element type."""
    etype = element_type.upper()
    if etype.startswith(("CPS", "CPE")):
        return 2
    if etype.startswith("C3D"):
        return 3
    if etype in {"T3D2", "B31"}:
        return 3
    if etype in {"TRUSS2", "BEAM2"}:
        remediation = "use T3D2" if etype == "TRUSS2" else "use B31"
        raise UnsupportedAbaqusFeatureError(
            f"retired Abaqus wire element alias {element_type!r}; {remediation}",
            code="abaqus.element_alias_retired",
            remediation=remediation,
        )
    raise ValueError(f"unsupported Abaqus element type: {element_type}")


def _element_type(element: AbaqusElement) -> str:
    """Map Abaqus element type to local element type."""
    aliases = {
        "T3D2": "Truss2",
        "B31": "Beam2",
    }
    mapped = aliases.get(element.type.upper())
    if mapped is not None:
        return mapped
    try:
        return canonical_element_type(element.type)
    except NotImplementedError as exc:
        raise ValueError(f"unsupported Abaqus element type: {element.type}") from exc


def _element_props(element: AbaqusElement) -> dict[str, Any]:
    """Return base properties for one mesh element."""
    props: dict[str, Any] = {"abaqus_type": element.type}
    if str(element.type).upper() == "B31":
        props[BEAM_DEFAULT_LOCAL_Y_REFERENCE_KEY] = (
            *BEAM_DEFAULT_LOCAL_Y_REFERENCE,
        )
    if element.element_set is not None:
        props["element_set"] = element.element_set
    if element.type.upper().startswith("CPS"):
        props["plane_type"] = "stress"
        props["thickness"] = 1.0
    elif element.type.upper().startswith("CPE"):
        props["plane_type"] = "strain"
        props["thickness"] = 1.0
    return props


def _constraint_ranges(
    boundary: AbaqusBoundary,
    dofs_per_node: int,
) -> list[tuple[int, int, float]]:
    """Return 1-based component ranges for a boundary line."""
    first = boundary.first_component
    if isinstance(first, str):
        label = first.upper()
        if label == "ENCASTRE":
            return [(1, dofs_per_node, 0.0)]
        if label == "XSYMM":
            return [(1, 1, 0.0)]
        if label == "YSYMM":
            return [(2, 2, 0.0)]
        if label == "ZSYMM":
            return [(3, 3, 0.0)]
        raise ValueError(f"unsupported Abaqus boundary label: {label}")

    last = boundary.last_component if boundary.last_component is not None else first
    return [(int(first), int(last), float(boundary.value))]


def _resolve_element_target(
    target: str | int,
    element_sets: dict[str, ElementSet],
) -> tuple[int, ...]:
    """Resolve an element id or element set name."""
    if isinstance(target, int):
        return (target,)
    if target not in element_sets:
        raise KeyError(f"element set {target} is not defined")
    return element_sets[target].element_ids


def _face_label_to_index(face_label: str, element_type: str | None = None) -> int:
    """Convert Abaqus S1-style labels to the project's local face index."""
    label = face_label.strip().upper()
    if not label.startswith("S"):
        raise ValueError(f"unsupported Abaqus face label: {face_label}")
    face_number = int(label[1:])
    if element_type is None:
        return face_number - 1

    etype = element_type.upper()
    if "TET" in etype or etype.startswith("C3D4") or etype.startswith("C3D10"):
        return _mapped_face_index(face_number, {1: 3, 2: 2, 3: 0, 4: 1}, face_label, element_type)
    if "HEX8" in etype or "HEX20" in etype or etype.startswith(("C3D8", "C3D20")):
        return _mapped_face_index(
            face_number,
            {1: 0, 2: 1, 3: 2, 4: 5, 5: 3, 6: 4},
            face_label,
            element_type,
        )
    return face_number - 1


def _mapped_face_index(
    face_number: int,
    mapping: dict[int, int],
    face_label: str,
    element_type: str,
) -> int:
    """Return a mapped local face index or raise a descriptive error."""
    if face_number not in mapping:
        raise ValueError(f"element type {element_type} does not have Abaqus face {face_label}")
    return mapping[face_number]


def _require_mesh_element(elem_lookup: dict[int, Any], element_id: int) -> Any:
    """Return a mesh element by id."""
    elem = elem_lookup.get(element_id)
    if elem is None:
        raise KeyError(f"element {element_id} is not defined")
    return elem


def _dload_face_label(load_label: str) -> str:
    """Convert Abaqus P1-style element pressure labels to S1-style faces."""
    label = load_label.strip().upper()
    if label.startswith("P") and len(label) > 1:
        return "S" + label[1:]
    raise ValueError(f"DLOAD pressure must use a face label like P1, got {load_label}")


def _generated_surface_name(step_name: str, step_index: int, load_index: int) -> str:
    """Return a stable generated surface name for a DLOAD entry."""
    safe_step = "".join(ch if ch.isalnum() else "_" for ch in step_name)
    return f"__DLOAD_{step_index}_{safe_step}_{load_index}"


def _unique_ids(ids: Any) -> tuple[int, ...]:
    """Return ids without duplicates while preserving order."""
    result: list[int] = []
    seen: set[int] = set()
    for value in ids:
        value = int(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _is_internal_element_set(name: str) -> bool:
    """Return whether an Abaqus element set should stay out of public model sets."""
    return str(name).startswith("_")
