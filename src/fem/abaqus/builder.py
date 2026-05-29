from __future__ import annotations

from typing import Any

from ..core.mesh import Element2D, Element3D, HexMesh3D, Node2D, Node3D, PlaneMesh2D, TetMesh3D
from ..core.model import (
    AnalysisStep,
    DisplacementConstraint,
    ElementFace,
    ElementSet,
    FEMModel,
    MaterialDefinition,
    NodalLoad,
    NodeSet,
    OutputRequest,
    SectionAssignment,
    Surface,
    SurfaceLoad,
)
from ..selection import faces as face_selection
from .deck import AbaqusBoundary, AbaqusDeck, AbaqusDistributedLoad, AbaqusElement, AbaqusStep


def build_model(deck: AbaqusDeck) -> FEMModel:
    """Build a FEMModel from a parsed Abaqus input deck."""
    mesh = _build_mesh(deck)
    node_sets = {
        name: NodeSet(name, _unique_ids(ids))
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
    sections: list[SectionAssignment] = []
    for section_index, section in enumerate(deck.sections):
        section_element_set = section.element_set
        section_element_ids = _unique_ids(section.element_ids)
        resolved_element_ids = _unique_ids(deck.element_sets.get(section.element_set, ()))
        if section_element_ids and section_element_ids != resolved_element_ids:
            section_element_set = f"_section_{section_index}_{section.element_set}"
            internal_element_sets[section_element_set] = ElementSet(
                section_element_set,
                section_element_ids,
            )
        sections.append(
            SectionAssignment(
                section_element_set,
                section.material,
                section.section_type,
            )
        )

    surfaces = _build_surfaces(mesh, deck, element_sets)
    steps = [
        _build_step(step, mesh, surfaces, element_sets, step_index)
        for step_index, step in enumerate(deck.steps)
    ]
    model = FEMModel(
        mesh=mesh,
        name=deck.name,
        node_sets=node_sets,
        element_sets=visible_element_sets,
        surfaces=surfaces,
        materials=materials,
        sections=sections,
        steps=steps,
        metadata={"_abaqus_internal_element_sets": internal_element_sets},
    )
    return model


def _build_mesh(deck: AbaqusDeck) -> Any:
    """Build a mesh from deck nodes and elements."""
    if not deck.nodes:
        raise ValueError("Abaqus deck has no nodes")
    if not deck.elements:
        raise ValueError("Abaqus deck has no elements")

    dimension = _mesh_dimension(deck.elements)
    if dimension == 2:
        nodes2d = [
            Node2D(node_id, coords[0], coords[1])
            for node_id, coords in sorted(deck.nodes.items())
        ]
        elements2d = [
            Element2D(
                element.id,
                list(element.node_ids),
                _element_type(element),
                _element_props(element),
            )
            for element in deck.elements
        ]
        return PlaneMesh2D(nodes2d, elements2d)

    nodes3d = [
        Node3D(node_id, coords[0], coords[1], coords[2])
        for node_id, coords in sorted(deck.nodes.items())
    ]
    elements3d = [
        Element3D(
            element.id,
            list(element.node_ids),
            _element_type(element),
            _element_props(element),
        )
        for element in deck.elements
    ]
    if all("tet" in elem.type.lower() for elem in elements3d):
        return TetMesh3D(nodes3d, elements3d)
    return HexMesh3D(nodes3d, elements3d)


def _build_surfaces(
    mesh: Any,
    deck: AbaqusDeck,
    element_sets: dict[str, ElementSet],
) -> dict[str, Surface]:
    """Build named model surfaces from deck surface entries."""
    face_lookup = {
        (elem_id, local_index): node_ids
        for elem_id, local_index, node_ids in face_selection.all(mesh)
    }
    elem_lookup = {elem.id: elem for elem in mesh.elements}
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


def _build_step(
    step: AbaqusStep,
    mesh: Any,
    surfaces: dict[str, Surface],
    element_sets: dict[str, ElementSet],
    step_index: int,
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
    surface_loads = [
        _build_surface_load(load, mesh, surfaces, element_sets, step.name, step_index, load_index)
        for load_index, load in enumerate(step.distributed_loads)
    ]
    outputs = [
        OutputRequest(output.kind, output.target, output.variables, output.metadata)
        for output in step.output_requests
    ]
    return AnalysisStep(
        step.name,
        procedure=step.procedure,
        boundaries=boundaries,
        cloads=cloads,
        surface_loads=surface_loads,
        outputs=outputs,
        metadata=dict(step.metadata),
    )


def _build_surface_load(
    load: AbaqusDistributedLoad,
    mesh: Any,
    surfaces: dict[str, Surface],
    element_sets: dict[str, ElementSet],
    step_name: str,
    step_index: int,
    load_index: int,
) -> SurfaceLoad:
    """Convert an Abaqus DLOAD/DSLOAD line to a model surface load."""
    label = load.label.upper()
    if load.source == "dsload":
        surface_name = str(load.target)
        if surface_name not in surfaces:
            raise KeyError(f"surface {surface_name} is not defined")
    elif load.source == "dload":
        face_label = _dload_face_label(label)
        surface_name = _generated_surface_name(step_name, step_index, load_index)
        surfaces[surface_name] = _surface_from_element_target(
            mesh,
            surface_name,
            load.target,
            face_label,
            element_sets,
        )
    else:
        raise ValueError(f"unsupported distributed load source: {load.source}")

    if label == "P" or label.startswith("P"):
        return SurfaceLoad(surface_name, magnitude=load.magnitude, load_type="pressure")
    if label == "TRVEC":
        return SurfaceLoad(surface_name, _scaled_traction_vector(load, mesh), load_type="traction")
    if label == "TRSHR":
        return SurfaceLoad(
            surface_name,
            _traction_direction(load, mesh, "TRSHR"),
            magnitude=load.magnitude,
            load_type="shear_traction",
        )
    raise ValueError(f"unsupported Abaqus distributed load label: {load.label}")


def _surface_from_element_target(
    mesh: Any,
    name: str,
    target: str | int,
    face_label: str,
    element_sets: dict[str, ElementSet],
) -> Surface:
    """Build a generated surface from an element target and face label."""
    face_lookup = {
        (elem_id, face_index): node_ids
        for elem_id, face_index, node_ids in face_selection.all(mesh)
    }
    elem_lookup = {elem.id: elem for elem in mesh.elements}
    model_faces = []
    for element_id in _resolve_element_target(target, element_sets):
        elem = _require_mesh_element(elem_lookup, element_id)
        local_index = _face_label_to_index(face_label, elem.type)
        node_ids = face_lookup.get((element_id, local_index))
        if node_ids is None:
            raise ValueError(f"element {element_id} does not have Abaqus face {face_label}")
        model_faces.append(ElementFace(element_id, local_index, node_ids))
    return Surface(name, model_faces)


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
    raise ValueError(f"unsupported Abaqus element type: {element_type}")


def _element_type(element: AbaqusElement) -> str:
    """Map Abaqus element type to local element type."""
    etype = element.type.upper()
    if etype in ("CPS3", "CPE3"):
        return "Tri3Plane"
    if etype in ("CPS4", "CPE4"):
        return "Quad4Plane"
    if etype in ("CPS8", "CPE8"):
        return "Quad8Plane"
    if etype == "C3D4":
        return "Tet4"
    if etype == "C3D10":
        return "Tet10"
    if etype == "C3D8":
        return "Hex8"
    raise ValueError(f"unsupported Abaqus element type: {element.type}")


def _element_props(element: AbaqusElement) -> dict[str, Any]:
    """Return base properties for one mesh element."""
    props: dict[str, Any] = {"abaqus_type": element.type}
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
    if "HEX8" in etype or etype.startswith("C3D8"):
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
