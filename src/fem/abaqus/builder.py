from __future__ import annotations

import math
from typing import Any

from ..core.mesh import Element2D, Element3D, Mesh2D, Mesh3D, Node2D, Node3D
from ..core.model import (
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
    SectionAssignment,
    Surface,
    SurfaceLoad,
)
from ..elements import canonical_element_type
from ..selection import edges as edge_selection
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
                section.properties,
            )
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
        return Mesh2D(nodes2d, elements2d)

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
        OutputRequest(output.kind, output.target, output.variables, output.metadata)
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
    if mesh.elements and mesh.elements[0].type in {"Beam2", "Truss2"}:
        if load.source != "dload":
            raise ValueError("line loads must use *Dload")
        if label not in {"QGLOBAL", "QLOCAL"}:
            raise ValueError(
                "line-element *Dload label must be QGLOBAL or QLOCAL"
            )
        vector = (load.magnitude, *load.extra)
        if len(vector) != 3:
            raise ValueError(
                f"{label} requires three vector components, got {len(vector)}"
            )
        return LineLoad(
            load.target,
            vector,
            "global" if label == "QGLOBAL" else "local",
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
    if etype in {"TRUSS2", "T3D2", "BEAM2", "B31"}:
        return 3
    raise ValueError(f"unsupported Abaqus element type: {element_type}")


def _element_type(element: AbaqusElement) -> str:
    """Map Abaqus element type to local element type."""
    aliases = {
        "T3D2": "Truss2",
        "TRUSS2": "Truss2",
        "B31": "Beam2",
        "BEAM2": "Beam2",
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
