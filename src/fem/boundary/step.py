from __future__ import annotations

from typing import Any

import numpy as np

from ..core.model import AnalysisStep, ElementFace, SurfaceLoad
from .condition import BoundaryCondition


def get_step(model: Any, step: str | int | AnalysisStep | None = None) -> AnalysisStep | None:
    """Return a model step by name or index."""
    if step is None:
        for candidate in model.steps:
            if candidate.name.lower() != "initial":
                return candidate
        return model.steps[0] if model.steps else None
    if isinstance(step, AnalysisStep):
        return step
    if isinstance(step, int):
        return model.steps[step]
    for candidate in model.steps:
        if candidate.name == step:
            return candidate
    raise KeyError(f"analysis step {step} is not defined")


def boundary_for_step(model: Any, step: str | int | AnalysisStep | None = None) -> BoundaryCondition:
    """Build solver boundary data for one model step."""
    selected_step = get_step(model, step)
    if selected_step is None:
        return BoundaryCondition()

    boundary = BoundaryCondition()
    for constraint in _step_boundaries(model, selected_step):
        for node_id in _resolve_node_target(model, constraint.target):
            for component in range(
                constraint.first_component,
                constraint.last_component + 1,
            ):
                _validate_component(model, component)
                boundary.add_displacement(
                    node_id,
                    component - 1,
                    constraint.value,
                    model.mesh,
                )

    for load in selected_step.cloads:
        _validate_component(model, load.component)
        for node_id in _resolve_node_target(model, load.target):
            boundary.add_nodal_force(
                node_id,
                load.component - 1,
                load.value,
                model.mesh,
            )

    for surface_load in selected_step.surface_loads:
        if surface_load.surface not in model.surfaces:
            raise KeyError(f"surface {surface_load.surface} is not defined")
        for face in model.surfaces[surface_load.surface].faces:
            if surface_load.load_type == "pressure":
                vector = _pressure_vector(model, face, surface_load)
            elif surface_load.load_type == "traction":
                vector = surface_load.vector
            elif surface_load.load_type == "shear_traction":
                vector = _shear_traction_vector(model, face, surface_load)
            else:
                raise ValueError(f"unsupported surface load type: {surface_load.load_type}")
            boundary.add_surface_traction(face.elem_id, face.local_index, *vector)

    return boundary


def _resolve_node_target(model: Any, target: str | int) -> tuple[int, ...]:
    """Resolve a node id or named node set."""
    if isinstance(target, int):
        return (target,)
    if target not in model.node_sets:
        raise KeyError(f"node set {target} is not defined")
    return model.node_sets[target].node_ids


def _step_boundaries(model: Any, step: AnalysisStep) -> tuple:
    """Return initial boundaries inherited by the selected step."""
    initial = next(
        (candidate for candidate in model.steps if candidate.name.lower() == "initial"),
        None,
    )
    if initial is None or initial is step:
        return tuple(step.boundaries)
    return tuple(initial.boundaries) + tuple(step.boundaries)


def _pressure_vector(
    model: Any,
    face: ElementFace,
    surface_load: SurfaceLoad,
) -> tuple[float, ...]:
    """Return an inward pressure vector for one surface face or edge."""
    if surface_load.magnitude is None:
        raise ValueError("pressure surface load requires a magnitude")

    if model.mesh.dofs_per_node == 2:
        return _pressure_vector_2d(model, face, surface_load)
    return _pressure_vector_3d(model, face, surface_load)


def _pressure_vector_2d(
    model: Any,
    face: ElementFace,
    surface_load: SurfaceLoad,
) -> tuple[float, float]:
    """Return an inward pressure vector for one 2D element edge."""
    node_lookup = {node.id: node for node in model.mesh.nodes}
    if len(face.node_ids) < 2:
        raise ValueError(f"surface edge {face} must contain at least 2 nodes for pressure")

    first = node_lookup[face.node_ids[0]]
    last = node_lookup[face.node_ids[-1]]
    p0 = np.array([float(first.x), float(first.y)], dtype=float)
    p1 = np.array([float(last.x), float(last.y)], dtype=float)
    tangent = p1 - p0
    length = float(np.linalg.norm(tangent))
    if length <= 0.0:
        raise ValueError(f"surface edge {face} has zero length")

    normal = np.array([tangent[1], -tangent[0]], dtype=float) / length

    elem_lookup = {elem.id: elem for elem in model.mesh.elements}
    elem = elem_lookup.get(face.elem_id)
    if elem is None:
        raise KeyError(f"element {face.elem_id} is not defined")
    elem_coords = np.array(
        [
            [float(node_lookup[node_id].x), float(node_lookup[node_id].y)]
            for node_id in elem.node_ids
        ],
        dtype=float,
    )
    edge_center = 0.5 * (p0 + p1)
    elem_center = np.mean(elem_coords, axis=0)
    inward = elem_center - edge_center
    if float(np.dot(normal, inward)) < 0.0:
        normal = -normal

    vector = float(surface_load.magnitude) * normal
    return float(vector[0]), float(vector[1])


def _pressure_vector_3d(
    model: Any,
    face: ElementFace,
    surface_load: SurfaceLoad,
) -> tuple[float, ...]:
    """Return an inward pressure vector for one 3D surface face."""
    node_lookup = {node.id: node for node in model.mesh.nodes}
    coords = []
    for node_id in face.node_ids:
        node = node_lookup[node_id]
        coords.append([float(node.x), float(node.y), float(getattr(node, "z", 0.0))])
    if len(coords) < 3:
        raise ValueError(f"surface face {face} must contain at least 3 nodes for pressure")

    p0 = np.array(coords[0], dtype=float)
    p1 = np.array(coords[1], dtype=float)
    p2 = np.array(coords[2], dtype=float)
    normal = np.cross(p1 - p0, p2 - p0)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError(f"surface face {face} has zero normal")

    elem_lookup = {elem.id: elem for elem in model.mesh.elements}
    elem = elem_lookup.get(face.elem_id)
    if elem is None:
        raise KeyError(f"element {face.elem_id} is not defined")
    elem_coords = []
    for node_id in elem.node_ids:
        node = node_lookup[node_id]
        elem_coords.append([float(node.x), float(node.y), float(getattr(node, "z", 0.0))])
    face_center = np.mean(np.array(coords, dtype=float), axis=0)
    elem_center = np.mean(np.array(elem_coords, dtype=float), axis=0)
    inward = elem_center - face_center
    if float(np.dot(normal, inward)) < 0.0:
        normal = -normal

    return tuple(float(value) for value in surface_load.magnitude * normal / norm)


def _shear_traction_vector(
    model: Any,
    face: ElementFace,
    surface_load: SurfaceLoad,
) -> tuple[float, ...]:
    """Return a surface shear traction projected onto one face tangent plane."""
    if surface_load.magnitude is None:
        raise ValueError("shear traction surface load requires a magnitude")
    if not surface_load.vector:
        raise ValueError("shear traction surface load requires a direction vector")

    normal = _face_unit_normal(model, face)
    direction = np.array(surface_load.vector, dtype=float)
    if direction.shape[0] != normal.shape[0]:
        raise ValueError(
            f"shear traction direction must have {normal.shape[0]} components, "
            f"got {direction.shape[0]}"
        )
    tangent = direction - float(np.dot(direction, normal)) * normal
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-12:
        return tuple(0.0 for _ in range(normal.shape[0]))
    vector = surface_load.magnitude * tangent / norm
    return tuple(float(value) for value in vector)


def _face_unit_normal(model: Any, face: ElementFace) -> np.ndarray:
    """Return a unit normal for a 3D surface face."""
    node_lookup = {node.id: node for node in model.mesh.nodes}
    coords = []
    for node_id in face.node_ids:
        node = node_lookup[node_id]
        coords.append([float(node.x), float(node.y), float(getattr(node, "z", 0.0))])
    if len(coords) < 3:
        raise ValueError(f"surface face {face} must contain at least 3 nodes for normal")

    p0 = np.array(coords[0], dtype=float)
    p1 = np.array(coords[1], dtype=float)
    p2 = np.array(coords[2], dtype=float)
    normal = np.cross(p1 - p0, p2 - p0)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError(f"surface face {face} has zero normal")
    unit_normal = normal / norm
    max_span = max(float(np.linalg.norm(np.array(coord, dtype=float) - p0)) for coord in coords)
    tolerance = 1e-8 * max(max_span, 1.0)
    for coord in coords[3:]:
        distance = abs(float(np.dot(np.array(coord, dtype=float) - p0, unit_normal)))
        if distance > tolerance:
            raise ValueError(
                f"surface face {face} is non-planar; "
                "TRSHR shear traction requires a planar face"
            )
    return unit_normal


def _validate_component(model: Any, component: int) -> None:
    """Validate a 1-based component against mesh DOFs."""
    if component < 1 or component > model.mesh.dofs_per_node:
        raise ValueError(
            f"component {component} is invalid for mesh with "
            f"{model.mesh.dofs_per_node} DOFs per node"
        )
