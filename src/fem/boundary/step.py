from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.model import (
    AnalysisStep,
    EdgeLoad,
    ElementEdge,
    ElementFace,
    SurfaceLoad,
    model_element_info,
)
from ._common import spatial_dim
from .condition import BoundaryCondition


_LookupFactory = Callable[[], dict[int, Any]]


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
    node_lookup_cache: dict[int, Any] | None = None
    elem_lookup_cache: dict[int, Any] | None = None

    def node_lookup() -> dict[int, Any]:
        nonlocal node_lookup_cache
        if node_lookup_cache is None:
            node_lookup_cache = {node.id: node for node in model.mesh.nodes}
        return node_lookup_cache

    def elem_lookup() -> dict[int, Any]:
        nonlocal elem_lookup_cache
        if elem_lookup_cache is None:
            elem_lookup_cache = {int(elem.id): elem for elem in model.mesh.elements}
        return elem_lookup_cache
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
        if spatial_dim(model.mesh) == 2:
            raise ValueError("2D surface loads are not supported; use edge loads")
        if surface_load.surface not in model.surfaces:
            raise KeyError(f"surface {surface_load.surface} is not defined")
        for face in model.surfaces[surface_load.surface].faces:
            if surface_load.load_type == "pressure":
                vector = _pressure_vector(
                    face,
                    surface_load,
                    node_lookup,
                    elem_lookup,
                )
            elif surface_load.load_type == "traction":
                vector = surface_load.vector
            elif surface_load.load_type == "shear_traction":
                vector = _shear_traction_vector(face, surface_load, node_lookup)
            else:
                raise ValueError(f"unsupported surface load type: {surface_load.load_type}")
            boundary.add_surface_traction(face.elem_id, face.local_index, *vector)

    for edge_load in selected_step.edge_loads:
        if edge_load.edge not in model.edges:
            raise KeyError(f"edge {edge_load.edge} is not defined")
        if spatial_dim(model.mesh) == 3:
            raise NotImplementedError("3D edge loads are not supported")
        for edge in model.edges[edge_load.edge].edges:
            if edge_load.load_type == "pressure":
                vector = _edge_pressure_vector_2d(
                    edge,
                    edge_load,
                    node_lookup,
                    elem_lookup,
                )
            elif edge_load.load_type == "traction":
                vector = edge_load.vector
            else:
                raise ValueError(f"unsupported edge load type: {edge_load.load_type}")
            boundary.add_edge_traction(edge.elem_id, edge.local_index, *vector)

    for line_load in selected_step.line_loads:
        vector = _validated_line_load_vector(line_load.vector)
        if line_load.coordinate_system not in {"global", "local"}:
            raise ValueError(
                "line load coordinate_system must be 'global' or 'local', "
                f"got {line_load.coordinate_system!r}"
            )
        for elem_id in _resolve_element_target(model, line_load.target):
            elem = elem_lookup().get(elem_id)
            if elem is None:
                raise KeyError(f"element {elem_id} is not defined")
            if str(elem.type).casefold() != "beam2":
                raise ValueError("line loads may target only Beam2 elements")
            boundary.add_line_load(
                elem_id,
                vector,
                line_load.coordinate_system,
            )

    _add_gravity_loads(model, selected_step, boundary)

    return boundary


def _resolve_node_target(model: Any, target: str | int) -> tuple[int, ...]:
    """Resolve a node id or named node set."""
    if isinstance(target, int):
        return (target,)
    if target not in model.node_sets:
        raise KeyError(f"node set {target} is not defined")
    return model.node_sets[target].node_ids


def _resolve_element_target(model: Any, target: str | int) -> tuple[int, ...]:
    """Resolve an element id or named element set."""
    return _resolve_element_target_from_sets(target, model.element_sets)


def _resolve_gravity_target(model: Any, target: str | int) -> tuple[int, ...]:
    """Resolve gravity targets, including importer-internal element sets."""
    element_sets = dict(model.element_sets)
    element_sets.update(model.metadata.get("_abaqus_internal_element_sets", {}))
    return _resolve_element_target_from_sets(target, element_sets)


def _resolve_element_target_from_sets(
    target: Any,
    element_sets: dict[str, Any],
) -> tuple[int, ...]:
    """Resolve an integral element id or a name from an element-set mapping."""
    if not isinstance(target, str):
        if isinstance(target, bool):
            raise TypeError("element target must be an integer id or element set name")
        try:
            return (int(operator.index(target)),)
        except TypeError as exc:
            raise TypeError(
                "element target must be an integer id or element set name"
            ) from exc
    if target not in element_sets:
        raise KeyError(f"element set {target} is not defined")
    return tuple(element_sets[target].element_ids)


def _add_gravity_loads(
    model: Any,
    step: AnalysisStep,
    boundary: BoundaryCondition,
) -> None:
    """Resolve a step's global and element-targeted gravity records."""
    if not step.gravity_loads:
        return
    dim = spatial_dim(model.mesh)
    global_acceleration = np.zeros(dim, dtype=float)
    has_global = False

    for load in step.gravity_loads:
        acceleration = _validated_gravity_vector(load.acceleration, dim)
        if load.target is None:
            global_acceleration += acceleration
            if not np.all(np.isfinite(global_acceleration)):
                raise ValueError("accumulated gravity acceleration must be finite")
            has_global = True
            continue

        for elem_id in _resolve_gravity_target(model, load.target):
            _validated_effective_density(model, step.name, load.target, elem_id)
            boundary.add_gravity_element(elem_id, *acceleration)

    if has_global:
        boundary.set_gravity(*(float(value) for value in global_acceleration))


def _validated_gravity_vector(vector: Any, dim: int) -> tuple[float, ...]:
    """Return a finite gravity acceleration matching the mesh dimension."""
    try:
        values = np.asarray(vector, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("gravity acceleration must contain finite numbers") from exc
    if values.shape != (dim,):
        raise ValueError(
            f"gravity acceleration must have {dim} components, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("gravity acceleration must contain finite numbers")
    return tuple(float(value) for value in values)


def _validated_effective_density(
    model: Any,
    step_name: str,
    target: str | int,
    elem_id: int,
) -> float:
    """Return effective density required by explicitly targeted gravity."""
    density = model_element_info(model, elem_id).properties.get("rho")
    context = (
        f"analysis step {step_name} gravity target {target!r} element {elem_id}"
    )
    if density is None:
        raise ValueError(f"{context} requires an effective density rho")
    try:
        value = float(density)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} density rho must be numeric") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            f"{context} density rho must be finite and >= 0, got {density!r}"
        )
    return value


def _validated_line_load_vector(vector: Any) -> tuple[float, float, float]:
    """Return a finite three-component line-load vector."""
    try:
        values = np.asarray(vector, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("line load vector must contain three finite numbers") from exc
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("line load vector must contain three finite numbers")
    return tuple(float(value) for value in values)


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
    face: ElementFace,
    surface_load: SurfaceLoad,
    node_lookup_factory: _LookupFactory,
    elem_lookup_factory: _LookupFactory,
) -> tuple[float, ...]:
    """Return an inward pressure vector for one surface face."""
    if surface_load.magnitude is None:
        raise ValueError("pressure surface load requires a magnitude")

    return _pressure_vector_3d(
        face,
        surface_load,
        node_lookup_factory,
        elem_lookup_factory,
    )


def _edge_pressure_vector_2d(
    edge: ElementEdge,
    edge_load: EdgeLoad,
    node_lookup_factory: _LookupFactory,
    elem_lookup_factory: _LookupFactory,
) -> tuple[float, float]:
    """Return an inward pressure vector for one 2D element edge."""
    if edge_load.magnitude is None:
        raise ValueError("pressure edge load requires a magnitude")

    node_lookup = node_lookup_factory()
    if len(edge.node_ids) < 2:
        raise ValueError(f"edge {edge} must contain at least 2 nodes for pressure")

    first = node_lookup[edge.node_ids[0]]
    last = node_lookup[edge.node_ids[-1]]
    p0 = np.array([float(first.x), float(first.y)], dtype=float)
    p1 = np.array([float(last.x), float(last.y)], dtype=float)
    tangent = p1 - p0
    length = float(np.linalg.norm(tangent))
    if length <= 0.0:
        raise ValueError(f"edge {edge} has zero length")

    normal = np.array([tangent[1], -tangent[0]], dtype=float) / length

    elem_lookup = elem_lookup_factory()
    elem = elem_lookup.get(edge.elem_id)
    if elem is None:
        raise KeyError(f"element {edge.elem_id} is not defined")
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

    vector = float(edge_load.magnitude) * normal
    return float(vector[0]), float(vector[1])


def _pressure_vector_3d(
    face: ElementFace,
    surface_load: SurfaceLoad,
    node_lookup_factory: _LookupFactory,
    elem_lookup_factory: _LookupFactory,
) -> tuple[float, ...]:
    """Return an inward pressure vector for one 3D surface face."""
    node_lookup = node_lookup_factory()
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

    elem_lookup = elem_lookup_factory()
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
    face: ElementFace,
    surface_load: SurfaceLoad,
    node_lookup_factory: _LookupFactory,
) -> tuple[float, ...]:
    """Return a surface shear traction projected onto one face tangent plane."""
    if surface_load.magnitude is None:
        raise ValueError("shear traction surface load requires a magnitude")
    if not surface_load.vector:
        raise ValueError("shear traction surface load requires a direction vector")

    normal = _face_unit_normal(face, node_lookup_factory())
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


def _face_unit_normal(face: ElementFace, node_lookup: dict[int, Any]) -> np.ndarray:
    """Return a unit normal for a 3D surface face."""
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
