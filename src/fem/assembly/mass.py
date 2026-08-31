"""Mass-matrix assembly for the displacement-based dynamic paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from fem.elements import canonical_element_type, get_element_capabilities, get_element_definition
from fem.model import MassMatrixPolicy


def assemble_mass_matrix(
    mesh: Any,
    *,
    policy: MassMatrixPolicy = MassMatrixPolicy.CONSISTENT,
) -> csr_matrix:
    """Assemble a consistent or row-sum lumped mass matrix.

    The first dynamic procedure supports plane/solid continuum and truss
    elements.  Beam rotary inertia is intentionally not guessed here; it will
    receive a dedicated section-mass contract in a later phase.
    """

    try:
        selected_policy = MassMatrixPolicy(str(policy).strip().casefold())
    except ValueError as error:
        raise ValueError("mass policy must be 'consistent' or 'lumped'") from error
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for element in tuple(getattr(mesh, "elements", ())):
        local_dofs = tuple(int(value) for value in mesh.element_dofs(element))
        local_mass = _element_mass(mesh, element)
        if selected_policy is MassMatrixPolicy.LUMPED:
            local_mass = np.diag(np.sum(local_mass, axis=1))
        if local_mass.shape != (len(local_dofs), len(local_dofs)):
            raise ValueError(
                f"element {element.id} mass shape {local_mass.shape} does not "
                f"match {len(local_dofs)} element DOFs"
            )
        for row, global_row in enumerate(local_dofs):
            for column, global_column in enumerate(local_dofs):
                value = float(local_mass[row, column])
                if value != 0.0:
                    rows.append(global_row)
                    columns.append(global_column)
                    values.append(value)
    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(int(mesh.num_dofs), int(mesh.num_dofs)),
        dtype=float,
    ).tocsr()
    matrix.sum_duplicates()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("assembled mass matrix must contain finite values")
    if matrix.nnz == 0:
        raise ValueError("dynamic analysis requires a non-empty mass matrix")
    return matrix


def _element_mass(mesh: Any, element: Any) -> np.ndarray:
    element_type = canonical_element_type(str(element.type))
    descriptor = get_element_capabilities(element_type)
    if descriptor.family == "beam":
        raise NotImplementedError(
            "dynamic analysis does not yet support Beam2 rotary inertia"
        )
    if descriptor.family not in {"plane_continuum", "solid_continuum", "truss"}:
        raise NotImplementedError(
            f"dynamic analysis does not support {element_type} mass"
        )
    rho = _density(getattr(element, "props", {}), element.id)
    nodes = _nodes(mesh, element)
    if descriptor.family == "truss":
        return _truss_mass(element, nodes, rho)
    spatial_dimension = int(descriptor.spatial_dimension)
    coordinates = np.asarray(
        [tuple(float(getattr(node, name)) for name in ("x", "y", "z")[:spatial_dimension]) for node in nodes],
        dtype=float,
    )
    definition = get_element_definition(element_type)
    scale = 1.0
    if descriptor.family == "plane_continuum":
        scale = _positive_property(
            getattr(element, "props", {}),
            "thickness",
            element.id,
            default=1.0,
        )
    local = np.zeros((len(nodes) * spatial_dimension, len(nodes) * spatial_dimension))
    for point in definition.gauss_points():
        parent = tuple(float(value) for value in point[:-1])
        weight = float(point[-1])
        shape = np.asarray(definition.shape_functions(*parent), dtype=float)
        gradients = np.asarray(definition.shape_gradients(*parent), dtype=float)
        if shape.shape != (len(nodes),):
            raise ValueError(f"{element_type} shape function size is invalid")
        jacobian = gradients @ coordinates
        determinant = float(np.linalg.det(jacobian))
        if determinant <= 0.0:
            raise ValueError(
                f"element {element.id} has non-positive mass Jacobian {determinant}"
            )
        local += rho * scale * determinant * weight * np.kron(
            np.outer(shape, shape),
            np.eye(spatial_dimension),
        )
    return local


def _truss_mass(element: Any, nodes: list[Any], rho: float) -> np.ndarray:
    if len(nodes) != 2:
        raise ValueError(f"Truss2 element {element.id} requires two nodes")
    area = _positive_property(getattr(element, "props", {}), "area", element.id)
    coordinates = np.asarray(
        [[float(node.x), float(node.y), float(node.z)] for node in nodes],
        dtype=float,
    )
    length = float(np.linalg.norm(coordinates[1] - coordinates[0]))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"Truss2 element {element.id} length must be > 0")
    scalar = rho * area * length / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    return np.kron(scalar, np.eye(3))


def _nodes(mesh: Any, element: Any) -> list[Any]:
    lookup = {int(node.id): node for node in mesh.nodes}
    try:
        return [lookup[int(node_id)] for node_id in element.node_ids]
    except KeyError as error:
        raise KeyError(f"element {element.id} references a missing node") from error


def _density(properties: Mapping[str, Any], element_id: int) -> float:
    try:
        value = float(properties["rho"])
    except KeyError as error:
        raise ValueError(
            f"dynamic analysis requires density rho for element {element_id}"
        ) from error
    except (TypeError, ValueError) as error:
        raise ValueError(f"element {element_id} density rho is invalid") from error
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"element {element_id} density rho must be finite and > 0"
        )
    return value


def _positive_property(
    properties: Mapping[str, Any],
    name: str,
    element_id: int,
    *,
    default: float | None = None,
) -> float:
    raw = properties.get(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"element {element_id} property {name} is invalid") from error
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"element {element_id} property {name} must be finite and > 0"
        )
    return value


__all__ = ["assemble_mass_matrix"]
