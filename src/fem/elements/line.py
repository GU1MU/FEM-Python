from __future__ import annotations

from typing import Any

import numpy as np

from .base import build_node_lookup


def _required_float_props(elem: Any, *names: str) -> tuple[float, ...]:
    """Return required, physically admissible element properties."""
    try:
        values = tuple(float(elem.props[name]) for name in names)
    except KeyError as exc:
        raise KeyError(
            f"Element {elem.id} missing property {exc.args[0]}, props={elem.props}"
        ) from exc

    for name, value in zip(names, values):
        if name in {"E", "area", "Iyy", "Izz", "J"} and (
            not np.isfinite(value) or value <= 0.0
        ):
            raise ValueError(
                f"Element {elem.id} property {name} must be finite and > 0, "
                f"got {elem.props[name]!r}"
            )
    return values


def _validate_optional_rho(elem: Any) -> None:
    """Validate optional non-negative density on a line element."""
    if "rho" not in elem.props:
        return
    try:
        rho = float(elem.props["rho"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Element {elem.id} property rho must be finite and >= 0, "
            f"got {elem.props['rho']!r}"
        ) from exc
    if not np.isfinite(rho) or rho < 0.0:
        raise ValueError(
            f"Element {elem.id} property rho must be finite and >= 0, "
            f"got {elem.props['rho']!r}"
        )


def _body_vector_3d(vector: tuple[float, float, float], element_type: str) -> np.ndarray:
    """Return a validated three-component body-force vector."""
    bvec = np.asarray(vector, dtype=float)
    if bvec.shape != (3,):
        raise ValueError(
            f"{element_type} body force must have 3 components, got {bvec.shape}"
        )
    if not np.all(np.isfinite(bvec)):
        raise ValueError(f"{element_type} body force components must be finite")
    return bvec


def line3_geometry(
    mesh: Any,
    elem: Any,
    node_lookup: dict[int, Any] | None = None,
) -> tuple[float, np.ndarray]:
    """Return length and spatial unit direction for a 2-node line element."""
    if len(elem.node_ids) != 2:
        raise ValueError(
            f"Line2 element {elem.id} requires 2 nodes, got {len(elem.node_ids)}; "
            f"node_ids={elem.node_ids}"
        )
    if node_lookup is None:
        node_lookup = build_node_lookup(mesh)

    try:
        ni = node_lookup[elem.node_ids[0]]
        nj = node_lookup[elem.node_ids[1]]
    except KeyError as exc:
        raise KeyError(
            f"Element {elem.id} references missing node {exc.args[0]}"
        ) from exc

    delta = np.array(
        [nj.x - ni.x, nj.y - ni.y, nj.z - ni.z],
        dtype=float,
    )
    length = float(np.linalg.norm(delta))
    if length <= 0.0:
        raise ValueError(f"Line2 element {elem.id} has zero length")
    return length, delta / length


def beam3_geometry(
    mesh: Any,
    elem: Any,
    node_lookup: dict[int, Any] | None = None,
) -> tuple[float, np.ndarray]:
    """Return Beam2 length and global-to-local right-handed rotation."""
    length, e_x = line3_geometry(mesh, elem, node_lookup)
    try:
        raw_local_y = elem.props["local_y"]
    except KeyError as exc:
        raise KeyError(f"Element {elem.id} missing property local_y") from exc
    try:
        reference = np.asarray(raw_local_y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Element {elem.id} local_y must contain three finite components"
        ) from exc
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ValueError(
            f"Element {elem.id} local_y must contain three finite components"
        )
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm <= 0.0:
        raise ValueError(f"Element {elem.id} local_y must be nonzero")
    projected = reference - float(reference @ e_x) * e_x
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm <= 1e-12 * reference_norm:
        raise ValueError(f"Element {elem.id} local_y is parallel to the beam axis")
    e_y = projected / projected_norm
    e_z = np.cross(e_x, e_y)
    return length, np.vstack([e_x, e_y, e_z])


def _beam3_transformation(rotation: np.ndarray) -> np.ndarray:
    """Return global-to-local Beam2 displacement transformation."""
    transformation = np.zeros((12, 12), dtype=float)
    for start in (0, 3, 6, 9):
        transformation[start : start + 3, start : start + 3] = rotation
    return transformation


def _beam2_local_stiffness(
    length: float,
    E: float,
    area: float,
    Iyy: float,
    Izz: float,
    G: float,
    J: float,
) -> np.ndarray:
    """Return the local 12-by-12 Euler-Bernoulli Beam2 stiffness."""
    stiffness = np.zeros((12, 12), dtype=float)

    def add(indices: tuple[int, ...], values: np.ndarray) -> None:
        stiffness[np.ix_(indices, indices)] += values

    add((0, 6), E * area / length * np.array([[1.0, -1.0], [-1.0, 1.0]]))
    add((3, 9), G * J / length * np.array([[1.0, -1.0], [-1.0, 1.0]]))

    bending_z = E * Izz / length**3 * np.array(
        [
            [12.0, 6.0 * length, -12.0, 6.0 * length],
            [6.0 * length, 4.0 * length**2, -6.0 * length, 2.0 * length**2],
            [-12.0, -6.0 * length, 12.0, -6.0 * length],
            [6.0 * length, 2.0 * length**2, -6.0 * length, 4.0 * length**2],
        ]
    )
    add((1, 5, 7, 11), bending_z)

    bending_y = E * Iyy / length**3 * np.array(
        [
            [12.0, -6.0 * length, -12.0, -6.0 * length],
            [-6.0 * length, 4.0 * length**2, 6.0 * length, 2.0 * length**2],
            [-12.0, 6.0 * length, 12.0, 6.0 * length],
            [-6.0 * length, 2.0 * length**2, 6.0 * length, 4.0 * length**2],
        ]
    )
    add((2, 4, 8, 10), bending_y)
    return stiffness


def _beam_properties(elem: Any) -> tuple[float, float, float, float, float, float]:
    """Return validated Beam2 elastic and section properties."""
    E, area, Iyy, Izz, J = _required_float_props(
        elem, "E", "area", "Iyy", "Izz", "J"
    )
    try:
        nu = float(elem.props["nu"])
    except KeyError as exc:
        raise KeyError(f"Element {elem.id} missing property nu") from exc
    if not np.isfinite(nu) or not -1.0 < nu < 0.5:
        raise ValueError(
            f"Element {elem.id} property nu must be finite and satisfy -1 < nu < 0.5, "
            f"got {elem.props['nu']!r}"
        )
    return E, nu, area, Iyy, Izz, J


class Truss2Kernel:
    """Two-node spatial truss element kernel."""
    type_names = ("Truss2",)
    edge_node_indices = ((0, 1),)

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return the 6-by-6 spatial truss stiffness matrix."""
        area, E = _required_float_props(elem, "area", "E")
        _validate_optional_rho(elem)
        length, direction = line3_geometry(mesh, elem, node_lookup)
        block = np.outer(direction, direction)
        return E * area / length * np.block([[block, -block], [-block, block]])

    def body_force(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return the consistent spatial truss body-force vector."""
        (area,) = _required_float_props(elem, "area")
        length, _ = line3_geometry(mesh, elem, node_lookup)
        nodal = _body_vector_3d(vector, "Truss2") * (area * length / 2.0)
        return np.tile(nodal, 2)

    def element_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
    ) -> tuple[float, float, float]:
        """Return axial strain, axial stress, and equivalent stress."""
        (E,) = _required_float_props(elem, "E")
        length, direction = line3_geometry(mesh, elem, node_lookup)
        ni_id, nj_id = elem.node_ids
        displacement_i = np.asarray(U[mesh.node_dofs(ni_id)], dtype=float)
        displacement_j = np.asarray(U[mesh.node_dofs(nj_id)], dtype=float)
        axial_strain = float(direction @ (displacement_j - displacement_i) / length)
        axial_stress = E * axial_strain
        return axial_strain, float(axial_stress), float(abs(axial_stress))


class Beam2Kernel:
    """Two-node spatial Euler-Bernoulli beam element kernel."""
    type_names = ("Beam2",)
    edge_node_indices = ((0, 1),)

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return the transformed 12-by-12 Beam2 stiffness matrix."""
        E, nu, area, Iyy, Izz, J = _beam_properties(elem)
        _validate_optional_rho(elem)
        length, rotation = beam3_geometry(mesh, elem, node_lookup)
        G = E / (2.0 * (1.0 + nu))
        local = _beam2_local_stiffness(length, E, area, Iyy, Izz, G, J)
        transformation = _beam3_transformation(rotation)
        return transformation.T @ local @ transformation

    def body_force(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return the consistent Beam2 body-force vector."""
        (area,) = _required_float_props(elem, "area")
        length, rotation = beam3_geometry(mesh, elem, node_lookup)
        global_vector = _body_vector_3d(vector, "Beam2")
        local_line_load = area * (rotation @ global_vector)
        local_force = _beam2_consistent_line_load(length, local_line_load)
        return _beam3_transformation(rotation).T @ local_force

    def line_load(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float, float],
        coordinate_system: str = "global",
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return the consistent Beam2 force for a constant line load."""
        line_vector = _body_vector_3d(vector, "line load")
        if coordinate_system not in {"global", "local"}:
            raise ValueError(
                "line load coordinate_system must be 'global' or 'local', "
                f"got {coordinate_system!r}"
            )
        length, rotation = beam3_geometry(mesh, elem, node_lookup)
        local_vector = rotation @ line_vector if coordinate_system == "global" else line_vector
        local_force = _beam2_consistent_line_load(length, local_vector)
        return _beam3_transformation(rotation).T @ local_force


def _beam2_consistent_line_load(length: float, vector: np.ndarray) -> np.ndarray:
    """Return consistent local nodal forces for a constant Beam2 line load."""
    qx, qy, qz = vector
    return np.array(
        [
            qx * length / 2.0,
            qy * length / 2.0,
            qz * length / 2.0,
            0.0,
            -qz * length**2 / 12.0,
            qy * length**2 / 12.0,
            qx * length / 2.0,
            qy * length / 2.0,
            qz * length / 2.0,
            0.0,
            qz * length**2 / 12.0,
            -qy * length**2 / 12.0,
        ],
        dtype=float,
    )
