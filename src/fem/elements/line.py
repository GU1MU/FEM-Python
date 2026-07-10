from __future__ import annotations

from typing import Any

import numpy as np

from .base import build_node_lookup


def _required_float_props(elem: Any, *names: str) -> tuple[float, ...]:
    """Return required element properties as floats."""
    try:
        return tuple(float(elem.props[name]) for name in names)
    except KeyError as exc:
        raise KeyError(
            f"Element {elem.id} missing property {exc.args[0]}, props={elem.props}"
        ) from exc


def _body_vector_2d(vector: tuple[float, float], element_type: str) -> np.ndarray:
    """Return a validated two-component body-force vector."""
    bvec = np.asarray(vector, dtype=float)
    if bvec.shape != (2,):
        raise ValueError(
            f"{element_type} body force must have 2 components, got {bvec.shape}"
        )
    return bvec


def line2_geometry(
    mesh: Any,
    elem: Any,
    node_lookup: dict[int, Any] | None = None,
) -> tuple[float, float, float]:
    """Return length and direction cosines for a 2-node line element."""
    if len(elem.node_ids) != 2:
        raise ValueError(
            f"Line2 element must have 2 nodes, elem {elem.id} node_ids={elem.node_ids}"
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

    dx = nj.x - ni.x
    dy = nj.y - ni.y
    length = float(np.hypot(dx, dy))
    if length <= 0.0:
        raise ValueError(f"Line2 element {elem.id} has zero length")
    return length, dx / length, dy / length


def _beam2d_transformation(c: float, s: float) -> np.ndarray:
    """Return the global-to-local Beam2D displacement transformation."""
    return np.array([
        [c, s, 0.0, 0.0, 0.0, 0.0],
        [-s, c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c, s, 0.0],
        [0.0, 0.0, 0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


class Truss2DKernel:
    """Two-node planar truss element kernel."""
    type_names = ("Truss2D",)

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return Truss2D element stiffness."""
        area, E = _required_float_props(elem, "area", "E")

        L, c, s = line2_geometry(mesh, elem, node_lookup)
        k = E * area / L
        return k * np.array([
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ], dtype=float)

    def body_force(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return the consistent Truss2D body force vector."""
        (area,) = _required_float_props(elem, "area")

        length, _, _ = line2_geometry(mesh, elem, node_lookup)
        bvec = _body_vector_2d(vector, "Truss2D")
        nodal = bvec * (area * length / 2.0)
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

        L, c, s = line2_geometry(mesh, elem, node_lookup)
        ni_id, nj_id = elem.node_ids
        uix = U[mesh.global_dof(ni_id, 0)]
        uiy = U[mesh.global_dof(ni_id, 1)]
        ujx = U[mesh.global_dof(nj_id, 0)]
        ujy = U[mesh.global_dof(nj_id, 1)]

        u_i_l = c * uix + s * uiy
        u_j_l = c * ujx + s * ujy
        axial_strain = (u_j_l - u_i_l) / L
        axial_stress = E * axial_strain
        return float(axial_strain), float(axial_stress), float(abs(axial_stress))


class Beam2DKernel:
    """Two-node Euler-Bernoulli beam element kernel."""
    type_names = ("Beam2D",)

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return Beam2D element stiffness."""
        E, area, I = _required_float_props(elem, "E", "area", "Izz")

        L, c, s = line2_geometry(mesh, elem, node_lookup)
        EA_L = E * area / L
        EI_L3 = E * I / (L**3)
        EI_L2 = E * I / (L**2)
        EI_L = E * I / L

        k_local = np.array([
            [EA_L, 0.0, 0.0, -EA_L, 0.0, 0.0],
            [0.0, 12 * EI_L3, 6 * EI_L2, 0.0, -12 * EI_L3, 6 * EI_L2],
            [0.0, 6 * EI_L2, 4 * EI_L, 0.0, -6 * EI_L2, 2 * EI_L],
            [-EA_L, 0.0, 0.0, EA_L, 0.0, 0.0],
            [0.0, -12 * EI_L3, -6 * EI_L2, 0.0, 12 * EI_L3, -6 * EI_L2],
            [0.0, 6 * EI_L2, 2 * EI_L, 0.0, -6 * EI_L2, 4 * EI_L],
        ], dtype=float)

        T = _beam2d_transformation(c, s)
        return T.T @ k_local @ T

    def body_force(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return the consistent Beam2D body force vector."""
        (area,) = _required_float_props(elem, "area")

        length, c, s = line2_geometry(mesh, elem, node_lookup)
        bvec = _body_vector_2d(vector, "Beam2D")

        qx, qy = area * np.array([
            c * bvec[0] + s * bvec[1],
            -s * bvec[0] + c * bvec[1],
        ])
        f_local = np.array([
            qx * length / 2.0,
            qy * length / 2.0,
            qy * length**2 / 12.0,
            qx * length / 2.0,
            qy * length / 2.0,
            -qy * length**2 / 12.0,
        ], dtype=float)
        return _beam2d_transformation(c, s).T @ f_local
