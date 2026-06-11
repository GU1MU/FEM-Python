from __future__ import annotations

from typing import Any

import numpy as np

from .base import build_node_lookup
from ..materials import linear_elastic


def hex8_shape_funcs_grads(xi: float, eta: float, zeta: float):
    """Return N and natural gradients for Hex8."""
    N = np.zeros(8, dtype=float)
    dN_dxi = np.zeros(8, dtype=float)
    dN_deta = np.zeros(8, dtype=float)
    dN_dzeta = np.zeros(8, dtype=float)

    N[0] = (1.0 - xi) * (1.0 - eta) * (1.0 - zeta) / 8.0
    N[1] = (1.0 + xi) * (1.0 - eta) * (1.0 - zeta) / 8.0
    N[2] = (1.0 + xi) * (1.0 + eta) * (1.0 - zeta) / 8.0
    N[3] = (1.0 - xi) * (1.0 + eta) * (1.0 - zeta) / 8.0
    N[4] = (1.0 - xi) * (1.0 - eta) * (1.0 + zeta) / 8.0
    N[5] = (1.0 + xi) * (1.0 - eta) * (1.0 + zeta) / 8.0
    N[6] = (1.0 + xi) * (1.0 + eta) * (1.0 + zeta) / 8.0
    N[7] = (1.0 - xi) * (1.0 + eta) * (1.0 + zeta) / 8.0

    dN_dxi[0] = -(1.0 - eta) * (1.0 - zeta) / 8.0
    dN_dxi[1] = (1.0 - eta) * (1.0 - zeta) / 8.0
    dN_dxi[2] = (1.0 + eta) * (1.0 - zeta) / 8.0
    dN_dxi[3] = -(1.0 + eta) * (1.0 - zeta) / 8.0
    dN_dxi[4] = -(1.0 - eta) * (1.0 + zeta) / 8.0
    dN_dxi[5] = (1.0 - eta) * (1.0 + zeta) / 8.0
    dN_dxi[6] = (1.0 + eta) * (1.0 + zeta) / 8.0
    dN_dxi[7] = -(1.0 + eta) * (1.0 + zeta) / 8.0

    dN_deta[0] = -(1.0 - xi) * (1.0 - zeta) / 8.0
    dN_deta[1] = -(1.0 + xi) * (1.0 - zeta) / 8.0
    dN_deta[2] = (1.0 + xi) * (1.0 - zeta) / 8.0
    dN_deta[3] = (1.0 - xi) * (1.0 - zeta) / 8.0
    dN_deta[4] = -(1.0 - xi) * (1.0 + zeta) / 8.0
    dN_deta[5] = -(1.0 + xi) * (1.0 + zeta) / 8.0
    dN_deta[6] = (1.0 + xi) * (1.0 + zeta) / 8.0
    dN_deta[7] = (1.0 - xi) * (1.0 + zeta) / 8.0

    dN_dzeta[0] = -(1.0 - xi) * (1.0 - eta) / 8.0
    dN_dzeta[1] = -(1.0 + xi) * (1.0 - eta) / 8.0
    dN_dzeta[2] = -(1.0 + xi) * (1.0 + eta) / 8.0
    dN_dzeta[3] = -(1.0 - xi) * (1.0 + eta) / 8.0
    dN_dzeta[4] = (1.0 - xi) * (1.0 - eta) / 8.0
    dN_dzeta[5] = (1.0 + xi) * (1.0 - eta) / 8.0
    dN_dzeta[6] = (1.0 + xi) * (1.0 + eta) / 8.0
    dN_dzeta[7] = (1.0 - xi) * (1.0 + eta) / 8.0

    return N, dN_dxi, dN_deta, dN_dzeta


def hex8_gauss_points(gauss_order: int = 2):
    """Return Gauss points for Hex8."""
    if gauss_order != 2:
        raise ValueError(f"Unsupported gauss_order {gauss_order}, only 2 supported")
    a = 1.0 / np.sqrt(3.0)
    return [
        (-a, -a, -a, 1.0),
        (a, -a, -a, 1.0),
        (a, a, -a, 1.0),
        (-a, a, -a, 1.0),
        (-a, -a, a, 1.0),
        (a, -a, a, 1.0),
        (a, a, a, 1.0),
        (-a, a, a, 1.0),
    ]


def _hex8_extrapolation_matrix() -> np.ndarray:
    """Return the constant 2x2x2 Gauss-to-node extrapolation matrix."""
    n_gp = np.array([
        hex8_shape_funcs_grads(xi, eta, zeta)[0]
        for xi, eta, zeta, _ in hex8_gauss_points()
    ], dtype=float)
    return np.linalg.solve(n_gp, np.eye(8, dtype=float))


HEX8_EXTRAPOLATION_MATRIX = _hex8_extrapolation_matrix()


class Hex8Kernel:
    """Hex8 solid element kernel."""
    type_names = ("Hex8", "C3D8")
    face_nodes = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
    ]

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
        gauss_order: int = 2,
    ) -> np.ndarray:
        """Return Hex8 element stiffness."""
        if len(elem.node_ids) != 8:
            raise ValueError(f"Hex8 element must have 8 nodes, got {len(elem.node_ids)}")

        D = self._material_matrix(elem)
        Ke = np.zeros((24, 24), dtype=float)

        for xi, eta, zeta, w in hex8_gauss_points(gauss_order):
            B, detJ = self._B_matrix(mesh, elem, xi, eta, zeta, node_lookup)
            Ke += (B.T @ D @ B) * detJ * w

        return Ke

    def body_force(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return consistent Hex8 body force vector."""
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y, z = self._coords(nodes)
        bvec = np.array(vector, dtype=float)
        fe = np.zeros(24, dtype=float)

        for xi, eta, zeta, w in hex8_gauss_points():
            N, dN_dxi, dN_deta, dN_dzeta = hex8_shape_funcs_grads(xi, eta, zeta)
            detJ = self._det_jacobian(elem, x, y, z, dN_dxi, dN_deta, dN_dzeta)
            for i in range(8):
                fe[3 * i:3 * i + 3] += N[i] * bvec * (detJ * w)

        return fe

    def face_traction(
        self,
        mesh: Any,
        elem: Any,
        local_face: int,
        traction: tuple[float, float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return consistent Hex8 face traction vector."""
        if local_face < 0 or local_face >= 6:
            raise ValueError(f"Invalid local_face {local_face}, must be 0-5")

        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)
        face_local = self.face_nodes[local_face]
        face_nodes = [node_lookup[elem.node_ids[i]] for i in face_local]
        xyz = np.array([[n.x, n.y, n.z] for n in face_nodes], dtype=float)
        tvec = np.array(traction, dtype=float)
        fe = np.zeros(24, dtype=float)

        a = 1.0 / np.sqrt(3.0)
        for xi, eta, w in [(-a, -a, 1.0), (a, -a, 1.0), (a, a, 1.0), (-a, a, 1.0)]:
            N_face = np.array([
                (1.0 - xi) * (1.0 - eta) / 4.0,
                (1.0 + xi) * (1.0 - eta) / 4.0,
                (1.0 + xi) * (1.0 + eta) / 4.0,
                (1.0 - xi) * (1.0 + eta) / 4.0,
            ], dtype=float)
            dN_dxi = 0.25 * np.array(
                [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
                dtype=float,
            )
            dN_deta = 0.25 * np.array(
                [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
                dtype=float,
            )
            area_scale = float(np.linalg.norm(np.cross(dN_dxi @ xyz, dN_deta @ xyz)))
            if area_scale <= 0.0:
                raise ValueError(f"Hex8 elem {elem.id} face {local_face} has zero area")

            for i, parent_local in enumerate(face_local):
                fe[3 * parent_local:3 * parent_local + 3] += N_face[i] * tvec * (area_scale * w)

        return fe

    def stress_at(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        xi: float,
        eta: float,
        zeta: float,
        node_lookup: dict[int, Any] | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        """Return stress at one natural coordinate point."""
        D = self._material_matrix(elem)
        B, _ = self._B_matrix(mesh, elem, xi, eta, zeta, node_lookup)
        sigma = D @ (B @ U[mesh.element_dofs(elem)])
        return tuple(float(v) for v in sigma)

    def nodal_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
        gauss_order: int = 2,
    ) -> np.ndarray:
        """Return element-nodal stresses extrapolated from 2x2x2 Gauss stresses."""
        gp_vals = np.array([
            self.stress_at(mesh, elem, U, xi, eta, zeta, node_lookup)
            for xi, eta, zeta, _ in hex8_gauss_points(gauss_order)
        ], dtype=float)
        return HEX8_EXTRAPOLATION_MATRIX @ gp_vals

    def _material_matrix(self, elem: Any) -> np.ndarray:
        """Return 3D material matrix from element props."""
        try:
            E = float(elem.props["E"])
            nu = float(elem.props["nu"])
        except KeyError as e:
            raise KeyError(f"Element {elem.id} missing property {e.args[0]}, props={elem.props}")
        return linear_elastic.solid_3d_matrix(E, nu)

    def _nodes(self, mesh: Any, elem: Any, node_lookup: dict[int, Any] | None):
        """Return element nodes in element order."""
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)
        return [node_lookup[nid] for nid in elem.node_ids]

    def _coords(self, nodes: list[Any]):
        """Return coordinate arrays for element nodes."""
        x = np.array([n.x for n in nodes], dtype=float)
        y = np.array([n.y for n in nodes], dtype=float)
        z = np.array([n.z for n in nodes], dtype=float)
        return x, y, z

    def _B_matrix(
        self,
        mesh: Any,
        elem: Any,
        xi: float,
        eta: float,
        zeta: float,
        node_lookup: dict[int, Any] | None,
    ):
        """Return B matrix and detJ at one natural coordinate point."""
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y, z = self._coords(nodes)
        _, dN_dxi, dN_deta, dN_dzeta = hex8_shape_funcs_grads(xi, eta, zeta)

        J = np.array([
            [np.sum(dN_dxi * x), np.sum(dN_dxi * y), np.sum(dN_dxi * z)],
            [np.sum(dN_deta * x), np.sum(dN_deta * y), np.sum(dN_deta * z)],
            [np.sum(dN_dzeta * x), np.sum(dN_dzeta * y), np.sum(dN_dzeta * z)],
        ], dtype=float)
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(f"Element {elem.id} has negative or zero Jacobian determinant")

        invJ = np.linalg.inv(J)
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta + invJ[0, 2] * dN_dzeta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta + invJ[1, 2] * dN_dzeta
        dN_dz = invJ[2, 0] * dN_dxi + invJ[2, 1] * dN_deta + invJ[2, 2] * dN_dzeta

        B = np.zeros((6, 24), dtype=float)
        for i in range(8):
            idx = 3 * i
            B[0, idx] = dN_dx[i]
            B[1, idx + 1] = dN_dy[i]
            B[2, idx + 2] = dN_dz[i]
            B[3, idx] = dN_dy[i]
            B[3, idx + 1] = dN_dx[i]
            B[4, idx + 1] = dN_dz[i]
            B[4, idx + 2] = dN_dy[i]
            B[5, idx] = dN_dz[i]
            B[5, idx + 2] = dN_dx[i]

        return B, detJ

    def _det_jacobian(
        self,
        elem: Any,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        dN_dxi: np.ndarray,
        dN_deta: np.ndarray,
        dN_dzeta: np.ndarray,
    ) -> float:
        """Return detJ from natural shape gradients."""
        J = np.array([
            [np.sum(dN_dxi * x), np.sum(dN_dxi * y), np.sum(dN_dxi * z)],
            [np.sum(dN_deta * x), np.sum(dN_deta * y), np.sum(dN_deta * z)],
            [np.sum(dN_dzeta * x), np.sum(dN_dzeta * y), np.sum(dN_dzeta * z)],
        ], dtype=float)
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(f"Hex8 elem {elem.id} has non-positive Jacobian")
        return detJ
