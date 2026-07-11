from __future__ import annotations

from typing import Any

import numpy as np

from .base import build_node_lookup, extrapolate_tensor_product
from ..materials import linear_elastic


def quad4_shape_grad_xi_eta(xi: float, eta: float) -> np.ndarray:
    """Return dN/dxi and dN/deta for bilinear Quad4."""
    dN_dxi = np.array(
        [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
        dtype=float,
    ) * 0.25
    dN_deta = np.array(
        [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        dtype=float,
    ) * 0.25
    return np.vstack([dN_dxi, dN_deta])


def quad4_gauss_points(gauss_order: int):
    """Return Gauss points for Quad4."""
    if gauss_order == 1:
        return [(0.0, 0.0, 4.0)]
    if gauss_order == 2:
        a = 1.0 / np.sqrt(3.0)
        return [(-a, -a, 1.0), (a, -a, 1.0), (a, a, 1.0), (-a, a, 1.0)]
    raise ValueError("gauss_order must be 1 or 2")


class Quad4PlaneKernel:
    """Quad4 plane stress/strain element kernel."""
    type_names = ("Quad4Plane", "Quad4", "CPS4", "CPE4")
    edge_node_indices = ((0, 1), (1, 2), (2, 3), (3, 0))

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
        gauss_order: int = 2,
    ) -> np.ndarray:
        """Return Quad4 plane element stiffness."""
        B_data = self._integration_data(mesh, elem, node_lookup, gauss_order)
        D, t = self._material_data(elem)
        Ke = np.zeros((8, 8), dtype=float)

        for B, detJ, w in B_data:
            Ke += (B.T @ D @ B) * (t * detJ * w)

        return Ke

    def stress_at(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        xi: float,
        eta: float,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return stress at one natural coordinate point."""
        D, _ = self._material_data(elem)
        B = self._B_matrix(mesh, elem, xi, eta, node_lookup)[0]
        Ue = U[mesh.element_dofs(elem)]
        return D @ (B @ Ue)

    def nodal_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
        gauss_order: int = 2,
    ):
        """Return extrapolated element-nodal stress, plane type, and nu."""
        if gauss_order != 2:
            raise ValueError("gauss_order must be 2 for Quad4 extrapolation")

        a = 1.0 / np.sqrt(3.0)
        xi_pts = [-a, a]
        eta_pts = [-a, a]
        gp_vals = np.vstack([
            self.stress_at(mesh, elem, U, xi, eta, node_lookup)
            for xi in xi_pts
            for eta in eta_pts
        ])
        node_coords = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        node_vals = extrapolate_tensor_product(gp_vals, xi_pts, eta_pts, node_coords)
        plane_type, nu = self._plane_data(elem)
        return node_vals, plane_type, nu

    def body_force(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return consistent Quad4 body force vector."""
        t = self._thickness(elem)
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y = self._coords(nodes)
        bvec = np.array(vector, dtype=float)
        fe = np.zeros(8, dtype=float)

        for xi, eta, w in quad4_gauss_points(2):
            N = self._shape_funcs(xi, eta)
            dN = quad4_shape_grad_xi_eta(xi, eta)
            detJ = self._det_jacobian(elem, x, y, dN)
            for i in range(4):
                fe[2 * i:2 * i + 2] += N[i] * bvec * (t * detJ * w)
        return fe

    def edge_traction(
        self,
        mesh: Any,
        elem: Any,
        local_edge: int,
        traction: tuple[float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return consistent Quad4 edge traction vector."""
        if local_edge < 0 or local_edge >= 4:
            raise ValueError(f"Quad4 local_edge must be 0/1/2/3, got {local_edge}")

        t = self._thickness(elem)
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y = self._coords(nodes)
        tvec = np.array(traction, dtype=float)
        fe = np.zeros(8, dtype=float)
        gp = 1.0 / np.sqrt(3.0)

        for s, w in [(-gp, 1.0), (gp, 1.0)]:
            xi, eta, dxi_ds, deta_ds = self._edge_point(local_edge, s)
            N = self._shape_funcs(xi, eta)
            dN = quad4_shape_grad_xi_eta(xi, eta)
            jac = self._edge_jacobian(elem, x, y, dN, dxi_ds, deta_ds)
            for i in range(4):
                fe[2 * i:2 * i + 2] += N[i] * tvec * (t * jac * w)
        return fe

    def _material_data(self, elem: Any):
        """Return D matrix and thickness from element props."""
        try:
            E = float(elem.props["E"])
            nu = float(elem.props["nu"])
        except KeyError as exc:
            raise KeyError(
                f"Element {elem.id} missing property {exc.args[0]}, props={elem.props}"
            ) from exc

        t = self._thickness(elem)
        pt, _ = self._plane_data(elem)
        D = linear_elastic.plane_matrix(E, nu, pt)
        return D, t

    def _plane_data(self, elem: Any):
        """Return plane type tag and Poisson ratio."""
        try:
            nu = float(elem.props["nu"])
        except KeyError as exc:
            raise KeyError(
                f"Element {elem.id} missing property {exc.args[0]}, props={elem.props}"
            ) from exc
        if "plane_type" in elem.props:
            pt = str(elem.props["plane_type"]).lower()
        elif str(elem.type).upper().startswith("CPE"):
            pt = "strain"
        else:
            pt = "stress"
        if pt.startswith("stress"):
            return "stress", nu
        if pt.startswith("strain"):
            return "strain", nu
        raise ValueError(
            f"Element {elem.id} has plane_type {elem.props.get('plane_type')!r}; "
            "expected 'stress' or 'strain'"
        )

    def _thickness(self, elem: Any) -> float:
        """Return plane element thickness."""
        thickness = float(elem.props.get("thickness", 1.0))
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError(
                f"Quad4 element {elem.id} thickness must be finite and > 0, "
                f"got {elem.props.get('thickness', 1.0)!r}"
            )
        return thickness

    def _nodes(self, mesh: Any, elem: Any, node_lookup: dict[int, Any] | None):
        """Return element nodes in element order."""
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)
        return [node_lookup[node_id] for node_id in elem.node_ids]

    def _coords(self, nodes: list[Any]):
        """Return x and y coordinate arrays."""
        return (
            np.array([n.x for n in nodes], dtype=float),
            np.array([n.y for n in nodes], dtype=float),
        )

    def _integration_data(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None,
        gauss_order: int,
    ):
        """Return B, detJ, and weight at integration points."""
        if len(elem.node_ids) != 4:
            raise ValueError(
                f"Quad4 element {elem.id} requires 4 nodes, got {len(elem.node_ids)}; "
                f"node_ids={elem.node_ids}"
            )
        return [
            (*self._B_matrix(mesh, elem, xi, eta, node_lookup), w)
            for xi, eta, w in quad4_gauss_points(gauss_order)
        ]

    def _B_matrix(
        self,
        mesh: Any,
        elem: Any,
        xi: float,
        eta: float,
        node_lookup: dict[int, Any] | None,
    ):
        """Return B matrix and detJ at one natural coordinate point."""
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)

        nodes = self._nodes(mesh, elem, node_lookup)
        x, y = self._coords(nodes)

        dN = quad4_shape_grad_xi_eta(xi, eta)
        J = self._jacobian(x, y, dN)
        detJ = self._checked_det_jacobian(elem, J)

        dN_xy = np.linalg.inv(J) @ dN
        B = np.zeros((3, 8), dtype=float)
        for a_i in range(4):
            dN_dx = dN_xy[0, a_i]
            dN_dy = dN_xy[1, a_i]
            c = 2 * a_i
            B[0, c] = dN_dx
            B[1, c + 1] = dN_dy
            B[2, c] = dN_dy
            B[2, c + 1] = dN_dx

        return B, detJ

    def _shape_funcs(self, xi: float, eta: float) -> np.ndarray:
        """Return Quad4 shape functions."""
        return 0.25 * np.array([
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ], dtype=float)

    def _jacobian(self, x: np.ndarray, y: np.ndarray, dN: np.ndarray) -> np.ndarray:
        """Return 2D isoparametric Jacobian."""
        return np.array(
            [[np.dot(dN[0], x), np.dot(dN[0], y)],
             [np.dot(dN[1], x), np.dot(dN[1], y)]],
            dtype=float,
        )

    def _checked_det_jacobian(self, elem: Any, J: np.ndarray) -> float:
        """Return detJ and reject singular or inverted elements."""
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(
                f"Quad4 element {elem.id} has non-positive Jacobian determinant "
                f"{detJ}; expected > 0"
            )
        return detJ

    def _det_jacobian(self, elem: Any, x: np.ndarray, y: np.ndarray, dN: np.ndarray) -> float:
        """Return detJ from natural gradients."""
        return self._checked_det_jacobian(elem, self._jacobian(x, y, dN))

    def _edge_point(self, local_edge: int, s: float):
        """Map edge parameter to natural coordinates."""
        if local_edge == 0:
            return s, -1.0, 1.0, 0.0
        if local_edge == 1:
            return 1.0, s, 0.0, 1.0
        if local_edge == 2:
            return -s, 1.0, -1.0, 0.0
        return -1.0, -s, 0.0, -1.0

    def _edge_jacobian(
        self,
        elem: Any,
        x: np.ndarray,
        y: np.ndarray,
        dN: np.ndarray,
        dxi_ds: float,
        deta_ds: float,
    ) -> float:
        """Return edge length scale from natural gradients."""
        dx_dxi = float(np.dot(dN[0], x))
        dy_dxi = float(np.dot(dN[0], y))
        dx_deta = float(np.dot(dN[1], x))
        dy_deta = float(np.dot(dN[1], y))
        jac = float(np.hypot(
            dx_dxi * dxi_ds + dx_deta * deta_ds,
            dy_dxi * dxi_ds + dy_deta * deta_ds,
        ))
        if jac == 0.0:
            raise ValueError(
                f"Quad4 element {elem.id} edge has zero Jacobian; expected > 0"
            )
        return jac


def quad8_shape_funcs_grads(xi: float, eta: float):
    """Return N, dN/dxi, and dN/deta for Quad8."""
    N = np.zeros(8, dtype=float)
    dN_dxi = np.zeros(8, dtype=float)
    dN_deta = np.zeros(8, dtype=float)

    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0)
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0)
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0)
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0)
    N[4] = 0.5 * (1.0 - xi * xi) * (1.0 - eta)
    N[5] = 0.5 * (1.0 + xi) * (1.0 - eta * eta)
    N[6] = 0.5 * (1.0 - xi * xi) * (1.0 + eta)
    N[7] = 0.5 * (1.0 - xi) * (1.0 - eta * eta)

    dN_dxi[0] = 0.25 * (-(1.0 - eta) * (-xi - eta - 1.0) + (1.0 - xi) * (1.0 - eta) * (-1.0))
    dN_dxi[1] = 0.25 * ((1.0 - eta) * (xi - eta - 1.0) + (1.0 + xi) * (1.0 - eta) * (1.0))
    dN_dxi[2] = 0.25 * ((1.0 + eta) * (xi + eta - 1.0) + (1.0 + xi) * (1.0 + eta) * (1.0))
    dN_dxi[3] = 0.25 * (-(1.0 + eta) * (-xi + eta - 1.0) + (1.0 - xi) * (1.0 + eta) * (-1.0))
    dN_dxi[4] = -xi * (1.0 - eta)
    dN_dxi[5] = 0.5 * (1.0 - eta * eta)
    dN_dxi[6] = -xi * (1.0 + eta)
    dN_dxi[7] = -0.5 * (1.0 - eta * eta)

    dN_deta[0] = 0.25 * (-(1.0 - xi) * (-xi - eta - 1.0) + (1.0 - xi) * (1.0 - eta) * (-1.0))
    dN_deta[1] = 0.25 * (-(1.0 + xi) * (xi - eta - 1.0) + (1.0 + xi) * (1.0 - eta) * (-1.0))
    dN_deta[2] = 0.25 * ((1.0 + xi) * (xi + eta - 1.0) + (1.0 + xi) * (1.0 + eta) * (1.0))
    dN_deta[3] = 0.25 * ((1.0 - xi) * (-xi + eta - 1.0) + (1.0 - xi) * (1.0 + eta) * (1.0))
    dN_deta[4] = -0.5 * (1.0 - xi * xi)
    dN_deta[5] = -(1.0 + xi) * eta
    dN_deta[6] = 0.5 * (1.0 - xi * xi)
    dN_deta[7] = -(1.0 - xi) * eta

    return N, dN_dxi, dN_deta


def quad8_gauss_points(gauss_order: int):
    """Return Gauss points for Quad8."""
    if gauss_order == 2:
        a = 1.0 / np.sqrt(3.0)
        return [(-a, -a, 1.0), (a, -a, 1.0), (a, a, 1.0), (-a, a, 1.0)]
    if gauss_order == 3:
        r = np.sqrt(3.0 / 5.0)
        one_d = [(-r, 5.0 / 9.0), (0.0, 8.0 / 9.0), (r, 5.0 / 9.0)]
        pts = []
        for xi, wx in one_d:
            for eta, wy in one_d:
                pts.append((xi, eta, wx * wy))
        return pts
    raise ValueError("gauss_order must be 2 or 3 for Quad8")


class Quad8PlaneKernel:
    """Quad8 plane stress/strain element kernel."""
    type_names = ("Quad8Plane", "Quad8", "CPS8", "CPE8")
    edge_node_indices = (
        (0, 4, 1),
        (1, 5, 2),
        (2, 6, 3),
        (3, 7, 0),
    )

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
        gauss_order: int = 3,
    ) -> np.ndarray:
        """Return Quad8 plane element stiffness."""
        if len(elem.node_ids) != 8:
            raise ValueError(
                f"Quad8 element {elem.id} requires 8 nodes, got {len(elem.node_ids)}; "
                f"node_ids={elem.node_ids}"
            )

        D, t = self._material_data(elem)
        Ke = np.zeros((16, 16), dtype=float)
        for xi, eta, w in quad8_gauss_points(gauss_order):
            B, detJ = self._B_matrix(mesh, elem, xi, eta, node_lookup)
            Ke += (B.T @ D @ B) * (t * detJ * w)
        return Ke

    def stress_at(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        xi: float,
        eta: float,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return stress at one natural coordinate point."""
        D, _ = self._material_data(elem)
        B, _ = self._B_matrix(mesh, elem, xi, eta, node_lookup)
        return D @ (B @ U[mesh.element_dofs(elem)])

    def nodal_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
        gauss_order: int = 3,
    ):
        """Return extrapolated element-nodal stress, plane type, and nu."""
        if gauss_order not in (2, 3):
            raise ValueError("gauss_order must be 2 or 3 for Quad8 extrapolation")

        if gauss_order == 2:
            a = 1.0 / np.sqrt(3.0)
            xi_pts = [-a, a]
            eta_pts = [-a, a]
        else:
            r = np.sqrt(3.0 / 5.0)
            xi_pts = [-r, 0.0, r]
            eta_pts = [-r, 0.0, r]

        gp_vals = np.vstack([
            self.stress_at(mesh, elem, U, xi, eta, node_lookup)
            for xi in xi_pts
            for eta in eta_pts
        ])
        node_coords = [
            (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
            (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0),
        ]
        node_vals = extrapolate_tensor_product(gp_vals, xi_pts, eta_pts, node_coords)
        plane_type, nu = self._plane_data(elem)
        return node_vals, plane_type, nu

    def body_force(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return consistent Quad8 body force vector."""
        t = self._thickness(elem)
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y = self._coords(nodes)
        bvec = np.array(vector, dtype=float)
        fe = np.zeros(16, dtype=float)

        for xi, eta, w in quad8_gauss_points(3):
            N, dN_dxi, dN_deta = quad8_shape_funcs_grads(xi, eta)
            detJ = self._det_jacobian(elem, x, y, dN_dxi, dN_deta)
            for i in range(8):
                fe[2 * i:2 * i + 2] += N[i] * bvec * (t * detJ * w)
        return fe

    def edge_traction(
        self,
        mesh: Any,
        elem: Any,
        local_edge: int,
        traction: tuple[float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return consistent Quad8 edge traction vector."""
        if local_edge < 0 or local_edge >= 4:
            raise ValueError(f"Quad8 local_edge must be 0/1/2/3, got {local_edge}")

        t = self._thickness(elem)
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y = self._coords(nodes)
        tvec = np.array(traction, dtype=float)
        fe = np.zeros(16, dtype=float)
        r = np.sqrt(3.0 / 5.0)

        for s, w in [(-r, 5.0 / 9.0), (0.0, 8.0 / 9.0), (r, 5.0 / 9.0)]:
            xi, eta, dxi_ds, deta_ds = self._edge_point(local_edge, s)
            N, dN_dxi, dN_deta = quad8_shape_funcs_grads(xi, eta)
            jac = self._edge_jacobian(elem, x, y, dN_dxi, dN_deta, dxi_ds, deta_ds)
            for i in range(8):
                fe[2 * i:2 * i + 2] += N[i] * tvec * (t * jac * w)
        return fe

    def _material_data(self, elem: Any):
        """Return D matrix and thickness from element props."""
        try:
            E = float(elem.props["E"])
            nu = float(elem.props["nu"])
        except KeyError as exc:
            raise KeyError(
                f"Element {elem.id} missing property {exc.args[0]}, props={elem.props}"
            ) from exc

        t = self._thickness(elem)
        pt, _ = self._plane_data(elem)
        D = linear_elastic.plane_matrix(E, nu, pt)
        return D, t

    def _plane_data(self, elem: Any):
        """Return plane type tag and Poisson ratio."""
        try:
            nu = float(elem.props["nu"])
        except KeyError as exc:
            raise KeyError(
                f"Element {elem.id} missing property {exc.args[0]}, props={elem.props}"
            ) from exc
        if "plane_type" in elem.props:
            pt = str(elem.props["plane_type"]).lower()
        elif str(elem.type).upper().startswith("CPE"):
            pt = "strain"
        else:
            pt = "stress"
        if pt.startswith("stress"):
            return "stress", nu
        if pt.startswith("strain"):
            return "strain", nu
        raise ValueError(
            f"Element {elem.id} has plane_type {elem.props.get('plane_type')!r}; "
            "expected 'stress' or 'strain'"
        )

    def _thickness(self, elem: Any) -> float:
        """Return plane element thickness."""
        thickness = float(elem.props.get("thickness", 1.0))
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError(
                f"Quad8 element {elem.id} thickness must be finite and > 0, "
                f"got {elem.props.get('thickness', 1.0)!r}"
            )
        return thickness

    def _nodes(self, mesh: Any, elem: Any, node_lookup: dict[int, Any] | None):
        """Return element nodes in element order."""
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)
        return [node_lookup[node_id] for node_id in elem.node_ids]

    def _coords(self, nodes: list[Any]):
        """Return x and y coordinate arrays."""
        return (
            np.array([n.x for n in nodes], dtype=float),
            np.array([n.y for n in nodes], dtype=float),
        )

    def _B_matrix(
        self,
        mesh: Any,
        elem: Any,
        xi: float,
        eta: float,
        node_lookup: dict[int, Any] | None,
    ):
        """Return B matrix and detJ at one natural coordinate point."""
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y = self._coords(nodes)

        _, dN_dxi, dN_deta = quad8_shape_funcs_grads(xi, eta)
        J = self._jacobian(x, y, dN_dxi, dN_deta)
        detJ = self._checked_det_jacobian(elem, J)

        dN_xy = np.linalg.inv(J) @ np.vstack([dN_dxi, dN_deta])
        B = np.zeros((3, 16), dtype=float)
        for a_i in range(8):
            dN_dx = dN_xy[0, a_i]
            dN_dy = dN_xy[1, a_i]
            c = 2 * a_i
            B[0, c] = dN_dx
            B[1, c + 1] = dN_dy
            B[2, c] = dN_dy
            B[2, c + 1] = dN_dx

        return B, detJ

    def _jacobian(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dN_dxi: np.ndarray,
        dN_deta: np.ndarray,
    ) -> np.ndarray:
        """Return 2D isoparametric Jacobian."""
        return np.array(
            [[np.dot(dN_dxi, x), np.dot(dN_dxi, y)],
             [np.dot(dN_deta, x), np.dot(dN_deta, y)]],
            dtype=float,
        )

    def _checked_det_jacobian(self, elem: Any, J: np.ndarray) -> float:
        """Return detJ and reject singular or inverted elements."""
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(
                f"Quad8 element {elem.id} has non-positive Jacobian determinant "
                f"{detJ}; expected > 0"
            )
        return detJ

    def _det_jacobian(
        self,
        elem: Any,
        x: np.ndarray,
        y: np.ndarray,
        dN_dxi: np.ndarray,
        dN_deta: np.ndarray,
    ) -> float:
        """Return detJ from natural gradients."""
        return self._checked_det_jacobian(elem, self._jacobian(x, y, dN_dxi, dN_deta))

    def _edge_point(self, local_edge: int, s: float):
        """Map edge parameter to natural coordinates."""
        if local_edge == 0:
            return s, -1.0, 1.0, 0.0
        if local_edge == 1:
            return 1.0, s, 0.0, 1.0
        if local_edge == 2:
            return -s, 1.0, -1.0, 0.0
        return -1.0, -s, 0.0, -1.0

    def _edge_jacobian(
        self,
        elem: Any,
        x: np.ndarray,
        y: np.ndarray,
        dN_dxi: np.ndarray,
        dN_deta: np.ndarray,
        dxi_ds: float,
        deta_ds: float,
    ) -> float:
        """Return edge length scale from natural gradients."""
        dx_dxi = float(np.dot(dN_dxi, x))
        dy_dxi = float(np.dot(dN_dxi, y))
        dx_deta = float(np.dot(dN_deta, x))
        dy_deta = float(np.dot(dN_deta, y))
        jac = float(np.hypot(
            dx_dxi * dxi_ds + dx_deta * deta_ds,
            dy_dxi * dxi_ds + dy_deta * deta_ds,
        ))
        if jac == 0.0:
            raise ValueError(
                f"Quad8 element {elem.id} edge has zero Jacobian; expected > 0"
            )
        return jac
