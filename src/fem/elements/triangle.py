from __future__ import annotations

from typing import Any

import numpy as np

from .base import build_node_lookup
from ..materials import linear_elastic


def _plane_type(elem: Any) -> str:
    """Return the plane formulation, inferring standard Abaqus aliases."""
    raw_value = elem.props.get("plane_type")
    if raw_value is None:
        raw_value = "strain" if str(elem.type).upper().startswith("CPE") else "stress"
    plane_type = str(raw_value).lower()
    if plane_type.startswith("stress"):
        return "stress"
    if plane_type.startswith("strain"):
        return "strain"
    raise ValueError(
        f"Element {elem.id} has plane_type {raw_value!r}; "
        "expected 'stress' or 'strain'"
    )


def _positive_thickness(elem: Any) -> float:
    """Return a finite positive plane-element thickness."""
    raw_value = elem.props.get("thickness", 1.0)
    thickness = float(raw_value)
    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValueError(
            f"Element {elem.id} property thickness must be finite and > 0, "
            f"got {raw_value!r}"
        )
    return thickness


class Tri3Kernel:
    """Tri3 plane stress/strain element kernel."""
    canonical_type = "Tri3"
    aliases = ("CPS3", "CPE3")
    edge_node_indices = ((0, 1), (1, 2), (2, 0))

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return Tri3 plane element stiffness."""
        B, area = self._B_matrix(mesh, elem, node_lookup)
        D, t = self._material_data(elem)
        return t * area * (B.T @ D @ B)

    def stress_at(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return constant Tri3 stress."""
        B, _ = self._B_matrix(mesh, elem, node_lookup)
        D, _ = self._material_data(elem)
        return D @ (B @ U[mesh.element_dofs(elem)])

    def nodal_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
    ):
        """Return element-nodal stress, plane type, and nu."""
        sigma = self.stress_at(mesh, elem, U, node_lookup)
        plane_type, nu = self._plane_data(elem)
        return np.tile(sigma, (3, 1)), plane_type, nu

    def body_force(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return consistent Tri3 body force vector."""
        _, area = self._B_matrix(mesh, elem, node_lookup)
        t = self._thickness(elem)
        bvec = np.array(vector, dtype=float)
        fe = np.zeros(6, dtype=float)
        for i in range(3):
            fe[2 * i:2 * i + 2] = bvec * (t * area / 3.0)
        return fe

    def edge_traction(
        self,
        mesh: Any,
        elem: Any,
        local_edge: int,
        traction: tuple[float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return consistent Tri3 edge traction vector."""
        if local_edge < 0 or local_edge >= 3:
            raise ValueError(f"Tri3 local_edge must be 0/1/2, got {local_edge}")
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)

        i, j = self.edge_node_indices[local_edge]
        ni = node_lookup[elem.node_ids[i]]
        nj = node_lookup[elem.node_ids[j]]
        length = float(np.hypot(nj.x - ni.x, nj.y - ni.y))
        if length <= 0.0:
            raise ValueError(f"Tri3 element {elem.id} edge length is zero; expected > 0")

        t = self._thickness(elem)
        tvec = np.array(traction, dtype=float)
        fe = np.zeros(6, dtype=float)
        for local_i in (i, j):
            fe[2 * local_i:2 * local_i + 2] += tvec * (t * length / 2.0)
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
        return _plane_type(elem), nu

    def _thickness(self, elem: Any) -> float:
        """Return plane element thickness."""
        return _positive_thickness(elem)

    def _B_matrix(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None,
    ):
        """Return B matrix and area for Tri3."""
        if len(elem.node_ids) != 3:
            raise ValueError(
                f"Tri3 element {elem.id} requires 3 nodes, got {len(elem.node_ids)}; "
                f"node_ids={elem.node_ids}"
            )
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)

        try:
            n1, n2, n3 = (node_lookup[node_id] for node_id in elem.node_ids)
        except KeyError as exc:
            raise KeyError(
                f"Element {elem.id} references missing node {exc.args[0]}"
            ) from exc

        x1, y1 = n1.x, n1.y
        x2, y2 = n2.x, n2.y
        x3, y3 = n3.x, n3.y

        detJ = (
            x2 * y3 - x3 * y2
            - x1 * y3 + x3 * y1
            + x1 * y2 - x2 * y1
        )
        area = 0.5 * detJ
        if area <= 0.0:
            raise ValueError(
                f"Tri3 element {elem.id} has non-positive signed area {area}; "
                "expected counter-clockwise, non-degenerate node ordering"
            )

        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1

        B = (1.0 / (2.0 * area)) * np.array([
            [b1, 0.0, b2, 0.0, b3, 0.0],
            [0.0, c1, 0.0, c2, 0.0, c3],
            [c1, b1, c2, b2, c3, b3],
        ], dtype=float)
        return B, area


def tri6_shape_funcs_grads(xi: float, eta: float):
    """Return N, dN/dxi, and dN/deta for quadratic Tri6."""
    l1 = 1.0 - xi - eta
    l2 = xi
    l3 = eta

    N = np.array([
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        l3 * (2.0 * l3 - 1.0),
        4.0 * l1 * l2,
        4.0 * l2 * l3,
        4.0 * l3 * l1,
    ], dtype=float)

    dN_dxi = np.array([
        1.0 - 4.0 * l1,
        4.0 * l2 - 1.0,
        0.0,
        4.0 * (l1 - l2),
        4.0 * l3,
        -4.0 * l3,
    ], dtype=float)

    dN_deta = np.array([
        1.0 - 4.0 * l1,
        0.0,
        4.0 * l3 - 1.0,
        -4.0 * l2,
        4.0 * l2,
        4.0 * (l1 - l3),
    ], dtype=float)

    return N, dN_dxi, dN_deta


def tri6_gauss_points():
    """Return reference-triangle Gauss points for Tri6."""
    return [
        (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    ]


def tri6_edge_gauss_points():
    """Return 3-point Gauss rule on [-1, 1] for quadratic edges."""
    r = np.sqrt(3.0 / 5.0)
    return [(-r, 5.0 / 9.0), (0.0, 8.0 / 9.0), (r, 5.0 / 9.0)]


class Tri6Kernel:
    """Tri6 plane stress/strain element kernel."""
    canonical_type = "Tri6"
    aliases = ("CPS6", "CPE6")
    edge_node_indices = ((0, 3, 1), (1, 4, 2), (2, 5, 0))

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return Tri6 plane element stiffness."""
        if len(elem.node_ids) != 6:
            raise ValueError(
                f"Tri6 element {elem.id} requires 6 nodes, got {len(elem.node_ids)}; "
                f"node_ids={elem.node_ids}"
            )

        D, t = self._material_data(elem)
        Ke = np.zeros((12, 12), dtype=float)
        for xi, eta, w in tri6_gauss_points():
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
        gauss_order: int | None = None,
    ):
        """Return element-nodal stress, plane type, and nu."""
        if gauss_order not in (None, 3):
            raise ValueError("gauss_order must be 3 for Tri6 nodal stress")
        node_coords = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.0, 0.5),
        ]
        node_vals = np.array(
            [
                self.stress_at(mesh, elem, U, xi, eta, node_lookup)
                for xi, eta in node_coords
            ],
            dtype=float,
        )
        plane_type, nu = self._plane_data(elem)
        return node_vals, plane_type, nu

    def body_force(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return consistent Tri6 body force vector."""
        if len(elem.node_ids) != 6:
            raise ValueError(
                f"Tri6 element {elem.id} requires 6 nodes, got {len(elem.node_ids)}; "
                f"node_ids={elem.node_ids}"
            )

        t = self._thickness(elem)
        bvec = np.array(vector, dtype=float)
        fe = np.zeros(12, dtype=float)
        for xi, eta, w in tri6_gauss_points():
            N, _, _ = tri6_shape_funcs_grads(xi, eta)
            _, detJ = self._B_matrix(mesh, elem, xi, eta, node_lookup)
            for i in range(6):
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
        """Return consistent Tri6 edge traction vector."""
        if local_edge < 0 or local_edge >= 3:
            raise ValueError(f"Tri6 local_edge must be 0/1/2, got {local_edge}")
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)

        local_ids = self.edge_node_indices[local_edge]
        nodes = [node_lookup[elem.node_ids[i]] for i in local_ids]
        x = np.array([node.x for node in nodes], dtype=float)
        y = np.array([node.y for node in nodes], dtype=float)
        t = self._thickness(elem)
        tvec = np.array(traction, dtype=float)
        fe = np.zeros(12, dtype=float)

        for s, w in tri6_edge_gauss_points():
            N = np.array([
                0.5 * s * (s - 1.0),
                1.0 - s * s,
                0.5 * s * (s + 1.0),
            ], dtype=float)
            dN_ds = np.array([s - 0.5, -2.0 * s, s + 0.5], dtype=float)
            jac = float(np.hypot(np.dot(dN_ds, x), np.dot(dN_ds, y)))
            if jac <= 0.0:
                raise ValueError(
                    f"Tri6 element {elem.id} edge has zero Jacobian; expected > 0"
                )
            for edge_pos, local_i in enumerate(local_ids):
                fe[2 * local_i:2 * local_i + 2] += N[edge_pos] * tvec * (t * jac * w)
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
        return _plane_type(elem), nu

    def _thickness(self, elem: Any) -> float:
        """Return plane element thickness."""
        return _positive_thickness(elem)

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
        nodes = [node_lookup[node_id] for node_id in elem.node_ids]
        x = np.array([node.x for node in nodes], dtype=float)
        y = np.array([node.y for node in nodes], dtype=float)

        _, dN_dxi, dN_deta = tri6_shape_funcs_grads(xi, eta)
        J = np.array(
            [
                [np.dot(dN_dxi, x), np.dot(dN_dxi, y)],
                [np.dot(dN_deta, x), np.dot(dN_deta, y)],
            ],
            dtype=float,
        )
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(
                f"Tri6 element {elem.id} has non-positive Jacobian determinant "
                f"{detJ}; expected > 0"
            )

        dN_xy = np.linalg.inv(J) @ np.vstack([dN_dxi, dN_deta])
        B = np.zeros((3, 12), dtype=float)
        for a_i in range(6):
            dN_dx = dN_xy[0, a_i]
            dN_dy = dN_xy[1, a_i]
            c = 2 * a_i
            B[0, c] = dN_dx
            B[1, c + 1] = dN_dy
            B[2, c] = dN_dy
            B[2, c + 1] = dN_dx
        return B, detJ
