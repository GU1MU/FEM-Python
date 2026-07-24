from __future__ import annotations

from typing import Any

import numpy as np

from .base import build_node_lookup
from .quadrilateral import quad8_gauss_points, quad8_shape_funcs_grads
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


HEX8_NATURAL_GRADIENTS = np.asarray(
    [
        hex8_shape_funcs_grads(xi, eta, zeta)[1:]
        for xi, eta, zeta, _weight in hex8_gauss_points()
    ],
    dtype=float,
)


def _hex8_extrapolation_matrix() -> np.ndarray:
    """Return the constant 2x2x2 Gauss-to-node extrapolation matrix."""
    n_gp = np.array(
        [
            hex8_shape_funcs_grads(xi, eta, zeta)[0]
            for xi, eta, zeta, _ in hex8_gauss_points()
        ],
        dtype=float,
    )
    return np.linalg.solve(n_gp, np.eye(8, dtype=float))


HEX8_EXTRAPOLATION_MATRIX = _hex8_extrapolation_matrix()


HEX20_NATURAL_NODE_COORDS = np.array([
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [0.0, -1.0, -1.0],
    [1.0, 0.0, -1.0],
    [0.0, 1.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, -1.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [-1.0, 0.0, 1.0],
    [-1.0, -1.0, 0.0],
    [1.0, -1.0, 0.0],
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
], dtype=float)

HEX20_FACE_NODE_INDICES = [
    [0, 3, 2, 1, 11, 10, 9, 8],
    [4, 5, 6, 7, 12, 13, 14, 15],
    [0, 1, 5, 4, 8, 17, 12, 16],
    [2, 3, 7, 6, 10, 19, 14, 18],
    [0, 4, 7, 3, 16, 15, 19, 11],
    [1, 2, 6, 5, 9, 18, 13, 17],
]


def hex20_shape_funcs_grads(xi: float, eta: float, zeta: float):
    """Return Hex20 shape functions and natural-coordinate gradients."""
    N = np.zeros(20, dtype=float)
    dN_dxi = np.zeros(20, dtype=float)
    dN_deta = np.zeros(20, dtype=float)
    dN_dzeta = np.zeros(20, dtype=float)

    for i, (a, b, c) in enumerate(HEX20_NATURAL_NODE_COORDS):
        if a != 0.0 and b != 0.0 and c != 0.0:
            N[i] = (
                (1.0 + a * xi)
                * (1.0 + b * eta)
                * (1.0 + c * zeta)
                * (a * xi + b * eta + c * zeta - 2.0)
                / 8.0
            )
            dN_dxi[i] = (
                a
                * (1.0 + b * eta)
                * (1.0 + c * zeta)
                * (2.0 * a * xi + b * eta + c * zeta - 1.0)
                / 8.0
            )
            dN_deta[i] = (
                b
                * (1.0 + a * xi)
                * (1.0 + c * zeta)
                * (a * xi + 2.0 * b * eta + c * zeta - 1.0)
                / 8.0
            )
            dN_dzeta[i] = (
                c
                * (1.0 + a * xi)
                * (1.0 + b * eta)
                * (a * xi + b * eta + 2.0 * c * zeta - 1.0)
                / 8.0
            )
        elif a == 0.0:
            N[i] = (1.0 - xi**2) * (1.0 + b * eta) * (1.0 + c * zeta) / 4.0
            dN_dxi[i] = -xi * (1.0 + b * eta) * (1.0 + c * zeta) / 2.0
            dN_deta[i] = b * (1.0 - xi**2) * (1.0 + c * zeta) / 4.0
            dN_dzeta[i] = c * (1.0 - xi**2) * (1.0 + b * eta) / 4.0
        elif b == 0.0:
            N[i] = (1.0 - eta**2) * (1.0 + a * xi) * (1.0 + c * zeta) / 4.0
            dN_dxi[i] = a * (1.0 - eta**2) * (1.0 + c * zeta) / 4.0
            dN_deta[i] = -eta * (1.0 + a * xi) * (1.0 + c * zeta) / 2.0
            dN_dzeta[i] = c * (1.0 - eta**2) * (1.0 + a * xi) / 4.0
        else:
            N[i] = (1.0 - zeta**2) * (1.0 + a * xi) * (1.0 + b * eta) / 4.0
            dN_dxi[i] = a * (1.0 - zeta**2) * (1.0 + b * eta) / 4.0
            dN_deta[i] = b * (1.0 - zeta**2) * (1.0 + a * xi) / 4.0
            dN_dzeta[i] = -zeta * (1.0 + a * xi) * (1.0 + b * eta) / 2.0

    return N, dN_dxi, dN_deta, dN_dzeta


def hex20_gauss_points(gauss_order: int = 3):
    """Return the 3x3x3 full-integration rule for Hex20."""
    if gauss_order != 3:
        raise ValueError("gauss_order must be 3 for Hex20")
    r = np.sqrt(3.0 / 5.0)
    one_d = [(-r, 5.0 / 9.0), (0.0, 8.0 / 9.0), (r, 5.0 / 9.0)]
    return [
        (xi, eta, zeta, wx * wy * wz)
        for xi, wx in one_d
        for eta, wy in one_d
        for zeta, wz in one_d
    ]


def _hex20_extrapolation_matrix() -> np.ndarray:
    """Return the 27-Gauss-point-to-20-node least-squares matrix."""
    n_gp = np.array(
        [
            hex20_shape_funcs_grads(xi, eta, zeta)[0]
            for xi, eta, zeta, _ in hex20_gauss_points()
        ],
        dtype=float,
    )
    return np.linalg.pinv(n_gp)


HEX20_EXTRAPOLATION_MATRIX = _hex20_extrapolation_matrix()


class _HexKernelBase:
    """Shared material and geometry helpers for hexahedral kernels."""

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
        return [node_lookup[node_id] for node_id in elem.node_ids]

    def _coords(self, nodes: list[Any]):
        """Return coordinate arrays for element nodes."""
        x = np.array([n.x for n in nodes], dtype=float)
        y = np.array([n.y for n in nodes], dtype=float)
        z = np.array([n.z for n in nodes], dtype=float)
        return x, y, z

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
            raise ValueError(
                f"{elem.type} element {elem.id} has non-positive Jacobian "
                f"determinant {detJ}; expected > 0"
            )
        return detJ


class Hex8Kernel(_HexKernelBase):
    """Hex8 solid element kernel."""
    canonical_type = "Hex8"
    aliases = ("C3D8",)
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
            raise ValueError(
                f"Hex8 element {elem.id} requires 8 nodes, got {len(elem.node_ids)}; "
                f"node_ids={elem.node_ids}"
            )
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)

        D = self._material_matrix(elem)
        Ke = np.zeros((24, 24), dtype=float)
        integration = self._integration_matrices(
            mesh,
            elem,
            node_lookup,
            gauss_order,
        )
        b_volume = self._average_volumetric_operator(integration)
        for B, detJ, w in integration:
            B_bar = self._apply_bbar(B, b_volume)
            Ke += (B_bar.T @ D @ B_bar) * detJ * w

        return Ke

    def _integration_matrices(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any],
        gauss_order: int,
    ) -> list[tuple[np.ndarray, float, float]]:
        """Build all Hex8 integration matrices with one coordinate lookup."""
        gauss_points = hex8_gauss_points(gauss_order)
        nodes = self._nodes(mesh, elem, node_lookup)
        coordinates = np.asarray(
            [[node.x, node.y, node.z] for node in nodes],
            dtype=float,
        )
        natural_gradients = HEX8_NATURAL_GRADIENTS
        jacobians = natural_gradients @ coordinates
        determinants = np.linalg.det(jacobians)
        invalid = np.flatnonzero(determinants <= 0.0)
        if invalid.size:
            detJ = float(determinants[int(invalid[0])])
            raise ValueError(
                f"Hex8 element {elem.id} has non-positive Jacobian determinant "
                f"{detJ}; expected > 0"
            )

        global_gradients = np.linalg.inv(jacobians) @ natural_gradients
        matrices = np.zeros((8, 6, 24), dtype=float)
        offsets = 3 * np.arange(8)
        matrices[:, 0, offsets] = global_gradients[:, 0, :]
        matrices[:, 1, offsets + 1] = global_gradients[:, 1, :]
        matrices[:, 2, offsets + 2] = global_gradients[:, 2, :]
        matrices[:, 3, offsets] = global_gradients[:, 1, :]
        matrices[:, 3, offsets + 1] = global_gradients[:, 0, :]
        matrices[:, 4, offsets + 1] = global_gradients[:, 2, :]
        matrices[:, 4, offsets + 2] = global_gradients[:, 1, :]
        matrices[:, 5, offsets] = global_gradients[:, 2, :]
        matrices[:, 5, offsets + 2] = global_gradients[:, 0, :]
        return [
            (matrices[index], float(determinants[index]), float(weight))
            for index, (*_point, weight) in enumerate(gauss_points)
        ]

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
                raise ValueError(
                    f"Hex8 element {elem.id} face {local_face} has zero area; expected > 0"
                )

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
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)
        D = self._material_matrix(elem)
        B, _ = self._B_matrix(mesh, elem, xi, eta, zeta, node_lookup)
        integration = [
            (*self._B_matrix(mesh, elem, gx, gy, gz, node_lookup), weight)
            for gx, gy, gz, weight in hex8_gauss_points()
        ]
        B_bar = self._apply_bbar(B, self._average_volumetric_operator(integration))
        sigma = D @ (B_bar @ U[mesh.element_dofs(elem)])
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
        _, integration_point_values = self.integration_point_stress(
            mesh, elem, U, node_lookup, gauss_order
        )
        return self.extrapolate_stress_to_nodes(
            integration_point_values, gauss_order
        )

    def integration_point_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
        gauss_order: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return all Hex8 B-bar integration-point stresses in one pass."""
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)
        D = self._material_matrix(elem)
        gauss_points = hex8_gauss_points(gauss_order)
        integration = [
            (*self._B_matrix(mesh, elem, xi, eta, zeta, node_lookup), w)
            for xi, eta, zeta, w in gauss_points
        ]
        b_volume = self._average_volumetric_operator(integration)
        element_u = U[mesh.element_dofs(elem)]
        values = np.array([
            D @ (self._apply_bbar(B, b_volume) @ element_u)
            for B, _detJ, _weight in integration
        ], dtype=float)
        points = np.asarray(
            [(xi, eta, zeta) for xi, eta, zeta, _ in gauss_points],
            dtype=float,
        )
        return points, values

    @staticmethod
    def extrapolate_stress_to_nodes(
        integration_point_values: np.ndarray,
        gauss_order: int = 2,
    ) -> np.ndarray:
        """Extrapolate eight Hex8 integration-point rows to eight nodes."""
        hex8_gauss_points(gauss_order)
        values = np.asarray(integration_point_values, dtype=float)
        if values.shape[0] != 8:
            raise ValueError(f"Hex8 requires 8 integration-point rows, got {values.shape}")
        return HEX8_EXTRAPOLATION_MATRIX @ values

    @staticmethod
    def interpolate_stress_to_centroid(
        integration_point_values: np.ndarray,
        gauss_order: int = 2,
    ) -> np.ndarray:
        """Interpolate the symmetric Hex8 integration-point field to the centroid."""
        hex8_gauss_points(gauss_order)
        values = np.asarray(integration_point_values, dtype=float)
        if values.shape[0] != 8:
            raise ValueError(f"Hex8 requires 8 integration-point rows, got {values.shape}")
        return np.mean(values, axis=0)

    @staticmethod
    def _average_volumetric_operator(integration) -> np.ndarray:
        """Return the element-average volumetric strain operator for C3D8 B-bar."""
        volume = sum(detJ * weight for _B, detJ, weight in integration)
        if volume <= 0.0:
            raise ValueError("Hex8 element integration volume must be positive")
        return sum(
            (B[0] + B[1] + B[2]) * detJ * weight
            for B, detJ, weight in integration
        ) / volume

    @staticmethod
    def _apply_bbar(B: np.ndarray, average_volume: np.ndarray) -> np.ndarray:
        """Replace the pointwise volumetric operator while preserving deviatoric strain."""
        B_bar = B.copy()
        correction = (average_volume - (B[0] + B[1] + B[2])) / 3.0
        B_bar[0] += correction
        B_bar[1] += correction
        B_bar[2] += correction
        return B_bar

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
            raise ValueError(
                f"Hex8 element {elem.id} has non-positive Jacobian determinant "
                f"{detJ}; expected > 0"
            )

        invJ = np.linalg.inv(J)
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta + invJ[0, 2] * dN_dzeta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta + invJ[1, 2] * dN_dzeta
        dN_dz = invJ[2, 0] * dN_dxi + invJ[2, 1] * dN_deta + invJ[2, 2] * dN_dzeta

        B = np.zeros((6, 24), dtype=float)
        for local_node in range(8):
            dof_offset = 3 * local_node
            B[0, dof_offset] = dN_dx[local_node]
            B[1, dof_offset + 1] = dN_dy[local_node]
            B[2, dof_offset + 2] = dN_dz[local_node]
            B[3, dof_offset] = dN_dy[local_node]
            B[3, dof_offset + 1] = dN_dx[local_node]
            B[4, dof_offset + 1] = dN_dz[local_node]
            B[4, dof_offset + 2] = dN_dy[local_node]
            B[5, dof_offset] = dN_dz[local_node]
            B[5, dof_offset + 2] = dN_dx[local_node]

        return B, detJ


class Hex20Kernel(_HexKernelBase):
    """Twenty-node quadratic serendipity hexahedron kernel."""
    canonical_type = "Hex20"
    aliases = ("C3D20",)
    face_nodes = HEX20_FACE_NODE_INDICES

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
        gauss_order: int = 3,
    ) -> np.ndarray:
        if len(elem.node_ids) != 20:
            raise ValueError(
                f"Hex20 element {elem.id} requires 20 nodes, got {len(elem.node_ids)}; "
                f"node_ids={elem.node_ids}"
            )
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)
        D = self._material_matrix(elem)
        Ke = np.zeros((60, 60), dtype=float)
        for xi, eta, zeta, w in hex20_gauss_points(gauss_order):
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
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y, z = self._coords(nodes)
        bvec = np.asarray(vector, dtype=float)
        if bvec.shape != (3,):
            raise ValueError(f"Hex20 body force must have 3 components, got {bvec.shape}")
        fe = np.zeros(60, dtype=float)
        for xi, eta, zeta, w in hex20_gauss_points():
            N, dN_dxi, dN_deta, dN_dzeta = hex20_shape_funcs_grads(xi, eta, zeta)
            detJ = self._det_jacobian(elem, x, y, z, dN_dxi, dN_deta, dN_dzeta)
            for i in range(20):
                fe[3 * i:3 * i + 3] += N[i] * bvec * detJ * w
        return fe

    def face_traction(
        self,
        mesh: Any,
        elem: Any,
        local_face: int,
        traction: tuple[float, float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        if local_face < 0 or local_face >= 6:
            raise ValueError(f"Invalid local_face {local_face}, must be 0-5")
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)
        face_local = self.face_nodes[local_face]
        xyz = np.array([
            [
                node_lookup[elem.node_ids[i]].x,
                node_lookup[elem.node_ids[i]].y,
                node_lookup[elem.node_ids[i]].z,
            ]
            for i in face_local
        ])
        tvec = np.asarray(traction, dtype=float)
        fe = np.zeros(60, dtype=float)
        for xi, eta, w in quad8_gauss_points(3):
            N, dN_dxi, dN_deta = quad8_shape_funcs_grads(xi, eta)
            area_scale = float(np.linalg.norm(np.cross(dN_dxi @ xyz, dN_deta @ xyz)))
            if area_scale <= 0.0:
                raise ValueError(
                    f"Hex20 element {elem.id} face {local_face} has zero area; expected > 0"
                )
            for face_i, parent_i in enumerate(face_local):
                fe[3 * parent_i:3 * parent_i + 3] += (
                    N[face_i] * tvec * area_scale * w
                )
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
        D = self._material_matrix(elem)
        B, _ = self._B_matrix(mesh, elem, xi, eta, zeta, node_lookup)
        sigma = D @ (B @ U[mesh.element_dofs(elem)])
        return tuple(float(value) for value in sigma)

    def nodal_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
        gauss_order: int = 3,
    ) -> np.ndarray:
        _, integration_point_values = self.integration_point_stress(
            mesh, elem, U, node_lookup, gauss_order
        )
        return self.extrapolate_stress_to_nodes(
            integration_point_values, gauss_order
        )

    def integration_point_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
        gauss_order: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Hex20 stresses at all 3x3x3 integration points."""
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)
        gauss_points = hex20_gauss_points(gauss_order)
        points = np.asarray(
            [(xi, eta, zeta) for xi, eta, zeta, _ in gauss_points],
            dtype=float,
        )
        values = np.asarray([
            self.stress_at(
                mesh,
                elem,
                U,
                xi,
                eta,
                zeta,
                node_lookup=node_lookup,
            )
            for xi, eta, zeta in points
        ], dtype=float)
        return points, values

    @staticmethod
    def extrapolate_stress_to_nodes(
        integration_point_values: np.ndarray,
        gauss_order: int = 3,
    ) -> np.ndarray:
        """Recover twenty Hex20 nodal rows from 27 integration points."""
        hex20_gauss_points(gauss_order)
        values = np.asarray(integration_point_values, dtype=float)
        if values.shape[0] != 27:
            raise ValueError(f"Hex20 requires 27 integration-point rows, got {values.shape}")
        return HEX20_EXTRAPOLATION_MATRIX @ values

    @staticmethod
    def interpolate_stress_to_centroid(
        integration_point_values: np.ndarray,
        gauss_order: int = 3,
    ) -> np.ndarray:
        """Return the Hex20 stress at its centroid integration point."""
        hex20_gauss_points(gauss_order)
        values = np.asarray(integration_point_values, dtype=float)
        if values.shape[0] != 27:
            raise ValueError(f"Hex20 requires 27 integration-point rows, got {values.shape}")
        return values[13].copy()

    def _B_matrix(self, mesh, elem, xi, eta, zeta, node_lookup):
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y, z = self._coords(nodes)
        _, dN_dxi, dN_deta, dN_dzeta = hex20_shape_funcs_grads(xi, eta, zeta)
        J = np.array([
            [dN_dxi @ x, dN_dxi @ y, dN_dxi @ z],
            [dN_deta @ x, dN_deta @ y, dN_deta @ z],
            [dN_dzeta @ x, dN_dzeta @ y, dN_dzeta @ z],
        ], dtype=float)
        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            raise ValueError(
                f"Hex20 element {elem.id} has non-positive Jacobian determinant "
                f"{detJ}; expected > 0"
            )
        gradients = np.linalg.inv(J) @ np.vstack([dN_dxi, dN_deta, dN_dzeta])
        dN_dx, dN_dy, dN_dz = gradients
        B = np.zeros((6, 60), dtype=float)
        for i in range(20):
            j = 3 * i
            B[0, j] = dN_dx[i]
            B[1, j + 1] = dN_dy[i]
            B[2, j + 2] = dN_dz[i]
            B[3, j:j + 2] = [dN_dy[i], dN_dx[i]]
            B[4, j + 1:j + 3] = [dN_dz[i], dN_dy[i]]
            B[5, [j, j + 2]] = [dN_dz[i], dN_dx[i]]
        return B, detJ
