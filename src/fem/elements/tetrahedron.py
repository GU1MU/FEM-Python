from __future__ import annotations

from typing import Any

import numpy as np

from .base import build_node_lookup
from .triangle import tri6_gauss_points, tri6_shape_funcs_grads
from ..materials import linear_elastic


def tet4_shape_funcs_grads(xi: float, eta: float, zeta: float):
    """Return N and natural gradients for Tet4."""
    N = np.array([
        1.0 - xi - eta - zeta,
        xi,
        eta,
        zeta,
    ], dtype=float)
    dN_dxi = np.array([-1.0, 1.0, 0.0, 0.0], dtype=float)
    dN_deta = np.array([-1.0, 0.0, 1.0, 0.0], dtype=float)
    dN_dzeta = np.array([-1.0, 0.0, 0.0, 1.0], dtype=float)
    return N, dN_dxi, dN_deta, dN_dzeta


def tet4_gauss_points():
    """Return centroid integration point for Tet4."""
    return [(0.25, 0.25, 0.25, 1.0 / 6.0)]


TET4_CENTROID = (0.25, 0.25, 0.25)


def tet10_shape_funcs_grads(xi: float, eta: float, zeta: float):
    """Return N and natural gradients for Tet10."""
    L1 = 1.0 - xi - eta - zeta
    L2 = xi
    L3 = eta
    L4 = zeta

    N = np.zeros(10, dtype=float)
    dN_dxi = np.zeros(10, dtype=float)
    dN_deta = np.zeros(10, dtype=float)
    dN_dzeta = np.zeros(10, dtype=float)

    N[0] = (2.0 * L1 - 1.0) * L1
    N[1] = (2.0 * L2 - 1.0) * L2
    N[2] = (2.0 * L3 - 1.0) * L3
    N[3] = (2.0 * L4 - 1.0) * L4
    N[4] = 4.0 * L1 * L2
    N[5] = 4.0 * L2 * L3
    N[6] = 4.0 * L1 * L3
    N[7] = 4.0 * L1 * L4
    N[8] = 4.0 * L2 * L4
    N[9] = 4.0 * L3 * L4

    dL1_dxi = -1.0
    dL1_deta = -1.0
    dL1_dzeta = -1.0
    dL2_dxi = 1.0
    dL2_deta = 0.0
    dL2_dzeta = 0.0
    dL3_dxi = 0.0
    dL3_deta = 1.0
    dL3_dzeta = 0.0
    dL4_dxi = 0.0
    dL4_deta = 0.0
    dL4_dzeta = 1.0

    dN_dxi[0] = (4.0 * L1 - 1.0) * dL1_dxi
    dN_deta[0] = (4.0 * L1 - 1.0) * dL1_deta
    dN_dzeta[0] = (4.0 * L1 - 1.0) * dL1_dzeta
    dN_dxi[1] = (4.0 * L2 - 1.0) * dL2_dxi
    dN_deta[1] = (4.0 * L2 - 1.0) * dL2_deta
    dN_dzeta[1] = (4.0 * L2 - 1.0) * dL2_dzeta
    dN_dxi[2] = (4.0 * L3 - 1.0) * dL3_dxi
    dN_deta[2] = (4.0 * L3 - 1.0) * dL3_deta
    dN_dzeta[2] = (4.0 * L3 - 1.0) * dL3_dzeta
    dN_dxi[3] = (4.0 * L4 - 1.0) * dL4_dxi
    dN_deta[3] = (4.0 * L4 - 1.0) * dL4_deta
    dN_dzeta[3] = (4.0 * L4 - 1.0) * dL4_dzeta

    dN_dxi[4] = 4.0 * (L2 * dL1_dxi + L1 * dL2_dxi)
    dN_deta[4] = 4.0 * (L2 * dL1_deta + L1 * dL2_deta)
    dN_dzeta[4] = 4.0 * (L2 * dL1_dzeta + L1 * dL2_dzeta)
    dN_dxi[5] = 4.0 * (L3 * dL2_dxi + L2 * dL3_dxi)
    dN_deta[5] = 4.0 * (L3 * dL2_deta + L2 * dL3_deta)
    dN_dzeta[5] = 4.0 * (L3 * dL2_dzeta + L2 * dL3_dzeta)
    dN_dxi[6] = 4.0 * (L3 * dL1_dxi + L1 * dL3_dxi)
    dN_deta[6] = 4.0 * (L3 * dL1_deta + L1 * dL3_deta)
    dN_dzeta[6] = 4.0 * (L3 * dL1_dzeta + L1 * dL3_dzeta)
    dN_dxi[7] = 4.0 * (L4 * dL1_dxi + L1 * dL4_dxi)
    dN_deta[7] = 4.0 * (L4 * dL1_deta + L1 * dL4_deta)
    dN_dzeta[7] = 4.0 * (L4 * dL1_dzeta + L1 * dL4_dzeta)
    dN_dxi[8] = 4.0 * (L4 * dL2_dxi + L2 * dL4_dxi)
    dN_deta[8] = 4.0 * (L4 * dL2_deta + L2 * dL4_deta)
    dN_dzeta[8] = 4.0 * (L4 * dL2_dzeta + L2 * dL4_dzeta)
    dN_dxi[9] = 4.0 * (L4 * dL3_dxi + L3 * dL4_dxi)
    dN_deta[9] = 4.0 * (L4 * dL3_deta + L3 * dL4_deta)
    dN_dzeta[9] = 4.0 * (L4 * dL3_dzeta + L3 * dL4_dzeta)

    return N, dN_dxi, dN_deta, dN_dzeta


def tet10_gauss_points():
    """Return 4-point Hammer integration rule for Tet10."""
    n = 0.58541020
    a = (1.0 - n) / 4.0
    b = (1.0 + 3.0 * n) / 4.0
    w = 1.0 / 24.0
    return [
        (a, a, a, w),
        (b, a, a, w),
        (a, b, a, w),
        (a, a, b, w),
    ]


TET10_NATURAL_NODE_COORDS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
]


def _tet10_linear_extrapolation_matrix() -> np.ndarray:
    """Return the constant Hammer-point linear-fit matrix for Tet10 nodes."""
    gp_coords = [(xi, eta, zeta) for xi, eta, zeta, _ in tet10_gauss_points()]
    a_gp = np.array([[1.0, xi, eta, zeta] for xi, eta, zeta in gp_coords], dtype=float)
    a_nodes = np.array(
        [[1.0, xi, eta, zeta] for xi, eta, zeta in TET10_NATURAL_NODE_COORDS],
        dtype=float,
    )
    return a_nodes @ np.linalg.solve(a_gp, np.eye(4, dtype=float))


TET10_LINEAR_EXTRAPOLATION_MATRIX = _tet10_linear_extrapolation_matrix()


def tet_physical_shape_gradients(
    elem: Any,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dN_dxi: np.ndarray,
    dN_deta: np.ndarray,
    dN_dzeta: np.ndarray,
):
    """Map tetra shape gradients to physical coordinates."""
    J = np.array([
        [np.sum(dN_dxi * x), np.sum(dN_dxi * y), np.sum(dN_dxi * z)],
        [np.sum(dN_deta * x), np.sum(dN_deta * y), np.sum(dN_deta * z)],
        [np.sum(dN_dzeta * x), np.sum(dN_dzeta * y), np.sum(dN_dzeta * z)],
    ], dtype=float)

    detJ = float(np.linalg.det(J))
    if detJ <= 0.0:
        raise ValueError(
            f"Element {elem.id} has non-positive Jacobian determinant {detJ}; "
            "expected > 0"
        )

    invJ = np.linalg.inv(J)
    dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta + invJ[0, 2] * dN_dzeta
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta + invJ[1, 2] * dN_dzeta
    dN_dz = invJ[2, 0] * dN_dxi + invJ[2, 1] * dN_deta + invJ[2, 2] * dN_dzeta
    return dN_dx, dN_dy, dN_dz, detJ


def build_tet_B_matrix(dN_dx: np.ndarray, dN_dy: np.ndarray, dN_dz: np.ndarray) -> np.ndarray:
    """Return 3D strain-displacement matrix for tetra nodes."""
    node_count = len(dN_dx)
    B = np.zeros((6, node_count * 3), dtype=float)
    for local_node in range(node_count):
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
    return B


class _TetKernelBase:
    """Shared tetrahedral solid element logic."""
    node_count: int
    gauss_points: Any
    shape_funcs_grads: Any

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return tetrahedral element stiffness."""
        if len(elem.node_ids) != self.node_count:
            raise ValueError(
                f"{self.type_names[0]} element {elem.id} requires {self.node_count} "
                f"nodes, got {len(elem.node_ids)}; node_ids={elem.node_ids}"
            )

        D = self._material_matrix(elem)
        Ke = np.zeros((self.node_count * 3, self.node_count * 3), dtype=float)
        for xi, eta, zeta, w in self.gauss_points():
            B, detJ = self._B_matrix(mesh, elem, xi, eta, zeta, node_lookup)
            Ke += (B.T @ D @ B) * detJ * w
        return Ke

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

    def volume(self, mesh: Any, elem: Any, node_lookup: dict[int, Any] | None = None) -> float:
        """Return element volume using the stiffness integration rule."""
        volume = 0.0
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y, z = self._coords(nodes)
        for xi, eta, zeta, w in self.gauss_points():
            _, dN_dxi, dN_deta, dN_dzeta = self.shape_funcs_grads(xi, eta, zeta)
            _, _, _, detJ = tet_physical_shape_gradients(
                elem, x, y, z, dN_dxi, dN_deta, dN_dzeta
            )
            volume += detJ * w
        return volume

    def body_force(
        self,
        mesh: Any,
        elem: Any,
        vector: tuple[float, float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return consistent tetrahedral body force vector."""
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y, z = self._coords(nodes)
        bvec = np.array(vector, dtype=float)
        fe = np.zeros(self.node_count * 3, dtype=float)

        for xi, eta, zeta, w in self.gauss_points():
            N, dN_dxi, dN_deta, dN_dzeta = self.shape_funcs_grads(xi, eta, zeta)
            _, _, _, detJ = tet_physical_shape_gradients(
                elem, x, y, z, dN_dxi, dN_deta, dN_dzeta
            )
            for i in range(self.node_count):
                fe[3 * i:3 * i + 3] += N[i] * bvec * (detJ * w)
        return fe

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
        _, dN_dxi, dN_deta, dN_dzeta = self.shape_funcs_grads(xi, eta, zeta)
        dN_dx, dN_dy, dN_dz, detJ = tet_physical_shape_gradients(
            elem, x, y, z, dN_dxi, dN_deta, dN_dzeta
        )
        return build_tet_B_matrix(dN_dx, dN_dy, dN_dz), detJ


class Tet4Kernel(_TetKernelBase):
    """Tet4 solid element kernel."""
    type_names = ("Tet4", "C3D4")
    node_count = 4
    gauss_points = staticmethod(tet4_gauss_points)
    shape_funcs_grads = staticmethod(tet4_shape_funcs_grads)
    face_node_indices = [
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2],
    ]

    def nodal_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return element-nodal stresses using constant Tet4 stress."""
        stress = self.stress_at(mesh, elem, U, *TET4_CENTROID, node_lookup)
        return np.tile(stress, (self.node_count, 1))

    def face_traction(
        self,
        mesh: Any,
        elem: Any,
        local_face: int,
        traction: tuple[float, float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return consistent Tet4 face traction vector."""
        if local_face < 0 or local_face >= 4:
            raise ValueError(f"Invalid local_face {local_face}, must be 0-3 for Tet4")
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)

        face_local = self.face_node_indices[local_face]
        face_nodes = [node_lookup[elem.node_ids[i]] for i in face_local]
        p1 = np.array([face_nodes[0].x, face_nodes[0].y, face_nodes[0].z], dtype=float)
        p2 = np.array([face_nodes[1].x, face_nodes[1].y, face_nodes[1].z], dtype=float)
        p3 = np.array([face_nodes[2].x, face_nodes[2].y, face_nodes[2].z], dtype=float)
        area = 0.5 * float(np.linalg.norm(np.cross(p2 - p1, p3 - p1)))
        if area <= 0.0:
            raise ValueError(
                f"Tet4 element {elem.id} face {local_face} has zero area; expected > 0"
            )

        tvec = np.array(traction, dtype=float)
        fe = np.zeros(12, dtype=float)
        for parent_local in face_local:
            fe[3 * parent_local:3 * parent_local + 3] += tvec * (area / 3.0)
        return fe


class Tet10Kernel(_TetKernelBase):
    """Tet10 solid element kernel."""
    type_names = ("Tet10", "C3D10")
    node_count = 10
    gauss_points = staticmethod(tet10_gauss_points)
    shape_funcs_grads = staticmethod(tet10_shape_funcs_grads)
    face_node_indices = [
        [1, 2, 3, 5, 9, 8],
        [0, 2, 3, 6, 9, 7],
        [0, 1, 3, 4, 8, 7],
        [0, 1, 2, 4, 5, 6],
    ]

    def nodal_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return element-nodal stresses from Hammer-point linear extrapolation."""
        gp_vals = np.array([
            self.stress_at(mesh, elem, U, xi, eta, zeta, node_lookup)
            for xi, eta, zeta, _ in self.gauss_points()
        ], dtype=float)
        return TET10_LINEAR_EXTRAPOLATION_MATRIX @ gp_vals

    def face_traction(
        self,
        mesh: Any,
        elem: Any,
        local_face: int,
        traction: tuple[float, float, float],
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return consistent Tet10 face traction vector."""
        if local_face < 0 or local_face >= 4:
            raise ValueError(f"Invalid local_face {local_face}, must be 0-3 for Tet10")
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)

        face_local = self.face_node_indices[local_face]
        face_nodes = [node_lookup[elem.node_ids[i]] for i in face_local]
        face_xyz = np.array([[n.x, n.y, n.z] for n in face_nodes], dtype=float)
        tvec = np.array(traction, dtype=float)
        fe = np.zeros(30, dtype=float)

        for xi, eta, w in tri6_gauss_points():
            N, dN_dxi, dN_deta = tri6_shape_funcs_grads(xi, eta)
            area_scale = float(np.linalg.norm(np.cross(dN_dxi @ face_xyz, dN_deta @ face_xyz)))
            if area_scale <= 0.0:
                raise ValueError(
                    f"Tet10 element {elem.id} face {local_face} has zero area; expected > 0"
                )
            for i, parent_local in enumerate(face_local):
                fe[3 * parent_local:3 * parent_local + 3] += N[i] * tvec * (area_scale * w)
        return fe
