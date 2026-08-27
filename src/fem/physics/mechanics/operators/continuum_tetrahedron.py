from __future__ import annotations

from typing import Any

import numpy as np

from .common import build_node_lookup
from ....materials import linear_elastic
from ....elements.tet4.definition import tet4_gauss_points, tet4_shape_funcs_grads
from ....elements.tet10.definition import (
    TET10_NATURAL_NODE_COORDS,
    tet10_gauss_points,
    tet10_shape_funcs_grads,
)
from ....elements.tri6.definition import tri6_gauss_points, tri6_shape_funcs_grads


TET4_CENTROID = (0.25, 0.25, 0.25)


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


class _TetLinearOperatorBase:
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
                f"{self.canonical_type} element {elem.id} requires {self.node_count} "
                f"nodes, got {len(elem.node_ids)}; node_ids={elem.node_ids}"
            )
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)

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

    def integration_point_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
        gauss_order: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return tetrahedral stress components at all integration points."""
        if gauss_order is not None:
            raise ValueError(
                f"gauss_order is not configurable for {self.type_names[0]}"
            )
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)
        gauss_points = self.gauss_points()
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


class Tet4LinearOperator(_TetLinearOperatorBase):
    """Tet4 solid linear operator."""
    canonical_type = "Tet4"
    aliases = ("C3D4",)
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
        _, integration_point_values = self.integration_point_stress(
            mesh, elem, U, node_lookup
        )
        return self.extrapolate_stress_to_nodes(integration_point_values)

    @staticmethod
    def extrapolate_stress_to_nodes(
        integration_point_values: np.ndarray,
        gauss_order: int | None = None,
    ) -> np.ndarray:
        """Replicate constant Tet4 stress to all four nodes."""
        if gauss_order is not None:
            raise ValueError("gauss_order is not configurable for Tet4")
        values = np.asarray(integration_point_values, dtype=float)
        if values.shape[0] != 1:
            raise ValueError(f"Tet4 requires 1 integration-point row, got {values.shape}")
        return np.tile(values[0], (4, 1))

    @staticmethod
    def interpolate_stress_to_centroid(
        integration_point_values: np.ndarray,
        gauss_order: int | None = None,
    ) -> np.ndarray:
        """Return the Tet4 centroid integration-point stress."""
        if gauss_order is not None:
            raise ValueError("gauss_order is not configurable for Tet4")
        values = np.asarray(integration_point_values, dtype=float)
        if values.shape[0] != 1:
            raise ValueError(f"Tet4 requires 1 integration-point row, got {values.shape}")
        return values[0].copy()

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


class Tet10LinearOperator(_TetLinearOperatorBase):
    """Tet10 solid linear operator."""
    canonical_type = "Tet10"
    aliases = ("C3D10",)
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
        _, integration_point_values = self.integration_point_stress(
            mesh, elem, U, node_lookup
        )
        return self.extrapolate_stress_to_nodes(integration_point_values)

    @staticmethod
    def extrapolate_stress_to_nodes(
        integration_point_values: np.ndarray,
        gauss_order: int | None = None,
    ) -> np.ndarray:
        """Linearly extrapolate four Hammer-point rows to ten nodes."""
        if gauss_order is not None:
            raise ValueError("gauss_order is not configurable for Tet10")
        values = np.asarray(integration_point_values, dtype=float)
        if values.shape[0] != 4:
            raise ValueError(f"Tet10 requires 4 integration-point rows, got {values.shape}")
        return TET10_LINEAR_EXTRAPOLATION_MATRIX @ values

    @staticmethod
    def interpolate_stress_to_centroid(
        integration_point_values: np.ndarray,
        gauss_order: int | None = None,
    ) -> np.ndarray:
        """Interpolate the symmetric Hammer-point field to the tetrahedron centroid."""
        if gauss_order is not None:
            raise ValueError("gauss_order is not configurable for Tet10")
        values = np.asarray(integration_point_values, dtype=float)
        if values.shape[0] != 4:
            raise ValueError(f"Tet10 requires 4 integration-point rows, got {values.shape}")
        return np.mean(values, axis=0)

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
