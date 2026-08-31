from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .common import build_node_lookup, tensor_product_recovery_matrix
from ...contracts import (
    LocalContribution,
    LocalOutputBatch,
    LocalPointOutput,
)
from .plane import PlaneProperties
from .continuum_quad4 import quad4_gauss_points, quad4_shape_grad_xi_eta
from ....elements.quad8.definition import quad8_gauss_points, quad8_shape_funcs_grads
from ....materials import linear_elastic

class Quad8LinearOperator:
    """Quad8 plane stress/strain linear operator."""
    canonical_type = "Quad8"
    aliases = ("CPS8", "CPE8")
    edge_node_indices = (
        (0, 4, 1),
        (1, 5, 2),
        (2, 6, 3),
        (3, 7, 0),
    )

    def linear_contribution(
        self,
        element: Any,
        reference_coordinates: np.ndarray,
        displacement: np.ndarray,
        dofs: tuple[int, ...],
        properties: Mapping[str, Any] | None = None,
        *,
        time: float = 0.0,
        temperature: float | None = None,
    ) -> LocalContribution:
        """Return the v2 stateless Quad8 contribution."""

        del time, temperature
        reference = _quad8_coordinates(
            reference_coordinates,
            "reference_coordinates",
        )
        local_displacement = _quad8_coordinates(displacement, "displacement")
        dof_tuple = tuple(int(value) for value in dofs)
        if len(dof_tuple) != 16:
            raise ValueError(
                f"Quad8 element {getattr(element, 'id', '?')} requires 16 DOFs; "
                f"got {len(dof_tuple)}"
            )
        source_properties = (
            getattr(element, "props", {}) if properties is None else properties
        )
        normalized = PlaneProperties.from_mapping(
            source_properties,
            element_id=getattr(element, "id", None),
            element_type=getattr(element, "type", None),
            label="Quad8 element",
        )
        D = linear_elastic.plane_matrix(
            normalized.E,
            normalized.nu,
            normalized.plane_type,
        )
        tangent = self._stiffness_from_reference_coordinates(
            element,
            reference,
            D,
            normalized.thickness,
            gauss_order=3,
        )
        records = []
        local_vector = local_displacement.reshape(-1)
        for point_id, (xi, eta, weight) in enumerate(
            quad8_gauss_points(3),
            start=1,
        ):
            B, detJ = self._B_matrix_from_coordinates(
                element,
                xi,
                eta,
                reference[:, 0],
                reference[:, 1],
            )
            strain = B @ local_vector
            records.append(
                LocalPointOutput.element_point(
                    element_id=int(element.id),
                    integration_point=point_id,
                    natural_coordinates=(float(xi), float(eta)),
                    weight=float(weight),
                    fields={
                        "engineering_strain": strain,
                        "cauchy_stress": D @ strain,
                        "jacobian": float(detJ),
                    },
                    history={},
                )
            )
        output_batch = LocalOutputBatch.element(int(element.id), tuple(records))
        return LocalContribution(
            dofs=dof_tuple,
            residual=tangent @ local_displacement.reshape(-1),
            tangent=tangent,
            outputs={"integration_points": output_batch.as_records()},
            output_batch=output_batch,
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
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)

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
        _, integration_point_values = self.integration_point_stress(
            mesh, elem, U, node_lookup, gauss_order
        )
        node_vals = self.extrapolate_stress_to_nodes(
            integration_point_values, gauss_order
        )
        plane_type, nu = self._plane_data(elem)
        return node_vals, plane_type, nu

    def integration_point_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
        gauss_order: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Quad8 stress components in integration-point order."""
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)
        gauss_points = quad8_gauss_points(gauss_order)
        points = np.asarray(
            [(xi, eta) for xi, eta, _ in gauss_points],
            dtype=float,
        )
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y = self._coords(nodes)
        D, _thickness = self._material_data(elem)
        element_u = U[mesh.element_dofs(elem)]
        values = np.asarray(
            [
                D
                @ (
                    self._B_matrix_from_coordinates(
                        elem,
                        xi,
                        eta,
                        x,
                        y,
                    )[0]
                    @ element_u
                )
                for xi, eta in points
            ],
            dtype=float,
        )
        return points, values

    @staticmethod
    def extrapolate_stress_to_nodes(
        integration_point_values: np.ndarray,
        gauss_order: int = 3,
    ) -> np.ndarray:
        """Extrapolate Quad8 integration-point components to eight nodes."""
        points = [(xi, eta) for xi, eta, _ in quad8_gauss_points(gauss_order)]
        targets = [
            (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
            (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0),
        ]
        values = np.asarray(integration_point_values, dtype=float)
        if values.shape[0] != len(points):
            raise ValueError(
                f"Quad8 requires {len(points)} integration-point rows, got {values.shape}"
            )
        return tensor_product_recovery_matrix(points, targets) @ values

    @staticmethod
    def interpolate_stress_to_centroid(
        integration_point_values: np.ndarray,
        gauss_order: int = 3,
    ) -> np.ndarray:
        """Interpolate Quad8 integration-point components to (0, 0)."""
        points = [(xi, eta) for xi, eta, _ in quad8_gauss_points(gauss_order)]
        values = np.asarray(integration_point_values, dtype=float)
        if values.shape[0] != len(points):
            raise ValueError(
                f"Quad8 requires {len(points)} integration-point rows, got {values.shape}"
            )
        return (tensor_product_recovery_matrix(points, [(0.0, 0.0)]) @ values)[0]

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

    def _stiffness_from_reference_coordinates(
        self,
        elem: Any,
        reference_coordinates: np.ndarray,
        D: np.ndarray,
        thickness: float,
        *,
        gauss_order: int,
    ) -> np.ndarray:
        """Integrate the Quad8 tangent from already-owned reference geometry."""

        reference = _quad8_coordinates(
            reference_coordinates,
            "reference_coordinates",
        )
        x = reference[:, 0]
        y = reference[:, 1]
        Ke = np.zeros((16, 16), dtype=float)
        for xi, eta, weight in quad8_gauss_points(gauss_order):
            B, detJ = self._B_matrix_from_coordinates(
                elem,
                xi,
                eta,
                x,
                y,
            )
            Ke += (B.T @ D @ B) * (thickness * detJ * weight)
        return Ke

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

        return self._B_matrix_from_coordinates(elem, xi, eta, x, y)

    def _B_matrix_from_coordinates(
        self,
        elem: Any,
        xi: float,
        eta: float,
        x: np.ndarray,
        y: np.ndarray,
    ):
        """Return B and detJ using coordinates already gathered for an element."""

        _, dN_dxi, dN_deta = quad8_shape_funcs_grads(xi, eta)
        J = self._jacobian(x, y, dN_dxi, dN_deta)
        detJ = self._checked_det_jacobian(elem, J)

        inverse = np.asarray(
            ((J[1, 1], -J[0, 1]), (-J[1, 0], J[0, 0])),
            dtype=float,
        ) / detJ
        dN_xy = inverse @ np.vstack([dN_dxi, dN_deta])
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
        detJ = float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
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


def _quad8_coordinates(value: Any, name: str) -> np.ndarray:
    """Validate finite two-component coordinates for an eight-node plane element."""

    try:
        coordinates = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if coordinates.shape != (8, 2) or not np.all(np.isfinite(coordinates)):
        raise ValueError(
            f"{name} must be a finite array with shape (8, 2); "
            f"got {coordinates.shape}"
        )
    return coordinates
