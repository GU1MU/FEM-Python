from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np

from .common import build_node_lookup, tensor_product_recovery_matrix
from ...contracts import (
    LocalContribution,
    LocalOutputBatch,
    LocalPointOutput,
)
from ....materials import linear_elastic
from ....state import EvaluationContext, LocalFieldState, StateManager
from ....elements.quad4.definition import (
    Quad4Definition,
    quad4_gauss_points,
    quad4_shape_functions,
    quad4_shape_grad_xi_eta,
)
from .plane import PlaneProperties, plane_thickness


@lru_cache(maxsize=2)
def _quad4_gauss_gradients(gauss_order: int) -> tuple[np.ndarray, ...]:
    """Return immutable natural gradients for one supported Gauss rule."""

    gradients = []
    for xi, eta, _weight in quad4_gauss_points(gauss_order):
        value = np.asarray(quad4_shape_grad_xi_eta(xi, eta), dtype=float)
        value.flags.writeable = False
        gradients.append(value)
    return tuple(gradients)


@lru_cache(maxsize=2)
def _quad4_extrapolation_matrix(gauss_order: int) -> np.ndarray:
    """Return the immutable Gauss-to-node recovery matrix."""

    points = [(xi, eta) for xi, eta, _ in quad4_gauss_points(gauss_order)]
    targets = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    matrix = np.asarray(
        tensor_product_recovery_matrix(points, targets),
        dtype=float,
    )
    matrix.flags.writeable = False
    return matrix


@lru_cache(maxsize=2)
def _quad4_centroid_recovery_matrix(gauss_order: int) -> np.ndarray:
    """Return the immutable Gauss-to-centroid recovery matrix."""

    points = [(xi, eta) for xi, eta, _ in quad4_gauss_points(gauss_order)]
    matrix = np.asarray(
        tensor_product_recovery_matrix(points, [(0.0, 0.0)]),
        dtype=float,
    )
    matrix.flags.writeable = False
    return matrix


def _quad4_coordinates(value: Any, name: str) -> np.ndarray:
    """Validate one finite Quad4 coordinate or displacement array."""

    try:
        coordinates = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if coordinates.ndim != 2 or not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{name} must be a finite 2D array")
    return coordinates


class Quad4LinearOperator:
    """Linear small-strain operator for the Quad4 plane element."""
    canonical_type = "Quad4"
    aliases = ("CPS4", "CPE4")
    edge_node_indices = ((0, 1), (1, 2), (2, 3), (3, 0))

    def initialize(
        self,
        element: Any,
        resources: Mapping[str, Any],
        state: StateManager | None,
        properties: Mapping[str, Any] | None = None,
        *,
        state_namespace: str,
    ) -> None:
        del element, resources, state, properties, state_namespace

    def evaluate(
        self,
        element: Any,
        reference_coordinates: np.ndarray,
        fields: Mapping[str, LocalFieldState],
        dofs: tuple[int, ...],
        resources: Mapping[str, Any],
        state: StateManager | None,
        properties: Mapping[str, Any] | None = None,
        *,
        context: EvaluationContext,
        state_namespace: str,
    ) -> LocalContribution:
        """Return the stateless small-strain Quad4 contribution.

        The local operator only consumes reference geometry, local
        displacement and an explicit property mapping.
        """

        del resources, state, context, state_namespace
        try:
            displacement = fields["U"].values
        except KeyError as exc:
            raise ValueError("linear solid mechanics requires local field 'U'") from exc
        reference = _quad4_coordinates(
            reference_coordinates,
            "reference_coordinates",
        )
        if reference.shape != (4, 2):
            raise ValueError(
                "Quad4 reference_coordinates must have shape (4, 2); "
                f"got {reference.shape}"
            )
        local_displacement = _quad4_coordinates(displacement, "displacement")
        if local_displacement.shape != (4, 2):
            raise ValueError(
                "Quad4 displacement must have shape (4, 2); "
                f"got {local_displacement.shape}"
            )
        dof_tuple = tuple(int(value) for value in dofs)
        if len(dof_tuple) != 8:
            raise ValueError(
                f"Quad4 element {getattr(element, 'id', '?')} requires 8 DOFs; "
                f"got {len(dof_tuple)}"
            )
        source_properties = (
            getattr(element, "props", {}) if properties is None else properties
        )
        normalized = PlaneProperties.from_mapping(
            source_properties,
            element_id=getattr(element, "id", None),
            element_type=getattr(element, "type", None),
        )
        D = linear_elastic.plane_matrix(
            normalized.E,
            normalized.nu,
            normalized.plane_type,
        )
        tangent = self._stiffness_from_reference_coordinates(
            element,
            reference,
            normalized,
            gauss_order=2,
        )
        records = []
        local_vector = local_displacement.reshape(-1)
        for point_id, (xi, eta, weight) in enumerate(
            quad4_gauss_points(2),
            start=1,
        ):
            B, detJ = self._B_matrix_from_coordinates(
                element,
                xi,
                eta,
                reference[:, 0],
                reference[:, 1],
            )
            records.append(
                LocalPointOutput.element_point(
                    element_id=int(element.id),
                    integration_point=point_id,
                    natural_coordinates=(float(xi), float(eta)),
                    weight=float(weight),
                    fields={
                        "engineering_strain": B @ local_vector,
                        "cauchy_stress": D @ (B @ local_vector),
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
        gauss_order: int = 2,
    ) -> np.ndarray:
        """Return Quad4 plane element stiffness."""
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y = self._coords(nodes)
        properties = PlaneProperties.from_mapping(
            elem.props,
            element_id=elem.id,
            element_type=getattr(elem, "type", None),
        )
        return self._stiffness_from_reference_coordinates(
            elem,
            np.column_stack((x, y)),
            properties,
            gauss_order=gauss_order,
        )

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
        gauss_order: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Quad4 stress components in integration-point order."""
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)
        gauss_points = quad4_gauss_points(gauss_order)
        points = np.asarray(
            [(xi, eta) for xi, eta, _ in gauss_points],
            dtype=float,
        )
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y = self._coords(nodes)
        D, _thickness = self._material_data(elem)
        element_u = U[mesh.element_dofs(elem)]
        gradients = _quad4_gauss_gradients(gauss_order)
        matrices = np.asarray(
            [
                self._B_matrix_from_coordinates(
                    elem,
                    float(xi),
                    float(eta),
                    x,
                    y,
                    natural_gradients=natural_gradient,
                )[0]
                for (xi, eta), natural_gradient in zip(points, gradients)
            ],
            dtype=float,
        )
        strains = np.matmul(matrices, element_u)
        values = np.matmul(strains, D.T)
        return points, values

    @staticmethod
    def extrapolate_stress_to_nodes(
        integration_point_values: np.ndarray,
        gauss_order: int = 2,
    ) -> np.ndarray:
        """Extrapolate Quad4 integration-point components to element nodes."""
        if gauss_order != 2:
            raise ValueError("gauss_order must be 2 for Quad4 extrapolation")
        values = np.asarray(integration_point_values, dtype=float)
        matrix = _quad4_extrapolation_matrix(gauss_order)
        if values.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Quad4 requires {matrix.shape[1]} integration-point rows, got {values.shape}"
            )
        return matrix @ values

    @staticmethod
    def interpolate_stress_to_centroid(
        integration_point_values: np.ndarray,
        gauss_order: int = 2,
    ) -> np.ndarray:
        """Interpolate Quad4 integration-point components to (0, 0)."""
        values = np.asarray(integration_point_values, dtype=float)
        matrix = _quad4_centroid_recovery_matrix(gauss_order)
        if values.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Quad4 requires {matrix.shape[1]} integration-point rows, got {values.shape}"
            )
        return (matrix @ values)[0]

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
        properties = PlaneProperties.from_mapping(
            elem.props,
            element_id=elem.id,
            element_type=getattr(elem, "type", None),
        )
        return (
            linear_elastic.plane_matrix(
                properties.E,
                properties.nu,
                properties.plane_type,
            ),
            properties.thickness,
        )

    def _plane_data(self, elem: Any):
        """Return plane type tag and Poisson ratio."""
        properties = PlaneProperties.from_mapping(
            elem.props,
            element_id=elem.id,
            element_type=getattr(elem, "type", None),
        )
        return properties.plane_type, properties.nu

    def _thickness(self, elem: Any) -> float:
        """Return plane element thickness."""
        return plane_thickness(
            elem.props,
            element_id=elem.id,
        )

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
        nodes = self._nodes(mesh, elem, node_lookup)
        x, y = self._coords(nodes)
        return [
            (*self._B_matrix_from_coordinates(elem, xi, eta, x, y), w)
            for xi, eta, w in quad4_gauss_points(gauss_order)
        ]

    def _stiffness_from_reference_coordinates(
        self,
        elem: Any,
        reference_coordinates: np.ndarray,
        properties: PlaneProperties,
        gauss_order: int,
    ) -> np.ndarray:
        """Integrate the small-strain tangent from already-owned geometry."""

        reference = _quad4_coordinates(
            reference_coordinates,
            "reference_coordinates",
        )
        if reference.shape != (4, 2):
            raise ValueError(
                "Quad4 reference_coordinates must have shape (4, 2); "
                f"got {reference.shape}"
            )
        x = reference[:, 0]
        y = reference[:, 1]
        D = linear_elastic.plane_matrix(
            properties.E,
            properties.nu,
            properties.plane_type,
        )
        Ke = np.zeros((8, 8), dtype=float)
        for xi, eta, weight in quad4_gauss_points(gauss_order):
            B, detJ = self._B_matrix_from_coordinates(
                elem,
                xi,
                eta,
                x,
                y,
            )
            Ke += (B.T @ D @ B) * (properties.thickness * detJ * weight)
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
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)

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
        *,
        natural_gradients: np.ndarray | None = None,
    ):
        """Return B and detJ using coordinates already gathered for an element."""

        dN = (
            quad4_shape_grad_xi_eta(xi, eta)
            if natural_gradients is None
            else natural_gradients
        )
        J = self._jacobian(x, y, dN)
        detJ = self._checked_det_jacobian(elem, J)

        inverse = np.asarray(
            ((J[1, 1], -J[0, 1]), (-J[1, 0], J[0, 0])),
            dtype=float,
        ) / detJ
        dN_xy = inverse @ dN
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
        """Return Quad4 shape functions from the canonical definition."""
        return quad4_shape_functions(xi, eta)

    def _jacobian(self, x: np.ndarray, y: np.ndarray, dN: np.ndarray) -> np.ndarray:
        """Return 2D isoparametric Jacobian."""
        return np.array(
            [[np.dot(dN[0], x), np.dot(dN[0], y)],
             [np.dot(dN[1], x), np.dot(dN[1], y)]],
            dtype=float,
        )

    def _checked_det_jacobian(self, elem: Any, J: np.ndarray) -> float:
        """Return detJ and reject singular or inverted elements."""
        detJ = float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
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


__all__ = [
    "Quad4Definition",
    "Quad4LinearOperator",
    "quad4_gauss_points",
    "quad4_shape_functions",
    "quad4_shape_grad_xi_eta",
]
