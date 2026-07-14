from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ElementKernel(Protocol):
    """Element type adapter for stiffness, loads, and stress."""
    canonical_type: str
    aliases: tuple[str, ...]

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return element stiffness matrix."""
        ...

    def stress_at(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        *coords: float,
        node_lookup: dict[int, Any] | None = None,
    ):
        """Return stress at one natural coordinate point when supported."""
        ...

    def nodal_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
        *args,
    ) -> np.ndarray:
        """Return element-nodal stresses when supported."""
        ...

    def element_stress(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
    ):
        """Return element stress data when supported."""
        ...


def build_node_lookup(mesh: Any) -> dict[int, Any]:
    """Return node lookup keyed by node id."""
    return {node.id: node for node in mesh.nodes}


def lagrange_weights_1d(points, x):
    """Return Lagrange weights at x."""
    weights = []
    for i, xi in enumerate(points):
        w = 1.0
        for j, xj in enumerate(points):
            if i != j:
                w *= (x - xj) / (xi - xj)
        weights.append(w)
    return weights


def extrapolate_tensor_product(gp_vals, xi_pts, eta_pts, node_coords):
    """Extrapolate Gauss-point values to nodes using tensor Lagrange."""
    n_eta = len(eta_pts)
    node_vals = []
    for xi_n, eta_n in node_coords:
        wx = lagrange_weights_1d(xi_pts, xi_n)
        wy = lagrange_weights_1d(eta_pts, eta_n)
        val = np.zeros(gp_vals.shape[1], dtype=float)
        for i in range(len(xi_pts)):
            for j in range(len(eta_pts)):
                idx = i * n_eta + j
                val += gp_vals[idx] * (wx[i] * wy[j])
        node_vals.append(val)
    return np.array(node_vals)
