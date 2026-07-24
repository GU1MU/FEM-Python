from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ._common import add_kernel_load, require_element, validate_vector
from .condition import ElementGravityLoad, ElementLoad


def _validate_finite_vector(
    vector: tuple[float, ...],
    expected_size: int,
    name: str,
) -> None:
    """Validate the size and finiteness of a body-load vector."""
    validate_vector(vector, expected_size, name)
    try:
        values = np.asarray(vector, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} vector components must be finite numbers") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} vector components must be finite numbers")


def add_forces(
    mesh: Any,
    loads: list[ElementLoad],
    F: np.ndarray,
    elem_lookup: Dict[int, Any],
    node_lookup: Dict[int, Any],
    spatial_dim: int,
) -> None:
    """Assemble constant element body forces."""
    for load in loads:
        elem = require_element(elem_lookup, load.elem_id)
        _validate_finite_vector(load.vector, spatial_dim, "body force")
        add_kernel_load(mesh, elem, node_lookup, F, "body_force", load.vector)


def add_gravity(
    mesh: Any,
    gravity: tuple[float, ...] | None,
    F: np.ndarray,
    node_lookup: Dict[int, Any],
    spatial_dim: int,
) -> None:
    """Assemble gravity as density-scaled body force."""
    if gravity is None:
        return

    _validate_finite_vector(gravity, spatial_dim, "gravity")
    for elem in mesh.elements:
        rho_value = _element_density(elem, required=False)
        if rho_value is not None:
            vector = tuple(rho_value * value for value in gravity)
            add_kernel_load(mesh, elem, node_lookup, F, "body_force", vector)


def add_element_gravities(
    mesh: Any,
    loads: list[ElementGravityLoad],
    F: np.ndarray,
    elem_lookup: Dict[int, Any],
    node_lookup: Dict[int, Any],
    spatial_dim: int,
) -> None:
    """Assemble density-scaled gravity resolved to individual elements."""
    for load in loads:
        elem = require_element(elem_lookup, load.elem_id)
        _validate_finite_vector(load.acceleration, spatial_dim, "gravity")
        rho_value = _element_density(elem, required=True)
        vector = tuple(rho_value * value for value in load.acceleration)
        add_kernel_load(mesh, elem, node_lookup, F, "body_force", vector)


def _element_density(elem: Any, *, required: bool) -> float | None:
    """Return a validated element density for gravity assembly."""
    rho = elem.props.get("rho")
    if rho is None:
        if required:
            raise ValueError(
                f"Element {elem.id} rho is required for targeted gravity"
            )
        return None
    try:
        rho_value = float(rho)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Element {elem.id} rho must be a finite non-negative number, "
            f"got {rho!r}"
        ) from exc
    if not np.isfinite(rho_value) or rho_value < 0.0:
        raise ValueError(
            f"Element {elem.id} rho must be finite and >= 0, got {rho!r}"
        )
    return rho_value
