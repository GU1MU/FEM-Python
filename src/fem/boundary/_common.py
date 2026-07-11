from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..elements import get_element_kernel


def require_element(elem_lookup: Dict[int, Any], elem_id: int) -> Any:
    """Return element by id."""
    elem = elem_lookup.get(elem_id)
    if elem is None:
        raise KeyError(f"Element {elem_id} not found in mesh")
    return elem


def spatial_dim(mesh: Any) -> int:
    """Return mesh coordinate dimension from node fields."""
    if not mesh.nodes:
        raise ValueError("mesh must contain at least one node")
    return 3 if hasattr(mesh.nodes[0], "z") else 2


def validate_vector(vector: tuple[float, ...], expected_size: int, name: str) -> None:
    """Validate a load vector size and numeric values."""
    try:
        values = np.asarray(vector, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} vector components must be finite numbers") from exc
    if values.shape != (expected_size,):
        raise ValueError(
            f"{name} vector must have {expected_size} components, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} vector components must be finite numbers")


def add_kernel_load(
    mesh: Any,
    elem: Any,
    node_lookup: Dict[int, Any],
    F: np.ndarray,
    method_name: str,
    vector: tuple[float, ...],
    local_index: int | None = None,
) -> None:
    """Assemble one element load through an element kernel method."""
    kernel = get_element_kernel(elem.type)
    method = getattr(kernel, method_name, None)
    if method is None:
        raise NotImplementedError(
            f"Unsupported element type for {method_name} assembly: {elem.type}"
        )

    if local_index is None:
        fe = method(mesh, elem, vector, node_lookup)
    else:
        fe = method(mesh, elem, local_index, vector, node_lookup)
    dofs = list(mesh.element_dofs(elem))
    fe = np.asarray(fe, dtype=float)
    if fe.shape != (len(dofs),):
        raise ValueError(
            f"Element {elem.id} {method_name} vector shape {fe.shape} "
            f"does not match {len(dofs)} element DOFs"
        )
    if not np.all(np.isfinite(fe)):
        raise ValueError(
            f"Element {elem.id} {method_name} vector contains non-finite values"
        )
    F[dofs] += fe
