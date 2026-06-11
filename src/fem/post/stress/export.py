from __future__ import annotations

from typing import Any, Sequence

from . import dispatch, element as element_export, nodal as nodal_export


def element(
    mesh: Any,
    U: Sequence[float],
    path: str,
    element_type: str | None = None,
    gauss_order: int | None = None,
) -> None:
    """Export element stresses to CSV. Element type is inferred when possible."""
    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    if len(type_keys) == 1:
        element_export.by_type(type_keys[0], mesh, U, path, gauss_order)
        return
    element_export.mixed(type_keys, mesh, U, path, gauss_order)


def nodal(
    mesh: Any,
    U: Sequence[float],
    path: str,
    element_type: str | None = None,
    gauss_order: int | None = None,
    model: Any | None = None,
    averaging_policy: Any | None = None,
) -> None:
    """Export nodal stresses to CSV. Element type is inferred when possible."""
    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    if len(type_keys) == 1:
        nodal_export.by_type(
            type_keys[0],
            mesh,
            U,
            path,
            gauss_order,
            model,
            averaging_policy,
        )
        return
    nodal_export.mixed(type_keys, mesh, U, path, gauss_order, model, averaging_policy)
