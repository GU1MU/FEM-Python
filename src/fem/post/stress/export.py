from __future__ import annotations

from typing import Any, Sequence

from . import beam as beam_export, dispatch, element as element_export, nodal as nodal_export


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
    threshold: float = 75.0,
) -> None:
    """Export nodal stresses to CSV. Element type is inferred when possible."""
    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    if "beam2" in type_keys:
        raise ValueError(
            "Beam2 nodal stress export requires ModelResult load context; "
            "use nodal_from_result(result, path)"
        )
    if len(type_keys) == 1:
        nodal_export.by_type(type_keys[0], mesh, U, path, gauss_order, threshold)
        return
    nodal_export.mixed(type_keys, mesh, U, path, gauss_order, threshold)


def nodal_from_result(
    result: Any,
    path: str,
    element_type: str | None = None,
    gauss_order: int | None = None,
    threshold: float = 75.0,
) -> None:
    """Export nodal stresses with access to analysis-step load context."""
    mesh = result.model.mesh
    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    if type_keys == ("beam2",):
        beam_export.export_nodal(result, path)
        return
    nodal(
        mesh,
        result.U,
        path,
        element_type=element_type,
        gauss_order=gauss_order,
        threshold=threshold,
    )
