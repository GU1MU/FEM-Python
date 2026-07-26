from __future__ import annotations

import csv as csv_module
from typing import Any, Sequence
import warnings

from .._paths import prepare_output_path
from . import (
    beam as beam_export,
    dispatch,
    element as element_export,
    nodal as nodal_export,
)
from .field import StressPosition, collect_stress


CSV_HEADER = [
    "position",
    "elem_id",
    "integration_point",
    "node_id",
    "local_node",
    "averaging_region",
    "x",
    "y",
    "z",
    "xi",
    "eta",
    "zeta",
    "S11",
    "S22",
    "S33",
    "S12",
    "S13",
    "S23",
    "Mises",
    "MaxPrincipal",
    "MidPrincipal",
    "MinPrincipal",
]


def csv(
    mesh: Any,
    U: Sequence[float],
    path: str,
    position: StressPosition | str = StressPosition.INTEGRATION_POINT,
    element_type: str | None = None,
    gauss_order: int | None = None,
) -> None:
    """Export a canonical plane or solid stress field to CSV."""
    stress_field = collect_stress(
        mesh,
        U,
        position=position,
        element_type=element_type,
        gauss_order=gauss_order,
    )
    output_path = prepare_output_path(path)
    region_ids: dict[object, int] = {}
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv_module.writer(stream)
        writer.writerow(CSV_HEADER)
        for record in stress_field.records:
            if record.region_key is None:
                region_id: int | str = ""
            else:
                region_id = region_ids.setdefault(
                    record.region_key, len(region_ids) + 1
                )
            coordinates = [*record.coordinates, "", ""]
            natural = [*(record.natural_coordinates or ()), "", "", ""]
            values = record.values(stress_field.component_names)
            writer.writerow([
                stress_field.position.value,
                _optional(record.elem_id),
                _optional(record.integration_point),
                _optional(record.node_id),
                _optional(record.local_node),
                region_id,
                coordinates[0],
                coordinates[1],
                coordinates[2],
                natural[0],
                natural[1],
                natural[2],
                values.get("S11", ""),
                values.get("S22", ""),
                values.get("S33", ""),
                values.get("S12", ""),
                values.get("S13", ""),
                values.get("S23", ""),
                values["Mises"],
                values["MaxPrincipal"],
                values["MidPrincipal"],
                values["MinPrincipal"],
            ])


def _optional(value: Any) -> Any:
    return "" if value is None else value


def element(
    mesh: Any,
    U: Sequence[float],
    path: str,
    element_type: str | None = None,
    gauss_order: int | None = None,
) -> None:
    """Compatibility export for legacy element or element-nodal stress output."""
    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    warnings.warn(
        "stress.export.element() is deprecated because its legacy position depends "
        "on element family; use stress.export.csv(..., position=...)",
        DeprecationWarning,
        stacklevel=2,
    )
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
    """Compatibility export for nodal-averaged continuum stress output."""
    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    warnings.warn(
        "stress.export.nodal() is deprecated; use stress.export.csv("
        "... position='nodal')",
        DeprecationWarning,
        stacklevel=2,
    )
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
