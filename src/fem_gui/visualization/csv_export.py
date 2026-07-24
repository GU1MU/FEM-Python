"""Export the currently selected GUI result field as a flat CSV table."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from fem.post.stress.field import StressPosition

from .result_adapter import ResultData


_HEADER = (
    "field",
    "position",
    "association",
    "node_id",
    "elem_id",
    "integration_point",
    "local_node",
    "x",
    "y",
    "z",
    "value",
)

_POSITIONS = {
    "IP": StressPosition.INTEGRATION_POINT,
    "CENTROID": StressPosition.CENTROID,
    "EN": StressPosition.ELEMENT_NODAL,
    "NODAL": StressPosition.NODAL,
}


def export_field_csv(data: ResultData, field_key: str, path: str | Path) -> Path:
    """Write one ready GUI scalar field with stable FEM identifiers."""
    key = str(field_key)
    scalar = data.fields.get(key)
    if scalar is None:
        raise KeyError(f"结果字段不存在：{key}")
    if not scalar.ready:
        raise ValueError(f"结果字段尚未恢复：{key}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(_HEADER)
        writer.writerows(_rows(data, key))
    return target


def _rows(data: ResultData, field_key: str) -> Iterable[tuple[object, ...]]:
    scalar = data.fields[field_key]
    prefix, separator, component = field_key.partition(":")
    position = _POSITIONS.get(prefix) if separator else None
    stress_field = data.stress_fields.get(position) if position is not None else None
    if stress_field is not None:
        for record in stress_field.records:
            values = record.values(stress_field.component_names)
            if component not in values:
                continue
            yield _row(
                component,
                position.value,
                scalar.association,
                record.node_id,
                record.elem_id,
                record.integration_point,
                record.local_node,
                record.coordinates,
                values[component],
            )
        return

    geometry = data._source_geometry
    if geometry is None:
        raise ValueError("结果缺少网格几何，无法导出 CSV")
    output_field = component if separator else field_key
    output_position = position.value if position is not None else "nodal"
    if scalar.association == "point":
        for point_index, value in enumerate(np.asarray(scalar.values, dtype=float)):
            yield _row(
                output_field,
                output_position,
                "point",
                geometry.point_index_to_node_id[point_index],
                None,
                None,
                None,
                geometry.points[point_index],
                value,
            )
        return
    if scalar.association == "cell":
        for cell_index, value in enumerate(np.asarray(scalar.values, dtype=float)):
            point_ids = geometry.cells[cell_index]
            coordinates = np.mean(geometry.points[list(point_ids)], axis=0)
            yield _row(
                output_field,
                output_position,
                "cell",
                None,
                geometry.cell_index_to_element_id[cell_index],
                None,
                None,
                coordinates,
                value,
            )
        return
    raise ValueError(f"CSV 导出不支持结果关联类型：{scalar.association}")


def _row(
    field_name: str,
    position: str,
    association: str,
    node_id: int | None,
    elem_id: int | None,
    integration_point: int | None,
    local_node: int | None,
    coordinates: Iterable[float],
    value: float,
) -> tuple[object, ...]:
    xyz = tuple(float(item) for item in coordinates)
    xyz = (*xyz, 0.0, 0.0, 0.0)[:3]
    return (
        field_name,
        position,
        association,
        _optional(node_id),
        _optional(elem_id),
        _optional(integration_point),
        _optional(local_node),
        *(_number(item) for item in xyz),
        _number(value),
    )


def _optional(value: int | None) -> object:
    return "" if value is None else int(value)


def _number(value: float) -> object:
    number = float(value)
    return "" if not math.isfinite(number) else format(number, ".17g")
