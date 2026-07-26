"""把应力位置语义转换为可直接绘制的 VTK 拓扑。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import numpy as np

from fem.post.averaging import NodalAveragingPolicy, resolve_nodal_stress
from fem.post.stress import field

from .model_adapter import ModelGeometry
from .result_adapter import ResultData, StressSample


@dataclass(frozen=True, slots=True)
class StressRenderGeometry:
    """应力专用拓扑及其真实有限元编号映射。"""

    points: np.ndarray
    cells: tuple[tuple[int, ...], ...]
    cell_array: np.ndarray
    cell_types: np.ndarray
    values: np.ndarray
    point_index_to_node_id: dict[int, int]
    point_index_to_element_id: dict[int, int | None]
    cell_index_to_element_id: dict[int, int]


def build_stress_render_geometry(
    geometry: ModelGeometry,
    data: ResultData,
    field_key: str,
    threshold: float = 75.0,
) -> StressRenderGeometry:
    """创建节点平均或单元节点不平均的应力绘图拓扑。"""

    prefix, component = field_key.split(":", 1)
    if prefix not in {"N", "NODAL", "EN"}:
        raise ValueError(f"不支持的应力绘图位置：{prefix}")
    averaging_policy = NodalAveragingPolicy(threshold)

    samples = _sample_lookup(data, component)
    nodal_mode = prefix in {"N", "NODAL"}
    if nodal_mode and not samples and data.nodal_stress:
        return _direct_nodal_geometry(geometry, data, component)
    decisions = (
        _average_decisions(geometry, data, component, averaging_policy)
        if nodal_mode
        else {}
    )
    point_keys: dict[tuple[object, ...], int] = {}
    points: list[np.ndarray] = []
    values: list[float] = []
    point_nodes: dict[int, int] = {}
    point_elements: dict[int, int | None] = {}
    cells: list[tuple[int, ...]] = []

    for cell_index, cell in enumerate(geometry.cells):
        element_id = geometry.cell_index_to_element_id[cell_index]
        local_cell: list[int] = []
        for source_point in cell:
            node_id = geometry.point_index_to_node_id[source_point]
            sample = samples[(element_id, node_id)]
            if not nodal_mode or not decisions[(node_id, sample.region_key)][0]:
                point_key = ("raw", element_id, node_id)
                scalar = sample.values[component]
                source_element: int | None = element_id
            else:
                point_key = ("average", node_id, sample.region_key)
                scalar = decisions[(node_id, sample.region_key)][1]
                source_element = None
            point_index = point_keys.get(point_key)
            if point_index is None:
                point_index = len(points)
                point_keys[point_key] = point_index
                points.append(geometry.points[source_point])
                values.append(float(scalar))
                point_nodes[point_index] = node_id
                point_elements[point_index] = source_element
            local_cell.append(point_index)
        cells.append(tuple(local_cell))

    flat = np.fromiter(
        chain.from_iterable((len(cell), *cell) for cell in cells),
        dtype=np.int64,
        count=sum(len(cell) + 1 for cell in cells),
    )
    return StressRenderGeometry(
        points=np.asarray(points, dtype=float).reshape((-1, 3)),
        cells=tuple(cells),
        cell_array=flat,
        cell_types=np.asarray(geometry.cell_types, dtype=np.uint8),
        values=np.asarray(values, dtype=float),
        point_index_to_node_id=point_nodes,
        point_index_to_element_id=point_elements,
        cell_index_to_element_id=dict(geometry.cell_index_to_element_id),
    )


def _direct_nodal_geometry(
    geometry: ModelGeometry,
    data: ResultData,
    component: str,
) -> StressRenderGeometry:
    """Use an already recovered nodal field without element-local samples."""
    values = np.asarray([
        data.nodal_stress[
            geometry.point_index_to_node_id[point_index]
        ][component]
        for point_index in range(len(geometry.points))
    ], dtype=float)
    return StressRenderGeometry(
        points=geometry.points.copy(),
        cells=tuple(geometry.cells),
        cell_array=geometry.cell_array.copy(),
        cell_types=geometry.cell_types.copy(),
        values=values,
        point_index_to_node_id=dict(geometry.point_index_to_node_id),
        point_index_to_element_id={
            point_index: None for point_index in range(len(geometry.points))
        },
        cell_index_to_element_id=dict(geometry.cell_index_to_element_id),
    )


def _sample_lookup(data: ResultData, component: str) -> dict[tuple[int, int], StressSample]:
    lookup: dict[tuple[int, int], StressSample] = {}
    for node_id, rows in data.nodal_stress_samples.items():
        for sample in rows:
            if component in sample.values:
                lookup[(sample.element_id, node_id)] = sample
    return lookup


def _average_decisions(
    geometry: ModelGeometry,
    data: ResultData,
    component: str,
    policy: NodalAveragingPolicy,
) -> dict[tuple[int, object], tuple[bool, float]]:
    """Choose canonical tensor-averaged N values or unaveraged EN values."""
    if field.StressPosition.ELEMENT_NODAL in data.stress_fields:
        return _canonical_average_decisions(
            geometry,
            data,
            component,
            policy,
        )
    return _legacy_average_decisions(
        data,
        component,
        policy.threshold_percent,
    )


def _canonical_average_decisions(
    geometry: ModelGeometry,
    data: ResultData,
    component: str,
    policy: NodalAveragingPolicy,
) -> dict[tuple[int, object], tuple[bool, float]]:
    """Map the post-processing resolver's tensor-level decisions for drawing."""

    element_nodal = data.stress_fields[field.StressPosition.ELEMENT_NODAL]
    node_ids = tuple(
        geometry.point_index_to_node_id[index]
        for index in range(len(geometry.points))
    )
    element_ids = tuple(
        geometry.cell_index_to_element_id[index]
        for index in range(len(geometry.cells))
    )
    resolved = resolve_nodal_stress(
        element_nodal,
        policy,
        node_ids=node_ids,
        element_ids=element_ids,
    )
    decisions: dict[tuple[int, object], tuple[bool, float]] = {}
    for record in resolved.records:
        if record.node_id is None or record.region_key is None:
            raise ValueError("canonical nodal stress rows require node and region ids")
        key = (record.node_id, record.region_key)
        if record.averaged:
            values = record.values(resolved.component_names)
            decisions[key] = (True, float(values[component]))
        else:
            decisions.setdefault(key, (False, float("nan")))
    return decisions


def _legacy_average_decisions(
    data: ResultData,
    component: str,
    threshold: float,
) -> dict[tuple[int, object], tuple[bool, float]]:
    """Compatibility path for synthetic ResultData without canonical N records."""

    region_values: dict[object, list[float]] = {}
    node_regions: dict[tuple[int, object], list[StressSample]] = {}
    for node_id, rows in data.nodal_stress_samples.items():
        for sample in rows:
            if component not in sample.values:
                continue
            region_values.setdefault(sample.region_key, []).append(sample.values[component])
            node_regions.setdefault((node_id, sample.region_key), []).append(sample)

    region_ranges = {
        key: float(np.ptp(np.asarray(values, dtype=float)))
        for key, values in region_values.items()
    }
    decisions: dict[tuple[int, object], tuple[bool, float]] = {}
    for key, rows in node_regions.items():
        scalars = np.asarray([row.values[component] for row in rows], dtype=float)
        weights = np.asarray([row.weight for row in rows], dtype=float)
        region_range = region_ranges[key[1]]
        variation = float(np.ptp(scalars))
        tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(scalars)))) * 32.0
        relative = 0.0 if region_range <= tolerance else 100.0 * variation / region_range
        averaged = len(rows) == 1 or (threshold > 0.0 and relative <= threshold)
        decisions[key] = (averaged, float(np.average(scalars, weights=weights)))
    return decisions
