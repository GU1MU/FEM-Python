"""Canonical continuum nodal averaging without application or GUI dependencies."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Real
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .fields import ResultRegionKey, result_region_sort_key

if TYPE_CHECKING:
    from .stress.field import StressRecord


@dataclass(frozen=True, slots=True)
class NodalAveragingPolicy:
    """Threshold policy for resolving element-nodal tensor contributions."""

    threshold_percent: float = 75.0
    preserve_region_boundaries: bool = True

    def __post_init__(self) -> None:
        threshold = self.threshold_percent
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise TypeError("threshold_percent must be a real number")
        threshold_value = float(threshold)
        if not math.isfinite(threshold_value):
            raise ValueError("threshold_percent must be finite")
        if not 0.0 <= threshold_value <= 100.0:
            raise ValueError("threshold_percent must be from 0.0 through 100.0")
        if type(self.preserve_region_boundaries) is not bool:
            raise TypeError("preserve_region_boundaries must be bool")
        if not self.preserve_region_boundaries:
            raise ValueError(
                "preserve_region_boundaries must be True for continuum stress"
            )
        object.__setattr__(self, "threshold_percent", threshold_value)


@dataclass(frozen=True, slots=True)
class ResolvedStressField:
    """Canonical resolved nodal tensor rows."""

    component_names: tuple[str, ...]
    records: tuple[StressRecord, ...]


def resolve_nodal_stress(
    element_nodal_field: object,
    policy: NodalAveragingPolicy | None = None,
    *,
    node_ids: Sequence[int],
    element_ids: Sequence[int],
) -> ResolvedStressField:
    """Resolve complete element-nodal tensors per node and result region.

    The supplied node and element sequences are the authoritative mesh order.
    Nodes without an element-nodal contribution intentionally produce no row.
    """

    from .stress.field import (
        CANONICAL_PLANE_COMPONENT_NAMES,
        CANONICAL_SOLID_COMPONENT_NAMES,
        StressField,
        StressPosition,
        StressRecord,
    )
    from .stress.invariants import derive_stress_invariants

    if type(element_nodal_field) is not StressField:
        raise TypeError("element_nodal_field must be StressField")
    if element_nodal_field.position is not StressPosition.ELEMENT_NODAL:
        raise ValueError(
            "element_nodal_field must have element_nodal position"
        )
    component_names = tuple(element_nodal_field.component_names)
    if component_names not in {
        CANONICAL_PLANE_COMPONENT_NAMES,
        CANONICAL_SOLID_COMPONENT_NAMES,
    }:
        raise ValueError(
            "element_nodal_field must use complete canonical stress components"
        )
    if policy is None:
        policy = NodalAveragingPolicy()
    if type(policy) is not NodalAveragingPolicy:
        raise TypeError("policy must be NodalAveragingPolicy")

    node_order = _ordered_id_map(node_ids, label="node_ids")
    element_order = _ordered_id_map(element_ids, label="element_ids")
    grouped: dict[tuple[int, ResultRegionKey], list[StressRecord]] = {}
    values_by_region: dict[ResultRegionKey, list[tuple[float, ...]]] = {}
    source_locations: set[tuple[int, int, int]] = set()

    for record in element_nodal_field.records:
        if type(record) is not StressRecord:
            raise TypeError("element_nodal_field records must be StressRecord")
        if record.position is not StressPosition.ELEMENT_NODAL:
            raise ValueError("all input records must have element_nodal position")
        if (
            record.node_id is None
            or record.elem_id is None
            or record.local_node is None
        ):
            raise ValueError(
                "element-nodal records require node, element, and local-node ids"
            )
        if type(record.region_key) is not ResultRegionKey:
            raise TypeError(
                "element-nodal records require exact ResultRegionKey identity"
            )
        if record.node_id not in node_order:
            raise ValueError(
                f"element-nodal node {record.node_id} is absent from node_ids"
            )
        if record.elem_id not in element_order:
            raise ValueError(
                f"element-nodal element {record.elem_id} is absent from element_ids"
            )
        if len(record.components) != len(component_names):
            raise ValueError(
                "element-nodal record component count does not match the field"
            )
        values = np.asarray(record.components, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("element-nodal stress components must be finite")
        _positive_finite_weight(record.weight)
        source_location = (
            int(record.elem_id),
            int(record.local_node),
            int(record.node_id),
        )
        if source_location in source_locations:
            raise ValueError(
                "element-nodal source locations must be unique"
            )
        source_locations.add(source_location)
        grouped.setdefault(
            (int(record.node_id), record.region_key),
            [],
        ).append(record)
        values_by_region.setdefault(record.region_key, []).append(
            tuple(float(value) for value in values)
        )

    region_ranges: dict[ResultRegionKey, np.ndarray] = {}
    region_tolerances: dict[ResultRegionKey, np.ndarray] = {}
    for region_key, values in values_by_region.items():
        array = np.asarray(values, dtype=float)
        region_ranges[region_key] = np.ptp(array, axis=0)
        region_tolerances[region_key] = (
            np.finfo(float).eps
            * np.maximum(1.0, np.max(np.abs(array), axis=0))
            * 32.0
        )

    records: list[StressRecord] = []
    grouped_by_node: dict[
        int,
        list[tuple[ResultRegionKey, list[StressRecord]]],
    ] = {}
    for (node_id, region_key), contributions in grouped.items():
        grouped_by_node.setdefault(node_id, []).append(
            (region_key, contributions)
        )

    for node_id in node_order:
        node_regions = sorted(
            grouped_by_node.get(node_id, ()),
            key=lambda item: result_region_sort_key(item[0]),
        )
        for region_key, contributions in node_regions:
            ordered = sorted(
                contributions,
                key=lambda record: (
                    element_order[int(record.elem_id)],
                    int(record.local_node),
                ),
            )
            if len(ordered) == 1 or policy.threshold_percent == 0.0:
                records.extend(
                    replace(
                        record,
                        position=StressPosition.NODAL,
                        averaged=False,
                    )
                    for record in ordered
                )
                continue

            node_values = np.asarray(
                [record.components for record in ordered],
                dtype=float,
            )
            node_ranges = np.ptp(node_values, axis=0)
            ranges = region_ranges[region_key]
            tolerances = region_tolerances[region_key]
            relative_variation = np.zeros_like(node_ranges)
            outside_tolerance = ranges > tolerances
            relative_variation[outside_tolerance] = (
                100.0
                * node_ranges[outside_tolerance]
                / ranges[outside_tolerance]
            )
            if not np.all(
                relative_variation <= policy.threshold_percent
            ):
                records.extend(
                    replace(
                        record,
                        position=StressPosition.NODAL,
                        averaged=False,
                    )
                    for record in ordered
                )
                continue

            weights = np.asarray(
                [_positive_finite_weight(record.weight) for record in ordered],
                dtype=float,
            )
            weight_sum = float(np.sum(weights))
            if not math.isfinite(weight_sum) or weight_sum <= 0.0:
                raise ValueError("element-nodal stress weights have invalid sum")
            averaged_components = np.average(
                node_values,
                axis=0,
                weights=weights,
            )
            if not np.isfinite(averaged_components).all():
                raise ValueError("averaged stress components must be finite")
            first = ordered[0]
            component_tuple = tuple(
                float(value) for value in averaged_components
            )
            records.append(
                StressRecord(
                    position=StressPosition.NODAL,
                    coordinates=first.coordinates,
                    components=component_tuple,
                    invariants=derive_stress_invariants(
                        component_tuple,
                        component_names,
                    ),
                    node_id=node_id,
                    region_key=region_key,
                    weight=weight_sum,
                    displacement=first.displacement,
                    averaged=True,
                )
            )

    return ResolvedStressField(component_names, tuple(records))


def _ordered_id_map(
    values: Sequence[int],
    *,
    label: str,
) -> dict[int, int]:
    result: dict[int, int] = {}
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must contain integer ids")
        if value in result:
            raise ValueError(f"{label} must not contain duplicate ids")
        result[value] = index
    return result


def _positive_finite_weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("StressRecord.weight must be a real number")
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("StressRecord.weight must be positive and finite")
    return weight


__all__ = [
    "NodalAveragingPolicy",
    "ResolvedStressField",
    "resolve_nodal_stress",
]
