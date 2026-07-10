from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from . import dispatch
from ._common import element_volume, nodal_stress, node_lookup, validated_u


PLANE_COMPONENT_NAMES = ("sig_x", "sig_y", "tau_xy")
SOLID_COMPONENT_NAMES = (
    "sig_x",
    "sig_y",
    "sig_z",
    "tau_xy",
    "tau_yz",
    "tau_zx",
)
MATERIAL_SIGNATURE_KEY = "_stress_material_signature"
SECTION_SIGNATURE_KEY = "_stress_section_signature"


@dataclass(frozen=True)
class StressRegionKey:
    """Material and section signatures that form a hard averaging boundary."""

    material_signature: Hashable
    section_signature: Hashable


@dataclass(frozen=True)
class ElementNodalStressContribution:
    """One element-local stress tensor recovered at a mesh node."""

    node_id: int
    elem_id: int
    local_node: int
    components: tuple[float, ...]
    weight: float
    region_key: StressRegionKey
    plane_type: str | None = None
    poisson_ratio: float | None = None


@dataclass(frozen=True)
class NodalStressField:
    """Raw element-nodal stress contributions grouped in mesh-node order."""

    component_names: tuple[str, ...]
    contributions_by_node: Mapping[int, tuple[ElementNodalStressContribution, ...]]
    node_ids: tuple[int, ...]


@dataclass(frozen=True)
class ResolvedNodalStressRow:
    """One averaged or element-local nodal stress value for export."""

    node_id: int
    components: tuple[float, ...]
    elem_id: int | None
    local_node: int | None
    averaged: bool
    plane_type: str | None = None
    poisson_ratio: float | None = None


@dataclass(frozen=True)
class ResolvedNodalStressField:
    """Resolved nodal stress rows in deterministic mesh and element order."""

    component_names: tuple[str, ...]
    rows: tuple[ResolvedNodalStressRow, ...]


def resolve(
    field: NodalStressField,
    threshold: float = 75.0,
) -> ResolvedNodalStressField:
    """Resolve raw contributions using component-wise relative variation."""

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 100.0
    ):
        raise ValueError("threshold must be a finite number from 0.0 through 100.0")
    threshold_value = float(threshold)
    region_ranges = _component_ranges_by_region(field)
    rows: list[ResolvedNodalStressRow] = []

    for node_id in field.node_ids:
        contributions = tuple(field.contributions_by_node.get(node_id, ()))
        if not contributions:
            rows.append(
                ResolvedNodalStressRow(
                    node_id=node_id,
                    components=(0.0,) * len(field.component_names),
                    elem_id=None,
                    local_node=None,
                    averaged=False,
                )
            )
            continue
        if len(contributions) == 1:
            rows.append(_raw_row(contributions[0]))
            continue

        region_keys = {item.region_key for item in contributions}
        if len(region_keys) != 1 or threshold_value == 0.0:
            rows.extend(_raw_row(item) for item in contributions)
            continue

        region_key = contributions[0].region_key
        values = np.asarray([item.components for item in contributions], dtype=float)
        node_ranges = np.ptp(values, axis=0)
        region_ranges_for_key = region_ranges[region_key]
        relative_variation = np.zeros_like(node_ranges)
        nonzero = region_ranges_for_key != 0.0
        relative_variation[nonzero] = (
            100.0 * node_ranges[nonzero] / region_ranges_for_key[nonzero]
        )
        if np.all(relative_variation <= threshold_value):
            weights = np.asarray([item.weight for item in contributions], dtype=float)
            averaged = np.average(values, axis=0, weights=weights)
            first = contributions[0]
            rows.append(
                ResolvedNodalStressRow(
                    node_id=node_id,
                    components=tuple(float(value) for value in averaged),
                    elem_id=None,
                    local_node=None,
                    averaged=True,
                    plane_type=first.plane_type,
                    poisson_ratio=first.poisson_ratio,
                )
            )
        else:
            rows.extend(_raw_row(item) for item in contributions)

    return ResolvedNodalStressField(field.component_names, tuple(rows))


def collect(
    mesh: Any,
    U: Sequence[float],
    element_type: str | None = None,
    gauss_order: int | None = None,
) -> NodalStressField:
    """Collect every selected element-local nodal stress without averaging."""

    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    if not dispatch.nodal_stress_supported(type_keys):
        raise ValueError(f"Nodal stress collection is not available for {type_keys}")
    group = dispatch.stress_group_for_keys(type_keys)
    component_names = PLANE_COMPONENT_NAMES if group == "plane" else SOLID_COMPONENT_NAMES
    selected = set(type_keys)
    U_array = validated_u(mesh, U)
    lookup = node_lookup(mesh)
    contributions: dict[int, list[ElementNodalStressContribution]] = {
        int(node_id): [] for node_id in mesh.node_ids
    }

    for elem in mesh.elements:
        type_key = dispatch.type_key_from_name(elem.type)
        if type_key not in selected:
            continue
        order = gauss_order if gauss_order is not None else dispatch.default_gauss_order(type_key)
        recovered = nodal_stress(mesh, elem, U_array, lookup, order)
        plane_type = None
        poisson_ratio = None
        if group == "plane":
            recovered, plane_type, poisson_ratio = recovered
        node_values = np.asarray(recovered, dtype=float)
        if node_values.shape != (len(elem.node_ids), len(component_names)):
            raise ValueError(
                f"Element {elem.id} nodal stress shape {node_values.shape} does not match "
                f"({len(elem.node_ids)}, {len(component_names)})"
            )
        weight = (
            element_volume(mesh, elem, lookup)
            if type_key in {"tet4", "tet10"}
            else 1.0
        )
        region_key = _region_key(elem)
        for local_node, (node_id, values) in enumerate(
            zip(elem.node_ids, node_values), start=1
        ):
            contributions[int(node_id)].append(
                ElementNodalStressContribution(
                    node_id=int(node_id),
                    elem_id=int(elem.id),
                    local_node=local_node,
                    components=tuple(float(value) for value in values),
                    weight=float(weight),
                    region_key=region_key,
                    plane_type=str(plane_type) if plane_type is not None else None,
                    poisson_ratio=(
                        float(poisson_ratio) if poisson_ratio is not None else None
                    ),
                )
            )

    return NodalStressField(
        component_names=component_names,
        contributions_by_node={
            node_id: tuple(values) for node_id, values in contributions.items()
        },
        node_ids=tuple(int(node_id) for node_id in mesh.node_ids),
    )


def _component_ranges_by_region(field: NodalStressField) -> dict[StressRegionKey, np.ndarray]:
    values_by_region: dict[StressRegionKey, list[tuple[float, ...]]] = {}
    for contributions in field.contributions_by_node.values():
        for item in contributions:
            values_by_region.setdefault(item.region_key, []).append(item.components)
    return {
        key: np.ptp(np.asarray(values, dtype=float), axis=0)
        for key, values in values_by_region.items()
    }


def _raw_row(item: ElementNodalStressContribution) -> ResolvedNodalStressRow:
    return ResolvedNodalStressRow(
        node_id=item.node_id,
        components=item.components,
        elem_id=item.elem_id,
        local_node=item.local_node,
        averaged=False,
        plane_type=item.plane_type,
        poisson_ratio=item.poisson_ratio,
    )


def _region_key(elem: Any) -> StressRegionKey:
    props = dict(getattr(elem, "props", {}))
    material_signature = props.get(MATERIAL_SIGNATURE_KEY)
    if material_signature is None:
        if "material" in props:
            material_signature = ("material", _freeze(props["material"]))
        elif "material_id" in props:
            material_signature = ("material_id", _freeze(props["material_id"]))
        else:
            material_signature = (
                "effective",
                tuple((name, _freeze(props[name])) for name in ("E", "nu", "rho") if name in props),
            )

    section_signature = props.get(SECTION_SIGNATURE_KEY)
    if section_signature is None:
        excluded = {
            MATERIAL_SIGNATURE_KEY,
            SECTION_SIGNATURE_KEY,
            "material",
            "material_id",
            "E",
            "nu",
            "rho",
            "section_type",
        }
        section_properties = {
            key: value for key, value in props.items() if key not in excluded
        }
        section_signature = (
            "section",
            _freeze(props.get("section_type")),
            _freeze(section_properties),
        )

    return StressRegionKey(_freeze(material_signature), _freeze(section_signature))


def _freeze(value: Any) -> Hashable:
    """Recursively convert signature data to a deterministic hashable value."""

    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    if isinstance(value, np.generic):
        return value.item()
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value
