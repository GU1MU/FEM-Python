"""正式求解结果到可视化场变量的适配。"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
from typing import Any

import numpy as np

from fem.elements import get_element_kernel
from fem.post.stress import beam, dispatch, field
from .model_adapter import ModelGeometry


@dataclass(frozen=True, slots=True)
class ScalarField:
    """一个节点或单元标量场。"""

    key: str
    label: str
    association: str
    values: np.ndarray
    ready: bool = True


@dataclass(frozen=True, slots=True)
class StressSample:
    """一个节点上的平均或单元侧应力值。"""

    element_id: int
    local_node: int
    region_key: object
    weight: float
    values: dict[str, float]


@dataclass(frozen=True, slots=True)
class ResultData:
    """位移、反力和当前内核可恢复的应力。"""

    displacement_vectors: np.ndarray
    reaction_vectors: np.ndarray
    fields: dict[str, ScalarField]
    nodal_values: dict[int, dict[str, float]]
    element_stress: dict[int, dict[str, float]]
    nodal_stress: dict[int, dict[str, float]]
    nodal_stress_samples: dict[int, tuple[StressSample, ...]]
    _source_result: Any | None = dataclass_field(
        default=None, repr=False, compare=False
    )
    _source_geometry: ModelGeometry | None = dataclass_field(
        default=None, repr=False, compare=False
    )
    _stress_cache: dict[str, Any] = dataclass_field(
        default_factory=dict, repr=False, compare=False
    )
    nodal_stress_by_region: dict[tuple[int, object], dict[str, float]] = (
        dataclass_field(default_factory=dict, repr=False, compare=False)
    )
    stress_fields: dict[field.StressPosition, field.StressField] = dataclass_field(
        default_factory=dict, repr=False, compare=False
    )
    stress_position_labels: dict[str, str] = dataclass_field(
        default_factory=dict, repr=False, compare=False
    )

    def field_ready(self, key: str) -> bool:
        """Return whether a catalog field already has numeric values."""
        scalar = self.fields.get(str(key))
        return scalar is not None and scalar.ready

    def available_stress_prefixes(self) -> tuple[str, ...]:
        """Return stress positions advertised by the current element family."""
        return tuple(
            prefix
            for prefix in ("IP", "CENTROID", "EN", "NODAL")
            if any(key.startswith(f"{prefix}:") for key in self.fields)
        )

    def stress_position_label(self, prefix: str) -> str:
        """Return the context-aware GUI label for one stress position."""
        return self.stress_position_labels.get(
            str(prefix),
            _DEFAULT_STRESS_POSITION_LABELS.get(str(prefix), str(prefix)),
        )


_DEFAULT_STRESS_POSITION_LABELS = {
    "IP": "积分点",
    "CENTROID": "单元质心",
    "EN": "单元节点（不平均）",
    "NODAL": "节点平均",
}


def field_family(field_key: str) -> str:
    """Return the shared GUI result family for a field key."""
    key = str(field_key)
    if ":" in key:
        return "S"
    if key in {"U", "U1", "U2", "U3"}:
        return "U"
    if key in {"R1", "R2", "R3"}:
        return "R"
    if key in {"RF", "RF1", "RF2", "RF3"}:
        return "RF"
    if key in {"RM1", "RM2", "RM3"}:
        return "RM"
    return "S"


def build_result_data(
    result: Any,
    geometry: ModelGeometry,
    *,
    include_stress: bool = True,
) -> ResultData:
    """从 ModelResult 读取数据，不复制求解逻辑。"""
    mesh = result.model.mesh
    stress_position_labels = dict(_DEFAULT_STRESS_POSITION_LABELS)
    try:
        type_keys = dispatch.resolve_type_keys(mesh, None)
        if type_keys == ("beam2",):
            stress_position_labels["NODAL"] = "节点包络"
    except ValueError:
        pass
    displacement = _nodal_vectors(mesh, result.U, geometry)
    reactions = _nodal_vectors(mesh, result.reactions, geometry)
    fields: dict[str, ScalarField] = {}
    labels = ("U1", "U2", "U3")
    translation_count = 3 if mesh.nodes and hasattr(mesh.nodes[0], "z") else 2
    for component in range(min(mesh.dofs_per_node, translation_count)):
        _add_field(fields, labels[component], labels[component], "point", displacement[:, component])
    _add_field(fields, "U", "总位移", "point", np.linalg.norm(displacement, axis=1))
    reaction_labels = ("RF1", "RF2", "RF3")
    for component in range(min(mesh.dofs_per_node, translation_count)):
        _add_field(
            fields,
            reaction_labels[component],
            reaction_labels[component],
            "point",
            reactions[:, component],
        )
    _add_field(fields, "RF", "总反力", "point", np.linalg.norm(reactions, axis=1))
    rotation_components = max(
        0,
        min(3, mesh.dofs_per_node - translation_count),
    )
    rotations = np.zeros((len(geometry.points), rotation_components), dtype=float)
    moments = np.zeros_like(rotations)
    rotation_labels = (
        (("R3", "RM3"),)
        if translation_count == 2 and rotation_components == 1
        else tuple(
            (f"R{component + 1}", f"RM{component + 1}")
            for component in range(rotation_components)
        )
    )
    for rotation_component, (rotation_key, moment_key) in enumerate(
        rotation_labels
    ):
        dof_component = translation_count + rotation_component
        rotations[:, rotation_component] = [
            result.U[
                mesh.global_dof(
                    geometry.point_index_to_node_id[index],
                    dof_component,
                )
            ]
            for index in range(len(geometry.points))
        ]
        moments[:, rotation_component] = [
            result.reactions[
                mesh.global_dof(
                    geometry.point_index_to_node_id[index],
                    dof_component,
                )
            ]
            for index in range(len(geometry.points))
        ]
        _add_field(
            fields,
            rotation_key,
            f"转角 {rotation_key}",
            "point",
            rotations[:, rotation_component],
        )
        _add_field(
            fields,
            moment_key,
            f"反力矩 {moment_key}",
            "point",
            moments[:, rotation_component],
        )
    nodal_values: dict[int, dict[str, float]] = {}
    for point_index, node_id in geometry.point_index_to_node_id.items():
        values = {
            "U1": float(displacement[point_index, 0]),
            "U2": float(displacement[point_index, 1]),
            "U": float(np.linalg.norm(displacement[point_index])),
            "RF1": float(reactions[point_index, 0]),
            "RF2": float(reactions[point_index, 1]),
            "RF": float(np.linalg.norm(reactions[point_index])),
        }
        if translation_count == 3:
            values["U3"] = float(displacement[point_index, 2])
            values["RF3"] = float(reactions[point_index, 2])
        for rotation_component, (rotation_key, moment_key) in enumerate(
            rotation_labels
        ):
            values[rotation_key] = float(
                rotations[point_index, rotation_component]
            )
            values[moment_key] = float(
                moments[point_index, rotation_component]
            )
        nodal_values[node_id] = values
    stress_fields: dict[field.StressPosition, field.StressField] = {}
    nodal_stress_by_region: dict[tuple[int, object], dict[str, float]] = {}
    if include_stress:
        (
            element_stress,
            nodal_stress,
            stress_samples,
            nodal_stress_by_region,
            stress_fields,
        ) = _stress_values(result)
    else:
        element_stress, nodal_stress, stress_samples = {}, {}, {}
    for key, label in _stress_labels(element_stress, nodal_stress, stress_samples):
        if nodal_stress and all(
            key in values for values in nodal_stress.values()
        ):
            values = np.asarray([
                nodal_stress.get(
                    geometry.point_index_to_node_id[index], {}
                ).get(key, np.nan)
                for index in range(len(geometry.points))
            ], dtype=float)
            _add_field(
                fields,
                f"NODAL:{key}",
                f"{stress_position_labels['NODAL']}{label}",
                "point",
                values,
            )
        if stress_samples and all(
            any(key in sample.values for sample in samples)
            for samples in stress_samples.values()
        ):
            element_nodal_values = np.asarray([
                sample.values[key]
                for node_id in geometry.point_index_to_node_id.values()
                for sample in stress_samples.get(node_id, ())
                if key in sample.values
            ], dtype=float)
            _add_field(
                fields,
                f"EN:{key}",
                f"单元节点（不平均）{label}",
                "element_node",
                element_nodal_values,
            )
        if element_stress and all(key in values for values in element_stress.values()):
            values = np.asarray([
                element_stress.get(geometry.cell_index_to_element_id[index], {}).get(key, np.nan)
                for index in range(len(geometry.cells))
            ], dtype=float)
            _add_field(fields, f"CENTROID:{key}", label, "cell", values)
    integration_field = stress_fields.get(field.StressPosition.INTEGRATION_POINT)
    if integration_field is not None:
        for component, label in _stress_field_labels(integration_field):
            _add_field(
                fields,
                f"IP:{component}",
                label,
                "integration_point",
                np.asarray([
                    record.values(integration_field.component_names)[component]
                    for record in integration_field.records
                ], dtype=float),
            )
    data = ResultData(
        displacement,
        reactions,
        fields,
        nodal_values,
        element_stress,
        nodal_stress,
        stress_samples,
        result,
        geometry,
        nodal_stress_by_region=nodal_stress_by_region,
        stress_fields=stress_fields,
        stress_position_labels=stress_position_labels,
    )
    _register_stress_catalog(data)
    return data


_STRESS_LABELS = {
    "LE11": "轴向应变",
    "S11": "S11",
    "S22": "S22",
    "S33": "S33",
    "S12": "S12",
    "S13": "S13",
    "S23": "S23",
    "Mises": "Mises",
    "MaxPrincipal": "最大主应力",
    "MidPrincipal": "中间主应力",
    "MinPrincipal": "最小主应力",
    "S11Max": "最大轴向应力",
    "S11Min": "最小轴向应力",
    "S11AbsMax": "最大绝对值轴向应力",
}


def _register_stress_catalog(data: ResultData) -> None:
    """Advertise supported stress fields without recovering their values."""
    result = data._source_result
    if result is None:
        return
    try:
        type_keys = dispatch.resolve_type_keys(result.model.mesh, None)
        group = dispatch.stress_group_for_keys(type_keys)
    except ValueError:
        return
    if group == "line":
        prefixes = ()
        components_by_prefix: dict[str, tuple[str, ...]] = {}
        if "truss2" in type_keys:
            prefixes += ("CENTROID",)
            components_by_prefix["CENTROID"] = ("LE11", "S11", "Mises")
        if type_keys == ("beam2",):
            prefixes += ("NODAL",)
            components_by_prefix["NODAL"] = (
                "S11Max",
                "S11Min",
                "S11AbsMax",
            )
    elif group == "plane":
        prefixes = ("IP", "CENTROID", "EN", "NODAL")
        components = (
            "S11", "S22", "S33", "S12", "Mises",
            "MaxPrincipal", "MidPrincipal", "MinPrincipal",
        )
        components_by_prefix = {
            prefix: components for prefix in prefixes
        }
    else:
        prefixes = ("IP", "CENTROID", "EN", "NODAL")
        components = (
            "S11", "S22", "S33", "S12", "S13", "S23", "Mises",
            "MaxPrincipal", "MidPrincipal", "MinPrincipal",
        )
        components_by_prefix = {
            prefix: components for prefix in prefixes
        }
    associations = {
        "IP": "integration_point",
        "CENTROID": "cell",
        "EN": "element_node",
        "NODAL": "point",
    }
    for prefix in prefixes:
        for component in components_by_prefix[prefix]:
            key = f"{prefix}:{component}"
            if key in data.fields:
                continue
            label = _STRESS_LABELS[component]
            if prefix == "NODAL":
                label = f"{data.stress_position_label(prefix)}{label}"
            elif prefix == "EN":
                label = f"单元节点（不平均）{label}"
            data.fields[key] = ScalarField(
                key,
                label,
                associations[prefix],
                np.empty(0, dtype=float),
                ready=False,
            )


def ensure_stress_data(
    data: ResultData,
    prefixes: str | tuple[str, ...] | None = None,
) -> bool:
    """Recover and cache stress data when a GUI consumer first requests it."""
    if prefixes is None:
        requested = data.available_stress_prefixes()
    elif isinstance(prefixes, str):
        requested = (prefixes,)
    else:
        requested = tuple(prefixes)
    aliases = {"N": "NODAL", "E": "CENTROID"}
    requested = tuple(aliases.get(prefix, prefix) for prefix in requested)
    requested_keys = [
        key
        for key, scalar in data.fields.items()
        if key.split(":", 1)[0] in requested and not scalar.ready
    ]
    if not requested_keys:
        return False
    result = data._source_result
    geometry = data._source_geometry
    if result is None or geometry is None:
        raise RuntimeError("结果数据缺少应力恢复所需的模型来源")

    group = data._stress_cache.get("group")
    if group is None:
        type_keys = dispatch.resolve_type_keys(result.model.mesh, None)
        group = dispatch.stress_group_for_keys(type_keys)
        data._stress_cache["group"] = group

    if group == "line":
        if "line" not in data._stress_cache:
            element_stress, nodal_stress = _line_stress(result)
            data._stress_cache["element"] = element_stress
            data._stress_cache["line"] = True
            data.element_stress.clear()
            data.element_stress.update(element_stress)
            data.nodal_stress.clear()
            data.nodal_stress.update(nodal_stress)
    elif "continuum" not in data._stress_cache:
        recovery = field.StressRecovery(result.model.mesh, result.U)
        stress_fields = {
            position: recovery.collect(position)
            for position in field.StressPosition
        }
        (
            element_stress,
            nodal_stress,
            stress_samples,
            nodal_by_region,
        ) = _adapt_core_stress_fields(stress_fields)
        data.stress_fields.clear()
        data.stress_fields.update(stress_fields)
        data.element_stress.clear()
        data.element_stress.update(element_stress)
        data.nodal_stress.clear()
        data.nodal_stress.update(nodal_stress)
        data.nodal_stress_samples.clear()
        data.nodal_stress_samples.update(stress_samples)
        data.nodal_stress_by_region.clear()
        data.nodal_stress_by_region.update(nodal_by_region)
        data._stress_cache["continuum"] = recovery

    _populate_cached_stress_fields(
        data,
        geometry,
        data.element_stress,
        data.nodal_stress,
        data.nodal_stress_samples,
    )
    return True


def recovered_stress_data(
    data: ResultData,
    prefixes: str | tuple[str, ...] | None = None,
) -> ResultData:
    """Recover stress into a detached container safe for background work."""
    detached = replace(
        data,
        fields=dict(data.fields),
        element_stress=dict(data.element_stress),
        nodal_stress=dict(data.nodal_stress),
        nodal_stress_samples=dict(data.nodal_stress_samples),
        _stress_cache=dict(data._stress_cache),
        nodal_stress_by_region=dict(data.nodal_stress_by_region),
        stress_fields=dict(data.stress_fields),
        stress_position_labels=dict(data.stress_position_labels),
    )
    ensure_stress_data(detached, prefixes)
    return detached


def _populate_cached_stress_fields(
    data: ResultData,
    geometry: ModelGeometry,
    element_stress: dict[int, dict[str, float]],
    nodal_stress: dict[int, dict[str, float]],
    stress_samples: dict[int, tuple[StressSample, ...]],
) -> None:
    """Replace lazy catalog entries with the recovered numeric arrays."""
    for component, label in _stress_labels(
        element_stress, nodal_stress, stress_samples
    ):
        if nodal_stress and all(
            component in values for values in nodal_stress.values()
        ):
            values = np.asarray([
                nodal_stress.get(
                    geometry.point_index_to_node_id[index], {}
                ).get(component, np.nan)
                for index in range(len(geometry.points))
            ], dtype=float)
            _add_field(
                data.fields,
                f"NODAL:{component}",
                f"{data.stress_position_label('NODAL')}{label}",
                "point",
                values,
            )
        if stress_samples and all(
            any(component in sample.values for sample in samples)
            for samples in stress_samples.values()
        ):
            element_nodal_values = np.asarray([
                sample.values[component]
                for node_id in geometry.point_index_to_node_id.values()
                for sample in stress_samples.get(node_id, ())
                if component in sample.values
            ], dtype=float)
            _add_field(
                data.fields,
                f"EN:{component}",
                f"单元节点（不平均）{label}",
                "element_node",
                element_nodal_values,
            )
        if element_stress and all(
            component in values for values in element_stress.values()
        ):
            values = np.asarray([
                element_stress.get(
                    geometry.cell_index_to_element_id[index], {}
                ).get(component, np.nan)
                for index in range(len(geometry.cells))
            ], dtype=float)
            _add_field(
                data.fields,
                f"CENTROID:{component}",
                label,
                "cell",
                values,
            )
    integration_field = data.stress_fields.get(
        field.StressPosition.INTEGRATION_POINT
    )
    if integration_field is not None:
        for component, label in _stress_field_labels(integration_field):
            _add_field(
                data.fields,
                f"IP:{component}",
                label,
                "integration_point",
                np.asarray([
                    record.values(integration_field.component_names)[component]
                    for record in integration_field.records
                ], dtype=float),
            )


def deformed_points(geometry: ModelGeometry, data: ResultData, scale: float) -> np.ndarray:
    """按指定比例返回变形坐标。"""
    return geometry.points + float(scale) * data.displacement_vectors


def automatic_deformation_scale(geometry: ModelGeometry, data: ResultData) -> float:
    """令最大位移约为模型包围盒 10% 的自动比例。"""
    if len(geometry.points) == 0:
        return 1.0
    span = float(np.linalg.norm(np.ptp(geometry.points, axis=0)))
    maximum = float(np.max(np.linalg.norm(data.displacement_vectors, axis=1)))
    return 1.0 if maximum <= 0.0 or span <= 0.0 else 0.1 * span / maximum


def _nodal_vectors(mesh: Any, values: np.ndarray, geometry: ModelGeometry) -> np.ndarray:
    vectors = np.zeros((len(geometry.points), 3), dtype=float)
    translation_count = 3 if mesh.nodes and hasattr(mesh.nodes[0], "z") else 2
    for node_id, point_index in geometry.node_id_to_point_index.items():
        for component in range(min(mesh.dofs_per_node, translation_count)):
            vectors[point_index, component] = float(values[mesh.global_dof(node_id, component)])
    return vectors


def _stress_values(
    result: Any,
) -> tuple[
    dict[int, dict[str, float]],
    dict[int, dict[str, float]],
    dict[int, tuple[StressSample, ...]],
    dict[tuple[int, object], dict[str, float]],
    dict[field.StressPosition, field.StressField],
]:
    mesh = result.model.mesh
    try:
        type_keys = dispatch.resolve_type_keys(mesh, None)
        group = dispatch.stress_group_for_keys(type_keys)
    except ValueError:
        return {}, {}, {}, {}, {}
    if group == "line":
        element_values, nodal_values = _line_stress(result)
        return element_values, nodal_values, {}, {}, {}
    recovery = field.StressRecovery(mesh, result.U)
    stress_fields = {
        position: recovery.collect(position)
        for position in field.StressPosition
    }
    (
        element_values,
        nodal_values,
        nodal_samples,
        nodal_by_region,
    ) = _adapt_core_stress_fields(stress_fields)
    return (
        element_values,
        nodal_values,
        nodal_samples,
        nodal_by_region,
        stress_fields,
    )


def _adapt_core_stress_fields(
    stress_fields: dict[field.StressPosition, field.StressField],
) -> tuple[
    dict[int, dict[str, float]],
    dict[int, dict[str, float]],
    dict[int, tuple[StressSample, ...]],
    dict[tuple[int, object], dict[str, float]],
]:
    """Adapt canonical core records to the existing GUI query containers."""
    centroid_field = stress_fields[field.StressPosition.CENTROID]
    element_values = {
        int(record.elem_id): record.values(centroid_field.component_names)
        for record in centroid_field.records
        if record.elem_id is not None
    }

    element_nodal_field = stress_fields[field.StressPosition.ELEMENT_NODAL]
    samples_by_node: dict[int, list[StressSample]] = {}
    for record in element_nodal_field.records:
        if (
            record.node_id is None
            or record.elem_id is None
            or record.local_node is None
        ):
            continue
        samples_by_node.setdefault(record.node_id, []).append(
            StressSample(
                record.elem_id,
                record.local_node,
                record.region_key,
                record.weight,
                record.values(element_nodal_field.component_names),
            )
        )
    nodal_samples = {
        node_id: tuple(samples)
        for node_id, samples in samples_by_node.items()
    }

    nodal_field = stress_fields[field.StressPosition.NODAL]
    nodal_by_region: dict[tuple[int, object], dict[str, float]] = {}
    records_by_node: dict[int, list[field.StressRecord]] = {}
    for record in nodal_field.records:
        if record.node_id is None:
            continue
        nodal_by_region[(record.node_id, record.region_key)] = record.values(
            nodal_field.component_names
        )
        records_by_node.setdefault(record.node_id, []).append(record)
    nodal_values = {
        node_id: records[0].values(nodal_field.component_names)
        for node_id, records in records_by_node.items()
        if len(records) == 1
    }
    return element_values, nodal_values, nodal_samples, nodal_by_region


def _line_stress(
    result: Any,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """Recover canonical Truss2 element and Beam2 nodal envelope results."""
    mesh = result.model.mesh
    element_values: dict[int, dict[str, float]] = {}
    nodal_values: dict[int, dict[str, float]] = {}
    lookup = {int(node.id): node for node in mesh.nodes}
    for element in mesh.elements:
        if dispatch.type_key_from_name(element.type) != "truss2":
            continue
        strain, stress, mises = get_element_kernel(element.type).element_stress(
            mesh, element, result.U, lookup
        )
        element_values[int(element.id)] = {
            "LE11": strain,
            "S11": stress,
            "Mises": mises,
        }
    type_keys = dispatch.resolve_type_keys(mesh, None)
    if type_keys == ("beam2",):
        nodal_values = {
            row.node_id: {
                "S11Max": row.maximum,
                "S11Min": row.minimum,
                "S11AbsMax": row.absolute_maximum,
            }
            for row in beam.nodal_envelope(result)
        }
    return element_values, nodal_values


def _stress_labels(
    element: dict[int, dict[str, float]],
    nodal: dict[int, dict[str, float]],
    samples: dict[int, tuple[StressSample, ...]],
):
    rows = [*element.values(), *nodal.values()]
    rows.extend(sample.values for group in samples.values() for sample in group)
    available = set().union(*(values.keys() for values in rows)) if rows else set()
    labels = {
        "LE11": "轴向应变", "S11": "S11", "S22": "S22", "S33": "S33",
        "S12": "S12", "S13": "S13", "S23": "S23", "Mises": "Mises",
        "MaxPrincipal": "最大主应力", "MidPrincipal": "中间主应力",
        "MinPrincipal": "最小主应力",
        "S11Max": "最大轴向应力", "S11Min": "最小轴向应力",
        "S11AbsMax": "最大绝对值轴向应力",
    }
    return tuple((key, labels[key]) for key in labels if key in available)


def _stress_field_labels(
    stress_field: field.StressField,
) -> tuple[tuple[str, str], ...]:
    available = {
        *stress_field.component_names,
        "Mises",
        "MaxPrincipal",
        "MidPrincipal",
        "MinPrincipal",
    }
    return tuple(
        (key, _STRESS_LABELS[key])
        for key in _STRESS_LABELS
        if key in available
    )


def _add_field(fields: dict[str, ScalarField], key: str, label: str, association: str, values: np.ndarray) -> None:
    fields[key] = ScalarField(key, label, association, np.asarray(values, dtype=float))
