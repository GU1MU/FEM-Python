"""正式求解结果到可视化场变量的适配。"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any

import numpy as np

from fem.elements import get_element_kernel
from fem.post.stress import dispatch, field, invariants
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

    def field_ready(self, key: str) -> bool:
        """Return whether a catalog field already has numeric values."""
        scalar = self.fields.get(str(key))
        return scalar is not None and scalar.ready

    def available_stress_prefixes(self) -> tuple[str, ...]:
        """Return stress positions advertised by the current element family."""
        return tuple(
            prefix
            for prefix in ("N", "EN", "E")
            if any(key.startswith(f"{prefix}:") for key in self.fields)
        )


def build_result_data(
    result: Any,
    geometry: ModelGeometry,
    *,
    include_stress: bool = True,
) -> ResultData:
    """从 ModelResult 读取数据，不复制求解逻辑。"""
    mesh = result.model.mesh
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
    rotations = None
    moments = None
    if translation_count == 2 and mesh.dofs_per_node >= 3:
        rotations = np.asarray([
            result.U[mesh.global_dof(geometry.point_index_to_node_id[index], 2)]
            for index in range(len(geometry.points))
        ])
        moments = np.asarray([
            result.reactions[mesh.global_dof(geometry.point_index_to_node_id[index], 2)]
            for index in range(len(geometry.points))
        ])
        _add_field(fields, "R3", "转角 R3", "point", rotations)
        _add_field(fields, "RM3", "反力矩 RM3", "point", moments)
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
        if rotations is not None and moments is not None:
            values["R3"] = float(rotations[point_index])
            values["RM3"] = float(moments[point_index])
        nodal_values[node_id] = values
    if include_stress:
        element_stress, nodal_stress, stress_samples = _stress_values(result)
    else:
        element_stress, nodal_stress, stress_samples = {}, {}, {}
    for key, label in _stress_labels(element_stress, nodal_stress, stress_samples):
        if stress_samples and all(
            any(key in sample.values for sample in samples)
            for samples in stress_samples.values()
        ):
            values = np.asarray([
                _fallback_nodal_scalar(
                    stress_samples.get(geometry.point_index_to_node_id[index], ()), key
                )
                for index in range(len(geometry.points))
            ], dtype=float)
            _add_field(fields, f"N:{key}", f"节点平均{label}", "point", values)
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
            _add_field(fields, f"E:{key}", label, "cell", values)
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
    "MinPrincipal": "最小主应力",
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
        prefixes = ("E",)
        components = ("LE11", "S11", "Mises")
    elif group == "plane":
        prefixes = ("N", "EN", "E")
        components = (
            "S11", "S22", "S12", "Mises", "MaxPrincipal", "MinPrincipal",
        )
    else:
        prefixes = ("N", "EN", "E")
        components = (
            "S11", "S22", "S33", "S12", "S13", "S23", "Mises",
            "MaxPrincipal", "MinPrincipal",
        )
    associations = {"N": "point", "EN": "element_node", "E": "cell"}
    for prefix in prefixes:
        for component in components:
            key = f"{prefix}:{component}"
            if key in data.fields:
                continue
            label = _STRESS_LABELS[component]
            if prefix == "N":
                label = f"节点平均{label}"
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

    needs_nodal = any(prefix in {"N", "EN"} for prefix in requested)
    if needs_nodal and "nodal" not in data._stress_cache:
        raw = data._stress_cache.get("raw")
        if raw is None:
            raw = field.collect(result.model.mesh, result.U)
            data._stress_cache["raw"] = raw
        nodal = _nodal_stress_from_raw(raw)
        data._stress_cache["nodal"] = nodal
        nodal_stress, stress_samples = nodal
        data.nodal_stress.clear()
        data.nodal_stress.update(nodal_stress)
        data.nodal_stress_samples.clear()
        data.nodal_stress_samples.update(stress_samples)

    if "E" in requested and "element" not in data._stress_cache:
        mesh = result.model.mesh
        if group == "line":
            element_stress = _line_element_stress(mesh, result.U)
        elif group == "solid":
            element_stress = _solid_element_stress(mesh, result.U)
        else:
            raw = data._stress_cache.get("raw")
            if raw is None:
                raw = field.collect(mesh, result.U)
                data._stress_cache["raw"] = raw
            element_stress = _plane_element_stress(raw)
        data._stress_cache["element"] = element_stress
        data.element_stress.clear()
        data.element_stress.update(element_stress)

    _populate_cached_stress_fields(
        data,
        geometry,
        data.element_stress,
        data.nodal_stress,
        data.nodal_stress_samples,
    )
    return True


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
        if stress_samples and all(
            any(component in sample.values for sample in samples)
            for samples in stress_samples.values()
        ):
            values = np.asarray([
                _fallback_nodal_scalar(
                    stress_samples.get(
                        geometry.point_index_to_node_id[index], ()
                    ),
                    component,
                )
                for index in range(len(geometry.points))
            ], dtype=float)
            _add_field(
                data.fields,
                f"N:{component}",
                f"节点平均{label}",
                "point",
                values,
            )
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
            _add_field(data.fields, f"E:{component}", label, "cell", values)


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
]:
    mesh = result.model.mesh
    try:
        type_keys = dispatch.resolve_type_keys(mesh, None)
        group = dispatch.stress_group_for_keys(type_keys)
    except ValueError:
        return {}, {}, {}
    if group == "line":
        return _line_element_stress(mesh, result.U), {}, {}
    raw = field.collect(mesh, result.U)
    element_values = (
        _solid_element_stress(mesh, result.U)
        if group == "solid"
        else _plane_element_stress(raw)
    )
    nodal_values, nodal_samples = _nodal_stress_from_raw(raw)
    return element_values, nodal_values, nodal_samples


def _nodal_stress_from_raw(
    raw: Any,
) -> tuple[
    dict[int, dict[str, float]],
    dict[int, tuple[StressSample, ...]],
]:
    """Convert one cached element-nodal recovery into GUI stress samples."""
    nodal_values: dict[int, dict[str, float]] = {}
    nodal_samples: dict[int, tuple[StressSample, ...]] = {}
    for node_id, contributions in raw.contributions_by_node.items():
        samples = tuple(
            StressSample(
                int(item.elem_id),
                int(item.local_node),
                item.region_key,
                float(item.weight),
                _stress_dict(
                    np.asarray(item.components),
                    len(raw.component_names),
                    item.plane_type,
                    item.poisson_ratio,
                ),
            )
            for item in contributions
        )
        nodal_samples[node_id] = samples
        if not samples or len({sample.region_key for sample in samples}) != 1:
            continue
        keys = set.intersection(*(set(sample.values) for sample in samples))
        weights = np.asarray([sample.weight for sample in samples], dtype=float)
        nodal_values[node_id] = {
            key: float(np.average(
                [sample.values[key] for sample in samples], weights=weights
            ))
            for key in keys
        }
    return nodal_values, nodal_samples


def _plane_element_stress(raw: Any) -> dict[int, dict[str, float]]:
    """按正式平面单元导出语义平均单元节点应力。"""
    groups: dict[int, list[dict[str, float]]] = {}
    for contributions in raw.contributions_by_node.values():
        for item in contributions:
            groups.setdefault(item.elem_id, []).append(
                _stress_dict(
                    np.asarray(item.components), len(raw.component_names),
                    item.plane_type, item.poisson_ratio,
                )
            )
    return {
        elem_id: {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        for elem_id, rows in groups.items()
    }


def _solid_element_stress(mesh: Any, displacement: np.ndarray) -> dict[int, dict[str, float]]:
    """使用现有单元核在 Hex 中心或 Tet 重心恢复实体应力。"""
    lookup = {int(node.id): node for node in mesh.nodes}
    values: dict[int, dict[str, float]] = {}
    for element in mesh.elements:
        type_key = dispatch.type_key_from_name(element.type)
        natural_coordinates = (
            (0.0, 0.0, 0.0)
            if type_key in {"hex8", "hex20"}
            else (0.25, 0.25, 0.25)
        )
        stress = get_element_kernel(element.type).stress_at(
            mesh, element, displacement, *natural_coordinates, lookup
        )
        values[int(element.id)] = _stress_dict(np.asarray(stress), 6)
    return values


def _line_element_stress(mesh: Any, displacement: np.ndarray) -> dict[int, dict[str, float]]:
    values: dict[int, dict[str, float]] = {}
    lookup = {int(node.id): node for node in mesh.nodes}
    for element in mesh.elements:
        if dispatch.type_key_from_name(element.type) != "truss2d":
            continue
        strain, stress, mises = get_element_kernel(element.type).element_stress(
            mesh, element, displacement, lookup
        )
        values[int(element.id)] = {"LE11": strain, "S11": stress, "Mises": mises}
    return values


def _stress_dict(
    components: np.ndarray,
    count: int,
    plane_type: str | None = None,
    poisson_ratio: float | None = None,
) -> dict[str, float]:
    values = np.asarray(components, dtype=float)
    if count == 3:
        s11, s22, s12 = values
        tensor = np.array([[s11, s12, 0.0], [s12, s22, 0.0], [0.0, 0.0, 0.0]])
        result = {
            "S11": float(s11), "S22": float(s22), "S12": float(s12),
            "Mises": float(invariants.von_mises_plane(
                s11, s22, s12, plane_type or "stress", poisson_ratio or 0.0
            )),
        }
    else:
        s11, s22, s33, s12, s23, s13 = values
        tensor = np.array([[s11, s12, s13], [s12, s22, s23], [s13, s23, s33]])
        result = {
            "S11": float(s11), "S22": float(s22), "S33": float(s33),
            "S12": float(s12), "S13": float(s13), "S23": float(s23),
            "Mises": float(invariants.von_mises_3d(s11, s22, s33, s12, s23, s13)),
        }
    principal = np.linalg.eigvalsh(tensor)
    result["MaxPrincipal"] = float(principal[-1])
    result["MinPrincipal"] = float(principal[0])
    return result


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
        "MaxPrincipal": "最大主应力", "MinPrincipal": "最小主应力",
    }
    return tuple((key, labels[key]) for key in labels if key in available)


def _fallback_nodal_scalar(samples: tuple[StressSample, ...], key: str) -> float:
    """为非绘图消费者提供完整节点数组；绘图仍保留区域硬边界。"""
    rows = [sample for sample in samples if key in sample.values]
    if not rows:
        return float("nan")
    return float(np.average(
        [sample.values[key] for sample in rows],
        weights=[sample.weight for sample in rows],
    ))


def _add_field(fields: dict[str, ScalarField], key: str, label: str, association: str, values: np.ndarray) -> None:
    fields[key] = ScalarField(key, label, association, np.asarray(values, dtype=float))
