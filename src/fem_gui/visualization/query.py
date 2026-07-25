"""结果查询的数据准备与编号解析。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from fem.post.stress.field import StressPosition
from .result_adapter import ResultData


NODE_DISPLACEMENT = "节点位移"
NODE_REACTION = "节点反力"
INTEGRATION_POINT_STRESS = "积分点应力"
CENTROID_STRESS = "单元质心应力"
ELEMENT_NODAL_STRESS = "单元节点应力（不平均）"
NODE_ELEMENT_NODAL_STRESS = "节点处单元贡献"
NODAL_STRESS = "节点平均应力"
NODAL_ENVELOPE_STRESS = "节点包络应力"


def _nodal_query_type(data: ResultData) -> str:
    return (
        NODAL_ENVELOPE_STRESS
        if data.stress_position_label("NODAL") == "节点包络"
        else NODAL_STRESS
    )


def _is_nodal_stress(query_type: str) -> bool:
    return query_type in {NODAL_STRESS, NODAL_ENVELOPE_STRESS}


@dataclass(frozen=True, slots=True)
class QueryRecord:
    """查询表格中的一行。"""

    object_id: int
    source_element_id: int | None
    values: dict[str, float]
    integration_point: int | None = None
    local_node: int | None = None


def available_query_types(data: ResultData, object_kind: str) -> tuple[str, ...]:
    """返回当前对象类型实际可查询的结果。"""
    if object_kind == "element":
        result = []
        if StressPosition.INTEGRATION_POINT in data.stress_fields:
            result.append(INTEGRATION_POINT_STRESS)
        if (
            StressPosition.CENTROID in data.stress_fields
            or data.element_stress
        ):
            result.append(CENTROID_STRESS)
        if StressPosition.ELEMENT_NODAL in data.stress_fields:
            result.append(ELEMENT_NODAL_STRESS)
        return tuple(result)
    result = [NODE_DISPLACEMENT, NODE_REACTION]
    if StressPosition.ELEMENT_NODAL in data.stress_fields:
        result.append(NODE_ELEMENT_NODAL_STRESS)
    if StressPosition.NODAL in data.stress_fields or data.nodal_stress:
        result.append(_nodal_query_type(data))
    return tuple(result)


def available_components(data: ResultData, query_type: str) -> tuple[str, ...]:
    """返回查询类型实际具有的分量。"""
    if query_type == NODE_DISPLACEMENT:
        return _ordered_keys(
            data.nodal_values,
            ("U", "U1", "U2", "U3", "R1", "R2", "R3"),
        )
    if query_type == NODE_REACTION:
        return _ordered_keys(
            data.nodal_values,
            ("RF", "RF1", "RF2", "RF3", "RM1", "RM2", "RM3"),
        )
    positions = {
        INTEGRATION_POINT_STRESS: StressPosition.INTEGRATION_POINT,
        CENTROID_STRESS: StressPosition.CENTROID,
        ELEMENT_NODAL_STRESS: StressPosition.ELEMENT_NODAL,
        NODE_ELEMENT_NODAL_STRESS: StressPosition.ELEMENT_NODAL,
        NODAL_STRESS: StressPosition.NODAL,
        NODAL_ENVELOPE_STRESS: StressPosition.NODAL,
    }
    position = positions.get(query_type)
    if position is None:
        return ()
    stress_field = data.stress_fields.get(position)
    if stress_field is None:
        if query_type == CENTROID_STRESS:
            return _ordered_keys(
                data.element_stress,
                ("S11", "Mises", "LE11"),
            )
        if _is_nodal_stress(query_type):
            return _ordered_keys(
                data.nodal_stress,
                (
                    "S11", "S22", "S33", "S12", "S13", "S23",
                    "Mises", "MaxPrincipal", "MidPrincipal", "MinPrincipal",
                    "S11Max", "S11Min", "S11AbsMax",
                ),
            )
        return ()
    source = {
        index: record.values(stress_field.component_names)
        for index, record in enumerate(stress_field.records)
    }
    return _ordered_keys(
        source,
        (
            "S11", "S22", "S33", "S12", "S13", "S23", "Mises",
            "MaxPrincipal", "MidPrincipal", "MinPrincipal", "LE11",
        ),
    )


def query_records(
    data: ResultData,
    query_type: str,
    object_ids: Iterable[int],
) -> tuple[QueryRecord, ...]:
    """按真实 FEM 编号生成查询记录。"""
    records: list[QueryRecord] = []
    for object_id in object_ids:
        object_id = int(object_id)
        if query_type == NODE_DISPLACEMENT:
            values = data.nodal_values.get(object_id)
            if values is not None:
                records.append(QueryRecord(
                    object_id,
                    None,
                    _prefix_values(
                        values,
                        ("U", "U1", "U2", "U3", "R1", "R2", "R3"),
                    ),
                ))
        elif query_type == NODE_REACTION:
            values = data.nodal_values.get(object_id)
            if values is not None:
                records.append(QueryRecord(
                    object_id,
                    None,
                    _prefix_values(
                        values,
                        (
                            "RF", "RF1", "RF2", "RF3",
                            "RM1", "RM2", "RM3",
                        ),
                    ),
                ))
        elif query_type == INTEGRATION_POINT_STRESS:
            stress_field = data.stress_fields.get(StressPosition.INTEGRATION_POINT)
            if stress_field is not None:
                for record in stress_field.records:
                    if record.elem_id == object_id:
                        records.append(QueryRecord(
                            object_id,
                            None,
                            record.values(stress_field.component_names),
                            integration_point=record.integration_point,
                        ))
        elif query_type == CENTROID_STRESS:
            stress_field = data.stress_fields.get(StressPosition.CENTROID)
            if stress_field is not None:
                for record in stress_field.records:
                    if record.elem_id == object_id:
                        records.append(QueryRecord(
                            object_id,
                            None,
                            record.values(stress_field.component_names),
                        ))
            else:
                values = data.element_stress.get(object_id)
                if values is not None:
                    records.append(QueryRecord(object_id, None, dict(values)))
        elif query_type == ELEMENT_NODAL_STRESS:
            stress_field = data.stress_fields.get(StressPosition.ELEMENT_NODAL)
            if stress_field is not None:
                for record in stress_field.records:
                    if record.elem_id == object_id:
                        records.append(QueryRecord(
                            object_id,
                            None,
                            record.values(stress_field.component_names),
                            local_node=record.local_node,
                        ))
        elif query_type == NODE_ELEMENT_NODAL_STRESS:
            stress_field = data.stress_fields.get(StressPosition.ELEMENT_NODAL)
            if stress_field is not None:
                for record in stress_field.records:
                    if record.node_id == object_id:
                        records.append(QueryRecord(
                            object_id,
                            record.elem_id,
                            record.values(stress_field.component_names),
                            local_node=record.local_node,
                        ))
        elif _is_nodal_stress(query_type):
            stress_field = data.stress_fields.get(StressPosition.NODAL)
            if stress_field is not None:
                for record in stress_field.records:
                    if record.node_id == object_id:
                        records.append(QueryRecord(
                            object_id,
                            None,
                            record.values(stress_field.component_names),
                        ))
            else:
                values = data.nodal_stress.get(object_id)
                if values is not None:
                    records.append(QueryRecord(object_id, None, dict(values)))
    return tuple(records)


def parse_object_ids(text: str, valid_ids: Iterable[int]) -> tuple[int, ...]:
    """解析逗号、空格及闭区间形式的 FEM 编号。"""
    valid = {int(value) for value in valid_ids}
    result: list[int] = []
    for token in re.split(r"[\s,，;；]+", text.strip()):
        if not token:
            continue
        match = re.fullmatch(r"(-?\d+)\s*[-~～]\s*(-?\d+)", token)
        if match:
            first, last = (int(value) for value in match.groups())
            step = 1 if last >= first else -1
            candidates = range(first, last + step, step)
        else:
            try:
                candidates = (int(token),)
            except ValueError as error:
                raise ValueError(f"无法识别的有限元编号：{token}") from error
        for candidate in candidates:
            if candidate not in valid:
                raise ValueError(f"有限元编号不存在：{candidate}")
            if candidate not in result:
                result.append(candidate)
    if not result:
        raise ValueError("请输入至少一个有限元编号")
    return tuple(result)


def _ordered_keys(source: dict[int, dict[str, float]], order: tuple[str, ...]) -> tuple[str, ...]:
    available = set().union(*(values.keys() for values in source.values())) if source else set()
    return tuple(key for key in order if key in available)


def _prefix_values(values: dict[str, float], order: tuple[str, ...]) -> dict[str, float]:
    return {key: values[key] for key in order if key in values}
