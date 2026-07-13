"""结果查询的数据准备与编号解析。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .result_adapter import ResultData


NODE_DISPLACEMENT = "节点位移"
NODE_REACTION = "节点反力"
ELEMENT_STRESS = "单元应力"
NODAL_STRESS = "节点应力"


@dataclass(frozen=True, slots=True)
class QueryRecord:
    """查询表格中的一行。"""

    object_id: int
    source_element_id: int | None
    values: dict[str, float]


def available_query_types(data: ResultData, object_kind: str) -> tuple[str, ...]:
    """返回当前对象类型实际可查询的结果。"""
    if object_kind == "element":
        return (ELEMENT_STRESS,) if data.element_stress else ()
    result = [NODE_DISPLACEMENT, NODE_REACTION]
    if data.nodal_stress_samples:
        result.append(NODAL_STRESS)
    return tuple(result)


def available_components(data: ResultData, query_type: str) -> tuple[str, ...]:
    """返回查询类型实际具有的分量。"""
    if query_type == NODE_DISPLACEMENT:
        return _ordered_keys(data.nodal_values, ("U", "U1", "U2", "U3", "R3"))
    if query_type == NODE_REACTION:
        return _ordered_keys(data.nodal_values, ("RF", "RF1", "RF2", "RF3", "RM3"))
    if query_type not in {ELEMENT_STRESS, NODAL_STRESS}:
        return ()
    source = data.element_stress if query_type == ELEMENT_STRESS else {
        node_id: sample.values
        for node_id, samples in data.nodal_stress_samples.items()
        for sample in samples[:1]
    }
    return _ordered_keys(
        source,
        ("S11", "S22", "S33", "S12", "S13", "S23", "Mises", "MaxPrincipal", "MinPrincipal", "LE11"),
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
                records.append(QueryRecord(object_id, None, _prefix_values(values, ("U", "U1", "U2", "U3", "R3"))))
        elif query_type == NODE_REACTION:
            values = data.nodal_values.get(object_id)
            if values is not None:
                records.append(QueryRecord(object_id, None, _prefix_values(values, ("RF", "RF1", "RF2", "RF3", "RM3"))))
        elif query_type == ELEMENT_STRESS:
            values = data.element_stress.get(object_id)
            if values is not None:
                records.append(QueryRecord(object_id, None, dict(values)))
        elif query_type == NODAL_STRESS:
            for sample in data.nodal_stress_samples.get(object_id, ()):
                records.append(QueryRecord(object_id, sample.element_id, dict(sample.values)))
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
