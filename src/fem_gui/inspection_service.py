"""有限元模型对象的只读索引与结构化信息准备。"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from typing import Any

from fem.application import RegionRef, resolve_effective_beam_frames
from fem.core._constraint_targets import (
    displacement_target_kind,
    resolve_displacement_node_ids,
)
from fem.application.results import (
    ElementResultInspectionRequest,
    FieldPosition,
    FieldState,
    NodeResultInspectionRequest,
    ResultInspectionField,
    ResultInspectionRequest,
    ResultInspectionResult,
    ResultProvider,
    ResultQueryValidationError,
)
from fem.post.fields import encode_result_region_key
from .result_presentation import (
    result_field_is_visible,
    result_position_label,
)


_RESULT_VARIABLE_LABELS = {
    "U": "位移 U",
    "UR": "转角 UR",
    "RF": "反力 RF",
    "RM": "反力矩 RM",
    "LE": "对数应变 LE",
    "S": "应力 S",
}
_RESULT_COMPONENT_LABELS = {
    "Magnitude": "模",
    "Mises": "Mises 等效应力",
    "MaxPrincipal": "最大主应力",
    "MidPrincipal": "中间主应力",
    "MinPrincipal": "最小主应力",
    "S11Max": "最大轴向应力",
    "S11Min": "最小轴向应力",
    "S11AbsMax": "最大绝对值轴向应力",
}
_RESULT_STATE_LABELS = {
    FieldState.READY: "就绪",
    FieldState.LAZY: "按需加载",
    FieldState.UNAVAILABLE: "不可用",
}
_RESULT_COLUMNS = (
    "状态",
    "分量",
    "数值",
    "节点",
    "单元",
    "积分点",
    "局部节点",
    "结果区域",
    "平均",
    "诊断",
)


@dataclass(frozen=True, slots=True)
class EntityReference:
    kind: str
    key: object


@dataclass(frozen=True, slots=True)
class InspectionTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    references: tuple[EntityReference | None, ...] = ()


@dataclass(frozen=True, slots=True)
class InspectionPage:
    title: str
    fields: tuple[tuple[str, str], ...] = ()
    tables: tuple[InspectionTable, ...] = ()


@dataclass(frozen=True, slots=True)
class EntityInspection:
    title: str
    kind: str
    key: object
    pages: tuple[InspectionPage, ...]


@dataclass(frozen=True, slots=True)
class EntitySelection:
    node_ids: tuple[int, ...] = ()
    element_ids: tuple[int, ...] = ()


class InspectionService:
    """为一个 FEMModel 建立一次只读反向索引。"""

    def __init__(
        self,
        model: Any,
        *,
        result_provider: ResultProvider | None = None,
        definitions: Any | None = None,
        effective_frame_query: (
            Callable[[RegionRef | int], Any] | None
        ) = None,
    ) -> None:
        self.model = model
        self.result_provider = _require_result_provider(result_provider)
        self.definitions = definitions
        self.section_definitions = tuple(
            getattr(definitions, "sections", ())
        )
        self.region_assignments = tuple(
            getattr(definitions, "assignments", ())
        )
        self._effective_frame_query = (
            effective_frame_query
            if effective_frame_query is not None
            else lambda target: resolve_effective_beam_frames(
                self.model,
                target,
            )
        )
        self.nodes = {int(node.id): node for node in model.mesh.nodes}
        self.elements = {int(element.id): element for element in model.mesh.elements}
        self.node_sets_by_node: dict[int, list[str]] = defaultdict(list)
        self.element_sets_by_element: dict[int, list[str]] = defaultdict(list)
        self.adjacent_elements: dict[int, list[int]] = defaultdict(list)
        self.element_material: dict[int, str] = {}
        self.element_section: dict[int, int] = {}
        self.section_elements: dict[int, tuple[int, ...]] = {}
        self.material_elements: dict[str, set[int]] = defaultdict(set)
        self.node_analysis: dict[int, list[tuple[str, str, str, float, EntityReference]]] = defaultdict(list)
        self.node_set_boundaries: dict[str, list[EntityReference]] = defaultdict(list)
        self.node_set_loads: dict[str, list[EntityReference]] = defaultdict(list)
        self.region_boundaries: dict[
            tuple[str, str],
            list[EntityReference],
        ] = defaultdict(list)
        self.region_loads: dict[tuple[str, str], list[EntityReference]] = defaultdict(list)
        self._all_element_sets = dict(model.element_sets)
        self._all_element_sets.update(model.metadata.get("_abaqus_internal_element_sets", {}))
        self._build_indexes()
        self._element_record_cached = lru_cache(maxsize=4096)(self._make_element_record)
        self._beam_frame_report_cached = lru_cache(maxsize=4096)(
            self._query_effective_beam_frames
        )

    def update_result_provider(
        self,
        result_provider: ResultProvider | None,
    ) -> None:
        """Install one exact immutable provider without materializing fields."""

        self.result_provider = _require_result_provider(result_provider)

    def _query_effective_beam_frames(
        self,
        target: RegionRef | int,
    ) -> Any:
        return self._effective_frame_query(target)

    def _build_indexes(self) -> None:
        for name, node_set in self.model.node_sets.items():
            for node_id in node_set.node_ids:
                self.node_sets_by_node[int(node_id)].append(str(name))
        for name, element_set in self.model.element_sets.items():
            for element_id in element_set.element_ids:
                self.element_sets_by_element[int(element_id)].append(str(name))
        for element_id, element in self.elements.items():
            for node_id in element.node_ids:
                self.adjacent_elements[int(node_id)].append(element_id)
        for section_index, section in enumerate(self.model.sections):
            element_set = self._all_element_sets.get(section.element_set)
            ids = () if element_set is None else tuple(int(value) for value in element_set.element_ids)
            self.section_elements[section_index] = ids
            for element_id in ids:
                self.element_section[element_id] = section_index
                self.element_material[element_id] = section.material
                self.material_elements[section.material].add(element_id)
        for element_id, element in self.elements.items():
            material = getattr(element, "props", {}).get("material")
            if material and element_id not in self.element_material:
                self.element_material[element_id] = str(material)
                self.material_elements[str(material)].add(element_id)
        self._index_analysis_definitions()

    def _index_analysis_definitions(self) -> None:
        for step_index, step in enumerate(self.model.steps):
            for index, boundary in enumerate(step.boundaries):
                reference = EntityReference("boundary", (step_index, index))
                target_kind = displacement_target_kind(boundary)
                if isinstance(boundary.target, str):
                    if target_kind == "node_set":
                        self.node_set_boundaries[str(boundary.target)].append(
                            reference
                        )
                    else:
                        self.region_boundaries[
                            (target_kind, str(boundary.target))
                        ].append(reference)
                component = _component_range(boundary.first_component, boundary.last_component)
                for node_id in resolve_displacement_node_ids(
                    self.model,
                    boundary,
                ):
                    self.node_analysis[node_id].append(
                        (step.name, "位移边界条件", component, float(boundary.value), reference)
                    )
            for index, load in enumerate(step.cloads):
                reference = EntityReference("cload", (step_index, index))
                if isinstance(load.target, str):
                    self.node_set_loads[str(load.target)].append(reference)
                for node_id in self.target_node_ids(load.target):
                    self.node_analysis[node_id].append(
                        (step.name, "节点力", f"U{load.component}", float(load.value), reference)
                    )
            for index, load in enumerate(step.surface_loads):
                self.region_loads[("surface", load.surface)].append(
                    EntityReference("surface_load", (step_index, index))
                )
            for index, load in enumerate(step.edge_loads):
                self.region_loads[("edge", load.edge)].append(
                    EntityReference("edge_load", (step_index, index))
                )

    def target_node_ids(self, target: str | int) -> tuple[int, ...]:
        if isinstance(target, int):
            return (int(target),) if int(target) in self.nodes else ()
        node_set = self.model.node_sets.get(str(target))
        return () if node_set is None else tuple(int(value) for value in node_set.node_ids)

    def element_record(self, element_id: int) -> dict[str, Any]:
        return self._element_record_cached(int(element_id))

    def _make_element_record(self, element_id: int) -> dict[str, Any]:
        element_id = int(element_id)
        element = self.elements[element_id]
        props = dict(getattr(element, "props", {}))
        section_index = self.element_section.get(element_id)
        section = self.model.sections[section_index] if section_index is not None else None
        material = self.element_material.get(element_id)
        effective: dict[str, Any] = {}
        if material in self.model.materials:
            effective.update(self.model.materials[material].properties)
        if section is not None:
            effective.update(section.properties)
        effective.update({key: value for key, value in props.items() if _is_physical_property(key, value)})
        return {
            "id": element_id,
            "type": str(element.type),
            "abaqus_type": props.get("abaqus_type"),
            "node_ids": tuple(int(value) for value in element.node_ids),
            "sets": tuple(self.element_sets_by_element.get(element_id, ())),
            "material": material,
            "section_index": section_index,
            "section_type": None if section is None else section.section_type,
            "properties": effective,
        }

    def node_row(self, node_id: int) -> tuple[object, ...]:
        node = self.nodes[int(node_id)]
        return (
            int(node.id), float(node.x), float(node.y),
            float(getattr(node, "z", 0.0)),
            "、".join(self.node_sets_by_node.get(int(node.id), ())),
        )

    def element_row(self, element_id: int) -> tuple[object, ...]:
        record = self.element_record(element_id)
        nodes = record["node_ids"]
        preview = ", ".join(str(value) for value in nodes[:4]) + (", …" if len(nodes) > 4 else "")
        section = "—" if record["section_index"] is None else f"截面 {record['section_index'] + 1}"
        return (
            record["id"], record["type"], preview,
            "、".join(record["sets"]), record["material"] or "—", section,
        )

    def inspect(self, kind: str, key: object) -> EntityInspection:
        handlers = {
            "model": self._inspect_model, "node": self._inspect_node,
            "element": self._inspect_element, "node_set": self._inspect_node_set,
            "element_set": self._inspect_element_set, "surface": self._inspect_surface,
            "edge": self._inspect_edge, "material": self._inspect_material,
            "section": self._inspect_section, "step": self._inspect_step,
            "boundary": self._inspect_boundary, "cload": self._inspect_cload,
            "surface_load": self._inspect_surface_load, "edge_load": self._inspect_edge_load,
            "line_load": self._inspect_line_load,
            "body_load": self._inspect_body_load,
            "gravity_load": self._inspect_gravity_load,
            "assignment": self._inspect_assignment,
            "output": self._inspect_output,
        }
        if kind not in handlers:
            raise KeyError(f"不支持的信息对象：{kind}")
        return handlers[kind](key)

    def selection_for(self, kind: str, key: object) -> EntitySelection:
        if kind == "node":
            return EntitySelection(node_ids=(int(key),))
        if kind == "element":
            return EntitySelection(element_ids=(int(key),))
        if kind == "node_set":
            return EntitySelection(node_ids=tuple(self.model.node_sets[str(key)].node_ids))
        if kind == "element_set":
            return EntitySelection(element_ids=tuple(self.model.element_sets[str(key)].element_ids))
        if kind == "surface":
            return EntitySelection(element_ids=tuple(sorted({face.elem_id for face in self.model.surfaces[str(key)].faces})))
        if kind == "edge":
            return EntitySelection(element_ids=tuple(sorted({edge.elem_id for edge in self.model.edges[str(key)].edges})))
        if kind == "material":
            return EntitySelection(element_ids=tuple(sorted(self.material_elements.get(str(key), ()))))
        if kind == "section":
            return EntitySelection(element_ids=self.section_elements.get(int(key), ()))
        if kind == "assignment":
            assignment = self.region_assignments[int(key)]
            return EntitySelection(
                element_ids=self.target_element_ids(
                    str(assignment.region_name)
                )
            )
        if kind == "boundary":
            step_index, index = key
            return EntitySelection(
                node_ids=resolve_displacement_node_ids(
                    self.model,
                    self.model.steps[step_index].boundaries[index],
                )
            )
        if kind == "cload":
            step_index, index = key
            return EntitySelection(node_ids=self.target_node_ids(self.model.steps[step_index].cloads[index].target))
        if kind in {"surface_load", "edge_load"}:
            step_index, index = key
            step = self.model.steps[step_index]
            region_kind = "surface" if kind == "surface_load" else "edge"
            region = step.surface_loads[index].surface if region_kind == "surface" else step.edge_loads[index].edge
            return self.selection_for(region_kind, region)
        if kind == "line_load":
            step_index, index = key
            return EntitySelection(element_ids=self.target_element_ids(
                self.model.steps[step_index].line_loads[index].target
            ))
        if kind == "body_load":
            step_index, index = key
            return EntitySelection(element_ids=self.target_element_ids(
                self.model.steps[step_index].body_loads[index].target
            ))
        if kind == "gravity_load":
            step_index, index = key
            target = self.model.steps[step_index].gravity_loads[index].target
            if target is None:
                return EntitySelection(
                    element_ids=tuple(sorted(self.elements))
                )
            return EntitySelection(
                element_ids=self.target_element_ids(target)
            )
        return EntitySelection()

    def target_element_ids(self, target: str | int) -> tuple[int, ...]:
        if isinstance(target, int):
            return (int(target),) if int(target) in self.elements else ()
        element_set = self._all_element_sets.get(str(target))
        return () if element_set is None else tuple(
            int(value) for value in element_set.element_ids
        )

    def _inspect_model(self, _key: object) -> EntityInspection:
        mesh = self.model.mesh
        spatial_dimension = getattr(mesh, "spatial_dimension", None)
        dimension = (
            f"{spatial_dimension}维"
            if spatial_dimension in {1, 2, 3}
            else "三维"
            if mesh.nodes and hasattr(mesh.nodes[0], "z")
            else "二维"
        )
        fields = (
            ("模型名称", str(self.model.name or "模型")), ("空间维度", dimension),
            ("节点数量", str(len(mesh.nodes))), ("单元数量", str(len(mesh.elements))),
            ("总自由度数量", str(mesh.num_dofs)),
            ("单元类型统计", _counter_text(element.type for element in mesh.elements)),
            ("节点集数量", str(len(self.model.node_sets))),
            ("单元集数量", str(len(self.model.element_sets))),
            ("表面和边数量", str(len(self.model.surfaces) + len(self.model.edges))),
            ("材料数量", str(len(self.model.materials))), ("截面数量", str(len(self.model.sections))),
            ("分析步数量", str(len(self.model.steps))),
        )
        return EntityInspection("模型概况", "model", None, (InspectionPage("概况", fields),))

    def _inspect_node(self, key: object) -> EntityInspection:
        node_id = int(key)
        node = self.nodes[node_id]
        coords = [float(node.x), float(node.y)]
        if getattr(self.model.mesh, "spatial_dimension", None) != 2 and hasattr(node, "z"):
            coords.append(float(node.z))
        adjacent = tuple(
            (str(element_id), str(self.elements[element_id].type))
            for element_id in self.adjacent_elements.get(node_id, ())
        )
        references = tuple(EntityReference("element", int(row[0])) for row in adjacent)
        pages = [InspectionPage(
            "基本信息",
            (("节点编号", str(node_id)), ("坐标", ", ".join(format_number(value) for value in coords)),
             ("所属节点集", "、".join(self.node_sets_by_node.get(node_id, ())) or "—")),
            (InspectionTable("相邻单元", ("单元编号", "单元类型"), adjacent, references),),
        )]
        analysis_rows = tuple(
            (step, kind, component, format_number(value))
            for step, kind, component, value, _reference in self.node_analysis.get(node_id, ())
        )
        if analysis_rows:
            pages.append(InspectionPage(
                "分析定义", tables=(InspectionTable(
                    "分析定义", ("分析步", "类型", "分量", "数值"), analysis_rows,
                    tuple(row[4] for row in self.node_analysis[node_id]),
                ),),
            ))
        result_page = self._provider_result_page(
            NodeResultInspectionRequest(node_id)
        )
        if result_page is not None:
            pages.append(result_page)
        return EntityInspection(
            f"节点 {node_id}",
            "node",
            node_id,
            tuple(pages),
        )

    def _inspect_element(self, key: object) -> EntityInspection:
        element_id = int(key)
        record = self.element_record(element_id)
        props = record["properties"]
        fields = [
            ("单元编号", str(element_id)), ("本地单元类型", record["type"]),
            ("所属单元集", "、".join(record["sets"]) or "—"),
            ("材料", record["material"] or "—"),
            ("截面", "—" if record["section_index"] is None else f"截面 {record['section_index'] + 1}"),
        ]
        if record["abaqus_type"]:
            fields.insert(2, ("Abaqus 单元类型", str(record["abaqus_type"])))
        if "plane_type" in props:
            fields.append(("平面类型", _plane_label(props["plane_type"])))
        if "thickness" in props:
            fields.append(("厚度", format_number(props["thickness"])))
        connection_rows = []
        references = []
        for local_index, node_id in enumerate(record["node_ids"], 1):
            node = self.nodes[node_id]
            connection_rows.append((
                str(local_index), str(node_id), format_number(node.x), format_number(node.y),
                format_number(getattr(node, "z", 0.0)),
            ))
            references.append(EntityReference("node", node_id))
        property_rows = tuple(
            (_property_label(name), format_number(value))
            for name, value in props.items()
            if name not in {"plane_type", "thickness"}
        )
        tables = [InspectionTable(
            "连接关系", ("局部节点", "全局节点", "X", "Y", "Z"),
            tuple(connection_rows), tuple(references),
        )]
        if property_rows:
            tables.append(InspectionTable("有效属性", ("属性", "数值"), property_rows))
        pages = [InspectionPage("基本信息", tuple(fields)), InspectionPage("连接与属性", tables=tuple(tables))]
        frame_report = self._beam_frame_report_cached(element_id)
        frame_entry = (
            frame_report.for_element(element_id)
            if hasattr(frame_report, "for_element")
            else next(
                (
                    entry
                    for entry in tuple(
                        getattr(frame_report, "entries", ())
                    )
                    if int(entry.element_id) == element_id
                ),
                None,
            )
        )
        if frame_entry is not None:
            frame = frame_entry.frame
            provenance = (
                "direct element"
                if frame_entry.assignment_index is None
                else f"截面分配 {int(frame_entry.assignment_index) + 1}"
            )
            frame_fields = [
                ("单元编号", str(element_id)),
                ("frame source", str(frame.source)),
                ("effective properties 来源", provenance),
                (
                    "assignment element set",
                    str(frame_entry.element_set or "—"),
                ),
                (
                    "effective section type",
                    str(frame_entry.section_type or "—"),
                ),
                ("local x", _vector_text(frame.local_x)),
                ("local y", _vector_text(frame.local_y)),
                ("local z", _vector_text(frame.local_z)),
            ]
            frame_diagnostics = tuple(
                getattr(frame_report, "diagnostics", ())
            )
            if frame_diagnostics:
                frame_fields.append(
                    (
                        "diagnostics",
                        _diagnostic_summary(frame_diagnostics),
                    )
                )
            pages.append(
                InspectionPage("Beam 局部坐标", tuple(frame_fields))
            )
        result_page = self._provider_result_page(
            ElementResultInspectionRequest(element_id)
        )
        if result_page is not None:
            pages.append(result_page)
        return EntityInspection(f"单元 {element_id}", "element", element_id, tuple(pages))

    def _provider_result_page(
        self,
        request: ResultInspectionRequest,
    ) -> InspectionPage | None:
        """Render only the provider's catalog-ordered typed inspection."""

        provider = self.result_provider
        if provider is None:
            return None
        try:
            result = provider.inspect_result(request)
        except ResultQueryValidationError:
            return None
        if type(result) is not ResultInspectionResult:
            raise TypeError(
                "ResultProvider.inspect_result() must return "
                "ResultInspectionResult"
            )
        fields = tuple(
            field_entry
            for field_entry in result.fields
            if result_field_is_visible(field_entry.availability)
        )
        if not fields:
            return None
        return InspectionPage(
            "结果",
            tables=tuple(
                _provider_result_table(field_entry)
                for field_entry in fields
            ),
        )

    def _inspect_node_set(self, key: object) -> EntityInspection:
        name = str(key)
        item = self.model.node_sets[name]
        rows = tuple(
            (str(node_id), format_number(self.nodes[node_id].x), format_number(self.nodes[node_id].y),
             format_number(getattr(self.nodes[node_id], "z", 0.0)))
            for node_id in item.node_ids
        )
        fields = (
            ("名称", name), ("节点数量", str(len(item.node_ids))),
            ("边界条件引用", self._reference_names(self.node_set_boundaries.get(name, ()))),
            ("节点力引用", self._reference_names(self.node_set_loads.get(name, ()))),
        )
        table = InspectionTable("成员节点", ("节点编号", "X", "Y", "Z"), rows,
                                tuple(EntityReference("node", int(row[0])) for row in rows))
        return EntityInspection(f"节点集 {name}", "node_set", name, (InspectionPage("节点集", fields, (table,)),))

    def _inspect_element_set(self, key: object) -> EntityInspection:
        name = str(key)
        item = self.model.element_sets[name]
        records = [self.element_record(element_id) for element_id in item.element_ids]
        rows = tuple((
            str(record["id"]), record["type"], record["material"] or "—",
            "—" if record["section_index"] is None else f"截面 {record['section_index'] + 1}",
        ) for record in records)
        fields = (
            ("名称", name), ("单元数量", str(len(records))),
            ("单元类型统计", _counter_text(record["type"] for record in records)),
            ("使用的材料", "、".join(sorted({record["material"] for record in records if record["material"]})) or "—"),
            ("使用的截面", "、".join(sorted({f"截面 {record['section_index'] + 1}" for record in records if record["section_index"] is not None})) or "—"),
        )
        table = InspectionTable("成员单元", ("单元编号", "类型", "材料", "截面"), rows,
                                tuple(EntityReference("element", int(row[0])) for row in rows))
        return EntityInspection(f"单元集 {name}", "element_set", name, (InspectionPage("单元集", fields, (table,)),))

    def _inspect_surface(self, key: object) -> EntityInspection:
        name = str(key)
        return self._inspect_region("surface", name, self.model.surfaces[name].faces)

    def _inspect_edge(self, key: object) -> EntityInspection:
        name = str(key)
        return self._inspect_region("edge", name, self.model.edges[name].edges)

    def _inspect_region(self, kind: str, name: str, members: tuple[Any, ...]) -> EntityInspection:
        label = "表面" if kind == "surface" else "边"
        rows = tuple((
            str(member.elem_id), str(member.local_index),
            ", ".join(str(value) for value in member.node_ids),
        ) for member in members)
        fields = (
            ("名称", name), ("类型", label), (f"{label}数量", str(len(members))),
            ("涉及单元数量", str(len({member.elem_id for member in members}))),
            ("边界条件引用", self._reference_names(self.region_boundaries.get((kind, name), ()))),
            ("载荷引用", self._reference_names(self.region_loads.get((kind, name), ()))),
        )
        table = InspectionTable(f"成员{label}", ("单元编号", f"局部{label}编号", "节点编号"), rows,
                                tuple(EntityReference("element", int(row[0])) for row in rows))
        return EntityInspection(f"{label} {name}", kind, name, (InspectionPage(label, fields, (table,)),))

    def _inspect_material(self, key: object) -> EntityInspection:
        name = str(key)
        material = self.model.materials[name]
        fields = [("材料名称", name)]
        for property_name in ("E", "nu", "rho", "density"):
            if property_name in material.properties:
                fields.append((_property_label(property_name), format_number(material.properties[property_name])))
        sections = [f"截面 {index + 1}" for index, section in enumerate(self.model.sections) if section.material == name]
        fields.extend((("引用截面", "、".join(sections) or "—"),
                       ("作用单元数量", str(len(self.material_elements.get(name, ()))))))
        return EntityInspection(f"材料 {name}", "material", name, (InspectionPage("材料", tuple(fields)),))

    def _inspect_section(self, key: object) -> EntityInspection:
        index = int(key)
        section = self.model.sections[index]
        ids = self.section_elements.get(index, ())
        fields = [
            ("截面编号", str(index + 1)), ("截面类型", section.section_type),
            ("材料", section.material), ("作用单元集", section.element_set),
            ("作用单元数量", str(len(ids))),
        ]
        plane_types = {self.element_record(value)["properties"].get("plane_type") for value in ids}
        thicknesses = {self.element_record(value)["properties"].get("thickness") for value in ids}
        plane_types.discard(None)
        thicknesses.discard(None)
        if plane_types:
            fields.append(("平面类型", "、".join(sorted(_plane_label(value) for value in plane_types))))
        if thicknesses:
            fields.append(("厚度", "、".join(sorted(format_number(value) for value in thicknesses))))
        for name, value in section.properties.items():
            if _is_physical_property(name, value) and name not in {"plane_type", "thickness"}:
                fields.append((_property_label(name), format_number(value)))
        return EntityInspection(f"截面 {index + 1}", "section", index, (InspectionPage("截面", tuple(fields)),))

    def _inspect_assignment(self, key: object) -> EntityInspection:
        index = int(key)
        assignment = self.region_assignments[index]
        section = next(
            (
                item
                for item in self.section_definitions
                if str(item.name) == str(assignment.section_name)
            ),
            None,
        )
        target = RegionRef(
            "element_set",
            str(assignment.region_name),
        )
        report = self._beam_frame_report_cached(target)
        entries = tuple(getattr(report, "entries", ()))
        element_ids = tuple(getattr(report, "element_ids", ()))
        diagnostics = tuple(getattr(report, "diagnostics", ()))
        orientation = getattr(assignment, "beam_orientation", None)
        not_applicable = (
            orientation is None
            and not entries
            and bool(diagnostics)
            and all(
                str(getattr(item, "code", ""))
                == "beam.orientation.unsupported_target"
                for item in diagnostics
            )
        )
        if not_applicable:
            diagnostics = ()
        reference = (
            None
            if orientation is None
            else getattr(orientation, "local_y_reference", None)
        )
        sources = sorted(
            {
                str(entry.frame.source)
                for entry in entries
            }
        )
        fields = [
            ("分配编号", str(index + 1)),
            ("截面", str(assignment.section_name)),
            ("目标单元集", str(assignment.region_name)),
            (
                "section type",
                str(getattr(section, "section_type", "—")),
            ),
            (
                "orientation source",
                (
                    "not applicable"
                    if not_applicable
                    else (
                        "explicit"
                        if orientation is not None
                        else "automatic"
                    )
                ),
            ),
            (
                "authored reference",
                "—" if reference is None else _vector_text(reference),
            ),
            (
                "effective frame source",
                (
                    "not applicable"
                    if not_applicable
                    else ("、".join(sources) or "—")
                ),
            ),
            (
                "有效元素数量",
                "0" if not_applicable else str(len(entries)),
            ),
            (
                "无效元素数量",
                (
                    "0"
                    if not_applicable
                    else str(max(0, len(element_ids) - len(entries)))
                ),
            ),
            (
                "validity",
                (
                    "not applicable"
                    if not_applicable
                    else (
                        "valid"
                        if bool(getattr(report, "passed", False))
                        else "invalid"
                    )
                ),
            ),
        ]
        properties = dict(getattr(section, "properties", {}))
        if (
            str(getattr(section, "section_type", "")).casefold()
            == "rectangle"
        ):
            fields.extend(
                (
                    ("矩形高度（local y）", format_number(properties.get("height"))),
                    ("矩形宽度（local z）", format_number(properties.get("width"))),
                    (
                        "截面轴映射",
                        "height → local y；width → local z",
                    ),
                )
            )
        if diagnostics:
            fields.append(("diagnostics", _diagnostic_summary(diagnostics)))
        tables = ()
        if diagnostics:
            tables = (
                InspectionTable(
                    "方向诊断",
                    ("代码", "严重性", "说明"),
                    tuple(
                        (
                            str(getattr(item, "code", "")),
                            str(
                                getattr(
                                    getattr(item, "severity", ""),
                                    "value",
                                    getattr(item, "severity", ""),
                                )
                            ),
                            str(getattr(item, "message", item)),
                        )
                        for item in diagnostics
                    ),
                ),
            )
        return EntityInspection(
            f"截面分配 {index + 1}",
            "assignment",
            index,
            (InspectionPage("截面分配", tuple(fields), tables),),
        )

    def _inspect_step(self, key: object) -> EntityInspection:
        step_index = int(key)
        step = self.model.steps[step_index]
        load_count = getattr(step, "summary_load_count", None)
        if load_count is None:
            load_count = (
                len(step.cloads)
                + len(step.surface_loads)
                + len(step.edge_loads)
                + len(step.line_loads)
                + len(step.body_loads)
                + len(step.gravity_loads)
            )
        boundary_count = getattr(
            step,
            "summary_boundary_count",
            len(step.boundaries),
        )
        output_count = getattr(
            step,
            "summary_output_count",
            len(step.outputs),
        )
        pages = [InspectionPage("概况", (
            ("分析步名称", step.name), ("分析类型", _procedure_label(step.procedure)),
            ("边界条件数量", str(boundary_count)), ("载荷数量", str(load_count)),
            ("输出请求数量", str(output_count)),
        ))]
        boundary_rows = tuple((str(index + 1), "位移边界条件", str(item.target),
                               _component_range(item.first_component, item.last_component), format_number(item.value))
                              for index, item in enumerate(step.boundaries))
        if boundary_rows:
            pages.append(InspectionPage("边界条件", tables=(InspectionTable(
                "边界条件", ("序号", "类型", "目标", "分量", "数值"), boundary_rows,
                tuple(EntityReference("boundary", (step_index, index)) for index in range(len(boundary_rows))),
            ),)))
        load_rows: list[tuple[str, ...]] = []
        load_refs: list[EntityReference] = []
        for index, item in enumerate(step.cloads):
            load_rows.append((str(len(load_rows) + 1), "节点力", str(item.target), f"U{item.component}", format_number(item.value)))
            load_refs.append(EntityReference("cload", (step_index, index)))
        for kind, items, target_name in (("surface_load", step.surface_loads, "surface"), ("edge_load", step.edge_loads, "edge")):
            for index, item in enumerate(items):
                target = getattr(item, target_name)
                direction = _load_direction(item)
                load_rows.append((str(len(load_rows) + 1), _load_type_label(item.load_type), target, direction, format_number(item.magnitude)))
                load_refs.append(EntityReference(kind, (step_index, index)))
        for index, item in enumerate(step.line_loads):
            load_rows.append((
                str(len(load_rows) + 1),
                "边力",
                str(item.target),
                (
                    "局部（Beam 已解析局部坐标）"
                    if item.coordinate_system == "local"
                    else "全局坐标"
                ),
                ", ".join(format_number(value) for value in item.vector),
            ))
            load_refs.append(EntityReference("line_load", (step_index, index)))
        for index, item in enumerate(step.body_loads):
            load_rows.append((
                str(len(load_rows) + 1),
                "体力",
                str(item.target),
                "全局坐标",
                ", ".join(format_number(value) for value in item.vector),
            ))
            load_refs.append(EntityReference("body_load", (step_index, index)))
        for index, item in enumerate(step.gravity_loads):
            load_rows.append((
                str(len(load_rows) + 1),
                "重力",
                "整个模型" if item.target is None else str(item.target),
                "全局坐标",
                ", ".join(
                    format_number(value)
                    for value in item.acceleration
                ),
            ))
            load_refs.append(
                EntityReference("gravity_load", (step_index, index))
            )
        if load_rows:
            pages.append(InspectionPage("载荷", tables=(InspectionTable(
                "载荷", ("序号", "类型", "目标", "分量或方向", "数值"), tuple(load_rows), tuple(load_refs),
            ),)))
        output_rows = tuple((str(index + 1), _output_kind(item.kind), _output_target(item.target), ", ".join(item.variables))
                            for index, item in enumerate(step.outputs))
        if output_rows:
            pages.append(InspectionPage("输出请求", tables=(InspectionTable(
                "输出请求", ("序号", "类型", "位置", "变量"), output_rows,
                tuple(EntityReference("output", (step_index, index)) for index in range(len(output_rows))),
            ),)))
        return EntityInspection(f"分析步 {step.name}", "step", step_index, tuple(pages))

    def _inspect_boundary(self, key: object) -> EntityInspection:
        step_index, index = key
        step = self.model.steps[step_index]
        item = step.boundaries[index]
        node_ids = resolve_displacement_node_ids(self.model, item)
        fields = (
            ("所属分析步", step.name), ("类型", "位移边界条件"), ("目标区域", str(item.target)),
            ("作用域类型", displacement_target_kind(item)),
            ("目标节点数量", str(len(node_ids))),
            ("约束分量", _component_range(item.first_component, item.last_component)),
            ("数值", format_number(item.value)),
        )
        rows = tuple((f"U{component}", format_number(item.value))
                     for component in range(item.first_component, item.last_component + 1))
        tables = (InspectionTable("分量与数值", ("分量", "数值"), rows),) if len(rows) > 1 else ()
        return EntityInspection("边界条件", "boundary", key, (InspectionPage("边界条件", fields, tables),))

    def _inspect_cload(self, key: object) -> EntityInspection:
        step_index, index = key
        step = self.model.steps[step_index]
        item = step.cloads[index]
        fields = (("所属分析步", step.name), ("类型", "节点力"), ("目标", str(item.target)),
                  ("目标节点数量", str(len(self.target_node_ids(item.target)))),
                  ("分量", f"U{item.component}"), ("数值", format_number(item.value)))
        return EntityInspection("节点力", "cload", key, (InspectionPage("载荷", fields),))

    def _inspect_surface_load(self, key: object) -> EntityInspection:
        return self._inspect_distributed_load("surface_load", key)

    def _inspect_edge_load(self, key: object) -> EntityInspection:
        return self._inspect_distributed_load("edge_load", key)

    def _inspect_line_load(self, key: object) -> EntityInspection:
        step_index, index = key
        step = self.model.steps[step_index]
        item = step.line_loads[index]
        fields = (
            ("所属分析步", step.name),
            ("类型", "边力（梁单元分布力）"),
            ("目标", str(item.target)),
            ("目标单元数量", str(len(self.target_element_ids(item.target)))),
            (
                "坐标系",
                (
                    "局部（Beam 已解析局部坐标）"
                    if item.coordinate_system == "local"
                    else "全局"
                ),
            ),
            ("载荷向量", ", ".join(format_number(value) for value in item.vector)),
        )
        return EntityInspection(
            "边力", "line_load", key,
            (InspectionPage("载荷", fields),),
        )

    def _inspect_gravity_load(self, key: object) -> EntityInspection:
        step_index, index = key
        step = self.model.steps[step_index]
        item = step.gravity_loads[index]
        target = (
            "整个模型"
            if item.target is None
            else str(item.target)
        )
        fields = (
            ("所属分析步", step.name),
            ("类型", "重力"),
            ("目标", target),
            (
                "加速度向量",
                ", ".join(
                    format_number(value)
                    for value in item.acceleration
                ),
            ),
        )
        return EntityInspection(
            "重力", "gravity_load", key,
            (InspectionPage("载荷", fields),),
        )

    def _inspect_body_load(self, key: object) -> EntityInspection:
        step_index, index = key
        step = self.model.steps[step_index]
        item = step.body_loads[index]
        fields = (
            ("所属分析步", step.name),
            ("类型", "体力"),
            ("目标", str(item.target)),
            (
                "力密度向量",
                ", ".join(format_number(value) for value in item.vector),
            ),
        )
        return EntityInspection(
            "体力",
            "body_load",
            key,
            (InspectionPage("载荷", fields),),
        )

    def _inspect_distributed_load(self, kind: str, key: object) -> EntityInspection:
        step_index, index = key
        step = self.model.steps[step_index]
        is_surface = kind == "surface_load"
        item = step.surface_loads[index] if is_surface else step.edge_loads[index]
        region = item.surface if is_surface else item.edge
        members = self.model.surfaces[region].faces if is_surface else self.model.edges[region].edges
        fields = [("所属分析步", step.name), ("类型", _load_type_label(item.load_type)),
                  ("目标", region), ("面或边数量", str(len(members)))]
        if item.load_type != "pressure" and item.vector:
            fields.append(("方向或向量", ", ".join(format_number(value) for value in item.vector)))
        if item.magnitude is not None:
            fields.append((("压力大小" if item.load_type == "pressure" else "数值"), format_number(item.magnitude)))
        return EntityInspection("面力" if is_surface else "边力", kind, key, (InspectionPage("载荷", tuple(fields)),))

    def _inspect_output(self, key: object) -> EntityInspection:
        step_index, index = key
        step = self.model.steps[step_index]
        item = step.outputs[index]
        fields = (("所属分析步", step.name), ("类型", _output_kind(item.kind)),
                  ("输出位置", _output_target(item.target)), ("变量", ", ".join(item.variables)))
        return EntityInspection("输出请求", "output", key, (InspectionPage("输出请求", fields),))

    def _reference_names(self, references: tuple[EntityReference, ...] | list[EntityReference]) -> str:
        names = []
        for reference in references:
            if reference.kind in {
                "boundary",
                "cload",
                "surface_load",
                "edge_load",
                "line_load",
                "body_load",
                "gravity_load",
            }:
                step_index, index = reference.key
                step = self.model.steps[step_index]
                labels = {
                    "boundary": "边界条件",
                    "cload": "节点力",
                    "surface_load": "面力",
                    "edge_load": "边力",
                    "line_load": "边力",
                    "body_load": "体力",
                    "gravity_load": "重力",
                }
                names.append(f"{step.name} / {labels[reference.kind]} {index + 1}")
        return "、".join(names) or "—"


def _require_result_provider(
    value: ResultProvider | None,
) -> ResultProvider | None:
    if value is not None and type(value) is not ResultProvider:
        raise TypeError("result_provider must be exactly ResultProvider or None")
    return value


def _provider_result_table(
    field_entry: ResultInspectionField,
) -> InspectionTable:
    if type(field_entry) is not ResultInspectionField:
        raise TypeError(
            "provider result fields must be ResultInspectionField values"
        )
    availability = field_entry.availability
    descriptor = availability.descriptor
    state_label = _RESULT_STATE_LABELS[availability.state]
    field_label = _localized_result_field(descriptor)
    if descriptor.unit_label is not None:
        field_label = f"{field_label} [{descriptor.unit_label}]"
    title = f"{field_label}（{state_label}）"
    diagnostic = _diagnostic_summary(availability.diagnostics)
    if availability.state is not FieldState.READY:
        variable = descriptor.field_id.variable.value
        rows = tuple(
            (
                state_label,
                _localized_result_component(component, variable),
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                diagnostic,
            )
            for component in descriptor.columns
        )
        return InspectionTable(
            title,
            _RESULT_COLUMNS,
            rows,
            tuple(None for _row in rows),
        )

    rows: list[tuple[str, ...]] = []
    references: list[EntityReference | None] = []
    variable = descriptor.field_id.variable.value
    for component_result in field_entry.component_results:
        component = _localized_result_component(
            component_result.query.component,
            variable,
        )
        if not component_result.records:
            rows.append(
                (
                    state_label,
                    component,
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    diagnostic,
                )
            )
            references.append(None)
            continue
        for record in component_result.records:
            location = record.location
            rows.append(
                (
                    state_label,
                    component,
                    format_number(record.value),
                    format_number(location.node_id),
                    format_number(location.element_id),
                    format_number(location.integration_point),
                    format_number(location.local_node),
                    (
                        "—"
                        if location.region_key is None
                        else encode_result_region_key(location.region_key)
                    ),
                    _averaged_label(location.averaged),
                    diagnostic,
                )
            )
            references.append(_result_location_reference(location))
    return InspectionTable(
        title,
        _RESULT_COLUMNS,
        tuple(rows),
        tuple(references),
    )


def _localized_result_component(component: str, variable: str) -> str:
    if component == "Magnitude":
        return {
            "U": "位移模",
            "RF": "反力模",
        }.get(variable, "模")
    return _RESULT_COMPONENT_LABELS.get(component, component)


def _localized_result_field(descriptor: Any) -> str:
    field_id = descriptor.field_id
    base = _RESULT_VARIABLE_LABELS.get(
        field_id.variable.value,
        descriptor.label_key,
    )
    if field_id.position is FieldPosition.NODE:
        return base
    position = result_position_label(field_id.position)
    return f"{base}（{position}）"


def _averaged_label(value: bool | None) -> str:
    if value is None:
        return "—"
    return "是" if value else "否"


def _result_location_reference(location: Any) -> EntityReference | None:
    if location.element_id is not None:
        return EntityReference("element", int(location.element_id))
    if location.node_id is not None:
        return EntityReference("node", int(location.node_id))
    return None


def format_number(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, Real):
        number = float(value)
        if abs(number) < 5.0e-13:
            return "0"
        return f"{number:.6g}"
    return str(value)


def _vector_text(values: object) -> str:
    return ", ".join(
        format_number(value)
        for value in tuple(values)
    )


def _diagnostic_summary(diagnostics: object) -> str:
    values = tuple(diagnostics)
    return "；".join(
        (
            f"[{getattr(item, 'code', 'diagnostic')}] "
            f"{getattr(item, 'message', str(item))}"
        )
        for item in values
    ) or "—"


def _counter_text(values: Any) -> str:
    return "、".join(f"{name}: {count}" for name, count in Counter(values).items()) or "—"


def _component_range(first: int, last: int) -> str:
    return f"U{first}" if first == last else "、".join(f"U{value}" for value in range(first, last + 1))


def _is_physical_property(name: str, value: object) -> bool:
    if name.startswith("_") or name in {"material", "element_set", "abaqus_type"}:
        return False
    return isinstance(value, (Real, str)) and name in {
        "E", "nu", "rho", "density", "thickness", "plane_type",
        "area", "A", "I", "Iyy", "Izz", "J", "section_type",
        "height", "width", "radius", "inner_radius", "outer_radius",
    }


def _property_label(name: str) -> str:
    return {
        "E": "弹性模量 E", "nu": "泊松比 ν", "rho": "密度 ρ",
        "density": "密度 ρ", "area": "截面积", "A": "截面积",
        "I": "惯性矩", "Iyy": "惯性矩 Iyy", "Izz": "惯性矩 Izz",
        "J": "扭转常数 J", "section_type": "截面类型",
        "height": "矩形高度（局部 y）", "width": "矩形宽度（局部 z）",
        "radius": "半径", "inner_radius": "内半径", "outer_radius": "外半径",
    }.get(name, name)


def _plane_label(value: object) -> str:
    return "平面应变" if str(value).lower().startswith("strain") else "平面应力"


def _procedure_label(value: str) -> str:
    return "线性静力" if str(value).lower() == "static" else str(value)


def _output_kind(value: str) -> str:
    return {"field": "场输出", "history": "历史输出"}.get(value, value)


def _output_target(value: str) -> str:
    return {"node": "节点", "element": "单元"}.get(value, value)


def _load_type_label(value: str) -> str:
    return "压力" if str(value).lower() == "pressure" else "面力或边力"


def _load_direction(item: Any) -> str:
    if str(item.load_type).lower() == "pressure":
        return "法向"
    return ", ".join(format_number(value) for value in item.vector) or "—"
