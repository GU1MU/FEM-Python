"""由当前真实模型生成的中文模型树。"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QMenu, QTreeWidget, QTreeWidgetItem

from ..icons import icon

ROLE_KIND = int(Qt.ItemDataRole.UserRole)
ROLE_KEY = ROLE_KIND + 1

_TREE_ICONS = {
    "model": "model",
    "mesh": "mesh",
    "node_set": "node_set",
    "element_set": "element_set",
    "surface": "surface",
    "edge": "surface",
    "material": "material",
    "section": "section",
    "assignment": "section",
    "step": "step",
    "boundary": "boundary",
    "cload": "load",
    "surface_load": "load",
    "edge_load": "load",
    "line_load": "load",
    "gravity_load": "load",
    "output": "output",
}

_CATEGORY_ICONS = {
    "节点集": "node_set",
    "单元集": "element_set",
    "表面": "surface",
    "材料": "material",
    "截面": "section",
    "分析": "step",
    "边界条件": "boundary",
    "载荷": "load",
    "输出请求": "output",
}

_EDITABLE_KINDS = {
    "material",
    "assignment",
    "step",
    "boundary",
    "cload",
    "edge_load",
    "surface_load",
    "line_load",
    "gravity_load",
    "output",
}


def _section_label(section: Any, element: Any | None = None) -> str:
    """Translate backend section identifiers into concise CAE terminology."""
    section_type = str(section.section_type).strip()
    normalized = section_type.casefold()
    properties = dict(getattr(section, "properties", {}))
    properties.update(getattr(element, "props", {}))
    if normalized == "solid":
        plane_type = str(properties.get("plane_type", "")).casefold()
        if plane_type.startswith("stress"):
            return "平面应力"
        if plane_type.startswith("strain"):
            return "平面应变"
        element_type = str(getattr(element, "type", "")).casefold()
        if element_type.startswith(("tri", "quad")):
            return "二维实体"
        return "三维实体"
    return {
        "beam": "梁截面",
        "truss": "杆截面",
        "shell": "壳截面",
    }.get(normalized, f"INP 截面（{section_type}）")


class ModelTree(QTreeWidget):
    """Model navigation tree with explicit inspect and edit actions."""

    highlightRequested = Signal(str, object)
    informationRequested = Signal(str, object)
    editRequested = Signal(str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("modelTree")
        self.setHeaderHidden(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.itemClicked.connect(self._on_clicked)
        self.itemDoubleClicked.connect(self._on_double_clicked)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.clear_model()

    def clear_model(self) -> None:
        self.clear()
        root = self._item("模型", "empty", None)
        root.addChild(self._item("未打开模型", "empty", None))
        self.addTopLevelItem(root)
        root.setExpanded(True)

    def set_model(
        self,
        model: Any,
        result: Any | None = None,
        *,
        feature_rows: tuple[str, ...] = (),
        part_name: str | None = None,
        section_definitions: tuple[Any, ...] = (),
        region_assignments: tuple[Any, ...] = (),
        scope_names: Collection[str] | None = None,
    ) -> None:
        self.clear()
        visible_scope_names = (
            None
            if scope_names is None
            else frozenset(str(name) for name in scope_names)
        )

        def visible_items(values):
            return tuple(
                (name, value)
                for name, value in values.items()
                if (
                    visible_scope_names is None
                    or str(name) in visible_scope_names
                )
            )

        root = self._item(str(model.name or "模型"), "model", None)
        part = None
        if part_name is not None:
            part = self._item(str(part_name), "part", None)
            for row in feature_rows:
                part.addChild(self._item(str(row), "feature", None))
            root.addChild(part)
        mesh = self._item("网格", "mesh", None)
        visible_node_sets = visible_items(model.node_sets)
        node_sets = self._category(mesh, "节点集", len(visible_node_sets))
        for name, node_set in visible_node_sets:
            node_sets.addChild(self._item(f"{name}  ({len(node_set.node_ids)})", "node_set", name))
        visible_element_sets = visible_items(model.element_sets)
        element_sets = self._category(mesh, "单元集", len(visible_element_sets))
        for name, element_set in visible_element_sets:
            element_sets.addChild(self._item(f"{name}  ({len(element_set.element_ids)})", "element_set", name))
        visible_surfaces = visible_items(model.surfaces)
        visible_edges = visible_items(model.edges)
        surface_count = len(visible_surfaces) + len(visible_edges)
        surfaces = self._category(mesh, "表面", surface_count)
        for name, surface in visible_surfaces:
            surfaces.addChild(self._item(f"{name}  ({len(surface.faces)})", "surface", name))
        for name, edge in visible_edges:
            surfaces.addChild(self._item(f"{name}  ({len(edge.edges)})", "edge", name))
        root.addChild(mesh)

        materials = self._category(root, "材料", len(model.materials))
        for name in model.materials:
            materials.addChild(self._item(name, "material", name))
        sections = self._category(root, "截面", len(model.sections))
        elements_by_id = {
            int(element.id): element
            for element in model.mesh.elements
            if getattr(element, "id", None) is not None
        }
        for index, section in enumerate(model.sections):
            element_set = model.element_sets.get(
                getattr(section, "element_set", "")
            )
            representative = (
                elements_by_id.get(element_set.element_ids[0])
                if element_set is not None and element_set.element_ids
                else None
            )
            sections.addChild(
                self._item(
                    f"截面 {index + 1}（"
                    f"{_section_label(section, representative)}）",
                    "section",
                    index,
                )
            )
        if region_assignments:
            assignments = self._category(
                root,
                "截面分配",
                len(region_assignments),
            )
            known_sections = {
                str(section.name)
                for section in section_definitions
            }
            for index, assignment in enumerate(region_assignments):
                section_name = str(assignment.section_name)
                region_name = str(assignment.region_name)
                label = f"{section_name} → {region_name}"
                if known_sections and section_name not in known_sections:
                    label += "（截面缺失）"
                item = self._item(
                    label,
                    "assignment",
                    index,
                )
                orientation = (
                    "explicit"
                    if getattr(assignment, "beam_orientation", None)
                    is not None
                    else "automatic"
                )
                item.addChild(
                    self._item(
                        f"orientation: {orientation}",
                        "detail",
                        None,
                    )
                )
                assignments.addChild(item)
        steps = self._category(root, "分析", len(model.steps))
        first_step_item = None
        for index, step in enumerate(model.steps):
            step_item = self._item(step.name, "step", index)
            if first_step_item is None and step.name.lower() != "initial":
                first_step_item = step_item
            bc_root = self._category(step_item, "边界条件", len(step.boundaries))
            for bc_index, boundary in enumerate(step.boundaries):
                bc_root.addChild(self._item(f"位移约束 {bc_index + 1}", "boundary", (index, bc_index)))
            load_count = (
                len(step.cloads)
                + len(step.surface_loads)
                + len(step.edge_loads)
                + len(step.line_loads)
                + len(getattr(step, "gravity_loads", ()))
            )
            load_root = self._category(step_item, "载荷", load_count)
            for load_index, _load in enumerate(step.cloads):
                load_root.addChild(self._item(f"节点载荷 {load_index + 1}", "cload", (index, load_index)))
            for load_index, _load in enumerate(step.surface_loads):
                load_root.addChild(self._item(f"面载荷 {load_index + 1}", "surface_load", (index, load_index)))
            for load_index, _load in enumerate(step.edge_loads):
                load_root.addChild(self._item(f"边载荷 {load_index + 1}", "edge_load", (index, load_index)))
            for load_index, _load in enumerate(step.line_loads):
                load_root.addChild(self._item(
                    f"梁均布载荷 {load_index + 1}",
                    "line_load",
                    (index, load_index),
                ))
            for load_index, _load in enumerate(
                getattr(step, "gravity_loads", ())
            ):
                load_root.addChild(self._item(
                    f"重力 {load_index + 1}",
                    "gravity_load",
                    (index, load_index),
                ))
            output_root = self._category(step_item, "输出请求", len(step.outputs))
            for output_index, output in enumerate(step.outputs):
                kind_label = {"field": "字段输出", "history": "历史输出"}.get(output.kind, "输出")
                target_label = {"node": "节点", "element": "单元"}.get(output.target, output.target)
                output_root.addChild(self._item(f"{kind_label}：{target_label}", "output", (index, output_index)))
            steps.addChild(step_item)
        self.addTopLevelItem(root)
        root.setExpanded(True)
        mesh.setExpanded(part is None)
        steps.setExpanded(part is None)
        if part is not None:
            part.setExpanded(True)
        if first_step_item is None and steps.childCount():
            first_step_item = steps.child(0)
        if first_step_item is not None and part is None:
            first_step_item.setExpanded(True)

    def set_geometry_preview(
        self,
        name: str,
        feature_rows: tuple[str, ...],
        *,
        part_name: str = "Part-1",
    ) -> None:
        """Show a deliberately shallow Model → Part → Feature history."""
        self.clear()
        root = self._item(str(name), "model", None)
        part = self._item(str(part_name), "part", None)
        for row in feature_rows:
            part.addChild(self._item(str(row), "feature", None))
        root.addChild(part)
        self.addTopLevelItem(root)
        root.setExpanded(True)
        part.setExpanded(True)

    def select_entity(self, kind: str, key: int) -> None:
        if kind in {"node", "element"}:
            mesh = self._find_kind("mesh")
            if mesh is not None:
                self.setCurrentItem(mesh)
                self.scrollToItem(mesh)
            return
        iterator = self.invisibleRootItem()
        stack = [iterator.child(index) for index in range(iterator.childCount())]
        while stack:
            item = stack.pop()
            if item.data(0, ROLE_KIND) == kind and item.data(0, ROLE_KEY) == key:
                self.setCurrentItem(item)
                self.scrollToItem(item)
                return
            stack.extend(item.child(index) for index in range(item.childCount()))

    def _find_kind(self, kind: str) -> QTreeWidgetItem | None:
        root = self.invisibleRootItem()
        stack = [root.child(index) for index in range(root.childCount())]
        while stack:
            item = stack.pop()
            if item.data(0, ROLE_KIND) == kind:
                return item
            stack.extend(item.child(index) for index in range(item.childCount()))
        return None

    def _category(self, parent: QTreeWidgetItem, text: str, count: int) -> QTreeWidgetItem:
        item = self._item(f"{text} ({count})", "category", None)
        icon_name = _CATEGORY_ICONS.get(text)
        if icon_name is not None:
            item.setIcon(0, icon(icon_name))
        parent.addChild(item)
        return item

    def _item(self, text: str, kind: str, key: object) -> QTreeWidgetItem:
        item = QTreeWidgetItem([text])
        item.setData(0, ROLE_KIND, kind)
        item.setData(0, ROLE_KEY, key)
        icon_name = _TREE_ICONS.get(kind)
        if icon_name is not None:
            item.setIcon(0, icon(icon_name))
        return item

    def _entry(self, item: QTreeWidgetItem | None) -> tuple[str, object] | None:
        if item is None:
            return None
        kind = str(item.data(0, ROLE_KIND))
        key = item.data(0, ROLE_KEY)
        if kind in {"empty", "category", "detail"}:
            return None
        return kind, key

    def _on_clicked(self, item: QTreeWidgetItem) -> None:
        entry = self._entry(item)
        if entry is not None:
            self.highlightRequested.emit(*entry)

    def _on_double_clicked(self, item: QTreeWidgetItem) -> None:
        entry = self._entry(item)
        if entry is not None:
            signal = (
                self.editRequested
                if entry[0] in _EDITABLE_KINDS
                else self.informationRequested
            )
            signal.emit(*entry)

    def _show_context_menu(self, position: QPoint) -> None:
        item = self.itemAt(position)
        entry = self._entry(item)
        if entry is None:
            return
        self.setCurrentItem(item)
        menu = QMenu(self)
        highlight = menu.addAction("高亮")
        edit = (
            menu.addAction("编辑")
            if entry[0] in _EDITABLE_KINDS
            else None
        )
        information = menu.addAction("查看信息")
        chosen = menu.exec(self.viewport().mapToGlobal(position))
        if chosen is highlight:
            self.highlightRequested.emit(*entry)
        elif edit is not None and chosen is edit:
            self.editRequested.emit(*entry)
        elif chosen is information:
            self.informationRequested.emit(*entry)
