"""由当前真实模型生成的中文模型树。"""

from __future__ import annotations

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
    "step": "step",
    "boundary": "boundary",
    "cload": "load",
    "surface_load": "load",
    "edge_load": "load",
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


class ModelTree(QTreeWidget):
    """只读模型导航树。"""

    highlightRequested = Signal(str, object)
    informationRequested = Signal(str, object)

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

    def set_model(self, model: Any, result: Any | None = None) -> None:
        self.clear()
        root = self._item(str(model.name or "模型"), "model", None)
        mesh = self._item(
            f"网格（{len(model.mesh.nodes)} 节点，{len(model.mesh.elements)} 单元）",
            "mesh",
            None,
        )
        node_sets = self._category(mesh, "节点集", len(model.node_sets))
        for name, node_set in model.node_sets.items():
            node_sets.addChild(self._item(f"{name}  ({len(node_set.node_ids)})", "node_set", name))
        element_sets = self._category(mesh, "单元集", len(model.element_sets))
        for name, element_set in model.element_sets.items():
            element_sets.addChild(self._item(f"{name}  ({len(element_set.element_ids)})", "element_set", name))
        surface_count = len(model.surfaces) + len(model.edges)
        surfaces = self._category(mesh, "表面", surface_count)
        for name, surface in model.surfaces.items():
            surfaces.addChild(self._item(f"{name}  ({len(surface.faces)})", "surface", name))
        for name, edge in model.edges.items():
            surfaces.addChild(self._item(f"{name}  ({len(edge.edges)})", "edge", name))
        root.addChild(mesh)

        materials = self._category(root, "材料", len(model.materials))
        for name in model.materials:
            materials.addChild(self._item(name, "material", name))
        sections = self._category(root, "截面", len(model.sections))
        for index, section in enumerate(model.sections):
            sections.addChild(self._item(f"截面 {index + 1}  ({section.section_type})", "section", index))
        steps = self._category(root, "分析", len(model.steps))
        first_step_item = None
        for index, step in enumerate(model.steps):
            step_item = self._item(step.name, "step", index)
            if first_step_item is None and step.name.lower() != "initial":
                first_step_item = step_item
            bc_root = self._category(step_item, "边界条件", len(step.boundaries))
            for bc_index, boundary in enumerate(step.boundaries):
                bc_root.addChild(self._item(f"位移约束 {bc_index + 1}", "boundary", (index, bc_index)))
            load_count = len(step.cloads) + len(step.surface_loads) + len(step.edge_loads)
            load_root = self._category(step_item, "载荷", load_count)
            for load_index, _load in enumerate(step.cloads):
                load_root.addChild(self._item(f"节点载荷 {load_index + 1}", "cload", (index, load_index)))
            for load_index, _load in enumerate(step.surface_loads):
                load_root.addChild(self._item(f"面载荷 {load_index + 1}", "surface_load", (index, load_index)))
            for load_index, _load in enumerate(step.edge_loads):
                load_root.addChild(self._item(f"边载荷 {load_index + 1}", "edge_load", (index, load_index)))
            output_root = self._category(step_item, "输出请求", len(step.outputs))
            for output_index, output in enumerate(step.outputs):
                kind_label = {"field": "字段输出", "history": "历史输出"}.get(output.kind, "输出")
                target_label = {"node": "节点", "element": "单元"}.get(output.target, output.target)
                output_root.addChild(self._item(f"{kind_label}：{target_label}", "output", (index, output_index)))
            steps.addChild(step_item)
        self.addTopLevelItem(root)
        root.setExpanded(True)
        mesh.setExpanded(True)
        steps.setExpanded(True)
        if first_step_item is None and steps.childCount():
            first_step_item = steps.child(0)
        if first_step_item is not None:
            first_step_item.setExpanded(True)

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
        if kind in {"empty", "category"}:
            return None
        return kind, key

    def _on_clicked(self, item: QTreeWidgetItem) -> None:
        entry = self._entry(item)
        if entry is not None:
            self.highlightRequested.emit(*entry)

    def _on_double_clicked(self, item: QTreeWidgetItem) -> None:
        entry = self._entry(item)
        if entry is not None:
            self.informationRequested.emit(*entry)

    def _show_context_menu(self, position: QPoint) -> None:
        item = self.itemAt(position)
        entry = self._entry(item)
        if entry is None:
            return
        self.setCurrentItem(item)
        menu = QMenu(self)
        highlight = menu.addAction("高亮")
        information = menu.addAction("查看信息")
        chosen = menu.exec(self.viewport().mapToGlobal(position))
        if chosen is highlight:
            self.highlightRequested.emit(*entry)
        elif chosen is information:
            self.informationRequested.emit(*entry)
