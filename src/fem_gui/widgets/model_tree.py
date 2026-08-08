"""由当前真实模型生成的中文模型树。"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QAbstractItemView, QMenu, QTreeWidget, QTreeWidgetItem

from ..icons import icon

ROLE_KIND = int(Qt.ItemDataRole.UserRole)
ROLE_KEY = ROLE_KIND + 1

_ACTIVE_PART_BACKGROUND = QColor("#d9ecff")

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
    "part",
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

_DELETABLE_KINDS = {
    "part",
    "boundary",
    "cload",
    "edge_load",
    "surface_load",
    "line_load",
    "body_load",
    "gravity_load",
}

_NATIVE_FEATURE_NAMES = {
    "Wire": "线体",
    "Sketch": "草图",
    "Move": "移动",
    "Rotate": "旋转",
    "Extrude": "拉伸",
    "Sweep": "扫掠",
    "Fuse": "合并",
    "Cut": "切除",
    "Partition": "分割",
    "Base": "基础体",
}


def _native_feature_label(value: object) -> str:
    """Translate canonical native-feature names for display only."""

    text = str(value)
    namespace, separator, leaf = text.rpartition("/")
    prefix = f"{namespace}/" if separator else ""
    if leaf.startswith(("拉伸合并-", "拉伸切除-")):
        return leaf
    for source, translated in _NATIVE_FEATURE_NAMES.items():
        suffix = leaf.removeprefix(source)
        if leaf == source or (
            leaf.startswith(f"{source}-")
            and suffix.removeprefix("-").isdigit()
        ):
            return f"{prefix}{translated}{suffix}"
    return text


def _native_part_label(native_part: Any) -> str:
    """Return a user-facing Part label without exposing internal IDs."""

    state = "（已抑制）" if native_part.suppressed else ""
    return f"{native_part.name}{state}"


def _style_native_part_item(
    item: QTreeWidgetItem,
    native_part: Any,
    *,
    active: bool,
) -> None:
    """Show current-Part state through color instead of label suffixes."""

    if active:
        item.setBackground(0, QBrush(_ACTIVE_PART_BACKGROUND))
    elif native_part.suppressed:
        item.setForeground(0, QBrush(Qt.GlobalColor.gray))


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
    deleteRequested = Signal(str, object)
    renameRequested = Signal(str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("modelTree")
        self.setHeaderHidden(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setStyleSheet(
            "QTreeWidget::item:selected {"
            " background-color: #d9ecff;"
            " color: #202020;"
            "}"
        )
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.itemClicked.connect(self._on_clicked)
        self.itemDoubleClicked.connect(self._on_double_clicked)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.clear_model()

    def clear_model(self) -> None:
        self._renamable_kinds: frozenset[str] = frozenset()
        self._non_highlightable_kinds: frozenset[str] = frozenset()
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
        model_name: str | None = None,
        section_definitions: tuple[Any, ...] = (),
        region_assignments: tuple[Any, ...] = (),
        scope_names: Collection[str] | None = None,
        native_parts: tuple[Any, ...] = (),
        active_part_id: str | None = None,
    ) -> None:
        self._renamable_kinds = (
            frozenset({"model", "part"})
            if part_name is not None or native_parts
            else frozenset()
        )
        self._non_highlightable_kinds = (
            frozenset({"model", "feature"})
            if native_parts
            else (
                frozenset({"model", "part", "feature"})
                if part_name is not None
                else frozenset()
            )
        )
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

        root = self._item(
            str(model_name or model.name or "模型"),
            "model",
            None,
        )
        part = None
        if native_parts:
            for native_part in native_parts:
                native_item = self._item(
                    _native_part_label(native_part),
                    "part",
                    native_part.id,
                )
                _style_native_part_item(
                    native_item,
                    native_part,
                    active=native_part.id == active_part_id,
                )
                for record in native_part.feature_history:
                    feature_item = self._item(
                        _native_feature_label(record.name),
                        "feature",
                        str(record.name),
                    )
                    summary = record.payload.get("summary")
                    if summary:
                        feature_item.setToolTip(0, str(summary))
                    native_item.addChild(feature_item)
                native_item.addChild(
                    self._item(
                        "网格设置",
                        "part_mesh_settings",
                        native_part.id,
                    )
                )
                root.addChild(native_item)
                native_item.setExpanded(
                    native_part.id == active_part_id
                )
            part = next(
                (
                    root.child(index)
                    for index in range(root.childCount())
                    if root.child(index).data(0, ROLE_KEY)
                    == active_part_id
                ),
                None,
            )
        elif part_name is not None:
            part = self._item(str(part_name), "part", None)
            for row in feature_rows:
                part.addChild(
                    self._item(
                        _native_feature_label(row),
                        "feature",
                        str(row),
                    )
                )
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
                identity = getattr(boundary, "name", None)
                bc_root.addChild(
                    self._item(
                        identity or f"位移约束 {bc_index + 1}",
                        "boundary",
                        (
                            (step.name, identity)
                            if identity is not None
                            else (index, bc_index)
                        ),
                    )
                )
            load_count = (
                len(step.cloads)
                + len(step.surface_loads)
                + len(step.edge_loads)
                + len(step.line_loads)
                + len(getattr(step, "body_loads", ()))
                + len(getattr(step, "gravity_loads", ()))
            )
            load_root = self._category(step_item, "载荷", load_count)
            for load_index, load in enumerate(step.cloads):
                identity = getattr(load, "name", None)
                load_root.addChild(self._item(
                    identity or f"节点力 {load_index + 1}",
                    "cload",
                    (step.name, identity) if identity is not None else (index, load_index),
                ))
            for load_index, load in enumerate(step.surface_loads):
                identity = getattr(load, "name", None)
                load_root.addChild(self._item(
                    identity or f"面力 {load_index + 1}",
                    "surface_load",
                    (step.name, identity) if identity is not None else (index, load_index),
                ))
            for load_index, load in enumerate(step.edge_loads):
                identity = getattr(load, "name", None)
                load_root.addChild(self._item(
                    identity or f"边力 {load_index + 1}",
                    "edge_load",
                    (step.name, identity) if identity is not None else (index, load_index),
                ))
            for load_index, load in enumerate(step.line_loads):
                identity = getattr(load, "name", None)
                load_root.addChild(self._item(
                    identity or f"边力 {load_index + 1}",
                    "line_load",
                    (
                        (step.name, identity)
                        if identity is not None
                        else (index, load_index)
                    ),
                ))
            for load_index, load in enumerate(
                getattr(step, "body_loads", ())
            ):
                identity = getattr(load, "name", None)
                load_root.addChild(self._item(
                    identity or f"体力 {load_index + 1}",
                    "body_load",
                    (
                        (step.name, identity)
                        if identity is not None
                        else (index, load_index)
                    ),
                ))
            for load_index, load in enumerate(
                getattr(step, "gravity_loads", ())
            ):
                identity = getattr(load, "name", None)
                load_root.addChild(self._item(
                    identity or f"重力 {load_index + 1}",
                    "gravity_load",
                    (
                        (step.name, identity)
                        if identity is not None
                        else (index, load_index)
                    ),
                ))
            output_count = sum(
                max(1, len(output.variables))
                for output in step.outputs
            )
            output_root = self._category(step_item, "输出请求", output_count)
            for output_index, output in enumerate(step.outputs):
                identity = getattr(output, "name", None)
                labels = tuple(output.variables) or ("输出请求",)
                for variable in labels:
                    output_root.addChild(self._item(
                        variable,
                        "output",
                        (
                            (step.name, identity)
                            if identity is not None
                            else (index, output_index)
                        ),
                    ))
            steps.addChild(step_item)
        self.addTopLevelItem(root)
        root.setExpanded(True)
        mesh.setExpanded(part is None)
        steps.setExpanded(part is None)
        if part is not None:
            part.setExpanded(True)
            self.setCurrentItem(part)
        if first_step_item is None and steps.childCount():
            first_step_item = steps.child(0)
        if first_step_item is not None and part is None:
            first_step_item.setExpanded(True)

    def set_geometry_preview(
        self,
        name: str,
        feature_rows: tuple[str, ...],
        *,
        part_name: str = "部件-1",
        bodies: tuple[tuple[str, str, tuple[str, ...]], ...] = (),
        parts: tuple[Any, ...] | None = None,
        active_part_id: str | None = None,
    ) -> None:
        """显示模型以及稳定的原生部件层级。"""
        self._renamable_kinds = frozenset({"model", "part"})
        self._non_highlightable_kinds = (
            frozenset({"model", "feature"})
            if parts is not None
            else frozenset({"model", "part", "feature"})
        )
        self.clear()
        root = self._item(str(name), "model", None)
        if parts is not None:
            active_item = None
            for native_part in parts:
                part = self._item(
                    _native_part_label(native_part),
                    "part",
                    native_part.id,
                )
                is_active = native_part.id == active_part_id
                _style_native_part_item(
                    part,
                    native_part,
                    active=is_active,
                )
                for row in native_part.feature_history:
                    feature_item = self._item(
                        _native_feature_label(row.name),
                        "feature",
                        str(row.name),
                    )
                    summary = row.payload.get("summary")
                    if summary:
                        feature_item.setToolTip(0, str(summary))
                    part.addChild(feature_item)
                provenance = native_part.provenance
                if provenance is not None:
                    operation = (
                        "布尔合并"
                        if provenance.operation == "fuse"
                        else "布尔切除"
                    )
                    part.addChild(
                        self._item(
                            f"{operation} [{provenance.feature_id}]",
                            "feature",
                            provenance.feature_id,
                        )
                    )
                    part.addChild(
                        self._item(
                            "源部件："
                            f"{provenance.target_part_id}、"
                            f"{provenance.tool_part_id}",
                            "detail",
                            None,
                        )
                    )
                part.addChild(
                    self._item("网格设置", "part_mesh_settings", native_part.id)
                )
                root.addChild(part)
                part.setExpanded(is_active)
                if is_active:
                    active_item = part
            self.addTopLevelItem(root)
            root.setExpanded(True)
            if active_item is not None:
                self.setCurrentItem(active_item)
            return
        part = self._item(str(part_name), "part", None)
        if bodies:
            for body_id, body_name, rows in bodies:
                body = self._item(
                    f"{body_name} [{body_id}]",
                    "geometry_body",
                    f"body:{body_id}",
                )
                for row in rows:
                    body.addChild(
                        self._item(
                            _native_feature_label(row),
                            "feature",
                            str(row),
                        )
                    )
                part.addChild(body)
                body.setExpanded(True)
        else:
            for row in feature_rows:
                part.addChild(
                    self._item(
                        _native_feature_label(row),
                        "feature",
                        str(row),
                    )
                )
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
        if (
            entry is not None
            and entry[0] not in self._non_highlightable_kinds
        ):
            self.highlightRequested.emit(*entry)

    def _on_double_clicked(self, item: QTreeWidgetItem) -> None:
        entry = self._entry(item)
        if entry is not None:
            editable = (
                entry[0] in _EDITABLE_KINDS
                and not (
                    entry[0] == "part"
                    and type(entry[1]) is not str
                )
            )
            signal = (
                self.editRequested
                if editable
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
        highlight = (
            menu.addAction("高亮")
            if entry[0] not in self._non_highlightable_kinds
            else None
        )
        rename = (
            menu.addAction("重命名")
            if entry[0] in self._renamable_kinds
            else None
        )
        edit = (
            menu.addAction("编辑")
            if (
                entry[0] in _EDITABLE_KINDS
                and not (
                    entry[0] == "part"
                    and type(entry[1]) is not str
                )
            )
            else None
        )
        delete = (
            menu.addAction("删除")
            if (
                entry[0] in _DELETABLE_KINDS
                and not (
                    entry[0] == "part"
                    and type(entry[1]) is not str
                )
            )
            else None
        )
        information = menu.addAction("查看信息")
        chosen = menu.exec(self.viewport().mapToGlobal(position))
        if highlight is not None and chosen is highlight:
            self.highlightRequested.emit(*entry)
        elif rename is not None and chosen is rename:
            self.renameRequested.emit(*entry)
        elif edit is not None and chosen is edit:
            self.editRequested.emit(*entry)
        elif delete is not None and chosen is delete:
            self.deleteRequested.emit(*entry)
        elif chosen is information:
            self.informationRequested.emit(*entry)
