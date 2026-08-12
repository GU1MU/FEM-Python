"""由当前真实模型生成的中文模型树。"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import QAbstractItemView, QMenu, QTreeWidget, QTreeWidgetItem

from fem.boundary.step import effective_step_boundaries

from ..icons import icon

ROLE_KIND = int(Qt.ItemDataRole.UserRole)
ROLE_KEY = ROLE_KIND + 1
ROLE_INHERITED = ROLE_KEY + 1
ROLE_DOCUMENT_ID = ROLE_INHERITED + 1


@dataclass(frozen=True, slots=True)
class _TreeViewState:
    expanded_paths: frozenset[tuple[tuple[str, str], ...]]
    current_path: tuple[tuple[str, str], ...] | None
    vertical_scroll: int


_ICON_ROOT = Path(__file__).resolve().parents[1] / "resources" / "icons"
_SCROLL_UP_ARROW = (_ICON_ROOT / "agent_chat_scroll_up.svg").as_posix()
_SCROLL_DOWN_ARROW = (_ICON_ROOT / "agent_chat_scroll_down.svg").as_posix()

_MODEL_TREE_STYLESHEET = f"""
QTreeWidget#modelTree::item:hover {{
    background-color: #d9ecff;
    color: #202020;
}}
QTreeWidget#modelTree QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 12px 0 12px 0;
}}
QTreeWidget#modelTree QScrollBar::handle:vertical {{
    background: rgba(76, 88, 98, 92);
    min-height: 34px;
    border-radius: 4px;
    margin: 1px 2px;
}}
QTreeWidget#modelTree QScrollBar::handle:vertical:hover {{
    background: rgba(76, 88, 98, 138);
}}
QTreeWidget#modelTree QScrollBar::add-line:vertical,
QTreeWidget#modelTree QScrollBar::sub-line:vertical {{
    background: transparent;
    border: none;
    height: 12px;
    subcontrol-origin: margin;
}}
QTreeWidget#modelTree QScrollBar::sub-line:vertical {{
    subcontrol-position: top;
}}
QTreeWidget#modelTree QScrollBar::add-line:vertical {{
    subcontrol-position: bottom;
}}
QTreeWidget#modelTree QScrollBar::up-arrow:vertical {{
    image: url("{_SCROLL_UP_ARROW}");
    width: 8px;
    height: 6px;
}}
QTreeWidget#modelTree QScrollBar::down-arrow:vertical {{
    image: url("{_SCROLL_DOWN_ARROW}");
    width: 8px;
    height: 6px;
}}
QTreeWidget#modelTree QScrollBar::add-page:vertical,
QTreeWidget#modelTree QScrollBar::sub-page:vertical {{
    background: transparent;
}}
"""

_TREE_ICONS = {
    "model": "model",
    "mesh": "mesh",
    "node_set": "node_set",
    "element_set": "element_set",
    "surface": "surface",
    "edge": "surface",
    "material": "material",
    "section": "section",
    "assignment": "section_assign",
    "step": "step",
    "boundary": "boundary",
    "inherited_boundary": "boundary",
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
    "截面分配": "section_assign",
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
    "PathSweep": "路径扫掠",
    "Fuse": "合并",
    "Cut": "切除",
    "Partition": "分割",
    "Base": "基础体",
}

_NATIVE_FEATURE_KIND_NAMES = {
    source.casefold(): translated
    for source, translated in _NATIVE_FEATURE_NAMES.items()
}
_NATIVE_FEATURE_KIND_NAMES.update({
    "face_sketch_boolean_fuse": "拉伸合并",
    "face_sketch_boolean_cut": "拉伸切除",
})


def native_feature_label(value: object) -> str:
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


def native_feature_kind_label(value: object) -> str:
    """Translate a canonical native-feature kind for display only."""

    text = str(value)
    return _NATIVE_FEATURE_KIND_NAMES.get(text.casefold(), text)


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
    """Show current-Part state without looking like a hovered tree row."""

    if active:
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
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

    # Keep the two-argument overload for callers from Phase 0/1 while making
    # the three-argument document-routed overload the default.  Emission
    # helpers below publish both overloads; Qt chooses the compatible overload
    # for each connected slot, so existing integrations remain source-safe.
    highlightRequested = Signal(
        (int, str, object),
        (str, object),
    )
    informationRequested = Signal(
        (int, str, object),
        (str, object),
    )
    editRequested = Signal(
        (int, str, object),
        (str, object),
    )
    deleteRequested = Signal(
        (int, str, object),
        (str, object),
    )
    renameRequested = Signal(
        (int, str, object),
        (str, object),
    )
    highlightResetRequested = Signal(bool)
    # Root-level lifecycle actions are routed with the owning document id so
    # the main window can activate an inactive document before applying the
    # command.  Child editing keeps the established routed signals above.
    rootActionRequested = Signal(int, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("modelTree")
        self.setHeaderHidden(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setMouseTracking(True)
        self.setStyleSheet(_MODEL_TREE_STYLESHEET)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.itemClicked.connect(self._on_clicked)
        self.itemDoubleClicked.connect(self._on_double_clicked)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self._roots: dict[int, QTreeWidgetItem] = {}
        self._active_document_id: int | None = None
        self._renamable_by_document: dict[int, frozenset[str]] = {}
        self._non_highlightable_by_document: dict[int, frozenset[str]] = {}
        self._building_document_id: int | None = None
        self.clear_model()

    def clear_model(self) -> None:
        self._renamable_kinds: frozenset[str] = frozenset()
        self._non_highlightable_kinds: frozenset[str] = frozenset()
        self._roots.clear()
        self._active_document_id = None
        self._renamable_by_document.clear()
        self._non_highlightable_by_document.clear()
        self._building_document_id = None
        self.clear()
        root = self._item("模型", "empty", None)
        root.addChild(self._item("未打开模型", "empty", None))
        self.addTopLevelItem(root)
        root.setExpanded(True)

    def _add_empty_placeholder(self) -> QTreeWidgetItem:
        self._building_document_id = None
        root = self._item("模型", "empty", None)
        root.addChild(self._item("未打开模型", "empty", None))
        self.addTopLevelItem(root)
        root.setExpanded(True)
        return root

    @property
    def roots(self) -> dict[int, QTreeWidgetItem]:
        """Return the document-root index used by incremental updates.

        The mapping is intentionally exposed as a read-only-by-convention
        dictionary: tests and the window use it for O(1) identity checks, while
        all mutation remains inside ``insert_document``/``remove_document``.
        """

        return self._roots

    def insert_document(
        self,
        document_id: int,
        projection: Any,
        **options: Any,
    ) -> QTreeWidgetItem:
        """Append one document root without touching any other root."""

        normalized = int(document_id)
        options = dict(options)
        source_path = options.pop("source_path", self._projection_path(projection))
        model_name = options.pop("model_name", self._projection_name(projection))
        if normalized in self._roots:
            return self.update_document(normalized, projection, **options)
        model = self._projection_model(projection)
        projection_options = self._projection_options(projection, options)
        if model is None:
            return self.set_geometry_preview(
                str(model_name or "模型"),
                tuple(projection_options.get("feature_rows", ())),
                parts=tuple(projection_options.get("native_parts", ()))
                or None,
                active_part_id=projection_options.get("active_part_id"),
                document_id=normalized,
                source_path=source_path,
            )
        return self.set_model(
            model,
            document_id=normalized,
            source_path=source_path,
            model_name=model_name,
            **projection_options,
        )

    def update_document(
        self,
        document_id: int,
        projection: Any,
        changed: object | None = None,
        **options: Any,
    ) -> QTreeWidgetItem:
        """Replace only one indexed root and preserve all other roots."""

        del changed  # Projection deltas are already coalesced by the Session.
        normalized = int(document_id)
        options = dict(options)
        source_path = options.pop("source_path", self._projection_path(projection))
        model_name = options.pop("model_name", self._projection_name(projection))
        model = self._projection_model(projection)
        projection_options = self._projection_options(projection, options)
        if model is None:
            return self.set_geometry_preview(
                str(model_name or "模型"),
                tuple(projection_options.get("feature_rows", ())),
                parts=tuple(projection_options.get("native_parts", ()))
                or None,
                active_part_id=projection_options.get("active_part_id"),
                document_id=normalized,
                source_path=source_path,
            )
        return self.set_model(
            model,
            document_id=normalized,
            source_path=source_path,
            model_name=model_name,
            **projection_options,
        )

    def remove_document(self, document_id: int) -> bool:
        """Remove one root by integer identity, leaving every other root."""

        normalized = int(document_id)
        root = self._roots.pop(normalized, None)
        if root is None:
            return False
        index = self.indexOfTopLevelItem(root)
        if index >= 0:
            self.takeTopLevelItem(index)
        self._renamable_by_document.pop(normalized, None)
        self._non_highlightable_by_document.pop(normalized, None)
        if self._active_document_id == normalized:
            self._active_document_id = None
            self.setCurrentItem(None)
        if not self._roots:
            self._add_empty_placeholder()
        return True

    def set_active_document(self, document_id: int | None) -> None:
        """Mark the active root without creating a persistent row highlight."""

        normalized = None if document_id is None else int(document_id)
        previous = self._active_document_id
        if previous == normalized:
            root = None if normalized is None else self._roots.get(normalized)
            if root is not None:
                self._set_item_bold(root, True)
                self.setCurrentItem(root)
            return
        if previous is not None:
            old_root = self._roots.get(previous)
            if old_root is not None:
                self._set_item_bold(old_root, False)
        self._active_document_id = normalized
        if normalized is None:
            self.setCurrentItem(None)
            return
        root = self._roots.get(normalized)
        if root is not None:
            self._set_item_bold(root, True)
            self.setCurrentItem(root)
        else:
            self.setCurrentItem(None)

    @staticmethod
    def _set_item_bold(item: QTreeWidgetItem, bold: bool) -> None:
        font = item.font(0)
        font.setBold(bold)
        item.setFont(0, font)

    def _interaction_kinds(
        self,
        document_id: int,
        mapping: dict[int, frozenset[str]],
        legacy: frozenset[str],
    ) -> frozenset[str]:
        return mapping.get(int(document_id), legacy)

    @staticmethod
    def _projection_model(projection: Any) -> Any:
        return getattr(projection, "model", projection)

    @staticmethod
    def _projection_path(projection: Any) -> str | Path | None:
        return getattr(projection, "source_path", None) or getattr(
            projection,
            "project_path",
            None,
        )

    @staticmethod
    def _projection_name(projection: Any) -> str | None:
        return (
            getattr(projection, "display_name", None)
            or getattr(projection, "model_name", None)
            or getattr(getattr(projection, "model", None), "name", None)
        )

    @staticmethod
    def _projection_options(
        projection: Any,
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        section_definitions = getattr(
            projection,
            "section_definitions",
            None,
        )
        if section_definitions is None:
            section_definitions = getattr(projection, "sections", ())
        region_assignments = getattr(
            projection,
            "region_assignments",
            None,
        )
        if region_assignments is None:
            region_assignments = getattr(projection, "assignments", ())
        scope_names = getattr(projection, "scope_names", None)
        if scope_names is None:
            named_regions = getattr(projection, "named_regions", None)
            scope_names = (
                tuple(named_regions)
                if named_regions is not None
                else None
            )
        feature_rows = getattr(projection, "feature_rows", None)
        if feature_rows is None:
            feature_rows = tuple(
                getattr(item, "name", item)
                for item in getattr(projection, "feature_history", ())
            )
        values = {
            "feature_rows": feature_rows or (),
            "part_name": getattr(projection, "part_name", None),
            "section_definitions": section_definitions or (),
            "region_assignments": region_assignments or (),
            "output_request_projections_by_step": getattr(
                projection,
                "output_request_projections_by_step",
                None,
            ),
            "scope_names": scope_names,
            "native_parts": tuple(getattr(projection, "parts", ())),
            "active_part_id": getattr(projection, "active_part_id", None),
        }
        values.update(options)
        return values

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
        output_request_projections_by_step: Mapping[
            str,
            Sequence[Any],
        ] | None = None,
        scope_names: Collection[str] | None = None,
        native_parts: tuple[Any, ...] = (),
        active_part_id: str | None = None,
        document_id: int | None = None,
        source_path: str | Path | None = None,
    ) -> QTreeWidgetItem:
        """Build one model root.

        Without ``document_id`` this keeps the legacy single-document
        replacement behavior.  With an ID, only that indexed root is replaced
        and the widget is never globally cleared.
        """
        previous_root = (
            None
            if document_id is None
            else self._roots.get(int(document_id))
        )
        view_state = (
            self._capture_view_state()
            if document_id is None
            else self._capture_view_state(previous_root)
            if previous_root is not None
            else None
        )
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
        if document_id is not None:
            normalized_document_id = int(document_id)
            self._renamable_by_document[normalized_document_id] = (
                self._renamable_kinds
            )
            self._non_highlightable_by_document[normalized_document_id] = (
                self._non_highlightable_kinds
            )
        if document_id is None:
            self._roots.clear()
            self._active_document_id = None
            self._renamable_by_document.clear()
            self._non_highlightable_by_document.clear()
            self.clear()
            self._building_document_id = 0
        else:
            normalized_document_id = int(document_id)
            previous_root = self._roots.pop(normalized_document_id, None)
            previous_index = (
                self.indexOfTopLevelItem(previous_root)
                if previous_root is not None
                else -1
            )
            if previous_root is not None:
                if previous_index >= 0:
                    self.takeTopLevelItem(previous_index)
            if self.topLevelItemCount() == 1:
                placeholder = self.topLevelItem(0)
                if placeholder.data(0, ROLE_KIND) == "empty":
                    self.takeTopLevelItem(0)
            self._building_document_id = normalized_document_id
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
            self._display_name(
                model_name or getattr(model, "name", None) or "模型"
            ),
            "model",
            None,
        )
        if source_path is not None:
            root.setToolTip(0, str(Path(source_path)))
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
                        native_feature_label(record.name),
                        "feature",
                        str(record.name),
                    )
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
                        native_feature_label(row),
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
            boundary_definitions = effective_step_boundaries(model, step)
            boundary_sources = tuple(
                (
                    source_index,
                    source_step,
                    source_boundary_index,
                )
                for source_index, source_step in enumerate(
                    model.steps[: index + 1]
                )
                for source_boundary_index, _boundary in enumerate(
                    source_step.boundaries
                )
            )
            boundary_count = (
                len(boundary_definitions)
                if boundary_definitions
                else getattr(step, "summary_boundary_count", 0)
            )
            bc_root = self._category(
                step_item,
                "边界条件",
                boundary_count,
            )
            for bc_index, boundary in enumerate(boundary_definitions):
                source_index, source_step, source_boundary_index = (
                    boundary_sources[bc_index]
                )
                inherited = source_index < index
                identity = getattr(boundary, "name", None)
                label = identity or f"位移约束 {source_boundary_index + 1}"
                source_key = (
                    (source_index, source_boundary_index)
                    if inherited or identity is None
                    else (source_step.name, identity)
                )
                boundary_item = self._item(
                    label,
                    "inherited_boundary" if inherited else "boundary",
                    source_key,
                )
                boundary_item.setData(0, ROLE_INHERITED, inherited)
                bc_root.addChild(boundary_item)
            load_count = (
                len(step.cloads)
                + len(step.surface_loads)
                + len(step.edge_loads)
                + len(step.line_loads)
                + len(getattr(step, "body_loads", ()))
                + len(getattr(step, "gravity_loads", ()))
            )
            load_root = self._category(
                step_item,
                "载荷",
                getattr(step, "summary_load_count", load_count),
            )
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
            output_entries = tuple(
                (output_index, output)
                for output_index, output in enumerate(step.outputs)
            )
            if output_request_projections_by_step is not None:
                output_entries = tuple(
                    (
                        projection.request_index,
                        projection.executable_authoring_request,
                    )
                    for projection in output_request_projections_by_step.get(
                        step.name,
                        (),
                    )
                    if projection.executable_authoring_request is not None
                )
            output_count = sum(
                max(1, len(output.variables))
                for _output_index, output in output_entries
            )
            output_root = self._category(
                step_item,
                "输出请求",
                (
                    output_count
                    if output_request_projections_by_step is not None
                    else getattr(step, "summary_output_count", output_count)
                ),
            )
            for output_index, output in output_entries:
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
        if document_id is not None:
            self._roots[int(document_id)] = root
            self.insertTopLevelItem(
                previous_index
                if previous_index >= 0
                else self.topLevelItemCount(),
                root,
            )
        else:
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
        self._restore_view_state(view_state, root)
        self._building_document_id = None
        return root

    def set_geometry_preview(
        self,
        name: str,
        feature_rows: tuple[str, ...],
        *,
        part_name: str = "部件-1",
        bodies: tuple[tuple[str, str, tuple[str, ...]], ...] = (),
        parts: tuple[Any, ...] | None = None,
        active_part_id: str | None = None,
        document_id: int | None = None,
        source_path: str | Path | None = None,
    ) -> QTreeWidgetItem:
        """显示模型以及稳定的原生部件层级。"""
        previous_root = (
            None
            if document_id is None
            else self._roots.get(int(document_id))
        )
        view_state = (
            self._capture_view_state()
            if document_id is None
            else self._capture_view_state(previous_root)
            if previous_root is not None
            else None
        )
        self._building_document_id = (
            None if document_id is None else int(document_id)
        )
        if document_id is not None:
            previous_root = self._roots.pop(int(document_id), None)
            previous_index = (
                self.indexOfTopLevelItem(previous_root)
                if previous_root is not None
                else -1
            )
            if previous_root is not None:
                if previous_index >= 0:
                    self.takeTopLevelItem(previous_index)
            if self.topLevelItemCount() == 1:
                placeholder = self.topLevelItem(0)
                if placeholder.data(0, ROLE_KIND) == "empty":
                    self.takeTopLevelItem(0)
        self._renamable_kinds = frozenset({"model", "part"})
        self._non_highlightable_kinds = (
            frozenset({"model", "feature"})
            if parts is not None
            else frozenset({"model", "part", "feature"})
        )
        if document_id is not None:
            normalized_document_id = int(document_id)
            self._renamable_by_document[normalized_document_id] = (
                self._renamable_kinds
            )
            self._non_highlightable_by_document[normalized_document_id] = (
                self._non_highlightable_kinds
            )
        if document_id is None:
            self._roots.clear()
            self._active_document_id = None
            self._renamable_by_document.clear()
            self._non_highlightable_by_document.clear()
            self.clear()
        root = self._item(self._display_name(name), "model", None)
        if source_path is not None:
            root.setToolTip(0, str(Path(source_path)))
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
                        native_feature_label(row.name),
                        "feature",
                        str(row.name),
                    )
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
            if document_id is not None:
                self._roots[int(document_id)] = root
                self.insertTopLevelItem(
                    previous_index
                    if previous_index >= 0
                    else self.topLevelItemCount(),
                    root,
                )
            else:
                self.addTopLevelItem(root)
            root.setExpanded(True)
            if active_item is not None:
                self.setCurrentItem(active_item)
            self._restore_view_state(view_state, root)
            self._building_document_id = None
            return root
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
                            native_feature_label(row),
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
                        native_feature_label(row),
                        "feature",
                        str(row),
                    )
                )
        root.addChild(part)
        if document_id is not None:
            self._roots[int(document_id)] = root
            self.insertTopLevelItem(
                previous_index
                if previous_index >= 0
                else self.topLevelItemCount(),
                root,
            )
        else:
            self.addTopLevelItem(root)
        root.setExpanded(True)
        part.setExpanded(True)
        self._restore_view_state(view_state, root)
        self._building_document_id = None
        return root

    def _capture_view_state(
        self,
        root: QTreeWidgetItem | None = None,
    ) -> _TreeViewState | None:
        """Capture navigation state before replacing the projected items."""

        if root is None:
            if self.topLevelItemCount() != 1:
                return None
            root = self.topLevelItem(0)
        if root.data(0, ROLE_KIND) != "model":
            return None
        expanded_paths: set[tuple[tuple[str, str], ...]] = set()
        current_path = None
        current = self.currentItem()

        def visit(
            item: QTreeWidgetItem,
            parent_path: tuple[tuple[str, str], ...],
        ) -> None:
            nonlocal current_path
            path = parent_path + (self._view_state_segment(item),)
            if item.isExpanded():
                expanded_paths.add(path)
            if item is current:
                current_path = path
            for index in range(item.childCount()):
                visit(item.child(index), path)

        visit(root, ())
        return _TreeViewState(
            frozenset(expanded_paths),
            current_path,
            self.verticalScrollBar().value(),
        )

    def _restore_view_state(
        self,
        state: _TreeViewState | None,
        root: QTreeWidgetItem | None = None,
    ) -> None:
        """Restore expansion, selection, and scrolling after a tree rebuild."""

        if state is None:
            return
        if root is None:
            if self.topLevelItemCount() != 1:
                return
            root = self.topLevelItem(0)
        current = None

        def visit(
            item: QTreeWidgetItem,
            parent_path: tuple[tuple[str, str], ...],
        ) -> None:
            nonlocal current
            path = parent_path + (self._view_state_segment(item),)
            item.setExpanded(path in state.expanded_paths)
            if path == state.current_path:
                current = item
            for index in range(item.childCount()):
                visit(item.child(index), path)

        visit(root, ())
        if current is not None:
            self.setCurrentItem(current)
        self.verticalScrollBar().setValue(state.vertical_scroll)

    @staticmethod
    def _view_state_segment(item: QTreeWidgetItem) -> tuple[str, str]:
        kind = str(item.data(0, ROLE_KIND))
        key = item.data(0, ROLE_KEY)
        if key is not None:
            return kind, repr(key)
        text = item.text(0)
        if kind == "category":
            text = text.rsplit(" (", 1)[0]
        elif kind in {"model", "mesh"}:
            text = ""
        return kind, text

    def select_entity(
        self,
        kind: str,
        key: int,
        document_id: int | None = None,
    ) -> None:
        if kind in {"node", "element"}:
            mesh = self._find_kind("mesh", document_id)
            if mesh is not None:
                self.setCurrentItem(mesh)
                self.scrollToItem(mesh)
            return
        if document_id is None:
            iterator = self.invisibleRootItem()
            stack = [
                iterator.child(index)
                for index in range(iterator.childCount())
            ]
        else:
            root = self._roots.get(int(document_id))
            if root is None:
                return
            stack = [root]
        while stack:
            item = stack.pop()
            if item.data(0, ROLE_KIND) == kind and item.data(0, ROLE_KEY) == key:
                self.setCurrentItem(item)
                self.scrollToItem(item)
                return
            stack.extend(item.child(index) for index in range(item.childCount()))

    def _find_kind(
        self,
        kind: str,
        document_id: int | None = None,
    ) -> QTreeWidgetItem | None:
        if document_id is None:
            root = self.invisibleRootItem()
            stack = [root.child(index) for index in range(root.childCount())]
        else:
            indexed_root = self._roots.get(int(document_id))
            if indexed_root is None:
                return None
            stack = [indexed_root]
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
        item.setData(0, ROLE_DOCUMENT_ID, self._building_document_id)
        icon_name = _TREE_ICONS.get(kind)
        if icon_name is not None:
            item.setIcon(0, icon(icon_name))
        return item

    @staticmethod
    def _display_name(value: object) -> str:
        text = str(value)
        suffix = Path(text).suffix.casefold()
        if suffix in {
            ".fempy",
            ".femproj",
            ".femres",
            ".inp",
            ".json",
        }:
            return Path(text).stem
        return text

    def _entry(
        self,
        item: QTreeWidgetItem | None,
    ) -> tuple[int, str, object] | None:
        if item is None:
            return None
        document_id = item.data(0, ROLE_DOCUMENT_ID)
        if document_id is None:
            document_id = 0
        kind = str(item.data(0, ROLE_KIND))
        key = item.data(0, ROLE_KEY)
        if kind in {"empty", "category", "detail"}:
            return None
        if kind == "inherited_boundary":
            return int(document_id), "boundary", key
        return int(document_id), kind, key

    @staticmethod
    def _emit_routed(signal: Signal, entry: tuple[int, str, object]) -> None:
        document_id, kind, key = entry
        # The first overload is the document-aware contract.  Emit the legacy
        # overload explicitly so old integrations continue to receive their
        # original (kind, key) pair.
        signal.emit(document_id, kind, key)
        signal[str, object].emit(kind, key)

    def _on_clicked(self, item: QTreeWidgetItem) -> None:
        entry = self._entry(item)
        non_highlightable = self._non_highlightable_kinds
        if entry is not None:
            non_highlightable = self._interaction_kinds(
                entry[0],
                self._non_highlightable_by_document,
                self._non_highlightable_kinds,
            )
        will_highlight = (
            entry is not None
            and entry[1] not in non_highlightable
        )
        self.highlightResetRequested.emit(will_highlight)
        if will_highlight:
            self._emit_routed(self.highlightRequested, entry)

    def _on_double_clicked(self, item: QTreeWidgetItem) -> None:
        entry = self._entry(item)
        if entry is not None:
            editable = (
                entry[1] in _EDITABLE_KINDS
                and not bool(item.data(0, ROLE_INHERITED))
                and not (
                    entry[1] == "part"
                    and type(entry[2]) is not str
                )
            )
            signal = (
                self.editRequested
                if editable
                else self.informationRequested
            )
            self._emit_routed(signal, entry)

    def _show_context_menu(self, position: QPoint) -> None:
        item = self.itemAt(position)
        entry = self._entry(item)
        if entry is None:
            return
        renamable = self._interaction_kinds(
            entry[0],
            self._renamable_by_document,
            self._renamable_kinds,
        )
        non_highlightable = self._interaction_kinds(
            entry[0],
            self._non_highlightable_by_document,
            self._non_highlightable_kinds,
        )
        self.setCurrentItem(item)
        menu = QMenu(self)
        root = (
            item.parent() is None
            and entry[1] == "model"
            and entry[0] in self._roots
        )
        if root:
            activate = menu.addAction("激活")
            save = menu.addAction("保存")
            save_as = menu.addAction("另存为")
            close = menu.addAction("关闭")
            chosen = menu.exec(self.viewport().mapToGlobal(position))
            if chosen is activate:
                self.rootActionRequested.emit(entry[0], "activate")
            elif chosen is save:
                self.rootActionRequested.emit(entry[0], "save")
            elif chosen is save_as:
                self.rootActionRequested.emit(entry[0], "save_as")
            elif chosen is close:
                self.rootActionRequested.emit(entry[0], "close")
            return
        highlight = (
            menu.addAction("高亮")
            if entry[1] not in non_highlightable
            else None
        )
        rename = (
            menu.addAction("重命名")
            if entry[1] in renamable
            else None
        )
        edit = (
            menu.addAction("编辑")
            if (
                entry[1] in _EDITABLE_KINDS
                and not bool(item.data(0, ROLE_INHERITED))
                and not (
                    entry[1] == "part"
                    and type(entry[2]) is not str
                )
            )
            else None
        )
        delete = (
            menu.addAction("删除")
            if (
                entry[1] in _DELETABLE_KINDS
                and not bool(item.data(0, ROLE_INHERITED))
                and not (
                    entry[1] == "part"
                    and type(entry[2]) is not str
                )
            )
            else None
        )
        information = menu.addAction("查看信息")
        chosen = menu.exec(self.viewport().mapToGlobal(position))
        if highlight is not None and chosen is highlight:
            self._emit_routed(self.highlightRequested, entry)
        elif rename is not None and chosen is rename:
            self._emit_routed(self.renameRequested, entry)
        elif edit is not None and chosen is edit:
            self._emit_routed(self.editRequested, entry)
        elif delete is not None and chosen is delete:
            self._emit_routed(self.deleteRequested, entry)
        elif chosen is information:
            self._emit_routed(self.informationRequested, entry)
