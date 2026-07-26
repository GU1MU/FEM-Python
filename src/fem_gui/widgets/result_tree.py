"""只展示真实可用结果族的精简结果树。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QTreeWidget, QTreeWidgetItem

from fem.application.results import (
    FieldAvailability,
    FieldState,
    ResultCatalog,
    ScalarFieldSelection,
)

if TYPE_CHECKING:
    from ..visualization.result_adapter import ResultData


# Compatibility role for the pre-Phase-8 string field path.
ROLE_FIELD = int(Qt.ItemDataRole.UserRole)
ROLE_SELECTION = ROLE_FIELD + 1
ROLE_MATERIALIZATION_KEY = ROLE_FIELD + 2
ROLE_FIELD_STATE = ROLE_FIELD + 3


_FIELD_LABELS = {
    "result.field.u.node": "位移 U",
    "result.field.ur.node": "转角 UR",
    "result.field.rf.node": "反力 RF",
    "result.field.rm.node": "反力矩 RM",
    "result.field.le.centroid": "对数应变 LE（单元质心）",
    "result.field.s.integration_point": "应力 S（积分点）",
    "result.field.s.centroid": "应力 S（单元质心）",
    "result.field.s.element_nodal": "应力 S（单元节点）",
    "result.field.s.node_region": "应力 S（节点区域）",
    "result.field.s.resolved_nodal": "应力 S（平均节点）",
    "result.field.s.section_end": "应力 S（截面端点）",
    "result.field.s.section_node_envelope": "应力 S（截面节点包络）",
}
_FIELD_STATE_LABELS = {
    FieldState.READY: "就绪",
    FieldState.LAZY: "按需加载",
    FieldState.UNAVAILABLE: "不可用",
}


class ResultTree(QTreeWidget):
    """按位移、反力和应力组织当前单步结果。"""

    fieldSelectionActivated = Signal(ScalarFieldSelection)
    # Compatibility signal for callers still projecting ResultData strings.
    fieldActivated = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("resultTree")
        self.setHeaderHidden(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.itemDoubleClicked.connect(self._activate_item)
        self.clear_result()

    def clear_result(self) -> None:
        self.clear()
        item = QTreeWidgetItem(["尚无分析结果"])
        item.setData(0, ROLE_FIELD, None)
        self.addTopLevelItem(item)

    def set_catalog(self, step_name: str, catalog: ResultCatalog) -> None:
        """Populate the tree from one immutable application catalog."""

        if type(step_name) is not str:
            raise TypeError("step_name must be a string")
        if type(catalog) is not ResultCatalog:
            raise TypeError("catalog must be a ResultCatalog")

        self.clear()
        root = QTreeWidgetItem(["分析结果"])
        step = QTreeWidgetItem([step_name or "当前分析步"])
        root.addChild(step)

        default_item: QTreeWidgetItem | None = None
        for availability in catalog.fields:
            field_item, selected_component = self._catalog_field_item(
                availability,
                catalog.default_selection,
            )
            step.addChild(field_item)
            if selected_component is not None:
                default_item = selected_component

        self.addTopLevelItem(root)
        root.setExpanded(True)
        step.setExpanded(True)
        if default_item is not None:
            default_item.parent().setExpanded(True)
            self.setCurrentItem(default_item)

    @staticmethod
    def _catalog_field_item(
        availability: FieldAvailability,
        default_selection: ScalarFieldSelection,
    ) -> tuple[QTreeWidgetItem, QTreeWidgetItem | None]:
        descriptor = availability.descriptor
        state_label = _FIELD_STATE_LABELS[availability.state]
        field_label = _FIELD_LABELS.get(
            descriptor.label_key,
            descriptor.label_key,
        )
        field_item = QTreeWidgetItem([f"{field_label}（{state_label}）"])
        field_selection = ScalarFieldSelection(
            availability.key,
            descriptor.default_component,
        )
        _set_typed_item_data(
            field_item,
            availability,
            field_selection,
        )

        selected_component: QTreeWidgetItem | None = None
        for component in descriptor.columns:
            selection = ScalarFieldSelection(availability.key, component)
            component_item = QTreeWidgetItem([component])
            _set_typed_item_data(
                component_item,
                availability,
                selection,
            )
            field_item.addChild(component_item)
            if selection == default_selection:
                selected_component = component_item

        if availability.state is FieldState.UNAVAILABLE:
            _disable_item(field_item)
            for index in range(field_item.childCount()):
                _disable_item(field_item.child(index))
        return field_item, selected_component

    def set_result(self, step_name: str, data: ResultData) -> None:
        """Populate the legacy string-key projection during migration."""

        self.clear()
        root = QTreeWidgetItem(["分析结果"])
        step = QTreeWidgetItem([step_name or "当前分析步"])
        root.addChild(step)
        for label, key in self._families(data):
            item = QTreeWidgetItem([label])
            item.setData(0, ROLE_FIELD, key)
            step.addChild(item)
        self.addTopLevelItem(root)
        root.setExpanded(True)
        step.setExpanded(True)

    def select_selection(
        self,
        selection: ScalarFieldSelection,
    ) -> bool:
        """Select the exact catalog component without rebuilding the tree."""

        item = self._selection_item(selection)
        if item is None:
            return False
        parent = item.parent()
        while parent is not None:
            parent.setExpanded(True)
            parent = parent.parent()
        self.setCurrentItem(item)
        return True

    def has_selection(
        self,
        selection: ScalarFieldSelection,
    ) -> bool:
        """Return whether the exact component is present without changing UI."""

        return self._selection_item(selection) is not None

    def _selection_item(
        self,
        selection: ScalarFieldSelection,
    ) -> QTreeWidgetItem | None:
        if type(selection) is not ScalarFieldSelection:
            raise TypeError("selection must be a ScalarFieldSelection")
        pending = [
            self.topLevelItem(index)
            for index in range(self.topLevelItemCount())
        ]
        fallback: QTreeWidgetItem | None = None
        while pending:
            item = pending.pop(0)
            if item.data(0, ROLE_SELECTION) == selection:
                if item.childCount() == 0:
                    fallback = item
                    break
                if fallback is None:
                    fallback = item
            pending.extend(
                item.child(index)
                for index in range(item.childCount())
            )
        return fallback

    @staticmethod
    def _families(data: ResultData) -> tuple[tuple[str, str], ...]:
        # Compatibility logic remains isolated from the typed catalog path.
        from ..visualization.result_adapter import field_family

        fields = data.fields
        families: list[tuple[str, str]] = []
        labels = {
            "U": "位移 U",
            "R": "转角 R",
            "RF": "反力 RF",
            "RM": "反力矩 RM",
            "S": "应力 S",
        }
        for family in ("U", "R", "RF", "RM", "S"):
            keys = [key for key in fields if field_family(key) == family]
            if not keys:
                continue
            preferred = {
                "U": "U",
                "RF": "RF",
            }.get(family)
            selected = preferred if preferred in fields else keys[0]
            if family == "S":
                selected = next(
                    (key for key in keys if key.endswith(":S11AbsMax")),
                    next(
                        (key for key in keys if key.endswith(":Mises")),
                        keys[0],
                    ),
                )
            families.append((labels[family], selected))
        return tuple(families)

    def _activate_item(self, item: QTreeWidgetItem) -> None:
        selection = item.data(0, ROLE_SELECTION)
        state = item.data(0, ROLE_FIELD_STATE)
        if (
            type(selection) is ScalarFieldSelection
            and state != FieldState.UNAVAILABLE.value
        ):
            self.fieldSelectionActivated.emit(selection)
            return

        field = item.data(0, ROLE_FIELD)
        if field:
            self.fieldActivated.emit(str(field))


def _set_typed_item_data(
    item: QTreeWidgetItem,
    availability: FieldAvailability,
    selection: ScalarFieldSelection,
) -> None:
    item.setData(0, ROLE_SELECTION, selection)
    item.setData(0, ROLE_MATERIALIZATION_KEY, availability.key)
    item.setData(0, ROLE_FIELD_STATE, availability.state.value)


def _disable_item(item: QTreeWidgetItem) -> None:
    item.setFlags(
        item.flags() & ~Qt.ItemFlag.ItemIsEnabled & ~Qt.ItemFlag.ItemIsSelectable
    )
