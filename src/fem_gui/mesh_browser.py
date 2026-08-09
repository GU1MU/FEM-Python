"""基于模型/代理模型的只读节点和单元浏览器。"""

from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .inspection_service import InspectionService, format_number


class _EntityTableModel(QAbstractTableModel):
    columns: tuple[str, ...] = ()
    kind = ""

    def __init__(self, service: InspectionService, ids: tuple[int, ...], parent=None) -> None:
        super().__init__(parent)
        self.service = service
        self.ids = ids

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.ids)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.columns[section]
        return super().headerData(section, orientation, role)

    def entity_id(self, row: int) -> int:
        return self.ids[row]


class NodeTableModel(_EntityTableModel):
    """按需从 InspectionService 读取节点表格值。"""

    columns = ("编号", "X", "Y", "Z", "所属节点集")
    kind = "node"

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.service.node_row(self.entity_id(index.row()))
        value = row[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return format_number(value) if index.column() in {1, 2, 3} else str(value)
        if role == Qt.ItemDataRole.UserRole:
            return value
        return None


class ElementTableModel(_EntityTableModel):
    """按需从 InspectionService 读取单元表格值。"""

    columns = ("编号", "类型", "连接节点预览", "所属单元集", "材料", "截面")
    kind = "element"

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.service.element_row(self.entity_id(index.row()))[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return str(value)
        if role == Qt.ItemDataRole.UserRole:
            return value
        if role == Qt.ItemDataRole.ToolTipRole and index.column() == 2:
            node_ids = self.service.element_record(self.entity_id(index.row()))["node_ids"]
            return ", ".join(str(node_id) for node_id in node_ids)
        return None


class MeshFilterProxyModel(QSortFilterProxyModel):
    """组合编号、类型和集合过滤条件。"""

    def __init__(self, kind: str, service: InspectionService, parent=None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.service = service
        self.search_text = ""
        self.type_filter = ""
        self.set_filter = ""
        self.setDynamicSortFilter(True)

    def set_search(self, value: str) -> None:
        self.search_text = value.strip()
        self.invalidate()

    def set_type_filter(self, value: str) -> None:
        self.type_filter = value
        self.invalidate()

    def set_set_filter(self, value: str) -> None:
        self.set_filter = value
        self.invalidate()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        source = self.sourceModel()
        entity_id = source.entity_id(source_row)
        if self.search_text and self.search_text not in str(entity_id):
            return False
        if self.kind == "node":
            return not self.set_filter or self.set_filter in self.service.node_sets_by_node.get(entity_id, ())
        element = self.service.elements[entity_id]
        if self.type_filter and str(element.type) != self.type_filter:
            return False
        return not self.set_filter or self.set_filter in self.service.element_sets_by_element.get(entity_id, ())


class MeshBrowserDialog(QDialog):
    """大网格安全的节点/单元浏览、过滤与定位窗口。"""

    entityInformationRequested = Signal(str, object)
    highlightRequested = Signal(str, object)
    locateRequested = Signal(str, object)

    def __init__(self, service: InspectionService, parent=None) -> None:
        super().__init__(parent)
        self.service = service
        self.setObjectName("meshBrowserDialog")
        self.setWindowTitle("网格浏览器")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(940, 620)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("meshBrowserTabs")
        self.node_model = NodeTableModel(service, tuple(sorted(service.nodes)), self)
        self.element_model = ElementTableModel(service, tuple(sorted(service.elements)), self)
        self.node_proxy = MeshFilterProxyModel("node", service, self)
        self.element_proxy = MeshFilterProxyModel("element", service, self)
        self.node_proxy.setSortRole(Qt.ItemDataRole.UserRole)
        self.element_proxy.setSortRole(Qt.ItemDataRole.UserRole)
        self.node_proxy.setSourceModel(self.node_model)
        self.element_proxy.setSourceModel(self.element_model)
        self.node_view = self._build_node_page()
        self.element_view = self._build_element_page()
        layout.addWidget(self.tabs, 1)
        buttons = QHBoxLayout()
        self.selection_buttons: list[QPushButton] = []
        for text, callback in (
            ("高亮", self.highlight_current), ("定位", self.locate_current),
            ("复制编号", self.copy_current_id),
        ):
            button = QPushButton(text, self)
            button.clicked.connect(callback)
            button.setEnabled(False)
            self.selection_buttons.append(button)
            buttons.addWidget(button)
        buttons.addStretch(1)
        close = QPushButton("关闭", self)
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.node_view.selectionModel().selectionChanged.connect(self._update_selection_buttons)
        self.element_view.selectionModel().selectionChanged.connect(self._update_selection_buttons)
        self.tabs.currentChanged.connect(self._update_selection_buttons)

    def _build_node_page(self) -> QTableView:
        page = QWidget(self.tabs)
        layout = QVBoxLayout(page)
        filters = QHBoxLayout()
        search = QLineEdit(page)
        search.setObjectName("nodeIdSearch")
        search.setPlaceholderText("按节点编号搜索")
        search.setMinimumWidth(260)
        search.setMaximumWidth(360)
        node_set = QComboBox(page)
        node_set.setObjectName("nodeSetFilter")
        node_set.setMinimumWidth(150)
        node_set.addItem("全部节点集", "")
        for name in self.service.model.node_sets:
            node_set.addItem(name, name)
        filters.addWidget(QLabel("编号", page))
        filters.addWidget(search)
        filters.addWidget(QLabel("节点集", page))
        filters.addWidget(node_set)
        filters.addStretch(1)
        layout.addLayout(filters)
        view = self._make_view(self.node_proxy, page)
        view.setObjectName("nodeBrowserTable")
        view.setColumnWidth(0, 80)
        for column in (1, 2, 3):
            view.setColumnWidth(column, 110)
        view.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(view, 1)
        search.textChanged.connect(self.node_proxy.set_search)
        node_set.currentIndexChanged.connect(lambda: self.node_proxy.set_set_filter(str(node_set.currentData())))
        self.tabs.addTab(page, "节点")
        return view

    def _build_element_page(self) -> QTableView:
        page = QWidget(self.tabs)
        layout = QVBoxLayout(page)
        filters = QHBoxLayout()
        search = QLineEdit(page)
        search.setObjectName("elementIdSearch")
        search.setPlaceholderText("按单元编号搜索")
        search.setMinimumWidth(260)
        search.setMaximumWidth(360)
        element_type = QComboBox(page)
        element_type.setObjectName("elementTypeFilter")
        element_type.setMinimumWidth(135)
        element_type.addItem("全部类型", "")
        for name in sorted({str(element.type) for element in self.service.elements.values()}):
            element_type.addItem(name, name)
        element_set = QComboBox(page)
        element_set.setObjectName("elementSetFilter")
        element_set.setMinimumWidth(155)
        element_set.addItem("全部单元集", "")
        for name in self.service.model.element_sets:
            element_set.addItem(name, name)
        filters.addWidget(QLabel("编号", page))
        filters.addWidget(search)
        filters.addWidget(QLabel("类型", page))
        filters.addWidget(element_type)
        filters.addWidget(QLabel("单元集", page))
        filters.addWidget(element_set)
        filters.addStretch(1)
        layout.addLayout(filters)
        view = self._make_view(self.element_proxy, page)
        view.setObjectName("elementBrowserTable")
        view.setColumnWidth(0, 80)
        view.setColumnWidth(1, 105)
        view.setColumnWidth(3, 160)
        view.setColumnWidth(4, 150)
        view.setColumnWidth(5, 120)
        view.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(view, 1)
        search.textChanged.connect(self.element_proxy.set_search)
        element_type.currentIndexChanged.connect(lambda: self.element_proxy.set_type_filter(str(element_type.currentData())))
        element_set.currentIndexChanged.connect(lambda: self.element_proxy.set_set_filter(str(element_set.currentData())))
        self.tabs.addTab(page, "单元")
        return view

    def _make_view(self, model: QSortFilterProxyModel, parent: QWidget) -> QTableView:
        view = QTableView(parent)
        view.setModel(model)
        view.setSortingEnabled(True)
        view.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        view.setAlternatingRowColors(True)
        view.verticalHeader().setVisible(False)
        view.verticalHeader().setDefaultSectionSize(22)
        view.horizontalHeader().setStretchLastSection(False)
        view.doubleClicked.connect(lambda index, source=view: self._open_index(source, index))
        return view

    def _update_selection_buttons(self, *_args) -> None:
        enabled = self._current_entity() is not None
        for button in self.selection_buttons:
            button.setEnabled(enabled)

    def _current_entity(self) -> tuple[str, int] | None:
        view = self.node_view if self.tabs.currentIndex() == 0 else self.element_view
        index = view.currentIndex()
        if not index.isValid():
            return None
        proxy = view.model()
        source_index = proxy.mapToSource(index)
        source = proxy.sourceModel()
        return source.kind, source.entity_id(source_index.row())

    def _open_index(self, view: QTableView, index: QModelIndex) -> None:
        proxy = view.model()
        source_index = proxy.mapToSource(index)
        source = proxy.sourceModel()
        self.entityInformationRequested.emit(source.kind, source.entity_id(source_index.row()))

    def highlight_current(self) -> None:
        entry = self._current_entity()
        if entry is not None:
            self.highlightRequested.emit(*entry)

    def locate_current(self) -> None:
        entry = self._current_entity()
        if entry is not None:
            self.locateRequested.emit(*entry)

    def copy_current_id(self) -> None:
        entry = self._current_entity()
        if entry is not None:
            QApplication.clipboard().setText(str(entry[1]))
