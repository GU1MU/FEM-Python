"""结构化有限元对象信息的统一只读窗口。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .inspection_service import EntityInspection, EntityReference, InspectionPage, InspectionTable


class InspectionTableModel(QAbstractTableModel):
    """显示服务已经准备好的只读表格。"""

    def __init__(self, table: InspectionTable, parent=None) -> None:
        super().__init__(parent)
        self.table = table

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.table.rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.table.columns)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}:
            return self.table.rows[index.row()][index.column()]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.table.columns[section]
        return super().headerData(section, orientation, role)

    def reference(self, row: int) -> EntityReference | None:
        if 0 <= row < len(self.table.references):
            return self.table.references[row]
        return None


class EntityInfoDialog(QDialog):
    """显示一个对象的字段、关联表格和结果页。"""

    highlightRequested = Signal(str, object)
    locateRequested = Signal(str, object)
    entityRequested = Signal(str, object)

    def __init__(self, inspection: EntityInspection, parent=None) -> None:
        super().__init__(parent)
        self.inspection = inspection
        self.table_views: list[QTableView] = []
        self.setObjectName("entityInfoDialog")
        self.setWindowTitle(inspection.title)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 9, 9, 8)
        layout.setSpacing(7)
        if len(inspection.pages) == 1:
            layout.addWidget(self._build_page(inspection.pages[0]), 1)
        else:
            tabs = QTabWidget(self)
            tabs.setObjectName("inspectionTabs")
            for page in inspection.pages:
                tabs.addTab(self._build_page(page), page.title)
            layout.addWidget(tabs, 1)
        buttons = QHBoxLayout()
        highlight = QPushButton("高亮", self)
        locate = QPushButton("定位", self)
        can_locate = inspection.kind in {
            "node", "element", "node_set", "element_set", "surface", "edge",
            "material", "section", "boundary", "cload", "surface_load", "edge_load",
            "gravity_load",
        }
        highlight.setEnabled(can_locate)
        locate.setEnabled(can_locate)
        copy = QPushButton("复制", self)
        close = QPushButton("关闭", self)
        highlight.clicked.connect(lambda: self.highlightRequested.emit(inspection.kind, inspection.key))
        locate.clicked.connect(lambda: self.locateRequested.emit(inspection.kind, inspection.key))
        export = QPushButton("导出", self)
        copy.clicked.connect(self.copy_information)
        export.clicked.connect(self.export_information)
        close.clicked.connect(self.close)
        for button in (highlight, locate, copy, export):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        maximum_columns = max(
            (len(table.columns) for page in inspection.pages for table in page.tables),
            default=2,
        )
        width = 560 if maximum_columns <= 2 else 680 if maximum_columns <= 4 else 800
        layout.activate()
        self.adjustSize()
        self.resize(width, min(650, max(260, self.sizeHint().height())))

    def _build_page(self, page: InspectionPage) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(3, 5, 3, 3)
        layout.setSpacing(6)
        if page.fields:
            table = InspectionTable("基本信息", ("项目", "内容"), tuple(page.fields))
            view = self._make_table(table, max_visible_rows=14)
            view.setColumnWidth(0, 165)
            layout.addWidget(view)
        for table in page.tables:
            if table.title:
                label = QLabel(table.title, widget)
                label.setObjectName("inspectionTableTitle")
                layout.addWidget(label)
            layout.addWidget(self._make_table(table))
        layout.addStretch(1)
        return widget

    def _make_table(self, table: InspectionTable, max_visible_rows: int = 10) -> QTableView:
        view = QTableView(self)
        view.setObjectName("inspectionTable")
        model = InspectionTableModel(table, view)
        view.setModel(model)
        view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        view.setAlternatingRowColors(True)
        view.verticalHeader().setVisible(False)
        view.verticalHeader().setDefaultSectionSize(23)
        view.horizontalHeader().setStretchLastSection(True)
        visible_rows = min(max_visible_rows, max(1, len(table.rows)))
        table_height = view.horizontalHeader().sizeHint().height() + visible_rows * 23 + 2
        view.setFixedHeight(table_height)
        view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            if len(table.rows) <= max_visible_rows
            else Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        view.doubleClicked.connect(lambda index, source=model: self._activate_reference(source, index.row()))
        self.table_views.append(view)
        return view

    def _activate_reference(self, model: InspectionTableModel, row: int) -> None:
        reference = model.reference(row)
        if reference is not None:
            self.entityRequested.emit(reference.kind, reference.key)

    def copy_information(self) -> None:
        QApplication.clipboard().setText("\n".join(self._information_lines()))

    def export_information(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self, "导出信息", f"{self.inspection.title}.tsv", "制表符文本 (*.tsv);;文本文件 (*.txt)"
        )
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".tsv")
        target.write_text("\n".join(self._information_lines()), encoding="utf-8-sig")

    def _information_lines(self) -> list[str]:
        lines = [self.inspection.title]
        for page in self.inspection.pages:
            if len(self.inspection.pages) > 1:
                lines.append(f"[{page.title}]")
            lines.extend(f"{name}\t{value}" for name, value in page.fields)
            for table in page.tables:
                lines.append(table.title)
                lines.append("\t".join(table.columns))
                lines.extend("\t".join(row) for row in table.rows)
        return lines
