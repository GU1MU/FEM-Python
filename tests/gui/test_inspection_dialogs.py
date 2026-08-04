from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QAbstractTableModel, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QHeaderView, QLineEdit

from fem.io.inp import read
from fem_gui.inspection_dialogs import EntityInfoDialog, InspectionTableModel
from fem_gui.inspection_service import InspectionService
from fem_gui.mesh_browser import MeshBrowserDialog, NodeTableModel


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_entity_dialog_uses_read_only_models_and_follows_related_rows(gui_inp_path):
    _application()
    service = InspectionService(read(gui_inp_path))
    dialog = EntityInfoDialog(service.inspect("element", 1))
    requested = []
    dialog.entityRequested.connect(lambda kind, key: requested.append((kind, key)))

    connection_view = next(view for view in dialog.table_views if view.model().columnCount() == 5)
    model = connection_view.model()
    assert isinstance(model, InspectionTableModel)
    assert isinstance(model, QAbstractTableModel)
    dialog._activate_reference(model, 0)
    assert requested == [("node", 1)]
    dialog.close()


def test_entity_dialog_export_contains_all_table_rows(monkeypatch, gui_inp_path, tmp_path):
    _application()
    service = InspectionService(read(gui_inp_path))
    inspection = service.inspect("node_set", "LEFT")
    dialog = EntityInfoDialog(inspection)
    target = tmp_path / "node_set.tsv"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *_args, **_kwargs: (str(target), "制表符文本 (*.tsv)")),
    )

    dialog.export_information()

    content = target.read_text(encoding="utf-8-sig")
    for row in inspection.pages[0].tables[0].rows:
        assert "\t".join(row) in content
    dialog.close()


def test_small_information_tables_fit_without_vertical_scrollbars(gui_inp_path):
    app = _application()
    service = InspectionService(read(gui_inp_path))
    for kind, key in (("model", None), ("element", 1), ("node_set", "LEFT")):
        dialog = EntityInfoDialog(service.inspect(kind, key))
        dialog.show()
        app.processEvents()
        for view in dialog.table_views:
            if view.model().rowCount() <= 10 or view.model().columnCount() == 2:
                assert view.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
                assert not view.verticalScrollBar().isVisible()
        dialog.close()


def test_mesh_browser_uses_table_models_filters_and_entity_signals(gui_inp_path):
    _application()
    service = InspectionService(read(gui_inp_path))
    dialog = MeshBrowserDialog(service)
    assert isinstance(dialog.node_model, NodeTableModel)
    assert isinstance(dialog.node_model, QAbstractTableModel)
    assert dialog.node_model.rowCount() == 4
    assert dialog.element_model.rowCount() == 1
    assert all(not button.isEnabled() for button in dialog.selection_buttons)
    assert dialog.findChild(QLineEdit, "elementIdSearch").maximumWidth() == 360
    assert dialog.findChild(QLineEdit, "elementIdSearch").minimumWidth() == 260
    assert (
        dialog.element_view.horizontalHeader().sectionResizeMode(2)
        == QHeaderView.ResizeMode.Stretch
    )
    assert (
        dialog.element_view.horizontalHeader().sectionResizeMode(5)
        != QHeaderView.ResizeMode.Stretch
    )
    tooltip = dialog.element_model.data(
        dialog.element_model.index(0, 2), Qt.ItemDataRole.ToolTipRole
    )
    assert tooltip == "1, 2, 3, 4"

    dialog.node_proxy.set_set_filter("RIGHT")
    assert dialog.node_proxy.rowCount() == 2
    dialog.node_proxy.set_search("2")
    assert dialog.node_proxy.rowCount() == 1
    dialog.element_proxy.set_type_filter("Quad4")
    dialog.element_proxy.set_set_filter("SOLID")
    assert dialog.element_proxy.rowCount() == 1

    requested = []
    dialog.entityInformationRequested.connect(lambda kind, key: requested.append((kind, key)))
    index = dialog.element_proxy.index(0, 0)
    dialog.tabs.setCurrentIndex(1)
    dialog.element_view.setCurrentIndex(index)
    assert all(button.isEnabled() for button in dialog.selection_buttons)
    dialog._open_index(dialog.element_view, index)
    assert requested == [("element", 1)]
    dialog.close()
