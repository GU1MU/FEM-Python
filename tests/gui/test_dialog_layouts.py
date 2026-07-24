from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QHeaderView, QToolButton

from fem.core.model import MaterialDefinition
from fem_gui.analysis_definition_dialogs import LoadDialog, OutputRequestDialog
from fem_gui.document import NamedRegion, SectionDefinition
from fem_gui.main_window import FEMMainWindow
from fem_gui.model_dialogs import (
    MaterialEditDialog,
    MaterialManagerDialog,
    RegionAssignmentDialog,
    SectionEditDialog,
    SectionManagerDialog,
)
from fem_gui.preprocessing_dialogs import (
    NamedRegionManagerDialog,
    SketchContourDialog,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _show(dialog) -> int:
    dialog.show()
    _application().processEvents()
    return dialog.width()


def test_parameter_dialogs_do_not_force_sparse_content_wide():
    _application()
    material = MaterialDefinition(
        "Steel",
        {"E": 210000.0, "nu": 0.3},
    )
    section = SectionDefinition("Section-1", "Steel")
    dialogs = (
        (SectionEditDialog([material], section), 350),
        (RegionAssignmentDialog([section], ["DOMAIN"]), 310),
        (LoadDialog(["Step-1"], ["Loaded"], [], [], 2), 360),
        (OutputRequestDialog(["Step-1"]), 340),
        (SketchContourDialog(), 350),
    )

    for dialog, maximum in dialogs:
        assert _show(dialog) <= maximum
        dialog.close()


def test_manager_dialogs_use_compact_initial_sizes_and_content_columns():
    _application()
    material = MaterialDefinition(
        "Steel",
        {"E": 210000.0, "nu": 0.3},
    )
    section = SectionDefinition("Section-1", "Steel")
    material_edit = MaterialEditDialog(material)
    material_manager = MaterialManagerDialog([material])
    section_manager = SectionManagerDialog([material], [section])
    region_manager = NamedRegionManagerDialog(
        {"EdgeSet-1": NamedRegion("EdgeSet-1", "edge", (1,))}
    )

    assert _show(material_edit) <= 430
    assert _show(material_manager) <= 530
    assert _show(section_manager) <= 510
    assert _show(region_manager) <= 530
    assert (
        section_manager.table.horizontalHeader().sectionResizeMode(2)
        == QHeaderView.ResizeMode.ResizeToContents
    )
    assert (
        region_manager.table.horizontalHeader().sectionResizeMode(1)
        == QHeaderView.ResizeMode.ResizeToContents
    )
    assert (
        region_manager.table.horizontalHeader().sectionResizeMode(2)
        == QHeaderView.ResizeMode.ResizeToContents
    )
    for dialog in (
        material_edit,
        material_manager,
        section_manager,
        region_manager,
    ):
        dialog.close()


def test_parameter_form_labels_share_right_aligned_compact_style():
    _application()
    material = MaterialDefinition(
        "Steel",
        {"E": 210000.0, "nu": 0.3},
    )
    dialog = SectionEditDialog(
        [material],
        SectionDefinition("Section-1", "Steel"),
    )

    assert (
        dialog.form.labelAlignment()
        & Qt.AlignmentFlag.AlignRight
    )
    assert dialog.form.horizontalSpacing() == 12
    assert dialog.form.verticalSpacing() == 8


def test_ribbon_command_labels_are_not_clipped_by_button_width():
    _application()
    window = FEMMainWindow()

    clipped = [
        button.text()
        for button in window.ribbon.findChildren(QToolButton)
        if button.maximumWidth() < button.sizeHint().width()
    ]

    assert clipped == []
    window.close()
