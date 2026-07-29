"""Focused integration tests for the public GUI command boundary."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.application import NativePart
from fem.geometry import RectangleGeometry
from fem_gui.commands import (
    CloseSessionCommand,
    GuiCommandStatus,
    NativeGeometryEdit,
    NewNativeProjectCommand,
)
from fem_gui.main_window import FEMMainWindow
from tests.helpers.gui_command_receipts import (
    await_succeeded,
    require_accepted,
    require_rejected,
)


FIXTURES = (
    Path(__file__).parents[1]
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)

PUBLIC_GUI_WORKFLOW_ENTRYPOINTS = (
    "new_native_project",
    "close_session",
    "open_inp_path",
    "open_project_path",
    "save_project_path",
    "reload_imported_source",
    "apply_native_geometry_edit",
    "apply_mesh_input_edit",
    "apply_named_region_edit",
    "apply_definition_edit",
    "generate_mesh",
    "check_step",
    "submit_run",
    "select_run_result",
    "export_result_csv",
    "export_result_vtk",
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_public_command_catalog_is_complete() -> None:
    _application()
    window = FEMMainWindow()

    missing = tuple(
        name
        for name in PUBLIC_GUI_WORKFLOW_ENTRYPOINTS
        if not callable(getattr(window, name, None))
    )

    assert missing == ()
    window.close()


def test_synchronous_public_edit_rejects_stale_revision_without_mutation() -> None:
    _application()
    window = FEMMainWindow()
    require_accepted(window.new_native_project(NewNativeProjectCommand()))
    base_revision = window.document.session_revision
    first = RectangleGeometry("Plate", 2.0, 1.0)
    require_accepted(
        window.apply_native_geometry_edit(
            NativeGeometryEdit(
                base_session_revision=base_revision,
                parts=(NativePart(),),
                recipe=first,
            )
        )
    )
    revision_after_first = window.document.session_revision

    rejected = window.apply_native_geometry_edit(
        NativeGeometryEdit(
            base_session_revision=base_revision,
            parts=(NativePart(),),
            recipe=RectangleGeometry("Stale", 4.0, 1.0),
        )
    )

    require_rejected(rejected, code="geometry.edit.rejected")
    assert rejected.command_id > 0
    assert window.document.session_revision == revision_after_first
    assert window.document.geometry_recipe == first
    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    window.close()


def test_pending_public_command_is_the_busy_gate_and_completion_handle() -> None:
    _application()
    window = FEMMainWindow()
    receipt = window.open_inp_path(FIXTURES / "truss2_tension.inp")

    assert receipt.status is GuiCommandStatus.PENDING
    require_rejected(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        ),
        code="task.busy",
    )
    terminal = await_succeeded(receipt)

    assert terminal.value is None
    assert receipt.completion is not None
    assert receipt.completion.task_id == terminal.task_id
    assert window.document.source_kind == "imported"
    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    window.close()


def test_public_inp_import_accepts_gb18030_comments(tmp_path) -> None:
    _application()
    source = FIXTURES / "truss2_tension.inp"
    path = tmp_path / "gb18030_truss.inp"
    path.write_bytes(
        "** 名称：完全固定；类型：位移/转角\n".encode("gb18030")
        + source.read_bytes()
    )
    window = FEMMainWindow()

    terminal = await_succeeded(window.open_inp_path(path))

    assert terminal.value is None
    assert window.document.source_kind == "imported"
    assert window.document.source_path == path
    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    window.close()
