from __future__ import annotations

import os
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from fem.core.model import MaterialDefinition
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from tests.helpers.file_builders import write_inp


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_task(window: FEMMainWindow, timeout: float = 20.0) -> None:
    deadline = monotonic() + timeout
    application = _application()
    while window.busy and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not window.busy


def _write_static_plate(tmp_path):
    return write_inp(
        tmp_path,
        "session_plate.inp",
        [
            "*Heading",
            "Session plate",
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "3, 1., 1.",
            "4, 0., 1.",
            "*Element, type=CPS4, elset=SOLID",
            "1, 1, 2, 3, 4",
            "*Nset, nset=LEFT",
            "1, 4",
            "*Nset, nset=RIGHT",
            "2, 3",
            "*Elset, elset=SOLID",
            "1",
            "*Material, name=STEEL",
            "*Elastic",
            "210000., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*Boundary",
            "LEFT, 1, 2, 0.",
            "*Step, name=Static-1",
            "*Static",
            "*Cload",
            "RIGHT, 1, 10.",
            "*Output, field",
            "*Node Output",
            "U, RF",
            "*Element Output",
            "S",
            "*End Step",
        ],
    )


def test_inp_open_check_solve_then_definition_edit_invalidates_every_projection(
    tmp_path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    gui_inp_path = _write_static_plate(tmp_path)
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(gui_inp_path), ""),
    )
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )

    window.open_inp()
    _wait_for_task(window)

    assert errors == []
    assert window.document.source_kind == "imported"
    assert window.document.source_path == gui_inp_path
    assert window.document.artifact is not None
    assert window.geometry.artifact_id == window.document.artifact.artifact_id
    assert window.viewport.artifact_id == window.document.artifact.artifact_id
    assert window.inspection_service is not None

    assert window.check_current_model(show_success=False), errors
    assert window.session.can_submit("Static-1")
    assert window.actions["submit_job"].isEnabled()
    run = window._submit_job("Job-1", "Static-1")
    assert run is not None
    _wait_for_task(window)

    assert errors == []
    current = window.session.current_result()
    assert current is not None
    assert current.provenance.run_id == run.run_id
    assert current.provenance.artifact_id == window.document.artifact.artifact_id
    assert window.result_data.run_id == run.run_id
    assert window.result_data.artifact_id == current.provenance.artifact_id
    assert window.viewport.run_id == run.run_id
    assert window.actions["query"].isEnabled()

    old_artifact_id = window.document.artifact.artifact_id
    materials = tuple(window.document.material_definitions)
    first = materials[0]
    properties = dict(first.properties)
    properties["E"] = float(properties["E"]) * 0.99
    changed_materials = (
        MaterialDefinition(first.name, properties),
        *materials[1:],
    )
    window._apply_model_definition_changes(
        "材料已修改",
        materials=changed_materials,
    )

    assert window.document.artifact.artifact_id != old_artifact_id
    assert window.session.validation_for("Static-1") is None
    assert not window.session.can_submit("Static-1")
    assert window.document.runs == ()
    assert window.session.current_result() is None
    assert window.result_data is None
    assert window.viewport.run_id is None
    assert window.result_tree.topLevelItem(0).text(0) == "尚无分析结果"
    for name in (
        "undeformed",
        "deformed",
        "contour",
        "field",
        "query",
        "export",
    ):
        assert not window.actions[name].isEnabled()
    assert not window.result_variable_combo.isEnabled()
    window.close()
