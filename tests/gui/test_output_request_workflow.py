from __future__ import annotations

import ast
from copy import deepcopy
import inspect
import os
from pathlib import Path
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QDialog

from fem.io.inp import read
from fem.application import (
    AuthoringCapability,
    AuthoringStatus,
    describe_session_authoring,
)
from fem.core.model import OutputRequest
from fem.geometry import RectangleGeometry
from fem.steps.factory import static
import fem_gui.analysis_definition_dialogs as definition_dialogs_module
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from fem_gui.visualization.model_adapter import build_model_geometry


_RELOAD_LOSS_MESSAGE = (
    "此修改只保留在当前 Session；"
    "重新加载原 INP 后会恢复源文件中的输出请求。"
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_task(window: FEMMainWindow, timeout: float = 10.0) -> None:
    application = _application()
    deadline = monotonic() + timeout
    while window.busy and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not window.busy


def _install_imported(window: FEMMainWindow, path: Path) -> None:
    model = read(path)
    window._model_loaded(
        path,
        (model, build_model_geometry(model)),
    )


def _editable_step(steps):
    return next(
        step
        for step in steps
        if step.name.strip().casefold() != "initial"
    )


def _accepted_candidate_dialog(predicate):
    class AcceptedCandidateDialog:
        observed_candidates = ()

        def __init__(
            self,
            step_names,
            _parent=None,
            *,
            candidates=(),
            current=None,
            existing_requests_by_step=None,
        ) -> None:
            assert current is None
            assert existing_requests_by_step is not None
            self._step_name = step_names[0]
            self._candidates = tuple(candidates)
            type(self).observed_candidates = self._candidates
            self._candidate = next(
                candidate
                for candidate in self._candidates
                if predicate(candidate.authoring_request)
            )

        def exec(self):
            return QDialog.DialogCode.Accepted

        def definition(self):
            return (
                self._step_name,
                deepcopy(self._candidate.authoring_request),
            )

    return AcceptedCandidateDialog


def _capture_warnings(monkeypatch):
    values: list[tuple[str, str]] = []

    class FakeMessageBox:
        @staticmethod
        def warning(_parent, title, message):
            values.append((str(title), str(message)))
            return None

    monkeypatch.setattr(main_window_module, "QMessageBox", FakeMessageBox)
    return values


def test_imported_create_uses_catalog_candidate_and_reload_restores_source(
    gui_inp_path,
    monkeypatch,
) -> None:
    _application()
    source_bytes = gui_inp_path.read_bytes()
    window = FEMMainWindow()
    _install_imported(window, gui_inp_path)
    original = tuple(_editable_step(window.document.steps).outputs)
    authoring = describe_session_authoring(window.document)
    dialog_type = _accepted_candidate_dialog(
        lambda request: request.variables == ("S",)
    )
    monkeypatch.setattr(
        main_window_module,
        "OutputRequestDialog",
        dialog_type,
    )
    warnings = _capture_warnings(monkeypatch)
    before_revision = window.document.session_revision

    window.create_output_request()

    created = tuple(_editable_step(window.document.steps).outputs)
    published = next(
        candidate.authoring_request
        for candidate in authoring.output_request_catalog.candidates
        if candidate.authoring_request.variables == ("S",)
    )
    assert tuple(dialog_type.observed_candidates) == (
        authoring.output_request_catalog.candidates
    )
    assert tuple(request.variables for request in created) == (
        ("U",),
        ("S",),
    )
    assert created[-1] == published
    assert window.document.session_revision == before_revision + 1
    assert not window.document.can_save
    assert warnings == [("输出请求", _RELOAD_LOSS_MESSAGE)]
    assert gui_inp_path.read_bytes() == source_bytes

    _install_imported(window, gui_inp_path)

    assert tuple(_editable_step(window.document.steps).outputs) == original
    assert gui_inp_path.read_bytes() == source_bytes
    window.close()


def test_imported_delete_checks_capability_warns_and_reload_restores_source(
    gui_inp_path,
    monkeypatch,
) -> None:
    _application()
    source_bytes = gui_inp_path.read_bytes()
    window = FEMMainWindow()
    _install_imported(window, gui_inp_path)
    original = tuple(_editable_step(window.document.steps).outputs)
    values = deepcopy(window.document.steps)
    editable = _editable_step(values)
    editable.outputs = tuple(editable.outputs[1:])

    class AcceptedManager:
        def exec(self):
            return QDialog.DialogCode.Accepted

        def values(self):
            return deepcopy(values)

    monkeypatch.setattr(
        window,
        "_analysis_manager_dialog",
        lambda: AcceptedManager(),
    )
    warnings = _capture_warnings(monkeypatch)

    window.show_analysis_manager()

    assert tuple(
        _editable_step(window.document.steps).outputs
    ) == original[1:]
    assert warnings == [("输出请求", _RELOAD_LOSS_MESSAGE)]
    assert gui_inp_path.read_bytes() == source_bytes

    _install_imported(window, gui_inp_path)

    assert tuple(_editable_step(window.document.steps).outputs) == original
    assert gui_inp_path.read_bytes() == source_bytes
    window.close()


def test_imported_existing_view_acceptance_has_zero_session_mutation(
    gui_inp_path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    _install_imported(window, gui_inp_path)
    before = window.document
    step_index = next(
        index
        for index, step in enumerate(window.document.steps)
        if step.name.strip().casefold() != "initial"
    )
    warnings = _capture_warnings(monkeypatch)
    monkeypatch.setattr(
        definition_dialogs_module.OutputRequestDialog,
        "exec",
        lambda _dialog: QDialog.DialogCode.Accepted,
    )

    window.edit_analysis_definition(
        "output",
        (step_index, 0),
    )

    assert window.document == before
    assert window.document.session_revision == before.session_revision
    assert warnings == []
    window.close()


def test_delete_rechecks_session_capability_before_mutation_or_warning(
    gui_inp_path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    _install_imported(window, gui_inp_path)
    before = window.document
    values = deepcopy(window.document.steps)
    editable = _editable_step(values)
    editable.outputs = tuple(editable.outputs[1:])

    class AcceptedManager:
        def exec(self):
            return QDialog.DialogCode.Accepted

        def values(self):
            return deepcopy(values)

    class DeniedProjection:
        @staticmethod
        def operation(name):
            assert name == "output_request.delete"
            return AuthoringCapability(
                name,
                AuthoringStatus.UNAVAILABLE,
            )

    monkeypatch.setattr(
        window,
        "_analysis_manager_dialog",
        lambda: AcceptedManager(),
    )
    monkeypatch.setattr(
        main_window_module,
        "describe_session_authoring",
        lambda _snapshot: DeniedProjection(),
    )
    errors = []
    monkeypatch.setattr(
        window,
        "_show_authoring_decision_error",
        lambda title, capability: errors.append(
            (title, capability.operation)
        ),
    )
    warnings = _capture_warnings(monkeypatch)

    window.show_analysis_manager()

    assert window.document == before
    assert warnings == []
    assert errors == [
        ("删除输出请求", "output_request.delete"),
    ]
    window.close()


def test_native_create_survives_project_save_and_reopen(
    tmp_path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    window._create_native_model("Model-1")
    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    window._analysis_definitions_changed(
        "测试分析步",
        [static("Load")],
    )
    dialog_type = _accepted_candidate_dialog(
        lambda request: request.variables == ("S",)
    )
    monkeypatch.setattr(
        main_window_module,
        "OutputRequestDialog",
        dialog_type,
    )
    warnings = _capture_warnings(monkeypatch)

    window.create_output_request()

    created = window.document.steps[0].outputs
    assert tuple(request.variables for request in created) == (
        ("U",),
        ("S",),
    )
    assert all(not request.metadata for request in created)
    assert all(request.source_evidence is None for request in created)
    assert warnings == []

    target = tmp_path / "output-request.femproj"
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    assert window.save_native_project()
    _wait_for_task(window)
    window.close()

    reopened = FEMMainWindow()
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    reopened.open_native_project()
    _wait_for_task(reopened)

    assert reopened.document.steps[0].outputs == created
    assert not reopened.document.steps[0].outputs[0].metadata
    assert reopened.document.steps[0].outputs[0].source_evidence is None
    reopened.close()


def test_native_create_accepts_multiple_output_variables(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    window._create_native_model("Model-1")
    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    window._analysis_definitions_changed(
        "测试分析步",
        [static("Load")],
    )

    class AcceptedMultipleDialog:
        def __init__(
            self,
            step_names,
            _parent=None,
            *,
            candidates=(),
            current=None,
            existing_requests_by_step=None,
        ) -> None:
            assert current is None
            assert existing_requests_by_step is not None
            self._step_name = step_names[0]
            self._requests = tuple(
                deepcopy(
                    next(
                        candidate.authoring_request
                        for candidate in candidates
                        if candidate.authoring_request.variables
                        == (variable,)
                    )
                )
                for variable in ("U", "RF", "S")
            )

        def exec(self):
            return QDialog.DialogCode.Accepted

        def definitions(self):
            return self._step_name, self._requests

    monkeypatch.setattr(
        main_window_module,
        "OutputRequestDialog",
        AcceptedMultipleDialog,
    )

    window.create_output_request()

    assert tuple(
        request.variables
        for request in window.document.steps[0].outputs
    ) == (("U",), ("RF",), ("S",))
    window.close()


def test_create_capability_rejection_has_zero_mutation(monkeypatch) -> None:
    _application()
    window = FEMMainWindow()
    before = window.document
    errors = []
    monkeypatch.setattr(
        window,
        "_show_authoring_decision_error",
        lambda title, capability: errors.append(
            (title, capability.operation)
        ),
    )
    monkeypatch.setattr(
        main_window_module,
        "OutputRequestDialog",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unavailable create must not open a dialog")
        ),
    )

    window.create_output_request()

    assert window.document == before
    assert errors == [
        ("创建输出请求", "output_request.create"),
    ]
    window.close()


def test_imported_cancelled_or_failed_create_does_not_warn_or_mutate(
    gui_inp_path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    _install_imported(window, gui_inp_path)
    before = window.document
    warnings = _capture_warnings(monkeypatch)

    class CancelledDialog:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def exec(self):
            return QDialog.DialogCode.Rejected

        def definition(self):
            raise AssertionError("cancelled dialogs cannot produce a DTO")

    monkeypatch.setattr(
        main_window_module,
        "OutputRequestDialog",
        CancelledDialog,
    )
    window.create_output_request()
    assert window.document == before
    assert warnings == []

    class FailedDialog(CancelledDialog):
        def exec(self):
            return QDialog.DialogCode.Accepted

        def definition(self):
            raise ValueError("invalid candidate")

    errors = []
    monkeypatch.setattr(
        main_window_module,
        "OutputRequestDialog",
        FailedDialog,
    )
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )
    window.create_output_request()

    assert window.document == before
    assert warnings == []
    assert errors == [
        ("创建输出请求", "invalid candidate"),
    ]
    window.close()


def test_output_collection_change_detection_includes_step_and_order() -> None:
    first = static("First")
    second = static("Second")
    outputs = (
        OutputRequest("field", "node", ("U",)),
        OutputRequest("field", "node", ("RF",)),
    )
    first.outputs = outputs
    before = [first, second]

    moved_first = static("First")
    moved_second = static("Second")
    moved_second.outputs = outputs
    assert FEMMainWindow._output_collections_changed(
        before,
        [moved_first, moved_second],
    )

    reordered = static("First")
    reordered.outputs = tuple(reversed(outputs))
    assert FEMMainWindow._output_collections_changed(
        before,
        [reordered, second],
    )


def test_main_window_output_workflow_has_no_support_or_dto_rebuild() -> None:
    source_path = Path(inspect.getsourcefile(FEMMainWindow) or "")
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    create = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
        and node.name == "create_output_request"
    )
    string_values = {
        node.value
        for node in ast.walk(create)
        if isinstance(node, ast.Constant)
        and type(node.value) is str
    }
    output_request_calls = [
        node
        for node in ast.walk(create)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OutputRequest"
    ]

    assert string_values.isdisjoint({"U", "UR", "RF", "RM", "S"})
    assert output_request_calls == []
    assert "output_request.create" in string_values
