from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from fem.io import save_result_archive
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from tests.helpers.result_archives import make_result_archive
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)

from tests.helpers.gui_result_workflows import (
    wait_for_result_idle,
    result_projection_identity,
    open_result_archive_window,
)


def test_agent_bridge_delegates_to_the_unified_result_invalidation_gate(
    monkeypatch,
) -> None:
    window = FEMMainWindow()
    calls: list[bool] = []
    monkeypatch.setattr(
        window,
        "_confirm_result_invalidation",
        lambda: calls.append(True) or False,
    )

    confirmation = (
        window.agent_authoring_bridge._result_invalidation_confirmation
    )
    assert confirmation is not None
    assert confirmation() is False
    assert calls == [True]
    window.close_model(confirm=False)
    window.close()


def test_close_event_cancel_preserves_unsaved_result_projection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = open_result_archive_window(tmp_path, "close-event-cancel")
    run_id = window.document.displayed_result_run_id
    assert run_id is not None
    window.session._result_file_states.pop(run_id, None)
    window.document = window.session.projection_snapshot(window.document)
    window._update_action_states()
    assert window.document.result_dirty
    before = result_projection_identity(window)

    class FakeMessageBox:
        class Icon:
            Warning = object()

        class ButtonRole:
            AcceptRole = object()
            DestructiveRole = object()
            RejectRole = object()

        def __init__(self, _parent) -> None:
            self._clicked = None

        def setWindowTitle(self, _title) -> None:
            pass

        def setIcon(self, _icon) -> None:
            pass

        def setText(self, _text) -> None:
            pass

        def addButton(self, _text, role):
            button = object()
            if role is self.ButtonRole.RejectRole:
                self._clicked = button
            return button

        def setDefaultButton(self, _button) -> None:
            pass

        def exec(self) -> None:
            pass

        def clickedButton(self):
            return self._clicked

    monkeypatch.setattr(main_window_module, "QMessageBox", FakeMessageBox)
    window.show()
    QApplication.instance().processEvents()
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert window.isVisible()
    assert result_projection_identity(window) == before

    window.close_model(confirm=False)
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
    window.close()
    QApplication.instance().processEvents()
    assert not window.isVisible()


def test_face_sketch_commit_cancel_preserves_pending_state(monkeypatch) -> None:
    window = FEMMainWindow()
    operation = object()
    direction = object()
    strategy = object()
    sketch = object()
    launch = SimpleNamespace(
        part_id="part-1",
        body_id="body:domain",
        session_id=window.document.session_id,
        session_revision=window.document.session_revision,
        part_revision=1,
        workplane=SimpleNamespace(
            support_face_id="face:domain",
            strategy=strategy,
        ),
    )
    current_sketch = SimpleNamespace(
        revision=3,
        external_references=(),
        external_coincidences=(),
    )
    geometry = SimpleNamespace(
        sketch=sketch,
        external_references=(),
        external_coincidences=(),
        support_face_id="face:domain",
        workplane_strategy=strategy,
        operation=operation,
        direction=direction,
        distance=2.0,
        participating_profile_ids=("profile-1",),
    )
    controller = SimpleNamespace(
        launch_snapshot=launch,
        sketch_snapshot=lambda: current_sketch,
        launch_is_current=lambda _document: True,
        draft=SimpleNamespace(to_sketch_geometry=lambda: sketch),
    )
    dialog = SimpleNamespace(preview_is_valid=True)
    result = SimpleNamespace(geometry=geometry)
    parameters = SimpleNamespace(
        operation=operation,
        direction=direction,
        distance=2.0,
        participating_profile_ids=("profile-1",),
    )

    class FakeRequest:
        def __init__(self, launch, geometry, sketch_revision, preview_generation):
            self.launch = launch
            self.geometry = geometry
            self.sketch_revision = sketch_revision
            self.preview_generation = preview_generation

    payload = FakeRequest(launch, geometry, current_sketch.revision, 4)
    monkeypatch.setattr(main_window_module, "FaceSketchBooleanFeatureRequest", FakeRequest)
    confirmation_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        window,
        "_confirm_result_invalidation",
        lambda **kwargs: confirmation_calls.append(kwargs) or False,
    )
    monkeypatch.setattr(
        window.session,
        "commit_face_sketch_boolean",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cancelled face sketch commit must not reach Session")
        ),
    )
    window._face_sketch_controller = controller
    window._face_sketch_preview_result = result
    window._face_sketch_dialog = dialog
    window._face_sketch_parameters = parameters
    window._face_sketch_preview_generation = 4

    window._commit_face_sketch_boolean_feature(payload)

    assert confirmation_calls == [{"preserve_editor": True}]
    assert window._face_sketch_controller is controller
    assert window._face_sketch_preview_result is result
    assert window._face_sketch_dialog is dialog
    assert window._face_sketch_parameters is parameters
    assert window.session.session_revision == 0
    window.close()


def test_public_edit_type_validation_precedes_result_confirmation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = open_result_archive_window(tmp_path, "invalid-edit-types")

    def unexpected_confirmation(**_kwargs):
        raise AssertionError("invalid command types must not open a modal")

    monkeypatch.setattr(window, "_confirm_result_invalidation", unexpected_confirmation)
    for command, expected_code in (
        (window.apply_native_geometry_edit, "command.type.invalid"),
        (window.apply_mesh_input_edit, "command.type.invalid"),
        (window.apply_named_region_edit, "command.type.invalid"),
    ):
        receipt = command(object())
        assert receipt.diagnostic is not None
        assert receipt.diagnostic.code == expected_code

    window.close_model(confirm=False)
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
    window.close()


def test_result_transition_confirmation_exposes_unsaved_run(monkeypatch, tmp_path: Path) -> None:
    window = open_result_archive_window(tmp_path, "confirm")
    # Remove the accepted file state to model a genuinely unsaved result run;
    # the count and job label must come from the live Session projection.
    run_id = window.document.displayed_result_run_id
    assert run_id is not None
    run = window.session.find_run(run_id)
    assert run is not None and run.name == "job"
    window.session._result_file_states.pop(run_id, None)
    window.document = window.session.projection_snapshot(window.document)
    window._update_action_states()
    assert window.document.unsaved_result_count == 1
    before = result_projection_identity(window)
    captured: list[str] = []
    button_texts: list[str] = []

    class FakeMessageBox:
        class Icon:
            Warning = object()

        class ButtonRole:
            AcceptRole = object()
            DestructiveRole = object()
            RejectRole = object()

        def __init__(self, _parent) -> None:
            self._clicked = None

        def setWindowTitle(self, _title) -> None:
            pass

        def setIcon(self, _icon) -> None:
            pass

        def setText(self, text) -> None:
            captured.append(text)

        def addButton(self, text, role):
            button = object()
            button_texts.append(str(text))
            if role is self.ButtonRole.DestructiveRole:
                # Simulate the user choosing Cancel, not Discard.
                self._discard = button
            elif role is self.ButtonRole.RejectRole:
                self._clicked = button
            return button

        def setDefaultButton(self, _button) -> None:
            pass

        def exec(self) -> None:
            pass

        def clickedButton(self):
            return self._clicked

    monkeypatch.setattr(main_window_module, "QMessageBox", FakeMessageBox)
    assert not window._confirm_document_transition()
    assert captured and "1" in captured[-1]
    assert "job" in captured[-1]
    assert "保存" not in button_texts
    after = result_projection_identity(window)
    assert after == before
    wait_for_result_idle(window)
    window.close_model(confirm=False)
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
    monkeypatch.setattr(window, "_confirm_workspace_context_close", lambda *_args: True)
    window.close()


def test_opening_documents_preserves_dirty_result_and_cancelled_close_keeps_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = open_result_archive_window(tmp_path, "transition-cancel")
    run_id = window.document.displayed_result_run_id
    assert run_id is not None
    window.session._result_file_states.pop(run_id, None)
    window.document = window.session.projection_snapshot(window.document)
    window._update_action_states()
    assert window.document.result_dirty
    dirty_context_id = window.workspace.active_document_id
    assert dirty_context_id is not None
    target = tmp_path / "other.femres"
    save_result_archive(target, make_result_archive(make_continuum_nodal_semantics_result, "other"))
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    monkeypatch.setattr(window, "_confirm_document_transition", lambda **_kwargs: False)
    window.open_result_file()
    wait_for_result_idle(window)
    assert window.workspace.document_count == 3
    dirty_context = window.workspace.document(dirty_context_id)
    assert dirty_context.projection.result_dirty
    assert window.workspace.active_document_id != dirty_context_id

    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: False)
    monkeypatch.setattr(
        window,
        "_confirm_workspace_context_close",
        lambda *_args: False,
    )
    monkeypatch.setattr(window, "_show_error", lambda *_args: None)
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: ("replacement", True),
    )
    window.new_native_model()
    assert window.workspace.document_count == 4
    assert window.workspace.document(dirty_context_id).projection.result_dirty
    assert not window.close_model(confirm=True, document_id=dirty_context_id)
    assert window.workspace.document(dirty_context_id).projection.result_dirty
    wait_for_result_idle(window)
    window.close_model(confirm=False)
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
    monkeypatch.setattr(
        window,
        "_confirm_workspace_context_close",
        lambda *_args: True,
    )
    window.close()
