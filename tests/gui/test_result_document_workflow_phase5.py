from __future__ import annotations

import os
import threading
import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QDialog, QMenu, QToolButton

from fem.application import ModelSession, NamedRegion, describe_session_authoring
from fem.geometry import LogicalEntityRef
from fem.io import save_project, save_result_archive
from fem.core.model import DisplacementConstraint, NodalLoad
from fem.application.results import ResultArchiveModelProjection
from fem.application.results import (
    FieldState,
    ResultQuery,
    ScalarFieldSelection,
)
from fem_gui.action_state import ACTION_DESCRIPTORS, GuiActionKey
from fem_gui.action_state import GuiActionContext, derive_action_availability
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from fem_gui.commands import MeshInputEdit
from fem_gui.inspection_service import InspectionService
from fem_gui.visualization.model_adapter import build_result_archive_model_view
from fem_gui.viewport_image_export_dialog import ViewportImageExportOptions
from fem_gui.task_controller import BackgroundTaskState
from tests.io.test_result_archive_v1 import _snapshot
from tests.helpers.phase8_result_characterization import (
    make_beam_field_characterization_result,
    make_continuum_nodal_semantics_result,
    make_truss_field_characterization_result,
)
from tests.gui.test_project_io import _native_project_snapshot
from fem.mesh.settings import MeshSettings


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_idle(window: FEMMainWindow, timeout_ms: int = 2000) -> None:
    app = _application()
    deadline = __import__("time").monotonic() + timeout_ms / 1000.0
    open_controller = window.workspace.open_controller
    while (
        (
            window.busy
            or (
                open_controller is not None
                and open_controller.busy
            )
        )
        and __import__("time").monotonic() < deadline
    ):
        app.processEvents()
        __import__("time").sleep(0.001)
    app.processEvents()
    if window.busy or (
        open_controller is not None and open_controller.busy
    ):
        print(
            "PHASE5 busy timeout",
            window.task_controller.state,
            window.task_controller.current_task_name,
            None
            if open_controller is None
            else open_controller.state,
            flush=True,
        )
        raise AssertionError(
            f"background task did not settle: {window.task_controller.state!r} "
            f"{window.task_controller.current_task_name!r}"
        )


def _projection_identity(window: FEMMainWindow) -> tuple[object, ...]:
    document = window.document
    catalog = window.result_tree.catalog
    payload = window.viewport._result_render_payload
    file_state = document.result_file_state
    return (
        document.session_id,
        document.session_revision,
        document.source_kind,
        document.source_path,
        id(catalog),
        None if catalog is None else catalog.source,
        window.viewport.artifact_id,
        id(payload),
        None
        if payload is None
        else (
            payload.topology.source,
            payload.topology.materialization_generation,
            payload.topology.selection,
        ),
        document.result_path,
        document.result_dirty,
        document.unsaved_result_count,
        file_state,
    )


def _open_archive_window(tmp_path: Path, run_name: str = "phase5") -> FEMMainWindow:
    archive = _snapshot(make_continuum_nodal_semantics_result, run_name)
    source = tmp_path / f"{run_name}.femres"
    save_result_archive(source, archive)
    window = FEMMainWindow()
    receipt = window.open_result_path(source)
    assert receipt.completion is not None
    assert receipt.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    _wait_idle(window)
    return window


def test_result_actions_have_canonical_descriptors_and_visible_layout(tmp_path: Path) -> None:
    _application()
    window = FEMMainWindow()
    descriptors = {item.key: item for item in ACTION_DESCRIPTORS}
    assert descriptors[GuiActionKey.SAVE_RESULT].handler == "save_current_result"
    assert descriptors[GuiActionKey.SAVE_RESULT].icon_name == "save_result"
    assert descriptors[GuiActionKey.SAVE_RESULT_AS].handler == "save_current_result_as"
    assert descriptors[GuiActionKey.SAVE_RESULT_AS].icon_name is None
    assert descriptors[GuiActionKey.OPEN_RESULT].handler == "open_result_file"
    assert descriptors[GuiActionKey.OPEN_RESULT].icon_name == "open_result"
    file_menu = window.findChild(QMenu, "menuFile")
    assert file_menu is not None
    file_actions = [item.objectName() for item in file_menu.actions()]
    assert file_actions == [
        "action_new_native",
        "action_open_project",
        "action_save_project",
        "action_save_project_as",
        "action_open",
        "action_save_result",
        "action_save_result_as",
        "action_open_result",
        "",
        "action_exit",
    ]
    project_index = next(
        index
        for index in range(window.ribbon.tab_bar.count())
        if window.ribbon.tab_bar.tabText(index) == "项目"
    )
    project_buttons = [
        button.defaultAction().objectName()
        for button in window.ribbon.stack.widget(project_index).findChildren(
            QToolButton
        )
        if button.defaultAction() is not None
    ]
    assert project_buttons[:9] == [
        "action_new_native",
        "action_open_project",
        "action_save_project",
        "action_save_project_as",
        "action_open",
        "action_save_result",
        "action_save_result_as",
        "action_open_result",
        "action_model_info",
    ]
    _wait_idle(window)
    window.close()


def test_open_result_path_installs_read_only_document_and_result_module(tmp_path: Path) -> None:
    _application()
    archive = _snapshot(make_continuum_nodal_semantics_result, "gui")
    path = tmp_path / "display.femres"
    save_result_archive(path, archive)
    window = FEMMainWindow()
    receipt = window.open_result_path(path)
    assert receipt.completion is not None
    terminal = receipt.completion.result(2.0)
    assert terminal.state.value == "succeeded"
    _wait_idle(window)
    assert window.document.result_only
    assert window.document.path == path
    assert "[结果只读]" in window.windowTitle()
    assert window.ribbon.tab_bar.tabText(window.ribbon.tab_bar.currentIndex()) == "结果"
    assert window.navigation.tabs.currentWidget() is window.result_tree
    assert window.result_provider is not None and window.result_provider.is_archived
    assert window.result_tree.catalog is not None
    assert window.inspection_service is not None
    assert isinstance(window.document.model_projection, ResultArchiveModelProjection)
    assert window.viewport.artifact_id == window.document.artifact.artifact_id
    assert window.agent_authoring_bridge.context is None
    assert window.agent_authoring_bridge.port._context is None
    assert not window.agent_authoring_controller.turn_snapshot.available
    assert window.agent_authoring_controller._binding_identity is None
    assert window.actions["save_result"].isEnabled()
    assert window.actions["open_result"].isEnabled()
    assert window.actions["close"].isEnabled()
    assert not window.actions["reload"].isEnabled()
    assert window.actions["screenshot"].toolTip() == "导出视口"
    assert not window.actions["save_project"].isEnabled()
    assert not window.actions["submit_job"].isEnabled()
    _wait_idle(window)
    window.close()


def test_result_archive_switches_between_result_and_mesh_modules(
    tmp_path: Path,
) -> None:
    window = _open_archive_window(tmp_path, "module-switch")
    artifact_id = window.document.artifact.artifact_id

    assert window._current_module_name() == "结果"
    assert window.navigation.tabs.currentWidget() is window.result_tree
    assert window.viewport._result_render_payload is not None

    for module_name in ("模型", "网格"):
        window.ribbon.set_current(module_name)
        assert window._current_module_name() == module_name
        assert window.navigation.tabs.currentWidget() is window.model_tree
        assert window.viewport._model is window._result_archive_model_view
        assert window.viewport._geometry is window.geometry
        assert window.viewport._result_render_payload is None
        assert window.viewport.artifact_id == artifact_id

    window.ribbon.set_current("结果")
    assert window.navigation.tabs.currentWidget() is window.result_tree
    assert window.viewport._result_render_payload is not None
    assert window.viewport.artifact_id == artifact_id
    _wait_idle(window)
    window.close()


def test_open_result_reprojects_when_result_module_is_already_current(
    tmp_path: Path,
) -> None:
    archive = _snapshot(make_continuum_nodal_semantics_result, "already-result")
    path = tmp_path / "already-result.femres"
    save_result_archive(path, archive)
    window = FEMMainWindow()
    window._set_selection_filter("face")
    window.ribbon.set_current("结果")

    receipt = window.open_result_path(path)
    assert receipt.completion is not None
    assert receipt.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    _wait_idle(window)

    assert window._current_module_name() == "结果"
    assert window.navigation.tabs.currentWidget() is window.result_tree
    assert window.viewport._result_render_payload is not None
    assert window.viewport._model is window._result_archive_model_view
    window.close()


def test_result_only_action_state_gates_all_mutating_commands(tmp_path: Path) -> None:
    window = _open_archive_window(tmp_path, "readonlyactions")
    disabled = {
        "save_project",
        "material_manager",
        "section_manager",
        "section_assign",
        "geometry_create",
        "geometry_sketch",
        "geometry_face_sketch",
        "geometry_wire",
        "geometry_move",
        "geometry_rotate",
        "geometry_extrude",
        "geometry_sweep",
        "geometry_fuse",
        "geometry_cut",
        "geometry_manager",
        "geometry_undo",
        "geometry_delete",
        "geometry_region",
        "geometry_regions",
        "mesh_settings",
        "mesh_generate",
        "mesh_clear",
        "mesh_controls",
        "mesh_local_control",
        "step_create",
        "boundary_create",
        "load_create",
        "output_create",
        "analysis_manager",
        "check_model",
        "submit_job",
        "resubmit_job",
        "job_manager",
    }
    assert disabled <= set(window.actions)
    for name in disabled:
        assert not window.actions[name].isEnabled(), name
        assert window.actions[name].toolTip()
    for name in (
        "query",
        "export_csv",
        "export_vtk",
        "mesh_statistics",
        "mesh_quality",
        "mesh_verify",
    ):
        assert window.actions[name].isEnabled(), name
    if window.viewport.can_capture and window.viewport.backend_available:
        assert window.actions["screenshot"].isEnabled()
    else:
        assert not window.actions["screenshot"].isEnabled()
    _wait_idle(window)
    window.close()


def test_result_only_query_inspection_exports_and_default_viewport_projection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = _open_archive_window(tmp_path, "result-consumers")
    provider = window.result_provider
    assert provider is not None and provider.is_archived
    selection = window.result_selection
    assert type(selection) is ScalarFieldSelection
    payload = window.viewport._result_render_payload
    assert payload is not None
    assert payload.topology.cells
    assert window.document.model_projection is not None
    assert window.viewport._run_id == provider.source.run_id
    if window.viewport._plotter is not None:
        assert "result" in window.viewport._actors

    ready = next(
        item
        for item in provider.catalog().fields
        if item.state is FieldState.READY
    )
    query = ResultQuery(ready.key, ready.descriptor.columns[0])
    queried = window.query_result(query)
    assert queried.outcome is not None
    assert queried.outcome.record_count is not None
    assert window.inspection_service is not None
    inspection = window.inspection_service.inspect("model", None)
    assert inspection.title
    assert inspection.pages

    csv_path = tmp_path / "result-consumers.csv"

    def accept_csv(dialog) -> QDialog.DialogCode:
        dialog.path_edit.setText(str(csv_path))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(window, "_exec_dialog", accept_csv)
    monkeypatch.setattr(window, "_show_save_success", lambda *_args, **_kwargs: None)
    window.export_csv()
    _wait_idle(window)
    assert csv_path.is_file()
    assert csv_path.read_text(encoding="utf-8").strip()

    vtk_path = tmp_path / "result-consumers.vtk"
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(vtk_path), ""),
    )
    window.export_vtk()
    _wait_idle(window)
    assert vtk_path.is_file()

    image_path = tmp_path / "result-consumers.png"
    screenshot_calls: list[tuple[object, ...]] = []

    class AcceptedImageDialog:
        options = ViewportImageExportOptions(
            1,
            None,
            False,
        )
        target_path = image_path

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Accepted

    def save_screenshot(path, **kwargs) -> None:
        screenshot_calls.append((path, kwargs))
        Path(path).write_bytes(b"fake-png")

    monkeypatch.setattr(
        main_window_module,
        "ViewportImageExportDialog",
        AcceptedImageDialog,
    )
    monkeypatch.setattr(window.viewport, "save_screenshot", save_screenshot)
    window.export_viewport_image()
    assert screenshot_calls and image_path.is_file()
    _wait_idle(window)
    window.close()


def test_save_result_path_completes_suffix_and_updates_session_state(tmp_path: Path) -> None:
    _application()
    archive = _snapshot(make_continuum_nodal_semantics_result, "save")
    source = tmp_path / "source.femres"
    save_result_archive(source, archive)
    window = FEMMainWindow()
    opened = window.open_result_path(source)
    assert opened.completion is not None
    opened.completion.result(2.0)
    _wait_idle(window)
    target = tmp_path / "copy-without-suffix"
    saved = window.save_result_path(target)
    assert saved.completion is not None
    assert saved.completion.result(2.0).state.value == "succeeded"
    result_path = target.with_suffix(".femres")
    assert result_path.exists()
    assert window.document.result_path == result_path
    assert window.document.unsaved_result_count == 0
    _wait_idle(window)
    window.close()


def test_result_dialog_handlers_route_to_archive_workers(tmp_path: Path, monkeypatch) -> None:
    _application()
    archive = _snapshot(make_continuum_nodal_semantics_result, "dialog")
    source = tmp_path / "dialog-source.femres"
    save_result_archive(source, archive)
    target = tmp_path / "dialog-copy"
    window = FEMMainWindow()
    open_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (
            open_calls.append((*args, kwargs)) or (str(source), "")
        ),
    )
    monkeypatch.setattr(window, "_confirm_document_transition", lambda: True)
    window.open_result_file()
    _wait_idle(window)
    assert window.document.result_only
    assert open_calls and open_calls[-1][3] == (
        "FEM-Python 结果 (*.femres);;所有文件 (*)"
    )

    save_calls: list[tuple[object, ...]] = []

    def choose_save_target(*args, **kwargs):
        save_calls.append((*args, kwargs))
        return str(target), ""

    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        choose_save_target,
    )
    assert window.save_current_result(wait=True)
    assert save_calls == []
    assert window.save_current_result_as(wait=True)
    assert window.save_current_result(wait=True)
    assert target.with_suffix(".femres").is_file()
    assert len(save_calls) == 1
    assert save_calls[0][2] == "dialog-source.femres"
    assert all(call[3] == "FEM-Python 结果 (*.femres)" for call in save_calls)
    _wait_idle(window)
    window.close()


def test_result_open_builds_archive_display_payload_off_gui_thread(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _application()
    archive = _snapshot(make_continuum_nodal_semantics_result, "thread")
    source = tmp_path / "thread-source.femres"
    save_result_archive(source, archive)
    gui_thread = threading.get_ident()
    calls: list[tuple[str, int]] = []
    original_load = main_window_module.load_result_archive
    original_geometry = main_window_module.build_result_archive_geometry
    original_model_view = main_window_module.build_result_archive_model_view

    def load(path):
        calls.append(("load", threading.get_ident()))
        return original_load(path)

    def geometry(projection):
        calls.append(("geometry", threading.get_ident()))
        return original_geometry(projection)

    def model_view(projection, profile, *, name):
        calls.append(("model_view", threading.get_ident()))
        return original_model_view(projection, profile, name=name)

    monkeypatch.setattr(main_window_module, "load_result_archive", load)
    monkeypatch.setattr(
        main_window_module,
        "build_result_archive_geometry",
        geometry,
    )
    monkeypatch.setattr(
        main_window_module,
        "build_result_archive_model_view",
        model_view,
    )
    window = FEMMainWindow()
    receipt = window.open_result_path(source)
    assert receipt.completion is not None
    assert receipt.completion.result(2.0).state.value == "succeeded"
    assert {name for name, _thread in calls} == {"load", "geometry", "model_view"}
    assert all(thread_id != gui_thread for _name, thread_id in calls)
    _wait_idle(window)
    window.close()


def test_result_archive_model_view_uses_profile_dimension_and_dofs() -> None:
    cases = (
        (make_continuum_nodal_semantics_result, 2, 2),
        (make_truss_field_characterization_result, 3, 3),
        (make_beam_field_characterization_result, 3, 6),
    )
    for builder, spatial_dimension, dofs_per_node in cases:
        archive = _snapshot(builder, builder.__name__)
        view = build_result_archive_model_view(
            archive.model_projection,
            archive.profile,
        )
        assert view.mesh.spatial_dimension == spatial_dimension
        assert view.mesh.dofs_per_node == dofs_per_node
        assert view.mesh.num_dofs == len(archive.topology.node_ids) * dofs_per_node


def test_result_archive_view_keeps_only_result_topology_and_regions() -> None:
    archive = _snapshot(make_continuum_nodal_semantics_result, "summaries")
    element_id = archive.topology.element_ids[0]
    projection = replace(
        archive.model_projection,
        named_region_element_ids={
            "REGION-A": (element_id,),
            "REGION-B": (element_id,),
        },
        summaries={
            "materials": ({"name": "STEEL", "properties": {"E": 2.1e11}},),
            "sections": (
                {
                    "name": "SOLID",
                    "material": "STEEL",
                    "section_type": "solid",
                    "properties": {},
                },
            ),
            "assignments": (
                {"section_name": "SOLID", "region_name": "REGION-A"},
                {"section_name": "SOLID", "region_name": "REGION-B"},
            ),
            "steps": (
                {
                    "name": "Step-1",
                    "procedure": "static",
                    "boundary_count": 2,
                    "load_count": 1,
                    "surface_load_count": 1,
                    "total_load_count": 4,
                    "output_count": 3,
                },
            ),
        },
    )
    view = build_result_archive_model_view(projection, archive.profile)
    service = InspectionService(view)

    assert set(view.element_sets) == {"REGION-A", "REGION-B"}
    assert view.materials == {}
    assert view.sections == ()
    assert view.steps == ()
    assert view.metadata == {}
    model_fields = dict(service.inspect("model", None).pages[0].fields)
    assert model_fields["空间维度"] == "2维"
    assert model_fields["总自由度数量"] == str(
        len(archive.topology.node_ids) * archive.profile.dofs_per_node
    )
    assert model_fields["分析步数量"] == "0"
    node_fields = dict(
        service.inspect("node", archive.topology.node_ids[0]).pages[0].fields
    )
    assert len(node_fields["坐标"].split(",")) == 2


def test_result_transition_confirmation_exposes_unsaved_run(monkeypatch, tmp_path: Path) -> None:
    _application()
    window = _open_archive_window(tmp_path, "confirm")
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
    before = _projection_identity(window)
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
    after = _projection_identity(window)
    assert after == before
    _wait_idle(window)
    window.close_model(confirm=False)
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
    monkeypatch.setattr(window, "_confirm_workspace_context_close", lambda *_args: True)
    window.close()


def test_transition_cancel_blocks_open_new_close_and_result_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = _open_archive_window(tmp_path, "transition-cancel")
    run_id = window.document.displayed_result_run_id
    assert run_id is not None
    window.session._result_file_states.pop(run_id, None)
    window.document = window.session.projection_snapshot(window.document)
    window._update_action_states()
    assert window.document.result_dirty
    dirty_context_id = window.workspace.active_document_id
    assert dirty_context_id is not None
    target = tmp_path / "other.femres"
    save_result_archive(target, _snapshot(make_continuum_nodal_semantics_result, "other"))
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    monkeypatch.setattr(window, "_confirm_document_transition", lambda **_kwargs: False)
    window.open_result_file()
    _wait_idle(window)
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
    _wait_idle(window)
    window.close_model(confirm=False)
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
    monkeypatch.setattr(
        window,
        "_confirm_workspace_context_close",
        lambda *_args: True,
    )
    window.close()


def test_close_event_cancel_preserves_unsaved_result_projection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = _open_archive_window(tmp_path, "close-event-cancel")
    run_id = window.document.displayed_result_run_id
    assert run_id is not None
    window.session._result_file_states.pop(run_id, None)
    window.document = window.session.projection_snapshot(window.document)
    window._update_action_states()
    assert window.document.result_dirty
    before = _projection_identity(window)

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
    _application().processEvents()
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert window.isVisible()
    assert _projection_identity(window) == before

    window.close_model(confirm=False)
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
    window.close()
    _application().processEvents()
    assert not window.isVisible()


def test_native_mutation_callpoints_share_result_invalidation_gate() -> None:
    for method_name in (
        "finish_wire_geometry",
        "finish_sketch_geometry",
        "extrude_geometry",
        "sweep_geometry",
        "finish_body_boolean",
        "show_geometry_manager",
        "undo_geometry_feature",
        "delete_geometry",
        "_set_native_geometry",
        "_commit_face_sketch_boolean_feature",
    ):
        source = inspect.getsource(getattr(FEMMainWindow, method_name))
        assert "_confirm_result_invalidation" in source, method_name


def test_agent_bridge_delegates_to_the_unified_result_invalidation_gate(
    monkeypatch,
) -> None:
    _application()
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


def test_face_sketch_commit_cancel_preserves_pending_state(monkeypatch) -> None:
    _application()
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
    window = _open_archive_window(tmp_path, "invalid-edit-types")

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


def test_real_fempy_project_vertical_remains_openable(
    tmp_path: Path,
    monkeypatch,
    dispose_gui_widget,
) -> None:
    _application()
    monkeypatch.setattr(FEMMainWindow, "_show_error", lambda *_args, **_kwargs: None)
    # Keep the native route end-to-end while using a coarse, fully constrained
    # sketch so this focused test does not become a mesh/solver benchmark.
    project = _native_project_snapshot()
    base_step = project.analysis_definitions[0]
    step = replace(
        base_step,
        boundaries=(DisplacementConstraint("FIXED", 1, 2, 0.0),),
        cloads=(NodalLoad("FIXED", 1, 100.0),),
        edge_loads=(),
        body_loads=(),
        gravity_loads=(),
    )
    project = replace(
        project,
        parts=(),
        mesh_settings=MeshSettings(25.0, order=1, cell_shape="triangle"),
        named_regions=(
            NamedRegion("DOMAIN_SET", (LogicalEntityRef("face:domain"),)),
            NamedRegion("FIXED", (LogicalEntityRef("edge:outer-loop"),)),
        ),
        region_assignments=(replace(project.region_assignments[0], region_name="DOMAIN_SET"),),
        analysis_definitions=(step,),
    )
    source = save_project(tmp_path / "vertical.fempy", project)
    window = FEMMainWindow()
    receipt = window.open_project_path(source)
    assert receipt.completion is not None
    assert receipt.completion.result(2.0).state.value == "succeeded"
    _wait_idle(window)
    assert window.document.source_kind == "native"
    assert window.document.project_path == source
    assert not window.legacy_project_extension
    assert window.generate_mesh()
    _wait_idle(window)
    assert window.check_current_model(show_success=False)
    assert window._submit_job("Vertical-Job", "Load") is not None
    _wait_idle(window)
    assert window.document.has_result
    result_target = tmp_path / "vertical-result"
    saved = window.save_result_path(result_target)
    assert saved.completion is not None
    assert saved.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    _wait_idle(window)
    result_path = result_target.with_suffix(".femres")
    assert window.close_model(confirm=False)
    _wait_idle(window)
    dispose_gui_widget(window)
    reopened = FEMMainWindow()
    opened = reopened.open_result_path(result_path)
    assert opened.completion is not None
    assert opened.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    _wait_idle(reopened)
    assert reopened.document.result_only
    reopened.close()


def test_result_action_reasons_are_typed_for_busy_and_no_result() -> None:
    snapshot = ModelSession().snapshot()
    authoring = describe_session_authoring(snapshot)
    idle = {
        item.key: item
        for item in derive_action_availability(
            snapshot,
            authoring,
            GuiActionContext(),
        )
    }
    busy = {
        item.key: item
        for item in derive_action_availability(
            snapshot,
            authoring,
            GuiActionContext(busy=True),
        )
    }
    assert not idle[GuiActionKey.SAVE_RESULT].enabled
    assert "成功结果" in idle[GuiActionKey.SAVE_RESULT].reason
    assert not idle[GuiActionKey.SAVE_RESULT_AS].enabled
    assert idle[GuiActionKey.SAVE_RESULT_AS].reason == idle[GuiActionKey.SAVE_RESULT].reason
    assert not busy[GuiActionKey.SAVE_RESULT].enabled
    assert "后台任务" in busy[GuiActionKey.SAVE_RESULT].reason
    assert not busy[GuiActionKey.SAVE_RESULT_AS].enabled
    assert busy[GuiActionKey.SAVE_RESULT_AS].reason == busy[GuiActionKey.SAVE_RESULT].reason
    assert not busy[GuiActionKey.OPEN_RESULT].enabled
    assert "后台任务" in busy[GuiActionKey.OPEN_RESULT].reason


def test_result_dialog_cancel_keeps_document_and_advertises_femres_filter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    open_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (
            open_calls.append((*args, kwargs)) or ("", "")
        ),
    )
    window.open_result_file()
    assert not window.busy
    assert window.document.source_kind is None
    assert open_calls and open_calls[0][3] == (
        "FEM-Python 结果 (*.femres);;所有文件 (*)"
    )
    _wait_idle(window)
    window.close()


def test_save_result_as_dialog_cancel_does_not_start_a_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _application()
    archive = _snapshot(make_continuum_nodal_semantics_result, "save-cancel")
    source = tmp_path / "save-cancel-source.femres"
    save_result_archive(source, archive)
    window = FEMMainWindow()
    window.open_result_path(source).completion.result(2.0)
    _wait_idle(window)
    before = _projection_identity(window)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (
            calls.append((*args, kwargs)) or ("", "")
        ),
    )
    assert not window.save_current_result_as()
    assert not window.busy
    assert calls and calls[0][3] == "FEM-Python 结果 (*.femres)"
    assert _projection_identity(window) == before
    _wait_idle(window)
    window.close()


def test_save_result_worker_runs_off_gui_thread(tmp_path: Path, monkeypatch) -> None:
    _application()
    archive = _snapshot(make_continuum_nodal_semantics_result, "save-thread")
    source = tmp_path / "save-thread-source.femres"
    save_result_archive(source, archive)
    window = FEMMainWindow()
    window.open_result_path(source).completion.result(2.0)
    _wait_idle(window)
    gui_thread = threading.get_ident()
    worker_threads: list[int] = []
    original_save = main_window_module.save_result_archive

    def save(path, snapshot, *, checkpoint=None):
        worker_threads.append(threading.get_ident())
        return original_save(path, snapshot, checkpoint=checkpoint)

    monkeypatch.setattr(main_window_module, "save_result_archive", save)
    receipt = window.save_result_path(tmp_path / "save-thread-copy")
    assert receipt.completion is not None
    assert receipt.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    assert worker_threads and worker_threads[0] != gui_thread
    _wait_idle(window)
    window.close()


def test_save_result_worker_failure_cleans_current_save_token(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _application()
    archive = _snapshot(make_continuum_nodal_semantics_result, "save-failure")
    source = tmp_path / "save-failure-source.femres"
    save_result_archive(source, archive)
    window = FEMMainWindow()
    window.open_result_path(source).completion.result(2.0)
    _wait_idle(window)
    before = _projection_identity(window)
    monkeypatch.setattr(window, "_show_error", lambda *_args, **_kwargs: None)

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(main_window_module, "save_result_archive", fail)
    receipt = window.save_result_path(tmp_path / "save-failure-copy")
    assert receipt.completion is not None
    assert receipt.completion.result(2.0).state is BackgroundTaskState.FAILED
    _wait_idle(window)
    assert not window.busy
    assert not window.session._active_result_save_tasks
    assert _projection_identity(window) == before
    _wait_idle(window)
    window.close()


def test_save_result_start_rejection_cleans_issued_save_token(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _application()
    archive = _snapshot(make_continuum_nodal_semantics_result, "save-start")
    source = tmp_path / "save-start-source.femres"
    save_result_archive(source, archive)
    window = FEMMainWindow()
    window.open_result_path(source).completion.result(2.0)
    _wait_idle(window)
    before = _projection_identity(window)
    monkeypatch.setattr(window, "_start_task", lambda *_args, **_kwargs: False)
    receipt = window.save_result_path(tmp_path / "save-start-copy")
    assert receipt.diagnostic is not None
    assert receipt.diagnostic.code == "task.start.rejected"
    assert not window.session._active_result_save_tasks
    assert not window.session._task_data
    assert _projection_identity(window) == before
    _wait_idle(window)
    window.close()


def test_open_result_decode_failure_preserves_current_document(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    window._apply_session_delta(window.session.new_native_project())
    before = _projection_identity(window)

    def fail(_path):
        raise ValueError("invalid archive")

    monkeypatch.setattr(main_window_module, "load_result_archive", fail)
    monkeypatch.setattr(window, "_show_error", lambda *_args, **_kwargs: None)
    receipt = window.open_result_path(tmp_path / "broken.femres")
    assert receipt.completion is not None
    assert receipt.completion.result(2.0).state is BackgroundTaskState.FAILED
    _wait_idle(window)
    assert _projection_identity(window) == before
    assert not window.document.result_only
    _wait_idle(window)
    window.close()


def test_result_save_overwrites_existing_archive_atomically(tmp_path: Path) -> None:
    _application()
    archive = _snapshot(make_continuum_nodal_semantics_result, "overwrite")
    source = tmp_path / "overwrite-source.femres"
    target = tmp_path / "overwrite-target.femres"
    save_result_archive(source, archive)
    target.write_bytes(b"stale bytes")
    window = FEMMainWindow()
    window.open_result_path(source).completion.result(2.0)
    _wait_idle(window)
    receipt = window.save_result_path(target)
    assert receipt.completion is not None
    assert receipt.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    assert target.read_bytes() != b"stale bytes"
    assert window.document.result_path == target
    _wait_idle(window)
    window.close()


def test_real_solve_save_close_and_reopen_result_roundtrip(
    gui_inp_path: Path,
    tmp_path: Path,
    dispose_gui_widget,
) -> None:
    _application()
    window = FEMMainWindow()
    window._load_path(gui_inp_path)
    _wait_idle(window)
    assert window.check_current_model(show_success=False)
    assert window._submit_job("Phase5-Job", "Static-1") is not None
    _wait_idle(window)
    assert window.document.has_result
    target = tmp_path / "phase5-roundtrip"
    saved = window.save_result_path(target)
    assert saved.completion is not None
    assert saved.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    _wait_idle(window)
    result_path = target.with_suffix(".femres")
    assert result_path.is_file()
    assert window.close_model(confirm=False)
    dispose_gui_widget(window)

    reopened = FEMMainWindow()
    opened = reopened.open_result_path(result_path)
    assert opened.completion is not None
    assert opened.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    _wait_idle(reopened)
    assert reopened.document.result_only
    assert reopened.result_provider is not None
    assert reopened.document.result_path == result_path
    reopened.close()
