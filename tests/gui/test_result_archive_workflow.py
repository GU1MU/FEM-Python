from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest

from PySide6.QtWidgets import QDialog, QMenu, QToolButton

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
from fem_gui.inspection_service import InspectionService
from fem_gui.visualization.model_adapter import build_result_archive_model_view
from fem_gui.viewport_image_export_dialog import ViewportImageExportOptions
from fem_gui.task_controller import BackgroundTaskState
from tests.helpers.result_archives import make_result_archive
from tests.helpers.phase8_result_characterization import (
    make_beam_field_characterization_result,
    make_continuum_nodal_semantics_result,
    make_truss_field_characterization_result,
)
from tests.helpers.gui_projects import make_native_project_snapshot
from fem.mesh.settings import MeshSettings

from tests.helpers.gui_result_workflows import (
    wait_for_result_idle,
    result_projection_identity,
    open_result_archive_window,
)


def test_result_actions_have_canonical_descriptors_and_visible_layout(tmp_path: Path) -> None:
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
    assert project_buttons[:8] == [
        "action_new_native",
        "action_delete_model",
        "action_open_project",
        "action_save_project",
        "action_open",
        "action_save_result",
        "action_open_result",
        "action_model_info",
    ]
    wait_for_result_idle(window)
    window.close()


def test_open_result_path_installs_read_only_document_and_result_module(tmp_path: Path) -> None:
    archive = make_result_archive(make_continuum_nodal_semantics_result, "gui")
    path = tmp_path / "display.femres"
    save_result_archive(path, archive)
    window = FEMMainWindow()
    receipt = window.open_result_path(path)
    assert receipt.completion is not None
    terminal = receipt.completion.result(2.0)
    assert terminal.state.value == "succeeded"
    wait_for_result_idle(window)
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
    wait_for_result_idle(window)
    window.close()


def test_result_archive_switches_between_result_and_mesh_modules(
    tmp_path: Path,
) -> None:
    window = open_result_archive_window(tmp_path, "module-switch")
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
    wait_for_result_idle(window)
    window.close()


def test_open_result_reprojects_when_result_module_is_already_current(
    tmp_path: Path,
) -> None:
    archive = make_result_archive(make_continuum_nodal_semantics_result, "already-result")
    path = tmp_path / "already-result.femres"
    save_result_archive(path, archive)
    window = FEMMainWindow()
    window._set_selection_filter("face")
    window.ribbon.set_current("结果")

    receipt = window.open_result_path(path)
    assert receipt.completion is not None
    assert receipt.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    wait_for_result_idle(window)

    assert window._current_module_name() == "结果"
    assert window.navigation.tabs.currentWidget() is window.result_tree
    assert window.viewport._result_render_payload is not None
    assert window.viewport._model is window._result_archive_model_view
    window.close()


def test_result_only_query_inspection_exports_and_default_viewport_projection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = open_result_archive_window(tmp_path, "result-consumers")
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
    wait_for_result_idle(window)
    assert csv_path.is_file()
    assert csv_path.read_text(encoding="utf-8").strip()

    vtk_path = tmp_path / "result-consumers.vtk"
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(vtk_path), ""),
    )
    window.export_vtk()
    wait_for_result_idle(window)
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
    wait_for_result_idle(window)
    window.close()


def test_save_result_path_completes_suffix_and_updates_session_state(tmp_path: Path) -> None:
    archive = make_result_archive(make_continuum_nodal_semantics_result, "save")
    source = tmp_path / "source.femres"
    save_result_archive(source, archive)
    window = FEMMainWindow()
    opened = window.open_result_path(source)
    assert opened.completion is not None
    opened.completion.result(2.0)
    wait_for_result_idle(window)
    target = tmp_path / "copy-without-suffix"
    saved = window.save_result_path(target)
    assert saved.completion is not None
    assert saved.completion.result(2.0).state.value == "succeeded"
    result_path = target.with_suffix(".femres")
    assert result_path.exists()
    assert window.document.result_path == result_path
    assert window.document.unsaved_result_count == 0
    wait_for_result_idle(window)
    window.close()


def test_result_dialog_handlers_route_to_archive_workers(tmp_path: Path, monkeypatch) -> None:
    archive = make_result_archive(make_continuum_nodal_semantics_result, "dialog")
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
    wait_for_result_idle(window)
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
    save_successes: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        window,
        "_show_save_success",
        lambda content_name, path: save_successes.append(
            (content_name, Path(path))
        ),
    )
    assert window.save_current_result(wait=True)
    assert save_calls == []
    assert window.save_current_result_as(wait=True)
    assert window.save_current_result(wait=True)
    assert target.with_suffix(".femres").is_file()
    assert len(save_calls) == 1
    assert save_calls[0][2] == "dialog-source.femres"
    assert all(call[3] == "FEM-Python 结果 (*.femres)" for call in save_calls)
    assert save_successes == [
        ("分析结果", source),
        ("分析结果", target.with_suffix(".femres")),
        ("分析结果", target.with_suffix(".femres")),
    ]
    wait_for_result_idle(window)
    window.close()


def test_result_open_builds_archive_display_payload_off_gui_thread(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = make_result_archive(make_continuum_nodal_semantics_result, "thread")
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
    wait_for_result_idle(window)
    window.close()


def test_result_archive_model_view_uses_profile_dimension_and_dofs() -> None:
    cases = (
        (make_continuum_nodal_semantics_result, 2, 2),
        (make_truss_field_characterization_result, 3, 3),
        (make_beam_field_characterization_result, 3, 6),
    )
    for builder, spatial_dimension, dofs_per_node in cases:
        archive = make_result_archive(builder, builder.__name__)
        view = build_result_archive_model_view(
            archive.model_projection,
            archive.profile,
        )
        assert view.mesh.spatial_dimension == spatial_dimension
        assert view.mesh.dofs_per_node == dofs_per_node
        assert view.mesh.num_dofs == len(archive.topology.node_ids) * dofs_per_node


def test_result_archive_view_keeps_only_result_topology_and_regions() -> None:
    archive = make_result_archive(make_continuum_nodal_semantics_result, "summaries")
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


@pytest.mark.gmsh
def test_real_fempy_project_vertical_remains_openable(
    tmp_path: Path,
    monkeypatch,
    dispose_gui_widget,
) -> None:
    monkeypatch.setattr(FEMMainWindow, "_show_error", lambda *_args, **_kwargs: None)
    # Keep the native route end-to-end while using a coarse, fully constrained
    # sketch so this focused test does not become a mesh/solver benchmark.
    project = make_native_project_snapshot()
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
    wait_for_result_idle(window)
    assert window.document.source_kind == "native"
    assert window.document.project_path == source
    assert not window.legacy_project_extension
    assert window.generate_mesh()
    wait_for_result_idle(window)
    assert window.check_current_model(show_success=False)
    assert window._submit_job("Vertical-Job", "Load") is not None
    wait_for_result_idle(window)
    assert window.document.has_result
    result_target = tmp_path / "vertical-result"
    saved = window.save_result_path(result_target)
    assert saved.completion is not None
    assert saved.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    wait_for_result_idle(window)
    result_path = result_target.with_suffix(".femres")
    assert window.close_model(confirm=False)
    wait_for_result_idle(window)
    dispose_gui_widget(window)
    reopened = FEMMainWindow()
    opened = reopened.open_result_path(result_path)
    assert opened.completion is not None
    assert opened.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    wait_for_result_idle(reopened)
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
    wait_for_result_idle(window)
    window.close()


def test_save_result_as_dialog_cancel_does_not_start_a_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = make_result_archive(make_continuum_nodal_semantics_result, "save-cancel")
    source = tmp_path / "save-cancel-source.femres"
    save_result_archive(source, archive)
    window = FEMMainWindow()
    window.open_result_path(source).completion.result(2.0)
    wait_for_result_idle(window)
    before = result_projection_identity(window)
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
    assert result_projection_identity(window) == before
    wait_for_result_idle(window)
    window.close()


def test_save_result_worker_runs_off_gui_thread(tmp_path: Path, monkeypatch) -> None:
    archive = make_result_archive(make_continuum_nodal_semantics_result, "save-thread")
    source = tmp_path / "save-thread-source.femres"
    save_result_archive(source, archive)
    window = FEMMainWindow()
    window.open_result_path(source).completion.result(2.0)
    wait_for_result_idle(window)
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
    wait_for_result_idle(window)
    window.close()


def test_save_result_worker_failure_cleans_current_save_token(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = make_result_archive(make_continuum_nodal_semantics_result, "save-failure")
    source = tmp_path / "save-failure-source.femres"
    save_result_archive(source, archive)
    window = FEMMainWindow()
    window.open_result_path(source).completion.result(2.0)
    wait_for_result_idle(window)
    before = result_projection_identity(window)
    monkeypatch.setattr(window, "_show_error", lambda *_args, **_kwargs: None)

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(main_window_module, "save_result_archive", fail)
    receipt = window.save_result_path(tmp_path / "save-failure-copy")
    assert receipt.completion is not None
    assert receipt.completion.result(2.0).state is BackgroundTaskState.FAILED
    wait_for_result_idle(window)
    assert not window.busy
    assert not window.session._active_result_save_tasks
    assert result_projection_identity(window) == before
    wait_for_result_idle(window)
    window.close()


def test_save_result_start_rejection_cleans_issued_save_token(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = make_result_archive(make_continuum_nodal_semantics_result, "save-start")
    source = tmp_path / "save-start-source.femres"
    save_result_archive(source, archive)
    window = FEMMainWindow()
    window.open_result_path(source).completion.result(2.0)
    wait_for_result_idle(window)
    before = result_projection_identity(window)
    monkeypatch.setattr(window, "_start_task", lambda *_args, **_kwargs: False)
    receipt = window.save_result_path(tmp_path / "save-start-copy")
    assert receipt.diagnostic is not None
    assert receipt.diagnostic.code == "task.start.rejected"
    assert not window.session._active_result_save_tasks
    assert not window.session._task_data
    assert result_projection_identity(window) == before
    wait_for_result_idle(window)
    window.close()


def test_open_result_decode_failure_preserves_current_document(
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = FEMMainWindow()
    window._apply_session_delta(window.session.new_native_project())
    before = result_projection_identity(window)

    def fail(_path):
        raise ValueError("invalid archive")

    monkeypatch.setattr(main_window_module, "load_result_archive", fail)
    monkeypatch.setattr(window, "_show_error", lambda *_args, **_kwargs: None)
    receipt = window.open_result_path(tmp_path / "broken.femres")
    assert receipt.completion is not None
    assert receipt.completion.result(2.0).state is BackgroundTaskState.FAILED
    wait_for_result_idle(window)
    assert result_projection_identity(window) == before
    assert not window.document.result_only
    wait_for_result_idle(window)
    window.close()


def test_result_save_overwrites_existing_archive_atomically(tmp_path: Path) -> None:
    archive = make_result_archive(make_continuum_nodal_semantics_result, "overwrite")
    source = tmp_path / "overwrite-source.femres"
    target = tmp_path / "overwrite-target.femres"
    save_result_archive(source, archive)
    target.write_bytes(b"stale bytes")
    window = FEMMainWindow()
    window.open_result_path(source).completion.result(2.0)
    wait_for_result_idle(window)
    receipt = window.save_result_path(target)
    assert receipt.completion is not None
    assert receipt.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    assert target.read_bytes() != b"stale bytes"
    assert window.document.result_path == target
    wait_for_result_idle(window)
    window.close()


def test_real_solve_save_close_and_reopen_result_roundtrip(
    gui_inp_path: Path,
    tmp_path: Path,
    dispose_gui_widget,
) -> None:
    window = FEMMainWindow()
    window._load_path(gui_inp_path)
    wait_for_result_idle(window)
    assert window.check_current_model(show_success=False)
    assert window._submit_job("Archive-Job", "Static-1") is not None
    wait_for_result_idle(window)
    assert window.document.has_result
    target = tmp_path / "archive-roundtrip"
    saved = window.save_result_path(target)
    assert saved.completion is not None
    assert saved.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    wait_for_result_idle(window)
    result_path = target.with_suffix(".femres")
    assert result_path.is_file()
    assert window.close_model(confirm=False)
    dispose_gui_widget(window)

    reopened = FEMMainWindow()
    opened = reopened.open_result_path(result_path)
    assert opened.completion is not None
    assert opened.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    wait_for_result_idle(reopened)
    assert reopened.document.result_only
    assert reopened.result_provider is not None
    assert reopened.document.result_path == result_path
    reopened.close()
