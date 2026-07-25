from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fem.application import NamedRegion, RegionAssignment, SectionDefinition
from fem.abaqus import read
from fem.core.model import (
    DisplacementConstraint,
    MaterialDefinition,
    NodalLoad,
)
from fem.solvers.static_linear import solve
from fem.steps.factory import static
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from fem_gui.preprocessing import (
    BoxGeometry,
    LocalMeshControl,
    MeshSettings,
    MovedGeometry,
    RectangleGeometry,
    RotatedGeometry,
)
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.result_adapter import build_result_data


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_actions_follow_document_and_result_context(gui_inp_path):
    _application()
    window = FEMMainWindow()
    assert window.actions["open"].isEnabled()
    assert not window.actions["geometry_sketch"].isEnabled()
    assert "请先新建模型" in window.actions["geometry_sketch"].toolTip()
    assert not window.actions["material_manager"].isEnabled()
    assert not window.actions["geometry_undo"].isEnabled()
    assert not window.actions["geometry_delete"].isEnabled()
    for name in (
        "geometry_rectangle",
        "geometry_plate_hole",
        "geometry_disk",
        "geometry_box",
        "geometry_cylinder",
        "geometry_fragment",
        "mesh_update",
    ):
        assert name not in window.actions
    for name in (
        "geometry_move", "geometry_rotate", "geometry_extrude",
        "geometry_fuse", "geometry_cut",
    ):
        assert not window.actions[name].isEnabled()
    assert not window.actions["mesh_settings"].isEnabled()
    assert not window.actions["mesh_generate"].isEnabled()
    for name in (
        "mesh_clear", "mesh_controls",
        "mesh_statistics", "mesh_quality",
    ):
        assert not window.actions[name].isEnabled()
    for name in ("reload", "close", "submit_job", "fit", "select_node", "model_info", "deformed", "overlay", "query", "export"):
        assert not window.actions[name].isEnabled()
    assert not window.result_variable_combo.isEnabled()

    model = read(gui_inp_path)
    geometry = build_model_geometry(model)
    window._model_loaded(gui_inp_path, (model, geometry))
    window._update_action_states()
    for name in ("reload", "close", "fit", "select_node", "model_info", "symbols", "step_info", "check_model", "job_manager"):
        assert window.actions[name].isEnabled()
    assert not window.actions["submit_job"].isEnabled()
    assert window.check_current_model(show_success=False)
    assert window.actions["submit_job"].isEnabled()
    assert window.actions["mesh_statistics"].isEnabled()
    assert window.actions["mesh_quality"].isEnabled()
    assert not window.actions["mesh_clear"].isEnabled()
    for name in ("deformed", "contour", "query", "export"):
        assert not window.actions[name].isEnabled()

    task = window.session.prepare_solve("Static-1", "Job-1")
    assert task.delta is not None
    window._apply_session_delta(task.delta)
    window._apply_session_delta(window.session.begin_run(task.token))
    result = solve(task.model, task.step_name)
    window._job_succeeded(
        task.token,
        (result, build_result_data(result, geometry)),
    )
    window._update_action_states()
    for name in ("undeformed", "deformed", "contour", "field", "query", "export"):
        assert window.actions[name].isEnabled()
    assert window.result_variable_combo.isEnabled()
    assert window.result_component_combo.isEnabled()

    window.close_model()
    assert "尚无分析结果" == window.result_tree.topLevelItem(0).text(0)
    assert window.status_panel.object_label.text() == "对象：—"
    assert window.status_panel.step_label.text() == "Step：—"
    assert window.status_panel.result_label.text() == "结果：—"
    assert not window.actions["submit_job"].isEnabled()
    window.close()


def test_new_model_unlocks_model_definition_and_sketch_commands():
    _application()
    window = FEMMainWindow()

    window._apply_session_delta(window.session.new_native_project())

    assert window.actions["geometry_sketch"].isEnabled()
    assert window.actions["material_manager"].isEnabled()
    assert window.actions["step_create"].isEnabled()
    window.close()


def test_every_disabled_action_explains_why_it_is_unavailable():
    _application()
    window = FEMMainWindow()

    unexplained = [
        name
        for name, action in window.actions.items()
        if not action.isEnabled()
        and action.toolTip().strip() == action.text().strip()
    ]

    assert unexplained == []
    window.close()


def test_load_action_uses_the_same_dimension_filtered_regions_as_dialog():
    _application()
    window = FEMMainWindow()
    model = SimpleNamespace(
        mesh=SimpleNamespace(
            elements=(SimpleNamespace(type="Hex8"),),
            nodes=(),
        ),
        node_sets={},
        edges={"EDGE_ONLY": object()},
        surfaces={},
        materials={},
        sections=(),
        steps=(static("Load"),),
    )
    task = window.session.prepare_import(Path("volume.inp"))
    delta = window.session.accept_imported_model(task.token, model)
    assert delta.accepted
    window.document = window.session.snapshot()
    window._applied_session_revision = window.document.session_revision
    window._current_step_name = "Load"

    window._update_action_states()

    assert not window.actions["load_create"].isEnabled()
    assert "三维面载荷区域" in window.actions["load_create"].toolTip()
    window.close()


def test_generated_model_uses_the_shared_install_path_without_enabling_reload(
    gui_inp_path,
):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    geometry = build_model_geometry(model)
    recipe = RectangleGeometry("Plate", 2.0, 1.0)
    settings = MeshSettings(0.25)

    window._set_native_geometry(recipe, "矩形")
    window._apply_session_delta(
        window.session.replace_mesh_settings(settings)
    )
    task = window.session.prepare_mesh_generation()
    window._generated_model_loaded(
        (model, geometry),
        token=task.token,
    )

    assert window.document.source_kind == "native"
    assert window.document.geometry_recipe == recipe
    assert window.document.mesh_settings == settings
    assert window.document.model is not None
    assert window.document.artifact is not None
    assert window.geometry is not None
    assert window.geometry.artifact_id == window.document.artifact.artifact_id
    assert not window.actions["reload"].isEnabled()
    assert window.actions["close"].isEnabled()
    assert window.actions["fit"].isEnabled()
    window.close()


def test_native_analysis_actions_are_available_before_meshing():
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(RectangleGeometry("Plate", 2.0, 1.0), "矩形")
    window._apply_session_delta(
        window.session.replace_named_regions(
            (NamedRegion("Fixed", "edge", (1,)),)
        )
    )
    window._apply_session_delta(
        window.session.replace_model_definitions(
            (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
            (SectionDefinition("Solid", "Steel"),),
            (),
            (static("Load"),),
        )
    )

    assert window.actions["section_assign"].isEnabled()
    assert window.actions["step_create"].isEnabled()
    assert window.actions["boundary_create"].isEnabled()
    assert window.actions["load_create"].isEnabled()
    assert window.actions["output_create"].isEnabled()
    assert window.actions["analysis_manager"].isEnabled()
    assert window.actions["close"].isEnabled()
    assert window.actions["model_info"].isEnabled()
    assert not window.actions["check_model"].isEnabled()
    window.close()


def test_short_action_labels_fit_the_ribbon_vocabulary():
    _application()
    window = FEMMainWindow()

    assert window.actions["new_native"].text() == "新建模型"
    assert window.actions["open_project"].text() == "打开项目"
    assert window.actions["save_project"].text() == "保存项目"
    assert window.actions["open"].text() == "打开 INP"
    assert window.actions["export"].text() == "导出 CSV"
    window.close()


def test_window_title_shows_source_and_unsaved_state(gui_inp_path):
    _application()
    window = FEMMainWindow()
    assert window.windowTitle() == "有限元分析"

    window._apply_session_delta(window.session.new_native_project())
    assert "[自主]" in window.windowTitle()
    assert not window.windowTitle().endswith("*")

    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    assert window.windowTitle().endswith("*")

    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    assert gui_inp_path.name in window.windowTitle()
    assert "[INP]" in window.windowTitle()
    assert not window.windowTitle().endswith("*")
    window.close()


def test_boundary_action_requests_a_viewport_region_before_opening_parameters(
    monkeypatch,
):
    application = _application()
    window = FEMMainWindow()
    window._set_native_geometry(RectangleGeometry("Plate", 2.0, 1.0), "矩形")
    window._apply_session_delta(
        window.session.replace_model_definitions(
            (),
            (),
            (),
            (static("Load"),),
        )
    )

    window.create_displacement_boundary()

    assert window._pending_analysis_selection == "boundary"
    assert window.viewport._selection_mode == "geometry_edge"
    calls: list[bool] = []
    monkeypatch.setattr(
        window,
        "create_displacement_boundary",
        lambda: calls.append(True),
    )
    window._on_viewport_pick("geometry_edge", 1)
    application.processEvents()

    assert calls == []
    assert window._pending_analysis_selection == "boundary"
    window._confirm_guided_selection()
    application.processEvents()

    assert calls == [True]
    assert window._pending_analysis_selection is None
    window.close()


@pytest.mark.parametrize(
    ("before", "after"),
    (
        (
            MovedGeometry(RectangleGeometry("Plate", 2.0, 1.0), 1.0, 0.0),
            MovedGeometry(RectangleGeometry("Plate", 3.0, 1.5), 2.0, -1.0),
        ),
        (
            RotatedGeometry(
                RectangleGeometry("Plate", 2.0, 1.0),
                "z",
                15.0,
            ),
            RotatedGeometry(
                RectangleGeometry("Plate", 3.0, 1.5),
                "z",
                60.0,
            ),
        ),
    ),
    ids=("move-parameters", "rotate-parameters"),
)
def test_geometry_parameter_edits_preserve_topology_references(before, after):
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(before, "变换后的")
    window._apply_session_delta(
        window.session.replace_named_regions(
            (NamedRegion("Fixed", "edge", (1,)),)
        )
    )
    window._apply_session_delta(
        window.session.replace_model_definitions(
            (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
            (SectionDefinition("Solid", "Steel"),),
            (RegionAssignment("Solid", "Fixed"),),
            (),
        )
    )
    window._apply_session_delta(
        window.session.replace_mesh_settings(
            MeshSettings(
                0.2,
                local_size=0.15,
                local_controls=(LocalMeshControl("edge", 1, 0.1),),
            )
        )
    )

    window._set_native_geometry(after, "参数修改后的")

    assert tuple(window.document.named_regions) == ("Fixed",)
    assert window.document.region_assignments == (
        RegionAssignment("Solid", "Fixed"),
    )
    assert window.document.mesh_settings.local_size == 0.15
    assert window.document.mesh_settings.local_controls == (
        LocalMeshControl("edge", 1, 0.1),
    )
    assert "已有拓扑引用已保留" in window.status_panel.state_label.text()
    assert "旧命名区域已失效" not in window.status_panel.state_label.text()
    window.close()


def test_geometry_topology_change_reports_cleared_references():
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        BoxGeometry("Box", 2.0, 1.0, 0.5),
        "长方体",
    )
    window._apply_session_delta(
        window.session.replace_named_regions(
            (NamedRegion("Fixed", "edge", (1,)),)
        )
    )
    window._apply_session_delta(
        window.session.replace_mesh_settings(
            MeshSettings(
                0.2,
                local_size=0.15,
                local_controls=(LocalMeshControl("edge", 1, 0.1),),
            )
        )
    )

    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )

    assert window.document.named_regions == {}
    assert window.document.mesh_settings.local_size is None
    assert window.document.mesh_settings.local_controls == ()
    message = window.status_panel.state_label.text()
    assert "1 个旧命名区域已失效" in message
    assert "旧局部网格设置已失效" in message
    window.close()


def test_cancelled_discard_confirmation_keeps_the_native_project(monkeypatch):
    _application()
    window = FEMMainWindow()
    window._apply_session_delta(window.session.new_native_project())
    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    assert window.document.dirty
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: False)

    assert not window.close_model()
    assert window.document.source_kind == "native"

    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
    assert window.close_model()
    assert window.document.source_kind is None
    assert not window.document.dirty
    window.close()


def test_geometry_ctrl_selection_accumulates_same_kind_entities(monkeypatch):
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    monkeypatch.setattr(window, "_geometry_pick_is_additive", lambda: False)
    window._on_viewport_pick("geometry_edge", 1)
    monkeypatch.setattr(window, "_geometry_pick_is_additive", lambda: True)
    window._on_viewport_pick("geometry_edge", 3)

    assert window._selected_geometry_kind == "edge"
    assert window._selected_geometry_ids == {1, 3}
    assert window.status_panel.object_label.text() == "对象：已选择 2 个边"
    assert window.actions["geometry_region"].isEnabled()

    window._on_viewport_pick("geometry_edge", 3)
    assert window._selected_geometry_ids == {1}
    assert window._selected_geometry_id == 1
    window.close()


def test_switching_geometry_selection_kind_clears_incompatible_selection(
    monkeypatch,
):
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    monkeypatch.setattr(window, "_geometry_pick_is_additive", lambda: False)
    window._set_geometry_selection_mode("edge")
    window._on_viewport_pick("geometry_edge", 1)

    window._set_geometry_selection_mode("face")

    assert window.viewport._selection_mode == "geometry_face"
    assert window._selected_geometry_kind is None
    assert window._selected_geometry_id is None
    assert window._selected_geometry_ids == set()
    assert "geometry_selection" not in window.viewport._actors
    window.close()


def test_switching_from_fem_to_geometry_selection_clears_stale_highlight(
    monkeypatch,
):
    _application()
    window = FEMMainWindow()
    window.selection.select_node(50)
    window.actions["selected_info"].setEnabled(True)
    cleared: list[bool] = []
    monkeypatch.setattr(
        window.viewport,
        "clear_selection",
        lambda: cleared.append(True),
    )

    window._set_geometry_selection_mode("face")

    assert window.selection.node_id is None
    assert window.selection.element_id is None
    assert cleared == [True]
    assert not window.actions["selected_info"].isEnabled()
    window.close()


def test_local_mesh_control_applies_once_to_all_selected_edges(monkeypatch):
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    window._selected_geometry_kind = "edge"
    window._selected_geometry_id = 1
    window._selected_geometry_ids = {1, 3}

    class AcceptedLocalMeshDialog:
        def __init__(self, kind, entity_id, global_size, parent):
            self._control = LocalMeshControl(kind, entity_id, global_size / 2.0)

        def exec(self):
            return True

        def control(self):
            return self._control

    monkeypatch.setattr(
        main_window_module,
        "LocalMeshControlDialog",
        AcceptedLocalMeshDialog,
    )

    window.set_local_mesh_control()

    controls = window.document.mesh_settings.local_controls
    assert {(item.entity_kind, item.entity_id) for item in controls} == {
        ("edge", 1),
        ("edge", 3),
    }
    assert {item.size for item in controls} == {
        window.document.mesh_settings.size / 2.0
    }
    window.close()


def test_named_region_default_names_do_not_expose_topology_ids():
    _application()
    window = FEMMainWindow()
    window._apply_session_delta(window.session.new_native_project())
    window._apply_session_delta(
        window.session.replace_named_regions(
            (NamedRegion("Surface-1", "face", (4,)),)
        )
    )

    assert window._next_named_region_name("face") == "Surface-2"
    assert window._next_named_region_name("edge") == "EdgeSet-1"
    window.close()


def test_named_region_rename_updates_analysis_and_section_references(
    monkeypatch,
):
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    window._apply_session_delta(
        window.session.replace_named_regions(
            (NamedRegion("Fixed", "edge", (1, 3)),)
        )
    )
    step = static("Load")
    step.boundaries = (DisplacementConstraint("Fixed", 1, 2, 0.0),)
    step.cloads = (NodalLoad("Fixed", 1, 10.0),)
    window._apply_session_delta(
        window.session.replace_model_definitions(
            (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
            (SectionDefinition("Solid", "Steel"),),
            (RegionAssignment("Solid", "Fixed"),),
            (step,),
        )
    )

    class FakeRegionManager:
        def __init__(self, _regions, _parent):
            pass

        def exec(self):
            return True

        def values(self):
            return {
                "Support": NamedRegion("Support", "edge", (1, 3)),
            }

    monkeypatch.setattr(
        "fem_gui.main_window.NamedRegionManagerDialog",
        FakeRegionManager,
    )
    window.show_named_region_manager()

    assert tuple(window.document.named_regions) == ("Support",)
    assert window.document.region_assignments[0].region_name == "Support"
    renamed_step = window.document.analysis_definitions[0]
    assert renamed_step.boundaries[0].target == "Support"
    assert renamed_step.cloads[0].target == "Support"
    window.close()
