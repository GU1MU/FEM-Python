from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fem.application import (
    DeleteIntent,
    MeshEntityRef,
    NamedRegion,
    RegionAssignment,
    RenameIntent,
    SectionDefinition,
    generate_fem_model,
)
from fem.application.results import build_solve_result_bundle
from fem.io.inp import read
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    GravityLoad,
    MaterialDefinition,
    NodalLoad,
)
from fem.geometry import (
    BoxGeometry,
    LogicalEntityRef,
    MovedGeometry,
    RectangleGeometry,
    RotatedGeometry,
    WireGeometry,
    WireMember,
    WirePoint,
    namespace_part_logical_id,
)
from fem.mesh import settings as mesh_settings_api
from fem.mesh.settings import LocalMeshControl, MeshSettings
from fem.solvers.static_linear import solve
from fem.steps.factory import static
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from fem_gui.visualization.model_adapter import build_model_geometry


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _named_region(name: str, *logical_ids: str) -> NamedRegion:
    return NamedRegion(
        name,
        tuple(_part_ref(logical_id) for logical_id in logical_ids),
    )


def _part_ref(logical_id: str, part_id: str = "P1") -> LogicalEntityRef:
    return LogicalEntityRef(
        namespace_part_logical_id(part_id, logical_id)
    )


def _global_local_control(
    logical_id: str,
    size: float,
) -> LocalMeshControl:
    return LocalMeshControl(
        _part_ref(logical_id),
        size,
        mesh_settings_api.MeshSizeFalloff(
            "global_size",
            0.0,
            2.0,
        ),
    )


def test_actions_follow_document_and_result_context(gui_inp_path):
    _application()
    window = FEMMainWindow()
    assert window.actions["open"].isEnabled()
    assert not window.actions["geometry_create"].isEnabled()
    assert "请先新建模型" in window.actions["geometry_create"].toolTip()
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
    for name in (
        "reload",
        "close",
        "submit_job",
        "fit",
        "select_node",
        "model_info",
        "deformed",
        "overlay",
        "query",
        "export_csv",
        "export_vtk",
    ):
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
    for name in ("deformed", "contour", "query", "export_csv", "export_vtk"):
        assert not window.actions[name].isEnabled()

    task = window.session.prepare_solve("Static-1", "Job-1")
    assert task.delta is not None
    window._apply_session_delta(task.delta)
    window._apply_session_delta(window.session.begin_run(task.token))
    result = solve(task.model, task.step_name)
    window._job_succeeded(
        task.token,
        (build_solve_result_bundle(task, result), {}),
    )
    window._update_action_states()
    for name in (
        "undeformed",
        "deformed",
        "contour",
        "field",
        "query",
        "export_csv",
        "export_vtk",
    ):
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

    assert window.actions["geometry_create"].isEnabled()
    assert window.actions["geometry_sketch"].isEnabled()
    assert window.actions["material_manager"].isEnabled()
    assert window.actions["step_create"].isEnabled()
    window.close()


def test_new_model_dialog_commits_entered_model_name_and_cancel_is_safe(
    monkeypatch,
):
    _application()
    responses = iter((("支架模型", True), ("", False)))
    prompts = []

    def get_text(_parent, title, prompt, **options):
        prompts.append((title, prompt, options.get("text")))
        return next(responses)

    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        get_text,
    )
    window = FEMMainWindow()

    window.new_native_model()

    assert prompts == [("新建模型", "模型名称：", "模型-1")]
    assert window.document.model_name == "支架模型"
    assert window.document.parts == ()
    assert window.document.active_part_id is None
    assert window.model_tree.topLevelItem(0).text(0) == "支架模型"
    assert window.model_tree.topLevelItem(0).childCount() == 0
    revision = window.document.session_revision

    window.new_native_model()

    assert window.document.session_revision == revision
    assert window.document.model_name == "支架模型"
    window.close()


def test_project_save_ui_follows_can_save_in_all_session_states(
    gui_inp_path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    message_boxes = []

    class FakeMessageBox:
        class Icon:
            Warning = object()

        class ButtonRole:
            AcceptRole = object()
            DestructiveRole = object()
            RejectRole = object()

        def __init__(self, _parent) -> None:
            self.buttons = []
            self._clicked = None
            message_boxes.append(self)

        def setWindowTitle(self, _title) -> None:
            pass

        def setIcon(self, _icon) -> None:
            pass

        def setText(self, _text) -> None:
            pass

        def addButton(self, text, role):
            button = object()
            self.buttons.append((text, role, button))
            if role is self.ButtonRole.DestructiveRole:
                self._clicked = button
            return button

        def setDefaultButton(self, _button) -> None:
            pass

        def exec(self) -> None:
            pass

        def clickedButton(self):
            return self._clicked

    monkeypatch.setattr(main_window_module, "QMessageBox", FakeMessageBox)

    def assert_save_ui(expected: bool) -> None:
        snapshot = window.session.snapshot()
        assert snapshot.can_save is expected
        window.document = replace(snapshot, dirty=True)
        window._update_action_states()

        assert window.actions["save_project"].isEnabled() is expected
        if not expected:
            assert not window.save_native_project()

        previous_count = len(message_boxes)
        assert window._confirm_discard_changes()
        assert len(message_boxes) == previous_count + 1
        labels = {text for text, _role, _button in message_boxes[-1].buttons}
        assert ("保存" in labels) is expected

    assert_save_ui(False)

    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    assert_save_ui(False)

    window.close_model(confirm=False)
    window._apply_session_delta(window.session.new_native_project())
    assert_save_ui(True)

    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    assert_save_ui(True)

    window.close_model(confirm=False)
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
    assert "[step.reference.invalid]" in window.actions["load_create"].toolTip()
    assert "请选择当前模型中存在的同类命名区域" in (
        window.actions["load_create"].toolTip()
    )
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


def test_native_scope_dependent_actions_require_meshing():
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(RectangleGeometry("Plate", 2.0, 1.0), "矩形")
    window._apply_session_delta(
        window.session.replace_named_regions(
            (_named_region("Fixed", "edge:bottom"),)
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

    assert not window.actions["section_assign"].isEnabled()
    assert window.actions["step_create"].isEnabled()
    assert not window.actions["boundary_create"].isEnabled()
    assert not window.actions["load_create"].isEnabled()
    assert window.actions["output_create"].isEnabled()
    assert window.actions["output_create"].toolTip() == "输出请求"
    assert window.actions["analysis_manager"].isEnabled()
    assert window.actions["close"].isEnabled()
    assert window.actions["model_info"].isEnabled()
    assert not window.actions["check_model"].isEnabled()
    window.close()


def test_truss_member_policy_disables_only_local_mesh_control():
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        WireGeometry(
            "Bar",
            (
                WirePoint("P1", 0.0, 0.0, 0.0),
                WirePoint("P2", 1.0, 0.0, 0.0),
            ),
            (WireMember("M1", "P1", "P2"),),
        ),
        "Wire",
    )
    window._apply_session_delta(
        window.session.replace_mesh_settings(
            MeshSettings(
                0.1,
                cell_shape="line",
                line_element_type="Truss2",
            )
        )
    )

    assert window.actions["mesh_generate"].isEnabled()
    assert window.actions["mesh_controls"].isEnabled()
    assert not window.actions["mesh_local_control"].isEnabled()
    assert "固定生成一个单元" in window.actions[
        "mesh_local_control"
    ].toolTip()

    window._apply_session_delta(
        window.session.replace_mesh_settings(
            MeshSettings(
                0.1,
                cell_shape="line",
                local_controls=(
                    _global_local_control("edge:M1", 0.05),
                ),
                line_element_type="Truss2",
            )
        )
    )
    assert not window.actions["mesh_generate"].isEnabled()
    assert "删除局部尺寸" in window.actions["mesh_generate"].toolTip()
    assert window.actions["mesh_controls"].isEnabled()
    window.close()


def test_short_action_labels_fit_the_ribbon_vocabulary():
    _application()
    window = FEMMainWindow()

    assert window.actions["new_native"].text() == "新建模型"
    assert window.actions["open_project"].text() == "打开项目"
    assert window.actions["save_project"].text() == "保存项目"
    assert window.actions["open"].text() == "打开 INP"
    assert window.actions["export_csv"].text() == "导出 CSV"
    assert window.actions["export_vtk"].text() == "导出 VTK"
    assert "export" not in window.actions
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


def test_boundary_action_does_not_open_scope_dialog_before_meshing(
    monkeypatch,
):
    _application()
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
    captured: dict[str, object] = {}

    class Dialog:
        def __init__(
            self,
            step_names,
            node_regions,
            dimensions,
            parent,
            **_kwargs,
        ):
            captured.update(
                step_names=step_names,
                node_regions=node_regions,
                dimensions=dimensions,
                parent=parent,
            )

        @staticmethod
        def exec():
            return False

    monkeypatch.setattr(
        main_window_module,
        "DisplacementDialog",
        Dialog,
    )

    window.create_displacement_boundary()

    assert captured == {}
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
            (
                _named_region("Fixed", "edge:bottom"),
                _named_region("SolidDomain", "body:domain"),
            )
        )
    )
    window._apply_session_delta(
        window.session.replace_model_definitions(
            (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
            (SectionDefinition("Solid", "Steel"),),
            (RegionAssignment("Solid", "SolidDomain"),),
            (),
        )
    )
    window._apply_session_delta(
        window.session.replace_mesh_settings(
            MeshSettings(
                0.2,
                local_controls=(
                    _global_local_control("edge:bottom", 0.1),
                ),
            )
        )
    )

    window._set_native_geometry(after, "参数修改后的")

    assert set(window.document.named_regions) == {"Fixed", "SolidDomain"}
    assert window.document.assignments == (
        RegionAssignment("Solid", "SolidDomain"),
    )
    assert window.document.mesh_settings.local_controls == (
        _global_local_control("edge:bottom", 0.1),
    )
    message = window.status_panel.state_label.text()
    assert "参数修改后的几何已创建" in message
    assert "旧命名区域已失效" not in message
    window.close()


def test_geometry_topology_change_clears_invalid_references():
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        BoxGeometry("Box", 2.0, 1.0, 0.5),
        "长方体",
    )
    window._apply_session_delta(
        window.session.replace_named_regions(
            (_named_region("Fixed", "edge:bottom-front"),)
        )
    )
    window._apply_session_delta(
        window.session.replace_mesh_settings(
            MeshSettings(
                0.2,
                local_controls=(
                    _global_local_control("edge:bottom-front", 0.1),
                ),
            )
        )
    )

    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )

    assert window.document.named_regions == {}
    assert window.document.mesh_settings.local_controls == ()
    message = window.status_panel.state_label.text()
    assert "矩形几何已创建" in message
    assert "旧局部网格设置已失效" in message
    window.close()


def test_geometry_topology_change_preserves_topology_independent_steps():
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        BoxGeometry("Box", 2.0, 1.0, 0.5),
        "长方体",
    )
    empty = AnalysisStep("Empty")
    global_gravity = AnalysisStep(
        "Global Gravity",
        gravity_loads=(GravityLoad((0.0, -9.81, 0.0)),),
    )
    window._apply_session_delta(
        window.session.replace_model_definitions(
            (),
            (),
            (),
            (empty, global_gravity),
        )
    )

    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )

    assert window.document.steps == (
        empty,
        global_gravity,
    )
    assert "矩形几何已创建" in window.status_panel.state_label.text()
    window.close()


def test_geometry_topology_change_invalidates_region_target_step():
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        BoxGeometry("Box", 2.0, 1.0, 0.5),
        "长方体",
    )
    window._apply_session_delta(
        window.session.replace_named_regions(
            (_named_region("SolidDomain", "body:domain"),)
        )
    )
    window._apply_session_delta(
        window.session.replace_model_definitions(
            (),
            (),
            (),
            (
                AnalysisStep(
                    "Region Gravity",
                    gravity_loads=(
                        GravityLoad(
                            (0.0, -9.81, 0.0),
                            "SolidDomain",
                        ),
                    ),
                ),
            ),
        )
    )

    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )

    assert window.document.steps == ()
    assert "矩形几何已创建" in window.status_panel.state_label.text()
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
    bottom = _part_ref("edge:bottom")
    top = _part_ref("edge:top")
    window._on_geometry_entity_pick(bottom)
    monkeypatch.setattr(window, "_geometry_pick_is_additive", lambda: True)
    window._on_geometry_entity_pick(top)

    assert window._geometry_selection_kind() == "edge"
    assert window._selected_geometry_refs == {bottom, top}
    assert window._canonical_geometry_selection() == (bottom, top)
    assert window.status_panel.object_label.text() == "对象：已选择 2 个边"
    assert not window.actions["geometry_region"].isEnabled()

    window._on_geometry_entity_pick(top)
    assert window._selected_geometry_refs == {bottom}
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
    window._on_geometry_entity_pick(
        _part_ref("edge:bottom")
    )

    window._set_geometry_selection_mode("face")

    assert window.viewport._selection_mode == "geometry_face"
    assert window._selected_geometry_refs == set()
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
    bottom = _part_ref("edge:bottom")
    top = _part_ref("edge:top")
    window._selected_geometry_refs = {bottom, top}

    class AcceptedLocalMeshDialog:
        def __init__(self, target, global_size, parent):
            self._control = _global_local_control(
                target.logical_id,
                global_size / 2.0,
            )

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
    assert {item.target for item in controls} == {
        bottom,
        top,
    }
    assert {item.size for item in controls} == {
        window.document.mesh_settings.size / 2.0
    }
    window.close()


def test_named_region_default_names_do_not_expose_topology_ids():
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    window._apply_session_delta(
        window.session.replace_named_regions(
            (_named_region("Surface-1", "face:domain"),)
        )
    )

    assert window._next_named_region_name("face") == "Surface-2"
    assert window._next_named_region_name("edge") == "EdgeSet-1"
    window.close()


def test_mesh_scope_ctrl_pick_toggles_selected_entity(monkeypatch) -> None:
    _application()
    window = FEMMainWindow()
    first = MeshEntityRef.node(1)
    second = MeshEntityRef.node(2)
    monkeypatch.setattr(window, "_geometry_pick_is_additive", lambda: False)
    window._on_mesh_scope_entity_pick(first)
    monkeypatch.setattr(window, "_geometry_pick_is_additive", lambda: True)
    window._on_mesh_scope_entity_pick(second)

    assert window._selected_mesh_scope_refs == {first, second}

    window._on_mesh_scope_entity_pick(second)
    assert window._selected_mesh_scope_refs == {first}
    window.close()


def test_former_builtin_region_name_can_be_created_as_a_mesh_scope(
    monkeypatch,
):
    _application()
    window = FEMMainWindow()
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )
    settings = MeshSettings(0.25)
    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    window._apply_session_delta(
        window.session.replace_mesh_settings(settings)
    )
    task = window.session.prepare_mesh_generation()
    model = generate_fem_model(task)
    geometry = build_model_geometry(model)
    window._generated_model_loaded(
        (model, geometry),
        token=task.token,
    )
    assert errors == []
    selected = MeshEntityRef.node(
        model.mesh.nodes[0].id,
        part_id=window.document.active_part_id,
    )
    window._selected_mesh_scope_refs = {selected}
    topology = window._scope_selection_topology()
    mesh = window.document.model.mesh
    assert mesh is window.session.projection_snapshot().model.mesh
    install_calls: list[object] = []
    monkeypatch.setattr(
        window,
        "_install_model",
        lambda *args, **kwargs: install_calls.append((args, kwargs)),
    )

    class FormerBuiltinRegionDialog:
        def __init__(self, *_args, **_kwargs):
            pass

        def exec(self):
            return True

        def region_name(self):
            return "bottom"

    monkeypatch.setattr(
        main_window_module,
        "NamedRegionDialog",
        FormerBuiltinRegionDialog,
    )

    created = window._create_region_from_current_mesh_selection()
    assert errors == []
    assert created == "bottom"
    assert window.document.named_regions["bottom"].references == (selected,)
    assert window.document.model.node_sets["bottom"].node_ids == (
        selected.node_id,
    )
    assert window.document.model.mesh is mesh
    assert window._scope_selection_topology_cache is topology
    assert window.geometry.artifact_id == window.document.artifact.artifact_id
    assert window.viewport.artifact_id == window.document.artifact.artifact_id
    assert install_calls == []
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
            (
                _named_region("Fixed", "edge:bottom", "edge:top"),
                _named_region("Volume", "body:domain"),
            )
        )
    )
    step = static("Load")
    step.boundaries = (DisplacementConstraint("Fixed", 1, 2, 0.0),)
    step.cloads = (NodalLoad("Fixed", 1, 10.0),)
    window._apply_session_delta(
        window.session.replace_model_definitions(
            (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
            (SectionDefinition("Solid", "Steel"),),
            (RegionAssignment("Solid", "Volume"),),
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
                "Support": _named_region(
                    "Support",
                    "edge:bottom",
                    "edge:top",
                ),
                "SolidDomain": _named_region(
                    "SolidDomain",
                    "body:domain",
                ),
            }

        @staticmethod
        def rename_intents() -> tuple[RenameIntent, ...]:
            return (
                RenameIntent("Fixed", "Support"),
                RenameIntent("Volume", "SolidDomain"),
            )

        @staticmethod
        def delete_intents() -> tuple[DeleteIntent, ...]:
            return ()

    monkeypatch.setattr(
        "fem_gui.main_window.NamedRegionManagerDialog",
        FakeRegionManager,
    )
    window.show_named_region_manager()

    assert set(window.document.named_regions) == {
        "Support",
        "SolidDomain",
    }
    assert (
        window.document.assignments[0].region_name
        == "SolidDomain"
    )
    renamed_step = window.document.steps[0]
    assert renamed_step.boundaries[0].target == "Support"
    assert renamed_step.cloads[0].target == "Support"
    window.close()
