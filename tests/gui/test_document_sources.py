from __future__ import annotations

from pathlib import Path

from fem.steps.factory import static
from fem_gui.document import FEMDocument, FeatureRecord
from tests.helpers.model_builders import make_static_pull_truss_model


def test_inp_source_keeps_reload_path_and_clears_generated_state() -> None:
    document = FEMDocument()
    document.set_generated_model(
        make_static_pull_truss_model(),
        geometry_recipe={"shape": "rectangle"},
        mesh_settings={"size": 0.25},
    )

    document.set_model("model.inp", make_static_pull_truss_model())

    assert document.source_kind == "inp"
    assert document.can_reload
    assert document.path is not None
    assert document.path.name == "model.inp"
    assert document.geometry_recipe is None
    assert document.mesh_settings is None


def test_generated_source_uses_the_same_document_without_a_fake_path() -> None:
    document = FEMDocument()
    recipe = {"shape": "rectangle"}
    settings = {"size": 0.25}

    document.set_generated_model(
        make_static_pull_truss_model(),
        geometry_recipe=recipe,
        mesh_settings=settings,
    )

    assert document.source_kind == "native"
    assert not document.can_reload
    assert document.path is None
    assert document.geometry_recipe is recipe
    assert document.mesh_settings is settings
    assert document.native_mesh_current
    assert document.mesh_is_current
    assert document.needs_model_check
    assert document.has_model


def test_close_clears_source_specific_and_shared_model_state() -> None:
    document = FEMDocument()
    document.set_generated_model(
        make_static_pull_truss_model(),
        geometry_recipe={"shape": "rectangle"},
        mesh_settings={"size": 0.25},
    )

    document.close()

    assert document.source_kind is None
    assert document.path is None
    assert document.geometry_recipe is None
    assert document.mesh_settings is None
    assert not document.native_mesh_current


def test_native_geometry_change_invalidates_mesh_model_and_results() -> None:
    document = FEMDocument()
    document.set_generated_model(
        make_static_pull_truss_model(),
        geometry_recipe={"shape": "rectangle"},
        mesh_settings={"size": 0.25},
    )
    document.result = object()
    document.mark_result_current()

    document.begin_native_model(
        {"shape": "circle"},
        feature=FeatureRecord("Sketch-1", "sketch"),
    )

    assert document.source_kind == "native"
    assert not document.has_model
    assert not document.has_result
    assert not document.native_mesh_current
    assert not document.workflow.mesh_current
    assert document.workflow.reason == "几何已修改，网格需要重新生成"
    assert [feature.name for feature in document.feature_history] == ["Sketch-1"]


def test_native_definitions_and_save_target_survive_geometry_regeneration() -> None:
    document = FEMDocument()
    document.new_native_model()
    document.native_project_path = Path("plate.femproj")
    document.analysis_definitions = [static("Load")]

    document.begin_native_model({"shape": "rectangle"})

    assert document.native_project_path == Path("plate.femproj")
    assert document.runnable_step_names() == ("Load",)
    assert document.step_name == "Load"


def test_model_input_change_requires_a_new_check_and_expires_result() -> None:
    document = FEMDocument()
    document.set_model("model.inp", make_static_pull_truss_model())
    document.mark_model_checked()
    document.result = object()
    document.mark_result_current()

    document.mark_model_definition_changed("材料已修改，模型需要重新检查")

    assert document.needs_model_check
    assert not document.has_result
    assert not document.workflow.results_current
    assert document.workflow.reason == "材料已修改，模型需要重新检查"


def test_inp_geometry_entry_explains_that_cad_cannot_be_reverse_engineered(monkeypatch, gui_inp_path) -> None:
    from PySide6.QtWidgets import QApplication
    from fem.abaqus import read
    from fem_gui.main_window import FEMMainWindow
    from fem_gui.visualization.model_adapter import build_model_geometry

    application = QApplication.instance() or QApplication([])
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(gui_inp_path, (model, build_model_geometry(model)))
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "_show_error", lambda title, message: messages.append((title, message)))

    window.create_sketch_geometry()

    assert messages == [("几何编辑不可用", "当前 INP 只包含有限元模型和网格，不能反向转换为可编辑 CAD；请新建自主模型。")]
    assert window.document.source_kind == "inp"
    window.close()
    application.processEvents()
