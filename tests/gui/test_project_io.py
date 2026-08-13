from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import threading
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from fem.application import (
    derive_feature_history,
    ModelSession,
    NamedRegion,
    NativePart,
    ProjectSnapshot,
    RegionAssignment,
    SectionDefinition,
    with_compatibility_analysis_names,
)
from fem.geometry import LogicalEntityRef
from fem.core.model import (
    BodyForce,
    DisplacementConstraint,
    EdgeLoad,
    GravityLoad,
    MaterialDefinition,
    NodalLoad,
    FEMModel,
)
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.geometry.recipes import (
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
    WireGeometry,
    WireMember,
    WirePoint,
)
from fem.io.project import (
    ProjectDecodeError,
    load_project,
    save_project,
)
from fem.mesh.settings import (
    LocalMeshControl,
    MeshSettings,
    MeshSizeFalloff,
)
from fem.steps.factory import static
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from tests.helpers.gui_command_receipts import await_succeeded


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_task(window: FEMMainWindow, timeout: float = 2.0) -> None:
    deadline = monotonic() + timeout
    application = _application()
    def busy() -> bool:
        controller = window.workspace.open_controller
        return window.busy or (controller is not None and controller.busy)

    while busy() and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not busy()


def _tree_texts(item) -> tuple[str, ...]:
    return (
        item.text(0),
        *(
            text
            for index in range(item.childCount())
            for text in _tree_texts(item.child(index))
        ),
    )


def _native_project_snapshot() -> ProjectSnapshot:
    recipe = SketchGeometry(
        "Plate",
        (
            SketchRectangle("material", 0.0, 0.0, 100.0, 50.0),
            SketchCircle("cut", 50.0, 25.0, 8.0),
        ),
    )
    step = static("Load")
    step.boundaries = (
        DisplacementConstraint("LEFT", 1, 2, 0.0),
    )
    step.cloads = (NodalLoad("RIGHT", 1, 100.0),)
    step.edge_loads = (EdgeLoad("TOP", (0.0, -5.0)),)
    step.body_loads = (BodyForce("DOMAIN", (1.0, -2.0)),)
    step.gravity_loads = (GravityLoad((0.0, -9.81)),)
    return ProjectSnapshot(
        source_kind="native",
        parts=(NativePart(),),
        geometry_recipe=recipe,
        mesh_settings=MeshSettings(
            5.0,
            order=2,
            cell_shape="quadrilateral",
            local_controls=(
                LocalMeshControl(
                    LogicalEntityRef("edge:outer-loop"),
                    1.0,
                ),
                LocalMeshControl(
                    LogicalEntityRef("edge:hole-loop"),
                    2.5,
                    MeshSizeFalloff("target_radius", 0.25, 2.0),
                ),
            ),
        ),
        feature_history=derive_feature_history(recipe),
        named_regions=(
            NamedRegion(
                "OuterBoundary",
                (LogicalEntityRef("edge:outer-loop"),),
            ),
        ),
        material_definitions=(
            MaterialDefinition(
                "Steel",
                {"E": 210000.0, "nu": 0.3},
            ),
        ),
        section_definitions=(
            SectionDefinition(
                "Section-1",
                "Steel",
                properties={"thickness": 2.0},
            ),
        ),
        region_assignments=(
            RegionAssignment("Section-1", "DOMAIN"),
        ),
        analysis_definitions=(step,),
    )


def test_gui_uses_only_the_version_neutral_project_router() -> None:
    assert importlib.util.find_spec("fem_gui.project_io") is None
    assert main_window_module.load_project is load_project
    assert main_window_module.save_project is save_project


def test_native_project_round_trip_returns_a_detached_snapshot(tmp_path) -> None:
    original = _native_project_snapshot()

    target = save_project(
        tmp_path / "plate.femproj",
        original,
    )
    loaded = load_project(target)
    reopened = loaded.snapshot

    assert isinstance(reopened, ProjectSnapshot)
    assert loaded.source_schema == 14
    assert loaded.notices == ()
    assert reopened.source_kind == "native"
    assert reopened.source_path == target
    assert reopened.geometry_recipe == original.geometry_recipe
    assert reopened.mesh_settings == reopened.parts[0].mesh_settings
    assert {
        control.target.logical_id
        for control in reopened.mesh_settings.local_controls
    } == {"edge:P1/outer-loop", "edge:P1/hole-loop"}
    assert reopened.material_definitions[0].name == "Steel"
    assert reopened.section_definitions[0].properties["thickness"] == 2.0
    assert reopened.region_assignments[0].region_name == "DOMAIN"
    assert reopened.analysis_definitions[0].cloads[0].value == 100.0
    expected_step = with_compatibility_analysis_names(
        original.analysis_definitions
    )[0]
    assert reopened.analysis_definitions[0].edge_loads == expected_step.edge_loads
    assert reopened.analysis_definitions[0].body_loads == expected_step.body_loads
    assert (
        reopened.analysis_definitions[0].gravity_loads
        == expected_step.gravity_loads
    )


def test_failed_detached_decode_cannot_change_a_live_session(tmp_path) -> None:
    session = ModelSession()
    session.new_native_project()
    before = session.snapshot()
    source = tmp_path / "broken.femproj"
    source.write_text('{"schema": 1, "source": "native"}', encoding="utf-8")

    with pytest.raises(ProjectDecodeError):
        load_project(source)

    assert session.snapshot() == before


def test_main_window_opens_current_fempy_project(tmp_path, monkeypatch) -> None:
    _application()
    source = save_project(
        tmp_path / "current.fempy",
        _native_project_snapshot(),
    )
    window = FEMMainWindow()
    open_dialog: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (
            open_dialog.append((*args, kwargs)) or (str(source), "")
        ),
    )
    monkeypatch.setattr(
        window,
        "_confirm_discard_changes",
        lambda: True,
    )

    window.open_native_project()
    _wait_for_task(window)

    assert window.document.source_kind == "native"
    assert window.document.project_path == source
    assert not window.legacy_project_extension
    assert open_dialog and "*.fempy" in str(open_dialog[0][3])
    assert "*.femproj" in str(open_dialog[0][3])
    assert window.document.geometry_recipe.name == "Plate"
    root = window.model_tree.roots[window.workspace.active_document_id]
    tree_texts = _tree_texts(root)
    assert "Steel" in tree_texts
    assert any(text.startswith("Section-1（") for text in tree_texts)
    assert "Load" in tree_texts
    assert any(text.startswith("位移-") for text in tree_texts)
    assert not window.document.dirty
    assert "compatibility migration" not in (
        window.status_panel.state_label.text()
    )
    window.close()


def test_main_window_opens_wire_project_and_projects_preview(tmp_path) -> None:
    _application()
    recipe = WireGeometry(
        "Wire",
        (WirePoint("P1", 0.0, 0.0), WirePoint("P2", 1.0, 0.0)),
        (WireMember("M1", "P1", "P2"),),
    )
    source = save_project(
        tmp_path / "wire.femproj",
        ProjectSnapshot(
            source_kind="native",
            parts=(NativePart(),),
            geometry_recipe=recipe,
            mesh_settings=MeshSettings(
                0.5,
                cell_shape="line",
                line_element_type="Truss2",
            ),
            feature_history=derive_feature_history(recipe),
        ),
    )
    window = FEMMainWindow()
    receipt = window.open_project_path(source)
    await_succeeded(receipt, timeout=2.0)

    assert receipt.diagnostic is None
    assert window.document.source_kind == "native"
    assert window.document.geometry_recipe == recipe
    assert window.viewport._geometry_preview is not None
    assert window.viewport._geometry_preview.dimension == 1
    window.close()


def test_main_window_v1_open_then_save_migrates_to_fempy(
    tmp_path,
    monkeypatch,
) -> None:
    _application()
    fixture = (
        Path(__file__).parents[1]
        / "fixtures"
        / "femproj"
        / "v1"
        / "minimal_rectangle.femproj"
    )
    source = tmp_path / "legacy.femproj"
    source.write_bytes(fixture.read_bytes())
    original = source.read_bytes()
    window = FEMMainWindow()
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(source), ""),
    )
    target = tmp_path / "legacy-migrated.fempy"
    save_dialog: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (
            save_dialog.append((*args, kwargs)) or (str(target), "")
        ),
    )
    monkeypatch.setattr(
        window,
        "_confirm_discard_changes",
        lambda: True,
    )
    save_successes: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        window,
        "_show_save_success",
        lambda content_name, path: save_successes.append(
            (content_name, Path(path))
        ),
    )

    window.open_native_project()
    _wait_for_task(window)

    assert window.document.project_path == source
    assert window.legacy_project_extension
    assert not window.document.dirty
    assert source.read_bytes() == original
    upgrade_notice = window.status_panel.state_label.text()
    assert "下次显式保存" in upgrade_notice
    assert "schema 10" in upgrade_notice
    assert "v10" in upgrade_notice

    assert window.save_native_project()
    _wait_for_task(window)
    assert save_dialog and save_dialog[0][2] == "legacy.fempy"
    assert source.read_bytes() == original
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8"))["schema"] == 14
    assert load_project(target).source_schema == 14
    assert window.document.project_path == target
    assert not window.legacy_project_extension
    assert not window.document.dirty
    assert save_successes == [("模型", target)]
    window.close()


def test_meshed_project_builds_display_geometry_off_gui_thread(
    tmp_path,
    monkeypatch,
) -> None:
    _application()
    model = FEMModel(
        Mesh2D(
            [
                Node2D(1, 0.0, 0.0),
                Node2D(2, 1.0, 0.0),
                Node2D(3, 0.0, 1.0),
            ],
            [Element2D(1, [1, 2, 3], "Tri3")],
        ),
        name="Meshed",
    )
    source = save_project(
        tmp_path / "meshed.fempy",
        ProjectSnapshot(model=model, model_name="Meshed"),
    )
    gui_thread = threading.get_ident()
    geometry_threads: list[int] = []
    original_builder = main_window_module.build_model_geometry

    def build_geometry(candidate):
        geometry_threads.append(threading.get_ident())
        return original_builder(candidate)

    monkeypatch.setattr(
        main_window_module,
        "build_model_geometry",
        build_geometry,
    )
    window = FEMMainWindow()

    await_succeeded(window.open_project_path(source), timeout=2.0)

    assert len(geometry_threads) == 1
    assert geometry_threads[0] != gui_thread
    assert window.document.model is not None
    assert window.geometry is not None
    window.close()


def test_legacy_project_save_cancel_preserves_document_and_source(
    tmp_path,
    monkeypatch,
) -> None:
    _application()
    fixture = (
        Path(__file__).parents[1]
        / "fixtures"
        / "femproj"
        / "v1"
        / "minimal_rectangle.femproj"
    )
    source = tmp_path / "legacy-cancel.femproj"
    source.write_bytes(fixture.read_bytes())
    original = source.read_bytes()
    window = FEMMainWindow()
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(source), ""),
    )
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)

    window.open_native_project()
    _wait_for_task(window)
    before_document = window.document
    before_session_id = window.session.session_id
    before_session_revision = window.session.session_revision

    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: ("", ""),
    )

    assert not window.save_native_project()
    assert not window.busy
    assert window.document is before_document
    assert window.session.session_id == before_session_id
    assert window.session.session_revision == before_session_revision
    assert window.document.project_path == source
    assert window.legacy_project_extension
    assert not window.document.dirty
    assert source.read_bytes() == original
    window.close()


def test_main_window_save_failure_keeps_project_dirty(
    tmp_path,
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        SketchGeometry(
            "Plate",
            (
                SketchRectangle(
                    "material",
                    0.0,
                    0.0,
                    2.0,
                    1.0,
                ),
            ),
        ),
        "草图",
    )
    target = tmp_path / "failed.femproj"
    errors = []
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    monkeypatch.setattr(
        main_window_module,
        "save_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("write failed")
        ),
    )
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )

    assert window.save_native_project()
    _wait_for_task(window)
    assert window.document.dirty
    assert window.document.project_path is None
    assert not target.exists()
    assert not target.with_suffix(".fempy").exists()
    assert errors and "write failed" in errors[-1][1]
    window.close()
