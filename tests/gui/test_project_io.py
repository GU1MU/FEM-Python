from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fem.application import (
    derive_feature_history,
    ModelSession,
    NamedRegion,
    NativePart,
    ProjectSnapshot,
    RegionAssignment,
    SectionDefinition,
)
from fem.geometry import LogicalEntityRef
from fem.core.model import (
    DisplacementConstraint,
    EdgeLoad,
    GravityLoad,
    MaterialDefinition,
    NodalLoad,
)
from fem.geometry.recipes import (
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
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


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


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
    assert loaded.source_schema == 2
    assert loaded.notices == ()
    assert reopened.source_kind == "native"
    assert reopened.source_path == target
    assert reopened.geometry_recipe == original.geometry_recipe
    assert reopened.mesh_settings == original.mesh_settings
    assert reopened.material_definitions[0].name == "Steel"
    assert reopened.section_definitions[0].properties["thickness"] == 2.0
    assert reopened.region_assignments[0].region_name == "DOMAIN"
    assert reopened.analysis_definitions[0].cloads[0].value == 100.0
    assert reopened.analysis_definitions[0].edge_loads == (
        EdgeLoad("TOP", (0.0, -5.0)),
    )
    assert reopened.analysis_definitions[0].gravity_loads == (
        GravityLoad((0.0, -9.81)),
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


def test_main_window_opens_current_v2_project(tmp_path, monkeypatch) -> None:
    _application()
    source = save_project(
        tmp_path / "current.femproj",
        _native_project_snapshot(),
    )
    window = FEMMainWindow()
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(source), ""),
    )
    monkeypatch.setattr(
        window,
        "_confirm_discard_changes",
        lambda: True,
    )

    window.open_native_project()

    assert window.document.source_kind == "native"
    assert window.document.project_path == source
    assert window.document.geometry_recipe.name == "Plate"
    assert not window.document.dirty
    assert "compatibility migration" not in (
        window.status_panel.state_label.text()
    )
    window.close()


def test_main_window_v1_open_then_save_upgrades_the_same_path(
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
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: pytest.fail(
            "existing v1 project path must not trigger Save As"
        ),
    )
    monkeypatch.setattr(
        window,
        "_confirm_discard_changes",
        lambda: True,
    )

    window.open_native_project()

    assert window.document.project_path == source
    assert not window.document.dirty
    assert source.read_bytes() == original
    upgrade_notice = window.status_panel.state_label.text()
    assert "下次显式保存" in upgrade_notice
    assert "schema 2" in upgrade_notice
    assert "v2" in upgrade_notice

    assert window.save_native_project()
    assert json.loads(source.read_text(encoding="utf-8"))["schema"] == 2
    assert load_project(source).source_schema == 2
    assert not window.document.dirty
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

    assert not window.save_native_project()
    assert window.document.dirty
    assert window.document.project_path is None
    assert not target.exists()
    assert errors and "write failed" in errors[-1][1]
    window.close()
