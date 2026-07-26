from __future__ import annotations

import os
from pathlib import Path
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from fem.abaqus import read
from fem.application import NamedRegion, RegionAssignment, SectionDefinition
from fem.core.model import DisplacementConstraint, MaterialDefinition, NodalLoad
from fem.geometry import LogicalEntityRef
from fem.steps.factory import static
from fem_gui.main_window import FEMMainWindow
from fem_gui.preprocessing import MeshSettings, SketchGeometry, SketchRectangle
from fem_gui.visualization.model_adapter import build_model_geometry


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait(window: FEMMainWindow) -> None:
    deadline = monotonic() + 15.0
    while window.busy and monotonic() < deadline:
        _application().processEvents()
        QThread.msleep(1)
    _application().processEvents()
    assert not window.busy


def _apply(window: FEMMainWindow, delta: object) -> None:
    assert window._apply_session_delta(delta)


def test_native_preprocess_check_job_result_workflow(monkeypatch):
    _application()
    window = FEMMainWindow()
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "_show_error", lambda title, message: errors.append((title, message)))
    window._set_native_geometry(
        SketchGeometry("Plate", (SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),)),
        "草图",
    )
    _apply(
        window,
        window.session.replace_named_regions(
            (
                NamedRegion(
                    "Fixed",
                    (LogicalEntityRef("edge:left"),),
                ),
                NamedRegion(
                    "Loaded",
                    (LogicalEntityRef("edge:right"),),
                ),
            )
        ),
    )
    step = static("Load")
    step.boundaries = (DisplacementConstraint("Fixed", 1, 2, 0.0),)
    step.cloads = (NodalLoad("Loaded", 1, 10.0),)
    _apply(
        window,
        window.session.replace_model_definitions(
            (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
            (
                SectionDefinition(
                    "Section-1",
                    "Steel",
                    properties={"thickness": 1.0},
                ),
            ),
            (RegionAssignment("Section-1", "DOMAIN"),),
            (step,),
        ),
    )
    _apply(
        window,
        window.session.replace_mesh_settings(MeshSettings(0.5)),
    )
    window.generate_native_mesh()
    _wait(window)

    model = window.document.model
    assert model is not None
    assert model.materials["Steel"].properties["E"] == 210000.0
    assert model.sections[0].element_set == "DOMAIN"
    assert model.steps[0].boundaries[0].target == "Fixed"
    assert model.steps[0].cloads[0].target == "Loaded"
    assert window.check_current_model(show_success=False), errors
    assert window.actions["submit_job"].isEnabled()
    assert window._submit_job("Job-1", "Load") is not None
    _wait(window)

    assert window.document.has_result
    current_result = window.session.current_result()
    assert current_result is not None
    assert (
        current_result.provenance.artifact_id
        == window.document.artifact.artifact_id
    )
    assert (
        current_result.provenance.model_revision
        == window.document.model_revision
    )
    assert window.result_data is not None and "U" in window.result_data.fields
    window.close()


def test_inp_check_job_result_workflow(monkeypatch, gui_inp_path):
    _application()
    window = FEMMainWindow()
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "_show_error", lambda title, message: errors.append((title, message)))
    model = read(gui_inp_path)
    window._model_loaded(Path(gui_inp_path), (model, build_model_geometry(model)))

    assert window.document.source_kind == "imported"
    assert not window.actions["submit_job"].isEnabled()
    assert window.check_current_model(show_success=False), errors
    assert window._submit_job("Job-1", "Static-1") is not None
    _wait(window)

    assert window.document.has_result
    current_result = window.session.current_result()
    assert current_result is not None
    assert (
        current_result.provenance.artifact_id
        == window.document.artifact.artifact_id
    )
    window.close()


def test_inp_check_accepts_importer_internal_section_element_set(monkeypatch):
    _application()
    window = FEMMainWindow()
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )
    path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "inp"
        / "internal_section_set.inp"
    )
    model = read(path)
    window._model_loaded(path, (model, build_model_geometry(model)))

    assert model.sections[0].element_set.startswith("_section_")
    assert window.check_current_model(show_success=False), errors
    assert errors == []
    window.close()


def test_model_check_rejects_an_underconstrained_native_model(monkeypatch):
    _application()
    window = FEMMainWindow()
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )
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
    _apply(
        window,
        window.session.replace_named_regions(
            (
                NamedRegion(
                    "Fixed",
                    (LogicalEntityRef("edge:left"),),
                ),
            )
        ),
    )
    step = static("Load")
    step.boundaries = (
        DisplacementConstraint("Fixed", 1, 1, 0.0),
    )
    _apply(
        window,
        window.session.replace_model_definitions(
            (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
            (
                SectionDefinition(
                    "Section-1",
                    "Steel",
                    properties={
                        "plane_type": "stress",
                        "thickness": 1.0,
                    },
                ),
            ),
            (RegionAssignment("Section-1", "DOMAIN"),),
            (step,),
        ),
    )
    _apply(
        window,
        window.session.replace_mesh_settings(MeshSettings(0.5)),
    )
    window.generate_native_mesh()
    _wait(window)

    assert not window.check_current_model(show_success=False)
    assert errors
    assert "约束不足或刚度矩阵奇异" in errors[-1][1]
    assert not window.session.can_submit("Load")
    window.close()
