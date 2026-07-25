from __future__ import annotations

import os
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from fem.application import NamedRegion, RegionAssignment, SectionDefinition
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    MaterialDefinition,
    NodalLoad,
)
from fem.geometry.recipes import SketchGeometry, SketchRectangle
from fem.mesh.settings import MeshSettings
from fem_gui.main_window import FEMMainWindow


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_task(window: FEMMainWindow, timeout: float = 30.0) -> None:
    deadline = monotonic() + timeout
    application = _application()
    while window.busy and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not window.busy


def test_native_authoring_mesh_check_solve_then_clear_fully_invalidates(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )

    window.new_native_model()
    assert window.document.source_kind == "native"
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
    assert window._apply_session_delta(
        window.session.replace_named_regions(
            (
                NamedRegion("Fixed", "edge", (4,)),
                NamedRegion("Loaded", "edge", (2,)),
            )
        )
    )
    assert window._apply_session_delta(
        window.session.replace_mesh_settings(MeshSettings(0.5))
    )
    step = AnalysisStep(
        "Load",
        boundaries=(DisplacementConstraint("Fixed", 1, 2, 0.0),),
        cloads=(NodalLoad("Loaded", 1, 10.0),),
    )
    assert window._apply_session_delta(
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
        )
    )

    window.generate_native_mesh()
    _wait_for_task(window)

    assert errors == []
    assert window.document.artifact is not None
    assert window.document.artifact.source_kind == "native"
    assert window.geometry.artifact_id == window.document.artifact.artifact_id
    assert window.viewport.artifact_id == window.document.artifact.artifact_id
    assert window.document.model.materials["Steel"].properties["E"] == 210000.0
    assert window.document.model.sections[0].element_set == "DOMAIN"
    assert window.document.model.steps[0].boundaries[0].target == "Fixed"
    assert window.document.model.steps[0].cloads[0].target == "Loaded"

    assert window.check_current_model(show_success=False), errors
    assert window.session.can_submit("Load")
    run = window._submit_job("Job-1", "Load")
    assert run is not None
    _wait_for_task(window)

    assert errors == []
    result = window.session.current_result()
    assert result is not None
    assert result.provenance.run_id == run.run_id
    assert window.result_data.run_id == run.run_id
    assert window.viewport.run_id == run.run_id
    assert window.actions["query"].isEnabled()

    window.delete_geometry()

    assert window.document.geometry_recipe is None
    assert not window.document.named_regions
    assert window.document.region_assignments == ()
    assert window.document.analysis_definitions == ()
    assert window.document.artifact is None
    assert not window.document.validations
    assert window.document.runs == ()
    assert window.session.current_result() is None
    assert window.geometry is None
    assert window.result_data is None
    assert window.viewport.artifact_id is None
    assert window.viewport.run_id is None
    assert window.inspection_service is None
    assert window.result_tree.topLevelItem(0).text(0) == "尚无分析结果"
    for name in (
        "undeformed",
        "deformed",
        "contour",
        "field",
        "query",
        "export",
    ):
        assert not window.actions[name].isEnabled()
    window.close()
