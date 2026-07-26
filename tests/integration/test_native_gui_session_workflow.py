"""Public-command characterization of the native GUI workflow."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.application import (
    DefinitionEditBatch,
    NamedRegion,
    NamedRegionEditBatch,
    NativePart,
    RegionAssignment,
    SectionDefinition,
)
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    MaterialDefinition,
    NodalLoad,
)
from fem.geometry.recipes import SketchGeometry, SketchRectangle
from fem.geometry.references import LogicalEntityRef
from fem.mesh.settings import MeshSettings
from fem_gui.commands import (
    CloseSessionCommand,
    MeshInputEdit,
    NativeGeometryEdit,
    NewNativeProjectCommand,
)
from fem_gui.main_window import FEMMainWindow
from tests.helpers.gui_command_receipts import (
    await_succeeded,
    require_accepted,
)


PUBLIC_GUI_WORKFLOW_ENTRYPOINTS = (
    "new_native_project",
    "apply_native_geometry_edit",
    "apply_named_region_edit",
    "apply_mesh_input_edit",
    "apply_definition_edit",
    "generate_mesh",
    "check_step",
    "submit_run",
    "save_project_path",
    "close_session",
    "open_project_path",
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _install_native_authoring(window: FEMMainWindow) -> None:
    recipe = SketchGeometry(
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
    )
    require_accepted(
        window.apply_native_geometry_edit(
            NativeGeometryEdit(
                base_session_revision=window.document.session_revision,
                parts=(NativePart("Plate", "Body"),),
                recipe=recipe,
            )
        )
    )
    require_accepted(
        window.apply_named_region_edit(
            NamedRegionEditBatch(
                base_session_revision=window.document.session_revision,
                regions=(
                    NamedRegion(
                        "Fixed",
                        (LogicalEntityRef("edge:left"),),
                    ),
                    NamedRegion(
                        "Loaded",
                        (LogicalEntityRef("edge:right"),),
                    ),
                ),
            )
        )
    )
    require_accepted(
        window.apply_mesh_input_edit(
            MeshInputEdit(
                base_session_revision=window.document.session_revision,
                settings=MeshSettings(0.5),
            )
        )
    )
    step = AnalysisStep(
        "Load",
        boundaries=(DisplacementConstraint("Fixed", 1, 2, 0.0),),
        cloads=(NodalLoad("Loaded", 1, 10.0),),
    )
    require_accepted(
        window.apply_definition_edit(
            DefinitionEditBatch(
                base_session_revision=window.document.session_revision,
                materials=(
                    MaterialDefinition(
                        "Steel",
                        {"E": 210000.0, "nu": 0.3},
                    ),
                ),
                sections=(
                    SectionDefinition(
                        "Section-1",
                        "Steel",
                        properties={
                            "plane_type": "stress",
                            "thickness": 1.0,
                        },
                    ),
                ),
                assignments=(
                    RegionAssignment("Section-1", "DOMAIN"),
                ),
                steps=(step,),
            )
        )
    )


def _mesh_check_and_solve(
    window: FEMMainWindow,
    *,
    run_name: str,
) -> str:
    await_succeeded(window.generate_mesh())
    artifact = window.document.artifact
    assert artifact is not None and artifact.source_kind == "native"
    assert window.geometry is not None
    assert window.geometry.artifact_id == artifact.artifact_id
    assert window.viewport.artifact_id == artifact.artifact_id
    assert window.actions["check_model"].isEnabled()

    await_succeeded(window.check_step("Load"))
    assert window.document.validation_current("Load")
    assert window.actions["submit_job"].isEnabled()

    await_succeeded(window.submit_run(run_name, "Load"))
    run = window.session.find_run(run_name)
    assert run is not None and run.has_result
    result = window.session.current_result()
    assert result is not None
    assert result.provenance.run_id == run.run_id
    assert result.provenance.artifact_id == artifact.artifact_id
    assert window.result_data is not None
    assert window.result_data.run_id == run.run_id
    assert window.result_data.field_ready("U")
    assert window.viewport.run_id == run.run_id
    assert window.actions["query"].isEnabled()
    return run.run_id


def test_native_public_workflow_saves_reopens_remeshes_and_resolves(
    tmp_path,
) -> None:
    _application()
    window = FEMMainWindow()

    require_accepted(
        window.new_native_project(NewNativeProjectCommand("Plate Project"))
    )
    assert window.document.source_kind == "native"
    assert window.actions["geometry_sketch"].isEnabled()

    _install_native_authoring(window)
    assert window.document.dirty
    assert window.actions["mesh_generate"].isEnabled()
    first_run_id = _mesh_check_and_solve(window, run_name="Job-1")

    project_path = tmp_path / "plate-public.femproj"
    require_accepted(window.save_project_path(project_path))
    assert project_path.is_file()
    assert window.document.project_path == project_path
    assert not window.document.dirty

    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    assert window.document.source_kind is None
    assert window.result_data is None
    assert window.viewport.run_id is None

    require_accepted(window.open_project_path(project_path))
    assert window.document.source_kind == "native"
    assert window.document.project_path == project_path
    assert window.document.geometry_recipe is not None
    assert window.document.mesh_settings == MeshSettings(0.5)
    assert window.document.runs == ()
    assert window.session.current_result() is None

    reopened_run_id = _mesh_check_and_solve(
        window,
        run_name="Job-Reopened",
    )
    assert reopened_run_id != first_run_id

    require_accepted(
        window.close_session(
            CloseSessionCommand(window.document.session_revision)
        )
    )
    window.close()
