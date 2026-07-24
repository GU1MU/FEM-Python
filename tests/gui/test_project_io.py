from __future__ import annotations

from fem.core.model import (
    DisplacementConstraint,
    GravityLoad,
    MaterialDefinition,
    NodalLoad,
)
from fem.steps.factory import static
from fem_gui.document import FEMDocument, RegionAssignment, SectionDefinition
from fem_gui.preprocessing import MeshSettings, SketchCircle, SketchGeometry, SketchRectangle
from fem_gui.project_io import load_native_project, save_native_project


def test_native_project_round_trip_preserves_editable_workflow_definitions(tmp_path):
    document = FEMDocument()
    recipe = SketchGeometry(
        "Plate",
        (
            SketchRectangle("material", 0.0, 0.0, 100.0, 50.0),
            SketchCircle("cut", 50.0, 25.0, 8.0),
        ),
    )
    document.begin_native_model(recipe)
    document.set_mesh_settings(MeshSettings(5.0, order=2, cell_shape="quadrilateral"))
    document.material_definitions = [MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3})]
    document.section_definitions = [SectionDefinition("Section-1", "Steel", properties={"thickness": 2.0})]
    document.region_assignments = [RegionAssignment("Section-1", "DOMAIN")]
    step = static("Load")
    step.boundaries = (DisplacementConstraint("LEFT", 1, 2, 0.0),)
    step.cloads = (NodalLoad("RIGHT", 1, 100.0),)
    step.gravity_loads = (GravityLoad((0.0, -9.81)),)
    document.analysis_definitions = [step]
    assert document.dirty
    target = save_native_project(tmp_path / "plate.femproj", document)
    assert not document.dirty

    reopened = FEMDocument()
    load_native_project(target, reopened)

    assert reopened.source_kind == "native"
    assert reopened.geometry_recipe == recipe
    assert reopened.mesh_settings.cell_shape == "quadrilateral"
    assert reopened.material_definitions[0].name == "Steel"
    assert reopened.section_definitions[0].properties["thickness"] == 2.0
    assert reopened.region_assignments[0].region_name == "DOMAIN"
    assert reopened.analysis_definitions[0].cloads[0].value == 100.0
    assert reopened.analysis_definitions[0].gravity_loads == (
        GravityLoad((0.0, -9.81)),
    )
    assert not reopened.has_model
    assert not reopened.dirty
    assert reopened.workflow.reason == "项目已打开，请重新生成网格后检查模型"
