from __future__ import annotations

import pytest

from fem.application import (
    FeatureRecord,
    ModelSession,
    NamedRegion,
    NativePart,
    ProjectSnapshot,
    RegionAssignment,
    SectionDefinition,
)
from fem.core.model import (
    DisplacementConstraint,
    GravityLoad,
    LineLoad,
    MaterialDefinition,
    NodalLoad,
)
from fem.geometry.recipes import (
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
)
from fem.io.project_v1 import load_project_v1, save_project_v1
from fem.mesh.settings import LocalMeshControl, MeshSettings
from fem.steps.factory import static
from fem_gui import project_io


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
    step.line_loads = (LineLoad("TOP", (0.0, -5.0)),)
    step.gravity_loads = (GravityLoad((0.0, -9.81)),)
    return ProjectSnapshot(
        source_kind="native",
        parts=(NativePart(),),
        geometry_recipe=recipe,
        mesh_settings=MeshSettings(
            5.0,
            order=2,
            cell_shape="quadrilateral",
            local_size=2.5,
            local_controls=(LocalMeshControl("edge", 1, 1.0),),
        ),
        feature_history=(FeatureRecord("Sketch-1", "sketch"),),
        named_regions=(NamedRegion("TOP", "edge", (1,)),),
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


def test_gui_project_io_is_only_a_canonical_codec_facade() -> None:
    assert project_io.load_native_project is load_project_v1
    assert project_io.load_project_v1 is load_project_v1
    assert project_io.save_native_project is save_project_v1
    assert project_io.save_project_v1 is save_project_v1


def test_native_project_round_trip_returns_a_detached_snapshot(tmp_path) -> None:
    original = _native_project_snapshot()

    target = project_io.save_native_project(
        tmp_path / "plate.femproj",
        original,
    )
    reopened = project_io.load_native_project(target)

    assert isinstance(reopened, ProjectSnapshot)
    assert reopened.source_kind == "native"
    assert reopened.source_path == target
    assert reopened.geometry_recipe == original.geometry_recipe
    assert reopened.mesh_settings == original.mesh_settings
    assert reopened.material_definitions[0].name == "Steel"
    assert reopened.section_definitions[0].properties["thickness"] == 2.0
    assert reopened.region_assignments[0].region_name == "DOMAIN"
    assert reopened.analysis_definitions[0].cloads[0].value == 100.0
    assert reopened.analysis_definitions[0].line_loads == (
        LineLoad("TOP", (0.0, -5.0)),
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

    with pytest.raises(project_io.ProjectV1DecodeError):
        project_io.load_native_project(source)

    assert session.snapshot() == before
