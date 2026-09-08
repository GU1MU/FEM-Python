from __future__ import annotations

from fem.application import (
    derive_feature_history,
    NamedRegion,
    NativePart,
    ProjectSnapshot,
    RegionAssignment,
    SectionDefinition,
)
from fem.geometry import LogicalEntityRef
from fem.core.model import (
    BodyForce,
    DisplacementConstraint,
    EdgeLoad,
    GravityLoad,
    MaterialDefinition,
    NodalLoad,
)
from fem.geometry.recipes import SketchCircle, SketchGeometry, SketchRectangle
from fem.mesh.settings import (
    LocalMeshControl,
    MeshSettings,
    MeshSizeFalloff,
)
from fem.steps.factory import static


def make_native_project_snapshot() -> ProjectSnapshot:
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
