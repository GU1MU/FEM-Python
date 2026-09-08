from __future__ import annotations

from fem.application import (
    MeshEntityRef,
    ModelSession,
    NamedRegion,
    ScopedDefinitionBatch,
    UnitContext,
)
from fem.application.native_scope_materialization import NATIVE_PART_OWNERSHIP_KEY
from fem.core.model import AnalysisStep, FEMModel, SurfaceLoad
from fem.geometry import BoxGeometry
from fem.mesh.settings import MeshSettings
from fem.selection import faces as mesh_faces
from tests.helpers.mesh_builders import make_selection_hex_mesh


def make_surface_load_session(unit_context: UnitContext | None = None) -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-三维块",
        unit_context or UnitContext("mm", "N", "MPa"),
        BoxGeometry("实体-三维块", 2.0, 3.0, 4.0),
        part_name="部件-三维块",
    )
    mesh = make_selection_hex_mesh()
    model = FEMModel(
        mesh,
        name="模型-三维块",
        metadata={
            NATIVE_PART_OWNERSHIP_KEY: {
                "P1": {
                    "node_ids": tuple(node.id for node in mesh.nodes),
                    "element_ids": tuple(
                        element.id for element in mesh.elements
                    ),
                }
            }
        },
    )
    task = session.prepare_agent_mesh_generation(
        "P1",
        MeshSettings(1.0, cell_shape="hexahedron"),
        "c" * 64,
        expected_session_revision=session.session_revision,
    )
    assert session.accept_agent_generated_model(task.token, model).accepted
    face = mesh_faces.boundary(mesh)[0]
    snapshot = session.snapshot()
    session.apply_scoped_definition_batch(
        ScopedDefinitionBatch(
            snapshot.session_revision,
            (
                NamedRegion(
                    "面-加载",
                    (MeshEntityRef.face(*face, part_id="P1"),),
                ),
            ),
            (),
            (),
            (),
            (
                AnalysisStep(
                    "分析步-静力",
                    surface_loads=(
                        SurfaceLoad(
                            "面-加载",
                            (1.0, 0.0, 0.0),
                            None,
                            "traction",
                            "载荷-表面",
                        ),
                    ),
                    metadata={"nlgeom": False},
                ),
            ),
        )
    )
    return session
