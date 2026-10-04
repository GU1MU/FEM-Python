from __future__ import annotations

from fem.application import (
    MeshEntityRef,
    ModelSession,
    NamedRegion,
    RegionAssignment,
    ScopedDefinitionBatch,
    SectionDefinition,
    UnitContext,
)
from fem.application.native_scope_materialization import (
    NATIVE_PART_OWNERSHIP_KEY,
    NATIVE_SCOPE_CATALOG_KEY,
)
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    EdgeLoad,
    FEMModel,
    MaterialDefinition,
    OutputRequest,
)
from fem.geometry import PlateWithHoleGeometry, RectangleGeometry
from fem.mesh.settings import MeshSettings
from fem.selection import edges as mesh_edges


def make_defined_plate_session(unit_context: UnitContext | None = None) -> ModelSession:
    """Two Tri3 elements with regions, material and section; no analysis step."""
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Model-板",
        unit_context or UnitContext("mm", "N", "MPa"),
        RectangleGeometry("Body-板", 10.0, 4.0),
        part_name="Part-板",
    )
    model = FEMModel(
        Mesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 10.0, 0.0),
                Node2D(3, 10.0, 4.0),
                Node2D(4, 0.0, 4.0),
            ],
            elements=[
                Element2D(1, (1, 2, 3), "Tri3"),
                Element2D(2, (1, 3, 4), "Tri3"),
            ],
        ),
        name="Model-板",
        metadata={
            NATIVE_PART_OWNERSHIP_KEY: {
                "P1": {
                    "node_ids": (1, 2, 3, 4),
                    "element_ids": (1, 2),
                }
            }
        },
    )
    task = session.prepare_agent_mesh_generation(
        "P1",
        MeshSettings(1.0),
        "a" * 64,
        expected_session_revision=session.session_revision,
    )
    assert session.accept_agent_generated_model(task.token, model).accepted
    session.apply_scoped_definition_batch(
        ScopedDefinitionBatch(
            session.session_revision,
            (
                NamedRegion(
                    "Edge-固定端",
                    (MeshEntityRef.edge(2, 2, (4, 1), part_id="P1"),),
                ),
                NamedRegion(
                    "Edge-加载端",
                    (MeshEntityRef.edge(1, 1, (2, 3), part_id="P1"),),
                ),
                NamedRegion(
                    "Domain-板体",
                    (
                        MeshEntityRef.element(1, part_id="P1"),
                        MeshEntityRef.element(2, part_id="P1"),
                    ),
                ),
            ),
            (MaterialDefinition("Material-钢", {"E": 210000.0, "nu": 0.3}),),
            (
                SectionDefinition(
                    "Section-平面应力",
                    "Material-钢",
                    "solid",
                    {"plane_type": "stress", "thickness": 1.0},
                ),
            ),
            (RegionAssignment("Section-平面应力", "Domain-板体"),),
            (),
        )
    )
    return session


def make_plate_static_step() -> AnalysisStep:
    return AnalysisStep(
        "Step-静力",
        procedure="static",
        boundaries=(
            DisplacementConstraint(
                "Edge-固定端", 1, 2, 0.0, "edge", "Displacement-固定端",
            ),
        ),
        edge_loads=(
            EdgeLoad("Edge-加载端", (10.0, 0.0), None, "traction", "Load-拉伸"),
        ),
        outputs=(
            OutputRequest("field", "node", ("U", "RF"), name="Output-位移反力"),
            OutputRequest("field", "element", ("S",), name="Output-应力"),
        ),
        metadata={"nlgeom": False},
    )


def make_eccentric_plate_recipe() -> PlateWithHoleGeometry:
    return PlateWithHoleGeometry(
        "Body-偏心孔板",
        10.0,
        6.0,
        6.5,
        2.0,
        1.0,
    )


def make_eccentric_plate_model() -> FEMModel:
    """Eight Tri3 elements with outer/hole boundary catalog and part ownership."""
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 10.0, 0.0),
            Node2D(3, 10.0, 6.0),
            Node2D(4, 0.0, 6.0),
            Node2D(5, 7.5, 2.0),
            Node2D(6, 6.5, 3.0),
            Node2D(7, 5.5, 2.0),
            Node2D(8, 6.5, 1.0),
        ],
        elements=[
            Element2D(1, [1, 2, 8], "Tri3"),
            Element2D(2, [2, 5, 8], "Tri3"),
            Element2D(3, [2, 3, 5], "Tri3"),
            Element2D(4, [3, 6, 5], "Tri3"),
            Element2D(5, [3, 4, 6], "Tri3"),
            Element2D(6, [4, 7, 6], "Tri3"),
            Element2D(7, [4, 1, 7], "Tri3"),
            Element2D(8, [1, 8, 7], "Tri3"),
        ],
    )
    boundary = tuple(mesh_edges.boundary(mesh))
    outer_nodes = {1, 2, 3, 4}
    hole_nodes = {5, 6, 7, 8}

    def rows(node_ids: set[int]):
        return tuple(
            (
                element_id,
                local_index,
                tuple(edge_node_ids),
            )
            for element_id, local_index, edge_node_ids in boundary
            if set(edge_node_ids).issubset(node_ids)
        )

    catalog = {
        "edge:P1/outer-loop": {
            "kind": "edge",
            "node_ids": tuple(sorted(outer_nodes)),
            "element_ids": (),
            "edges": rows(outer_nodes),
            "faces": (),
        },
        "edge:P1/hole-loop": {
            "kind": "edge",
            "node_ids": tuple(sorted(hole_nodes)),
            "element_ids": (),
            "edges": rows(hole_nodes),
            "faces": (),
        },
        "edge:P1/bottom": {
            "kind": "edge",
            "node_ids": (1, 2),
            "element_ids": (),
            "edges": rows({1, 2}),
            "faces": (),
        },
        "edge:P1/right": {
            "kind": "edge",
            "node_ids": (2, 3),
            "element_ids": (),
            "edges": rows({2, 3}),
            "faces": (),
        },
        "edge:P1/top": {
            "kind": "edge",
            "node_ids": (3, 4),
            "element_ids": (),
            "edges": rows({3, 4}),
            "faces": (),
        },
        "edge:P1/left": {
            "kind": "edge",
            "node_ids": (1, 4),
            "element_ids": (),
            "edges": rows({1, 4}),
            "faces": (),
        },
        "face:P1/domain": {
            "kind": "face",
            "node_ids": tuple(range(1, 9)),
            "element_ids": tuple(range(1, 9)),
            "edges": (),
            "faces": (),
        },
    }
    return FEMModel(
        mesh,
        name="Model-偏心孔板",
        metadata={
            NATIVE_SCOPE_CATALOG_KEY: catalog,
            NATIVE_PART_OWNERSHIP_KEY: {
                "P1": {
                    "node_ids": tuple(range(1, 9)),
                    "element_ids": tuple(range(1, 9)),
                }
            },
        },
    )


def make_meshed_eccentric_plate_session() -> ModelSession:
    """Install the hole plate mesh without material or analysis definitions."""
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Model-偏心孔板",
        UnitContext("mm", "N", "MPa"),
        make_eccentric_plate_recipe(),
        part_name="Part-偏心孔板",
    )
    task = session.prepare_agent_mesh_generation(
        "P1",
        MeshSettings(1.0),
        "a" * 64,
        expected_session_revision=session.session_revision,
    )
    assert session.accept_agent_generated_model(
        task.token,
        make_eccentric_plate_model(),
    ).accepted
    return session
