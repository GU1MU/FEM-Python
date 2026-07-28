from __future__ import annotations

import pytest

from fem.application import (
    MeshEntityRef,
    ModelSession,
    NamedRegion,
    NamedRegionEditBatch,
    NativePart,
    RegionAssignment,
    SectionDefinition,
)
from fem.application.native_scope_materialization import (
    materialize_native_scopes,
    mesh_references_for_logical_entities,
)
from fem.application.preprocessing import generate_fem_model
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    MaterialDefinition,
)
from fem.geometry import (
    BoxGeometry,
    LogicalEntityRef,
    RectangleGeometry,
    WireGeometry,
    WireMember,
    WirePoint,
)
from fem.mesh.settings import MeshSettings
from fem.selection import edges as mesh_edges
from fem.selection import faces as mesh_faces


@pytest.mark.gmsh
def test_rectangle_scope_materializes_on_the_existing_mesh() -> None:
    recipe = RectangleGeometry("ScopeRectangle", 2.0, 1.0)
    model = generate_fem_model(recipe, MeshSettings(0.25))
    before_nodes = tuple(model.mesh.nodes)
    before_elements = tuple(model.mesh.elements)
    edge = mesh_edges.boundary(model.mesh)[0]

    updated = materialize_native_scopes(
        model,
        previous_names=(),
        regions=(
            NamedRegion(
                "Boundary",
                (MeshEntityRef.edge(*edge),),
            ),
        ),
    )

    assert tuple(updated.mesh.nodes) == before_nodes
    assert tuple(updated.mesh.elements) == before_elements
    assert updated.edges["Boundary"].edges
    assert "Boundary" not in model.node_sets
    assert "Boundary" not in model.edges
    assert not updated.node_sets
    assert not updated.element_sets


@pytest.mark.gmsh
def test_native_catalog_expands_one_geometry_edge_to_its_mesh_chain() -> None:
    recipe = RectangleGeometry("ScopeCatalogRectangle", 2.0, 1.0)
    model = generate_fem_model(recipe, MeshSettings(0.25))

    references = mesh_references_for_logical_entities(
        model,
        (LogicalEntityRef("edge:bottom"),),
        mesh_kind="edge",
    )

    assert len(references) > 1
    assert {reference.kind for reference in references} == {"edge"}
    assert not model.node_sets
    assert not model.element_sets
    assert not model.edges
    assert not model.surfaces


@pytest.mark.gmsh
def test_imported_mesh_can_add_a_mesh_scope_without_remeshing(
    tmp_path,
) -> None:
    model = generate_fem_model(
        RectangleGeometry("ImportedScope", 2.0, 1.0),
        MeshSettings(0.25),
    )
    model.metadata.clear()
    edge = mesh_edges.boundary(model.mesh)[0]
    session = ModelSession()
    task = session.prepare_import(tmp_path / "imported.inp")
    session.accept_imported_model(task.token, model)
    before = session.snapshot()

    session.apply_named_region_edit(
        NamedRegionEditBatch(
            before.session_revision,
            (
                NamedRegion(
                    "ImportedEdge",
                    (MeshEntityRef.edge(*edge),),
                ),
            ),
        )
    )
    after = session.snapshot()

    assert after.source_kind == "imported"
    assert after.mesh_input_revision == before.mesh_input_revision
    assert tuple(after.model.mesh.nodes) == tuple(before.model.mesh.nodes)
    assert tuple(after.model.mesh.elements) == tuple(
        before.model.mesh.elements
    )
    assert "ImportedEdge" in after.model.edges


@pytest.mark.gmsh
def test_box_face_scope_materializes_nodes_and_element_faces() -> None:
    recipe = BoxGeometry("ScopeBox", 2.0, 1.0, 0.5)
    model = generate_fem_model(
        recipe,
        MeshSettings(0.4, cell_shape="tetrahedron"),
    )
    face = mesh_faces.boundary(model.mesh)[0]

    updated = materialize_native_scopes(
        model,
        previous_names=(),
        regions=(
            NamedRegion(
                "LoadedFace",
                (MeshEntityRef.face(*face),),
            ),
        ),
    )

    assert updated.surfaces["LoadedFace"].faces
    assert not updated.node_sets
    assert "LoadedFace" not in updated.element_sets
    assert "LoadedFace" not in updated.edges


@pytest.mark.gmsh
def test_wire_point_and_member_scopes_use_the_existing_line_mesh() -> None:
    recipe = WireGeometry(
        "ScopeWire",
        (
            WirePoint("P1", 0.0, 0.0),
            WirePoint("P2", 1.0, 0.0),
        ),
        (WireMember("M1", "P1", "P2"),),
    )
    model = generate_fem_model(
        recipe,
        MeshSettings(
            0.25,
            cell_shape="line",
            line_element_type="Truss2",
        ),
    )
    node_id = int(model.mesh.nodes[0].id)
    element_id = int(model.mesh.elements[0].id)

    updated = materialize_native_scopes(
        model,
        previous_names=(),
        regions=(
            NamedRegion(
                "Joint",
                (MeshEntityRef.node(node_id),),
            ),
            NamedRegion(
                "Member",
                (MeshEntityRef.element(element_id),),
            ),
        ),
    )

    assert updated.node_sets["Joint"].node_ids
    assert updated.element_sets["Member"].element_ids
    assert "Joint" not in updated.element_sets
    assert not updated.edges
    assert not updated.surfaces


@pytest.mark.gmsh
def test_remeshing_invalidates_mesh_scopes_and_their_dependents() -> None:
    recipe = RectangleGeometry("RemeshScope", 2.0, 1.0)
    settings = MeshSettings(0.25)
    model = generate_fem_model(recipe, settings)
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry((NativePart(),), recipe)
    session.replace_mesh_settings(settings)
    task = session.prepare_mesh_generation()
    session.accept_generated_model(task.token, model)
    before_scope_edit = session.snapshot()
    regions = (
        NamedRegion(
            "Pinned",
            (MeshEntityRef.node(model.mesh.nodes[0].id),),
        ),
        NamedRegion(
            "Domain",
            tuple(
                MeshEntityRef.element(element.id)
                for element in model.mesh.elements
            ),
        ),
    )
    session.replace_named_regions(regions)
    after_scope_edit = session.snapshot()

    assert (
        after_scope_edit.mesh_input_revision
        == before_scope_edit.mesh_input_revision
    )
    assert after_scope_edit.artifact is not None
    assert (
        after_scope_edit.artifact.mesh_input_revision
        == before_scope_edit.mesh_input_revision
    )
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
        (SectionDefinition("Plate", "Steel", properties={"thickness": 1.0}),),
        (RegionAssignment("Plate", "Domain"),),
        (
            AnalysisStep(
                "Load",
                boundaries=(
                    DisplacementConstraint("Pinned", 1, 2, 0.0),
                ),
            ),
        ),
    )

    delta = session.replace_mesh_settings(MeshSettings(0.2))
    snapshot = session.snapshot()

    assert not snapshot.named_regions
    assert not snapshot.assignments
    assert not snapshot.steps
    assert snapshot.model is None
    assert snapshot.artifact is None
    assert delta.effects
