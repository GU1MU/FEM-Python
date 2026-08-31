from __future__ import annotations

import pytest

from fem.application import (
    DeleteIntent,
    MeshEntityRef,
    ModelSession,
    NamedRegion,
    NamedRegionEditBatch,
    NativePart,
    RegionAssignment,
    SessionStateError,
    SectionDefinition,
)
from fem.application.native_scope_materialization import (
    materialize_native_scopes,
    mesh_references_for_logical_entities,
)
from fem.application import native_scope_materialization as scope_materialization
from fem.application.preprocessing import generate_fem_model
from fem.model import (
    AnalysisStep,
    DisplacementConstraint,
    MaterialDefinition,
    UnitContext,
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
def test_scope_edit_reuses_mesh_and_builds_edge_lookup_once(monkeypatch) -> None:
    model = generate_fem_model(
        RectangleGeometry("ScopeLookupRectangle", 2.0, 1.0),
        MeshSettings(0.25),
    )
    rows = tuple(mesh_edges.boundary(model.mesh)[:2])
    calls = 0
    original = scope_materialization.mesh_edges.all

    def counted_all(mesh):
        nonlocal calls
        calls += 1
        return original(mesh)

    monkeypatch.setattr(scope_materialization.mesh_edges, "all", counted_all)
    updated = materialize_native_scopes(
        model,
        previous_names=(),
        regions=tuple(
            NamedRegion(name, (MeshEntityRef.edge(*row),))
            for name, row in zip(("First", "Second"), rows, strict=True)
        ),
        reuse_mesh=True,
    )

    assert updated is not model
    assert updated.mesh is model.mesh
    assert updated.edges is not model.edges
    assert tuple(updated.edges) == ("First", "Second")
    assert calls == 1


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

    scoped = session.projection_snapshot()
    session.apply_named_region_edit(
        NamedRegionEditBatch(
            scoped.session_revision,
            (),
            deletes=(DeleteIntent("ImportedEdge"),),
        )
    )
    deleted = session.projection_snapshot()

    assert deleted.mesh_input_revision == scoped.mesh_input_revision
    assert deleted.model.mesh is scoped.model.mesh
    assert "ImportedEdge" not in deleted.model.edges


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
def test_remeshing_rejects_legacy_mesh_scopes_without_clearing_inputs() -> None:
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

    before_remesh = session.snapshot()
    with pytest.raises(SessionStateError, match="无法安全迁移"):
        session.replace_mesh_settings(MeshSettings(0.2))
    after_remesh = session.snapshot()

    assert after_remesh == before_remesh


@pytest.mark.gmsh
def test_remeshing_rebuilds_logical_scopes_and_preserves_definitions() -> None:
    recipe = RectangleGeometry("LogicalRemeshScope", 2.0, 1.0)
    settings = MeshSettings(0.25)
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry((NativePart(),), recipe)
    session.replace_named_regions(
        (
            NamedRegion(
                "Pinned",
                (LogicalEntityRef("point:bottom-left"),),
            ),
            NamedRegion(
                "PlateDomain",
                (LogicalEntityRef("face:domain"),),
            ),
        )
    )
    session.replace_mesh_settings(settings)
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
        (SectionDefinition("Plate", "Steel", properties={"thickness": 1.0}),),
        (RegionAssignment("Plate", "PlateDomain"),),
        (
            AnalysisStep(
                "Load",
                boundaries=(
                    DisplacementConstraint("Pinned", 1, 2, 0.0),
                ),
            ),
        ),
    )
    first_task = session.prepare_mesh_generation()
    first_model = generate_fem_model(first_task)
    session.accept_generated_model(first_task.token, first_model)
    before_remesh = session.snapshot()

    delta = session.replace_mesh_settings(MeshSettings(0.2))
    after_remesh = session.snapshot()

    assert not delta.effects
    assert after_remesh.named_regions == before_remesh.named_regions
    assert after_remesh.materials == before_remesh.materials
    assert after_remesh.sections == before_remesh.sections
    assert after_remesh.assignments == before_remesh.assignments
    assert after_remesh.steps == before_remesh.steps
    assert after_remesh.model is None
    assert after_remesh.artifact is None

    second_task = session.prepare_mesh_generation()
    second_model = generate_fem_model(second_task)
    session.accept_generated_model(second_task.token, second_model)
    installed = session.snapshot()
    assert installed.model is not None
    assert installed.model.node_sets["Pinned"].node_ids
    assert installed.model.element_sets["PlateDomain"].element_ids
    assert installed.assignments == before_remesh.assignments
    assert installed.steps == before_remesh.steps


@pytest.mark.gmsh
def test_part_mesh_setting_change_invalidates_face_scope_before_remesh() -> None:
    recipe = BoxGeometry("PartRemeshScope", 2.0, 1.0, 0.5)
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Part remesh scope",
        UnitContext("mm", "N", "MPa"),
        recipe,
    )
    session.replace_part_mesh_settings(
        "P1",
        MeshSettings(0.5, cell_shape="tetrahedron"),
    )
    first_task = session.prepare_mesh_generation()
    first_model = generate_fem_model(first_task)
    assert session.accept_generated_model(
        first_task.token,
        first_model,
    ).accepted

    face = mesh_faces.boundary(first_model.mesh)[0]
    session.replace_named_regions(
        (
            NamedRegion(
                "LoadedFace",
                (MeshEntityRef.face(*face, part_id="P1"),),
            ),
        )
    )
    assert session.snapshot().named_regions

    before_remesh = session.snapshot()
    with pytest.raises(SessionStateError, match="无法安全迁移"):
        session.replace_part_mesh_settings(
            "P1",
            MeshSettings(0.2, cell_shape="tetrahedron"),
        )
    assert session.snapshot() == before_remesh
