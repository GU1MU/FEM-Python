from __future__ import annotations

import pytest

from fem.application import (
    ChangeKind,
    ModelSession,
    NamedRegion,
    NativePart,
    RegionAssignment,
    RevisionConflictError,
    SectionDefinition,
    TransitionEffect,
)
from fem.core.model import (
    AnalysisStep,
    ElementSet,
    FEMModel,
    GravityLoad,
    MaterialDefinition,
)
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.geometry import LogicalEntityRef
from fem.geometry.recipe_topology import describe_recipe_topology
from fem.geometry.recipes import (
    BoxGeometry,
    CylinderGeometry,
    ExtrudedGeometry,
    RectangleGeometry,
    WireGeometry,
    WireMember,
    WirePoint,
)
from fem.mesh.settings import LocalMeshControl, MeshSettings
from tests.helpers.preflight_builders import passing_preflight_report
from tests.helpers.result_builders import (
    assert_result_records_equivalent,
    make_solve_result_bundle,
)
from tests.geometry.test_profile_extrusion import (
    profile_face_id,
    two_profile_sketch,
)


def _reference(recipe, kind: str) -> LogicalEntityRef:
    entity = describe_recipe_topology(recipe).entities_of(kind)[0]
    return LogicalEntityRef(entity.logical_id)


def _wire_recipe(
    *,
    moved: bool = False,
    extra_member: bool = False,
    reconnected: bool = False,
) -> WireGeometry:
    points = (
        WirePoint("P1", 0.0 if not moved else 10.0, 0.0),
        WirePoint("P2", 1.0 if not moved else 11.0, 0.0),
        WirePoint("P3", 1.0 if not moved else 11.0, 1.0),
    )
    members = [
        WireMember("M1", "P1", "P2"),
        WireMember("M2", "P2", "P3"),
    ]
    if extra_member:
        points = (*points, WirePoint("P4", 2.0, 1.0))
        members.append(WireMember("M3", "P3", "P4"))
    if reconnected:
        members[1] = WireMember("M2", "P1", "P3")
    return WireGeometry("Wire", points, tuple(members))


def test_profile_selection_edit_preserves_only_surviving_lineage() -> None:
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")
    second = profile_face_id(sketch, "L5")
    first_name = first.split(":", 1)[1]
    before = ExtrudedGeometry(sketch, 1.0, (first, second))
    after = ExtrudedGeometry(sketch, 1.0, (first,))
    removed_side = LogicalEntityRef(f"face:side/{first_name}/L1")
    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        before,
        mesh_settings=MeshSettings(
            0.5,
            cell_shape="tetrahedron",
            local_controls=(LocalMeshControl(removed_side, 0.2),),
        ),
    )
    session.replace_named_regions(
        (
            NamedRegion(
                "Body",
                (LogicalEntityRef("body:domain"),),
            ),
            NamedRegion("ProfileSide", (removed_side,)),
        )
    )

    delta = session.replace_native_geometry_inputs(
        (NativePart(),),
        after,
    )
    snapshot = session.snapshot()

    assert tuple(snapshot.named_regions) == ("Body",)
    assert snapshot.mesh_settings is not None
    assert snapshot.mesh_settings.local_controls == ()
    assert TransitionEffect.NAMED_REGIONS_CLEARED in delta.effects
    assert TransitionEffect.LOCAL_CONTROLS_CLEARED in delta.effects


def test_new_geometry_creates_dimension_aware_default_mesh_settings() -> None:
    session = ModelSession()
    session.new_native_project()
    before = session.snapshot()

    delta = session.replace_native_geometry_inputs(
        (NativePart(),),
        RectangleGeometry("Plate", 20.0, 10.0),
    )
    after = session.snapshot()

    assert delta.session_revision == before.session_revision + 1
    assert after.session_revision == before.session_revision + 1
    assert after.project_revision == before.project_revision + 1
    assert after.mesh_input_revision == before.mesh_input_revision + 1
    assert after.model_revision == before.model_revision + 1
    assert after.mesh_settings == MeshSettings(
        1.0,
        cell_shape="triangle",
    )
    assert ChangeKind.GEOMETRY in delta.changed
    assert ChangeKind.MESH_SETTINGS in delta.changed


def test_new_wire_without_explicit_mesh_settings_keeps_mesh_intent_unset() -> None:
    session = ModelSession()
    session.new_native_project()

    session.replace_native_geometry_inputs(
        (NativePart(),),
        _wire_recipe(),
    )

    assert session.snapshot().mesh_settings is None


def test_wire_with_explicit_line_settings_is_installed_atomically() -> None:
    session = ModelSession()
    session.new_native_project()
    settings = MeshSettings(
        0.25,
        cell_shape="line",
        line_element_type="Beam2",
    )

    session.replace_native_geometry_inputs(
        (NativePart(),),
        _wire_recipe(),
        mesh_settings=settings,
    )

    after = session.snapshot()
    assert after.geometry_recipe == _wire_recipe()
    assert after.mesh_settings == settings


def test_wire_coordinate_edit_preserves_line_settings_and_local_controls() -> None:
    session = ModelSession()
    session.new_native_project()
    first = _wire_recipe()
    settings = MeshSettings(
        0.5,
        cell_shape="line",
        line_element_type="Truss2",
        local_controls=(
            LocalMeshControl(LogicalEntityRef("edge:M1"), 0.2),
        ),
    )
    session.replace_native_geometry_inputs(
        (NativePart(),),
        first,
        mesh_settings=settings,
    )

    delta = session.replace_native_geometry_inputs(
        (NativePart(),),
        _wire_recipe(moved=True),
    )

    assert session.snapshot().mesh_settings == settings
    assert delta.effects == {TransitionEffect.REFERENCES_PRESERVED}


def test_wire_topology_edit_keeps_formulation_but_clears_entity_controls() -> None:
    session = ModelSession()
    session.new_native_project()
    first = _wire_recipe()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        first,
        mesh_settings=MeshSettings(
            0.5,
            cell_shape="line",
            line_element_type="Beam2",
            local_controls=(
                LocalMeshControl(LogicalEntityRef("edge:M1"), 0.2),
            ),
        ),
    )

    delta = session.replace_native_geometry_inputs(
        (NativePart(),),
        _wire_recipe(extra_member=True),
    )

    assert session.snapshot().mesh_settings == MeshSettings(
        0.5,
        cell_shape="line",
        line_element_type="Beam2",
    )
    assert delta.effects == {
        TransitionEffect.LOCAL_CONTROLS_CLEARED,
    }


def test_wire_member_reconnection_clears_entity_controls() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        _wire_recipe(),
        mesh_settings=MeshSettings(
            0.5,
            cell_shape="line",
            line_element_type="Truss2",
            local_controls=(
                LocalMeshControl(LogicalEntityRef("edge:M1"), 0.2),
            ),
        ),
    )

    delta = session.replace_native_geometry_inputs(
        (NativePart(),),
        _wire_recipe(reconnected=True),
    )

    assert session.snapshot().mesh_settings == MeshSettings(
        0.5,
        cell_shape="line",
        line_element_type="Truss2",
    )
    assert delta.effects == {TransitionEffect.LOCAL_CONTROLS_CLEARED}


def test_dimension_transitions_clear_or_remove_line_intent_atomically() -> None:
    session = ModelSession()
    session.new_native_project()
    continuum = RectangleGeometry("Plate", 2.0, 1.0)
    session.replace_native_geometry_inputs((NativePart(),), continuum)
    session.replace_native_geometry_inputs(
        (NativePart(),),
        _wire_recipe(),
    )
    assert session.snapshot().mesh_settings is None

    session.replace_native_geometry_inputs(
        (NativePart(),),
        _wire_recipe(),
        mesh_settings=MeshSettings(
            0.25,
            cell_shape="line",
            line_element_type="Truss2",
        ),
    )
    session.replace_native_geometry_inputs(
        (NativePart(),),
        continuum,
    )

    assert session.snapshot().mesh_settings == MeshSettings(
        0.25,
        cell_shape="triangle",
    )


def test_wire_rejects_continuum_mesh_shape_without_mutating_session() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        _wire_recipe(),
        mesh_settings=MeshSettings(
            0.25,
            cell_shape="line",
            line_element_type="Beam2",
        ),
    )
    before = session.snapshot()

    with pytest.raises(ValueError, match="not supported"):
        session.replace_native_geometry_inputs(
            (NativePart(),),
            _wire_recipe(moved=True),
            mesh_settings=MeshSettings(0.25, cell_shape="triangle"),
        )

    assert session.snapshot() == before


def test_continuum_rejects_line_mesh_settings_without_mutating_session() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        RectangleGeometry("Plate", 2.0, 1.0),
    )
    before = session.snapshot()

    with pytest.raises(ValueError, match="not supported"):
        session.replace_mesh_settings(
            MeshSettings(
                0.25,
                cell_shape="line",
                line_element_type="Truss2",
            )
        )

    assert session.snapshot() == before


def test_explicit_none_clears_mesh_settings_in_the_geometry_commit() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        RectangleGeometry("Plate", 2.0, 1.0),
    )
    before = session.snapshot()

    delta = session.replace_native_geometry_inputs(
        (NativePart(),),
        RectangleGeometry("Resized", 3.0, 2.0),
        mesh_settings=None,
        expected_session_revision=before.session_revision,
    )
    after = session.snapshot()

    assert delta.session_revision == before.session_revision + 1
    assert after.mesh_settings is None
    assert TransitionEffect.REFERENCES_PRESERVED in delta.effects
    assert ChangeKind.MESH_SETTINGS in delta.changed


def test_explicit_mesh_settings_are_installed_atomically() -> None:
    session = ModelSession()
    session.new_native_project()
    before = session.snapshot()
    settings = MeshSettings(
        0.25,
        order=2,
        cell_shape="tetrahedron",
    )

    session.replace_native_geometry_inputs(
        (NativePart(),),
        BoxGeometry("Box", 2.0, 1.0, 0.5),
        mesh_settings=settings,
        expected_session_revision=before.session_revision,
    )
    after = session.snapshot()

    assert after.session_revision == before.session_revision + 1
    assert after.mesh_settings == settings


def test_invalid_explicit_mesh_settings_reject_the_whole_transition() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        RectangleGeometry("Plate", 2.0, 1.0),
    )
    recipe = session.snapshot().geometry_recipe
    session.replace_named_regions(
        (NamedRegion("Domain-A", (_reference(recipe, "body"),)),)
    )
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 1.0, "nu": 0.3}),),
        (SectionDefinition("Solid", "Steel"),),
        (RegionAssignment("Solid", "Domain-A"),),
        (AnalysisStep("Step-A"),),
    )
    generated_model = FEMModel(
        mesh=Mesh2D(
            nodes=(
                Node2D(1, 0.0, 0.0),
                Node2D(2, 1.0, 0.0),
                Node2D(3, 0.0, 1.0),
            ),
            elements=(
                Element2D(
                    1,
                    (1, 2, 3),
                    "Tri3",
                    {
                        "E": 1.0,
                        "nu": 0.3,
                        "plane_type": "stress",
                        "thickness": 1.0,
                    },
                ),
            ),
        ),
        steps=[AnalysisStep("Step-A")],
        element_sets={"Domain-A": ElementSet("Domain-A", (1,))},
    )
    mesh_task = session.prepare_mesh_generation()
    session.accept_generated_model(mesh_task.token, generated_model)
    validation = session.prepare_validation("Step-A")
    session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    solve = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(solve.token)
    session.accept_run_succeeded(
        solve.token,
        make_solve_result_bundle(solve, marker=1.0),
    )
    session.select_result(solve.run_id)
    before = session.snapshot()

    with pytest.raises(ValueError, match="not supported"):
        session.replace_native_geometry_inputs(
            (NativePart(),),
            CylinderGeometry("Cylinder", 1.0, 2.0),
            mesh_settings=MeshSettings(
                0.25,
                cell_shape="hexahedron",
            ),
            expected_session_revision=before.session_revision,
        )

    after = session.snapshot()
    assert after.session_revision == before.session_revision
    assert after.project_revision == before.project_revision
    assert after.mesh_input_revision == before.mesh_input_revision
    assert after.model_revision == before.model_revision
    assert after.geometry_recipe == before.geometry_recipe
    assert after.mesh_settings == before.mesh_settings
    assert after.feature_history == before.feature_history
    assert after.artifact.artifact_id == before.artifact.artifact_id
    assert after.validations == before.validations
    assert after.runs == before.runs
    assert_result_records_equivalent(
        after.displayed_result,
        before.displayed_result,
    )


def test_unset_preserves_compatible_local_controls_and_reports_effect() -> None:
    session = ModelSession()
    session.new_native_project()
    recipe = RectangleGeometry("Plate", 2.0, 1.0)
    session.replace_native_geometry_inputs(
        (NativePart(),),
        recipe,
        mesh_settings=MeshSettings(
            0.5,
            cell_shape="quadrilateral",
            local_controls=(LocalMeshControl(_reference(recipe, "edge"), 0.2),),
        ),
    )
    before = session.snapshot()

    delta = session.replace_native_geometry_inputs(
        (NativePart(),),
        RectangleGeometry("Resized", 4.0, 3.0),
    )
    after = session.snapshot()

    assert after.session_revision == before.session_revision + 1
    assert after.mesh_settings == before.mesh_settings
    assert delta.effects == {
        TransitionEffect.REFERENCES_PRESERVED,
    }
    assert ChangeKind.MESH_SETTINGS not in delta.changed


def test_unset_clears_incompatible_dependencies_and_normalizes_hex() -> None:
    session = ModelSession()
    session.new_native_project()
    recipe = BoxGeometry("Box", 2.0, 1.0, 0.5)
    session.replace_native_geometry_inputs(
        (NativePart(),),
        recipe,
        mesh_settings=MeshSettings(
            0.5,
            cell_shape="hexahedron",
            local_controls=(LocalMeshControl(_reference(recipe, "edge"), 0.2),),
        ),
    )
    session.replace_named_regions(
        (NamedRegion("Domain-A", (_reference(recipe, "body"),)),)
    )
    dependent = AnalysisStep(
        "Dependent",
        gravity_loads=(GravityLoad((0.0, -9.81, 0.0), "Domain-A"),),
    )
    independent = AnalysisStep("Independent")
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 1.0}),),
        (SectionDefinition("Solid", "Steel"),),
        (RegionAssignment("Solid", "Domain-A"),),
        (dependent, independent),
    )
    before = session.snapshot()

    delta = session.replace_native_geometry_inputs(
        (NativePart(),),
        CylinderGeometry("Cylinder", 1.0, 2.0),
        expected_session_revision=before.session_revision,
    )
    after = session.snapshot()

    assert after.session_revision == before.session_revision + 1
    assert after.project_revision == before.project_revision + 1
    assert after.mesh_input_revision == before.mesh_input_revision + 1
    assert after.model_revision == before.model_revision + 1
    assert after.mesh_settings == MeshSettings(
        0.5,
        cell_shape="tetrahedron",
    )
    assert not after.named_regions
    assert not after.assignments
    assert after.steps == (independent,)
    assert delta.effects == {
        TransitionEffect.NAMED_REGIONS_CLEARED,
        TransitionEffect.LOCAL_CONTROLS_CLEARED,
        TransitionEffect.ASSIGNMENTS_CLEARED,
        TransitionEffect.STEPS_CLEARED,
        TransitionEffect.MESH_SHAPE_NORMALIZED,
    }


def test_atomic_geometry_command_rejects_a_stale_base_revision() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_native_geometry_inputs(
        (NativePart(),),
        RectangleGeometry("Plate", 2.0, 1.0),
    )
    stale_revision = session.session_revision
    session.replace_mesh_settings(MeshSettings(0.25))
    before = session.snapshot()

    with pytest.raises(RevisionConflictError):
        session.replace_native_geometry_inputs(
            (NativePart(),),
            RectangleGeometry("Stale", 4.0, 2.0),
            expected_session_revision=stale_revision,
        )

    after = session.snapshot()
    assert after.session_revision == before.session_revision
    assert after.project_revision == before.project_revision
    assert after.geometry_recipe == before.geometry_recipe
    assert after.mesh_settings == before.mesh_settings
