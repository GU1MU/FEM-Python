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
    RectangleGeometry,
)
from fem.mesh.settings import LocalMeshControl, MeshSettings
from tests.helpers.preflight_builders import passing_preflight_report
from tests.helpers.result_builders import (
    assert_result_records_equivalent,
    make_solve_result_bundle,
)


def _reference(recipe, kind: str) -> LogicalEntityRef:
    entity = describe_recipe_topology(recipe).entities_of(kind)[0]
    return LogicalEntityRef(entity.logical_id)


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
