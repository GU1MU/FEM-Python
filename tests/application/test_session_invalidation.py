from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from fem.io.inp import read
from fem.application import (
    ArtifactKind,
    BeamOrientation,
    ChangeKind,
    DefinitionRejected,
    ModelSession,
    NamedRegion,
    NativePart,
    RegionAssignment,
    SectionDefinition,
    TokenStatus,
)
from fem.core.model import (
    AnalysisStep,
    ElementSet,
    FEMModel,
    GravityLoad,
    MaterialDefinition,
)
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.geometry.recipes import (
    BooleanGeometry,
    BoxGeometry,
    MovedGeometry,
    RectangleGeometry,
)
from fem.geometry import LogicalEntityRef
from fem.geometry.recipe_topology import describe_recipe_topology
from fem.mesh.settings import LocalMeshControl, MeshSettings
from tests.helpers.preflight_builders import passing_preflight_report
from tests.helpers.result_builders import (
    assert_result_records_equivalent,
    make_solve_result_bundle,
)


_FIXTURES = (
    Path(__file__).parents[1]
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)


def _first_reference(recipe, kind: str) -> LogicalEntityRef:
    entity = describe_recipe_topology(recipe).entities_of(kind)[0]
    return LogicalEntityRef(entity.logical_id)


def _model(*step_names: str) -> FEMModel:
    return FEMModel(
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
        steps=[AnalysisStep(name) for name in step_names],
        element_sets={
            "Region-A": ElementSet("Region-A", (1,)),
            "Region-B": ElementSet("Region-B", (1,)),
        },
    )


def _session_with_artifacts() -> ModelSession:
    session = ModelSession()
    session.new_native_project()
    recipe = BoxGeometry("Box", 1.0, 1.0, 1.0)
    session.replace_geometry(
        (NativePart(),),
        recipe,
    )
    session.replace_named_regions(
        (NamedRegion("Region-A", (_first_reference(recipe, "body"),)),)
    )
    session.replace_mesh_settings(
        MeshSettings(
            1.0,
            local_controls=(
                LocalMeshControl(_first_reference(recipe, "edge"), 0.2),
            ),
        )
    )
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 1.0, "nu": 0.3}),),
        (SectionDefinition("Solid", "Steel"),),
        (RegionAssignment("Solid", "Region-A"),),
        (AnalysisStep("Step-A"),),
    )
    mesh = session.prepare_mesh_generation()
    session.accept_generated_model(mesh.token, _model("Step-A"))
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
    return session


def _accept_beam_computations(session: ModelSession) -> None:
    validation = session.prepare_validation("UniformLoad")
    session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    run_name = (
        "Beam-Job"
        if session.find_run("Beam-Job") is None
        else session.next_run_name()
    )
    solve = session.prepare_solve("UniformLoad", run_name)
    session.begin_run(solve.token)
    session.accept_run_succeeded(
        solve.token,
        make_solve_result_bundle(solve, marker=1.0),
    )
    session.select_result(solve.run_id)


def _beam_session_with_artifacts() -> ModelSession:
    session = ModelSession()
    task = session.prepare_import(
        _FIXTURES / "beam2_rectangle_uniform_load.inp"
    )
    session.accept_imported_model(
        task.token,
        read(_FIXTURES / "beam2_rectangle_uniform_load.inp"),
    )
    _accept_beam_computations(session)
    return session


def _session_with_exact_geometry_artifacts(recipe) -> ModelSession:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry(
        (NativePart(),),
        recipe,
    )
    session.replace_named_regions(
        (NamedRegion("Region-A", (_first_reference(recipe, "body"),)),)
    )
    session.replace_mesh_settings(
        MeshSettings(
            1.0,
            local_controls=(
                LocalMeshControl(_first_reference(recipe, "edge"), 0.2),
            ),
        )
    )
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 1.0, "nu": 0.3}),),
        (SectionDefinition("Solid", "Steel"),),
        (RegionAssignment("Solid", "Region-A"),),
        (AnalysisStep("Step-A"),),
    )
    mesh = session.prepare_mesh_generation()
    session.accept_generated_model(mesh.token, _model("Step-A"))
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
    return session


def test_topology_compatible_geometry_edit_preserves_inputs_and_drops_artifacts() -> None:
    session = _session_with_exact_geometry_artifacts(
        RectangleGeometry("Plate", 2.0, 1.0)
    )

    delta = session.replace_geometry(
        (NativePart(),),
        RectangleGeometry("Resized", 4.0, 3.0),
    )
    snapshot = session.snapshot()

    assert delta.changed == {
        ChangeKind.PROJECT_INPUTS,
        ChangeKind.GEOMETRY,
        ChangeKind.MODEL,
        ChangeKind.VALIDATIONS,
        ChangeKind.RUNS,
        ChangeKind.DISPLAYED_RESULT,
    }
    assert snapshot.named_regions["Region-A"] == NamedRegion(
        "Region-A", (LogicalEntityRef("body:domain"),)
    )
    assert snapshot.mesh_settings.local_controls == (
        LocalMeshControl(LogicalEntityRef("edge:bottom"), 0.2),
    )
    assert snapshot.assignments == (RegionAssignment("Solid", "Region-A"),)
    assert tuple(step.name for step in snapshot.steps) == ("Step-A",)
    assert snapshot.artifact is None
    assert not snapshot.validations
    assert tuple(run.name for run in snapshot.runs) == ("Job-1",)
    assert snapshot.runs[0].has_result
    assert snapshot.displayed_result is None
    assert {
        ArtifactKind.MODEL,
        ArtifactKind.DISPLAYED_RESULT,
    }.issubset(delta.invalidated)
    assert ArtifactKind.RESULTS not in delta.invalidated


@pytest.mark.parametrize(
    ("before", "after"),
    (
        (
            BoxGeometry("Box", 2.0, 1.0, 0.5),
            RectangleGeometry("Plate", 2.0, 1.0),
        ),
        (
            RectangleGeometry("Plate", 2.0, 1.0),
            BooleanGeometry(
                "Union",
                "fuse",
                RectangleGeometry("Object", 2.0, 1.0),
                MovedGeometry(
                    RectangleGeometry("Tool", 1.0, 0.5),
                    0.5,
                    0.25,
                ),
            ),
        ),
    ),
    ids=("box-to-rectangle", "unproven-boolean"),
)
def test_topology_incompatible_geometry_edit_clears_dependent_inputs(
    before,
    after,
) -> None:
    session = _session_with_exact_geometry_artifacts(before)

    delta = session.replace_geometry((NativePart(),), after)
    snapshot = session.snapshot()

    assert delta.changed == {
        ChangeKind.PROJECT_INPUTS,
        ChangeKind.GEOMETRY,
        ChangeKind.MESH_SETTINGS,
        ChangeKind.NAMED_REGIONS,
        ChangeKind.DEFINITIONS,
        ChangeKind.MODEL,
        ChangeKind.VALIDATIONS,
        ChangeKind.RUNS,
        ChangeKind.DISPLAYED_RESULT,
    }
    assert snapshot.mesh_settings.local_controls == ()
    assert not snapshot.named_regions
    assert snapshot.assignments == ()
    assert tuple(step.name for step in snapshot.steps) == ("Step-A",)
    assert snapshot.artifact is None


def test_topology_change_preserves_only_geometry_independent_steps() -> None:
    session = ModelSession()
    session.new_native_project()
    recipe = BoxGeometry("Box", 2.0, 1.0, 0.5)
    region = NamedRegion(
        "Region-A",
        (_first_reference(recipe, "body"),),
    )
    global_gravity = AnalysisStep(
        "Global Gravity",
        gravity_loads=(GravityLoad((0.0, -9.81, 0.0)),),
    )
    empty = AnalysisStep("Empty")
    region_gravity = AnalysisStep(
        "Region Gravity",
        gravity_loads=(
            GravityLoad((0.0, -9.81, 0.0), "Region-A"),
        ),
    )
    session.replace_geometry((NativePart(),), recipe)
    session.replace_named_regions((region,))
    session.replace_mesh_settings(
        MeshSettings(
            1.0,
            local_controls=(
                LocalMeshControl(_first_reference(recipe, "edge"), 0.2),
            ),
        )
    )
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 1.0}),),
        (SectionDefinition("Solid", "Steel"),),
        (RegionAssignment("Solid", "Region-A"),),
        (empty, global_gravity, region_gravity),
    )
    before = session.snapshot()

    delta = session.replace_geometry(
        (NativePart(),),
        RectangleGeometry("Plate", 3.0, 2.0),
    )
    after = session.snapshot()

    assert delta.session_revision == before.session_revision + 1
    assert delta.changed == {
        ChangeKind.PROJECT_INPUTS,
        ChangeKind.GEOMETRY,
        ChangeKind.MESH_SETTINGS,
        ChangeKind.NAMED_REGIONS,
        ChangeKind.DEFINITIONS,
        ChangeKind.MODEL,
        ChangeKind.VALIDATIONS,
        ChangeKind.RUNS,
        ChangeKind.DISPLAYED_RESULT,
    }
    assert delta.invalidated == {
        ArtifactKind.MODEL,
        ArtifactKind.VALIDATIONS,
        ArtifactKind.RUNS,
        ArtifactKind.DISPLAYED_RESULT,
    }
    assert after.session_revision == before.session_revision + 1
    assert after.project_revision == before.project_revision + 1
    assert after.mesh_input_revision == before.mesh_input_revision + 1
    assert after.model_revision == before.model_revision + 1
    assert after.materials == before.materials
    assert after.sections == before.sections
    assert after.steps == (empty, global_gravity)
    assert after.assignments == ()
    assert not after.named_regions
    assert after.mesh_settings.local_controls == ()


@pytest.mark.parametrize(
    "operation",
    [
        lambda session: session.replace_mesh_settings(MeshSettings(0.25)),
        lambda session: session.replace_named_regions(
            (
                session.snapshot().named_regions["Region-A"],
                NamedRegion(
                    "Region-B",
                    (LogicalEntityRef("point:bottom-front-left"),),
                ),
            )
        ),
        lambda session: session.clear_generated_model(),
    ],
    ids=["mesh-settings", "named-regions", "clear-generated-model"],
)
def test_model_input_changes_retain_completed_run_history(operation) -> None:
    session = _session_with_artifacts()
    previous = session.snapshot()

    delta = operation(session)
    current = session.snapshot()

    assert ArtifactKind.MODEL in delta.invalidated
    assert ArtifactKind.VALIDATIONS in delta.invalidated
    assert ArtifactKind.RUNS in delta.invalidated
    assert ArtifactKind.RESULTS not in delta.invalidated
    assert current.artifact is None
    assert not current.validations
    assert current.runs == previous.runs
    assert current.displayed_result is None
    assert not current.has_result
    assert current.project_revision == previous.project_revision + 1


def test_definition_change_recompiles_model_and_retains_run_history() -> None:
    session = _session_with_artifacts()
    previous = session.snapshot()

    delta = session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 2.0, "nu": 0.3}),),
        (SectionDefinition("Solid", "Steel"),),
        (RegionAssignment("Solid", "Region-A"),),
        (AnalysisStep("Step-A"),),
    )
    current = session.snapshot()

    assert ArtifactKind.MODEL in delta.invalidated
    assert current.artifact is not None
    assert current.artifact.artifact_id != previous.artifact.artifact_id
    assert current.model_revision == previous.model_revision + 1
    assert current.mesh_input_revision == previous.mesh_input_revision
    assert not current.validations
    assert current.runs == previous.runs
    assert current.displayed_result is None
    assert current.model.materials["Steel"].properties["E"] == 2.0


def test_beam_orientation_edit_and_clear_recompile_and_invalidate() -> None:
    session = _beam_session_with_artifacts()

    for orientation in (
        BeamOrientation((0.0, 1.0, 0.0)),
        None,
    ):
        before = session.snapshot()
        assignment = before.assignments[0]

        delta = session.replace_model_definitions(
            before.materials,
            before.sections,
            (
                RegionAssignment(
                    assignment.section_name,
                    assignment.region_name,
                    orientation,
                ),
            ),
            before.steps,
        )
        current = session.snapshot()

        assert current.session_revision == before.session_revision + 1
        assert current.project_revision == before.project_revision + 1
        assert current.model_revision == before.model_revision + 1
        assert current.artifact is not None
        assert current.artifact.artifact_id != (
            before.artifact.artifact_id
        )
        assert current.assignments[0].beam_orientation == (
            orientation
        )
        assert not current.validations
        assert all(run.has_result for run in current.runs)
        assert current.displayed_result is None
        assert not current.has_result
        assert {
            ArtifactKind.MODEL,
            ArtifactKind.VALIDATIONS,
            ArtifactKind.RUNS,
        }.issubset(delta.invalidated)
        assert ArtifactKind.RESULTS not in delta.invalidated

        if orientation is not None:
            _accept_beam_computations(session)


def test_parallel_orientation_rejection_preserves_all_accepted_state() -> None:
    session = _beam_session_with_artifacts()
    validation = session.prepare_validation("UniformLoad")
    before = session.snapshot()
    assignment = before.assignments[0]

    with pytest.raises(DefinitionRejected) as caught:
        session.replace_model_definitions(
            before.materials,
            before.sections,
            (
                RegionAssignment(
                    assignment.section_name,
                    assignment.region_name,
                    BeamOrientation((1.0, 0.0, 0.0)),
                ),
            ),
            before.steps,
        )

    after = session.snapshot()
    assert {
        item.code for item in caught.value.diagnostics
    } == {"beam.orientation.parallel"}
    assert after.session_revision == before.session_revision
    assert after.project_revision == before.project_revision
    assert after.model_revision == before.model_revision
    assert after.artifact.artifact_id == before.artifact.artifact_id
    assert after.assignments == before.assignments
    assert after.validations == before.validations
    assert after.runs == before.runs
    assert_result_records_equivalent(
        after.displayed_result,
        before.displayed_result,
    )
    assert after.has_result == before.has_result
    assert session.validate_task_token(validation.token) is (
        TokenStatus.CURRENT
    )


def test_clear_geometry_removes_all_topology_dependent_inputs() -> None:
    session = ModelSession()
    session.new_native_project()
    recipe = BoxGeometry("Box", 1.0, 1.0, 1.0)
    session.replace_geometry(
        (NativePart(),),
        recipe,
    )
    session.replace_named_regions(
        (NamedRegion("Region-A", (_first_reference(recipe, "body"),)),)
    )
    session.replace_mesh_settings(
        MeshSettings(
            1.0,
            local_controls=(
                LocalMeshControl(_first_reference(recipe, "edge"), 0.2),
            ),
        )
    )
    step = AnalysisStep(
        "Step-A",
        gravity_loads=(GravityLoad((0.0, -9.81), "Region-A"),),
    )
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {}),),
        (SectionDefinition("Solid", "Steel"),),
        (RegionAssignment("Solid", "Region-A"),),
        (step,),
    )

    session.clear_geometry()
    snapshot = session.snapshot()

    assert snapshot.parts == ()
    assert snapshot.geometry_recipe is None
    assert snapshot.feature_history == ()
    assert snapshot.mesh_settings.size == 1.0
    assert snapshot.mesh_settings.local_controls == ()
    assert not snapshot.named_regions
    assert snapshot.assignments == ()
    assert snapshot.steps == ()
    assert snapshot.artifact is None
    assert not snapshot.validations
    assert not snapshot.runs
    assert snapshot.displayed_result is None

    session.replace_geometry(
        (), BoxGeometry("New Box", 2.0, 1.0, 1.0)
    )
    recreated = session.snapshot()
    assert recreated.parts == (NativePart(),)
    assert not recreated.named_regions
    assert recreated.assignments == ()
    assert recreated.steps == ()


def test_named_region_rename_updates_every_reference_atomically() -> None:
    session = ModelSession()
    session.new_native_project()
    recipe = BoxGeometry("Box", 1.0, 1.0, 1.0)
    session.replace_geometry((NativePart(),), recipe)
    session.replace_named_regions(
        (NamedRegion("Region-A", (_first_reference(recipe, "body"),)),)
    )
    step = AnalysisStep(
        "Step-A",
        gravity_loads=(GravityLoad((0.0, -9.81), "Region-A"),),
    )
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {}),),
        (SectionDefinition("Solid", "Steel"),),
        (RegionAssignment("Solid", "Region-A"),),
        (step,),
    )

    session.replace_named_regions(
        (NamedRegion("Region-B", (_first_reference(recipe, "body"),)),),
        renames={"Region-A": "Region-B"},
    )
    snapshot = session.snapshot()
    renamed_step = snapshot.steps[0]

    assert tuple(snapshot.named_regions) == ("Region-B",)
    assert snapshot.assignments[0].region_name == "Region-B"
    assert renamed_step.gravity_loads[0].target == "Region-B"


def test_removing_a_referenced_named_region_is_rejected_without_side_effects() -> None:
    session = ModelSession()
    session.new_native_project()
    recipe = BoxGeometry("Box", 1.0, 1.0, 1.0)
    session.replace_geometry((NativePart(),), recipe)
    session.replace_named_regions(
        (NamedRegion("Region-A", (_first_reference(recipe, "body"),)),)
    )
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {}),),
        (SectionDefinition("Solid", "Steel"),),
        (RegionAssignment("Solid", "Region-A"),),
        (AnalysisStep("Step-A"),),
    )
    before = session.snapshot()

    with pytest.raises(ValueError, match="referenced"):
        session.replace_named_regions(())

    after = session.snapshot()
    assert after.session_revision == before.session_revision
    assert after.project_revision == before.project_revision
    assert dict(after.named_regions) == dict(before.named_regions)
    assert after.assignments == before.assignments


def test_snapshots_and_task_inputs_do_not_expose_authoritative_mutable_objects() -> None:
    session = _session_with_artifacts()
    snapshot = session.snapshot()
    snapshot.model.materials["Steel"].properties["E"] = 99.0
    snapshot.sections[0].properties["tag"] = "changed"
    with pytest.raises(FrozenInstanceError):
        snapshot.named_regions["Region-A"].references += (
            LogicalEntityRef("body:other"),
        )
    with pytest.raises(TypeError):
        snapshot.named_regions["Other"] = NamedRegion(
            "Other", (LogicalEntityRef("body:domain"),)
        )

    validation = session.prepare_validation("Step-A")
    validation.model.materials["Steel"].properties["E"] = 77.0

    fresh = session.snapshot()
    assert fresh.model.materials["Steel"].properties["E"] == 1.0
    assert "tag" not in fresh.sections[0].properties
    assert fresh.named_regions["Region-A"].references == (
        LogicalEntityRef("body:domain"),
    )
