from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from fem.application import (
    ArtifactKind,
    FeatureRecord,
    ModelSession,
    NamedRegion,
    NativePart,
    RegionAssignment,
    SectionDefinition,
)
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    EdgeLoad,
    ElementSet,
    GravityLoad,
    LineLoad,
    MaterialDefinition,
    NodalLoad,
    SurfaceLoad,
)
from fem.mesh.settings import LocalMeshControl, MeshSettings


def _model(*step_names: str) -> SimpleNamespace:
    return SimpleNamespace(
        materials={},
        sections=[],
        steps=[AnalysisStep(name) for name in step_names],
        element_sets={
            "Region-A": ElementSet("Region-A", (1,)),
            "Region-B": ElementSet("Region-B", (1,)),
        },
        metadata={},
        mesh=SimpleNamespace(nodes=[], elements=[]),
    )


def _session_with_artifacts() -> ModelSession:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry(
        (NativePart(),),
        {"kind": "box"},
        feature_history=(FeatureRecord("Base-1", "base"),),
    )
    session.replace_named_regions(
        (NamedRegion("Region-A", "body", (1,)),)
    )
    session.replace_mesh_settings(
        MeshSettings(
            1.0,
            local_size=0.25,
            local_controls=(LocalMeshControl("edge", 1, 0.2),),
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
    session.accept_validation(validation.token, {"passed": True})
    solve = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(solve.token)
    session.accept_run_result(solve.token, {"value": 1})
    return session


@pytest.mark.parametrize(
    "operation",
    [
        lambda session: session.replace_mesh_settings({"size": 0.25}),
        lambda session: session.replace_named_regions(
            (NamedRegion("Region-A", "body", (1, 2)),)
        ),
        lambda session: session.clear_generated_model(),
    ],
    ids=["mesh-settings", "named-regions", "clear-generated-model"],
)
def test_model_input_changes_invalidate_all_computations(operation) -> None:
    session = _session_with_artifacts()
    previous = session.snapshot()

    delta = operation(session)
    current = session.snapshot()

    assert ArtifactKind.MODEL in delta.invalidated
    assert ArtifactKind.VALIDATIONS in delta.invalidated
    assert ArtifactKind.RUNS in delta.invalidated
    assert ArtifactKind.RESULTS in delta.invalidated
    assert current.artifact is None
    assert not current.validations
    assert not current.runs
    assert current.displayed_result is None
    assert not current.has_result
    assert current.project_revision == previous.project_revision + 1


def test_definition_change_recompiles_model_and_invalidates_derived_state() -> None:
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
    assert not current.runs
    assert current.displayed_result is None
    assert current.model.materials["Steel"].properties["E"] == 2.0


def test_clear_geometry_removes_all_topology_dependent_inputs() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry(
        (NativePart(),),
        {"kind": "box"},
        feature_history=(FeatureRecord("Base-1", "base"),),
    )
    session.replace_named_regions(
        (NamedRegion("Region-A", "body", (1,)),)
    )
    session.replace_mesh_settings(
        MeshSettings(
            1.0,
            local_size=0.25,
            local_controls=(LocalMeshControl("edge", 1, 0.2),),
        )
    )
    step = AnalysisStep(
        "Step-A",
        boundaries=(DisplacementConstraint("Region-A", 1, 1),),
        cloads=(NodalLoad("Region-A", 1, 1.0),),
        edge_loads=(EdgeLoad("Region-A", (1.0, 0.0)),),
        surface_loads=(SurfaceLoad("Region-A", (1.0, 0.0)),),
        line_loads=(LineLoad("Region-A", (1.0, 0.0, 0.0)),),
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
    assert snapshot.mesh_settings.local_size is None
    assert snapshot.mesh_settings.local_controls == ()
    assert not snapshot.named_regions
    assert snapshot.assignments == ()
    assert snapshot.steps == ()
    assert snapshot.artifact is None
    assert not snapshot.validations
    assert not snapshot.runs
    assert snapshot.displayed_result is None

    session.replace_geometry((), {"kind": "new-box"})
    recreated = session.snapshot()
    assert recreated.parts == (NativePart(),)
    assert not recreated.named_regions
    assert recreated.assignments == ()
    assert recreated.steps == ()


def test_named_region_rename_updates_every_reference_atomically() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry((NativePart(),), {"kind": "box"})
    session.replace_named_regions(
        (NamedRegion("Region-A", "body", (1,)),)
    )
    step = AnalysisStep(
        "Step-A",
        boundaries=(DisplacementConstraint("Region-A", 1, 1),),
        cloads=(NodalLoad("Region-A", 1, 1.0),),
        edge_loads=(EdgeLoad("Region-A", (1.0, 0.0)),),
        surface_loads=(SurfaceLoad("Region-A", (1.0, 0.0)),),
        line_loads=(LineLoad("Region-A", (1.0, 0.0, 0.0)),),
        gravity_loads=(GravityLoad((0.0, -9.81), "Region-A"),),
    )
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {}),),
        (SectionDefinition("Solid", "Steel"),),
        (RegionAssignment("Solid", "Region-A"),),
        (step,),
    )

    session.replace_named_regions(
        (NamedRegion("Region-B", "body", (1,)),),
        renames={"Region-A": "Region-B"},
    )
    snapshot = session.snapshot()
    renamed_step = snapshot.steps[0]

    assert tuple(snapshot.named_regions) == ("Region-B",)
    assert snapshot.assignments[0].region_name == "Region-B"
    assert renamed_step.boundaries[0].target == "Region-B"
    assert renamed_step.cloads[0].target == "Region-B"
    assert renamed_step.edge_loads[0].edge == "Region-B"
    assert renamed_step.surface_loads[0].surface == "Region-B"
    assert renamed_step.line_loads[0].target == "Region-B"
    assert renamed_step.gravity_loads[0].target == "Region-B"


def test_removing_a_referenced_named_region_is_rejected_without_side_effects() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry((NativePart(),), {"kind": "box"})
    session.replace_named_regions(
        (NamedRegion("Region-A", "body", (1,)),)
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
        snapshot.named_regions["Region-A"].entity_ids += (99,)
    with pytest.raises(TypeError):
        snapshot.named_regions["Other"] = NamedRegion("Other", "body", ())

    validation = session.prepare_validation("Step-A")
    validation.model.materials["Steel"].properties["E"] = 77.0

    fresh = session.snapshot()
    assert fresh.model.materials["Steel"].properties["E"] == 1.0
    assert "tag" not in fresh.sections[0].properties
    assert fresh.named_regions["Region-A"].entity_ids == (1,)
