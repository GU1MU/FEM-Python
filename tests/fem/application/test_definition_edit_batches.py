from __future__ import annotations

import pytest

from fem.application import (
    ArtifactKind,
    DefinitionEditBatch,
    DeleteIntent,
    ModelSession,
    NamedRegion,
    NamedRegionEditBatch,
    NativePart,
    RegionAssignment,
    RenameIntent,
    RevisionConflictError,
    SectionDefinition,
)
from tests.helpers.preflight_builders import passing_preflight_report
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    ElementSet,
    FEMModel,
    GravityLoad,
    MaterialDefinition,
)
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.geometry import LogicalEntityRef
from fem.geometry.recipe_topology import describe_recipe_topology
from fem.geometry.recipes import BoxGeometry
from tests.helpers.result_builders import make_solve_result_bundle


def _reference(recipe: BoxGeometry, kind: str) -> LogicalEntityRef:
    entity = describe_recipe_topology(recipe).entities_of(kind)[0]
    return LogicalEntityRef(entity.logical_id)


def _definition_session(
    *,
    include_spares: bool = False,
) -> tuple[ModelSession, BoxGeometry]:
    session = ModelSession()
    session.new_native_project()
    recipe = BoxGeometry("Box", 2.0, 1.0, 0.5)
    session.replace_native_geometry_inputs((NativePart(),), recipe)
    session.replace_named_regions(
        (
            NamedRegion("Domain-A", (_reference(recipe, "body"),)),
            NamedRegion("Boundary-A", (_reference(recipe, "face"),)),
        )
    )
    materials = [MaterialDefinition("Steel", {"E": 210_000.0, "nu": 0.3})]
    sections = [SectionDefinition("Solid", "Steel")]
    if include_spares:
        materials.append(MaterialDefinition("Spare-Material", {"E": 1.0}))
        sections.append(SectionDefinition("Spare-Section", "Steel"))
    session.replace_model_definitions(
        tuple(materials),
        tuple(sections),
        (RegionAssignment("Solid", "Domain-A"),),
        (
            AnalysisStep(
                "Step-A",
                boundaries=(DisplacementConstraint("Boundary-A", 1, 1),),
                gravity_loads=(GravityLoad((0.0, -9.81, 0.0), "Domain-A"),),
            ),
        ),
    )
    return session, recipe


def _definition_batch(
    session: ModelSession,
    *,
    materials=None,
    sections=None,
    assignments=None,
    steps=None,
    material_renames=(),
    section_renames=(),
    material_deletes=(),
    section_deletes=(),
) -> DefinitionEditBatch:
    snapshot = session.snapshot()
    return DefinitionEditBatch(
        snapshot.session_revision,
        snapshot.materials if materials is None else tuple(materials),
        snapshot.sections if sections is None else tuple(sections),
        snapshot.assignments if assignments is None else tuple(assignments),
        snapshot.steps if steps is None else tuple(steps),
        tuple(material_renames),
        tuple(section_renames),
        tuple(material_deletes),
        tuple(section_deletes),
    )


def test_material_rename_cascades_to_sections_in_one_commit() -> None:
    session, _recipe = _definition_session()
    before = session.snapshot()
    renamed = MaterialDefinition("Structural Steel", {"E": 210_000.0})
    batch = _definition_batch(
        session,
        materials=(renamed,),
        material_renames=(RenameIntent("Steel", "Structural Steel"),),
    )

    delta = session.apply_definition_edit(batch)
    after = session.snapshot()

    assert delta.session_revision == before.session_revision + 1
    assert after.session_revision == before.session_revision + 1
    assert after.project_revision == before.project_revision + 1
    assert after.model_revision == before.model_revision + 1
    assert after.mesh_input_revision == before.mesh_input_revision
    assert after.dirty
    assert tuple(material.name for material in after.materials) == ("Structural Steel",)
    assert after.sections[0].material == "Structural Steel"

    renamed.properties["E"] = 1.0
    assert after.materials[0].properties["E"] == 210_000.0


def test_material_delete_create_same_count_does_not_retarget_sections() -> None:
    session, _recipe = _definition_session(include_spares=True)
    before = session.snapshot()
    materials = (
        before.materials[0],
        MaterialDefinition("New-Material", {"E": 2.0}),
    )
    batch = _definition_batch(
        session,
        materials=materials,
        sections=(before.sections[0],),
        material_deletes=(DeleteIntent("Spare-Material"),),
        section_deletes=(DeleteIntent("Spare-Section"),),
    )

    session.apply_definition_edit(batch)
    after = session.snapshot()

    assert len(after.materials) == len(before.materials)
    assert after.sections == (before.sections[0],)
    assert after.sections[0].material == "Steel"
    assert {material.name for material in after.materials} == {
        "Steel",
        "New-Material",
    }


def test_referenced_material_delete_is_fail_closed() -> None:
    session, _recipe = _definition_session()
    before = session.snapshot()
    batch = _definition_batch(
        session,
        materials=(MaterialDefinition("Replacement", {"E": 1.0}),),
        sections=(SectionDefinition("Solid", "Replacement"),),
        material_deletes=(DeleteIntent("Steel"),),
    )

    with pytest.raises(ValueError, match="referenced materials"):
        session.apply_definition_edit(batch)

    after = session.snapshot()
    assert after.session_revision == before.session_revision
    assert after.project_revision == before.project_revision
    assert after.model_revision == before.model_revision
    assert after.materials == before.materials
    assert after.sections == before.sections
    assert after.assignments == before.assignments


def test_section_rename_cascades_to_assignments_in_one_commit() -> None:
    session, _recipe = _definition_session()
    before = session.snapshot()
    batch = _definition_batch(
        session,
        sections=(SectionDefinition("Continuum", "Steel"),),
        section_renames=(RenameIntent("Solid", "Continuum"),),
    )

    session.apply_definition_edit(batch)
    after = session.snapshot()

    assert after.session_revision == before.session_revision + 1
    assert after.sections[0].name == "Continuum"
    assert after.assignments[0].section_name == "Continuum"


def test_section_delete_create_same_count_does_not_retarget_assignments() -> None:
    session, _recipe = _definition_session(include_spares=True)
    before = session.snapshot()
    sections = (
        before.sections[0],
        SectionDefinition("New-Section", "Spare-Material"),
    )
    batch = _definition_batch(
        session,
        sections=sections,
        section_deletes=(DeleteIntent("Spare-Section"),),
    )

    session.apply_definition_edit(batch)
    after = session.snapshot()

    assert len(after.sections) == len(before.sections)
    assert after.assignments[0].section_name == "Solid"
    assert {section.name for section in after.sections} == {
        "Solid",
        "New-Section",
    }


def test_assigned_section_delete_is_fail_closed() -> None:
    session, _recipe = _definition_session()
    before = session.snapshot()
    batch = _definition_batch(
        session,
        sections=(SectionDefinition("Replacement", "Steel"),),
        assignments=(RegionAssignment("Replacement", "Domain-A"),),
        section_deletes=(DeleteIntent("Solid"),),
    )

    with pytest.raises(ValueError, match="assigned sections"):
        session.apply_definition_edit(batch)

    assert session.snapshot().session_revision == before.session_revision
    assert session.snapshot().sections == before.sections
    assert session.snapshot().assignments == before.assignments


def test_named_region_rename_cascades_assignment_boundary_and_load_targets() -> None:
    session, recipe = _definition_session()
    before = session.snapshot()
    regions = (
        NamedRegion("Domain-B", (_reference(recipe, "body"),)),
        NamedRegion("Boundary-B", (_reference(recipe, "face"),)),
    )
    batch = NamedRegionEditBatch(
        before.session_revision,
        regions,
        renames=(
            RenameIntent("Domain-A", "Domain-B"),
            RenameIntent("Boundary-A", "Boundary-B"),
        ),
    )

    session.apply_named_region_edit(batch)
    after = session.snapshot()

    assert after.session_revision == before.session_revision + 1
    assert after.project_revision == before.project_revision + 1
    assert after.mesh_input_revision == before.mesh_input_revision + 1
    assert after.model_revision == before.model_revision + 1
    assert tuple(after.named_regions) == ("Domain-B", "Boundary-B")
    assert after.assignments[0].region_name == "Domain-B"
    assert after.steps[0].boundaries[0].target == "Boundary-B"
    assert after.steps[0].gravity_loads[0].target == "Domain-B"


def test_referenced_named_region_delete_is_fail_closed() -> None:
    session, _recipe = _definition_session()
    before = session.snapshot()
    batch = NamedRegionEditBatch(
        before.session_revision,
        (before.named_regions["Boundary-A"],),
        deletes=(DeleteIntent("Domain-A"),),
    )

    with pytest.raises(ValueError, match="referenced named regions"):
        session.apply_named_region_edit(batch)

    after = session.snapshot()
    assert after.session_revision == before.session_revision
    assert dict(after.named_regions) == dict(before.named_regions)
    assert after.assignments == before.assignments
    assert after.steps == before.steps


def test_distinct_named_regions_may_share_the_same_canonical_references() -> None:
    session = ModelSession()
    session.new_native_project()
    recipe = BoxGeometry("Box", 1.0, 1.0, 1.0)
    session.replace_native_geometry_inputs((NativePart(),), recipe)
    shared = (_reference(recipe, "face"),)
    session.replace_named_regions((NamedRegion("First", shared),))
    before = session.snapshot()

    session.apply_named_region_edit(
        NamedRegionEditBatch(
            before.session_revision,
            (
                before.named_regions["First"],
                NamedRegion("Second", shared),
            ),
        )
    )

    after = session.snapshot()
    assert tuple(after.named_regions) == ("First", "Second")
    assert after.named_regions["First"].references == shared
    assert after.named_regions["Second"].references == shared


def test_named_region_delete_create_same_count_is_not_inferred_as_rename() -> None:
    session = ModelSession()
    session.new_native_project()
    recipe = BoxGeometry("Box", 1.0, 1.0, 1.0)
    session.replace_native_geometry_inputs((NativePart(),), recipe)
    shared = (_reference(recipe, "face"),)
    session.replace_named_regions((NamedRegion("Old", shared),))
    before = session.snapshot()

    session.apply_named_region_edit(
        NamedRegionEditBatch(
            before.session_revision,
            (NamedRegion("New", shared),),
            deletes=(DeleteIntent("Old"),),
        )
    )

    assert tuple(session.snapshot().named_regions) == ("New",)


@pytest.mark.parametrize(
    "batch_factory",
    (
        lambda before: DefinitionEditBatch(
            before.session_revision,
            (MaterialDefinition("Other", {}),),
            before.sections,
            before.assignments,
            before.steps,
        ),
        lambda before: DefinitionEditBatch(
            before.session_revision,
            (
                MaterialDefinition("Steel", {}),
                MaterialDefinition("Other", {}),
            ),
            before.sections,
            before.assignments,
            before.steps,
            material_renames=(RenameIntent("Steel", "Other"),),
        ),
        lambda before: DefinitionEditBatch(
            before.session_revision,
            before.materials,
            before.sections,
            before.assignments,
            before.steps,
            material_renames=(
                RenameIntent("Steel", "Other"),
                RenameIntent("Steel", "Another"),
            ),
        ),
    ),
    ids=("implicit-removal", "rename-collision", "duplicate-source"),
)
def test_invalid_definition_identity_ledgers_are_atomic(batch_factory) -> None:
    session, _recipe = _definition_session()
    before = session.snapshot()

    with pytest.raises((KeyError, ValueError)):
        session.apply_definition_edit(batch_factory(before))

    after = session.snapshot()
    assert after.session_revision == before.session_revision
    assert after.project_revision == before.project_revision
    assert after.materials == before.materials
    assert after.sections == before.sections
    assert after.assignments == before.assignments
    assert after.steps == before.steps


def test_named_region_removal_requires_explicit_identity_intent() -> None:
    session, _recipe = _definition_session()
    before = session.snapshot()

    with pytest.raises(ValueError, match="explicit rename or delete"):
        session.apply_named_region_edit(
            NamedRegionEditBatch(
                before.session_revision,
                (before.named_regions["Boundary-A"],),
            )
        )

    assert session.snapshot().session_revision == before.session_revision


def test_definition_batch_compare_and_swap_conflict_is_atomic() -> None:
    session, _recipe = _definition_session()
    stale = _definition_batch(session)
    session.replace_model_definitions(
        session.snapshot().materials,
        session.snapshot().sections,
        session.snapshot().assignments,
        session.snapshot().steps,
    )
    before = session.snapshot()

    with pytest.raises(RevisionConflictError):
        session.apply_definition_edit(stale)

    after = session.snapshot()
    assert after.session_revision == before.session_revision
    assert after.project_revision == before.project_revision
    assert after.materials == before.materials


def test_named_region_batch_compare_and_swap_conflict_is_atomic() -> None:
    session, _recipe = _definition_session()
    before_change = session.snapshot()
    stale = NamedRegionEditBatch(
        before_change.session_revision,
        tuple(before_change.named_regions.values()),
    )
    session.replace_model_definitions(
        before_change.materials,
        before_change.sections,
        before_change.assignments,
        before_change.steps,
    )
    before = session.snapshot()

    with pytest.raises(RevisionConflictError):
        session.apply_named_region_edit(stale)

    after = session.snapshot()
    assert after.session_revision == before.session_revision
    assert dict(after.named_regions) == dict(before.named_regions)


def test_definition_batch_recompiles_artifact_and_retains_completed_result() -> None:
    session, _recipe = _definition_session()
    generated_model = FEMModel(
        mesh=Mesh3D(
            nodes=(
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 1.0, 0.0, 0.0),
                Node3D(3, 0.0, 1.0, 0.0),
                Node3D(4, 0.0, 0.0, 1.0),
            ),
            elements=(
                Element3D(
                    1,
                    (1, 2, 3, 4),
                    "Tet4",
                    {"E": 1.0, "nu": 0.3},
                ),
            ),
        ),
        steps=[AnalysisStep("Step-A")],
        element_sets={"Domain-A": ElementSet("Domain-A", (1,))},
    )
    mesh_task = session.prepare_mesh_generation()
    assert session.accept_generated_model(
        mesh_task.token,
        generated_model,
    ).accepted
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
    properties = {"E": 200_000.0, "nu": 0.3}

    delta = session.apply_definition_edit(
        _definition_batch(
            session,
            materials=(MaterialDefinition("Steel", properties),),
        )
    )
    after = session.snapshot()

    assert after.session_revision == before.session_revision + 1
    assert after.artifact is not None
    assert after.artifact.artifact_id != before.artifact.artifact_id
    assert not after.validations
    assert tuple(run.run_id for run in after.runs) == (solve.run_id,)
    assert after.runs[0].has_result
    assert after.displayed_result is None
    assert {
        ArtifactKind.MODEL,
        ArtifactKind.VALIDATIONS,
        ArtifactKind.RUNS,
        ArtifactKind.DISPLAYED_RESULT,
    }.issubset(delta.invalidated)
    assert ArtifactKind.RESULTS not in delta.invalidated

    session.select_result(solve.run_id)
    assert session.current_result() is not None

    properties["E"] = 1.0
    assert after.materials[0].properties["E"] == 200_000.0
