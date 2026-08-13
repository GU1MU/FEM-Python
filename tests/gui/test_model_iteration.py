from __future__ import annotations

import json

import pytest

from fem.application import (
    ModelSession,
    NamedRegion,
    RegionAssignment,
    RevisionConflictError,
    SectionDefinition,
    UnitContext,
)
from fem.core.model import AnalysisStep, GravityLoad, MaterialDefinition
from fem.geometry import LogicalEntityRef, namespace_part_reference
from fem.geometry.recipe_topology import describe_recipe_topology
from fem.geometry.recipes import BoxGeometry, CylinderGeometry, RectangleGeometry
from fem.mesh.settings import LocalMeshControl, MeshSettings
from fem_agent.workspace_catalog import (
    WORKSPACE_DOCUMENT_CATALOG_MAX_BYTES,
    WorkspaceCatalogBridge,
)
from fem_gui.agent_workspace_catalog import FEMWorkspaceCatalogPort
from fem_gui.model_iteration import (
    GeometryEditPolicy,
    GeometryEditUnavailableError,
    MigrationItems,
    ModelIterationService,
    geometry_edit_policy,
)
from fem_gui.workspace import DocumentLineage, FEMWorkspace


def _reference(recipe, kind: str) -> LogicalEntityRef:
    entity = describe_recipe_topology(recipe).entities_of(kind)[0]
    return LogicalEntityRef(entity.logical_id)


def _part_reference(recipe, kind: str) -> LogicalEntityRef:
    return namespace_part_reference("P1", _reference(recipe, kind))


def _rectangle_source() -> tuple[FEMWorkspace, object]:
    recipe = RectangleGeometry("Plate", 10.0, 4.0)
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Plate-Model",
        UnitContext("mm", "N", "MPa"),
        recipe,
    )
    session.replace_part_geometry(
        "P1",
        recipe,
        mesh_settings=MeshSettings(
            1.0,
            cell_shape="quadrilateral",
            local_controls=(LocalMeshControl(_reference(recipe, "edge"), 0.5),),
        ),
    )
    session.replace_named_regions(
        (NamedRegion("Plate-Domain", (_part_reference(recipe, "face"),)),)
    )
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
        (SectionDefinition("Solid", "Steel"),),
        (RegionAssignment("Solid", "Plate-Domain"),),
        (
            AnalysisStep(
                "Step-1",
                gravity_loads=(GravityLoad((0.0, -9.81, 0.0), "Plate-Domain"),),
            ),
        ),
    )
    workspace = FEMWorkspace()
    document = workspace.add_model(session, session.projection_snapshot())
    workspace.activate(document)
    return workspace, document


def _box_source() -> tuple[FEMWorkspace, object]:
    recipe = BoxGeometry("Box", 2.0, 1.0, 0.5)
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Solid-Model",
        UnitContext("mm", "N", "MPa"),
        recipe,
    )
    session.replace_part_geometry(
        "P1",
        recipe,
        mesh_settings=MeshSettings(
            0.5,
            cell_shape="hexahedron",
            local_controls=(LocalMeshControl(_reference(recipe, "edge"), 0.2),),
        ),
    )
    session.replace_named_regions(
        (NamedRegion("Solid-Domain", (_part_reference(recipe, "body"),)),)
    )
    dependent = AnalysisStep(
        "Dependent",
        gravity_loads=(GravityLoad((0.0, -9.81, 0.0), "Solid-Domain"),),
    )
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 1.0}),),
        (SectionDefinition("Solid", "Steel"),),
        (RegionAssignment("Solid", "Solid-Domain"),),
        (dependent, AnalysisStep("Independent")),
    )
    workspace = FEMWorkspace()
    document = workspace.add_model(session, session.projection_snapshot())
    workspace.activate(document)
    return workspace, document


def test_geometry_edit_policy_uses_in_place_until_downstream_state_exists() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Geometry",
        UnitContext("mm", "N", "MPa"),
        RectangleGeometry("Plate", 2.0, 1.0),
    )
    workspace = FEMWorkspace()
    document = workspace.add_model(session, session.projection_snapshot())

    assert geometry_edit_policy(document) is GeometryEditPolicy.IN_PLACE

    session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 1.0}),),
        (),
        (),
        (),
    )
    workspace.update_projection(document, session.projection_snapshot())
    assert geometry_edit_policy(document) is GeometryEditPolicy.BRANCH

    result_document = workspace.add_result(ModelSession())
    with pytest.raises(GeometryEditUnavailableError, match="model document"):
        geometry_edit_policy(result_document)


def test_migration_items_are_bounded_without_blocking_a_model_branch() -> None:
    items = MigrationItems(
        preserved=(f"preserved-{index}" for index in range(12)),
        rewritten=(f"rewritten-{index}" for index in range(10)),
        dropped=(f"dropped-{index}" for index in range(8)),
    )

    assert len(items.preserved) == 6
    assert len(items.rewritten) == 6
    assert len(items.dropped) == 6
    assert items.truncated
    assert items.omitted_item_count == 12
    assert items.to_dict()["truncated"] is True


def test_compatible_geometry_branch_migrates_inputs_and_resets_derived_state() -> None:
    workspace, source = _rectangle_source()
    source_before = source.session.project_snapshot_for_branch()
    source_revision = source.session.session_revision
    source_project_revision = source.session.project_revision

    result = ModelIterationService(workspace).branch_geometry_edit(
        source.document_id,
        "P1",
        RectangleGeometry("Plate-Resized", 12.0, 5.0),
    )

    child = result.document
    after = child.session.projection_snapshot()
    assert source.session.project_snapshot_for_branch() == source_before
    assert child.document_id != source.document_id
    assert child.session.session_id != source.session.session_id
    assert workspace.active_document_id == child.document_id
    assert child.source_path is None
    assert child.lineage is not None
    assert child.lineage.source_document_id == source.document_id
    assert child.lineage.source_session_id == source.session.session_id
    assert child.lineage.source_session_revision == source_revision
    assert child.lineage.source_project_revision == source_project_revision
    assert child.lineage.source_run_id is None
    assert after.parts[0].geometry_recipe == RectangleGeometry(
        "Plate-Resized", 12.0, 5.0
    )
    assert tuple(after.named_regions) == ("Plate-Domain",)
    assert [value.name for value in after.materials] == ["Steel"]
    assert [value.name for value in after.sections] == ["Solid"]
    assert len(after.assignments) == 1
    assert [value.name for value in after.steps] == ["Step-1"]
    assert after.artifact is None
    assert not after.validations
    assert not after.runs
    assert not after.result_generations
    assert result.report.requires_remesh
    assert result.report.named_regions.preserved == ("Plate-Domain",)
    assert result.report.mesh_settings.preserved == ("global",)
    assert result.report.local_mesh_controls.preserved
    assert result.report.materials.preserved == ("Steel",)
    assert result.report.sections.preserved == ("Solid",)
    assert result.report.assignments.preserved == ("Plate-Domain:Solid",)
    assert result.report.analysis_steps.preserved == ("Step-1",)
    assert result.report.to_dict()["runs"] == "not_migrated"

    catalog = (
        WorkspaceCatalogBridge(FEMWorkspaceCatalogPort(workspace)).catalog().to_dict()
    )
    child_summary = next(
        item
        for item in catalog["documents"]
        if item["target"]["document_id"] == str(child.document_id)
    )
    assert child_summary["lineage"] == {
        "source_document_id": str(source.document_id),
        "source_session_id": source.session.session_id,
        "source_session_revision": source_revision,
        "source_project_revision": source_project_revision,
        "source_run_id": None,
        "reason": "geometry_edit",
    }
    encoded = json.dumps(
        catalog,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert len(encoded) <= WORKSPACE_DOCUMENT_CATALOG_MAX_BYTES
    assert "path" not in str(catalog).casefold()


def test_topology_change_reports_dropped_dependencies_and_retained_definitions() -> (
    None
):
    workspace, source = _box_source()

    result = ModelIterationService(workspace).branch_geometry_edit(
        source.document_id,
        "P1",
        CylinderGeometry("Cylinder", 1.0, 2.0),
    )
    after = result.document.session.projection_snapshot()

    assert not after.named_regions
    assert not after.assignments
    assert [step.name for step in after.steps] == ["Independent"]
    assert [material.name for material in after.materials] == ["Steel"]
    assert [section.name for section in after.sections] == ["Solid"]
    assert result.report.named_regions.dropped == ("Solid-Domain",)
    assert result.report.local_mesh_controls.dropped
    assert result.report.mesh_settings.rewritten == ("global",)
    assert result.report.materials.preserved == ("Steel",)
    assert result.report.sections.preserved == ("Solid",)
    assert result.report.assignments.dropped == ("Solid-Domain:Solid",)
    assert result.report.analysis_steps.preserved == ("Independent",)
    assert result.report.analysis_steps.dropped == ("Dependent",)


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    (
        ({"source_run_id": "run-foreign"}, ValueError),
        ({"expected_source_session_revision": -1}, RevisionConflictError),
    ),
)
def test_branch_precondition_failures_leave_workspace_and_source_unchanged(
    kwargs,
    error_type,
) -> None:
    workspace, source = _rectangle_source()
    before = source.session.project_snapshot_for_branch()
    active_before = workspace.active_document_id

    with pytest.raises(error_type):
        ModelIterationService(workspace).branch_geometry_edit(
            source.document_id,
            "P1",
            RectangleGeometry("Changed", 8.0, 3.0),
            **kwargs,
        )

    assert workspace.document_count == 1
    assert workspace.active_document_id == active_before
    assert source.session.project_snapshot_for_branch() == before


def test_invalid_recipe_and_activation_failure_roll_back_workspace(monkeypatch) -> None:
    workspace, source = _rectangle_source()
    service = ModelIterationService(workspace)
    source_before = source.session.project_snapshot_for_branch()
    active_before = workspace.active_document_id

    with pytest.raises((TypeError, ValueError)):
        service.branch_geometry_edit(source.document_id, "P1", object())
    assert workspace.document_count == 1
    assert workspace.active_document_id == active_before

    original_activate = workspace.activate

    def fail_child_activation(document):
        target_id = getattr(document, "document_id", document)
        if target_id != source.document_id:
            raise RuntimeError("activation failed")
        return original_activate(document)

    monkeypatch.setattr(workspace, "activate", fail_child_activation)
    with pytest.raises(RuntimeError, match="activation failed"):
        service.branch_geometry_edit(
            source.document_id,
            "P1",
            RectangleGeometry("Changed", 8.0, 3.0),
        )

    assert workspace.document_count == 1
    assert workspace.active_document_id == active_before
    assert workspace.active_kind == "model"
    assert source.session.project_snapshot_for_branch() == source_before


def test_lineage_catalog_truncates_to_budget_and_retains_active_document() -> None:
    workspace = FEMWorkspace()
    source = workspace.add_model(ModelSession(), display_name="Source")
    lineage = DocumentLineage(
        source_document_id=source.document_id,
        source_session_id=source.session.session_id,
        source_session_revision=source.session.session_revision,
        source_project_revision=source.session.project_revision,
    )
    active = source
    for index in range(80):
        active = workspace.add_model(
            ModelSession(),
            display_name=f"Iteration-{index}-" + "几何" * 30,
            lineage=lineage,
        )
    workspace.activate(active)

    catalog = (
        WorkspaceCatalogBridge(FEMWorkspaceCatalogPort(workspace)).catalog().to_dict()
    )
    encoded = json.dumps(
        catalog,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert len(encoded) <= WORKSPACE_DOCUMENT_CATALOG_MAX_BYTES
    assert catalog["truncated"] is True
    assert catalog["active_target"]["document_id"] == str(active.document_id)
    assert all("path" not in str(item).casefold() for item in catalog["documents"])
