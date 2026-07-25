from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from fem.application import (
    FeatureRecord,
    ModelSession,
    NamedRegion,
    NativePart,
    ProjectSnapshot,
    RegionAssignment,
    SectionDefinition,
    SessionSnapshot,
)
from fem_gui import document
from tests.helpers.model_builders import make_static_pull_truss_model


def test_document_module_is_only_a_headless_compatibility_facade() -> None:
    assert document.FEMDocument is SessionSnapshot
    assert document.ModelSession is ModelSession
    assert document.ProjectSnapshot is ProjectSnapshot
    assert document.NativePart is NativePart
    assert document.FeatureRecord is FeatureRecord
    assert document.NamedRegion is NamedRegion
    assert document.SectionDefinition is SectionDefinition
    assert document.RegionAssignment is RegionAssignment
    assert not hasattr(document, "WorkflowState")


def test_session_snapshot_has_no_stored_legacy_currentness_flags() -> None:
    stored_fields = {item.name for item in fields(SessionSnapshot)}

    assert "mesh_current" not in stored_fields
    assert "model_checked" not in stored_fields
    assert "results_current" not in stored_fields


def test_document_snapshot_is_frozen_and_does_not_expose_mutable_collections() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry(
        (NativePart(),),
        {"shape": "rectangle"},
        feature_history=(FeatureRecord("Sketch-1", "sketch"),),
    )
    session.replace_named_regions(
        (NamedRegion("Fixed", "edge", (1,)),)
    )
    snapshot = session.snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.source_kind = "imported"
    with pytest.raises(TypeError):
        snapshot.named_regions["Loaded"] = NamedRegion(
            "Loaded",
            "edge",
            (2,),
        )

    assert isinstance(snapshot.parts, tuple)
    assert isinstance(snapshot.feature_history, tuple)
    assert isinstance(snapshot.material_definitions, tuple)
    assert isinstance(snapshot.section_definitions, tuple)
    assert isinstance(snapshot.region_assignments, tuple)
    assert isinstance(snapshot.analysis_definitions, tuple)


def test_imported_session_derives_reload_and_mesh_current_from_artifact() -> None:
    session = ModelSession()
    source = Path("model.inp")
    task = session.prepare_import(source)

    session.accept_imported_model(
        task.token,
        make_static_pull_truss_model(),
    )
    snapshot = session.snapshot()

    assert snapshot.source_kind == "imported"
    assert snapshot.path == source
    assert snapshot.can_reload
    assert snapshot.has_model
    assert snapshot.mesh_current
    assert not hasattr(snapshot, "model_checked")
    assert not hasattr(snapshot, "results_current")


def test_geometry_change_drops_the_previous_model_artifact() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry(
        (NativePart(),),
        {"shape": "rectangle"},
    )
    task = session.prepare_mesh_generation()
    session.accept_generated_model(
        task.token,
        make_static_pull_truss_model(),
    )
    assert session.snapshot().has_model

    session.replace_geometry(
        (NativePart(),),
        {"shape": "circle"},
        feature_history=(FeatureRecord("Sketch-1", "sketch"),),
    )
    snapshot = session.snapshot()

    assert snapshot.source_kind == "native"
    assert not snapshot.has_model
    assert not snapshot.has_result
    assert not snapshot.mesh_current
    assert [feature.name for feature in snapshot.feature_history] == [
        "Sketch-1"
    ]


def test_close_replaces_the_session_identity_and_clears_project_state() -> None:
    session = ModelSession()
    session.new_native_project()
    previous_session_id = session.session_id

    session.close()
    snapshot = session.snapshot()

    assert snapshot.session_id != previous_session_id
    assert snapshot.source_kind is None
    assert snapshot.geometry_recipe is None
    assert snapshot.mesh_settings is None
    assert snapshot.parts == ()
    assert snapshot.artifact is None
    assert not snapshot.dirty
