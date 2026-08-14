from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fem.application import ModelSession, RevisionConflictError
from fem.application.capabilities import describe_session_authoring
from fem.application.results import (
    ElementResultInspectionRequest,
    FieldState,
    NodeResultInspectionRequest,
    ResultMaterializationUnavailableError,
    ResultQuery,
    build_archived_result_provider,
    prepare_result_export_snapshot,
)
import fem.application.session as session_module
from fem.io import (
    dumps_result_csv,
    dumps_result_vtk,
    load_result_archive,
    read_result_csv,
    read_result_vtk,
    save_result_archive,
    write_result_csv,
    write_result_vtk,
)
from fem.application.results import project_scalar_field_topology
from tests.io.test_result_archive_v1 import _snapshot
from tests.helpers.phase8_result_characterization import (
    make_beam_field_characterization_result,
    make_continuum_nodal_semantics_result,
    make_truss_field_characterization_result,
)


@pytest.mark.parametrize(
    "builder",
    (
        make_continuum_nodal_semantics_result,
        make_truss_field_characterization_result,
        make_beam_field_characterization_result,
    ),
)
def test_result_archive_install_is_result_only_and_consumable(
    builder,
    tmp_path: Path,
) -> None:
    archive = _snapshot(builder, "install")
    path = tmp_path / "saved.femres"
    save_result_archive(path, archive)
    loaded = load_result_archive(path)

    session = ModelSession()
    delta = session.replace_from_result_archive(loaded)
    snapshot = session.snapshot()
    provider = session.current_result_provider()

    assert delta.accepted
    assert snapshot.source_kind == "result"
    assert snapshot.result_only
    assert snapshot.path == path
    assert snapshot.result_path == path
    assert snapshot.saved_result_generation == archive.materialization.generation
    assert snapshot.unsaved_result_count == 0
    assert provider is not None and provider.is_archived
    assert provider.model_result is None
    assert provider.snapshot.topology is provider.model_projection.topology
    assert provider.archive_id == archive.archive_id
    assert session.current_result().result is None
    assert session.current_result().result_id != archive.source.result_id
    assert session.current_result().provenance.run_id != archive.source.run_id
    assert session.result_origin == archive.origin
    assert session.can_save is False
    assert session.can_check() is False
    assert session.can_submit() is False

    ready = next(item for item in provider.catalog().fields if item.state.value == "ready")
    query = provider.query(
        ResultQuery(
            field_key=ready.key,
            component=ready.descriptor.default_component,
        )
    )
    assert query.records
    inspection = provider.inspect_result(
        ElementResultInspectionRequest(archive.topology.element_ids[0])
    )
    assert inspection.fields
    export = prepare_result_export_snapshot(
        provider.snapshot,
        provider.catalog().default_selection,
    )
    assert dumps_result_csv(export)
    assert dumps_result_vtk(export)


@pytest.mark.parametrize(
    "builder",
    (
        make_continuum_nodal_semantics_result,
        make_truss_field_characterization_result,
        make_beam_field_characterization_result,
    ),
)
def test_result_archive_save_after_install_preserves_origin_and_generation(
    builder,
    tmp_path: Path,
) -> None:
    archive = _snapshot(builder, "resave")
    source_path = tmp_path / "source.femres"
    target_path = tmp_path / "resaved.femres"
    save_result_archive(source_path, archive)
    session = ModelSession()
    session.replace_from_result_archive(load_result_archive(source_path))
    run_id = session.snapshot().selected_run_id
    save = session.prepare_result_archive_save(run_id)

    assert save.archive.origin == archive.origin
    assert save.archive.model_projection.summaries == archive.model_projection.summaries
    save_result_archive(target_path, save.archive)
    accepted = session.accept_result_archive_saved(save.token, target_path)
    assert accepted.accepted, accepted
    reloaded = load_result_archive(target_path).snapshot

    assert reloaded.origin == archive.origin
    assert reloaded.materialization.generation == archive.materialization.generation
    assert reloaded.source == save.source
    assert session.snapshot().result_path == target_path
    assert session.snapshot().saved_result_generation == archive.materialization.generation
    assert session.snapshot().unsaved_result_count == 0


def test_result_archive_install_rebinds_all_source_keys_without_array_copy() -> None:
    archive = _snapshot(make_continuum_nodal_semantics_result, "rebind")
    before_topology = archive.topology
    session = ModelSession()
    session.replace_from_result_archive(archive, Path("result.femres"))
    provider = session.current_result_provider()
    assert provider is not None
    assert provider.snapshot.source == provider.catalog().source
    assert provider.snapshot.topology.source == provider.source
    assert provider.snapshot.topology is provider.model_projection.topology
    assert provider.snapshot.topology is not before_topology
    assert provider.catalog().source == provider.source
    assert session.current_result().output_report.source == provider.source
    for installed, original in zip(provider.snapshot.fields, archive.fields, strict=True):
        assert installed.source == provider.source
        assert installed is not original
        assert installed._values is original._values
        assert installed.locations is original.locations
        assert installed.key == original.key
    assert provider.snapshot.topology._node_coordinates is before_topology._node_coordinates
    assert provider.snapshot.topology._nodal_displacements is before_topology._nodal_displacements
    assert provider.source.result_id != archive.source.result_id
    assert provider.source.session_id != archive.source.session_id
    assert provider.source.artifact_id != archive.source.artifact_id
    assert provider.archive_origin == archive.origin
    assert provider.archive_origin.provenance == archive.origin.provenance
    assert session.find_run(session.snapshot().selected_run_id).name == archive.run.name


def test_result_archive_install_without_path_remains_unsaved_in_memory() -> None:
    archive = _snapshot(make_continuum_nodal_semantics_result, "in-memory")
    session = ModelSession()

    delta = session.replace_from_result_archive(archive)
    snapshot = session.snapshot()

    assert delta.accepted
    assert snapshot.result_only
    assert snapshot.path is None
    assert snapshot.result_path is None
    assert snapshot.unsaved_result_count == 1
    assert snapshot.unsaved_result_run_ids == (snapshot.displayed_result_run_id,)
    assert session.current_result_provider() is not None


def test_result_archive_install_failure_is_atomic() -> None:
    archive = _snapshot(make_truss_field_characterization_result, "atomic")
    session = ModelSession()
    session.new_native_project("Before")
    before = session.snapshot()
    for empty_path in ("", "   ", Path(""), Path(".")):
        with pytest.raises(ValueError):
            session.replace_from_result_archive(archive, empty_path)
        assert session.snapshot().session_id == before.session_id
    after = session.snapshot()
    assert after.session_id == before.session_id
    assert after.session_revision == before.session_revision
    assert after.source_kind == before.source_kind
    assert after.model_name == before.model_name


def test_result_archive_replaces_nonempty_session_and_rejects_stale_revision(
    tmp_path: Path,
) -> None:
    first = _snapshot(make_continuum_nodal_semantics_result, "first")
    second = _snapshot(make_truss_field_characterization_result, "second")
    session = ModelSession()
    session.replace_from_result_archive(first, tmp_path / "first.femres")
    before = session.snapshot()
    session.replace_from_result_archive(
        second,
        tmp_path / "second.femres",
        expected_session_revision=before.session_revision,
    )
    replaced = session.snapshot()
    assert replaced.session_id != before.session_id
    assert replaced.artifact.artifact_id != before.artifact.artifact_id
    assert replaced.displayed_result.result_id != before.displayed_result.result_id
    stable = session.snapshot()
    with pytest.raises(RevisionConflictError):
        session.replace_from_result_archive(
            first,
            tmp_path / "stale.femres",
            expected_session_revision=before.session_revision,
        )
    after = session.snapshot()
    assert after.session_id == stable.session_id
    assert after.session_revision == stable.session_revision
    assert after.displayed_result.result_id == stable.displayed_result.result_id


def test_result_archive_candidate_failure_preserves_current_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first = _snapshot(make_beam_field_characterization_result, "stable")
    second = _snapshot(make_continuum_nodal_semantics_result, "candidate")
    session = ModelSession()
    session.replace_from_result_archive(first, tmp_path / "stable.femres")
    before = session.snapshot()

    def reject(_archive):
        raise ValueError("candidate provider rejected")

    monkeypatch.setattr(session_module, "build_archived_result_provider", reject)
    with pytest.raises(ValueError, match="candidate provider rejected"):
        session.replace_from_result_archive(second, tmp_path / "candidate.femres")
    after = session.snapshot()
    assert after.session_id == before.session_id
    assert after.session_revision == before.session_revision
    assert after.result_path == before.result_path
    assert after.displayed_result.result_id == before.displayed_result.result_id


def test_archive_provider_does_not_derive_missing_fields() -> None:
    archive = _snapshot(make_continuum_nodal_semantics_result, "missing")
    executed = {
        key
        for request in archive.run.output_report.requests
        for variable in request.variables
        if variable.status.value == "executed"
        for key in variable.field_keys
    }
    target = next(
        item.key
        for item in archive.catalog.fields
        if item.key not in executed
    )
    catalog = replace(
        archive.catalog,
        fields=tuple(
            replace(item, state=FieldState.LAZY)
            if item.key == target
            else item
            for item in archive.catalog.fields
        ),
    )
    materialization = replace(
        archive.materialization,
        fields=tuple(item for item in archive.fields if item.key != target),
    )
    archive = replace(
        archive,
        catalog=catalog,
        materialization=materialization,
        model_projection=replace(
            archive.model_projection,
            topology=materialization.topology,
        ),
    )
    provider = build_archived_result_provider(archive)
    with pytest.raises(ResultMaterializationUnavailableError) as error:
        provider.materialize((target,))
    assert error.value.code == "result.materialization.unavailable"
    assert error.value.keys == (target,)


def test_result_only_authoring_projection_is_unavailable() -> None:
    archive = _snapshot(make_beam_field_characterization_result, "caps")
    session = ModelSession()
    session.replace_from_result_archive(archive, Path("caps.femres"))
    projection = describe_session_authoring(session.snapshot())
    assert projection.targets == ()
    assert projection.step_lifecycle == ()
    assert projection.operation("section.create").can_enter is False
    assert projection.operation("output_request.create").can_submit is False


def _query_signature(result):
    return tuple(
        (record.location, float(record.value))
        for record in result.records
    )


def _inspection_signature(result):
    return tuple(
        (
            field.availability.key,
            field.availability.descriptor,
            field.availability.state,
            tuple(_query_signature(component) for component in field.component_results),
        )
        for field in result.fields
    )


@pytest.mark.parametrize(
    "builder",
    (
        make_continuum_nodal_semantics_result,
        make_truss_field_characterization_result,
        make_beam_field_characterization_result,
    ),
)
def test_result_archive_three_model_query_inspection_export_and_contour_parity(
    builder,
    tmp_path: Path,
) -> None:
    archive = _snapshot(builder, "parity")
    before = build_archived_result_provider(archive)
    source_path = tmp_path / "parity.femres"
    save_result_archive(source_path, archive)
    session = ModelSession()
    session.replace_from_result_archive(load_result_archive(source_path))
    after = session.current_result_provider()
    assert after is not None

    assert tuple(before.catalog().fields) == tuple(after.catalog().fields)
    assert before.catalog().default_selection == after.catalog().default_selection
    assert before.model_projection.summaries == after.model_projection.summaries
    assert before.model_projection.named_region_node_ids == after.model_projection.named_region_node_ids
    assert before.model_projection.named_region_element_ids == after.model_projection.named_region_element_ids
    for name, ids in archive.model_projection.named_region_node_ids.items():
        assert after.named_region_node_ids(name) == tuple(ids)
    for name, ids in archive.model_projection.named_region_element_ids.items():
        assert after.named_region_element_ids(name) == tuple(ids)

    selection = before.catalog().default_selection
    assert selection is not None
    before_query = before.query(
        ResultQuery(field_key=selection.field_key, component=selection.component)
    )
    after_query = after.query(
        ResultQuery(field_key=selection.field_key, component=selection.component)
    )
    assert _query_signature(before_query) == _query_signature(after_query)

    node_id = archive.topology.node_ids[0]
    element_id = archive.topology.element_ids[0]
    assert _inspection_signature(
        before.inspect_result(NodeResultInspectionRequest(node_id))
    ) == _inspection_signature(
        after.inspect_result(NodeResultInspectionRequest(node_id))
    )
    assert _inspection_signature(
        before.inspect_result(ElementResultInspectionRequest(element_id))
    ) == _inspection_signature(
        after.inspect_result(ElementResultInspectionRequest(element_id))
    )

    before_export = prepare_result_export_snapshot(before.snapshot, selection)
    after_export = prepare_result_export_snapshot(after.snapshot, selection)
    before_csv = tmp_path / "before.csv"
    after_csv = tmp_path / "after.csv"
    before_vtk = tmp_path / "before.vtk"
    after_vtk = tmp_path / "after.vtk"
    write_result_csv(before_csv, before_export)
    write_result_csv(after_csv, after_export)
    write_result_vtk(before_vtk, before_export, deformation_scale=0.0)
    write_result_vtk(after_vtk, after_export, deformation_scale=0.0)
    csv_before = read_result_csv(before_csv)
    csv_after = read_result_csv(after_csv)
    assert csv_before.materialization_generation == csv_after.materialization_generation
    assert tuple((item.location, item.value) for item in csv_before.records) == tuple(
        (item.location, item.value) for item in csv_after.records
    )
    vtk_before = read_result_vtk(before_vtk)
    vtk_after = read_result_vtk(after_vtk)
    assert vtk_before.materialization_generation == vtk_after.materialization_generation
    assert vtk_before.points == vtk_after.points
    assert vtk_before.cells == vtk_after.cells
    assert vtk_before.values == vtk_after.values

    for scale in (0.0, 1.0):
        projected_before = project_scalar_field_topology(before_export, scale)
        projected_after = project_scalar_field_topology(after_export, scale)
        np.testing.assert_array_equal(projected_before.points, projected_after.points)
        assert projected_before.cells == projected_after.cells
        np.testing.assert_array_equal(projected_before.values, projected_after.values)
