from __future__ import annotations

from pathlib import Path

import pytest

from fem.application.results import (
    FieldPosition,
    ResultQuery,
    ResultSourceKey,
    ScalarFieldSelection,
    build_result_provider,
    execute_output_requests,
    prepare_result_export_snapshot,
)
from fem.core.model import OutputRequest
from fem.io.result_csv import (
    dumps_result_components_csv,
    dumps_result_csv,
    read_result_csv,
    write_result_csv,
)
from fem.io.result_vtk import read_result_vtk, write_result_vtk
from fem.post.stress import beam
from tests.helpers.phase8_result_characterization import (
    make_beam_field_characterization_result,
)


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        "beam-result",
        "session",
        "artifact",
        1,
        "Step-1",
        "beam-run",
    )


def _executed_provider(monkeypatch: pytest.MonkeyPatch | None = None):
    provider = build_result_provider(
        _source(),
        make_beam_field_characterization_result(),
    )
    calls: list[object] = []
    if monkeypatch is not None:
        original = beam.recover_integration_point_stress

        def counted(result, *, checkpoint=None):
            calls.append(result)
            return original(result, checkpoint=checkpoint)

        monkeypatch.setattr(beam, "recover_integration_point_stress", counted)
    outcome = execute_output_requests(
        provider,
        (OutputRequest("field", "element", ("S",)),),
    )
    return outcome.provider_draft, outcome, calls


def test_s_request_recovers_once_and_publishes_four_integration_point_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, outcome, calls = _executed_provider(monkeypatch)

    assert len(calls) == 1
    fields = outcome.eager_patch.fields
    assert len(fields) == 4
    point_fields = tuple(
        field
        for field in fields
        if field.key.request.field_id.position is FieldPosition.INTEGRATION_POINT
    )
    assert tuple(
        field.key.request.field_id.section_point_number
        for field in point_fields
    ) == (1, 2, 3, 4)
    assert all(
        field.descriptor.columns
        == (
            "S11",
            "S22",
            "S12",
            "Mises",
            "MaxPrincipal",
            "MidPrincipal",
            "MinPrincipal",
        )
        for field in point_fields
    )
    assert all(provider.field(field.key) is field for field in fields)


def test_four_point_keys_query_same_beam_ip_without_crossing_points() -> None:
    provider, _outcome, _calls = _executed_provider()
    point_fields = tuple(
        field
        for field in provider.snapshot.fields
        if field.key.request.field_id.position is FieldPosition.INTEGRATION_POINT
        and field.key.request.field_id.section_point_number is not None
    )

    queried = []
    for field in point_fields:
        point_number = field.key.request.field_id.section_point_number
        result = provider.query(
            ResultQuery(
                field.key,
                "S11",
                element_ids=(30,),
            )
        )
        assert len(result.records) == 1
        location = result.records[0].location
        assert location.local_node is None
        assert location.integration_point == 1
        assert location.section_point is not None
        assert location.section_point.number == point_number
        queried.append(
            (
                point_number,
                location.section_point.local_y,
                location.section_point.local_z,
                result.records[0].value,
            )
        )

    assert [row[:3] for row in queried] == [
        (1, 0.5, 1.0),
        (2, -0.5, 1.0),
        (3, -0.5, -1.0),
        (4, 0.5, -1.0),
    ]
    assert len({row[3] for row in queried}) > 1


def test_csv_and_vtk_preserve_one_selected_section_point_identity(
    tmp_path: Path,
) -> None:
    provider, _outcome, _calls = _executed_provider()
    field = next(
        field
        for field in provider.snapshot.fields
        if field.key.request.field_id.position is FieldPosition.INTEGRATION_POINT
        and field.key.request.field_id.section_point_number == 2
    )
    export = prepare_result_export_snapshot(
        provider.snapshot,
        ScalarFieldSelection(field.key, "S11"),
    )

    csv_path = tmp_path / "section-point.csv"
    csv_text = dumps_result_csv(export)
    assert "section_point_number" in csv_text.splitlines()[0]
    assert "section_point_local_y" in csv_text.splitlines()[0]
    assert "section_point_local_z" in csv_text.splitlines()[0]
    write_result_csv(csv_path, export)
    csv_readback = read_result_csv(csv_path)
    assert csv_readback.selection == export.selection
    assert {
        record.location.section_point
        for record in csv_readback.records
    } == {location.section_point for location in field.locations}

    vtk_path = tmp_path / "section-point.vtk"
    write_result_vtk(vtk_path, export)
    vtk_readback = read_result_vtk(vtk_path)
    assert vtk_readback.selection == export.selection
    projected_points = {
        identity.section_point
        for identity in (
            *vtk_readback.point_locations,
            *vtk_readback.cell_locations,
        )
        if identity is not None and identity.section_point is not None
    }
    assert projected_points == {location.section_point for location in field.locations}


def test_selected_csv_exports_ip_and_section_point_identity() -> None:
    provider, _outcome, _calls = _executed_provider()
    field = next(
        field
        for field in provider.snapshot.fields
        if field.key.request.field_id.position is FieldPosition.INTEGRATION_POINT
        and field.key.request.field_id.section_point_number == 1
    )
    snapshot = prepare_result_export_snapshot(
        provider.snapshot,
        ScalarFieldSelection(field.key, "S11"),
    )

    lines = dumps_result_components_csv((snapshot,)).splitlines()

    assert lines[0].endswith(",S11")
    assert "integration_point" in lines[0]
    assert "section_point_number" in lines[0]
