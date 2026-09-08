from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path


import numpy as np
import pytest

from fem.application.results import (
    FieldPosition,
    PhysicalQuantity,
    ResultQuery,
    ResultVariable,
    ScalarFieldSelection,
    prepare_result_export_snapshot,
)
from fem.elements import (
    get_element_kernel,
)
from fem.io import load_result_archive, save_result_archive
from fem.io.result_csv import dumps_result_csv
from fem.io.result_vtk import read_result_vtk, write_result_vtk


from tests.helpers.beam_section_builders import (
    _LENGTH,
    _AXIAL_FORCE,
    _LOCAL_Y_FORCE,
    _LOCAL_Z_FORCE,
    _TORQUE,
    _SECTION_CASES,
    _POINT_COLUMNS,
    _solve_and_request_stress,
    _independent_integration_point_oracle,
    _stress_fields,
    _field,
    _archive_snapshot,
)


@pytest.mark.parametrize(("section_type", "dimensions"), _SECTION_CASES)
def test_section_points_keep_identity_through_query_csv_and_archive(
    section_type: str,
    dimensions: dict[str, float],
    tmp_path: Path,
) -> None:
    result, provider, outcome = _solve_and_request_stress(
        section_type,
        dimensions,
    )
    point_oracle = _independent_integration_point_oracle(
        section_type,
        dimensions,
    )
    point_fields = _stress_fields(provider)
    assert tuple(
        field.key.request.field_id.section_point_number for field in point_fields
    ) == (1, 2, 3, 4)
    for field in point_fields:
        assert field.descriptor.columns == _POINT_COLUMNS
        number = field.key.request.field_id.section_point_number
        row = next(
            index for index, location in enumerate(field.locations)
            if location.element_id == 10 and location.integration_point == 1
        )
        np.testing.assert_allclose(
            field.values[row, :1], point_oracle[number], rtol=2e-10, atol=1e-8,
        )

    selected_field = point_fields[2]
    query = ResultQuery(
        selected_field.key,
        "S11",
        element_ids=(10,),
    )
    queried = provider.query(query)
    assert len(queried.records) == 1
    queried_location = queried.records[0].location
    assert queried_location.section_point is not None
    assert queried_location.section_point.number == 3
    assert queried.records[0].value == pytest.approx(point_oracle[3][0])

    selection = ScalarFieldSelection(selected_field.key, "S11")
    export = prepare_result_export_snapshot(provider.snapshot, selection)
    csv_rows = tuple(csv.DictReader(StringIO(dumps_result_csv(export, queried))))
    assert len(csv_rows) == 1
    assert csv_rows[0]["field_position"] == "integration_point"
    assert csv_rows[0]["integration_point"] == "1"
    assert csv_rows[0]["section_point_number"] == "3"
    assert float(csv_rows[0]["value"]) == pytest.approx(point_oracle[3][0])

    archive_path = tmp_path / f"{section_type}.femres"
    save_result_archive(
        archive_path,
        _archive_snapshot(provider, outcome, section_type),
    )
    loaded = load_result_archive(archive_path).snapshot
    loaded_points = tuple(
        field
        for field in loaded.fields
        if field.key.request.field_id.position is FieldPosition.INTEGRATION_POINT
        and field.key.request.field_id.section_point_number is not None
    )
    assert tuple(
        field.key.request.field_id.section_point_number for field in loaded_points
    ) == (1, 2, 3, 4)
    assert all(
        location.section_point is not None
        and location.section_point.number
        == field.key.request.field_id.section_point_number
        for field in loaded_points
        for location in field.locations
    )


def test_public_section_forces_keep_typed_ip_identity_across_consumers(
    tmp_path: Path,
) -> None:
    result, provider, outcome = _solve_and_request_stress(
        "rectangle",
        {"height": 0.4, "width": 0.2},
    )
    sf = _field(provider, ResultVariable.SF)
    sm = _field(provider, ResultVariable.SM)

    assert sf.descriptor.quantity is PhysicalQuantity.FORCE
    assert sf.descriptor.columns == ("N", "Vy", "Vz")
    assert sm.descriptor.quantity is PhysicalQuantity.MOMENT
    assert sm.descriptor.columns == ("T", "My", "Mz")
    assert sf.key.request.field_id.section_point_number is None
    assert sm.key.request.field_id.section_point_number is None
    assert len(sf.locations) == len(sm.locations) == 1
    assert all(
        location.element_id == 10
        and location.integration_point == 1
        and location.section_point is None
        for field in (sf, sm)
        for location in field.locations
    )
    owner = get_element_kernel("Beam2").local_integration_point_forces(
        result.model.mesh,
        result.model.mesh.elements[0],
        result.U,
    )
    assert (owner.N, owner.Vy, owner.Vz, owner.T) == pytest.approx(
        (_AXIAL_FORCE, _LOCAL_Y_FORCE, _LOCAL_Z_FORCE, _TORQUE)
    )
    assert sf.values[0] == pytest.approx((owner.N, owner.Vy, owner.Vz))
    assert sm.values[0] == pytest.approx((owner.T, owner.My, owner.Mz))
    assert (owner.My, owner.Mz) == pytest.approx(
        (
            -_LOCAL_Z_FORCE * _LENGTH / 2.0,
            _LOCAL_Y_FORCE * _LENGTH / 2.0,
        )
    )

    for field, component in (
        (sf, "N"),
        (sf, "Vy"),
        (sf, "Vz"),
        (sm, "T"),
        (sm, "My"),
        (sm, "Mz"),
    ):
        queried = provider.query(
            ResultQuery(field.key, component, element_ids=(10,))
        )
        assert len(queried.records) == 1
        assert queried.records[0].location.integration_point == 1
        assert queried.records[0].location.section_point is None
        export = prepare_result_export_snapshot(
            provider.snapshot,
            ScalarFieldSelection(field.key, component),
        )
        csv_row = next(
            csv.DictReader(StringIO(dumps_result_csv(export, queried)))
        )
        assert (
            csv_row["field_variable"]
            == field.descriptor.field_id.variable.value
        )
        assert csv_row["field_position"] == "integration_point"
        assert csv_row["integration_point"] == "1"
        assert csv_row["section_point_number"] == ""
        vtk_path = tmp_path / f"{field.descriptor.field_id.variable.value}-{component}.vtk"
        write_result_vtk(vtk_path, export)
        vtk = read_result_vtk(vtk_path)
        assert vtk.quantity is field.descriptor.quantity
        assert vtk.selection.component == component
        assert vtk.values == pytest.approx((queried.records[0].value,))
        assert vtk.cell_locations[0].integration_point == 1
        assert vtk.cell_locations[0].section_point is None

    archive_path = tmp_path / "typed-section-forces.femres"
    save_result_archive(
        archive_path,
        _archive_snapshot(provider, outcome, "rectangle"),
    )
    loaded = load_result_archive(archive_path).snapshot
    loaded_by_variable = {
        field.key.request.field_id.variable: field
        for field in loaded.fields
        if field.key.request.field_id.variable
        in {ResultVariable.SF, ResultVariable.SM}
    }
    assert tuple(loaded_by_variable) == (ResultVariable.SF, ResultVariable.SM)
    for variable, original in (
        (ResultVariable.SF, sf),
        (ResultVariable.SM, sm),
    ):
        restored = loaded_by_variable[variable]
        assert restored.key == original.key
        assert restored.descriptor == original.descriptor
        assert restored.locations == original.locations
        np.testing.assert_array_equal(restored.values, original.values)
