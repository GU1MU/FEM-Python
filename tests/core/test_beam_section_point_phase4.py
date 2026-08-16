from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
import math
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from fem.application.results import (
    FieldPosition,
    PhysicalQuantity,
    ResultArchiveModelProjection,
    ResultArchiveOrigin,
    ResultArchiveRun,
    ResultArchiveSnapshot,
    ResultQuery,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
    build_archived_result_provider,
    build_result_provider,
    execute_output_requests,
    prepare_result_export_snapshot,
)
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    FEMModel,
    NodalLoad,
    OutputRequest,
)
from fem.elements import (
    BEAM_FRAME_FIELD_KEY,
    BeamFrameField,
    get_element_kernel,
)
from fem.io import load_result_archive, save_result_archive
from fem.io.result_csv import dumps_result_csv
from fem.io.result_vtk import read_result_vtk, write_result_vtk
from fem.post.stress import beam as beam_stress
from fem.post.stress.beam import recover_integration_point_stress
from fem.solvers import static_linear
from fem_gui.result_presentation import (
    result_provider_section_point_labels,
)
from fem_gui.widgets.result_tree import ResultTree


_LENGTH = 2.0
_AXIAL_FORCE = 12.0
_LOCAL_Y_FORCE = 3.0
_LOCAL_Z_FORCE = -2.0
_TORQUE = 1.5
_POINT_COLUMNS = (
    "S11",
    "S22",
    "S12",
    "Mises",
    "MaxPrincipal",
    "MidPrincipal",
    "MinPrincipal",
)
_SECTION_CASES = (
    ("rectangle", {"height": 0.4, "width": 0.2}),
    ("solid_circle", {"radius": 0.2}),
    ("hollow_circle", {"outer_radius": 0.2, "inner_radius": 0.1}),
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _inline_cantilever(
    section_type: str,
    dimensions: dict[str, float],
) -> FEMModel:
    frame = BeamFrameField.from_rotations(
        _LENGTH,
        np.eye(3),
        np.eye(3),
    )
    mesh = Mesh3D(
        nodes=(
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, _LENGTH, 0.0, 0.0),
        ),
        elements=(
            Element3D(
                10,
                (1, 2),
                "Beam2",
                {
                    "E": 210.0e9,
                    "nu": 0.3,
                    "section_type": section_type,
                    **dimensions,
                    BEAM_FRAME_FIELD_KEY: frame,
                },
            ),
        ),
        dofs_per_node=6,
    )
    return FEMModel(
        mesh=mesh,
        name=f"inline-{section_type}-cantilever",
        steps=(
            AnalysisStep(
                "Load",
                boundaries=(DisplacementConstraint(1, 1, 6),),
                cloads=(
                    NodalLoad(2, 1, _AXIAL_FORCE),
                    NodalLoad(2, 2, _LOCAL_Y_FORCE),
                    NodalLoad(2, 3, _LOCAL_Z_FORCE),
                    NodalLoad(2, 4, _TORQUE),
                ),
            ),
        ),
    )


def _solve_and_request_stress(
    section_type: str,
    dimensions: dict[str, float],
):
    result = static_linear.solve(
        _inline_cantilever(section_type, dimensions),
        "Load",
    )
    source = ResultSourceKey(
        result_id=f"result-{section_type}",
        session_id="phase4-session",
        artifact_id=f"artifact-{section_type}",
        model_revision=1,
        step_name="Load",
        run_id=f"run-{section_type}",
    )
    provider = build_result_provider(source, result)
    outcome = execute_output_requests(
        provider,
        (OutputRequest("field", "element", ("SF", "SM", "S")),),
    )
    return result, outcome.provider_draft, outcome


def _rectangle_torsion_constant(height: float, width: float) -> float:
    long_side = max(height, width)
    short_side = min(height, width)
    series = 0.0
    for odd in range(1, 10000, 2):
        term = math.tanh(odd * math.pi * long_side / (2.0 * short_side)) / odd**5
        series += term
        if term < 1.0e-15:
            break
    correction = 192.0 * short_side * series / (math.pi**5 * long_side)
    return long_side * short_side**3 * (1.0 - correction) / 3.0


def _rectangle_torsion_shear(
    height: float,
    width: float,
    torsion_constant: float,
) -> float:
    long_side = max(height, width)
    short_side = min(height, width)
    inverse_cosh_series = 0.0
    for odd in range(1, 10000, 2):
        argument = odd * math.pi * long_side / (2.0 * short_side)
        term = (0.0 if argument > 40.0 else 1.0 / math.cosh(argument)) / odd**2
        inverse_cosh_series += term
        if term < 1.0e-15:
            break
    series = math.pi**2 / 8.0 - inverse_cosh_series
    shear_per_twist = 8.0 * short_side * series / math.pi**2
    return abs(_TORQUE) * shear_per_twist / torsion_constant


def _independent_integration_point_oracle(
    section_type: str,
    dimensions: dict[str, float],
) -> dict[int, tuple[float, ...]]:
    if section_type == "rectangle":
        height = dimensions["height"]
        width = dimensions["width"]
        area = height * width
        iyy = width * height**3 / 12.0
        izz = height * width**3 / 12.0
        points = (
            (width / 2.0, height / 2.0),
            (-width / 2.0, height / 2.0),
            (-width / 2.0, -height / 2.0),
            (width / 2.0, -height / 2.0),
        )
    else:
        outer_radius = (
            dimensions["radius"]
            if section_type == "solid_circle"
            else dimensions["outer_radius"]
        )
        inner_radius = (
            0.0 if section_type == "solid_circle" else dimensions["inner_radius"]
        )
        area = math.pi * (outer_radius**2 - inner_radius**2)
        iyy = izz = math.pi * (outer_radius**4 - inner_radius**4) / 4.0
        point_radius = (
            outer_radius
            if section_type == "solid_circle"
            else (outer_radius + inner_radius) / 2.0
        )
        points = (
            (point_radius, 0.0),
            (0.0, point_radius),
            (-point_radius, 0.0),
            (0.0, -point_radius),
        )

    axial = _AXIAL_FORCE / area
    moment_y = -_LOCAL_Z_FORCE * _LENGTH / 2.0
    moment_z = _LOCAL_Y_FORCE * _LENGTH / 2.0
    point_values: dict[int, tuple[float, ...]] = {}
    s11_values = []
    for number, (local_y, local_z) in enumerate(points, start=1):
        s11 = axial + moment_y * local_z / iyy - moment_z * local_y / izz
        values = (s11,)
        point_values[number] = values
        s11_values.append(s11)

    return point_values


def _stress_fields(provider):
    return tuple(
        field
        for field in provider.snapshot.fields
        if field.key.request.field_id.variable is ResultVariable.S
        and field.key.request.field_id.position is FieldPosition.INTEGRATION_POINT
        and field.key.request.field_id.section_point_number is not None
    )


def _integration_point_row(field) -> int:
    return next(
        index
        for index, location in enumerate(field.locations)
        if location.element_id == 10 and location.integration_point == 1
    )


def _field(provider, variable: ResultVariable):
    return next(
        field
        for field in provider.snapshot.fields
        if field.key.request.field_id.variable is variable
        and field.key.request.field_id.position
        is FieldPosition.INTEGRATION_POINT
        and field.key.request.field_id.section_point_number is None
    )


def _archive_snapshot(provider, outcome, section_type: str):
    now = datetime(2026, 8, 9, tzinfo=timezone.utc)
    return ResultArchiveSnapshot(
        archive_id=f"archive-{section_type}",
        created_at=now,
        producer_version="phase4-test",
        origin=ResultArchiveOrigin(
            model_name=f"inline-{section_type}-cantilever",
            model_fingerprint="0" * 64,
        ),
        run=ResultArchiveRun(
            "Job-1",
            "Load",
            now,
            output_report=outcome.report,
        ),
        profile=provider.profile,
        catalog=provider.catalog(),
        materialization=provider.snapshot,
        model_projection=ResultArchiveModelProjection(
            provider.snapshot.topology,
        ),
    )


@pytest.mark.parametrize(("section_type", "dimensions"), _SECTION_CASES)
def test_inline_sections_complete_solve_query_csv_archive_and_gui_contract(
    section_type: str,
    dimensions: dict[str, float],
    tmp_path: Path,
) -> None:
    _application()
    result, provider, outcome = _solve_and_request_stress(
        section_type,
        dimensions,
    )
    point_oracle = _independent_integration_point_oracle(
        section_type,
        dimensions,
    )
    fields = _stress_fields(provider)

    assert np.all(np.isfinite(result.U))
    assert np.all(np.isfinite(result.reactions))
    assert len(fields) == 4
    assert all(np.all(np.isfinite(field.values)) for field in fields)

    point_fields = tuple(
        field
        for field in fields
        if field.key.request.field_id.position is FieldPosition.INTEGRATION_POINT
    )
    assert tuple(
        field.key.request.field_id.section_point_number for field in point_fields
    ) == (1, 2, 3, 4)
    assert all(field.descriptor.columns == _POINT_COLUMNS for field in point_fields)

    for field in point_fields:
        point_number = field.key.request.field_id.section_point_number
        assert point_number is not None
        row = _integration_point_row(field)
        np.testing.assert_allclose(
            field.values[row, :1],
            point_oracle[point_number],
            rtol=2.0e-10,
            atol=1.0e-8,
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
    archived_provider = build_archived_result_provider(loaded)
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
    expected_position_labels = (
        {1: "右上", 2: "左上", 3: "左下", 4: "右下"}
        if section_type == "rectangle"
        else {}
    )
    assert (
        result_provider_section_point_labels(archived_provider)
        == expected_position_labels
    )

    tree = ResultTree()
    tree.set_catalog(
        "Load",
        provider.catalog(),
        section_point_labels=result_provider_section_point_labels(provider),
    )
    step_item = tree.topLevelItem(0).child(0)
    stress_item = next(
        step_item.child(index)
        for index in range(step_item.childCount())
        if step_item.child(index).text(0) == "应力 S"
    )
    gui_stress_labels = tuple(
        stress_item.child(index).text(0)
        for index in range(stress_item.childCount())
    )
    expected_point_labels = (
        ("右上", "左上", "左下", "右下")
        if section_type == "rectangle"
        else ("截面点 1", "截面点 2", "截面点 3", "截面点 4")
    )
    assert gui_stress_labels == expected_point_labels
    tree.close()


def test_public_section_forces_keep_typed_ip_identity_across_consumers(
    tmp_path: Path,
) -> None:
    _application()
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

    tree = ResultTree()
    tree.set_catalog("Load", provider.catalog())
    step = tree.topLevelItem(0).child(0)
    variables = {
        step.child(index).text(0): step.child(index)
        for index in range(step.childCount())
    }
    for label, expected_components in (
        ("截面力 SF（积分点）", ("N", "Vy", "Vz")),
        ("截面矩 SM（积分点）", ("T", "My", "Mz")),
    ):
        item = variables[label]
        assert "积分点" in item.text(0)
        assert tuple(
            item.child(index).text(0)
            for index in range(item.childCount())
        ) == expected_components
    tree.close()


def test_sf_sm_and_s_share_one_constitutive_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = static_linear.solve(
        _inline_cantilever("rectangle", {"height": 0.4, "width": 0.2}),
        "Load",
    )
    provider = build_result_provider(
        ResultSourceKey(
            "shared-recovery-result",
            "shared-recovery-session",
            "shared-recovery-artifact",
            1,
            "Load",
            "shared-recovery-run",
        ),
        result,
    )
    original = beam_stress.recover_integration_point_stress
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        beam_stress,
        "recover_integration_point_stress",
        counted,
    )
    outcome = execute_output_requests(
        provider,
        (OutputRequest("field", "element", ("SF", "SM", "S")),),
    )

    assert outcome.report.diagnostics == ()
    assert calls == 1
    assert {
        field.key.request.field_id.variable
        for field in outcome.provider_draft.snapshot.fields
    }.issuperset({ResultVariable.SF, ResultVariable.SM, ResultVariable.S})


def _best_batch_time(action, *, repeats: int) -> float:
    best = math.inf
    for _sample in range(3):
        checksum = 0.0
        started = perf_counter()
        for _repeat in range(repeats):
            checksum += float(action())
        best = min(best, perf_counter() - started)
        assert math.isfinite(checksum)
    return best


def _section_resultant_workload(result) -> float:
    mesh = result.model.mesh
    element = mesh.elements[0]
    kernel = get_element_kernel("Beam2")
    forces = kernel.local_integration_point_forces(mesh, element, result.U)
    return sum(
        abs(value)
        for value in (
            forces.N,
            forces.Vy,
            forces.Vz,
            forces.T,
            forces.My,
            forces.Mz,
        )
    )


def _four_point_workload(result) -> float:
    recovered = recover_integration_point_stress(result)
    point_values = sum(
        abs(component)
        for field in recovered.section_points
        for row in field.rows
        for component in row.values().values()
    )
    return point_values


def _export_switch_checksum(export) -> float:
    return float(
        export.materialization_generation
        + len(export.field.locations)
        + len(export.field.descriptor.columns)
    )


def test_four_point_data_and_operation_budgets_are_bounded(
    request: pytest.FixtureRequest,
) -> None:
    result = static_linear.solve(
        _inline_cantilever("rectangle", {"height": 0.4, "width": 0.2}),
        "Load",
    )
    source = ResultSourceKey(
        "performance-result",
        "performance-session",
        "performance-artifact",
        1,
        "Load",
        "performance-run",
    )
    base_provider = build_result_provider(source, result)
    output_requests = (OutputRequest("field", "element", ("S",)),)
    outcome = execute_output_requests(base_provider, output_requests)
    provider = outcome.provider_draft
    stress_fields = _stress_fields(provider)

    value_bytes = sum(field.values.nbytes for field in stress_fields)
    section_resultant_bytes = 6 * np.dtype(float).itemsize
    byte_ratio = value_bytes / section_resultant_bytes

    section_resultant_seconds = _best_batch_time(
        lambda: _section_resultant_workload(result),
        repeats=30,
    )
    four_point_materialization_seconds = _best_batch_time(
        lambda: _four_point_workload(result),
        repeats=30,
    )

    point_selections = tuple(
        ScalarFieldSelection(field.key, "S11")
        for field in stress_fields
        if field.key.request.field_id.position is FieldPosition.INTEGRATION_POINT
    )
    base_switch_seconds = _best_batch_time(
        lambda: _export_switch_checksum(
            prepare_result_export_snapshot(
                provider.snapshot,
                point_selections[0],
            )
        ),
        repeats=1200,
    )
    point_index = 0

    def switch_point() -> float:
        nonlocal point_index
        export = prepare_result_export_snapshot(
            provider.snapshot,
            point_selections[point_index % len(point_selections)],
        )
        point_index += 1
        return _export_switch_checksum(export)

    four_point_switch_seconds = _best_batch_time(
        switch_point,
        repeats=1200,
    )

    metrics = {
        "section_resultant_value_bytes": section_resultant_bytes,
        "four_point_section_value_bytes": value_bytes,
        "four_point_value_byte_ratio": byte_ratio,
        "section_resultant_materialization_seconds_30": (
            section_resultant_seconds
        ),
        "four_point_materialization_seconds_30": (four_point_materialization_seconds),
        "four_point_materialization_time_ratio": (
            four_point_materialization_seconds / section_resultant_seconds
        ),
        "base_switch_seconds_1200": base_switch_seconds,
        "four_point_switch_seconds_1200": four_point_switch_seconds,
    }
    request.node.user_properties.extend(metrics.items())

    assert value_bytes == 4 * len(_POINT_COLUMNS) * np.dtype(float).itemsize
    assert byte_ratio <= 10.0
    assert four_point_materialization_seconds <= max(
        12.0 * section_resultant_seconds,
        0.5,
    )
    assert four_point_switch_seconds <= max(
        10.0 * base_switch_seconds,
        0.25,
    )
    assert four_point_materialization_seconds < 2.0
    assert four_point_switch_seconds < 1.0


def test_phase4_sources_have_no_external_fixture_path_dependency() -> None:
    source = Path(__file__).read_text(encoding="utf-8").casefold()
    forbidden = ("da" + "ta/", "da" + "ta" + chr(92))

    assert all(token not in source for token in forbidden)
