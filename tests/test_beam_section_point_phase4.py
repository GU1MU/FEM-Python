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
from fem.elements.beam_section import (
    axial_stress_extrema,
    parse_beam2_section,
)
from fem.io import load_result_archive, save_result_archive
from fem.io.result_csv import dumps_result_csv
from fem.post.stress.beam import recover_section_stress
from fem.solvers import static_linear
from fem_gui.result_presentation import result_provider_section_point_labels
from fem_gui.widgets.result_tree import ResultTree


_LENGTH = 2.0
_AXIAL_FORCE = 12.0
_LOCAL_Y_FORCE = 3.0
_LOCAL_Z_FORCE = -2.0
_TORQUE = 1.5
_POINT_COLUMNS = (
    "S11",
    "S12",
    "Mises",
    "MaxPrincipal",
    "MidPrincipal",
    "MinPrincipal",
)
_SECTION_COLUMNS = (
    "S11Max",
    "S11Min",
    "S11AbsMax",
    "S12AbsMax",
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
        (OutputRequest("field", "element", ("S",)),),
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


def _independent_root_oracle(
    section_type: str,
    dimensions: dict[str, float],
) -> tuple[dict[int, tuple[float, ...]], tuple[float, ...]]:
    if section_type == "rectangle":
        height = dimensions["height"]
        width = dimensions["width"]
        area = height * width
        iyy = width * height**3 / 12.0
        izz = height * width**3 / 12.0
        torsion_constant = _rectangle_torsion_constant(height, width)
        points = (
            (width / 2.0, height / 2.0),
            (-width / 2.0, height / 2.0),
            (-width / 2.0, -height / 2.0),
            (width / 2.0, -height / 2.0),
        )
        point_shear = 0.0
        section_shear = _rectangle_torsion_shear(
            height,
            width,
            torsion_constant,
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
        torsion_constant = 2.0 * iyy
        points = (
            (outer_radius, 0.0),
            (0.0, outer_radius),
            (-outer_radius, 0.0),
            (0.0, -outer_radius),
        )
        point_shear = _TORQUE * outer_radius / torsion_constant
        section_shear = abs(point_shear)

    axial = _AXIAL_FORCE / area
    moment_y = -_LOCAL_Z_FORCE * _LENGTH
    moment_z = _LOCAL_Y_FORCE * _LENGTH
    point_values: dict[int, tuple[float, ...]] = {}
    s11_values = []
    for number, (local_y, local_z) in enumerate(points, start=1):
        s11 = axial - moment_y * local_z / iyy + moment_z * local_y / izz
        principal_span = math.sqrt(s11**2 + 4.0 * point_shear**2)
        values = (
            s11,
            point_shear,
            math.sqrt(s11**2 + 3.0 * point_shear**2),
            (s11 + principal_span) / 2.0,
            0.0,
            (s11 - principal_span) / 2.0,
        )
        point_values[number] = values
        s11_values.append(s11)

    if section_type == "rectangle":
        maximum = max(s11_values)
        minimum = min(s11_values)
    else:
        outer_radius = points[0][0]
        increment = outer_radius * math.hypot(
            moment_y / iyy,
            moment_z / izz,
        )
        maximum = axial + increment
        minimum = axial - increment
    section_values = (
        maximum,
        minimum,
        max(abs(maximum), abs(minimum)),
        section_shear,
    )
    return point_values, section_values


def _stress_fields(provider):
    return tuple(
        field
        for field in provider.snapshot.fields
        if field.key.request.field_id.variable is ResultVariable.S
        and field.key.request.field_id.position
        in (FieldPosition.SECTION_POINT, FieldPosition.SECTION_END)
    )


def _root_row(field) -> int:
    return next(
        index
        for index, location in enumerate(field.locations)
        if location.element_id == 10 and location.local_node == 1
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
    point_oracle, section_oracle = _independent_root_oracle(
        section_type,
        dimensions,
    )
    fields = _stress_fields(provider)

    assert np.all(np.isfinite(result.U))
    assert np.all(np.isfinite(result.reactions))
    assert len(fields) == 5
    assert all(np.all(np.isfinite(field.values)) for field in fields)

    point_fields = tuple(
        field
        for field in fields
        if field.key.request.field_id.position is FieldPosition.SECTION_POINT
    )
    section_field = next(
        field
        for field in fields
        if field.key.request.field_id.position is FieldPosition.SECTION_END
    )
    assert tuple(
        field.key.request.field_id.section_point_number for field in point_fields
    ) == (1, 2, 3, 4)
    assert all(field.descriptor.columns == _POINT_COLUMNS for field in point_fields)
    assert section_field.descriptor.columns == _SECTION_COLUMNS

    for field in point_fields:
        point_number = field.key.request.field_id.section_point_number
        assert point_number is not None
        row = _root_row(field)
        np.testing.assert_allclose(
            field.values[row],
            point_oracle[point_number],
            rtol=2.0e-10,
            atol=1.0e-8,
        )
    np.testing.assert_allclose(
        section_field.values[_root_row(section_field)],
        section_oracle,
        rtol=2.0e-10,
        atol=1.0e-8,
    )

    selected_field = point_fields[2]
    query = ResultQuery(
        selected_field.key,
        "S11",
        node_ids=(1,),
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
    assert csv_rows[0]["field_position"] == "section_point"
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
        if field.key.request.field_id.position is FieldPosition.SECTION_POINT
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
    assert gui_stress_labels == (*expected_point_labels, "截面")
    tree.close()


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


def _legacy_three_component_workload(result) -> float:
    mesh = result.model.mesh
    element = mesh.elements[0]
    kernel = get_element_kernel("Beam2")
    end_actions = kernel.local_end_actions(mesh, element, result.U)
    section = parse_beam2_section(element.props)
    values = tuple(
        axial_stress_extrema(
            section,
            float(axial_force),
            float(moment_y),
            float(moment_z),
        )
        for axial_force, moment_y, moment_z in end_actions
    )
    return sum(abs(component) for row in values for component in row)


def _four_point_workload(result) -> float:
    recovered = recover_section_stress(result)
    section_values = sum(
        abs(component)
        for row in recovered.section_end.rows
        for component in row.section_values().values()
    )
    point_values = sum(
        abs(component)
        for field in recovered.section_points
        for row in field.rows
        for component in row.values().values()
    )
    return section_values + point_values


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
    section_rows = next(
        field.values.shape[0]
        for field in stress_fields
        if field.key.request.field_id.position is FieldPosition.SECTION_END
    )
    legacy_three_component_bytes = section_rows * 3 * np.dtype(float).itemsize
    byte_ratio = value_bytes / legacy_three_component_bytes

    legacy_three_component_seconds = _best_batch_time(
        lambda: _legacy_three_component_workload(result),
        repeats=30,
    )
    four_point_materialization_seconds = _best_batch_time(
        lambda: _four_point_workload(result),
        repeats=30,
    )

    point_selections = tuple(
        ScalarFieldSelection(field.key, "S11")
        for field in stress_fields
        if field.key.request.field_id.position is FieldPosition.SECTION_POINT
    )
    section_selection = ScalarFieldSelection(
        next(
            field.key
            for field in stress_fields
            if field.key.request.field_id.position is FieldPosition.SECTION_END
        ),
        "S11AbsMax",
    )
    legacy_switch_seconds = _best_batch_time(
        lambda: _export_switch_checksum(
            prepare_result_export_snapshot(
                provider.snapshot,
                section_selection,
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
        "legacy_section_value_bytes": legacy_three_component_bytes,
        "four_point_section_value_bytes": value_bytes,
        "four_point_value_byte_ratio": byte_ratio,
        "legacy_three_component_materialization_seconds_30": (
            legacy_three_component_seconds
        ),
        "four_point_materialization_seconds_30": (four_point_materialization_seconds),
        "four_point_materialization_time_ratio": (
            four_point_materialization_seconds / legacy_three_component_seconds
        ),
        "legacy_switch_seconds_1200": legacy_switch_seconds,
        "four_point_switch_seconds_1200": four_point_switch_seconds,
    }
    request.node.user_properties.extend(metrics.items())

    assert value_bytes == 56 * np.dtype(float).itemsize
    assert byte_ratio <= 10.0
    assert four_point_materialization_seconds <= max(
        12.0 * legacy_three_component_seconds,
        0.5,
    )
    assert four_point_switch_seconds <= max(
        10.0 * legacy_switch_seconds,
        0.25,
    )
    assert four_point_materialization_seconds < 2.0
    assert four_point_switch_seconds < 1.0


def test_phase4_sources_have_no_external_fixture_path_dependency() -> None:
    source = Path(__file__).read_text(encoding="utf-8").casefold()
    forbidden = ("da" + "ta/", "da" + "ta" + chr(92))

    assert all(token not in source for token in forbidden)
