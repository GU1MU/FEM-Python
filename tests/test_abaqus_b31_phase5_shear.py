from __future__ import annotations

import csv
from io import StringIO
import math
from pathlib import Path

import numpy as np
import pytest

from fem.application.results import (
    FieldPosition,
    ResultQuery,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
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
from fem.elements.beam_section import (
    BeamIntegrationPointForces,
    parse_beam2_section,
    recover_integration_point_s11,
)
from fem.io.result_csv import dumps_result_csv
from fem.io.result_vtk import read_result_vtk, write_result_vtk
from fem.solvers import static_linear


_SECTION_ORACLE = (
    (
        "RECT",
        {"section_type": "rectangle", "width": 0.2, "height": 0.4},
        44200.3671875,
        ((0.1, 0.2), (-0.1, 0.2), (-0.1, -0.2), (0.1, -0.2)),
        (25, 21, 1, 5),
    ),
    (
        "CIRC",
        {"section_type": "solid_circle", "radius": 0.2},
        39788.734375,
        ((0.2, 0.0), (0.0, 0.2), (-0.2, 0.0), (0.0, -0.2)),
        (7, 11, 15, 3),
    ),
    (
        "THICK PIPE",
        {
            "section_type": "hollow_circle",
            "outer_radius": 0.2,
            "inner_radius": 0.1,
        },
        31830.98828125,
        ((0.15, 0.0), (0.0, 0.15), (-0.15, 0.0), (0.0, -0.15)),
        (8, 14, 20, 2),
    ),
)
_ABAQUS_POINT_MAPPING = {
    "RECT": (25, 21, 1, 5),
    "CIRC": (7, 11, 15, 3),
    "THICK PIPE": (8, 14, 20, 2),
}


@pytest.mark.parametrize(
    ("profile", "properties", "expected_s12", "coordinates", "abaqus_points"),
    _SECTION_ORACLE,
)
def test_positive_and_negative_torsion_match_abaqus_2023_section_points(
    profile: str,
    properties: dict[str, float | str],
    expected_s12: float,
    coordinates: tuple[tuple[float, float], ...],
    abaqus_points: tuple[int, ...],
) -> None:
    section = parse_beam2_section(properties)

    for sign in (1.0, -1.0):
        stresses = recover_integration_point_s11(
            section,
            BeamIntegrationPointForces(0.0, 0.0, 0.0, sign * 500.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(
            tuple(row.point.local_coordinates for row in stresses),
            coordinates,
        )
        assert [row.s12 for row in stresses] == pytest.approx(
            [sign * expected_s12] * 4,
            rel=2.0e-2,
        )
        assert all(row.s22 == 0.0 for row in stresses)
        assert all("S13" not in row.values() for row in stresses)

    assert _ABAQUS_POINT_MAPPING[profile] == abaqus_points


@pytest.mark.parametrize(
    "properties",
    tuple(case[1] for case in _SECTION_ORACLE),
)
@pytest.mark.parametrize(("shear_y", "shear_z"), ((1000.0, 0.0), (0.0, 1000.0)))
def test_transverse_section_forces_follow_abaqus_missing_point_stress_semantics(
    properties: dict[str, float | str],
    shear_y: float,
    shear_z: float,
) -> None:
    stresses = recover_integration_point_s11(
        parse_beam2_section(properties),
        BeamIntegrationPointForces(0.0, 0.0, 0.0, 0.0, shear_y, shear_z),
    )

    assert all(row.s12 == 0.0 for row in stresses)
    assert all("S13" not in row.values() for row in stresses)
    assert all(row.mises == 0.0 for row in stresses)


def test_invariants_are_recomputed_from_each_same_point_tensor() -> None:
    section = parse_beam2_section(
        {"section_type": "rectangle", "width": 0.2, "height": 0.4}
    )
    stresses = recover_integration_point_s11(
        section,
        BeamIntegrationPointForces(700.0, 120.0, -80.0, 500.0, 1000.0, -1500.0),
    )
    maximum_output = max(
        abs(value)
        for row in stresses
        for value in row.values().values()
    )
    maximum_difference = 0.0

    for row in stresses:
        values = row.values()
        expected_mises = math.sqrt(
            values["S11"] ** 2
            - values["S11"] * values["S22"]
            + values["S22"] ** 2
            + 3.0 * values["S12"] ** 2
        )
        span = math.sqrt(
            (values["S11"] - values["S22"]) ** 2
            + 4.0 * values["S12"] ** 2
        )
        plane = (
            (values["S11"] + values["S22"] + span) / 2.0,
            (values["S11"] + values["S22"] - span) / 2.0,
        )
        expected_principals = sorted((*plane, 0.0), reverse=True)
        differences = (
            abs(values["Mises"] - expected_mises),
            abs(values["MaxPrincipal"] - expected_principals[0]),
            abs(values["MidPrincipal"] - expected_principals[1]),
            abs(values["MinPrincipal"] - expected_principals[2]),
        )
        maximum_difference = max(maximum_difference, *differences)

    assert maximum_difference <= maximum_output * 1.0e-12


@pytest.mark.parametrize(
    ("axial", "moment_y", "moment_z"),
    ((1200.0, 0.0, 0.0), (0.0, 125.0, 0.0), (0.0, 0.0, -80.0)),
)
def test_no_shear_cases_preserve_exact_uniaxial_invariants(
    axial: float,
    moment_y: float,
    moment_z: float,
) -> None:
    stresses = recover_integration_point_s11(
        parse_beam2_section(
            {"section_type": "rectangle", "width": 0.2, "height": 0.4}
        ),
        BeamIntegrationPointForces(
            axial,
            moment_y,
            moment_z,
            0.0,
            0.0,
            0.0,
        ),
    )

    for row in stresses:
        assert row.mises == pytest.approx(abs(row.s11), rel=1.0e-8, abs=1.0e-8)
        assert row.max_principal == pytest.approx(max(row.s11, 0.0), abs=1.0e-8)
        assert row.min_principal == pytest.approx(min(row.s11, 0.0), abs=1.0e-8)


def test_default_rect_5x5_corner_owner_matches_frozen_aspect_ratio_matrix() -> None:
    oracle = (
        (1.0, 94141.4375),
        (1.25, 69463.78125),
        (1.5, 56651.1875),
        (2.0, 44200.3671875),
        (3.0, 34847.37890625),
        (4.0, 30420.5),
        (6.0, 24489.22265625),
        (10.0, 16962.0546875),
        (20.0, 9130.43359375),
    )

    for aspect_ratio, expected in oracle:
        stresses = recover_integration_point_s11(
            parse_beam2_section(
                {
                    "section_type": "rectangle",
                    "width": 0.2,
                    "height": 0.2 * aspect_ratio,
                }
            ),
            BeamIntegrationPointForces(0.0, 0.0, 0.0, 500.0, 0.0, 0.0),
        )
        assert [row.s12 for row in stresses] == pytest.approx(
            [expected] * 4,
            rel=2.0e-2,
        )


def test_query_csv_and_vtk_reuse_the_stored_s12_field(tmp_path: Path) -> None:
    mesh = Mesh3D(
        nodes=(Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 2.0, 0.0, 0.0)),
        elements=(
            Element3D(
                10,
                (1, 2),
                "Beam2",
                {
                    "E": 210.0e9,
                    "nu": 0.3,
                    "section_type": "rectangle",
                    "width": 0.2,
                    "height": 0.4,
                },
            ),
        ),
        dofs_per_node=6,
    )
    model = FEMModel(
        mesh=mesh,
        steps=(
            AnalysisStep(
                "Load",
                boundaries=(DisplacementConstraint(1, 1, 6),),
                cloads=(NodalLoad(2, 4, 500.0),),
            ),
        ),
    )
    result = static_linear.solve(model, "Load")
    provider = build_result_provider(
        ResultSourceKey("phase5", "session", "artifact", 1, "Load", "run"),
        result,
    )
    provider = execute_output_requests(
        provider,
        (OutputRequest("field", "element", ("S",)),),
    ).provider_draft
    field = next(
        value
        for value in provider.snapshot.fields
        if value.key.request.field_id.variable is ResultVariable.S
        and value.key.request.field_id.position is FieldPosition.INTEGRATION_POINT
        and value.key.request.field_id.section_point_number == 1
    )
    assert field.descriptor.components == ("S11", "S22", "S12")
    assert "S13" not in field.descriptor.columns
    stored = float(field.values[0, field.descriptor.columns.index("S12")])
    queried = provider.query(ResultQuery(field.key, "S12", element_ids=(10,)))
    assert queried.records[0].value == stored

    export = prepare_result_export_snapshot(
        provider.snapshot,
        ScalarFieldSelection(field.key, "S12"),
    )
    csv_row = next(csv.DictReader(StringIO(dumps_result_csv(export, queried))))
    assert float(csv_row["value"]) == stored

    vtk_path = tmp_path / "phase5-s12.vtk"
    write_result_vtk(vtk_path, export)
    vtk = read_result_vtk(vtk_path)
    assert vtk.values == pytest.approx((stored,))


def test_phase5_test_source_has_no_external_fixture_path_dependency() -> None:
    source = Path(__file__).read_text(encoding="utf-8").casefold()
    forbidden = ("da" + "ta/", "da" + "ta" + chr(92))

    assert all(token not in source for token in forbidden)
