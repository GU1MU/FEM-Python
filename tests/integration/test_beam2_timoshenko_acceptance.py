from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from fem.application.results import (
    FieldPosition,
    ResultArchiveModelProjection,
    ResultArchiveOrigin,
    ResultArchiveRun,
    ResultArchiveSnapshot,
    ResultSourceKey,
    ResultVariable,
    build_result_provider,
    execute_output_requests,
)
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    FEMModel,
    NodalLoad,
)
from fem.elements import get_element_kernel
from fem.elements.beam_section import parse_beam2_section
from fem.io import inp, load_result_archive, save_result_archive
from fem.solvers import static_linear


_E = 210.0e9
_NU = 0.3
_G = _E / (2.0 * (1.0 + _NU))
_SECTION_CASES = (
    ("rectangle", {"height": 0.4, "width": 0.2}),
    ("solid_circle", {"radius": 0.2}),
    (
        "hollow_circle",
        {"outer_radius": 0.2, "inner_radius": 0.1},
    ),
)


def _cantilever(
    section_type: str,
    dimensions: dict[str, float],
    *,
    length: float,
    loads: tuple[float, float, float, float],
) -> FEMModel:
    axial, transverse_y, transverse_z, torque = loads
    mesh = Mesh3D(
        nodes=(
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, length, 0.0, 0.0),
        ),
        elements=(
            Element3D(
                10,
                (1, 2),
                "Beam2",
                {
                    "E": _E,
                    "nu": _NU,
                    "section_type": section_type,
                    **dimensions,
                },
            ),
        ),
        dofs_per_node=6,
    )
    return FEMModel(
        mesh=mesh,
        name=f"inline-{section_type}-cantilever",
        steps=[
            AnalysisStep(
                "Combined",
                boundaries=(DisplacementConstraint(1, 1, 6),),
                cloads=tuple(
                    NodalLoad(2, component, value)
                    for component, value in enumerate(loads, start=1)
                    if value != 0.0
                ),
            )
        ],
    )


def _bending_compliance(
    section_type: str,
    dimensions: dict[str, float],
    *,
    length: float,
    component: int,
) -> tuple[float, float]:
    section = parse_beam2_section(
        {"section_type": section_type, **dimensions}
    )
    shear_y, shear_z = section.abaqus_b31_shear_rigidities(
        _G,
        _NU,
        length,
    )
    if component == 2:
        inertia = section.Izz
        shear_rigidity = shear_y
    else:
        inertia = section.Iyy
        shear_rigidity = shear_z
    euler_bernoulli = length**3 / (3.0 * _E * inertia)
    b31 = length**3 / (4.0 * _E * inertia) + length / shear_rigidity
    return euler_bernoulli, b31


@pytest.mark.parametrize(("section_type", "dimensions"), _SECTION_CASES)
@pytest.mark.parametrize("component", (2, 3), ids=("local-y", "local-z"))
@pytest.mark.parametrize(
    ("length", "regime"),
    ((0.5, "thick"), (12.0, "slender")),
)
def test_three_sections_match_abaqus_b31_bending_limits(
    section_type: str,
    dimensions: dict[str, float],
    component: int,
    length: float,
    regime: str,
) -> None:
    load = 1.7e3
    loads = [0.0, 0.0, 0.0, 0.0]
    loads[component - 1] = load
    result = static_linear.solve(
        _cantilever(
            section_type,
            dimensions,
            length=length,
            loads=tuple(loads),
        ),
        "Combined",
    )
    euler_bernoulli, b31 = _bending_compliance(
        section_type,
        dimensions,
        length=length,
        component=component,
    )
    displacement = result.nodal_displacement(2, component)

    assert displacement == pytest.approx(load * b31, rel=2.0e-10)
    if regime == "thick":
        assert b31 > 1.05 * euler_bernoulli
    else:
        assert b31 < euler_bernoulli


@pytest.mark.parametrize(("section_type", "dimensions"), _SECTION_CASES)
def test_combined_load_balances_displacement_reaction_energy_and_end_forces(
    section_type: str,
    dimensions: dict[str, float],
) -> None:
    length = 1.2
    axial = 15.0e3
    transverse_y = -6.0e3
    transverse_z = 4.0e3
    torque = 2.0e3
    model = _cantilever(
        section_type,
        dimensions,
        length=length,
        loads=(axial, transverse_y, transverse_z, torque),
    )
    result = static_linear.solve(model, "Combined")
    section = parse_beam2_section(result.model.mesh.elements[0].props)
    shear_y, shear_z = section.abaqus_b31_shear_rigidities(
        _G,
        _NU,
        length,
    )

    expected_tip = (
        axial * length / (_E * section.area),
        transverse_y
        * (length**3 / (4.0 * _E * section.Izz) + length / shear_y),
        transverse_z
        * (length**3 / (4.0 * _E * section.Iyy) + length / shear_z),
        torque * length / (_G * section.J),
    )
    assert tuple(
        result.nodal_displacement(2, index) for index in range(1, 5)
    ) == pytest.approx(expected_tip)

    expected_root_reactions = (
        -axial,
        -transverse_y,
        -transverse_z,
        -torque,
        length * transverse_z,
        -length * transverse_y,
    )
    assert tuple(
        result.nodal_reaction(1, index) for index in range(1, 7)
    ) == pytest.approx(expected_root_reactions, abs=1.0e-8)
    assert result.reactions[result.model.mesh.node_dofs(2)] == pytest.approx(
        np.zeros(6),
        abs=1.0e-8,
    )

    kernel = get_element_kernel("Beam2")
    stiffness = kernel.stiffness(
        result.model.mesh,
        result.model.mesh.elements[0],
    )
    element_displacement = result.U[
        list(result.model.mesh.element_dofs(result.model.mesh.elements[0]))
    ]
    strain_energy = 0.5 * float(
        element_displacement @ stiffness @ element_displacement
    )
    applied = np.zeros(result.model.mesh.num_dofs)
    for component, value in enumerate(
        (axial, transverse_y, transverse_z, torque)
    ):
        applied[result.model.mesh.global_dof(2, component)] = value
    external_work = 0.5 * float(applied @ result.U)
    assert strain_energy == pytest.approx(external_work, rel=2.0e-10)

    start, end = kernel.local_section_end_forces(
        result.model.mesh,
        result.model.mesh.elements[0],
        result.U,
    )
    assert (start.N, end.N) == pytest.approx((axial, axial))
    assert (start.Vy, end.Vy) == pytest.approx(
        (transverse_y, transverse_y),
        abs=1.0e-8,
    )
    assert (start.Vz, end.Vz) == pytest.approx(
        (transverse_z, transverse_z),
        abs=1.0e-8,
    )
    assert (start.My, end.My) == pytest.approx(
        (-length * transverse_z, 0.0),
        abs=1.0e-8,
    )
    assert (start.Mz, end.Mz) == pytest.approx(
        (length * transverse_y, 0.0),
        abs=1.0e-8,
    )
    assert (start.T, end.T) == pytest.approx((torque, torque))


def _write_inline_b31(tmp_path: Path) -> Path:
    path = tmp_path / "inline_b31_acceptance.inp"
    path.write_text(
        "\n".join(
            (
                "*Heading",
                "*Node",
                "1, 0.0, 0.0, 0.0",
                "2, 1.2, 0.0, 0.0",
                "*Element, type=B31, elset=BEAM",
                "10, 1, 2",
                "*Nset, nset=FIXED",
                "1",
                "*Material, name=STEEL",
                "*Elastic",
                "2.10E11, 0.30",
                "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
                "0.4, 0.2",
                "0.0, 1.0, 0.0",
                "*Step, name=Combined",
                "*Static",
                "*Boundary",
                "FIXED, 1, 6, 0.0",
                "*Cload",
                "2, 1, 15000.0",
                "2, 2, -6000.0",
                "2, 3, 4000.0",
                "2, 4, 2000.0",
                "*Output, FIELD",
                "*Node Output",
                "U, RF",
                "*Element Output, elset=BEAM",
                "S",
                "*End Step",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="b31-acceptance-result",
        session_id="b31-acceptance-session",
        artifact_id="b31-acceptance-artifact",
        model_revision=1,
        step_name="Combined",
        run_id="b31-acceptance-run",
    )


def _archive_snapshot(provider, outcome) -> ResultArchiveSnapshot:
    created_at = datetime(2026, 8, 10, tzinfo=timezone.utc)
    return ResultArchiveSnapshot(
        archive_id="b31-acceptance-archive",
        created_at=created_at,
        producer_version="beam2-timoshenko-acceptance",
        origin=ResultArchiveOrigin(
            model_name="inline-b31-acceptance",
            model_fingerprint="0" * 64,
        ),
        run=ResultArchiveRun(
            "B31-Acceptance",
            "Combined",
            created_at,
            output_report=outcome.report,
        ),
        profile=provider.profile,
        catalog=provider.catalog(),
        materialization=provider.snapshot,
        model_projection=ResultArchiveModelProjection(
            provider.snapshot.topology,
        ),
    )


def test_inline_b31_solve_s_fields_and_femres_round_trip(tmp_path: Path) -> None:
    imported = inp.read_with_report(_write_inline_b31(tmp_path))
    assert tuple(notice.code for notice in imported.notices) == (
        "abaqus.b31.linear_timoshenko_support_boundary",
    )
    assert "Timoshenko" in imported.notices[0].message
    assert "Euler" not in imported.notices[0].message

    step = imported.model.steps[0]
    result = static_linear.solve(imported.model, step)
    provider = build_result_provider(_source(), result)
    outcome = execute_output_requests(provider, tuple(step.outputs))
    materialized = outcome.provider_draft
    stress_fields = tuple(
        field
        for field in materialized.snapshot.fields
        if field.key.request.field_id.variable is ResultVariable.S
        and field.key.request.field_id.position
        in (FieldPosition.SECTION_POINT, FieldPosition.SECTION_END)
    )

    assert len(stress_fields) == 5
    point_fields = tuple(
        field
        for field in stress_fields
        if field.key.request.field_id.position is FieldPosition.SECTION_POINT
    )
    section_field = next(
        field
        for field in stress_fields
        if field.key.request.field_id.position is FieldPosition.SECTION_END
    )
    assert tuple(
        field.key.request.field_id.section_point_number
        for field in point_fields
    ) == (1, 2, 3, 4)
    assert all(
        field.descriptor.components == ("S11", "S12")
        for field in point_fields
    )
    assert section_field.descriptor.components == ("S11Max", "S11Min")
    assert all(np.all(np.isfinite(field.values)) for field in stress_fields)

    archive_path = tmp_path / "inline_b31_acceptance.femres"
    save_result_archive(
        archive_path,
        _archive_snapshot(materialized, outcome),
    )
    loaded = load_result_archive(archive_path).snapshot
    loaded_stress = tuple(
        field
        for field in loaded.fields
        if field.key.request.field_id.variable is ResultVariable.S
        and field.key.request.field_id.position
        in (FieldPosition.SECTION_POINT, FieldPosition.SECTION_END)
    )
    assert tuple(field.key for field in loaded_stress) == tuple(
        field.key for field in stress_fields
    )
    for expected, actual in zip(stress_fields, loaded_stress, strict=True):
        assert actual.locations == expected.locations
        np.testing.assert_allclose(actual.values, expected.values)
