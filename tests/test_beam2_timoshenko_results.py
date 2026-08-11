from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import AnalysisStep, FEMModel
from fem.core.result import ModelResult
from fem.elements import BeamSectionEndForces, get_element_kernel
from fem.elements.beam_section import (
    parse_beam2_section,
    recover_section_point_stress,
)
from fem.post.stress import beam


def _beam_mesh() -> Mesh3D:
    return Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 2.0, 0.0, 0.0)],
        elements=[
            Element3D(
                1,
                [1, 2],
                "Beam2",
                {
                    "E": 210.0e9,
                    "nu": 0.3,
                    "section_type": "rectangle",
                    "height": 0.4,
                    "width": 0.2,
                },
            )
        ],
        dofs_per_node=6,
    )


def test_end_force_contract_preserves_legacy_four_position_constructor() -> None:
    legacy = BeamSectionEndForces(1.0, 4.0, 5.0, 6.0)
    complete = BeamSectionEndForces(
        axial_force=1.0,
        moment_y=4.0,
        moment_z=5.0,
        torque=6.0,
        shear_y=2.0,
        shear_z=3.0,
    )

    assert (legacy.N, legacy.Vy, legacy.Vz, legacy.My, legacy.Mz, legacy.T) == (
        1.0,
        0.0,
        0.0,
        4.0,
        5.0,
        6.0,
    )
    assert (
        complete.N,
        complete.Vy,
        complete.Vz,
        complete.My,
        complete.Mz,
        complete.T,
    ) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    with pytest.raises(FrozenInstanceError):
        complete.shear_y = 0.0
    with pytest.raises(ValueError, match="shear_y"):
        BeamSectionEndForces(1.0, 4.0, 5.0, 6.0, float("nan"), 3.0)


@pytest.mark.parametrize(
    "section_props",
    (
        {"section_type": "rectangle", "height": 0.4, "width": 0.2},
        {"section_type": "solid_circle", "radius": 0.2},
        {
            "section_type": "hollow_circle",
            "outer_radius": 0.2,
            "inner_radius": 0.1,
        },
    ),
)
def test_section_point_stress_follows_abaqus_missing_transverse_shear_semantics(
    section_props: dict[str, float | str],
) -> None:
    section = parse_beam2_section(section_props)
    without_shear = recover_section_point_stress(
        section,
        BeamSectionEndForces(11.0, 12.0, 13.0, 14.0),
    )
    with_shear = recover_section_point_stress(
        section,
        BeamSectionEndForces(11.0, 12.0, 13.0, 14.0, 101.0, -202.0),
    )

    assert with_shear.section_values() == pytest.approx(
        without_shear.section_values()
    )
    for actual, expected in zip(
        with_shear.point_stresses,
        without_shear.point_stresses,
        strict=True,
    ):
        assert actual.values() == pytest.approx(expected.values())


@pytest.mark.parametrize(
    ("translation_component", "result_name", "zero_result_name"),
    ((1, "Vy", "Vz"), (2, "Vz", "Vy")),
)
def test_kernel_publishes_both_transverse_end_shears_without_changing_legacy_view(
    translation_component: int,
    result_name: str,
    zero_result_name: str,
) -> None:
    mesh = _beam_mesh()
    element = mesh.elements[0]
    displacement = np.zeros(mesh.num_dofs)
    displacement[mesh.global_dof(2, translation_component)] = 0.01
    kernel = get_element_kernel("Beam2")

    forces = kernel.local_section_end_forces(mesh, element, displacement)
    legacy = kernel.local_end_actions(mesh, element, displacement)

    assert legacy.shape == (2, 3)
    assert legacy == pytest.approx(
        np.asarray([(row.N, row.My, row.Mz) for row in forces])
    )
    assert [getattr(row, result_name) for row in forces] == pytest.approx(
        [getattr(forces[0], result_name)] * 2
    )
    assert abs(getattr(forces[0], result_name)) > 0.0
    assert [getattr(row, zero_result_name) for row in forces] == pytest.approx(
        [0.0, 0.0],
        abs=1.0e-12,
    )


def test_end_resultants_reconstruct_balanced_local_nodal_actions() -> None:
    mesh = _beam_mesh()
    element = mesh.elements[0]
    kernel = get_element_kernel("Beam2")
    local_load = kernel.local_line_load(
        mesh,
        element,
        (3.0, 5.0, -7.0),
        "local",
    )

    start, end = kernel.local_section_end_forces(
        mesh,
        element,
        np.zeros(mesh.num_dofs),
        local_load,
    )
    reconstructed_action = np.asarray(
        (
            -start.N,
            -start.Vy,
            -start.Vz,
            -start.T,
            -start.My,
            -start.Mz,
            end.N,
            end.Vy,
            end.Vz,
            end.T,
            end.My,
            end.Mz,
        )
    )

    assert reconstructed_action == pytest.approx(-local_load)
    assert reconstructed_action + local_load == pytest.approx(np.zeros(12))


def test_stress_recovery_retains_shear_source_actions_without_new_stress_fields() -> None:
    mesh = _beam_mesh()
    displacement = np.zeros(mesh.num_dofs)
    displacement[mesh.global_dof(2, 1)] = 0.01
    result = ModelResult(
        FEMModel(mesh=mesh),
        AnalysisStep("Load"),
        displacement,
        np.zeros(mesh.num_dofs),
    )

    recovered = beam.recover_section_stress(result)

    assert recovered.section_end.component_names == (
        "S11Max",
        "S11Min",
        "S11AbsMax",
    )
    assert all(row.shear_y != 0.0 for row in recovered.section_end.rows)
    assert all(row.shear_z == pytest.approx(0.0) for row in recovered.section_end.rows)
    assert [row.Vy for row in recovered.section_end.rows] == pytest.approx(
        [row.shear_y for row in recovered.section_end.rows]
    )
    assert [row.Vz for row in recovered.section_end.rows] == pytest.approx(
        [row.shear_z for row in recovered.section_end.rows]
    )
    assert [
        (row.N, row.My, row.Mz, row.T) for row in recovered.section_end.rows
    ] == pytest.approx(
        [
            (row.axial_force, row.moment_y, row.moment_z, row.torque)
            for row in recovered.section_end.rows
        ]
    )
    assert all(
        tuple(row.values()) == ("S11Max", "S11Min", "S11AbsMax")
        for row in recovered.section_end.rows
    )
    assert all(
        field.component_names
        == (
            "S11",
            "S22",
            "S12",
            "Mises",
            "MaxPrincipal",
            "MidPrincipal",
            "MinPrincipal",
        )
        for field in recovered.section_points
    )
