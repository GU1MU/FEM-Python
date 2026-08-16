from __future__ import annotations

import numpy as np
import pytest

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import AnalysisStep, FEMModel
from fem.core.result import ModelResult
from fem.elements import get_element_kernel
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


@pytest.mark.parametrize(
    ("translation_component", "result_name", "zero_result_name"),
    ((1, "Vy", "Vz"), (2, "Vz", "Vy")),
)
def test_internal_end_actions_publish_both_transverse_shears(
    translation_component: int,
    result_name: str,
    zero_result_name: str,
) -> None:
    mesh = _beam_mesh()
    element = mesh.elements[0]
    displacement = np.zeros(mesh.num_dofs)
    displacement[mesh.global_dof(2, translation_component)] = 0.01
    kernel = get_element_kernel("Beam2")

    forces = kernel.local_section_end_actions(mesh, element, displacement)
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

    start, end = kernel.local_section_end_actions(
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


def test_integration_point_recovery_keeps_shear_resultant_and_point_stress_semantics() -> None:
    mesh = _beam_mesh()
    displacement = np.zeros(mesh.num_dofs)
    displacement[mesh.global_dof(2, 1)] = 0.01
    result = ModelResult(
        FEMModel(mesh=mesh),
        AnalysisStep("Load"),
        displacement,
        np.zeros(mesh.num_dofs),
    )

    recovered = beam.recover_integration_point_stress(result)

    assert recovered.section_forces.component_names == (
        "N",
        "Vy",
        "Vz",
        "T",
        "My",
        "Mz",
    )
    force = recovered.section_forces.rows[0]
    assert force.Vy != 0.0
    assert force.Vz == pytest.approx(0.0)
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
    assert all(
        row.s12 == pytest.approx(0.0)
        for field in recovered.section_points
        for row in field.rows
    )
