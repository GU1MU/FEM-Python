from __future__ import annotations

import numpy as np
import pytest

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    FEMModel,
    LineLoad,
    NodalLoad,
)
from fem.core.result import ModelResult
from fem.elements import get_element_kernel, resolve_beam_frame
from fem.elements.beam_section import parse_beam2_section
from fem.post.stress import beam
from fem.solvers import static_linear
from tests.helpers.model_builders import make_line_load_beam_model


def _beam_mesh(*, end=(4.0, 0.0, 0.0), props=None):
    properties = {
        "E": 210.0,
        "nu": 0.25,
        "section_type": "rectangle",
        "height": 3.0,
        "width": 2.0,
        "rho": 4.0,
    }
    if props:
        properties.update(props)
    return Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, *end)],
        elements=[Element3D(1, [1, 2], "Beam2", properties)],
        dofs_per_node=6,
    )


def _beam_result(mesh, U, step=None):
    model = FEMModel(mesh=mesh)
    selected_step = step or AnalysisStep("result")
    return ModelResult(
        model,
        selected_step,
        np.asarray(U, dtype=float),
        np.zeros(mesh.num_dofs),
    )


def test_beam2_rigid_motion_recovers_zero_nodal_axial_stress():
    mesh = _beam_mesh(end=(2.0, 3.0, 6.0))
    rigid = np.tile([0.4, -0.2, 0.7, 0.0, 0.0, 0.0], 2)

    rows = beam.nodal_envelope(_beam_result(mesh, rigid))

    assert [(row.maximum, row.minimum, row.absolute_maximum) for row in rows] == pytest.approx(
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], abs=1e-10
    )


def test_beam2_pure_tension_recovers_positive_stress_at_both_ends():
    mesh = _beam_mesh()
    U = np.zeros(mesh.num_dofs)
    U[mesh.global_dof(2, 0)] = 0.04
    result = _beam_result(mesh, U)

    rows = beam.nodal_envelope(result)

    expected = 210.0 * 0.04 / 4.0
    assert [(row.maximum, row.minimum, row.absolute_maximum) for row in rows] == pytest.approx(
        [(expected, expected, expected), (expected, expected, expected)]
    )
    assert beam.absolute_maximum(result) == pytest.approx(expected)


def test_beam2_inclined_and_reversed_elements_preserve_physical_extrema():
    inclined = _beam_mesh(end=(2.0, 3.0, 6.0))
    rotation = resolve_beam_frame(
        inclined,
        inclined.elements[0],
    ).rotation
    local_displacement = np.zeros(12)
    local_displacement[6] = 0.07
    local_displacement[7] = 0.02
    local_displacement[11] = 0.01
    global_displacement = np.zeros(12)
    for start in (0, 3, 6, 9):
        global_displacement[start : start + 3] = rotation.T @ local_displacement[start : start + 3]

    forward = beam.nodal_envelope(_beam_result(inclined, global_displacement))
    reversed_mesh = _beam_mesh(end=(2.0, 3.0, 6.0))
    reversed_mesh.elements[0].node_ids = [2, 1]
    reversed_rows = beam.nodal_envelope(
        _beam_result(reversed_mesh, global_displacement)
    )

    forward_by_node = {row.node_id: row for row in forward}
    reversed_by_node = {row.node_id: row for row in reversed_rows}
    for node_id in inclined.node_ids:
        assert (
            reversed_by_node[node_id].maximum,
            reversed_by_node[node_id].minimum,
            reversed_by_node[node_id].absolute_maximum,
        ) == pytest.approx(
            (
                forward_by_node[node_id].maximum,
                forward_by_node[node_id].minimum,
                forward_by_node[node_id].absolute_maximum,
            )
        )


def test_beam2_shared_node_uses_maximum_minimum_envelope_without_averaging():
    props = dict(_beam_mesh().elements[0].props)
    mesh = Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 1.0, 0.0, 0.0),
            Node3D(3, 2.0, 0.0, 0.0),
        ],
        elements=[
            Element3D(1, [1, 2], "Beam2", dict(props)),
            Element3D(2, [2, 3], "Beam2", dict(props)),
        ],
        dofs_per_node=6,
    )
    U = np.zeros(mesh.num_dofs)
    U[mesh.global_dof(2, 0)] = 0.1

    rows = beam.nodal_envelope(_beam_result(mesh, U))

    assert [row.node_id for row in rows] == [1, 2, 3]
    assert [(row.maximum, row.minimum, row.absolute_maximum) for row in rows] == pytest.approx(
        [(21.0, 21.0, 21.0), (21.0, -21.0, 21.0), (-21.0, -21.0, 21.0)]
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
    # Rectangle 3 by 2: Izz=2, Iyy=4.5; G=84, kappa=.85, B31 compensation=.25.
    inertia = 2.0 if translation_component == 1 else 4.5
    shear_rigidity = 0.85 * 84.0 * 6.0 / (1.0 + 0.25 * 4.0**2 * 6.0 / (12.0 * inertia))
    expected_shear = shear_rigidity * 0.01 / 4.0
    assert [getattr(row, result_name) for row in forces] == pytest.approx(
        [expected_shear, expected_shear]
    )
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
    assert force.Vy == pytest.approx(0.85 * 84.0 * 6.0 / 2.0 * 0.01 / 4.0)
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


def test_zero_strain_has_zero_constitutive_stress_with_local_or_global_line_loads():
    model = make_line_load_beam_model(
        inclined=True,
        orientation=(0.0, 1.0, 0.0),
    )
    mesh = model.mesh
    elem = mesh.elements[0]
    frame = resolve_beam_frame(mesh, elem)
    rotation = frame.rotation
    first_local = np.array([2.0, 3.0, 4.0])
    second_local = np.array([-1.0, 2.0, -2.0])
    combined_local = first_local + second_local
    multi_global_step = AnalysisStep(
        "multi_global",
        line_loads=(
            LineLoad(10, rotation.T @ first_local, "global"),
            LineLoad("beams", rotation.T @ second_local, "global"),
        ),
    )
    combined_local_step = AnalysisStep(
        "combined_local",
        line_loads=(LineLoad(10, combined_local, "local"),),
    )

    def recover(step):
        result = ModelResult(
            model,
            step,
            np.zeros(mesh.num_dofs),
            np.zeros(mesh.num_dofs),
        )
        return beam.nodal_envelope(result)

    multi_rows = recover(multi_global_step)
    local_rows = recover(combined_local_step)

    multi_values = [
        (row.maximum, row.minimum, row.absolute_maximum) for row in multi_rows
    ]
    local_values = [
        (row.maximum, row.minimum, row.absolute_maximum) for row in local_rows
    ]
    assert np.allclose(multi_values, local_values)

    assert np.allclose(multi_values, np.zeros((2, 3)))


def test_inclined_nodal_tip_load_recovers_axial_and_biaxial_bending_results():
    model = make_line_load_beam_model(
        inclined=True,
        orientation=(0.0, 1.0, 0.0),
    )
    mesh = model.mesh
    elem = mesh.elements[0]
    frame = resolve_beam_frame(mesh, elem)
    length = frame.length
    rotation = frame.rotation
    axial_force = 12.0
    force_y = 5.0
    force_z = -3.0
    local_tip_force = np.array([axial_force, force_y, force_z])
    global_tip_force = rotation.T @ local_tip_force
    step = AnalysisStep(
        "combined_tip_load",
        boundaries=(DisplacementConstraint(1, 1, 6, 0.0),),
        cloads=tuple(
            NodalLoad(2, component, value)
            for component, value in enumerate(global_tip_force, start=1)
        ),
    )
    model.steps.append(step)

    result = static_linear.solve(model, step)

    end_action_rows = get_element_kernel("Beam2").local_section_end_actions(
        mesh,
        elem,
        result.U,
    )
    moment_y = -force_z * length
    moment_z = force_y * length
    assert np.allclose(
        [(row.N, row.My, row.Mz) for row in end_action_rows],
        [
            (axial_force, moment_y, moment_z),
            (axial_force, 0.0, 0.0),
        ],
        atol=1e-10,
    )

    section = parse_beam2_section(elem.props)
    axial_stress = axial_force / section.area
    increment = (
        abs(0.5 * moment_y / section.Iyy) * section.height / 2.0
        + abs(0.5 * moment_z / section.Izz) * section.width / 2.0
    )
    rows = beam.nodal_envelope(result)
    recovered = [
        (row.maximum, row.minimum, row.absolute_maximum) for row in rows
    ]
    assert np.allclose(
        recovered,
        [
            (
                axial_stress + increment,
                axial_stress - increment,
                axial_stress + increment,
            ),
            (
                axial_stress + increment,
                axial_stress - increment,
                axial_stress + increment,
            ),
        ],
        atol=1e-10,
    )


def test_explicit_orientation_section_end_actions_follow_reversal_convention():
    model = make_line_load_beam_model(inclined=True, orientation=(0.0, 1.0, 0.0))
    reversed_model = make_line_load_beam_model(inclined=True, orientation=(0.0, 1.0, 0.0))
    reversed_model.mesh.elements[0].node_ids = [2, 1]
    kernel = get_element_kernel("Beam2")
    local_axis_reversal = np.diag([-1.0, 1.0, -1.0])
    action_component_reversal = np.array([1.0, -1.0, 1.0])
    local_vector = np.array([2.0, 3.0, 4.0])
    displacement = np.arange(model.mesh.num_dofs, dtype=float) / 10.0
    local_load = kernel.local_line_load(
        model.mesh,
        model.mesh.elements[0],
        local_vector,
        "local",
    )
    reversed_local_load = kernel.local_line_load(
        reversed_model.mesh,
        reversed_model.mesh.elements[0],
        local_axis_reversal @ local_vector,
        "local",
    )

    action_rows = kernel.local_section_end_actions(
        model.mesh,
        model.mesh.elements[0],
        displacement,
        local_load,
    )
    reversed_action_rows = kernel.local_section_end_actions(
        reversed_model.mesh,
        reversed_model.mesh.elements[0],
        displacement,
        reversed_local_load,
    )

    actions = np.asarray([(row.N, row.My, row.Mz) for row in action_rows])
    reversed_actions = np.asarray(
        [(row.N, row.My, row.Mz) for row in reversed_action_rows]
    )
    assert reversed_actions == pytest.approx(
        actions[::-1] * action_component_reversal,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("translation", "rotation", "sign", "curvature", "half_depth"),
    [(1, 5, 1.0, 0.03, 1.0), (2, 4, -1.0, 0.02, 1.5)],
    ids=["bend-about-z", "bend-about-y"],
)
def test_pure_bending_recovers_analytic_section_extrema_at_both_ends(
    translation, rotation, sign, curvature, half_depth,
):
    mesh = _beam_mesh()
    displacement = np.zeros(mesh.num_dofs)
    displacement[mesh.global_dof(2, translation)] = sign * 0.5 * curvature * 4.0**2
    displacement[mesh.global_dof(2, rotation)] = curvature * 4.0

    rows = beam.nodal_envelope(_beam_result(mesh, displacement))

    increment = 210.0 * curvature * half_depth
    np.testing.assert_allclose(
        [(row.maximum, row.minimum, row.absolute_maximum) for row in rows],
        [(increment, -increment, increment)] * 2,
    )
