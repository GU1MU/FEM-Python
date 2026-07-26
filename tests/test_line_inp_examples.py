from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fem.abaqus import read
from fem.elements import get_element_kernel
from fem.post.stress import beam
from fem.solvers.static_linear import solve


FIXTURES = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)


def test_truss2_tension_inp_solves_to_the_expected_axial_stress():
    model = read(FIXTURES / "truss2_tension.inp")
    result = solve(model, "Tension")
    element = model.mesh.elements[0]
    strain, stress, mises = get_element_kernel(element.type).element_stress(
        model.mesh,
        element,
        result.U,
    )

    assert element.type == "Truss2"
    assert model.mesh.dofs_per_node == 3
    assert strain == pytest.approx(1.0e-3)
    assert stress == pytest.approx(2.10e8)
    assert mises == pytest.approx(2.10e8)


@pytest.mark.parametrize(
    "name, section_type",
    (
        ("beam2_rectangle_tip_load.inp", "rectangle"),
        ("beam2_rectangle_uniform_load.inp", "rectangle"),
        ("beam2_hollow_circle_uniform_load.inp", "hollow_circle"),
        ("beam2_solid_circle_inclined.inp", "solid_circle"),
    ),
)
def test_beam2_inp_examples_solve_and_recover_nodal_envelopes(
    name: str,
    section_type: str,
):
    model = read(FIXTURES / name)
    step = next(step for step in model.steps if step.name.lower() != "initial")
    result = solve(model, step.name)
    envelope = beam.nodal_envelope(result)

    assert {element.type for element in model.mesh.elements} == {"Beam2"}
    assert model.mesh.dofs_per_node == 6
    assert model.sections[0].section_type == section_type
    assert np.isfinite(result.U).all()
    assert len(envelope) == len(model.mesh.nodes)
    assert max(row.absolute_maximum for row in envelope) > 0.0


def test_beam2_uniform_load_inp_preserves_global_and_local_vectors():
    local_model = read(FIXTURES / "beam2_rectangle_uniform_load.inp")
    global_model = read(FIXTURES / "beam2_hollow_circle_uniform_load.inp")
    local_step = next(step for step in local_model.steps if step.name == "UniformLoad")
    global_step = next(step for step in global_model.steps if step.name == "UniformLoad")

    assert local_step.line_loads[0].coordinate_system == "local"
    assert local_step.line_loads[0].vector == (0.0, -500.0, 0.0)
    assert global_step.line_loads[0].coordinate_system == "global"
    assert global_step.line_loads[0].vector == (0.0, 0.0, -300.0)
