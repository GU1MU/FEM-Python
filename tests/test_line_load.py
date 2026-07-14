import numpy as np
import pytest

from fem import steps
from fem.boundary.loads import build_load_vector
from fem.boundary.step import boundary_for_step
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core import validate_model
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    ElementSet,
    FEMModel,
    LineLoad,
    NodalLoad,
)
from fem.core.result import ModelResult
from fem.elements import get_element_kernel
from fem.elements.beam_section import parse_beam2_section
from fem.elements.line import beam3_geometry
from fem.post.stress import beam as beam_stress
from fem.solvers import static_linear


def _beam_model(*, inclined=False):
    end = (2.0, 3.0, 6.0) if inclined else (4.0, 0.0, 0.0)
    mesh = Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, *end)],
        elements=[
            Element3D(
                10,
                [1, 2],
                "Beam2",
                {
                    "E": 210.0,
                    "nu": 0.25,
                    "section_type": "rectangle",
                    "height": 3.0,
                    "width": 2.0,
                    "rho": 99.0,
                },
            )
        ],
        dofs_per_node=6,
    )
    return FEMModel(
        mesh=mesh,
        element_sets={"beams": ElementSet("beams", (10,))},
    )


def _step_force(model, step):
    return build_load_vector(model.mesh, boundary_for_step(model, step))


def test_line_load_public_model_and_step_api_preserve_definition():
    step = steps.static("distributed")

    load = steps.line_load(step, "beams", (1.0, 2.0, 3.0), "local")

    assert load == LineLoad("beams", (1.0, 2.0, 3.0), "local")
    assert step.line_loads == (load,)


def test_line_load_element_id_and_element_set_targets_are_equivalent():
    model = _beam_model()
    by_id = AnalysisStep("id", line_loads=(LineLoad(10, (0.0, 2.0, 0.0)),))
    by_set = AnalysisStep(
        "set", line_loads=(LineLoad("beams", (0.0, 2.0, 0.0)),)
    )

    assert _step_force(model, by_id) == pytest.approx(_step_force(model, by_set))


@pytest.mark.parametrize(
    ("vector", "expected"),
    [
        ((2.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ((0.0, 3.0, 0.0), (0.0, 6.0, 0.0, 0.0, 0.0, 4.0, 0.0, 6.0, 0.0, 0.0, 0.0, -4.0)),
        ((0.0, 0.0, 3.0), (0.0, 0.0, 6.0, 0.0, -4.0, 0.0, 0.0, 0.0, 6.0, 0.0, 4.0, 0.0)),
    ],
    ids=["local-x", "local-y", "local-z"],
)
def test_line_load_three_local_directions_match_consistent_nodal_vector(vector, expected):
    model = _beam_model()
    step = AnalysisStep("load", line_loads=(LineLoad(10, vector, "local"),))

    assert _step_force(model, step) == pytest.approx(expected)


def test_global_and_equivalent_local_line_loads_match_on_inclined_beam():
    model = _beam_model(inclined=True)
    global_vector = np.array([1.5, -2.0, 0.25])
    _, rotation = beam3_geometry(model.mesh, model.mesh.elements[0])
    local_vector = rotation @ global_vector
    global_step = AnalysisStep(
        "global", line_loads=(LineLoad(10, global_vector, "global"),)
    )
    local_step = AnalysisStep(
        "local", line_loads=(LineLoad(10, local_vector, "local"),)
    )

    assert _step_force(model, global_step) == pytest.approx(
        _step_force(model, local_step)
    )


def test_multiple_line_loads_accumulate_without_area_density_or_gravity_scaling():
    model = _beam_model()
    first = AnalysisStep("first", line_loads=(LineLoad(10, (0.0, 2.0, 0.0)),))
    combined = AnalysisStep(
        "combined",
        line_loads=(
            LineLoad(10, (0.0, 2.0, 0.0)),
            LineLoad("beams", (0.0, 3.0, 0.0)),
        ),
    )

    assert _step_force(model, combined) == pytest.approx(2.5 * _step_force(model, first))


def test_inclined_global_line_loads_accumulate_in_recovered_stress_envelope():
    model = _beam_model(inclined=True)
    mesh = model.mesh
    elem = mesh.elements[0]
    length, rotation = beam3_geometry(mesh, elem)
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
        return beam_stress.nodal_envelope(result)

    multi_rows = recover(multi_global_step)
    local_rows = recover(combined_local_step)

    multi_values = [
        (row.maximum, row.minimum, row.absolute_maximum) for row in multi_rows
    ]
    local_values = [
        (row.maximum, row.minimum, row.absolute_maximum) for row in local_rows
    ]
    assert np.allclose(multi_values, local_values)

    section = parse_beam2_section(elem.props)
    qx, qy, qz = combined_local
    axial = qx * length / (2.0 * section.area)
    moment_y = -qz * length**2 / 12.0
    moment_z = qy * length**2 / 12.0
    increment = (
        abs(moment_y / section.Iyy) * section.width / 2.0
        + abs(moment_z / section.Izz) * section.height / 2.0
    )
    expected = [
        (axial + increment, axial - increment, axial + increment),
        (-axial + increment, -axial - increment, axial + increment),
    ]
    assert np.allclose(multi_values, expected)


def test_line_load_preserves_global_resultant_and_moment_about_arbitrary_origin():
    model = _beam_model(inclined=True)
    q = np.array([1.5, -2.0, 0.25])
    force = _step_force(
        model,
        AnalysisStep("load", line_loads=(LineLoad(10, q, "global"),)),
    )
    origin = np.array([-1.0, 2.0, 0.5])
    start = np.zeros(3)
    end = np.array([2.0, 3.0, 6.0])
    resultant = q * 7.0
    assembled_moment = (
        np.cross(start - origin, force[:3])
        + force[3:6]
        + np.cross(end - origin, force[6:9])
        + force[9:12]
    )

    assert force[:3] + force[6:9] == pytest.approx(resultant)
    assert assembled_moment == pytest.approx(np.cross((start + end) / 2.0 - origin, resultant))


def test_uniform_transverse_line_load_cantilever_matches_closed_form_and_reactions():
    model = _beam_model()
    q = 12.0
    model.steps.append(
        AnalysisStep(
            "bend",
            boundaries=(DisplacementConstraint(1, 1, 6, 0.0),),
            line_loads=(LineLoad(10, (0.0, q, 0.0), "local"),),
        )
    )

    result = static_linear.solve(model, "bend")
    mesh = model.mesh
    length = 4.0
    E = model.mesh.elements[0].props["E"]
    Izz = parse_beam2_section(model.mesh.elements[0].props).Izz

    assert result.U[mesh.global_dof(2, 1)] == pytest.approx(q * length**4 / (8.0 * E * Izz))
    assert result.U[mesh.global_dof(2, 5)] == pytest.approx(q * length**3 / (6.0 * E * Izz))
    assert result.reactions[mesh.global_dof(1, 1)] == pytest.approx(-q * length)
    assert result.reactions[mesh.global_dof(1, 5)] == pytest.approx(-q * length**2 / 2.0)


def test_inclined_cantilever_solution_recovers_combined_axial_and_biaxial_bending():
    model = _beam_model(inclined=True)
    mesh = model.mesh
    elem = mesh.elements[0]
    length, rotation = beam3_geometry(mesh, elem)
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

    end_actions = get_element_kernel("Beam2").local_end_actions(
        mesh,
        elem,
        result.U,
    )
    moment_y = -force_z * length
    moment_z = force_y * length
    assert np.allclose(
        end_actions,
        [
            (axial_force, moment_y, moment_z),
            (axial_force, 0.0, 0.0),
        ],
        atol=1e-10,
    )

    section = parse_beam2_section(elem.props)
    axial_stress = axial_force / section.area
    increment = (
        abs(moment_y / section.Iyy) * section.width / 2.0
        + abs(moment_z / section.Izz) * section.height / 2.0
    )
    rows = beam_stress.nodal_envelope(result)
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
            (axial_stress, axial_stress, axial_stress),
        ],
        atol=1e-10,
    )


def test_fixed_beam_uniform_line_load_recovers_bending_stress_with_zero_displacement():
    model = _beam_model()
    q = 12.0
    step = AnalysisStep(
        "fixed",
        line_loads=(LineLoad(10, (0.0, q, 0.0), "local"),),
    )
    result = ModelResult(
        model,
        step,
        np.zeros(model.mesh.num_dofs),
        np.zeros(model.mesh.num_dofs),
    )

    rows = beam_stress.nodal_envelope(result)

    section = parse_beam2_section(model.mesh.elements[0].props)
    moment = q * 4.0**2 / 12.0
    increment = abs(moment / section.Izz) * section.height / 2.0
    assert [(row.maximum, row.minimum, row.absolute_maximum) for row in rows] == pytest.approx(
        [(increment, -increment, increment), (increment, -increment, increment)]
    )


@pytest.mark.parametrize("coordinate_system", ["GLOBAL ", "cylindrical", ""])
def test_line_load_rejects_invalid_coordinate_system(coordinate_system):
    model = _beam_model()
    step = AnalysisStep(
        "bad", line_loads=(LineLoad(10, (1.0, 2.0, 3.0), coordinate_system),)
    )

    with pytest.raises(ValueError, match="coordinate_system"):
        _step_force(model, step)


@pytest.mark.parametrize("vector", [(1.0, 2.0), (1.0, 2.0, np.nan)])
def test_line_load_rejects_invalid_vector(vector):
    model = _beam_model()
    step = AnalysisStep("bad", line_loads=(LineLoad(10, vector),))

    with pytest.raises(ValueError, match="line load vector"):
        _step_force(model, step)


def test_line_load_rejects_non_beam_and_mixed_element_set_targets():
    model = _beam_model()
    model.mesh.elements.append(
        Element3D(20, [1, 2], "Truss2", {"E": 1.0, "area": 1.0})
    )
    model.element_sets["mixed"] = ElementSet("mixed", (10, 20))

    for target in (20, "mixed"):
        step = AnalysisStep("bad", line_loads=(LineLoad(target, (1.0, 0.0, 0.0)),))
        with pytest.raises(ValueError, match="only Beam2"):
            boundary_for_step(model, step)


@pytest.mark.parametrize("target", [999, "missing"])
def test_model_validation_rejects_missing_line_load_targets(target):
    model = _beam_model()
    step = AnalysisStep("bad", line_loads=(LineLoad(target, (1.0, 0.0, 0.0)),))

    with pytest.raises(KeyError, match="missing (element|element set)"):
        validate_model(model, step)


@pytest.mark.parametrize(
    ("load", "message"),
    [
        (LineLoad(10, (1.0, 2.0)), "line load vector"),
        (LineLoad(10, (1.0, 2.0, np.nan)), "line load vector"),
        (LineLoad(10, (1.0, 2.0, 3.0), "GLOBAL"), "coordinate_system"),
    ],
)
def test_model_validation_rejects_invalid_line_load_definitions(load, message):
    model = _beam_model()

    with pytest.raises(ValueError, match=message):
        validate_model(model, AnalysisStep("bad", line_loads=(load,)))
