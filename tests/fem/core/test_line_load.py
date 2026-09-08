import numpy as np
import pytest

from fem import steps
from fem.boundary.loads import build_load_vector
from fem.boundary.step import boundary_for_step
from fem.core import validate_model
from fem.core.mesh import Element3D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    ElementSet,
    GravityLoad,
    LineLoad,
)
from fem.elements import (
    get_element_kernel,
    resolve_beam_frame,
)
from fem.elements.beam_section import parse_beam2_section
from fem.solvers import static_linear
from tests.helpers.model_builders import make_line_load_beam_model


def _step_force(model, step):
    return build_load_vector(model.mesh, boundary_for_step(model, step))


def test_line_load_public_model_and_step_api_preserve_definition():
    step = steps.static("distributed")

    load = steps.line_load(step, "beams", (1.0, 2.0, 3.0), "local")

    assert load == LineLoad("beams", (1.0, 2.0, 3.0), "local")
    assert step.line_loads == (load,)


def test_line_load_element_id_and_element_set_targets_are_equivalent():
    model = make_line_load_beam_model()
    by_id = AnalysisStep("id", line_loads=(LineLoad(10, (0.0, 2.0, 0.0)),))
    by_set = AnalysisStep(
        "set", line_loads=(LineLoad("beams", (0.0, 2.0, 0.0)),)
    )

    assert _step_force(model, by_id) == pytest.approx(_step_force(model, by_set))


def test_line_load_resolves_importer_internal_element_set():
    model = make_line_load_beam_model()
    model.element_sets.clear()
    model.metadata["_abaqus_internal_element_sets"] = {
        "imported": ElementSet("imported", (10,))
    }
    by_id = AnalysisStep(
        "id",
        line_loads=(LineLoad(10, (0.0, 2.0, 0.0)),),
    )
    imported = AnalysisStep(
        "set",
        line_loads=(LineLoad("imported", (0.0, 2.0, 0.0)),),
    )

    validate_model(model, imported)
    assert _step_force(model, imported) == pytest.approx(
        _step_force(model, by_id)
    )


def test_public_line_load_set_wins_over_same_named_internal_set():
    model = make_line_load_beam_model()
    model.element_sets["shared"] = ElementSet("shared", (10,))
    model.metadata["_abaqus_internal_element_sets"] = {
        "shared": ElementSet("shared", (999,))
    }
    by_id = AnalysisStep(
        "id",
        line_loads=(LineLoad(10, (0.0, 2.0, 0.0)),),
    )
    shared = AnalysisStep(
        "set",
        line_loads=(LineLoad("shared", (0.0, 2.0, 0.0)),),
    )

    validate_model(model, shared)
    assert _step_force(model, shared) == pytest.approx(
        _step_force(model, by_id)
    )


@pytest.mark.parametrize(
    ("vector", "expected"),
    [
        ((2.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ((0.0, 3.0, 0.0), (0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0, 3.0), (0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0)),
    ],
    ids=["local-x", "local-y", "local-z"],
)
def test_line_load_three_local_directions_match_consistent_nodal_vector(vector, expected):
    model = make_line_load_beam_model()
    step = AnalysisStep("load", line_loads=(LineLoad(10, vector, "local"),))

    assert _step_force(model, step) == pytest.approx(expected)


def test_global_and_equivalent_local_line_loads_match_on_inclined_beam():
    model = make_line_load_beam_model(
        inclined=True,
        orientation=(0.0, 1.0, 0.0),
    )
    global_vector = np.array([1.5, -2.0, 0.25])
    frame = resolve_beam_frame(model.mesh, model.mesh.elements[0])
    local_vector = frame.rotation @ global_vector
    global_step = AnalysisStep(
        "global", line_loads=(LineLoad(10, global_vector, "global"),)
    )
    local_step = AnalysisStep(
        "local", line_loads=(LineLoad(10, local_vector, "local"),)
    )

    assert _step_force(model, global_step) == pytest.approx(
        _step_force(model, local_step)
    )


@pytest.mark.parametrize("coordinate_system", ["global", "local"])
def test_explicit_orientation_line_load_reversal_preserves_physical_force(coordinate_system):
    model = make_line_load_beam_model(inclined=True, orientation=(0.0, 1.0, 0.0))
    reversed_model = make_line_load_beam_model(inclined=True, orientation=(0.0, 1.0, 0.0))
    reversed_model.mesh.elements[0].node_ids = [2, 1]
    kernel = get_element_kernel("Beam2")
    permutation = np.eye(12)[[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]]
    vector = np.array([2.0, 3.0, 4.0])
    # Reversing connectivity flips local x and z while retaining local y.
    reversed_vector = (
        vector * [-1.0, 1.0, -1.0] if coordinate_system == "local" else vector
    )

    force = kernel.line_load(
        model.mesh, model.mesh.elements[0], vector, coordinate_system,
    )
    reversed_force = kernel.line_load(
        reversed_model.mesh,
        reversed_model.mesh.elements[0],
        reversed_vector,
        coordinate_system,
    )

    assert reversed_force == pytest.approx(permutation @ force, abs=1e-12)


def test_multiple_line_loads_accumulate_without_area_or_density_scaling():
    model = make_line_load_beam_model()
    first = AnalysisStep("first", line_loads=(LineLoad(10, (0.0, 2.0, 0.0)),))
    combined = AnalysisStep(
        "combined",
        line_loads=(
            LineLoad(10, (0.0, 2.0, 0.0)),
            LineLoad("beams", (0.0, 3.0, 0.0)),
        ),
    )

    expected = np.zeros(model.mesh.num_dofs)
    # Each end receives (2 + 3) * L / 2 = 10, independent of rho=99 and area=6.
    expected[[model.mesh.global_dof(1, 1), model.mesh.global_dof(2, 1)]] = 10.0
    combined_force = _step_force(model, combined)
    assert combined_force == pytest.approx(expected)
    assert combined_force == pytest.approx(2.5 * _step_force(model, first))


def test_beam_gravity_uses_rho_area_then_the_b31_line_load_path():
    model = make_line_load_beam_model(inclined=True, orientation=(0.0, 1.0, 0.0))
    element = model.mesh.elements[0]
    section = parse_beam2_section(element.props)
    acceleration = np.array((1.25, -2.5, 0.75))
    gravity = AnalysisStep(
        "gravity",
        gravity_loads=(GravityLoad(acceleration),),
    )
    equivalent_line = AnalysisStep(
        "line",
        line_loads=(
            LineLoad(
                10,
                element.props["rho"] * section.area * acceleration,
                "global",
            ),
        ),
    )

    assert _step_force(model, gravity) == pytest.approx(
        _step_force(model, equivalent_line),
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_line_load_preserves_global_resultant_and_moment_about_arbitrary_origin():
    model = make_line_load_beam_model(inclined=True)
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

    expected_moment = np.cross((start + end) / 2.0 - origin, resultant)
    force_error = np.linalg.norm(force[:3] + force[6:9] - resultant)
    moment_error = np.linalg.norm(assembled_moment - expected_moment)

    assert force_error <= 1.0e-12 * max(1.0, np.linalg.norm(resultant))
    assert moment_error <= 1.0e-12 * max(1.0, np.linalg.norm(expected_moment))


def test_uniform_transverse_line_load_cantilever_matches_b31_discrete_response():
    model = make_line_load_beam_model()
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
    nu = model.mesh.elements[0].props["nu"]
    section = parse_beam2_section(model.mesh.elements[0].props)
    Izz = section.Izz
    shear_modulus = E / (2.0 * (1.0 + nu))
    shear_y, _ = section.abaqus_b31_shear_rigidities(
        shear_modulus,
        nu,
        length,
    )

    expected_tip = (
        q * length**4 / (8.0 * E * Izz)
        + q * length**2 / (2.0 * shear_y)
    )
    assert result.U[mesh.global_dof(2, 1)] == pytest.approx(expected_tip)
    assert result.U[mesh.global_dof(2, 5)] == pytest.approx(q * length**3 / (4.0 * E * Izz))
    assert result.reactions[mesh.global_dof(1, 1)] == pytest.approx(-q * length)
    assert result.reactions[mesh.global_dof(1, 5)] == pytest.approx(-q * length**2 / 2.0)


@pytest.mark.parametrize(
    ("load", "message"),
    [
        (LineLoad(10, (1.0, 2.0, 3.0), "cylindrical"), "coordinate_system"),
        (LineLoad(10, (1.0, 2.0)), "line load vector"),
        (LineLoad(10, (1.0, 2.0, np.nan)), "line load vector"),
    ],
    ids=(
        "unsupported-coordinates",
        "short-vector",
        "nonfinite-vector",
    ),
)
def test_line_load_assembly_rejects_invalid_definitions(load, message):
    model = make_line_load_beam_model()
    step = AnalysisStep("bad", line_loads=(load,))

    with pytest.raises(ValueError, match=message):
        _step_force(model, step)


@pytest.mark.parametrize("target", [20, "mixed"], ids=("non-beam", "mixed-set"))
def test_line_load_rejects_non_beam_and_mixed_element_set_targets(target):
    model = make_line_load_beam_model()
    model.mesh.elements.append(
        Element3D(20, [1, 2], "Truss2", {"E": 1.0, "area": 1.0})
    )
    model.element_sets["mixed"] = ElementSet("mixed", (10, 20))

    step = AnalysisStep("bad", line_loads=(LineLoad(target, (1.0, 0.0, 0.0)),))
    with pytest.raises(ValueError, match="only Beam2"):
        boundary_for_step(model, step)


@pytest.mark.parametrize("target", [999, "missing"])
def test_model_validation_rejects_missing_line_load_targets(target):
    model = make_line_load_beam_model()
    step = AnalysisStep("bad", line_loads=(LineLoad(target, (1.0, 0.0, 0.0)),))

    with pytest.raises(KeyError, match="missing (element|element set)"):
        validate_model(model, step)


@pytest.mark.parametrize(
    ("load", "message"),
    [
        (LineLoad(10, (1.0, 2.0)), "line load vector"),
        (LineLoad(10, (1.0, 2.0, np.nan)), "line load vector"),
        (LineLoad(10, (1.0, 2.0, 3.0), "cylindrical"), "coordinate_system"),
    ],
    ids=("short-vector", "nonfinite-vector", "coordinate-system"),
)
def test_model_validation_rejects_invalid_line_load_definitions(load, message):
    model = make_line_load_beam_model()

    with pytest.raises(ValueError, match=message):
        validate_model(model, AnalysisStep("bad", line_loads=(load,)))
