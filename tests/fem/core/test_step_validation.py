import pytest

from fem import materials
from fem.core import validate_analysis_step, validate_model, validate_model_structure
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    ElementSet,
    FEMModel,
    GravityLoad,
    LineLoad,
    MaterialDefinition,
    NodalLoad,
    SectionAssignment,
)
from tests.helpers.mesh_builders import make_beam_stiffness_mesh, make_truss_stiffness_mesh
from tests.helpers.model_builders import (
    make_static_pull_truss_model,
    make_two_step_static_pull_truss_model,
    make_truss_workflow_model,
)


def test_selected_step_validation_ignores_unrelated_step_reference_errors():
    model = make_two_step_static_pull_truss_model()
    selected = model.steps[1]
    invalid = model.steps[2]
    invalid.cloads = (NodalLoad("MISSING_IN_OTHER_STEP", 1, 1.0),)

    validate_model_structure(model)
    validate_analysis_step(model, selected)
    validate_model(model, selected)

    with pytest.raises(KeyError, match="MISSING_IN_OTHER_STEP"):
        validate_analysis_step(model, invalid)
    with pytest.raises(KeyError, match="MISSING_IN_OTHER_STEP"):
        validate_model(model)


@pytest.mark.parametrize(
    ("source_index", "selected_index", "missing_target"),
    [(0, 1, "MISSING_INITIAL_TARGET"), (1, 2, "MISSING_PREVIOUS_TARGET")],
    ids=["initial-boundaries", "previous-step-boundaries"],
)
def test_selected_step_validation_includes_inherited_boundaries(
    source_index, selected_index, missing_target
):
    model = make_two_step_static_pull_truss_model()
    selected = model.steps[selected_index]
    model.steps[source_index].boundaries = (
        DisplacementConstraint(missing_target, 1, 3, 0.0),
    )

    validate_model_structure(model)
    with pytest.raises(KeyError, match=missing_target):
        validate_analysis_step(model, selected)
    with pytest.raises(KeyError, match=missing_target):
        validate_model(model, selected)


def test_selected_step_does_not_inherit_initial_loads():
    model = make_two_step_static_pull_truss_model()
    selected = model.steps[1]
    model.steps[0].cloads = (NodalLoad("MISSING_INITIAL_LOAD", 1, 1.0),)

    validate_model(model, selected)

    with pytest.raises(KeyError, match="MISSING_INITIAL_LOAD"):
        validate_model(model)


@pytest.mark.parametrize(
    ("first", "last"),
    [(0, 1), (2, 1), (1, 4)],
)
def test_validate_model_rejects_invalid_constraint_component_ranges(first, last):
    model = make_static_pull_truss_model()
    model.steps[0].boundaries = (
        DisplacementConstraint("FIXED", first, last, 0.0),
    )

    with pytest.raises(ValueError, match="constraint components must satisfy"):
        validate_model(model)


@pytest.mark.parametrize("component", [0, 4])
def test_validate_model_rejects_invalid_load_components(component):
    model = make_static_pull_truss_model()
    model.steps[0].cloads = (NodalLoad("TIP", component, 1.0),)

    with pytest.raises(ValueError, match="load component must be from 1 through 3"):
        validate_model(model)


@pytest.mark.parametrize("kind", ["constraint", "load"])
def test_validate_model_rejects_nonfinite_step_values(kind):
    model = make_static_pull_truss_model()
    if kind == "constraint":
        model.steps[0].boundaries = (
            DisplacementConstraint("FIXED", 1, 1, float("nan")),
        )
    else:
        model.steps[0].cloads = (NodalLoad("TIP", 1, float("inf")),)

    with pytest.raises(ValueError, match=rf"{kind} value must be finite"):
        validate_model(model)


def test_validate_model_accepts_global_id_and_set_gravity_with_effective_density():
    model = make_truss_workflow_model()
    materials.add(
        model,
        materials.linear_elastic.material("steel", E=100.0, nu=0.3, rho=0.0),
    )
    materials.assign(model, "steel", "bar", area=2.0)
    model.steps.append(
        AnalysisStep(
            "gravity",
            gravity_loads=(
                GravityLoad((0.0, -9.81, 0.0)),
                GravityLoad((0.0, 0.0, 0.0), 1),
                GravityLoad((1.0, 0.0, 0.0), "bar"),
            ),
        )
    )

    validate_model(model)

    assert "rho" not in model.mesh.elements[0].props


def test_validate_model_allows_global_gravity_on_massless_elements():
    model = make_truss_workflow_model()
    model.steps.append(
        AnalysisStep("gravity", gravity_loads=(GravityLoad((0.0, -9.81, 0.0)),))
    )

    validate_model(model)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (99, "gravity target references missing element 99"),
        ("missing", "gravity target references missing element set missing"),
    ],
)
def test_validate_model_rejects_unknown_gravity_targets_with_step_context(
    target,
    message,
):
    model = make_truss_workflow_model()
    model.steps.append(
        AnalysisStep("gravity_case", gravity_loads=(GravityLoad((0.0, -1.0, 0.0), target),))
    )

    with pytest.raises(KeyError, match=rf"analysis step gravity_case {message}"):
        validate_model(model)


@pytest.mark.parametrize(
    ("acceleration", "error", "message"),
    [
        ((0.0, -1.0), ValueError, "must have 3 components"),
        ((0.0, float("nan"), 0.0), ValueError, "component must be finite"),
        ((0.0, "bad", 0.0), TypeError, "component must be numeric"),
    ],
)
def test_validate_model_rejects_invalid_gravity_acceleration(
    acceleration,
    error,
    message,
):
    model = make_truss_workflow_model()
    model.steps.append(
        AnalysisStep("gravity", gravity_loads=(GravityLoad(acceleration),))
    )

    with pytest.raises(error, match=message):
        validate_model(model)


@pytest.mark.parametrize("rho", [None, -1.0, float("nan")])
def test_validate_model_requires_valid_effective_density_for_targeted_gravity(rho):
    model = make_truss_workflow_model()
    properties = {"E": 100.0, "nu": 0.3}
    if rho is not None:
        properties["rho"] = rho
    model.materials["steel"] = MaterialDefinition("steel", properties)
    model.sections.append(SectionAssignment("bar", "steel", properties={"area": 2.0}))
    model.steps.append(
        AnalysisStep(
            "gravity",
            gravity_loads=(GravityLoad((0.0, -1.0, 0.0), "bar"),),
        )
    )

    with pytest.raises(ValueError, match=r"gravity target 'bar'.*density rho"):
        validate_model(model)


def test_targeted_gravity_validation_ignores_stale_section_density():
    model = make_truss_workflow_model()
    model.materials["steel"] = MaterialDefinition(
        "steel",
        {"E": 100.0, "nu": 0.3, "rho": 2.0},
    )
    model.sections.append(SectionAssignment("bar", "steel", properties={"area": 2.0}))
    materials.apply_sections(model)
    model.materials["steel"] = MaterialDefinition("steel", {"E": 100.0, "nu": 0.3})
    model.steps.append(
        AnalysisStep(
            "gravity",
            gravity_loads=(GravityLoad((0.0, -1.0, 0.0), 1),),
        )
    )

    with pytest.raises(ValueError, match="requires an effective density rho"):
        validate_model(model)


def test_public_element_set_wins_over_same_named_internal_set_for_gravity():
    mesh = Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 1.0, 0.0, 0.0),
            Node3D(3, 2.0, 0.0, 0.0),
        ],
        elements=[
            Element3D(
                1,
                [1, 2],
                "Truss2",
                {"E": 100.0, "area": 1.0, "rho": 1.0},
            ),
            Element3D(2, [2, 3], "Truss2", {"E": 100.0, "area": 1.0}),
        ],
    )
    model = FEMModel(
        mesh=mesh,
        element_sets={"shared": ElementSet("shared", (1,))},
        steps=[
            AnalysisStep(
                "gravity",
                gravity_loads=(GravityLoad((0.0, -1.0, 0.0), "shared"),),
            )
        ],
        metadata={
            "_abaqus_internal_element_sets": {
                "shared": ElementSet("shared", (2,)),
            }
        },
    )

    validate_model(model)


@pytest.mark.parametrize(
    "mesh_factory",
    [make_truss_stiffness_mesh, make_beam_stiffness_mesh],
    ids=["truss", "beam"],
)
def test_line_element_gravity_requires_three_spatial_components(
    mesh_factory,
):
    model = FEMModel(
        mesh=mesh_factory(),
        steps=[
            AnalysisStep(
                "gravity",
                gravity_loads=[GravityLoad((0.0, -1.0, 0.0))],
            )
        ],
    )

    validate_model(model)

    model.steps[0].gravity_loads = (GravityLoad((0.0, -1.0)),)
    with pytest.raises(ValueError, match="must have 3 components"):
        validate_model(model)


def test_step_validation_rejects_line_loads_on_truss_elements():
    model = FEMModel(
        mesh=make_truss_stiffness_mesh(),
        element_sets={"bar": ElementSet("bar", (1,))},
    )
    step = AnalysisStep(
        "line",
        line_loads=(LineLoad("bar", (0.0, -1.0, 0.0)),),
    )

    validate_model_structure(model)
    with pytest.raises(ValueError, match="line loads may target only Beam2"):
        validate_analysis_step(model, step)
