from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fem import materials
from fem.materials import assignment as material_assignment
from fem.assemble import assemble_global_stiffness_sparse
from fem.assemble import stiffness as stiffness_module
from fem.core import validate_mesh, validate_model
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    ElementSet,
    FEMModel,
    GravityLoad,
    MaterialDefinition,
    NodalLoad,
    SectionAssignment,
)
from fem.core.result import ModelResult
from fem.solvers import static_linear
from tests.helpers.mesh_builders import make_beam_stiffness_mesh, make_truss_stiffness_mesh
from tests.helpers.model_builders import (
    make_static_pull_truss_model,
    make_truss_workflow_model,
)


def test_validate_mesh_rejects_stale_dof_map_until_explicit_rebuild():
    mesh = make_truss_stiffness_mesh()
    mesh.nodes.append(Node3D(3, 3.0, 0.0, 0.0))

    with pytest.raises(ValueError, match=r"DofMap.*rebuild_dof_map"):
        validate_mesh(mesh)

    mesh.rebuild_dof_map()

    validate_mesh(mesh)
    assert mesh.node_ids == [1, 2, 3]
    assert mesh.num_dofs == 9


def test_validate_mesh_rejects_empty_nodes_and_elements_before_assembly():
    no_nodes = Mesh3D(nodes=[], elements=[])
    with pytest.raises(ValueError, match="at least one node"):
        assemble_global_stiffness_sparse(no_nodes)

    no_elements = Mesh3D(nodes=[Node3D(1, 0.0, 0.0, 0.0)], elements=[])
    with pytest.raises(ValueError, match="at least one element"):
        assemble_global_stiffness_sparse(no_elements)


def test_validate_mesh_rejects_duplicate_node_and_element_ids():
    duplicate_node_mesh = make_truss_stiffness_mesh()
    duplicate_node_mesh.nodes.append(Node3D(1, 3.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="node ids must be unique"):
        validate_mesh(duplicate_node_mesh)

    duplicate_element_mesh = make_truss_stiffness_mesh()
    duplicate_element_mesh.elements.append(deepcopy(duplicate_element_mesh.elements[0]))
    with pytest.raises(ValueError, match="element ids must be unique"):
        assemble_global_stiffness_sparse(duplicate_element_mesh)


def test_validate_mesh_rejects_missing_connectivity_and_nonfinite_coordinates():
    missing_node_mesh = make_truss_stiffness_mesh()
    missing_node_mesh.elements[0].node_ids[-1] = 999
    with pytest.raises(KeyError, match="element 1 references missing node 999"):
        validate_mesh(missing_node_mesh)

    nonfinite_mesh = make_truss_stiffness_mesh()
    nonfinite_mesh.nodes[0].x = float("nan")
    with pytest.raises(ValueError, match="coordinate x must be finite"):
        validate_mesh(nonfinite_mesh)


@pytest.mark.parametrize("invalid_coordinate", ["1.0", True])
def test_validate_mesh_rejects_non_real_coordinates(invalid_coordinate):
    mesh = make_truss_stiffness_mesh()
    mesh.nodes[0].x = invalid_coordinate

    with pytest.raises(TypeError, match="coordinate x must be a real number"):
        validate_mesh(mesh)


def test_validate_mesh_rejects_repeated_node_ids_within_an_element():
    mesh = make_truss_stiffness_mesh()
    mesh.elements[0].node_ids = [1, 1]

    with pytest.raises(ValueError, match="element 1 node_ids must be unique"):
        validate_mesh(mesh)


def test_validate_model_rejects_invalid_set_and_section_references():
    invalid_set_model = make_truss_workflow_model()
    invalid_set_model.element_sets["bad"] = ElementSet("bad", (999,))
    with pytest.raises(KeyError, match="element set bad references missing element 999"):
        validate_model(invalid_set_model)

    missing_material_model = make_truss_workflow_model()
    missing_material_model.sections.append(SectionAssignment("bar", "missing"))
    with pytest.raises(KeyError, match="material missing is not defined"):
        validate_model(missing_material_model)

    missing_set_model = make_truss_workflow_model()
    materials.add(
        missing_set_model,
        materials.linear_elastic.material("steel", E=100.0, nu=0.3),
    )
    missing_set_model.sections.append(SectionAssignment("missing", "steel"))
    with pytest.raises(KeyError, match="element set missing is not defined"):
        validate_model(missing_set_model)


def test_validate_model_rejects_case_insensitive_duplicate_step_names():
    model = make_static_pull_truss_model()
    model.steps.append(AnalysisStep("PULL"))

    with pytest.raises(ValueError, match="step names must be unique ignoring case"):
        validate_model(model)


@pytest.mark.parametrize(
    ("procedure", "nlgeom", "message"),
    [
        ("dynamic", None, "requires procedure 'static'"),
        ("static", True, "does not support nlgeom"),
        ("static", "YES", "does not support nlgeom"),
    ],
)
def test_static_solver_rejects_unsupported_step_semantics(
    procedure,
    nlgeom,
    message,
):
    model = make_static_pull_truss_model()
    model.steps[0].procedure = procedure
    if nlgeom is not None:
        model.steps[0].metadata["nlgeom"] = nlgeom

    with pytest.raises(ValueError, match=message):
        static_linear.solve(model, "pull")


def test_static_solver_accepts_explicit_false_nlgeom():
    model = make_static_pull_truss_model()
    model.steps[0].metadata["nlgeom"] = "NO"

    result = static_linear.solve(model, "pull")

    assert result.U[model.mesh.global_dof(2, 0)] == pytest.approx(0.5)


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


@pytest.mark.parametrize("rho", [None, -1.0, float("nan"), float("inf")])
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


def test_gravity_vector_size_uses_spatial_dimension_for_beam_mesh():
    model = FEMModel(
        mesh=make_beam_stiffness_mesh(),
        steps=[AnalysisStep("gravity", gravity_loads=[GravityLoad((0.0, -1.0, 0.0))])],
    )

    validate_model(model)


def test_apply_sections_restores_original_properties_after_change_and_removal():
    original_props = {"E": 10.0, "area": 2.0, "custom": "base"}
    model = make_truss_workflow_model(element_props=dict(original_props))
    materials.add(
        model,
        materials.linear_elastic.material("steel", E=100.0, nu=0.3),
    )
    materials.assign(model, "steel", "bar", area=3.0, first_only="old")

    materials.apply_sections(model)
    elem = model.mesh.elements[0]
    assert elem.props["E"] == 100.0
    assert elem.props["area"] == 3.0
    assert elem.props["first_only"] == "old"

    model.sections.clear()
    materials.add(
        model,
        materials.linear_elastic.material("aluminum", E=50.0, nu=0.25),
    )
    materials.assign(model, "aluminum", "bar", area=4.0)
    materials.apply_sections(model)

    assert elem.props["E"] == 50.0
    assert elem.props["area"] == 4.0
    assert elem.props["custom"] == "base"
    assert "first_only" not in elem.props

    model.sections.clear()
    materials.apply_sections(model)

    assert elem.props == original_props


def _beam_assignment_model():
    mesh = Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 2.0, 0.0, 0.0)],
        elements=[Element3D(1, [1, 2], "Beam2")],
        dofs_per_node=6,
    )
    model = FEMModel(mesh=mesh)
    model.element_sets["beam"] = ElementSet("beam", (1,))
    aluminum = materials.linear_elastic.material(
        "aluminum", E=70.0, nu=0.33, rho=2.7
    )
    materials.add(model, aluminum)
    return model, aluminum


def test_apply_sections_assigns_effective_beam2_material_and_section_properties():
    model, aluminum = _beam_assignment_model()
    materials.assign(
        model,
        aluminum,
        model.element_sets["beam"],
        section_type="solid_circle",
        radius=0.02,
    )

    materials.apply_sections(model)

    elem = model.mesh.elements[0]
    assert elem.props["material"] == "aluminum"
    assert elem.props["E"] == 70.0
    assert elem.props["nu"] == 0.33
    assert elem.props["rho"] == 2.7
    assert elem.props["section_type"] == "solid_circle"
    assert elem.props["radius"] == 0.02


def test_apply_sections_rejects_invalid_beam2_section_transactionally():
    model, aluminum = _beam_assignment_model()
    materials.assign(
        model,
        aluminum,
        "beam",
        section_type="solid_circle",
        radius=0.02,
    )
    materials.apply_sections(model)
    elem = model.mesh.elements[0]
    props_before = deepcopy(elem.props)
    metadata_before = deepcopy(model.metadata)

    model.sections.clear()
    materials.assign(
        model,
        aluminum,
        "beam",
        section_type="solid_circle",
        radius=-1.0,
    )

    with pytest.raises(ValueError, match=r"Element 1.*radius"):
        materials.apply_sections(model)

    assert elem.props == props_before
    assert model.metadata == metadata_before


def test_apply_sections_rejects_missing_effective_beam2_section():
    model, _ = _beam_assignment_model()

    with pytest.raises(ValueError, match=r"Element 1.*section_type"):
        materials.apply_sections(model)


def test_apply_sections_preserves_last_matching_section_semantics():
    model = make_truss_workflow_model(element_props={"area": 2.0})
    materials.add(
        model,
        materials.linear_elastic.material("first", E=100.0, nu=0.3),
    )
    materials.add(
        model,
        materials.linear_elastic.material("last", E=50.0, nu=0.25),
    )
    materials.assign(model, "first", "bar", first_only="remove-me")
    materials.assign(model, "last", "bar", area=4.0)

    materials.apply_sections(model)

    elem = model.mesh.elements[0]
    assert elem.props["material"] == "last"
    assert elem.props["E"] == 50.0
    assert elem.props["area"] == 4.0
    assert "first_only" not in elem.props


def test_apply_sections_does_not_restore_old_baseline_into_replaced_element():
    model = make_truss_workflow_model(element_props={"E": 10.0, "area": 2.0})
    materials.add(
        model,
        materials.linear_elastic.material("steel", E=100.0, nu=0.3),
    )
    materials.assign(model, "steel", "bar")
    materials.apply_sections(model)

    replacement = Element3D(
        1,
        [1, 2],
        "Truss2",
        {"E": 9.0, "area": 9.0, "replacement": True},
    )
    model.mesh.elements[0] = replacement
    materials.apply_sections(model)
    model.sections.clear()
    materials.apply_sections(model)

    assert replacement.props == {
        "E": 9.0,
        "area": 9.0,
        "replacement": True,
    }


@pytest.mark.parametrize("failure", ["material", "set", "element"])
def test_apply_sections_resolution_failure_preserves_previous_state(failure):
    model = make_truss_workflow_model(element_props={"E": 10.0, "area": 2.0})
    materials.add(
        model,
        materials.linear_elastic.material("steel", E=100.0, nu=0.3),
    )
    materials.assign(model, "steel", "bar", area=3.0)
    materials.apply_sections(model)
    elem = model.mesh.elements[0]
    props_before = deepcopy(elem.props)
    metadata_before = deepcopy(model.metadata)

    if failure == "material":
        model.sections[:] = [SectionAssignment("bar", "missing")]
        message = "material missing is not defined"
    elif failure == "set":
        model.sections[:] = [SectionAssignment("missing", "steel")]
        message = "element set missing is not defined"
    else:
        model.element_sets["bar"] = ElementSet("bar", (999,))
        message = "element 999 is not defined"

    with pytest.raises(KeyError, match=message):
        materials.apply_sections(model)

    assert elem.props == props_before
    assert model.metadata == metadata_before


def test_apply_sections_commit_failure_rolls_back_props_and_metadata(monkeypatch):
    model = make_truss_workflow_model(element_props={"E": 10.0, "area": 2.0})
    materials.add(
        model,
        materials.linear_elastic.material("steel", E=100.0, nu=0.3),
    )
    materials.assign(model, "steel", "bar", area=3.0)
    materials.apply_sections(model)
    elem = model.mesh.elements[0]
    props_before = deepcopy(elem.props)
    metadata_before = deepcopy(model.metadata)
    original_restore = material_assignment._restore_tracked_keys

    def restore_then_fail(elem, keys, baseline):
        original_restore(elem, keys, baseline)
        raise RuntimeError("commit failed")

    monkeypatch.setattr(
        material_assignment,
        "_restore_tracked_keys",
        restore_then_fail,
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        materials.apply_sections(model)

    assert elem.props == props_before
    assert model.metadata == metadata_before


def test_model_result_owns_validated_one_dimensional_vectors():
    model = make_static_pull_truss_model()
    num_dofs = model.mesh.num_dofs
    U = np.arange(num_dofs, dtype=float)
    reactions = -U

    result = ModelResult(model, model.steps[0], U, reactions)
    U[:] = 99.0
    reactions[:] = 88.0

    assert np.array_equal(result.U, np.arange(num_dofs, dtype=float))
    assert np.array_equal(result.reactions, -np.arange(num_dofs, dtype=float))


def test_model_result_queries_one_based_nodal_components():
    model = make_static_pull_truss_model()
    num_dofs = model.mesh.num_dofs
    result = ModelResult(
        model,
        model.steps[0],
        np.arange(num_dofs, dtype=float),
        -np.arange(num_dofs, dtype=float),
    )

    dof = model.mesh.global_dof(2, 1)
    assert result.nodal_displacement(2, component=2) == float(dof)
    assert result.nodal_reaction(2, component=2) == float(-dof)


@pytest.mark.parametrize("component", [True, 1.0, "1"])
def test_model_result_nodal_queries_reject_noninteger_components(component):
    model = make_static_pull_truss_model()
    result = ModelResult(
        model,
        model.steps[0],
        np.zeros(model.mesh.num_dofs),
        np.zeros(model.mesh.num_dofs),
    )

    with pytest.raises(TypeError, match="component must be an integer"):
        result.nodal_displacement(2, component=component)


@pytest.mark.parametrize("component", [0, 4])
def test_model_result_nodal_queries_reject_out_of_range_components(component):
    model = make_static_pull_truss_model()
    result = ModelResult(
        model,
        model.steps[0],
        np.zeros(model.mesh.num_dofs),
        np.zeros(model.mesh.num_dofs),
    )

    with pytest.raises(IndexError, match="components are 1-based"):
        result.nodal_reaction(2, component=component)


def test_model_result_rejects_invalid_vectors():
    model = make_static_pull_truss_model()
    num_dofs = model.mesh.num_dofs

    with pytest.raises(ValueError, match="U must be one-dimensional"):
        ModelResult(
            model,
            model.steps[0],
            np.zeros((num_dofs, 1)),
            np.zeros(num_dofs),
        )
    with pytest.raises(ValueError, match=rf"U must have length {num_dofs}"):
        ModelResult(
            model,
            model.steps[0],
            np.zeros(num_dofs - 1),
            np.zeros(num_dofs),
        )
    with pytest.raises(ValueError, match="reactions must contain only finite"):
        ModelResult(
            model,
            model.steps[0],
            np.zeros(num_dofs),
            np.full(num_dofs, np.nan),
        )


@pytest.mark.parametrize("failure", ["nonfinite", "asymmetric"])
def test_assembly_rejects_invalid_element_stiffness(monkeypatch, failure):
    mesh = make_truss_stiffness_mesh()
    Ke = np.eye(6)
    if failure == "nonfinite":
        Ke[0, 0] = np.nan
        message = "contains non-finite values"
    else:
        Ke[0, 1] = 1.0
        message = "stiffness is not symmetric"

    class Kernel:
        def stiffness(self, mesh, elem, node_lookup=None):
            return Ke

    monkeypatch.setattr(
        stiffness_module,
        "get_element_kernel",
        lambda element_type: Kernel(),
    )

    with pytest.raises(ValueError, match=message):
        assemble_global_stiffness_sparse(mesh)
