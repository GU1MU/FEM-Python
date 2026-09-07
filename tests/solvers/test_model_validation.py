from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fem import materials
from fem.assemble import assemble_global_stiffness_sparse
from fem.core import (
    validate_mesh,
    validate_model,
)
from fem.core.mesh import Mesh3D, Node3D
from fem.core.model import (
    AnalysisStep,
    ElementSet,
    SectionAssignment,
)
from fem.core.result import ModelResult
from fem.solvers import static_linear
from tests.helpers.mesh_builders import make_truss_stiffness_mesh
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
