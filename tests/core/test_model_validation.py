from copy import deepcopy

import pytest

from fem.assemble import assemble_global_stiffness_sparse
from fem.core import validate_mesh, validate_model, validate_model_structure
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import AnalysisStep, ElementSet, MaterialDefinition, SectionAssignment
from tests.helpers.mesh_builders import make_truss_stiffness_mesh
from tests.helpers.model_builders import make_static_pull_truss_model, make_truss_workflow_model


def test_node_table_changes_require_rebuild_before_connectivity_uses_new_dof_order():
    mesh = Mesh3D(
        nodes=[Node3D(20, 0.0, 0.0, 0.0), Node3D(40, 2.0, 0.0, 0.0)],
        elements=[Element3D(1, [40, 20], "Truss2", {"E": 100.0, "area": 2.0})],
    )
    mesh.nodes.insert(1, Node3D(10, -1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match=r"DofMap.*rebuild_dof_map"):
        validate_mesh(mesh)

    mesh.rebuild_dof_map()
    validate_mesh(mesh)

    assert mesh.node_ids == [10, 20, 40]
    assert mesh.num_dofs == 9
    assert mesh.global_dof(20, 0) == 3
    assert list(mesh.element_dofs(mesh.elements[0])) == [6, 7, 8, 3, 4, 5]


@pytest.mark.parametrize(
    ("invalid_mesh", "message"),
    [("no-nodes", "at least one node"),
     ("no-elements", "at least one element"),
     ("duplicate-element", "element ids.*unique")],
)
def test_assembly_rejects_invalid_mesh_structure(invalid_mesh, message):
    mesh = make_truss_stiffness_mesh()
    if invalid_mesh == "no-nodes":
        mesh = Mesh3D(nodes=[], elements=[])
    elif invalid_mesh == "no-elements":
        mesh.elements.clear()
    else:
        mesh.elements.append(deepcopy(mesh.elements[0]))

    with pytest.raises(ValueError, match=message):
        assemble_global_stiffness_sparse(mesh)


@pytest.mark.parametrize("entity", ["node", "element"])
def test_mesh_validation_rejects_duplicate_entity_ids_after_edit(entity):
    mesh = make_truss_stiffness_mesh()
    if entity == "node":
        mesh.nodes.append(Node3D(1, 3.0, 0.0, 0.0))
    else:
        mesh.elements.append(deepcopy(mesh.elements[0]))

    with pytest.raises(ValueError, match=rf"{entity} ids.*unique"):
        validate_mesh(mesh)


@pytest.mark.parametrize(
    ("node_ids", "error", "message"),
    [([1, 999], KeyError, "element 1.*missing node 999"),
     ([1, 1], ValueError, "element 1.*node_ids.*unique")],
    ids=["missing-node", "repeated-node"],
)
def test_mesh_validation_rejects_invalid_element_connectivity(node_ids, error, message):
    mesh = make_truss_stiffness_mesh()
    mesh.elements[0].node_ids = node_ids

    with pytest.raises(error, match=message):
        validate_mesh(mesh)


@pytest.mark.parametrize(
    ("coordinate", "error", "message"),
    [(float("nan"), ValueError, "coordinate x.*finite"),
     ("1.0", TypeError, "coordinate x.*real number"),
     (True, TypeError, "coordinate x.*real number")],
    ids=["nonfinite", "numeric-string", "boolean"],
)
def test_mesh_validation_rejects_nonfinite_or_nonreal_coordinates(
    coordinate, error, message,
):
    mesh = make_truss_stiffness_mesh()
    mesh.nodes[0].x = coordinate

    with pytest.raises(error, match=message):
        validate_mesh(mesh)


@pytest.mark.parametrize(
    ("reference", "message"),
    [("set-member", "element set bad.*missing element 999"),
     ("section-material", "material missing.*not defined"),
     ("section-set", "element set missing.*not defined")],
)
def test_model_structure_rejects_missing_set_and_section_references(reference, message):
    model = make_truss_workflow_model()
    if reference == "set-member":
        model.element_sets["bad"] = ElementSet("bad", (999,))
    elif reference == "section-material":
        model.sections.append(SectionAssignment("bar", "missing"))
    else:
        model.materials["steel"] = MaterialDefinition("steel", {"E": 100.0, "nu": 0.3})
        model.sections.append(SectionAssignment("missing", "steel"))

    with pytest.raises(KeyError, match=message):
        validate_model_structure(model)


def test_model_validation_rejects_case_insensitive_duplicate_step_names():
    model = make_static_pull_truss_model()
    model.steps.append(AnalysisStep("PULL"))

    with pytest.raises(ValueError, match="step names.*unique ignoring case"):
        validate_model(model)
