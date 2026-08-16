import numpy as np
import pytest

from fem import materials, steps
from fem.core.model import ElementSet, FEMModel, MaterialDefinition, NodeSet
from fem.solvers import static_linear
from tests.helpers.mesh_builders import (
    make_hex20_stiffness_mesh,
    make_mixed_hex8_tet4_mesh,
    make_mixed_tri6_quad8_mesh,
)
from tests.helpers.model_builders import make_truss_workflow_model


def test_core_model_supports_hand_written_mesh_model_solve_result_flow():
    model = make_truss_workflow_model(name="manual_bar", loaded_set_name="tip")
    mesh = model.mesh

    material = materials.linear_elastic.material("steel", E=100.0, nu=0.3)
    materials.add(model, material)
    section = materials.assign(model, "steel", "bar", area=2.0)
    step = steps.static("pull")
    steps.displacement(step, "fixed", components=(1, 2, 3))
    steps.displacement(step, 2, components=(2, 3))
    steps.nodal_load(step, "tip", component=1, value=100.0)
    steps.add(model, step)

    result = static_linear.solve(model, "pull")

    assert material == MaterialDefinition("steel", {"E": 100.0, "nu": 0.3})
    assert model.element_sets["bar"] == ElementSet("bar", (1,))
    assert section.element_set == "bar"
    assert section.properties["area"] == 2.0
    assert model.node_sets["fixed"] == NodeSet("fixed", (1,))
    assert model.node_sets["tip"] == NodeSet("tip", (2,))
    assert step.name == "pull"
    assert len(step.boundaries) == 2
    assert len(step.cloads) == 1
    assert mesh.elements[0].props["E"] == 100.0
    assert mesh.elements[0].props["area"] == 2.0
    assert result.U[mesh.global_dof(2, 0)] == pytest.approx(0.5)
    assert result.reactions[mesh.global_dof(1, 0)] == pytest.approx(-100.0)


def test_gravity_and_unified_solve_keep_static_steps_independent():
    model = make_truss_workflow_model(
        name="gravity_bar",
        loaded_set_name="tip",
    )
    mesh = model.mesh
    materials.add(
        model,
        materials.linear_elastic.material(
            "steel",
            E=100.0,
            nu=0.3,
            rho=2.0,
        ),
    )
    materials.assign(model, "steel", "bar", area=2.0)

    initial = steps.static("Initial")
    steps.displacement(initial, "fixed", components=(1, 2, 3))
    steps.displacement(initial, "tip", components=(2, 3))
    gravity_case = steps.static("gravity_case")
    steps.gravity(gravity_case, (3.0, 0.0, 0.0))
    steps.nodal_load(gravity_case, "tip", component=1, value=5.0)
    nodal_case = steps.static("nodal_case")
    steps.nodal_load(nodal_case, "tip", component=1, value=5.0)
    for step in (initial, gravity_case, nodal_case):
        steps.add(model, step)

    results = static_linear.solve(model, steps="all")
    gravity_result, nodal_result = results.results

    assert tuple(result.step.name for result in results.results) == (
        "gravity_case",
        "nodal_case",
    )
    assert gravity_result.reactions[mesh.global_dof(1, 0)] == pytest.approx(-17.0)
    assert nodal_result.reactions[mesh.global_dof(1, 0)] == pytest.approx(-5.0)
    assert nodal_result.U[mesh.global_dof(2, 0)] == pytest.approx(0.025)


def test_mixed_solid_model_assigns_materials_by_element_set_and_solves():
    mesh = make_mixed_hex8_tet4_mesh()
    model = FEMModel(mesh=mesh, name="mixed_hex8_tet4")
    model.element_sets["hexes"] = ElementSet("hexes", (1,))
    model.element_sets["tets"] = ElementSet("tets", (2,))
    model.node_sets["fixed"] = NodeSet("fixed", (1, 4, 5, 8))
    model.node_sets["tip"] = NodeSet("tip", (9,))

    steel = materials.linear_elastic.material("steel", E=210.0, nu=0.3)
    aluminum = materials.linear_elastic.material("aluminum", E=120.0, nu=0.25)
    materials.add(model, steel)
    materials.add(model, aluminum)
    materials.assign(model, "steel", "hexes")
    materials.assign(model, "aluminum", "tets")

    step = steps.static("pull")
    steps.displacement(step, "fixed", components=(1, 2, 3))
    steps.nodal_load(step, "tip", component=1, value=1.0)
    steps.add(model, step)

    result = static_linear.solve(model, "pull")

    assert mesh.elements[0].type == "Hex8"
    assert mesh.elements[1].type == "Tet4"
    assert mesh.elements[0].props["material"] == "steel"
    assert mesh.elements[1].props["material"] == "aluminum"
    assert np.all(np.isfinite(result.U))
    assert abs(float(result.U[mesh.global_dof(9, 0)])) > 0.0


def test_fully_constrained_hex20_model_solves_with_nonzero_displacement():
    mesh = make_hex20_stiffness_mesh()
    model = FEMModel(mesh=mesh, name="hex20_static")
    model.node_sets["fixed"] = NodeSet("fixed", tuple(range(1, 20)))
    model.node_sets["loaded"] = NodeSet("loaded", (20,))

    step = steps.static("pull")
    steps.displacement(step, "fixed", components=(1, 2, 3))
    steps.displacement(step, "loaded", components=(2, 3))
    steps.nodal_load(step, "loaded", component=1, value=1.0e-3)
    steps.add(model, step)

    result = static_linear.solve(model, "pull")
    loaded_displacement = result.U[mesh.global_dof(20, 0)]

    assert np.all(np.isfinite(result.U))
    assert np.isfinite(loaded_displacement)
    assert abs(float(loaded_displacement)) > 0.0


def test_mixed_quadratic_plane_model_assigns_materials_by_element_set_and_solves():
    mesh = make_mixed_tri6_quad8_mesh()
    model = FEMModel(mesh=mesh, name="mixed_tri6_quad8")
    model.element_sets["triangles"] = ElementSet("triangles", (1,))
    model.element_sets["quads"] = ElementSet("quads", (2,))
    model.node_sets["fixed"] = NodeSet("fixed", (1, 3, 6, 7, 10, 14))
    model.node_sets["tip"] = NodeSet("tip", (8,))

    steel = materials.linear_elastic.material("steel", E=210.0, nu=0.3)
    aluminum = materials.linear_elastic.material("aluminum", E=120.0, nu=0.25)
    materials.add(model, steel)
    materials.add(model, aluminum)
    materials.assign(model, "steel", "triangles")
    materials.assign(model, "aluminum", "quads")

    step = steps.static("pull")
    steps.displacement(step, "fixed", components=(1, 2))
    steps.nodal_load(step, "tip", component=1, value=1.0)
    steps.add(model, step)

    result = static_linear.solve(model, "pull")

    assert mesh.elements[0].type == "Tri6"
    assert mesh.elements[1].type == "Quad8"
    assert mesh.elements[0].props["material"] == "steel"
    assert mesh.elements[1].props["material"] == "aluminum"
    assert np.all(np.isfinite(result.U))
    assert abs(float(result.U[mesh.global_dof(8, 0)])) > 0.0
