import numpy as np
import pytest

from fem import boundary
from fem.boundary.condition import BoundaryCondition
from fem.boundary.loads import build_load_vector
from fem.boundary.step import boundary_for_step
from fem.core.model import AnalysisStep, Edge, EdgeLoad, ElementEdge, ElementFace, FEMModel, Surface, SurfaceLoad
from fem.elements import get_element_kernel
from tests.helpers.mesh_builders import (
    make_beam_stiffness_mesh,
    make_hex20_stiffness_mesh,
    make_mixed_hex8_tet4_mesh,
    make_quad4_boundary_mesh,
    make_selection_hex_mesh,
    make_tet4_stiffness_mesh,
    make_truss_stiffness_mesh,
)


def test_boundary_load_vector_matches_kernel_dispatch():
    mesh = make_quad4_boundary_mesh()
    elem = mesh.elements[0]
    bc = boundary.condition.BoundaryCondition()
    bc.add_body_force_element(elem.id, 4.0, -5.0)
    bc.add_edge_traction(elem.id, 0, 7.0, -11.0)

    F = boundary.loads.build_load_vector(mesh, bc)
    kernel = get_element_kernel(elem.type)
    expected = kernel.body_force(mesh, elem, (4.0, -5.0))
    expected += kernel.edge_traction(mesh, elem, 0, (7.0, -11.0))

    assert np.allclose(F, expected)

    mesh3d = make_tet4_stiffness_mesh()
    elem3d = mesh3d.elements[0]
    bc3d = boundary.condition.BoundaryCondition()
    bc3d.add_body_force_element(elem3d.id, 0.0, 0.0, -6.0)
    bc3d.add_surface_traction(elem3d.id, 3, 0.0, 0.0, -2.0)

    F3d = boundary.loads.build_load_vector(mesh3d, bc3d)
    kernel3d = get_element_kernel(elem3d.type)
    expected3d = kernel3d.body_force(mesh3d, elem3d, (0.0, 0.0, -6.0))
    expected3d += kernel3d.face_traction(mesh3d, elem3d, 3, (0.0, 0.0, -2.0))

    assert np.allclose(F3d, expected3d)


def test_boundary_package_exposes_explicit_modules_only():
    assert hasattr(boundary, "body")
    assert hasattr(boundary, "condition")
    assert hasattr(boundary, "loads")
    assert hasattr(boundary, "nodal")
    assert hasattr(boundary, "constraints")
    assert hasattr(boundary, "traction")
    assert callable(boundary.loads.build_load_vector)
    assert not hasattr(boundary.traction, "add_forces")
    assert not hasattr(boundary, "BoundaryCondition2D")
    assert not hasattr(boundary, "BoundaryCondition3D")
    assert not hasattr(boundary, "build_load_vector_3d")


def test_boundary_condition_preserves_positional_order():
    bc = BoundaryCondition({}, {}, [], [], (0.0, -9.81))

    assert bc.surface_tractions == []
    assert bc.gravity == (0.0, -9.81)
    assert bc.edge_tractions == []


def test_3d_nodal_forces_accumulate_like_2d():
    mesh = make_tet4_stiffness_mesh()
    bc = boundary.condition.BoundaryCondition()
    bc.add_nodal_force(node_id=1, component=2, value=-2.0, mesh=mesh)
    bc.add_nodal_force(node_id=1, component=2, value=-3.0, mesh=mesh)

    F = boundary.loads.build_load_vector(mesh, bc)

    assert F[mesh.global_dof(1, 2)] == pytest.approx(-5.0)


def test_mixed_solid_surface_tractions_dispatch_by_element_type():
    mesh = make_mixed_hex8_tet4_mesh()
    bc = BoundaryCondition()
    bc.add_surface_traction(1, 1, 0.0, 0.0, 1.0)
    bc.add_surface_traction(2, 0, 1.0, 0.0, 0.0)

    F = build_load_vector(mesh, bc)

    assert F.shape == (mesh.num_dofs,)
    assert float(np.linalg.norm(F)) > 0.0


def test_mixed_solid_body_forces_and_gravity_dispatch_by_element_type():
    mesh = make_mixed_hex8_tet4_mesh()
    mesh.elements[0].props["rho"] = 2.0
    mesh.elements[1].props["rho"] = 3.0
    bc = BoundaryCondition()
    bc.add_body_force_element(1, 0.0, 0.0, -1.0)
    bc.add_body_force_element(2, 1.0, 0.0, 0.0)
    bc.set_gravity(0.0, 0.0, -9.81)

    F = build_load_vector(mesh, bc)

    assert F.shape == (mesh.num_dofs,)
    assert np.all(np.isfinite(F))
    assert float(np.linalg.norm(F)) > 0.0


@pytest.mark.parametrize(
    "builder",
    [make_truss_stiffness_mesh, make_beam_stiffness_mesh],
    ids=["truss2", "beam2"],
)
def test_line_element_gravity_dispatches_through_body_force(builder):
    mesh = builder()
    elem = mesh.elements[0]
    elem.props["rho"] = 3.0
    bc = BoundaryCondition()
    bc.set_gravity(0.0, -2.0, 0.0)

    F = build_load_vector(mesh, bc)
    node_lookup = {node.id: node for node in mesh.nodes}
    ni = node_lookup[elem.node_ids[0]]
    nj = node_lookup[elem.node_ids[1]]
    length = float(np.hypot(nj.x - ni.x, nj.y - ni.y))
    expected_y = -2.0 * 3.0 * float(elem.props["area"]) * length
    y_dofs = [mesh.global_dof(node.id, 1) for node in mesh.nodes]

    assert F.shape == (mesh.num_dofs,)
    assert np.all(np.isfinite(F))
    assert float(F[y_dofs].sum()) == pytest.approx(expected_y)


def test_boundary_step_builds_2d_edge_traction():
    mesh = make_quad4_boundary_mesh()
    model = FEMModel(
        mesh=mesh,
        edges={"LOAD_EDGE": Edge("LOAD_EDGE", [ElementEdge(1, 1, (2, 3))])},
        steps=[
            AnalysisStep(
                "load",
                edge_loads=[EdgeLoad("LOAD_EDGE", (7.0, -11.0), load_type="traction")],
            )
        ],
    )

    bc = boundary_for_step(model, "load")

    assert len(bc.edge_tractions) == 1
    assert bc.edge_tractions[0].elem_id == 1
    assert bc.edge_tractions[0].local_index == 1
    assert bc.edge_tractions[0].vector == (7.0, -11.0)


def test_boundary_load_vector_assembles_2d_edge_traction():
    mesh = make_quad4_boundary_mesh()
    elem = mesh.elements[0]
    bc = BoundaryCondition()
    bc.add_edge_traction(elem.id, 0, 7.0, -11.0)

    F = build_load_vector(mesh, bc)
    kernel = get_element_kernel(elem.type)
    expected = kernel.edge_traction(mesh, elem, 0, (7.0, -11.0))

    assert np.allclose(F, expected)


def test_boundary_step_builds_2d_edge_pressure():
    mesh = make_quad4_boundary_mesh()
    model = FEMModel(
        mesh=mesh,
        edges={"RIGHT": Edge("RIGHT", [ElementEdge(1, 1, (2, 3))])},
        steps=[AnalysisStep("load", edge_loads=[EdgeLoad("RIGHT", magnitude=2.0, load_type="pressure")])],
    )

    bc = boundary_for_step(model, "load")

    assert len(bc.edge_tractions) == 1
    assert bc.edge_tractions[0].elem_id == 1
    assert bc.edge_tractions[0].local_index == 1
    assert np.allclose(bc.edge_tractions[0].vector, (-2.0, 0.0))


def test_boundary_step_rejects_2d_surface_loads():
    mesh = make_quad4_boundary_mesh()
    model = FEMModel(
        mesh=mesh,
        surfaces={"RIGHT": Surface("RIGHT", [ElementFace(1, 1, (2, 3))])},
        steps=[
            AnalysisStep(
                "load",
                surface_loads=[SurfaceLoad("RIGHT", magnitude=2.0, load_type="pressure")],
            )
        ],
    )

    with pytest.raises(ValueError, match="2D surface loads are not supported"):
        boundary_for_step(model, "load")


def test_boundary_step_reports_3d_edge_loads_not_supported():
    mesh = make_selection_hex_mesh()
    model = FEMModel(
        mesh=mesh,
        edges={"TOP": Edge("TOP", [ElementEdge(1, 4, (5, 6))])},
        steps=[AnalysisStep("load", edge_loads=[EdgeLoad("TOP", (0.0, 0.0, -1.0))])],
    )

    with pytest.raises(NotImplementedError, match="3D edge loads are not supported"):
        boundary_for_step(model, "load")


def test_3d_edge_traction_assembly_reports_not_supported():
    mesh = make_selection_hex_mesh()
    bc = BoundaryCondition()
    bc.add_edge_traction(1, 4, 0.0, 0.0, -1.0)

    with pytest.raises(NotImplementedError, match="3D edge loads are not supported"):
        build_load_vector(mesh, bc)


def test_3d_surface_traction_assembly_still_uses_face_traction():
    mesh = make_tet4_stiffness_mesh()
    elem = mesh.elements[0]
    bc = BoundaryCondition()
    bc.add_surface_traction(elem.id, 3, 0.0, 0.0, -2.0)

    F = build_load_vector(mesh, bc)
    kernel = get_element_kernel(elem.type)
    expected = kernel.face_traction(mesh, elem, 3, (0.0, 0.0, -2.0))

    assert np.allclose(F, expected)


def test_hex20_surface_traction_dispatches_through_element_kernel():
    mesh = make_hex20_stiffness_mesh()
    elem = mesh.elements[0]
    bc = BoundaryCondition()
    bc.add_surface_traction(elem.id, 1, 0.0, 0.0, -5.0)

    F = build_load_vector(mesh, bc)
    expected = get_element_kernel(elem.type).face_traction(
        mesh,
        elem,
        1,
        (0.0, 0.0, -5.0),
    )

    assert np.allclose(F, expected)
