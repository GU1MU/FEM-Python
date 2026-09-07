from copy import deepcopy

import numpy as np
import pytest

from fem.boundary.condition import BoundaryCondition
from fem.boundary.loads import build_load_vector
from fem.boundary.step import boundary_for_step, effective_step_boundaries
from fem.core.model import (
    AnalysisStep,
    BodyForce,
    DisplacementConstraint,
    Edge,
    EdgeLoad,
    ElementEdge,
    ElementFace,
    ElementSet,
    FEMModel,
    GravityLoad,
    NodalLoad,
    Surface,
    SurfaceLoad,
)
from tests.helpers.mesh_builders import (
    make_beam_stiffness_mesh,
    make_hex20_stiffness_mesh,
    make_mixed_hex8_tet4_mesh,
    make_quad4_boundary_mesh,
    make_selection_hex_mesh,
    make_tet4_stiffness_mesh,
    make_truss_stiffness_mesh,
)
from tests.helpers.model_builders import make_two_step_static_pull_truss_model


def test_effective_step_boundaries_accumulate_all_preceding_steps() -> None:
    model = make_two_step_static_pull_truss_model()
    previous = DisplacementConstraint("TIP", 1, 1, 0.125)
    current = DisplacementConstraint("TIP", 1, 1, 0.25)
    model.steps[1].boundaries = (previous,)
    model.steps[2].boundaries = (current,)

    initial = effective_step_boundaries(model, "Initial")
    inherited = effective_step_boundaries(model, "pull2")

    assert initial == tuple(model.steps[0].boundaries)
    assert inherited == (*model.steps[0].boundaries, previous, current)

    resolved = boundary_for_step(model, "pull2")
    tip_ux = model.mesh.global_dof(2, 0)
    assert resolved.prescribed_displacements[tip_ux] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("target_kind", "expected_node_ids"),
    (
        ("edge", (5, 6)),
        ("surface", (1, 2, 3, 4)),
    ),
)
def test_displacement_constraint_expands_edge_and_surface_nodes(
    target_kind,
    expected_node_ids,
):
    mesh = make_selection_hex_mesh()
    model = FEMModel(
        mesh=mesh,
        edges={
            "FIXED": Edge(
                "FIXED",
                [ElementEdge(1, 4, (5, 6))],
            )
        },
        surfaces={
            "FIXED": Surface(
                "FIXED",
                [ElementFace(1, 0, (1, 2, 3, 4))],
            )
        },
        steps=[
            AnalysisStep(
                "load",
                boundaries=[
                    DisplacementConstraint(
                        "FIXED",
                        1,
                        1,
                        target_kind=target_kind,
                    )
                ],
            )
        ],
    )
    authoring_before = deepcopy(
        (
            model.node_sets,
            model.edges,
            model.surfaces,
            tuple(model.steps[0].boundaries),
        )
    )

    resolved = boundary_for_step(model, "load")

    assert set(resolved.prescribed_displacements) == {
        mesh.global_dof(node_id, 0)
        for node_id in expected_node_ids
    }
    assert (
        model.node_sets,
        model.edges,
        model.surfaces,
        tuple(model.steps[0].boundaries),
    ) == authoring_before
    assert model.node_sets == {}
    assert model.steps[0].boundaries[0].target == "FIXED"
    assert model.steps[0].boundaries[0].target_kind == target_kind


@pytest.mark.parametrize(
    ("surface_load", "message"),
    [
        (SurfaceLoad("LOADED", load_type="pressure"), "requires a magnitude"),
        (
            SurfaceLoad("LOADED", magnitude=2.0, load_type="shear_traction"),
            "requires a direction vector",
        ),
    ],
    ids=["pressure", "shear-traction"],
)
def test_surface_load_rejects_missing_required_parameters(surface_load, message):
    mesh = make_selection_hex_mesh()
    surface = Surface("LOADED", (ElementFace(1, 0, (1, 2, 3, 4)),))
    step = AnalysisStep("load", surface_loads=(surface_load,))
    model = FEMModel(mesh=mesh, surfaces={"LOADED": surface}, steps=[step])

    with pytest.raises(ValueError, match=message):
        boundary_for_step(model, step)


def test_edge_pressure_rejects_missing_magnitude():
    mesh = make_quad4_boundary_mesh()
    edge = Edge("LOADED", (ElementEdge(1, 1, (2, 3)),))
    step = AnalysisStep(
        "load",
        edge_loads=(EdgeLoad("LOADED", load_type="pressure"),),
    )
    model = FEMModel(mesh=mesh, edges={"LOADED": edge}, steps=[step])

    with pytest.raises(ValueError, match="requires a magnitude"):
        boundary_for_step(model, step)


@pytest.mark.parametrize(
    "builder",
    [make_truss_stiffness_mesh, make_beam_stiffness_mesh],
    ids=["truss2", "beam2"],
)
def test_line_element_gravity_has_density_times_volume_resultant(builder):
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
    area = 0.5  # Both builders use area 0.5 (beam rectangle 1 by 0.5).
    expected_y = -2.0 * 3.0 * area * length
    y_dofs = [mesh.global_dof(node.id, 1) for node in mesh.nodes]

    assert F.shape == (mesh.num_dofs,)
    assert np.all(np.isfinite(F))
    assert float(F[y_dofs].sum()) == pytest.approx(expected_y)


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


@pytest.mark.parametrize(
    "builder", [make_quad4_boundary_mesh, make_tet4_stiffness_mesh], ids=["2d", "3d"]
)
@pytest.mark.parametrize("nodal_only", [False, True], ids=["empty", "nodal-only"])
def test_empty_and_repeated_nodal_loads_assemble(builder, nodal_only):
    mesh = builder()
    component = mesh.dofs_per_node - 1
    step = AnalysisStep("load", cloads=(
        NodalLoad(1, component + 1, -2.0),
        NodalLoad(1, component + 1, -3.0),
    ) if nodal_only else ())
    model = FEMModel(mesh=mesh, steps=[step])

    force = build_load_vector(mesh, boundary_for_step(model, step))

    expected = np.zeros(mesh.num_dofs)
    if nodal_only:
        expected[mesh.global_dof(1, component)] = -5.0
    np.testing.assert_allclose(force, expected)


def test_2d_body_force_and_edge_traction_add_at_shared_nodes():
    mesh = make_quad4_boundary_mesh()
    bc = BoundaryCondition()
    bc.add_body_force_element(1, 4.0, -5.0)
    bc.add_edge_traction(1, 0, 7.0, -11.0)

    force = build_load_vector(mesh, bc)

    # Rectangle volume = 2 * 1 * 2; bottom edge measure = 2 * 2.
    np.testing.assert_allclose(force.reshape(-1, 2), [
        (18.0, -27.0), (18.0, -27.0), (4.0, -5.0), (4.0, -5.0),
    ])


def test_tetrahedron_body_and_face_loads_add_at_shared_nodes():
    mesh = make_tet4_stiffness_mesh()
    bc = BoundaryCondition()
    bc.add_body_force_element(1, 0.0, 0.0, -6.0)
    bc.add_surface_traction(1, 3, 0.0, 0.0, -2.0)

    force = build_load_vector(mesh, bc).reshape(-1, 3)

    # Volume 1/6 shared by four nodes, area 1/2 shared by bottom three.
    expected = np.zeros((4, 3))
    expected[:, 2] = -0.25
    expected[:3, 2] -= 1.0 / 3.0
    np.testing.assert_allclose(force, expected, atol=1e-14)


@pytest.mark.parametrize("load, nodal_vector", [
    (EdgeLoad("RIGHT", (7.0, -11.0), load_type="traction"), (7.0, -11.0)),
    (EdgeLoad("RIGHT", magnitude=2.0, load_type="pressure"), (-2.0, 0.0)),
], ids=["traction", "inward-pressure"])
def test_2d_edge_loads_resolve_and_assemble(load, nodal_vector):
    mesh = make_quad4_boundary_mesh()
    model = FEMModel(
        mesh=mesh,
        edges={"RIGHT": Edge("RIGHT", [ElementEdge(1, 1, (2, 3))])},
        steps=[AnalysisStep("load", edge_loads=[load])],
    )

    force = build_load_vector(mesh, boundary_for_step(model, "load"))

    expected = np.zeros((4, 2))
    # Right edge length 1, thickness 2, two equal nodal shares.
    expected[1:3] = nodal_vector
    np.testing.assert_allclose(force.reshape(-1, 2), expected, atol=1e-14)


def test_pressure_on_adjacent_faces_points_inward_and_accumulates():
    mesh = make_selection_hex_mesh()
    model = FEMModel(
        mesh=mesh,
        surfaces={"LOADED": Surface("LOADED", [
            ElementFace(1, 0, (1, 2, 3, 4)),
            ElementFace(1, 5, (2, 3, 7, 6)),
        ])},
        steps=[AnalysisStep("load", surface_loads=[
            SurfaceLoad("LOADED", magnitude=2.0, load_type="pressure")
        ])],
    )

    force = build_load_vector(mesh, boundary_for_step(model, "load"))

    # Bottom area 6 -> +z force 12; right area 12 -> -x force 24.
    expected = np.zeros((8, 3))
    expected[:4, 2] = 3.0
    expected[[1, 2, 5, 6], 0] = -6.0
    np.testing.assert_allclose(force.reshape(-1, 3), expected, atol=1e-14)


def test_mixed_solid_surface_tractions_scatter_to_shared_nodes():
    mesh = make_mixed_hex8_tet4_mesh()
    bc = BoundaryCondition()
    bc.add_surface_traction(1, 1, 0.0, 0.0, 1.0)
    bc.add_surface_traction(2, 0, 1.0, 0.0, 0.0)

    force = build_load_vector(mesh, bc)

    expected = np.zeros((9, 3))
    expected[4:8, 2] = 0.25  # Unit cube top area = 1.
    expected[[8, 2, 5], 0] = np.sqrt(3.0) / 6.0  # Tet inclined area = sqrt(3)/2.
    np.testing.assert_allclose(force.reshape(-1, 3), expected, atol=1e-14)


@pytest.mark.parametrize("with_gravity", [False, True], ids=["body-only", "body-and-gravity"])
def test_body_force_ignores_density_and_gravity_scales_by_density(with_gravity):
    mesh = make_mixed_hex8_tet4_mesh()
    mesh.elements[0].props["rho"] = 2.0
    mesh.elements[1].props["rho"] = 3.0
    step = AnalysisStep(
        "load", body_loads=(BodyForce("all", (1.0, -2.0, 3.0)),),
        gravity_loads=(GravityLoad((0.0, 0.0, -2.0)),) if with_gravity else (),
    )
    model = FEMModel(mesh=mesh, element_sets={"all": ElementSet("all", (1, 2))}, steps=[step])

    force = build_load_vector(mesh, boundary_for_step(model, step))

    # Cube volume 1 and tetrahedron volume 1/6; shared nodes sum both shares.
    cube_force = np.array((1.0, -2.0, -1.0 if with_gravity else 3.0))
    tet_force = np.array((1.0, -2.0, -3.0 if with_gravity else 3.0))
    expected = np.zeros((9, 3))
    expected[:8] = cube_force / 8.0
    expected[[1, 8, 2, 5]] += tet_force / 24.0
    np.testing.assert_allclose(force.reshape(-1, 3), expected, atol=1e-14)


def test_global_id_and_set_gravity_accumulate_with_mixed_element_density():
    mesh = make_mixed_hex8_tet4_mesh()
    mesh.elements[0].props["rho"] = 2.0
    mesh.elements[1].props["rho"] = 3.0
    step = AnalysisStep("gravity", gravity_loads=(
        GravityLoad((0.0, -1.0, 0.0)),
        GravityLoad((1.0, 0.0, -2.0)),
        GravityLoad((0.0, 0.0, -3.0), np.int64(1)),
        GravityLoad((4.0, 0.0, 0.0), "all"),
    ))
    model = FEMModel(mesh=mesh, element_sets={"all": ElementSet("all", (1, 2))}, steps=[step])

    force = build_load_vector(mesh, boundary_for_step(model, step))

    expected = np.zeros((9, 3))
    expected[:8] = np.array((10.0, -2.0, -10.0)) / 8.0
    expected[[1, 8, 2, 5]] += np.array((15.0, -3.0, -6.0)) / 24.0
    np.testing.assert_allclose(force.reshape(-1, 3), expected, atol=1e-14)


def test_hex20_face_traction_has_consistent_corner_and_midside_forces():
    mesh = make_hex20_stiffness_mesh()
    bc = BoundaryCondition()
    bc.add_surface_traction(1, 1, 0.0, 0.0, -5.0)

    force = build_load_vector(mesh, bc).reshape(-1, 3)

    # A unit-area quadratic face has weights -1/12 at corners and 1/3 at midsides.
    expected = np.zeros((20, 3))
    expected[4:8, 2] = 5.0 / 12.0
    expected[12:16, 2] = -5.0 / 3.0
    np.testing.assert_allclose(force, expected, atol=1e-14)
    xyz = np.array([(node.x, node.y, node.z) for node in mesh.nodes])
    np.testing.assert_allclose(force.sum(axis=0), (0.0, 0.0, -5.0), atol=1e-14)
    np.testing.assert_allclose(np.cross(xyz, force).sum(axis=0), (-2.5, 2.5, 0.0), atol=1e-14)
