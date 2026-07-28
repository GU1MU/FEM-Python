from copy import deepcopy

import numpy as np
import pytest

from fem import boundary
from fem.boundary.condition import BoundaryCondition
from fem.boundary.loads import build_load_vector
from fem.boundary.step import boundary_for_step
from fem.boundary import step as boundary_step
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
from fem.elements import get_element_kernel
from fem.elements.beam_section import parse_beam2_section
from tests.helpers.mesh_builders import (
    make_beam_stiffness_mesh,
    make_hex20_stiffness_mesh,
    make_mixed_hex8_tet4_mesh,
    make_quad4_boundary_mesh,
    make_selection_hex_mesh,
    make_tet4_stiffness_mesh,
    make_truss_stiffness_mesh,
)


class _IterationCountingList(list):
    def __init__(self, values):
        super().__init__(values)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


def _track_mesh_iterations(mesh):
    nodes = _IterationCountingList(mesh.nodes)
    elements = _IterationCountingList(mesh.elements)
    mesh.nodes = nodes
    mesh.elements = elements
    return nodes, elements


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


def test_3d_nodal_forces_accumulate_like_2d():
    mesh = make_tet4_stiffness_mesh()
    bc = boundary.condition.BoundaryCondition()
    bc.add_nodal_force(node_id=1, component=2, value=-2.0, mesh=mesh)
    bc.add_nodal_force(node_id=1, component=2, value=-3.0, mesh=mesh)

    F = boundary.loads.build_load_vector(mesh, bc)

    assert F[mesh.global_dof(1, 2)] == pytest.approx(-5.0)


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


@pytest.mark.parametrize("nodal_only", [False, True], ids=["empty", "nodal-only"])
def test_boundary_step_skips_mesh_lookups_without_geometric_loads(nodal_only):
    mesh = make_tet4_stiffness_mesh()
    step = AnalysisStep(
        "load",
        cloads=(NodalLoad(1, 3, -5.0),) if nodal_only else (),
    )
    model = FEMModel(mesh=mesh, steps=[step])
    nodes, elements = _track_mesh_iterations(mesh)

    bc = boundary_for_step(model, step)

    assert len(bc.nodal_forces) == int(nodal_only)
    assert nodes.iterations == 0
    assert elements.iterations == 0


@pytest.mark.parametrize("nodal_only", [False, True], ids=["empty", "nodal-only"])
def test_load_vector_skips_mesh_lookups_without_distributed_loads(nodal_only):
    mesh = make_tet4_stiffness_mesh()
    bc = BoundaryCondition()
    if nodal_only:
        bc.add_nodal_force(1, 2, -5.0, mesh)
    nodes, elements = _track_mesh_iterations(mesh)

    F = build_load_vector(mesh, bc)

    assert np.count_nonzero(F) == int(nodal_only)
    assert nodes.iterations == 0
    assert elements.iterations == 0


def test_boundary_step_reuses_mesh_lookups_across_pressure_faces():
    mesh = make_selection_hex_mesh()
    surface = Surface(
        "LOADED",
        (
            ElementFace(1, 0, (1, 2, 3, 4)),
            ElementFace(1, 1, (2, 3, 7, 6)),
        ),
    )
    step = AnalysisStep(
        "load",
        surface_loads=(SurfaceLoad("LOADED", magnitude=2.0, load_type="pressure"),),
    )
    model = FEMModel(mesh=mesh, surfaces={"LOADED": surface}, steps=[step])
    nodes, elements = _track_mesh_iterations(mesh)

    bc = boundary_for_step(model, step)

    assert len(bc.surface_tractions) == 2
    assert nodes.iterations == 1
    assert elements.iterations == 1


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
def test_boundary_step_validates_surface_load_before_mesh_lookups(surface_load, message):
    mesh = make_selection_hex_mesh()
    surface = Surface("LOADED", (ElementFace(1, 0, (1, 2, 3, 4)),))
    step = AnalysisStep("load", surface_loads=(surface_load,))
    model = FEMModel(mesh=mesh, surfaces={"LOADED": surface}, steps=[step])
    nodes, elements = _track_mesh_iterations(mesh)

    with pytest.raises(ValueError, match=message):
        boundary_for_step(model, step)

    assert nodes.iterations == 0
    assert elements.iterations == 0


def test_boundary_step_validates_edge_pressure_before_mesh_lookups():
    mesh = make_quad4_boundary_mesh()
    edge = Edge("LOADED", (ElementEdge(1, 1, (2, 3)),))
    step = AnalysisStep(
        "load",
        edge_loads=(EdgeLoad("LOADED", load_type="pressure"),),
    )
    model = FEMModel(mesh=mesh, edges={"LOADED": edge}, steps=[step])
    nodes, elements = _track_mesh_iterations(mesh)

    with pytest.raises(ValueError, match="requires a magnitude"):
        boundary_for_step(model, step)

    assert nodes.iterations == 0
    assert elements.iterations == 0


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


def test_high_level_body_force_targets_elements_without_density_scaling():
    mesh = make_mixed_hex8_tet4_mesh()
    mesh.elements[0].props["rho"] = 20.0
    mesh.elements[1].props["rho"] = 30.0
    vector = (1.0, -2.0, 3.0)
    step = AnalysisStep(
        "body",
        body_loads=(BodyForce("all", vector),),
    )
    model = FEMModel(
        mesh=mesh,
        element_sets={"all": ElementSet("all", (1, 2))},
        steps=[step],
    )

    bc = boundary_for_step(model, step)
    F = build_load_vector(mesh, bc)
    expected = np.zeros(mesh.num_dofs)
    node_lookup = {node.id: node for node in mesh.nodes}
    for elem in mesh.elements:
        expected[mesh.element_dofs(elem)] += get_element_kernel(
            elem.type
        ).body_force(mesh, elem, vector, node_lookup)

    assert tuple(load.vector for load in bc.body_forces) == (vector, vector)
    assert np.allclose(F, expected)


def test_boundary_step_resolves_and_accumulates_global_and_targeted_gravity():
    mesh = make_mixed_hex8_tet4_mesh()
    mesh.elements[0].props["rho"] = 2.0
    mesh.elements[1].props["rho"] = 3.0
    step = AnalysisStep(
        "gravity",
        gravity_loads=(
            GravityLoad((0.0, -1.0, 0.0)),
            GravityLoad((1.0, 0.0, -2.0)),
            GravityLoad((0.0, 0.0, -3.0), np.int64(1)),
            GravityLoad((4.0, 0.0, 0.0), "all"),
        ),
    )
    model = FEMModel(
        mesh=mesh,
        element_sets={"all": ElementSet("all", (1, 2))},
        steps=[step],
    )

    bc = boundary_for_step(model, step)

    assert bc.gravity == (1.0, -1.0, -2.0)
    assert tuple(load.elem_id for load in bc.element_gravities) == (1, 1, 2)
    assert tuple(load.acceleration for load in bc.element_gravities) == (
        (0.0, 0.0, -3.0),
        (4.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
    )


def test_high_level_global_and_targeted_gravity_superpose_through_kernels():
    mesh = make_mixed_hex8_tet4_mesh()
    mesh.elements[0].props["rho"] = 2.0
    mesh.elements[1].props["rho"] = 3.0
    step = AnalysisStep(
        "gravity",
        gravity_loads=(
            GravityLoad((0.0, 0.0, -2.0)),
            GravityLoad((1.0, 0.0, 0.0), "tets"),
            GravityLoad((1.0, 0.0, 0.0), 2),
        ),
    )
    model = FEMModel(
        mesh=mesh,
        element_sets={"tets": ElementSet("tets", (2,))},
        steps=[step],
    )

    F = build_load_vector(mesh, boundary_for_step(model, step))
    expected = np.zeros(mesh.num_dofs)
    node_lookup = {node.id: node for node in mesh.nodes}
    contributions = {
        1: ((0.0, 0.0, -4.0),),
        2: ((0.0, 0.0, -6.0), (3.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
    }
    for elem in mesh.elements:
        kernel = get_element_kernel(elem.type)
        dofs = mesh.element_dofs(elem)
        for vector in contributions[elem.id]:
            expected[dofs] += kernel.body_force(mesh, elem, vector, node_lookup)

    assert np.allclose(F, expected)

    global_step = AnalysisStep(
        "global",
        gravity_loads=(GravityLoad((0.0, 0.0, -2.0)),),
    )
    global_model = FEMModel(mesh=mesh, steps=[global_step])
    direct = BoundaryCondition()
    direct.set_gravity(0.0, 0.0, -2.0)
    assert np.allclose(
        build_load_vector(mesh, boundary_for_step(global_model, global_step)),
        build_load_vector(mesh, direct),
    )


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
    area = (
        parse_beam2_section(elem.props).area
        if str(elem.type).casefold() == "beam2"
        else float(elem.props["area"])
    )
    expected_y = -2.0 * 3.0 * area * length
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


def test_pressure_faces_share_boundary_step_lookup_tables(monkeypatch):
    mesh = make_selection_hex_mesh()
    model = FEMModel(
        mesh=mesh,
        surfaces={"LOADED": Surface("LOADED", [
            ElementFace(1, 0, (1, 2, 3, 4)),
            ElementFace(1, 1, (5, 6, 7, 8)),
        ])},
        steps=[AnalysisStep(
            "load", surface_loads=[SurfaceLoad("LOADED", magnitude=2.0, load_type="pressure")]
        )],
    )
    lookup_ids = []
    original = boundary_step._pressure_vector_3d

    def counted(face, load, node_lookup_factory, element_lookup_factory):
        lookup_ids.append((
            id(node_lookup_factory()),
            id(element_lookup_factory()),
        ))
        return original(
            face,
            load,
            node_lookup_factory,
            element_lookup_factory,
        )

    monkeypatch.setattr(boundary_step, "_pressure_vector_3d", counted)
    boundary_for_step(model, "load")

    assert len(lookup_ids) == 2
    assert len(set(lookup_ids)) == 1


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
