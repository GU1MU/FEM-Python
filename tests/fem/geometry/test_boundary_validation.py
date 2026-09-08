import numpy as np
import pytest

from fem.boundary.loads import build_load_vector
from fem.boundary.step import boundary_for_step
from fem.core import validate_model
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    Edge,
    EdgeLoad,
    ElementEdge,
    ElementFace,
    FEMModel,
    NodalLoad,
    NodeSet,
    Surface,
    SurfaceLoad,
)
from fem.elements import get_element_kernel
from fem.solvers import static_linear
from tests.helpers.mesh_builders import (
    make_beam_stiffness_mesh,
    make_hex20_stiffness_mesh,
    make_hex8_stiffness_mesh,
    make_quad4_stiffness_mesh,
    make_quad8_stiffness_mesh,
    make_tet4_stiffness_mesh,
    make_tet10_stiffness_mesh,
)


REDUCED_INTEGRATION_TYPES = (
    "C3D8R",
    "CPS4R",
    "CPE4R",
    "CPS8R",
    "CPE8R",
    "C3D20R",
)


@pytest.mark.parametrize(
    ("builder", "element_label", "inverted_node_ids"),
    [
        (make_quad4_stiffness_mesh, "Quad4", [1, 4, 3, 2]),
        (make_quad8_stiffness_mesh, "Quad8", [1, 4, 3, 2, 8, 7, 6, 5]),
    ],
)
@pytest.mark.parametrize(
    "operation",
    ("stiffness", "stress_at", "nodal_stress", "body_force"),
)
def test_inverted_quad_is_rejected_on_every_geometry_dependent_path(
    builder,
    element_label,
    inverted_node_ids,
    operation,
):
    mesh = builder()
    elem = mesh.elements[0]
    elem.node_ids = inverted_node_ids
    kernel = get_element_kernel(elem.type)
    displacement = np.zeros(mesh.num_dofs, dtype=float)

    with pytest.raises(
        ValueError,
        match=rf"{element_label} element 1 has non-positive Jacobian determinant .*expected > 0",
    ):
        if operation == "stiffness":
            kernel.stiffness(mesh, elem)
        elif operation == "stress_at":
            kernel.stress_at(mesh, elem, displacement, 0.0, 0.0)
        elif operation == "nodal_stress":
            kernel.nodal_stress(mesh, elem, displacement)
        else:
            kernel.body_force(mesh, elem, (0.0, -1.0))


@pytest.mark.parametrize("operation", ("body_force", "edge_traction"))
def test_quad_load_consumers_reject_invalid_shared_thickness(operation):
    mesh = make_quad4_stiffness_mesh()
    elem = mesh.elements[0]
    elem.props["thickness"] = 0.0
    kernel = get_element_kernel(elem.type)

    with pytest.raises(
        ValueError,
        match=r"Quad4 element 1 thickness must be finite and > 0",
    ):
        if operation == "body_force":
            kernel.body_force(mesh, elem, (0.0, -1.0))
        else:
            kernel.edge_traction(mesh, elem, 0, (0.0, -1.0))


@pytest.mark.parametrize(
    ("builder", "element_type", "expected_plane_type"),
    [
        (make_quad4_stiffness_mesh, "CPE4", "strain"),
        (make_quad4_stiffness_mesh, "CPS4", "stress"),
        (make_quad4_stiffness_mesh, "Quad4", "stress"),
        (make_quad8_stiffness_mesh, "CPE8", "strain"),
        (make_quad8_stiffness_mesh, "CPS8", "stress"),
        (make_quad8_stiffness_mesh, "Quad8", "stress"),
    ],
)
def test_quad_plane_type_default_follows_explicit_element_family(
    builder,
    element_type,
    expected_plane_type,
):
    mesh = builder()
    elem = mesh.elements[0]
    elem.type = element_type
    elem.props.pop("plane_type", None)

    _, plane_type, _ = get_element_kernel(element_type).nodal_stress(
        mesh,
        elem,
        np.zeros(mesh.num_dofs, dtype=float),
    )

    assert plane_type == expected_plane_type


@pytest.mark.parametrize("element_type", REDUCED_INTEGRATION_TYPES)
def test_registry_rejects_unimplemented_reduced_integration(element_type):
    with pytest.raises(
        NotImplementedError,
        match=rf"Unsupported element type: {element_type}; reduced integration is not implemented",
    ):
        get_element_kernel(element_type)


def test_registry_does_not_dispatch_arbitrary_substring_matches():
    with pytest.raises(
        NotImplementedError,
        match=r"Unsupported element type: wrapped-Quad4-variant",
    ):
        get_element_kernel("wrapped-Quad4-variant")


@pytest.mark.parametrize(
    ("local_index", "node_ids", "message"),
    [
        (0, (2, 3), r"local_index 0 expects node_ids \(1, 2\), got \(2, 3\)"),
        (4, (1, 2), r"edge local_index 4 is invalid.*expected 0 through 3"),
    ],
)
def test_validate_model_matches_edge_index_to_kernel_topology(
    local_index,
    node_ids,
    message,
):
    mesh = make_quad4_stiffness_mesh()
    model = FEMModel(
        mesh=mesh,
        edges={"LOAD": Edge("LOAD", [ElementEdge(1, local_index, node_ids)])},
    )

    with pytest.raises(ValueError, match=message):
        validate_model(model)


@pytest.mark.parametrize(
    ("builder", "node_ids", "message"),
    [
        (
            make_hex8_stiffness_mesh,
            (5, 6, 7, 8),
            r"surface LOAD element 1 local_index 0 expects node_ids "
            r"\(1, 4, 3, 2\), got \(5, 6, 7, 8\)",
        ),
        (
            make_tet4_stiffness_mesh,
            (1, 2, 3),
            r"surface LOAD element 1 local_index 0 expects node_ids "
            r"\(2, 3, 4\), got \(1, 2, 3\)",
        ),
    ],
    ids=["hex8", "tet4"],
)
def test_validate_model_matches_face_index_to_kernel_topology(
    builder,
    node_ids,
    message,
):
    mesh = builder()
    model = FEMModel(
        mesh=mesh,
        surfaces={"LOAD": Surface("LOAD", [ElementFace(1, 0, node_ids)])},
    )

    with pytest.raises(ValueError, match=message):
        validate_model(model)


def test_validate_model_accepts_kernel_boundary_topology():
    plane_mesh = make_quad4_stiffness_mesh()
    plane_model = FEMModel(
        mesh=plane_mesh,
        edges={
            "VALID": Edge(
                "VALID",
                [
                    ElementEdge(1, 0, (2, 1)),
                    ElementEdge(1, 1, (2, 3)),
                ],
            )
        },
    )
    solid_mesh = make_hex8_stiffness_mesh()
    solid_model = FEMModel(
        mesh=solid_mesh,
        surfaces={
            "VALID": Surface(
                "VALID",
                [
                    ElementFace(1, 0, (4, 3, 2, 1)),
                    ElementFace(1, 5, (2, 6, 7, 3)),
                ],
            )
        },
    )

    validate_model(plane_model)
    validate_model(solid_model)


def test_validate_model_preserves_quadratic_edge_midside_role():
    mesh = make_quad8_stiffness_mesh()
    reversed_model = FEMModel(
        mesh=mesh,
        edges={"VALID": Edge("VALID", [ElementEdge(1, 0, (2, 5, 1))])},
    )
    shifted_model = FEMModel(
        mesh=mesh,
        edges={"BAD": Edge("BAD", [ElementEdge(1, 0, (5, 2, 1))])},
    )

    validate_model(reversed_model)
    with pytest.raises(
        ValueError,
        match=(
            r"edge BAD element 1 local_index 0 expects node_ids "
            r"\(1, 5, 2\), got \(5, 2, 1\)"
        ),
    ):
        validate_model(shifted_model)


@pytest.mark.parametrize(
    ("builder", "rotated_node_ids", "reversed_node_ids", "shifted_node_ids"),
    [
        (
            make_hex20_stiffness_mesh,
            (4, 3, 2, 1, 11, 10, 9, 12),
            (1, 2, 3, 4, 9, 10, 11, 12),
            (4, 3, 2, 12, 11, 10, 9, 1),
        ),
        (
            make_tet10_stiffness_mesh,
            (3, 4, 2, 10, 9, 6),
            (2, 4, 3, 9, 10, 6),
            (3, 4, 2, 6, 10, 9),
        ),
    ],
    ids=["hex20", "tet10"],
)
def test_validate_model_preserves_quadratic_face_midside_ring(
    builder,
    rotated_node_ids,
    reversed_node_ids,
    shifted_node_ids,
):
    mesh = builder()
    rotated_model = FEMModel(
        mesh=mesh,
        surfaces={
            "VALID": Surface(
                "VALID",
                [ElementFace(1, 0, rotated_node_ids)],
            )
        },
    )
    reversed_model = FEMModel(
        mesh=mesh,
        surfaces={
            "VALID": Surface(
                "VALID",
                [ElementFace(1, 0, reversed_node_ids)],
            )
        },
    )
    shifted_model = FEMModel(
        mesh=mesh,
        surfaces={
            "BAD": Surface(
                "BAD",
                [ElementFace(1, 0, shifted_node_ids)],
            )
        },
    )

    validate_model(rotated_model)
    validate_model(reversed_model)
    with pytest.raises(ValueError, match=r"surface BAD element 1 local_index 0"):
        validate_model(shifted_model)


def test_unused_hex8_named_edge_does_not_block_validation_or_solve():
    mesh = make_hex8_stiffness_mesh()
    model = FEMModel(
        mesh=mesh,
        node_sets={"FIXED": NodeSet("FIXED", (1, 4, 5, 8))},
        edges={"UNUSED": Edge("UNUSED", [ElementEdge(1, 4, (5, 6))])},
        steps=[
            AnalysisStep(
                "load",
                boundaries=[DisplacementConstraint("FIXED", 1, 3, 0.0)],
                cloads=[NodalLoad(2, 1, 1.0)],
            )
        ],
    )

    validate_model(model)
    result = static_linear.solve(model, "load")

    assert np.all(np.isfinite(result.U))


def test_beam2_rejects_generic_3d_edge_loads_in_favor_of_line_loads():
    mesh = make_beam_stiffness_mesh()
    model = FEMModel(
        mesh=mesh,
        edges={"LINE": Edge("LINE", [ElementEdge(1, 0, (1, 2))])},
        steps=[
            AnalysisStep(
                "load",
                edge_loads=[EdgeLoad("LINE", (0.0, -2.0, 0.0), load_type="traction")],
            )
        ],
    )

    with pytest.raises(NotImplementedError, match=r"3D edge loads are not supported"):
        boundary_for_step(model, "load")


def test_beam2_rejects_generic_surface_traction_assembly():
    mesh = make_beam_stiffness_mesh()
    model = FEMModel(
        mesh=mesh,
        surfaces={"LINE": Surface("LINE", [ElementFace(1, 0, (1, 2))])},
        steps=[
            AnalysisStep(
                "load",
                surface_loads=[
                    SurfaceLoad("LINE", (0.0, -2.0, 0.0), load_type="traction")
                ],
            )
        ],
    )

    boundary = boundary_for_step(model, "load")
    with pytest.raises(
        NotImplementedError,
        match=r"Unsupported element type for face_traction assembly: Beam2",
    ):
        build_load_vector(mesh, boundary)
