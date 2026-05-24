from fem.core.mesh import (
    BeamMesh2D,
    Element2D,
    Element3D,
    HexMesh3D,
    Node2D,
    Node3D,
    PlaneMesh2D,
    TetMesh3D,
    TrussMesh2D,
)


def make_minimal_hex_mesh():
    return HexMesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
        elements=[Element3D(1, [1, 2], type="Hex8")],
    )


def make_dof_order_meshes():
    return [
        TrussMesh2D(
            nodes=[Node2D(20, 0.0, 0.0), Node2D(10, 1.0, 0.0)],
            elements=[Element2D(1, [20, 10], type="Truss2D")],
        ),
        BeamMesh2D(
            nodes=[Node2D(20, 0.0, 0.0), Node2D(10, 1.0, 0.0)],
            elements=[Element2D(1, [20, 10], type="Beam2D")],
        ),
        PlaneMesh2D(
            nodes=[Node2D(20, 0.0, 0.0), Node2D(10, 1.0, 0.0)],
            elements=[Element2D(1, [20, 10], type="Tri3Plane")],
        ),
        HexMesh3D(
            nodes=[Node3D(20, 0.0, 0.0, 0.0), Node3D(10, 1.0, 0.0, 0.0)],
            elements=[Element3D(1, [20, 10], type="Hex8")],
        ),
        TetMesh3D(
            nodes=[Node3D(20, 0.0, 0.0, 0.0), Node3D(10, 1.0, 0.0, 0.0)],
            elements=[Element3D(1, [20, 10], type="Tet4")],
        ),
    ]


def make_selection_quad_mesh():
    return PlaneMesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 1.0, 1.0),
            Node2D(4, 0.0, 1.0),
        ],
        elements=[Element2D(1, [1, 2, 3, 4], type="Quad4")],
    )


def make_selection_mixed_plane_mesh():
    return PlaneMesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 1.0, 1.0),
            Node2D(4, 0.0, 1.0),
        ],
        elements=[
            Element2D(1, [1, 2, 3], type="Tri3"),
            Element2D(2, [1, 3, 4], type="Quad4"),
        ],
    )


def make_selection_hex_mesh():
    return HexMesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 2.0, 0.0, 0.0),
            Node3D(3, 2.0, 3.0, 0.0),
            Node3D(4, 0.0, 3.0, 0.0),
            Node3D(5, 0.0, 0.0, 4.0),
            Node3D(6, 2.0, 0.0, 4.0),
            Node3D(7, 2.0, 3.0, 4.0),
            Node3D(8, 0.0, 3.0, 4.0),
        ],
        elements=[Element3D(1, [1, 2, 3, 4, 5, 6, 7, 8], type="Hex8")],
    )


def make_truss_stiffness_mesh():
    return TrussMesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 2.0, 0.0),
        ],
        elements=[
            Element2D(
                id=1,
                node_ids=[1, 2],
                type="Truss2D",
                props={"E": 210.0, "area": 0.5},
            )
        ],
    )


def make_beam_stiffness_mesh():
    return BeamMesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 2.0, 1.0),
        ],
        elements=[
            Element2D(
                id=1,
                node_ids=[1, 2],
                type="Beam2D",
                props={"E": 210.0, "area": 0.5, "Izz": 0.25},
            )
        ],
    )


def make_quad4_stiffness_mesh():
    return PlaneMesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 2.0, 0.0),
            Node2D(3, 2.0, 1.0),
            Node2D(4, 0.0, 1.0),
        ],
        elements=[
            Element2D(
                id=1,
                node_ids=[1, 2, 3, 4],
                type="Quad4Plane",
                props={
                    "E": 210.0,
                    "nu": 0.3,
                    "thickness": 1.0,
                    "plane_type": "stress",
                },
            )
        ],
    )


def make_quad4_boundary_mesh():
    return PlaneMesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 2.0, 0.0),
            Node2D(3, 2.0, 1.0),
            Node2D(4, 0.0, 1.0),
        ],
        elements=[
            Element2D(
                id=1,
                node_ids=[1, 2, 3, 4],
                type="Quad4Plane",
                props={"E": 210.0, "nu": 0.3, "thickness": 2.0},
            )
        ],
    )


def make_tri3_load_mesh():
    return PlaneMesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 2.0, 0.0),
            Node2D(3, 0.0, 1.0),
        ],
        elements=[
            Element2D(
                id=1,
                node_ids=[1, 2, 3],
                type="Tri3Plane",
                props={"E": 210.0, "nu": 0.3, "thickness": 2.0},
            )
        ],
    )


def make_quad8_load_mesh():
    return PlaneMesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 2.0, 0.0),
            Node2D(3, 2.0, 2.0),
            Node2D(4, 0.0, 2.0),
            Node2D(5, 1.0, 0.0),
            Node2D(6, 2.0, 1.0),
            Node2D(7, 1.0, 2.0),
            Node2D(8, 0.0, 1.0),
        ],
        elements=[
            Element2D(
                id=1,
                node_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                type="Quad8Plane",
                props={"E": 210.0, "nu": 0.3, "thickness": 1.5},
            )
        ],
    )


def make_tri3_stiffness_mesh():
    return PlaneMesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 2.0, 0.0),
            Node2D(3, 0.0, 1.0),
        ],
        elements=[
            Element2D(
                id=1,
                node_ids=[1, 2, 3],
                type="Tri3Plane",
                props={
                    "E": 210.0,
                    "nu": 0.3,
                    "thickness": 1.0,
                    "plane_type": "stress",
                },
            )
        ],
    )


def make_quad8_stiffness_mesh():
    return PlaneMesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 2.0, 0.0),
            Node2D(3, 2.0, 2.0),
            Node2D(4, 0.0, 2.0),
            Node2D(5, 1.0, 0.0),
            Node2D(6, 2.0, 1.0),
            Node2D(7, 1.0, 2.0),
            Node2D(8, 0.0, 1.0),
        ],
        elements=[
            Element2D(
                id=1,
                node_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                type="Quad8Plane",
                props={
                    "E": 210.0,
                    "nu": 0.3,
                    "thickness": 1.0,
                    "plane_type": "stress",
                },
            )
        ],
    )


def make_hex8_stiffness_mesh():
    nodes = [
        Node3D(1, 0.0, 0.0, 0.0),
        Node3D(2, 2.0, 0.0, 0.0),
        Node3D(3, 2.0, 3.0, 0.0),
        Node3D(4, 0.0, 3.0, 0.0),
        Node3D(5, 0.0, 0.0, 4.0),
        Node3D(6, 2.0, 0.0, 4.0),
        Node3D(7, 2.0, 3.0, 4.0),
        Node3D(8, 0.0, 3.0, 4.0),
    ]
    elem = Element3D(
        id=1,
        node_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        type="Hex8",
        props={"E": 1.0, "nu": 0.25},
    )
    return HexMesh3D(nodes=nodes, elements=[elem])


def make_hex8_solid_stress_mesh():
    mesh = make_hex8_stiffness_mesh()
    mesh.elements[0].props = {"E": 210.0, "nu": 0.3}
    return mesh


def make_unit_hex8_mesh():
    return HexMesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 1.0, 0.0, 0.0),
            Node3D(3, 1.0, 1.0, 0.0),
            Node3D(4, 0.0, 1.0, 0.0),
            Node3D(5, 0.0, 0.0, 1.0),
            Node3D(6, 1.0, 0.0, 1.0),
            Node3D(7, 1.0, 1.0, 1.0),
            Node3D(8, 0.0, 1.0, 1.0),
        ],
        elements=[
            Element3D(
                id=1,
                node_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                type="Hex8",
                props={"E": 210.0, "nu": 0.3},
            )
        ],
    )


def make_tet4_stiffness_mesh():
    return TetMesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 1.0, 0.0, 0.0),
            Node3D(3, 0.0, 1.0, 0.0),
            Node3D(4, 0.0, 0.0, 1.0),
        ],
        elements=[
            Element3D(
                id=1,
                node_ids=[1, 2, 3, 4],
                type="Tet4",
                props={"E": 210.0, "nu": 0.3},
            )
        ],
    )


def make_tet10_stiffness_mesh():
    return TetMesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 1.0, 0.0, 0.0),
            Node3D(3, 0.0, 1.0, 0.0),
            Node3D(4, 0.0, 0.0, 1.0),
            Node3D(5, 0.5, 0.0, 0.0),
            Node3D(6, 0.5, 0.5, 0.0),
            Node3D(7, 0.0, 0.5, 0.0),
            Node3D(8, 0.0, 0.0, 0.5),
            Node3D(9, 0.5, 0.0, 0.5),
            Node3D(10, 0.0, 0.5, 0.5),
        ],
        elements=[
            Element3D(
                id=1,
                node_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                type="Tet10",
                props={"E": 210.0, "nu": 0.3},
            )
        ],
    )
