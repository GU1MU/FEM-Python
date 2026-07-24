from fem.core.mesh import (
    Element2D,
    Element3D,
    Mesh2D,
    Mesh3D,
    Node2D,
    Node3D,
)


def make_minimal_hex_mesh():
    return Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0)],
        elements=[Element3D(1, [1, 2], type="Hex8")],
    )


def make_dof_order_meshes():
    return [
        Mesh3D(
            nodes=[Node3D(20, 0.0, 0.0, 0.0), Node3D(10, 1.0, 0.0, 0.0)],
            elements=[Element3D(1, [20, 10], type="Truss2")],
        ),
        Mesh3D(
            nodes=[Node3D(20, 0.0, 0.0, 0.0), Node3D(10, 1.0, 0.0, 0.0)],
            elements=[Element3D(1, [20, 10], type="Beam2")],
            dofs_per_node=6,
        ),
        Mesh2D(
            nodes=[Node2D(20, 0.0, 0.0), Node2D(10, 1.0, 0.0)],
            elements=[Element2D(1, [20, 10], type="Tri3")],
        ),
        Mesh3D(
            nodes=[Node3D(20, 0.0, 0.0, 0.0), Node3D(10, 1.0, 0.0, 0.0)],
            elements=[Element3D(1, [20, 10], type="Hex8")],
        ),
        Mesh3D(
            nodes=[Node3D(20, 0.0, 0.0, 0.0), Node3D(10, 1.0, 0.0, 0.0)],
            elements=[Element3D(1, [20, 10], type="Tet4")],
        ),
    ]


def make_selection_quad_mesh():
    return Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 1.0, 1.0),
            Node2D(4, 0.0, 1.0),
        ],
        elements=[Element2D(1, [1, 2, 3, 4], type="Quad4")],
    )


def make_selection_mixed_plane_mesh():
    return Mesh2D(
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
    return Mesh3D(
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
    return Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 2.0, 0.0, 0.0),
        ],
        elements=[
            Element3D(
                id=1,
                node_ids=[1, 2],
                type="Truss2",
                props={"E": 210.0, "area": 0.5},
            )
        ],
    )


def make_beam_stiffness_mesh():
    return Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 2.0, 1.0, 0.0),
        ],
        elements=[
            Element3D(
                id=1,
                node_ids=[1, 2],
                type="Beam2",
                props={
                    "E": 210.0,
                    "nu": 0.3,
                    "section_type": "rectangle",
                    "height": 1.0,
                    "width": 0.5,
                },
            )
        ],
        dofs_per_node=6,
    )


def make_quad4_stiffness_mesh():
    return Mesh2D(
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
                type="Quad4",
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
    return Mesh2D(
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
                type="Quad4",
                props={"E": 210.0, "nu": 0.3, "thickness": 2.0},
            )
        ],
    )


def make_tri3_load_mesh():
    return Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 2.0, 0.0),
            Node2D(3, 0.0, 1.0),
        ],
        elements=[
            Element2D(
                id=1,
                node_ids=[1, 2, 3],
                type="Tri3",
                props={"E": 210.0, "nu": 0.3, "thickness": 2.0},
            )
        ],
    )


def make_quad8_load_mesh():
    return Mesh2D(
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
                type="Quad8",
                props={"E": 210.0, "nu": 0.3, "thickness": 1.5},
            )
        ],
    )


def make_tri3_stiffness_mesh():
    return Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 2.0, 0.0),
            Node2D(3, 0.0, 1.0),
        ],
        elements=[
            Element2D(
                id=1,
                node_ids=[1, 2, 3],
                type="Tri3",
                props={
                    "E": 210.0,
                    "nu": 0.3,
                    "thickness": 1.0,
                    "plane_type": "stress",
                },
            )
        ],
    )


def make_tri6_stiffness_mesh():
    return Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 2.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, 1.0, 0.0),
            Node2D(5, 1.0, 0.5),
            Node2D(6, 0.0, 0.5),
        ],
        elements=[
            Element2D(
                id=1,
                node_ids=[1, 2, 3, 4, 5, 6],
                type="Tri6",
                props={
                    "E": 210.0,
                    "nu": 0.3,
                    "thickness": 1.0,
                    "plane_type": "stress",
                },
            )
        ],
    )


def make_tri6_load_mesh():
    return Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 2.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, 1.0, 0.0),
            Node2D(5, 1.0, 0.5),
            Node2D(6, 0.0, 0.5),
        ],
        elements=[
            Element2D(
                id=1,
                node_ids=[1, 2, 3, 4, 5, 6],
                type="Tri6",
                props={"E": 210.0, "nu": 0.3, "thickness": 2.0},
            )
        ],
    )


def make_quad8_stiffness_mesh():
    return Mesh2D(
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
                type="Quad8",
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
    return Mesh3D(nodes=nodes, elements=[elem])


def make_hex20_stiffness_mesh(curved=False):
    from fem.elements.hexahedron import HEX20_NATURAL_NODE_COORDS

    nodes = [
        Node3D(
            i + 1,
            (xi + 1.0) / 2.0,
            (eta + 1.0) / 2.0,
            (zeta + 1.0) / 2.0,
        )
        for i, (xi, eta, zeta) in enumerate(HEX20_NATURAL_NODE_COORDS)
    ]
    if curved:
        nodes[8].z += 0.05
    elem = Element3D(
        1,
        list(range(1, 21)),
        "Hex20",
        {"E": 210.0, "nu": 0.3},
    )
    return Mesh3D(nodes=nodes, elements=[elem])


def make_hex8_solid_stress_mesh():
    mesh = make_hex8_stiffness_mesh()
    mesh.elements[0].props = {"E": 210.0, "nu": 0.3}
    return mesh


def make_unit_hex8_mesh():
    return Mesh3D(
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
    return Mesh3D(
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
    return Mesh3D(
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


def make_mixed_hex8_tet4_mesh():
    nodes = [
        Node3D(1, 0.0, 0.0, 0.0),
        Node3D(2, 1.0, 0.0, 0.0),
        Node3D(3, 1.0, 1.0, 0.0),
        Node3D(4, 0.0, 1.0, 0.0),
        Node3D(5, 0.0, 0.0, 1.0),
        Node3D(6, 1.0, 0.0, 1.0),
        Node3D(7, 1.0, 1.0, 1.0),
        Node3D(8, 0.0, 1.0, 1.0),
        Node3D(9, 2.0, 0.0, 0.0),
    ]
    elements = [
        Element3D(
            id=1,
            node_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            type="Hex8",
            props={"E": 210.0, "nu": 0.3},
        ),
        Element3D(
            id=2,
            node_ids=[2, 9, 3, 6],
            type="Tet4",
            props={"E": 120.0, "nu": 0.25},
        ),
    ]
    return Mesh3D(nodes=nodes, elements=elements)


def make_mixed_hex8_hex20_mesh():
    hex8 = make_unit_hex8_mesh()
    hex20 = make_hex20_stiffness_mesh()
    shifted_nodes = [
        Node3D(node.id + 8, node.x + 2.0, node.y, node.z)
        for node in hex20.nodes
    ]
    return Mesh3D(
        nodes=list(hex8.nodes) + shifted_nodes,
        elements=[
            hex8.elements[0],
            Element3D(
                2,
                [node_id + 8 for node_id in hex20.elements[0].node_ids],
                "Hex20",
                {"E": 120.0, "nu": 0.25},
            ),
        ],
    )


def make_mixed_hex20_tet10_mesh():
    hex20 = make_hex20_stiffness_mesh()
    tet10 = make_tet10_stiffness_mesh()
    shifted_nodes = [
        Node3D(node.id + 20, node.x + 2.0, node.y, node.z)
        for node in tet10.nodes
    ]
    return Mesh3D(
        nodes=list(hex20.nodes) + shifted_nodes,
        elements=[
            hex20.elements[0],
            Element3D(
                2,
                [node_id + 20 for node_id in tet10.elements[0].node_ids],
                "Tet10",
                {"E": 120.0, "nu": 0.25},
            ),
        ],
    )


def make_mixed_tri3_quad4_mesh():
    nodes = [
        Node2D(1, 0.0, 0.0),
        Node2D(2, 1.0, 0.0),
        Node2D(3, 1.0, 1.0),
        Node2D(4, 0.0, 1.0),
        Node2D(5, 2.0, 0.0),
    ]
    elements = [
        Element2D(
            id=1,
            node_ids=[1, 2, 4],
            type="Tri3",
            props={"E": 100.0, "nu": 0.25, "thickness": 1.0, "plane_type": "stress"},
        ),
        Element2D(
            id=2,
            node_ids=[2, 5, 3, 4],
            type="Quad4",
            props={"E": 90.0, "nu": 0.3, "thickness": 1.0, "plane_type": "stress"},
        ),
    ]
    return Mesh2D(nodes=nodes, elements=elements)


def make_mixed_tet4_tet10_mesh():
    nodes = [
        Node3D(1, 0.0, 0.0, 0.0),
        Node3D(2, 1.0, 0.0, 0.0),
        Node3D(3, 0.0, 1.0, 0.0),
        Node3D(4, 0.0, 0.0, 1.0),
        Node3D(5, 2.0, 0.0, 0.0),
        Node3D(6, 3.0, 0.0, 0.0),
        Node3D(7, 2.0, 1.0, 0.0),
        Node3D(8, 2.0, 0.0, 1.0),
        Node3D(9, 2.5, 0.0, 0.0),
        Node3D(10, 2.5, 0.5, 0.0),
        Node3D(11, 2.0, 0.5, 0.0),
        Node3D(12, 2.0, 0.0, 0.5),
        Node3D(13, 2.5, 0.0, 0.5),
        Node3D(14, 2.0, 0.5, 0.5),
    ]
    elements = [
        Element3D(
            id=1,
            node_ids=[1, 2, 3, 4],
            type="Tet4",
            props={"E": 210.0, "nu": 0.3},
        ),
        Element3D(
            id=2,
            node_ids=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            type="Tet10",
            props={"E": 120.0, "nu": 0.25},
        ),
    ]
    return Mesh3D(nodes=nodes, elements=elements)


def make_mixed_tri6_quad8_mesh():
    nodes = [
        Node2D(1, 0.0, 0.0),
        Node2D(2, 2.0, 0.0),
        Node2D(3, 0.0, 1.0),
        Node2D(4, 1.0, 0.0),
        Node2D(5, 1.0, 0.5),
        Node2D(6, 0.0, 0.5),
        Node2D(7, 3.0, 0.0),
        Node2D(8, 5.0, 0.0),
        Node2D(9, 5.0, 2.0),
        Node2D(10, 3.0, 2.0),
        Node2D(11, 4.0, 0.0),
        Node2D(12, 5.0, 1.0),
        Node2D(13, 4.0, 2.0),
        Node2D(14, 3.0, 1.0),
    ]
    elements = [
        Element2D(
            id=1,
            node_ids=[1, 2, 3, 4, 5, 6],
            type="Tri6",
            props={"E": 100.0, "nu": 0.25, "thickness": 1.0, "plane_type": "stress"},
        ),
        Element2D(
            id=2,
            node_ids=[7, 8, 9, 10, 11, 12, 13, 14],
            type="Quad8",
            props={"E": 90.0, "nu": 0.3, "thickness": 1.0, "plane_type": "stress"},
        ),
    ]
    return Mesh2D(nodes=nodes, elements=elements)
