import pytest

from tests.helpers.file_builders import write_inp


def _hex20_node_lines():
    from fem.elements.hexahedron import HEX20_NATURAL_NODE_COORDS

    return [
        f"{node_id}, {(xi + 1.0) / 2.0}, {(eta + 1.0) / 2.0}, {(zeta + 1.0) / 2.0}"
        for node_id, (xi, eta, zeta) in enumerate(
            HEX20_NATURAL_NODE_COORDS,
            start=1,
        )
    ]


def _mixed3d_element_row(elem_id, elem_type, node_ids):
    node_values = [str(node_id) for node_id in node_ids]
    return ",".join(
        [str(elem_id), elem_type, *node_values, *("" for _ in range(20 - len(node_values)))]
    )


def test_csv_read_mixed3d_reads_hex8_and_tet4_from_sectioned_csv(tmp_path):
    from fem.io import csv as csv_io

    mesh_path = write_inp(
        tmp_path,
        "mixed_hex8_tet4.csv",
        [
            "# NODES",
            "node_id,x,y,z",
            "1,0.0,0.0,0.0",
            "2,1.0,0.0,0.0",
            "3,1.0,1.0,0.0",
            "4,0.0,1.0,0.0",
            "5,0.0,0.0,1.0",
            "6,1.0,0.0,1.0",
            "7,1.0,1.0,1.0",
            "8,0.0,1.0,1.0",
            "9,2.0,0.0,0.0",
            "",
            "# ELEMENTS",
            "elem_id,type,node1,node2,node3,node4,node5,node6,node7,node8",
            "1,Hex8,1,2,3,4,5,6,7,8",
            "2,Tet4,2,9,3,6,,,,",
        ],
    )

    mesh = csv_io.read_mixed3d(mesh_path)

    assert mesh.num_nodes == 9
    assert mesh.num_elements == 2
    assert [elem.type for elem in mesh.elements] == ["Hex8", "Tet4"]
    assert mesh.elements[0].node_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert mesh.elements[1].node_ids == [2, 9, 3, 6]
    assert all(elem.props == {} for elem in mesh.elements)
    assert mesh.dofs_per_node == 3


def test_csv_read_mixed3d_flows_through_model_and_vtk_post(tmp_path):
    from fem.io import csv as csv_io
    from fem.post import vtk
    from tests.helpers.result_builders import make_zero_result

    element_header = ",".join(
        ["elem_id", "type", *(f"node{index}" for index in range(1, 21)), "material_id"]
    )
    mesh_path = write_inp(
        tmp_path,
        "mixed_reader_to_post.csv",
        [
            "node_id,x,y,z",
            "1,0.0,0.0,0.0",
            "2,1.0,0.0,0.0",
            "3,1.0,1.0,0.0",
            "4,0.0,1.0,0.0",
            "5,0.0,0.0,1.0",
            "6,1.0,0.0,1.0",
            "7,1.0,1.0,1.0",
            "8,0.0,1.0,1.0",
            "9,2.0,0.0,0.0",
            element_header,
            f"{_mixed3d_element_row(1, 'Hex8', range(1, 9))},1",
            f"{_mixed3d_element_row(2, 'Tet4', (2, 9, 3, 6))},1",
        ],
    )
    material_path = write_inp(
        tmp_path,
        "mixed_reader_to_post_materials.csv",
        ["material_id,E,nu", "1,210000,0.3"],
    )

    mesh = csv_io.read_mixed3d(mesh_path, material_path)
    result = make_zero_result(mesh, "mixed_reader_to_post")
    vtk.export.from_result(result, output_dir=tmp_path)

    vtk_lines = (tmp_path / "mixed_reader_to_post.vtk").read_text(
        encoding="utf-8"
    ).splitlines()
    cell_types_index = vtk_lines.index("CELL_TYPES 2")

    assert result.model.mesh is mesh
    assert [int(value) for value in vtk_lines[cell_types_index + 1 : cell_types_index + 3]] == [
        12,
        10,
    ]
    assert (tmp_path / "mixed_reader_to_post_nodal_displacement.csv").exists()
    assert (tmp_path / "mixed_reader_to_post_element_stress.csv").exists()
    assert (tmp_path / "mixed_reader_to_post_nodal_stress.csv").exists()


def test_csv_read_mixed3d_reads_four_supported_element_types(tmp_path):
    from fem.io import csv as csv_io

    element_header = ",".join(
        ["elem_id", "type", *(f"node{index}" for index in range(1, 21))]
    )
    mesh_path = write_inp(
        tmp_path,
        "mixed_four_types.csv",
        [
            "# NODES",
            "node_id,x,y,z",
            *_hex20_node_lines(),
            "",
            "# ELEMENTS",
            element_header,
            _mixed3d_element_row(1, "Hex8", range(1, 9)),
            _mixed3d_element_row(2, "Hex20", range(1, 21)),
            _mixed3d_element_row(3, "Tet4", range(1, 5)),
            _mixed3d_element_row(4, "Tet10", range(1, 11)),
        ],
    )

    mesh = csv_io.read_mixed3d(mesh_path)

    assert [elem.type for elem in mesh.elements] == ["Hex8", "Hex20", "Tet4", "Tet10"]
    assert [elem.node_ids for elem in mesh.elements] == [
        list(range(1, 9)),
        list(range(1, 21)),
        list(range(1, 5)),
        list(range(1, 11)),
    ]
    assert all(elem.props == {} for elem in mesh.elements)


def test_csv_read_mixed3d_material_props_include_only_available_values(tmp_path):
    from fem.io import csv as csv_io

    element_header = ",".join(
        ["elem_id", "type", *(f"node{index}" for index in range(1, 21)), "material_id"]
    )
    mesh_path = write_inp(
        tmp_path,
        "mixed_material.csv",
        [
            "node_id,x,y,z",
            *_hex20_node_lines(),
            element_header,
            f"{_mixed3d_element_row(1, 'Hex8', range(1, 9))},1",
        ],
    )
    material_path = write_inp(
        tmp_path,
        "solid_materials.csv",
        ["material_id,E", "1,210000"],
    )

    mesh = csv_io.read_mixed3d(mesh_path, material_path)

    assert mesh.elements[0].props == {"material_id": 1, "E": 210000.0}


@pytest.mark.parametrize(
    ("reader_name", "element_header", "element_row", "material_id"),
    (
        (
            "read_mixed3d",
            ",".join(
                [
                    "elem_id",
                    "type",
                    *(f"node{index}" for index in range(1, 21)),
                    "material_id",
                ]
            ),
            f"{_mixed3d_element_row(1, 'Hex8', range(1, 9))},7",
            7,
        ),
        (
            "read_hex8",
            "elem_id,node1,node2,node3,node4,node5,node6,node7,node8,material_id",
            "1,1,2,3,4,5,6,7,8,17",
            17,
        ),
        (
            "read_tet4",
            "elem_id,node1,node2,node3,node4,material_id",
            "1,1,2,3,4,23",
            23,
        ),
    ),
)
def test_csv_readers_keep_material_id_without_material_table(
    tmp_path,
    reader_name,
    element_header,
    element_row,
    material_id,
):
    from fem.io import csv as csv_io

    mesh_path = write_inp(
        tmp_path,
        f"{reader_name}_material_id_without_table.csv",
        [
            "node_id,x,y,z",
            *_hex20_node_lines(),
            element_header,
            element_row,
        ],
    )

    mesh = getattr(csv_io, reader_name)(mesh_path)

    assert mesh.elements[0].props == {"material_id": material_id}


def test_csv_read_mixed3d_validates_material_id_without_material_table(tmp_path):
    from fem.io import csv as csv_io

    element_header = ",".join(
        ["elem_id", "type", *(f"node{index}" for index in range(1, 21)), "material_id"]
    )
    mesh_path = write_inp(
        tmp_path,
        "mixed_invalid_material_id.csv",
        [
            "node_id,x,y,z",
            *_hex20_node_lines(),
            element_header,
            f"{_mixed3d_element_row(1, 'Hex8', range(1, 9))},steel",
        ],
    )

    with pytest.raises(ValueError) as exc_info:
        csv_io.read_mixed3d(mesh_path)

    message = str(exc_info.value)
    assert "read_mixed3d" in message
    assert str(mesh_path) in message
    assert "line 23" in message
    assert "field material_id" in message
    assert "raw value 'steel'" in message
    assert "expected an integer" in message


@pytest.mark.parametrize(
    ("elem_type", "node_ids", "message"),
    [
        ("Hex20", range(1, 20), "Hex20 row is missing node20"),
        ("Tet10", range(1, 12), "Tet10 row has extra node11"),
    ],
)
def test_csv_read_mixed3d_validates_missing_and_extra_nodes(
    tmp_path,
    elem_type,
    node_ids,
    message,
):
    from fem.io import csv as csv_io

    element_header = ",".join(
        ["elem_id", "type", *(f"node{index}" for index in range(1, 21))]
    )
    mesh_path = write_inp(
        tmp_path,
        f"mixed_{elem_type.lower()}_invalid.csv",
        [
            "# NODES",
            "node_id,x,y,z",
            *_hex20_node_lines(),
            "# ELEMENTS",
            element_header,
            _mixed3d_element_row(1, elem_type, node_ids),
        ],
    )

    with pytest.raises(ValueError, match=message):
        csv_io.read_mixed3d(mesh_path)


def test_csv_read_mixed3d_rejects_populated_node21_header_column_for_hex20(tmp_path):
    from fem.io import csv as csv_io

    element_header = ",".join(
        ["elem_id", "type", *(f"node{index}" for index in range(1, 22))]
    )
    mesh_path = write_inp(
        tmp_path,
        "mixed_hex20_node21.csv",
        [
            "# NODES",
            "node_id,x,y,z",
            *_hex20_node_lines(),
            "# ELEMENTS",
            element_header,
            f"{_mixed3d_element_row(1, 'Hex20', range(1, 21))},21",
        ],
    )

    with pytest.raises(ValueError, match="line 25 Hex20 row has extra node21"):
        csv_io.read_mixed3d(mesh_path)


def test_csv_read_mixed3d_rejects_nonempty_physical_field_beyond_header(tmp_path):
    from fem.io import csv as csv_io

    element_header = ",".join(
        ["elem_id", "type", *(f"node{index}" for index in range(1, 21))]
    )
    mesh_path = write_inp(
        tmp_path,
        "mixed_hex20_trailing_field.csv",
        [
            "# NODES",
            "node_id,x,y,z",
            *_hex20_node_lines(),
            "# ELEMENTS",
            element_header,
            f"{_mixed3d_element_row(1, 'Hex20', range(1, 21))},21",
        ],
    )

    with pytest.raises(
        ValueError,
        match="line 25 element row has nonempty trailing field beyond header",
    ):
        csv_io.read_mixed3d(mesh_path)


@pytest.mark.parametrize(
    ("reader_name", "lines", "field", "raw_value", "expected_condition"),
    (
        ("read_truss2", ["node_id,x,y,z", "bad,0,0,0"], "node_id", "bad", "integer"),
        (
            "read_beam2",
            [
                "elem_id,node_i,node_j",
                "bad,1,2",
            ],
            "elem_id",
            "bad",
            "integer",
        ),
        (
            "read_tri3",
            ["elem_id,node1,node2,node3,thickness,material_id", "1,1,2,3,bad,1"],
            "thickness",
            "bad",
            "numeric value",
        ),
        ("read_mixed3d", ["node_id,x,y,z", "1,bad,0,0"], "x", "bad", "numeric value"),
        (
            "read_hex8",
            [
                "elem_id,node1,node2,node3,node4,node5,node6,node7,node8,material_id",
                "1,1,2,3,4,5,6,bad,8,1",
            ],
            "node7",
            "bad",
            "integer",
        ),
        ("read_tet4", ["node_id,x,y,z", "1,0,bad,0"], "y", "bad", "numeric value"),
    ),
)
def test_csv_public_readers_report_invalid_numeric_fields_with_context(
    tmp_path,
    reader_name,
    lines,
    field,
    raw_value,
    expected_condition,
):
    from fem.io import csv as csv_io

    mesh_path = write_inp(tmp_path, f"{reader_name}_invalid.csv", lines)

    with pytest.raises(ValueError) as exc_info:
        getattr(csv_io, reader_name)(mesh_path)

    message = str(exc_info.value)
    assert reader_name in message
    assert str(mesh_path) in message
    assert "line 2" in message
    assert f"field {field}" in message
    assert f"raw value {raw_value!r}" in message
    assert f"expected a{ 'n' if expected_condition == 'integer' else ''} {expected_condition}" in message


def test_material_csv_builds_named_linear_elastic_material(tmp_path):
    from fem.io import materials as materials_io

    material_path = write_inp(
        tmp_path,
        "materials.csv",
        [
            "material_id,name,E,rho,nu",
            "1,steel,220e3,7800,0.3",
            "2,aluminum,70e3,2700,0.33",
        ],
    )

    steel = materials_io.linear_elastic(material_path, "steel")
    aluminum = materials_io.linear_elastic(material_path, "aluminum")

    assert steel.name == "steel"
    assert steel.properties["E"] == 220000.0
    assert steel.properties["rho"] == 7800.0
    assert steel.properties["nu"] == 0.3
    assert aluminum.name == "aluminum"
    assert aluminum.properties["E"] == 70000.0
    assert aluminum.properties["rho"] == 2700.0
    assert aluminum.properties["nu"] == 0.33
