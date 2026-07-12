from __future__ import annotations

import csv as csv_lib
from typing import Dict, List, Optional

from ..core.mesh import BeamMesh3D, Element2D, Element3D, HexMesh3D, Node2D, Node3D, PlaneMesh2D, TetMesh3D, TrussMesh3D
from ..elements.beam_section import parse_beam2_section
from .materials import _get_float_from_material, read


_TRUSS2_NODE_HEADER = ["node_id", "x", "y", "z"]
_TRUSS2_ELEMENT_HEADER = ["elem_id", "node_i", "node_j", "area", "material_id"]
_BEAM2_NODE_HEADER = ["node_id", "x", "y", "z"]
_BEAM2_ELEMENT_HEADER = [
    "elem_id", "node_i", "node_j", "section_type", "radius",
    "outer_radius", "inner_radius", "size_y", "size_z", "local_y_x",
    "local_y_y", "local_y_z", "material_id",
]


def _integer_field(
    value: object,
    *,
    reader_name: str,
    mesh_path: str,
    line_no: int,
    field: str,
) -> int:
    """Parse one CSV integer field with reader and source context."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{reader_name}: mesh CSV {str(mesh_path)} line {line_no} field {field} "
            f"has raw value {value!r}; expected an integer"
        ) from exc


def _numeric_field(
    value: object,
    *,
    reader_name: str,
    mesh_path: str,
    line_no: int,
    field: str,
) -> float:
    """Parse one CSV numeric field with reader and source context."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{reader_name}: mesh CSV {str(mesh_path)} line {line_no} field {field} "
            f"has raw value {value!r}; expected a numeric value"
        ) from exc


def read_truss2(
    mesh_path: str,
    material_path: Optional[str] = None,
) -> TrussMesh3D:
    """Read a spatial Truss2 mesh CSV with optional materials."""

    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials_dict = read(material_path)

    nodes: List[Node3D] = []
    elements: List[Element3D] = []

    mode: Optional[str] = None

    with open(mesh_path, "r", encoding="utf-8") as f:
        reader = csv_lib.reader(f)

        for line_no, row in enumerate(reader, start=1):
            row = [col.strip() for col in row]

            if not row or all(col == "" for col in row):
                continue

            if row[0].startswith("#"):
                continue

            # 节点表头
            if row[0] == "node_id":
                if row != _TRUSS2_NODE_HEADER:
                    raise ValueError("Truss2 node header must be node_id,x,y,z")
                mode = "nodes"
                continue

            # 单元表头
            if row[0] == "elem_id":
                if row != _TRUSS2_ELEMENT_HEADER:
                    raise ValueError(
                        "Truss2 element header must be "
                        + ",".join(_TRUSS2_ELEMENT_HEADER)
                    )
                mode = "elements"
                continue

            if mode == "nodes":
                if len(row) < 4:
                    raise ValueError(
                        f"Truss2 mesh CSV {mesh_path!r} line {line_no} node row "
                        f"requires node_id,x,y,z; got {row!r}"
                    )
                node_id = _integer_field(
                    row[0], reader_name="read_truss2", mesh_path=mesh_path,
                    line_no=line_no, field="node_id",
                )
                x = _numeric_field(
                    row[1], reader_name="read_truss2", mesh_path=mesh_path,
                    line_no=line_no, field="x",
                )
                y = _numeric_field(
                    row[2], reader_name="read_truss2", mesh_path=mesh_path,
                    line_no=line_no, field="y",
                )
                z = _numeric_field(
                    row[3], reader_name="read_truss2", mesh_path=mesh_path,
                    line_no=line_no, field="z",
                )
                nodes.append(Node3D(id=node_id, x=x, y=y, z=z))

            elif mode == "elements":
                if len(row) < 5:
                    raise ValueError(
                        f"Truss2 mesh CSV {mesh_path!r} line {line_no} element row "
                        f"requires elem_id,node_i,node_j,area,material_id; got {row!r}"
                    )
                elem_id = _integer_field(
                    row[0], reader_name="read_truss2", mesh_path=mesh_path,
                    line_no=line_no, field="elem_id",
                )
                node_i = _integer_field(
                    row[1], reader_name="read_truss2", mesh_path=mesh_path,
                    line_no=line_no, field="node_i",
                )
                node_j = _integer_field(
                    row[2], reader_name="read_truss2", mesh_path=mesh_path,
                    line_no=line_no, field="node_j",
                )
                area = _numeric_field(
                    row[3], reader_name="read_truss2", mesh_path=mesh_path,
                    line_no=line_no, field="area",
                )
                mid = _integer_field(
                    row[4], reader_name="read_truss2", mesh_path=mesh_path,
                    line_no=line_no, field="material_id",
                )

                props: Dict[str, object] = {
                    "area": area,
                    "material_id": mid,
                }

                if materials_dict:
                    mat_row = materials_dict.get(mid)
                    if mat_row is not None:
                        raw_E = _get_float_from_material(mat_row, ["E"])
                        raw_rho = _get_float_from_material(mat_row, ["rho"])
                        if raw_E is not None:
                            props["E"] = raw_E
                        if raw_rho is not None:
                            props["rho"] = raw_rho

                elements.append(
                    Element3D(
                        id=elem_id,
                        node_ids=[node_i, node_j],
                        type="Truss2",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"Truss2 mesh CSV {mesh_path!r} line {line_no} has data before "
                    f"a recognized node_id or elem_id header: {row!r}"
                )

    if not nodes:
        raise ValueError(f"Truss2 mesh CSV {mesh_path!r} contains no node rows")
    if not elements:
        raise ValueError(f"Truss2 mesh CSV {mesh_path!r} contains no element rows")

    return TrussMesh3D(nodes=nodes, elements=elements)


def read_beam2(
    mesh_path: str,
    material_path: Optional[str] = None,
) -> BeamMesh3D:
    """Read a spatial Beam2 mesh CSV with optional materials."""

    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials_dict = read(material_path)

    nodes: List[Node3D] = []
    elements: List[Element3D] = []

    mode: Optional[str] = None

    with open(mesh_path, "r", encoding="utf-8") as f:
        reader = csv_lib.reader(f)

        for line_no, row in enumerate(reader, start=1):
            row = [col.strip() for col in row]

            if not row or all(col == "" for col in row):
                continue

            if row[0].startswith("#"):
                continue

            # 表头：节点
            if row[0] == "node_id":
                if row != _BEAM2_NODE_HEADER:
                    raise ValueError("Beam2 node header must be node_id,x,y,z")
                mode = "nodes"
                continue

            # 表头：单元
            if row[0] == "elem_id":
                if row != _BEAM2_ELEMENT_HEADER:
                    raise ValueError(
                        "Beam2 element header must be "
                        + ",".join(_BEAM2_ELEMENT_HEADER)
                    )
                mode = "elements"
                continue

            if mode == "nodes":
                if len(row) < 4:
                    raise ValueError(
                        f"Beam2 mesh CSV {mesh_path!r} line {line_no} node row "
                        f"requires node_id,x,y,z; got {row!r}"
                    )
                node_id = _integer_field(
                    row[0], reader_name="read_beam2", mesh_path=mesh_path,
                    line_no=line_no, field="node_id",
                )
                x = _numeric_field(
                    row[1], reader_name="read_beam2", mesh_path=mesh_path,
                    line_no=line_no, field="x",
                )
                y = _numeric_field(
                    row[2], reader_name="read_beam2", mesh_path=mesh_path,
                    line_no=line_no, field="y",
                )
                z = _numeric_field(
                    row[3], reader_name="read_beam2", mesh_path=mesh_path,
                    line_no=line_no, field="z",
                )
                nodes.append(Node3D(id=node_id, x=x, y=y, z=z))

            elif mode == "elements":
                if len(row) != len(_BEAM2_ELEMENT_HEADER):
                    raise ValueError(
                        f"Beam2 mesh CSV {mesh_path!r} line {line_no} element row "
                        f"requires {','.join(_BEAM2_ELEMENT_HEADER)}; got {row!r}"
                    )
                elem_id = _integer_field(
                    row[0], reader_name="read_beam2", mesh_path=mesh_path,
                    line_no=line_no, field="elem_id",
                )
                node_i = _integer_field(
                    row[1], reader_name="read_beam2", mesh_path=mesh_path,
                    line_no=line_no, field="node_i",
                )
                node_j = _integer_field(
                    row[2], reader_name="read_beam2", mesh_path=mesh_path,
                    line_no=line_no, field="node_j",
                )
                section_props: Dict[str, object] = {"section_type": row[3]}
                for index, field in zip(
                    (4, 5, 6, 7, 8),
                    ("radius", "outer_radius", "inner_radius", "size_y", "size_z"),
                ):
                    if row[index] != "":
                        section_props[field] = _numeric_field(
                            row[index], reader_name="read_beam2", mesh_path=mesh_path,
                            line_no=line_no, field=field,
                        )
                local_y = tuple(
                    _numeric_field(
                        row[index], reader_name="read_beam2", mesh_path=mesh_path,
                        line_no=line_no, field=field,
                    )
                    for index, field in zip(
                        (9, 10, 11), ("local_y_x", "local_y_y", "local_y_z")
                    )
                )
                mid = _integer_field(
                    row[12], reader_name="read_beam2", mesh_path=mesh_path,
                    line_no=line_no, field="material_id",
                )

                parse_beam2_section(section_props)
                props: Dict[str, object] = {
                    **section_props,
                    "local_y": local_y,
                    "material_id": mid,
                }

                if materials_dict:
                    mat_row = materials_dict.get(mid)
                    if mat_row is not None:
                        raw_E = _get_float_from_material(mat_row, ["E"])
                        raw_nu = _get_float_from_material(mat_row, ["nu", "poisson"])
                        raw_rho = _get_float_from_material(mat_row, ["rho"])
                        if raw_E is not None:
                            props["E"] = raw_E
                        if raw_nu is not None:
                            props["nu"] = raw_nu
                        if raw_rho is not None:
                            props["rho"] = raw_rho

                elements.append(
                    Element3D(
                        id=elem_id,
                        node_ids=[node_i, node_j],
                        type="Beam2",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"Beam2 mesh CSV {mesh_path!r} line {line_no} has data before "
                    f"a recognized node_id or elem_id header: {row!r}"
                )

    if not nodes:
        raise ValueError(f"Beam2 mesh CSV {mesh_path!r} contains no node rows")
    if not elements:
        raise ValueError(f"Beam2 mesh CSV {mesh_path!r} contains no element rows")

    return BeamMesh3D(nodes=nodes, elements=elements)


def read_tri3(
    mesh_path: str,
    material_path: Optional[str] = None,
    plane_type: str = "stress",
) -> PlaneMesh2D:
    """Read a Tri3 plane mesh CSV with optional materials."""

    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        from .materials import _get_float_from_material, read
        materials_dict = read(material_path)

    nodes: List[Node2D] = []
    elements: List[Element2D] = []

    mode: Optional[str] = None

    with open(mesh_path, "r", encoding="utf-8") as f:
        reader = csv_lib.reader(f)

        for line_no, row in enumerate(reader, start=1):
            row = [col.strip() for col in row]

            if not row or all(col == "" for col in row):
                continue

            if row[0].startswith("#"):
                continue

            if row[0] == "node_id":
                mode = "nodes"
                continue

            if row[0] == "elem_id":
                mode = "elements"
                continue

            if mode == "nodes":
                if len(row) < 3:
                    raise ValueError(
                        f"Tri3 mesh CSV {mesh_path!r} line {line_no} node row requires "
                        f"node_id,x,y; got {row!r}"
                    )
                node_id = _integer_field(
                    row[0], reader_name="read_tri3", mesh_path=mesh_path,
                    line_no=line_no, field="node_id",
                )
                x = _numeric_field(
                    row[1], reader_name="read_tri3", mesh_path=mesh_path,
                    line_no=line_no, field="x",
                )
                y = _numeric_field(
                    row[2], reader_name="read_tri3", mesh_path=mesh_path,
                    line_no=line_no, field="y",
                )
                nodes.append(Node2D(id=node_id, x=x, y=y))

            elif mode == "elements":
                # elem_id,node1,node2,node3,thickness,material_id
                if len(row) < 6:
                    raise ValueError(
                        f"Tri3 mesh CSV {mesh_path!r} line {line_no} element row requires "
                        f"elem_id,node1,node2,node3,thickness,material_id; got {row!r}"
                    )
                elem_id = _integer_field(
                    row[0], reader_name="read_tri3", mesh_path=mesh_path,
                    line_no=line_no, field="elem_id",
                )
                n1 = _integer_field(
                    row[1], reader_name="read_tri3", mesh_path=mesh_path,
                    line_no=line_no, field="node1",
                )
                n2 = _integer_field(
                    row[2], reader_name="read_tri3", mesh_path=mesh_path,
                    line_no=line_no, field="node2",
                )
                n3 = _integer_field(
                    row[3], reader_name="read_tri3", mesh_path=mesh_path,
                    line_no=line_no, field="node3",
                )
                thickness = _numeric_field(
                    row[4], reader_name="read_tri3", mesh_path=mesh_path,
                    line_no=line_no, field="thickness",
                )
                mid = _integer_field(
                    row[5], reader_name="read_tri3", mesh_path=mesh_path,
                    line_no=line_no, field="material_id",
                )

                props: Dict[str, object] = {
                    "thickness": thickness,
                    "material_id": mid,
                    "plane_type": plane_type,
                }

                if materials_dict:
                    from .materials import _get_float_from_material

                    mat_row = materials_dict.get(mid)
                    if mat_row is not None:
                        E_val = _get_float_from_material(mat_row, ["E"])
                        nu_val = _get_float_from_material(mat_row, ["nu"])
                        rho_val = _get_float_from_material(mat_row, ["rho"])

                        if E_val is None or nu_val is None:
                            raise KeyError(
                                f"Material {mid} for Tri3 element {elem_id} is missing E or nu; "
                                f"material row={mat_row}"
                            )

                        props["E"] = E_val
                        props["nu"] = nu_val
                        if rho_val is not None:
                            props["rho"] = rho_val

                elements.append(
                    Element2D(
                        id=elem_id,
                        node_ids=[n1, n2, n3],
                        type="Tri3Plane",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"Tri3 mesh CSV {mesh_path!r} line {line_no} has data before "
                    f"a recognized node_id or elem_id header: {row!r}"
                )

    if not nodes:
        raise ValueError(f"Tri3 mesh CSV {mesh_path!r} contains no node rows")
    if not elements:
        raise ValueError(f"Tri3 mesh CSV {mesh_path!r} contains no element rows")

    return PlaneMesh2D(nodes=nodes, elements=elements)


def read_mixed3d(
    mesh_path: str,
    material_path: Optional[str] = None,
) -> HexMesh3D:
    """Read a mixed 3D mesh CSV with Hex8, Hex20, Tet4, and Tet10 elements."""

    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials_dict = read(material_path)

    nodes: List[Node3D] = []
    elements: List[Element3D] = []

    mode: Optional[str] = None
    element_header: List[str] = []

    with open(mesh_path, "r", encoding="utf-8") as f:
        reader = csv_lib.reader(f)

        for line_no, row in enumerate(reader, start=1):
            row = [col.strip() for col in row]

            if not row or all(col == "" for col in row):
                continue

            if row[0].startswith("#"):
                continue

            if row[0] == "node_id":
                mode = "nodes"
                continue

            if row[0] == "elem_id":
                mode = "elements"
                element_header = row
                continue

            if mode == "nodes":
                if len(row) < 4:
                    raise ValueError(f"line {line_no} node row must contain node_id,x,y,z: {row!r}")
                nodes.append(
                    Node3D(
                        id=_integer_field(
                            row[0], reader_name="read_mixed3d", mesh_path=mesh_path,
                            line_no=line_no, field="node_id",
                        ),
                        x=_numeric_field(
                            row[1], reader_name="read_mixed3d", mesh_path=mesh_path,
                            line_no=line_no, field="x",
                        ),
                        y=_numeric_field(
                            row[2], reader_name="read_mixed3d", mesh_path=mesh_path,
                            line_no=line_no, field="y",
                        ),
                        z=_numeric_field(
                            row[3], reader_name="read_mixed3d", mesh_path=mesh_path,
                            line_no=line_no, field="z",
                        ),
                    )
                )

            elif mode == "elements":
                if any(value != "" for value in row[len(element_header):]):
                    raise ValueError(
                        f"line {line_no} element row has nonempty trailing field beyond header"
                    )
                values = _row_by_header(element_header, row)
                elem_id = _integer_field(
                    values["elem_id"], reader_name="read_mixed3d", mesh_path=mesh_path,
                    line_no=line_no, field="elem_id",
                )
                elem_type = _canonical_mixed3d_element_type(values.get("type", ""))
                node_count = _mixed3d_node_count(elem_type)
                node_ids = _mixed3d_node_ids(
                    values, elem_type, node_count, mesh_path, line_no
                )
                props = _mixed3d_element_props(
                    values, materials_dict, mesh_path, line_no
                )

                elements.append(
                    Element3D(
                        id=elem_id,
                        node_ids=node_ids,
                        type=elem_type,
                        props=props,
                    )
                )

            else:
                raise ValueError(f"data row before a recognized header at line {line_no}: {row!r}")

    if not nodes:
        raise ValueError("mixed 3D mesh csv has no nodes")
    if not elements:
        raise ValueError("mixed 3D mesh csv has no elements")

    return HexMesh3D(nodes=nodes, elements=elements)


def _row_by_header(header: List[str], row: List[str]) -> Dict[str, str]:
    return {
        name: row[index] if index < len(row) else ""
        for index, name in enumerate(header)
    }


def _canonical_mixed3d_element_type(raw_type: str) -> str:
    mapping = {
        "hex8": "Hex8",
        "hex20": "Hex20",
        "tet4": "Tet4",
        "tet10": "Tet10",
    }
    normalized = raw_type.strip().lower()
    if normalized not in mapping:
        raise ValueError(f"unsupported mixed 3D element type: {raw_type!r}")
    return mapping[normalized]


def _mixed3d_node_count(elem_type: str) -> int:
    counts = {"Hex8": 8, "Hex20": 20, "Tet4": 4, "Tet10": 10}
    if elem_type not in counts:
        raise ValueError(f"unsupported mixed 3D element type: {elem_type!r}")
    return counts[elem_type]


def _mixed3d_node_ids(
    values: Dict[str, str],
    elem_type: str,
    node_count: int,
    mesh_path: str,
    line_no: int,
) -> List[int]:
    node_ids: List[int] = []
    for index in range(1, node_count + 1):
        value = values.get(f"node{index}", "")
        if value == "":
            raise ValueError(f"line {line_no} {elem_type} row is missing node{index}")
        node_ids.append(
            _integer_field(
                value,
                reader_name="read_mixed3d",
                mesh_path=mesh_path,
                line_no=line_no,
                field=f"node{index}",
            )
        )

    for index in range(node_count + 1, 21):
        if values.get(f"node{index}", "") != "":
            raise ValueError(f"line {line_no} {elem_type} row has extra node{index}")

    for name, value in values.items():
        node_index = name[4:]
        if (
            name.startswith("node")
            and node_index.isdigit()
            and int(node_index) > 20
            and value != ""
        ):
            raise ValueError(f"line {line_no} {elem_type} row has extra node{node_index}")

    return node_ids


def _mixed3d_element_props(
    values: Dict[str, str],
    materials_dict: Dict[int, Dict[str, str]],
    mesh_path: str,
    line_no: int,
) -> Dict[str, object]:
    material_id = values.get("material_id", "")
    if str(material_id).strip() == "":
        return {}
    parsed_material_id = _integer_field(
        material_id,
        reader_name="read_mixed3d",
        mesh_path=mesh_path,
        line_no=line_no,
        field="material_id",
    )
    return _solid_material_props(
        parsed_material_id,
        materials_dict,
    )


def _solid_material_props(
    material_id: int | str | None,
    materials_dict: Dict[int, Dict[str, str]],
) -> Dict[str, object]:
    """Build optional solid material properties without inventing missing values."""
    if material_id is None or str(material_id).strip() == "":
        return {}

    mid = int(material_id)
    props: Dict[str, object] = {"material_id": mid}
    if not materials_dict:
        return props

    mat_row = materials_dict.get(mid)
    if mat_row is not None:
        raw_E = _get_float_from_material(mat_row, ["E"])
        raw_nu = _get_float_from_material(mat_row, ["nu", "poisson"])
        raw_rho = _get_float_from_material(mat_row, ["rho"])
        if raw_E is not None:
            props["E"] = raw_E
        if raw_nu is not None:
            props["nu"] = raw_nu
        if raw_rho is not None:
            props["rho"] = raw_rho
    return props


def read_hex8(
    mesh_path: str,
    material_path: Optional[str] = None,
) -> HexMesh3D:
    """Read a Hex8 mesh CSV with optional materials."""

    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials_dict = read(material_path)

    nodes: List[Node3D] = []
    elements: List[Element3D] = []

    mode: Optional[str] = None

    with open(mesh_path, "r", encoding="utf-8") as f:
        reader = csv_lib.reader(f)

        for line_no, row in enumerate(reader, start=1):
            row = [col.strip() for col in row]

            if not row or all(col == "" for col in row):
                continue

            if row[0].startswith("#"):
                continue

            # 节点表头
            if row[0] == "node_id":
                mode = "nodes"
                continue

            # 单元表头
            if row[0] == "elem_id":
                mode = "elements"
                continue

            if mode == "nodes":
                if len(row) < 4:
                    raise ValueError(
                        f"Hex8 mesh CSV {mesh_path!r} line {line_no} node row requires "
                        f"node_id,x,y,z; got {row!r}"
                    )
                node_id = _integer_field(
                    row[0], reader_name="read_hex8", mesh_path=mesh_path,
                    line_no=line_no, field="node_id",
                )
                x = _numeric_field(
                    row[1], reader_name="read_hex8", mesh_path=mesh_path,
                    line_no=line_no, field="x",
                )
                y = _numeric_field(
                    row[2], reader_name="read_hex8", mesh_path=mesh_path,
                    line_no=line_no, field="y",
                )
                z = _numeric_field(
                    row[3], reader_name="read_hex8", mesh_path=mesh_path,
                    line_no=line_no, field="z",
                )
                nodes.append(Node3D(id=node_id, x=x, y=y, z=z))

            elif mode == "elements":
                if len(row) < 10:
                    raise ValueError(
                        f"Hex8 mesh CSV {mesh_path!r} line {line_no} element row requires "
                        f"elem_id,node1..node8,material_id; got {row!r}"
                    )
                elem_id = _integer_field(
                    row[0], reader_name="read_hex8", mesh_path=mesh_path,
                    line_no=line_no, field="elem_id",
                )
                node_ids = [
                    _integer_field(
                        row[index], reader_name="read_hex8", mesh_path=mesh_path,
                        line_no=line_no, field=f"node{index}",
                    )
                    for index in range(1, 9)
                ]
                mid = _integer_field(
                    row[9], reader_name="read_hex8", mesh_path=mesh_path,
                    line_no=line_no, field="material_id",
                )

                props = _solid_material_props(mid, materials_dict)

                elements.append(
                    Element3D(
                        id=elem_id,
                        node_ids=node_ids,
                        type="Hex8",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"Hex8 mesh CSV {mesh_path!r} line {line_no} has data before "
                    f"a recognized node_id or elem_id header: {row!r}"
                )

    if not nodes:
        raise ValueError(f"Hex8 mesh CSV {mesh_path!r} contains no node rows")
    if not elements:
        raise ValueError(f"Hex8 mesh CSV {mesh_path!r} contains no element rows")

    return HexMesh3D(nodes=nodes, elements=elements)


def read_tet4(
    mesh_path: str,
    material_path: Optional[str] = None,
) -> TetMesh3D:
    """Read a Tet4 mesh CSV with optional materials."""
    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials_dict = read(material_path)

    nodes: List[Node3D] = []
    elements: List[Element3D] = []

    mode: Optional[str] = None

    with open(mesh_path, "r", encoding="utf-8") as f:
        reader = csv_lib.reader(f)

        for line_no, row in enumerate(reader, start=1):
            row = [col.strip() for col in row]

            if not row or all(col == "" for col in row):
                continue

            if row[0].startswith("#"):
                continue

            if row[0] == "node_id":
                mode = "nodes"
                continue

            if row[0] == "elem_id":
                mode = "elements"
                continue

            if mode == "nodes":
                if len(row) < 4:
                    raise ValueError(
                        f"Tet4 mesh CSV {mesh_path!r} line {line_no} node row requires "
                        f"node_id,x,y,z; got {row!r}"
                    )
                node_id = _integer_field(
                    row[0], reader_name="read_tet4", mesh_path=mesh_path,
                    line_no=line_no, field="node_id",
                )
                x = _numeric_field(
                    row[1], reader_name="read_tet4", mesh_path=mesh_path,
                    line_no=line_no, field="x",
                )
                y = _numeric_field(
                    row[2], reader_name="read_tet4", mesh_path=mesh_path,
                    line_no=line_no, field="y",
                )
                z = _numeric_field(
                    row[3], reader_name="read_tet4", mesh_path=mesh_path,
                    line_no=line_no, field="z",
                )
                nodes.append(Node3D(id=node_id, x=x, y=y, z=z))

            elif mode == "elements":
                if len(row) < 6:
                    raise ValueError(
                        f"Tet4 mesh CSV {mesh_path!r} line {line_no} element row requires "
                        f"elem_id,node1..node4,material_id; got {row!r}"
                    )
                elem_id = _integer_field(
                    row[0], reader_name="read_tet4", mesh_path=mesh_path,
                    line_no=line_no, field="elem_id",
                )
                node_ids = [
                    _integer_field(
                        row[index], reader_name="read_tet4", mesh_path=mesh_path,
                        line_no=line_no, field=f"node{index}",
                    )
                    for index in range(1, 5)
                ]
                mid = _integer_field(
                    row[5], reader_name="read_tet4", mesh_path=mesh_path,
                    line_no=line_no, field="material_id",
                )

                props = _solid_material_props(mid, materials_dict)

                elements.append(
                    Element3D(
                        id=elem_id,
                        node_ids=node_ids,
                        type="Tet4",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"Tet4 mesh CSV {mesh_path!r} line {line_no} has data before "
                    f"a recognized node_id or elem_id header: {row!r}"
                )

    if not nodes:
        raise ValueError(f"Tet4 mesh CSV {mesh_path!r} contains no node rows")
    if not elements:
        raise ValueError(f"Tet4 mesh CSV {mesh_path!r} contains no element rows")

    return TetMesh3D(nodes=nodes, elements=elements)
