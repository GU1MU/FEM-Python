from __future__ import annotations

import csv as csv_lib
from typing import Dict, List, Optional

from ..core.mesh import BeamMesh2D, Element2D, Element3D, HexMesh3D, Node2D, Node3D, PlaneMesh2D, TetMesh3D, TrussMesh2D
from .materials import _get_float_from_material, read


def read_truss2d(
    mesh_path: str,
    material_path: Optional[str] = None,
) -> TrussMesh2D:
    """Read a Truss2D mesh CSV with optional materials."""

    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
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

            # 节点表头
            if row[0] == "node_id":
                mode = "nodes"
                continue

            # 单元表头
            if row[0] == "elem_id":
                mode = "elements"
                continue

            if mode == "nodes":
                if len(row) < 3:
                    raise ValueError(f"第 {line_no} 行节点格式错误: {row!r}")
                nid = int(row[0])
                x = float(row[1])
                y = float(row[2])
                nodes.append(Node2D(id=nid, x=x, y=y))

            elif mode == "elements":
                if len(row) < 5:
                    raise ValueError(f"第 {line_no} 行单元格式错误: {row!r}")
                eid = int(row[0])
                ni = int(row[1])
                nj = int(row[2])
                area = float(row[3])
                mid = int(row[4])

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
                    Element2D(
                        id=eid,
                        node_ids=[ni, nj],
                        type="Truss2D",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"在未识别出表头前遇到数据行（第 {line_no} 行）: {row!r}"
                )

    if not nodes:
        raise ValueError("mesh csv 中没有读到节点")
    if not elements:
        raise ValueError("mesh csv 中没有读到单元")

    return TrussMesh2D(nodes=nodes, elements=elements)


def read_beam2d(
    mesh_path: str,
    material_path: Optional[str] = None,
) -> BeamMesh2D:
    """Read a Beam2D mesh CSV with optional materials."""

    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
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

            # 表头：节点
            if row[0] == "node_id":
                mode = "nodes"
                continue

            # 表头：单元
            if row[0] == "elem_id":
                mode = "elements"
                continue

            if mode == "nodes":
                if len(row) < 3:
                    raise ValueError(f"第 {line_no} 行节点格式错误: {row!r}")
                nid = int(row[0])
                x = float(row[1])
                y = float(row[2])
                nodes.append(Node2D(id=nid, x=x, y=y))

            elif mode == "elements":
                # elem_id,node_i,node_j,area,Izz,material_id
                if len(row) < 6:
                    raise ValueError(f"第 {line_no} 行 Beam 单元格式错误: {row!r}")
                eid = int(row[0])
                ni = int(row[1])
                nj = int(row[2])
                area = float(row[3])
                Izz = float(row[4])
                mid = int(row[5])

                props: Dict[str, object] = {
                    "area": area,        # A
                    "Izz": Izz,          # I
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
                    Element2D(
                        id=eid,
                        node_ids=[ni, nj],
                        type="Beam2D",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"在未识别出表头前遇到数据行（第 {line_no} 行）: {row!r}"
                )

    if not nodes:
        raise ValueError("beam mesh csv 中没有读到节点")
    if not elements:
        raise ValueError("beam mesh csv 中没有读到单元")

    return BeamMesh2D(nodes=nodes, elements=elements)


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
                    raise ValueError(f"第 {line_no} 行节点格式错误: {row!r}")
                nid = int(row[0])
                x = float(row[1])
                y = float(row[2])
                nodes.append(Node2D(id=nid, x=x, y=y))

            elif mode == "elements":
                # elem_id,node1,node2,node3,thickness,material_id
                if len(row) < 6:
                    raise ValueError(f"第 {line_no} 行 Tri3 单元格式错误: {row!r}")
                eid = int(row[0])
                n1 = int(row[1])
                n2 = int(row[2])
                n3 = int(row[3])
                thickness = float(row[4])
                mid = int(row[5])

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
                                f"材料 {mid} 未找到 E/nu 信息，mat_row={mat_row}"
                            )

                        props["E"] = E_val
                        props["nu"] = nu_val
                        if rho_val is not None:
                            props["rho"] = rho_val

                elements.append(
                    Element2D(
                        id=eid,
                        node_ids=[n1, n2, n3],
                        type="Tri3Plane",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"在未识别出表头前遇到数据行（第 {line_no} 行）: {row!r}"
                )

    if not nodes:
        raise ValueError("plane tri3 mesh csv 中没有读到节点")
    if not elements:
        raise ValueError("plane tri3 mesh csv 中没有读到单元")

    return PlaneMesh2D(nodes=nodes, elements=elements)


def read_mixed3d(
    mesh_path: str,
    material_path: Optional[str] = None,
) -> HexMesh3D:
    """Read a mixed 3D mesh CSV with Hex8 and Tet4 elements."""

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
                        id=int(row[0]),
                        x=float(row[1]),
                        y=float(row[2]),
                        z=float(row[3]),
                    )
                )

            elif mode == "elements":
                values = _row_by_header(element_header, row)
                elem_id = int(values["elem_id"])
                elem_type = _canonical_mixed3d_element_type(values.get("type", ""))
                node_count = _mixed3d_node_count(elem_type)
                node_ids = _mixed3d_node_ids(values, elem_type, node_count, line_no)
                props = _mixed3d_element_props(values, materials_dict)

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
    normalized = raw_type.strip().lower()
    if normalized == "hex8":
        return "Hex8"
    if normalized == "tet4":
        return "Tet4"
    raise ValueError(f"unsupported mixed 3D element type: {raw_type!r}")


def _mixed3d_node_count(elem_type: str) -> int:
    if elem_type == "Hex8":
        return 8
    if elem_type == "Tet4":
        return 4
    raise ValueError(f"unsupported mixed 3D element type: {elem_type!r}")


def _mixed3d_node_ids(
    values: Dict[str, str],
    elem_type: str,
    node_count: int,
    line_no: int,
) -> List[int]:
    node_ids: List[int] = []
    for index in range(1, node_count + 1):
        value = values.get(f"node{index}", "")
        if value == "":
            raise ValueError(f"line {line_no} {elem_type} row is missing node{index}")
        node_ids.append(int(value))

    for index in range(node_count + 1, 9):
        if values.get(f"node{index}", "") != "":
            raise ValueError(f"line {line_no} {elem_type} row has extra node{index}")

    return node_ids


def _mixed3d_element_props(
    values: Dict[str, str],
    materials_dict: Dict[int, Dict[str, str]],
) -> Dict[str, object]:
    raw_mid = values.get("material_id", "")
    if raw_mid == "":
        return {}

    mid = int(raw_mid)
    props: Dict[str, object] = {"material_id": mid}
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
                    raise ValueError(f"第 {line_no} 行节点格式错误: {row!r}")
                nid = int(row[0])
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                nodes.append(Node3D(id=nid, x=x, y=y, z=z))

            elif mode == "elements":
                if len(row) < 10:
                    raise ValueError(f"第 {line_no} 行 Hex8 单元格式错误: {row!r}")
                eid = int(row[0])
                n1 = int(row[1])
                n2 = int(row[2])
                n3 = int(row[3])
                n4 = int(row[4])
                n5 = int(row[5])
                n6 = int(row[6])
                n7 = int(row[7])
                n8 = int(row[8])
                mid = int(row[9])

                props: Dict[str, object] = {
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
                        id=eid,
                        node_ids=[n1, n2, n3, n4, n5, n6, n7, n8],
                        type="Hex8",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"在未识别出表头前遇到数据行（第 {line_no} 行）: {row!r}"
                )

    if not nodes:
        raise ValueError("mesh csv 中没有读到节点")
    if not elements:
        raise ValueError("mesh csv 中没有读到单元")

    return HexMesh3D(nodes=nodes, elements=elements)


def read_tet4(
    mesh_path: str,
    material_path: Optional[str] = None,
) -> TetMesh3D:
    """Read a Tet4 mesh CSV with optional materials."""
    import numpy as np

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
                    raise ValueError(f"第 {line_no} 行节点格式错误: {row!r}")
                nid = int(row[0])
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                nodes.append(Node3D(id=nid, x=x, y=y, z=z))

            elif mode == "elements":
                if len(row) < 6:
                    raise ValueError(f"第 {line_no} 行 Tet4 单元格式错误: {row!r}")
                eid = int(row[0])
                n1 = int(row[1])
                n2 = int(row[2])
                n3 = int(row[3])
                n4 = int(row[4])
                mid = int(row[5])

                props: Dict[str, object] = {
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
                        id=eid,
                        node_ids=[n1, n2, n3, n4],
                        type="Tet4",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"在未识别出表头前遇到数据行（第 {line_no} 行）: {row!r}"
                )

    if not nodes:
        raise ValueError("mesh csv 中没有读到节点")
    if not elements:
        raise ValueError("mesh csv 中没有读到单元")

    return TetMesh3D(nodes=nodes, elements=elements)
