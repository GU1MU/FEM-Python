from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ..core.mesh import Element2D, Element3D, HexMesh3D, Node2D, Node3D, PlaneMesh2D, TetMesh3D


_MIXED2D_TYPES = {
    "CPS3": ("Tri3Plane", 3, "linear"),
    "CPE3": ("Tri3Plane", 3, "linear"),
    "CPS4": ("Quad4Plane", 4, "linear"),
    "CPE4": ("Quad4Plane", 4, "linear"),
    "CPS4R": ("Quad4Plane", 4, "linear"),
    "CPE4R": ("Quad4Plane", 4, "linear"),
    "CPS6": ("Tri6Plane", 6, "quadratic"),
    "CPE6": ("Tri6Plane", 6, "quadratic"),
    "CPS8": ("Quad8Plane", 8, "quadratic"),
    "CPE8": ("Quad8Plane", 8, "quadratic"),
    "CPS8R": ("Quad8Plane", 8, "quadratic"),
    "CPE8R": ("Quad8Plane", 8, "quadratic"),
}


def _split_nums(line: str) -> List[str]:
    parts = [p.strip() for p in line.strip().split(",")]
    return [p for p in parts if p]


def _keyword_type(kw: str) -> str | None:
    for part in [p.strip() for p in kw.split(",")]:
        if part.startswith("TYPE="):
            return part.split("=", 1)[1].strip().upper()
    return None


def _normalize_plane_type(plane_type: Optional[str], elem_type: str) -> str:
    if plane_type is None:
        return "strain" if elem_type.upper().startswith("CPE") else "stress"
    pt = str(plane_type).lower()
    if pt.startswith("stress"):
        return "stress"
    if pt.startswith("strain"):
        return "strain"
    raise ValueError("plane_type must be 'stress' or 'strain'")


def _signed_area_2d(node_lookup: Dict[int, Node2D], node_ids: List[int]) -> float:
    corners = node_ids[:3] if len(node_ids) in (3, 6) else node_ids[:4]
    area = 0.0
    for i, node_id in enumerate(corners):
        current = node_lookup[node_id]
        nxt = node_lookup[corners[(i + 1) % len(corners)]]
        area += current.x * nxt.y - nxt.x * current.y
    return 0.5 * area


def _fix_plane_orientation(elem: Element2D, node_lookup: Dict[int, Node2D]) -> None:
    area = _signed_area_2d(node_lookup, elem.node_ids)
    if area >= 0.0:
        return
    if elem.type == "Tri3Plane":
        n1, n2, n3 = elem.node_ids
        elem.node_ids = [n1, n3, n2]
    elif elem.type == "Tri6Plane":
        n1, n2, n3, n4, n5, n6 = elem.node_ids
        elem.node_ids = [n1, n3, n2, n6, n5, n4]
    elif elem.type == "Quad4Plane":
        n1, n2, n3, n4 = elem.node_ids
        elem.node_ids = [n1, n4, n3, n2]
    elif elem.type == "Quad8Plane":
        n1, n2, n3, n4, n5, n6, n7, n8 = elem.node_ids
        elem.node_ids = [n1, n4, n3, n2, n8, n7, n6, n5]


def read_tri3(
    inp_path: str,
    default_thickness: float = 1.0,
    plane_type: Optional[str] = None,
) -> PlaneMesh2D:
    """Read a Tri3 plane mesh from Abaqus .inp files."""
    nodes: List[Node2D] = []
    elements: List[Element2D] = []

    node_lookup: Dict[int, Node2D] = {}

    in_node_block = False
    in_elem_block = False
    elem_abaqus_type: Optional[str] = None  # "CPS3" or "CPE3"

    def _parse_keyword(line: str) -> str:
        return line.strip()

    def _parse_csv_like_numbers(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip().split(",")]
        parts = [p for p in parts if p != ""]
        return parts

    def _infer_plane_type_from_elem_type(et: str) -> str:
        etu = et.upper()
        if etu.startswith("CPS3"):
            return "stress"
        if etu.startswith("CPE3"):
            return "strain"
        return "stress"

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "" or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = _parse_keyword(line).upper()

                in_node_block = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem_block = False
                    elem_abaqus_type = None

                    if "TYPE=" in kw:
                        parts = [p.strip() for p in kw.split(",")]
                        etype = None
                        for p in parts:
                            if p.startswith("TYPE="):
                                etype = p.split("=", 1)[1].strip()
                                break
                        if etype in ("CPS3", "CPE3"):
                            in_elem_block = True
                            elem_abaqus_type = etype
                    continue

                if not in_node_block:
                    pass
                if not kw.startswith("*ELEMENT"):
                    in_elem_block = False
                    elem_abaqus_type = None

                continue

            # 数据行
            if in_node_block:
                parts = _parse_csv_like_numbers(line)
                if len(parts) < 3:
                    continue
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                node = Node2D(id=nid, x=x, y=y)
                nodes.append(node)
                node_lookup[nid] = node

            elif in_elem_block and elem_abaqus_type is not None:
                parts = _parse_csv_like_numbers(line)
                if len(parts) < 4:
                    continue
                eid = int(parts[0])
                n1 = int(parts[1])
                n2 = int(parts[2])
                n3 = int(parts[3])

                if plane_type is None:
                    pt = _infer_plane_type_from_elem_type(elem_abaqus_type)
                else:
                    pt = str(plane_type).lower()
                    if pt.startswith("stress"):
                        pt = "stress"
                    elif pt.startswith("strain"):
                        pt = "strain"
                    else:
                        raise ValueError("plane_type 必须是 'stress' 或 'strain'")

                props: Dict[str, any] = {
                    "thickness": float(default_thickness),
                    "plane_type": pt,
                }

                elem = Element2D(
                    id=eid,
                    node_ids=[n1, n2, n3],
                    type="Tri3Plane",
                    props=props,
                )
                elements.append(elem)

    if not nodes:
        raise ValueError(f"未在 {inp_path} 中解析到 *Node 数据")
    if not elements:
        raise ValueError(f"未在 {inp_path} 中解析到 CPS3/CPE3 的 *Element 数据")

    return PlaneMesh2D(nodes=nodes, elements=elements)


def read_tri6(
    inp_path: str,
    default_thickness: float = 1.0,
    plane_type: Optional[str] = None,
    fix_orientation: bool = True,
) -> PlaneMesh2D:
    """Read Tri6 plane mesh (CPS6/CPE6) from Abaqus INP file."""
    nodes: List[Node2D] = []
    elements: List[Element2D] = []
    node_lookup: Dict[int, Node2D] = {}
    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                kw = line.strip().upper()
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    elem_abaqus_type = _keyword_type(kw)
                    in_elem = elem_abaqus_type in ("CPS6", "CPE6")
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = _split_nums(line)
                if len(parts) < 3:
                    continue
                node = Node2D(id=int(parts[0]), x=float(parts[1]), y=float(parts[2]))
                nodes.append(node)
                node_lookup[node.id] = node
                continue

            if in_elem and elem_abaqus_type is not None:
                parts = _split_nums(line)
                if len(parts) < 7:
                    continue
                props: Dict[str, Any] = {
                    "thickness": float(default_thickness),
                    "plane_type": _normalize_plane_type(plane_type, elem_abaqus_type),
                }
                elements.append(
                    Element2D(
                        id=int(parts[0]),
                        node_ids=[int(value) for value in parts[1:7]],
                        type="Tri6Plane",
                        props=props,
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No CPS6/CPE6 *Element data found in {inp_path}")
    if fix_orientation:
        for elem in elements:
            _fix_plane_orientation(elem, node_lookup)
    return PlaneMesh2D(nodes=nodes, elements=elements)


def read_quad4(
    inp_path: str,
    default_thickness: float = 1.0,
    plane_type: Optional[str] = None,
    fix_orientation: bool = True,
    enforce_parallelogram: bool = False,
    tol: float = 1e-10,
) -> PlaneMesh2D:
    """Read Quad4 plane mesh (CPS4/CPE4) from Abaqus INP file."""
    nodes: List[Node2D] = []
    elements: List[Element2D] = []
    node_lookup: Dict[int, Node2D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    def split_nums(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip().split(",")]
        return [p for p in parts if p]

    def infer_plane_type(et: str) -> str:
        etu = et.upper()
        if etu.startswith("CPS4"):
            return "stress"
        if etu.startswith("CPE4"):
            return "strain"
        return "stress"

    def signed_area_quad(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], p4: Tuple[float, float]) -> float:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        return 0.5 * (
            x1 * y2 - x2 * y1 +
            x2 * y3 - x3 * y2 +
            x3 * y4 - x4 * y3 +
            x4 * y1 - x1 * y4
        )

    def is_parallelogram(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        d1 = (x1 + x3 - x2 - x4, y1 + y3 - y2 - y4)  # diag midpoints: p1+p3 == p2+p4
        return (d1[0] * d1[0] + d1[1] * d1[1]) <= tol

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = line.strip().upper()
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    parts = [p.strip() for p in kw.split(",")]
                    et = None
                    for p in parts:
                        if p.startswith("TYPE="):
                            et = p.split("=", 1)[1].strip()
                            break
                    if et in ("CPS4", "CPE4", "CPS4R", "CPE4R"):
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = split_nums(line)
                if len(parts) < 3:
                    continue
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                n = Node2D(id=nid, x=x, y=y)
                nodes.append(n)
                node_lookup[nid] = n
                continue

            if in_elem and elem_abaqus_type is not None:
                parts = split_nums(line)
                if len(parts) < 5:
                    continue
                eid = int(parts[0])
                n1, n2, n3, n4 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])

                pt = infer_plane_type(elem_abaqus_type) if plane_type is None else str(plane_type).lower()
                if pt.startswith("stress"):
                    pt = "stress"
                elif pt.startswith("strain"):
                    pt = "strain"
                else:
                    raise ValueError("plane_type must be 'stress' or 'strain'")

                props: Dict[str, any] = {
                    "thickness": float(default_thickness),
                    "plane_type": pt,
                }

                elements.append(
                    Element2D(
                        id=eid,
                        node_ids=[n1, n2, n3, n4],
                        type="Quad4Plane",
                        props=props,
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No CPS4/CPE4 *Element data found in {inp_path}")

    if enforce_parallelogram or fix_orientation:
        for e in elements:
            n1, n2, n3, n4 = e.node_ids
            try:
                p1 = (node_lookup[n1].x, node_lookup[n1].y)
                p2 = (node_lookup[n2].x, node_lookup[n2].y)
                p3 = (node_lookup[n3].x, node_lookup[n3].y)
                p4 = (node_lookup[n4].x, node_lookup[n4].y)
            except KeyError as ex:
                raise KeyError(f"Element {e.id} references missing node {ex.args[0]}")

            if enforce_parallelogram and not is_parallelogram(p1, p2, p3, p4):
                raise ValueError(f"Element {e.id} is not a parallelogram by tolerance {tol}")

            if fix_orientation:
                A = signed_area_quad(p1, p2, p3, p4)
                if A < 0.0:
                    e.node_ids = [n1, n4, n3, n2]

    return PlaneMesh2D(nodes=nodes, elements=elements)


def read_quad8(
    inp_path: str,
    default_thickness: float = 1.0,
    plane_type: Optional[str] = None,
    fix_orientation: bool = True,
) -> PlaneMesh2D:
    """Read Quad8 plane mesh (CPS8/CPE8) from Abaqus INP file."""
    nodes: List[Node2D] = []
    elements: List[Element2D] = []
    node_lookup: Dict[int, Node2D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    def split_nums(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip().split(",")]
        return [p for p in parts if p]

    def infer_plane_type(et: str) -> str:
        etu = et.upper()
        if etu.startswith("CPS8"):
            return "stress"
        if etu.startswith("CPE8"):
            return "strain"
        return "stress"

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = line.strip().upper()
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    parts = [p.strip() for p in kw.split(",")]
                    et = None
                    for p in parts:
                        if p.startswith("TYPE="):
                            et = p.split("=", 1)[1].strip()
                            break
                    if et in ("CPS8", "CPE8", "CPS8R", "CPE8R"):
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = split_nums(line)
                if len(parts) < 3:
                    continue
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                n = Node2D(id=nid, x=x, y=y)
                nodes.append(n)
                node_lookup[nid] = n
                continue

            if in_elem and elem_abaqus_type is not None:
                parts = split_nums(line)
                if len(parts) < 9:
                    continue
                eid = int(parts[0])
                nids = [int(p) for p in parts[1:9]]

                pt = infer_plane_type(elem_abaqus_type) if plane_type is None else str(plane_type).lower()
                if pt.startswith("stress"):
                    pt = "stress"
                elif pt.startswith("strain"):
                    pt = "strain"
                else:
                    raise ValueError("plane_type must be 'stress' or 'strain'")

                props: Dict[str, any] = {
                    "thickness": float(default_thickness),
                    "plane_type": pt,
                }

                elements.append(
                    Element2D(
                        id=eid,
                        node_ids=nids,
                        type="Quad8Plane",
                        props=props,
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No CPS8/CPE8 *Element data found in {inp_path}")

    if fix_orientation:
        for e in elements:
            try:
                n1, n2, n3, n4 = (node_lookup[e.node_ids[i]] for i in range(4))
            except KeyError as ex:
                raise KeyError(f"Element {e.id} references missing node {ex.args[0]}")
            area = 0.5 * (
                n1.x * n2.y - n2.x * n1.y
                + n2.x * n3.y - n3.x * n2.y
                + n3.x * n4.y - n4.x * n3.y
                + n4.x * n1.y - n1.x * n4.y
            )
            if area < 0.0:
                if len(e.node_ids) != 8:
                    raise ValueError(f"Element {e.id} expected 8 nodes for orientation fix, got {len(e.node_ids)}")
                n1_id, n2_id, n3_id, n4_id, n5_id, n6_id, n7_id, n8_id = e.node_ids
                e.node_ids = [n1_id, n4_id, n3_id, n2_id, n8_id, n7_id, n6_id, n5_id]

    return PlaneMesh2D(nodes=nodes, elements=elements)


def read_mixed2d(
    inp_path: str,
    default_thickness: float = 1.0,
    plane_type: Optional[str] = None,
    fix_orientation: bool = True,
) -> PlaneMesh2D:
    """Read a same-order mixed 2D plane mesh from Abaqus INP file."""
    nodes: List[Node2D] = []
    elements: List[Element2D] = []
    node_lookup: Dict[int, Node2D] = {}
    orders: set[str] = set()
    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                kw = line.strip().upper()
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    elem_abaqus_type = _keyword_type(kw)
                    in_elem = elem_abaqus_type in _MIXED2D_TYPES
                    if (
                        elem_abaqus_type is not None
                        and not in_elem
                        and elem_abaqus_type.startswith(("CPS", "CPE"))
                    ):
                        raise ValueError(
                            f"unsupported mixed 2D element type: {elem_abaqus_type}"
                        )
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = _split_nums(line)
                if len(parts) < 3:
                    continue
                node = Node2D(id=int(parts[0]), x=float(parts[1]), y=float(parts[2]))
                nodes.append(node)
                node_lookup[node.id] = node
                continue

            if in_elem and elem_abaqus_type is not None:
                local_type, node_count, order = _MIXED2D_TYPES[elem_abaqus_type]
                parts = _split_nums(line)
                if len(parts) < node_count + 1:
                    continue
                orders.add(order)
                props: Dict[str, Any] = {
                    "thickness": float(default_thickness),
                    "plane_type": _normalize_plane_type(plane_type, elem_abaqus_type),
                }
                elements.append(
                    Element2D(
                        id=int(parts[0]),
                        node_ids=[int(value) for value in parts[1:node_count + 1]],
                        type=local_type,
                        props=props,
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No supported 2D *Element data found in {inp_path}")
    if len(orders) > 1:
        raise ValueError("read_mixed2d requires elements with the same polynomial order")
    if fix_orientation:
        for elem in elements:
            _fix_plane_orientation(elem, node_lookup)
    return PlaneMesh2D(nodes=nodes, elements=elements)


def read_tet10(inp_path: str) -> TetMesh3D:
    """Read a Tet10 3D mesh from Abaqus .inp file (C3D10 elements).

    Node ordering (Abaqus convention):
        Corner nodes:  1-4
        Edge midnodes: 5=edge(1,2), 6=edge(3,4), 7=edge(1,4),
                       8=edge(1,3), 9=edge(2,4), 10=edge(2,3)
    """
    nodes: List[Node3D] = []
    elements: List[Element3D] = []
    node_lookup: Dict[int, Node3D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None
    pending_elem_parts: List[str] = []

    def split_nums(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip().split(",")]
        return [p for p in parts if p]

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                if in_elem and pending_elem_parts:
                    raise ValueError(
                        f"Incomplete C3D10 connectivity record in {inp_path}: {pending_elem_parts}"
                    )
                kw = line.strip().upper()
                if kw.startswith("*ASSEMBLY"):
                    # This reader consumes the part-level Tet10 mesh only.
                    # Assembly sections may contain reference points that reuse node ids.
                    break
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    pending_elem_parts = []
                    parts = [p.strip() for p in kw.split(",")]
                    et = None
                    for p in parts:
                        if p.startswith("TYPE="):
                            et = p.split("=", 1)[1].strip()
                            break
                    if et in ("C3D10", "C3D10T"):
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = split_nums(line)
                if len(parts) < 4:
                    continue
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                n = Node3D(id=nid, x=x, y=y, z=z)
                nodes.append(n)
                node_lookup[nid] = n
                continue

            if in_elem and elem_abaqus_type is not None:
                pending_elem_parts.extend(split_nums(line))
                while len(pending_elem_parts) >= 11:
                    eid = int(pending_elem_parts[0])
                    nids = [int(p) for p in pending_elem_parts[1:11]]
                    pending_elem_parts = pending_elem_parts[11:]

                    elements.append(
                        Element3D(
                            id=eid,
                            node_ids=nids,
                            type="Tet10",
                        )
                    )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if pending_elem_parts:
        raise ValueError(
            f"Incomplete C3D10 connectivity record at end of file {inp_path}: {pending_elem_parts}"
        )
    if not elements:
        raise ValueError(f"No C3D10 *Element data found in {inp_path}")

    from ..elements.tetrahedron import tet10_gauss_points, tet10_shape_funcs_grads

    for e in elements:
        coords = [node_lookup[nid] for nid in e.node_ids]
        x = np.array([n.x for n in coords], dtype=float)
        y = np.array([n.y for n in coords], dtype=float)
        z = np.array([n.z for n in coords], dtype=float)

        for xi, eta, zeta, _ in tet10_gauss_points():
            _, dN_dxi, dN_deta, dN_dzeta = tet10_shape_funcs_grads(xi, eta, zeta)
            J = np.array([
                [np.sum(dN_dxi * x), np.sum(dN_dxi * y), np.sum(dN_dxi * z)],
                [np.sum(dN_deta * x), np.sum(dN_deta * y), np.sum(dN_deta * z)],
                [np.sum(dN_dzeta * x), np.sum(dN_dzeta * y), np.sum(dN_dzeta * z)],
            ], dtype=float)
            if np.linalg.det(J) <= 0.0:
                raise ValueError(
                    f"Element {e.id} has zero or negative Jacobian determinant. Check node ordering."
                )

    return TetMesh3D(nodes=nodes, elements=elements)


def read_tet4(inp_path: str) -> TetMesh3D:
    """Read a Tet4 3D mesh from Abaqus .inp file (C3D4 / C3D4T elements)."""
    nodes: List[Node3D] = []
    elements: List[Element3D] = []
    node_lookup: Dict[int, Node3D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    def split_nums(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip().split(",")]
        return [p for p in parts if p]

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = line.strip().upper()
                if kw.startswith("*ASSEMBLY"):
                    # This reader consumes the part-level Tet4 mesh only.
                    # Assembly sections may contain reference points that reuse node ids.
                    break
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    parts = [p.strip() for p in kw.split(",")]
                    et = None
                    for p in parts:
                        if p.startswith("TYPE="):
                            et = p.split("=", 1)[1].strip()
                            break
                    if et in ("C3D4", "C3D4T"):
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = split_nums(line)
                if len(parts) < 4:
                    continue
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                n = Node3D(id=nid, x=x, y=y, z=z)
                nodes.append(n)
                node_lookup[nid] = n
                continue

            if in_elem and elem_abaqus_type is not None:
                parts = split_nums(line)
                if len(parts) < 5:
                    continue
                eid = int(parts[0])
                n1, n2, n3, n4 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])

                elements.append(
                    Element3D(
                        id=eid,
                        node_ids=[n1, n2, n3, n4],
                        type="Tet4",
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No C3D4/C3D4T *Element data found in {inp_path}")

    # Check volume (Jacobian determinant) for each element
    for e in elements:
        n1, n2, n3, n4 = (node_lookup[nid] for nid in e.node_ids)
        # Volume = det(J)/6 where J columns are (x2-x1, x3-x1, x4-x1)
        v1 = np.array([n2.x - n1.x, n2.y - n1.y, n2.z - n1.z])
        v2 = np.array([n3.x - n1.x, n3.y - n1.y, n3.z - n1.z])
        v3 = np.array([n4.x - n1.x, n4.y - n1.y, n4.z - n1.z])
        vol = np.dot(v1, np.cross(v2, v3)) / 6.0
        if vol <= 0.0:
            raise ValueError(
                f"Element {e.id} has zero or negative volume "
                f"(nodes: {e.node_ids}). Check node ordering."
            )

    return TetMesh3D(nodes=nodes, elements=elements)


def read_hex8(inp_path: str) -> HexMesh3D:
    """Read a Hex8 3D mesh from Abaqus .inp file (C3D8 elements)."""
    nodes: List[Node3D] = []
    elements: List[Element3D] = []
    node_lookup: Dict[int, Node3D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    def split_nums(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip().split(",")]
        return [p for p in parts if p]

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = line.strip().upper()
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    parts = [p.strip() for p in kw.split(",")]
                    et = None
                    for p in parts:
                        if p.startswith("TYPE="):
                            et = p.split("=", 1)[1].strip()
                            break
                    if et == "C3D8":
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = split_nums(line)
                if len(parts) < 4:
                    continue
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                n = Node3D(id=nid, x=x, y=y, z=z)
                nodes.append(n)
                node_lookup[nid] = n
                continue

            if in_elem and elem_abaqus_type is not None:
                parts = split_nums(line)
                if len(parts) < 9:
                    continue
                eid = int(parts[0])
                nids = [int(p) for p in parts[1:9]]

                elements.append(
                    Element3D(
                        id=eid,
                        node_ids=nids,
                        type="Hex8",
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No C3D8 *Element data found in {inp_path}")

    return HexMesh3D(nodes=nodes, elements=elements)
