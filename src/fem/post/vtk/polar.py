from __future__ import annotations

from typing import Dict, Sequence

from ...core.mesh import Mesh2DProtocol
from .fields import NodalStressCsv, NodalStressCsvRow


def basis(x: float, y: float, center: Sequence[float]):
    """Return cos/sin of polar basis at (x, y)."""
    dx = x - float(center[0])
    dy = y - float(center[1])
    r = (dx * dx + dy * dy) ** 0.5
    if r == 0.0:
        return 1.0, 0.0
    return dx / r, dy / r


def displacement(c: float, s: float, ux: float, uy: float):
    """Return (ur, ut) from (ux, uy)."""
    ur = c * ux + s * uy
    ut = -s * ux + c * uy
    return ur, ut


def stress(c: float, s: float, sig_x: float, sig_y: float, tau_xy: float):
    """Return (sig_r, sig_t, tau_rt) from (sig_x, sig_y, tau_xy)."""
    sig_r = c * c * sig_x + s * s * sig_y + 2.0 * s * c * tau_xy
    sig_t = s * s * sig_x + c * c * sig_y - 2.0 * s * c * tau_xy
    tau_rt = -s * c * sig_x + s * c * sig_y + (c * c - s * s) * tau_xy
    return sig_r, sig_t, tau_rt


def convert_nodal_displacement(
    mesh: Mesh2DProtocol,
    node_disp: Dict[int, Dict[str, float]],
    center: Sequence[float],
) -> Dict[int, Dict[str, float]]:
    """Convert nodal displacement dict into polar components."""
    if len(center) != 2:
        raise ValueError("center must have 2 values")

    node_lookup = {node.id: node for node in mesh.nodes}
    polar_disp: Dict[int, Dict[str, float]] = {}

    for node in mesh.nodes:
        disp = node_disp.get(node.id, {"ux": 0.0, "uy": 0.0, "rz": 0.0})
        c, s = basis(node.x, node.y, center)
        ur, ut = displacement(c, s, float(disp.get("ux", 0.0)), float(disp.get("uy", 0.0)))
        polar_disp[node.id] = {"ux": ur, "uy": ut, "rz": float(disp.get("rz", 0.0))}

    for nid, disp in node_disp.items():
        if nid in polar_disp:
            continue
        node = node_lookup.get(nid)
        if node is None:
            continue
        c, s = basis(node.x, node.y, center)
        ur, ut = displacement(c, s, float(disp.get("ux", 0.0)), float(disp.get("uy", 0.0)))
        polar_disp[nid] = {"ux": ur, "uy": ut, "rz": float(disp.get("rz", 0.0))}

    return polar_disp


def convert_nodal_stress_fields(
    mesh: Mesh2DProtocol,
    nodal_fields: Dict[str, Dict[int, float]],
    center: Sequence[float],
) -> Dict[str, Dict[int, float]]:
    """Convert nodal stress fields to polar components."""
    required = {"sig_x", "sig_y", "tau_xy"}
    polar_names = {"sig_r", "sig_t", "tau_rt"}
    if not required.issubset(nodal_fields) or polar_names.intersection(nodal_fields):
        return nodal_fields

    node_lookup = {node.id: node for node in mesh.nodes}
    new_fields = {name: vals for name, vals in nodal_fields.items() if name not in required}
    sig_r: Dict[int, float] = {}
    sig_t: Dict[int, float] = {}
    tau_rt: Dict[int, float] = {}

    for node in mesh.nodes:
        nid = node.id
        sx = float(nodal_fields["sig_x"].get(nid, 0.0))
        sy = float(nodal_fields["sig_y"].get(nid, 0.0))
        txy = float(nodal_fields["tau_xy"].get(nid, 0.0))
        c, s = basis(node.x, node.y, center)
        sr, st, trt = stress(c, s, sx, sy, txy)
        sig_r[nid] = sr
        sig_t[nid] = st
        tau_rt[nid] = trt

    for nid in nodal_fields["sig_x"]:
        if nid in sig_r:
            continue
        node = node_lookup.get(nid)
        if node is None:
            continue
        sx = float(nodal_fields["sig_x"].get(nid, 0.0))
        sy = float(nodal_fields["sig_y"].get(nid, 0.0))
        txy = float(nodal_fields["tau_xy"].get(nid, 0.0))
        c, s = basis(node.x, node.y, center)
        sr, st, trt = stress(c, s, sx, sy, txy)
        sig_r[nid] = sr
        sig_t[nid] = st
        tau_rt[nid] = trt

    new_fields["sig_r"] = sig_r
    new_fields["sig_t"] = sig_t
    new_fields["tau_rt"] = tau_rt
    return new_fields


def convert_nodal_stress_rows(
    mesh: Mesh2DProtocol,
    data: NodalStressCsv,
    center: Sequence[float],
) -> NodalStressCsv:
    """Convert every resolved CSV row while preserving duplicate provenance."""
    required = {"sig_x", "sig_y", "tau_xy"}
    polar_names = {"sig_r", "sig_t", "tau_rt"}
    if not required.issubset(data.field_names) or polar_names.intersection(data.field_names):
        return data
    if len(center) != 2:
        raise ValueError("center must have 2 values")

    node_lookup = {node.id: node for node in mesh.nodes}
    field_names = tuple(
        name for name in data.field_names if name not in required
    ) + ("sig_r", "sig_t", "tau_rt")
    converted: list[NodalStressCsvRow] = []
    for row in data.rows:
        node = node_lookup.get(row.node_id)
        if node is None:
            continue
        values = {name: value for name, value in row.values.items() if name not in required}
        c, s = basis(node.x, node.y, center)
        sig_r, sig_t, tau_rt = stress(
            c,
            s,
            float(row.values.get("sig_x", 0.0)),
            float(row.values.get("sig_y", 0.0)),
            float(row.values.get("tau_xy", 0.0)),
        )
        values.update(sig_r=sig_r, sig_t=sig_t, tau_rt=tau_rt)
        converted.append(
            NodalStressCsvRow(
                node_id=row.node_id,
                elem_id=row.elem_id,
                local_node=row.local_node,
                averaged=row.averaged,
                values=values,
                legacy=row.legacy,
            )
        )
    return NodalStressCsv(field_names, tuple(converted))


def convert_element_stress_fields(
    mesh: Mesh2DProtocol,
    field_data: Dict[str, Dict[int, float]],
    center: Sequence[float],
) -> Dict[str, Dict[int, float]]:
    """Convert element stress fields to polar components."""
    required = {"sig_x", "sig_y", "tau_xy"}
    polar_names = {"sig_r", "sig_t", "tau_rt"}
    if not required.issubset(field_data) or polar_names.intersection(field_data):
        return field_data

    node_lookup = {node.id: node for node in mesh.nodes}
    elem_lookup = {elem.id: elem for elem in mesh.elements}

    new_fields = {name: vals for name, vals in field_data.items() if name not in required}
    sig_r: Dict[int, float] = {}
    sig_t: Dict[int, float] = {}
    tau_rt: Dict[int, float] = {}

    for eid, elem in elem_lookup.items():
        sx = float(field_data["sig_x"].get(eid, 0.0))
        sy = float(field_data["sig_y"].get(eid, 0.0))
        txy = float(field_data["tau_xy"].get(eid, 0.0))
        xs = [node_lookup[nid].x for nid in elem.node_ids if nid in node_lookup]
        ys = [node_lookup[nid].y for nid in elem.node_ids if nid in node_lookup]
        if not xs or not ys:
            continue
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        c, s = basis(cx, cy, center)
        sr, st, trt = stress(c, s, sx, sy, txy)
        sig_r[eid] = sr
        sig_t[eid] = st
        tau_rt[eid] = trt

    new_fields["sig_r"] = sig_r
    new_fields["sig_t"] = sig_t
    new_fields["tau_rt"] = tau_rt
    return new_fields
