from __future__ import annotations

from typing import Dict, Sequence

from ...core.mesh import Mesh2D
from .. import polar as _polar
from .fields import NodalStressCsv, NodalStressCsvRow


def convert_nodal_displacement(
    mesh: Mesh2D,
    node_disp: Dict[int, Dict[str, float]],
    center: Sequence[float],
) -> Dict[int, Dict[str, float]]:
    """Convert nodal displacement dict into polar components."""
    if len(center) != 2:
        raise ValueError(f"center must have 2 values, got {len(center)}: {center!r}")

    polar_disp: Dict[int, Dict[str, float]] = {}

    for node in mesh.nodes:
        disp = node_disp.get(node.id, {"ux": 0.0, "uy": 0.0, "rz": 0.0})
        c, s = _polar._basis(node.x, node.y, center)
        ur, ut = _polar._displacement(
            c,
            s,
            float(disp.get("ux", 0.0)),
            float(disp.get("uy", 0.0)),
        )
        polar_disp[node.id] = {"ux": ur, "uy": ut, "rz": float(disp.get("rz", 0.0))}

    return polar_disp


def convert_nodal_stress_rows(
    mesh: Mesh2D,
    data: NodalStressCsv,
    center: Sequence[float],
) -> NodalStressCsv:
    """Convert every resolved CSV row while preserving duplicate provenance."""
    required = {"sig_x", "sig_y", "tau_xy"}
    polar_names = {"sig_r", "sig_t", "tau_rt"}
    if not required.issubset(data.field_names) or polar_names.intersection(data.field_names):
        return data
    if len(center) != 2:
        raise ValueError(f"center must have 2 values, got {len(center)}: {center!r}")

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
        c, s = _polar._basis(node.x, node.y, center)
        sig_r, sig_t, tau_rt = _polar._stress(
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
            )
        )
    return NodalStressCsv(field_names, tuple(converted))


def convert_element_stress_fields(
    mesh: Mesh2D,
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
        c, s = _polar._basis(cx, cy, center)
        sr, st, trt = _polar._stress(c, s, sx, sy, txy)
        sig_r[eid] = sr
        sig_t[eid] = st
        tau_rt[eid] = trt

    new_fields["sig_r"] = sig_r
    new_fields["sig_t"] = sig_t
    new_fields["tau_rt"] = tau_rt
    return new_fields
