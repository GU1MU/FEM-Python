from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class NodalStressCsvRow:
    """One already-resolved nodal stress CSV row."""

    node_id: int
    elem_id: int | None
    local_node: int | None
    averaged: bool
    values: dict[str, float]
    legacy: bool = False


@dataclass(frozen=True)
class NodalStressCsv:
    """Ordered resolved rows and their scalar field names."""

    field_names: tuple[str, ...]
    rows: tuple[NodalStressCsvRow, ...]


def read_displacement(mesh, path: str) -> Dict[int, Dict[str, float]]:
    """Read nodal displacement CSV into a node keyed field dict."""
    node_disp: Dict[int, Dict[str, float]] = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"node_id", "ux", "uy"}
        if len(mesh.nodes) > 0 and hasattr(mesh.nodes[0], "z"):
            required_cols.add("uz")
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(f"Disp CSV requires columns {required_cols}, got {reader.fieldnames}")

        has_rz = "rz" in reader.fieldnames
        has_uz = "uz" in reader.fieldnames

        for row in reader:
            nid = int(row["node_id"])
            ux = float(row["ux"])
            uy = float(row["uy"])
            rz = float(row["rz"]) if has_rz and row.get("rz", "") != "" else 0.0
            uz = float(row["uz"]) if has_uz and row.get("uz", "") != "" else 0.0
            node_disp[nid] = {"ux": ux, "uy": uy, "uz": uz, "rz": rz}

    for node in mesh.nodes:
        if node.id not in node_disp:
            node_disp[node.id] = {"ux": 0.0, "uy": 0.0, "rz": 0.0}

    return node_disp


def read_nodal_stress(path: str) -> Dict[str, Dict[int, float]]:
    """Read unique nodal stress rows into compatibility field dictionaries."""
    data = read_nodal_stress_rows(path)
    nodal_fields: Dict[str, Dict[int, float]] = {
        name: {} for name in data.field_names
    }
    seen: set[int] = set()
    for row in data.rows:
        if row.node_id in seen:
            raise ValueError(
                f"Nodal stress CSV contains repeated node_id {row.node_id}; "
                "use read_nodal_stress_rows() to preserve contributions"
            )
        seen.add(row.node_id)
        for name, value in row.values.items():
            nodal_fields[name][row.node_id] = value
    return nodal_fields


def read_nodal_stress_rows(path: str) -> NodalStressCsv:
    """Read resolved nodal rows without collapsing repeated node ids."""
    rows: list[NodalStressCsvRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = tuple(reader.fieldnames or ())
        if "node_id" not in fieldnames:
            raise ValueError(f"Nodal stress CSV requires 'node_id', got {reader.fieldnames}")

        metadata = {"elem_id", "local_node", "averaged"}
        present_metadata = metadata.intersection(fieldnames)
        if present_metadata and present_metadata != metadata:
            raise ValueError(
                "Nodal stress CSV metadata requires elem_id, local_node, and averaged"
            )
        is_legacy = not present_metadata
        ignore_exact = {"node_id", "x", "y", "z", *metadata}
        stress_names = tuple(name for name in fieldnames if name not in ignore_exact)

        for row in reader:
            nid = int(row["node_id"])
            values: dict[str, float] = {}
            for name in stress_names:
                val_str = row.get(name, "")
                if val_str == "":
                    continue
                try:
                    values[name] = float(val_str)
                except ValueError:
                    values[name] = 0.0
            if is_legacy:
                elem_id = None
                local_node = None
                averaged = True
            else:
                elem_value = row.get("elem_id", "").strip()
                local_value = row.get("local_node", "").strip()
                elem_id = int(elem_value) if elem_value else None
                local_node = int(local_value) if local_value else None
                averaged_value = row.get("averaged", "").strip().lower()
                if averaged_value not in {"true", "false"}:
                    raise ValueError(
                        f"Nodal stress CSV node {nid} has invalid averaged value "
                        f"{row.get('averaged')!r}"
                    )
                averaged = averaged_value == "true"
            rows.append(
                NodalStressCsvRow(
                    node_id=nid,
                    elem_id=elem_id,
                    local_node=local_node,
                    averaged=averaged,
                    values=values,
                    legacy=is_legacy,
                )
            )

    return NodalStressCsv(stress_names, tuple(rows))


def point_fields(
    data: NodalStressCsv,
    point_rows: tuple[NodalStressCsvRow | None, ...],
) -> Dict[str, list[float]]:
    """Expand ordered CSV values to match result-topology points."""
    return {
        name: [
            0.0 if row is None else float(row.values.get(name, 0.0))
            for row in point_rows
        ]
        for name in data.field_names
    }


def read_element_stress(path: str) -> Dict[str, Dict[int, float]]:
    """Read element stress CSV into field dictionaries."""
    field_data: Dict[str, Dict[int, float]] = {}
    counts: Dict[str, Dict[int, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "elem_id" not in (reader.fieldnames or []):
            raise ValueError(f"Element stress CSV requires 'elem_id', got {reader.fieldnames}")

        ignore_prefixes = ("node", "nid")
        ignore_exact = {"elem_id", "local_node"}

        stress_field_names = [
            name for name in (reader.fieldnames or [])
            if name not in ignore_exact and not name.startswith(ignore_prefixes)
        ]

        for name in stress_field_names:
            field_data[name] = {}
            counts[name] = {}

        for row in reader:
            eid = int(row["elem_id"])
            for name in stress_field_names:
                val_str = row.get(name, "")
                if val_str == "":
                    continue
                try:
                    val = float(val_str)
                except ValueError:
                    val = 0.0
                field_data[name][eid] = field_data[name].get(eid, 0.0) + val
                counts[name][eid] = counts[name].get(eid, 0) + 1

    for name, values in field_data.items():
        for eid, total in list(values.items()):
            values[eid] = total / counts[name][eid]

    return field_data
