from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict

from .._csv import (
    _NODAL_STRESS_METADATA_FIELDS,
    parse_csv_integer,
    parse_csv_number,
    validate_nodal_stress_header,
    validate_nodal_stress_row,
)


@dataclass(frozen=True)
class NodalStressCsvRow:
    """One already-resolved nodal stress CSV row."""

    node_id: int
    elem_id: int | None
    local_node: int | None
    averaged: bool
    values: dict[str, float]


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
            node_id = parse_csv_integer(
                row.get("node_id"),
                path,
                reader.line_num,
                "node_id",
                source="Displacement CSV",
            )
            ux = parse_csv_number(
                row.get("ux"), path, reader.line_num, "ux", source="Displacement CSV"
            )
            uy = parse_csv_number(
                row.get("uy"), path, reader.line_num, "uy", source="Displacement CSV"
            )
            rz = (
                parse_csv_number(
                    row.get("rz"),
                    path,
                    reader.line_num,
                    "rz",
                    source="Displacement CSV",
                )
                if has_rz and row.get("rz", "") != ""
                else 0.0
            )
            uz = (
                parse_csv_number(
                    row.get("uz"),
                    path,
                    reader.line_num,
                    "uz",
                    source="Displacement CSV",
                )
                if has_uz and row.get("uz", "") != ""
                else 0.0
            )
            node_disp[node_id] = {"ux": ux, "uy": uy, "uz": uz, "rz": rz}

    for node in mesh.nodes:
        if node.id not in node_disp:
            node_disp[node.id] = {"ux": 0.0, "uy": 0.0, "rz": 0.0}

    return node_disp


def read_nodal_stress_rows(path: str) -> NodalStressCsv:
    """Read resolved nodal rows without collapsing repeated node ids."""
    rows: list[NodalStressCsvRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = tuple(reader.fieldnames or ())
        validate_nodal_stress_header(fieldnames, path)
        ignore_exact = {"node_id", "x", "y", "z", *_NODAL_STRESS_METADATA_FIELDS}
        stress_names = tuple(name for name in fieldnames if name not in ignore_exact)

        for row in reader:
            node_id, elem_id, local_node, averaged = validate_nodal_stress_row(
                row, path, reader.line_num
            )
            values: dict[str, float] = {}
            for name in stress_names:
                val_str = row.get(name, "")
                if val_str == "":
                    continue
                try:
                    values[name] = float(val_str)
                except ValueError:
                    values[name] = 0.0
            rows.append(
                NodalStressCsvRow(
                    node_id=node_id,
                    elem_id=elem_id,
                    local_node=local_node,
                    averaged=averaged,
                    values=values,
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
            elem_id = parse_csv_integer(
                row.get("elem_id"),
                path,
                reader.line_num,
                "elem_id",
                source="Element stress CSV",
            )
            for name in stress_field_names:
                val_str = row.get(name, "")
                if val_str == "":
                    continue
                try:
                    val = float(val_str)
                except ValueError:
                    val = 0.0
                field_data[name][elem_id] = field_data[name].get(elem_id, 0.0) + val
                counts[name][elem_id] = counts[name].get(elem_id, 0) + 1

    for name, values in field_data.items():
        for elem_id, total in list(values.items()):
            values[elem_id] = total / counts[name][elem_id]

    return field_data
