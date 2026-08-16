from __future__ import annotations

import json
from pathlib import Path

from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.post import vtk


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "helpers" / "fixtures" / "phase8"


def _write_utf8_lf(path: Path, text: str) -> Path:
    path.write_bytes(text.encode("utf-8"))
    return path


def _parse_legacy_vtk_semantics(path: Path) -> dict[str, object]:
    """Read the small ASCII subset emitted by the compatibility writer."""

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cursor = 0
    format_rows = lines[cursor : cursor + 4]
    cursor += 4

    point_header = lines[cursor].split()
    cursor += 1
    point_count = int(point_header[1])
    points = [
        [float(value) for value in lines[index].split()]
        for index in range(cursor, cursor + point_count)
    ]
    cursor += point_count

    cell_header = lines[cursor].split()
    cursor += 1
    cell_count = int(cell_header[1])
    cells = [
        [int(value) for value in lines[index].split()]
        for index in range(cursor, cursor + cell_count)
    ]
    cursor += cell_count

    cell_type_header = lines[cursor].split()
    cursor += 1
    cell_type_count = int(cell_type_header[1])
    cell_types = [
        int(lines[index])
        for index in range(cursor, cursor + cell_type_count)
    ]
    cursor += cell_type_count

    point_data_header = lines[cursor].split()
    cursor += 1
    point_data_count = int(point_data_header[1])
    point_vectors: dict[str, list[list[float]]] = {}
    point_scalars: dict[str, list[float]] = {}
    while cursor < len(lines) and not lines[cursor].startswith("CELL_DATA "):
        header = lines[cursor].split()
        cursor += 1
        if header[0] == "VECTORS":
            point_vectors[header[1]] = [
                [float(value) for value in lines[index].split()]
                for index in range(cursor, cursor + point_data_count)
            ]
            cursor += point_data_count
            continue
        if header[0] == "SCALARS":
            assert lines[cursor] == "LOOKUP_TABLE default"
            cursor += 1
            point_scalars[header[1]] = [
                float(lines[index])
                for index in range(cursor, cursor + point_data_count)
            ]
            cursor += point_data_count
            continue
        raise AssertionError(f"Unexpected POINT_DATA row: {' '.join(header)}")

    cell_data_count = 0
    cell_scalars: dict[str, list[float]] = {}
    if cursor < len(lines):
        cell_data_header = lines[cursor].split()
        cursor += 1
        cell_data_count = int(cell_data_header[1])
        while cursor < len(lines):
            header = lines[cursor].split()
            cursor += 1
            assert header[0] == "SCALARS"
            assert lines[cursor] == "LOOKUP_TABLE default"
            cursor += 1
            cell_scalars[header[1]] = [
                float(lines[index])
                for index in range(cursor, cursor + cell_data_count)
            ]
            cursor += cell_data_count

    return {
        "format": {
            "version": format_rows[0],
            "title": format_rows[1],
            "encoding": format_rows[2],
            "dataset": format_rows[3],
        },
        "points": points,
        "cells": cells,
        "cell_types": cell_types,
        "point_data": {
            "count": point_data_count,
            "vectors": point_vectors,
            "scalars": point_scalars,
        },
        "cell_data": {
            "count": cell_data_count,
            "scalars": cell_scalars,
        },
    }


def test_legacy_vtk_from_csv_semantics_match_golden(tmp_path: Path) -> None:
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 0.0, 1.0),
            Node2D(4, -1.0, 0.0),
            Node2D(5, 0.0, -1.0),
        ],
        elements=[
            Element2D(1, [1, 2, 3], "Tri3"),
            Element2D(2, [3, 4, 5], "Tri3"),
        ],
    )
    displacement_path = _write_utf8_lf(
        tmp_path / "displacement.csv",
        (
            "node_id,x,y,ux,uy\n"
            "1,0,0,0,0\n"
            "2,1,0,1,0\n"
            "3,0,1,3,4\n"
            "4,-1,0,-1,0\n"
            "5,0,-1,0,-1\n"
        ),
    )
    element_path = _write_utf8_lf(
        tmp_path / "element.csv",
        "elem_id,sig_x,mises\n1,100,111\n2,200,222\n",
    )
    nodal_path = _write_utf8_lf(
        tmp_path / "nodal.csv",
        (
            "node_id,x,y,elem_id,local_node,averaged,sig_x,mises\n"
            "1,0,0,1,1,false,0,0\n"
            "2,1,0,1,2,false,1,1\n"
            "3,0,1,1,3,false,10,10\n"
            "3,0,1,2,1,false,20,20\n"
            "4,-1,0,2,2,false,4,4\n"
            "5,0,-1,2,3,false,5,5\n"
        ),
    )
    vtk_path = tmp_path / "legacy-roundtrip.vtk"

    vtk.export.from_csv(
        mesh,
        displacement_path,
        element_path,
        vtk_path,
        nodal_path,
    )

    expected = json.loads(
        (FIXTURE_ROOT / "legacy_vtk_csv_roundtrip_golden.json").read_text(
            encoding="utf-8"
        )
    )
    assert _parse_legacy_vtk_semantics(vtk_path) == expected
