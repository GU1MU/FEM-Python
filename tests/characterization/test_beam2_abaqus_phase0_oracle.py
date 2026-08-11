"""Focused, data-free checks for the B31 Phase 0 validation oracle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPARE = _load_script("compare_frame_b31_odb.py")
EXPORT = _load_script("export_frame_b31_program_snapshot.py")


def _selection_args(**overrides):
    values = {
        "program_section_point_number": 1,
        "odb_section_point_number": None,
        "section_point_local_y": 0.05,
        "section_point_local_z": 0.01,
        "section_type": "RECT",
        "coordinate_tolerance": 1.0e-12,
        "legacy_displacement_csv": None,
        "legacy_stress_csv": None,
        "program_snapshot": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_cantilever(tmp_path: Path) -> Path:
    path = tmp_path / "phase0_minimal_b31.inp"
    path.write_text(
        "\n".join(
            (
                "*Heading",
                "Independent Phase 0 cantilever",
                "*Node",
                "1, 0.0, 0.0, 0.0",
                "2, 1.0, 0.0, 0.0",
                "*Element, type=B31, elset=BEAM",
                "1, 1, 2",
                "*Nset, nset=FIXED",
                "1",
                "*Nset, nset=TIP",
                "2",
                "*Material, name=STEEL",
                "*Elastic",
                "2.10e11, 0.3",
                "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
                "0.10, 0.02",
                "0.0, 1.0, 0.0",
                "*Boundary",
                "FIXED, ENCASTRE",
                "*Step, name=Load",
                "*Static",
                "*Cload",
                "TIP, 2, -100.0",
                "*End Step",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_rect_coordinate_mapping_is_explicit_and_freezes_program_point_one() -> None:
    args = _selection_args()
    assert COMPARE._section_point_number(args) == 25

    with pytest.raises(ValueError, match="select the Abaqus section point"):
        COMPARE._section_point_number(
            _selection_args(
                section_point_local_y=None,
                section_point_local_z=None,
            )
        )

    with pytest.raises(ValueError, match="conflicts"):
        COMPARE._section_point_number(
            _selection_args(odb_section_point_number=21)
        )


def test_comparator_json_writer_is_utf8_deterministic_without_trailing_space(
    tmp_path: Path,
) -> None:
    target = tmp_path / "oracle.json"
    payload = {
        "label": "矩形截面",
        "rows": [[1, 2], [3, 4]],
    }

    COMPARE._json_write(str(target), payload)
    first = target.read_bytes()
    COMPARE._json_write(str(target), payload)

    assert target.read_bytes() == first
    assert first.endswith(b"\n")
    assert b"\r\n" not in first
    assert all(line == line.rstrip(b" \t") for line in first.splitlines())
    assert target.read_text(encoding="utf-8")


def test_program_snapshot_exports_nodal_dofs_and_b31_ip_identity(
    tmp_path: Path,
) -> None:
    source = _write_cantilever(tmp_path)

    snapshot = EXPORT.build_snapshot(source, step="Load")

    assert snapshot["schema"] == EXPORT.SNAPSHOT_SCHEMA
    assert snapshot["positions"] == {
        "nodal": "NODE",
        "section_force_results": "INTEGRATION_POINT",
        "section_results": "INTEGRATION_POINT",
        "integration_point_available": True,
    }
    assert snapshot["counts"] == {
        "nodes": 2,
        "elements": 1,
        "section_force_result_rows": 1,
        "section_result_rows": 4,
    }
    assert snapshot["public_section_result_fields"] == [
        {
            "variable": "SF",
            "position": "integration_point",
            "association": "integration_point",
            "quantity": "force",
            "components": ["N"],
            "section_point_number": None,
            "recovery_contract": 3,
            "row_count": 1,
        },
        {
            "variable": "SM",
            "position": "integration_point",
            "association": "integration_point",
            "quantity": "moment",
            "components": ["My", "Mz"],
            "section_point_number": None,
            "recovery_contract": 3,
            "row_count": 1,
        },
        *(
            {
                "variable": "S",
                "position": "integration_point",
                "association": "integration_point",
                "quantity": "stress",
                "components": ["S11"],
                "section_point_number": point,
                "recovery_contract": 3,
                "row_count": 1,
            }
            for point in range(1, 5)
        ),
    ]
    assert tuple(row["node_id"] for row in snapshot["nodes"]) == (1, 2)
    assert all(
        len(row[field]) == 3
        for row in snapshot["nodes"]
        for field in ("U", "UR", "RF", "RM")
    )
    point_one = [
        row
        for row in snapshot["section_results"]
        if row["section_point"]["number"] == 1
    ]
    assert len(point_one) == 1
    assert all(row["position"] == "INTEGRATION_POINT" for row in point_one)
    assert all(row["integration_point"] == 1 for row in point_one)
    assert all(row["node_id"] is None for row in point_one)
    assert all(
        row["section_point"]["local_y"] > 0.0
        and row["section_point"]["local_z"] > 0.0
        for row in point_one
    )
    assert set(point_one[0]["components"]) == {"S11"}
    force_row = snapshot["section_force_results"][0]
    assert force_row["position"] == "INTEGRATION_POINT"
    assert force_row["element_id"] == 1
    assert force_row["integration_point"] == 1
    assert force_row["section"] == point_one[0]["section"]
    assert set(force_row["components"]) == {"N", "MY", "MZ"}
    assert all(math.isfinite(value) for value in force_row["components"].values())


def _oracle_for_snapshot(snapshot):
    nodes = {row["node_id"]: row for row in snapshot["nodes"]}
    point_one = [
        row
        for row in snapshot["section_results"]
        if row["section_point"]["number"] == 1
    ]
    extrapolated = []
    for node_id in (1, 2):
        row = point_one[0]
        extrapolated.append(
            {
                "position": "ELEMENT_NODAL",
                "element_id": row["element_id"],
                "node_id": node_id,
                "integration_point": None,
                "section_point": {"number": 25, "description": "Top Right Corner"},
                "components": dict(row["components"]),
            }
        )
    integration = {
        "position": "INTEGRATION_POINT",
        "element_id": 1,
        "node_id": None,
        "integration_point": 1,
        "section_point": {"number": 25, "description": "Top Right Corner"},
        "components": {
            "S11": 1.0,
            "S12": 2.0,
            "Mises": 3.0,
            "MaxPrincipal": 4.0,
        },
    }
    return {
        "metadata": {
            "path": "synthetic.odb",
            "sha256": "0" * 64,
            "step": "Load",
            "frame_index": 1,
            "instance": "PART-1-1",
        },
        "coordinates": {
            node_id: row["coordinates"] for node_id, row in nodes.items()
        },
        "connectivity": {1: [1, 2]},
        "groups": {
            "columns": [1],
            "arch_ribs": [1],
            "purlins": [1],
            "side_rails": [1],
            "roof_bracing": [1],
        },
        "node_fields": {
            field: {node_id: row[field] for node_id, row in nodes.items()}
            for field in ("U", "UR", "RF", "RM")
        },
        "field_components": {
            field: [f"{field}1", f"{field}2", f"{field}3"]
            for field in ("U", "UR", "RF", "RM")
        },
        "stress": {
            "stored_position": "INTEGRATION_POINT",
            "integration_point": [integration],
            "element_nodal": extrapolated,
            "section_output_available": {"SF": False, "SM": False},
        },
    }


def test_comparison_never_counts_extrapolated_copies_as_integration_points(
    tmp_path: Path,
) -> None:
    source = _write_cantilever(tmp_path)
    snapshot = EXPORT.build_snapshot(source, step="Load")
    snapshot_path = tmp_path / "program_snapshot.json"
    snapshot_path.write_text(
        json.dumps(snapshot, sort_keys=True),
        encoding="utf-8",
    )
    args = _selection_args(program_snapshot=str(snapshot_path))
    loaded, program_nodes, program_sections = COMPARE.load_program_snapshot(
        str(snapshot_path),
        args,
        25,
    )

    summary, details, _worst_nodes, _worst_elements = COMPARE.compare_snapshots(
        loaded,
        program_nodes,
        program_sections,
        _oracle_for_snapshot(snapshot),
        args,
        25,
    )

    identity = summary["identity"]
    assert identity["odb_target_integration_point_rows"] == 1
    assert identity["odb_element_nodal_extrapolated_rows"] == 2
    contract = summary["position_contract"]
    assert contract["formal_program_rows"] == 1
    assert contract["diagnostic_is_acceptance_evidence"] is False
    assert (
        contract[
            "element_nodal_rows_are_extrapolations_not_independent_integration_points"
        ]
        is True
    )
    assert summary["metrics"]["formal_integration_point"]["S11"][
        "matched_rows"
    ] == 1
    assert summary["metrics"]["diagnostic_section_end_vs_element_nodal"][
        "S11"
    ]["matched_rows"] == 0
    assert all(
        metrics["diagnostic_section_end_vs_element_nodal"]["S11"][
            "matched_rows"
        ]
        == 0
        for metrics in summary["metrics"]["groups"].values()
    )
    assert summary["metrics"]["global_nodal"]["U"]["vector_relative_l2"] == 0.0
    assert {
        (row["position_program"], row["position_abaqus"])
        for row in details
        if row["comparison"] == "formal" and row["status"] == "matched"
    } == {("INTEGRATION_POINT", "INTEGRATION_POINT")}


def test_report_writes_are_deterministic_and_reject_data_output(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    value = {"z": [2, 1], "a": {"value": 3.25}}

    COMPARE._json_write(str(first), value)
    COMPARE._json_write(str(second), value)

    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(ValueError, match="outside data"):
        COMPARE._output_directory(str(PROJECT_ROOT / "data" / "validation"))
