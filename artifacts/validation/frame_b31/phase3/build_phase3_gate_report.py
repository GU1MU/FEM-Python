from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PHASE_DIR = Path(__file__).resolve().parent


def _read(name: str):
    return json.loads((PHASE_DIR / name).read_text(encoding="utf-8"))


def _metric(value: float, threshold: float) -> dict[str, float | bool]:
    return {
        "actual": float(value),
        "threshold": float(threshold),
        "passed": bool(value < threshold),
    }


def main() -> None:
    frame_gate = _read("frame_gate.json")
    summary = _read("summary.json")
    oracle = _read("abaqus_oracle_snapshot.json")
    program = _read("program_snapshot.json")
    previous = _read("../phase2/program_snapshot.json")

    global_nodal = summary["metrics"]["global_nodal"]
    displacement = global_nodal["U"]
    rotation = global_nodal["UR"]
    maximum_z = max(float(value[2]) for value in oracle["coordinates"].values())
    ridge_ids = tuple(
        sorted(
            int(node_id)
            for node_id, coordinates in oracle["coordinates"].items()
            if abs(float(coordinates[0]) - 12.0) <= 1.0e-6
            and abs(float(coordinates[2]) - maximum_z) <= 1.0e-6
        )
    )
    program_by_node = {int(item["node_id"]): item for item in program["nodes"]}
    oracle_ridge = np.asarray(
        [oracle["node_fields"]["UR"][str(node_id)] for node_id in ridge_ids],
        dtype=float,
    )
    program_ridge = np.asarray(
        [program_by_node[node_id]["UR"] for node_id in ridge_ids],
        dtype=float,
    )
    ridge_relative_l2 = float(
        np.linalg.norm(program_ridge - oracle_ridge)
        / np.linalg.norm(oracle_ridge)
    )

    previous_by_node = {
        int(item["node_id"]): item for item in previous["nodes"]
    }
    phase3_change = max(
        abs(float(program_by_node[node_id][field][component]) - float(previous_by_node[node_id][field][component]))
        for node_id in program_by_node
        for field in ("U", "UR")
        for component in range(3)
    )

    gates = {
        "frame_direction_cosine_max_absolute": _metric(
            frame_gate["max_direction_cosine_absolute_error"],
            1.0e-10,
        ),
        "translation_vector_relative_l2": _metric(
            displacement["vector_relative_l2"],
            0.01,
        ),
        "translation_significant_node_max_vector_relative": _metric(
            displacement["significant_node_max_vector_relative_error"],
            0.03,
        ),
        "u2_relative_l2": _metric(
            displacement["components"]["2"]["relative_l2"],
            0.15,
        ),
        "u2_max_absolute_mm": _metric(
            displacement["components"]["2"]["max_absolute_error"] * 1000.0,
            0.010,
        ),
        "rotation_vector_relative_l2": _metric(
            rotation["vector_relative_l2"],
            0.05,
        ),
        "ur1_max_absolute_rad": _metric(
            rotation["components"]["1"]["max_absolute_error"],
            5.0e-5,
        ),
        "ur3_max_absolute_rad": _metric(
            rotation["components"]["3"]["max_absolute_error"],
            5.0e-5,
        ),
        "roof_ridge_rotation_vector_relative_l2": _metric(
            ridge_relative_l2,
            0.15,
        ),
    }
    payload = {
        "passed": all(item["passed"] for item in gates.values()),
        "gates": gates,
        "ridge_node_ids": ridge_ids,
        "matched_nodes": summary["identity"]["matched_nodes"],
        "target_integration_points": summary["identity"]["odb_target_integration_point_rows"],
        "phase2_to_phase3_max_u_ur_absolute_change": phase3_change,
        "residual_error_owner": {
            "formal_integration_point_rows": summary["metrics"]["formal_integration_point"]["S11"]["matched_rows"],
            "diagnostic_section_end_s11_significant_relative_l2": summary["metrics"]["diagnostic_section_end_vs_element_nodal"]["S11"]["significant_relative_l2"],
            "conclusion": "Displacement and rotation gates pass; remaining mismatch is Phase 4/5 integration-point result recovery.",
        },
        "notice_codes": frame_gate["notices"],
    }
    (PHASE_DIR / "phase3_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not payload["passed"]:
        raise SystemExit("one or more Phase 3 gates failed")


if __name__ == "__main__":
    main()
