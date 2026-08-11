"""Build the Phase 4 portal-frame gate from frozen point-specific oracles."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROGRAM = ROOT / "program_snapshot.json"
POINT_ORACLES = {
    1: ROOT / "abaqus_oracle_snapshot.json",
    2: ROOT / "point21" / "abaqus_oracle_snapshot.json",
    3: ROOT / "point1" / "abaqus_oracle_snapshot.json",
    4: ROOT / "point5" / "abaqus_oracle_snapshot.json",
}
POINT_SUMMARIES = {
    1: ROOT / "summary.json",
    2: ROOT / "point21" / "summary.json",
    3: ROOT / "point1" / "summary.json",
    4: ROOT / "point5" / "summary.json",
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_l2(program, abaqus) -> float:
    numerator = math.sqrt(sum((left - right) ** 2 for left, right in zip(program, abaqus)))
    denominator = math.sqrt(sum(value**2 for value in abaqus))
    return numerator / denominator


def _component_metrics(program, abaqus):
    differences = [left - right for left, right in zip(program, abaqus)]
    return {
        "relative_l2": _relative_l2(program, abaqus),
        "mae": sum(abs(value) for value in differences) / len(differences),
        "max_absolute_error": max(abs(value) for value in differences),
    }


def _balance_metrics(summary):
    totals = summary["totals"]
    applied = totals["program_applied"]
    reaction = totals["program_reaction"]
    result = {}
    for key in ("force", "moment_about_origin"):
        residual = [
            float(left) + float(right)
            for left, right in zip(applied[key], reaction[key])
        ]
        scale = max(
            math.sqrt(sum(float(value) ** 2 for value in applied[key])),
            1.0,
        )
        norm = math.sqrt(sum(value**2 for value in residual))
        result[key] = {
            "residual": residual,
            "residual_l2": norm,
            "relative_l2": norm / scale,
        }
    return result


def main() -> None:
    program = _load(PROGRAM)
    program_rows = {
        (int(row["element_id"]), int(row["section_point"]["number"])): row
        for row in program["section_results"]
    }
    program_force_rows = {
        int(row["element_id"]): row
        for row in program["section_force_results"]
    }
    public_fields = program["public_section_result_fields"]
    public_field_identities = [
        {
            "variable": row["variable"],
            "position": row["position"],
            "association": row["association"],
            "quantity": row["quantity"],
            "components": row["components"],
            "section_point_number": row["section_point_number"],
            "recovery_contract": row["recovery_contract"],
            "row_count": row["row_count"],
        }
        for row in public_fields
    ]
    expected_public_fields = [
        {
            "variable": "SF",
            "position": "integration_point",
            "association": "integration_point",
            "quantity": "force",
            "components": ["N"],
            "section_point_number": None,
            "recovery_contract": 3,
            "row_count": 276,
        },
        {
            "variable": "SM",
            "position": "integration_point",
            "association": "integration_point",
            "quantity": "moment",
            "components": ["My", "Mz"],
            "section_point_number": None,
            "recovery_contract": 3,
            "row_count": 276,
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
                "row_count": 276,
            }
            for point in range(1, 5)
        ),
    ]
    abaqus_by_point = {}
    group_ids = None
    for point, path in POINT_ORACLES.items():
        payload = _load(path)
        abaqus_by_point[point] = {
            int(row["element_id"]): float(row["components"]["S11"])
            for row in payload["stress"]["integration_point"]
        }
        if group_ids is None:
            group_ids = {
                name: tuple(sorted(int(value) for value in values))
                for name, values in payload["groups"].items()
            }
    assert group_ids is not None

    element_ids = sorted(element_id for element_id, point in program_rows if point == 1)
    assert len(element_ids) == 276
    assert set(program_force_rows) == set(element_ids)
    assert all(set(values) == set(element_ids) for values in abaqus_by_point.values())

    program_forces = {name: [] for name in ("N", "MY", "MZ")}
    abaqus_forces = {name: [] for name in ("N", "MY", "MZ")}
    for element_id in element_ids:
        stress_row = program_rows[(element_id, 1)]
        force_row = program_force_rows[element_id]
        dimensions = stress_row["section"]["dimensions"]
        width = float(dimensions["width"])
        height = float(dimensions["height"])
        area = width * height
        iyy = width * height**3 / 12.0
        izz = height * width**3 / 12.0
        s_pp = abaqus_by_point[1][element_id]
        s_mp = abaqus_by_point[2][element_id]
        s_mm = abaqus_by_point[3][element_id]
        s_pm = abaqus_by_point[4][element_id]
        reconstructed = {
            "N": area * (s_pp + s_mp + s_mm + s_pm) / 4.0,
            "MY": iyy * ((s_pp + s_mp) - (s_mm + s_pm)) / (2.0 * height),
            "MZ": izz * ((s_mp + s_mm) - (s_pp + s_pm)) / (2.0 * width),
        }
        for name in program_forces:
            program_forces[name].append(float(force_row["components"][name]))
            abaqus_forces[name].append(reconstructed[name])

    force_metrics = {
        name: _component_metrics(program_forces[name], abaqus_forces[name])
        for name in program_forces
    }
    point_metrics = {
        str(point): _load(path)["metrics"]["formal_integration_point"]["S11"]
        for point, path in POINT_SUMMARIES.items()
    }
    primary_s11 = point_metrics["1"]
    group_metrics = {
        name: _load(POINT_SUMMARIES[1])["metrics"]["groups"][name][
            "formal_integration_point"
        ]["S11"]
        for name in sorted(group_ids)
    }
    force_gates = {
        "N_relative_l2_below_1_5_percent": force_metrics["N"]["relative_l2"] < 0.015,
        "MY_relative_l2_below_1_percent": force_metrics["MY"]["relative_l2"] < 0.01,
        "MZ_relative_l2_below_1_percent": force_metrics["MZ"]["relative_l2"] < 0.01,
    }
    s11_gates = {
        "significant_relative_l2_below_2_percent": primary_s11["significant_relative_l2"] < 0.02,
        "mae_below_0_20_mpa": primary_s11["mae"] < 0.20e6,
        "max_below_1_mpa": primary_s11["max_absolute_error"] < 1.0e6,
        "all_groups_mae_below_0_35_mpa": all(
            metrics["mae"] < 0.35e6 for metrics in group_metrics.values()
        ),
    }
    identity_gates = {
        "276_unique_integration_points": len(element_ids) == 276,
        "one_program_ip_per_element_per_section_point": len(program_rows) == 4 * 276,
        "one_public_section_force_ip_per_element": (
            len(program_force_rows) == 276
        ),
        "public_sf_sm_s_use_typed_integration_point_fields": (
            public_field_identities == expected_public_fields
        ),
    }
    payload = {
        "schema": "fem-python-b31-phase4-gate-v1",
        "passed": all((*force_gates.values(), *s11_gates.values(), *identity_gates.values())),
        "inputs": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (PROGRAM, *POINT_ORACLES.values(), *POINT_SUMMARIES.values())
        },
        "identity": {
            "elements": len(element_ids),
            "longitudinal_integration_points": len(element_ids),
            "program_section_point_rows": len(program_rows),
            "program_section_force_rows": len(program_force_rows),
            "public_section_result_fields": public_field_identities,
        },
        "rect_mapping": {
            "program_1": {"coordinates": "+y,+z", "abaqus": 25},
            "program_2": {"coordinates": "-y,+z", "abaqus": 21},
            "program_3": {"coordinates": "-y,-z", "abaqus": 1},
            "program_4": {"coordinates": "+y,-z", "abaqus": 5},
        },
        "s11_point_metrics": point_metrics,
        "s11_group_metrics_point_1": group_metrics,
        "section_force_metrics_reconstructed_from_four_corner_s11": force_metrics,
        "gates": {**force_gates, **s11_gates, **identity_gates},
        "balance": {
            "program_equilibrium": _balance_metrics(_load(POINT_SUMMARIES[1])),
            "comparison_totals": _load(POINT_SUMMARIES[1])["totals"],
        },
    }
    output = ROOT / "phase4_gate.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
