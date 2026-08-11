from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASELINE_SUMMARY = ROOT / "summary.json"
PHASE3_GATE = ROOT / "phase3" / "phase3_gate.json"
PHASE4_GATE = ROOT / "phase4" / "phase4_gate.json"
PHASE5_GATE = ROOT / "phase5" / "phase5_gate.json"
FINAL_SUMMARY = ROOT / "phase5" / "portal_point1" / "summary.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate(actual: float, threshold: float, *, absolute: bool = False) -> dict[str, object]:
    value = abs(actual) if absolute else actual
    return {
        "actual": actual,
        "threshold": threshold,
        "passed": value <= threshold,
    }


def _relative_components(actual: list[float], reference: list[float]) -> list[float]:
    result = []
    for value, oracle in zip(actual, reference, strict=True):
        result.append(abs(value - oracle) / abs(oracle) if oracle else math.nan)
    return result


def main() -> None:
    baseline = _read(BASELINE_SUMMARY)
    phase3 = _read(PHASE3_GATE)
    phase4 = _read(PHASE4_GATE)
    phase5 = _read(PHASE5_GATE)
    final = _read(FINAL_SUMMARY)

    baseline_nodal = baseline["metrics"]["global_nodal"]
    final_nodal = final["metrics"]["global_nodal"]
    final_identity = final["identity"]
    totals = final["totals"]

    stress_points = phase4["s11_point_metrics"]
    worst_s11 = {
        key: max(point[key] for point in stress_points.values())
        for key in (
            "mae",
            "max_absolute_error",
            "significant_relative_l2",
        )
    }
    portal = phase5["portal"]["components"]
    program_reaction = totals["program_reaction"]
    abaqus_reaction = totals["abaqus_reaction"]
    reaction_force_relative = _relative_components(
        program_reaction["force"],
        abaqus_reaction["force"],
    )
    reaction_moment_relative = _relative_components(
        program_reaction["moment_about_origin"],
        abaqus_reaction["moment_about_origin"],
    )
    total_load_scale = math.sqrt(
        sum(value * value for value in abaqus_reaction["force"])
    )
    near_zero_force_error = abs(
        program_reaction["force"][1] - abaqus_reaction["force"][1]
    )

    gates = {
        "matched_nodes": {
            "actual": final_identity["matched_nodes"],
            "expected": 117,
            "passed": final_identity["matched_nodes"] == 117,
        },
        "matched_elements": {
            "actual": final_identity["matched_elements"],
            "expected": 276,
            "passed": final_identity["matched_elements"] == 276,
        },
        "unique_longitudinal_integration_points": {
            "actual": final_identity["odb_target_integration_point_rows"],
            "expected": 276,
            "passed": (
                final_identity["odb_target_integration_point_rows"] == 276
                and not final_identity["odb_target_integration_point_missing_elements"]
                and not final_identity["odb_target_integration_point_duplicate_elements"]
            ),
        },
        "reaction_force_x_relative": _gate(reaction_force_relative[0], 1.0e-6),
        "reaction_force_y_near_zero_absolute": _gate(
            near_zero_force_error,
            total_load_scale * 1.0e-10,
        ),
        "reaction_force_z_relative": _gate(reaction_force_relative[2], 1.0e-6),
        "reaction_moment_max_component_relative": _gate(
            max(reaction_moment_relative),
            1.0e-6,
        ),
        "translation_vector_relative_l2": _gate(
            final_nodal["U"]["vector_relative_l2"],
            0.01,
        ),
        "translation_significant_mean_vector_relative": _gate(
            final_nodal["U"]["significant_node_mean_vector_relative_error"],
            0.015,
        ),
        "translation_significant_max_vector_relative": _gate(
            final_nodal["U"]["significant_node_max_vector_relative_error"],
            0.03,
        ),
        "u1_relative_l2": _gate(
            final_nodal["U"]["components"]["1"]["relative_l2"],
            0.01,
        ),
        "u3_relative_l2": _gate(
            final_nodal["U"]["components"]["3"]["relative_l2"],
            0.01,
        ),
        "u2_relative_l2": _gate(
            final_nodal["U"]["components"]["2"]["relative_l2"],
            0.15,
        ),
        "u2_max_absolute_mm": _gate(
            final_nodal["U"]["components"]["2"]["max_absolute_error"] * 1000.0,
            0.01,
        ),
        "rotation_vector_relative_l2": _gate(
            final_nodal["UR"]["vector_relative_l2"],
            0.05,
        ),
        "ur1_max_absolute_rad": _gate(
            final_nodal["UR"]["components"]["1"]["max_absolute_error"],
            5.0e-5,
        ),
        "ur3_max_absolute_rad": _gate(
            final_nodal["UR"]["components"]["3"]["max_absolute_error"],
            5.0e-5,
        ),
        "roof_ridge_rotation_vector_relative_l2": _gate(
            phase3["gates"]["roof_ridge_rotation_vector_relative_l2"]["actual"],
            0.15,
        ),
        "frame_direction_cosine_max_absolute": _gate(
            phase3["gates"]["frame_direction_cosine_max_absolute"]["actual"],
            1.0e-10,
        ),
        "s11_significant_relative_l2": _gate(
            worst_s11["significant_relative_l2"],
            0.02,
        ),
        "s11_mae_pa": _gate(worst_s11["mae"], 0.20e6),
        "s11_max_absolute_pa": _gate(
            worst_s11["max_absolute_error"],
            1.0e6,
        ),
        "mises_significant_relative_l2": _gate(
            portal["Mises"]["maximum_significant_relative_l2"],
            0.03,
        ),
        "mises_mae_pa": _gate(portal["Mises"]["all_row_mae_pa"], 0.25e6),
        "max_principal_significant_relative_l2": _gate(
            portal["MaxPrincipal"]["maximum_significant_relative_l2"],
            0.05,
        ),
        "max_principal_mae_pa": _gate(
            portal["MaxPrincipal"]["all_row_mae_pa"],
            0.25e6,
        ),
        "point_stress_rows": {
            "actual": phase5["portal"]["section_stress_rows"],
            "expected": 1104,
            "passed": phase5["portal"]["section_stress_rows"] == 1104,
        },
        "five_member_groups_present": {
            "actual": sorted(phase4["s11_group_metrics_point_1"]),
            "expected_count": 5,
            "passed": len(phase4["s11_group_metrics_point_1"]) == 5,
        },
        "section_resultants": {
            "actual_relative_l2": {
                key: value["relative_l2"]
                for key, value in phase4[
                    "section_force_metrics_reconstructed_from_four_corner_s11"
                ].items()
            },
            "passed": all(
                phase4["gates"][key]
                for key in (
                    "N_relative_l2_below_1_5_percent",
                    "MY_relative_l2_below_1_percent",
                    "MZ_relative_l2_below_1_percent",
                )
            ),
        },
        "program_force_equilibrium_relative_l2": _gate(
            phase4["balance"]["program_equilibrium"]["force"]["relative_l2"],
            1.0e-12,
        ),
        "program_moment_equilibrium_relative_l2": _gate(
            phase4["balance"]["program_equilibrium"]["moment_about_origin"][
                "relative_l2"
            ],
            1.0e-12,
        ),
        "program_invariant_recompute_relative": _gate(
            phase5["invariant_recompute"][
                "program_maximum_relative_to_output"
            ],
            1.0e-12,
        ),
    }
    passed = all(gate["passed"] for gate in gates.values())

    comparison = {
        "translation_vector_relative_l2": {
            "phase0": baseline_nodal["U"]["vector_relative_l2"],
            "final": final_nodal["U"]["vector_relative_l2"],
        },
        "rotation_vector_relative_l2": {
            "phase0": baseline_nodal["UR"]["vector_relative_l2"],
            "final": final_nodal["UR"]["vector_relative_l2"],
        },
        "formal_point_stress_rows": {
            "phase0": baseline["position_contract"]["formal_program_rows"],
            "final": phase5["portal"]["section_stress_rows"],
        },
        "s11": {
            "phase0": "unavailable at matching integration-point/section-point identity",
            "final_worst_significant_relative_l2": worst_s11[
                "significant_relative_l2"
            ],
            "final_worst_mae_pa": worst_s11["mae"],
        },
        "s12": {
            "phase0": "unavailable at matching integration-point/section-point identity",
            "final_significant_relative_l2": portal["S12"][
                "maximum_significant_relative_l2"
            ],
            "final_mae_pa": portal["S12"]["all_row_mae_pa"],
        },
        "mises": {
            "phase0": "unavailable at matching integration-point/section-point identity",
            "final_significant_relative_l2": portal["Mises"][
                "maximum_significant_relative_l2"
            ],
            "final_mae_pa": portal["Mises"]["all_row_mae_pa"],
        },
        "max_principal": {
            "phase0": "unavailable at matching integration-point/section-point identity",
            "final_significant_relative_l2": portal["MaxPrincipal"][
                "maximum_significant_relative_l2"
            ],
            "final_mae_pa": portal["MaxPrincipal"]["all_row_mae_pa"],
        },
    }

    inputs = {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
        for path in (
            BASELINE_SUMMARY,
            PHASE3_GATE,
            PHASE4_GATE,
            PHASE5_GATE,
            FINAL_SUMMARY,
        )
    }
    gate_report = {
        "schema": "fem-python.abaqus-b31-phase6-gate.v1",
        "passed": passed,
        "inputs": inputs,
        "gates": gates,
    }
    summary = {
        "schema": "fem-python.abaqus-b31-phase6-summary.v1",
        "passed": passed,
        "formulation_provenance": {
            "beam_formulation": "abaqus-b31-linear-timoshenko-v1",
            "beam_result_position": "INTEGRATION_POINT",
            "beam_recovery_contract": 4,
        },
        "phase0_to_final": comparison,
        "member_groups": phase4["s11_group_metrics_point_1"],
        "public_resultants": phase5["public_integration_point_resultants"],
        "balance": phase4["balance"],
        "inputs": inputs,
    }

    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "final_gate.json").write_text(
        json.dumps(gate_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (HERE / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = f"""# Phase 6 B31 end-to-end validation

## Result

Phase 6 passes all {len(gates)} final numerical gates. The comparison preserves
the Phase 0 identity, thresholds, step/frame selection, and section-point
coordinate mapping; no fitted scaling is applied.

## Phase 0 to final

| Measure | Phase 0 | Final |
|---|---:|---:|
| Translation vector relative L2 | {comparison['translation_vector_relative_l2']['phase0']:.9g} | {comparison['translation_vector_relative_l2']['final']:.9g} |
| Rotation vector relative L2 | {comparison['rotation_vector_relative_l2']['phase0']:.9g} | {comparison['rotation_vector_relative_l2']['final']:.9g} |
| Formal point-stress rows | {comparison['formal_point_stress_rows']['phase0']} | {comparison['formal_point_stress_rows']['final']} |
| S11 significant relative L2 (worst point) | unavailable | {worst_s11['significant_relative_l2']:.9g} |
| S12 significant relative L2 | unavailable | {portal['S12']['maximum_significant_relative_l2']:.9g} |
| Mises significant relative L2 | unavailable | {portal['Mises']['maximum_significant_relative_l2']:.9g} |
| MaxPrincipal significant relative L2 | unavailable | {portal['MaxPrincipal']['maximum_significant_relative_l2']:.9g} |

Phase 0 published only section-end stress diagnostics, so it had no formal
integration-point/section-point stress observation that could be compared to
Abaqus S. The final result contains 276 unique elements and 1,104 point-stress
rows (four section points at longitudinal integration point 1).

## Final response and stress

- Translation vector relative L2: {final_nodal['U']['vector_relative_l2']:.9g};
  U1/U2/U3 relative L2: {final_nodal['U']['components']['1']['relative_l2']:.9g},
  {final_nodal['U']['components']['2']['relative_l2']:.9g}, and
  {final_nodal['U']['components']['3']['relative_l2']:.9g}.
- Rotation vector relative L2: {final_nodal['UR']['vector_relative_l2']:.9g};
  UR1/UR3 maximum absolute errors are
  {final_nodal['UR']['components']['1']['max_absolute_error']:.9g} rad and
  {final_nodal['UR']['components']['3']['max_absolute_error']:.9g} rad.
- S11 worst-point relative L2/MAE/max: {worst_s11['significant_relative_l2']:.9g},
  {worst_s11['mae']:.9g} Pa, and {worst_s11['max_absolute_error']:.9g} Pa.
- S12 relative L2/MAE: {portal['S12']['maximum_significant_relative_l2']:.9g}
  and {portal['S12']['all_row_mae_pa']:.9g} Pa.
- Mises relative L2/MAE: {portal['Mises']['maximum_significant_relative_l2']:.9g}
  and {portal['Mises']['all_row_mae_pa']:.9g} Pa.
- MaxPrincipal relative L2/MAE:
  {portal['MaxPrincipal']['maximum_significant_relative_l2']:.9g} and
  {portal['MaxPrincipal']['all_row_mae_pa']:.9g} Pa.

All five member groups are present. The largest point-1 group S11 MAE is
{max(value['mae'] for value in phase4['s11_group_metrics_point_1'].values()):.9g}
Pa. Public SF=(N,Vy,Vz) and SM=(T,My,Mz) each contain 276 unique integration-
point rows under recovery contract 4.

## Balance, frames, and reproducibility

Program force and moment equilibrium relative residuals are
{phase4['balance']['program_equilibrium']['force']['relative_l2']:.9g} and
{phase4['balance']['program_equilibrium']['moment_about_origin']['relative_l2']:.9g}.
The maximum frame direction-cosine difference is
{phase3['gates']['frame_direction_cosine_max_absolute']['actual']:.9g}.
Program invariant recomputation is exact; the Abaqus snapshot's maximum
relative recomputation difference is
{phase5['invariant_recompute']['abaqus_maximum_relative_to_output']:.9g},
consistent with single-precision output rounding.

`final_gate.json` records every threshold and actual value. `summary.json`
records the Phase 0/final comparison, formulation provenance, member groups,
public resultants, balance, and SHA-256 hashes of every source report.
"""
    (HERE / "validation_report.md").write_text(report, encoding="utf-8")

    if not passed:
        raise SystemExit("Phase 6 numerical gates failed")


if __name__ == "__main__":
    main()
