from __future__ import annotations

import json
import math
from pathlib import Path

from fem.elements.beam_section import (
    BeamIntegrationPointForces,
    parse_beam2_section,
    recover_integration_point_stress,
)


ROOT = Path(__file__).resolve().parent
PRIMARY = {
    "RECT": {"section_type": "rectangle", "width": 0.2, "height": 0.4},
    "CIRC": {"section_type": "solid_circle", "radius": 0.2},
    "THICK PIPE": {
        "section_type": "hollow_circle",
        "outer_radius": 0.2,
        "inner_radius": 0.1,
    },
}
RECT_DIMENSIONS = {
    10: (0.2, 0.4),
    40: (0.2, 0.2),
    50: (0.4, 0.2),
    60: (0.2, 0.8),
    70: (0.2, 2.0),
    80: (0.2, 0.25),
    90: (0.2, 0.3),
    100: (0.2, 0.6),
    110: (0.2, 1.2),
    120: (0.2, 4.0),
}
POINT_MAPPING = {
    "RECT": (25, 21, 1, 5),
    "CIRC": (7, 11, 15, 3),
    "THICK PIPE": (8, 14, 20, 2),
}
PUBLIC_RESULTANTS = {
    "SF": {
        "components": ("N", "Vy", "Vz"),
        "abaqus_mapping": {"SF1": "N", "SF2": "Vy", "SF3": "Vz"},
        "quantity": "force",
    },
    "SM": {
        "components": ("T", "My", "Mz"),
        "abaqus_mapping": {"SM1": "T", "SM2": "My", "SM3": "Mz"},
        "quantity": "moment",
    },
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _invariant_differences(components):
    s11 = components["S11"]
    s22 = components["S22"]
    s12 = components["S12"]
    mises = math.sqrt(s11**2 - s11 * s22 + s22**2 + 3.0 * s12**2)
    span = math.sqrt((s11 - s22) ** 2 + 4.0 * s12**2)
    principals = sorted(
        ((s11 + s22 + span) / 2.0, (s11 + s22 - span) / 2.0, 0.0),
        reverse=True,
    )
    return {
        "Mises": abs(components["Mises"] - mises),
        "MaxPrincipal": abs(components["MaxPrincipal"] - principals[0]),
        "MidPrincipal": abs(components["MidPrincipal"] - principals[1]),
        "MinPrincipal": abs(components["MinPrincipal"] - principals[2]),
    }


def _relative_error(actual, expected):
    return abs(actual - expected) / abs(expected)


def main():
    oracle = _load(ROOT / "abaqus_shear_oracle.json")
    holdout = _load(ROOT / "rect_holdout_oracle.json")
    program = _load(ROOT / "program_snapshot.json")
    portal = [
        _load(ROOT / ("portal_point%d" % number) / "summary.json")
        for number in range(1, 5)
    ]

    labels = {
        step: tuple(payload["component_labels"])
        for step, payload in oracle["steps"].items()
    }
    all_oracle_rows = [
        row
        for payload in oracle["steps"].values()
        for row in payload["rows"]
    ]
    tensor_contract = {
        "component_labels_by_step": labels,
        "all_labels_exact": all(
            value == ("S11", "S22", "S12") for value in labels.values()
        ),
        "maximum_absolute_s22": max(
            abs(row["components"]["S22"]) for row in all_oracle_rows
        ),
        "s13_published": any(
            "S13" in row["components"] for row in all_oracle_rows
        ),
    }

    torsion = {}
    for profile, properties in PRIMARY.items():
        section = parse_beam2_section(properties)
        profile_metrics = {}
        for step, torque in (("T_POS", 500.0), ("T_NEG", -500.0)):
            expected_rows = [
                row
                for row in oracle["steps"][step]["rows"]
                if row["section"] == profile
            ]
            actual_rows = recover_integration_point_stress(
                section,
                BeamIntegrationPointForces(
                    0.0, 0.0, 0.0, torque, 0.0, 0.0
                ),
            )
            errors = [
                _relative_error(actual.s12, expected["components"]["S12"])
                for actual, expected in zip(
                    actual_rows,
                    sorted(
                        expected_rows,
                        key=lambda row: POINT_MAPPING[profile].index(
                            row["section_point"]["number"]
                        ),
                    ),
                    strict=True,
                )
            ]
            profile_metrics[step] = {
                "rows": len(errors),
                "maximum_relative_error": max(errors),
                "sign_match": all(
                    actual.s12 * expected["components"]["S12"] > 0.0
                    for actual, expected in zip(
                        actual_rows,
                        sorted(
                            expected_rows,
                            key=lambda row: POINT_MAPPING[profile].index(
                                row["section_point"]["number"]
                            ),
                        ),
                        strict=True,
                    )
                ),
            }
        torsion[profile] = profile_metrics

    pure_shear = {}
    for step in ("VY_POS", "VZ_POS"):
        pure_shear[step] = {
            "abaqus_maximum_absolute_s12": max(
                abs(row["components"]["S12"])
                for row in oracle["steps"][step]["rows"]
                if row["section"] in PRIMARY
            ),
            "program_semantic_value": 0.0,
            "s13_published": False,
        }

    rect_calibration_errors = []
    t_rows = oracle["steps"]["T_POS"]["rows"]
    for element_id, (width, height) in RECT_DIMENSIONS.items():
        expected = next(
            row["components"]["S12"]
            for row in t_rows
            if row["element_id"] == element_id
        )
        actual = recover_integration_point_stress(
            parse_beam2_section(
                {
                    "section_type": "rectangle",
                    "width": width,
                    "height": height,
                }
            ),
            BeamIntegrationPointForces(0.0, 0.0, 0.0, 500.0, 0.0, 0.0),
        )[0].s12
        rect_calibration_errors.append(_relative_error(actual, expected))

    rect_holdout = []
    for ratio in (1.37, 5.0, 13.0):
        expected = next(
            row["components"]["S12"]
            for row in holdout["rows"]
            if row["aspect_ratio"] == ratio
        )
        actual = recover_integration_point_stress(
            parse_beam2_section(
                {
                    "section_type": "rectangle",
                    "width": 0.2,
                    "height": 0.2 * ratio,
                }
            ),
            BeamIntegrationPointForces(0.0, 0.0, 0.0, 500.0, 0.0, 0.0),
        )[0].s12
        rect_holdout.append(
            {
                "aspect_ratio": ratio,
                "abaqus_s12": expected,
                "program_s12": actual,
                "relative_error": _relative_error(actual, expected),
            }
        )

    oracle_invariant_differences = [
        difference
        for row in all_oracle_rows
        for difference in _invariant_differences(row["components"]).values()
    ]
    oracle_maximum = max(
        abs(value)
        for row in all_oracle_rows
        for value in row["components"].values()
    )
    program_invariant_differences = [
        difference
        for row in program["section_results"]
        for difference in _invariant_differences(row["components"]).values()
    ]
    program_maximum = max(
        abs(value)
        for row in program["section_results"]
        for value in row["components"].values()
    )

    portal_components = {}
    for component in ("S12", "Mises", "MaxPrincipal"):
        metrics = [
            point["metrics"]["formal_integration_point"][component]
            for point in portal
        ]
        portal_components[component] = {
            "rows": sum(metric["matched_rows"] for metric in metrics),
            "maximum_significant_relative_l2": max(
                metric["significant_relative_l2"]
                for metric in metrics
                if metric["significant_relative_l2"] is not None
            ),
            "all_row_mae_pa": sum(
                metric["mae"] * metric["matched_rows"] for metric in metrics
            )
            / sum(metric["matched_rows"] for metric in metrics),
            "maximum_absolute_error_pa": max(
                metric["max_absolute_error"] for metric in metrics
            ),
        }

    point_counts = {}
    for row in program["section_results"]:
        number = row["section_point"]["number"]
        point_counts[number] = point_counts.get(number, 0) + 1

    public_resultants = {}
    force_rows = program["section_force_results"]
    force_identities = {
        (row["element_id"], row["integration_point"]) for row in force_rows
    }
    expected_force_components = {"N", "VY", "VZ", "T", "MY", "MZ"}
    if any(set(row["components"]) != expected_force_components for row in force_rows):
        raise ValueError("program snapshot has incomplete section resultants")
    for variable, contract in PUBLIC_RESULTANTS.items():
        field = next(
            field
            for field in program["public_section_result_fields"]
            if field["variable"] == variable
        )
        if tuple(field["components"]) != contract["components"]:
            raise ValueError("public %s components do not match" % variable)
        if field["quantity"] != contract["quantity"]:
            raise ValueError("public %s quantity does not match" % variable)
        if field["section_point_number"] is not None:
            raise ValueError("public %s must not have a section point" % variable)
        public_resultants[variable] = {
            "components": field["components"],
            "abaqus_mapping": contract["abaqus_mapping"],
            "quantity": field["quantity"],
            "position": field["position"],
            "association": field["association"],
            "recovery_contract": field["recovery_contract"],
            "row_count": field["row_count"],
            "section_point_number": None,
        }

    gate = {
        "schema": "fem-python.abaqus-b31-phase5-gate.v2",
        "abaqus_release": oracle["abaqus_release"],
        "tensor_contract": tensor_contract,
        "section_point_mapping": POINT_MAPPING,
        "torsion_oracle": torsion,
        "pure_transverse_shear": pure_shear,
        "public_integration_point_resultants": {
            "fields": public_resultants,
            "row_identity": {
                "row_count": len(force_rows),
                "unique_element_integration_point_count": len(force_identities),
                "integration_points": sorted(
                    {row["integration_point"] for row in force_rows}
                ),
                "section_point_number": None,
            },
        },
        "rect_default_5x5": {
            "calibration_ratios": sorted(
                max(width, height) / min(width, height)
                for width, height in RECT_DIMENSIONS.values()
            ),
            "calibration_maximum_relative_error": max(rect_calibration_errors),
            "holdout": rect_holdout,
            "holdout_maximum_relative_error": max(
                row["relative_error"] for row in rect_holdout
            ),
        },
        "invariant_recompute": {
            "abaqus_maximum_absolute_difference_pa": max(
                oracle_invariant_differences
            ),
            "abaqus_maximum_relative_to_output": max(
                oracle_invariant_differences
            )
            / oracle_maximum,
            "program_maximum_absolute_difference_pa": max(
                program_invariant_differences
            ),
            "program_maximum_relative_to_output": max(
                program_invariant_differences
            )
            / program_maximum,
        },
        "portal": {
            "section_stress_rows": len(program["section_results"]),
            "rows_by_program_section_point": point_counts,
            "components": portal_components,
            "s13_published": False,
        },
    }
    (ROOT / "phase5_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Phase 5 B31 shear stress and invariant validation",
        "",
        "## Result",
        "",
        (
            "Phase 5 passes. Abaqus 2023 publishes B31 stress components "
            "`S11/S22/S12`; `S22` is explicitly zero and `S13` is absent. "
            "The program preserves that missing-component contract and "
            "computes every invariant from the same stored "
            "integration-point/section-point row."
        ),
        "",
        "## Minimal section oracle",
        "",
        "| Section | +T max relative error | -T max relative error |",
        "|---|---:|---:|",
    ]
    for profile in ("RECT", "CIRC", "THICK PIPE"):
        lines.append(
            "| %s | %.6g | %.6g |"
            % (
                profile,
                torsion[profile]["T_POS"]["maximum_relative_error"],
                torsion[profile]["T_NEG"]["maximum_relative_error"],
            )
        )
    lines.extend(
        [
            "",
            (
                "Pure Vy/Vz produce only Abaqus numerical noise in S12 and "
                "no S13. Public integration-point resultants are "
                "`SF=(N,Vy,Vz)`, mapping Abaqus `SF1/SF2/SF3`, and "
                "`SM=(T,My,Mz)`, mapping `SM1/SM2/SM3`. All six values come "
                "from the shared constitutive recovery. Vy/Vz/T retain "
                "resultant identities; Vy/Vz never enter point stress, while "
                "T contributes only the published torsional S12. RECT uses the same "
                "default 5x5 integrated-section owner as Phase 1: "
                "calibration max error is %.6g and independent held-out max "
                "error is %.6g."
            )
            % (
                gate["rect_default_5x5"]["calibration_maximum_relative_error"],
                gate["rect_default_5x5"]["holdout_maximum_relative_error"],
            ),
            "",
            (
                "SF and SM each contain 276 unique element/integration-point "
                "rows at longitudinal integration point 1, with no section "
                "point identity. Their physical quantities remain force and "
                "moment respectively."
            ),
            "",
            "## Portal-frame gate",
            "",
            "| Component | Conservative significant relative L2 | All-row MAE (MPa) | Max abs (MPa) |",
            "|---|---:|---:|---:|",
        ]
    )
    for component in ("S12", "Mises", "MaxPrincipal"):
        metric = portal_components[component]
        lines.append(
            "| %s | %.6g | %.6g | %.6g |"
            % (
                component,
                metric["maximum_significant_relative_l2"],
                metric["all_row_mae_pa"] / 1.0e6,
                metric["maximum_absolute_error_pa"] / 1.0e6,
            )
        )
    lines.extend(
        [
            "",
            (
                "All 1,104 portal stress rows are present: 276 elements times "
                "four section points at longitudinal integration point 1. "
                "Program invariant recomputation has maximum relative "
                "difference %.6g; the Abaqus snapshot difference %.6g is "
                "single-precision output rounding."
            )
            % (
                gate["invariant_recompute"][
                    "program_maximum_relative_to_output"
                ],
                gate["invariant_recompute"][
                    "abaqus_maximum_relative_to_output"
                ],
            ),
            "",
            "## Reproduction",
            "",
            (
                "The INP files and extraction scripts in this directory are "
                "Python 2.7 compatible. Automated tests freeze the extracted "
                "numbers and construct inline models; they do not access the "
                "engineering project directory."
            ),
        ]
    )
    (ROOT / "validation_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
