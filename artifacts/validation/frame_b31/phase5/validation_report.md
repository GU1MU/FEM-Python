# Phase 5 B31 shear stress and invariant validation

## Result

Phase 5 passes. Abaqus 2023 publishes B31 stress components `S11/S22/S12`; `S22` is explicitly zero and `S13` is absent. The program preserves that missing-component contract and computes every invariant from the same stored integration-point/section-point row.

## Minimal section oracle

| Section | +T max relative error | -T max relative error |
|---|---:|---:|
| RECT | 0.000395111 | 0.000395111 |
| CIRC | 3.51349e-08 | 3.51349e-08 |
| THICK PIPE | 1.05912e-08 | 1.05912e-08 |

Pure Vy/Vz produce only Abaqus numerical noise in S12 and no S13. Public integration-point resultants are `SF=(N,Vy,Vz)`, mapping Abaqus `SF1/SF2/SF3`, and `SM=(T,My,Mz)`, mapping `SM1/SM2/SM3`. All six values come from the shared constitutive recovery. Vy/Vz/T retain resultant identities; Vy/Vz never enter point stress, while T contributes only the published torsional S12. RECT uses the same default 5x5 integrated-section owner as Phase 1: calibration max error is 0.00499483 and independent held-out max error is 0.00248692.

SF and SM each contain 276 unique element/integration-point rows at longitudinal integration point 1, with no section point identity. Their physical quantities remain force and moment respectively.

## Portal-frame gate

| Component | Conservative significant relative L2 | All-row MAE (MPa) | Max abs (MPa) |
|---|---:|---:|---:|
| S12 | 1.16182e-07 | 3.72656e-05 | 0.000275622 |
| Mises | 2.4386e-06 | 5.58065e-06 | 0.000218528 |
| MaxPrincipal | 1.76371e-06 | 3.54147e-06 | 0.000128698 |

All 1,104 portal stress rows are present: 276 elements times four section points at longitudinal integration point 1. Program invariant recomputation has maximum relative difference 0; the Abaqus snapshot difference 7.95213e-08 is single-precision output rounding.

## Reproduction

The INP files and extraction scripts in this directory are Python 2.7 compatible. Automated tests freeze the extracted numbers and construct inline models; they do not access the engineering project directory.
