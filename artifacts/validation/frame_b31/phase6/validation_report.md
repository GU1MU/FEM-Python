# Phase 6 B31 end-to-end validation

## Result

Phase 6 passes all 32 final numerical gates. The comparison preserves
the Phase 0 identity, thresholds, step/frame selection, and section-point
coordinate mapping; no fitted scaling is applied.

## Phase 0 to final

| Measure | Phase 0 | Final |
|---|---:|---:|
| Translation vector relative L2 | 0.0304111949 | 1.19435853e-07 |
| Rotation vector relative L2 | 0.216320402 | 1.46142988e-07 |
| Formal point-stress rows | 0 | 1104 |
| S11 significant relative L2 (worst point) | unavailable | 1.43879662e-07 |
| S12 significant relative L2 | unavailable | 1.16182345e-07 |
| Mises significant relative L2 | unavailable | 2.43859945e-06 |
| MaxPrincipal significant relative L2 | unavailable | 1.76371313e-06 |

Phase 0 published only section-end stress diagnostics, so it had no formal
integration-point/section-point stress observation that could be compared to
Abaqus S. The final result contains 276 unique elements and 1,104 point-stress
rows (four section points at longitudinal integration point 1).

## Final response and stress

- Translation vector relative L2: 1.19435853e-07;
  U1/U2/U3 relative L2: 9.88897462e-08,
  8.42046955e-07, and
  1.22272014e-07.
- Rotation vector relative L2: 1.46142988e-07;
  UR1/UR3 maximum absolute errors are
  3.5015182e-11 rad and
  9.57344511e-12 rad.
- S11 worst-point relative L2/MAE/max: 1.43879662e-07,
  0.361646814 Pa, and 2.1214111 Pa.
- S12 relative L2/MAE: 1.16182345e-07
  and 37.2656049 Pa.
- Mises relative L2/MAE: 2.43859945e-06
  and 5.58064617 Pa.
- MaxPrincipal relative L2/MAE:
  1.76371313e-06 and
  3.54146693 Pa.

All five member groups are present. The largest point-1 group S11 MAE is
1.01562732
Pa. Public SF=(N,Vy,Vz) and SM=(T,My,Mz) each contain 276 unique integration-
point rows under recovery contract 4.

## Balance, frames, and reproducibility

Program force and moment equilibrium relative residuals are
2.4426796e-14 and
2.33062057e-14.
The maximum frame direction-cosine difference is
5.47339951e-14.
Program invariant recomputation is exact; the Abaqus snapshot's maximum
relative recomputation difference is
7.95212958e-08,
consistent with single-precision output rounding.

`final_gate.json` records every threshold and actual value. `summary.json`
records the Phase 0/final comparison, formulation provenance, member groups,
public resultants, balance, and SHA-256 hashes of every source report.
