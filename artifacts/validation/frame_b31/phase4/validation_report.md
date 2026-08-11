# Phase 4 B31 integration-point S11 validation

## Result

Phase 4 passes. FEM-Python publishes exactly one longitudinal integration
point for each of the 276 Beam2 elements and keeps each of the four section
coordinates as a qualified identity on that integration point. The public
Beam `S` catalog no longer publishes the two section-end action rows.

The frozen rectangle mapping is:

| Program point | Local coordinate | Abaqus RECT point |
|---:|---|---:|
| 1 | (+y, +z) | 25 |
| 2 | (-y, +z) | 21 |
| 3 | (-y, -z) | 1 |
| 4 | (+y, -z) | 5 |

## Portal-frame metrics

All stress values below are in Pa unless an MPa conversion is shown.

| Metric | Actual | Gate |
|---|---:|---:|
| Point 1 S11 significant relative L2 | 1.38644e-7 | < 2.0e-2 |
| Point 1 S11 all-row MAE | 0.346299 Pa (3.46299e-7 MPa) | < 0.20 MPa |
| Point 1 S11 maximum absolute error | 2.12141 Pa (2.12141e-6 MPa) | < 1.0 MPa |
| N relative L2 | 1.82083e-7 | < 1.5e-2 |
| My relative L2 | 1.47904e-6 | < 1.0e-2 |
| Mz relative L2 | 1.06869e-7 | < 1.0e-2 |

The Abaqus ODB reports `SF=false` and `SM=false`. Therefore the N/My/Mz
reference is derived, without fitting, from the four matched corner S11 values
at the same element and longitudinal integration point:

- N = A (S(+,+) + S(-,+) + S(-,-) + S(+,-)) / 4
- My = Iyy ((S(+,+) + S(-,+)) - (S(-,-) + S(+,-))) / (2 h)
- Mz = Izz ((S(-,+) + S(-,-)) - (S(+,+) + S(+,-))) / (2 w)

Here width w spans local y and height h spans local z. These equations are the
inverse of S11 = N/A + My z/Iyy - Mz y/Izz and preserve the program's frozen
local signs. Every reconstruction has one unique Abaqus record at integration
point 1 for each of the 276 elements.

### Five member groups, program point 1

| Group | Rows | S11 MAE (Pa) | S11 MAE (MPa) | Max abs (Pa) |
|---|---:|---:|---:|---:|
| Columns | 36 | 0.140387 | 1.40387e-7 | 0.495800 |
| Arch ribs | 72 | 1.01563 | 1.01563e-6 | 2.12141 |
| Purlins | 72 | 0.110487 | 1.10487e-7 | 0.334206 |
| Side rails | 32 | 0.0379499 | 3.79499e-8 | 0.104950 |
| Roof bracing | 64 | 0.128591 | 1.28591e-7 | 0.442039 |

All five group MAEs are below 0.35 MPa. No group or near-zero record is
silently removed from MAE/max reporting.

## Identity, ownership, and balance

- Program section rows: 1,104 = 276 elements x 4 section points.
- Program SF/SM source rows: 276 = one unqualified longitudinal integration
  point per element; N/My/Mz are not repeated as section-point fields.
- Unique longitudinal integration points: 276; missing, extra, and duplicate
  element integration points: zero.
- N/My/Mz and S11 originate from the same midpoint B31 `B` matrix and material
  matrix used by stiffness assembly (`D @ B @ u`).
- Section-end actions remain a distinct type and continue to use
  `k @ u - f_eq`; public integration-point recovery does not call that path.
- Program force-equilibrium relative L2 residual: 2.44e-14.
- Program moment-equilibrium relative L2 residual: 2.33e-14.
- ODB force/moment comparisons retain Abaqus single-precision extraction
  differences in `phase4_gate.json`; program internal balance is double
  precision.

## Result-system contract

The four public `S` fields use `FieldPosition.INTEGRATION_POINT`, integration
point number 1, and an explicit section-point number and local y/z coordinate.
The public `SF` field is a single integration-point `FORCE` field with component
`N`; the public `SM` field is a single integration-point `MOMENT` field with
components `My/Mz`. Each has 276 rows and neither is repeated for the four
section points. These six exact identities are recorded under
`identity.public_section_result_fields` in `phase4_gate.json` from the
production result registry used by the program snapshot exporter.
Query, CSV, result archive, VTK projection, tree/ribbon selection, inspection,
and viewport labels consume this stored identity. Schema-v1 section-end archive
fields with recovery contract 1 remain readable as `SECTION_END`; the loader
retains that position and provenance and does not fabricate integration points.

S12, S13, Mises, and principal-stress equivalence remain assigned to Phase 5.
They are deliberately absent from the new Phase 4 integration-point descriptor.

## Reproduction

```text
.venv\Scripts\python.exe scripts\export_frame_b31_program_snapshot.py --inp data\portal_frame_b31_wind_snow.inp --output artifacts\validation\frame_b31\phase4\program_snapshot.json

abaqus python scripts\compare_frame_b31_odb.py --odb data\frame_b31.odb --program-snapshot artifacts\validation\frame_b31\phase4\program_snapshot.json --output-directory <point-output> --program-section-point-number <1..4> --odb-section-point-number <25|21|1|5> --section-type RECT --expected-nodes 117 --expected-elements 276

.venv\Scripts\python.exe artifacts\validation\frame_b31\phase4\build_phase4_gate_report.py
```

The portal INP and ODB are read only by these manual engineering-validation
commands. Automated tests construct independent minimal models and do not read,
copy, or derive any file under `data/`.
