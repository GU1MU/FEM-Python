# Phase 3 B31 frame and nodal-normal validation

## Outcome

Phase 3 passes. The existing element-end frame resolver already matches the
Abaqus 2023 oracle, so this phase makes no frame-formula change. The production
change canonicalizes B31 Abaqus node/element/set identities and removes the
retired `abaqus.b31.nodal_normal_generation_approximation` notice. The remaining
`abaqus.b31.linear_timoshenko_support_boundary` notice describes capabilities
outside this phase and remains present.

## Frozen semantics

The implementation follows the official Abaqus beam-orientation contract:

- the right-handed axes are `(t, n1, n2)`, with `t` directed from the first
  structural node to the second;
- an additional orientation node defines approximate `n1` from the first node
  and takes precedence over section `n1`;
- section `n1` takes precedence over the default `(0, 0, -1)`;
- a user `*NORMAL` for an exact element/node identity takes precedence over a
  normal supplied on `*NODE`;
- the remaining generated normals use the official 20-degree connected-group
  algorithm; a connected non-clique is fully split, separate cliques remain
  separate, and more than 30 remaining elements are not averaged;
- generated groups and final frames are ordered by exact
  `(element_id, local_end, node_id)` identity. One shared node can therefore
  retain multiple member-direction groups.

Official sources:

- <https://docs.software.vt.edu/abaqusv2025/English/SIMACAEELMRefMap/simaelm-c-beamcrosssection.htm>
- <https://docs.software.vt.edu/abaqusv2024/English/SIMACAEMODRefMap/simamod-c-nodalnormals.htm>
- <https://docs.software.vt.edu/abaqusv2024/English/SIMACAEKEYRefMap/simakey-r-normal.htm>

The shared numerical policy remains `1e-8` for a nonparallel direction and
`1e-10` for direction comparison. Zero, non-finite, parallel, missing, and
conflicting sources continue to fail closed through typed import errors.

## Abaqus frame oracle

`phase3_frame_sources.inp` covers default and section `n1`, an orientation node,
node normals, `*NORMAL`, reversed connectivity, a three-member averaged group,
a non-clique split group, and two disjoint groups. Abaqus 2023 was run with full
output precision. For each fixed-free B31, three unit global tip-load steps
recover the sole longitudinal integration-point frame from
`SF=(t dot p, n2 dot p, n1 dot p)`.

The comparison in `frame_gate.json` covers 14 elements. The maximum absolute
direction-cosine error is `5.473399511402022e-14`, below the `1e-10` gate. Case
maxima are:

- default `n1`: `2.50e-16`;
- orientation node over section `n1`: `1.69e-15`;
- node normal over generated normal: `2.50e-16`;
- `*NORMAL` over node normal: `1.05e-15`;
- reversed connectivity: `1.69e-15`;
- averaged-normal cases: `2.81e-14`;
- non-clique split cases: `2.94e-14`;
- disjoint-group cases: `5.47e-14`.

## Portal-frame gate

The normal Phase 3 portal report was regenerated in this directory with 117
matched nodes and 276 target integration points. All response gates pass:

| Gate | Actual | Threshold |
| --- | ---: | ---: |
| translation vector relative L2 | `1.19436e-7` | `< 1.0e-2` |
| significant-node max vector relative | `1.65055e-7` | `< 3.0e-2` |
| U2 relative L2 | `8.42047e-7` | `< 1.5e-1` |
| U2 max absolute error | `2.99299e-8 mm` | `< 0.010 mm` |
| rotation vector relative L2 | `1.46143e-7` | `< 5.0e-2` |
| UR1 max absolute error | `3.50152e-11 rad` | `< 5e-5 rad` |
| UR3 max absolute error | `9.57345e-12 rad` | `< 5e-5 rad` |
| roof-ridge rotation vector relative L2 | `3.47289e-7` | `< 1.5e-1` |

The Phase 2 and Phase 3 program snapshots have an exact maximum `U/UR` change
of `0.0`. This establishes that no extra portal-frame direction capability was
needed. The formal integration-point stress row count remains zero and the
diagnostic section-end-to-element-nodal S11 significant relative L2 error is
`0.6660`; the residual mismatch is owned by the planned Phase 4/5 result
recovery work.

## Focused verification

- `219 passed` across Phase 1/2/3/normal-group, line topology, builder report,
  Beam frame, stiffness, local-load, recovery, and architecture-focused tests;
- the narrowed slenderness-gate architecture assertion passed alone, followed
  by `129 passed, 1 deselected` across the complete Phase 3-related files;
- the B31 GUI import notice projection test also passed; two heavier GUI
  integration cases were deselected by their existing environment gate;
- all new Phase 3 input tests write independent UTF-8 minimal INP files under
  `tmp_path` and do not read or derive the repository `data/` baselines.

Reproduction commands and machine-readable gate values are preserved in
`extract_phase3_frame_oracle.py`, `compare_phase3_frame_oracle.py`,
`build_phase3_gate_report.py`, `frame_gate.json`, and `phase3_gate.json`.
