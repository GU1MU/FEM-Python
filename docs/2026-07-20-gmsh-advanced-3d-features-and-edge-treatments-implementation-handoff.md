# Gmsh Advanced 3D Features and Edge Treatments Implementation Handoff

## Prerequisite Baseline

- Plan 1 baseline commit: `4293a1a653584f5d884b499b7952b2e2fe4bd076`
- Supported Gmsh version at the start of Plan 2: `4.15.2`
- Plan 2 implementation started from branch `develop` at that exact commit.
- The initial worktree already contained a user-owned deletion of
  `docs/2026-07-20-gmsh-structured-feature-results-and-foundational-operations-implementation-handoff.md`;
  Plan 2 does not restore, modify, or include that deletion.

## Verification Results

### Prerequisite gate

- Bytecode compilation: passed with `PYTHONPYCACHEPREFIX` redirected to a
  writable workspace directory because pre-existing cache files were locked.
- Focused architecture, geometry, meshing, and IO suite: `737 passed`.
- Complete baseline suite: `1374 passed`.
- Production and example sources contain no `fem.meshing` imports; the only
  matching text is the architecture regression assertion that forbids them.

## Implemented Public Contract

- Added owner-bound, frozen `WireRef` values for ordered open and closed OCC
  wires. Validation covers ownership, liveness, duplicate members, oriented
  endpoint continuity, 2D planarity, and dependency-scoped invalidation.
- Added `GeometryModel.revolve()` for curve-to-surface and surface-to-volume
  features, including partial, negative, full-turn, and global-XY 2D cases.
- Added `GeometryModel.sweep()` for curve and surface profiles along open or
  closed `WireRef` paths. The public `SweepFrame` values and native mappings
  are:
  - `discrete` -> `DiscreteTrihedron`
  - `corrected_frenet` -> `CorrectedFrenet`
  - `frenet` -> `Frenet`
  - `fixed` -> `Fixed`
  - `constant_normal` -> `ConstantNormal`
  - `darboux` -> `Darboux`
- Added `GeometryModel.loft()` for solid closed-section and surface
  open-section workflows, with `ruled`, `max_degree`, and `smoothing` options.
  Supported continuity values are `C0`, `G1`, `C1`, `G2`, `C2`, `C3`, and
  `CN`; supported parametrizations are `chord_length`, `centripetal`, and
  `iso_parametric`. Every typed continuity and parametrization value was
  forwarded through real Gmsh. The accepted combined smoothing case was
  exercised with a surface loft, `max_degree=3`, `continuity="C1"`,
  `parametrization="chord_length"`, and `smoothing=True`.
- Added volume `GeometryModel.fillet()` and `GeometryModel.chamfer()` with
  selected-volume closure and adjacency preflight, destructive and preserving
  modes, single/multiple-volume support, and native value vectors of lengths
  1, N, and 2N where Gmsh accepts them.
- Generalized `FeatureResult` for dimension-raising features and
  same-dimensional replacement features. Added one `LoftResult` wrapper
  because flattened `FeatureResult.inputs` cannot retain the ordered grouping
  and identity of the section wires; it delegates the topology fields and
  stores `sections` explicitly.
- All native-mutation paths use owner-local wrapping and fail-closed
  invalidation. Numeric OCC tags remain non-semantic and may be reused with a
  fresh typed identity.

## Architecture Audit

- Public exports and runtime annotations include `WireRef`, `FeatureResult`,
  `LoftResult`, `SweepFrame`, `LoftContinuity`, and `LoftParametrization`.
- `wire`, `revolve`, `sweep`, `loft`, `fillet`, and `chamfer` are geometry
  operations and are absent from `fem.mesh.gmsh.Mesher`.
- `fem.geometry.gmsh` has no FEM runtime or IO imports, and none of the new
  operations accepts mesh-control arguments.
- The new examples use `fem.geometry.gmsh`, `fem.mesh.gmsh`, and
  `fem.io.gmsh` explicitly. They contain no `raw_occ` or physical-group use.
- Production and example sources contain no `fem.meshing` imports; the only
  matching repository text remains the architecture regression assertion that
  forbids them.

## Final Verification

- Supported Gmsh version: `4.15.2`.
- Ruff: `ruff check src tests examples` passed.
- Bytecode compilation: `python -m compileall -q src tests examples` passed
  with `PYTHONPYCACHEPREFIX` redirected to a writable workspace directory.
- Advanced real/fake backend suite: `50 passed`.
- The plan's documented focused command: `802 passed`.
- Expanded architecture and all-Gmsh focused suite: `894 passed`.
- Complete repository suite: `1489 passed`.
- `git diff --check` passed.

### End-to-end examples

- `gmsh_geometry_revolved_solid.py`: solved an imported mesh with 145 nodes
  and 379 Tet4 elements, verified finite displacement/reaction arrays, and
  wrote `results/gmsh_geometry_revolved_solid/gmsh_geometry_revolved_solid.vtk`.
- `gmsh_geometry_swept_solid.py`: generated and imported 60 nodes and 127 Tet4
  elements.
- `gmsh_geometry_filleted_box.py`: generated and imported 139 nodes and 358
  Tet4 elements.
- Real-Gmsh tests additionally mesh and import representative lofted and
  chamfered topology.

## Intentionally Deferred

The plan's declared out-of-scope behavior remains deferred: structured layer
revolution/sweep; guide/contact and point-sweep variants; unrestricted 2D pipe
semantics; shell, draft, thickening, offset, and midsurface features; 2D sketch
fillet/chamfer; face-face variable fillet laws beyond Gmsh radius vectors;
stable topological naming and persistent history; physical-group transport and
automatic FEM regions; CAD import/healing/defeaturing/deduplication; feature-
specific new FEM cell types; additional CAD backends, GUI modeling, and undo.
No README or historical plan document was changed.
