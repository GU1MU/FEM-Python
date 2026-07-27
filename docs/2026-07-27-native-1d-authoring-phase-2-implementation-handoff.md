# Native 1D Authoring Phase 2 Implementation Handoff

Date: 2026-07-27

## Scope completed

Phase 2 extends the Phase 1 native authoring contract through the headless
workflow while keeping GUI wire editing out of scope.

- `NativeMeshContract` is the single dimension/shape/order/formulation
  resolver. Incomplete `WireGeometry` intent remains incomplete until
  `Truss2` or `Beam2` is explicit.
- `WireGeometry` compiles to one shared CAD point per declared point and one
  CAD line per declared member. Point, member, and domain references are
  audited after Gmsh import.
- Native line preprocessing imports first-order `Truss2` and `Beam2` meshes,
  produces stable node/element sets, and rejects unexpected edges/surfaces.
- Native regions, capabilities, project validation, Beam orientation, and
  Beam-only line loads follow the selected formulation.
- Schema v3 persists wire geometry, line formulation, topology links, local
  mesh controls, named regions, definitions, loads, outputs, and nullable
  mesh settings. The generic router reads schemas 1/2/3 and writes schema 3.
  The explicit schema-v2 codec remains wire-incompatible and frozen.
- GUI project-open rejects native 1D projects with
  `native_1d.gui_pending` before `Session.replace_from_snapshot()`; the
  current Session and viewport remain unchanged.
- Truss2 and Beam2 vertical slices cover generation, formulation DOFs,
  analytical response checks, result-provider fields, CSV/VTK export, and a
  schema-v3 reopen/regenerate/validate/solve path.

## Verification

- Phase 1 baseline from the preceding handoff: `4450 passed, 6 skipped`.
- Focused Phase 2 tests cover application contracts, native regions and
  capabilities, project codecs/router/migration, GUI project I/O, native
  preprocessing, low-level Gmsh line behavior, and the Truss2/Beam2 vertical
  slices.
- The final focused regression after the review fixes passed: native
  preprocessing boundary `3 passed`, schema v3 `6 passed`, and the
  capabilities/session projection batch `32 passed`.
- The final full suite reported `4473 passed, 6 skipped, 2 failed` in
  `178.87s`. The two failures are the pre-existing architecture environment
  contract tests that invoke `git` through a child process; both raised
  Windows `FileNotFoundError` inside the managed full-suite process. Running
  `tests/architecture/test_environment_contracts.py` alone passed `7 passed`,
  and no Phase 2 source or test change touches those checks.
- Gmsh version: `4.15.2`.
- Ruff is run from the global environment because the project virtual
  environment does not provide the module; all changed Python files pass.
- Independent-agent review found two P1 issues in the first pass. They were
  fixed with regression tests for strict member-chain/ownership auditing and
  duplicate schema-v3 topology IDs. The follow-up review found no P0/P1
  blocker and explicitly confirmed that capabilities now use the shared mesh
  contract without an independent shape dispatch table.

## Phase 3 boundary

The GUI still does not create, preview, edit, pick, or mesh native wire
geometry. Phase 3 must reuse the Phase 1 DTOs and the Phase 2 headless
services for those workflows.
