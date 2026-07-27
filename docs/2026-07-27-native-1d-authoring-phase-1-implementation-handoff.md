# Native 1D Authoring Phase 1: Implementation Handoff

## Status

Phase 1 is implemented. The headless authoring layer now accepts validated
spatial straight-member wire recipes and explicit homogeneous line-mesh
formulation intent while keeping execution, persistence, capability, and GUI
support behind the Phase 2/3 boundary.

## Delivered contract

- Added frozen, slotted `WirePoint`, `WireMember`, and `WireGeometry` values,
  with strict string and finite-real normalization, tuple ownership, and graph
  validation.
- Added `WireGeometry` to the native geometry unions and public
  `fem.geometry` exports.
- Extended dimension-aware topology and recipe projections to dimension one:
  stable `point:<name>`, `edge:<name>`, and `body:domain` entities,
  reorder-independent fingerprints that include canonical member endpoint
  links, maximum-member-length characteristic size, wire feature history, and
  generic point/member/domain regions.
- Added explicit `MeshSettings.line_element_type` after the existing fields.
  Line settings require `Truss2` or `Beam2` and first order; continuum settings
  cannot carry a line formulation.
- Updated Session transitions so a new wire without compatible explicit intent
  keeps `mesh_settings=None`, coordinate edits preserve valid line controls,
  topology edits clear entity controls while retaining formulation, and
  dimension transitions replace mesh fields atomically.
- Added the pre-Gmsh native-1D preprocessing guard with the explicit Phase 2
  diagnostic.
- Kept v2 strict and non-persistable for wire state. The v1 encoder recognizes
  the appended in-memory field for existing continuum compatibility and fails
  closed if a non-`None` line formulation is supplied; it does not add wire
  persistence.
- Did not add compiler, capability, solver, Gmsh, or GUI workflow support.

## Boundary behavior

- Rigid move and rotate wrappers preserve wire logical identities.
- Wire extrusion and all one-dimensional Boolean operations are rejected.
- Target-radius falloff on wire point/member targets reports an unsupported
  target diagnostic.
- Native preprocessing rejects a wire before importing or initializing Gmsh.
- Schema-v2 encoding rejects a wire snapshot instead of producing a partial
  project file.

## Verification

The project virtual environment does not contain Ruff, so the configured global
Ruff executable was used as allowed by the repository instructions.

- Scoped Ruff check over every changed Python file: `All checks passed!`
- Focused geometry, mesh, session, v1-session, public-export, and boundary
  tests after the final session guard: `89 passed`.
- Existing regression batches completed before the final one-test addition:
  `1008 passed`, `31 passed`, `237 passed, 4 skipped`, and `499 passed` for
  application/geometry/mesh/io, integration, characterization/materials/post/
  performance, and GUI coverage respectively.
- The final top-level `tests/test_*.py` batch completed with exit code `0`.
- Complete post-fix `pytest -q`: `4450 passed, 6 skipped in 155.07s`.
- `git diff --check` is required before staging.

The repository-wide Ruff command still reports eight pre-existing violations in
unmodified agent/tool test files. None are in the Phase 1 change set; the
changed-file Ruff check is clean.

## Phase 2 entry points

Phase 2 can consume the existing stable IDs and settings directly:

1. Compile one owned OCC point per `WirePoint` and one straight OCC line per
   `WireMember`.
2. Bind the point, member, and domain logical IDs to native entities.
3. Apply global and supported local line sizing.
4. Generate dimension-one Gmsh meshes and pass the explicit formulation to the
   existing importer.
5. Add native-1D capability, formulation-aware validation, and a new strict
   persistence schema together with headless solve/export coverage.

GUI draft and editing work remains Phase 3.
