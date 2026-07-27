# Native 1D Authoring Phase 3: GUI Implementation Handoff

## Scope

Phase 3 exposes the landed native `WireGeometry` and explicit line-mesh
contracts through the existing FEM GUI workflow. The implementation keeps an
incomplete graph detached from `ModelSession`; only a valid Finish operation
submits one revision-bound `NativeGeometryEdit`.

The Phase 1 and Phase 2 prerequisite commits were already present at the
start of this work:

- Phase 1: `d00c704` — native 1D wire authoring contract;
- Phase 2: `466fe39` — native 1D headless authoring and GUI-open guard;
- Gmsh used by the focused verification environment: `4.15.2`.

## Delivered behavior

- `GeometryPreview` now carries an explicit topological dimension and builds
  face-free Wire previews with declared point/member order and stable
  `point:<name>`, `edge:<name>`, and `body:domain` identities.
- One-dimensional committed previews render member edges and visible joint
  markers. Point, member, and domain-body picking/highlighting use the existing
  logical reference pipeline; face selection is disabled with the planned
  diagnostic.
- `WireDraftController` and immutable draft values support incomplete graphs,
  deterministic P/M naming, table edits, endpoint rewrites, deletion guards,
  coincident-point reporting, finish diagnostics, detached snapshots, and
  strict conversion back to `WireGeometry`.
- The non-modal `WireEditorPanel` shares one draft controller with the main
  viewport. It provides point/member/select modes, XY/XZ/YZ work planes,
  offsets, optional snapping, point/member tables, validation, coincident
  point confirmation, and Finish/Cancel.
- The viewport authoring mode routes point placement and two-click member
  drawing through typed signals. Enter/Escape precedence, pending interaction
  cleanup, and Ctrl+Alt camera controls remain separate from ordinary FEM and
  geometry picking.
- Geometry actions include `New Wire`; active editing disables state-changing
  project, geometry, mesh, definition, and analysis actions while leaving
  standard view actions available. Wire move/rotate are spatial; extrusion and
  boolean actions are unavailable.
- Geometry Manager can edit a root Wire directly or through move/rotate
  wrappers. Cancel restores the Session projection; Finish preserves the
  wrapper chain and relies on Session reference effects.
- `MeshSettingsDialog` has an explicit dimension-one branch with no default
  formulation for a new Wire, fixed line/first-order controls, and Truss2 or
  Beam2 selection. `MeshControlsDialog` preserves `line_element_type`.
- Native Wire project opening no longer returns `native_1d.gui_pending`; the
  normal schema-v3 Session projection rebuilds the committed Wire preview.
- Wire point/member/body region and analysis selection paths use node-set or
  element-set products as supplied by the existing capability descriptors.
  Member/domain Beam line loads are not coerced to continuum edge loads, and
  Wire displacement selection defaults to joints.

## Verification

Focused commands used the project `.venv` because the configured `uv` cache
was not accessible in this workspace.

- Phase 1/2 prerequisite and focused GUI batch: **214 passed**.
- Main-window layout, preprocessing, Session projection, line-element, and
  native GUI projection batch: **95 passed**.
- Changed-file Ruff check: **passed**.
- `git diff --check`: **passed**.
- Post-review GUI regression batch: **107 passed**; existing imported line,
  result, and end-to-end workflow batch: **12 passed**.
- Full pytest was intentionally not run because the repository’s full suite is
  materially more expensive and the requested Phase 3 risk was covered by the
  focused matrix above.

The repository-wide Ruff command still reports eight pre-existing findings in
`src/fem_agent/engine.py`, `src/fem_agent/tools/results.py`,
`tests/test_agent_tool_registry.py`, and `tests/test_gmsh_geometry.py`; none
are in the Phase 3 change set.

An independent read-only broad review found no P0 issue. Its two actionable
P1 findings were fixed before handoff: discard confirmation is now atomic
across both draft and document prompts, and incremental draft redraws retain
the current camera instead of refitting on every edit.

## Deferred or intentionally unchanged

- No new element kernel, Gmsh importer, solver, result schema, README, or CI
  behavior was added.
- The editor uses straight two-point members only. Curves, releases, offsets,
  mixed formulations, second-order line cells, and exact member-division
  controls remain outside Phase 3.
- Full end-to-end solve/result GUI exercise and manual high-DPI Windows
  interaction remain follow-up validation; the existing imported line and
  native GUI projection tests remain green in the focused run.
