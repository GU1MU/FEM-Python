# `fem.geometry` Stateful Runtime and Meshing Port Phase 3 Implementation Handoff

## Outcome

Phase 3 is implemented as a behavior-preserving extraction of the stateful
Gmsh runtime boundary from `GeometryModel`. Model lifecycle state, native
session/model/option ownership, typed geometry-reference identity, committed
mesh-control topology dependencies, and the geometry-owned meshing capability
now have separate private owners.

`fem.mesh.gmsh.Mesher` retains only one `_BoundMeshingPort` and reaches
geometry through the single `_acquire_meshing_port()` seam. Mesh-only errors,
fields, policies, generated-mesh references, and native generation remain in
`_gmsh.model` for Phase 4, as planned.

The exact 16-name `fem.geometry` API, exact 10-name `fem.mesh.gmsh` API, and
all public signatures are unchanged. External `gmsh` remains lazily loaded.

The implementation is committed as one Phase 3 change. The pre-existing
user-owned deletion of
`docs/2026-07-20-gmsh-structured-feature-results-and-foundational-operations-implementation-handoff.md`
was left untouched.

## Starting Boundary

Implementation started from:

~~~text
45f0058b77f8e0f0ed24a7bec49add359cb594c8
45f0058 refactor(geometry): isolate gmsh backend and stateless core
~~~

The verified environment is:

~~~text
Python 3.13.11
Gmsh 4.15.2
pytest 9.1.1
Ruff 0.14.10
~~~

## Implementation

### Model state

`src/fem/geometry/_gmsh/state.py` owns `_State`, the three exact allowed-state
sets, and `_ModelStateMachine`. `GeometryModel` delegates allowed-operation
checks, contextual state errors, legal transitions, and idempotent public close
to that object. Mesh-attempt, generated-mesh, bound-port, and structured-
extrusion subphase state remain model-local for Phase 4.

### Native session and numeric options

`src/fem/geometry/_gmsh/session.py` owns lazy backend loading, process-global
session acquisition, facade-model creation and activation, prior-current model
restoration, facade-only removal, owned finalization, retryable ownership, and
the per-model pending numeric-option ledger.

Entry order remains backend load, initialized-state inspection/initialization,
prior-current and model-list capture, name validation/collision rejection,
model add, and activation. Owned-session state is recorded before
`initialize()`, so a partially successful initialize can be finalized.

`activate(operation)` is the session's exclusive raw-facade return gate.
Context entry returns no facade, the session exposes no facade accessor, and
the model's private native accessor delegates through activation. Generated-
mesh borrowing activates inside its stale-handle exception-translation guard.

Numeric-option transactions read every original before applying replacements,
write replacements in request order, restore every pending entry after later
failure, remove each successfully restored entry immediately, and retain only
failed restorations for retry. Nested model sessions keep independent ledgers.

### Intentional cleanup correction

Gmsh initialized-state inspection is now part of captured cleanup. An entry or
context-body exception remains primary and receives every cleanup failure as a
note. With no primary exception, cleanup raises one contextual `GeometryError`
from the first failure and retains later failures as notes. All safe cleanup
steps continue after earlier failures.

An inspection failure is reported as `inspect Gmsh session state`. Because the
native state cannot be confirmed in that case, resource ownership is retained
and finalization is skipped. A later explicit `__exit__()` retries cleanup.
This is the phase's only intended observable behavior correction.

### Typed reference identity

`src/fem/geometry/_gmsh/reference_registry.py` contains:

- `_EntityRegistry`, which owns the model owner token and stable entity tokens;
- `_TopologyReferenceRegistry`, which owns separate curve-loop and wire tag,
  token, and dependency namespaces;
- `_ReferenceRegistry`, which composes entity and topology invalidation.

`GeometryModel` no longer stores `_owner_token`, `_entity_tokens`,
`_curve_loop_tokens`, `_curve_loop_dependencies`, `_wire_tokens`, or
`_wire_dependencies`. Native liveness and recursive-boundary queries remain in
the model and pass normalized keys/dependencies to the registry.

Targeted entity invalidation also invalidates dependent loops and wires;
topology-only invalidation preserves entity identity; malformed or duplicate
loop/wire identities clear only their own namespace; raw access and unknown OCC
mutation clear all typed geometry identity. Native wire failure retains its
existing global fail-closed asymmetry.

### Mesh-control dependencies

`src/fem/geometry/_gmsh/control_dependencies.py` owns committed entity keys,
the transform-unsafe subset, and the unknown-scope flag. Destructive operations
and transforms provide the normalized keys/closures computed by the model.

Raw access marks dependency scope unknown only when committed dependencies
exist. Unknown controlled mutation marks it unconditionally. Structured
extrusion snapshots and restores only the unknown flag around successful
dependency registration. Automatic-mesh blockers, size precedence, fields,
and generation state remain separate.

### Bound meshing capability

`src/fem/geometry/_gmsh/meshing_port.py` owns `_BoundMeshingPort`. It exposes
only the 12 complete user-level mesh transactions required by `Mesher`, keeps
its owner name-mangled, and exposes no native model/OCC handle, session,
registry, option, token, or arbitrary state operation.

`GeometryModel._acquire_meshing_port()` creates and records exactly one port,
activates the native model before binding, opens structured extrusion, and
enters `CONFIGURING_MESH`. Failed pre-binding state or activation checks do not
consume the capability. Every delegated transaction validates exact port
identity and mesh-control state. Lifecycle cleanup invalidates the stored port.

`fem.mesh.gmsh.Mesher` now has only the `_port` slot. Its constructor validates
the public geometry object and obtains the port through the single acquisition
seam. Every public operation routes through the port, and the redundant
cross-package `_complete()` layer is removed.

## Architecture and API Verification

Architecture tests enforce explicit import allowlists for every new module,
fresh-process Gmsh laziness, moved state/container ownership, one Mesher
acquisition seam, absence of private model-operation backchannels in
`src/fem/mesh`, and the restricted port surface.

An independent AST comparison against `45f0058` confirmed:

- the exact 16-name `fem.geometry` export snapshot is unchanged;
- the exact 10-name `fem.mesh.gmsh` export snapshot is unchanged;
- all public `GeometryModel`, `model()`, spec, and `Mesher` signatures match;
- geometry has no reverse dependency on mesh, IO, or FEM runtime layers;
- every new runtime module is acyclic and lazy with respect to external Gmsh.

`_gmsh.model` now has 5,015 physical lines, reduced from the plan-recorded
5,373 lines.

## Test Additions and Count Reconciliation

The complete count reconciles exactly:

~~~text
1,640 baseline + 80 newly collected items = 1,720 tests
~~~

There are 64 new test functions producing 80 items. Five existing functions,
representing eight parametrized items, were renamed with no count change. No
tests were moved or removed.

### `tests/test_gmsh_state.py` — 9 functions / 14 items

- `test_state_names_and_operation_sets_are_exact`
- `test_successful_mesh_lifecycle_follows_the_exact_state_graph`
- `test_failed_mesh_lifecycle_follows_the_exact_state_graph`
- `test_close_is_legal_from_every_nonclosed_state`
- `test_repeat_entry_preserves_the_existing_contextual_error`
- `test_illegal_transition_preserves_state_and_sorted_allowed_names`
- `test_allowed_operation_checks_match_each_lifecycle_phase`
- `test_mesh_terminal_states_reject_further_mesh_transitions`
- `test_error_factory_preserves_the_existing_context`

### `tests/test_gmsh_session.py` — 20 functions / 20 items

- `test_entry_preserves_backend_session_capture_validation_and_add_order`
- `test_invalid_name_is_detected_after_owned_session_and_model_inspection`
- `test_external_session_removes_only_facade_and_restores_prior_model`
- `test_nested_like_sessions_restore_lifo_and_only_outer_finalizes`
- `test_activate_reselects_only_when_needed_and_returns_facade`
- `test_activate_reports_inactive_session_and_externally_missing_model`
- `test_valid_empty_prior_model_name_is_restored`
- `test_missing_prior_model_is_skipped_without_disturbing_current_model`
- `test_model_name_collision_fails_before_add_or_external_finalization`
- `test_partially_successful_initialize_is_owned_and_finalized`
- `test_cleanup_inspection_failure_retains_ownership_for_retry`
- `test_cleanup_attempts_every_step_and_retains_each_failure_for_retry`
- `test_already_finalized_session_relinquishes_stale_cleanup_ownership`
- `test_numeric_options_read_all_originals_before_ordered_writes_and_restore`
- `test_numeric_option_partial_read_performs_no_writes_and_retains_snapshot`
- `test_numeric_option_partial_write_keeps_every_snapshot_for_restoration`
- `test_numeric_option_partial_restore_continues_and_retries_only_failures`
- `test_second_numeric_option_transaction_is_rejected_before_native_access`
- `test_nested_sessions_keep_independent_numeric_option_ledgers`
- `test_private_session_import_does_not_eagerly_import_external_gmsh`

### `tests/test_gmsh_reference_registry.py` — 20 functions / 20 items

- `test_entity_registry_reuses_live_token_and_refreshes_reused_tag`
- `test_entity_registry_preserves_validation_order_and_messages`
- `test_entity_registry_rejects_malformed_native_entity_keys`
- `test_loop_and_wire_with_the_same_tag_have_independent_identities`
- `test_invalid_loop_identity_clears_only_the_loop_namespace`
- `test_duplicate_loop_identity_clears_only_the_loop_namespace`
- `test_invalid_wire_identity_clears_only_the_wire_namespace`
- `test_duplicate_wire_identity_clears_only_the_wire_namespace`
- `test_topology_validation_covers_type_owner_token_and_duplicates`
- `test_topology_validation_detects_dependency_drift_and_shared_members`
- `test_entity_invalidation_also_invalidates_intersecting_topology`
- `test_topology_only_invalidation_keeps_entity_identity_live`
- `test_full_registry_clear_invalidates_all_typed_geometry_references`
- `test_control_ledger_rejects_removal_of_the_lowest_conflicting_key`
- `test_control_ledger_guards_only_transform_unsafe_subset`
- `test_control_ledger_marks_raw_scope_unknown_only_with_dependencies`
- `test_control_ledger_unknown_mutation_is_unconditional_but_empty_removal_is_safe`
- `test_control_ledger_structured_snapshot_restores_only_unknown_flag`
- `test_reference_clearing_does_not_clear_committed_control_guards`
- `test_private_registry_modules_do_not_import_external_gmsh`

### `tests/test_gmsh_meshing_port.py` — 3 functions / 14 items

- `test_bound_meshing_port_forwards_itself_as_the_only_authority`
- `test_bound_meshing_port_exposes_only_complete_mesh_transactions`
- `test_mesher_stores_one_port_and_routes_every_public_operation`

### Existing test modules — 12 functions / 12 items

`tests/test_gmsh_geometry.py` adds:

- `test_session_inspection_failure_does_not_mask_primary_and_is_retryable`
- `test_session_inspection_failure_without_primary_is_contextual_and_retryable`
- `test_cleanup_retains_later_failures_as_notes_and_retries_every_resource`
- `test_internal_facade_access_is_session_activation_gated`
- `test_meshing_port_activation_failure_does_not_consume_binding`

`tests/test_gmsh_profiles.py` adds:

- `test_fake_cleanup_failure_still_invalidates_entity_loop_and_wire_identity`

`tests/test_project_layout.py` adds:

- `test_geometry_stateful_modules_follow_explicit_dependency_boundaries`
- `test_phase_3_runtime_boundaries_do_not_eagerly_import_external_gmsh`
- `test_private_gmsh_model_delegates_extracted_stateful_ownership`
- `test_fem_mesh_uses_exactly_one_private_geometry_acquisition_seam`
- `test_bound_meshing_port_has_only_restricted_transaction_surface`
- `test_fem_mesh_public_api_snapshots_remain_exact`

The existing
`test_mesher_requires_live_geometry_and_failed_new_binding_is_retryable` also
gained a count-neutral assertion that an old bound port fails after context
close.

### Count-neutral renames — 5 functions / 8 items

- `test_transfinite_surface_rejects_cross_model_and_stale_corners_pre_mutation`
  became `test_transfinite_surface_rejects_cross_model_and_missing_native_corners`;
- `test_every_mesh_control_rejects_foreign_and_stale_targets_pre_mutation`
  became `test_every_mesh_control_rejects_foreign_and_missing_native_targets`
  and remains four-way parametrized;
- `test_transfinite_volume_rejects_foreign_and_stale_corner_tokens_pre_mutation`
  became `test_transfinite_volume_rejects_foreign_and_missing_native_corners`;
- `test_mesh_size_rejects_foreign_and_stale_points_before_native_mutation`
  became `test_mesh_size_rejects_foreign_and_missing_native_points_before_mutation`;
- `test_distance_sources_rejects_foreign_and_stale_entities_pre_mutation`
  became `test_distance_sources_rejects_foreign_and_missing_native_entities`.

These renames replace direct model-dictionary mutation with observable fake
native deletion. Registry unit tests now own direct stale-token validation.

## Verification Results

Static and structural verification:

- `ruff check src tests examples`: passed;
- project-environment `python -m compileall -q src tests examples`: passed with
  workspace-contained `PYTHONPYCACHEPREFIX`;
- Phase 3 architecture/state/session/registry/port tests: 89 passed;
- all four required structural searches: no matches;
- `git diff --check`: passed, with only Windows LF-to-CRLF checkout warnings.

Behavioral verification:

~~~text
Phase 3 focused runtime/architecture:       89 passed
Profiles/foundational/advanced/geometry/
local-refinement/meshing/IO:               898 passed in 4.54s
Complete repository suite:               1720 passed in 14.64s
~~~

The authoritative full-suite run ran outside the Windows sandbox because
sandbox-created pytest temporary directories have unusable ACLs.

Required real-Gmsh examples all exited with status 0 and reproduced the Phase
2 results:

~~~text
gmsh_geometry_irregular_plate.py
Solved 607 irregular-profile Tri3 elements and wrote results\gmsh_geometry_irregular_plate\gmsh_geometry_irregular_plate.vtk

gmsh_geometry_revolved_solid.py
Solved 145 nodes and 379 Tet4 elements; wrote D:\gu1mu\Code\projects\FEM-Python\results\gmsh_geometry_revolved_solid\gmsh_geometry_revolved_solid.vtk

gmsh_geometry_swept_solid.py
Imported swept solid: 60 nodes, 127 Tet4 elements

gmsh_geometry_filleted_box.py
Imported filleted box: 139 nodes, 358 Tet4 elements
~~~

## Working Tree and Phase 4 Boundary

Phase 3 is committed as one cohesive implementation. The user-owned
documentation deletion remains present, outside the commit, and untouched.

Phase 4 can now move mesh errors, fields, policies, controls, native generation,
and the generated-mesh runtime lease behind the established port. Structured
extrusion remains a geometry-owned OCC transaction reached through that port.
