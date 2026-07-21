# `fem.mesh.gmsh` Runtime Ownership Phase 4 Implementation Handoff

## Outcome

Phase 4 is implemented from the committed Phase 3 baseline:

~~~text
89dff5e refactor(gmsh): add stateful runtime and meshing port
~~~

The work remains uncommitted at the collaborator's request. The branch is
`develop`, and no implementation commit was created.

`fem.mesh.gmsh` is now a package whose exact public facade is:

~~~text
AutoMeshSpec
GmshMeshRef
MeshCellShapeError
MeshControlConflictError
MeshFieldOwnershipError
MeshFieldRef
MeshSpec
Mesher
StaleGmshMeshError
StaleMeshFieldError
~~~

Mesh errors, specifications, references, configuration, field identity,
policies, controls, generation, validation, and generated-mesh lease authority
now have canonical mesh owners. `GeometryModel` retains OCC geometry,
topology/reference state, structured extrusion, session ownership, and one
restricted geometry-issued meshing capability.

The pre-existing user-owned deletion of
`docs/2026-07-20-gmsh-structured-feature-results-and-foundational-operations-implementation-handoff.md`
was not restored, modified, or staged.

## Production Ownership Migration

The former single file `src/fem/mesh/gmsh.py` was removed and replaced by:

- `src/fem/mesh/gmsh/__init__.py`: exact ten-name public facade;
- `src/fem/mesh/gmsh/errors.py`: the five mesh error classes;
- `src/fem/mesh/gmsh/types.py`: `MeshFieldRef`, `GmshMeshRef`, and the sealed
  generated-mesh lease;
- `src/fem/mesh/gmsh/specs.py`: `MeshSpec` and `AutoMeshSpec`;
- `src/fem/mesh/gmsh/mesher.py`: the thin public `Mesher` facade;
- `src/fem/mesh/gmsh/_protocols.py`: structural geometry-port and native-borrow
  contracts;
- `src/fem/mesh/gmsh/_validation.py`: mesh-only primitive validation;
- `src/fem/mesh/gmsh/_configuration.py`: size mode, blocker, and one-attempt
  state;
- `src/fem/mesh/gmsh/_field_registry.py`: runtime-local field identity and
  rollback;
- `src/fem/mesh/gmsh/_policies.py`: explicit and automatic generation policy;
- `src/fem/mesh/gmsh/_runtime.py`: control, field, option, generation, strict
  cell-shape, and lease-preparation orchestration.

`src/fem/mesh/__init__.py` now resolves the package directly while preserving
the exact `fem.mesh` public API.

The geometry host changed as follows:

- `meshing_port.py` exposes only immutable metadata, validation and entity
  services, scoped native callbacks, dependency registration, option
  transactions, structured extrusion, generation state transitions, and
  native-borrow preparation/completion;
- `session.py` owns independent revocation epochs, exact facade-model and
  owned-session-baseline incarnation markers, and a dormant native-model
  borrow capability; context exit revokes all borrows before native cleanup;
- `state.py` provides the prevalidated no-fail `MESHED` transition;
- `model.py` no longer defines or stores mesh contracts, fields, policies,
  blockers, size mode, attempt state, generation identity, generated-mesh
  borrowing, high-level `_mesher_*` methods, or direct `.model.mesh` calls;
- geometry reports only neutral topology provenance and retains the OCC body
  of structured extrusion;
- `fem.io.gmsh.read()` validates its public inputs and calls the canonical
  `GmshMeshRef._borrow_model` implementation exactly once, including for a
  subclass instance.

`Mesher` validates a public `GeometryModel`, crosses the geometry boundary
through one `_acquire_meshing_port()` call, constructs one mesh runtime, and
stores only that runtime. The runtime retains the structural port and
mesh-owned state; it never retains a raw native model outside a scoped
callback.

## Preserved Behavior and Authority Hardening

The characterized Phase 3 validation and mutation order remains in place for
ordinary controls, fields, structured extrusion, explicit generation, and
automatic generation. In particular:

- preflight failures leave structured extrusion open, while the first native
  mesh mutation closes it;
- native control failures leave the generation attempt available;
- missing top-dimensional topology remains retryable;
- failures after the generation commit consume the sole attempt, enter
  `MESH_FAILED`, and retain restoration diagnostics;
- explicit generation always synchronizes OCC after native generation, even
  when strict cell-shape validation is disabled;
- Distance, Threshold, and Min fields use runtime-local owner and fresh
  identity tokens, including after native tag reuse;
- post-allocation field failure rolls back the native field and records a
  rollback failure as a note on the primary exception;
- deterministic option snapshots are restored after success and failure, with
  pending failures retained for cleanup retry;
- structured extrusion remains geometry-owned internally and mesh-owned at the
  public routing and automatic-blocker level;
- native activation occurs before an ordinary control closes the structured
  subphase, so activation failure remains retryable;
- successful generation prepares the native capability, sealed mesh lease,
  bearer token, and public reference, then completes every fallible
  authority/state/incarnation validation before the assignment-only borrow,
  `MESHED`, prepared-slot, and lease activation tail;
- generation completion consumes the port's prepared native-borrow capability,
  and the owner revalidates port identity plus `CONFIGURING_MESH` before
  activation, so missing, replayed, failed-state, or closed-state completion
  cannot touch owner state or resurrect a terminal lifecycle;
- generated references retain no direct or indirect `GeometryModel`, entity
  owner token, or geometry generation token;
- facade activation, generated borrowing, completion, model removal, and
  owned-session finalization all verify exact persistent incarnations, so a
  same-name facade or initialization-baseline replacement is never borrowed,
  removed, or finalized as the original model;
- owned-session finalization compares the full model-name multiset and verifies
  every captured baseline incarnation; additional models, duplicate empty-name
  models, replaced baseline models, and identity-inspection failures retain the
  process-global session rather than deleting foreign state;
- repeated reads and bearer-preserving copies remain valid, while altered
  metadata, malformed/look-alike leases, revoked sessions, closed contexts,
  missing native models, and nested-model lifetime violations become stale
  with preserved causes.

The independent and follow-up reviews drove regressions for unconditional
post-generation synchronization, activation-before-control-commit ordering,
one-shot consumption of prepared completion authority, owner-side rejection of
terminal-state completion, generated-handle revocation across every cleanup
failure path, exact native model incarnation, and the assignment-only success
commit tail. Real-Gmsh regressions reproduce the reported 12-node original and
98-node same-name replacement, and separately cover normal default-model
finalization, default-model replacement, and duplicate empty-name models.

## Test Ownership Migration

Seventy-three existing test definitions, expanding to 220 cases, moved
unchanged from `tests/test_gmsh_geometry.py` to
`tests/test_gmsh_mesh_runtime.py`. OCC operations, the geometry dependency
ledger, topology provenance, structured extrusion, and real-Gmsh integration
remain in the geometry module. The split reconciles as follows:

~~~text
before split:                              609 cases
retained test_gmsh_geometry.py:            389 cases
moved runtime cases:                       220 cases
new runtime ordering regressions:            2 cases
new cleanup lifetime regressions:            6 cases
follow-up incarnation/no-fail regressions:  11 cases
final geometry + runtime split:            628 cases
~~~

The moved test definitions are:

- `test_transfinite_curve_and_recombine_forward_typed_targets_while_building`
- `test_invalid_curve_and_recombine_controls_fail_before_backend_mutation`
- `test_mesh_controls_reject_cross_model_and_stale_targets_before_mutation`
- `test_native_mesh_control_failure_preserves_state_and_mesh_attempt`
- `test_transfinite_surface_forwards_automatic_and_explicit_corners_in_order`
- `test_invalid_surface_corner_shape_fails_before_native_control`
- `test_transfinite_surface_rejects_nonboundary_corner_before_native_control`
- `test_transfinite_surface_rejects_cross_model_and_missing_native_corners`
- `test_transfinite_volume_forwards_automatic_six_and_eight_corners_in_order`
- `test_invalid_volume_corner_shape_fails_before_native_control`
- `test_transfinite_volume_requires_3d_facade_and_recursive_boundary_corners`
- `test_mesh_controls_reject_new_and_closed_states_contextually`
- `test_mesh_controls_reject_meshed_and_mesh_failed_states_contextually`
- `test_every_mesh_control_rejects_invalid_target_before_backend_mutation`
- `test_every_mesh_control_rejects_foreign_and_missing_native_targets`
- `test_transfinite_volume_rejects_foreign_and_missing_native_corners`
- `test_native_control_failures_preserve_exception_state_and_generation_attempt`
- `test_mesh_controls_reactivate_owned_model_and_stay_nested_model_local`
- `test_mesh_size_forwards_ordered_batches_while_building`
- `test_invalid_mesh_size_inputs_fail_before_synchronization_or_mutation`
- `test_mesh_size_materializes_generators_before_native_mutation`
- `test_mesh_size_rejects_foreign_and_missing_native_points_before_mutation`
- `test_distance_field_forwards_dimension_specific_lists_in_source_order`
- `test_invalid_distance_field_inputs_fail_before_native_mutation`
- `test_distance_field_materializes_every_source_before_native_mutation`
- `test_threshold_min_and_background_build_an_ordered_inert_field_graph`
- `test_invalid_threshold_inputs_fail_before_native_mutation`
- `test_invalid_min_inputs_fail_before_native_mutation`
- `test_distance_sources_reject_foreign_and_missing_native_entities`
- `test_mesh_fields_are_owned_live_model_local_and_use_fresh_tokens_on_reuse`
- `test_distance_field_rolls_back_after_each_configuration_failure`
- `test_threshold_field_rolls_back_after_each_configuration_failure`
- `test_min_field_rolls_back_after_each_configuration_failure`
- `test_field_constructor_rejects_invalid_or_inactive_allocated_tags`
- `test_field_rollback_failure_preserves_primary_error_note_and_mesh_attempt`
- `test_background_selection_failure_is_retryable_and_keeps_fields_inert`
- `test_background_rejects_non_size_fields_and_repeated_selection_pre_mutation`
- `test_typed_size_mode_conflicts_are_retryable_before_native_generation`
- `test_typed_size_modes_request_and_restore_all_deterministic_options`
- `test_typed_size_generation_failures_restore_every_external_option`
- `test_nested_typed_size_modes_keep_fields_model_local_and_restore_options`
- `test_invalid_mesh_arguments_fail_before_mesh_or_option_mutation`
- `test_generation_surface_is_owned_only_by_mesher_specs`
- `test_missing_top_dimensional_entity_leaves_mesher_retryable_but_geometry_sealed`
- `test_generate_mesh_assigns_size_isolates_options_and_returns_live_handle`
- `test_generated_handle_reactivates_owner_across_nested_contexts`
- `test_generated_handle_rejects_forged_generation_identity`
- `test_generated_handle_rejects_malformed_lease_before_dispatch`
- `test_generated_handle_detects_missing_native_model`
- `test_1d_mesh_contract_is_validated_before_mesh_or_option_mutation`
- `test_1d_generate_mesh_returns_native_handle_and_restores_options`
- `test_1d_missing_curve_preflight_keeps_mesher_retryable_and_seals_geometry`
- `test_failed_generation_restores_options_and_disallows_retry`
- `test_size_assignment_without_points_consumes_mesh_attempt`
- `test_option_set_failure_restores_snapshot_and_marks_mesh_failed`
- `test_auto_mesh_legal_shape_matrix_uses_exact_fixed_policy`
- `test_auto_mesh_rejects_invalid_shape_matrix_before_native_mutation`
- `test_auto_mesh_rejects_invalid_levels_before_native_mutation`
- `test_auto_mesh_levels_set_dimension_aware_absolute_size_factor`
- `test_auto_mesh_preflight_validation_is_retryable`
- `test_auto_mesh_strict_pure_families_return_native_handles`
- `test_auto_mesh_tri_quad_accepts_each_permitted_family_union`
- `test_auto_mesh_strict_validation_rejects_empty_and_malformed_output`
- `test_auto_mesh_shape_error_reports_aggregated_named_actual_cells`
- `test_auto_mesh_shape_diagnostic_falls_back_to_unknown_numeric_type`
- `test_auto_mesh_get_elements_failure_preserves_native_error_and_attempt`
- `test_auto_mesh_strict_validation_ignores_lower_dimensional_blocks`
- `test_auto_mesh_typed_size_modes_compose_factor_once_and_restore_options`
- `test_auto_mesh_failures_restore_every_external_option_and_consume_attempt`
- `test_auto_mesh_successful_generation_restoration_failure_is_retried_on_exit`
- `test_auto_mesh_shape_failure_preserves_error_when_restoration_also_fails`
- `test_auto_mesh_missing_top_entity_keeps_mesher_retryable_and_seals_geometry`
- `test_nested_auto_mesh_models_isolate_current_model_policy_and_level`

No test was removed. These count-neutral renames describe the new ownership:

- `test_generated_handle_rejects_malformed_owner_before_dispatch` became
  `test_generated_handle_rejects_malformed_lease_before_dispatch` and moved to
  the runtime module;
- `test_canonical_exports_share_owner_coupled_class_identity` became
  `test_canonical_exports_have_mesh_owned_class_identity`;
- `test_bound_meshing_port_forwards_itself_as_the_only_authority` became
  `test_bound_meshing_port_forwards_itself_as_the_only_geometry_authority`;
- `test_bound_meshing_port_exposes_only_complete_mesh_transactions` became
  `test_bound_meshing_port_exposes_only_geometry_host_capabilities`;
- `test_mesher_stores_one_port_and_routes_every_public_operation` became
  `test_mesher_stores_one_runtime_and_routes_every_public_operation`;
- `test_phase_3_runtime_boundaries_do_not_eagerly_import_external_gmsh` became
  `test_phase_4_runtime_boundaries_do_not_eagerly_import_external_gmsh`;
- `test_private_gmsh_model_does_not_redefine_extracted_stateless_names` became
  `test_private_gmsh_model_does_not_redefine_moved_contracts_or_policies`;
- `test_gmsh_meshing_depends_only_on_geometry_and_backend_layers` became
  `test_gmsh_meshing_recursively_depends_only_on_public_geometry_contracts`.

## Added Contract Tests and Exact Count

The repository baseline was 1,720 passing tests. The initial Phase 4
implementation added 52 cases:

- 28 specification, field-registry, and generated-lease cases:
  - `test_mesh_package_exports_exact_canonical_contracts`;
  - `test_mesh_spec_fields_defaults_and_signatures_are_unchanged`;
  - `test_mesh_specs_remain_frozen_slotted_and_normalize_size`;
  - `test_all_mesh_errors_preserve_geometry_error_hierarchy`;
  - `test_mesh_spec_size_validation_text_is_preserved` (six cases);
  - `test_auto_mesh_spec_vocabulary_validation_text_is_preserved`;
  - `test_field_owner_identity_is_runtime_local_and_tag_reuse_is_fresh`;
  - `test_native_field_disappearance_invalidates_only_matching_identity`;
  - `test_field_inputs_reject_duplicates_before_native_liveness`;
  - `test_post_allocation_failure_rolls_back_and_preserves_primary_exception`;
  - `test_rollback_failure_is_a_note_on_the_primary_exception`;
  - `test_active_native_field_tags_reject_malformed_values` (four cases);
  - `test_generated_reference_is_frozen_slotted_and_borrow_is_nonconsuming`;
  - `test_bearer_preserving_copies_remain_usable`;
  - `test_altered_bearer_metadata_is_stale` (three cases);
  - `test_lookalike_lease_is_rejected_before_dispatch`;
  - `test_nominal_lease_is_sealed_against_subclasses_and_direct_construction`;
  - `test_native_borrow_failure_is_translated_with_preserved_cause`;
- eight port cases:
  - one additional forwarding-matrix case for the final capability surface;
  - `test_bound_meshing_port_snapshots_read_only_model_metadata`;
  - `test_bound_meshing_port_accepts_only_backend_free_callback_results`
    (two cases);
  - `test_bound_meshing_port_completes_with_its_prepared_native_borrow`;
  - `test_bound_meshing_port_rejects_missing_or_replayed_generation_completion`;
  - `test_owner_rejects_prepared_generation_completion_after_failure`;
  - `test_owner_rejects_prepared_generation_completion_after_close`;
- four architecture cases:
  - `test_geometry_model_contains_no_mesh_owned_runtime_or_native_mesh_calls`;
  - `test_generated_mesh_reference_has_no_concrete_geometry_backchannel`;
  - `test_gmsh_mesh_facade_routes_public_names_to_canonical_modules`;
  - `test_gmsh_mesh_backend_is_a_package_and_old_single_file_is_absent`;
- three session cases:
  - `test_native_borrow_is_dormant_then_reactivates_repeatedly`;
  - `test_native_borrow_revocation_is_idempotent_and_native_call_free`;
  - `test_nested_session_borrow_epochs_are_isolated`;
- one IO case:
  - `test_read_dispatches_canonical_borrow_once_for_a_reference_subclass`;
- two review-driven runtime cases:
  - `test_explicit_generation_synchronizes_after_non_strict_native_generate`;
  - `test_native_control_activation_failure_leaves_structured_subphase_open`;
- six generated-handle cleanup lifetime cases:
  - `test_generated_handle_is_stale_when_cleanup_step_fails` for session
    inspection, facade-model removal, prior-model restoration, and
    finalization failures;
  - `test_generated_handle_stays_stale_across_combined_cleanup_failures_and_retry`;
  - `test_inner_cleanup_failure_revokes_only_inner_generated_handle`.

The follow-up ownership and completion audits add another 25 cases:

- eight exact-incarnation cases:
  - `test_entry_marker_failure_distinguishes_verified_partial_installation`
    (two cases);
  - `test_cleanup_attribute_read_failure_retains_exact_model_for_retry`;
  - `test_same_name_replacement_is_neither_activated_borrowed_nor_removed`;
  - `test_outer_session_owner_does_not_finalize_inner_same_name_replacement`;
  - `test_owned_session_finalization_revalidates_sole_remaining_owned_model`;
  - `test_generated_handle_rejects_same_name_replacement_and_preserves_it`;
  - `test_real_gmsh_same_name_replacement_cannot_satisfy_old_reference`;
- eleven no-fail completion cases:
  - `test_owner_generation_completion_tail_is_prevalidated_assignment_only`;
  - `test_owner_rejects_prepared_generation_completion_after_meshed`;
  - `test_success_completion_uses_only_prevalidated_assignment_steps_in_order`;
  - `test_generation_completion_preparation_failure_is_terminal_and_dormant`
    for native-borrow, bearer-token, lease, reference, and completion-guard
    failures (five cases);
  - `test_completion_failure_preserves_primary_when_terminalization_also_fails`
    for `MESH_FAILED` and `CLOSED` (two cases);
  - `test_same_name_replacement_during_completion_preparation_fails_dormant`;
- six owned-session baseline cases:
  - `test_owned_session_default_model_does_not_block_finalization`;
  - `test_missing_model_clears_prior_identity_inspection_failure`;
  - `test_owned_session_baseline_same_name_replacement_blocks_finalization`;
  - `test_real_owned_session_close_finalizes_native_default_model`;
  - `test_real_owned_session_preserves_added_duplicate_empty_model`;
  - `test_real_owned_session_preserves_replaced_default_model`.

Therefore:

~~~text
Phase 3 baseline                 1,720
count-neutral moves/renames          0
initial Phase 4 cases               52
follow-up audit cases               25
final repository total          1,797
~~~

## Final Verification

The final implementation used:

- Python 3.13.11;
- Gmsh 4.15.2;
- pytest 9.1.1;
- global Ruff 0.14.10 because Ruff remains absent from the project virtual
  environment.

The final working tree passed:

- focused Phase 4 ownership/runtime suite: 351 passed in 4.15 seconds;
- retained geometry/Gmsh compatibility suite: 682 passed in 4.13 seconds;
- complete repository suite: 1,797 passed in 16.09 seconds;
- Python compilation of `src`, `tests`, and `examples` with a workspace
  `PYTHONPYCACHEPREFIX`;
- Ruff checks for `src`, `tests`, and `examples`;
- architecture tests for recursive dependency direction, exact exports,
  package identity, the sole acquisition seam, runtime storage, the exact port
  allowlist, geometry ownership removal, canonical IO dispatch, and lazy
  external-Gmsh loading;
- all five structural searches with no matches;
- `git diff --check`.

The sandbox-created Windows pytest base directory again acquired unusable ACLs
during the broad run. The final complete suite was therefore run outside that
sandbox boundary with a workspace-contained base directory; it completed
cleanly.

The four required real-Gmsh examples passed without source changes:

- irregular plate: 607 irregular-profile Tri3 elements;
- revolved solid: 145 nodes and 379 Tet4 elements;
- swept solid: 60 nodes and 127 Tet4 elements;
- filleted box: 139 nodes and 358 Tet4 elements.

No files were staged or committed. The worktree is ready for collaborator
review while preserving the unrelated documentation deletion exactly as it
was found.
