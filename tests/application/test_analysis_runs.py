from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import pytest

from fem.application import (
    ChangeKind,
    ModelSession,
    NativePart,
    RunStatus,
    TokenStatus,
)
from fem.application.results import (
    ResultMaterializationSnapshot,
    ResultTopologyProjection,
    SolveResultBundle,
    build_solve_result_bundle,
)
from fem.core.model import AnalysisStep, FEMModel
from fem.core.result import ModelResult
from fem.geometry.recipes import BoxGeometry
from fem.solvers import static_linear
from tests.helpers.preflight_builders import passing_preflight_report
from tests.helpers.result_builders import make_solve_result_bundle
from tests.helpers.model_builders import (
    make_simple_truss_mesh,
    make_static_pull_truss_model,
)
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)


def _model() -> FEMModel:
    return FEMModel(
        mesh=make_simple_truss_mesh(),
        steps=[AnalysisStep("Step-A")],
    )


def _session() -> ModelSession:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry(
        (NativePart(),), BoxGeometry("Box", 1.0, 1.0, 1.0)
    )
    session.replace_model_definitions((), (), (), (AnalysisStep("Step-A"),))
    mesh = session.prepare_mesh_generation()
    session.accept_generated_model(mesh.token, _model())
    validation = session.prepare_validation("Step-A")
    session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    return session


def test_cached_prepared_system_isolated_from_exposed_solve_task_model() -> None:
    session = ModelSession()
    imported = session.prepare_import("prepared-cache.inp")
    session.accept_imported_model(
        imported.token,
        make_static_pull_truss_model(),
    )
    validation = session.prepare_validation("pull")
    prepared = static_linear.prepare(validation.model)
    session.accept_validation_with_prepared_system(
        validation.token,
        passing_preflight_report(validation.token),
        prepared,
    )

    first = session.prepare_solve("pull", "Job-1")
    original_x = first.model.mesh.nodes[1].x
    first.model.mesh.nodes[1].x = original_x + 10.0
    second = session.prepare_solve("pull", "Job-2")

    assert second.prepared_system is not None
    assert second.model.mesh.nodes[1].x == original_x
    result = static_linear.solve(
        second.model,
        second.step_name,
        _prepared_system=second.prepared_system,
    )
    assert result.U[second.model.mesh.global_dof(2, 0)] == pytest.approx(0.5)


def test_first_solve_installs_unexposed_cache_clone(monkeypatch) -> None:
    factor_calls = 0
    original_factor = static_linear.factorize_spd

    def factor(stiffness):
        nonlocal factor_calls
        factor_calls += 1
        return original_factor(stiffness)

    monkeypatch.setattr(static_linear, "factorize_spd", factor)
    session = ModelSession()
    imported = session.prepare_import("quick-cache.inp")
    session.accept_imported_model(
        imported.token,
        make_static_pull_truss_model(),
    )
    validation = session.prepare_validation("pull")
    session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    first = session.prepare_solve("pull", "Job-1")
    session.begin_run(first.token)
    run_prepared = static_linear.prepare(
        first.model,
        copy_model=False,
    )
    result = static_linear.solve(
        first.model,
        first.step_name,
        _prepared_system=run_prepared,
    )
    cache_candidate = run_prepared.clone()
    session.accept_run_succeeded_with_prepared_system(
        first.token,
        build_solve_result_bundle(first, result),
        run_prepared,
        cache_candidate=cache_candidate,
    )

    original_x = first.model.mesh.nodes[1].x
    first.model.mesh.nodes[1].x = original_x + 10.0
    second = session.prepare_solve("pull", "Job-2")

    assert second.prepared_system is not None
    assert second.model.mesh.nodes[1].x == original_x
    second_result = static_linear.solve(
        second.model,
        second.step_name,
        _prepared_system=second.prepared_system,
    )
    assert second_result.U[
        second.model.mesh.global_dof(2, 0)
    ] == pytest.approx(0.5)
    assert factor_calls == 1


def test_stale_validation_cannot_install_prepared_system() -> None:
    session = ModelSession()
    imported = session.prepare_import("old.inp")
    session.accept_imported_model(
        imported.token,
        make_static_pull_truss_model(),
    )
    validation = session.prepare_validation("pull")
    prepared = static_linear.prepare(validation.model)
    replacement = session.prepare_import("replacement.inp")
    session.accept_imported_model(
        replacement.token,
        make_static_pull_truss_model(),
    )

    delta = session.accept_validation_with_prepared_system(
        validation.token,
        passing_preflight_report(validation.token),
        prepared,
    )

    assert not delta.accepted
    assert delta.token_status is TokenStatus.STALE_SESSION
    assert session._current_prepared_system() is None


def test_stale_solve_completion_cannot_install_cache_candidate(
    monkeypatch,
) -> None:
    factor_calls = 0
    original_factor = static_linear.factorize_spd

    def factor(stiffness):
        nonlocal factor_calls
        factor_calls += 1
        return original_factor(stiffness)

    monkeypatch.setattr(static_linear, "factorize_spd", factor)
    session = ModelSession()
    imported = session.prepare_import("old-solve.inp")
    session.accept_imported_model(
        imported.token,
        make_static_pull_truss_model(),
    )
    validation = session.prepare_validation("pull")
    session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    solve = session.prepare_solve("pull", "Job-1")
    session.begin_run(solve.token)
    run_prepared = static_linear.prepare(
        solve.model,
        copy_model=False,
    )
    result = static_linear.solve(
        solve.model,
        solve.step_name,
        _prepared_system=run_prepared,
    )
    bundle = build_solve_result_bundle(solve, result)
    cache_candidate = run_prepared.clone()
    replacement = session.prepare_import("new-solve.inp")
    session.accept_imported_model(
        replacement.token,
        make_static_pull_truss_model(),
    )

    delta = session.accept_run_succeeded_with_prepared_system(
        solve.token,
        bundle,
        run_prepared,
        cache_candidate=cache_candidate,
    )

    assert not delta.accepted
    assert delta.token_status is TokenStatus.STALE_SESSION
    assert session._current_prepared_system() is None

    validation = session.prepare_validation("pull")
    session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    replacement_solve = session.prepare_solve("pull", "Job-2")
    replacement_prepared = static_linear.prepare(
        replacement_solve.model,
        copy_model=False,
    )
    static_linear.solve(
        replacement_solve.model,
        replacement_solve.step_name,
        _prepared_system=replacement_prepared,
    )
    assert factor_calls == 2


def test_pending_running_succeeded_lifecycle_and_provenance() -> None:
    session = _session()
    solve = session.prepare_solve("Step-A", "Job-1")
    pending = session.find_run(solve.run_id)
    assert pending.status is RunStatus.PENDING

    session.begin_run(solve.token)
    assert session.find_run(solve.run_id).status is RunStatus.RUNNING

    delta = session.accept_run_succeeded(
        solve.token,
        make_solve_result_bundle(solve, marker=42.0),
        timings={"solve": 0.25},
    )
    succeeded = session.find_run(solve.run_id)
    current = session.current_result()
    snapshot = session.snapshot()

    assert succeeded.status is RunStatus.SUCCEEDED
    assert succeeded.has_result
    assert succeeded.result_id == solve.result_id
    assert current.result_id == solve.result_id
    assert succeeded.timings == {"solve": 0.25}
    assert type(current.result) is ModelResult
    assert current.result.U[0] == 42.0
    assert current.result.U.flags.writeable is False
    assert current.result.reactions.flags.writeable is False
    assert current.output_report.source.result_id == solve.result_id
    assert current.materialization.source.result_id == solve.result_id
    assert current.materialization.generation == 0
    assert current.provenance.session_id == snapshot.session_id
    assert current.provenance.artifact_id == snapshot.artifact.artifact_id
    assert current.provenance.model_revision == snapshot.model_revision
    assert current.provenance.step_name == "Step-A"
    assert current.provenance.run_id == solve.run_id
    assert snapshot.displayed_result_run_id == solve.run_id
    assert delta.changed == {
        ChangeKind.RUNS,
        ChangeKind.RESULTS,
        ChangeKind.DISPLAYED_RESULT,
    }


def test_result_acceptance_deep_owns_all_public_result_views() -> None:
    session = _session()
    solve = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(solve.token)
    bundle = make_solve_result_bundle(solve, marker=4.0)

    session.accept_run_succeeded(solve.token, bundle)
    assert bundle._provider is None
    bundle.result.U[0] = 99.0
    bundle.result.model.name = "mutated-worker-model"

    current = session.current_result()
    assert current.result.U[0] == 4.0
    assert current.result.model.name != "mutated-worker-model"
    assert current.result.U.flags.writeable is False
    assert current.result.reactions.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        current.result.U[0] = 8.0
    current.result.model.name = "mutated-public-copy"
    assert (
        session.current_result().result.model.name
        != "mutated-public-copy"
    )

    displayed = session.snapshot().displayed_result
    projection = session.prepare_result_projection(solve.run_id)
    assert displayed.result.U.flags.writeable is False
    assert displayed.result.reactions.flags.writeable is False
    assert projection.record.result.U.flags.writeable is False
    assert projection.record.result.reactions.flags.writeable is False
    assert (
        displayed.materialization.topology._node_coordinates.flags.writeable
        is False
    )
    assert all(
        field_data._values.flags.writeable is False
        for field_data in displayed.materialization.fields
    )


def test_wrong_reserved_source_is_rejected_atomically_and_can_fail() -> None:
    session = _session()
    solve = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(solve.token)
    wrong_result_id = "result-forged"
    wrong_task = replace(
        solve,
        token=replace(solve.token, result_id=wrong_result_id),
        result_id=wrong_result_id,
    )
    bundle = make_solve_result_bundle(wrong_task, marker=1.0)
    revision = session.session_revision

    with pytest.raises(ValueError, match="reserved result"):
        session.accept_run_succeeded(solve.token, bundle)

    assert session.session_revision == revision
    assert session.find_run(solve.run_id).status is RunStatus.RUNNING
    assert session.current_result() is None
    assert session.validate_task_token(solve.token) is TokenStatus.CURRENT
    assert session.accept_run_failed(solve.token, "invalid bundle").accepted


def test_coherent_but_wrong_topology_is_rejected_before_mutation() -> None:
    session = _session()
    solve = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(solve.token)
    valid = make_solve_result_bundle(solve, marker=2.0)
    topology = valid.initial_materialization.topology
    coordinates = topology.node_coordinates.copy()
    coordinates[0, 0] += 0.5
    forged_topology = ResultTopologyProjection(
        source=valid.source,
        node_ids=topology.node_ids,
        node_coordinates=coordinates,
        nodal_displacements=topology.nodal_displacements,
        element_ids=topology.element_ids,
        element_types=topology.element_types,
        connectivity=topology.connectivity,
        element_region_keys=topology.element_region_keys,
    )
    forged_snapshot = ResultMaterializationSnapshot(
        source=valid.source,
        generation=0,
        topology=forged_topology,
        fields=valid.initial_materialization.fields,
    )
    forged = SolveResultBundle(
        source=valid.source,
        result=valid.result,
        execution_report=valid.execution_report,
        initial_materialization=forged_snapshot,
    )
    revision = session.session_revision

    with pytest.raises(ValueError, match="topology"):
        session.accept_run_succeeded(solve.token, forged)

    assert session.session_revision == revision
    assert session.find_run(solve.run_id).status is RunStatus.RUNNING
    assert session.current_result() is None
    assert session.validate_task_token(solve.token) is TokenStatus.CURRENT


def test_coherent_foreign_model_bundle_is_rejected_atomically() -> None:
    session = _session()
    solve = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(solve.token)
    foreign_result = make_continuum_nodal_semantics_result()
    foreign_step = AnalysisStep("Step-A")
    foreign_result.model.steps = [foreign_step]
    foreign_result.step = foreign_step
    forged_task = replace(solve, model=foreign_result.model)
    forged = build_solve_result_bundle(forged_task, foreign_result)
    revision = session.session_revision

    with pytest.raises(ValueError, match="exact model and step"):
        session.accept_run_succeeded(solve.token, forged)

    assert session.session_revision == revision
    assert session.find_run(solve.run_id).status is RunStatus.RUNNING
    assert session.current_result() is None
    assert session.validate_task_token(solve.token) is TokenStatus.CURRENT


def test_same_topology_with_forged_material_is_rejected_atomically() -> None:
    session = _session()
    solve = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(solve.token)
    foreign_model = deepcopy(solve.model)
    foreign_model.mesh.elements[0].props["E"] = 999.0
    forged_task = replace(solve, model=foreign_model)
    forged = make_solve_result_bundle(forged_task, marker=3.0)
    revision = session.session_revision

    with pytest.raises(ValueError, match="exact model and step"):
        session.accept_run_succeeded(solve.token, forged)

    assert session.session_revision == revision
    assert session.find_run(solve.run_id).status is RunStatus.RUNNING
    assert session.current_result() is None
    assert session.validate_task_token(solve.token) is TokenStatus.CURRENT


def test_failed_new_run_preserves_previous_successful_display() -> None:
    session = _session()
    first = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(first.token)
    session.accept_run_succeeded(
        first.token,
        make_solve_result_bundle(first, marker=1.0),
    )

    second = session.prepare_solve("Step-A", "Job-2")
    session.begin_run(second.token)
    assert session.current_result().result.U[0] == 1.0
    session.accept_run_failed(second.token, "solver failed")

    assert session.find_run(second.run_id).status is RunStatus.FAILED
    assert session.current_result().result.U[0] == 1.0
    assert session.snapshot().displayed_result_run_id == first.run_id


def test_cancelled_new_run_preserves_previous_successful_display() -> None:
    session = _session()
    first = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(first.token)
    session.accept_run_succeeded(
        first.token,
        make_solve_result_bundle(first, marker=1.0),
    )

    second = session.prepare_solve("Step-A", "Job-2")
    session.begin_run(second.token)
    session.request_cancel(second.run_id)
    session.accept_run_cancelled(second.token)

    cancelled = session.find_run(second.run_id)
    assert cancelled.status is RunStatus.CANCELLED
    assert cancelled.cancellation_requested
    assert session.current_result().result.U[0] == 1.0
    assert session.snapshot().displayed_result_run_id == first.run_id


def test_model_revision_change_clears_run_history_and_display() -> None:
    session = _session()
    solve = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(solve.token)
    session.accept_run_succeeded(
        solve.token,
        make_solve_result_bundle(solve, marker=1.0),
    )

    session.replace_model_definitions((), (), (), (AnalysisStep("Step-A"),))
    snapshot = session.snapshot()

    assert snapshot.runs == ()
    assert snapshot.displayed_result_run_id is None
    assert snapshot.displayed_result is None
    assert session.current_result() is None
