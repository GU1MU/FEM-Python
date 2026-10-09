from __future__ import annotations

from dataclasses import replace

import pytest

from fem_agent.authoring import (
    AuthoringContext,
    CapabilitySummary,
    LocalModelBinding,
    MeshSummary,
    PartSummary,
    ProposalState,
    RequirementStatus,
)
from fem_agent.authoring_runtime import AuthoringWorkflowController
from fem_agent.authoring_values import AuthoringToolOutcome, AuthoringWorkflowStage
from fem_agent.tools.registry import ToolExecutionContext


_CONTEXT_TOOLS = frozenset({
    "request_project_save",
    "read_deletable_objects",
    "prepare_delete_proposal",
    "read_editable_model_objects",
    "edit_model_object",
    "read_geometry_edit_context",
    "prepare_geometry_edit",
    "read_geometry_feature_catalog",
    "read_profile_transform_context",
    "prepare_profile_extrusion",
    "prepare_profile_revolution",
    "prepare_profile_path_sweep",
})
_RESULT_TOOLS = frozenset({
    "read_analysis_run_catalog",
    "read_accepted_result_catalog",
    "query_accepted_result",
    "compare_accepted_results",
    "read_result_display_context",
    "export_accepted_result_csv",
})
_READ_ONLY_STAGES = (
    AuthoringWorkflowStage.GEOMETRY_PENDING,
    AuthoringWorkflowStage.MESH_PENDING,
    AuthoringWorkflowStage.PREFLIGHT_PENDING,
    AuthoringWorkflowStage.SOLVE_PENDING,
    AuthoringWorkflowStage.PROJECT_SAVE_PENDING,
    AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING,
    AuthoringWorkflowStage.STALE,
    AuthoringWorkflowStage.CANCELLED,
)


def _context(dimension: int = 2) -> AuthoringContext:
    return AuthoringContext(
        binding=LocalModelBinding("document:publication", "native-publication", 3, "native", True),
        model_name="Model-publication",
        active_part_id="part-publication",
        parts=(PartSummary("part-publication", "Part-publication", "planar_sketch", dimension, False),),
        mesh=MeshSummary(True, True, 20, 10),
        capabilities=tuple(
            CapabilitySummary(operation, True)
            for operation in (
                "request_project_save", "delete_model_objects", "edit_model_objects",
                "edit_native_geometry", "read_geometry_feature_catalog",
            )
        ),
    )


def _handler(_arguments, _controller) -> AuthoringToolOutcome:
    return AuthoringToolOutcome("Read locally.", {})


def _published_names(controller: AuthoringWorkflowController) -> set[str]:
    return {tool.name for tool in controller.definitions}


def test_publication_reads_capabilities_once_without_changing_observed_state() -> None:
    observed = replace(
        _context(1), mesh=MeshSummary(False, False), run_count=1, result_count=1,
    )
    live = replace(_context(3), run_count=2, result_count=2)
    reads = []
    export_reads = []

    def context_reader():
        reads.append("context")
        return live

    def export_reader():
        export_reads.append("export")
        return True

    controller = AuthoringWorkflowController(
        context_reader,
        {name: _handler for name in _CONTEXT_TOOLS | _RESULT_TOOLS | {"apply_model_definition"}},
        workspace_export_available=export_reader,
    )
    controller.observe_binding(observed)
    snapshot = controller.turn_snapshot
    entries = controller.ledger.entries
    identity = controller._binding_identity

    tools = controller.definitions
    names = {item.name for item in tools}

    assert reads == ["context"]
    assert export_reads == ["export"]
    assert _CONTEXT_TOOLS <= names
    assert "apply_model_definition" not in names
    assert "compare_accepted_results" not in names
    assert "export_accepted_result_csv" in names
    requirements = next(item for item in tools if item.name == "set_authoring_requirements")
    properties = requirements.parameters["properties"]["requirements"]["properties"]
    assert properties["mesh_cell_shape"]["enum"] == ["line"]
    assert "line_element_type" in properties
    assert controller._observed_context is observed
    assert controller._binding_identity == identity
    assert controller.ledger.entries == entries
    assert controller.turn_snapshot is snapshot


@pytest.mark.parametrize("reader_result", ["missing", "mapping", "exception"])
def test_publication_failure_never_reuses_previous_live_or_observed_capabilities(reader_result: str) -> None:
    calls = []

    def reader():
        calls.append("read")
        if len(calls) == 1:
            return _context()
        if reader_result == "exception":
            raise RuntimeError("capability context unavailable")
        return {} if reader_result == "mapping" else None

    controller = AuthoringWorkflowController(reader, {name: _handler for name in _CONTEXT_TOOLS})
    observed = _context()
    controller.observe_binding(observed)
    assert _CONTEXT_TOOLS <= _published_names(controller)

    assert not _CONTEXT_TOOLS.intersection(_published_names(controller))
    assert calls == ["read", "read"]
    assert controller._observed_context is observed


@pytest.mark.parametrize("stage", _READ_ONLY_STAGES)
def test_read_only_publication_does_not_read_live_context_or_export(stage) -> None:
    def unexpected_read():
        pytest.fail("read-only publication must not read live eligibility")

    controller = AuthoringWorkflowController(
        unexpected_read,
        {name: _handler for name in _CONTEXT_TOOLS | _RESULT_TOOLS | {"read_workspace_documents"}},
        workspace_export_available=unexpected_read,
    )
    controller.observe_binding(replace(_context(), run_count=2, result_count=2))
    controller._stage = stage

    assert _published_names(controller) == {"read_authoring_context", "read_workspace_documents"}


def test_workspace_only_publication_samples_inventory_and_export_without_live_context() -> None:
    calls = []

    def inventory_reader():
        calls.append("inventory")
        return 2, 2

    def export_reader():
        calls.append("export")
        return True

    def unexpected_context_read():
        pytest.fail("global workspace reads must not read authoring context")

    controller = AuthoringWorkflowController(
        unexpected_context_read,
        {name: _handler for name in _CONTEXT_TOOLS | _RESULT_TOOLS | {"read_workspace_documents"}},
        workspace_result_inventory=inventory_reader,
        workspace_export_available=export_reader,
    )

    assert _published_names(controller) == _RESULT_TOOLS | {"read_workspace_documents"}
    assert calls == ["inventory", "export"]
    assert controller._observed_context is None


@pytest.mark.parametrize("raises", [False, True])
def test_export_failure_is_false_and_is_sampled_once_per_publication(raises: bool) -> None:
    reads = []

    def export_reader():
        reads.append("export")
        if raises:
            raise RuntimeError("no selected result")
        return False

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {name: _handler for name in _RESULT_TOOLS},
        workspace_export_available=export_reader,
    )
    controller.observe_binding(replace(_context(), run_count=2, result_count=2))

    assert "export_accepted_result_csv" not in _published_names(controller)
    assert reads == ["export"]


@pytest.mark.parametrize("has_observed_context", [False, True])
def test_export_callback_is_skipped_when_handler_is_absent_despite_results(has_observed_context: bool) -> None:
    def unexpected_export_read():
        pytest.fail("unregistered export tools do not need export eligibility")

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {name: _handler for name in _RESULT_TOOLS - {"export_accepted_result_csv"}},
        workspace_result_inventory=lambda: (2, 2),
        workspace_export_available=unexpected_export_read,
    )
    if has_observed_context:
        controller.observe_binding(_context())

    names = _published_names(controller)
    assert "export_accepted_result_csv" not in names
    assert "query_accepted_result" in names


def test_ready_publication_skips_live_read_for_incomplete_registration_groups() -> None:
    def unexpected_read():
        pytest.fail("incomplete handler groups cannot publish context tools")

    controller = AuthoringWorkflowController(
        unexpected_read,
        {"read_geometry_edit_context": _handler, "read_deletable_objects": _handler},
    )
    controller.observe_binding(_context())

    assert _published_names(controller) == {"read_authoring_context", "set_authoring_requirements"}


def test_unobserved_ready_publication_does_not_promote_live_context_to_observed_state() -> None:
    controller = AuthoringWorkflowController(
        lambda: _context(1),
        {"request_project_save": _handler},
    )

    assert "request_project_save" in _published_names(controller)
    assert controller._observed_context is None
    assert controller._binding_identity is None
    assert controller.stage is AuthoringWorkflowStage.REQUIREMENTS
    assert controller.ledger.entries == ()


def test_invalidated_requirement_is_not_used_for_mesh_publication() -> None:
    controller = AuthoringWorkflowController(
        lambda: _context(), {"prepare_mesh_proposal": _handler},
    )
    controller.observe_binding(replace(_context(), mesh=MeshSummary(False, False)))
    for key, field_type, value in (
        ("mesh_cell_shape", "string", "triangle"),
        ("mesh_order", "integer", 1),
        ("mesh_global_size", "number", 5.0),
    ):
        controller.ledger.record(
            key, field_type=field_type, stage="mesh", value=value,
            source_turn_id="turn-publication", status=RequirementStatus.PROPOSED,
        )
    assert "prepare_mesh_proposal" in _published_names(controller)

    controller.ledger.invalidate("mesh_global_size", "active geometry changed")

    assert "prepare_mesh_proposal" not in _published_names(controller)


@pytest.mark.parametrize("terminal_stage", [AuthoringWorkflowStage.STALE, AuthoringWorkflowStage.CANCELLED])
def test_read_dispatch_resynchronizes_binding_and_scoped_geometry_schema(terminal_stage) -> None:
    current = [_context()]

    def geometry_context(_arguments, _controller):
        return AuthoringToolOutcome("Read geometry context.", {
            "part_id": current[0].active_part_id,
            "kind": "planar_sketch",
            "supported_edits": ["add_circle"],
        })

    controller = AuthoringWorkflowController(
        lambda: current[0],
        {"read_geometry_edit_context": geometry_context, "prepare_geometry_edit": _handler},
    )
    controller.observe_binding(current[0])
    execution = ToolExecutionContext("agent-publication", 3, "key-geometry-context")
    assert controller.dispatch("read_geometry_edit_context", {}, execution).ok
    previous_scope = controller._geometry_edit_schema_scope
    current[0] = replace(
        current[0],
        binding=LocalModelBinding("document:rebound", "native-rebound", 0, "native", True),
        active_part_id="part-rebound",
        parts=(PartSummary("part-rebound", "Part-rebound", "planar_sketch", 2, False),),
    )
    if terminal_stage is AuthoringWorkflowStage.STALE:
        assert not controller.observe_binding(current[0])
    else:
        controller._stage = terminal_stage
    assert "read_geometry_edit_context" not in _published_names(controller)

    result = controller.dispatch("read_geometry_edit_context", {}, execution)

    assert result.ok
    assert controller._observed_context is current[0]
    assert controller._binding_identity == ("document:rebound", "native-rebound", 0)
    assert controller._geometry_edit_schema_scope != previous_scope
    assert controller._geometry_edit_schema_scope.part_id == "part-rebound"
    assert "prepare_geometry_edit" in _published_names(controller)


@pytest.mark.parametrize("restriction", ["pending", "missing_handler", "disabled", "unsupported", "imported"])
def test_save_preview_keeps_policy_guard_without_reading_context(restriction: str) -> None:
    def unexpected_read():
        pytest.fail("save preview must use its explicitly supplied context")

    context = _context()
    controller = AuthoringWorkflowController(
        unexpected_read,
        {} if restriction == "missing_handler" else {"request_project_save": _handler},
    )
    if restriction == "pending":
        controller._stage = AuthoringWorkflowStage.SOLVE_PENDING
    elif restriction == "disabled":
        context = replace(context, capabilities=())
    elif restriction == "unsupported":
        context = replace(context, binding=replace(context.binding, supported=False))
    elif restriction == "imported":
        context = replace(context, binding=replace(context.binding, source_kind="imported"))

    with pytest.raises(ValueError, match="current native project is not ready to save"):
        controller.preview_project_save_proposal("proposal-save-publication", context)
    assert controller.project_save_record is None


def test_save_acceptance_still_requires_pending_gui_identity_and_current_revision() -> None:
    context = _context()
    controller = AuthoringWorkflowController(lambda: context, {"request_project_save": _handler})
    controller.observe_binding(context)
    record = controller.register_project_save_proposal("proposal-save-publication", context)
    assert not controller.can_accept_project_save_from_gui(record.proposal_id, record.proposal_hash, context)
    controller._stage = AuthoringWorkflowStage.PROJECT_SAVE_PENDING
    controller._pending_operation = "project_save"
    assert controller.can_accept_project_save_from_gui(record.proposal_id, record.proposal_hash, context)
    for changed_context in (
        replace(context, binding=replace(context.binding, session_revision=4)),
        replace(context, binding=replace(context.binding, document_id="document:other")),
        replace(context, capabilities=()),
    ):
        assert not controller.can_accept_project_save_from_gui(
            record.proposal_id, record.proposal_hash, changed_context,
        )
    assert not controller.can_accept_project_save_from_gui("proposal:other", record.proposal_hash, context)
    controller.begin_project_save_from_gui(record.proposal_id, record.proposal_hash, context)
    assert controller.project_save_record.state is ProposalState.RUNNING
    assert not controller.can_accept_project_save_from_gui(record.proposal_id, record.proposal_hash, context)
