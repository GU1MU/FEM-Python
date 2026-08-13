from __future__ import annotations

from dataclasses import replace

import pytest
from PySide6.QtWidgets import QApplication

from fem.application import ModelSession, UnitContext
from fem.geometry import (
    RectangleGeometry,
    SketchCircle,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
)
from fem_agent.authoring import (
    AgentProposal,
    AuthoringAuthorizationError,
    ModelOperation,
    OperationKind,
    ProposalKind,
    ProposalState,
    UnitContextSummary,
)
from fem_agent.geometry_authoring import (
    add_planar_circle,
    create_geometry_edit_proposal,
    create_geometry_proposal,
    disk_geometry,
    geometry_draft,
    plate_with_hole_geometry,
)
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    authoring_context_from_snapshot,
)
from fem_gui.main_window import FEMMainWindow


def _units() -> UnitContextSummary:
    return UnitContextSummary(
        length="mm",
        force="N",
        stress="MPa",
        density=None,
        acceleration=None,
        convention="N-mm-MPa",
    )


def _application_units() -> UnitContext:
    return UnitContext.from_dict(_units().to_dict())


def _proposal(
    session: ModelSession,
    *,
    proposal_id: str,
    draft=None,
    part_function: str = "偏心孔板",
    project_function: str | None = None,
    document_id: str | int | None = None,
) -> AgentProposal:
    context = authoring_context_from_snapshot(
        session.snapshot(),
        document_id=document_id,
    )
    return create_geometry_proposal(
        proposal_id=proposal_id,
        agent_session_id="agent-session-a2",
        turn_id=f"turn-{proposal_id}",
        source_tool_call_ids=(f"call-{proposal_id}",),
        context=context,
        draft_revision=1,
        draft=(
            plate_with_hole_geometry(
                "实体-偏心孔板",
                width=10.0,
                height=6.0,
                hole_radius=1.0,
                center_offset=(1.5, -1.0),
            )
            if draft is None
            else draft
        ),
        part_function=part_function,
        project_function=project_function,
        unit_context=_units(),
    )


def _bridge(session: ModelSession, refreshes: list[int]) -> AgentAuthoringBridge:
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(
            session,
            lambda: refreshes.append(session.session_revision),
        )
    )
    bridge.bind_snapshot(session.snapshot())
    return bridge


def _open_sketch() -> SketchGeometry:
    return SketchGeometry(
        "草图-开放轮廓",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 2.0, 0.0),
            SketchPoint("P3", 2.0, 1.0),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
        ),
    )


def _self_intersecting_sketch() -> SketchGeometry:
    return SketchGeometry(
        "草图-自交轮廓",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 2.0, 1.0),
            SketchPoint("P3", 0.0, 1.0),
            SketchPoint("P4", 2.0, 0.0),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
        ),
    )


def test_a2_blank_creation_is_atomic_and_refreshes_once_only_after_accept() -> None:
    session = ModelSession()
    refreshes: list[int] = []
    bridge = _bridge(session, refreshes)
    proposal = _proposal(
        session,
        proposal_id="proposal-create",
        project_function="偏心孔板",
    )
    before = session.snapshot()

    assert proposal.display_summary["title"] == "加入部件"

    bridge.register_proposal(proposal)

    pending = session.snapshot()
    assert pending.session_revision == before.session_revision
    assert pending.source_kind is None
    assert pending.parts == ()
    assert refreshes == []

    receipt = bridge.accept_from_gui_control(proposal.proposal_id)
    accepted = session.snapshot()

    assert receipt.state is ProposalState.SUCCEEDED
    assert accepted.session_revision == before.session_revision + 1
    assert accepted.source_kind == "native"
    assert accepted.model_name == "模型-偏心孔板"
    assert accepted.unit_context == _application_units()
    assert len(accepted.parts) == 1
    assert accepted.parts[0].name == "部件-偏心孔板"
    assert accepted.parts[0].mesh_settings is None
    assert refreshes == [accepted.session_revision]


def test_a2_reject_keeps_blank_session_tree_actor_proxy_and_revision_unchanged() -> (
    None
):
    session = ModelSession()
    refreshes: list[int] = []
    bridge = _bridge(session, refreshes)
    proposal = _proposal(
        session,
        proposal_id="proposal-reject",
        project_function="偏心孔板",
    )
    before = session.snapshot()
    bridge.register_proposal(proposal)

    receipt = bridge.reject_from_gui_control(proposal.proposal_id)
    after = session.snapshot()

    assert receipt.state is ProposalState.REJECTED
    assert after.session_revision == before.session_revision
    assert after.parts == before.parts
    assert after.artifact is before.artifact is None
    assert refreshes == []


def test_a2_native_accept_adds_exactly_one_allocated_part_and_one_refresh() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-组合件",
        _application_units(),
        RectangleGeometry("实体-矩形", 4.0, 2.0),
        part_name="部件-板",
    )
    refreshes: list[int] = []
    bridge = _bridge(session, refreshes)
    proposal = _proposal(
        session,
        proposal_id="proposal-add",
        draft=disk_geometry("实体-圆盘", radius=1.0),
        part_function="圆盘",
    )
    before = session.snapshot()

    assert proposal.display_summary["title"] == "加入部件"
    bridge.register_proposal(proposal)

    receipt = bridge.accept_from_gui_control(proposal.proposal_id)
    after = session.snapshot()

    assert receipt.state is ProposalState.SUCCEEDED
    assert after.session_revision == before.session_revision + 1
    assert [part.name for part in after.parts] == ["部件-板", "部件-圆盘"]
    assert refreshes == [after.session_revision]


def test_a2_existing_part_adds_second_hole_without_delete_or_recreate() -> None:
    session = ModelSession()
    original = plate_with_hole_geometry(
        "实体-旧孔板",
        width=100.0,
        height=200.0,
        hole_radius=10.0,
        hole_center=(50.0, 100.0),
    )
    session.create_native_project_with_first_part(
        "模型-双孔板",
        _application_units(),
        original.recipe,
        part_name="部件-孔板",
    )
    refreshes: list[int] = []
    bridge = _bridge(session, refreshes)
    before = session.snapshot()
    part_id = str(before.parts[0].id)
    context = authoring_context_from_snapshot(before)
    assert context.parts[0].recipe_kind == "planar_sketch"
    edited = add_planar_circle(
        before.parts[0].geometry_recipe,
        center_x=50.0,
        center_y=130.0,
        radius=5.0,
    )
    proposal = create_geometry_edit_proposal(
        proposal_id="proposal-add-second-hole",
        agent_session_id="agent-session-a2",
        turn_id="turn-add-second-hole",
        source_tool_call_ids=("call-add-second-hole",),
        context=context,
        draft_revision=1,
        part_id=part_id,
        draft=edited,
        summary="增加第二个圆孔",
    )

    bridge.register_proposal(proposal)
    receipt = bridge.accept_from_gui_control(proposal.proposal_id)
    after = session.snapshot()

    assert receipt.state is ProposalState.SUCCEEDED
    assert len(after.parts) == 1
    assert str(after.parts[0].id) == part_id
    assert after.parts[0].name == "部件-孔板"
    assert type(after.parts[0].geometry_recipe) is SketchGeometry
    assert len(
        [
            curve
            for curve in after.parts[0].geometry_recipe.curves
            if isinstance(curve, SketchCircle)
        ]
    ) == 2
    assert after.session_revision == before.session_revision + 1
    assert refreshes == [after.session_revision]


def test_a2_stale_geometry_proposal_cannot_commit() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-组合件",
        _application_units(),
        RectangleGeometry("实体-矩形", 4.0, 2.0),
        part_name="部件-板",
    )
    refreshes: list[int] = []
    bridge = _bridge(session, refreshes)
    proposal = _proposal(
        session,
        proposal_id="proposal-stale",
        draft=disk_geometry("实体-圆盘", radius=1.0),
        part_function="圆盘",
    )
    bridge.register_proposal(proposal)
    session.add_native_part(
        RectangleGeometry("实体-外部", 1.0, 1.0),
        name="部件-外部",
        mesh_settings=None,
        unit_context=_application_units(),
    )

    stale_ids = bridge.bind_snapshot(session.snapshot())
    count_after_external_edit = len(session.snapshot().parts)

    assert stale_ids == (proposal.proposal_id,)
    assert bridge.state(proposal.proposal_id) is ProposalState.STALE
    with pytest.raises(AuthoringAuthorizationError):
        bridge.accept_from_gui_control(proposal.proposal_id)
    assert len(session.snapshot().parts) == count_after_external_edit
    assert refreshes == []


def test_a2_invalid_hole_commit_failure_is_atomic() -> None:
    session = ModelSession()
    refreshes: list[int] = []
    bridge = _bridge(session, refreshes)
    context = authoring_context_from_snapshot(session.snapshot())
    invalid = AgentProposal.create(
        proposal_id="proposal-invalid-hole",
        proposal_kind=ProposalKind.GEOMETRY,
        agent_session_id="agent-session-a2",
        turn_id="turn-invalid-hole",
        source_tool_call_ids=("call-invalid-hole",),
        target_document_id=context.binding.document_id,
        target_session_id=context.binding.session_id,
        base_session_revision=context.binding.session_revision,
        draft_revision=1,
        operations=(
            ModelOperation(
                OperationKind.CREATE_NATIVE_PROJECT,
                {
                    "project_name": "模型-孔板",
                    "part_name": "部件-孔板",
                    "unit_context": _units().to_dict(),
                    "recipe": {
                        "kind": "plate_with_hole",
                        "name": "实体-孔板",
                        "width": 10.0,
                        "height": 6.0,
                        "hole_x": 0.5,
                        "hole_y": 3.0,
                        "hole_radius": 1.0,
                    },
                },
            ),
        ),
        preconditions={"source_kind": "blank"},
        expected_changes={"part_count_delta": 1},
        invalidation_impact={},
        display_summary={"title": "无效孔"},
    )
    before = session.snapshot()
    bridge.register_proposal(invalid)

    receipt = bridge.accept_from_gui_control(invalid.proposal_id)
    after = session.snapshot()

    assert receipt.state is ProposalState.FAILED
    assert after.session_id == before.session_id
    assert after.session_revision == before.session_revision
    assert after.source_kind is None
    assert after.parts == ()
    assert refreshes == []


@pytest.mark.parametrize(
    "invalid_recipe",
    (_open_sketch(), _self_intersecting_sketch()),
    ids=("open-profile", "self-intersecting-profile"),
)
@pytest.mark.parametrize("replace_existing", (False, True))
def test_a2_invalid_strict_profile_create_or_replace_is_atomic(
    invalid_recipe: SketchGeometry,
    replace_existing: bool,
) -> None:
    session = ModelSession()
    if replace_existing:
        session.create_native_project_with_first_part(
            "模型-现有板",
            _application_units(),
            RectangleGeometry("实体-现有板", 4.0, 2.0),
            part_name="部件-现有板",
        )
    refreshes: list[int] = []
    bridge = _bridge(session, refreshes)
    before = session.snapshot()
    draft = geometry_draft(invalid_recipe)
    suffix = "open" if len(invalid_recipe.curves) == 2 else "self-intersecting"
    if replace_existing:
        part_id = str(before.parts[0].id)
        proposal = create_geometry_edit_proposal(
            proposal_id=f"proposal-invalid-replace-{suffix}",
            agent_session_id="agent-session-a2",
            turn_id="turn-invalid-replace",
            source_tool_call_ids=("call-invalid-replace",),
            context=authoring_context_from_snapshot(before),
            draft_revision=1,
            part_id=part_id,
            draft=draft,
            summary="替换为无效严格草图",
        )
    else:
        proposal = _proposal(
            session,
            proposal_id=f"proposal-invalid-create-{suffix}",
            draft=draft,
            project_function="无效严格草图",
        )

    bridge.register_proposal(proposal)
    receipt = bridge.accept_from_gui_control(proposal.proposal_id)

    assert receipt.state is ProposalState.FAILED
    assert session.snapshot() == before
    assert refreshes == []


@pytest.mark.parametrize(
    "invalid_recipe",
    (_open_sketch(), _self_intersecting_sketch()),
    ids=("open-profile", "self-intersecting-profile"),
)
def test_a2_invalid_strict_profile_add_is_atomic(
    invalid_recipe: SketchGeometry,
) -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-现有板",
        _application_units(),
        RectangleGeometry("实体-现有板", 4.0, 2.0),
        part_name="部件-现有板",
    )
    refreshes: list[int] = []
    bridge = _bridge(session, refreshes)
    before = session.snapshot()
    suffix = "open" if len(invalid_recipe.curves) == 2 else "self-intersecting"
    proposal = _proposal(
        session,
        proposal_id=f"proposal-invalid-add-{suffix}",
        draft=geometry_draft(invalid_recipe),
        part_function="无效严格草图",
    )

    bridge.register_proposal(proposal)
    receipt = bridge.accept_from_gui_control(proposal.proposal_id)

    assert receipt.state is ProposalState.FAILED
    assert session.snapshot() == before
    assert refreshes == []


def test_a2_real_port_rejects_name_allocator_bypass_without_mutation() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-组合件",
        _application_units(),
        RectangleGeometry("实体-矩形", 4.0, 2.0),
        part_name="部件-圆盘",
    )
    refreshes: list[int] = []
    bridge = _bridge(session, refreshes)
    proposal = _proposal(
        session,
        proposal_id="proposal-bypass",
        draft=disk_geometry("实体-圆盘", radius=1.0),
        part_function="圆盘",
    )
    operation = proposal.operations[0]
    tampered = AgentProposal.create(
        proposal_id="proposal-bypass-manual",
        proposal_kind=ProposalKind.GEOMETRY,
        agent_session_id=proposal.agent_session_id,
        turn_id="turn-bypass-manual",
        source_tool_call_ids=("call-bypass-manual",),
        target_document_id=proposal.target_document_id,
        target_session_id=proposal.target_session_id,
        base_session_revision=proposal.base_session_revision,
        draft_revision=proposal.draft_revision,
        operations=(
            replace(
                operation,
                parameters={
                    **operation.parameters,
                    "part_name": "部件-圆盘",
                },
            ),
        ),
        preconditions=proposal.preconditions,
        expected_changes=proposal.expected_changes,
        invalidation_impact=proposal.invalidation_impact,
        display_summary=proposal.display_summary,
    )
    before = session.snapshot()
    bridge.register_proposal(tampered)

    receipt = bridge.accept_from_gui_control(tampered.proposal_id)

    assert receipt.state is ProposalState.FAILED
    assert session.snapshot().session_revision == before.session_revision
    assert len(session.snapshot().parts) == len(before.parts)
    assert refreshes == []


def test_a2_main_window_projects_one_accepted_geometry_refresh(
    monkeypatch,
) -> None:
    QApplication.instance() or QApplication([])
    window = FEMMainWindow()
    rebuild_count = 0
    original = window._rebuild_full_projection

    def counted_rebuild() -> None:
        nonlocal rebuild_count
        rebuild_count += 1
        original()

    window.agent_authoring_bridge.port._refresh_callback = counted_rebuild
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
    active_document = window.workspace.active_document()
    assert active_document is not None
    proposal = _proposal(
        window.session,
        proposal_id="proposal-main-window",
        project_function="偏心孔板",
        document_id=active_document.document_id,
    )
    before_revision = window.document.session_revision
    before_preview = window.viewport._geometry_preview
    window.agent_authoring_bridge.register_proposal(proposal)

    assert window.document.session_revision == before_revision
    assert window.document.parts == ()
    assert window.viewport._geometry_preview is before_preview

    receipt = window.agent_authoring_bridge.accept_from_gui_control(
        proposal.proposal_id
    )

    assert receipt.state is ProposalState.SUCCEEDED
    assert rebuild_count == 1
    assert [part.name for part in window.document.parts] == ["部件-偏心孔板"]
    assert window.viewport._geometry_preview is not None
    window.close()
