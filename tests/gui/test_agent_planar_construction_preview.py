from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from fem.application import (
    ModelSession,
    PlanarConstructionCompileError,
    PlanarConstructionDiagnostic,
)
from fem_agent.authoring import AuthoringContractError, ProposalState
from fem_agent.geometry_authoring import geometry_recipe_from_payload
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import AgentProposalPreview
from fem_gui.geometry_preview import GeometryPreview
from fem_gui.main_window import FEMMainWindow
from tests.helpers.agent_planar_construction import (
    build_rectangle_arguments,
    make_planar_authoring_controller,
)


def _dispatch(controller, arguments, key: str, turn_id: str = "turn-a"):
    return controller.dispatch(
        "prepare_planar_construction_proposal",
        arguments,
        ToolExecutionContext(
            "phase5-planar-preview",
            0,
            key,
            turn_id=turn_id,
        ),
    )


def _missing_reference(name: str = "missing-a") -> dict[str, object]:
    return {
        "part_function": "引用恢复测试板",
        "construction": {
            "schema_version": 1,
            "name": "missing reference",
            "plane": "XY",
            "nodes": [
                {
                    "id": "plate",
                    "kind": "rectangle",
                    "x": 0.0,
                    "y": 0.0,
                    "width": 10.0,
                    "height": 4.0,
                },
                {
                    "id": "result",
                    "kind": "difference",
                    "base": "plate",
                    "subtract": [name],
                },
            ],
            "result_node_id": "result",
        },
        "output": "planar",
    }


def _split_then_cut_arguments() -> dict[str, object]:
    return {
        "part_function": "带内部切除和圆孔的平板",
        "construction": {
            "schema_version": 1,
            "name": "split material diagnostic",
            "plane": "XY",
            "nodes": [
                {
                    "id": "plate",
                    "kind": "rectangle",
                    "x": 0.0,
                    "y": 0.0,
                    "width": 100.0,
                    "height": 300.0,
                },
                {
                    "id": "divider",
                    "kind": "rectangle",
                    "x": 0.0,
                    "y": 140.0,
                    "width": 100.0,
                    "height": 20.0,
                },
                {
                    "id": "hole",
                    "kind": "circle",
                    "center_x": 10.0,
                    "center_y": 290.0,
                    "radius": 2.0,
                },
                {
                    "id": "result",
                    "kind": "difference",
                    "base": "plate",
                    "subtract": ["divider", "hole"],
                },
            ],
            "result_node_id": "result",
        },
        "output": "planar",
    }


def test_planar_preview_is_gui_only_recipe_bound_and_cleared() -> None:
    session = ModelSession()
    before = session.snapshot()
    bridge, controller = make_planar_authoring_controller(session)
    changes: list[tuple[str, AgentProposalPreview | None]] = []
    bridge.set_preview_listener(
        lambda proposal_id, preview: changes.append((proposal_id, preview))
    )

    result = _dispatch(controller, build_rectangle_arguments(), "planar-preview")

    assert result.ok, result.summary
    proposal_id = result.data["proposal_id"]
    preview = bridge._proposal_previews[proposal_id]
    proposal = bridge._records[proposal_id].proposal
    evidence = proposal.preconditions["local_evidence"]
    assert "preview" not in proposal.display_summary
    assert preview.dimension == 2
    assert preview.faces
    assert preview.edges
    assert preview.recipe_digest == evidence["output_recipe_digest"]
    assert preview.proof_digest == evidence["output_proof_digest"]
    provider_payload = json.dumps(result.data, sort_keys=True)
    assert '"faces"' not in provider_payload
    assert '"points"' not in provider_payload
    assert changes[-1] == (proposal_id, preview)

    other_bridge, _controller_unused = make_planar_authoring_controller(session)
    with pytest.raises(AuthoringContractError, match="does not match"):
        other_bridge.register_proposal(
            bridge._records[proposal_id].proposal,
            replace(preview, recipe_digest="0" * 64),
        )
    assert not other_bridge._records

    receipt = bridge.reject_from_gui_control(proposal_id)
    controller.record_proposal_state("geometry", receipt.state, "rejected")

    assert session.snapshot() == before
    assert receipt.state is ProposalState.REJECTED
    assert proposal_id not in bridge._proposal_previews
    assert changes[-1] == (proposal_id, None)
    assert controller.planar_construction_audit[-1].terminal_state == "rejected"


def test_direct_3d_preview_has_real_surface_cells_and_stale_clears() -> None:
    session = ModelSession()
    before = session.snapshot()
    bridge, controller = make_planar_authoring_controller(session)
    arguments = deepcopy(build_rectangle_arguments())
    arguments["output"] = {
        "kind": "extrusion",
        "profile_selection": "unique_material_profile",
        "height": 3.0,
    }

    result = _dispatch(controller, arguments, "3d-preview")

    assert result.ok, result.summary
    proposal_id = result.data["proposal_id"]
    preview = bridge._proposal_previews[proposal_id]
    proposal = bridge._records[proposal_id].proposal
    evidence = proposal.preconditions["local_evidence"]
    assert "preview" not in proposal.display_summary
    assert preview.dimension == 3
    assert preview.faces
    assert preview.edges
    assert preview.proof_digest == evidence["output_proof_digest"]
    assert bridge.stale_pending_proposals_from_gui("test session switch") == (proposal_id,)
    assert bridge.state(proposal_id) is ProposalState.STALE
    assert session.snapshot() == before
    assert proposal_id not in bridge._proposal_previews


def test_accept_success_failure_cancel_and_detach_clear_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = ModelSession()
    bridge, controller = make_planar_authoring_controller(session)
    result = _dispatch(controller, build_rectangle_arguments(), "accept-success")
    proposal_id = result.data["proposal_id"]
    proposal = bridge._records[proposal_id].proposal
    prepared_recipe = geometry_recipe_from_payload(proposal.operations[0].parameters["recipe"])
    preview = bridge._proposal_previews[proposal_id]
    receipt = bridge.accept_from_gui_control(proposal_id)
    committed_recipe = session.snapshot().parts[0].geometry_recipe
    assert committed_recipe == prepared_recipe
    assert receipt.state is ProposalState.SUCCEEDED
    assert proposal_id not in bridge._proposal_previews

    session = ModelSession()
    bridge, controller = make_planar_authoring_controller(session)
    before = session.snapshot()
    result = _dispatch(controller, build_rectangle_arguments(), "accept-failure")
    proposal_id = result.data["proposal_id"]

    def fail_accept(*_args, **_kwargs):
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(session, "create_native_project_with_first_part", fail_accept)
    receipt = bridge.accept_from_gui_control(proposal_id)
    assert receipt.state is ProposalState.FAILED
    assert session.snapshot() == before
    assert proposal_id not in bridge._proposal_previews

    session = ModelSession()
    before = session.snapshot()
    bridge, controller = make_planar_authoring_controller(session)
    result = _dispatch(controller, build_rectangle_arguments(), "cancel")
    controller.cancel_turn("provider operation cancelled")
    assert session.snapshot() == before
    proposal_id = result.data["proposal_id"]
    assert bridge.cancel_pending_proposals_from_gui("provider cancelled") == (
        proposal_id,
    )
    assert bridge._records[proposal_id].state is ProposalState.CANCELLED
    assert proposal_id not in bridge._proposal_previews

    bridge, controller = make_planar_authoring_controller(ModelSession())
    result = _dispatch(controller, build_rectangle_arguments(), "detach")
    proposal_id = result.data["proposal_id"]
    changes = []
    bridge.set_preview_listener(lambda item, preview: changes.append((item, preview)))
    bridge.detach_gui_callbacks()
    assert proposal_id not in bridge._proposal_previews
    assert changes == [(proposal_id, None)]


def test_session_rebind_and_drawer_close_restore_committed_preview() -> None:
    bridge, controller = make_planar_authoring_controller(ModelSession())
    result = _dispatch(controller, build_rectangle_arguments(), "session-switch")
    proposal_id = result.data["proposal_id"]
    other = ModelSession()
    bridge.bind_snapshot(other.snapshot(), document_id="document-other")
    assert proposal_id not in bridge._proposal_previews

    committed = GeometryPreview(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        (),
        ((0, 1),),
        topological_dimension=2,
    )
    proposal_preview = AgentProposalPreview(
        2,
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        ((0, 1, 2),),
        ((0, 1), (1, 2), (2, 0)),
        "a" * 64,
        "b" * 64,
    )

    class _Viewport:
        def __init__(self):
            self.current = committed

        def show_geometry_preview(self, preview, **_kwargs):
            self.current = preview

    class _Bridge:
        def __init__(self):
            self.cleared = False

        def clear_proposal_previews(self):
            self.cleared = True

    class _Harness:
        _agent_proposal_preview_changed = FEMMainWindow._agent_proposal_preview_changed
        _close_agent_proposal_preview = FEMMainWindow._close_agent_proposal_preview
        _restore_after_agent_proposal_preview = (
            FEMMainWindow._restore_after_agent_proposal_preview
        )

        def __init__(self):
            self._agent_proposal_preview_id = None
            self.viewport = _Viewport()
            self.agent_authoring_bridge = _Bridge()

        def _current_native_geometry_preview(self):
            return committed

    harness = _Harness()
    harness._agent_proposal_preview_changed("proposal", proposal_preview)
    assert harness.viewport.current.faces == proposal_preview.faces
    harness._close_agent_proposal_preview()
    assert harness.viewport.current is committed
    assert harness.agent_authoring_bridge.cleared is True


def test_actual_diagnostic_retry_can_revise_same_ir() -> None:
    _bridge, controller = make_planar_authoring_controller(ModelSession())
    request = _missing_reference()
    failed = _dispatch(controller, request, "actual-reference-first")
    assert failed.data["diagnostic"]["code"] == "planar-ir.reference-missing"
    repaired = deepcopy(request)
    repaired["construction"]["nodes"].insert(1, {
        "id": "hole", "kind": "circle", "center_x": 2, "center_y": 2,
        "radius": 0.5,
    })
    repaired["construction"]["nodes"][-1]["subtract"] = ["hole"]
    succeeded = _dispatch(controller, repaired, "actual-reference-repaired")
    assert succeeded.ok, succeeded.summary


@pytest.mark.parametrize(
    ("code", "node_id", "retryable", "allowed_fields"),
    [
        ("planar-ir.invalid-primitive", "plate", True, ("width",)),
        ("planar-ir.equivalence-failed", None, False, ()),
    ],
)
def test_stable_diagnostics_are_provider_safe(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    node_id: str | None,
    retryable: bool,
    allowed_fields: tuple[str, ...],
) -> None:
    def fail(_construction, **_kwargs):
        raise PlanarConstructionCompileError(
            PlanarConstructionDiagnostic(
                code,
                "bounded backend detail",
                node_id,
                retryable,
                allowed_fields,
            )
        )

    monkeypatch.setattr("fem_gui.agent_authoring.compile_planar_construction", fail)
    bridge, controller = make_planar_authoring_controller(ModelSession())

    result = _dispatch(controller, build_rectangle_arguments(), f"diagnostic-{code.replace('.', '-')}")

    assert not result.ok
    assert result.data["diagnostic"] == {
        "code": code,
        "message": "bounded backend detail",
        "node_id": node_id,
        "retryable": retryable,
        "allowed_fields": list(allowed_fields),
        "model_unchanged": True,
    }
    assert result.data["required_action"] == "revise_same_planar_construction_ir"
    assert not bridge._records


def test_retry_requires_allowed_slice_change_and_resets_next_turn() -> None:
    _bridge, controller = make_planar_authoring_controller(ModelSession())
    first = _dispatch(controller, _missing_reference(), "first")
    record = controller.planar_construction_audit[-1]
    assert len(record.construction_digest) == 64
    assert record.stage == "validation"
    assert record.diagnostic_code == "planar-ir.reference-missing"
    assert not any(hasattr(record, field) for field in ("construction", "points", "faces"))
    same = _dispatch(controller, _missing_reference(), "same")
    assert first.data["retry"]["retryable"] is True
    assert same.data["retry"]["retryable"] is False
    assert "modify" in same.data["retry"]["blocker"]

    next_turn = _dispatch(
        controller,
        _missing_reference(),
        "next-turn",
        turn_id="turn-b",
    )
    assert next_turn.data["retry"] == {
        **next_turn.data["retry"],
        "attempt": 1,
        "retryable": True,
        "blocker": None,
    }

    _bridge, controller = make_planar_authoring_controller(ModelSession())
    first = _dispatch(controller, _missing_reference(), "allowed-first")
    unrelated = _missing_reference()
    unrelated["construction"]["name"] = "unrelated change"
    blocked = _dispatch(controller, unrelated, "unrelated")
    assert first.data["retry"]["retryable"] is True
    assert blocked.data["retry"]["retryable"] is False

    _bridge, controller = make_planar_authoring_controller(ModelSession())
    _dispatch(controller, _missing_reference("missing-a"), "changed-first")
    changed = _dispatch(controller, _missing_reference("missing-b"), "changed-second")
    exhausted = _dispatch(controller, _missing_reference("missing-c"), "changed-third")
    assert changed.data["retry"]["retryable"] is True
    assert exhausted.data["retry"]["retryable"] is False
    assert "three attempts" in exhausted.data["retry"]["blocker"]


def test_split_material_diagnostic_identifies_source_and_topology() -> None:
    _bridge, controller = make_planar_authoring_controller(ModelSession())
    first = _dispatch(controller, _split_then_cut_arguments(), "split-first")

    assert not first.ok
    assert first.data["diagnostic"]["code"] == (
        "planar-ir.feature-splits-material"
    )
    assert first.data["diagnostic"]["node_id"] == "divider"
    evidence = first.data["diagnostic"]["evidence"]
    assert {
        key: value
        for key, value in evidence.items()
        if key != "tool_bounding_box"
    } == {
        "material_profile_count_before": 1,
        "material_profile_count_after": 2,
        "boundary_contact": "detected_or_crossed",
        "positive_clearance": False,
    }
    assert evidence["tool_bounding_box"] == pytest.approx(
        [0.0, 140.0, 100.0, 160.0],
        abs=2.0e-7,
    )
    assert first.data["retry"]["retryable"] is True

    equivalent = _split_then_cut_arguments()
    equivalent["construction"]["nodes"][1] = {
        "id": "divider",
        "kind": "polygon",
        "vertices": [
            [0.0, 140.0],
            [100.0, 140.0],
            [100.0, 160.0],
            [0.0, 160.0],
        ],
    }
    same_topology = _dispatch(controller, equivalent, "split-equivalent")

    assert not same_topology.ok
    assert same_topology.data["retry"]["retryable"] is False
    assert "same topology failure evidence" in same_topology.data["retry"][
        "blocker"
    ]

    corrected = _split_then_cut_arguments()
    corrected["construction"]["nodes"][1].update(
        {"x": 20.0, "width": 60.0}
    )
    accepted = _dispatch(controller, corrected, "split-corrected")

    assert accepted.ok, accepted.data


def test_invalid_output_is_rejected_before_cad_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_compile(_construction, **_kwargs):
        raise AssertionError("CAD compilation must not start for invalid output")

    monkeypatch.setattr(
        "fem_gui.agent_authoring.compile_planar_construction",
        unexpected_compile,
    )
    bridge, controller = make_planar_authoring_controller(ModelSession())
    arguments = build_rectangle_arguments()
    arguments["output"] = {"kind": "planar", "height": 10.0}

    result = _dispatch(controller, arguments, "invalid-output")

    assert not result.ok
    assert result.data["diagnostic"]["code"] == "planar-ir.transform-invalid"
    assert "only kind='planar'" in result.data["diagnostic"]["message"]
    assert not bridge._records


def test_fourth_planar_attempt_is_blocked_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls = 0

    def fail_compile(_construction, **_kwargs):
        nonlocal compile_calls
        compile_calls += 1
        raise PlanarConstructionCompileError(
            PlanarConstructionDiagnostic(
                "planar-ir.invalid-primitive",
                "injected primitive failure",
                "plate",
                True,
                ("width",),
            )
        )

    monkeypatch.setattr(
        "fem_gui.agent_authoring.compile_planar_construction",
        fail_compile,
    )
    _bridge, controller = make_planar_authoring_controller(ModelSession())
    results = []
    for index in range(4):
        arguments = build_rectangle_arguments()
        arguments["construction"]["nodes"][0]["width"] = 100.0 + index
        results.append(_dispatch(controller, arguments, f"attempt-{index}"))

    assert compile_calls == 3
    assert results[-1].data["required_action"] == (
        "restart_planar_construction_in_new_turn"
    )
    assert results[-1].data["retry"]["retryable"] is False


def test_retry_accepts_allowed_top_level_result_change() -> None:
    _bridge, controller = make_planar_authoring_controller(ModelSession())
    request = _missing_reference()
    request["construction"]["result_node_id"] = "missing-result"
    first = _dispatch(controller, request, "result-first")
    changed = deepcopy(request)
    changed["construction"]["result_node_id"] = "result"
    second = _dispatch(controller, changed, "result-second")
    assert first.data["retry"]["retryable"] is True
    assert second.data["retry"]["attempt"] == 2
