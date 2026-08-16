"""Phase 5 bounded Profile-transform diagnostics and atomic recovery tests."""

from __future__ import annotations

import pytest

from fem.application import ModelSession, UnitContext
from fem.geometry import (
    CylinderGeometry,
    SketchCircle,
    SketchGeometry,
)
from fem_agent.diagnostics import (
    PROFILE_TRANSFORM_DIAGNOSTIC_CODES,
    ProfileTransformDiagnostic,
    profile_transform_diagnostic,
)
from fem_agent.geometry_authoring import planar_sketch_geometry
from fem_agent.authoring import AuthoringContractError
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui import agent_authoring
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from tests.geometry.test_profile_extrusion import two_profile_sketch


def _controller(session: ModelSession, refresh=None):
    if refresh is None:
        def refresh():
            return None
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, refresh)
    )
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    return bridge, controller


def _session(recipe) -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Phase 5 diagnostics",
        UnitContext("mm", "N", "MPa"),
        recipe,
        part_name="Sketch",
    )
    return session


def _context(controller, name: str = "phase5") -> ToolExecutionContext:
    return ToolExecutionContext(name, 0, name)


def _prepare_extrusion(controller, **arguments):
    return controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "height": 2.0,
            **arguments,
        },
        _context(controller, "extrusion"),
    )


def _valid_path() -> dict[str, object]:
    return {
        "points": [
            {"name": "A", "x": 0.0, "y": 0.0, "z": 0.0},
            {"name": "B", "x": 0.0, "y": 0.0, "z": 1.0},
        ],
        "members": [{"name": "AB", "start": "A", "end": "B"}],
    }


@pytest.mark.parametrize("code", PROFILE_TRANSFORM_DIAGNOSTIC_CODES)
def test_phase5_every_profile_transform_code_is_bounded_and_recoverable(code: str):
    diagnostic = profile_transform_diagnostic(
        code,
        operation="Profile extrusion",
        detail="local failure detail",
    ).to_dict()
    assert set(
        ("code", "message", "operation", "retryable", "required_fields", "preserve_draft")
    ) <= set(diagnostic)
    assert diagnostic["code"] == code
    assert isinstance(diagnostic["retryable"], bool)
    assert isinstance(diagnostic["required_fields"], list)
    assert isinstance(diagnostic["preserve_draft"], bool)
    assert len(diagnostic["message"].encode("utf-8")) <= 512
    assert "traceback" not in diagnostic["message"].casefold()
    assert "mesh workaround" not in diagnostic["message"].casefold()


def test_phase5_unknown_codes_fail_fast_and_recovery_text_matches_flags() -> None:
    with pytest.raises(ValueError, match="unknown Profile transform diagnostic code"):
        profile_transform_diagnostic(
            "profile-transform.unknown",
            operation="Profile extrusion",
        )
    with pytest.raises(ValueError, match="unknown Profile transform diagnostic code"):
        ProfileTransformDiagnostic(
            "profile-transform.unknown",
            "bad",
            "Profile extrusion",
            False,
            (),
        )
    boundary = profile_transform_diagnostic(
        "profile-transform.topology-unproven",
        operation="Profile extrusion",
        retryable=False,
        required_fields=(),
    ).to_dict()
    assert "reread" not in boundary["message"].casefold()
    assert "supported boundary" in boundary["message"]
    retry = profile_transform_diagnostic(
        "profile-transform.preflight-failed",
        operation="Profile extrusion",
        retryable=True,
        required_fields=(),
    ).to_dict()
    assert "reread" in retry["message"].casefold()


def test_phase5_profile_messages_are_utf8_bounded_and_hide_local_paths() -> None:
    diagnostic = profile_transform_diagnostic(
        "profile-transform.preflight-failed",
        operation="拉伸" * 100,
        detail=("详细失败信息" * 300) + " C:\\private\\model.brep traceback: secret",
        candidates=["候选" * 200],
        first_failed_member="成员" * 200,
    ).to_dict()
    assert len(diagnostic["message"].encode("utf-8")) <= 512
    assert len(diagnostic["operation"].encode("utf-8")) <= 96
    assert len(diagnostic["candidates"][0].encode("utf-8")) <= 192
    assert len(diagnostic["first_failed_member"].encode("utf-8")) <= 128
    assert "C:\\private" not in diagnostic["message"]
    assert "traceback" not in diagnostic["message"].casefold()


@pytest.mark.parametrize(
    ("retryable", "required_fields", "recovery"),
    [
        (
            True,
            ("height", "profile_selection"),
            "Next input: height, profile_selection.",
        ),
        (
            True,
            (),
            "Next action: reread the current context and retry.",
        ),
        (
            False,
            (),
            "Next action: revise the request or geometry to a supported boundary.",
        ),
    ],
)
def test_phase5_utf8_detail_keeps_complete_recovery_text(
    retryable: bool,
    required_fields: tuple[str, ...],
    recovery: str,
) -> None:
    diagnostic = profile_transform_diagnostic(
        "profile-transform.preflight-failed",
        operation="变换操作" * 100,
        detail="详细失败信息" * 400,
        retryable=retryable,
        required_fields=required_fields,
    ).to_dict()
    message = diagnostic["message"]
    assert len(message.encode("utf-8")) <= 512
    assert recovery in message
    assert "Next input:" in message or "Next action:" in message


def test_phase5_required_fields_are_bounded_and_deduplicated_before_recovery() -> None:
    diagnostic = profile_transform_diagnostic(
        "profile-transform.invalid-source-id",
        operation="Profile extrusion",
        detail="详细失败信息" * 400,
        required_fields=("字段" * 100, "字段" * 100, "另一个字段" * 100),
    ).to_dict()
    assert len(diagnostic["message"].encode("utf-8")) <= 512
    assert len(diagnostic["required_fields"]) == 2
    assert "Next input:" in diagnostic["message"]


def test_phase5_part_not_found_is_typed_and_keeps_snapshot() -> None:
    session = _session(planar_sketch_geometry("Sketch", contours=(SketchCircle("material", 0, 0, 1),)).recipe)
    _bridge, controller = _controller(session)
    before = session.snapshot()
    result = controller.dispatch(
        "read_profile_transform_context",
        {"part_id": "P404"},
        _context(controller, "missing"),
    )
    assert not result.ok
    assert result.data["diagnostic"]["code"] == "profile-transform.part-not-found"
    assert result.data["diagnostic"]["required_fields"] == ["part_id"]
    assert session.snapshot() == before


def test_phase5_non_planar_blocks_while_legacy_exact_topology_is_transformable() -> None:
    non_planar = _session(CylinderGeometry("Solid", 1.0, 2.0))
    non_planar_refresh = []
    _bridge, non_planar_controller = _controller(
        non_planar,
        lambda: non_planar_refresh.append("refresh"),
    )
    non_planar_before = non_planar.snapshot()
    result = non_planar_controller.dispatch(
        "read_profile_transform_context",
        {"part_id": "P1"},
        _context(non_planar_controller, "non-planar"),
    )
    assert (
        result.data["operations"]["extrusion"]["blocking_code"]
        == "profile-transform.source-not-planar"
    )
    prepared = non_planar_controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "height": 2.0,
        },
        _context(non_planar_controller, "non-planar-prepare"),
    )
    non_planar_diagnostic = prepared.data["diagnostic"]
    assert non_planar_diagnostic == {
        "code": "profile-transform.source-not-planar",
        "message": non_planar_diagnostic["message"],
        "operation": "Profile 拉伸",
        "retryable": False,
        "required_fields": [],
        "preserve_draft": True,
    }
    assert non_planar.snapshot() == non_planar_before
    assert non_planar_refresh == []

    # The Profile-transform contract is now defined over exact topology rather
    # than sketch strictness: a legacy non-strict sketch whose topology proof
    # is exact exposes its feature-history Profile and stays transformable.
    legacy = _session(
        SketchGeometry("Legacy", (SketchCircle("material", 0, 0, 1),))
    )
    legacy_refresh = []
    _bridge, legacy_controller = _controller(
        legacy,
        lambda: legacy_refresh.append("refresh"),
    )
    legacy_before = legacy.snapshot()
    result = legacy_controller.dispatch(
        "read_profile_transform_context",
        {"part_id": "P1"},
        _context(legacy_controller, "legacy"),
    )
    extrusion = result.data["operations"]["extrusion"]
    assert extrusion["blocking_code"] is None
    assert extrusion["available"] is True
    assert result.data["topology_exact"] is True
    assert [profile["face_id"] for profile in result.data["profiles"]] == [
        "face:domain",
    ]
    prepared = legacy_controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "height": 2.0,
        },
        _context(legacy_controller, "legacy-prepare"),
    )
    assert prepared.ok, prepared.summary
    assert prepared.data["proposal_id"]
    assert legacy.snapshot() == legacy_before
    assert legacy_refresh == []


def test_phase5_missing_profile_height_and_ambiguous_selection_have_next_inputs() -> None:
    session = _session(two_profile_sketch())
    _bridge, controller = _controller(session)
    missing_height = controller.dispatch(
        "prepare_profile_extrusion",
        {"part_id": "P1", "profile_selection": "unique_material_profile"},
        _context(controller, "missing-height"),
    )
    assert missing_height.data["diagnostic"]["code"] == "profile-transform.nonpositive-height"
    assert missing_height.data["diagnostic"]["required_fields"] == ["height"]

    ambiguous = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "height": 2.0,
        },
        _context(controller, "ambiguous"),
    )
    diagnostic = ambiguous.data["diagnostic"]
    assert diagnostic["code"] == "profile-transform.ambiguous-material-profiles"
    assert diagnostic["required_fields"] == ["profile_selection"]
    assert diagnostic["candidates"]


def test_phase5_invalid_source_and_stale_context_do_not_mutate_session() -> None:
    session = _session(planar_sketch_geometry("Sketch", contours=(SketchCircle("material", 0, 0, 1),)).recipe)
    _bridge, controller = _controller(session)
    before = session.snapshot()
    invalid = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": ["face:missing"],
            "context_revision": before.session_revision,
            "height": 2.0,
        },
        _context(controller, "invalid-source"),
    )
    assert invalid.data["diagnostic"]["code"] == "profile-transform.invalid-source-id"
    stale = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": ["face:domain"],
            "context_revision": before.session_revision + 1,
            "height": 2.0,
        },
        _context(controller, "stale"),
    )
    assert stale.data["diagnostic"]["code"] == "profile-transform.stale-context"
    assert session.snapshot() == before


def test_phase5_path_boundary_reports_first_invalid_member_and_frame(monkeypatch) -> None:
    session = _session(planar_sketch_geometry("Sketch", contours=(SketchCircle("material", 0, 0, 1),)).recipe)
    _bridge, controller = _controller(session)
    broken = _valid_path()
    broken["members"] = [
        {"name": "AB", "start": "A", "end": "B"},
        {"name": "CD", "start": "C", "end": "D"},
    ]
    broken["points"] = [
        *broken["points"],
        {"name": "C", "x": 2.0, "y": 0.0, "z": 0.0},
        {"name": "D", "x": 2.0, "y": 0.0, "z": 1.0},
    ]
    invalid_path = controller.dispatch(
        "prepare_profile_path_sweep",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "path": broken,
            "frame_strategy": "transport",
        },
        _context(controller, "broken-path"),
    )
    assert invalid_path.data["diagnostic"]["code"] == "profile-transform.invalid-path"
    assert invalid_path.data["diagnostic"]["required_fields"] == ["path"]
    assert invalid_path.data["diagnostic"]["first_failed_member"] == "CD"

    unsupported_frame = controller.dispatch(
        "prepare_profile_path_sweep",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "path": _valid_path(),
            "frame_strategy": "curved",
        },
        _context(controller, "unsupported-frame"),
    )
    assert unsupported_frame.data["diagnostic"]["code"] == "profile-transform.unsupported-frame"
    assert unsupported_frame.data["diagnostic"]["required_fields"] == ["frame_strategy"]

    missing_frame = controller.dispatch(
        "prepare_profile_path_sweep",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "path": _valid_path(),
        },
        _context(controller, "missing-frame"),
    )
    assert missing_frame.data["diagnostic"]["code"] == "profile-transform.unsupported-frame"
    assert missing_frame.data["diagnostic"]["required_fields"] == ["frame_strategy"]

    missing_path = controller.dispatch(
        "prepare_profile_path_sweep",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "frame_strategy": "transport",
        },
        _context(controller, "missing-path"),
    )
    assert missing_path.data["diagnostic"]["code"] == "profile-transform.invalid-path"
    assert missing_path.data["diagnostic"]["required_fields"] == ["path"]

    assert session.snapshot().session_revision == 1


def test_phase5_topology_and_preflight_failures_are_typed_and_atomic(monkeypatch) -> None:
    session = _session(planar_sketch_geometry("Sketch", contours=(SketchCircle("material", 0, 0, 1),)).recipe)
    refresh_count = []
    bridge, controller = _controller(session, lambda: refresh_count.append("refresh"))
    before = session.snapshot()

    monkeypatch.setattr(
        agent_authoring,
        "profile_transform_context",
        lambda *_args, **_kwargs: {
            "dimension": 2,
            "recipe_kind": "SketchGeometry",
            "session_revision": 0,
            "topology_exact": False,
            "material_profile_count": 0,
            "profiles": [],
            "extrusion": {
                "available": False,
                "blocking_code": "profile-transform.topology-unproven",
                "blocking_reason": "topology proof unavailable",
            },
        },
    )
    topology = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "height": 2.0,
        },
        _context(controller, "topology"),
    )
    assert topology.data["diagnostic"]["code"] == "profile-transform.topology-unproven"
    assert session.snapshot() == before

    monkeypatch.setattr(
        agent_authoring,
        "profile_transform_context",
        lambda *_args, **_kwargs: {
            "dimension": 2,
            "recipe_kind": "SketchGeometry",
            "session_revision": 0,
            "topology_exact": True,
            "material_profile_count": 0,
            "profiles": [],
            "extrusion": {
                "available": False,
                "blocking_code": "profile-transform.no-material-profile",
                "blocking_reason": "no material Profile",
            },
        },
    )
    no_material = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "height": 2.0,
        },
        _context(controller, "no-material"),
    )
    assert no_material.data["diagnostic"]["code"] == "profile-transform.no-material-profile"
    assert bridge._records == {}
    assert session.snapshot() == before

    monkeypatch.undo()
    monkeypatch.setattr(
        agent_authoring,
        "_preflight_profile_extrusions",
        lambda _recipes: (_ for _ in ()).throw(
            RuntimeError("OCC/Gmsh preflight failed at C:\\private\\model.brep")
        ),
    )
    context = controller.dispatch(
        "read_profile_transform_context",
        {"part_id": "P1"},
        _context(controller, "read-before-preflight"),
    )
    source = context.data["profiles"][0]["face_id"]
    failed = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": [source],
            "context_revision": before.session_revision,
            "height": 2.0,
        },
        _context(controller, "preflight"),
    )
    assert failed.data["diagnostic"]["code"] == "profile-transform.preflight-failed"
    assert "C:\\private" not in failed.data["diagnostic"]["message"]
    assert bridge._records == {}
    assert session.snapshot() == before
    assert refresh_count == []


def test_phase5_unexpected_body_count_diagnostic_preserves_draft(monkeypatch) -> None:
    session = _session(planar_sketch_geometry("Sketch", contours=(SketchCircle("material", 0, 0, 1),)).recipe)
    _bridge, controller = _controller(session)
    monkeypatch.setattr(
        agent_authoring,
        "_preflight_derived_geometry",
        lambda _recipe: (_ for _ in ()).throw(
            AuthoringContractError(
                "profile-transform.unexpected-body-count: two volumes"
            )
        ),
    )
    result = controller.dispatch(
        "prepare_profile_revolution",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "axis": "z",
            "angle_degrees": 180.0,
        },
        _context(controller, "body-count"),
    )
    diagnostic = result.data["diagnostic"]
    assert diagnostic["code"] == "profile-transform.unexpected-body-count"
    assert diagnostic["preserve_draft"] is True


@pytest.mark.parametrize("dependency_error", [KeyError("dependency"), AttributeError("dependency")])
def test_phase5_programming_errors_keep_generic_tool_contract(
    monkeypatch,
    dependency_error,
) -> None:
    session = _session(planar_sketch_geometry("Sketch", contours=(SketchCircle("material", 0, 0, 1),)).recipe)
    refresh_count = []
    bridge, controller = _controller(session, lambda: refresh_count.append("refresh"))
    before = session.snapshot()

    def fail(*_args, **_kwargs):
        raise dependency_error

    monkeypatch.setattr(agent_authoring, "profile_transform_context", fail)
    result = controller.dispatch(
        "read_profile_transform_context",
        {"part_id": "P1"},
        _context(controller, "programming-error"),
    )
    assert not result.ok
    assert result.data is None
    assert result.diagnostics[0].code == "INVALID_TOOL_ARGUMENTS"
    assert bridge._records == {}
    assert refresh_count == []
    assert session.snapshot() == before
