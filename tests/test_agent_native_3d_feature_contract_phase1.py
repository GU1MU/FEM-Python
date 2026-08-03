from __future__ import annotations

from copy import deepcopy

import pytest

from fem.application.definitions import NativePart
from fem.application.feature_history import derive_feature_history
from fem.application.session import ProjectSnapshot
from fem.geometry import (
    BooleanBodyContext,
    BooleanGeometry,
    BoxGeometry,
    ExtrudedGeometry,
    MultiBodyGeometry,
    MovedGeometry,
    PathSweptGeometry,
    RectangleGeometry,
    RevolvedGeometry,
    SolidBody,
    WireGeometry,
    WireMember,
    WirePoint,
    describe_recipe_topology,
    is_single_solid_recipe,
)
from fem.geometry.recipe_topology import topology_fingerprint_for_recipe
from fem.io.project import decode_project, encode_project
from fem_agent.geometry_authoring import (
    create_geometry_proposal,
    feature_topology_catalog,
    geometry_contract_proof,
    geometry_draft,
    geometry_recipe_from_payload,
    geometry_recipe_to_payload,
)
from fem_agent.authoring import AuthoringContext, LocalModelBinding, UnitContextSummary


def _profile() -> tuple[RectangleGeometry, str]:
    sketch = RectangleGeometry("Profile", 2.0, 1.0)
    face_id = next(
        item.logical_id
        for item in describe_recipe_topology(sketch).entities
        if item.kind == "face"
    )
    return sketch, face_id


def _path() -> WireGeometry:
    return WireGeometry(
        "Path",
        (
            WirePoint("A", 0.0, 0.0, 0.0),
            WirePoint("B", 0.0, 0.0, 2.0),
            WirePoint("C", 1.0, 0.0, 3.0),
        ),
        (WireMember("AB", "A", "B"), WireMember("BC", "B", "C")),
    )


def _recipes() -> tuple[object, ...]:
    sketch, face_id = _profile()
    extrusion = ExtrudedGeometry(sketch, 3.0, (face_id,))
    revolve = RevolvedGeometry(sketch, "x", 180.0, (face_id,))
    path_sweep = PathSweptGeometry(
        sketch, _path(), (face_id,), "transport"
    )
    boolean = BooleanGeometry(
        "Joined",
        "fuse",
        BoxGeometry("Target", 2.0, 2.0, 2.0),
        MovedGeometry(BoxGeometry("Tool", 1.0, 1.0, 1.0), 1.0, 0.0, 0.0),
    )
    multi_body = MultiBodyGeometry(
        "Assembly",
        (
            SolidBody("B1", "Extrusion", extrusion),
            SolidBody("B2", "Revolve", revolve),
        ),
    )
    return extrusion, revolve, path_sweep, boolean, multi_body


def test_phase1_boolean_body_context_payload_round_trip() -> None:
    recipe = BooleanGeometry(
        "Joined", "fuse", BoxGeometry("Target", 2.0, 2.0, 2.0),
        MovedGeometry(BoxGeometry("Tool", 1.0, 1.0, 1.0), 1.0, 0.0, 0.0),
        body_context=BooleanBodyContext("BF1", "B1", "B2", "Tool"),
    )

    assert geometry_recipe_from_payload(geometry_recipe_to_payload(recipe)) == recipe


@pytest.mark.parametrize("recipe", _recipes())
def test_phase1_versioned_3d_payload_round_trip_is_strict(recipe: object) -> None:
    payload = geometry_recipe_to_payload(recipe)

    assert payload["schema_version"] == 1
    assert geometry_recipe_from_payload(payload) == recipe

    unknown = deepcopy(payload)
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        geometry_recipe_from_payload(unknown)


def test_phase1_payload_rejects_unknown_type_version_and_budgets() -> None:
    with pytest.raises(ValueError, match="fields"):
        geometry_recipe_from_payload({"schema_version": 1, "kind": "loft"})
    with pytest.raises(ValueError, match="schema_version"):
        geometry_recipe_from_payload(
            {"schema_version": 2, "kind": "box", "name": "B", "width": 1,
             "depth": 1, "height": 1}
        )
    nested: dict[str, object] = {
        "kind": "box", "name": "B", "width": 1, "depth": 1, "height": 1
    }
    for _ in range(17):
        nested = {"kind": "translated", "base": nested, "dx": 0, "dy": 0, "dz": 0}
    with pytest.raises(ValueError, match="depth budget"):
        geometry_recipe_from_payload({"schema_version": 1, **nested})
    oversized = {
        "schema_version": 1,
        "kind": "wire",
        "name": "W",
        "points": [
            {"name": f"P{i}", "x": float(i), "y": 0.0, "z": 0.0}
            for i in range(129)
        ],
        "members": [{"name": "M", "start": "P0", "end": "P1"}],
    }
    with pytest.raises(ValueError, match="bounded|budget"):
        geometry_recipe_from_payload(oversized)


def test_phase1_legacy_a2_payload_without_version_remains_readable() -> None:
    legacy = {"kind": "rectangle", "name": "Legacy", "width": 2.0, "height": 1.0}

    assert geometry_recipe_from_payload(legacy) == RectangleGeometry(
        "Legacy", 2.0, 1.0
    )


def test_phase1_proposal_summary_exposes_feature_contract_and_invalidation() -> None:
    recipe = _recipes()[0]
    context = AuthoringContext(
        binding=LocalModelBinding("doc", "session", 0, "blank", True),
        model_name=None,
        active_part_id=None,
    )
    units = UnitContextSummary("mm", "N", "MPa")

    proposal = create_geometry_proposal(
        proposal_id="phase1-summary",
        agent_session_id="agent",
        turn_id="turn",
        source_tool_call_ids=("call",),
        context=context,
        draft_revision=1,
        draft=geometry_draft(recipe),
        part_function="拉伸体",
        project_function="三维项目",
        unit_context=units,
    )

    summary = proposal.display_summary
    assert summary["source"] == list(recipe.source_face_ids)
    assert summary["feature_operation"] == "extrude"
    assert summary["key_dimensions"] == {"height": 3.0}
    assert summary["expected_entity_count"] == 1
    assert summary["invalidation_impact"] == {
        "mesh": False, "definitions": False, "results": False,
    }
    assert summary["proof"]["kind"] == "local_recipe_topology_proof"


@pytest.mark.parametrize("recipe", _recipes())
def test_phase1_catalog_proof_and_project_round_trip_preserve_identity(
    recipe: object,
) -> None:
    before_catalog = feature_topology_catalog(recipe, part_id="P1")
    before_fingerprint = topology_fingerprint_for_recipe(recipe)
    proof = geometry_contract_proof(recipe)
    if isinstance(recipe, PathSweptGeometry):
        restored = geometry_recipe_from_payload(geometry_recipe_to_payload(recipe))
        assert type(restored) is PathSweptGeometry
        assert topology_fingerprint_for_recipe(restored) == before_fingerprint
        assert feature_topology_catalog(restored, part_id="P1") == before_catalog
        assert proof.exact
        assert proof.expected_body_count == 1
        assert is_single_solid_recipe(restored)
        return
    snapshot = ProjectSnapshot(
        source_kind="native",
        parts=(NativePart(),),
        geometry_recipe=recipe,
        feature_history=derive_feature_history(recipe),
    )

    reopened = decode_project(encode_project(snapshot)).snapshot

    if isinstance(recipe, MultiBodyGeometry):
        assert before_catalog["canonical_part_ownership"] == [
            {"body_id": "B1", "part_id": "P1"},
            {"body_id": "B2", "part_id": "P2"},
        ]
        assert len(reopened.parts) == len(recipe.bodies)
        for body, part in zip(recipe.bodies, reopened.parts, strict=True):
            assert str(part.id) == f"P{body.id[1:]}"
            assert part.name == body.name
            assert part.geometry_recipe == body.recipe
            assert topology_fingerprint_for_recipe(
                part.geometry_recipe
            ) == topology_fingerprint_for_recipe(body.recipe)
        assert str(reopened.active_part_id) == "P1"
    else:
        assert reopened.geometry_recipe == recipe
        assert reopened.feature_history == derive_feature_history(recipe)
        assert topology_fingerprint_for_recipe(reopened.geometry_recipe) == before_fingerprint
        assert feature_topology_catalog(reopened.geometry_recipe, part_id="P1") == before_catalog
    assert proof.topology_fingerprint["entities"]
    if isinstance(recipe, (PathSweptGeometry, BooleanGeometry)):
        assert not proof.exact
        assert proof.expected_body_count == 0
        assert proof.diagnostics
    else:
        assert proof.exact
        assert proof.expected_body_count >= 1


def test_phase3_proves_path_body_while_unproven_boolean_never_guesses() -> None:
    path_sweep = _recipes()[2]
    boolean = _recipes()[3]

    path_proof = geometry_contract_proof(path_sweep)
    assert path_proof.to_dict()["body_count_proven"] is True
    assert path_proof.expected_body_count == 1
    assert feature_topology_catalog(path_sweep)["exact"]
    assert is_single_solid_recipe(path_sweep)

    boolean_proof = geometry_contract_proof(boolean)
    assert boolean_proof.to_dict()["body_count_proven"] is False
    assert boolean_proof.expected_body_count == 0
    assert not feature_topology_catalog(boolean)["exact"]


def test_phase1_feature_catalog_rejects_an_unbounded_topology() -> None:
    points = tuple(
        WirePoint(f"P{index}", float(index), 0.0, 0.0)
        for index in range(65)
    )
    members = tuple(
        WireMember(f"M{index}", f"P{index}", f"P{index + 1}")
        for index in range(64)
    )

    with pytest.raises(ValueError, match="bounded contract"):
        feature_topology_catalog(WireGeometry("Long", points, members))
