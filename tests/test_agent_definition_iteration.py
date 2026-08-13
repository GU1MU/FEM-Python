from __future__ import annotations

import json

import pytest

from fem.application import (
    MeshEntityRef,
    ModelSession,
    NamedRegion,
    RegionAssignment,
    RevisionConflictError,
    ScopedDefinitionBatch,
    SectionDefinition,
)
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    EdgeLoad,
    MaterialDefinition,
    OutputRequest,
    OutputSourceEvidence,
)
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from fem_agent.editing_authoring import (
    apply_edit_operation,
    create_edit_patch,
    editable_object_catalog,
)
from fem_gui.agent_authoring import authoring_context_from_snapshot
from tests.test_agent_authoring_phase_a5 import _session as _base_session


STEP = "分析步-静力"


def _session() -> ModelSession:
    session = _base_session()
    snapshot = session.snapshot()
    session.apply_scoped_definition_batch(
        ScopedDefinitionBatch(
            snapshot.session_revision,
            tuple(snapshot.named_regions.values())
            + (
                NamedRegion(
                    "域-备用",
                    (MeshEntityRef.element(1, part_id="P1"),),
                ),
            ),
            tuple(snapshot.materials)
            + (MaterialDefinition("材料-铝", {"E": 70000.0, "nu": 0.33}),),
            tuple(snapshot.sections)
            + (SectionDefinition("截面-备用", "材料-铝"),),
            tuple(snapshot.assignments),
            (
                AnalysisStep(
                    STEP,
                    boundaries=(
                        DisplacementConstraint(
                            "边-固定端", 1, 2, 0.0, "edge", "位移-固定端"
                        ),
                    ),
                    edge_loads=(
                        EdgeLoad(
                            "边-加载端",
                            (10.0, 0.0),
                            None,
                            "traction",
                            "载荷-拉伸",
                        ),
                    ),
                    outputs=(
                        OutputRequest(
                            "field",
                            "node",
                            ("U", "RF"),
                            {"frequency": 1},
                            OutputSourceEvidence("native"),
                            "结果请求-节点",
                        ),
                    ),
                    metadata={"nlgeom": False},
                ),
            ),
        )
    )
    return session


def _patch(
    session: ModelSession,
    object_type: str,
    target_id: str,
    changes: dict[str, object],
    *,
    step_name: str | None = None,
):
    snapshot = session.snapshot()
    return create_edit_patch(
        patch_id=f"patch-{object_type}-{snapshot.session_revision}",
        agent_session_id="agent-definition-iteration",
        turn_id="turn-definition-iteration",
        source_tool_call_ids=("call-definition-iteration",),
        context=authoring_context_from_snapshot(snapshot),
        snapshot=snapshot,
        draft_revision=1,
        object_type=object_type,
        target_id=target_id,
        changes=changes,
        step_name=step_name,
    )[0]


def _apply(session: ModelSession, patch) -> None:
    apply_edit_operation(
        session,
        patch.operations[0],
        base_session_revision=patch.base_session_revision,
    )


def test_catalog_exposes_all_definition_edit_types_with_bounded_details() -> None:
    catalog = editable_object_catalog(_session().snapshot(), limit=128)
    by_type = {item.object_type: item for item in catalog}

    assert {
        "named_region",
        "material",
        "section",
        "section_assignment",
        "analysis_step",
        "boundary_condition",
        "load",
        "result_request",
    }.issubset(by_type)
    assert by_type["material"].details["editable_fields"] == [
        "new_name",
        "properties",
    ]
    assert by_type["section_assignment"].target_id == "域-板体"
    assert by_type["result_request"].step_name == STEP
    payload = [item.to_provider_dict() for item in catalog]
    assert len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) < 32768
    assert "artifact" not in json.dumps(payload, ensure_ascii=False).casefold()


def test_material_and_section_renames_cascade_atomically() -> None:
    session = _session()
    material = _patch(
        session,
        "material",
        "材料-钢",
        {
            "new_name": "材料-结构钢",
            "properties": {"E": 200000.0, "density": 7.85e-9},
        },
    )
    _apply(session, material)
    after_material = session.snapshot()
    assert after_material.materials[0].name == "材料-结构钢"
    assert after_material.materials[0].properties["density"] == 7.85e-9
    assert after_material.materials[0].properties["nu"] == 0.3
    assert after_material.sections[0].material == "材料-结构钢"

    section = _patch(
        session,
        "section",
        "截面-平面应力",
        {
            "new_name": "截面-板",
            "material": "材料-结构钢",
            "section_type": "solid",
            "properties": {"plane_type": "stress", "thickness": 2.0},
        },
    )
    _apply(session, section)
    after_section = session.snapshot()
    assert after_section.sections[0].name == "截面-板"
    assert after_section.assignments[0].section_name == "截面-板"


def test_assignment_step_and_result_request_edits_preserve_child_state() -> None:
    session = _session()
    assignment = _patch(
        session,
        "section_assignment",
        "域-板体",
        {"region_name": "域-备用", "section_name": "截面-备用"},
    )
    _apply(session, assignment)
    assert session.snapshot().assignments == (
        RegionAssignment("截面-备用", "域-备用"),
    )

    before_step = session.snapshot().steps[0]
    step_patch = _patch(
        session,
        "analysis_step",
        STEP,
        {
            "new_name": "分析步-复算",
            "procedure": "static",
            "metadata": {"increments": 20},
        },
    )
    _apply(session, step_patch)
    edited_step = session.snapshot().steps[0]
    assert edited_step.name == "分析步-复算"
    assert edited_step.boundaries == before_step.boundaries
    assert edited_step.edge_loads == before_step.edge_loads
    assert edited_step.outputs == before_step.outputs
    assert edited_step.metadata == {"nlgeom": False, "increments": 20}

    evidence = edited_step.outputs[0].source_evidence
    output_patch = _patch(
        session,
        "result_request",
        "结果请求-节点",
        {
            "new_name": "结果请求-应力",
            "output_kind": "field",
            "target": "element",
            "variables": ["S"],
            "metadata": {"position": "centroid"},
        },
        step_name="分析步-复算",
    )
    _apply(session, output_patch)
    output = session.snapshot().steps[0].outputs[0]
    assert output.name == "结果请求-应力"
    assert output.target == "element"
    assert output.variables == ("S",)
    assert output.metadata == {"frequency": 1, "position": "centroid"}
    assert output.source_evidence == evidence


@pytest.mark.parametrize(
    ("object_type", "target_id", "changes", "step_name", "message"),
    (
        ("material", "材料-钢", {"new_name": "材料-铝"}, None, "unique"),
        (
            "section",
            "截面-平面应力",
            {"material": "材料-不存在"},
            None,
            "unavailable",
        ),
        (
            "section_assignment",
            "域-板体",
            {"region_name": "域-不存在"},
            None,
            "unavailable",
        ),
        ("analysis_step", STEP, {"procedure": "dynamic"}, None, "unsupported"),
        (
            "result_request",
            "结果请求-节点",
            {"target": "node", "variables": ["S"]},
            STEP,
            "do not match",
        ),
        ("material", "材料-钢", {"properties": {"E": 210000.0, "nu": 0.3}}, None, "do not modify"),
    ),
)
def test_invalid_definition_edits_fail_atomically(
    object_type: str,
    target_id: str,
    changes: dict[str, object],
    step_name: str | None,
    message: str,
) -> None:
    session = _session()
    before = session.snapshot()
    if object_type == "material" and changes == {"new_name": "材料-铝"}:
        patch = _patch(session, object_type, target_id, changes)
        with pytest.raises(ValueError, match=message):
            _apply(session, patch)
    else:
        with pytest.raises(ValueError, match=message):
            _patch(
                session,
                object_type,
                target_id,
                changes,
                step_name=step_name,
            )
    assert session.snapshot() == before


def test_stale_definition_edit_fails_without_partial_mutation() -> None:
    session = _session()
    patch = _patch(
        session,
        "material",
        "材料-钢",
        {"properties": {"E": 205000.0, "nu": 0.3}},
    )
    session.rename_native_model(
        "模型-迭代",
        expected_session_revision=session.session_revision,
    )
    before = session.snapshot()

    with pytest.raises(RevisionConflictError):
        _apply(session, patch)

    assert session.snapshot() == before


def test_definition_patch_reports_history_retention_and_preflight_reset() -> None:
    session = _session()
    patch = _patch(
        session,
        "material",
        "材料-钢",
        {"properties": {"E": 205000.0, "nu": 0.3}},
    )

    assert patch.invalidation_impact == {
        "model": True,
        "validation": True,
        "results": False,
        "historical_results_retained": True,
        "current_validation_reset": True,
        "current_result_display_reset": True,
    }
    assert AuthoringWorkflowStage.PREFLIGHT_READY.value == "preflight_ready"
