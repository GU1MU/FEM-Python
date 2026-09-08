from __future__ import annotations

import pytest

from fem_agent.authoring import AuthoringContext, LocalModelBinding
from fem_agent.authoring_runtime import (
    AuthoringToolOutcome,
    AuthoringWorkflowController,
    provider_safe_authoring_payload,
)

from tests.helpers.agent_authoring_workflow_fixtures import (
    _context,
    _requirements_for,
    _dispatch,
)


def test_blank_project_seeds_default_units_without_a_clarification_gate() -> None:
    context = AuthoringContext(
        binding=LocalModelBinding(
            "document:blank-a8",
            "blank-a8",
            0,
            "blank",
            True,
        ),
        model_name=None,
        active_part_id=None,
    )
    controller = AuthoringWorkflowController(
        lambda: context,
        {
            "prepare_geometry_proposal": lambda _arguments, _controller: (
                AuthoringToolOutcome(
                    "Geometry proposal registered.",
                    {"state": "pending_confirmation"},
                )
            ),
        },
    )

    controller.observe_binding(context)

    assert controller.collected_requirements("geometry") == {
        "length_unit": "mm",
        "force_unit": "N",
        "stress_unit": "MPa",
    }
    assert controller.defaulted_requirement_keys("geometry") == (
        "length_unit",
        "force_unit",
        "stress_unit",
    )
    assert "prepare_geometry_proposal" in {
        item.name for item in controller.definitions
    }
    context_result = _dispatch(
        controller,
        "read_authoring_context",
        {},
        20,
    )
    assert context_result.data["missing_requirements"] == []
    assert context_result.data["defaulted_requirements"] == [
        "length_unit",
        "force_unit",
        "stress_unit",
    ]

    overridden = _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-explicit-units",
            "requirements": {
                "length_unit": "m",
                "force_unit": "kN",
                "stress_unit": "GPa",
            },
        },
        21,
    )
    assert overridden.ok
    assert controller.collected_requirements("geometry") == {
        "length_unit": "m",
        "force_unit": "kN",
        "stress_unit": "GPa",
    }
    assert controller.defaulted_requirement_keys("geometry") == ()


def test_requirement_batch_validation_is_atomic() -> None:
    controller = AuthoringWorkflowController(lambda: _context(), {})
    before_revision = controller.ledger.revision
    before_entries = controller.ledger.entries

    rejected = _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-invalid",
            "requirements": {
                "length_unit": "mm",
                "force_unit": 1,
            },
        },
        7,
    )

    assert rejected.ok is False
    assert controller.ledger.revision == before_revision
    assert controller.ledger.entries == before_entries


def test_provider_payload_rejects_paths_bulk_arrays_and_unsafe_summaries() -> None:
    for payload in (
        {"message": "local file D:\\private\\model.femproj"},
        {"node_ids": [1, 2]},
        {"coordinates": [[0.0, 0.0]]},
        {"model_patch": {"schema_version": "1.0"}},
    ):
        with pytest.raises(ValueError):
            provider_safe_authoring_payload(payload)

    def unsafe(_arguments, _controller):
        raise ValueError("failed at D:\\private\\secret.femproj")

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {"prepare_geometry_proposal": unsafe},
    )
    assert _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-geometry",
            "requirements": _requirements_for("geometry"),
        },
        9,
    ).ok
    result = _dispatch(controller, "prepare_geometry_proposal", {}, 10)
    serialized = result.to_json()
    assert "D:" not in serialized
    assert "private" not in serialized
    assert "ValueError" in serialized
