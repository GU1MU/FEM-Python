from __future__ import annotations

from collections.abc import Callable

import pytest

from fem.geometry import GeometryStateError
from fem.geometry._gmsh.state import (
    _GEOMETRY_MUTATION_STATES,
    _MESH_CONTROL_STATES,
    _QUERY_STATES,
    _ModelStateMachine,
    _State,
)


def test_state_names_and_operation_sets_are_exact() -> None:
    assert tuple(state.name for state in _State) == (
        "NEW",
        "BUILDING_GEOMETRY",
        "CONFIGURING_MESH",
        "MESHED",
        "MESH_FAILED",
        "CLOSED",
    )
    assert _QUERY_STATES == frozenset(
        {
            _State.BUILDING_GEOMETRY,
            _State.CONFIGURING_MESH,
            _State.MESH_FAILED,
        }
    )
    assert _GEOMETRY_MUTATION_STATES == frozenset({_State.BUILDING_GEOMETRY})
    assert _MESH_CONTROL_STATES == frozenset({_State.CONFIGURING_MESH})


def test_successful_mesh_lifecycle_follows_the_exact_state_graph() -> None:
    states = _ModelStateMachine("part")

    assert states.state is _State.NEW
    states.enter_geometry()
    assert states.state is _State.BUILDING_GEOMETRY
    states.begin_mesh_configuration()
    assert states.state is _State.CONFIGURING_MESH
    states.mark_meshed()
    assert states.state is _State.MESHED
    states.close()
    assert states.state is _State.CLOSED
    states.close()
    assert states.state is _State.CLOSED


def test_failed_mesh_lifecycle_follows_the_exact_state_graph() -> None:
    states = _ModelStateMachine("part")

    states.enter_geometry()
    states.begin_mesh_configuration()
    states.mark_mesh_failed()
    assert states.state is _State.MESH_FAILED
    states.close()
    assert states.state is _State.CLOSED


@pytest.mark.parametrize(
    "advance",
    [
        lambda states: None,
        lambda states: states.enter_geometry(),
        lambda states: (
            states.enter_geometry(),
            states.begin_mesh_configuration(),
        ),
        lambda states: (
            states.enter_geometry(),
            states.begin_mesh_configuration(),
            states.mark_meshed(),
        ),
        lambda states: (
            states.enter_geometry(),
            states.begin_mesh_configuration(),
            states.mark_mesh_failed(),
        ),
    ],
    ids=["new", "building", "configuring", "meshed", "mesh-failed"],
)
def test_close_is_legal_from_every_nonclosed_state(
    advance: Callable[[_ModelStateMachine], object],
) -> None:
    states = _ModelStateMachine("part")
    advance(states)

    states.close()

    assert states.state is _State.CLOSED


def test_repeat_entry_preserves_the_existing_contextual_error() -> None:
    states = _ModelStateMachine("part")
    states.enter_geometry()

    with pytest.raises(
        GeometryStateError,
        match=(
            "^geometry model 'part': context entry failed because model context "
            "is not new$"
        ),
    ):
        states.enter_geometry()

    assert states.state is _State.BUILDING_GEOMETRY


def test_illegal_transition_preserves_state_and_sorted_allowed_names() -> None:
    states = _ModelStateMachine("part")

    with pytest.raises(
        GeometryStateError,
        match=(
            "^geometry model 'part': Mesher binding failed because state NEW does "
            "not permit this operation \\(expected BUILDING_GEOMETRY\\)$"
        ),
    ):
        states.begin_mesh_configuration()

    assert states.state is _State.NEW


def test_allowed_operation_checks_match_each_lifecycle_phase() -> None:
    states = _ModelStateMachine("part")
    states.enter_geometry()
    states.check("entities", _QUERY_STATES)
    states.check("rectangle", _GEOMETRY_MUTATION_STATES)

    states.begin_mesh_configuration()
    states.check("entities", _QUERY_STATES)
    states.check("transfinite_curve", _MESH_CONTROL_STATES)

    states.mark_mesh_failed()
    states.check("entities", _QUERY_STATES)
    with pytest.raises(
        GeometryStateError,
        match=(
            "^geometry model 'part': entities failed because state MESH_FAILED "
            "does not permit this operation \\(expected BUILDING_GEOMETRY, "
            "CONFIGURING_MESH\\)$"
        ),
    ):
        states.check(
            "entities",
            frozenset(
                {_State.CONFIGURING_MESH, _State.BUILDING_GEOMETRY}
            ),
        )


@pytest.mark.parametrize("terminal", ["meshed", "failed"])
def test_mesh_terminal_states_reject_further_mesh_transitions(terminal: str) -> None:
    states = _ModelStateMachine("part")
    states.enter_geometry()
    states.begin_mesh_configuration()
    if terminal == "meshed":
        states.mark_meshed()
        expected = _State.MESHED
    else:
        states.mark_mesh_failed()
        expected = _State.MESH_FAILED

    with pytest.raises(GeometryStateError, match=expected.name):
        states.mark_meshed()
    with pytest.raises(GeometryStateError, match=expected.name):
        states.mark_mesh_failed()

    assert states.state is expected


def test_error_factory_preserves_the_existing_context() -> None:
    states = _ModelStateMachine("part")

    error = states.error("entity", "facade-owned Gmsh model is missing")

    assert type(error) is GeometryStateError
    assert str(error) == (
        "geometry model 'part': entity failed because "
        "facade-owned Gmsh model is missing"
    )
