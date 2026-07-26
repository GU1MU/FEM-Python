from __future__ import annotations

from threading import Thread

import pytest

from fem.application import NativePart, SessionDelta, UNSET
from fem.geometry import LogicalEntityRef, RectangleGeometry
from fem.mesh.settings import LocalMeshControl, MeshSettings
from fem_gui.commands import (
    CloseSessionCommand,
    GuiCommandCompletion,
    GuiCommandDiagnostic,
    GuiCommandReceipt,
    GuiCommandStatus,
    MeshInputEdit,
    NativeGeometryEdit,
    NewNativeProjectCommand,
)
from fem_gui.task_controller import (
    BackgroundTaskState,
    TaskApplyStatus,
    TaskCompletion,
)


def _accepted_delta() -> SessionDelta:
    return SessionDelta(session_revision=4)


def _terminal(
    *,
    task_id: int = 7,
    value: object = None,
) -> TaskCompletion:
    return TaskCompletion(
        task_id=task_id,
        task_name="mesh",
        state=BackgroundTaskState.SUCCEEDED,
        apply_status=TaskApplyStatus.ACCEPTED,
        value=value,
    )


def test_command_dtos_normalize_and_validate_lifecycle_inputs() -> None:
    assert NewNativeProjectCommand("  Beam  ", 3) == NewNativeProjectCommand(
        "Beam",
        3,
    )
    assert CloseSessionCommand(0).expected_session_revision == 0

    with pytest.raises(ValueError, match="project name"):
        NewNativeProjectCommand(" ")
    with pytest.raises(ValueError, match="non-negative"):
        CloseSessionCommand(True)


def test_native_geometry_edit_is_typed_detached_and_revision_bound() -> None:
    parts = [NativePart(" Part ", " Body ")]
    recipe = RectangleGeometry("Plate", 2.0, 1.0)

    edit = NativeGeometryEdit(6, parts, recipe)
    parts.clear()

    assert edit.base_session_revision == 6
    assert edit.parts == (NativePart("Part", "Body"),)
    assert edit.recipe == recipe
    assert edit.mesh_settings is UNSET

    with pytest.raises(TypeError, match="NativePart"):
        NativeGeometryEdit(6, ("Part-1",), recipe)
    with pytest.raises(ValueError, match="at least one part"):
        NativeGeometryEdit(6, (), recipe)
    with pytest.raises(TypeError, match="NativeGeometry"):
        NativeGeometryEdit(6, (NativePart(),), object())


def test_native_geometry_clear_has_explicit_empty_parts() -> None:
    edit = NativeGeometryEdit(
        base_session_revision=2,
        parts=(),
        recipe=None,
        mesh_settings=None,
    )

    assert edit.recipe is None
    assert edit.mesh_settings is None

    with pytest.raises(ValueError, match="must not retain parts"):
        NativeGeometryEdit(2, (NativePart(),), None)


def test_mesh_input_edit_covers_global_shape_order_and_local_controls() -> None:
    settings = MeshSettings(
        size=0.5,
        order=2,
        cell_shape="quadrilateral",
        local_controls=(
            LocalMeshControl(
                LogicalEntityRef("edge:0"),
                0.2,
            ),
        ),
    )

    edit = MeshInputEdit(9, settings)

    assert edit.settings == settings
    assert edit.settings is not settings
    assert edit.settings.order == 2
    assert edit.settings.cell_shape == "quadrilateral"
    assert len(edit.settings.local_controls) == 1

    with pytest.raises(TypeError, match="MeshSettings"):
        MeshInputEdit(9, object())


def test_receipt_factories_enforce_one_status_payload_shape() -> None:
    accepted = GuiCommandReceipt.accepted(1, _accepted_delta())
    diagnostic = GuiCommandDiagnostic(
        "busy",
        "Another task is running.",
        "Wait for it to finish.",
    )
    rejected = GuiCommandReceipt.rejected(2, diagnostic)
    completion = GuiCommandCompletion(3)
    pending = GuiCommandReceipt.pending(3, completion)

    assert accepted.status is GuiCommandStatus.ACCEPTED
    assert accepted.delta == _accepted_delta()
    assert rejected.status is GuiCommandStatus.REJECTED
    assert rejected.diagnostic == diagnostic
    assert pending.status is GuiCommandStatus.PENDING
    assert pending.completion is completion


@pytest.mark.parametrize(
    ("status", "kwargs"),
    [
        (GuiCommandStatus.ACCEPTED, {}),
        (
            GuiCommandStatus.ACCEPTED,
            {"delta": SessionDelta(0, accepted=False)},
        ),
        (GuiCommandStatus.REJECTED, {}),
        (GuiCommandStatus.PENDING, {}),
        (
            GuiCommandStatus.PENDING,
            {"completion": GuiCommandCompletion(99)},
        ),
    ],
)
def test_receipt_rejects_mismatched_status_payloads(
    status: GuiCommandStatus,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        GuiCommandReceipt(command_id=5, status=status, **kwargs)


def test_completion_observes_once_before_and_after_terminal() -> None:
    handle = GuiCommandCompletion(3, task_id=7)
    observed: list[TaskCompletion] = []
    late: list[TaskCompletion] = []

    handle.observe(observed.append)
    assert handle.complete(_terminal())
    assert not handle.complete(_terminal())
    handle.observe(late.append)

    assert handle.done
    assert handle.task_id == 7
    assert observed == late == [handle.terminal]


def test_completion_waits_offline_and_drops_worker_payload_ownership() -> None:
    handle = GuiCommandCompletion(1)
    payload = {"model": object(), "result": object()}
    worker = Thread(target=lambda: handle.complete(_terminal(value=payload)))
    worker.start()

    terminal = handle.result(timeout=1.0)
    worker.join()

    assert terminal.state is BackgroundTaskState.SUCCEEDED
    assert terminal.value is None
    assert handle.wait(0.0) is terminal


def test_completion_timeout_and_task_binding_fail_closed() -> None:
    handle = GuiCommandCompletion(1)

    assert handle.wait(0.0) is None
    with pytest.raises(TimeoutError):
        handle.result(0.0)
    handle.bind_task_id(4)
    with pytest.raises(RuntimeError, match="another task"):
        handle.bind_task_id(5)
    with pytest.raises(ValueError, match="does not match"):
        handle.complete(_terminal(task_id=5))


def test_command_boundary_rejects_bool_ids_and_untyped_values() -> None:
    with pytest.raises(ValueError, match="command_id"):
        GuiCommandCompletion(True)
    with pytest.raises(TypeError, match="TaskCompletion"):
        GuiCommandCompletion(1).complete(object())
    with pytest.raises(TypeError, match="GuiCommandStatus"):
        GuiCommandReceipt(1, "accepted", delta=_accepted_delta())
    with pytest.raises(TypeError, match="GuiCommandDiagnostic"):
        GuiCommandReceipt(1, GuiCommandStatus.REJECTED, diagnostic="busy")
