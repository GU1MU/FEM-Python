from __future__ import annotations

import gc
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from threading import Thread
from typing import get_overloads
import weakref

import pytest

from fem.application import NativePart, SessionDelta, UNSET
from fem.application.results import (
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
)
from fem.geometry import LogicalEntityRef, RectangleGeometry
from fem.mesh.settings import LocalMeshControl, MeshSettings
from fem_gui.commands import (
    CloseSessionCommand,
    GuiCommandCompletion,
    GuiCommandDiagnostic,
    GuiCommandOutcome,
    GuiCommandReceipt,
    GuiCommandStatus,
    MeshInputEdit,
    NativeGeometryEdit,
    NewNativeProjectCommand,
    ResultCsvExportSpec,
    ResultVtkExportSpec,
)
from fem_gui.task_controller import (
    BackgroundTaskState,
    TaskApplyStatus,
    TaskCompletion,
)


def _accepted_delta() -> SessionDelta:
    return SessionDelta(session_revision=4)


def _result_source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="result-1",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=4,
        step_name="Step-1",
        run_id="run-1",
    )


def _selection() -> ScalarFieldSelection:
    return ScalarFieldSelection(
        FieldMaterializationKey(
            FieldRequest(
                ResultFieldId(
                    ResultVariable.U,
                    FieldPosition.NODE,
                )
            ),
            recovery_contract=2,
        ),
        "U1",
    )


def _result_outcome(
    *,
    output_path: Path | None = None,
    record_count: int | None = 8,
) -> GuiCommandOutcome:
    return GuiCommandOutcome(
        output_path=output_path,
        source=_result_source(),
        materialization_generation=3,
        selection=_selection(),
        record_count=record_count,
        diagnostic_summary=" exported with stable warning ",
    )


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


def test_command_outcome_is_strict_sanitized_and_deeply_owned() -> None:
    source = _result_source()
    selection = _selection()
    outcome = GuiCommandOutcome(
        output_path=Path("exports/result.csv"),
        source=source,
        materialization_generation=5,
        selection=selection,
        record_count=0,
        diagnostic_summary="  first line\n second line  ",
    )

    assert outcome.output_path == Path("exports/result.csv")
    assert outcome.source == source
    assert outcome.source is not source
    assert outcome.selection == selection
    assert outcome.selection is not selection
    assert outcome.selection.field_key is not selection.field_key
    assert outcome.materialization_generation == 5
    assert outcome.record_count == 0
    assert outcome.diagnostic_summary == "first line second line"
    assert not hasattr(outcome, "__dict__")
    assert {field.name for field in fields(GuiCommandOutcome)} == {
        "output_path",
        "source",
        "materialization_generation",
        "selection",
        "record_count",
        "diagnostic_summary",
    }
    with pytest.raises(FrozenInstanceError):
        outcome.record_count = 1  # type: ignore[misc]

    object.__setattr__(source, "result_id", "mutated-result")
    object.__setattr__(selection, "component", "U2")
    assert outcome.source.result_id == "result-1"
    assert outcome.selection.component == "U1"


def test_command_outcome_allows_only_coherent_optional_shapes() -> None:
    source = _result_source()
    selection = _selection()

    assert GuiCommandOutcome(output_path=Path("report.txt")).source is None
    assert (
        GuiCommandOutcome(
            source=source,
            materialization_generation=0,
            selection=selection,
        ).record_count
        is None
    )

    for kwargs in (
        {},
        {"diagnostic_summary": "warning only"},
        {"source": source},
        {
            "source": source,
            "materialization_generation": 1,
        },
        {
            "materialization_generation": 1,
            "selection": selection,
        },
        {"record_count": 1},
    ):
        with pytest.raises(ValueError):
            GuiCommandOutcome(**kwargs)

    with pytest.raises(TypeError, match="output_path"):
        GuiCommandOutcome(output_path="result.csv")
    with pytest.raises(ValueError, match="identify a file"):
        GuiCommandOutcome(output_path=Path("."))
    with pytest.raises(TypeError, match="source"):
        GuiCommandOutcome(
            source=object(),
            materialization_generation=1,
            selection=selection,
        )
    with pytest.raises(TypeError, match="selection"):
        GuiCommandOutcome(
            source=source,
            materialization_generation=1,
            selection=object(),
        )
    with pytest.raises(TypeError, match="record_count"):
        GuiCommandOutcome(
            source=source,
            materialization_generation=1,
            selection=selection,
            record_count=True,
        )
    with pytest.raises(TypeError, match="record_count"):
        GuiCommandOutcome(
            source=source,
            materialization_generation=1,
            selection=selection,
            record_count=1.5,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="record_count"):
        GuiCommandOutcome(
            source=source,
            materialization_generation=1,
            selection=selection,
            record_count=-1,
        )
    with pytest.raises(TypeError, match="diagnostic_summary"):
        GuiCommandOutcome(
            output_path=Path("result.csv"),
            diagnostic_summary=None,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="control"):
        GuiCommandOutcome(
            output_path=Path("result.csv"),
            diagnostic_summary="bad\x00summary",
        )
    with pytest.raises(TypeError):
        GuiCommandOutcome(  # type: ignore[call-arg]
            output_path=Path("result.csv"),
            provider=object(),
        )


def test_result_export_specs_are_complete_detached_and_typed() -> None:
    source = _result_source()
    selection = _selection()

    csv_spec = ResultCsvExportSpec(source, 6, selection)
    second_selection = ScalarFieldSelection(selection.field_key, "U2")
    multi_csv_spec = ResultCsvExportSpec(
        source,
        6,
        (selection, second_selection),
    )
    vtk_spec = ResultVtkExportSpec(source, 6, selection, 2)

    assert csv_spec.source == vtk_spec.source == source
    assert csv_spec.source is not source
    assert vtk_spec.source is not source
    assert csv_spec.selection == vtk_spec.selection == selection
    assert multi_csv_spec.selections == (selection, second_selection)
    assert multi_csv_spec.selection == selection
    assert all(
        owned is not original
        for owned, original in zip(
            multi_csv_spec.selections,
            (selection, second_selection),
            strict=True,
        )
    )
    assert csv_spec.selection is not selection
    assert vtk_spec.selection is not selection
    assert vtk_spec.deformation_scale == 2.0
    with pytest.raises(FrozenInstanceError):
        csv_spec.materialization_generation = 7  # type: ignore[misc]

    object.__setattr__(source, "run_id", "mutated-run")
    object.__setattr__(selection, "component", "U2")
    assert csv_spec.source.run_id == "run-1"
    assert vtk_spec.source.run_id == "run-1"
    assert csv_spec.selection.component == "U1"
    assert vtk_spec.selection.component == "U1"


@pytest.mark.parametrize(
    ("generation", "expected_error"),
    (
        (True, TypeError),
        (1.5, TypeError),
        (-1, ValueError),
    ),
)
def test_result_export_specs_reject_invalid_generation(
    generation: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error, match="materialization_generation"):
        GuiCommandOutcome(
            source=_result_source(),
            materialization_generation=generation,  # type: ignore[arg-type]
            selection=_selection(),
        )
    with pytest.raises(expected_error, match="materialization_generation"):
        ResultCsvExportSpec(
            _result_source(),
            generation,  # type: ignore[arg-type]
            _selection(),
        )
    with pytest.raises(expected_error, match="materialization_generation"):
        ResultVtkExportSpec(
            _result_source(),
            generation,  # type: ignore[arg-type]
            _selection(),
            0.0,
        )


@pytest.mark.parametrize(
    "scale",
    (True, "1", float("nan"), float("inf"), float("-inf")),
)
def test_vtk_export_spec_requires_finite_deformation_scale(
    scale: object,
) -> None:
    expected_error = TypeError if scale is True or scale == "1" else ValueError
    with pytest.raises(expected_error, match="deformation_scale"):
        ResultVtkExportSpec(
            _result_source(),
            1,
            _selection(),
            scale,  # type: ignore[arg-type]
        )


def test_result_export_specs_reject_untyped_source_and_selection() -> None:
    with pytest.raises(TypeError):
        ResultVtkExportSpec(  # type: ignore[call-arg]
            _result_source(),
            1,
            _selection(),
        )
    with pytest.raises(TypeError, match="source"):
        ResultCsvExportSpec(
            object(),  # type: ignore[arg-type]
            1,
            _selection(),
        )
    with pytest.raises(TypeError, match="selection"):
        ResultCsvExportSpec(
            _result_source(),
            1,
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="selection"):
        ResultVtkExportSpec(
            _result_source(),
            1,
            object(),  # type: ignore[arg-type]
            0.0,
        )


def test_receipt_factories_enforce_one_status_payload_shape() -> None:
    accepted = GuiCommandReceipt.accepted(1, _accepted_delta())
    outcome = _result_outcome(output_path=Path("result.csv"))
    accepted_outcome = GuiCommandReceipt.accepted(4, outcome=outcome)
    diagnostic = GuiCommandDiagnostic(
        "busy",
        "Another task is running.",
        "Wait for it to finish.",
    )
    rejected = GuiCommandReceipt.rejected(2, diagnostic)
    completion = GuiCommandCompletion(3)
    pending = GuiCommandReceipt.pending(3, completion)

    assert len(get_overloads(GuiCommandReceipt.accepted)) == 2
    assert accepted.status is GuiCommandStatus.ACCEPTED
    assert accepted.delta == _accepted_delta()
    assert accepted.outcome is None
    assert accepted_outcome.status is GuiCommandStatus.ACCEPTED
    assert accepted_outcome.delta is None
    assert accepted_outcome.outcome == outcome
    assert accepted_outcome.outcome is not outcome
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
        (
            GuiCommandStatus.ACCEPTED,
            {
                "delta": _accepted_delta(),
                "outcome": _result_outcome(),
            },
        ),
        (GuiCommandStatus.REJECTED, {}),
        (
            GuiCommandStatus.REJECTED,
            {
                "diagnostic": GuiCommandDiagnostic("failed", "failed"),
                "outcome": _result_outcome(),
            },
        ),
        (GuiCommandStatus.PENDING, {}),
        (
            GuiCommandStatus.PENDING,
            {"completion": GuiCommandCompletion(99)},
        ),
        (
            GuiCommandStatus.PENDING,
            {
                "completion": GuiCommandCompletion(5),
                "outcome": _result_outcome(),
            },
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
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert terminal.state is BackgroundTaskState.SUCCEEDED
    assert terminal.value is None
    assert handle.outcome is None
    assert handle.wait(0.0) is terminal


def test_pending_completion_installs_typed_value_outcome_independently() -> None:
    handle = GuiCommandCompletion(8)
    receipt = GuiCommandReceipt.pending(8, handle)
    outcome = _result_outcome(output_path=Path("result.vtk"))

    assert receipt.completion is handle
    assert handle.outcome is None
    assert handle.complete(_terminal(value=outcome))

    assert handle.terminal is not None
    assert handle.terminal.value is None
    assert handle.outcome == outcome
    assert handle.outcome is not outcome
    assert receipt.completion.outcome == outcome


def test_completion_accepts_explicit_outcome_and_drops_worker_payload() -> None:
    handle = GuiCommandCompletion(9)
    payload = {"provider": object(), "field_data": object()}
    outcome = _result_outcome(record_count=3)

    assert handle.complete(
        _terminal(value=payload),
        outcome=outcome,
    )

    assert handle.terminal is not None
    assert handle.terminal.value is None
    assert handle.outcome == outcome
    assert handle.outcome is not outcome


def test_completion_rejects_ambiguous_or_invalid_outcome_sources() -> None:
    outcome = _result_outcome()

    with pytest.raises(TypeError, match="GuiCommandOutcome"):
        GuiCommandCompletion(1).complete(
            _terminal(),
            outcome=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="one explicit source"):
        GuiCommandCompletion(1).complete(
            _terminal(value=outcome),
            outcome=outcome,
        )
    failed = TaskCompletion(
        task_id=7,
        task_name="query",
        state=BackgroundTaskState.FAILED,
        message="failed",
        value=outcome,
    )
    with pytest.raises(ValueError, match="succeeded"):
        GuiCommandCompletion(1).complete(failed)


def test_completed_handle_rejects_every_later_terminal_as_ordered_noop() -> None:
    handle = GuiCommandCompletion(1)
    terminal = _terminal(value=_result_outcome())

    assert handle.complete(terminal)
    assert not handle.complete(terminal, outcome=object())  # type: ignore[arg-type]
    assert handle.outcome == _result_outcome()


def test_completion_does_not_own_forbidden_worker_objects() -> None:
    class ForbiddenWorkerPayload:
        pass

    payload = ForbiddenWorkerPayload()
    payload.provider = object()
    payload.model = object()
    payload.field_data = object()
    payload.session = object()
    payload.qt_object = object()
    payload.pyvista_object = object()
    reference = weakref.ref(payload)
    completion = _terminal(value=payload)
    handle = GuiCommandCompletion(1)

    assert handle.complete(completion)
    del completion
    del payload
    gc.collect()

    assert reference() is None
    assert handle.terminal is not None
    assert handle.terminal.value is None
    assert handle.outcome is None


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
    with pytest.raises(TypeError, match="GuiCommandOutcome"):
        GuiCommandReceipt(
            1,
            GuiCommandStatus.ACCEPTED,
            outcome=object(),
        )
