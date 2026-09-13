from __future__ import annotations

from pathlib import Path
import time

from PySide6.QtWidgets import QApplication

from fem.io import save_result_archive
from fem_gui.main_window import FEMMainWindow
from fem_gui.task_controller import BackgroundTaskState
from tests.helpers.result_archives import make_result_archive
from tests.helpers.result_field_fixtures import (
    make_continuum_nodal_semantics_result,
)


def wait_for_result_idle(window: FEMMainWindow, timeout_ms: int = 2000) -> None:
    app = QApplication.instance()
    deadline = time.monotonic() + timeout_ms / 1000.0
    open_controller = window.workspace.open_controller
    while (
        (
            window.busy
            or (
                open_controller is not None
                and open_controller.busy
            )
        )
        and time.monotonic() < deadline
    ):
        app.processEvents()
        time.sleep(0.001)
    app.processEvents()
    if window.busy or (
        open_controller is not None and open_controller.busy
    ):
        print(
            "Result task timeout",
            window.task_controller.state,
            window.task_controller.current_task_name,
            None
            if open_controller is None
            else open_controller.state,
            flush=True,
        )
        raise AssertionError(
            f"background task did not settle: {window.task_controller.state!r} "
            f"{window.task_controller.current_task_name!r}"
        )


def result_projection_identity(window: FEMMainWindow) -> tuple[object, ...]:
    document = window.document
    catalog = window.result_tree.catalog
    payload = window.viewport._result_render_payload
    file_state = document.result_file_state
    return (
        document.session_id,
        document.session_revision,
        document.source_kind,
        document.source_path,
        id(catalog),
        None if catalog is None else catalog.source,
        window.viewport.artifact_id,
        id(payload),
        None
        if payload is None
        else (
            payload.topology.source,
            payload.topology.materialization_generation,
            payload.topology.selection,
        ),
        document.result_path,
        document.result_dirty,
        document.unsaved_result_count,
        file_state,
    )


def open_result_archive_window(tmp_path: Path, run_name: str = "archive") -> FEMMainWindow:
    archive = make_result_archive(make_continuum_nodal_semantics_result, run_name)
    source = tmp_path / f"{run_name}.femres"
    save_result_archive(source, archive)
    window = FEMMainWindow()
    receipt = window.open_result_path(source)
    assert receipt.completion is not None
    assert receipt.completion.result(2.0).state is BackgroundTaskState.SUCCEEDED
    wait_for_result_idle(window)
    return window
