from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.application.results import (
    build_solve_result_bundle,
    restore_result_provider,
)
from fem.io.result_csv import read_result_csv
from fem.io.result_vtk import read_result_vtk
from fem.solvers.static_linear import solve
from fem_gui.commands import (
    ResultCsvExportSpec,
    ResultVtkExportSpec,
)
from fem_gui import main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from fem_gui.visualization.model_adapter import build_model_geometry
from tests.helpers.gui_command_receipts import (
    await_succeeded,
    require_rejected,
)
from tests.helpers.model_builders import make_static_pull_truss_model
from tests.helpers.preflight_builders import passing_preflight_report


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _solved_window() -> FEMMainWindow:
    _application()
    window = FEMMainWindow()
    model = make_static_pull_truss_model()
    imported = window.session.prepare_import(Path("result-export.inp"))
    delta = window.session.accept_imported_model(imported.token, model)
    assert window._apply_session_delta(
        delta,
        model_geometry=build_model_geometry(model),
        source_label="result-export.inp",
    )

    validation = window.session.prepare_validation("pull")
    assert window._apply_session_delta(
        window.session.accept_validation(
            validation.token,
            passing_preflight_report(validation.token),
        )
    )
    task = window.session.prepare_solve("pull", "Job-1")
    assert task.delta is not None
    assert window._apply_session_delta(task.delta)
    assert window._apply_session_delta(window.session.begin_run(task.token))
    result = solve(task.model, task.step_name, name=task.run_name)
    assert window._apply_session_delta(
        window.session.accept_run_succeeded(
            task.token,
            build_solve_result_bundle(task, result),
        )
    )
    return window


def _specs(
    window: FEMMainWindow,
) -> tuple[ResultCsvExportSpec, ResultVtkExportSpec]:
    record = window.session.current_result()
    assert record is not None
    provider = restore_result_provider(
        record.result,
        record.materialization,
    )
    selection = provider.catalog().default_selection
    return (
        ResultCsvExportSpec(
            provider.source,
            provider.snapshot.generation,
            selection,
        ),
        ResultVtkExportSpec(
            provider.source,
            provider.snapshot.generation,
            selection,
            1.25,
        ),
    )


def test_public_result_exports_use_canonical_snapshot_bound_writers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = _solved_window()
    csv_spec, vtk_spec = _specs(window)
    revision = window.document.session_revision

    csv_receipt = window.export_result_csv(
        tmp_path / "result.csv",
        csv_spec,
    )
    await_succeeded(csv_receipt)
    assert csv_receipt.completion is not None
    csv_outcome = csv_receipt.completion.outcome
    assert csv_outcome is not None
    assert csv_outcome.output_path == tmp_path / "result.csv"
    assert csv_outcome.source == csv_spec.source
    assert csv_outcome.selection == csv_spec.selection
    assert csv_outcome.materialization_generation == (
        csv_spec.materialization_generation
    )
    assert csv_outcome.record_count == 2
    csv_readback = read_result_csv(tmp_path / "result.csv")
    assert csv_readback.source == csv_spec.source
    assert csv_readback.selection == csv_spec.selection

    monkeypatch.setattr(
        main_window_module,
        "write_result_csv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("VTK export must not round-trip through CSV")
        ),
    )
    vtk_receipt = window.export_result_vtk(
        tmp_path / "result.vtk",
        vtk_spec,
    )
    await_succeeded(vtk_receipt)
    assert vtk_receipt.completion is not None
    vtk_outcome = vtk_receipt.completion.outcome
    assert vtk_outcome is not None
    assert vtk_outcome.output_path == tmp_path / "result.vtk"
    assert vtk_outcome.record_count == 2
    vtk_readback = read_result_vtk(tmp_path / "result.vtk")
    assert vtk_readback.source == vtk_spec.source
    assert vtk_readback.selection == vtk_spec.selection
    assert vtk_readback.deformation_scale == 1.25

    assert window.document.session_revision == revision
    window.close()


def test_result_export_commands_reject_untyped_stale_and_wrong_suffix(
    tmp_path: Path,
) -> None:
    window = _solved_window()
    csv_spec, vtk_spec = _specs(window)

    require_rejected(
        window.export_result_csv(
            tmp_path / "result.csv",
            object(),  # type: ignore[arg-type]
        ),
        code="command.type.invalid",
    )
    require_rejected(
        window.export_result_csv(
            tmp_path / "result.txt",
            csv_spec,
        ),
        code="result.csv_export.rejected",
    )
    require_rejected(
        window.export_result_vtk(
            tmp_path / "result.vtu",
            vtk_spec,
        ),
        code="result.vtk_export.rejected",
    )
    require_rejected(
        window.export_result_csv(
            tmp_path / "stale.csv",
            ResultCsvExportSpec(
                csv_spec.source,
                csv_spec.materialization_generation + 1,
                csv_spec.selection,
            ),
        ),
        code="result.csv_export.rejected",
    )

    assert not (tmp_path / "result.txt").exists()
    assert not (tmp_path / "result.vtu").exists()
    assert not (tmp_path / "stale.csv").exists()
    window.close()
