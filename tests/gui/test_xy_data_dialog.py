from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
import pytest

from fem.results import (
    ResultProbeKind,
    ResultProbeTarget,
    ResultXYRequest,
    ResultXYSeries,
    build_result_xy_series,
    build_result_provider,
)
from fem_gui.xy_data_dialog import XYDataDialog
from tests.application.results.test_xy_data import _provider


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_xy_data_dialog_emits_typed_request_and_displays_series() -> None:
    _application()
    provider = _provider()
    selection = provider.catalog().default_selection
    assert selection is not None
    dialog = XYDataDialog(
        provider.catalog(),
        frame_catalog=provider.frame_catalog,
        current_selection=selection,
        node_ids=provider.snapshot.topology.node_ids,
        element_ids=provider.snapshot.topology.element_ids,
    )
    emitted: list[object] = []
    dialog.generateRequested.connect(emitted.append)

    dialog.target_combo.setCurrentIndex(0)
    dialog.target_id_spin.setValue(2)
    dialog.request_curve()

    assert len(emitted) == 1
    request = emitted[0]
    assert isinstance(request, ResultXYRequest)
    assert request.target == ResultProbeTarget(
        ResultProbeKind.NODE,
        node_id=2,
    )

    series = build_result_xy_series(
        tuple(
            (frame_index, provider.frame_provider(frame_index))
            for frame_index in provider.frame_indices
        ),
        request,
        frame_catalog=provider.frame_catalog,
    )
    assert isinstance(series, ResultXYSeries)
    dialog.set_series(series)
    assert dialog.table.rowCount() == 2
    assert dialog.chart.title().startswith(request.selection.component)
    dialog.close()


def test_xy_data_dialog_rejects_a_target_incompatible_with_field_position() -> None:
    _application()
    provider = _provider()
    selection = provider.catalog().default_selection
    assert selection is not None
    dialog = XYDataDialog(
        provider.catalog(),
        frame_catalog=provider.frame_catalog,
        current_selection=selection,
        node_ids=provider.snapshot.topology.node_ids,
        element_ids=provider.snapshot.topology.element_ids,
    )
    dialog.target_combo.setCurrentIndex(1)
    with pytest.raises(ValueError, match="目标"):
        dialog.current_request()
    dialog.close()
