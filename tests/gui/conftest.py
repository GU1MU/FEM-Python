from __future__ import annotations

import gc
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FEM_GUI_OFFSCREEN", "1")

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMessageBox,
)

from tests.helpers.file_builders import write_inp


_GUI_TEST_ROOT = Path(__file__).resolve().parent
_gui_cleanup_count = 0
_NATIVE_GUI_TEST_FILES = frozenset(
    {
        "test_agent_composite_geometry_phase4.py",
        "test_agent_exact_boolean_phase4.py",
        "test_agent_profile_sweep_phase3.py",
        "test_agent_profile_transform_baseline_phase0.py",
        "test_agent_profile_transform_phase6.py",
        "test_preprocessing_workflow.py",
        "test_scope_selection.py",
    }
)


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Keep native geometry/rendering integration out of routine GUI tests."""

    if any(
        os.environ.get(name) == "1"
        for name in ("FEM_RUN_GUI_NATIVE", "FEM_RUN_NATIVE_TESTS")
    ):
        return
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        path = Path(str(item.path)).resolve()
        native_gui_test = (
            path.parent == _GUI_TEST_ROOT
            and (
                path.name in _NATIVE_GUI_TEST_FILES
                or item.get_closest_marker("gmsh") is not None
                or item.get_closest_marker("gui_native") is not None
            )
        )
        (deselected if native_gui_test else selected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture(autouse=True)
def reject_unstubbed_modal_dialogs(monkeypatch):
    """Turn accidental native modal loops into immediate test failures."""

    def blocked(*_args, **_kwargs):
        pytest.fail("GUI tests must stub modal dialogs before opening them")

    monkeypatch.setattr(QDialog, "exec", blocked)
    monkeypatch.setattr(QMessageBox, "exec", blocked)
    for method in (
        "critical",
        "information",
        "question",
        "warning",
    ):
        monkeypatch.setattr(QMessageBox, method, blocked)
    for method in ("getOpenFileName", "getSaveFileName"):
        monkeypatch.setattr(QFileDialog, method, blocked)
    for method in ("getDouble", "getInt", "getItem", "getText"):
        monkeypatch.setattr(QInputDialog, method, blocked)

    yield

    global _gui_cleanup_count

    application = QApplication.instance()
    if application is None:
        return
    application.processEvents()
    widgets = tuple(
        widget
        for widget in application.topLevelWidgets()
        if widget.parentWidget() is None
    )
    for widget in widgets:
        viewport = getattr(widget, "viewport", None)
        shutdown_backend = getattr(viewport, "shutdown_backend", None)
        if callable(shutdown_backend):
            shutdown_backend()
        widget.hide()
        widget.deleteLater()
    application.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    _gui_cleanup_count += 1
    gc.collect(0)
    if _gui_cleanup_count % 50 == 0:
        gc.collect()


@pytest.fixture
def gui_inp_path(tmp_path):
    """生成可稳定求解的小型平面应力验收模型。"""
    return write_inp(
        tmp_path,
        "gui_plate.inp",
        [
            "*Heading",
            "GUI plate",
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "3, 1., 1.",
            "4, 0., 1.",
            "*Element, type=CPS4, elset=SOLID",
            "1, 1, 2, 3, 4",
            "*Nset, nset=LEFT",
            "1, 4",
            "*Nset, nset=RIGHT",
            "2, 3",
            "*Elset, elset=SOLID",
            "1",
            "*Material, name=STEEL",
            "*Elastic",
            "210000., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*Boundary",
            "LEFT, 1, 2, 0.",
            "*Step, name=Static-1",
            "*Static",
            "*Cload",
            "RIGHT, 1, 10.",
            "*Output, field",
            "*Node Output",
            "U, RF",
            "*Element Output",
            "S",
            "*End Step",
        ],
    )
