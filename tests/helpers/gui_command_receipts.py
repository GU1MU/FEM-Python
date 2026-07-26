"""Assertions for public GUI command receipts used by workflow tests."""

from __future__ import annotations

from time import monotonic

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from fem_gui.commands import GuiCommandReceipt, GuiCommandStatus
from fem_gui.task_controller import BackgroundTaskState, TaskCompletion


def require_accepted(receipt: GuiCommandReceipt) -> GuiCommandReceipt:
    """Require one synchronous command to return exactly one accepted payload."""

    assert type(receipt) is GuiCommandReceipt
    assert receipt.status is GuiCommandStatus.ACCEPTED
    assert (receipt.delta is None) != (receipt.outcome is None)
    if receipt.delta is not None:
        assert receipt.delta.accepted
    assert receipt.diagnostic is None
    assert receipt.completion is None
    return receipt


def require_rejected(
    receipt: GuiCommandReceipt,
    *,
    code: str,
) -> GuiCommandReceipt:
    """Require one synchronous command to reject with a typed diagnostic."""

    assert type(receipt) is GuiCommandReceipt
    assert receipt.status is GuiCommandStatus.REJECTED
    assert receipt.delta is None
    assert receipt.outcome is None
    assert receipt.diagnostic is not None
    assert receipt.diagnostic.code == code
    assert receipt.completion is None
    return receipt


def await_succeeded(
    receipt: GuiCommandReceipt,
    *,
    timeout: float = 30.0,
) -> TaskCompletion:
    """Drive Qt until one pending public command reaches ``SUCCEEDED``."""

    assert type(receipt) is GuiCommandReceipt
    assert receipt.status is GuiCommandStatus.PENDING
    assert receipt.delta is None
    assert receipt.outcome is None
    assert receipt.diagnostic is None
    completion = receipt.completion
    assert completion is not None

    application = QApplication.instance() or QApplication([])
    deadline = monotonic() + timeout
    while not completion.done and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()

    terminal = completion.result(0.0)
    assert terminal.state is BackgroundTaskState.SUCCEEDED
    return terminal
