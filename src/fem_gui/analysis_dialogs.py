"""内存分析作业的创建与管理窗口。"""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from fem.application import AnalysisRun, RunStatus

from .dialogs import configure_form_layout


_STATUS_LABELS = {
    RunStatus.PENDING: "等待中",
    RunStatus.RUNNING: "运行中",
    RunStatus.SUCCEEDED: "已完成",
    RunStatus.FAILED: "失败",
    RunStatus.CANCELLED: "已取消",
}


class JobSubmitDialog(QDialog):
    """收集一个最小的线性静力作业名称和分析步。"""

    def __init__(
        self,
        default_name: str,
        step_names: Iterable[str],
        current_step: str | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("jobSubmitDialog")
        self.setWindowTitle("创建分析作业")
        self.setMinimumWidth(390)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        configure_form_layout(form)
        self.name_edit = QLineEdit(default_name, self)
        self.name_edit.setMaxLength(64)
        self.step_combo = QComboBox(self)
        for name in step_names:
            self.step_combo.addItem(str(name), str(name))
        index = self.step_combo.findData(current_step)
        self.step_combo.setCurrentIndex(index if index >= 0 else 0)
        solver_type = QLabel("线性静力", self)
        form.addRow("作业名称", self.name_edit)
        form.addRow("分析步", self.step_combo)
        form.addRow("求解类型", solver_type)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("提交")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def job_name(self) -> str:
        return self.name_edit.text().strip()

    @property
    def step_name(self) -> str:
        return str(self.step_combo.currentData() or "")


class JobManagerDialog(QDialog):
    """显示会话作业、选中作业日志及历史结果操作。"""

    terminateRequested = Signal(str)
    openResultRequested = Signal(str)

    def __init__(self, jobs: Iterable[AnalysisRun], parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("jobManagerDialog")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("作业管理器")
        self.resize(680, 450)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 5, self)
        self.table.setObjectName("jobTable")
        self.table.setHorizontalHeaderLabels(("作业名称", "分析步", "状态", "开始时间", "耗时"))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._update_selection)
        layout.addWidget(self.table, 1)
        layout.addWidget(QLabel("日志", self))
        self.log_view = QPlainTextEdit(self)
        self.log_view.setObjectName("jobLogView")
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(125)
        self._displayed_job_name: str | None = None
        layout.addWidget(self.log_view)
        buttons = QHBoxLayout()
        self.terminate_button = QPushButton("终止求解", self)
        self.open_result_button = QPushButton("打开结果", self)
        close = QPushButton("关闭", self)
        self.terminate_button.clicked.connect(self._emit_terminate)
        self.open_result_button.clicked.connect(self._emit_open_result)
        close.clicked.connect(self.close)
        buttons.addWidget(self.terminate_button)
        buttons.addWidget(self.open_result_button)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh(jobs)

    def refresh(self, jobs: Iterable[AnalysisRun] | None = None) -> None:
        """刷新表格并保留原选择。"""
        if jobs is not None:
            self._jobs = list(jobs)
        elif not hasattr(self, "_jobs"):
            self._jobs = []
        selected = self.selected_job_name()
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._jobs))
        for row, job in enumerate(self._jobs):
            entries = (
                job.name,
                job.step_name,
                (
                    "终止中"
                    if job.cancellation_requested
                    and job.status is RunStatus.RUNNING
                    else _STATUS_LABELS.get(job.status, str(job.status.value))
                ),
                job.started_at.strftime("%H:%M:%S") if job.started_at else "—",
                _elapsed_text(job),
            )
            for column, text in enumerate(entries):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setData(Qt.ItemDataRole.UserRole, job.name)
                self.table.setItem(row, column, item)
            if job.name == selected:
                self.table.selectRow(row)
        self.table.blockSignals(False)
        if self.table.currentRow() < 0 and self._jobs:
            self.table.selectRow(0)
        self._update_selection()

    def selected_job_name(self) -> str | None:
        item = self.table.item(self.table.currentRow(), 0)
        return None if item is None else str(item.data(Qt.ItemDataRole.UserRole))

    def _selected_job(self) -> AnalysisRun | None:
        name = self.selected_job_name()
        return next((job for job in getattr(self, "_jobs", []) if job.name == name), None)

    def _update_selection(self) -> None:
        job = self._selected_job()
        job_name = job.name if job else None
        log_text = "\n".join(job.messages) if job else "尚无作业记录"
        if job_name != self._displayed_job_name or log_text != self.log_view.toPlainText():
            same_job = job_name == self._displayed_job_name
            scroll_bar = self.log_view.verticalScrollBar()
            scroll_value = scroll_bar.value()
            was_at_bottom = scroll_value == scroll_bar.maximum()
            self.log_view.setPlainText(log_text)
            if same_job:
                scroll_bar.setValue(scroll_bar.maximum() if was_at_bottom else scroll_value)
            self._displayed_job_name = job_name
        self.terminate_button.setEnabled(
            job is not None
            and job.status is RunStatus.RUNNING
            and not job.cancellation_requested
        )
        self.open_result_button.setEnabled(job is not None and job.has_result)

    def _emit_terminate(self) -> None:
        name = self.selected_job_name()
        if name is not None:
            self.terminateRequested.emit(name)

    def _emit_open_result(self) -> None:
        name = self.selected_job_name()
        if name is not None:
            self.openResultRequested.emit(name)


def _elapsed_text(job: AnalysisRun) -> str:
    elapsed = job.elapsed_seconds
    return "—" if elapsed is None else f"{elapsed:.2f} s"
