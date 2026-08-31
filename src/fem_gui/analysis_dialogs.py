"""内存分析作业的创建与管理窗口。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
)

from fem.application import (
    AnalysisRun,
    AttemptRecord,
    AttemptStatus,
    RunDiagnosticsSnapshot,
    RunMessage,
    RunStatus,
)
from fem.model import StaticFormulation

from .dialogs import configure_form_layout
from .analysis_presentation import analysis_formulation_label


_STATUS_LABELS = {
    RunStatus.PENDING: "已创建",
    RunStatus.RUNNING: "运行中",
    RunStatus.SUCCEEDED: "已完成",
    RunStatus.FAILED: "失败",
    RunStatus.CANCELLED: "已取消",
}


class JobSubmitDialog(QDialog):
    """收集作业名称、分析步和该步解析出的求解类型。"""

    def __init__(
        self,
        default_name: str,
        step_formulations: Mapping[str, StaticFormulation | str],
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
        if not isinstance(step_formulations, Mapping):
            raise TypeError("step_formulations must be a mapping")
        labels: dict[str, str] = {}
        for name, formulation in step_formulations.items():
            if type(name) is not str or not name.strip():
                raise ValueError("analysis step names must be nonblank strings")
            if isinstance(formulation, StaticFormulation):
                labels[name] = analysis_formulation_label(formulation)
            elif isinstance(formulation, str) and formulation.strip():
                labels[name] = formulation.strip()
            else:
                raise TypeError(
                    "step descriptions must be StaticFormulation values or nonblank strings"
                )
            self.step_combo.addItem(name, name)
        self._step_formulations = labels
        index = self.step_combo.findData(current_step)
        self.step_combo.setCurrentIndex(index if index >= 0 else 0)
        self.solver_type = QLabel(self)
        self.solver_type.setObjectName("solverTypeLabel")
        form.addRow("作业名称：", self.name_edit)
        form.addRow("分析步：", self.step_combo)
        form.addRow("求解类型：", self.solver_type)
        self.step_combo.currentIndexChanged.connect(self._refresh_solver_type)
        self._refresh_solver_type()
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("创建")
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

    def _refresh_solver_type(self) -> None:
        formulation = self._step_formulations.get(self.step_name)
        self.solver_type.setText(
            "—"
            if formulation is None
            else formulation
        )


class JobManagerDialog(QDialog):
    """Abaqus-style list of submitted jobs and their main actions."""

    submitRequested = Signal(str)
    terminateRequested = Signal(str)
    openResultRequested = Signal(str)
    monitorRequested = Signal(str)
    createRequested = Signal()
    copyRequested = Signal(str)
    renameRequested = Signal(str)
    deleteRequested = Signal(str)

    def __init__(
        self,
        jobs: Iterable[AnalysisRun],
        parent=None,
        *,
        diagnostics: Mapping[str, RunDiagnosticsSnapshot] | None = None,
        diagnostics_provider: Callable[[], Mapping[str, RunDiagnosticsSnapshot]]
        | None = None,
        model_name_provider: Callable[[], str | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("jobManagerDialog")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("作业管理器")
        self.setMinimumSize(700, 260)
        self.resize(820, 360)
        layout = QVBoxLayout(self)
        body = QHBoxLayout()
        self.table = QTableWidget(0, 4, self)
        self.table.setObjectName("jobTable")
        self.table.setHorizontalHeaderLabels(
            ("名称", "模型", "类型", "状态")
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for column, width in enumerate((120, 140, 90), start=1):
            self.table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.Interactive
            )
            self.table.setColumnWidth(column, width)
        self.table.itemSelectionChanged.connect(self._update_selection)
        body.addWidget(self.table, 1)

        actions = QVBoxLayout()
        actions.setSpacing(6)
        self.submit_button = QPushButton("提交求解", self)
        self.create_button = QPushButton("新建", self)
        self.copy_button = QPushButton("复制", self)
        self.rename_button = QPushButton("重命名", self)
        self.delete_button = QPushButton("删除", self)
        self.monitor_button = QPushButton("监视器", self)
        self.open_result_button = QPushButton("打开结果", self)
        self.terminate_button = QPushButton("终止求解", self)
        self.submit_button.clicked.connect(self._emit_submit)
        self.create_button.clicked.connect(self.createRequested)
        self.copy_button.clicked.connect(self._emit_copy)
        self.rename_button.clicked.connect(self._emit_rename)
        self.delete_button.clicked.connect(self._emit_delete)
        self.monitor_button.clicked.connect(self._emit_monitor)
        self.open_result_button.clicked.connect(self._emit_open_result)
        self.terminate_button.clicked.connect(self._emit_terminate)
        actions.addWidget(self.submit_button)
        actions.addWidget(self.create_button)
        actions.addWidget(self.copy_button)
        actions.addWidget(self.rename_button)
        actions.addWidget(self.delete_button)
        actions.addWidget(self.monitor_button)
        actions.addWidget(self.open_result_button)
        actions.addWidget(self.terminate_button)
        actions.addStretch(1)
        close = QPushButton("关闭", self)
        close.clicked.connect(self.close)
        actions.addWidget(close)
        body.addLayout(actions)
        layout.addLayout(body, 1)

        # Kept as a non-visible compatibility surface for existing callers.
        # Visible logs now belong to JobMonitorDialog, matching Abaqus.
        self.log_view = QPlainTextEdit(self)
        self.log_view.setObjectName("jobLogView")
        self.log_view.setReadOnly(True)
        self.log_view.hide()
        self._displayed_job_name: str | None = None
        self._diagnostics = dict(diagnostics or {})
        self._diagnostics_provider = diagnostics_provider
        self._model_name_provider = model_name_provider
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh(jobs, diagnostics=self._diagnostics)

    def refresh(
        self,
        jobs: Iterable[AnalysisRun] | None = None,
        *,
        diagnostics: Mapping[str, RunDiagnosticsSnapshot] | None = None,
    ) -> None:
        """刷新表格并保留原选择。"""
        if self._diagnostics_provider is not None:
            diagnostics = self._diagnostics_provider()
        if jobs is not None:
            self._jobs = list(jobs)
        elif not hasattr(self, "_jobs"):
            self._jobs = []
        if diagnostics is not None:
            self._diagnostics = dict(diagnostics)
        selected = self.selected_job_name()
        vertical = self.table.verticalScrollBar()
        horizontal = self.table.horizontalScrollBar()
        old_vertical = vertical.value()
        old_horizontal = horizontal.value()
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._jobs))
        for row, job in enumerate(self._jobs):
            diagnostic = self._diagnostics.get(job.run_id)
            entries = (
                job.name,
                (
                    diagnostic.model_name
                    if diagnostic is not None and diagnostic.model_name.strip()
                    else self._model_name()
                ),
                diagnostic.analysis_type if diagnostic else "—",
                (
                    "终止中"
                    if job.cancellation_requested
                    and job.status is RunStatus.RUNNING
                    else _STATUS_LABELS.get(job.status, str(job.status.value))
                ),
            )
            for column, text in enumerate(entries):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setData(Qt.ItemDataRole.UserRole, job.name)
                self.table.setItem(row, column, item)
            if job.name == selected:
                self.table.selectRow(row)
        self.table.blockSignals(False)
        vertical.setValue(min(old_vertical, vertical.maximum()))
        horizontal.setValue(min(old_horizontal, horizontal.maximum()))
        if self.table.currentRow() < 0 and self._jobs:
            self.table.selectRow(0)
        self._update_selection()

    def selected_job_name(self) -> str | None:
        item = self.table.item(self.table.currentRow(), 0)
        return None if item is None else str(item.data(Qt.ItemDataRole.UserRole))

    def _selected_job(self) -> AnalysisRun | None:
        name = self.selected_job_name()
        return next((job for job in getattr(self, "_jobs", []) if job.name == name), None)

    def _model_name(self) -> str:
        if self._model_name_provider is None:
            return "—"
        value = self._model_name_provider()
        return str(value).strip() if value and str(value).strip() else "—"

    def _update_selection(self) -> None:
        job = self._selected_job()
        solve_running = any(
            candidate.status is RunStatus.RUNNING
            for candidate in getattr(self, "_jobs", ())
        )
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
        self.submit_button.setEnabled(
            job is not None
            and job.status is RunStatus.PENDING
            and not job.cancellation_requested
            and not solve_running
        )
        self.terminate_button.setEnabled(
            job is not None
            and job.status is RunStatus.RUNNING
            and not job.cancellation_requested
        )
        terminal = job is not None and job.status is not RunStatus.RUNNING
        self.copy_button.setEnabled(terminal)
        self.rename_button.setEnabled(terminal)
        self.delete_button.setEnabled(terminal)
        self.open_result_button.setEnabled(
            job is not None and job.has_displayable_result
        )
        self.open_result_button.setText(
            "打开部分结果"
            if job is not None and job.has_partial_result
            else "打开结果"
        )
        self.monitor_button.setEnabled(job is not None)

    def _emit_submit(self) -> None:
        name = self.selected_job_name()
        if name is not None:
            self.submitRequested.emit(name)

    def _emit_terminate(self) -> None:
        name = self.selected_job_name()
        if name is not None:
            self.terminateRequested.emit(name)

    def _emit_monitor(self) -> None:
        name = self.selected_job_name()
        if name is not None:
            self.monitorRequested.emit(name)

    def _emit_copy(self) -> None:
        name = self.selected_job_name()
        if name is not None:
            self.copyRequested.emit(name)

    def _emit_rename(self) -> None:
        name = self.selected_job_name()
        if name is not None:
            self.renameRequested.emit(name)

    def _emit_delete(self) -> None:
        name = self.selected_job_name()
        if name is not None:
            self.deleteRequested.emit(name)

    def _emit_open_result(self) -> None:
        name = self.selected_job_name()
        if name is not None:
            self.openResultRequested.emit(name)


class JobMonitorDialog(QDialog):
    """Abaqus-style monitor for increments, attempts and solver messages."""

    cancelRequested = Signal(str)
    rerunRequested = Signal(str)
    openResultRequested = Signal(str)
    resultFrameRequested = Signal(str, int)

    def __init__(
        self,
        job: AnalysisRun,
        snapshot: RunDiagnosticsSnapshot | None = None,
        parent=None,
        *,
        snapshot_provider: Callable[[], RunDiagnosticsSnapshot | None]
        | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("jobMonitorDialog")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setMinimumSize(860, 520)
        self.resize(1040, 640)
        self._job = job
        self._snapshot = snapshot
        self._snapshot_provider = snapshot_provider

        layout = QVBoxLayout(self)
        self.header = QLabel(self)
        self.header.setObjectName("jobMonitorHeader")
        layout.addWidget(self.header)

        self.table = QTableWidget(0, 10, self)
        self.table.setObjectName("monitorIncrementTable")
        self.table.setHorizontalHeaderLabels(
            (
                "分析步",
                "增量",
                "尝试",
                "平衡迭代",
                "总迭代",
                "载荷因子",
                "载荷增量",
                "残差",
                "状态",
                "耗时",
            )
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        for column, width in enumerate(
            (100, 72, 58, 82, 82, 104, 104, 118, 90, 88)
        ):
            self.table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.Interactive
            )
            self.table.setColumnWidth(column, width)
        self._table_mode = "static"
        self.tabs = QTabWidget(self)
        self._tab_views: dict[str, QPlainTextEdit] = {}
        self._tab_texts: dict[str, str] = {}
        for title, key in (
            ("日志", "log"),
            ("错误", "errors"),
            ("警告", "warnings"),
            ("输出", "output"),
            ("数据文件", "data"),
            ("消息文件", "messages"),
            ("状态文件", "status"),
        ):
            view = QPlainTextEdit(self)
            view.setObjectName(f"jobMonitor{key.title()}View")
            view.setReadOnly(True)
            self._tab_views[key] = view
            self.tabs.addTab(view, title)
        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setObjectName("jobMonitorSplitter")
        splitter.addWidget(self.table)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes((390, 220))
        layout.addWidget(splitter, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_button = QPushButton("取消求解", self)
        self.cancel_button.setObjectName("jobMonitorCancelButton")
        self.cancel_button.clicked.connect(
            lambda: self.cancelRequested.emit(self.job_name)
        )
        buttons.addWidget(self.cancel_button)
        self.rerun_button = QPushButton("重新提交", self)
        self.rerun_button.setObjectName("jobMonitorRerunButton")
        self.rerun_button.clicked.connect(
            lambda: self.rerunRequested.emit(self.job_name)
        )
        buttons.addWidget(self.rerun_button)
        self.open_result_button = QPushButton("打开结果", self)
        self.open_result_button.setObjectName("jobMonitorOpenResultButton")
        self.open_result_button.clicked.connect(
            lambda: self.openResultRequested.emit(self.job_name)
        )
        buttons.addWidget(self.open_result_button)
        self.jump_result_button = QPushButton("跳转结果", self)
        self.jump_result_button.setObjectName("jobMonitorJumpResultButton")
        self.jump_result_button.clicked.connect(self._emit_selected_result_frame)
        buttons.addWidget(self.jump_result_button)
        self.close_button = QPushButton("关闭", self)
        self.close_button.clicked.connect(self.close)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)
        self.table.itemSelectionChanged.connect(self._refresh_action_buttons)
        self.table.cellDoubleClicked.connect(
            lambda row, _column: self._emit_result_frame(row)
        )
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh_live)
        self._timer.start()
        self.refresh(job, snapshot)

    def _configure_table(
        self,
        mode: str,
        headers: tuple[str, ...],
        widths: tuple[int, ...],
    ) -> None:
        """Switch the increment table between static and dynamic schemas."""

        if self._table_mode == mode:
            return
        selected_row = self.table.currentRow()
        self.table.clearContents()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        for column, width in enumerate(widths):
            self.table.setColumnWidth(column, width)
        self._table_mode = mode
        if selected_row >= 0:
            self.table.selectRow(min(selected_row, self.table.rowCount() - 1))

    def _refresh_live(self) -> None:
        snapshot = (
            self._snapshot_provider()
            if self._snapshot_provider is not None
            else self._snapshot
        )
        self.refresh(snapshot=snapshot)

    def refresh(
        self,
        job: AnalysisRun | None = None,
        snapshot: RunDiagnosticsSnapshot | None = None,
    ) -> None:
        if job is not None:
            self._job = job
        self._snapshot = snapshot
        job = self._job
        snapshot = self._snapshot
        status = _monitor_status_text(
            snapshot.status if snapshot is not None else job.status.value,
            job,
        )
        self.setWindowTitle(f"{job.name} 监视器")
        header = [f"作业：{job.name}", f"状态：{status}"]
        if snapshot is not None:
            if snapshot.dynamic_increments:
                latest = snapshot.dynamic_increments[-1]
                header.extend(
                    (
                        f"增量：{latest.increment}",
                        f"时间：{_number_text(latest.step_time)}",
                        f"时间增量：{_number_text(latest.time_increment)}",
                    )
                )
            elif snapshot.current_increment is not None:
                header.extend(
                    (
                        f"增量：{snapshot.current_increment}",
                        f"Newton：{snapshot.current_iteration or '—'}",
                        f"残差：{_number_text(snapshot.current_residual)}",
                    )
                )
        self.header.setText("    ".join(header))
        selected_row = self.table.currentRow()
        vertical = self.table.verticalScrollBar()
        horizontal = self.table.horizontalScrollBar()
        old_vertical = vertical.value()
        old_horizontal = horizontal.value()
        dynamic_increments = (
            ()
            if snapshot is None
            else snapshot.dynamic_increments
        )
        if dynamic_increments:
            self._configure_table(
                "dynamic",
                (
                    "分析步",
                    "增量",
                    "尝试",
                    "步时间",
                    "时间增量",
                    "稳定时间步",
                    "Newton迭代",
                    "残差",
                    "动能",
                    "内能",
                    "总能量",
                    "状态",
                    "耗时",
                ),
                (100, 62, 58, 92, 92, 102, 88, 112, 104, 104, 104, 86, 88),
            )
            self._refresh_dynamic_rows(job, dynamic_increments)
            vertical.setValue(min(old_vertical, vertical.maximum()))
            horizontal.setValue(min(old_horizontal, horizontal.maximum()))
            self._refresh_tabs(job, snapshot)
            self._refresh_action_buttons()
            return
        self._configure_table(
            "static",
            (
                "分析步",
                "增量",
                "尝试",
                "平衡迭代",
                "总迭代",
                "载荷因子",
                "载荷增量",
                "残差",
                "状态",
                "耗时",
            ),
            (100, 72, 58, 82, 82, 104, 104, 118, 90, 88),
        )
        attempts = () if snapshot is None else snapshot.attempts
        self.table.setRowCount(len(attempts))
        total_iterations = 0
        for row, record in enumerate(attempts):
            total_iterations += record.iterations
            values = (
                job.step_name,
                str(record.increment),
                _attempt_text(record),
                str(record.iterations),
                str(total_iterations),
                _number_text(
                    record.load_factor
                    if record.load_factor is not None
                    else record.target_load_factor
                ),
                _number_text(
                    record.target_load_factor - record.previous_load_factor
                ),
                _number_text(record.residual_norm),
                _attempt_status_text(record.status),
                _seconds_text(record.duration_seconds),
            )
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)
        if 0 <= selected_row < len(attempts):
            self.table.selectRow(selected_row)
        vertical.setValue(min(old_vertical, vertical.maximum()))
        horizontal.setValue(min(old_horizontal, horizontal.maximum()))
        self._refresh_tabs(job, snapshot)
        self._refresh_action_buttons()

    def _refresh_dynamic_rows(
        self,
        job: AnalysisRun,
        records: tuple,
    ) -> None:
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = (
                job.step_name,
                str(record.increment),
                str(record.attempt),
                _number_text(record.step_time),
                _number_text(record.time_increment),
                _number_text(record.stable_time_increment),
                "—" if record.iterations is None else str(record.iterations),
                _number_text(record.residual_norm),
                _number_text(record.kinetic_energy),
                _number_text(record.internal_energy),
                _number_text(record.total_energy),
                _dynamic_status_text(record.status),
                _seconds_text(record.duration_seconds),
            )
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)

    @property
    def job_name(self) -> str:
        return self._job.name

    @property
    def job_id(self) -> str:
        return self._job.run_id

    def _selected_increment(self) -> int | None:
        """Return the solver increment represented by the selected row."""

        row = self.table.currentRow()
        if row < 0 or self._snapshot is None:
            return None
        if self._table_mode == "dynamic":
            records = self._snapshot.dynamic_increments
        else:
            records = self._snapshot.attempts
        if row < 0 or row >= len(records):
            return None
        value = getattr(records[row], "increment", None)
        return value if type(value) is int and value > 0 else None

    def _emit_result_frame(self, row: int | None = None) -> None:
        if row is not None:
            self.table.selectRow(int(row))
        increment = self._selected_increment()
        if increment is not None and self._job.has_displayable_result:
            self.resultFrameRequested.emit(self.job_name, increment)

    def _emit_selected_result_frame(self) -> None:
        self._emit_result_frame()

    def _refresh_action_buttons(self, *_args: object) -> None:
        """Enable only actions valid for the current run lifecycle."""

        status = self._job.status
        self.cancel_button.setEnabled(
            status is RunStatus.RUNNING and not self._job.cancellation_requested
        )
        self.rerun_button.setEnabled(
            status
            in {
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }
        )
        self.open_result_button.setEnabled(self._job.has_displayable_result)
        self.open_result_button.setText(
            "打开部分结果"
            if self._job.has_partial_result
            else "打开结果"
        )
        self.jump_result_button.setEnabled(
            self._job.has_displayable_result
            and self._selected_increment() is not None
        )

    def _refresh_tabs(
        self,
        job: AnalysisRun,
        snapshot: RunDiagnosticsSnapshot | None,
    ) -> None:
        messages = () if snapshot is None else snapshot.messages
        log_lines = [_message_text(message) for message in messages]
        if not log_lines:
            log_lines = list(job.messages)
        self._set_tab_text("log", "\n".join(log_lines) or "暂无日志")
        self._set_tab_text(
            "errors",
            "\n".join(
                _message_text(message)
                for message in messages
                if message.level == "error"
            )
            or "暂无错误",
        )
        self._set_tab_text(
            "warnings",
            "\n".join(
                _message_text(message)
                for message in messages
                if message.level == "warning"
            )
            or "暂无警告",
        )
        if snapshot is None:
            output = "暂无结构化求解信息"
            data = f"分析步：{job.step_name}\n模型：—"
            status = _monitor_status_text(job.status.value, job)
        else:
            output = _output_text(snapshot)
            if snapshot.dynamic_increments:
                latest = snapshot.dynamic_increments[-1]
                data = (
                    f"分析步：{snapshot.step_name}\n"
                    f"模型：{snapshot.model_name}\n"
                    f"作业类型：{snapshot.analysis_type}\n"
                    f"分析过程：{snapshot.procedure}\n"
                    f"动力学求解器：{latest.solver_kind}\n"
                    f"已完成增量：{len(snapshot.dynamic_increments)}\n"
                    f"最后步时间：{_number_text(latest.step_time)}\n"
                    f"最后时间增量：{_number_text(latest.time_increment)}\n"
                    f"稳定时间步：{_number_text(latest.stable_time_increment)}"
                )
            else:
                data = (
                    f"分析步：{snapshot.step_name}\n"
                    f"模型：{snapshot.model_name}\n"
                    f"作业类型：{snapshot.analysis_type}\n"
                    f"分析过程：{snapshot.procedure}\n"
                    f"NLGEOM：{'On' if snapshot.nlgeom else 'Off'}\n"
                    f"已收敛增量：{snapshot.completed_increments}\n"
                    f"最后收敛载荷因子："
                    f"{_number_text(snapshot.last_converged_load_factor)}"
                )
            status = _status_text(snapshot)
        self._set_tab_text("output", output)
        self._set_tab_text("data", data)
        self._set_tab_text(
            "messages",
            "\n".join(_message_text(message) for message in messages)
            or "暂无消息",
        )
        self._set_tab_text("status", status)

    def _set_tab_text(self, key: str, text: str) -> None:
        """Update one monitor page without moving the user's scroll position."""

        view = self._tab_views[key]
        if self._tab_texts.get(key) == text:
            return
        vertical = view.verticalScrollBar()
        horizontal = view.horizontalScrollBar()
        old_vertical = vertical.value()
        old_horizontal = horizontal.value()
        was_at_bottom = old_vertical >= vertical.maximum()
        self._tab_texts[key] = text
        view.setPlainText(text)
        if was_at_bottom:
            vertical.setValue(vertical.maximum())
        else:
            vertical.setValue(min(old_vertical, vertical.maximum()))
        horizontal.setValue(min(old_horizontal, horizontal.maximum()))


def _monitor_status_text(status: str, job: AnalysisRun) -> str:
    if job.cancellation_requested and job.status is RunStatus.RUNNING:
        return "终止中"
    return {
        "pending": "已创建",
        "running": "运行中",
        "succeeded": "已完成",
        "failed": "失败",
        "cancelled": "已取消",
    }.get(str(status), _STATUS_LABELS.get(job.status, str(status)))


def _attempt_text(record: AttemptRecord) -> str:
    suffix = "U" if record.status in {
        AttemptStatus.CUTBACK,
        AttemptStatus.FAILED,
    } else ""
    return f"{record.attempt}{suffix}"


def _attempt_status_text(status: AttemptStatus) -> str:
    return {
        AttemptStatus.RUNNING: "运行中",
        AttemptStatus.CONVERGED: "已收敛",
        AttemptStatus.CUTBACK: "自动切步",
        AttemptStatus.FAILED: "失败",
    }.get(status, str(status.value))


def _dynamic_status_text(status: str) -> str:
    return {
        "running": "运行中",
        "converged": "已完成",
        "failed": "失败",
        "cutback": "自动切步",
    }.get(str(status), str(status))


def _number_text(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3e}"


def _seconds_text(value: float | None) -> str:
    return "—" if value is None else f"{float(value):.3f} s"


def _message_text(message: RunMessage) -> str:
    stamp = message.created_at.astimezone().strftime("%H:%M:%S")
    level = {
        "info": "信息",
        "warning": "警告",
        "error": "错误",
    }.get(message.level, message.level)
    scope = ""
    if message.increment is not None:
        scope = f" 增量 {message.increment}"
        if message.attempt is not None:
            scope += f" / 尝试 {message.attempt}"
    return f"[{stamp}] [{level}]{scope} {message.text}"


def _output_text(snapshot: RunDiagnosticsSnapshot) -> str:
    if snapshot.dynamic_increments:
        return _dynamic_output_text(snapshot)
    lines = [
        f"阶段：{snapshot.stage}",
        f"已收敛增量：{snapshot.completed_increments}",
        f"当前增量：{snapshot.current_increment or '—'}",
        f"当前尝试：{snapshot.current_attempt or '—'}",
        f"当前 Newton：{snapshot.current_iteration or '—'}",
        f"当前载荷因子：{_number_text(snapshot.current_load_factor)}",
        f"当前残差：{_number_text(snapshot.current_residual)}",
    ]
    return "\n".join(lines)


def _dynamic_output_text(snapshot: RunDiagnosticsSnapshot) -> str:
    latest = snapshot.dynamic_increments[-1]
    lines = [
        f"阶段：{snapshot.stage}",
        f"已完成增量：{len(snapshot.dynamic_increments)}",
        f"求解器：{latest.solver_kind}",
        f"当前增量：{latest.increment}",
        f"当前步时间：{_number_text(snapshot.current_step_time)}",
        f"当前时间增量：{_number_text(snapshot.current_time_increment)}",
        f"稳定时间步：{_number_text(snapshot.current_stable_time_increment)}",
        f"当前 Newton：{latest.iterations if latest.iterations is not None else '—'}",
        f"当前残差：{_number_text(snapshot.current_residual)}",
        f"当前动能：{_number_text(snapshot.current_kinetic_energy)}",
        f"当前内能：{_number_text(snapshot.current_internal_energy)}",
        f"当前外力功：{_number_text(latest.external_work)}",
        f"当前阻尼耗散：{_number_text(latest.damping_dissipation)}",
        f"当前总能量：{_number_text(snapshot.current_total_energy)}",
        f"能量平衡误差：{_number_text(latest.energy_balance_error)}",
    ]
    if latest.critical_element is not None:
        lines.append(f"控制稳定步单元：{latest.critical_element}")
    return "\n".join(lines)


def _status_text(snapshot: RunDiagnosticsSnapshot) -> str:
    label = {
        "pending": "已创建",
        "running": "运行中",
        "succeeded": "已完成",
        "failed": "失败",
        "cancelled": "已取消",
    }.get(snapshot.status, snapshot.status)
    return _output_text(snapshot) + f"\n状态：{label}"
