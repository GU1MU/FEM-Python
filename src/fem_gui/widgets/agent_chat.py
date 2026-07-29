"""FEM Agent 的覆盖式聊天界面与阶段 3 结构化事件预览。

本模块不创建真实 Agent 会话，不导入或调用 ``fem_agent``。工作区文件候选
只使用有界元数据索引，不读取文件内容；对话展示只消费结构化事件投影状态。
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QPoint,
    Property,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QKeyEvent, QMouseEvent, QRegion, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..agent_events import (
    AgentEvent,
    AgentEventProjector,
    ConfirmationView,
    DiagnosticView,
    FakeAgentEventStream,
    MessageStatus,
    MessageView,
    SessionPresentation,
    TimelineKind,
    ToolGroupView,
    ToolStatus,
    TurnStatus,
)
from ..agent_workspace import (
    MAX_VISIBLE_WORKSPACE_CANDIDATES,
    WorkspaceCommandHandler,
    WorkspaceFileReference,
    WorkspaceSelectionResult,
)


_BOUNDARY_EVENT_TYPES = frozenset(
    {
        QEvent.Type.MouseButtonPress,
        QEvent.Type.MouseButtonRelease,
        QEvent.Type.MouseButtonDblClick,
        QEvent.Type.MouseMove,
        QEvent.Type.Wheel,
        QEvent.Type.KeyPress,
        QEvent.Type.KeyRelease,
        QEvent.Type.ShortcutOverride,
        QEvent.Type.ContextMenu,
        QEvent.Type.DragEnter,
        QEvent.Type.DragMove,
        QEvent.Type.DragLeave,
        QEvent.Type.Drop,
        QEvent.Type.TouchBegin,
        QEvent.Type.TouchUpdate,
        QEvent.Type.TouchEnd,
        QEvent.Type.TabletPress,
        QEvent.Type.TabletMove,
        QEvent.Type.TabletRelease,
    }
)


_AGENT_CHAT_STYLESHEET = """
QFrame#agentChatDrawer {
    background: #f8f9fb;
    border-left: 1px solid #c7ccd2;
}
QFrame#agentChatDrawer QLabel {
    background: transparent;
}
QFrame#agentChatHeader {
    background: #ffffff;
    border: none;
    border-bottom: 1px solid #dfe3e7;
}
QLabel#agentChatTitle {
    color: #20262d;
    font-size: 10.5pt;
    font-weight: 600;
}
QLabel#agentChatPreviewBadge {
    color: #426b88;
    background: #e8f0f6;
    border: 1px solid #c8dbe8;
    border-radius: 7px;
    padding: 1px 6px;
    font-size: 7.5pt;
}
QLabel#agentChatSubtitle, QLabel#agentChatMuted,
QLabel#agentChatComposerHint, QLabel#agentChatToolMeta {
    color: #6c7680;
    font-size: 8pt;
}
QToolButton#agentChatHeaderButton {
    color: #4a545e;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    min-width: 26px;
    max-width: 26px;
    min-height: 26px;
    max-height: 26px;
    padding: 0;
    font-size: 12pt;
}
QToolButton#agentChatHeaderButton:hover {
    background: #edf2f5;
    border-color: #d7dde2;
}
QScrollArea#agentChatScroll {
    background: #f8f9fb;
    border: none;
}
QWidget#agentChatConversation {
    background: #f8f9fb;
}
QFrame#agentChatWelcome {
    background: #eef4f8;
    border: 1px solid #d5e2ea;
    border-radius: 7px;
}
QLabel#agentChatWelcomeTitle {
    color: #2d4f68;
    font-weight: 600;
}
QFrame#agentChatUserMessage {
    background: #e7eff5;
    border: 1px solid #d1e0eb;
    border-radius: 9px;
}
QLabel#agentChatUserLabel {
    color: #29333c;
}
QLabel#agentChatSpeaker {
    color: #334b5d;
    font-weight: 600;
}
QLabel#agentChatAgentMessage {
    color: #28313a;
}
QFrame#agentChatToolActivity {
    background: transparent;
    border: none;
    border-top: 1px solid #e1e4e7;
    border-bottom: 1px solid #e1e4e7;
}
QToolButton#agentChatToolSummary {
    color: #707981;
    background: transparent;
    border: none;
    padding: 5px 1px;
    text-align: left;
    font-size: 8pt;
}
QToolButton#agentChatToolSummary:hover {
    color: #45515b;
    background: #f1f3f5;
}
QFrame#agentChatToolDetails {
    background: #f3f4f6;
    border: 1px solid #e2e5e8;
    border-radius: 5px;
}
QLabel#agentChatToolColumn {
    color: #879099;
    font-size: 7.5pt;
    font-weight: 600;
}
QLabel#agentChatToolValue {
    color: #68727b;
    font-size: 7.5pt;
}
QLabel#agentChatToolSuccess {
    color: #4f775f;
    font-size: 7.5pt;
}
QLabel#agentChatToolWarning {
    color: #8a681d;
    font-size: 7.5pt;
}
QLabel#agentChatToolFailure {
    color: #a14444;
    font-size: 7.5pt;
}
QLabel#agentChatToolDetail {
    color: #707981;
    font-size: 7pt;
}
QFrame#agentChatDiagnostic {
    background: #fff8e7;
    border: 1px solid #ead9a6;
    border-left: 3px solid #c99a2e;
    border-radius: 5px;
}
QLabel#agentChatDiagnosticTitle {
    color: #795e1c;
    font-weight: 600;
}
QLabel#agentChatDiagnosticText {
    color: #65562f;
}
QFrame#agentChatDiagnostic[severity="info"] {
    background: #edf5fa;
    border-color: #cbdde8;
    border-left-color: #5c8aa8;
}
QFrame#agentChatDiagnostic[severity="error"],
QFrame#agentChatDiagnostic[severity="blocking"] {
    background: #fff0f0;
    border-color: #e5c2c2;
    border-left-color: #b34f4f;
}
QFrame#agentChatConfirmation {
    background: #f1f4f8;
    border: 1px solid #cad3dc;
    border-left: 3px solid #587b98;
    border-radius: 5px;
}
QLabel#agentChatConfirmationTitle {
    color: #36556d;
    font-weight: 600;
}
QLabel#agentChatConfirmationText {
    color: #435665;
}
QLabel#agentChatConfirmationRevision {
    color: #6c7680;
    font-size: 7pt;
}
QToolButton#agentChatConfirmationButton {
    color: #76818a;
    background: #e5e9ed;
    border: 1px solid #d2d8dd;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 7.5pt;
}
QLabel#agentChatTurnStatus {
    color: #a14444;
    background: #fff0f0;
    border: 1px solid #e5c2c2;
    border-radius: 4px;
    padding: 5px 7px;
    font-size: 7.5pt;
}
QFrame#agentChatComposer {
    background: #ffffff;
    border: none;
    border-top: 1px solid #dfe3e7;
}
QLabel#agentChatWorkspaceState {
    color: #66717b;
    background: #f3f5f7;
    border: 1px solid #dde1e5;
    border-radius: 6px;
    padding: 2px 7px;
    font-size: 7.5pt;
}
QFrame#agentChatSuggestion {
    background: #ffffff;
    border: 1px solid #ced5db;
    border-radius: 5px;
}
QLabel#agentChatSuggestionTitle {
    color: #66717b;
    font-size: 7.5pt;
    font-weight: 600;
}
QListWidget#agentChatSuggestionList {
    color: #34434f;
    background: #ffffff;
    border: none;
    outline: none;
}
QListWidget#agentChatSuggestionList::item {
    padding: 4px 5px;
    border-radius: 3px;
}
QListWidget#agentChatSuggestionList::item:selected {
    color: #263946;
    background: #e8f0f6;
}
QPlainTextEdit#agentChatInput {
    color: #20262d;
    background: #ffffff;
    border: 1px solid #cbd2d8;
    border-radius: 7px;
    padding: 7px;
    selection-background-color: #dce9f2;
}
QPlainTextEdit#agentChatInput:focus {
    border-color: #4c7fa5;
}
QPlainTextEdit#agentChatInput:disabled {
    color: #8f979e;
    background: #f5f6f7;
}
QToolButton#agentChatAddButton, QToolButton#agentChatSendButton,
QToolButton#agentChatStopButton {
    border-radius: 6px;
    min-width: 30px;
    max-width: 30px;
    min-height: 30px;
    max-height: 30px;
    padding: 0;
    font-size: 12pt;
}
QToolButton#agentChatAddButton {
    color: #4c5963;
    background: transparent;
    border: 1px solid transparent;
}
QToolButton#agentChatAddButton:hover {
    background: #edf2f5;
    border-color: #d7dde2;
}
QToolButton#agentChatSendButton {
    color: #ffffff;
    background: #4c7fa5;
    border: 1px solid #4c7fa5;
}
QToolButton#agentChatSendButton:hover {
    background: #3f6f92;
}
QToolButton#agentChatSendButton:disabled {
    color: #9ea8af;
    background: #e6e9eb;
    border-color: #e0e3e5;
}
QToolButton#agentChatStopButton {
    color: #ffffff;
    background: #a85252;
    border: 1px solid #964949;
}
QToolButton#agentChatStopButton:hover {
    background: #914747;
}
QMenu#agentChatAddMenu {
    background: #ffffff;
    border: 1px solid #cbd2d8;
    padding: 4px;
}
QMenu#agentChatAddMenu::item {
    padding: 6px 24px 6px 9px;
}
QMenu#agentChatAddMenu::item:selected {
    background: #e8f0f6;
}
QToolButton#agentChatLauncher {
    color: #365d78;
    background: #ffffff;
    border: 1px solid #c4d1da;
    border-radius: 17px;
    min-width: 34px;
    max-width: 34px;
    min-height: 34px;
    max-height: 34px;
    padding: 0;
    font-size: 11pt;
    font-weight: 600;
}
QToolButton#agentChatLauncher:hover {
    background: #e8f0f6;
    border-color: #8faec2;
}
QFrame#agentChatResizeHandle {
    background: transparent;
    border: none;
}
QFrame#agentChatResizeHandle:hover {
    background: #8bb0c8;
}
"""


class _EventBoundaryMixin:
    """Accept input locally so an ignored event cannot cross into a sibling."""

    def event(self, event: QEvent) -> bool:
        handled = super().event(event)
        if event.type() in _BOUNDARY_EVENT_TYPES:
            event.accept()
            return True
        return handled


class _BoundaryFrame(_EventBoundaryMixin, QFrame):
    pass


class _BoundaryToolButton(_EventBoundaryMixin, QToolButton):
    pass


class _BoundaryListWidget(_EventBoundaryMixin, QListWidget):
    pass


class _ChatInput(QPlainTextEdit):
    """在输入框内路由发送与候选键盘操作。"""

    submitRequested = Signal()
    suggestionMoveRequested = Signal(int)
    suggestionAcceptRequested = Signal()
    suggestionDismissRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._suggestions_active = False

    def set_suggestions_active(self, active: bool) -> None:
        self._suggestions_active = bool(active)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        modifiers = event.modifiers()
        if self._suggestions_active:
            if key == Qt.Key.Key_Down:
                self.suggestionMoveRequested.emit(1)
                event.accept()
                return
            if key == Qt.Key.Key_Up:
                self.suggestionMoveRequested.emit(-1)
                event.accept()
                return
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
                modifiers & Qt.KeyboardModifier.ShiftModifier
            ):
                self.suggestionAcceptRequested.emit()
                event.accept()
                return
            if key == Qt.Key.Key_Escape:
                self.suggestionDismissRequested.emit()
                event.accept()
                return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            modifiers & Qt.KeyboardModifier.ShiftModifier
        ):
            self.submitRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class _DrawerResizeHandle(_BoundaryFrame):
    dragDelta = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("agentChatResizeHandle")
        self.setFixedWidth(6)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self._last_global_x: int | None = None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_global_x = round(event.globalPosition().x())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._last_global_x is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            current_x = round(event.globalPosition().x())
            self.dragDelta.emit(self._last_global_x - current_x)
            self._last_global_x = current_x
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            self._last_global_x is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._last_global_x = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


_TOOL_STATUS_LABELS = {
    ToolStatus.REQUESTED: "等待",
    ToolStatus.RUNNING: "执行中",
    ToolStatus.COMPLETED: "完成",
    ToolStatus.WARNING: "警告",
    ToolStatus.FAILED: "失败",
    ToolStatus.CANCELLED: "已取消",
}


def _restricted_markdown_html(markdown: str) -> str:
    """把受限 Markdown 转成不含链接、图片或原始 HTML 的 Qt 富文本。"""
    escaped = html.escape(markdown, quote=True)
    escaped = re.sub(
        r"`([^`\n]+)`",
        r"<span style='font-family:monospace'>\1</span>",
        escaped,
    )
    escaped = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped.replace("\n", "<br>")


def _plain_label(text: str, parent: QWidget) -> QLabel:
    label = QLabel(parent)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setText(text)
    return label


def _tool_summary_text(group: ToolGroupView) -> str:
    parts = [f"已调用 {len(group.calls)} 个工具"]
    if group.completed_count:
        parts.append(f"{group.completed_count} 项完成")
    if group.warning_count:
        parts.append(f"{group.warning_count} 项警告")
    if group.failed_count:
        parts.append(f"{group.failed_count} 项失败")
    if group.cancelled_count:
        parts.append(f"{group.cancelled_count} 项取消")
    running = sum(
        call.status in {ToolStatus.REQUESTED, ToolStatus.RUNNING}
        for call in group.calls
    )
    if running:
        parts.append(f"{running} 项进行中")
    parts.append(f"{group.total_duration_ms / 1000:.1f} 秒")
    return " · ".join(parts)


class ToolActivityPreview(_BoundaryFrame):
    """由结构化 ``ToolGroupView`` 驱动的可折叠工具活动。"""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        group: ToolGroupView | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("agentChatToolActivity")
        self._group = group or ToolGroupView(group_id="empty")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(3)

        self.summary_button = _BoundaryToolButton(self)
        self.summary_button.setObjectName("agentChatToolSummary")
        self.summary_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.summary_button.setCheckable(True)
        self.summary_button.setChecked(False)
        self.summary_button.toggled.connect(self._set_expanded)
        layout.addWidget(self.summary_button)

        self.details = _BoundaryFrame(self)
        self.details.setObjectName("agentChatToolDetails")
        self.details_layout = QVBoxLayout(self.details)
        self.details_layout.setContentsMargins(8, 6, 8, 6)
        self.details_layout.setSpacing(6)
        layout.addWidget(self.details)
        self._populate_details()
        self._set_expanded(False)

    @property
    def group(self) -> ToolGroupView:
        return self._group

    def _populate_details(self) -> None:
        while self.details_layout.count():
            item = self.details_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        for call in self._group.calls:
            row = QWidget(self.details)
            row_layout = QGridLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setHorizontalSpacing(8)
            row_layout.setVerticalSpacing(2)

            name = _plain_label(call.display_name, row)
            name.setObjectName("agentChatToolValue")
            row_layout.addWidget(name, 0, 0)
            status = QLabel(_TOOL_STATUS_LABELS[call.status], row)
            if call.status is ToolStatus.COMPLETED:
                status.setObjectName("agentChatToolSuccess")
            elif call.status is ToolStatus.WARNING:
                status.setObjectName("agentChatToolWarning")
            elif call.status in {
                ToolStatus.FAILED,
                ToolStatus.CANCELLED,
            }:
                status.setObjectName("agentChatToolFailure")
            else:
                status.setObjectName("agentChatToolValue")
            row_layout.addWidget(status, 0, 1)
            duration = QLabel(f"{call.duration_ms / 1000:.1f} s", row)
            duration.setObjectName("agentChatToolValue")
            row_layout.addWidget(duration, 0, 2)
            row_layout.setColumnStretch(0, 1)

            request = _plain_label(
                f"请求 · {call.request_summary}",
                row,
            )
            request.setObjectName("agentChatToolDetail")
            request.setWordWrap(True)
            row_layout.addWidget(request, 1, 0, 1, 3)
            if call.result_summary:
                result = _plain_label(
                    f"返回 · {call.result_summary}",
                    row,
                )
                result.setObjectName("agentChatToolDetail")
                result.setWordWrap(True)
                row_layout.addWidget(result, 2, 0, 1, 3)
            for offset, diagnostic_text in enumerate(
                call.diagnostics,
                start=3,
            ):
                diagnostic = _plain_label(
                    f"诊断 · {diagnostic_text}",
                    row,
                )
                diagnostic.setObjectName(
                    "agentChatToolWarning"
                    if call.status is ToolStatus.WARNING
                    else "agentChatToolFailure"
                )
                diagnostic.setWordWrap(True)
                row_layout.addWidget(diagnostic, offset, 0, 1, 3)
            self.details_layout.addWidget(row)

    def _set_expanded(self, expanded: bool) -> None:
        self.details.setVisible(expanded)
        self.summary_button.setArrowType(
            Qt.ArrowType.DownArrow
            if expanded
            else Qt.ArrowType.RightArrow
        )
        self.summary_button.setText(_tool_summary_text(self._group))


class AgentChatDrawer(_BoundaryFrame):
    """覆盖在模型画布之上的 FEM Agent 聊天面板。"""

    closeRequested = Signal()
    messagePreviewRequested = Signal(str, object)
    agentEventApplied = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        workspace_commands: WorkspaceCommandHandler | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("agentChatDrawer")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self._busy_preview = False
        self._expanded_tool_group_ids: set[str] = set()
        self.event_projector = AgentEventProjector()
        self.workspace_commands = (
            workspace_commands or WorkspaceCommandHandler()
        )
        self._workspace_index = self.workspace_commands.workspace_index
        self._workspace_references: list[
            WorkspaceFileReference
        ] = []

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.resize_handle = _DrawerResizeHandle(self)
        root.addWidget(self.resize_handle)

        pane = QWidget(self)
        pane.setObjectName("agentChatPane")
        pane_layout = QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)
        pane_layout.setSpacing(0)
        root.addWidget(pane, 1)

        pane_layout.addWidget(self._build_header(pane))
        pane_layout.addWidget(self._build_conversation(pane), 1)
        pane_layout.addWidget(self._build_composer(pane))
        self._update_workspace_state()
        self.replay_agent_events(FakeAgentEventStream().review_preview())

    def _build_header(self, parent: QWidget) -> QWidget:
        header = _BoundaryFrame(parent)
        header.setObjectName("agentChatHeader")
        layout = QGridLayout(header)
        layout.setContentsMargins(12, 8, 8, 8)
        layout.setHorizontalSpacing(7)
        layout.setVerticalSpacing(1)

        title = QLabel("FEM Agent", header)
        title.setObjectName("agentChatTitle")
        layout.addWidget(title, 0, 0)
        badge = QLabel("事件预览", header)
        badge.setObjectName("agentChatPreviewBadge")
        layout.addWidget(badge, 0, 1)
        layout.setColumnStretch(2, 1)

        self.new_session_button = _BoundaryToolButton(header)
        self.new_session_button.setObjectName("agentChatHeaderButton")
        self.new_session_button.setText("＋")
        self.new_session_button.setToolTip("清空当前输入预览")
        self.new_session_button.clicked.connect(self._reset_preview)
        layout.addWidget(self.new_session_button, 0, 3, 2, 1)

        self.close_button = _BoundaryToolButton(header)
        self.close_button.setObjectName("agentChatHeaderButton")
        self.close_button.setText("×")
        self.close_button.setToolTip("关闭聊天框")
        self.close_button.clicked.connect(self.closeRequested)
        layout.addWidget(self.close_button, 0, 4, 2, 1)

        subtitle = QLabel("纯内存 Fake 流 · 尚未接入 Agent", header)
        subtitle.setObjectName("agentChatSubtitle")
        layout.addWidget(subtitle, 1, 0, 1, 3)
        return header

    def _build_conversation(self, parent: QWidget) -> QWidget:
        scroll = QScrollArea(parent)
        scroll.setObjectName("agentChatScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        conversation = QWidget(scroll)
        conversation.setObjectName("agentChatConversation")
        layout = QVBoxLayout(conversation)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(11)

        welcome = _BoundaryFrame(conversation)
        welcome.setObjectName("agentChatWelcome")
        welcome_layout = QVBoxLayout(welcome)
        welcome_layout.setContentsMargins(10, 8, 10, 8)
        welcome_layout.setSpacing(3)
        welcome_title = QLabel("FEM Agent 结构化事件预览", welcome)
        welcome_title.setObjectName("agentChatWelcomeTitle")
        welcome_layout.addWidget(welcome_title)
        welcome_text = QLabel(
            "回复、工具、诊断和确认均来自纯内存 Fake 事件。"
            "不会执行工具、求解或网络请求。",
            welcome,
        )
        welcome_text.setObjectName("agentChatMuted")
        welcome_text.setWordWrap(True)
        welcome_layout.addWidget(welcome_text)
        layout.addWidget(welcome)

        self.event_feed = QWidget(conversation)
        self.event_feed.setObjectName("agentChatEventFeed")
        self.event_feed_layout = QVBoxLayout(self.event_feed)
        self.event_feed_layout.setContentsMargins(0, 0, 0, 0)
        self.event_feed_layout.setSpacing(11)
        layout.addWidget(self.event_feed)
        layout.addStretch(1)

        scroll.setWidget(conversation)
        self.conversation_widget = conversation
        self.conversation_layout = layout
        self.conversation_scroll = scroll
        return scroll

    @property
    def event_presentation(self) -> SessionPresentation:
        """返回与 Qt 控件分离的结构化展示快照。"""
        return self.event_projector.presentation

    def replay_agent_events(
        self,
        events: Iterable[AgentEvent],
    ) -> None:
        """用完整事件日志替换当前 Fake 展示并一次性重绘。"""
        self.event_projector = AgentEventProjector.replay(events)
        self._expanded_tool_group_ids.clear()
        self._render_event_presentation(preserve_tool_expansion=False)

    def apply_agent_event(self, event: AgentEvent) -> None:
        """消费一个已验证事件；不解析 CLI 文本。"""
        self.event_projector.apply(event)
        self._render_event_presentation()
        self.agentEventApplied.emit(event)

    def _clear_event_feed(
        self,
        *,
        preserve_tool_expansion: bool,
    ) -> None:
        if preserve_tool_expansion:
            for tools in self.event_feed.findChildren(ToolActivityPreview):
                group_id = tools.group.group_id
                if tools.summary_button.isChecked():
                    self._expanded_tool_group_ids.add(group_id)
                else:
                    self._expanded_tool_group_ids.discard(group_id)
        while self.event_feed_layout.count():
            item = self.event_feed_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    def _render_event_presentation(
        self,
        *,
        preserve_tool_expansion: bool = True,
    ) -> None:
        self._clear_event_feed(
            preserve_tool_expansion=preserve_tool_expansion,
        )
        for turn in self.event_projector.presentation.turns:
            self._add_user_message(turn.user_message, turn.turn_id)
            speaker = QLabel("FEM Agent", self.event_feed)
            speaker.setObjectName("agentChatSpeaker")
            self.event_feed_layout.addWidget(speaker)

            messages = {
                message.message_id: message
                for message in turn.messages
            }
            tool_groups = {
                group.group_id: group
                for group in turn.tool_groups
            }
            diagnostics = {
                diagnostic.diagnostic_id: diagnostic
                for diagnostic in turn.diagnostics
            }
            confirmations = {
                confirmation.confirmation_id: confirmation
                for confirmation in turn.confirmations
            }
            for timeline_item in turn.timeline:
                if timeline_item.kind is TimelineKind.MESSAGE:
                    self._add_agent_message(
                        messages[timeline_item.item_id]
                    )
                elif timeline_item.kind is TimelineKind.TOOL_GROUP:
                    tools = ToolActivityPreview(
                        self.event_feed,
                        group=tool_groups[timeline_item.item_id],
                    )
                    tools.setProperty("groupId", timeline_item.item_id)
                    tools.summary_button.setChecked(
                        timeline_item.item_id
                        in self._expanded_tool_group_ids
                    )
                    self.event_feed_layout.addWidget(tools)
                elif timeline_item.kind is TimelineKind.DIAGNOSTIC:
                    self._add_diagnostic_card(
                        diagnostics[timeline_item.item_id]
                    )
                elif timeline_item.kind is TimelineKind.CONFIRMATION:
                    self._add_confirmation_card(
                        confirmations[timeline_item.item_id]
                    )
            if turn.status in {TurnStatus.CANCELLED, TurnStatus.FAILED}:
                status = _plain_label(
                    (
                        "本轮已取消 · "
                        if turn.status is TurnStatus.CANCELLED
                        else "本轮失败 · "
                    )
                    + turn.failure_reason,
                    self.event_feed,
                )
                status.setObjectName("agentChatTurnStatus")
                status.setWordWrap(True)
                self.event_feed_layout.addWidget(status)

    def _add_user_message(self, text: str, turn_id: str) -> None:
        user_row = QWidget(self.event_feed)
        user_row.setObjectName("agentChatUserRow")
        user_row.setProperty("turnId", turn_id)
        user_layout = QHBoxLayout(user_row)
        user_layout.setContentsMargins(34, 0, 0, 0)
        user_layout.setSpacing(0)
        user_bubble = _BoundaryFrame(user_row)
        user_bubble.setObjectName("agentChatUserMessage")
        bubble_layout = QVBoxLayout(user_bubble)
        bubble_layout.setContentsMargins(10, 7, 10, 7)
        user_text = _plain_label(text, user_bubble)
        user_text.setObjectName("agentChatUserLabel")
        user_text.setWordWrap(True)
        bubble_layout.addWidget(user_text)
        user_layout.addWidget(user_bubble)
        self.event_feed_layout.addWidget(user_row)

    def _add_agent_message(self, message: MessageView) -> None:
        text = message.text
        if message.status is MessageStatus.STREAMING:
            text += " ▌"
        label = QLabel(self.event_feed)
        label.setObjectName("agentChatAgentMessage")
        label.setProperty("messageId", message.message_id)
        label.setProperty("messageStatus", message.status.value)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setText(_restricted_markdown_html(text))
        label.setOpenExternalLinks(False)
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.event_feed_layout.addWidget(label)
        if message.status in {
            MessageStatus.CANCELLED,
            MessageStatus.INTERRUPTED,
        }:
            suffix = QLabel(
                (
                    "回复已取消"
                    if message.status is MessageStatus.CANCELLED
                    else "回复因失败中断"
                ),
                self.event_feed,
            )
            suffix.setObjectName("agentChatMuted")
            self.event_feed_layout.addWidget(suffix)

    def _add_diagnostic_card(self, diagnostic: DiagnosticView) -> None:
        card = _BoundaryFrame(self.event_feed)
        card.setObjectName("agentChatDiagnostic")
        card.setProperty("diagnosticId", diagnostic.diagnostic_id)
        card.setProperty("severity", diagnostic.severity.value)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(9, 7, 9, 7)
        layout.setSpacing(2)
        title = _plain_label(diagnostic.title, card)
        title.setObjectName("agentChatDiagnosticTitle")
        layout.addWidget(title)
        text = _plain_label(diagnostic.message, card)
        text.setObjectName("agentChatDiagnosticText")
        text.setWordWrap(True)
        layout.addWidget(text)
        if diagnostic.code:
            code = _plain_label(diagnostic.code, card)
            code.setObjectName("agentChatMuted")
            layout.addWidget(code)
        self.event_feed_layout.addWidget(card)

    def _add_confirmation_card(
        self,
        confirmation: ConfirmationView,
    ) -> None:
        card = _BoundaryFrame(self.event_feed)
        card.setObjectName("agentChatConfirmation")
        card.setProperty("confirmationId", confirmation.confirmation_id)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(9, 7, 9, 7)
        layout.setSpacing(4)
        title = _plain_label(confirmation.title, card)
        title.setObjectName("agentChatConfirmationTitle")
        layout.addWidget(title)
        text = _plain_label(confirmation.summary, card)
        text.setObjectName("agentChatConfirmationText")
        text.setWordWrap(True)
        layout.addWidget(text)
        revision = QLabel(
            f"revision {confirmation.revision} · "
            f"sha256 {confirmation.revision_hash[:16]}…",
            card,
        )
        revision.setObjectName("agentChatConfirmationRevision")
        revision.setToolTip(confirmation.revision_hash)
        layout.addWidget(revision)
        button = _BoundaryToolButton(card)
        button.setObjectName("agentChatConfirmationButton")
        button.setText("确认（阶段 3 不执行）")
        button.setEnabled(False)
        button.setProperty(
            "confirmationId",
            confirmation.confirmation_id,
        )
        button.setProperty("revision", confirmation.revision)
        button.setProperty("revisionHash", confirmation.revision_hash)
        button.setProperty("authorized", confirmation.authorized)
        layout.addWidget(
            button,
            0,
            Qt.AlignmentFlag.AlignLeft,
        )
        self.event_feed_layout.addWidget(card)

    def _build_composer(self, parent: QWidget) -> QWidget:
        composer = _BoundaryFrame(parent)
        composer.setObjectName("agentChatComposer")
        layout = QVBoxLayout(composer)
        layout.setContentsMargins(10, 8, 10, 9)
        layout.setSpacing(6)

        self.workspace_state = QLabel("工作区  尚未选择", composer)
        self.workspace_state.setObjectName("agentChatWorkspaceState")
        self.workspace_state.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(
            self.workspace_state,
            0,
            Qt.AlignmentFlag.AlignLeft,
        )

        self.suggestion = _BoundaryFrame(composer)
        self.suggestion.setObjectName("agentChatSuggestion")
        suggestion_layout = QVBoxLayout(self.suggestion)
        suggestion_layout.setContentsMargins(7, 5, 7, 5)
        suggestion_layout.setSpacing(2)
        self.suggestion_title = QLabel("", self.suggestion)
        self.suggestion_title.setObjectName("agentChatSuggestionTitle")
        suggestion_layout.addWidget(self.suggestion_title)
        self.suggestion_list = _BoundaryListWidget(self.suggestion)
        self.suggestion_list.setObjectName("agentChatSuggestionList")
        self.suggestion_list.setMaximumHeight(126)
        self.suggestion_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.suggestion_list.itemClicked.connect(
            self._activate_suggestion
        )
        self.suggestion_item = QListWidgetItem("")
        suggestion_layout.addWidget(self.suggestion_list)
        layout.addWidget(self.suggestion)
        self.suggestion.hide()

        self.input = _ChatInput(composer)
        self.input.setObjectName("agentChatInput")
        self.input.setPlaceholderText(
            "询问 FEM Agent；使用 @ 引用工作区文件…"
        )
        self.input.setMaximumHeight(86)
        self.input.setTabChangesFocus(True)
        self.input.textChanged.connect(self._input_changed)
        self.input.document().contentsChange.connect(
            self._adjust_reference_ranges
        )
        self.input.submitRequested.connect(self._submit_current_input)
        self.input.suggestionMoveRequested.connect(
            self._move_suggestion_selection
        )
        self.input.suggestionAcceptRequested.connect(
            self._activate_current_suggestion
        )
        self.input.suggestionDismissRequested.connect(
            self._hide_suggestions
        )
        layout.addWidget(self.input)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(5)

        self.add_button = _BoundaryToolButton(composer)
        self.add_button.setObjectName("agentChatAddButton")
        self.add_button.setText("＋")
        self.add_button.setToolTip("添加上下文")
        self.add_menu = QMenu(self.add_button)
        self.add_menu.setObjectName("agentChatAddMenu")
        self.workspace_action = self.add_menu.addAction("选择工作区…")
        self.workspace_action.setObjectName("agentChatWorkspaceAction")
        self.workspace_action.triggered.connect(
            self._run_workspace_command
        )
        self.add_button.clicked.connect(self._show_add_menu)
        footer.addWidget(self.add_button)

        self.composer_hint = QLabel(
            "Enter 发送 · Shift+Enter 换行",
            composer,
        )
        self.composer_hint.setObjectName("agentChatComposerHint")
        footer.addWidget(self.composer_hint)
        footer.addStretch(1)

        self.send_state = QStackedWidget(composer)
        self.send_state.setObjectName("agentChatSendState")
        self.send_state.setFixedSize(30, 30)
        self.send_button = _BoundaryToolButton(self.send_state)
        self.send_button.setObjectName("agentChatSendButton")
        self.send_button.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_ArrowUp
            )
        )
        self.send_button.setIconSize(QSize(16, 16))
        self.send_button.setToolTip("发送（当前仍为界面预览）")
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(self._submit_current_input)
        self.stop_button = _BoundaryToolButton(self.send_state)
        self.stop_button.setObjectName("agentChatStopButton")
        self.stop_button.setText("■")
        self.stop_button.setToolTip("停止界面预览状态")
        self.stop_button.clicked.connect(
            lambda: self.set_busy_preview(False)
        )
        self.send_state.addWidget(self.send_button)
        self.send_state.addWidget(self.stop_button)
        self.send_state.setCurrentWidget(self.send_button)
        footer.addWidget(self.send_state)
        layout.addLayout(footer)
        return composer

    def _show_add_menu(self) -> None:
        menu_size = self.add_menu.sizeHint()
        self.add_menu.popup(
            self.add_button.mapToGlobal(
                QPoint(0, -menu_size.height())
            )
        )

    @property
    def workspace_file_references(
        self,
    ) -> tuple[WorkspaceFileReference, ...]:
        """返回输入框当前保存的结构化工作区引用。"""
        return tuple(self._workspace_references)

    def _active_token(self) -> tuple[int, int, str] | None:
        text = self.input.toPlainText()
        end = self.input.textCursor().position()
        if end == 0 and text:
            end = len(text)
        start = end
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        token = text[start:end]
        if not token.startswith(("@", "/")):
            return None
        return start, end, token

    def _input_changed(self) -> None:
        text = self.input.toPlainText()
        self.send_button.setEnabled(
            bool(text.strip()) and not self._busy_preview
        )
        active = self._active_token()
        if active is None:
            self._hide_suggestions()
            return
        _start, _end, token = active
        if token.startswith("/"):
            if "/workspace".startswith(token.casefold()):
                self._show_slash_suggestion()
            else:
                self._hide_suggestions()
            return
        self._show_workspace_suggestions(token[1:])

    def _show_slash_suggestion(self) -> None:
        self.suggestion_title.setText("斜杠命令")
        self.suggestion_list.clear()
        item = QListWidgetItem("/workspace  选择工作区")
        item.setData(Qt.ItemDataRole.UserRole, "/workspace")
        self.suggestion_list.addItem(item)
        self.suggestion_item = item
        self.suggestion_list.setCurrentRow(0)
        self._show_suggestions()

    def _show_workspace_suggestions(self, query: str) -> None:
        self.suggestion_list.clear()
        snapshot = self._workspace_index
        if snapshot is None:
            self.suggestion_title.setText("@ 工作区文件")
            item = QListWidgetItem(
                "请先选择工作区，再引用各种类型的普通文件"
            )
            self.suggestion_list.addItem(item)
            self.suggestion_item = item
            self.suggestion_list.setCurrentRow(0)
            self._show_suggestions()
            return

        matches = snapshot.matching_files(
            query,
            limit=MAX_VISIBLE_WORKSPACE_CANDIDATES + 1,
        )
        visible = matches[:MAX_VISIBLE_WORKSPACE_CANDIDATES]
        labels: list[str] = []
        if snapshot.truncated:
            labels.append("索引已截断")
        if len(matches) > MAX_VISIBLE_WORKSPACE_CANDIDATES:
            labels.append("候选已截断")
        suffix = f" · {' · '.join(labels)}" if labels else ""
        self.suggestion_title.setText(f"@ 工作区文件{suffix}")

        if not visible:
            item = QListWidgetItem("没有匹配的工作区文件")
            self.suggestion_list.addItem(item)
            self.suggestion_item = item
        else:
            for reference in visible:
                item = QListWidgetItem(reference.relative_path)
                item.setData(Qt.ItemDataRole.UserRole, reference)
                item.setToolTip(
                    f"{reference.file_type} · "
                    f"{reference.size_bytes} 字节"
                )
                self.suggestion_list.addItem(item)
            self.suggestion_item = self.suggestion_list.item(0)
        self.suggestion_list.setCurrentRow(0)
        self._show_suggestions()

    def _show_suggestions(self) -> None:
        rows = max(1, min(4, self.suggestion_list.count()))
        self.suggestion_list.setFixedHeight(8 + rows * 27)
        self.suggestion.show()
        self.suggestion.raise_()
        self.input.set_suggestions_active(True)

    def _hide_suggestions(self) -> None:
        self.suggestion.hide()
        self.input.set_suggestions_active(False)

    def _move_suggestion_selection(self, delta: int) -> None:
        count = self.suggestion_list.count()
        if not count:
            return
        row = self.suggestion_list.currentRow()
        self.suggestion_list.setCurrentRow((row + int(delta)) % count)

    def _activate_current_suggestion(self) -> None:
        self._activate_suggestion(
            self.suggestion_list.currentItem()
        )

    def _activate_suggestion(
        self,
        item: QListWidgetItem | None,
    ) -> None:
        if item is None:
            return
        payload = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(payload, WorkspaceFileReference):
            self._insert_workspace_reference(payload)
            return
        if payload == "/workspace":
            if self.input.toPlainText().strip().casefold() == "/workspace":
                self._hide_suggestions()
                self._submit_current_input()
                return
            self._replace_active_token("/workspace")

    def _replace_active_token(
        self,
        replacement: str,
        *,
        trailing_space: bool = False,
    ) -> tuple[int, int] | None:
        active = self._active_token()
        if active is None:
            return None
        start, end, _token = active
        cursor = self.input.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(replacement + (" " if trailing_space else ""))
        self.input.setTextCursor(cursor)
        return start, start + len(replacement)

    def _insert_workspace_reference(
        self,
        reference: WorkspaceFileReference,
    ) -> None:
        text_range = self._replace_active_token(
            reference.mention_text,
            trailing_space=True,
        )
        if text_range is None:
            return
        self._workspace_references.append(
            reference.at_text_range(*text_range)
        )
        self._hide_suggestions()
        self.input.setFocus()

    def _adjust_reference_ranges(
        self,
        position: int,
        chars_removed: int,
        chars_added: int,
    ) -> None:
        delta = int(chars_added) - int(chars_removed)
        changed_end = int(position) + int(chars_removed)
        adjusted: list[WorkspaceFileReference] = []
        for reference in self._workspace_references:
            start = reference.mention_start
            end = reference.mention_end
            if start is None or end is None:
                continue
            if changed_end <= start:
                adjusted.append(
                    reference.at_text_range(start + delta, end + delta)
                )
            elif int(position) >= end:
                adjusted.append(reference)
        self._workspace_references = adjusted

    def _submit_current_input(self) -> None:
        text = self.input.toPlainText().strip()
        if not text:
            return
        if text.casefold() == "/workspace":
            self.input.clear()
            self._hide_suggestions()
            self._run_workspace_command()
            return
        self._preview_send()

    def _run_workspace_command(
        self,
        _checked: bool = False,
    ) -> WorkspaceSelectionResult:
        result = self.workspace_commands.execute(
            "/workspace",
            parent=self,
        )
        if result.succeeded:
            self._workspace_index = result.index
            self._workspace_references.clear()
            self._update_workspace_state()
            count = len(result.index.files) if result.index else 0
            suffix = "，索引已截断" if result.index and result.index.truncated else ""
            self._show_preview_notice(
                f"已选择工作区 · {count} 个文件{suffix}"
            )
        elif result.cancelled:
            self._show_preview_notice("已取消选择，工作区保持不变")
        else:
            self._show_preview_notice(
                f"无法选择工作区：{result.error or '未知错误'}"
            )
        return result

    def _update_workspace_state(self) -> None:
        snapshot = self._workspace_index
        private_root = self.workspace_commands.agent_data_root
        if snapshot is None:
            self.workspace_state.setText("工作区  尚未选择")
            self.workspace_state.setToolTip(
                f"Agent 私有数据位置：{private_root}\n"
                "尚未创建任何目录"
            )
            return
        workspace = snapshot.workspace
        truncated = " · 索引已截断" if snapshot.truncated else ""
        self.workspace_state.setText(
            f"工作区  {workspace.root.name or workspace.root} · "
            f"{len(snapshot.files)} 个文件{truncated}"
        )
        self.workspace_state.setToolTip(
            f"用户工作区：{workspace.root}\n"
            f"Agent 私有数据位置：{private_root}\n"
            "两者独立；选择工作区不会写入文件"
        )

    def _preview_send(self) -> None:
        text = self.input.toPlainText()
        if not text.strip():
            return
        self.messagePreviewRequested.emit(
            text,
            self.workspace_file_references,
        )
        self.set_busy_preview(True)

    def _show_preview_notice(self, text: str) -> None:
        self.composer_hint.setText(text)

    def _reset_preview(self) -> None:
        self.set_busy_preview(False)
        self.input.clear()
        self._workspace_references.clear()
        self._show_preview_notice("本地上下文预览 · 未创建会话")

    def set_busy_preview(self, busy: bool) -> None:
        """切换发送/停止按钮，且不启动任何后台工作。"""
        self._busy_preview = bool(busy)
        self.input.setEnabled(not self._busy_preview)
        self.send_state.setCurrentWidget(
            self.stop_button if self._busy_preview else self.send_button
        )
        if self._busy_preview:
            self._show_preview_notice(
                "运行状态预览 · 消息未发送，点击停止返回"
            )
        else:
            self._show_preview_notice("Enter 发送 · Shift+Enter 换行")
        self.send_button.setEnabled(
            bool(self.input.toPlainText().strip())
            and not self._busy_preview
        )


class AgentChatLauncher(_BoundaryToolButton):
    """聊天框关闭后留在模型画布右上角的覆盖式入口。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("agentChatLauncher")
        self.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground,
            True,
        )
        self.setText("FA")
        self.setToolTip("打开 FEM Agent")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.setMask(
            QRegion(
                self.rect(),
                QRegion.RegionType.Ellipse,
            )
        )


class ModelViewportOverlayHost(QWidget):
    """只叠放 Qt 控件、不改变模型视口布局或计算的宿主。"""

    drawerOpenChanged = Signal(bool)
    drawerWidthChanged = Signal(int)

    DEFAULT_DRAWER_WIDTH = 384
    MIN_DRAWER_WIDTH = 300
    LAUNCHER_MARGIN = 12
    ANIMATION_DURATION_MS = 180

    def __init__(
        self,
        viewport: QWidget,
        parent: QWidget | None = None,
        *,
        workspace_commands: WorkspaceCommandHandler | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("modelViewportOverlayHost")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setStyleSheet(_AGENT_CHAT_STYLESHEET)

        self.viewport = viewport
        base_layout = QVBoxLayout(self)
        base_layout.setContentsMargins(0, 0, 0, 0)
        base_layout.setSpacing(0)
        base_layout.addWidget(self.viewport)

        self.agent_chat_drawer = AgentChatDrawer(
            self,
            workspace_commands=workspace_commands,
        )
        self.chat_launcher = AgentChatLauncher(self)
        self.chat_launcher.clicked.connect(
            lambda: self.set_drawer_open(True)
        )
        self.agent_chat_drawer.closeRequested.connect(
            lambda: self.set_drawer_open(False)
        )
        self.agent_chat_drawer.resize_handle.dragDelta.connect(
            self._resize_drawer_by
        )

        self._drawer_width = self.DEFAULT_DRAWER_WIDTH
        self._drawer_reveal = 0
        self._drawer_open = True
        self._animation = QPropertyAnimation(
            self,
            b"drawerReveal",
            self,
        )
        self._animation.setDuration(self.ANIMATION_DURATION_MS)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.finished.connect(self._animation_finished)

        self.chat_launcher.hide()
        self.agent_chat_drawer.show()
        self._settle_overlay_geometry()

    def _get_drawer_reveal(self) -> int:
        return self._drawer_reveal

    def _set_drawer_reveal(self, value: int) -> None:
        maximum = self._effective_drawer_width()
        self._drawer_reveal = max(0, min(int(value), maximum))
        self._position_overlays()

    drawerReveal = Property(
        int,
        _get_drawer_reveal,
        _set_drawer_reveal,
    )

    @property
    def drawer_is_open(self) -> bool:
        return self._drawer_open

    @property
    def drawer_width(self) -> int:
        """返回当前宿主尺寸下聊天框实际可见的目标宽度。"""
        return self._effective_drawer_width()

    def set_drawer_open(
        self,
        opened: bool,
        *,
        animated: bool = True,
    ) -> None:
        """拉出或收起聊天框；仅改变覆盖控件自身的状态。"""
        opened = bool(opened)
        target = self._effective_drawer_width() if opened else 0
        if (
            opened == self._drawer_open
            and self._animation.state()
            != QAbstractAnimation.State.Running
            and self._drawer_reveal == target
        ):
            return

        self._animation.stop()
        self._drawer_open = opened
        if opened:
            self.agent_chat_drawer.show()
            self.agent_chat_drawer.raise_()
            self.chat_launcher.hide()

        self.drawerOpenChanged.emit(opened)
        if (
            not animated
            or not self.isVisible()
            or self._drawer_reveal == target
        ):
            self._set_drawer_reveal(target)
            self._animation_finished()
            return

        self._animation.setStartValue(self._drawer_reveal)
        self._animation.setEndValue(target)
        self._animation.start()

    def set_drawer_width(self, width: int) -> None:
        """设置用户偏好宽度；宿主较窄时只裁剪聊天框自身。"""
        width = max(self.MIN_DRAWER_WIDTH, int(width))
        if width == self._drawer_width:
            return
        self._animation.stop()
        self._drawer_width = width
        if self._drawer_open:
            self._drawer_reveal = self._effective_drawer_width()
        else:
            self._drawer_reveal = 0
        self._position_overlays()
        self.drawerWidthChanged.emit(self.drawer_width)

    def _resize_drawer_by(self, delta: int) -> None:
        self.set_drawer_width(self._drawer_width + int(delta))

    def _effective_drawer_width(self) -> int:
        return min(self._drawer_width, max(0, self.width()))

    def _settle_overlay_geometry(self) -> None:
        self._animation.stop()
        self._drawer_reveal = (
            self._effective_drawer_width() if self._drawer_open else 0
        )
        self._position_overlays()
        self._animation_finished()

    def _position_overlays(self) -> None:
        drawer_width = self._effective_drawer_width()
        drawer_x = self.width() - self._drawer_reveal
        self.agent_chat_drawer.setGeometry(
            QRect(drawer_x, 0, drawer_width, self.height())
        )

        launcher_size = self.chat_launcher.sizeHint()
        launcher_width = max(34, launcher_size.width())
        launcher_height = max(34, launcher_size.height())
        launcher_x = max(
            0,
            self.width() - launcher_width - self.LAUNCHER_MARGIN,
        )
        launcher_y = min(
            self.LAUNCHER_MARGIN,
            max(0, self.height() - launcher_height),
        )
        self.chat_launcher.setGeometry(
            QRect(
                launcher_x,
                launcher_y,
                min(launcher_width, self.width()),
                min(launcher_height, self.height()),
            )
        )
        if self._drawer_open or self._drawer_reveal:
            self.agent_chat_drawer.raise_()
        else:
            self.chat_launcher.raise_()

    def _animation_finished(self) -> None:
        if self._drawer_open:
            self.agent_chat_drawer.show()
            self.agent_chat_drawer.raise_()
            self.chat_launcher.hide()
        else:
            self.agent_chat_drawer.hide()
            self.chat_launcher.show()
            self.chat_launcher.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._settle_overlay_geometry()
