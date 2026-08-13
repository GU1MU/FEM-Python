"""FEM Agent 的覆盖式聊天界面与结构化事件展示。

工作区文件候选使用有界元数据索引；文件内容只由后台 Agent 适配器按限制
读取。对话展示只消费结构化事件投影状态。
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QPoint,
    Property,
    QPropertyAnimation,
    QRect,
    QSize,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QFocusEvent,
    QKeyEvent,
    QMouseEvent,
    QRegion,
    QTextCursor,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..agent_authoring import (
    AgentProposal,
    AuthoringWorkflowController,
    AuthoringWorkflowStage,
    ProposalState,
)

from ..agent_events import (
    AgentEvent,
    AgentEventProjector,
    ConfirmationView,
    DiagnosticView,
    EventType,
    MessageStatus,
    MessageView,
    ProposalView,
    ProposalViewStatus,
    SessionPresentation,
    TimelineKind,
    ToolGroupView,
    ToolStatus,
    TurnStatus,
)
from ..agent_runtime import QtAgentRuntime
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


def _shutdown_runtime_safely(
    runtime: QtAgentRuntime,
    *,
    wait: bool = False,
) -> None:
    try:
        runtime.shutdown(wait=wait)
    except RuntimeError:
        # Python may already be finalizing its thread executors.
        pass


_AGENT_CHAT_ICON_ROOT = (
    Path(__file__).resolve().parents[1] / "resources" / "icons"
)
_AGENT_CHAT_SCROLL_UP_ARROW = (
    _AGENT_CHAT_ICON_ROOT / "agent_chat_scroll_up.svg"
).as_posix()
_AGENT_CHAT_SCROLL_DOWN_ARROW = (
    _AGENT_CHAT_ICON_ROOT / "agent_chat_scroll_down.svg"
).as_posix()


_AGENT_CHAT_STYLESHEET = """
QFrame#agentChatDrawer {
    background: #ffffff;
    border-left: 1px solid #c7ccd2;
    font-size: 9.5pt;
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
    font-size: 12pt;
    font-weight: 600;
}
QLabel#agentChatMuted,
QLabel#agentChatComposerHint, QLabel#agentChatToolMeta {
    color: #6c7680;
    font-size: 8.5pt;
}
QToolButton#agentChatHeaderButton {
    color: #4a545e;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    min-width: 28px;
    max-width: 28px;
    min-height: 28px;
    max-height: 28px;
    padding: 0;
    font-size: 13pt;
}
QToolButton#agentChatHeaderButton:hover {
    background: #edf2f5;
    border-color: #d7dde2;
}
QScrollArea#agentChatScroll {
    background: #ffffff;
    border: none;
}
QWidget#agentChatScrollViewport,
QWidget#agentChatConversation,
QWidget#agentChatEventFeed,
QWidget#agentChatUserRow {
    background: #ffffff;
}
QFrame#agentChatDrawer QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 12px 0 12px 0;
}
QFrame#agentChatDrawer QScrollBar::handle:vertical {
    background: rgba(76, 88, 98, 92);
    min-height: 34px;
    border-radius: 4px;
    margin: 1px 2px;
}
QFrame#agentChatDrawer QScrollBar::handle:vertical:hover {
    background: rgba(76, 88, 98, 138);
}
QFrame#agentChatDrawer QScrollBar::add-line:vertical,
QFrame#agentChatDrawer QScrollBar::sub-line:vertical {
    background: transparent;
    border: none;
    height: 12px;
    subcontrol-origin: margin;
}
QFrame#agentChatDrawer QScrollBar::sub-line:vertical {
    subcontrol-position: top;
}
QFrame#agentChatDrawer QScrollBar::add-line:vertical {
    subcontrol-position: bottom;
}
QFrame#agentChatDrawer QScrollBar::up-arrow:vertical {
    image: url("__AGENT_CHAT_SCROLL_UP_ARROW__");
    width: 8px;
    height: 6px;
}
QFrame#agentChatDrawer QScrollBar::down-arrow:vertical {
    image: url("__AGENT_CHAT_SCROLL_DOWN_ARROW__");
    width: 8px;
    height: 6px;
}
QFrame#agentChatDrawer QScrollBar::add-page:vertical,
QFrame#agentChatDrawer QScrollBar::sub-page:vertical {
    background: transparent;
}
QFrame#agentChatUserMessage {
    background: #e7eff5;
    border: 1px solid #d1e0eb;
    border-radius: 9px;
}
QLabel#agentChatUserLabel {
    color: #29333c;
    font-size: 9.5pt;
}
QLabel#agentChatSpeaker {
    color: #334b5d;
    font-size: 9.5pt;
    font-weight: 600;
}
QLabel#agentChatAgentMessage {
    color: #28313a;
    font-size: 9.5pt;
}
QLabel#agentChatLiveActivity {
    color: #6f7f8b;
    font-size: 8.5pt;
    padding: 1px 0;
}
QFrame#agentChatToolActivity {
    background: #f2f4f6;
    border: 1px solid #e0e4e7;
    border-radius: 6px;
}
QToolButton#agentChatToolSummary {
    color: #707981;
    background: transparent;
    border: none;
    padding: 4px 0;
    text-align: left;
    font-size: 8.5pt;
}
QToolButton#agentChatToolSummary:hover {
    color: #45515b;
    background: #e8ecef;
}
QFrame#agentChatToolDetails {
    background: #e9edf0;
    border: 1px solid #dce2e6;
    border-radius: 5px;
}
QLabel#agentChatToolColumn {
    color: #879099;
    font-size: 8pt;
    font-weight: 600;
}
QLabel#agentChatToolValue {
    color: #68727b;
    font-size: 8pt;
}
QLabel#agentChatToolSuccess {
    color: #4f775f;
    font-size: 8pt;
}
QLabel#agentChatToolWarning {
    color: #8a681d;
    font-size: 8pt;
}
QLabel#agentChatToolFailure {
    color: #a14444;
    font-size: 8pt;
}
QLabel#agentChatToolDetail {
    color: #707981;
    font-size: 7.5pt;
}
QFrame#agentChatDiagnostic {
    background: #fff8e7;
    border: 1px solid #ead9a6;
    border-left: 3px solid #c99a2e;
    border-radius: 5px;
}
QLabel#agentChatDiagnosticTitle {
    color: #795e1c;
    font-size: 9.5pt;
    font-weight: 600;
}
QLabel#agentChatDiagnosticTitle[severity="error"],
QLabel#agentChatDiagnosticTitle[severity="blocking"] {
    color: #b42318;
}
QLabel#agentChatDiagnosticText {
    color: #20262d;
    font-size: 9.5pt;
    font-weight: 600;
}
QLabel#agentChatDiagnosticCode {
    color: #8a9299;
    font-size: 7.5pt;
    font-weight: 400;
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
    font-size: 9.5pt;
    font-weight: 600;
}
QLabel#agentChatConfirmationText {
    color: #435665;
    font-size: 9.5pt;
}
QToolButton#agentChatConfirmationButton {
    color: #ffffff;
    background: #477a9f;
    border: 1px solid #477a9f;
    border-radius: 5px;
    padding: 6px 14px;
    font-size: 9pt;
    font-weight: 600;
}
QToolButton#agentChatConfirmationButton:hover {
    background: #3c6b8d;
    border-color: #3c6b8d;
}
QToolButton#agentChatConfirmationButton:disabled {
    color: #7c8790;
    background: #e3e7ea;
    border-color: #d2d8dd;
}
QFrame#agentChatProposal {
    background: #f5f8fb;
    border: 1px solid #cad7e1;
    border-left: 3px solid #4c7fa5;
    border-radius: 6px;
}
QLabel#agentChatProposalTitle {
    color: #294b63;
    font-size: 9.5pt;
    font-weight: 600;
}
QLabel#agentChatProposalSummary {
    color: #344b5b;
    font-size: 9.5pt;
}
QLabel#agentChatProposalStatus {
    color: #55758c;
    font-size: 8.5pt;
}
QToolButton#agentChatProposalAcceptButton,
QToolButton#agentChatProposalRejectButton {
    border-radius: 5px;
    padding: 6px 18px;
    min-width: 52px;
    font-size: 9pt;
    font-weight: 600;
}
QToolButton#agentChatProposalAcceptButton {
    color: #ffffff;
    background: #477a9f;
    border: 1px solid #477a9f;
}
QToolButton#agentChatProposalAcceptButton:hover {
    background: #3c6b8d;
    border-color: #3c6b8d;
}
QToolButton#agentChatProposalAcceptButton:pressed {
    background: #315d7c;
    border-color: #315d7c;
}
QToolButton#agentChatProposalRejectButton {
    color: #3f5666;
    background: #ffffff;
    border: 1px solid #9eacb7;
}
QToolButton#agentChatProposalRejectButton:hover {
    color: #263d4d;
    background: #edf2f5;
    border-color: #718591;
}
QToolButton#agentChatProposalRejectButton:pressed {
    background: #dfe8ed;
}
QFrame#agentChatAppliedPatch {
    background: #f5f8fb;
    border: 1px solid #cad7e1;
    border-radius: 5px;
}
QLabel#agentChatAppliedPatchText {
    color: #294b63;
    font-size: 9pt;
    font-weight: 600;
}
QToolButton#agentChatPatchUndoButton {
    color: #315d7c;
    background: #ffffff;
    border: 1px solid #8faec2;
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 8.5pt;
    font-weight: 600;
}
QToolButton#agentChatPatchUndoButton:hover {
    color: #244f6d;
    background: #e8f0f6;
    border-color: #6f98b2;
}
QToolButton#agentChatPatchUndoButton:pressed {
    background: #d9e7f0;
}
QToolButton#agentChatPatchUndoButton:disabled {
    color: #8a949c;
    background: #eceff1;
    border-color: #d4dadd;
}
QToolButton#agentChatProposalAcceptButton:disabled,
QToolButton#agentChatProposalRejectButton:disabled {
    color: #8a949c;
    background: #eceff1;
    border-color: #d9dee2;
}
QLabel#agentChatTurnStatus {
    color: #a14444;
    background: #fff0f0;
    border: 1px solid #e5c2c2;
    border-radius: 4px;
    padding: 5px 7px;
    font-size: 8.5pt;
}
QFrame#agentChatComposer {
    background: #ffffff;
    border: none;
    border-top: 1px solid #dfe3e7;
}
QFrame#agentChatComposerSurface {
    background: #ffffff;
    border: 1px solid #cbd2d8;
    border-radius: 10px;
}
QFrame#agentChatComposerSurface[focused="true"] {
    border-color: #4c7fa5;
}
QFrame#agentChatComposerTaskSurface {
    background: #f7f9fb;
    border: 1px solid #cbd2d8;
    border-radius: 10px;
}
QLabel#agentChatComposerTaskTitle {
    color: #294b63;
    font-size: 9.5pt;
    font-weight: 600;
}
QLabel#agentChatComposerTaskSummary,
QLabel#agentChatComposerTaskImpact,
QLabel#agentChatComposerTaskStatus {
    color: #344b5b;
    font-size: 9pt;
}
QProgressBar#agentChatComposerProgress {
    min-height: 5px;
    max-height: 5px;
    border: none;
    border-radius: 2px;
    background: #dfe6eb;
    text-align: center;
}
QProgressBar#agentChatComposerProgress::chunk {
    border-radius: 2px;
    background: #4c7fa5;
}
QLabel#agentChatWorkspaceState {
    color: #66717b;
    background: #f3f5f7;
    border: 1px solid #dde1e5;
    border-radius: 6px;
    padding: 2px 7px;
    font-size: 8.5pt;
}
QFrame#agentChatSuggestion {
    background: #ffffff;
    border: 1px solid #ced5db;
    border-radius: 5px;
}
QLabel#agentChatSuggestionTitle {
    color: #66717b;
    font-size: 8.5pt;
    font-weight: 600;
}
QListWidget#agentChatSuggestionList {
    color: #34434f;
    background: #ffffff;
    border: none;
    outline: none;
    font-size: 9pt;
}
QListWidget#agentChatSuggestionList::item {
    padding: 6px 7px;
    border-radius: 3px;
}
QListWidget#agentChatSuggestionList::item:selected {
    color: #263946;
    background: #e8f0f6;
}
QPlainTextEdit#agentChatInput {
    color: #20262d;
    background: transparent;
    border: none;
    padding: 6px 7px 2px 7px;
    font-size: 10pt;
    selection-background-color: #dce9f2;
}
QPlainTextEdit#agentChatInput:disabled {
    color: #8f979e;
    background: transparent;
}
QToolButton#agentChatSendButton, QToolButton#agentChatStopButton {
    border-radius: 6px;
    min-width: 30px;
    max-width: 30px;
    min-height: 30px;
    max-height: 30px;
    padding: 0;
    font-size: 13pt;
}
QToolButton#agentChatAddButton {
    color: #4c5963;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    padding: 0;
    font-size: 12pt;
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
    font-size: 9pt;
}
QMenu#agentChatAddMenu::item {
    padding: 8px 26px 8px 11px;
}
QMenu#agentChatAddMenu::item:selected {
    background: #e8f0f6;
}
QToolButton#agentChatLauncher {
    color: #365d78;
    background: #ffffff;
    border: 1px solid #c4d1da;
    border-radius: 19px;
    min-width: 38px;
    max-width: 38px;
    min-height: 38px;
    max-height: 38px;
    padding: 0;
    font-size: 12pt;
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
QFrame#agentChatResizePreview {
    background: #4f88ad;
    border: none;
}
""".replace(
    "__AGENT_CHAT_SCROLL_UP_ARROW__",
    _AGENT_CHAT_SCROLL_UP_ARROW,
).replace(
    "__AGENT_CHAT_SCROLL_DOWN_ARROW__",
    _AGENT_CHAT_SCROLL_DOWN_ARROW,
)


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


class _CurrentPageStack(QStackedWidget):
    """只用当前页面计算堆叠区的布局尺寸。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.currentChanged.connect(self._current_page_changed)

    def sizeHint(self) -> QSize:
        current = self.currentWidget()
        return (
            current.sizeHint()
            if current is not None
            else super().sizeHint()
        )

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _current_page_changed(self, _index: int) -> None:
        self.updateGeometry()


class _ChatInput(QPlainTextEdit):
    """在输入框内路由发送与候选键盘操作。"""

    COLLAPSED_HEIGHT = 44
    MAXIMUM_VISIBLE_LINES = 5

    submitRequested = Signal()
    suggestionMoveRequested = Signal(int)
    suggestionAcceptRequested = Signal()
    suggestionDismissRequested = Signal()
    focusChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._suggestions_active = False
        self._idle_placeholder = ""
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.setFixedHeight(self.COLLAPSED_HEIGHT)
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.document().documentLayout().documentSizeChanged.connect(
            self._fit_height_to_content
        )
        self.textChanged.connect(self._fit_height_to_content)

    def setPlaceholderText(self, placeholder_text: str) -> None:
        self._idle_placeholder = placeholder_text
        super().setPlaceholderText(
            "" if self.hasFocus() else self._idle_placeholder
        )

    def focusInEvent(self, event: QFocusEvent) -> None:
        super().setPlaceholderText("")
        super().focusInEvent(event)
        self.focusChanged.emit(True)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        super().setPlaceholderText(self._idle_placeholder)
        self.focusChanged.emit(False)

    def _fit_height_to_content(self, *_args: object) -> None:
        document = self.document()
        document_layout = document.documentLayout()
        block = document.begin()
        visual_line_count = 0
        while block.isValid():
            document_layout.blockBoundingRect(block)
            visual_line_count += max(block.layout().lineCount(), 1)
            block = block.next()
        visible_lines = min(
            max(visual_line_count, 1),
            self.MAXIMUM_VISIBLE_LINES,
        )
        target_height = self.COLLAPSED_HEIGHT + (
            visible_lines - 1
        ) * self.fontMetrics().lineSpacing()
        scroll_policy = (
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
            if visual_line_count > self.MAXIMUM_VISIBLE_LINES
            else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        if self.verticalScrollBarPolicy() != scroll_policy:
            self.setVerticalScrollBarPolicy(scroll_policy)
        if self.height() != target_height:
            self.setFixedHeight(target_height)

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
    dragStarted = Signal()
    dragPreviewChanged = Signal(int)
    dragFinished = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("agentChatResizeHandle")
        self.setFixedWidth(6)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self._press_global_x: int | None = None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_global_x = round(event.globalPosition().x())
            self.dragStarted.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._press_global_x is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            current_x = round(event.globalPosition().x())
            self.dragPreviewChanged.emit(self._press_global_x - current_x)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            self._press_global_x is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            current_x = round(event.globalPosition().x())
            delta = self._press_global_x - current_x
            self._press_global_x = None
            self.dragFinished.emit(delta)
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


_UNORDERED_LIST_PATTERN = re.compile(r"^\s{0,3}[-+*]\s+(.+)$")
_ORDERED_LIST_PATTERN = re.compile(r"^\s{0,3}(\d+)[.)]\s+(.+)$")
_TABLE_DELIMITER_PATTERN = re.compile(r"^:?-{3,}:?$")


def _restricted_inline_markdown_html(markdown: str) -> str:
    escaped = html.escape(markdown, quote=True)
    escaped = re.sub(
        r"`([^`\n]+)`",
        r"<span style='font-family:monospace'>\1</span>",
        escaped,
    )
    escaped = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def _split_markdown_table_row(line: str) -> tuple[str, ...] | None:
    """Split one pipe row while preserving escaped and inline-code pipes."""

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    in_code = False
    separators = 0
    for character in line.strip():
        if escaped:
            current.append(character)
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "`":
            in_code = not in_code
            current.append(character)
            continue
        if character == "|" and not in_code:
            cells.append("".join(current).strip())
            current = []
            separators += 1
            continue
        current.append(character)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    if separators == 0:
        return None
    if cells and not cells[0]:
        cells.pop(0)
    if cells and not cells[-1]:
        cells.pop()
    return tuple(cells) if cells else None


def _table_alignments(
    delimiter: tuple[str, ...] | None,
    column_count: int,
) -> tuple[str, ...] | None:
    if (
        delimiter is None
        or len(delimiter) != column_count
        or any(
            _TABLE_DELIMITER_PATTERN.fullmatch(cell.replace(" ", ""))
            is None
            for cell in delimiter
        )
    ):
        return None
    alignments: list[str] = []
    for cell in delimiter:
        normalized = cell.replace(" ", "")
        alignments.append(
            "center"
            if normalized.startswith(":") and normalized.endswith(":")
            else "right"
            if normalized.endswith(":")
            else "left"
        )
    return tuple(alignments)


def _restricted_table_html(
    header: tuple[str, ...],
    alignments: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> str:
    cell_style = (
        "border:1px solid #d6dce3;"
        "padding:4px 7px;"
        "vertical-align:top;"
    )
    rendered = [
        "<table width='100%' border='1' bordercolor='#d6dce3' "
        "cellspacing='0' cellpadding='0' "
        "style='border-collapse:collapse; margin-top:2px; "
        "margin-bottom:2px;'>",
        "<tr>",
    ]
    for cell, alignment in zip(header, alignments, strict=True):
        rendered.append(
            f"<th style='{cell_style}background-color:#f2f4f6;"
            f"text-align:{alignment};'>"
            f"{_restricted_inline_markdown_html(cell)}</th>"
        )
    rendered.append("</tr>")
    for row in rows:
        rendered.append("<tr>")
        for cell, alignment in zip(row, alignments, strict=True):
            rendered.append(
                f"<td style='{cell_style}text-align:{alignment};'>"
                f"{_restricted_inline_markdown_html(cell)}</td>"
            )
        rendered.append("</tr>")
    rendered.append("</table>")
    return "".join(rendered)


def _restricted_markdown_html(markdown: str) -> str:
    """把受限 Markdown 转成不含链接、图片或原始 HTML 的 Qt 富文本。"""
    rendered: list[str] = []
    active_list: str | None = None
    lines = markdown.split("\n")

    def close_list() -> None:
        nonlocal active_list
        if active_list is not None:
            rendered.append(f"</{active_list}>")
            active_list = None

    def drop_break_before_block() -> None:
        if rendered and rendered[-1] == "<br>":
            rendered.pop()

    index = 0
    while index < len(lines):
        line = lines[index]
        header = _split_markdown_table_row(line)
        delimiter = (
            _split_markdown_table_row(lines[index + 1])
            if index + 1 < len(lines)
            else None
        )
        alignments = (
            _table_alignments(delimiter, len(header))
            if header is not None
            else None
        )
        if header is not None and alignments is not None:
            close_list()
            drop_break_before_block()
            rows: list[tuple[str, ...]] = []
            index += 2
            while index < len(lines):
                row = _split_markdown_table_row(lines[index])
                if row is None or len(row) != len(header):
                    break
                rows.append(row)
                index += 1
            rendered.append(
                _restricted_table_html(header, alignments, rows)
            )
            continue

        unordered = _UNORDERED_LIST_PATTERN.match(line)
        ordered = _ORDERED_LIST_PATTERN.match(line)
        list_type = "ul" if unordered is not None else (
            "ol" if ordered is not None else None
        )
        if list_type is not None:
            if active_list != list_type:
                close_list()
                drop_break_before_block()
                rendered.append(
                    f"<{list_type} style='margin-top:2px; "
                    "margin-bottom:2px;'>"
                )
                active_list = list_type
            content = (
                unordered.group(1)
                if unordered is not None
                else ordered.group(2)
            )
            rendered.append(
                "<li>"
                + _restricted_inline_markdown_html(content)
                + "</li>"
            )
            index += 1
            continue

        close_list()
        if line.strip():
            rendered.append(_restricted_inline_markdown_html(line))
            rendered.append("<br>")
        index += 1

    close_list()
    if rendered and rendered[-1] == "<br>":
        rendered.pop()
    return "".join(rendered)


def _plain_label(text: str, parent: QWidget) -> QLabel:
    label = QLabel(parent)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setText(text)
    return label


def _tool_summary_text(group: ToolGroupView) -> str:
    if len(group.calls) == 1:
        call = group.calls[0]
        prefix = {
            ToolStatus.REQUESTED: "准备调用",
            ToolStatus.RUNNING: "正在执行",
            ToolStatus.COMPLETED: "已完成",
            ToolStatus.WARNING: "已完成，有警告",
            ToolStatus.FAILED: "调用失败",
            ToolStatus.CANCELLED: "已取消",
        }[call.status]
        return f"{prefix} · {call.display_name}"
    parts = [f"工具 {len(group.calls)}"]
    if group.completed_count:
        parts.append(f"完成 {group.completed_count}")
    if group.warning_count:
        parts.append(f"警告 {group.warning_count}")
    if group.failed_count:
        parts.append(f"失败 {group.failed_count}")
    if group.cancelled_count:
        parts.append(f"取消 {group.cancelled_count}")
    running = sum(
        call.status in {ToolStatus.REQUESTED, ToolStatus.RUNNING}
        for call in group.calls
    )
    if running:
        parts.append(f"进行中 {running}")
    return " · ".join(parts)


def _tool_summary_tooltip(group: ToolGroupView) -> str:
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
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(3)

        self.summary_button = _BoundaryToolButton(self)
        self.summary_button.setObjectName("agentChatToolSummary")
        self.summary_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.summary_button.setCheckable(True)
        self.summary_button.setChecked(False)
        self.summary_button.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self.summary_button.toggled.connect(self._set_expanded)
        layout.addWidget(
            self.summary_button,
            0,
            Qt.AlignmentFlag.AlignLeft,
        )

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
        self.summary_button.setToolTip(
            _tool_summary_tooltip(self._group)
        )


class AgentChatDrawer(_BoundaryFrame):
    """覆盖在模型画布之上的 FEM Agent 聊天面板。"""

    closeRequested = Signal()
    messagePreviewRequested = Signal(str, object)
    messageSubmitted = Signal(str, object)
    agentEventApplied = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        workspace_commands: WorkspaceCommandHandler | None = None,
        agent_runtime: QtAgentRuntime | None = None,
        authoring_bridge: object | None = None,
        authoring_controller: AuthoringWorkflowController | None = None,
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
        self._shutting_down = False
        self._runtime_busy = False
        self._composer_proposal: tuple[ProposalView, str] | None = None
        self._manual_composer_proposal: tuple[ProposalView, str] | None = None
        self._continuation_active = False
        self._composer_accepting_id: str | None = None
        self._composer_task_was_visible = False
        self._rendering_event_presentation = False
        self._expanded_tool_group_ids: set[str] = set()
        self._pending_solve_confirmations: set[tuple[int, str]] = set()
        self._completed_solve_confirmations: set[tuple[int, str]] = set()
        self._applied_patch_records: dict[str, object] = {}
        self.event_projector = AgentEventProjector()
        self._message_widgets: dict[str, QLabel] = {}
        self._tool_group_widgets: dict[str, ToolActivityPreview] = {}
        self._diagnostic_widgets: dict[str, QWidget] = {}
        self._pending_message_refreshes: set[str] = set()
        self._stream_refresh_timer = QTimer(self)
        self._stream_refresh_timer.setSingleShot(True)
        self._stream_refresh_timer.setInterval(30)
        self._stream_refresh_timer.timeout.connect(
            self._flush_streaming_message_updates
        )
        self.authoring_bridge = authoring_bridge
        self._project_save_handler: (
            Callable[[Callable[[ProposalState, str], None]], bool] | None
        ) = None
        self.workspace_commands = (
            workspace_commands or WorkspaceCommandHandler()
        )
        self.agent_runtime = agent_runtime or QtAgentRuntime(
            self.workspace_commands.agent_data_root,
            self,
            authoring_controller=authoring_controller,
        )
        patch_listener = getattr(
            self.authoring_bridge,
            "set_patch_listener",
            None,
        )
        if callable(patch_listener):
            patch_listener(self.show_applied_patch)
        lifecycle_listener = getattr(
            self.authoring_bridge,
            "set_lifecycle_listener",
            None,
        )
        if callable(lifecycle_listener):
            lifecycle_listener(self._record_bridge_proposal_lifecycle)
        self._workspace_index = self.workspace_commands.workspace_index
        self._workspace_references: list[
            WorkspaceFileReference
        ] = []
        self._conversation_auto_follow = True
        self._conversation_scroll_suspended = False
        self._conversation_scroll_update_pending = False
        self._conversation_scroll_restore_value: int | None = None
        self._conversation_scroll_timer = QTimer(self)
        self._conversation_scroll_timer.setSingleShot(True)
        self._conversation_scroll_timer.timeout.connect(
            self._apply_queued_conversation_scroll
        )
        self._continuation_settle_timer = QTimer(self)
        self._continuation_settle_timer.setSingleShot(True)
        self._continuation_settle_timer.timeout.connect(
            self._settle_continuation_state
        )
        self._composer_focus_timer = QTimer(self)
        self._composer_focus_timer.setSingleShot(True)
        self._composer_focus_timer.timeout.connect(self._focus_composer_input)
        self._composer_accept_timer = QTimer(self)
        self._composer_accept_timer.setSingleShot(True)
        self._composer_accept_timer.timeout.connect(
            self._settle_composer_accepting
        )
        self._live_activity_label: QLabel | None = None
        self._live_activity_base = ""
        self._live_activity_tick = 0
        self._live_activity_timer = QTimer(self)
        self._live_activity_timer.setInterval(320)
        self._live_activity_timer.timeout.connect(
            self._animate_live_activity
        )
        self._live_activity_timer.start()

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
        self.agent_runtime.agentEventReady.connect(
            self._apply_runtime_event,
            Qt.ConnectionType.QueuedConnection,
        )
        self.agent_runtime.sessionReset.connect(
            self._reset_runtime_session,
            Qt.ConnectionType.QueuedConnection,
        )
        self.agent_runtime.busyChanged.connect(
            self.set_runtime_busy,
            Qt.ConnectionType.QueuedConnection,
        )
        self.agent_runtime.solveFinished.connect(
            self._solve_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self.agent_runtime.operationRejected.connect(
            self._show_runtime_notice,
            Qt.ConnectionType.QueuedConnection,
        )
        self.agent_runtime.eventRejected.connect(
            self._show_runtime_notice,
            Qt.ConnectionType.QueuedConnection,
        )
        runtime = self.agent_runtime
        self.destroyed.connect(
            lambda _object=None: _shutdown_runtime_safely(runtime, wait=False)
        )
        application = QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self._application_quit)
        self._update_workspace_state()
        self._render_event_presentation(preserve_tool_expansion=False)

    def set_project_save_handler(
        self,
        handler: Callable[
            [Callable[[ProposalState, str], None]],
            bool,
        ],
    ) -> None:
        if not callable(handler):
            raise TypeError("project save handler must be callable")
        self._project_save_handler = handler

    def _build_header(self, parent: QWidget) -> QWidget:
        header = _BoundaryFrame(parent)
        header.setObjectName("agentChatHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(14, 4, 8, 4)
        layout.setSpacing(6)

        title = QLabel("FEM Agent", header)
        title.setObjectName("agentChatTitle")
        layout.addWidget(title)

        layout.addStretch(1)

        self.close_button = _BoundaryToolButton(header)
        self.close_button.setObjectName("agentChatHeaderButton")
        self.close_button.setText("×")
        self.close_button.setToolTip("关闭聊天框")
        self.close_button.clicked.connect(self.closeRequested)
        layout.addWidget(self.close_button)
        return header

    def _build_conversation(self, parent: QWidget) -> QWidget:
        scroll = QScrollArea(parent)
        scroll.setObjectName("agentChatScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.viewport().setObjectName("agentChatScrollViewport")
        scroll.viewport().setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )

        conversation = QWidget(scroll)
        conversation.setObjectName("agentChatConversation")
        conversation.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )
        layout = QVBoxLayout(conversation)
        layout.setContentsMargins(6, 14, 14, 14)
        layout.setSpacing(12)

        self.event_feed = QWidget(conversation)
        self.event_feed.setObjectName("agentChatEventFeed")
        self.event_feed.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )
        self.event_feed_layout = QVBoxLayout(self.event_feed)
        self.event_feed_layout.setContentsMargins(0, 0, 0, 0)
        self.event_feed_layout.setSpacing(6)
        layout.addWidget(self.event_feed)
        layout.addStretch(1)

        scroll.setWidget(conversation)
        scroll_bar = scroll.verticalScrollBar()
        scroll_bar.valueChanged.connect(
            self._conversation_scroll_value_changed
        )
        scroll_bar.rangeChanged.connect(
            self._conversation_scroll_range_changed
        )
        self.conversation_widget = conversation
        self.conversation_layout = layout
        self.conversation_scroll = scroll
        return scroll

    def _conversation_is_near_bottom(self) -> bool:
        scroll_bar = self.conversation_scroll.verticalScrollBar()
        return scroll_bar.maximum() - scroll_bar.value() <= 24

    def _conversation_scroll_value_changed(self, _value: int) -> None:
        if not self._conversation_scroll_suspended:
            self._conversation_auto_follow = (
                self._conversation_is_near_bottom()
            )

    def _conversation_scroll_range_changed(
        self,
        _minimum: int,
        _maximum: int,
    ) -> None:
        if (
            self._conversation_auto_follow
            and not self._conversation_scroll_suspended
        ):
            self._queue_conversation_scroll()

    def _install_conversation_wheel_filters(self) -> None:
        widgets = (
            self.conversation_widget,
            *self.conversation_widget.findChildren(QWidget),
        )
        for widget in widgets:
            widget.installEventFilter(self)

    def _scroll_conversation_from_wheel(
        self,
        event: QWheelEvent,
    ) -> None:
        scroll_bar = self.conversation_scroll.verticalScrollBar()
        pixel_delta = event.pixelDelta().y()
        if pixel_delta:
            distance = pixel_delta
        else:
            angle_delta = event.angleDelta().y()
            line_step = max(
                scroll_bar.singleStep(),
                self.fontMetrics().lineSpacing(),
            )
            distance = round(
                angle_delta
                / 120
                * line_step
                * max(QApplication.wheelScrollLines(), 1)
            )
        if event.inverted():
            distance = -distance
        if distance:
            scroll_bar.setValue(scroll_bar.value() - distance)
        event.accept()

    def eventFilter(self, watched: object, event: object) -> bool:
        if (
            isinstance(watched, QWidget)
            and isinstance(event, QWheelEvent)
            and (
                watched is self.conversation_widget
                or self.conversation_widget.isAncestorOf(watched)
            )
        ):
            self._scroll_conversation_from_wheel(event)
            return True
        return super().eventFilter(watched, event)

    def _queue_conversation_scroll(
        self,
        restore_value: int | None = None,
    ) -> None:
        self._conversation_scroll_suspended = True
        self._conversation_scroll_restore_value = restore_value
        if self._conversation_scroll_update_pending:
            return
        self._conversation_scroll_update_pending = True
        self._conversation_scroll_timer.start(0)

    def _apply_queued_conversation_scroll(self) -> None:
        self._conversation_scroll_update_pending = False
        scroll_bar = self.conversation_scroll.verticalScrollBar()
        if self._conversation_auto_follow:
            scroll_bar.setValue(scroll_bar.maximum())
        elif self._conversation_scroll_restore_value is not None:
            scroll_bar.setValue(
                min(
                    self._conversation_scroll_restore_value,
                    scroll_bar.maximum(),
                )
            )
        self._conversation_scroll_restore_value = None
        self._conversation_scroll_suspended = False

    @property
    def event_presentation(self) -> SessionPresentation:
        """返回与 Qt 控件分离的结构化展示快照。"""
        return self.event_projector.presentation

    def replay_agent_events(
        self,
        events: Iterable[AgentEvent],
    ) -> None:
        """用完整事件日志替换当前展示并一次性重绘。"""
        self._stream_refresh_timer.stop()
        self._conversation_scroll_timer.stop()
        self._conversation_scroll_update_pending = False
        self._pending_message_refreshes.clear()
        self.event_projector = AgentEventProjector.replay(events)
        self.agent_runtime.synchronize_event_projection_from_gui(
            self.event_projector.presentation_view
        )
        self._expanded_tool_group_ids.clear()
        self._conversation_auto_follow = True
        latest = self._latest_projected_proposal()
        self._continuation_active = bool(
            latest is not None
            and latest[0].status
            in {
                ProposalViewStatus.SUCCEEDED,
                ProposalViewStatus.REJECTED,
                ProposalViewStatus.STALE,
                ProposalViewStatus.FAILED,
            }
            and self._runtime_busy
        )
        self._render_event_presentation(preserve_tool_expansion=False)

    def apply_agent_event(self, event: AgentEvent) -> None:
        """消费一个已验证事件；不解析 CLI 文本。"""
        self.event_projector.apply_in_place(event)
        if event.event_type is EventType.CONTINUATION_STARTED:
            self._continuation_active = True
        elif event.event_type in {
            EventType.PROPOSAL_SUCCEEDED,
            EventType.PROPOSAL_REJECTED,
            EventType.PROPOSAL_STALE,
            EventType.PROPOSAL_FAILED,
        }:
            self._continuation_active = True
            self._continuation_settle_timer.start(0)
        if event.event_type in {
            EventType.PROPOSAL_ACCEPTED,
            EventType.PROPOSAL_STARTED,
            EventType.PROPOSAL_REJECTED,
            EventType.PROPOSAL_STALE,
            EventType.PROPOSAL_SUCCEEDED,
            EventType.PROPOSAL_FAILED,
            EventType.PROPOSAL_CANCELLED,
        }:
            proposal_id = str(event.payload["proposal_id"])
            if self._composer_accepting_id == proposal_id:
                self._composer_accepting_id = None
        if event.event_type is EventType.MESSAGE_DELTA:
            self._pending_message_refreshes.add(
                str(event.payload["message_id"])
            )
            if not self._stream_refresh_timer.isActive():
                self._stream_refresh_timer.start()
        elif event.event_type is EventType.MESSAGE_COMPLETE:
            self._pending_message_refreshes.add(
                str(event.payload["message_id"])
            )
            self._flush_streaming_message_updates()
        else:
            self._flush_streaming_message_updates()
            self._render_event_presentation()
        self.agentEventApplied.emit(event)

    def _flush_streaming_message_updates(self) -> None:
        self._stream_refresh_timer.stop()
        if not self._pending_message_refreshes:
            return
        scroll_bar = self.conversation_scroll.verticalScrollBar()
        previous_scroll_value = scroll_bar.value()
        follow_latest = self._conversation_auto_follow
        message_ids = tuple(self._pending_message_refreshes)
        self._pending_message_refreshes.clear()
        for message_id in message_ids:
            label = self._message_widgets.get(message_id)
            if label is None:
                continue
            message = self.event_projector.message_view(message_id)
            self._update_agent_message_widget(label, message)
        self._queue_conversation_scroll(
            None if follow_latest else previous_scroll_value
        )

    def _clear_event_feed(
        self,
        *,
        preserve_tool_expansion: bool,
    ) -> None:
        self._live_activity_label = None
        self._message_widgets.clear()
        self._tool_group_widgets.clear()
        self._diagnostic_widgets.clear()
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
        scroll_bar = self.conversation_scroll.verticalScrollBar()
        previous_scroll_value = scroll_bar.value()
        follow_latest = self._conversation_auto_follow
        self._conversation_scroll_suspended = True
        self._clear_event_feed(
            preserve_tool_expansion=preserve_tool_expansion,
        )
        self._rendering_event_presentation = True
        for turn in self.event_projector.presentation_view.turns:
            if turn.user_message:
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
            proposals = {
                proposal.proposal_id: proposal
                for proposal in turn.proposals
            }
            deferred_proposal_ids: list[str] = []
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
                    self._tool_group_widgets[timeline_item.item_id] = tools
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
                elif timeline_item.kind is TimelineKind.PROPOSAL:
                    deferred_proposal_ids.append(timeline_item.item_id)
            if turn.status is TurnStatus.CANCELLED:
                status = _plain_label(
                    "本轮已取消 · " + turn.failure_reason,
                    self.event_feed,
                )
                status.setObjectName("agentChatTurnStatus")
                status.setWordWrap(True)
                self.event_feed_layout.addWidget(status)
            for proposal_id in deferred_proposal_ids:
                self._add_proposal_card(
                    proposals[proposal_id],
                    turn.turn_id,
                )
            if turn.status is TurnStatus.RUNNING:
                self._add_live_activity(turn)
        self._rendering_event_presentation = False
        for record in self._applied_patch_records.values():
            self._add_applied_patch_card(record)
        self._sync_composer_state()
        self._install_conversation_wheel_filters()
        self._conversation_auto_follow = follow_latest
        self._queue_conversation_scroll(
            None if follow_latest else previous_scroll_value
        )

    def _add_live_activity(self, turn: object) -> None:
        running_calls = [
            call
            for group in getattr(turn, "tool_groups", ())
            for call in group.calls
            if call.status in {
                ToolStatus.REQUESTED,
                ToolStatus.RUNNING,
            }
        ]
        messages = tuple(getattr(turn, "messages", ()))
        tool_groups = tuple(getattr(turn, "tool_groups", ()))
        if running_calls:
            base = f"正在执行 · {running_calls[-1].display_name}"
        elif any(
            message.status is MessageStatus.STREAMING
            for message in messages
        ):
            base = "正在生成回复"
        elif tool_groups:
            base = "正在整理工具结果"
        else:
            base = "正在分析请求"
        label = _plain_label("", self.event_feed)
        label.setObjectName("agentChatLiveActivity")
        self._live_activity_label = label
        self._live_activity_base = base
        self._update_live_activity_text()
        self.event_feed_layout.addWidget(
            label,
            0,
            Qt.AlignmentFlag.AlignLeft,
        )

    def _animate_live_activity(self) -> None:
        label = self._live_activity_label
        if label is None or not label.isVisible():
            return
        self._live_activity_tick = (self._live_activity_tick + 1) % 3
        self._update_live_activity_text()

    def _update_live_activity_text(self) -> None:
        label = self._live_activity_label
        if label is None:
            return
        dots = "·" * (self._live_activity_tick + 1)
        label.setText(f"●  {self._live_activity_base} {dots}")

    def show_applied_patch(self, record: object) -> None:
        """Show one local automatic edit with its revision-gated undo entry."""

        patch = getattr(record, "patch", None)
        patch_id = getattr(patch, "patch_id", None)
        if type(patch_id) is not str or not patch_id:
            raise TypeError("record must contain one applied ModelPatch")
        self._applied_patch_records.clear()
        self._applied_patch_records[patch_id] = record
        self._render_event_presentation()

    def _add_applied_patch_card(self, record: object) -> None:
        patch = getattr(record, "patch")
        patch_id = str(getattr(patch, "patch_id"))
        bridge = self.authoring_bridge
        if bridge is not None:
            getter = getattr(getattr(bridge, "port", None), "patch_record", None)
            if callable(getter):
                try:
                    record = getter(patch_id)
                    self._applied_patch_records[patch_id] = record
                except Exception:
                    pass
        summary = getattr(record, "display_summary", {})
        summary = summary if isinstance(summary, Mapping) else {}
        card = _BoundaryFrame(self.event_feed)
        card.setObjectName("agentChatAppliedPatch")
        card.setProperty("patchId", patch_id)
        card.setMaximumWidth(520)
        card.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )
        layout = QHBoxLayout(card)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(10)
        raw_title = str(
            summary.get("title", "Agent 已应用模型修改")
        )
        raw_detail = str(summary.get("summary", "")).strip()
        visible_title = (
            f"Agent {raw_detail}"
            if raw_title == "Agent 修改已同步" and raw_detail
            else raw_title
        )
        state_value = str(
            getattr(getattr(record, "state", None), "value", "")
        )
        if state_value == "undone":
            visible_title = "Agent 已撤销修改"
        elif state_value == "stale":
            visible_title = "Agent 修改已无法撤销"
        title = _plain_label(
            visible_title,
            card,
        )
        title.setObjectName("agentChatAppliedPatchText")
        layout.addWidget(title)
        if state_value and state_value != "applied":
            self.event_feed_layout.addWidget(
                card,
                0,
                Qt.AlignmentFlag.AlignLeft,
            )
            return
        undo = _BoundaryToolButton(card)
        undo.setObjectName("agentChatPatchUndoButton")
        undo.setProperty("patchId", patch_id)
        undo.setText("撤销修改")
        undo.setToolTip(
            str(summary.get("undo_label", "撤销本次 Agent 修改"))
        )
        undo.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        undo.setEnabled(
            not self._runtime_busy
            and bridge is not None
            and bool(bridge.can_undo_patch(patch_id))
        )
        undo.clicked.connect(
            lambda _checked=False, value=patch_id: (
                self._undo_applied_patch(value)
            )
        )
        layout.addWidget(
            undo,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        self.event_feed_layout.addWidget(
            card,
            0,
            Qt.AlignmentFlag.AlignLeft,
        )

    def _undo_applied_patch(self, patch_id: str) -> None:
        bridge = self.authoring_bridge
        if bridge is None:
            return
        try:
            record = bridge.undo_patch_from_gui_control(patch_id)
        except Exception as error:
            self._show_runtime_notice(str(error))
        else:
            self._applied_patch_records[patch_id] = record
            self._show_preview_notice("已撤销 Agent 修改")
        self._render_event_presentation()

    def _add_user_message(self, text: str, turn_id: str) -> None:
        user_row = QWidget(self.event_feed)
        user_row.setObjectName("agentChatUserRow")
        user_row.setProperty("turnId", turn_id)
        user_layout = QHBoxLayout(user_row)
        user_layout.setContentsMargins(34, 0, 0, 0)
        user_layout.setSpacing(0)
        user_layout.addStretch(0)
        user_bubble = _BoundaryFrame(user_row)
        user_bubble.setObjectName("agentChatUserMessage")
        bubble_layout = QVBoxLayout(user_bubble)
        bubble_layout.setContentsMargins(10, 7, 10, 7)
        user_text = _plain_label(text, user_bubble)
        user_text.setObjectName("agentChatUserLabel")
        user_text.setWordWrap(True)
        # 气泡宽度随内容自适应：以最宽行的文本宽度加内边距作为上限，
        # 短消息收窄为内容宽度，长消息仍占满可用宽度并自动换行。
        user_text.ensurePolished()
        metrics = user_text.fontMetrics()
        natural_width = max(
            (
                metrics.horizontalAdvance(line)
                for line in text.splitlines()
            ),
            default=0,
        )
        user_bubble.setMinimumWidth(48)
        user_bubble.setMaximumWidth(max(natural_width + 26, 48))
        bubble_layout.addWidget(user_text)
        user_layout.addWidget(user_bubble, 1)
        self.event_feed_layout.addWidget(user_row)

    def _add_agent_message(self, message: MessageView) -> None:
        label = QLabel(self.event_feed)
        label.setObjectName("agentChatAgentMessage")
        label.setProperty("messageId", message.message_id)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setOpenExternalLinks(False)
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._update_agent_message_widget(label, message)
        self._message_widgets[message.message_id] = label
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

    @staticmethod
    def _update_agent_message_widget(
        label: QLabel,
        message: MessageView,
    ) -> None:
        if message.status is MessageStatus.STREAMING:
            rendered_text = html.escape(
                message.text,
                quote=True,
            ).replace("\n", "<br>") + " ▌"
        else:
            rendered_text = _restricted_markdown_html(message.text)
        label.setProperty("messageStatus", message.status.value)
        label.setText(rendered_text)

    def _add_diagnostic_card(self, diagnostic: DiagnosticView) -> None:
        card = _BoundaryFrame(self.event_feed)
        card.setObjectName("agentChatDiagnostic")
        card.setProperty("diagnosticId", diagnostic.diagnostic_id)
        card.setProperty("severity", diagnostic.severity.value)
        self._diagnostic_widgets[diagnostic.diagnostic_id] = card
        card.setMaximumWidth(520)
        card.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(9, 7, 9, 7)
        layout.setSpacing(2)
        title = _plain_label(diagnostic.title, card)
        title.setObjectName("agentChatDiagnosticTitle")
        title.setProperty("severity", diagnostic.severity.value)
        layout.addWidget(title)
        text = _plain_label(diagnostic.message, card)
        text.setObjectName("agentChatDiagnosticText")
        text.setWordWrap(True)
        layout.addWidget(text)
        if diagnostic.code:
            code = _plain_label(diagnostic.code, card)
            code.setObjectName("agentChatDiagnosticCode")
            layout.addWidget(code)
        self.event_feed_layout.addWidget(
            card,
            0,
            Qt.AlignmentFlag.AlignLeft,
        )

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
        key = (
            confirmation.revision,
            confirmation.revision_hash,
        )
        button = _BoundaryToolButton(card)
        button.setObjectName("agentChatConfirmationButton")
        if key in self._completed_solve_confirmations:
            button.setText("求解已完成")
            button.setEnabled(False)
        elif key in self._pending_solve_confirmations:
            button.setText("正在求解…")
            button.setEnabled(False)
        else:
            button.setText("开始求解")
            button.setEnabled(
                not self._runtime_busy
                and self._confirmation_targets_live_session()
            )
            button.clicked.connect(
                lambda _checked=False, item=confirmation: (
                    self._confirm_runtime_solve(item)
                )
            )
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

    def _add_proposal_card(
        self,
        proposal: ProposalView,
        turn_id: str,
    ) -> None:
        if not self._rendering_event_presentation:
            self._manual_composer_proposal = (proposal, turn_id)
            self._sync_composer_state()
        if proposal.status is ProposalViewStatus.PENDING_CONFIRMATION:
            return

        card = _BoundaryFrame(self.event_feed)
        card.setObjectName("agentChatProposal")
        card.setProperty("proposalId", proposal.proposal_id)
        card.setProperty("proposalHash", proposal.proposal_hash)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(9, 7, 9, 7)
        layout.setSpacing(4)

        title = _plain_label(proposal.title, card)
        title.setObjectName("agentChatProposalTitle")
        layout.addWidget(title)
        summary = _plain_label(proposal.summary, card)
        summary.setObjectName("agentChatProposalSummary")
        summary.setWordWrap(True)
        layout.addWidget(summary)
        status = proposal.status
        status_labels = {
            ProposalViewStatus.PENDING_CONFIRMATION: "等待 GUI 确认",
            ProposalViewStatus.ACCEPTED: "已接受",
            ProposalViewStatus.REJECTED: "已拒绝",
            ProposalViewStatus.STALE: "提案已陈旧",
            ProposalViewStatus.RUNNING: "正在执行",
            ProposalViewStatus.SUCCEEDED: "已完成",
            ProposalViewStatus.FAILED: "执行失败",
            ProposalViewStatus.CANCELLED: "已取消",
        }
        status_text = status_labels[status]
        if (
            status is not ProposalViewStatus.SUCCEEDED
            and proposal.status_message
        ):
            status_text += f" · {proposal.status_message}"
        state_label = _plain_label(status_text, card)
        state_label.setObjectName("agentChatProposalStatus")
        layout.addWidget(state_label)

        actions = QWidget(card)
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 2, 0, 0)
        actions_layout.setSpacing(6)
        accept = _BoundaryToolButton(actions)
        accept.setObjectName("agentChatProposalHistoryAcceptButton")
        accept.setText("确认")
        accept.setToolTip(proposal.confirm_label)
        accept.setProperty("proposalId", proposal.proposal_id)
        accept.setProperty("proposalHash", proposal.proposal_hash)
        accept.setProperty("proposalKind", proposal.proposal_kind)
        accept.setProperty(
            "targetDocumentId",
            proposal.target_document_id,
        )
        accept.setProperty(
            "targetSessionId",
            proposal.target_session_id,
        )
        accept.setProperty(
            "baseSessionRevision",
            proposal.base_session_revision,
        )
        accept.setEnabled(False)
        actions_layout.addWidget(accept)

        reject = _BoundaryToolButton(actions)
        reject.setObjectName("agentChatProposalHistoryRejectButton")
        reject.setText("拒绝")
        reject.setProperty("proposalId", proposal.proposal_id)
        reject.setProperty("proposalHash", proposal.proposal_hash)
        reject.setProperty("proposalKind", proposal.proposal_kind)
        reject.setProperty(
            "targetDocumentId",
            proposal.target_document_id,
        )
        reject.setProperty(
            "targetSessionId",
            proposal.target_session_id,
        )
        reject.setProperty(
            "baseSessionRevision",
            proposal.base_session_revision,
        )
        reject.setEnabled(False)
        actions_layout.addWidget(reject)
        actions_layout.addStretch(1)
        layout.addWidget(actions)
        actions.hide()
        self.event_feed_layout.addWidget(card)

    def _build_composer(self, parent: QWidget) -> QWidget:
        composer = _BoundaryFrame(parent)
        composer.setObjectName("agentChatComposer")
        layout = QVBoxLayout(composer)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(5)

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
        self.suggestion_list.setMaximumHeight(150)
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

        self.composer_surface = _BoundaryFrame(composer)
        self.composer_surface.setObjectName("agentChatComposerSurface")
        self.composer_surface.setProperty("focused", False)
        surface_layout = QVBoxLayout(self.composer_surface)
        surface_layout.setContentsMargins(5, 4, 5, 5)
        surface_layout.setSpacing(3)

        self.input = _ChatInput(self.composer_surface)
        self.input.setObjectName("agentChatInput")
        self.input.setPlaceholderText(
            "询问 FEM Agent；使用 @ 引用工作区文件…"
        )
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
        self.input.focusChanged.connect(self._set_composer_surface_focused)
        surface_layout.addWidget(self.input)

        footer = QHBoxLayout()
        footer.setContentsMargins(2, 0, 0, 0)
        footer.setSpacing(4)

        self.add_button = _BoundaryToolButton(self.composer_surface)
        self.add_button.setObjectName("agentChatAddButton")
        self.add_button.setText("＋")
        self.add_button.setFixedSize(24, 24)
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

        self.workspace_state = QLabel(
            "工作区  尚未选择",
            self.composer_surface,
        )
        self.workspace_state.setObjectName("agentChatWorkspaceState")
        self.workspace_state.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )
        footer.addWidget(self.workspace_state)

        self.composer_hint = QLabel("", self.composer_surface)
        self.composer_hint.setObjectName("agentChatComposerHint")
        self.composer_hint.hide()
        footer.addWidget(self.composer_hint)
        footer.addStretch(1)

        self.send_state = QStackedWidget(self.composer_surface)
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
        self.send_button.setToolTip("发送到配置的 FEM Agent Provider")
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(self._submit_current_input)
        self.stop_button = _BoundaryToolButton(self.send_state)
        self.stop_button.setObjectName("agentChatStopButton")
        self.stop_button.setText("■")
        self.stop_button.setToolTip("取消当前 Agent 操作")
        self.stop_button.clicked.connect(self._cancel_runtime_operation)
        self.send_state.addWidget(self.send_button)
        self.send_state.addWidget(self.stop_button)
        self.send_state.setCurrentWidget(self.send_button)
        footer.addWidget(self.send_state)
        surface_layout.addLayout(footer)
        self.composer_stack = _CurrentPageStack(composer)
        self.composer_stack.setObjectName("agentChatComposerStack")
        self.composer_stack.addWidget(self.composer_surface)
        self.composer_task_surface = _BoundaryFrame(self.composer_stack)
        self.composer_task_surface.setObjectName("agentChatComposerTaskSurface")
        task_layout = QVBoxLayout(self.composer_task_surface)
        task_layout.setContentsMargins(10, 8, 10, 8)
        task_layout.setSpacing(4)
        self.composer_task_title = _plain_label("", self.composer_task_surface)
        self.composer_task_title.setObjectName("agentChatComposerTaskTitle")
        task_layout.addWidget(self.composer_task_title)
        self.composer_task_summary = _plain_label("", self.composer_task_surface)
        self.composer_task_summary.setObjectName("agentChatComposerTaskSummary")
        self.composer_task_summary.setWordWrap(True)
        task_layout.addWidget(self.composer_task_summary)
        self.composer_task_impact = _plain_label("", self.composer_task_surface)
        self.composer_task_impact.setObjectName("agentChatComposerTaskImpact")
        self.composer_task_impact.setWordWrap(True)
        task_layout.addWidget(self.composer_task_impact)
        self.composer_progress = QProgressBar(self.composer_task_surface)
        self.composer_progress.setObjectName("agentChatComposerProgress")
        self.composer_progress.setTextVisible(False)
        self.composer_progress.setRange(0, 100)
        task_layout.addWidget(self.composer_progress)
        self.composer_task_status = _plain_label("", self.composer_task_surface)
        self.composer_task_status.setObjectName("agentChatComposerTaskStatus")
        self.composer_task_status.setWordWrap(True)
        task_layout.addWidget(self.composer_task_status)
        task_actions = QHBoxLayout()
        task_actions.setContentsMargins(0, 2, 0, 0)
        task_actions.setSpacing(6)
        self.composer_accept_button = _BoundaryToolButton(
            self.composer_task_surface
        )
        self.composer_accept_button.setObjectName(
            "agentChatProposalAcceptButton"
        )
        self.composer_accept_button.clicked.connect(
            self._accept_composer_proposal
        )
        task_actions.addWidget(self.composer_accept_button)
        self.composer_reject_button = _BoundaryToolButton(
            self.composer_task_surface
        )
        self.composer_reject_button.setObjectName(
            "agentChatProposalRejectButton"
        )
        self.composer_reject_button.setText("拒绝")
        self.composer_reject_button.clicked.connect(
            self._reject_composer_proposal
        )
        task_actions.addWidget(self.composer_reject_button)
        self.composer_stop_button = _BoundaryToolButton(
            self.composer_task_surface
        )
        self.composer_stop_button.setObjectName("agentChatComposerStopButton")
        self.composer_stop_button.setText("停止")
        self.composer_stop_button.clicked.connect(
            self._cancel_runtime_operation
        )
        task_actions.addWidget(self.composer_stop_button)
        task_actions.addStretch(1)
        task_layout.addLayout(task_actions)
        self.composer_stack.addWidget(self.composer_task_surface)
        self.composer_stack.setCurrentWidget(self.composer_surface)
        layout.addWidget(self.composer_stack)
        return composer

    def _set_composer_surface_focused(self, focused: bool) -> None:
        self.composer_surface.setProperty("focused", focused)
        style = self.composer_surface.style()
        style.unpolish(self.composer_surface)
        style.polish(self.composer_surface)
        self.composer_surface.update()

    def _latest_projected_proposal(
        self,
    ) -> tuple[ProposalView, str] | None:
        latest: tuple[ProposalView, str] | None = None
        for turn in self.event_projector.presentation_view.turns:
            for proposal in turn.proposals:
                latest = (proposal, turn.turn_id)
        return latest

    def _set_composer_proposal_properties(
        self,
        proposal: ProposalView,
    ) -> None:
        for button in (
            self.composer_accept_button,
            self.composer_reject_button,
        ):
            button.setProperty("proposalId", proposal.proposal_id)
            button.setProperty("proposalHash", proposal.proposal_hash)
            button.setProperty("proposalKind", proposal.proposal_kind)
            button.setProperty(
                "targetDocumentId",
                proposal.target_document_id,
            )
            button.setProperty(
                "targetSessionId",
                proposal.target_session_id,
            )
            button.setProperty(
                "baseSessionRevision",
                proposal.base_session_revision,
            )

    def _sync_composer_state(self) -> None:
        projected = self._latest_projected_proposal()
        if projected is not None:
            self._manual_composer_proposal = None
        manual = projected is None and self._manual_composer_proposal is not None
        latest = projected or self._manual_composer_proposal
        self._composer_proposal = latest
        proposal = None if latest is None else latest[0]
        turn_id = None if latest is None else latest[1]

        pending = (
            proposal is not None
            and proposal.status is ProposalViewStatus.PENDING_CONFIRMATION
            and proposal.proposal_id != self._composer_accepting_id
        )
        local = (
            proposal is not None
            and (
                proposal.status
                in {ProposalViewStatus.ACCEPTED, ProposalViewStatus.RUNNING}
                or proposal.proposal_id == self._composer_accepting_id
            )
        )
        continuing = self._continuation_active and not pending and not local
        show_task = pending or local or continuing
        self.composer_stack.setCurrentWidget(
            self.composer_task_surface if show_task else self.composer_surface
        )
        self.suggestion.setVisible(
            self.suggestion.isVisible() and not show_task
        )

        if not show_task:
            if self._composer_task_was_visible and self.isVisible():
                self._composer_focus_timer.start(0)
            self._composer_task_was_visible = False
            return
        self._composer_task_was_visible = True

        self.composer_accept_button.setVisible(pending)
        self.composer_reject_button.setVisible(pending)
        self.composer_stop_button.setVisible(
            (local or continuing) and self._runtime_busy
        )
        self.composer_progress.setVisible(local or continuing)
        self.composer_task_summary.setVisible(not pending)
        self.composer_task_status.setVisible(not pending)
        self.composer_task_impact.hide()

        if pending and proposal is not None and turn_id is not None:
            self.composer_task_title.setText(proposal.title)
            self.composer_task_summary.clear()
            self.composer_task_impact.clear()
            self.composer_task_status.clear()
            self.composer_accept_button.setText(proposal.confirm_label)
            self.composer_accept_button.setToolTip(proposal.confirm_label)
            self._set_composer_proposal_properties(proposal)
            enabled = (
                not self._runtime_busy
                and self._proposal_targets_live_binding(
                    proposal,
                    None if manual else turn_id,
                )
            )
            self.composer_accept_button.setEnabled(enabled)
            self.composer_reject_button.setEnabled(enabled)
            return

        self.composer_accept_button.setEnabled(False)
        self.composer_reject_button.setEnabled(False)
        self.composer_task_impact.clear()
        if local and proposal is not None:
            self.composer_task_title.setText(proposal.title)
            self.composer_task_summary.setText(proposal.summary)
            progress = max(0, min(100, round(proposal.progress * 100)))
            if proposal.status is ProposalViewStatus.ACCEPTED and progress == 0:
                self.composer_progress.setRange(0, 0)
            else:
                self.composer_progress.setRange(0, 100)
                self.composer_progress.setValue(progress)
            self.composer_task_status.setText(
                (
                    "正在启动本地任务…"
                    if proposal.proposal_id == self._composer_accepting_id
                    else proposal.status_message or "正在执行本地任务…"
                )
            )
            return

        self.composer_task_title.setText("FEM Agent 正在续跑")
        self.composer_task_summary.setText(
            "本地任务已进入终态，Agent 正在继续下一阶段。"
        )
        self.composer_progress.setRange(0, 0)
        self.composer_task_status.setText(
            "可等待下一项确认，或在需要时停止 Agent。"
        )

    def _accept_composer_proposal(self) -> None:
        if self._composer_proposal is None:
            return
        proposal, turn_id = self._composer_proposal
        if proposal.status is not ProposalViewStatus.PENDING_CONFIRMATION:
            return
        self._composer_accepting_id = proposal.proposal_id
        self._sync_composer_state()
        self._accept_authoring_proposal(proposal, turn_id)
        self._composer_accept_timer.start(0)

    def _reject_composer_proposal(self) -> None:
        if self._composer_proposal is None:
            return
        proposal, turn_id = self._composer_proposal
        if proposal.status is not ProposalViewStatus.PENDING_CONFIRMATION:
            return
        self._reject_authoring_proposal(proposal, turn_id)
        self._sync_composer_state()

    def _settle_composer_accepting(self) -> None:
        if self._shutting_down:
            return
        if self._composer_accepting_id is None:
            return
        proposal = None
        if self._composer_proposal is not None:
            proposal = self._composer_proposal[0]
        if (
            proposal is None
            or proposal.status is ProposalViewStatus.PENDING_CONFIRMATION
        ):
            self._composer_accepting_id = None
        self._sync_composer_state()

    def _settle_continuation_state(self) -> None:
        if self._shutting_down:
            return
        if not self.agent_runtime.busy:
            self._continuation_active = False
            self._sync_composer_state()

    def _focus_composer_input(self) -> None:
        if not self._shutting_down:
            self.input.setFocus()

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
            bool(text.strip()) and not self._runtime_busy
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
        self.suggestion_title.show()
        self.suggestion_list.clear()
        item = QListWidgetItem("/workspace  选择工作区")
        item.setData(Qt.ItemDataRole.UserRole, "/workspace")
        self.suggestion_list.addItem(item)
        self.suggestion_item = item
        self.suggestion_list.setCurrentRow(0)
        self._show_suggestions()

    def _show_workspace_suggestions(self, query: str) -> None:
        self.suggestion_title.clear()
        self.suggestion_title.hide()
        self.suggestion_list.clear()
        snapshot = self._workspace_index
        if snapshot is None:
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
        self._send_to_runtime()

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
            self._show_preview_notice("")
        elif result.cancelled:
            self._show_preview_notice("已取消选择，工作区保持不变")
        else:
            self._show_preview_notice(
                f"无法选择工作区：{result.error or '未知错误'}"
            )
        return result

    def _update_workspace_state(self) -> None:
        snapshot = self._workspace_index
        if snapshot is None:
            self.workspace_state.setText("工作区  尚未选择")
            self.workspace_state.setToolTip(
                "Agent 私有会话目录尚未创建；其位置不会在界面中披露"
            )
            return
        workspace = snapshot.workspace
        self.workspace_state.setText(
            f"工作区  {workspace.root.name or workspace.root}"
        )
        self.workspace_state.setToolTip(
            f"用户工作区：{workspace.root}\n"
            "Agent 私有数据与用户工作区相互独立；"
            "选择工作区不会写入文件"
        )

    def _send_to_runtime(self) -> None:
        text = self.input.toPlainText()
        if not text.strip():
            return
        references = self.workspace_file_references
        workspace_root = (
            None
            if self._workspace_index is None
            else self._workspace_index.workspace.root
        )
        if not self.agent_runtime.send_message(
            text,
            references,
            workspace_root=workspace_root,
        ):
            return
        self._applied_patch_records.clear()
        self._render_event_presentation()
        self._conversation_auto_follow = True
        self._queue_conversation_scroll()
        self.messagePreviewRequested.emit(
            text,
            references,
        )
        self.messageSubmitted.emit(text, references)
        self.input.clear()
        self._workspace_references.clear()
        self._hide_suggestions()
        self.set_runtime_busy(True)

    def _show_preview_notice(self, text: str) -> None:
        notice = text.strip()
        self.composer_hint.setText(notice)
        self.composer_hint.setVisible(bool(notice))

    def _cancel_runtime_operation(self) -> None:
        if self.agent_runtime.cancel():
            self._show_preview_notice("正在取消当前操作…")

    def _confirm_runtime_solve(
        self,
        confirmation: ConfirmationView,
    ) -> None:
        key = (
            confirmation.revision,
            confirmation.revision_hash,
        )
        if self.agent_runtime.confirm_solve(*key):
            self._pending_solve_confirmations.add(key)
            self._show_preview_notice("已确认当前 revision，正在开始求解…")
            self.set_runtime_busy(True)
            self._render_event_presentation(
                preserve_tool_expansion=True,
            )

    def _accept_authoring_proposal(
        self,
        proposal: ProposalView,
        turn_id: str,
    ) -> None:
        bridge = self.authoring_bridge
        if bridge is None:
            return
        if proposal.proposal_kind == "project_save":
            controller = self.agent_runtime.authoring_controller
            context = bridge.context
            handler = self._project_save_handler
            if (
                controller is None
                or context is None
                or handler is None
                or not self._proposal_targets_live_binding(proposal, turn_id)
            ):
                self._show_preview_notice("保存请求已陈旧，请重新生成")
                return
            try:
                controller.begin_project_save_from_gui(
                    proposal.proposal_id,
                    proposal.proposal_hash,
                    context,
                )
            except Exception as exc:
                self._show_preview_notice(
                    str(exc).strip() or "保存请求无法接受"
                )
                return
            self._record_projected_proposal_lifecycle(
                proposal,
                turn_id,
                ProposalState.RUNNING,
                "等待本地保存完成",
            )
            try:
                started = handler(
                    lambda state, message, item=proposal, item_turn=turn_id: (
                        self._finish_project_save(
                            item,
                            state,
                            message,
                            item_turn,
                        )
                    )
                )
            except Exception:
                started = False
            if not started and proposal.status is ProposalViewStatus.RUNNING:
                self._finish_project_save(
                    proposal,
                    ProposalState.FAILED,
                    "保存任务未能启动",
                )
            return
        if proposal.proposal_kind == "requirement_review":
            if not self._proposal_targets_live_binding(proposal, turn_id):
                self._show_preview_notice("需求审查已陈旧，请重新生成")
                return
            controller = self.agent_runtime.authoring_controller
            review = (
                None
                if controller is None
                else controller.pending_review
            )
            if not self._requirement_review_is_current(
                proposal.proposal_id,
                proposal.proposal_hash,
            ):
                self._show_preview_notice("需求审查已陈旧，请重新生成")
                return
            try:
                confirmed = bridge.confirm_requirement_review_from_gui(
                    controller.ledger,
                    review,
                )
                self.agent_runtime.resolve_requirement_review_from_gui(
                    confirmed
                )
            except Exception as exc:
                self._show_preview_notice(
                    str(exc).strip() or "需求确认失败"
                )
            else:
                self._record_projected_proposal_lifecycle(
                    proposal,
                    turn_id,
                    ProposalState.SUCCEEDED,
                    "需求审查已确认",
                )
                self._show_preview_notice("需求审查已由 GUI 控件确认")
            return
        try:
            receipt = bridge.accept_from_gui_control(proposal.proposal_id)
        except Exception as exc:
            self._show_preview_notice(
                str(exc).strip() or "提案接受失败"
            )
        else:
            try:
                self.agent_runtime.record_authoring_proposal_state_from_gui(
                    proposal.proposal_kind,
                    receipt.state.value,
                    receipt.message,
                )
            except (RuntimeError, ValueError):
                pass
            self._show_preview_notice(
                receipt.message
                or (
                    "提案已由 GUI 控件接受；A1 Fake Port 未修改模型"
                    if receipt.state.value == "accepted"
                    else (
                        (
                            "几何已加入模型"
                            if proposal.proposal_kind == "geometry"
                            else (
                                "模型修改已完成"
                                if proposal.proposal_kind
                                == "destructive_edit"
                                else "提案已完成"
                            )
                        )
                        if receipt.state.value == "succeeded"
                        else "提案处理失败"
                    )
                )
            )

    def _reject_authoring_proposal(
        self,
        proposal: ProposalView,
        turn_id: str,
    ) -> None:
        bridge = self.authoring_bridge
        if bridge is None:
            return
        if proposal.proposal_kind == "project_save":
            controller = self.agent_runtime.authoring_controller
            context = bridge.context
            if (
                controller is None
                or context is None
                or not controller.can_accept_project_save_from_gui(
                    proposal.proposal_id,
                    proposal.proposal_hash,
                    context,
                )
            ):
                self._show_preview_notice("保存请求已陈旧，请重新生成")
                return
            controller.record_project_save_state(
                proposal.proposal_id,
                proposal.proposal_hash,
                ProposalState.REJECTED,
                "用户拒绝保存请求",
            )
            self._record_projected_proposal_lifecycle(
                proposal,
                turn_id,
                ProposalState.REJECTED,
                "用户拒绝保存请求",
            )
            self._show_preview_notice("保存请求已拒绝，未写入文件")
            return
        if proposal.proposal_kind == "requirement_review":
            if not self._proposal_targets_live_binding(proposal, turn_id):
                self._show_preview_notice("需求审查已陈旧，请重新生成")
                return
            controller = self.agent_runtime.authoring_controller
            review = (
                None
                if controller is None
                else controller.pending_review
            )
            if not self._requirement_review_is_current(
                proposal.proposal_id,
                proposal.proposal_hash,
            ):
                self._show_preview_notice("需求审查已陈旧，请重新生成")
                return
            try:
                rejected = bridge.reject_requirement_review_from_gui(
                    controller.ledger,
                    review,
                )
                self.agent_runtime.resolve_requirement_review_from_gui(
                    rejected
                )
            except Exception as exc:
                self._show_preview_notice(
                    str(exc).strip() or "需求拒绝失败"
                )
            else:
                self._record_projected_proposal_lifecycle(
                    proposal,
                    turn_id,
                    ProposalState.REJECTED,
                    "用户拒绝需求审查",
                )
                self._show_preview_notice("需求审查已拒绝，模型保持不变")
            return
        try:
            receipt = bridge.reject_from_gui_control(proposal.proposal_id)
        except Exception as exc:
            self._show_preview_notice(
                str(exc).strip() or "提案拒绝失败"
            )
        else:
            try:
                self.agent_runtime.record_authoring_proposal_state_from_gui(
                    proposal.proposal_kind,
                    receipt.state.value,
                    receipt.message,
                )
            except (RuntimeError, ValueError):
                pass
            self._show_preview_notice(
                receipt.message or "提案已拒绝，当前模型保持不变"
            )

    def _proposal_targets_live_binding(
        self,
        proposal: ProposalView,
        turn_id: str | None = None,
    ) -> bool:
        bridge = self.authoring_bridge
        if bridge is None or bridge.context is None:
            return False
        if turn_id is not None and proposal.proposal_kind not in {
            "project_save",
            "requirement_review",
        }:
            source_turn = self.agent_runtime.proposal_source_turn_from_gui(
                proposal.proposal_id,
                proposal.proposal_hash,
                self.event_projector.presentation_view.session_id,
                turn_id,
            )
            identity_check = getattr(
                bridge,
                "ensure_display_identity_from_gui",
                None,
            )
            if (
                source_turn is None
                or (
                    callable(identity_check)
                    and not identity_check(
                        proposal.proposal_id,
                        proposal.proposal_hash,
                        self.event_projector.presentation_view.session_id,
                        source_turn,
                    )
                )
            ):
                return False
        binding = bridge.context.binding
        if (
            not binding.supported
            or binding.document_id != proposal.target_document_id
            or binding.session_id != proposal.target_session_id
            or binding.session_revision != proposal.base_session_revision
        ):
            if turn_id is not None and proposal.proposal_kind in {
                "project_save",
                "requirement_review",
            }:
                self._record_projected_proposal_lifecycle(
                    proposal,
                    turn_id,
                    ProposalState.STALE,
                    "绑定文档、session 或 revision 已改变",
                )
            return False
        if proposal.proposal_kind == "requirement_review":
            current = self._requirement_review_is_current(
                proposal.proposal_id,
                proposal.proposal_hash,
            )
            if not current and turn_id is not None:
                self._record_projected_proposal_lifecycle(
                    proposal,
                    turn_id,
                    ProposalState.STALE,
                    "需求审查 identity 已改变",
                )
            return current
        if proposal.proposal_kind == "project_save":
            controller = self.agent_runtime.authoring_controller
            current = bool(
                controller is not None
                and controller.can_accept_project_save_from_gui(
                    proposal.proposal_id,
                    proposal.proposal_hash,
                    bridge.context,
                )
            )
            if not current and turn_id is not None and controller is not None:
                self._record_projected_proposal_lifecycle(
                    proposal,
                    turn_id,
                    ProposalState.STALE,
                    "保存请求 identity 已改变",
                )
            return current
        try:
            check = getattr(bridge, "can_accept_from_gui_control", None)
            if callable(check):
                return bool(check(proposal.proposal_id))
            return bridge.state(proposal.proposal_id).value == "pending_confirmation"
        except Exception:
            return False

    def refresh_authoring_binding(self) -> None:
        self._render_event_presentation(preserve_tool_expansion=True)

    def _finish_project_save(
        self,
        proposal: ProposalView,
        state: ProposalState | str,
        message: str,
        turn_id: str | None = None,
    ) -> None:
        normalized = ProposalState(state)
        controller = self.agent_runtime.authoring_controller
        if controller is not None:
            try:
                controller.record_project_save_state(
                    proposal.proposal_id,
                    proposal.proposal_hash,
                    normalized,
                    message,
                )
            except ValueError:
                record = controller.project_save_record
                normalized = (
                    record.state
                    if (
                        record is not None
                        and record.proposal_id == proposal.proposal_id
                        and record.proposal_hash == proposal.proposal_hash
                        and record.state
                        in {
                            ProposalState.SUCCEEDED,
                            ProposalState.FAILED,
                            ProposalState.CANCELLED,
                            ProposalState.STALE,
                            ProposalState.REJECTED,
                        }
                    )
                    else ProposalState.STALE
                )
        turn_id = turn_id or self._proposal_turn_id(proposal.proposal_id)
        if turn_id is not None:
            self._record_projected_proposal_lifecycle(
                proposal,
                turn_id,
                normalized,
                message,
            )
        notices = {
            ProposalState.SUCCEEDED: "自主项目保存完成",
            ProposalState.FAILED: "自主项目保存失败",
            ProposalState.CANCELLED: "已取消保存，未写入文件",
            ProposalState.STALE: "保存快照已陈旧，当前修改仍未保存",
            ProposalState.REJECTED: "保存请求已拒绝",
        }
        self._show_preview_notice(notices[normalized])

    def _record_bridge_proposal_lifecycle(
        self,
        proposal: AgentProposal,
        state: ProposalState,
        message: str,
    ) -> None:
        if self.agent_runtime.proposal_lifecycle_matches_from_gui(
            proposal.proposal_id,
            proposal.proposal_hash,
            proposal.agent_session_id,
            proposal.turn_id,
        ):
            try:
                self.agent_runtime.record_authoring_proposal_state_from_gui(
                    proposal.proposal_kind.value,
                    state,
                    message,
                )
            except (RuntimeError, ValueError):
                pass
        self.agent_runtime.record_proposal_lifecycle_from_gui(
            proposal.proposal_id,
            proposal.proposal_hash,
            proposal.agent_session_id,
            proposal.turn_id,
            state,
            message,
        )

    def _record_projected_proposal_lifecycle(
        self,
        proposal: ProposalView,
        turn_id: str,
        state: ProposalState,
        message: str,
    ) -> None:
        emitted = self.agent_runtime.record_proposal_lifecycle_from_gui(
            proposal.proposal_id,
            proposal.proposal_hash,
            self.event_projector.presentation_view.session_id,
            turn_id,
            state,
            message,
        )
        if not emitted:
            proposal.status = ProposalViewStatus(state.value)
            proposal.status_message = str(message).strip()

    def _proposal_turn_id(self, proposal_id: str) -> str | None:
        for turn in self.event_projector.presentation_view.turns:
            if any(
                proposal.proposal_id == proposal_id
                for proposal in turn.proposals
            ):
                return turn.turn_id
        return None

    def _solve_finished(
        self,
        revision: int,
        revision_hash: str,
        succeeded: bool,
    ) -> None:
        if self._shutting_down:
            return
        key = (revision, revision_hash)
        self._pending_solve_confirmations.discard(key)
        if succeeded:
            self._completed_solve_confirmations.add(key)
        self._render_event_presentation(
            preserve_tool_expansion=True,
        )

    def _reset_runtime_session(self, _session_id: str) -> None:
        if self._shutting_down:
            return
        bridge = self.authoring_bridge
        stale_pending = getattr(
            bridge,
            "stale_pending_proposals_from_gui",
            None,
        )
        if callable(stale_pending):
            stale_pending("Agent session changed")
        controller = self.agent_runtime.authoring_controller
        if controller is not None:
            controller.reset_for_binding()
            context = None if bridge is None else bridge.context
            if context is not None:
                controller.observe_binding(context)
        self._stream_refresh_timer.stop()
        self._pending_message_refreshes.clear()
        self.event_projector = AgentEventProjector()
        self._expanded_tool_group_ids.clear()
        self._pending_solve_confirmations.clear()
        self._completed_solve_confirmations.clear()
        self._applied_patch_records.clear()
        self._manual_composer_proposal = None
        self._composer_proposal = None
        self._composer_accepting_id = None
        self._continuation_active = False
        self._conversation_auto_follow = True
        self._render_event_presentation(preserve_tool_expansion=False)
        self.input.clear()
        self._workspace_references.clear()
        self._show_preview_notice("新的 Agent 会话已就绪")

    def _apply_runtime_event(self, event: AgentEvent) -> None:
        if self._shutting_down:
            return
        self.apply_agent_event(event)

    def _show_runtime_notice(self, text: str) -> None:
        if self._shutting_down:
            return
        self._show_preview_notice(text)

    def _confirmation_targets_live_session(self) -> bool:
        presentation_session = self.event_projector.presentation.session_id
        return (
            bool(presentation_session)
            and self.agent_runtime.session_id == presentation_session
        )

    def set_runtime_busy(self, _busy: bool) -> None:
        """投影后台 runtime 的串行操作状态。"""
        if self._shutting_down:
            return
        self._runtime_busy = self.agent_runtime.busy
        if not self._runtime_busy:
            self._continuation_active = False
        self.input.setEnabled(not self._runtime_busy)
        self.add_button.setEnabled(not self._runtime_busy)
        self.send_state.setCurrentWidget(
            self.stop_button if self._runtime_busy else self.send_button
        )
        if self._runtime_busy:
            self._show_preview_notice(
                "FEM Agent 正在后台运行 · 可点击停止"
            )
        else:
            self._show_preview_notice("")
        self.send_button.setEnabled(
            bool(self.input.toPlainText().strip())
            and not self._runtime_busy
        )
        for button in self.findChildren(
            QToolButton,
            "agentChatConfirmationButton",
        ):
            revision = button.property("revision")
            revision_hash = button.property("revisionHash")
            key = (revision, revision_hash)
            if key in self._completed_solve_confirmations:
                button.setText("求解已完成")
                button.setEnabled(False)
            elif key in self._pending_solve_confirmations:
                button.setText("正在求解…")
                button.setEnabled(False)
            else:
                button.setText("开始求解")
                button.setEnabled(
                    not self._runtime_busy
                    and self._confirmation_targets_live_session()
                )
        for button in self.findChildren(
            QToolButton,
            "agentChatProposalAcceptButton",
        ):
            button.setEnabled(
                not self._runtime_busy
                and self._proposal_button_targets_live_binding(button)
            )
        for button in self.findChildren(
            QToolButton,
            "agentChatProposalRejectButton",
        ):
            button.setEnabled(
                not self._runtime_busy
                and self._proposal_button_targets_live_binding(button)
            )
        for button in self.findChildren(
            QToolButton,
            "agentChatPatchUndoButton",
        ):
            bridge = self.authoring_bridge
            button.setEnabled(
                not self._runtime_busy
                and bridge is not None
                and bridge.can_undo_patch(str(button.property("patchId")))
            )
        self._sync_composer_state()

    def _proposal_button_targets_live_binding(
        self,
        button: QToolButton,
    ) -> bool:
        bridge = self.authoring_bridge
        if bridge is None or bridge.context is None:
            return False
        binding = bridge.context.binding
        if (
            not binding.supported
            or binding.document_id != button.property("targetDocumentId")
            or binding.session_id != button.property("targetSessionId")
            or binding.session_revision
            != button.property("baseSessionRevision")
        ):
            return False
        if button.property("proposalKind") == "requirement_review":
            return self._requirement_review_is_current(
                str(button.property("proposalId")),
                str(button.property("proposalHash")),
            )
        if button.property("proposalKind") == "project_save":
            controller = self.agent_runtime.authoring_controller
            return bool(
                controller is not None
                and controller.can_accept_project_save_from_gui(
                    str(button.property("proposalId")),
                    str(button.property("proposalHash")),
                    bridge.context,
                )
            )
        try:
            proposal_id = str(button.property("proposalId"))
            check = getattr(bridge, "can_accept_from_gui_control", None)
            if callable(check):
                return bool(check(proposal_id))
            return bridge.state(proposal_id).value == "pending_confirmation"
        except Exception:
            return False

    def _requirement_review_is_current(
        self,
        review_id: str,
        review_hash: str,
    ) -> bool:
        controller = self.agent_runtime.authoring_controller
        if (
            controller is None
            or controller.stage is not AuthoringWorkflowStage.REVIEW_PENDING
        ):
            return False
        pending = controller.pending_review
        return (
            pending is not None
            and pending.review_id == review_id
            and pending.review_hash == review_hash
        )

    def _application_quit(self) -> None:
        self.shutdown_runtime(wait=False)

    def shutdown_runtime(self, *, wait: bool = False) -> None:
        """安全关闭后台执行边界；收起聊天框不会调用本方法。"""
        if not self._shutting_down:
            self._shutting_down = True
            self._project_save_handler = None
            detach_callbacks = getattr(
                self.authoring_bridge,
                "detach_gui_callbacks",
                None,
            )
            if callable(detach_callbacks):
                detach_callbacks()
            for timer in (
                self._stream_refresh_timer,
                self._conversation_scroll_timer,
                self._continuation_settle_timer,
                self._composer_focus_timer,
                self._composer_accept_timer,
                self._live_activity_timer,
            ):
                timer.stop()
        self.agent_runtime.shutdown(wait=wait)


class AgentChatLauncher(_BoundaryToolButton):
    """聊天框关闭后留在模型画布上的可拖动覆盖式入口。"""

    dragDelta = Signal(QPoint)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("agentChatLauncher")
        self.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground,
            True,
        )
        self.setText("FA")
        self.setToolTip("点击打开 FEM Agent；拖动可移动")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._press_global_position: QPoint | None = None
        self._last_global_position: QPoint | None = None
        self._dragging = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            position = event.globalPosition().toPoint()
            self._press_global_position = position
            self._last_global_position = position
            self._dragging = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._press_global_position is not None
            and self._last_global_position is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            position = event.globalPosition().toPoint()
            if not self._dragging:
                distance = (
                    position - self._press_global_position
                ).manhattanLength()
                if distance < QApplication.startDragDistance():
                    event.accept()
                    return
                self._dragging = True
                self.setDown(False)
            delta = position - self._last_global_position
            self._last_global_position = position
            if not delta.isNull():
                self.dragDelta.emit(delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            self._press_global_position is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            was_dragging = self._dragging
            self._press_global_position = None
            self._last_global_position = None
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            if was_dragging:
                self.setDown(False)
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.setMask(
            QRegion(
                self.rect(),
                QRegion.RegionType.Ellipse,
            )
        )


class ModelViewportOverlayHost(QWidget):
    """叠放 Qt 控件，并为打开的聊天框预留模型视口宽度。"""

    drawerOpenChanged = Signal(bool)
    drawerWidthChanged = Signal(int)
    viewportGeometryCommitted = Signal()

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
        agent_runtime: QtAgentRuntime | None = None,
        authoring_bridge: object | None = None,
        authoring_controller: AuthoringWorkflowController | None = None,
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
        self._viewport_layout = base_layout
        self._viewport_reserved_width = 0

        self.agent_chat_drawer = AgentChatDrawer(
            self,
            workspace_commands=workspace_commands,
            agent_runtime=agent_runtime,
            authoring_bridge=authoring_bridge,
            authoring_controller=authoring_controller,
        )
        self.chat_launcher = AgentChatLauncher(self)
        self._launcher_position: QPoint | None = None
        self._bottom_overlay: QWidget | None = None
        self._bottom_overlay_visible = False
        self.agent_chat_drawer.layout().setSizeConstraint(
            QLayout.SizeConstraint.SetNoConstraint
        )
        tool_flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.agent_chat_drawer.setWindowFlags(tool_flags)
        self.chat_launcher.setWindowFlags(
            tool_flags | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        for overlay in (self.agent_chat_drawer, self.chat_launcher):
            overlay.setAttribute(
                Qt.WidgetAttribute.WA_ShowWithoutActivating,
                True,
            )
        self._drawer_resize_preview = _BoundaryFrame(self)
        self._drawer_resize_preview.setObjectName("agentChatResizePreview")
        self._drawer_resize_preview.setFixedWidth(2)
        self._drawer_resize_preview.setWindowFlags(
            tool_flags | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self._drawer_resize_preview.setAttribute(
            Qt.WidgetAttribute.WA_ShowWithoutActivating,
            True,
        )
        self._drawer_resize_preview.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        self.chat_launcher.clicked.connect(
            lambda: self.set_drawer_open(True, animated=False)
        )
        self.chat_launcher.dragDelta.connect(self._move_launcher_by)
        self.agent_chat_drawer.closeRequested.connect(
            lambda: self.set_drawer_open(False, animated=False)
        )
        self.agent_chat_drawer.resize_handle.dragStarted.connect(
            self._begin_drawer_resize
        )
        self.agent_chat_drawer.resize_handle.dragPreviewChanged.connect(
            self._preview_drawer_resize
        )
        self.agent_chat_drawer.resize_handle.dragFinished.connect(
            self._finish_drawer_resize
        )

        self._drawer_width = self.DEFAULT_DRAWER_WIDTH
        self._drawer_reveal = 0
        self._drawer_open = False
        self._drawer_resize_initial_width: int | None = None
        self._drawer_resize_delta = 0
        self._pending_drawer_width: int | None = None
        self._shutting_down = False
        self._anchor_window: QWidget | None = None
        self._animation = QPropertyAnimation(
            self,
            b"drawerReveal",
            self,
        )
        self._animation.setDuration(self.ANIMATION_DURATION_MS)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.finished.connect(self._animation_finished)
        self._overlay_sync_timer = QTimer(self)
        self._overlay_sync_timer.setSingleShot(True)
        self._overlay_sync_timer.timeout.connect(self._sync_overlay_windows)
        self._drawer_resize_commit_timer = QTimer(self)
        self._drawer_resize_commit_timer.setSingleShot(True)
        self._drawer_resize_commit_timer.timeout.connect(
            self._commit_drawer_resize
        )
        native_surface_updated = getattr(
            self.viewport,
            "nativeSurfaceUpdated",
            None,
        )
        if native_surface_updated is not None:
            native_surface_updated.connect(
                self._schedule_overlay_window_sync
            )

        self.chat_launcher.hide()
        self.agent_chat_drawer.hide()
        self._drawer_resize_preview.hide()
        self._settle_overlay_geometry()

    def closeEvent(self, event) -> None:
        self.shutdown(wait=False)
        super().closeEvent(event)

    def shutdown(self, *, wait: bool = False) -> None:
        """Stop native overlays and the Agent runtime before owner teardown."""

        if not self._shutting_down:
            self._shutting_down = True
            self._animation.stop()
            self._overlay_sync_timer.stop()
            self._cancel_drawer_resize()
        self.agent_chat_drawer.shutdown_runtime(wait=wait)
        self.agent_chat_drawer.hide()
        self.chat_launcher.hide()
        if self._bottom_overlay is not None:
            self._bottom_overlay.hide()
        if self._anchor_window is not None:
            self._anchor_window.removeEventFilter(self)
            self._anchor_window = None

    def _get_drawer_reveal(self) -> int:
        return self._drawer_reveal

    def _set_drawer_reveal(self, value: int) -> None:
        maximum = self._effective_drawer_width()
        reveal = max(0, min(int(value), maximum))
        if reveal == self._drawer_reveal:
            return
        self._drawer_reveal = reveal
        self._position_overlays(sync_visibility=False)

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
        animated: bool = False,
    ) -> None:
        """拉出或收起聊天框，并同步模型视口的可用宽度。"""
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
        self.drawerOpenChanged.emit(opened)
        if (
            not animated
            or not self.isVisible()
            or self._drawer_reveal == target
        ):
            self._commit_drawer_open_state(target)
            return

        if opened:
            self._sync_viewport_reservation()
            self.chat_launcher.hide()
            self._sync_overlay_window_visibility()
        self._animation.setStartValue(self._drawer_reveal)
        self._animation.setEndValue(target)
        self._animation.start()

    def _commit_drawer_open_state(self, reveal: int) -> None:
        """Apply one fast visibility transaction without native-window frames."""

        if not self._drawer_open:
            # Remove the native drawer before the VTK surface grows underneath it.
            self.agent_chat_drawer.hide()
        self._drawer_reveal = int(reveal)
        self._sync_viewport_reservation()
        self._position_overlays(sync_visibility=False)
        self._sync_overlay_window_visibility()

    def set_drawer_width(self, width: int) -> None:
        """设置用户偏好宽度；打开时同步提交模型视口宽度。"""
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
        self._sync_viewport_reservation()
        self.drawerWidthChanged.emit(self.drawer_width)

    def set_bottom_overlay(self, overlay: QWidget) -> None:
        """注册一个不占用模型视口布局的底部工具浮层。"""
        if overlay is self._bottom_overlay:
            return
        if self._bottom_overlay is not None:
            self._bottom_overlay.hide()
        self._bottom_overlay = overlay
        overlay.setParent(self)
        overlay.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        overlay.setAttribute(
            Qt.WidgetAttribute.WA_ShowWithoutActivating,
            True,
        )
        overlay_layout = overlay.layout()
        if overlay_layout is not None:
            overlay_layout.setSizeConstraint(
                QLayout.SizeConstraint.SetNoConstraint
            )
        overlay.hide()
        self._position_overlays()

    def set_bottom_overlay_visible(self, visible: bool) -> None:
        """显示或隐藏底部工具浮层，同时保持模型视口几何不变。"""
        visible = bool(visible)
        if visible == self._bottom_overlay_visible:
            return
        self._bottom_overlay_visible = visible
        self._position_overlays()

    def _begin_drawer_resize(self) -> None:
        self._drawer_resize_commit_timer.stop()
        self._pending_drawer_width = None
        self._drawer_resize_initial_width = self._drawer_width
        self._drawer_resize_delta = 0
        self._position_drawer_resize_preview()
        self._drawer_resize_preview.show()
        self._drawer_resize_preview.raise_()

    def _preview_drawer_resize(self, delta: int) -> None:
        if self._drawer_resize_initial_width is None:
            return
        self._drawer_resize_delta = int(delta)
        self._position_drawer_resize_preview()

    def _finish_drawer_resize(self, delta: int) -> None:
        initial_width = self._drawer_resize_initial_width
        if initial_width is None:
            return
        self._drawer_resize_initial_width = None
        self._drawer_resize_delta = 0
        self._drawer_resize_preview.hide()
        self._pending_drawer_width = initial_width + int(delta)
        self._drawer_resize_commit_timer.start(0)

    def _commit_drawer_resize(self) -> None:
        """Commit only after the preview window's hide event has settled."""

        width = self._pending_drawer_width
        self._pending_drawer_width = None
        if width is not None and not self._shutting_down:
            self.set_drawer_width(width)

    def _cancel_drawer_resize(self) -> None:
        self._drawer_resize_commit_timer.stop()
        self._pending_drawer_width = None
        self._drawer_resize_initial_width = None
        self._drawer_resize_delta = 0
        self._drawer_resize_preview.hide()

    def _position_drawer_resize_preview(self) -> None:
        initial_width = self._drawer_resize_initial_width
        if initial_width is None:
            return
        requested_width = max(
            self.MIN_DRAWER_WIDTH,
            initial_width + self._drawer_resize_delta,
        )
        preview_width = min(requested_width, max(0, self.width()))
        host_origin = self.mapToGlobal(QPoint(0, 0))
        self._drawer_resize_preview.setGeometry(
            host_origin.x() + self.width() - preview_width,
            host_origin.y(),
            2,
            self.height(),
        )

    def _move_launcher_by(self, delta: QPoint) -> None:
        if self._launcher_position is None:
            host_origin = self.mapToGlobal(QPoint(0, 0))
            self._launcher_position = (
                self.chat_launcher.geometry().topLeft() - host_origin
            )
        self._launcher_position = self._bounded_launcher_position(
            self._launcher_position + delta
        )
        self._position_overlays()

    def _bounded_launcher_position(self, position: QPoint) -> QPoint:
        return QPoint(
            max(
                0,
                min(
                    position.x(),
                    max(0, self.width() - self.chat_launcher.width()),
                ),
            ),
            max(
                0,
                min(
                    position.y(),
                    max(0, self.height() - self.chat_launcher.height()),
                ),
            ),
        )

    def _effective_drawer_width(self) -> int:
        return min(self._drawer_width, max(0, self.width()))

    def _sync_viewport_reservation(self) -> None:
        reserved_width = (
            self._effective_drawer_width() if self._drawer_open else 0
        )
        if reserved_width == self._viewport_reserved_width:
            return
        self._viewport_reserved_width = reserved_width
        self._viewport_layout.setContentsMargins(0, 0, reserved_width, 0)
        self._viewport_layout.activate()
        self.viewportGeometryCommitted.emit()

    def _settle_overlay_geometry(self) -> None:
        self._animation.stop()
        self._drawer_reveal = (
            self._effective_drawer_width() if self._drawer_open else 0
        )
        self._sync_viewport_reservation()
        self._position_overlays()
        self._animation_finished()

    def _position_overlays(self, *, sync_visibility: bool = True) -> None:
        host_origin = self.mapToGlobal(QPoint(0, 0))
        drawer_width = self._effective_drawer_width()
        drawer_x = (
            host_origin.x() + self.width() - self._drawer_reveal
        )
        drawer_geometry = QRect(
            drawer_x,
            host_origin.y(),
            drawer_width,
            self.height(),
        )
        if self.agent_chat_drawer.geometry() != drawer_geometry:
            self.agent_chat_drawer.setGeometry(drawer_geometry)
        if 0 < self._drawer_reveal < drawer_width:
            self.agent_chat_drawer.setMask(
                QRegion(
                    0,
                    0,
                    self._drawer_reveal,
                    self.height(),
                )
            )
        elif self._drawer_reveal >= drawer_width:
            self.agent_chat_drawer.clearMask()
        else:
            self.agent_chat_drawer.setMask(QRegion())

        if not self._drawer_open and self._drawer_reveal == 0:
            launcher_size = self.chat_launcher.sizeHint()
            requested_launcher_width = min(
                max(34, launcher_size.width()),
                self.width(),
            )
            requested_launcher_height = min(
                max(34, launcher_size.height()),
                self.height(),
            )
            self.chat_launcher.resize(
                requested_launcher_width,
                requested_launcher_height,
            )
            launcher_width = self.chat_launcher.width()
            launcher_height = self.chat_launcher.height()
            if self._launcher_position is None:
                launcher_position = QPoint(
                    max(
                        0,
                        self.width() - launcher_width - self.LAUNCHER_MARGIN,
                    ),
                    min(
                        self.LAUNCHER_MARGIN,
                        max(0, self.height() - launcher_height),
                    ),
                )
            else:
                self._launcher_position = self._bounded_launcher_position(
                    self._launcher_position
                )
                launcher_position = self._launcher_position
            self.chat_launcher.move(
                host_origin + launcher_position,
            )
        bottom_overlay = self._bottom_overlay
        if bottom_overlay is not None and self._bottom_overlay_visible:
            overlay_height = min(
                max(1, bottom_overlay.sizeHint().height()),
                self.height(),
            )
            overlay_width = max(
                0,
                self.width() - self._drawer_reveal,
            )
            bottom_overlay.setGeometry(
                QRect(
                    host_origin.x(),
                    host_origin.y() + self.height() - overlay_height,
                    overlay_width,
                    overlay_height,
                )
            )
        self._position_drawer_resize_preview()
        if sync_visibility:
            self._sync_overlay_window_visibility()

    def _refresh_anchor_window(self) -> None:
        anchor_window = self.window()
        if anchor_window is self._anchor_window:
            return
        if self._anchor_window is not None:
            self._anchor_window.removeEventFilter(self)
        self._anchor_window = anchor_window
        self._anchor_window.installEventFilter(self)

    def _window_group_is_active(self) -> bool:
        if (
            QApplication.applicationState()
            != Qt.ApplicationState.ApplicationActive
        ):
            return False
        active_window = QApplication.activeWindow()
        if active_window is None:
            return True
        group_windows = {
            self._anchor_window,
            self.agent_chat_drawer,
            self.chat_launcher,
            self._drawer_resize_preview,
            self._bottom_overlay,
        }
        current = active_window
        while current is not None:
            if current in group_windows:
                return True
            current = current.parentWidget()
        return False

    def _overlay_windows_can_show(self) -> bool:
        anchor_window = self._anchor_window
        return bool(
            not self._shutting_down
            and self.isVisible()
            and anchor_window is not None
            and anchor_window.isVisible()
            and not anchor_window.isMinimized()
            and self._window_group_is_active()
        )

    def _sync_overlay_window_visibility(self) -> None:
        if not self._overlay_windows_can_show():
            self.agent_chat_drawer.hide()
            self.chat_launcher.hide()
            self._drawer_resize_preview.hide()
            if self._bottom_overlay is not None:
                self._bottom_overlay.hide()
            return
        if (
            self._bottom_overlay is not None
            and self._bottom_overlay_visible
            and self._bottom_overlay.width() > 0
        ):
            self._bottom_overlay.show()
            self._bottom_overlay.raise_()
        elif self._bottom_overlay is not None:
            self._bottom_overlay.hide()
        if self._drawer_open or self._drawer_reveal:
            self.chat_launcher.hide()
            self.agent_chat_drawer.show()
            self.agent_chat_drawer.raise_()
        else:
            self.agent_chat_drawer.hide()
            self.chat_launcher.show()
            self.chat_launcher.raise_()

    def _schedule_overlay_window_sync(self) -> None:
        """在宿主或 VTK 更新后同步独立覆盖窗口。"""
        if self._shutting_down or self._overlay_sync_timer.isActive():
            return
        self._overlay_sync_timer.start(0)

    def _sync_overlay_windows(self) -> None:
        if self._shutting_down:
            return
        self._refresh_anchor_window()
        self._position_overlays()

    def _animation_finished(self) -> None:
        if self._shutting_down:
            return
        if not self._drawer_open:
            self._sync_viewport_reservation()
        self._sync_overlay_window_visibility()

    def eventFilter(self, watched: object, event: object) -> bool:
        if watched is self._anchor_window:
            event_type = event.type()
            if event_type in {
                QEvent.Type.Hide,
                QEvent.Type.Close,
            }:
                self.agent_chat_drawer.hide()
                self.chat_launcher.hide()
                self._drawer_resize_preview.hide()
                if self._bottom_overlay is not None:
                    self._bottom_overlay.hide()
            elif event_type in {
                QEvent.Type.Move,
                QEvent.Type.Resize,
                QEvent.Type.Show,
                QEvent.Type.WindowActivate,
                QEvent.Type.WindowDeactivate,
                QEvent.Type.WindowStateChange,
                QEvent.Type.ZOrderChange,
            }:
                self._schedule_overlay_window_sync()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._settle_overlay_geometry()

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        self._schedule_overlay_window_sync()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_anchor_window()
        self._schedule_overlay_window_sync()

    def hideEvent(self, event) -> None:
        self.agent_chat_drawer.hide()
        self.chat_launcher.hide()
        self._drawer_resize_preview.hide()
        if self._bottom_overlay is not None:
            self._bottom_overlay.hide()
        super().hideEvent(event)
