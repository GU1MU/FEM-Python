"""浅色工业软件主题。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QProxyStyle, QStyle, QStyleOption, QWidget


COLORS = {
    "background": "#f4f5f6",
    "chrome": "#f7f8f9",
    "menu": "#f7f8f9",
    "border": "#d1d5da",
    "soft_border": "#e2e5e8",
    "text": "#20262d",
    "muted": "#66717b",
    "disabled": "#a0a6ac",
    "hover": "#e8f0f6",
    "selected": "#e3edf4",
    "checked": "#edf3f7",
    "accent": "#4c7fa5",
}

_COMBO_DOWN_ARROW = (
    Path(__file__).with_name("resources") / "icons" / "combo_down_arrow.svg"
).resolve().as_posix()


class FEMStyle(QProxyStyle):
    """为较难辨认的原生控件提供清晰、稳定的绘制。"""

    CHECKBOX_SIZE = 16
    CHECKBOX_BORDER_WIDTH = 2.0

    def pixelMetric(
        self,
        metric: QStyle.PixelMetric,
        option: QStyleOption | None = None,
        widget: QWidget | None = None,
    ) -> int:
        if metric in (
            QStyle.PixelMetric.PM_IndicatorWidth,
            QStyle.PixelMetric.PM_IndicatorHeight,
        ):
            return self.CHECKBOX_SIZE
        return super().pixelMetric(metric, option, widget)

    def drawPrimitive(
        self,
        element: QStyle.PrimitiveElement,
        option: QStyleOption,
        painter: QPainter,
        widget: QWidget | None = None,
    ) -> None:
        if element in (
            QStyle.PrimitiveElement.PE_IndicatorCheckBox,
            QStyle.PrimitiveElement.PE_IndicatorItemViewItemCheck,
        ):
            self._draw_checkbox_indicator(option, painter)
            return
        if element == QStyle.PrimitiveElement.PE_IndicatorRadioButton:
            self._draw_radio_indicator(option, painter)
            return
        super().drawPrimitive(element, option, painter, widget)

    def _draw_checkbox_indicator(
        self,
        option: QStyleOption,
        painter: QPainter,
    ) -> None:
        state = option.state
        enabled = bool(state & QStyle.StateFlag.State_Enabled)
        hovered = bool(state & QStyle.StateFlag.State_MouseOver)
        focused = bool(state & QStyle.StateFlag.State_HasFocus)
        checked = bool(state & QStyle.StateFlag.State_On)
        partial = bool(state & QStyle.StateFlag.State_NoChange)

        side = min(self.CHECKBOX_SIZE, option.rect.width(), option.rect.height())
        left = option.rect.x() + (option.rect.width() - side) / 2
        top = option.rect.y() + (option.rect.height() - side) / 2
        box = QRectF(left + 1.25, top + 1.25, side - 2.5, side - 2.5)

        if not enabled:
            border = QColor("#aeb5bb")
            fill = QColor("#aeb8c0") if checked or partial else QColor("#f1f2f3")
        elif checked or partial:
            border = QColor("#315f82")
            fill = QColor("#3f759d") if hovered else QColor(COLORS["accent"])
        else:
            border = QColor(COLORS["accent"] if hovered or focused else "#65727d")
            fill = QColor("#f5f9fc" if hovered else "#ffffff")

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(border, self.CHECKBOX_BORDER_WIDTH))
        painter.setBrush(fill)
        painter.drawRoundedRect(box, 2.0, 2.0)

        if checked:
            check = QPainterPath(QPointF(left + 3.7, top + 8.1))
            check.lineTo(QPointF(left + 6.8, top + 11.1))
            check.lineTo(QPointF(left + 12.4, top + 4.8))
            check_pen = QPen(
                QColor("#f7f9fa" if enabled else "#ffffff"),
                2.2,
            )
            check_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            check_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(check_pen)
            painter.setBrush(QColor("transparent"))
            painter.drawPath(check)
        elif partial:
            partial_pen = QPen(
                QColor("#f7f9fa" if enabled else "#ffffff"),
                2.2,
            )
            partial_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(partial_pen)
            painter.drawLine(
                QPointF(left + 4.2, top + 8.0),
                QPointF(left + 11.8, top + 8.0),
            )
        painter.restore()

    def _draw_radio_indicator(
        self,
        option: QStyleOption,
        painter: QPainter,
    ) -> None:
        state = option.state
        enabled = bool(state & QStyle.StateFlag.State_Enabled)
        hovered = bool(state & QStyle.StateFlag.State_MouseOver)
        focused = bool(state & QStyle.StateFlag.State_HasFocus)
        checked = bool(state & QStyle.StateFlag.State_On)

        side = min(self.CHECKBOX_SIZE, option.rect.width(), option.rect.height())
        left = option.rect.x() + (option.rect.width() - side) / 2
        top = option.rect.y() + (option.rect.height() - side) / 2
        outer = QRectF(left + 1.25, top + 1.25, side - 2.5, side - 2.5)
        border = QColor(
            "#aeb5bb"
            if not enabled
            else COLORS["accent"]
            if hovered or focused or checked
            else "#65727d"
        )

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(border, self.CHECKBOX_BORDER_WIDTH))
        painter.setBrush(QColor("#f1f2f3" if not enabled else "#ffffff"))
        painter.drawEllipse(outer)
        if checked:
            dot = QRectF(
                outer.center().x() - 3.25,
                outer.center().y() - 3.25,
                6.5,
                6.5,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(
                QColor("#8d969e" if not enabled else COLORS["accent"])
            )
            painter.drawEllipse(dot)
        painter.restore()


def build_stylesheet() -> str:
    """返回紧凑的浅色 CAE 风格样式。"""
    return f"""
QMainWindow, QWidget {{ background: {COLORS['background']}; color: {COLORS['text']};
  font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif; font-size: 9pt; }}
QMenuBar {{ background: {COLORS['menu']}; border-bottom: 1px solid {COLORS['border']}; padding: 1px 5px; }}
QMenuBar::item {{ padding: 4px 9px; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {COLORS['hover']}; }}
QMenu {{ background: white; border: 1px solid {COLORS['border']}; padding: 3px; }}
QMenu::item {{ padding: 5px 28px 5px 10px; }}
QMenu::item:disabled {{ color: {COLORS['disabled']}; }}
QToolBar#viewportToolbar {{ background: {COLORS['chrome']}; border: none; border-bottom: 1px solid {COLORS['border']}; spacing: 1px; padding: 2px 4px; }}
QToolBar#viewportToolbar::separator {{ width: 1px; background: {COLORS['soft_border']}; margin: 3px 5px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 2px; padding: 2px; min-width: 26px; min-height: 24px; }}
QToolButton:hover {{ background: {COLORS['hover']}; border-color: {COLORS['border']}; }}
QToolButton:checked {{ background: {COLORS['checked']}; border-color: {COLORS['soft_border']}; border-bottom: 2px solid {COLORS['accent']}; }}
QToolButton:disabled {{ color: {COLORS['disabled']}; }}
QTreeWidget {{ background: white; border: none; show-decoration-selected: 1; }}
QTreeWidget::item {{ min-height: 23px; padding: 0 3px; }}
QTreeWidget::item:hover {{ background: {COLORS['hover']}; }}
QTreeWidget::item:selected {{ background: {COLORS['selected']}; color: {COLORS['text']}; }}
QHeaderView::section {{ background: {COLORS['chrome']}; border: none; border-bottom: 1px solid {COLORS['border']}; padding: 4px; }}
QSplitter::handle {{ background: {COLORS['border']}; width: 1px; }}
QStatusBar {{ background: #eef0f2; color: {COLORS['muted']}; border-top: 1px solid {COLORS['border']}; min-height: 20px; max-height: 22px; padding: 0 4px; }}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ background: transparent; padding: 0 5px; font-size: 8.5pt; }}
QFrame#statusSeparator {{ color: {COLORS['border']}; max-width: 1px; }}
QWidget#ribbonWidget {{ background: {COLORS['chrome']}; border-bottom: 1px solid {COLORS['border']}; }}
QTabBar#ribbonTabs {{ background: {COLORS['chrome']}; }}
QTabBar#ribbonTabs::tab {{ background: transparent; border: none; border-bottom: 2px solid transparent; min-width: 68px; height: 26px; padding: 0 10px; }}
QTabBar#ribbonTabs::tab:hover {{ background: {COLORS['hover']}; }}
QTabBar#ribbonTabs::tab:selected {{ background: #ffffff; border-bottom-color: {COLORS['accent']}; }}
QStackedWidget#ribbonStack, QWidget#ribbonPage {{ background: {COLORS['chrome']}; border: none; }}
QFrame#ribbonGroup {{ background: {COLORS['chrome']}; border: none; border-right: 1px solid {COLORS['border']}; }}
QLabel#ribbonGroupTitle {{ background: transparent; color: {COLORS['muted']}; font-size: 8pt; min-height: 14px; }}
QToolButton#ribbonLargeButton {{ min-width: 64px; min-height: 52px; padding: 1px 6px; }}
QToolButton#ribbonSmallButton {{ min-height: 23px; padding: 0 5px; text-align: left; }}
QToolButton#ribbonCompactButton {{ min-width: 0; max-width: 128px; min-height: 25px; padding: 0 5px; text-align: left; }}
QWidget#navigationPanel {{ background: white; border-right: 1px solid {COLORS['border']}; }}
QTabWidget#navigationTabs::pane {{ background: white; border: none; border-top: 1px solid {COLORS['border']}; }}
QTabWidget#navigationTabs QTabBar::tab {{ background: {COLORS['chrome']}; border: none; border-right: 1px solid {COLORS['border']}; min-width: 72px; height: 25px; padding: 0 8px; }}
QTabWidget#navigationTabs QTabBar::tab:selected {{ background: white; color: {COLORS['text']}; border-top: 2px solid {COLORS['accent']}; }}
QTabWidget#inspectionTabs::pane, QTabWidget#meshBrowserTabs::pane {{ background: white; border: 1px solid {COLORS['border']}; }}
QTabWidget#inspectionTabs QTabBar::tab, QTabWidget#meshBrowserTabs QTabBar::tab {{ background: {COLORS['chrome']}; border: 1px solid {COLORS['border']}; border-bottom: none; min-width: 64px; height: 25px; padding: 0 9px; }}
QTabWidget#inspectionTabs QTabBar::tab:selected, QTabWidget#meshBrowserTabs QTabBar::tab:selected {{ background: white; border-top: 2px solid {COLORS['accent']}; }}
QWidget#viewportPanel {{ background: white; }}
QWidget#scopeCreationBar {{ background: {COLORS['chrome']}; border-top: 1px solid {COLORS['border']}; }}
QLabel#scopeCreationType {{ color: {COLORS['accent']}; font-weight: 600; min-width: 54px; }}
QPushButton#scopeCreationSubmit {{ background: {COLORS['accent']}; color: white; border-color: {COLORS['accent']}; font-weight: 600; min-width: 72px; }}
QPushButton#scopeCreationSubmit:hover {{ background: #3f6f92; }}
QPushButton#scopeCreationSubmit:disabled {{ background: {COLORS['chrome']}; color: {COLORS['disabled']}; border-color: {COLORS['border']}; }}
QPushButton {{ background: {COLORS['chrome']}; border: 1px solid {COLORS['border']}; border-radius: 2px; padding: 4px 12px; min-height: 23px; }}
QPushButton:hover {{ background: {COLORS['hover']}; }}
QPushButton:focus {{ border-color: {COLORS['accent']}; }}
QPushButton:disabled {{ color: {COLORS['disabled']}; }}
QCheckBox, QRadioButton {{ spacing: 7px; min-height: 22px; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: {COLORS['disabled']}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget, QTableView, QPlainTextEdit {{ background: white; border: 1px solid {COLORS['border']}; padding: 3px 5px; selection-background-color: {COLORS['selected']}; selection-color: {COLORS['text']}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {{ border-color: {COLORS['accent']}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ background: #e9ecef; color: {COLORS['disabled']}; border-color: #c3c8cd; }}
QDoubleSpinBox#resultScaleValue:disabled {{ background: {COLORS['background']}; color: {COLORS['disabled']}; border-color: {COLORS['soft_border']}; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; height: 0; border: none; }}
QComboBox {{ min-height: 22px; padding-right: 22px; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right; width: 20px; border-left: 1px solid {COLORS['soft_border']}; background: {COLORS['chrome']}; }}
QComboBox::down-arrow {{ image: url("{_COMBO_DOWN_ARROW}"); width: 8px; height: 5px; }}
QTableView {{ alternate-background-color: #fafbfc; gridline-color: {COLORS['soft_border']}; padding: 0; }}
QTableView::item {{ padding: 2px 5px; border: none; }}
QTableView::item:hover {{ background: {COLORS['hover']}; }}
QTableView::item:selected {{ background: {COLORS['selected']}; color: {COLORS['text']}; }}
QScrollBar:vertical {{ background: #f3f4f5; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #bcc3c9; min-height: 24px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: #9fa9b1; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: #f3f4f5; height: 11px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #bcc3c9; min-width: 24px; margin: 2px; }}
QScrollBar::handle:horizontal:hover {{ background: #9fa9b1; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QGroupBox {{ border: 1px solid {COLORS['border']}; margin-top: 14px; padding: 12px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
"""
