from PySide6.QtCore import QPoint, QRect
import pytest
from PySide6.QtGui import QColor, QImage, QPainter, QPalette
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox, QStyle,
    QStyleOption, QStyleOptionSpinBox,
)

from fem_gui.theme import FEMStyle, build_stylesheet


@pytest.fixture
def themed_application(gui_application):
    previous_stylesheet = gui_application.styleSheet()
    gui_application.setStyleSheet(build_stylesheet())
    try:
        yield gui_application
    finally:
        gui_application.setStyleSheet(previous_stylesheet)


def test_themed_checkbox_visually_distinguishes_checked_state(themed_application):
    checkbox = QCheckBox()
    checkbox.resize(30, 30)
    checkbox.show()
    themed_application.processEvents()
    unchecked = checkbox.grab().toImage()

    checkbox.setChecked(True)
    themed_application.processEvents()

    assert checkbox.grab().toImage() != unchecked


def test_fem_style_draws_a_larger_high_contrast_checkbox_indicator(gui_application):
    style = FEMStyle()
    option = QStyleOption()
    option.rect = QRect(0, 0, 16, 16)
    option.state = QStyle.StateFlag.State_Enabled
    image = QImage(16, 16, QImage.Format.Format_ARGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)
    style.drawPrimitive(
        QStyle.PrimitiveElement.PE_IndicatorCheckBox,
        option,
        painter,
    )
    painter.end()

    assert (
        style.pixelMetric(QStyle.PixelMetric.PM_IndicatorWidth) == style.CHECKBOX_SIZE
    )
    assert (
        style.pixelMetric(QStyle.PixelMetric.PM_IndicatorHeight) == style.CHECKBOX_SIZE
    )
    dark_border_pixels = sum(
        1
        for y in range(image.height())
        for x in range(image.width())
        if max(
            image.pixelColor(x, y).red(),
            image.pixelColor(x, y).green(),
            image.pixelColor(x, y).blue(),
        )
        < 160
    )
    assert dark_border_pixels >= 40

    option.state |= QStyle.StateFlag.State_On
    image.fill(QColor("transparent"))
    painter = QPainter(image)
    style.drawPrimitive(
        QStyle.PrimitiveElement.PE_IndicatorCheckBox,
        option,
        painter,
    )
    painter.end()

    light_check_pixels = sum(
        1
        for y in range(image.height())
        for x in range(image.width())
        if min(
            image.pixelColor(x, y).red(),
            image.pixelColor(x, y).green(),
            image.pixelColor(x, y).blue(),
        )
        > 220
    )
    assert light_check_pixels >= 8


def test_fem_style_draws_a_visible_checked_radio_indicator(gui_application):
    style = FEMStyle()
    option = QStyleOption()
    option.rect = QRect(0, 0, 16, 16)
    option.state = (
        QStyle.StateFlag.State_Enabled
        | QStyle.StateFlag.State_On
    )
    image = QImage(16, 16, QImage.Format.Format_ARGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)

    style.drawPrimitive(
        QStyle.PrimitiveElement.PE_IndicatorRadioButton,
        option,
        painter,
    )
    painter.end()

    center = image.pixelColor(8, 8)
    assert center.blue() > center.red()
    assert center.blue() > center.green()


def test_combo_theme_draws_a_visible_down_arrow(themed_application):
    combo = QComboBox()
    combo.addItem("Step-1")
    combo.resize(145, 30)
    combo.show()
    themed_application.processEvents()

    image = QImage(combo.size(), QImage.Format.Format_ARGB32)
    image.fill(QColor("transparent"))
    painter = QPainter(image)
    combo.render(painter, QPoint())
    painter.end()

    arrow_pixels = 0
    for y in range(image.height()):
        for x in range(image.width() - 20, image.width()):
            color = image.pixelColor(x, y)
            if color.alpha() and max(color.red(), color.green(), color.blue()) < 150:
                arrow_pixels += 1

    combo.close()

    assert arrow_pixels > 0


@pytest.mark.parametrize("spin_box_type", [QSpinBox, QDoubleSpinBox])
def test_spin_box_theme_hides_increment_and_decrement_buttons(
    themed_application, spin_box_type,
):
    spin_box = spin_box_type()
    spin_box.ensurePolished()
    option = QStyleOptionSpinBox()
    spin_box.initStyleOption(option)
    for button in (QStyle.SubControl.SC_SpinBoxUp, QStyle.SubControl.SC_SpinBoxDown):
        rectangle = spin_box.style().subControlRect(
            QStyle.ComplexControl.CC_SpinBox, option, button, spin_box,
        )
        assert rectangle.isEmpty()


@pytest.mark.parametrize("object_name", ["", "resultScaleValue"])
def test_disabled_spin_box_uses_muted_text(themed_application, object_name):
    spin_box = QDoubleSpinBox()
    spin_box.setObjectName(object_name)
    spin_box.ensurePolished()
    palette = spin_box.palette()

    enabled_text = palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Text)
    disabled_text = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)

    assert disabled_text.lightness() > enabled_text.lightness()
