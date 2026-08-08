import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QComboBox, QStyle, QStyleOption

from fem_gui.theme import FEMStyle, build_stylesheet


def test_checkbox_theme_delegates_indicator_drawing_to_the_application_style():
    stylesheet = build_stylesheet()

    assert "QCheckBox, QRadioButton" in stylesheet
    assert "QCheckBox::indicator" not in stylesheet
    assert "standardbutton-apply" not in stylesheet
    assert "QComboBox::drop-down" in stylesheet


def test_fem_style_draws_a_larger_high_contrast_checkbox_indicator():
    QApplication.instance() or QApplication([])
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


def test_combo_theme_draws_a_visible_down_arrow():
    stylesheet = build_stylesheet()
    assert "QComboBox::down-arrow" in stylesheet

    application = QApplication.instance() or QApplication([])
    previous_stylesheet = application.styleSheet()
    application.setStyleSheet(stylesheet)
    combo = QComboBox()
    combo.addItem("Step-1")
    combo.resize(145, 30)
    combo.show()
    application.processEvents()

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
    application.setStyleSheet(previous_stylesheet)

    assert arrow_pixels > 0


def test_spin_box_theme_hides_increment_and_decrement_buttons():
    stylesheet = build_stylesheet()

    assert "QSpinBox::up-button" in stylesheet
    assert "QSpinBox::down-button" in stylesheet
    assert "QDoubleSpinBox::up-button" in stylesheet
    assert "QDoubleSpinBox::down-button" in stylesheet


def test_disabled_result_scale_uses_a_muted_input_style():
    stylesheet = build_stylesheet()

    assert "QDoubleSpinBox#resultScaleValue:disabled" in stylesheet
    assert "background: #f4f5f6; color: #a0a6ac;" in stylesheet
