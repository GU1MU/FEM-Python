import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QComboBox

from fem_gui.theme import build_stylesheet


def test_checkbox_theme_uses_the_native_indicator_instead_of_a_fragile_icon_resource():
    stylesheet = build_stylesheet()

    assert "QCheckBox, QRadioButton" in stylesheet
    assert "QCheckBox::indicator" not in stylesheet
    assert "standardbutton-apply" not in stylesheet
    assert "QComboBox::drop-down" in stylesheet


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
