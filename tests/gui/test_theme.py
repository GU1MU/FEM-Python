from fem_gui.theme import build_stylesheet


def test_checkbox_theme_uses_the_native_indicator_instead_of_a_fragile_icon_resource():
    stylesheet = build_stylesheet()

    assert "QCheckBox, QRadioButton" in stylesheet
    assert "QCheckBox::indicator" not in stylesheet
    assert "standardbutton-apply" not in stylesheet
    assert "QComboBox::drop-down" in stylesheet
