from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from fem_gui.visualization.result_adapter import ScalarField
from fem_gui.visualization.scene import DisplayState
from fem_gui.viewport_background import (
    ViewportBackgroundSettings,
    load_background_settings,
    save_background_settings,
)
from fem_gui.widgets.viewport import FEMViewport


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_background_settings_persist_only_when_requested(tmp_path):
    store = QSettings(str(tmp_path / "background.ini"), QSettings.Format.IniFormat)
    selected = ViewportBackgroundSettings("solid", "#16191c", "#16191c")

    save_background_settings(store, selected, True)
    store.sync()
    loaded, remembered = load_background_settings(store)
    assert remembered
    assert loaded == selected

    save_background_settings(store, selected, False)
    store.sync()
    loaded, remembered = load_background_settings(store)
    assert not remembered
    assert loaded == ViewportBackgroundSettings()


def test_default_background_matches_light_blue_gradient():
    settings = ViewportBackgroundSettings()

    assert settings.style == "gradient"
    assert settings.bottom_color == "#e1f1f8"
    assert settings.top_color == "#ffffff"


def test_background_normalization_keeps_default_preset_case_insensitive():
    selected = ViewportBackgroundSettings("gradient", "#E1F1F8", "#FFFFFF")

    assert selected.normalized() == ViewportBackgroundSettings()


def test_background_contrast_follows_average_luminance():
    dark = ViewportBackgroundSettings("gradient", "#607d92", "#1e3448")
    light = ViewportBackgroundSettings("solid", "#ffffff", "#ffffff")

    assert dark.is_dark
    assert dark.foreground_color == "#f2f5f7"
    assert not light.is_dark
    assert light.foreground_color == "#20262d"


def test_background_refresh_reuses_rendered_stress_grid_and_scalar(monkeypatch):
    _application()
    viewport = FEMViewport()
    rendered_grid = object()
    rendered_scalar = ScalarField(
        "EN:Mises",
        "Mises",
        "point",
        np.asarray([1.0, 2.0, 3.0]),
    )
    viewport._result_grid = rendered_grid
    viewport._result_scalar = rendered_scalar
    viewport._display = DisplayState(
        "undeformed",
        True,
        rendered_scalar.key,
    )
    viewport._contour["show_maximum"] = True
    calls = []
    monkeypatch.setattr(
        viewport,
        "_add_extrema_labels",
        lambda grid, scalar: calls.append((grid, scalar)),
    )

    viewport._refresh_extrema_for_background()

    assert calls == [(rendered_grid, rendered_scalar)]
    viewport.close()
