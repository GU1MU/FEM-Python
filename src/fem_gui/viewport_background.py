"""Viewport background state, derived colors, and persistence."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor


@dataclass(frozen=True, slots=True)
class ViewportBackgroundSettings:
    """Describe a solid or gradient viewport background."""

    style: str = "gradient"
    bottom_color: str = "#e1f1f8"
    top_color: str = "#ffffff"
    auto_contrast: bool = True

    def normalized(self) -> "ViewportBackgroundSettings":
        style = "gradient" if self.style == "gradient" else "solid"
        bottom_input = QColor(self.bottom_color)
        bottom = bottom_input.name() if bottom_input.isValid() else "#e1f1f8"
        top_input = QColor(self.top_color)
        top = top_input.name() if top_input.isValid() else bottom
        return ViewportBackgroundSettings(style, bottom, top, bool(self.auto_contrast))

    @property
    def is_dark(self) -> bool:
        colors = [QColor(self.bottom_color)]
        if self.style == "gradient":
            colors.append(QColor(self.top_color))
        luminance = sum(
            0.2126 * color.redF() + 0.7152 * color.greenF() + 0.0722 * color.blueF()
            for color in colors
        ) / len(colors)
        return luminance < 0.48

    @property
    def foreground_color(self) -> str:
        if not self.auto_contrast:
            return "#20262d"
        return "#f2f5f7" if self.is_dark else "#20262d"


def load_background_settings(store: QSettings) -> tuple[ViewportBackgroundSettings, bool]:
    """Read the application viewport background, using defaults if no settings are remembered."""
    remember = store.value("viewportBackground/remember", False, type=bool)
    if not remember:
        return ViewportBackgroundSettings(), False
    settings = ViewportBackgroundSettings(
        style=str(store.value("viewportBackground/style", "gradient")),
        bottom_color=str(store.value("viewportBackground/bottom", "#e1f1f8")),
        top_color=str(store.value("viewportBackground/top", "#ffffff")),
        auto_contrast=store.value("viewportBackground/autoContrast", True, type=bool),
    ).normalized()
    return settings, True


def save_background_settings(
    store: QSettings,
    settings: ViewportBackgroundSettings,
    remember: bool,
) -> None:
    """Save or clear the application viewport background."""
    store.setValue("viewportBackground/remember", bool(remember))
    if not remember:
        store.remove("viewportBackground/style")
        store.remove("viewportBackground/bottom")
        store.remove("viewportBackground/top")
        store.remove("viewportBackground/autoContrast")
        return
    value = settings.normalized()
    store.setValue("viewportBackground/style", value.style)
    store.setValue("viewportBackground/bottom", value.bottom_color)
    store.setValue("viewportBackground/top", value.top_color)
    store.setValue("viewportBackground/autoContrast", value.auto_contrast)
