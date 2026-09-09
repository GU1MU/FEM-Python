"""PySide6 application entry point."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from .main_window import FEMMainWindow
from .theme import FEMStyle, build_stylesheet


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Create or reuse the QApplication."""
    application = QApplication.instance()
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs)
    if application is None:
        application = QApplication(list(sys.argv if argv is None else argv))
    _configure_font(application)
    application.setApplicationDisplayName("Finite Element Analysis")
    application.setStyle(FEMStyle())
    application.setStyleSheet(build_stylesheet())
    return application


def _configure_font(application: QApplication) -> None:
    """Use an available English UI font."""
    preferred = ("Segoe UI", "Noto Sans", "DejaVu Sans")
    installed = set(QFontDatabase.families())
    family = next((name for name in preferred if name in installed), None)
    if family is not None:
        application.setFont(QFont(family, 9))


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the main window."""
    application = create_application(argv)
    window = FEMMainWindow()
    window.showMaximized()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
