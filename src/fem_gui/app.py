"""PySide6 应用入口。"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from .main_window import FEMMainWindow
from .theme import build_stylesheet


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """创建或复用 QApplication。"""
    application = QApplication.instance()
    if application is None:
        application = QApplication(list(sys.argv if argv is None else argv))
    _configure_chinese_font(application)
    application.setApplicationDisplayName("有限元分析")
    application.setStyleSheet(build_stylesheet())
    return application


def _configure_chinese_font(application: QApplication) -> None:
    """必要时注册系统中文字体，兼容 Qt offscreen 会话。"""
    preferred = ("Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC")
    installed = set(QFontDatabase.families())
    family = next((name for name in preferred if name in installed), None)
    if family is None:
        candidates = (
            Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc",
            Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "NotoSansSC-VF.ttf",
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        )
        for path in candidates:
            if not path.is_file():
                continue
            font_id = QFontDatabase.addApplicationFont(str(path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                family = families[0]
                break
    if family is not None:
        application.setFont(QFont(family, 9))


def main(argv: Sequence[str] | None = None) -> int:
    """启动主窗口。"""
    application = create_application(argv)
    window = FEMMainWindow()
    window.showMaximized()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
