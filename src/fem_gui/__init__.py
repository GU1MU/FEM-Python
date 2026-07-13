"""中文有限元桌面界面，与数值内核保持同级。"""

from __future__ import annotations

__all__ = ["create_application", "main"]


def create_application(argv=None):
    """延迟导入 Qt 应用入口。"""
    from .app import create_application as _create_application

    return _create_application(argv)


def main(argv=None) -> int:
    """启动中文 GUI。"""
    from .app import main as _main

    return _main(argv)
