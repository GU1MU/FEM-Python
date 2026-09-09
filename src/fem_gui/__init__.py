"""Finite element desktop interface alongside the numerical kernel."""

from __future__ import annotations

__all__ = ["create_application", "main"]


def create_application(argv=None):
    """Lazily import the Qt application entry point."""
    from .app import create_application as _create_application

    return _create_application(argv)


def main(argv=None) -> int:
    """Launch the GUI."""
    from .app import main as _main

    return _main(argv)
