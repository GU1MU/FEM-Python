from __future__ import annotations

from pathlib import Path

from ..core.model import FEMModel
from .builder import AbaqusBuildResult, build_model, build_model_with_report
from .parser import parse_file


def read(path: str | Path) -> FEMModel:
    """Read a model while intentionally discarding import notices.

    Use :func:`read_with_report` for user-facing import workflows.
    """

    return build_model(parse_file(path))


def read_with_report(path: str | Path) -> AbaqusBuildResult:
    """Read an Abaqus input file and retain source-level import notices."""

    return build_model_with_report(parse_file(path))
