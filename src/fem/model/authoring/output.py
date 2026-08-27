from __future__ import annotations

from typing import Any, Sequence

from ..entities import AnalysisStep, OutputRequest


def output(
    step: AnalysisStep,
    kind: str,
    target: str,
    variables: Sequence[str] = (),
    **metadata: Any,
) -> OutputRequest:
    """Add an output request to a step."""
    request = OutputRequest(kind, target, variables, metadata)
    step.outputs = tuple(step.outputs) + (request,)
    return request
