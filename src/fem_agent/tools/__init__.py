"""Typed local tools exposed through the Agent registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .importing import AbaqusImportResult, inspect_abaqus, runnable_steps
from .inspection import AbaqusKeywordInspection, inspect_abaqus_keywords

if TYPE_CHECKING:
    from .registry import AgentToolRegistry, ToolExecutionContext


def __getattr__(name: str) -> Any:
    if name in {"AgentToolRegistry", "ToolExecutionContext"}:
        from .registry import AgentToolRegistry, ToolExecutionContext

        return {
            "AgentToolRegistry": AgentToolRegistry,
            "ToolExecutionContext": ToolExecutionContext,
        }[name]
    raise AttributeError(name)

__all__ = [
    "AgentToolRegistry",
    "AbaqusImportResult",
    "AbaqusKeywordInspection",
    "ToolExecutionContext",
    "inspect_abaqus",
    "inspect_abaqus_keywords",
    "runnable_steps",
]
