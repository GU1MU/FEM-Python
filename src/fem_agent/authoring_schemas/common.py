"""Shared constructors for authoring tool definitions."""

from __future__ import annotations

from collections.abc import Mapping

from ..providers.base import ToolDefinition


_NO_ARGUMENTS = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}


def _tool(
    name: str,
    description: str,
    parameters: Mapping[str, object],
) -> ToolDefinition:
    return ToolDefinition(name, description, parameters)


def _result_tool_definition(schema: Mapping[str, object]) -> ToolDefinition:
    return ToolDefinition(
        str(schema["name"]),
        str(schema["description"]),
        schema["input_schema"],  # type: ignore[arg-type]
    )
