"""Publish existing read and export contracts as authoring tool definitions."""

from __future__ import annotations

from ..export_authoring import (
    export_result_csv_tool_schema,
    result_display_context_tool_schema,
)
from ..geometry_authoring import geometry_feature_catalog_tool_schema
from ..result_authoring import (
    analysis_run_catalog_tool_schema,
    result_catalog_tool_schema,
    result_comparison_tool_schema,
    result_query_tool_schema,
)
from ..workspace_catalog import workspace_documents_tool_schema
from .common import _result_tool_definition


_RESULT_CATALOG = _result_tool_definition(result_catalog_tool_schema())
_RESULT_QUERY = _result_tool_definition(result_query_tool_schema())
_RESULT_COMPARISON = _result_tool_definition(result_comparison_tool_schema())
_ANALYSIS_RUN_CATALOG = _result_tool_definition(analysis_run_catalog_tool_schema())
_WORKSPACE_DOCUMENTS = _result_tool_definition(workspace_documents_tool_schema())
_GEOMETRY_FEATURE_CATALOG = _result_tool_definition(
    geometry_feature_catalog_tool_schema()
)
_EXPORT_RESULT_CSV = _result_tool_definition(export_result_csv_tool_schema())
_READ_RESULT_DISPLAY_CONTEXT = _result_tool_definition(
    result_display_context_tool_schema()
)
