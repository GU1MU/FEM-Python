"""Pure publication policy for the native authoring tool catalog."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .authoring import AuthoringContext
from .authoring_values import AuthoringWorkflowStage
from .providers.base import ToolDefinition
from .authoring_schemas.session import (
    _CREATE_NATIVE_MODEL_DOCUMENT,
    _READ_CONTEXT,
    _REQUEST_PROJECT_SAVE,
    _REQUIREMENT_SPECS,
    _SET_REQUIREMENTS,
    _stage_requirement_tool,
)
from .authoring_schemas.geometry import (
    _PREPARE_GEOMETRY,
    _PREPARE_GEOMETRY_EDIT,
    _PREPARE_PLANAR_CONSTRUCTION,
    _PREPARE_PROFILE_EXTRUSION,
    _PREPARE_PROFILE_PATH_SWEEP,
    _PREPARE_PROFILE_REVOLUTION,
    _READ_GEOMETRY_EDIT_CONTEXT,
    _READ_PROFILE_TRANSFORM_CONTEXT,
    _geometry_edit_tool_definition,
)
from .authoring_schemas.definitions import (
    _APPLY_DEFINITION,
    _EDIT_MODEL_OBJECT,
    _PREPARE_DELETE,
    _READ_DELETABLE_OBJECTS,
    _READ_EDITABLE_OBJECTS,
    _READ_MODEL_TOPOLOGY_CONTEXT,
)
from .authoring_schemas.analysis import (
    _PREPARE_MESH,
    _PREPARE_SOLVE,
    _READ_MESH_REFINEMENT_CONTEXT,
    _RUN_PREFLIGHT,
)
from .authoring_schemas.catalogs import (
    _ANALYSIS_RUN_CATALOG,
    _EXPORT_RESULT_CSV,
    _GEOMETRY_FEATURE_CATALOG,
    _READ_RESULT_DISPLAY_CONTEXT,
    _RESULT_CATALOG,
    _RESULT_COMPARISON,
    _RESULT_QUERY,
    _WORKSPACE_DOCUMENTS,
)

_GEOMETRY_REQUIREMENTS = (
    "length_unit",
    "force_unit",
    "stress_unit",
)
_MESH_REQUIREMENTS = (
    "mesh_cell_shape",
    "mesh_order",
    "mesh_global_size",
)
_LINE_MESH_REQUIREMENTS = (
    "mesh_cell_shape",
    "mesh_order",
    "mesh_global_size",
    "line_element_type",
)
_REQUIREMENT_GROUPS: dict[str, tuple[str, ...]] = {
    "geometry": _GEOMETRY_REQUIREMENTS,
    "mesh": _MESH_REQUIREMENTS,
}
_REQUIREMENT_GATE_BY_STAGE = {
    AuthoringWorkflowStage.REQUIREMENTS: "geometry",
    AuthoringWorkflowStage.GEOMETRY_READY: "geometry",
    AuthoringWorkflowStage.MESH_READY: "mesh",
    AuthoringWorkflowStage.DEFINITIONS_READY: "mesh",
    AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY: "mesh",
    AuthoringWorkflowStage.PREFLIGHT_READY: "mesh",
    AuthoringWorkflowStage.SOLVE_READY: "mesh",
    AuthoringWorkflowStage.RESULTS_READY: "mesh",
}
_GATED_OPERATIONS_BY_STAGE = {
    AuthoringWorkflowStage.REQUIREMENTS: frozenset(
        {"prepare_geometry_proposal", "prepare_planar_construction_proposal"}
    ),
    AuthoringWorkflowStage.GEOMETRY_READY: frozenset(
        {"prepare_geometry_proposal", "prepare_planar_construction_proposal"}
    ),
    AuthoringWorkflowStage.MESH_READY: frozenset({"prepare_mesh_proposal"}),
    AuthoringWorkflowStage.DEFINITIONS_READY: frozenset({"prepare_mesh_proposal"}),
    AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY: frozenset(
        {"prepare_mesh_proposal"}
    ),
    AuthoringWorkflowStage.PREFLIGHT_READY: frozenset({"prepare_mesh_proposal"}),
    AuthoringWorkflowStage.SOLVE_READY: frozenset({"prepare_mesh_proposal"}),
    AuthoringWorkflowStage.RESULTS_READY: frozenset({"prepare_mesh_proposal"}),
}


_PROJECT_SAVE_READY_STAGES = frozenset(
    {
        AuthoringWorkflowStage.REQUIREMENTS,
        AuthoringWorkflowStage.GEOMETRY_READY,
        AuthoringWorkflowStage.MESH_READY,
        AuthoringWorkflowStage.DEFINITIONS_READY,
        AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY,
        AuthoringWorkflowStage.PREFLIGHT_READY,
        AuthoringWorkflowStage.SOLVE_READY,
        AuthoringWorkflowStage.RESULTS_READY,
    }
)
_DESTRUCTIVE_EDIT_TOOLS = frozenset(
    {_READ_DELETABLE_OBJECTS.name, _PREPARE_DELETE.name}
)
_MODEL_EDIT_TOOLS = frozenset({_READ_EDITABLE_OBJECTS.name, _EDIT_MODEL_OBJECT.name})
_GEOMETRY_EDIT_TOOLS = frozenset(
    {
        _READ_GEOMETRY_EDIT_CONTEXT.name,
        _PREPARE_GEOMETRY_EDIT.name,
    }
)
_GEOMETRY_CATALOG_TOOLS = frozenset({_GEOMETRY_FEATURE_CATALOG.name})
_PROFILE_TRANSFORM_TOOLS = frozenset(
    {
        _READ_PROFILE_TRANSFORM_CONTEXT.name,
        _PREPARE_PROFILE_EXTRUSION.name,
        _PREPARE_PROFILE_REVOLUTION.name,
        _PREPARE_PROFILE_PATH_SWEEP.name,
    }
)


_STAGE_TOOLS: dict[AuthoringWorkflowStage, tuple[ToolDefinition, ...]] = {
    AuthoringWorkflowStage.REQUIREMENTS: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _PREPARE_GEOMETRY,
        _PREPARE_PLANAR_CONSTRUCTION,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _APPLY_DEFINITION,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.GEOMETRY_READY: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _PREPARE_GEOMETRY,
        _PREPARE_PLANAR_CONSTRUCTION,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _APPLY_DEFINITION,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.GEOMETRY_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.MESH_READY: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _READ_MESH_REFINEMENT_CONTEXT,
        _PREPARE_MESH,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _APPLY_DEFINITION,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.MESH_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.DEFINITIONS_READY: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _READ_MESH_REFINEMENT_CONTEXT,
        _PREPARE_MESH,
        _READ_MODEL_TOPOLOGY_CONTEXT,
        _APPLY_DEFINITION,
        _RUN_PREFLIGHT,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _READ_MESH_REFINEMENT_CONTEXT,
        _PREPARE_MESH,
        _READ_MODEL_TOPOLOGY_CONTEXT,
        _APPLY_DEFINITION,
        _RUN_PREFLIGHT,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.PREFLIGHT_READY: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _READ_MESH_REFINEMENT_CONTEXT,
        _PREPARE_MESH,
        _READ_MODEL_TOPOLOGY_CONTEXT,
        _APPLY_DEFINITION,
        _RUN_PREFLIGHT,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.PREFLIGHT_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.SOLVE_READY: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _READ_MESH_REFINEMENT_CONTEXT,
        _PREPARE_MESH,
        _READ_MODEL_TOPOLOGY_CONTEXT,
        _APPLY_DEFINITION,
        _RUN_PREFLIGHT,
        _PREPARE_SOLVE,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.SOLVE_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.RESULTS_READY: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _READ_MESH_REFINEMENT_CONTEXT,
        _PREPARE_MESH,
        _READ_MODEL_TOPOLOGY_CONTEXT,
        _RUN_PREFLIGHT,
        _PREPARE_SOLVE,
        _RESULT_CATALOG,
        _RESULT_QUERY,
        _RESULT_COMPARISON,
        _READ_RESULT_DISPLAY_CONTEXT,
        _EXPORT_RESULT_CSV,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _APPLY_DEFINITION,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.PROJECT_SAVE_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.STALE: (_READ_CONTEXT,),
    AuthoringWorkflowStage.CANCELLED: (_READ_CONTEXT,),
}

# Keep the dedicated tools adjacent to the feature catalog in every ready
# native-authoring stage.  Pending/terminal stages intentionally publish only
# their read context and cannot create a proposal.
_PROFILE_TRANSFORM_STAGE_TOOLS = (
    _READ_PROFILE_TRANSFORM_CONTEXT,
    _PREPARE_PROFILE_EXTRUSION,
    _PREPARE_PROFILE_REVOLUTION,
    _PREPARE_PROFILE_PATH_SWEEP,
)
for _stage, _definitions in tuple(_STAGE_TOOLS.items()):
    if _stage not in _PROJECT_SAVE_READY_STAGES:
        continue
    expanded: list[ToolDefinition] = []
    for _definition in _definitions:
        expanded.append(_definition)
        if _definition.name == _GEOMETRY_FEATURE_CATALOG.name:
            expanded.extend(_PROFILE_TRANSFORM_STAGE_TOOLS)
    _STAGE_TOOLS[_stage] = tuple(expanded)


PROFILE_TRANSFORM_TOOL_NAMES = _PROFILE_TRANSFORM_TOOLS


@dataclass(frozen=True, slots=True)
class GeometryEditSchemaScope:
    binding_identity: tuple[str, str, int]
    part_id: str
    recipe_kind: str
    supported_operations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuthoringToolPolicyState:
    """Controller-owned observations copied for one publication evaluation.

    Observed context supplies counts, mesh state and requirement dimension.
    Fresh capability context is a separate input to the publication function.
    """

    stage: AuthoringWorkflowStage
    registered_tool_names: frozenset[str]
    observed_context: AuthoringContext | None = None
    proposed_requirement_keys: frozenset[str] = frozenset()
    binding_identity: tuple[str, str, int] | None = None
    geometry_edit_scope: GeometryEditSchemaScope | None = None
    workspace_result_inventory: tuple[int, int] | None = None


# These groups require all of their handlers before any member is published.
_PROJECT_SAVE_TOOL_GROUP = (
    frozenset({_REQUEST_PROJECT_SAVE.name}),
    "request_project_save",
)
_CONTEXT_TOOL_GROUPS = (
    _PROJECT_SAVE_TOOL_GROUP,
    (_DESTRUCTIVE_EDIT_TOOLS, "delete_model_objects"),
    (_MODEL_EDIT_TOOLS, "edit_model_objects"),
    (_GEOMETRY_EDIT_TOOLS, "edit_native_geometry"),
    (_GEOMETRY_CATALOG_TOOLS, "read_geometry_feature_catalog"),
    (_PROFILE_TRANSFORM_TOOLS, "edit_native_geometry"),
)
_CONTEXT_TOOL_NAMES = frozenset(
    name for names, _operation in _CONTEXT_TOOL_GROUPS for name in names
)


def current_requirement_group(stage: AuthoringWorkflowStage) -> str | None:
    return _REQUIREMENT_GATE_BY_STAGE.get(stage)


def active_part_dimension(context: AuthoringContext | None) -> int | None:
    if context is None or context.active_part_id is None:
        return None
    part = next(
        (
            item
            for item in context.parts
            if item.part_id == context.active_part_id and not item.suppressed
        ),
        None,
    )
    return None if part is None else part.dimension


def requirement_keys(
    group: str,
    observed_context: AuthoringContext | None,
) -> tuple[str, ...]:
    try:
        keys = _REQUIREMENT_GROUPS[group]
    except KeyError as exc:
        raise ValueError("unknown requirement group") from exc
    if group == "mesh" and active_part_dimension(observed_context) == 1:
        return _LINE_MESH_REQUIREMENTS
    return keys


def requirement_spec(
    key: str,
    observed_context: AuthoringContext | None,
) -> Mapping[str, object]:
    spec = _REQUIREMENT_SPECS[key]
    dimension = active_part_dimension(observed_context)
    if key == "mesh_cell_shape":
        if dimension == 1:
            return {"type": "string", "enum": ["line"]}
        if dimension == 3:
            return {"type": "string", "enum": ["tetrahedron", "hexahedron"]}
        return {"type": "string", "enum": ["triangle", "quadrilateral"]}
    if key == "mesh_order" and dimension == 1:
        return {"type": "integer", "enum": [1]}
    return spec


def requirement_stage(key: str) -> str:
    if key == "line_element_type":
        return "mesh"
    for stage, keys in _REQUIREMENT_GROUPS.items():
        if key in keys:
            return stage
    raise ValueError("unknown requirement field")


def capability_enabled(context: AuthoringContext, operation: str) -> bool:
    return any(
        item.operation == operation and item.enabled for item in context.capabilities
    )


def _context_group_available(
    stage: AuthoringWorkflowStage,
    registered_tool_names: frozenset[str],
    context: AuthoringContext | None,
    names: frozenset[str],
    operation: str,
) -> bool:
    if (
        stage not in _PROJECT_SAVE_READY_STAGES
        or not names.issubset(registered_tool_names)
        or type(context) is not AuthoringContext
    ):
        return False
    binding = context.binding
    if not (
        binding.supported
        and binding.source_kind == "native"
        and capability_enabled(context, operation)
    ):
        return False
    if names == _GEOMETRY_EDIT_TOOLS:
        return bool(context.parts)
    if names in (_GEOMETRY_CATALOG_TOOLS, _PROFILE_TRANSFORM_TOOLS):
        return any(not part.suppressed for part in context.parts)
    return True


def project_save_available(
    stage: AuthoringWorkflowStage,
    registered_tool_names: frozenset[str],
    context: AuthoringContext | None,
) -> bool:
    names, operation = _PROJECT_SAVE_TOOL_GROUP
    return _context_group_available(
        stage, registered_tool_names, context, names, operation
    )


def _global_reads_only(state: AuthoringToolPolicyState) -> bool:
    return (
        state.observed_context is None
        and state.workspace_result_inventory is not None
    )


def _result_counts(state: AuthoringToolPolicyState) -> tuple[int, int]:
    if state.workspace_result_inventory is not None:
        return state.workspace_result_inventory
    if state.observed_context is not None:
        return state.observed_context.run_count, state.observed_context.result_count
    return 0, 0


def publication_needs_live_context(state: AuthoringToolPolicyState) -> bool:
    """Request a fresh read only if a registered context group can be shown."""

    return bool(
        not _global_reads_only(state)
        and state.stage in _PROJECT_SAVE_READY_STAGES
        and any(
            names.issubset(state.registered_tool_names)
            for names, _operation in _CONTEXT_TOOL_GROUPS
        )
    )


def publication_needs_workspace_export(state: AuthoringToolPolicyState) -> bool:
    run_count, result_count = _result_counts(state)
    return bool(
        _EXPORT_RESULT_CSV.name in state.registered_tool_names
        and result_count > 0
        and (
            _global_reads_only(state)
            or (state.stage in _PROJECT_SAVE_READY_STAGES and run_count > 0)
        )
    )


def _current_mesh_available(state: AuthoringToolPolicyState) -> bool:
    context = state.observed_context
    return bool(
        type(context) is AuthoringContext
        and context.binding.supported
        and context.binding.source_kind == "native"
        and context.mesh.present
        and context.mesh.current
        and _APPLY_DEFINITION.name in state.registered_tool_names
    )


def publish_authoring_tools(
    state: AuthoringToolPolicyState,
    *,
    live_context: AuthoringContext | None = None,
    workspace_export_selected: bool = False,
) -> tuple[ToolDefinition, ...]:
    """Select ordered definitions without reading or changing external state."""

    registered = state.registered_tool_names
    run_count, result_count = _result_counts(state)
    if _global_reads_only(state):
        global_reads = (
            *(
                (_WORKSPACE_DOCUMENTS,)
                if _WORKSPACE_DOCUMENTS.name in registered
                else ()
            ),
            *((_ANALYSIS_RUN_CATALOG,) if run_count > 0 else ()),
            *((_RESULT_CATALOG, _RESULT_QUERY) if result_count > 0 else ()),
            *((_RESULT_COMPARISON,) if result_count >= 2 else ()),
            *((_READ_RESULT_DISPLAY_CONTEXT,) if result_count > 0 else ()),
            *(
                (_EXPORT_RESULT_CSV,)
                if result_count > 0 and workspace_export_selected
                else ()
            ),
        )
        return tuple(item for item in global_reads if item.name in registered)

    definitions = list(_STAGE_TOOLS[state.stage])
    if (
        state.stage in _PROJECT_SAVE_READY_STAGES
        and _CREATE_NATIVE_MODEL_DOCUMENT.name in registered
    ):
        definitions.insert(1, _CREATE_NATIVE_MODEL_DOCUMENT)
    if _WORKSPACE_DOCUMENTS.name in registered:
        definitions.append(_WORKSPACE_DOCUMENTS)
    if state.stage in _PROJECT_SAVE_READY_STAGES and run_count > 0:
        historical_definitions = [_ANALYSIS_RUN_CATALOG]
        if result_count > 0:
            historical_definitions.extend((_RESULT_CATALOG, _RESULT_QUERY))
            historical_definitions.append(_READ_RESULT_DISPLAY_CONTEXT)
            if workspace_export_selected:
                historical_definitions.append(_EXPORT_RESULT_CSV)
        if result_count >= 2:
            historical_definitions.append(_RESULT_COMPARISON)
        for result_definition in historical_definitions:
            if all(item.name != result_definition.name for item in definitions):
                definitions.append(result_definition)

    requirement_group = current_requirement_group(state.stage)
    requirements_complete = (
        requirement_group is not None
        and set(requirement_keys(requirement_group, state.observed_context))
        <= state.proposed_requirement_keys
    )
    gated_operations = _GATED_OPERATIONS_BY_STAGE.get(state.stage, frozenset())
    context_tool_names = frozenset(
        name
        for names, operation in _CONTEXT_TOOL_GROUPS
        if _context_group_available(
            state.stage, registered, live_context, names, operation
        )
        for name in names
    )
    visible: list[ToolDefinition] = []
    for item in definitions:
        name = item.name
        if (
            name not in {_READ_CONTEXT.name, _SET_REQUIREMENTS.name}
            and name not in registered
        ):
            continue
        if name in gated_operations and not requirements_complete:
            continue
        if (
            name == _APPLY_DEFINITION.name
            and state.stage in {
                AuthoringWorkflowStage.REQUIREMENTS,
                AuthoringWorkflowStage.GEOMETRY_READY,
                AuthoringWorkflowStage.MESH_READY,
            }
            and not _current_mesh_available(state)
        ):
            continue
        if name == _ANALYSIS_RUN_CATALOG.name and run_count <= 0:
            continue
        if name in {_RESULT_CATALOG.name, _RESULT_QUERY.name, _RESULT_COMPARISON.name}:
            if result_count < (2 if name == _RESULT_COMPARISON.name else 1):
                continue
        if name == _READ_RESULT_DISPLAY_CONTEXT.name and result_count <= 0:
            continue
        if name == _EXPORT_RESULT_CSV.name and not (
            result_count > 0 and workspace_export_selected
        ):
            continue
        if name in _CONTEXT_TOOL_NAMES and name not in context_tool_names:
            continue
        if name == _SET_REQUIREMENTS.name and requirement_group is not None:
            keys = requirement_keys(requirement_group, state.observed_context)
            item = _stage_requirement_tool(
                requirement_group,
                keys,
                {key: requirement_spec(key, state.observed_context) for key in keys},
            )
        if name == _PREPARE_GEOMETRY_EDIT.name:
            scope = state.geometry_edit_scope
            if scope is not None and scope.binding_identity == state.binding_identity:
                item = _geometry_edit_tool_definition(
                    scope.part_id, scope.recipe_kind, scope.supported_operations
                )
        visible.append(item)
    return tuple(visible)
