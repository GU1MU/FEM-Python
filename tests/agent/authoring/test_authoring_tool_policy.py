from __future__ import annotations

from dataclasses import replace

import pytest

from fem_agent.authoring import (
    AuthoringContext,
    CapabilitySummary,
    LocalModelBinding,
    MeshSummary,
    PartSummary,
)
from fem_agent.authoring_schemas.geometry import (
    _PREPARE_GEOMETRY_EDIT,
    _geometry_edit_tool_definition,
)
from fem_agent.authoring_schemas.session import _stage_requirement_tool
from fem_agent.authoring_tool_policy import (
    AuthoringToolPolicyState,
    GeometryEditSchemaScope,
    project_save_available,
    publication_needs_live_context,
    publication_needs_workspace_export,
    publish_authoring_tools,
    requirement_keys,
    requirement_spec,
    requirement_stage,
)
from fem_agent.authoring_values import AuthoringWorkflowStage


_READY_TOOL_ORDER = (
    "read_authoring_context",
    "create_native_model_document",
    "read_geometry_feature_catalog",
    "read_profile_transform_context",
    "prepare_profile_extrusion",
    "prepare_profile_revolution",
    "prepare_profile_path_sweep",
    "set_authoring_requirements",
    "prepare_geometry_proposal",
    "prepare_planar_construction_proposal",
    "read_geometry_edit_context",
    "prepare_geometry_edit",
    "apply_model_definition",
    "request_project_save",
    "read_deletable_objects",
    "prepare_delete_proposal",
    "read_editable_model_objects",
    "edit_model_object",
    "read_workspace_documents",
)
_RESULT_TOOLS = frozenset({
    "read_analysis_run_catalog",
    "read_accepted_result_catalog",
    "query_accepted_result",
    "compare_accepted_results",
    "read_result_display_context",
    "export_accepted_result_csv",
})
_GROUPED_TOOLS = (
    ("request_project_save",),
    ("read_deletable_objects", "prepare_delete_proposal"),
    ("read_editable_model_objects", "edit_model_object"),
    ("read_geometry_edit_context", "prepare_geometry_edit"),
    ("read_geometry_feature_catalog",),
    (
        "read_profile_transform_context",
        "prepare_profile_extrusion",
        "prepare_profile_revolution",
        "prepare_profile_path_sweep",
    ),
)
_READ_ONLY_STAGES = (
    AuthoringWorkflowStage.GEOMETRY_PENDING,
    AuthoringWorkflowStage.MESH_PENDING,
    AuthoringWorkflowStage.PREFLIGHT_PENDING,
    AuthoringWorkflowStage.SOLVE_PENDING,
    AuthoringWorkflowStage.PROJECT_SAVE_PENDING,
    AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING,
    AuthoringWorkflowStage.STALE,
    AuthoringWorkflowStage.CANCELLED,
)


def _native_context(dimension: int = 2) -> AuthoringContext:
    return AuthoringContext(
        binding=LocalModelBinding("document:policy", "native-policy", 3, "native", True),
        model_name="Model-policy",
        active_part_id="part-policy",
        parts=(PartSummary("part-policy", "Part-policy", "planar_sketch", dimension, False),),
        mesh=MeshSummary(True, True, 20, 10),
        capabilities=tuple(
            CapabilitySummary(operation, True)
            for operation in (
                "request_project_save",
                "delete_model_objects",
                "edit_model_objects",
                "edit_native_geometry",
                "read_geometry_feature_catalog",
            )
        ),
    )


def _policy_state(**changes) -> AuthoringToolPolicyState:
    return replace(
        AuthoringToolPolicyState(
            stage=AuthoringWorkflowStage.REQUIREMENTS,
            registered_tool_names=frozenset(_READY_TOOL_ORDER) | _RESULT_TOOLS,
            observed_context=_native_context(),
            proposed_requirement_keys=frozenset({"length_unit", "force_unit", "stress_unit"}),
            binding_identity=("document:policy", "native-policy", 3),
        ),
        **changes,
    )


def _published_names(state, **arguments) -> tuple[str, ...]:
    return tuple(item.name for item in publish_authoring_tools(state, **arguments))


def test_ready_tool_order_keeps_optional_creation_and_profile_tools_adjacent() -> None:
    state = _policy_state()

    assert _published_names(state, live_context=_native_context()) == _READY_TOOL_ORDER
    definitions = publish_authoring_tools(state, live_context=_native_context())
    requirements = next(item for item in definitions if item.name == "set_authoring_requirements")
    assert requirements == _stage_requirement_tool(
        "geometry",
        ("length_unit", "force_unit", "stress_unit"),
        {key: {"type": "string"} for key in ("length_unit", "force_unit", "stress_unit")},
    )
    save = next(item for item in definitions if item.name == "request_project_save")
    assert save.parameters == {
        "type": "object", "properties": {}, "required": [], "additionalProperties": False,
    }


@pytest.mark.parametrize("dimension", [None, 1, 2, 3])
def test_requirement_schema_uses_observed_active_dimension(dimension: int | None) -> None:
    observed = _native_context(dimension or 2)
    if dimension is None:
        observed = replace(observed, active_part_id=None)
    state = _policy_state(stage=AuthoringWorkflowStage.MESH_READY, observed_context=observed)
    tools = publish_authoring_tools(state, live_context=_native_context(3 if dimension == 1 else 1))
    requirement_tool = next(item for item in tools if item.name == "set_authoring_requirements")
    specs = {
        "mesh_cell_shape": {
            "type": "string",
            "enum": ["line"] if dimension == 1 else
            ["tetrahedron", "hexahedron"] if dimension == 3 else
            ["triangle", "quadrilateral"],
        },
        "mesh_order": {"type": "integer", "enum": [1] if dimension == 1 else [1, 2]},
        "mesh_global_size": {"type": "number", "exclusiveMinimum": 0},
    }
    if dimension == 1:
        specs["line_element_type"] = {"type": "string", "enum": ["Truss2", "Beam2"]}

    assert requirement_tool == _stage_requirement_tool("mesh", tuple(specs), specs)


@pytest.mark.parametrize(
    ("stage", "dimension", "required_names", "operation_names"),
    [
        (AuthoringWorkflowStage.REQUIREMENTS, 2,
         {"length_unit", "force_unit", "stress_unit"},
         {"prepare_geometry_proposal", "prepare_planar_construction_proposal"}),
        (AuthoringWorkflowStage.GEOMETRY_READY, 2,
         {"length_unit", "force_unit", "stress_unit"},
         {"prepare_geometry_proposal", "prepare_planar_construction_proposal"}),
        (AuthoringWorkflowStage.MESH_READY, 1,
         {"mesh_cell_shape", "mesh_order", "mesh_global_size", "line_element_type"},
         {"prepare_mesh_proposal"}),
        (AuthoringWorkflowStage.DEFINITIONS_READY, 3,
         {"mesh_cell_shape", "mesh_order", "mesh_global_size"},
         {"prepare_mesh_proposal"}),
    ],
)
def test_requirement_gates_require_every_current_group_key(
    stage, dimension, required_names, operation_names,
) -> None:
    state = _policy_state(
        stage=stage,
        observed_context=_native_context(dimension),
        registered_tool_names=frozenset(operation_names),
        proposed_requirement_keys=frozenset(required_names),
    )
    assert operation_names <= set(_published_names(state))
    for key in required_names:
        incomplete = replace(state, proposed_requirement_keys=state.proposed_requirement_keys - {key})
        assert not operation_names.intersection(_published_names(incomplete))


@pytest.mark.parametrize("group_names", _GROUPED_TOOLS)
def test_context_tool_groups_require_complete_registration(group_names) -> None:
    registered = frozenset(group_names)
    state = _policy_state(registered_tool_names=registered)
    assert publication_needs_live_context(state)
    assert registered <= set(_published_names(state, live_context=_native_context()))
    for missing in group_names:
        partial = replace(state, registered_tool_names=registered - {missing})
        assert not publication_needs_live_context(partial)
        assert not registered.intersection(_published_names(partial, live_context=_native_context()))


@pytest.mark.parametrize("live_context_kind", ["missing", "unsupported", "imported", "disabled"])
def test_context_tools_reject_unavailable_native_capabilities(live_context_kind: str) -> None:
    live = _native_context()
    if live_context_kind == "missing":
        live = None
    elif live_context_kind == "unsupported":
        live = replace(live, binding=replace(live.binding, supported=False))
    elif live_context_kind == "imported":
        live = replace(live, binding=replace(live.binding, source_kind="imported"))
    else:
        live = replace(live, capabilities=())

    names = set(_published_names(_policy_state(), live_context=live))
    assert not names.intersection(name for group in _GROUPED_TOOLS for name in group)


@pytest.mark.parametrize("suppressed", [False, True])
def test_geometry_edit_tools_remain_published_for_suppressed_parts_while_catalog_and_profiles_require_visible_parts(
    suppressed: bool,
) -> None:
    context = _native_context()
    context = replace(context, parts=(replace(context.parts[0], suppressed=suppressed),))
    names = set(_published_names(_policy_state(), live_context=context))

    assert {"read_geometry_edit_context", "prepare_geometry_edit"} <= names
    assert ("read_geometry_feature_catalog" in names) is not suppressed
    assert ("prepare_profile_extrusion" in names) is not suppressed


def test_mesh_eligibility_uses_observed_context_and_stage() -> None:
    observed = replace(_native_context(), mesh=MeshSummary(False, False))
    state = _policy_state(observed_context=observed)
    assert "apply_model_definition" not in _published_names(state, live_context=_native_context())
    assert "apply_model_definition" in _published_names(
        replace(state, observed_context=_native_context()), live_context=observed,
    )
    assert "apply_model_definition" in _published_names(
        replace(state, stage=AuthoringWorkflowStage.DEFINITIONS_READY), live_context=observed,
    )


@pytest.mark.parametrize("binding_matches", [False, True])
def test_geometry_edit_schema_is_scoped_to_current_binding(binding_matches: bool) -> None:
    state = _policy_state(geometry_edit_scope=GeometryEditSchemaScope(
        ("document:policy", "native-policy", 3) if binding_matches else
        ("document:old", "native-old", 1),
        "part-policy", "planar_sketch", ("add_circle",),
    ))
    edit = next(
        item for item in publish_authoring_tools(state, live_context=_native_context())
        if item.name == "prepare_geometry_edit"
    )
    assert edit == (
        _geometry_edit_tool_definition("part-policy", "planar_sketch", ("add_circle",))
        if binding_matches else _PREPARE_GEOMETRY_EDIT
    )


@pytest.mark.parametrize("stage", _READ_ONLY_STAGES)
def test_pending_and_terminal_stages_publish_only_context_and_registered_workspace(stage) -> None:
    state = _policy_state(stage=stage)

    assert not publication_needs_live_context(state)
    assert not publication_needs_workspace_export(state)
    assert _published_names(state, live_context=_native_context(), workspace_export_selected=True) == (
        "read_authoring_context", "read_workspace_documents",
    )


@pytest.mark.parametrize("stage", [AuthoringWorkflowStage.REQUIREMENTS, AuthoringWorkflowStage.RESULTS_READY])
def test_historical_result_tools_keep_stage_specific_order(stage) -> None:
    state = _policy_state(
        stage=stage,
        registered_tool_names=_RESULT_TOOLS | {"read_workspace_documents"},
        workspace_result_inventory=(3, 2),
    )
    assert publication_needs_workspace_export(state)
    names = _published_names(state, workspace_export_selected=True)
    expected = (
        "read_authoring_context", "set_authoring_requirements", "read_workspace_documents",
        "read_analysis_run_catalog", "read_accepted_result_catalog", "query_accepted_result",
        "read_result_display_context", "export_accepted_result_csv", "compare_accepted_results",
    ) if stage is AuthoringWorkflowStage.REQUIREMENTS else (
        "read_authoring_context", "set_authoring_requirements", "read_accepted_result_catalog",
        "query_accepted_result", "compare_accepted_results", "read_result_display_context",
        "export_accepted_result_csv", "read_workspace_documents", "read_analysis_run_catalog",
    )
    assert names == expected


@pytest.mark.parametrize(
    ("inventory", "expected_result_names"),
    [
        ((0, 0), ()),
        ((1, 0), ("read_analysis_run_catalog",)),
        ((2, 1), ("read_analysis_run_catalog", "read_accepted_result_catalog", "query_accepted_result",
                  "read_result_display_context", "export_accepted_result_csv")),
        ((3, 2), ("read_analysis_run_catalog", "read_accepted_result_catalog", "query_accepted_result",
                  "compare_accepted_results", "read_result_display_context", "export_accepted_result_csv")),
    ],
)
def test_no_observed_context_publishes_only_available_workspace_reads(inventory, expected_result_names) -> None:
    state = _policy_state(observed_context=None, workspace_result_inventory=inventory)

    assert not publication_needs_live_context(state)
    assert publication_needs_workspace_export(state) is (inventory[1] > 0)
    assert _published_names(state, live_context=_native_context(), workspace_export_selected=True) == (
        "read_workspace_documents", *expected_result_names,
    )


def test_result_counts_prefer_workspace_inventory_then_observed_context() -> None:
    observed = replace(_native_context(), run_count=2, result_count=2)
    state = _policy_state(observed_context=observed, registered_tool_names=_RESULT_TOOLS)

    assert _RESULT_TOOLS <= set(_published_names(
        state, live_context=_native_context(), workspace_export_selected=True,
    ))
    assert _published_names(replace(state, workspace_result_inventory=(0, 0))) == (
        "read_authoring_context", "set_authoring_requirements",
    )
    assert "export_accepted_result_csv" not in _published_names(state, workspace_export_selected=False)


def test_requirement_helpers_reject_unknown_groups_and_fields() -> None:
    with pytest.raises(ValueError, match="unknown requirement group"):
        requirement_keys("analysis", None)
    with pytest.raises(ValueError, match="unknown requirement field"):
        requirement_stage("unknown")
    with pytest.raises(KeyError):
        requirement_spec("unknown", None)


@pytest.mark.parametrize("stage", tuple(AuthoringWorkflowStage))
def test_save_policy_guards_ready_stages_registration_and_native_capability(stage) -> None:
    expected = stage not in _READ_ONLY_STAGES
    assert project_save_available(stage, frozenset({"request_project_save"}), _native_context()) is expected
    assert not project_save_available(stage, frozenset(), _native_context())
    assert not project_save_available(stage, frozenset({"request_project_save"}), None)
