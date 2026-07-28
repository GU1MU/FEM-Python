from dataclasses import replace

import pytest

from fem.application import ModelSession, describe_session_authoring
from fem.application.results import FieldState
from fem.core.model import AnalysisStep
from fem.geometry import LogicalEntityRef, RectangleGeometry
from fem.mesh.settings import MeshSettings
from fem_gui.action_state import (
    ACTION_DESCRIPTORS,
    GuiActionContext,
    GuiActionKey,
    derive_action_availability,
)


def _by_key(values):
    return {item.key: item for item in values}


def _result_action_states(context: GuiActionContext):
    base = ModelSession().snapshot()
    snapshot = replace(
        base,
        displayed_result=object(),  # type: ignore[arg-type]
    )
    return _by_key(
        derive_action_availability(
            snapshot,
            describe_session_authoring(base),
            context,
        )
    )


def test_registry_and_projector_cover_every_action_exactly_once() -> None:
    snapshot = ModelSession().snapshot()
    values = derive_action_availability(
        snapshot,
        describe_session_authoring(snapshot),
        GuiActionContext(),
    )

    assert tuple(item.key for item in ACTION_DESCRIPTORS) == tuple(
        item.key for item in values
    )
    assert len(values) == len(GuiActionKey)
    assert {item.key for item in values} == set(GuiActionKey)


def test_closed_and_busy_context_are_projected_without_qt() -> None:
    snapshot = ModelSession().snapshot()
    idle = _by_key(
        derive_action_availability(
            snapshot,
            describe_session_authoring(snapshot),
            GuiActionContext(),
        )
    )
    busy = _by_key(
        derive_action_availability(
            snapshot,
            describe_session_authoring(snapshot),
            GuiActionContext(busy=True),
        )
    )

    assert idle[GuiActionKey.OPEN].enabled
    assert not idle[GuiActionKey.GEOMETRY_SKETCH].enabled
    assert "请先新建模型" in idle[GuiActionKey.GEOMETRY_SKETCH].reason
    assert not idle[GuiActionKey.OUTPUT_CREATE].enabled
    assert not busy[GuiActionKey.OPEN].enabled
    assert "后台任务" in busy[GuiActionKey.OPEN].reason


def test_sketch_editor_context_gates_mutating_actions() -> None:
    session = ModelSession()
    session.new_native_project()
    snapshot = session.snapshot()
    states = _by_key(
        derive_action_availability(
            snapshot,
            describe_session_authoring(snapshot),
            GuiActionContext(sketch_editor_active=True),
        )
    )

    assert not states[GuiActionKey.GEOMETRY_CREATE].enabled
    assert not states[GuiActionKey.OPEN_PROJECT].enabled
    assert not states[GuiActionKey.MESH_SETTINGS].enabled
    assert "草图编辑" in states[GuiActionKey.GEOMETRY_CREATE].reason
    assert states[GuiActionKey.TOP].enabled


def test_native_scope_actions_remain_disabled_until_meshing() -> None:
    session = ModelSession()
    session.new_native_project()
    base = session.snapshot()
    snapshot = replace(
        base,
        geometry_recipe=RectangleGeometry("Part-1", 4.0, 2.0),
        mesh_settings=MeshSettings(size=0.4),
        steps=(AnalysisStep("Step-1"),),
        can_save=True,
    )
    projection = describe_session_authoring(snapshot)
    unselected = _by_key(
        derive_action_availability(
            snapshot,
            projection,
            GuiActionContext(selected_step_name="Step-1"),
        )
    )
    selected = _by_key(
        derive_action_availability(
            snapshot,
            projection,
            GuiActionContext(
                selected_step_name="Step-1",
                geometry_selection=(LogicalEntityRef("edge:left"),),
            ),
        )
    )

    assert unselected[GuiActionKey.SAVE_PROJECT].enabled
    assert unselected[GuiActionKey.MESH_GENERATE].enabled
    assert not unselected[GuiActionKey.GEOMETRY_REGION].enabled
    assert not selected[GuiActionKey.GEOMETRY_REGION].enabled
    assert not selected[GuiActionKey.BOUNDARY_CREATE].enabled
    assert selected[GuiActionKey.OUTPUT_CREATE].enabled

    busy = _by_key(
        derive_action_availability(
            snapshot,
            projection,
            GuiActionContext(busy=True),
        )
    )
    assert not busy[GuiActionKey.OUTPUT_CREATE].enabled


def test_geometry_selection_rejects_noncanonical_values() -> None:
    with pytest.raises(TypeError, match="LogicalEntityRef"):
        GuiActionContext(geometry_selection=("edge:left",))


def test_result_action_descriptors_use_canonical_export_keys_and_handlers() -> None:
    descriptors = {item.key: item for item in ACTION_DESCRIPTORS}

    assert descriptors[GuiActionKey.FIELD].handler == ("show_result_display_dialog")
    assert (
        descriptors[GuiActionKey.QUERY].handler
        == "show_result_query_dialog"
    )
    assert descriptors[GuiActionKey.EXPORT_CSV].handler == "export_csv"
    assert descriptors[GuiActionKey.EXPORT_VTK].handler == "export_vtk"
    assert descriptors[GuiActionKey.SCREENSHOT].handler == ("export_viewport_image")
    assert tuple(
        key.value
        for key in (
            GuiActionKey.EXPORT_CSV,
            GuiActionKey.EXPORT_VTK,
            GuiActionKey.SCREENSHOT,
        )
    ) == ("export_csv", "export_vtk", "screenshot")
    assert "export" not in {key.value for key in GuiActionKey}


@pytest.mark.parametrize(
    ("changes", "field_enabled", "query_enabled", "export_enabled"),
    (
        ({}, True, True, True),
        ({"result_source_current": False}, False, False, False),
        ({"catalog_available": False}, False, False, False),
        ({"busy": True}, False, False, False),
        ({"materialization_pending": True}, False, False, False),
        ({"result_task_busy": True}, False, False, False),
        (
            {
                "selected_field_exists": False,
                "selected_field_state": None,
            },
            True,
            True,
            False,
        ),
        ({"selected_field_state": FieldState.LAZY}, True, True, False),
        (
            {"selected_field_state": FieldState.UNAVAILABLE},
            True,
            True,
            False,
        ),
        ({"selected_field_state": None}, True, True, False),
    ),
)
def test_result_action_readiness_is_a_typed_fact_truth_table(
    changes: dict[str, object],
    field_enabled: bool,
    query_enabled: bool,
    export_enabled: bool,
) -> None:
    context = replace(GuiActionContext(), **changes)

    states = _result_action_states(context)

    assert states[GuiActionKey.FIELD].enabled is field_enabled
    assert states[GuiActionKey.QUERY].enabled is query_enabled
    assert states[GuiActionKey.EXPORT_CSV].enabled is export_enabled
    assert states[GuiActionKey.EXPORT_VTK].enabled is export_enabled
    for key in (
        GuiActionKey.FIELD,
        GuiActionKey.QUERY,
        GuiActionKey.EXPORT_CSV,
        GuiActionKey.EXPORT_VTK,
    ):
        if not states[key].enabled:
            assert states[key].reason


def test_query_can_submit_a_lazy_field_for_materialization() -> None:
    states = _result_action_states(
        GuiActionContext(
            selected_field_state=FieldState.LAZY,
            materialization_pending=False,
            result_task_busy=False,
        )
    )

    assert states[GuiActionKey.FIELD].enabled
    assert states[GuiActionKey.QUERY].enabled
    assert not states[GuiActionKey.EXPORT_CSV].enabled
    assert not states[GuiActionKey.EXPORT_VTK].enabled


@pytest.mark.parametrize(
    (
        "scene_available",
        "backend_available",
        "capture_active",
        "busy",
        "expected",
    ),
    (
        (True, True, False, False, True),
        (True, True, False, True, True),
        (False, True, False, False, False),
        (True, False, False, False, False),
        (True, True, True, False, False),
    ),
)
def test_screenshot_depends_only_on_capture_scene_and_backend_facts(
    scene_available: bool,
    backend_available: bool,
    capture_active: bool,
    busy: bool,
    expected: bool,
) -> None:
    snapshot = ModelSession().snapshot()
    states = _by_key(
        derive_action_availability(
            snapshot,
            describe_session_authoring(snapshot),
            GuiActionContext(
                busy=busy,
                viewport_scene_available=scene_available,
                display_backend_available=backend_available,
                viewport_capture_active=capture_active,
                result_source_current=False,
                catalog_available=False,
            ),
        )
    )

    assert states[GuiActionKey.SCREENSHOT].enabled is expected
    if not expected:
        assert states[GuiActionKey.SCREENSHOT].reason


@pytest.mark.parametrize(
    "field_name",
    (
        "busy",
        "display_backend_available",
        "viewport_capture_active",
        "result_source_current",
        "catalog_available",
        "selected_field_exists",
        "materialization_pending",
        "result_task_busy",
        "viewport_scene_available",
    ),
)
def test_action_context_rejects_non_boolean_result_facts(
    field_name: str,
) -> None:
    with pytest.raises(TypeError, match=field_name):
        GuiActionContext(**{field_name: 1})  # type: ignore[arg-type]


def test_action_context_requires_typed_selected_field_state() -> None:
    with pytest.raises(TypeError, match="selected_field_state"):
        GuiActionContext(
            selected_field_state="ready",  # type: ignore[arg-type]
        )
