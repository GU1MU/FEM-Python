from dataclasses import replace

import pytest

from fem.application import ModelSession, describe_session_authoring
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
    assert not busy[GuiActionKey.OPEN].enabled
    assert "后台任务" in busy[GuiActionKey.OPEN].reason


def test_native_targets_and_selection_drive_authoring_actions() -> None:
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
    assert selected[GuiActionKey.GEOMETRY_REGION].enabled
    assert selected[GuiActionKey.BOUNDARY_CREATE].enabled
    assert not selected[GuiActionKey.OUTPUT_CREATE].enabled
    assert "不会执行输出请求" in selected[GuiActionKey.OUTPUT_CREATE].reason


def test_geometry_selection_rejects_noncanonical_values() -> None:
    with pytest.raises(TypeError, match="LogicalEntityRef"):
        GuiActionContext(geometry_selection=("edge:left",))
