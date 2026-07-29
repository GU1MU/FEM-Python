from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from fem.application import (
    ModelSession,
    NamedRegion,
    TopologyResolutionError,
    describe_session_authoring,
)
from fem.geometry import ExtrudedGeometry, LogicalEntityRef
from fem_gui.action_state import (
    GuiActionContext,
    GuiActionKey,
    derive_action_availability,
)
from fem_gui.commands import NativeGeometryEdit
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from fem_gui.geometry_preview import build_geometry_preview
from fem_gui.preprocessing_dialogs import ExtrudeGeometryDialog
from fem_gui.widgets.viewport import FEMViewport
from tests.geometry.test_profile_extrusion import (
    hole_profile_sketch,
    profile_face_id,
    two_profile_sketch,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _extrude_state(recipe, selection=()):
    session = ModelSession()
    session.new_native_project()
    session.add_native_part(recipe)
    snapshot = session.snapshot()
    states = {
        state.key: state
        for state in derive_action_availability(
            snapshot,
            describe_session_authoring(snapshot),
            GuiActionContext(geometry_selection=selection),
        )
    }
    return states[GuiActionKey.GEOMETRY_EXTRUDE]


def test_multi_profile_action_requires_valid_face_selection() -> None:
    sketch = two_profile_sketch()
    first = LogicalEntityRef(profile_face_id(sketch, "L1"))

    unselected = _extrude_state(sketch)
    selected = _extrude_state(sketch, (first,))
    wrong_kind = _extrude_state(
        sketch,
        (LogicalEntityRef("edge:L1"),),
    )

    assert not unselected.enabled
    assert "多个 Profile" in unselected.reason
    assert selected.enabled
    assert not wrong_kind.enabled
    assert "非面" in wrong_kind.reason


def test_dialog_hides_source_descriptions_and_builds_selected_recipe() -> None:
    _application()
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")
    second = profile_face_id(sketch, "L5")

    dialog = ExtrudeGeometryDialog(
        sketch,
        source_face_ids=(second, first),
    )
    dialog.height_spin.setValue(4.5)
    recipe = dialog.recipe()

    label_texts = {label.text() for label in dialog.findChildren(QLabel)}
    assert "源 Profiles" not in label_texts
    assert "+Z（Phase 2）" not in label_texts
    assert "+Z" in label_texts
    assert recipe.height == 4.5
    assert recipe.source_face_ids == (first, second)
    dialog.close()


def test_extruded_profile_caps_sides_and_body_are_pickable() -> None:
    _application()
    sketch = hole_profile_sketch()
    source = profile_face_id(sketch, "L1")
    preview = build_geometry_preview(
        ExtrudedGeometry(sketch, 2.0, (source,))
    )
    viewport = FEMViewport()

    viewport._install_geometry_pick_bindings(preview)

    for logical_id in (
        "face:bottom",
        "face:top",
        "face:side/L1",
        "face:side/L5",
        "body:domain",
    ):
        assert (
            LogicalEntityRef(logical_id)
            in viewport._geometry_ref_to_pick_ids
        )
    viewport.close()


def test_gui_extrusion_commits_selected_profile_and_clears_selection(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")
    window._set_native_geometry(sketch, "测试草图")
    window._set_geometry_selection_mode("face")
    window._selected_geometry_refs = {LogicalEntityRef(first)}
    window._update_action_states()
    assert window.actions["geometry_extrude"].isEnabled()
    monkeypatch.setattr(window, "_exec_dialog", lambda _dialog: True)

    window.extrude_geometry()

    recipe = window.document.geometry_recipe
    assert isinstance(recipe, ExtrudedGeometry)
    assert recipe.source_face_ids == (first,)
    assert window._selected_geometry_refs == set()
    assert window._geometry_selection_mode == "body"
    window.close()


def test_gui_extrusion_cancel_preserves_recipe_and_face_selection(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    sketch = two_profile_sketch()
    first_ref = LogicalEntityRef(profile_face_id(sketch, "L1"))
    window._set_native_geometry(sketch, "测试草图")
    window._set_geometry_selection_mode("face")
    window._selected_geometry_refs = {first_ref}
    monkeypatch.setattr(window, "_exec_dialog", lambda _dialog: False)

    window.extrude_geometry()

    assert window.document.geometry_recipe == sketch
    assert window._selected_geometry_refs == {first_ref}
    window.close()


def test_gui_extrusion_revision_conflict_preserves_face_selection(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    sketch = two_profile_sketch()
    first_ref = LogicalEntityRef(profile_face_id(sketch, "L1"))
    window._set_native_geometry(sketch, "测试草图")
    window._set_geometry_selection_mode("face")
    window._selected_geometry_refs = {first_ref}
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )

    def mutate_while_dialog_is_open(_dialog) -> bool:
        window.session.rename_native_part(
            "P1",
            "并发修改的部件",
        )
        return True

    monkeypatch.setattr(
        window,
        "_exec_dialog",
        mutate_while_dialog_is_open,
    )

    window.extrude_geometry()

    assert window.document.geometry_recipe == sketch
    assert window._selected_geometry_refs == {first_ref}
    assert errors
    assert errors[0][0] == "编辑几何"
    window.close()


def test_geometry_edit_prunes_stale_profile_selection() -> None:
    _application()
    window = FEMMainWindow()
    sketch = two_profile_sketch()
    first_ref = LogicalEntityRef(profile_face_id(sketch, "L1"))
    remaining = replace(
        sketch,
        points=sketch.points[4:],
        curves=sketch.curves[4:],
    )
    window._set_native_geometry(sketch, "测试草图")
    window._set_geometry_selection_mode("face")
    window._selected_geometry_refs = {first_ref}

    receipt = window.apply_native_geometry_edit(
        NativeGeometryEdit(
            base_session_revision=window.document.session_revision,
            parts=tuple(window.document.parts),
            recipe=remaining,
        )
    )

    assert receipt.diagnostic is None
    assert window.document.geometry_recipe == remaining
    assert window._selected_geometry_refs == set()
    window.close()


def test_occ_preflight_failure_keeps_recipe_selection_and_dialog(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    sketch = two_profile_sketch()
    first_ref = LogicalEntityRef(profile_face_id(sketch, "L1"))
    window._set_native_geometry(sketch, "测试草图")
    window._set_geometry_selection_mode("face")
    window._selected_geometry_refs = {first_ref}
    dialog_results = iter((True, False))
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_exec_dialog",
        lambda _dialog: next(dialog_results),
    )
    monkeypatch.setattr(
        window,
        "_preflight_extruded_geometry",
        lambda _recipe: (_ for _ in ()).throw(
            TopologyResolutionError("injected OCC failure")
        ),
    )
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )

    window.extrude_geometry()

    assert window.document.geometry_recipe == sketch
    assert window._selected_geometry_refs == {first_ref}
    assert errors
    assert "injected OCC failure" in errors[0][1]
    window.close()


def test_occ_preflight_failure_releases_temporary_runtime(
    monkeypatch,
) -> None:
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")
    recipe = ExtrudedGeometry(sketch, 1.0, (first,))
    released: list[bool] = []

    class FakeRuntime:
        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            released.append(True)

    monkeypatch.setattr(
        main_window_module.geometry_runtime,
        "model",
        lambda *_args, **_kwargs: FakeRuntime(),
    )
    monkeypatch.setattr(
        main_window_module,
        "compile_recipe",
        lambda *_args: (_ for _ in ()).throw(
            TopologyResolutionError("injected OCC failure")
        ),
    )

    with pytest.raises(
        TopologyResolutionError,
        match="injected OCC failure",
    ):
        FEMMainWindow._preflight_extruded_geometry(recipe)

    assert released == [True]


def test_profile_change_preserves_only_surviving_part_regions() -> None:
    _application()
    window = FEMMainWindow()
    sketch = two_profile_sketch()
    first = profile_face_id(sketch, "L1")
    second = profile_face_id(sketch, "L5")
    window._set_native_geometry(
        ExtrudedGeometry(sketch, 1.0, (first,)),
        "测试拉伸体",
    )
    window._apply_session_delta(
        window.session.replace_named_regions(
            (
                NamedRegion(
                    "Body",
                    (LogicalEntityRef("body:P1/domain"),),
                ),
                NamedRegion(
                    "Side",
                    (
                        LogicalEntityRef(
                            "face:P1/side/L1"
                        ),
                    ),
                ),
            )
        )
    )

    window._set_native_geometry(
        ExtrudedGeometry(sketch, 1.0, (second,)),
        "选择修改后的",
    )

    assert set(window.document.named_regions) == {"Body"}
    assert "选择修改后的几何已创建" in window.status_panel.state_label.text()
    window.close()
