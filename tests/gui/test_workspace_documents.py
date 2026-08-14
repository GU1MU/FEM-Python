from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fem.application import ModelSession
from fem_gui.main_window import FEMMainWindow
from fem_gui.workspace import FEMWorkspace, canonical_path


def test_workspace_allocates_monotonic_ids_and_keeps_path_indexes(
    tmp_path,
    gui_application,
):
    workspace = FEMWorkspace()
    model_path = tmp_path / "models" / "beam.femproj"
    first = workspace.add_model(source_path=model_path)
    second = workspace.add_model(source_path=tmp_path / "other.femproj")
    duplicate = workspace.add_model(
        session=ModelSession(),
        source_path=model_path.parent / "." / model_path.name,
    )

    assert first.document_id == 1
    assert second.document_id == 2
    assert duplicate is first
    assert workspace.model_paths[canonical_path(model_path)] == first.document_id
    assert workspace.document(first.document_id) is first

    workspace.remove(first)
    replacement = workspace.add_model(source_path=model_path)
    assert replacement.document_id == 3
    assert workspace.model_paths[canonical_path(model_path)] == replacement.document_id


def test_workspace_add_activate_remove_and_result_indexes(
    tmp_path,
    gui_application,
):
    workspace = FEMWorkspace()
    model = workspace.add_model(display_name="Model-A")
    result = workspace.add_result(
        display_name="Result-A",
        source_path=tmp_path / "result.femres",
    )

    assert workspace.active_document() is model
    assert workspace.active_kind == "model"
    assert workspace.activate(result.document_id) is result
    assert workspace.active_kind == "result"
    assert workspace.active_document() is result

    removed = workspace.remove(result.document_id)
    assert removed is result
    assert workspace.active_document() is model
    assert result.document_id not in workspace.results
    assert result.document_id not in workspace.result_paths.values()

    workspace.remove(model.document_id)
    assert workspace.active_document() is None
    assert workspace.active_document_id is None


def test_workspace_projection_update_preserves_identity_and_revision(
    gui_application,
):
    session = ModelSession()
    initial = session.projection_snapshot()
    workspace = FEMWorkspace()
    context = workspace.add_model(session=session, projection=initial)

    assert context.session is session
    assert context.projection is initial
    assert context.revision == session.session_revision == initial.session_revision

    session.new_native_project("identity")
    updated = session.projection_snapshot(initial)
    workspace.update_projection(context, updated)
    assert context.projection is updated
    assert context.revision == updated.session_revision == session.session_revision
    assert context.display_name == "identity"


def test_projection_path_changes_keep_duplicate_lookup_and_remove_consistent(
    tmp_path,
    gui_application,
):
    session = ModelSession()
    workspace = FEMWorkspace()
    initial = session.projection_snapshot()
    context = workspace.add_model(session=session, projection=initial)
    first_path = tmp_path / "first.femproj"
    second_path = tmp_path / "second.femproj"

    first = replace(initial, source_path=first_path, project_path=first_path)
    workspace.update_projection(context, first)
    assert workspace.model_paths[canonical_path(first_path)] == context.document_id
    assert (
        workspace.add_model(source_path=first_path.parent / "." / first_path.name)
        is context
    )

    second = replace(first, source_path=second_path, project_path=second_path)
    workspace.update_projection(context, second)
    assert canonical_path(first_path) not in workspace.model_paths
    assert workspace.model_paths[canonical_path(second_path)] == context.document_id
    assert workspace.add_model(source_path=second_path) is context

    workspace.remove(context)
    assert canonical_path(second_path) not in workspace.model_paths


def test_unnamed_result_context_uses_result_identity(gui_application):
    workspace = FEMWorkspace()
    result = workspace.add_result()
    model = workspace.add_model()

    assert result.display_name == f"Result-{result.document_id}"
    assert model.display_name == "模型-1"


def test_model_default_names_use_model_sequence_across_result_ids(gui_application):
    workspace = FEMWorkspace()
    first = workspace.add_model()
    workspace.add_result()
    second = workspace.add_model()

    assert first.display_name == "模型-1"
    assert second.display_name == "模型-2"


def test_model_default_number_skips_imported_model_names(
    tmp_path,
    gui_application,
):
    workspace = FEMWorkspace()
    workspace.add_model(display_name="模型-1")
    workspace.add_model(
        display_name="模型-2",
        source_path=tmp_path / "imported.fempy",
    )

    assert workspace.model_name_exists(" 模型-2 ")
    assert workspace.next_model_number == 3


def test_workspace_disambiguates_real_model_and_result_names(gui_application, tmp_path):
    workspace = FEMWorkspace()

    first_model = workspace.add_model(display_name="模型-1")
    second_model = workspace.add_model(display_name="模型-1")
    first_result = workspace.add_result(
        display_name="plate",
        source_path=tmp_path / "a" / "plate.femres",
    )
    second_result = workspace.add_result(
        display_name="plate",
        source_path=tmp_path / "b" / "plate.femres",
    )

    assert first_model.display_name == "模型-1"
    assert second_model.display_name == "模型-1(1)"
    assert first_result.display_name == "plate"
    assert second_result.display_name == "plate(1)"


def test_workspace_job_numbers_are_global_and_never_reused(gui_application):
    workspace = FEMWorkspace()

    assert workspace.next_job_name() == "作业-1"
    workspace.remember_job_name("作业-1")
    assert workspace.next_job_name() == "作业-2"
    assert workspace.job_name_exists("作业-1")
    workspace.remember_job_name("作业-7")
    assert workspace.next_job_name() == "作业-8"


def test_idle_contexts_do_not_create_threads(gui_application):
    workspace = FEMWorkspace()
    model = workspace.add_model()
    result = workspace.add_result()

    assert workspace.open_controller is None
    assert not model.task_controller.busy
    assert not result.task_controller.busy
    assert model.task_controller.current_task_id is None
    assert result.task_controller.current_task_id is None


def test_main_window_aliases_active_workspace_context(
    gui_application,
    dispose_gui_widget,
):
    window = FEMMainWindow()
    context = window.workspace.active_document()
    assert context is not None
    assert window.session is context.session
    assert window.document is context.projection
    assert window.task_controller is context.task_controller

    delta = window.session.new_native_project("alias")
    assert window._apply_session_delta(delta)
    assert context.projection is window.document
    assert context.projection.session_revision == window.session.session_revision

    dispose_gui_widget(window)
