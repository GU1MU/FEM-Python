from pathlib import Path
from dataclasses import replace

from fem.abaqus import read
from fem.application import (
    AnalysisRun,
    AuthoringStatus,
    ModelSession,
    RegionRef,
    RunStatus,
    describe_session_authoring,
)
from fem.core.model import AnalysisStep, OutputRequest
from fem.geometry import RectangleGeometry
from fem.mesh.settings import MeshSettings


_FIXTURES = (
    Path(__file__).parents[1]
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)


def test_closed_session_projection_is_typed_and_empty() -> None:
    projection = describe_session_authoring(ModelSession().snapshot())

    assert projection.targets == ()
    assert projection.step_lifecycle == ()
    assert projection.report.status is AuthoringStatus.UNAVAILABLE


def test_native_projection_preserves_target_namespaces_and_order() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry(
        (),
        RectangleGeometry("Part-1", 4.0, 2.0),
    )
    session.replace_mesh_settings(MeshSettings(size=0.4))

    projection = describe_session_authoring(session.snapshot())
    target_refs = tuple(target.region for target in projection.targets)

    assert RegionRef("element_set", "DOMAIN") in target_refs
    assert RegionRef("node_set", "LEFT") in target_refs
    assert RegionRef("edge", "LEFT") in target_refs
    assert target_refs == tuple(
        sorted(
            target_refs,
            key=lambda item: (
                {"node_set": 0, "element_set": 1, "edge": 2, "surface": 3}[
                    item.kind
                ],
                item.name.casefold(),
                item.name,
            ),
        )
    )
    assert projection.target(
        RegionRef("node_set", "LEFT")
    ).operation("boundary.displacement").can_submit


def test_imported_beam_projection_uses_canonical_model_regions() -> None:
    path = _FIXTURES / "beam2_rectangle_uniform_load.inp"
    session = ModelSession()
    task = session.prepare_import(path)
    session.accept_imported_model(task.token, read(path))

    projection = describe_session_authoring(session.snapshot())
    beam = projection.target(RegionRef("element_set", "BEAM"))

    assert beam.operation("load.line.global").can_submit
    assert beam.operation("load.line.local").can_submit
    assert projection.step("UniformLoad") is not None
    assert projection.step("UniformLoad").can_check
    assert not projection.step("UniformLoad").can_submit


def test_authoring_capability_separates_enter_and_submit_semantics() -> None:
    path = _FIXTURES / "beam2_rectangle_tip_load.inp"
    session = ModelSession()
    task = session.prepare_import(path)
    session.accept_imported_model(task.token, read(path))

    projection = describe_session_authoring(session.snapshot())
    capability = projection.report.region(
        RegionRef("element_set", "BEAM")
    ).operation("section.rectangle")

    assert capability.status in {
        AuthoringStatus.ENABLED,
        AuthoringStatus.LIMITED,
    }
    assert capability.can_enter
    assert capability.can_submit is (
        capability.status is AuthoringStatus.ENABLED
    )


def test_output_operations_are_session_contextual_and_independent() -> None:
    path = _FIXTURES / "truss2_tension.inp"
    session = ModelSession()
    task = session.prepare_import(path)
    session.accept_imported_model(task.token, read(path))

    projection = describe_session_authoring(session.snapshot())

    assert tuple(item.operation for item in projection.operations) == (
        "output_request.create",
        "output_request.view",
        "output_request.delete",
    )
    assert projection.operation(
        "output_request.create"
    ).status is AuthoringStatus.ENABLED
    assert projection.operation(
        "output_request.view"
    ).status is AuthoringStatus.READ_ONLY
    assert not projection.operation("output_request.view").can_submit
    assert projection.operation(
        "output_request.delete"
    ).status is AuthoringStatus.ENABLED
    assert (
        projection.output_request_catalog
        is projection.report.output_request_catalog
    )


def test_unsupported_existing_output_remains_viewable_and_deletable() -> None:
    path = _FIXTURES / "truss2_tension.inp"
    session = ModelSession()
    task = session.prepare_import(path)
    session.accept_imported_model(task.token, read(path))
    snapshot = replace(
        session.snapshot(),
        steps=(
            AnalysisStep(
                "Tension",
                outputs=(
                    OutputRequest(
                        "history",
                        "preselect",
                        ("Future",),
                    ),
                ),
            ),
        ),
    )

    projection = describe_session_authoring(snapshot)

    assert projection.operation(
        "output_request.view"
    ).status is AuthoringStatus.READ_ONLY
    assert projection.operation(
        "output_request.delete"
    ).status is AuthoringStatus.ENABLED


def test_output_create_and_delete_require_idle_but_view_does_not() -> None:
    path = _FIXTURES / "truss2_tension.inp"
    session = ModelSession()
    task = session.prepare_import(path)
    session.accept_imported_model(task.token, read(path))
    snapshot = session.snapshot()
    busy = replace(
        snapshot,
        runs=(
            AnalysisRun(
                run_id="run-1",
                name="Job-1",
                step_name="Tension",
                artifact_id=snapshot.artifact.artifact_id,
                model_revision=snapshot.model_revision,
                status=RunStatus.RUNNING,
            ),
        ),
    )

    projection = describe_session_authoring(busy)

    assert projection.operation(
        "output_request.create"
    ).status is AuthoringStatus.UNAVAILABLE
    assert projection.operation(
        "output_request.view"
    ).status is AuthoringStatus.READ_ONLY
    assert projection.operation(
        "output_request.delete"
    ).status is AuthoringStatus.UNAVAILABLE


def test_output_operations_require_step_and_request_existence() -> None:
    path = _FIXTURES / "truss2_tension.inp"
    session = ModelSession()
    task = session.prepare_import(path)
    session.accept_imported_model(task.token, read(path))
    snapshot = session.snapshot()

    empty_step = describe_session_authoring(
        replace(snapshot, steps=(AnalysisStep("Tension"),))
    )
    assert empty_step.operation(
        "output_request.create"
    ).status is AuthoringStatus.ENABLED
    assert empty_step.operation(
        "output_request.view"
    ).status is AuthoringStatus.UNAVAILABLE
    assert empty_step.operation(
        "output_request.delete"
    ).status is AuthoringStatus.UNAVAILABLE

    no_step = describe_session_authoring(replace(snapshot, steps=()))
    assert no_step.operation(
        "output_request.create"
    ).status is AuthoringStatus.UNAVAILABLE
