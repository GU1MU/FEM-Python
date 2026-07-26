from pathlib import Path

from fem.abaqus import read
from fem.application import (
    AuthoringStatus,
    ModelSession,
    RegionRef,
    describe_session_authoring,
)
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
