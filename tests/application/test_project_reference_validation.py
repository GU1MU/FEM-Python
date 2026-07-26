from __future__ import annotations

from dataclasses import replace

import pytest

from fem.application import ModelSession, NamedRegion, NativePart, ProjectSnapshot
from fem.application.feature_history import derive_feature_history
from fem.application.project_validation import NativeProjectValidationError
from fem.core.model import AnalysisStep, DisplacementConstraint
from fem.geometry import LogicalEntityRef
from fem.geometry.recipes import (
    BoxGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
)
from fem.mesh.settings import LocalMeshControl, MeshSettings, MeshSizeFalloff


def _rectangle_snapshot() -> ProjectSnapshot:
    recipe = RectangleGeometry("Plate", 4.0, 2.0)
    return ProjectSnapshot(
        source_kind="native",
        parts=(NativePart(),),
        geometry_recipe=recipe,
        mesh_settings=MeshSettings(
            0.5,
            local_controls=(
                LocalMeshControl(LogicalEntityRef("edge:bottom"), 0.2),
            ),
        ),
        feature_history=derive_feature_history(recipe),
        named_regions=(
            NamedRegion(
                "Fixed",
                (LogicalEntityRef("edge:left"),),
            ),
        ),
    )


def test_valid_snapshot_install_owns_canonical_logical_references() -> None:
    session = ModelSession()

    session.replace_from_snapshot(_rectangle_snapshot())
    installed = session.snapshot()

    assert installed.named_regions["Fixed"].references == (
        LogicalEntityRef("edge:left"),
    )
    assert installed.mesh_settings.local_controls[0].target == (
        LogicalEntityRef("edge:bottom")
    )
    assert not installed.dirty


@pytest.mark.parametrize(
    "mutation",
    (
        lambda snapshot: replace(
            snapshot,
            named_regions=(
                NamedRegion(
                    "Missing",
                    (LogicalEntityRef("edge:missing"),),
                ),
            ),
        ),
        lambda snapshot: replace(
            snapshot,
            mesh_settings=MeshSettings(
                0.5,
                local_controls=(
                    LocalMeshControl(
                        LogicalEntityRef("edge:missing"),
                        0.2,
                    ),
                ),
            ),
        ),
    ),
    ids=("unknown-region-reference", "unknown-control-reference"),
)
def test_invalid_snapshot_install_is_atomic(mutation) -> None:
    session = ModelSession()
    session.replace_from_snapshot(_rectangle_snapshot())
    before = session.snapshot()

    with pytest.raises(
        (NativeProjectValidationError, ValueError),
        match="unknown logical reference",
    ):
        session.replace_from_snapshot(mutation(_rectangle_snapshot()))

    assert session.snapshot() == before


def test_replace_named_regions_failure_is_atomic() -> None:
    session = ModelSession()
    session.replace_from_snapshot(_rectangle_snapshot())
    before = session.snapshot()

    with pytest.raises(ValueError, match="unknown logical reference"):
        session.replace_named_regions(
            (
                NamedRegion(
                    "Invalid",
                    (LogicalEntityRef("edge:missing"),),
                ),
            )
        )

    assert session.snapshot() == before


def test_replace_mesh_settings_failure_is_atomic() -> None:
    session = ModelSession()
    session.replace_from_snapshot(_rectangle_snapshot())
    before = session.snapshot()

    with pytest.raises(ValueError, match="unknown logical reference"):
        session.replace_mesh_settings(
            MeshSettings(
                0.5,
                local_controls=(
                    LocalMeshControl(
                        LogicalEntityRef("edge:missing"),
                        0.1,
                    ),
                ),
            )
        )

    assert session.snapshot() == before


def test_same_fingerprint_geometry_edit_preserves_target_radius_profile() -> None:
    first = PlateWithHoleGeometry("Plate", 6.0, 4.0, 3.0, 2.0, 0.5)
    second = PlateWithHoleGeometry("Plate", 8.0, 5.0, 4.0, 2.5, 0.75)
    hole = LogicalEntityRef("edge:hole-loop")
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry((NativePart(),), first)
    session.replace_named_regions((NamedRegion("HoleBoundary", (hole,)),))
    session.replace_mesh_settings(
        MeshSettings(
            0.5,
            local_controls=(
                LocalMeshControl(
                    hole,
                    0.1,
                    MeshSizeFalloff("target_radius", 0.25, 2.0),
                ),
            ),
        )
    )

    session.replace_geometry((NativePart(),), second)
    after = session.snapshot()

    assert after.named_regions["HoleBoundary"].references == (hole,)
    assert after.mesh_settings.local_controls == (
        LocalMeshControl(
            hole,
            0.1,
            MeshSizeFalloff("target_radius", 0.25, 2.0),
        ),
    )


def test_fingerprint_change_clears_every_geometry_reference() -> None:
    session = ModelSession()
    session.replace_from_snapshot(_rectangle_snapshot())

    session.replace_geometry(
        (NativePart(),),
        BoxGeometry("Block", 4.0, 2.0, 1.0),
    )
    after = session.snapshot()

    assert not after.named_regions
    assert after.mesh_settings.local_controls == ()


@pytest.mark.parametrize("target", [1, True])
def test_integer_analysis_target_rejected_before_session_mutation(
    target: object,
) -> None:
    session = ModelSession()
    session.replace_from_snapshot(_rectangle_snapshot())
    before = session.snapshot()
    step = AnalysisStep(
        "Load",
        boundaries=(DisplacementConstraint(target, 1, 2),),
    )

    with pytest.raises(ValueError, match="stable region name"):
        session.replace_model_definitions((), (), (), (step,))

    assert session.snapshot() == before
