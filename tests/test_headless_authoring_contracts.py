"""Contract tests for authoring state extracted from the GUI package."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict

import pytest

from fem.application.definitions import (
    FeatureRecord,
    NamedRegion,
    NativePart,
    RegionAssignment,
    SectionDefinition,
)
from fem.geometry import LogicalEntityRef
from fem.geometry import recipes
from fem.mesh.settings import LocalMeshControl, MeshSettings


def test_gui_preprocessing_reexports_the_headless_contract_types() -> None:
    from fem_gui import preprocessing

    names = (
        "RectangleGeometry",
        "DiskGeometry",
        "BoxGeometry",
        "CylinderGeometry",
        "PlateWithHoleGeometry",
        "SketchRectangle",
        "SketchCircle",
        "SketchGeometry",
        "MovedGeometry",
        "RotatedGeometry",
        "ExtrudedGeometry",
        "BooleanGeometry",
    )
    for name in names:
        assert getattr(preprocessing, name) is getattr(recipes, name)
    assert preprocessing.LocalMeshControl is LocalMeshControl
    assert preprocessing.MeshSettings is MeshSettings


@pytest.mark.parametrize(
    ("value", "field_name"),
    (
        (NativePart(), "name"),
        (FeatureRecord("Sketch-1", "sketch"), "kind"),
        (
            NamedRegion(
                "Fixed",
                (
                    LogicalEntityRef("edge:left"),
                    LogicalEntityRef("edge:right"),
                ),
            ),
            "name",
        ),
        (SectionDefinition("Section-1", "Steel"), "material"),
        (RegionAssignment("Section-1", "DOMAIN"), "region_name"),
    ),
)
def test_editable_definition_contracts_are_frozen(value: object, field_name: str) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(value, field_name, "changed")


def test_definition_serialisation_shape_remains_compatible() -> None:
    assert asdict(NativePart()) == {"name": "Part-1", "body_name": "Body-1"}
    assert asdict(FeatureRecord("Sketch-1", "sketch")) == {
        "name": "Sketch-1",
        "kind": "sketch",
        "payload": {},
    }
    assert asdict(
        NamedRegion(
            "Fixed",
            (
                LogicalEntityRef("edge:left"),
                LogicalEntityRef("edge:right"),
            ),
        )
    ) == {
        "name": "Fixed",
        "references": (
            {"logical_id": "edge:left"},
            {"logical_id": "edge:right"},
        ),
    }
    assert asdict(SectionDefinition("Section-1", "Steel")) == {
        "name": "Section-1",
        "material": "Steel",
        "section_type": "solid",
        "properties": {},
    }
    assert asdict(RegionAssignment("Section-1", "DOMAIN")) == {
        "section_name": "Section-1",
        "region_name": "DOMAIN",
        "beam_orientation": None,
    }
