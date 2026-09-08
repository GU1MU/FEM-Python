from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from fem.application.native_regions import (
    CompiledDomainRegionSource,
    LogicalReferencesRegionSource,
    NativeRegionValidationError,
    RecipeRegionSelector,
    RecipeRegionSource,
    describe_native_regions,
    require_native_region_product,
    validate_logical_references,
    validate_native_authoring_context,
)
from fem.geometry import LogicalEntityRef
from fem.geometry.recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    WireGeometry,
    WireMember,
    WirePoint,
)


@dataclass(frozen=True)
class _Region:
    name: str
    references: tuple[LogicalEntityRef, ...]


@pytest.mark.parametrize(
    "recipe,expected",
    [
        (
            RectangleGeometry("Rectangle", 4.0, 2.0),
            ("DOMAIN", "BOTTOM", "RIGHT", "TOP", "LEFT"),
        ),
        (
            DiskGeometry("Disk", 1.0),
            ("DOMAIN", "OUTER"),
        ),
        (
            PlateWithHoleGeometry("Plate", 4.0, 2.0, 2.0, 1.0, 0.25),
            ("DOMAIN", "BOTTOM", "RIGHT", "TOP", "LEFT", "HOLE"),
        ),
        (
            BoxGeometry("Box", 2.0, 3.0, 4.0),
            ("DOMAIN", "BOTTOM", "RIGHT", "TOP", "LEFT", "FRONT", "BACK"),
        ),
        (
            CylinderGeometry("Cylinder", 1.0, 2.0),
            ("DOMAIN", "BOTTOM", "TOP", "OUTER"),
        ),
        (
            ExtrudedGeometry(RectangleGeometry("Rectangle", 4.0, 2.0), 3.0),
            ("DOMAIN", "BOTTOM", "TOP", "OUTER"),
        ),
    ],
)
def test_native_region_catalog_has_one_typed_builtin_source(recipe, expected) -> None:
    descriptors = describe_native_regions(recipe)

    assert tuple(descriptor.name for descriptor in descriptors) == expected
    assert isinstance(descriptors[0].source, CompiledDomainRegionSource)
    assert descriptors[0].products == frozenset({"element_set"})
    assert all(
        isinstance(descriptor.source, RecipeRegionSource)
        for descriptor in descriptors[1:]
    )
    assert all(
        descriptor.source.selector.value == descriptor.name
        for descriptor in descriptors[1:]
    )


def test_non_exact_recipe_publishes_only_domain() -> None:
    recipe = BooleanGeometry(
        "Fuse",
        "fuse",
        RectangleGeometry("A", 2.0, 2.0),
        RectangleGeometry("B", 1.0, 1.0),
    )

    descriptors = describe_native_regions(recipe)

    assert tuple(descriptor.name for descriptor in descriptors) == ("DOMAIN",)


def test_user_regions_are_canonical_and_publish_dimension_products() -> None:
    recipe = RectangleGeometry("Rectangle", 4.0, 2.0)
    regions = (
        _Region(
            "Supports",
            (
                LogicalEntityRef("edge:right"),
                LogicalEntityRef("edge:left"),
            ),
        ),
        _Region("WholeFace", (LogicalEntityRef("face:domain"),)),
    )

    descriptors = describe_native_regions(recipe, regions)
    supports = next(item for item in descriptors if item.name == "Supports")
    whole_face = next(item for item in descriptors if item.name == "WholeFace")

    assert isinstance(supports.source, LogicalReferencesRegionSource)
    assert supports.source.references == (
        LogicalEntityRef("edge:left"),
        LogicalEntityRef("edge:right"),
    )
    assert supports.products == frozenset({"node_set", "edge"})
    assert whole_face.products == frozenset({"element_set"})


@pytest.mark.parametrize(
    "region",
    [
        _Region("domain", (LogicalEntityRef("face:domain"),)),
        _Region("Empty", ()),
        _Region(
            "Mixed",
            (
                LogicalEntityRef("point:bottom-left"),
                LogicalEntityRef("edge:left"),
            ),
        ),
        _Region("Unknown", (LogicalEntityRef("edge:missing"),)),
    ],
)
def test_invalid_user_region_fails_before_cad_or_mesh(region) -> None:
    with pytest.raises(NativeRegionValidationError):
        describe_native_regions(
            RectangleGeometry("Rectangle", 4.0, 2.0),
            (region,),
        )


def test_non_exact_user_reference_is_rejected() -> None:
    recipe = BooleanGeometry(
        "Fuse",
        "fuse",
        RectangleGeometry("A", 2.0, 2.0),
        RectangleGeometry("B", 1.0, 1.0),
    )

    with pytest.raises(NativeRegionValidationError):
        describe_native_regions(
            recipe,
            (_Region("Boundary", (LogicalEntityRef("body:result"),)),),
        )


def test_reference_and_product_capability_helpers_are_pure() -> None:
    recipe = BoxGeometry("Box", 2.0, 3.0, 4.0)
    references = validate_logical_references(
        recipe,
        (
            LogicalEntityRef("face:top"),
            LogicalEntityRef("face:bottom"),
        ),
    )
    descriptors = describe_native_regions(recipe)

    assert references == (
        LogicalEntityRef("face:bottom"),
        LogicalEntityRef("face:top"),
    )
    assert require_native_region_product(descriptors, "TOP", "surface").name == "TOP"
    with pytest.raises(NativeRegionValidationError):
        require_native_region_product(descriptors, "TOP", "element_set")


def test_authoring_context_validates_target_radius_and_requirements() -> None:
    recipe = PlateWithHoleGeometry("Plate", 4.0, 2.0, 2.0, 1.0, 0.25)
    control = SimpleNamespace(
        target=LogicalEntityRef("edge:hole-loop"),
        falloff=SimpleNamespace(reference="target_radius"),
    )

    descriptors = validate_native_authoring_context(
        recipe,
        local_controls=(control,),
        region_requirements=(("HOLE", "edge"),),
    )

    assert (
        next(
            descriptor.source.selector
            for descriptor in descriptors
            if descriptor.name == "HOLE"
        )
        is RecipeRegionSelector.HOLE
    )


def test_wire_regions_expose_points_as_nodes_and_members_as_elements() -> None:
    recipe = WireGeometry(
        "Wire",
        (
            WirePoint("P1", 0.0, 0.0),
            WirePoint("P2", 1.0, 0.0),
            WirePoint("P3", 1.0, 1.0),
        ),
        (
            WireMember("M1", "P1", "P2"),
            WireMember("M2", "P2", "P3"),
        ),
    )
    regions = (
        _Region("Node", (LogicalEntityRef("point:P1"),)),
        _Region("Member", (LogicalEntityRef("edge:M1"),)),
    )

    descriptors = describe_native_regions(recipe, regions)

    assert tuple(descriptor.name for descriptor in descriptors) == (
        "DOMAIN",
        "Member",
        "Node",
    )
    assert next(item for item in descriptors if item.name == "DOMAIN").products == (
        frozenset({"element_set"})
    )
    assert next(item for item in descriptors if item.name == "Node").products == (
        frozenset({"node_set"})
    )
    assert next(item for item in descriptors if item.name == "Member").products == (
        frozenset({"element_set"})
    )


def test_wire_target_radius_falloff_reports_unsupported_measure() -> None:
    recipe = WireGeometry(
        "Wire",
        (WirePoint("P1", 0.0, 0.0), WirePoint("P2", 1.0, 0.0)),
        (WireMember("M1", "P1", "P2"),),
    )
    control = SimpleNamespace(
        target=LogicalEntityRef("edge:M1"),
        falloff=SimpleNamespace(reference="target_radius"),
    )

    with pytest.raises(ValueError, match="target_radius.*not supported"):
        validate_native_authoring_context(recipe, local_controls=(control,))
