from __future__ import annotations

import math

import pytest

from fem.geometry import (
    BASE_GEOMETRY_TYPES,
    BooleanGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    NATIVE_GEOMETRY_TYPES,
    RotatedGeometry,
    WireGeometry,
    WireMember,
    WirePoint,
    geometry_dimension,
)


def _wire() -> WireGeometry:
    return WireGeometry(
        "Portal",
        (
            WirePoint("P1", 0.0, 0.0, 0.0),
            WirePoint("P2", 0.0, 3.0, 0.0),
            WirePoint("P3", 4.0, 3.0, 1.0),
            WirePoint("P4", 4.0, 0.0, 1.0),
        ),
        (
            WireMember("M1", "P1", "P2"),
            WireMember("M2", "P2", "P3"),
            WireMember("M3", "P3", "P4"),
        ),
    )


def test_wire_values_normalize_names_and_materialize_collections() -> None:
    points = [WirePoint(" P1 ", 0, 0), WirePoint("P2", 1, 0)]
    members = [WireMember(" M1 ", " P1 ", "P2")]

    wire = WireGeometry(" Portal ", points, members)
    points.append(WirePoint("P3", 2, 0))
    members.clear()

    assert wire.name == "Portal"
    assert wire.points == (WirePoint("P1", 0.0, 0.0), WirePoint("P2", 1.0, 0.0))
    assert wire.members == (WireMember("M1", "P1", "P2"),)
    assert isinstance(wire.points, tuple)
    assert isinstance(wire.members, tuple)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: WirePoint(1, 0.0, 0.0),
        lambda: WirePoint("P", "0", 0.0),
        lambda: WirePoint("P", math.nan, 0.0),
        lambda: WirePoint("P", math.inf, 0.0),
        lambda: WirePoint("P", True, 0.0),
        lambda: WireMember(1, "P1", "P2"),
        lambda: WireMember("M", 1, "P2"),
        lambda: WireMember("M", "P1", True),
    ),
)
def test_wire_scalar_validation_rejects_noncanonical_input(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "points,members",
    (
        ((WirePoint("P1", 0, 0),), (WireMember("M1", "P1", "P1"),)),
        (
            (WirePoint("P1", 0, 0), WirePoint("p1", 1, 0)),
            (WireMember("M1", "P1", "p1"),),
        ),
        (
            (WirePoint("P1", 0, 0), WirePoint("P2", 1, 0)),
            (WireMember("M1", "P1", "P3"),),
        ),
        (
            (WirePoint("P1", 0, 0), WirePoint("P2", 0, 0)),
            (WireMember("M1", "P1", "P2"),),
        ),
        (
            (
                WirePoint("P1", 0, 0),
                WirePoint("P2", 1, 0),
                WirePoint("P3", 0, 1),
            ),
            (
                WireMember("M1", "P1", "P2"),
                WireMember("M2", "P2", "P1"),
            ),
        ),
        (
            (WirePoint("P1", 0, 0), WirePoint("P2", 1, 0)),
            (
                WireMember("M1", "P1", "P2"),
                WireMember("m1", "P2", "P1"),
            ),
        ),
        (
            (
                WirePoint("P1", 0, 0),
                WirePoint("P2", 1, 0),
                WirePoint("P3", 2, 0),
            ),
            (WireMember("M1", "P1", "P2"),),
        ),
    ),
)
def test_wire_graph_validation_rejects_invalid_graphs(points, members) -> None:
    with pytest.raises(ValueError):
        WireGeometry("Invalid", points, members)


def test_disconnected_components_and_coincident_named_points_remain_distinct() -> None:
    wire = WireGeometry(
        "Disconnected",
        (
            WirePoint("A", 0.0, 0.0),
            WirePoint("B", 1.0, 0.0),
            WirePoint("C", 0.0, 0.0),
            WirePoint("D", 0.0, 1.0),
        ),
        (
            WireMember("AB", "A", "B"),
            WireMember("CD", "C", "D"),
        ),
    )

    assert wire.points[0].name != wire.points[2].name
    assert wire.points[0].x == wire.points[2].x
    assert wire.members[0].start == "A"
    assert wire.members[1].start == "C"


def test_wire_is_a_native_base_and_rigid_transforms_preserve_dimension() -> None:
    wire = _wire()

    assert WireGeometry in BASE_GEOMETRY_TYPES
    assert WireGeometry in NATIVE_GEOMETRY_TYPES
    assert geometry_dimension(wire) == 1
    assert geometry_dimension(MovedGeometry(wire, 1.0, 2.0, 3.0)) == 1
    assert geometry_dimension(RotatedGeometry(wire, "x", 30.0)) == 1


def test_wire_cannot_be_extruded_or_used_in_a_boolean() -> None:
    wire = _wire()

    with pytest.raises(ValueError, match="二维"):
        ExtrudedGeometry(wire, 1.0)
    with pytest.raises(ValueError, match="一维"):
        BooleanGeometry("Invalid", "fuse", wire, wire)
