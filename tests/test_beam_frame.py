from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.elements import (
    BEAM_LOCAL_Y_REFERENCE_KEY,
    BeamFrame,
    BeamOrientation,
    BeamOrientationInvalidError,
    BeamOrientationParallelError,
    parse_beam_orientation,
    resolve_beam_frame,
)


def _mesh(
    end=(1.0, 0.0, 0.0),
    *,
    reference=None,
    reversed_nodes=False,
):
    props = {}
    if reference is not None:
        props[BEAM_LOCAL_Y_REFERENCE_KEY] = reference
    return Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, *end),
        ],
        elements=[
            Element3D(
                7,
                [2, 1] if reversed_nodes else [1, 2],
                "Beam2",
                props,
            )
        ],
        dofs_per_node=6,
    )


def test_orientation_parser_takes_owned_float_tuple() -> None:
    source = [0, 2, 0]

    orientation = parse_beam_orientation(source)
    source[1] = 9

    assert orientation == BeamOrientation((0.0, 2.0, 0.0))
    assert orientation.local_y_reference == (0.0, 2.0, 0.0)
    assert parse_beam_orientation(orientation) is orientation


@pytest.mark.parametrize(
    "value",
    (
        None,
        (),
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
        "0,1,0",
        {"x": 0.0, "y": 1.0, "z": 0.0},
        (True, 1.0, 0.0),
        (0.0, float("nan"), 0.0),
        (0.0, float("inf"), 0.0),
        (0.0, 0.0, 0.0),
        (10**10000, 0.0, 0.0),
    ),
)
def test_orientation_parser_rejects_invalid_values(value) -> None:
    with pytest.raises(BeamOrientationInvalidError) as exc_info:
        parse_beam_orientation(value)

    assert exc_info.value.code == "beam.orientation.invalid"


@pytest.mark.parametrize(
    ("end", "reference", "expected"),
    (
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            np.eye(3),
        ),
        (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
        ),
        (
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        ),
    ),
)
def test_explicit_frame_for_global_axes(end, reference, expected) -> None:
    mesh = _mesh(end, reference=reference)

    frame = resolve_beam_frame(mesh, mesh.elements[0])

    assert frame.source == "explicit"
    assert frame.orientation == BeamOrientation(reference)
    assert frame.rotation == pytest.approx(expected, abs=1e-12)
    assert np.cross(frame.local_x, frame.local_y) == pytest.approx(
        frame.local_z,
        abs=1e-12,
    )


def test_explicit_inclined_frame_is_orthonormal_right_handed_and_scale_invariant() -> None:
    first = _mesh((2.0, 3.0, 6.0), reference=(4.0, -2.0, 1.0))
    scaled = _mesh((2.0, 3.0, 6.0), reference=(40.0, -20.0, 10.0))

    frame = resolve_beam_frame(first, first.elements[0])
    scaled_frame = resolve_beam_frame(scaled, scaled.elements[0])

    assert frame.length == pytest.approx(7.0)
    assert frame.rotation @ frame.rotation.T == pytest.approx(
        np.eye(3),
        abs=1e-12,
    )
    assert np.linalg.det(frame.rotation) == pytest.approx(1.0)
    assert scaled_frame.rotation == pytest.approx(frame.rotation, abs=1e-12)


@pytest.mark.parametrize("transverse", (0.0, 0.5e-8, 1.0e-8))
def test_explicit_parallel_and_near_parallel_reference_is_rejected(
    transverse,
) -> None:
    axial = 1.0
    mesh = _mesh(reference=(axial, transverse, 0.0))

    with pytest.raises(BeamOrientationParallelError) as exc_info:
        resolve_beam_frame(mesh, mesh.elements[0])

    error = exc_info.value
    assert error.code == "beam.orientation.parallel"
    assert error.element_id == 7
    assert error.reference == pytest.approx((axial, transverse, 0.0))
    assert error.tangent == pytest.approx((1.0, 0.0, 0.0))


def test_explicit_reference_above_parallel_tolerance_is_accepted() -> None:
    transverse = 2.0e-8
    axial = math.sqrt(1.0 - transverse**2)
    mesh = _mesh(reference=(axial, transverse, 0.0))

    frame = resolve_beam_frame(mesh, mesh.elements[0])

    assert frame.source == "explicit"
    assert frame.local_y == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)


def test_properties_argument_is_the_complete_authoritative_source() -> None:
    mesh = _mesh(reference=(0.0, 1.0, 0.0))
    elem = mesh.elements[0]

    direct = resolve_beam_frame(mesh, elem)
    covered_automatic = resolve_beam_frame(mesh, elem, properties={})
    covered_explicit = resolve_beam_frame(
        mesh,
        elem,
        properties={BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 0.0, 1.0)},
    )

    assert direct.source == "explicit"
    assert covered_automatic.source == "automatic"
    assert covered_explicit.source == "explicit"
    assert covered_explicit.local_y == pytest.approx((0.0, 0.0, 1.0))


def test_explicit_frame_reverse_connectivity_keeps_y_and_reverses_x_z() -> None:
    forward = _mesh(reference=(0.0, 1.0, 0.0))
    reversed_mesh = _mesh(
        reference=(0.0, 1.0, 0.0),
        reversed_nodes=True,
    )

    first = resolve_beam_frame(forward, forward.elements[0])
    reversed_frame = resolve_beam_frame(
        reversed_mesh,
        reversed_mesh.elements[0],
    )

    assert reversed_frame.local_x == pytest.approx(-first.local_x)
    assert reversed_frame.local_y == pytest.approx(first.local_y)
    assert reversed_frame.local_z == pytest.approx(-first.local_z)
    assert np.linalg.det(reversed_frame.rotation) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("end", "expected"),
    (
        ((1.0, 0.0, 0.0), np.eye(3)),
        (
            (0.0, 1.0, 0.0),
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            (0.0, 0.0, 1.0),
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        ),
    ),
)
def test_automatic_frame_preserves_phase_3_global_axis_results(
    end,
    expected,
) -> None:
    mesh = _mesh(end)

    frame = resolve_beam_frame(mesh, mesh.elements[0])

    assert frame.source == "automatic"
    assert frame.orientation is None
    assert frame.rotation == pytest.approx(expected, abs=1e-12)


def test_automatic_frame_preserves_phase_3_near_global_z_fallback() -> None:
    mesh = _mesh((1e-14, 0.0, 1.0))

    frame = resolve_beam_frame(mesh, mesh.elements[0])

    assert frame.local_z == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert frame.rotation @ frame.rotation.T == pytest.approx(
        np.eye(3),
        abs=1e-12,
    )


def test_frame_rotation_and_axis_views_cannot_be_made_writable() -> None:
    mesh = _mesh(reference=(0.0, 1.0, 0.0))
    frame = resolve_beam_frame(mesh, mesh.elements[0])

    assert isinstance(frame, BeamFrame)
    assert not frame.rotation.flags.writeable
    assert not frame.local_y.flags.writeable
    with pytest.raises(ValueError):
        frame.rotation[0, 0] = 2.0
    with pytest.raises(ValueError):
        frame.rotation.setflags(write=True)
    with pytest.raises(ValueError):
        frame.local_y.setflags(write=True)

    copied = deepcopy(frame)
    assert copied is frame
    assert not copied.rotation.flags.writeable
    with pytest.raises(ValueError):
        copied.rotation.setflags(write=True)
