from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
from typing import Any, Sequence

import numpy as np
import pytest

from fem import geometry
from fem.mesh import gmsh as gmsh_meshing
from fem.mesh.gmsh import _runtime as runtime_module
from fem.mesh.gmsh import types as mesh_types

from tests.test_gmsh_geometry import (
    _AUTO_OPTION_ORIGINALS,
    _FakeGmsh,
    _build_fake_topology,
    _fake_entities,
    _fake_threshold,
    _first_requested_options,
    _generate_auto_mesh,
    _generate_mesh,
    _install_backend,
    _mesher,
    _set_fake_element_blocks,
)


def test_transfinite_curve_and_recombine_forward_typed_targets_while_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("controls", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].add((1, 7))
        curve = cad.entity(1, 7)

        assert _mesher(cad).transfinite_curve(curve, num_nodes=np.int64(5)) is None
        assert _mesher(cad).recombine(surface) is None
        assert _mesher(cad).transfinite_curve(curve, num_nodes=3) is None
        assert _mesher(cad).recombine(surface) is None

    assert backend.model.mesh.calls == [
        ("setTransfiniteCurve", 7, 5, "controls"),
        ("setRecombine", 2, 1, "controls"),
        ("setTransfiniteCurve", 7, 3, "controls"),
        ("setRecombine", 2, 1, "controls"),
    ]
    assert backend.option.calls == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, surface, curve: _mesher(cad).transfinite_curve(
            surface, num_nodes=3
        ),
        lambda cad, surface, curve: _mesher(cad).transfinite_curve(1, num_nodes=3),
        lambda cad, surface, curve: _mesher(cad).transfinite_curve(
            curve, num_nodes=True
        ),
        lambda cad, surface, curve: _mesher(cad).transfinite_curve(curve, num_nodes=1),
        lambda cad, surface, curve: _mesher(cad).transfinite_curve(
            curve, num_nodes=2.5
        ),
        lambda cad, surface, curve: _mesher(cad).recombine(curve),
        lambda cad, surface, curve: _mesher(cad).recombine((2, 1)),
    ],
)
def test_invalid_curve_and_recombine_controls_fail_before_backend_mutation(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-controls", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].add((1, 1))
        curve = cad.entity(1, 1)
        synchronize_calls = backend.model.occ.synchronize_calls

        with pytest.raises((TypeError, ValueError)):
            operation(cad, surface, curve)

        assert backend.model.mesh.calls == []
        assert backend.model.occ.synchronize_calls == synchronize_calls


def test_mesh_controls_reject_cross_model_and_stale_targets_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-controls", dimension=2) as outer:
        outer_surface = outer.rectangle(0, 0, 1, 1)
        with geometry.model("inner-controls", dimension=2) as inner:
            inner_surface = inner.rectangle(0, 0, 1, 1)
            synchronize_calls = backend.model.occ.synchronize_calls
            with pytest.raises(geometry.EntityOwnershipError):
                _mesher(inner).recombine(outer_surface)
            assert backend.model.occ.synchronize_calls == synchronize_calls

            with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
                _ = inner.raw_model
            assert _mesher(inner).recombine(inner_surface) is None

    assert sum(call[0] == "setRecombine" for call in backend.model.mesh.calls) == 1


def test_native_mesh_control_failure_preserves_state_and_mesh_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("retry-controls", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model.mesh.fail_next.add("setRecombine")
        with pytest.raises(RuntimeError, match="fake setRecombine failure"):
            _mesher(cad).recombine(surface)

        native_mesh = _generate_mesh(cad, recombine=False)
        assert isinstance(native_mesh, gmsh_meshing.GmshMeshRef)


def test_transfinite_surface_forwards_automatic_and_explicit_corners_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("surface-controls", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].update(
            (0, tag) for tag in range(1, 5)
        )
        points = tuple(cad.entity(0, tag) for tag in range(1, 5))
        backend.model.boundary_result = [
            (0, point.tag) for point in points
        ]

        assert _mesher(cad).transfinite_surface(surface) is None
        assert (
            _mesher(cad).transfinite_surface(
                surface,
                corners=(points[3], points[1], points[0], points[2]),
            )
            is None
        )
        assert _mesher(cad).transfinite_surface(surface, corners=points[:3]) is None
        assert cad.entity(0, points[0].tag) == points[0]

    assert backend.model.mesh.calls == [
        ("setTransfiniteSurface", 1, "Left", (), "surface-controls"),
        (
            "setTransfiniteSurface",
            1,
            "Left",
            (4, 2, 1, 3),
            "surface-controls",
        ),
        (
            "setTransfiniteSurface",
            1,
            "Left",
            (1, 2, 3),
            "surface-controls",
        ),
    ]


@pytest.mark.parametrize(
    "corners",
    [
        None,
        lambda points, curve: points[:2],
        lambda points, curve: (*points, points[0]),
        lambda points, curve: (points[0], points[0], points[1]),
        lambda points, curve: (points[0], points[1], curve),
    ],
)
def test_invalid_surface_corner_shape_fails_before_native_control(
    monkeypatch: pytest.MonkeyPatch,
    corners: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-surface-corners", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].update(
            {(0, 1), (0, 2), (0, 3), (0, 4), (1, 1)}
        )
        points = tuple(cad.entity(0, tag) for tag in range(1, 5))
        curve = cad.entity(1, 1)
        supplied = corners(points, curve) if callable(corners) else corners
        synchronize_calls = backend.model.occ.synchronize_calls

        with pytest.raises((TypeError, ValueError)):
            _mesher(cad).transfinite_surface(surface, corners=supplied)

        assert backend.model.mesh.calls == []
        assert backend.model.occ.synchronize_calls == synchronize_calls


def test_transfinite_surface_rejects_nonboundary_corner_before_native_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("surface-membership", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].update(
            (0, tag) for tag in range(1, 5)
        )
        points = tuple(cad.entity(0, tag) for tag in range(1, 5))
        backend.model.boundary_result = [(0, 1), (0, 2), (0, 3)]

        with pytest.raises(ValueError, match="boundary"):
            _mesher(cad).transfinite_surface(
                surface,
                corners=(points[0], points[1], points[3]),
            )

    assert backend.model.mesh.calls == []


def test_transfinite_surface_rejects_cross_model_and_missing_native_corners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-surface", dimension=2) as outer:
        backend.model._current_data()["entities"].add((0, 1))
        foreign = outer.entity(0, 1)
        with geometry.model("inner-surface", dimension=2) as inner:
            surface = inner.rectangle(0, 0, 1, 1)
            backend.model._current_data()["entities"].update(
                (0, tag) for tag in range(1, 4)
            )
            points = tuple(inner.entity(0, tag) for tag in range(1, 4))
            synchronize_calls = backend.model.occ.synchronize_calls

            with pytest.raises(geometry.EntityOwnershipError):
                _mesher(inner).transfinite_surface(
                    surface,
                    corners=(points[0], points[1], foreign),
                )
            assert backend.model.occ.synchronize_calls == synchronize_calls

            backend.model._current_data()["entities"].discard(
                (0, points[2].tag)
            )
            with pytest.raises(geometry.StaleEntityError):
                _mesher(inner).transfinite_surface(surface, corners=points)
            assert backend.model.occ.synchronize_calls == synchronize_calls + 1

    assert backend.model.mesh.calls == []


def test_transfinite_volume_forwards_automatic_six_and_eight_corners_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("volume-controls", dimension=3) as cad:
        volume = cad.box(0, 0, 0, 1, 1, 1)
        backend.model._current_data()["entities"].update(
            (0, tag) for tag in range(1, 9)
        )
        points = tuple(cad.entity(0, tag) for tag in range(1, 9))
        backend.model.boundary_result = [
            (0, point.tag) for point in points
        ]

        assert _mesher(cad).transfinite_volume(volume) is None
        assert (
            _mesher(cad).transfinite_volume(
                volume,
                corners=(
                    points[5],
                    points[2],
                    points[0],
                    points[4],
                    points[1],
                    points[3],
                ),
            )
            is None
        )
        assert _mesher(cad).transfinite_volume(volume, corners=tuple(reversed(points))) is None

    assert backend.model.mesh.calls == [
        ("setTransfiniteVolume", 1, (), "volume-controls"),
        ("setTransfiniteVolume", 1, (6, 3, 1, 5, 2, 4), "volume-controls"),
        ("setTransfiniteVolume", 1, (8, 7, 6, 5, 4, 3, 2, 1), "volume-controls"),
    ]


@pytest.mark.parametrize(
    "corners",
    [
        None,
        lambda points, surface: points[:5],
        lambda points, surface: points[:7],
        lambda points, surface: (*points, points[0]),
        lambda points, surface: (*points[:5], points[0]),
        lambda points, surface: (*points[:5], surface),
    ],
)
def test_invalid_volume_corner_shape_fails_before_native_control(
    monkeypatch: pytest.MonkeyPatch,
    corners: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-volume-corners", dimension=3) as cad:
        volume = cad.box(0, 0, 0, 1, 1, 1)
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].update(
            (0, tag) for tag in range(1, 9)
        )
        points = tuple(cad.entity(0, tag) for tag in range(1, 9))
        supplied = corners(points, surface) if callable(corners) else corners
        synchronize_calls = backend.model.occ.synchronize_calls

        with pytest.raises((TypeError, ValueError)):
            _mesher(cad).transfinite_volume(volume, corners=supplied)

        assert backend.model.mesh.calls == []
        assert backend.model.occ.synchronize_calls == synchronize_calls


def test_transfinite_volume_requires_3d_facade_and_recursive_boundary_corners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("two-dimensional-volume", dimension=2) as cad:
        backend.model._current_data()["entities"].add((3, 1))
        synchronize_calls = backend.model.occ.synchronize_calls
        with pytest.raises(ValueError, match="model dimension"):
            cad.entity(3, 1)
        assert backend.model.occ.synchronize_calls == synchronize_calls

    with geometry.model("volume-membership", dimension=3) as cad:
        volume = cad.box(0, 0, 0, 1, 1, 1)
        backend.model._current_data()["entities"].update(
            (0, tag) for tag in range(1, 9)
        )
        points = tuple(cad.entity(0, tag) for tag in range(1, 9))
        backend.model.boundary_result = [(0, tag) for tag in range(1, 8)]

        with pytest.raises(ValueError, match="boundary"):
            _mesher(cad).transfinite_volume(volume, corners=points)

    assert backend.model.mesh.calls == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad: _mesher(cad).transfinite_curve(None, num_nodes=3),
        lambda cad: _mesher(cad).transfinite_surface(None),
        lambda cad: _mesher(cad).transfinite_volume(None),
        lambda cad: _mesher(cad).recombine(None),
        lambda cad: _mesher(cad).mesh_size([], size=0.1),
        lambda cad: _mesher(cad).distance_field(),
        lambda cad: _mesher(cad).threshold_field(
            None,
            size_min=0.1,
            size_max=0.2,
            dist_min=0.0,
            dist_max=1.0,
        ),
        lambda cad: _mesher(cad).min_field([]),
        lambda cad: _mesher(cad).background_field(None),
    ],
)
def test_mesh_controls_reject_new_and_closed_states_contextually(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    cad = geometry.GeometryModel("control-states", dimension=3)

    with pytest.raises(geometry.GeometryStateError, match="NEW"):
        operation(cad)
    with cad:
        pass
    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        operation(cad)


def test_mesh_controls_reject_meshed_and_mesh_failed_states_contextually(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    operations = (
        lambda cad: _mesher(cad).transfinite_curve(None, num_nodes=3),
        lambda cad: _mesher(cad).transfinite_surface(None),
        lambda cad: _mesher(cad).transfinite_volume(None),
        lambda cad: _mesher(cad).recombine(None),
        lambda cad: _mesher(cad).mesh_size([], size=0.1),
        lambda cad: _mesher(cad).distance_field(),
        lambda cad: _mesher(cad).threshold_field(
            None,
            size_min=0.1,
            size_max=0.2,
            dist_min=0.0,
            dist_max=1.0,
        ),
        lambda cad: _mesher(cad).min_field([]),
        lambda cad: _mesher(cad).background_field(None),
    )

    with geometry.model("meshed-controls", dimension=3) as cad:
        cad.box(0, 0, 0, 1, 1, 1)
        _generate_mesh(cad, )
        for operation in operations:
            with pytest.raises(geometry.GeometryStateError, match="MESHED"):
                operation(cad)

    with geometry.model("failed-controls", dimension=3) as cad:
        cad.box(0, 0, 0, 1, 1, 1)
        backend.model.mesh.fail_generate = True
        with pytest.raises(RuntimeError, match="fake mesh failure"):
            _generate_mesh(cad, )
        for operation in operations:
            with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
                operation(cad)


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, curve, surface, volume: _mesher(cad).transfinite_curve(
            surface, num_nodes=3
        ),
        lambda cad, curve, surface, volume: _mesher(cad).transfinite_surface(curve),
        lambda cad, curve, surface, volume: _mesher(cad).transfinite_surface(2),
        lambda cad, curve, surface, volume: _mesher(cad).transfinite_volume(surface),
        lambda cad, curve, surface, volume: _mesher(cad).transfinite_volume((3, 1)),
        lambda cad, curve, surface, volume: _mesher(cad).recombine(volume),
    ],
)
def test_every_mesh_control_rejects_invalid_target_before_backend_mutation(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-control-targets", dimension=3) as cad:
        volume = cad.box(0, 0, 0, 1, 1, 1)
        surface = cad.rectangle(2, 0, 1, 1)
        backend.model._current_data()["entities"].add((1, 1))
        curve = cad.entity(1, 1)
        synchronize_calls = backend.model.occ.synchronize_calls

        with pytest.raises((TypeError, ValueError)):
            operation(cad, curve, surface, volume)

        assert backend.model.mesh.calls == []
        assert backend.model.occ.synchronize_calls == synchronize_calls


@pytest.mark.parametrize(
    ("dimension", "operation"),
    [
        (1, lambda cad, target: _mesher(cad).transfinite_curve(target, num_nodes=3)),
        (2, lambda cad, target: _mesher(cad).transfinite_surface(target)),
        (3, lambda cad, target: _mesher(cad).transfinite_volume(target)),
        (2, lambda cad, target: _mesher(cad).recombine(target)),
    ],
)
def test_every_mesh_control_rejects_foreign_and_missing_native_targets(
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-control-target", dimension=3) as outer:
        backend.model._current_data()["entities"].add((dimension, 9))
        foreign = outer.entity(dimension, 9)
        with geometry.model("inner-control-target", dimension=3) as inner:
            backend.model._current_data()["entities"].add((dimension, 9))
            local = inner.entity(dimension, 9)
            synchronize_calls = backend.model.occ.synchronize_calls

            with pytest.raises(geometry.EntityOwnershipError):
                operation(inner, foreign)
            assert backend.model.occ.synchronize_calls == synchronize_calls

            backend.model._current_data()["entities"].discard((dimension, 9))
            with pytest.raises(geometry.StaleEntityError):
                operation(inner, local)
            assert backend.model.occ.synchronize_calls == synchronize_calls + 1

    assert backend.model.mesh.calls == []


def test_transfinite_volume_rejects_foreign_and_missing_native_corners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-volume-corner", dimension=3) as outer:
        backend.model._current_data()["entities"].add((0, 1))
        foreign = outer.entity(0, 1)
        with geometry.model("inner-volume-corner", dimension=3) as inner:
            volume = inner.box(0, 0, 0, 1, 1, 1)
            backend.model._current_data()["entities"].update(
                (0, tag) for tag in range(1, 7)
            )
            points = tuple(inner.entity(0, tag) for tag in range(1, 7))
            synchronize_calls = backend.model.occ.synchronize_calls

            with pytest.raises(geometry.EntityOwnershipError):
                _mesher(inner).transfinite_volume(
                    volume,
                    corners=(*points[:5], foreign),
                )
            assert backend.model.occ.synchronize_calls == synchronize_calls

            backend.model._current_data()["entities"].discard(
                (0, points[5].tag)
            )
            with pytest.raises(geometry.StaleEntityError):
                _mesher(inner).transfinite_volume(volume, corners=points)
            assert backend.model.occ.synchronize_calls == synchronize_calls + 1

    assert backend.model.mesh.calls == []


@pytest.mark.parametrize(
    ("native_operation", "target_name", "operation"),
    [
        (
            "setTransfiniteCurve",
            "curve",
            lambda cad, target: _mesher(cad).transfinite_curve(target, num_nodes=3),
        ),
        (
            "setTransfiniteSurface",
            "surface",
            lambda cad, target: _mesher(cad).transfinite_surface(target),
        ),
        (
            "setTransfiniteVolume",
            "volume",
            lambda cad, target: _mesher(cad).transfinite_volume(target),
        ),
        ("setRecombine", "surface", lambda cad, target: _mesher(cad).recombine(target)),
    ],
)
def test_native_control_failures_preserve_exception_state_and_generation_attempt(
    monkeypatch: pytest.MonkeyPatch,
    native_operation: str,
    target_name: str,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"failed-{native_operation}", dimension=3) as cad:
        targets = {
            "volume": cad.box(0, 0, 0, 1, 1, 1),
            "surface": cad.rectangle(2, 0, 1, 1),
        }
        backend.model._current_data()["entities"].add((1, 1))
        targets["curve"] = cad.entity(1, 1)
        target = targets[target_name]
        backend.model.mesh.fail_next.add(native_operation)

        with pytest.raises(RuntimeError, match=f"fake {native_operation} failure"):
            operation(cad, target)

        native_mesh = _generate_mesh(cad, recombine=False)
        assert isinstance(native_mesh, gmsh_meshing.GmshMeshRef)


def test_mesh_controls_reactivate_owned_model_and_stay_nested_model_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(
        initialized=True,
        names=("external",),
        current="external",
    )
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-mesh-controls", dimension=3) as outer:
        outer_volume = outer.box(0, 0, 0, 1, 1, 1)
        outer_surface = outer.rectangle(2, 0, 1, 1)
        backend.model._current_data()["entities"].add((1, 1))
        outer_curve = outer.entity(1, 1)
        outer_operations = (
            lambda: _mesher(outer).transfinite_curve(outer_curve, num_nodes=3),
            lambda: _mesher(outer).transfinite_surface(outer_surface),
            lambda: _mesher(outer).transfinite_volume(outer_volume),
            lambda: _mesher(outer).recombine(outer_surface),
        )
        for operation in outer_operations:
            backend.model.setCurrent("external")
            operation()

        with geometry.model("inner-mesh-controls", dimension=3) as inner:
            inner_volume = inner.box(0, 0, 0, 1, 1, 1)
            _mesher(inner).transfinite_volume(inner_volume)

        _mesher(outer).transfinite_volume(outer_volume)

    control_calls = [
        call
        for call in backend.model.mesh.calls
        if call[0]
        in {
            "setTransfiniteCurve",
            "setTransfiniteSurface",
            "setTransfiniteVolume",
            "setRecombine",
        }
    ]
    assert [call[-1] for call in control_calls] == [
        "outer-mesh-controls",
        "outer-mesh-controls",
        "outer-mesh-controls",
        "outer-mesh-controls",
        "inner-mesh-controls",
        "outer-mesh-controls",
    ]
    assert backend.model.current == "external"


def test_mesh_size_forwards_ordered_batches_while_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("point-sizes", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point_3, point_1 = _fake_entities(cad, backend, 0, 3, 1)

        synchronize_calls = backend.model.occ.synchronize_calls
        assert (
            _mesher(cad).mesh_size(
                (point for point in (point_3, point_1)),
                size=np.float64(0.2),
            )
            is None
        )
        assert backend.model.occ.synchronize_calls == synchronize_calls + 1

        synchronize_calls = backend.model.occ.synchronize_calls
        assert _mesher(cad).mesh_size([point_1], size=0.025) is None
        assert backend.model.occ.synchronize_calls == synchronize_calls + 1

    assert backend.model.mesh.calls == [
        ("setSize", ((0, 3), (0, 1)), 0.2, "point-sizes"),
        ("setSize", ((0, 1),), 0.025, "point-sizes"),
    ]
    assert backend.model.mesh.field.calls == []
    assert backend.option.calls == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, point, surface: _mesher(cad).mesh_size([], size=0.1),
        lambda cad, point, surface: _mesher(cad).mesh_size([point, point], size=0.1),
        lambda cad, point, surface: _mesher(cad).mesh_size([surface], size=0.1),
        lambda cad, point, surface: _mesher(cad).mesh_size([object()], size=0.1),
        lambda cad, point, surface: _mesher(cad).mesh_size([point], size=True),
        lambda cad, point, surface: _mesher(cad).mesh_size([point], size=0.0),
        lambda cad, point, surface: _mesher(cad).mesh_size([point], size=-0.1),
        lambda cad, point, surface: _mesher(cad).mesh_size([point], size=float("inf")),
        lambda cad, point, surface: _mesher(cad).mesh_size([point], size=float("nan")),
    ],
)
def test_invalid_mesh_size_inputs_fail_before_synchronization_or_mutation(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-point-sizes", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        synchronize_calls = backend.model.occ.synchronize_calls

        with pytest.raises((TypeError, ValueError)):
            operation(cad, point, surface)

        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert backend.model.mesh.calls == []


def test_mesh_size_materializes_generators_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("point-size-generator", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]

        def broken_points():
            yield point
            raise RuntimeError("point generator failed")

        synchronize_calls = backend.model.occ.synchronize_calls
        with pytest.raises(RuntimeError, match="point generator failed"):
            _mesher(cad).mesh_size(broken_points(), size=0.1)
        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert backend.model.mesh.calls == []


def test_mesh_size_rejects_foreign_and_missing_native_points_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-point-size", dimension=2) as outer:
        foreign = _fake_entities(outer, backend, 0, 1)[0]
        with geometry.model("inner-point-size", dimension=2) as inner:
            local = _fake_entities(inner, backend, 0, 1)[0]
            synchronize_calls = backend.model.occ.synchronize_calls

            with pytest.raises(geometry.EntityOwnershipError):
                _mesher(inner).mesh_size([foreign], size=0.1)
            assert backend.model.occ.synchronize_calls == synchronize_calls

            backend.model._current_data()["entities"].discard((0, local.tag))
            with pytest.raises(geometry.StaleEntityError):
                _mesher(inner).mesh_size([local], size=0.1)
            assert backend.model.occ.synchronize_calls == synchronize_calls + 1

    assert backend.model.mesh.calls == []


@pytest.mark.parametrize("source_case", ["points", "curves", "surfaces", "mixed"])
def test_distance_field_forwards_dimension_specific_lists_in_source_order(
    monkeypatch: pytest.MonkeyPatch,
    source_case: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"distance-{source_case}", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        point_2, point_1 = _fake_entities(cad, backend, 0, 2, 1)
        curve_4, curve_3 = _fake_entities(cad, backend, 1, 4, 3)
        kwargs: dict[str, Any]
        expected_options: list[tuple[Any, ...]]
        if source_case == "points":
            kwargs = {"points": (point for point in (point_2, point_1))}
            expected_options = [("setNumbers", 1, "PointsList", (2, 1))]
        elif source_case == "curves":
            kwargs = {"curves": [curve_4, curve_3]}
            expected_options = [("setNumbers", 1, "CurvesList", (4, 3))]
        elif source_case == "surfaces":
            kwargs = {"surfaces": [surface]}
            expected_options = [("setNumbers", 1, "SurfacesList", (1,))]
        else:
            kwargs = {
                "points": [point_2, point_1],
                "curves": (curve for curve in (curve_4, curve_3)),
                "surfaces": [surface],
                "sampling": np.int64(40),
            }
            expected_options = [
                ("setNumbers", 1, "PointsList", (2, 1)),
                ("setNumbers", 1, "CurvesList", (4, 3)),
                ("setNumbers", 1, "SurfacesList", (1,)),
            ]

        synchronize_calls = backend.model.occ.synchronize_calls
        distance = _mesher(cad).distance_field(**kwargs)
        assert backend.model.occ.synchronize_calls == synchronize_calls + 1
        assert (distance.tag, distance.field_type) == (1, "Distance")

    model_name = f"distance-{source_case}"
    sampling = 40.0 if source_case == "mixed" else 20.0
    assert backend.model.mesh.field.calls == [
        ("add", "Distance", -1, model_name),
        *(option + (model_name,) for option in expected_options),
        ("setNumber", 1, "Sampling", sampling, model_name),
        ("list", model_name),
    ]
    assert backend.option.calls == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, point, curve, surface: _mesher(cad).distance_field(),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(
            points=[point, point]
        ),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(points=[curve]),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(curves=[point]),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(surfaces=[curve]),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(points=[object()]),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(
            points=[point], sampling=True
        ),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(
            points=[point], sampling=1
        ),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(
            points=[point], sampling=2.5
        ),
    ],
)
def test_invalid_distance_field_inputs_fail_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-distance", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        curve = _fake_entities(cad, backend, 1, 1)[0]
        synchronize_calls = backend.model.occ.synchronize_calls

        with pytest.raises((TypeError, ValueError)):
            operation(cad, point, curve, surface)

        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert backend.model.mesh.field.calls == []


def test_distance_field_materializes_every_source_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("distance-generator", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]

        def broken_curves():
            raise RuntimeError("curve generator failed")
            yield point

        synchronize_calls = backend.model.occ.synchronize_calls
        with pytest.raises(RuntimeError, match="curve generator failed"):
            _mesher(cad).distance_field(points=[point], curves=broken_curves())
        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert backend.model.mesh.field.calls == []


def test_threshold_min_and_background_build_an_ordered_inert_field_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("field-graph", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        curve = _fake_entities(cad, backend, 1, 7)[0]
        distance = _mesher(cad).distance_field(curves=[curve], sampling=30)
        first = _fake_threshold(cad, distance)
        assert backend.model._current_data()["background_mesh_field"] is None

        second = _fake_threshold(
            cad,
            distance,
            size_min=0.02,
            size_max=0.3,
            dist_min=0.05,
            dist_max=0.6,
        )
        minimum = _mesher(cad).min_field((field for field in (second, first)))
        nested = _mesher(cad).min_field([minimum, second])
        assert _mesher(cad).background_field(nested) is None

        fields = backend.model._current_data()["mesh_fields"]
        assert fields[distance.tag]["type"] == "Distance"
        assert fields[first.tag]["numbers"] == {
            "InField": 1.0,
            "SizeMin": 0.05,
            "SizeMax": 0.4,
            "DistMin": 0.1,
            "DistMax": 0.8,
        }
        assert fields[second.tag]["numbers"] == {
            "InField": 1.0,
            "SizeMin": 0.02,
            "SizeMax": 0.3,
            "DistMin": 0.05,
            "DistMax": 0.6,
        }
        assert fields[minimum.tag]["number_lists"] == {
            "FieldsList": (second.tag, first.tag)
        }
        assert fields[nested.tag]["number_lists"] == {
            "FieldsList": (minimum.tag, second.tag)
        }
        assert backend.model._current_data()["background_mesh_field"] == nested.tag

    assert [
        call[:4]
        for call in backend.model.mesh.field.calls
        if call[0] == "setNumbers" and call[2] == "FieldsList"
    ] == [
        ("setNumbers", minimum.tag, "FieldsList", (second.tag, first.tag)),
        ("setNumbers", nested.tag, "FieldsList", (minimum.tag, second.tag)),
    ]
    assert [
        call[:4]
        for call in backend.model.mesh.field.calls
        if call[0] == "setNumber" and call[2] != "Sampling"
    ] == [
        ("setNumber", first.tag, "InField", float(distance.tag)),
        ("setNumber", first.tag, "SizeMin", 0.05),
        ("setNumber", first.tag, "SizeMax", 0.4),
        ("setNumber", first.tag, "DistMin", 0.1),
        ("setNumber", first.tag, "DistMax", 0.8),
        ("setNumber", second.tag, "InField", float(distance.tag)),
        ("setNumber", second.tag, "SizeMin", 0.02),
        ("setNumber", second.tag, "SizeMax", 0.3),
        ("setNumber", second.tag, "DistMin", 0.05),
        ("setNumber", second.tag, "DistMax", 0.6),
    ]
    assert backend.option.calls == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, distance, threshold: _mesher(cad).threshold_field(
            object(),
            size_min=0.1,
            size_max=0.2,
            dist_min=0.0,
            dist_max=1.0,
        ),
        lambda cad, distance, threshold: _fake_threshold(cad, threshold),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, size_min=True
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, size_min=0.0
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, size_min=float("nan")
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, size_max=float("inf")
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, size_min=0.4, size_max=0.4
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, size_min=0.5, size_max=0.4
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, dist_min=-0.1
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, dist_min=float("nan")
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, dist_min=0.2, dist_max=0.2
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, dist_min=0.2, dist_max=0.1
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, dist_max=float("inf")
        ),
    ],
)
def test_invalid_threshold_inputs_fail_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-threshold", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = _mesher(cad).distance_field(points=[point])
        threshold = _fake_threshold(cad, distance)
        calls = list(backend.model.mesh.field.calls)

        with pytest.raises((TypeError, ValueError)):
            operation(cad, distance, threshold)

        assert backend.model.mesh.field.calls == calls


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, distance, first, second: _mesher(cad).min_field([]),
        lambda cad, distance, first, second: _mesher(cad).min_field([first]),
        lambda cad, distance, first, second: _mesher(cad).min_field([first, first]),
        lambda cad, distance, first, second: _mesher(cad).min_field([distance, first]),
        lambda cad, distance, first, second: _mesher(cad).min_field([object(), first]),
    ],
)
def test_invalid_min_inputs_fail_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-min", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = _mesher(cad).distance_field(points=[point])
        first = _fake_threshold(cad, distance)
        second = _fake_threshold(
            cad,
            distance,
            size_min=0.02,
            size_max=0.3,
        )
        calls = list(backend.model.mesh.field.calls)

        with pytest.raises((TypeError, ValueError)):
            operation(cad, distance, first, second)

        assert backend.model.mesh.field.calls == calls


def test_distance_sources_reject_foreign_and_missing_native_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-distance-source", dimension=2) as outer:
        foreign = _fake_entities(outer, backend, 0, 1)[0]
        with geometry.model("inner-distance-source", dimension=2) as inner:
            local = _fake_entities(inner, backend, 0, 1)[0]
            synchronize_calls = backend.model.occ.synchronize_calls

            with pytest.raises(geometry.EntityOwnershipError):
                _mesher(inner).distance_field(points=[foreign])
            assert backend.model.occ.synchronize_calls == synchronize_calls

            backend.model._current_data()["entities"].discard((0, local.tag))
            with pytest.raises(geometry.StaleEntityError):
                _mesher(inner).distance_field(points=[local])
            assert backend.model.occ.synchronize_calls == synchronize_calls + 1

    assert backend.model.mesh.field.calls == []


def test_mesh_fields_are_owned_live_model_local_and_use_fresh_tokens_on_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(
        initialized=True,
        names=("external",),
        current="external",
    )
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-fields", dimension=2) as outer:
        outer_point = _fake_entities(outer, backend, 0, 1)[0]
        outer_distance = _mesher(outer).distance_field(points=[outer_point])
        outer_threshold = _fake_threshold(outer, outer_distance)
        with geometry.model("inner-fields", dimension=2) as inner:
            inner_point = _fake_entities(inner, backend, 0, 1)[0]
            inner_distance = _mesher(inner).distance_field(points=[inner_point])
            assert outer_distance.tag == inner_distance.tag == 1

            calls = list(backend.model.mesh.field.calls)
            with pytest.raises(gmsh_meshing.MeshFieldOwnershipError):
                _fake_threshold(inner, outer_distance)
            assert backend.model.mesh.field.calls == calls

            backend.model.mesh.field.remove(inner_distance.tag)
            with pytest.raises(gmsh_meshing.StaleMeshFieldError, match="no longer exists"):
                _fake_threshold(inner, inner_distance)

            replacement = _mesher(inner).distance_field(points=[inner_point])
            assert replacement.tag == inner_distance.tag
            assert replacement != inner_distance
            with pytest.raises(gmsh_meshing.StaleMeshFieldError, match="stale"):
                _fake_threshold(inner, inner_distance)

            threshold = _fake_threshold(inner, replacement)
            assert threshold.field_type == "Threshold"
            calls = list(backend.model.mesh.field.calls)
            with pytest.raises(gmsh_meshing.MeshFieldOwnershipError):
                _mesher(inner).background_field(outer_threshold)
            with pytest.raises(gmsh_meshing.MeshFieldOwnershipError):
                _mesher(inner).min_field([threshold, outer_threshold])
            assert backend.model.mesh.field.calls == calls

            with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
                _ = inner.raw_model
            assert _mesher(inner).background_field(threshold) is None

        assert _mesher(outer).background_field(outer_threshold) is None

    assert any(
        call[0] == "add" and call[-1] == "outer-fields"
        for call in backend.model.mesh.field.calls
    )
    assert any(
        call[0] == "add" and call[-1] == "inner-fields"
        for call in backend.model.mesh.field.calls
    )
    assert backend.model.current == "external"


@pytest.mark.parametrize(
    ("native_operation", "option"),
    [
        ("setNumbers", "PointsList"),
        ("setNumbers", "CurvesList"),
        ("setNumbers", "SurfacesList"),
        ("setNumber", "Sampling"),
        ("list", None),
    ],
)
def test_distance_field_rolls_back_after_each_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
    native_operation: str,
    option: str | None,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("distance-rollback", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        curve = _fake_entities(cad, backend, 1, 1)[0]
        backend.model.mesh.field.fail_next.add((native_operation, option))

        with pytest.raises(RuntimeError, match=f"fake field {native_operation}"):
            _mesher(cad).distance_field(
                points=[point],
                curves=[curve],
                surfaces=[surface],
            )

        assert backend.model._current_data()["mesh_fields"] == {}
        assert backend.model.mesh.field.calls[-1] == (
            "remove",
            1,
            "distance-rollback",
        )


@pytest.mark.parametrize(
    "option",
    ["InField", "SizeMin", "SizeMax", "DistMin", "DistMax", None],
)
def test_threshold_field_rolls_back_after_each_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
    option: str | None,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("threshold-rollback", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = _mesher(cad).distance_field(points=[point])
        if option is None:
            backend.model.mesh.field.fail_after[("list", None)] = 1
            expected = "list"
        else:
            backend.model.mesh.field.fail_next.add(("setNumber", option))
            expected = "setNumber"

        with pytest.raises(RuntimeError, match=f"fake field {expected}"):
            _fake_threshold(cad, distance)

        assert set(backend.model._current_data()["mesh_fields"]) == {distance.tag}
        assert backend.model.mesh.field.calls[-1] == (
            "remove",
            2,
            "threshold-rollback",
        )


@pytest.mark.parametrize("failure", ["FieldsList", "list"])
def test_min_field_rolls_back_after_each_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("min-rollback", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = _mesher(cad).distance_field(points=[point])
        first = _fake_threshold(cad, distance)
        second = _fake_threshold(
            cad,
            distance,
            size_min=0.02,
            size_max=0.3,
        )
        if failure == "list":
            backend.model.mesh.field.fail_after[("list", None)] = 1
        else:
            backend.model.mesh.field.fail_next.add(("setNumbers", "FieldsList"))

        with pytest.raises(RuntimeError, match=f"fake field .*{failure}"):
            _mesher(cad).min_field([second, first])

        assert set(backend.model._current_data()["mesh_fields"]) == {
            distance.tag,
            first.tag,
            second.tag,
        }
        assert backend.model.mesh.field.calls[-1] == (
            "remove",
            4,
            "min-rollback",
        )


def test_field_constructor_rejects_invalid_or_inactive_allocated_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-field-tag", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        backend.model.mesh.field.add_results.append(0)
        with pytest.raises(ValueError, match="positive"):
            _mesher(cad).distance_field(points=[point])
        assert backend.model._current_data()["mesh_fields"] == {}

        backend.model.mesh.field.hidden_tags.add(1)
        with pytest.raises(geometry.GeometryError, match="not active"):
            _mesher(cad).distance_field(points=[point])
        assert backend.model._current_data()["mesh_fields"] == {}

        live = _mesher(cad).distance_field(points=[point])
        assert (live.tag, live.field_type) == (1, "Distance")


def test_field_rollback_failure_preserves_primary_error_note_and_mesh_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("rollback-note", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        backend.model.mesh.field.fail_next.update(
            {("setNumber", "Sampling"), ("remove", None)}
        )

        with pytest.raises(RuntimeError, match="setNumber") as captured:
            _mesher(cad).distance_field(points=[point])
        assert any(
            "rollback" in note and "remove" in note
            for note in getattr(captured.value, "__notes__", ())
        )
        assert set(backend.model._current_data()["mesh_fields"]) == {1}

        replacement = _mesher(cad).distance_field(points=[point])
        assert replacement.tag == 2
        assert isinstance(_generate_mesh(cad, size=0.2), gmsh_meshing.GmshMeshRef)


def test_background_selection_failure_is_retryable_and_keeps_fields_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("background-retry", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = _mesher(cad).distance_field(points=[point])
        threshold = _fake_threshold(cad, distance)
        backend.model.mesh.field.fail_next.add(("setAsBackgroundMesh", None))

        with pytest.raises(RuntimeError, match="setAsBackgroundMesh"):
            _mesher(cad).background_field(threshold)
        assert backend.model._current_data()["background_mesh_field"] is None
        assert _mesher(cad).background_field(threshold) is None
        assert backend.model._current_data()["background_mesh_field"] == threshold.tag

    background_calls = [
        call
        for call in backend.model.mesh.field.calls
        if call[0] == "setAsBackgroundMesh"
    ]
    assert background_calls == [
        ("setAsBackgroundMesh", threshold.tag, "background-retry"),
        ("setAsBackgroundMesh", threshold.tag, "background-retry"),
    ]
    assert backend.option.calls == []


def test_background_rejects_non_size_fields_and_repeated_selection_pre_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-background", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = _mesher(cad).distance_field(points=[point])
        threshold = _fake_threshold(cad, distance)
        calls = list(backend.model.mesh.field.calls)

        with pytest.raises(TypeError):
            _mesher(cad).background_field(object())
        with pytest.raises(ValueError, match="Threshold or Min"):
            _mesher(cad).background_field(distance)
        assert backend.model.mesh.field.calls == calls

        _mesher(cad).background_field(threshold)
        calls = list(backend.model.mesh.field.calls)
        with pytest.raises(ValueError, match="only once"):
            _mesher(cad).background_field(threshold)
        assert backend.model.mesh.field.calls == calls


@pytest.mark.parametrize("mode", ["point", "background"])
def test_typed_size_mode_conflicts_are_retryable_before_native_generation(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{mode}-mesh-conflict", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = _mesher(cad).distance_field(points=[point])
        threshold = _fake_threshold(cad, distance)
        if mode == "point":
            _mesher(cad).mesh_size([point], size=0.1)
            field_calls = list(backend.model.mesh.field.calls)
            with pytest.raises(ValueError, match="point sizes"):
                _mesher(cad).background_field(threshold)
            assert backend.model.mesh.field.calls == field_calls
        else:
            _mesher(cad).background_field(threshold)
            mesh_calls = list(backend.model.mesh.calls)
            with pytest.raises(ValueError, match="background field"):
                _mesher(cad).mesh_size([point], size=0.1)
            assert backend.model.mesh.calls == mesh_calls

        mesh_calls = list(backend.model.mesh.calls)
        option_calls = list(backend.option.calls)
        with pytest.raises(ValueError, match="size cannot be supplied"):
            _generate_mesh(cad, size=0.2)
        assert backend.model.mesh.calls == mesh_calls
        assert backend.option.calls == option_calls

        assert isinstance(_generate_mesh(cad, ), gmsh_meshing.GmshMeshRef)


@pytest.mark.parametrize("mode", ["point", "background"])
def test_typed_size_modes_request_and_restore_all_deterministic_options(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    backend = _FakeGmsh()
    original = {
        "Mesh.ElementOrder": 7.0,
        "Mesh.SecondOrderIncomplete": 0.25,
        "Mesh.RecombineAll": 0.75,
        "Mesh.MeshSizeFromPoints": 0.3,
        "Mesh.MeshSizeFromCurvature": 4.0,
        "Mesh.MeshSizeExtendFromBoundary": 0.2,
        "Mesh.MeshSizeMin": 0.03,
        "Mesh.MeshSizeMax": 0.9,
        "Mesh.MeshSizeFactor": 2.5,
    }
    backend.option.values.update(original)
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{mode}-options", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        if mode == "point":
            _mesher(cad).mesh_size([point], size=0.1)
        else:
            distance = _mesher(cad).distance_field(points=[point])
            _mesher(cad).background_field(_fake_threshold(cad, distance))
        backend.option.calls.clear()

        assert isinstance(
            _generate_mesh(cad, order=2, recombine=True),
            gmsh_meshing.GmshMeshRef,
        )

    assert backend.option.values == original
    requested = [
        call for call in backend.option.calls if call[0] == "setNumber"
    ][:9]
    assert requested == [
        ("setNumber", "Mesh.ElementOrder", 2.0),
        ("setNumber", "Mesh.SecondOrderIncomplete", 1.0),
        ("setNumber", "Mesh.RecombineAll", 1.0),
        (
            "setNumber",
            "Mesh.MeshSizeFromPoints",
            1.0 if mode == "point" else 0.0,
        ),
        ("setNumber", "Mesh.MeshSizeFromCurvature", 0.0),
        (
            "setNumber",
            "Mesh.MeshSizeExtendFromBoundary",
            1.0 if mode == "point" else 0.0,
        ),
        ("setNumber", "Mesh.MeshSizeMin", 0.0),
        ("setNumber", "Mesh.MeshSizeMax", 1.0e22),
        ("setNumber", "Mesh.MeshSizeFactor", 1.0),
    ]


@pytest.mark.parametrize(
    ("failure", "mode"),
    [("mesh", "background"), ("option", "point")],
)
def test_typed_size_generation_failures_restore_every_external_option(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    mode: str,
) -> None:
    backend = _FakeGmsh()
    original = {
        "Mesh.ElementOrder": 4.0,
        "Mesh.SecondOrderIncomplete": 0.5,
        "Mesh.RecombineAll": 0.25,
        "Mesh.MeshSizeFromPoints": 0.3,
        "Mesh.MeshSizeFromCurvature": 2.0,
        "Mesh.MeshSizeExtendFromBoundary": 0.4,
        "Mesh.MeshSizeMin": 0.06,
        "Mesh.MeshSizeMax": 0.8,
        "Mesh.MeshSizeFactor": 1.7,
    }
    backend.option.values.update(original)
    backend.model.mesh.fail_generate = failure == "mesh"
    if failure == "option":
        backend.option.fail_set_names.add("Mesh.MeshSizeMax")
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{mode}-{failure}-restore", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        if mode == "point":
            _mesher(cad).mesh_size([point], size=0.1)
        else:
            distance = _mesher(cad).distance_field(points=[point])
            _mesher(cad).background_field(_fake_threshold(cad, distance))

        expected = {
            "mesh": "fake mesh failure",
            "option": "Mesh.MeshSizeMax",
        }[failure]
        with pytest.raises(RuntimeError, match=expected):
            _generate_mesh(cad, order=2, recombine=True)

        assert backend.option.values == original
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_mesh(cad, )


def test_nested_typed_size_modes_keep_fields_model_local_and_restore_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(
        initialized=True,
        names=("external",),
        current="external",
    )
    original = {
        "Mesh.ElementOrder": 3.0,
        "Mesh.SecondOrderIncomplete": 0.2,
        "Mesh.RecombineAll": 0.4,
        "Mesh.MeshSizeFromPoints": 0.6,
        "Mesh.MeshSizeFromCurvature": 2.0,
        "Mesh.MeshSizeExtendFromBoundary": 0.5,
        "Mesh.MeshSizeMin": 0.02,
        "Mesh.MeshSizeMax": 0.9,
        "Mesh.MeshSizeFactor": 1.8,
    }
    backend.option.values.update(original)
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-size-mode", dimension=2) as outer:
        outer.rectangle(0, 0, 1, 1)
        outer_point = _fake_entities(outer, backend, 0, 1)[0]
        distance = _mesher(outer).distance_field(points=[outer_point])
        threshold = _fake_threshold(outer, distance)
        _mesher(outer).background_field(threshold)
        assert set(backend.model._current_data()["mesh_fields"]) == {1, 2}

        with geometry.model("inner-size-mode", dimension=2) as inner:
            inner.rectangle(0, 0, 1, 1)
            inner_point = _fake_entities(inner, backend, 0, 1)[0]
            _mesher(inner).mesh_size([inner_point], size=0.1)
            _generate_mesh(inner, )
            assert backend.option.values == original

        assert backend.model.current == "outer-size-mode"
        assert set(backend.model._current_data()["mesh_fields"]) == {1, 2}
        _generate_mesh(outer, )
        assert backend.option.values == original

    assert [
        call for call in backend.model.mesh.calls if call[0] == "generate"
    ] == [
        ("generate", 2, "inner-size-mode"),
        ("generate", 2, "outer-size-mode"),
    ]
    assert backend.model.current == "external"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": 0},
        {"size": float("nan")},
        {"order": True},
        {"order": 3},
        {"recombine": 1},
    ],
)
def test_invalid_mesh_arguments_fail_before_mesh_or_option_mutation(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("mesh", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        with pytest.raises((TypeError, ValueError)):
            _generate_mesh(cad, **kwargs)
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []


def test_generation_surface_is_owned_only_by_mesher_specs() -> None:
    removed = {
        "generate_mesh",
        "generate_auto_mesh",
        "transfinite_curve",
        "transfinite_surface",
        "transfinite_volume",
        "recombine",
        "mesh_size",
        "distance_field",
        "threshold_field",
        "min_field",
        "background_field",
    }
    assert all(not hasattr(geometry.GeometryModel, name) for name in removed)
    assert tuple(inspect.signature(gmsh_meshing.Mesher.generate).parameters) == (
        "self",
        "spec",
    )
    assert tuple(inspect.signature(gmsh_meshing.MeshSpec).parameters) == (
        "size",
        "order",
        "recombine",
    )
    assert tuple(inspect.signature(gmsh_meshing.AutoMeshSpec).parameters) == (
        "level",
        "cell_shape",
        "order",
    )


def test_missing_top_dimensional_entity_leaves_mesher_retryable_but_geometry_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("mesh", dimension=2) as cad:
        with pytest.raises(ValueError, match="top-dimensional"):
            _generate_mesh(cad, )
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []

        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            cad.rectangle(0, 0, 1, 1)
        with pytest.raises(ValueError, match="top-dimensional"):
            _generate_mesh(cad, )


def test_generate_mesh_assigns_size_isolates_options_and_returns_live_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(
        {
            "Mesh.ElementOrder": 7.0,
            "Mesh.SecondOrderIncomplete": 0.25,
            "Mesh.RecombineAll": 0.75,
            "Mesh.MeshSizeFromPoints": 0.0,
        }
    )
    _install_backend(monkeypatch, backend)

    with geometry.model("mesh", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].update({(0, 3), (0, 1)})
        result = _generate_mesh(cad,
            size=0.2,
            order=2,
            recombine=True,
        )

        assert isinstance(result, gmsh_meshing.GmshMeshRef)
        assert (result.dimension, result.model_name) == (2, "mesh")
        assert "GeometryModel" not in repr(result)
        assert "object" not in repr(result)
        assert result._borrow_model() is backend.model
        assert result._borrow_model() is backend.model
        with pytest.raises(FrozenInstanceError):
            result.model_name = "renamed"  # type: ignore[misc]
        with pytest.raises(geometry.GeometryStateError, match="MESHED"):
            _generate_mesh(cad, )
        with pytest.raises(geometry.GeometryStateError, match="MESHED"):
            cad.entities(2)

    with pytest.raises(
        gmsh_meshing.StaleGmshMeshError,
        match="mesh.*inside the owning geometry model context",
    ):
        result._borrow_model()

    assert backend.model.mesh.calls == [
        ("setSize", ((0, 1), (0, 3)), 0.2, "mesh"),
        ("generate", 2, "mesh"),
    ]
    assert backend.option.values == {
        "Mesh.ElementOrder": 7.0,
        "Mesh.SecondOrderIncomplete": 0.25,
        "Mesh.RecombineAll": 0.75,
        "Mesh.MeshSizeFromPoints": 0.0,
    }
    set_calls = [call for call in backend.option.calls if call[0] == "setNumber"]
    assert set_calls[:4] == [
        ("setNumber", "Mesh.ElementOrder", 2.0),
        ("setNumber", "Mesh.SecondOrderIncomplete", 1.0),
        ("setNumber", "Mesh.RecombineAll", 1.0),
        ("setNumber", "Mesh.MeshSizeFromPoints", 1.0),
    ]


def test_success_completion_uses_only_prevalidated_assignment_steps_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("no-fail-completion", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        builder = _mesher(cad)
        session_type = type(cad._session)
        state_type = type(cad._states)
        events: list[str] = []

        original_borrow_validation = session_type.validate_native_borrow
        original_activate_borrow = session_type.activate_native_borrow
        original_mark_meshed = state_type.mark_meshed_prevalidated
        original_activate_lease = mesh_types._GeneratedMeshLease._activate

        def validate_borrow(session: Any, capability: Any, operation: str) -> None:
            events.append("validate-native-borrow")
            original_borrow_validation(session, capability, operation)

        def activate_borrow(session: Any, capability: Any) -> None:
            events.append("activate-native-borrow")
            original_activate_borrow(session, capability)

        def mark_meshed(states: Any) -> None:
            events.append("mark-meshed")
            original_mark_meshed(states)

        def activate_lease(lease: Any) -> None:
            events.append("activate-mesh-lease")
            original_activate_lease(lease)

        monkeypatch.setattr(
            session_type,
            "validate_native_borrow",
            validate_borrow,
        )
        monkeypatch.setattr(
            session_type,
            "activate_native_borrow",
            activate_borrow,
        )
        monkeypatch.setattr(state_type, "mark_meshed_prevalidated", mark_meshed)
        monkeypatch.setattr(mesh_types._GeneratedMeshLease, "_activate", activate_lease)

        result = builder.generate(gmsh_meshing.MeshSpec())

        assert events == [
            "validate-native-borrow",
            "activate-native-borrow",
            "mark-meshed",
            "activate-mesh-lease",
        ]
        assert cad._states.state.name == "MESHED"
        assert result._borrow_model() is backend.model


@pytest.mark.parametrize(
    "failure_boundary",
    [
        "native-borrow-preparation",
        "bearer-token-allocation",
        "lease-allocation",
        "reference-allocation",
        "completion-prevalidation",
    ],
)
def test_generation_completion_preparation_failure_is_terminal_and_dormant(
    monkeypatch: pytest.MonkeyPatch,
    failure_boundary: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    captured: dict[str, Any] = {}

    with geometry.model(f"completion-{failure_boundary}", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        builder = _mesher(cad)
        port_type = type(builder._runtime._port)

        if failure_boundary == "native-borrow-preparation":
            def fail_native_borrow(port: Any, operation: str) -> Any:
                raise RuntimeError("injected native borrow preparation failure")

            monkeypatch.setattr(port_type, "prepare_native_borrow", fail_native_borrow)
        elif failure_boundary == "bearer-token-allocation":
            def fail_bearer_token_allocation() -> object:
                raise RuntimeError("injected bearer token allocation failure")

            monkeypatch.setattr(
                mesh_types,
                "_new_bearer_token",
                fail_bearer_token_allocation,
            )
        elif failure_boundary == "lease-allocation":
            def fail_lease_allocation(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("injected mesh lease allocation failure")

            monkeypatch.setattr(
                mesh_types._GeneratedMeshLease,
                "__init__",
                fail_lease_allocation,
            )
        elif failure_boundary == "reference-allocation":
            def fail_reference_allocation(*args: Any, **kwargs: Any) -> None:
                captured["lease"] = args[2]
                raise RuntimeError("injected mesh reference allocation failure")

            monkeypatch.setattr(mesh_types, "GmshMeshRef", fail_reference_allocation)
        else:
            original_prepare = runtime_module._prepare_generated_mesh_reference

            def capture_prepared_reference(*args: Any, **kwargs: Any) -> Any:
                reference, lease = original_prepare(*args, **kwargs)
                captured["reference"] = reference
                captured["lease"] = lease
                return reference, lease

            def fail_prevalidation(port: Any, operation: str) -> None:
                raise RuntimeError("injected completion prevalidation failure")

            monkeypatch.setattr(
                runtime_module,
                "_prepare_generated_mesh_reference",
                capture_prepared_reference,
            )
            monkeypatch.setattr(
                port_type,
                "complete_generation",
                fail_prevalidation,
            )

        with pytest.raises(RuntimeError, match="injected") as failure:
            builder.generate(gmsh_meshing.MeshSpec())

        assert any(
            "mesh generation failed" in note
            for note in getattr(failure.value, "__notes__", ())
        )
        assert cad._states.state.name == "MESH_FAILED"
        assert backend.model.mesh.generate_calls == [2]
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            builder.generate(gmsh_meshing.MeshSpec())

        lease = captured.get("lease")
        if lease is not None:
            assert lease._GeneratedMeshLease__active is False
        if failure_boundary != "native-borrow-preparation":
            native_borrow = (
                builder._runtime._port._BoundMeshingPort__prepared_native_borrow
            )
            assert native_borrow._active is False
        reference = captured.get("reference")
        if reference is not None:
            with pytest.raises(gmsh_meshing.StaleGmshMeshError):
                reference._borrow_model()


@pytest.mark.parametrize("terminal_state", ["MESH_FAILED", "CLOSED"])
def test_completion_failure_preserves_primary_when_terminalization_also_fails(
    monkeypatch: pytest.MonkeyPatch,
    terminal_state: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    captured: dict[str, Any] = {}

    with geometry.model(f"completion-primary-{terminal_state}", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        builder = _mesher(cad)
        port_type = type(builder._runtime._port)
        original_prepare = runtime_module._prepare_generated_mesh_reference

        def capture_prepared_reference(*args: Any, **kwargs: Any) -> Any:
            reference, lease = original_prepare(*args, **kwargs)
            captured["lease"] = lease
            return reference, lease

        def fail_after_terminal_state(port: Any, operation: str) -> None:
            if terminal_state == "MESH_FAILED":
                cad._states.mark_mesh_failed(operation)
            else:
                cad._states.close()
            raise RuntimeError("primary guarded completion failure")

        monkeypatch.setattr(
            runtime_module,
            "_prepare_generated_mesh_reference",
            capture_prepared_reference,
        )
        monkeypatch.setattr(
            port_type,
            "complete_generation",
            fail_after_terminal_state,
        )

        with pytest.raises(
            RuntimeError,
            match="primary guarded completion failure",
        ) as failure:
            builder.generate(gmsh_meshing.MeshSpec())

        assert cad._states.state.name == terminal_state
        assert captured["lease"]._GeneratedMeshLease__active is False
        assert any(
            "failed to enter terminal mesh-failure state" in note
            for note in getattr(failure.value, "__notes__", ())
        )
        assert any(
            "mesh generation failed" in note
            for note in getattr(failure.value, "__notes__", ())
        )


def test_same_name_replacement_during_completion_preparation_fails_dormant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    model_name = "completion-incarnation-replacement"
    captured: dict[str, Any] = {}
    original_prepare = runtime_module._prepare_generated_mesh_reference

    def prepare_then_replace(*args: Any, **kwargs: Any) -> Any:
        reference, lease = original_prepare(*args, **kwargs)
        captured["reference"] = reference
        captured["lease"] = lease
        backend.model.remove()
        backend.model.add(model_name)
        replacement_data = backend.model._current_data()
        replacement_data["entities"].add((2, 98))
        captured["replacement"] = replacement_data
        return reference, lease

    monkeypatch.setattr(
        runtime_module,
        "_prepare_generated_mesh_reference",
        prepare_then_replace,
    )

    with geometry.model(model_name, dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)

        with pytest.raises(
            geometry.GeometryStateError,
            match="incarnation is missing or replaced",
        ) as failure:
            _generate_mesh(cad)

        assert any(
            "mesh generation failed" in note
            for note in getattr(failure.value, "__notes__", ())
        )
        assert cad._states.state.name == "MESH_FAILED"
        assert captured["lease"]._GeneratedMeshLease__active is False
        with pytest.raises(gmsh_meshing.StaleGmshMeshError):
            captured["reference"]._borrow_model()

    replacement_data = captured["replacement"]
    assert model_name in backend.model.models
    assert backend.model.models[model_name] is replacement_data
    assert replacement_data["entities"] == {(2, 98)}


@pytest.mark.parametrize(
    ("failure", "cleanup_operation"),
    [
        pytest.param(
            "session-inspection",
            "inspect Gmsh session state",
            id="session-inspection",
        ),
        pytest.param(
            "facade-model-removal",
            "remove facade model",
            id="facade-model-removal",
        ),
        pytest.param(
            "prior-model-restoration",
            "restore prior model",
            id="prior-model-restoration",
        ),
        pytest.param(
            "finalization",
            "finalize owned session",
            id="finalization",
        ),
    ],
)
def test_generated_handle_is_stale_when_cleanup_step_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    cleanup_operation: str,
) -> None:
    if failure == "finalization":
        backend = _FakeGmsh()
    else:
        backend = _FakeGmsh(
            initialized=True,
            names=("prior",),
            current="prior",
        )
    _install_backend(monkeypatch, backend)
    model_name = f"cleanup-{failure}"
    cad = geometry.model(model_name, dimension=2)
    cad.__enter__()
    cad.rectangle(0.0, 0.0, 1.0, 1.0)
    native_mesh = _generate_mesh(cad)
    assert native_mesh._borrow_model() is backend.model

    if failure == "session-inspection":
        backend.fail_is_initialized_count = 1
    elif failure == "facade-model-removal":
        backend.model.fail_remove = True
    elif failure == "prior-model-restoration":
        backend.model.fail_set_current_names.add("prior")
    else:
        backend.fail_finalize = True

    with pytest.raises(geometry.GeometryError, match=cleanup_operation):
        cad.__exit__(None, None, None)

    with pytest.raises(
        gmsh_meshing.StaleGmshMeshError,
        match=model_name,
    ) as captured:
        native_mesh._borrow_model()
    assert isinstance(captured.value.__cause__, geometry.GeometryStateError)
    assert "inactive or revoked" in str(captured.value.__cause__)

    if failure in {"session-inspection", "facade-model-removal"}:
        assert model_name in backend.model.models
    backend.model.fail_remove = False
    cad.__exit__(None, None, None)
    assert model_name not in backend.model.models


def test_generated_handle_stays_stale_across_combined_cleanup_failures_and_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(names=("prior",), current="prior")
    _install_backend(monkeypatch, backend)
    model_name = "combined-cleanup-failures"
    cad = geometry.model(model_name, dimension=2)
    cad.__enter__()
    cad.rectangle(0.0, 0.0, 1.0, 1.0)
    native_mesh = _generate_mesh(cad)
    assert native_mesh._borrow_model() is backend.model
    assert backend.initialize_calls == 1

    backend.model.fail_remove = True
    backend.model.fail_set_current_names.add("prior")
    backend.fail_finalize = True
    with pytest.raises(
        geometry.GeometryError,
        match="combined-cleanup-failures.*remove facade model",
    ) as captured:
        cad.__exit__(None, None, None)

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert any(
        "restore prior model" in note
        for note in captured.value.__notes__
    )
    assert any(
        "finalize owned session" in note
        for note in captured.value.__notes__
    )
    assert model_name in backend.model.models
    assert backend.initialized
    with pytest.raises(
        gmsh_meshing.StaleGmshMeshError,
        match=model_name,
    ) as stale:
        native_mesh._borrow_model()
    assert isinstance(stale.value.__cause__, geometry.GeometryStateError)
    assert "inactive or revoked" in str(stale.value.__cause__)

    backend.model.fail_remove = False
    cad.__exit__(None, None, None)
    assert model_name not in backend.model.models
    assert backend.model.current == "prior"
    assert backend.finalize_calls == 2
    assert not backend.initialized
    with pytest.raises(gmsh_meshing.StaleGmshMeshError, match=model_name):
        native_mesh._borrow_model()


def test_inner_cleanup_failure_revokes_only_inner_generated_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-cleanup-isolation", dimension=2) as outer:
        outer.rectangle(0.0, 0.0, 1.0, 1.0)
        outer_mesh = _generate_mesh(outer)

        inner = geometry.model("inner-cleanup-isolation", dimension=2)
        inner.__enter__()
        inner.rectangle(0.0, 0.0, 1.0, 1.0)
        inner_mesh = _generate_mesh(inner)
        assert inner_mesh._borrow_model() is backend.model

        backend.model.fail_remove = True
        with pytest.raises(
            geometry.GeometryError,
            match="inner-cleanup-isolation.*remove facade model",
        ) as captured:
            inner.__exit__(None, None, None)

        assert isinstance(captured.value.__cause__, RuntimeError)
        assert "inner-cleanup-isolation" in backend.model.models
        with pytest.raises(
            gmsh_meshing.StaleGmshMeshError,
            match="inner-cleanup-isolation",
        ) as stale:
            inner_mesh._borrow_model()
        assert isinstance(stale.value.__cause__, geometry.GeometryStateError)
        assert "inactive or revoked" in str(stale.value.__cause__)

        backend.model.setCurrent("inner-cleanup-isolation")
        assert outer_mesh._borrow_model() is backend.model
        assert backend.model.current == "outer-cleanup-isolation"

        backend.model.fail_remove = False
        inner.__exit__(None, None, None)
        assert "inner-cleanup-isolation" not in backend.model.models
        with pytest.raises(
            gmsh_meshing.StaleGmshMeshError,
            match="inner-cleanup-isolation",
        ):
            inner_mesh._borrow_model()
        assert outer_mesh._borrow_model() is backend.model
        assert backend.model.current == "outer-cleanup-isolation"

    with pytest.raises(
        gmsh_meshing.StaleGmshMeshError,
        match="outer-cleanup-isolation",
    ):
        outer_mesh._borrow_model()


def test_generated_handle_reactivates_owner_across_nested_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(
        initialized=True,
        names=("external",),
        current="external",
    )
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-native-mesh", dimension=2) as outer:
        outer.rectangle(0.0, 0.0, 1.0, 1.0)
        outer_mesh = _generate_mesh(outer, )

        with geometry.model("inner-native-mesh", dimension=3) as inner:
            inner.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            assert backend.model.current == "inner-native-mesh"

            assert outer_mesh._borrow_model() is backend.model
            assert backend.model.current == "outer-native-mesh"
            assert len(inner.entities(3)) == 1
            assert backend.model.current == "inner-native-mesh"

        assert outer_mesh._borrow_model() is backend.model
        assert backend.model.current == "outer-native-mesh"

    assert backend.model.current == "external"
    with pytest.raises(gmsh_meshing.StaleGmshMeshError, match="outer-native-mesh"):
        outer_mesh._borrow_model()


def test_generated_handle_rejects_forged_generation_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("native-identity", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        native_mesh = _generate_mesh(cad, )
        forged = gmsh_meshing.GmshMeshRef(
            native_mesh.dimension,
            native_mesh.model_name,
            cad,
            object(),
        )

        with pytest.raises(gmsh_meshing.StaleGmshMeshError, match="native-identity"):
            forged._borrow_model()


def test_generated_handle_rejects_malformed_lease_before_dispatch() -> None:
    malformed = gmsh_meshing.GmshMeshRef(
        2,
        "malformed-owner",
        object(),  # type: ignore[arg-type]
        object(),
    )

    with pytest.raises(gmsh_meshing.StaleGmshMeshError, match="malformed-owner"):
        malformed._borrow_model()


def test_generated_handle_detects_missing_native_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("missing-native-model", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        native_mesh = _generate_mesh(cad, )
        del backend.model.models["missing-native-model"]

        with pytest.raises(
            gmsh_meshing.StaleGmshMeshError,
            match="missing-native-model",
        ):
            native_mesh._borrow_model()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"order": 2}, "order"),
        ({"recombine": True}, "recombine"),
    ],
)
def test_1d_mesh_contract_is_validated_before_mesh_or_option_mutation(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("members", dimension=1) as cad:
        start = cad.point(0, 0, 0)
        end = cad.point(1, 0, 0)
        cad.line(start, end)
        with pytest.raises(ValueError, match=message):
            _generate_mesh(cad, **kwargs)
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []


def test_1d_generate_mesh_returns_native_handle_and_restores_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(
        {
            "Mesh.ElementOrder": 7.0,
            "Mesh.SecondOrderIncomplete": 0.25,
            "Mesh.RecombineAll": 0.75,
            "Mesh.MeshSizeFromPoints": 0.0,
        }
    )
    _install_backend(monkeypatch, backend)

    with geometry.model("members", dimension=1) as cad:
        start = cad.point(0, 0, 1)
        end = cad.point(2, 3, 4)
        cad.line(start, end)
        result = _generate_mesh(cad, size=0.25)
        assert isinstance(result, gmsh_meshing.GmshMeshRef)
        assert (result.dimension, result.model_name) == (1, "members")

    assert backend.model.mesh.calls == [
        ("setSize", ((0, 1), (0, 2)), 0.25, "members"),
        ("generate", 1, "members"),
    ]
    assert backend.option.values == {
        "Mesh.ElementOrder": 7.0,
        "Mesh.SecondOrderIncomplete": 0.25,
        "Mesh.RecombineAll": 0.75,
        "Mesh.MeshSizeFromPoints": 0.0,
    }


def test_1d_missing_curve_preflight_keeps_mesher_retryable_and_seals_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("members", dimension=1) as cad:
        cad.point(0, 0, 0)
        with pytest.raises(ValueError, match="top-dimensional"):
            _generate_mesh(cad, )
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []

        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            cad.point(1, 0, 0)
        with pytest.raises(ValueError, match="top-dimensional"):
            _generate_mesh(cad, )


def test_failed_generation_restores_options_and_disallows_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(
        {
            "Mesh.ElementOrder": 4.0,
            "Mesh.SecondOrderIncomplete": 0.0,
            "Mesh.RecombineAll": 0.5,
            "Mesh.MeshSizeFromPoints": 0.0,
        }
    )
    backend.model.mesh.fail_generate = True
    _install_backend(monkeypatch, backend)

    with geometry.model("mesh", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].add((0, 1))
        with pytest.raises(RuntimeError, match="fake mesh failure") as captured:
            _generate_mesh(cad, size=0.2, order=2, recombine=True)

        assert any(
            "mesh generation failed" in note
            for note in getattr(captured.value, "__notes__", ())
        )

        assert backend.option.values == {
            "Mesh.ElementOrder": 4.0,
            "Mesh.SecondOrderIncomplete": 0.0,
            "Mesh.RecombineAll": 0.5,
            "Mesh.MeshSizeFromPoints": 0.0,
        }
        assert len(cad.entities(2)) == 1
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_mesh(cad, )


def test_size_assignment_without_points_consumes_mesh_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("mesh", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        with pytest.raises(geometry.GeometryError, match="point"):
            _generate_mesh(cad, size=0.25)
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_mesh(cad, )
    assert backend.option.calls == []


def test_option_set_failure_restores_snapshot_and_marks_mesh_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    original = {
        "Mesh.ElementOrder": 3.0,
        "Mesh.SecondOrderIncomplete": 0.0,
        "Mesh.RecombineAll": 0.0,
    }
    backend.option.values.update(original)
    backend.option.fail_set_names.add("Mesh.RecombineAll")
    _install_backend(monkeypatch, backend)

    with geometry.model("mesh", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        with pytest.raises(RuntimeError, match="Mesh.RecombineAll"):
            _generate_mesh(cad, order=2, recombine=True)
        assert backend.option.values == original
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_mesh(cad, )


@pytest.mark.parametrize(
    (
        "dimension",
        "cell_shape",
        "order",
        "element_type",
        "policy_options",
    ),
    [
        (
            1,
            None,
            1,
            1,
            {
                "Mesh.RecombineAll": 0.0,
                "Mesh.SubdivisionAlgorithm": 0.0,
            },
        ),
        (
            2,
            None,
            1,
            2,
            {
                "Mesh.Algorithm": 6.0,
                "Mesh.RecombineAll": 1.0,
                "Mesh.RecombinationAlgorithm": 1.0,
                "Mesh.SubdivisionAlgorithm": 0.0,
            },
        ),
        (
            2,
            "tri",
            2,
            9,
            {
                "Mesh.Algorithm": 6.0,
                "Mesh.RecombineAll": 0.0,
                "Mesh.SubdivisionAlgorithm": 0.0,
            },
        ),
        (
            2,
            "tri-quad",
            2,
            16,
            {
                "Mesh.Algorithm": 6.0,
                "Mesh.RecombineAll": 1.0,
                "Mesh.RecombinationAlgorithm": 1.0,
                "Mesh.SubdivisionAlgorithm": 0.0,
            },
        ),
        (
            2,
            "quad",
            2,
            16,
            {
                "Mesh.Algorithm": 6.0,
                "Mesh.RecombineAll": 1.0,
                "Mesh.RecombinationAlgorithm": 3.0,
                "Mesh.SubdivisionAlgorithm": 0.0,
            },
        ),
        (
            3,
            None,
            1,
            4,
            {
                "Mesh.Algorithm": 6.0,
                "Mesh.Algorithm3D": 1.0,
                "Mesh.RecombineAll": 0.0,
                "Mesh.Recombine3DAll": 0.0,
                "Mesh.SubdivisionAlgorithm": 0.0,
            },
        ),
        (
            3,
            "tet",
            2,
            11,
            {
                "Mesh.Algorithm": 6.0,
                "Mesh.Algorithm3D": 1.0,
                "Mesh.RecombineAll": 0.0,
                "Mesh.Recombine3DAll": 0.0,
                "Mesh.SubdivisionAlgorithm": 0.0,
            },
        ),
        (
            3,
            "hex",
            2,
            17,
            {
                "Mesh.Algorithm": 6.0,
                "Mesh.Algorithm3D": 1.0,
                "Mesh.RecombineAll": 0.0,
                "Mesh.Recombine3DAll": 0.0,
                "Mesh.SubdivisionAlgorithm": 2.0,
            },
        ),
    ],
)
def test_auto_mesh_legal_shape_matrix_uses_exact_fixed_policy(
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    cell_shape: Any,
    order: int,
    element_type: int,
    policy_options: dict[str, float],
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    _install_backend(monkeypatch, backend)

    with geometry.model(
        f"auto-policy-{dimension}-{cell_shape}-{order}",
        dimension=dimension,
    ) as cad:
        _build_fake_topology(cad)
        _set_fake_element_blocks(backend, dimension, (element_type, (101, 102)))
        result = _generate_auto_mesh(cad,
            cell_shape=cell_shape,
            order=order,
        )

    expected_options = {
        "Mesh.ElementOrder": float(order),
        "Mesh.SecondOrderIncomplete": 1.0 if order == 2 else 0.0,
        "Mesh.MeshSizeFactor": 1.0,
        **policy_options,
    }
    assert isinstance(result, gmsh_meshing.GmshMeshRef)
    assert result.dimension == dimension
    assert _first_requested_options(backend) == expected_options
    assert backend.option.values == _AUTO_OPTION_ORIGINALS
    assert backend.model.mesh.generate_calls == [dimension]
    assert backend.model.mesh.refine_calls == 0
    assert [call for call in backend.model.mesh.calls if call[0] == "getElements"] == [
        (
            "getElements",
            dimension,
            -1,
            f"auto-policy-{dimension}-{cell_shape}-{order}",
        )
    ]


@pytest.mark.parametrize(
    ("dimension", "cell_shape"),
    [
        *((1, shape) for shape in ("tri", "tri-quad", "quad", "tet", "hex")),
        (2, "tet"),
        (2, "hex"),
        (3, "tri"),
        (3, "tri-quad"),
        (3, "quad"),
        (2, "TRI"),
        (2, "triangle"),
        (2, 3),
    ],
)
def test_auto_mesh_rejects_invalid_shape_matrix_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    cell_shape: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(
        f"invalid-auto-shape-{dimension}-{cell_shape}",
        dimension=dimension,
    ) as cad:
        _build_fake_topology(cad)
        synchronize_calls = backend.model.occ.synchronize_calls
        with pytest.raises((TypeError, ValueError)) as captured:
            _generate_auto_mesh(cad, cell_shape=cell_shape)

        assert "cell_shape" in str(captured.value)
        assert "cell_shape" in str(captured.value)
        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []


@pytest.mark.parametrize("level", [True, False, 1.0, "3", 0, 6, -1])
def test_auto_mesh_rejects_invalid_levels_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
    level: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"invalid-auto-level-{level}", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        synchronize_calls = backend.model.occ.synchronize_calls
        with pytest.raises((TypeError, ValueError)) as captured:
            _generate_auto_mesh(cad, level=level)

        assert "level" in str(captured.value)
        assert "level" in str(captured.value)
        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_auto_mesh_levels_set_dimension_aware_absolute_size_factor(
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    level: int,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    _install_backend(monkeypatch, backend)
    default_type = {1: 1, 2: 2, 3: 4}[dimension]

    with geometry.model(f"auto-level-{dimension}-{level}", dimension=dimension) as cad:
        _build_fake_topology(cad)
        _set_fake_element_blocks(backend, dimension, (default_type, (1,)))
        _generate_auto_mesh(cad, level=level)

    assert _first_requested_options(backend)["Mesh.MeshSizeFactor"] == pytest.approx(
        2.0 ** ((3 - level) / dimension)
    )
    assert backend.option.values == _AUTO_OPTION_ORIGINALS


def test_auto_mesh_preflight_validation_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("auto-validation-retry", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        with pytest.raises((TypeError, ValueError)):
            _generate_auto_mesh(cad, level=True)

        _set_fake_element_blocks(backend, 2, (2, (1,)))
        assert isinstance(_generate_auto_mesh(cad, level=3), gmsh_meshing.GmshMeshRef)


@pytest.mark.parametrize(
    ("dimension", "cell_shape", "order", "element_type"),
    [
        (1, None, 1, 1),
        (2, "tri", 1, 2),
        (2, "tri", 2, 9),
        (2, "quad", 1, 3),
        (2, "quad", 2, 16),
        (3, "tet", 1, 4),
        (3, "tet", 2, 11),
        (3, "hex", 1, 5),
        (3, "hex", 2, 17),
    ],
)
def test_auto_mesh_strict_pure_families_return_native_handles(
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    cell_shape: Any,
    order: int,
    element_type: int,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(
        f"strict-pure-{dimension}-{cell_shape}-{order}",
        dimension=dimension,
    ) as cad:
        _build_fake_topology(cad)
        _set_fake_element_blocks(backend, dimension, (element_type, (1, 2)))
        result = _generate_auto_mesh(cad,
            cell_shape=cell_shape,
            order=order,
        )

    assert isinstance(result, gmsh_meshing.GmshMeshRef)
    assert result.dimension == dimension
    assert backend.model.mesh.generate_calls == [dimension]
    assert backend.model.mesh.refine_calls == 0


@pytest.mark.parametrize(
    ("order", "blocks"),
    [
        (1, ((2, (1, 2)),)),
        (1, ((3, (1, 2)),)),
        (1, ((2, (1,)), (3, (2, 3)))),
        (2, ((9, (1, 2)),)),
        (2, ((16, (1, 2)),)),
        (2, ((9, (1,)), (16, (2, 3)))),
    ],
)
def test_auto_mesh_tri_quad_accepts_each_permitted_family_union(
    monkeypatch: pytest.MonkeyPatch,
    order: int,
    blocks: tuple[tuple[int, Sequence[int]], ...],
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"strict-mixed-{order}-{len(blocks)}", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        _set_fake_element_blocks(backend, 2, *blocks)
        result = _generate_auto_mesh(cad, cell_shape="tri-quad", order=order)

    assert isinstance(result, gmsh_meshing.GmshMeshRef)
    assert backend.model.mesh.generate_calls == [2]
    assert backend.model.mesh.refine_calls == 0


@pytest.mark.parametrize(
    ("raw_blocks", "message"),
    [
        (([], [], []), "no top-dimensional cells"),
        (([3], [[]], [[]]), "no top-dimensional cells"),
        (
            ([3, 2], [[1]], [[], []]),
            "malformed top-dimensional element blocks",
        ),
        (([3], [None], [[]]), "malformed element tags"),
        (([True], [[1]], [[]]), "non-integer element type"),
    ],
)
def test_auto_mesh_strict_validation_rejects_empty_and_malformed_output(
    monkeypatch: pytest.MonkeyPatch,
    raw_blocks: Any,
    message: str,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    _install_backend(monkeypatch, backend)

    with geometry.model(f"strict-malformed-{message}", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        backend.model._current_data()["element_blocks"][2] = raw_blocks
        with pytest.raises(gmsh_meshing.MeshCellShapeError, match=message):
            _generate_auto_mesh(cad, cell_shape="quad")

        assert len(cad.entities(2)) == 1
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_auto_mesh(cad, cell_shape="quad")

    assert backend.model.mesh.generate_calls == [2]
    assert backend.option.values == _AUTO_OPTION_ORIGINALS


def test_auto_mesh_shape_error_reports_aggregated_named_actual_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    _install_backend(monkeypatch, backend)

    with geometry.model("strict-named-diagnostic", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        _set_fake_element_blocks(
            backend,
            2,
            (2, (1, 2)),
            (3, (3,)),
            (2, (4,)),
        )
        with pytest.raises(gmsh_meshing.MeshCellShapeError) as captured:
            _generate_auto_mesh(cad, cell_shape="quad", order=2)

    message = str(captured.value)
    for fragment in (
        "strict-named-diagnostic",
        "AutoMeshSpec",
        "cell_shape='quad'",
        "dimension=2",
        "order=2",
        "Quadrilateral 8",
        "Triangle 3=3",
        "Quadrilateral 4=1",
        "automatic fallback is disabled",
    ):
        assert fragment in message
    assert [
        call[1]
        for call in backend.model.mesh.calls
        if call[0] == "getElementProperties"
    ] == [2, 3]
    assert backend.option.values == _AUTO_OPTION_ORIGINALS


def test_auto_mesh_shape_diagnostic_falls_back_to_unknown_numeric_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("strict-unknown-diagnostic", dimension=3) as cad:
        cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        _set_fake_element_blocks(backend, 3, (99, (1, 2)))
        backend.model.mesh.fail_element_properties.add(99)
        with pytest.raises(
            gmsh_meshing.MeshCellShapeError,
            match=r"Gmsh type 99=2",
        ):
            _generate_auto_mesh(cad, cell_shape="hex")

    assert 99 not in backend.model.mesh.fail_element_properties


def test_auto_mesh_get_elements_failure_preserves_native_error_and_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    backend.model.mesh.fail_get_elements_dimensions.add(2)
    _install_backend(monkeypatch, backend)

    with geometry.model("strict-query-failure", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        with pytest.raises(RuntimeError, match="fake getElements failure"):
            _generate_auto_mesh(cad, cell_shape="tri")
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_mesh(cad, )

    assert backend.model.mesh.generate_calls == [2]
    assert backend.option.values == _AUTO_OPTION_ORIGINALS


def test_auto_mesh_strict_validation_ignores_lower_dimensional_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("strict-top-dimension-only", dimension=3) as cad:
        cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        _set_fake_element_blocks(backend, 2, (2, (20,)), (3, (21,)))
        _set_fake_element_blocks(backend, 3, (17, (30, 31)))
        result = _generate_auto_mesh(cad, cell_shape="hex", order=2)

    assert isinstance(result, gmsh_meshing.GmshMeshRef)
    assert [
        call[1] for call in backend.model.mesh.calls if call[0] == "getElements"
    ] == [3]


@pytest.mark.parametrize("mode", ["point", "background"])
def test_auto_mesh_typed_size_modes_compose_factor_once_and_restore_options(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    _install_backend(monkeypatch, backend)

    with geometry.model(f"auto-typed-size-{mode}", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        point = _fake_entities(cad, backend, 0, 11)[0]
        if mode == "point":
            _mesher(cad).mesh_size([point], size=0.1)
        else:
            distance = _mesher(cad).distance_field(points=[point])
            _mesher(cad).background_field(_fake_threshold(cad, distance))
        _set_fake_element_blocks(backend, 2, (16, (1, 2)))
        backend.option.calls.clear()

        assert isinstance(
            _generate_auto_mesh(cad, level=4, cell_shape="quad", order=2),
            gmsh_meshing.GmshMeshRef,
        )

    requested = _first_requested_options(backend)
    assert requested == {
        "Mesh.ElementOrder": 2.0,
        "Mesh.SecondOrderIncomplete": 1.0,
        "Mesh.RecombineAll": 1.0,
        "Mesh.Algorithm": 6.0,
        "Mesh.RecombinationAlgorithm": 3.0,
        "Mesh.SubdivisionAlgorithm": 0.0,
        "Mesh.MeshSizeFromPoints": 1.0 if mode == "point" else 0.0,
        "Mesh.MeshSizeFromCurvature": 0.0,
        "Mesh.MeshSizeExtendFromBoundary": 1.0 if mode == "point" else 0.0,
        "Mesh.MeshSizeMin": 0.0,
        "Mesh.MeshSizeMax": 1.0e22,
        "Mesh.MeshSizeFactor": pytest.approx(2.0**-0.5),
    }
    get_names = [call[1] for call in backend.option.calls if call[0] == "getNumber"]
    set_names = [call[1] for call in backend.option.calls if call[0] == "setNumber"]
    assert len(get_names) == len(set(get_names))
    assert set(get_names) == set(requested)
    assert all(set_names.count(name) == 2 for name in requested)
    assert backend.option.values == _AUTO_OPTION_ORIGINALS


@pytest.mark.parametrize(
    "failure",
    ["option_get", "option_set", "generate", "strict"],
)
def test_auto_mesh_failures_restore_every_external_option_and_consume_attempt(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    if failure == "option_get":
        backend.option.fail_get_names.add("Mesh.Algorithm")
    elif failure == "option_set":
        backend.option.fail_set_names.add("Mesh.RecombinationAlgorithm")
    elif failure == "generate":
        backend.model.mesh.fail_generate = True
    _install_backend(monkeypatch, backend)

    with geometry.model(f"auto-failure-{failure}", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        element_type = 2 if failure == "strict" else 16
        _set_fake_element_blocks(backend, 2, (element_type, (1, 2)))
        if failure == "strict":
            expected_error: type[BaseException] = gmsh_meshing.MeshCellShapeError
            expected_message = "Triangle 3=2"
        else:
            expected_error = RuntimeError
            expected_message = {
                "option_get": "option get failure",
                "option_set": "Mesh.RecombinationAlgorithm",
                "generate": "fake mesh failure",
            }[failure]

        with pytest.raises(expected_error, match=expected_message):
            _generate_auto_mesh(cad, cell_shape="quad", order=2)

        assert backend.option.values == _AUTO_OPTION_ORIGINALS
        assert len(cad.entities(2)) == 1
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_auto_mesh(cad, cell_shape="quad")


def test_auto_mesh_successful_generation_restoration_failure_is_retried_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    backend.option.fail_set_after["Mesh.Algorithm"] = 1
    _install_backend(monkeypatch, backend)

    with geometry.model("auto-restore-failure", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        _set_fake_element_blocks(backend, 2, (3, (1, 2)))
        with pytest.raises(
            geometry.GeometryError,
            match="restoring global Gmsh options failed",
        ):
            _generate_auto_mesh(cad, cell_shape="quad")

        assert backend.option.values["Mesh.Algorithm"] == 6.0
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_auto_mesh(cad, cell_shape="quad")

    assert backend.option.values == _AUTO_OPTION_ORIGINALS


def test_auto_mesh_shape_failure_preserves_error_when_restoration_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    backend.option.fail_set_after["Mesh.Algorithm"] = 1
    _install_backend(monkeypatch, backend)

    with geometry.model("auto-shape-and-restore-failure", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        _set_fake_element_blocks(backend, 2, (2, (1,)))
        with pytest.raises(gmsh_meshing.MeshCellShapeError) as captured:
            _generate_auto_mesh(cad, cell_shape="quad")

        assert any(
            "additionally failed to restore" in note
            for note in getattr(captured.value, "__notes__", ())
        )

    assert backend.option.values == _AUTO_OPTION_ORIGINALS


def test_auto_mesh_missing_top_entity_keeps_mesher_retryable_and_seals_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("auto-missing-topology", dimension=2) as cad:
        with pytest.raises(ValueError, match="top-dimensional"):
            _generate_auto_mesh(cad, )
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []

        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            cad.rectangle(0.0, 0.0, 1.0, 1.0)
        with pytest.raises(ValueError, match="top-dimensional"):
            _generate_auto_mesh(cad, )


def test_nested_auto_mesh_models_isolate_current_model_policy_and_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(
        initialized=True,
        names=("external",),
        current="external",
    )
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-auto", dimension=2) as outer:
        outer.rectangle(0.0, 0.0, 1.0, 1.0)
        _set_fake_element_blocks(backend, 2, (3, (1, 2)))

        with geometry.model("inner-auto", dimension=3) as inner:
            inner.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            _set_fake_element_blocks(backend, 3, (4, (3, 4)))
            _generate_auto_mesh(inner, level=1, cell_shape="tet")
            assert backend.option.values == _AUTO_OPTION_ORIGINALS

        assert backend.model.current == "outer-auto"
        _generate_auto_mesh(outer, level=5, cell_shape="quad")
        assert backend.option.values == _AUTO_OPTION_ORIGINALS

    assert backend.model.current == "external"
    assert backend.model.mesh.generate_calls == [3, 2]


def test_explicit_generation_synchronizes_after_non_strict_native_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    events: list[str] = []
    synchronize = backend.model.occ.synchronize
    generate = backend.model.mesh.generate

    def record_synchronize() -> None:
        events.append("synchronize")
        synchronize()

    def record_generate(dimension: int) -> None:
        events.append("generate")
        generate(dimension)

    monkeypatch.setattr(backend.model.occ, "synchronize", record_synchronize)
    monkeypatch.setattr(backend.model.mesh, "generate", record_generate)

    with geometry.model("explicit_sync", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        _generate_mesh(cad)

    assert events[-2:] == ["generate", "synchronize"]


def test_native_control_activation_failure_leaves_structured_subphase_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(
        initialized=True,
        names=("outer",),
        current="outer",
    )
    _install_backend(monkeypatch, backend)

    with geometry.model("activation", dimension=2) as cad:
        (curve,) = _fake_entities(cad, backend, 1, 1)
        mesh_builder = _mesher(cad)
        get_boundary = backend.model.getBoundary

        def switch_current_after_boundary(*args: Any, **kwargs: Any) -> Any:
            result = get_boundary(*args, **kwargs)
            backend.model.current = "outer"
            return result

        monkeypatch.setattr(
            backend.model,
            "getBoundary",
            switch_current_after_boundary,
        )
        backend.model.fail_set_current_names.add("activation")

        with pytest.raises(RuntimeError, match="fake setCurrent failure"):
            mesh_builder.transfinite_curve(curve, num_nodes=3)

        backend.model.current = "activation"
        with pytest.raises(ValueError, match="at least one"):
            mesh_builder.structured_extrude(
                (),
                1.0,
                0.0,
                0.0,
                num_elements=(1,),
            )


def test_generated_handle_rejects_same_name_replacement_and_preserves_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(
        initialized=True,
        names=("prior",),
        current="prior",
    )
    _install_backend(monkeypatch, backend)

    with geometry.model("same-name-replacement", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        generated = _generate_mesh(cad)
        assert generated._borrow_model() is backend.model

        backend.model.remove()
        backend.model.add("same-name-replacement")
        replacement_data = backend.model._current_data()
        replacement_data["entities"].add((2, 98))
        backend.model.setCurrent("prior")

        with pytest.raises(
            gmsh_meshing.StaleGmshMeshError,
            match="same-name-replacement.*owning geometry model context",
        ) as captured:
            generated._borrow_model()

        assert "incarnation is missing or replaced" in str(
            captured.value.__cause__
        )

    assert "same-name-replacement" in backend.model.models
    assert backend.model.models["same-name-replacement"] is replacement_data
    assert replacement_data["entities"] == {(2, 98)}
    assert backend.model.current == "prior"


def test_real_gmsh_same_name_replacement_cannot_satisfy_old_reference() -> None:
    native_gmsh = pytest.importorskip("gmsh")
    from fem.io import gmsh as gmsh_io

    owns_session = not bool(native_gmsh.isInitialized())
    if owns_session:
        native_gmsh.initialize()
    previous_terminal = native_gmsh.option.getNumber("General.Terminal")
    native_gmsh.option.setNumber("General.Terminal", 0)
    native_gmsh.clear()

    model_name = "same-name-incarnation-e2e"
    prior_model_name = "same-name-incarnation-e2e-prior"
    try:
        native_gmsh.model.add(prior_model_name)
        with geometry.model(model_name, dimension=2) as cad:
            cad.rectangle(0.0, 0.0, 1.0, 1.0)
            original = _generate_mesh(cad, size=0.5)
            original_node_count = len(native_gmsh.model.mesh.getNodes()[0])

            native_gmsh.model.remove()
            native_gmsh.model.add(model_name)
            native_gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0)
            native_gmsh.model.occ.synchronize()
            native_gmsh.model.mesh.setSize(
                native_gmsh.model.getEntities(0),
                0.14,
            )
            native_gmsh.model.mesh.generate(2)
            replacement_node_count = len(native_gmsh.model.mesh.getNodes()[0])
            native_gmsh.model.setCurrent(prior_model_name)

            assert (original_node_count, replacement_node_count) == (12, 98)
            with pytest.raises(gmsh_meshing.StaleGmshMeshError):
                gmsh_io.read(original)

        assert model_name in tuple(str(item) for item in native_gmsh.model.list())
        assert str(native_gmsh.model.getCurrent()) == prior_model_name
        native_gmsh.model.setCurrent(model_name)
        assert len(native_gmsh.model.mesh.getNodes()[0]) == 98
    finally:
        native_gmsh.clear()
        native_gmsh.option.setNumber("General.Terminal", previous_terminal)
        if owns_session and bool(native_gmsh.isInitialized()):
            native_gmsh.finalize()
