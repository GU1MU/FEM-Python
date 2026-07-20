from __future__ import annotations

from typing import Any

import pytest

from fem.geometry._gmsh.meshing_port import _BoundMeshingPort
from fem.mesh import gmsh as meshing


class _RecordingOwner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.result = object()

    def __getattr__(self, name: str) -> Any:
        if not name.startswith(("_mesher_", "_structured_extrude")):
            raise AttributeError(name)

        def record(*args: Any, **kwargs: Any) -> object:
            self.calls.append((name, args, kwargs))
            return self.result

        return record


@pytest.mark.parametrize(
    ("owner_method", "invoke", "returns_owner_result"),
    [
        (
            "_mesher_transfinite_curve",
            lambda port: port.transfinite_curve(object(), num_nodes=3),
            False,
        ),
        (
            "_mesher_transfinite_surface",
            lambda port: port.transfinite_surface(object(), corners=()),
            False,
        ),
        (
            "_mesher_transfinite_volume",
            lambda port: port.transfinite_volume(object(), corners=()),
            False,
        ),
        ("_mesher_recombine", lambda port: port.recombine(object()), False),
        (
            "_mesher_mesh_size",
            lambda port: port.mesh_size((object(),), size=0.5),
            False,
        ),
        (
            "_mesher_distance_field",
            lambda port: port.distance_field(
                points=(), curves=(), surfaces=(), sampling=7
            ),
            True,
        ),
        (
            "_mesher_threshold_field",
            lambda port: port.threshold_field(
                object(),
                size_min=0.1,
                size_max=1.0,
                dist_min=0.0,
                dist_max=2.0,
            ),
            True,
        ),
        (
            "_mesher_min_field",
            lambda port: port.min_field((object(), object())),
            True,
        ),
        (
            "_mesher_background_field",
            lambda port: port.background_field(object()),
            False,
        ),
        (
            "_structured_extrude",
            lambda port: port.structured_extrude(
                (object(),),
                1.0,
                2.0,
                3.0,
                num_elements=(4,),
                heights=(1.0,),
                recombine=True,
            ),
            True,
        ),
        (
            "_mesher_generate_mesh",
            lambda port: port.generate_mesh(size=0.2, order=2, recombine=True),
            True,
        ),
        (
            "_mesher_generate_auto_mesh",
            lambda port: port.generate_auto_mesh(
                level=4,
                cell_shape="quad",
                order=2,
            ),
            True,
        ),
    ],
)
def test_bound_meshing_port_forwards_itself_as_the_only_authority(
    owner_method,
    invoke,
    returns_owner_result,
):
    owner = _RecordingOwner()
    port = _BoundMeshingPort(owner)

    result = invoke(port)

    assert len(owner.calls) == 1
    called_method, args, _ = owner.calls[0]
    assert called_method == owner_method
    assert args[0] is port
    if returns_owner_result:
        assert result is owner.result
    else:
        assert result is None


def test_bound_meshing_port_exposes_only_complete_mesh_transactions():
    port = _BoundMeshingPort(_RecordingOwner())

    assert {name for name in dir(port) if not name.startswith("_")} == {
        "background_field",
        "distance_field",
        "generate_auto_mesh",
        "generate_mesh",
        "mesh_size",
        "min_field",
        "recombine",
        "structured_extrude",
        "threshold_field",
        "transfinite_curve",
        "transfinite_surface",
        "transfinite_volume",
    }


class _RecordingPort:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.results = {
            "distance_field": object(),
            "threshold_field": object(),
            "min_field": object(),
            "structured_extrude": object(),
            "generate_mesh": object(),
            "generate_auto_mesh": object(),
        }

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> object | None:
            self.calls.append((name, args, kwargs))
            return self.results.get(name)

        return record


class _FakeGeometryModel:
    def __init__(self, port: _RecordingPort) -> None:
        self.port = port
        self.acquisitions = 0

    def _acquire_meshing_port(self) -> _RecordingPort:
        self.acquisitions += 1
        return self.port


def test_mesher_stores_one_port_and_routes_every_public_operation(monkeypatch):
    port = _RecordingPort()
    geometry = _FakeGeometryModel(port)
    monkeypatch.setattr(meshing._geometry, "GeometryModel", _FakeGeometryModel)

    builder = meshing.Mesher(geometry)
    curve = object()
    surface = object()
    volume = object()
    point = object()
    corners = (object(), object(), object())
    distance = object()
    fields = (object(), object())

    builder.transfinite_curve(curve, num_nodes=5)
    builder.transfinite_surface(surface, corners=corners)
    builder.transfinite_volume(volume, corners=corners)
    builder.recombine(surface)
    builder.mesh_size((point,), size=0.25)
    assert (
        builder.distance_field(
            points=(point,),
            curves=(curve,),
            surfaces=(surface,),
            sampling=11,
        )
        is port.results["distance_field"]
    )
    assert (
        builder.threshold_field(
            distance,
            size_min=0.1,
            size_max=1.0,
            dist_min=0.0,
            dist_max=2.0,
        )
        is port.results["threshold_field"]
    )
    assert builder.min_field(fields) is port.results["min_field"]
    builder.background_field(fields[0])
    assert (
        builder.structured_extrude(
            (surface,),
            1.0,
            2.0,
            3.0,
            num_elements=(4,),
            heights=(1.0,),
            recombine=True,
        )
        is port.results["structured_extrude"]
    )
    assert (
        builder.generate(meshing.MeshSpec(size=0.2, order=2, recombine=True))
        is port.results["generate_mesh"]
    )
    assert (
        builder.generate(
            meshing.AutoMeshSpec(level=4, cell_shape="quad", order=2)
        )
        is port.results["generate_auto_mesh"]
    )

    assert geometry.acquisitions == 1
    assert builder._port is port
    assert not hasattr(builder, "_geometry")
    assert not hasattr(builder, "_mesher_token")
    assert not hasattr(builder, "_complete")
    assert [name for name, _, _ in port.calls] == [
        "transfinite_curve",
        "transfinite_surface",
        "transfinite_volume",
        "recombine",
        "mesh_size",
        "distance_field",
        "threshold_field",
        "min_field",
        "background_field",
        "structured_extrude",
        "generate_mesh",
        "generate_auto_mesh",
    ]
