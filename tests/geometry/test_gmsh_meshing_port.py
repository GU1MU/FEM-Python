from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

import pytest

from fem import geometry
from fem.geometry._gmsh.meshing_port import _BoundMeshingPort
from fem.mesh import gmsh as meshing
from fem.mesh.gmsh import mesher as mesher_module
from tests.helpers.gmsh_fake import _FakeGmsh, _install_backend


class _RecordingOwner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.result = object()
        self.results: dict[str, Any] = {}
        self.name = "port-owner"
        self.dimension = 3
        self._topology_provenance_unknown = True

    def __getattr__(self, name: str) -> Any:
        if not name.startswith(("_meshing_", "_structured_extrude")):
            raise AttributeError(name)

        def record(*args: Any, **kwargs: Any) -> object:
            self.calls.append((name, args, kwargs))
            return self.results.get(name, self.result)

        return record


@pytest.mark.parametrize(
    ("owner_method", "invoke", "returns_owner_result"),
    [
        (
            "_meshing_validate",
            lambda port: port.validate("validation"),
            False,
        ),
        (
            "_meshing_normalize_entities",
            lambda port: port.normalize_entities(
                (object(),),
                operation="normalization",
            ),
            True,
        ),
        (
            "_meshing_normalize_optional_entities",
            lambda port: port.normalize_optional_entities(
                (object(),),
                operation="optional normalization",
                label="sources",
            ),
            True,
        ),
        (
            "_meshing_assert_entities_live",
            lambda port: port.assert_entities_live(
                (object(),),
                operation="liveness",
            ),
            False,
        ),
        (
            "_meshing_boundary_closure",
            lambda port: port.boundary_closure(
                (object(),),
                operation="boundary closure",
            ),
            True,
        ),
        (
            "_meshing_assert_corners_on_boundary",
            lambda port: port.assert_corners_on_boundary(
                object(),
                (object(), object()),
                operation="corner validation",
            ),
            False,
        ),
        (
            "_meshing_commit_generation_attempt",
            lambda port: port.commit_generation_attempt("generation"),
            False,
        ),
        (
            "_meshing_register_control_dependencies",
            lambda port: port.register_control_dependencies(
                ((2, 1), (1, 2)),
                transform_unsafe=True,
            ),
            False,
        ),
        (
            "_meshing_has_pending_numeric_options",
            lambda port: port.has_pending_numeric_options,
            True,
        ),
        (
            "_meshing_apply_numeric_options",
            lambda port: port.apply_numeric_options(
                (("Mesh.ElementOrder", 2.0),),
            ),
            False,
        ),
        (
            "_meshing_restore_numeric_options",
            lambda port: port.restore_numeric_options(),
            False,
        ),
        (
            "_meshing_fail_generation",
            lambda port: port.fail_generation("generation"),
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
    ],
)
def test_bound_meshing_port_forwards_itself_as_the_only_geometry_authority(
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


def test_bound_meshing_port_snapshots_read_only_model_metadata():
    owner = _RecordingOwner()
    port = _BoundMeshingPort(owner)
    owner.name = "changed"
    owner.dimension = 1
    owner._topology_provenance_unknown = False

    assert port.model_name == "port-owner"
    assert port.dimension == 3
    assert port.topology_provenance_unknown is True
    with pytest.raises(AttributeError):
        port.model_name = "forged"
    with pytest.raises(AttributeError):
        port.dimension = 2
    with pytest.raises(AttributeError):
        port.topology_provenance_unknown = False


def test_bound_meshing_port_exposes_only_geometry_host_capabilities():
    port = _BoundMeshingPort(_RecordingOwner())

    assert {name for name in dir(port) if not name.startswith("_")} == {
        "apply_numeric_options",
        "assert_corners_on_boundary",
        "assert_entities_live",
        "boundary_closure",
        "commit_generation_attempt",
        "complete_generation",
        "dimension",
        "fail_generation",
        "has_pending_numeric_options",
        "model_name",
        "native_control",
        "native_query",
        "normalize_entities",
        "normalize_optional_entities",
        "prepare_native_borrow",
        "register_control_dependencies",
        "restore_numeric_options",
        "structured_extrude",
        "topology_provenance_unknown",
        "validate",
    }


@pytest.mark.parametrize(
    ("port_method", "owner_method"),
    [
        ("native_query", "_meshing_native_query"),
        ("native_control", "_meshing_native_control"),
    ],
)
def test_bound_meshing_port_accepts_only_backend_free_callback_results(
    port_method,
    owner_method,
):
    owner = _RecordingOwner()
    safe_result = (1, (2.0, "value"), True, None)
    owner.results[owner_method] = safe_result
    port = _BoundMeshingPort(owner)

    def callback(native_model):
        return native_model

    assert getattr(port, port_method)("native operation", callback) == safe_result
    assert owner.calls == [
        (owner_method, (port, "native operation", callback), {})
    ]

    for unsafe_result in (object(), [], {}, (object(),)):
        owner.results[owner_method] = unsafe_result
        with pytest.raises(
            TypeError,
            match="native operation native callback must return backend-free",
        ):
            getattr(port, port_method)("native operation", callback)


def test_bound_meshing_port_completes_with_its_prepared_native_borrow():
    owner = _RecordingOwner()
    native_borrow = object()
    owner.results["_meshing_prepare_native_borrow"] = native_borrow
    port = _BoundMeshingPort(owner)

    assert port.prepare_native_borrow("generation") is native_borrow
    port.complete_generation("generation")

    assert owner.calls == [
        (
            "_meshing_prepare_native_borrow",
            (port, "generation"),
            {},
        ),
        (
            "_meshing_complete_generation",
            (port, "generation", native_borrow),
            {},
        ),
    ]


def test_owner_generation_completion_tail_is_prevalidated_assignment_only():
    source = textwrap.dedent(
        inspect.getsource(geometry.GeometryModel._meshing_complete_generation)
    )
    function = ast.parse(source).body[0]

    assert isinstance(function, ast.FunctionDef)
    assert [ast.unparse(statement) for statement in function.body] == [
        "self._meshing_validate(authority, operation)",
        "self._session.validate_native_borrow(native_borrow, operation)",
        "self._session.activate_native_borrow(native_borrow)",
        "self._states.mark_meshed_prevalidated()",
    ]


def test_bound_meshing_port_rejects_missing_or_replayed_generation_completion():
    owner = _RecordingOwner()
    port = _BoundMeshingPort(owner)

    with pytest.raises(RuntimeError, match="requires a prepared native model borrow"):
        port.complete_generation("generation")
    assert owner.calls == []

    owner.results["_meshing_prepare_native_borrow"] = object()
    port.prepare_native_borrow("generation")
    port.complete_generation("generation")
    completed_calls = list(owner.calls)

    with pytest.raises(RuntimeError, match="requires a prepared native model borrow"):
        port.complete_generation("replayed generation")
    assert owner.calls == completed_calls


def test_owner_rejects_prepared_generation_completion_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("failed-generation-completion", dimension=2) as cad:
        port = cad._acquire_meshing_port()
        native_borrow = port.prepare_native_borrow("mesh generation")
        port.fail_generation("mesh generation")

        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            port.complete_generation("mesh generation")

        assert native_borrow._active is False
        assert cad._states.state.name == "MESH_FAILED"
        assert (
            port._BoundMeshingPort__prepared_native_borrow is native_borrow
        )


def test_owner_rejects_prepared_generation_completion_after_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("closed-generation-completion", dimension=2) as cad:
        port = cad._acquire_meshing_port()
        native_borrow = port.prepare_native_borrow("mesh generation")

    with pytest.raises(
        geometry.GeometryStateError,
        match="bound Mesher capability is invalid",
    ):
        port.complete_generation("mesh generation")

    assert native_borrow._active is False
    assert cad._states.state.name == "CLOSED"
    assert port._BoundMeshingPort__prepared_native_borrow is native_borrow


def test_owner_rejects_prepared_generation_completion_after_meshed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("meshed-generation-completion", dimension=2) as cad:
        port = cad._acquire_meshing_port()
        native_borrow = port.prepare_native_borrow("mesh generation")
        cad._states.mark_meshed_prevalidated()

        with pytest.raises(geometry.GeometryStateError, match="MESHED"):
            port.complete_generation("mesh generation")

        assert native_borrow._active is False
        assert cad._states.state.name == "MESHED"
        assert port._BoundMeshingPort__prepared_native_borrow is native_borrow


class _RecordingRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.results = {
            "distance_field": object(),
            "threshold_field": object(),
            "min_field": object(),
            "structured_extrude": object(),
            "generate": object(),
        }

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> object | None:
            self.calls.append((name, args, kwargs))
            return self.results.get(name)

        return record


class _FakeGeometryModel:
    def __init__(self, port: object) -> None:
        self.port = port
        self.acquisitions = 0

    def _acquire_meshing_port(self) -> object:
        self.acquisitions += 1
        return self.port


def test_mesher_stores_one_runtime_and_routes_every_public_operation(monkeypatch):
    port = object()
    runtime = _RecordingRuntime()
    geometry = _FakeGeometryModel(port)
    runtime_ports = []

    def runtime_factory(runtime_port):
        runtime_ports.append(runtime_port)
        return runtime

    monkeypatch.setattr(
        mesher_module._geometry,
        "GeometryModel",
        _FakeGeometryModel,
    )
    monkeypatch.setattr(
        mesher_module,
        "_GmshMeshRuntime",
        runtime_factory,
    )

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
        is runtime.results["distance_field"]
    )
    assert (
        builder.threshold_field(
            distance,
            size_min=0.1,
            size_max=1.0,
            dist_min=0.0,
            dist_max=2.0,
        )
        is runtime.results["threshold_field"]
    )
    assert builder.min_field(fields) is runtime.results["min_field"]
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
        is runtime.results["structured_extrude"]
    )
    assert (
        builder.generate(meshing.MeshSpec(size=0.2, order=2, recombine=True))
        is runtime.results["generate"]
    )
    assert (
        builder.generate(
            meshing.AutoMeshSpec(level=4, cell_shape="quad", order=2)
        )
        is runtime.results["generate"]
    )

    assert geometry.acquisitions == 1
    assert runtime_ports == [port]
    assert builder._runtime is runtime
    assert [name for name, _, _ in runtime.calls] == [
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
        "generate",
        "generate",
    ]
