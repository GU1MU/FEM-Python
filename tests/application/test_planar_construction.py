from __future__ import annotations

import pytest

from fem.application.planar_construction import (
    PlanarConstructionCompileError,
    _NativeFacts,
    _evaluate,
    _require_equivalent,
    compile_planar_construction,
)
from fem.geometry import GeometryError, PlanarConstructionIR


def _rectangle_ir() -> PlanarConstructionIR:
    return PlanarConstructionIR.from_dict(
        {
            "schema_version": 1,
            "name": "rectangle",
            "plane": "XY",
            "nodes": [
                {
                    "id": "result",
                    "kind": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": 2,
                    "height": 1,
                }
            ],
            "result_node_id": "result",
        }
    )


def test_compile_failure_releases_owned_runtime_and_returns_stable_diagnostic() -> None:
    events: list[str] = []

    class Runtime:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_args: object) -> None:
            events.append("exit")

        def rectangle(self, *_args: object) -> None:
            raise RuntimeError("native failure")

    def factory(*_args: object, **_kwargs: object) -> Runtime:
        return Runtime()

    with pytest.raises(PlanarConstructionCompileError) as caught:
        compile_planar_construction(_rectangle_ir(), model_factory=factory)

    assert events == ["enter", "exit"]
    assert caught.value.diagnostic.code == "planar-ir.invalid-primitive"
    assert caught.value.diagnostic.node_id == "result"
    assert caught.value.diagnostic.model_unchanged is True


def test_compile_rejects_non_ir_before_opening_runtime() -> None:
    opened = False

    def factory(*_args: object, **_kwargs: object) -> object:
        nonlocal opened
        opened = True
        return object()

    with pytest.raises(TypeError, match="PlanarConstructionIR"):
        compile_planar_construction(object(), model_factory=factory)  # type: ignore[arg-type]
    assert opened is False


def test_raw_boolean_failure_uses_materialization_diagnostic() -> None:
    construction = PlanarConstructionIR.from_dict(
        {
            "schema_version": 1,
            "name": "union",
            "plane": "XY",
            "nodes": [
                {
                    "id": "left",
                    "kind": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": 1,
                    "height": 1,
                },
                {
                    "id": "right",
                    "kind": "rectangle",
                    "x": 0.5,
                    "y": 0,
                    "width": 1,
                    "height": 1,
                },
                {"id": "result", "kind": "union", "operands": ["left", "right"]},
            ],
            "result_node_id": "result",
        }
    )

    class Cad:
        def rectangle(self, *_args: object) -> object:
            return object()

        def area(self, _surface: object) -> float:
            return 1.0

        def boundary(self, _surfaces: tuple[object, ...]) -> tuple[object, ...]:
            return ()

        def copy(self, surfaces: tuple[object, ...]) -> tuple[object, ...]:
            return surfaces

        def fuse(self, *_args: object) -> None:
            raise RuntimeError("native Boolean failure")

    with pytest.raises(PlanarConstructionCompileError) as caught:
        _evaluate(Cad(), construction)

    assert caught.value.diagnostic.code == "planar-ir.materialization-failed"
    assert caught.value.diagnostic.node_id == "result"


def test_equivalence_proof_rejects_curve_type_changes() -> None:
    source = _NativeFacts(1.0, (0.0, 0.0, 1.0, 1.0), 1, 0, (("line", 4),))
    recipe = _NativeFacts(1.0, (0.0, 0.0, 1.0, 1.0), 1, 0, (("arc", 4),))

    with pytest.raises(PlanarConstructionCompileError) as caught:
        _require_equivalent(source, recipe, "result")

    assert caught.value.diagnostic.code == "planar-ir.equivalence-failed"
    assert "curve types" in caught.value.diagnostic.message.casefold()


def test_unsupported_boundary_uses_stable_diagnostic(monkeypatch) -> None:
    class Runtime:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def planar_boundary_loops(self, _surfaces: object) -> None:
            raise GeometryError("unsupported boundary spline")

    monkeypatch.setattr(
        "fem.application.planar_construction._evaluate",
        lambda _cad, _construction: ((object(),), ("result",)),
    )

    with pytest.raises(PlanarConstructionCompileError) as caught:
        compile_planar_construction(
            _rectangle_ir(),
            model_factory=lambda *_args, **_kwargs: Runtime(),
        )

    assert caught.value.diagnostic.code == "planar-ir.unsupported-boundary"
