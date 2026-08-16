from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from typing import Any

import pytest

from fem.core import Mesh3D
from fem import geometry
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing


_SWEEP_FRAMES = {
    "discrete": "DiscreteTrihedron",
    "corrected_frenet": "CorrectedFrenet",
    "frenet": "Frenet",
    "fixed": "Fixed",
    "constant_normal": "ConstantNormal",
    "darboux": "Darboux",
}
_LOFT_OPTION_MAPPINGS = (
    *(("continuity", value, value) for value in ("C0", "G1", "C1", "G2", "C2", "C3", "CN")),
    ("parametrization", "chord_length", "ChordLength"),
    ("parametrization", "centripetal", "Centripetal"),
    ("parametrization", "iso_parametric", "IsoParametric"),
)


def _open_path(
    cad: geometry.GeometryModel,
    *,
    length: float = 1.0,
) -> geometry.WireRef:
    start = cad.point(0.0, 0.0, 0.0)
    end = cad.point(0.0, 0.0, length)
    curve = cad.line(start, end)
    return cad.wire((cad.orient(curve),), closed=False)


def _closed_square_wire(
    cad: geometry.GeometryModel,
    *,
    z: float,
    half_width: float,
) -> tuple[geometry.WireRef, tuple[geometry.EntityRef, ...]]:
    points = (
        cad.point(-half_width, -half_width, z),
        cad.point(half_width, -half_width, z),
        cad.point(half_width, half_width, z),
        cad.point(-half_width, half_width, z),
    )
    curves = tuple(
        cad.line(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )
    wire = cad.wire(tuple(cad.orient(curve) for curve in curves), closed=True)
    return wire, curves


def _open_section_wire(
    cad: geometry.GeometryModel,
    *,
    z: float,
    half_width: float,
) -> tuple[geometry.WireRef, geometry.EntityRef]:
    start = cad.point(-half_width, 0.0, z)
    end = cad.point(half_width, 0.0, z)
    curve = cad.line(start, end)
    return cad.wire((cad.orient(curve),), closed=False), curve


def _closed_circular_path(cad: geometry.GeometryModel) -> geometry.WireRef:
    center = cad.point(0.0, 0.0, 0.0)
    points = (
        cad.point(2.0, 0.0, 0.0),
        cad.point(0.0, 2.0, 0.0),
        cad.point(-2.0, 0.0, 0.0),
        cad.point(0.0, -2.0, 0.0),
    )
    arcs = tuple(
        cad.circular_arc(points[index], center, points[(index + 1) % 4])
        for index in range(4)
    )
    return cad.wire(tuple(cad.orient(arc) for arc in arcs), closed=True)


def test_generalized_feature_result_and_loft_result_contracts() -> None:
    owner = object()
    first_curve = geometry.EntityRef(1, 1, owner, object())
    second_curve = geometry.EntityRef(1, 2, owner, object())
    revolved_surface = geometry.EntityRef(2, 1, owner, object())
    revolved_side = geometry.EntityRef(1, 3, owner, object())

    revolution = geometry.FeatureResult(
        "revolve",
        (first_curve,),
        (first_curve, revolved_surface, revolved_side),
        (revolved_surface,),
        (),
        (revolved_side,),
    )

    assert revolution.outputs[0] is first_curve
    assert revolution.ends == ()
    assert revolution.primary == (revolved_surface,)
    assert revolution.sides == (revolved_side,)

    first_section = geometry.WireRef(
        1,
        (geometry.OrientedCurveRef(first_curve),),
        True,
        owner,
        object(),
    )
    second_section = geometry.WireRef(
        2,
        (geometry.OrientedCurveRef(second_curve),),
        True,
        owner,
        object(),
    )
    volume = geometry.EntityRef(3, 1, owner, object())
    first_cap = geometry.EntityRef(2, 2, owner, object())
    side = geometry.EntityRef(2, 3, owner, object())
    topology = geometry.FeatureResult(
        "loft",
        (first_curve, second_curve),
        (first_cap, volume, side),
        (volume,),
        (first_cap,),
        (side,),
    )
    loft = geometry.LoftResult(topology, (first_section, second_section))

    assert loft.topology is topology
    assert loft.sections == (first_section, second_section)
    assert loft.operation == "loft"
    assert loft.primary == (volume,)
    assert loft.ends == (first_cap,)
    assert loft.sides == (side,)
    assert loft.of_dimension(2) == (first_cap, side)
    assert not hasattr(loft, "__dict__")
    with pytest.raises(FrozenInstanceError):
        loft.topology = topology  # type: ignore[misc]


@pytest.mark.parametrize(
    ("angle", "end_y_sign"),
    ((0.5 * math.pi, 1.0), (-0.5 * math.pi, -1.0)),
)
def test_real_partial_and_negative_revolve_classify_topology(
    real_gmsh: Any,
    angle: float,
    end_y_sign: float,
) -> None:
    with geometry.model(
        f"advanced-revolve-{end_y_sign:+.0f}",
        dimension=3,
    ) as cad:
        start = cad.point(1.0, 0.0, 0.0)
        end = cad.point(2.0, 0.0, 0.0)
        source = cad.line(start, end)

        result = cad.revolve(
            (source,),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            angle,
        )

        assert isinstance(result, geometry.FeatureResult)
        assert result.operation == "revolve"
        assert result.inputs == (source,)
        assert tuple(entity.dimension for entity in result.primary) == (2,)
        assert len(result.ends) == 1
        assert all(entity.dimension == 1 for entity in (*result.ends, *result.sides))
        assert source not in result.outputs
        assert set(cad.boundary(result.primary, combined=False)) == {
            source,
            *result.ends,
            *result.sides,
        }
        end_center = cad.center_of_mass(result.ends[0])
        assert end_center[1] * end_y_sign > 0.0


def test_real_full_revolve_retains_source_echo_without_terminal_end(
    real_gmsh: Any,
) -> None:
    with geometry.model("advanced-revolve-full", dimension=3) as cad:
        start = cad.point(1.0, 0.0, 0.0)
        end = cad.point(2.0, 0.0, 0.0)
        source = cad.line(start, end)

        result = cad.revolve(
            (source,),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            2.0 * math.pi,
        )

        assert result.outputs[0] == source
        assert result.outputs.count(source) == 1
        assert result.ends == ()
        assert tuple(entity.dimension for entity in result.primary) == (2,)
        assert set(cad.boundary(result.primary, combined=False)) == set(result.sides)
        assert cad.length(source) == pytest.approx(1.0)


def test_real_full_revolve_accepts_cancelling_periodic_curve_seam(
    real_gmsh: Any,
) -> None:
    with geometry.model("advanced-revolve-periodic-full", dimension=3) as cad:
        points = (
            cad.point(1.0, 0.0, -0.5),
            cad.point(2.0, 0.0, -0.5),
            cad.point(2.0, 0.0, 0.5),
            cad.point(1.0, 0.0, 0.5),
        )
        periodic = cad.spline((*points, points[0]))

        result = cad.revolve(
            (periodic,),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            2.0 * math.pi,
        )

        assert result.outputs.count(periodic) == 1
        assert result.ends == ()
        assert tuple(entity.dimension for entity in result.primary) == (2,)
        assert periodic in cad.boundary(result.primary, combined=False)
        assert periodic not in cad.boundary(result.primary, combined=True)


def test_real_revolve_validation_precedes_native_topology_creation(
    real_gmsh: Any,
) -> None:
    with geometry.model("advanced-revolve-validation", dimension=3) as cad:
        first = cad.point(1.0, 0.0, 0.0)
        second = cad.point(2.0, 0.0, 0.0)
        source = cad.line(first, second)
        before = (cad.entities(2), cad.entities(3))
        invalid_calls = (
            lambda: cad.revolve((), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
            lambda: cad.revolve(
                (source, source),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            ),
            lambda: cad.revolve(
                (first,), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0
            ),
            lambda: cad.revolve(
                (source,), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
            ),
            lambda: cad.revolve(
                (source,), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
            ),
            lambda: cad.revolve(
                (source,),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                2.0 * math.pi + 0.1,
            ),
            lambda: cad.revolve(
                (source,), 0.0, 0.0, 0.0, math.nan, 0.0, 1.0, 1.0
            ),
        )
        for invalid_call in invalid_calls:
            with pytest.raises(ValueError):
                invalid_call()
            assert (cad.entities(2), cad.entities(3)) == before

    with geometry.model("advanced-revolve-validation-2d", dimension=2) as cad:
        first = cad.point(1.0, 0.0)
        second = cad.point(2.0, 0.0)
        source = cad.line(first, second)
        before = cad.entities(2)
        with pytest.raises(ValueError, match="global Z axis"):
            cad.revolve(
                (source,), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, math.pi
            )
        assert cad.entities(2) == before


def test_real_two_dimensional_revolve_preserves_global_xy_plane(
    real_gmsh: Any,
) -> None:
    with geometry.model("advanced-revolve-2d", dimension=2) as cad:
        first = cad.point(1.0, 0.0)
        second = cad.point(2.0, 0.0)
        source = cad.line(first, second)

        result = cad.revolve(
            (source,),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.5 * math.pi,
        )

        assert tuple(entity.dimension for entity in result.primary) == (2,)
        assert all(
            abs(cad.bounding_box(entity)[2]) <= 1.0e-6
            and abs(cad.bounding_box(entity)[5]) <= 1.0e-6
            for entity in result.outputs
        )


def test_real_revolve_native_failure_is_contextual_and_fail_closed(
    real_gmsh: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_revolve(*args: Any, **kwargs: Any) -> list[tuple[int, int]]:
        raise RuntimeError("injected revolve failure")

    monkeypatch.setattr(real_gmsh.model.occ, "revolve", fail_revolve)
    with geometry.model("advanced-revolve-failure", dimension=3) as cad:
        first = cad.point(1.0, 0.0, 0.0)
        second = cad.point(2.0, 0.0, 0.0)
        source = cad.line(first, second)

        with pytest.raises(geometry.GeometryError, match="native OCC revolve") as caught:
            cad.revolve(
                (source,), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, math.pi
            )

        assert isinstance(caught.value.__cause__, RuntimeError)
        with pytest.raises(geometry.StaleEntityError, match="stale"):
            cad.length(source)


def test_real_open_sweep_classifies_both_ends_and_lateral_topology(
    real_gmsh: Any,
) -> None:
    with geometry.model("advanced-sweep-open", dimension=3) as cad:
        profile = cad.disk(0.0, 0.0, 0.25)
        path = _open_path(cad, length=1.5)

        result = cad.sweep((profile,), path)

        assert isinstance(result, geometry.FeatureResult)
        assert result.operation == "sweep"
        assert result.inputs == (profile,)
        assert tuple(entity.dimension for entity in result.primary) == (3,)
        assert len(result.ends) == 2
        assert len(result.sides) == 1
        assert all(entity.dimension == 2 for entity in (*result.ends, *result.sides))
        assert set(cad.boundary(result.primary, combined=False)) == {
            *result.ends,
            *result.sides,
        }
        assert cad.area(profile) == pytest.approx(math.pi * 0.25**2)


def test_real_symmetric_cube_sweep_classifies_terminal_topologically(
    real_gmsh: Any,
) -> None:
    with geometry.model("advanced-sweep-symmetric-cube", dimension=3) as cad:
        profile = cad.rectangle(-0.5, -0.5, 1.0, 1.0)
        path = _open_path(cad)

        result = cad.sweep((profile,), path)

        assert tuple(entity.dimension for entity in result.primary) == (3,)
        assert len(result.ends) == 2
        assert len(result.sides) == 4
        assert sorted(
            cad.center_of_mass(entity)[2] for entity in result.ends
        ) == pytest.approx([0.0, 1.0])
        for entity in (*result.ends, *result.sides):
            assert cad.area(entity) == pytest.approx(1.0)
        assert set(cad.boundary(result.primary, combined=False)) == {
            *result.ends,
            *result.sides,
        }


@pytest.mark.parametrize(
    ("frame", "backend_name"),
    tuple(_SWEEP_FRAMES.items()),
)
def test_real_sweep_forwards_typed_frame_name(
    real_gmsh: Any,
    monkeypatch: pytest.MonkeyPatch,
    frame: str,
    backend_name: str,
) -> None:
    original_add_pipe = real_gmsh.model.occ.addPipe
    forwarded: list[str] = []

    def recording_add_pipe(
        dim_tags: list[tuple[int, int]],
        wire_tag: int,
        trihedron: str = "",
    ) -> list[tuple[int, int]]:
        forwarded.append(trihedron)
        return original_add_pipe(dim_tags, wire_tag, trihedron)

    monkeypatch.setattr(real_gmsh.model.occ, "addPipe", recording_add_pipe)
    with geometry.model(f"advanced-sweep-frame-{frame}", dimension=3) as cad:
        profile = cad.disk(0.0, 0.0, 0.2)
        path = _open_path(cad)

        result = cad.sweep((profile,), path, frame=frame)  # type: ignore[arg-type]

        assert result.primary
        assert forwarded == [backend_name]


def test_real_closed_sweep_has_no_duplicate_terminal_seam(
    real_gmsh: Any,
) -> None:
    with geometry.model("advanced-sweep-closed", dimension=3) as cad:
        profile_start = cad.point(2.0, 0.0, -0.25)
        profile_end = cad.point(2.0, 0.0, 0.25)
        profile = cad.line(profile_start, profile_end)
        path = _closed_circular_path(cad)

        result = cad.sweep((profile,), path)

        assert result.primary
        assert all(entity.dimension == 2 for entity in result.primary)
        assert len(result.ends) <= 1
        assert len(set(result.ends)) == len(result.ends)
        assert set(cad.boundary(result.primary, combined=False)) == {
            *result.ends,
            *result.sides,
        }


def test_real_sweep_preflight_rejects_invalid_reused_and_stale_paths(
    real_gmsh: Any,
) -> None:
    with geometry.model("advanced-sweep-validation", dimension=3) as cad:
        profile = cad.disk(0.0, 0.0, 0.2)
        path = _open_path(cad)
        before = cad.entities(3)
        with pytest.raises(ValueError, match="unsupported sweep frame"):
            cad.sweep((profile,), path, frame="parallel")  # type: ignore[arg-type]
        assert cad.entities(3) == before

        reused_start = cad.point(0.0, 0.0, 2.0)
        reused_end = cad.point(0.0, 0.0, 3.0)
        reused_curve = cad.line(reused_start, reused_end)
        reused_path = cad.wire((cad.orient(reused_curve),), closed=False)
        with pytest.raises(ValueError, match="reuse a profile"):
            cad.sweep((reused_curve,), reused_path)
        assert cad.entities(3) == before

        stale_path = _open_path(cad, length=0.5)
        cad.translate((stale_path.curves[0].curve,), 0.25, 0.0, 0.0)
        with pytest.raises(geometry.StaleEntityError, match="stale wire"):
            cad.sweep((profile,), stale_path)
        assert cad.entities(3) == before


def test_real_sweep_rejects_foreign_path_before_native_mutation(
    real_gmsh: Any,
) -> None:
    with geometry.model("advanced-sweep-outer", dimension=3) as outer:
        foreign_path = _open_path(outer)
        with geometry.model("advanced-sweep-inner", dimension=3) as inner:
            profile = inner.disk(0.0, 0.0, 0.2)
            before = inner.entities(3)
            with pytest.raises(geometry.EntityOwnershipError, match="another"):
                inner.sweep((profile,), foreign_path)
            assert inner.entities(3) == before


def test_real_sweep_native_failure_is_contextual_and_fail_closed(
    real_gmsh: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_pipe(*args: Any, **kwargs: Any) -> list[tuple[int, int]]:
        raise RuntimeError("injected sweep failure")

    monkeypatch.setattr(real_gmsh.model.occ, "addPipe", fail_pipe)
    with geometry.model("advanced-sweep-failure", dimension=3) as cad:
        profile = cad.disk(0.0, 0.0, 0.2)
        path = _open_path(cad)

        with pytest.raises(geometry.GeometryError, match="native OCC sweep") as caught:
            cad.sweep((profile,), path)

        assert isinstance(caught.value.__cause__, RuntimeError)
        with pytest.raises(geometry.StaleEntityError, match="stale"):
            cad.area(profile)


def test_real_solid_loft_preserves_grouped_section_history(
    real_gmsh: Any,
) -> None:
    with geometry.model("advanced-loft-solid", dimension=3) as cad:
        first, first_curves = _closed_square_wire(
            cad,
            z=0.0,
            half_width=0.5,
        )
        second, second_curves = _closed_square_wire(
            cad,
            z=1.0,
            half_width=0.3,
        )

        result = cad.loft((first, second))

        assert isinstance(result, geometry.LoftResult)
        assert result.operation == "loft"
        assert result.sections == (first, second)
        assert result.inputs == (*first_curves, *second_curves)
        assert tuple(entity.dimension for entity in result.primary) == (3,)
        assert result.primary == result.of_dimension(3)
        assert result.ends == ()
        assert result.sides == ()
        assert cad.boundary(result.primary, combined=False)
        assert all(cad.length(curve) > 0.0 for curve in result.inputs)


@pytest.mark.parametrize("ruled", (False, True))
def test_real_open_surface_loft_supports_smooth_and_ruled_modes(
    real_gmsh: Any,
    ruled: bool,
) -> None:
    with geometry.model(f"advanced-loft-surface-{ruled}", dimension=3) as cad:
        first, first_curve = _open_section_wire(
            cad,
            z=0.0,
            half_width=0.5,
        )
        second, second_curve = _open_section_wire(
            cad,
            z=1.0,
            half_width=0.25,
        )

        result = cad.loft((first, second), solid=False, ruled=ruled)

        assert result.sections == (first, second)
        assert result.inputs == (first_curve, second_curve)
        assert tuple(entity.dimension for entity in result.primary) == (2,)
        assert result.primary == result.of_dimension(2)
        assert result.ends == ()
        assert result.sides == ()
        assert all(cad.area(surface) > 0.0 for surface in result.primary)


def test_real_surface_loft_supports_typed_smoothing_options(
    real_gmsh: Any,
) -> None:
    with geometry.model("advanced-loft-smoothing", dimension=3) as cad:
        first, _ = _open_section_wire(cad, z=0.0, half_width=0.5)
        second, _ = _open_section_wire(cad, z=1.0, half_width=0.25)
        third, _ = _open_section_wire(cad, z=2.0, half_width=0.4)

        result = cad.loft(
            (first, second, third),
            solid=False,
            max_degree=3,
            continuity="C1",
            parametrization="chord_length",
            smoothing=True,
        )

        assert result.primary
        assert all(entity.dimension == 2 for entity in result.primary)


@pytest.mark.parametrize(
    ("option_name", "option_value", "backend_value"),
    _LOFT_OPTION_MAPPINGS,
)
def test_real_loft_forwards_typed_option_names(
    real_gmsh: Any,
    monkeypatch: pytest.MonkeyPatch,
    option_name: str,
    option_value: str,
    backend_value: str,
) -> None:
    original_add_sections = real_gmsh.model.occ.addThruSections
    forwarded: list[tuple[str, str]] = []

    def recording_add_sections(
        wire_tags: list[int],
        tag: int = -1,
        make_solid: bool = True,
        make_ruled: bool = False,
        max_degree: int = -1,
        continuity: str = "",
        parametrization: str = "",
        smoothing: bool = False,
    ) -> list[tuple[int, int]]:
        forwarded.append((continuity, parametrization))
        return original_add_sections(
            wire_tags,
            tag,
            make_solid,
            make_ruled,
            max_degree,
            continuity,
            parametrization,
            smoothing,
        )

    monkeypatch.setattr(
        real_gmsh.model.occ,
        "addThruSections",
        recording_add_sections,
    )
    with geometry.model(
        f"advanced-loft-option-{option_name}-{option_value}",
        dimension=3,
    ) as cad:
        first, _ = _open_section_wire(cad, z=0.0, half_width=0.5)
        second, _ = _open_section_wire(cad, z=1.0, half_width=0.25)
        kwargs = {option_name: option_value}
        result = cad.loft(
            (first, second),
            solid=False,
            **kwargs,  # type: ignore[arg-type]
        )

        assert result.primary
        expected = (
            (backend_value, "")
            if option_name == "continuity"
            else ("", backend_value)
        )
        assert forwarded == [expected]


def test_real_loft_validation_precedes_native_topology_creation(
    real_gmsh: Any,
) -> None:
    with geometry.model("advanced-loft-validation", dimension=3) as cad:
        closed, _ = _closed_square_wire(cad, z=0.0, half_width=0.5)
        open_section, _ = _open_section_wire(cad, z=1.0, half_width=0.25)
        before = (cad.entities(2), cad.entities(3))

        invalid_calls = (
            lambda: cad.loft((closed,)),
            lambda: cad.loft((closed, closed)),
            lambda: cad.loft((closed, open_section), solid=False),
            lambda: cad.loft((open_section, open_section), solid=True),
            lambda: cad.loft((closed, open_section), solid=1),  # type: ignore[arg-type]
            lambda: cad.loft((closed, open_section), ruled=1),  # type: ignore[arg-type]
            lambda: cad.loft((closed, open_section), max_degree=0),
            lambda: cad.loft(
                (closed, open_section),
                continuity="C4",  # type: ignore[arg-type]
            ),
            lambda: cad.loft(
                (closed, open_section),
                parametrization="uniform",  # type: ignore[arg-type]
            ),
            lambda: cad.loft(
                (closed, open_section),
                smoothing=1,  # type: ignore[arg-type]
            ),
        )
        for invalid_call in invalid_calls:
            with pytest.raises((TypeError, ValueError)):
                invalid_call()
            assert (cad.entities(2), cad.entities(3)) == before


def test_real_two_dimensional_loft_is_rejected_before_native_call(
    real_gmsh: Any,
) -> None:
    with geometry.model("advanced-loft-2d", dimension=2) as cad:
        first, _ = _open_section_wire(cad, z=0.0, half_width=0.5)
        second_start = cad.point(-0.25, 1.0)
        second_end = cad.point(0.25, 1.0)
        second_curve = cad.line(second_start, second_end)
        second = cad.wire((cad.orient(second_curve),), closed=False)
        before = cad.entities(2)

        with pytest.raises(ValueError, match="three-dimensional"):
            cad.loft((first, second), solid=False)

        assert cad.entities(2) == before


def test_real_loft_malformed_native_output_fails_closed(
    real_gmsh: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with geometry.model("advanced-loft-malformed", dimension=3) as cad:
        first, first_curve = _open_section_wire(
            cad,
            z=0.0,
            half_width=0.5,
        )
        second, _ = _open_section_wire(cad, z=1.0, half_width=0.25)

        def malformed_sections(*args: Any, **kwargs: Any) -> list[tuple[int, int]]:
            return [(1, first_curve.tag)]

        monkeypatch.setattr(
            real_gmsh.model.occ,
            "addThruSections",
            malformed_sections,
        )
        with pytest.raises(geometry.GeometryError, match="unexpected dimension"):
            cad.loft((first, second), solid=False)

        with pytest.raises(geometry.StaleEntityError, match="stale"):
            cad.length(first_curve)


def test_real_loft_missing_native_output_fails_closed(
    real_gmsh: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with geometry.model("advanced-loft-missing", dimension=3) as cad:
        first, first_curve = _open_section_wire(cad, z=0.0, half_width=0.5)
        second, _ = _open_section_wire(cad, z=1.0, half_width=0.25)

        monkeypatch.setattr(
            real_gmsh.model.occ,
            "addThruSections",
            lambda *args, **kwargs: [(2, 999999)],
        )
        with pytest.raises(geometry.GeometryError, match="missing entity"):
            cad.loft((first, second), solid=False)

        with pytest.raises(geometry.StaleEntityError, match="stale"):
            cad.length(first_curve)


def test_real_loft_aliased_native_output_fails_closed(
    real_gmsh: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with geometry.model("advanced-loft-aliased", dimension=3) as cad:
        unrelated = cad.rectangle(2.0, 2.0, 1.0, 1.0)
        first, first_curve = _open_section_wire(cad, z=0.0, half_width=0.5)
        second, _ = _open_section_wire(cad, z=1.0, half_width=0.25)

        monkeypatch.setattr(
            real_gmsh.model.occ,
            "addThruSections",
            lambda *args, **kwargs: [(2, unrelated.tag)],
        )
        with pytest.raises(geometry.GeometryError, match="existing entity"):
            cad.loft((first, second), solid=False)

        with pytest.raises(geometry.StaleEntityError, match="stale"):
            cad.length(first_curve)


@pytest.mark.parametrize("operation", ("wire", "revolve", "sweep", "loft"))
def test_mesher_binding_seals_advanced_geometry_mutation(
    real_gmsh: Any,
    operation: str,
) -> None:
    with geometry.model(f"advanced-mesher-seal-{operation}", dimension=3) as cad:
        if operation == "wire":
            first = cad.point(0.0, 0.0, 0.0)
            second = cad.point(0.0, 0.0, 1.0)
            curve = cad.line(first, second)

            def mutate() -> object:
                return cad.wire((cad.orient(curve),), closed=False)

        elif operation == "revolve":
            first = cad.point(1.0, 0.0, 0.0)
            second = cad.point(2.0, 0.0, 0.0)
            source = cad.line(first, second)

            def mutate() -> object:
                return cad.revolve(
                    (source,), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, math.pi
                )

        elif operation == "sweep":
            profile = cad.disk(0.0, 0.0, 0.2)
            path = _open_path(cad)

            def mutate() -> object:
                return cad.sweep((profile,), path)

        else:
            first_section, _ = _closed_square_wire(
                cad, z=0.0, half_width=0.5
            )
            second_section, _ = _closed_square_wire(
                cad, z=1.0, half_width=0.25
            )

            def mutate() -> object:
                return cad.loft((first_section, second_section))

        before = tuple(real_gmsh.model.occ.getEntities())
        gmsh_meshing.Mesher(cad)

        with pytest.raises(geometry.GeometryStateError, match="does not permit"):
            mutate()

        assert tuple(real_gmsh.model.occ.getEntities()) == before


@pytest.mark.parametrize("feature", ("revolve", "sweep", "loft"))
def test_real_advanced_volume_is_meshable_and_importable(
    real_gmsh: Any,
    feature: str,
) -> None:
    with geometry.model(f"advanced-mesh-{feature}", dimension=3) as cad:
        if feature == "revolve":
            profile = cad.rectangle(0.5, 0.0, 0.5, 1.0)
            result = cad.revolve(
                (profile,),
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                2.0 * math.pi,
            )
        elif feature == "sweep":
            profile = cad.disk(0.0, 0.0, 0.25)
            result = cad.sweep((profile,), _open_path(cad, length=1.0))
        else:
            first, _ = _closed_square_wire(cad, z=0.0, half_width=0.5)
            second, _ = _closed_square_wire(cad, z=1.0, half_width=0.3)
            result = cad.loft((first, second))

        assert result.primary
        assert all(entity.dimension == 3 for entity in result.primary)
        native_mesh = gmsh_meshing.Mesher(cad).generate(
            gmsh_meshing.MeshSpec(size=0.3)
        )
        mesh = gmsh_io.read(native_mesh)

    assert isinstance(mesh, Mesh3D)
    assert mesh.num_nodes > 0
    assert mesh.num_elements > 0
    assert {element.type for element in mesh.elements} == {"Tet4"}


@pytest.mark.parametrize("feature", ("fillet", "chamfer"))
def test_real_edge_treated_volume_is_meshable_and_importable(
    real_gmsh: Any,
    feature: str,
) -> None:
    with geometry.model(f"advanced-mesh-{feature}", dimension=3) as cad:
        source = cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        surface = cad.boundary((source,), combined=False)[0]
        curve = cad.boundary((surface,), combined=False)[0]
        if feature == "fillet":
            result = cad.fillet((source,), (curve,), (0.1,))
        else:
            result = cad.chamfer(
                (source,),
                (curve,),
                (surface,),
                (0.1,),
            )

        assert result.primary
        assert all(entity.dimension == 3 for entity in result.primary)
        assert result.ends == ()
        assert result.sides == ()
        native_mesh = gmsh_meshing.Mesher(cad).generate(
            gmsh_meshing.MeshSpec(size=0.25)
        )
        mesh = gmsh_io.read(native_mesh)

    assert isinstance(mesh, Mesh3D)
    assert mesh.num_nodes > 0
    assert mesh.num_elements > 0
    assert {element.type for element in mesh.elements} == {"Tet4"}


@pytest.mark.parametrize("feature", ("fillet", "chamfer"))
def test_real_edge_treatment_preserve_mode_supports_two_value_law(
    real_gmsh: Any,
    feature: str,
) -> None:
    with geometry.model(f"advanced-preserve-{feature}", dimension=3) as cad:
        source = cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        surface = cad.boundary((source,), combined=False)[0]
        curve = cad.boundary((surface,), combined=False)[0]
        original_closure = (
            source,
            surface,
            curve,
            *cad.boundary((curve,), combined=False),
        )
        if feature == "fillet":
            result = cad.fillet(
                (source,),
                (curve,),
                (0.05, 0.08),
                remove_volumes=False,
            )
        else:
            result = cad.chamfer(
                (source,),
                (curve,),
                (surface,),
                (0.05, 0.08),
                remove_volumes=False,
            )

        assert len(result.primary) == 1
        assert result.primary[0] != source
        assert (result.primary[0].dimension, result.primary[0].tag) != (
            source.dimension,
            source.tag,
        )
        for original in original_closure:
            cad.bounding_box(original)


@pytest.mark.parametrize("feature", ("fillet", "chamfer"))
def test_real_edge_treatment_supports_multiple_volumes_and_per_target_values(
    real_gmsh: Any,
    feature: str,
) -> None:
    with geometry.model(f"advanced-multiple-{feature}", dimension=3) as cad:
        volumes = (
            cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            cad.box(2.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        )
        surfaces = tuple(
            cad.boundary((volume,), combined=False)[0] for volume in volumes
        )
        curves = tuple(
            cad.boundary((surface,), combined=False)[0] for surface in surfaces
        )
        if feature == "fillet":
            result = cad.fillet(volumes, curves, (0.05, 0.08))
        else:
            result = cad.chamfer(
                volumes,
                curves,
                surfaces,
                (0.05, 0.08, 0.06, 0.09),
            )

        assert len(result.primary) == 2
        assert all(entity.dimension == 3 for entity in result.primary)


@pytest.mark.parametrize("feature", ("fillet", "chamfer"))
def test_real_edge_treatment_invalidates_dependent_wire_only(
    real_gmsh: Any,
    feature: str,
) -> None:
    with geometry.model(f"advanced-wire-invalidation-{feature}", dimension=3) as cad:
        source = cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        surface = cad.boundary((source,), combined=False)[0]
        curve = cad.boundary((surface,), combined=False)[0]
        dependent = cad.wire((cad.orient(curve),), closed=False)

        profile = cad.disk(3.0, 0.0, 0.2)
        path_start = cad.point(3.0, 0.0, 0.0)
        path_end = cad.point(3.0, 0.0, 1.0)
        path_curve = cad.line(path_start, path_end)
        unrelated = cad.wire((cad.orient(path_curve),), closed=False)

        if feature == "fillet":
            cad.fillet((source,), (curve,), (0.1,))
        else:
            cad.chamfer((source,), (curve,), (surface,), (0.1,))

        with pytest.raises(geometry.StaleEntityError, match="stale wire"):
            cad.sweep((profile,), dependent)
        swept = cad.sweep((profile,), unrelated)
        assert swept.primary
