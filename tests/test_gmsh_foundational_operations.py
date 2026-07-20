from __future__ import annotations

from typing import Any

import pytest

from fem.core import Mesh2D, Mesh3D
from fem import geometry
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing


@pytest.fixture
def real_gmsh() -> Any:
    import gmsh

    owns_session = not gmsh.isInitialized()
    if owns_session:
        gmsh.initialize()
    terminal = gmsh.option.getNumber("General.Terminal")
    try:
        gmsh.clear()
        gmsh.option.setNumber("General.Terminal", 0.0)
        yield gmsh
    finally:
        gmsh.clear()
        gmsh.option.setNumber("General.Terminal", terminal)
        if owns_session:
            gmsh.finalize()


def _generate(
    cad: geometry.GeometryModel,
    *,
    size: float,
) -> gmsh_meshing.GmshMeshRef:
    return gmsh_meshing.Mesher(cad).generate(
        gmsh_meshing.MeshSpec(size=size)
    )


def _entity_mesh_nodes(
    gmsh: Any,
    dimension: int,
    tags: tuple[int, ...],
) -> set[int]:
    node_tags: set[int] = set()
    for tag in tags:
        raw_tags, _, _ = gmsh.model.mesh.getNodes(
            dimension,
            tag,
            includeBoundary=True,
            returnParametricCoord=False,
        )
        node_tags.update(int(value) for value in raw_tags)
    return node_tags


def _entity_element_nodes(
    gmsh: Any,
    dimension: int,
    tags: tuple[int, ...],
) -> set[int]:
    node_tags: set[int] = set()
    for tag in tags:
        _, _, raw_node_blocks = gmsh.model.mesh.getElements(dimension, tag)
        for raw_nodes in raw_node_blocks:
            node_tags.update(int(value) for value in raw_nodes)
    return node_tags


def _entity_measure(
    cad: geometry.GeometryModel,
    gmsh: Any,
    entity: geometry.EntityRef,
) -> float:
    if entity.dimension == 0:
        return 0.0
    if entity.dimension == 1:
        return cad.length(entity)
    if entity.dimension == 2:
        return cad.area(entity)
    return float(gmsh.model.occ.getMass(entity.dimension, entity.tag))


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    ("kind", "facade_dimension", "vector", "side_count"),
    [
        ("point", 2, (1.0, 0.0, 0.0), 0),
        ("curve", 2, (0.0, 1.0, 0.0), 2),
        ("surface", 3, (0.0, 0.0, 1.0), 4),
    ],
)
def test_real_extrusion_classifies_feature_topology_for_every_input_dimension(
    real_gmsh: Any,
    structured: bool,
    kind: str,
    facade_dimension: int,
    vector: tuple[float, float, float],
    side_count: int,
) -> None:
    with geometry.model(
        f"feature_{kind}_{'structured' if structured else 'pure'}",
        dimension=facade_dimension,
    ) as cad:
        if kind == "point":
            source = cad.point(0.0, 0.0, 0.0)
        elif kind == "curve":
            source = cad.line(
                cad.point(0.0, 0.0, 0.0),
                cad.point(1.0, 0.0, 0.0),
            )
        else:
            source = cad.rectangle(0.0, 0.0, 1.0, 1.0)

        if structured:
            result = gmsh_meshing.Mesher(cad).structured_extrude(
                [source],
                *vector,
                num_elements=(2,),
            )
        else:
            result = cad.extrude([source], *vector)

        assert isinstance(result, geometry.FeatureResult)
        assert result.operation == (
            "structured_extrude" if structured else "extrude"
        )
        assert result.inputs == (source,)
        assert len(result.primary) == 1
        assert result.primary[0].dimension == source.dimension + 1
        assert len(result.ends) == 1
        assert result.ends[0].dimension == source.dimension
        assert len(result.sides) == side_count
        assert set(result.primary + result.ends + result.sides) == set(
            result.outputs
        )


def test_real_extrusion_handles_holes_negative_vectors_and_shared_sides(
    real_gmsh: Any,
) -> None:
    with geometry.model("feature_hole_negative", dimension=3) as cad:
        plate = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        hole = cad.disk(1.0, 0.5, 0.2)
        source = cad.cut([plate], [hole]).of_dimension(2)[0]
        result = cad.extrude([source], 0.0, 0.0, -1.0)

        assert len(result.primary) == 1
        assert len(result.ends) == 1
        assert len(result.sides) == 5
        assert cad.center_of_mass(result.ends[0])[2] == pytest.approx(-1.0)

    with geometry.model("feature_shared_side", dimension=3) as cad:
        left = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        right = cad.rectangle(1.0, 0.0, 1.0, 1.0)
        sources = cad.fragment([left], [right]).of_dimension(2)
        assert len(sources) == 2

        result = cad.extrude(sources, 0.0, 0.0, 1.0)

        assert len(result.primary) == 2
        assert len(result.ends) == 2
        assert len(result.sides) == 7
        assert len(result.outputs) > len(set(result.outputs))


def test_real_structured_extrusion_handles_holed_surface_and_negative_vector(
    real_gmsh: Any,
) -> None:
    with geometry.model("feature_structured_hole_negative", dimension=3) as cad:
        plate = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        hole = cad.disk(1.0, 0.5, 0.2)
        source = cad.cut([plate], [hole]).of_dimension(2)[0]

        result = gmsh_meshing.Mesher(cad).structured_extrude(
            [source],
            0.0,
            0.0,
            -1.0,
            num_elements=(2,),
        )

        assert len(result.primary) == 1
        assert len(result.ends) == 1
        assert len(result.sides) == 5
        assert cad.center_of_mass(result.ends[0])[2] == pytest.approx(-1.0)


@pytest.mark.parametrize("structured", [False, True])
def test_real_extrusion_classifies_short_curve_far_from_origin(
    real_gmsh: Any,
    structured: bool,
) -> None:
    origin = 1.0e9
    length = 0.0625
    with geometry.model(
        f"feature_far_origin_{'structured' if structured else 'pure'}",
        dimension=2,
    ) as cad:
        source = cad.line(
            cad.point(origin, origin, 0.0),
            cad.point(origin + length, origin, 0.0),
        )
        if structured:
            result = gmsh_meshing.Mesher(cad).structured_extrude(
                [source],
                0.0,
                length,
                0.0,
                num_elements=(1,),
            )
        else:
            result = cad.extrude([source], 0.0, length, 0.0)

        assert len(result.primary) == 1
        assert len(result.ends) == 1
        assert len(result.sides) == 2
        end_center = cad.center_of_mass(result.ends[0])
        assert end_center[0] - origin == pytest.approx(0.5 * length)
        assert end_center[1] - origin == pytest.approx(length)


@pytest.mark.parametrize("structured", [False, True])
def test_real_extrusion_classifies_multiple_disjoint_inputs(
    real_gmsh: Any,
    structured: bool,
) -> None:
    with geometry.model(
        f"feature_disjoint_{'structured' if structured else 'pure'}",
        dimension=3,
    ) as cad:
        first = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        second = cad.rectangle(2.0, 0.0, 1.0, 1.0)
        if structured:
            result = gmsh_meshing.Mesher(cad).structured_extrude(
                [second, first],
                0.0,
                0.0,
                1.0,
                num_elements=(1,),
            )
        else:
            result = cad.extrude([second, first], 0.0, 0.0, 1.0)

        assert result.inputs == (second, first)
        assert len(result.primary) == 2
        assert len(result.ends) == 2
        assert len(result.sides) == 8
        assert len(result.outputs) == len(set(result.outputs))


def test_real_copy_batches_mixed_dimensions_and_restores_caller_order(
    real_gmsh: Any,
) -> None:
    with geometry.model("copy_mixed", dimension=3) as cad:
        isolated_point = cad.point(5.0, 0.0, 0.0)
        curve = cad.line(
            cad.point(0.0, 0.0, 0.0),
            cad.point(1.0, 0.0, 0.0),
        )
        surface = cad.rectangle(0.0, 2.0, 1.0, 1.0)
        volume = cad.box(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)
        sources = (surface, isolated_point, volume, curve)

        copies = cad.copy(sources)

        assert tuple(entity.dimension for entity in copies) == (2, 0, 3, 1)
        assert len(set(copies)) == len(sources)
        assert all(copy != source for source, copy in zip(sources, copies, strict=True))
        for source, copied in zip(sources, copies, strict=True):
            assert cad.bounding_box(copied) == pytest.approx(
                cad.bounding_box(source)
            )
            assert cad.center_of_mass(copied) == pytest.approx(
                cad.center_of_mass(source)
            )
            assert _entity_measure(cad, real_gmsh, copied) == pytest.approx(
                _entity_measure(cad, real_gmsh, source)
            )


def test_real_copy_reactivates_outer_nested_model(real_gmsh: Any) -> None:
    with geometry.model("copy_outer", dimension=2) as outer:
        source = outer.rectangle(0.0, 0.0, 1.0, 1.0)
        with geometry.model("copy_inner", dimension=3) as inner:
            inner.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        copied = outer.copy([source])[0]

        assert real_gmsh.model.getCurrent() == "copy_outer"
        assert copied.dimension == 2
        assert outer.area(copied) == pytest.approx(outer.area(source))


def test_real_mirror_and_negative_scale_preserve_top_level_references(
    real_gmsh: Any,
) -> None:
    with geometry.model("mirror_scale", dimension=2) as cad:
        surface = cad.rectangle(1.0, 0.0, 1.0, 1.0)
        old_boundary = cad.boundary([surface])
        unrelated = cad.rectangle(4.0, 0.0, 1.0, 1.0)

        assert cad.mirror([surface], 1.0, 0.0, 0.0, 0.0) == (surface,)
        assert cad.center_of_mass(surface) == pytest.approx((-1.5, 0.5, 0.0))
        assert cad.bounding_box(surface) == pytest.approx(
            (-2.0, 0.0, 0.0, -1.0, 1.0, 0.0),
            abs=2.0e-7,
        )
        with pytest.raises(geometry.StaleEntityError):
            cad.bounding_box(old_boundary[0])
        assert cad.area(unrelated) == pytest.approx(1.0)

        assert cad.scale([surface], 0.0, 0.0, 0.0, -1.0, 2.0, 1.0) == (
            surface,
        )
        assert cad.center_of_mass(surface) == pytest.approx((1.5, 1.0, 0.0))
        assert cad.bounding_box(surface) == pytest.approx(
            (1.0, 0.0, 0.0, 2.0, 2.0, 0.0),
            abs=3.0e-7,
        )
        assert cad.area(surface) == pytest.approx(2.0)


def test_real_mirror_and_negative_scale_support_curves_and_volumes(
    real_gmsh: Any,
) -> None:
    with geometry.model("mirror_scale_3d", dimension=3) as cad:
        curve = cad.line(
            cad.point(3.0, 0.0, 0.0),
            cad.point(4.0, 0.0, 0.0),
        )
        volume = cad.box(1.0, 0.0, 0.0, 1.0, 1.0, 1.0)

        assert cad.mirror([curve, volume], 1.0, 0.0, 0.0, 0.0) == (
            curve,
            volume,
        )
        assert cad.scale(
            [curve, volume],
            0.0,
            0.0,
            0.0,
            -1.0,
            2.0,
            0.5,
        ) == (curve, volume)
        assert cad.center_of_mass(curve) == pytest.approx((3.5, 0.0, 0.0))
        assert cad.center_of_mass(volume) == pytest.approx((1.5, 1.0, 0.25))
        assert cad.bounding_box(curve) == pytest.approx(
            (3.0, 0.0, 0.0, 4.0, 0.0, 0.0),
            abs=3.0e-7,
        )
        assert cad.bounding_box(volume) == pytest.approx(
            (1.0, 0.0, 0.0, 2.0, 2.0, 0.5),
            abs=3.0e-7,
        )
        assert cad.length(curve) == pytest.approx(1.0)
        assert _entity_measure(cad, real_gmsh, volume) == pytest.approx(1.0)

        native_mesh = _generate(cad, size=0.5)
        mesh = gmsh_io.read(native_mesh)

    assert isinstance(mesh, Mesh3D)
    assert mesh.num_elements > 0


def test_real_intersection_supports_mixed_dimensions_and_empty_results(
    real_gmsh: Any,
) -> None:
    with geometry.model("intersect_mixed", dimension=2) as cad:
        surface = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        crossing = cad.line(
            cad.point(-1.0, 0.5, 0.0),
            cad.point(3.0, 0.5, 0.0),
        )

        result = cad.intersect([surface], [crossing])

        assert len(result.outputs) == 1
        assert result.outputs[0].dimension == 1
        assert tuple(len(group) for group in result.input_map) == (0, 1)

    with geometry.model("intersect_empty", dimension=2) as cad:
        first = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        second = cad.rectangle(2.0, 0.0, 1.0, 1.0)

        empty = cad.intersect([first], [second])

        assert empty.outputs == ()
        assert empty.input_map == ((), ())


def _contained_dimension_pair(
    cad: geometry.GeometryModel,
    lower_dimension: int,
    higher_dimension: int,
) -> tuple[geometry.EntityRef, geometry.EntityRef]:
    embedded_z = 0.0 if higher_dimension == 2 else 0.5
    if higher_dimension == 1:
        higher = cad.line(
            cad.point(0.0, 0.5, embedded_z),
            cad.point(1.0, 0.5, embedded_z),
        )
    elif higher_dimension == 2:
        higher = cad.rectangle(0.0, 0.0, 1.0, 1.0, z=embedded_z)
    else:
        higher = cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    if lower_dimension == 0:
        lower = cad.point(0.5, 0.5, embedded_z)
    elif lower_dimension == 1:
        lower = cad.line(
            cad.point(0.2, 0.5, embedded_z),
            cad.point(0.8, 0.5, embedded_z),
        )
    else:
        lower = cad.rectangle(0.2, 0.2, 0.6, 0.6, z=embedded_z)
    return lower, higher


@pytest.mark.parametrize(
    ("lower_dimension", "higher_dimension"),
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
)
@pytest.mark.parametrize("lower_is_object", [False, True])
def test_real_intersection_supports_every_cross_dimensional_pair(
    real_gmsh: Any,
    lower_dimension: int,
    higher_dimension: int,
    lower_is_object: bool,
) -> None:
    with geometry.model(
        f"intersect_{lower_dimension}_{higher_dimension}_{lower_is_object}",
        dimension=higher_dimension,
    ) as cad:
        lower, higher = _contained_dimension_pair(
            cad,
            lower_dimension,
            higher_dimension,
        )
        objects, tools = (
            ((lower,), (higher,))
            if lower_is_object
            else ((higher,), (lower,))
        )

        result = cad.intersect(objects, tools)

        assert len(result.outputs) == 1
        assert result.outputs[0].dimension == lower_dimension
        assert tuple(len(group) for group in result.input_map) == (
            (1, 0) if lower_is_object else (0, 1)
        )


def test_real_point_curve_fragment_creates_one_shared_mesh_node(
    real_gmsh: Any,
) -> None:
    with geometry.model("fragment_point_curve", dimension=1) as cad:
        curve = cad.line(
            cad.point(0.0, 0.0, 0.0),
            cad.point(2.0, 0.0, 0.0),
        )
        point = cad.point(1.0, 0.0, 0.0)

        result = cad.fragment([curve], [point])
        members = result.of_dimension(1)
        embedded = result.input_map[1]
        assert len(members) == 2
        assert len(embedded) == 1
        assert embedded[0] in result.outputs
        point_tags = tuple(entity.tag for entity in embedded)
        member_tags = tuple(entity.tag for entity in members)

        native_mesh = _generate(cad, size=0.4)
        point_nodes = _entity_mesh_nodes(real_gmsh, 0, point_tags)
        member_nodes = _entity_element_nodes(real_gmsh, 1, member_tags)
        mesh = gmsh_io.read(native_mesh, line_element_type="Truss2")

    assert len(point_nodes) == 1
    assert point_nodes <= member_nodes
    assert isinstance(mesh, Mesh3D)


def test_real_crossing_curve_fragment_creates_one_shared_mesh_node(
    real_gmsh: Any,
) -> None:
    with geometry.model("fragment_crossing_curves", dimension=2) as cad:
        cad.rectangle(2.0, 2.0, 1.0, 1.0)
        horizontal = cad.line(
            cad.point(0.0, 0.5, 0.0),
            cad.point(1.0, 0.5, 0.0),
        )
        vertical = cad.line(
            cad.point(0.5, 0.0, 0.0),
            cad.point(0.5, 1.0, 0.0),
        )

        result = cad.fragment([horizontal], [vertical])

        assert len(result.of_dimension(1)) == 4
        assert tuple(len(group) for group in result.input_map) == (2, 2)
        horizontal_tags = tuple(entity.tag for entity in result.input_map[0])
        vertical_tags = tuple(entity.tag for entity in result.input_map[1])

        _generate(cad, size=0.2)
        horizontal_nodes = _entity_element_nodes(real_gmsh, 1, horizontal_tags)
        vertical_nodes = _entity_element_nodes(real_gmsh, 1, vertical_tags)

    assert len(horizontal_nodes & vertical_nodes) == 1


def test_real_curve_surface_fragment_retains_and_meshes_embedded_curve(
    real_gmsh: Any,
) -> None:
    with geometry.model("fragment_curve_surface", dimension=2) as cad:
        surface = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        curve = cad.line(
            cad.point(1.0, 0.2, 0.0),
            cad.point(1.0, 0.8, 0.0),
        )

        result = cad.fragment([surface], [curve])
        embedded = result.input_map[1]
        top = result.of_dimension(2)
        assert len(embedded) == 1
        assert embedded[0] in result.outputs
        assert len(top) == 1
        embedded_tags = tuple(entity.tag for entity in embedded)
        top_tags = tuple(entity.tag for entity in top)

        native_mesh = _generate(cad, size=0.2)
        embedded_nodes = _entity_mesh_nodes(real_gmsh, 1, embedded_tags)
        top_nodes = _entity_element_nodes(real_gmsh, 2, top_tags)
        mesh = gmsh_io.read(native_mesh)

    assert embedded_nodes
    assert embedded_nodes <= top_nodes
    assert isinstance(mesh, Mesh2D)


def test_real_surface_volume_fragment_creates_conformal_volume_partition(
    real_gmsh: Any,
) -> None:
    with geometry.model("fragment_surface_volume", dimension=3) as cad:
        volume = cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        surface = cad.rectangle(0.0, 0.0, 1.0, 1.0, z=0.5)

        result = cad.fragment([volume], [surface])
        embedded = result.input_map[1]
        top = result.of_dimension(3)
        assert len(embedded) == 1
        assert embedded[0] in result.outputs
        assert len(top) == 2
        embedded_tags = tuple(entity.tag for entity in embedded)
        top_tags = tuple(entity.tag for entity in top)

        native_mesh = _generate(cad, size=0.4)
        embedded_nodes = _entity_mesh_nodes(real_gmsh, 2, embedded_tags)
        top_nodes = _entity_element_nodes(real_gmsh, 3, top_tags)
        mesh = gmsh_io.read(native_mesh)

    assert embedded_nodes
    assert embedded_nodes <= top_nodes
    assert isinstance(mesh, Mesh3D)


def test_real_fragment_accepts_multiple_lower_dimensional_tools(
    real_gmsh: Any,
) -> None:
    with geometry.model("fragment_multiple_tools", dimension=2) as cad:
        surface = cad.rectangle(0.0, 0.0, 3.0, 1.0)
        first = cad.line(
            cad.point(1.0, 0.0, 0.0),
            cad.point(1.0, 1.0, 0.0),
        )
        second = cad.line(
            cad.point(2.0, 0.0, 0.0),
            cad.point(2.0, 1.0, 0.0),
        )

        result = cad.fragment([surface], [first, second])

        assert len(result.input_map) == 3
        assert len(result.of_dimension(2)) == 3
        assert result.input_map[1]
        assert result.input_map[2]
