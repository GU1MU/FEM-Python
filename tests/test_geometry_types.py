from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from itertools import permutations
from pathlib import Path
import subprocess
import sys
from typing import Any, get_args, get_type_hints

import pytest

from fem import geometry
from fem.geometry import errors as geometry_errors
from fem.geometry import types as geometry_types
from fem.mesh import gmsh as gmsh_meshing


ERROR_NAMES = (
    "EntityOwnershipError",
    "GeometryError",
    "GeometryStateError",
    "StaleEntityError",
)
TYPE_NAMES = (
    "BooleanResult",
    "CurveLoopRef",
    "EntityRef",
    "FeatureResult",
    "LoftContinuity",
    "LoftParametrization",
    "LoftResult",
    "OrientedCurveRef",
    "StrictBodyBooleanPreview",
    "SurfaceTessellation",
    "SweepFrame",
    "WireRef",
)
VALUE_OBJECT_NAMES = (
    "BooleanResult",
    "CurveLoopRef",
    "EntityRef",
    "FeatureResult",
    "LoftResult",
    "OrientedCurveRef",
    "StrictBodyBooleanPreview",
    "SurfaceTessellation",
    "WireRef",
)
GEOMETRY_PUBLIC_API = [
    "BASE_GEOMETRY_TYPES",
    "BodyOverlapError",
    "BodyRelation",
    "BooleanBodyContext",
    "BooleanResult",
    "BooleanGeometry",
    "BooleanLineageEntity",
    "BooleanLineageProof",
    "BooleanLineageResolutionError",
    "BooleanLineageMapping",
    "BoxGeometry",
    "CircleFrame",
    "CurveLoopRef",
    "CylinderGeometry",
    "DiskGeometry",
    "EntityOwnershipError",
    "EntityKind",
    "EntityRef",
    "ExtrudedGeometry",
    "ExtrusionSourceResolutionError",
    "ExtrusionSourceSelection",
    "FeatureResult",
    "GeometryError",
    "GeometryModel",
    "GeometryStateError",
    "LoftContinuity",
    "LoftParametrization",
    "LoftResult",
    "LogicalEntityRef",
    "MovedGeometry",
    "MultiBodyGeometry",
    "NATIVE_GEOMETRY_TYPES",
    "NativeGeometry",
    "OrientedCurveRef",
    "PRIMITIVE_GEOMETRY_TYPES",
    "PlateWithHoleGeometry",
    "PrimitiveGeometry",
    "RectangleFrame",
    "RectangleGeometry",
    "RotatedGeometry",
    "SKETCH_CONTOUR_TYPES",
    "STRICT_SKETCH_CURVE_TYPES",
    "SketchArc",
    "SketchCircle",
    "SketchContour",
    "SketchCurve",
    "SketchDiagnostic",
    "SketchGeometry",
    "SketchLine",
    "SketchPlane",
    "SketchPoint",
    "SketchProfile",
    "SketchProfileAnalysis",
    "SketchRectangle",
    "SolidBody",
    "StrictBodyBooleanPreview",
    "SurfaceTessellation",
    "StaleEntityError",
    "SweepFrame",
    "TargetRadiusResolutionError",
    "WireRef",
    "WireGeometry",
    "WireMember",
    "WirePoint",
    "add_solid_body",
    "axis_aligned_rectangle",
    "analyze_sketch_profiles",
    "analyze_body_relations",
    "delete_solid_body",
    "expand_sketch_recipe",
    "geometry_dimension",
    "historical_recipe_ids",
    "capture_boolean_operand_evidence",
    "install_proven_body_boolean",
    "logical_ref_sort_key",
    "legacy_sketch_to_strict",
    "legacy_sketches_to_strict",
    "materialize_multi_body",
    "model",
    "next_body_id",
    "next_boolean_feature_id",
    "provisional_body_boolean",
    "rename_solid_body",
    "retired_recipe_ids",
    "recipe_characteristic_size",
    "resolve_legacy_hole_target",
    "resolve_extrusion_source_faces",
    "resolve_target_radius",
    "resolve_solid_boolean_lineage",
    "require_meshable_body_relations",
    "supports_structured_hexahedron",
    "transform_solid_body",
    "transformed_circle",
    "undo_solid_body_feature",
]


class _IndexValue:
    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"IndexValue({self.value})"


def _entity(dimension: int, tag: int, owner: object) -> geometry_types.EntityRef:
    return geometry_types.EntityRef(dimension, tag, owner, object())


def _valid_loft() -> tuple[
    geometry_types.LoftResult,
    geometry_types.WireRef,
    geometry_types.WireRef,
]:
    owner = object()
    first_curve = _entity(1, 1, owner)
    second_curve = _entity(1, 2, owner)
    first_section = geometry_types.WireRef(
        1,
        (geometry_types.OrientedCurveRef(first_curve),),
        True,
        owner,
        object(),
    )
    second_section = geometry_types.WireRef(
        2,
        (geometry_types.OrientedCurveRef(second_curve),),
        True,
        owner,
        object(),
    )
    volume = _entity(3, 3, owner)
    first_cap = _entity(2, 4, owner)
    side = _entity(2, 5, owner)
    topology = geometry_types.FeatureResult(
        "loft",
        (first_curve, second_curve),
        (first_cap, volume, side),
        (volume,),
        (first_cap,),
        (side,),
    )
    return (
        geometry_types.LoftResult(
            topology,
            (section for section in (first_section, second_section)),
        ),
        first_section,
        second_section,
    )


def test_public_error_and_type_modules_have_exact_exports() -> None:
    assert geometry_errors.__all__ == list(ERROR_NAMES)
    assert geometry_types.__all__ == list(TYPE_NAMES)
    assert geometry.__all__ == GEOMETRY_PUBLIC_API
    assert all(hasattr(geometry, name) for name in GEOMETRY_PUBLIC_API)


def test_facade_submodules_share_one_public_identity() -> None:
    for name in ERROR_NAMES:
        canonical = getattr(geometry_errors, name)
        assert getattr(geometry, name) is canonical

    for name in TYPE_NAMES:
        canonical = getattr(geometry_types, name)
        assert getattr(geometry, name) is canonical

    cad = geometry.model("type-contract", dimension=2)
    assert isinstance(cad, geometry.GeometryModel)


@pytest.mark.parametrize(
    "module_order",
    tuple(permutations(("fem.geometry.types", "fem.geometry", "fem.mesh.gmsh"))),
)
def test_supported_fresh_process_import_orders_are_cycle_free_and_lazy(
    module_order: tuple[str, ...],
) -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import builtins
import importlib
import sys

sys.path.insert(0, {str(src_dir)!r})
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "gmsh" or name.startswith("gmsh."):
        raise AssertionError("external gmsh was imported eagerly")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
for module_name in {module_order!r}:
    importlib.import_module(module_name)
for module_name in (
    "fem.geometry.errors",
    "fem.geometry._validation",
    "fem.geometry._gmsh.constants",
    "fem.geometry._gmsh.predicates",
):
    importlib.import_module(module_name)

from fem import geometry
from fem.geometry import errors, types

for name in {ERROR_NAMES!r}:
    assert getattr(geometry, name) is getattr(errors, name)
for name in {TYPE_NAMES!r}:
    assert getattr(geometry, name) is getattr(types, name)
assert "gmsh" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_public_classes_have_canonical_modules_and_error_inheritance() -> None:
    for name in ERROR_NAMES:
        assert getattr(geometry_errors, name).__module__ == "fem.geometry.errors"
    for name in VALUE_OBJECT_NAMES:
        assert getattr(geometry_types, name).__module__ == "fem.geometry.types"

    assert geometry_errors.GeometryError.__bases__ == (RuntimeError,)
    for error_type in (
        geometry_errors.EntityOwnershipError,
        geometry_errors.GeometryStateError,
        geometry_errors.StaleEntityError,
    ):
        assert error_type.__bases__ == (geometry_errors.GeometryError,)


def test_public_literal_aliases_retain_their_exact_values() -> None:
    assert get_args(geometry_types.SweepFrame) == (
        "discrete",
        "corrected_frenet",
        "frenet",
        "fixed",
        "constant_normal",
        "darboux",
    )
    assert get_args(geometry_types.LoftContinuity) == (
        "C0",
        "G1",
        "C1",
        "G2",
        "C2",
        "C3",
        "CN",
    )
    assert get_args(geometry_types.LoftParametrization) == (
        "chord_length",
        "centripetal",
        "iso_parametric",
    )


def test_mesh_annotations_resolve_to_the_canonical_geometry_types() -> None:
    transfinite_hints = get_type_hints(gmsh_meshing.Mesher.transfinite_curve)
    extrusion_hints = get_type_hints(gmsh_meshing.Mesher.structured_extrude)

    assert transfinite_hints["curve"] is geometry_types.EntityRef
    assert extrusion_hints["return"] is geometry_types.FeatureResult


def test_entity_ref_is_frozen_slotted_and_does_not_store_index_normalization() -> None:
    owner = object()
    tag = _IndexValue(7)
    entity = geometry_types.EntityRef(1, tag, owner, object())  # type: ignore[arg-type]

    assert entity.dimension == 1
    assert entity.tag is tag
    assert not hasattr(entity, "__dict__")
    assert "object" not in repr(entity)
    with pytest.raises(FrozenInstanceError):
        entity.tag = 8  # type: ignore[misc]


@pytest.mark.parametrize("dimension", (True, -1, 4, 1.0, "1", _IndexValue(1)))
def test_entity_ref_rejects_invalid_dimensions(dimension: Any) -> None:
    with pytest.raises(ValueError, match="entity dimension"):
        geometry_types.EntityRef(dimension, 1, object(), object())


@pytest.mark.parametrize("tag", (False, 0, -1, 1.0, "1"))
def test_entity_ref_rejects_invalid_tags(tag: Any) -> None:
    with pytest.raises(ValueError, match="entity tag must be a positive integer"):
        geometry_types.EntityRef(1, tag, object(), object())


def test_oriented_curve_ref_validates_curve_dimension_and_boolean_orientation() -> None:
    owner = object()
    curve = _entity(1, 1, owner)
    oriented = geometry_types.OrientedCurveRef(curve, True)

    assert oriented.curve is curve
    assert oriented.reversed is True
    assert not hasattr(oriented, "__dict__")
    with pytest.raises(FrozenInstanceError):
        oriented.reversed = False  # type: ignore[misc]
    with pytest.raises(ValueError, match="dimension-one EntityRef"):
        geometry_types.OrientedCurveRef(_entity(2, 2, owner))
    with pytest.raises(ValueError, match="dimension-one EntityRef"):
        geometry_types.OrientedCurveRef(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reversed must be a boolean"):
        geometry_types.OrientedCurveRef(curve, 1)  # type: ignore[arg-type]


def test_curve_loop_ref_materializes_and_validates_oriented_curves() -> None:
    owner = object()
    oriented = geometry_types.OrientedCurveRef(_entity(1, 1, owner))
    loop = geometry_types.CurveLoopRef(
        2,
        (item for item in (oriented,)),
        owner,
        object(),
    )

    assert loop.curves == (oriented,)
    assert not hasattr(loop, "__dict__")
    with pytest.raises(ValueError, match="at least one oriented curve"):
        geometry_types.CurveLoopRef(2, (), owner, object())
    with pytest.raises(TypeError, match="iterable of OrientedCurveRef"):
        geometry_types.CurveLoopRef(2, None, owner, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="only OrientedCurveRef"):
        geometry_types.CurveLoopRef(2, (object(),), owner, object())  # type: ignore[arg-type]


def test_wire_ref_materializes_curves_and_requires_a_boolean_closed_flag() -> None:
    owner = object()
    oriented = geometry_types.OrientedCurveRef(_entity(1, 1, owner), True)
    wire = geometry_types.WireRef(
        3,
        (item for item in (oriented,)),
        False,
        owner,
        object(),
    )

    assert wire.curves == (oriented,)
    assert wire.closed is False
    assert not hasattr(wire, "__dict__")
    with pytest.raises(ValueError, match="at least one oriented curve"):
        geometry_types.WireRef(3, (), False, owner, object())
    with pytest.raises(TypeError, match="only OrientedCurveRef"):
        geometry_types.WireRef(3, (object(),), False, owner, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="closed must be a boolean"):
        geometry_types.WireRef(3, (oriented,), 0, owner, object())  # type: ignore[arg-type]


def test_boolean_result_has_no_implicit_coercion_or_ownership_policy() -> None:
    first_owner = object()
    second_owner = object()
    curve = _entity(1, 1, first_owner)
    foreign_surface = _entity(2, 2, second_owner)
    outputs = [curve, curve, foreign_surface]
    input_map = [[curve], [foreign_surface]]

    result = geometry_types.BooleanResult(outputs, input_map)  # type: ignore[arg-type]

    assert result.outputs is outputs
    assert result.input_map is input_map
    assert result.of_dimension(1) == (curve, curve)
    assert result.of_dimension(2) == (foreign_surface,)
    assert not hasattr(result, "__dict__")
    with pytest.raises(ValueError, match="entity dimension"):
        result.of_dimension(True)


def test_feature_result_materializes_fields_and_preserves_output_repeats() -> None:
    owner = object()
    source = _entity(1, 1, owner)
    first_side = _entity(1, 2, owner)
    primary = _entity(2, 3, owner)
    end = _entity(1, 4, owner)
    second_side = _entity(1, 5, owner)
    result = geometry_types.FeatureResult(
        "extrude",
        [source],  # type: ignore[arg-type]
        [first_side, primary, end, first_side, second_side],  # type: ignore[arg-type]
        [primary],  # type: ignore[arg-type]
        [end],  # type: ignore[arg-type]
        [first_side, second_side],  # type: ignore[arg-type]
    )

    assert result.inputs == (source,)
    assert result.outputs == (
        first_side,
        primary,
        end,
        first_side,
        second_side,
    )
    assert result.primary == (primary,)
    assert result.ends == (end,)
    assert result.sides == (first_side, second_side)
    assert result.of_dimension(1) == (
        first_side,
        end,
        first_side,
        second_side,
    )
    assert result.of_dimension(2) == (primary,)
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.operation = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("case", "error_type", "message"),
    (
        ("empty_operation", ValueError, "operation"),
        ("duplicate_inputs", ValueError, "inputs must be duplicate-free"),
        ("foreign_owner", geometry_errors.EntityOwnershipError, "one geometry model"),
        ("mixed_inputs", ValueError, "one common dimension"),
        ("duplicate_semantic", ValueError, "semantic fields must be duplicate-free"),
        ("overlap", ValueError, "must be disjoint"),
        ("incomplete_partition", ValueError, "must partition"),
        ("wrong_order", ValueError, "first-seen output order"),
        ("generated_dimension", ValueError, "generated feature outputs"),
        ("primary_below_inputs", ValueError, "must not be below"),
        ("wrong_boundary_dimension", ValueError, "boundary dimension"),
        ("same_dimension_ends", ValueError, "cannot report ends or sides"),
    ),
)
def test_feature_result_rejects_malformed_topology_partitions(
    case: str,
    error_type: type[Exception],
    message: str,
) -> None:
    owner = object()
    source = _entity(1, 1, owner)
    primary = _entity(2, 2, owner)
    end = _entity(1, 3, owner)
    first_side = _entity(1, 4, owner)
    second_side = _entity(1, 5, owner)
    kwargs: dict[str, Any] = {
        "operation": "extrude",
        "inputs": (source,),
        "outputs": (first_side, primary, end, second_side),
        "primary": (primary,),
        "ends": (end,),
        "sides": (first_side, second_side),
    }
    if case == "empty_operation":
        kwargs["operation"] = ""
    elif case == "duplicate_inputs":
        kwargs["inputs"] = (source, source)
    elif case == "foreign_owner":
        foreign = _entity(1, 6, object())
        kwargs["outputs"] = (foreign, primary, end)
        kwargs["sides"] = (foreign,)
    elif case == "mixed_inputs":
        kwargs["inputs"] = (source, _entity(0, 6, owner))
    elif case == "duplicate_semantic":
        kwargs["sides"] = (first_side, first_side, second_side)
    elif case == "overlap":
        kwargs["sides"] = (first_side, end, second_side)
    elif case == "incomplete_partition":
        kwargs["sides"] = (first_side,)
    elif case == "wrong_order":
        kwargs["outputs"] = (second_side, primary, end, first_side)
    elif case == "generated_dimension":
        kwargs["outputs"] = (
            _entity(0, 7, owner),
            first_side,
            primary,
            end,
            second_side,
        )
    elif case == "primary_below_inputs":
        surface = _entity(2, 7, owner)
        replacement_curve = _entity(1, 8, owner)
        kwargs.update(
            inputs=(surface,),
            outputs=(replacement_curve,),
            primary=(replacement_curve,),
            ends=(),
            sides=(),
        )
    elif case == "wrong_boundary_dimension":
        wrong_end = _entity(2, 7, owner)
        kwargs["outputs"] = (first_side, primary, wrong_end, second_side)
        kwargs["ends"] = (wrong_end,)
    else:
        surface = _entity(2, 7, owner)
        replacement = _entity(2, 8, owner)
        boundary = _entity(1, 9, owner)
        kwargs.update(
            inputs=(surface,),
            outputs=(replacement, boundary),
            primary=(replacement,),
            ends=(boundary,),
            sides=(),
        )

    with pytest.raises(error_type, match=message):
        geometry_types.FeatureResult(**kwargs)


def test_same_dimensional_feature_accepts_historical_source_references() -> None:
    owner = object()
    historical_source = _entity(2, 1, owner)
    replacement = _entity(2, 2, owner)

    result = geometry_types.FeatureResult(
        "fillet",
        (historical_source,),
        (replacement,),
        (replacement,),
    )

    assert result.inputs == (historical_source,)
    assert result.outputs == (replacement,)
    assert result.primary == (replacement,)
    assert result.ends == ()
    assert result.sides == ()


def test_loft_result_preserves_grouped_sections_and_delegates_topology() -> None:
    loft, first_section, second_section = _valid_loft()

    assert loft.sections == (first_section, second_section)
    assert loft.operation == "loft"
    assert loft.inputs == loft.topology.inputs
    assert loft.outputs == loft.topology.outputs
    assert loft.primary == loft.topology.primary
    assert loft.ends == loft.topology.ends
    assert loft.sides == loft.topology.sides
    assert loft.of_dimension(2) == (*loft.ends, *loft.sides)
    assert not hasattr(loft, "__dict__")
    with pytest.raises(FrozenInstanceError):
        loft.topology = loft.topology  # type: ignore[misc]


def test_loft_result_validates_operation_sections_owner_and_flattened_order() -> None:
    loft, first_section, second_section = _valid_loft()
    topology = loft.topology

    with pytest.raises(TypeError, match="topology must be a FeatureResult"):
        geometry_types.LoftResult(object(), ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="operation 'loft'"):
        geometry_types.LoftResult(
            replace(topology, operation="sweep"),
            (first_section, second_section),
        )
    with pytest.raises(ValueError, match="at least two WireRef"):
        geometry_types.LoftResult(topology, (first_section,))
    with pytest.raises(TypeError, match="only WireRef"):
        geometry_types.LoftResult(
            topology,
            (first_section, object()),  # type: ignore[arg-type]
        )

    foreign_owner = object()
    foreign_curve = _entity(1, 20, foreign_owner)
    foreign_section = geometry_types.WireRef(
        20,
        (geometry_types.OrientedCurveRef(foreign_curve),),
        True,
        foreign_owner,
        object(),
    )
    with pytest.raises(geometry_errors.EntityOwnershipError, match="one geometry model"):
        geometry_types.LoftResult(topology, (first_section, foreign_section))
    with pytest.raises(ValueError, match="grouped section-curve order"):
        geometry_types.LoftResult(
            replace(topology, inputs=tuple(reversed(topology.inputs))),
            (first_section, second_section),
        )
