from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from fem.geometry import (
    CurveLoopRef,
    EntityOwnershipError,
    EntityRef,
    GeometryError,
    GeometryStateError,
    OrientedCurveRef,
    StaleEntityError,
    WireRef,
)
from fem.geometry._gmsh.control_dependencies import _ControlDependencyLedger
from fem.geometry._gmsh.reference_registry import (
    _EntityRegistry,
    _ReferenceRegistry,
)


def _curve(
    registry: _ReferenceRegistry,
    tag: int,
) -> tuple[EntityRef, tuple[OrientedCurveRef, ...], frozenset[tuple[int, int]]]:
    curve = registry.wrap_entity((1, tag))
    registry.wrap_entity((0, 2 * tag - 1))
    registry.wrap_entity((0, 2 * tag))
    oriented = (OrientedCurveRef(curve),)
    dependencies = frozenset(
        {(1, tag), (0, 2 * tag - 1), (0, 2 * tag)}
    )
    return curve, oriented, dependencies


def _normalize_loop(
    registry: _ReferenceRegistry,
    loop: CurveLoopRef,
    dependencies: frozenset[tuple[int, int]],
) -> tuple[CurveLoopRef, ...]:
    return registry.normalize_curve_loops(
        (loop,),
        operation="plane_surface",
        dependency_resolver=lambda _reference: dependencies,
    )


def _normalize_wire(
    registry: _ReferenceRegistry,
    wire: WireRef,
    dependencies: frozenset[tuple[int, int]],
) -> tuple[WireRef, ...]:
    return registry.normalize_wires(
        (wire,),
        operation="loft",
        dependency_resolver=lambda _reference: dependencies,
    )


def test_entity_registry_reuses_live_token_and_refreshes_reused_tag() -> None:
    registry = _EntityRegistry("identity")

    first = registry.wrap((2, 7))
    repeated = registry.wrap([2, 7])

    assert repeated == first
    assert repeated._owner_token is first._owner_token
    assert repeated._entity_token is first._entity_token

    registry.invalidate(((2, 7),))
    with pytest.raises(StaleEntityError, match=r"identity.*stale entity \(2, 7\)"):
        registry.normalize((first,), operation="remove")

    reused = registry.wrap((2, 7))
    assert reused._owner_token is first._owner_token
    assert reused._entity_token is not first._entity_token
    assert registry.normalize((reused,), operation="remove") == (reused,)


def test_entity_registry_preserves_validation_order_and_messages() -> None:
    registry = _EntityRegistry("validation")
    entity = registry.wrap((1, 3))
    foreign = _EntityRegistry("foreign").wrap((1, 3))

    with pytest.raises(TypeError, match="edges entities must be iterable"):
        registry.normalize(None, operation="edges")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="edges requires at least one entity"):
        registry.normalize((), operation="edges")
    with pytest.raises(TypeError, match="edges requires EntityRef values"):
        registry.normalize((object(),), operation="edges")  # type: ignore[arg-type]
    with pytest.raises(EntityOwnershipError, match="owned by another"):
        registry.normalize((foreign,), operation="edges")

    registry.invalidate(((1, 3),))
    with pytest.raises(StaleEntityError, match=r"stale entity \(1, 3\)"):
        registry.normalize((entity, entity), operation="edges")

    current = registry.wrap((1, 3))
    with pytest.raises(ValueError, match="entity inputs must be duplicate-free"):
        registry.normalize((current, current), operation="edges")


def test_entity_registry_rejects_malformed_native_entity_keys() -> None:
    registry = _EntityRegistry("malformed")

    with pytest.raises(GeometryError, match="invalid Gmsh entity reference"):
        registry.wrap((1, 2, 3))
    with pytest.raises(ValueError, match="entity dimension"):
        registry.wrap((4, 1))
    with pytest.raises(ValueError, match="entity tag"):
        registry.wrap((1, 0))


def test_loop_and_wire_with_the_same_tag_have_independent_identities() -> None:
    registry = _ReferenceRegistry("namespaces")
    _entity, oriented, dependencies = _curve(registry, 1)

    loop = registry.register_curve_loop(8, oriented, dependencies)
    wire = registry.register_wire(8, oriented, False, dependencies)

    assert loop.tag == wire.tag == 8
    assert loop._loop_token is not wire._wire_token
    assert _normalize_loop(registry, loop, dependencies) == (loop,)
    assert _normalize_wire(registry, wire, dependencies) == (wire,)


def test_invalid_loop_identity_clears_only_the_loop_namespace() -> None:
    registry = _ReferenceRegistry("invalid-loop")
    _entity, oriented, dependencies = _curve(registry, 1)
    loop = registry.register_curve_loop(3, oriented, dependencies)
    wire = registry.register_wire(3, oriented, False, dependencies)

    with pytest.raises(GeometryError, match="invalid loop tag"):
        registry.register_curve_loop(0, oriented, dependencies)

    with pytest.raises(StaleEntityError, match="stale curve loop 3"):
        _normalize_loop(registry, loop, dependencies)
    assert _normalize_wire(registry, wire, dependencies) == (wire,)


def test_duplicate_loop_identity_clears_only_the_loop_namespace() -> None:
    registry = _ReferenceRegistry("duplicate-loop")
    _entity, oriented, dependencies = _curve(registry, 1)
    loop = registry.register_curve_loop(3, oriented, dependencies)
    wire = registry.register_wire(3, oriented, False, dependencies)

    with pytest.raises(GeometryError, match="duplicate loop tag 3"):
        registry.register_curve_loop(3, oriented, dependencies)

    with pytest.raises(StaleEntityError, match="stale curve loop 3"):
        _normalize_loop(registry, loop, dependencies)
    assert _normalize_wire(registry, wire, dependencies) == (wire,)


def test_invalid_wire_identity_clears_only_the_wire_namespace() -> None:
    registry = _ReferenceRegistry("invalid-wire")
    _entity, oriented, dependencies = _curve(registry, 1)
    loop = registry.register_curve_loop(3, oriented, dependencies)
    wire = registry.register_wire(3, oriented, False, dependencies)

    with pytest.raises(GeometryError, match="invalid wire tag"):
        registry.register_wire(-1, oriented, False, dependencies)

    assert _normalize_loop(registry, loop, dependencies) == (loop,)
    with pytest.raises(StaleEntityError, match="stale wire 3"):
        _normalize_wire(registry, wire, dependencies)


def test_duplicate_wire_identity_clears_only_the_wire_namespace() -> None:
    registry = _ReferenceRegistry("duplicate-wire")
    _entity, oriented, dependencies = _curve(registry, 1)
    loop = registry.register_curve_loop(3, oriented, dependencies)
    wire = registry.register_wire(3, oriented, False, dependencies)

    with pytest.raises(GeometryError, match="duplicate wire tag 3"):
        registry.register_wire(3, oriented, False, dependencies)

    assert _normalize_loop(registry, loop, dependencies) == (loop,)
    with pytest.raises(StaleEntityError, match="stale wire 3"):
        _normalize_wire(registry, wire, dependencies)


def test_topology_validation_covers_type_owner_token_and_duplicates() -> None:
    registry = _ReferenceRegistry("topology")
    _entity, oriented, dependencies = _curve(registry, 1)
    loop = registry.register_curve_loop(4, oriented, dependencies)
    wire = registry.register_wire(5, oriented, False, dependencies)

    with pytest.raises(TypeError, match="curve loops must be iterable"):
        registry.normalize_curve_loops(  # type: ignore[arg-type]
            None,
            operation="surface",
            dependency_resolver=lambda _reference: dependencies,
        )
    with pytest.raises(ValueError, match="requires at least one wire"):
        registry.normalize_wires(
            (),
            operation="loft",
            dependency_resolver=lambda _reference: dependencies,
        )
    with pytest.raises(TypeError, match="requires CurveLoopRef values"):
        registry.normalize_curve_loops(  # type: ignore[arg-type]
            (object(),),
            operation="surface",
            dependency_resolver=lambda _reference: dependencies,
        )

    foreign = _ReferenceRegistry("foreign")
    _foreign_entity, foreign_oriented, foreign_dependencies = _curve(foreign, 2)
    foreign_wire = foreign.register_wire(
        5,
        foreign_oriented,
        False,
        foreign_dependencies,
    )
    with pytest.raises(EntityOwnershipError, match="wire owned by another"):
        registry.normalize_wires(
            (foreign_wire,),
            operation="loft",
            dependency_resolver=lambda _reference: foreign_dependencies,
        )

    with pytest.raises(ValueError, match="curve loops must be duplicate-free"):
        registry.normalize_curve_loops(
            (loop, loop),
            operation="surface",
            dependency_resolver=lambda _reference: dependencies,
        )

    registry.clear_wires()
    with pytest.raises(StaleEntityError, match="stale wire 5"):
        _normalize_wire(registry, wire, dependencies)


def test_topology_validation_detects_dependency_drift_and_shared_members() -> None:
    registry = _ReferenceRegistry("dependencies")
    _first, first_oriented, first_dependencies = _curve(registry, 1)
    _second, second_oriented, second_dependencies = _curve(registry, 2)
    first_loop = registry.register_curve_loop(
        10,
        first_oriented,
        first_dependencies,
    )
    shared_dependencies = frozenset(
        {*second_dependencies, min(first_dependencies)}
    )
    second_loop = registry.register_curve_loop(
        11,
        second_oriented,
        shared_dependencies,
    )

    with pytest.raises(StaleEntityError, match="stale curve loop 10"):
        registry.normalize_curve_loops(
            (first_loop,),
            operation="surface",
            dependency_resolver=lambda _reference: {(1, 999)},
        )

    dependencies_by_tag = {
        first_loop.tag: first_dependencies,
        second_loop.tag: shared_dependencies,
    }
    with pytest.raises(ValueError, match="must not share member curves"):
        registry.normalize_curve_loops(
            (first_loop, second_loop),
            operation="surface",
            dependency_resolver=lambda reference: dependencies_by_tag[reference.tag],
        )


def test_entity_invalidation_also_invalidates_intersecting_topology() -> None:
    registry = _ReferenceRegistry("entity-invalidation")
    first, first_oriented, first_dependencies = _curve(registry, 1)
    second, second_oriented, second_dependencies = _curve(registry, 2)
    first_loop = registry.register_curve_loop(7, first_oriented, first_dependencies)
    second_wire = registry.register_wire(
        8,
        second_oriented,
        False,
        second_dependencies,
    )

    registry.invalidate_entities(((1, 1),))

    with pytest.raises(StaleEntityError, match=r"stale entity \(1, 1\)"):
        registry.normalize_entities((first,), operation="query")
    with pytest.raises(StaleEntityError, match="stale curve loop 7"):
        _normalize_loop(registry, first_loop, first_dependencies)
    assert registry.normalize_entities((second,), operation="query") == (second,)
    assert _normalize_wire(registry, second_wire, second_dependencies) == (
        second_wire,
    )


def test_topology_only_invalidation_keeps_entity_identity_live() -> None:
    registry = _ReferenceRegistry("topology-invalidation")
    curve, oriented, dependencies = _curve(registry, 1)
    loop = registry.register_curve_loop(7, oriented, dependencies)
    wire = registry.register_wire(8, oriented, False, dependencies)

    registry.invalidate_topology(((0, 1),))

    assert registry.normalize_entities((curve,), operation="query") == (curve,)
    with pytest.raises(StaleEntityError, match="stale curve loop 7"):
        _normalize_loop(registry, loop, dependencies)
    with pytest.raises(StaleEntityError, match="stale wire 8"):
        _normalize_wire(registry, wire, dependencies)


def test_full_registry_clear_invalidates_all_typed_geometry_references() -> None:
    registry = _ReferenceRegistry("clear")
    curve, oriented, dependencies = _curve(registry, 1)
    loop = registry.register_curve_loop(7, oriented, dependencies)
    wire = registry.register_wire(8, oriented, False, dependencies)

    registry.clear()

    with pytest.raises(StaleEntityError, match=r"stale entity \(1, 1\)"):
        registry.normalize_entities((curve,), operation="query")
    with pytest.raises(StaleEntityError, match="stale curve loop 7"):
        _normalize_loop(registry, loop, dependencies)
    with pytest.raises(StaleEntityError, match="stale wire 8"):
        _normalize_wire(registry, wire, dependencies)


def test_control_ledger_rejects_removal_of_the_lowest_conflicting_key() -> None:
    ledger = _ControlDependencyLedger("guards")
    ledger.register({(2, 9), (1, 4)}, transform_unsafe=False)

    with pytest.raises(
        GeometryStateError,
        match=r"guards.*remove.*topology \(1, 4\)",
    ):
        ledger.check_removal("remove", {(2, 9), (1, 4), (0, 1)})

    ledger.check_removal("remove", {(0, 1)})


def test_control_ledger_guards_only_transform_unsafe_subset() -> None:
    ledger = _ControlDependencyLedger("transforms")
    ledger.register({(1, 1)}, transform_unsafe=False)
    ledger.register({(2, 2)}, transform_unsafe=True)

    ledger.check_transform("translate", {(1, 1)})
    with pytest.raises(
        GeometryStateError,
        match=r"translate.*discard.*\(2, 2\)",
    ):
        ledger.check_transform("translate", {(1, 1), (2, 2)})


def test_control_ledger_marks_raw_scope_unknown_only_with_dependencies() -> None:
    ledger = _ControlDependencyLedger("raw")

    ledger.mark_unknown_after_raw_access()
    assert not ledger.scope_unknown

    ledger.register({(1, 1)}, transform_unsafe=False)
    ledger.mark_unknown_after_raw_access()
    assert ledger.scope_unknown
    with pytest.raises(GeometryStateError, match="dependencies unknown"):
        ledger.check_scope_known("remove")


def test_control_ledger_unknown_mutation_is_unconditional_but_empty_removal_is_safe(
) -> None:
    ledger = _ControlDependencyLedger("unknown")

    ledger.mark_unknown_after_unknown_mutation()

    assert ledger.scope_unknown
    ledger.check_removal("remove", ())
    with pytest.raises(GeometryStateError, match="dependencies unknown"):
        ledger.check_removal("remove", {(1, 1)})
    with pytest.raises(GeometryStateError, match="dependencies unknown"):
        ledger.check_transform("rotate", ())


def test_control_ledger_structured_snapshot_restores_only_unknown_flag() -> None:
    ledger = _ControlDependencyLedger("structured")
    snapshot = ledger.snapshot_unknown_scope()

    ledger.mark_unknown_after_unknown_mutation()
    ledger.register({(2, 6)}, transform_unsafe=True)
    ledger.restore_unknown_scope(snapshot)

    assert not ledger.scope_unknown
    assert ledger.has_dependencies
    assert ledger.has_transform_unsafe_dependencies
    with pytest.raises(GeometryStateError, match=r"topology \(2, 6\)"):
        ledger.check_removal("remove", {(2, 6)})


def test_reference_clearing_does_not_clear_committed_control_guards() -> None:
    registry = _ReferenceRegistry("separate")
    ledger = _ControlDependencyLedger("separate")
    entity = registry.wrap_entity((2, 4))
    ledger.register({(2, 4)}, transform_unsafe=False)

    registry.clear()

    with pytest.raises(StaleEntityError):
        registry.normalize_entities((entity,), operation="query")
    with pytest.raises(GeometryStateError, match=r"topology \(2, 4\)"):
        ledger.check_removal("remove", {(2, 4)})

    ledger.clear()
    assert not ledger.has_dependencies
    assert not ledger.scope_unknown
    ledger.check_removal("remove", {(2, 4)})


def test_private_registry_modules_do_not_import_external_gmsh() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import builtins
import sys

sys.path.insert(0, {str(src_dir)!r})
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "gmsh" or name.startswith("gmsh."):
        raise AssertionError("external gmsh was imported eagerly")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from fem.geometry._gmsh.control_dependencies import _ControlDependencyLedger
from fem.geometry._gmsh.reference_registry import _ReferenceRegistry

assert _ControlDependencyLedger("lazy") is not None
assert _ReferenceRegistry("lazy") is not None
assert "gmsh" not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
