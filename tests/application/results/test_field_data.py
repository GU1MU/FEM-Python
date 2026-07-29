from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fem.application.results import (
    FieldAssociation,
    FieldAvailability,
    FieldData,
    FieldDescriptor,
    FieldLocation,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    FieldState,
    PhysicalQuantity,
    ResultCatalog,
    ResultDiagnostic,
    ResultExportSnapshot,
    ResultFieldId,
    ResultMaterializationPatch,
    ResultMaterializationSnapshot,
    ResultSourceKey,
    ResultTopologyProjection,
    ResultVariable,
    ScalarFieldSelection,
    advance_materialization,
    build_initial_materialization,
    prepare_result_export_snapshot,
)
from fem.post.fields import (
    ResultRegionKey,
    make_result_region_signature,
)


def _source(suffix: str = "1") -> ResultSourceKey:
    return ResultSourceKey(
        result_id=f"result-{suffix}",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=3,
        step_name="Step-1",
        run_id=f"run-{suffix}",
    )


def _region(name: str = "steel") -> ResultRegionKey:
    return ResultRegionKey(
        make_result_region_signature({"material": name}),
        make_result_region_signature({"section": "solid"}),
    )


def _key(
    *,
    variable: ResultVariable = ResultVariable.S,
    position: FieldPosition = FieldPosition.CENTROID,
    contract: int = 1,
) -> FieldMaterializationKey:
    return FieldMaterializationKey(
        FieldRequest(ResultFieldId(variable, position)),
        contract,
    )


def _descriptor(
    *,
    variable: ResultVariable = ResultVariable.S,
    position: FieldPosition = FieldPosition.CENTROID,
    association: FieldAssociation = FieldAssociation.ELEMENT,
    components: tuple[str, ...] = ("S11",),
    derived: tuple[str, ...] = ("Mises",),
    default: str = "Mises",
    order: int = 10,
) -> FieldDescriptor:
    quantity = {
        ResultVariable.U: PhysicalQuantity.DISPLACEMENT,
        ResultVariable.UR: PhysicalQuantity.ROTATION,
        ResultVariable.RF: PhysicalQuantity.FORCE,
        ResultVariable.RM: PhysicalQuantity.MOMENT,
        ResultVariable.S: PhysicalQuantity.STRESS,
        ResultVariable.LE: PhysicalQuantity.STRAIN,
    }[variable]
    return FieldDescriptor(
        field_id=ResultFieldId(variable, position),
        association=association,
        quantity=quantity,
        components=components,
        derived_components=derived,
        label_key=f"result.{variable.value}.{position.value}",
        unit_label=None,
        default_component=default,
        order=order,
    )


def _topology(source: ResultSourceKey | None = None) -> ResultTopologyProjection:
    actual_source = source or _source()
    return ResultTopologyProjection(
        source=actual_source,
        node_ids=(10, 20, 30),
        node_coordinates=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            dtype=float,
        ),
        nodal_displacements=np.asarray(
            ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)),
            dtype=float,
        ),
        element_ids=(100, 200),
        element_types=("Truss2", "Truss2"),
        connectivity=((10, 20), (20, 30)),
        element_region_keys=(_region(), _region()),
    )


def _centroid_field(
    *,
    source: ResultSourceKey | None = None,
    contract: int = 1,
) -> FieldData:
    actual_source = source or _source()
    return FieldData(
        descriptor=_descriptor(),
        source=actual_source,
        key=_key(contract=contract),
        locations=(
            FieldLocation(
                FieldAssociation.ELEMENT,
                (0.5, 0.0, 0.0),
                (0.05, 0.0, 0.0),
                element_id=100,
            ),
            FieldLocation(
                FieldAssociation.ELEMENT,
                (1.5, 0.0, 0.0),
                (0.15, 0.0, 0.0),
                element_id=200,
            ),
        ),
        values=np.asarray(((10.0, 10.0), (20.0, 20.0))),
    )


def _node_field(
    *,
    source: ResultSourceKey | None = None,
) -> FieldData:
    actual_source = source or _source()
    return FieldData(
        descriptor=_descriptor(
            variable=ResultVariable.U,
            position=FieldPosition.NODE,
            association=FieldAssociation.NODE,
            components=("U1", "U2"),
            derived=("Magnitude",),
            default="Magnitude",
            order=0,
        ),
        source=actual_source,
        key=_key(
            variable=ResultVariable.U,
            position=FieldPosition.NODE,
        ),
        locations=tuple(
            FieldLocation(
                FieldAssociation.NODE,
                coordinates,
                displacement,
                node_id=node_id,
            )
            for node_id, coordinates, displacement in (
                (10, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                (20, (1.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
                (30, (2.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
            )
        ),
        values=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 0.1),
                (0.2, 0.0, 0.2),
            )
        ),
    )


def _diagnostic() -> ResultDiagnostic:
    return ResultDiagnostic(
        code="result.field.unavailable",
        severity="warning",
        message="Field is not available.",
        path=("outputs", 0),
        remediation="Choose a supported field.",
        details={"family": "mixed", "supported": ["U", "RF"]},
    )


def test_field_descriptor_freezes_complete_column_order() -> None:
    descriptor = _descriptor()

    assert descriptor.columns == ("S11", "Mises")
    assert descriptor.default_component == "Mises"
    with pytest.raises(FrozenInstanceError):
        descriptor.order = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"association": FieldAssociation.NODE},
        {"components": ("S11", "S11")},
        {"components": ("S11",), "derived": ("S11",)},
        {"components": (), "derived": ()},
        {"default": "S22"},
        {"order": -1},
    ),
)
def test_field_descriptor_rejects_inconsistent_contract(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _descriptor(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "location",
    (
        FieldLocation(
            FieldAssociation.NODE,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            node_id=10,
        ),
        FieldLocation(
            FieldAssociation.ELEMENT,
            (0.5, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            element_id=100,
        ),
        FieldLocation(
            FieldAssociation.INTEGRATION_POINT,
            (0.25, 0.0, 0.0),
            None,
            element_id=100,
            integration_point=1,
        ),
        FieldLocation(
            FieldAssociation.ELEMENT_NODE,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            node_id=10,
            element_id=100,
            local_node=1,
        ),
        FieldLocation(
            FieldAssociation.NODE_REGION,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            node_id=10,
            region_key=_region(),
        ),
        FieldLocation(
            FieldAssociation.RESOLVED_NODAL,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            node_id=10,
            region_key=_region(),
            averaged=True,
        ),
        FieldLocation(
            FieldAssociation.RESOLVED_NODAL,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            node_id=10,
            element_id=100,
            local_node=1,
            region_key=_region(),
            averaged=False,
        ),
    ),
)
def test_field_location_accepts_each_exact_identity_matrix(
    location: FieldLocation,
) -> None:
    assert location.coordinates == pytest.approx((0.0, 0.0, 0.0), abs=0.5)


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "association": FieldAssociation.NODE,
            "element_id": 100,
        },
        {
            "association": FieldAssociation.ELEMENT,
            "element_id": None,
        },
        {
            "association": FieldAssociation.INTEGRATION_POINT,
            "element_id": 100,
        },
        {
            "association": FieldAssociation.ELEMENT_NODE,
            "element_id": 100,
            "local_node": 1,
        },
        {
            "association": FieldAssociation.NODE_REGION,
            "node_id": 10,
        },
        {
            "association": FieldAssociation.RESOLVED_NODAL,
            "node_id": 10,
            "region_key": _region(),
            "averaged": False,
        },
        {
            "association": FieldAssociation.RESOLVED_NODAL,
            "node_id": 10,
            "element_id": 100,
            "local_node": 1,
            "region_key": _region(),
            "averaged": True,
        },
    ),
)
def test_field_location_rejects_missing_or_extra_identity(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        FieldLocation(
            coordinates=(0.0, 0.0, 0.0),
            displacement=None,
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("coordinates", "error"),
    (
        ([0.0, 0.0, 0.0], TypeError),
        ((0.0, 0.0), ValueError),
        ((0.0, True, 0.0), TypeError),
        ((0.0, float("nan"), 0.0), ValueError),
        ((0.0, float("inf"), 0.0), ValueError),
    ),
)
def test_field_location_requires_finite_triplets(
    coordinates: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        FieldLocation(
            FieldAssociation.NODE,
            coordinates,  # type: ignore[arg-type]
            None,
            node_id=1,
        )


def test_result_diagnostic_deep_freezes_nested_json() -> None:
    details = {
        "family": "continuum",
        "nested": {"orders": [1, 2]},
    }
    diagnostic = ResultDiagnostic(
        "result.recovery.lazy",
        "info",
        "Recovery is deferred.",
        ("fields", {"index": 1}),
        "Materialize the field when needed.",
        details,
    )
    details["family"] = "changed"
    details["nested"]["orders"].append(3)  # type: ignore[index, union-attr]

    assert diagnostic.details["family"] == "continuum"
    assert diagnostic.details["nested"]["orders"] == (1, 2)
    assert diagnostic.path[1]["index"] == 1  # type: ignore[index]
    assert deepcopy(diagnostic.details) is diagnostic.details
    with pytest.raises(TypeError):
        diagnostic.details["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        diagnostic.details._items = ()  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        diagnostic.details["nested"]._items = ()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "details",
    (
        {"bad": float("nan")},
        {"bad": object()},
        {1: "non-string-key"},
    ),
)
def test_result_diagnostic_rejects_non_json_details(
    details: dict[object, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ResultDiagnostic(
            "result.invalid",
            "error",
            "Invalid result.",
            (),
            "Correct the result.",
            details,  # type: ignore[arg-type]
        )


def test_topology_projection_deep_owns_arrays_and_hides_internal_bases() -> None:
    coordinates = np.arange(9.0).reshape(3, 3)
    displacements = coordinates / 10.0
    topology = ResultTopologyProjection(
        _source(),
        (10, 20, 30),
        coordinates,
        displacements,
        (100,),
        ("Tri3",),
        ((10, 20, 30),),
        (_region(),),
    )
    coordinates[:] = -1.0
    displacements[:] = -1.0

    first_coordinates = topology.node_coordinates
    first_displacements = topology.nodal_displacements
    assert first_coordinates.flags.owndata
    assert first_coordinates.base is None
    assert not first_coordinates.flags.writeable
    assert not first_displacements.flags.writeable
    np.testing.assert_allclose(
        first_coordinates,
        np.arange(9.0).reshape(3, 3),
    )
    first_coordinates.setflags(write=True)
    first_coordinates[:] = 999.0
    np.testing.assert_allclose(
        topology.node_coordinates,
        np.arange(9.0).reshape(3, 3),
    )


@pytest.mark.parametrize(
    "replacement",
    (
        {"node_coordinates": np.zeros((3, 2))},
        {"nodal_displacements": np.full((3, 3), np.nan)},
        {"node_ids": (10, 10, 30)},
        {"element_ids": (100, 100)},
        {"connectivity": ((10, 99), (20, 30))},
        {"element_types": ("Truss2",)},
        {"element_region_keys": (_region(),)},
    ),
)
def test_topology_projection_rejects_shape_identity_and_order_damage(
    replacement: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "source": _source(),
        "node_ids": (10, 20, 30),
        "node_coordinates": np.zeros((3, 3)),
        "nodal_displacements": np.zeros((3, 3)),
        "element_ids": (100, 200),
        "element_types": ("Truss2", "Truss2"),
        "connectivity": ((10, 20), (20, 30)),
        "element_region_keys": (_region(), _region()),
    }
    arguments.update(replacement)

    with pytest.raises((TypeError, ValueError)):
        ResultTopologyProjection(**arguments)  # type: ignore[arg-type]


def test_field_data_owns_finite_values_in_complete_column_shape() -> None:
    values = np.asarray(((10, 10), (20, 20)), dtype=np.int64)
    field_data = FieldData(
        _descriptor(),
        _source(),
        _key(),
        _centroid_field().locations,
        values,
    )
    values[:] = -1

    first = field_data.values
    assert first.dtype == np.dtype(float)
    assert first.flags.c_contiguous
    assert first.flags.owndata
    assert first.base is None
    assert not first.flags.writeable
    np.testing.assert_allclose(first, ((10.0, 10.0), (20.0, 20.0)))
    first.setflags(write=True)
    first[:] = 999.0
    np.testing.assert_allclose(
        field_data.values,
        ((10.0, 10.0), (20.0, 20.0)),
    )


def test_field_data_returns_one_detached_component_column() -> None:
    field_data = _centroid_field()

    component = field_data.component_values("Mises")

    np.testing.assert_array_equal(component, (10.0, 20.0))
    assert component.shape == (2,)
    assert component.flags.owndata
    assert not component.flags.writeable
    component.setflags(write=True)
    component[:] = -1.0
    np.testing.assert_array_equal(
        field_data.component_values("Mises"),
        (10.0, 20.0),
    )


@pytest.mark.parametrize(
    "values",
    (
        np.zeros((2,)),
        np.zeros((2, 1)),
        np.zeros((1, 2)),
        np.asarray(((True, False), (False, True))),
        np.asarray(((1.0 + 0.0j, 2.0), (3.0, 4.0))),
        np.asarray(((1.0, np.nan), (3.0, 4.0))),
        np.asarray(((1.0, np.inf), (3.0, 4.0))),
    ),
)
def test_field_data_rejects_invalid_value_matrix(values: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        FieldData(
            _descriptor(),
            _source(),
            _key(),
            _centroid_field().locations,
            values,
        )


def test_field_data_requires_matching_descriptor_key_and_locations() -> None:
    descriptor = _descriptor()
    source = _source()
    locations = _centroid_field().locations

    with pytest.raises(ValueError, match="field_id"):
        FieldData(
            descriptor,
            source,
            _key(
                variable=ResultVariable.LE,
                position=FieldPosition.CENTROID,
            ),
            locations,
            np.zeros((2, 2)),
        )
    with pytest.raises(ValueError, match="association"):
        FieldData(
            descriptor,
            source,
            _key(),
            (
                FieldLocation(
                    FieldAssociation.NODE,
                    (0.0, 0.0, 0.0),
                    None,
                    node_id=10,
                ),
            ),
            np.zeros((1, 2)),
        )
    with pytest.raises(ValueError, match="unique"):
        FieldData(
            descriptor,
            source,
            _key(),
            (locations[0], locations[0]),
            np.zeros((2, 2)),
        )


def test_availability_and_catalog_bind_full_key_descriptor_and_default() -> None:
    field_data = _centroid_field()
    availability = FieldAvailability(
        field_data.key,
        field_data.descriptor,
        FieldState.LAZY,
        (_diagnostic(),),
    )
    catalog = ResultCatalog(
        field_data.source,
        (availability,),
        ScalarFieldSelection(field_data.key, "Mises"),
        (_diagnostic(),),
    )

    assert catalog.default_selection.field_key == field_data.key
    assert tuple(item.code for item in catalog.diagnostics) == (
        "result.field.unavailable",
    )
    with pytest.raises(TypeError, match="diagnostics must be a tuple"):
        ResultCatalog(
            field_data.source,
            (availability,),
            ScalarFieldSelection(field_data.key, "Mises"),
            [_diagnostic()],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="ResultDiagnostic"):
        ResultCatalog(
            field_data.source,
            (availability,),
            ScalarFieldSelection(field_data.key, "Mises"),
            (object(),),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unavailable"):
        ResultCatalog(
            field_data.source,
            (
                FieldAvailability(
                    field_data.key,
                    field_data.descriptor,
                    FieldState.UNAVAILABLE,
                    (_diagnostic(),),
                ),
            ),
            ScalarFieldSelection(field_data.key, "Mises"),
        )
    with pytest.raises(ValueError, match="component"):
        ResultCatalog(
            field_data.source,
            (availability,),
            ScalarFieldSelection(field_data.key, "S22"),
        )


def test_catalog_rejects_duplicate_or_noncanonical_key_order() -> None:
    first = _centroid_field(contract=1)
    second = _centroid_field(contract=2)
    first_availability = FieldAvailability(
        first.key,
        first.descriptor,
        FieldState.READY,
    )
    second_availability = FieldAvailability(
        second.key,
        second.descriptor,
        FieldState.LAZY,
    )

    with pytest.raises(ValueError, match="unique"):
        ResultCatalog(
            first.source,
            (first_availability, first_availability),
            ScalarFieldSelection(first.key, "S11"),
        )
    with pytest.raises(ValueError, match="order"):
        ResultCatalog(
            first.source,
            (second_availability, first_availability),
            ScalarFieldSelection(first.key, "S11"),
        )


def test_initial_materialization_combines_and_sorts_without_advancing() -> None:
    source = _source()
    topology = _topology(source)
    base_field = _node_field(source=source)
    eager_field = _centroid_field(source=source)
    patch = ResultMaterializationPatch(
        source,
        (eager_field,),
        (_diagnostic(),),
    )

    snapshot = build_initial_materialization(
        source,
        topology,
        (base_field,),
        (patch,),
    )

    assert snapshot.generation == 0
    assert snapshot.topology is topology
    assert snapshot.fields == (base_field, eager_field)


def test_snapshot_and_patch_require_exact_source_unique_sorted_fields() -> None:
    source = _source()
    foreign = _source("foreign")
    topology = _topology(source)
    first = _centroid_field(source=source, contract=1)
    second = _centroid_field(source=source, contract=2)

    with pytest.raises(ValueError, match="source"):
        ResultMaterializationPatch(
            source,
            (_centroid_field(source=foreign),),
        )
    with pytest.raises(ValueError, match="unique"):
        ResultMaterializationPatch(source, (first, first))
    with pytest.raises(ValueError, match="order"):
        ResultMaterializationPatch(source, (second, first))
    with pytest.raises(ValueError, match="source"):
        ResultMaterializationSnapshot(
            source,
            0,
            _topology(foreign),
            (),
        )
    with pytest.raises(TypeError, match="generation"):
        ResultMaterializationSnapshot(
            source,
            True,  # type: ignore[arg-type]
            topology,
            (),
        )


def test_advance_materialization_adds_only_new_keys_and_reuses_topology() -> None:
    source = _source()
    topology = _topology(source)
    original = _node_field(source=source)
    current = build_initial_materialization(
        source,
        topology,
        (original,),
    )
    added = _centroid_field(source=source)

    advanced = advance_materialization(
        current,
        ResultMaterializationPatch(source, (added,)),
    )

    assert current.generation == 0
    assert current.fields == (original,)
    assert advanced.generation == 1
    assert advanced.topology is topology
    assert advanced.fields == (original, added)
    assert advanced.fields[0] is original
    assert advanced.fields[1] is added


def test_advance_materialization_rejects_empty_overlap_and_foreign_patch() -> None:
    source = _source()
    current = build_initial_materialization(
        source,
        _topology(source),
        (_node_field(source=source),),
    )

    with pytest.raises(ValueError, match="at least one"):
        advance_materialization(
            current,
            ResultMaterializationPatch(source, ()),
        )
    with pytest.raises(ValueError, match="replace"):
        advance_materialization(
            current,
            ResultMaterializationPatch(
                source,
                (_node_field(source=source),),
            ),
        )
    foreign = _source("foreign")
    with pytest.raises(ValueError, match="source"):
        advance_materialization(
            current,
            ResultMaterializationPatch(
                foreign,
                (_centroid_field(source=foreign),),
            ),
        )


def test_prepare_export_snapshot_binds_exact_generation_field_and_component() -> None:
    source = _source()
    field_data = _centroid_field(source=source)
    materialization = build_initial_materialization(
        source,
        _topology(source),
        (field_data,),
    )
    selection = ScalarFieldSelection(field_data.key, "Mises")

    export = prepare_result_export_snapshot(materialization, selection)

    assert export.source == source
    assert export.materialization_generation == 0
    assert export.topology is materialization.topology
    assert export.field is field_data
    assert export.selection is selection
    with pytest.raises(TypeError, match="prepare_result_export_snapshot"):
        ResultExportSnapshot(
            source,
            0,
            materialization.topology,
            field_data,
            selection,
        )


def test_prepare_export_snapshot_rejects_partial_or_wrong_selection() -> None:
    source = _source()
    field_data = _centroid_field(source=source)
    materialization = build_initial_materialization(
        source,
        _topology(source),
        (field_data,),
    )

    with pytest.raises(KeyError, match="not materialized"):
        prepare_result_export_snapshot(
            materialization,
            ScalarFieldSelection(
                _key(contract=2),
                "Mises",
            ),
        )
    with pytest.raises(ValueError, match="component"):
        prepare_result_export_snapshot(
            materialization,
            ScalarFieldSelection(field_data.key, "S22"),
        )
