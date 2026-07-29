from __future__ import annotations

import numpy as np
import pytest

import fem.application.results.topology as topology_module
from fem.application.results import (
    FieldAssociation,
    FieldData,
    FieldDescriptor,
    FieldLocation,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    PhysicalQuantity,
    ResultCellKind,
    ResultFieldId,
    ResultFieldTopology,
    ResultMaterializationSnapshot,
    ResultSourceKey,
    ResultTopologyProjection,
    ResultValueLayout,
    ResultVariable,
    ScalarFieldSelection,
    build_result_field_topology_template,
    prepare_result_export_snapshot,
    project_scalar_field_topology,
    project_scalar_field_topology_from_template,
)
from fem.post.averaging import NodalAveragingPolicy
from fem.post.fields import (
    ResultRegionKey,
    make_result_region_signature,
)


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="result-1",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=4,
        step_name="Step-1",
        run_id="run-1",
    )


def _region(name: str) -> ResultRegionKey:
    return ResultRegionKey(
        make_result_region_signature({"material": name}),
        make_result_region_signature({"section": "solid"}),
    )


REGION_A = _region("A")
REGION_B = _region("B")


def _topology() -> ResultTopologyProjection:
    return ResultTopologyProjection(
        source=_source(),
        node_ids=(10, 20, 30, 40, 50),
        node_coordinates=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
            )
        ),
        nodal_displacements=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 0.0),
                (0.0, 0.2, 0.0),
                (0.3, 0.0, 0.0),
                (0.4, 0.0, 0.0),
            )
        ),
        element_ids=(100, 200, 300),
        element_types=("Tri3", "Tri3", "Tri3"),
        connectivity=(
            (10, 20, 30),
            (20, 40, 30),
            (20, 50, 30),
        ),
        element_region_keys=(REGION_A, REGION_A, REGION_B),
    )


def _field_contract(
    position: FieldPosition,
) -> tuple[
    FieldAssociation,
    ResultVariable,
    PhysicalQuantity,
    FieldMaterializationKey,
]:
    association = {
        FieldPosition.NODE: FieldAssociation.NODE,
        FieldPosition.CENTROID: FieldAssociation.ELEMENT,
        FieldPosition.INTEGRATION_POINT: (FieldAssociation.INTEGRATION_POINT),
        FieldPosition.ELEMENT_NODAL: FieldAssociation.ELEMENT_NODE,
        FieldPosition.SECTION_END: FieldAssociation.ELEMENT_NODE,
        FieldPosition.NODE_REGION: FieldAssociation.NODE_REGION,
        FieldPosition.RESOLVED_NODAL: FieldAssociation.RESOLVED_NODAL,
    }[position]
    variable = ResultVariable.U if position is FieldPosition.NODE else ResultVariable.S
    quantity = (
        PhysicalQuantity.DISPLACEMENT
        if variable is ResultVariable.U
        else PhysicalQuantity.STRESS
    )
    policy = (
        NodalAveragingPolicy() if position is FieldPosition.RESOLVED_NODAL else None
    )
    key = FieldMaterializationKey(
        FieldRequest(
            ResultFieldId(variable, position),
            averaging_policy=policy,
        ),
        recovery_contract=2,
    )
    return association, variable, quantity, key


def _field(
    position: FieldPosition,
    locations: tuple[FieldLocation, ...],
    values: tuple[float, ...],
) -> FieldData:
    association, variable, quantity, key = _field_contract(position)
    component = "U1" if variable is ResultVariable.U else "S11"
    descriptor = FieldDescriptor(
        field_id=key.request.field_id,
        association=association,
        quantity=quantity,
        components=(component,),
        derived_components=(),
        label_key=f"result.{variable.value}.{position.value}",
        unit_label=None,
        default_component=component,
        order=0,
    )
    return FieldData(
        descriptor=descriptor,
        source=_source(),
        key=key,
        locations=locations,
        values=np.asarray(values, dtype=float).reshape((-1, 1)),
    )


def _export(field_data: FieldData):
    materialization = ResultMaterializationSnapshot(
        source=_source(),
        generation=3,
        topology=_topology(),
        fields=(field_data,),
    )
    selection = ScalarFieldSelection(
        field_data.key,
        field_data.descriptor.default_component,
    )
    return prepare_result_export_snapshot(materialization, selection)


def _node_location(
    association: FieldAssociation,
    node_id: int,
    **identity: object,
) -> FieldLocation:
    topology = _topology()
    index = topology.node_ids.index(node_id)
    return FieldLocation(
        association=association,
        coordinates=tuple(topology.node_coordinates[index]),
        displacement=tuple(topology.nodal_displacements[index]),
        node_id=node_id,
        **identity,
    )


def _all_element_node_locations() -> tuple[FieldLocation, ...]:
    topology = _topology()
    return tuple(
        _node_location(
            FieldAssociation.ELEMENT_NODE,
            node_id,
            element_id=element_id,
            local_node=local_node,
        )
        for element_id, connected in zip(
            topology.element_ids,
            topology.connectivity,
            strict=True,
        )
        for local_node, node_id in enumerate(connected, start=1)
    )


def _node_region_locations() -> tuple[FieldLocation, ...]:
    return (
        _node_location(
            FieldAssociation.NODE_REGION,
            30,
            region_key=REGION_B,
        ),
        _node_location(
            FieldAssociation.NODE_REGION,
            20,
            region_key=REGION_A,
        ),
        _node_location(
            FieldAssociation.NODE_REGION,
            10,
            region_key=REGION_A,
        ),
        _node_location(
            FieldAssociation.NODE_REGION,
            30,
            region_key=REGION_A,
        ),
        _node_location(
            FieldAssociation.NODE_REGION,
            50,
            region_key=REGION_B,
        ),
        _node_location(
            FieldAssociation.NODE_REGION,
            40,
            region_key=REGION_A,
        ),
        _node_location(
            FieldAssociation.NODE_REGION,
            20,
            region_key=REGION_B,
        ),
    )


def _resolved_locations() -> tuple[FieldLocation, ...]:
    return (
        _node_location(
            FieldAssociation.RESOLVED_NODAL,
            30,
            region_key=REGION_A,
            averaged=False,
            element_id=200,
            local_node=3,
        ),
        _node_location(
            FieldAssociation.RESOLVED_NODAL,
            20,
            region_key=REGION_A,
            averaged=True,
        ),
        _node_location(
            FieldAssociation.RESOLVED_NODAL,
            30,
            region_key=REGION_A,
            averaged=False,
            element_id=100,
            local_node=3,
        ),
        _node_location(
            FieldAssociation.RESOLVED_NODAL,
            10,
            region_key=REGION_A,
            averaged=True,
        ),
        _node_location(
            FieldAssociation.RESOLVED_NODAL,
            40,
            region_key=REGION_A,
            averaged=True,
        ),
        _node_location(
            FieldAssociation.RESOLVED_NODAL,
            20,
            region_key=REGION_B,
            averaged=True,
        ),
        _node_location(
            FieldAssociation.RESOLVED_NODAL,
            30,
            region_key=REGION_B,
            averaged=True,
        ),
        _node_location(
            FieldAssociation.RESOLVED_NODAL,
            50,
            region_key=REGION_B,
            averaged=True,
        ),
    )


def test_node_projection_uses_mesh_order_connectivity_and_owned_deformation() -> None:
    locations = tuple(
        _node_location(FieldAssociation.NODE, node_id)
        for node_id in reversed(_topology().node_ids)
    )
    field_data = _field(
        FieldPosition.NODE,
        locations,
        tuple(float(node_id) for node_id in reversed(_topology().node_ids)),
    )

    projected = project_scalar_field_topology(
        _export(field_data),
        deformation_scale=2.0,
    )

    assert projected.source == _source()
    assert projected.materialization_generation == 3
    assert projected.selection.field_key == field_data.key
    assert projected.deformation_scale == 2.0
    assert projected.value_layout is ResultValueLayout.POINT
    assert projected.cells == (
        (0, 1, 2),
        (1, 3, 2),
        (1, 4, 2),
    )
    assert projected.cell_kinds == (ResultCellKind.FEM_ELEMENT,) * 3
    assert projected.canonical_element_types == ("Tri3",) * 3
    np.testing.assert_allclose(
        projected.points,
        (
            (0.0, 0.0, 0.0),
            (1.2, 0.0, 0.0),
            (0.0, 1.4, 0.0),
            (2.6, 0.0, 0.0),
            (3.8, 0.0, 0.0),
        ),
    )
    np.testing.assert_array_equal(
        projected.values,
        (10.0, 20.0, 30.0, 40.0, 50.0),
    )
    assert (
        tuple(
            location.node_id
            for location in projected.point_locations
            if location is not None
        )
        == _topology().node_ids
    )

    public_points = projected.points
    public_values = projected.values
    public_points.setflags(write=True)
    public_values.setflags(write=True)
    public_points[0, 0] = 999.0
    public_values[0] = 999.0
    assert projected.points[0, 0] == 0.0
    assert projected.values[0] == 10.0
    assert not projected.points.flags.writeable
    assert not projected.values.flags.writeable


def test_element_projection_keeps_full_deformed_cells_and_cell_order() -> None:
    field_data = _field(
        FieldPosition.CENTROID,
        (
            FieldLocation(
                FieldAssociation.ELEMENT,
                (1.5, 0.5, 0.0),
                (99.0, 0.0, 0.0),
                element_id=300,
            ),
            FieldLocation(
                FieldAssociation.ELEMENT,
                (0.5, 0.5, 0.0),
                (99.0, 0.0, 0.0),
                element_id=100,
            ),
            FieldLocation(
                FieldAssociation.ELEMENT,
                (1.0, 0.5, 0.0),
                (99.0, 0.0, 0.0),
                element_id=200,
            ),
        ),
        (30.0, 10.0, 20.0),
    )

    projected = project_scalar_field_topology(
        _export(field_data),
        deformation_scale=1.0,
    )

    assert projected.value_layout is ResultValueLayout.CELL
    np.testing.assert_array_equal(projected.values, (10.0, 20.0, 30.0))
    np.testing.assert_allclose(
        projected.points,
        _topology().node_coordinates + _topology().nodal_displacements,
    )
    assert tuple(
        location.element_id
        for location in projected.cell_locations
        if location is not None
    ) == (100, 200, 300)


def test_integration_points_become_sample_vertices_in_field_row_order() -> None:
    locations = (
        FieldLocation(
            FieldAssociation.INTEGRATION_POINT,
            (1.5, 0.2, 0.0),
            (0.25, 0.0, 0.0),
            element_id=300,
            integration_point=2,
        ),
        FieldLocation(
            FieldAssociation.INTEGRATION_POINT,
            (0.2, 0.2, 0.0),
            (0.05, 0.1, 0.0),
            element_id=100,
            integration_point=1,
        ),
    )
    field_data = _field(
        FieldPosition.INTEGRATION_POINT,
        locations,
        (32.0, 11.0),
    )

    projected = project_scalar_field_topology(
        _export(field_data),
        deformation_scale=2.0,
    )

    assert projected.cells == ((0,), (1,))
    assert projected.cell_kinds == (
        ResultCellKind.SAMPLE_VERTEX,
        ResultCellKind.SAMPLE_VERTEX,
    )
    assert projected.canonical_element_types == (None, None)
    assert projected.value_layout is ResultValueLayout.POINT
    np.testing.assert_allclose(
        projected.points,
        ((2.0, 0.2, 0.0), (0.3, 0.4, 0.0)),
    )
    np.testing.assert_array_equal(projected.values, (32.0, 11.0))
    assert projected.point_locations == locations
    assert projected.cell_locations == locations


@pytest.mark.parametrize(
    "position",
    (FieldPosition.ELEMENT_NODAL, FieldPosition.SECTION_END),
)
def test_element_node_positions_duplicate_every_element_local_point(
    position: FieldPosition,
) -> None:
    canonical = _all_element_node_locations()
    shuffled = tuple(reversed(canonical))
    values_by_identity = {
        (location.element_id, location.local_node): float(index)
        for index, location in enumerate(canonical, start=1)
    }
    field_data = _field(
        position,
        shuffled,
        tuple(
            values_by_identity[(location.element_id, location.local_node)]
            for location in shuffled
        ),
    )

    projected = project_scalar_field_topology(
        _export(field_data),
        deformation_scale=1.0,
    )

    assert projected.cells == (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
    )
    assert projected.cell_kinds == (ResultCellKind.FEM_ELEMENT,) * 3
    np.testing.assert_array_equal(
        projected.values,
        np.arange(1.0, 10.0),
    )
    assert tuple(
        (location.element_id, location.local_node, location.node_id)
        for location in projected.point_locations
        if location is not None
    ) == tuple(
        (location.element_id, location.local_node, location.node_id)
        for location in canonical
    )


def test_component_projection_reuses_cached_element_node_layout() -> None:
    canonical = _all_element_node_locations()
    shuffled = tuple(reversed(canonical))
    association, variable, quantity, key = _field_contract(
        FieldPosition.ELEMENT_NODAL
    )
    descriptor = FieldDescriptor(
        field_id=key.request.field_id,
        association=association,
        quantity=quantity,
        components=("S11", "S22"),
        derived_components=(),
        label_key=f"result.{variable.value}.element_nodal",
        unit_label=None,
        default_component="S11",
        order=0,
    )
    values_by_location = {
        location: (float(index), float(index + 100))
        for index, location in enumerate(canonical, start=1)
    }
    field_data = FieldData(
        descriptor=descriptor,
        source=_source(),
        key=key,
        locations=shuffled,
        values=np.asarray(
            tuple(values_by_location[location] for location in shuffled)
        ),
    )
    materialization = ResultMaterializationSnapshot(
        source=_source(),
        generation=3,
        topology=_topology(),
        fields=(field_data,),
    )
    first_export = prepare_result_export_snapshot(
        materialization,
        ScalarFieldSelection(key, "S11"),
    )
    second_export = prepare_result_export_snapshot(
        materialization,
        ScalarFieldSelection(key, "S22"),
    )
    first = project_scalar_field_topology(
        first_export,
        deformation_scale=1.0,
    )
    template = build_result_field_topology_template(first, field_data)

    second = project_scalar_field_topology_from_template(
        second_export,
        template,
        deformation_scale=1.0,
    )

    assert second.cells is first.cells
    assert second.point_locations is first.point_locations
    assert second.cell_locations is first.cell_locations
    np.testing.assert_array_equal(second.points, first.points)
    np.testing.assert_array_equal(
        second.values,
        np.arange(101.0, 110.0),
    )


def test_element_node_deformation_is_batched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _all_element_node_locations()
    field_data = _field(
        FieldPosition.ELEMENT_NODAL,
        canonical,
        tuple(float(index) for index in range(1, 10)),
    )

    def reject_per_location_deformation(*_args: object) -> None:
        raise AssertionError("element-node projection must batch deformation")

    monkeypatch.setattr(
        topology_module,
        "_deformed_location",
        reject_per_location_deformation,
    )

    projected = project_scalar_field_topology(
        _export(field_data),
        deformation_scale=1.0,
    )

    np.testing.assert_array_equal(
        projected.values,
        np.arange(1.0, 10.0),
    )


def test_node_region_projection_reuses_only_matching_node_region_points() -> None:
    locations = _node_region_locations()
    values_by_identity = {
        (location.node_id, location.region_key): float(index)
        for index, location in enumerate(locations, start=1)
    }
    field_data = _field(
        FieldPosition.NODE_REGION,
        locations,
        tuple(
            values_by_identity[(location.node_id, location.region_key)]
            for location in locations
        ),
    )

    projected = project_scalar_field_topology(_export(field_data))

    assert projected.cells == (
        (0, 1, 2),
        (1, 3, 2),
        (4, 5, 6),
    )
    assert tuple(
        (location.node_id, location.region_key)
        for location in projected.point_locations
        if location is not None
    ) == (
        (10, REGION_A),
        (20, REGION_A),
        (30, REGION_A),
        (40, REGION_A),
        (20, REGION_B),
        (50, REGION_B),
        (30, REGION_B),
    )
    np.testing.assert_array_equal(
        projected.values,
        tuple(
            values_by_identity[identity]
            for identity in (
                (10, REGION_A),
                (20, REGION_A),
                (30, REGION_A),
                (40, REGION_A),
                (20, REGION_B),
                (50, REGION_B),
                (30, REGION_B),
            )
        ),
    )


def test_resolved_projection_prefers_matching_average_then_exact_raw_row() -> None:
    locations = _resolved_locations()
    values_by_location = {
        location: float(index) for index, location in enumerate(locations, start=1)
    }
    field_data = _field(
        FieldPosition.RESOLVED_NODAL,
        locations,
        tuple(values_by_location[location] for location in locations),
    )

    projected = project_scalar_field_topology(_export(field_data))

    assert projected.cells == (
        (0, 1, 2),
        (1, 3, 4),
        (5, 6, 7),
    )
    identities = tuple(
        (
            location.node_id,
            location.region_key,
            location.averaged,
            location.element_id,
            location.local_node,
        )
        for location in projected.point_locations
        if location is not None
    )
    assert identities == (
        (10, REGION_A, True, None, None),
        (20, REGION_A, True, None, None),
        (30, REGION_A, False, 100, 3),
        (40, REGION_A, True, None, None),
        (30, REGION_A, False, 200, 3),
        (20, REGION_B, True, None, None),
        (50, REGION_B, True, None, None),
        (30, REGION_B, True, None, None),
    )
    np.testing.assert_array_equal(
        projected.values,
        tuple(
            values_by_location[location]
            for location in projected.point_locations
            if location is not None
        ),
    )


@pytest.mark.parametrize("case", ("missing", "wrong_region", "extra"))
def test_node_region_projection_rejects_non_exact_row_coverage(
    case: str,
) -> None:
    locations = list(_node_region_locations())
    if case == "missing":
        locations = [
            location
            for location in locations
            if not (location.node_id == 50 and location.region_key == REGION_B)
        ]
    elif case == "wrong_region":
        locations = [
            (
                _node_location(
                    FieldAssociation.NODE_REGION,
                    50,
                    region_key=REGION_A,
                )
                if location.node_id == 50 and location.region_key == REGION_B
                else location
            )
            for location in locations
        ]
    else:
        locations.append(
            _node_location(
                FieldAssociation.NODE_REGION,
                10,
                region_key=REGION_B,
            )
        )
    field_data = _field(
        FieldPosition.NODE_REGION,
        tuple(locations),
        tuple(float(index) for index in range(len(locations))),
    )

    with pytest.raises(ValueError, match="missing exact|outside"):
        project_scalar_field_topology(_export(field_data))


def test_resolved_projection_rejects_missing_raw_and_unused_competing_raw() -> None:
    locations = list(_resolved_locations())
    missing = tuple(
        location
        for location in locations
        if not (location.element_id == 200 and location.local_node == 3)
    )
    missing_field = _field(
        FieldPosition.RESOLVED_NODAL,
        missing,
        tuple(float(index) for index in range(len(missing))),
    )
    with pytest.raises(ValueError, match="missing exact"):
        project_scalar_field_topology(_export(missing_field))

    locations.append(
        _node_location(
            FieldAssociation.RESOLVED_NODAL,
            20,
            region_key=REGION_A,
            averaged=False,
            element_id=100,
            local_node=2,
        )
    )
    competing_field = _field(
        FieldPosition.RESOLVED_NODAL,
        tuple(locations),
        tuple(float(index) for index in range(len(locations))),
    )
    with pytest.raises(ValueError, match="outside"):
        project_scalar_field_topology(_export(competing_field))


@pytest.mark.parametrize(
    ("scale", "error"),
    (
        (True, TypeError),
        ("1", TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ),
)
def test_projector_rejects_invalid_deformation_scale(
    scale: object,
    error: type[Exception],
) -> None:
    field_data = _field(
        FieldPosition.NODE,
        tuple(
            _node_location(FieldAssociation.NODE, node_id)
            for node_id in _topology().node_ids
        ),
        tuple(float(node_id) for node_id in _topology().node_ids),
    )

    with pytest.raises(error, match="deformation_scale"):
        project_scalar_field_topology(
            _export(field_data),
            deformation_scale=scale,  # type: ignore[arg-type]
        )


def test_result_field_topology_rejects_non_projected_indexes_and_cardinality() -> None:
    field_data = _field(
        FieldPosition.NODE,
        tuple(
            _node_location(FieldAssociation.NODE, node_id)
            for node_id in _topology().node_ids
        ),
        tuple(float(node_id) for node_id in _topology().node_ids),
    )
    export = _export(field_data)
    source_points = np.zeros((1, 3))
    source_values = np.asarray((7.0,))
    owned = ResultFieldTopology(
        source=export.source,
        materialization_generation=export.materialization_generation,
        selection=export.selection,
        deformation_scale=0.0,
        points=source_points,
        cells=((0,),),
        cell_kinds=(ResultCellKind.SAMPLE_VERTEX,),
        canonical_element_types=(None,),
        values=source_values,
        value_layout=ResultValueLayout.POINT,
        point_locations=(None,),
        cell_locations=(None,),
    )
    source_points[0, 0] = 9.0
    source_values[0] = 9.0
    assert owned.points[0, 0] == 0.0
    assert owned.values[0] == 7.0

    with pytest.raises(ValueError, match="zero-based"):
        ResultFieldTopology(
            source=export.source,
            materialization_generation=export.materialization_generation,
            selection=export.selection,
            deformation_scale=0.0,
            points=np.zeros((1, 3)),
            cells=((10,),),
            cell_kinds=(ResultCellKind.FEM_ELEMENT,),
            canonical_element_types=("Tri3",),
            values=np.zeros(1),
            value_layout=ResultValueLayout.CELL,
            point_locations=(None,),
            cell_locations=(None,),
        )
    with pytest.raises(ValueError, match="values"):
        ResultFieldTopology(
            source=export.source,
            materialization_generation=export.materialization_generation,
            selection=export.selection,
            deformation_scale=0.0,
            points=np.zeros((1, 3)),
            cells=((0,),),
            cell_kinds=(ResultCellKind.SAMPLE_VERTEX,),
            canonical_element_types=(None,),
            values=np.zeros(0),
            value_layout=ResultValueLayout.POINT,
            point_locations=(None,),
            cell_locations=(None,),
        )
