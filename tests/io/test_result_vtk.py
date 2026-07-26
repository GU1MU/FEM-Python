from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any

import numpy as np
import pytest

import fem.io as fem_io
import fem.io.result_vtk as result_vtk_module
from fem.application.results import (
    FieldAssociation,
    FieldData,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultCellKind,
    ResultExportSnapshot,
    ResultFieldId,
    ResultMaterializationSnapshot,
    ResultSourceKey,
    ResultTopologyProjection,
    ResultValueLayout,
    ResultVariable,
    ScalarFieldSelection,
    advance_materialization,
    build_result_provider,
    prepare_result_export_snapshot,
    project_scalar_field_topology,
)
from fem.io.result_vtk import (
    RESULT_VTK_FORMAT_NAME,
    RESULT_VTK_SCHEMA_VERSION,
    RESULT_VTK_TITLE,
    ResultVtkDecodeError,
    ResultVtkEmptySelectionError,
    ResultVtkEncodeError,
    ResultVtkError,
    ResultVtkLocationIdentity,
    dumps_result_vtk,
    read_result_vtk,
    write_result_vtk,
)
from fem.post.averaging import NodalAveragingPolicy
from fem.post.fields import (
    ResultRegionKey,
    encode_result_region_key,
    make_result_region_signature,
)
from fem.post.vtk.cells import vtk_cell_type
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)


def _source(*, result_id: str = "result,结果-canonical") -> ResultSourceKey:
    return ResultSourceKey(
        result_id=result_id,
        session_id="session-会话",
        artifact_id="artifact-工件",
        model_revision=7,
        step_name="载荷步-1",
        run_id="run-运行",
    )


def _export(
    variable: ResultVariable,
    position: FieldPosition,
    *,
    policy: NodalAveragingPolicy | None = None,
    gauss_order: int | None = None,
    source: ResultSourceKey | None = None,
) -> ResultExportSnapshot:
    provider = build_result_provider(
        _source() if source is None else source,
        make_continuum_nodal_semantics_result(),
    )
    key = provider.resolve_request(
        FieldRequest(
            ResultFieldId(variable, position),
            averaging_policy=policy,
            gauss_order=gauss_order,
        )
    )
    snapshot = provider.snapshot
    if not any(field.key == key for field in snapshot.fields):
        snapshot = advance_materialization(
            snapshot,
            provider.materialize((key,)),
        )
    component = "U1" if variable is ResultVariable.U else "S11"
    return prepare_result_export_snapshot(
        snapshot,
        ScalarFieldSelection(key, component),
    )


def _with_key(
    export: ResultExportSnapshot,
    key: FieldMaterializationKey,
) -> ResultExportSnapshot:
    field = FieldData(
        descriptor=export.field.descriptor,
        source=export.source,
        key=key,
        locations=export.field.locations,
        values=export.field.values,
    )
    snapshot = ResultMaterializationSnapshot(
        source=export.source,
        generation=export.materialization_generation,
        topology=export.topology,
        fields=(field,),
    )
    return prepare_result_export_snapshot(
        snapshot,
        ScalarFieldSelection(key, export.selection.component),
    )


def _identity(location: Any) -> ResultVtkLocationIdentity | None:
    if location is None:
        return None
    return ResultVtkLocationIdentity.from_location(location)


def _write_serialized(path: Path, serialized: str) -> None:
    path.write_bytes(serialized.encode("ascii", errors="strict"))


def _mutate_field_values(
    serialized: str,
    name: str,
    replacement: str,
) -> str:
    lines = serialized.splitlines()
    for index, line in enumerate(lines):
        parts = line.split()
        if parts and parts[0] == name:
            lines[index + 1] = replacement
            return "\n".join(lines) + "\n"
    raise AssertionError(f"missing FIELD array {name}")


def _mutate_utf8_field(
    serialized: str,
    name: str,
    replacement: str,
) -> str:
    encoded = replacement.encode("utf-8", errors="strict")
    lines = serialized.splitlines()
    for index, line in enumerate(lines):
        parts = line.split()
        if parts and parts[0] == name:
            parts[2] = str(len(encoded))
            lines[index] = " ".join(parts)
            lines[index + 1] = " ".join(str(byte) for byte in encoded)
            return "\n".join(lines) + "\n"
    raise AssertionError(f"missing UTF-8 FIELD array {name}")


def _mutate_structural_value(serialized: str, name: str) -> str:
    lines = serialized.splitlines()
    if name == "point":
        declaration = next(
            index for index, line in enumerate(lines) if line.startswith("POINTS ")
        )
        lines[declaration + 1] = "9 0 0"
    elif name == "connectivity":
        declaration = next(
            index for index, line in enumerate(lines) if line.startswith("CELLS ")
        )
        lines[declaration + 1] = "3 0 1 3"
    elif name == "cell_type":
        declaration = next(
            index for index, line in enumerate(lines) if line.startswith("CELL_TYPES ")
        )
        lines[declaration + 1] = "9"
    elif name == "scalar":
        declaration = lines.index("SCALARS selected_scalar double 1")
        lines[declaration + 2] = "99"
    elif name == "identity":
        return _mutate_first_vector_value(
            serialized,
            "fem_node_id",
            "99",
        )
    else:
        raise AssertionError(f"unknown structural mutation {name}")
    return "\n".join(lines) + "\n"


def _mutate_first_vector_value(
    serialized: str,
    name: str,
    replacement: str,
) -> str:
    lines = serialized.splitlines()
    for index, line in enumerate(lines):
        parts = line.split()
        if parts and parts[0] == name:
            values = lines[index + 1].split()
            values[0] = replacement
            lines[index + 1] = " ".join(values)
            return "\n".join(lines) + "\n"
    raise AssertionError(f"missing vector FIELD array {name}")


def _identity_vector_values(
    serialized: str,
    field_name: str,
    array_name: str,
) -> tuple[str, ...]:
    lines = serialized.splitlines()
    section = lines.index(f"FIELD {field_name} 11")
    for index in range(section + 1, len(lines)):
        parts = lines[index].split()
        if parts and parts[0] == array_name:
            return tuple(lines[index + 1].split())
    raise AssertionError(f"missing {field_name} array {array_name}")


def _mutate_identity_vector_value(
    serialized: str,
    field_name: str,
    array_name: str,
    *,
    value_index: int,
    replacement: str,
) -> str:
    lines = serialized.splitlines()
    section = lines.index(f"FIELD {field_name} 11")
    for index in range(section + 1, len(lines)):
        parts = lines[index].split()
        if parts and parts[0] == array_name:
            values = lines[index + 1].split()
            values[value_index] = replacement
            lines[index + 1] = " ".join(values)
            return "\n".join(lines) + "\n"
    raise AssertionError(f"missing {field_name} array {array_name}")


def _coherent_region_table_tamper(serialized: str) -> str:
    new_region = ResultRegionKey(
        make_result_region_signature(["material_id", 3]),
        make_result_region_signature(
            [
                "section",
                None,
                {"plane_type": "stress", "thickness": 1.0},
            ]
        ),
    )
    region_text = encode_result_region_key(new_region)
    encoded = region_text.encode("utf-8", errors="strict")
    lines = serialized.splitlines()

    metadata = lines.index("FIELD ResultMetadata 24")
    lines[metadata] = "FIELD ResultMetadata 25"
    region_count = lines.index("region_count 1 1 long")
    lines[region_count + 1] = "3"
    point_data = next(
        index
        for index in range(region_count + 1, len(lines))
        if lines[index].startswith("POINT_DATA ")
    )
    lines[point_data:point_data] = [
        f"region_2_utf8 1 {len(encoded)} unsigned_char",
        " ".join(str(byte) for byte in encoded),
    ]
    tampered = "\n".join(lines) + "\n"
    return _mutate_identity_vector_value(
        tampered,
        "PointIdentity",
        "region_index",
        value_index=-1,
        replacement="2",
    )


def test_result_vtk_public_api_is_exported_from_fem_io() -> None:
    public_names = (
        "RESULT_VTK_FORMAT_NAME",
        "RESULT_VTK_SCHEMA_VERSION",
        "RESULT_VTK_TITLE",
        "ResultVtkDecodeError",
        "ResultVtkEmptySelectionError",
        "ResultVtkEncodeError",
        "ResultVtkError",
        "ResultVtkLocationIdentity",
        "ResultVtkReadback",
        "dumps_result_vtk",
        "read_result_vtk",
        "write_result_vtk",
    )

    for name in public_names:
        assert getattr(fem_io, name) is getattr(result_vtk_module, name)
        assert name in fem_io.__all__


@pytest.mark.parametrize(
    (
        "variable",
        "position",
        "policy",
        "association",
        "value_layout",
    ),
    (
        (
            ResultVariable.U,
            FieldPosition.NODE,
            None,
            FieldAssociation.NODE,
            ResultValueLayout.POINT,
        ),
        (
            ResultVariable.S,
            FieldPosition.CENTROID,
            None,
            FieldAssociation.ELEMENT,
            ResultValueLayout.CELL,
        ),
        (
            ResultVariable.S,
            FieldPosition.INTEGRATION_POINT,
            None,
            FieldAssociation.INTEGRATION_POINT,
            ResultValueLayout.POINT,
        ),
        (
            ResultVariable.S,
            FieldPosition.ELEMENT_NODAL,
            None,
            FieldAssociation.ELEMENT_NODE,
            ResultValueLayout.POINT,
        ),
        (
            ResultVariable.S,
            FieldPosition.NODE_REGION,
            None,
            FieldAssociation.NODE_REGION,
            ResultValueLayout.POINT,
        ),
        (
            ResultVariable.S,
            FieldPosition.RESOLVED_NODAL,
            NodalAveragingPolicy(100.0),
            FieldAssociation.RESOLVED_NODAL,
            ResultValueLayout.POINT,
        ),
    ),
)
def test_round_trip_covers_every_neutral_topology_association(
    tmp_path: Path,
    variable: ResultVariable,
    position: FieldPosition,
    policy: NodalAveragingPolicy | None,
    association: FieldAssociation,
    value_layout: ResultValueLayout,
) -> None:
    export = _export(variable, position, policy=policy)
    scale = 1.75
    expected = project_scalar_field_topology(export, scale)
    serialized = dumps_result_vtk(export, scale)
    target = tmp_path / f"{position.value}.vtk"
    _write_serialized(target, serialized)

    readback = read_result_vtk(target)
    expected_cell_types = tuple(
        (1 if kind is ResultCellKind.SAMPLE_VERTEX else vtk_cell_type(element_type))
        for kind, element_type in zip(
            expected.cell_kinds,
            expected.canonical_element_types,
            strict=True,
        )
    )

    assert serialized.splitlines()[:4] == [
        "# vtk DataFile Version 3.0",
        RESULT_VTK_TITLE,
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
    ]
    assert f"POINTS {len(expected.points)} double" in serialized
    assert "SCALARS selected_scalar double 1" in serialized
    assert "\r" not in serialized
    assert serialized.endswith("\n")
    assert export.source.result_id not in serialized
    assert readback.source == export.source
    assert readback.materialization_generation == export.materialization_generation
    assert readback.selection == export.selection
    assert readback.quantity is export.field.descriptor.quantity
    assert readback.association is association
    assert readback.deformation_scale == scale
    assert readback.points == tuple(
        tuple(float(component) for component in point) for point in expected.points
    )
    assert readback.cells == expected.cells
    assert readback.cell_types == expected_cell_types
    assert readback.values == tuple(float(value) for value in expected.values)
    assert readback.value_layout is value_layout
    assert readback.point_locations == tuple(
        _identity(location) for location in expected.point_locations
    )
    assert readback.cell_locations == tuple(
        _identity(location) for location in expected.cell_locations
    )
    assert all(
        0 <= point_index < len(readback.points)
        for cell in readback.cells
        for point_index in cell
    )


def test_model_connectivity_is_zero_based_and_not_fem_node_ids(
    tmp_path: Path,
) -> None:
    export = _export(ResultVariable.U, FieldPosition.NODE)
    target = tmp_path / "node.vtk"

    write_result_vtk(target, export)
    readback = read_result_vtk(target)

    assert export.topology.connectivity[0] == (1, 2, 3)
    assert readback.cells[0] == (0, 1, 2)
    assert readback.cell_types == (5, 5, 5)


def test_node_only_topology_round_trips_point_values_with_zero_cells(
    tmp_path: Path,
) -> None:
    export = _export(ResultVariable.U, FieldPosition.NODE)
    topology = ResultTopologyProjection(
        source=export.source,
        node_ids=export.topology.node_ids,
        node_coordinates=export.topology.node_coordinates,
        nodal_displacements=export.topology.nodal_displacements,
        element_ids=(),
        element_types=(),
        connectivity=(),
        element_region_keys=(),
    )
    snapshot = ResultMaterializationSnapshot(
        source=export.source,
        generation=export.materialization_generation,
        topology=topology,
        fields=(export.field,),
    )
    node_only = prepare_result_export_snapshot(
        snapshot,
        export.selection,
    )
    target = tmp_path / "node-only.vtk"

    serialized = dumps_result_vtk(node_only)
    write_result_vtk(target, node_only)
    readback = read_result_vtk(target)

    assert "CELLS 0 0\nCELL_TYPES 0\n" in serialized
    assert "CELL_DATA 0\nFIELD CellIdentity 11\n" in serialized
    assert readback.value_layout is ResultValueLayout.POINT
    assert len(readback.values) == len(export.field.locations)
    assert readback.cells == ()
    assert readback.cell_types == ()
    assert readback.cell_locations == ()


def test_region_table_and_three_state_averaged_identity_round_trip(
    tmp_path: Path,
) -> None:
    node_region = _export(
        ResultVariable.S,
        FieldPosition.NODE_REGION,
    )
    resolved = _export(
        ResultVariable.S,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(100.0),
    )

    for name, export in (("region", node_region), ("resolved", resolved)):
        target = tmp_path / f"{name}.vtk"
        write_result_vtk(target, export)
        readback = read_result_vtk(target)
        encoded = tuple(
            encode_result_region_key(region) for region in readback.region_table
        )
        referenced = {
            identity.region_key
            for identity in (readback.point_locations + readback.cell_locations)
            if identity is not None and identity.region_key is not None
        }

        assert encoded == tuple(sorted(set(encoded)))
        assert set(readback.region_table) == referenced

    region_readback = read_result_vtk(tmp_path / "region.vtk")
    resolved_readback = read_result_vtk(tmp_path / "resolved.vtk")
    assert {
        identity.averaged
        for identity in region_readback.point_locations
        if identity is not None
    } == {None}
    assert {
        identity.averaged
        for identity in resolved_readback.point_locations
        if identity is not None
    } == {False, True}


def test_coherent_region_table_count_string_and_index_tamper_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _export(ResultVariable.S, FieldPosition.NODE_REGION)
    serialize = result_vtk_module._serialize_result_vtk
    tampered = _coherent_region_table_tamper(dumps_result_vtk(export))
    readable = tmp_path / "coherent-region.vtk"
    _write_serialized(readable, tampered)

    readback = read_result_vtk(readable)

    assert len(readback.region_table) == 3
    assert readback.point_locations[-1] is not None
    assert readback.point_locations[-1].region_key == readback.region_table[2]

    target = tmp_path / "region-target.vtk"
    target.write_bytes(b"old-result")
    monkeypatch.setattr(
        result_vtk_module,
        "_serialize_result_vtk",
        lambda projected: _coherent_region_table_tamper(serialize(projected)),
    )

    with pytest.raises(
        ResultVtkEncodeError,
        match="semantic verification",
    ):
        write_result_vtk(target, export)

    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize(
    ("position", "field_name", "validity_name"),
    (
        (
            FieldPosition.NODE,
            "PointIdentity",
            "fem_node_id_valid",
        ),
        (
            FieldPosition.CENTROID,
            "CellIdentity",
            "fem_element_id_valid",
        ),
    ),
)
def test_point_and_cell_identity_validity_mask_tamper_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    position: FieldPosition,
    field_name: str,
    validity_name: str,
) -> None:
    variable = ResultVariable.U if position is FieldPosition.NODE else ResultVariable.S
    export = _export(variable, position)
    serialize = result_vtk_module._serialize_result_vtk

    def tampered_serialize(projected: Any) -> str:
        return _mutate_identity_vector_value(
            serialize(projected),
            field_name,
            validity_name,
            value_index=0,
            replacement="0",
        )

    readable = tmp_path / f"{field_name}.vtk"
    _write_serialized(
        readable, tampered_serialize(result_vtk_module._project_result_vtk(export, 0.0))
    )
    with pytest.raises(ResultVtkDecodeError):
        read_result_vtk(readable)

    target = tmp_path / f"{field_name}-target.vtk"
    target.write_bytes(b"old-result")
    monkeypatch.setattr(
        result_vtk_module,
        "_serialize_result_vtk",
        tampered_serialize,
    )
    with pytest.raises(ResultVtkDecodeError):
        write_result_vtk(target, export)

    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_averaged_state_wire_distinguishes_missing_false_and_true(
    tmp_path: Path,
) -> None:
    node = _export(ResultVariable.U, FieldPosition.NODE)
    resolved = _export(
        ResultVariable.S,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(100.0),
    )
    node_wire = dumps_result_vtk(node)
    resolved_wire = dumps_result_vtk(resolved)

    assert set(
        _identity_vector_values(
            node_wire,
            "PointIdentity",
            "averaged_state",
        )
    ) == {"0"}
    assert set(
        _identity_vector_values(
            resolved_wire,
            "PointIdentity",
            "averaged_state",
        )
    ) == {"1", "2"}

    for name, serialized in (("node", node_wire), ("resolved", resolved_wire)):
        _write_serialized(tmp_path / f"{name}.vtk", serialized)
    assert {
        identity.averaged
        for identity in read_result_vtk(tmp_path / "node.vtk").point_locations
        if identity is not None
    } == {None}
    assert {
        identity.averaged
        for identity in read_result_vtk(tmp_path / "resolved.vtk").point_locations
        if identity is not None
    } == {False, True}


def test_policy_gauss_order_and_recovery_contract_are_wire_distinct(
    tmp_path: Path,
) -> None:
    raw = _export(
        ResultVariable.S,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(0.0),
    )
    averaged = _export(
        ResultVariable.S,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(100.0),
    )
    integration = _export(
        ResultVariable.S,
        FieldPosition.INTEGRATION_POINT,
    )
    gauss_key = FieldMaterializationKey(
        FieldRequest(
            integration.selection.field_key.request.field_id,
            gauss_order=2,
        ),
        integration.selection.field_key.recovery_contract,
    )
    gauss_two = _with_key(integration, gauss_key)
    changed_contract = _with_key(
        integration,
        FieldMaterializationKey(
            integration.selection.field_key.request,
            integration.selection.field_key.recovery_contract + 1,
        ),
    )

    variants = (raw, averaged, integration, gauss_two, changed_contract)
    serialized = tuple(dumps_result_vtk(export) for export in variants)

    assert len(set(serialized)) == len(serialized)
    for index, export in enumerate(variants):
        target = tmp_path / f"variant-{index}.vtk"
        _write_serialized(target, serialized[index])
        assert read_result_vtk(target).selection == export.selection


def test_wire_uses_ascii_utf8_bytes_and_round_trip_safe_doubles(
    tmp_path: Path,
) -> None:
    export = _export(ResultVariable.U, FieldPosition.NODE)
    serialized = dumps_result_vtk(export)
    target = tmp_path / "wire.vtk"
    _write_serialized(target, serialized)

    assert serialized.isascii()
    assert RESULT_VTK_FORMAT_NAME not in serialized
    assert f"schema 1 1 int\n{RESULT_VTK_SCHEMA_VERSION}\n" in serialized
    assert "0.10000000000000001" in serialized
    assert read_result_vtk(target).source == export.source


@pytest.mark.parametrize(
    "mutation",
    ("header", "crlf", "final-newline", "noncanonical-float"),
)
def test_read_requires_canonical_legacy_ascii_wire(
    tmp_path: Path,
    mutation: str,
) -> None:
    export = _export(ResultVariable.U, FieldPosition.NODE)
    serialized = dumps_result_vtk(export)
    if mutation == "header":
        raw = serialized.replace(
            "# vtk DataFile Version 3.0",
            "# vtk DataFile Version 2.0",
            1,
        ).encode("ascii")
    elif mutation == "crlf":
        raw = serialized.replace("\n", "\r\n").encode("ascii")
    elif mutation == "final-newline":
        raw = serialized.removesuffix("\n").encode("ascii")
    else:
        raw = serialized.replace("\n0 0 0\n", "\n0.0 0 0\n", 1).encode("ascii")
    target = tmp_path / f"{mutation}.vtk"
    target.write_bytes(raw)

    with pytest.raises(ResultVtkDecodeError):
        read_result_vtk(target)


def test_zero_rows_are_rejected_before_parent_or_temp_creation(
    tmp_path: Path,
) -> None:
    export = _export(ResultVariable.U, FieldPosition.NODE)
    empty_field = FieldData(
        descriptor=export.field.descriptor,
        source=export.source,
        key=export.field.key,
        locations=(),
        values=np.empty(
            (0, len(export.field.descriptor.columns)),
            dtype=float,
        ),
    )
    snapshot = ResultMaterializationSnapshot(
        source=export.source,
        generation=export.materialization_generation,
        topology=export.topology,
        fields=(empty_field,),
    )
    empty_export = prepare_result_export_snapshot(
        snapshot,
        export.selection,
    )
    parent = tmp_path / "not-created"
    target = parent / "empty.vtk"

    with pytest.raises(ResultVtkEmptySelectionError):
        dumps_result_vtk(empty_export)
    with pytest.raises(ResultVtkEmptySelectionError):
        write_result_vtk(target, empty_export)

    assert not parent.exists()


def test_unpaired_surrogate_is_typed_and_rejected_before_temp_creation(
    tmp_path: Path,
) -> None:
    export = _export(
        ResultVariable.U,
        FieldPosition.NODE,
        source=_source(result_id="bad-\ud800-result"),
    )
    parent = tmp_path / "not-created"
    target = parent / "surrogate.vtk"

    with pytest.raises(ResultVtkEncodeError, match="Unicode"):
        dumps_result_vtk(export)
    with pytest.raises(ResultVtkEncodeError, match="Unicode"):
        write_result_vtk(target, export)

    assert not parent.exists()


def test_atomic_write_observes_old_target_at_once_only_final_hook(
    tmp_path: Path,
) -> None:
    export = _export(ResultVariable.U, FieldPosition.NODE)
    target = tmp_path / "result.vtk"
    target.write_bytes(b"old-result")
    observations: list[bytes] = []
    checkpoints: list[int] = []

    returned = write_result_vtk(
        target,
        export,
        checkpoint=lambda: checkpoints.append(len(checkpoints) + 1),
        before_replace=lambda: observations.append(target.read_bytes()),
    )

    assert returned == target
    assert checkpoints == [1, 2, 3, 4, 5]
    assert observations == [b"old-result"]
    assert read_result_vtk(target).selection == export.selection
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize("cancel_at", (1, 2, 3, 4, 5))
def test_cancellation_at_each_checkpoint_preserves_old_target(
    tmp_path: Path,
    cancel_at: int,
) -> None:
    export = _export(ResultVariable.U, FieldPosition.NODE)
    target = tmp_path / "result.vtk"
    target.write_bytes(b"old-result")
    calls = 0

    def checkpoint() -> None:
        nonlocal calls
        calls += 1
        if calls == cancel_at:
            raise RuntimeError(f"cancelled at checkpoint {cancel_at}")

    with pytest.raises(
        RuntimeError,
        match=f"cancelled at checkpoint {cancel_at}",
    ):
        write_result_vtk(target, export, checkpoint=checkpoint)

    assert calls == cancel_at
    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_late_cancel_after_replace_is_not_observed_and_completion_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _export(ResultVariable.U, FieldPosition.NODE)
    target = tmp_path / "late-cancel.vtk"
    target.write_bytes(b"old-result")
    shared_writer = result_vtk_module.atomic_write_verified_text
    real_replace = shared_writer.__globals__["os"].replace
    pending = False
    checkpoint_calls = 0

    def checkpoint() -> None:
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        if pending:
            raise RuntimeError("late cancellation must not be observed")

    def replace_then_cancel(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        nonlocal pending
        real_replace(source, destination)
        pending = True

    def injected_writer(*args: Any, **kwargs: Any) -> Path:
        kwargs["replace_func"] = replace_then_cancel
        return shared_writer(*args, **kwargs)

    monkeypatch.setattr(
        result_vtk_module,
        "atomic_write_verified_text",
        injected_writer,
    )

    returned = write_result_vtk(
        target,
        export,
        checkpoint=checkpoint,
    )

    assert returned == target
    assert pending is True
    assert checkpoint_calls == 5
    assert read_result_vtk(target).selection == export.selection
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


class _FaultingTextStream:
    def __init__(self, stream: Any, stage: str) -> None:
        self._stream = stream
        self._stage = stage

    def write(self, value: str) -> int:
        if self._stage == "write":
            raise OSError("injected write failure")
        return self._stream.write(value)

    def flush(self) -> None:
        if self._stage == "flush":
            raise OSError("injected flush failure")
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def close(self) -> None:
        self._stream.close()


@pytest.mark.parametrize("stage", ("write", "flush", "fsync", "replace"))
def test_atomic_fault_preserves_old_target_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    export = _export(ResultVariable.U, FieldPosition.NODE)
    target = tmp_path / "result.vtk"
    target.write_bytes(b"old-result")
    shared_writer = result_vtk_module.atomic_write_verified_text

    def injected_writer(*args: Any, **kwargs: Any) -> Path:
        if stage in {"write", "flush"}:
            default_fdopen = shared_writer.__globals__["os"].fdopen

            def faulting_fdopen(
                descriptor: int,
                *open_args: Any,
                **open_kwargs: Any,
            ) -> _FaultingTextStream:
                stream = default_fdopen(
                    descriptor,
                    *open_args,
                    **open_kwargs,
                )
                return _FaultingTextStream(stream, stage)

            kwargs["_fdopen_func"] = faulting_fdopen
        elif stage == "fsync":

            def fail_fsync(_descriptor: int) -> None:
                raise OSError("injected fsync failure")

            kwargs["_fsync_func"] = fail_fsync
        else:

            def fail_replace(
                _source: str | Path,
                _destination: str | Path,
            ) -> None:
                raise OSError("injected replace failure")

            kwargs["replace_func"] = fail_replace
        return shared_writer(*args, **kwargs)

    monkeypatch.setattr(
        result_vtk_module,
        "atomic_write_verified_text",
        injected_writer,
    )

    with pytest.raises(OSError, match=f"injected {stage} failure"):
        write_result_vtk(target, export)

    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_readback_failure_preserves_old_target_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _export(ResultVariable.U, FieldPosition.NODE)
    target = tmp_path / "result.vtk"
    target.write_bytes(b"old-result")

    def fail_readback(_path: str | Path) -> Any:
        raise OSError("injected readback failure")

    monkeypatch.setattr(
        result_vtk_module,
        "read_result_vtk",
        fail_readback,
    )

    with pytest.raises(OSError, match="injected readback failure"):
        write_result_vtk(target, export)

    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_semantic_readback_mismatch_preserves_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _export(ResultVariable.U, FieldPosition.NODE)
    target = tmp_path / "result.vtk"
    target.write_bytes(b"old-result")
    reader = result_vtk_module.read_result_vtk

    def drifted_readback(path: str | Path):
        readback = reader(path)
        return replace(
            readback,
            materialization_generation=(readback.materialization_generation + 1),
        )

    monkeypatch.setattr(
        result_vtk_module,
        "read_result_vtk",
        drifted_readback,
    )

    with pytest.raises(
        ResultVtkEncodeError,
        match="semantic verification",
    ):
        write_result_vtk(target, export)

    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize(
    ("metadata_name", "replacement"),
    (
        ("model_revision", "8"),
        ("materialization_generation", "99"),
        ("averaging_threshold_percent", "25"),
        ("gauss_order", "3"),
        ("recovery_contract", "99"),
        ("deformation_scale", "2"),
    ),
)
def test_complete_semantics_tamper_is_caught_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_name: str,
    replacement: str,
) -> None:
    if metadata_name == "averaging_threshold_percent":
        export = _export(
            ResultVariable.S,
            FieldPosition.RESOLVED_NODAL,
            policy=NodalAveragingPolicy(0.0),
        )
    elif metadata_name == "gauss_order":
        base = _export(
            ResultVariable.S,
            FieldPosition.INTEGRATION_POINT,
        )
        export = _with_key(
            base,
            FieldMaterializationKey(
                FieldRequest(
                    base.selection.field_key.request.field_id,
                    gauss_order=2,
                ),
                base.selection.field_key.recovery_contract,
            ),
        )
    else:
        export = _export(ResultVariable.U, FieldPosition.NODE)
    target = tmp_path / f"tamper-{metadata_name}.vtk"
    target.write_bytes(b"old-result")
    serialize = result_vtk_module._serialize_result_vtk

    def tampered_serialize(projected: Any) -> str:
        return _mutate_field_values(
            serialize(projected),
            metadata_name,
            replacement,
        )

    monkeypatch.setattr(
        result_vtk_module,
        "_serialize_result_vtk",
        tampered_serialize,
    )

    with pytest.raises(
        ResultVtkEncodeError,
        match="semantic verification",
    ):
        write_result_vtk(target, export)

    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize(
    "metadata_name",
    (
        "format_utf8",
        "schema",
        "result_id_utf8",
        "session_id_utf8",
        "artifact_id_utf8",
        "run_id_utf8",
        "step_name_utf8",
        "field_variable_utf8",
        "field_position_utf8",
        "component_utf8",
        "averaging_policy_present",
        "averaging_preserve_region_boundaries",
        "gauss_order_present",
        "field_quantity_utf8",
        "field_association_utf8",
    ),
)
def test_remaining_metadata_tamper_is_typed_and_preserves_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_name: str,
) -> None:
    if metadata_name.startswith("averaging_"):
        export = _export(
            ResultVariable.S,
            FieldPosition.RESOLVED_NODAL,
            policy=NodalAveragingPolicy(0.0),
        )
    elif metadata_name == "gauss_order_present":
        base = _export(
            ResultVariable.S,
            FieldPosition.INTEGRATION_POINT,
        )
        export = _with_key(
            base,
            FieldMaterializationKey(
                FieldRequest(
                    base.selection.field_key.request.field_id,
                    gauss_order=2,
                ),
                base.selection.field_key.recovery_contract,
            ),
        )
    elif metadata_name == "field_quantity_utf8":
        export = _export(
            ResultVariable.S,
            FieldPosition.CENTROID,
        )
    else:
        export = _export(ResultVariable.U, FieldPosition.NODE)
    target = tmp_path / f"tamper-{metadata_name}.vtk"
    target.write_bytes(b"old-result")
    serialize = result_vtk_module._serialize_result_vtk

    def tampered_serialize(projected: Any) -> str:
        serialized = serialize(projected)
        string_replacements = {
            "format_utf8": "changed-format",
            "result_id_utf8": "changed-result",
            "session_id_utf8": "changed-session",
            "artifact_id_utf8": "changed-artifact",
            "run_id_utf8": "changed-run",
            "step_name_utf8": "changed-step",
            "field_variable_utf8": "INVALID_VARIABLE",
            "field_position_utf8": "invalid_position",
            "component_utf8": "U2",
            "field_quantity_utf8": "strain",
            "field_association_utf8": "element_node",
        }
        if metadata_name in string_replacements:
            return _mutate_utf8_field(
                serialized,
                metadata_name,
                string_replacements[metadata_name],
            )
        if metadata_name == "schema":
            return _mutate_field_values(serialized, "schema", "2")
        if metadata_name == "averaging_policy_present":
            return _mutate_field_values(
                serialized,
                metadata_name,
                "0",
            )
        if metadata_name == "averaging_preserve_region_boundaries":
            original = projected.selection.field_key.request.averaging_policy
            assert original is not None
            replacement = "0" if original.preserve_region_boundaries else "1"
            return _mutate_field_values(
                serialized,
                metadata_name,
                replacement,
            )
        assert metadata_name == "gauss_order_present"
        serialized = _mutate_field_values(
            serialized,
            "gauss_order_present",
            "0",
        )
        return _mutate_field_values(serialized, "gauss_order", "0")

    monkeypatch.setattr(
        result_vtk_module,
        "_serialize_result_vtk",
        tampered_serialize,
    )

    with pytest.raises(ResultVtkError):
        write_result_vtk(target, export)

    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize(
    "semantic_part",
    ("point", "connectivity", "cell_type", "scalar", "identity"),
)
def test_topology_value_and_identity_tamper_fails_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    semantic_part: str,
) -> None:
    export = _export(ResultVariable.U, FieldPosition.NODE)
    target = tmp_path / f"tamper-{semantic_part}.vtk"
    target.write_bytes(b"old-result")
    serialize = result_vtk_module._serialize_result_vtk

    def tampered_serialize(projected: Any) -> str:
        return _mutate_structural_value(
            serialize(projected),
            semantic_part,
        )

    monkeypatch.setattr(
        result_vtk_module,
        "_serialize_result_vtk",
        tampered_serialize,
    )

    with pytest.raises(ResultVtkError):
        write_result_vtk(target, export)

    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_canonical_codec_has_no_csv_or_optional_runtime_dependencies() -> None:
    module_path = Path(result_vtk_module.__file__)
    syntax = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(syntax)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(syntax)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    forbidden = (
        "fem.io.result_csv",
        "fem.post.vtk.export",
        "fem_gui",
        "pyvista",
        "vtkmodules",
    )

    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported_modules
        for prefix in forbidden
    )


def test_vtk_write_never_calls_csv_or_legacy_from_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fem.io.result_csv as result_csv_module
    import fem.post.vtk.export as legacy_vtk_export

    def forbidden_call(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("VTK export must not call CSV adapters")

    for name in (
        "dumps_result_csv",
        "read_result_csv",
        "write_result_csv",
    ):
        monkeypatch.setattr(result_csv_module, name, forbidden_call)
    monkeypatch.setattr(
        legacy_vtk_export,
        "from_csv",
        forbidden_call,
    )
    export = _export(ResultVariable.U, FieldPosition.NODE)
    target = tmp_path / "no-csv.vtk"

    write_result_vtk(target, export)

    assert read_result_vtk(target).selection == export.selection


def test_fresh_base_only_process_writes_and_reads_with_optional_imports_blocked(
    tmp_path: Path,
) -> None:
    target = tmp_path / "base-only.vtk"
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        for name in ("pyvista", "vtkmodules", "PySide6", "PyQt6"):
            sys.modules[name] = None

        from fem.application.results import (
            FieldPosition,
            FieldRequest,
            ResultFieldId,
            ResultSourceKey,
            ResultVariable,
            ScalarFieldSelection,
            build_result_provider,
            prepare_result_export_snapshot,
        )
        from fem.io.result_vtk import read_result_vtk, write_result_vtk
        from tests.helpers.phase8_result_characterization import (
            make_continuum_nodal_semantics_result,
        )

        source = ResultSourceKey(
            result_id="base-result",
            session_id="base-session",
            artifact_id="base-artifact",
            model_revision=1,
            step_name="base-step",
            run_id="base-run",
        )
        provider = build_result_provider(
            source,
            make_continuum_nodal_semantics_result(),
        )
        key = provider.resolve_request(
            FieldRequest(
                ResultFieldId(ResultVariable.U, FieldPosition.NODE)
            )
        )
        export = prepare_result_export_snapshot(
            provider.snapshot,
            ScalarFieldSelection(key, "U1"),
        )
        target = Path(sys.argv[1])
        write_result_vtk(target, export)
        readback = read_result_vtk(target)
        assert readback.source == source
        assert readback.selection == export.selection
        assert len(readback.values) == len(export.field.locations)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(target)],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert target.exists()
    assert (
        read_result_vtk(target).selection
        == _export(
            ResultVariable.U,
            FieldPosition.NODE,
            source=ResultSourceKey(
                result_id="base-result",
                session_id="base-session",
                artifact_id="base-artifact",
                model_revision=1,
                step_name="base-step",
                run_id="base-run",
            ),
        ).selection
    )
