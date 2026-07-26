from __future__ import annotations

import ast
import csv
from dataclasses import replace
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fem.application.results import (
    FieldAssociation,
    FieldData,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultMaterializationSnapshot,
    ResultQuery,
    ResultQueryResult,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
    advance_materialization,
    build_result_provider,
    evaluate_result_query,
    prepare_result_export_snapshot,
)
from fem.io import (
    RESULT_CSV_FORMAT_NAME,
    RESULT_CSV_HEADER,
    RESULT_CSV_SCHEMA_VERSION,
    ResultCsvDecodeError,
    ResultCsvEmptySelectionError,
    ResultCsvEncodeError,
    ResultCsvError,
    dumps_result_csv,
    read_result_csv,
    write_result_csv,
)
from fem.io import result_csv as result_csv_module
from fem.post.averaging import NodalAveragingPolicy
from fem.post.fields import encode_result_region_key
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)


def _source(suffix: str = "canonical") -> ResultSourceKey:
    return ResultSourceKey(
        result_id=f"result,结果-{suffix}",
        session_id=f"session-{suffix}",
        artifact_id=f"artifact-{suffix}",
        model_revision=7,
        step_name="载荷步-1",
        run_id=f"run-{suffix}",
    )


def _export(
    variable: ResultVariable,
    position: FieldPosition,
    *,
    policy: NodalAveragingPolicy | None = None,
    gauss_order: int | None = None,
):
    provider = build_result_provider(
        _source(),
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
    selection = ScalarFieldSelection(key, component)
    return (
        prepare_result_export_snapshot(snapshot, selection),
        snapshot,
    )


def _with_key(export, key: FieldMaterializationKey):
    field_data = FieldData(
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
        fields=(field_data,),
    )
    return prepare_result_export_snapshot(
        snapshot,
        ScalarFieldSelection(key, export.selection.component),
    )


def _with_unit(export, unit_label: str):
    descriptor = replace(export.field.descriptor, unit_label=unit_label)
    field_data = FieldData(
        descriptor=descriptor,
        source=export.source,
        key=export.field.key,
        locations=export.field.locations,
        values=export.field.values,
    )
    snapshot = ResultMaterializationSnapshot(
        source=export.source,
        generation=export.materialization_generation,
        topology=export.topology,
        fields=(field_data,),
    )
    return prepare_result_export_snapshot(snapshot, export.selection)


def _write_serialized(path: Path, serialized: str) -> None:
    path.write_bytes(serialized.encode("utf-8"))


def _rows(serialized: str) -> list[list[str]]:
    assert serialized.startswith("\ufeff")
    return list(
        csv.reader(
            io.StringIO(serialized.removeprefix("\ufeff"), newline=""),
            strict=True,
        )
    )


def _mutate(
    serialized: str,
    column: str,
    value: str,
    *,
    data_row: int | None = 0,
) -> str:
    rows = _rows(serialized)
    column_index = rows[0].index(column)
    if data_row is None:
        for row in rows[1:]:
            row[column_index] = value
    else:
        rows[data_row + 1][column_index] = value
    output = io.StringIO(newline="")
    output.write("\ufeff")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue()


@pytest.mark.parametrize(
    ("variable", "position", "policy", "association"),
    (
        (
            ResultVariable.U,
            FieldPosition.NODE,
            None,
            FieldAssociation.NODE,
        ),
        (
            ResultVariable.S,
            FieldPosition.CENTROID,
            None,
            FieldAssociation.ELEMENT,
        ),
        (
            ResultVariable.S,
            FieldPosition.INTEGRATION_POINT,
            None,
            FieldAssociation.INTEGRATION_POINT,
        ),
        (
            ResultVariable.S,
            FieldPosition.ELEMENT_NODAL,
            None,
            FieldAssociation.ELEMENT_NODE,
        ),
        (
            ResultVariable.S,
            FieldPosition.NODE_REGION,
            None,
            FieldAssociation.NODE_REGION,
        ),
        (
            ResultVariable.S,
            FieldPosition.RESOLVED_NODAL,
            NodalAveragingPolicy(100.0),
            FieldAssociation.RESOLVED_NODAL,
        ),
    ),
)
def test_round_trip_covers_every_result_association(
    tmp_path: Path,
    variable: ResultVariable,
    position: FieldPosition,
    policy: NodalAveragingPolicy | None,
    association: FieldAssociation,
) -> None:
    export, _snapshot = _export(variable, position, policy=policy)
    serialized = dumps_result_csv(export)
    target = tmp_path / f"{position.value}.csv"
    _write_serialized(target, serialized)

    readback = read_result_csv(target)
    rows = _rows(serialized)

    assert serialized.encode("utf-8").startswith(b"\xef\xbb\xbf")
    assert "\r" not in serialized
    assert serialized.endswith("\n")
    assert tuple(rows[0]) == RESULT_CSV_HEADER
    assert readback.source == export.source
    assert (
        readback.materialization_generation
        == export.materialization_generation
    )
    assert readback.selection == export.selection
    assert readback.quantity == export.field.descriptor.quantity
    assert readback.association is association
    assert readback.unit_label == export.field.descriptor.unit_label
    assert tuple(record.location for record in readback.records) == tuple(
        replace(location, displacement=None)
        for location in export.field.locations
    )


def test_fixed_metadata_and_unicode_unit_are_serialized_on_every_row(
    tmp_path: Path,
) -> None:
    export, _snapshot = _export(
        ResultVariable.S,
        FieldPosition.CENTROID,
    )
    export = _with_unit(export, "兆帕,等效")
    serialized = dumps_result_csv(export)
    rows = _rows(serialized)
    header = rows[0]

    assert len(rows) > 2
    for row in rows[1:]:
        projected = dict(zip(header, row, strict=True))
        assert projected["format"] == RESULT_CSV_FORMAT_NAME
        assert projected["schema"] == str(RESULT_CSV_SCHEMA_VERSION)
        assert projected["result_id"] == export.source.result_id
        assert projected["materialization_generation"] == str(
            export.materialization_generation
        )
        assert projected["component"] == "S11"
        assert projected["quantity"] == "stress"
        assert projected["association"] == "element"
        assert projected["unit"] == "兆帕,等效"

    target = tmp_path / "unit.csv"
    _write_serialized(target, serialized)
    assert read_result_csv(target).unit_label == "兆帕,等效"


def test_region_codec_and_resolved_raw_or_averaged_identity_round_trip(
    tmp_path: Path,
) -> None:
    node_region, _ = _export(
        ResultVariable.S,
        FieldPosition.NODE_REGION,
    )
    raw, _ = _export(
        ResultVariable.S,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(0.0),
    )
    averaged, _ = _export(
        ResultVariable.S,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(100.0),
    )

    node_rows = _rows(dumps_result_csv(node_region))
    region_index = node_rows[0].index("region")
    assert {
        row[region_index] for row in node_rows[1:]
    } == {
        encode_result_region_key(location.region_key)
        for location in node_region.field.locations
    }
    node_one_regions = {
        location.region_key
        for location in node_region.field.locations
        if location.node_id == 1
    }
    assert len(node_one_regions) == 2

    for name, export in (
        ("raw", raw),
        ("averaged", averaged),
    ):
        target = tmp_path / f"{name}.csv"
        write_result_csv(target, export)
        readback = read_result_csv(target)
        assert tuple(
            record.location.averaged for record in readback.records
        ) == tuple(
            location.averaged for location in export.field.locations
        )
        assert all(
            record.location.region_key is not None
            for record in readback.records
        )
    assert {location.averaged for location in raw.field.locations} == {False}
    assert True in {
        location.averaged for location in averaged.field.locations
    }


def test_policy_gauss_order_and_recovery_contract_are_wire_distinct(
    tmp_path: Path,
) -> None:
    raw, _ = _export(
        ResultVariable.S,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(0.0),
    )
    averaged, _ = _export(
        ResultVariable.S,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(100.0),
    )
    gauss_default, _ = _export(
        ResultVariable.S,
        FieldPosition.INTEGRATION_POINT,
    )
    gauss_two = _with_key(
        gauss_default,
        FieldMaterializationKey(
            FieldRequest(
                gauss_default.selection.field_key.request.field_id,
                gauss_order=2,
            ),
            gauss_default.selection.field_key.recovery_contract,
        ),
    )
    changed_contract = _with_key(
        gauss_default,
        FieldMaterializationKey(
            gauss_default.selection.field_key.request,
            gauss_default.selection.field_key.recovery_contract + 1,
        ),
    )

    pairs = (
        (raw, averaged),
        (gauss_default, gauss_two),
        (gauss_default, changed_contract),
    )
    for left, right in pairs:
        assert dumps_result_csv(left) != dumps_result_csv(right)

    for index, export in enumerate(
        (raw, averaged, gauss_default, gauss_two, changed_contract)
    ):
        target = tmp_path / f"variant-{index}.csv"
        _write_serialized(target, dumps_result_csv(export))
        assert read_result_csv(target).selection == export.selection


def test_exact_query_exports_only_the_ordered_snapshot_subset(
    tmp_path: Path,
) -> None:
    export, snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    query = ResultQuery(
        export.selection.field_key,
        export.selection.component,
        node_ids=(8, 1),
    )
    query_result = evaluate_result_query(snapshot, query)
    target = tmp_path / "query.csv"

    write_result_csv(target, export, query_result)
    readback = read_result_csv(target)

    assert tuple(
        record.location.node_id for record in readback.records
    ) == (1, 8)
    assert tuple(record.value for record in readback.records) == tuple(
        record.value for record in query_result.records
    )


def test_query_binding_and_exact_rows_are_revalidated() -> None:
    export, snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    query_result = evaluate_result_query(
        snapshot,
        ResultQuery(
            export.selection.field_key,
            export.selection.component,
            node_ids=(1, 8),
        ),
    )
    foreign = ResultQueryResult(
        source=_source("foreign"),
        materialization_generation=query_result.materialization_generation,
        query=query_result.query,
        records=(),
    )
    wrong_key = FieldMaterializationKey(
        query_result.query.field_key.request,
        query_result.query.field_key.recovery_contract + 1,
    )
    wrong_key_result = ResultQueryResult(
        source=query_result.source,
        materialization_generation=query_result.materialization_generation,
        query=ResultQuery(wrong_key, query_result.query.component),
        records=(),
    )
    wrong_component = ResultQueryResult(
        source=query_result.source,
        materialization_generation=query_result.materialization_generation,
        query=ResultQuery(
            query_result.query.field_key,
            "U2",
        ),
        records=(),
    )

    for invalid in (
        foreign,
        replace(
            query_result,
            materialization_generation=(
                query_result.materialization_generation + 1
            ),
        ),
        wrong_key_result,
        wrong_component,
        replace(query_result, records=query_result.records[:-1]),
    ):
        with pytest.raises(ResultCsvEncodeError):
            dumps_result_csv(export, invalid)


def test_zero_rows_are_rejected_before_parent_or_temp_creation(
    tmp_path: Path,
) -> None:
    export, _snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
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
    empty_snapshot = ResultMaterializationSnapshot(
        source=export.source,
        generation=export.materialization_generation,
        topology=export.topology,
        fields=(empty_field,),
    )
    empty_export = prepare_result_export_snapshot(
        empty_snapshot,
        export.selection,
    )
    parent = tmp_path / "not-created"
    target = parent / "empty.csv"

    with pytest.raises(ResultCsvEmptySelectionError):
        write_result_csv(target, empty_export)

    assert not parent.exists()


def test_empty_exact_query_is_rejected() -> None:
    export, snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    empty = evaluate_result_query(
        snapshot,
        ResultQuery(
            export.selection.field_key,
            export.selection.component,
            node_ids=(1,),
            element_ids=(1,),
        ),
    )
    assert empty.records == ()

    with pytest.raises(ResultCsvEmptySelectionError):
        dumps_result_csv(export, empty)


def test_zero_row_query_is_rejected_before_parent_or_temp_creation(
    tmp_path: Path,
) -> None:
    export, snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    empty = evaluate_result_query(
        snapshot,
        ResultQuery(
            export.selection.field_key,
            export.selection.component,
            node_ids=(1,),
            element_ids=(1,),
        ),
    )
    parent = tmp_path / "query-parent"
    target = parent / "empty-query.csv"

    with pytest.raises(ResultCsvEmptySelectionError):
        write_result_csv(target, export, empty)

    assert not parent.exists()


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("format", "legacy-result"),
        ("schema", "01"),
        ("model_revision", "07"),
        ("materialization_generation", "-1"),
        ("recovery_contract", "0"),
        ("x", "1.0"),
        ("value", "nan"),
    ),
)
def test_read_rejects_tampered_or_noncanonical_scalar_metadata(
    tmp_path: Path,
    column: str,
    value: str,
) -> None:
    export, _snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    target = tmp_path / f"tampered-{column}.csv"
    _write_serialized(
        target,
        _mutate(dumps_result_csv(export), column, value),
    )

    with pytest.raises(ResultCsvDecodeError):
        read_result_csv(target)


def test_read_rejects_noncanonical_region_and_mixed_row_metadata(
    tmp_path: Path,
) -> None:
    export, _snapshot = _export(
        ResultVariable.S,
        FieldPosition.NODE_REGION,
    )
    serialized = dumps_result_csv(export)
    noncanonical = tmp_path / "region.csv"
    mixed = tmp_path / "mixed.csv"
    _write_serialized(
        noncanonical,
        _mutate(
            serialized,
            "region",
            '{"section":{},"material":{}}',
        ),
    )
    _write_serialized(
        mixed,
        _mutate(
            serialized,
            "materialization_generation",
            str(export.materialization_generation + 1),
            data_row=1,
        ),
    )

    with pytest.raises(ResultCsvDecodeError):
        read_result_csv(noncanonical)
    with pytest.raises(ResultCsvDecodeError, match="changes field metadata"):
        read_result_csv(mixed)


@pytest.mark.parametrize("mutation", ("bom", "crlf", "final-newline"))
def test_read_requires_bom_lf_and_final_newline(
    tmp_path: Path,
    mutation: str,
) -> None:
    export, _snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    raw = dumps_result_csv(export).encode("utf-8")
    if mutation == "bom":
        raw = raw.removeprefix(b"\xef\xbb\xbf")
    elif mutation == "crlf":
        raw = raw.replace(b"\n", b"\r\n")
    else:
        raw = raw.removesuffix(b"\n")
    target = tmp_path / f"{mutation}.csv"
    target.write_bytes(raw)

    with pytest.raises(ResultCsvDecodeError):
        read_result_csv(target)


def test_atomic_write_observes_old_target_at_final_checkpoint(
    tmp_path: Path,
) -> None:
    export, _snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    target = tmp_path / "result.csv"
    target.write_bytes(b"old-result")
    observed: list[bytes] = []

    returned = write_result_csv(
        target,
        export,
        before_replace=lambda: observed.append(target.read_bytes()),
    )

    assert returned == target
    assert observed == [b"old-result"]
    assert read_result_csv(target).selection == export.selection
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_cancellation_preserves_old_target_and_cleans_temp(
    tmp_path: Path,
) -> None:
    export, _snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    target = tmp_path / "result.csv"
    target.write_bytes(b"old-result")

    def cancel() -> None:
        raise RuntimeError("cancelled before result commit")

    with pytest.raises(RuntimeError, match="cancelled before result commit"):
        write_result_csv(target, export, before_replace=cancel)

    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize(
    ("cancel_on", "stage"),
    (
        (1, "write"),
        (2, "readback"),
        (3, "compare"),
        (4, "replace"),
    ),
)
def test_public_checkpoint_cancels_each_atomic_stage(
    tmp_path: Path,
    cancel_on: int,
    stage: str,
) -> None:
    export, _snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    target = tmp_path / "result.csv"
    target.write_bytes(b"old-result")
    atomic_calls = 0

    def cancel_at_stage() -> None:
        nonlocal atomic_calls
        if not list(tmp_path.glob(f".{target.name}.*.tmp")):
            return
        atomic_calls += 1
        if atomic_calls == cancel_on:
            raise RuntimeError(f"cancelled before {stage}")

    with pytest.raises(RuntimeError, match=f"cancelled before {stage}"):
        write_result_csv(
            target,
            export,
            checkpoint=cancel_at_stage,
        )

    assert atomic_calls == cancel_on
    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_csv_completion_wins_after_four_atomic_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export, _snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    target = tmp_path / "result.csv"
    target.write_bytes(b"old-result")
    shared_writer = result_csv_module.atomic_write_verified_text
    default_replace = shared_writer.__globals__["os"].replace
    atomic_calls = 0
    final_hooks = 0
    cancellation_pending = False

    def checkpoint() -> None:
        nonlocal atomic_calls
        if cancellation_pending:
            raise RuntimeError("late cancellation must not be observed")
        if list(tmp_path.glob(f".{target.name}.*.tmp")):
            atomic_calls += 1

    def before_replace() -> None:
        nonlocal final_hooks
        final_hooks += 1

    def replace_and_cancel(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        nonlocal cancellation_pending
        default_replace(source, destination)
        cancellation_pending = True

    def injected_writer(*args: Any, **kwargs: Any) -> Path:
        kwargs["replace_func"] = replace_and_cancel
        return shared_writer(*args, **kwargs)

    monkeypatch.setattr(
        result_csv_module,
        "atomic_write_verified_text",
        injected_writer,
    )

    returned = write_result_csv(
        target,
        export,
        checkpoint=checkpoint,
        before_replace=before_replace,
    )

    assert returned == target
    assert atomic_calls == 4
    assert final_hooks == 1
    assert cancellation_pending
    assert read_result_csv(target).selection == export.selection
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_csv_row_serialization_polls_before_filesystem_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export, _snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    parent = tmp_path / "not-created"
    target = parent / "result.csv"
    project = result_csv_module._project_result_csv
    serializing = False
    serialization_calls = 0

    def tracked_project(*args: Any, **kwargs: Any):
        nonlocal serializing
        projected = project(*args, **kwargs)
        serializing = True
        return projected

    def cancel_during_first_row() -> None:
        nonlocal serialization_calls
        if not serializing:
            return
        serialization_calls += 1
        if serialization_calls == 2:
            raise RuntimeError("cancelled during CSV row serialization")

    monkeypatch.setattr(
        result_csv_module,
        "_project_result_csv",
        tracked_project,
    )

    with pytest.raises(
        RuntimeError,
        match="cancelled during CSV row serialization",
    ):
        write_result_csv(
            target,
            export,
            checkpoint=cancel_during_first_row,
        )

    assert serialization_calls == 2
    assert not parent.exists()


def test_invalid_surrogate_is_typed_before_filesystem_creation(
    tmp_path: Path,
) -> None:
    export, _snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    invalid = _with_unit(export, "\ud800")
    parent = tmp_path / "not-created"
    target = parent / "result.csv"

    with pytest.raises(
        ResultCsvEncodeError,
        match="valid strict UTF-8",
    ):
        dumps_result_csv(invalid)
    with pytest.raises(
        ResultCsvEncodeError,
        match="valid strict UTF-8",
    ):
        write_result_csv(target, invalid)

    assert not parent.exists()


@pytest.mark.parametrize("stage", ("fsync", "replace"))
def test_atomic_fault_preserves_old_target_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    export, _snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    target = tmp_path / "result.csv"
    target.write_bytes(b"old-result")
    shared_writer = result_csv_module.atomic_write_verified_text

    def injected_writer(*args: Any, **kwargs: Any) -> Path:
        if stage == "fsync":
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
        result_csv_module,
        "atomic_write_verified_text",
        injected_writer,
    )

    with pytest.raises(OSError, match=f"injected {stage} failure"):
        write_result_csv(target, export)

    assert target.read_bytes() == b"old-result"
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


@pytest.mark.parametrize("stage", ("write", "flush"))
def test_atomic_stream_fault_preserves_old_target_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    export, _snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    target = tmp_path / "result.csv"
    target.write_bytes(b"old-result")
    shared_writer = result_csv_module.atomic_write_verified_text

    def injected_writer(*args: Any, **kwargs: Any) -> Path:
        default_fdopen = shared_writer.__globals__["os"].fdopen

        def faulting_fdopen(
            descriptor: int,
            *open_args: Any,
            **open_kwargs: Any,
        ) -> _FaultingTextStream:
            return _FaultingTextStream(
                default_fdopen(
                    descriptor,
                    *open_args,
                    **open_kwargs,
                ),
                stage,
            )

        kwargs["_fdopen_func"] = faulting_fdopen
        return shared_writer(*args, **kwargs)

    monkeypatch.setattr(
        result_csv_module,
        "atomic_write_verified_text",
        injected_writer,
    )

    with pytest.raises(OSError, match=f"injected {stage} failure"):
        write_result_csv(target, export)

    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_readback_failure_preserves_old_target_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export, _snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    target = tmp_path / "result.csv"
    target.write_bytes(b"old-result")

    def fail_readback(_path: str | Path):
        raise OSError("injected readback failure")

    monkeypatch.setattr(
        result_csv_module,
        "read_result_csv",
        fail_readback,
    )

    with pytest.raises(OSError, match="injected readback failure"):
        write_result_csv(target, export)

    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_semantic_readback_mismatch_preserves_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export, _snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    target = tmp_path / "result.csv"
    target.write_bytes(b"old-result")
    reader = result_csv_module.read_result_csv

    def drifted_readback(path: str | Path):
        readback = reader(path)
        return replace(
            readback,
            materialization_generation=(
                readback.materialization_generation + 1
            ),
        )

    monkeypatch.setattr(
        result_csv_module,
        "read_result_csv",
        drifted_readback,
    )

    with pytest.raises(ResultCsvError, match="semantic verification"):
        write_result_csv(target, export)

    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize(
    ("column", "replacement"),
    (
        ("averaging_threshold_percent", "25"),
        ("gauss_order", "2"),
        ("recovery_contract", "999"),
    ),
)
def test_complete_key_tamper_triggers_semantic_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    replacement: str,
) -> None:
    if column == "averaging_threshold_percent":
        export, _snapshot = _export(
            ResultVariable.S,
            FieldPosition.RESOLVED_NODAL,
            policy=NodalAveragingPolicy(0.0),
        )
    else:
        export, _snapshot = _export(
            ResultVariable.S,
            FieldPosition.INTEGRATION_POINT,
        )
    target = tmp_path / f"tamper-{column}.csv"
    target.write_bytes(b"old-result")
    serialize = result_csv_module._serialize_result_csv

    def tampered_serialize(
        projected,
        *,
        checkpoint=None,
    ) -> str:
        return _mutate(
            serialize(projected, checkpoint=checkpoint),
            column,
            replacement,
            data_row=None,
            )

    monkeypatch.setattr(
        result_csv_module,
        "_serialize_result_csv",
        tampered_serialize,
    )

    with pytest.raises(ResultCsvError, match="semantic verification"):
        write_result_csv(target, export)

    assert target.read_bytes() == b"old-result"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_cleanup_failure_is_attached_to_primary_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export, _snapshot = _export(
        ResultVariable.U,
        FieldPosition.NODE,
    )
    target = tmp_path / "result.csv"
    target.write_bytes(b"old-result")
    shared_writer = result_csv_module.atomic_write_verified_text
    primary = OSError("injected fsync failure")

    def injected_writer(*args: Any, **kwargs: Any) -> Path:
        def fail_fsync(_descriptor: int) -> None:
            raise primary

        def fail_cleanup(path: Path) -> None:
            raise PermissionError(
                f"injected cleanup failure for {path.name}"
            )

        kwargs["_fsync_func"] = fail_fsync
        kwargs["unlink_func"] = fail_cleanup
        return shared_writer(*args, **kwargs)

    monkeypatch.setattr(
        result_csv_module,
        "atomic_write_verified_text",
        injected_writer,
    )

    with pytest.raises(OSError, match="injected fsync failure") as caught:
        write_result_csv(target, export)

    assert caught.value is primary
    assert any(
        "delete temporary text file failed" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert target.read_bytes() == b"old-result"
    remaining = list(tmp_path.glob(f".{target.name}.*.tmp"))
    assert len(remaining) == 1
    remaining[0].unlink()


def test_canonical_codec_has_no_legacy_exporter_dependencies() -> None:
    module_path = Path(result_csv_module.__file__)
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
        "fem.io.csv",
        "fem.post._csv",
        "fem.post.displacement",
        "fem.post.stress",
        "fem_gui",
    )

    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported_modules
        for prefix in forbidden
    )
