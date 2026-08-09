from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import warnings
import zipfile

import numpy as np
import pytest

from fem.application.results import (
    FieldData,
    FieldMaterializationKey,
    FieldPosition,
    FieldState,
    ResultArchiveModelProjection,
    ResultArchiveOrigin,
    ResultArchiveRun,
    ResultArchiveSnapshot,
    ResultSourceKey,
    ResultTopologyProjection,
    build_result_provider,
    execute_output_requests,
)
import fem.application.results.archive as archive_contract
from fem.application.units import UnitContext
from fem.core.model import OutputRequest
from fem.io import (
    ResultArchiveDecodeError,
    UnsupportedResultArchiveSchemaError,
    decode_result_archive,
    encode_result_archive,
    load_result_archive,
    save_result_archive,
)
from fem.post.fields import ResultRegionKey, make_result_region_signature
from tests.helpers.phase8_result_characterization import (
    make_beam_field_characterization_result,
    make_continuum_nodal_semantics_result,
    make_truss_field_characterization_result,
)


def _snapshot(builder, name: str) -> ResultArchiveSnapshot:
    source = ResultSourceKey(
        result_id=f"result-{name}",
        session_id="session",
        artifact_id="artifact",
        model_revision=3,
        step_name="Step-1",
        run_id=f"run-{name}",
    )
    provider = build_result_provider(source, builder())
    lazy_keys = tuple(
        item.key
        for item in provider.catalog().fields
        if item.state.value == "lazy"
    )
    if lazy_keys:
        provider = provider.advance(provider.materialize(lazy_keys))
    provider = provider.publish_fields(
        tuple(field_data.key for field_data in provider.snapshot.fields)
    )
    report = execute_output_requests(
        provider,
        (OutputRequest("field", "node", ("U",)),),
    ).report
    now = datetime(2026, 8, 9, tzinfo=timezone.utc)
    units = UnitContext("m", "N", "Pa")
    return ResultArchiveSnapshot(
        archive_id=f"archive-{name}",
        created_at=now,
        producer_version="test",
        origin=ResultArchiveOrigin(
            model_name=f"model-{name}",
            source_basename=f"model-{name}.fempy",
            model_fingerprint="a" * 64,
            provenance={"run_id": source.run_id},
        ),
        run=ResultArchiveRun("job", source.step_name, now, output_report=report),
        profile=provider.profile,
        catalog=provider.catalog(),
        materialization=provider.snapshot,
        model_projection=ResultArchiveModelProjection(
            provider.snapshot.topology,
            unit_context=units,
            named_region_node_ids={"all_nodes": provider.snapshot.topology.node_ids},
            named_region_element_ids={"all_elements": provider.snapshot.topology.element_ids},
            summaries={"model_family": provider.profile.family.value},
        ),
        unit_context=units,
    )


def _entries(data: bytes) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(BytesIO(data)) as archive:
        return [(name, archive.read(name)) for name in archive.namelist()]


def _rewrite(entries: list[tuple[str, bytes]]) -> bytes:
    output = BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in entries:
                archive.writestr(name, payload)
    return output.getvalue()


def _manifest_and_entries(data: bytes) -> tuple[dict[str, object], list[tuple[str, bytes]]]:
    entries = _entries(data)
    manifest = json.loads(dict(entries)["manifest.json"])
    return manifest, entries


def _replace_manifest(manifest: dict[str, object], entries: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    payload = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return [("manifest.json", payload), *[(name, value) for name, value in entries if name != "manifest.json"]]


@pytest.mark.parametrize(
    "name,builder",
    (
        ("continuum", make_continuum_nodal_semantics_result),
        ("truss2", make_truss_field_characterization_result),
        ("beam2", make_beam_field_characterization_result),
    ),
)
def test_schema_v1_roundtrip_preserves_result_contract(name, builder):
    expected = _snapshot(builder, name)
    encoded = encode_result_archive(expected)
    actual = decode_result_archive(encoded)
    assert actual.archive_id == expected.archive_id
    assert actual.source == expected.source
    assert actual.profile == expected.profile
    assert actual.catalog == expected.catalog
    assert actual.origin == expected.origin
    assert actual.run == expected.run
    assert actual.materialization.generation == expected.materialization.generation == 1
    assert actual.unit_context == expected.unit_context
    assert actual.model_projection.named_region_node_ids == expected.model_projection.named_region_node_ids
    assert actual.model_projection.named_region_element_ids == expected.model_projection.named_region_element_ids
    assert actual.model_projection.summaries == expected.model_projection.summaries
    assert actual.topology.node_ids == expected.topology.node_ids
    assert actual.topology.element_types == expected.topology.element_types
    assert actual.topology.connectivity == expected.topology.connectivity
    assert actual.topology.element_region_keys == expected.topology.element_region_keys
    assert len(actual.fields) == len(expected.fields)
    for left, right in zip(actual.fields, expected.fields, strict=True):
        assert left.key == right.key
        assert left.descriptor == right.descriptor
        assert left.locations == right.locations
        np.testing.assert_array_equal(left.values, right.values)
        assert left.values.flags.writeable is False


def test_beam_section_point_archive_roundtrip_preserves_point_identity() -> None:
    expected = _snapshot(make_beam_field_characterization_result, "beam-points")

    encoded = encode_result_archive(expected)
    manifest, _entries_value = _manifest_and_entries(encoded)
    point_entries = tuple(
        field
        for field in manifest["fields"]
        if field["key"]["request"]["field_id"].get("section_point_number")
        is not None
    )
    assert len(point_entries) == 4
    assert all(
        {
            "section_point_number",
            "section_point_local_y",
            "section_point_local_z",
        }
        <= set(field["arrays"])
        for field in point_entries
    )

    actual = decode_result_archive(encoded)
    point_fields = tuple(
        field
        for field in actual.fields
        if field.key.request.field_id.position is FieldPosition.SECTION_POINT
    )

    assert tuple(
        field.key.request.field_id.section_point_number
        for field in point_fields
    ) == (1, 2, 3, 4)
    assert all(
        location.section_point is not None
        and location.section_point.number
        == field.key.request.field_id.section_point_number
        for field in point_fields
        for location in field.locations
    )


def test_legacy_beam_archive_remains_readable_without_fabricated_points() -> None:
    current = _snapshot(make_beam_field_characterization_result, "legacy-beam")
    section = next(
        field
        for field in current.fields
        if field.key.request.field_id.position is FieldPosition.SECTION_END
    )
    legacy_key = FieldMaterializationKey(section.key.request, 1)
    legacy_descriptor = replace(
        section.descriptor,
        derived_components=("S11AbsMax",),
    )
    legacy_section = FieldData(
        descriptor=legacy_descriptor,
        source=section.source,
        key=legacy_key,
        locations=section.locations,
        values=section.values[:, :3],
    )
    legacy_fields = tuple(
        legacy_section if field is section else field
        for field in current.fields
        if field.key.request.field_id.position is not FieldPosition.SECTION_POINT
    )
    legacy_catalog_fields = tuple(
        (
            replace(
                availability,
                key=legacy_key,
                descriptor=legacy_descriptor,
            )
            if availability.key == section.key
            else availability
        )
        for availability in current.catalog.fields
        if availability.key.request.field_id.position
        is not FieldPosition.SECTION_POINT
    )
    legacy = replace(
        current,
        catalog=replace(current.catalog, fields=legacy_catalog_fields),
        materialization=replace(
            current.materialization,
            fields=legacy_fields,
        ),
    )

    loaded = decode_result_archive(encode_result_archive(legacy))

    assert all(
        field.key.request.field_id.position is not FieldPosition.SECTION_POINT
        for field in loaded.fields
    )
    assert all(
        location.section_point is None
        for field in loaded.fields
        for location in field.locations
    )


def test_schema_v1_entry_order_is_deterministic(tmp_path: Path):
    snapshot = _snapshot(make_truss_field_characterization_result, "order")
    first = encode_result_archive(snapshot)
    second = encode_result_archive(snapshot)
    assert first == second
    target = tmp_path / "result.femres"
    save_result_archive(target, snapshot)
    loaded = load_result_archive(target)
    assert loaded.path == target
    assert loaded.source_schema == 1
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
    assert names[0] == "manifest.json"
    assert names[1:9] == [
        "topology/node_ids.npy",
        "topology/node_coordinates.npy",
        "topology/nodal_displacements.npy",
        "topology/element_ids.npy",
        "topology/connectivity_offsets.npy",
        "topology/connectivity.npy",
        "topology/element_type_indices.npy",
        "topology/region_indices.npy",
    ]


def test_schema_v1_rejects_duplicate_unknown_and_checksum_entries(tmp_path: Path):
    snapshot = _snapshot(make_truss_field_characterization_result, "security")
    encoded = encode_result_archive(snapshot)
    # Rebuild a mutated archive while preserving deterministic entry bytes.
    source = tmp_path / "source.femres"
    source.write_bytes(encoded)
    with zipfile.ZipFile(source) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = __import__("json").loads(entries["manifest.json"])
    any_array = next(iter(manifest["arrays"]))
    entries[any_array] = entries[any_array][:-1] + bytes([entries[any_array][-1] ^ 1])
    mutated = tmp_path / "mutated.femres"
    with zipfile.ZipFile(mutated, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    with pytest.raises(ResultArchiveDecodeError, match="checksum|valid"):
        load_result_archive(mutated)

    unknown = tmp_path / "unknown.femres"
    with zipfile.ZipFile(unknown, "w") as archive:
        for name, payload in {**entries, "unknown.bin": b"x"}.items():
            archive.writestr(name, payload)
    with pytest.raises(ResultArchiveDecodeError, match="entries"):
        load_result_archive(unknown)


@pytest.mark.parametrize("dangerous_name", ("../evil.npy", "/absolute.npy", "C:/absolute.npy", "fields\\evil.npy"))
def test_schema_v1_rejects_duplicate_missing_and_dangerous_entries(
    tmp_path: Path,
    dangerous_name: str,
):
    snapshot = _snapshot(make_truss_field_characterization_result, "entry-security")
    original = encode_result_archive(snapshot)
    entries = _entries(original)
    source = tmp_path / "dangerous.femres"
    source.write_bytes(_rewrite([*entries, (dangerous_name, b"evil")]))
    with pytest.raises(ResultArchiveDecodeError):
        load_result_archive(source)

    duplicate = tmp_path / "duplicate.femres"
    duplicate.write_bytes(_rewrite([*entries, entries[1]]))
    with pytest.raises(ResultArchiveDecodeError, match="duplicate"):
        load_result_archive(duplicate)

    missing = tmp_path / "missing.femres"
    missing.write_bytes(_rewrite(entries[:-1]))
    with pytest.raises(ResultArchiveDecodeError, match="missing|entries"):
        load_result_archive(missing)


@pytest.mark.parametrize("metadata_key", ("dtype", "shape", "nbytes"))
def test_schema_v1_rejects_dtype_shape_and_byte_size_mismatch(
    tmp_path: Path,
    metadata_key: str,
):
    snapshot = _snapshot(make_truss_field_characterization_result, f"metadata-{metadata_key}")
    manifest, entries = _manifest_and_entries(encode_result_archive(snapshot))
    array_name = "topology/node_ids.npy"
    metadata = manifest["arrays"][array_name]
    if metadata_key == "dtype":
        metadata[metadata_key] = "<f4"
    elif metadata_key == "shape":
        metadata[metadata_key] = [999]
    else:
        metadata[metadata_key] += 1
    target = tmp_path / f"{metadata_key}.femres"
    target.write_bytes(_rewrite(_replace_manifest(manifest, entries)))
    with pytest.raises(ResultArchiveDecodeError, match="dtype|shape|byte"):
        load_result_archive(target)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda manifest: manifest["model_projection"]["named_region_node_ids"].update(
            {"all_nodes": "not-an-array"}
        ),
        lambda manifest: manifest["model_projection"]["named_region_node_ids"].update(
            {"all_nodes": [1.5]}
        ),
        lambda manifest: manifest["model_projection"]["named_region_node_ids"].update(
            {"all_nodes": [999999]}
        ),
        lambda manifest: manifest["model_projection"]["named_region_node_ids"].update(
            {"all_nodes": [1, 1]}
        ),
    ),
)
def test_schema_v1_rejects_malformed_named_region_projection(
    tmp_path: Path,
    mutation,
):
    snapshot = _snapshot(make_continuum_nodal_semantics_result, "projection-security")
    manifest, entries = _manifest_and_entries(encode_result_archive(snapshot))
    mutation(manifest)
    target = tmp_path / "projection-security.femres"
    target.write_bytes(_rewrite(_replace_manifest(manifest, entries)))
    with pytest.raises(ResultArchiveDecodeError, match="named_region|integer|unknown|duplicate"):
        load_result_archive(target)


@pytest.mark.parametrize("mutation", ("duplicate", "unused", "region-order"))
def test_schema_v1_rejects_dictionary_tampering(tmp_path: Path, mutation: str):
    snapshot = _snapshot(make_continuum_nodal_semantics_result, "dictionary-security")
    manifest, entries = _manifest_and_entries(encode_result_archive(snapshot))
    if mutation == "duplicate":
        manifest["topology"]["element_types"].append(
            manifest["topology"]["element_types"][0]
        )
    elif mutation == "unused":
        manifest["topology"]["element_types"].append("UnusedType")
    else:
        manifest["topology"]["region_keys"].reverse()
    target = tmp_path / f"{mutation}.femres"
    target.write_bytes(_rewrite(_replace_manifest(manifest, entries)))
    with pytest.raises(ResultArchiveDecodeError, match="dictionary|duplicates|order|used"):
        load_result_archive(target)


def test_schema_v1_rejects_unused_region_dictionary_entry(tmp_path: Path):
    snapshot = _snapshot(make_continuum_nodal_semantics_result, "unused-region")
    manifest, entries = _manifest_and_entries(encode_result_archive(snapshot))
    manifest["topology"]["region_keys"].append(
        '{"material":["material_id",99],"section":["section",null,{"plane_type":"stress","thickness":1.0}]}'
    )
    target = tmp_path / "unused-region.femres"
    target.write_bytes(_rewrite(_replace_manifest(manifest, entries)))
    with pytest.raises(ResultArchiveDecodeError, match="exactly the used"):
        load_result_archive(target)


def test_schema_v1_rejects_truncated_corrupt_and_nonfinite_payloads(tmp_path: Path):
    snapshot = _snapshot(make_truss_field_characterization_result, "corrupt")
    encoded = encode_result_archive(snapshot)
    truncated = tmp_path / "truncated.femres"
    truncated.write_bytes(encoded[:-12])
    with pytest.raises(ResultArchiveDecodeError):
        load_result_archive(truncated)
    corrupt = tmp_path / "corrupt.femres"
    corrupt.write_bytes(b"not-a-zip")
    with pytest.raises(ResultArchiveDecodeError):
        load_result_archive(corrupt)

    manifest, entries = _manifest_and_entries(encoded)
    array_name = "topology/node_coordinates.npy"
    values = np.load(BytesIO(dict(entries)[array_name]), allow_pickle=False)
    values[0, 0] = np.nan
    raw = BytesIO()
    np.save(raw, values, allow_pickle=False)
    dict_entries = dict(entries)
    dict_entries[array_name] = raw.getvalue()
    manifest["arrays"][array_name]["sha256"] = hashlib.sha256(raw.getvalue()).hexdigest()
    nonfinite = tmp_path / "nonfinite.femres"
    nonfinite.write_bytes(_rewrite(_replace_manifest(manifest, list(dict_entries.items()))))
    with pytest.raises(ResultArchiveDecodeError, match="non-finite"):
        load_result_archive(nonfinite)

    manifest, entries = _manifest_and_entries(encoded)
    manifest["run"]["timings"]["nonfinite"] = float("nan")
    nonfinite_manifest = tmp_path / "nonfinite-manifest.femres"
    nonfinite_manifest.write_bytes(_rewrite(_replace_manifest(manifest, entries)))
    with pytest.raises(ResultArchiveDecodeError, match="non-finite|JSON"):
        load_result_archive(nonfinite_manifest)


def test_schema_v1_rejects_object_dtype_payload(tmp_path: Path):
    snapshot = _snapshot(make_truss_field_characterization_result, "object")
    manifest, entries = _manifest_and_entries(encode_result_archive(snapshot))
    array_name = "topology/node_ids.npy"
    raw = BytesIO()
    np.save(raw, np.asarray(["not-safe"], dtype=object), allow_pickle=True)
    dict_entries = dict(entries)
    dict_entries[array_name] = raw.getvalue()
    manifest["arrays"][array_name]["dtype"] = "|O"
    manifest["arrays"][array_name]["shape"] = [1]
    manifest["arrays"][array_name]["nbytes"] = np.asarray(["not-safe"], dtype=object).nbytes
    manifest["arrays"][array_name]["sha256"] = hashlib.sha256(raw.getvalue()).hexdigest()
    target = tmp_path / "object.femres"
    target.write_bytes(_rewrite(_replace_manifest(manifest, list(dict_entries.items()))))
    with pytest.raises(ResultArchiveDecodeError, match="object|dtype"):
        load_result_archive(target)


def test_schema_v1_readback_semantic_mismatch_preserves_old_target_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import fem.io.result_archive_v1 as codec

    snapshot = _snapshot(make_truss_field_characterization_result, "semantic")
    target = tmp_path / "semantic.femres"
    target.write_bytes(b"old-target")
    original = codec.encode_result_archive_v1
    calls = 0

    def altered(value):
        nonlocal calls
        calls += 1
        encoded = original(value)
        return encoded if calls == 1 else encoded + b"changed"

    monkeypatch.setattr(codec, "encode_result_archive_v1", altered)
    with pytest.raises(codec.ResultArchiveEncodeError, match="semantic"):
        codec.save_result_archive_v1(target, snapshot)
    assert target.read_bytes() == b"old-target"
    assert tuple(tmp_path.glob(".*.tmp")) == ()


def test_schema_v1_rejects_unsupported_schema_and_preserves_target_on_failure(tmp_path: Path):
    snapshot = _snapshot(make_truss_field_characterization_result, "atomic")
    target = tmp_path / "result.femres"
    target.write_bytes(b"old-target")
    with pytest.raises(RuntimeError, match="cancel"):
        save_result_archive(target, snapshot, checkpoint=lambda: (_ for _ in ()).throw(RuntimeError("cancel")))
    assert target.read_bytes() == b"old-target"
    assert tuple(tmp_path.glob(".*.tmp")) == ()

    encoded = encode_result_archive(snapshot)
    source = tmp_path / "schema.femres"
    source.write_bytes(encoded)
    with zipfile.ZipFile(source) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = __import__("json").loads(entries["manifest.json"])
    manifest["schema"] = 2
    entries["manifest.json"] = __import__("json").dumps(manifest, separators=(",", ":")).encode()
    source.write_bytes(b"")
    with zipfile.ZipFile(source, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    with pytest.raises(UnsupportedResultArchiveSchemaError):
        load_result_archive(source)


def test_version_neutral_router_rejects_future_schema_before_array_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fem.io.result_archive_v1 as codec

    snapshot = _snapshot(make_truss_field_characterization_result, "future-router")
    manifest, entries = _manifest_and_entries(encode_result_archive(snapshot))
    manifest["schema"] = 2
    manifest["future_contract"] = {"frames": 1}
    encoded = _rewrite(_replace_manifest(manifest, entries))
    original_read = codec._read_zip_entry_bounded
    reads: list[str] = []

    def tracking_read(archive, name: str, limit: int) -> bytes:
        reads.append(name)
        return original_read(archive, name, limit)

    monkeypatch.setattr(codec, "_read_zip_entry_bounded", tracking_read)
    with pytest.raises(UnsupportedResultArchiveSchemaError):
        decode_result_archive(encoded)
    assert reads == ["manifest.json"]

    reads.clear()
    source = tmp_path / "future.femres"
    source.write_bytes(encoded)
    with pytest.raises(UnsupportedResultArchiveSchemaError):
        load_result_archive(source)
    assert reads == ["manifest.json"]


def test_schema_v1_wraps_manifest_numeric_overflow_as_typed_decode_error() -> None:
    snapshot = _snapshot(make_truss_field_characterization_result, "overflow")
    manifest, entries = _manifest_and_entries(encode_result_archive(snapshot))
    manifest["run"]["timings"] = {"hostile": 10**400}
    encoded = _rewrite(_replace_manifest(manifest, entries))

    with pytest.raises(ResultArchiveDecodeError, match="finite real number"):
        decode_result_archive(encoded)


def test_snapshot_factory_detaches_an_accepted_result_record():
    from tests.characterization.test_phase0_result_contracts import _session_with_success

    session, _solve = _session_with_success()
    record = session.current_result()
    assert record is not None
    snapshot = ResultArchiveSnapshot.from_result_record(record)
    reopened = decode_result_archive(encode_result_archive(snapshot))
    assert reopened.source == snapshot.source
    assert reopened.origin.model_fingerprint
    session.close()


def test_snapshot_rejects_projection_identity_unit_and_catalog_mismatches():
    snapshot = _snapshot(make_continuum_nodal_semantics_result, "dto-validation")
    topology = snapshot.topology
    coordinates = np.array(topology.node_coordinates, copy=True)
    coordinates[0, 0] += 1.0
    changed_topology = ResultTopologyProjection(
        source=topology.source,
        node_ids=topology.node_ids,
        node_coordinates=coordinates,
        nodal_displacements=topology.nodal_displacements,
        element_ids=topology.element_ids,
        element_types=topology.element_types,
        connectivity=topology.connectivity,
        element_region_keys=topology.element_region_keys,
    )
    with pytest.raises(ValueError, match="topology"):
        replace(
            snapshot,
            model_projection=replace(snapshot.model_projection, topology=changed_topology),
        )

    with pytest.raises(ValueError, match="unit"):
        replace(
            snapshot,
            model_projection=replace(
                snapshot.model_projection,
                unit_context=UnitContext("mm", "N", "Pa"),
            ),
        )

    availability = replace(snapshot.catalog.fields[0], state=FieldState.LAZY)
    with pytest.raises(ValueError, match="READY"):
        replace(snapshot, catalog=replace(snapshot.catalog, fields=(availability, *snapshot.catalog.fields[1:])))

    inherited = replace(
        snapshot,
        unit_context=None,
        model_projection=replace(snapshot.model_projection, unit_context=snapshot.unit_context),
    )
    assert inherited.unit_context == snapshot.unit_context


def test_model_fingerprint_changes_with_profile_regions_and_units():
    snapshot = _snapshot(make_continuum_nodal_semantics_result, "fingerprint")
    baseline = archive_contract._result_model_fingerprint(
        snapshot.topology,
        snapshot.profile,
        step_name=snapshot.source.step_name,
        unit_context=snapshot.unit_context,
    )
    profile = replace(
        snapshot.profile,
        dofs_per_node=3,
        dof_labels=(*snapshot.profile.dof_labels, "U3"),
        force_labels=(*snapshot.profile.force_labels, "Fz"),
    )
    assert archive_contract._result_model_fingerprint(
        snapshot.topology,
        profile,
        step_name=snapshot.source.step_name,
        unit_context=snapshot.unit_context,
    ) != baseline

    region = ResultRegionKey(
        material_signature=make_result_region_signature(["changed", 1]),
        section_signature=snapshot.topology.element_region_keys[0].section_signature,
    )
    regions = (region, *snapshot.topology.element_region_keys[1:])
    changed_topology = ResultTopologyProjection(
        source=snapshot.topology.source,
        node_ids=snapshot.topology.node_ids,
        node_coordinates=snapshot.topology.node_coordinates,
        nodal_displacements=snapshot.topology.nodal_displacements,
        element_ids=snapshot.topology.element_ids,
        element_types=snapshot.topology.element_types,
        connectivity=snapshot.topology.connectivity,
        element_region_keys=regions,
    )
    assert archive_contract._result_model_fingerprint(
        changed_topology,
        snapshot.profile,
        step_name=snapshot.source.step_name,
        unit_context=snapshot.unit_context,
    ) != baseline
    assert archive_contract._result_model_fingerprint(
        snapshot.topology,
        snapshot.profile,
        step_name=snapshot.source.step_name,
        unit_context=UnitContext("mm", "N", "Pa"),
    ) != baseline
