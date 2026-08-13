from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pytest

import fem.application.results.archive as archive_contract
import fem.io.result_archive_v1 as archive_codec
from fem.io import (
    ResultArchiveDecodeError,
    encode_result_archive,
    load_result_archive,
)
from fem.application.results.data import FieldData, ResultTopologyProjection
from tests.helpers.phase8_result_characterization import (
    make_beam_field_characterization_result,
    make_continuum_nodal_semantics_result,
)
from tests.io.test_result_archive_v1 import _snapshot
from tests.io.test_result_archive_v1 import (
    _manifest_and_entries,
    _replace_manifest,
    _rewrite,
)


@pytest.fixture
def archive_snapshot():
    return _snapshot(make_continuum_nodal_semantics_result, "phase5-streaming")


def test_path_loader_streams_zip_and_matches_bytes_semantics(
    archive_snapshot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = encode_result_archive(archive_snapshot)
    source = tmp_path / "streamed.femres"
    source.write_bytes(encoded)

    monkeypatch.setattr(
        archive_codec,
        "_read_path_bounded",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("path loading must not materialize the container")
        ),
    )
    loaded = load_result_archive(source).snapshot
    decoded = archive_codec.decode_result_archive_v1(encoded)
    assert archive_codec.encode_result_archive_v1(loaded) == (
        archive_codec.encode_result_archive_v1(decoded)
    )


def test_path_loader_accepts_canonical_npy_v1_and_v2(
    archive_snapshot,
    tmp_path: Path,
) -> None:
    encoded = encode_result_archive(archive_snapshot)
    manifest, entries = _manifest_and_entries(encoded)
    v1_payload = dict(entries)["topology/node_ids.npy"]
    assert np.lib.format.read_magic(BytesIO(v1_payload)) == (1, 0)

    node_ids = np.load(BytesIO(v1_payload), allow_pickle=False)
    v2_stream = BytesIO()
    np.lib.format.write_array(v2_stream, node_ids, version=(2, 0), allow_pickle=False)
    v2_payload = v2_stream.getvalue()
    assert np.lib.format.read_magic(BytesIO(v2_payload)) == (2, 0)
    replacement = []
    for name, payload in entries:
        if name == "topology/node_ids.npy":
            payload = v2_payload
            manifest["arrays"][name]["sha256"] = hashlib.sha256(payload).hexdigest()
        replacement.append((name, payload))
    source = tmp_path / "npy-v2.femres"
    source.write_bytes(_rewrite(_replace_manifest(manifest, replacement)))
    assert load_result_archive(source).snapshot.materialization.fields


def test_public_path_router_parses_manifest_once_before_arrays(
    archive_snapshot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "manifest-once.femres"
    source.write_bytes(encode_result_archive(archive_snapshot))
    original_parse = archive_codec._parse_manifest
    original_read = archive_codec._read_zip_entry_bounded
    parses: list[bytes] = []
    reads: list[str] = []

    def parse_once(data: bytes, **kwargs):
        parses.append(data)
        return original_parse(data, **kwargs)

    def track_read(archive, name: str, limit: int) -> bytes:
        reads.append(name)
        return original_read(archive, name, limit)

    monkeypatch.setattr(archive_codec, "_parse_manifest", parse_once)
    monkeypatch.setattr(archive_codec, "_read_zip_entry_bounded", track_read)
    load_result_archive(source)
    assert len(parses) == 1
    assert reads[0] == archive_codec.MANIFEST_NAME
    assert archive_codec.MANIFEST_NAME not in reads[1:]


def test_path_validation_error_does_not_fallback_to_container_bytes(
    archive_snapshot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "corrupt-stream.femres"
    _manifest, entries = _manifest_and_entries(encode_result_archive(archive_snapshot))
    corrupted: list[tuple[str, bytes]] = []
    changed = False
    for name, payload in entries:
        if not changed and name.endswith(".npy"):
            payload = payload[:-1] + bytes([payload[-1] ^ 1])
            changed = True
        corrupted.append((name, payload))
    assert changed
    source.write_bytes(_rewrite(corrupted))
    monkeypatch.setattr(
        archive_codec,
        "_read_path_bounded",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("corrupt path must not fall back to bytes API")
        ),
    )
    with pytest.raises(ResultArchiveDecodeError, match="SHA-256 checksum mismatch"):
        load_result_archive(source)


def test_path_array_source_is_not_a_raw_bytes_mapping(
    archive_snapshot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "array-source.femres"
    source.write_bytes(encode_result_archive(archive_snapshot))
    with zipfile.ZipFile(source, "r") as archive:
        manifest, arrays = archive_codec._open_archive_zip(
            archive,
            materialize_arrays=False,
        )
        assert manifest["format"] == archive_codec.FORMAT_NAME
        assert isinstance(arrays, archive_codec._ArchiveArrayReader)
        assert not isinstance(arrays, dict)
    monkeypatch.setattr(
        archive_codec._ArchiveArrayReader,
        "__getitem__",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("path decoder must not request raw array bytes")
        ),
    )
    loaded = load_result_archive(source).snapshot
    assert loaded.materialization.fields


def test_path_arrays_are_owned_readonly_and_no_unconditional_array_copy(
    archive_snapshot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "owned-arrays.femres"
    source.write_bytes(encode_result_archive(archive_snapshot))
    original_validate = archive_codec._validate_decoded_array
    copies: list[bool] = []

    def track_validate(*args, **kwargs):
        copies.append(bool(kwargs.get("copy", True)))
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(archive_codec, "_validate_decoded_array", track_validate)
    loaded = load_result_archive(source).snapshot
    assert loaded.materialization.topology._node_coordinates.flags.owndata
    assert not loaded.materialization.topology._node_coordinates.flags.writeable
    assert loaded.materialization.fields[0]._values.flags.owndata
    assert not loaded.materialization.fields[0]._values.flags.writeable
    assert copies
    assert set(copies) == {False}


def test_path_locations_reuse_matrix_finite_validation(
    archive_snapshot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fem.application.results.data as result_data

    source = tmp_path / "validated-location-matrices.femres"
    source.write_bytes(encode_result_archive(archive_snapshot))
    original = result_data._finite_triplet
    calls: list[str] = []

    def track_triplet(value, *, label: str):
        calls.append(label)
        return original(value, label=label)

    monkeypatch.setattr(result_data, "_finite_triplet", track_triplet)
    replace(archive_snapshot.fields[0].locations[0])
    assert calls
    calls.clear()

    loaded = load_result_archive(source).snapshot
    assert loaded.fields
    assert calls == []


def test_path_reader_rejects_unknown_named_region_ids(
    archive_snapshot,
    tmp_path: Path,
) -> None:
    manifest, entries = _manifest_and_entries(encode_result_archive(archive_snapshot))
    manifest["model_projection"]["named_region_node_ids"]["bad"] = [999999999]
    source = tmp_path / "unknown-region.femres"
    source.write_bytes(_rewrite(_replace_manifest(manifest, entries)))
    with pytest.raises(ResultArchiveDecodeError, match="unknown|references"):
        load_result_archive(source)


def test_named_region_mapping_accepts_full_and_subset_ids() -> None:
    allowed = (1, 2, 3)
    assert archive_codec._decode_named_region_mapping(
        {"all": [1, 2, 3]},
        label="named_region_node_ids",
        allowed_ids=allowed,
    ) == {"all": (1, 2, 3)}
    assert archive_codec._decode_named_region_mapping(
        {"subset": [1, 3]},
        label="named_region_node_ids",
        allowed_ids=allowed,
    ) == {"subset": (1, 3)}
    with pytest.raises(ResultArchiveDecodeError, match="unknown|references"):
        archive_codec._decode_named_region_mapping(
            {"bad": [4]},
            label="named_region_node_ids",
            allowed_ids=allowed,
        )


def test_path_reader_preserves_strict_topology_element_type_validation(
    archive_snapshot,
    tmp_path: Path,
) -> None:
    manifest, entries = _manifest_and_entries(encode_result_archive(archive_snapshot))
    manifest["topology"]["element_types"][0] = "   "
    source = tmp_path / "blank-element-type.femres"
    source.write_bytes(_rewrite(_replace_manifest(manifest, entries)))
    with pytest.raises(ResultArchiveDecodeError, match="element_types"):
        load_result_archive(source)


def test_path_reader_preserves_section_point_contract(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(make_beam_field_characterization_result, "section-point-contract")
    manifest, entries = _manifest_and_entries(encode_result_archive(snapshot))
    field = next(
        item
        for item in manifest["fields"]
        if item["key"]["request"]["field_id"].get("section_point_number") is not None
    )
    point_name = field["arrays"]["section_point_number"]
    payloads = dict(entries)
    values = np.load(BytesIO(payloads[point_name]), allow_pickle=False).copy()
    values[:] = 99
    changed = BytesIO()
    np.save(changed, values, allow_pickle=False)
    payloads[point_name] = changed.getvalue()
    manifest["arrays"][point_name]["sha256"] = hashlib.sha256(
        payloads[point_name]
    ).hexdigest()
    source = tmp_path / "section-point-contract.femres"
    source.write_bytes(
        _rewrite(_replace_manifest(manifest, list(payloads.items())))
    )
    with pytest.raises(ResultArchiveDecodeError, match="section point"):
        load_result_archive(source)


@pytest.mark.parametrize(
    ("association", "array_key", "replacement", "message"),
    (
        ("node", "node_id", -1, "required"),
        ("node", "node_id", 0, "positive"),
        ("node", "element_id", 1, "not allowed"),
        ("node_region", "region_index", 999, "region index"),
        ("resolved_nodal", "averaged_mask", 0, "averaged is required"),
    ),
)
def test_path_reader_preserves_vectorized_location_identity_validation(
    archive_snapshot,
    tmp_path: Path,
    association: str,
    array_key: str,
    replacement: int,
    message: str,
) -> None:
    manifest, entries = _manifest_and_entries(
        encode_result_archive(archive_snapshot)
    )
    field = next(
        item
        for item in manifest["fields"]
        if item["descriptor"]["association"] == association
    )
    array_name = field["arrays"][array_key]
    payloads = dict(entries)
    values = np.load(BytesIO(payloads[array_name]), allow_pickle=False).copy()
    values[0] = replacement
    changed = BytesIO()
    np.save(changed, values, allow_pickle=False)
    payloads[array_name] = changed.getvalue()
    manifest["arrays"][array_name]["sha256"] = hashlib.sha256(
        payloads[array_name]
    ).hexdigest()
    source = tmp_path / f"invalid-{association}-{array_key}.femres"
    source.write_bytes(
        _rewrite(_replace_manifest(manifest, list(payloads.items())))
    )

    with pytest.raises(ResultArchiveDecodeError, match=message):
        load_result_archive(source)


def test_topology_semantic_identity_fast_path_and_distinct_value_check(
    archive_snapshot,
) -> None:
    topology = archive_snapshot.materialization.topology
    assert archive_contract._topology_semantically_equal(topology, topology)
    clone = ResultTopologyProjection(
        topology.source,
        topology.node_ids,
        topology._node_coordinates,
        topology._nodal_displacements,
        topology.element_ids,
        topology.element_types,
        topology.connectivity,
        topology.element_region_keys,
    )
    assert clone is not topology
    assert archive_contract._topology_semantically_equal(topology, clone)
    changed_coordinates = np.array(topology._node_coordinates, copy=True)
    changed_coordinates[0, 0] += 1.0
    changed = ResultTopologyProjection(
        topology.source,
        topology.node_ids,
        changed_coordinates,
        topology._nodal_displacements,
        topology.element_ids,
        topology.element_types,
        topology.connectivity,
        topology.element_region_keys,
    )
    assert not archive_contract._topology_semantically_equal(topology, changed)


def test_trusted_transfer_is_private_and_public_constructors_still_copy(
    archive_snapshot,
) -> None:
    topology = archive_snapshot.materialization.topology
    coordinates = np.array(topology._node_coordinates, copy=True, order="C")
    displacements = np.array(topology._nodal_displacements, copy=True, order="C")
    coordinates.setflags(write=False)
    displacements.setflags(write=False)
    public = ResultTopologyProjection(
        topology.source,
        topology.node_ids,
        coordinates,
        displacements,
        topology.element_ids,
        topology.element_types,
        topology.connectivity,
        topology.element_region_keys,
    )
    assert public._node_coordinates is not coordinates
    changed = np.array(coordinates, copy=True)
    changed.setflags(write=True)
    changed[0, 0] += 123.0
    assert public._node_coordinates[0, 0] != changed[0, 0]

    transferred = ResultTopologyProjection._from_owned_arrays(
        topology.source,
        topology.node_ids,
        coordinates,
        displacements,
        topology.element_ids,
        topology.element_types,
        topology.connectivity,
        topology.element_region_keys,
    )
    assert transferred._node_coordinates is coordinates
    assert not transferred._node_coordinates.flags.writeable

    field = archive_snapshot.materialization.fields[0]
    values = np.array(field._values, copy=True, order="C")
    values.setflags(write=False)
    public_field = FieldData(
        field.descriptor,
        field.source,
        field.key,
        field.locations,
        values,
    )
    assert public_field._values is not values
    transferred_field = FieldData._from_owned_values(
        field.descriptor,
        field.source,
        field.key,
        field.locations,
        values,
    )
    assert transferred_field._values is values
    assert not transferred_field._values.flags.writeable
