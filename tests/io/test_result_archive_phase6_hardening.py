from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np
import pytest

from fem.application import ModelSession
from fem.application.results.data import FieldData
from fem.io import (
    ResultArchiveDecodeError,
    ResultArchiveEncodeError,
    decode_result_archive,
    encode_result_archive,
    load_result_archive,
    save_result_archive,
)
from fem.io._atomic_binary import atomic_write_verified_binary
import fem.io.result_archive_v1 as archive_codec
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)
from tests.io.test_result_archive_v1 import (
    _manifest_and_entries,
    _rewrite,
    _snapshot,
)


def _expanded_location(location, copy_index: int):
    """Give each repeated field row a deterministic, unique FEM identity."""

    offset = copy_index * 1_000_000
    values = {}
    if location.node_id is not None:
        values["node_id"] = location.node_id + offset
    if location.element_id is not None:
        values["element_id"] = location.element_id + offset
    return replace(location, **values)


@pytest.fixture(scope="module")
def result_archive():
    return _snapshot(make_continuum_nodal_semantics_result, "phase6-small")


@pytest.fixture(scope="module")
def large_result_archive(result_archive):
    """Deterministic multi-field fixture with tens of thousands of records."""

    base = result_archive
    repetitions = 2_048
    fields: list[FieldData] = []
    for field_index, field in enumerate(base.fields):
        locations = tuple(
            _expanded_location(location, copy_index)
            for copy_index in range(repetitions)
            for location in field.locations
        )
        values = np.tile(field._values, (repetitions, 1))
        # A tiny row marker keeps the fixture deterministic without making
        # every field a highly compressible repeated block.
        values += np.arange(len(locations), dtype=float)[:, None] * (
            (field_index + 1) * 1.0e-9
        )
        fields.append(
            FieldData(
                descriptor=field.descriptor,
                source=field.source,
                key=field.key,
                locations=locations,
                values=values,
            )
        )
    materialization = replace(base.materialization, fields=tuple(fields))
    return replace(base, materialization=materialization)


@pytest.mark.slow
def test_large_archive_roundtrip_records_measurements_and_reuses_arrays(
    large_result_archive,
    tmp_path: Path,
) -> None:
    archive = large_result_archive
    encoded = encode_result_archive(archive)
    assert len(encoded) > 1_000_000

    target = tmp_path / "phase6-large.femres"
    tracemalloc.start()
    started = time.perf_counter()
    save_result_archive(target, archive)
    write_seconds = time.perf_counter() - started
    write_current, write_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    started = time.perf_counter()
    loaded = load_result_archive(target)
    read_seconds = time.perf_counter() - started
    read_current, read_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(loaded.snapshot.fields) == len(archive.fields) == 7
    assert sum(len(field.locations) for field in loaded.snapshot.fields) > 50_000
    np.testing.assert_array_equal(
        loaded.snapshot.fields[-1]._values,
        archive.fields[-1]._values,
    )

    # Result installation and save preparation transfer ownership of the
    # immutable matrices.  They must not invoke a second full-matrix copy.
    session = ModelSession()
    session.replace_from_result_archive(loaded)
    provider = session.current_result_provider()
    assert provider is not None
    prepared = session.prepare_result_archive_save(session.snapshot().selected_run_id)
    assert prepared.archive.topology._node_coordinates is (
        provider.snapshot.topology._node_coordinates
    )
    for prepared_field, provider_field in zip(
        prepared.archive.fields,
        provider.snapshot.fields,
        strict=True,
    ):
        assert prepared_field._values is provider_field._values

    metrics = {
        "archive_bytes": len(encoded),
        "field_count": len(archive.fields),
        "record_count": sum(len(field.locations) for field in archive.fields),
        "write_seconds": write_seconds,
        "read_seconds": read_seconds,
        "write_peak_bytes": write_peak,
        "read_peak_bytes": read_peak,
        "write_current_bytes": write_current,
        "read_current_bytes": read_current,
    }
    print("PHASE6_RESULT_ARCHIVE_METRICS " + json.dumps(metrics, sort_keys=True))
    assert write_seconds >= 0.0
    assert read_seconds >= 0.0


def test_archive_rejects_declared_array_size_overflow(
    result_archive,
    tmp_path: Path,
) -> None:
    manifest, entries = _manifest_and_entries(encode_result_archive(result_archive))
    metadata = manifest["arrays"]["topology/node_ids.npy"]
    metadata["shape"] = [2**60]
    metadata["nbytes"] = 2**60 * 8
    target = tmp_path / "oversized-metadata.femres"
    target.write_bytes(_rewrite([("manifest.json", json.dumps(manifest).encode()), *[(name, value) for name, value in entries if name != "manifest.json"]]))
    with pytest.raises(ResultArchiveDecodeError, match="size|byte"):
        load_result_archive(target)


def test_archive_compression_ratio_guard_is_checked_before_read(
    result_archive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = encode_result_archive(result_archive)
    monkeypatch.setattr(archive_codec, "_MAX_ZIP_COMPRESSION_RATIO", 1)
    with pytest.raises(ResultArchiveDecodeError, match="compression-ratio"):
        decode_result_archive(encoded)


def test_archive_container_limit_rejects_bytes_and_paths_before_zip_parse(
    result_archive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = encode_result_archive(result_archive)
    source = tmp_path / "container-limit.femres"
    source.write_bytes(encoded)
    monkeypatch.setattr(archive_codec, "_MAX_ZIP_CONTAINER_BYTES", len(encoded) - 1)

    def zip_must_not_open(*_args, **_kwargs):
        raise AssertionError("container limit must run before ZipFile")

    monkeypatch.setattr(archive_codec.zipfile, "ZipFile", zip_must_not_open)
    with pytest.raises(ResultArchiveDecodeError, match="container"):
        archive_codec.decode_result_archive_v1(encoded)
    with pytest.raises(ResultArchiveDecodeError, match="container"):
        decode_result_archive(bytearray(encoded))
    with pytest.raises(ResultArchiveDecodeError, match="container"):
        archive_codec.load_result_archive_v1(source)


def test_archive_path_limit_checks_stat_before_open(
    result_archive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = encode_result_archive(result_archive)
    source = tmp_path / "container-stat-limit.femres"
    source.write_bytes(encoded)
    monkeypatch.setattr(archive_codec, "_MAX_ZIP_CONTAINER_BYTES", len(encoded) - 1)

    def path_open_must_not_run(*_args, **_kwargs):
        raise AssertionError("stat size must reject before Path.open")

    monkeypatch.setattr(Path, "open", path_open_must_not_run)
    with pytest.raises(ResultArchiveDecodeError, match="container"):
        archive_codec.load_result_archive_v1(source)

    # A small input should only request its declared size plus one byte from
    # each bounded stream; the safety limit must not become the read size.
    monkeypatch.undo()
    payload = b"small archive payload"
    reads: list[int] = []

    class RecordingStream:
        def __init__(self, stream) -> None:
            self._stream = stream

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            self._stream.close()

        def read(self, size: int = -1) -> bytes:
            reads.append(size)
            return self._stream.read(size)

    original_path_open = Path.open
    small_path = tmp_path / "small.femres"
    small_path.write_bytes(payload)

    def recording_path_open(path, *args, **kwargs):
        return RecordingStream(original_path_open(path, *args, **kwargs))

    monkeypatch.setattr(Path, "open", recording_path_open)
    assert archive_codec._read_path_bounded(small_path, 1024) == payload
    assert reads == [len(payload) + 1]

    reads.clear()

    class Info:
        file_size = len(payload)

    class FakeArchive:
        def getinfo(self, _name):
            return Info()

        def open(self, _name, _mode):
            return RecordingStream(BytesIO(payload))

    assert archive_codec._read_zip_entry_bounded(FakeArchive(), "entry", 1024) == payload
    assert reads == [len(payload) + 1]


def test_archive_entry_count_and_manifest_limits_run_before_manifest_parse(
    result_archive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = encode_result_archive(result_archive)
    monkeypatch.setattr(archive_codec, "_MAX_ZIP_ENTRY_COUNT", 1)
    with pytest.raises(ResultArchiveDecodeError, match="too many ZIP entries"):
        decode_result_archive(encoded)

    monkeypatch.setattr(archive_codec, "_MAX_ZIP_ENTRY_COUNT", 4096)
    monkeypatch.setattr(archive_codec, "_MAX_MANIFEST_BYTES", 1)
    with pytest.raises(ResultArchiveDecodeError, match="manifest"):
        decode_result_archive(encoded)


@pytest.mark.parametrize(
    "limit_name",
    (
        "_MAX_ARRAY_BYTES",
        "_MAX_ZIP_ENTRY_BYTES",
        "_MAX_ZIP_TOTAL_BYTES",
        "_MAX_ZIP_ENTRY_COUNT",
        "_MAX_MANIFEST_BYTES",
        "_MAX_ZIP_CONTAINER_BYTES",
    ),
)
def test_archive_writer_rejects_limits_before_readback(
    result_archive,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
) -> None:
    monkeypatch.setattr(archive_codec, limit_name, 1)
    with pytest.raises(ResultArchiveEncodeError):
        encode_result_archive(result_archive)


def test_archive_writer_checks_compression_ratio_like_reader(
    result_archive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = encode_result_archive(result_archive)
    decode_result_archive(encoded)

    monkeypatch.setattr(archive_codec, "_MAX_ZIP_COMPRESSION_RATIO", 1)
    with pytest.raises(ResultArchiveEncodeError, match="compression-ratio"):
        encode_result_archive(result_archive)


class _FailingStream:
    def __init__(self, stage: str) -> None:
        self.stage = stage

    def write(self, _data: bytes) -> int:
        if self.stage == "write":
            raise OSError("write failure")
        return len(_data)

    def flush(self) -> None:
        if self.stage == "flush":
            raise OSError("flush failure")

    def fileno(self) -> int:
        return 17

    def close(self) -> None:
        if self.stage == "close":
            raise OSError("close failure")


@pytest.mark.parametrize(
    "stage",
    ("create", "open", "write", "flush", "fsync", "readback", "verify", "replace", "cleanup"),
)
def test_atomic_binary_fault_matrix_preserves_target_and_retries(
    stage: str,
    tmp_path: Path,
) -> None:
    target = tmp_path / "fault-injection.femres"
    target.write_bytes(b"old-target")
    temporary_paths: list[Path] = []

    def mkstemp(**_kwargs):
        if stage == "create":
            raise OSError("create failure")
        path = tmp_path / ".fault-injection.femres.test.tmp"
        path.write_bytes(b"")
        temporary_paths.append(path)
        return 17, str(path)

    def fdopen(_descriptor, _mode):
        if stage == "open":
            raise OSError("open failure")
        return _FailingStream(stage)

    def fsync(_descriptor):
        if stage == "fsync":
            raise OSError("fsync failure")

    def verifier(path: Path):
        if stage == "replace":
            return b"new-target"
        if stage in {"readback", "verify"}:
            raise OSError(f"{stage} failure")
        return path.read_bytes()

    def replace_func(_temporary, _target):
        if stage == "replace":
            raise OSError("replace failure")
        raise AssertionError("replace must not run for this injected failure")

    def unlink_func(path: Path):
        if stage == "cleanup":
            raise OSError("cleanup failure")
        path.unlink()

    with pytest.raises((OSError, ValueError)):
        atomic_write_verified_binary(
            target,
            b"new-target",
            verifier=verifier,
            semantic_encoder=lambda value: value,
            expected_semantic=b"new-target",
            replace_func=replace_func,
            unlink_func=unlink_func,
            _mkstemp_func=mkstemp,
            _fdopen_func=fdopen,
            _fsync_func=fsync,
            _close_func=lambda _descriptor: None,
        )
    assert target.read_bytes() == b"old-target"
    if stage == "cleanup":
        assert temporary_paths and temporary_paths[0].exists()
        temporary_paths[0].unlink()
    else:
        assert not temporary_paths or not temporary_paths[0].exists()

    # A failed attempt does not poison the destination: the same target can
    # be retried through the real helper once all injected failures are gone.
    atomic_write_verified_binary(
        target,
        b"new-target",
        verifier=lambda path: path.read_bytes(),
        semantic_encoder=lambda value: value,
        expected_semantic=b"new-target",
    )
    assert target.read_bytes() == b"new-target"
