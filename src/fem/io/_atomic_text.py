"""Shared crash-safe installation for semantically verified text files."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, TextIO, TypeVar


_VerifiedT = TypeVar("_VerifiedT")


def atomic_write_verified_text(
    path: str | Path,
    serialized: str,
    *,
    verifier: Callable[[Path], _VerifiedT],
    semantic_encoder: Callable[[_VerifiedT], Any],
    expected_semantic: Any,
    error_type: type[Exception] = ValueError,
    mismatch_message: str = "temporary file semantic verification failed",
    checkpoint: Callable[[], Any] | None = None,
    before_replace: Callable[[], Any] | None = None,
    replace_func: Callable[[str | Path, str | Path], Any] | None = None,
    unlink_func: Callable[[Path], Any] | None = None,
    _mkstemp_func: Callable[..., tuple[int, str]] | None = None,
    _fdopen_func: Callable[..., TextIO] | None = None,
    _fsync_func: Callable[[int], Any] | None = None,
    _close_func: Callable[[int], Any] | None = None,
) -> Path:
    """Durably write, semantically verify, then atomically install text.

    ``checkpoint`` is observed immediately before write, readback, semantic
    comparison, and replacement.  ``before_replace`` remains the final,
    once-only hook immediately before the atomic commit.  Once replacement
    succeeds, completion wins and neither callback is observed again.  Cleanup
    failures are attached to the primary exception instead of masking it.
    """

    if not isinstance(serialized, str):
        raise TypeError("serialized must be str")
    if not isinstance(error_type, type) or not issubclass(
        error_type,
        Exception,
    ):
        raise TypeError("error_type must be an Exception type")
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable or None")
    if before_replace is not None and not callable(before_replace):
        raise TypeError("before_replace must be callable or None")
    try:
        serialized.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise error_type(
            "serialized text must be valid strict UTF-8"
        ) from error

    expected = deepcopy(expected_semantic)

    def write_serialized(stream: TextIO) -> None:
        stream.write(serialized)

    def compare_verified(verified: _VerifiedT) -> None:
        actual_semantic = semantic_encoder(verified)
        if actual_semantic != expected:
            raise error_type(mismatch_message)

    return _atomic_write_text_transaction(
        path,
        writer=write_serialized,
        verifier=verifier,
        compare_verified=compare_verified,
        checkpoint=checkpoint,
        before_replace=before_replace,
        replace_func=replace_func,
        unlink_func=unlink_func,
        _mkstemp_func=_mkstemp_func,
        _fdopen_func=_fdopen_func,
        _fsync_func=_fsync_func,
        _close_func=_close_func,
    )


def atomic_write_verified_text_stream(
    path: str | Path,
    writer: Callable[[TextIO], Any],
    *,
    error_type: type[Exception] = ValueError,
    mismatch_message: str = "temporary file byte verification failed",
    checkpoint: Callable[[], Any] | None = None,
    before_replace: Callable[[], Any] | None = None,
    replace_func: Callable[[str | Path, str | Path], Any] | None = None,
    unlink_func: Callable[[Path], Any] | None = None,
    _mkstemp_func: Callable[..., tuple[int, str]] | None = None,
    _fdopen_func: Callable[..., TextIO] | None = None,
    _fsync_func: Callable[[int], Any] | None = None,
    _close_func: Callable[[int], Any] | None = None,
) -> Path:
    """Stream UTF-8 text into a verified, atomically installed file.

    The temporary file is hashed while text is emitted and read back in
    bounded chunks after ``fsync``.  This retains byte-for-byte verification
    without materializing either the complete text or its readback in memory.
    """

    if not callable(writer):
        raise TypeError("writer must be callable")
    if not isinstance(error_type, type) or not issubclass(
        error_type,
        Exception,
    ):
        raise TypeError("error_type must be an Exception type")
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable or None")
    if before_replace is not None and not callable(before_replace):
        raise TypeError("before_replace must be callable or None")

    expected_digest: bytes | None = None

    def write_and_hash(stream: TextIO) -> None:
        nonlocal expected_digest
        digest = hashlib.sha256()
        writer(_DigestingTextWriter(stream, digest, error_type))
        expected_digest = digest.digest()

    def compare_digest(actual_digest: bytes) -> None:
        if expected_digest is None or actual_digest != expected_digest:
            raise error_type(mismatch_message)

    return _atomic_write_text_transaction(
        path,
        writer=write_and_hash,
        verifier=_sha256_file,
        compare_verified=compare_digest,
        checkpoint=checkpoint,
        before_replace=before_replace,
        replace_func=replace_func,
        unlink_func=unlink_func,
        _mkstemp_func=_mkstemp_func,
        _fdopen_func=_fdopen_func,
        _fsync_func=_fsync_func,
        _close_func=_close_func,
    )


class _DigestingTextWriter:
    """Minimal text stream proxy that hashes successfully written UTF-8."""

    __slots__ = ("_stream", "_digest", "_error_type")

    def __init__(
        self,
        stream: TextIO,
        digest: Any,
        error_type: type[Exception],
    ) -> None:
        self._stream = stream
        self._digest = digest
        self._error_type = error_type

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError("streamed text must be str")
        try:
            encoded = value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise self._error_type(
                "serialized text must be valid strict UTF-8"
            ) from error
        written = self._stream.write(value)
        if written != len(value):
            raise OSError("short text write")
        self._digest.update(encoded)
        return written


def _sha256_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.digest()


def _atomic_write_text_transaction(
    path: str | Path,
    *,
    writer: Callable[[TextIO], Any],
    verifier: Callable[[Path], _VerifiedT],
    compare_verified: Callable[[_VerifiedT], Any],
    checkpoint: Callable[[], Any] | None,
    before_replace: Callable[[], Any] | None,
    replace_func: Callable[[str | Path, str | Path], Any] | None,
    unlink_func: Callable[[Path], Any] | None,
    _mkstemp_func: Callable[..., tuple[int, str]] | None,
    _fdopen_func: Callable[..., TextIO] | None,
    _fsync_func: Callable[[int], Any] | None,
    _close_func: Callable[[int], Any] | None,
) -> Path:
    target = Path(path)
    replace = os.replace if replace_func is None else replace_func
    unlink = (
        (lambda temporary: temporary.unlink())
        if unlink_func is None
        else unlink_func
    )
    mkstemp = tempfile.mkstemp if _mkstemp_func is None else _mkstemp_func
    fdopen = os.fdopen if _fdopen_func is None else _fdopen_func
    fsync = os.fsync if _fsync_func is None else _fsync_func
    close = os.close if _close_func is None else _close_func

    descriptor = -1
    temporary: Path | None = None
    installed = False
    primary_error: BaseException | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        stream = fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        )
        descriptor = -1
        stream_error: BaseException | None = None
        try:
            _invoke_checkpoint(checkpoint)
            writer(stream)
            stream.flush()
            fsync(stream.fileno())
        except BaseException as error:
            stream_error = error
            raise
        finally:
            try:
                stream.close()
            except BaseException as cleanup_error:
                if stream_error is None:
                    raise
                _add_cleanup_note(
                    stream_error,
                    cleanup_error,
                    action="close temporary text file",
                    temporary=temporary,
                )

        _invoke_checkpoint(checkpoint)
        verified = verifier(temporary)
        _invoke_checkpoint(checkpoint)
        compare_verified(verified)
        _invoke_checkpoint(checkpoint)
        if before_replace is not None:
            before_replace()

        replace(temporary, target)
        installed = True
        return target
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if descriptor >= 0:
            try:
                close(descriptor)
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                _add_cleanup_note(
                    primary_error,
                    cleanup_error,
                    action="close temporary text file descriptor",
                    temporary=temporary,
                )
        if temporary is not None and not installed:
            try:
                unlink(temporary)
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                _add_cleanup_note(
                    primary_error,
                    cleanup_error,
                    action="delete temporary text file",
                    temporary=temporary,
                )


def _invoke_checkpoint(checkpoint: Callable[[], Any] | None) -> None:
    if checkpoint is not None:
        checkpoint()


def _add_cleanup_note(
    primary_error: BaseException,
    cleanup_error: BaseException,
    *,
    action: str,
    temporary: Path | None,
) -> None:
    location = "<not-created>" if temporary is None else str(temporary)
    primary_error.add_note(
        f"{action} failed; temporary path {location}; "
        f"{type(cleanup_error).__name__}: {cleanup_error}"
    )


__all__ = [
    "atomic_write_verified_text",
    "atomic_write_verified_text_stream",
]
