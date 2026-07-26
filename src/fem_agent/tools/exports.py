"""Run-confined CSV/VTK result export with artifact hashing."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import secrets
import shutil
import tempfile
from typing import Any

from fem import post

from ..diagnostics import (
    DiagnosticCode,
    exception_diagnostic,
    has_errors,
    make_diagnostic,
)
from ..schemas import (
    ArtifactRecord,
    Diagnostic,
    ExportFormat,
    ResourceLimits,
)


_SOURCE = "fem.exports"


@dataclass(frozen=True)
class ExportOutcome:
    """Actual committed export artifacts plus normalized diagnostics."""

    artifacts: tuple[ArtifactRecord, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def ok(self) -> bool:
        return not has_errors(self.diagnostics)


def export_results(
    result: Any,
    formats: Iterable[ExportFormat | str],
    *,
    run_id: str,
    run_directory: str | Path,
    exports_directory: str | Path,
    resource_limits: ResourceLimits | None = None,
) -> ExportOutcome:
    """Export into one pre-created ``<run>/exports`` directory.

    Generation happens in a temporary child directory.  Files are committed
    only after every output has passed path, count, and byte-limit checks.
    Existing destination files are never overwritten.
    """

    limits = resource_limits or ResourceLimits()
    committed: list[Path] = []
    run_path: Path | None = None
    exports_path: Path | None = None
    try:
        requested = _normalize_formats(formats)
        if not requested:
            raise _ExportFailure("at least one export format is required")
        _validate_run_id(run_id)
        run_path, exports_path = _resolve_export_boundary(
            run_directory,
            exports_directory,
        )
        prefix = f"result-{run_id}"

        with _temporary_export_directory(exports_path) as staging:
            _generate_exports(result, requested, staging, prefix)
            staged_files = _validated_staged_files(staging, limits)
            destinations = tuple(exports_path / path.name for path in staged_files)
            collisions = [path.name for path in destinations if path.exists()]
            if collisions:
                raise _ExportFailure(
                    f"export destination already exists: {collisions[0]}"
                )

            metadata = tuple(
                (path, _sha256(path), path.stat().st_size)
                for path in staged_files
            )
            try:
                for staged, destination in zip(staged_files, destinations):
                    staged.rename(destination)
                    committed.append(destination)
            except Exception:
                _remove_committed_files(committed)
                committed.clear()
                raise

        artifacts = tuple(
            _artifact_record(
                run_id=run_id,
                path=destination,
                sha256=digest,
                size_bytes=size_bytes,
            )
            for destination, (_, digest, size_bytes) in zip(
                committed,
                metadata,
            )
        )
        return ExportOutcome(artifacts=artifacts)
    except Exception as error:
        if committed:
            _remove_committed_files(committed)
        return ExportOutcome(
            diagnostics=(
                make_diagnostic(
                    DiagnosticCode.EXPORT_FAILED,
                    _safe_export_error(error, run_path, exports_path),
                    source=_SOURCE,
                    remediation=(
                        "Use the active run exports directory and ensure the "
                        "requested format is supported."
                    ),
                ),
            )
        )


@contextmanager
def _temporary_export_directory(exports_path: Path) -> Iterator[Path]:
    """Create a private staging child that remains usable in Windows sandboxes."""

    if os.name != "nt":
        with tempfile.TemporaryDirectory(
            prefix=".fem-agent-export-",
            dir=exports_path,
        ) as temporary:
            yield Path(temporary).resolve(strict=True)
        return

    # Python 3.13 translates mode 0o700 into a restrictive Windows ACL.
    # A managed worker token may then be unable to reopen its own temporary
    # directory. The already-confined exports parent supplies the security
    # boundary, so inherit its ACL for this short-lived staging child.
    staging: Path | None = None
    for _attempt in range(100):
        candidate = exports_path / (
            f".fem-agent-export-{secrets.token_hex(8)}"
        )
        try:
            candidate.mkdir(mode=0o777)
        except FileExistsError:
            continue
        staging = candidate.resolve(strict=True)
        break
    if staging is None:
        raise _ExportFailure(
            "could not allocate a unique export staging directory"
        )

    try:
        yield staging
    finally:
        shutil.rmtree(staging)


def _normalize_formats(
    formats: Iterable[ExportFormat | str],
) -> tuple[ExportFormat, ...]:
    normalized: list[ExportFormat] = []
    for value in formats:
        try:
            item = value if isinstance(value, ExportFormat) else ExportFormat(value)
        except (TypeError, ValueError) as error:
            raise _ExportFailure(f"unsupported export format {value!r}") from error
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _resolve_export_boundary(
    run_directory: str | Path,
    exports_directory: str | Path,
) -> tuple[Path, Path]:
    raw_run = Path(run_directory)
    raw_exports = Path(exports_directory)
    if not raw_run.is_absolute() or not raw_exports.is_absolute():
        raise _ExportFailure("run and exports directories must be absolute")
    if ".." in raw_run.parts or ".." in raw_exports.parts:
        raise _ExportFailure("run and exports directories must already be resolved")
    if raw_run.is_symlink() or raw_exports.is_symlink():
        raise _ExportFailure("run and exports directories must not be symbolic links")
    try:
        run_path = raw_run.resolve(strict=True)
        exports_path = raw_exports.resolve(strict=True)
    except OSError as error:
        raise _ExportFailure(
            "the active run and exports directories must already exist"
        ) from error
    if not run_path.is_dir() or not exports_path.is_dir():
        raise _ExportFailure("run and exports paths must be directories")
    if exports_path != run_path / "exports":
        raise _ExportFailure(
            "exports directory must be the active run's direct exports child"
        )
    return run_path, exports_path


def _validate_run_id(run_id: str) -> None:
    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 100
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in run_id
        )
    ):
        raise _ExportFailure(
            "run_id may contain only lowercase letters, digits, underscores, and hyphens"
        )


def _generate_exports(
    result: Any,
    formats: tuple[ExportFormat, ...],
    staging: Path,
    prefix: str,
) -> None:
    if ExportFormat.VTK in formats:
        post.vtk.export.from_result(
            result,
            output_dir=staging,
            name=prefix,
            overwrite=False,
        )
        return
    if ExportFormat.CSV in formats:
        _generate_csv_bundle(result, staging, prefix)


def _generate_csv_bundle(result: Any, staging: Path, prefix: str) -> None:
    mesh = result.model.mesh
    post.displacement.export.nodal(
        mesh,
        result.U,
        staging / f"{prefix}_nodal_displacement.csv",
    )

    try:
        type_keys = post.stress.dispatch.resolve_type_keys(mesh, None)
        post.stress.dispatch.stress_group_for_keys(type_keys)
    except ValueError:
        return

    if post.stress.dispatch.element_stress_supported(type_keys):
        _write_element_stress_csv(
            type_keys,
            mesh,
            result.U,
            staging / f"{prefix}_element_stress.csv",
        )
    if post.stress.dispatch.nodal_stress_supported(type_keys):
        nodal_path = staging / f"{prefix}_nodal_stress.csv"
        if type_keys == ("beam2",):
            post.stress.export.nodal_from_result(result, nodal_path)
        elif len(type_keys) == 1:
            post.stress.nodal.by_type(
                type_keys[0],
                mesh,
                result.U,
                nodal_path,
            )
        else:
            post.stress.nodal.mixed(
                type_keys,
                mesh,
                result.U,
                nodal_path,
            )


def _write_element_stress_csv(
    type_keys: tuple[str, ...],
    mesh: Any,
    displacement: Any,
    path: Path,
) -> None:
    """Write current element stress without entering deprecated wrappers."""
    if len(type_keys) == 1:
        post.stress.element.by_type(
            type_keys[0],
            mesh,
            displacement,
            path,
        )
        return
    post.stress.element.mixed(type_keys, mesh, displacement, path)


def _validated_staged_files(
    staging: Path,
    limits: ResourceLimits,
) -> tuple[Path, ...]:
    entries = tuple(sorted(staging.iterdir(), key=lambda path: path.name))
    if not entries:
        raise _ExportFailure("the exporter produced no files")
    files: list[Path] = []
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise _ExportFailure("the exporter produced an unsupported filesystem entry")
        resolved = path.resolve(strict=True)
        if resolved.parent != staging:
            raise _ExportFailure("an exported file escaped the staging directory")
        files.append(resolved)
    if len(files) > limits.max_output_files:
        raise _ExportFailure(
            f"export produced {len(files)} files; limit is {limits.max_output_files}"
        )
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > limits.max_output_bytes:
        raise _ExportFailure(
            f"export produced {total_bytes} bytes; limit is {limits.max_output_bytes}"
        )
    return tuple(files)


def _artifact_record(
    *,
    run_id: str,
    path: Path,
    sha256: str,
    size_bytes: int,
) -> ArtifactRecord:
    identity = hashlib.sha256(
        f"{run_id}\0{path.name}\0{sha256}".encode("utf-8")
    ).hexdigest()[:24]
    return ArtifactRecord(
        artifact_id=f"artifact-{identity}",
        kind=path.suffix.lstrip(".").casefold() or "result",
        display_path=f"exports/{path.name}",
        sha256=sha256,
        size_bytes=int(size_bytes),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_committed_files(paths: Iterable[Path]) -> None:
    for path in reversed(tuple(paths)):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _safe_export_error(
    error: Exception,
    run_path: Path | None,
    exports_path: Path | None,
) -> str:
    if isinstance(error, _ExportFailure):
        message = str(error)
    elif isinstance(error, ValueError):
        message = exception_diagnostic(
            DiagnosticCode.EXPORT_FAILED,
            error,
            source=_SOURCE,
        ).message
    else:
        message = f"{type(error).__name__}: result export could not be completed"
    for path in (run_path, exports_path):
        if path is not None:
            message = message.replace(str(path), "<run>")
    if len(message) > 1200:
        message = message[:1197] + "..."
    return f"Result export failed: {message}"


class _ExportFailure(RuntimeError):
    pass


export_result_artifacts = export_results


__all__ = [
    "ExportOutcome",
    "export_result_artifacts",
    "export_results",
]
