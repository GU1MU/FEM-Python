"""Phase 0 baseline for the multi-document/multi-result workspace plan.

The benchmark deliberately lives outside production code.  It builds small
inline meshes and a temporary result archive, then records the existing
single-document pipeline as stable JSON.  GUI dependencies are optional: a
headless checkout without Qt/PyVista still produces the model, geometry,
inspection, and archive measurements and reports the unavailable GUI metrics.

Examples (from the repository root)::

    python tests/performance/benchmark_multi_document_workspace.py
    python tests/performance/benchmark_multi_document_workspace.py --scenario 5000
    python tests/performance/benchmark_multi_document_workspace.py --scenario archive
    python tests/performance/benchmark_multi_document_workspace.py --output phase0.json

The default run covers the empty document, 5,000 and 20,000 element models,
and a medium temporary result archive.  ``--scenario`` selects exactly one
scenario (or ``all``); the legacy ``--elements`` option selects model sizes
only and therefore does not run the archive.  ``--output`` is optional;
without it the JSON is written to stdout.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import tracemalloc
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator


# A direct ``python tests/performance/...py`` invocation does not put the
# repository root on sys.path.  Keep the script runnable without installing the
# package while remaining a no-op when the root is already present (for pytest
# and ``python -m`` callers).
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

# Qt/VTK are intentionally kept off-screen for this baseline.  This avoids
# creating a native OpenGL surface while still exercising the public viewport
# methods and measuring their call boundary.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FEM_GUI_OFFSCREEN", "1")

import numpy as np

_PROJECT_IMPORT_ERROR: BaseException | None = None
try:
    from fem.application.results import (
        ResultArchiveModelProjection,
        ResultArchiveOrigin,
        ResultArchiveRun,
        ResultArchiveSnapshot,
        ResultSourceKey,
        build_result_provider,
        execute_output_requests,
        result_model_fingerprint,
    )
    from fem.core.model import AnalysisStep, FEMModel, OutputRequest
    from fem.core.mesh import Element2D, Mesh2D, Node2D
    from fem.core.result import ModelResult
    from fem.io import encode_result_archive
    from fem_gui import inspection_service
    from fem_gui.visualization import model_adapter
except Exception as error:  # report missing project runtime without installing it
    _PROJECT_IMPORT_ERROR = error


_FIXED_TIME = datetime(2026, 8, 12, tzinfo=timezone.utc)
_DEFAULT_ELEMENT_COUNTS = (0, 5_000, 20_000)
_DEFAULT_ARCHIVE_ELEMENTS = 5_000


class _CallSpy:
    """Minimal call counter used by test-only monkey patches."""

    __slots__ = ("calls",)

    def __init__(self) -> None:
        self.calls = 0


@contextmanager
def _method_spy(owner: Any, name: str) -> Iterator[_CallSpy]:
    """Wrap one callable for the duration of a measurement.

    The helper is intentionally local to this script.  It changes no
    production source and restores inherited Qt methods exactly after the
    measurement, including methods (such as ``QTreeWidget.clear``) that are
    not present in the subclass ``__dict__``.
    """

    sentinel = object()
    local_value = getattr(owner, "__dict__", {}).get(name, sentinel)
    original = getattr(owner, name)
    spy = _CallSpy()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        spy.calls += 1
        return original(*args, **kwargs)

    setattr(owner, name, wrapped)
    try:
        yield spy
    finally:
        if local_value is sentinel:
            delattr(owner, name)
        else:
            setattr(owner, name, local_value)


def _measure(function: Callable[[], Any]) -> tuple[Any, float, int]:
    """Return value, wall time, and Python allocation peak for one call."""

    gc.collect()
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.stop()
    tracemalloc.start()
    started = perf_counter()
    try:
        value = function()
    finally:
        elapsed = perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if was_tracing:
            tracemalloc.start()
    return value, elapsed, int(peak_bytes)


@contextmanager
def _workspace_temp_directory() -> Iterator[Path]:
    """Create an accessible, exact-scope temporary directory on Windows.

    The managed Windows sandbox applies a restrictive ACL to
    :func:`tempfile.mkdtemp`'s default ``0o700`` mode.  A directory explicitly
    created with ``0o777`` under the repository inherits the workspace ACL and
    remains temporary because this context removes only its generated path.
    """

    directory = _REPOSITORY_ROOT / f".phase0-{uuid.uuid4().hex}"
    directory.mkdir(mode=0o777)
    try:
        yield directory
    finally:
        for child in tuple(directory.iterdir()):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                # The benchmark only creates one archive file.  Keep cleanup
                # bounded to this generated directory if a codec leaves a
                # nested temporary path behind.
                for nested in tuple(child.rglob("*")):
                    if nested.is_file() or nested.is_symlink():
                        nested.unlink()
                for nested in sorted(
                    (item for item in child.rglob("*") if item.is_dir()),
                    reverse=True,
                ):
                    nested.rmdir()
                child.rmdir()
        directory.rmdir()


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _record_measure(
    timings: dict[str, float | None],
    peaks: dict[str, int | None],
    errors: dict[str, str],
    name: str,
    function: Callable[[], Any],
) -> Any | None:
    """Measure a named operation, retaining an error instead of aborting run."""

    try:
        value, elapsed, peak_bytes = _measure(function)
    except Exception as error:  # pragma: no cover - exercised by optional GUI
        timings[f"{name}_seconds"] = None
        peaks[f"{name}_peak_bytes"] = None
        errors[name] = _error_text(error)
        return None
    timings[f"{name}_seconds"] = float(elapsed)
    peaks[f"{name}_peak_bytes"] = int(peak_bytes)
    return value


def _plate_model(element_count: int) -> FEMModel:
    """Build a deterministic inline Quad4 plate with ``element_count`` cells."""

    if type(element_count) is not int or element_count < 0:
        raise ValueError("element_count must be a non-negative integer")
    if element_count == 0:
        return FEMModel(Mesh2D([], []), name="phase0-empty")

    columns = max(int(element_count**0.5), 1)
    rows = (element_count + columns - 1) // columns
    nodes = [
        Node2D(
            row * (columns + 1) + column + 1,
            float(column),
            float(row),
        )
        for row in range(rows + 1)
        for column in range(columns + 1)
    ]
    elements = []
    for index in range(element_count):
        row, column = divmod(index, columns)
        lower_left = row * (columns + 1) + column + 1
        elements.append(
            Element2D(
                index + 1,
                [
                    lower_left,
                    lower_left + 1,
                    lower_left + columns + 2,
                    lower_left + columns + 1,
                ],
                "Quad4",
            )
        )
    return FEMModel(
        Mesh2D(nodes, elements),
        name=f"phase0-plate-{element_count}",
        steps=[AnalysisStep("load")],
    )


def _runtime_info() -> dict[str, Any]:
    """Collect versions without making optional GUI packages mandatory."""

    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "qt": None,
        "pyside6": None,
        "pyvista": None,
        "vtk": None,
    }
    errors: dict[str, str] = {}
    try:
        import PySide6
        from PySide6.QtCore import qVersion

        versions["pyside6"] = str(getattr(PySide6, "__version__", "unknown"))
        versions["qt"] = str(qVersion())
    except Exception as error:  # optional dependency
        errors["qt"] = _error_text(error)
    try:
        import pyvista

        versions["pyvista"] = str(getattr(pyvista, "__version__", "unknown"))
    except Exception as error:  # optional dependency
        errors["pyvista"] = _error_text(error)
    try:
        import vtk

        versions["vtk"] = str(vtk.vtkVersion.GetVTKVersion())
    except Exception as error:  # optional dependency
        errors["vtk"] = _error_text(error)

    return {"versions": versions, "errors": errors}


def _hardware_info() -> dict[str, Any]:
    """Return stable stdlib-only hardware descriptors for baseline context."""

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def _count_objects(predicate: Callable[[object], bool]) -> int:
    gc.collect()
    count = 0
    for candidate in gc.get_objects():
        try:
            if predicate(candidate):
                count += 1
        except Exception:
            continue
    return count


def _named_object_count(names: set[str]) -> int:
    return _count_objects(lambda candidate: type(candidate).__name__ in names)


def _gui_metrics(model: FEMModel | None, geometry: Any | None) -> dict[str, Any]:
    """Exercise tree/viewport boundaries and return GUI metrics or a skip record."""

    timings: dict[str, float | None] = {}
    peaks: dict[str, int | None] = {}
    errors: dict[str, str] = {}
    spy_calls = {
        "ModelTree.clear": 0,
        "ResultTree.clear": 0,
        "FEMViewport.render": 0,
        "render": 0,
    }
    object_counts: dict[str, int | None] = {
        "viewport_before": None,
        "viewport_live": None,
        "viewport_after_cleanup": None,
        "plotter_before": None,
        "plotter_live": None,
        "plotter_after_cleanup": None,
    }

    try:
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QApplication

        from fem_gui.widgets.model_tree import ModelTree
        from fem_gui.widgets.result_tree import ResultTree
        from fem_gui.widgets.viewport import FEMViewport
    except Exception as error:  # optional GUI dependency
        return {
            "available": False,
            "errors": {"gui": _error_text(error)},
            "timings_seconds": timings,
            "tracemalloc_peak_bytes": peaks,
            "object_counts": object_counts,
            "spy_calls": spy_calls,
        }

    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    object_counts["viewport_before"] = _count_objects(
        lambda candidate: isinstance(candidate, FEMViewport)
    )
    object_counts["plotter_before"] = _named_object_count(
        {"QtInteractor", "BackgroundPlotter", "Plotter", "BasePlotter"}
    )

    viewport = None
    model_tree = None
    result_tree = None
    plotter = None
    plotter_type: type[object] | None = None
    with _method_spy(ModelTree, "clear") as model_tree_spy:
        with _method_spy(ResultTree, "clear") as result_tree_spy:
            with _method_spy(FEMViewport, "render") as public_render_spy:
                with _method_spy(FEMViewport, "_render") as render_spy:
                    try:
                        viewport = FEMViewport()
                        model_tree = ModelTree()
                        result_tree = ResultTree()
                        model_tree.clear_model()
                        result_tree.clear_result()
                        if model is None:
                            _record_measure(
                                timings,
                                peaks,
                                errors,
                                "viewport_clear_model",
                                viewport.clear_model,
                            )
                        else:
                            _record_measure(
                                timings,
                                peaks,
                                errors,
                                "viewport_set_model",
                                lambda: viewport.set_model(
                                    model,
                                    geometry,
                                    refresh_symbols=False,
                                    render=False,
                                ),
                            )
                        _record_measure(
                            timings,
                            peaks,
                            errors,
                            "render",
                            viewport.render,
                        )
                        plotter = getattr(viewport, "_plotter", None)
                        if plotter is not None:
                            plotter_type = type(plotter)
                    except Exception as error:  # optional backend/runtime failure
                        errors.setdefault("gui", _error_text(error))
                    finally:
                        spy_calls["ModelTree.clear"] = model_tree_spy.calls
                        spy_calls["ResultTree.clear"] = result_tree_spy.calls
                        spy_calls["FEMViewport.render"] = public_render_spy.calls
                        spy_calls["render"] = render_spy.calls

    object_counts["viewport_live"] = _count_objects(
        lambda candidate: isinstance(candidate, FEMViewport)
    )
    if plotter_type is None:
        object_counts["plotter_live"] = _named_object_count(
            {"QtInteractor", "BackgroundPlotter", "Plotter", "BasePlotter"}
        )
    else:
        object_counts["plotter_live"] = _count_objects(
            lambda candidate: type(candidate) is plotter_type
        )

    # Release native resources before the next model scenario.  The live
    # counts above intentionally include the current local references; the
    # after-cleanup counts are taken only after those references are dropped
    # and Qt's deferred deletes are delivered.
    if viewport is not None:
        try:
            shutdown = getattr(viewport, "shutdown_backend", None)
            if callable(shutdown):
                shutdown()
            viewport.deleteLater()
        except Exception as error:
            errors.setdefault("gui_cleanup", _error_text(error))
        shutdown = None
    viewport = None
    model_tree = None
    result_tree = None
    plotter = None
    application.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()
    gc.collect()
    object_counts["viewport_after_cleanup"] = _count_objects(
        lambda candidate: isinstance(candidate, FEMViewport)
    )
    if plotter_type is None:
        object_counts["plotter_after_cleanup"] = _named_object_count(
            {"QtInteractor", "BackgroundPlotter", "Plotter", "BasePlotter"}
        )
    else:
        object_counts["plotter_after_cleanup"] = _count_objects(
            lambda candidate: type(candidate) is plotter_type
        )

    return {
        "available": True,
        "errors": errors,
        "timings_seconds": timings,
        "tracemalloc_peak_bytes": peaks,
        "object_counts": object_counts,
        "spy_calls": spy_calls,
    }


def _run_model_scenario(element_count: int) -> dict[str, Any]:
    timings: dict[str, float | None] = {}
    peaks: dict[str, int | None] = {}
    errors: dict[str, str] = {}
    model = _record_measure(
        timings,
        peaks,
        errors,
        "model_construction",
        lambda: _plate_model(element_count),
    )
    result: dict[str, Any] = {
        "elements": element_count,
        "nodes": None,
        "timings_seconds": timings,
        "tracemalloc_peak_bytes": peaks,
        "errors": errors,
        "object_counts": {},
        "spy_calls": {
            "geometry_builder": 0,
            "ModelTree.clear": 0,
            "ResultTree.clear": 0,
            "FEMViewport.render": 0,
            "render": 0,
        },
    }
    if model is None:
        return result
    result["nodes"] = len(model.mesh.nodes)

    with _method_spy(model_adapter, "build_model_geometry") as geometry_spy:
        geometry = _record_measure(
            timings,
            peaks,
            errors,
            "geometry_construction",
            lambda: model_adapter.build_model_geometry(model),
        )
    result["spy_calls"]["geometry_builder"] = geometry_spy.calls
    if geometry is None:
        result["gui"] = _gui_metrics(model, None)
        result["object_counts"] = result["gui"]["object_counts"]
        result["spy_calls"].update(result["gui"]["spy_calls"])
        return result

    _record_measure(
        timings,
        peaks,
        errors,
        "inspection_construction",
        lambda: inspection_service.InspectionService(model),
    )
    result["gui"] = _gui_metrics(model, geometry)
    result["object_counts"] = result["gui"]["object_counts"]
    result["spy_calls"].update(result["gui"]["spy_calls"])
    return result


def _build_archive_snapshot(element_count: int) -> ResultArchiveSnapshot:
    """Build one medium archive payload from an inline ModelResult."""

    model = _plate_model(element_count)
    step = AnalysisStep("load")
    result = ModelResult(
        model,
        step,
        np.zeros(model.mesh.num_dofs, dtype=float),
        np.zeros(model.mesh.num_dofs, dtype=float),
        name="phase0-result",
    )
    source = ResultSourceKey(
        result_id="phase0-result",
        session_id="phase0-session",
        artifact_id="phase0-artifact",
        model_revision=1,
        step_name="load",
        run_id="phase0-run",
    )
    provider = build_result_provider(source, result)
    outcome = execute_output_requests(
        provider,
        (OutputRequest("field", "node", ("U", "RF")),),
    )
    provider = outcome.provider_draft
    topology = provider.snapshot.topology
    fingerprint = result_model_fingerprint(
        topology,
        provider.profile,
        step_name=source.step_name,
        unit_context=None,
    )
    projection = ResultArchiveModelProjection(
        topology=topology,
        named_region_node_ids={"all_nodes": tuple(topology.node_ids)},
        named_region_element_ids={"all_elements": tuple(topology.element_ids)},
        summaries={
            "model_family": provider.profile.family.value,
            "element_count": element_count,
        },
    )
    return ResultArchiveSnapshot(
        archive_id="phase0-archive",
        created_at=_FIXED_TIME,
        producer_version="phase0-baseline",
        origin=ResultArchiveOrigin(
            model_name="phase0-archive-model",
            source_basename="phase0-model.fempy",
            model_fingerprint=fingerprint,
            provenance={"run_id": source.run_id},
        ),
        run=ResultArchiveRun(
            name="phase0-run",
            step_name=source.step_name,
            created_at=_FIXED_TIME,
            output_report=outcome.report,
        ),
        profile=provider.profile,
        catalog=provider.catalog(),
        materialization=provider.snapshot,
        model_projection=projection,
    )


def _run_archive_scenario(element_count: int) -> dict[str, Any]:
    timings: dict[str, float | None] = {}
    peaks: dict[str, int | None] = {}
    errors: dict[str, str] = {}
    result: dict[str, Any] = {
        "elements": element_count,
        "timings_seconds": timings,
        "tracemalloc_peak_bytes": peaks,
        "errors": errors,
        "spy_calls": {"archive_loader": 0},
        "archive_bytes": None,
    }
    snapshot = _record_measure(
        timings,
        peaks,
        errors,
        "archive_fixture_construction",
        lambda: _build_archive_snapshot(element_count),
    )
    if snapshot is None:
        return result

    with _workspace_temp_directory() as directory:
        path = directory / "phase0-result.femres"
        encoded = _record_measure(
            timings,
            peaks,
            errors,
            "archive_encode",
            lambda: encode_result_archive(snapshot),
        )
        if encoded is not None:
            try:
                path.write_bytes(encoded)
            except OSError as error:
                errors["archive_write"] = _error_text(error)
        try:
            result["archive_bytes"] = path.stat().st_size
        except OSError as error:
            errors["archive_size"] = _error_text(error)

        try:
            import fem.io.result_archive as archive_module
        except Exception as error:  # pragma: no cover - package import failure
            errors["archive_loader"] = _error_text(error)
            return result
        with _method_spy(archive_module, "load_result_archive") as archive_spy:
            _record_measure(
                timings,
                peaks,
                errors,
                "archive_load",
                lambda: archive_module.load_result_archive(path),
            )
        result["spy_calls"]["archive_loader"] = archive_spy.calls
    return result


def _blocked_scenarios(
    element_counts: tuple[int, ...],
    *,
    include_archive: bool,
) -> dict[str, Any]:
    """Keep the CLI useful when an optional/project dependency is absent."""

    message = _error_text(_PROJECT_IMPORT_ERROR) if _PROJECT_IMPORT_ERROR else "unknown"
    scenarios: dict[str, Any] = {}
    if 0 in element_counts:
        scenarios["empty_document"] = {
            "available": False,
            "errors": {"project_import": message},
        }
    for element_count in element_counts:
        if element_count:
            scenarios[f"{element_count}_elements"] = {
                "available": False,
                "elements": element_count,
                "errors": {"project_import": message},
            }
    if include_archive:
        scenarios["medium_result_archive"] = {
            "available": False,
            "elements": _DEFAULT_ARCHIVE_ELEMENTS,
            "errors": {"project_import": message},
        }
    return scenarios


def run(
    element_counts: tuple[int, ...] = _DEFAULT_ELEMENT_COUNTS,
    *,
    include_archive: bool = True,
) -> dict[str, Any]:
    """Run selected Phase 0 scenarios and return a JSON-serializable record."""

    if _PROJECT_IMPORT_ERROR is not None:
        return {
            "schema": "phase-0-multi-document-workspace-baseline-v1",
            "phase": 0,
            "status": "blocked",
            "blocker": _error_text(_PROJECT_IMPORT_ERROR),
            "runtime": _runtime_info(),
            "hardware": _hardware_info(),
            "scenarios": _blocked_scenarios(
                element_counts,
                include_archive=include_archive,
            ),
        }

    scenarios: dict[str, Any] = {}
    for element_count in element_counts:
        if element_count == 0:
            scenarios["empty_document"] = _gui_metrics(None, None)
            continue
        scenarios[f"{element_count}_elements"] = _run_model_scenario(element_count)
    if include_archive:
        scenarios["medium_result_archive"] = _run_archive_scenario(
            _DEFAULT_ARCHIVE_ELEMENTS
        )
    return {
        "schema": "phase-0-multi-document-workspace-baseline-v1",
        "phase": 0,
        "runtime": _runtime_info(),
        "hardware": _hardware_info(),
        "scenarios": scenarios,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("empty", "5000", "20000", "archive", "all"),
        help=(
            "run exactly one named scenario; 'all' runs every Phase 0 scenario "
            "(default when no selector is supplied)"
        ),
    )
    parser.add_argument(
        "--elements",
        type=int,
        action="append",
        choices=(0, 5_000, 20_000),
        help=(
            "legacy model-size selector; repeat to select multiple sizes; "
            "explicit use excludes the archive"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the stable JSON record to this path as well as stdout",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(argv)
    if arguments.scenario is not None and arguments.elements:
        raise SystemExit("--scenario and --elements cannot be combined")
    if arguments.scenario == "empty":
        selected = (0,)
        include_archive = False
    elif arguments.scenario == "5000":
        selected = (5_000,)
        include_archive = False
    elif arguments.scenario == "20000":
        selected = (20_000,)
        include_archive = False
    elif arguments.scenario == "archive":
        selected = ()
        include_archive = True
    elif arguments.scenario == "all":
        selected = _DEFAULT_ELEMENT_COUNTS
        include_archive = True
    elif arguments.elements:
        selected = tuple(arguments.elements)
        include_archive = False
    else:
        selected = _DEFAULT_ELEMENT_COUNTS
        include_archive = True
    record = run(selected, include_archive=include_archive)
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        separators=(",", ": "),
    )
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
