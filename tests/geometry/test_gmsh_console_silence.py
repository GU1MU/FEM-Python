"""Console-silence contract for the owned Gmsh runtime sessions.

Geometry compilation and mesh generation must not emit Gmsh Info-level
progress (OCC Boolean steps, tessellation meshing) to the terminal.  The
application silences the console by pinning ``General.Terminal`` to zero
whenever it owns the process-global Gmsh session.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from fem.geometry._gmsh import session as _session_module
from fem.geometry._gmsh.session import _GmshModelSession

_TERMINAL_OPTION = "General.Terminal"

_GEOMETRY_COMPILE_SCRIPT = (
    "from fem.geometry import model\n"
    "with model('silence-geometry', dimension=2) as cad:\n"
    "    rect = cad.rectangle(0, 0, 10, 10)\n"
    "    hole = cad.disk(5, 5, 2)\n"
    "    cad.cut((rect,), (hole,))\n"
    "    faces = cad.entities(2)\n"
    "    edges = cad.boundary(faces, combined=False)\n"
    "    points = cad.boundary(edges, combined=False)\n"
    "    tessellation = cad.tessellate_surfaces(faces, edges, points)\n"
    "print('SILENCE_OK', len(tessellation.points), len(tessellation.faces))\n"
)

_MESH_GENERATION_SCRIPT = (
    "from fem.application.preprocessing import generate_fem_model\n"
    "from fem.geometry import RectangleGeometry\n"
    "from fem.mesh.settings import MeshSettings\n"
    "model = generate_fem_model(\n"
    "    RectangleGeometry('silence-mesh', 2.0, 1.0),\n"
    "    MeshSettings(0.5),\n"
    ")\n"
    "print('SILENCE_OK', len(model.mesh.nodes), len(model.mesh.elements))\n"
)


class _RecordingOption:
    def __init__(self, backend: _RecordingGmsh) -> None:
        self._backend = backend
        self.values: dict[str, float] = {}

    def getNumber(self, name: str) -> float:
        return self.values.get(name, 0.0)

    def setNumber(self, name: str, value: float) -> None:
        self._backend.option_calls.append((name, float(value)))
        self.values[name] = float(value)


class _RecordingModel:
    def __init__(self) -> None:
        self.models: dict[str, dict[str, list[str]]] = {}
        self.current = ""

    def list(self) -> list[str]:
        return list(self.models)

    def getCurrent(self) -> str:
        return self.current

    def setCurrent(self, name: str) -> None:
        self.current = name

    def add(self, name: str) -> None:
        self.models[name] = {}
        self.current = name

    def remove(self) -> None:
        del self.models[self.current]
        self.current = next(iter(self.models), "")

    def getAttribute(self, name: str) -> list[str]:
        return list(self.models[self.current].get(name, ()))

    def setAttribute(self, name: str, values: list[str]) -> None:
        self.models[self.current][name] = [str(item) for item in values]


class _RecordingGmsh:
    def __init__(self, *, initialized: bool = False) -> None:
        self.initialized = initialized
        self.option_calls: list[tuple[str, float]] = []
        self.model = _RecordingModel()
        self.option = _RecordingOption(self)

    def isInitialized(self) -> bool:
        return self.initialized

    def initialize(self, *, interruptible: bool = True) -> None:
        self.initialized = True

    def finalize(self) -> None:
        self.initialized = False


def _install_backend(
    monkeypatch: pytest.MonkeyPatch,
    gmsh: _RecordingGmsh,
) -> None:
    monkeypatch.setattr(_session_module.backend, "load_gmsh", lambda: gmsh)


def test_owned_session_pins_terminal_option_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _RecordingGmsh()
    _install_backend(monkeypatch, gmsh)
    session = _GmshModelSession("silence-owned")

    assert session.enter() is None

    assert gmsh.option_calls == [(_TERMINAL_OPTION, 0.0)]


def test_external_session_keeps_caller_terminal_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _RecordingGmsh(initialized=True)
    gmsh.option.values[_TERMINAL_OPTION] = 5.0
    _install_backend(monkeypatch, gmsh)
    session = _GmshModelSession("silence-external")

    assert session.enter() is None

    assert gmsh.option_calls == []
    assert gmsh.option.values[_TERMINAL_OPTION] == 5.0


def _run_isolated_script(script: str) -> subprocess.CompletedProcess[str]:
    pytest.importorskip(
        "gmsh",
        reason="the optional native Gmsh runtime is not installed",
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "SILENCE_OK" in completed.stdout
    return completed


def _assert_console_silent(completed: subprocess.CompletedProcess[str]) -> None:
    for stream_name, stream_text in (
        ("stdout", completed.stdout),
        ("stderr", completed.stderr),
    ):
        assert "Info    :" not in stream_text, (
            f"Gmsh Info output leaked to {stream_name}:\n{stream_text}"
        )


@pytest.mark.gmsh
def test_geometry_compilation_keeps_terminal_silent() -> None:
    completed = _run_isolated_script(_GEOMETRY_COMPILE_SCRIPT)
    _assert_console_silent(completed)


@pytest.mark.gmsh
def test_fem_model_generation_keeps_terminal_silent() -> None:
    completed = _run_isolated_script(_MESH_GENERATION_SCRIPT)
    _assert_console_silent(completed)
