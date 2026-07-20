from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from fem.geometry import GeometryStateError
from fem.geometry._gmsh import session as _session_module
from fem.geometry._gmsh.session import _GmshModelSession


class _FakeModel:
    def __init__(
        self,
        backend: _FakeGmsh,
        *,
        names: tuple[str, ...],
        current: str,
    ) -> None:
        self._backend = backend
        self.models: dict[str, None] = dict.fromkeys(names)
        self.current = current
        self.fail_remove_count = 0
        self.fail_set_current_counts: dict[str, int] = {}

    def list(self) -> list[str]:
        self._backend.calls.append(("model.list",))
        return list(self.models)

    def getCurrent(self) -> str:
        self._backend.calls.append(("model.getCurrent",))
        return self.current

    def setCurrent(self, name: str) -> None:
        self._backend.calls.append(("model.setCurrent", name))
        remaining = self.fail_set_current_counts.get(name, 0)
        if remaining:
            self.fail_set_current_counts[name] = remaining - 1
            raise RuntimeError(f"fake setCurrent failure for {name!r}")
        if name not in self.models:
            raise RuntimeError(f"unknown model {name!r}")
        self.current = name

    def add(self, name: str) -> None:
        self._backend.calls.append(("model.add", name))
        if name in self.models:
            raise RuntimeError(f"duplicate model {name!r}")
        self.models[name] = None
        self.current = name

    def remove(self) -> None:
        self._backend.calls.append(("model.remove", self.current))
        if self.fail_remove_count:
            self.fail_remove_count -= 1
            raise RuntimeError("fake remove failure")
        if self.current not in self.models:
            raise RuntimeError(f"unknown model {self.current!r}")
        del self.models[self.current]
        self.current = next(iter(self.models), "")


class _FakeOption:
    def __init__(self, backend: _FakeGmsh) -> None:
        self._backend = backend
        self.values: dict[str, float] = {}
        self.fail_get_counts: dict[str, int] = {}
        self.fail_set_counts: dict[str, int] = {}

    def fail_next_get(self, name: str) -> None:
        self.fail_get_counts[name] = self.fail_get_counts.get(name, 0) + 1

    def fail_next_set(self, name: str) -> None:
        self.fail_set_counts[name] = self.fail_set_counts.get(name, 0) + 1

    def getNumber(self, name: str) -> float:
        self._backend.calls.append(("option.getNumber", name))
        remaining = self.fail_get_counts.get(name, 0)
        if remaining:
            self.fail_get_counts[name] = remaining - 1
            raise RuntimeError(f"fake option get failure for {name}")
        return self.values.get(name, 0.0)

    def setNumber(self, name: str, value: float) -> None:
        numeric_value = float(value)
        self._backend.calls.append(("option.setNumber", name, numeric_value))
        remaining = self.fail_set_counts.get(name, 0)
        if remaining:
            self.fail_set_counts[name] = remaining - 1
            raise RuntimeError(f"fake option set failure for {name}")
        self.values[name] = numeric_value


class _FakeGmsh:
    def __init__(
        self,
        *,
        initialized: bool = False,
        names: tuple[str, ...] = (),
        current: str = "",
    ) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.initialized = initialized
        self.fail_initialize_after_state = False
        self.fail_inspect_count = 0
        self.fail_finalize_count = 0
        self.initialize_calls = 0
        self.finalize_calls = 0
        self.model = _FakeModel(self, names=names, current=current)
        self.option = _FakeOption(self)

    def isInitialized(self) -> bool:
        self.calls.append(("isInitialized",))
        if self.fail_inspect_count:
            self.fail_inspect_count -= 1
            raise RuntimeError("fake session inspection failure")
        return self.initialized

    def initialize(self) -> None:
        self.calls.append(("initialize",))
        self.initialize_calls += 1
        self.initialized = True
        if self.fail_initialize_after_state:
            raise RuntimeError("fake initialize failure")

    def finalize(self) -> None:
        self.calls.append(("finalize",))
        self.finalize_calls += 1
        if self.fail_finalize_count:
            self.fail_finalize_count -= 1
            raise RuntimeError("fake finalize failure")
        self.initialized = False


def _install_backend(
    monkeypatch: pytest.MonkeyPatch,
    gmsh: _FakeGmsh,
) -> None:
    def load_gmsh() -> _FakeGmsh:
        gmsh.calls.append(("load backend",))
        return gmsh

    monkeypatch.setattr(_session_module.backend, "load_gmsh", load_gmsh)


def _enter_session(
    monkeypatch: pytest.MonkeyPatch,
    gmsh: _FakeGmsh,
    name: str = "facade",
) -> _GmshModelSession:
    _install_backend(monkeypatch, gmsh)
    session = _GmshModelSession(name)
    assert not hasattr(session, "facade")
    assert session.enter() is None
    return session


def test_entry_preserves_backend_session_capture_validation_and_add_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh()
    _install_backend(monkeypatch, gmsh)
    session = _GmshModelSession("facade")

    assert gmsh.calls == []

    assert session.enter() is None

    assert gmsh.calls == [
        ("load backend",),
        ("isInitialized",),
        ("initialize",),
        ("model.getCurrent",),
        ("model.list",),
        ("model.add", "facade"),
        ("model.setCurrent", "facade"),
    ]
    assert not hasattr(session, "facade")
    assert session.created_model
    assert gmsh.model.current == "facade"


def test_invalid_name_is_detected_after_owned_session_and_model_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh()
    _install_backend(monkeypatch, gmsh)
    session = _GmshModelSession(" \t")

    with pytest.raises(GeometryStateError, match="model name must be a nonempty"):
        session.enter()

    assert gmsh.calls == [
        ("load backend",),
        ("isInitialized",),
        ("initialize",),
        ("model.getCurrent",),
        ("model.list",),
    ]
    assert session.cleanup_after_failed_entry() == ()
    assert gmsh.initialize_calls == 1
    assert gmsh.finalize_calls == 1
    assert not gmsh.initialized


def test_external_session_removes_only_facade_and_restores_prior_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(
        initialized=True,
        names=("prior", "other"),
        current="prior",
    )
    session = _enter_session(monkeypatch, gmsh)
    gmsh.model.setCurrent("other")

    assert session.cleanup_after_failed_entry() == ()

    assert tuple(gmsh.model.models) == ("prior", "other")
    assert gmsh.model.current == "prior"
    assert gmsh.initialize_calls == 0
    assert gmsh.finalize_calls == 0
    assert not session.created_model


def test_nested_like_sessions_restore_lifo_and_only_outer_finalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh()
    _install_backend(monkeypatch, gmsh)
    outer = _GmshModelSession("outer")
    inner = _GmshModelSession("inner")

    outer.enter()
    assert gmsh.model.current == "outer"
    inner.enter()
    assert gmsh.model.current == "inner"

    assert inner.cleanup_after_failed_entry() == ()
    assert gmsh.model.current == "outer"
    assert gmsh.initialized
    assert gmsh.finalize_calls == 0

    assert outer.cleanup_after_failed_entry() == ()
    assert gmsh.initialize_calls == 1
    assert gmsh.finalize_calls == 1
    assert not gmsh.initialized


def test_activate_reselects_only_when_needed_and_returns_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(
        initialized=True,
        names=("external",),
        current="external",
    )
    session = _enter_session(monkeypatch, gmsh)
    gmsh.model.setCurrent("external")
    gmsh.calls.clear()

    assert session.activate("entities") is gmsh
    assert gmsh.calls == [
        ("isInitialized",),
        ("model.list",),
        ("model.getCurrent",),
        ("model.setCurrent", "facade"),
    ]

    gmsh.calls.clear()
    assert session.activate("entities") is gmsh
    assert gmsh.calls == [
        ("isInitialized",),
        ("model.list",),
        ("model.getCurrent",),
    ]


def test_activate_reports_inactive_session_and_externally_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(initialized=True)
    session = _enter_session(monkeypatch, gmsh)

    gmsh.initialized = False
    with pytest.raises(
        GeometryStateError,
        match="facade.*query.*Gmsh session is not active",
    ):
        session.activate("query")

    gmsh.initialized = True
    del gmsh.model.models["facade"]
    with pytest.raises(
        GeometryStateError,
        match="facade.*query.*facade-owned Gmsh model is missing",
    ):
        session.activate("query")


def test_valid_empty_prior_model_name_is_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(initialized=True, names=("",), current="")
    session = _enter_session(monkeypatch, gmsh)
    gmsh.calls.clear()

    session.remove_created_model()
    session.restore_prior_model()

    assert tuple(gmsh.model.models) == ("",)
    assert gmsh.model.current == ""
    assert ("model.setCurrent", "") in gmsh.calls


def test_missing_prior_model_is_skipped_without_disturbing_current_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(
        initialized=True,
        names=("prior", "other"),
        current="prior",
    )
    session = _enter_session(monkeypatch, gmsh)
    del gmsh.model.models["prior"]
    gmsh.calls.clear()

    session.remove_created_model()
    session.restore_prior_model()

    assert gmsh.model.current == "other"
    assert ("model.setCurrent", "prior") not in gmsh.calls


def test_model_name_collision_fails_before_add_or_external_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(initialized=True, names=("taken",), current="taken")
    _install_backend(monkeypatch, gmsh)
    session = _GmshModelSession("taken")

    with pytest.raises(GeometryStateError, match="already exists"):
        session.enter()

    assert ("model.add", "taken") not in gmsh.calls
    assert session.cleanup_after_failed_entry() == ()
    assert gmsh.model.current == "taken"
    assert gmsh.finalize_calls == 0


def test_partially_successful_initialize_is_owned_and_finalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh()
    gmsh.fail_initialize_after_state = True
    _install_backend(monkeypatch, gmsh)
    session = _GmshModelSession("facade")

    with pytest.raises(RuntimeError, match="fake initialize failure"):
        session.enter()

    assert gmsh.calls == [
        ("load backend",),
        ("isInitialized",),
        ("initialize",),
    ]
    assert gmsh.initialized

    assert session.cleanup_after_failed_entry() == ()
    assert gmsh.finalize_calls == 1
    assert not gmsh.initialized
    assert not session.created_model


def test_cleanup_inspection_failure_retains_ownership_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh()
    session = _enter_session(monkeypatch, gmsh)
    gmsh.fail_inspect_count = 1
    gmsh.calls.clear()

    errors = session.cleanup_after_failed_entry()

    assert len(errors) == 1
    operation, error = errors[0]
    assert operation == "inspect Gmsh session state"
    assert str(error) == "fake session inspection failure"
    assert gmsh.calls == [("isInitialized",)]
    assert session.created_model
    assert "facade" in gmsh.model.models
    assert gmsh.initialized
    assert gmsh.finalize_calls == 0

    assert session.cleanup_after_failed_entry() == ()
    assert not session.created_model
    assert "facade" not in gmsh.model.models
    assert gmsh.finalize_calls == 1
    assert not gmsh.initialized


def test_cleanup_attempts_every_step_and_retains_each_failure_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(names=("prior",), current="prior")
    gmsh.option.values["Mesh.Algorithm"] = 1.0
    session = _enter_session(monkeypatch, gmsh)
    session.set_numeric_options((("Mesh.Algorithm", 7.0),))
    gmsh.option.fail_next_set("Mesh.Algorithm")
    gmsh.model.fail_remove_count = 1
    gmsh.model.fail_set_current_counts["prior"] = 1
    gmsh.fail_finalize_count = 1
    gmsh.calls.clear()

    errors = session.cleanup_after_failed_entry()

    assert tuple(operation for operation, _error in errors) == (
        "restore mesh options",
        "remove facade model",
        "restore prior model",
        "finalize owned session",
    )
    assert tuple(str(error) for _operation, error in errors) == (
        "fake option set failure for Mesh.Algorithm",
        "fake remove failure",
        "fake setCurrent failure for 'prior'",
        "fake finalize failure",
    )
    assert gmsh.calls == [
        ("isInitialized",),
        ("option.setNumber", "Mesh.Algorithm", 1.0),
        ("model.list",),
        ("model.getCurrent",),
        ("model.remove", "facade"),
        ("model.list",),
        ("model.setCurrent", "prior"),
        ("finalize",),
    ]
    assert session.has_pending_options
    assert session.created_model
    assert gmsh.initialized

    assert session.cleanup_after_failed_entry() == ()
    assert not session.has_pending_options
    assert not session.created_model
    assert gmsh.option.values["Mesh.Algorithm"] == 1.0
    assert tuple(gmsh.model.models) == ("prior",)
    assert gmsh.model.current == "prior"
    assert gmsh.finalize_calls == 2
    assert not gmsh.initialized


def test_already_finalized_session_relinquishes_stale_cleanup_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(
        initialized=True,
        names=("prior",),
        current="prior",
    )
    session = _enter_session(monkeypatch, gmsh)
    gmsh.initialized = False
    gmsh.calls.clear()

    assert session.cleanup_after_failed_entry() == ()

    assert gmsh.calls == [("isInitialized",)]
    assert not session.created_model
    assert gmsh.finalize_calls == 0


def test_numeric_options_read_all_originals_before_ordered_writes_and_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(initialized=True)
    gmsh.option.values.update({"Mesh.A": 1.0, "Mesh.B": 2.0})
    session = _enter_session(monkeypatch, gmsh)
    gmsh.calls.clear()

    session.set_numeric_options((("Mesh.A", 10), ("Mesh.B", 20)))

    assert gmsh.calls == [
        ("option.getNumber", "Mesh.A"),
        ("option.getNumber", "Mesh.B"),
        ("option.setNumber", "Mesh.A", 10.0),
        ("option.setNumber", "Mesh.B", 20.0),
    ]
    assert session.has_pending_options

    gmsh.calls.clear()
    session.restore_pending_options()
    assert gmsh.calls == [
        ("option.setNumber", "Mesh.A", 1.0),
        ("option.setNumber", "Mesh.B", 2.0),
    ]
    assert gmsh.option.values == {"Mesh.A": 1.0, "Mesh.B": 2.0}
    assert not session.has_pending_options


def test_numeric_option_partial_read_performs_no_writes_and_retains_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(initialized=True)
    gmsh.option.values.update({"Mesh.A": 1.0, "Mesh.B": 2.0})
    session = _enter_session(monkeypatch, gmsh)
    gmsh.option.fail_next_get("Mesh.B")
    gmsh.calls.clear()

    with pytest.raises(RuntimeError, match="get failure for Mesh.B"):
        session.set_numeric_options((("Mesh.A", 10), ("Mesh.B", 20)))

    assert gmsh.calls == [
        ("option.getNumber", "Mesh.A"),
        ("option.getNumber", "Mesh.B"),
    ]
    assert gmsh.option.values == {"Mesh.A": 1.0, "Mesh.B": 2.0}
    assert session.has_pending_options

    gmsh.calls.clear()
    session.restore_pending_options()
    assert gmsh.calls == [("option.setNumber", "Mesh.A", 1.0)]
    assert not session.has_pending_options


def test_numeric_option_partial_write_keeps_every_snapshot_for_restoration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(initialized=True)
    gmsh.option.values.update(
        {"Mesh.A": 1.0, "Mesh.B": 2.0, "Mesh.C": 3.0}
    )
    session = _enter_session(monkeypatch, gmsh)
    gmsh.option.fail_next_set("Mesh.B")
    gmsh.calls.clear()

    with pytest.raises(RuntimeError, match="set failure for Mesh.B"):
        session.set_numeric_options(
            (("Mesh.A", 10), ("Mesh.B", 20), ("Mesh.C", 30))
        )

    assert gmsh.calls == [
        ("option.getNumber", "Mesh.A"),
        ("option.getNumber", "Mesh.B"),
        ("option.getNumber", "Mesh.C"),
        ("option.setNumber", "Mesh.A", 10.0),
        ("option.setNumber", "Mesh.B", 20.0),
    ]
    assert gmsh.option.values == {
        "Mesh.A": 10.0,
        "Mesh.B": 2.0,
        "Mesh.C": 3.0,
    }
    assert session.has_pending_options

    gmsh.calls.clear()
    session.restore_pending_options()
    assert gmsh.calls == [
        ("option.setNumber", "Mesh.A", 1.0),
        ("option.setNumber", "Mesh.B", 2.0),
        ("option.setNumber", "Mesh.C", 3.0),
    ]
    assert gmsh.option.values == {
        "Mesh.A": 1.0,
        "Mesh.B": 2.0,
        "Mesh.C": 3.0,
    }
    assert not session.has_pending_options


def test_numeric_option_partial_restore_continues_and_retries_only_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(initialized=True)
    gmsh.option.values.update(
        {"Mesh.A": 1.0, "Mesh.B": 2.0, "Mesh.C": 3.0}
    )
    session = _enter_session(monkeypatch, gmsh)
    session.set_numeric_options(
        (("Mesh.A", 10), ("Mesh.B", 20), ("Mesh.C", 30))
    )
    gmsh.option.fail_next_set("Mesh.A")
    gmsh.option.fail_next_set("Mesh.C")
    gmsh.calls.clear()

    with pytest.raises(RuntimeError, match="set failure for Mesh.A"):
        session.restore_pending_options()

    assert gmsh.calls == [
        ("option.setNumber", "Mesh.A", 1.0),
        ("option.setNumber", "Mesh.B", 2.0),
        ("option.setNumber", "Mesh.C", 3.0),
    ]
    assert gmsh.option.values == {
        "Mesh.A": 10.0,
        "Mesh.B": 2.0,
        "Mesh.C": 30.0,
    }
    assert session.has_pending_options

    gmsh.calls.clear()
    session.restore_pending_options()
    assert gmsh.calls == [
        ("option.setNumber", "Mesh.A", 1.0),
        ("option.setNumber", "Mesh.C", 3.0),
    ]
    assert gmsh.option.values == {
        "Mesh.A": 1.0,
        "Mesh.B": 2.0,
        "Mesh.C": 3.0,
    }
    assert not session.has_pending_options


def test_second_numeric_option_transaction_is_rejected_before_native_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(initialized=True)
    gmsh.option.values["Mesh.A"] = 1.0
    session = _enter_session(monkeypatch, gmsh)
    session.set_numeric_options((("Mesh.A", 10),))
    gmsh.calls.clear()

    with pytest.raises(GeometryStateError, match="pending restoration"):
        session.set_numeric_options((("Mesh.B", 20),))

    assert gmsh.calls == []
    assert session.has_pending_options


def test_nested_sessions_keep_independent_numeric_option_ledgers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(
        initialized=True,
        names=("prior",),
        current="prior",
    )
    gmsh.option.values["Mesh.Algorithm"] = 1.0
    _install_backend(monkeypatch, gmsh)
    outer = _GmshModelSession("outer")
    inner = _GmshModelSession("inner")

    outer.enter()
    outer.set_numeric_options((("Mesh.Algorithm", 10),))
    inner.enter()
    inner.set_numeric_options((("Mesh.Algorithm", 20),))

    assert outer.has_pending_options
    assert inner.has_pending_options
    assert gmsh.option.values["Mesh.Algorithm"] == 20.0

    inner.restore_pending_options()
    assert not inner.has_pending_options
    assert outer.has_pending_options
    assert gmsh.option.values["Mesh.Algorithm"] == 10.0

    outer.restore_pending_options()
    assert not outer.has_pending_options
    assert gmsh.option.values["Mesh.Algorithm"] == 1.0

    assert inner.cleanup_after_failed_entry() == ()
    assert gmsh.model.current == "outer"
    assert outer.cleanup_after_failed_entry() == ()
    assert gmsh.model.current == "prior"
    assert gmsh.finalize_calls == 0


def test_private_session_import_does_not_eagerly_import_external_gmsh() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import builtins
import sys

sys.path.insert(0, {str(src_dir)!r})
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "gmsh" or name.startswith("gmsh."):
        raise AssertionError("external gmsh was imported eagerly")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from fem.geometry._gmsh.session import _GmshModelSession

assert not hasattr(_GmshModelSession("lazy"), "facade")
assert "gmsh" not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
