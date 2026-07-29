from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from fem.geometry import GeometryStateError
from fem.geometry._gmsh import session as _session_module
from fem.geometry._gmsh.session import (
    _MODEL_INCARNATION_ATTRIBUTE,
    _SESSION_BASELINE_INCARNATION_ATTRIBUTE,
    _GmshModelSession,
)


class _FakeModel:
    def __init__(
        self,
        backend: _FakeGmsh,
        *,
        names: tuple[str, ...],
        current: str,
    ) -> None:
        self._backend = backend
        self.models: dict[str, dict[str, list[str]]] = {
            name: {} for name in names
        }
        self.current = current
        self.fail_remove_count = 0
        self.fail_set_current_counts: dict[str, int] = {}
        self.fail_get_attribute_count = 0
        self.fail_set_attribute_count = 0
        self.fail_set_attribute_after_state = False

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
        self.models[name] = {}
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

    def getAttribute(self, name: str) -> list[str]:
        self._backend.calls.append(("model.getAttribute", name))
        if self.fail_get_attribute_count:
            self.fail_get_attribute_count -= 1
            raise RuntimeError("fake getAttribute failure")
        return list(self.models[self.current].get(name, ()))

    def setAttribute(self, name: str, values: list[str]) -> None:
        materialized = [str(item) for item in values]
        self._backend.calls.append(
            ("model.setAttribute", name, tuple(materialized))
        )
        if self.fail_set_attribute_after_state:
            self.models[self.current][name] = materialized
        if self.fail_set_attribute_count:
            self.fail_set_attribute_count -= 1
            raise RuntimeError("fake setAttribute failure")
        self.models[self.current][name] = materialized


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
        self.initialize_interruptible: list[bool] = []
        self.finalize_calls = 0
        self.model = _FakeModel(self, names=names, current=current)
        self.option = _FakeOption(self)

    def isInitialized(self) -> bool:
        self.calls.append(("isInitialized",))
        if self.fail_inspect_count:
            self.fail_inspect_count -= 1
            raise RuntimeError("fake session inspection failure")
        return self.initialized

    def initialize(self, *, interruptible: bool = True) -> None:
        self.calls.append(("initialize",))
        self.initialize_calls += 1
        self.initialize_interruptible.append(bool(interruptible))
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
        (
            "model.setAttribute",
            _MODEL_INCARNATION_ATTRIBUTE,
            (session._model_incarnation,),
        ),
        ("model.getAttribute", _MODEL_INCARNATION_ATTRIBUTE),
        ("model.setCurrent", "facade"),
    ]
    assert session.created_model
    assert gmsh.model.current == "facade"
    assert gmsh.initialize_interruptible == [False]


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
        ("model.getAttribute", _MODEL_INCARNATION_ATTRIBUTE),
    ]

    gmsh.calls.clear()
    assert session.activate("entities") is gmsh
    assert gmsh.calls == [
        ("isInitialized",),
        ("model.list",),
        ("model.getCurrent",),
        ("model.getAttribute", _MODEL_INCARNATION_ATTRIBUTE),
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


@pytest.mark.parametrize("marker_written_before_error", [False, True])
def test_entry_marker_failure_distinguishes_verified_partial_installation(
    monkeypatch: pytest.MonkeyPatch,
    marker_written_before_error: bool,
) -> None:
    gmsh = _FakeGmsh(initialized=True, names=("prior",), current="prior")
    gmsh.model.fail_set_attribute_count = 1
    gmsh.model.fail_set_attribute_after_state = marker_written_before_error
    _install_backend(monkeypatch, gmsh)
    session = _GmshModelSession("facade")

    with pytest.raises(RuntimeError, match="fake setAttribute failure"):
        session.enter()

    if marker_written_before_error:
        assert tuple(gmsh.model.models) == ("prior",)
        assert not session.created_model
        assert session.cleanup_after_failed_entry() == ()
    else:
        assert tuple(gmsh.model.models) == ("prior", "facade")
        assert session.created_model
        ((operation, cleanup_error),) = session.cleanup_after_failed_entry()
        assert operation == "remove facade model"
        assert "incarnation was never verified" in str(cleanup_error)
        assert session.created_model
    assert gmsh.finalize_calls == 0


def test_cleanup_attribute_read_failure_retains_exact_model_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh()
    session = _enter_session(monkeypatch, gmsh)
    gmsh.model.fail_get_attribute_count = 2

    for _ in range(2):
        ((operation, cleanup_error),) = session.cleanup_after_failed_entry()
        assert operation == "remove facade model"
        assert str(cleanup_error) == "fake getAttribute failure"
        assert session.created_model
        assert "facade" in gmsh.model.models
        assert gmsh.initialized
        assert gmsh.finalize_calls == 0

    assert session.cleanup_after_failed_entry() == ()

    assert "facade" not in gmsh.model.models
    assert not session.created_model
    assert not gmsh.initialized
    assert gmsh.finalize_calls == 1


def test_native_borrow_is_dormant_then_reactivates_repeatedly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(
        initialized=True,
        names=("external",),
        current="external",
    )
    session = _enter_session(monkeypatch, gmsh)
    capability = session.prepare_native_borrow()
    gmsh.calls.clear()

    with pytest.raises(GeometryStateError, match="inactive or revoked"):
        capability.borrow()
    assert gmsh.calls == []

    session.validate_native_borrow(capability, "complete mesh generation")
    session.activate_native_borrow(capability)
    gmsh.model.setCurrent("external")
    gmsh.calls.clear()

    assert capability.borrow() is gmsh.model
    assert capability.borrow() is gmsh.model
    assert gmsh.model.current == "facade"
    assert gmsh.calls.count(("model.setCurrent", "facade")) == 1


def test_same_name_replacement_is_neither_activated_borrowed_nor_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh()
    session = _enter_session(monkeypatch, gmsh)
    capability = session.prepare_native_borrow()
    session.validate_native_borrow(capability, "complete mesh generation")
    session.activate_native_borrow(capability)

    gmsh.model.remove()
    gmsh.model.add("facade")
    gmsh.model.models["facade"]["replacement"] = ["retained"]

    with pytest.raises(GeometryStateError, match="incarnation.*replaced"):
        session.activate("query")
    with pytest.raises(GeometryStateError, match="incarnation.*replaced"):
        capability.borrow()

    session.revoke_borrows()
    session.remove_created_model()
    session.finalize_owned_session(initialized=True)

    assert not session.created_model
    assert gmsh.model.models["facade"]["replacement"] == ["retained"]
    assert gmsh.initialized
    assert gmsh.finalize_calls == 0


def test_outer_session_owner_does_not_finalize_inner_same_name_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh()
    _install_backend(monkeypatch, gmsh)
    outer = _GmshModelSession("outer")
    inner = _GmshModelSession("inner")
    outer.enter()
    inner.enter()

    gmsh.model.remove()
    gmsh.model.add("inner")
    gmsh.model.models["inner"]["replacement"] = ["retained"]

    inner.remove_created_model()
    inner.restore_prior_model()
    inner.finalize_owned_session(initialized=True)
    outer.remove_created_model()
    outer.restore_prior_model()
    outer.finalize_owned_session(initialized=True)

    assert tuple(gmsh.model.models) == ("inner",)
    assert gmsh.model.models["inner"]["replacement"] == ["retained"]
    assert gmsh.initialized
    assert gmsh.finalize_calls == 0


def test_native_borrow_revocation_is_idempotent_and_native_call_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(initialized=True)
    session = _enter_session(monkeypatch, gmsh)
    capability = session.prepare_native_borrow()
    session.activate_native_borrow(capability)
    gmsh.calls.clear()

    session.revoke_borrows()
    session.revoke_borrows()

    assert gmsh.calls == []
    with pytest.raises(GeometryStateError, match="inactive or revoked"):
        capability.borrow()
    assert gmsh.calls == []
    with pytest.raises(GeometryStateError, match="revoked"):
        session.prepare_native_borrow()


def test_nested_session_borrow_epochs_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(initialized=True)
    _install_backend(monkeypatch, gmsh)
    outer = _GmshModelSession("outer")
    inner = _GmshModelSession("inner")
    outer.enter()
    outer_borrow = outer.prepare_native_borrow()
    outer.activate_native_borrow(outer_borrow)
    inner.enter()
    inner_borrow = inner.prepare_native_borrow()
    inner.activate_native_borrow(inner_borrow)

    inner.revoke_borrows()

    with pytest.raises(GeometryStateError, match="inactive or revoked"):
        inner_borrow.borrow()
    assert outer_borrow.borrow() is gmsh.model
    assert gmsh.model.current == "outer"


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


def test_owned_session_default_model_does_not_block_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(names=("",), current="")
    session = _enter_session(monkeypatch, gmsh)

    session.remove_created_model()
    session.restore_prior_model()
    session.finalize_owned_session(initialized=True)

    assert tuple(gmsh.model.models) == ("",)
    assert gmsh.finalize_calls == 1
    assert not gmsh.initialized


def test_owned_session_baseline_same_name_replacement_blocks_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(names=("prior",), current="prior")
    session = _enter_session(monkeypatch, gmsh)
    session.remove_created_model()
    gmsh.model.models["prior"] = {"replacement": ["retained"]}
    gmsh.model.current = "prior"

    session.finalize_owned_session(initialized=True)

    assert gmsh.model.models["prior"]["replacement"] == ["retained"]
    assert gmsh.finalize_calls == 0
    assert gmsh.initialized


def test_missing_model_clears_prior_identity_inspection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh(names=("",), current="")
    session = _enter_session(monkeypatch, gmsh)
    gmsh.model.fail_get_attribute_count = 1

    with pytest.raises(RuntimeError, match="fake getAttribute failure"):
        session.remove_created_model()

    del gmsh.model.models["facade"]
    gmsh.model.current = ""
    session.remove_created_model()
    session.finalize_owned_session(initialized=True)

    assert not session.created_model
    assert gmsh.finalize_calls == 1
    assert not gmsh.initialized


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
        ("model.getAttribute", _MODEL_INCARNATION_ATTRIBUTE),
        ("model.remove", "facade"),
        ("model.list",),
        ("model.setCurrent", "prior"),
        ("model.list",),
        ("model.getCurrent",),
        ("model.setCurrent", "prior"),
        (
            "model.getAttribute",
            _SESSION_BASELINE_INCARNATION_ATTRIBUTE,
        ),
        ("model.getCurrent",),
        ("model.setCurrent", "facade"),
        ("model.getAttribute", _MODEL_INCARNATION_ATTRIBUTE),
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


def test_owned_session_finalization_revalidates_sole_remaining_owned_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gmsh = _FakeGmsh()
    session = _enter_session(monkeypatch, gmsh)
    gmsh.model.fail_remove_count = 1
    gmsh.fail_finalize_count = 1

    errors = session.cleanup_after_failed_entry()

    assert tuple(operation for operation, _error in errors) == (
        "remove facade model",
        "finalize owned session",
    )
    assert tuple(str(error) for _operation, error in errors) == (
        "fake remove failure",
        "fake finalize failure",
    )
    assert session.created_model
    assert gmsh.initialized
    assert gmsh.finalize_calls == 1

    assert session.cleanup_after_failed_entry() == ()
    assert not session.created_model
    assert not gmsh.initialized
    assert gmsh.finalize_calls == 2


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


def test_real_owned_session_close_finalizes_native_default_model() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import sys

sys.path.insert(0, {str(src_dir)!r})

import gmsh
from fem import geometry

assert not bool(gmsh.isInitialized())
with geometry.model("owned-session-finalization", dimension=1):
    assert bool(gmsh.isInitialized())
    assert gmsh.model.list().count("") == 1
assert not bool(gmsh.isInitialized())
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_real_owned_session_can_initialize_in_background_worker() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, {str(src_dir)!r})

import gmsh
from fem import geometry

assert not bool(gmsh.isInitialized())

def build():
    with geometry.model("background-worker", dimension=2) as cad:
        surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        return cad.area(surface)

with ThreadPoolExecutor(max_workers=1) as executor:
    assert executor.submit(build).result() == 1.0

assert not bool(gmsh.isInitialized())
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_real_owned_session_preserves_added_duplicate_empty_model() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import sys

sys.path.insert(0, {str(src_dir)!r})

import gmsh
from fem import geometry

assert not bool(gmsh.isInitialized())
try:
    with geometry.model("duplicate-empty-model", dimension=1):
        gmsh.model.add("")
        assert gmsh.model.list().count("") == 2
    assert bool(gmsh.isInitialized())
    assert gmsh.model.list().count("") == 2
finally:
    if bool(gmsh.isInitialized()):
        gmsh.clear()
        gmsh.finalize()
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_real_owned_session_preserves_replaced_default_model() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import sys

sys.path.insert(0, {str(src_dir)!r})

import gmsh
from fem import geometry

assert not bool(gmsh.isInitialized())
try:
    with geometry.model("replaced-default-model", dimension=1):
        gmsh.model.setCurrent("")
        gmsh.model.remove()
        gmsh.model.add("")
        assert gmsh.model.list().count("") == 1
    assert bool(gmsh.isInitialized())
    assert gmsh.model.list().count("") == 1
finally:
    if bool(gmsh.isInitialized()):
        gmsh.clear()
        gmsh.finalize()
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
