from __future__ import annotations

from copy import copy
from dataclasses import FrozenInstanceError, replace

import pytest

from fem.mesh import gmsh as meshing
from fem.mesh.gmsh.types import (
    _GeneratedMeshLease,
    _prepare_generated_mesh_reference,
)


class _RecordingNativeBorrow:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0
        self.error: BaseException | None = None

    def borrow(self) -> object:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def _active_reference(
    name: str = "lease-model",
) -> tuple[meshing.GmshMeshRef, _GeneratedMeshLease, _RecordingNativeBorrow]:
    native = _RecordingNativeBorrow(object())
    reference, lease = _prepare_generated_mesh_reference(
        native,
        dimension=2,
        model_name=name,
    )
    lease._activate()
    return reference, lease, native


def test_generated_reference_is_frozen_slotted_and_borrow_is_nonconsuming() -> None:
    reference, _, native = _active_reference()

    assert not hasattr(reference, "__dict__")
    assert reference._borrow_model() is native.result
    assert reference._borrow_model() is native.result
    assert native.calls == 2
    with pytest.raises(FrozenInstanceError):
        reference.model_name = "other"  # type: ignore[misc]


def test_bearer_preserving_copies_remain_usable() -> None:
    reference, _, native = _active_reference()

    assert copy(reference)._borrow_model() is native.result
    assert replace(reference)._borrow_model() is native.result
    assert native.calls == 2


@pytest.mark.parametrize(
    "replacement",
    [
        {"dimension": 3},
        {"model_name": "altered"},
        {"_bearer_token": object()},
    ],
)
def test_altered_bearer_metadata_is_stale(replacement: dict[str, object]) -> None:
    reference, _, native = _active_reference()
    altered = replace(reference, **replacement)

    with pytest.raises(meshing.StaleGmshMeshError, match="inside"):
        altered._borrow_model()
    assert native.calls == 0


def test_lookalike_lease_is_rejected_before_dispatch() -> None:
    native = _RecordingNativeBorrow(object())
    malformed = meshing.GmshMeshRef(2, "lookalike", native, object())

    with pytest.raises(meshing.StaleGmshMeshError, match="lookalike"):
        malformed._borrow_model()
    assert native.calls == 0


def test_nominal_lease_is_sealed_against_subclasses_and_direct_construction() -> None:
    with pytest.raises(TypeError, match="sealed"):

        class _LeaseSubclass(_GeneratedMeshLease):
            pass

    with pytest.raises(TypeError, match="sealed factory"):
        _GeneratedMeshLease(
            _RecordingNativeBorrow(object()),
            2,
            "direct",
            object(),
            _factory_authority=object(),
        )


def test_native_borrow_failure_is_translated_with_preserved_cause() -> None:
    reference, _, native = _active_reference("failed-native")
    native.error = RuntimeError("activation failed")

    with pytest.raises(meshing.StaleGmshMeshError, match="failed-native") as caught:
        reference._borrow_model()

    assert caught.value.__cause__ is native.error
