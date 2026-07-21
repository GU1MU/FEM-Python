from __future__ import annotations

import builtins
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from fem import geometry, materials, post, steps
from fem.core import FEMModel, Mesh2D, Mesh3D, validate_model
from fem.elements import get_element_kernel
from fem.elements.beam_section import parse_beam2_section
from fem.geometry._gmsh import backend as _gmsh_backend
from fem.geometry._gmsh import predicates as _gmsh_predicates
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing
from fem.selection import edges, elements, nodes
from fem.solvers import static_linear
from tests.helpers.gmsh_fake import (
    _AUTO_OPTION_ORIGINALS,
    _ENTITY_DEPENDENT_MESH_CONTROLS,
    _TRANSFORM_UNSAFE_ENTITY_CONTROLS,
    _FakeGmsh,
    _apply_edge_treatment,
    _apply_entity_dependent_mesh_control,
    _apply_foundational_operation,
    _apply_typed_transform,
    _build_fake_topology,
    _entity_control_target,
    _fake_control_boundary_dependency,
    _fake_edge_treatment_topology,
    _fake_entities,
    _fake_mesh_control_targets,
    _fake_threshold,
    _first_requested_options,
    _generate_auto_mesh,
    _generate_mesh,
    _install_backend,
    _mesher,
    _occ_operation_call_count,
    _set_fake_element_blocks,
    _structured_extrude,
)


def test_missing_dependency_message_is_actionable_and_closes_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing_gmsh(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gmsh":
            raise ModuleNotFoundError("No module named 'gmsh'", name="gmsh")
        return real_import(name, *args, **kwargs)

    cad = geometry.model("missing", dimension=2)
    assert isinstance(cad, geometry.GeometryModel)
    monkeypatch.setattr(builtins, "__import__", missing_gmsh)

    with pytest.raises(ModuleNotFoundError, match=r"optional 'cad'.*pip install -e"):
        cad.__enter__()

    with pytest.raises(geometry.GeometryStateError, match="missing.*rectangle"):
        cad.rectangle(0.0, 0.0, 1.0, 1.0)


def test_backend_loader_preserves_internal_dependency_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__
    internal_error = ModuleNotFoundError(
        "No module named 'gmsh_internal_dependency'",
        name="gmsh_internal_dependency",
    )

    def broken_gmsh(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gmsh":
            raise internal_error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_gmsh)

    with pytest.raises(ModuleNotFoundError) as captured:
        _gmsh_backend.load_gmsh()

    assert captured.value is internal_error
    assert "optional 'cad'" not in str(captured.value)


def test_translated_signature_uses_local_scale_far_from_origin() -> None:
    origin = 1.0e9
    length = 0.1
    source = (
        (origin, origin, 0.0, origin + length, origin, 0.0),
        (origin + 0.5 * length, origin, 0.0),
        length,
    )
    terminal = (
        (
            origin,
            origin + length,
            0.0,
            origin + length,
            origin + length,
            0.0,
        ),
        (origin + 0.5 * length, origin + length, 0.0),
        length,
    )
    lateral = (
        (origin, origin, 0.0, origin, origin + length, 0.0),
        (origin, origin + 0.5 * length, 0.0),
        length,
    )
    vector = (0.0, length, 0.0)

    assert _gmsh_predicates._matches_translated_signature(source, terminal, vector)
    assert not _gmsh_predicates._matches_translated_signature(source, lateral, vector)


@pytest.mark.parametrize("dimension", [0, 4, True, "2", None])
def test_model_rejects_invalid_mesh_dimension(dimension: Any) -> None:
    with pytest.raises(ValueError, match="dimension must be 1, 2, or 3"):
        geometry.model("part", dimension=dimension)


def test_owned_session_is_initialized_then_model_is_removed_and_finalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    cad = geometry.model("owned", dimension=2)
    _install_backend(monkeypatch, backend)

    with cad:
        assert cad.name == "owned"
        assert backend.initialized
        assert backend.model.current == "owned"

    assert backend.initialize_calls == 1
    assert backend.finalize_calls == 1
    assert "owned" not in backend.model.models


def test_internal_facade_access_is_session_activation_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    cad = geometry.model("activation-gate", dimension=2)
    _install_backend(monkeypatch, backend)

    with pytest.raises(
        geometry.GeometryStateError,
        match="native facade access.*Gmsh session is not active",
    ):
        _ = cad._gmsh

    with cad:
        backend.model.add("external")
        assert backend.model.current == "external"
        assert cad._gmsh is backend
        assert backend.model.current == "activation-gate"

    with pytest.raises(
        geometry.GeometryStateError,
        match="native facade access.*facade-owned Gmsh model is missing",
    ):
        _ = cad._gmsh
    assert backend.initialized
    assert "external" in backend.model.models


def test_partially_successful_initialize_is_finalized_after_entry_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.fail_initialize_after_state = True
    _install_backend(monkeypatch, backend)
    cad = geometry.model("entry_failure", dimension=2)

    with pytest.raises(RuntimeError, match="fake initialize failure"):
        cad.__enter__()

    assert backend.initialize_calls == 1
    assert backend.finalize_calls == 1
    assert not backend.initialized
    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        cad.entities(2)


def test_failed_entry_reports_model_removal_failure_and_retains_retry_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(initialized=True, names=("prior",), current="prior")
    backend.model.fail_set_current_names.add("facade")
    backend.model.fail_remove = True
    _install_backend(monkeypatch, backend)
    cad = geometry.model("facade", dimension=2)

    with pytest.raises(RuntimeError, match="setCurrent") as captured:
        cad.__enter__()

    assert any(
        "remove facade model" in note
        for note in getattr(captured.value, "__notes__", ())
    )
    assert "facade" in backend.model.models
    with pytest.raises(geometry.GeometryStateError, match="already exists"):
        with geometry.model("facade", dimension=2):
            pass

    backend.model.fail_remove = False
    cad.__exit__(None, None, None)
    assert "facade" not in backend.model.models
    assert backend.model.current == "prior"


def test_failed_entry_reports_prior_model_restoration_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(initialized=True, names=("prior",), current="prior")
    backend.model.fail_set_current_names.update({"facade", "prior"})
    _install_backend(monkeypatch, backend)
    cad = geometry.model("facade", dimension=2)

    with pytest.raises(RuntimeError, match="facade") as captured:
        cad.__enter__()

    assert any(
        "restore prior model" in note
        for note in getattr(captured.value, "__notes__", ())
    )
    cad.__exit__(None, None, None)
    assert backend.model.current == "prior"


def test_failed_entry_reports_finalize_failure_and_retains_session_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.fail_finalize = True
    _install_backend(monkeypatch, backend)
    cad = geometry.model(" ", dimension=2)

    with pytest.raises(geometry.GeometryStateError, match="nonempty") as captured:
        cad.__enter__()

    assert any(
        "finalize owned session" in note
        for note in getattr(captured.value, "__notes__", ())
    )
    assert backend.initialized
    cad.__exit__(None, None, None)
    assert backend.finalize_calls == 2
    assert not backend.initialized


def test_external_session_restores_prior_model_and_removes_only_facade_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(
        initialized=True,
        names=("prior", "other"),
        current="prior",
    )
    _install_backend(monkeypatch, backend)

    with geometry.model("facade", dimension=3):
        backend.model.setCurrent("other")

    assert backend.finalize_calls == 0
    assert tuple(backend.model.models) == ("prior", "other")
    assert backend.model.current == "prior"


def test_model_identity_is_read_only_and_cleanup_uses_the_created_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(
        initialized=True,
        names=("prior", "other"),
        current="prior",
    )
    _install_backend(monkeypatch, backend)

    with geometry.model("facade", dimension=2) as cad:
        with pytest.raises(AttributeError):
            cad.name = "other"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            cad.dimension = 3  # type: ignore[misc]
        assert cad.name == "facade"
        assert cad.dimension == 2
        backend.model.setCurrent("other")

    assert tuple(backend.model.models) == ("prior", "other")
    assert backend.model.current == "prior"
    assert [call for call in backend.model.calls if call[0] == "remove"] == [
        ("remove", "facade")
    ]


def test_valid_empty_name_model_is_restored_after_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(initialized=True, names=("",), current="")
    _install_backend(monkeypatch, backend)

    with geometry.model("facade", dimension=2):
        pass

    assert tuple(backend.model.models) == ("",)
    assert backend.model.current == ""
    assert ("setCurrent", "") in backend.model.calls


def test_model_name_collision_is_rejected_before_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(initialized=True, names=("taken",), current="taken")
    _install_backend(monkeypatch, backend)

    with pytest.raises(geometry.GeometryStateError, match="already exists"):
        with geometry.model("taken", dimension=2):
            pass

    assert ("add", "taken") not in backend.model.calls
    assert backend.model.current == "taken"


def test_user_exception_restores_external_model_without_being_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(initialized=True, names=("prior",), current="prior")
    _install_backend(monkeypatch, backend)

    with pytest.raises(LookupError, match="primary"):
        with geometry.model("facade", dimension=2):
            raise LookupError("primary")

    assert backend.model.current == "prior"
    assert "facade" not in backend.model.models


def test_cleanup_failure_is_noted_without_masking_primary_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(initialized=True)
    _install_backend(monkeypatch, backend)

    with pytest.raises(LookupError, match="primary") as captured:
        with geometry.model("facade", dimension=2):
            backend.model.fail_remove = True
            raise LookupError("primary")

    assert any("remove facade model" in note for note in captured.value.__notes__)


def test_cleanup_failure_without_primary_raises_contextual_geometry_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(initialized=True)
    _install_backend(monkeypatch, backend)

    with pytest.raises(geometry.GeometryError, match="facade.*remove facade model"):
        with geometry.model("facade", dimension=2):
            backend.model.fail_remove = True


def test_session_inspection_failure_does_not_mask_primary_and_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(initialized=True)
    _install_backend(monkeypatch, backend)
    cad = geometry.model("inspection-primary", dimension=2)

    with pytest.raises(LookupError, match="primary") as captured:
        with cad:
            backend.fail_is_initialized_count = 1
            raise LookupError("primary")

    assert any(
        "inspect Gmsh session state" in note
        for note in captured.value.__notes__
    )
    assert "inspection-primary" in backend.model.models

    cad.__exit__(None, None, None)
    assert "inspection-primary" not in backend.model.models


def test_session_inspection_failure_without_primary_is_contextual_and_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(initialized=True)
    _install_backend(monkeypatch, backend)
    cad = geometry.model("inspection-cleanup", dimension=2)

    with pytest.raises(
        geometry.GeometryError,
        match="inspection-cleanup.*inspect Gmsh session state",
    ) as captured:
        with cad:
            backend.fail_is_initialized_count = 1

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert "inspection-cleanup" in backend.model.models

    cad.__exit__(None, None, None)
    assert "inspection-cleanup" not in backend.model.models


def test_cleanup_retains_later_failures_as_notes_and_retries_every_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(names=("prior",), current="prior")
    _install_backend(monkeypatch, backend)
    cad = geometry.model("multi-cleanup", dimension=2)
    cad.__enter__()
    backend.model.fail_remove = True
    backend.model.fail_set_current_names.add("prior")
    backend.fail_finalize = True

    with pytest.raises(
        geometry.GeometryError,
        match="multi-cleanup.*remove facade model",
    ) as captured:
        cad.__exit__(None, None, None)

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert any(
        "restore prior model" in note for note in captured.value.__notes__
    )
    assert any(
        "finalize owned session" in note for note in captured.value.__notes__
    )
    assert backend.initialized
    assert "multi-cleanup" in backend.model.models
    assert "prior" in backend.model.models

    backend.model.fail_remove = False
    cad.__exit__(None, None, None)
    assert backend.model.current == "prior"
    assert "multi-cleanup" not in backend.model.models
    assert backend.finalize_calls == 2
    assert not backend.initialized


def test_nested_contexts_restore_current_models_in_lifo_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer", dimension=2):
        assert backend.model.current == "outer"
        with geometry.model("inner", dimension=2):
            assert backend.model.current == "inner"
        assert backend.model.current == "outer"
    assert backend.finalize_calls == 1


def test_operations_reactivate_facade_model_and_missing_model_is_contextual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(initialized=True, names=("external",), current="external")
    _install_backend(monkeypatch, backend)

    with geometry.model("facade", dimension=2) as cad:
        backend.model.setCurrent("external")
        assert cad.entities(2) == ()
        assert backend.model.current == "facade"
        backend.model.remove()
        with pytest.raises(geometry.GeometryStateError, match="facade.*entities"):
            cad.entities(2)


def test_calls_before_entry_and_after_exit_raise_contextual_state_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    cad = geometry.model("part", dimension=2)

    with pytest.raises(geometry.GeometryStateError, match="part.*entities"):
        cad.entities(2)
    with cad:
        assert cad.entities(2) == ()
    with pytest.raises(geometry.GeometryStateError, match="part.*entities"):
        cad.entities(2)


def test_occ_primitives_forward_normalized_arguments_and_return_typed_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("surfaces", dimension=2) as cad:
        rectangle = cad.rectangle(
            1,
            2,
            3,
            4,
            rounded_radius=0.25,
        )
        disk = cad.disk(5, 6, 2, radius_y=1)
        y_major_disk = cad.disk(8, 9, 1, radius_y=2)

    assert (rectangle.dimension, rectangle.tag) == (2, 1)
    assert (disk.dimension, disk.tag) == (2, 2)
    assert (y_major_disk.dimension, y_major_disk.tag) == (2, 3)
    assert ("addRectangle", 1.0, 2.0, 0.0, 3.0, 4.0, -1, 0.25) in (
        backend.model.occ.calls
    )
    assert ("addDisk", 5.0, 6.0, 0.0, 2.0, 1.0, -1, (), ()) in (
        backend.model.occ.calls
    )
    assert (
        "addDisk",
        8.0,
        9.0,
        0.0,
        2.0,
        1.0,
        -1,
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
    ) in backend.model.occ.calls


def test_line_primitives_forward_spatial_coordinates_and_return_typed_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("members", dimension=1) as cad:
        start = cad.point(1, 2, 3)
        end = cad.point(4, 5)
        member = cad.line(start, end)

    assert (start.dimension, start.tag) == (0, 1)
    assert (end.dimension, end.tag) == (0, 2)
    assert (member.dimension, member.tag) == (1, 1)
    assert ("addPoint", 1.0, 2.0, 3.0, 0.0, -1) in backend.model.occ.calls
    assert ("addPoint", 4.0, 5.0, 0.0, 0.0, -1) in backend.model.occ.calls
    assert ("addLine", 1, 2, -1) in backend.model.occ.calls


def test_line_requires_distinct_live_point_references_before_add_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("members", dimension=1) as cad:
        start = cad.point(0, 0, 0)
        end = cad.point(1, 0, 0)
        other_at_end = cad.point(1, 0, 0)
        assert end != other_at_end
        member = cad.line(start, end)

        before = list(backend.model.occ.calls)
        with pytest.raises(ValueError, match="distinct|duplicate"):
            cad.line(start, start)
        with pytest.raises(TypeError, match="EntityRef"):
            cad.line(start, object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="dimension-zero|point"):
            cad.line(start, member)
        assert backend.model.occ.calls == before

        backend.model._current_data()["entities"].remove((0, end.tag))
        with pytest.raises(geometry.StaleEntityError, match="no longer exists"):
            cad.line(start, end)
        assert not any(call[0] == "addLine" for call in backend.model.occ.calls[len(before) :])


def test_line_rejects_cross_model_endpoint_before_add_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer", dimension=1) as outer:
        outer_point = outer.point(0, 0, 0)
        with geometry.model("inner", dimension=1) as inner:
            inner_point = inner.point(1, 0, 0)
            before = list(backend.model.occ.calls)
            with pytest.raises(geometry.EntityOwnershipError, match="inner"):
                inner.line(outer_point, inner_point)
            assert backend.model.occ.calls == before


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, point: cad.point(float("nan"), 0, 0),
        lambda cad, point: cad.rectangle(0, 0, 1, 1),
        lambda cad, point: cad.disk(0, 0, 1),
        lambda cad, point: cad.box(0, 0, 0, 1, 1, 1),
        lambda cad, point: cad.cylinder(0, 0, 0, 1, 0, 0, 1),
        lambda cad, point: cad.extrude([point], 1, 0, 0),
    ],
)
def test_1d_facade_rejects_invalid_or_higher_dimensional_primitives_pre_backend(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("members", dimension=1) as cad:
        point = cad.point(0, 0, 0)
        before = list(backend.model.occ.calls)
        with pytest.raises(ValueError):
            operation(cad, point)
        assert backend.model.occ.calls == before


def test_1d_transform_is_spatial_and_topology_remains_editable_before_meshing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("members", dimension=1) as cad:
        start = cad.point(0, 0, 0)
        end = cad.point(1, 0, 0)
        member = cad.line(start, end)
        assert cad.translate([member], 1, 2, 3) == (member,)
        assert cad.rotate([member], 0, 0, 0, 1, 1, 0, 0.5) == (member,)
        third = cad.point(2, 0, 0)
        assert cad.line(end, third).dimension == 1

    assert ("translate", ((1, 1),), 1.0, 2.0, 3.0) in backend.model.occ.calls
    assert (
        "rotate",
        ((1, 1),),
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.5,
    ) in backend.model.occ.calls


def test_volume_primitives_forward_arguments_in_three_dimensional_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("volumes", dimension=3) as cad:
        box = cad.box(1, 2, 3, 4, 5, 6)
        cylinder = cad.cylinder(0, 1, 2, 0, 0, 3, 4, angle=1.5)

    assert (box.dimension, box.tag) == (3, 1)
    assert (cylinder.dimension, cylinder.tag) == (3, 2)
    assert ("addBox", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1) in (
        backend.model.occ.calls
    )
    assert (
        "addCylinder",
        0.0,
        1.0,
        2.0,
        0.0,
        0.0,
        3.0,
        4.0,
        -1,
        1.5,
    ) in backend.model.occ.calls


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad: cad.rectangle(0, 0, 0, 1),
        lambda cad: cad.rectangle(float("nan"), 0, 1, 1),
        lambda cad: cad.rectangle(0, 0, 1, 1, rounded_radius=-1),
        lambda cad: cad.rectangle(0, 0, 1, 2, rounded_radius=0.5),
        lambda cad: cad.rectangle(0, 0, 1, 2, rounded_radius=0.6),
        lambda cad: cad.rectangle(0, 0, 1, 1, z=2.0e-10),
        lambda cad: cad.disk(0, 0, 0),
        lambda cad: cad.disk(0, 0, 1, radius_y=-1),
        lambda cad: cad.box(0, 0, 0, 1, 1, 1),
        lambda cad: cad.cylinder(0, 0, 0, 0, 0, 1, 1),
    ],
)
def test_invalid_2d_primitive_inputs_fail_before_occ_call(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid", dimension=2) as cad:
        before = list(backend.model.occ.calls)
        with pytest.raises((ValueError, geometry.GeometryStateError)):
            operation(cad)
        assert backend.model.occ.calls == before


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad: cad.box(0, 0, 0, 1, -1, 1),
        lambda cad: cad.cylinder(0, 0, 0, 0, 0, 0, 1),
        lambda cad: cad.cylinder(0, 0, 0, 0, 0, 1, 0),
        lambda cad: cad.cylinder(0, 0, 0, 0, 0, 1, 1, angle=0),
        lambda cad: cad.cylinder(0, 0, 0, 0, 0, 1, 1, angle=7),
    ],
)
def test_invalid_3d_primitive_inputs_fail_before_occ_call(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid", dimension=3) as cad:
        before = list(backend.model.occ.calls)
        with pytest.raises(ValueError):
            operation(cad)
        assert backend.model.occ.calls == before


def test_cross_model_reference_is_rejected_even_when_dimension_and_tag_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer", dimension=2) as outer:
        outer_surface = outer.rectangle(0, 0, 1, 1)
        with geometry.model("inner", dimension=2) as inner:
            inner_surface = inner.rectangle(0, 0, 1, 1)
            assert (outer_surface.dimension, outer_surface.tag) == (
                inner_surface.dimension,
                inner_surface.tag,
            )
            with pytest.raises(geometry.EntityOwnershipError, match="inner"):
                inner.translate([outer_surface], 1, 0, 0)
        assert outer.translate([outer_surface], 1, 0, 0) == (outer_surface,)


def test_raw_escape_invalidates_references_and_entity_reacquires_current_occ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("raw", dimension=2) as cad:
        original = cad.rectangle(0, 0, 1, 1)
        raw_occ = cad.raw_occ
        with pytest.raises(geometry.StaleEntityError, match="raw"):
            cad.translate([original], 1, 0, 0)

        raw_tag = raw_occ.addRectangle(2, 0, 0, 1, 1)
        reacquired = cad.entity(2, raw_tag)
        assert cad.entity(2, raw_tag) == reacquired
        assert cad.translate([reacquired], 1, 0, 0) == (reacquired,)
        raw_model = cad.raw_model
        assert raw_model is backend.model
        with pytest.raises(geometry.StaleEntityError):
            cad.translate([reacquired], 1, 0, 0)


def test_entity_rejects_missing_occ_pair_and_external_removal_becomes_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("liveness", dimension=2) as cad:
        with pytest.raises(geometry.StaleEntityError, match="2, 99"):
            cad.entity(2, 99)
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].remove((2, surface.tag))
        with pytest.raises(geometry.StaleEntityError, match="no longer exists"):
            cad.translate([surface], 1, 0, 0)


def test_copy_batches_by_dimension_and_restores_caller_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("copy-order", dimension=3) as cad:
        point = cad.point(0, 0, 0)
        surface = cad.rectangle(0, 0, 1, 1)
        volume = cad.box(0, 0, 0, 1, 1, 1)
        sources = (surface, point, volume)

        copied = cad.copy(item for item in sources)

        assert tuple(item.dimension for item in copied) == (2, 0, 3)
        assert len(set(copied)) == len(copied)
        assert all(output != source for output, source in zip(copied, sources, strict=True))
        assert all(cad.entity(item.dimension, item.tag) == item for item in (*sources, *copied))

    assert [
        call for call in backend.model.occ.calls if call[0] == "copy"
    ] == [
        ("copy", ((2, surface.tag),)),
        ("copy", ((0, point.tag),)),
        ("copy", ((3, volume.tag),)),
    ]


@pytest.mark.parametrize(
    ("malformation", "message"),
    [
        ("count", "unexpected entity count"),
        ("dimension", "unexpected dimension"),
        ("duplicate", "duplicate entities"),
        ("source_reuse", "fresh entities"),
        ("missing", "missing entity"),
    ],
)
def test_malformed_copy_result_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    message: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"copy-malformed-{malformation}", dimension=2) as cad:
        first = cad.rectangle(0, 0, 1, 1)
        second = cad.rectangle(2, 0, 1, 1)
        unrelated = cad.rectangle(4, 0, 1, 1)
        sources = [first, second] if malformation == "duplicate" else [first]
        configured = {
            "count": [],
            "dimension": [(1, 90)],
            "duplicate": [(2, 90), (2, 90)],
            "source_reuse": [(2, first.tag)],
            "missing": [(2, 90)],
        }[malformation]
        backend.model.occ.copy_results[2] = configured
        if malformation == "missing":
            backend.model.occ.copy_register_outputs = False

        with pytest.raises(geometry.GeometryError, match=message):
            cad.copy(sources)

        for old_reference in (first, second, unrelated):
            with pytest.raises(geometry.StaleEntityError):
                cad.translate([old_reference], 1, 0, 0)
        reacquired = cad.entity(2, unrelated.tag)
        assert cad.entity(2, unrelated.tag) == reacquired
        with pytest.raises(geometry.GeometryStateError, match="dependencies unknown"):
            cad.translate([reacquired], 1, 0, 0)


def test_native_copy_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("copy-native-failure", dimension=2) as cad:
        source = cad.rectangle(0, 0, 1, 1)
        unrelated = cad.rectangle(2, 0, 1, 1)
        backend.model.occ.fail_next.add("copy")

        with pytest.raises(RuntimeError, match="fake copy failure"):
            cad.copy([source])

        for old_reference in (source, unrelated):
            with pytest.raises(geometry.StaleEntityError):
                cad.translate([old_reference], 1, 0, 0)
        reacquired = cad.entity(2, unrelated.tag)
        assert cad.entity(2, unrelated.tag) == reacquired
        with pytest.raises(geometry.GeometryStateError, match="dependencies unknown"):
            cad.translate([reacquired], 1, 0, 0)


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, entity: cad.copy([]),
        lambda cad, entity: cad.copy([entity, entity]),
        lambda cad, entity: cad.copy([object()]),
    ],
)
def test_invalid_copy_inputs_fail_before_occ_call(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("copy-invalid", dimension=2) as cad:
        entity = cad.rectangle(0, 0, 1, 1)
        copy_calls = _occ_operation_call_count(backend, "copy")
        with pytest.raises((ValueError, TypeError)):
            operation(cad, entity)
        assert _occ_operation_call_count(backend, "copy") == copy_calls


@pytest.mark.parametrize(
    "operation",
    ["copy", "mirror", "scale", "intersect", "fragment"],
)
def test_foundational_operations_reject_foreign_entities_before_native_call(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    native_operation = "dilate" if operation == "scale" else operation

    with geometry.model("foundational-owner-outer", dimension=2) as outer:
        foreign = outer.rectangle(0, 0, 1, 1)
        with geometry.model("foundational-owner-inner", dimension=2) as inner:
            tool = inner.rectangle(2, 0, 1, 1)
            native_calls = _occ_operation_call_count(backend, native_operation)

            with pytest.raises(geometry.EntityOwnershipError, match="another"):
                _apply_foundational_operation(inner, operation, foreign, tool)

            assert (
                _occ_operation_call_count(backend, native_operation) == native_calls
            )


@pytest.mark.parametrize(
    "operation",
    ["copy", "mirror", "scale", "intersect", "fragment"],
)
def test_foundational_operations_reject_externally_stale_entities_pre_native(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    native_operation = "dilate" if operation == "scale" else operation

    with geometry.model(f"foundational-stale-{operation}", dimension=2) as cad:
        source = cad.rectangle(0, 0, 1, 1)
        tool = cad.rectangle(2, 0, 1, 1)
        backend.model._current_data()["entities"].remove((2, source.tag))
        native_calls = _occ_operation_call_count(backend, native_operation)

        with pytest.raises(geometry.StaleEntityError, match="no longer exists"):
            _apply_foundational_operation(cad, operation, source, tool)

        assert _occ_operation_call_count(backend, native_operation) == native_calls


@pytest.mark.parametrize(
    "operation",
    ["copy", "mirror", "scale", "intersect", "fragment"],
)
def test_foundational_operations_reactivate_owner_and_stay_model_local(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("foundational-local-outer", dimension=2) as outer:
        outer_source = outer.rectangle(0, 0, 1, 1)
        outer_tool = outer.rectangle(2, 0, 1, 1)
        with geometry.model("foundational-local-inner", dimension=2) as inner:
            inner.rectangle(10, 0, 1, 1)
            inner.rectangle(12, 0, 1, 1)
            inner_snapshot = set(
                backend.model.models["foundational-local-inner"]["entities"]
            )

            _apply_foundational_operation(
                outer,
                operation,
                outer_source,
                outer_tool,
            )

            assert backend.model.current == "foundational-local-outer"
            assert (
                backend.model.models["foundational-local-inner"]["entities"]
                == inner_snapshot
            )
            assert inner.entities(2)
            assert backend.model.current == "foundational-local-inner"


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad: cad.copy([]),
        lambda cad: cad.mirror([], 1, 0, 0, 0),
        lambda cad: cad.scale([], 0, 0, 0, 1, 1, 1),
        lambda cad: cad.intersect([], []),
        lambda cad: cad.fragment([], []),
    ],
)
def test_foundational_operations_reject_new_and_closed_states(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    cad = geometry.GeometryModel("foundational-states", dimension=2)

    with pytest.raises(geometry.GeometryStateError, match="NEW"):
        operation(cad)
    with cad:
        pass
    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        operation(cad)


@pytest.mark.parametrize(
    "operation",
    ["copy", "mirror", "scale", "intersect", "fragment"],
)
def test_raw_occ_access_stales_foundational_operation_inputs_pre_native(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    native_operation = "dilate" if operation == "scale" else operation

    with geometry.model(f"foundational-raw-{operation}", dimension=2) as cad:
        source = cad.rectangle(0, 0, 1, 1)
        tool = cad.rectangle(2, 0, 1, 1)
        assert cad.raw_occ is backend.model.occ
        native_calls = _occ_operation_call_count(backend, native_operation)

        with pytest.raises(geometry.StaleEntityError):
            _apply_foundational_operation(cad, operation, source, tool)

        assert _occ_operation_call_count(backend, native_operation) == native_calls


def test_destructive_boolean_preserves_mapping_and_replaces_reused_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("boolean", dimension=2) as cad:
        first = cad.rectangle(0, 0, 2, 1)
        tool = cad.disk(1, 0.5, 0.25)
        backend.model._current_data()["entities"].add((1, 7))
        backend.model.boundary_result = [(1, 7)]
        old_boundary = cad.entity(1, 7)
        backend.model.occ.boolean_results["cut"] = (
            [(2, first.tag)],
            [[(2, first.tag)], []],
        )

        result = cad.cut((item for item in [first]), [tool])

        assert result.outputs == result.input_map[0]
        assert result.input_map[1] == ()
        assert result.outputs[0] != first
        with pytest.raises(geometry.StaleEntityError):
            cad.translate([first], 1, 0, 0)
        with pytest.raises(geometry.StaleEntityError):
            cad.translate([tool], 1, 0, 0)
        with pytest.raises(geometry.StaleEntityError):
            cad.translate([old_boundary], 1, 0, 0)
        assert cad.translate(result.outputs, 1, 0, 0) == result.outputs


@pytest.mark.parametrize("operation", ["fragment", "intersect"])
def test_non_destructive_boolean_preserves_input_references(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("boolean", dimension=2) as cad:
        first = cad.rectangle(0, 0, 2, 1)
        tool = cad.disk(1, 0.5, 0.25)

        getattr(cad, operation)(
            [first],
            [tool],
            remove_objects=False,
            remove_tools=False,
        )

        assert cad.translate([first, tool], 1, 0, 0) == (first, tool)


def test_partially_destructive_boolean_preserves_only_kept_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("boolean", dimension=2) as cad:
        first = cad.rectangle(0, 0, 2, 1)
        tool = cad.disk(1, 0.5, 0.25)
        backend.model.occ.boolean_results["cut"] = (
            [(2, first.tag)],
            [[(2, first.tag)], []],
        )

        result = cad.cut(
            [first],
            [tool],
            remove_objects=False,
            remove_tools=True,
        )

        assert result.outputs == (first,)
        assert cad.translate([first], 1, 0, 0) == (first,)
        with pytest.raises(geometry.StaleEntityError):
            cad.translate([tool], 1, 0, 0)


@pytest.mark.parametrize("operation", ["fuse", "intersect"])
def test_failed_boolean_preserves_input_liveness(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("boolean", dimension=2) as cad:
        first = cad.rectangle(0, 0, 2, 1)
        second = cad.rectangle(2, 0, 1, 1)
        backend.model.occ.fail_next.add(operation)

        with pytest.raises(RuntimeError, match=f"fake {operation} failure"):
            getattr(cad, operation)([first], [second])

        assert cad.translate([first, second], 1, 0, 0) == (first, second)


@pytest.mark.parametrize(
    ("malformation", "message"),
    [
        ("map_length", "invalid input map"),
        ("invalid_native_dimension", "invalid boolean output data"),
        ("facade_dimension", "above the facade dimension"),
        ("missing_map_entity", "missing entity"),
    ],
)
@pytest.mark.parametrize("operation", ["cut", "intersect"])
def test_malformed_destructive_boolean_result_invalidates_changed_inputs(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    message: str,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("boolean", dimension=2) as cad:
        first = cad.rectangle(0, 0, 2, 1)
        tool = cad.disk(1, 0.5, 0.25)
        unrelated = cad.rectangle(10, 0, 1, 1)
        backend.model._current_data()["entities"].add((1, 7))
        backend.model.boundary_result = [(1, 7)]
        old_boundary = cad.entity(1, 7)
        if malformation == "map_length":
            backend.model.occ.boolean_results[operation] = (
                [(2, first.tag)],
                [[(2, first.tag)]],
            )
        elif malformation == "invalid_native_dimension":
            backend.model.occ.boolean_results[operation] = (
                [(2, first.tag)],
                [[(4, first.tag)], []],
            )
        elif malformation == "facade_dimension":
            backend.model.occ.boolean_results[operation] = (
                [(2, first.tag)],
                [[(3, 90)], []],
            )
        else:
            backend.model.occ.boolean_results[operation] = (
                [(2, first.tag)],
                [[(2, first.tag)], [(1, 999)]],
            )
            backend.model.occ.boolean_register_map_outputs = False

        with pytest.raises(geometry.GeometryError, match=message):
            getattr(cad, operation)([first], [tool])

        for old_reference in (first, tool, unrelated, old_boundary):
            with pytest.raises(geometry.StaleEntityError):
                cad.translate([old_reference], 1, 0, 0)
        reacquired = cad.entity(2, unrelated.tag)
        assert cad.entity(2, unrelated.tag) == reacquired
        assert cad.translate([reacquired], 1, 0, 0) == (reacquired,)


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, first, second: cad.fuse([], [second]),
        lambda cad, first, second: cad.cut([first, first], [second]),
        lambda cad, first, second: cad.fragment([first], [first]),
        lambda cad, first, second: cad.fuse([first], [second], remove_objects=1),
    ],
)
def test_invalid_boolean_inputs_fail_before_occ_call(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("boolean", dimension=2) as cad:
        first = cad.rectangle(0, 0, 1, 1)
        second = cad.rectangle(2, 0, 1, 1)
        before = list(backend.model.occ.calls)
        with pytest.raises((ValueError, TypeError)):
            operation(cad, first, second)
        assert backend.model.occ.calls == before


@pytest.mark.parametrize("operation", ["fuse", "cut"])
def test_fuse_and_cut_require_one_common_dimension(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("mixed", dimension=3) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        volume = cad.box(0, 0, 0, 1, 1, 1)
        with pytest.raises(ValueError, match="common dimension"):
            getattr(cad, operation)([surface], [volume])


def test_intersect_accepts_different_homogeneous_group_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("intersect-cross-dimension", dimension=3) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        volume = cad.box(0, 0, 0, 1, 1, 1)

        result = cad.intersect([surface], [volume])

        assert tuple(item.dimension for item in result.outputs) == (2,)
        assert result.input_map == (result.outputs, ())
        assert backend.model.occ.calls[-1] == (
            "intersect",
            ((2, surface.tag),),
            ((3, volume.tag),),
            -1,
            True,
            True,
        )


@pytest.mark.parametrize("mixed_group", ["objects", "tools"])
def test_intersect_rejects_mixed_dimensions_inside_either_input_group(
    monkeypatch: pytest.MonkeyPatch,
    mixed_group: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"intersect-mixed-{mixed_group}", dimension=3) as cad:
        curve = _fake_entities(cad, backend, 1, 90)[0]
        surface = cad.rectangle(0, 0, 1, 1)
        volume = cad.box(0, 0, 0, 1, 1, 1)
        objects = [surface, volume] if mixed_group == "objects" else [surface]
        tools = [curve] if mixed_group == "objects" else [curve, volume]
        intersect_calls = _occ_operation_call_count(backend, "intersect")

        with pytest.raises(ValueError, match="each have one common dimension"):
            cad.intersect(objects, tools)

        assert _occ_operation_call_count(backend, "intersect") == intersect_calls


def test_empty_intersection_is_a_valid_boolean_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("intersect-empty", dimension=2) as cad:
        first = cad.rectangle(0, 0, 1, 1)
        second = cad.rectangle(2, 0, 1, 1)
        backend.model.occ.boolean_results["intersect"] = ([], [[], []])

        result = cad.intersect([first], [second])

        assert result.outputs == ()
        assert result.input_map == ((), ())
        for removed in (first, second):
            with pytest.raises(geometry.StaleEntityError):
                cad.translate([removed], 1, 0, 0)


def test_fragment_accepts_fully_mixed_dimensions_and_exports_map_only_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("fragment-mixed", dimension=3) as cad:
        point = cad.point(0, 0, 0)
        curve = _fake_entities(cad, backend, 1, 90)[0]
        surface = cad.rectangle(0, 0, 1, 1)
        volume = cad.box(0, 0, 0, 1, 1, 1)
        backend.model.occ.boolean_results["fragment"] = (
            [(3, 90)],
            [[(2, 91)], [(3, 90)], [(1, 92)], [(0, 93)]],
        )

        result = cad.fragment([surface, volume], [curve, point])

        assert tuple((item.dimension, item.tag) for item in result.outputs) == (
            (3, 90),
            (2, 91),
            (1, 92),
            (0, 93),
        )
        assert tuple(
            tuple((item.dimension, item.tag) for item in group)
            for group in result.input_map
        ) == (
            ((2, 91),),
            ((3, 90),),
            ((1, 92),),
            ((0, 93),),
        )
        assert backend.model.occ.calls[-1] == (
            "fragment",
            ((2, surface.tag), (3, volume.tag)),
            ((1, curve.tag), (0, point.tag)),
            -1,
            True,
            True,
        )


@pytest.mark.parametrize(
    ("operation", "values"),
    [
        ("fillet", [0.125, np.float64(0.25)]),
        ("chamfer", [0.2, np.float64(0.3)]),
    ],
)
def test_edge_treatments_forward_native_arguments_and_return_modifying_result(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    values: Sequence[float],
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{operation}-forward", dimension=3) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)

        result = _apply_edge_treatment(
            cad,
            operation,
            topology,
            values,
            remove_volumes=False,
        )

        assert result.operation == operation
        assert result.inputs == (topology["volume"],)
        assert result.outputs == result.primary
        assert len(result.primary) == 1
        assert result.primary[0].dimension == 3
        assert result.primary[0].tag != topology["volume"].tag
        assert result.ends == ()
        assert result.sides == ()
        expected_values = tuple(float(value) for value in values)
        if operation == "fillet":
            expected_call = (
                "fillet",
                (topology["volume"].tag,),
                (topology["curve"].tag,),
                expected_values,
                False,
            )
        else:
            expected_call = (
                "chamfer",
                (topology["volume"].tag,),
                (topology["curve"].tag,),
                (topology["surface"].tag,),
                expected_values,
                False,
            )
        assert expected_call in backend.model.occ.calls


@pytest.mark.parametrize("operation", ["fillet", "chamfer"])
@pytest.mark.parametrize("value_count", [2, 4])
def test_edge_treatments_accept_per_edge_and_endpoint_value_vectors(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    value_count: int,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(
        f"{operation}-value-cardinality-{value_count}",
        dimension=3,
    ) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)
        values = [0.05 * (index + 1) for index in range(value_count)]
        if operation == "fillet":
            result = cad.fillet(
                [topology["volume"]],
                [topology["curve"], topology["other_curve"]],
                values,
                remove_volumes=False,
            )
        else:
            result = cad.chamfer(
                [topology["volume"]],
                [topology["curve"], topology["other_curve"]],
                [topology["surface"], topology["nonadjacent_surface"]],
                values,
                remove_volumes=False,
            )

        assert result.primary[0].dimension == 3
        assert _occ_operation_call_count(backend, operation) == 1


@pytest.mark.parametrize(
    "invalid_operation",
    [
        pytest.param(
            lambda cad, topology: cad.fillet(
                [], [topology["curve"]], [0.1]
            ),
            id="empty-volumes",
        ),
        pytest.param(
            lambda cad, topology: cad.fillet(
                [topology["surface"]], [topology["curve"]], [0.1]
            ),
            id="fillet-volume-dimension",
        ),
        pytest.param(
            lambda cad, topology: cad.fillet(
                [topology["volume"]], [topology["surface"]], [0.1]
            ),
            id="fillet-curve-dimension",
        ),
        pytest.param(
            lambda cad, topology: cad.fillet(
                [topology["volume"]],
                [topology["curve"], topology["curve"]],
                [0.1],
            ),
            id="duplicate-curves",
        ),
        pytest.param(
            lambda cad, topology: cad.fillet(
                [topology["volume"]], [topology["curve"]], []
            ),
            id="empty-radii",
        ),
        pytest.param(
            lambda cad, topology: cad.fillet(
                [topology["volume"]], [topology["curve"]], [0.1, 0.2, 0.3]
            ),
            id="invalid-radii-cardinality",
        ),
        pytest.param(
            lambda cad, topology: cad.fillet(
                [topology["volume"]], [topology["curve"]], [0.0]
            ),
            id="zero-radius",
        ),
        pytest.param(
            lambda cad, topology: cad.fillet(
                [topology["volume"]], [topology["curve"]], [math.nan]
            ),
            id="nonfinite-radius",
        ),
        pytest.param(
            lambda cad, topology: cad.fillet(
                [topology["volume"]],
                [topology["curve"]],
                [0.1],
                remove_volumes=1,
            ),
            id="fillet-nonboolean-remove",
        ),
        pytest.param(
            lambda cad, topology: cad.chamfer(
                [topology["volume"]],
                [topology["curve"]],
                [topology["curve"]],
                [0.1],
            ),
            id="chamfer-surface-dimension",
        ),
        pytest.param(
            lambda cad, topology: cad.chamfer(
                [topology["volume"]],
                [topology["curve"], topology["other_curve"]],
                [topology["surface"]],
                [0.1],
            ),
            id="curve-surface-count-mismatch",
        ),
        pytest.param(
            lambda cad, topology: cad.chamfer(
                [topology["volume"]],
                [topology["curve"]],
                [topology["surface"]],
                [0.1, 0.2, 0.3],
            ),
            id="invalid-distance-cardinality",
        ),
        pytest.param(
            lambda cad, topology: cad.chamfer(
                [topology["volume"]],
                [topology["curve"], topology["other_curve"]],
                [topology["surface"], topology["surface"]],
                [0.1],
            ),
            id="duplicate-surfaces",
        ),
        pytest.param(
            lambda cad, topology: cad.chamfer(
                [topology["volume"]],
                [topology["curve"]],
                [topology["surface"]],
                [-0.1],
            ),
            id="negative-distance",
        ),
        pytest.param(
            lambda cad, topology: cad.chamfer(
                [topology["volume"]],
                [topology["curve"]],
                [topology["surface"]],
                [0.1],
                remove_volumes=1,
            ),
            id="chamfer-nonboolean-remove",
        ),
    ],
)
def test_edge_treatment_preflight_rejects_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    invalid_operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("edge-treatment-preflight", dimension=3) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)
        fillet_calls = _occ_operation_call_count(backend, "fillet")
        chamfer_calls = _occ_operation_call_count(backend, "chamfer")

        with pytest.raises((TypeError, ValueError)):
            invalid_operation(cad, topology)

        assert _occ_operation_call_count(backend, "fillet") == fillet_calls
        assert _occ_operation_call_count(backend, "chamfer") == chamfer_calls


@pytest.mark.parametrize("operation", ["fillet", "chamfer"])
def test_edge_treatments_reject_non_3d_facade_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{operation}-2d", dimension=2) as cad:
        surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        curve = cad.boundary([surface])[0]
        calls = _occ_operation_call_count(backend, operation)

        with pytest.raises(ValueError, match="three-dimensional|3D"):
            if operation == "fillet":
                cad.fillet([surface], [curve], [0.1])
            else:
                cad.chamfer([surface], [curve], [surface], [0.1])

        assert _occ_operation_call_count(backend, operation) == calls


@pytest.mark.parametrize(
    ("operation", "curve_name", "surface_name"),
    [
        ("fillet", "unrelated_curve", None),
        ("chamfer", "curve", "nonadjacent_surface"),
        ("chamfer", "curve", "outside_surface"),
    ],
)
def test_edge_treatments_validate_curve_and_surface_volume_adjacency_pre_native(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    curve_name: str,
    surface_name: str | None,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{operation}-adjacency", dimension=3) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)
        calls = _occ_operation_call_count(backend, operation)

        with pytest.raises(ValueError, match="adjacen|belong|boundary"):
            if operation == "fillet":
                cad.fillet(
                    [topology["volume"]],
                    [topology[curve_name]],
                    [0.1],
                )
            else:
                assert surface_name is not None
                cad.chamfer(
                    [topology["volume"]],
                    [topology[curve_name]],
                    [topology[surface_name]],
                    [0.1],
                )

        assert _occ_operation_call_count(backend, operation) == calls


@pytest.mark.parametrize("operation", ["fillet", "chamfer"])
def test_edge_treatments_reject_foreign_and_raw_stale_references_pre_native(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{operation}-ownership", dimension=3) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)
        foreign_volume = geometry.EntityRef(
            3,
            topology["volume"].tag,
            object(),
            object(),
        )
        calls = _occ_operation_call_count(backend, operation)

        with pytest.raises(geometry.EntityOwnershipError):
            if operation == "fillet":
                cad.fillet([foreign_volume], [topology["curve"]], [0.1])
            else:
                cad.chamfer(
                    [foreign_volume],
                    [topology["curve"]],
                    [topology["surface"]],
                    [0.1],
                )
        assert _occ_operation_call_count(backend, operation) == calls

        assert cad.raw_occ is backend.model.occ
        with pytest.raises(geometry.StaleEntityError):
            _apply_edge_treatment(cad, operation, topology, [0.1])
        assert _occ_operation_call_count(backend, operation) == calls


@pytest.mark.parametrize("operation", ["fillet", "chamfer"])
def test_destructive_edge_treatment_reuses_tag_with_fresh_identity_and_stales_closure(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{operation}-destructive", dimension=3) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)
        old_closure = tuple(
            topology[name]
            for name in (
                "volume",
                "surface",
                "nonadjacent_surface",
                "curve",
                "other_curve",
                "start",
                "end",
                "other_start",
                "other_end",
            )
        )
        unrelated = tuple(
            topology[name]
            for name in (
                "unrelated_volume",
                "unrelated_surface",
                "unrelated_curve",
                "unrelated_start",
            )
        )

        result = _apply_edge_treatment(cad, operation, topology, [0.1])

        replacement = result.primary[0]
        assert (replacement.dimension, replacement.tag) == (
            topology["volume"].dimension,
            topology["volume"].tag,
        )
        assert replacement != topology["volume"]
        for reference in old_closure:
            with pytest.raises(geometry.StaleEntityError):
                cad.boundary([reference], combined=False)
        for reference in (*unrelated, replacement):
            cad.boundary([reference], combined=False)


@pytest.mark.parametrize("operation", ["fillet", "chamfer"])
def test_preserving_edge_treatment_keeps_original_closure_and_returns_fresh_body(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{operation}-preserve", dimension=3) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)
        original_closure = tuple(
            topology[name]
            for name in (
                "volume",
                "surface",
                "nonadjacent_surface",
                "curve",
                "other_curve",
                "start",
                "end",
                "other_start",
                "other_end",
            )
        )

        result = _apply_edge_treatment(
            cad,
            operation,
            topology,
            [0.1],
            remove_volumes=False,
        )

        added = result.primary[0]
        assert added != topology["volume"]
        assert (added.dimension, added.tag) != (
            topology["volume"].dimension,
            topology["volume"].tag,
        )
        for reference in original_closure:
            cad.boundary([reference], combined=False)
        added_boundary = cad.boundary([added], combined=False)
        assert added_boundary
        assert set(added_boundary).isdisjoint(original_closure)


@pytest.mark.parametrize("operation", ["fillet", "chamfer"])
def test_edge_treatment_exposes_lower_dimensional_native_outputs(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{operation}-lower-outputs", dimension=3) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)
        backend.model.occ.edge_treatment_results[operation] = [
            (3, 900),
            (2, 901),
        ]

        result = _apply_edge_treatment(
            cad,
            operation,
            topology,
            [0.1],
            remove_volumes=False,
        )

        assert tuple(entity.dimension for entity in result.outputs) == (3, 2)
        assert result.primary == (result.outputs[0],)
        assert result.of_dimension(2) == (result.outputs[1],)
        assert result.ends == ()
        assert result.sides == ()


@pytest.mark.parametrize("operation", ["fillet", "chamfer"])
@pytest.mark.parametrize(
    "malformation",
    [
        "empty",
        "wrong_dimension",
        "missing",
        "reused_preserved",
        "unrelated_lower",
    ],
)
def test_malformed_edge_treatment_result_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    malformation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(
        f"{operation}-malformed-{malformation}",
        dimension=3,
    ) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)
        if malformation == "empty":
            outputs: list[tuple[int, int]] = []
        elif malformation == "wrong_dimension":
            outputs = [(2, 900)]
        elif malformation == "missing":
            outputs = [(3, 900)]
            backend.model.occ.edge_treatment_register_outputs = False
        elif malformation == "unrelated_lower":
            outputs = [(3, 900), (2, 901)]
            backend.model.occ.edge_treatment_attach_lower_outputs = False
        else:
            outputs = [(3, topology["volume"].tag)]
        backend.model.occ.edge_treatment_results[operation] = outputs
        preserve = malformation in {"reused_preserved", "unrelated_lower"}

        with pytest.raises(geometry.GeometryError):
            _apply_edge_treatment(
                cad,
                operation,
                topology,
                [0.1],
                remove_volumes=not preserve,
            )

        for name in ("volume", "curve", "unrelated_volume", "unrelated_curve"):
            with pytest.raises(geometry.StaleEntityError):
                cad.boundary([topology[name]], combined=False)
        reacquired = cad.entity(3, topology["unrelated_volume"].tag)
        cad.boundary([reacquired], combined=False)


@pytest.mark.parametrize("operation", ["fillet", "chamfer"])
def test_malformed_edge_treatment_pair_has_operation_context(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{operation}-invalid-pair", dimension=3) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)
        backend.model.occ.edge_treatment_results[operation] = [
            (4, 900),
        ]

        with pytest.raises(
            geometry.GeometryError,
            match=rf"geometry model .*{operation} returned invalid entity data",
        ):
            _apply_edge_treatment(cad, operation, topology, [0.1])

        with pytest.raises(geometry.StaleEntityError):
            cad.boundary((topology["unrelated_volume"],), combined=False)


@pytest.mark.parametrize("operation", ["fillet", "chamfer"])
def test_preserving_edge_treatment_detects_removed_original_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{operation}-preserve-violation", dimension=3) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)
        backend.model.occ.edge_treatment_remove_preserved.add(operation)

        with pytest.raises(geometry.GeometryError, match="preserv|removed"):
            _apply_edge_treatment(
                cad,
                operation,
                topology,
                [0.1],
                remove_volumes=False,
            )

        for name in ("volume", "curve", "unrelated_volume"):
            with pytest.raises(geometry.StaleEntityError):
                cad.boundary([topology[name]], combined=False)


@pytest.mark.parametrize("operation", ["fillet", "chamfer"])
def test_destructive_edge_treatment_rejects_unreported_surviving_input(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{operation}-surviving-input", dimension=3) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)
        backend.model.occ.edge_treatment_results[operation] = [(3, 900)]
        backend.model.occ.edge_treatment_preserve_destructive.add(operation)

        with pytest.raises(geometry.GeometryError, match="left an input volume"):
            _apply_edge_treatment(cad, operation, topology, [0.1])

        for name in ("volume", "curve", "unrelated_volume", "unrelated_curve"):
            with pytest.raises(geometry.StaleEntityError):
                cad.boundary([topology[name]], combined=False)


@pytest.mark.parametrize("operation", ["fillet", "chamfer"])
def test_edge_treatment_supports_multiple_selected_volumes(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{operation}-multiple-volumes", dimension=3) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)
        volumes = (topology["volume"], topology["unrelated_volume"])
        curves = (topology["curve"], topology["unrelated_curve"])
        if operation == "fillet":
            result = cad.fillet(volumes, curves, (0.1, 0.12))
        else:
            result = cad.chamfer(
                volumes,
                curves,
                (topology["surface"], topology["unrelated_surface"]),
                (0.1, 0.12),
            )

        assert len(result.primary) == 2
        assert all(entity.dimension == 3 for entity in result.primary)
        for source in (*volumes, *curves):
            with pytest.raises(geometry.StaleEntityError):
                cad.boundary((source,), combined=False)


@pytest.mark.parametrize("operation", ["fillet", "chamfer"])
def test_native_edge_treatment_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{operation}-native-failure", dimension=3) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)
        backend.model.occ.fail_next.add(operation)

        with pytest.raises(
            geometry.GeometryError,
            match=rf"native OCC {operation} failed",
        ) as caught:
            _apply_edge_treatment(cad, operation, topology, [0.1])
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert str(caught.value.__cause__) == f"fake {operation} failure"

        for name in ("volume", "surface", "curve", "unrelated_volume"):
            with pytest.raises(geometry.StaleEntityError):
                cad.boundary([topology[name]], combined=False)


@pytest.mark.parametrize("operation", ["fillet", "chamfer"])
def test_mesher_binding_seals_edge_treatments_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{operation}-mesher-sealed", dimension=3) as cad:
        topology = _fake_edge_treatment_topology(cad, backend)
        _mesher(cad)
        calls = _occ_operation_call_count(backend, operation)

        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            _apply_edge_treatment(cad, operation, topology, [0.1])

        assert _occ_operation_call_count(backend, operation) == calls


def test_translate_and_rotate_forward_and_return_same_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("transform", dimension=2) as cad:
        first = cad.rectangle(0, 0, 1, 1)
        second = cad.disk(2, 0, 0.5)
        assert cad.translate([first, second], 1, 2, 0) == (first, second)
        assert cad.rotate([first], 0, 0, 0, 0, 0, 2, 0.5) == (first,)

    assert ("translate", ((2, 1), (2, 2)), 1.0, 2.0, 0.0) in (
        backend.model.occ.calls
    )
    assert (
        "rotate",
        ((2, 1),),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.5,
    ) in backend.model.occ.calls


@pytest.mark.parametrize("operation", ["mirror", "scale"])
def test_mirror_and_scale_forward_preserve_sources_and_invalidate_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"transform-{operation}", dimension=2) as cad:
        source = cad.rectangle(0, 0, 1, 1)
        unrelated = cad.rectangle(3, 0, 1, 1)
        old_boundaries = cad.boundary([source], combined=False)

        if operation == "mirror":
            result = cad.mirror([source], 1, 0, 0, -1)
            expected_call = ("mirror", ((2, source.tag),), 1.0, 0.0, 0.0, -1.0)
        else:
            result = cad.scale([source], 0, 0, 0, -2, 3, -1)
            expected_call = (
                "dilate",
                ((2, source.tag),),
                0.0,
                0.0,
                0.0,
                -2.0,
                3.0,
                -1.0,
            )

        assert result == (source,)
        assert expected_call in backend.model.occ.calls
        for old_boundary in old_boundaries:
            with pytest.raises(geometry.StaleEntityError):
                cad.translate([old_boundary], 1, 0, 0)
        assert cad.translate([source, unrelated], 1, 0, 0) == (
            source,
            unrelated,
        )


@pytest.mark.parametrize("operation", ["mirror", "scale"])
def test_valid_2d_mirror_and_scale_plane_preservation_cases_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"transform-valid-plane-{operation}", dimension=2) as cad:
        source = cad.rectangle(0, 0, 1, 1)

        if operation == "mirror":
            assert cad.mirror([source], 0, 0, 2, 0) == (source,)
            expected = ("mirror", ((2, source.tag),), 0.0, 0.0, 2.0, 0.0)
        else:
            assert cad.scale([source], 0, 0, 5, 2, 3, 1) == (source,)
            expected = (
                "dilate",
                ((2, source.tag),),
                0.0,
                0.0,
                5.0,
                2.0,
                3.0,
                1.0,
            )

        assert expected in backend.model.occ.calls


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, entity: cad.translate([], 1, 0, 0),
        lambda cad, entity: cad.translate([entity, entity], 1, 0, 0),
        lambda cad, entity: cad.translate([entity], 0, 0, 1),
        lambda cad, entity: cad.translate([entity], float("nan"), 0, 0),
        lambda cad, entity: cad.rotate([entity], 0, 0, 0, 0, 0, 0, 1),
        lambda cad, entity: cad.rotate([entity], 0, 0, 0, 1, 0, 1, 1),
        lambda cad, entity: cad.rotate(
            [entity], 0, 0, 0, 0, 0, 1, float("nan")
        ),
        lambda cad, entity: cad.mirror([], 1, 0, 0, 0),
        lambda cad, entity: cad.mirror([entity, entity], 1, 0, 0, 0),
        lambda cad, entity: cad.mirror([entity], float("nan"), 0, 0, 0),
        lambda cad, entity: cad.mirror([entity], 0, 0, 0, 1),
        lambda cad, entity: cad.mirror([entity], 1, 0, 1, 0),
        lambda cad, entity: cad.mirror([entity], 0, 0, 1, 1),
        lambda cad, entity: cad.scale([], 0, 0, 0, 1, 1, 1),
        lambda cad, entity: cad.scale([entity, entity], 0, 0, 0, 1, 1, 1),
        lambda cad, entity: cad.scale(
            [entity], float("nan"), 0, 0, 1, 1, 1
        ),
        lambda cad, entity: cad.scale([entity], 0, 0, 0, 1, 0, 1),
        lambda cad, entity: cad.scale([entity], 0, 0, 1, 1, 1, 2),
    ],
)
def test_invalid_transform_inputs_fail_before_occ_call(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("transform", dimension=2) as cad:
        entity = cad.rectangle(0, 0, 1, 1)
        before = list(backend.model.occ.calls)
        with pytest.raises((ValueError, TypeError)):
            operation(cad, entity)
        assert backend.model.occ.calls == before


@pytest.mark.parametrize(
    ("operation", "native_operation"),
    [("mirror", "mirror"), ("scale", "dilate")],
)
@pytest.mark.parametrize("failure_mode", ["native", "postcheck"])
def test_mirror_and_scale_native_or_postcheck_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    native_operation: str,
    failure_mode: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(
        f"transform-failure-{operation}-{failure_mode}",
        dimension=2,
    ) as cad:
        source = cad.rectangle(0, 0, 1, 1)
        unrelated = cad.rectangle(3, 0, 1, 1)
        if failure_mode == "native":
            backend.model.occ.fail_next.add(native_operation)
            error_type: type[Exception] = RuntimeError
            message = "fake"
        else:
            backend.model.occ.nonplanar_after.add(native_operation)
            error_type = ValueError
            message = "global XY plane"

        with pytest.raises(error_type, match=message):
            _apply_typed_transform(cad, operation, source)

        for old_reference in (source, unrelated):
            with pytest.raises(geometry.StaleEntityError):
                cad.translate([old_reference], 1, 0, 0)
        reacquired = cad.entity(2, unrelated.tag)
        assert cad.entity(2, unrelated.tag) == reacquired
        with pytest.raises(geometry.GeometryStateError, match="dependencies unknown"):
            cad.translate([reacquired], 1, 0, 0)


@pytest.mark.parametrize(
    "axis",
    [
        (1.0e-10, 0.0, 2.0e-10),
        (5.0e-11, 0.0, 1.0),
    ],
)
def test_2d_rotation_rejects_every_tilted_axis(
    monkeypatch: pytest.MonkeyPatch,
    axis: tuple[float, float, float],
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("transform", dimension=2) as cad:
        entity = cad.rectangle(0, 0, 1, 1)
        before = list(backend.model.occ.calls)
        with pytest.raises(ValueError, match="parallel to the global Z axis"):
            cad.rotate([entity], 0, 0, 0, *axis, 1)
        assert backend.model.occ.calls == before


@pytest.mark.parametrize("operation", ["translate", "extrude"])
def test_2d_transform_rejects_nonzero_dz_that_could_accumulate(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("transform", dimension=2) as cad:
        if operation == "translate":
            entity = cad.rectangle(0, 0, 1, 1)
        else:
            backend.model._current_data()["entities"].add((1, 1))
            entity = cad.entity(1, 1)
        before = list(backend.model.occ.calls)
        with pytest.raises(ValueError, match="global XY"):
            getattr(cad, operation)([entity], 1, 0, 5.0e-11)
        assert backend.model.occ.calls == before


def test_extrude_validates_and_forwards_layer_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("extrude", dimension=3) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        result = _structured_extrude(
            cad,
            [surface],
            0,
            0,
            2,
            num_elements=(2, 3),
            heights=(0.4, 1.0),
            recombine=True,
        )

    assert result.operation == "structured_extrude"
    assert result.inputs == (surface,)
    assert tuple((item.dimension, item.tag) for item in result.outputs) == (
        (2, 2),
        (3, 1),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
    )
    assert tuple((item.dimension, item.tag) for item in result.primary) == ((3, 1),)
    assert tuple((item.dimension, item.tag) for item in result.ends) == ((2, 2),)
    assert tuple((item.dimension, item.tag) for item in result.sides) == (
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
    )
    assert (
        "extrude",
        ((2, 1),),
        0.0,
        0.0,
        2.0,
        (2, 3),
        (0.4, 1.0),
        True,
    ) in backend.model.occ.calls


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vector": (0, 0, 0)},
        {"vector": (0, 0, 1), "num_elements": (0,)},
        {"vector": (0, 0, 1), "num_elements": (True,)},
        {"vector": (0, 0, 1), "heights": (1.0,)},
        {
            "vector": (0, 0, 1),
            "num_elements": (1, 1),
            "heights": (1.0,),
        },
        {
            "vector": (0, 0, 1),
            "num_elements": (1, 1),
            "heights": (0.6, 0.5),
        },
        {
            "vector": (0, 0, 1),
            "num_elements": (1,),
            "heights": (0.9,),
        },
        {"vector": (0, 0, 1), "recombine": 1},
    ],
)
def test_invalid_extrusion_controls_fail_before_occ_call(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("extrude", dimension=3) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        options = dict(kwargs)
        vector = options.pop("vector")
        before = list(backend.model.occ.calls)
        with pytest.raises((ValueError, TypeError)):
            if options:
                _structured_extrude(cad, [surface], *vector, **options)
            else:
                cad.extrude([surface], *vector)
        assert backend.model.occ.calls == before


def test_2d_extrusion_rejects_out_of_plane_and_too_high_input_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("extrude", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        with pytest.raises(ValueError, match="dimension"):
            cad.extrude([surface], 1, 0, 0)
        backend.model._current_data()["entities"].add((1, 1))
        curve = cad.entity(1, 1)
        with pytest.raises(ValueError, match="global XY"):
            cad.extrude([curve], 0, 0, 1)


def test_entities_and_boundary_synchronize_and_sort_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("queries", dimension=2) as cad:
        backend.model._current_data()["entities"].update(
            {(2, 4), (2, 1), (1, 3), (1, 1)}
        )
        surfaces = cad.entities(2)
        backend.model.boundary_result = [(1, 3), (1, 1), (1, 3)]
        boundaries = cad.boundary(
            surfaces,
            combined=False,
            recursive=True,
        )

    assert tuple(item.tag for item in surfaces) == (1, 4)
    assert tuple(item.tag for item in boundaries) == (1, 3)
    assert backend.model.occ.synchronize_calls == 2
    assert (
        "getBoundary",
        ((2, 1), (2, 4)),
        False,
        False,
        True,
        "queries",
    ) in backend.model.calls


def test_coordinate_selection_checks_both_bounding_box_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("selection", dimension=2) as cad:
        backend.model._current_data()["entities"].update({(1, 1), (1, 2), (1, 3)})
        backend.model._current_data()["boxes"].update(
            {
                (1, 1): (-1e-9, 0.0, 0.0, 1e-9, 1.0, 0.0),
                (1, 2): (0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
                (1, 3): (-1e-9, 2.0, 0.0, 1e-9, 2.0, 0.0),
            }
        )
        curves = cad.entities(1)
        at_x_zero = cad.select(curves, x=0.0)
        at_point = cad.select(curves, x=0.0, y=2.0)
        empty = cad.select(curves, x=5.0)

    assert tuple(item.tag for item in at_x_zero) == (1, 3)
    assert tuple(item.tag for item in at_point) == (3,)
    assert empty == ()


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, entity: cad.boundary([]),
        lambda cad, entity: cad.select([entity]),
        lambda cad, entity: cad.select([], x=0),
        lambda cad, entity: cad.select([entity], x=float("nan")),
        lambda cad, entity: cad.select([entity], x=0, tolerance=-1),
    ],
)
def test_invalid_query_inputs_fail_before_model_level_call(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("queries", dimension=2) as cad:
        entity = cad.rectangle(0, 0, 1, 1)
        model_calls = list(backend.model.calls)
        synchronize_calls = backend.model.occ.synchronize_calls
        with pytest.raises((ValueError, TypeError)):
            operation(cad, entity)
        assert backend.model.calls == model_calls
        assert backend.model.occ.synchronize_calls == synchronize_calls


























def test_meshing_port_activation_failure_does_not_consume_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("binding-activation-retry", dimension=2) as cad:
        surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        original_list = backend.model.list

        def fail_list() -> list[str]:
            raise RuntimeError("injected model-list failure")

        monkeypatch.setattr(backend.model, "list", fail_list)
        with pytest.raises(RuntimeError, match="model-list failure"):
            gmsh_meshing.Mesher(cad)

        monkeypatch.setattr(backend.model, "list", original_list)
        builder = gmsh_meshing.Mesher(cad)
        assert builder.recombine(surface) is None












@pytest.mark.parametrize("operation", ["fuse", "cut", "intersect", "fragment"])
@pytest.mark.parametrize("control", _ENTITY_DEPENDENT_MESH_CONTROLS)
def test_entity_dependency_guard_rejects_boolean_removing_control_closure(
    monkeypatch: pytest.MonkeyPatch,
    control: str,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(
        f"dependency-closure-{control}-{operation}",
        dimension=3,
    ) as cad:
        point, curve, surface, volume = _fake_mesh_control_targets(cad, backend)
        dependency = _fake_control_boundary_dependency(
            cad,
            backend,
            control,
            point=point,
            curve=curve,
            surface=surface,
            volume=volume,
        )
        backend.model.boundary_result = (
            []
            if control == "mesh_size"
            else [(dependency.dimension, dependency.tag)]
        )

        _apply_entity_dependent_mesh_control(
            cad,
            control,
            point=point,
            curve=curve,
            surface=surface,
            volume=volume,
        )
        removed, kept = _fake_entities(
            cad,
            backend,
            dependency.dimension + 1,
            80,
            81,
        )
        backend.model.boundary_result = [(dependency.dimension, dependency.tag)]
        boolean_calls = _occ_operation_call_count(backend, operation)

        with pytest.raises(
            geometry.GeometryStateError,
            match="CONFIGURING_MESH",
        ):
            getattr(cad, operation)(
                [removed],
                [kept],
                remove_objects=True,
                remove_tools=False,
            )

        assert _occ_operation_call_count(backend, operation) == boolean_calls


@pytest.mark.parametrize("control", _ENTITY_DEPENDENT_MESH_CONTROLS)
def test_native_control_failure_keeps_geometry_sealed_without_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
    control: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"failed-dependency-{control}", dimension=3) as cad:
        point, curve, surface, volume = _fake_mesh_control_targets(cad, backend)
        target = _entity_control_target(
            control,
            point=point,
            curve=curve,
            surface=surface,
            volume=volume,
        )
        unrelated = _fake_entities(cad, backend, target.dimension, 99)[0]
        backend.model.boundary_result = []
        if control.startswith("transfinite_"):
            native_operation = {
                "transfinite_curve": "setTransfiniteCurve",
                "transfinite_surface": "setTransfiniteSurface",
                "transfinite_volume": "setTransfiniteVolume",
            }[control]
            backend.model.mesh.fail_next.add(native_operation)
        elif control == "recombine":
            backend.model.mesh.fail_next.add("setRecombine")
        elif control == "mesh_size":
            backend.model.mesh.fail_set_size = True
        elif control == "distance_field":
            backend.model.mesh.field.fail_next.add(("setNumber", "Sampling"))
        else:
            backend.model.occ.fail_next.add("extrude")

        with pytest.raises(RuntimeError, match="fake"):
            _apply_entity_dependent_mesh_control(
                cad,
                control,
                point=point,
                curve=curve,
                surface=surface,
                volume=volume,
            )

        expected_state = (
            "MESH_FAILED"
            if control in {"layered_extrude", "recombined_extrude"}
            else "CONFIGURING_MESH"
        )
        translate_calls = _occ_operation_call_count(backend, "translate")
        with pytest.raises(geometry.GeometryStateError, match=expected_state):
            _apply_typed_transform(cad, "translate", target)
        assert _occ_operation_call_count(backend, "translate") == translate_calls
        fuse_calls = _occ_operation_call_count(backend, "fuse")
        with pytest.raises(geometry.GeometryStateError, match=expected_state):
            cad.fuse([target], [unrelated])
        assert _occ_operation_call_count(backend, "fuse") == fuse_calls


@pytest.mark.parametrize(
    "operation",
    [
        "point",
        "line",
        "rectangle",
        "disk",
        "box",
        "cylinder",
        "copy",
        "plain_extrude",
    ],
)
def test_mesher_binding_seals_additive_geometry_topology(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    dimension = 1 if operation in {"point", "line"} else 3

    with geometry.model(
        f"dependency-allows-{operation}",
        dimension=dimension,
    ) as cad:
        if dimension == 1:
            start = cad.point(0, 0, 0)
            end = cad.point(1, 0, 0)
            backend.model.boundary_result = []
            _mesher(cad).mesh_size([start], size=0.1)
            mutation = {
                "point": lambda: cad.point(2, 0, 0),
                "line": lambda: cad.line(start, end),
            }[operation]
        else:
            surface = cad.rectangle(0, 0, 1, 1)
            point = _fake_entities(cad, backend, 0, 20)[0]
            backend.model.boundary_result = []
            _mesher(cad).mesh_size([point], size=0.1)
            mutation = {
                "rectangle": lambda: cad.rectangle(4, 0, 1, 1),
                "disk": lambda: cad.disk(4, 0, 1),
                "box": lambda: cad.box(4, 0, 0, 1, 1, 1),
                "cylinder": lambda: cad.cylinder(4, 0, 0, 0, 0, 1, 1),
                "copy": lambda: cad.copy([surface]),
                "plain_extrude": lambda: cad.extrude([surface], 0, 0, 1),
            }[operation]

        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            mutation()


def test_entity_dependency_guard_allows_multiple_controlled_extrusions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("multiple-controlled-extrusions", dimension=3) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model.boundary_result = []
        first_result = _structured_extrude(
            cad,
            [surface],
            0,
            0,
            1,
            num_elements=[2],
            heights=[1.0],
        )
        top = first_result.ends[0]

        second_result = _structured_extrude(
            cad,
            [top],
            0,
            0,
            1,
            recombine=True,
        )

        assert tuple(entity.dimension for entity in second_result.primary) == (3,)
        assert tuple(entity.dimension for entity in second_result.ends) == (2,)
        assert second_result.sides == ()
        assert sum(call[0] == "extrude" for call in backend.model.occ.calls) == 2


def test_controlled_extrude_preserves_valid_duplicate_native_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("controlled-extrude-shared-side", dimension=3) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model.occ.configure_extrude_result(
            [(2, surface.tag)],
            (0, 0, 1),
            [
                (2, 2),
                (3, 1),
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
                (2, 3),
            ],
            ends=[(2, 2)],
            primary=[(3, 1)],
        )

        result = _structured_extrude(
            cad,
            [surface],
            0,
            0,
            1,
            num_elements=[1],
        )

        assert tuple((item.dimension, item.tag) for item in result.outputs) == (
            (2, 2),
            (3, 1),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 3),
        )
        assert result.primary == result.of_dimension(3)
        assert tuple(item.tag for item in result.ends) == (2,)
        assert tuple(item.tag for item in result.sides) == (3, 4, 5, 6)
        assert result.outputs[2] == result.outputs[-1]


def test_extrude_rejects_omitted_generated_primary_boundary_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("extrude-omitted-boundary", dimension=3) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model.occ.configure_extrude_result(
            [(2, surface.tag)],
            (0, 0, 1),
            [(2, 2), (3, 1), (2, 3), (2, 4), (2, 5), (2, 6)],
            ends=[(2, 2)],
            primary=[(3, 1)],
        )
        backend.model.occ.extrude_extra_primary_boundaries[(3, 1)] = [(2, 99)]

        with pytest.raises(
            geometry.GeometryError,
            match="same-dimensional output topology completely",
        ):
            cad.extrude([surface], 0, 0, 1)


def test_extrude_rejects_duplicate_side_assignment_with_omitted_source_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("extrude-duplicate-side-contact", dimension=3) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model.occ.configure_extrude_result(
            [(2, surface.tag)],
            (0, 0, 1),
            [(2, 2), (3, 1), (2, 3), (2, 4), (2, 5), (2, 6)],
            ends=[(2, 2)],
            primary=[(3, 1)],
        )
        backend.model.occ.extrude_side_contact_indices[(3, 1)] = (0, 0, 1, 2)

        with pytest.raises(
            geometry.GeometryError,
            match="side topology classification is incomplete or ambiguous",
        ):
            cad.extrude([surface], 0, 0, 1)


@pytest.mark.parametrize("operation", ["fuse", "cut", "intersect", "fragment"])
def test_mesher_binding_seals_non_destructive_booleans(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"non-destructive-{operation}", dimension=2) as cad:
        first = cad.rectangle(0, 0, 1, 1)
        second = cad.rectangle(2, 0, 1, 1)
        backend.model.boundary_result = []
        _mesher(cad).recombine(first)
        boolean_calls = _occ_operation_call_count(backend, operation)

        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            getattr(cad, operation)(
                [first],
                [second],
                remove_objects=False,
                remove_tools=False,
            )

        assert _occ_operation_call_count(backend, operation) == boolean_calls


@pytest.mark.parametrize("operation", ["fuse", "cut", "intersect", "fragment"])
@pytest.mark.parametrize(
    "removed_scope",
    ["objects", "tools", "objects_and_tools"],
)
def test_mesher_binding_seals_unrelated_destructive_booleans(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    removed_scope: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"unrelated-removal-{operation}", dimension=2) as cad:
        protected = cad.rectangle(0, 0, 1, 1)
        unrelated_a = cad.rectangle(2, 0, 1, 1)
        unrelated_b = cad.rectangle(4, 0, 1, 1)
        backend.model.boundary_result = []
        _mesher(cad).recombine(protected)
        if removed_scope == "objects":
            objects = [unrelated_a]
            tools = [protected]
            remove_objects, remove_tools = True, False
        elif removed_scope == "tools":
            objects = [protected]
            tools = [unrelated_a]
            remove_objects, remove_tools = False, True
        else:
            objects = [unrelated_a]
            tools = [unrelated_b]
            remove_objects, remove_tools = True, True
        backend.model.boundary_result = []
        boolean_calls = _occ_operation_call_count(backend, operation)

        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            getattr(cad, operation)(
                objects,
                tools,
                remove_objects=remove_objects,
                remove_tools=remove_tools,
            )

        assert _occ_operation_call_count(backend, operation) == boolean_calls


@pytest.mark.parametrize("operation", ["translate", "rotate", "mirror", "scale"])
@pytest.mark.parametrize("control", _TRANSFORM_UNSAFE_ENTITY_CONTROLS)
def test_entity_dependency_guard_rejects_transform_of_control_closure_only(
    monkeypatch: pytest.MonkeyPatch,
    control: str,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"controlled-transform-{control}-{operation}", dimension=3) as cad:
        point, curve, surface, volume = _fake_mesh_control_targets(cad, backend)
        dependency = _fake_control_boundary_dependency(
            cad,
            backend,
            control,
            point=point,
            curve=curve,
            surface=surface,
            volume=volume,
        )
        backend.model.boundary_result = (
            []
            if control == "mesh_size"
            else [(dependency.dimension, dependency.tag)]
        )
        _apply_entity_dependent_mesh_control(
            cad,
            control,
            point=point,
            curve=curve,
            surface=surface,
            volume=volume,
        )
        controlled_parent, unrelated = _fake_entities(
            cad,
            backend,
            dependency.dimension + 1,
            80,
            81,
        )
        backend.model.boundary_result = [(dependency.dimension, dependency.tag)]
        transform_calls = _occ_operation_call_count(backend, operation)

        with pytest.raises(
            geometry.GeometryStateError,
            match="CONFIGURING_MESH",
        ):
            _apply_typed_transform(cad, operation, controlled_parent)

        assert _occ_operation_call_count(backend, operation) == transform_calls
        backend.model.boundary_result = []
        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            _apply_typed_transform(cad, operation, unrelated)
        assert _occ_operation_call_count(backend, operation) == transform_calls


@pytest.mark.parametrize("operation", ["translate", "rotate", "mirror", "scale"])
def test_distance_source_transform_is_sealed_after_mesher_binding(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"distance-{operation}", dimension=2) as cad:
        source = cad.rectangle(0, 0, 1, 1)
        backend.model.boundary_result = []
        _mesher(cad).distance_field(surfaces=[source])
        backend.model.boundary_result = []
        transform_calls = _occ_operation_call_count(backend, operation)

        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            _apply_typed_transform(cad, operation, source)
        assert _occ_operation_call_count(backend, operation) == transform_calls


@pytest.mark.parametrize(
    "options",
    [
        {"remove_objects": 1, "remove_tools": True},
        {"remove_objects": False, "remove_tools": 1},
    ],
)
def test_mesher_seal_precedes_boolean_remove_flag_validation(
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any],
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("guarded-invalid-remove-flags", dimension=2) as cad:
        first = cad.rectangle(0, 0, 1, 1)
        second = cad.rectangle(2, 0, 1, 1)
        backend.model.boundary_result = []
        _mesher(cad).recombine(first)
        fuse_calls = _occ_operation_call_count(backend, "fuse")

        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            cad.fuse([first], [second], **options)

        assert _occ_operation_call_count(backend, "fuse") == fuse_calls


def test_entity_dependency_guard_allows_more_controls_and_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("guarded-mesh-workflow", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        backend.model.boundary_result = []
        _mesher(cad).mesh_size([point], size=0.1)

        backend.model.boundary_result = []
        assert _mesher(cad).recombine(surface) is None
        assert isinstance(_generate_mesh(cad, ), gmsh_meshing.GmshMeshRef)


@pytest.mark.parametrize("raw_access", ["raw_model", "raw_occ"])
def test_raw_access_is_rejected_after_mesher_binding_without_invalidating_refs(
    monkeypatch: pytest.MonkeyPatch,
    raw_access: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"unknown-after-{raw_access}", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        tool = cad.rectangle(2, 0, 1, 1)
        backend.model.boundary_result = []
        _mesher(cad).recombine(surface)
        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            getattr(cad, raw_access)
        assert cad.entity(2, surface.tag) == surface
        assert cad.entity(2, tool.tag) == tool


@pytest.mark.parametrize(
    ("native_result", "error_type", "message"),
    [
        ([(4, 1)], ValueError, "dimension"),
        ([], geometry.GeometryError, "no entities"),
        ([(2, 2)], geometry.GeometryError, "dimension-3"),
    ],
)
def test_malformed_structured_extrude_enters_terminal_mesh_failed_state(
    monkeypatch: pytest.MonkeyPatch,
    native_result: list[tuple[int, int]],
    error_type: type[Exception],
    message: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(
        f"malformed-controlled-extrude-{message}",
        dimension=3,
    ) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        tool = cad.rectangle(2, 0, 1, 1)
        backend.model.boundary_result = []
        backend.model.occ.extrude_result = native_result

        with pytest.raises(error_type, match=message):
            _structured_extrude(
                cad,
                [surface],
                0,
                0,
                1,
                num_elements=[1],
            )
        backend.model.boundary_result = []
        fragment_calls = _occ_operation_call_count(backend, "fragment")
        with pytest.raises(
            geometry.GeometryStateError,
            match="MESH_FAILED",
        ):
            cad.fragment([surface], [tool])

        assert _occ_operation_call_count(backend, "fragment") == fragment_calls
        rotate_calls = _occ_operation_call_count(backend, "rotate")
        with pytest.raises(
            geometry.GeometryStateError,
            match="MESH_FAILED",
        ):
            cad.rotate([surface], 0, 0, 0, 0, 0, 1, 0.5)

        assert _occ_operation_call_count(backend, "rotate") == rotate_calls












































































































@pytest.mark.parametrize(
    ("control", "reported_blocker"),
    [
        ("transfinite_curve", "transfinite_curve"),
        ("transfinite_surface", "transfinite_surface"),
        ("transfinite_volume", "transfinite_volume"),
        ("recombine", "recombine"),
        ("num_elements_extrude", "structured_extrude"),
        ("heights_extrude", "structured_extrude"),
        ("recombined_extrude", "structured_extrude"),
    ],
)
def test_auto_mesh_rejects_every_explicit_topology_control_retryably(
    monkeypatch: pytest.MonkeyPatch,
    control: str,
    reported_blocker: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"auto-blocker-{control}", dimension=3) as cad:
        point, curve, surface, volume = _fake_mesh_control_targets(cad, backend)
        backend.model.boundary_result = []
        if control in {
            "transfinite_curve",
            "transfinite_surface",
            "transfinite_volume",
            "recombine",
        }:
            _apply_entity_dependent_mesh_control(
                cad,
                control,
                point=point,
                curve=curve,
                surface=surface,
                volume=volume,
            )
        elif control == "num_elements_extrude":
            _structured_extrude(
                cad,
                [surface],
                0.0,
                0.0,
                1.0,
                num_elements=[2],
            )
        elif control == "heights_extrude":
            _structured_extrude(
                cad,
                [surface],
                0.0,
                0.0,
                1.0,
                num_elements=[1, 1],
                heights=[0.5, 1.0],
            )
        else:
            _structured_extrude(
                cad,
                [surface],
                0.0,
                0.0,
                1.0,
                recombine=True,
            )

        mesh_calls = list(backend.model.mesh.calls)
        option_calls = list(backend.option.calls)
        synchronize_calls = backend.model.occ.synchronize_calls
        with pytest.raises(gmsh_meshing.MeshControlConflictError) as captured:
            _generate_auto_mesh(cad, cell_shape="tet")

        assert reported_blocker in str(captured.value)
        assert "MeshSpec" in str(captured.value)
        assert backend.model.mesh.calls == mesh_calls
        assert backend.option.calls == option_calls
        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert isinstance(_generate_mesh(cad, ), gmsh_meshing.GmshMeshRef)


@pytest.mark.parametrize("raw_access", ["raw_model", "raw_occ"])
def test_auto_mesh_raw_access_conflict_is_retryable_through_low_level_path(
    monkeypatch: pytest.MonkeyPatch,
    raw_access: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"auto-raw-{raw_access}", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        getattr(cad, raw_access)
        mesh_calls = list(backend.model.mesh.calls)
        option_calls = list(backend.option.calls)
        synchronize_calls = backend.model.occ.synchronize_calls

        with pytest.raises(
            gmsh_meshing.MeshControlConflictError,
            match="scope unknown",
        ):
            _generate_auto_mesh(cad, )

        assert backend.model.mesh.calls == mesh_calls
        assert backend.option.calls == option_calls
        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert isinstance(_generate_mesh(cad, ), gmsh_meshing.GmshMeshRef)


@pytest.mark.parametrize("control", ["point", "background", "plain_extrude"])
def test_auto_mesh_accepts_compatible_typed_size_and_plain_topology_controls(
    monkeypatch: pytest.MonkeyPatch,
    control: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    dimension = 3 if control == "plain_extrude" else 2

    with geometry.model(f"auto-compatible-{control}", dimension=dimension) as cad:
        if control == "plain_extrude":
            surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
            cad.extrude([surface], 0.0, 0.0, 1.0)
            element_type = 4
        else:
            cad.rectangle(0.0, 0.0, 1.0, 1.0)
            point = _fake_entities(cad, backend, 0, 11)[0]
            backend.model.boundary_result = []
            if control == "point":
                _mesher(cad).mesh_size([point], size=0.1)
            else:
                distance = _mesher(cad).distance_field(points=[point])
                _mesher(cad).background_field(_fake_threshold(cad, distance))
            element_type = 2
        _set_fake_element_blocks(backend, dimension, (element_type, (1, 2)))

        assert isinstance(_generate_auto_mesh(cad, ), gmsh_meshing.GmshMeshRef)


@pytest.mark.parametrize("failure", ["transfinite", "extrude"])
def test_failed_control_state_distinguishes_precommit_and_native_occ_mutation(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"auto-precommit-{failure}", dimension=3) as cad:
        volume = cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        surface = cad.rectangle(2.0, 0.0, 1.0, 1.0)
        backend.model.boundary_result = []
        if failure == "transfinite":
            backend.model.mesh.fail_next.add("setTransfiniteVolume")
            with pytest.raises(RuntimeError, match="setTransfiniteVolume"):
                _mesher(cad).transfinite_volume(volume)
        else:
            backend.model.occ.fail_next.add("extrude")
            with pytest.raises(RuntimeError, match="fake extrude failure"):
                _structured_extrude(
                    cad,
                    [surface],
                    0.0,
                    0.0,
                    1.0,
                    num_elements=[1],
                )

        _set_fake_element_blocks(backend, 3, (4, (1, 2)))
        if failure == "transfinite":
            assert isinstance(_generate_auto_mesh(cad, ), gmsh_meshing.GmshMeshRef)
        else:
            with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
                _generate_auto_mesh(cad, )


def test_malformed_structured_extrusion_blocks_all_generation_terminally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("auto-malformed-controlled-extrude", dimension=3) as cad:
        surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        cad.box(2.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        backend.model.occ.extrude_result = []
        with pytest.raises(geometry.GeometryError, match="no entities"):
            _structured_extrude(
                cad,
                [surface],
                0.0,
                0.0,
                1.0,
                num_elements=[1],
            )

        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_auto_mesh(cad, )
        assert backend.model.mesh.generate_calls == []
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_mesh(cad, )


def _assert_positive_top_dimensional_jacobians(
    gmsh: Any,
    dimension: int,
) -> None:
    element_types, element_tags, _ = gmsh.model.mesh.getElements(dimension)
    checked_elements = 0
    for element_type, tags in zip(element_types, element_tags):
        if len(tags) == 0:
            continue
        local_coordinates, weights = gmsh.model.mesh.getIntegrationPoints(
            element_type,
            "Gauss2",
        )
        _, determinants, _ = gmsh.model.mesh.getJacobians(
            element_type,
            local_coordinates,
        )
        determinant_array = np.asarray(determinants, dtype=float)
        assert determinant_array.size == len(tags) * len(weights)
        assert np.all(np.isfinite(determinant_array))
        assert np.all(determinant_array > 0.0)
        checked_elements += len(tags)
    assert checked_elements > 0


def _assert_vtk_cell_type(mesh: Mesh2D | Mesh3D, expected: int) -> None:
    cells, cell_types, elements = post.vtk.cells.build(mesh)
    assert len(cells) == mesh.num_elements
    assert cell_types == [expected] * mesh.num_elements
    assert len(elements) == mesh.num_elements


def _top_dimensional_element_counts(
    gmsh: Any,
    dimension: int,
) -> dict[int, int]:
    element_types, element_tags, _ = gmsh.model.mesh.getElements(dimension)
    return {
        int(element_type): len(tags)
        for element_type, tags in zip(element_types, element_tags, strict=True)
        if len(tags) > 0
    }


def _tri3_areas_by_centroid_x(mesh: Mesh2D) -> list[tuple[float, float]]:
    nodes = {node.id: node for node in mesh.nodes}
    samples: list[tuple[float, float]] = []
    for element in mesh.elements:
        assert element.type == "Tri3"
        first, second, third = (
            nodes[node_id] for node_id in element.node_ids[:3]
        )
        centroid_x = (first.x + second.x + third.x) / 3.0
        area = 0.5 * abs(
            (second.x - first.x) * (third.y - first.y)
            - (third.x - first.x) * (second.y - first.y)
        )
        samples.append((centroid_x, area))
    return samples


def test_real_auto_line_levels_refine_monotonically(
    real_gmsh: Any,
) -> None:
    counts: list[int] = []
    for level in range(1, 6):
        with geometry.model(
            f"auto_line_{level}",
            dimension=1,
        ) as cad:
            start = cad.point(0.0, 0.0, 0.0)
            end = cad.point(8.0, 0.0, 0.0)
            cad.line(start, end)
            _generate_auto_mesh(cad, level=level)
            native_counts = _top_dimensional_element_counts(real_gmsh, 1)

        assert set(native_counts) == {1}
        counts.append(sum(native_counts.values()))

    assert all(coarse < fine for coarse, fine in zip(counts, counts[1:]))


@pytest.mark.parametrize(
    (
        "cell_shape",
        "order",
        "expected_native_types",
        "expected_fem_types",
        "expected_vtk_types",
    ),
    [
        ("tri", 1, {2}, {"Tri3"}, {5}),
        ("tri", 2, {9}, {"Tri6"}, {22}),
        ("tri-quad", 1, {2, 3}, {"Tri3", "Quad4"}, {5, 9}),
        ("tri-quad", 2, {9, 16}, {"Tri6", "Quad8"}, {22, 23}),
        ("quad", 1, {3}, {"Quad4"}, {9}),
        ("quad", 2, {16}, {"Quad8"}, {23}),
    ],
)
def test_real_auto_2d_policies_preserve_strict_native_and_fem_families(
    real_gmsh: Any,
    cell_shape: str,
    order: int,
    expected_native_types: set[int],
    expected_fem_types: set[str],
    expected_vtk_types: set[int],
) -> None:
    with geometry.model(
        f"auto_2d_{cell_shape}_{order}",
        dimension=2,
    ) as cad:
        if cell_shape == "quad":
            cad.disk(0.0, 0.0, 1.0)
        else:
            cad.rectangle(0.0, 0.0, 2.0, 1.0)
        native_mesh = _generate_auto_mesh(cad,
            level=2,
            cell_shape=cell_shape,
            order=order,
        )
        mesh = gmsh_io.read(native_mesh)
        native_counts = _top_dimensional_element_counts(real_gmsh, 2)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 2)

    actual_native_types = set(native_counts)
    actual_fem_types = {element.type for element in mesh.elements}
    assert actual_native_types
    assert actual_native_types <= expected_native_types
    assert actual_fem_types
    assert actual_fem_types <= expected_fem_types
    if cell_shape != "tri-quad":
        assert actual_native_types == expected_native_types
        assert actual_fem_types == expected_fem_types
    _, vtk_types, _ = post.vtk.cells.build(mesh)
    assert set(vtk_types) <= expected_vtk_types


@pytest.mark.parametrize(
    (
        "cell_shape",
        "order",
        "expected_native_type",
        "expected_fem_type",
        "expected_vtk_type",
    ),
    [
        ("tet", 1, 4, "Tet4", 10),
        ("tet", 2, 11, "Tet10", 24),
        ("hex", 1, 5, "Hex8", 12),
        ("hex", 2, 17, "Hex20", 25),
    ],
)
def test_real_auto_3d_policies_preserve_strict_native_and_fem_families(
    real_gmsh: Any,
    cell_shape: str,
    order: int,
    expected_native_type: int,
    expected_fem_type: str,
    expected_vtk_type: int,
) -> None:
    with geometry.model(
        f"auto_3d_{cell_shape}_{order}",
        dimension=3,
    ) as cad:
        cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        native_mesh = _generate_auto_mesh(cad,
            level=1 if cell_shape == "hex" else 2,
            cell_shape=cell_shape,
            order=order,
        )
        mesh = gmsh_io.read(native_mesh)
        native_counts = _top_dimensional_element_counts(real_gmsh, 3)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 3)

    assert set(native_counts) == {expected_native_type}
    assert not {6, 7, 13, 14, 18, 19}.intersection(native_counts)
    assert {element.type for element in mesh.elements} == {expected_fem_type}
    _assert_vtk_cell_type(mesh, expected_vtk_type)


@pytest.mark.parametrize("cell_shape", ["tri", "quad"])
def test_real_auto_2d_all_levels_refine_monotonically(
    real_gmsh: Any,
    cell_shape: str,
) -> None:
    counts: list[int] = []
    expected_type = "Tri3" if cell_shape == "tri" else "Quad4"
    for level in range(1, 6):
        with geometry.model(
            f"auto_2d_progression_{cell_shape}_{level}",
            dimension=2,
        ) as cad:
            cad.disk(0.0, 0.0, 1.0)
            native_mesh = _generate_auto_mesh(cad,
                level=level,
                cell_shape=cell_shape,
            )
            mesh = gmsh_io.read(native_mesh)

        assert {element.type for element in mesh.elements} == {expected_type}
        counts.append(mesh.num_elements)

    assert all(coarse < fine for coarse, fine in zip(counts, counts[1:]))


@pytest.mark.parametrize("cell_shape", ["tet", "hex"])
def test_real_auto_3d_selected_levels_refine_monotonically(
    real_gmsh: Any,
    cell_shape: str,
) -> None:
    counts: list[int] = []
    expected_type = "Tet4" if cell_shape == "tet" else "Hex8"
    for level in (1, 3, 5):
        with geometry.model(
            f"auto_3d_progression_{cell_shape}_{level}",
            dimension=3,
        ) as cad:
            cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            native_mesh = _generate_auto_mesh(cad,
                level=level,
                cell_shape=cell_shape,
            )
            mesh = gmsh_io.read(native_mesh)

        assert {element.type for element in mesh.elements} == {expected_type}
        counts.append(mesh.num_elements)

    assert all(coarse < fine for coarse, fine in zip(counts, counts[1:]))


@pytest.mark.parametrize("control", ["point", "background"])
def test_real_auto_typed_size_controls_preserve_near_far_refinement(
    real_gmsh: Any,
    control: str,
) -> None:
    counts: list[int] = []
    for level in (2, 4):
        with geometry.model(
            f"auto_local_refinement_{control}_{level}",
            dimension=2,
        ) as cad:
            surface = cad.rectangle(0.0, 0.0, 4.0, 1.0)
            boundary = cad.boundary([surface])
            left_curves = cad.select(boundary, x=0.0)
            assert len(left_curves) == 1
            if control == "point":
                boundary_points = cad.boundary(boundary, combined=False)
                left_points = cad.select(boundary_points, x=0.0)
                assert len(left_points) == 2
                _mesher(cad).mesh_size(left_points, size=0.04)
            else:
                distance = _mesher(cad).distance_field(curves=left_curves, sampling=100)
                threshold = _mesher(cad).threshold_field(
                    distance,
                    size_min=0.04,
                    size_max=0.35,
                    dist_min=0.15,
                    dist_max=1.5,
                )
                _mesher(cad).background_field(threshold)
            native_mesh = _generate_auto_mesh(cad,
                level=level,
                cell_shape="tri",
            )
            mesh = gmsh_io.read(native_mesh)
            _assert_positive_top_dimensional_jacobians(real_gmsh, 2)

        assert isinstance(mesh, Mesh2D)
        samples = _tri3_areas_by_centroid_x(mesh)
        near_areas = [area for x, area in samples if x < 0.75]
        far_areas = [area for x, area in samples if x > 3.25]
        assert near_areas
        assert far_areas
        assert np.median(near_areas) < np.median(far_areas)
        counts.append(mesh.num_elements)

    assert counts[0] < counts[1]


def test_real_auto_mesh_restores_external_algorithm_and_size_options(
    real_gmsh: Any,
) -> None:
    external_values = {
        "Mesh.RecombineAll": 1.0,
        "Mesh.MeshSizeFactor": 1.8,
        "Mesh.Algorithm": 5.0,
        "Mesh.Algorithm3D": 7.0,
        "Mesh.RecombinationAlgorithm": 2.0,
        "Mesh.Recombine3DAll": 1.0,
        "Mesh.SubdivisionAlgorithm": 1.0,
    }
    prior_values = {
        name: real_gmsh.option.getNumber(name) for name in external_values
    }
    try:
        for name, value in external_values.items():
            real_gmsh.option.setNumber(name, value)

        with geometry.model("auto_restore_quad", dimension=2) as cad:
            cad.disk(0.0, 0.0, 1.0)
            native_mesh = _generate_auto_mesh(cad,
                level=2,
                cell_shape="quad",
            )
            quad = gmsh_io.read(native_mesh)
        assert {element.type for element in quad.elements} == {"Quad4"}
        assert {
            name: real_gmsh.option.getNumber(name) for name in external_values
        } == external_values

        with geometry.model("auto_restore_hex", dimension=3) as cad:
            cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            native_mesh = _generate_auto_mesh(cad,
                level=1,
                cell_shape="hex",
            )
            hexahedra = gmsh_io.read(native_mesh)
        assert {element.type for element in hexahedra.elements} == {"Hex8"}
        assert {
            name: real_gmsh.option.getNumber(name) for name in external_values
        } == external_values
    finally:
        for name, value in prior_values.items():
            real_gmsh.option.setNumber(name, value)


def test_real_1d_facade_reuses_shared_point_in_connected_spatial_mesh(
    real_gmsh: Any,
) -> None:
    middle_coordinates = (1.0, 0.5, 0.75)
    with geometry.model("facade_connected_lines", dimension=1) as cad:
        start = cad.point(0.0, 0.0, 0.25)
        middle = cad.point(*middle_coordinates)
        end = cad.point(2.0, -0.5, 1.25)
        cad.line(start, middle)
        cad.line(middle, end)
        native_mesh = _generate_mesh(cad, size=0.4)
        mesh = gmsh_io.read(
            native_mesh,
            line_element_type="Truss2",
        )

    assert isinstance(mesh, Mesh3D)
    assert mesh.dofs_per_node == 3
    assert {element.type for element in mesh.elements} == {"Truss2"}
    assert all(len(element.node_ids) == 2 for element in mesh.elements)
    middle_node = next(
        node
        for node in mesh.nodes
        if (node.x, node.y, node.z) == pytest.approx(middle_coordinates)
    )
    assert sum(
        middle_node.id in element.node_ids for element in mesh.elements
    ) == 2
    _assert_vtk_cell_type(mesh, 3)


def test_real_1d_fragment_splits_intersections_into_shared_mesh_node(
    real_gmsh: Any,
) -> None:
    with geometry.model("facade_fragmented_lines", dimension=1) as cad:
        left = cad.point(-1.0, 0.0, 0.0)
        right = cad.point(1.0, 0.0, 0.0)
        bottom = cad.point(0.0, -1.0, 0.0)
        top = cad.point(0.0, 1.0, 0.0)
        horizontal = cad.line(left, right)
        vertical = cad.line(bottom, top)

        fragmented = cad.fragment([horizontal], [vertical])
        members = fragmented.of_dimension(1)
        assert len(members) == 4
        assert tuple(len(group) for group in fragmented.input_map) == (2, 2)
        with pytest.raises(geometry.StaleEntityError):
            cad.translate([horizontal], 0.0, 0.0, 0.0)

        center = cad.select(cad.entities(0), x=0.0, y=0.0, z=0.0)
        assert len(center) == 1
        native_mesh = _generate_mesh(cad, size=0.4)
        mesh = gmsh_io.read(
            native_mesh,
            line_element_type="Truss2",
        )

    center_node_id = nodes.by_coord(mesh, x=0.0, y=0.0, z=0.0)[0]
    assert sum(
        center_node_id in element.node_ids for element in mesh.elements
    ) == 4
    assert mesh.num_elements >= len(members)


def test_real_truss2_vertical_slice_matches_bar_solution_and_exports_vtk(
    real_gmsh: Any,
    tmp_path: Path,
) -> None:
    length = 2.0
    elastic_modulus = 210.0e9
    area = 1.0e-4
    force = 1.0e4
    with geometry.model("truss_vertical_slice", dimension=1) as cad:
        start = cad.point(0.0, 0.5, -0.25)
        end = cad.point(length, 0.5, -0.25)
        cad.line(start, end)
        native_mesh = _generate_mesh(cad, size=0.5)
        mesh = gmsh_io.read(
            native_mesh,
            line_element_type="Truss2",
        )

    model = FEMModel(mesh=mesh, name="truss_vertical_slice")
    member_set = elements.set_all(mesh, "MEMBERS")
    fixed_set = nodes.set_by_coord(
        mesh,
        "FIXED",
        x=0.0,
        y=0.5,
        z=-0.25,
    )
    tip_set = nodes.set_by_coord(
        mesh,
        "TIP",
        x=length,
        y=0.5,
        z=-0.25,
    )
    model.element_sets[member_set.name] = member_set
    model.node_sets[fixed_set.name] = fixed_set
    model.node_sets[tip_set.name] = tip_set

    steel = materials.linear_elastic.material(
        "steel",
        E=elastic_modulus,
        nu=0.3,
    )
    materials.add(model, steel)
    materials.assign(model, steel, "MEMBERS", area=area)
    load_step = steps.static("pull")
    steps.displacement(load_step, "FIXED", components=(1, 2, 3))
    fixed_id = model.node_sets["FIXED"].node_ids[0]
    for node_id in model.mesh.node_ids:
        if node_id != fixed_id:
            steps.displacement(load_step, node_id, components=(2, 3))
    steps.nodal_load(load_step, "TIP", component=1, value=force)
    steps.add(model, load_step)

    result = static_linear.solve(model, load_step)
    tip_id = model.node_sets["TIP"].node_ids[0]
    assert result.U[model.mesh.global_dof(tip_id, 0)] == pytest.approx(
        force * length / (elastic_modulus * area)
    )
    stresses = [
        get_element_kernel(element.type).element_stress(
            model.mesh,
            element,
            result.U,
        )[1]
        for element in model.mesh.elements
    ]
    assert stresses == pytest.approx([force / area] * model.mesh.num_elements)

    post.vtk.export.from_result(result, output_dir=tmp_path, name="truss_slice")
    vtk_path = tmp_path / "truss_slice.vtk"
    vtk_text = vtk_path.read_text(encoding="utf-8")
    assert f"CELL_TYPES {model.mesh.num_elements}" in vtk_text
    assert "\n3\n" in vtk_text
    assert "VECTORS displacement float" in vtk_text
    assert "SCALARS axial_stress float 1" in vtk_text


def test_real_beam2_vertical_slice_uses_fixed_rectangle_axes_and_line_load(
    real_gmsh: Any,
    tmp_path: Path,
) -> None:
    length = 2.0
    elastic_modulus = 210.0e9
    tip_force = 1.0e3
    line_load = 5.0e2
    with geometry.model("beam_vertical_slice", dimension=1) as cad:
        root = cad.point(0.0, 0.0, 0.0)
        tip = cad.point(length, 0.0, 0.0)
        cad.line(root, tip)
        native_mesh = _generate_mesh(cad, size=0.5)
        mesh = gmsh_io.read(
            native_mesh,
            line_element_type="Beam2",
        )

    model = FEMModel(mesh=mesh, name="beam_vertical_slice")
    member_set = elements.set_all(mesh, "MEMBERS")
    fixed_set = nodes.set_by_coord(mesh, "FIXED", x=0.0, y=0.0, z=0.0)
    tip_set = nodes.set_by_coord(mesh, "TIP", x=length, y=0.0, z=0.0)
    model.element_sets[member_set.name] = member_set
    model.node_sets[fixed_set.name] = fixed_set
    model.node_sets[tip_set.name] = tip_set

    steel = materials.linear_elastic.material(
        "steel",
        E=elastic_modulus,
        nu=0.3,
    )
    materials.add(model, steel)
    materials.assign(
        model,
        steel,
        "MEMBERS",
        section_type="rectangle",
        height=0.2,
        width=0.1,
    )

    def fixed_step(name: str):
        step = steps.static(name)
        steps.displacement(step, "FIXED", components=(1, 2, 3, 4, 5, 6))
        steps.add(model, step)
        return step

    tip_y_step = fixed_step("tip_y")
    steps.nodal_load(tip_y_step, "TIP", component=2, value=tip_force)
    tip_z_step = fixed_step("tip_z")
    steps.nodal_load(tip_z_step, "TIP", component=3, value=tip_force)
    distributed_step = fixed_step("distributed_y")
    steps.line_load(distributed_step, "MEMBERS", (0.0, line_load, 0.0))

    tip_y_result = static_linear.solve(model, tip_y_step)
    section = parse_beam2_section(model.mesh.elements[0].props)
    tip_z_result = static_linear.solve(model, tip_z_step)
    distributed_result = static_linear.solve(model, distributed_step)
    tip_id = model.node_sets["TIP"].node_ids[0]
    assert tip_y_result.U[model.mesh.global_dof(tip_id, 1)] == pytest.approx(
        tip_force * length**3 / (3.0 * elastic_modulus * section.Izz)
    )
    assert tip_z_result.U[model.mesh.global_dof(tip_id, 2)] == pytest.approx(
        tip_force * length**3 / (3.0 * elastic_modulus * section.Iyy)
    )
    assert distributed_result.U[model.mesh.global_dof(tip_id, 1)] == pytest.approx(
        line_load * length**4 / (8.0 * elastic_modulus * section.Izz)
    )
    envelope = post.stress.beam.nodal_envelope(distributed_result)
    assert max(row.absolute_maximum for row in envelope) > 0.0

    post.vtk.export.from_result(
        distributed_result,
        output_dir=tmp_path,
        name="beam_slice",
    )
    vtk_text = (tmp_path / "beam_slice.vtk").read_text(encoding="utf-8")
    assert "\n3\n" in vtk_text
    assert "VECTORS displacement float" in vtk_text
    assert "VECTORS rotation float" in vtk_text
    assert "SCALARS axial_stress_max float 1" in vtk_text
    assert "SCALARS axial_stress_min float 1" in vtk_text
    assert "SCALARS axial_stress_abs_max float 1" in vtk_text


def test_real_facade_rectangle_selects_regions_solves_and_survives_cleanup(
    real_gmsh: Any,
    tmp_path: Path,
) -> None:
    with geometry.model("facade_rectangle", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 2.0, 1.0)
        native_mesh = _generate_mesh(cad, size=0.35)
        mesh = gmsh_io.read(native_mesh)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 2)

    assert real_gmsh.isInitialized()
    assert "facade_rectangle" not in real_gmsh.model.list()
    assert isinstance(mesh, Mesh2D)
    assert {element.type for element in mesh.elements} == {"Tri3"}
    _assert_vtk_cell_type(mesh, 5)

    model = FEMModel(mesh=mesh, name="facade_rectangle")
    domain = elements.set_all(mesh, "DOMAIN")
    left = nodes.set_by_x(mesh, "LEFT", 0.0)
    right = nodes.set_by_x(mesh, "RIGHT", 2.0)
    right_edge = edges.edge_by_x(mesh, "RIGHT", 2.0)
    model.element_sets[domain.name] = domain
    model.node_sets[left.name] = left
    model.node_sets[right.name] = right
    model.edges[right_edge.name] = right_edge
    elastic = materials.linear_elastic.material("elastic", E=1000.0, nu=0.3)
    materials.add(model, elastic)
    materials.assign(model, "elastic", "DOMAIN")
    load_step = steps.static("pull")
    steps.displacement(load_step, "LEFT", components=(1, 2))
    steps.edge_traction(load_step, "RIGHT", vector=(2.0, 0.0))
    steps.add(model, load_step)
    validate_model(model)

    result = static_linear.solve(model, "pull")
    assert np.all(np.isfinite(result.U))
    assert np.all(np.isfinite(result.reactions))
    assert np.linalg.norm(result.reactions) > 0.0
    post.vtk.export.from_result(
        result,
        output_dir=tmp_path,
        name="facade_rectangle",
    )
    vtk_path = tmp_path / "facade_rectangle.vtk"
    vtk_text = vtk_path.read_text(encoding="utf-8")
    points_line = next(
        line for line in vtk_text.splitlines() if line.startswith("POINTS ")
    )
    vtk_point_count = int(points_line.split()[1])
    assert vtk_point_count >= model.mesh.num_nodes
    assert f"CELLS {model.mesh.num_elements}" in vtk_text
    assert f"POINT_DATA {vtk_point_count}" in vtk_text
    vtk_lines = vtk_text.splitlines()
    cell_types_index = vtk_lines.index(f"CELL_TYPES {model.mesh.num_elements}")
    assert [
        int(value)
        for value in vtk_lines[
            cell_types_index + 1 : cell_types_index + 1 + model.mesh.num_elements
        ]
    ] == [5] * model.mesh.num_elements


def test_real_facade_cut_creates_quadratic_tri6_hole_mesh(real_gmsh: Any) -> None:
    with geometry.model("facade_hole", dimension=2) as cad:
        plate = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        hole = cad.disk(1.0, 0.5, 0.2)
        cut = cad.cut([plate], [hole])
        domain = cut.of_dimension(2)
        assert len(domain) == 1
        assert len(cad.boundary(domain)) == 5
        native_mesh = _generate_mesh(cad, size=0.25, order=2)
        mesh = gmsh_io.read(native_mesh)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 2)

    assert isinstance(mesh, Mesh2D)
    assert {element.type for element in mesh.elements} == {"Tri6"}
    assert all(len(element.node_ids) == 6 for element in mesh.elements)
    _assert_vtk_cell_type(mesh, 22)


def test_real_destructive_fragment_invalidates_old_boundary_references(
    real_gmsh: Any,
) -> None:
    with geometry.model("facade_fragment_stale", dimension=2) as cad:
        first = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        second = cad.rectangle(1.0, -0.5, 1.0, 2.0)
        old_bottom = cad.select(cad.boundary([first]), y=0.0)[0]
        old_points = cad.boundary([old_bottom], combined=False)
        assert old_points

        fragmented = cad.fragment([first], [second])

        with pytest.raises(geometry.StaleEntityError):
            cad.select([old_bottom], y=0.0)
        for point in old_points:
            with pytest.raises(geometry.StaleEntityError):
                cad.translate([point], 0.0, 0.0, 0.0)
        surfaces = fragmented.of_dimension(2)
        assert surfaces
        assert cad.boundary(surfaces)


def test_real_facade_supports_y_major_disk_and_strictly_valid_rounding(
    real_gmsh: Any,
) -> None:
    with geometry.model("facade_occ_parameters", dimension=2) as cad:
        ellipse = cad.disk(0.0, 0.0, 1.0, radius_y=2.0)
        rounded = cad.rectangle(3.0, 0.0, 2.0, 1.0, rounded_radius=0.49)
        ellipse_tag = ellipse.tag
        rounded_tag = rounded.tag

        raw_model = cad.raw_model
        raw_model.occ.synchronize()
        bounds = raw_model.getBoundingBox(2, ellipse_tag)
        assert bounds[3] - bounds[0] == pytest.approx(2.0, abs=1.0e-6)
        assert bounds[4] - bounds[1] == pytest.approx(4.0, abs=1.0e-6)
        assert raw_model.occ.getMass(2, rounded_tag) > 0.0


def test_real_size_control_overrides_and_restores_external_point_size_option(
    real_gmsh: Any,
) -> None:
    option_name = "Mesh.MeshSizeFromPoints"
    original = real_gmsh.option.getNumber(option_name)
    real_gmsh.option.setNumber(option_name, 0.0)
    try:
        with geometry.model("facade_size_fine", dimension=2) as cad:
            cad.rectangle(0.0, 0.0, 1.0, 1.0)
            native_mesh = _generate_mesh(cad, size=0.1)
            fine = gmsh_io.read(native_mesh)
            assert real_gmsh.option.getNumber(option_name) == 0.0

        with geometry.model("facade_size_coarse", dimension=2) as cad:
            cad.rectangle(0.0, 0.0, 1.0, 1.0)
            native_mesh = _generate_mesh(cad, size=0.5)
            coarse = gmsh_io.read(native_mesh)
            assert real_gmsh.option.getNumber(option_name) == 0.0
    finally:
        real_gmsh.option.setNumber(option_name, original)

    assert fine.num_elements > coarse.num_elements


def test_real_transfinite_line_creates_exact_truss2_mesh(real_gmsh: Any) -> None:
    with geometry.model("facade_transfinite_line", dimension=1) as cad:
        start = cad.point(0.0, 0.0, 0.0)
        end = cad.point(2.0, 0.0, 0.0)
        member = cad.line(start, end)
        _mesher(cad).transfinite_curve(member, num_nodes=5)
        native_mesh = _generate_mesh(cad, )
        mesh = gmsh_io.read(native_mesh, line_element_type="Truss2")

    assert isinstance(mesh, Mesh3D)
    assert mesh.num_nodes == 5
    assert mesh.num_elements == 4
    assert {element.type for element in mesh.elements} == {"Truss2"}
    _assert_vtk_cell_type(mesh, 3)


def test_real_facade_structured_rectangle_creates_quad8(real_gmsh: Any) -> None:
    with geometry.model("facade_quad8", dimension=2) as cad:
        surface = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        curves = cad.boundary([surface])
        for curve in curves:
            _mesher(cad).transfinite_curve(curve, num_nodes=3)
        _mesher(cad).transfinite_surface(surface)
        _mesher(cad).recombine(surface)

        native_mesh = _generate_mesh(cad, order=2, recombine=False)
        mesh = gmsh_io.read(native_mesh)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 2)

    assert isinstance(mesh, Mesh2D)
    assert mesh.num_elements == 4
    assert {element.type for element in mesh.elements} == {"Quad8"}
    assert all(len(element.node_ids) == 8 for element in mesh.elements)
    assert edges.edge_by_x(mesh, "LEFT", 0.0).edges
    _assert_vtk_cell_type(mesh, 23)


def test_real_entity_recombine_leaves_unselected_surface_triangular(
    real_gmsh: Any,
) -> None:
    real_gmsh.option.setNumber("Mesh.RecombineAll", 1.0)
    with geometry.model("facade_selective_recombine", dimension=2) as cad:
        structured = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        cad.rectangle(2.0, 0.0, 1.0, 1.0)
        structured_curves = cad.boundary([structured])
        for curve in structured_curves:
            _mesher(cad).transfinite_curve(curve, num_nodes=3)
        _mesher(cad).transfinite_surface(structured)
        _mesher(cad).recombine(structured)

        native_mesh = _generate_mesh(cad, size=0.3, recombine=False)
        mesh = gmsh_io.read(native_mesh)
        assert real_gmsh.option.getNumber("Mesh.RecombineAll") == 1.0

    element_types = [element.type for element in mesh.elements]
    assert element_types.count("Quad4") == 4
    assert "Tri3" in element_types
    assert edges.edge_by_x(mesh, "STRUCTURED_LEFT", 0.0).edges


def test_real_facade_box_creates_tet10(real_gmsh: Any) -> None:
    with geometry.model("facade_tet10", dimension=3) as cad:
        cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        native_mesh = _generate_mesh(cad, size=0.7, order=2)
        mesh = gmsh_io.read(native_mesh)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 3)

    assert isinstance(mesh, Mesh3D)
    assert {element.type for element in mesh.elements} == {"Tet10"}
    _assert_vtk_cell_type(mesh, 24)


def test_real_facade_transfinite_box_creates_exact_hex20_mesh(
    real_gmsh: Any,
) -> None:
    with geometry.model("facade_transfinite_hex20", dimension=3) as cad:
        volume = cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        faces = cad.boundary([volume])
        edges = cad.boundary(faces, combined=False)
        assert len(faces) == 6
        assert len(edges) == 12

        for edge in edges:
            _mesher(cad).transfinite_curve(edge, num_nodes=3)
        for face in faces:
            _mesher(cad).transfinite_surface(face)
            _mesher(cad).recombine(face)
        _mesher(cad).transfinite_volume(volume)

        native_mesh = _generate_mesh(cad, order=2, recombine=False)
        mesh = gmsh_io.read(native_mesh)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 3)

    assert isinstance(mesh, Mesh3D)
    assert mesh.num_elements == 8
    assert {element.type for element in mesh.elements} == {"Hex20"}
    assert all(len(element.node_ids) == 20 for element in mesh.elements)
    _assert_vtk_cell_type(mesh, 25)


def test_real_facade_structured_extrusion_creates_hex20(real_gmsh: Any) -> None:
    with geometry.model("facade_hex20", dimension=3) as cad:
        surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        extruded = _structured_extrude(
            cad,
            [surface],
            0.0,
            0.0,
            1.0,
            num_elements=(2,),
            recombine=True,
        )
        assert len(extruded.primary) == 1
        assert extruded.of_dimension(3) == extruded.primary
        native_mesh = _generate_mesh(cad, size=0.5, order=2, recombine=True)
        mesh = gmsh_io.read(native_mesh)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 3)

    assert isinstance(mesh, Mesh3D)
    assert {element.type for element in mesh.elements} == {"Hex20"}
    assert all(len(element.node_ids) == 20 for element in mesh.elements)
    _assert_vtk_cell_type(mesh, 25)
