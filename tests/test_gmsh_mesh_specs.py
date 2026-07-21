from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import inspect

import pytest

from fem import geometry
from fem.mesh import gmsh as meshing
from fem.mesh.gmsh import errors, mesher, specs, types


def test_mesh_package_exports_exact_canonical_contracts() -> None:
    assert meshing.__all__ == [
        "AutoMeshSpec",
        "GmshMeshRef",
        "MeshCellShapeError",
        "MeshControlConflictError",
        "MeshFieldOwnershipError",
        "MeshFieldRef",
        "MeshSpec",
        "Mesher",
        "StaleGmshMeshError",
        "StaleMeshFieldError",
    ]
    owners = {
        "AutoMeshSpec": specs,
        "GmshMeshRef": types,
        "MeshCellShapeError": errors,
        "MeshControlConflictError": errors,
        "MeshFieldOwnershipError": errors,
        "MeshFieldRef": types,
        "MeshSpec": specs,
        "Mesher": mesher,
        "StaleGmshMeshError": errors,
        "StaleMeshFieldError": errors,
    }
    for name, owner in owners.items():
        value = getattr(meshing, name)
        assert value is getattr(owner, name)
        assert value.__module__ == owner.__name__


def test_mesh_spec_fields_defaults_and_signatures_are_unchanged() -> None:
    assert [(item.name, item.default) for item in fields(meshing.MeshSpec)] == [
        ("size", None),
        ("order", 1),
        ("recombine", False),
    ]
    assert [
        (item.name, item.default) for item in fields(meshing.AutoMeshSpec)
    ] == [
        ("level", 3),
        ("cell_shape", None),
        ("order", 1),
    ]
    assert tuple(inspect.signature(meshing.MeshSpec).parameters) == (
        "size",
        "order",
        "recombine",
    )
    assert tuple(inspect.signature(meshing.AutoMeshSpec).parameters) == (
        "level",
        "cell_shape",
        "order",
    )


def test_mesh_specs_remain_frozen_slotted_and_normalize_size() -> None:
    explicit = meshing.MeshSpec(size=2, order=2, recombine=True)
    automatic = meshing.AutoMeshSpec(level=4, cell_shape="quad", order=2)

    assert explicit.size == 2.0
    assert not hasattr(explicit, "__dict__")
    assert not hasattr(automatic, "__dict__")
    with pytest.raises(FrozenInstanceError):
        explicit.size = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        automatic.level = 3  # type: ignore[misc]


def test_all_mesh_errors_preserve_geometry_error_hierarchy() -> None:
    for error_type in (
        meshing.MeshCellShapeError,
        meshing.MeshControlConflictError,
        meshing.MeshFieldOwnershipError,
        meshing.StaleMeshFieldError,
        meshing.StaleGmshMeshError,
    ):
        assert issubclass(error_type, geometry.GeometryError)


@pytest.mark.parametrize(
    "value",
    [0, -1, float("inf"), float("nan"), True, "bad"],
)
def test_mesh_spec_size_validation_text_is_preserved(value: object) -> None:
    with pytest.raises(ValueError, match="size must be finite and > 0"):
        meshing.MeshSpec(size=value)  # type: ignore[arg-type]


def test_auto_mesh_spec_vocabulary_validation_text_is_preserved() -> None:
    with pytest.raises(ValueError, match="cell_shape must be exactly"):
        meshing.AutoMeshSpec(cell_shape="line")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="level must be a Python integer"):
        meshing.AutoMeshSpec(level=True)  # type: ignore[arg-type]
