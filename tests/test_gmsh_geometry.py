from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest

from fem import materials, post, steps
from fem.core import Mesh2D, Mesh3D, validate_model
from fem.elements import get_element_kernel
from fem.elements.beam_section import parse_beam2_section
from fem.geometry import gmsh as geometry
from fem.solvers import static_linear


class _FakeOcc:
    def __init__(self, model: _FakeModel) -> None:
        self._model = model
        self.synchronize_calls = 0
        self.calls: list[tuple[Any, ...]] = []
        self.boolean_results: dict[
            str,
            tuple[list[tuple[int, int]], list[list[tuple[int, int]]]],
        ] = {}
        self.fail_next: set[str] = set()
        self.extrude_result: list[tuple[int, int]] | None = None

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self.calls.append(("synchronize", self._model.current))

    def getEntities(self, dimension: int = -1) -> list[tuple[int, int]]:
        entities = self._model._current_data()["entities"]
        return sorted(
            pair for pair in entities if dimension == -1 or pair[0] == dimension
        )

    def _allocate(self, dimension: int) -> int:
        data = self._model._current_data()
        next_tags = data["next_tags"]
        tag = next_tags.get(dimension, 1)
        next_tags[dimension] = tag + 1
        data["entities"].add((dimension, tag))
        return tag

    def addPoint(
        self,
        x: float,
        y: float,
        z: float,
        meshSize: float = 0.0,
        tag: int = -1,
    ) -> int:
        self.calls.append(("addPoint", x, y, z, meshSize, tag))
        return self._allocate(0)

    def addLine(self, start_tag: int, end_tag: int, tag: int = -1) -> int:
        self.calls.append(("addLine", start_tag, end_tag, tag))
        return self._allocate(1)

    def addRectangle(
        self,
        x: float,
        y: float,
        z: float,
        dx: float,
        dy: float,
        tag: int = -1,
        roundedRadius: float = 0.0,
    ) -> int:
        self.calls.append(
            ("addRectangle", x, y, z, dx, dy, tag, roundedRadius)
        )
        return self._allocate(2)

    def addDisk(
        self,
        x: float,
        y: float,
        z: float,
        radius_x: float,
        radius_y: float,
        tag: int = -1,
        zAxis: Sequence[float] = (),
        xAxis: Sequence[float] = (),
    ) -> int:
        self.calls.append(
            (
                "addDisk",
                x,
                y,
                z,
                radius_x,
                radius_y,
                tag,
                tuple(zAxis),
                tuple(xAxis),
            )
        )
        return self._allocate(2)

    def addBox(
        self,
        x: float,
        y: float,
        z: float,
        dx: float,
        dy: float,
        dz: float,
        tag: int = -1,
    ) -> int:
        self.calls.append(("addBox", x, y, z, dx, dy, dz, tag))
        return self._allocate(3)

    def addCylinder(
        self,
        x: float,
        y: float,
        z: float,
        axis_x: float,
        axis_y: float,
        axis_z: float,
        radius: float,
        tag: int = -1,
        angle: float = 2.0 * 3.141592653589793,
    ) -> int:
        self.calls.append(
            (
                "addCylinder",
                x,
                y,
                z,
                axis_x,
                axis_y,
                axis_z,
                radius,
                tag,
                angle,
            )
        )
        return self._allocate(3)

    def fuse(
        self,
        objects: list[tuple[int, int]],
        tools: list[tuple[int, int]],
        tag: int = -1,
        removeObject: bool = True,
        removeTool: bool = True,
    ) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
        return self._boolean(
            "fuse", objects, tools, tag, removeObject, removeTool
        )

    def cut(
        self,
        objects: list[tuple[int, int]],
        tools: list[tuple[int, int]],
        tag: int = -1,
        removeObject: bool = True,
        removeTool: bool = True,
    ) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
        return self._boolean(
            "cut", objects, tools, tag, removeObject, removeTool
        )

    def fragment(
        self,
        objects: list[tuple[int, int]],
        tools: list[tuple[int, int]],
        tag: int = -1,
        removeObject: bool = True,
        removeTool: bool = True,
    ) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
        return self._boolean(
            "fragment", objects, tools, tag, removeObject, removeTool
        )

    def _boolean(
        self,
        name: str,
        objects: list[tuple[int, int]],
        tools: list[tuple[int, int]],
        tag: int,
        remove_objects: bool,
        remove_tools: bool,
    ) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
        self.calls.append(
            (
                name,
                tuple(objects),
                tuple(tools),
                tag,
                remove_objects,
                remove_tools,
            )
        )
        if name in self.fail_next:
            self.fail_next.remove(name)
            raise RuntimeError(f"fake {name} failure")
        outputs, input_map = self.boolean_results.get(
            name,
            ([objects[0]], [[pair] for pair in (*objects, *tools)]),
        )
        entities = self._model._current_data()["entities"]
        if remove_objects:
            entities.difference_update(objects)
        if remove_tools:
            entities.difference_update(tools)
        entities.update(outputs)
        return list(outputs), [list(group) for group in input_map]

    def translate(
        self,
        entities: list[tuple[int, int]],
        dx: float,
        dy: float,
        dz: float,
    ) -> None:
        self.calls.append(("translate", tuple(entities), dx, dy, dz))

    def rotate(
        self,
        entities: list[tuple[int, int]],
        x: float,
        y: float,
        z: float,
        axis_x: float,
        axis_y: float,
        axis_z: float,
        angle: float,
    ) -> None:
        self.calls.append(
            (
                "rotate",
                tuple(entities),
                x,
                y,
                z,
                axis_x,
                axis_y,
                axis_z,
                angle,
            )
        )

    def extrude(
        self,
        entities: list[tuple[int, int]],
        dx: float,
        dy: float,
        dz: float,
        numElements: list[int] | None = None,
        heights: list[float] | None = None,
        recombine: bool = False,
    ) -> list[tuple[int, int]]:
        self.calls.append(
            (
                "extrude",
                tuple(entities),
                dx,
                dy,
                dz,
                tuple(numElements or ()),
                tuple(heights or ()),
                recombine,
            )
        )
        if "extrude" in self.fail_next:
            self.fail_next.remove("extrude")
            raise RuntimeError("fake extrude failure")
        outputs = self.extrude_result
        if outputs is None:
            outputs = [entities[0], (entities[0][0] + 1, self._allocate(entities[0][0] + 1))]
        self._model._current_data()["entities"].update(outputs)
        return list(outputs)


class _FakeMeshField:
    def __init__(self, model: _FakeModel) -> None:
        self._model = model
        self.calls: list[tuple[Any, ...]] = []
        self.fail_next: set[tuple[str, str | None]] = set()
        self.fail_after: dict[tuple[str, str | None], int] = {}
        self.add_results: list[int] = []
        self.hidden_tags: set[int] = set()

    def _maybe_fail(self, operation: str, option: str | None = None) -> None:
        keys = (
            ((operation, None),)
            if option is None
            else ((operation, option), (operation, None))
        )
        for key in keys:
            if key in self.fail_after:
                remaining = self.fail_after[key]
                if remaining == 0:
                    del self.fail_after[key]
                    suffix = f" for {option}" if option is not None else ""
                    raise RuntimeError(f"fake field {operation} failure{suffix}")
                self.fail_after[key] = remaining - 1
            if key in self.fail_next:
                self.fail_next.remove(key)
                suffix = f" for {option}" if option is not None else ""
                raise RuntimeError(f"fake field {operation} failure{suffix}")

    def _fields(self) -> dict[int, dict[str, Any]]:
        return self._model._current_data()["mesh_fields"]

    def add(self, field_type: str, tag: int = -1) -> int:
        self.calls.append(("add", field_type, tag, self._model.current))
        self._maybe_fail("add", field_type)
        fields = self._fields()
        if self.add_results:
            allocated_tag = self.add_results.pop(0)
        elif tag >= 0:
            allocated_tag = tag
        else:
            allocated_tag = 1
            while allocated_tag in fields:
                allocated_tag += 1
        if allocated_tag > 0:
            if allocated_tag in fields:
                raise RuntimeError(f"fake duplicate field tag {allocated_tag}")
            fields[allocated_tag] = {
                "type": field_type,
                "numbers": {},
                "number_lists": {},
            }
        return allocated_tag

    def setNumber(self, tag: int, option: str, value: float) -> None:
        numeric_value = float(value)
        self.calls.append(
            ("setNumber", tag, option, numeric_value, self._model.current)
        )
        self._maybe_fail("setNumber", option)
        try:
            field = self._fields()[tag]
        except KeyError as exc:
            raise RuntimeError(f"fake unknown field tag {tag}") from exc
        field["numbers"][option] = numeric_value

    def setNumbers(
        self,
        tag: int,
        option: str,
        values: Sequence[float],
    ) -> None:
        materialized = tuple(values)
        self.calls.append(
            ("setNumbers", tag, option, materialized, self._model.current)
        )
        self._maybe_fail("setNumbers", option)
        try:
            field = self._fields()[tag]
        except KeyError as exc:
            raise RuntimeError(f"fake unknown field tag {tag}") from exc
        field["number_lists"][option] = materialized

    def setAsBackgroundMesh(self, tag: int) -> None:
        self.calls.append(("setAsBackgroundMesh", tag, self._model.current))
        self._maybe_fail("setAsBackgroundMesh")
        if tag not in self._fields():
            raise RuntimeError(f"fake unknown field tag {tag}")
        self._model._current_data()["background_mesh_field"] = tag

    def remove(self, tag: int) -> None:
        self.calls.append(("remove", tag, self._model.current))
        self._maybe_fail("remove")
        try:
            del self._fields()[tag]
        except KeyError as exc:
            raise RuntimeError(f"fake unknown field tag {tag}") from exc
        data = self._model._current_data()
        if data["background_mesh_field"] == tag:
            data["background_mesh_field"] = None
        self.hidden_tags.discard(tag)

    def list(self) -> list[int]:
        self.calls.append(("list", self._model.current))
        self._maybe_fail("list")
        return sorted(set(self._fields()).difference(self.hidden_tags))


class _FakeMesh:
    def __init__(self, model: _FakeModel) -> None:
        self._model = model
        self.field = _FakeMeshField(model)
        self.generate_calls: list[int] = []
        self.calls: list[tuple[Any, ...]] = []
        self.fail_next: set[str] = set()
        self.fail_generate = False
        self.fail_set_size = False

    def setTransfiniteCurve(self, tag: int, num_nodes: int) -> None:
        self._record_control("setTransfiniteCurve", tag, num_nodes)

    def setTransfiniteSurface(
        self,
        tag: int,
        arrangement: str = "Left",
        cornerTags: Sequence[int] = (),
    ) -> None:
        self._record_control(
            "setTransfiniteSurface",
            tag,
            arrangement,
            tuple(cornerTags),
        )

    def setTransfiniteVolume(
        self,
        tag: int,
        cornerTags: Sequence[int] = (),
    ) -> None:
        self._record_control("setTransfiniteVolume", tag, tuple(cornerTags))

    def setRecombine(self, dimension: int, tag: int) -> None:
        self._record_control("setRecombine", dimension, tag)

    def _record_control(self, operation: str, *args: Any) -> None:
        self.calls.append((operation, *args, self._model.current))
        if operation in self.fail_next:
            self.fail_next.remove(operation)
            raise RuntimeError(f"fake {operation} failure")

    def setSize(
        self,
        points: list[tuple[int, int]],
        size: float,
    ) -> None:
        self.calls.append(("setSize", tuple(points), size, self._model.current))
        if self.fail_set_size:
            raise RuntimeError("fake setSize failure")

    def generate(self, dimension: int) -> None:
        self.calls.append(("generate", dimension, self._model.current))
        self.generate_calls.append(dimension)
        if self.fail_generate:
            raise RuntimeError("fake mesh failure")


class _FakeModel:
    def __init__(
        self,
        *,
        names: tuple[str, ...] = (),
        current: str = "",
    ) -> None:
        self.models: dict[str, dict[str, Any]] = {
            name: self._new_data() for name in names
        }
        self.current = current
        self.calls: list[tuple[Any, ...]] = []
        self.occ = _FakeOcc(self)
        self.mesh = _FakeMesh(self)
        self.boundary_result: list[tuple[int, int]] = []
        self.fail_next_physical = False
        self.fail_remove = False
        self.fail_set_current_names: set[str] = set()

    @staticmethod
    def _new_data() -> dict[str, Any]:
        return {
            "entities": set(),
            "next_tags": {},
            "boxes": {},
            "physical_groups": {},
            "next_physical_tags": {},
            "mesh_fields": {},
            "background_mesh_field": None,
        }

    def _current_data(self) -> dict[str, Any]:
        if self.current not in self.models:
            raise RuntimeError(f"model {self.current!r} is not available")
        return self.models[self.current]

    def list(self) -> list[str]:
        self.calls.append(("list",))
        return list(self.models)

    def getCurrent(self) -> str:
        self.calls.append(("getCurrent",))
        return self.current

    def setCurrent(self, name: str) -> None:
        self.calls.append(("setCurrent", name))
        if name in self.fail_set_current_names:
            self.fail_set_current_names.remove(name)
            raise RuntimeError(f"fake setCurrent failure for {name!r}")
        if name not in self.models:
            raise RuntimeError(f"unknown model {name!r}")
        self.current = name

    def add(self, name: str) -> None:
        self.calls.append(("add", name))
        if name in self.models:
            raise RuntimeError(f"duplicate model {name!r}")
        self.models[name] = self._new_data()
        self.current = name

    def remove(self) -> None:
        self.calls.append(("remove", self.current))
        if self.fail_remove:
            raise RuntimeError("fake remove failure")
        if self.current not in self.models:
            raise RuntimeError(f"unknown model {self.current!r}")
        del self.models[self.current]
        self.current = next(iter(self.models), "")

    def getEntities(self, dimension: int = -1) -> list[tuple[int, int]]:
        self.calls.append(("getEntities", dimension, self.current))
        entities = self._current_data()["entities"]
        return sorted(
            pair for pair in entities if dimension == -1 or pair[0] == dimension
        )

    def getBoundary(
        self,
        entities: list[tuple[int, int]],
        combined: bool = True,
        oriented: bool = True,
        recursive: bool = False,
    ) -> list[tuple[int, int]]:
        self.calls.append(
            (
                "getBoundary",
                tuple(entities),
                combined,
                oriented,
                recursive,
                self.current,
            )
        )
        return list(self.boundary_result)

    def getBoundingBox(
        self,
        dimension: int,
        tag: int,
    ) -> tuple[float, float, float, float, float, float]:
        self.calls.append(("getBoundingBox", dimension, tag, self.current))
        return self._current_data()["boxes"][(dimension, tag)]

    def addPhysicalGroup(
        self,
        dimension: int,
        tags: list[int],
        tag: int = -1,
        name: str = "",
    ) -> int:
        self.calls.append(
            (
                "addPhysicalGroup",
                dimension,
                tuple(tags),
                tag,
                name,
                self.current,
            )
        )
        if self.fail_next_physical:
            self.fail_next_physical = False
            raise RuntimeError("fake physical failure")
        data = self._current_data()
        next_tags = data["next_physical_tags"]
        physical_tag = next_tags.get(dimension, 1) if tag == -1 else tag
        next_tags[dimension] = max(physical_tag + 1, next_tags.get(dimension, 1))
        data["physical_groups"][(dimension, physical_tag)] = (name, tuple(tags))
        return physical_tag


class _FakeOption:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.calls: list[tuple[Any, ...]] = []
        self.fail_set_names: set[str] = set()

    def getNumber(self, name: str) -> float:
        self.calls.append(("getNumber", name))
        return self.values.get(name, 0.0)

    def setNumber(self, name: str, value: float) -> None:
        self.calls.append(("setNumber", name, float(value)))
        if name in self.fail_set_names:
            self.fail_set_names.remove(name)
            raise RuntimeError(f"fake option failure for {name}")
        self.values[name] = float(value)


class _FakeGmsh:
    def __init__(
        self,
        *,
        initialized: bool = False,
        names: tuple[str, ...] = (),
        current: str = "",
    ) -> None:
        self.initialized = initialized
        self.initialize_calls = 0
        self.finalize_calls = 0
        self.model = _FakeModel(names=names, current=current)
        self.option = _FakeOption()
        self.__version__ = "4.15.2-fake"
        self.fail_initialize_after_state = False
        self.fail_finalize = False

    def isInitialized(self) -> bool:
        return self.initialized

    def initialize(self) -> None:
        self.initialize_calls += 1
        self.initialized = True
        if self.fail_initialize_after_state:
            raise RuntimeError("fake initialize failure")

    def finalize(self) -> None:
        self.finalize_calls += 1
        if self.fail_finalize:
            self.fail_finalize = False
            raise RuntimeError("fake finalize failure")
        self.initialized = False


class _FakeImportResult:
    def __init__(
        self,
        model_result: Any = None,
        *,
        conversion_error: BaseException | None = None,
    ) -> None:
        self.model_result = model_result
        self.conversion_error = conversion_error
        self.to_fem_model_calls: list[str | None] = []

    def to_fem_model(self, name: str | None = None) -> Any:
        self.to_fem_model_calls.append(name)
        if self.conversion_error is not None:
            raise self.conversion_error
        return self.model_result


def _install_backend(monkeypatch: pytest.MonkeyPatch, backend: _FakeGmsh) -> None:
    monkeypatch.setattr(geometry, "_load_gmsh", lambda: backend)


def _fake_entities(
    cad: geometry.GeometryModel,
    backend: _FakeGmsh,
    dimension: int,
    *tags: int,
) -> tuple[geometry.EntityRef, ...]:
    backend.model._current_data()["entities"].update(
        (dimension, tag) for tag in tags
    )
    return tuple(cad.entity(dimension, tag) for tag in tags)


def _fake_threshold(
    cad: geometry.GeometryModel,
    distance: geometry.MeshFieldRef,
    *,
    size_min: float = 0.05,
    size_max: float = 0.4,
    dist_min: float = 0.1,
    dist_max: float = 0.8,
) -> geometry.MeshFieldRef:
    return cad.threshold_field(
        distance,
        size_min=size_min,
        size_max=size_max,
        dist_min=dist_min,
        dist_max=dist_max,
    )


_ENTITY_DEPENDENT_MESH_CONTROLS = (
    "transfinite_curve",
    "transfinite_surface",
    "transfinite_volume",
    "recombine",
    "mesh_size",
    "distance_field",
    "layered_extrude",
    "recombined_extrude",
)

_TRANSFORM_UNSAFE_ENTITY_CONTROLS = tuple(
    control
    for control in _ENTITY_DEPENDENT_MESH_CONTROLS
    if control != "distance_field"
)


def _apply_entity_dependent_mesh_control(
    cad: geometry.GeometryModel,
    control: str,
    *,
    point: geometry.EntityRef,
    curve: geometry.EntityRef,
    surface: geometry.EntityRef,
    volume: geometry.EntityRef,
) -> None:
    operations = {
        "transfinite_curve": lambda: cad.transfinite_curve(curve, num_nodes=3),
        "transfinite_surface": lambda: cad.transfinite_surface(surface),
        "transfinite_volume": lambda: cad.transfinite_volume(volume),
        "recombine": lambda: cad.recombine(surface),
        "mesh_size": lambda: cad.mesh_size([point], size=0.1),
        "distance_field": lambda: cad.distance_field(surfaces=[surface]),
        "layered_extrude": lambda: cad.extrude(
            [surface],
            0,
            0,
            1,
            num_elements=[2],
            heights=[1.0],
        ),
        "recombined_extrude": lambda: cad.extrude(
            [surface],
            0,
            0,
            1,
            recombine=True,
        ),
    }
    operations[control]()


def _fake_mesh_control_targets(
    cad: geometry.GeometryModel,
    backend: _FakeGmsh,
) -> tuple[
    geometry.EntityRef,
    geometry.EntityRef,
    geometry.EntityRef,
    geometry.EntityRef,
]:
    surface = cad.rectangle(0, 0, 1, 1)
    volume = cad.box(0, 0, 0, 1, 1, 1)
    point = _fake_entities(cad, backend, 0, 11)[0]
    curve = _fake_entities(cad, backend, 1, 12)[0]
    return point, curve, surface, volume


def _entity_control_target(
    control: str,
    *,
    point: geometry.EntityRef,
    curve: geometry.EntityRef,
    surface: geometry.EntityRef,
    volume: geometry.EntityRef,
) -> geometry.EntityRef:
    return {
        "transfinite_curve": curve,
        "transfinite_surface": surface,
        "transfinite_volume": volume,
        "recombine": surface,
        "mesh_size": point,
        "distance_field": surface,
        "layered_extrude": surface,
        "recombined_extrude": surface,
    }[control]


def _fake_control_boundary_dependency(
    cad: geometry.GeometryModel,
    backend: _FakeGmsh,
    control: str,
    *,
    point: geometry.EntityRef,
    curve: geometry.EntityRef,
    surface: geometry.EntityRef,
    volume: geometry.EntityRef,
) -> geometry.EntityRef:
    target = _entity_control_target(
        control,
        point=point,
        curve=curve,
        surface=surface,
        volume=volume,
    )
    if target.dimension == 0:
        return target
    return _fake_entities(cad, backend, target.dimension - 1, 70)[0]


def _apply_typed_transform(
    cad: geometry.GeometryModel,
    operation: str,
    entity: geometry.EntityRef,
) -> tuple[geometry.EntityRef, ...]:
    if operation == "translate":
        return cad.translate([entity], 1, 0, 0)
    return cad.rotate([entity], 0, 0, 0, 0, 0, 1, 0.5)


def _occ_operation_call_count(backend: _FakeGmsh, operation: str) -> int:
    return sum(call[0] == operation for call in backend.model.occ.calls)


def test_importing_geometry_gmsh_does_not_import_external_gmsh() -> None:
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
from fem.geometry import gmsh as geometry
assert geometry.__name__ == "fem.geometry.gmsh"
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_missing_dependency_message_is_actionable_and_closes_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> Any:
        raise ModuleNotFoundError("No module named 'gmsh'", name="gmsh")

    monkeypatch.setattr(geometry, "_load_gmsh", missing)
    cad = geometry.model("missing", dimension=2)

    with pytest.raises(ModuleNotFoundError, match=r"optional 'cad'.*pip install -e"):
        cad.__enter__()

    with pytest.raises(geometry.GeometryStateError, match="missing.*rectangle"):
        cad.rectangle(0.0, 0.0, 1.0, 1.0)


def test_public_reference_types_are_immutable_and_boolean_filter_is_typed() -> None:
    owner = object()
    surface = geometry.EntityRef(2, 8, owner, object())
    curve = geometry.EntityRef(1, 3, owner, object())
    result = geometry.BooleanResult((surface, curve), ((surface,), (curve,)))
    group = geometry.PhysicalGroupRef(2, 4, "DOMAIN", owner)
    mesh_field = geometry.MeshFieldRef(5, "Distance", owner, object())

    assert result.of_dimension(2) == (surface,)
    assert group.name == "DOMAIN"
    assert "object" not in repr(surface)
    assert (mesh_field.tag, mesh_field.field_type) == (5, "Distance")
    assert "object" not in repr(mesh_field)
    assert issubclass(geometry.MeshFieldOwnershipError, geometry.GeometryError)
    assert issubclass(geometry.StaleMeshFieldError, geometry.GeometryError)
    assert {
        "MeshFieldRef",
        "MeshFieldOwnershipError",
        "StaleMeshFieldError",
    }.issubset(geometry.__all__)
    with pytest.raises(FrozenInstanceError):
        surface.tag = 9  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        mesh_field.tag = 9  # type: ignore[misc]
    with pytest.raises(ValueError, match="dimension"):
        result.of_dimension(4)
    with pytest.raises(ValueError, match="mesh field type"):
        geometry.MeshFieldRef(1, "Box", owner, object())  # type: ignore[arg-type]


@pytest.mark.parametrize("dimension", [0, 4, True, "2", None])
def test_model_rejects_invalid_mesh_dimension(dimension: Any) -> None:
    with pytest.raises(ValueError, match="dimension must be 1, 2, or 3"):
        geometry.model("part", dimension=dimension)


def test_owned_session_is_initialized_then_model_is_removed_and_finalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("owned", dimension=2) as cad:
        assert cad.name == "owned"
        assert backend.initialized
        assert backend.model.current == "owned"

    assert backend.initialize_calls == 1
    assert backend.finalize_calls == 1
    assert "owned" not in backend.model.models


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


def test_1d_transform_is_spatial_and_topology_freezes_after_physical_group(
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
        cad.physical("MEMBERS", [member])
        with pytest.raises(geometry.GeometryStateError, match="LABELED"):
            cad.point(2, 0, 0)
        with pytest.raises(geometry.GeometryStateError, match="LABELED"):
            cad.line(start, end)

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


def test_non_destructive_boolean_preserves_input_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("boolean", dimension=2) as cad:
        first = cad.rectangle(0, 0, 2, 1)
        tool = cad.disk(1, 0.5, 0.25)

        cad.fragment(
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


def test_failed_boolean_preserves_input_liveness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("boolean", dimension=2) as cad:
        first = cad.rectangle(0, 0, 2, 1)
        second = cad.rectangle(2, 0, 1, 1)
        backend.model.occ.fail_next.add("fuse")

        with pytest.raises(RuntimeError, match="fake fuse failure"):
            cad.fuse([first], [second])

        assert cad.translate([first, second], 1, 0, 0) == (first, second)


@pytest.mark.parametrize(
    ("malformation", "message"),
    [
        ("map_length", "invalid input map"),
        ("entity_dimension", "invalid boolean output data"),
    ],
)
def test_malformed_destructive_boolean_result_invalidates_changed_inputs(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    message: str,
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
            backend.model.occ.boolean_results["cut"] = (
                [(2, first.tag)],
                [[(2, first.tag)]],
            )
        else:
            backend.model.occ.boolean_results["cut"] = (
                [(2, first.tag)],
                [[(4, first.tag)], []],
            )

        with pytest.raises(geometry.GeometryError, match=message):
            cad.cut([first], [tool])

        for old_reference in (first, tool, unrelated, old_boundary):
            with pytest.raises(geometry.StaleEntityError):
                cad.translate([old_reference], 1, 0, 0)
        reacquired = cad.entity(2, unrelated.tag)
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


def test_boolean_requires_one_common_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("mixed", dimension=3) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        volume = cad.box(0, 0, 0, 1, 1, 1)
        with pytest.raises(ValueError, match="common dimension"):
            cad.fuse([surface], [volume])


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
        backend.model.occ.extrude_result = [(2, 2), (3, 1), (2, 3)]
        result = cad.extrude(
            [surface],
            0,
            0,
            2,
            num_elements=(2, 3),
            heights=(0.4, 1.0),
            recombine=True,
        )

    assert tuple((item.dimension, item.tag) for item in result) == (
        (2, 2),
        (3, 1),
        (2, 3),
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
            cad.extrude([surface], *vector, **options)
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


def test_physical_group_is_named_trimmed_and_freezes_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("labels", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        group = cad.physical("  DOMAIN  ", [surface])

        assert (group.dimension, group.tag, group.name) == (2, 1, "DOMAIN")
        assert cad.entities(2) == (surface,)
        with pytest.raises(geometry.GeometryStateError, match="LABELED"):
            cad.rectangle(2, 0, 1, 1)
        with pytest.raises(geometry.GeometryStateError, match="LABELED"):
            cad.translate([surface], 1, 0, 0)
        with pytest.raises(geometry.GeometryStateError, match="LABELED"):
            _ = cad.raw_occ

    assert (
        "addPhysicalGroup",
        2,
        (1,),
        -1,
        "DOMAIN",
        "labels",
    ) in backend.model.calls


def test_physical_names_follow_adapter_element_and_node_set_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("namespaces", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].update({(1, 1), (0, 1)})
        curve = cad.entity(1, 1)
        point = cad.entity(0, 1)

        top = cad.physical("SHARED", [surface])
        lower = cad.physical("SHARED", [curve])
        assert top.name == lower.name == "SHARED"
        with pytest.raises(ValueError, match="node-set namespace"):
            cad.physical("SHARED", [point])
        with pytest.raises(ValueError, match="element-set namespace"):
            cad.physical("SHARED", [surface])


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, surface, curve: cad.physical("", [surface]),
        lambda cad, surface, curve: cad.physical("NAME", []),
        lambda cad, surface, curve: cad.physical("NAME", [surface, surface]),
        lambda cad, surface, curve: cad.physical("NAME", [surface, curve]),
    ],
)
def test_invalid_physical_inputs_fail_before_backend_call(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("labels", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].add((1, 1))
        curve = cad.entity(1, 1)
        before = list(backend.model.calls)
        synchronize_calls = backend.model.occ.synchronize_calls
        with pytest.raises((TypeError, ValueError)):
            operation(cad, surface, curve)
        assert backend.model.calls == before
        assert backend.model.occ.synchronize_calls == synchronize_calls


def test_failed_physical_group_does_not_reserve_name_or_freeze_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("labels", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model.fail_next_physical = True
        with pytest.raises(RuntimeError, match="fake physical failure"):
            cad.physical("DOMAIN", [surface])

        second = cad.rectangle(2, 0, 1, 1)
        group = cad.physical("DOMAIN", [surface, second])
        assert group.name == "DOMAIN"


def test_additional_physical_groups_are_allowed_after_topology_freezes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("labels", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].update({(1, 1), (1, 2)})
        first = cad.entity(1, 1)
        second = cad.entity(1, 2)
        cad.physical("DOMAIN", [surface])
        cad.physical("LEFT", [first])
        cad.physical("RIGHT", [second])

    physical_calls = [
        call for call in backend.model.calls if call[0] == "addPhysicalGroup"
    ]
    assert [call[4] for call in physical_calls] == ["DOMAIN", "LEFT", "RIGHT"]


def test_transfinite_curve_and_recombine_forward_typed_targets_in_both_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("controls", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].add((1, 7))
        curve = cad.entity(1, 7)

        assert cad.transfinite_curve(curve, num_nodes=np.int64(5)) is None
        assert cad.recombine(surface) is None
        cad.physical("DOMAIN", [surface])
        assert cad.transfinite_curve(curve, num_nodes=3) is None
        assert cad.recombine(surface) is None
        cad.physical("EDGE", [curve])

    assert backend.model.mesh.calls == [
        ("setTransfiniteCurve", 7, 5, "controls"),
        ("setRecombine", 2, 1, "controls"),
        ("setTransfiniteCurve", 7, 3, "controls"),
        ("setRecombine", 2, 1, "controls"),
    ]
    assert backend.option.calls == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, surface, curve: cad.transfinite_curve(
            surface, num_nodes=3
        ),
        lambda cad, surface, curve: cad.transfinite_curve(1, num_nodes=3),
        lambda cad, surface, curve: cad.transfinite_curve(
            curve, num_nodes=True
        ),
        lambda cad, surface, curve: cad.transfinite_curve(curve, num_nodes=1),
        lambda cad, surface, curve: cad.transfinite_curve(
            curve, num_nodes=2.5
        ),
        lambda cad, surface, curve: cad.recombine(curve),
        lambda cad, surface, curve: cad.recombine((2, 1)),
    ],
)
def test_invalid_curve_and_recombine_controls_fail_before_backend_mutation(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-controls", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].add((1, 1))
        curve = cad.entity(1, 1)
        synchronize_calls = backend.model.occ.synchronize_calls

        with pytest.raises((TypeError, ValueError)):
            operation(cad, surface, curve)

        assert backend.model.mesh.calls == []
        assert backend.model.occ.synchronize_calls == synchronize_calls


def test_mesh_controls_reject_cross_model_and_stale_targets_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-controls", dimension=2) as outer:
        outer_surface = outer.rectangle(0, 0, 1, 1)
        with geometry.model("inner-controls", dimension=2) as inner:
            inner_surface = inner.rectangle(0, 0, 1, 1)
            synchronize_calls = backend.model.occ.synchronize_calls
            with pytest.raises(geometry.EntityOwnershipError):
                inner.recombine(outer_surface)
            assert backend.model.occ.synchronize_calls == synchronize_calls

            _ = inner.raw_model
            with pytest.raises(geometry.StaleEntityError):
                inner.recombine(inner_surface)
            assert backend.model.occ.synchronize_calls == synchronize_calls

    assert backend.model.mesh.calls == []


def test_native_mesh_control_failure_preserves_state_and_mesh_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    imported = _FakeImportResult()
    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=lambda **kwargs: imported),
        raising=False,
    )

    with geometry.model("retry-controls", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model.mesh.fail_next.add("setRecombine")
        with pytest.raises(RuntimeError, match="fake setRecombine failure"):
            cad.recombine(surface)

        cad.physical("DOMAIN", [surface])
        assert cad.generate_mesh(recombine=False) is imported


def test_transfinite_surface_forwards_automatic_and_explicit_corners_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("surface-controls", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].update(
            (0, tag) for tag in range(1, 5)
        )
        points = tuple(cad.entity(0, tag) for tag in range(1, 5))
        backend.model.boundary_result = [
            (0, point.tag) for point in points
        ]

        assert cad.transfinite_surface(surface) is None
        assert (
            cad.transfinite_surface(
                surface,
                corners=(points[3], points[1], points[0], points[2]),
            )
            is None
        )
        cad.physical("DOMAIN", [surface])
        assert cad.transfinite_surface(surface, corners=points[:3]) is None
        cad.physical("CORNERS", points)  # References remain live after controls.

    assert backend.model.mesh.calls == [
        ("setTransfiniteSurface", 1, "Left", (), "surface-controls"),
        (
            "setTransfiniteSurface",
            1,
            "Left",
            (4, 2, 1, 3),
            "surface-controls",
        ),
        (
            "setTransfiniteSurface",
            1,
            "Left",
            (1, 2, 3),
            "surface-controls",
        ),
    ]


@pytest.mark.parametrize(
    "corners",
    [
        None,
        lambda points, curve: points[:2],
        lambda points, curve: (*points, points[0]),
        lambda points, curve: (points[0], points[0], points[1]),
        lambda points, curve: (points[0], points[1], curve),
    ],
)
def test_invalid_surface_corner_shape_fails_before_native_control(
    monkeypatch: pytest.MonkeyPatch,
    corners: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-surface-corners", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].update(
            {(0, 1), (0, 2), (0, 3), (0, 4), (1, 1)}
        )
        points = tuple(cad.entity(0, tag) for tag in range(1, 5))
        curve = cad.entity(1, 1)
        supplied = corners(points, curve) if callable(corners) else corners
        synchronize_calls = backend.model.occ.synchronize_calls

        with pytest.raises((TypeError, ValueError)):
            cad.transfinite_surface(surface, corners=supplied)

        assert backend.model.mesh.calls == []
        assert backend.model.occ.synchronize_calls == synchronize_calls


def test_transfinite_surface_rejects_nonboundary_corner_before_native_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("surface-membership", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].update(
            (0, tag) for tag in range(1, 5)
        )
        points = tuple(cad.entity(0, tag) for tag in range(1, 5))
        backend.model.boundary_result = [(0, 1), (0, 2), (0, 3)]

        with pytest.raises(ValueError, match="boundary"):
            cad.transfinite_surface(
                surface,
                corners=(points[0], points[1], points[3]),
            )

    assert backend.model.mesh.calls == []


def test_transfinite_surface_rejects_cross_model_and_stale_corners_pre_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-surface", dimension=2) as outer:
        backend.model._current_data()["entities"].add((0, 1))
        foreign = outer.entity(0, 1)
        with geometry.model("inner-surface", dimension=2) as inner:
            surface = inner.rectangle(0, 0, 1, 1)
            backend.model._current_data()["entities"].update(
                (0, tag) for tag in range(1, 4)
            )
            points = tuple(inner.entity(0, tag) for tag in range(1, 4))
            synchronize_calls = backend.model.occ.synchronize_calls

            with pytest.raises(geometry.EntityOwnershipError):
                inner.transfinite_surface(
                    surface,
                    corners=(points[0], points[1], foreign),
                )
            assert backend.model.occ.synchronize_calls == synchronize_calls

            inner._entity_tokens.pop((0, points[2].tag))
            with pytest.raises(geometry.StaleEntityError):
                inner.transfinite_surface(surface, corners=points)
            assert backend.model.occ.synchronize_calls == synchronize_calls

    assert backend.model.mesh.calls == []


def test_transfinite_volume_forwards_automatic_six_and_eight_corners_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("volume-controls", dimension=3) as cad:
        volume = cad.box(0, 0, 0, 1, 1, 1)
        backend.model._current_data()["entities"].update(
            (0, tag) for tag in range(1, 9)
        )
        points = tuple(cad.entity(0, tag) for tag in range(1, 9))
        backend.model.boundary_result = [
            (0, point.tag) for point in points
        ]

        assert cad.transfinite_volume(volume) is None
        assert (
            cad.transfinite_volume(
                volume,
                corners=(
                    points[5],
                    points[2],
                    points[0],
                    points[4],
                    points[1],
                    points[3],
                ),
            )
            is None
        )
        cad.physical("DOMAIN", [volume])
        assert cad.transfinite_volume(volume, corners=tuple(reversed(points))) is None

    assert backend.model.mesh.calls == [
        ("setTransfiniteVolume", 1, (), "volume-controls"),
        ("setTransfiniteVolume", 1, (6, 3, 1, 5, 2, 4), "volume-controls"),
        ("setTransfiniteVolume", 1, (8, 7, 6, 5, 4, 3, 2, 1), "volume-controls"),
    ]


@pytest.mark.parametrize(
    "corners",
    [
        None,
        lambda points, surface: points[:5],
        lambda points, surface: points[:7],
        lambda points, surface: (*points, points[0]),
        lambda points, surface: (*points[:5], points[0]),
        lambda points, surface: (*points[:5], surface),
    ],
)
def test_invalid_volume_corner_shape_fails_before_native_control(
    monkeypatch: pytest.MonkeyPatch,
    corners: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-volume-corners", dimension=3) as cad:
        volume = cad.box(0, 0, 0, 1, 1, 1)
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].update(
            (0, tag) for tag in range(1, 9)
        )
        points = tuple(cad.entity(0, tag) for tag in range(1, 9))
        supplied = corners(points, surface) if callable(corners) else corners
        synchronize_calls = backend.model.occ.synchronize_calls

        with pytest.raises((TypeError, ValueError)):
            cad.transfinite_volume(volume, corners=supplied)

        assert backend.model.mesh.calls == []
        assert backend.model.occ.synchronize_calls == synchronize_calls


def test_transfinite_volume_requires_3d_facade_and_recursive_boundary_corners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("two-dimensional-volume", dimension=2) as cad:
        backend.model._current_data()["entities"].add((3, 1))
        raw_volume = cad.entity(3, 1)
        synchronize_calls = backend.model.occ.synchronize_calls
        with pytest.raises(ValueError, match="facade dimension"):
            cad.transfinite_volume(raw_volume)
        assert backend.model.occ.synchronize_calls == synchronize_calls

    with geometry.model("volume-membership", dimension=3) as cad:
        volume = cad.box(0, 0, 0, 1, 1, 1)
        backend.model._current_data()["entities"].update(
            (0, tag) for tag in range(1, 9)
        )
        points = tuple(cad.entity(0, tag) for tag in range(1, 9))
        backend.model.boundary_result = [(0, tag) for tag in range(1, 8)]

        with pytest.raises(ValueError, match="boundary"):
            cad.transfinite_volume(volume, corners=points)

    assert backend.model.mesh.calls == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad: cad.transfinite_curve(None, num_nodes=3),
        lambda cad: cad.transfinite_surface(None),
        lambda cad: cad.transfinite_volume(None),
        lambda cad: cad.recombine(None),
        lambda cad: cad.mesh_size([], size=0.1),
        lambda cad: cad.distance_field(),
        lambda cad: cad.threshold_field(
            None,
            size_min=0.1,
            size_max=0.2,
            dist_min=0.0,
            dist_max=1.0,
        ),
        lambda cad: cad.min_field([]),
        lambda cad: cad.background_field(None),
    ],
)
def test_mesh_controls_reject_new_and_closed_states_contextually(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    cad = geometry.GeometryModel("control-states", dimension=3)

    with pytest.raises(geometry.GeometryStateError, match="NEW"):
        operation(cad)
    with cad:
        pass
    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        operation(cad)


def test_mesh_controls_reject_meshed_and_mesh_failed_states_contextually(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    imported = _FakeImportResult()
    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=lambda **kwargs: imported),
        raising=False,
    )
    operations = (
        lambda cad: cad.transfinite_curve(None, num_nodes=3),
        lambda cad: cad.transfinite_surface(None),
        lambda cad: cad.transfinite_volume(None),
        lambda cad: cad.recombine(None),
        lambda cad: cad.mesh_size([], size=0.1),
        lambda cad: cad.distance_field(),
        lambda cad: cad.threshold_field(
            None,
            size_min=0.1,
            size_max=0.2,
            dist_min=0.0,
            dist_max=1.0,
        ),
        lambda cad: cad.min_field([]),
        lambda cad: cad.background_field(None),
    )

    with geometry.model("meshed-controls", dimension=3) as cad:
        cad.box(0, 0, 0, 1, 1, 1)
        cad.generate_mesh()
        for operation in operations:
            with pytest.raises(geometry.GeometryStateError, match="MESHED"):
                operation(cad)

    with geometry.model("failed-controls", dimension=3) as cad:
        cad.box(0, 0, 0, 1, 1, 1)
        backend.model.mesh.fail_generate = True
        with pytest.raises(RuntimeError, match="fake mesh failure"):
            cad.generate_mesh()
        for operation in operations:
            with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
                operation(cad)


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, curve, surface, volume: cad.transfinite_curve(
            surface, num_nodes=3
        ),
        lambda cad, curve, surface, volume: cad.transfinite_surface(curve),
        lambda cad, curve, surface, volume: cad.transfinite_surface(2),
        lambda cad, curve, surface, volume: cad.transfinite_volume(surface),
        lambda cad, curve, surface, volume: cad.transfinite_volume((3, 1)),
        lambda cad, curve, surface, volume: cad.recombine(volume),
    ],
)
def test_every_mesh_control_rejects_invalid_target_before_backend_mutation(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-control-targets", dimension=3) as cad:
        volume = cad.box(0, 0, 0, 1, 1, 1)
        surface = cad.rectangle(2, 0, 1, 1)
        backend.model._current_data()["entities"].add((1, 1))
        curve = cad.entity(1, 1)
        synchronize_calls = backend.model.occ.synchronize_calls

        with pytest.raises((TypeError, ValueError)):
            operation(cad, curve, surface, volume)

        assert backend.model.mesh.calls == []
        assert backend.model.occ.synchronize_calls == synchronize_calls


@pytest.mark.parametrize(
    ("dimension", "operation"),
    [
        (1, lambda cad, target: cad.transfinite_curve(target, num_nodes=3)),
        (2, lambda cad, target: cad.transfinite_surface(target)),
        (3, lambda cad, target: cad.transfinite_volume(target)),
        (2, lambda cad, target: cad.recombine(target)),
    ],
)
def test_every_mesh_control_rejects_foreign_and_stale_targets_pre_mutation(
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-control-target", dimension=3) as outer:
        backend.model._current_data()["entities"].add((dimension, 9))
        foreign = outer.entity(dimension, 9)
        with geometry.model("inner-control-target", dimension=3) as inner:
            backend.model._current_data()["entities"].add((dimension, 9))
            local = inner.entity(dimension, 9)
            synchronize_calls = backend.model.occ.synchronize_calls

            with pytest.raises(geometry.EntityOwnershipError):
                operation(inner, foreign)
            assert backend.model.occ.synchronize_calls == synchronize_calls

            inner._entity_tokens.pop((dimension, 9))
            with pytest.raises(geometry.StaleEntityError):
                operation(inner, local)
            assert backend.model.occ.synchronize_calls == synchronize_calls

    assert backend.model.mesh.calls == []


def test_transfinite_volume_rejects_foreign_and_stale_corner_tokens_pre_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-volume-corner", dimension=3) as outer:
        backend.model._current_data()["entities"].add((0, 1))
        foreign = outer.entity(0, 1)
        with geometry.model("inner-volume-corner", dimension=3) as inner:
            volume = inner.box(0, 0, 0, 1, 1, 1)
            backend.model._current_data()["entities"].update(
                (0, tag) for tag in range(1, 7)
            )
            points = tuple(inner.entity(0, tag) for tag in range(1, 7))
            synchronize_calls = backend.model.occ.synchronize_calls

            with pytest.raises(geometry.EntityOwnershipError):
                inner.transfinite_volume(
                    volume,
                    corners=(*points[:5], foreign),
                )
            assert backend.model.occ.synchronize_calls == synchronize_calls

            inner._entity_tokens.pop((0, points[5].tag))
            with pytest.raises(geometry.StaleEntityError):
                inner.transfinite_volume(volume, corners=points)
            assert backend.model.occ.synchronize_calls == synchronize_calls

    assert backend.model.mesh.calls == []


@pytest.mark.parametrize(
    ("native_operation", "target_name", "operation"),
    [
        (
            "setTransfiniteCurve",
            "curve",
            lambda cad, target: cad.transfinite_curve(target, num_nodes=3),
        ),
        (
            "setTransfiniteSurface",
            "surface",
            lambda cad, target: cad.transfinite_surface(target),
        ),
        (
            "setTransfiniteVolume",
            "volume",
            lambda cad, target: cad.transfinite_volume(target),
        ),
        ("setRecombine", "surface", lambda cad, target: cad.recombine(target)),
    ],
)
def test_native_control_failures_preserve_exception_state_and_generation_attempt(
    monkeypatch: pytest.MonkeyPatch,
    native_operation: str,
    target_name: str,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    imported = _FakeImportResult()
    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=lambda **kwargs: imported),
        raising=False,
    )

    with geometry.model(f"failed-{native_operation}", dimension=3) as cad:
        targets = {
            "volume": cad.box(0, 0, 0, 1, 1, 1),
            "surface": cad.rectangle(2, 0, 1, 1),
        }
        backend.model._current_data()["entities"].add((1, 1))
        targets["curve"] = cad.entity(1, 1)
        target = targets[target_name]
        backend.model.mesh.fail_next.add(native_operation)

        with pytest.raises(RuntimeError, match=f"fake {native_operation} failure"):
            operation(cad, target)

        cad.physical("TARGET", [target])
        assert cad.generate_mesh(recombine=False) is imported


@pytest.mark.parametrize("operation", ["fuse", "cut", "fragment"])
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
            match="entity-dependent mesh control",
        ):
            getattr(cad, operation)(
                [removed],
                [kept],
                remove_objects=True,
                remove_tools=False,
            )

        assert _occ_operation_call_count(backend, operation) == boolean_calls


@pytest.mark.parametrize("control", _ENTITY_DEPENDENT_MESH_CONTROLS)
def test_native_control_failure_does_not_register_entity_dependencies(
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

        translate_calls = _occ_operation_call_count(backend, "translate")
        assert _apply_typed_transform(cad, "translate", target) == (target,)
        assert (
            _occ_operation_call_count(backend, "translate")
            == translate_calls + 1
        )
        backend.model.boundary_result = []
        fuse_calls = _occ_operation_call_count(backend, "fuse")
        assert cad.fuse([target], [unrelated]).outputs
        assert _occ_operation_call_count(backend, "fuse") == fuse_calls + 1


@pytest.mark.parametrize(
    "operation",
    [
        "point",
        "line",
        "rectangle",
        "disk",
        "box",
        "cylinder",
        "plain_extrude",
    ],
)
def test_entity_dependency_guard_allows_additive_topology(
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
            cad.mesh_size([start], size=0.1)
            mutation = {
                "point": lambda: cad.point(2, 0, 0),
                "line": lambda: cad.line(start, end),
            }[operation]
        else:
            surface = cad.rectangle(0, 0, 1, 1)
            point = _fake_entities(cad, backend, 0, 20)[0]
            backend.model.boundary_result = []
            cad.mesh_size([point], size=0.1)
            mutation = {
                "rectangle": lambda: cad.rectangle(4, 0, 1, 1),
                "disk": lambda: cad.disk(4, 0, 1),
                "box": lambda: cad.box(4, 0, 0, 1, 1, 1),
                "cylinder": lambda: cad.cylinder(4, 0, 0, 0, 0, 1, 1),
                "plain_extrude": lambda: cad.extrude([surface], 0, 0, 1),
            }[operation]

        assert mutation() is not None


def test_entity_dependency_guard_allows_multiple_controlled_extrusions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("multiple-controlled-extrusions", dimension=3) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model.boundary_result = []
        backend.model.occ.extrude_result = [(2, 2), (3, 1)]
        first_outputs = cad.extrude(
            [surface],
            0,
            0,
            1,
            num_elements=[2],
            heights=[1.0],
        )
        top = next(entity for entity in first_outputs if entity.dimension == 2)
        backend.model.boundary_result = []
        backend.model.occ.extrude_result = [(2, 3), (3, 2)]

        second_outputs = cad.extrude([top], 0, 0, 1, recombine=True)

        assert {entity.dimension for entity in second_outputs} == {2, 3}
        assert sum(call[0] == "extrude" for call in backend.model.occ.calls) == 2


def test_controlled_extrude_preserves_valid_duplicate_native_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("controlled-extrude-shared-side", dimension=3) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model.boundary_result = []
        backend.model.occ.extrude_result = [(2, 2), (3, 1), (2, 2)]

        outputs = cad.extrude([surface], 0, 0, 1, num_elements=[1])

        assert tuple((item.dimension, item.tag) for item in outputs) == (
            (2, 2),
            (3, 1),
            (2, 2),
        )
        assert outputs[0] == outputs[2]


@pytest.mark.parametrize("operation", ["fuse", "cut", "fragment"])
def test_entity_dependency_guard_allows_non_destructive_booleans(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"non-destructive-{operation}", dimension=2) as cad:
        first = cad.rectangle(0, 0, 1, 1)
        second = cad.rectangle(2, 0, 1, 1)
        backend.model.boundary_result = []
        cad.recombine(first)
        boolean_calls = _occ_operation_call_count(backend, operation)

        result = getattr(cad, operation)(
            [first],
            [second],
            remove_objects=False,
            remove_tools=False,
        )

        assert result.outputs
        assert _occ_operation_call_count(backend, operation) == boolean_calls + 1


@pytest.mark.parametrize("operation", ["fuse", "cut", "fragment"])
@pytest.mark.parametrize(
    "removed_scope",
    ["objects", "tools", "objects_and_tools"],
)
def test_entity_dependency_guard_allows_precisely_unrelated_removal(
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
        cad.recombine(protected)
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

        result = getattr(cad, operation)(
            objects,
            tools,
            remove_objects=remove_objects,
            remove_tools=remove_tools,
        )

        assert result.outputs
        assert _occ_operation_call_count(backend, operation) == boolean_calls + 1


@pytest.mark.parametrize("operation", ["translate", "rotate"])
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
            match="entity-dependent mesh control",
        ):
            _apply_typed_transform(cad, operation, controlled_parent)

        assert _occ_operation_call_count(backend, operation) == transform_calls
        backend.model.boundary_result = []
        assert _apply_typed_transform(cad, operation, unrelated) == (unrelated,)
        assert _occ_operation_call_count(backend, operation) == transform_calls + 1


@pytest.mark.parametrize("operation", ["translate", "rotate"])
def test_distance_source_transform_remains_allowed(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"distance-{operation}", dimension=2) as cad:
        source = cad.rectangle(0, 0, 1, 1)
        backend.model.boundary_result = []
        cad.distance_field(surfaces=[source])
        backend.model.boundary_result = []
        transform_calls = _occ_operation_call_count(backend, operation)

        assert _apply_typed_transform(cad, operation, source) == (source,)
        assert _occ_operation_call_count(backend, operation) == transform_calls + 1


@pytest.mark.parametrize(
    "options",
    [
        {"remove_objects": 1, "remove_tools": True},
        {"remove_objects": False, "remove_tools": 1},
    ],
)
def test_guarded_boolean_validates_remove_flags_before_dependency_check(
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any],
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("guarded-invalid-remove-flags", dimension=2) as cad:
        first = cad.rectangle(0, 0, 1, 1)
        second = cad.rectangle(2, 0, 1, 1)
        backend.model.boundary_result = []
        cad.recombine(first)
        fuse_calls = _occ_operation_call_count(backend, "fuse")

        with pytest.raises(TypeError, match="must be a boolean"):
            cad.fuse([first], [second], **options)

        assert _occ_operation_call_count(backend, "fuse") == fuse_calls


def test_entity_dependency_guard_allows_labels_more_controls_and_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    imported = _FakeImportResult()
    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=lambda **kwargs: imported),
        raising=False,
    )

    with geometry.model("guarded-mesh-workflow", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        backend.model.boundary_result = []
        cad.mesh_size([point], size=0.1)

        assert cad.physical("DOMAIN", [surface]).name == "DOMAIN"
        backend.model.boundary_result = []
        assert cad.recombine(surface) is None
        assert cad.generate_mesh() is imported


@pytest.mark.parametrize("raw_access", ["raw_model", "raw_occ"])
def test_raw_access_makes_typed_removal_and_transform_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw_access: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"unknown-after-{raw_access}", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        tool = cad.rectangle(2, 0, 1, 1)
        backend.model.boundary_result = []
        cad.recombine(surface)
        getattr(cad, raw_access)
        reacquired_surface = cad.entity(2, surface.tag)
        reacquired_tool = cad.entity(2, tool.tag)
        backend.model.boundary_result = []
        cut_calls = _occ_operation_call_count(backend, "cut")

        with pytest.raises(
            geometry.GeometryStateError,
            match="dependencies unknown",
        ):
            cad.cut([reacquired_surface], [reacquired_tool])

        assert _occ_operation_call_count(backend, "cut") == cut_calls
        translate_calls = _occ_operation_call_count(backend, "translate")
        with pytest.raises(
            geometry.GeometryStateError,
            match="dependencies unknown",
        ):
            cad.translate([reacquired_surface], 1, 0, 0)

        assert _occ_operation_call_count(backend, "translate") == translate_calls


@pytest.mark.parametrize(
    ("native_result", "error_type", "message"),
    [
        ([(4, 1)], ValueError, "dimension"),
        ([], geometry.GeometryError, "no entities"),
        ([(2, 2)], geometry.GeometryError, "dimension-3"),
    ],
)
def test_malformed_controlled_extrude_makes_dependency_scope_unknown(
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
            cad.extrude([surface], 0, 0, 1, num_elements=[1])
        backend.model.boundary_result = []
        fragment_calls = _occ_operation_call_count(backend, "fragment")
        with pytest.raises(
            geometry.GeometryStateError,
            match="dependencies unknown",
        ):
            cad.fragment([surface], [tool])

        assert _occ_operation_call_count(backend, "fragment") == fragment_calls
        rotate_calls = _occ_operation_call_count(backend, "rotate")
        with pytest.raises(
            geometry.GeometryStateError,
            match="dependencies unknown",
        ):
            cad.rotate([surface], 0, 0, 0, 0, 0, 1, 0.5)

        assert _occ_operation_call_count(backend, "rotate") == rotate_calls


def test_mesh_controls_reactivate_owned_model_and_stay_nested_model_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(
        initialized=True,
        names=("external",),
        current="external",
    )
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-mesh-controls", dimension=3) as outer:
        outer_volume = outer.box(0, 0, 0, 1, 1, 1)
        outer_surface = outer.rectangle(2, 0, 1, 1)
        backend.model._current_data()["entities"].add((1, 1))
        outer_curve = outer.entity(1, 1)
        outer_operations = (
            lambda: outer.transfinite_curve(outer_curve, num_nodes=3),
            lambda: outer.transfinite_surface(outer_surface),
            lambda: outer.transfinite_volume(outer_volume),
            lambda: outer.recombine(outer_surface),
        )
        for operation in outer_operations:
            backend.model.setCurrent("external")
            operation()

        with geometry.model("inner-mesh-controls", dimension=3) as inner:
            inner_volume = inner.box(0, 0, 0, 1, 1, 1)
            inner.transfinite_volume(inner_volume)

        outer.transfinite_volume(outer_volume)

    control_calls = [
        call
        for call in backend.model.mesh.calls
        if call[0]
        in {
            "setTransfiniteCurve",
            "setTransfiniteSurface",
            "setTransfiniteVolume",
            "setRecombine",
        }
    ]
    assert [call[-1] for call in control_calls] == [
        "outer-mesh-controls",
        "outer-mesh-controls",
        "outer-mesh-controls",
        "outer-mesh-controls",
        "inner-mesh-controls",
        "outer-mesh-controls",
    ]
    assert backend.model.current == "external"


def test_mesh_size_forwards_ordered_batches_in_building_and_labeled_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("point-sizes", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        point_3, point_1 = _fake_entities(cad, backend, 0, 3, 1)

        synchronize_calls = backend.model.occ.synchronize_calls
        assert (
            cad.mesh_size(
                (point for point in (point_3, point_1)),
                size=np.float64(0.2),
            )
            is None
        )
        assert backend.model.occ.synchronize_calls == synchronize_calls + 1

        cad.physical("DOMAIN", [surface])
        synchronize_calls = backend.model.occ.synchronize_calls
        assert cad.mesh_size([point_1], size=0.025) is None
        assert backend.model.occ.synchronize_calls == synchronize_calls + 1
        cad.physical("SIZED_POINTS", [point_3, point_1])

    assert backend.model.mesh.calls == [
        ("setSize", ((0, 3), (0, 1)), 0.2, "point-sizes"),
        ("setSize", ((0, 1),), 0.025, "point-sizes"),
    ]
    assert backend.model.mesh.field.calls == []
    assert backend.option.calls == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, point, surface: cad.mesh_size([], size=0.1),
        lambda cad, point, surface: cad.mesh_size([point, point], size=0.1),
        lambda cad, point, surface: cad.mesh_size([surface], size=0.1),
        lambda cad, point, surface: cad.mesh_size([object()], size=0.1),
        lambda cad, point, surface: cad.mesh_size([point], size=True),
        lambda cad, point, surface: cad.mesh_size([point], size=0.0),
        lambda cad, point, surface: cad.mesh_size([point], size=-0.1),
        lambda cad, point, surface: cad.mesh_size([point], size=float("inf")),
        lambda cad, point, surface: cad.mesh_size([point], size=float("nan")),
    ],
)
def test_invalid_mesh_size_inputs_fail_before_synchronization_or_mutation(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-point-sizes", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        synchronize_calls = backend.model.occ.synchronize_calls

        with pytest.raises((TypeError, ValueError)):
            operation(cad, point, surface)

        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert backend.model.mesh.calls == []


def test_mesh_size_materializes_generators_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("point-size-generator", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]

        def broken_points():
            yield point
            raise RuntimeError("point generator failed")

        synchronize_calls = backend.model.occ.synchronize_calls
        with pytest.raises(RuntimeError, match="point generator failed"):
            cad.mesh_size(broken_points(), size=0.1)
        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert backend.model.mesh.calls == []


def test_mesh_size_rejects_foreign_and_stale_points_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-point-size", dimension=2) as outer:
        foreign = _fake_entities(outer, backend, 0, 1)[0]
        with geometry.model("inner-point-size", dimension=2) as inner:
            local = _fake_entities(inner, backend, 0, 1)[0]
            synchronize_calls = backend.model.occ.synchronize_calls

            with pytest.raises(geometry.EntityOwnershipError):
                inner.mesh_size([foreign], size=0.1)
            assert backend.model.occ.synchronize_calls == synchronize_calls

            inner._entity_tokens.pop((0, local.tag))
            with pytest.raises(geometry.StaleEntityError):
                inner.mesh_size([local], size=0.1)
            assert backend.model.occ.synchronize_calls == synchronize_calls

    assert backend.model.mesh.calls == []


@pytest.mark.parametrize("source_case", ["points", "curves", "surfaces", "mixed"])
def test_distance_field_forwards_dimension_specific_lists_in_source_order(
    monkeypatch: pytest.MonkeyPatch,
    source_case: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"distance-{source_case}", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        point_2, point_1 = _fake_entities(cad, backend, 0, 2, 1)
        curve_4, curve_3 = _fake_entities(cad, backend, 1, 4, 3)
        kwargs: dict[str, Any]
        expected_options: list[tuple[Any, ...]]
        if source_case == "points":
            kwargs = {"points": (point for point in (point_2, point_1))}
            expected_options = [("setNumbers", 1, "PointsList", (2, 1))]
        elif source_case == "curves":
            kwargs = {"curves": [curve_4, curve_3]}
            expected_options = [("setNumbers", 1, "CurvesList", (4, 3))]
        elif source_case == "surfaces":
            kwargs = {"surfaces": [surface]}
            expected_options = [("setNumbers", 1, "SurfacesList", (1,))]
        else:
            cad.physical("DOMAIN", [surface])
            kwargs = {
                "points": [point_2, point_1],
                "curves": (curve for curve in (curve_4, curve_3)),
                "surfaces": [surface],
                "sampling": np.int64(40),
            }
            expected_options = [
                ("setNumbers", 1, "PointsList", (2, 1)),
                ("setNumbers", 1, "CurvesList", (4, 3)),
                ("setNumbers", 1, "SurfacesList", (1,)),
            ]

        synchronize_calls = backend.model.occ.synchronize_calls
        distance = cad.distance_field(**kwargs)
        assert backend.model.occ.synchronize_calls == synchronize_calls + 1
        assert (distance.tag, distance.field_type) == (1, "Distance")
        cad.physical("SOURCES", [point_2, point_1])

    model_name = f"distance-{source_case}"
    sampling = 40.0 if source_case == "mixed" else 20.0
    assert backend.model.mesh.field.calls == [
        ("add", "Distance", -1, model_name),
        *(option + (model_name,) for option in expected_options),
        ("setNumber", 1, "Sampling", sampling, model_name),
        ("list", model_name),
    ]
    assert backend.option.calls == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, point, curve, surface: cad.distance_field(),
        lambda cad, point, curve, surface: cad.distance_field(
            points=[point, point]
        ),
        lambda cad, point, curve, surface: cad.distance_field(points=[curve]),
        lambda cad, point, curve, surface: cad.distance_field(curves=[point]),
        lambda cad, point, curve, surface: cad.distance_field(surfaces=[curve]),
        lambda cad, point, curve, surface: cad.distance_field(points=[object()]),
        lambda cad, point, curve, surface: cad.distance_field(
            points=[point], sampling=True
        ),
        lambda cad, point, curve, surface: cad.distance_field(
            points=[point], sampling=1
        ),
        lambda cad, point, curve, surface: cad.distance_field(
            points=[point], sampling=2.5
        ),
    ],
)
def test_invalid_distance_field_inputs_fail_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-distance", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        curve = _fake_entities(cad, backend, 1, 1)[0]
        synchronize_calls = backend.model.occ.synchronize_calls

        with pytest.raises((TypeError, ValueError)):
            operation(cad, point, curve, surface)

        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert backend.model.mesh.field.calls == []


def test_distance_field_materializes_every_source_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("distance-generator", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]

        def broken_curves():
            raise RuntimeError("curve generator failed")
            yield point

        synchronize_calls = backend.model.occ.synchronize_calls
        with pytest.raises(RuntimeError, match="curve generator failed"):
            cad.distance_field(points=[point], curves=broken_curves())
        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert backend.model.mesh.field.calls == []


def test_threshold_min_and_background_build_an_ordered_inert_field_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("field-graph", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        curve = _fake_entities(cad, backend, 1, 7)[0]
        distance = cad.distance_field(curves=[curve], sampling=30)
        first = _fake_threshold(cad, distance)
        assert backend.model._current_data()["background_mesh_field"] is None

        cad.physical("DOMAIN", [surface])
        second = _fake_threshold(
            cad,
            distance,
            size_min=0.02,
            size_max=0.3,
            dist_min=0.05,
            dist_max=0.6,
        )
        minimum = cad.min_field((field for field in (second, first)))
        nested = cad.min_field([minimum, second])
        assert cad.background_field(nested) is None
        cad.physical("SOURCE", [curve])

        fields = backend.model._current_data()["mesh_fields"]
        assert fields[distance.tag]["type"] == "Distance"
        assert fields[first.tag]["numbers"] == {
            "InField": 1.0,
            "SizeMin": 0.05,
            "SizeMax": 0.4,
            "DistMin": 0.1,
            "DistMax": 0.8,
        }
        assert fields[second.tag]["numbers"] == {
            "InField": 1.0,
            "SizeMin": 0.02,
            "SizeMax": 0.3,
            "DistMin": 0.05,
            "DistMax": 0.6,
        }
        assert fields[minimum.tag]["number_lists"] == {
            "FieldsList": (second.tag, first.tag)
        }
        assert fields[nested.tag]["number_lists"] == {
            "FieldsList": (minimum.tag, second.tag)
        }
        assert backend.model._current_data()["background_mesh_field"] == nested.tag

    assert [
        call[:4]
        for call in backend.model.mesh.field.calls
        if call[0] == "setNumbers" and call[2] == "FieldsList"
    ] == [
        ("setNumbers", minimum.tag, "FieldsList", (second.tag, first.tag)),
        ("setNumbers", nested.tag, "FieldsList", (minimum.tag, second.tag)),
    ]
    assert [
        call[:4]
        for call in backend.model.mesh.field.calls
        if call[0] == "setNumber" and call[2] != "Sampling"
    ] == [
        ("setNumber", first.tag, "InField", float(distance.tag)),
        ("setNumber", first.tag, "SizeMin", 0.05),
        ("setNumber", first.tag, "SizeMax", 0.4),
        ("setNumber", first.tag, "DistMin", 0.1),
        ("setNumber", first.tag, "DistMax", 0.8),
        ("setNumber", second.tag, "InField", float(distance.tag)),
        ("setNumber", second.tag, "SizeMin", 0.02),
        ("setNumber", second.tag, "SizeMax", 0.3),
        ("setNumber", second.tag, "DistMin", 0.05),
        ("setNumber", second.tag, "DistMax", 0.6),
    ]
    assert backend.option.calls == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, distance, threshold: cad.threshold_field(
            object(),
            size_min=0.1,
            size_max=0.2,
            dist_min=0.0,
            dist_max=1.0,
        ),
        lambda cad, distance, threshold: _fake_threshold(cad, threshold),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, size_min=True
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, size_min=0.0
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, size_min=float("nan")
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, size_max=float("inf")
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, size_min=0.4, size_max=0.4
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, size_min=0.5, size_max=0.4
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, dist_min=-0.1
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, dist_min=float("nan")
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, dist_min=0.2, dist_max=0.2
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, dist_min=0.2, dist_max=0.1
        ),
        lambda cad, distance, threshold: _fake_threshold(
            cad, distance, dist_max=float("inf")
        ),
    ],
)
def test_invalid_threshold_inputs_fail_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-threshold", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = cad.distance_field(points=[point])
        threshold = _fake_threshold(cad, distance)
        calls = list(backend.model.mesh.field.calls)

        with pytest.raises((TypeError, ValueError)):
            operation(cad, distance, threshold)

        assert backend.model.mesh.field.calls == calls


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, distance, first, second: cad.min_field([]),
        lambda cad, distance, first, second: cad.min_field([first]),
        lambda cad, distance, first, second: cad.min_field([first, first]),
        lambda cad, distance, first, second: cad.min_field([distance, first]),
        lambda cad, distance, first, second: cad.min_field([object(), first]),
    ],
)
def test_invalid_min_inputs_fail_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-min", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = cad.distance_field(points=[point])
        first = _fake_threshold(cad, distance)
        second = _fake_threshold(
            cad,
            distance,
            size_min=0.02,
            size_max=0.3,
        )
        calls = list(backend.model.mesh.field.calls)

        with pytest.raises((TypeError, ValueError)):
            operation(cad, distance, first, second)

        assert backend.model.mesh.field.calls == calls


def test_distance_sources_reject_foreign_and_stale_entities_pre_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-distance-source", dimension=2) as outer:
        foreign = _fake_entities(outer, backend, 0, 1)[0]
        with geometry.model("inner-distance-source", dimension=2) as inner:
            local = _fake_entities(inner, backend, 0, 1)[0]
            synchronize_calls = backend.model.occ.synchronize_calls

            with pytest.raises(geometry.EntityOwnershipError):
                inner.distance_field(points=[foreign])
            assert backend.model.occ.synchronize_calls == synchronize_calls

            inner._entity_tokens.pop((0, local.tag))
            with pytest.raises(geometry.StaleEntityError):
                inner.distance_field(points=[local])
            assert backend.model.occ.synchronize_calls == synchronize_calls

    assert backend.model.mesh.field.calls == []


def test_mesh_fields_are_owned_live_model_local_and_use_fresh_tokens_on_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(
        initialized=True,
        names=("external",),
        current="external",
    )
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-fields", dimension=2) as outer:
        outer_point = _fake_entities(outer, backend, 0, 1)[0]
        outer_distance = outer.distance_field(points=[outer_point])
        outer_threshold = _fake_threshold(outer, outer_distance)
        with geometry.model("inner-fields", dimension=2) as inner:
            inner_point = _fake_entities(inner, backend, 0, 1)[0]
            inner_distance = inner.distance_field(points=[inner_point])
            assert outer_distance.tag == inner_distance.tag == 1

            calls = list(backend.model.mesh.field.calls)
            with pytest.raises(geometry.MeshFieldOwnershipError):
                _fake_threshold(inner, outer_distance)
            assert backend.model.mesh.field.calls == calls

            backend.model.mesh.field.remove(inner_distance.tag)
            with pytest.raises(geometry.StaleMeshFieldError, match="no longer exists"):
                _fake_threshold(inner, inner_distance)

            replacement = inner.distance_field(points=[inner_point])
            assert replacement.tag == inner_distance.tag
            assert replacement != inner_distance
            with pytest.raises(geometry.StaleMeshFieldError, match="stale"):
                _fake_threshold(inner, inner_distance)

            threshold = _fake_threshold(inner, replacement)
            assert threshold.field_type == "Threshold"
            calls = list(backend.model.mesh.field.calls)
            with pytest.raises(geometry.MeshFieldOwnershipError):
                inner.background_field(outer_threshold)
            with pytest.raises(geometry.MeshFieldOwnershipError):
                inner.min_field([threshold, outer_threshold])
            assert backend.model.mesh.field.calls == calls

            _ = inner.raw_model
            with pytest.raises(geometry.StaleMeshFieldError):
                inner.background_field(threshold)

        assert outer.background_field(outer_threshold) is None

    assert any(
        call[0] == "add" and call[-1] == "outer-fields"
        for call in backend.model.mesh.field.calls
    )
    assert any(
        call[0] == "add" and call[-1] == "inner-fields"
        for call in backend.model.mesh.field.calls
    )
    assert backend.model.current == "external"


@pytest.mark.parametrize(
    ("native_operation", "option"),
    [
        ("setNumbers", "PointsList"),
        ("setNumbers", "CurvesList"),
        ("setNumbers", "SurfacesList"),
        ("setNumber", "Sampling"),
        ("list", None),
    ],
)
def test_distance_field_rolls_back_after_each_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
    native_operation: str,
    option: str | None,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("distance-rollback", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        curve = _fake_entities(cad, backend, 1, 1)[0]
        backend.model.mesh.field.fail_next.add((native_operation, option))

        with pytest.raises(RuntimeError, match=f"fake field {native_operation}"):
            cad.distance_field(
                points=[point],
                curves=[curve],
                surfaces=[surface],
            )

        assert backend.model._current_data()["mesh_fields"] == {}
        assert backend.model.mesh.field.calls[-1] == (
            "remove",
            1,
            "distance-rollback",
        )


@pytest.mark.parametrize(
    "option",
    ["InField", "SizeMin", "SizeMax", "DistMin", "DistMax", None],
)
def test_threshold_field_rolls_back_after_each_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
    option: str | None,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("threshold-rollback", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = cad.distance_field(points=[point])
        if option is None:
            backend.model.mesh.field.fail_after[("list", None)] = 1
            expected = "list"
        else:
            backend.model.mesh.field.fail_next.add(("setNumber", option))
            expected = "setNumber"

        with pytest.raises(RuntimeError, match=f"fake field {expected}"):
            _fake_threshold(cad, distance)

        assert set(backend.model._current_data()["mesh_fields"]) == {distance.tag}
        assert backend.model.mesh.field.calls[-1] == (
            "remove",
            2,
            "threshold-rollback",
        )


@pytest.mark.parametrize("failure", ["FieldsList", "list"])
def test_min_field_rolls_back_after_each_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("min-rollback", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = cad.distance_field(points=[point])
        first = _fake_threshold(cad, distance)
        second = _fake_threshold(
            cad,
            distance,
            size_min=0.02,
            size_max=0.3,
        )
        if failure == "list":
            backend.model.mesh.field.fail_after[("list", None)] = 1
        else:
            backend.model.mesh.field.fail_next.add(("setNumbers", "FieldsList"))

        with pytest.raises(RuntimeError, match=f"fake field .*{failure}"):
            cad.min_field([second, first])

        assert set(backend.model._current_data()["mesh_fields"]) == {
            distance.tag,
            first.tag,
            second.tag,
        }
        assert backend.model.mesh.field.calls[-1] == (
            "remove",
            4,
            "min-rollback",
        )


def test_field_constructor_rejects_invalid_or_inactive_allocated_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-field-tag", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        backend.model.mesh.field.add_results.append(0)
        with pytest.raises(ValueError, match="positive"):
            cad.distance_field(points=[point])
        assert backend.model._current_data()["mesh_fields"] == {}

        backend.model.mesh.field.hidden_tags.add(1)
        with pytest.raises(geometry.GeometryError, match="not active"):
            cad.distance_field(points=[point])
        assert backend.model._current_data()["mesh_fields"] == {}

        live = cad.distance_field(points=[point])
        assert (live.tag, live.field_type) == (1, "Distance")


def test_field_rollback_failure_preserves_primary_error_note_and_mesh_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    imported = _FakeImportResult()
    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=lambda **kwargs: imported),
        raising=False,
    )

    with geometry.model("rollback-note", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        backend.model.mesh.field.fail_next.update(
            {("setNumber", "Sampling"), ("remove", None)}
        )

        with pytest.raises(RuntimeError, match="setNumber") as captured:
            cad.distance_field(points=[point])
        assert any(
            "rollback" in note and "remove" in note
            for note in getattr(captured.value, "__notes__", ())
        )
        assert set(backend.model._current_data()["mesh_fields"]) == {1}

        replacement = cad.distance_field(points=[point])
        assert replacement.tag == 2
        assert cad.generate_mesh(size=0.2) is imported


def test_background_selection_failure_is_retryable_and_keeps_fields_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("background-retry", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = cad.distance_field(points=[point])
        threshold = _fake_threshold(cad, distance)
        backend.model.mesh.field.fail_next.add(("setAsBackgroundMesh", None))

        with pytest.raises(RuntimeError, match="setAsBackgroundMesh"):
            cad.background_field(threshold)
        assert backend.model._current_data()["background_mesh_field"] is None
        assert cad.background_field(threshold) is None
        assert backend.model._current_data()["background_mesh_field"] == threshold.tag

    background_calls = [
        call
        for call in backend.model.mesh.field.calls
        if call[0] == "setAsBackgroundMesh"
    ]
    assert background_calls == [
        ("setAsBackgroundMesh", threshold.tag, "background-retry"),
        ("setAsBackgroundMesh", threshold.tag, "background-retry"),
    ]
    assert backend.option.calls == []


def test_background_rejects_non_size_fields_and_repeated_selection_pre_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("invalid-background", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = cad.distance_field(points=[point])
        threshold = _fake_threshold(cad, distance)
        calls = list(backend.model.mesh.field.calls)

        with pytest.raises(TypeError):
            cad.background_field(object())
        with pytest.raises(ValueError, match="Threshold or Min"):
            cad.background_field(distance)
        assert backend.model.mesh.field.calls == calls

        cad.background_field(threshold)
        calls = list(backend.model.mesh.field.calls)
        with pytest.raises(ValueError, match="only once"):
            cad.background_field(threshold)
        assert backend.model.mesh.field.calls == calls


@pytest.mark.parametrize("mode", ["point", "background"])
@pytest.mark.parametrize("generation_operation", ["mesh", "fem"])
def test_typed_size_mode_conflicts_are_retryable_before_native_generation(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    generation_operation: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    imported = _FakeImportResult(object())
    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=lambda **kwargs: imported),
        raising=False,
    )

    with geometry.model(f"{mode}-{generation_operation}-conflict", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = cad.distance_field(points=[point])
        threshold = _fake_threshold(cad, distance)
        if mode == "point":
            cad.mesh_size([point], size=0.1)
            field_calls = list(backend.model.mesh.field.calls)
            with pytest.raises(ValueError, match="point sizes"):
                cad.background_field(threshold)
            assert backend.model.mesh.field.calls == field_calls
        else:
            cad.background_field(threshold)
            mesh_calls = list(backend.model.mesh.calls)
            with pytest.raises(ValueError, match="background field"):
                cad.mesh_size([point], size=0.1)
            assert backend.model.mesh.calls == mesh_calls

        mesh_calls = list(backend.model.mesh.calls)
        option_calls = list(backend.option.calls)
        with pytest.raises(ValueError, match="size cannot be supplied"):
            if generation_operation == "mesh":
                cad.generate_mesh(size=0.2)
            else:
                cad.generate_fem_model("conflict", size=0.2)
        assert backend.model.mesh.calls == mesh_calls
        assert backend.option.calls == option_calls

        assert cad.generate_mesh() is imported


@pytest.mark.parametrize("mode", ["point", "background"])
def test_typed_size_modes_request_and_restore_all_deterministic_options(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    backend = _FakeGmsh()
    original = {
        "Mesh.ElementOrder": 7.0,
        "Mesh.SecondOrderIncomplete": 0.25,
        "Mesh.RecombineAll": 0.75,
        "Mesh.MeshSizeFromPoints": 0.3,
        "Mesh.MeshSizeFromCurvature": 4.0,
        "Mesh.MeshSizeExtendFromBoundary": 0.2,
        "Mesh.MeshSizeMin": 0.03,
        "Mesh.MeshSizeMax": 0.9,
        "Mesh.MeshSizeFactor": 2.5,
    }
    backend.option.values.update(original)
    _install_backend(monkeypatch, backend)
    imported = _FakeImportResult()
    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=lambda **kwargs: imported),
        raising=False,
    )

    with geometry.model(f"{mode}-options", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        if mode == "point":
            cad.mesh_size([point], size=0.1)
        else:
            distance = cad.distance_field(points=[point])
            cad.background_field(_fake_threshold(cad, distance))
        backend.option.calls.clear()

        assert cad.generate_mesh(order=2, recombine=True) is imported

    assert backend.option.values == original
    requested = [
        call for call in backend.option.calls if call[0] == "setNumber"
    ][:9]
    assert requested == [
        ("setNumber", "Mesh.ElementOrder", 2.0),
        ("setNumber", "Mesh.SecondOrderIncomplete", 1.0),
        ("setNumber", "Mesh.RecombineAll", 1.0),
        (
            "setNumber",
            "Mesh.MeshSizeFromPoints",
            1.0 if mode == "point" else 0.0,
        ),
        ("setNumber", "Mesh.MeshSizeFromCurvature", 0.0),
        (
            "setNumber",
            "Mesh.MeshSizeExtendFromBoundary",
            1.0 if mode == "point" else 0.0,
        ),
        ("setNumber", "Mesh.MeshSizeMin", 0.0),
        ("setNumber", "Mesh.MeshSizeMax", 1.0e22),
        ("setNumber", "Mesh.MeshSizeFactor", 1.0),
    ]


@pytest.mark.parametrize(
    ("failure", "mode"),
    [("mesh", "background"), ("adapter", "point"), ("option", "background")],
)
def test_typed_size_generation_failures_restore_every_external_option(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    mode: str,
) -> None:
    backend = _FakeGmsh()
    original = {
        "Mesh.ElementOrder": 4.0,
        "Mesh.SecondOrderIncomplete": 0.5,
        "Mesh.RecombineAll": 0.25,
        "Mesh.MeshSizeFromPoints": 0.3,
        "Mesh.MeshSizeFromCurvature": 2.0,
        "Mesh.MeshSizeExtendFromBoundary": 0.4,
        "Mesh.MeshSizeMin": 0.06,
        "Mesh.MeshSizeMax": 0.8,
        "Mesh.MeshSizeFactor": 1.7,
    }
    backend.option.values.update(original)
    backend.model.mesh.fail_generate = failure == "mesh"
    if failure == "option":
        backend.option.fail_set_names.add("Mesh.MeshSizeMax")
    _install_backend(monkeypatch, backend)

    def from_model(**kwargs: Any) -> _FakeImportResult:
        if failure == "adapter":
            raise RuntimeError("fake adapter failure")
        return _FakeImportResult()

    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=from_model),
        raising=False,
    )

    with geometry.model(f"{mode}-{failure}-restore", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        if mode == "point":
            cad.mesh_size([point], size=0.1)
        else:
            distance = cad.distance_field(points=[point])
            cad.background_field(_fake_threshold(cad, distance))

        expected = {
            "mesh": "fake mesh failure",
            "adapter": "fake adapter failure",
            "option": "Mesh.MeshSizeMax",
        }[failure]
        with pytest.raises(RuntimeError, match=expected):
            cad.generate_mesh(order=2, recombine=True)

        assert backend.option.values == original
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            cad.generate_mesh()


def test_nested_typed_size_modes_keep_fields_model_local_and_restore_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(
        initialized=True,
        names=("external",),
        current="external",
    )
    original = {
        "Mesh.ElementOrder": 3.0,
        "Mesh.SecondOrderIncomplete": 0.2,
        "Mesh.RecombineAll": 0.4,
        "Mesh.MeshSizeFromPoints": 0.6,
        "Mesh.MeshSizeFromCurvature": 2.0,
        "Mesh.MeshSizeExtendFromBoundary": 0.5,
        "Mesh.MeshSizeMin": 0.02,
        "Mesh.MeshSizeMax": 0.9,
        "Mesh.MeshSizeFactor": 1.8,
    }
    backend.option.values.update(original)
    _install_backend(monkeypatch, backend)
    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=lambda **kwargs: _FakeImportResult()),
        raising=False,
    )

    with geometry.model("outer-size-mode", dimension=2) as outer:
        outer.rectangle(0, 0, 1, 1)
        outer_point = _fake_entities(outer, backend, 0, 1)[0]
        distance = outer.distance_field(points=[outer_point])
        threshold = _fake_threshold(outer, distance)
        outer.background_field(threshold)
        assert set(backend.model._current_data()["mesh_fields"]) == {1, 2}

        with geometry.model("inner-size-mode", dimension=2) as inner:
            inner.rectangle(0, 0, 1, 1)
            inner_point = _fake_entities(inner, backend, 0, 1)[0]
            inner.mesh_size([inner_point], size=0.1)
            inner.generate_mesh()
            assert backend.option.values == original

        assert backend.model.current == "outer-size-mode"
        assert set(backend.model._current_data()["mesh_fields"]) == {1, 2}
        outer.generate_mesh()
        assert backend.option.values == original

    assert [
        call for call in backend.model.mesh.calls if call[0] == "generate"
    ] == [
        ("generate", 2, "inner-size-mode"),
        ("generate", 2, "outer-size-mode"),
    ]
    assert backend.model.current == "external"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": 0},
        {"size": float("nan")},
        {"order": True},
        {"order": 3},
        {"recombine": 1},
        {"plane_type": "plane"},
        {"thickness": 0},
        {"z_tolerance": -1},
    ],
)
def test_invalid_mesh_arguments_fail_before_mesh_or_option_mutation(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("mesh", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        with pytest.raises((TypeError, ValueError)):
            cad.generate_mesh(**kwargs)
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []


def test_missing_top_dimensional_entity_is_retryable_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    imported = _FakeImportResult()
    importer_calls: list[dict[str, Any]] = []

    def from_model(**kwargs: Any) -> _FakeImportResult:
        importer_calls.append(kwargs)
        return imported

    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=from_model),
        raising=False,
    )

    with geometry.model("mesh", dimension=2) as cad:
        with pytest.raises(ValueError, match="top-dimensional"):
            cad.generate_mesh()
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []

        cad.rectangle(0, 0, 1, 1)
        assert cad.generate_mesh() is imported

    assert len(importer_calls) == 1


def test_generate_mesh_assigns_size_isolates_options_and_delegates_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(
        {
            "Mesh.ElementOrder": 7.0,
            "Mesh.SecondOrderIncomplete": 0.25,
            "Mesh.RecombineAll": 0.75,
            "Mesh.MeshSizeFromPoints": 0.0,
        }
    )
    _install_backend(monkeypatch, backend)
    imported = _FakeImportResult()
    importer_calls: list[tuple[dict[str, Any], str]] = []

    def from_model(**kwargs: Any) -> _FakeImportResult:
        importer_calls.append((kwargs, backend.model.current))
        return imported

    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=from_model),
        raising=False,
    )

    with geometry.model("mesh", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].update({(0, 3), (0, 1)})
        cad.physical("DOMAIN", [surface])
        result = cad.generate_mesh(
            size=0.2,
            order=2,
            recombine=True,
            plane_type="STRAIN",
            thickness=2,
            z_tolerance=0.1,
        )

        assert result is imported
        with pytest.raises(geometry.GeometryStateError, match="MESHED"):
            cad.generate_mesh()
        with pytest.raises(geometry.GeometryStateError, match="MESHED"):
            cad.entities(2)

    assert backend.model.mesh.calls == [
        ("setSize", ((0, 1), (0, 3)), 0.2, "mesh"),
        ("generate", 2, "mesh"),
    ]
    assert backend.option.values == {
        "Mesh.ElementOrder": 7.0,
        "Mesh.SecondOrderIncomplete": 0.25,
        "Mesh.RecombineAll": 0.75,
        "Mesh.MeshSizeFromPoints": 0.0,
    }
    assert importer_calls == [
        (
            {
                "dimension": 2,
                "gmsh_model": backend.model,
                "line_element_type": None,
                "plane_type": "strain",
                "thickness": 2.0,
                "z_tolerance": 0.1,
            },
            "mesh",
        )
    ]
    set_calls = [call for call in backend.option.calls if call[0] == "setNumber"]
    assert set_calls[:4] == [
        ("setNumber", "Mesh.ElementOrder", 2.0),
        ("setNumber", "Mesh.SecondOrderIncomplete", 1.0),
        ("setNumber", "Mesh.RecombineAll", 1.0),
        ("setNumber", "Mesh.MeshSizeFromPoints", 1.0),
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "line_element_type"),
        ({"line_element_type": "truss2"}, "line_element_type"),
        ({"line_element_type": "Line2"}, "line_element_type"),
        ({"line_element_type": "Truss2", "order": 2}, "order"),
        ({"line_element_type": "Beam2", "recombine": True}, "recombine"),
    ],
)
def test_1d_mesh_contract_is_validated_before_mesh_or_option_mutation(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("members", dimension=1) as cad:
        start = cad.point(0, 0, 0)
        end = cad.point(1, 0, 0)
        cad.line(start, end)
        with pytest.raises(ValueError, match=message):
            cad.generate_mesh(**kwargs)
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []


@pytest.mark.parametrize("dimension", [2, 3])
def test_other_dimensions_reject_line_formulation_before_backend_mutation(
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("part", dimension=dimension) as cad:
        if dimension == 2:
            cad.rectangle(0, 0, 1, 1)
        else:
            cad.box(0, 0, 0, 1, 1, 1)
        with pytest.raises(ValueError, match="line_element_type.*dimension 1"):
            cad.generate_mesh(line_element_type="Truss2")
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []


def test_1d_generate_mesh_forwards_formulation_once_and_restores_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(
        {
            "Mesh.ElementOrder": 7.0,
            "Mesh.SecondOrderIncomplete": 0.25,
            "Mesh.RecombineAll": 0.75,
            "Mesh.MeshSizeFromPoints": 0.0,
        }
    )
    _install_backend(monkeypatch, backend)
    imported = _FakeImportResult()
    importer_calls: list[dict[str, Any]] = []

    def from_model(**kwargs: Any) -> _FakeImportResult:
        importer_calls.append(kwargs)
        return imported

    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=from_model),
        raising=False,
    )

    with geometry.model("members", dimension=1) as cad:
        start = cad.point(0, 0, 1)
        end = cad.point(2, 3, 4)
        member = cad.line(start, end)
        cad.physical("MEMBERS", [member])
        result = cad.generate_mesh(
            size=0.25,
            line_element_type="Beam2",
        )
        assert result is imported

    assert backend.model.mesh.calls == [
        ("setSize", ((0, 1), (0, 2)), 0.25, "members"),
        ("generate", 1, "members"),
    ]
    assert importer_calls == [
        {
            "dimension": 1,
            "gmsh_model": backend.model,
            "line_element_type": "Beam2",
            "plane_type": "stress",
            "thickness": 1.0,
            "z_tolerance": 1.0e-10,
        }
    ]
    assert backend.option.values == {
        "Mesh.ElementOrder": 7.0,
        "Mesh.SecondOrderIncomplete": 0.25,
        "Mesh.RecombineAll": 0.75,
        "Mesh.MeshSizeFromPoints": 0.0,
    }


def test_1d_missing_curve_is_retryable_before_mesh_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    imported = _FakeImportResult()
    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=lambda **kwargs: imported),
        raising=False,
    )

    with geometry.model("members", dimension=1) as cad:
        start = cad.point(0, 0, 0)
        with pytest.raises(ValueError, match="top-dimensional"):
            cad.generate_mesh(line_element_type="Truss2")
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []

        end = cad.point(1, 0, 0)
        cad.line(start, end)
        assert cad.generate_mesh(line_element_type="Truss2") is imported


def test_1d_generate_fem_model_forwards_formulation_and_converts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    sentinel_model = object()
    imported = _FakeImportResult(sentinel_model)
    importer_calls: list[dict[str, Any]] = []

    def from_model(**kwargs: Any) -> _FakeImportResult:
        importer_calls.append(kwargs)
        return imported

    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=from_model),
        raising=False,
    )

    with geometry.model("members", dimension=1) as cad:
        start = cad.point(0, 0, 0)
        end = cad.point(1, 0, 0)
        cad.line(start, end)
        result = cad.generate_fem_model(
            "beam",
            line_element_type="Beam2",
        )

    assert result is sentinel_model
    assert imported.to_fem_model_calls == ["beam"]
    assert [call["line_element_type"] for call in importer_calls] == ["Beam2"]


def test_generate_fem_model_delegates_to_import_result_and_is_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    sentinel_model = object()
    imported = _FakeImportResult(sentinel_model)
    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=lambda **kwargs: imported),
        raising=False,
    )

    with geometry.model("mesh", dimension=3) as cad:
        cad.box(0, 0, 0, 1, 1, 1)
        assert cad.generate_fem_model("solid", order=1) is sentinel_model
        with pytest.raises(geometry.GeometryStateError, match="MESHED"):
            cad.generate_mesh()

    assert imported.to_fem_model_calls == ["solid"]
    assert all(
        call[1] != "Mesh.MeshSizeFromPoints"
        for call in backend.option.calls
    )


def test_generate_fem_model_state_errors_name_the_public_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    imported = _FakeImportResult(object())
    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=lambda **kwargs: imported),
        raising=False,
    )
    cad = geometry.model("fem-state", dimension=2)

    with pytest.raises(
        geometry.GeometryStateError,
        match="fem-state.*generate_fem_model",
    ):
        cad.generate_fem_model()
    with cad:
        cad.rectangle(0, 0, 1, 1)
        cad.generate_fem_model()
        with pytest.raises(
            geometry.GeometryStateError,
            match="fem-state.*generate_fem_model.*MESHED",
        ):
            cad.generate_fem_model()
    with pytest.raises(
        geometry.GeometryStateError,
        match="fem-state.*generate_fem_model.*CLOSED",
    ):
        cad.generate_fem_model()


def test_generate_fem_model_conversion_failure_marks_mesh_failed_and_allows_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    imported = _FakeImportResult(
        conversion_error=RuntimeError("fake FEM conversion failure")
    )
    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=lambda **kwargs: imported),
        raising=False,
    )

    with geometry.model("fem-conversion", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        with pytest.raises(RuntimeError, match="FEM conversion") as captured:
            cad.generate_fem_model("invalid")

        assert any(
            "FEM model conversion failed" in note
            for note in getattr(captured.value, "__notes__", ())
        )
        assert len(cad.entities(2)) == 1
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            cad.generate_fem_model()
    assert imported.to_fem_model_calls == ["invalid"]


@pytest.mark.parametrize("failure", ["mesh", "adapter"])
def test_failed_generation_restores_options_and_disallows_retry(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(
        {
            "Mesh.ElementOrder": 4.0,
            "Mesh.SecondOrderIncomplete": 0.0,
            "Mesh.RecombineAll": 0.5,
            "Mesh.MeshSizeFromPoints": 0.0,
        }
    )
    backend.model.mesh.fail_generate = failure == "mesh"
    _install_backend(monkeypatch, backend)

    def from_model(**kwargs: Any) -> _FakeImportResult:
        if failure == "adapter":
            raise RuntimeError("fake adapter failure")
        return _FakeImportResult()

    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=from_model),
        raising=False,
    )

    with geometry.model("mesh", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].add((0, 1))
        expected = "fake mesh failure" if failure == "mesh" else "fake adapter failure"
        with pytest.raises(RuntimeError, match=expected):
            cad.generate_mesh(size=0.2, order=2, recombine=True)

        assert backend.option.values == {
            "Mesh.ElementOrder": 4.0,
            "Mesh.SecondOrderIncomplete": 0.0,
            "Mesh.RecombineAll": 0.5,
            "Mesh.MeshSizeFromPoints": 0.0,
        }
        assert len(cad.entities(2)) == 1
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            cad.generate_mesh()
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            cad.physical("DOMAIN", cad.entities(2))


def test_size_assignment_without_points_consumes_mesh_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=lambda **kwargs: _FakeImportResult()),
        raising=False,
    )

    with geometry.model("mesh", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        with pytest.raises(geometry.GeometryError, match="point"):
            cad.generate_mesh(size=0.25)
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            cad.generate_mesh()
    assert backend.option.calls == []


def test_option_set_failure_restores_snapshot_and_marks_mesh_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    original = {
        "Mesh.ElementOrder": 3.0,
        "Mesh.SecondOrderIncomplete": 0.0,
        "Mesh.RecombineAll": 0.0,
    }
    backend.option.values.update(original)
    backend.option.fail_set_names.add("Mesh.RecombineAll")
    _install_backend(monkeypatch, backend)
    monkeypatch.setattr(
        geometry,
        "gmsh_io",
        SimpleNamespace(from_model=lambda **kwargs: _FakeImportResult()),
        raising=False,
    )

    with geometry.model("mesh", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        with pytest.raises(RuntimeError, match="Mesh.RecombineAll"):
            cad.generate_mesh(order=2, recombine=True)
        assert backend.option.values == original
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            cad.generate_mesh()


@pytest.fixture
def real_gmsh() -> Any:
    import gmsh

    owns_session = not gmsh.isInitialized()
    if owns_session:
        gmsh.initialize()
    option_names = (
        "General.Terminal",
        "Mesh.ElementOrder",
        "Mesh.SecondOrderIncomplete",
        "Mesh.RecombineAll",
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature",
        "Mesh.MeshSizeExtendFromBoundary",
        "Mesh.MeshSizeMin",
        "Mesh.MeshSizeMax",
        "Mesh.MeshSizeFactor",
    )
    saved_options = {
        name: gmsh.option.getNumber(name) for name in option_names
    }
    try:
        gmsh.clear()
        gmsh.option.setNumber("General.Terminal", 0)
        yield gmsh
    finally:
        gmsh.clear()
        for name, value in saved_options.items():
            gmsh.option.setNumber(name, value)
        if owns_session:
            gmsh.finalize()


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


def test_real_1d_facade_reuses_shared_point_in_connected_spatial_mesh(
    real_gmsh: Any,
) -> None:
    middle_coordinates = (1.0, 0.5, 0.75)
    with geometry.model("facade_connected_lines", dimension=1) as cad:
        start = cad.point(0.0, 0.0, 0.25)
        middle = cad.point(*middle_coordinates)
        end = cad.point(2.0, -0.5, 1.25)
        first = cad.line(start, middle)
        second = cad.line(middle, end)
        cad.physical("MEMBERS", [first, second])
        cad.physical("FIXED", [start])
        cad.physical("TIP", [end])
        imported = cad.generate_mesh(
            size=0.4,
            line_element_type="Truss2",
        )

    assert isinstance(imported.mesh, Mesh3D)
    assert imported.mesh.dofs_per_node == 3
    assert {element.type for element in imported.mesh.elements} == {"Truss2"}
    assert all(len(element.node_ids) == 2 for element in imported.mesh.elements)
    middle_node = next(
        node
        for node in imported.mesh.nodes
        if (node.x, node.y, node.z) == pytest.approx(middle_coordinates)
    )
    assert sum(
        middle_node.id in element.node_ids for element in imported.mesh.elements
    ) == 2
    assert imported.element_sets["MEMBERS"].element_ids
    assert len(imported.node_sets["FIXED"].node_ids) == 1
    assert len(imported.node_sets["TIP"].node_ids) == 1
    assert imported.edges == {}
    assert imported.surfaces == {}
    _assert_vtk_cell_type(imported.mesh, 3)


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
        cad.physical("MEMBERS", members)
        cad.physical("CENTER", center)
        imported = cad.generate_mesh(
            size=0.4,
            line_element_type="Truss2",
        )

    center_node_id = imported.node_sets["CENTER"].node_ids[0]
    assert sum(
        center_node_id in element.node_ids for element in imported.mesh.elements
    ) == 4
    assert set(imported.element_sets["MEMBERS"].element_ids) == {
        element.id for element in imported.mesh.elements
    }


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
        member = cad.line(start, end)
        cad.physical("MEMBERS", [member])
        cad.physical("FIXED", [start])
        cad.physical("TIP", [end])
        model = cad.generate_fem_model(
            "truss_vertical_slice",
            size=0.5,
            line_element_type="Truss2",
        )

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
        member = cad.line(root, tip)
        cad.physical("MEMBERS", [member])
        cad.physical("FIXED", [root])
        cad.physical("TIP", [tip])
        model = cad.generate_fem_model(
            "beam_vertical_slice",
            size=0.5,
            line_element_type="Beam2",
        )

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


def test_real_facade_rectangle_labels_solve_vtk_and_survive_cleanup(
    real_gmsh: Any,
    tmp_path: Path,
) -> None:
    with geometry.model("facade_rectangle", dimension=2) as cad:
        surface = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        boundary = cad.boundary([surface])
        left = cad.select(boundary, x=0.0)
        right = cad.select(boundary, x=2.0)
        assert left and right
        cad.physical("DOMAIN", [surface])
        cad.physical("LEFT", left)
        cad.physical("RIGHT", right)
        imported = cad.generate_mesh(size=0.35)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 2)

    assert real_gmsh.isInitialized()
    assert "facade_rectangle" not in real_gmsh.model.list()
    assert isinstance(imported.mesh, Mesh2D)
    assert {element.type for element in imported.mesh.elements} == {"Tri3"}
    assert imported.element_sets["DOMAIN"].element_ids
    assert imported.node_sets["LEFT"].node_ids
    assert imported.node_sets["RIGHT"].node_ids
    assert imported.edges["LEFT"].edges
    assert imported.edges["RIGHT"].edges
    _assert_vtk_cell_type(imported.mesh, 5)

    model = imported.to_fem_model("facade_rectangle")
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
        cad.physical("DOMAIN", domain)
        imported = cad.generate_mesh(size=0.25, order=2)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 2)

    assert isinstance(imported.mesh, Mesh2D)
    assert {element.type for element in imported.mesh.elements} == {"Tri6"}
    assert all(len(element.node_ids) == 6 for element in imported.mesh.elements)
    _assert_vtk_cell_type(imported.mesh, 22)


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
            fine = cad.generate_mesh(size=0.1)
            assert real_gmsh.option.getNumber(option_name) == 0.0

        with geometry.model("facade_size_coarse", dimension=2) as cad:
            cad.rectangle(0.0, 0.0, 1.0, 1.0)
            coarse = cad.generate_mesh(size=0.5)
            assert real_gmsh.option.getNumber(option_name) == 0.0
    finally:
        real_gmsh.option.setNumber(option_name, original)

    assert fine.mesh.num_elements > coarse.mesh.num_elements


def test_real_transfinite_line_creates_exact_truss2_mesh(real_gmsh: Any) -> None:
    with geometry.model("facade_transfinite_line", dimension=1) as cad:
        start = cad.point(0.0, 0.0, 0.0)
        end = cad.point(2.0, 0.0, 0.0)
        member = cad.line(start, end)
        cad.physical("MEMBERS", [member])
        cad.physical("ENDS", [start, end])
        cad.transfinite_curve(member, num_nodes=5)
        imported = cad.generate_mesh(line_element_type="Truss2")

    assert isinstance(imported.mesh, Mesh3D)
    assert imported.mesh.num_nodes == 5
    assert imported.mesh.num_elements == 4
    assert {element.type for element in imported.mesh.elements} == {"Truss2"}
    assert len(imported.element_sets["MEMBERS"].element_ids) == 4
    assert len(imported.node_sets["ENDS"].node_ids) == 2
    _assert_vtk_cell_type(imported.mesh, 3)


def test_real_facade_structured_rectangle_creates_quad8(real_gmsh: Any) -> None:
    with geometry.model("facade_quad8", dimension=2) as cad:
        surface = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        curves = cad.boundary([surface])
        for curve in curves:
            cad.transfinite_curve(curve, num_nodes=3)
        cad.transfinite_surface(surface)
        cad.recombine(surface)

        left = cad.select(curves, x=0.0)
        assert len(left) == 1
        cad.physical("DOMAIN", [surface])
        cad.physical("LEFT", left)
        imported = cad.generate_mesh(order=2, recombine=False)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 2)

    assert isinstance(imported.mesh, Mesh2D)
    assert imported.mesh.num_elements == 4
    assert {element.type for element in imported.mesh.elements} == {"Quad8"}
    assert all(len(element.node_ids) == 8 for element in imported.mesh.elements)
    assert len(imported.element_sets["DOMAIN"].element_ids) == 4
    assert imported.node_sets["LEFT"].node_ids
    assert imported.edges["LEFT"].edges
    _assert_vtk_cell_type(imported.mesh, 23)


def test_real_entity_recombine_leaves_unselected_surface_triangular(
    real_gmsh: Any,
) -> None:
    real_gmsh.option.setNumber("Mesh.RecombineAll", 1.0)
    with geometry.model("facade_selective_recombine", dimension=2) as cad:
        structured = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        unselected = cad.rectangle(2.0, 0.0, 1.0, 1.0)
        structured_curves = cad.boundary([structured])
        for curve in structured_curves:
            cad.transfinite_curve(curve, num_nodes=3)
        cad.transfinite_surface(structured)
        cad.recombine(structured)

        structured_left = cad.select(structured_curves, x=0.0)
        assert len(structured_left) == 1
        cad.physical("STRUCTURED", [structured])
        cad.physical("UNSELECTED", [unselected])
        cad.physical("STRUCTURED_LEFT", structured_left)
        imported = cad.generate_mesh(size=0.3, recombine=False)
        assert real_gmsh.option.getNumber("Mesh.RecombineAll") == 1.0

    elements_by_id = {
        element.id: element for element in imported.mesh.elements
    }
    structured_elements = [
        elements_by_id[element_id]
        for element_id in imported.element_sets["STRUCTURED"].element_ids
    ]
    unselected_elements = [
        elements_by_id[element_id]
        for element_id in imported.element_sets["UNSELECTED"].element_ids
    ]
    assert len(structured_elements) == 4
    assert {element.type for element in structured_elements} == {"Quad4"}
    assert unselected_elements
    assert {element.type for element in unselected_elements} == {"Tri3"}
    assert imported.edges["STRUCTURED_LEFT"].edges


def test_real_facade_box_creates_tet10_and_named_surface(real_gmsh: Any) -> None:
    with geometry.model("facade_tet10", dimension=3) as cad:
        volume = cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        faces = cad.boundary([volume])
        left = cad.select(faces, x=0.0)
        assert left
        cad.physical("VOLUME", [volume])
        cad.physical("LEFT", left)
        imported = cad.generate_mesh(size=0.7, order=2)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 3)

    assert isinstance(imported.mesh, Mesh3D)
    assert {element.type for element in imported.mesh.elements} == {"Tet10"}
    assert imported.element_sets["VOLUME"].element_ids
    assert imported.node_sets["LEFT"].node_ids
    assert imported.surfaces["LEFT"].faces
    _assert_vtk_cell_type(imported.mesh, 24)


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
            cad.transfinite_curve(edge, num_nodes=3)
        for face in faces:
            cad.transfinite_surface(face)
            cad.recombine(face)
        cad.transfinite_volume(volume)

        left = cad.select(faces, x=0.0)
        assert len(left) == 1
        cad.physical("VOLUME", [volume])
        cad.physical("LEFT", left)
        imported = cad.generate_mesh(order=2, recombine=False)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 3)

    assert isinstance(imported.mesh, Mesh3D)
    assert imported.mesh.num_elements == 8
    assert {element.type for element in imported.mesh.elements} == {"Hex20"}
    assert all(len(element.node_ids) == 20 for element in imported.mesh.elements)
    assert len(imported.element_sets["VOLUME"].element_ids) == 8
    assert imported.node_sets["LEFT"].node_ids
    assert imported.surfaces["LEFT"].faces
    _assert_vtk_cell_type(imported.mesh, 25)


def test_real_facade_structured_extrusion_creates_hex20(real_gmsh: Any) -> None:
    with geometry.model("facade_hex20", dimension=3) as cad:
        surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        extruded = cad.extrude(
            [surface],
            0.0,
            0.0,
            1.0,
            num_elements=(2,),
            recombine=True,
        )
        volumes = tuple(
            entity for entity in extruded if entity.dimension == 3
        )
        assert len(volumes) == 1
        cad.physical("VOLUME", volumes)
        imported = cad.generate_mesh(size=0.5, order=2, recombine=True)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 3)

    assert isinstance(imported.mesh, Mesh3D)
    assert {element.type for element in imported.mesh.elements} == {"Hex20"}
    assert all(len(element.node_ids) == 20 for element in imported.mesh.elements)
    _assert_vtk_cell_type(imported.mesh, 25)
