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
        outputs = self.extrude_result
        if outputs is None:
            outputs = [entities[0], (entities[0][0] + 1, self._allocate(entities[0][0] + 1))]
        self._model._current_data()["entities"].update(outputs)
        return list(outputs)


class _FakeMesh:
    def __init__(self, model: _FakeModel) -> None:
        self._model = model
        self.generate_calls: list[int] = []
        self.calls: list[tuple[Any, ...]] = []
        self.fail_generate = False
        self.fail_set_size = False

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

    assert result.of_dimension(2) == (surface,)
    assert group.name == "DOMAIN"
    assert "object" not in repr(surface)
    with pytest.raises(FrozenInstanceError):
        surface.tag = 9  # type: ignore[misc]
    with pytest.raises(ValueError, match="dimension"):
        result.of_dimension(4)


@pytest.mark.parametrize("dimension", [0, 1, 4, True, "2", None])
def test_model_rejects_invalid_mesh_dimension(dimension: Any) -> None:
    with pytest.raises(ValueError, match="dimension must be 2 or 3"):
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


def test_real_facade_structured_rectangle_creates_quad8(real_gmsh: Any) -> None:
    with geometry.model("facade_quad8", dimension=2) as cad:
        surface = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        curves = cad.boundary([surface])
        surface_tag = surface.tag
        curve_tags = tuple(curve.tag for curve in curves)

        raw_model = cad.raw_model
        raw_model.occ.synchronize()
        for curve_tag in curve_tags:
            raw_model.mesh.setTransfiniteCurve(curve_tag, 3)
        raw_model.mesh.setTransfiniteSurface(surface_tag)
        raw_model.mesh.setRecombine(2, surface_tag)

        surface = cad.entity(2, surface_tag)
        cad.physical("DOMAIN", [surface])
        imported = cad.generate_mesh(order=2, recombine=True)
        _assert_positive_top_dimensional_jacobians(real_gmsh, 2)

    assert isinstance(imported.mesh, Mesh2D)
    assert {element.type for element in imported.mesh.elements} == {"Quad8"}
    assert all(len(element.node_ids) == 8 for element in imported.mesh.elements)
    _assert_vtk_cell_type(imported.mesh, 23)


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
