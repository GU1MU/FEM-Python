from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError
import inspect
import math
from pathlib import Path
import subprocess
import sys
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


_FAKE_ELEMENT_PROPERTIES = {
    1: ("Line 2", 1, 1, 2, [], 2),
    2: ("Triangle 3", 2, 1, 3, [], 3),
    3: ("Quadrilateral 4", 2, 1, 4, [], 4),
    4: ("Tetrahedron 4", 3, 1, 4, [], 4),
    5: ("Hexahedron 8", 3, 1, 8, [], 8),
    6: ("Prism 6", 3, 1, 6, [], 6),
    7: ("Pyramid 5", 3, 1, 5, [], 5),
    9: ("Triangle 6", 2, 2, 6, [], 3),
    11: ("Tetrahedron 10", 3, 2, 10, [], 4),
    16: ("Quadrilateral 8", 2, 2, 8, [], 4),
    17: ("Hexahedron 20", 3, 2, 20, [], 8),
}


def _mesher(cad: geometry.GeometryModel) -> gmsh_meshing.Mesher:
    bound = getattr(cad, "_test_bound_mesher", None)
    if bound is None:
        bound = gmsh_meshing.Mesher(cad)
        cad._test_bound_mesher = bound
    return bound


def _generate_mesh(
    cad: geometry.GeometryModel,
    **kwargs: Any,
) -> gmsh_meshing.GmshMeshRef:
    spec = gmsh_meshing.MeshSpec(**kwargs)
    return _mesher(cad).generate(spec)


def _generate_auto_mesh(
    cad: geometry.GeometryModel,
    **kwargs: Any,
) -> gmsh_meshing.GmshMeshRef:
    spec = gmsh_meshing.AutoMeshSpec(**kwargs)
    return _mesher(cad).generate(spec)


def _structured_extrude(
    cad: geometry.GeometryModel,
    *args: Any,
    **kwargs: Any,
) -> geometry.FeatureResult:
    return _mesher(cad).structured_extrude(*args, **kwargs)


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
        self.copy_results: dict[int, list[tuple[int, int]]] = {}
        self.copy_register_outputs = True
        self.boolean_register_map_outputs = True
        self.edge_treatment_results: dict[str, list[tuple[int, int]]] = {}
        self.edge_treatment_register_outputs = True
        self.edge_treatment_attach_lower_outputs = True
        self.edge_treatment_preserve_destructive: set[str] = set()
        self.edge_treatment_remove_preserved: set[str] = set()
        self.nonplanar_after: set[str] = set()
        self.extrude_result: list[tuple[int, int]] | None = None
        self.extrude_extra_primary_boundaries: dict[
            tuple[int, int], list[tuple[int, int]]
        ] = {}
        self.extrude_side_contact_indices: dict[
            tuple[int, int], tuple[int, ...]
        ] = {}
        self._extrude_configuration: tuple[
            tuple[tuple[int, int], ...],
            tuple[float, float, float],
            tuple[tuple[int, int], ...],
            tuple[tuple[int, int], ...],
            tuple[tuple[tuple[int, int], ...], ...],
        ] | None = None

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
        while (dimension, tag) in data["entities"]:
            tag += 1
        next_tags[dimension] = tag + 1
        data["entities"].add((dimension, tag))
        return tag

    def _register_pair(self, pair: tuple[int, int]) -> None:
        dimension, tag = pair
        data = self._model._current_data()
        data["entities"].add(pair)
        data["next_tags"][dimension] = max(
            data["next_tags"].get(dimension, 1),
            tag + 1,
        )

    @staticmethod
    def _is_valid_native_pair(pair: Any) -> bool:
        return (
            isinstance(pair, (tuple, list))
            and len(pair) == 2
            and isinstance(pair[0], int)
            and not isinstance(pair[0], bool)
            and 0 <= pair[0] <= 3
            and isinstance(pair[1], int)
            and not isinstance(pair[1], bool)
            and pair[1] > 0
        )

    def _stored_boundary_closure(
        self,
        sources: Sequence[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        data = self._model._current_data()
        closure = set(sources)
        frontier = list(sources)
        while frontier:
            source = frontier.pop()
            for boundary in data["boundaries"].get(source, ()):
                if boundary not in closure:
                    closure.add(boundary)
                    frontier.append(boundary)
        return closure

    def _clone_entity_geometry(
        self,
        source: tuple[int, int],
        copied: tuple[int, int],
        memo: dict[tuple[int, int], tuple[int, int]],
    ) -> None:
        data = self._model._current_data()
        memo[source] = copied
        if source in data["boxes"]:
            self._store_geometry(
                copied,
                data["boxes"][source],
                data["masses"].get(source, 0.0),
                center=data["centers"].get(source),
            )
        copied_boundaries: list[tuple[int, int]] = []
        for boundary in data["boundaries"].get(source, ()):
            copied_boundary = memo.get(boundary)
            if copied_boundary is None:
                copied_boundary = (
                    boundary[0],
                    self._allocate(boundary[0]),
                )
                self._clone_entity_geometry(
                    boundary,
                    copied_boundary,
                    memo,
                )
            copied_boundaries.append(copied_boundary)
        if copied_boundaries:
            data["boundaries"][copied] = copied_boundaries
            data["boundary_priority"].add(copied)

    @staticmethod
    def _transformed_box(
        box: Sequence[float],
        transform: Any,
    ) -> tuple[float, float, float, float, float, float]:
        corners = [
            transform((x, y, z))
            for x in (float(box[0]), float(box[3]))
            for y in (float(box[1]), float(box[4]))
            for z in (float(box[2]), float(box[5]))
        ]
        return (
            *(min(point[axis] for point in corners) for axis in range(3)),
            *(max(point[axis] for point in corners) for axis in range(3)),
        )  # type: ignore[return-value]

    def _transform_stored_geometry(
        self,
        entities: Sequence[tuple[int, int]],
        transform: Any,
        measure_factors: Sequence[float],
    ) -> None:
        data = self._model._current_data()
        for pair in self._stored_boundary_closure(entities):
            box = data["boxes"].get(pair)
            if box is None:
                continue
            data["boxes"][pair] = self._transformed_box(box, transform)
            center = data["centers"].get(pair, self._box_center(box))
            data["centers"][pair] = tuple(transform(center))
            data["masses"][pair] = data["masses"].get(pair, 0.0) * float(
                measure_factors[pair[0]]
            )

    @staticmethod
    def _box_center(
        box: Sequence[float],
    ) -> tuple[float, float, float]:
        return tuple(
            0.5 * (float(box[axis]) + float(box[axis + 3]))
            for axis in range(3)
        )  # type: ignore[return-value]

    @staticmethod
    def _union_boxes(
        boxes: Sequence[Sequence[float]],
    ) -> tuple[float, float, float, float, float, float]:
        return (
            *(min(float(box[axis]) for box in boxes) for axis in range(3)),
            *(max(float(box[axis + 3]) for box in boxes) for axis in range(3)),
        )  # type: ignore[return-value]

    @staticmethod
    def _translated_box(
        box: Sequence[float],
        vector: Sequence[float],
    ) -> tuple[float, float, float, float, float, float]:
        return tuple(
            float(value) + float(vector[axis % 3])
            for axis, value in enumerate(box)
        )  # type: ignore[return-value]

    def _store_geometry(
        self,
        pair: tuple[int, int],
        box: Sequence[float],
        measure: float,
        *,
        center: Sequence[float] | None = None,
    ) -> None:
        data = self._model._current_data()
        normalized_box = tuple(float(value) for value in box)
        data["boxes"][pair] = normalized_box
        data["centers"][pair] = tuple(
            float(value)
            for value in (
                self._box_center(normalized_box) if center is None else center
            )
        )
        data["masses"][pair] = float(measure)

    def _add_boundary_entity(
        self,
        dimension: int,
        box: Sequence[float],
        measure: float,
    ) -> tuple[int, int]:
        pair = (dimension, self._allocate(dimension))
        self._store_geometry(pair, box, measure)
        return pair

    def _rectangle_boundaries(
        self,
        x: float,
        y: float,
        z: float,
        dx: float,
        dy: float,
    ) -> list[tuple[int, int]]:
        return [
            self._add_boundary_entity(1, (x, y, z, x + dx, y, z), dx),
            self._add_boundary_entity(
                1,
                (x + dx, y, z, x + dx, y + dy, z),
                dy,
            ),
            self._add_boundary_entity(
                1,
                (x, y + dy, z, x + dx, y + dy, z),
                dx,
            ),
            self._add_boundary_entity(1, (x, y, z, x, y + dy, z), dy),
        ]

    def addPoint(
        self,
        x: float,
        y: float,
        z: float,
        meshSize: float = 0.0,
        tag: int = -1,
    ) -> int:
        self.calls.append(("addPoint", x, y, z, meshSize, tag))
        allocated = self._allocate(0)
        self._store_geometry((0, allocated), (
            x,
            y,
            z,
            x,
            y,
            z,
        ), 0.0, center=(x, y, z))
        return allocated

    def getBoundingBox(
        self,
        dimension: int,
        tag: int,
    ) -> tuple[float, float, float, float, float, float]:
        return self._model._current_data()["boxes"][(dimension, tag)]

    def getMass(self, dimension: int, tag: int) -> float:
        return self._model._current_data()["masses"][(dimension, tag)]

    def getCenterOfMass(
        self,
        dimension: int,
        tag: int,
    ) -> tuple[float, float, float]:
        return self._model._current_data()["centers"][(dimension, tag)]

    def addLine(self, start_tag: int, end_tag: int, tag: int = -1) -> int:
        self.calls.append(("addLine", start_tag, end_tag, tag))
        allocated = self._allocate(1)
        data = self._model._current_data()
        start_box = data["boxes"][(0, start_tag)]
        end_box = data["boxes"][(0, end_tag)]
        start = self._box_center(start_box)
        end = self._box_center(end_box)
        self._store_geometry(
            (1, allocated),
            self._union_boxes((start_box, end_box)),
            math.dist(start, end),
            center=tuple(
                0.5 * (first + second)
                for first, second in zip(start, end, strict=True)
            ),
        )
        data["boundaries"][(1, allocated)] = [
            (0, start_tag),
            (0, end_tag),
        ]
        return allocated

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
        allocated = self._allocate(2)
        pair = (2, allocated)
        self._store_geometry(pair, (x, y, z, x + dx, y + dy, z), dx * dy)
        self._model._current_data()["boundaries"][pair] = (
            self._rectangle_boundaries(x, y, z, dx, dy)
        )
        return allocated

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
        allocated = self._allocate(2)
        pair = (2, allocated)
        box = (
            x - radius_x,
            y - radius_y,
            z,
            x + radius_x,
            y + radius_y,
            z,
        )
        self._store_geometry(
            pair,
            box,
            math.pi * radius_x * radius_y,
            center=(x, y, z),
        )
        boundary = self._add_boundary_entity(
            1,
            box,
            2.0 * math.pi * max(radius_x, radius_y),
        )
        self._model._current_data()["boundaries"][pair] = [boundary]
        return allocated

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
        allocated = self._allocate(3)
        pair = (3, allocated)
        box = (x, y, z, x + dx, y + dy, z + dz)
        self._store_geometry(pair, box, dx * dy * dz)
        face_areas = (dx * dy, dx * dy, dx * dz, dx * dz, dy * dz, dy * dz)
        boundaries = [
            self._add_boundary_entity(2, box, area)
            for area in face_areas
        ]
        self._model._current_data()["boundaries"][pair] = boundaries
        return allocated

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
        allocated = self._allocate(3)
        pair = (3, allocated)
        end = (x + axis_x, y + axis_y, z + axis_z)
        box = (
            min(x, end[0]) - radius,
            min(y, end[1]) - radius,
            min(z, end[2]) - radius,
            max(x, end[0]) + radius,
            max(y, end[1]) + radius,
            max(z, end[2]) + radius,
        )
        axis_length = math.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
        self._store_geometry(
            pair,
            box,
            math.pi * radius**2 * axis_length * angle / (2.0 * math.pi),
            center=(
                x + 0.5 * axis_x,
                y + 0.5 * axis_y,
                z + 0.5 * axis_z,
            ),
        )
        boundaries = [
            self._add_boundary_entity(2, box, math.pi * radius**2)
            for _ in range(3)
        ]
        self._model._current_data()["boundaries"][pair] = boundaries
        return allocated

    def copy(
        self,
        entities: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        materialized = tuple(entities)
        self.calls.append(("copy", materialized))
        if "copy" in self.fail_next:
            self.fail_next.remove("copy")
            raise RuntimeError("fake copy failure")
        dimension = materialized[0][0]
        configured = self.copy_results.get(dimension)
        if configured is None:
            outputs = [
                (source[0], self._allocate(source[0]))
                for source in materialized
            ]
        else:
            outputs = list(configured)
            if self.copy_register_outputs:
                for output in outputs:
                    self._register_pair(output)

        if self.copy_register_outputs:
            memo: dict[tuple[int, int], tuple[int, int]] = {}
            for source, output in zip(materialized, outputs):
                if source[0] == output[0] and output != source:
                    self._clone_entity_geometry(source, output, memo)
        return outputs

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

    def intersect(
        self,
        objects: list[tuple[int, int]],
        tools: list[tuple[int, int]],
        tag: int = -1,
        removeObject: bool = True,
        removeTool: bool = True,
    ) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
        return self._boolean(
            "intersect", objects, tools, tag, removeObject, removeTool
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
        configured = self.boolean_results.get(name)
        if configured is None:
            outputs = [objects[0]]
            input_map = [
                *[
                    [pair] if not remove_objects else [outputs[0]]
                    for pair in objects
                ],
                *[[pair] if not remove_tools else [] for pair in tools],
            ]
        else:
            outputs, input_map = configured
        entities = self._model._current_data()["entities"]
        if remove_objects:
            entities.difference_update(objects)
        if remove_tools:
            entities.difference_update(tools)
        for pair in outputs:
            if self._is_valid_native_pair(pair):
                self._register_pair(pair)
        input_pairs = {*objects, *tools}
        if self.boolean_register_map_outputs:
            for group in input_map:
                for pair in group:
                    if self._is_valid_native_pair(pair) and (
                        pair not in input_pairs or pair in outputs
                    ):
                        self._register_pair(pair)
        return list(outputs), [list(group) for group in input_map]

    def fillet(
        self,
        volumeTags: list[int],
        curveTags: list[int],
        radii: list[float],
        removeVolume: bool = True,
    ) -> list[tuple[int, int]]:
        return self._edge_treatment(
            "fillet",
            volumeTags,
            curveTags,
            (),
            radii,
            removeVolume,
        )

    def chamfer(
        self,
        volumeTags: list[int],
        curveTags: list[int],
        surfaceTags: list[int],
        distances: list[float],
        removeVolume: bool = True,
    ) -> list[tuple[int, int]]:
        return self._edge_treatment(
            "chamfer",
            volumeTags,
            curveTags,
            surfaceTags,
            distances,
            removeVolume,
        )

    def _edge_treatment(
        self,
        name: str,
        volume_tags: Sequence[int],
        curve_tags: Sequence[int],
        surface_tags: Sequence[int],
        values: Sequence[float],
        remove_volume: bool,
    ) -> list[tuple[int, int]]:
        volumes = tuple(int(tag) for tag in volume_tags)
        curves = tuple(int(tag) for tag in curve_tags)
        surfaces = tuple(int(tag) for tag in surface_tags)
        normalized_values = tuple(float(value) for value in values)
        if name == "fillet":
            self.calls.append(
                (name, volumes, curves, normalized_values, remove_volume)
            )
        else:
            self.calls.append(
                (
                    name,
                    volumes,
                    curves,
                    surfaces,
                    normalized_values,
                    remove_volume,
                )
            )
        if name in self.fail_next:
            self.fail_next.remove(name)
            raise RuntimeError(f"fake {name} failure")

        source_pairs = tuple((3, tag) for tag in volumes)
        configured = self.edge_treatment_results.get(name)
        if configured is None:
            if remove_volume:
                outputs = list(source_pairs)
            else:
                outputs = [(3, self._allocate(3)) for _ in source_pairs]
        else:
            outputs = list(configured)

        data = self._model._current_data()
        if (
            remove_volume and name not in self.edge_treatment_preserve_destructive
        ) or name in self.edge_treatment_remove_preserved:
            data["entities"].difference_update(source_pairs)
        if self.edge_treatment_register_outputs:
            for output in outputs:
                if self._is_valid_native_pair(output):
                    self._register_pair(output)

        memo: dict[tuple[int, int], tuple[int, int]] = {}
        for source, output in zip(source_pairs, outputs):
            if source[0] == output[0] and source != output:
                self._clone_entity_geometry(source, output, memo)
        primary_outputs = [pair for pair in outputs if pair[0] == 3]
        lower_outputs = [pair for pair in outputs if pair[0] < 3]
        if (
            self.edge_treatment_attach_lower_outputs
            and primary_outputs
            and lower_outputs
        ):
            primary = primary_outputs[0]
            data["boundaries"].setdefault(primary, []).extend(lower_outputs)
            data["boundary_priority"].add(primary)
        return outputs

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

    def mirror(
        self,
        entities: list[tuple[int, int]],
        a: float,
        b: float,
        c: float,
        d: float,
    ) -> None:
        materialized = tuple(entities)
        self.calls.append(("mirror", materialized, a, b, c, d))
        if "mirror" in self.fail_next:
            self.fail_next.remove("mirror")
            raise RuntimeError("fake mirror failure")
        denominator = a * a + b * b + c * c

        def reflected(point: Sequence[float]) -> tuple[float, float, float]:
            distance = (
                a * float(point[0])
                + b * float(point[1])
                + c * float(point[2])
                + d
            ) / denominator
            return (
                float(point[0]) - 2.0 * distance * a,
                float(point[1]) - 2.0 * distance * b,
                float(point[2]) - 2.0 * distance * c,
            )

        self._transform_stored_geometry(
            materialized,
            reflected,
            (1.0, 1.0, 1.0, 1.0),
        )
        self._force_nonplanar_if_requested("mirror", materialized)

    def dilate(
        self,
        entities: list[tuple[int, int]],
        x: float,
        y: float,
        z: float,
        factor_x: float,
        factor_y: float,
        factor_z: float,
    ) -> None:
        materialized = tuple(entities)
        self.calls.append(
            (
                "dilate",
                materialized,
                x,
                y,
                z,
                factor_x,
                factor_y,
                factor_z,
            )
        )
        if "dilate" in self.fail_next:
            self.fail_next.remove("dilate")
            raise RuntimeError("fake dilate failure")

        def scaled(point: Sequence[float]) -> tuple[float, float, float]:
            return (
                x + factor_x * (float(point[0]) - x),
                y + factor_y * (float(point[1]) - y),
                z + factor_z * (float(point[2]) - z),
            )

        absolute_factors = (
            abs(factor_x),
            abs(factor_y),
            abs(factor_z),
        )
        self._transform_stored_geometry(
            materialized,
            scaled,
            (
                1.0,
                max(absolute_factors),
                absolute_factors[0] * absolute_factors[1],
                math.prod(absolute_factors),
            ),
        )
        self._force_nonplanar_if_requested("dilate", materialized)

    def _force_nonplanar_if_requested(
        self,
        operation: str,
        entities: Sequence[tuple[int, int]],
    ) -> None:
        if operation not in self.nonplanar_after:
            return
        data = self._model._current_data()
        for pair in entities:
            box = data["boxes"][pair]
            data["boxes"][pair] = (
                box[0],
                box[1],
                -1.0,
                box[3],
                box[4],
                1.0,
            )

    def configure_extrude_result(
        self,
        sources: Sequence[tuple[int, int]],
        vector: Sequence[float],
        outputs: Sequence[tuple[int, int]],
        *,
        ends: Sequence[tuple[int, int]],
        primary: Sequence[tuple[int, int]],
    ) -> None:
        source_pairs = tuple(sources)
        vector_values = tuple(float(value) for value in vector)
        output_pairs = tuple(outputs)
        end_pairs = tuple(ends)
        primary_pairs = tuple(primary)
        if len(vector_values) != 3:
            raise ValueError("fake extrusion vector must have three components")
        if not (
            len(source_pairs) == len(end_pairs) == len(primary_pairs)
        ):
            raise ValueError(
                "fake extrusion sources, ends, and primary entities must align"
            )

        semantic_pairs = {*end_pairs, *primary_pairs}
        unique_sides = tuple(
            dict.fromkeys(pair for pair in output_pairs if pair not in semantic_pairs)
        )
        data = self._model._current_data()
        side_groups: list[tuple[tuple[int, int], ...]] = []
        remaining = list(unique_sides)
        for index, source in enumerate(source_pairs):
            if index == len(source_pairs) - 1:
                count = len(remaining)
            else:
                count = min(
                    len(self._model._boundary_for_pair(source)),
                    len(remaining),
                )
            group = tuple(remaining[:count])
            del remaining[:count]
            if group and not self._model._boundary_for_pair(source):
                source_dimension = source[0]
                if source_dimension == 0:
                    raise ValueError("point extrusions cannot have fake side entities")
                source_box = data["boxes"][source]
                source_measure = data["masses"].get(source, 0.0)
                contact = self._add_boundary_entity(
                    source_dimension - 1,
                    source_box,
                    source_measure,
                )
                data["boundaries"][source] = [contact]
            side_groups.append(group)

        self.extrude_result = list(output_pairs)
        self._extrude_configuration = (
            source_pairs,
            vector_values,  # type: ignore[arg-type]
            end_pairs,
            primary_pairs,
            tuple(side_groups),
        )

    def _seed_extrusion_geometry(
        self,
        sources: Sequence[tuple[int, int]],
        vector: tuple[float, float, float],
        ends: Sequence[tuple[int, int]],
        primary: Sequence[tuple[int, int]],
        side_groups: Sequence[Sequence[tuple[int, int]]],
    ) -> None:
        data = self._model._current_data()
        for source, end, body, sides in zip(
            sources,
            ends,
            primary,
            side_groups,
            strict=True,
        ):
            source_box = data["boxes"][source]
            source_center = data["centers"].get(
                source,
                self._box_center(source_box),
            )
            source_measure = data["masses"].get(source, 0.0)
            end_box = self._translated_box(source_box, vector)
            end_center = tuple(
                float(value) + vector[axis]
                for axis, value in enumerate(source_center)
            )
            self._store_geometry(
                end,
                end_box,
                source_measure,
                center=end_center,
            )

            source_boundaries = list(self._model._boundary_for_pair(source))
            translated_boundaries: list[tuple[int, int]] = []
            for boundary in source_boundaries:
                boundary_box = data["boxes"].get(boundary, source_box)
                boundary_measure = data["masses"].get(
                    boundary,
                    source_measure,
                )
                if boundary not in data["boxes"]:
                    self._register_pair(boundary)
                    self._store_geometry(
                        boundary,
                        boundary_box,
                        boundary_measure,
                    )
                translated_boundary = (
                    boundary[0],
                    self._allocate(boundary[0]),
                )
                self._store_geometry(
                    translated_boundary,
                    self._translated_box(boundary_box, vector),
                    boundary_measure,
                    center=tuple(
                        value + vector[axis]
                        for axis, value in enumerate(
                            data["centers"].get(
                                boundary,
                                self._box_center(boundary_box),
                            )
                        )
                    ),
                )
                translated_boundaries.append(translated_boundary)
            data["boundaries"][end] = translated_boundaries
            data["boundary_priority"].add(end)

            body_boundaries = [source, end]
            configured_contact_indices = self.extrude_side_contact_indices.get(body)
            if configured_contact_indices is not None and len(
                configured_contact_indices
            ) != len(sides):
                raise ValueError(
                    "fake extrusion side-contact indices must align with sides"
                )
            for side_index, side in enumerate(sides):
                if not source_boundaries:
                    raise ValueError(
                        "fake extrusion sides require source boundary topology"
                    )
                contact_index = (
                    configured_contact_indices[side_index]
                    if configured_contact_indices is not None
                    else side_index % len(source_boundaries)
                )
                source_contact = source_boundaries[contact_index]
                end_contact = translated_boundaries[contact_index]
                contact_box = data["boxes"][source_contact]
                translated_contact_box = data["boxes"][end_contact]
                self._store_geometry(
                    side,
                    self._union_boxes((contact_box, translated_contact_box)),
                    max(
                        data["masses"].get(source_contact, 0.0),
                        math.dist((0.0, 0.0, 0.0), vector),
                    ),
                    center=tuple(
                        0.5 * (first + second)
                        for first, second in zip(
                            data["centers"][source_contact],
                            data["centers"][end_contact],
                            strict=True,
                        )
                    ),
                )
                data["boundaries"][side] = [source_contact, end_contact]
                data["boundary_priority"].add(side)
                body_boundaries.append(side)

            for extra_boundary in self.extrude_extra_primary_boundaries.get(
                body,
                (),
            ):
                self._register_pair(extra_boundary)
                if extra_boundary not in data["boxes"]:
                    self._store_geometry(
                        extra_boundary,
                        self._union_boxes((source_box, end_box)),
                        source_measure,
                    )
                body_boundaries.append(extra_boundary)

            self._store_geometry(
                body,
                self._union_boxes((source_box, end_box)),
                source_measure * math.dist((0.0, 0.0, 0.0), vector),
            )
            data["boundaries"][body] = body_boundaries
            data["boundary_priority"].add(body)

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
            generated_outputs: list[tuple[int, int]] = []
            ends: list[tuple[int, int]] = []
            primary: list[tuple[int, int]] = []
            side_groups: list[tuple[tuple[int, int], ...]] = []
            for source in entities:
                end = (source[0], self._allocate(source[0]))
                body = (source[0] + 1, self._allocate(source[0] + 1))
                sides = tuple(
                    (source[0], self._allocate(source[0]))
                    for _ in self._model._boundary_for_pair(source)
                )
                ends.append(end)
                primary.append(body)
                side_groups.append(sides)
                generated_outputs.extend((end, body, *sides))
            outputs = generated_outputs
            self._seed_extrusion_geometry(
                entities,
                (dx, dy, dz),
                ends,
                primary,
                side_groups,
            )
        elif self._extrude_configuration is not None:
            (
                configured_sources,
                configured_vector,
                ends,
                primary,
                side_groups,
            ) = self._extrude_configuration
            if tuple(entities) != configured_sources or (dx, dy, dz) != (
                configured_vector
            ):
                raise RuntimeError("fake extrusion call does not match configuration")
            for pair in outputs:
                self._register_pair(pair)
            self._seed_extrusion_geometry(
                configured_sources,
                configured_vector,
                ends,
                primary,
                side_groups,
            )
        else:
            for pair in outputs:
                self._register_pair(pair)
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
        self.fail_get_elements_dimensions: set[int] = set()
        self.fail_element_properties: set[int] = set()
        self.refine_calls = 0

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

    def getElements(
        self,
        dimension: int = -1,
        tag: int = -1,
    ) -> Any:
        self.calls.append(("getElements", dimension, tag, self._model.current))
        if dimension in self.fail_get_elements_dimensions:
            self.fail_get_elements_dimensions.remove(dimension)
            raise RuntimeError("fake getElements failure")
        blocks = self._model._current_data()["element_blocks"]
        return blocks.get((dimension, tag), blocks.get(dimension, ([], [], [])))

    def getElementProperties(self, element_type: int) -> Any:
        self.calls.append(("getElementProperties", element_type, self._model.current))
        if element_type in self.fail_element_properties:
            self.fail_element_properties.remove(element_type)
            raise RuntimeError(f"fake getElementProperties failure for {element_type}")
        return self._model._current_data()["element_properties"][element_type]

    def refine(self) -> None:
        self.calls.append(("refine", self._model.current))
        self.refine_calls += 1


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
        self._boundary_result: list[tuple[int, int]] = []
        self._boundary_result_overridden = False
        self.fail_remove = False
        self.fail_set_current_names: set[str] = set()

    @property
    def boundary_result(self) -> list[tuple[int, int]]:
        return self._boundary_result

    @boundary_result.setter
    def boundary_result(self, result: Sequence[tuple[int, int]]) -> None:
        self._boundary_result = list(result)
        self._boundary_result_overridden = True

    @staticmethod
    def _new_data() -> dict[str, Any]:
        return {
            "entities": set(),
            "next_tags": {},
            "boxes": {},
            "centers": {},
            "masses": {},
            "boundaries": {},
            "boundary_priority": set(),
            "mesh_fields": {},
            "background_mesh_field": None,
            "element_blocks": {},
            "element_properties": dict(_FAKE_ELEMENT_PROPERTIES),
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
        boundary: list[tuple[int, int]] = []
        for dimension, raw_tag in entities:
            pair = (dimension, abs(raw_tag))
            boundary.extend(self._boundary_for_pair(pair))
        return boundary

    def _boundary_for_pair(
        self,
        pair: tuple[int, int],
    ) -> list[tuple[int, int]]:
        data = self._current_data()
        per_entity = data["boundaries"]
        if pair in data["boundary_priority"]:
            return list(per_entity[pair])
        if self._boundary_result_overridden:
            return list(self._boundary_result)
        if pair in per_entity:
            return list(per_entity[pair])
        return []

    def getAdjacencies(
        self,
        dimension: int,
        tag: int,
    ) -> tuple[list[int], list[int]]:
        self.calls.append(("getAdjacencies", dimension, tag, self.current))
        pair = (dimension, tag)
        data = self._current_data()
        upward = sorted(
            source_tag
            for (source_dimension, source_tag), boundaries in data[
                "boundaries"
            ].items()
            if source_dimension == dimension + 1 and pair in boundaries
        )
        downward = sorted(
            boundary_tag
            for boundary_dimension, boundary_tag in self._boundary_for_pair(pair)
            if boundary_dimension == dimension - 1
        )
        return upward, downward

    def getBoundingBox(
        self,
        dimension: int,
        tag: int,
    ) -> tuple[float, float, float, float, float, float]:
        self.calls.append(("getBoundingBox", dimension, tag, self.current))
        return self._current_data()["boxes"][(dimension, tag)]

class _FakeOption:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.calls: list[tuple[Any, ...]] = []
        self.fail_get_names: set[str] = set()
        self.fail_set_names: set[str] = set()
        self.fail_set_after: dict[str, int] = {}

    def getNumber(self, name: str) -> float:
        self.calls.append(("getNumber", name))
        if name in self.fail_get_names:
            self.fail_get_names.remove(name)
            raise RuntimeError(f"fake option get failure for {name}")
        return self.values.get(name, 0.0)

    def setNumber(self, name: str, value: float) -> None:
        self.calls.append(("setNumber", name, float(value)))
        if name in self.fail_set_after:
            remaining = self.fail_set_after[name]
            if remaining == 0:
                del self.fail_set_after[name]
                raise RuntimeError(f"fake option failure for {name}")
            self.fail_set_after[name] = remaining - 1
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
        self.fail_is_initialized_count = 0
        self.fail_finalize = False

    def isInitialized(self) -> bool:
        if self.fail_is_initialized_count:
            self.fail_is_initialized_count -= 1
            raise RuntimeError("fake session inspection failure")
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


def _install_backend(monkeypatch: pytest.MonkeyPatch, backend: _FakeGmsh) -> None:
    monkeypatch.setattr(_gmsh_backend, "load_gmsh", lambda: backend)


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


def _fake_edge_treatment_topology(
    cad: geometry.GeometryModel,
    backend: _FakeGmsh,
) -> dict[str, geometry.EntityRef]:
    volume, unrelated_volume, outside_volume = _fake_entities(
        cad,
        backend,
        3,
        70,
        71,
        72,
    )
    surface, nonadjacent_surface, unrelated_surface, outside_surface = (
        _fake_entities(cad, backend, 2, 80, 81, 82, 83)
    )
    curve, other_curve, unrelated_curve = _fake_entities(
        cad,
        backend,
        1,
        90,
        91,
        92,
    )
    start, end, other_start, other_end, unrelated_start, unrelated_end = (
        _fake_entities(cad, backend, 0, 100, 101, 102, 103, 104, 105)
    )

    data = backend.model._current_data()
    boundaries = {
        (3, volume.tag): [
            (2, surface.tag),
            (2, nonadjacent_surface.tag),
        ],
        (2, surface.tag): [(1, curve.tag)],
        (2, nonadjacent_surface.tag): [(1, other_curve.tag)],
        (1, curve.tag): [(0, start.tag), (0, end.tag)],
        (1, other_curve.tag): [
            (0, other_start.tag),
            (0, other_end.tag),
        ],
        (3, unrelated_volume.tag): [(2, unrelated_surface.tag)],
        (2, unrelated_surface.tag): [(1, unrelated_curve.tag)],
        (1, unrelated_curve.tag): [
            (0, unrelated_start.tag),
            (0, unrelated_end.tag),
        ],
        (3, outside_volume.tag): [(2, outside_surface.tag)],
        (2, outside_surface.tag): [(1, curve.tag)],
    }
    data["boundaries"].update(boundaries)
    data["boundary_priority"].update(boundaries)
    return {
        "volume": volume,
        "surface": surface,
        "nonadjacent_surface": nonadjacent_surface,
        "curve": curve,
        "other_curve": other_curve,
        "start": start,
        "end": end,
        "other_start": other_start,
        "other_end": other_end,
        "unrelated_volume": unrelated_volume,
        "unrelated_surface": unrelated_surface,
        "unrelated_curve": unrelated_curve,
        "unrelated_start": unrelated_start,
        "outside_volume": outside_volume,
        "outside_surface": outside_surface,
    }


def _fake_threshold(
    cad: geometry.GeometryModel,
    distance: gmsh_meshing.MeshFieldRef,
    *,
    size_min: float = 0.05,
    size_max: float = 0.4,
    dist_min: float = 0.1,
    dist_max: float = 0.8,
) -> gmsh_meshing.MeshFieldRef:
    return _mesher(cad).threshold_field(
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
        "transfinite_curve": lambda: _mesher(cad).transfinite_curve(curve, num_nodes=3),
        "transfinite_surface": lambda: _mesher(cad).transfinite_surface(surface),
        "transfinite_volume": lambda: _mesher(cad).transfinite_volume(volume),
        "recombine": lambda: _mesher(cad).recombine(surface),
        "mesh_size": lambda: _mesher(cad).mesh_size([point], size=0.1),
        "distance_field": lambda: _mesher(cad).distance_field(surfaces=[surface]),
        "layered_extrude": lambda: _structured_extrude(
            cad,
            [surface],
            0,
            0,
            1,
            num_elements=[2],
            heights=[1.0],
        ),
        "recombined_extrude": lambda: _structured_extrude(
            cad,
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
    operations = {
        "translate": lambda: cad.translate([entity], 1, 0, 0),
        "rotate": lambda: cad.rotate([entity], 0, 0, 0, 0, 0, 1, 0.5),
        "mirror": lambda: cad.mirror([entity], 1, 0, 0, 0),
        "scale": lambda: cad.scale([entity], 0, 0, 0, 2, 1, 1),
    }
    return operations[operation]()


def _apply_foundational_operation(
    cad: geometry.GeometryModel,
    operation: str,
    entity: geometry.EntityRef,
    tool: geometry.EntityRef,
) -> Any:
    operations = {
        "copy": lambda: cad.copy([entity]),
        "mirror": lambda: cad.mirror([entity], 1, 0, 0, 0),
        "scale": lambda: cad.scale([entity], 0, 0, 0, 2, 1, 1),
        "intersect": lambda: cad.intersect([entity], [tool]),
        "fragment": lambda: cad.fragment([entity], [tool]),
    }
    return operations[operation]()


def _apply_edge_treatment(
    cad: geometry.GeometryModel,
    operation: str,
    topology: dict[str, geometry.EntityRef],
    values: Sequence[float],
    *,
    remove_volumes: bool = True,
) -> geometry.FeatureResult:
    if operation == "fillet":
        return cad.fillet(
            [topology["volume"]],
            [topology["curve"]],
            values,
            remove_volumes=remove_volumes,
        )
    return cad.chamfer(
        [topology["volume"]],
        [topology["curve"]],
        [topology["surface"]],
        values,
        remove_volumes=remove_volumes,
    )


def _occ_operation_call_count(backend: _FakeGmsh, operation: str) -> int:
    return sum(call[0] == operation for call in backend.model.occ.calls)


def _build_fake_topology(cad: geometry.GeometryModel) -> None:
    if cad.dimension == 1:
        start = cad.point(0.0, 0.0, 0.0)
        end = cad.point(1.0, 0.0, 0.0)
        cad.line(start, end)
    elif cad.dimension == 2:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
    else:
        cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)


def _set_fake_element_blocks(
    backend: _FakeGmsh,
    dimension: int,
    *blocks: tuple[int, Sequence[int]],
) -> None:
    backend.model._current_data()["element_blocks"][dimension] = (
        [element_type for element_type, _ in blocks],
        [list(tags) for _, tags in blocks],
        [[] for _ in blocks],
    )


def _first_requested_options(backend: _FakeGmsh) -> dict[str, float]:
    requested: dict[str, float] = {}
    for operation, name, *values in backend.option.calls:
        if operation == "setNumber" and name not in requested:
            requested[name] = float(values[0])
    return requested


_AUTO_OPTION_ORIGINALS = {
    "Mesh.ElementOrder": 7.0,
    "Mesh.SecondOrderIncomplete": 0.25,
    "Mesh.RecombineAll": 0.75,
    "Mesh.MeshSizeFactor": 2.5,
    "Mesh.Algorithm": 4.0,
    "Mesh.Algorithm3D": 7.0,
    "Mesh.RecombinationAlgorithm": 2.0,
    "Mesh.Recombine3DAll": 0.5,
    "Mesh.SubdivisionAlgorithm": 1.0,
    "Mesh.MeshSizeFromPoints": 0.3,
    "Mesh.MeshSizeFromCurvature": 4.0,
    "Mesh.MeshSizeExtendFromBoundary": 0.2,
    "Mesh.MeshSizeMin": 0.03,
    "Mesh.MeshSizeMax": 0.9,
}


def test_public_geometry_api_is_exact() -> None:
    assert geometry.__name__ == "fem.geometry"
    assert geometry.__all__ == [
        "BooleanResult",
        "CurveLoopRef",
        "EntityOwnershipError",
        "EntityRef",
        "FeatureResult",
        "GeometryError",
        "GeometryModel",
        "GeometryStateError",
        "LoftContinuity",
        "LoftParametrization",
        "LoftResult",
        "OrientedCurveRef",
        "StaleEntityError",
        "SweepFrame",
        "WireRef",
        "model",
    ]
    assert all(getattr(geometry, name) is not None for name in geometry.__all__)
    assert not hasattr(geometry, "gmsh")
    assert not hasattr(geometry, "load_gmsh")


def test_importing_public_geometry_does_not_import_external_gmsh() -> None:
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
import fem
from fem import geometry
from fem.geometry import EntityRef, GeometryModel, model

assert fem.geometry is geometry
assert geometry.__name__ == "fem.geometry"
assert EntityRef is geometry.EntityRef
assert GeometryModel is geometry.GeometryModel
assert model is geometry.model
assert "gmsh" not in sys.modules
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


def test_public_reference_types_are_immutable_and_boolean_filter_is_typed() -> None:
    owner = object()
    surface = geometry.EntityRef(2, 8, owner, object())
    curve = geometry.EntityRef(1, 3, owner, object())
    result = geometry.BooleanResult((surface, curve), ((surface,), (curve,)))
    mesh_field = gmsh_meshing.MeshFieldRef(5, "Distance", owner, object())

    assert result.of_dimension(2) == (surface,)
    assert "object" not in repr(surface)
    assert (mesh_field.tag, mesh_field.field_type) == (5, "Distance")
    assert "object" not in repr(mesh_field)
    assert issubclass(gmsh_meshing.MeshFieldOwnershipError, geometry.GeometryError)
    assert issubclass(gmsh_meshing.StaleMeshFieldError, geometry.GeometryError)
    assert issubclass(gmsh_meshing.StaleGmshMeshError, geometry.GeometryError)
    meshing_names = {
        "GmshMeshRef",
        "MeshFieldRef",
        "MeshFieldOwnershipError",
        "StaleGmshMeshError",
        "StaleMeshFieldError",
    }
    assert meshing_names.isdisjoint(geometry.__all__)
    assert meshing_names.issubset(gmsh_meshing.__all__)
    assert "FeatureResult" not in gmsh_meshing.__all__
    with pytest.raises(FrozenInstanceError):
        surface.tag = 9  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        mesh_field.tag = 9  # type: ignore[misc]
    with pytest.raises(ValueError, match="dimension"):
        result.of_dimension(4)
    with pytest.raises(ValueError, match="mesh field type"):
        gmsh_meshing.MeshFieldRef(1, "Box", owner, object())  # type: ignore[arg-type]


def test_feature_result_is_frozen_slotted_and_preserves_output_multiplicity() -> None:
    owner = object()
    source = geometry.EntityRef(1, 1, owner, object())
    first_side = geometry.EntityRef(1, 7, owner, object())
    primary = geometry.EntityRef(2, 4, owner, object())
    end = geometry.EntityRef(1, 9, owner, object())
    second_side = geometry.EntityRef(1, 8, owner, object())

    result = geometry.FeatureResult(
        "extrude",
        [source],  # type: ignore[arg-type]
        [first_side, primary, end, first_side, second_side],  # type: ignore[arg-type]
        [primary],  # type: ignore[arg-type]
        [end],  # type: ignore[arg-type]
        [first_side, second_side],  # type: ignore[arg-type]
    )

    assert result.inputs == (source,)
    assert result.outputs == (
        first_side,
        primary,
        end,
        first_side,
        second_side,
    )
    assert result.primary == (primary,)
    assert result.ends == (end,)
    assert result.sides == (first_side, second_side)
    assert result.of_dimension(1) == (
        first_side,
        end,
        first_side,
        second_side,
    )
    assert result.of_dimension(2) == (primary,)
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.operation = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="dimension"):
        result.of_dimension(4)


@pytest.mark.parametrize(
    ("case", "error_type", "message"),
    [
        ("empty_operation", ValueError, "operation"),
        ("foreign_owner", geometry.EntityOwnershipError, "one geometry model"),
        ("mixed_inputs", ValueError, "common dimension"),
        ("duplicate_semantic", ValueError, "duplicate-free"),
        ("overlap", ValueError, "disjoint"),
        ("incomplete_partition", ValueError, "partition"),
        ("wrong_order", ValueError, "first-seen output order"),
    ],
)
def test_feature_result_rejects_malformed_semantic_partitions(
    case: str,
    error_type: type[Exception],
    message: str,
) -> None:
    owner = object()
    source = geometry.EntityRef(1, 1, owner, object())
    primary = geometry.EntityRef(2, 2, owner, object())
    end = geometry.EntityRef(1, 3, owner, object())
    first_side = geometry.EntityRef(1, 4, owner, object())
    second_side = geometry.EntityRef(1, 5, owner, object())
    kwargs: dict[str, Any] = {
        "operation": "extrude",
        "inputs": (source,),
        "outputs": (first_side, primary, end, second_side),
        "primary": (primary,),
        "ends": (end,),
        "sides": (first_side, second_side),
    }
    if case == "empty_operation":
        kwargs["operation"] = ""
    elif case == "foreign_owner":
        foreign = geometry.EntityRef(1, 6, object(), object())
        kwargs["outputs"] = (foreign, primary, end)
        kwargs["sides"] = (foreign,)
    elif case == "mixed_inputs":
        kwargs["inputs"] = (
            source,
            geometry.EntityRef(0, 6, owner, object()),
        )
    elif case == "duplicate_semantic":
        kwargs["sides"] = (first_side, first_side, second_side)
    elif case == "overlap":
        kwargs["sides"] = (first_side, end, second_side)
    elif case == "incomplete_partition":
        kwargs["sides"] = (first_side,)
    else:
        kwargs["outputs"] = (second_side, primary, end, first_side)

    with pytest.raises(error_type, match=message):
        geometry.FeatureResult(**kwargs)


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
        match="native facade access.*Gmsh session is not active",
    ):
        _ = cad._gmsh


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


def test_transfinite_curve_and_recombine_forward_typed_targets_while_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("controls", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].add((1, 7))
        curve = cad.entity(1, 7)

        assert _mesher(cad).transfinite_curve(curve, num_nodes=np.int64(5)) is None
        assert _mesher(cad).recombine(surface) is None
        assert _mesher(cad).transfinite_curve(curve, num_nodes=3) is None
        assert _mesher(cad).recombine(surface) is None

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
        lambda cad, surface, curve: _mesher(cad).transfinite_curve(
            surface, num_nodes=3
        ),
        lambda cad, surface, curve: _mesher(cad).transfinite_curve(1, num_nodes=3),
        lambda cad, surface, curve: _mesher(cad).transfinite_curve(
            curve, num_nodes=True
        ),
        lambda cad, surface, curve: _mesher(cad).transfinite_curve(curve, num_nodes=1),
        lambda cad, surface, curve: _mesher(cad).transfinite_curve(
            curve, num_nodes=2.5
        ),
        lambda cad, surface, curve: _mesher(cad).recombine(curve),
        lambda cad, surface, curve: _mesher(cad).recombine((2, 1)),
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
                _mesher(inner).recombine(outer_surface)
            assert backend.model.occ.synchronize_calls == synchronize_calls

            with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
                _ = inner.raw_model
            assert _mesher(inner).recombine(inner_surface) is None

    assert sum(call[0] == "setRecombine" for call in backend.model.mesh.calls) == 1


def test_native_mesh_control_failure_preserves_state_and_mesh_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("retry-controls", dimension=2) as cad:
        surface = cad.rectangle(0, 0, 1, 1)
        backend.model.mesh.fail_next.add("setRecombine")
        with pytest.raises(RuntimeError, match="fake setRecombine failure"):
            _mesher(cad).recombine(surface)

        native_mesh = _generate_mesh(cad, recombine=False)
        assert isinstance(native_mesh, gmsh_meshing.GmshMeshRef)


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

        assert _mesher(cad).transfinite_surface(surface) is None
        assert (
            _mesher(cad).transfinite_surface(
                surface,
                corners=(points[3], points[1], points[0], points[2]),
            )
            is None
        )
        assert _mesher(cad).transfinite_surface(surface, corners=points[:3]) is None
        assert cad.entity(0, points[0].tag) == points[0]

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
            _mesher(cad).transfinite_surface(surface, corners=supplied)

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
            _mesher(cad).transfinite_surface(
                surface,
                corners=(points[0], points[1], points[3]),
            )

    assert backend.model.mesh.calls == []


def test_transfinite_surface_rejects_cross_model_and_missing_native_corners(
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
                _mesher(inner).transfinite_surface(
                    surface,
                    corners=(points[0], points[1], foreign),
                )
            assert backend.model.occ.synchronize_calls == synchronize_calls

            backend.model._current_data()["entities"].discard(
                (0, points[2].tag)
            )
            with pytest.raises(geometry.StaleEntityError):
                _mesher(inner).transfinite_surface(surface, corners=points)
            assert backend.model.occ.synchronize_calls == synchronize_calls + 1

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

        assert _mesher(cad).transfinite_volume(volume) is None
        assert (
            _mesher(cad).transfinite_volume(
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
        assert _mesher(cad).transfinite_volume(volume, corners=tuple(reversed(points))) is None

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
            _mesher(cad).transfinite_volume(volume, corners=supplied)

        assert backend.model.mesh.calls == []
        assert backend.model.occ.synchronize_calls == synchronize_calls


def test_transfinite_volume_requires_3d_facade_and_recursive_boundary_corners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("two-dimensional-volume", dimension=2) as cad:
        backend.model._current_data()["entities"].add((3, 1))
        synchronize_calls = backend.model.occ.synchronize_calls
        with pytest.raises(ValueError, match="model dimension"):
            cad.entity(3, 1)
        assert backend.model.occ.synchronize_calls == synchronize_calls

    with geometry.model("volume-membership", dimension=3) as cad:
        volume = cad.box(0, 0, 0, 1, 1, 1)
        backend.model._current_data()["entities"].update(
            (0, tag) for tag in range(1, 9)
        )
        points = tuple(cad.entity(0, tag) for tag in range(1, 9))
        backend.model.boundary_result = [(0, tag) for tag in range(1, 8)]

        with pytest.raises(ValueError, match="boundary"):
            _mesher(cad).transfinite_volume(volume, corners=points)

    assert backend.model.mesh.calls == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad: _mesher(cad).transfinite_curve(None, num_nodes=3),
        lambda cad: _mesher(cad).transfinite_surface(None),
        lambda cad: _mesher(cad).transfinite_volume(None),
        lambda cad: _mesher(cad).recombine(None),
        lambda cad: _mesher(cad).mesh_size([], size=0.1),
        lambda cad: _mesher(cad).distance_field(),
        lambda cad: _mesher(cad).threshold_field(
            None,
            size_min=0.1,
            size_max=0.2,
            dist_min=0.0,
            dist_max=1.0,
        ),
        lambda cad: _mesher(cad).min_field([]),
        lambda cad: _mesher(cad).background_field(None),
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


def test_mesh_controls_reject_meshed_and_mesh_failed_states_contextually(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    operations = (
        lambda cad: _mesher(cad).transfinite_curve(None, num_nodes=3),
        lambda cad: _mesher(cad).transfinite_surface(None),
        lambda cad: _mesher(cad).transfinite_volume(None),
        lambda cad: _mesher(cad).recombine(None),
        lambda cad: _mesher(cad).mesh_size([], size=0.1),
        lambda cad: _mesher(cad).distance_field(),
        lambda cad: _mesher(cad).threshold_field(
            None,
            size_min=0.1,
            size_max=0.2,
            dist_min=0.0,
            dist_max=1.0,
        ),
        lambda cad: _mesher(cad).min_field([]),
        lambda cad: _mesher(cad).background_field(None),
    )

    with geometry.model("meshed-controls", dimension=3) as cad:
        cad.box(0, 0, 0, 1, 1, 1)
        _generate_mesh(cad, )
        for operation in operations:
            with pytest.raises(geometry.GeometryStateError, match="MESHED"):
                operation(cad)

    with geometry.model("failed-controls", dimension=3) as cad:
        cad.box(0, 0, 0, 1, 1, 1)
        backend.model.mesh.fail_generate = True
        with pytest.raises(RuntimeError, match="fake mesh failure"):
            _generate_mesh(cad, )
        for operation in operations:
            with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
                operation(cad)


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, curve, surface, volume: _mesher(cad).transfinite_curve(
            surface, num_nodes=3
        ),
        lambda cad, curve, surface, volume: _mesher(cad).transfinite_surface(curve),
        lambda cad, curve, surface, volume: _mesher(cad).transfinite_surface(2),
        lambda cad, curve, surface, volume: _mesher(cad).transfinite_volume(surface),
        lambda cad, curve, surface, volume: _mesher(cad).transfinite_volume((3, 1)),
        lambda cad, curve, surface, volume: _mesher(cad).recombine(volume),
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
        (1, lambda cad, target: _mesher(cad).transfinite_curve(target, num_nodes=3)),
        (2, lambda cad, target: _mesher(cad).transfinite_surface(target)),
        (3, lambda cad, target: _mesher(cad).transfinite_volume(target)),
        (2, lambda cad, target: _mesher(cad).recombine(target)),
    ],
)
def test_every_mesh_control_rejects_foreign_and_missing_native_targets(
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

            backend.model._current_data()["entities"].discard((dimension, 9))
            with pytest.raises(geometry.StaleEntityError):
                operation(inner, local)
            assert backend.model.occ.synchronize_calls == synchronize_calls + 1

    assert backend.model.mesh.calls == []


def test_transfinite_volume_rejects_foreign_and_missing_native_corners(
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
                _mesher(inner).transfinite_volume(
                    volume,
                    corners=(*points[:5], foreign),
                )
            assert backend.model.occ.synchronize_calls == synchronize_calls

            backend.model._current_data()["entities"].discard(
                (0, points[5].tag)
            )
            with pytest.raises(geometry.StaleEntityError):
                _mesher(inner).transfinite_volume(volume, corners=points)
            assert backend.model.occ.synchronize_calls == synchronize_calls + 1

    assert backend.model.mesh.calls == []


@pytest.mark.parametrize(
    ("native_operation", "target_name", "operation"),
    [
        (
            "setTransfiniteCurve",
            "curve",
            lambda cad, target: _mesher(cad).transfinite_curve(target, num_nodes=3),
        ),
        (
            "setTransfiniteSurface",
            "surface",
            lambda cad, target: _mesher(cad).transfinite_surface(target),
        ),
        (
            "setTransfiniteVolume",
            "volume",
            lambda cad, target: _mesher(cad).transfinite_volume(target),
        ),
        ("setRecombine", "surface", lambda cad, target: _mesher(cad).recombine(target)),
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

        native_mesh = _generate_mesh(cad, recombine=False)
        assert isinstance(native_mesh, gmsh_meshing.GmshMeshRef)


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
            lambda: _mesher(outer).transfinite_curve(outer_curve, num_nodes=3),
            lambda: _mesher(outer).transfinite_surface(outer_surface),
            lambda: _mesher(outer).transfinite_volume(outer_volume),
            lambda: _mesher(outer).recombine(outer_surface),
        )
        for operation in outer_operations:
            backend.model.setCurrent("external")
            operation()

        with geometry.model("inner-mesh-controls", dimension=3) as inner:
            inner_volume = inner.box(0, 0, 0, 1, 1, 1)
            _mesher(inner).transfinite_volume(inner_volume)

        _mesher(outer).transfinite_volume(outer_volume)

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


def test_mesh_size_forwards_ordered_batches_while_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("point-sizes", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point_3, point_1 = _fake_entities(cad, backend, 0, 3, 1)

        synchronize_calls = backend.model.occ.synchronize_calls
        assert (
            _mesher(cad).mesh_size(
                (point for point in (point_3, point_1)),
                size=np.float64(0.2),
            )
            is None
        )
        assert backend.model.occ.synchronize_calls == synchronize_calls + 1

        synchronize_calls = backend.model.occ.synchronize_calls
        assert _mesher(cad).mesh_size([point_1], size=0.025) is None
        assert backend.model.occ.synchronize_calls == synchronize_calls + 1

    assert backend.model.mesh.calls == [
        ("setSize", ((0, 3), (0, 1)), 0.2, "point-sizes"),
        ("setSize", ((0, 1),), 0.025, "point-sizes"),
    ]
    assert backend.model.mesh.field.calls == []
    assert backend.option.calls == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, point, surface: _mesher(cad).mesh_size([], size=0.1),
        lambda cad, point, surface: _mesher(cad).mesh_size([point, point], size=0.1),
        lambda cad, point, surface: _mesher(cad).mesh_size([surface], size=0.1),
        lambda cad, point, surface: _mesher(cad).mesh_size([object()], size=0.1),
        lambda cad, point, surface: _mesher(cad).mesh_size([point], size=True),
        lambda cad, point, surface: _mesher(cad).mesh_size([point], size=0.0),
        lambda cad, point, surface: _mesher(cad).mesh_size([point], size=-0.1),
        lambda cad, point, surface: _mesher(cad).mesh_size([point], size=float("inf")),
        lambda cad, point, surface: _mesher(cad).mesh_size([point], size=float("nan")),
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
            _mesher(cad).mesh_size(broken_points(), size=0.1)
        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert backend.model.mesh.calls == []


def test_mesh_size_rejects_foreign_and_missing_native_points_before_mutation(
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
                _mesher(inner).mesh_size([foreign], size=0.1)
            assert backend.model.occ.synchronize_calls == synchronize_calls

            backend.model._current_data()["entities"].discard((0, local.tag))
            with pytest.raises(geometry.StaleEntityError):
                _mesher(inner).mesh_size([local], size=0.1)
            assert backend.model.occ.synchronize_calls == synchronize_calls + 1

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
        distance = _mesher(cad).distance_field(**kwargs)
        assert backend.model.occ.synchronize_calls == synchronize_calls + 1
        assert (distance.tag, distance.field_type) == (1, "Distance")

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
        lambda cad, point, curve, surface: _mesher(cad).distance_field(),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(
            points=[point, point]
        ),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(points=[curve]),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(curves=[point]),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(surfaces=[curve]),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(points=[object()]),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(
            points=[point], sampling=True
        ),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(
            points=[point], sampling=1
        ),
        lambda cad, point, curve, surface: _mesher(cad).distance_field(
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
            _mesher(cad).distance_field(points=[point], curves=broken_curves())
        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert backend.model.mesh.field.calls == []


def test_threshold_min_and_background_build_an_ordered_inert_field_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("field-graph", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        curve = _fake_entities(cad, backend, 1, 7)[0]
        distance = _mesher(cad).distance_field(curves=[curve], sampling=30)
        first = _fake_threshold(cad, distance)
        assert backend.model._current_data()["background_mesh_field"] is None

        second = _fake_threshold(
            cad,
            distance,
            size_min=0.02,
            size_max=0.3,
            dist_min=0.05,
            dist_max=0.6,
        )
        minimum = _mesher(cad).min_field((field for field in (second, first)))
        nested = _mesher(cad).min_field([minimum, second])
        assert _mesher(cad).background_field(nested) is None

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
        lambda cad, distance, threshold: _mesher(cad).threshold_field(
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
        distance = _mesher(cad).distance_field(points=[point])
        threshold = _fake_threshold(cad, distance)
        calls = list(backend.model.mesh.field.calls)

        with pytest.raises((TypeError, ValueError)):
            operation(cad, distance, threshold)

        assert backend.model.mesh.field.calls == calls


@pytest.mark.parametrize(
    "operation",
    [
        lambda cad, distance, first, second: _mesher(cad).min_field([]),
        lambda cad, distance, first, second: _mesher(cad).min_field([first]),
        lambda cad, distance, first, second: _mesher(cad).min_field([first, first]),
        lambda cad, distance, first, second: _mesher(cad).min_field([distance, first]),
        lambda cad, distance, first, second: _mesher(cad).min_field([object(), first]),
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
        distance = _mesher(cad).distance_field(points=[point])
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


def test_distance_sources_reject_foreign_and_missing_native_entities(
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
                _mesher(inner).distance_field(points=[foreign])
            assert backend.model.occ.synchronize_calls == synchronize_calls

            backend.model._current_data()["entities"].discard((0, local.tag))
            with pytest.raises(geometry.StaleEntityError):
                _mesher(inner).distance_field(points=[local])
            assert backend.model.occ.synchronize_calls == synchronize_calls + 1

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
        outer_distance = _mesher(outer).distance_field(points=[outer_point])
        outer_threshold = _fake_threshold(outer, outer_distance)
        with geometry.model("inner-fields", dimension=2) as inner:
            inner_point = _fake_entities(inner, backend, 0, 1)[0]
            inner_distance = _mesher(inner).distance_field(points=[inner_point])
            assert outer_distance.tag == inner_distance.tag == 1

            calls = list(backend.model.mesh.field.calls)
            with pytest.raises(gmsh_meshing.MeshFieldOwnershipError):
                _fake_threshold(inner, outer_distance)
            assert backend.model.mesh.field.calls == calls

            backend.model.mesh.field.remove(inner_distance.tag)
            with pytest.raises(gmsh_meshing.StaleMeshFieldError, match="no longer exists"):
                _fake_threshold(inner, inner_distance)

            replacement = _mesher(inner).distance_field(points=[inner_point])
            assert replacement.tag == inner_distance.tag
            assert replacement != inner_distance
            with pytest.raises(gmsh_meshing.StaleMeshFieldError, match="stale"):
                _fake_threshold(inner, inner_distance)

            threshold = _fake_threshold(inner, replacement)
            assert threshold.field_type == "Threshold"
            calls = list(backend.model.mesh.field.calls)
            with pytest.raises(gmsh_meshing.MeshFieldOwnershipError):
                _mesher(inner).background_field(outer_threshold)
            with pytest.raises(gmsh_meshing.MeshFieldOwnershipError):
                _mesher(inner).min_field([threshold, outer_threshold])
            assert backend.model.mesh.field.calls == calls

            with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
                _ = inner.raw_model
            assert _mesher(inner).background_field(threshold) is None

        assert _mesher(outer).background_field(outer_threshold) is None

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
            _mesher(cad).distance_field(
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
        distance = _mesher(cad).distance_field(points=[point])
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
        distance = _mesher(cad).distance_field(points=[point])
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
            _mesher(cad).min_field([second, first])

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
            _mesher(cad).distance_field(points=[point])
        assert backend.model._current_data()["mesh_fields"] == {}

        backend.model.mesh.field.hidden_tags.add(1)
        with pytest.raises(geometry.GeometryError, match="not active"):
            _mesher(cad).distance_field(points=[point])
        assert backend.model._current_data()["mesh_fields"] == {}

        live = _mesher(cad).distance_field(points=[point])
        assert (live.tag, live.field_type) == (1, "Distance")


def test_field_rollback_failure_preserves_primary_error_note_and_mesh_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("rollback-note", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        backend.model.mesh.field.fail_next.update(
            {("setNumber", "Sampling"), ("remove", None)}
        )

        with pytest.raises(RuntimeError, match="setNumber") as captured:
            _mesher(cad).distance_field(points=[point])
        assert any(
            "rollback" in note and "remove" in note
            for note in getattr(captured.value, "__notes__", ())
        )
        assert set(backend.model._current_data()["mesh_fields"]) == {1}

        replacement = _mesher(cad).distance_field(points=[point])
        assert replacement.tag == 2
        assert isinstance(_generate_mesh(cad, size=0.2), gmsh_meshing.GmshMeshRef)


def test_background_selection_failure_is_retryable_and_keeps_fields_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("background-retry", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = _mesher(cad).distance_field(points=[point])
        threshold = _fake_threshold(cad, distance)
        backend.model.mesh.field.fail_next.add(("setAsBackgroundMesh", None))

        with pytest.raises(RuntimeError, match="setAsBackgroundMesh"):
            _mesher(cad).background_field(threshold)
        assert backend.model._current_data()["background_mesh_field"] is None
        assert _mesher(cad).background_field(threshold) is None
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
        distance = _mesher(cad).distance_field(points=[point])
        threshold = _fake_threshold(cad, distance)
        calls = list(backend.model.mesh.field.calls)

        with pytest.raises(TypeError):
            _mesher(cad).background_field(object())
        with pytest.raises(ValueError, match="Threshold or Min"):
            _mesher(cad).background_field(distance)
        assert backend.model.mesh.field.calls == calls

        _mesher(cad).background_field(threshold)
        calls = list(backend.model.mesh.field.calls)
        with pytest.raises(ValueError, match="only once"):
            _mesher(cad).background_field(threshold)
        assert backend.model.mesh.field.calls == calls


@pytest.mark.parametrize("mode", ["point", "background"])
def test_typed_size_mode_conflicts_are_retryable_before_native_generation(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"{mode}-mesh-conflict", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        distance = _mesher(cad).distance_field(points=[point])
        threshold = _fake_threshold(cad, distance)
        if mode == "point":
            _mesher(cad).mesh_size([point], size=0.1)
            field_calls = list(backend.model.mesh.field.calls)
            with pytest.raises(ValueError, match="point sizes"):
                _mesher(cad).background_field(threshold)
            assert backend.model.mesh.field.calls == field_calls
        else:
            _mesher(cad).background_field(threshold)
            mesh_calls = list(backend.model.mesh.calls)
            with pytest.raises(ValueError, match="background field"):
                _mesher(cad).mesh_size([point], size=0.1)
            assert backend.model.mesh.calls == mesh_calls

        mesh_calls = list(backend.model.mesh.calls)
        option_calls = list(backend.option.calls)
        with pytest.raises(ValueError, match="size cannot be supplied"):
            _generate_mesh(cad, size=0.2)
        assert backend.model.mesh.calls == mesh_calls
        assert backend.option.calls == option_calls

        assert isinstance(_generate_mesh(cad, ), gmsh_meshing.GmshMeshRef)


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

    with geometry.model(f"{mode}-options", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        if mode == "point":
            _mesher(cad).mesh_size([point], size=0.1)
        else:
            distance = _mesher(cad).distance_field(points=[point])
            _mesher(cad).background_field(_fake_threshold(cad, distance))
        backend.option.calls.clear()

        assert isinstance(
            _generate_mesh(cad, order=2, recombine=True),
            gmsh_meshing.GmshMeshRef,
        )

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
    [("mesh", "background"), ("option", "point")],
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

    with geometry.model(f"{mode}-{failure}-restore", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        point = _fake_entities(cad, backend, 0, 1)[0]
        if mode == "point":
            _mesher(cad).mesh_size([point], size=0.1)
        else:
            distance = _mesher(cad).distance_field(points=[point])
            _mesher(cad).background_field(_fake_threshold(cad, distance))

        expected = {
            "mesh": "fake mesh failure",
            "option": "Mesh.MeshSizeMax",
        }[failure]
        with pytest.raises(RuntimeError, match=expected):
            _generate_mesh(cad, order=2, recombine=True)

        assert backend.option.values == original
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_mesh(cad, )


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

    with geometry.model("outer-size-mode", dimension=2) as outer:
        outer.rectangle(0, 0, 1, 1)
        outer_point = _fake_entities(outer, backend, 0, 1)[0]
        distance = _mesher(outer).distance_field(points=[outer_point])
        threshold = _fake_threshold(outer, distance)
        _mesher(outer).background_field(threshold)
        assert set(backend.model._current_data()["mesh_fields"]) == {1, 2}

        with geometry.model("inner-size-mode", dimension=2) as inner:
            inner.rectangle(0, 0, 1, 1)
            inner_point = _fake_entities(inner, backend, 0, 1)[0]
            _mesher(inner).mesh_size([inner_point], size=0.1)
            _generate_mesh(inner, )
            assert backend.option.values == original

        assert backend.model.current == "outer-size-mode"
        assert set(backend.model._current_data()["mesh_fields"]) == {1, 2}
        _generate_mesh(outer, )
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
            _generate_mesh(cad, **kwargs)
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []


def test_generation_surface_is_owned_only_by_mesher_specs() -> None:
    removed = {
        "generate_mesh",
        "generate_auto_mesh",
        "transfinite_curve",
        "transfinite_surface",
        "transfinite_volume",
        "recombine",
        "mesh_size",
        "distance_field",
        "threshold_field",
        "min_field",
        "background_field",
    }
    assert all(not hasattr(geometry.GeometryModel, name) for name in removed)
    assert tuple(inspect.signature(gmsh_meshing.Mesher.generate).parameters) == (
        "self",
        "spec",
    )
    assert tuple(inspect.signature(gmsh_meshing.MeshSpec).parameters) == (
        "size",
        "order",
        "recombine",
    )
    assert tuple(inspect.signature(gmsh_meshing.AutoMeshSpec).parameters) == (
        "level",
        "cell_shape",
        "order",
    )


def test_missing_top_dimensional_entity_leaves_mesher_retryable_but_geometry_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("mesh", dimension=2) as cad:
        with pytest.raises(ValueError, match="top-dimensional"):
            _generate_mesh(cad, )
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []

        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            cad.rectangle(0, 0, 1, 1)
        with pytest.raises(ValueError, match="top-dimensional"):
            _generate_mesh(cad, )


def test_generate_mesh_assigns_size_isolates_options_and_returns_live_handle(
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

    with geometry.model("mesh", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].update({(0, 3), (0, 1)})
        result = _generate_mesh(cad,
            size=0.2,
            order=2,
            recombine=True,
        )

        assert isinstance(result, gmsh_meshing.GmshMeshRef)
        assert (result.dimension, result.model_name) == (2, "mesh")
        assert "GeometryModel" not in repr(result)
        assert "object" not in repr(result)
        assert result._borrow_model() is backend.model
        assert result._borrow_model() is backend.model
        with pytest.raises(FrozenInstanceError):
            result.model_name = "renamed"  # type: ignore[misc]
        with pytest.raises(geometry.GeometryStateError, match="MESHED"):
            _generate_mesh(cad, )
        with pytest.raises(geometry.GeometryStateError, match="MESHED"):
            cad.entities(2)

    with pytest.raises(
        gmsh_meshing.StaleGmshMeshError,
        match="mesh.*inside the owning geometry model context",
    ):
        result._borrow_model()

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
    set_calls = [call for call in backend.option.calls if call[0] == "setNumber"]
    assert set_calls[:4] == [
        ("setNumber", "Mesh.ElementOrder", 2.0),
        ("setNumber", "Mesh.SecondOrderIncomplete", 1.0),
        ("setNumber", "Mesh.RecombineAll", 1.0),
        ("setNumber", "Mesh.MeshSizeFromPoints", 1.0),
    ]


def test_generated_handle_reactivates_owner_across_nested_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(
        initialized=True,
        names=("external",),
        current="external",
    )
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-native-mesh", dimension=2) as outer:
        outer.rectangle(0.0, 0.0, 1.0, 1.0)
        outer_mesh = _generate_mesh(outer, )

        with geometry.model("inner-native-mesh", dimension=3) as inner:
            inner.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            assert backend.model.current == "inner-native-mesh"

            assert outer_mesh._borrow_model() is backend.model
            assert backend.model.current == "outer-native-mesh"
            assert len(inner.entities(3)) == 1
            assert backend.model.current == "inner-native-mesh"

        assert outer_mesh._borrow_model() is backend.model
        assert backend.model.current == "outer-native-mesh"

    assert backend.model.current == "external"
    with pytest.raises(gmsh_meshing.StaleGmshMeshError, match="outer-native-mesh"):
        outer_mesh._borrow_model()


def test_generated_handle_rejects_forged_generation_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("native-identity", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        native_mesh = _generate_mesh(cad, )
        forged = gmsh_meshing.GmshMeshRef(
            native_mesh.dimension,
            native_mesh.model_name,
            cad,
            object(),
            object(),
        )

        with pytest.raises(gmsh_meshing.StaleGmshMeshError, match="native-identity"):
            forged._borrow_model()


def test_generated_handle_rejects_malformed_owner_before_dispatch() -> None:
    malformed = gmsh_meshing.GmshMeshRef(
        2,
        "malformed-owner",
        object(),  # type: ignore[arg-type]
        object(),
        object(),
    )

    with pytest.raises(gmsh_meshing.StaleGmshMeshError, match="malformed-owner"):
        malformed._borrow_model()


def test_generated_handle_detects_missing_native_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("missing-native-model", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        native_mesh = _generate_mesh(cad, )
        del backend.model.models["missing-native-model"]

        with pytest.raises(
            gmsh_meshing.StaleGmshMeshError,
            match="missing-native-model",
        ):
            native_mesh._borrow_model()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"order": 2}, "order"),
        ({"recombine": True}, "recombine"),
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
            _generate_mesh(cad, **kwargs)
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []

def test_1d_generate_mesh_returns_native_handle_and_restores_options(
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

    with geometry.model("members", dimension=1) as cad:
        start = cad.point(0, 0, 1)
        end = cad.point(2, 3, 4)
        cad.line(start, end)
        result = _generate_mesh(cad, size=0.25)
        assert isinstance(result, gmsh_meshing.GmshMeshRef)
        assert (result.dimension, result.model_name) == (1, "members")

    assert backend.model.mesh.calls == [
        ("setSize", ((0, 1), (0, 2)), 0.25, "members"),
        ("generate", 1, "members"),
    ]
    assert backend.option.values == {
        "Mesh.ElementOrder": 7.0,
        "Mesh.SecondOrderIncomplete": 0.25,
        "Mesh.RecombineAll": 0.75,
        "Mesh.MeshSizeFromPoints": 0.0,
    }


def test_1d_missing_curve_preflight_keeps_mesher_retryable_and_seals_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("members", dimension=1) as cad:
        cad.point(0, 0, 0)
        with pytest.raises(ValueError, match="top-dimensional"):
            _generate_mesh(cad, )
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []

        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            cad.point(1, 0, 0)
        with pytest.raises(ValueError, match="top-dimensional"):
            _generate_mesh(cad, )


def test_failed_generation_restores_options_and_disallows_retry(
    monkeypatch: pytest.MonkeyPatch,
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
    backend.model.mesh.fail_generate = True
    _install_backend(monkeypatch, backend)

    with geometry.model("mesh", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        backend.model._current_data()["entities"].add((0, 1))
        with pytest.raises(RuntimeError, match="fake mesh failure") as captured:
            _generate_mesh(cad, size=0.2, order=2, recombine=True)

        assert any(
            "mesh generation failed" in note
            for note in getattr(captured.value, "__notes__", ())
        )

        assert backend.option.values == {
            "Mesh.ElementOrder": 4.0,
            "Mesh.SecondOrderIncomplete": 0.0,
            "Mesh.RecombineAll": 0.5,
            "Mesh.MeshSizeFromPoints": 0.0,
        }
        assert len(cad.entities(2)) == 1
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_mesh(cad, )


def test_size_assignment_without_points_consumes_mesh_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("mesh", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        with pytest.raises(geometry.GeometryError, match="point"):
            _generate_mesh(cad, size=0.25)
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_mesh(cad, )
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

    with geometry.model("mesh", dimension=2) as cad:
        cad.rectangle(0, 0, 1, 1)
        with pytest.raises(RuntimeError, match="Mesh.RecombineAll"):
            _generate_mesh(cad, order=2, recombine=True)
        assert backend.option.values == original
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_mesh(cad, )


@pytest.mark.parametrize(
    (
        "dimension",
        "cell_shape",
        "order",
        "element_type",
        "policy_options",
    ),
    [
        (
            1,
            None,
            1,
            1,
            {
                "Mesh.RecombineAll": 0.0,
                "Mesh.SubdivisionAlgorithm": 0.0,
            },
        ),
        (
            2,
            None,
            1,
            2,
            {
                "Mesh.Algorithm": 6.0,
                "Mesh.RecombineAll": 1.0,
                "Mesh.RecombinationAlgorithm": 1.0,
                "Mesh.SubdivisionAlgorithm": 0.0,
            },
        ),
        (
            2,
            "tri",
            2,
            9,
            {
                "Mesh.Algorithm": 6.0,
                "Mesh.RecombineAll": 0.0,
                "Mesh.SubdivisionAlgorithm": 0.0,
            },
        ),
        (
            2,
            "tri-quad",
            2,
            16,
            {
                "Mesh.Algorithm": 6.0,
                "Mesh.RecombineAll": 1.0,
                "Mesh.RecombinationAlgorithm": 1.0,
                "Mesh.SubdivisionAlgorithm": 0.0,
            },
        ),
        (
            2,
            "quad",
            2,
            16,
            {
                "Mesh.Algorithm": 6.0,
                "Mesh.RecombineAll": 1.0,
                "Mesh.RecombinationAlgorithm": 3.0,
                "Mesh.SubdivisionAlgorithm": 0.0,
            },
        ),
        (
            3,
            None,
            1,
            4,
            {
                "Mesh.Algorithm": 6.0,
                "Mesh.Algorithm3D": 1.0,
                "Mesh.RecombineAll": 0.0,
                "Mesh.Recombine3DAll": 0.0,
                "Mesh.SubdivisionAlgorithm": 0.0,
            },
        ),
        (
            3,
            "tet",
            2,
            11,
            {
                "Mesh.Algorithm": 6.0,
                "Mesh.Algorithm3D": 1.0,
                "Mesh.RecombineAll": 0.0,
                "Mesh.Recombine3DAll": 0.0,
                "Mesh.SubdivisionAlgorithm": 0.0,
            },
        ),
        (
            3,
            "hex",
            2,
            17,
            {
                "Mesh.Algorithm": 6.0,
                "Mesh.Algorithm3D": 1.0,
                "Mesh.RecombineAll": 0.0,
                "Mesh.Recombine3DAll": 0.0,
                "Mesh.SubdivisionAlgorithm": 2.0,
            },
        ),
    ],
)
def test_auto_mesh_legal_shape_matrix_uses_exact_fixed_policy(
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    cell_shape: Any,
    order: int,
    element_type: int,
    policy_options: dict[str, float],
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    _install_backend(monkeypatch, backend)

    with geometry.model(
        f"auto-policy-{dimension}-{cell_shape}-{order}",
        dimension=dimension,
    ) as cad:
        _build_fake_topology(cad)
        _set_fake_element_blocks(backend, dimension, (element_type, (101, 102)))
        result = _generate_auto_mesh(cad,
            cell_shape=cell_shape,
            order=order,
        )

    expected_options = {
        "Mesh.ElementOrder": float(order),
        "Mesh.SecondOrderIncomplete": 1.0 if order == 2 else 0.0,
        "Mesh.MeshSizeFactor": 1.0,
        **policy_options,
    }
    assert isinstance(result, gmsh_meshing.GmshMeshRef)
    assert result.dimension == dimension
    assert _first_requested_options(backend) == expected_options
    assert backend.option.values == _AUTO_OPTION_ORIGINALS
    assert backend.model.mesh.generate_calls == [dimension]
    assert backend.model.mesh.refine_calls == 0
    assert [call for call in backend.model.mesh.calls if call[0] == "getElements"] == [
        (
            "getElements",
            dimension,
            -1,
            f"auto-policy-{dimension}-{cell_shape}-{order}",
        )
    ]


@pytest.mark.parametrize(
    ("dimension", "cell_shape"),
    [
        *((1, shape) for shape in ("tri", "tri-quad", "quad", "tet", "hex")),
        (2, "tet"),
        (2, "hex"),
        (3, "tri"),
        (3, "tri-quad"),
        (3, "quad"),
        (2, "TRI"),
        (2, "triangle"),
        (2, 3),
    ],
)
def test_auto_mesh_rejects_invalid_shape_matrix_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    cell_shape: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(
        f"invalid-auto-shape-{dimension}-{cell_shape}",
        dimension=dimension,
    ) as cad:
        _build_fake_topology(cad)
        synchronize_calls = backend.model.occ.synchronize_calls
        with pytest.raises((TypeError, ValueError)) as captured:
            _generate_auto_mesh(cad, cell_shape=cell_shape)

        assert "cell_shape" in str(captured.value)
        assert "cell_shape" in str(captured.value)
        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []


@pytest.mark.parametrize("level", [True, False, 1.0, "3", 0, 6, -1])
def test_auto_mesh_rejects_invalid_levels_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
    level: Any,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"invalid-auto-level-{level}", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        synchronize_calls = backend.model.occ.synchronize_calls
        with pytest.raises((TypeError, ValueError)) as captured:
            _generate_auto_mesh(cad, level=level)

        assert "level" in str(captured.value)
        assert "level" in str(captured.value)
        assert backend.model.occ.synchronize_calls == synchronize_calls
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_auto_mesh_levels_set_dimension_aware_absolute_size_factor(
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    level: int,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    _install_backend(monkeypatch, backend)
    default_type = {1: 1, 2: 2, 3: 4}[dimension]

    with geometry.model(f"auto-level-{dimension}-{level}", dimension=dimension) as cad:
        _build_fake_topology(cad)
        _set_fake_element_blocks(backend, dimension, (default_type, (1,)))
        _generate_auto_mesh(cad, level=level)

    assert _first_requested_options(backend)["Mesh.MeshSizeFactor"] == pytest.approx(
        2.0 ** ((3 - level) / dimension)
    )
    assert backend.option.values == _AUTO_OPTION_ORIGINALS


def test_auto_mesh_preflight_validation_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("auto-validation-retry", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        with pytest.raises((TypeError, ValueError)):
            _generate_auto_mesh(cad, level=True)

        _set_fake_element_blocks(backend, 2, (2, (1,)))
        assert isinstance(_generate_auto_mesh(cad, level=3), gmsh_meshing.GmshMeshRef)


@pytest.mark.parametrize(
    ("dimension", "cell_shape", "order", "element_type"),
    [
        (1, None, 1, 1),
        (2, "tri", 1, 2),
        (2, "tri", 2, 9),
        (2, "quad", 1, 3),
        (2, "quad", 2, 16),
        (3, "tet", 1, 4),
        (3, "tet", 2, 11),
        (3, "hex", 1, 5),
        (3, "hex", 2, 17),
    ],
)
def test_auto_mesh_strict_pure_families_return_native_handles(
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    cell_shape: Any,
    order: int,
    element_type: int,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(
        f"strict-pure-{dimension}-{cell_shape}-{order}",
        dimension=dimension,
    ) as cad:
        _build_fake_topology(cad)
        _set_fake_element_blocks(backend, dimension, (element_type, (1, 2)))
        result = _generate_auto_mesh(cad,
            cell_shape=cell_shape,
            order=order,
        )

    assert isinstance(result, gmsh_meshing.GmshMeshRef)
    assert result.dimension == dimension
    assert backend.model.mesh.generate_calls == [dimension]
    assert backend.model.mesh.refine_calls == 0


@pytest.mark.parametrize(
    ("order", "blocks"),
    [
        (1, ((2, (1, 2)),)),
        (1, ((3, (1, 2)),)),
        (1, ((2, (1,)), (3, (2, 3)))),
        (2, ((9, (1, 2)),)),
        (2, ((16, (1, 2)),)),
        (2, ((9, (1,)), (16, (2, 3)))),
    ],
)
def test_auto_mesh_tri_quad_accepts_each_permitted_family_union(
    monkeypatch: pytest.MonkeyPatch,
    order: int,
    blocks: tuple[tuple[int, Sequence[int]], ...],
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model(f"strict-mixed-{order}-{len(blocks)}", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        _set_fake_element_blocks(backend, 2, *blocks)
        result = _generate_auto_mesh(cad, cell_shape="tri-quad", order=order)

    assert isinstance(result, gmsh_meshing.GmshMeshRef)
    assert backend.model.mesh.generate_calls == [2]
    assert backend.model.mesh.refine_calls == 0


@pytest.mark.parametrize(
    ("raw_blocks", "message"),
    [
        (([], [], []), "no top-dimensional cells"),
        (([3], [[]], [[]]), "no top-dimensional cells"),
        (
            ([3, 2], [[1]], [[], []]),
            "malformed top-dimensional element blocks",
        ),
        (([3], [None], [[]]), "malformed element tags"),
        (([True], [[1]], [[]]), "non-integer element type"),
    ],
)
def test_auto_mesh_strict_validation_rejects_empty_and_malformed_output(
    monkeypatch: pytest.MonkeyPatch,
    raw_blocks: Any,
    message: str,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    _install_backend(monkeypatch, backend)

    with geometry.model(f"strict-malformed-{message}", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        backend.model._current_data()["element_blocks"][2] = raw_blocks
        with pytest.raises(gmsh_meshing.MeshCellShapeError, match=message):
            _generate_auto_mesh(cad, cell_shape="quad")

        assert len(cad.entities(2)) == 1
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_auto_mesh(cad, cell_shape="quad")

    assert backend.model.mesh.generate_calls == [2]
    assert backend.option.values == _AUTO_OPTION_ORIGINALS


def test_auto_mesh_shape_error_reports_aggregated_named_actual_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    _install_backend(monkeypatch, backend)

    with geometry.model("strict-named-diagnostic", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        _set_fake_element_blocks(
            backend,
            2,
            (2, (1, 2)),
            (3, (3,)),
            (2, (4,)),
        )
        with pytest.raises(gmsh_meshing.MeshCellShapeError) as captured:
            _generate_auto_mesh(cad, cell_shape="quad", order=2)

    message = str(captured.value)
    for fragment in (
        "strict-named-diagnostic",
        "AutoMeshSpec",
        "cell_shape='quad'",
        "dimension=2",
        "order=2",
        "Quadrilateral 8",
        "Triangle 3=3",
        "Quadrilateral 4=1",
        "automatic fallback is disabled",
    ):
        assert fragment in message
    assert [
        call[1]
        for call in backend.model.mesh.calls
        if call[0] == "getElementProperties"
    ] == [2, 3]
    assert backend.option.values == _AUTO_OPTION_ORIGINALS


def test_auto_mesh_shape_diagnostic_falls_back_to_unknown_numeric_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("strict-unknown-diagnostic", dimension=3) as cad:
        cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        _set_fake_element_blocks(backend, 3, (99, (1, 2)))
        backend.model.mesh.fail_element_properties.add(99)
        with pytest.raises(
            gmsh_meshing.MeshCellShapeError,
            match=r"Gmsh type 99=2",
        ):
            _generate_auto_mesh(cad, cell_shape="hex")

    assert 99 not in backend.model.mesh.fail_element_properties


def test_auto_mesh_get_elements_failure_preserves_native_error_and_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    backend.model.mesh.fail_get_elements_dimensions.add(2)
    _install_backend(monkeypatch, backend)

    with geometry.model("strict-query-failure", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        with pytest.raises(RuntimeError, match="fake getElements failure"):
            _generate_auto_mesh(cad, cell_shape="tri")
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_mesh(cad, )

    assert backend.model.mesh.generate_calls == [2]
    assert backend.option.values == _AUTO_OPTION_ORIGINALS


def test_auto_mesh_strict_validation_ignores_lower_dimensional_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("strict-top-dimension-only", dimension=3) as cad:
        cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        _set_fake_element_blocks(backend, 2, (2, (20,)), (3, (21,)))
        _set_fake_element_blocks(backend, 3, (17, (30, 31)))
        result = _generate_auto_mesh(cad, cell_shape="hex", order=2)

    assert isinstance(result, gmsh_meshing.GmshMeshRef)
    assert [
        call[1] for call in backend.model.mesh.calls if call[0] == "getElements"
    ] == [3]


@pytest.mark.parametrize("mode", ["point", "background"])
def test_auto_mesh_typed_size_modes_compose_factor_once_and_restore_options(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    _install_backend(monkeypatch, backend)

    with geometry.model(f"auto-typed-size-{mode}", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        point = _fake_entities(cad, backend, 0, 11)[0]
        if mode == "point":
            _mesher(cad).mesh_size([point], size=0.1)
        else:
            distance = _mesher(cad).distance_field(points=[point])
            _mesher(cad).background_field(_fake_threshold(cad, distance))
        _set_fake_element_blocks(backend, 2, (16, (1, 2)))
        backend.option.calls.clear()

        assert isinstance(
            _generate_auto_mesh(cad, level=4, cell_shape="quad", order=2),
            gmsh_meshing.GmshMeshRef,
        )

    requested = _first_requested_options(backend)
    assert requested == {
        "Mesh.ElementOrder": 2.0,
        "Mesh.SecondOrderIncomplete": 1.0,
        "Mesh.RecombineAll": 1.0,
        "Mesh.Algorithm": 6.0,
        "Mesh.RecombinationAlgorithm": 3.0,
        "Mesh.SubdivisionAlgorithm": 0.0,
        "Mesh.MeshSizeFromPoints": 1.0 if mode == "point" else 0.0,
        "Mesh.MeshSizeFromCurvature": 0.0,
        "Mesh.MeshSizeExtendFromBoundary": 1.0 if mode == "point" else 0.0,
        "Mesh.MeshSizeMin": 0.0,
        "Mesh.MeshSizeMax": 1.0e22,
        "Mesh.MeshSizeFactor": pytest.approx(2.0**-0.5),
    }
    get_names = [call[1] for call in backend.option.calls if call[0] == "getNumber"]
    set_names = [call[1] for call in backend.option.calls if call[0] == "setNumber"]
    assert len(get_names) == len(set(get_names))
    assert set(get_names) == set(requested)
    assert all(set_names.count(name) == 2 for name in requested)
    assert backend.option.values == _AUTO_OPTION_ORIGINALS


@pytest.mark.parametrize(
    "failure",
    ["option_get", "option_set", "generate", "strict"],
)
def test_auto_mesh_failures_restore_every_external_option_and_consume_attempt(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    if failure == "option_get":
        backend.option.fail_get_names.add("Mesh.Algorithm")
    elif failure == "option_set":
        backend.option.fail_set_names.add("Mesh.RecombinationAlgorithm")
    elif failure == "generate":
        backend.model.mesh.fail_generate = True
    _install_backend(monkeypatch, backend)

    with geometry.model(f"auto-failure-{failure}", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        element_type = 2 if failure == "strict" else 16
        _set_fake_element_blocks(backend, 2, (element_type, (1, 2)))
        if failure == "strict":
            expected_error: type[BaseException] = gmsh_meshing.MeshCellShapeError
            expected_message = "Triangle 3=2"
        else:
            expected_error = RuntimeError
            expected_message = {
                "option_get": "option get failure",
                "option_set": "Mesh.RecombinationAlgorithm",
                "generate": "fake mesh failure",
            }[failure]

        with pytest.raises(expected_error, match=expected_message):
            _generate_auto_mesh(cad, cell_shape="quad", order=2)

        assert backend.option.values == _AUTO_OPTION_ORIGINALS
        assert len(cad.entities(2)) == 1
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_auto_mesh(cad, cell_shape="quad")

def test_auto_mesh_successful_generation_restoration_failure_is_retried_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    backend.option.fail_set_after["Mesh.Algorithm"] = 1
    _install_backend(monkeypatch, backend)

    with geometry.model("auto-restore-failure", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        _set_fake_element_blocks(backend, 2, (3, (1, 2)))
        with pytest.raises(
            geometry.GeometryError,
            match="restoring global Gmsh options failed",
        ):
            _generate_auto_mesh(cad, cell_shape="quad")

        assert backend.option.values["Mesh.Algorithm"] == 6.0
        with pytest.raises(geometry.GeometryStateError, match="MESH_FAILED"):
            _generate_auto_mesh(cad, cell_shape="quad")

    assert backend.option.values == _AUTO_OPTION_ORIGINALS


def test_auto_mesh_shape_failure_preserves_error_when_restoration_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    backend.option.fail_set_after["Mesh.Algorithm"] = 1
    _install_backend(monkeypatch, backend)

    with geometry.model("auto-shape-and-restore-failure", dimension=2) as cad:
        cad.rectangle(0.0, 0.0, 1.0, 1.0)
        _set_fake_element_blocks(backend, 2, (2, (1,)))
        with pytest.raises(gmsh_meshing.MeshCellShapeError) as captured:
            _generate_auto_mesh(cad, cell_shape="quad")

        assert any(
            "additionally failed to restore" in note
            for note in getattr(captured.value, "__notes__", ())
        )

    assert backend.option.values == _AUTO_OPTION_ORIGINALS


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


def test_auto_mesh_missing_top_entity_keeps_mesher_retryable_and_seals_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)

    with geometry.model("auto-missing-topology", dimension=2) as cad:
        with pytest.raises(ValueError, match="top-dimensional"):
            _generate_auto_mesh(cad, )
        assert backend.model.mesh.calls == []
        assert backend.option.calls == []

        with pytest.raises(geometry.GeometryStateError, match="CONFIGURING_MESH"):
            cad.rectangle(0.0, 0.0, 1.0, 1.0)
        with pytest.raises(ValueError, match="top-dimensional"):
            _generate_auto_mesh(cad, )


def test_nested_auto_mesh_models_isolate_current_model_policy_and_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeGmsh(
        initialized=True,
        names=("external",),
        current="external",
    )
    backend.option.values.update(_AUTO_OPTION_ORIGINALS)
    _install_backend(monkeypatch, backend)

    with geometry.model("outer-auto", dimension=2) as outer:
        outer.rectangle(0.0, 0.0, 1.0, 1.0)
        _set_fake_element_blocks(backend, 2, (3, (1, 2)))

        with geometry.model("inner-auto", dimension=3) as inner:
            inner.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            _set_fake_element_blocks(backend, 3, (4, (3, 4)))
            _generate_auto_mesh(inner, level=1, cell_shape="tet")
            assert backend.option.values == _AUTO_OPTION_ORIGINALS

        assert backend.model.current == "outer-auto"
        _generate_auto_mesh(outer, level=5, cell_shape="quad")
        assert backend.option.values == _AUTO_OPTION_ORIGINALS

    assert backend.model.current == "external"
    assert backend.model.mesh.generate_calls == [3, 2]


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
        "Mesh.Algorithm",
        "Mesh.Algorithm3D",
        "Mesh.RecombinationAlgorithm",
        "Mesh.Recombine3DAll",
        "Mesh.SubdivisionAlgorithm",
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
