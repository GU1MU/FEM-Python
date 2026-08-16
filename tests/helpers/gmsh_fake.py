from __future__ import annotations

import math
from typing import Any, Sequence

import pytest

from fem import geometry
from fem.geometry._gmsh import backend as _gmsh_backend
from fem.mesh import gmsh as gmsh_meshing

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
        self.get_entities_calls = 0
        self.calls: list[tuple[Any, ...]] = []
        self.distance_results: dict[tuple[int, int, int, int], Any] = {}
        self.distance_failures: dict[
            tuple[int, int, int, int], BaseException
        ] = {}
        self.distance_default: Any = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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
        self.get_entities_calls += 1
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

    def getDistance(
        self,
        left_dimension: int,
        left_tag: int,
        right_dimension: int,
        right_tag: int,
    ) -> Any:
        key = (left_dimension, left_tag, right_dimension, right_tag)
        self.calls.append(("getDistance", *key))
        failure = self.distance_failures.get(key)
        if failure is not None:
            raise failure
        return self.distance_results.get(key, self.distance_default)

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
            "attributes": {},
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

    def getAttribute(self, name: str) -> list[str]:
        return list(self._current_data()["attributes"].get(name, ()))

    def setAttribute(self, name: str, values: Sequence[str]) -> None:
        self._current_data()["attributes"][name] = [
            str(item) for item in values
        ]

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
        # Console verbosity is a session-startup baseline, not a mesh option,
        # so it stays out of the recorded option log and stored values.
        if name == "General.Terminal":
            return
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

    def initialize(self, *, interruptible: bool = True) -> None:
        del interruptible
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
