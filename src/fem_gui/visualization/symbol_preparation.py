"""Camera-independent interpretation and sampling of model symbols."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from fem.application import RegionRef
from fem.application.beam_frames import BeamFrameReport
from fem.boundary.condition import BoundaryCondition
from fem.boundary.step import boundary_for_step, effective_step_boundaries, get_step
from fem.core._constraint_targets import displacement_target_kind, resolve_displacement_node_ids
from fem.core.model import FEMModel
from fem.elements import BeamFrame

from .model_adapter import ArchiveModelView, ModelGeometry
from .symbols import (
    SymbolSettings,
    constraint_sample_indices,
    constraint_spatial_regions,
    face_display_node_ids,
    sample_distributed_faces,
    sample_distributed_polyline,
    symbol_length,
)


BEAM_FRAME_CACHE_LIMIT = 256


@dataclass(frozen=True, slots=True)
class ConstraintSample:
    node_id: int
    point: tuple[float, float, float]
    components: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class SymbolRenderData:
    """Prepared symbol anchors use the original model geometry."""

    reference_length: float
    model_center: tuple[float, float, float]
    is_3d: bool
    constraints: tuple[ConstraintSample, ...]
    load_anchors: np.ndarray
    load_vectors: np.ndarray
    load_starts_at_anchor: tuple[bool, ...]
    load_labels: tuple[str, ...]
    moment_points: np.ndarray
    moment_axes: np.ndarray
    moment_labels: tuple[str, ...]
    face_cells: tuple[int, ...]
    edge_lines: tuple[int, ...]
    gravity: tuple[float, float, float]


def _point_array(values: Sequence[np.ndarray]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape((-1, 3))
    array.setflags(write=False)
    return array


def effective_line_load_vector(
    vector: object,
    coordinate_system: str,
    frame: BeamFrame | None,
) -> np.ndarray | None:
    """Project local components with the application-resolved Beam rotation."""

    values = np.asarray(vector, dtype=float)
    if coordinate_system != "local":
        return values
    if frame is None:
        return None
    return np.asarray(frame.rotation, dtype=float).T @ values


class SymbolPreparation:
    """Own one model binding, expanded boundaries and the latest sampled data."""

    def __init__(self) -> None:
        self._model: FEMModel | ArchiveModelView | None = None
        self._geometry: ModelGeometry | None = None
        self._frame_query: Callable[[RegionRef | int], BeamFrameReport | None] | None = None
        self._boundary_cache: dict[str | None, BoundaryCondition] = {}
        self._frame_cache: dict[RegionRef | int, BeamFrameReport | None] = {}
        self._constraint_sample_node_ids: dict[
            tuple[tuple[str, str | int], str], tuple[int, ...]
        ] = {}
        self._reference_length: float | None = None
        self._model_center: tuple[float, float, float] | None = None
        self._constraint_reference_axis_cache: np.ndarray | None = None
        self._element_lookup_cache: dict[int, Any] | None = None
        self._prepared_key: tuple[object, ...] | None = None
        self._prepared_data: SymbolRenderData | None = None

    def bind(
        self, model: FEMModel | ArchiveModelView, geometry: ModelGeometry,
        frame_query: Callable[[RegionRef | int], BeamFrameReport | None] | None = None,
    ) -> None:
        self.clear()
        self._model = model
        self._geometry = geometry
        self._frame_query = frame_query

    def clear(self) -> None:
        self._model = None
        self._geometry = None
        self._frame_query = None
        self.invalidate_definitions()
        self._reference_length = None
        self._model_center = None
        self._constraint_reference_axis_cache = None

    def invalidate_definitions(self) -> None:
        self._boundary_cache.clear()
        self._constraint_sample_node_ids.clear()
        self._frame_cache.clear()
        self._element_lookup_cache = None
        self._prepared_key = None
        self._prepared_data = None

    def reference_length(self) -> float:
        if self._geometry is None:
            return 1.0e-9
        if self._reference_length is None:
            self._reference_length = symbol_length(self._geometry.points)
        return self._reference_length

    def model_center(self) -> tuple[float, float, float]:
        if self._geometry is None:
            raise RuntimeError("symbol preparation has no model binding")
        if self._model_center is None:
            center = 0.5 * (np.min(self._geometry.points, axis=0) + np.max(self._geometry.points, axis=0))
            self._model_center = tuple(float(value) for value in center)
        return self._model_center

    def _constraint_reference_axis(self) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("symbol preparation has no model binding")
        if self._constraint_reference_axis_cache is None:
            points = np.asarray(self._geometry.points, dtype=float)
            centered = points - np.mean(points, axis=0)
            _left, _singular_values, right = np.linalg.svd(centered, full_matrices=False)
            self._constraint_reference_axis_cache = np.asarray(right[0], dtype=float)
        return self._constraint_reference_axis_cache

    def element_lookup(self) -> dict[int, Any]:
        if self._model is None:
            return {}
        if self._element_lookup_cache is None:
            self._element_lookup_cache = {
                int(element.id): element for element in self._model.mesh.elements
            }
        return self._element_lookup_cache

    def frame_report(self, target: RegionRef | int) -> BeamFrameReport | None:
        if target in self._frame_cache:
            return self._frame_cache[target]
        if self._frame_query is None:
            return None
        report = self._frame_query(target)
        if len(self._frame_cache) >= BEAM_FRAME_CACHE_LIMIT:
            self._frame_cache.pop(next(iter(self._frame_cache)))
        self._frame_cache[target] = report
        return report

    def prepare(
        self, step_name: str | None, settings: SymbolSettings, sampling_density: str,
    ) -> SymbolRenderData | None:
        if self._model is None or self._geometry is None:
            raise RuntimeError("symbol preparation has no model binding")
        key = (
            step_name, sampling_density, settings.show_constraints, settings.show_nodal_loads,
            settings.show_edge_loads, settings.show_surface_loads, settings.show_line_loads,
        )
        if self._prepared_key == key and self._prepared_data is not None:
            return self._prepared_data
        if step_name not in self._boundary_cache:
            try:
                self._boundary_cache[step_name] = boundary_for_step(self._model, step_name)
            except Exception:
                return None
        boundary = self._boundary_cache[step_name]
        selected_definition = get_step(self._model, step_name)
        gravity_vector = np.zeros(3, dtype=float)
        if selected_definition is not None:
            for gravity_load in selected_definition.gravity_loads:
                acceleration = np.asarray(gravity_load.acceleration, dtype=float).reshape(-1)
                gravity_vector[:min(3, len(acceleration))] += acceleration[:3]
        is_3d = bool(self._model.mesh.nodes and hasattr(self._model.mesh.nodes[0], "z"))
        translation_count = 3 if is_3d else 2
        constraint_samples: list[ConstraintSample] = []
        if settings.show_constraints:
            boundary_definitions = effective_step_boundaries(
                self._model,
                selected_definition,
            )
            constraints_by_target: dict[
                tuple[str, str | int],
                dict[int, float],
            ] = {}
            representative_by_target: dict[
                tuple[str, str | int],
                object,
            ] = {}
            for definition in boundary_definitions:
                target_key = (
                    displacement_target_kind(definition),
                    definition.target,
                )
                representative_by_target[target_key] = definition
                components = constraints_by_target.setdefault(target_key, {})
                for component in range(definition.first_component - 1, definition.last_component):
                    components[component] = float(definition.value)
            for target_key, components in constraints_by_target.items():
                sample_key = (target_key, sampling_density)
                sampled_node_ids = self._constraint_sample_node_ids.get(sample_key)
                if sampled_node_ids is None:
                    try:
                        node_ids = resolve_displacement_node_ids(
                            self._model,
                            representative_by_target[target_key],
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
                    target_points = np.asarray([
                        self._geometry.points[
                            self._geometry.node_id_to_point_index[node_id]
                        ]
                        for node_id in node_ids
                    ])
                    regions = constraint_spatial_regions(
                        target_points,
                        self._geometry.points,
                        reference_axis=self._constraint_reference_axis(),
                    )
                    sampled: list[int] = []
                    for region_indices in regions:
                        region_node_ids = tuple(
                            node_ids[int(index)] for index in region_indices
                        )
                        selected = constraint_sample_indices(
                            target_points[region_indices],
                            sampling_density,
                        )
                        sampled.extend(
                            region_node_ids[int(index)] for index in selected
                        )
                    sampled_node_ids = tuple(sampled)
                    self._constraint_sample_node_ids[sample_key] = sampled_node_ids
                for node_id in sampled_node_ids:
                    constraint_samples.append(ConstraintSample(
                        int(node_id),
                        tuple(float(value) for value in self._geometry.points[
                            self._geometry.node_id_to_point_index[node_id]
                        ]),
                        tuple(sorted(components.items())),
                    ))
        origins: list[np.ndarray] = []
        vectors: list[np.ndarray] = []
        load_starts_at_anchor: list[bool] = []
        load_labels: list[str] = []
        moment_points: list[np.ndarray] = []
        moment_labels: list[str] = []
        if settings.show_nodal_loads:
            nodal_vectors: dict[int, np.ndarray] = {}
            nodal_moments: dict[int, np.ndarray] = {}
            for dof, value in boundary.nodal_forces.items():
                node_index, component = divmod(int(dof), self._model.mesh.dofs_per_node)
                node_id = self._model.mesh.dof_map.node_ids[node_index]
                if component < translation_count:
                    nodal_vectors.setdefault(node_id, np.zeros(3))[component] += float(value)
                else:
                    axis = component - translation_count if is_3d else 2
                    nodal_moments.setdefault(node_id, np.zeros(3))[axis] += float(value)
            for node_id, vector in nodal_vectors.items():
                if float(np.linalg.norm(vector)) <= 0.0:
                    continue
                base = self._geometry.points[self._geometry.node_id_to_point_index[node_id]]
                origins.append(base)
                vectors.append(vector)
                load_starts_at_anchor.append(False)
                load_labels.append(f"F={tuple(float(value) for value in vector[:translation_count])}")
            for node_id, moment in nodal_moments.items():
                if float(np.linalg.norm(moment)) <= 0.0:
                    continue
                moment_points.append(self._geometry.points[self._geometry.node_id_to_point_index[node_id]])
                moment_labels.append(
                    "M=" + str(tuple(float(value) for value in moment))
                )
        face_cells: list[int] = []
        edge_lines: list[int] = []

        def node_point(node_id: int) -> np.ndarray:
            return self._geometry.points[
                self._geometry.node_id_to_point_index[int(node_id)]
            ]

        def add_distributed_group(definition, members, tractions, kind: str) -> None:
            member_points: list[np.ndarray] = []
            member_vectors: list[np.ndarray] = []
            for member, traction in zip(members, tractions):
                ids = member.node_ids
                points = np.asarray([node_point(node_id) for node_id in ids])
                vector = np.pad(
                    np.asarray(traction.vector, dtype=float),
                    (0, max(0, 3 - len(traction.vector))),
                )[:3]
                if kind == "face":
                    display_ids = face_display_node_ids(tuple(ids))
                    points = np.asarray([
                        node_point(node_id) for node_id in display_ids
                    ])
                member_points.append(points)
                member_vectors.append(vector)
                indices = [self._geometry.node_id_to_point_index[int(node_id)] for node_id in ids]
                target = face_cells if kind == "face" else edge_lines
                target.extend((len(indices), *indices))
            if not member_points:
                return
            if kind == "face":
                sampled_points, sampled_vectors = sample_distributed_faces(
                    member_points,
                    member_vectors,
                    sampling_density,
                )
            else:
                sampled_points, sampled_vectors = sample_distributed_polyline(
                    member_points,
                    member_vectors,
                    sampling_density,
                )
            if len(sampled_points) == 0:
                return
            label_index = int(np.argmin(np.linalg.norm(
                sampled_points - np.mean(sampled_points, axis=0), axis=1
            )))
            for index, (sample, vector) in enumerate(
                zip(sampled_points, sampled_vectors)
            ):
                magnitude = float(np.linalg.norm(vector))
                if magnitude <= 0.0:
                    continue
                origins.append(sample)
                vectors.append(vector)
                load_starts_at_anchor.append(kind == "edge")
                prefix = "P" if definition.load_type == "pressure" else "T"
                load_labels.append(
                    f"{prefix}={magnitude:.6g}" if index == label_index else ""
                )

        if selected_definition is not None:
            surface_offset = 0
            if settings.show_surface_loads:
                for definition in selected_definition.surface_loads:
                    members = self._model.surfaces[definition.surface].faces
                    count = len(members)
                    add_distributed_group(
                        definition, members,
                        boundary.surface_tractions[surface_offset:surface_offset + count],
                        "face",
                    )
                    surface_offset += count
            edge_offset = 0
            if settings.show_edge_loads:
                for definition in selected_definition.edge_loads:
                    members = self._model.edges[definition.edge].edges
                    count = len(members)
                    add_distributed_group(
                        definition, members,
                        boundary.edge_tractions[edge_offset:edge_offset + count],
                        "edge",
                    )
                    edge_offset += count
            if settings.show_line_loads:
                element_lookup = self.element_lookup()
                frame_by_element: dict[int, BeamFrame] = {}
                for definition in selected_definition.line_loads:
                    if definition.coordinate_system != "local":
                        continue
                    frame_target = (
                        int(definition.target)
                        if isinstance(definition.target, int)
                        else RegionRef(
                            "element_set",
                            str(definition.target),
                        )
                    )
                    report = self.frame_report(
                        frame_target
                    )
                    for entry in tuple(
                        getattr(report, "entries", ())
                    ):
                        frame_by_element[int(entry.element_id)] = (
                            entry.frame
                        )
                element_sets = dict(
                    getattr(self._model, "metadata", {}).get(
                        "_abaqus_internal_element_sets",
                        {},
                    )
                )
                element_sets.update(getattr(self._model, "element_sets", {}))
                line_offset = 0
                for definition in selected_definition.line_loads:
                    target_ids = (
                        tuple(element_sets[str(definition.target)].element_ids)
                        if isinstance(definition.target, str)
                        else (int(definition.target),)
                    )
                    count = len(target_ids)
                    group_loads = boundary.line_loads[
                        line_offset:line_offset + count
                    ]
                    line_offset += count
                    member_points: list[np.ndarray] = []
                    member_vectors: list[np.ndarray] = []
                    label_vector: tuple[float, ...] | None = None
                    for load in group_loads:
                        element = element_lookup.get(int(load.elem_id))
                        if element is None:
                            continue
                        points = np.asarray([
                            node_point(node_id)
                            for node_id in element.node_ids
                        ])
                        vector = effective_line_load_vector(
                            load.vector,
                            load.coordinate_system,
                            frame_by_element.get(int(load.elem_id)),
                        )
                        if vector is None or float(np.linalg.norm(vector)) <= 0.0:
                            continue
                        member_points.append(points)
                        member_vectors.append(vector)
                        label_vector = tuple(float(value) for value in load.vector)
                        indices = [
                            self._geometry.node_id_to_point_index[int(node_id)]
                            for node_id in element.node_ids
                        ]
                        edge_lines.extend((len(indices), *indices))
                    samples, sample_vectors = sample_distributed_polyline(
                        member_points,
                        member_vectors,
                        sampling_density,
                    )
                    if len(samples) == 0:
                        continue
                    label_index = int(np.argmin(np.linalg.norm(
                        samples - np.mean(samples, axis=0), axis=1
                    )))
                    for sample_index, (sample, vector) in enumerate(
                        zip(samples, sample_vectors)
                    ):
                        origins.append(sample)
                        vectors.append(vector)
                        load_starts_at_anchor.append(False)
                        load_labels.append(
                            f"q={label_vector}"
                            if sample_index == label_index
                            else ""
                        )

        moment_axes = (
            [moment / np.linalg.norm(moment) for moment in nodal_moments.values() if np.linalg.norm(moment) > 0.0]
            if moment_points else []
        )
        data = SymbolRenderData(
            self.reference_length(), self.model_center(), is_3d, tuple(constraint_samples),
            _point_array(origins), _point_array(vectors), tuple(load_starts_at_anchor), tuple(load_labels),
            _point_array(moment_points), _point_array(moment_axes), tuple(moment_labels),
            tuple(face_cells), tuple(edge_lines), tuple(float(value) for value in gravity_vector),
        )
        self._prepared_key = key
        self._prepared_data = data
        return data
