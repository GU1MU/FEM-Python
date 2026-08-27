"""Spatial result-path extraction over one accepted result frame."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from .data import FieldLocation, FieldState
from .fields import ResultSourceKey, ScalarFieldSelection
from .provider import ResultProvider


@dataclass(frozen=True, slots=True)
class ResultPathRequest:
    """One straight or mesh-edge path sampled from a result field."""

    selection: ScalarFieldSelection
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    sample_count: int = 20
    edge_node_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.selection) is not ScalarFieldSelection:
            raise TypeError("selection must be ScalarFieldSelection")
        object.__setattr__(self, "start", _finite_triplet(self.start, "start"))
        object.__setattr__(self, "end", _finite_triplet(self.end, "end"))
        if type(self.sample_count) is not int or self.sample_count < 2:
            raise ValueError("sample_count must be an integer >= 2")
        if type(self.edge_node_ids) is not tuple:
            raise TypeError("edge_node_ids must be a tuple")
        if any(type(value) is not int or value <= 0 for value in self.edge_node_ids):
            raise ValueError("edge_node_ids must contain positive integers")
        if len(set(self.edge_node_ids)) != len(self.edge_node_ids):
            raise ValueError("edge_node_ids must not contain duplicates")


@dataclass(frozen=True, slots=True)
class ResultPathSample:
    """One spatial path sample and the stored FEM row used for its value."""

    distance: float
    coordinates: tuple[float, float, float]
    value: float
    source_location: FieldLocation

    def __post_init__(self) -> None:
        object.__setattr__(self, "distance", _finite_real(self.distance, "distance"))
        object.__setattr__(
            self,
            "coordinates",
            _finite_triplet(self.coordinates, "coordinates"),
        )
        object.__setattr__(self, "value", _finite_real(self.value, "value"))
        if type(self.source_location) is not FieldLocation:
            raise TypeError("source_location must be FieldLocation")


@dataclass(frozen=True, slots=True)
class ResultPathResult:
    """Accepted, immutable spatial samples bound to one result source."""

    source: ResultSourceKey
    materialization_generation: int
    request: ResultPathRequest
    samples: tuple[ResultPathSample, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(self.materialization_generation) is not int or self.materialization_generation < 0:
            raise ValueError("materialization_generation must be an integer >= 0")
        if type(self.request) is not ResultPathRequest:
            raise TypeError("request must be ResultPathRequest")
        if type(self.samples) is not tuple or len(self.samples) < 2:
            raise ValueError("samples must contain at least two points")
        if any(type(sample) is not ResultPathSample for sample in self.samples):
            raise TypeError("samples must contain ResultPathSample values")
        distances = tuple(sample.distance for sample in self.samples)
        if distances != tuple(sorted(distances)) or len(set(distances)) != len(distances):
            raise ValueError("path sample distances must be strictly increasing")


def build_result_path_result(
    provider: ResultProvider,
    request: ResultPathRequest,
) -> ResultPathResult:
    """Sample the nearest stored FEM result row at each path position.

    The result layer deliberately reports the source location used for each
    sample.  It never silently interpolates or averages across incompatible
    element/integration-point records; callers can therefore see whether a
    coarse path reused a nearby stored result row.
    """

    if type(provider) is not ResultProvider:
        raise TypeError("provider must be ResultProvider")
    if type(request) is not ResultPathRequest:
        raise TypeError("request must be ResultPathRequest")
    availability = provider.field_status(request.selection.field_key)
    if availability.state is not FieldState.READY:
        raise ValueError("路径提取要求场变量已经物化")
    field_data = provider.field(request.selection.field_key)
    try:
        component_index = field_data.descriptor.columns.index(
            request.selection.component
        )
    except ValueError as error:
        raise ValueError(
            f"路径场变量没有分量 {request.selection.component!r}"
        ) from error
    if not field_data.locations:
        raise ValueError("当前场变量没有可用于路径提取的结果位置")

    coordinates = np.asarray(
        [location.coordinates for location in field_data.locations],
        dtype=float,
    )
    values = np.asarray(field_data.values[:, component_index], dtype=float)
    path_points = _path_points(provider, request)
    samples: list[ResultPathSample] = []
    for distance, point in path_points:
        nearest = int(np.argmin(np.linalg.norm(coordinates - point, axis=1)))
        samples.append(
            ResultPathSample(
                distance,
                tuple(float(value) for value in point),
                float(values[nearest]),
                field_data.locations[nearest],
            )
        )
    return ResultPathResult(
        provider.source,
        provider.snapshot.generation,
        request,
        tuple(samples),
    )


def _path_points(
    provider: ResultProvider,
    request: ResultPathRequest,
) -> tuple[tuple[float, np.ndarray], ...]:
    if not request.edge_node_ids:
        start = np.asarray(request.start, dtype=float)
        end = np.asarray(request.end, dtype=float)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 1.0e-14:
            raise ValueError("路径起点和终点不能重合")
        parameters = np.linspace(0.0, 1.0, request.sample_count)
        return tuple(
            (float(parameter * length), start + parameter * vector)
            for parameter in parameters
        )

    node_coordinates = {
        int(node_id): np.asarray(coordinate, dtype=float)
        for node_id, coordinate in zip(
            provider.snapshot.topology.node_ids,
            provider.snapshot.topology.node_coordinates,
            strict=True,
        )
    }
    try:
        vertices = [node_coordinates[node_id] for node_id in request.edge_node_ids]
    except KeyError as error:
        raise ValueError(f"路径边引用了未知节点 {error.args[0]}") from error
    lengths = np.asarray(
        [float(np.linalg.norm(end - start)) for start, end in zip(vertices, vertices[1:])],
        dtype=float,
    )
    if not len(lengths) or np.any(lengths <= 1.0e-14):
        raise ValueError("路径边不能包含零长度线段")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    distances = np.linspace(0.0, total, request.sample_count)
    points: list[tuple[float, np.ndarray]] = []
    for index, distance in enumerate(distances):
        if index == 0:
            points.append((0.0, vertices[0].copy()))
            continue
        if index == len(distances) - 1:
            points.append((total, vertices[-1].copy()))
            continue
        segment = min(int(np.searchsorted(cumulative, distance, side="right")) - 1, len(lengths) - 1)
        fraction = (distance - cumulative[segment]) / lengths[segment]
        point = vertices[segment] + fraction * (vertices[segment + 1] - vertices[segment])
        points.append((float(distance), point))
    return tuple(points)


def _finite_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _finite_triplet(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    return tuple(_finite_real(item, f"{label}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


__all__ = [
    "ResultPathRequest",
    "ResultPathResult",
    "ResultPathSample",
    "build_result_path_result",
]
