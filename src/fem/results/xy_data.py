"""Typed XY data extracted from one exact result location across frames."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real

from .data import FieldState
from .fields import ResultSourceKey, ScalarFieldSelection
from .frames import ResultFrameCatalog
from .probe import (
    ResultProbeRequest,
    ResultProbeTarget,
    probe_result_from_query_result,
    result_query_for_probe,
)
from .provider import ResultProvider


class ResultXYAxis(str, Enum):
    """Supported independent variables for increment-history curves."""

    INCREMENT = "increment"
    STEP_TIME = "step_time"
    TOTAL_TIME = "total_time"
    LOAD_FACTOR = "load_factor"


class ResultXYValidationError(ValueError):
    """Typed validation failure while building one XY series."""


@dataclass(frozen=True, slots=True)
class ResultXYRequest:
    """One scalar result selection and one exact FEM location."""

    selection: ScalarFieldSelection
    target: ResultProbeTarget
    axis: ResultXYAxis = ResultXYAxis.INCREMENT

    def __post_init__(self) -> None:
        if type(self.selection) is not ScalarFieldSelection:
            raise TypeError("selection must be ScalarFieldSelection")
        if type(self.target) is not ResultProbeTarget:
            raise TypeError("target must be ResultProbeTarget")
        if type(self.axis) is not ResultXYAxis:
            raise TypeError("axis must be ResultXYAxis")

    @property
    def probe_request(self) -> ResultProbeRequest:
        """Return the equivalent exact probe request."""

        return ResultProbeRequest(self.selection, self.target)


@dataclass(frozen=True, slots=True)
class ResultXYPoint:
    """One finite point in an increment-history curve."""

    frame_index: int
    x_value: float
    y_value: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be an integer >= 0")
        object.__setattr__(
            self,
            "x_value",
            _finite_real(self.x_value, label="x_value"),
        )
        object.__setattr__(
            self,
            "y_value",
            _finite_real(self.y_value, label="y_value"),
        )


@dataclass(frozen=True, slots=True)
class ResultXYSeries:
    """One typed XY curve bound to a result source and exact request."""

    source: ResultSourceKey
    request: ResultXYRequest
    points: tuple[ResultXYPoint, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(self.request) is not ResultXYRequest:
            raise TypeError("request must be ResultXYRequest")
        if type(self.points) is not tuple:
            raise TypeError("points must be a tuple")
        if not self.points:
            raise ValueError("XY series must contain at least one point")
        if any(type(point) is not ResultXYPoint for point in self.points):
            raise TypeError("points must contain ResultXYPoint values")
        frame_indices = tuple(point.frame_index for point in self.points)
        if len(set(frame_indices)) != len(frame_indices):
            raise ValueError("XY series frame indices must be unique")
        if frame_indices != tuple(sorted(frame_indices)):
            raise ValueError("XY series points must be sorted by frame_index")


def build_result_xy_series(
    frame_providers: Iterable[tuple[int, ResultProvider]],
    request: ResultXYRequest,
    *,
    frame_catalog: ResultFrameCatalog | None = None,
) -> ResultXYSeries:
    """Extract one exact scalar value from each supplied result frame.

    The caller owns frame construction and may materialize lazy fields before
    passing providers here.  This function only reads accepted immutable
    providers and reuses the existing Probe filtering semantics, so element
    nodal, integration-point, and duplicate region records are never silently
    collapsed into an arbitrary value.
    """

    if type(request) is not ResultXYRequest:
        raise TypeError("request must be ResultXYRequest")
    try:
        items = tuple(frame_providers)
    except TypeError as error:
        raise TypeError(
            "frame_providers must be an iterable of (frame_index, provider)"
        ) from error
    if not items:
        raise ResultXYValidationError("至少需要一个结果增量才能生成 XY 曲线")
    if frame_catalog is not None:
        if type(frame_catalog) is not ResultFrameCatalog:
            raise TypeError("frame_catalog must be ResultFrameCatalog or None")

    source: ResultSourceKey | None = None
    points: list[ResultXYPoint] = []
    probe_request = request.probe_request
    query = result_query_for_probe(probe_request)
    previous_frame = -1
    for frame_index, provider in items:
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise TypeError("frame indices must be integers >= 0")
        if type(provider) is not ResultProvider:
            raise TypeError("frame providers must contain ResultProvider values")
        if frame_index <= previous_frame:
            raise ResultXYValidationError(
                "结果帧必须按增量编号递增且不能重复"
            )
        previous_frame = frame_index
        if source is None:
            source = provider.source
        elif provider.source != source:
            raise ResultXYValidationError("XY 曲线不能混用不同结果源")
        if provider.frame_key is not None:
            if provider.frame_key.frame_index != frame_index:
                raise ResultXYValidationError(
                    f"结果帧身份不一致：期望增量 {frame_index}"
                )
            if provider.frame_key.source != provider.source:
                raise ResultXYValidationError("结果帧来源与 Provider 不一致")

        try:
            availability = provider.validate_query(query)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ResultXYValidationError(
                f"增量 {frame_index} 的 XY 查询无效：{error}"
            ) from error
        if availability.state is not FieldState.READY:
            raise ResultXYValidationError(
                f"增量 {frame_index} 的场变量尚未就绪，无法生成 XY 曲线"
            )
        try:
            query_result = provider.query(query)
            probe_result = probe_result_from_query_result(
                query_result,
                probe_request,
                frame_key=provider.frame_key,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ResultXYValidationError(
                f"增量 {frame_index} 的 XY 查询失败：{error}"
            ) from error
        if len(probe_result.records) != 1:
            raise ResultXYValidationError(
                f"增量 {frame_index} 的目标位置命中 {len(probe_result.records)} "
                "条记录，无法形成唯一曲线值"
            )
        points.append(
            ResultXYPoint(
                frame_index,
                _x_value(
                    request.axis,
                    frame_index,
                    frame_catalog,
                ),
                probe_result.records[0].value,
            )
        )

    if source is None:
        raise ResultXYValidationError("没有可用的结果源")
    return ResultXYSeries(source, request, tuple(points))


def _x_value(
    axis: ResultXYAxis,
    frame_index: int,
    frame_catalog: ResultFrameCatalog | None,
) -> float:
    if axis is ResultXYAxis.INCREMENT:
        return float(frame_index)
    if frame_catalog is None:
        raise ResultXYValidationError(
            f"横轴 {axis.value} 需要结果帧元数据"
        )
    try:
        metadata = frame_catalog.metadata_for(frame_index)
    except KeyError as error:
        raise ResultXYValidationError(
            f"结果帧元数据缺少增量 {frame_index}"
        ) from error
    value = (
        metadata.step_time
        if axis is ResultXYAxis.STEP_TIME
        else metadata.total_time
        if axis is ResultXYAxis.TOTAL_TIME
        else metadata.load_factor
    )
    if value is None:
        label = {
            ResultXYAxis.STEP_TIME: "步时间",
            ResultXYAxis.TOTAL_TIME: "总时间",
            ResultXYAxis.LOAD_FACTOR: "载荷因子",
        }[axis]
        raise ResultXYValidationError(
            f"增量 {frame_index} 没有可用的{label}"
        )
    return _finite_real(value, label=f"{axis.value} value")


def _finite_real(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


__all__ = [
    "ResultXYAxis",
    "ResultXYPoint",
    "ResultXYRequest",
    "ResultXYSeries",
    "ResultXYValidationError",
    "build_result_xy_series",
]
