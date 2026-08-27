"""Deterministic operations on retained result XY series.

The operations live beside the typed XY data contract so the GUI only
coordinates selection and presentation.  They never mutate an existing
series and therefore remain safe to reuse from exports or future automation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .xy_data import ResultXYPoint, ResultXYSeries, ResultXYValidationError


def _validate_series(series: ResultXYSeries) -> ResultXYSeries:
    if type(series) is not ResultXYSeries:
        raise TypeError("series must be ResultXYSeries")
    return series


def _series_from_values(
    series: ResultXYSeries,
    values: np.ndarray,
) -> ResultXYSeries:
    source = _validate_series(series)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(source.points),) or not np.all(np.isfinite(values)):
        raise ResultXYValidationError("XY 运算结果必须是一维有限数值")
    return replace(
        source,
        points=tuple(
            ResultXYPoint(point.frame_index, point.x_value, value)
            for point, value in zip(source.points, values, strict=True)
        ),
    )


def difference(
    left: ResultXYSeries,
    right: ResultXYSeries,
) -> ResultXYSeries:
    """Return ``left - right`` after matching frame/x identities."""

    left = _validate_series(left)
    right = _validate_series(right)
    if left.source != right.source:
        raise ResultXYValidationError("XY 差值不能混用不同结果源")
    if len(left.points) != len(right.points):
        raise ResultXYValidationError("XY 差值要求两条曲线点数相同")
    if any(
        a.frame_index != b.frame_index or not np.isclose(a.x_value, b.x_value)
        for a, b in zip(left.points, right.points, strict=True)
    ):
        raise ResultXYValidationError("XY 差值要求两条曲线的横轴一致")
    return _series_from_values(
        left,
        np.asarray([a.y_value - b.y_value for a, b in zip(left.points, right.points)], dtype=float),
    )


def derivative(series: ResultXYSeries) -> ResultXYSeries:
    """Return the numerical derivative ``dy/dx`` using finite differences."""

    series = _validate_series(series)
    x = np.asarray([point.x_value for point in series.points], dtype=float)
    y = np.asarray([point.y_value for point in series.points], dtype=float)
    if len(series.points) == 1:
        values = np.zeros(1, dtype=float)
    else:
        dx = np.diff(x)
        if np.any(np.isclose(dx, 0.0)):
            raise ResultXYValidationError("XY 导数要求横轴不能重复")
        values = np.gradient(y, x)
    return _series_from_values(series, values)


def integral(series: ResultXYSeries) -> ResultXYSeries:
    """Return the cumulative trapezoidal integral from the first point."""

    series = _validate_series(series)
    x = np.asarray([point.x_value for point in series.points], dtype=float)
    y = np.asarray([point.y_value for point in series.points], dtype=float)
    values = np.zeros(len(series.points), dtype=float)
    if len(series.points) > 1:
        dx = np.diff(x)
        if np.any(np.isclose(dx, 0.0)):
            raise ResultXYValidationError("XY 积分要求横轴不能重复")
        values[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return _series_from_values(series, values)


def envelope(series: ResultXYSeries) -> ResultXYSeries:
    """Return the running maximum absolute envelope of one curve."""

    series = _validate_series(series)
    values = np.maximum.accumulate(
        np.abs(np.asarray([point.y_value for point in series.points], dtype=float))
    )
    return _series_from_values(series, values)


__all__ = ["difference", "derivative", "envelope", "integral"]
