"""First-class frame identities used during the result-layer migration.

The existing solver result objects remain compatible.  These contracts give
application and GUI consumers a stable identity for one output frame without
requiring the current provider implementation to be replaced in one step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from .fields import ResultSourceKey


@dataclass(frozen=True, slots=True)
class ResultFrameKey:
    """Identity of one output frame within one accepted result source."""

    source: ResultSourceKey
    frame_index: int

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be an integer >= 0")


@dataclass(frozen=True, slots=True)
class ResultFrameMetadata:
    """Stable metadata for one frame, independent of its stored fields."""

    key: ResultFrameKey
    increment_number: int | None = None
    step_time: float | None = None
    load_factor: float | None = None
    description: str = ""
    converged: bool = True
    iterations: int | None = None
    residual_norm: float | None = None
    cutbacks: int = 0
    attempt_number: int | None = None
    previous_load_factor: float | None = None
    load_factor_increment: float | None = None
    duration_seconds: float | None = None
    status: str = "converged"
    total_time: float | None = None
    time_increment: float | None = None
    amplitude: float | None = None
    stable_time_increment: float | None = None
    solver_kind: str | None = None

    def __post_init__(self) -> None:
        if type(self.key) is not ResultFrameKey:
            raise TypeError("key must be ResultFrameKey")
        if self.increment_number is not None:
            if (
                isinstance(self.increment_number, bool)
                or not isinstance(self.increment_number, int)
                or self.increment_number < 0
            ):
                raise ValueError(
                    "increment_number must be an integer >= 0 or None"
                )
        for name in (
            "step_time",
            "load_factor",
            "residual_norm",
            "total_time",
            "time_increment",
            "amplitude",
            "stable_time_increment",
        ):
            value = getattr(self, name)
            if value is not None:
                numeric = _finite_real(value, name=name)
                if name in {"step_time", "total_time", "time_increment", "stable_time_increment"} and numeric < 0.0:
                    raise ValueError(f"{name} must be >= 0")
        if type(self.description) is not str:
            raise TypeError("description must be a string")
        if type(self.converged) is not bool:
            raise TypeError("converged must be a bool")
        if self.iterations is not None:
            if (
                isinstance(self.iterations, bool)
                or not isinstance(self.iterations, int)
                or self.iterations < 0
            ):
                raise ValueError("iterations must be an integer >= 0 or None")
        if (
            isinstance(self.cutbacks, bool)
            or not isinstance(self.cutbacks, int)
            or self.cutbacks < 0
        ):
            raise ValueError("cutbacks must be an integer >= 0")
        if self.attempt_number is not None:
            if (
                isinstance(self.attempt_number, bool)
                or not isinstance(self.attempt_number, int)
                or self.attempt_number < 1
            ):
                raise ValueError("attempt_number must be an integer >= 1 or None")
        for name in (
            "previous_load_factor",
            "load_factor_increment",
            "duration_seconds",
        ):
            value = getattr(self, name)
            if value is not None:
                numeric = _finite_real(value, name=name)
                if name == "duration_seconds" and numeric < 0.0:
                    raise ValueError("duration_seconds must be >= 0")
        if type(self.status) is not str or not self.status.strip():
            raise ValueError("status must be a non-empty string")
        if self.solver_kind is not None and (
            not isinstance(self.solver_kind, str) or not self.solver_kind.strip()
        ):
            raise ValueError("solver_kind must be a non-empty string or None")


@dataclass(frozen=True, slots=True)
class ResultFrameCatalog:
    """Ordered frame metadata for one result source.

    Static and dynamic result sources contain the accepted output frames
    produced by the active analysis procedure. An initial-state frame is not
    synthesized when the solver did not publish one.
    """

    source: ResultSourceKey
    frames: tuple[ResultFrameMetadata, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(self.frames) is not tuple:
            raise TypeError("frames must be a tuple")
        previous = -1
        seen: set[int] = set()
        for frame in self.frames:
            if type(frame) is not ResultFrameMetadata:
                raise TypeError(
                    "frames must contain ResultFrameMetadata values"
                )
            if frame.key.source != self.source:
                raise ValueError("frame source must match catalog source")
            index = frame.key.frame_index
            if index in seen:
                raise ValueError("frame indices must be unique")
            if index <= previous:
                raise ValueError("frames must be sorted by frame_index")
            seen.add(index)
            previous = index

    @property
    def frame_keys(self) -> tuple[ResultFrameKey, ...]:
        """Return the ordered identities without exposing mutable storage."""

        return tuple(frame.key for frame in self.frames)

    @property
    def last(self) -> ResultFrameMetadata | None:
        """Return the last real output frame, not a synthetic final frame."""

        return self.frames[-1] if self.frames else None

    def metadata_for(self, frame_index: int) -> ResultFrameMetadata:
        """Return one frame metadata record by zero-based output index."""

        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise TypeError("frame_index must be an integer")
        for frame in self.frames:
            if frame.key.frame_index == frame_index:
                return frame
        raise KeyError(frame_index)


def frame_catalog_from_model_result(
    source: ResultSourceKey,
    result: object,
) -> ResultFrameCatalog:
    """Adapt the current ``ModelResult.frames`` without changing its owner.

    This is intentionally an adapter, not a second result store.  The current
    solver remains responsible for producing ``ModelResult`` and
    ``ResultFrame`` values; application consumers can start using stable
    frame identities before the provider/archive layers become frame-native.
    Legacy results without captured frames produce an empty catalog rather
    than inventing a synthetic final frame.
    """

    if type(source) is not ResultSourceKey:
        raise TypeError("source must be ResultSourceKey")
    from fem.results import ModelResult, ResultFrame

    if type(result) is not ModelResult:
        raise TypeError("result must be ModelResult")
    metadata: list[ResultFrameMetadata] = []
    for frame in result.frames:
        if type(frame) is not ResultFrame:
            raise TypeError("result.frames must contain ResultFrame values")
        outputs = frame.outputs
        dynamic = frame.dynamic_data
        step_time = outputs.get("step_time")
        if step_time is None:
            step_time = outputs.get("time")
        total_time = None
        time_increment = None
        amplitude = None
        stable_time_increment = None
        solver_kind = None
        if dynamic is not None:
            step_time = dynamic.step_time
            total_time = dynamic.total_time
            time_increment = dynamic.time_increment
            amplitude = dynamic.amplitude
            stable_time_increment = (
                None
                if dynamic.diagnostics is None
                else dynamic.diagnostics.stable_time_increment
            )
            solver_kind = dynamic.solver_kind
        cutbacks = outputs.get("cutbacks", 0)
        if isinstance(cutbacks, bool) or not isinstance(cutbacks, int):
            cutbacks = 0
        increment_number = outputs.get("increment_number", frame.frame_index)
        if (
            isinstance(increment_number, bool)
            or not isinstance(increment_number, int)
            or increment_number < 0
        ):
            increment_number = frame.frame_index
        attempt_number = outputs.get("attempt_number")
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number < 1
        ):
            attempt_number = None
        previous_load_factor = _optional_finite_output(
            outputs.get("previous_load_factor"),
            name="previous_load_factor",
        )
        load_factor_increment = _optional_finite_output(
            outputs.get("load_factor_increment"),
            name="load_factor_increment",
        )
        duration_seconds = _optional_finite_output(
            outputs.get("increment_duration_seconds"),
            name="duration_seconds",
        )
        if duration_seconds is not None and duration_seconds < 0.0:
            duration_seconds = None
        status = outputs.get("increment_status", "converged")
        if type(status) is not str or not status.strip():
            status = "converged"
        metadata.append(
            ResultFrameMetadata(
                key=ResultFrameKey(source, frame.frame_index),
                increment_number=increment_number,
                step_time=(
                    None
                    if step_time is None
                    else _finite_real(step_time, name="step_time")
                ),
                load_factor=(None if dynamic is not None else frame.load_factor),
                description=frame.name or "",
                converged=frame.converged,
                iterations=frame.iterations,
                residual_norm=frame.residual_norm,
                cutbacks=cutbacks,
                attempt_number=attempt_number,
                previous_load_factor=previous_load_factor,
                load_factor_increment=load_factor_increment,
                duration_seconds=duration_seconds,
                status=status,
                total_time=total_time,
                time_increment=time_increment,
                amplitude=amplitude,
                stable_time_increment=stable_time_increment,
                solver_kind=solver_kind,
            )
        )
    return ResultFrameCatalog(source, tuple(metadata))


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _optional_finite_output(value: object, *, name: str) -> float | None:
    """Decode optional frame metadata without changing legacy results."""

    if value is None:
        return None
    return _finite_real(value, name=name)


__all__ = [
    "ResultFrameCatalog",
    "ResultFrameKey",
    "ResultFrameMetadata",
    "frame_catalog_from_model_result",
]
