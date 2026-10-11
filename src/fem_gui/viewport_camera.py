"""Camera framing values and backend snapshot operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ViewportCameraState:
    """Camera values that control framing and on-screen magnification."""

    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]
    parallel_scale: float
    view_angle: float
    parallel_projection: int


def capture_camera_state(plotter: object) -> ViewportCameraState | None:
    """Capture framing before transient actors are rebuilt."""

    try:
        camera = plotter.camera
        return ViewportCameraState(
            tuple(float(value) for value in camera.GetPosition()),
            tuple(float(value) for value in camera.GetFocalPoint()),
            tuple(float(value) for value in camera.GetViewUp()),
            float(camera.GetParallelScale()),
            float(camera.GetViewAngle()),
            int(camera.GetParallelProjection()),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def restore_camera_state(plotter: object, state: ViewportCameraState) -> None:
    """Restore framing while allowing the clipping range to follow new actors."""

    camera = plotter.camera
    camera.SetPosition(*state.position)
    camera.SetFocalPoint(*state.focal_point)
    camera.SetViewUp(*state.view_up)
    camera.SetParallelScale(state.parallel_scale)
    camera.SetViewAngle(state.view_angle)
    camera.SetParallelProjection(state.parallel_projection)
    camera.OrthogonalizeViewUp()
    reset_clipping = getattr(plotter, "reset_camera_clipping_range", None)
    if reset_clipping is not None:
        reset_clipping()
