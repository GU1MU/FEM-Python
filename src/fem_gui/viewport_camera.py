"""Camera framing values and backend snapshot operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def trackball_point(x: float, y: float, width: float, height: float) -> np.ndarray:
    """Project a viewport position onto Abaqus' virtual rotation sphere."""
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    radius = max(0.42 * min(width, height), 1.0)
    point_x = (float(x) - width * 0.5) / radius
    point_y = (height * 0.5 - float(y)) / radius
    distance_squared = point_x * point_x + point_y * point_y
    if distance_squared <= 1.0:
        point_z = float(np.sqrt(max(0.0, 1.0 - distance_squared)))
        return np.array((point_x, point_y, point_z), dtype=float)
    inverse_length = 1.0 / float(np.sqrt(distance_squared))
    return np.array(
        (point_x * inverse_length, point_y * inverse_length, 0.0),
        dtype=float,
    )


def _rotate_vector(
    vector: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate a vector using Rodrigues' formula."""
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return (
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * float(np.dot(axis, vector)) * (1.0 - cosine)
    )


def rotate_camera_trackball(camera: object, previous: np.ndarray, current: np.ndarray) -> bool:
    """Rotate the camera freely around its focal point using an arcball."""
    axis_camera = np.cross(current, previous)
    axis_length = float(np.linalg.norm(axis_camera))
    dot = float(np.clip(np.dot(previous, current), -1.0, 1.0))
    if axis_length <= 1.0e-12:
        return False
    axis_camera /= axis_length
    angle = float(np.arctan2(axis_length, dot))

    position = np.asarray(camera.GetPosition(), dtype=float)
    focal = np.asarray(camera.GetFocalPoint(), dtype=float)
    view_up = np.asarray(camera.GetViewUp(), dtype=float)
    direction = focal - position
    direction_length = float(np.linalg.norm(direction))
    up_length = float(np.linalg.norm(view_up))
    if direction_length <= 1.0e-12 or up_length <= 1.0e-12:
        return False
    direction /= direction_length
    view_up /= up_length
    view_right = np.cross(direction, view_up)
    right_length = float(np.linalg.norm(view_right))
    if right_length <= 1.0e-12:
        return False
    view_right /= right_length
    view_up = np.cross(view_right, direction)

    axis_world = (
        axis_camera[0] * view_right
        + axis_camera[1] * view_up
        - axis_camera[2] * direction
    )
    axis_world_length = float(np.linalg.norm(axis_world))
    if axis_world_length <= 1.0e-12:
        return False
    axis_world /= axis_world_length

    offset = position - focal
    rotated_offset = _rotate_vector(offset, axis_world, angle)
    rotated_up = _rotate_vector(view_up, axis_world, angle)
    camera.SetPosition(*(focal + rotated_offset))
    camera.SetViewUp(*rotated_up)
    camera.OrthogonalizeViewUp()
    return True
