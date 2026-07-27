"""Detached authoring state and pure helpers for native one-dimensional wires.

The values in this module deliberately sit on the GUI side of the Session
boundary.  A draft may be incomplete or invalid while it is being edited;
only :meth:`WireDraftController.to_geometry` creates the strict domain value
owned by :mod:`fem.geometry`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from fem.geometry import WireGeometry, WireMember, WirePoint


WORK_PLANES = ("XY", "XZ", "YZ")


@dataclass(frozen=True, slots=True)
class WireDraftPoint:
    """One editable point, allowing non-finite coordinates during editing."""

    name: str
    x: float
    y: float
    z: float = 0.0

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            raise TypeError("draft point name must be a string")
        for label, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"draft point {label} must be numeric")
            object.__setattr__(self, label, float(value))


@dataclass(frozen=True, slots=True)
class WireDraftMember:
    """One editable member whose endpoints are point names."""

    name: str
    start: str
    end: str

    def __post_init__(self) -> None:
        for label in ("name", "start", "end"):
            if type(getattr(self, label)) is not str:
                raise TypeError(f"draft member {label} must be a string")


@dataclass(frozen=True, slots=True)
class WireDraftSnapshot:
    """Detached immutable state emitted by a draft controller."""

    name: str
    points: tuple[WireDraftPoint, ...] = ()
    members: tuple[WireDraftMember, ...] = ()

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            raise TypeError("draft wire name must be a string")
        points = tuple(self.points)
        members = tuple(self.members)
        if any(type(point) is not WireDraftPoint for point in points):
            raise TypeError("draft points must contain WireDraftPoint values")
        if any(type(member) is not WireDraftMember for member in members):
            raise TypeError("draft members must contain WireDraftMember values")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "members", members)


@dataclass(frozen=True, slots=True)
class WireDraftDiagnostic:
    """A stable, GUI-friendly validation message."""

    code: str
    message: str
    blocking: bool = True


class WireDraftValidationError(ValueError):
    """Raised when an incomplete draft cannot be materialized as a wire."""

    def __init__(self, diagnostics: Sequence[WireDraftDiagnostic]):
        self.diagnostics = tuple(diagnostics)
        message = "; ".join(item.message for item in self.diagnostics)
        super().__init__(message or "wire draft is not complete")


def _plane_name(plane: str) -> str:
    if type(plane) is not str or plane.upper() not in WORK_PLANES:
        raise ValueError("work plane must be XY, XZ, or YZ")
    return plane.upper()


def intersect_ray_with_work_plane(
    near: Iterable[float],
    far: Iterable[float],
    plane: str,
    offset: float = 0.0,
    *,
    tolerance: float = 1.0e-12,
) -> tuple[float, float, float] | None:
    """Intersect a display ray with one finite-offset orthogonal plane.

    ``near`` and ``far`` are world-space points obtained from the viewport's
    display-to-world conversion.  A parallel ray returns ``None``.
    """

    clean_plane = _plane_name(plane)
    near_values = tuple(float(value) for value in near)
    far_values = tuple(float(value) for value in far)
    if len(near_values) != 3 or len(far_values) != 3:
        raise ValueError("ray endpoints must contain three coordinates")
    if not math.isfinite(float(offset)):
        raise ValueError("work plane offset must be finite")
    axis = {"XY": 2, "XZ": 1, "YZ": 0}[clean_plane]
    direction = far_values[axis] - near_values[axis]
    if abs(direction) <= float(tolerance):
        return None
    fraction = (float(offset) - near_values[axis]) / direction
    point = tuple(
        near_values[index] + fraction * (far_values[index] - near_values[index])
        for index in range(3)
    )
    if not all(math.isfinite(value) for value in point):
        return None
    return point


def snap_work_plane_point(
    point: Iterable[float],
    plane: str,
    spacing: float | None,
) -> tuple[float, float, float]:
    """Snap only in-plane coordinates and preserve the plane offset exactly."""

    clean_plane = _plane_name(plane)
    values = tuple(float(value) for value in point)
    if len(values) != 3:
        raise ValueError("point must contain three coordinates")
    if spacing is None:
        return values
    if isinstance(spacing, bool) or not isinstance(spacing, (int, float)):
        raise ValueError("grid spacing must be positive")
    grid = float(spacing)
    if not math.isfinite(grid) or grid <= 0.0:
        raise ValueError("grid spacing must be positive")
    fixed_axis = {"XY": 2, "XZ": 1, "YZ": 0}[clean_plane]
    snapped = list(values)
    for index in range(3):
        if index == fixed_axis:
            continue
        snapped[index] = math.floor(values[index] / grid + 0.5) * grid
        if abs(snapped[index]) <= 1.0e-15:
            snapped[index] = 0.0
    return tuple(snapped)


# Friendly aliases for small viewport/test helpers.
ray_plane_intersection = intersect_ray_with_work_plane
intersect_ray_with_plane = intersect_ray_with_work_plane
snap_to_work_plane_grid = snap_work_plane_point


def _next_name(prefix: str, values: Iterable[str]) -> str:
    occupied = {value.casefold() for value in values if isinstance(value, str)}
    index = 1
    while f"{prefix}{index}".casefold() in occupied:
        index += 1
    return f"{prefix}{index}"


class WireDraftController:
    """Mutable editor state with immutable snapshots and strict conversion."""

    def __init__(
        self,
        snapshot: WireDraftSnapshot | None = None,
        *,
        root: WireGeometry | None = None,
        name: str = "Wire-1",
    ) -> None:
        if snapshot is not None and root is not None:
            raise ValueError("provide either snapshot or root, not both")
        if root is not None:
            if type(root) is not WireGeometry:
                raise TypeError("root must be a WireGeometry")
            snapshot = self.snapshot_from_geometry(root)
        if snapshot is not None and type(snapshot) is not WireDraftSnapshot:
            raise TypeError("snapshot must be a WireDraftSnapshot")
        initial = snapshot or WireDraftSnapshot(str(name))
        self._name = initial.name
        self._points = list(initial.points)
        self._members = list(initial.members)
        self._selection: tuple[str, str] | None = None
        self._initial_snapshot = self.snapshot()

    @classmethod
    def from_geometry(cls, root: WireGeometry) -> "WireDraftController":
        return cls(root=root)

    @staticmethod
    def snapshot_from_geometry(root: WireGeometry) -> WireDraftSnapshot:
        if type(root) is not WireGeometry:
            raise TypeError("root must be a WireGeometry")
        return WireDraftSnapshot(
            root.name,
            tuple(WireDraftPoint(item.name, item.x, item.y, item.z) for item in root.points),
            tuple(WireDraftMember(item.name, item.start, item.end) for item in root.members),
        )

    @property
    def dirty(self) -> bool:
        return self.snapshot() != self._initial_snapshot

    @property
    def selection(self) -> tuple[str, str] | None:
        return self._selection

    @property
    def current_snapshot(self) -> WireDraftSnapshot:
        return self.snapshot()

    def snapshot(self) -> WireDraftSnapshot:
        return WireDraftSnapshot(
            self._name,
            tuple(self._points),
            tuple(self._members),
        )

    def restore_snapshot(self, snapshot: WireDraftSnapshot) -> None:
        if type(snapshot) is not WireDraftSnapshot:
            raise TypeError("snapshot must be a WireDraftSnapshot")
        self._name = snapshot.name
        self._points = list(snapshot.points)
        self._members = list(snapshot.members)
        self._selection = None

    def set_wire_name(self, name: str) -> None:
        if type(name) is not str:
            raise TypeError("wire name must be a string")
        self._name = name

    set_name = set_wire_name

    def _point_index(self, name: str) -> int:
        for index, point in enumerate(self._points):
            if point.name == name:
                return index
        raise KeyError(name)

    def _member_index(self, name: str) -> int:
        for index, member in enumerate(self._members):
            if member.name == name:
                return index
        raise KeyError(name)

    def add_point(
        self,
        name: str | None = None,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> WireDraftPoint:
        point = WireDraftPoint(
            _next_name("P", (item.name for item in self._points))
            if name is None
            else name,
            x,
            y,
            z,
        )
        self._points.append(point)
        self.select_point(point.name)
        return point

    def update_point(
        self,
        name: str,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> WireDraftPoint:
        index = self._point_index(name)
        current = self._points[index]
        updated = WireDraftPoint(
            current.name,
            current.x if x is None else x,
            current.y if y is None else y,
            current.z if z is None else z,
        )
        self._points[index] = updated
        return updated

    set_point_coordinates = update_point

    def rename_point(self, old_name: str, new_name: str) -> WireDraftPoint:
        index = self._point_index(old_name)
        current = self._points[index]
        updated = WireDraftPoint(new_name, current.x, current.y, current.z)
        self._points[index] = updated
        self._members = [
            WireDraftMember(
                member.name,
                new_name if member.start == old_name else member.start,
                new_name if member.end == old_name else member.end,
            )
            for member in self._members
        ]
        if self._selection == ("point", old_name):
            self._selection = ("point", new_name)
        return updated

    def delete_point(self, name: str) -> None:
        index = self._point_index(name)
        references = tuple(
            member.name
            for member in self._members
            if member.start == name or member.end == name
        )
        if references:
            raise ValueError(
                f"cannot delete point {name!r}; members reference it: "
                + ", ".join(references)
            )
        del self._points[index]
        if self._selection == ("point", name):
            self._selection = None

    def add_member(
        self,
        name: str | None = None,
        start: str = "",
        end: str = "",
    ) -> WireDraftMember:
        if not start and len(self._points) >= 1:
            start = self._points[0].name
        if not end and len(self._points) >= 2:
            end = self._points[1].name
        member = WireDraftMember(
            _next_name("M", (item.name for item in self._members))
            if name is None
            else name,
            start,
            end,
        )
        self._members.append(member)
        self.select_member(member.name)
        return member

    def update_member(
        self,
        name: str,
        start: str | None = None,
        end: str | None = None,
    ) -> WireDraftMember:
        index = self._member_index(name)
        current = self._members[index]
        updated = WireDraftMember(
            current.name,
            current.start if start is None else start,
            current.end if end is None else end,
        )
        self._members[index] = updated
        return updated

    change_member_endpoints = update_member

    def rename_member(self, old_name: str, new_name: str) -> WireDraftMember:
        index = self._member_index(old_name)
        current = self._members[index]
        updated = WireDraftMember(new_name, current.start, current.end)
        self._members[index] = updated
        if self._selection == ("member", old_name):
            self._selection = ("member", new_name)
        return updated

    def delete_member(self, name: str) -> None:
        del self._members[self._member_index(name)]
        if self._selection == ("member", name):
            self._selection = None

    def select_point(self, name: str) -> None:
        self._point_index(name)
        self._selection = ("point", name)

    def select_member(self, name: str) -> None:
        self._member_index(name)
        self._selection = ("member", name)

    def clear_selection(self) -> None:
        self._selection = None

    def coincident_point_groups(self) -> tuple[tuple[str, ...], ...]:
        groups: dict[tuple[float, float, float], list[str]] = {}
        for point in self._points:
            groups.setdefault((point.x, point.y, point.z), []).append(point.name)
        return tuple(
            tuple(names)
            for names in groups.values()
            if len(names) > 1
        )

    @property
    def coincident_groups(self) -> tuple[tuple[str, ...], ...]:
        return self.coincident_point_groups()

    def editing_diagnostics(self) -> tuple[WireDraftDiagnostic, ...]:
        diagnostics: list[WireDraftDiagnostic] = []
        if not self._name.strip():
            diagnostics.append(
                WireDraftDiagnostic("wire.name.empty", "Wire name is required", False)
            )
        point_names = [point.name for point in self._points]
        member_names = [member.name for member in self._members]
        for name in point_names:
            if not name.strip():
                diagnostics.append(
                    WireDraftDiagnostic(
                        "point.name.empty",
                        "Point names must not be blank",
                    )
                )
        for name in member_names:
            if not name.strip():
                diagnostics.append(
                    WireDraftDiagnostic(
                        "member.name.empty",
                        "Member names must not be blank",
                    )
                )
        if len({name.casefold() for name in point_names}) != len(point_names):
            diagnostics.append(
                WireDraftDiagnostic("point.name.duplicate", "Point names must be unique")
            )
        if len({name.casefold() for name in member_names}) != len(member_names):
            diagnostics.append(
                WireDraftDiagnostic("member.name.duplicate", "Member names must be unique")
            )
        return tuple(diagnostics)

    def finish_diagnostics(self) -> tuple[WireDraftDiagnostic, ...]:
        diagnostics = list(self.editing_diagnostics())
        if len(self._points) < 2:
            diagnostics.append(
                WireDraftDiagnostic("wire.points.minimum", "A wire requires at least two points")
            )
        if not self._members:
            diagnostics.append(
                WireDraftDiagnostic("wire.members.minimum", "A wire requires at least one member")
            )
        point_names = {point.name for point in self._points}
        used_points: set[str] = set()
        endpoint_pairs: set[frozenset[str]] = set()
        point_by_name = {point.name: point for point in self._points}
        for point in self._points:
            if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
                diagnostics.append(
                    WireDraftDiagnostic(
                        "point.coordinate.nonfinite",
                        f"Point {point.name!r} coordinates must be finite",
                    )
                )
        for member in self._members:
            if member.start not in point_names or member.end not in point_names:
                diagnostics.append(
                    WireDraftDiagnostic(
                        "member.endpoint.unknown",
                        f"Member {member.name!r} references an unknown point",
                    )
                )
                continue
            used_points.update((member.start, member.end))
            if member.start == member.end:
                diagnostics.append(
                    WireDraftDiagnostic(
                        "member.endpoint.same",
                        f"Member {member.name!r} must use two distinct points",
                    )
                )
                continue
            start = point_by_name[member.start]
            end = point_by_name[member.end]
            if (start.x, start.y, start.z) == (end.x, end.y, end.z):
                diagnostics.append(
                    WireDraftDiagnostic(
                        "member.length.zero",
                        f"Member {member.name!r} has zero length",
                    )
                )
            pair = frozenset((member.start, member.end))
            if pair in endpoint_pairs:
                diagnostics.append(
                    WireDraftDiagnostic(
                        "member.endpoint.duplicate",
                        f"Member {member.name!r} duplicates an undirected endpoint pair",
                    )
                )
            endpoint_pairs.add(pair)
        unused = tuple(point.name for point in self._points if point.name not in used_points)
        if unused:
            diagnostics.append(
                WireDraftDiagnostic(
                    "point.unused",
                    "Every point must be used by at least one member: "
                    + ", ".join(unused),
                )
            )
        return tuple(diagnostics)

    @property
    def can_finish(self) -> bool:
        return not self.finish_diagnostics()

    def to_geometry(self) -> WireGeometry:
        diagnostics = self.finish_diagnostics()
        if diagnostics:
            raise WireDraftValidationError(diagnostics)
        try:
            return WireGeometry(
                self._name.strip(),
                tuple(
                    WirePoint(
                        point.name.strip(),
                        point.x,
                        point.y,
                        point.z,
                    )
                    for point in self._points
                ),
                tuple(
                    WireMember(
                        member.name.strip(),
                        member.start.strip(),
                        member.end.strip(),
                    )
                    for member in self._members
                ),
            )
        except (TypeError, ValueError) as error:
            raise WireDraftValidationError(
                (WireDraftDiagnostic("wire.domain.invalid", str(error)),)
            ) from error

    build_geometry = to_geometry
    serialize_complete_domain = to_geometry

    def restore_from_geometry(self, root: WireGeometry) -> None:
        self.restore_snapshot(self.snapshot_from_geometry(root))
        self._initial_snapshot = self.snapshot()


__all__ = [
    "WORK_PLANES",
    "WireDraftController",
    "WireDraftDiagnostic",
    "WireDraftMember",
    "WireDraftPoint",
    "WireDraftSnapshot",
    "WireDraftValidationError",
    "intersect_ray_with_plane",
    "intersect_ray_with_work_plane",
    "ray_plane_intersection",
    "snap_to_work_plane_grid",
    "snap_work_plane_point",
]
