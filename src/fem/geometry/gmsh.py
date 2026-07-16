"""Typed scripted-geometry facade for Gmsh's OpenCASCADE kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import operator
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from fem.core import FEMModel
from fem.io import gmsh as gmsh_io


_CAD_DEPENDENCY_MESSAGE = (
    "Gmsh geometry support requires the optional 'cad' dependencies. "
    'Install the project with: pip install -e ".[cad]"'
)
_PLANAR_TOLERANCE = 1.0e-10
# OpenCASCADE expands Gmsh bounding boxes by this numerical safety gap.
_OCC_BOUNDING_BOX_PADDING = 1.0e-7
_MESH_OPTION_NAMES = (
    "Mesh.ElementOrder",
    "Mesh.SecondOrderIncomplete",
    "Mesh.RecombineAll",
)
_POINT_SIZE_OPTION_NAME = "Mesh.MeshSizeFromPoints"


class GeometryError(RuntimeError):
    """Base error raised by the scripted Gmsh geometry facade."""


class GeometryStateError(GeometryError):
    """Raised when a geometry operation is invalid in the current state."""


class EntityOwnershipError(GeometryError):
    """Raised when an entity belongs to a different geometry model."""


class StaleEntityError(GeometryError):
    """Raised when an entity reference no longer denotes a live OCC entity."""


@dataclass(frozen=True, slots=True)
class EntityRef:
    """Immutable reference to one entity owned by a geometry model."""

    dimension: int
    tag: int
    _owner_token: object = field(repr=False)
    _entity_token: object = field(repr=False)

    def __post_init__(self) -> None:
        _validate_entity_dimension(self.dimension)
        _validate_positive_tag(self.tag, "entity tag")


@dataclass(frozen=True, slots=True)
class BooleanResult:
    """Typed outputs and input-to-output mapping from an OCC boolean."""

    outputs: tuple[EntityRef, ...]
    input_map: tuple[tuple[EntityRef, ...], ...]

    def of_dimension(self, dimension: int) -> tuple[EntityRef, ...]:
        """Return boolean outputs having the requested entity dimension."""
        normalized = _validate_entity_dimension(dimension)
        return tuple(
            entity for entity in self.outputs if entity.dimension == normalized
        )


@dataclass(frozen=True, slots=True)
class PhysicalGroupRef:
    """Immutable reference to a named Gmsh physical group."""

    dimension: int
    tag: int
    name: str
    _owner_token: object = field(repr=False)

    def __post_init__(self) -> None:
        _validate_entity_dimension(self.dimension)
        _validate_positive_tag(self.tag, "physical group tag")


class _State(Enum):
    NEW = auto()
    BUILDING = auto()
    LABELED = auto()
    MESHED = auto()
    MESH_FAILED = auto()
    CLOSED = auto()


_QUERY_STATES = frozenset({_State.BUILDING, _State.LABELED, _State.MESH_FAILED})
_MESH_CONTROL_STATES = frozenset({_State.BUILDING, _State.LABELED})


def _load_gmsh() -> Any:
    try:
        import gmsh
    except ModuleNotFoundError as exc:
        if exc.name != "gmsh":
            raise
        raise ModuleNotFoundError(_CAD_DEPENDENCY_MESSAGE) from exc
    return gmsh


class GeometryModel:
    """Context-managed owner of one scripted OCC model and its mesh attempt.

    Gmsh model, session, and option state is process-global. This facade
    supports same-thread nested contexts but is not thread-safe.
    """

    def __init__(self, name: str, *, dimension: Literal[1, 2, 3]) -> None:
        self._name = name
        self._dimension = _validate_mesh_dimension(dimension)
        self._state = _State.NEW
        self._owner_token = object()
        self._entity_tokens: dict[tuple[int, int], object] = {}
        self._gmsh: Any | None = None
        self._owns_session = False
        self._created_model = False
        self._prior_current: str | None = None
        self._pending_options: dict[str, float] = {}
        self._element_group_names: set[str] = set()
        self._node_group_names: set[str] = set()
        self._mesh_attempted = False

    @property
    def name(self) -> str:
        """Return the immutable Gmsh model name owned by this facade."""
        return self._name

    @property
    def dimension(self) -> Literal[1, 2, 3]:
        """Return the immutable topological mesh dimension."""
        return self._dimension

    def __enter__(self) -> GeometryModel:
        """Enter the facade context and create its isolated Gmsh model."""
        if self._state is not _State.NEW:
            raise self._state_error("context entry", "model context is not new")

        try:
            try:
                self._gmsh = _load_gmsh()
            except ModuleNotFoundError as exc:
                if exc.name == "gmsh" or "optional 'cad'" in str(exc):
                    raise ModuleNotFoundError(_CAD_DEPENDENCY_MESSAGE) from exc
                raise

            if not bool(self._gmsh.isInitialized()):
                self._owns_session = True
                self._gmsh.initialize()

            self._prior_current = str(self._gmsh.model.getCurrent())
            existing_models = tuple(str(item) for item in self._gmsh.model.list())
            if not isinstance(self.name, str) or not self.name.strip():
                raise GeometryStateError(
                    f"geometry model {self.name!r}: model name must be a nonempty string"
                )
            if self.name in existing_models:
                raise GeometryStateError(
                    f"geometry model {self.name!r}: a Gmsh model with this name "
                    "already exists"
                )

            self._gmsh.model.add(self.name)
            self._created_model = True
            self._gmsh.model.setCurrent(self.name)
            self._entity_tokens.clear()
            self._element_group_names.clear()
            self._node_group_names.clear()
            self._mesh_attempted = False
            self._state = _State.BUILDING
            return self
        except BaseException as error:
            cleanup_errors = self._cleanup_after_failed_entry()
            self._state = _State.CLOSED
            for operation, cleanup_error in cleanup_errors:
                error.add_note(
                    f"geometry model {self.name!r}: entry cleanup also failed "
                    f"while trying to {operation}: {cleanup_error}"
                )
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool:
        """Remove the facade model and release only lifecycle state it owns."""
        cleanup_error: tuple[str, BaseException] | None = None
        gmsh = self._gmsh
        try:
            if gmsh is not None and bool(gmsh.isInitialized()):
                cleanup_error = self._capture_cleanup_error(
                    cleanup_error,
                    "restore mesh options",
                    self._restore_pending_options,
                )
                cleanup_error = self._capture_cleanup_error(
                    cleanup_error,
                    "remove facade model",
                    self._remove_created_model,
                )
                cleanup_error = self._capture_cleanup_error(
                    cleanup_error,
                    "restore prior model",
                    self._restore_prior_model,
                )
        finally:
            self._entity_tokens.clear()
            self._state = _State.CLOSED
            if gmsh is not None and self._owns_session:
                if bool(gmsh.isInitialized()):
                    try:
                        gmsh.finalize()
                    except BaseException as error:
                        if cleanup_error is None:
                            cleanup_error = ("finalize owned session", error)
                    else:
                        self._owns_session = False
                        self._created_model = False
                else:
                    self._owns_session = False
                    self._created_model = False

        if exc_value is None and cleanup_error is not None:
            operation, error = cleanup_error
            raise GeometryError(
                f"geometry model {self.name!r}: cleanup failed while trying to "
                f"{operation}"
            ) from error
        if exc_value is not None and cleanup_error is not None:
            operation, error = cleanup_error
            exc_value.add_note(
                f"geometry model {self.name!r}: cleanup also failed while "
                f"trying to {operation}: {error}"
            )
        return False

    def entities(self, dimension: int) -> tuple[EntityRef, ...]:
        """Return synchronized current entities of one dimension."""
        self._check_state("entities", _QUERY_STATES)
        normalized_dimension = _validate_entity_dimension(dimension)
        self._activate("entities")
        self._gmsh.model.occ.synchronize()
        pairs = self._gmsh.model.getEntities(normalized_dimension)
        return tuple(
            self._wrap_entity(pair)
            for pair in sorted(
                (_normalize_dim_tag(pair) for pair in pairs),
                key=lambda item: (item[0], item[1]),
            )
        )

    def point(
        self,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> EntityRef:
        """Create an OCC point in a one-dimensional facade."""
        operation = "point"
        self._check_state(operation, frozenset({_State.BUILDING}))
        if self.dimension != 1:
            raise ValueError("point requires a one-dimensional geometry model")
        coordinates = (
            _finite_float(x, "x"),
            _finite_float(y, "y"),
            _finite_float(z, "z"),
        )
        self._activate(operation)
        tag = self._gmsh.model.occ.addPoint(*coordinates)
        return self._wrap_entity((0, tag))

    def line(
        self,
        start: EntityRef,
        end: EntityRef,
    ) -> EntityRef:
        """Create a straight OCC line between two facade-owned points."""
        operation = "line"
        self._check_state(operation, frozenset({_State.BUILDING}))
        if self.dimension != 1:
            raise ValueError("line requires a one-dimensional geometry model")
        endpoints = self._normalize_entities((start, end), operation=operation)
        if any(endpoint.dimension != 0 for endpoint in endpoints):
            raise ValueError("line endpoints must be dimension-zero point references")
        self._activate(operation)
        self._assert_occ_liveness(endpoints, operation)
        tag = self._gmsh.model.occ.addLine(endpoints[0].tag, endpoints[1].tag)
        return self._wrap_entity((1, tag))

    def rectangle(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        z: float = 0.0,
        rounded_radius: float = 0.0,
    ) -> EntityRef:
        """Create a rectangular OCC surface."""
        self._check_state("rectangle", frozenset({_State.BUILDING}))
        if self.dimension == 1:
            raise ValueError(
                "rectangle requires a two- or three-dimensional geometry model"
            )
        x_value = _finite_float(x, "x")
        y_value = _finite_float(y, "y")
        z_value = _finite_float(z, "z")
        width_value = _positive_float(width, "width")
        height_value = _positive_float(height, "height")
        radius_value = _nonnegative_float(rounded_radius, "rounded_radius")
        if radius_value >= 0.5 * min(width_value, height_value):
            raise ValueError(
                "rounded_radius must be strictly less than half the shorter "
                "rectangle side"
            )
        self._validate_2d_z(z_value, "rectangle")
        self._activate("rectangle")
        tag = self._gmsh.model.occ.addRectangle(
            x_value,
            y_value,
            z_value,
            width_value,
            height_value,
            -1,
            radius_value,
        )
        return self._wrap_entity((2, tag))

    def disk(
        self,
        x: float,
        y: float,
        radius_x: float,
        *,
        z: float = 0.0,
        radius_y: float | None = None,
    ) -> EntityRef:
        """Create an elliptical or circular OCC disk surface."""
        self._check_state("disk", frozenset({_State.BUILDING}))
        if self.dimension == 1:
            raise ValueError("disk requires a two- or three-dimensional geometry model")
        x_value = _finite_float(x, "x")
        y_value = _finite_float(y, "y")
        z_value = _finite_float(z, "z")
        radius_x_value = _positive_float(radius_x, "radius_x")
        radius_y_value = (
            radius_x_value
            if radius_y is None
            else _positive_float(radius_y, "radius_y")
        )
        self._validate_2d_z(z_value, "disk")
        self._activate("disk")
        if radius_y_value > radius_x_value:
            tag = self._gmsh.model.occ.addDisk(
                x_value,
                y_value,
                z_value,
                radius_y_value,
                radius_x_value,
                -1,
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            )
        else:
            tag = self._gmsh.model.occ.addDisk(
                x_value,
                y_value,
                z_value,
                radius_x_value,
                radius_y_value,
                -1,
            )
        return self._wrap_entity((2, tag))

    def box(
        self,
        x: float,
        y: float,
        z: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> EntityRef:
        """Create an OCC box volume in a three-dimensional facade."""
        self._check_state("box", frozenset({_State.BUILDING}))
        if self.dimension != 3:
            raise ValueError("box requires a three-dimensional geometry model")
        values = (
            _finite_float(x, "x"),
            _finite_float(y, "y"),
            _finite_float(z, "z"),
            _positive_float(dx, "dx"),
            _positive_float(dy, "dy"),
            _positive_float(dz, "dz"),
        )
        self._activate("box")
        tag = self._gmsh.model.occ.addBox(*values, -1)
        return self._wrap_entity((3, tag))

    def cylinder(
        self,
        x: float,
        y: float,
        z: float,
        axis_x: float,
        axis_y: float,
        axis_z: float,
        radius: float,
        *,
        angle: float = 2.0 * math.pi,
    ) -> EntityRef:
        """Create an OCC cylinder volume in a three-dimensional facade."""
        self._check_state("cylinder", frozenset({_State.BUILDING}))
        if self.dimension != 3:
            raise ValueError("cylinder requires a three-dimensional geometry model")
        coordinates = (
            _finite_float(x, "x"),
            _finite_float(y, "y"),
            _finite_float(z, "z"),
        )
        axis = (
            _finite_float(axis_x, "axis_x"),
            _finite_float(axis_y, "axis_y"),
            _finite_float(axis_z, "axis_z"),
        )
        if not any(component != 0.0 for component in axis):
            raise ValueError("cylinder axis must be nonzero")
        radius_value = _positive_float(radius, "radius")
        angle_value = _positive_float(angle, "angle")
        if angle_value > 2.0 * math.pi:
            raise ValueError("cylinder angle must be no greater than 2 * pi")
        self._activate("cylinder")
        tag = self._gmsh.model.occ.addCylinder(
            *coordinates,
            *axis,
            radius_value,
            -1,
            angle_value,
        )
        return self._wrap_entity((3, tag))

    def fuse(
        self,
        objects: Iterable[EntityRef],
        tools: Iterable[EntityRef],
        *,
        remove_objects: bool = True,
        remove_tools: bool = True,
    ) -> BooleanResult:
        """Fuse same-dimensional OCC entities and retain the Gmsh input map."""
        return self._boolean(
            "fuse",
            objects,
            tools,
            remove_objects=remove_objects,
            remove_tools=remove_tools,
        )

    def cut(
        self,
        objects: Iterable[EntityRef],
        tools: Iterable[EntityRef],
        *,
        remove_objects: bool = True,
        remove_tools: bool = True,
    ) -> BooleanResult:
        """Subtract tool entities from objects and retain the Gmsh input map."""
        return self._boolean(
            "cut",
            objects,
            tools,
            remove_objects=remove_objects,
            remove_tools=remove_tools,
        )

    def fragment(
        self,
        objects: Iterable[EntityRef],
        tools: Iterable[EntityRef],
        *,
        remove_objects: bool = True,
        remove_tools: bool = True,
    ) -> BooleanResult:
        """Fragment same-dimensional OCC entities and retain the input map."""
        return self._boolean(
            "fragment",
            objects,
            tools,
            remove_objects=remove_objects,
            remove_tools=remove_tools,
        )

    def translate(
        self,
        entities: Iterable[EntityRef],
        dx: float,
        dy: float,
        dz: float,
    ) -> tuple[EntityRef, ...]:
        """Translate entities in place and return their existing references."""
        operation = "translate"
        self._check_state(operation, frozenset({_State.BUILDING}))
        normalized = self._normalize_entities(entities, operation=operation)
        vector = (
            _finite_float(dx, "dx"),
            _finite_float(dy, "dy"),
            _finite_float(dz, "dz"),
        )
        if self.dimension == 2 and vector[2] != 0.0:
            raise ValueError("2D translation must remain in the global XY plane")
        self._activate(operation)
        self._assert_occ_liveness(normalized, operation)
        self._gmsh.model.occ.translate(_dim_tags(normalized), *vector)
        return normalized

    def rotate(
        self,
        entities: Iterable[EntityRef],
        x: float,
        y: float,
        z: float,
        axis_x: float,
        axis_y: float,
        axis_z: float,
        angle: float,
    ) -> tuple[EntityRef, ...]:
        """Rotate entities in place and return their existing references."""
        operation = "rotate"
        self._check_state(operation, frozenset({_State.BUILDING}))
        normalized = self._normalize_entities(entities, operation=operation)
        center = (
            _finite_float(x, "x"),
            _finite_float(y, "y"),
            _finite_float(z, "z"),
        )
        axis = (
            _finite_float(axis_x, "axis_x"),
            _finite_float(axis_y, "axis_y"),
            _finite_float(axis_z, "axis_z"),
        )
        angle_value = _finite_float(angle, "angle")
        if not any(component != 0.0 for component in axis):
            raise ValueError("rotation axis must be nonzero")
        if self.dimension == 2 and (axis[0] != 0.0 or axis[1] != 0.0):
            raise ValueError(
                "2D rotation axis must be parallel to the global Z axis"
            )
        self._activate(operation)
        self._assert_occ_liveness(normalized, operation)
        self._gmsh.model.occ.rotate(
            _dim_tags(normalized),
            *center,
            *axis,
            angle_value,
        )
        return normalized

    def extrude(
        self,
        entities: Iterable[EntityRef],
        dx: float,
        dy: float,
        dz: float,
        *,
        num_elements: Sequence[int] = (),
        heights: Sequence[float] = (),
        recombine: bool = False,
    ) -> tuple[EntityRef, ...]:
        """Extrude OCC entities with optional structured layer controls."""
        operation = "extrude"
        self._check_state(operation, frozenset({_State.BUILDING}))
        if self.dimension == 1:
            raise ValueError("extrude is unavailable in a one-dimensional geometry model")
        normalized = self._normalize_entities(entities, operation=operation)
        dimensions = {entity.dimension for entity in normalized}
        if len(dimensions) != 1:
            raise ValueError("extrude inputs must have one common dimension")
        input_dimension = next(iter(dimensions))
        if input_dimension + 1 > self.dimension:
            raise ValueError(
                "extrude input dimension plus one exceeds the facade dimension"
            )
        vector = (
            _finite_float(dx, "dx"),
            _finite_float(dy, "dy"),
            _finite_float(dz, "dz"),
        )
        if not any(component != 0.0 for component in vector):
            raise ValueError("extrusion vector must be nonzero")
        if self.dimension == 2 and vector[2] != 0.0:
            raise ValueError("2D extrusion must remain in the global XY plane")
        layer_counts = _positive_integer_sequence(num_elements, "num_elements")
        normalized_heights = tuple(
            _finite_float(value, "height") for value in tuple(heights)
        )
        if normalized_heights:
            if not layer_counts:
                raise ValueError("heights require nonempty num_elements")
            if len(normalized_heights) != len(layer_counts):
                raise ValueError("heights and num_elements must have equal lengths")
            if any(value <= 0.0 or value > 1.0 for value in normalized_heights):
                raise ValueError("heights must lie in the normalized interval (0, 1]")
            if any(
                current <= previous
                for previous, current in zip(
                    normalized_heights,
                    normalized_heights[1:],
                )
            ):
                raise ValueError("heights must be strictly increasing")
            if not math.isclose(
                normalized_heights[-1],
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("the final normalized height must be 1.0")
        if not isinstance(recombine, bool):
            raise TypeError(f"recombine must be a boolean, got {recombine!r}")
        self._activate(operation)
        self._assert_occ_liveness(normalized, operation)
        output_pairs = self._gmsh.model.occ.extrude(
            _dim_tags(normalized),
            *vector,
            list(layer_counts),
            list(normalized_heights),
            recombine,
        )
        validated_pairs = tuple(_normalize_dim_tag(pair) for pair in output_pairs)
        return tuple(self._wrap_entity(pair) for pair in validated_pairs)

    def entity(self, dimension: int, tag: int) -> EntityRef:
        """Acquire a typed reference for an existing raw OCC entity."""
        operation = "entity"
        self._check_state(operation, _QUERY_STATES)
        normalized = (
            _validate_entity_dimension(dimension),
            _validate_positive_tag(tag, "entity tag"),
        )
        self._activate(operation)
        existing = {
            _normalize_dim_tag(pair)
            for pair in self._gmsh.model.occ.getEntities(normalized[0])
        }
        if normalized not in existing:
            raise StaleEntityError(
                f"geometry model {self.name!r}: OCC entity "
                f"({normalized[0]}, {normalized[1]}) does not exist"
            )
        return self._wrap_entity(normalized)

    def boundary(
        self,
        entities: Iterable[EntityRef],
        *,
        combined: bool = True,
        recursive: bool = False,
    ) -> tuple[EntityRef, ...]:
        """Return the deterministic, unoriented boundary of OCC entities."""
        operation = "boundary"
        self._check_state(operation, _QUERY_STATES)
        normalized = self._normalize_entities(entities, operation=operation)
        if not isinstance(combined, bool):
            raise TypeError(f"combined must be a boolean, got {combined!r}")
        if not isinstance(recursive, bool):
            raise TypeError(f"recursive must be a boolean, got {recursive!r}")
        self._activate(operation)
        self._assert_occ_liveness(normalized, operation)
        self._gmsh.model.occ.synchronize()
        result = self._gmsh.model.getBoundary(
            _dim_tags(normalized),
            combined=combined,
            oriented=False,
            recursive=recursive,
        )
        pairs = sorted(
            {_normalize_dim_tag(pair) for pair in result},
            key=lambda item: (item[0], item[1]),
        )
        return tuple(self._wrap_entity(pair) for pair in pairs)

    def select(
        self,
        entities: Iterable[EntityRef],
        *,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        tolerance: float = 1.0e-8,
    ) -> tuple[EntityRef, ...]:
        """Select entities lying on supplied axis-aligned coordinates."""
        operation = "select"
        self._check_state(operation, _QUERY_STATES)
        normalized = self._normalize_entities(entities, operation=operation)
        raw_coordinates = (x, y, z)
        if all(value is None for value in raw_coordinates):
            raise ValueError("select requires at least one coordinate")
        coordinates = tuple(
            None if value is None else _finite_float(value, axis)
            for axis, value in zip(("x", "y", "z"), raw_coordinates)
        )
        tolerance_value = _nonnegative_float(tolerance, "tolerance")
        bounding_tolerance = tolerance_value + _OCC_BOUNDING_BOX_PADDING
        self._activate(operation)
        self._assert_occ_liveness(normalized, operation)
        self._gmsh.model.occ.synchronize()
        matches: list[EntityRef] = []
        for entity in normalized:
            bounds = tuple(
                float(value)
                for value in self._gmsh.model.getBoundingBox(
                    entity.dimension,
                    entity.tag,
                )
            )
            if len(bounds) != 6:
                raise GeometryError(
                    f"geometry model {self.name!r}: invalid bounding box for "
                    f"entity ({entity.dimension}, {entity.tag})"
                )
            if all(
                coordinate is None
                or (
                    abs(bounds[axis] - coordinate) <= bounding_tolerance
                    and abs(bounds[axis + 3] - coordinate) <= bounding_tolerance
                )
                for axis, coordinate in enumerate(coordinates)
            ):
                matches.append(entity)
        return tuple(sorted(matches, key=lambda item: (item.dimension, item.tag)))

    def physical(
        self,
        name: str,
        entities: Iterable[EntityRef],
    ) -> PhysicalGroupRef:
        """Create a named physical group and freeze topology after success."""
        operation = "physical"
        self._check_state(
            operation,
            frozenset({_State.BUILDING, _State.LABELED}),
        )
        if not isinstance(name, str) or not name.strip():
            raise ValueError("physical group name must be a nonempty string")
        normalized_name = name.strip()
        normalized = self._normalize_entities(entities, operation=operation)
        dimensions = {entity.dimension for entity in normalized}
        if len(dimensions) != 1:
            raise ValueError("physical group entities must have one common dimension")
        dimension = next(iter(dimensions))
        if dimension > self.dimension:
            raise ValueError(
                "physical group entity dimension exceeds the facade dimension"
            )
        namespace = (
            self._element_group_names
            if dimension == self.dimension
            else self._node_group_names
        )
        namespace_label = (
            "element-set namespace"
            if dimension == self.dimension
            else "node-set namespace"
        )
        if normalized_name in namespace:
            raise ValueError(
                f"physical name {normalized_name!r} already exists in the "
                f"{namespace_label}"
            )
        self._activate(operation)
        self._assert_occ_liveness(normalized, operation)
        self._gmsh.model.occ.synchronize()
        tag = self._gmsh.model.addPhysicalGroup(
            dimension,
            [entity.tag for entity in normalized],
            name=normalized_name,
        )
        reference = PhysicalGroupRef(
            dimension,
            _validate_positive_tag(tag, "physical group tag"),
            normalized_name,
            self._owner_token,
        )
        namespace.add(normalized_name)
        self._state = _State.LABELED
        return reference

    def transfinite_curve(
        self,
        curve: EntityRef,
        *,
        num_nodes: int,
    ) -> None:
        """Set Gmsh's primary-node count, not an element count, on one curve."""
        operation = "transfinite_curve"
        self._check_state(operation, _MESH_CONTROL_STATES)
        node_count = _integer_at_least(num_nodes, "num_nodes", minimum=2)
        target = self._prepare_mesh_control_target(
            curve,
            dimension=1,
            operation=operation,
        )
        self._gmsh.model.mesh.setTransfiniteCurve(target.tag, node_count)

    def transfinite_surface(
        self,
        surface: EntityRef,
        *,
        corners: Sequence[EntityRef] = (),
    ) -> None:
        """Mark one surface as transfinite with optional boundary corners.

        Gmsh remains responsible for topology suitability. Call ``recombine``
        separately when quadrilateral output is desired.
        """
        operation = "transfinite_surface"
        self._check_state(operation, _MESH_CONTROL_STATES)
        normalized_corners = self._normalize_mesh_control_corners(
            corners,
            allowed_counts=(0, 3, 4),
            operation=operation,
        )
        target = self._prepare_mesh_control_target(
            surface,
            dimension=2,
            operation=operation,
            related_entities=normalized_corners,
        )
        self._assert_mesh_control_corners_on_boundary(
            target,
            normalized_corners,
            operation=operation,
        )
        self._gmsh.model.mesh.setTransfiniteSurface(
            target.tag,
            cornerTags=[corner.tag for corner in normalized_corners],
        )

    def transfinite_volume(
        self,
        volume: EntityRef,
        *,
        corners: Sequence[EntityRef] = (),
    ) -> None:
        """Mark one volume as transfinite without configuring its boundary.

        The caller must constrain suitable boundary curves and surfaces; Gmsh
        remains responsible for rejecting incompatible or unsuitable topology.
        """
        operation = "transfinite_volume"
        self._check_state(operation, _MESH_CONTROL_STATES)
        normalized_corners = self._normalize_mesh_control_corners(
            corners,
            allowed_counts=(0, 6, 8),
            operation=operation,
        )
        target = self._prepare_mesh_control_target(
            volume,
            dimension=3,
            operation=operation,
            related_entities=normalized_corners,
        )
        self._assert_mesh_control_corners_on_boundary(
            target,
            normalized_corners,
            operation=operation,
        )
        self._gmsh.model.mesh.setTransfiniteVolume(
            target.tag,
            cornerTags=[corner.tag for corner in normalized_corners],
        )

    def recombine(self, surface: EntityRef) -> None:
        """Request native Gmsh recombination on one surface.

        The entity-local request retains Gmsh's default angle and does not
        guarantee an all-quadrilateral mesh for unsuitable topology.
        """
        operation = "recombine"
        self._check_state(operation, _MESH_CONTROL_STATES)
        target = self._prepare_mesh_control_target(
            surface,
            dimension=2,
            operation=operation,
        )
        self._gmsh.model.mesh.setRecombine(2, target.tag)

    def generate_mesh(
        self,
        *,
        size: float | None = None,
        order: Literal[1, 2] = 1,
        recombine: bool = False,
        line_element_type: Literal["Truss2", "Beam2"] | None = None,
        plane_type: Literal["stress", "strain"] = "stress",
        thickness: float = 1.0,
        z_tolerance: float = 1.0e-10,
    ) -> gmsh_io.GmshImportResult:
        """Generate and import the one mesh permitted for this facade model."""
        imported = self._generate_mesh(
            operation="generate_mesh",
            size=size,
            order=order,
            recombine=recombine,
            line_element_type=line_element_type,
            plane_type=plane_type,
            thickness=thickness,
            z_tolerance=z_tolerance,
        )
        self._state = _State.MESHED
        return imported

    def generate_fem_model(
        self,
        name: str | None = None,
        *,
        size: float | None = None,
        order: Literal[1, 2] = 1,
        recombine: bool = False,
        line_element_type: Literal["Truss2", "Beam2"] | None = None,
        plane_type: Literal["stress", "strain"] = "stress",
        thickness: float = 1.0,
        z_tolerance: float = 1.0e-10,
    ) -> FEMModel:
        """Generate, import, and convert the mesh through ``to_fem_model``."""
        imported = self._generate_mesh(
            operation="generate_fem_model",
            size=size,
            order=order,
            recombine=recombine,
            line_element_type=line_element_type,
            plane_type=plane_type,
            thickness=thickness,
            z_tolerance=z_tolerance,
        )
        try:
            fem_model = imported.to_fem_model(name)
        except BaseException as error:
            self._state = _State.MESH_FAILED
            error.add_note(
                f"geometry model {self.name!r}: FEM model conversion failed"
            )
            raise
        self._state = _State.MESHED
        return fem_model

    def _generate_mesh(
        self,
        *,
        operation: Literal["generate_mesh", "generate_fem_model"],
        size: float | None,
        order: Literal[1, 2],
        recombine: bool,
        line_element_type: Literal["Truss2", "Beam2"] | None,
        plane_type: Literal["stress", "strain"],
        thickness: float,
        z_tolerance: float,
    ) -> gmsh_io.GmshImportResult:
        self._check_state(
            operation,
            frozenset({_State.BUILDING, _State.LABELED}),
        )
        if self._mesh_attempted:
            raise self._state_error(operation, "the one mesh attempt was already used")
        size_value = None if size is None else _positive_float(size, "size")
        if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2):
            raise ValueError(f"order must be integer 1 or 2, got {order!r}")
        if not isinstance(recombine, bool):
            raise TypeError(f"recombine must be a boolean, got {recombine!r}")
        normalized_line_element_type = _validate_line_element_type(
            self.dimension,
            line_element_type,
        )
        if self.dimension == 1 and order != 1:
            raise ValueError("order must be 1 for a one-dimensional geometry model")
        if self.dimension == 1 and recombine:
            raise ValueError("recombine must be False for a one-dimensional geometry model")
        if not isinstance(plane_type, str) or plane_type.lower() not in {
            "stress",
            "strain",
        }:
            raise ValueError(
                f"plane_type must be 'stress' or 'strain', got {plane_type!r}"
            )
        normalized_plane_type = plane_type.lower()
        thickness_value = _positive_float(thickness, "thickness")
        tolerance_value = _nonnegative_float(z_tolerance, "z_tolerance")

        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        top_entities = sorted(
            _normalize_dim_tag(pair)
            for pair in self._gmsh.model.getEntities(self.dimension)
        )
        if not top_entities:
            raise ValueError(
                f"geometry model {self.name!r}: mesh generation requires at "
                "least one top-dimensional OCC entity"
            )

        self._mesh_attempted = True
        try:
            if size_value is not None:
                points = sorted(
                    _normalize_dim_tag(pair)
                    for pair in self._gmsh.model.getEntities(0)
                )
                if not points:
                    raise GeometryError(
                        f"geometry model {self.name!r}: uniform mesh size "
                        "requires at least one point entity"
                    )
                self._gmsh.model.mesh.setSize(points, size_value)

            self._snapshot_and_set_mesh_options(
                order,
                recombine,
                use_point_sizes=size_value is not None,
            )
            self._gmsh.model.mesh.generate(self.dimension)
            self._gmsh.model.occ.synchronize()
            imported = gmsh_io.from_model(
                dimension=self.dimension,
                gmsh_model=self._gmsh.model,
                line_element_type=normalized_line_element_type,
                plane_type=normalized_plane_type,
                thickness=thickness_value,
                z_tolerance=tolerance_value,
            )
        except BaseException as error:
            self._state = _State.MESH_FAILED
            try:
                self._restore_pending_options()
            except BaseException as restore_error:
                error.add_note(
                    f"geometry model {self.name!r}: additionally failed to "
                    f"restore Gmsh mesh options: {restore_error}"
                )
            error.add_note(
                f"geometry model {self.name!r}: mesh generation/import failed"
            )
            raise

        try:
            self._restore_pending_options()
        except BaseException as error:
            self._state = _State.MESH_FAILED
            raise GeometryError(
                f"geometry model {self.name!r}: mesh generation succeeded but "
                "restoring global Gmsh options failed"
            ) from error
        return imported

    def _snapshot_and_set_mesh_options(
        self,
        order: Literal[1, 2],
        recombine: bool,
        *,
        use_point_sizes: bool,
    ) -> None:
        if self._pending_options:
            raise GeometryStateError(
                f"geometry model {self.name!r}: mesh options already have a "
                "pending restoration"
            )
        requested = {
            "Mesh.ElementOrder": float(order),
            "Mesh.SecondOrderIncomplete": 1.0 if order == 2 else 0.0,
            "Mesh.RecombineAll": 1.0 if recombine else 0.0,
        }
        option_names = _MESH_OPTION_NAMES
        if use_point_sizes:
            requested[_POINT_SIZE_OPTION_NAME] = 1.0
            option_names = (*option_names, _POINT_SIZE_OPTION_NAME)
        for option_name in option_names:
            self._pending_options[option_name] = float(
                self._gmsh.option.getNumber(option_name)
            )
        for option_name in option_names:
            self._gmsh.option.setNumber(option_name, requested[option_name])

    @property
    def raw_model(self) -> Any:
        """Return the active raw Gmsh model after invalidating typed refs."""
        return self._raw_handle("raw_model", "model")

    @property
    def raw_occ(self) -> Any:
        """Return the active raw OCC handle after invalidating typed refs."""
        return self._raw_handle("raw_occ", "occ")

    def _raw_handle(self, operation: str, kind: Literal["model", "occ"]) -> Any:
        self._check_state(operation, frozenset({_State.BUILDING}))
        self._activate(operation)
        self._entity_tokens.clear()
        if kind == "model":
            return self._gmsh.model
        return self._gmsh.model.occ

    def _boolean(
        self,
        operation: Literal["fuse", "cut", "fragment"],
        objects: Iterable[EntityRef],
        tools: Iterable[EntityRef],
        *,
        remove_objects: bool,
        remove_tools: bool,
    ) -> BooleanResult:
        self._check_state(operation, frozenset({_State.BUILDING}))
        normalized_objects = self._normalize_entities(objects, operation=operation)
        normalized_tools = self._normalize_entities(tools, operation=operation)
        if set(normalized_objects) & set(normalized_tools):
            raise ValueError(f"{operation} inputs must not overlap")
        dimensions = {
            entity.dimension for entity in (*normalized_objects, *normalized_tools)
        }
        if len(dimensions) != 1:
            raise ValueError(f"{operation} inputs must have one common dimension")
        if not isinstance(remove_objects, bool):
            raise TypeError(
                f"remove_objects must be a boolean, got {remove_objects!r}"
            )
        if not isinstance(remove_tools, bool):
            raise TypeError(f"remove_tools must be a boolean, got {remove_tools!r}")
        all_inputs = (*normalized_objects, *normalized_tools)
        self._activate(operation)
        self._assert_occ_liveness(all_inputs, operation)
        removed_inputs = (
            *(normalized_objects if remove_objects else ()),
            *(normalized_tools if remove_tools else ()),
        )
        invalidated_keys = self._entity_boundary_closure_keys(removed_inputs)
        backend_operation = getattr(self._gmsh.model.occ, operation)
        raw_outputs, raw_map = backend_operation(
            _dim_tags(normalized_objects),
            _dim_tags(normalized_tools),
            -1,
            remove_objects,
            remove_tools,
        )
        self._invalidate_entity_keys(invalidated_keys)
        try:
            output_pairs = tuple(_normalize_dim_tag(pair) for pair in raw_outputs)
            map_pairs = tuple(
                tuple(_normalize_dim_tag(pair) for pair in group)
                for group in raw_map
            )
            if len(map_pairs) != len(all_inputs):
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} returned "
                    "an invalid input map"
                )
        except BaseException as error:
            if invalidated_keys:
                self._entity_tokens.clear()
            if isinstance(error, GeometryError) or not isinstance(error, Exception):
                raise
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} returned invalid "
                "boolean output data"
            ) from error
        outputs = tuple(self._wrap_entity(pair) for pair in output_pairs)
        input_map = tuple(
            tuple(self._wrap_entity(pair) for pair in group) for group in map_pairs
        )
        return BooleanResult(outputs, input_map)

    def _normalize_entities(
        self,
        entities: Iterable[EntityRef],
        *,
        operation: str,
    ) -> tuple[EntityRef, ...]:
        try:
            normalized = tuple(entities)
        except TypeError as exc:
            raise TypeError(f"{operation} entities must be iterable") from exc
        if not normalized:
            raise ValueError(f"{operation} requires at least one entity")
        seen: set[EntityRef] = set()
        for entity in normalized:
            if not isinstance(entity, EntityRef):
                raise TypeError(
                    f"{operation} requires EntityRef values, got {entity!r}"
                )
            if entity._owner_token is not self._owner_token:
                raise EntityOwnershipError(
                    f"geometry model {self.name!r}: {operation} received an "
                    "entity owned by another geometry model"
                )
            current_token = self._entity_tokens.get(
                (entity.dimension, entity.tag)
            )
            if current_token is not entity._entity_token:
                raise StaleEntityError(
                    f"geometry model {self.name!r}: {operation} received stale "
                    f"entity ({entity.dimension}, {entity.tag})"
                )
            if entity in seen:
                raise ValueError(f"{operation} entity inputs must be duplicate-free")
            seen.add(entity)
        return normalized

    def _prepare_mesh_control_target(
        self,
        entity: EntityRef,
        *,
        dimension: int,
        operation: str,
        related_entities: Iterable[EntityRef] = (),
    ) -> EntityRef:
        target = self._normalize_entities((entity,), operation=operation)[0]
        if target.dimension != dimension:
            raise ValueError(
                f"{operation} target must be a dimension-{dimension} entity"
            )
        if dimension > self.dimension:
            raise ValueError(
                f"{operation} target dimension exceeds the facade dimension"
            )
        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        self._assert_occ_liveness((target, *related_entities), operation)
        return target

    def _normalize_mesh_control_corners(
        self,
        corners: Sequence[EntityRef],
        *,
        allowed_counts: tuple[int, ...],
        operation: str,
    ) -> tuple[EntityRef, ...]:
        try:
            materialized = tuple(corners)
        except TypeError as exc:
            raise TypeError(f"{operation} corners must be iterable") from exc
        if len(materialized) not in allowed_counts:
            allowed_text = ", ".join(str(count) for count in allowed_counts)
            raise ValueError(
                f"{operation} requires {allowed_text} corners, got "
                f"{len(materialized)}"
            )
        if not materialized:
            return ()
        normalized = self._normalize_entities(
            materialized,
            operation=f"{operation} corners",
        )
        if any(corner.dimension != 0 for corner in normalized):
            raise ValueError(
                f"{operation} corners must be dimension-zero point references"
            )
        return normalized

    def _assert_mesh_control_corners_on_boundary(
        self,
        target: EntityRef,
        corners: tuple[EntityRef, ...],
        *,
        operation: str,
    ) -> None:
        if not corners:
            return
        boundary_keys = self._entity_boundary_closure_keys(
            (target,),
            synchronize=False,
        )
        missing = [
            corner
            for corner in corners
            if (corner.dimension, corner.tag) not in boundary_keys
        ]
        if missing:
            raise ValueError(
                f"{operation} corners must belong to the target's recursive "
                "boundary closure"
            )

    def _assert_occ_liveness(
        self,
        entities: Iterable[EntityRef],
        operation: str,
    ) -> None:
        existing = {
            _normalize_dim_tag(pair) for pair in self._gmsh.model.occ.getEntities()
        }
        for entity in entities:
            key = (entity.dimension, entity.tag)
            if key not in existing:
                if self._entity_tokens.get(key) is entity._entity_token:
                    del self._entity_tokens[key]
                raise StaleEntityError(
                    f"geometry model {self.name!r}: {operation} entity "
                    f"({entity.dimension}, {entity.tag}) no longer exists"
                )

    def _entity_boundary_closure_keys(
        self,
        entities: Iterable[EntityRef],
        *,
        synchronize: bool = True,
    ) -> set[tuple[int, int]]:
        keys = {(entity.dimension, entity.tag) for entity in entities}
        frontier = set(keys)
        if not frontier:
            return keys

        if synchronize:
            self._gmsh.model.occ.synchronize()
        while frontier:
            next_frontier: set[tuple[int, int]] = set()
            dimensions = sorted(
                {dimension for dimension, _ in frontier if dimension > 0},
                reverse=True,
            )
            for dimension in dimensions:
                sources = sorted(
                    pair for pair in frontier if pair[0] == dimension
                )
                raw_boundary = self._gmsh.model.getBoundary(
                    sources,
                    combined=False,
                    oriented=False,
                    recursive=False,
                )
                for raw_pair in raw_boundary:
                    pair = _normalize_dim_tag(raw_pair)
                    if pair[0] < dimension and pair not in keys:
                        next_frontier.add(pair)
            if not next_frontier:
                break
            keys.update(next_frontier)
            frontier = next_frontier
        return keys

    def _invalidate_entity_keys(
        self,
        keys: Iterable[tuple[int, int]],
    ) -> None:
        for key in keys:
            self._entity_tokens.pop(key, None)

    def _validate_2d_z(self, z_value: float, operation: str) -> None:
        if self.dimension == 2 and abs(z_value) > _PLANAR_TOLERANCE:
            raise ValueError(
                f"{operation} in a 2D facade must lie in the global XY plane"
            )

    def _check_state(
        self,
        operation: str,
        allowed: frozenset[_State],
    ) -> None:
        if self._state not in allowed:
            allowed_text = ", ".join(
                state.name for state in sorted(allowed, key=lambda item: item.name)
            )
            raise self._state_error(
                operation,
                f"state {self._state.name} does not permit this operation "
                f"(expected {allowed_text})",
            )

    def _activate(self, operation: str) -> None:
        if self._gmsh is None or not bool(self._gmsh.isInitialized()):
            raise self._state_error(operation, "Gmsh session is not active")
        models = tuple(str(item) for item in self._gmsh.model.list())
        if self.name not in models:
            raise self._state_error(operation, "facade-owned Gmsh model is missing")
        if str(self._gmsh.model.getCurrent()) != self.name:
            self._gmsh.model.setCurrent(self.name)

    def _wrap_entity(self, pair: Any) -> EntityRef:
        dimension, tag = _normalize_dim_tag(pair)
        key = (dimension, tag)
        token = self._entity_tokens.get(key)
        if token is None:
            token = object()
            self._entity_tokens[key] = token
        return EntityRef(dimension, tag, self._owner_token, token)

    def _cleanup_after_failed_entry(
        self,
    ) -> tuple[tuple[str, BaseException], ...]:
        gmsh = self._gmsh
        if gmsh is None:
            return ()
        cleanup_errors: list[tuple[str, BaseException]] = []
        try:
            initialized = bool(gmsh.isInitialized())
        except BaseException as error:
            return (("inspect Gmsh session state", error),)
        if initialized:
            try:
                self._remove_created_model()
            except BaseException as error:
                cleanup_errors.append(("remove facade model", error))
            try:
                self._restore_prior_model()
            except BaseException as error:
                cleanup_errors.append(("restore prior model", error))
            if self._owns_session:
                try:
                    gmsh.finalize()
                except BaseException as error:
                    cleanup_errors.append(("finalize owned session", error))
                else:
                    self._owns_session = False
                    self._created_model = False
        else:
            self._owns_session = False
            self._created_model = False
        return tuple(cleanup_errors)

    def _remove_created_model(self) -> None:
        if not self._created_model or self._gmsh is None:
            return
        model_names = tuple(str(item) for item in self._gmsh.model.list())
        if self.name in model_names:
            if str(self._gmsh.model.getCurrent()) != self.name:
                self._gmsh.model.setCurrent(self.name)
            self._gmsh.model.remove()
        self._created_model = False

    def _restore_prior_model(self) -> None:
        if self._gmsh is None or self._prior_current is None:
            return
        model_names = tuple(str(item) for item in self._gmsh.model.list())
        if self._prior_current in model_names:
            self._gmsh.model.setCurrent(self._prior_current)

    def _restore_pending_options(self) -> None:
        if self._gmsh is None:
            return
        first_error: BaseException | None = None
        for option_name, value in tuple(self._pending_options.items()):
            try:
                self._gmsh.option.setNumber(option_name, value)
            except BaseException as error:
                if first_error is None:
                    first_error = error
            else:
                del self._pending_options[option_name]
        if first_error is not None:
            raise first_error

    @staticmethod
    def _capture_cleanup_error(
        current: tuple[str, BaseException] | None,
        operation: str,
        callback: Any,
    ) -> tuple[str, BaseException] | None:
        try:
            callback()
        except BaseException as error:
            return current if current is not None else (operation, error)
        return current

    def _state_error(self, operation: str, detail: str) -> GeometryStateError:
        return GeometryStateError(
            f"geometry model {self.name!r}: {operation} failed because {detail}"
        )


def model(name: str, *, dimension: Literal[1, 2, 3]) -> GeometryModel:
    """Return a context manager for one scripted Gmsh OCC model."""
    return GeometryModel(name, dimension=dimension)


def _validate_mesh_dimension(value: Any) -> Literal[1, 2, 3]:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 3):
        raise ValueError(f"dimension must be 1, 2, or 3, got {value!r}")
    return value


def _validate_line_element_type(
    dimension: Literal[1, 2, 3],
    value: Any,
) -> Literal["Truss2", "Beam2"] | None:
    if dimension == 1:
        if value not in ("Truss2", "Beam2"):
            raise ValueError(
                "line_element_type must be exactly 'Truss2' or 'Beam2' for "
                f"dimension 1, got {value!r}"
            )
        return value
    if value is not None:
        raise ValueError(
            "line_element_type is only valid for dimension 1, "
            f"got {value!r} for dimension {dimension}"
        )
    return None


def _validate_entity_dimension(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in range(4):
        raise ValueError(f"entity dimension must be an integer from 0 through 3, got {value!r}")
    return value


def _validate_positive_tag(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    try:
        normalized = int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{label} must be a positive integer, got {value!r}") from exc
    if normalized <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return normalized


def _normalize_dim_tag(value: Any) -> tuple[int, int]:
    try:
        dimension, tag = value
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"invalid Gmsh entity reference {value!r}") from exc
    return (
        _validate_entity_dimension(dimension),
        _validate_positive_tag(tag, "entity tag"),
    )


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite, got {value!r}")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite, got {value!r}") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return normalized


def _positive_float(value: Any, label: str) -> float:
    normalized = _finite_float(value, label)
    if normalized <= 0.0:
        raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    return normalized


def _nonnegative_float(value: Any, label: str) -> float:
    normalized = _finite_float(value, label)
    if normalized < 0.0:
        raise ValueError(f"{label} must be finite and >= 0, got {value!r}")
    return normalized


def _positive_integer_sequence(values: Sequence[int], label: str) -> tuple[int, ...]:
    try:
        materialized = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be a sequence of positive integers") from exc
    result: list[int] = []
    for value in materialized:
        if isinstance(value, bool):
            raise ValueError(f"{label} values must be positive integers")
        try:
            normalized = int(operator.index(value))
        except TypeError as exc:
            raise ValueError(f"{label} values must be positive integers") from exc
        if normalized <= 0:
            raise ValueError(f"{label} values must be positive integers")
        result.append(normalized)
    return tuple(result)


def _integer_at_least(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    try:
        normalized = int(operator.index(value))
    except TypeError as exc:
        raise ValueError(
            f"{label} must be an integer >= {minimum}, got {value!r}"
        ) from exc
    if normalized < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return normalized


def _dim_tags(entities: Iterable[EntityRef]) -> list[tuple[int, int]]:
    return [(entity.dimension, entity.tag) for entity in entities]


__all__ = [
    "BooleanResult",
    "EntityOwnershipError",
    "EntityRef",
    "GeometryError",
    "GeometryModel",
    "GeometryStateError",
    "PhysicalGroupRef",
    "StaleEntityError",
    "model",
]
