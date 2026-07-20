"""Typed scripted-geometry facade for Gmsh's OpenCASCADE kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import operator
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from .._validation import (
    _finite_float,
    _integer_at_least,
    _nonnegative_float,
    _positive_feature_vector,
    _positive_float,
    _validate_entity_dimension,
    _validate_mesh_dimension,
    _validate_positive_tag,
)
from ..errors import (
    EntityOwnershipError,
    GeometryError,
    GeometryStateError,
    StaleEntityError,
)
from ..types import (
    BooleanResult,
    CurveLoopRef,
    EntityRef,
    FeatureResult,
    LoftContinuity,
    LoftParametrization,
    LoftResult,
    OrientedCurveRef,
    SweepFrame,
    WireRef,
    _unique_first_seen,
)
from . import backend
from .constants import (
    _LOOP_WINDING_REFINEMENTS,
    _OCC_BOUNDING_BOX_PADDING,
    _PLANAR_TOLERANCE,
)
from .predicates import (
    _GeometrySignature,
    _PlaneFrame,
    _Point3D,
    _RigidShapeSignature,
    _coordinate_distance,
    _matches_rigid_shape_signature,
    _matches_rotated_signature,
    _matches_translated_signature,
    _plane_frame,
    _point_axis_distance,
    _polyline_has_self_contact,
    _polyline_winding,
    _project_plane_point,
    _project_plane_points,
    _validate_elliptical_arc_geometry,
    _vector_norm,
)


_POINT_SIZE_OPTION_NAME = "Mesh.MeshSizeFromPoints"
_MESH_FIELD_TYPES = frozenset({"Distance", "Threshold", "Min"})
_GMSH_TOP_CELL_TYPE_NAMES = {
    1: "Line 2",
    2: "Triangle 3",
    3: "Quadrilateral 4",
    4: "Tetrahedron 4",
    5: "Hexahedron 8",
    9: "Triangle 6",
    11: "Tetrahedron 10",
    16: "Quadrilateral 8",
    17: "Hexahedron 20",
}

_AutoCellShape = Literal["tri", "tri-quad", "quad", "tet", "hex"]
_AutoMeshMode = Literal["line", "tri", "tri-quad", "quad", "tet", "hex"]
_GenerationOperation = Literal["MeshSpec generation", "AutoMeshSpec generation"]
_GenerationSizeMode = Literal["none", "uniform", "point", "background"]
_SWEEP_FRAME_NAMES: dict[SweepFrame, str] = {
    "discrete": "DiscreteTrihedron",
    "corrected_frenet": "CorrectedFrenet",
    "frenet": "Frenet",
    "fixed": "Fixed",
    "constant_normal": "ConstantNormal",
    "darboux": "Darboux",
}
_LOFT_CONTINUITY_NAMES: dict[LoftContinuity, str] = {
    value: value for value in ("C0", "G1", "C1", "G2", "C2", "C3", "CN")
}
_LOFT_PARAMETRIZATION_NAMES: dict[LoftParametrization, str] = {
    "chord_length": "ChordLength",
    "centripetal": "Centripetal",
    "iso_parametric": "IsoParametric",
}


class MeshCellShapeError(GeometryError):
    """Raised when an automatic mesh violates its top-cell contract."""


class MeshControlConflictError(GeometryError):
    """Raised when explicit topology controls conflict with automatic meshing."""


class MeshFieldOwnershipError(GeometryError):
    """Raised when a mesh field belongs to a different geometry model."""


class StaleMeshFieldError(GeometryError):
    """Raised when a mesh-field reference no longer denotes a live field."""


class StaleGmshMeshError(GeometryError):
    """Raised when a generated native mesh is no longer available to import."""


def _stale_gmsh_mesh_error(model_name: str) -> StaleGmshMeshError:
    return StaleGmshMeshError(
        f"generated Gmsh mesh for model {model_name!r} is stale; import it "
        "with fem.io.gmsh.read() inside the owning geometry model context"
    )


@dataclass(frozen=True, slots=True)
class _AutoMeshPolicy:
    mode: _AutoMeshMode
    option_overrides: tuple[tuple[str, float], ...]
    order_one_types: frozenset[int]
    order_two_types: frozenset[int] | None

    def allowed_types(self, order: Literal[1, 2]) -> frozenset[int]:
        if order == 1:
            return self.order_one_types
        if self.order_two_types is None:
            raise ValueError(
                f"order must be 1 for automatic {self.mode!r} mesh generation"
            )
        return self.order_two_types


@dataclass(frozen=True, slots=True)
class _MeshGenerationPolicy:
    operation: _GenerationOperation
    order: Literal[1, 2]
    option_overrides: tuple[tuple[str, float], ...]
    mesh_size_factor: float | None = None
    requested_cell_shape: _AutoCellShape | None = None
    resolved_cell_shape: _AutoMeshMode | None = None
    allowed_top_cell_types: frozenset[int] | None = None
    strict_cell_shape: bool = False


_AUTO_MESH_POLICIES = {
    "line": _AutoMeshPolicy(
        "line",
        (
            ("Mesh.RecombineAll", 0.0),
            ("Mesh.SubdivisionAlgorithm", 0.0),
        ),
        frozenset({1}),
        None,
    ),
    "tri": _AutoMeshPolicy(
        "tri",
        (
            ("Mesh.RecombineAll", 0.0),
            ("Mesh.Algorithm", 6.0),
            ("Mesh.SubdivisionAlgorithm", 0.0),
        ),
        frozenset({2}),
        frozenset({9}),
    ),
    "tri-quad": _AutoMeshPolicy(
        "tri-quad",
        (
            ("Mesh.RecombineAll", 1.0),
            ("Mesh.Algorithm", 6.0),
            ("Mesh.RecombinationAlgorithm", 1.0),
            ("Mesh.SubdivisionAlgorithm", 0.0),
        ),
        frozenset({2, 3}),
        frozenset({9, 16}),
    ),
    "quad": _AutoMeshPolicy(
        "quad",
        (
            ("Mesh.RecombineAll", 1.0),
            ("Mesh.Algorithm", 6.0),
            ("Mesh.RecombinationAlgorithm", 3.0),
            ("Mesh.SubdivisionAlgorithm", 0.0),
        ),
        frozenset({3}),
        frozenset({16}),
    ),
    "tet": _AutoMeshPolicy(
        "tet",
        (
            ("Mesh.RecombineAll", 0.0),
            ("Mesh.Algorithm", 6.0),
            ("Mesh.Algorithm3D", 1.0),
            ("Mesh.Recombine3DAll", 0.0),
            ("Mesh.SubdivisionAlgorithm", 0.0),
        ),
        frozenset({4}),
        frozenset({11}),
    ),
    "hex": _AutoMeshPolicy(
        "hex",
        (
            ("Mesh.RecombineAll", 0.0),
            ("Mesh.Algorithm", 6.0),
            ("Mesh.Algorithm3D", 1.0),
            ("Mesh.Recombine3DAll", 0.0),
            ("Mesh.SubdivisionAlgorithm", 2.0),
        ),
        frozenset({5}),
        frozenset({17}),
    ),
}


@dataclass(frozen=True, slots=True)
class MeshFieldRef:
    """Immutable reference to one mesh field owned by a geometry model."""

    tag: int
    field_type: Literal["Distance", "Threshold", "Min"]
    _owner_token: object = field(repr=False)
    _field_token: object = field(repr=False)

    def __post_init__(self) -> None:
        _validate_positive_tag(self.tag, "mesh field tag")
        _validate_mesh_field_type(self.field_type)


@dataclass(frozen=True, slots=True)
class GmshMeshRef:
    """Read-only reference to a generated mesh in a live Gmsh model context."""

    dimension: Literal[1, 2, 3]
    model_name: str
    _owner: GeometryModel = field(repr=False)
    _owner_token: object = field(repr=False)
    _generation_token: object = field(repr=False)

    def __post_init__(self) -> None:
        _validate_mesh_dimension(self.dimension)
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a nonempty string")

    def _borrow_model(self) -> Any:
        """Return the live owning Gmsh model for the FEM IO adapter."""
        if not isinstance(self._owner, GeometryModel):
            raise _stale_gmsh_mesh_error(self.model_name)
        return GeometryModel._borrow_generated_mesh(self._owner, self)


class _State(Enum):
    NEW = auto()
    BUILDING_GEOMETRY = auto()
    CONFIGURING_MESH = auto()
    MESHED = auto()
    MESH_FAILED = auto()
    CLOSED = auto()


_QUERY_STATES = frozenset(
    {_State.BUILDING_GEOMETRY, _State.CONFIGURING_MESH, _State.MESH_FAILED}
)
_GEOMETRY_MUTATION_STATES = frozenset({_State.BUILDING_GEOMETRY})
_MESH_CONTROL_STATES = frozenset({_State.CONFIGURING_MESH})


class GeometryModel:
    """Context-managed owner of one scripted OCC model and its mesh attempt.

    Gmsh model, session, and option state is process-global. This facade
    supports same-thread nested contexts but is not thread-safe. Successful
    entity-dependent mesh controls protect their referenced topology from
    typed removal and from OCC transforms that would discard native controls.
    """

    def __init__(self, name: str, *, dimension: Literal[1, 2, 3]) -> None:
        self._name = name
        self._dimension = _validate_mesh_dimension(dimension)
        self._state = _State.NEW
        self._owner_token = object()
        self._entity_tokens: dict[tuple[int, int], object] = {}
        self._curve_loop_tokens: dict[int, object] = {}
        self._curve_loop_dependencies: dict[int, frozenset[tuple[int, int]]] = {}
        self._wire_tokens: dict[int, object] = {}
        self._wire_dependencies: dict[int, frozenset[tuple[int, int]]] = {}
        self._mesh_field_tokens: dict[int, object] = {}
        self._mesh_field_types: dict[
            int,
            Literal["Distance", "Threshold", "Min"],
        ] = {}
        self._mesh_size_mode: Literal["none", "point", "background"] = "none"
        self._background_field: MeshFieldRef | None = None
        self._entity_control_dependencies: set[tuple[int, int]] = set()
        self._transform_unsafe_control_dependencies: set[tuple[int, int]] = set()
        self._control_dependency_scope_unknown = False
        self._auto_mesh_blockers: set[str] = set()
        self._auto_mesh_scope_unknown = False
        self._gmsh: Any | None = None
        self._owns_session = False
        self._created_model = False
        self._prior_current: str | None = None
        self._pending_options: dict[str, float] = {}
        self._mesh_attempted = False
        self._generation_token: object | None = None
        self._mesher_token: object | None = None
        self._structured_extrusion_open = False

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
            self._gmsh = backend.load_gmsh()

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
            self._curve_loop_tokens.clear()
            self._curve_loop_dependencies.clear()
            self._wire_tokens.clear()
            self._wire_dependencies.clear()
            self._mesh_field_tokens.clear()
            self._mesh_field_types.clear()
            self._mesh_size_mode = "none"
            self._background_field = None
            self._entity_control_dependencies.clear()
            self._transform_unsafe_control_dependencies.clear()
            self._control_dependency_scope_unknown = False
            self._auto_mesh_blockers.clear()
            self._auto_mesh_scope_unknown = False
            self._mesh_attempted = False
            self._generation_token = None
            self._mesher_token = None
            self._structured_extrusion_open = False
            self._state = _State.BUILDING_GEOMETRY
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
            self._curve_loop_tokens.clear()
            self._curve_loop_dependencies.clear()
            self._wire_tokens.clear()
            self._wire_dependencies.clear()
            self._mesh_field_tokens.clear()
            self._mesh_field_types.clear()
            self._background_field = None
            self._entity_control_dependencies.clear()
            self._transform_unsafe_control_dependencies.clear()
            self._control_dependency_scope_unknown = False
            self._auto_mesh_blockers.clear()
            self._auto_mesh_scope_unknown = False
            self._generation_token = None
            self._mesher_token = None
            self._structured_extrusion_open = False
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
        if normalized_dimension > self.dimension:
            raise ValueError(
                "entities dimension must not exceed the geometry model dimension"
            )
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
        """Create an OCC point whose dimension does not exceed the model target."""
        operation = "point"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        coordinates = (
            _finite_float(x, "x"),
            _finite_float(y, "y"),
            _finite_float(z, "z"),
        )
        self._validate_2d_z(coordinates[2], operation)
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
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        endpoints = self._normalize_entities((start, end), operation=operation)
        if any(endpoint.dimension != 0 for endpoint in endpoints):
            raise ValueError("line endpoints must be dimension-zero point references")
        self._activate(operation)
        self._assert_occ_liveness(endpoints, operation)
        self._assert_planar_entities(endpoints, operation)
        coordinates = self._point_coordinates(endpoints, operation)
        if _coordinate_distance(coordinates[0], coordinates[1]) <= _PLANAR_TOLERANCE:
            raise ValueError("line endpoints must have distinct coordinates")
        tag = self._gmsh.model.occ.addLine(endpoints[0].tag, endpoints[1].tag)
        return self._wrap_entity((1, tag))

    def circular_arc(
        self,
        start: EntityRef,
        center: EntityRef,
        end: EntityRef,
    ) -> EntityRef:
        """Create a circular arc from ``start`` to ``end`` about ``center``."""
        operation = "circular_arc"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        points = self._normalize_entities(
            (start, center, end),
            operation=operation,
        )
        if any(point.dimension != 0 for point in points):
            raise ValueError(
                "circular_arc inputs must be dimension-zero point references"
            )
        self._activate(operation)
        self._assert_occ_liveness(points, operation)
        self._assert_planar_entities(points, operation)
        coordinates = self._point_coordinates(points, operation)
        start_radius = _coordinate_distance(coordinates[0], coordinates[1])
        end_radius = _coordinate_distance(coordinates[2], coordinates[1])
        if min(start_radius, end_radius) <= _PLANAR_TOLERANCE:
            raise ValueError("circular_arc radius must be positive")
        if not math.isclose(
            start_radius,
            end_radius,
            rel_tol=1.0e-10,
            abs_tol=_PLANAR_TOLERANCE,
        ):
            raise ValueError(
                "circular_arc start and end must be equidistant from center"
            )
        if _coordinate_distance(coordinates[0], coordinates[2]) <= _PLANAR_TOLERANCE:
            raise ValueError("circular_arc start and end must be distinct")
        tag = self._gmsh.model.occ.addCircleArc(
            points[0].tag,
            points[1].tag,
            points[2].tag,
        )
        return self._wrap_entity((1, tag))

    def elliptical_arc(
        self,
        start: EntityRef,
        center: EntityRef,
        major_axis_point: EntityRef,
        end: EntityRef,
    ) -> EntityRef:
        """Create an elliptical arc with an explicit major-axis point."""
        operation = "elliptical_arc"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        points = tuple(
            self._normalize_entities((point,), operation=f"{operation} {role}")[0]
            for role, point in zip(
                ("start", "center", "major_axis_point", "end"),
                (start, center, major_axis_point, end),
                strict=True,
            )
        )
        if any(point.dimension != 0 for point in points):
            raise ValueError(
                "elliptical_arc inputs must be dimension-zero point references"
            )
        self._activate(operation)
        self._assert_occ_liveness(points, operation)
        self._assert_planar_entities(points, operation)
        coordinates = self._point_coordinates(points, operation)
        if _coordinate_distance(coordinates[1], coordinates[2]) <= _PLANAR_TOLERANCE:
            raise ValueError(
                "elliptical_arc center and major_axis_point must be distinct"
            )
        if (
            _coordinate_distance(coordinates[0], coordinates[1])
            <= _PLANAR_TOLERANCE
            or _coordinate_distance(coordinates[3], coordinates[1])
            <= _PLANAR_TOLERANCE
        ):
            raise ValueError("elliptical_arc endpoints must differ from center")
        if _coordinate_distance(coordinates[0], coordinates[3]) <= _PLANAR_TOLERANCE:
            raise ValueError("elliptical_arc start and end must be distinct")
        _validate_elliptical_arc_geometry(coordinates)
        tag = self._gmsh.model.occ.addEllipseArc(
            points[0].tag,
            points[1].tag,
            points[2].tag,
            points[3].tag,
        )
        return self._wrap_entity((1, tag))

    def spline(self, points: Sequence[EntityRef]) -> EntityRef:
        """Create an interpolating spline through ordered control points."""
        return self._point_curve("spline", points, backend_name="addSpline")

    def bspline(self, points: Sequence[EntityRef]) -> EntityRef:
        """Create a B-spline through ordered control points."""
        return self._point_curve("bspline", points, backend_name="addBSpline")

    def orient(
        self,
        curve: EntityRef,
        *,
        reversed: bool = False,
    ) -> OrientedCurveRef:
        """Return an explicit traversal orientation for one live curve."""
        operation = "orient"
        self._check_state(operation, _QUERY_STATES)
        if not isinstance(reversed, bool):
            raise TypeError(f"reversed must be a boolean, got {reversed!r}")
        normalized = self._normalize_entities((curve,), operation=operation)[0]
        if normalized.dimension != 1:
            raise ValueError("orient requires a dimension-one curve reference")
        self._activate(operation)
        self._assert_occ_liveness((normalized,), operation)
        return OrientedCurveRef(normalized, reversed)

    def curve_loop(
        self,
        curves: Sequence[OrientedCurveRef],
    ) -> CurveLoopRef:
        """Create an owner-local closed loop from ordered, oriented curves."""
        operation = "curve_loop"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        try:
            oriented = tuple(curves)
        except TypeError as exc:
            raise TypeError("curve_loop curves must be iterable") from exc
        if not oriented:
            raise ValueError("curve_loop requires at least one oriented curve")
        if any(not isinstance(item, OrientedCurveRef) for item in oriented):
            raise TypeError(
                "curve_loop requires only OrientedCurveRef values returned by orient()"
            )
        normalized_curves = self._normalize_entities(
            tuple(item.curve for item in oriented),
            operation=operation,
        )
        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        self._assert_occ_liveness(normalized_curves, operation)
        self._assert_planar_entities(normalized_curves, operation)

        endpoints: list[tuple[tuple[int, int], tuple[int, int]]] = []
        curve_tags: list[int] = []
        for item in oriented:
            signed_tag = -item.curve.tag if item.reversed else item.curve.tag
            curve_tags.append(item.curve.tag)
            raw_boundary = self._gmsh.model.getBoundary(
                [(1, signed_tag)],
                combined=False,
                oriented=True,
                recursive=False,
            )
            boundary = tuple(_normalize_dim_tag(pair) for pair in raw_boundary)
            if len(boundary) != 2 or any(dimension != 0 for dimension, _ in boundary):
                raise GeometryError(
                    f"geometry model {self.name!r}: curve_loop could not determine "
                    f"two ordered endpoints for curve {item.curve.tag}"
                )
            endpoints.append((boundary[0], boundary[1]))
        for index, (_, end_point) in enumerate(endpoints):
            next_start = endpoints[(index + 1) % len(endpoints)][0]
            if end_point != next_start:
                raise ValueError(
                    "curve_loop curves must be continuous in the supplied order "
                    "and close at the final endpoint"
                )
        ordered_start_points = tuple(start_point for start_point, _ in endpoints)
        if len(oriented) > 1 and len(set(ordered_start_points)) != len(
            ordered_start_points
        ):
            raise ValueError(
                "curve_loop must not revisit a boundary point before final closure"
            )
        dependency_keys = frozenset(
            self._entity_boundary_closure_keys(
                normalized_curves,
                synchronize=False,
            )
        )

        # OCC curve loops auto-orient their positive curve tags. Negative tags
        # are intentionally kept out of this call: Gmsh's OCC layer can
        # interpret them as additive inner wires. The explicit typed
        # orientation above is used to validate traversal continuity.
        raw_tag = self._gmsh.model.occ.addCurveLoop(curve_tags)
        try:
            tag = _validate_positive_tag(raw_tag, "curve loop tag")
        except ValueError as error:
            self._curve_loop_tokens.clear()
            self._curve_loop_dependencies.clear()
            raise GeometryError(
                f"geometry model {self.name!r}: curve_loop returned an invalid "
                "loop tag; typed loop identities were invalidated"
            ) from error
        if tag in self._curve_loop_tokens:
            self._curve_loop_tokens.clear()
            self._curve_loop_dependencies.clear()
            raise GeometryError(
                f"geometry model {self.name!r}: curve_loop returned duplicate "
                f"loop tag {tag}; typed loop identities were invalidated"
            )
        token = object()
        reference = CurveLoopRef(tag, oriented, self._owner_token, token)
        self._curve_loop_tokens[tag] = token
        self._curve_loop_dependencies[tag] = dependency_keys
        return reference

    def wire(
        self,
        curves: Sequence[OrientedCurveRef],
        *,
        closed: bool,
    ) -> WireRef:
        """Create an owner-local ordered open or closed OCC wire."""
        operation = "wire"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        if not isinstance(closed, bool):
            raise TypeError(f"closed must be a boolean, got {closed!r}")
        try:
            oriented = tuple(curves)
        except TypeError as exc:
            raise TypeError("wire curves must be iterable") from exc
        if not oriented:
            raise ValueError("wire requires at least one oriented curve")
        if any(not isinstance(item, OrientedCurveRef) for item in oriented):
            raise TypeError(
                "wire requires only OrientedCurveRef values returned by orient()"
            )
        normalized_curves = self._normalize_entities(
            tuple(item.curve for item in oriented),
            operation=operation,
        )
        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        self._assert_occ_liveness(normalized_curves, operation)
        self._assert_planar_entities(normalized_curves, operation)

        endpoints = self._oriented_curve_endpoints(oriented, operation=operation)
        for (_, end_point), (next_start, _) in zip(
            endpoints,
            endpoints[1:],
        ):
            if end_point != next_start:
                raise ValueError(
                    "wire curves must form one continuous chain in the supplied order"
                )
        first_start = endpoints[0][0]
        final_end = endpoints[-1][1]
        if closed and final_end != first_start:
            raise ValueError("closed wire curves must close at the final endpoint")
        if not closed and final_end == first_start:
            raise ValueError("open wire start and end points must be distinct")

        traversal_points = (first_start, *(end for _, end in endpoints))
        unique_traversal = (
            traversal_points[:-1] if closed else traversal_points
        )
        if len(set(unique_traversal)) != len(unique_traversal):
            raise ValueError(
                "wire must not revisit a boundary point before its final endpoint"
            )

        dependency_keys = frozenset(
            self._entity_boundary_closure_keys(
                normalized_curves,
                synchronize=False,
            )
        )
        signed_tags = [
            -item.curve.tag if item.reversed else item.curve.tag
            for item in oriented
        ]
        try:
            raw_tag = self._gmsh.model.occ.addWire(signed_tags, -1, closed)
        except BaseException as error:
            self._fail_closed_after_unknown_occ_mutation()
            if isinstance(error, GeometryError) or not isinstance(error, Exception):
                raise
            raise GeometryError(
                f"geometry model {self.name!r}: native OCC wire failed"
            ) from error
        try:
            tag = _validate_positive_tag(raw_tag, "wire tag")
        except ValueError as error:
            self._wire_tokens.clear()
            self._wire_dependencies.clear()
            raise GeometryError(
                f"geometry model {self.name!r}: wire returned an invalid wire "
                "tag; typed wire identities were invalidated"
            ) from error
        if tag in self._wire_tokens:
            self._wire_tokens.clear()
            self._wire_dependencies.clear()
            raise GeometryError(
                f"geometry model {self.name!r}: wire returned duplicate wire "
                f"tag {tag}; typed wire identities were invalidated"
            )
        token = object()
        reference = WireRef(tag, oriented, closed, self._owner_token, token)
        self._wire_tokens[tag] = token
        self._wire_dependencies[tag] = dependency_keys
        return reference

    def plane_surface(
        self,
        outer: CurveLoopRef,
        *,
        holes: Sequence[CurveLoopRef] = (),
    ) -> EntityRef:
        """Create a planar surface from one outer loop and optional hole loops."""
        operation = "plane_surface"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        if self.dimension < 2:
            raise ValueError(
                "plane_surface requires a two- or three-dimensional geometry model"
            )
        try:
            materialized_holes = tuple(holes)
        except TypeError as exc:
            raise TypeError("plane_surface holes must be iterable") from exc
        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        loops = self._normalize_curve_loops(
            (outer, *materialized_holes),
            operation=operation,
        )
        member_curves = tuple(
            oriented.curve for loop in loops for oriented in loop.curves
        )
        self._assert_occ_liveness(member_curves, operation)
        self._assert_planar_entities(member_curves, operation)
        self._assert_plane_surface_loop_compatibility(loops, operation=operation)
        tag = self._gmsh.model.occ.addPlaneSurface([loop.tag for loop in loops])
        return self._wrap_entity((2, tag))

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
        self._check_state("rectangle", _GEOMETRY_MUTATION_STATES)
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
        self._check_state("disk", _GEOMETRY_MUTATION_STATES)
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
        self._check_state("box", _GEOMETRY_MUTATION_STATES)
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
        self._check_state("cylinder", _GEOMETRY_MUTATION_STATES)
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

    def intersect(
        self,
        objects: Iterable[EntityRef],
        tools: Iterable[EntityRef],
        *,
        remove_objects: bool = True,
        remove_tools: bool = True,
    ) -> BooleanResult:
        """Intersect internally homogeneous OCC input groups."""
        return self._boolean(
            "intersect",
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
        """Fragment or imprint OCC entities of same or mixed dimensions."""
        return self._boolean(
            "fragment",
            objects,
            tools,
            remove_objects=remove_objects,
            remove_tools=remove_tools,
        )

    def copy(
        self,
        entities: Iterable[EntityRef],
    ) -> tuple[EntityRef, ...]:
        """Copy OCC entities and return fresh references in caller order."""
        operation = "copy"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        normalized = self._normalize_entities(entities, operation=operation)
        self._activate(operation)
        self._assert_occ_liveness(normalized, operation)
        existing_pairs = {
            _normalize_dim_tag(pair) for pair in self._gmsh.model.occ.getEntities()
        }
        native_started = False
        try:
            grouped: dict[int, list[tuple[int, EntityRef]]] = {}
            for index, entity in enumerate(normalized):
                grouped.setdefault(entity.dimension, []).append((index, entity))
            reordered_pairs: list[tuple[int, int] | None] = [None] * len(normalized)
            for dimension_group in grouped.values():
                native_started = True
                raw_outputs = self._gmsh.model.occ.copy(
                    _dim_tags(entity for _, entity in dimension_group)
                )
                try:
                    batch_pairs = tuple(
                        _normalize_dim_tag(pair) for pair in raw_outputs
                    )
                except (TypeError, ValueError, GeometryError) as exc:
                    raise GeometryError(
                        f"geometry model {self.name!r}: copy returned invalid "
                        "entity data"
                    ) from exc
                if len(batch_pairs) != len(dimension_group):
                    raise GeometryError(
                        f"geometry model {self.name!r}: copy returned an "
                        "unexpected entity count"
                    )
                for (index, source), output in zip(
                    dimension_group,
                    batch_pairs,
                    strict=True,
                ):
                    if output[0] != source.dimension:
                        raise GeometryError(
                            f"geometry model {self.name!r}: copy returned an "
                            "entity with an unexpected dimension"
                        )
                    reordered_pairs[index] = output

            if any(pair is None for pair in reordered_pairs):
                raise GeometryError(
                    f"geometry model {self.name!r}: copy returned incomplete "
                    "entity data"
                )
            output_pairs = tuple(
                pair for pair in reordered_pairs if pair is not None
            )
            if len(set(output_pairs)) != len(output_pairs):
                raise GeometryError(
                    f"geometry model {self.name!r}: copy returned duplicate entities"
                )
            if any(pair in existing_pairs for pair in output_pairs):
                raise GeometryError(
                    f"geometry model {self.name!r}: copy did not return fresh entities"
                )
            current_pairs = {
                _normalize_dim_tag(pair)
                for pair in self._gmsh.model.occ.getEntities()
            }
            if any(pair not in current_pairs for pair in output_pairs):
                raise GeometryError(
                    f"geometry model {self.name!r}: copy returned a missing entity"
                )
            if any(
                (source.dimension, source.tag) not in current_pairs
                for source in normalized
            ):
                raise GeometryError(
                    f"geometry model {self.name!r}: copy removed a source entity"
                )
            return tuple(self._wrap_entity(pair) for pair in output_pairs)
        except BaseException:
            if native_started:
                self._fail_closed_after_unknown_occ_mutation()
            raise

    def translate(
        self,
        entities: Iterable[EntityRef],
        dx: float,
        dy: float,
        dz: float,
    ) -> tuple[EntityRef, ...]:
        """Translate entities in place and return their existing references."""
        operation = "translate"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        normalized = self._normalize_entities(entities, operation=operation)
        vector = (
            _finite_float(dx, "dx"),
            _finite_float(dy, "dy"),
            _finite_float(dz, "dz"),
        )
        if self.dimension == 2 and vector[2] != 0.0:
            raise ValueError("2D translation must remain in the global XY plane")
        self._check_control_dependency_scope_known(operation)
        self._activate(operation)
        self._assert_occ_liveness(normalized, operation)
        self._check_controlled_transform_allowed(operation, normalized)
        transformed_keys = self._entity_boundary_closure_keys(normalized)
        self._gmsh.model.occ.translate(_dim_tags(normalized), *vector)
        self._invalidate_curve_topology_for_keys(transformed_keys)
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
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
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
        self._check_control_dependency_scope_known(operation)
        self._activate(operation)
        self._assert_occ_liveness(normalized, operation)
        self._check_controlled_transform_allowed(operation, normalized)
        transformed_keys = self._entity_boundary_closure_keys(normalized)
        self._gmsh.model.occ.rotate(
            _dim_tags(normalized),
            *center,
            *axis,
            angle_value,
        )
        self._invalidate_curve_topology_for_keys(transformed_keys)
        return normalized

    def mirror(
        self,
        entities: Iterable[EntityRef],
        a: float,
        b: float,
        c: float,
        d: float,
    ) -> tuple[EntityRef, ...]:
        """Mirror entities in place about ``a*x + b*y + c*z + d = 0``."""
        operation = "mirror"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        normalized = self._normalize_entities(entities, operation=operation)
        plane = (
            _finite_float(a, "a"),
            _finite_float(b, "b"),
            _finite_float(c, "c"),
            _finite_float(d, "d"),
        )
        if not any(coefficient != 0.0 for coefficient in plane[:3]):
            raise ValueError("mirror plane normal must be nonzero")
        if self.dimension == 2 and not (
            plane[2] == 0.0
            or (plane[0] == 0.0 and plane[1] == 0.0 and plane[3] == 0.0)
        ):
            raise ValueError("2D mirror plane must preserve the global XY plane")
        self._check_control_dependency_scope_known(operation)
        self._activate(operation)
        self._assert_occ_liveness(normalized, operation)
        self._check_controlled_transform_allowed(operation, normalized)
        transformed_keys = self._entity_boundary_closure_keys(normalized)
        input_keys = {(entity.dimension, entity.tag) for entity in normalized}
        native_started = False
        try:
            native_started = True
            self._gmsh.model.occ.mirror(_dim_tags(normalized), *plane)
            self._assert_occ_liveness(normalized, operation)
            self._assert_planar_entities(normalized, operation)
            self._invalidate_entity_keys(transformed_keys - input_keys)
            self._invalidate_curve_topology_for_keys(input_keys)
            return normalized
        except BaseException:
            if native_started:
                self._fail_closed_after_unknown_occ_mutation()
            raise

    def scale(
        self,
        entities: Iterable[EntityRef],
        x: float,
        y: float,
        z: float,
        factor_x: float,
        factor_y: float,
        factor_z: float,
    ) -> tuple[EntityRef, ...]:
        """Scale entities in place about a center using nonzero factors.

        Negative factors are supported by Gmsh 4.15.2 and reverse the
        corresponding scaling direction without exposing oriented tags.
        """
        operation = "scale"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        normalized = self._normalize_entities(entities, operation=operation)
        center = (
            _finite_float(x, "x"),
            _finite_float(y, "y"),
            _finite_float(z, "z"),
        )
        factors = (
            _finite_float(factor_x, "factor_x"),
            _finite_float(factor_y, "factor_y"),
            _finite_float(factor_z, "factor_z"),
        )
        if any(factor == 0.0 for factor in factors):
            raise ValueError("scale factors must be nonzero")
        if (
            self.dimension == 2
            and abs(center[2] * (1.0 - factors[2])) > _PLANAR_TOLERANCE
        ):
            raise ValueError("2D scaling must preserve the global XY plane")
        self._check_control_dependency_scope_known(operation)
        self._activate(operation)
        self._assert_occ_liveness(normalized, operation)
        self._check_controlled_transform_allowed(operation, normalized)
        transformed_keys = self._entity_boundary_closure_keys(normalized)
        input_keys = {(entity.dimension, entity.tag) for entity in normalized}
        native_started = False
        try:
            native_started = True
            self._gmsh.model.occ.dilate(
                _dim_tags(normalized),
                *center,
                *factors,
            )
            self._assert_occ_liveness(normalized, operation)
            self._assert_planar_entities(
                normalized,
                operation,
                bounding_box_padding_scale=max(
                    1.0,
                    *(abs(factor) for factor in factors),
                ),
            )
            self._invalidate_entity_keys(transformed_keys - input_keys)
            self._invalidate_curve_topology_for_keys(input_keys)
            return normalized
        except BaseException:
            if native_started:
                self._fail_closed_after_unknown_occ_mutation()
            raise

    def revolve(
        self,
        entities: Iterable[EntityRef],
        x: float,
        y: float,
        z: float,
        axis_x: float,
        axis_y: float,
        axis_z: float,
        angle: float,
    ) -> FeatureResult:
        """Revolve curves or surfaces about an axis as a pure OCC feature."""
        operation = "revolve"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        normalized = self._normalize_entities(entities, operation=operation)
        dimensions = {entity.dimension for entity in normalized}
        if len(dimensions) != 1:
            raise ValueError("revolve inputs must have one common dimension")
        input_dimension = next(iter(dimensions))
        if input_dimension not in {1, 2}:
            raise ValueError("revolve supports dimension-one or dimension-two inputs")
        if input_dimension + 1 > self.dimension:
            raise ValueError(
                "revolve input dimension plus one exceeds the facade dimension"
            )
        axis_point = (
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
            raise ValueError("revolution axis must be nonzero")
        if angle_value == 0.0:
            raise ValueError("revolution angle must be nonzero")
        if abs(angle_value) > 2.0 * math.pi:
            raise ValueError("absolute revolution angle must not exceed 2*pi")
        if self.dimension == 2 and (axis[0] != 0.0 or axis[1] != 0.0):
            raise ValueError(
                "2D revolution axis must be parallel to the global Z axis"
            )

        self._activate(operation)
        self._assert_occ_liveness(normalized, operation)
        self._gmsh.model.occ.synchronize()
        source_signatures = tuple(
            self._entity_geometry_signature(entity, operation)
            for entity in normalized
        )
        source_boundaries = tuple(
            self._immediate_boundary_keys(entity, operation)
            for entity in normalized
        )
        native_started = False
        try:
            native_started = True
            raw_outputs = self._gmsh.model.occ.revolve(
                _dim_tags(normalized),
                *axis_point,
                *axis,
                angle_value,
                [],
                [],
                False,
            )
            output_pairs = tuple(
                _normalize_dim_tag(pair) for pair in raw_outputs
            )
            if not output_pairs:
                raise GeometryError(
                    f"geometry model {self.name!r}: revolve returned no entities"
                )
            allowed_dimensions = {input_dimension, input_dimension + 1}
            if any(pair[0] not in allowed_dimensions for pair in output_pairs):
                raise GeometryError(
                    f"geometry model {self.name!r}: revolve returned an entity "
                    "with an unexpected dimension"
                )
            if not any(
                pair[0] == input_dimension + 1 for pair in output_pairs
            ):
                raise GeometryError(
                    f"geometry model {self.name!r}: revolve returned no "
                    f"dimension-{input_dimension + 1} entity"
                )
            outputs = tuple(self._wrap_entity(pair) for pair in output_pairs)
            result = self._classify_revolution_result(
                inputs=normalized,
                outputs=outputs,
                axis_point=axis_point,
                axis=axis,
                angle=angle_value,
                source_signatures=source_signatures,
                source_boundaries=source_boundaries,
            )
            self._assert_occ_liveness(normalized, operation)
            self._assert_planar_entities(result.outputs, operation)
            return result
        except BaseException as error:
            if native_started:
                self._fail_closed_after_unknown_occ_mutation()
            if isinstance(error, GeometryError) or not isinstance(error, Exception):
                raise
            raise GeometryError(
                f"geometry model {self.name!r}: native OCC revolve failed"
            ) from error

    def sweep(
        self,
        entities: Iterable[EntityRef],
        path: WireRef,
        *,
        frame: SweepFrame = "discrete",
    ) -> FeatureResult:
        """Sweep curve or surface profiles along an owner-local path wire."""
        operation = "sweep"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        if self.dimension != 3:
            raise ValueError("sweep requires a three-dimensional geometry model")
        normalized = self._normalize_entities(entities, operation=operation)
        dimensions = {entity.dimension for entity in normalized}
        if len(dimensions) != 1:
            raise ValueError("sweep profiles must have one common dimension")
        input_dimension = next(iter(dimensions))
        if input_dimension not in {1, 2}:
            raise ValueError("sweep supports dimension-one or dimension-two profiles")
        if input_dimension + 1 > self.dimension:
            raise ValueError(
                "sweep profile dimension plus one exceeds the facade dimension"
            )
        if not isinstance(frame, str) or frame not in _SWEEP_FRAME_NAMES:
            raise ValueError(
                f"unsupported sweep frame {frame!r}; expected one of "
                f"{tuple(_SWEEP_FRAME_NAMES)}"
            )
        mapped_frame = _SWEEP_FRAME_NAMES[frame]  # type: ignore[index]

        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        normalized_path = self._normalize_wires((path,), operation=operation)[0]
        path_curves = tuple(item.curve for item in normalized_path.curves)
        if set(normalized) & set(path_curves):
            raise ValueError("sweep path must not reuse a profile entity")
        self._assert_occ_liveness((*normalized, *path_curves), operation)
        source_signatures = tuple(
            self._entity_geometry_signature(entity, operation)
            for entity in normalized
        )
        source_shapes = tuple(
            self._entity_rigid_shape_signature(entity, operation)
            for entity in normalized
        )
        native_started = False
        try:
            native_started = True
            raw_outputs = self._gmsh.model.occ.addPipe(
                _dim_tags(normalized),
                normalized_path.tag,
                mapped_frame,
            )
            output_pairs = tuple(
                _normalize_dim_tag(pair) for pair in raw_outputs
            )
            primary_dimension = input_dimension + 1
            if not output_pairs:
                raise GeometryError(
                    f"geometry model {self.name!r}: sweep returned no entities"
                )
            if any(pair[0] != primary_dimension for pair in output_pairs):
                raise GeometryError(
                    f"geometry model {self.name!r}: sweep returned an entity "
                    "with an unexpected dimension"
                )
            primary = tuple(self._wrap_entity(pair) for pair in output_pairs)
            result = self._classify_sweep_result(
                inputs=normalized,
                primary=primary,
                path=normalized_path,
                source_signatures=source_signatures,
                source_shapes=source_shapes,
            )
            self._assert_occ_liveness((*normalized, *path_curves), operation)
            return result
        except BaseException as error:
            if native_started:
                self._fail_closed_after_unknown_occ_mutation()
            if isinstance(error, GeometryError) or not isinstance(error, Exception):
                raise
            raise GeometryError(
                f"geometry model {self.name!r}: native OCC sweep failed"
            ) from error

    def loft(
        self,
        sections: Iterable[WireRef],
        *,
        solid: bool = True,
        ruled: bool = False,
        max_degree: int | None = None,
        continuity: LoftContinuity | None = None,
        parametrization: LoftParametrization | None = None,
        smoothing: bool = False,
    ) -> LoftResult:
        """Loft ordered wire sections into a surface or solid."""
        operation = "loft"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        if self.dimension != 3:
            raise ValueError("loft requires a three-dimensional geometry model")
        if not isinstance(solid, bool):
            raise TypeError(f"solid must be a boolean, got {solid!r}")
        if not isinstance(ruled, bool):
            raise TypeError(f"ruled must be a boolean, got {ruled!r}")
        if not isinstance(smoothing, bool):
            raise TypeError(f"smoothing must be a boolean, got {smoothing!r}")
        degree = (
            -1
            if max_degree is None
            else _integer_at_least(max_degree, "max_degree", minimum=1)
        )
        if continuity is None:
            mapped_continuity = ""
        elif isinstance(continuity, str) and continuity in _LOFT_CONTINUITY_NAMES:
            mapped_continuity = _LOFT_CONTINUITY_NAMES[continuity]  # type: ignore[index]
        else:
            raise ValueError(
                f"unsupported loft continuity {continuity!r}; expected one of "
                f"{tuple(_LOFT_CONTINUITY_NAMES)}"
            )
        if parametrization is None:
            mapped_parametrization = ""
        elif (
            isinstance(parametrization, str)
            and parametrization in _LOFT_PARAMETRIZATION_NAMES
        ):
            mapped_parametrization = _LOFT_PARAMETRIZATION_NAMES[parametrization]  # type: ignore[index]
        else:
            raise ValueError(
                f"unsupported loft parametrization {parametrization!r}; expected "
                f"one of {tuple(_LOFT_PARAMETRIZATION_NAMES)}"
            )

        try:
            materialized_sections = tuple(sections)
        except TypeError as exc:
            raise TypeError("loft sections must be iterable") from exc
        if len(materialized_sections) < 2:
            raise ValueError("loft requires at least two section wires")
        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        normalized_sections = self._normalize_wires(
            materialized_sections,
            operation=operation,
        )
        closed_states = {section.closed for section in normalized_sections}
        if len(closed_states) != 1:
            raise ValueError("loft sections must be either all open or all closed")
        if solid and not normalized_sections[0].closed:
            raise ValueError("solid loft requires closed section wires")
        flattened_inputs = tuple(
            item.curve
            for section in normalized_sections
            for item in section.curves
        )
        if len(set(flattened_inputs)) != len(flattened_inputs):
            raise ValueError("loft section member curves must be duplicate-free")
        self._assert_occ_liveness(flattened_inputs, operation)
        primary_dimension = 3 if solid else 2
        existing_before = {
            _normalize_dim_tag(pair) for pair in self._gmsh.model.occ.getEntities()
        }
        native_started = False
        try:
            native_started = True
            raw_outputs = self._gmsh.model.occ.addThruSections(
                [section.tag for section in normalized_sections],
                -1,
                solid,
                ruled,
                degree,
                mapped_continuity,
                mapped_parametrization,
                smoothing,
            )
            output_pairs = tuple(
                _normalize_dim_tag(pair) for pair in raw_outputs
            )
            if not output_pairs:
                raise GeometryError(
                    f"geometry model {self.name!r}: loft returned no entities"
                )
            if any(pair[0] != primary_dimension for pair in output_pairs):
                raise GeometryError(
                    f"geometry model {self.name!r}: loft returned an entity "
                    "with an unexpected dimension"
                )
            if len(set(output_pairs)) != len(output_pairs):
                raise GeometryError(
                    f"geometry model {self.name!r}: loft returned duplicate entities"
                )
            self._gmsh.model.occ.synchronize()
            existing_after = {
                _normalize_dim_tag(pair)
                for pair in self._gmsh.model.occ.getEntities()
            }
            missing_outputs = set(output_pairs) - existing_after
            if missing_outputs:
                missing = min(missing_outputs)
                raise GeometryError(
                    f"geometry model {self.name!r}: loft returned a missing "
                    f"entity {missing}"
                )
            aliased_outputs = set(output_pairs) & existing_before
            if aliased_outputs:
                aliased = min(aliased_outputs)
                raise GeometryError(
                    f"geometry model {self.name!r}: loft returned an existing "
                    f"entity {aliased}"
                )
            self._assert_occ_liveness(flattened_inputs, operation)
            outputs = tuple(self._wrap_entity(pair) for pair in output_pairs)
            topology = FeatureResult(
                operation,
                flattened_inputs,
                outputs,
                _unique_first_seen(outputs),
            )
            return LoftResult(topology, normalized_sections)
        except BaseException as error:
            if native_started:
                self._fail_closed_after_unknown_occ_mutation()
            if isinstance(error, GeometryError) or not isinstance(error, Exception):
                raise
            raise GeometryError(
                f"geometry model {self.name!r}: native OCC loft failed"
            ) from error

    def fillet(
        self,
        volumes: Iterable[EntityRef],
        curves: Iterable[EntityRef],
        radii: Sequence[float],
        *,
        remove_volumes: bool = True,
    ) -> FeatureResult:
        """Fillet selected boundary curves of three-dimensional volumes."""
        operation = "fillet"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        if self.dimension != 3:
            raise ValueError("fillet requires a three-dimensional geometry model")
        normalized_volumes = self._normalize_entities(volumes, operation=operation)
        normalized_curves = self._normalize_entities(
            curves,
            operation=f"{operation} curves",
        )
        if any(volume.dimension != 3 for volume in normalized_volumes):
            raise ValueError("fillet volumes must be dimension-three entities")
        if any(curve.dimension != 1 for curve in normalized_curves):
            raise ValueError("fillet curves must be dimension-one entities")
        normalized_radii = _positive_feature_vector(
            radii,
            count=len(normalized_curves),
            label="radii",
        )
        if not isinstance(remove_volumes, bool):
            raise TypeError(
                f"remove_volumes must be a boolean, got {remove_volumes!r}"
            )
        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        self._assert_occ_liveness(
            (*normalized_volumes, *normalized_curves),
            operation,
        )
        closures = tuple(
            self._entity_boundary_closure_keys((volume,), synchronize=False)
            for volume in normalized_volumes
        )
        if any(
            not any((curve.dimension, curve.tag) in closure for closure in closures)
            for curve in normalized_curves
        ):
            raise ValueError(
                "every fillet curve must belong to at least one selected volume"
            )
        return self._apply_volume_edge_treatment(
            operation=operation,
            volumes=normalized_volumes,
            curves=normalized_curves,
            surfaces=(),
            values=normalized_radii,
            closures=closures,
            remove_volumes=remove_volumes,
        )

    def chamfer(
        self,
        volumes: Iterable[EntityRef],
        curves: Iterable[EntityRef],
        surfaces: Iterable[EntityRef],
        distances: Sequence[float],
        *,
        remove_volumes: bool = True,
    ) -> FeatureResult:
        """Chamfer explicit boundary curve and adjacent-surface pairs."""
        operation = "chamfer"
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        if self.dimension != 3:
            raise ValueError("chamfer requires a three-dimensional geometry model")
        normalized_volumes = self._normalize_entities(volumes, operation=operation)
        normalized_curves = self._normalize_entities(
            curves,
            operation=f"{operation} curves",
        )
        normalized_surfaces = self._normalize_entities(
            surfaces,
            operation=f"{operation} surfaces",
        )
        if any(volume.dimension != 3 for volume in normalized_volumes):
            raise ValueError("chamfer volumes must be dimension-three entities")
        if any(curve.dimension != 1 for curve in normalized_curves):
            raise ValueError("chamfer curves must be dimension-one entities")
        if any(surface.dimension != 2 for surface in normalized_surfaces):
            raise ValueError("chamfer surfaces must be dimension-two entities")
        if len(normalized_curves) != len(normalized_surfaces):
            raise ValueError(
                "chamfer curve and surface sequences must have equal nonzero lengths"
            )
        normalized_distances = _positive_feature_vector(
            distances,
            count=len(normalized_curves),
            label="distances",
        )
        if not isinstance(remove_volumes, bool):
            raise TypeError(
                f"remove_volumes must be a boolean, got {remove_volumes!r}"
            )
        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        self._assert_occ_liveness(
            (*normalized_volumes, *normalized_curves, *normalized_surfaces),
            operation,
        )
        closures = tuple(
            self._entity_boundary_closure_keys((volume,), synchronize=False)
            for volume in normalized_volumes
        )
        for curve, surface in zip(
            normalized_curves,
            normalized_surfaces,
            strict=True,
        ):
            curve_key = (curve.dimension, curve.tag)
            surface_key = (surface.dimension, surface.tag)
            if curve_key not in self._immediate_boundary_keys(surface, operation):
                raise ValueError(
                    "each chamfer curve must be adjacent to its paired surface"
                )
            if not any(
                curve_key in closure and surface_key in closure
                for closure in closures
            ):
                raise ValueError(
                    "each chamfer pair must belong to a selected volume"
                )
        return self._apply_volume_edge_treatment(
            operation=operation,
            volumes=normalized_volumes,
            curves=normalized_curves,
            surfaces=normalized_surfaces,
            values=normalized_distances,
            closures=closures,
            remove_volumes=remove_volumes,
        )

    def extrude(
        self,
        entities: Iterable[EntityRef],
        dx: float,
        dy: float,
        dz: float,
    ) -> FeatureResult:
        """Extrude OCC entities without applying native mesh-layer controls."""
        self._check_state("extrude", _GEOMETRY_MUTATION_STATES)
        return self._extrude(
            entities,
            dx,
            dy,
            dz,
            operation="extrude",
            layer_counts=(),
            normalized_heights=(),
            recombine=False,
            structured=False,
        )

    def _structured_extrude(
        self,
        mesher_token: object,
        entities: Iterable[EntityRef],
        dx: float,
        dy: float,
        dz: float,
        *,
        num_elements: Sequence[int],
        heights: Sequence[float],
        recombine: bool,
    ) -> FeatureResult:
        operation = "structured_extrude"
        self._assert_mesher_authority(mesher_token, operation)
        if not self._structured_extrusion_open:
            raise self._state_error(
                operation,
                "the structured-extrusion subphase was closed by an ordinary "
                "mesh control, field, or generation attempt",
            )
        layer_counts = _positive_integer_sequence(num_elements, "num_elements")
        try:
            materialized_heights = tuple(heights)
        except TypeError as exc:
            raise TypeError("heights must be a sequence of finite numbers") from exc
        normalized_heights = tuple(
            _finite_float(value, "height") for value in materialized_heights
        )
        if not isinstance(recombine, bool):
            raise TypeError(f"recombine must be a boolean, got {recombine!r}")
        if not layer_counts and not recombine:
            raise ValueError(
                "structured_extrude requires nonempty num_elements or "
                "recombine=True; use GeometryModel.extrude() for pure geometry"
            )
        return self._extrude(
            entities,
            dx,
            dy,
            dz,
            operation=operation,
            layer_counts=layer_counts,
            normalized_heights=normalized_heights,
            recombine=recombine,
            structured=True,
        )

    def _extrude(
        self,
        entities: Iterable[EntityRef],
        dx: float,
        dy: float,
        dz: float,
        *,
        operation: str,
        layer_counts: tuple[int, ...],
        normalized_heights: tuple[float, ...],
        recombine: bool,
        structured: bool,
    ) -> FeatureResult:
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
        self._activate(operation)
        self._assert_occ_liveness(normalized, operation)
        self._gmsh.model.occ.synchronize()
        source_signatures = tuple(
            self._entity_geometry_signature(entity, operation)
            for entity in normalized
        )
        source_boundaries = tuple(
            self._immediate_boundary_keys(entity, operation)
            for entity in normalized
        )
        native_started = False
        prior_unknown_scope = self._control_dependency_scope_unknown
        prior_auto_mesh_scope_unknown = self._auto_mesh_scope_unknown
        try:
            native_started = True
            output_pairs = self._gmsh.model.occ.extrude(
                _dim_tags(normalized),
                *vector,
                list(layer_counts),
                list(normalized_heights),
                recombine,
            )
            if structured:
                self._control_dependency_scope_unknown = True
                self._auto_mesh_scope_unknown = True
            validated_pairs = tuple(
                _normalize_dim_tag(pair) for pair in output_pairs
            )
            if not validated_pairs:
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} returned no entities"
                )
            allowed_output_dimensions = {input_dimension, input_dimension + 1}
            if any(
                dimension not in allowed_output_dimensions
                for dimension, _ in validated_pairs
            ):
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} returned an "
                    "entity with an unexpected dimension"
                )
            if not any(
                dimension == input_dimension + 1
                for dimension, _ in validated_pairs
            ):
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} returned no "
                    f"dimension-{input_dimension + 1} entity"
                )
            outputs = tuple(self._wrap_entity(pair) for pair in validated_pairs)
            result = self._classify_extrusion_result(
                operation=operation,
                inputs=normalized,
                outputs=outputs,
                vector=vector,
                source_signatures=source_signatures,
                source_boundaries=source_boundaries,
            )
            if structured:
                dependency_keys = self._entity_boundary_closure_keys(
                    (*normalized, *outputs)
                )
                self._register_control_dependencies(
                    dependency_keys,
                    transform_unsafe=True,
                )
                self._auto_mesh_blockers.add(operation)
                self._control_dependency_scope_unknown = prior_unknown_scope
                self._auto_mesh_scope_unknown = prior_auto_mesh_scope_unknown
            return result
        except BaseException as error:
            if native_started:
                self._fail_closed_after_unknown_occ_mutation()
                if structured:
                    self._state = _State.MESH_FAILED
                    self._structured_extrusion_open = False
                    error.add_note(
                        f"geometry model {self.name!r}: structured extrusion failed "
                        "after native OCC mutation began"
                    )
            raise

    def entity(self, dimension: int, tag: int) -> EntityRef:
        """Acquire a typed reference for an existing raw OCC entity."""
        operation = "entity"
        self._check_state(operation, _QUERY_STATES)
        normalized = (
            _validate_entity_dimension(dimension),
            _validate_positive_tag(tag, "entity tag"),
        )
        if normalized[0] > self.dimension:
            raise ValueError(
                "entity dimension must not exceed the geometry model dimension"
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

    def bounding_box(
        self,
        entity: EntityRef,
    ) -> tuple[float, float, float, float, float, float]:
        """Return the native OCC bounding box of one live entity."""
        operation = "bounding_box"
        target = self._prepare_geometry_query_entity(entity, operation=operation)
        bounds = tuple(
            _finite_float(value, "bounding box coordinate")
            for value in self._gmsh.model.getBoundingBox(
                target.dimension,
                target.tag,
            )
        )
        if len(bounds) != 6:
            raise GeometryError(
                f"geometry model {self.name!r}: invalid bounding box for "
                f"entity ({target.dimension}, {target.tag})"
            )
        if any(bounds[axis] > bounds[axis + 3] for axis in range(3)):
            raise GeometryError(
                f"geometry model {self.name!r}: inverted bounding box for "
                f"entity ({target.dimension}, {target.tag})"
            )
        return bounds  # type: ignore[return-value]

    def length(self, curve: EntityRef) -> float:
        """Return the OCC length of one live curve."""
        target = self._prepare_geometry_query_entity(curve, operation="length")
        if target.dimension != 1:
            raise ValueError("length requires a dimension-one curve reference")
        return _nonnegative_float(
            self._gmsh.model.occ.getMass(1, target.tag),
            "curve length",
        )

    def area(self, surface: EntityRef) -> float:
        """Return the OCC area of one live surface."""
        target = self._prepare_geometry_query_entity(surface, operation="area")
        if target.dimension != 2:
            raise ValueError("area requires a dimension-two surface reference")
        return _nonnegative_float(
            self._gmsh.model.occ.getMass(2, target.tag),
            "surface area",
        )

    def center_of_mass(
        self,
        entity: EntityRef,
    ) -> tuple[float, float, float]:
        """Return the OCC center of mass of one live entity."""
        operation = "center_of_mass"
        target = self._prepare_geometry_query_entity(entity, operation=operation)
        if target.dimension == 0:
            bounds = self.bounding_box(target)
            values = tuple(
                0.5 * (bounds[axis] + bounds[axis + 3]) for axis in range(3)
            )
        else:
            values = tuple(
                _finite_float(value, "center-of-mass coordinate")
                for value in self._gmsh.model.occ.getCenterOfMass(
                    target.dimension,
                    target.tag,
                )
            )
        if len(values) != 3:
            raise GeometryError(
                f"geometry model {self.name!r}: invalid center of mass for "
                f"entity ({target.dimension}, {target.tag})"
            )
        return values  # type: ignore[return-value]

    def adjacent(
        self,
        entity: EntityRef,
        *,
        dimension: int,
    ) -> tuple[EntityRef, ...]:
        """Return deterministic immediate adjacencies at one neighboring dimension."""
        operation = "adjacent"
        target = self._prepare_geometry_query_entity(entity, operation=operation)
        requested_dimension = _validate_entity_dimension(dimension)
        if requested_dimension > self.dimension:
            raise ValueError(
                "adjacent dimension must not exceed the geometry model dimension"
            )
        if requested_dimension not in {
            target.dimension - 1,
            target.dimension + 1,
        }:
            raise ValueError(
                "adjacent dimension must differ from the entity dimension by one"
            )
        raw_upward, raw_downward = self._gmsh.model.getAdjacencies(
            target.dimension,
            target.tag,
        )
        raw_tags = raw_upward if requested_dimension > target.dimension else raw_downward
        tags = sorted(
            {
                _validate_positive_tag(value, "adjacent entity tag")
                for value in raw_tags
            }
        )
        existing = {
            _normalize_dim_tag(pair)
            for pair in self._gmsh.model.occ.getEntities(requested_dimension)
        }
        missing = [
            tag for tag in tags if (requested_dimension, tag) not in existing
        ]
        if missing:
            raise GeometryError(
                f"geometry model {self.name!r}: adjacent returned missing "
                f"entity ({requested_dimension}, {missing[0]})"
            )
        return tuple(
            self._wrap_entity((requested_dimension, tag)) for tag in tags
        )

    def _bind_mesher(self) -> object:
        """Atomically seal geometry mutation and issue the sole mesher capability."""
        operation = "Mesher binding"
        if self._mesher_token is not None:
            raise self._state_error(operation, "a Mesher is already bound")
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        self._activate(operation)
        token = object()
        self._mesher_token = token
        self._structured_extrusion_open = True
        self._state = _State.CONFIGURING_MESH
        return token

    def _assert_mesher_authority(
        self,
        mesher_token: object,
        operation: str,
    ) -> None:
        if self._mesher_token is None or mesher_token is not self._mesher_token:
            raise self._state_error(operation, "the bound Mesher capability is invalid")
        self._check_state(operation, _MESH_CONTROL_STATES)

    def _complete_mesh_configuration_operation(
        self,
        mesher_token: object,
        operation: str,
    ) -> None:
        self._assert_mesher_authority(mesher_token, operation)
        self._structured_extrusion_open = False

    def _mesher_transfinite_curve(
        self,
        mesher_token: object,
        curve: EntityRef,
        *,
        num_nodes: int,
    ) -> None:
        """Set Gmsh's primary-node count, not an element count, on one curve."""
        operation = "transfinite_curve"
        self._assert_mesher_authority(mesher_token, operation)
        node_count = _integer_at_least(num_nodes, "num_nodes", minimum=2)
        target = self._prepare_mesh_control_target(
            curve,
            dimension=1,
            operation=operation,
        )
        dependency_keys = self._entity_boundary_closure_keys(
            (target,),
            synchronize=False,
        )
        self._complete_mesh_configuration_operation(mesher_token, operation)
        self._gmsh.model.mesh.setTransfiniteCurve(target.tag, node_count)
        self._register_control_dependencies(dependency_keys, transform_unsafe=True)
        self._auto_mesh_blockers.add(operation)

    def _mesher_transfinite_surface(
        self,
        mesher_token: object,
        surface: EntityRef,
        *,
        corners: Sequence[EntityRef] = (),
    ) -> None:
        """Mark one surface as transfinite with optional boundary corners.

        Gmsh remains responsible for topology suitability. Call ``recombine``
        separately when quadrilateral output is desired.
        """
        operation = "transfinite_surface"
        self._assert_mesher_authority(mesher_token, operation)
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
        dependency_keys = self._entity_boundary_closure_keys(
            (target, *normalized_corners),
            synchronize=False,
        )
        self._complete_mesh_configuration_operation(mesher_token, operation)
        self._gmsh.model.mesh.setTransfiniteSurface(
            target.tag,
            cornerTags=[corner.tag for corner in normalized_corners],
        )
        self._register_control_dependencies(dependency_keys, transform_unsafe=True)
        self._auto_mesh_blockers.add(operation)

    def _mesher_transfinite_volume(
        self,
        mesher_token: object,
        volume: EntityRef,
        *,
        corners: Sequence[EntityRef] = (),
    ) -> None:
        """Mark one volume as transfinite without configuring its boundary.

        The caller must constrain suitable boundary curves and surfaces; Gmsh
        remains responsible for rejecting incompatible or unsuitable topology.
        """
        operation = "transfinite_volume"
        self._assert_mesher_authority(mesher_token, operation)
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
        dependency_keys = self._entity_boundary_closure_keys(
            (target, *normalized_corners),
            synchronize=False,
        )
        self._complete_mesh_configuration_operation(mesher_token, operation)
        self._gmsh.model.mesh.setTransfiniteVolume(
            target.tag,
            cornerTags=[corner.tag for corner in normalized_corners],
        )
        self._register_control_dependencies(dependency_keys, transform_unsafe=True)
        self._auto_mesh_blockers.add(operation)

    def _mesher_recombine(
        self,
        mesher_token: object,
        surface: EntityRef,
    ) -> None:
        """Request native Gmsh recombination on one surface.

        The entity-local request retains Gmsh's default angle and does not
        guarantee an all-quadrilateral mesh for unsuitable topology.
        """
        operation = "recombine"
        self._assert_mesher_authority(mesher_token, operation)
        target = self._prepare_mesh_control_target(
            surface,
            dimension=2,
            operation=operation,
        )
        dependency_keys = self._entity_boundary_closure_keys(
            (target,),
            synchronize=False,
        )
        self._complete_mesh_configuration_operation(mesher_token, operation)
        self._gmsh.model.mesh.setRecombine(2, target.tag)
        self._register_control_dependencies(dependency_keys, transform_unsafe=True)
        self._auto_mesh_blockers.add(operation)

    def _mesher_mesh_size(
        self,
        mesher_token: object,
        points: Iterable[EntityRef],
        *,
        size: float,
    ) -> None:
        """Assign one mesh size to selected live OCC points."""
        operation = "mesh_size"
        self._assert_mesher_authority(mesher_token, operation)
        if self._mesh_size_mode == "background":
            raise ValueError(
                "mesh_size cannot be combined with a selected background field"
            )
        size_value = _positive_float(size, "size")
        normalized = self._normalize_entities(points, operation=operation)
        if any(point.dimension != 0 for point in normalized):
            raise ValueError("mesh_size requires dimension-zero point references")

        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        self._assert_occ_liveness(normalized, operation)
        self._complete_mesh_configuration_operation(mesher_token, operation)
        self._gmsh.model.mesh.setSize(_dim_tags(normalized), size_value)
        self._mesh_size_mode = "point"
        self._register_control_dependencies(
            _dim_tags(normalized),
            transform_unsafe=True,
        )

    def _mesher_distance_field(
        self,
        mesher_token: object,
        *,
        points: Iterable[EntityRef] = (),
        curves: Iterable[EntityRef] = (),
        surfaces: Iterable[EntityRef] = (),
        sampling: int = 20,
    ) -> MeshFieldRef:
        """Create a field measuring distance from selected OCC entities."""
        operation = "distance_field"
        self._assert_mesher_authority(mesher_token, operation)
        normalized_points = self._normalize_optional_entities(
            points,
            operation=operation,
            label="points",
        )
        normalized_curves = self._normalize_optional_entities(
            curves,
            operation=operation,
            label="curves",
        )
        normalized_surfaces = self._normalize_optional_entities(
            surfaces,
            operation=operation,
            label="surfaces",
        )
        source_groups = (
            ("points", 0, normalized_points),
            ("curves", 1, normalized_curves),
            ("surfaces", 2, normalized_surfaces),
        )
        all_sources = tuple(
            entity
            for _, _, group in source_groups
            for entity in group
        )
        if not all_sources:
            raise ValueError("distance_field requires at least one source entity")
        for label, dimension, group in source_groups:
            if any(entity.dimension != dimension for entity in group):
                raise ValueError(
                    f"distance_field {label} must be dimension-{dimension} "
                    "entity references"
                )
        source_keys = [(entity.dimension, entity.tag) for entity in all_sources]
        if len(set(source_keys)) != len(source_keys):
            raise ValueError("distance_field source entities must be duplicate-free")
        sampling_value = _integer_at_least(sampling, "sampling", minimum=2)

        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        self._assert_occ_liveness(all_sources, operation)
        dependency_keys = self._entity_boundary_closure_keys(
            all_sources,
            synchronize=False,
        )

        def configure(field_tag: int) -> None:
            manager = self._gmsh.model.mesh.field
            if normalized_points:
                manager.setNumbers(
                    field_tag,
                    "PointsList",
                    [point.tag for point in normalized_points],
                )
            if normalized_curves:
                manager.setNumbers(
                    field_tag,
                    "CurvesList",
                    [curve.tag for curve in normalized_curves],
                )
            if normalized_surfaces:
                manager.setNumbers(
                    field_tag,
                    "SurfacesList",
                    [surface.tag for surface in normalized_surfaces],
                )
            manager.setNumber(field_tag, "Sampling", sampling_value)

        self._complete_mesh_configuration_operation(mesher_token, operation)
        mesh_field = self._construct_mesh_field("Distance", configure)
        self._register_control_dependencies(
            dependency_keys,
            transform_unsafe=False,
        )
        return mesh_field

    def _mesher_threshold_field(
        self,
        mesher_token: object,
        distance: MeshFieldRef,
        *,
        size_min: float,
        size_max: float,
        dist_min: float,
        dist_max: float,
    ) -> MeshFieldRef:
        """Map one distance field to near- and far-field mesh sizes."""
        operation = "threshold_field"
        self._assert_mesher_authority(mesher_token, operation)
        normalized_distance = self._normalize_mesh_fields(
            (distance,),
            operation=operation,
        )[0]
        if normalized_distance.field_type != "Distance":
            raise ValueError("threshold_field requires a Distance field")
        size_min_value = _positive_float(size_min, "size_min")
        size_max_value = _positive_float(size_max, "size_max")
        if size_min_value >= size_max_value:
            raise ValueError("size_min must be less than size_max")
        dist_min_value = _nonnegative_float(dist_min, "dist_min")
        dist_max_value = _finite_float(dist_max, "dist_max")
        if dist_max_value <= dist_min_value:
            raise ValueError("dist_max must be greater than dist_min")

        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        self._assert_mesh_field_liveness((normalized_distance,), operation)

        def configure(field_tag: int) -> None:
            manager = self._gmsh.model.mesh.field
            manager.setNumber(field_tag, "InField", normalized_distance.tag)
            manager.setNumber(field_tag, "SizeMin", size_min_value)
            manager.setNumber(field_tag, "SizeMax", size_max_value)
            manager.setNumber(field_tag, "DistMin", dist_min_value)
            manager.setNumber(field_tag, "DistMax", dist_max_value)

        self._complete_mesh_configuration_operation(mesher_token, operation)
        return self._construct_mesh_field("Threshold", configure)

    def _mesher_min_field(
        self,
        mesher_token: object,
        fields: Sequence[MeshFieldRef],
    ) -> MeshFieldRef:
        """Create the pointwise minimum of two or more size fields."""
        operation = "min_field"
        self._assert_mesher_authority(mesher_token, operation)
        try:
            materialized = tuple(fields)
        except TypeError as exc:
            raise TypeError("min_field fields must be iterable") from exc
        if len(materialized) < 2:
            raise ValueError("min_field requires at least two fields")
        normalized = self._normalize_mesh_fields(
            materialized,
            operation=operation,
        )
        if any(field.field_type not in {"Threshold", "Min"} for field in normalized):
            raise ValueError(
                "min_field accepts only Threshold and Min size fields"
            )

        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        self._assert_mesh_field_liveness(normalized, operation)

        def configure(field_tag: int) -> None:
            self._gmsh.model.mesh.field.setNumbers(
                field_tag,
                "FieldsList",
                [item.tag for item in normalized],
            )

        self._complete_mesh_configuration_operation(mesher_token, operation)
        return self._construct_mesh_field("Min", configure)

    def _mesher_background_field(
        self,
        mesher_token: object,
        field: MeshFieldRef,
    ) -> None:
        """Select exactly one size-producing field as the background field."""
        operation = "background_field"
        self._assert_mesher_authority(mesher_token, operation)
        if self._background_field is not None:
            raise ValueError("background_field may be selected only once")
        if self._mesh_size_mode == "point":
            raise ValueError(
                "background_field cannot be combined with typed point sizes"
            )
        normalized = self._normalize_mesh_fields(
            (field,),
            operation=operation,
        )[0]
        if normalized.field_type not in {"Threshold", "Min"}:
            raise ValueError(
                "background_field requires a Threshold or Min size field"
            )

        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        self._assert_mesh_field_liveness((normalized,), operation)
        self._complete_mesh_configuration_operation(mesher_token, operation)
        self._gmsh.model.mesh.field.setAsBackgroundMesh(normalized.tag)
        self._background_field = normalized
        self._mesh_size_mode = "background"

    def _mesher_generate_mesh(
        self,
        mesher_token: object,
        *,
        size: float | None = None,
        order: Literal[1, 2] = 1,
        recombine: bool = False,
    ) -> GmshMeshRef:
        """Generate the one native mesh permitted for this facade model."""
        self._assert_mesher_authority(mesher_token, "generate")
        return self._generate_mesh(
            size=size,
            order=order,
            recombine=recombine,
        )

    def _mesher_generate_auto_mesh(
        self,
        mesher_token: object,
        *,
        level: Literal[1, 2, 3, 4, 5] = 3,
        cell_shape: Literal[
            "tri",
            "tri-quad",
            "quad",
            "tet",
            "hex",
        ]
        | None = None,
        order: Literal[1, 2] = 1,
    ) -> GmshMeshRef:
        """Generate one level-scaled, strict-shape native mesh."""
        self._assert_mesher_authority(mesher_token, "generate")
        return self._generate_auto_mesh(
            level=level,
            cell_shape=cell_shape,
            order=order,
        )

    def _generate_mesh(
        self,
        *,
        size: float | None,
        order: Literal[1, 2],
        recombine: bool,
    ) -> GmshMeshRef:
        operation = "MeshSpec generation"
        self._check_state(
            operation,
            _MESH_CONTROL_STATES,
        )
        if self._mesh_attempted:
            raise self._state_error(operation, "the one mesh attempt was already used")
        size_value = None if size is None else _positive_float(size, "size")
        if size_value is not None and self._mesh_size_mode != "none":
            raise ValueError(
                "size cannot be supplied after typed point sizes or a typed "
                "background field has been configured"
            )
        if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2):
            raise ValueError(f"order must be integer 1 or 2, got {order!r}")
        if not isinstance(recombine, bool):
            raise TypeError(f"recombine must be a boolean, got {recombine!r}")
        if self.dimension == 1 and order != 1:
            raise ValueError("order must be 1 for a one-dimensional geometry model")
        if self.dimension == 1 and recombine:
            raise ValueError("recombine must be False for a one-dimensional geometry model")

        policy = _MeshGenerationPolicy(
            operation=operation,
            order=order,
            option_overrides=(
                ("Mesh.ElementOrder", float(order)),
                (
                    "Mesh.SecondOrderIncomplete",
                    1.0 if order == 2 else 0.0,
                ),
                ("Mesh.RecombineAll", 1.0 if recombine else 0.0),
            ),
        )
        return self._generate_native_mesh(
            policy=policy,
            size_value=size_value,
        )

    def _generate_auto_mesh(
        self,
        *,
        level: Literal[1, 2, 3, 4, 5],
        cell_shape: _AutoCellShape | None,
        order: Literal[1, 2],
    ) -> GmshMeshRef:
        operation = "AutoMeshSpec generation"
        self._check_state(
            operation,
            _MESH_CONTROL_STATES,
        )
        if self._mesh_attempted:
            raise self._state_error(operation, "the one mesh attempt was already used")

        normalized_level = _validate_auto_mesh_level(level)
        resolved_cell_shape = _resolve_auto_mesh_mode(
            self.dimension,
            cell_shape,
        )
        if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2):
            raise ValueError(f"order must be integer 1 or 2, got {order!r}")
        if self.dimension == 1 and order != 1:
            raise ValueError("order must be 1 for a one-dimensional geometry model")
        self._check_auto_mesh_controls()

        auto_policy = _AUTO_MESH_POLICIES[resolved_cell_shape]
        size_factor = 2.0 ** ((3 - normalized_level) / self.dimension)
        policy = _MeshGenerationPolicy(
            operation=operation,
            order=order,
            option_overrides=(
                ("Mesh.ElementOrder", float(order)),
                (
                    "Mesh.SecondOrderIncomplete",
                    1.0 if order == 2 else 0.0,
                ),
                *auto_policy.option_overrides,
            ),
            mesh_size_factor=size_factor,
            requested_cell_shape=cell_shape,
            resolved_cell_shape=resolved_cell_shape,
            allowed_top_cell_types=auto_policy.allowed_types(order),
            strict_cell_shape=True,
        )
        return self._generate_native_mesh(
            policy=policy,
            size_value=None,
        )

    def _generate_native_mesh(
        self,
        *,
        policy: _MeshGenerationPolicy,
        size_value: float | None,
    ) -> GmshMeshRef:
        operation = policy.operation

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

        self._structured_extrusion_open = False
        self._mesh_attempted = True
        try:
            generation_size_mode: _GenerationSizeMode = self._mesh_size_mode
            if size_value is not None:
                generation_size_mode = "uniform"
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
                policy,
                size_mode=generation_size_mode,
            )
            self._gmsh.model.mesh.generate(self.dimension)
            self._gmsh.model.occ.synchronize()
            if policy.strict_cell_shape:
                self._validate_generated_top_cell_shape(policy)
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
                f"geometry model {self.name!r}: mesh generation failed"
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
        generation_token = object()
        self._generation_token = generation_token
        self._state = _State.MESHED
        return GmshMeshRef(
            self.dimension,
            self.name,
            self,
            self._owner_token,
            generation_token,
        )

    def _snapshot_and_set_mesh_options(
        self,
        policy: _MeshGenerationPolicy,
        *,
        size_mode: _GenerationSizeMode,
    ) -> None:
        if self._pending_options:
            raise GeometryStateError(
                f"geometry model {self.name!r}: mesh options already have a "
                "pending restoration"
            )
        requested: dict[str, float] = {}
        for option_name, option_value in policy.option_overrides:
            if option_name in requested:
                raise GeometryStateError(
                    f"geometry model {self.name!r}: generation policy contains "
                    f"duplicate Gmsh option {option_name!r}"
                )
            requested[option_name] = float(option_value)

        if size_mode == "uniform":
            requested[_POINT_SIZE_OPTION_NAME] = 1.0
        elif size_mode in {"point", "background"}:
            requested.update(
                {
                    _POINT_SIZE_OPTION_NAME: 1.0 if size_mode == "point" else 0.0,
                    "Mesh.MeshSizeFromCurvature": 0.0,
                    "Mesh.MeshSizeExtendFromBoundary": (
                        1.0 if size_mode == "point" else 0.0
                    ),
                    "Mesh.MeshSizeMin": 0.0,
                    "Mesh.MeshSizeMax": 1.0e22,
                }
            )
        if policy.mesh_size_factor is not None:
            requested["Mesh.MeshSizeFactor"] = policy.mesh_size_factor
        elif size_mode in {"point", "background"}:
            requested["Mesh.MeshSizeFactor"] = 1.0

        option_names = tuple(requested)
        for option_name in option_names:
            self._pending_options[option_name] = float(
                self._gmsh.option.getNumber(option_name)
            )
        for option_name in option_names:
            self._gmsh.option.setNumber(option_name, requested[option_name])

    def _validate_generated_top_cell_shape(
        self,
        policy: _MeshGenerationPolicy,
    ) -> None:
        allowed_types = policy.allowed_top_cell_types
        if allowed_types is None or policy.resolved_cell_shape is None:
            raise GeometryStateError(
                f"geometry model {self.name!r}: strict mesh policy is incomplete"
            )

        raw_blocks = self._gmsh.model.mesh.getElements(self.dimension)
        try:
            raw_types, raw_element_tags, _ = raw_blocks
            element_types = list(raw_types)
            element_tag_blocks = list(raw_element_tags)
        except Exception as exc:
            raise self._mesh_cell_shape_error(
                policy,
                "generated malformed top-dimensional element blocks",
            ) from exc

        if len(element_types) != len(element_tag_blocks):
            raise self._mesh_cell_shape_error(
                policy,
                "generated malformed top-dimensional element blocks "
                f"({len(element_types)} type blocks and "
                f"{len(element_tag_blocks)} element-tag blocks)",
            )

        counts: dict[int, int] = {}
        for block_index, (raw_type, raw_tags) in enumerate(
            zip(element_types, element_tag_blocks, strict=True)
        ):
            if isinstance(raw_type, bool):
                raise self._mesh_cell_shape_error(
                    policy,
                    f"generated a non-integer element type in block {block_index}",
                )
            try:
                element_type = int(operator.index(raw_type))
            except (TypeError, ValueError, OverflowError) as exc:
                raise self._mesh_cell_shape_error(
                    policy,
                    f"generated a non-integer element type in block {block_index}",
                ) from exc
            try:
                element_count = len(raw_tags)
            except Exception as exc:
                raise self._mesh_cell_shape_error(
                    policy,
                    "generated malformed element tags in top-dimensional "
                    f"block {block_index}",
                ) from exc
            if element_count:
                counts[element_type] = counts.get(element_type, 0) + element_count

        if not counts:
            raise self._mesh_cell_shape_error(
                policy,
                "generated no top-dimensional cells",
            )
        if set(counts).issubset(allowed_types):
            return

        actual = " and ".join(
            f"{self._element_type_diagnostic_name(element_type)}={counts[element_type]}"
            for element_type in sorted(counts)
        )
        raise self._mesh_cell_shape_error(
            policy,
            f"generated {actual}",
        )

    def _mesh_cell_shape_error(
        self,
        policy: _MeshGenerationPolicy,
        actual_detail: str,
    ) -> MeshCellShapeError:
        allowed_types = policy.allowed_top_cell_types or frozenset()
        expected_names = [
            _GMSH_TOP_CELL_TYPE_NAMES.get(
                element_type,
                f"Gmsh type {element_type}",
            )
            for element_type in sorted(allowed_types)
        ]
        expected = " or ".join(expected_names) if expected_names else "no cell type"
        requested = f"cell_shape={policy.requested_cell_shape!r}"
        if policy.requested_cell_shape is None:
            requested += f" (resolved to {policy.resolved_cell_shape!r})"
        return MeshCellShapeError(
            f"geometry model {self.name!r}: AutoMeshSpec requested "
            f"{requested} for dimension={self.dimension} and order={policy.order}; "
            f"expected only {expected}, but {actual_detail}; automatic fallback "
            "is disabled"
        )

    def _element_type_diagnostic_name(self, element_type: int) -> str:
        try:
            properties = self._gmsh.model.mesh.getElementProperties(element_type)
            name = properties[0]
            if not isinstance(name, str) or not name:
                raise ValueError("element name is unavailable")
        except BaseException:
            return f"Gmsh type {element_type}"
        return name

    @property
    def raw_model(self) -> Any:
        """Return the raw model and transfer size-precedence ownership to caller.

        Access invalidates typed entity and mesh-field references. Any already
        committed typed size mode or entity-control guard remains committed.
        """
        return self._raw_handle("raw_model", "model")

    @property
    def raw_occ(self) -> Any:
        """Return raw OCC and transfer size-precedence ownership to caller.

        Access invalidates typed entity and mesh-field references. Any already
        committed typed size mode or entity-control guard remains committed.
        """
        return self._raw_handle("raw_occ", "occ")

    def _raw_handle(self, operation: str, kind: Literal["model", "occ"]) -> Any:
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        self._activate(operation)
        self._auto_mesh_scope_unknown = True
        if self._entity_control_dependencies:
            self._control_dependency_scope_unknown = True
        self._entity_tokens.clear()
        self._curve_loop_tokens.clear()
        self._curve_loop_dependencies.clear()
        self._wire_tokens.clear()
        self._wire_dependencies.clear()
        self._mesh_field_tokens.clear()
        self._mesh_field_types.clear()
        if kind == "model":
            return self._gmsh.model
        return self._gmsh.model.occ

    def _point_curve(
        self,
        operation: Literal["spline", "bspline"],
        points: Sequence[EntityRef],
        *,
        backend_name: Literal["addSpline", "addBSpline"],
    ) -> EntityRef:
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        try:
            materialized = tuple(points)
        except TypeError as exc:
            raise TypeError(f"{operation} points must be iterable") from exc
        if len(materialized) < 2:
            raise ValueError(f"{operation} requires at least two points")
        normalized = tuple(
            self._normalize_entities(
                (point,),
                operation=f"{operation} point {index}",
            )[0]
            for index, point in enumerate(materialized)
        )
        duplicate_positions: dict[EntityRef, list[int]] = {}
        for index, point in enumerate(normalized):
            duplicate_positions.setdefault(point, []).append(index)
        repeated = {
            point: positions
            for point, positions in duplicate_positions.items()
            if len(positions) > 1
        }
        periodic_repeat = (
            len(normalized) >= 3
            and len(repeated) == 1
            and repeated.get(normalized[0]) == [0, len(normalized) - 1]
        )
        if repeated and not periodic_repeat:
            raise ValueError(
                f"{operation} point inputs must be duplicate-free except for "
                "a repeated first point that closes a periodic curve"
            )
        if any(point.dimension != 0 for point in normalized):
            raise ValueError(
                f"{operation} inputs must be dimension-zero point references"
            )
        self._activate(operation)
        self._assert_occ_liveness(normalized, operation)
        self._assert_planar_entities(normalized, operation)
        coordinates = self._point_coordinates(normalized, operation)
        if any(
            _coordinate_distance(first, second) <= _PLANAR_TOLERANCE
            for first, second in zip(coordinates, coordinates[1:])
        ):
            raise ValueError(
                f"{operation} consecutive points must have distinct coordinates"
            )
        backend = getattr(self._gmsh.model.occ, backend_name)
        tag = backend([point.tag for point in normalized])
        return self._wrap_entity((1, tag))

    def _oriented_curve_endpoints(
        self,
        curves: tuple[OrientedCurveRef, ...],
        *,
        operation: str,
    ) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
        endpoints: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for item in curves:
            signed_tag = -item.curve.tag if item.reversed else item.curve.tag
            try:
                raw_boundary = self._gmsh.model.getBoundary(
                    [(1, signed_tag)],
                    combined=False,
                    oriented=True,
                    recursive=False,
                )
                boundary = tuple(
                    _normalize_dim_tag(pair) for pair in raw_boundary
                )
            except GeometryError:
                raise
            except Exception as exc:
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} could not "
                    f"determine ordered endpoints for curve {item.curve.tag}"
                ) from exc
            if len(boundary) != 2 or any(
                dimension != 0 for dimension, _ in boundary
            ):
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} could not "
                    f"determine two ordered endpoints for curve {item.curve.tag}"
                )
            endpoints.append((boundary[0], boundary[1]))
        return tuple(endpoints)

    def _normalize_curve_loops(
        self,
        loops: Iterable[CurveLoopRef],
        *,
        operation: str,
    ) -> tuple[CurveLoopRef, ...]:
        try:
            normalized = tuple(loops)
        except TypeError as exc:
            raise TypeError(f"{operation} curve loops must be iterable") from exc
        if not normalized:
            raise ValueError(f"{operation} requires at least one curve loop")
        seen_tags: set[int] = set()
        member_keys: set[tuple[int, int]] = set()
        for loop in normalized:
            if not isinstance(loop, CurveLoopRef):
                raise TypeError(
                    f"{operation} requires CurveLoopRef values, got {loop!r}"
                )
            if loop._owner_token is not self._owner_token:
                raise EntityOwnershipError(
                    f"geometry model {self.name!r}: {operation} received a "
                    "curve loop owned by another geometry model"
                )
            if self._curve_loop_tokens.get(loop.tag) is not loop._loop_token:
                raise StaleEntityError(
                    f"geometry model {self.name!r}: {operation} received stale "
                    f"curve loop {loop.tag}"
                )
            if loop.tag in seen_tags:
                raise ValueError(f"{operation} curve loops must be duplicate-free")
            seen_tags.add(loop.tag)
            dependency_keys = self._curve_loop_dependencies.get(loop.tag)
            expected_keys = frozenset(
                self._entity_boundary_closure_keys(
                    tuple(item.curve for item in loop.curves),
                    synchronize=False,
                )
            )
            if dependency_keys != expected_keys:
                raise StaleEntityError(
                    f"geometry model {self.name!r}: {operation} received stale "
                    f"curve loop {loop.tag}"
                )
            overlap = member_keys & set(dependency_keys)
            if overlap:
                raise ValueError(
                    f"{operation} curve loops must not share member curves or "
                    "boundary points"
                )
            member_keys.update(dependency_keys)
            self._normalize_entities(
                tuple(item.curve for item in loop.curves),
                operation=f"{operation} curve loop {loop.tag}",
            )
        return normalized

    def _normalize_wires(
        self,
        wires: Iterable[WireRef],
        *,
        operation: str,
    ) -> tuple[WireRef, ...]:
        try:
            normalized = tuple(wires)
        except TypeError as exc:
            raise TypeError(f"{operation} wires must be iterable") from exc
        if not normalized:
            raise ValueError(f"{operation} requires at least one wire")
        seen_tags: set[int] = set()
        for wire in normalized:
            if not isinstance(wire, WireRef):
                raise TypeError(
                    f"{operation} requires WireRef values, got {wire!r}"
                )
            if wire._owner_token is not self._owner_token:
                raise EntityOwnershipError(
                    f"geometry model {self.name!r}: {operation} received a "
                    "wire owned by another geometry model"
                )
            if self._wire_tokens.get(wire.tag) is not wire._wire_token:
                raise StaleEntityError(
                    f"geometry model {self.name!r}: {operation} received stale "
                    f"wire {wire.tag}"
                )
            if wire.tag in seen_tags:
                raise ValueError(f"{operation} wires must be duplicate-free")
            seen_tags.add(wire.tag)
            member_curves = tuple(item.curve for item in wire.curves)
            dependency_keys = self._wire_dependencies.get(wire.tag)
            expected_keys = frozenset(
                self._entity_boundary_closure_keys(
                    member_curves,
                    synchronize=False,
                )
            )
            if dependency_keys != expected_keys:
                raise StaleEntityError(
                    f"geometry model {self.name!r}: {operation} received stale "
                    f"wire {wire.tag}"
                )
            self._normalize_entities(
                member_curves,
                operation=f"{operation} wire {wire.tag}",
            )
        return normalized

    def _assert_plane_surface_loop_compatibility(
        self,
        loops: tuple[CurveLoopRef, ...],
        *,
        operation: str,
    ) -> None:
        self._assert_curve_loop_boundaries_separated(loops, operation=operation)
        initial_samples = tuple(
            self._sample_curve_loop(loop, divisions=8, operation=operation)
            for loop in loops
        )
        frame = _plane_frame(
            initial_samples[0],
            fixed_xy=self.dimension == 2,
            operation=operation,
        )
        for loop, points in zip(loops, initial_samples, strict=True):
            _project_plane_points(points, frame, operation=operation)
            self._assert_sampled_curve_loop_is_simple(
                loop,
                frame,
                operation=operation,
            )
        if len(loops) == 1:
            return

        probes = tuple(points[0] for points in initial_samples[1:])
        outer = loops[0]
        for probe in probes:
            winding = self._curve_loop_winding_about_point(
                outer,
                probe,
                frame,
                operation=operation,
            )
            if abs(winding) != 1:
                raise ValueError(
                    f"{operation} hole loops must lie strictly inside the outer loop"
                )

        holes = loops[1:]
        for left_index, left_hole in enumerate(holes):
            for right_index in range(left_index + 1, len(holes)):
                right_hole = holes[right_index]
                right_inside_left = self._curve_loop_winding_about_point(
                    left_hole,
                    probes[right_index],
                    frame,
                    operation=operation,
                )
                left_inside_right = self._curve_loop_winding_about_point(
                    right_hole,
                    probes[left_index],
                    frame,
                    operation=operation,
                )
                if right_inside_left != 0 or left_inside_right != 0:
                    raise ValueError(
                        f"{operation} hole loops must have disjoint enclosed regions"
                    )

    def _assert_sampled_curve_loop_is_simple(
        self,
        loop: CurveLoopRef,
        frame: _PlaneFrame,
        *,
        operation: str,
    ) -> None:
        contacts: list[bool] = []
        for divisions in (8, 16, 32, 64):
            sampled = self._sample_curve_loop(
                loop,
                divisions=divisions,
                operation=operation,
            )
            projected = _project_plane_points(sampled, frame, operation=operation)
            contacts.append(_polyline_has_self_contact(projected, frame[-1]))
        if contacts[-2:] == [True, True]:
            raise ValueError(f"{operation} curve loops must not self-intersect")

    def _assert_curve_loop_boundaries_separated(
        self,
        loops: tuple[CurveLoopRef, ...],
        *,
        operation: str,
    ) -> None:
        for loop in loops:
            curves = tuple(item.curve for item in loop.curves)
            for left_index, left_curve in enumerate(curves):
                for right_index in range(left_index + 1, len(curves)):
                    if right_index == left_index + 1 or (
                        left_index == 0 and right_index == len(curves) - 1
                    ):
                        continue
                    distance = self._occ_curve_distance(
                        left_curve,
                        curves[right_index],
                        operation=operation,
                    )
                    if distance <= _PLANAR_TOLERANCE:
                        raise ValueError(
                            f"{operation} curve loops must not self-intersect"
                        )

        for left_index, left_loop in enumerate(loops):
            for right_loop in loops[left_index + 1 :]:
                for left_item in left_loop.curves:
                    for right_item in right_loop.curves:
                        distance = self._occ_curve_distance(
                            left_item.curve,
                            right_item.curve,
                            operation=operation,
                        )
                        if distance <= _PLANAR_TOLERANCE:
                            raise ValueError(
                                f"{operation} curve-loop boundaries must not "
                                "touch or intersect"
                            )

    def _occ_curve_distance(
        self,
        left: EntityRef,
        right: EntityRef,
        *,
        operation: str,
    ) -> float:
        try:
            raw_result = tuple(
                self._gmsh.model.occ.getDistance(
                    left.dimension,
                    left.tag,
                    right.dimension,
                    right.tag,
                )
            )
            if len(raw_result) != 7:
                raise ValueError("expected seven distance result values")
            result = tuple(
                _finite_float(value, "curve-distance result")
                for value in raw_result
            )
        except Exception as exc:
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} could not verify "
                "curve-loop boundary separation"
            ) from exc
        if result[0] < 0.0:
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} received a failed "
                "curve-distance result"
            )
        return result[0]

    def _sample_curve_loop(
        self,
        loop: CurveLoopRef,
        *,
        divisions: int,
        operation: str,
    ) -> tuple[_Point3D, ...]:
        points: list[_Point3D] = []
        for item in loop.curves:
            curve_points = self._sample_oriented_curve(
                item,
                divisions=divisions,
                operation=operation,
            )
            if points:
                scale = max(
                    1.0,
                    _vector_norm(points[-1]),
                    _vector_norm(curve_points[0]),
                )
                if _coordinate_distance(points[-1], curve_points[0]) > (
                    _PLANAR_TOLERANCE * scale
                ):
                    raise GeometryError(
                        f"geometry model {self.name!r}: {operation} sampled a "
                        "discontinuous curve loop"
                    )
                points.extend(curve_points[1:])
            else:
                points.extend(curve_points)
        scale = max(1.0, *(_vector_norm(point) for point in points))
        if _coordinate_distance(points[-1], points[0]) > (
            _PLANAR_TOLERANCE * scale
        ):
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} sampled an open curve loop"
            )
        points[-1] = points[0]
        return tuple(points)

    def _sample_oriented_curve(
        self,
        oriented: OrientedCurveRef,
        *,
        divisions: int,
        operation: str,
    ) -> tuple[_Point3D, ...]:
        try:
            raw_minimum, raw_maximum = self._gmsh.model.getParametrizationBounds(
                1,
                oriented.curve.tag,
            )
            minimum_values = tuple(raw_minimum)
            maximum_values = tuple(raw_maximum)
            if len(minimum_values) != 1 or len(maximum_values) != 1:
                raise ValueError("expected one curve parameter bound")
            minimum = _finite_float(minimum_values[0], "curve parameter minimum")
            maximum = _finite_float(maximum_values[0], "curve parameter maximum")
            if maximum <= minimum:
                raise ValueError("curve parameter bounds must be increasing")
            parameters = [
                minimum + (maximum - minimum) * index / divisions
                for index in range(divisions + 1)
            ]
            raw_coordinates = tuple(
                self._gmsh.model.getValue(1, oriented.curve.tag, parameters)
            )
            if len(raw_coordinates) != 3 * len(parameters):
                raise ValueError("curve evaluation returned an invalid coordinate count")
            coordinates = tuple(
                _finite_float(value, "curve evaluation coordinate")
                for value in raw_coordinates
            )
        except Exception as exc:
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} could not evaluate "
                f"curve {oriented.curve.tag} for loop validation"
            ) from exc

        points = tuple(
            (coordinates[index], coordinates[index + 1], coordinates[index + 2])
            for index in range(0, len(coordinates), 3)
        )
        if oriented.reversed:
            return tuple(reversed(points))
        return points

    def _curve_loop_winding_about_point(
        self,
        loop: CurveLoopRef,
        point: _Point3D,
        frame: _PlaneFrame,
        *,
        operation: str,
    ) -> int:
        projected_point = _project_plane_point(point, frame, operation=operation)
        previous_winding: int | None = None
        for divisions in _LOOP_WINDING_REFINEMENTS:
            sampled = self._sample_curve_loop(
                loop,
                divisions=divisions,
                operation=operation,
            )
            projected = _project_plane_points(sampled, frame, operation=operation)
            winding = _polyline_winding(projected, projected_point, frame[-1])
            if winding is None:
                previous_winding = None
                continue
            if winding == previous_winding:
                return winding
            previous_winding = winding
        raise GeometryError(
            f"geometry model {self.name!r}: {operation} could not determine "
            "curve-loop containment reliably"
        )

    def _assert_planar_entities(
        self,
        entities: Iterable[EntityRef],
        operation: str,
        *,
        bounding_box_padding_scale: float = 1.0,
    ) -> None:
        if self.dimension != 2:
            return
        z_tolerance = (
            _OCC_BOUNDING_BOX_PADDING * bounding_box_padding_scale
            + _PLANAR_TOLERANCE
        )
        for entity in entities:
            bounds = tuple(
                _finite_float(value, "bounding box coordinate")
                for value in self._gmsh.model.occ.getBoundingBox(
                    entity.dimension,
                    entity.tag,
                )
            )
            if len(bounds) != 6:
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} received an "
                    "entity with an invalid bounding box"
                )
            if (
                bounds[2] < -z_tolerance
                or bounds[5] > z_tolerance
            ):
                raise ValueError(
                    f"{operation} in a 2D facade must lie in the global XY plane"
                )

    def _point_coordinates(
        self,
        points: Iterable[EntityRef],
        operation: str,
    ) -> tuple[tuple[float, float, float], ...]:
        result: list[tuple[float, float, float]] = []
        for point in points:
            bounds = tuple(
                _finite_float(value, "point coordinate")
                for value in self._gmsh.model.occ.getBoundingBox(0, point.tag)
            )
            if len(bounds) != 6:
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} received a "
                    "point with invalid coordinates"
                )
            values = tuple(
                0.5 * (bounds[axis] + bounds[axis + 3]) for axis in range(3)
            )
            result.append(values)  # type: ignore[arg-type]
        return tuple(result)

    def _entity_geometry_signature(
        self,
        entity: EntityRef,
        operation: str,
    ) -> tuple[
        tuple[float, float, float, float, float, float],
        tuple[float, float, float],
        float,
    ]:
        try:
            bounds = tuple(
                _finite_float(value, "bounding box coordinate")
                for value in self._gmsh.model.occ.getBoundingBox(
                    entity.dimension,
                    entity.tag,
                )
            )
        except GeometryError:
            raise
        except Exception as exc:
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} could not read a "
                f"finite bounding box for entity ({entity.dimension}, {entity.tag})"
            ) from exc
        if len(bounds) != 6 or any(
            bounds[axis] > bounds[axis + 3] for axis in range(3)
        ):
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} received an invalid "
                f"bounding box for entity ({entity.dimension}, {entity.tag})"
            )

        if entity.dimension == 0:
            center = tuple(
                0.5 * (bounds[axis] + bounds[axis + 3]) for axis in range(3)
            )
            measure = 0.0
        else:
            try:
                center = tuple(
                    _finite_float(value, "center-of-mass coordinate")
                    for value in self._gmsh.model.occ.getCenterOfMass(
                        entity.dimension,
                        entity.tag,
                    )
                )
                measure = _nonnegative_float(
                    self._gmsh.model.occ.getMass(entity.dimension, entity.tag),
                    "entity measure",
                )
            except GeometryError:
                raise
            except Exception as exc:
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} could not read "
                    f"finite geometric invariants for entity "
                    f"({entity.dimension}, {entity.tag})"
                ) from exc
            if len(center) != 3:
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} received an "
                    f"invalid center of mass for entity "
                    f"({entity.dimension}, {entity.tag})"
                )
        return bounds, center, measure  # type: ignore[return-value]

    def _immediate_boundary_keys(
        self,
        entity: EntityRef,
        operation: str,
        *,
        combined: bool = False,
    ) -> frozenset[tuple[int, int]]:
        if entity.dimension == 0:
            return frozenset()
        try:
            pairs = tuple(
                _normalize_dim_tag(pair)
                for pair in self._gmsh.model.getBoundary(
                    [(entity.dimension, entity.tag)],
                    combined=combined,
                    oriented=False,
                    recursive=False,
                )
            )
        except GeometryError:
            raise
        except Exception as exc:
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} received invalid "
                f"boundary topology for entity ({entity.dimension}, {entity.tag})"
            ) from exc
        if any(dimension != entity.dimension - 1 for dimension, _ in pairs):
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} received unexpected "
                f"boundary topology for entity ({entity.dimension}, {entity.tag})"
            )
        return frozenset(pairs)

    def _classify_extrusion_result(
        self,
        *,
        operation: str,
        inputs: tuple[EntityRef, ...],
        outputs: tuple[EntityRef, ...],
        vector: tuple[float, float, float],
        source_signatures: tuple[
            tuple[
                tuple[float, float, float, float, float, float],
                tuple[float, float, float],
                float,
            ],
            ...,
        ],
        source_boundaries: tuple[frozenset[tuple[int, int]], ...],
    ) -> FeatureResult:
        input_dimension = inputs[0].dimension
        primary_dimension = input_dimension + 1
        unique_outputs = _unique_first_seen(outputs)
        input_keys = {(entity.dimension, entity.tag) for entity in inputs}
        if any(
            (entity.dimension, entity.tag) in input_keys
            for entity in unique_outputs
        ):
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} reported a source "
                "entity as generated topology"
            )
        primary = tuple(
            entity
            for entity in unique_outputs
            if entity.dimension == primary_dimension
        )
        same_dimension = tuple(
            entity
            for entity in unique_outputs
            if entity.dimension == input_dimension
        )
        if not primary:
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} could not classify "
                "a primary generated entity"
            )

        self._gmsh.model.occ.synchronize()
        primary_boundaries = {
            (entity.dimension, entity.tag): self._immediate_boundary_keys(
                entity,
                operation,
            )
            for entity in primary
        }
        primary_boundary_union = frozenset().union(*primary_boundaries.values())
        same_keys = {(entity.dimension, entity.tag) for entity in same_dimension}
        generated_boundary_keys = primary_boundary_union - input_keys
        if generated_boundary_keys != same_keys:
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} could not classify "
                "same-dimensional output topology completely"
            )

        candidate_signatures = {
            (entity.dimension, entity.tag): self._entity_geometry_signature(
                entity,
                operation,
            )
            for entity in same_dimension
        }
        candidate_boundaries = {
            (entity.dimension, entity.tag): self._immediate_boundary_keys(
                entity,
                operation,
            )
            for entity in same_dimension
        }
        end_keys: set[tuple[int, int]] = set()
        assigned_primary_keys: list[tuple[int, int]] = []
        for source, source_signature, source_boundary in zip(
            inputs,
            source_signatures,
            source_boundaries,
            strict=True,
        ):
            source_key = (source.dimension, source.tag)
            matches = [
                key
                for key, candidate_signature in candidate_signatures.items()
                if (
                    not (candidate_boundaries[key] & source_boundary)
                    and any(
                        source_key in boundary_keys and key in boundary_keys
                        for boundary_keys in primary_boundaries.values()
                    )
                    and _matches_translated_signature(
                        source_signature,
                        candidate_signature,
                        vector,
                    )
                )
            ]
            if len(matches) != 1:
                detail = "ambiguous" if len(matches) > 1 else "incomplete"
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} terminal-entity "
                    f"classification is {detail}"
                )
            end_key = matches[0]
            containing_primaries = [
                primary_key
                for primary_key, boundary_keys in primary_boundaries.items()
                if source_key in boundary_keys and end_key in boundary_keys
            ]
            if len(containing_primaries) != 1:
                detail = (
                    "ambiguous" if len(containing_primaries) > 1 else "incomplete"
                )
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} source-to-primary "
                    f"topology classification is {detail}"
                )
            if end_key in end_keys:
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} terminal-entity "
                    "classification is ambiguous"
                )
            end_keys.add(end_key)
            assigned_primary_keys.append(containing_primaries[0])

        primary_keys = set(primary_boundaries)
        if (
            len(set(assigned_primary_keys)) != len(assigned_primary_keys)
            or set(assigned_primary_keys) != primary_keys
        ):
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} primary-entity "
                "classification is incomplete or ambiguous"
            )

        side_keys = same_keys - end_keys
        if input_dimension == 0 and side_keys:
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} returned unexpected "
                "point-extrusion side entities"
            )
        source_boundary_union = frozenset().union(*source_boundaries)
        side_entities = {
            (entity.dimension, entity.tag): entity
            for entity in same_dimension
            if (entity.dimension, entity.tag) in side_keys
        }
        assigned_source_boundary_keys: list[tuple[int, int]] = []
        for side_key, side in side_entities.items():
            if side_key not in primary_boundary_union:
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} side topology "
                    "classification is incomplete"
                )
            source_contacts = (
                self._immediate_boundary_keys(side, operation)
                & source_boundary_union
            )
            if len(source_contacts) != 1:
                detail = "ambiguous" if len(source_contacts) > 1 else "incomplete"
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} side topology "
                    f"classification is {detail}"
                )
            assigned_source_boundary_keys.append(next(iter(source_contacts)))
        if (
            len(set(assigned_source_boundary_keys))
            != len(assigned_source_boundary_keys)
            or set(assigned_source_boundary_keys) != source_boundary_union
        ):
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} side topology "
                "classification is incomplete or ambiguous"
            )

        ends = tuple(
            entity
            for entity in unique_outputs
            if (entity.dimension, entity.tag) in end_keys
        )
        sides = tuple(
            entity
            for entity in unique_outputs
            if (entity.dimension, entity.tag) in side_keys
        )
        return FeatureResult(operation, inputs, outputs, primary, ends, sides)

    def _entity_rigid_shape_signature(
        self,
        entity: EntityRef,
        operation: str,
    ) -> _RigidShapeSignature:
        measure = self._entity_geometry_signature(entity, operation)[2]
        boundary_measures: list[float] = []
        for dimension, tag in self._immediate_boundary_keys(entity, operation):
            if dimension == 0:
                boundary_measures.append(0.0)
                continue
            try:
                boundary_measures.append(
                    _nonnegative_float(
                        self._gmsh.model.occ.getMass(dimension, tag),
                        "boundary measure",
                    )
                )
            except GeometryError:
                raise
            except Exception as exc:
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} could not read "
                    f"boundary measure for entity ({dimension}, {tag})"
                ) from exc
        return measure, tuple(sorted(boundary_measures))

    def _entity_key_lies_on_axis(
        self,
        key: tuple[int, int],
        *,
        axis_point: _Point3D,
        axis: _Point3D,
        operation: str,
    ) -> bool:
        try:
            bounds = tuple(
                _finite_float(value, "bounding box coordinate")
                for value in self._gmsh.model.occ.getBoundingBox(*key)
            )
        except GeometryError:
            raise
        except Exception as exc:
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} could not inspect "
                f"axis coincidence for entity {key}"
            ) from exc
        if len(bounds) != 6:
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} received an invalid "
                f"bounding box for entity {key}"
            )
        extent = max(
            bounds[index + 3] - bounds[index] for index in range(3)
        )
        tolerance = (
            2.0 * _OCC_BOUNDING_BOX_PADDING
            + _PLANAR_TOLERANCE * max(1.0, extent)
        )
        return all(
            _point_axis_distance(corner, axis_point, axis) <= tolerance
            for corner in (
                (x_value, y_value, z_value)
                for x_value in (bounds[0], bounds[3])
                for y_value in (bounds[1], bounds[4])
                for z_value in (bounds[2], bounds[5])
            )
        )

    def _classify_revolution_result(
        self,
        *,
        inputs: tuple[EntityRef, ...],
        outputs: tuple[EntityRef, ...],
        axis_point: _Point3D,
        axis: _Point3D,
        angle: float,
        source_signatures: tuple[_GeometrySignature, ...],
        source_boundaries: tuple[frozenset[tuple[int, int]], ...],
    ) -> FeatureResult:
        operation = "revolve"
        input_dimension = inputs[0].dimension
        primary_dimension = input_dimension + 1
        unique_outputs = _unique_first_seen(outputs)
        input_keys = {(entity.dimension, entity.tag) for entity in inputs}
        echoed_inputs = tuple(
            entity
            for entity in unique_outputs
            if (entity.dimension, entity.tag) in input_keys
        )
        full_turn = math.isclose(
            abs(angle),
            2.0 * math.pi,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        if full_turn:
            if {
                (entity.dimension, entity.tag) for entity in echoed_inputs
            } != input_keys:
                raise GeometryError(
                    f"geometry model {self.name!r}: full revolve did not "
                    "report each preserved source seam exactly once"
                )
        elif echoed_inputs:
            raise GeometryError(
                f"geometry model {self.name!r}: partial revolve reported a "
                "source entity as generated topology"
            )

        generated_outputs = tuple(
            entity
            for entity in unique_outputs
            if (entity.dimension, entity.tag) not in input_keys
        )
        primary = tuple(
            entity
            for entity in generated_outputs
            if entity.dimension == primary_dimension
        )
        same_dimension = tuple(
            entity
            for entity in generated_outputs
            if entity.dimension == input_dimension
        )
        if not primary:
            raise GeometryError(
                f"geometry model {self.name!r}: revolve could not classify a "
                "primary generated entity"
            )

        self._gmsh.model.occ.synchronize()
        existing_pairs = {
            _normalize_dim_tag(pair) for pair in self._gmsh.model.occ.getEntities()
        }
        if any(
            (entity.dimension, entity.tag) not in existing_pairs
            for entity in unique_outputs
        ):
            raise GeometryError(
                f"geometry model {self.name!r}: revolve returned a missing entity"
            )
        primary_boundaries = {
            (entity.dimension, entity.tag): self._immediate_boundary_keys(
                entity,
                operation,
            )
            for entity in primary
        }
        primary_boundary_union = frozenset().union(*primary_boundaries.values())
        same_keys = {(entity.dimension, entity.tag) for entity in same_dimension}
        if primary_boundary_union - input_keys != same_keys:
            raise GeometryError(
                f"geometry model {self.name!r}: revolve could not classify "
                "same-dimensional output topology completely"
            )

        if full_turn:
            uncancelled_source_seams = set().union(
                *(
                    self._immediate_boundary_keys(
                        entity,
                        operation,
                        combined=True,
                    )
                    & input_keys
                    for entity in primary
                )
            )
            if uncancelled_source_seams:
                raise GeometryError(
                    f"geometry model {self.name!r}: full revolve exposed a "
                    "coincident source seam as a terminal boundary"
                )
            if len(primary) != len(inputs):
                raise GeometryError(
                    f"geometry model {self.name!r}: full revolve primary "
                    "classification is incomplete or ambiguous"
                )
            sides = tuple(
                entity
                for entity in unique_outputs
                if (entity.dimension, entity.tag) in same_keys
            )
            return FeatureResult(
                operation,
                inputs,
                outputs,
                primary,
                (),
                sides,
            )

        candidate_signatures = {
            (entity.dimension, entity.tag): self._entity_geometry_signature(
                entity,
                operation,
            )
            for entity in same_dimension
        }
        candidate_boundaries = {
            (entity.dimension, entity.tag): self._immediate_boundary_keys(
                entity,
                operation,
            )
            for entity in same_dimension
        }
        end_keys: set[tuple[int, int]] = set()
        assigned_primary_keys: list[tuple[int, int]] = []
        moving_boundaries: list[frozenset[tuple[int, int]]] = []
        for source, source_signature, source_boundary in zip(
            inputs,
            source_signatures,
            source_boundaries,
            strict=True,
        ):
            source_key = (source.dimension, source.tag)
            moving_boundary = frozenset(
                key
                for key in source_boundary
                if not self._entity_key_lies_on_axis(
                    key,
                    axis_point=axis_point,
                    axis=axis,
                    operation=operation,
                )
            )
            moving_boundaries.append(moving_boundary)
            matches = [
                key
                for key, candidate_signature in candidate_signatures.items()
                if (
                    not (candidate_boundaries[key] & moving_boundary)
                    and any(
                        source_key in boundary_keys and key in boundary_keys
                        for boundary_keys in primary_boundaries.values()
                    )
                    and _matches_rotated_signature(
                        source_signature,
                        candidate_signature,
                        axis_point,
                        axis,
                        angle,
                    )
                )
            ]
            if len(matches) != 1:
                detail = "ambiguous" if len(matches) > 1 else "incomplete"
                raise GeometryError(
                    f"geometry model {self.name!r}: revolve terminal-entity "
                    f"classification is {detail}"
                )
            end_key = matches[0]
            containing_primaries = [
                primary_key
                for primary_key, boundary_keys in primary_boundaries.items()
                if source_key in boundary_keys and end_key in boundary_keys
            ]
            if len(containing_primaries) != 1 or end_key in end_keys:
                raise GeometryError(
                    f"geometry model {self.name!r}: revolve source-to-primary "
                    "classification is incomplete or ambiguous"
                )
            end_keys.add(end_key)
            assigned_primary_keys.append(containing_primaries[0])
        if (
            len(set(assigned_primary_keys)) != len(assigned_primary_keys)
            or set(assigned_primary_keys) != set(primary_boundaries)
        ):
            raise GeometryError(
                f"geometry model {self.name!r}: revolve primary-entity "
                "classification is incomplete or ambiguous"
            )

        side_keys = same_keys - end_keys
        moving_boundary_union = frozenset().union(*moving_boundaries)
        assigned_source_boundary_keys: list[tuple[int, int]] = []
        for side in same_dimension:
            side_key = (side.dimension, side.tag)
            if side_key not in side_keys:
                continue
            source_contacts = (
                self._immediate_boundary_keys(side, operation)
                & moving_boundary_union
            )
            if len(source_contacts) != 1:
                detail = "ambiguous" if len(source_contacts) > 1 else "incomplete"
                raise GeometryError(
                    f"geometry model {self.name!r}: revolve side topology "
                    f"classification is {detail}"
                )
            assigned_source_boundary_keys.append(next(iter(source_contacts)))
        if (
            len(set(assigned_source_boundary_keys))
            != len(assigned_source_boundary_keys)
            or set(assigned_source_boundary_keys) != moving_boundary_union
        ):
            raise GeometryError(
                f"geometry model {self.name!r}: revolve side topology "
                "classification is incomplete or ambiguous"
            )
        ends = tuple(
            entity
            for entity in unique_outputs
            if (entity.dimension, entity.tag) in end_keys
        )
        sides = tuple(
            entity
            for entity in unique_outputs
            if (entity.dimension, entity.tag) in side_keys
        )
        return FeatureResult(operation, inputs, outputs, primary, ends, sides)

    def _classify_sweep_result(
        self,
        *,
        inputs: tuple[EntityRef, ...],
        primary: tuple[EntityRef, ...],
        path: WireRef,
        source_signatures: tuple[_GeometrySignature, ...],
        source_shapes: tuple[_RigidShapeSignature, ...],
    ) -> FeatureResult:
        operation = "sweep"
        unique_primary = _unique_first_seen(primary)
        if not unique_primary or (
            not path.closed and len(unique_primary) != len(inputs)
        ):
            raise GeometryError(
                f"geometry model {self.name!r}: sweep primary-entity "
                "classification is incomplete or ambiguous"
            )
        input_keys = {(entity.dimension, entity.tag) for entity in inputs}
        if any(
            (entity.dimension, entity.tag) in input_keys
            for entity in unique_primary
        ):
            raise GeometryError(
                f"geometry model {self.name!r}: sweep reported a profile as "
                "generated topology"
            )
        self._gmsh.model.occ.synchronize()
        primary_boundaries: dict[
            tuple[int, int], tuple[EntityRef, ...]
        ] = {}
        all_boundaries: list[EntityRef] = []
        for entity in unique_primary:
            raw_boundary = self._gmsh.model.getBoundary(
                [(entity.dimension, entity.tag)],
                combined=False,
                oriented=False,
                recursive=False,
            )
            boundary_pairs = tuple(
                _normalize_dim_tag(pair) for pair in raw_boundary
            )
            if not boundary_pairs or any(
                dimension != inputs[0].dimension
                for dimension, _ in boundary_pairs
            ):
                raise GeometryError(
                    f"geometry model {self.name!r}: sweep returned invalid "
                    "primary boundary topology"
                )
            boundary_refs = tuple(
                self._wrap_entity(pair) for pair in boundary_pairs
            )
            primary_boundaries[(entity.dimension, entity.tag)] = (
                _unique_first_seen(boundary_refs)
            )
            all_boundaries.extend(boundary_refs)
        unique_boundaries = _unique_first_seen(all_boundaries)
        boundary_keys = {
            (entity.dimension, entity.tag) for entity in unique_boundaries
        }
        outputs = (*unique_primary, *unique_boundaries)
        if path.closed:
            return FeatureResult(
                operation,
                inputs,
                outputs,
                unique_primary,
                (),
                unique_boundaries,
            )

        end_keys: set[tuple[int, int]] = set()
        assigned_primary: set[tuple[int, int]] = set()
        for source, source_signature, source_shape in zip(
            inputs,
            source_signatures,
            source_shapes,
            strict=True,
        ):
            start_matches: list[
                tuple[tuple[int, int], tuple[int, int]]
            ] = []
            for primary_key, candidates in primary_boundaries.items():
                for candidate in candidates:
                    candidate_key = (candidate.dimension, candidate.tag)
                    candidate_signature = self._entity_geometry_signature(
                        candidate,
                        operation,
                    )
                    if _matches_translated_signature(
                        source_signature,
                        candidate_signature,
                        (0.0, 0.0, 0.0),
                    ):
                        start_matches.append((primary_key, candidate_key))
            if len(start_matches) != 1:
                detail = "ambiguous" if len(start_matches) > 1 else "incomplete"
                raise GeometryError(
                    f"geometry model {self.name!r}: sweep start-profile "
                    f"classification is {detail}"
                )
            primary_key, start_key = start_matches[0]
            if primary_key in assigned_primary or start_key in end_keys:
                raise GeometryError(
                    f"geometry model {self.name!r}: sweep profile-to-primary "
                    "classification is ambiguous"
                )
            terminal_matches = [
                candidate
                for candidate in primary_boundaries[primary_key]
                if (
                    (candidate.dimension, candidate.tag) != start_key
                    and _matches_rigid_shape_signature(
                        source_shape,
                        self._entity_rigid_shape_signature(candidate, operation),
                    )
                )
            ]
            if len(terminal_matches) > 1:
                distances = self._boundary_adjacency_distances(
                    primary_boundaries[primary_key],
                    start_key=start_key,
                    operation=operation,
                )
                terminal_keys = {
                    (candidate.dimension, candidate.tag)
                    for candidate in terminal_matches
                }
                if terminal_keys <= distances.keys():
                    greatest_distance = max(
                        distances[key] for key in terminal_keys
                    )
                    terminal_matches = [
                        candidate
                        for candidate in terminal_matches
                        if distances[(candidate.dimension, candidate.tag)]
                        == greatest_distance
                    ]
            if len(terminal_matches) != 1:
                detail = "ambiguous" if len(terminal_matches) > 1 else "incomplete"
                raise GeometryError(
                    f"geometry model {self.name!r}: sweep terminal-profile "
                    f"classification is {detail}"
                )
            terminal_key = (
                terminal_matches[0].dimension,
                terminal_matches[0].tag,
            )
            if terminal_key in end_keys:
                raise GeometryError(
                    f"geometry model {self.name!r}: sweep terminal-profile "
                    "classification is ambiguous"
                )
            assigned_primary.add(primary_key)
            end_keys.update((start_key, terminal_key))
        if assigned_primary != set(primary_boundaries):
            raise GeometryError(
                f"geometry model {self.name!r}: sweep primary-entity "
                "classification is incomplete"
            )
        side_keys = boundary_keys - end_keys
        ends = tuple(
            entity
            for entity in unique_boundaries
            if (entity.dimension, entity.tag) in end_keys
        )
        sides = tuple(
            entity
            for entity in unique_boundaries
            if (entity.dimension, entity.tag) in side_keys
        )
        return FeatureResult(
            operation,
            inputs,
            outputs,
            unique_primary,
            ends,
            sides,
        )

    def _boundary_adjacency_distances(
        self,
        entities: tuple[EntityRef, ...],
        *,
        start_key: tuple[int, int],
        operation: str,
    ) -> dict[tuple[int, int], int]:
        boundary_keys = {
            (entity.dimension, entity.tag): self._immediate_boundary_keys(
                entity,
                operation,
            )
            for entity in entities
        }
        if start_key not in boundary_keys:
            raise GeometryError(
                f"geometry model {self.name!r}: {operation} boundary adjacency "
                "is missing its start entity"
            )
        incidence: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for entity_key, lower_keys in boundary_keys.items():
            for lower_key in lower_keys:
                incidence.setdefault(lower_key, set()).add(entity_key)

        distances = {start_key: 0}
        frontier = [start_key]
        while frontier:
            current = frontier.pop(0)
            for lower_key in boundary_keys[current]:
                for neighbor in incidence[lower_key]:
                    if neighbor in distances:
                        continue
                    distances[neighbor] = distances[current] + 1
                    frontier.append(neighbor)
        return distances

    def _apply_volume_edge_treatment(
        self,
        *,
        operation: Literal["fillet", "chamfer"],
        volumes: tuple[EntityRef, ...],
        curves: tuple[EntityRef, ...],
        surfaces: tuple[EntityRef, ...],
        values: tuple[float, ...],
        closures: tuple[set[tuple[int, int]], ...],
        remove_volumes: bool,
    ) -> FeatureResult:
        invalidated_keys = set().union(*closures)
        if remove_volumes:
            self._check_control_dependency_scope_known(operation)
            self._check_controlled_removal_allowed(operation, invalidated_keys)
        existing_before = {
            _normalize_dim_tag(pair) for pair in self._gmsh.model.occ.getEntities()
        }
        native_started = False
        try:
            native_started = True
            if operation == "fillet":
                raw_outputs = self._gmsh.model.occ.fillet(
                    [volume.tag for volume in volumes],
                    [curve.tag for curve in curves],
                    list(values),
                    remove_volumes,
                )
            else:
                raw_outputs = self._gmsh.model.occ.chamfer(
                    [volume.tag for volume in volumes],
                    [curve.tag for curve in curves],
                    [surface.tag for surface in surfaces],
                    list(values),
                    remove_volumes,
                )
            try:
                output_pairs = tuple(
                    _normalize_dim_tag(pair) for pair in raw_outputs
                )
            except Exception as exc:
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} returned "
                    "invalid entity data"
                ) from exc
            if not output_pairs:
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} returned no entities"
                )
            primary_pairs = tuple(
                pair for pair in dict.fromkeys(output_pairs) if pair[0] == 3
            )
            if not primary_pairs:
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} returned no "
                    "dimension-three primary entity"
                )
            self._gmsh.model.occ.synchronize()
            existing_after = {
                _normalize_dim_tag(pair)
                for pair in self._gmsh.model.occ.getEntities()
            }
            if any(pair not in existing_after for pair in output_pairs):
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} returned a "
                    "missing entity"
                )
            if remove_volumes:
                input_keys = {
                    (volume.dimension, volume.tag) for volume in volumes
                }
                unreported_survivors = (
                    input_keys & existing_after
                ) - set(output_pairs)
                if unreported_survivors:
                    raise GeometryError(
                        f"geometry model {self.name!r}: {operation} left an "
                        "input volume alive without reporting it"
                    )
                aliased_outputs = {
                    pair
                    for pair in output_pairs
                    if pair in existing_before and pair not in invalidated_keys
                }
                if aliased_outputs:
                    raise GeometryError(
                        f"geometry model {self.name!r}: {operation} returned an "
                        "unrelated existing entity as new topology"
                    )
                self._invalidate_entity_keys(invalidated_keys)
            else:
                missing_preserved = invalidated_keys - existing_after
                if missing_preserved:
                    raise GeometryError(
                        f"geometry model {self.name!r}: {operation} removed "
                        "topology while remove_volumes was false"
                    )
                if any(pair in existing_before for pair in output_pairs):
                    raise GeometryError(
                        f"geometry model {self.name!r}: {operation} did not "
                        "return fresh preserving-mode topology"
                    )
            outputs = tuple(self._wrap_entity(pair) for pair in output_pairs)
            primary = tuple(
                entity
                for entity in _unique_first_seen(outputs)
                if entity.dimension == 3
            )
            primary_closure = self._entity_boundary_closure_keys(
                primary,
                synchronize=False,
            )
            unrelated_lower_outputs = {
                (entity.dimension, entity.tag)
                for entity in outputs
                if entity.dimension < 3
                and (entity.dimension, entity.tag) not in primary_closure
            }
            if unrelated_lower_outputs:
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} returned lower-"
                    "dimensional topology outside its primary volume closure"
                )
            return FeatureResult(operation, volumes, outputs, primary)
        except BaseException as error:
            if native_started:
                self._fail_closed_after_unknown_occ_mutation()
            if isinstance(error, GeometryError) or not isinstance(error, Exception):
                raise
            raise GeometryError(
                f"geometry model {self.name!r}: native OCC {operation} failed"
            ) from error

    def _fail_closed_after_unknown_occ_mutation(self) -> None:
        self._entity_tokens.clear()
        self._curve_loop_tokens.clear()
        self._curve_loop_dependencies.clear()
        self._wire_tokens.clear()
        self._wire_dependencies.clear()
        self._control_dependency_scope_unknown = True
        self._auto_mesh_scope_unknown = True

    def _boolean(
        self,
        operation: Literal["fuse", "cut", "intersect", "fragment"],
        objects: Iterable[EntityRef],
        tools: Iterable[EntityRef],
        *,
        remove_objects: bool,
        remove_tools: bool,
    ) -> BooleanResult:
        self._check_state(operation, _GEOMETRY_MUTATION_STATES)
        normalized_objects = self._normalize_entities(objects, operation=operation)
        normalized_tools = self._normalize_entities(tools, operation=operation)
        if set(normalized_objects) & set(normalized_tools):
            raise ValueError(f"{operation} inputs must not overlap")
        object_dimensions = {entity.dimension for entity in normalized_objects}
        tool_dimensions = {entity.dimension for entity in normalized_tools}
        dimensions = object_dimensions | tool_dimensions
        if operation in {"fuse", "cut"} and len(dimensions) != 1:
            raise ValueError(f"{operation} inputs must have one common dimension")
        if operation == "intersect" and (
            len(object_dimensions) != 1 or len(tool_dimensions) != 1
        ):
            raise ValueError(
                "intersect objects and tools must each have one common dimension"
            )
        if not isinstance(remove_objects, bool):
            raise TypeError(
                f"remove_objects must be a boolean, got {remove_objects!r}"
            )
        if not isinstance(remove_tools, bool):
            raise TypeError(f"remove_tools must be a boolean, got {remove_tools!r}")
        if remove_objects or remove_tools:
            self._check_control_dependency_scope_known(operation)
        all_inputs = (*normalized_objects, *normalized_tools)
        self._activate(operation)
        self._assert_occ_liveness(all_inputs, operation)
        removed_inputs = (
            *(normalized_objects if remove_objects else ()),
            *(normalized_tools if remove_tools else ()),
        )
        preserved_inputs = (
            *(normalized_objects if not remove_objects else ()),
            *(normalized_tools if not remove_tools else ()),
        )
        removed_closure = self._entity_boundary_closure_keys(removed_inputs)
        preserved_closure = self._entity_boundary_closure_keys(preserved_inputs)
        invalidated_keys = removed_closure - preserved_closure
        self._check_controlled_removal_allowed(operation, invalidated_keys)
        backend_operation = getattr(self._gmsh.model.occ, operation)
        raw_result = backend_operation(
            _dim_tags(normalized_objects),
            _dim_tags(normalized_tools),
            -1,
            remove_objects,
            remove_tools,
        )
        self._invalidate_entity_keys(invalidated_keys)
        try:
            raw_outputs, raw_map = raw_result
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
            if operation == "fragment":
                maximum_input_dimension = max(dimensions)
                seen_outputs = set(output_pairs)
                map_only_outputs: list[tuple[int, int]] = []
                for group in map_pairs:
                    for pair in group:
                        if (
                            pair[0] < maximum_input_dimension
                            and pair not in seen_outputs
                        ):
                            seen_outputs.add(pair)
                            map_only_outputs.append(pair)
                output_pairs = (*output_pairs, *map_only_outputs)

            existing_pairs = {
                _normalize_dim_tag(pair)
                for pair in self._gmsh.model.occ.getEntities()
            }
            represented_pairs = set(output_pairs)
            represented_pairs.update(
                pair for group in map_pairs for pair in group
            )
            if any(
                dimension > self.dimension
                for dimension, _ in represented_pairs
            ):
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} returned an "
                    "entity above the facade dimension"
                )
            missing_pairs = represented_pairs - existing_pairs
            if missing_pairs:
                dimension, tag = min(missing_pairs)
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} returned a "
                    f"missing entity ({dimension}, {tag})"
                )
            missing_preserved = [
                entity
                for entity in preserved_inputs
                if (entity.dimension, entity.tag) not in existing_pairs
            ]
            if missing_preserved:
                self._entity_tokens.clear()
                self._curve_loop_tokens.clear()
                self._curve_loop_dependencies.clear()
                self._wire_tokens.clear()
                self._wire_dependencies.clear()
                raise GeometryError(
                    f"geometry model {self.name!r}: {operation} removed an input "
                    "whose remove flag was false"
                )
        except BaseException as error:
            self._entity_tokens.clear()
            self._curve_loop_tokens.clear()
            self._curve_loop_dependencies.clear()
            self._wire_tokens.clear()
            self._wire_dependencies.clear()
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

    def _normalize_optional_entities(
        self,
        entities: Iterable[EntityRef],
        *,
        operation: str,
        label: str,
    ) -> tuple[EntityRef, ...]:
        try:
            materialized = tuple(entities)
        except TypeError as exc:
            raise TypeError(f"{operation} {label} must be iterable") from exc
        if not materialized:
            return ()
        return self._normalize_entities(
            materialized,
            operation=f"{operation} {label}",
        )

    def _normalize_mesh_fields(
        self,
        fields: Iterable[MeshFieldRef],
        *,
        operation: str,
    ) -> tuple[MeshFieldRef, ...]:
        try:
            normalized = tuple(fields)
        except TypeError as exc:
            raise TypeError(f"{operation} fields must be iterable") from exc
        seen_tags: set[int] = set()
        for mesh_field in normalized:
            if not isinstance(mesh_field, MeshFieldRef):
                raise TypeError(
                    f"{operation} requires MeshFieldRef values, got "
                    f"{mesh_field!r}"
                )
            if mesh_field._owner_token is not self._owner_token:
                raise MeshFieldOwnershipError(
                    f"geometry model {self.name!r}: {operation} received a "
                    "mesh field owned by another geometry model"
                )
            current_token = self._mesh_field_tokens.get(mesh_field.tag)
            registered_type = self._mesh_field_types.get(mesh_field.tag)
            if (
                current_token is not mesh_field._field_token
                or registered_type != mesh_field.field_type
            ):
                raise StaleMeshFieldError(
                    f"geometry model {self.name!r}: {operation} received stale "
                    f"mesh field {mesh_field.tag}"
                )
            if mesh_field.tag in seen_tags:
                raise ValueError(f"{operation} field inputs must be duplicate-free")
            seen_tags.add(mesh_field.tag)
        return normalized

    def _active_mesh_field_tags(self) -> set[int]:
        try:
            return {
                _validate_positive_tag(tag, "mesh field tag")
                for tag in self._gmsh.model.mesh.field.list()
            }
        except (TypeError, ValueError) as exc:
            raise GeometryError(
                f"geometry model {self.name!r}: Gmsh returned an invalid mesh "
                "field list"
            ) from exc

    def _assert_mesh_field_liveness(
        self,
        fields: Iterable[MeshFieldRef],
        operation: str,
    ) -> None:
        active_tags = self._active_mesh_field_tags()
        for mesh_field in fields:
            token = self._mesh_field_tokens.get(mesh_field.tag)
            registered_type = self._mesh_field_types.get(mesh_field.tag)
            if (
                token is not mesh_field._field_token
                or registered_type != mesh_field.field_type
                or mesh_field.tag not in active_tags
            ):
                self._mesh_field_tokens.pop(mesh_field.tag, None)
                self._mesh_field_types.pop(mesh_field.tag, None)
                raise StaleMeshFieldError(
                    f"geometry model {self.name!r}: {operation} mesh field "
                    f"{mesh_field.tag} no longer exists"
                )

    def _construct_mesh_field(
        self,
        field_type: Literal["Distance", "Threshold", "Min"],
        configure: Any,
    ) -> MeshFieldRef:
        manager = self._gmsh.model.mesh.field
        allocated_tag: int | None = None
        try:
            allocated_tag = _validate_positive_tag(
                manager.add(field_type, tag=-1),
                "mesh field tag",
            )
            configure(allocated_tag)
            if allocated_tag not in self._active_mesh_field_tags():
                raise GeometryError(
                    f"geometry model {self.name!r}: newly allocated "
                    f"{field_type} field {allocated_tag} is not active"
                )
            token = object()
            reference = MeshFieldRef(
                allocated_tag,
                field_type,
                self._owner_token,
                token,
            )
            self._mesh_field_tokens[allocated_tag] = token
            self._mesh_field_types[allocated_tag] = field_type
        except BaseException as error:
            if allocated_tag is not None:
                self._mesh_field_tokens.pop(allocated_tag, None)
                self._mesh_field_types.pop(allocated_tag, None)
                try:
                    manager.remove(allocated_tag)
                except BaseException as rollback_error:
                    error.add_note(
                        f"geometry model {self.name!r}: mesh-field rollback "
                        f"also failed while removing field {allocated_tag}: "
                        f"{rollback_error}"
                    )
            raise
        return reference

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

    def _prepare_geometry_query_entity(
        self,
        entity: EntityRef,
        *,
        operation: str,
    ) -> EntityRef:
        self._check_state(operation, _QUERY_STATES)
        target = self._normalize_entities((entity,), operation=operation)[0]
        if target.dimension > self.dimension:
            raise ValueError(
                f"{operation} entity dimension exceeds the geometry model dimension"
            )
        self._activate(operation)
        self._gmsh.model.occ.synchronize()
        self._assert_occ_liveness((target,), operation)
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
                    self._invalidate_entity_keys((key,))
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
        materialized = set(keys)
        for key in materialized:
            self._entity_tokens.pop(key, None)
        self._invalidate_curve_topology_for_keys(materialized)

    def _invalidate_curve_topology_for_keys(
        self,
        keys: Iterable[tuple[int, int]],
    ) -> None:
        materialized = set(keys)
        invalid_loops = [
            tag
            for tag, dependencies in self._curve_loop_dependencies.items()
            if dependencies & materialized
        ]
        for tag in invalid_loops:
            self._curve_loop_tokens.pop(tag, None)
            self._curve_loop_dependencies.pop(tag, None)
        invalid_wires = [
            tag
            for tag, dependencies in self._wire_dependencies.items()
            if dependencies & materialized
        ]
        for tag in invalid_wires:
            self._wire_tokens.pop(tag, None)
            self._wire_dependencies.pop(tag, None)

    def _validate_2d_z(self, z_value: float, operation: str) -> None:
        if self.dimension == 2 and abs(z_value) > _PLANAR_TOLERANCE:
            raise ValueError(
                f"{operation} in a 2D facade must lie in the global XY plane"
            )

    def _check_controlled_removal_allowed(
        self,
        operation: str,
        removed_keys: set[tuple[int, int]],
    ) -> None:
        if not removed_keys:
            return
        if self._control_dependency_scope_unknown:
            raise self._state_error(
                operation,
                "raw access or a failed controlled topology operation made "
                "mesh-control dependencies unknown",
            )
        conflicts = removed_keys & self._entity_control_dependencies
        if conflicts:
            dimension, tag = min(conflicts)
            raise self._state_error(
                operation,
                "destructive topology replacement would invalidate an "
                "entity-dependent mesh control on topology "
                f"({dimension}, {tag})",
            )

    def _check_control_dependency_scope_known(self, operation: str) -> None:
        if self._control_dependency_scope_unknown:
            raise self._state_error(
                operation,
                "raw access or a failed controlled topology operation made "
                "mesh-control dependencies unknown",
            )

    def _check_auto_mesh_controls(self) -> None:
        if self._auto_mesh_scope_unknown:
            raise MeshControlConflictError(
                f"geometry model {self.name!r}: AutoMeshSpec generation cannot own "
                "the mesh topology because raw access or a failed controlled "
                "topology operation made the automatic topology scope unknown; "
                "use Mesher.generate(MeshSpec(...)) for this model"
            )
        if self._auto_mesh_blockers:
            blockers = ", ".join(sorted(self._auto_mesh_blockers))
            raise MeshControlConflictError(
                f"geometry model {self.name!r}: AutoMeshSpec generation conflicts "
                f"with explicit topology controls: {blockers}; use "
                "Mesher.generate(MeshSpec(...)) for this model"
            )

    def _check_controlled_transform_allowed(
        self,
        operation: str,
        entities: tuple[EntityRef, ...],
    ) -> None:
        if self._control_dependency_scope_unknown:
            raise self._state_error(
                operation,
                "raw access or a failed controlled topology operation made "
                "mesh-control dependencies unknown",
            )
        if not self._transform_unsafe_control_dependencies:
            return
        transformed_keys = self._entity_boundary_closure_keys(entities)
        conflicts = transformed_keys & self._transform_unsafe_control_dependencies
        if conflicts:
            dimension, tag = min(conflicts)
            raise self._state_error(
                operation,
                "the OCC transform would discard an entity-dependent mesh "
                f"control on topology ({dimension}, {tag})",
            )

    def _register_control_dependencies(
        self,
        keys: Iterable[tuple[int, int]],
        *,
        transform_unsafe: bool,
    ) -> None:
        materialized = set(keys)
        self._entity_control_dependencies.update(materialized)
        if transform_unsafe:
            self._transform_unsafe_control_dependencies.update(materialized)

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

    def _borrow_generated_mesh(self, source: GmshMeshRef) -> Any:
        """Validate and reactivate one live generated mesh for the IO layer."""
        stale_error = _stale_gmsh_mesh_error(source.model_name)
        if (
            source._owner is not self
            or source._owner_token is not self._owner_token
            or source.dimension != self.dimension
            or source.model_name != self.name
            or self._state is not _State.MESHED
            or self._generation_token is None
            or source._generation_token is not self._generation_token
            or not self._created_model
        ):
            raise stale_error

        gmsh = self._gmsh
        try:
            if gmsh is None or not bool(gmsh.isInitialized()):
                raise stale_error
            model_names = tuple(str(item) for item in gmsh.model.list())
            if self.name not in model_names:
                raise stale_error
            if str(gmsh.model.getCurrent()) != self.name:
                gmsh.model.setCurrent(self.name)
        except StaleGmshMeshError:
            raise
        except BaseException as error:
            raise stale_error from error
        return gmsh.model

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
        self._entity_tokens.clear()
        self._curve_loop_tokens.clear()
        self._curve_loop_dependencies.clear()
        self._wire_tokens.clear()
        self._wire_dependencies.clear()
        self._mesh_field_tokens.clear()
        self._mesh_field_types.clear()
        self._mesh_size_mode = "none"
        self._background_field = None
        self._entity_control_dependencies.clear()
        self._transform_unsafe_control_dependencies.clear()
        self._control_dependency_scope_unknown = False
        self._auto_mesh_blockers.clear()
        self._auto_mesh_scope_unknown = False
        self._mesher_token = None
        self._structured_extrusion_open = False
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


def _validate_auto_mesh_level(value: Any) -> Literal[1, 2, 3, 4, 5]:
    if isinstance(value, bool) or not isinstance(value, int) or value not in range(1, 6):
        raise ValueError(
            "AutoMeshSpec level must be a Python integer from 1 through "
            f"5, got {value!r}"
        )
    return value


def _resolve_auto_mesh_mode(
    dimension: Literal[1, 2, 3],
    cell_shape: Any,
) -> _AutoMeshMode:
    if dimension == 1:
        if cell_shape is not None:
            raise ValueError(
                "AutoMeshSpec cell_shape must be None for dimension 1, "
                f"got {cell_shape!r}"
            )
        return "line"

    if dimension == 2:
        if cell_shape is None:
            return "tri-quad"
        if not isinstance(cell_shape, str) or cell_shape not in {
            "tri",
            "tri-quad",
            "quad",
        }:
            raise ValueError(
                "AutoMeshSpec cell_shape for dimension 2 must be exactly "
                f"'tri', 'tri-quad', or 'quad', got {cell_shape!r}"
            )
        return cell_shape

    if cell_shape is None:
        return "tet"
    if not isinstance(cell_shape, str) or cell_shape not in {"tet", "hex"}:
        raise ValueError(
            "AutoMeshSpec cell_shape for dimension 3 must be exactly "
            f"'tet' or 'hex', got {cell_shape!r}"
        )
    return cell_shape


def _validate_mesh_field_type(
    value: Any,
) -> Literal["Distance", "Threshold", "Min"]:
    if not isinstance(value, str) or value not in _MESH_FIELD_TYPES:
        raise ValueError(
            "mesh field type must be 'Distance', 'Threshold', or 'Min', "
            f"got {value!r}"
        )
    return value


def _normalize_dim_tag(value: Any) -> tuple[int, int]:
    try:
        dimension, tag = value
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"invalid Gmsh entity reference {value!r}") from exc
    return (
        _validate_entity_dimension(dimension),
        _validate_positive_tag(tag, "entity tag"),
    )


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


def _dim_tags(entities: Iterable[EntityRef]) -> list[tuple[int, int]]:
    return [(entity.dimension, entity.tag) for entity in entities]


__all__ = [
    "GeometryModel",
    "model",
]
