"""Headless native geometry-to-canonical-FEM-model workflow."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any, Protocol

from fem import geometry
from fem.core.model import (
    Edge,
    ElementEdge,
    ElementFace,
    ElementSet,
    FEMModel,
    NodeSet,
    Surface,
)
from fem.geometry.measurements import resolve_target_radius
from fem.geometry.recipe_analysis import (
    recipe_characteristic_size,
    supports_structured_hexahedron,
)
from fem.geometry.references import LogicalEntityRef
from fem.geometry.recipes import (
    NATIVE_GEOMETRY_TYPES,
    NativeGeometry,
    geometry_dimension,
)
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing
from fem.mesh.settings import MeshSettings
from fem.selection import edges as mesh_edges
from fem.selection import faces as mesh_faces

from .recipe_compiler import (
    CompiledRecipeTopology,
    TopologyResolutionError,
    compile_recipe,
)
from .native_regions import (
    CompiledDomainRegionSource,
    LogicalReferencesRegionSource,
    NativeRegionDescriptor,
    RecipeRegionSource,
    validate_native_authoring_context,
)
from .revisions import MeshTaskSnapshot


class RecipeTopologyResolver(Protocol):
    """Application seam for compiling and resolving one-session CAD topology."""

    def build(
        self,
        cad: Any,
        recipe: NativeGeometry,
    ) -> CompiledRecipeTopology:
        """Compile one recipe in the active CAD session."""
        ...

    def resolve(
        self,
        cad: Any,
        recipe: NativeGeometry,
        topology: CompiledRecipeTopology,
        reference: LogicalEntityRef,
    ) -> tuple[Any, ...]:
        """Resolve one logical entity to live CAD references."""
        ...

    def characteristic_size(self, recipe: NativeGeometry) -> float:
        """Return a representative dimension for structured meshing."""
        ...

    def supports_hexahedron(self, recipe: NativeGeometry) -> bool:
        """Return whether the recipe supports the structured Hex workflow."""
        ...


class LogicalRecipeTopologyResolver:
    """Default resolver backed by the tag-independent recipe topology catalog."""

    def build(
        self,
        cad: Any,
        recipe: NativeGeometry,
    ) -> CompiledRecipeTopology:
        return compile_recipe(cad, recipe)

    def resolve(
        self,
        cad: Any,
        recipe: NativeGeometry,
        topology: CompiledRecipeTopology,
        reference: LogicalEntityRef,
    ) -> tuple[Any, ...]:
        del cad, recipe
        return topology.resolve(reference)

    def characteristic_size(self, recipe: NativeGeometry) -> float:
        return recipe_characteristic_size(recipe)

    def supports_hexahedron(self, recipe: NativeGeometry) -> bool:
        return supports_structured_hexahedron(recipe)


DEFAULT_TOPOLOGY_RESOLVER: RecipeTopologyResolver = LogicalRecipeTopologyResolver()


def generate_fem_model(
    recipe_or_snapshot: NativeGeometry | MeshTaskSnapshot,
    settings: MeshSettings | None = None,
    *,
    named_regions: Iterable[Any] | Mapping[str, Any] | None = None,
    resolver: RecipeTopologyResolver | None = None,
) -> FEMModel:
    """Generate a canonical model from explicit inputs or a mesh task snapshot."""
    recipe, mesh_settings, regions = _normalize_inputs(
        recipe_or_snapshot,
        settings,
        named_regions,
    )
    region_descriptors = validate_native_authoring_context(
        recipe,
        regions,
        local_controls=mesh_settings.local_controls,
    )
    topology_resolver = resolver or DEFAULT_TOPOLOGY_RESOLVER

    try:
        import gmsh
    except ModuleNotFoundError as error:
        if error.name != "gmsh":
            raise
        raise ModuleNotFoundError(
            "几何与网格功能需要 Gmsh；请安装项目的 cad 可选依赖"
        ) from error

    owns_session = not bool(gmsh.isInitialized())
    try:
        if owns_session:
            # Gmsh can install its SIGINT handler only on the Python main
            # thread. GUI callers intentionally execute this function in a
            # worker, so the application retains signal ownership.
            gmsh.initialize(interruptible=False)
        dimension = geometry_dimension(recipe)
        with geometry.model(recipe.name, dimension=dimension) as cad:
            topology = topology_resolver.build(cad, recipe)
            mesher = gmsh_meshing.Mesher(cad)
            if mesh_settings.cell_shape == "hexahedron":
                _configure_hexahedral_mesh(
                    cad,
                    mesher,
                    recipe,
                    mesh_settings,
                    topology,
                    topology_resolver,
                )

            entity_groups = {
                descriptor.name: _resolve_region_descriptor(
                    cad,
                    recipe,
                    topology,
                    topology_resolver,
                    descriptor,
                )
                for descriptor in region_descriptors
            }

            mesh_size = mesh_settings.size
            refinements: list[Any] = []
            for control in mesh_settings.local_controls:
                entities = topology_resolver.resolve(
                    cad,
                    recipe,
                    topology,
                    control.target,
                )
                scale = (
                    mesh_settings.size
                    if control.falloff.reference == "global_size"
                    else resolve_target_radius(recipe, control.target)
                )
                distance = mesher.distance_field(
                    **_distance_field_sources(entities),
                    sampling=100,
                )
                refinements.append(
                    mesher.threshold_field(
                        distance,
                        size_min=control.size,
                        size_max=mesh_settings.size,
                        dist_min=scale * control.falloff.start_factor,
                        dist_max=scale * control.falloff.end_factor,
                    )
                )
            if refinements:
                background = (
                    refinements[0]
                    if len(refinements) == 1
                    else mesher.min_field(refinements)
                )
                mesher.background_field(background)
                mesh_size = None

            native_mesh = mesher.generate(
                gmsh_meshing.MeshSpec(
                    size=mesh_size,
                    order=mesh_settings.order,
                    recombine=mesh_settings.cell_shape
                    in {"quadrilateral", "hexahedron"},
                )
            )
            mesh = gmsh_io.read(native_mesh)
            return _build_native_fem_model(
                mesh,
                native_mesh,
                recipe.name,
                dimension,
                entity_groups,
            )
    finally:
        if owns_session and bool(gmsh.isInitialized()):
            gmsh.finalize()


def _normalize_inputs(
    recipe_or_snapshot: NativeGeometry | MeshTaskSnapshot,
    settings: MeshSettings | None,
    named_regions: Iterable[Any] | Mapping[str, Any] | None,
) -> tuple[NativeGeometry, MeshSettings, tuple[Any, ...]]:
    if isinstance(recipe_or_snapshot, MeshTaskSnapshot):
        if settings is not None:
            raise TypeError("MeshTaskSnapshot 已包含网格设置，不能重复传入 settings")
        if named_regions is not None:
            raise TypeError(
                "MeshTaskSnapshot 已包含命名区域，不能重复传入 named_regions"
            )
        recipe = recipe_or_snapshot.geometry_recipe
        mesh_settings = recipe_or_snapshot.mesh_settings
        region_source = recipe_or_snapshot.named_regions
    else:
        recipe = recipe_or_snapshot
        mesh_settings = settings
        region_source = () if named_regions is None else named_regions

    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise TypeError("网格生成需要有效的原生几何配方")
    if not isinstance(mesh_settings, MeshSettings):
        raise TypeError("网格生成需要 MeshSettings")
    regions = (
        tuple(region_source.values())
        if isinstance(region_source, Mapping)
        else tuple(region_source)
    )
    return recipe, mesh_settings, regions


def _configure_hexahedral_mesh(
    cad: Any,
    mesher: Any,
    recipe: NativeGeometry,
    settings: MeshSettings,
    topology: CompiledRecipeTopology,
    resolver: RecipeTopologyResolver,
) -> None:
    if not resolver.supports_hexahedron(recipe):
        raise ValueError("六面体结构化网格当前仅支持长方体或矩形草图拉伸体")
    curves = _unique_entities(
        entity
        for entity in cad.boundary(topology.boundary, combined=False)
        if entity.dimension == 1
    )
    node_count = max(
        2,
        int(math.ceil(resolver.characteristic_size(recipe) / settings.size)) + 1,
    )
    for curve in curves:
        mesher.transfinite_curve(curve, num_nodes=node_count)
    for surface in topology.boundary:
        mesher.transfinite_surface(surface)
        mesher.recombine(surface)
    if len(topology.domain) != 1:
        raise TopologyResolutionError("结构化六面体网格要求唯一计算域")
    mesher.transfinite_volume(topology.domain[0])


def _resolve_region_descriptor(
    cad: Any,
    recipe: NativeGeometry,
    topology: CompiledRecipeTopology,
    resolver: RecipeTopologyResolver,
    descriptor: NativeRegionDescriptor,
) -> tuple[Any, ...]:
    source = descriptor.source
    if isinstance(source, CompiledDomainRegionSource):
        return tuple(topology.domain)
    if isinstance(source, RecipeRegionSource):
        entities = tuple(topology.region_bindings.get(source.selector, ()))
        if not entities:
            raise TopologyResolutionError(
                f"内建区域 {descriptor.name!r} 没有对应的 CAD 实体"
            )
        return entities
    if not isinstance(source, LogicalReferencesRegionSource):
        raise TypeError(
            f"不支持的 native region source: {type(source).__name__}"
        )
    entities: list[Any] = []
    for reference in source.references:
        entities.extend(
            resolver.resolve(
                cad,
                recipe,
                topology,
                reference,
            )
        )
    resolved = _unique_entities(entities)
    if not resolved:
        raise TopologyResolutionError(
            f"命名区域 {descriptor.name!r} 没有解析到 CAD 实体"
        )
    return resolved


def _distance_field_sources(
    entities: Iterable[Any],
) -> Mapping[str, tuple[Any, ...]]:
    resolved = _unique_entities(entities)
    if not resolved:
        raise TopologyResolutionError("局部网格控制没有解析到几何实体")
    dimensions = {entity.dimension for entity in resolved}
    if len(dimensions) != 1:
        raise TopologyResolutionError("局部网格控制不能混用不同维度的几何实体")
    dimension = next(iter(dimensions))
    source_name = {0: "points", 1: "curves", 2: "surfaces"}.get(dimension)
    if source_name is None:
        raise TopologyResolutionError("局部网格控制只支持点、边或面")
    return {source_name: resolved}


def _build_native_fem_model(
    mesh: Any,
    native_mesh: Any,
    name: str,
    dimension: int,
    entity_groups: Mapping[str, tuple[Any, ...]],
) -> FEMModel:
    """Convert CAD entity groups to FEM sets without leaking CAD identifiers."""
    model = FEMModel(mesh=mesh, name=name)
    boundary_edges = mesh_edges.boundary(mesh) if dimension == 2 else ()
    boundary_faces = mesh_faces.boundary(mesh) if dimension == 3 else ()

    for group_name, entities in entity_groups.items():
        if not entities:
            continue
        entity_dimension = entities[0].dimension
        if any(entity.dimension != entity_dimension for entity in entities):
            raise ValueError(f"几何区域 {group_name} 包含不同维度的实体")
        if entity_dimension == dimension:
            element_ids = gmsh_io.entity_element_ids(native_mesh, entities)
            model.element_sets[group_name] = ElementSet(
                group_name,
                element_ids,
            )
            continue

        node_ids = gmsh_io.entity_node_ids(native_mesh, entities)
        model.node_sets[group_name] = NodeSet(group_name, node_ids)
        node_id_set = set(node_ids)
        if dimension == 2 and entity_dimension == 1:
            model.edges[group_name] = Edge(
                group_name,
                [
                    ElementEdge(
                        element_id,
                        local_edge,
                        edge_node_ids,
                    )
                    for (
                        element_id,
                        local_edge,
                        edge_node_ids,
                    ) in boundary_edges
                    if set(edge_node_ids).issubset(node_id_set)
                ],
            )
        elif dimension == 3 and entity_dimension == 2:
            model.surfaces[group_name] = Surface(
                group_name,
                [
                    ElementFace(
                        element_id,
                        local_face,
                        face_node_ids,
                    )
                    for (
                        element_id,
                        local_face,
                        face_node_ids,
                    ) in boundary_faces
                    if set(face_node_ids).issubset(node_id_set)
                ],
            )
    return model


def _unique_entities(entities: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(entities))


__all__ = [
    "DEFAULT_TOPOLOGY_RESOLVER",
    "LogicalRecipeTopologyResolver",
    "RecipeTopologyResolver",
    "TopologyResolutionError",
    "generate_fem_model",
]
