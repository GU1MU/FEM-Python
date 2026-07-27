"""Pure native-region catalog and detached geometry-reference validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from fem.geometry.measurements import resolve_target_radius
from fem.geometry.recipe_topology import (
    LogicalEntity,
    RecipeTopology,
    describe_recipe_topology,
)
from fem.geometry.references import (
    EntityKind,
    LogicalEntityRef,
    logical_ref_sort_key,
)
from fem.geometry.recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    NativeGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchGeometry,
)
from fem.mesh.settings import MeshSettings

from .native_mesh_contract import (
    NativeMeshContract,
    describe_native_mesh_contract,
)

NativeRegionProduct = Literal[
    "node_set",
    "element_set",
    "edge",
    "surface",
    "beam_element_set",
]

_PRODUCTS = frozenset(
    {"node_set", "element_set", "edge", "surface", "beam_element_set"}
)


class NativeRegionValidationError(ValueError):
    """A detached native region or reference is inconsistent with its recipe."""


class RecipeRegionSelector(str, Enum):
    """Typed application-only selectors for built-in recipe regions."""

    BOTTOM = "BOTTOM"
    RIGHT = "RIGHT"
    TOP = "TOP"
    LEFT = "LEFT"
    FRONT = "FRONT"
    BACK = "BACK"
    OUTER = "OUTER"
    HOLE = "HOLE"


_SELECTOR_ORDER = (
    RecipeRegionSelector.BOTTOM,
    RecipeRegionSelector.RIGHT,
    RecipeRegionSelector.TOP,
    RecipeRegionSelector.LEFT,
    RecipeRegionSelector.FRONT,
    RecipeRegionSelector.BACK,
    RecipeRegionSelector.OUTER,
    RecipeRegionSelector.HOLE,
)


@dataclass(frozen=True, slots=True)
class CompiledDomainRegionSource:
    """The compiled full-dimensional recipe domain."""


@dataclass(frozen=True, slots=True)
class LogicalReferencesRegionSource:
    """A canonical non-empty group of user-owned logical references."""

    references: tuple[LogicalEntityRef, ...]

    def __post_init__(self) -> None:
        references = tuple(self.references)
        if not references:
            raise NativeRegionValidationError(
                "logical-reference region source must not be empty"
            )
        if any(type(reference) is not LogicalEntityRef for reference in references):
            raise TypeError(
                "logical-reference region source requires LogicalEntityRef values"
            )
        if len(set(references)) != len(references):
            raise NativeRegionValidationError(
                "logical-reference region source contains duplicate references"
            )
        if len({reference.kind for reference in references}) != 1:
            raise NativeRegionValidationError(
                "one native region cannot mix logical entity kinds"
            )
        object.__setattr__(
            self,
            "references",
            tuple(sorted(references, key=logical_ref_sort_key)),
        )


@dataclass(frozen=True, slots=True)
class RecipeRegionSource:
    """One built-in region resolved by a typed recipe selector."""

    selector: RecipeRegionSelector

    def __post_init__(self) -> None:
        if type(self.selector) is not RecipeRegionSelector:
            raise TypeError("recipe region selector must be RecipeRegionSelector")


NativeRegionSource = (
    CompiledDomainRegionSource | LogicalReferencesRegionSource | RecipeRegionSource
)


@dataclass(frozen=True, slots=True)
class NativeRegionDescriptor:
    """One stable native region and the products it can generate."""

    name: str
    source: NativeRegionSource
    products: frozenset[NativeRegionProduct]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise NativeRegionValidationError(
                "native region name must be a non-empty string"
            )
        if self.name != self.name.strip():
            raise NativeRegionValidationError(
                "native region name must not contain surrounding whitespace"
            )
        if not isinstance(
            self.source,
            (
                CompiledDomainRegionSource,
                LogicalReferencesRegionSource,
                RecipeRegionSource,
            ),
        ):
            raise TypeError("unsupported native region source")
        products = frozenset(self.products)
        if not products or not products.issubset(_PRODUCTS):
            raise NativeRegionValidationError(
                f"unsupported native region products: {sorted(products)!r}"
            )
        object.__setattr__(self, "products", products)


def describe_native_regions(
    recipe: NativeGeometry,
    named_regions: Iterable[Any] | Mapping[str, Any] = (),
    *,
    mesh_settings: MeshSettings | None = None,
    mesh_contract: NativeMeshContract | None = None,
) -> tuple[NativeRegionDescriptor, ...]:
    """Describe every built-in and user region without constructing CAD or mesh."""

    contract = _resolve_mesh_contract(recipe, mesh_settings, mesh_contract)
    topology = describe_recipe_topology(recipe)
    domain_products = _products_for_entity_dimension(
        topology.dimension,
        topology.dimension,
        contract,
    )
    descriptors: list[NativeRegionDescriptor] = [
        NativeRegionDescriptor(
            "DOMAIN",
            CompiledDomainRegionSource(),
            domain_products,
        )
    ]
    if topology.exact:
        boundary_products: frozenset[NativeRegionProduct] = (
            frozenset({"node_set", "edge"})
            if topology.dimension == 2
            else frozenset({"node_set", "surface"})
        )
        selectors = set(_builtin_region_selectors(recipe, topology))
        descriptors.extend(
            NativeRegionDescriptor(
                selector.value,
                RecipeRegionSource(selector),
                boundary_products,
            )
            for selector in _SELECTOR_ORDER
            if selector in selectors
        )

    raw_regions = (
        tuple(named_regions.values())
        if isinstance(named_regions, Mapping)
        else tuple(named_regions)
    )
    user_descriptors: list[NativeRegionDescriptor] = []
    occupied = {
        descriptor.name.casefold(): descriptor.name for descriptor in descriptors
    }
    for region in raw_regions:
        name = getattr(region, "name", None)
        if type(name) is not str or not name.strip():
            raise NativeRegionValidationError(
                "named region name must be a non-empty string"
            )
        if name != name.strip():
            raise NativeRegionValidationError(
                "named region name must not contain surrounding whitespace"
            )
        folded = name.casefold()
        if folded in occupied:
            raise NativeRegionValidationError(
                f"named region {name!r} conflicts with {occupied[folded]!r}"
            )
        references = getattr(region, "references", None)
        if references is None:
            raise NativeRegionValidationError(
                f"named region {name!r} does not provide logical references"
            )
        source = LogicalReferencesRegionSource(tuple(references))
        validated = validate_logical_references(recipe, source.references)
        source = LogicalReferencesRegionSource(validated)
        descriptor = NativeRegionDescriptor(
            name,
            source,
            _products_for_references(topology, validated, contract),
        )
        occupied[folded] = name
        user_descriptors.append(descriptor)
    descriptors.extend(
        sorted(user_descriptors, key=lambda descriptor: descriptor.name.casefold())
    )
    return tuple(descriptors)


def validate_logical_reference(
    recipe: NativeGeometry,
    reference: LogicalEntityRef,
    *,
    allowed_kinds: Iterable[EntityKind] | None = None,
    require_exact: bool = True,
) -> LogicalEntity:
    """Validate one detached reference and return its catalog entity."""

    if type(reference) is not LogicalEntityRef:
        raise TypeError("reference must be a LogicalEntityRef")
    topology = describe_recipe_topology(recipe)
    if require_exact and not topology.exact:
        raise NativeRegionValidationError(
            "geometry-dependent references require exact recipe topology"
        )
    if allowed_kinds is not None and reference.kind not in frozenset(allowed_kinds):
        raise NativeRegionValidationError(
            f"logical reference kind {reference.kind!r} is not allowed"
        )
    try:
        entity = topology.entity(reference.logical_id)
    except KeyError as error:
        raise NativeRegionValidationError(
            f"unknown logical reference {reference.logical_id!r}"
        ) from error
    if entity.kind != reference.kind:
        raise NativeRegionValidationError(
            f"logical reference kind mismatch for {reference.logical_id!r}"
        )
    if not entity.selectable:
        raise NativeRegionValidationError(
            f"logical reference {reference.logical_id!r} is not selectable"
        )
    return entity


def validate_logical_references(
    recipe: NativeGeometry,
    references: Iterable[LogicalEntityRef],
    *,
    allowed_kinds: Iterable[EntityKind] | None = None,
    require_exact: bool = True,
) -> tuple[LogicalEntityRef, ...]:
    """Validate and canonicalize one non-empty homogeneous reference group."""

    source = LogicalReferencesRegionSource(tuple(references))
    for reference in source.references:
        validate_logical_reference(
            recipe,
            reference,
            allowed_kinds=allowed_kinds,
            require_exact=require_exact,
        )
    return source.references


def require_native_region_product(
    descriptors: Iterable[NativeRegionDescriptor],
    region_name: str,
    product: NativeRegionProduct,
) -> NativeRegionDescriptor:
    """Return one named region or reject a missing product capability."""

    if product not in _PRODUCTS:
        raise NativeRegionValidationError(
            f"unsupported native region product {product!r}"
        )
    matches = tuple(
        descriptor for descriptor in descriptors if descriptor.name == region_name
    )
    if len(matches) != 1:
        raise NativeRegionValidationError(f"unknown native region {region_name!r}")
    descriptor = matches[0]
    if product not in descriptor.products:
        raise NativeRegionValidationError(
            f"native region {region_name!r} cannot produce {product!r}"
        )
    return descriptor


def validate_native_authoring_context(
    recipe: NativeGeometry,
    named_regions: Iterable[Any] | Mapping[str, Any] = (),
    *,
    local_controls: Iterable[Any] = (),
    region_requirements: Iterable[tuple[str, NativeRegionProduct]] = (),
    mesh_settings: MeshSettings | None = None,
    mesh_contract: NativeMeshContract | None = None,
) -> tuple[NativeRegionDescriptor, ...]:
    """Validate detached regions, local controls, and named capabilities together."""

    contract = _resolve_mesh_contract(recipe, mesh_settings, mesh_contract)
    descriptors = describe_native_regions(
        recipe,
        named_regions,
        mesh_contract=contract,
    )
    for control in local_controls:
        target = getattr(control, "target", None)
        validate_logical_reference(
            recipe,
            target,
            allowed_kinds=("point", "edge", "face"),
        )
        falloff = getattr(control, "falloff", None)
        if getattr(falloff, "reference", None) == "target_radius":
            if contract.dimension == 1:
                raise NativeRegionValidationError(
                    "target_radius falloff is not supported for native line "
                    "local mesh controls; use global_size"
                )
            resolve_target_radius(recipe, target)
    for region_name, product in region_requirements:
        require_native_region_product(descriptors, region_name, product)
    return descriptors


def _builtin_region_selectors(
    recipe: NativeGeometry,
    topology: RecipeTopology,
) -> tuple[RecipeRegionSelector, ...]:
    if isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        return _builtin_region_selectors(
            recipe.base,
            describe_recipe_topology(recipe.base),
        )
    if isinstance(recipe, RectangleGeometry):
        return (
            RecipeRegionSelector.BOTTOM,
            RecipeRegionSelector.RIGHT,
            RecipeRegionSelector.TOP,
            RecipeRegionSelector.LEFT,
        )
    if isinstance(recipe, DiskGeometry):
        return (RecipeRegionSelector.OUTER,)
    if isinstance(recipe, PlateWithHoleGeometry):
        return (
            RecipeRegionSelector.BOTTOM,
            RecipeRegionSelector.RIGHT,
            RecipeRegionSelector.TOP,
            RecipeRegionSelector.LEFT,
            RecipeRegionSelector.HOLE,
        )
    if isinstance(recipe, BoxGeometry):
        return (
            RecipeRegionSelector.BOTTOM,
            RecipeRegionSelector.TOP,
            RecipeRegionSelector.FRONT,
            RecipeRegionSelector.RIGHT,
            RecipeRegionSelector.BACK,
            RecipeRegionSelector.LEFT,
        )
    if isinstance(recipe, CylinderGeometry):
        return (
            RecipeRegionSelector.BOTTOM,
            RecipeRegionSelector.TOP,
            RecipeRegionSelector.OUTER,
        )
    if isinstance(recipe, ExtrudedGeometry):
        selectors = [
            RecipeRegionSelector.BOTTOM,
            RecipeRegionSelector.TOP,
            RecipeRegionSelector.OUTER,
        ]
        base_topology = describe_recipe_topology(recipe.base)
        if _has_hole_boundary(base_topology):
            selectors.append(RecipeRegionSelector.HOLE)
        return tuple(selectors)
    if isinstance(recipe, (SketchGeometry, BooleanGeometry)):
        if _has_hole_boundary(topology):
            return (
                RecipeRegionSelector.BOTTOM,
                RecipeRegionSelector.RIGHT,
                RecipeRegionSelector.TOP,
                RecipeRegionSelector.LEFT,
                RecipeRegionSelector.HOLE,
            )
        logical_ids = topology.signature.logical_ids
        if "edge:bottom" in logical_ids:
            return (
                RecipeRegionSelector.BOTTOM,
                RecipeRegionSelector.RIGHT,
                RecipeRegionSelector.TOP,
                RecipeRegionSelector.LEFT,
            )
        if "edge:outer" in logical_ids:
            return (RecipeRegionSelector.OUTER,)
    return ()


def _has_hole_boundary(topology: RecipeTopology) -> bool:
    return any(
        entity.kind == "edge"
        and entity.semantic_role == "boundary.hole-loop"
        and entity.selectable
        for entity in topology.entities
    )


def _products_for_references(
    topology: RecipeTopology,
    references: tuple[LogicalEntityRef, ...],
    contract: NativeMeshContract,
) -> frozenset[NativeRegionProduct]:
    dimensions = {
        topology.entity(reference.logical_id).dimension for reference in references
    }
    if len(dimensions) != 1:
        raise NativeRegionValidationError("one native region cannot mix CAD dimensions")
    entity_dimension = next(iter(dimensions))
    return _products_for_entity_dimension(
        entity_dimension,
        topology.dimension,
        contract,
    )


def _products_for_entity_dimension(
    entity_dimension: int,
    recipe_dimension: int,
    contract: NativeMeshContract,
) -> frozenset[NativeRegionProduct]:
    if entity_dimension == recipe_dimension:
        products: set[NativeRegionProduct] = {"element_set"}
        if contract.line_element_type == "Beam2" and recipe_dimension == 1:
            products.add("beam_element_set")
        return frozenset(products)
    if entity_dimension == 0:
        return frozenset({"node_set"})
    if recipe_dimension == 2 and entity_dimension == 1:
        return frozenset({"node_set", "edge"})
    if recipe_dimension == 3 and entity_dimension == 2:
        return frozenset({"node_set", "surface"})
    return frozenset({"node_set"})


def _resolve_mesh_contract(
    recipe: NativeGeometry,
    mesh_settings: MeshSettings | None,
    mesh_contract: NativeMeshContract | None,
) -> NativeMeshContract:
    if mesh_contract is not None:
        if type(mesh_contract) is not NativeMeshContract:
            raise TypeError("mesh_contract must be NativeMeshContract or None")
        return mesh_contract
    # The helper intentionally returns an incomplete contract for an
    # unconfigured wire, while keeping continuum defaults unchanged.
    return describe_native_mesh_contract(recipe, mesh_settings)


__all__ = [
    "CompiledDomainRegionSource",
    "LogicalReferencesRegionSource",
    "NativeRegionDescriptor",
    "NativeRegionProduct",
    "NativeRegionSource",
    "NativeRegionValidationError",
    "RecipeRegionSelector",
    "RecipeRegionSource",
    "describe_native_regions",
    "require_native_region_product",
    "validate_logical_reference",
    "validate_logical_references",
    "validate_native_authoring_context",
]
