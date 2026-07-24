"""Versioned JSON persistence for editable native GUI projects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    EdgeLoad,
    MaterialDefinition,
    NodalLoad,
    OutputRequest,
    SurfaceLoad,
)

from .document import FEMDocument, FeatureRecord, NativePart, NamedRegion, RegionAssignment, SectionDefinition, WorkflowState
from .preprocessing import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    LocalMeshControl,
    MeshSettings,
    MovedGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
)


SCHEMA_VERSION = 1


def save_native_project(path: str | Path, document: FEMDocument) -> Path:
    if document.source_kind != "native":
        raise ValueError("只有自主模型可以保存为项目文件")
    if document.geometry_recipe is None:
        raise ValueError("请先创建草图或几何后再保存项目")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA_VERSION,
        "source": "native",
        "parts": [{"name": part.name, "body_name": part.body_name} for part in document.parts],
        "geometry": _encode_geometry(document.geometry_recipe),
        "mesh_settings": _encode_mesh_settings(document.mesh_settings),
        "named_regions": [
            {"name": region.name, "entity_kind": region.entity_kind, "entity_ids": list(region.entity_ids)}
            for region in document.named_regions.values()
        ],
        "materials": [
            {"name": material.name, "properties": dict(material.properties)}
            for material in document.material_definitions
        ],
        "sections": [
            {"name": section.name, "material": section.material, "section_type": section.section_type, "properties": section.properties}
            for section in document.section_definitions
        ],
        "assignments": [
            {"section_name": assignment.section_name, "region_name": assignment.region_name}
            for assignment in document.region_assignments
        ],
        "steps": [_encode_step(step) for step in document.analysis_definitions],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    document.native_project_path = target
    document.dirty = False
    return target


def load_native_project(path: str | Path, document: FEMDocument) -> None:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA_VERSION or payload.get("source") != "native":
        raise ValueError("不是受支持的自主项目文件")
    recipe = _decode_geometry(payload["geometry"])
    document.close()
    document.source_kind = "native"
    document.native_project_path = source
    document.geometry_recipe = recipe
    document.mesh_settings = _decode_mesh_settings(payload.get("mesh_settings"))
    document.parts = [NativePart(**item) for item in payload.get("parts", ())] or [NativePart()]
    document.feature_history = _history_for_recipe(recipe)
    document.named_regions = {
        item["name"]: NamedRegion(item["name"], item["entity_kind"], tuple(item.get("entity_ids", ())))
        for item in payload.get("named_regions", ())
    }
    document.material_definitions = [
        MaterialDefinition(item["name"], dict(item.get("properties", {})))
        for item in payload.get("materials", ())
    ]
    document.section_definitions = [
        SectionDefinition(item["name"], item["material"], item.get("section_type", "solid"), dict(item.get("properties", {})))
        for item in payload.get("sections", ())
    ]
    document.region_assignments = [RegionAssignment(**item) for item in payload.get("assignments", ())]
    document.analysis_definitions = [_decode_step(item) for item in payload.get("steps", ())]
    document.workflow = WorkflowState(reason="项目已打开，请重新生成网格后检查模型")
    document.dirty = False


def _encode_geometry(recipe: Any) -> dict[str, Any]:
    if isinstance(recipe, SketchGeometry):
        return {"type": "SketchGeometry", "name": recipe.name, "contours": [_encode_contour(item) for item in recipe.contours]}
    if isinstance(recipe, RectangleGeometry): return {"type": "RectangleGeometry", "name": recipe.name, "width": recipe.width, "height": recipe.height}
    if isinstance(recipe, DiskGeometry): return {"type": "DiskGeometry", "name": recipe.name, "radius": recipe.radius}
    if isinstance(recipe, BoxGeometry): return {"type": "BoxGeometry", "name": recipe.name, "width": recipe.width, "depth": recipe.depth, "height": recipe.height}
    if isinstance(recipe, CylinderGeometry): return {"type": "CylinderGeometry", "name": recipe.name, "radius": recipe.radius, "height": recipe.height}
    if isinstance(recipe, PlateWithHoleGeometry): return {"type": "PlateWithHoleGeometry", "name": recipe.name, "width": recipe.width, "height": recipe.height, "hole_x": recipe.hole_x, "hole_y": recipe.hole_y, "hole_radius": recipe.hole_radius}
    if isinstance(recipe, MovedGeometry): return {"type": "MovedGeometry", "base": _encode_geometry(recipe.base), "dx": recipe.dx, "dy": recipe.dy, "dz": recipe.dz}
    if isinstance(recipe, RotatedGeometry): return {"type": "RotatedGeometry", "base": _encode_geometry(recipe.base), "axis": recipe.axis, "angle_degrees": recipe.angle_degrees}
    if isinstance(recipe, ExtrudedGeometry): return {"type": "ExtrudedGeometry", "base": _encode_geometry(recipe.base), "height": recipe.height}
    if isinstance(recipe, BooleanGeometry): return {"type": "BooleanGeometry", "name": recipe.name, "operation": recipe.operation, "object": _encode_geometry(recipe.object_geometry), "tool": _encode_geometry(recipe.tool_geometry)}
    raise TypeError(f"不能保存几何类型：{type(recipe).__name__}")


def _decode_geometry(data: dict[str, Any]):
    kind = data["type"]
    if kind == "SketchGeometry": return SketchGeometry(data["name"], tuple(_decode_contour(item) for item in data["contours"]))
    if kind == "RectangleGeometry": return RectangleGeometry(data["name"], data["width"], data["height"])
    if kind == "DiskGeometry": return DiskGeometry(data["name"], data["radius"])
    if kind == "BoxGeometry": return BoxGeometry(data["name"], data["width"], data["depth"], data["height"])
    if kind == "CylinderGeometry": return CylinderGeometry(data["name"], data["radius"], data["height"])
    if kind == "PlateWithHoleGeometry": return PlateWithHoleGeometry(data["name"], data["width"], data["height"], data["hole_x"], data["hole_y"], data["hole_radius"])
    if kind == "MovedGeometry": return MovedGeometry(_decode_geometry(data["base"]), data["dx"], data["dy"], data.get("dz", 0.0))
    if kind == "RotatedGeometry": return RotatedGeometry(_decode_geometry(data["base"]), data["axis"], data["angle_degrees"])
    if kind == "ExtrudedGeometry": return ExtrudedGeometry(_decode_geometry(data["base"]), data["height"])
    if kind == "BooleanGeometry": return BooleanGeometry(data["name"], data["operation"], _decode_geometry(data["object"]), _decode_geometry(data["tool"]))
    raise ValueError(f"未知几何类型：{kind}")


def _encode_contour(contour: Any) -> dict[str, Any]:
    if isinstance(contour, SketchRectangle): return {"type": "rectangle", "operation": contour.operation, "x": contour.x, "y": contour.y, "width": contour.width, "height": contour.height}
    return {"type": "circle", "operation": contour.operation, "x": contour.x, "y": contour.y, "radius": contour.radius}


def _decode_contour(data: dict[str, Any]):
    if data["type"] == "rectangle": return SketchRectangle(data["operation"], data["x"], data["y"], data["width"], data["height"])
    return SketchCircle(data["operation"], data["x"], data["y"], data["radius"])


def _encode_mesh_settings(settings: Any) -> dict[str, Any] | None:
    if settings is None: return None
    return {"size": settings.size, "order": settings.order, "cell_shape": settings.cell_shape, "local_controls": [{"entity_kind": control.entity_kind, "entity_id": control.entity_id, "size": control.size} for control in settings.local_controls]}


def _decode_mesh_settings(data: dict[str, Any] | None):
    if data is None: return None
    return MeshSettings(data["size"], data.get("order", 1), data.get("cell_shape", "triangle"), local_controls=tuple(LocalMeshControl(**control) for control in data.get("local_controls", ())))


def _encode_step(step: AnalysisStep) -> dict[str, Any]:
    return {"name": step.name, "procedure": step.procedure, "metadata": dict(step.metadata), "boundaries": [item.__dict__ for item in step.boundaries], "cloads": [item.__dict__ for item in step.cloads], "edge_loads": [item.__dict__ for item in step.edge_loads], "surface_loads": [item.__dict__ for item in step.surface_loads], "outputs": [{"kind": item.kind, "target": item.target, "variables": list(item.variables), "metadata": dict(item.metadata)} for item in step.outputs]}


def _decode_step(data: dict[str, Any]) -> AnalysisStep:
    return AnalysisStep(data["name"], data.get("procedure", "static"), boundaries=[DisplacementConstraint(**item) for item in data.get("boundaries", ())], cloads=[NodalLoad(**item) for item in data.get("cloads", ())], edge_loads=[EdgeLoad(**item) for item in data.get("edge_loads", ())], surface_loads=[SurfaceLoad(**item) for item in data.get("surface_loads", ())], outputs=[OutputRequest(item["kind"], item["target"], item.get("variables", ()), item.get("metadata", {})) for item in data.get("outputs", ())], metadata=data.get("metadata", {}))


def _history_for_recipe(recipe: Any) -> list[FeatureRecord]:
    # A persisted recipe is authoritative; labels are reconstructed for the shallow tree.
    if isinstance(recipe, SketchGeometry): return [FeatureRecord("Sketch-1", "sketch")]
    if isinstance(recipe, ExtrudedGeometry): return _history_for_recipe(recipe.base) + [FeatureRecord("Extrude-1", "extrude")]
    if isinstance(recipe, MovedGeometry): return _history_for_recipe(recipe.base) + [FeatureRecord("Move-1", "move")]
    if isinstance(recipe, RotatedGeometry): return _history_for_recipe(recipe.base) + [FeatureRecord("Rotate-1", "rotate")]
    if isinstance(recipe, BooleanGeometry): return _history_for_recipe(recipe.object_geometry) + [FeatureRecord({"fuse": "Fuse-1", "cut": "Cut-1", "fragment": "Partition-1"}[recipe.operation], recipe.operation)]
    return [FeatureRecord("Base-1", "base")]
