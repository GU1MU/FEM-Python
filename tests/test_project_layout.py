import ast
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TESTS_ROOT = PROJECT_ROOT / "tests"


def _string_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value.replace("\\", "/")


def _resolved_import_targets(path, module_name):
    """Yield resolved import targets and source lines for a project module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_parts = module_name.split(".")
    package_parts = (
        module_parts if path.name == "__init__.py" else module_parts[:-1]
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level:
            parent_count = node.level - 1
            base_parts = package_parts[: len(package_parts) - parent_count]
            if node.module:
                base_parts += node.module.split(".")
            base = ".".join(base_parts)
        else:
            base = node.module or ""

        for alias in node.names:
            target = f"{base}.{alias.name}" if base else alias.name
            yield target, node.lineno


def _module_name(path):
    relative = path.relative_to(SRC_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_standard_library_or_gmsh(target):
    root = target.split(".", 1)[0]
    return root == "gmsh" or root in sys.stdlib_module_names


def test_tests_do_not_reference_example_data_or_results_outputs():
    offenders = []
    for path in TESTS_ROOT.rglob("test_*.py"):
        if path.name == Path(__file__).name:
            continue
        for literal in _string_literals(path):
            if literal == "examples" or literal.startswith("examples/"):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)} -> {literal}")
            if literal == "results" or literal.startswith("results/"):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)} -> {literal}")

    assert offenders == []


def test_gmsh_geometry_has_no_reverse_or_third_party_runtime_dependencies():
    paths = (
        SRC_ROOT / "fem" / "geometry" / "__init__.py",
        SRC_ROOT / "fem" / "geometry" / "gmsh.py",
    )
    allowed_fem_targets = {"fem.geometry.gmsh"}
    offenders = []
    for path in paths:
        for target, lineno in _resolved_import_targets(path, _module_name(path)):
            if target == "fem" or (
                target.startswith("fem.") and target not in allowed_fem_targets
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )
            elif not target.startswith("fem.") and not _is_standard_library_or_gmsh(
                target
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert offenders == []


def test_gmsh_meshing_depends_only_on_geometry_and_backend_layers():
    paths = (
        SRC_ROOT / "fem" / "mesh" / "__init__.py",
        SRC_ROOT / "fem" / "mesh" / "gmsh.py",
    )
    offenders = []
    for path in paths:
        for target, lineno in _resolved_import_targets(path, _module_name(path)):
            allowed_fem_target = (
                target == "fem.mesh.gmsh"
                or target == "fem.geometry.gmsh"
                or target.startswith("fem.geometry.gmsh.")
            )
            if target == "fem" or (
                target.startswith("fem.") and not allowed_fem_target
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )
            elif not target.startswith("fem.") and not _is_standard_library_or_gmsh(
                target
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert offenders == []


def test_gmsh_io_imports_only_mesh_level_fem_core_types():
    path = SRC_ROOT / "fem" / "io" / "gmsh.py"
    allowed_fem_targets = {
        "fem.core.Element2D",
        "fem.core.Element3D",
        "fem.core.Mesh2D",
        "fem.core.Mesh3D",
        "fem.core.Node2D",
        "fem.core.Node3D",
        "fem.mesh.gmsh.GmshMeshRef",
    }
    resolved_imports = tuple(
        _resolved_import_targets(path, "fem.io.gmsh")
    )
    offenders = []
    for target, lineno in resolved_imports:
        if (
            target == "fem"
            or (
                target.startswith("fem.")
                and target not in allowed_fem_targets
            )
        ):
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}")

    assert offenders == []
    assert "fem.mesh.gmsh.GmshMeshRef" in {
        target for target, _ in resolved_imports
    }


def test_fem_runtime_layers_do_not_import_geometry_or_meshing():
    package_roots = (
        SRC_ROOT / "fem" / "core",
        SRC_ROOT / "fem" / "elements",
        SRC_ROOT / "fem" / "assemble",
        SRC_ROOT / "fem" / "solvers",
    )
    missing_roots = [
        str(package_root.relative_to(PROJECT_ROOT))
        for package_root in package_roots
        if not package_root.is_dir()
    ]
    assert missing_roots == []

    paths = [
        path
        for package_root in package_roots
        for path in package_root.rglob("*.py")
    ]

    forbidden_roots = ("fem.geometry", "fem.mesh")
    offenders = []
    for path in sorted(paths):
        for target, lineno in _resolved_import_targets(path, _module_name(path)):
            if any(
                target == root or target.startswith(f"{root}.")
                for root in forbidden_roots
            ):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}"
                )

    assert offenders == []
