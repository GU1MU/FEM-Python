import ast
from pathlib import Path


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
    package_parts = module_name.split(".")[:-1]

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


def test_gmsh_geometry_has_no_fem_runtime_dependencies():
    path = SRC_ROOT / "fem" / "geometry" / "gmsh.py"
    offenders = []
    for target, lineno in _resolved_import_targets(path, "fem.geometry.gmsh"):
        if target == "fem" or target.startswith("fem."):
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}")

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
        "fem.geometry.gmsh.GmshMeshRef",
    }
    offenders = []
    for target, lineno in _resolved_import_targets(path, "fem.io.gmsh"):
        if (
            target == "fem"
            or (
                target.startswith("fem.")
                and target not in allowed_fem_targets
            )
        ):
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno} -> {target}")

    assert offenders == []
