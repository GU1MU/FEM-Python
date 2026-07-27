from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
GUI_ROOT = SRC_ROOT / "fem_gui"
POST_ROOT = SRC_ROOT / "fem" / "post"
PROVIDER_ROOT = SRC_ROOT.joinpath(*"fem.application.results".split("."))


def _module_name(path: Path) -> str:
    relative = path.relative_to(SRC_ROOT).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolved_imports(path: Path) -> tuple[tuple[str, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    imports: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, alias.name) for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            package_parts = package.split(".")
            keep = len(package_parts) - (node.level - 1)
            prefix = package_parts[:keep]
            if node.module:
                prefix.extend(node.module.split("."))
            imported_module = ".".join(prefix)
        else:
            imported_module = node.module or ""
        imports.extend(
            (
                imported_module,
                (
                    f"{imported_module}.{alias.name}"
                    if imported_module
                    else alias.name
                ),
            )
            for alias in node.names
        )
    return tuple(imports)


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.py")))


def _forbidden_imports(
    root: Path,
    prefixes: tuple[str, ...],
) -> list[tuple[Path, str]]:
    offenders: list[tuple[Path, str]] = []
    for path in _python_files(root):
        for imported_module, imported_symbol in _resolved_imports(path):
            if imported_module.startswith(prefixes):
                offenders.append(
                    (path.relative_to(PROJECT_ROOT), imported_symbol)
                )
    return offenders


def test_retired_gui_result_modules_are_absent() -> None:
    retired = (
        GUI_ROOT / "visualization" / "result_adapter.py",
        GUI_ROOT / "visualization" / "stress_adapter.py",
        GUI_ROOT / "visualization" / "query.py",
        GUI_ROOT / "visualization" / "csv_export.py",
    )

    assert [path.relative_to(PROJECT_ROOT) for path in retired if path.exists()] == []


def test_gui_has_no_retired_result_state_or_recovery_names() -> None:
    retired_names = {
        "ResultData",
        "_source_result",
        "_source_geometry",
        "_stress_cache",
        "ensure_stress_data",
        "recovered_stress_data",
        "_average_decisions",
        "_tensor_average_decisions",
        "_legacy_average_decisions",
        "build_stress_render_geometry",
    }
    offenders: list[tuple[Path, str]] = []
    for path in _python_files(GUI_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        names.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        )
        offenders.extend(
            (path.relative_to(PROJECT_ROOT), name)
            for name in sorted(names & retired_names)
        )

    assert offenders == []


def test_gui_does_not_import_numeric_result_kernels() -> None:
    assert _forbidden_imports(
        GUI_ROOT,
        (
            "fem.elements",
            "fem.post.averaging",
            "fem.post.stress",
        ),
    ) == []


def test_post_has_no_application_io_or_consumer_dependencies() -> None:
    assert _forbidden_imports(
        POST_ROOT,
        (
            "fem.application",
            "fem.io",
            "fem_gui",
            "fem_agent",
            "PySide6",
            "pyvista",
        ),
    ) == []


def test_provider_has_no_gui_agent_loader_or_renderer_dependencies() -> None:
    assert _forbidden_imports(
        PROVIDER_ROOT,
        (
            "fem.io",
            "fem.abaqus",
            "fem_gui",
            "fem_agent",
            "PySide6",
            "pyvista",
        ),
    ) == []


def test_retired_output_and_projection_alias_tokens_are_absent() -> None:
    retired_tokens = {
        "output_request.existing",
        "output.request.not_executed",
        "ResultProjectionTaskSnapshot",
    }
    offenders: list[tuple[Path, str]] = []
    for path in _python_files(SRC_ROOT):
        source = path.read_text(encoding="utf-8")
        offenders.extend(
            (path.relative_to(PROJECT_ROOT), token)
            for token in sorted(retired_tokens)
            if token in source
        )

    assert offenders == []


def test_fem_and_gui_do_not_import_deferred_agent_implementation() -> None:
    roots = (SRC_ROOT / "fem", GUI_ROOT)
    offenders = [
        offender
        for root in roots
        for offender in _forbidden_imports(root, ("fem_agent",))
    ]

    assert offenders == []
