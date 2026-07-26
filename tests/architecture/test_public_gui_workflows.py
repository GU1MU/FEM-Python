"""Structural gates for public-command GUI workflow tests."""

from __future__ import annotations

import ast
from pathlib import Path


TESTS_ROOT = Path(__file__).parents[1]
INTEGRATION_ROOT = TESTS_ROOT / "integration"
PUBLIC_GUI_WORKFLOW = (
    TESTS_ROOT / "gui" / "test_public_workflow_commands.py"
)


def _main_window_aliases(tree: ast.AST) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "fem_gui.main_window"
        ):
            aliases.update(
                item.asname or item.name
                for item in node.names
                if item.name == "FEMMainWindow"
            )
    return aliases


def _annotation_name(annotation: ast.expr | None) -> str | None:
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Constant) and isinstance(
        annotation.value,
        str,
    ):
        return annotation.value
    return None


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {
            name
            for item in target.elts
            for name in _assigned_names(item)
        }
    return set()


def _main_window_receivers(
    tree: ast.AST,
    aliases: set[str],
) -> set[str]:
    receivers = set(aliases)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
            receivers.update(
                argument.arg
                for argument in arguments
                if _annotation_name(argument.annotation) in aliases
            )
        if isinstance(node, ast.Assign):
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in aliases
            ):
                for target in node.targets:
                    receivers.update(_assigned_names(target))
        if isinstance(node, ast.AnnAssign):
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in aliases
            ):
                receivers.update(_assigned_names(node.target))
    return receivers


def _workflow_modules() -> tuple[tuple[Path, ast.Module], ...]:
    candidates = (
        *sorted(INTEGRATION_ROOT.glob("*gui*.py")),
        PUBLIC_GUI_WORKFLOW,
    )
    modules: list[tuple[Path, ast.Module]] = []
    for path in candidates:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _main_window_aliases(tree):
            modules.append((path, tree))
    return tuple(modules)


def _declared_entrypoints(tree: ast.Module) -> tuple[str, ...] | None:
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else (statement.target,)
        )
        if not any(
            isinstance(target, ast.Name)
            and target.id == "PUBLIC_GUI_WORKFLOW_ENTRYPOINTS"
            for target in targets
        ):
            continue
        try:
            value = ast.literal_eval(statement.value)
        except (TypeError, ValueError):
            return None
        if (
            isinstance(value, tuple)
            and value
            and all(type(item) is str for item in value)
        ):
            return value
        return None
    return None


def test_public_gui_workflows_declare_their_public_entrypoints() -> None:
    missing_or_invalid = []
    for path, tree in _workflow_modules():
        entrypoints = _declared_entrypoints(tree)
        if (
            entrypoints is None
            or any(name.startswith("_") for name in entrypoints)
        ):
            missing_or_invalid.append(path.relative_to(TESTS_ROOT).as_posix())

    assert missing_or_invalid == []


def test_public_gui_workflows_do_not_call_main_window_private_seams() -> None:
    violations: list[str] = []
    for path, tree in _workflow_modules():
        aliases = _main_window_aliases(tree)
        receivers = _main_window_receivers(tree, aliases)
        for call in (
            node for node in ast.walk(tree) if isinstance(node, ast.Call)
        ):
            function = call.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr.startswith("_")
                and isinstance(function.value, ast.Name)
                and function.value.id in receivers
            ):
                relative = path.relative_to(TESTS_ROOT).as_posix()
                violations.append(
                    f"{relative}:{call.lineno}:"
                    f"{function.value.id}.{function.attr}"
                )

    assert violations == []
