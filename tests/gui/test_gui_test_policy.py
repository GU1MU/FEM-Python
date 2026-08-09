from __future__ import annotations

import ast
from pathlib import Path


GUI_TEST_ROOT = Path(__file__).resolve().parent


def _number(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Name):
        return function.id
    return ""


def _timeout_argument(node: ast.Call) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg in {"timeout", "timeout_seconds"}:
            return keyword.value
    return node.args[0] if node.args else None


def _is_thread_join(node: ast.Call) -> bool:
    function = node.func
    if not isinstance(function, ast.Attribute) or function.attr != "join":
        return False
    receiver = function.value
    return isinstance(receiver, ast.Name) and any(
        token in receiver.id.casefold()
        for token in ("closer", "thread", "worker")
    )


def _function_defaults(node: ast.FunctionDef) -> tuple[tuple[str, ast.AST], ...]:
    positional = node.args.posonlyargs + node.args.args
    aligned = zip(positional[-len(node.args.defaults) :], node.args.defaults)
    keyword_only = (
        (argument, default)
        for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
        if default is not None
    )
    return tuple((argument.arg, default) for argument, default in (*aligned, *keyword_only))


def test_gui_tests_keep_real_waits_within_two_seconds() -> None:
    violations: list[str] = []
    for path in sorted(GUI_TEST_ROOT.glob("test_*.py")):
        if path == Path(__file__).resolve():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node)
                if name == "wait" or _is_thread_join(node):
                    timeout = _number(_timeout_argument(node))
                    if timeout is None or timeout > 2.0:
                        violations.append(f"{path.name}:{node.lineno} {name}")
                elif name == "result":
                    timeout = _number(_timeout_argument(node))
                    if timeout is not None and timeout > 2.0:
                        violations.append(f"{path.name}:{node.lineno} result")
                elif name == "qWait":
                    delay_ms = _number(node.args[0] if node.args else None)
                    if delay_ms is None or delay_ms > 10.0:
                        violations.append(f"{path.name}:{node.lineno} qWait")
                elif name == "sleep":
                    delay = _number(node.args[0] if node.args else None)
                    if delay is None or delay > 0.01:
                        violations.append(f"{path.name}:{node.lineno} sleep")
            elif isinstance(node, ast.FunctionDef):
                for name, default in _function_defaults(node):
                    if name not in {"timeout", "timeout_ms"}:
                        continue
                    value = _number(default)
                    limit = 2_000.0 if name == "timeout_ms" else 2.0
                    if value is None or value > limit:
                        violations.append(
                            f"{path.name}:{node.lineno} {name} default"
                        )
            elif (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Add)
                and isinstance(node.left, ast.Call)
                and _call_name(node.left) == "monotonic"
            ):
                budget = _number(node.right)
                if budget is not None and budget > 2.0:
                    violations.append(
                        f"{path.name}:{node.lineno} monotonic deadline"
                    )

    assert violations == []
