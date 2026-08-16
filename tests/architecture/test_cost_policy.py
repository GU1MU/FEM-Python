from __future__ import annotations

import ast
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent
GUI_TEST_ROOT = TEST_ROOT / "gui"


def _number(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


def _attribute_chain(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _has_slow_marker(nodes: tuple[ast.AST, ...] | list[ast.AST]) -> bool:
    return any(
        _attribute_chain(node) == ("pytest", "mark", "slow")
        for node in nodes
    )


def _module_is_slow(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if not any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in targets
        ):
            continue
        value = node.value
        if value is not None and any(
            _attribute_chain(candidate) == ("pytest", "mark", "slow")
            for candidate in ast.walk(value)
        ):
            return True
    return False


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


class _CostVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, *, module_slow: bool) -> None:
        self.path = path
        self.slow_scope = module_slow
        self.violations: list[str] = []

    def _record(self, node: ast.AST, rule: str) -> None:
        self.violations.append(f"{self.path.name}:{node.lineno} {rule}")

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        previous = self.slow_scope
        self.slow_scope = previous or _has_slow_marker(node.decorator_list)
        self.generic_visit(node)
        self.slow_scope = previous

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        chain = _attribute_chain(node.func)
        terminal = chain[-1] if chain else ""
        if terminal == "wait" or _is_thread_join(node):
            timeout = _number(_timeout_argument(node))
            if timeout is None or timeout > 2.0:
                self._record(node, f"{terminal} exceeds 2 seconds")
        elif terminal == "sleep":
            delay = _number(node.args[0] if node.args else None)
            if delay is None or delay > 0.05:
                self._record(node, "sleep exceeds 50 milliseconds")
        elif chain == ("subprocess", "run"):
            timeout_node = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "timeout"
                ),
                None,
            )
            timeout = _number(timeout_node)
            if timeout is None or timeout > 15.0:
                self._record(node, "subprocess.run lacks a <=15 second timeout")
        elif terminal == "range" and not self.slow_scope:
            values = tuple(_number(argument) for argument in node.args)
            if any(value is not None and abs(value) > 50_000 for value in values):
                self._record(node, "large range requires pytest.mark.slow")
        self.generic_visit(node)


def test_non_gui_tests_keep_runtime_and_allocation_costs_bounded() -> None:
    violations: list[str] = []
    for path in sorted(TEST_ROOT.rglob("test_*.py")):
        if path == Path(__file__).resolve() or path.is_relative_to(GUI_TEST_ROOT):
            continue
        source = path.read_text(encoding="utf-8")
        if not any(
            token in source
            for token in (".wait(", ".join(", "sleep(", "subprocess.run(", "range(")
        ):
            continue
        tree = ast.parse(source, filename=str(path))
        visitor = _CostVisitor(path, module_slow=_module_is_slow(tree))
        visitor.visit(tree)
        violations.extend(visitor.violations)

    assert violations == []
