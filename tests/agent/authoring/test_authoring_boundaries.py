from __future__ import annotations

import ast
from pathlib import Path

from fem_agent import authoring_payload, authoring_runtime, authoring_values


def test_authoring_contract_modules_do_not_import_the_workflow_controller() -> None:
    package_root = Path(authoring_runtime.__file__).parent
    contract_sources = [
        package_root / "authoring_payload.py",
        package_root / "authoring_validation.py",
        package_root / "authoring_values.py",
        *sorted((package_root / "authoring_schemas").glob("*.py")),
    ]

    for source_path in contract_sources:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                imported_names = {alias.name for alias in node.names}
                assert "authoring_runtime" not in module_name.split("."), source_path
                assert "authoring_runtime" not in imported_names, source_path
                assert "AuthoringWorkflowController" not in imported_names, source_path
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "authoring_runtime" not in alias.name.split("."), source_path


def test_runtime_public_value_imports_retain_the_owned_type_identity() -> None:
    for name in (
        "AuthoringTerminalRecord",
        "AuthoringTurnSnapshot",
        "AuthoringToolOutcome",
        "AuthoringWorkflowStage",
        "ProjectSaveProposalRecord",
    ):
        assert getattr(authoring_runtime, name) is getattr(authoring_values, name)
    assert (
        authoring_runtime.provider_safe_authoring_payload
        is authoring_payload.provider_safe_authoring_payload
    )
