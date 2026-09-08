from pathlib import Path


def test_fem_package_does_not_import_agent_package():
    fem_root = Path(__file__).resolve().parents[2] / "src" / "fem"

    offenders = [
        str(path.relative_to(fem_root))
        for path in fem_root.rglob("*.py")
        if "fem_agent" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
