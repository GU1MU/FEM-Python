from pathlib import Path

import pytest


pytest_plugins = ["pytester"]
TESTS_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture_project(pytester, monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    pytester.makeini("[pytest]\naddopts = -p no:cacheprovider\n")
    pytester.makeconftest((TESTS_ROOT / "conftest.py").read_text(encoding="utf-8"))
    return pytester


def test_missing_gmsh_skips_dependent_test(fixture_project):
    with (fixture_project.path / "conftest.py").open("a", encoding="utf-8") as stream:
        stream.write('''
import sys

class MissingGmsh:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "gmsh":
            raise ModuleNotFoundError("Gmsh is unavailable", name="gmsh")

sys.meta_path.insert(0, MissingGmsh())
''')
    fixture_project.makepyfile('''
def test_native_operation(real_gmsh):
    raise AssertionError("An unavailable runtime must not reach the test body")
''')
    result = fixture_project.runpytest_subprocess(
        "--confcutdir", str(fixture_project.path), "-q", timeout=15,
    )
    result.assert_outcomes(skipped=1)


@pytest.mark.parametrize("borrowed", [False, True], ids=["owned", "borrowed"])
def test_gmsh_fixture_restores_resources(fixture_project, borrowed):
    pytest.importorskip("gmsh", reason="[optional-native-runtime] Gmsh is unavailable")
    with (fixture_project.path / "conftest.py").open("a", encoding="utf-8") as stream:
        stream.write(f'''

@pytest.fixture(autouse=True)
def surrounding_session():
    import gmsh
    assert not gmsh.isInitialized()
    if {borrowed!r}:
        gmsh.initialize()
        gmsh.model.add("preserved")
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        models = list(gmsh.model.list())
    yield
    try:
        if {borrowed!r}:
            assert gmsh.isInitialized()
            assert gmsh.model.list() == models
            assert gmsh.model.getCurrent() == "preserved"
            assert gmsh.option.getNumber("General.Terminal") == 1
            assert gmsh.option.getNumber("Mesh.Algorithm") == 6
        else:
            assert not gmsh.isInitialized()
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
''')
    fixture_project.makepyfile('''
def test_native_operation(real_gmsh):
    assert real_gmsh.isInitialized()
    assert real_gmsh.option.getNumber("General.Terminal") == 0
    real_gmsh.model.add("temporary")
    real_gmsh.model.add("temporary")
    real_gmsh.option.setNumber("Mesh.Algorithm", 2)
''')
    result = fixture_project.runpytest_subprocess(
        "--confcutdir", str(fixture_project.path), "-q", timeout=15,
    )
    result.assert_outcomes(passed=1)
