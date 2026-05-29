from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_TEMP_ROOT = PROJECT_ROOT / "temp" / "tests"


def temporary_directory():
    TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return TemporaryDirectory(dir=TEST_TEMP_ROOT)


def test_output_path(*parts):
    path = TEST_TEMP_ROOT.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def fresh_test_output_dir(*parts):
    path = TEST_TEMP_ROOT.joinpath(*parts)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path
