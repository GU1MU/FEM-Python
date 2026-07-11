from pathlib import Path


def prepare_output_path(path: str | Path) -> Path:
    """Create the output parent directory and return a Path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
