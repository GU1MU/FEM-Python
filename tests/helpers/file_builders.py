from pathlib import Path


def write_inp(directory, filename, lines):
    path = Path(directory) / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
