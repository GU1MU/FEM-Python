"""从仓库根目录启动中文有限元 GUI。"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    from fem_gui.app import main as run_gui

    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
