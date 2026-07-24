"""讓 tests/ 底下的測試可以直接 import scripts/ 內的模組(例如 prepare_le2i)。

scripts/ 不是 src layout 的一部分、不會被 uv 安裝成套件,預設不在 sys.path 上;
這裡只在測試階段補這一條路徑,不影響正式執行(scripts/*.py 平常都是直接
`uv run python scripts/xxx.py` 執行,不透過 import)。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
