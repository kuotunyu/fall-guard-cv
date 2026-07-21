"""README/PLAN 必含章節守門(docs/PLAN.md 第 5 章/第 14 章收尾清單),避免發布前漏填 TODO。"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_README_SECTIONS = [
    "系統架構",
    "模型選型",
    "資料集與授權",
    "快速開始",
    "即時偵測",
    "評估結果",
    "隱私設計",
    "成本估算",
    "關鍵套件版本",
    "開發紀錄與授權",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_readme_has_required_sections():
    text = _read(REPO_ROOT / "README.md")
    missing = [s for s in REQUIRED_README_SECTIONS if s not in text]
    assert not missing, f"README.md 缺少章節:{missing}"


def test_readme_has_no_leftover_phase_todos():
    text = _read(REPO_ROOT / "README.md")
    assert "TODO(Phase" not in text, "README.md 仍有未填的 TODO(Phase N) 標記"


def test_readme_references_demo_gif_and_file_exists():
    text = _read(REPO_ROOT / "README.md")
    assert "docs/assets/demo.gif" in text, "README.md 未嵌入 demo.gif"
    gif_path = REPO_ROOT / "docs" / "assets" / "demo.gif"
    assert gif_path.exists(), "docs/assets/demo.gif 不存在"
    size_mb = gif_path.stat().st_size / (1024 * 1024)
    assert size_mb <= 8.0, f"demo.gif 超過 8MB 上限(現為 {size_mb:.2f}MB,見 docs/PLAN.md 第 9 章)"


def test_readme_has_urfd_citation():
    text = _read(REPO_ROOT / "README.md")
    assert "Kwolek" in text and "Kepski" in text, "README.md 缺少 URFD 引用(Kwolek & Kepski 2014)"
    assert "CC BY-NC-SA" in text, "README.md 缺少 URFD 授權標示"


def test_plan_has_decision_log_and_phase_sections():
    text = _read(REPO_ROOT / "docs" / "PLAN.md")
    assert "Decision Log" in text
    for phase_marker in ["Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4"]:
        assert phase_marker in text, f"docs/PLAN.md 缺少 {phase_marker} 段落"
