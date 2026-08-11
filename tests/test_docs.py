"""Public documentation and repository-safety contracts."""

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
    "評估紀錄與授權",
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
    assert size_mb <= 8.0, f"demo.gif 超過 8MB 上限（現為 {size_mb:.2f}MB）"


def test_readme_has_urfd_citation():
    text = _read(REPO_ROOT / "README.md")
    assert "Kwolek" in text and "Kepski" in text, "README.md 缺少 URFD 引用(Kwolek & Kepski 2014)"
    assert "CC BY-NC-SA" in text, "README.md 缺少 URFD 授權標示"


def test_agent_guardrail_identifies_the_canonical_repository():
    text = _read(REPO_ROOT / "AGENTS.md")
    assert "https://github.com/kuotunyu/fall-guard-cv" in text
    assert "fall-detection-pose" in text
    assert "不得修改" in text


def test_public_docs_do_not_depend_on_private_session_files():
    result_paths = [
        path
        for path in sorted((REPO_ROOT / "docs" / "results").glob("*.md"))
        if path.name != "vlm_comparison_detail.md"
    ]
    paths = [REPO_ROOT / "README.md", *result_paths]
    forbidden = ["PROGRESS.md", "PLAN2.md", "CLAUDE.md"]
    for path in paths:
        text = _read(path)
        for term in forbidden:
            assert term not in text, f"{path.relative_to(REPO_ROOT)} 仍依賴 {term}"


def test_ci_enforces_locked_quality_gates():
    workflow = _read(REPO_ROOT / ".github" / "workflows" / "test.yml")
    for command in [
        "uv sync --locked",
        "uv run ruff check .",
        "uv run pytest -q",
        "uv run python scripts/check_public_text.py --tracked",
    ]:
        assert command in workflow, f"CI 缺少品質門檻：{command}"
