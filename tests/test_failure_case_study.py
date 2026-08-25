from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_failure_case_study_preserves_generalization_limits() -> None:
    case_study_path = ROOT / "docs" / "FAILURE_ANALYSIS_CASE_STUDY.md"
    assert case_study_path.is_file(), "缺少 failure analysis case study"

    text = case_study_path.read_text(encoding="utf-8")
    required_sections = (
        "## Question",
        "## Frozen protocol",
        "## Observed failure",
        "## Root-cause hypotheses",
        "## What was deliberately not tuned",
        "## Engineering lessons",
        "## Next experiment contract",
    )
    required_markers = (
        "Sensitivity 0.559",
        "Specificity 0.000",
        "only 3 non-fall ADL videos",
        "0.3-second evaluation",
        "10-second runtime",
        "not deployment evidence",
    )

    for marker in (*required_sections, *required_markers):
        assert marker in text, f"failure case study 缺少必要標記：{marker}"
