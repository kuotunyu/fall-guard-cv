"""Phase 0 冒煙測試:套件可匯入、設定有預設值、金鑰缺席時不炸。"""

from fallguard.config import REPO_ROOT, settings


def test_package_importable():
    import fallguard

    assert fallguard.__version__


def test_settings_defaults():
    assert settings.gemini_model
    assert settings.pose_model.endswith("-pose.pt")
    assert settings.fall_confirm_seconds > 0
    assert settings.alert_cooldown_seconds > 0
    assert isinstance(settings.local_only, bool)
    assert isinstance(settings.send_image, bool)


def test_repo_layout():
    assert (REPO_ROOT / "docs" / "PLAN.md").exists()
    assert (REPO_ROOT / "PROGRESS.md").exists()
    assert (REPO_ROOT / ".env.example").exists()
