"""套件可匯入、設定有預設值，且公開 checkout 具備必要檔案。"""

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
    assert (REPO_ROOT / "README.md").exists()
    assert (REPO_ROOT / ".env.example").exists()
