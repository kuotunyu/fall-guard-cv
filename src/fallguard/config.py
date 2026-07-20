"""環境設定載入:金鑰與模型字串一律讀 .env,程式不寫死(CLAUDE.md 模型政策)。

金鑰命名對齊(docs/PLAN.md D7):langchain-google-genai 官方以 GOOGLE_API_KEY 為主,
.env 只有 GEMINI_API_KEY 時自動補上 GOOGLE_API_KEY。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")

if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    gemini_model: str = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    gemini_lite_model: str = os.environ.get("GEMINI_LITE_MODEL", "gemini-2.5-flash-lite")
    openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    pose_model: str = os.environ.get("POSE_MODEL", "yolo26m-pose.pt")
    fall_confirm_seconds: float = float(os.environ.get("FALL_CONFIRM_SECONDS", "10"))
    alert_cooldown_seconds: float = float(os.environ.get("ALERT_COOLDOWN_SECONDS", "120"))
    local_only: bool = _bool("LOCAL_ONLY", False)
    send_image: bool = _bool("SEND_IMAGE", True)
    discord_webhook_url: str | None = os.environ.get("DISCORD_WEBHOOK_URL") or None

    data_dir: Path = REPO_ROOT / "data"
    models_dir: Path = REPO_ROOT / "models"
    events_dir: Path = REPO_ROOT / "events"


settings = Settings()
