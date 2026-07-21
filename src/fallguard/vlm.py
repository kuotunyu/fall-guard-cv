"""Gemini VLM 現場描述(docs/PLAN.md D4/§8.2)。

跌倒事件確認後,對截圖做現場描述+嚴重程度評估,供 Discord 通報文字使用。
**告警送達是安全關鍵,VLM 只是增強**:任何失敗(安全過濾擋下、網路錯誤、金鑰缺失)
一律回傳預留文字、不丟例外,呼叫端(notify.py)永遠照常送出通報。
"""

from __future__ import annotations

import base64
from pathlib import Path

PROMPT = (
    "你是居家照護助理。這是跌倒偵測系統確認跌倒後的現場截圖。請描述:"
    "(1) 人物姿態與位置 (2) 周遭環境是否有危險物 (3) 可見的受傷跡象 "
    "(4) 嚴重程度 1-5 分與理由。100 字內,供家人快速判讀。"
)

FALLBACK_TEXT = "(VLM 描述暫缺)"


def _build_model():
    from google.genai.types import HarmBlockThreshold, HarmCategory
    from langchain.chat_models import init_chat_model

    from .config import settings

    return init_chat_model(
        settings.gemini_model,
        model_provider="google_genai",
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )


def describe_scene(image_path: Path) -> str:
    """回傳現場描述文字;任何失敗都回傳 FALLBACK_TEXT,絕不拋例外。"""
    try:
        from langchain_core.messages import HumanMessage

        model = _build_model()
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": PROMPT},
                {"type": "image", "base64": image_b64, "mime_type": "image/jpeg"},
            ]
        )
        response = model.invoke([message])
        # response.content 可能是純字串,也可能是多段 content block 的 list(視 provider/版本而定);
        # `.text` 是 langchain_core 提供的正規化屬性,一律回傳串接後的純文字,不用自己判斷型別。
        text = (response.text or "").strip()
        return text if text else FALLBACK_TEXT
    except Exception:  # noqa: BLE001 - VLM 是增強功能,任何失敗都不可中斷通報鏈
        return FALLBACK_TEXT
