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


def _build_model(model: str | None = None, provider: str | None = None):
    """預設(不傳參數)= 現行 Gemini 行為,呼叫端(detect.py)零改動。

    `model`/`provider` 是 Phase 6(docs/PLAN2.md,VLM 描述品質對照)加的可選參數,
    讓 scripts/compare_vlm.py 能建立 OPENAI_MODEL 的模型做對照。safety_settings 是
    google_genai 專屬參數,只在走預設 Gemini 路徑時才帶入,OpenAI 路徑不適用。
    """
    from langchain.chat_models import init_chat_model

    from .config import settings

    if model is None and provider is None:
        from google.genai.types import HarmBlockThreshold, HarmCategory

        return init_chat_model(
            settings.gemini_model,
            model_provider="google_genai",
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
    return init_chat_model(model, model_provider=provider)


def _describe_scene_raw(image_path: Path, model: str | None = None, provider: str | None = None) -> str:
    """實際呼叫模型的核心邏輯,失敗時直接拋出例外(不吞例外)。

    `describe_scene()` 包一層 try/except 把這裡的例外轉成 FALLBACK_TEXT,是生產路徑
    (detect.py→notify.py)用的安全版本。scripts/compare_vlm.py(Phase 6,docs/PLAN2.md)
    需要看到真正的失敗原因(被安全過濾 vs 網路錯誤 vs 其他),故直接呼叫這個函式、
    自己接例外——單一實作來源,不重造一份呼叫邏輯(呼應 D18/D20 的教訓)。
    """
    from langchain_core.messages import HumanMessage

    # 零參數呼叫時保持跟改動前完全一樣的呼叫方式(_build_model()),
    # 讓既有測試的 monkeypatch(替換成 0 參數 lambda)不受影響。
    model_obj = _build_model() if model is None and provider is None else _build_model(model, provider)
    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": PROMPT},
            {"type": "image", "base64": image_b64, "mime_type": "image/jpeg"},
        ]
    )
    response = model_obj.invoke([message])
    # response.content 可能是純字串,也可能是多段 content block 的 list(視 provider/版本而定);
    # `.text` 是 langchain_core 提供的正規化屬性,一律回傳串接後的純文字,不用自己判斷型別。
    text = (response.text or "").strip()
    return text if text else FALLBACK_TEXT


def describe_scene(image_path: Path, model: str | None = None, provider: str | None = None) -> str:
    """回傳現場描述文字;任何失敗都回傳 FALLBACK_TEXT,絕不拋例外。

    `model`/`provider` 不傳時走現行 Gemini 行為;Phase 6 的 compare_vlm.py 用這兩個
    參數切到 OPENAI_MODEL 做對照(見 docs/PLAN2.md)。
    """
    try:
        return _describe_scene_raw(image_path, model, provider)
    except Exception:  # noqa: BLE001 - VLM 是增強功能,任何失敗都不可中斷通報鏈
        return FALLBACK_TEXT
