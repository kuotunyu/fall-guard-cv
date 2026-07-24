"""vlm.py 單元測試:VLM 失敗(安全過濾/例外/空回應)一律回傳 fallback 文字,不拋例外(docs/PLAN.md §8.2)。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

import fallguard.vlm as vlm


class _FakeModel:
    """回傳真正的 AIMessage(不是自製假物件),確保測試的是 langchain_core 實際的
    `.text` 正規化行為,而不是測試自己寫的替身邏輯。"""

    def __init__(self, content):
        self._content = content

    def invoke(self, messages):
        return AIMessage(content=self._content)


def _dummy_image(tmp_path):
    img = tmp_path / "snap.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
    return img


def test_describe_scene_returns_fallback_on_exception(tmp_path, monkeypatch):
    def _raise():
        raise RuntimeError("safety filter blocked / network error")

    monkeypatch.setattr(vlm, "_build_model", _raise)
    result = vlm.describe_scene(_dummy_image(tmp_path))
    assert result == vlm.FALLBACK_TEXT


def test_describe_scene_returns_model_text_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm, "_build_model", lambda: _FakeModel("有人躺在地上,嚴重程度 3 分"))
    result = vlm.describe_scene(_dummy_image(tmp_path))
    assert result == "有人躺在地上,嚴重程度 3 分"


def test_describe_scene_returns_fallback_on_empty_response(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm, "_build_model", lambda: _FakeModel(""))
    result = vlm.describe_scene(_dummy_image(tmp_path))
    assert result == vlm.FALLBACK_TEXT


def test_describe_scene_flattens_list_content_blocks_to_plain_text(tmp_path, monkeypatch):
    """迴歸測試:某些 provider/版本回傳的 content 是 list of content block(非純字串),
    早期實作直接 str(response.content) 會把整個 list/dict 結構原樣塞進通報文字裡
    (例如 "[{'type': 'text', 'text': '...', 'extras': {...}}]"),必須用 `.text` 正規化。"""
    monkeypatch.setattr(
        vlm,
        "_build_model",
        lambda: _FakeModel([{"type": "text", "text": "有人倒地,嚴重程度 3 分", "extras": {"signature": "abc"}}]),
    )
    result = vlm.describe_scene(_dummy_image(tmp_path))
    assert result == "有人倒地,嚴重程度 3 分"
    assert "{" not in result and "extras" not in result


def test_describe_scene_returns_fallback_on_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm, "_build_model", lambda: _FakeModel("不應該被呼叫到"))
    result = vlm.describe_scene(tmp_path / "does-not-exist.jpg")
    assert result == vlm.FALLBACK_TEXT


def test_describe_scene_forwards_model_and_provider_to_init_chat_model(tmp_path, monkeypatch):
    """Phase 6(docs/PLAN2.md,VLM 描述品質對照):describe_scene 帶 model/provider 參數時,
    要把它們原封不動轉給 init_chat_model,且不能帶 google_genai 專屬的 safety_settings
    (OpenAI 不支援這個參數)。攔截 init_chat_model 本身,不真的呼叫任何 API。"""
    captured = {}

    def _fake_init_chat_model(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return _FakeModel("測試用回應")

    monkeypatch.setattr("langchain.chat_models.init_chat_model", _fake_init_chat_model)
    result = vlm.describe_scene(_dummy_image(tmp_path), model="gpt-5-mini", provider="openai")

    assert captured["model"] == "gpt-5-mini"
    assert captured["kwargs"] == {"model_provider": "openai"}
    assert result == "測試用回應"


def test_describe_scene_default_call_unchanged(tmp_path, monkeypatch):
    """不傳 model/provider 時,_build_model 呼叫方式必須跟改動前完全一樣(零參數),
    確保 detect.py 的既有呼叫端(vlm.describe_scene(confirm_path))行為零改動。"""
    calls = []
    monkeypatch.setattr(vlm, "_build_model", lambda *a, **kw: calls.append((a, kw)) or _FakeModel("ok"))
    vlm.describe_scene(_dummy_image(tmp_path))
    assert calls == [((), {})]


def test_build_model_default_branch_uses_google_genai_safety_settings(monkeypatch):
    """涵蓋 `_build_model()` 零參數(預設)分支本身——這是 detect.py 實際會呼叫到的路徑,
    但先前所有零參數測試都直接 monkeypatch 掉 `_build_model` 整個函式,從未真正執行過
    這段組裝 google_genai safety_settings 的程式碼(收尾複查發現)。這裡改成攔截更底層的
    `langchain.chat_models.init_chat_model`,讓 `_build_model()` 本體真正跑一次。"""
    from google.genai.types import HarmBlockThreshold, HarmCategory

    from fallguard.config import settings

    captured = {}

    def _fake_init_chat_model(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "不重要,這裡只驗證傳入參數"

    monkeypatch.setattr("langchain.chat_models.init_chat_model", _fake_init_chat_model)

    vlm._build_model()

    assert captured["model"] == settings.gemini_model
    assert captured["kwargs"]["model_provider"] == "google_genai"
    safety_settings = captured["kwargs"]["safety_settings"]
    assert safety_settings[HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT] == HarmBlockThreshold.BLOCK_NONE
