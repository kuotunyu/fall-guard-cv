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
