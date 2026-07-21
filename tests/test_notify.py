"""notify.py 單元測試:mock Discord webhook,涵蓋成功送出/附圖/429 重送/未設定 webhook(docs/PLAN.md §8.3)。"""

from __future__ import annotations

import types

import fallguard.notify as notify


class _FakeResponse:
    def __init__(self, status_code, json_body=None, text=""):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.text = text

    def json(self):
        return self._json_body


def _fake_settings(discord_webhook_url=None, send_image=True):
    return types.SimpleNamespace(discord_webhook_url=discord_webhook_url, send_image=send_image)


def test_send_fall_alert_no_webhook_url_returns_false(monkeypatch):
    monkeypatch.setattr(notify, "settings", _fake_settings())
    ok = notify.send_fall_alert("測試描述", image_path=None, webhook_url="")
    assert ok is False


def test_send_fall_alert_success_text_only(monkeypatch):
    monkeypatch.setattr(notify, "settings", _fake_settings())
    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs)
        return _FakeResponse(204)

    monkeypatch.setattr(notify.requests, "post", fake_post)
    ok = notify.send_fall_alert("測試描述", image_path=None, webhook_url="https://example.invalid/webhook")
    assert ok is True
    assert len(calls) == 1
    assert "files" not in calls[0]


def test_send_fall_alert_with_image_attaches_file(tmp_path, monkeypatch):
    img = tmp_path / "snap.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")

    monkeypatch.setattr(notify, "settings", _fake_settings(send_image=True))
    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs)
        return _FakeResponse(200)

    monkeypatch.setattr(notify.requests, "post", fake_post)
    ok = notify.send_fall_alert("測試描述", image_path=img, webhook_url="https://example.invalid/webhook")
    assert ok is True
    assert "files" in calls[0]


def test_send_fall_alert_send_image_false_sends_text_only(tmp_path, monkeypatch):
    img = tmp_path / "snap.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")

    monkeypatch.setattr(notify, "settings", _fake_settings(send_image=False))
    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs)
        return _FakeResponse(200)

    monkeypatch.setattr(notify.requests, "post", fake_post)
    ok = notify.send_fall_alert("測試描述", image_path=img, webhook_url="https://example.invalid/webhook")
    assert ok is True
    assert "files" not in calls[0]


def test_send_fall_alert_429_then_success_retries_once(monkeypatch):
    monkeypatch.setattr(notify, "settings", _fake_settings())
    responses = [
        _FakeResponse(429, json_body={"retry_after": 0.01}),
        _FakeResponse(204),
    ]
    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr(notify.requests, "post", fake_post)
    ok = notify.send_fall_alert("測試描述", image_path=None, webhook_url="https://example.invalid/webhook")
    assert ok is True
    assert len(calls) == 2


def test_send_fall_alert_429_exhausts_retry_returns_false(monkeypatch):
    monkeypatch.setattr(notify, "settings", _fake_settings())

    def fake_post(url, **kwargs):
        return _FakeResponse(429, json_body={"retry_after": 0.01})

    monkeypatch.setattr(notify.requests, "post", fake_post)
    ok = notify.send_fall_alert("測試描述", image_path=None, webhook_url="https://example.invalid/webhook")
    assert ok is False
