"""Discord webhook 跌倒通報(docs/PLAN.md D5/§8.3)。

multipart 送出 embed + 附圖(`SEND_IMAGE=false` 時只送文字);429 依回應 body 的
`retry_after` 秒數等待後重送一次。**告警送達是安全關鍵**:任何無法重試的錯誤
一律印出原因、回傳 False,不拋例外——呼叫端(detect.py 的 alert worker)靠這個
回傳值記錄失敗,但主偵測迴圈不會被中斷。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .config import settings

EMBED_COLOR_ALERT = 0xE74C3C  # 紅
TIMEOUT_S = 10


def _build_payload(description: str, escalation: bool, attach_image: bool) -> dict:
    embed = {
        "title": "偵測到跌倒(升級再告警)" if escalation else "偵測到跌倒",
        "description": description,
        "color": EMBED_COLOR_ALERT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if attach_image:
        embed["image"] = {"url": "attachment://snapshot.jpg"}
    return {"embeds": [embed]}


def send_fall_alert(
    description: str,
    image_path: Path | None = None,
    escalation: bool = False,
    webhook_url: str | None = None,
) -> bool:
    """送出跌倒通報。`image_path` 為 None 或 `SEND_IMAGE=false` 時只送文字 embed(D8)。"""
    url = webhook_url or settings.discord_webhook_url
    if not url:
        print("[notify] DISCORD_WEBHOOK_URL 未設定,略過送出(請至 .env 補上)")
        return False

    attach = image_path is not None and settings.send_image and Path(image_path).exists()
    payload = _build_payload(description, escalation, attach)
    return _post_with_retry(url, payload, Path(image_path) if attach else None)


def _post_with_retry(url: str, payload: dict, image_path: Path | None, max_retries: int = 1) -> bool:
    for attempt in range(max_retries + 1):
        fh = None
        try:
            data = {"payload_json": json.dumps(payload)}
            if image_path is not None:
                fh = image_path.open("rb")
                resp = requests.post(url, data=data, files={"files[0]": ("snapshot.jpg", fh, "image/jpeg")}, timeout=TIMEOUT_S)
            else:
                resp = requests.post(url, data=data, timeout=TIMEOUT_S)
        finally:
            if fh:
                fh.close()

        if resp.status_code in (200, 204):
            return True

        if resp.status_code == 429 and attempt < max_retries:
            retry_after = 1.0
            try:
                retry_after = float(resp.json().get("retry_after", 1.0))
            except Exception:  # noqa: BLE001
                pass
            print(f"[notify] 429 rate limited,{retry_after:.1f}s 後重送")
            time.sleep(retry_after)
            continue

        print(f"[notify] Discord 送出失敗:HTTP {resp.status_code} {resp.text[:200]}")
        return False

    return False
