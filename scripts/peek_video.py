"""獨立播放指定影片,純觀看用,完全不影響 annotate_urfd.py 的標註進度/佇列位置。

用法：
    uv run python scripts/peek_video.py fall-01           # 播放 fall-01,迴圈播放
    uv run python scripts/peek_video.py fall-01 fall-07    # 依序播放多支(看完一支按任意鍵放下一支)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
URFD_DIR = REPO_ROOT / "data" / "raw" / "urfd"
WINDOW_NAME = "peek(純觀看,不影響標註進度)"
FONT_PATH = "C:/Windows/Fonts/msjh.ttc"


def font(size: int):
    if Path(FONT_PATH).exists():
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default()


def banner(width: int, text: str) -> np.ndarray:
    img = Image.new("RGB", (width, 40), (35, 35, 35))
    draw = ImageDraw.Draw(img)
    draw.text((8, 6), text, font=font(20), fill=(255, 220, 120))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def play_once(video_id: str) -> None:
    path = URFD_DIR / f"{video_id}-cam0.mp4"
    if not path.exists():
        print(f"找不到 {path}")
        return

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = max(1, int(1000 / fps))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    label = banner(720, f"{video_id}(迴圈播放,按任意鍵換下一支/結束)")

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        h, w = frame.shape[:2]
        scale = 720 / w
        resized = cv2.resize(frame, (720, int(h * scale)))
        cv2.imshow(WINDOW_NAME, np.vstack([label, resized]))
        if cv2.waitKey(delay) != -1:
            break
    cap.release()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_ids", nargs="+", help="要看的影片代號,例如 fall-01 fall-07")
    args = parser.parse_args()

    for vid in args.video_ids:
        play_once(vid)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
