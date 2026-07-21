"""從目前的 data/urfd_meta.csv 產出「受試者對照表」:P1-P5 各挑幾段影片的站立畫面
放大排在一起,供人工標註時另開視窗對照,比工具內建的小縮圖列更清楚。

用法：
    uv run python scripts/subject_sheet.py                 # 產出 docs/assets/subject_sheet.png 並用預設看圖軟體開啟
    uv run python scripts/subject_sheet.py --samples 4      # 每人取樣張數(預設 3)
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
URFD_DIR = REPO_ROOT / "data" / "raw" / "urfd"
META_PATH = REPO_ROOT / "data" / "urfd_meta.csv"
OUT_PATH = REPO_ROOT / "docs" / "assets" / "subject_sheet.png"

CELL_W, CELL_H = 240, 180
FONT_PATH = "C:/Windows/Fonts/msjh.ttc"


def font(size: int):
    if Path(FONT_PATH).exists():
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default()


def fit_thumb(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def standing_frame(video_id: str) -> np.ndarray | None:
    """抓影片約 15% 處的畫面,通常人還站著(跌倒多發生在中後段)。"""
    path = URFD_DIR / f"{video_id}-cam0.mp4"
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * 0.15)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def load_subject_videos() -> dict[str, list[str]]:
    by_subject: dict[str, list[str]] = {}
    with META_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            subj = (row.get("subject_id") or "").strip()
            if subj and subj != "unknown":
                by_subject.setdefault(subj, []).append(row["video_id"])
    return by_subject


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--no-open", action="store_true", help="只產出檔案,不自動開啟")
    args = parser.parse_args()

    if not META_PATH.exists():
        print(f"找不到 {META_PATH},請先開始標註(scripts/annotate_urfd.py)")
        sys.exit(1)

    by_subject = load_subject_videos()
    if not by_subject:
        print("目前還沒有任何已標註的受試者,先標幾段再跑本腳本")
        sys.exit(1)

    subjects = sorted(by_subject.keys())
    n_cols = max(len(v[: args.samples]) for v in by_subject.values())

    label_h = 40
    row_h = label_h + CELL_H
    canvas = Image.new("RGB", (CELL_W * n_cols, row_h * len(subjects)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    f_label = font(22)

    for r, subj in enumerate(subjects):
        vids = by_subject[subj][: args.samples]
        draw.text((8, r * row_h + 6), f"{subj}（{len(by_subject[subj])} 段已標）", font=f_label, fill=(255, 220, 120))
        for c, vid in enumerate(vids):
            frame = standing_frame(vid)
            thumb = fit_thumb(frame, CELL_W, CELL_H) if frame is not None else np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            canvas.paste(Image.fromarray(thumb_rgb), (c * CELL_W, r * row_h + label_h))
            draw.text((c * CELL_W + 6, r * row_h + label_h + CELL_H - 24), vid, font=font(16), fill=(200, 255, 200))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PATH)
    print(f"已產出 {OUT_PATH}（{len(subjects)} 位受試者 × 最多 {args.samples} 張樣本）")

    if not args.no_open:
        try:
            os.startfile(str(OUT_PATH))  # Windows 預設看圖軟體開啟
        except Exception as exc:  # noqa: BLE001
            print(f"自動開啟失敗({exc}),請手動開啟 {OUT_PATH}")


if __name__ == "__main__":
    main()
