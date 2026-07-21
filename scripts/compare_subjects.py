"""兩位受試者的近距離放大比較圖:用 prepare_data.py 已算好的人物框(bbox)裁切放大,
比 subject_sheet.py 的整間房間縮圖更看得清楚臉型/身形細節,適合「這兩人很像分不出來」時用。

用法：
    uv run python scripts/compare_subjects.py P1 P2            # 比較 P1 vs P2
    uv run python scripts/compare_subjects.py P1 P2 --samples 4
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
URFD_DIR = REPO_ROOT / "data" / "raw" / "urfd"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
META_PATH = REPO_ROOT / "data" / "urfd_meta.csv"
OUT_PATH = REPO_ROOT / "docs" / "assets" / "subject_compare.png"

CELL_W, CELL_H = 260, 340
PAD_RATIO = 0.35  # bbox 四周留白比例,避免裁太緊看不到全身輪廓
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


def load_subject_videos() -> dict[str, list[str]]:
    by_subject: dict[str, list[str]] = {}
    with META_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            subj = (row.get("subject_id") or "").strip()
            if subj and subj != "unknown":
                by_subject.setdefault(subj, []).append(row["video_id"])
    return by_subject


def cropped_person(video_id: str) -> np.ndarray | None:
    """從 npz 找一個「站著、有偵測到人」的幀,用該幀的 bbox 去原始影片裁切放大。"""
    npz_path = PROCESSED_DIR / f"{video_id}.npz"
    video_path = URFD_DIR / f"{video_id}-cam0.mp4"
    if not npz_path.exists() or not video_path.exists():
        return None

    with np.load(npz_path) as d:
        bbox = d["bbox_xywh"]
        raw_label = d["raw_label"]
        valid = ~np.isnan(bbox[:, 0])
        standing = valid & (raw_label == -1)  # 未跌倒/未躺下的幀,人通常還站著
        candidates = np.where(standing)[0]
        if len(candidates) == 0:
            candidates = np.where(valid)[0]
        if len(candidates) == 0:
            return None
        idx = int(candidates[len(candidates) // 3])  # 取偏前段但不要第一幀
        cx, cy, w, h = bbox[idx]

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    fh, fw = frame.shape[:2]
    pad_w, pad_h = w * PAD_RATIO, h * PAD_RATIO
    x0 = max(0, int(cx - w / 2 - pad_w))
    y0 = max(0, int(cy - h / 2 - pad_h))
    x1 = min(fw, int(cx + w / 2 + pad_w))
    y1 = min(fh, int(cy + h / 2 + pad_h))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subjects", nargs=2, help="兩個要比較的受試者代號,例如 P1 P2")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    by_subject = load_subject_videos()
    for s in args.subjects:
        if s not in by_subject:
            print(f"找不到已標註的 {s}")
            return

    label_h = 40
    row_h = label_h + CELL_H
    n_cols = args.samples
    canvas = Image.new("RGB", (CELL_W * n_cols, row_h * 2), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    f_label = font(24)

    for r, subj in enumerate(args.subjects):
        vids = by_subject[subj][: args.samples]
        draw.text((8, r * row_h + 6), f"{subj}", font=f_label, fill=(255, 220, 120))
        for c, vid in enumerate(vids):
            crop = cropped_person(vid)
            thumb = fit_thumb(crop, CELL_W, CELL_H) if crop is not None else np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            canvas.paste(Image.fromarray(thumb_rgb), (c * CELL_W, r * row_h + label_h))
            draw.text((c * CELL_W + 6, r * row_h + label_h + CELL_H - 24), vid, font=font(16), fill=(200, 255, 200))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PATH)
    print(f"已產出 {OUT_PATH}")

    if not args.no_open:
        try:
            os.startfile(str(OUT_PATH))
        except Exception as exc:  # noqa: BLE001
            print(f"自動開啟失敗({exc}),請手動開啟 {OUT_PATH}")


if __name__ == "__main__":
    main()
