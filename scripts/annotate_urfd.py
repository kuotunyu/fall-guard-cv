"""URFD 人工標註工具:互動式 GUI,標 70 段影片的 subject_id(+ ADL 動作類別)。

依據 docs/PLAN.md D6 / Phase 1 DoD:LOSO 評估協定需要受試者級標籤,
URFD 官方未提供 subject↔sequence 對照表,故人工看預覽片標註。

用法：
    uv run python scripts/annotate_urfd.py            # 標註模式:只跳過已標註的影片(可分次做完)
    uv run python scripts/annotate_urfd.py --review    # 複查模式(第二輪自我一致性檢查):重新過一遍全部 70 段,逐一確認/修改
    uv run python scripts/annotate_urfd.py --restart    # 忽略既有標註,從頭開始(較少用)

視窗內操作(畫面下方會顯示同一份說明)：
    1-5   標為受試者 P1-P5(URFD 官方稱共 5 人;不確定時可用 u)
    u     標為不確定(unknown,只會被排進訓練集,不進 LOSO 測試折)
    p     播放這段影片(正常速度;按任意鍵停止播放回到縮圖)
    b     回上一段(改標籤用)
    n     替這段加備註(僅限英數字,cv2 視窗無法輸入中文)
    q     儲存並離開(進度自動存檔,下次啟動會接著做)

輸出：data/urfd_meta.csv(欄位:video_id, kind, subject_id, action_category, note)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
URFD_DIR = REPO_ROOT / "data" / "raw" / "urfd"
META_PATH = REPO_ROOT / "data" / "urfd_meta.csv"

FALL_COUNT = 30
ADL_COUNT = 40
MAX_SUBJECTS = 9  # URFD 官方稱 5 人;多留幾格當安全閥

ACTION_CATEGORIES = ["走動", "坐下", "蹲下/綁鞋帶", "撿東西/彎腰", "躺床", "其他"]

THUMB_W, THUMB_H = 280, 210
GRID_COLS, GRID_ROWS = 3, 2
REF_THUMB = 96
CANVAS_W = THUMB_W * GRID_COLS
WINDOW_NAME = "URFD 標註工具"

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/mingliu.ttc",
    "C:/Windows/Fonts/simsun.ttc",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = _load_font(size)
    return _FONT_CACHE[size]


def text_block(width: int, lines: list[tuple[str, tuple[int, int, int]]], size: int = 18, pad: int = 8, bg=(24, 24, 24)) -> np.ndarray:
    """畫一塊左對齊多行文字的區塊(BGR numpy),cv2.putText 不支援中文,一律走 PIL。"""
    line_h = size + 10
    height = pad * 2 + line_h * max(len(lines), 1)
    img = Image.new("RGB", (width, height), (bg[2], bg[1], bg[0]))
    draw = ImageDraw.Draw(img)
    f = font(size)
    y = pad
    for text, color in lines:
        draw.text((pad, y), text, font=f, fill=(color[2], color[1], color[0]))
        y += line_h
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def fit_thumb(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def sample_frames(path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, max(total - 1, 0), n, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        frames.append(frame if ok else np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8))
    cap.release()
    return frames


def build_montage(path: Path) -> np.ndarray:
    frames = sample_frames(path, GRID_COLS * GRID_ROWS)
    if not frames:
        return np.zeros((THUMB_H * GRID_ROWS, THUMB_W * GRID_COLS, 3), dtype=np.uint8)
    thumbs = [fit_thumb(f, THUMB_W, THUMB_H) for f in frames]
    rows = [np.hstack(thumbs[r * GRID_COLS : (r + 1) * GRID_COLS]) for r in range(GRID_ROWS)]
    return np.vstack(rows)


def first_frame(path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


class MetaStore:
    """管理 data/urfd_meta.csv 的讀寫(每次標註後整檔重寫,支援回上一段修改)。"""

    FIELDS = ["video_id", "kind", "subject_id", "action_category", "note"]

    def __init__(self, videos: list[tuple[str, str]]):
        self.order = [v for v, _ in videos]
        self.kind_of = dict(videos)
        self.records: dict[str, dict[str, str]] = {
            vid: {"video_id": vid, "kind": self.kind_of[vid], "subject_id": "", "action_category": "", "note": ""}
            for vid in self.order
        }
        self.load()

    def load(self) -> None:
        if not META_PATH.exists():
            return
        with META_PATH.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                vid = row.get("video_id", "")
                if vid in self.records:
                    self.records[vid].update({k: row.get(k, "") for k in self.FIELDS})

    def save(self) -> None:
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        with META_PATH.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            for vid in self.order:
                writer.writerow(self.records[vid])

    def is_labeled(self, vid: str) -> bool:
        return bool(self.records[vid]["subject_id"])

    def set_subject(self, vid: str, subject: str) -> None:
        self.records[vid]["subject_id"] = subject
        self.save()

    def set_action(self, vid: str, action: str) -> None:
        self.records[vid]["action_category"] = action
        self.save()

    def set_note(self, vid: str, note: str) -> None:
        self.records[vid]["note"] = note
        self.save()

    def known_subjects(self) -> list[str]:
        seen = []
        for vid in self.order:
            s = self.records[vid]["subject_id"]
            if s and s != "unknown" and s not in seen:
                seen.append(s)
        return sorted(seen)

    def summary(self) -> str:
        total = len(self.order)
        labeled = sum(1 for v in self.order if self.is_labeled(v))
        subjects = self.known_subjects()
        counts = {s: sum(1 for v in self.order if self.records[v]["subject_id"] == s) for s in subjects}
        unk = sum(1 for v in self.order if self.records[v]["subject_id"] == "unknown")
        lines = [f"已標註 {labeled}/{total}", f"unknown: {unk}"]
        lines += [f"{s}: {counts[s]}" for s in subjects]
        return "  ".join(lines)


def list_videos() -> list[tuple[str, str]]:
    videos = [(f"fall-{i:02d}", "fall") for i in range(1, FALL_COUNT + 1)]
    videos += [(f"adl-{i:02d}", "adl") for i in range(1, ADL_COUNT + 1)]
    return videos


def video_path(video_id: str) -> Path:
    return URFD_DIR / f"{video_id}-cam0.mp4"


def build_reference_strip(store: MetaStore) -> np.ndarray | None:
    subjects = store.known_subjects()
    if not subjects:
        return None
    tiles = []
    for s in subjects:
        rep_vid = next(v for v in store.order if store.records[v]["subject_id"] == s)
        frame = first_frame(video_path(rep_vid))
        thumb = fit_thumb(frame, REF_THUMB, REF_THUMB) if frame is not None else np.zeros((REF_THUMB, REF_THUMB, 3), dtype=np.uint8)
        label = text_block(REF_THUMB, [(s, (255, 255, 255))], size=16, pad=4, bg=(50, 50, 50))
        tiles.append(np.vstack([thumb, label]))
    strip = np.hstack(tiles)
    if strip.shape[1] < CANVAS_W:
        pad = np.zeros((strip.shape[0], CANVAS_W - strip.shape[1], 3), dtype=np.uint8)
        strip = np.hstack([strip, pad])
    return strip[:, :CANVAS_W]


def compose_screen(montage: np.ndarray, ref_strip: np.ndarray | None, header_lines, footer_lines) -> np.ndarray:
    header = text_block(CANVAS_W, header_lines, size=20, bg=(35, 35, 60))
    footer = text_block(CANVAS_W, footer_lines, size=16, bg=(35, 35, 35))
    parts = [header]
    if ref_strip is not None:
        ref_header = text_block(CANVAS_W, [("已標註受試者參考(比對外觀用):", (200, 200, 200))], size=14, pad=4, bg=(20, 20, 20))
        parts += [ref_header, ref_strip]
    parts += [montage, footer]
    return np.vstack(parts)


def play_video(path: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = max(1, int(1000 / fps))
    banner = text_block(CANVAS_W, [("播放中... 按任意鍵停止並回到縮圖", (255, 220, 120))], size=18, bg=(35, 35, 35))
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        thumb = fit_thumb(frame, CANVAS_W, int(CANVAS_W * frame.shape[0] / frame.shape[1]))
        cv2.imshow(WINDOW_NAME, np.vstack([banner, thumb]))
        if cv2.waitKey(delay) != -1:
            break
    cap.release()


def capture_note(prompt: str) -> str:
    """簡易文字輸入(僅英數字/基本符號,cv2 視窗無法處理中文輸入法)。"""
    buf = ""
    while True:
        block = text_block(
            CANVAS_W,
            [(prompt, (255, 255, 255)), (f"> {buf}_", (150, 220, 150)), ("Enter 確認 / Esc 取消", (150, 150, 150))],
            size=18,
            bg=(20, 20, 40),
        )
        cv2.imshow(WINDOW_NAME, block)
        key = cv2.waitKey(0)
        if key in (13, 10):  # Enter
            return buf
        if key == 27:  # Esc
            return ""
        if key == 8:  # Backspace
            buf = buf[:-1]
        elif 32 <= key < 127:
            buf += chr(key)


def ask_action_category(video_id: str) -> str | None:
    lines = [(f"{i+1}. {name}", (255, 255, 255)) for i, name in enumerate(ACTION_CATEGORIES)]
    lines.append(("按 1-6 選動作類別 · s 跳過 · b 取消回上一步", (180, 180, 180)))
    block = text_block(CANVAS_W, [(f"{video_id}:這段 ADL 主要動作是?", (255, 220, 120))] + lines, size=18, bg=(20, 40, 20))
    cv2.imshow(WINDOW_NAME, block)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if ord("1") <= key <= ord("6"):
            return ACTION_CATEGORIES[key - ord("1")]
        if key in (ord("s"), ord("S")):
            return ""
        if key in (ord("b"), ord("B")):
            return None


def run(mode: str) -> None:
    videos = list_videos()
    store = MetaStore(videos)

    if mode == "review":
        queue = list(store.order)
    else:
        queue = [v for v in store.order if not store.is_labeled(v)]

    if not queue:
        print("所有影片都已標註。若要重新逐一確認(第二輪自我一致性檢查),請加 --review。")
        return

    print(f"待處理 {len(queue)} 段。視窗操作說明見腳本開頭 docstring,畫面下方也會顯示。")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    i = 0
    changed_in_review: list[str] = []
    while 0 <= i < len(queue):
        vid = queue[i]
        path = video_path(vid)
        if not path.exists():
            print(f"[警告] 找不到 {path.name},跳過(下載可能未完成)")
            i += 1
            continue

        montage = build_montage(path)
        ref_strip = build_reference_strip(store)
        current = store.records[vid]
        kind_label = "跌倒(fall)" if current["kind"] == "fall" else "日常活動(adl)"
        header = [
            (f"[{i+1}/{len(queue)}] {vid}  ({kind_label})", (255, 255, 255)),
            (f"目前標籤:subject={current['subject_id'] or '(未標)'}  action={current['action_category'] or '-'}  note={current['note'] or '-'}", (180, 220, 180)),
        ]
        footer = [
            ("1-5=標受試者P1-P5  u=不確定  p=播放影片  b=上一段  n=備註  q=存檔離開", (255, 255, 255)),
            (f"進度:{store.summary()}", (150, 200, 150)),
        ]
        screen = compose_screen(montage, ref_strip, header, footer)
        cv2.imshow(WINDOW_NAME, screen)
        cv2.setWindowTitle(WINDOW_NAME, f"{WINDOW_NAME} - {vid} ({i+1}/{len(queue)})")

        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("b"):
            i = max(0, i - 1)
            continue
        if key == ord("p"):
            play_video(path)
            continue
        if key == ord("n"):
            note = capture_note(f"{vid} 備註(英數字):")
            if note:
                store.set_note(vid, note)
            continue

        subject = None
        if key == ord("u"):
            subject = "unknown"
        elif ord("1") <= key <= ord("9") and (key - ord("0")) <= MAX_SUBJECTS:
            subject = f"P{chr(key)}"

        if subject is None:
            continue  # 未知按鍵,忽略

        prev = current["subject_id"]
        store.set_subject(vid, subject)
        if mode == "review" and prev and prev != subject:
            changed_in_review.append(f"{vid}: {prev} -> {subject}")

        if current["kind"] == "adl":
            action = ask_action_category(vid)
            if action is None:  # b:取消,不前進
                continue
            store.set_action(vid, action)

        i += 1

    store.save()
    print()
    print("=== 標註進度摘要 ===")
    print(store.summary())
    if mode == "review" and changed_in_review:
        print(f"複查時變更了 {len(changed_in_review)} 筆:")
        for line in changed_in_review:
            print(" ", line)
    print(f"已寫入 {META_PATH}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", action="store_true", help="複查模式:重新逐一確認全部影片(第二輪自我一致性檢查)")
    parser.add_argument("--restart", action="store_true", help="忽略既有 data/urfd_meta.csv,從頭標註")
    args = parser.parse_args()

    if args.restart and META_PATH.exists():
        META_PATH.unlink()

    if not URFD_DIR.exists():
        print(f"找不到 {URFD_DIR},請先執行 scripts/download_data.py")
        sys.exit(1)

    run("review" if args.review else "label")


if __name__ == "__main__":
    main()
