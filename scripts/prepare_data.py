"""URFD 影片 → 關鍵點序列 .npz(本機 GPU 抽取)。

依據 docs/PLAN.md D2(YOLO26-pose)/ D12(ADL label 語意) / Phase 1 DoD。

每支影片輸出 data/processed/{video_id}.npz,欄位：
    xyn         (T,17,2) float32   正規化關鍵點座標;未偵測到人的幀為 NaN
    conf        (T,17)   float32   每個關鍵點信心度;未偵測為 NaN
    bbox_xywh   (T,4)    float32   人物 bbox(中心點+寬高);未偵測為 NaN
    track_id    (T,)     int32     ByteTrack 追蹤 ID;未偵測為 -1
    raw_label   (T,)     int8      URFD 官方逐幀標籤原始值(-1/0/1);CSV 無該幀時為 -128(sentinel)
    label_present (T,)   bool      這一幀是否有 CSV 標籤覆蓋(True/False,對應上面 sentinel)
    fps         scalar   float32
    timestamps  (T,)     float32   frame_idx / fps(秒)
    video_id    scalar   str       例如 "fall-01"
    kind        scalar   str       "fall" 或 "adl"

*** 重要(D12)***：raw_label 是 URFD 官方 CSV 的「姿態是否水平」幾何特徵,
不是「是否發生跌倒事件」的語意標籤。ADL 影片定義上不含跌倒事件,即使
raw_label==1(躺姿),也絕不代表跌倒——這些是「已躺床」的困難負樣本
(誤報分析的關鍵素材)。下游程式(Phase 2 features.py / evaluate.py)必須
以 kind=="fall" 且 raw_label==1 才視為正例；kind=="adl" 一律是負例,
不論 raw_label 為何。CSV 對齊方式:frame_num(CSV,1-indexed) - 1 = 影片幀索引
(已用 fall-01/adl-01 實測驗證,精確對齊、非模糊 ±1 容忍)。

用法：
    uv run python scripts/prepare_data.py                  # 抽取全部 70 支
    uv run python scripts/prepare_data.py --limit 3         # 只抽前 3 支(快速測試)
    uv run python scripts/prepare_data.py --force            # 已存在的 npz 也重新抽取
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
URFD_DIR = REPO_ROOT / "data" / "raw" / "urfd"
OUT_DIR = REPO_ROOT / "data" / "processed"

FALL_COUNT = 30
ADL_COUNT = 40
LABEL_SENTINEL = -128  # raw_label 找不到對應 CSV 幀時的填值(域外值,{-1,0,1} 之外)
POSE_MODEL_NAME = "yolo26m-pose.pt"
WEIGHTS_CACHE_DIR = REPO_ROOT / "models" / "pretrained"


def resolve_pose_weights(name: str) -> str:
    """回傳權重本機路徑;首次執行會觸發 ultralytics 自動下載(落在 CWD),下載後搬進
    models/pretrained/(已 gitignore)避免權重散落在 repo 根目錄。"""
    WEIGHTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = WEIGHTS_CACHE_DIR / name
    if cached.exists():
        return str(cached)

    from ultralytics import YOLO

    YOLO(name)  # 觸發下載到 CWD
    downloaded = REPO_ROOT / name
    if downloaded.exists():
        downloaded.rename(cached)
        return str(cached)
    return name  # 找不到就讓呼叫端照原樣處理(ultralytics 會再嘗試一次)


def list_videos() -> list[tuple[str, str]]:
    videos = [(f"fall-{i:02d}", "fall") for i in range(1, FALL_COUNT + 1)]
    videos += [(f"adl-{i:02d}", "adl") for i in range(1, ADL_COUNT + 1)]
    return videos


def load_labels(kind: str) -> dict[str, dict[int, int]]:
    """讀 urfall-cam0-{falls,adls}.csv → {video_id: {frame_num(1-indexed): label}}。"""
    csv_path = URFD_DIR / f"urfall-cam0-{kind}s.csv"
    per_video: dict[str, dict[int, int]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            vid, frame_num, label = row[0], int(row[1]), int(row[2])
            per_video.setdefault(vid, {})[frame_num] = label
    return per_video


def extract_one(model, video_id: str, kind: str, labels: dict[int, int]) -> dict:
    path = URFD_DIR / f"{video_id}-cam0.mp4"

    xyn_list: list[np.ndarray] = []
    conf_list: list[np.ndarray] = []
    bbox_list: list[np.ndarray] = []
    track_ids: list[int] = []

    results = model.track(
        str(path),
        stream=True,
        device=0,
        quantize=16,
        max_det=1,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    )
    fps = None
    for r in results:
        if fps is None:
            fps = float(r.speed.get("fps", 0)) or None  # 保底,下面用 orig video fps 覆蓋

        if r.keypoints is not None and len(r.keypoints) > 0:
            xyn_list.append(r.keypoints.xyn[0].cpu().numpy())
            conf_list.append(r.keypoints.conf[0].cpu().numpy())
            bbox_list.append(r.boxes.xywh[0].cpu().numpy())
            tid = int(r.boxes.id[0].item()) if r.boxes.id is not None else -1
            track_ids.append(tid)
        else:
            xyn_list.append(np.full((17, 2), np.nan, dtype=np.float32))
            conf_list.append(np.full((17,), np.nan, dtype=np.float32))
            bbox_list.append(np.full((4,), np.nan, dtype=np.float32))
            track_ids.append(-1)

    # 用 cv2 直接讀原始 fps(比 track 結果的 speed 欄位可靠)
    import cv2

    cap = cv2.VideoCapture(str(path))
    real_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    T = len(xyn_list)
    raw_label = np.full((T,), LABEL_SENTINEL, dtype=np.int8)
    label_present = np.zeros((T,), dtype=bool)
    for i in range(T):
        frame_num = i + 1  # CSV 為 1-indexed(已實測驗證 fall-01/adl-01)
        if frame_num in labels:
            raw_label[i] = labels[frame_num]
            label_present[i] = True

    return {
        "xyn": np.stack(xyn_list).astype(np.float32),
        "conf": np.stack(conf_list).astype(np.float32),
        "bbox_xywh": np.stack(bbox_list).astype(np.float32),
        "track_id": np.array(track_ids, dtype=np.int32),
        "raw_label": raw_label,
        "label_present": label_present,
        "fps": np.float32(real_fps),
        "timestamps": (np.arange(T, dtype=np.float32) / real_fps),
        "video_id": video_id,
        "kind": kind,
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="只處理前 N 支影片(測試用)")
    parser.add_argument("--force", action="store_true", help="已存在的 npz 也重新抽取")
    args = parser.parse_args()

    if not URFD_DIR.exists():
        print(f"找不到 {URFD_DIR},請先執行 scripts/download_data.py")
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(resolve_pose_weights(POSE_MODEL_NAME))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fall_labels = load_labels("fall")
    adl_labels = load_labels("adl")

    videos = list_videos()
    if args.limit:
        videos = videos[: args.limit]

    ok = 0
    fail: list[str] = []
    low_coverage: list[str] = []
    detection_rates: list[float] = []
    t0 = time.time()

    for i, (vid, kind) in enumerate(videos, start=1):
        out_path = OUT_DIR / f"{vid}.npz"
        if out_path.exists() and not args.force:
            print(f"[{i}/{len(videos)}] {vid} 已存在,略過(--force 可重抽)")
            ok += 1
            continue

        labels = fall_labels.get(vid, {}) if kind == "fall" else adl_labels.get(vid, {})
        if not labels:
            print(f"[{i}/{len(videos)}] {vid} 找不到對應標籤,略過")
            fail.append(vid)
            continue

        try:
            data = extract_one(model, vid, kind, labels)
        except Exception as exc:  # noqa: BLE001 - 單支失敗不中斷整批
            print(f"[{i}/{len(videos)}] {vid} 抽取失敗:{exc}")
            fail.append(vid)
            continue

        np.savez(out_path, **data)

        det_rate = float(np.mean(~np.isnan(data["xyn"][:, 0, 0]))) * 100
        cov_rate = float(np.mean(data["label_present"])) * 100
        detection_rates.append(det_rate)
        if cov_rate < 50:
            low_coverage.append(f"{vid}(覆蓋率 {cov_rate:.0f}%)")

        print(f"[{i}/{len(videos)}] {vid} OK  幀數={len(data['xyn'])}  偵測率={det_rate:.1f}%  標籤覆蓋率={cov_rate:.1f}%")
        ok += 1

    elapsed = time.time() - t0
    print()
    print("=== 關鍵點抽取摘要 ===")
    print(f"成功:{ok}/{len(videos)}  失敗:{len(fail)}  耗時:{elapsed:.0f}s")
    if detection_rates:
        print(f"平均偵測率:{np.mean(detection_rates):.1f}%(最低 {np.min(detection_rates):.1f}%)")
    if low_coverage:
        print(f"標籤覆蓋率偏低(<50%,建議人工檢查)：{', '.join(low_coverage)}")
    if fail:
        print(f"失敗清單:{', '.join(fail)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
