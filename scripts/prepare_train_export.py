"""把 data/processed/*.npz 的關鍵點特徵轉成視窗統計特徵,打包成一個小檔案供上傳 Colab 訓練用
(docs/PLAN.md Phase 3)。輸出只有數字(54 維特徵 × 標籤 × video_id),遠比原始關鍵點小,
不含任何影像,上傳無隱私疑慮。

用法：
    uv run python scripts/prepare_train_export.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import VideoData, load_all_videos, window_ground_truth  # noqa: E402

from fallguard.config import REPO_ROOT  # noqa: E402
from fallguard.features import STAT_FEATURE_NAMES, make_windows, window_stat_vector  # noqa: E402

OUT_PATH = REPO_ROOT / "data" / "export" / "xgb_windows.npz"


def build_dataset(videos: dict[str, VideoData]) -> dict[str, np.ndarray]:
    X_list, y_list, video_id_list, start_t_list = [], [], [], []
    for vid, video in videos.items():
        for w in make_windows(video.features):
            gt = window_ground_truth(video, w.start_t, w.end_t)
            if gt is None:
                continue
            vec = window_stat_vector(video.features, w)
            if np.all(vec == 0):  # 整段缺失,無有效資料
                continue
            X_list.append(vec)
            y_list.append(gt)
            video_id_list.append(vid)
            start_t_list.append(w.start_t)

    return {
        "X": np.stack(X_list).astype(np.float32),
        "y": np.array(y_list, dtype=np.int8),
        "video_id": np.array(video_id_list),
        "start_t": np.array(start_t_list, dtype=np.float32),
        "feature_names": np.array(STAT_FEATURE_NAMES),
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    print("載入影片與特徵...")
    videos = load_all_videos()
    print("轉換為視窗統計特徵...")
    dataset = build_dataset(videos)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, **dataset)

    n_pos = int(dataset["y"].sum())
    n_total = len(dataset["y"])
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"已寫入 {OUT_PATH}({size_kb:.0f} KB)")
    print(f"視窗總數:{n_total}(正例 {n_pos}, {n_pos/n_total:.1%})；特徵維度:{dataset['X'].shape[1]}")
    print("上傳 Colab 時一併帶上 data/splits.json 與 data/urfd_meta.csv(皆已在 git 版控中,體積很小)。")


if __name__ == "__main__":
    main()
