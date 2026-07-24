"""YOLO26-pose 推論包裝(docs/PLAN.md D2/§8.4)。居家單人場景假設,`max_det=1`(§1 非目標)。

`resolve_pose_weights()` 與 `extract_video_pose()` 由 scripts/prepare_data.py(URFD)與
scripts/prepare_le2i.py(Le2i,docs/PLAN2.md Phase 7)共用,避免離線批次抽取邏輯各自維護
一份重複程式碼而silently飄移(呼應 D18/D20 的教訓:train/eval 用不同套邏輯會飄移)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import REPO_ROOT

POSE_WEIGHTS_CACHE_DIR = REPO_ROOT / "models" / "pretrained"


def resolve_pose_weights(name: str) -> str:
    """回傳權重本機路徑;首次執行觸發 ultralytics 自動下載(落在 CWD),下載後搬進
    models/pretrained/(已 gitignore)避免權重散落在 repo 根目錄。"""
    POSE_WEIGHTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = POSE_WEIGHTS_CACHE_DIR / name
    if cached.exists():
        return str(cached)

    from ultralytics import YOLO

    YOLO(name)  # 觸發下載到 CWD
    downloaded = REPO_ROOT / name
    if downloaded.exists():
        downloaded.rename(cached)
        return str(cached)
    return name


def extract_video_pose(model, video_path: Path, persist: bool = False) -> dict:
    """對整支影片檔跑 pose 追蹤,回傳逐幀陣列;不含任何標籤資訊——標籤語意因資料集而異
    (URFD 用逐幀 CSV、Le2i 用 fall 起訖幀標註),由呼叫端在這份陣列之上自行套用。

    回傳 dict 含 xyn(T,17,2)/conf(T,17)/bbox_xywh(T,4)/track_id(T,)/fps(float)/
    timestamps(T,),T 為總幀數。追蹤參數(quantize/max_det/tracker)與 detect.py 的
    即時推論刻意保持同一套,避免離線抽取與線上推論的行為不一致——但 `persist` 是例外:
    這裡預設 False,因為批次抽取(prepare_data.py/prepare_le2i.py)在同一個 model 物件上
    對「一批互不相關的影片」逐支呼叫,不是單一連續串流,`persist=True` 會讓 ByteTrack 的
    frame_id 計數跨影片不歸零,導致每支影片(除了批次裡第一支)第一次出現的人在第一幀
    可能因 `STrack.is_activated` 只在 `frame_id==1` 才立即為真而被整幀濾掉(見 D50)。
    detect.py 的即時推論走的是真正的單一連續串流,`PoseEstimator.infer()` 仍用 `persist=True`。
    """
    import cv2

    xyn_list: list[np.ndarray] = []
    conf_list: list[np.ndarray] = []
    bbox_list: list[np.ndarray] = []
    track_ids: list[int] = []

    results = model.track(
        str(video_path),
        stream=True,
        device=0,
        quantize=16,
        max_det=1,
        tracker="bytetrack.yaml",
        persist=persist,
        verbose=False,
    )
    for r in results:
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
    cap = cv2.VideoCapture(str(video_path))
    real_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    T = len(xyn_list)
    return {
        "xyn": np.stack(xyn_list).astype(np.float32),
        "conf": np.stack(conf_list).astype(np.float32),
        "bbox_xywh": np.stack(bbox_list).astype(np.float32),
        "track_id": np.array(track_ids, dtype=np.int32),
        "fps": np.float32(real_fps),
        "timestamps": (np.arange(T, dtype=np.float32) / real_fps),
    }


@dataclass
class PoseFrame:
    xyn: np.ndarray  # (17,2) 正規化座標,無偵測時全 NaN
    conf: np.ndarray  # (17,)
    bbox_xywh: np.ndarray  # (4,)
    detected: bool


class PoseEstimator:
    """包裝 `model.track()`,逐幀餵入 BGR frame,回傳單人關鍵點結果。

    `persist=True` 讓 ByteTrack 在連續呼叫間維持追蹤狀態(官方即時推論標準寫法);
    `max_det=1` 已限制單人,故不需額外挑「最大 bbox」的 sticky 邏輯。
    """

    def __init__(self, weights_name: str, device: int | str = 0):
        from ultralytics import YOLO

        self.model = YOLO(resolve_pose_weights(weights_name))
        self.device = device

    def infer(self, frame: np.ndarray) -> tuple[PoseFrame, np.ndarray]:
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
            quantize=16,
            max_det=1,
            verbose=False,
        )
        r = results[0]
        annotated = r.plot()

        if r.keypoints is not None and len(r.keypoints) > 0 and r.keypoints.conf is not None:
            pose = PoseFrame(
                xyn=r.keypoints.xyn[0].cpu().numpy(),
                conf=r.keypoints.conf[0].cpu().numpy(),
                bbox_xywh=r.boxes.xywh[0].cpu().numpy(),
                detected=True,
            )
        else:
            pose = PoseFrame(
                xyn=np.full((17, 2), np.nan, dtype=np.float32),
                conf=np.full((17,), np.nan, dtype=np.float32),
                bbox_xywh=np.full((4,), np.nan, dtype=np.float32),
                detected=False,
            )
        return pose, annotated
