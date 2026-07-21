"""YOLO26-pose 即時推論包裝(docs/PLAN.md D2/§8.4)。居家單人場景假設,`max_det=1`(§1 非目標)。

與 scripts/prepare_data.py 共用「權重快取到 models/pretrained/」邏輯,避免離線批次抽取
與線上即時推論各自維護一份重複程式碼。
"""

from __future__ import annotations

from dataclasses import dataclass

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
