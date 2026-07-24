"""規則式視窗分類器(docs/PLAN.md §8.1 的閾值語意,套用在單一視窗的無狀態判斷)。

用途:視窗級 precision/recall/F1/PR-AUC 評估(§7.2)。與 fsm.py 共用相同的預設閾值,
差別是這裡判斷「這個 1.5 秒視窗本身看起來像不像跌倒觸發+落地」,不像 fsm.py
是跨視窗持續追蹤狀態(事件級評估用 fsm.py,不是這裡)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fsm import FSMConfig
from .features import FrameFeatures, Window

# window_score() 的「整個視窗完全沒偵測到人」哨兵值。語意上是「這個視窗絕不可能是跌倒」,
# 理論上該用 -inf,但 sklearn 的 average_precision_score/precision_recall_curve 不接受
# 非有限值的分數陣列會直接拋例外。改用夠大的有限負數:分類判斷(> 0)結果不變,PR-AUC
# 計算也不會再炸。URFD 的 70 支影片從未真的觸發這個分支(否則早就在既有評估中炸過),
# 是 Phase 7 導入 Le2i(偵測條件較嚴苛,部分視窗整段無偵測)才第一次踩到,見 docs/PLAN.md D46。
NO_DETECTION_SCORE = -1e6


@dataclass
class RuleThresholds:
    v_y_threshold: float = 2.0
    omega_threshold: float = 120.0
    theta_threshold: float = 60.0
    rho_threshold: float = 1.0
    hip_height_threshold: float = 0.5

    @classmethod
    def literature_default(cls) -> "RuleThresholds":
        cfg = FSMConfig()
        return cls(
            v_y_threshold=cfg.v_y_threshold,
            omega_threshold=cfg.omega_threshold,
            theta_threshold=cfg.on_ground_theta_threshold,
            rho_threshold=cfg.on_ground_rho_threshold,
            hip_height_threshold=cfg.on_ground_hip_height_threshold,
        )


def window_arrays(features: FrameFeatures, window: Window) -> dict[str, np.ndarray]:
    sl = slice(window.start_idx, window.end_idx)
    return {
        "v_y": features.v_y[sl],
        "omega": features.omega[sl],
        "theta": features.theta[sl],
        "rho": features.rho[sl],
        "hip_height": features.hip_height[sl],
    }


def window_score(arrays: dict[str, np.ndarray], thresholds: RuleThresholds) -> float:
    """連續分數(非機率):觸發後最大躺姿餘裕(margin),供 PR-AUC 掃描閾值使用。

    規則本身是二元判斷,這裡取「觸發點之後,躺姿三條件中最保守的餘裕值」當連續分數,
    分數越高代表越像跌倒,threshold=0 對應 classify_window 的二元判斷邊界。
    """
    v_y, omega = arrays["v_y"], arrays["omega"]
    theta, rho, hip_height = arrays["theta"], arrays["rho"], arrays["hip_height"]

    valid = ~(np.isnan(v_y) | np.isnan(omega) | np.isnan(theta) | np.isnan(rho) | np.isnan(hip_height))
    if not valid.any():
        return NO_DETECTION_SCORE

    trigger_mask = valid & ((v_y > thresholds.v_y_threshold) | (np.abs(omega) > thresholds.omega_threshold))
    if not trigger_mask.any():
        # 沒觸發:分數為「離觸發閾值還差多少」的負值(越負代表越不像)
        return float(np.nanmax(v_y[valid]) - thresholds.v_y_threshold) if valid.any() else float("-inf")

    trigger_idx = int(np.argmax(trigger_mask))
    margin_theta = theta[trigger_idx:] - thresholds.theta_threshold
    margin_rho = rho[trigger_idx:] - thresholds.rho_threshold
    margin_hip = thresholds.hip_height_threshold - hip_height[trigger_idx:]  # 越低於門檻分數越高
    with np.errstate(all="ignore"):
        combined = np.minimum(np.minimum(margin_theta, margin_rho), margin_hip)
        combined = combined[~np.isnan(combined)]
    if len(combined) == 0:
        return 0.0
    return float(np.nanmax(combined))


def classify_window(arrays: dict[str, np.ndarray], thresholds: RuleThresholds) -> int:
    return 1 if window_score(arrays, thresholds) > 0 else 0
