"""rules.py 單元測試:window_score/classify_window/RuleThresholds(docs/PLAN.md §8.1)。

先前完全沒有測試覆蓋(收尾複查發現)——NO_DETECTION_SCORE 的分支過去只有靠 Le2i 資料集
才第一次被觸發(D46),本檔用合成陣列直接測,不必依賴真實資料集存在。
"""

from __future__ import annotations

import numpy as np

from fallguard.fsm import FSMConfig
from fallguard.rules import NO_DETECTION_SCORE, RuleThresholds, classify_window, window_score

DEFAULT_TH = RuleThresholds(v_y_threshold=2.0, omega_threshold=120.0, theta_threshold=60.0, rho_threshold=1.0, hip_height_threshold=0.5)


def _all_nan_arrays(n=5):
    return {k: np.full(n, np.nan) for k in ("v_y", "omega", "theta", "rho", "hip_height")}


def test_window_score_no_detection_returns_sentinel():
    """整個視窗五個特徵全為 NaN(完全沒偵測到人)⇒ NO_DETECTION_SCORE,不是 -inf(D46)。"""
    score = window_score(_all_nan_arrays(), DEFAULT_TH)
    assert score == NO_DETECTION_SCORE
    assert np.isfinite(score)  # sklearn 的 PR-AUC 計算不接受非有限值,這是 D46 修復的核心前提


def test_window_score_untriggered_returns_negative_margin():
    """v_y/omega 全程未超過觸發閾值 ⇒ 分數為「離觸發閾值還差多少」的負值。"""
    arrays = {
        "v_y": np.array([0.5, 1.0, 1.5]),
        "omega": np.array([0.0, 0.0, 0.0]),
        "theta": np.array([10.0, 10.0, 10.0]),
        "rho": np.array([0.3, 0.3, 0.3]),
        "hip_height": np.array([0.8, 0.8, 0.8]),
    }
    score = window_score(arrays, DEFAULT_TH)
    assert score == 1.5 - 2.0  # max(v_y) - v_y_threshold
    assert classify_window(arrays, DEFAULT_TH) == 0


def test_window_score_triggered_takes_worst_case_margin_after_trigger():
    """觸發後(idx=1)兩幀的躺姿三條件margin取逐幀最小值,再取這些最小值裡的最大值。"""
    arrays = {
        "v_y": np.array([0.0, 3.0, 3.0]),
        "omega": np.array([0.0, 0.0, 0.0]),
        "theta": np.array([0.0, 70.0, 80.0]),
        "rho": np.array([0.0, 1.5, 1.2]),
        "hip_height": np.array([0.0, 0.3, 0.4]),
    }
    # idx1: min(70-60, 1.5-1.0, 0.5-0.3) = min(10, 0.5, 0.2) = 0.2
    # idx2: min(80-60, 1.2-1.0, 0.5-0.4) = min(20, 0.2, 0.1) = 0.1
    # nanmax(0.2, 0.1) = 0.2
    score = window_score(arrays, DEFAULT_TH)
    assert score == 0.2
    assert classify_window(arrays, DEFAULT_TH) == 1


def test_classify_window_boundary_is_strictly_greater_than_zero():
    """score 恰好等於 0 時分類仍為 0(未觸發),驗證邊界用的是嚴格 >,不是 >=。"""
    arrays = {
        "v_y": np.array([3.0]),
        "omega": np.array([0.0]),
        "theta": np.array([70.0]),
        "rho": np.array([1.5]),
        "hip_height": np.array([0.5]),  # margin_hip = 0.5 - 0.5 = 0,是三者中最小值
    }
    assert window_score(arrays, DEFAULT_TH) == 0.0
    assert classify_window(arrays, DEFAULT_TH) == 0


def test_literature_default_matches_fsm_config():
    """RuleThresholds.literature_default() 是手動抄一份 FSMConfig 對應欄位,防止未來漂移。"""
    cfg = FSMConfig()
    th = RuleThresholds.literature_default()
    assert th.v_y_threshold == cfg.v_y_threshold
    assert th.omega_threshold == cfg.omega_threshold
    assert th.theta_threshold == cfg.on_ground_theta_threshold
    assert th.rho_threshold == cfg.on_ground_rho_threshold
    assert th.hip_height_threshold == cfg.on_ground_hip_height_threshold
