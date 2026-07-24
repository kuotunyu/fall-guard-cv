"""scripts/evaluate.py 單元測試(收尾複查發現先前完全零測試覆蓋)。

涵蓋:window_ground_truth() 的 D12(ADL 覆寫)/D48(URFD vs Le2i 標籤語意)規則、
event_level_metrics() 的 D15(無 adl 測試樣本)分支、run_cross_evaluation() 缺
Le2i 資料目錄時的 exit path。用合成資料建構,不需要真實 npz 存在。
"""

from __future__ import annotations

import numpy as np
import pytest
from evaluate import VideoData, event_level_metrics, run_cross_evaluation, window_ground_truth
import evaluate as evaluate_module

from fallguard.features import FrameFeatures
from fallguard.fsm import FSMConfig


def _video(kind, raw_label, label_present, raw_timestamps, has_ambiguous_label=True, features=None):
    return VideoData(
        video_id="test-video",
        kind=kind,
        features=features,
        raw_label=np.array(raw_label, dtype=np.int8),
        label_present=np.array(label_present, dtype=bool),
        raw_timestamps=np.array(raw_timestamps, dtype=np.float32),
        has_ambiguous_label=has_ambiguous_label,
    )


def _standing_features(n=5):
    """全程站立、從不觸發跌倒判定的合成特徵,供 event_level_metrics 測試用。"""
    return FrameFeatures(
        timestamps=np.arange(n, dtype=np.float32) / 25.0,
        theta=np.zeros(n),
        omega=np.zeros(n),
        v_y=np.zeros(n),
        rho=np.full(n, 0.4),
        drho=np.zeros(n),
        head_ankle_diff=np.zeros(n),
        hip_height=np.full(n, 0.8),
        y_std=np.zeros(n),
        missing_rate=np.zeros(n),
        torso_missing=np.zeros(n, dtype=bool),
        s_t=np.ones(n),
    )


# ---------- window_ground_truth() ----------


def test_window_ground_truth_adl_video_always_negative_regardless_of_raw_label():
    """D12:kind!='fall' 一律負例,即使 raw_label 誤標成 1(躺姿幾何特徵非跌倒事件)。"""
    video = _video("adl", raw_label=[1, 1, 1], label_present=[True, True, True], raw_timestamps=[0.0, 0.1, 0.2])
    assert window_ground_truth(video, 0.0, 0.2) == 0


def test_window_ground_truth_fall_video_with_positive_label_is_positive():
    video = _video("fall", raw_label=[0, 1, 0], label_present=[True, True, True], raw_timestamps=[0.0, 0.1, 0.2])
    assert window_ground_truth(video, 0.0, 0.2) == 1


def test_window_ground_truth_urfd_all_zero_window_is_ambiguous_and_excluded():
    """URFD(has_ambiguous_label=True):只含 0,官方三值標籤裡屬模糊過渡帶,剔除(既有 D12 行為)。"""
    video = _video(
        "fall", raw_label=[0, 0, 0], label_present=[True, True, True], raw_timestamps=[0.0, 0.1, 0.2], has_ambiguous_label=True
    )
    assert window_ground_truth(video, 0.0, 0.2) is None


def test_window_ground_truth_le2i_all_zero_window_is_negative_not_excluded():
    """Le2i(has_ambiguous_label=False):只含 0 就是明確的跌倒區間外,算負例、不剔除(D48 修正)。"""
    video = _video(
        "fall", raw_label=[0, 0, 0], label_present=[True, True, True], raw_timestamps=[0.0, 0.1, 0.2], has_ambiguous_label=False
    )
    assert window_ground_truth(video, 0.0, 0.2) == 0


def test_window_ground_truth_no_label_present_in_range_is_excluded():
    video = _video("fall", raw_label=[0, 0, 0], label_present=[False, False, False], raw_timestamps=[0.0, 0.1, 0.2])
    assert window_ground_truth(video, 0.0, 0.2) is None


# ---------- event_level_metrics() ----------


def test_event_level_metrics_no_adl_samples_leaves_specificity_none():
    """D15:test 集沒有 adl 影片時,specificity/CI/FP-per-hour/adl_total_hours 留 None,
    不可誤算成 0 或拋除以零例外。"""
    video = _video("fall", raw_label=[0] * 5, label_present=[True] * 5, raw_timestamps=[0, 0.04, 0.08, 0.12, 0.16], features=_standing_features())
    videos = {"fall-only": video}
    result = event_level_metrics(["fall-only"], videos, FSMConfig())
    assert result["n_fall"] == 1
    assert result["n_adl"] == 0
    assert result["event_specificity"] is None
    assert result["event_specificity_ci"] is None
    assert result["false_alarms_per_hour"] is None
    assert result["adl_total_hours"] is None


def test_event_level_metrics_with_adl_samples_computes_specificity():
    fall = _video("fall", raw_label=[0] * 5, label_present=[True] * 5, raw_timestamps=[0, 0.04, 0.08, 0.12, 0.16], features=_standing_features())
    adl = _video("adl", raw_label=[0] * 5, label_present=[True] * 5, raw_timestamps=[0, 0.04, 0.08, 0.12, 0.16], features=_standing_features())
    videos = {"fall": fall, "adl": adl}
    result = event_level_metrics(["fall", "adl"], videos, FSMConfig())
    assert result["n_adl"] == 1
    assert result["event_specificity"] == 1.0  # 全程站立,adl 影片不會被誤判成跌倒
    assert result["event_specificity_ci"] is not None
    assert result["adl_total_hours"] is not None and result["adl_total_hours"] > 0


# ---------- run_cross_evaluation() ----------


def test_run_cross_evaluation_missing_le2i_dir_exits(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate_module, "LE2I_PROCESSED_DIR", tmp_path / "does-not-exist")
    with pytest.raises(SystemExit) as exc_info:
        run_cross_evaluation("rule")
    assert exc_info.value.code == 1


def test_run_cross_evaluation_rejects_xgb_model():
    with pytest.raises(SystemExit) as exc_info:
        run_cross_evaluation("xgb")
    assert exc_info.value.code == 1
