"""features.py 單元測試:用合成關鍵點驗證已知值(docs/PLAN.md Phase 2 DoD)。"""

from __future__ import annotations

import numpy as np
import pytest

from fallguard import features as feat


def _make_rigid_pose(torso_len: float, theta_deg: float, cx: float = 0.5, cy: float = 0.5) -> np.ndarray:
    """建一個 17 點姿態:肩、髖依 theta_deg 決定的軀幹角排列,其餘點放合理位置,conf 皆設 1.0。

    theta_deg=0 → 肩在髖正上方(站姿);theta_deg=90 → 肩髖同高、左右錯開(躺姿)。
    """
    kp = np.zeros((17, 2), dtype=np.float32)
    theta = np.radians(theta_deg)
    half = torso_len / 2
    dx, dy = half * np.sin(theta), half * np.cos(theta)
    shoulder_mid = np.array([cx, cy - dy])
    hip_mid = np.array([cx, cy + dy])
    # 站姿時 dx=0,肩髖同一 x;躺姿時 dy=0,肩髖同一 y、左右錯開 dx
    shoulder_mid = np.array([cx - dx, cy - dy])
    hip_mid = np.array([cx + dx, cy + dy])

    kp[feat.LEFT_SHOULDER] = shoulder_mid + [0.05, 0]
    kp[feat.RIGHT_SHOULDER] = shoulder_mid - [0.05, 0]
    kp[feat.LEFT_HIP] = hip_mid + [0.03, 0]
    kp[feat.RIGHT_HIP] = hip_mid - [0.03, 0]
    kp[feat.NOSE] = shoulder_mid - [0, 0.1]
    kp[feat.LEFT_ANKLE] = hip_mid + [0.02, 0.4]
    kp[feat.RIGHT_ANKLE] = hip_mid - [0.02, 0.4]
    # 其餘點(眼耳肘腕膝)放在肩髖之間,避免全 0 造成 y_std 之類特徵異常
    for idx in [feat.LEFT_EYE, feat.RIGHT_EYE, feat.LEFT_EAR, feat.RIGHT_EAR]:
        kp[idx] = shoulder_mid - [0, 0.08]
    for idx in [feat.LEFT_ELBOW, feat.RIGHT_ELBOW, feat.LEFT_WRIST, feat.RIGHT_WRIST]:
        kp[idx] = (shoulder_mid + hip_mid) / 2
    for idx in [feat.LEFT_KNEE, feat.RIGHT_KNEE]:
        kp[idx] = (hip_mid + kp[feat.LEFT_ANKLE]) / 2
    return kp


def _make_sequence(n_frames: int, fps: float, pose_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """pose_fn(t) -> (17,2) keypoints。回傳 xyn, conf(全 1), bbox(依 pose 外接框), timestamps。"""
    timestamps = np.arange(n_frames) / fps
    xyn = np.stack([pose_fn(t) for t in timestamps]).astype(np.float32)
    conf = np.ones((n_frames, 17), dtype=np.float32)
    xmin, ymin = xyn[:, :, 0].min(axis=1), xyn[:, :, 1].min(axis=1)
    xmax, ymax = xyn[:, :, 0].max(axis=1), xyn[:, :, 1].max(axis=1)
    w, h = xmax - xmin, ymax - ymin
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    bbox = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
    return xyn, conf, bbox, timestamps


def test_standing_torso_angle_near_zero():
    xyn, conf, bbox, ts = _make_sequence(150, 30.0, lambda t: _make_rigid_pose(0.2, theta_deg=0.0))
    ff = feat.compute_features(xyn, conf, bbox, ts)
    mid = len(ff.theta) // 2
    assert ff.theta[mid] == pytest.approx(0.0, abs=3.0)


def test_lying_torso_angle_near_90():
    xyn, conf, bbox, ts = _make_sequence(150, 30.0, lambda t: _make_rigid_pose(0.2, theta_deg=90.0))
    ff = feat.compute_features(xyn, conf, bbox, ts)
    mid = len(ff.theta) // 2
    assert ff.theta[mid] == pytest.approx(90.0, abs=3.0)


def test_constant_descent_gives_expected_v_y():
    """勻速下降:整個剛體以固定速度往下平移,torso 長度不變,v_y(torso/s) 應約等於 (像素速度/torso長)。"""
    torso_len = 0.2
    rate = 0.1  # 正規化座標/秒
    n_frames, fps = 150, 30.0  # 5 秒,足夠 s(t) 的 3 秒滑動中位數穩定

    def pose_fn(t: float) -> np.ndarray:
        base = _make_rigid_pose(torso_len, theta_deg=0.0, cy=0.3)
        base[:, 1] += rate * t
        return base

    xyn, conf, bbox, ts = _make_sequence(n_frames, fps, pose_fn)
    ff = feat.compute_features(xyn, conf, bbox, ts)

    idx_4s = int(np.searchsorted(ff.timestamps, 4.0))
    expected_v_y = rate / torso_len
    assert ff.v_y[idx_4s] == pytest.approx(expected_v_y, rel=0.15)


def test_hip_height_distinguishes_crouch_vs_lying():
    """蹲下(髖仍高於腳踝一段距離)hip_height 應明顯大於躺地(髖踝同高,趨近 0)。"""

    def crouch_pose(t: float) -> np.ndarray:
        kp = np.zeros((17, 2), dtype=np.float32)
        kp[feat.LEFT_SHOULDER] = [0.45, 0.35]
        kp[feat.RIGHT_SHOULDER] = [0.55, 0.35]
        kp[feat.LEFT_HIP] = [0.47, 0.55]
        kp[feat.RIGHT_HIP] = [0.53, 0.55]
        kp[feat.LEFT_ANKLE] = [0.47, 0.75]  # 髖踝差距大 → hip_height 大
        kp[feat.RIGHT_ANKLE] = [0.53, 0.75]
        kp[feat.NOSE] = [0.5, 0.25]
        for idx in [feat.LEFT_EYE, feat.RIGHT_EYE, feat.LEFT_EAR, feat.RIGHT_EAR]:
            kp[idx] = [0.5, 0.27]
        for idx in [feat.LEFT_ELBOW, feat.RIGHT_ELBOW, feat.LEFT_WRIST, feat.RIGHT_WRIST]:
            kp[idx] = [0.5, 0.45]
        for idx in [feat.LEFT_KNEE, feat.RIGHT_KNEE]:
            kp[idx] = [0.5, 0.65]
        return kp

    def lying_pose(t: float) -> np.ndarray:
        kp = np.zeros((17, 2), dtype=np.float32)
        kp[feat.LEFT_SHOULDER] = [0.3, 0.55]
        kp[feat.RIGHT_SHOULDER] = [0.3, 0.45]
        kp[feat.LEFT_HIP] = [0.5, 0.55]
        kp[feat.RIGHT_HIP] = [0.5, 0.45]
        kp[feat.LEFT_ANKLE] = [0.7, 0.55]  # 髖踝幾乎同高 → hip_height 趨近 0
        kp[feat.RIGHT_ANKLE] = [0.7, 0.45]
        kp[feat.NOSE] = [0.2, 0.5]
        for idx in [feat.LEFT_EYE, feat.RIGHT_EYE, feat.LEFT_EAR, feat.RIGHT_EAR]:
            kp[idx] = [0.22, 0.5]
        for idx in [feat.LEFT_ELBOW, feat.RIGHT_ELBOW, feat.LEFT_WRIST, feat.RIGHT_WRIST]:
            kp[idx] = [0.4, 0.5]
        for idx in [feat.LEFT_KNEE, feat.RIGHT_KNEE]:
            kp[idx] = [0.6, 0.5]
        return kp

    xyn_c, conf_c, bbox_c, ts_c = _make_sequence(120, 30.0, crouch_pose)
    xyn_l, conf_l, bbox_l, ts_l = _make_sequence(120, 30.0, lying_pose)
    ff_c = feat.compute_features(xyn_c, conf_c, bbox_c, ts_c)
    ff_l = feat.compute_features(xyn_l, conf_l, bbox_l, ts_l)

    mid_c, mid_l = len(ff_c.hip_height) // 2, len(ff_l.hip_height) // 2
    assert ff_c.hip_height[mid_c] > 0.3
    assert abs(ff_l.hip_height[mid_l]) < 0.2
    assert ff_c.hip_height[mid_c] > ff_l.hip_height[mid_l]


def test_missing_keypoint_short_gap_is_interpolated():
    """<=0.3s 的關鍵點缺失(conf 掉到 0)應被內插補上,不應整段變 NaN。"""
    torso_len = 0.2
    n_frames, fps = 150, 30.0

    xyn, conf, bbox, ts = _make_sequence(n_frames, fps, lambda t: _make_rigid_pose(torso_len, theta_deg=0.0))
    # 在 t=2.0~2.2s(6 幀,0.2s < 0.3s 上限)把肩膀關鍵點信心壓到 0,模擬短暫遮擋
    gap_start, gap_end = int(2.0 * fps), int(2.2 * fps)
    conf[gap_start:gap_end, feat.LEFT_SHOULDER] = 0.0
    conf[gap_start:gap_end, feat.RIGHT_SHOULDER] = 0.0

    ff = feat.compute_features(xyn, conf, bbox, ts)
    idx_in_gap = int(np.searchsorted(ff.timestamps, 2.1))
    assert not np.isnan(ff.theta[idx_in_gap]), "0.2s 短缺口應被內插,不應仍是 NaN"


def test_missing_keypoint_long_gap_stays_nan():
    """>0.3s 的缺失不內插,維持 NaN(供訓練端剔除該視窗)。"""
    torso_len = 0.2
    n_frames, fps = 150, 30.0

    xyn, conf, bbox, ts = _make_sequence(n_frames, fps, lambda t: _make_rigid_pose(torso_len, theta_deg=0.0))
    gap_start, gap_end = int(2.0 * fps), int(2.6 * fps)  # 0.6s,超過 0.3s 上限
    conf[gap_start:gap_end, feat.LEFT_SHOULDER] = 0.0
    conf[gap_start:gap_end, feat.RIGHT_SHOULDER] = 0.0
    conf[gap_start:gap_end, feat.LEFT_HIP] = 0.0
    conf[gap_start:gap_end, feat.RIGHT_HIP] = 0.0

    ff = feat.compute_features(xyn, conf, bbox, ts)
    idx_in_gap = int(np.searchsorted(ff.timestamps, 2.3))
    assert np.isnan(ff.theta[idx_in_gap]), "0.6s 長缺口不該被內插填補"


def test_make_windows_covers_expected_count():
    xyn, conf, bbox, ts = _make_sequence(150, 30.0, lambda t: _make_rigid_pose(0.2, theta_deg=0.0))
    ff = feat.compute_features(xyn, conf, bbox, ts)
    windows = feat.make_windows(ff, window_s=1.5, stride_s=0.2)
    assert len(windows) > 0
    for w in windows:
        assert w.end_idx - w.start_idx == round(1.5 * feat.TARGET_HZ)
        assert w.end_t > w.start_t
