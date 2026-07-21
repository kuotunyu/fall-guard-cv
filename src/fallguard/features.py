"""跌倒偵測特徵工程(docs/PLAN.md §7.3)。

輸入為 scripts/prepare_data.py 產出的原始關鍵點序列(xyn/conf/bbox_xywh/timestamps),
輸出為距離無關、fps 無關的逐幀特徵,供規則式狀態機(fsm.py)與視窗分類(rules.py)使用。

管線順序：逐幀基礎特徵(用真實 timestamp)→ 短缺口內插 → 重採樣到固定 25Hz 網格
→ 導數類特徵(在等間隔網格上計算,結果等同真實時間差分)。

規格中「dρ/dt 同前平滑法」一句有歧義,本檔採用與 v_y 相同的做法
(差分 + 5 幀移動平均),而非 Savitzky-Golay(該法規格明文只用於 ω)。
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from scipy.signal import savgol_filter

# COCO 17 關鍵點索引(docs/PLAN.md §3 對照表)
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

CONF_THRESH = 0.5
TORSO_SCALE_WINDOW_S = 3.0  # s(t) 滑動中位數視窗
MAX_GAP_INTERP_S = 0.3  # 缺口內插上限
TARGET_HZ = 25.0
V_DIFF_S = 0.2  # v_y / dρ/dt 差分間隔
SMOOTH_WINDOW_FRAMES = 5  # v_y / dρ/dt 的移動平均視窗(以重採樣後的幀數計)
SAVGOL_WINDOW = 7  # ω 的 Savitzky-Golay 平滑視窗(幀數,需為奇數)


@dataclass
class FrameFeatures:
    """逐幀特徵(重採樣到 25Hz 網格之後)。每個陣列 shape=(T,)。"""

    timestamps: np.ndarray
    theta: np.ndarray  # 軀幹角(度),站≈0、躺≈90
    omega: np.ndarray  # 軀幹角速度(度/秒)
    v_y: np.ndarray  # 質心垂直速度(torso/秒,向下為正)
    rho: np.ndarray  # bbox 長寬比
    drho: np.ndarray  # dρ/dt
    head_ankle_diff: np.ndarray  # (y_ankle_max - y_nose) / s(t)
    hip_height: np.ndarray  # (y_ankle_mean - y_hip_mid) / s(t)
    y_std: np.ndarray  # std(y, conf>=0.5) / s(t)
    missing_rate: np.ndarray  # 該幀 17 點中 conf<0.5 的比例
    torso_missing: np.ndarray  # bool:肩與髖是否都缺失(FSM 凍結判斷用)
    s_t: np.ndarray  # 人身尺度(軀幹長滑動中位數),debug/測試用

    def __len__(self) -> int:
        return len(self.timestamps)

    def frame(self, i: int) -> dict:
        """取第 i 幀的純量字典,方便 fsm.py 逐幀處理。"""
        return {f.name: getattr(self, f.name)[i] for f in fields(self) if f.name != "timestamps"} | {
            "t": self.timestamps[i]
        }


def _midpoint_with_fallback(xyn: np.ndarray, conf: np.ndarray, idx_a: int, idx_b: int) -> np.ndarray:
    """兩點中點,任一點 conf<0.5 時退回只用另一點(至少一肩+一髖原則)。"""
    T = xyn.shape[0]
    out = np.full((T, 2), np.nan, dtype=np.float32)
    ok_a = conf[:, idx_a] >= CONF_THRESH
    ok_b = conf[:, idx_b] >= CONF_THRESH
    both = ok_a & ok_b
    only_a = ok_a & ~ok_b
    only_b = ok_b & ~ok_a
    out[both] = (xyn[both, idx_a] + xyn[both, idx_b]) / 2
    out[only_a] = xyn[only_a, idx_a]
    out[only_b] = xyn[only_b, idx_b]
    return out


def _weighted_centroid(xyn: np.ndarray, conf: np.ndarray, hip_mid: np.ndarray) -> np.ndarray:
    """conf>=0.5 關鍵點的信心加權質心;整幀都缺失時退回 hip_mid。"""
    T = xyn.shape[0]
    mask = conf >= CONF_THRESH  # (T,17)
    w = np.where(mask, conf, 0.0)
    w_sum = w.sum(axis=1)  # (T,)
    cy = np.full(T, np.nan, dtype=np.float32)
    has_any = w_sum > 0
    cy[has_any] = (xyn[has_any, :, 1] * w[has_any]).sum(axis=1) / w_sum[has_any]
    cy[~has_any] = hip_mid[~has_any, 1]
    return cy


def _interpolate_short_gaps(arr: np.ndarray, timestamps: np.ndarray, max_gap_s: float = MAX_GAP_INTERP_S) -> np.ndarray:
    """線性內插長度 <= max_gap_s 的 NaN 缺口;更長的缺口維持 NaN。"""
    arr = arr.copy()
    isnan = np.isnan(arr)
    if not isnan.any() or isnan.all():
        return arr

    valid_idx = np.flatnonzero(~isnan)
    # 逐段找連續 NaN 區間
    nan_runs = []
    start = None
    for i in range(len(arr)):
        if isnan[i] and start is None:
            start = i
        elif not isnan[i] and start is not None:
            nan_runs.append((start, i - 1))
            start = None
    if start is not None:
        nan_runs.append((start, len(arr) - 1))

    for lo, hi in nan_runs:
        if lo == 0 or hi == len(arr) - 1:
            continue  # 頭尾缺口不內插(缺前後端點)
        gap_s = timestamps[hi] - timestamps[lo] + (timestamps[lo] - timestamps[lo - 1])
        if gap_s > max_gap_s:
            continue
        arr[lo : hi + 1] = np.interp(timestamps[lo : hi + 1], [timestamps[lo - 1], timestamps[hi + 1]], [arr[lo - 1], arr[hi + 1]])
    return arr


def _rolling_median(arr: np.ndarray, timestamps: np.ndarray, window_s: float) -> np.ndarray:
    """以「過去 window_s 秒」為視窗的滑動中位數(不對稱、只看過去,避免用到未來資訊)。"""
    out = np.full_like(arr, np.nan, dtype=np.float64)
    valid = ~np.isnan(arr)
    for i in range(len(arr)):
        lo_t = timestamps[i] - window_s
        lo = np.searchsorted(timestamps, lo_t, side="left")
        window_vals = arr[lo : i + 1][valid[lo : i + 1]]
        if len(window_vals) > 0:
            out[i] = np.median(window_vals)
    return out


def _diff_with_smoothing(arr: np.ndarray, timestamps: np.ndarray, smooth_frames: int = SMOOTH_WINDOW_FRAMES) -> np.ndarray:
    """差分(用實際 Δt)後做移動平均去抖,對應 v_y / dρ/dt 的規格寫法。"""
    d = np.full_like(arr, np.nan, dtype=np.float64)
    dt = np.diff(timestamps, prepend=timestamps[0] - (timestamps[1] - timestamps[0] if len(timestamps) > 1 else 1))
    dt[dt <= 0] = np.nan
    raw = np.empty_like(arr)
    raw[0] = np.nan
    raw[1:] = (arr[1:] - arr[:-1]) / dt[1:]
    if smooth_frames > 1:
        kernel = np.ones(smooth_frames) / smooth_frames
        pad = smooth_frames // 2
        padded = np.pad(raw, (pad, pad), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")[: len(raw)]
        return smoothed
    return raw


def _smooth_only(arr: np.ndarray, smooth_frames: int = SMOOTH_WINDOW_FRAMES) -> np.ndarray:
    """純移動平均(不含差分),NaN 值先用鄰近有效值填補邊界避免整段被抹成 NaN。"""
    if smooth_frames <= 1:
        return arr
    kernel = np.ones(smooth_frames) / smooth_frames
    pad = smooth_frames // 2
    valid = ~np.isnan(arr)
    filled = np.where(valid, arr, 0.0)
    padded = np.pad(filled, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")[: len(arr)]
    # 視窗內全 NaN 的位置維持 NaN,其餘照常輸出(移動平均對零散 NaN 的容忍度足夠此用途)
    out = smoothed
    out[~valid] = np.nan
    return out


def compute_frame_features(
    xyn: np.ndarray,
    conf: np.ndarray,
    bbox_xywh: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, np.ndarray]:
    """逐幀基礎特徵(原始 fps,尚未重採樣)。"""
    shoulder_mid = _midpoint_with_fallback(xyn, conf, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_mid = _midpoint_with_fallback(xyn, conf, LEFT_HIP, RIGHT_HIP)

    torso_len = np.linalg.norm(hip_mid - shoulder_mid, axis=1)
    torso_len[np.isnan(shoulder_mid[:, 0]) | np.isnan(hip_mid[:, 0])] = np.nan
    s_t = _rolling_median(torso_len, timestamps, TORSO_SCALE_WINDOW_S)

    v = hip_mid - shoulder_mid
    theta = np.degrees(np.arctan2(np.abs(v[:, 0]), v[:, 1]))

    torso_missing = np.isnan(shoulder_mid[:, 0]) & np.isnan(hip_mid[:, 0])

    cy = _weighted_centroid(xyn, conf, hip_mid)

    ankle_ok = conf[:, [LEFT_ANKLE, RIGHT_ANKLE]] >= CONF_THRESH
    ankle_y = xyn[:, [LEFT_ANKLE, RIGHT_ANKLE], 1]
    ankle_y_masked = np.where(ankle_ok, ankle_y, np.nan)
    with np.errstate(all="ignore"):
        ankle_y_max = np.nanmax(ankle_y_masked, axis=1)
        ankle_y_mean = np.nanmean(ankle_y_masked, axis=1)

    nose_y = np.where(conf[:, NOSE] >= CONF_THRESH, xyn[:, NOSE, 1], np.nan)
    head_ankle_diff = (ankle_y_max - nose_y) / s_t
    hip_height = (ankle_y_mean - hip_mid[:, 1]) / s_t

    mask = conf >= CONF_THRESH
    y_all = np.where(mask, xyn[:, :, 1], np.nan)
    with np.errstate(all="ignore"):
        y_std = np.nanstd(y_all, axis=1) / s_t

    missing_rate = 1.0 - mask.mean(axis=1)

    w = bbox_xywh[:, 2]
    h = bbox_xywh[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(h > 0, w / h, np.nan)

    return {
        "theta": theta,
        "v_y_raw": cy,  # 尚未取差分,後續 resample 後再做 Δt=0.2s 差分
        "rho": rho,
        "head_ankle_diff": head_ankle_diff,
        "hip_height": hip_height,
        "y_std": y_std,
        "missing_rate": missing_rate,
        "torso_missing": torso_missing.astype(np.float64),  # 先轉 float 方便內插/重採樣,FSM 端再轉回 bool
        "s_t": s_t,
    }


def resample_to_grid(
    raw: dict[str, np.ndarray], timestamps: np.ndarray, target_hz: float = TARGET_HZ
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """內插短缺口後,把逐幀特徵重採樣到固定 target_hz 網格(fps 無關化,防洩漏)。"""
    grid_t = np.arange(timestamps[0], timestamps[-1], 1.0 / target_hz)
    if len(grid_t) == 0:
        grid_t = timestamps[:1]

    out: dict[str, np.ndarray] = {}
    for key, arr in raw.items():
        filled = _interpolate_short_gaps(arr.astype(np.float64), timestamps)
        out[key] = np.interp(grid_t, timestamps, filled, left=np.nan, right=np.nan)
        # np.interp 對 NaN 來源值不會自動傳播 NaN(它會把 NaN 當成普通浮點數內插出怪值),
        # 需手動把「兩端最近的原始有效點都太遠」的網格點標回 NaN。
        still_nan_src = np.isnan(filled)
        if still_nan_src.any():
            nan_at_grid = np.interp(grid_t, timestamps, still_nan_src.astype(np.float64)) > 0
            out[key][nan_at_grid] = np.nan
    return out, grid_t


def compute_features(xyn: np.ndarray, conf: np.ndarray, bbox_xywh: np.ndarray, timestamps: np.ndarray) -> FrameFeatures:
    """對外主入口:原始關鍵點序列 → 重採樣後的完整逐幀特徵。"""
    raw = compute_frame_features(xyn, conf, bbox_xywh, timestamps)
    grid, grid_t = resample_to_grid(raw, timestamps)

    theta = grid["theta"]
    if len(theta) >= SAVGOL_WINDOW:
        valid = ~np.isnan(theta)
        theta_filled = np.where(valid, theta, np.nanmean(theta) if valid.any() else 0.0)
        theta_smooth = savgol_filter(theta_filled, SAVGOL_WINDOW, polyorder=2)
        theta_smooth = np.where(valid, theta_smooth, np.nan)
    else:
        theta_smooth = theta
    omega = _diff_with_smoothing(theta_smooth, grid_t, smooth_frames=1)  # ω 本身已由 SG 平滑,差分不再二次平滑

    # v_y 規格單位為「torso/s」(§7.3 開頭:速度類特徵一律以 s(t) 為單位),
    # 故先算原始像素座標差分,除以當下 s(t) 正規化後才做 5 幀移動平均去抖。
    v_y_pixel_rate = _diff_with_smoothing(grid["v_y_raw"], grid_t, smooth_frames=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_y_normalized = np.where(grid["s_t"] > 0, v_y_pixel_rate / grid["s_t"], np.nan)
    v_y = _smooth_only(v_y_normalized, smooth_frames=SMOOTH_WINDOW_FRAMES)

    drho = _diff_with_smoothing(grid["rho"], grid_t)

    torso_missing = grid["torso_missing"] > 0.5  # 重採樣內插後可能變成 0~1 之間,>0.5 視為缺失

    return FrameFeatures(
        timestamps=grid_t,
        theta=theta,
        omega=omega,
        v_y=v_y,
        rho=grid["rho"],
        drho=drho,
        head_ankle_diff=grid["head_ankle_diff"],
        hip_height=grid["hip_height"],
        y_std=grid["y_std"],
        missing_rate=grid["missing_rate"],
        torso_missing=torso_missing,
        s_t=grid["s_t"],
    )


@dataclass
class Window:
    start_idx: int
    end_idx: int  # 不含(exclusive)
    start_t: float
    end_t: float


def make_windows(features: FrameFeatures, window_s: float = 1.5, stride_s: float = 0.2) -> list[Window]:
    """在重採樣後的 25Hz 網格上切滑動視窗(訓練/評估用;推論時改用 ring buffer,見 §8.4)。"""
    t = features.timestamps
    if len(t) == 0:
        return []
    dt = 1.0 / TARGET_HZ
    window_frames = max(1, round(window_s / dt))
    stride_frames = max(1, round(stride_s / dt))

    windows = []
    start = 0
    while start + window_frames <= len(t):
        end = start + window_frames
        windows.append(Window(start_idx=start, end_idx=end, start_t=t[start], end_t=t[end - 1]))
        start += stride_frames
    return windows


# Phase 3(XGBoost)用:每個視窗的統計聚合特徵,~9 基礎特徵 × 6 統計量 = 54 維(docs/PLAN.md 第 4 章)。
STAT_BASE_FEATURES = ["theta", "omega", "v_y", "rho", "drho", "head_ankle_diff", "hip_height", "y_std", "missing_rate"]
STAT_AGGREGATES = ["mean", "std", "min", "max", "last_minus_first", "max_abs_derivative"]
STAT_FEATURE_NAMES = [f"{base}_{agg}" for base in STAT_BASE_FEATURES for agg in STAT_AGGREGATES]


def window_stat_vector(features: FrameFeatures, window: Window) -> np.ndarray:
    """視窗內每個基礎特徵取 {mean,std,min,max,last-first,max|Δ|} 六個統計量,串成固定長度向量。
    全缺失(NaN)的基礎特徵該組六格填 0(missing_rate 這個特徵本身已經標記缺失狀態,不需要另外加 flag)。
    """
    sl = slice(window.start_idx, window.end_idx)
    out = np.zeros(len(STAT_FEATURE_NAMES), dtype=np.float32)
    for i, base in enumerate(STAT_BASE_FEATURES):
        arr = getattr(features, base)[sl]
        valid = arr[~np.isnan(arr)]
        off = i * len(STAT_AGGREGATES)
        if len(valid) == 0:
            continue
        out[off + 0] = np.mean(valid)
        out[off + 1] = np.std(valid) if len(valid) > 1 else 0.0
        out[off + 2] = np.min(valid)
        out[off + 3] = np.max(valid)
        first = arr[0] if not np.isnan(arr[0]) else valid[0]
        last = arr[-1] if not np.isnan(arr[-1]) else valid[-1]
        out[off + 4] = last - first
        out[off + 5] = np.max(np.abs(np.diff(valid))) if len(valid) > 1 else 0.0
    return out
