"""規則式 baseline 評估(docs/PLAN.md §7.2 / Phase 2 DoD)。

視窗級:precision/recall/F1/PR-AUC + 混淆矩陣(文獻預設閾值 vs 折內調參後閾值)。
事件級:用 fsm.py 完整跑過每支測試影片,算 Event Sensitivity/Specificity/
       false alarms per hour/偵測延遲(演算法延遲、告警延遲分開報)。

`--protocol cross`(Phase 7,docs/PLAN2.md):URFD 全量訓練/調參 → Le2i 純測試,
只報事件級指標,兌現 docs/PLAN.md §7.1 P3 協定。

用法：
    uv run python scripts/evaluate.py --model rule --protocol loso
    uv run python scripts/evaluate.py --model rule --protocol groupkfold
    uv run python scripts/evaluate.py --model rule --protocol cross
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

from fallguard.config import REPO_ROOT
from fallguard.features import FrameFeatures, compute_features, make_windows, window_stat_vector
from fallguard.fsm import FallStateMachine, FSMConfig, State
from fallguard.rules import RuleThresholds, window_arrays, window_score
from fallguard.stats import wilson_interval

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
LE2I_PROCESSED_DIR = REPO_ROOT / "data" / "processed_le2i"  # Phase 7,docs/PLAN2.md;與 URFD 分開避免混撈
META_PATH = REPO_ROOT / "data" / "urfd_meta.csv"
SPLITS_PATH = REPO_ROOT / "data" / "splits.json"
RESULTS_DIR = REPO_ROOT / "docs" / "results"
XGB_MODELS_DIR = REPO_ROOT / "models" / "xgboost"

EVAL_CONFIRM_SECONDS = 2.0  # 評估用 N(D11);部署預設 N=10s 由 config.py 的 FALL_CONFIRM_SECONDS 另行套用

TUNE_V_Y_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
TUNE_THETA_GRID = [45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
TUNE_FALLING_TIMEOUT_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
# D16 實測發現：25/25 進到 ON_GROUND 的影片,進入後剩餘時長全部 < 2.0s(中位數僅 0.77s)——
# URFD 片段短 + 本管線判定「已躺平」偏晚,D11 原訂評估用 N=2s 對這批資料系統性過嚴,故 confirm_seconds
# 也需要當作可調參數搜尋(而非視為固定值),grid 選在遠低於 2.0s 的範圍。
TUNE_CONFIRM_SECONDS_GRID = [0.3, 0.5, 0.8, 1.0, 1.5]


# ---------- 資料載入 ----------


class VideoData:
    __slots__ = ("video_id", "kind", "features", "raw_label", "label_present", "raw_timestamps", "has_ambiguous_label")

    def __init__(self, video_id, kind, features, raw_label, label_present, raw_timestamps, has_ambiguous_label=True):
        self.video_id = video_id
        self.kind = kind
        self.features = features
        self.raw_label = raw_label
        self.label_present = label_present
        self.raw_timestamps = raw_timestamps
        # URFD 官方標籤是 -1/0/1 三值(0=躺姿判定的模糊過渡帶);Le2i 只有 0/1 兩值(0=明確的跌倒區間外)。
        # 兩者對「0」的語意不同,window_ground_truth() 靠這個旗標分辨要不要把「只含 0」的視窗當歧義剔除。見 D48。
        self.has_ambiguous_label = has_ambiguous_label


def load_all_videos(processed_dir: Path = PROCESSED_DIR, has_ambiguous_label: bool | None = None) -> dict[str, VideoData]:
    """預設讀 URFD 的 data/processed/;Phase 7 傳 LE2I_PROCESSED_DIR 讀 Le2i 的另一批 npz。

    has_ambiguous_label 未指定時,依 processed_dir 是否為 LE2I_PROCESSED_DIR 自動判斷(URFD=True、
    Le2i=False);必要時可手動覆寫。細節見 VideoData.has_ambiguous_label / window_ground_truth()。
    """
    if has_ambiguous_label is None:
        has_ambiguous_label = processed_dir != LE2I_PROCESSED_DIR
    out: dict[str, VideoData] = {}
    for npz_path in sorted(processed_dir.glob("*.npz")):
        with np.load(npz_path) as d:
            xyn, conf, bbox = d["xyn"], d["conf"], d["bbox_xywh"]
            raw_timestamps = d["timestamps"]
            raw_label, label_present = d["raw_label"], d["label_present"]
            kind = str(d["kind"])
            video_id = str(d["video_id"])
        features = compute_features(xyn, conf, bbox, raw_timestamps)
        out[video_id] = VideoData(video_id, kind, features, raw_label, label_present, raw_timestamps, has_ambiguous_label)
    return out


def load_splits(protocol: str) -> list[dict]:
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    if protocol == "loso":
        section = splits["loso"]
        if section["status"] != "ready":
            print(f"LOSO 尚未就緒:{section.get('reason', '')}")
            sys.exit(1)
        return section["folds"]
    return splits["groupkfold"]["folds"]


# ---------- 視窗級 ground truth(D12 kind 覆寫規則) ----------


def window_ground_truth(video: VideoData, start_t: float, end_t: float) -> int | None:
    """含 ≥1 幀 label=1 ⇒ 正例。ADL 一律負例(D12)。

    fall 影片「只含 0」的視窗如何判定,依資料集標籤語意而定(D48)：
    URFD(has_ambiguous_label=True)的 0 是官方三值標籤(-1/0/1)裡「躺姿判定的模糊過渡帶」,視為歧義予以剔除;
    只有全 -1(可能混 0)才是明確負例。
    Le2i(has_ambiguous_label=False)只有 0/1 兩值,0 就是明確的「跌倒區間外」,直接算負例、不剔除。
    """
    if video.kind != "fall":
        return 0
    mask = (video.raw_timestamps >= start_t) & (video.raw_timestamps <= end_t) & video.label_present
    present = video.raw_label[mask]
    if len(present) == 0:
        return None
    if (present == 1).any():
        return 1
    if video.has_ambiguous_label and (present == 0).all():
        return None
    return 0


# ---------- 視窗資料集建構 ----------


class WindowSample:
    __slots__ = ("video_id", "arrays", "stat_vec", "gt")

    def __init__(self, video_id, arrays, stat_vec, gt):
        self.video_id = video_id
        self.arrays = arrays
        self.stat_vec = stat_vec
        self.gt = gt


def build_window_samples(video_ids: list[str], videos: dict[str, VideoData]) -> list[WindowSample]:
    samples = []
    for vid in video_ids:
        video = videos[vid]
        for w in make_windows(video.features):
            gt = window_ground_truth(video, w.start_t, w.end_t)
            if gt is None:
                continue
            arrays = window_arrays(video.features, w)
            if any(np.all(np.isnan(v)) for v in arrays.values()):
                continue
            stat_vec = window_stat_vector(video.features, w)
            samples.append(WindowSample(vid, arrays, stat_vec, gt))
    return samples


def build_xgb_stat_samples(video_ids: list[str], videos: dict[str, VideoData]) -> list[tuple[str, np.ndarray, int]]:
    """XGBoost 用的視窗資料集(video_id, stat_vec, gt)。

    排除條件刻意跟 `build_window_samples`(規則式分類器用,以 5 個原始特徵陣列是否全 NaN 判斷)不同——
    XGBoost 吃的是 54 維統計向量,只有在全部 9 個基礎特徵都整段缺失(向量全零)時才真的沒有可用資料,
    這也是 `prepare_train_export.py` 訓練資料採用的同一套邏輯(此函式被兩邊共用,避免 train/eval
    視窗集合不一致——這正是 D18 發現的 bug 成因,曾經因為兩邊各自維護邏輯而各算各的)。
    """
    samples = []
    for vid in video_ids:
        video = videos[vid]
        for w in make_windows(video.features):
            gt = window_ground_truth(video, w.start_t, w.end_t)
            if gt is None:
                continue
            stat_vec = window_stat_vector(video.features, w)
            if np.all(stat_vec == 0):
                continue
            samples.append((vid, stat_vec, gt))
    return samples


# ---------- 閾值調參(只用 train 折) ----------


def tune_thresholds(train_samples: list[WindowSample]) -> RuleThresholds:
    base = RuleThresholds.literature_default()
    if not train_samples:
        return base

    y_true = np.array([s.gt for s in train_samples])
    best_f1, best = -1.0, base
    for v_y_th, theta_th in itertools.product(TUNE_V_Y_GRID, TUNE_THETA_GRID):
        cand = RuleThresholds(v_y_threshold=v_y_th, omega_threshold=base.omega_threshold, theta_threshold=theta_th, rho_threshold=base.rho_threshold, hip_height_threshold=base.hip_height_threshold)
        scores = np.array([window_score(s.arrays, cand) for s in train_samples])
        y_pred = (scores > 0).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best = f1, cand
    return best


# ---------- 視窗級指標 ----------


def window_level_metrics(samples: list[WindowSample], thresholds: RuleThresholds) -> dict:
    if not samples:
        return {"n": 0}
    y_true = np.array([s.gt for s in samples])
    scores = np.array([window_score(s.arrays, thresholds) for s in samples])
    y_pred = (scores > 0).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pr_auc = average_precision_score(y_true, scores) if len(set(y_true.tolist())) > 1 else float("nan")

    return {
        "n": len(samples),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),  # [[TN,FP],[FN,TP]]
    }


# ---------- 事件級指標 ----------


def run_fsm_on_video(video: VideoData, fsm_config: FSMConfig) -> FallStateMachine:
    fsm = FallStateMachine(fsm_config)
    f = video.features
    for i in range(len(f)):
        fsm.step(f.frame(i))
    return fsm


def video_duration_hours(video: VideoData) -> float:
    if len(video.raw_timestamps) < 2:
        return 0.0
    return float(video.raw_timestamps[-1] - video.raw_timestamps[0]) / 3600.0


def event_level_metrics(test_video_ids: list[str], videos: dict[str, VideoData], fsm_config: FSMConfig) -> dict:
    fall_ids = [v for v in test_video_ids if videos[v].kind == "fall"]
    adl_ids = [v for v in test_video_ids if videos[v].kind == "adl"]

    algo_delays, alert_delays = [], []
    confirmed_fall = 0
    for vid in fall_ids:
        video = videos[vid]
        fsm = run_fsm_on_video(video, fsm_config)
        impact_idx = np.flatnonzero(video.label_present & (video.raw_label == 1))
        impact_t = float(video.raw_timestamps[impact_idx[0]]) if len(impact_idx) else None

        on_ground_t = next((tr.t for tr in fsm.log if tr.to_state == State.ON_GROUND), None)
        confirmed_t = fsm.confirmed_at
        if confirmed_t is not None:
            confirmed_fall += 1
        if impact_t is not None and on_ground_t is not None:
            algo_delays.append(on_ground_t - impact_t)
        if impact_t is not None and confirmed_t is not None:
            alert_delays.append(confirmed_t - impact_t)

    sensitivity = confirmed_fall / len(fall_ids) if fall_ids else float("nan")
    # Wilson score 95% CI(Phase 5,docs/PLAN2.md):小樣本折(如 P3/P4/P5 每折僅 6 段 fall)的點估計
    # 需要搭配信賴區間解讀,不能只看單一數字。
    sensitivity_ci = wilson_interval(confirmed_fall, len(fall_ids)) if fall_ids else None

    specificity = None
    specificity_ci = None
    fp_per_hour = None
    adl_total_hours = None
    if adl_ids:  # D15:P3/P4/P5 折沒有 adl test 樣本,specificity 留 None(不可算)
        confirmed_adl = 0
        total_hours = 0.0
        for vid in adl_ids:
            video = videos[vid]
            fsm = run_fsm_on_video(video, fsm_config)
            if fsm.confirmed_at is not None:
                confirmed_adl += 1
            total_hours += video_duration_hours(video)
        specificity = 1.0 - (confirmed_adl / len(adl_ids))
        true_negatives = len(adl_ids) - confirmed_adl
        specificity_ci = wilson_interval(true_negatives, len(adl_ids))
        fp_per_hour = (confirmed_adl / total_hours) if total_hours > 0 else float("nan")
        adl_total_hours = total_hours  # FP/小時的分母,供報告呈現樣本時長脈絡用(D48)

    return {
        "n_fall": len(fall_ids),
        "n_adl": len(adl_ids),
        "event_sensitivity": sensitivity,
        "event_sensitivity_ci": sensitivity_ci,
        "event_specificity": specificity,
        "event_specificity_ci": specificity_ci,
        "false_alarms_per_hour": fp_per_hour,
        "adl_total_hours": adl_total_hours,
        "algo_delay_s_mean": float(np.mean(algo_delays)) if algo_delays else None,
        "alert_delay_s_mean": float(np.mean(alert_delays)) if alert_delays else None,
        "n_delay_samples": len(alert_delays),
    }


def tune_fsm_timing(train_ids: list[str], videos: dict[str, VideoData], base_config: FSMConfig) -> tuple[float, float]:
    """聯合搜尋 falling_timeout_s 與 confirm_seconds(D16):
    用 train 折的 fall 影片算 sensitivity、adl 影片算誤報數,選「sensitivity 最高、同分則誤報最少」的組合。
    """
    fall_ids = [v for v in train_ids if videos[v].kind == "fall"]
    adl_ids = [v for v in train_ids if videos[v].kind == "adl"]
    if not fall_ids:
        return base_config.falling_timeout_s, base_config.confirm_seconds

    best = (base_config.falling_timeout_s, base_config.confirm_seconds)
    best_score = (-1.0, -1.0)
    for timeout, confirm_s in itertools.product(TUNE_FALLING_TIMEOUT_GRID, TUNE_CONFIRM_SECONDS_GRID):
        cfg = FSMConfig(**{**base_config.__dict__, "falling_timeout_s": timeout, "confirm_seconds": confirm_s})
        confirmed = sum(1 for vid in fall_ids if run_fsm_on_video(videos[vid], cfg).confirmed_at is not None)
        sensitivity = confirmed / len(fall_ids)
        false_alarms = sum(1 for vid in adl_ids if run_fsm_on_video(videos[vid], cfg).confirmed_at is not None)
        fa_rate = false_alarms / len(adl_ids) if adl_ids else 0.0
        score = (sensitivity, -fa_rate)
        if score > best_score:
            best_score, best = score, (timeout, confirm_s)
    return best


# ---------- 主流程 ----------


def run_fold(fold: dict, protocol: str, videos: dict[str, VideoData]) -> dict:
    train_ids = [v for v in fold["train"] if v in videos]
    test_ids = [v for v in fold["test"] if v in videos]

    train_samples = build_window_samples(train_ids, videos)
    test_samples = build_window_samples(test_ids, videos)

    default_th = RuleThresholds.literature_default()
    tuned_th = tune_thresholds(train_samples)

    default_fsm_cfg = FSMConfig(confirm_seconds=EVAL_CONFIRM_SECONDS)
    tuned_falling_timeout, tuned_confirm_seconds = tune_fsm_timing(train_ids, videos, default_fsm_cfg)
    tuned_fsm_cfg = FSMConfig(
        confirm_seconds=tuned_confirm_seconds,
        v_y_threshold=tuned_th.v_y_threshold,
        on_ground_theta_threshold=tuned_th.theta_threshold,
        falling_timeout_s=tuned_falling_timeout,
    )

    result = {
        "fold_id": fold["fold"],
        "fold_name": fold.get("subject", f"groupkfold-{fold['fold']}"),
        "n_train_videos": len(train_ids),
        "n_test_videos": len(test_ids),
        "window_default": window_level_metrics(test_samples, default_th),
        "window_tuned": window_level_metrics(test_samples, tuned_th),
        "tuned_thresholds": {
            "v_y_threshold": tuned_th.v_y_threshold,
            "theta_threshold": tuned_th.theta_threshold,
            "falling_timeout_s": tuned_falling_timeout,
            "confirm_seconds": tuned_confirm_seconds,
        },
        "event_default": event_level_metrics(test_ids, videos, default_fsm_cfg),
        "event_tuned": event_level_metrics(test_ids, videos, tuned_fsm_cfg),
    }
    return result


def _fmt(x, digits=3) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, float) and np.isnan(x):
        return "N/A"
    return f"{x:.{digits}f}"


def _fmt_ci(ci) -> str:
    """格式化 Wilson score 信賴區間(Phase 5)。ci 為 None 時代表沒有樣本可算。"""
    if ci is None:
        return "N/A"
    lo, hi = ci
    return f"[{lo:.2f}, {hi:.2f}]"


def write_report(protocol: str, fold_results: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "rule_baseline.md"

    lines = [
        "# 規則式 Baseline 評估結果",
        "",
        f"協定：{protocol}(docs/PLAN.md §7.1);評估用 N={EVAL_CONFIRM_SECONDS}s(D11,部署另用 N=10s)。",
        "",
        "## 視窗級指標(文獻預設閾值 vs 折內調參後閾值)",
        "",
        "| 折 | n(視窗) | P(預設) | R(預設) | F1(預設) | PR-AUC(預設) | P(調參) | R(調參) | F1(調參) | PR-AUC(調參) | 調參後 v_y/θ |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in fold_results:
        d, t = r["window_default"], r["window_tuned"]
        th = r["tuned_thresholds"]
        lines.append(
            f"| {r['fold_name']} | {d.get('n', 0)} | {_fmt(d.get('precision'))} | {_fmt(d.get('recall'))} | {_fmt(d.get('f1'))} | {_fmt(d.get('pr_auc'))} "
            f"| {_fmt(t.get('precision'))} | {_fmt(t.get('recall'))} | {_fmt(t.get('f1'))} | {_fmt(t.get('pr_auc'))} | {th['v_y_threshold']}/{th['theta_threshold']}° |"
        )

    lines += [
        "",
        "**注意**：調參閾值只在各折的 train 影片上搜尋(不碰 test),grid = v_y∈{1.0,1.5,2.0,2.5,3.0}、θ∈{45,50,55,60,65,70}°。",
        "",
        "## 事件級指標(文獻預設 vs 折內調參後)",
        "",
        "FP/小時分母＝ADL 影片總時長（不含 fall 影片）。",
        "",
        "### 文獻預設(v_y>2.0、θ>60°、FALLING 逾時 1.0s)",
        "",
        "| 折 | fall 段數 | adl 段數 | Sensitivity | Sensitivity 95% CI | Specificity | Specificity 95% CI | FP/小時 | 演算法延遲(s) | 告警延遲(s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in fold_results:
        e = r["event_default"]
        spec_note = "" if e["n_adl"] > 0 else "（無 adl 測試樣本,D15）"
        lines.append(
            f"| {r['fold_name']} | {e['n_fall']} | {e['n_adl']} | {_fmt(e['event_sensitivity'])} | {_fmt_ci(e.get('event_sensitivity_ci'))} "
            f"| {_fmt(e['event_specificity'])}{spec_note} | {_fmt_ci(e.get('event_specificity_ci'))} "
            f"| {_fmt(e['false_alarms_per_hour'])} | {_fmt(e['algo_delay_s_mean'])} | {_fmt(e['alert_delay_s_mean'])} |"
        )

    lines += [
        "",
        "### 折內調參後(v_y/θ 沿用視窗級調參結果;falling_timeout_s × confirm_seconds 以 train 折聯合搜尋,D16)",
        "",
        "| 折 | fall 段數 | adl 段數 | Sensitivity | Sensitivity 95% CI | Specificity | Specificity 95% CI | FP/小時 | 演算法延遲(s) | 告警延遲(s) | 逾時/確認秒數 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in fold_results:
        e = r["event_tuned"]
        th = r["tuned_thresholds"]
        spec_note = "" if e["n_adl"] > 0 else "（無 adl 測試樣本,D15）"
        lines.append(
            f"| {r['fold_name']} | {e['n_fall']} | {e['n_adl']} | {_fmt(e['event_sensitivity'])} | {_fmt_ci(e.get('event_sensitivity_ci'))} "
            f"| {_fmt(e['event_specificity'])}{spec_note} | {_fmt_ci(e.get('event_specificity_ci'))} "
            f"| {_fmt(e['false_alarms_per_hour'])} | {_fmt(e['algo_delay_s_mean'])} | {_fmt(e['alert_delay_s_mean'])} | {th['falling_timeout_s']}s / {th['confirm_seconds']}s |"
        )

    lines += [
        "",
        "**Wilson score 95% 信賴區間（Phase 5，docs/PLAN2.md）**：每折的測試影片數很少（P3/P4/P5 折各只有 6 段 fall），"
        "Sensitivity/Specificity 只是點估計，務必搭配 CI 解讀——CI 越寬代表這個數字越不穩固，不是模型表現不好，是樣本量本來就小。"
        "視窗級 F1 不附 CI：F1 沒有封閉解公式，要用 bootstrap 重抽樣才能估，這個資料量下投入產出比不高，暫不做。",
        "",
        "**重要發現（D16）**：文獻預設的 `FALLING→ON_GROUND` 1.0 秒逾時窗對本資料集(YOLO26-pose bbox + URFD 攝影機視角)偏緊,"
        "實測 30 段 fall 中 23 段在「已確認倒地」期間內存在同時滿足 θ>60°/ρ>1.0/髖高<0.5 三條件的瞬間,但常發生在觸發後 1.0–1.5 秒左右。"
        "**更關鍵的發現**：即使放寬逾時窗、成功進入 ON_GROUND(25/30 段),進入後到影片結束的剩餘時長全部 <2.0 秒(中位數僅 0.77 秒)——"
        "URFD 片段短 + 本管線判定「已躺平」偏晚,使 D11 原訂的評估用 N=2s 對這批資料系統性過嚴(文獻預設事件級 Sensitivity 恆為 0)。"
        "故 `confirm_seconds` 也納入折內調參範圍(grid {0.3,0.5,0.8,1.0,1.5}s),不再視為固定的評估值,此發現連帶更新 D11。",
        "",
        "**局限**：`TUNE_CONFIRM_SECONDS_GRID` 的候選範圍(0.3–1.5s)是根據 URFD 全部 30 段 fall 影片"
        "(涵蓋每折未來的 test 影片)的探索性分析定案,非嚴格巢狀 CV；`tune_fsm_timing()` 選最終值時"
        "只用 train_ids,但候選邊界本身已隱含全資料集資訊。佐證：P1-P5 五折最終全部選中同一組邊界值"
        "(1.5s/0.3s),顯示這個邊界對結果有實質影響，可能讓 Sensitivity 有輕微樂觀偏誤。",
        "",
        "**LOSO 折指標可用性不對稱（D15）**：ADL 只有 P1/P2 兩位受試者出現。P1、P2 折的 test 集同時含 fall+adl,"
        "可算完整 Sensitivity+Specificity;P3/P4/P5 折的 test 集只有 fall,Specificity/FP 標 N/A,不可跟 P1/P2 折平均後當完整指標呈現。",
        "",
        "**站姿 vs 坐姿分層報告（§7.2 要求）**：目前找不到 URFD 官方提供的逐段「站姿跌倒/坐姿跌倒」對照表"
        "(僅知全體 15+15 的總數,無法對應到個別 fall-XX 影片),故本次報告從缺,誠實記錄此限制。"
        "若之後找到可信來源或決定人工判讀,可回補。",
        "",
        "**偵測延遲定義**：演算法延遲 = GT 撞擊幀(第一個 raw_label==1 的幀)→ 進入 ON_GROUND;"
        "告警延遲 = 撞擊 → CONFIRMED(含刻意設計的 N 秒確認)。",
        "",
        "## 混淆矩陣（視窗級，折內調參後）",
        "",
        "| 折 | TN | FP | FN | TP |",
        "|---|---|---|---|---|",
    ]
    agg_tn = agg_fp = agg_fn = agg_tp = 0
    for r in fold_results:
        cm = r["window_tuned"].get("confusion_matrix")
        if not cm:
            continue
        (tn, fp), (fn, tp) = cm
        agg_tn, agg_fp, agg_fn, agg_tp = agg_tn + tn, agg_fp + fp, agg_fn + fn, agg_tp + tp
        lines.append(f"| {r['fold_name']} | {tn} | {fp} | {fn} | {tp} |")
    lines += [
        f"| **加總** | **{agg_tn}** | **{agg_fp}** | **{agg_fn}** | **{agg_tp}** |",
        "",
        "TN=真陰性(正確判斷非跌倒)、FP=誤報(把非跌倒判成跌倒)、FN=漏報(把跌倒判成非跌倒)、TP=真陽性(正確判斷跌倒)；"
        "此表為視窗級(1.5 秒滑動視窗)統計，非事件級(整段影片)統計，兩者不可互換解讀。",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------- 跨資料集泛化(Phase 7,docs/PLAN2.md;URFD 訓練 → Le2i 純測試) ----------


def write_cross_report(result: dict) -> Path:
    """只報事件級指標(docs/PLAN.md §7.1 P3、§7.2):Le2i 的視窗級標籤語意跟 URFD 是否
    完全對等尚未像事件級那樣經過同等驗證,不延伸比較基礎。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cross_dataset.md"

    e = result["event_tuned"]
    th = result["tuned_thresholds"]

    lines = [
        "# 跨資料集泛化：URFD 訓練 → Le2i 純測試",
        "",
        "協定：docs/PLAN.md §7.1 P3。門檻與時間參數只用 URFD(70 段)train 資料調參，"
        "Le2i 完全沒被看過、也沒有參與任何調參——受試者/場景/攝影機皆天然不相交。"
        "只報事件級指標，不報視窗級：Le2i 的視窗級標籤語意（哪些幀算跌倒中）跟 URFD "
        "是否完全對等，還沒有像事件級（整段影片是否判定跌倒）那樣經過同等程度的驗證。",
        "",
        f"URFD(train)：{result['n_train_videos']} 段。Le2i(test)：{result['n_test_videos']} 段"
        f"（{e.get('n_fall', 0)} 段跌倒 + {e.get('n_adl', 0)} 段日常活動）。",
        "",
        "## 事件級指標（套用 URFD 調參後的門檻，Wilson 95% 信賴區間見 docs/PLAN2.md Phase 5）",
        "",
        "| 指標 | 數值 | 95% CI |",
        "|---|---|---|",
        f"| Sensitivity | {_fmt(e.get('event_sensitivity'))} | {_fmt_ci(e.get('event_sensitivity_ci'))} |",
        f"| Specificity | {_fmt(e.get('event_specificity'))} | {_fmt_ci(e.get('event_specificity_ci'))} |",
        f"| FP/小時 | {_fmt(e.get('false_alarms_per_hour'))} | — |",
        "",
    ]
    adl_hours = e.get("adl_total_hours")
    if adl_hours and adl_hours > 0:
        extrap = 1.0 / adl_hours
        lines += [
            f"**FP/小時分母說明**：分母＝ADL 影片總時長（不含 fall 影片），此處僅 {e.get('n_adl', 0)} 段共 "
            f"{adl_hours * 3600:.1f} 秒，外推倍數約 {extrap:.0f}×，數字不具統計穩定性，解讀請以 Specificity "
            "及其信賴區間為準。",
            "",
        ]
    lines += [
        f"調參後參數（只在 URFD train 上搜尋）：v_y={th['v_y_threshold']}、θ={th['theta_threshold']}°、"
        f"falling_timeout_s={th['falling_timeout_s']}s、confirm_seconds={th['confirm_seconds']}s。",
        "",
        "**對照**：URFD 內部 LOSO 各折事件級指標見 [rule_baseline.md](rule_baseline.md)"
        "（注意：那是同資料集內部交叉驗證，跟這裡的跨資料集純測試不是同一種協定，"
        "數字不可直接相減當「下滑幅度」，只能定性比較量級）。",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def run_cross_evaluation(model_kind: str) -> None:
    if model_kind != "rule":
        print("跨資料集泛化目前只支援 --model rule（docs/PLAN2.md Phase 7 範圍未涵蓋 XGBoost 版）。")
        sys.exit(1)

    if not LE2I_PROCESSED_DIR.exists():
        print(f"找不到 {LE2I_PROCESSED_DIR}，請先執行 scripts/prepare_le2i.py。")
        sys.exit(1)

    print("載入 URFD(train)關鍵點與特徵中...")
    urfd_videos = load_all_videos(PROCESSED_DIR)
    print(f"已載入 {len(urfd_videos)} 支 URFD 影片。")

    print("載入 Le2i(test)關鍵點與特徵中...")
    le2i_videos = load_all_videos(LE2I_PROCESSED_DIR)
    if not le2i_videos:
        print(f"{LE2I_PROCESSED_DIR} 是空的，無法評估。")
        sys.exit(1)
    print(f"已載入 {len(le2i_videos)} 支 Le2i 影片。")

    all_videos = {**urfd_videos, **le2i_videos}
    fold = {
        "fold": 0,
        "fold_name": "cross-le2i",
        "train": list(urfd_videos.keys()),
        "test": list(le2i_videos.keys()),
    }
    result = run_fold(fold, "cross", all_videos)
    out_path = write_cross_report(result)
    print(f"已寫入 {out_path}")

    e = result["event_tuned"]
    print(
        f"  cross-le2i: sensitivity={_fmt(e.get('event_sensitivity'))} "
        f"specificity={_fmt(e.get('event_specificity'))} FP/hr={_fmt(e.get('false_alarms_per_hour'))}"
    )


# ---------- XGBoost(Phase 3;讀 Colab 訓練回來的權重,本機重現評估) ----------


def load_xgb_fold_models() -> dict[str, "xgb.Booster"]:
    import xgboost as xgb

    if not XGB_MODELS_DIR.exists():
        return {}
    models = {}
    for path in sorted(XGB_MODELS_DIR.glob("xgb_fold_*.json")):
        subject = path.stem.replace("xgb_fold_", "")
        booster = xgb.Booster()
        booster.load_model(str(path))
        models[subject] = booster
    return models


def xgb_window_metrics(samples: list[tuple[str, np.ndarray, int]], booster) -> dict:
    if not samples:
        return {"n": 0}
    import xgboost as xgb

    X = np.stack([s[1] for s in samples])
    y_true = np.array([s[2] for s in samples])
    scores = booster.predict(xgb.DMatrix(X))
    y_pred = (scores >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pr_auc = average_precision_score(y_true, scores) if len(set(y_true.tolist())) > 1 else float("nan")
    return {
        "n": len(samples),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
    }


def run_xgb_evaluation(protocol: str, videos: dict[str, VideoData]) -> None:
    if not XGB_MODELS_DIR.exists():
        print(f"找不到 {XGB_MODELS_DIR}——請先完成 Colab 訓練(notebooks/fall-guard-cv_train_xgboost_colab.ipynb),")
        print("把下載回來的 xgb_fold_*.json / xgb_final.json / xgb_loso_results.json 放進這個資料夾。")
        sys.exit(1)

    models = load_xgb_fold_models()
    if not models:
        print(f"{XGB_MODELS_DIR} 存在但找不到任何 xgb_fold_*.json,無法評估。")
        sys.exit(1)
    print(f"已載入 {len(models)} 個折模型:{sorted(models.keys())}")

    folds = load_splits(protocol)
    local_results = []
    for fold in folds:
        subject = fold.get("subject")
        if subject not in models:
            print(f"警告:找不到 {subject} 的模型,跳過此折")
            continue
        test_ids = [v for v in fold["test"] if v in videos]
        test_samples = build_xgb_stat_samples(test_ids, videos)
        metrics = xgb_window_metrics(test_samples, models[subject])
        metrics["fold"] = subject
        local_results.append(metrics)
        print(f"  {subject}(本機重現): n={metrics['n']} P={_fmt(metrics.get('precision'))} R={_fmt(metrics.get('recall'))} F1={_fmt(metrics.get('f1'))}")

    colab_results_path = XGB_MODELS_DIR / "xgb_loso_results.json"
    lines = ["# XGBoost 本機重現結果", "", "模型來源：`models/xgboost/`（Colab 訓練，見 notebooks/fall-guard-cv_train_xgboost_colab.ipynb）", ""]
    if colab_results_path.exists():
        colab_data = json.loads(colab_results_path.read_text(encoding="utf-8"))
        colab_by_fold = {m["fold"]: m for m in colab_data["folds"]}
        lines += ["## Colab vs 本機重現對照（容許 ±0.01 誤差）", "", "| 折 | 指標 | Colab | 本機 | 差異 | 通過 |", "|---|---|---|---|---|---|"]
        all_ok = True
        for r in local_results:
            colab_m = colab_by_fold.get(r["fold"])
            if colab_m is None:
                continue
            for key in ["precision", "recall", "f1"]:
                diff = abs(r[key] - colab_m[key])
                ok = diff <= 0.01
                all_ok = all_ok and ok
                lines.append(f"| {r['fold']} | {key} | {colab_m[key]:.3f} | {r[key]:.3f} | {diff:.3f} | {'✅' if ok else '❌'} |")
        print(f"\n=== 重現驗收：{'✅ 全部通過(±0.01)' if all_ok else '❌ 有指標誤差超過 0.01,請檢查 xgboost 版本是否一致'} ===")
    else:
        lines += ["（找不到 `xgb_loso_results.json`,無法自動比對 Colab 數字,僅列本機重現結果）", ""]
        lines += ["| 折 | n | Precision | Recall | F1 | PR-AUC |", "|---|---|---|---|---|---|"]
        for r in local_results:
            lines.append(f"| {r['fold']} | {r['n']} | {_fmt(r.get('precision'))} | {_fmt(r.get('recall'))} | {_fmt(r.get('f1'))} | {_fmt(r.get('pr_auc'))} |")

    out_path = RESULTS_DIR / "xgb_baseline.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已寫入 {out_path}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["rule", "xgb"], default="rule")
    parser.add_argument("--protocol", choices=["loso", "groupkfold", "cross"], default="loso")
    args = parser.parse_args()

    if args.protocol == "cross":
        run_cross_evaluation(args.model)
        return

    print("載入 70 支影片的關鍵點與特徵中...")
    videos = load_all_videos()
    print(f"已載入 {len(videos)} 支影片的特徵。")

    if args.model == "xgb":
        run_xgb_evaluation(args.protocol, videos)
        return

    folds = load_splits(args.protocol)
    fold_results = []
    for fold in folds:
        print(f"評估 fold {fold.get('subject', fold['fold'])} ...")
        fold_results.append(run_fold(fold, args.protocol, videos))

    out_path = write_report(args.protocol, fold_results)
    print(f"已寫入 {out_path}")

    for r in fold_results:
        ed, et = r["event_default"], r["event_tuned"]
        print(
            f"  {r['fold_name']}: window F1(調參)={_fmt(r['window_tuned'].get('f1'))} "
            f"event sensitivity 預設={_fmt(ed['event_sensitivity'])} 調參={_fmt(et['event_sensitivity'])} "
            f"specificity 預設={_fmt(ed['event_specificity'])} 調參={_fmt(et['event_specificity'])}"
        )


if __name__ == "__main__":
    main()
