"""規則式 baseline 評估(docs/PLAN.md §7.2 / Phase 2 DoD)。

視窗級:precision/recall/F1/PR-AUC + 混淆矩陣(文獻預設閾值 vs 折內調參後閾值)。
事件級:用 fsm.py 完整跑過每支測試影片,算 Event Sensitivity/Specificity/
       false alarms per hour/偵測延遲(演算法延遲、告警延遲分開報)。

用法：
    uv run python scripts/evaluate.py --model rule --protocol loso
    uv run python scripts/evaluate.py --model rule --protocol groupkfold
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

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
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
    __slots__ = ("video_id", "kind", "features", "raw_label", "label_present", "raw_timestamps")

    def __init__(self, video_id, kind, features, raw_label, label_present, raw_timestamps):
        self.video_id = video_id
        self.kind = kind
        self.features = features
        self.raw_label = raw_label
        self.label_present = label_present
        self.raw_timestamps = raw_timestamps


def load_all_videos() -> dict[str, VideoData]:
    out: dict[str, VideoData] = {}
    for npz_path in sorted(PROCESSED_DIR.glob("*.npz")):
        with np.load(npz_path) as d:
            xyn, conf, bbox = d["xyn"], d["conf"], d["bbox_xywh"]
            raw_timestamps = d["timestamps"]
            raw_label, label_present = d["raw_label"], d["label_present"]
            kind = str(d["kind"])
            video_id = str(d["video_id"])
        features = compute_features(xyn, conf, bbox, raw_timestamps)
        out[video_id] = VideoData(video_id, kind, features, raw_label, label_present, raw_timestamps)
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
    """含 ≥1 幀 label=1 ⇒ 正例;全 -1(可能混 0)⇒ 負例;只含 0 ⇒ 剔除。ADL 一律負例(D12)。"""
    if video.kind != "fall":
        return 0
    mask = (video.raw_timestamps >= start_t) & (video.raw_timestamps <= end_t) & video.label_present
    present = video.raw_label[mask]
    if len(present) == 0:
        return None
    if (present == 1).any():
        return 1
    if (present == 0).all():
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

    specificity = None
    fp_per_hour = None
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
        fp_per_hour = (confirmed_adl / total_hours) if total_hours > 0 else float("nan")

    return {
        "n_fall": len(fall_ids),
        "n_adl": len(adl_ids),
        "event_sensitivity": sensitivity,
        "event_specificity": specificity,
        "false_alarms_per_hour": fp_per_hour,
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
        "### 文獻預設(v_y>2.0、θ>60°、FALLING 逾時 1.0s)",
        "",
        "| 折 | fall 段數 | adl 段數 | Sensitivity | Specificity | FP/小時 | 演算法延遲(s) | 告警延遲(s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in fold_results:
        e = r["event_default"]
        spec_note = "" if e["n_adl"] > 0 else "（無 adl 測試樣本,D15）"
        lines.append(
            f"| {r['fold_name']} | {e['n_fall']} | {e['n_adl']} | {_fmt(e['event_sensitivity'])} | {_fmt(e['event_specificity'])}{spec_note} "
            f"| {_fmt(e['false_alarms_per_hour'])} | {_fmt(e['algo_delay_s_mean'])} | {_fmt(e['alert_delay_s_mean'])} |"
        )

    lines += [
        "",
        "### 折內調參後(v_y/θ 沿用視窗級調參結果;falling_timeout_s × confirm_seconds 以 train 折聯合搜尋,D16)",
        "",
        "| 折 | fall 段數 | adl 段數 | Sensitivity | Specificity | FP/小時 | 演算法延遲(s) | 告警延遲(s) | 逾時/確認秒數 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in fold_results:
        e = r["event_tuned"]
        th = r["tuned_thresholds"]
        spec_note = "" if e["n_adl"] > 0 else "（無 adl 測試樣本,D15）"
        lines.append(
            f"| {r['fold_name']} | {e['n_fall']} | {e['n_adl']} | {_fmt(e['event_sensitivity'])} | {_fmt(e['event_specificity'])}{spec_note} "
            f"| {_fmt(e['false_alarms_per_hour'])} | {_fmt(e['algo_delay_s_mean'])} | {_fmt(e['alert_delay_s_mean'])} | {th['falling_timeout_s']}s / {th['confirm_seconds']}s |"
        )

    lines += [
        "",
        "**重要發現（D16）**：文獻預設的 `FALLING→ON_GROUND` 1.0 秒逾時窗對本資料集(YOLO26-pose bbox + URFD 攝影機視角)偏緊,"
        "實測 30 段 fall 中 23 段在「已確認倒地」期間內存在同時滿足 θ>60°/ρ>1.0/髖高<0.5 三條件的瞬間,但常發生在觸發後 1.0–1.5 秒左右。"
        "**更關鍵的發現**：即使放寬逾時窗、成功進入 ON_GROUND(25/30 段),進入後到影片結束的剩餘時長全部 <2.0 秒(中位數僅 0.77 秒)——"
        "URFD 片段短 + 本管線判定「已躺平」偏晚,使 D11 原訂的評估用 N=2s 對這批資料系統性過嚴(文獻預設事件級 Sensitivity 恆為 0)。"
        "故 `confirm_seconds` 也納入折內調參範圍(grid {0.3,0.5,0.8,1.0,1.5}s),不再視為固定的評估值,此發現連帶更新 D11。",
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
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


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


def xgb_window_metrics(samples: list[WindowSample], booster) -> dict:
    if not samples:
        return {"n": 0}
    import xgboost as xgb

    X = np.stack([s.stat_vec for s in samples])
    y_true = np.array([s.gt for s in samples])
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
        print(f"找不到 {XGB_MODELS_DIR}——請先完成 Colab 訓練(notebooks/train_colab.ipynb),")
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
        test_samples = build_window_samples(test_ids, videos)
        metrics = xgb_window_metrics(test_samples, models[subject])
        metrics["fold"] = subject
        local_results.append(metrics)
        print(f"  {subject}(本機重現): n={metrics['n']} P={_fmt(metrics.get('precision'))} R={_fmt(metrics.get('recall'))} F1={_fmt(metrics.get('f1'))}")

    colab_results_path = XGB_MODELS_DIR / "xgb_loso_results.json"
    lines = ["# XGBoost 本機重現結果", "", f"模型來源：`{XGB_MODELS_DIR}`（Colab 訓練，見 notebooks/train_colab.ipynb）", ""]
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
    parser.add_argument("--protocol", choices=["loso", "groupkfold"], default="loso")
    args = parser.parse_args()

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
