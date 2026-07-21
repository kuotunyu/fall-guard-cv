"""規則式 baseline 誤報分析(docs/PLAN.md §1.3 / Phase 2 DoD)。

用各 ADL 影片所屬 LOSO 折的折內調參後設定(不碰 test 的方法論一致)跑狀態機,
統計「動作類別 × 誤報」表,並畫跌倒/躺床/蹲下三聯特徵曲線圖說明規則為何(不)區分得開。

用法：
    uv run python scripts/error_analysis.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# matplotlib 預設字型(DejaVu Sans)不含中文字形,圖上中文會顯示空白方塊,改用系統內建中文字型。
_CJK_FONT_PATH = "C:/Windows/Fonts/msjh.ttc"
if Path(_CJK_FONT_PATH).exists():
    fm.fontManager.addfont(_CJK_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_CJK_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False  # 中文字型常缺負號字形,改用純 ASCII 減號

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (  # noqa: E402
    RESULTS_DIR,
    RuleThresholds,
    VideoData,
    build_window_samples,
    load_all_videos,
    load_splits,
    run_fsm_on_video,
    tune_fsm_timing,
    tune_thresholds,
)
from fallguard.config import REPO_ROOT  # noqa: E402
from fallguard.fsm import FSMConfig, FallStateMachine  # noqa: E402

META_PATH = REPO_ROOT / "data" / "urfd_meta.csv"
ASSETS_DIR = REPO_ROOT / "docs" / "assets"


def load_meta() -> dict[str, dict]:
    with META_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["video_id"]: row for row in csv.DictReader(f)}


def tuned_config_per_subject_fold(videos: dict[str, VideoData]) -> dict[str, FSMConfig]:
    """LOSO 每折(P1/P2/...)算一次折內調參後的 FSMConfig,回傳 {fold_subject: config}。"""
    folds = load_splits("loso")
    out = {}
    for fold in folds:
        train_ids = [v for v in fold["train"] if v in videos]
        train_samples = build_window_samples(train_ids, videos)
        tuned_th = tune_thresholds(train_samples)
        base_cfg = FSMConfig()
        falling_timeout, confirm_s = tune_fsm_timing(train_ids, videos, base_cfg)
        out[fold["subject"]] = FSMConfig(
            confirm_seconds=confirm_s,
            v_y_threshold=tuned_th.v_y_threshold,
            on_ground_theta_threshold=tuned_th.theta_threshold,
            falling_timeout_s=falling_timeout,
        )
    return out


def analyze_false_positives(videos: dict[str, VideoData], meta: dict[str, dict], configs: dict[str, FSMConfig]) -> tuple[dict, list[str]]:
    """回傳 (action_category -> {total, fp}), 以及誤觸(FP)的 video_id 清單。"""
    stats: dict[str, dict] = {}
    fp_videos: list[str] = []
    for vid, video in videos.items():
        if video.kind != "adl":
            continue
        row = meta.get(vid, {})
        action = row.get("action_category") or "(未標)"
        subject = row.get("subject_id")
        cfg = configs.get(subject)
        if cfg is None:
            continue
        fsm = run_fsm_on_video(video, cfg)
        is_fp = fsm.confirmed_at is not None

        s = stats.setdefault(action, {"total": 0, "fp": 0})
        s["total"] += 1
        if is_fp:
            s["fp"] += 1
            fp_videos.append(vid)
    return stats, fp_videos


def write_fp_table(stats: dict) -> list[str]:
    lines = ["| 動作類別 | 段數 | 誤報數 | 誤報率 |", "|---|---|---|---|"]
    for action, s in sorted(stats.items(), key=lambda kv: -kv[1]["fp"] / kv[1]["total"]):
        rate = s["fp"] / s["total"] if s["total"] else 0.0
        lines.append(f"| {action} | {s['total']} | {s['fp']} | {rate:.1%} |")
    return lines


def plot_triplet(fall_video: VideoData, lying_video: VideoData, crouch_video: VideoData, out_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=False)
    specs = [
        ("theta", "軀幹角 θ(度)", 60.0, "θ 閾值(躺姿判定用)"),
        ("v_y", "質心垂直速度 v_y(torso/s)", 2.0, "v_y 觸發閾值(預設)"),
        ("rho", "bbox 長寬比 ρ", 1.0, "ρ 閾值(躺姿判定用)"),
        ("hip_height", "髖高(torso 單位)", 0.5, "髖高閾值(躺姿判定用)"),
    ]
    videos_labeled = [
        (f"跌倒({fall_video.video_id})", fall_video, "#d62728"),
        (f"躺床({lying_video.video_id})", lying_video, "#1f77b4"),
        (f"蹲下({crouch_video.video_id})", crouch_video, "#2ca02c"),
    ]

    for ax, (key, ylabel, thresh, thresh_label) in zip(axes, specs):
        for label, video, color in videos_labeled:
            t = video.features.timestamps
            y = getattr(video.features, key)
            ax.plot(t, y, label=label, color=color, linewidth=1.5)
        ax.axhline(thresh, color="gray", linestyle="--", linewidth=1, label=thresh_label)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("時間(秒,各影片皆從 0 起算)")
    fig.suptitle("跌倒 vs 躺床 vs 蹲下:特徵曲線對照(docs/PLAN.md §1.3 誤報分析)")
    fig.tight_layout()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def pick_example(videos: dict[str, VideoData], meta: dict[str, dict], kind: str, action: str | None = None) -> VideoData | None:
    for vid, video in videos.items():
        if video.kind != kind:
            continue
        if action is not None and meta.get(vid, {}).get("action_category") != action:
            continue
        return video
    return None


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    print("載入影片與特徵...")
    videos = load_all_videos()
    meta = load_meta()

    print("計算各 LOSO 折的折內調參設定...")
    configs = tuned_config_per_subject_fold(videos)

    print("跑 ADL 誤報統計...")
    stats, fp_videos = analyze_false_positives(videos, meta, configs)

    fall_example = videos.get("fall-01")
    lying_example = pick_example(videos, meta, "adl", "躺床")
    crouch_example = pick_example(videos, meta, "adl", "蹲下/綁鞋帶")

    plot_path = ASSETS_DIR / "error_analysis_triplet.png"
    if fall_example and lying_example and crouch_example:
        plot_triplet(fall_example, lying_example, crouch_example, plot_path)
        print(f"已產出 {plot_path}")
    else:
        print("警告:找不到跌倒/躺床/蹲下三種範例其中之一,略過畫圖")

    lines = [
        "# 誤報案例分析(規則式 Baseline)",
        "",
        "採用各 LOSO 折折內調參後的設定(v_y/θ/falling_timeout/confirm_seconds,不碰 test),",
        "對全部 40 段 ADL 影片跑狀態機,統計哪些日常動作最容易被誤判為跌倒。",
        "",
        "## 動作類別 × 誤報率",
        "",
    ]
    lines += write_fp_table(stats)
    lines += [
        "",
        f"總計 40 段 ADL 中 {len(fp_videos)} 段觸發誤報：{', '.join(fp_videos) if fp_videos else '(無)'}。",
        "",
        "## 跌倒 vs 躺床 vs 蹲下 特徵曲線對照",
        "",
        f"見 `docs/assets/error_analysis_triplet.png`(範例：{fall_example.video_id if fall_example else 'N/A'} / "
        f"{lying_example.video_id if lying_example else 'N/A'} / {crouch_example.video_id if crouch_example else 'N/A'})。",
        "",
        "**為何規則能區分躺床與跌倒**：躺床是緩慢受控下降，`v_y` 全程不會超過觸發閾值，"
        "狀態機根本不會離開 NORMAL——這是躺床 vs 跌倒唯一可靠的判別子（PLAN.md §8.1）。",
        "",
        "**為何規則能區分蹲下與跌倒**：蹲下時髖高通常仍 > 0.5 torso（人還沒真正趴平），"
        "即使短暫觸發 FALLING，也達不到「躺姿」三條件，逾時退回 NORMAL。",
        "",
        "**已知弱點**：上表誤報率 > 0 的動作類別即為規則式方法目前的困難案例，"
        "通常是快速蹲下/彎腰接近跌倒的下墜特徵，或關鍵點缺失導致誤判——這些正是 Phase 3 "
        "ML 模型(學習更細緻的決策邊界)預期能改善的地方。",
    ]

    out_path = RESULTS_DIR / "error_analysis.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已寫入 {out_path}")


if __name__ == "__main__":
    main()
