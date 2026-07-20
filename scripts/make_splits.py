"""產出 data/splits.json:P1 影片級 GroupKFold(最低防線)+ P2 受試者級 LOSO(主協定)。

依據 docs/PLAN.md D6 / §7.1。GroupKFold 只需影片清單,現在就能跑;
LOSO 需要 data/urfd_meta.csv 的人工 subject_id 標註,標註完成前會輸出
status="pending_annotation" 並跳過,不阻塞其他 Phase 1 工作。

用法：
    uv run python scripts/make_splits.py            # 產生/更新 data/splits.json
    uv run python scripts/make_splits.py --seed 7    # 換隨機種子(預設 42)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
META_PATH = REPO_ROOT / "data" / "urfd_meta.csv"
OUT_PATH = REPO_ROOT / "data" / "splits.json"

FALL_COUNT = 30
ADL_COUNT = 40
N_SPLITS = 5


def list_videos() -> list[tuple[str, str]]:
    videos = [(f"fall-{i:02d}", "fall") for i in range(1, FALL_COUNT + 1)]
    videos += [(f"adl-{i:02d}", "adl") for i in range(1, ADL_COUNT + 1)]
    return videos


def build_groupkfold(videos: list[tuple[str, str]], seed: int) -> dict:
    """影片級、依 fall/adl 分層的 5 折(每支影片恰為一個 group,同影片的幀/視窗永不跨折)。"""
    ids = [v for v, _ in videos]
    kinds = [k for _, k in videos]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(ids, kinds)):
        train = sorted(ids[i] for i in train_idx)
        test = sorted(ids[i] for i in test_idx)
        assert not (set(train) & set(test)), "GroupKFold 產生了重疊,不應發生"
        folds.append({"fold": fold_idx, "train": train, "test": test})
    return {"n_splits": N_SPLITS, "stratify_by": "kind", "seed": seed, "folds": folds}


def load_subject_labels() -> dict[str, str] | None:
    if not META_PATH.exists():
        return None
    labels: dict[str, str] = {}
    with META_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            vid = row.get("video_id", "")
            subj = (row.get("subject_id") or "").strip()
            if vid and subj:
                labels[vid] = subj
    return labels


def build_loso(videos: list[tuple[str, str]]) -> dict:
    all_ids = [v for v, _ in videos]
    labels = load_subject_labels()

    if labels is None:
        return {"status": "pending_annotation", "reason": f"{META_PATH.name} 不存在;先跑 scripts/annotate_urfd.py"}

    labeled_count = sum(1 for v in all_ids if v in labels)
    if labeled_count < len(all_ids):
        return {
            "status": "pending_annotation",
            "reason": f"僅標註 {labeled_count}/{len(all_ids)} 段;跑 scripts/annotate_urfd.py 補完",
        }

    subjects = sorted({s for s in labels.values() if s != "unknown"})
    if not subjects:
        return {"status": "pending_annotation", "reason": "所有影片都標為 unknown,無法做 LOSO"}

    folds = []
    for fold_idx, subject in enumerate(subjects):
        test = sorted(v for v in all_ids if labels[v] == subject)
        train = sorted(v for v in all_ids if labels[v] != subject)
        assert not (set(train) & set(test))
        folds.append({"fold": fold_idx, "subject": subject, "train": train, "test": test})

    unknown_count = sum(1 for s in labels.values() if s == "unknown")
    return {
        "status": "ready",
        "n_subjects": len(subjects),
        "subjects": subjects,
        "unknown_count": unknown_count,
        "folds": folds,
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    videos = list_videos()
    groupkfold = build_groupkfold(videos, args.seed)
    loso = build_loso(videos)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps({"groupkfold": groupkfold, "loso": loso}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"已寫入 {OUT_PATH}")
    print(f"GroupKFold:{groupkfold['n_splits']} 折,種子={groupkfold['seed']}")
    if loso["status"] == "ready":
        print(f"LOSO:{loso['n_subjects']} 位受試者({', '.join(loso['subjects'])}),unknown {loso['unknown_count']} 段")
    else:
        print(f"LOSO:尚未就緒 — {loso['reason']}")


if __name__ == "__main__":
    main()
