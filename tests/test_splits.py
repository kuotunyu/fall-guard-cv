"""data/splits.json 防洩漏守門:fold 群組交集為空、涵蓋全部影片、LOSO 結構正確。

data/splits.json 不進 git,不存在時優雅跳過。
"""

from __future__ import annotations

import json

import pytest

from fallguard.config import REPO_ROOT

SPLITS_PATH = REPO_ROOT / "data" / "splits.json"

FALL_COUNT = 30
ADL_COUNT = 40
ALL_VIDEO_IDS = {f"fall-{i:02d}" for i in range(1, FALL_COUNT + 1)} | {f"adl-{i:02d}" for i in range(1, ADL_COUNT + 1)}


@pytest.fixture()
def splits() -> dict:
    if not SPLITS_PATH.exists():
        pytest.skip(f"{SPLITS_PATH} 不存在(尚未跑 scripts/make_splits.py)")
    return json.loads(SPLITS_PATH.read_text(encoding="utf-8"))


def test_groupkfold_folds_disjoint_and_complete(splits):
    gk = splits["groupkfold"]
    assert gk["n_splits"] == len(gk["folds"])

    seen_as_test = set()
    for fold in gk["folds"]:
        train, test = set(fold["train"]), set(fold["test"])
        assert not (train & test), f"fold {fold['fold']} train/test 有交集"
        assert train | test == ALL_VIDEO_IDS, f"fold {fold['fold']} 未涵蓋全部 70 支影片"
        assert not (seen_as_test & test), "同一支影片出現在多個 fold 的 test 集"
        seen_as_test |= test

    assert seen_as_test == ALL_VIDEO_IDS, "並非每支影片都恰好被當過一次 test"


def test_groupkfold_stratification_reasonable(splits):
    """每折的 test 集不應該是清一色 fall 或清一色 adl(分層有生效)。"""
    for fold in splits["groupkfold"]["folds"]:
        test = fold["test"]
        fall_n = sum(1 for v in test if v.startswith("fall-"))
        adl_n = sum(1 for v in test if v.startswith("adl-"))
        assert fall_n > 0 and adl_n > 0, f"fold {fold['fold']} 分層失敗:fall={fall_n} adl={adl_n}"


def test_loso_when_ready(splits):
    loso = splits["loso"]
    if loso["status"] != "ready":
        pytest.skip(f"LOSO 尚未就緒:{loso.get('reason', '')}")

    all_labeled = set()
    seen_as_test = set()
    for fold in loso["folds"]:
        train, test = set(fold["train"]), set(fold["test"])
        assert not (train & test), f"LOSO fold {fold['fold']}({fold['subject']}) train/test 有交集"
        assert not (seen_as_test & test), "同一支影片出現在多個 LOSO fold 的 test 集"
        seen_as_test |= test
        all_labeled |= train | test

    assert loso["n_subjects"] == len(loso["folds"])
    # unknown 標註的影片不該出現在任何 test 集(D6:unknown 只進訓練集)
    for fold in loso["folds"]:
        assert fold["subject"] != "unknown"
