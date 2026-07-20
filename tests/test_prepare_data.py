"""scripts/prepare_data.py 產出的 npz 的 schema 與標籤對齊測試。

資料不進 git(見 .gitignore),故 data/processed/ 不存在時優雅跳過,
不影響尚未跑過資料管線的 clone/CI 環境。
"""

from __future__ import annotations

import numpy as np
import pytest

from fallguard.config import REPO_ROOT

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

EXPECTED_KEYS = {
    "xyn",
    "conf",
    "bbox_xywh",
    "track_id",
    "raw_label",
    "label_present",
    "fps",
    "timestamps",
    "video_id",
    "kind",
}


def _require_npz(video_id: str) -> dict:
    path = PROCESSED_DIR / f"{video_id}.npz"
    if not path.exists():
        pytest.skip(f"{path} 不存在(尚未跑 scripts/prepare_data.py)")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def test_processed_dir_has_all_70_when_present():
    if not PROCESSED_DIR.exists():
        pytest.skip("data/processed/ 不存在(尚未跑 scripts/prepare_data.py)")
    npz_files = list(PROCESSED_DIR.glob("*.npz"))
    if not npz_files:
        pytest.skip("data/processed/ 是空的")
    assert len(npz_files) <= 70
    # 沒硬性要求剛好 70(允許分批跑),但檔名必須都是合法 video_id
    for f in npz_files:
        assert f.stem.startswith("fall-") or f.stem.startswith("adl-")


def test_npz_schema_fall01():
    data = _require_npz("fall-01")
    assert EXPECTED_KEYS <= set(data.keys())

    T = data["xyn"].shape[0]
    assert data["xyn"].shape == (T, 17, 2)
    assert data["conf"].shape == (T, 17)
    assert data["bbox_xywh"].shape == (T, 4)
    assert data["track_id"].shape == (T,)
    assert data["raw_label"].shape == (T,)
    assert data["label_present"].shape == (T,)
    assert data["timestamps"].shape == (T,)

    assert data["xyn"].dtype == np.float32
    assert data["raw_label"].dtype == np.int8
    assert data["label_present"].dtype == np.bool_
    assert float(data["fps"]) > 0
    assert str(data["video_id"]) == "fall-01"
    assert str(data["kind"]) == "fall"


def test_label_alignment_fall01_full_coverage():
    """fall-01 實測:160 幀,CSV frame_num 1-160 完整覆蓋,無缺口(見 PLAN.md D12)。"""
    data = _require_npz("fall-01")
    assert data["label_present"].all()
    assert set(np.unique(data["raw_label"])) <= {-1, 0, 1}


def test_label_alignment_adl01_has_known_gap():
    """adl-01 實測:CSV 從 frame_num=6 開始覆蓋,且 frame_num=7 缺一列(見 PLAN.md D12)。
    對齊公式 csv_frame_num - 1 = 影片幀索引(0-indexed),故:
      video index 0-4(frame_num 1-5)  → 無覆蓋
      video index 5  (frame_num 6)    → 有覆蓋
      video index 6  (frame_num 7)    → 缺口,無覆蓋
      video index 7  (frame_num 8)    → 有覆蓋
    """
    data = _require_npz("adl-01")
    label_present = data["label_present"]
    assert not label_present[0:5].any()
    assert label_present[5]
    assert not label_present[6]
    assert label_present[7]


def test_adl_raw_label_is_not_task_label():
    """D12 守門:ADL npz 允許 raw_label==1(躺姿幾何特徵),但 kind 必須誠實標為 adl,
    下游程式才知道不能把這些幀當跌倒正例。"""
    data = _require_npz("adl-10")  # 已知含大量 label=1 幀的 ADL 影片之一
    assert str(data["kind"]) == "adl"
    assert (data["raw_label"] == 1).any(), "adl-10 預期含躺姿幀(D12 發現的依據樣本)"


def test_detection_rate_sane_across_available_videos():
    """回歸防呆:若未來重跑管線因某種原因整批偵測失敗,這裡會先炸,而不是悄悄產出全 NaN 的 npz。"""
    if not PROCESSED_DIR.exists():
        pytest.skip("data/processed/ 不存在")
    npz_files = list(PROCESSED_DIR.glob("*.npz"))
    if not npz_files:
        pytest.skip("data/processed/ 是空的")
    rates = []
    for f in npz_files:
        with np.load(f) as data:
            valid = ~np.isnan(data["xyn"][:, 0, 0])
            rates.append(float(valid.mean()))
    assert np.mean(rates) > 0.5, f"平均偵測率過低({np.mean(rates):.2%}),疑似管線異常"
