"""scripts/prepare_le2i.py 的標註解析測試(Phase 7,docs/PLAN2.md)。

`parse_annotation()` 用合成標註檔測(不需要真的下載 Le2i);npz 產出的 schema/
標籤測試則跟 test_prepare_data.py 同慣例,data/processed_le2i/ 不存在時優雅跳過。
"""

from __future__ import annotations

import numpy as np
import pytest
from prepare_le2i import list_videos, parse_annotation

from fallguard.config import REPO_ROOT

LE2I_PROCESSED_DIR = REPO_ROOT / "data" / "processed_le2i"


def test_parse_annotation_fall_video(tmp_path):
    """有跌倒的標註檔:前兩行是純數字(起訖幀),D45 實地盤點的真實格式。"""
    ann = tmp_path / "video (1).txt"
    ann.write_text("48\n80\n1,1,0,0,0,0\n2,1,0,0,0,0\n5,1,292,152,311,240\n", encoding="utf-8")
    kind, start, end = parse_annotation(ann)
    assert kind == "fall"
    assert start == 48
    assert end == 80


def test_parse_annotation_adl_video_no_header(tmp_path):
    """沒有跌倒的標註檔:沒有起訖幀表頭,直接是逐幀資料(第一行就有逗號),
    D45 實地盤點發現的真實格式——官方 README 完全沒提到這個省略表頭的細節。"""
    ann = tmp_path / "video (26).txt"
    ann.write_text("1,1,72,58,132,170\n2,1,72,58,132,170\n3,1,72,58,132,170\n", encoding="utf-8")
    kind, start, end = parse_annotation(ann)
    assert kind == "adl"
    assert start is None
    assert end is None


def test_parse_annotation_fall_at_very_start(tmp_path):
    """邊界情境:跌倒從第 1 幀就開始(fall_start=1)。"""
    ann = tmp_path / "video (x).txt"
    ann.write_text("1\n10\n1,8,0,0,0,0\n", encoding="utf-8")
    kind, start, end = parse_annotation(ann)
    assert kind == "fall"
    assert start == 1
    assert end == 10


def test_list_videos_only_covers_four_annotated_scenes():
    """D45:只用有標註可驗證的 4 個場景(Coffee_room_01/02、Home_01/02),
    Office/Lecture_room 因為完全沒有標註檔案而整批排除。資料夾不存在時優雅跳過
    (符合本專案 data/ 不進 git 的慣例)。"""
    le2i_raw = REPO_ROOT / "data" / "raw" / "le2i"
    if not le2i_raw.exists():
        pytest.skip(f"{le2i_raw} 不存在(尚未跑 scripts/download_data.py --fallback le2i)")
    videos = list_videos()
    if not videos:
        pytest.skip("data/raw/le2i/ 存在但找不到任何可用影片")
    for video_id, _, _ in videos:
        assert video_id.startswith("le2i-")
        assert any(scene in video_id for scene in ("coffee01", "coffee02", "home01", "home02"))
        assert "office" not in video_id and "lecture" not in video_id


def _require_npz(video_id: str) -> dict:
    path = LE2I_PROCESSED_DIR / f"{video_id}.npz"
    if not path.exists():
        pytest.skip(f"{path} 不存在(尚未跑 scripts/prepare_le2i.py)")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def test_le2i_npz_schema_matches_urfd():
    """跟 URFD 的 data/processed/*.npz 用同一套 schema,evaluate.py 才能不改邏輯直接套用。"""
    data = _require_npz("le2i-coffee01-01")
    expected_keys = {
        "xyn", "conf", "bbox_xywh", "track_id",
        "raw_label", "label_present", "fps", "timestamps",
        "video_id", "kind",
    }
    assert expected_keys <= set(data.keys())
    T = data["xyn"].shape[0]
    assert data["xyn"].shape == (T, 17, 2)
    assert data["raw_label"].dtype == np.int8
    assert data["label_present"].dtype == np.bool_
    assert str(data["kind"]) == "fall"
    assert float(data["fps"]) == pytest.approx(25.0)


def test_le2i_label_present_always_true():
    """跟 URFD 不同:Le2i 對每一幀都有明確標註,沒有 CSV 覆蓋缺口這種情況。"""
    data = _require_npz("le2i-coffee01-01")
    assert data["label_present"].all()


def test_le2i_adl_video_has_no_positive_labels():
    """已知的無跌倒影片(D45 盤點確認):raw_label 應全為 0。"""
    data = _require_npz("le2i-coffee01-26")
    assert str(data["kind"]) == "adl"
    assert (data["raw_label"] == 0).all()
