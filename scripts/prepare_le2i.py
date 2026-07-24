"""Le2i 影片 → 關鍵點序列 .npz(本機 GPU 抽取,Phase 7,docs/PLAN2.md)。

實地盤點結果(2026-07-24,見 docs/PLAN.md D45,非憑網路二手資訊硬做)：
- 解壓後是 6 個場景資料夾：Coffee_room_01/02、Home_01/02(各自有 Annotation_files
  或 Annotations_files,可驗證真偽)、Lecture_room、Office(完全沒有任何標註檔案)。
- **只用有標註可驗證的 4 個資料夾**；Office/Lecture_room 整批排除——沒有 ground
  truth 就無法判斷影片裡到底有沒有跌倒,不編造標籤。
- 標註檔案 `video (i).txt`：有跌倒的檔案前兩行各是一個純數字,依序是「跌倒起始幀」
  「跌倒結束幀」(1-indexed),第三行起才是逐幀資料；不含跌倒的檔案沒有這兩行表頭,
  直接從逐幀資料開始(第一行就有逗號)。逐幀資料本身(activity code + bbox)是
  Le2i 自己的標註,本專案用自己的 YOLO26-pose 重新抽取,不採用、不比對。
- 4 個資料夾合計 130 段有標註影片：127 段含跌倒、3 段不含跌倒(Coffee_room_01
  ×1、Coffee_room_02 ×2)。
- 舊版 .avi 容器的 cv2 `CAP_PROP_FRAME_COUNT` 常態性比實際解碼幀數少報 1(抽查
  3 段皆是),故不信任該欄位做對齊基準；一律以 `extract_video_pose()` 實際解碼出
  的幀數為準,超出這個範圍的標註列安靜忽略,不 assert 崩潰。
- **「有無表頭」判斷式的獨立驗證(D51)**：這個規則是從資料反推出來的(非官方文件),
  曾嘗試用逐幀 activity code 欄位交叉驗證,但全量掃描 130 個標註檔後發現該欄位語意
  不明(官方 README 完全未定義)、且不足以在同一影片內部乾淨區分跌倒/非跌倒區段——
  127 段 fall 檔案裡有 78 段(61%)找不到「跌倒區間專屬」的 code,3 段 adl 檔案裡有
  2 段本身就含有 fall 範例影片中的「區間專屬」code,故不採用此欄位做二次驗證。改用
  更直接的方式：肉眼抽查 3 段 adl 影片(le2i-coffee01-26、le2i-coffee02-50、
  le2i-coffee02-52)的取樣幀,確認 3 段皆是同一種佈景——受試者刻意在地上鋪一張床墊、
  緩慢走近後躺上去(近似 URFD「躺床」ADL 的居家版本),不是真的跌倒,adl 標籤正確。
  這同時也解釋了為何這 3 段全部觸發跨資料集測試的誤報(README「跨資料集泛化」一節)：
  緩慢受控地躺上床墊,在極短的 `confirm_seconds=0.3s` 判定窗口下,跟真跌倒的姿態終態
  難以區分,是評估用門檻套錯情境,不是資料標籤本身有問題。

每支影片輸出 data/processed_le2i/{video_id}.npz,schema 與 data/processed/*.npz
(URFD)相同,讓 scripts/evaluate.py 的既有評估邏輯完全不需改動即可直接套用：
    xyn/conf/bbox_xywh/track_id/fps/timestamps  (extract_video_pose() 共用邏輯產出,
                                                   與 URFD 用同一份追蹤參數)
    raw_label     (T,) int8   frame 落在 [fall_start,fall_end] 區間內為 1,否則 0
    label_present (T,) bool   一律 True(Le2i 對每一幀都有明確標註,沒有 URFD 那種
                                          CSV 覆蓋缺口)
    video_id      scalar str  例如 "le2i-coffee01-05"
    kind          scalar str  "fall" 或 "adl"(依標註檔是否有起訖幀表頭判定)

用法：
    uv run python scripts/prepare_le2i.py                  # 抽取全部可用影片(130 支)
    uv run python scripts/prepare_le2i.py --limit 3         # 只抽前 3 支(快速測試)
    uv run python scripts/prepare_le2i.py --force            # 已存在的 npz 也重新抽取
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from fallguard.pose import extract_video_pose, resolve_pose_weights

REPO_ROOT = Path(__file__).resolve().parents[1]
LE2I_RAW_DIR = REPO_ROOT / "data" / "raw" / "le2i"
OUT_DIR = REPO_ROOT / "data" / "processed_le2i"
POSE_MODEL_NAME = "yolo26m-pose.pt"

# 只用有標註可驗證的資料夾(D45);(場景資料夾名, video_id 用的短代碼, 標註子資料夾名)。
# Coffee_room_02 的標註子資料夾名多一個 s("Annotations_files"),下載下來就是這樣,
# 不是筆誤——盤點時發現的真實不一致,原樣保留而非「修正」成看起來一致的假象。
ANNOTATED_SCENES = [
    ("Coffee_room_01", "coffee01", "Annotation_files"),
    ("Coffee_room_02", "coffee02", "Annotations_files"),
    ("Home_01", "home01", "Annotation_files"),
    ("Home_02", "home02", "Annotation_files"),
]


def list_videos() -> list[tuple[str, Path, Path]]:
    """回傳 (video_id, video_path, annotation_path) 清單,只含四個有標註的場景。"""
    out: list[tuple[str, Path, Path]] = []
    for scene_dir, scene_code, ann_dirname in ANNOTATED_SCENES:
        base = LE2I_RAW_DIR / scene_dir / scene_dir
        videos_dir = base / "Videos"
        ann_dir = base / ann_dirname
        if not videos_dir.exists() or not ann_dir.exists():
            print(f"警告:{base} 底下找不到 Videos/ 或 {ann_dirname}/,略過整個場景")
            continue
        for video_path in sorted(videos_dir.glob("*.avi")):
            stem = video_path.stem  # 例如 "video (5)"
            ann_path = ann_dir / f"{stem}.txt"
            if not ann_path.exists():
                print(f"警告:{video_path.name} 找不到對應標註,略過")
                continue
            num = stem.split("(")[1].rstrip(")")
            video_id = f"le2i-{scene_code}-{int(num):02d}"
            out.append((video_id, video_path, ann_path))
    return out


def parse_annotation(ann_path: Path) -> tuple[str, int | None, int | None]:
    """回傳 (kind, fall_start_frame, fall_end_frame),起訖幀為 1-indexed。

    有跌倒的標註檔前兩行是純數字(跌倒起訖幀);沒有跌倒的標註檔沒有這兩行,
    直接是逗號分隔的逐幀資料。用「第一行是否含逗號」判斷有無表頭(D45 實地盤點確認,
    非文件描述,官方 README 對這個「無跌倒檔案省略表頭」的細節完全沒提)。
    """
    lines = ann_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or "," in lines[0].strip():
        return "adl", None, None
    fall_start = int(lines[0].strip())
    fall_end = int(lines[1].strip())
    return "fall", fall_start, fall_end


def extract_one(model, video_id: str, video_path: Path, ann_path: Path) -> dict:
    kind, fall_start, fall_end = parse_annotation(ann_path)
    data = extract_video_pose(model, video_path)

    T = len(data["xyn"])
    raw_label = np.zeros((T,), dtype=np.int8)
    label_present = np.ones((T,), dtype=bool)
    if kind == "fall":
        for i in range(T):
            frame_num = i + 1  # 1-indexed,對齊 URFD 慣例,D45 實測確認同一套邏輯適用
            if fall_start <= frame_num <= fall_end:
                raw_label[i] = 1

    data["raw_label"] = raw_label
    data["label_present"] = label_present
    data["video_id"] = video_id
    data["kind"] = kind
    return data


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="只處理前 N 支影片(測試用)")
    parser.add_argument("--force", action="store_true", help="已存在的 npz 也重新抽取")
    args = parser.parse_args()

    if not LE2I_RAW_DIR.exists():
        print(f"找不到 {LE2I_RAW_DIR},請先執行 scripts/download_data.py --fallback le2i")
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(resolve_pose_weights(POSE_MODEL_NAME))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = list_videos()
    if args.limit:
        videos = videos[: args.limit]
    print(f"共 {len(videos)} 支可用影片(僅 Coffee_room_01/02、Home_01/02,Office/Lecture_room 無標註已排除)")

    ok = 0
    fail: list[str] = []
    t0 = time.time()

    for i, (video_id, video_path, ann_path) in enumerate(videos, start=1):
        out_path = OUT_DIR / f"{video_id}.npz"
        if out_path.exists() and not args.force:
            print(f"[{i}/{len(videos)}] {video_id} 已存在,略過(--force 可重抽)")
            ok += 1
            continue

        try:
            data = extract_one(model, video_id, video_path, ann_path)
        except Exception as exc:  # noqa: BLE001 - 單支失敗不中斷整批
            print(f"[{i}/{len(videos)}] {video_id} 抽取失敗:{exc}")
            fail.append(video_id)
            continue

        np.savez(out_path, **data)

        det_rate = float(np.mean(~np.isnan(data["xyn"][:, 0, 0]))) * 100
        print(f"[{i}/{len(videos)}] {video_id} OK  kind={data['kind']}  幀數={len(data['xyn'])}  偵測率={det_rate:.1f}%")
        ok += 1

    elapsed = time.time() - t0

    # 摘要統計一律從磁碟上實際存在的 npz 重新讀取,不用迴圈內累加——這樣不管是全新抽取
    # 還是「已存在,略過」的情況都會被算進去,避免分批跑(--limit 測試過的影片在正式全量
    # 跑時被略過)導致統計數字漏算(曾經真的因此漏算 3 支,見 docs/PLAN.md D45)。
    n_fall = n_adl = 0
    detection_rates: list[float] = []
    for video_id, _, _ in videos:
        out_path = OUT_DIR / f"{video_id}.npz"
        if not out_path.exists():
            continue
        with np.load(out_path) as d:
            if str(d["kind"]) == "fall":
                n_fall += 1
            else:
                n_adl += 1
            detection_rates.append(float(np.mean(~np.isnan(d["xyn"][:, 0, 0]))) * 100)

    print()
    print("=== Le2i 關鍵點抽取摘要 ===")
    print(f"成功:{ok}/{len(videos)}(fall={n_fall}, adl={n_adl})  失敗:{len(fail)}  耗時(本次執行):{elapsed:.0f}s")
    if detection_rates:
        print(f"平均偵測率:{np.mean(detection_rates):.1f}%(最低 {np.min(detection_rates):.1f}%)")
    if fail:
        print(f"失敗清單:{', '.join(fail)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
