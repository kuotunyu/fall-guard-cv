"""下載 URFD(主資料集)或 Le2i/IMVIA(備援,--fallback le2i)到 data/raw/。

用法：
    uv run python scripts/download_data.py                # 下載 URFD:70 mp4 + 2 CSV
    uv run python scripts/download_data.py --fallback le2i  # 改走 Kaggle 下載 Le2i/IMVIA

依據：docs/PLAN.md D1。支援斷點續傳(Range request);結束印檔數/總大小 summary。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
URFD_BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data/"
URFD_DIR = REPO_ROOT / "data" / "raw" / "urfd"
LE2I_DIR = REPO_ROOT / "data" / "raw" / "le2i"
KAGGLE_SLUG = "tuyenldvn/falldataset-imvia"

FALL_COUNT = 30
ADL_COUNT = 40
CHUNK = 1024 * 1024  # 1 MiB


def _urfd_filenames() -> list[str]:
    names = [f"fall-{i:02d}-cam0.mp4" for i in range(1, FALL_COUNT + 1)]
    names += [f"adl-{i:02d}-cam0.mp4" for i in range(1, ADL_COUNT + 1)]
    names += ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]
    return names


def download_with_resume(url: str, dest: Path, session: requests.Session) -> tuple[bool, int]:
    """下載單一檔案,支援斷點續傳。回傳 (是否有新下載動作, 最終檔案大小)。"""
    dest.parent.mkdir(parents=True, exist_ok=True)

    head = session.head(url, timeout=30, allow_redirects=True)
    head.raise_for_status()
    remote_size = int(head.headers.get("Content-Length", "0"))

    existing = dest.stat().st_size if dest.exists() else 0
    if remote_size and existing == remote_size:
        return False, existing

    headers = {}
    mode = "wb"
    if existing and existing < remote_size:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
    elif existing and existing > remote_size:
        # 本機比遠端大,檔案損毀,重下
        existing = 0
        mode = "wb"

    resp = session.get(url, headers=headers, stream=True, timeout=60)
    resp.raise_for_status()
    with dest.open(mode) as f:
        for chunk in resp.iter_content(chunk_size=CHUNK):
            if chunk:
                f.write(chunk)

    return True, dest.stat().st_size


def download_urfd() -> None:
    URFD_DIR.mkdir(parents=True, exist_ok=True)
    filenames = _urfd_filenames()
    session = requests.Session()

    total_bytes = 0
    downloaded = 0
    skipped = 0
    failed: list[str] = []

    for i, name in enumerate(filenames, start=1):
        url = URFD_BASE + name
        dest = URFD_DIR / name
        print(f"[{i}/{len(filenames)}] {name} ...", end=" ", flush=True)
        try:
            did_download, size = download_with_resume(url, dest, session)
            total_bytes += size
            if did_download:
                downloaded += 1
                print(f"OK ({size / 1e6:.1f} MB)")
            else:
                skipped += 1
                print("已存在,略過")
        except requests.RequestException as exc:
            failed.append(name)
            print(f"失敗:{exc}")

    print()
    print("=== 下載摘要 ===")
    print(f"目標目錄:{URFD_DIR}")
    print(f"新下載:{downloaded}  已存在略過:{skipped}  失敗:{len(failed)}  總數:{len(filenames)}")
    print(f"目前總大小:{total_bytes / 1e9:.2f} GB")
    if failed:
        print("失敗清單:", ", ".join(failed))
        print("提示:URFD 站況若持續異常,改用 --fallback le2i")
        sys.exit(1)

    mp4_count = len(list(URFD_DIR.glob("*.mp4")))
    csv_count = len(list(URFD_DIR.glob("*.csv")))
    print(f"驗收:mp4 {mp4_count}/{FALL_COUNT + ADL_COUNT}  csv {csv_count}/2")
    if mp4_count != FALL_COUNT + ADL_COUNT or csv_count != 2:
        print("警告:檔案數與預期不符,請檢查上方失敗清單或重跑本腳本(支援續傳)")
        sys.exit(1)


def download_le2i() -> None:
    """備援:透過 Kaggle API 下載 Le2i/IMVIA(需先 uv sync --extra le2i 並設好 kaggle token)。"""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("缺少 kaggle 套件。請先執行:uv sync --extra le2i")
        print("並確認 Kaggle token 已放在 ~/.kaggle/kaggle.json(見 https://www.kaggle.com/settings → API → Create New Token)")
        sys.exit(1)

    from kaggle.api.kaggle_api_extended import KaggleApi

    LE2I_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"下載 Kaggle 資料集 {KAGGLE_SLUG} → {LE2I_DIR} ...")
    api.dataset_download_files(KAGGLE_SLUG, path=str(LE2I_DIR), unzip=True, quiet=False)

    files = list(LE2I_DIR.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    print()
    print("=== 下載摘要(Le2i/IMVIA) ===")
    print(f"目標目錄:{LE2I_DIR}")
    print(f"檔案數:{file_count}  總大小:{total_bytes / 1e9:.2f} GB")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 舊終端不支援時照常執行
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fallback",
        choices=["le2i"],
        default=None,
        help="URFD 失效時改走此備援資料集",
    )
    args = parser.parse_args()

    if args.fallback == "le2i":
        download_le2i()
    else:
        download_urfd()


if __name__ == "__main__":
    main()
