"""把 models/xgboost/ 的權重與模型卡上傳 Hugging Face(CLAUDE.md:模型權重不進 git,上傳 HF)。

用法：
    uv run python scripts/upload_to_hf.py --repo-id <帳號>/fall-guard-cv-xgboost
    uv run python scripts/upload_to_hf.py --repo-id <帳號>/fall-guard-cv-xgboost --private
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import HfApi

from fallguard.config import REPO_ROOT, settings

MODELS_DIR = REPO_ROOT / "models" / "xgboost"


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="格式:帳號/repo名稱")
    parser.add_argument("--private", action="store_true", help="建立私有 repo(預設公開)")
    args = parser.parse_args()

    if not settings.hf_token:
        print("找不到 HF_TOKEN,請確認 .env 已設定")
        sys.exit(1)
    if not MODELS_DIR.exists():
        print(f"找不到 {MODELS_DIR},請先完成 Colab 訓練並把權重放進這個資料夾")
        sys.exit(1)

    expected = ["xgb_final.json", "xgb_loso_results.json", "README.md"]
    missing = [f for f in expected if not (MODELS_DIR / f).exists()]
    if missing:
        print(f"缺少檔案:{missing}")
        sys.exit(1)

    api = HfApi(token=settings.hf_token)

    print(f"建立/確認 repo:{args.repo_id}(private={args.private})")
    repo_url = api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"repo: {repo_url}")

    print(f"上傳 {MODELS_DIR} 的內容...")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(MODELS_DIR),
        commit_message="上傳 XGBoost 跌倒偵測分類器(LOSO 5 折 + 最終部署模型)",
    )
    print(f"完成:https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
