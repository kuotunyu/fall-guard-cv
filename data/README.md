# data/

本目錄內容不進 git(見 .gitignore)。取得方式:

- URFD(主資料集):`uv run python scripts/download_data.py`(Phase 1 提供;下載 cam0 mp4 ×70 + frame 級標註 CSV ×2,支援斷點續傳)
- Le2i/IMVIA(備案與泛化測試):`uv run python scripts/download_data.py --fallback le2i`(需 Kaggle API token,放 `~/.kaggle/kaggle.json`)

授權:URFD 為 CC BY-NC-SA 4.0(引用 Kwolek & Kepski 2014),本 repo 不重新散佈原始影像,僅提供下載腳本。
