# fall-guard-cv 進度追蹤

<!-- 使用規則：
  🧭 快速回憶區：直接改寫、不留歷史、全區 ≤30 行。每次收工前更新（用 /update-progress skill）。
  📜 Phase 日誌：只增不改。每 Phase 收尾寫一條，附驗證證據（指令+輸出數字）與 git tag。
  回來時的 10 分鐘流程：讀快速回憶區 → git log --oneline -10 → uv run pytest -q → 讀 docs/PLAN.md 當前 Phase DoD。
  （或直接用 /resume-context skill 自動跑完上述流程。） -->

## 🧭 快速回憶區

**上次收工日期**：2026-07-21
**現在做到哪**：Phase 0 完成（tag: phase-0）——git repo 已建（3 commits）、uv 環境好了（torch 2.11.0+cu128、CUDA True on RTX 4090）、公開文案守門 hooks 實測會擋、pytest 3 passed、README 骨架就位。
**下一步（開機第一件事）**：開 Phase 1——先寫 `scripts/download_data.py`，跑 `uv run python scripts/download_data.py` 下載 URFD（cam0 mp4 ×70 + urfall CSV ×2，帶斷點續傳與 summary 輸出）；動工前先用 Context7 查 ultralytics 現行 API（PLAN.md 第 6 章開頭規範）。
**未決問題**：URFD ADL 躺床幀的 label 語意（是否標 1=lying）——Phase 1 下載後檢查 `urfall-cam0-adls.csv` 定案，記入 Decision Log。
**待使用者人工處理**：(1) 建立 Discord Webhook 並填入 .env 的 `DISCORD_WEBHOOK_URL`（Phase 4 前完成即可）。(2) Phase 1 收尾時人工標註 `data/urfd_meta.csv`（subject_id + ADL 動作類別，約 1–2 小時）。
**已知坑**：torch 必須走 cu128 index（已寫進 pyproject，重建環境直接 `uv sync` 即可）。clone 後要跑一次 `git config core.hooksPath .githooks` 守門才會生效。公開文件裡不要寫出「磁碟機:\Users」的字面路徑示例——會被自家 hook 擋下，用文字描述代替。

## 📜 Phase 日誌（append-only）

### 計畫階段（2026-07-20 完成）

- 完成：前置研究（URFD 現況實測、YOLO26-pose 選型、LangChain 1.x 多模態寫法、Discord webhook 規格、既有專案慣例盤點）→ 四項決策拍板（XGBoost 必做+GRU 選做；LOSO 人工標註；巢狀 repo；先文件後實作）→ 產出 docs/PLAN.md + PROGRESS.md + 三支 skills + CLAUDE.md 複本。
- 關鍵驗證證據（2026-07-20 實測）：
  - URFD 官方頁與下載 URL pattern 全部 HTTP 200（`https://fenix.ur.edu.pl/~mkepski/ds/`）
  - YOLO26-pose 為 ultralytics 官方 latest/recommended（2026-01-14 發布；ultralytics PyPI 8.4.102）
  - langchain 1.3.14 / langchain-google-genai 4.2.7；1.x 影像 content block 標準格式確認
- 決策記錄：PLAN.md 第 2 章 D1–D11。
- 尚未執行：git init（本條目完成時 repo 還不是 git repo，故無 commit 範圍/tag）。

### Phase 0 — 環境與骨架（2026-07-21 完成，tag: phase-0）

- 完成：git init（main）+ hooksPath 設定 + 3 個 Conventional Commits；pyproject（cu128 explicit index、torch+torchvision 同鎖、src layout + hatchling）；`uv sync`；.env 補齊專案變數（`DISCORD_WEBHOOK_URL` 留白待使用者填）+ `.env.example`；自既有專案移植 `scripts/check_public_text.py` + `.githooks/` + `.claude/private/redlist.txt`（不進 git）；README 骨架（十章節標題 + TODO 標記）；`src/fallguard/`（`config.py` 含金鑰 mapping D7）+ 冒煙測試。
- 驗證證據（2026-07-21 實測）：
  - `uv run python -c "import torch; print(torch.cuda.is_available())"` → `True`（RTX 4090、torch 2.11.0+cu128）
  - `uv run pytest -q` → `3 passed`
  - 含本機路徑樣式的測試 commit 被 pre-commit 擋下（exit 1）；正式 3 commits 全數通過 public-copy-check
- 決策連結：PLAN.md D7（金鑰 mapping）、D9（cu128 同鎖 torchvision）、D10（src layout）。
- commit 範圍：`69f7c09..0835351`
- 插曲：public-copy-check SKILL.md 原文含字面路徑示例被自家 hook 擋下，已改為文字描述（教訓收進「已知坑」）。
