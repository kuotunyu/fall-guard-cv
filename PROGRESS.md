# fall-guard-cv 進度追蹤

<!-- 使用規則：
  🧭 快速回憶區：直接改寫、不留歷史、全區 ≤30 行。每次收工前更新（用 /update-progress skill）。
  📜 Phase 日誌：只增不改。每 Phase 收尾寫一條，附驗證證據（指令+輸出數字）與 git tag。
  回來時的 10 分鐘流程：讀快速回憶區 → git log --oneline -10 → uv run pytest -q → 讀 docs/PLAN.md 當前 Phase DoD。
  （或直接用 /resume-context skill 自動跑完上述流程。） -->

## 🧭 快速回憶區

**上次收工日期**：2026-07-20
**現在做到哪**：計畫階段完成——docs/PLAN.md（14 章開發藍圖，含 D1–D11 決策與已驗證的外部資源事實）、三支習慣 skills（resume-context / update-progress / public-copy-check）已建立。尚未 git init、尚未建 uv 環境（皆屬 Phase 0）。
**下一步（開機第一件事）**：使用者審閱 docs/PLAN.md；確認後依 PLAN.md 第 6 章開 Phase 0（git init → pyproject cu128 → `uv run python -c "import torch; print(torch.cuda.is_available())"` 須為 True）。
**未決問題**：URFD ADL 躺床幀的 label 語意（是否標 1=lying）——Phase 1 下載後檢查 `urfall-cam0-adls.csv` 定案，記入 Decision Log。
**待使用者人工處理**：(1) 建立 Discord Webhook 並填入 .env 的 `DISCORD_WEBHOOK_URL`（Phase 4 前完成即可）。(2) Phase 1 收尾時人工標註 `data/urfd_meta.csv`（subject_id + ADL 動作類別，約 1–2 小時）。
**已知坑**：Windows 上 PyPI 預設 torch 是 CPU wheel——pyproject 必須用 cu128 explicit index 並同時鎖 torch+torchvision（PLAN.md D9 有完整寫法）。URFD 舊網域 fenix.univ.rzeszow.pl 已死，一律用 fenix.ur.edu.pl（HTTPS）。

## 📜 Phase 日誌（append-only）

### 計畫階段（2026-07-20 完成）

- 完成：前置研究（URFD 現況實測、YOLO26-pose 選型、LangChain 1.x 多模態寫法、Discord webhook 規格、既有專案慣例盤點）→ 四項決策拍板（XGBoost 必做+GRU 選做；LOSO 人工標註；巢狀 repo；先文件後實作）→ 產出 docs/PLAN.md + PROGRESS.md + 三支 skills + CLAUDE.md 複本。
- 關鍵驗證證據（2026-07-20 實測）：
  - URFD 官方頁與下載 URL pattern 全部 HTTP 200（`https://fenix.ur.edu.pl/~mkepski/ds/`）
  - YOLO26-pose 為 ultralytics 官方 latest/recommended（2026-01-14 發布；ultralytics PyPI 8.4.102）
  - langchain 1.3.14 / langchain-google-genai 4.2.7；1.x 影像 content block 標準格式確認
- 決策記錄：PLAN.md 第 2 章 D1–D11。
- 尚未執行：git init（本條目完成時 repo 還不是 git repo，故無 commit 範圍/tag）。
