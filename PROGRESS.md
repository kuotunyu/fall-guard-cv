# fall-guard-cv 進度追蹤

<!-- 使用規則：
  🧭 快速回憶區：直接改寫、不留歷史、全區 ≤30 行。每次收工前更新（用 /update-progress skill）。
  📜 Phase 日誌：只增不改。每 Phase 收尾寫一條，附驗證證據（指令+輸出數字）與 git tag。
  回來時的 10 分鐘流程：讀快速回憶區 → git log --oneline -10 → uv run pytest -q → 讀 docs/PLAN.md 當前 Phase DoD。
  （或直接用 /resume-context skill 自動跑完上述流程。） -->

## 🧭 快速回憶區

**上次收工日期**：2026-07-21
**現在做到哪**：**Phase 1 完成**——下載、關鍵點抽取、人工標註（含一次因 GUI bug 導致的 ADL 複查修正）、GroupKFold + LOSO 切分全數就緒，`uv run pytest -q` 12 passed。準備打 `git tag phase-1` 並開 Phase 2。
**下一步（開機第一件事）**：確認 `git tag phase-1` 已打（若還沒，先補上）；接著開 Phase 2——寫 `src/fallguard/features.py`（PLAN.md §7.3 特徵定義）與規則式 baseline，動工前先查 Context7/官方文件確認任何新用到的套件 API。
**未決問題**：無。
**待使用者人工處理**：建立 Discord Webhook 並填入 .env 的 `DISCORD_WEBHOOK_URL`（Phase 4 前完成即可，不阻塞 Phase 2/3）。
**已知坑**：torch 必須走 cu128 index（已寫進 pyproject，重建環境直接 `uv sync` 即可）。clone 後要跑一次 `git config core.hooksPath .githooks` 守門才會生效。公開文件裡不要寫出「磁碟機:\Users」的字面路徑示例——會被自家 hook 擋下，用文字描述代替。ultralytics `half=True` 已棄用，一律用 `quantize=16`（D14）。ultralytics 下載的 `.pt` 權重會先落在 CWD，`prepare_data.py` 已自動搬進 `models/pretrained/`（已 gitignore），若手動測試模型別忘了清理 repo 根目錄。**cv2 GUI 視窗失焦時按鍵會被 OS 排隊，恢復焦點瞬間可能暴衝連續觸發**——`annotate_urfd.py` 已加 400ms 按鍵沖刷防護（D15），未來若寫其他互動式 cv2 工具要記得比照處理。**LOSO 的 P3/P4/P5 折沒有 ADL 測試樣本**（ADL 只有 P1/P2 出現，D15），Phase 2 evaluate.py 寫報告邏輯時要處理這個不對稱，不能三折一起平均 specificity。

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

### Phase 1 — 資料下載 + 關鍵點抽取 + 人工標註（2026-07-21 完成，tag: phase-1）

- 完成：
  - `scripts/download_data.py`：URFD 下載（斷點續傳、summary），實測 70 mp4 + 2 CSV 全數 200、共 0.10 GB。
  - CSV 格式偵查：`urfall-cam0-{falls,adls}.csv` 無標頭，欄位 = video_id, frame_num(1-indexed), label(-1/0/1), 8 個深度衍生特徵；frame_num 可能有缺口（非連續）。
  - **D12 發現**：ADL 40 段中 16 段含 label=1 幀（姿態水平幾何特徵，非跌倒事件），已定案 kind 覆寫規則並回頭修正 §7.2。
  - Context7 MCP 在本環境未連接，依 CLAUDE.md 改讀官方文件 + 實測 API 確認 ultralytics 8.4.102 用法；**發現 `half=True` 已棄用，改用 `quantize=16`（D14）**，順手全文改正。
  - `scripts/prepare_data.py`：YOLO26m-pose + ByteTrack 抽取 70 支影片關鍵點 → `data/processed/*.npz`；權重快取到 `models/pretrained/`（gitignore）。
  - `scripts/annotate_urfd.py`：互動標註工具（縮圖蒙太奇、已標受試者參考列、PIL 中文疊字、播放/回上一段/備註/複查模式），含 ADL 躺姿比例提示（用 D12 的發現輔助人工判斷）。
  - `scripts/subject_sheet.py`、`scripts/compare_subjects.py`、`scripts/peek_video.py`：標註輔助小工具（全受試者對照圖、兩人放大比對、獨立播放不影響進度）。
  - `scripts/make_splits.py` + `tests/test_splits.py`：GroupKFold + LOSO 皆已產生。
  - `tests/test_prepare_data.py`：npz schema + fall-01/adl-01 標籤對齊(含已知缺口)驗證。
  - **D13**：修正 `.gitignore`，`urfd_meta.csv` 與 `splits.json` 例外進 git（不可重現的人工標註）。
  - **人工標註完成**：70/70 段，unknown 0；過程中發現 cv2 視窗失焦累積按鍵、恢復焦點瞬間暴衝連續觸發的 bug（40 段 ADL 曾被誤標成只有 P1/P2 兩種且分佈不合理），已修正工具加入 400ms 按鍵沖刷（D15）並用 `--review --kind adl` 只複查受影響的 40 段（30 段 fall 未受影響、未重做）。複查後確認 ADL **真的只有 P1、P2 兩位受試者出現**（使用者親眼確認、官方頁未反駁此分佈），P3/P4/P5 只在 fall 中出現，連帶造成 LOSO 折指標可用性不對稱（記入 D15，Phase 2 需處理）。
- 驗證證據（2026-07-21 實測）：
  - `uv run python scripts/prepare_data.py` → `70/70 ok`，耗時 287s，平均偵測率 90.0%（最低 fall-19 53.0%，跌倒瞬間遮擋，符合預期）
  - `uv run python scripts/make_splits.py` → GroupKFold 5 折 + LOSO 5 折（P1 test=30 段 fall6+adl24、P2 test=22 段 fall6+adl16、P3/P4/P5 test 皆 6 段 fall6+adl0）已寫入 `data/splits.json`
  - `uv run pytest -q` → `12 passed`（含 LOSO 測試，原本 skip 現轉 pass）
- 決策連結：PLAN.md D12（ADL label 語意）、D13（.gitignore 例外）、D14（quantize 取代 half）、D15（ADL 僅 2 受試者 + LOSO 指標不對稱 + 按鍵暴衝 bug 修正）。
- commit 範圍：`de1f321..`（本 Phase 內多次小 commit，見 `git log --oneline`）
